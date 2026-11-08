import numpy as np
import pytest
import speech_recognition as sr

from lecture_summarizer.transcribe import (
    CONTEXT_SECONDS,
    TranscriptionError,
    _slice,
    duration_seconds,
    load_audio_file,
    record,
    split_on_pauses,
    transcribe,
    trim_overlap,
)

RATE, WIDTH = 16000, 2


def tone(seconds: float) -> bytes:
    """A loud 220 Hz tone, standing in for speech."""
    t = np.arange(int(RATE * seconds)) / RATE
    return (8000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16).tobytes()


def quiet(seconds: float) -> bytes:
    return b"\x00\x00" * int(RATE * seconds)


def audio(*parts: bytes) -> sr.AudioData:
    return sr.AudioData(b"".join(parts), RATE, WIDTH)


class FakeRecognizer:
    """Stands in for Google: returns scripted results instead of using the internet."""

    def __init__(self, results=(), record_effects=()):
        self.results = list(results)
        self.record_effects = list(record_effects)
        self.clip_lengths = []

    def recognize_google(self, clip):
        self.clip_lengths.append(duration_seconds(clip))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def record(self, source, duration=None):
        effect = self.record_effects.pop(0) if self.record_effects else None
        if isinstance(effect, BaseException):
            raise effect
        return audio(quiet(duration))


class FakeMic:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- splitting at pauses -------------------------------------------------------------


def test_stretches_are_joined_into_pieces_of_about_20_seconds():
    # Five 8-second stretches of "speech" separated by half-second pauses.
    clip = audio(*[tone(8) + quiet(0.5) for _ in range(5)])
    pieces = split_on_pauses(clip)
    lengths = [end - start for start, end in pieces]
    assert len(pieces) == 2
    assert lengths[0] == pytest.approx(25, abs=0.2)  # three stretches: 8 + 0.5 + 8 + 0.5 + 8
    assert lengths[1] == pytest.approx(16.5, abs=0.2)  # the remaining two


def test_a_piece_never_goes_over_30_seconds():
    clip = audio(*[tone(12) + quiet(0.5) for _ in range(4)])
    assert all(end - start <= 30 for start, end in split_on_pauses(clip))


def test_short_gaps_do_not_count_as_pauses():
    clip = audio(tone(5), quiet(0.1), tone(5))
    assert split_on_pauses(clip, target=1) == [pytest.approx((0, 10.1), abs=0.05)]


def test_speech_without_pauses_is_cut_into_equal_parts():
    pieces = split_on_pauses(audio(tone(70)))
    assert len(pieces) == 3
    assert all(end - start == pytest.approx(70 / 3, abs=0.1) for start, end in pieces)


def test_silence_gives_no_pieces():
    assert split_on_pauses(audio(quiet(10))) == []


def test_slicing_never_splits_a_sample():
    clip = audio(tone(2))
    for start in (0.00003, 0.5, 1.23456):  # times that land between bytes
        piece = _slice(clip, start, 2)
        assert len(piece.frame_data) % WIDTH == 0


# --- trimming the warm-up overlap ----------------------------------------------------


def test_repeated_words_from_the_warm_up_are_trimmed():
    previous = "So the candle burns because the wax melts."
    assert trim_overlap(previous, "the wax melts and then it rises") == "and then it rises"


def test_nothing_is_trimmed_without_an_overlap():
    assert trim_overlap("The wax melts.", "air flows upward") == "air flows upward"


def test_a_single_matching_word_is_not_trimmed():
    assert trim_overlap("We light the candle.", "candle flames are bright") == "candle flames are bright"


# --- transcription -------------------------------------------------------------------


def test_each_piece_becomes_a_sentence_with_warm_up_context():
    clip = audio(*[tone(8) + quiet(0.5) for _ in range(5)])
    fake = FakeRecognizer(["the flame is hot", "of the candle and the wick draws up wax"])
    text = transcribe(clip, recognizer=fake)
    assert text == "The flame is hot. Of the candle and the wick draws up wax."
    first, second = fake.clip_lengths
    assert first == pytest.approx(25.15, abs=0.2)  # the first piece has nothing before it
    assert second == pytest.approx(16.5 + CONTEXT_SECONDS + 0.15, abs=0.2)  # later pieces get warm-up audio


def test_unclear_piece_is_skipped():
    clip = audio(*[tone(10) + quiet(0.5) for _ in range(6)])  # three ~20 s pieces
    fake = FakeRecognizer(["hello there", sr.UnknownValueError(), "goodbye now"])
    assert transcribe(clip, recognizer=fake) == "Hello there. Goodbye now."


def test_no_speech_at_all_is_an_error():
    fake = FakeRecognizer([sr.UnknownValueError()])
    with pytest.raises(TranscriptionError, match="No speech"):
        transcribe(audio(tone(5)), recognizer=fake)
    with pytest.raises(TranscriptionError, match="No speech"):
        transcribe(audio(quiet(5)), recognizer=FakeRecognizer())


def test_no_internet_is_a_clear_error():
    fake = FakeRecognizer([sr.RequestError("connection failed")])
    with pytest.raises(TranscriptionError, match="internet"):
        transcribe(audio(tone(5)), recognizer=fake)


def test_missing_flac_encoder_suggests_a_fix():
    fake = FakeRecognizer([OSError("Bad CPU type in executable")])
    with pytest.raises(TranscriptionError, match="brew install flac"):
        transcribe(audio(tone(5)), recognizer=fake)


# --- recording and files -------------------------------------------------------------


def test_records_for_the_requested_time():
    clip = record(12, recognizer=FakeRecognizer(), microphone=FakeMic)
    assert duration_seconds(clip) == pytest.approx(12)


def test_ctrl_c_keeps_what_was_recorded():
    fake = FakeRecognizer(record_effects=[None, None, KeyboardInterrupt()])
    clip = record(60, recognizer=fake, microphone=FakeMic)
    assert duration_seconds(clip) == pytest.approx(10)  # two 5-second pieces


def test_missing_pyaudio_explains_setup():
    def no_pyaudio():
        raise AttributeError("Could not find PyAudio; check installation")

    with pytest.raises(TranscriptionError, match="brew install portaudio"):
        record(5, recognizer=FakeRecognizer(), microphone=no_pyaudio)


def test_unreadable_audio_file(tmp_path):
    bad = tmp_path / "lecture.wav"
    bad.write_bytes(b"not really audio")
    with pytest.raises(TranscriptionError, match="WAV, AIFF or FLAC"):
        load_audio_file(bad)
    with pytest.raises(TranscriptionError, match="Couldn't find"):
        load_audio_file(tmp_path / "missing.wav")


def test_reads_a_real_wav_file(tmp_path):
    import wave

    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(WIDTH)
        w.setframerate(RATE)
        w.writeframes(tone(2))
    assert duration_seconds(load_audio_file(path)) == pytest.approx(2)
