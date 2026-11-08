"""Turn speech into text with Google's free web speech service (needs internet).

Audio comes from the microphone or from a WAV, AIFF or FLAC file.

Three quirks of the free service shape how audio is sent:
  * It rejects or cuts off long clips, so audio goes in pieces of at most 30 seconds.
  * It often drops the first few seconds of each clip. Each piece is therefore sent
    with 4 seconds of the audio before it as a warm-up, and any words that repeat
    the end of the previous piece are trimmed.
  * Its text has no punctuation. Pieces are cut at pauses in the speech, and each
    piece becomes one "sentence" for the summarizer.

Tested on 84 seconds of lecture audio: about 93% of the words came back, compared
with 80% when sending plain 30-second chunks.
"""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import speech_recognition as sr

RECORD_PIECE_SECONDS = 5  # Ctrl+C loses at most this much of a recording
TARGET_PIECE_SECONDS = 20  # pieces are built up from pauses until at least this long...
MAX_PIECE_SECONDS = 30  # ...but never longer than this
CONTEXT_SECONDS = 4  # warm-up audio sent before each piece
MIN_PAUSE_SECONDS = 0.3  # a quiet stretch at least this long counts as a pause
EDGE_SECONDS = 0.15  # a little extra audio after each piece so the last word isn't clipped
FRAME_SECONDS = 0.03  # loudness is measured in 30 ms frames
QUIET_FRACTION = 0.1  # a frame is "quiet" below 10% of the speech's typical peak loudness
MIC_SETUP = "brew install portaudio && uv sync --extra mic"

Status = Callable[[str], None]


class TranscriptionError(Exception):
    """Something went wrong getting text from audio. The message says how to fix it."""


def _quiet(_: str) -> None:
    pass


def duration_seconds(audio: sr.AudioData) -> float:
    return len(audio.frame_data) / (audio.sample_rate * audio.sample_width)


def _slice(audio: sr.AudioData, start: float, end: float) -> sr.AudioData:
    """Cut audio between two times (seconds), on whole-sample boundaries.

    AudioData.get_segment cuts at byte positions, and a cut that lands in the middle of a
    2-byte sample scrambles everything after it into noise.
    """
    width, rate = audio.sample_width, audio.sample_rate
    first, last = int(start * rate) * width, int(end * rate) * width
    return sr.AudioData(audio.frame_data[first:last], rate, width)


# --- Getting audio ---------------------------------------------------------------------


def record(
    seconds: float,
    status: Status = _quiet,
    recognizer: sr.Recognizer | None = None,
    microphone: Callable[[], sr.Microphone] = sr.Microphone,
) -> sr.AudioData:
    """Record from the default microphone for `seconds`, or until Ctrl+C."""
    if seconds <= 0:
        raise TranscriptionError("The recording length must be more than 0 seconds.")
    recognizer = recognizer or sr.Recognizer()
    try:
        source = microphone()
    except AttributeError as error:  # raised by SpeechRecognition when PyAudio is missing
        raise TranscriptionError(
            f"Microphone support isn't installed. Run:\n  {MIC_SETUP}"
        ) from error

    pieces: list[sr.AudioData] = []
    try:
        with source:
            status(f"Recording for up to {seconds:g} seconds. Press Ctrl+C to stop early.")
            remaining = seconds
            while remaining > 0:
                pieces.append(recognizer.record(source, duration=min(RECORD_PIECE_SECONDS, remaining)))
                remaining -= RECORD_PIECE_SECONDS
    except KeyboardInterrupt:
        status("Stopped recording.")
    except OSError as error:
        raise TranscriptionError(
            f"Couldn't use the microphone ({error}). Check that one is connected and that your "
            "terminal app is allowed to use it (System Settings > Privacy & Security > Microphone)."
        ) from error

    if not pieces:
        raise TranscriptionError("Nothing was recorded.")
    first = pieces[0]
    return sr.AudioData(b"".join(p.frame_data for p in pieces), first.sample_rate, first.sample_width)


def load_audio_file(path: Path | str, recognizer: sr.Recognizer | None = None) -> sr.AudioData:
    """Read a WAV, AIFF or FLAC file."""
    recognizer = recognizer or sr.Recognizer()
    try:
        with sr.AudioFile(str(path)) as source:
            return recognizer.record(source)
    except FileNotFoundError as error:
        raise TranscriptionError(f"Couldn't find the audio file {path}.") from error
    except ValueError as error:
        raise TranscriptionError(
            f"Couldn't read {path}. Use a WAV, AIFF or FLAC file "
            "(MP3 and M4A need converting first)."
        ) from error


# --- Splitting at pauses ---------------------------------------------------------------


def split_on_pauses(
    audio: sr.AudioData,
    min_pause: float = MIN_PAUSE_SECONDS,
    target: float = TARGET_PIECE_SECONDS,
    longest: float = MAX_PIECE_SECONDS,
) -> list[tuple[float, float]]:
    """Plan the pieces to send, as (start, end) times in seconds.

    1. Measure loudness in short frames and treat quiet runs of at least `min_pause`
       as pauses. Speech between pauses is a "stretch".
    2. Join neighboring stretches until a piece is at least `target` seconds long,
       without going over `longest`.
    3. A single stretch longer than `longest` (no pauses at all) is cut into equal parts.
    Silent audio gives no pieces.
    """
    samples = np.frombuffer(audio.get_raw_data(convert_width=2), dtype=np.int16).astype(np.float64)
    frame = max(1, int(audio.sample_rate * FRAME_SECONDS))
    frames = len(samples) // frame
    if frames == 0:
        return []
    loudness = np.sqrt(np.mean(samples[: frames * frame].reshape(frames, frame) ** 2, axis=1))
    threshold = QUIET_FRACTION * np.percentile(loudness, 95)
    loud_frames = np.flatnonzero(loudness >= threshold) if threshold > 0 else []
    if len(loud_frames) == 0:
        return []

    min_gap = max(1, round(min_pause / FRAME_SECONDS))
    stretches = []
    start = last = loud_frames[0]
    for i in loud_frames[1:]:
        if i - last > min_gap:
            stretches.append((start * FRAME_SECONDS, (last + 1) * FRAME_SECONDS))
            start = i
        last = i
    stretches.append((start * FRAME_SECONDS, (last + 1) * FRAME_SECONDS))

    pieces: list[tuple[float, float]] = []
    for begin, end in stretches:
        if pieces and pieces[-1][1] - pieces[-1][0] < target and end - pieces[-1][0] <= longest:
            pieces[-1] = (pieces[-1][0], end)
            continue
        parts = int(np.ceil((end - begin) / longest))
        step = (end - begin) / parts
        pieces.extend((begin + k * step, begin + (k + 1) * step) for k in range(parts))
    return [(float(a), float(b)) for a, b in pieces]


# --- Talking to Google -----------------------------------------------------------------


def _normalize(token: str) -> str:
    return re.sub(r"[^\w']", "", token.lower())


def trim_overlap(previous: str, current: str, look_back: int = 30, longest: int = 25) -> str:
    """Remove words at the start of `current` that repeat the end of `previous`.

    The warm-up audio before each piece is sometimes transcribed too, which would
    repeat a few words. At least 2 words must match, so a single common word like
    "the" is never removed.
    """
    tail = [_normalize(t) for t in previous.split()][-look_back:]
    tokens = current.split()
    words = [_normalize(t) for t in tokens]
    for k in range(min(longest, len(words)), 1, -1):
        if any(tail[j : j + k] == words[:k] for j in range(len(tail) - k + 1)):
            return " ".join(tokens[k:])
    return current


def _as_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!" else text + "."


def transcribe(audio: sr.AudioData, status: Status = _quiet, recognizer: sr.Recognizer | None = None) -> str:
    """Send each piece to Google and join the results, one sentence per piece."""
    recognizer = recognizer or sr.Recognizer()
    pieces = split_on_pauses(audio)
    total = duration_seconds(audio)

    sentences: list[str] = []
    for number, (start, end) in enumerate(pieces, 1):
        status(f"Transcribing part {number} of {len(pieces)}...")
        clip = _slice(audio, max(0.0, start - CONTEXT_SECONDS), min(total, end + EDGE_SECONDS))
        try:
            text = recognizer.recognize_google(clip)
        except sr.UnknownValueError:
            continue  # unclear speech or noise in this piece; keep going
        except sr.RequestError as error:
            raise TranscriptionError(
                f"Couldn't reach Google's speech service ({error}). Check your internet connection."
            ) from error
        except OSError as error:
            # The FLAC encoder bundled with SpeechRecognition is Intel-only on macOS; Apple-silicon
            # Macs without Rosetta need a native one, which SpeechRecognition picks up automatically.
            raise TranscriptionError(
                f"Couldn't prepare the audio for upload ({error}). Install FLAC with: brew install flac"
            ) from error
        if sentences:
            text = trim_overlap(sentences[-1], text)
        if sentence := _as_sentence(text):
            sentences.append(sentence)

    if not sentences:
        raise TranscriptionError("No speech was recognized. Try speaking closer to the microphone.")
    return " ".join(sentences)
