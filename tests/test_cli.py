from pathlib import Path

import pytest

from lecture_summarizer import cli
from lecture_summarizer.summarize import load_nlp, score_sentences

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "faraday_candle_lecture1.txt"


def test_summarize_text_file(capsys):
    assert cli.main(["summarize", str(SAMPLE)]) == 0
    out = capsys.readouterr().out.strip()
    # The summary is exactly three of the lecture's own sentences, in their original order.
    sentences = [s.text for s in score_sentences(load_nlp()(SAMPLE.read_text(encoding="utf-8")))]
    kept = [s for s in sentences if s in out]
    assert len(kept) == 3
    assert " ".join(kept) == out


def test_output_file_is_saved(tmp_path, capsys):
    target = tmp_path / "summary.txt"
    assert cli.main(["summarize", str(SAMPLE), "-n", "2", "-o", str(target)]) == 0
    printed = capsys.readouterr().out.strip()
    assert target.read_text(encoding="utf-8").strip() == printed


def test_missing_file_is_a_clear_error(capsys):
    assert cli.main(["summarize", "does-not-exist.txt"]) == 1
    assert "Couldn't find" in capsys.readouterr().err


def test_sentences_and_ratio_together_are_rejected():
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["summarize", str(SAMPLE), "-n", "2", "-r", "0.5"])
    assert exit_info.value.code == 2


def test_audio_file_is_transcribed_and_transcript_saved(tmp_path, monkeypatch, capsys):
    audio_path = tmp_path / "lecture.wav"
    audio_path.write_bytes(b"")
    transcript = "Volcanoes form at plate boundaries. Lunch was fine. Volcanoes erupt magma."
    monkeypatch.setattr(cli, "load_audio_file", lambda path: "audio")
    monkeypatch.setattr(cli, "transcribe", lambda audio, status: transcript)

    saved = tmp_path / "transcript.txt"
    assert cli.main(["summarize", str(audio_path), "-n", "2", "-t", str(saved)]) == 0
    assert saved.read_text(encoding="utf-8").strip() == transcript
    assert "Volcanoes" in capsys.readouterr().out


def test_record_command(monkeypatch, capsys):
    monkeypatch.setattr(cli, "record", lambda seconds, status: f"{seconds} seconds of audio")
    monkeypatch.setattr(cli, "transcribe", lambda audio, status: "Glaciers carve valleys. Glaciers move slowly.")
    assert cli.main(["record", "-s", "10", "-n", "1"]) == 0
    assert "Glaciers" in capsys.readouterr().out


def test_transcription_errors_exit_cleanly(monkeypatch, capsys):
    def offline(audio, status):
        raise cli.TranscriptionError("Couldn't reach Google's speech service.")

    monkeypatch.setattr(cli, "record", lambda seconds, status: "audio")
    monkeypatch.setattr(cli, "transcribe", offline)
    assert cli.main(["record", "-s", "5"]) == 1
    assert "error: Couldn't reach" in capsys.readouterr().err


def test_phone_recordings_get_a_conversion_hint(tmp_path, capsys):
    recording = tmp_path / "lecture.m4a"
    recording.write_bytes(b"")
    assert cli.main(["summarize", str(recording)]) == 1
    err = capsys.readouterr().err
    assert "needs converting to WAV" in err and "afconvert" in err
