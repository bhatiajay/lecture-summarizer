"""Command-line interface.

    lecture-summarizer record --seconds 300            record, transcribe, summarize
    lecture-summarizer summarize notes.txt             summarize a text file
    lecture-summarizer summarize lecture.wav           transcribe an audio file, then summarize

The summary goes to standard output; progress messages go to standard error, so
`lecture-summarizer summarize notes.txt > summary.txt` saves only the summary.
"""

import argparse
import sys
from pathlib import Path

from lecture_summarizer import __version__
from lecture_summarizer.summarize import DEFAULT_SENTENCES, summarize
from lecture_summarizer.transcribe import TranscriptionError, load_audio_file, record, transcribe

AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".aifc", ".flac"}
CONVERT_SUFFIXES = {".m4a", ".mp3", ".mp4", ".aac", ".ogg", ".opus", ".webm", ".caf"}


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _ratio(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1, like 0.2")
    return number


def _seconds(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be more than 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lecture-summarizer",
        description="Record a lecture or load a transcript, then keep its most important sentences.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    length = argparse.ArgumentParser(add_help=False)
    group = length.add_mutually_exclusive_group()
    group.add_argument("-n", "--sentences", type=_positive_int,
                       help=f"how many sentences to keep (default {DEFAULT_SENTENCES})")
    group.add_argument("-r", "--ratio", type=_ratio, help="share of sentences to keep, like 0.2 for 20%%")
    length.add_argument("-o", "--output", type=Path, help="also save the summary to this file")
    length.add_argument("-t", "--transcript", type=Path, help="save the full transcript to this file")

    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record", parents=[length], help="record from the microphone, then summarize")
    rec.add_argument("-s", "--seconds", type=_seconds, default=60,
                     help="how long to record (default 60); Ctrl+C stops early")
    summ = commands.add_parser("summarize", parents=[length],
                               help="summarize a text file, or an audio file (WAV, AIFF, FLAC)")
    summ.add_argument("path", type=Path, help="a .txt file, an audio file, or - to read text from stdin")
    return parser


def _get_text(args: argparse.Namespace) -> tuple[str, bool]:
    """Return (text, came_from_audio)."""
    if args.command == "record":
        audio = record(args.seconds, status=_status)
        return transcribe(audio, status=_status), True
    if str(args.path) == "-":
        return sys.stdin.read(), False
    if args.path.suffix.lower() in CONVERT_SUFFIXES:
        wav = args.path.with_suffix(".wav")
        raise TranscriptionError(
            f"{args.path.name} needs converting to WAV first. On a Mac:\n"
            f"  afconvert -f WAVE -d LEI16@16000 '{args.path}' '{wav}'"
        )
    if args.path.suffix.lower() in AUDIO_SUFFIXES:
        audio = load_audio_file(args.path)
        return transcribe(audio, status=_status), True
    try:
        return args.path.read_text(encoding="utf-8"), False
    except FileNotFoundError:
        raise TranscriptionError(f"Couldn't find {args.path}.") from None
    except UnicodeDecodeError:
        raise TranscriptionError(
            f"{args.path} isn't a plain-text file. For audio, use WAV, AIFF or FLAC."
        ) from None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        text, from_audio = _get_text(args)
        if args.transcript:
            if not from_audio:
                _status("Note: --transcript only applies to audio; the input is already text.")
            else:
                args.transcript.write_text(text + "\n", encoding="utf-8")
                _status(f"Saved the transcript to {args.transcript}")
        summary = summarize(text, sentences=args.sentences, ratio=args.ratio)
    except (TranscriptionError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(summary)
    if args.output:
        args.output.write_text(summary + "\n", encoding="utf-8")
        _status(f"Saved the summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
