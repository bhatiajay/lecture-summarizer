# lecture-summarizer

[![tests](https://github.com/bhatiajay/lecture-summarizer/actions/workflows/tests.yml/badge.svg)](https://github.com/bhatiajay/lecture-summarizer/actions/workflows/tests.yml)

A command-line tool that records a lecture (or loads a transcript or audio file) and
keeps its most important sentences, so you can review the summary later without
replaying the recording. Speech is transcribed with Google's free speech service, and
the summary is built with [spaCy](https://spacy.io/).

The original version is preserved at tag
[`v1-original`](https://github.com/bhatiajay/lecture-summarizer/tree/v1-original).

## Quick start

Needs [uv](https://docs.astral.sh/uv/). It installs Python 3.12, the exact package
versions from `uv.lock`, and spaCy's English model.

```bash
git clone https://github.com/bhatiajay/lecture-summarizer.git
cd lecture-summarizer
uv sync
uv run lecture-summarizer summarize samples/faraday_candle_lecture1.txt
```

### Recording from the microphone

Recording needs PyAudio, which on a Mac has to be built against PortAudio:

```bash
brew install portaudio
uv sync --extra mic
uv run lecture-summarizer record --seconds 600 -t lecture.txt -o summary.txt
```

Press **Ctrl+C** to stop early; everything recorded so far is kept. The first time, macOS
asks whether your terminal app may use the microphone. If you said no, allow it in
System Settings → Privacy & Security → Microphone.

### Summarizing a recording you already have

WAV, AIFF and FLAC files work directly. Convert phone recordings (M4A) or MP3s first with
the converter built into macOS:

```bash
afconvert -f WAVE -d LEI16@16000 lecture.m4a lecture.wav
uv run lecture-summarizer summarize lecture.wav -t lecture.txt
```

## Options

| Option | Meaning |
|---|---|
| `-n`, `--sentences N` | how many sentences to keep (default 3) |
| `-r`, `--ratio R` | keep a share of the sentences instead, like `0.2` for 20% |
| `-o`, `--output FILE` | also save the summary to a file |
| `-t`, `--transcript FILE` | save the full transcript (audio input only) |
| `-s`, `--seconds S` | `record` only: how long to record (default 60) |

The summary is printed to standard output and progress messages to standard error, so
`lecture-summarizer summarize notes.txt > summary.txt` saves only the summary. Use `-` as
the path to read text from standard input.

## How it works

```
microphone or audio file ──► transcribe.py ──► transcript ──► summarize.py ──► summary
                         (Google speech service)          (spaCy word frequencies)
text file ───────────────────────────────────►
```

**Summarizing** (`src/lecture_summarizer/summarize.py`) is *extractive*: it picks
sentences from the text rather than writing new ones.
1. spaCy splits the text into sentences and words.
2. Common words like "the" and punctuation are skipped. The rest are counted in lowercase
   base form, so "Candles" and "candle" count as the same word.
3. Each word gets a weight: its count divided by the count of the most frequent word.
4. Each sentence's score is the sum of its word weights.
5. The highest-scoring sentences are printed in the order they were spoken.

Adding up weights favors longer, detail-rich sentences. Averaging them would favor short
ones, but then a tiny sentence with one key word can beat everything else.

**Transcribing** (`src/lecture_summarizer/transcribe.py`) works around three quirks of
Google's free service:
- **It rejects long clips.** Audio is sent in pieces of 20–30 seconds.
- **It often drops the first few seconds of a clip.** Each piece is sent with the 4
  seconds of audio before it as a warm-up, and words that repeat the previous piece are
  trimmed.
- **It returns no punctuation.** Pieces are cut at pauses in the speech, and each piece
  becomes one "sentence". Summaries of recordings are therefore whole passages rather
  than single sentences.

Measured on synthetic speech of the sample lecture, about 80–93% of the words come back
(about 80% for plain 30-second chunks). Expect some missing or misheard words, especially
in noisy rooms.

## Limitations

- **Internet required.** Transcription needs internet. Summarizing text works offline.
- **Free service.** Google's free service is meant for light personal use and may slow
  down or refuse heavy use. Very long lectures mean many requests.
- **Apple-silicon Macs without Rosetta.** SpeechRecognition's bundled FLAC encoder is
  Intel-only. Run `brew install flac` if transcription reports it couldn't prepare the
  audio.
- **English only.** The summarizer uses spaCy's small English model.

## Project layout

```
├── src/lecture_summarizer/
│   ├── cli.py          command-line interface (record / summarize)
│   ├── summarize.py    sentence scoring and selection
│   └── transcribe.py   recording, audio files, splitting at pauses, Google requests
├── samples/            public-domain sample lecture (Faraday, 1860)
├── tests/              pytest suite; uses a stand-in for Google, so no internet needed
├── pyproject.toml      dependencies; `mic` extra adds PyAudio
└── uv.lock             exact versions
```

## Tests

```bash
uv run pytest
```

GitHub Actions runs the tests and summarizes the sample lecture on every push and pull
request.

## Changes from the original version

- **Scores no longer build up.** The first script never reset `sentenceScore`, so each
  sentence's score included every earlier sentence's and the summary always leaned
  toward the end of the lecture.
- **Capitalized words count.** The second script counted words with their capitals but
  looked them up in lowercase, so key terms like "Mitochondria" never scored.
- **Readable output.** Sentences come out in their original order with spaces between
  them; they used to be glued together ("…end.Next…") and ordered by score.
- **Common words ignored.** The first script let "the" and "," decide the scores.
- **Summaries of recordings work.** Google returns text without punctuation, so a
  recording used to be one giant "sentence". Audio is now split at pauses.
- **Clear errors.** No internet, no speech, a missing microphone or setup step, and
  unsupported files each get a plain message instead of a crash.
- **Public-domain sample.** The copied encyclopedia article was replaced with Faraday's
  public-domain lecture.
- **Real command-line tool.** One `lecture-summarizer` command with options replaces two
  scripts that used `input()`, plus tests, locked dependencies and CI.
