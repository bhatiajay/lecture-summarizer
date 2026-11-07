from pathlib import Path

import pytest

from lecture_summarizer.summarize import load_nlp, score_sentences, sentence_count, summarize

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "faraday_candle_lecture1.txt"


def test_keeps_sentences_in_original_order_with_spaces():
    text = (
        "Photosynthesis turns light into chemical energy. Lunch was late today. "
        "Plants store that energy from photosynthesis as sugar. Photosynthesis happens in chloroplasts."
    )
    # Sentences 1 and 3 share the most frequent words (photosynthesis, energy). They come
    # back in their original order, joined by a single space.
    assert summarize(text, sentences=2) == (
        "Photosynthesis turns light into chemical energy. "
        "Plants store that energy from photosynthesis as sugar."
    )


def test_capitalized_words_count():
    # The original version counted "Mitochondria" but looked up "mitochondria", so
    # capitalized words never scored. Here the key word only ever appears capitalized.
    text = "Mitochondria make energy. The bus was late. Mitochondria have DNA. Rain fell outside."
    summary = summarize(text, sentences=2)
    assert summary == "Mitochondria make energy. Mitochondria have DNA."


def test_score_does_not_depend_on_position():
    # The original first version added every earlier sentence's score into later ones.
    nlp = load_nlp()
    forward = {s.text: s.score for s in score_sentences(nlp("Cats purr. Dogs bark loudly. Cats sleep."))}
    backward = {s.text: s.score for s in score_sentences(nlp("Cats sleep. Dogs bark loudly. Cats purr."))}
    assert forward == pytest.approx(backward)


def test_sentence_count_rules():
    assert sentence_count(25) == 3
    assert sentence_count(25, sentences=5) == 5
    assert sentence_count(2, sentences=5) == 2  # can't keep more than exist
    assert sentence_count(25, ratio=0.2) == 5
    assert sentence_count(4, ratio=0.1) == 1  # always at least one
    with pytest.raises(ValueError):
        sentence_count(25, sentences=2, ratio=0.5)
    with pytest.raises(ValueError):
        sentence_count(25, ratio=1.5)
    with pytest.raises(ValueError):
        sentence_count(25, sentences=0)


def test_empty_text_is_rejected():
    with pytest.raises(ValueError, match="no text"):
        summarize("   \n ")


def test_sample_lecture_summary_uses_real_sentences():
    text = SAMPLE.read_text(encoding="utf-8")
    summary = summarize(text)
    kept = [s.text for s in score_sentences(load_nlp()(text)) if s.text in summary]
    assert len(kept) == 3
    assert all(sentence in text for sentence in kept)
    assert "candle" in summary.lower()
