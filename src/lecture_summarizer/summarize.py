"""Extractive summarization: keep the sentences that use the text's most frequent words.

How it works:
  1. Split the text into sentences and words with spaCy.
  2. Count each content word (skip stop words like "the" and punctuation). Words are
     lowercased and reduced to their base form, so "Candles" and "candle" count as one.
  3. Give each word a weight: its count divided by the count of the most common word (0-1).
  4. Score each sentence by adding up the weights of its words.
  5. Keep the top-scoring sentences, in the order they appeared in the text.

Adding weights favors longer, detail-rich sentences. Averaging them would favor short
ones, but then a tiny sentence with a single key word can beat everything else.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language
from spacy.tokens import Doc, Token

MODEL = "en_core_web_sm"
DEFAULT_SENTENCES = 3


@dataclass(frozen=True)
class ScoredSentence:
    index: int  # position in the original text
    text: str
    score: float


@lru_cache(maxsize=1)
def load_nlp() -> Language:
    """Load the spaCy English model once and reuse it."""
    return spacy.load(MODEL)


def _is_content(token: Token) -> bool:
    return not (token.is_punct or token.is_space or token.lower_ in STOP_WORDS)


def _key(token: Token) -> str:
    return token.lemma_.lower()


def word_weights(doc: Doc) -> dict[str, float]:
    """Each content word's count divided by the most common word's count."""
    counts = Counter(_key(t) for t in doc if _is_content(t))
    if not counts:
        return {}
    top = max(counts.values())
    return {word: count / top for word, count in counts.items()}


def score_sentences(doc: Doc) -> list[ScoredSentence]:
    """Score every sentence on its own words only, so its position doesn't matter."""
    weights = word_weights(doc)
    scored = []
    for index, sentence in enumerate(doc.sents):
        text = sentence.text.strip()
        if not text:
            continue
        score = sum(weights[_key(t)] for t in sentence if _is_content(t))
        scored.append(ScoredSentence(index=index, text=text, score=score))
    return scored


def sentence_count(total: int, sentences: int | None = None, ratio: float | None = None) -> int:
    """How many sentences to keep: an exact number, or a share of the total (at least 1)."""
    if sentences is not None and ratio is not None:
        raise ValueError("Choose either a number of sentences or a ratio, not both.")
    if ratio is not None:
        if not 0 < ratio <= 1:
            raise ValueError("The ratio must be between 0 and 1, like 0.2 for 20%.")
        wanted = round(total * ratio)
    else:
        wanted = DEFAULT_SENTENCES if sentences is None else sentences
        if wanted < 1:
            raise ValueError("Ask for at least 1 sentence.")
    return max(1, min(total, wanted))


def summarize(text: str, sentences: int | None = None, ratio: float | None = None) -> str:
    """Return the most important sentences of `text`, in their original order."""
    if not text or not text.strip():
        raise ValueError("There's no text to summarize.")
    scored = score_sentences(load_nlp()(text.strip()))
    keep = sentence_count(len(scored), sentences=sentences, ratio=ratio)
    best = sorted(scored, key=lambda s: s.score, reverse=True)[:keep]
    return " ".join(s.text for s in sorted(best, key=lambda s: s.index))
