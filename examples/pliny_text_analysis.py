#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""
Text analysis of Pliny's Natural History, demonstrating text preprocessing
(normalization, tokenization), chunking, and trigram fuzzy matching.
Pure stdlib -- no external deps.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

TEXT_PATH = Path(__file__).resolve().parent.parent / "natural-history.djvu.txt"

# Sample from three regions of the text for topical variety:
#   Book II (cosmology/elements), Book III (geography), Book V (Africa).
REGIONS = [(2319, 2500), (9800, 10000), (23369, 23550)]
MAX_CHARS = 8000


def _is_footnote(line: str) -> bool:
    """Heuristic: skip footnote lines (start with digit or symbol marker)."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(re.match(r"^[\d^*\u2020\u2021]", stripped))


def load_text() -> str:
    """Load body text from multiple regions, skipping footnotes."""
    all_lines = TEXT_PATH.read_text(encoding="utf-8").splitlines()
    buf: list[str] = []
    total = 0
    for start, end in REGIONS:
        for line in all_lines[start:end]:
            if _is_footnote(line):
                continue
            if total + len(line) > MAX_CHARS:
                break
            buf.append(line)
            total += len(line)
    return "\n".join(buf)


def scrub(text: str) -> str:
    """Normalize text to a canonical form.

    NFC normalization, case folding, diacritics stripping, whitespace collapse.
    """
    # NFC normalize
    text = unicodedata.normalize("NFC", text)
    # Strip combining marks (diacritics)
    text = "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    # Case fold
    text = text.casefold()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def words(text: str) -> list[str]:
    """Tokenize into words.

    Splits on non-alphabetic boundaries, drops short tokens.
    """
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def sentences(text: str) -> list[str]:
    """Split into sentences."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


raw = load_text()
normalized = scrub(raw)
word_tokens = words(normalized)
vocab = sorted(set(word_tokens))

print("=" * 60)
print("1. TEXT PREPROCESSING")
print("=" * 60)
print(f"   Raw text length:  {len(raw):,} chars")
print(f"   Normalized length: {len(normalized):,} chars")
print(f"   Word tokens:       {len(word_tokens):,}")
print(f"   Vocabulary size:   {len(vocab):,}")
print(f"   Sample (first 12): {word_tokens[:12]}")
print()

CHUNK_SIZE_WORDS = 80


def chunk_fixed(tokens: list[str], size: int) -> list[list[str]]:
    """Fixed-size word chunks."""
    return [tokens[i : i + size] for i in range(0, len(tokens), size)]


def chunk_sentences(text: str, max_words: int) -> list[str]:
    """Sentence-based chunking.

    Greedily packs sentences until the word budget is reached.
    """
    sents = sentences(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for s in sents:
        sw = len(s.split())
        if buf_words + sw > max_words and buf:
            chunks.append(" ".join(buf))
            buf, buf_words = [], 0
        buf.append(s)
        buf_words += sw
    if buf:
        chunks.append(" ".join(buf))
    return chunks


fixed_chunks = chunk_fixed(word_tokens, CHUNK_SIZE_WORDS)
sentence_chunks = chunk_sentences(raw, CHUNK_SIZE_WORDS)

print("=" * 60)
print("2. CHUNKING")
print("=" * 60)
print(f"   Fixed chunks ({CHUNK_SIZE_WORDS} words each): {len(fixed_chunks)}")
print(f"   Sentence chunks (max {CHUNK_SIZE_WORDS} words): {len(sentence_chunks)}")
if fixed_chunks:
    print(f"   Fixed chunk 0 preview: {' '.join(fixed_chunks[0][:15])}...")
if sentence_chunks:
    preview = sentence_chunks[0][:120]
    print(f"   Sentence chunk 0 preview: {preview}...")
print()

def char_trigrams(s: str) -> set[str]:
    """Generate character trigrams."""
    s = s.lower()
    if len(s) < 3:
        return {s}
    return {s[i : i + 3] for i in range(len(s) - 2)}


def trigram_jaccard(a: str, b: str) -> float:
    """Trigram Jaccard similarity."""
    ta, tb = char_trigrams(a), char_trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class TrigramIndex:
    """Trigram inverted index for candidate generation."""

    def __init__(self) -> None:
        self._index: dict[str, set[int]] = {}
        self._docs: dict[int, str] = {}

    def add(self, doc_id: int, term: str) -> None:
        self._docs[doc_id] = term
        for tri in char_trigrams(term):
            self._index.setdefault(tri, set()).add(doc_id)

    def candidates(self, query: str) -> list[int]:
        """Union of posting lists for query trigrams."""
        hits: set[int] = set()
        for tri in char_trigrams(query):
            hits |= self._index.get(tri, set())
        return sorted(hits)

    def search(self, query: str, threshold: float = 0.2) -> list[tuple[str, float]]:
        """Candidate generation + trigram Jaccard verification."""
        results = []
        for doc_id in self.candidates(query):
            term = self._docs[doc_id]
            sim = trigram_jaccard(query, term)
            if sim >= threshold:
                results.append((term, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# Build index over vocabulary
ix = TrigramIndex()
for i, term in enumerate(vocab):
    ix.add(i, term)

# Pliny's OCR text has spelling variants -- search for some.
# These demonstrate the real-world use case: recovering correct terms
# from noisy OCR output or archaic/variant spellings.
QUERIES = [
    ("historie", "OCR variant of 'history'"),
    ("naturall", "early-modern variant of 'natural'"),
    ("mountaine", "archaic variant of 'mountain'"),
    ("provynce", "archaic variant of 'province'"),
    ("elemenns", "OCR variant of 'elements'"),
    ("Spaine", "archaic variant of 'spain'"),
]

print("=" * 60)
print("3. FUZZY MATCHING (trigram index)")
print("=" * 60)
print(f"   Index size: {len(vocab)} terms")
print()
for query, desc in QUERIES:
    matches = ix.search(query, threshold=0.3)[:5]
    match_str = ", ".join(f"{t} ({s:.2f})" for t, s in matches) if matches else "(none)"
    print(f"   '{query}' ({desc})")
    print(f"     -> {match_str}")
print()

freq = Counter(word_tokens)

# Common English stopwords (inlined for zero-dependency)
STOPWORDS = frozenset(
    "the of and in to it is that by as are this which not there he "
    "was for with his from but or be at an on had all its they were "
    "have been has no their may we more than so do her she de et ii "
    "lib cap see also who did would what into any one them our him".split()
)
freq_content = Counter({w: c for w, c in freq.items() if w not in STOPWORDS})

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "astronomy": ["stars", "sun", "moon", "planets", "eclipse", "heavens", "comet"],
    "geography": ["earth", "world", "sea", "ocean", "land", "region", "country"],
    "elements": ["fire", "water", "air", "wind", "rain", "thunder", "lightning"],
    "philosophy": ["nature", "god", "divine", "reason", "opinion", "doctrine"],
}

print("=" * 60)
print("4. ANALYSIS")
print("=" * 60)
print()
print("   Top 20 content words (stopwords removed):")
for word, count in freq_content.most_common(20):
    bar = "#" * min(count, 40)
    print(f"     {word:<14} {count:>4}  {bar}")

print()
print("   Topic presence (keyword group counts):")
for topic, keywords in TOPIC_KEYWORDS.items():
    total = sum(freq.get(k, 0) for k in keywords)
    found = [k for k in keywords if freq.get(k, 0) > 0]
    print(f"     {topic:<14} {total:>4} hits  terms: {', '.join(found) or '(none)'}")

# Hapax legomena (words appearing exactly once)
hapax = [w for w, c in freq.items() if c == 1]
print()
print(
    f"   Hapax legomena: {len(hapax)} / {len(vocab)} ({100 * len(hapax) / max(len(vocab), 1):.0f}% of vocabulary)"
)
print(f"   Sample hapax:   {hapax[:8]}")

# Type-token ratio
ttr = len(vocab) / max(len(word_tokens), 1)
print(f"   Type-token ratio: {ttr:.3f}")
