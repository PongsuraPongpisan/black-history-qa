"""Chunk the article sections and search them with BM25 (plus a naive word-overlap baseline)."""
import math
import re
from collections import Counter
from dataclasses import dataclass

from app.corpus import Section, load_sections

STOPWORDS = set(
    "a an the and or of to in on for at by with is are was were be been it its this that i my me we our "
    "you your do does did can how what when where who which much many there their they as from if per "
    "after before about any into than then so will would he she his her him".split()
)
MAX_WORDS = 250  # a section longer than this is split at paragraph boundaries


@dataclass
class Chunk:
    id: str           # "harriet-tubman#early-life:2" (internal)
    section_id: str   # "harriet-tubman#early-life" (what answers cite)
    article: str
    title: str        # "Harriet Tubman — Early life"
    text: str


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def chunk_by_section(sections: list[Section]) -> list[Chunk]:
    """Paragraphs of one section packed into chunks of at most MAX_WORDS words."""
    chunks = []
    for s in sections:
        parts, buf, words = [], [], 0
        for para in s.text.split("\n"):
            n = len(para.split())
            if buf and words + n > MAX_WORDS:
                parts.append(buf); buf, words = [], 0
            buf.append(para); words += n
        if buf:
            parts.append(buf)
        for i, p in enumerate(parts, 1):
            chunks.append(Chunk(f"{s.id}:{i}", s.id, s.article, f"{s.article} — {s.heading}", "\n".join(p)))
    return chunks


def chunk_fixed(sections: list[Section], size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Naive default: fixed word windows over each whole article, ignoring its headings."""
    by_article: dict[str, list[Section]] = {}
    for s in sections:
        by_article.setdefault(s.article, []).append(s)
    chunks = []
    for article, secs in by_article.items():
        words, owner = [], []
        for s in secs:
            for w in s.text.split():
                words.append(w); owner.append(s.id)
        for i, start in enumerate(range(0, len(words), size - overlap)):
            end = min(start + size, len(words))
            chunks.append(Chunk(f"{secs[0].id.split('#')[0]}#w{i}", owner[start], article, article,
                                " ".join(words[start:end])))
            if end == len(words):
                break
    return chunks


CHUNKERS = {"section": chunk_by_section, "fixed": chunk_fixed}


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        ranked = sorted(zip(self.chunks, self.score(query)), key=lambda x: x[1], reverse=True)
        return [(c, s) for c, s in ranked[:k] if s > 0 and s >= min_score]


class OverlapRetriever(BM25Retriever):
    """Baseline: score = number of distinct query words in the chunk (no IDF, no length normalisation)."""

    def score(self, query: str) -> list[float]:
        q = set(tokenize(query))
        return [float(len(q & tf.keys())) for tf in self.tfs]


def build_retriever(chunking: str = "section", kind: str = "bm25",
                    sections: list[Section] | None = None) -> BM25Retriever:
    chunks = CHUNKERS[chunking](sections if sections is not None else load_sections())
    return BM25Retriever(chunks) if kind == "bm25" else OverlapRetriever(chunks)
