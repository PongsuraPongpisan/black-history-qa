"""Retrieval-augmented answering with section citations, plus a no-retrieval baseline."""
import re

from app.config import MIN_SCORE, REFUSAL, TOP_K
from app.llm import chat
from app.retriever import BM25Retriever

RAG_SYSTEM = f"""You answer questions about Black history using ONLY the encyclopedia excerpts provided.
Rules:
- Answer in 1-3 short, factual sentences.
- After each fact, cite the id of the excerpt it came from in square brackets, e.g. [harriet-tubman#early-life].
- Do not use outside knowledge, and do not guess.
- Do not give opinions, and do not make judgments or generalisations about any group of people.
- If the excerpts do not contain the answer, or the question asks for an opinion, reply exactly: {REFUSAL}"""

BASELINE_SYSTEM = f"""You answer questions about Black history in 1-3 short, factual sentences.
Do not give opinions or make judgments about any group of people.
If you do not know the answer, or the question asks for an opinion, reply exactly: {REFUSAL}"""

CITATION_RE = re.compile(r"\[([a-z0-9-]+#[a-z0-9-]+)\]")


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Model output is untrusted: keep only citations that point at real sections."""
    return [c for c in dict.fromkeys(CITATION_RE.findall(answer)) if c in valid_ids]


def format_context(hits) -> str:
    return "\n\n".join(f"[{c.section_id}] {c.title}\n{c.text}" for c, _ in hits)


def valid_ids(retriever: BM25Retriever) -> set[str]:
    return {c.section_id for c in retriever.chunks}


def answer_rag(question: str, retriever: BM25Retriever, k: int = TOP_K, min_score: float = MIN_SCORE) -> dict:
    hits = retriever.search(question, k=k, min_score=min_score)
    if not hits:  # nothing passed the score floor: refuse without calling the model
        return {"answer": REFUSAL, "citations": [], "retrieved": []}
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"Excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"},
    ]
    answer = chat(messages).strip()
    return {
        "answer": answer,
        "citations": extract_citations(answer, valid_ids(retriever)),
        "retrieved": [{"id": c.id, "score": round(s, 2)} for c, s in hits],
    }


def answer_no_retrieval(question: str) -> dict:
    """Baseline: the same model with no access to the articles."""
    messages = [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": question}]
    return {"answer": chat(messages).strip(), "citations": [], "retrieved": []}
