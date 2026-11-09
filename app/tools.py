"""Tools the agent may call. Only names in the allow-list can run, and inputs are checked first."""
import re

from app.config import TOP_K
from app.corpus import Section
from app.rag import format_context
from app.retriever import BM25Retriever

SECTION_ID = re.compile(r"^[a-z0-9-]+#[a-z0-9-]+$")
MAX_OBSERVATION_CHARS = 3000


def make_tools(retriever: BM25Retriever, sections: list[Section], min_score: float) -> dict:
    by_id = {s.id: s for s in sections}

    def search_history(query: str) -> str:
        hits = retriever.search(query, k=TOP_K, min_score=min_score)
        return format_context(hits) if hits else "No relevant excerpts found."

    def read_section(section_id: str) -> str:
        """Return a whole section, e.g. "harriet-tubman#early-life"."""
        section_id = section_id.strip().strip("[]")
        if not SECTION_ID.match(section_id):
            return "Error: give a section id such as harriet-tubman#early-life"
        if section_id not in by_id:
            return f"Error: section {section_id} does not exist"
        s = by_id[section_id]
        return f"[{s.id}] {s.article} — {s.heading}\n{s.text}"

    return {"search_history": search_history, "read_section": read_section}


def run_tool(tools: dict, name: str, arg: str) -> str:
    if name not in tools:  # allow-list check
        return f"Error: unknown tool '{name}'"
    return tools[name](arg)[:MAX_OBSERVATION_CHARS]  # cap what goes back to the model
