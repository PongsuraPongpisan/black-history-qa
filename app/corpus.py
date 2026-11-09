"""Split Wikipedia plain-text extracts into sections and load/save them.

The MediaWiki API (prop=extracts, explaintext) marks headings as "== Heading ==", "=== Sub ===".
Each heading becomes one Section; the text before the first heading is the "Overview".
"""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import SECTIONS_JSONL

HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
SKIP_SECTIONS = {"see also", "references", "notes", "further reading", "external links", "bibliography",
                 "sources", "citations", "footnotes", "works cited", "general and cited sources",
                 "explanatory notes", "primary sources", "secondary sources"}


@dataclass
class Section:
    id: str          # "harriet-tubman#early-life": what answers cite
    article: str     # "Harriet Tubman"
    heading: str     # "Early life" (sub-sections are "Parent > Child")
    url: str         # link to the section on Wikipedia
    text: str


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def section_url(article: str, heading: str) -> str:
    base = "https://en.wikipedia.org/wiki/" + article.replace(" ", "_")
    last = heading.split(" > ")[-1]
    return base if heading == "Overview" else f"{base}#{last.replace(' ', '_')}"


def parse_extract(article: str, extract: str) -> list[Section]:
    sections, path, buf = [], ["Overview"], []

    def flush():
        text = "\n".join(line for line in buf if line.strip()).strip()
        skipped = any(p.lower() in SKIP_SECTIONS for p in path)
        if text and not skipped:
            heading = " > ".join(path)
            sid, n = f"{slug(article)}#{slug(heading)}", 2
            while any(s.id == sid for s in sections):
                sid, n = f"{slug(article)}#{slug(heading)}-{n}", n + 1
            sections.append(Section(sid, article, heading, section_url(article, heading), text))

    for line in extract.split("\n"):
        m = HEADING.match(line.strip())
        if m:
            flush()
            buf = []
            level = len(m.group(1)) - 1          # "==" -> 1, "===" -> 2
            parents = path[:level - 1] if path[0] != "Overview" else []
            path = parents + [m.group(2)]
        else:
            buf.append(line)
    flush()
    return sections


def save_sections(sections: list[Section], path: Path = SECTIONS_JSONL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(s), ensure_ascii=False) for s in sections) + "\n", encoding="utf-8")


def load_sections(path: Path = SECTIONS_JSONL) -> list[Section]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m scripts.fetch_articles` first (see README).")
    return [Section(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
