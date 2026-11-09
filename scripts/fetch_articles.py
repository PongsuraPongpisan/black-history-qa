"""Download the Wikipedia articles listed in app/config.py and split them into sections.

    python -m scripts.fetch_articles

Each article is saved to data/raw/ as soon as it arrives, so if the run stops halfway,
running it again only downloads the missing ones. Then it writes data/processed/sections.jsonl
(the corpus) and data/ATTRIBUTION.md (licence + exact revisions). Commit all three.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

from app.config import ARTICLES, ATTRIBUTION_MD, ROOT
from app.corpus import parse_extract, save_sections, slug

API = "https://en.wikipedia.org/w/api.php"
RAW_DIR = ROOT / "data" / "raw"
# Wikipedia asks API users to identify themselves: https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = f"black-history-qa/1.0 (university course project; {os.getenv('CONTACT_EMAIL', 'no contact given')})"
PAUSE_SECONDS = 3      # between articles
MAX_RETRIES = 5


def fetch(title: str) -> dict:
    params = {"action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "wiki",
              "inprop": "url", "redirects": 1, "titles": title, "format": "json", "formatversion": 2}
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                page = json.load(resp)["query"]["pages"][0]
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == MAX_RETRIES:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 30 * attempt
            print(f"   Wikipedia says slow down (HTTP {e.code}); waiting {wait} s, then retry {attempt}/{MAX_RETRIES - 1}")
            time.sleep(wait)
    if page.get("missing") or not page.get("extract"):
        raise ValueError(f"Wikipedia has no article called {title!r}")
    return page


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sections, rows = [], []
    for title in ARTICLES:
        cache = RAW_DIR / f"{slug(title)}.json"
        cached = cache.exists()
        if cached:
            page = json.loads(cache.read_text(encoding="utf-8"))
        else:
            page = fetch(title)
            cache.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
            time.sleep(PAUSE_SECONDS)
        found = parse_extract(page["title"], page["extract"])
        sections.extend(found)
        rows.append(f"| [{page['title']}]({page['fullurl']}) | {page['lastrevid']} | {len(found)} |")
        print(f"{page['title']:45s} {len(found):3d} sections{'  (already downloaded)' if cached else ''}")
    save_sections(sections)
    ATTRIBUTION_MD.write_text(
        "# Attribution\n\n"
        "The corpus in `data/processed/sections.jsonl` is text from English Wikipedia, licensed under "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). It was split into sections; "
        "reference lists were removed. Authors: see each article's history page.\n\n"
        f"Downloaded {date.today().isoformat()} by `scripts/fetch_articles.py`.\n\n"
        "| Article | Revision | Sections |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n",
        encoding="utf-8")
    print(f"\n{len(sections)} sections from {len(ARTICLES)} articles saved.")


if __name__ == "__main__":
    main()
