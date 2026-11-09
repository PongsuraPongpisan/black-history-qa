"""Web app + REST API.

    uvicorn server:app --reload      then open http://127.0.0.1:8000  (API docs at /docs)
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import DAILY_LIMIT, RATE_LIMIT_PER_MINUTE
from app.ratelimit import RateLimiter
from app.schemas import AskRequest, AskResponse, Source
from app.service import ask, get_sections

app = FastAPI(title="Black History Q&A")
limiter = RateLimiter(RATE_LIMIT_PER_MINUTE, DAILY_LIMIT)
INDEX = Path(__file__).resolve().parent / "static" / "index.html"
MAX_SOURCE_CHARS = 1500


def client_id(request: Request) -> str:
    """Hosting platforms put the visitor's real IP first in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def section_sources(citations: list[str]) -> list[Source]:
    """The cited sections (trimmed at a paragraph boundary), so readers can check the answer."""
    by_id = {sec.id: sec for sec in get_sections()}
    out = []
    for c in citations:
        sec = by_id.get(c)
        if not sec:
            continue
        text = ""
        for para in sec.text.split("\n"):
            if text and len(text) + len(para) > MAX_SOURCE_CHARS:
                text += "\n…"
                break
            text += ("\n" if text else "") + para
        out.append(Source(id=sec.id, title=f"{sec.article} — {sec.heading}", url=sec.url, text=text))
    return out


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest, request: Request) -> AskResponse:
    limited = limiter.check(client_id(request))
    if limited:
        raise HTTPException(status_code=429, detail=limited)
    try:
        result = ask(req.question, mode=req.mode)
    except Exception as e:  # the model service is down, slow or refused the request
        raise HTTPException(status_code=502, detail="The model service didn't respond. Try again in a minute.") from e
    return AskResponse(answer=result["answer"], citations=result["citations"],
                       sources=section_sources(result["citations"]), mode=req.mode)
