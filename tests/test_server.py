"""Offline tests for the web server: the model and the corpus are faked."""
from fastapi.testclient import TestClient

import server
from app.corpus import Section
from app.ratelimit import RateLimiter

FAKE = [Section("example-person#early-life", "Example Person", "Early life",
                "https://en.wikipedia.org/wiki/Example_Person#Early_life", "Born in 1820.\nRaised on a farm.")]


def make_client(monkeypatch, per_minute=6, per_day=300):
    monkeypatch.setattr(server, "ask", lambda q, mode: {"answer": "Born in 1820 [example-person#early-life].",
                                                         "citations": ["example-person#early-life"]})
    monkeypatch.setattr(server, "get_sections", lambda: FAKE)
    monkeypatch.setattr(server, "limiter", RateLimiter(per_minute, per_day))
    return TestClient(server.app)


def test_index_page_is_served(monkeypatch):
    res = make_client(monkeypatch).get("/")
    assert res.status_code == 200 and "Black history" in res.text


def test_ask_returns_cited_section(monkeypatch):
    res = make_client(monkeypatch).post("/ask", json={"question": "When was she born?", "mode": "rag"})
    assert res.status_code == 200
    src = res.json()["sources"][0]
    assert src["title"] == "Example Person — Early life" and src["url"].endswith("#Early_life")


def test_per_minute_limit(monkeypatch):
    client = make_client(monkeypatch, per_minute=2)
    codes = [client.post("/ask", json={"question": "When was she born?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_model_failure_gives_clear_error(monkeypatch):
    client = make_client(monkeypatch)
    def boom(q, mode):
        raise RuntimeError("proxy down")
    monkeypatch.setattr(server, "ask", boom)
    res = client.post("/ask", json={"question": "When was she born?"})
    assert res.status_code == 502 and "Try again" in res.json()["detail"]


def test_invalid_question_rejected(monkeypatch):
    assert make_client(monkeypatch).post("/ask", json={"question": "x"}).status_code == 422
