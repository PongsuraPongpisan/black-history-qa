"""Offline tests: no LLM calls and no downloads (a small invented article is used as the corpus)."""
from pathlib import Path

import pytest

import app.agent as agent_mod
from app.config import REFUSAL
from app.corpus import parse_extract
from app.rag import answer_rag, extract_citations
from app.retriever import build_retriever, chunk_by_section, chunk_fixed
from app.schemas import parse_agent_step
from app.tools import make_tools, run_tool
from eval.metrics import first_relevant_rank, score_answer

FIXTURE = (Path(__file__).parent / "fixtures" / "sample_extract.txt").read_text()


@pytest.fixture(scope="module")
def sections():
    return parse_extract("Example Person", FIXTURE)


@pytest.fixture(scope="module")
def retriever(sections):
    return build_retriever("section", "bm25", sections)


def test_parser_builds_sections_and_drops_reference_lists(sections):
    ids = [s.id for s in sections]
    assert ids == ["example-person#overview", "example-person#early-life",
                   "example-person#early-life-family", "example-person#the-river-line"]
    assert sections[3].url == "https://en.wikipedia.org/wiki/Example_Person#The_River_Line"


def test_nested_headings():
    secs = parse_extract("T", "x\n== A ==\na\n=== A1 ===\na1\n=== A2 ===\na2\n== B ==\nb\n== Notes ==\nn\n")
    assert [s.heading for s in secs] == ["Overview", "A", "A > A1", "A > A2", "B"]


def test_chunkers_keep_track_of_sections(sections):
    assert {c.section_id for c in chunk_by_section(sections)} == {s.id for s in sections}
    assert all(c.article == "Example Person" for c in chunk_fixed(sections))


def test_retriever_finds_the_right_section(retriever):
    chunk, _ = retriever.search("how many people did she guide to freedom", k=1)[0]
    assert chunk.section_id == "example-person#the-river-line"


def test_min_score_refuses_without_calling_the_model(retriever):
    assert answer_rag("stock market prices", retriever, min_score=5.0)["answer"] == REFUSAL


def test_read_section_validates_input(sections, retriever):
    tools = make_tools(retriever, sections, 0.0)
    assert "70 people" in run_tool(tools, "read_section", "example-person#the-river-line")
    assert run_tool(tools, "read_section", "nope#missing").startswith("Error")
    assert run_tool(tools, "read_section", "../../etc/passwd").startswith("Error")
    assert run_tool(tools, "delete_files", "x").startswith("Error: unknown tool")


def test_parse_agent_step_handles_fences_and_rejects_bad_actions():
    step = parse_agent_step('```json\n{"thought": "t", "action": "read_section", "action_input": "a#b"}\n```')
    assert step.action == "read_section"
    with pytest.raises(ValueError):
        parse_agent_step('{"action": "run_shell", "action_input": "ls"}')
    with pytest.raises(ValueError):
        parse_agent_step("The answer is 1849.")


def test_citations_are_filtered_to_real_sections():
    valid = {"example-person#early-life"}
    assert extract_citations("Born 1820 [example-person#early-life] [fake#id]", valid) == ["example-person#early-life"]


def test_agent_loop_with_fake_model(monkeypatch, sections, retriever):
    replies = iter([
        "not json",  # invalid -> the agent asks again
        '{"thought": "search", "action": "search_history", "action_input": "River Line"}',
        '{"thought": "read it", "action": "read_section", "action_input": "example-person#the-river-line"}',
        '{"thought": "done", "action": "final_answer", "action_input": "She guided 70 people [example-person#the-river-line]."}',
    ])
    monkeypatch.setattr(agent_mod, "chat", lambda messages: next(replies))
    out = agent_mod.run_agent("How many people did she guide?", retriever, sections)
    assert out["answer"].startswith("She guided 70 people")
    assert out["citations"] == ["example-person#the-river-line"] and out["steps"] == 4


def test_agent_stops_at_step_limit(monkeypatch, sections, retriever):
    monkeypatch.setattr(agent_mod, "chat", lambda m: '{"thought": "", "action": "search_history", "action_input": "x"}')
    out = agent_mod.run_agent("loop forever", retriever, sections, max_steps=3)
    assert out["steps"] == 3 and "step limit" in out["answer"]


def test_scoring_rules():
    assert score_answer({"type": "single", "must_include": ["1863"]}, "It was issued in 1863 [x#y].")
    assert not score_answer({"type": "single", "must_include": ["1863"]}, REFUSAL)
    assert score_answer({"type": "unanswerable", "must_include": []}, REFUSAL)
    assert first_relevant_rank(["A", "B"], ["B"]) == 2


def test_agent_is_told_to_answer_on_its_last_step(monkeypatch, sections, retriever):
    seen = []
    def fake_chat(messages):
        seen.append(messages[-1]["content"])
        if "This is your last step" in messages[-1]["content"]:
            return '{"thought": "", "action": "final_answer", "action_input": "70 people [example-person#the-river-line]."}'
        return '{"thought": "", "action": "search_history", "action_input": "River Line"}'
    monkeypatch.setattr(agent_mod, "chat", fake_chat)
    out = agent_mod.run_agent("How many people?", retriever, sections, max_steps=3)
    assert out["answer"].startswith("70 people") and out["steps"] == 3
    assert sum("This is your last step" in m for m in seen) == 1
