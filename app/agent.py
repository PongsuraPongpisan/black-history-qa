"""A small tool-using agent loop.

The model replies with one JSON object per step (validated by AgentStep). We run the chosen tool,
feed the result back as an observation, and stop on final_answer or after AGENT_MAX_STEPS steps.
"""
from pydantic import ValidationError

from app.config import AGENT_MAX_STEPS, MIN_SCORE, REFUSAL
from app.corpus import Section
from app.llm import chat
from app.rag import extract_citations, valid_ids
from app.retriever import BM25Retriever
from app.schemas import parse_agent_step
from app.tools import make_tools, run_tool

AGENT_SYSTEM = f"""You answer questions about Black history. You can use tools to read encyclopedia articles.

Tools:
- search_history: input is a short search query. Returns matching excerpts, each with an id in square brackets.
- read_section: input is an excerpt id such as harriet-tubman#early-life. Returns that whole section.
  Use it when an excerpt is cut off or you need more of the same section.
- final_answer: input is your answer for the user (1-3 factual sentences), citing ids like [harriet-tubman#early-life].

Rules:
- Always use search_history before answering. Use only facts from the excerpts you read.
- Do not give opinions, and do not make judgments or generalisations about any group of people.
- If the excerpts do not contain the answer, or the question asks for an opinion, your final answer must be exactly: {REFUSAL}

Reply with ONE JSON object and nothing else, in this format:
{{"thought": "what you will do next", "action": "search_history" | "read_section" | "final_answer", "action_input": "..."}}"""


def run_agent(question: str, retriever: BM25Retriever, sections: list[Section], max_steps: int = AGENT_MAX_STEPS,
              min_score: float = MIN_SCORE, verbose: bool = False) -> dict:
    tools = make_tools(retriever, sections, min_score)
    messages = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": question}]
    trace = []

    for step_no in range(1, max_steps + 1):
        if step_no == max_steps:
            # Last chance: without this, the model can keep searching until the limit and give no answer.
            # Appended to the last user message so the conversation still alternates user/assistant.
            messages[-1]["content"] += ("\n\nThis is your last step. Reply now with action final_answer, "
                                        f"using only what you have read. If nothing you read answers it, give: {REFUSAL}")
        raw = chat(messages)
        messages.append({"role": "assistant", "content": raw})
        try:
            step = parse_agent_step(raw)
        except (ValueError, ValidationError) as e:
            trace.append({"step": step_no, "error": f"invalid step: {str(e)[:200]}"})
            messages.append({"role": "user", "content":
                             "Your reply was not a valid JSON object in the required format. "
                             "Reply with exactly one JSON object with keys thought, action, action_input."})
            continue

        if verbose:
            print(f"[step {step_no}] {step.action}: {step.action_input}")

        if step.action == "final_answer":  # stop condition 1: the model is done
            trace.append({"step": step_no, "action": "final_answer"})
            return {"answer": step.action_input.strip(),
                    "citations": extract_citations(step.action_input, valid_ids(retriever)),
                    "steps": step_no, "trace": trace}

        observation = run_tool(tools, step.action, step.action_input)
        trace.append({"step": step_no, "action": step.action, "input": step.action_input,
                      "observation": observation[:200]})
        if verbose:
            print(f"    -> {observation[:150]!r}")
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    # stop condition 2: step limit reached
    return {"answer": "Sorry, I couldn't finish answering within the step limit.",
            "citations": [], "steps": max_steps, "trace": trace}
