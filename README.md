# Black History Q&A

<!-- TODO (write in your own words): what the app does, who it is for, and why you chose this topic. -->

A question-answering website about the history of Black people, from the Atlantic slave trade to the civil rights movement and the end of apartheid. Answers come only from 22 English Wikipedia articles, and every fact is footnoted with a link to the section it came from. The app refuses to give opinions or make judgments about groups of people.

All LLM calls go to the class proxy: `OpenAI()` with `model="gemma-4-E4B-it"`.

**Live app:** <!-- TODO: your Render link -->

## Course capability families covered

| # | Family | Where |
|---|---|---|
| 1 | Structured output with schema validation | `app/schemas.py`: every agent step is JSON validated by Pydantic, and the API request and response are validated too |
| 2 | RAG with citations | `app/corpus.py` (Wikipedia → sections), `app/retriever.py` (section chunks + BM25), `app/rag.py` (section citations, `min_score` floor) |
| 3 | Retrieval and answer evaluation | `eval/`: labelled questions, recall@k, MRR, accuracy, refusal and citation accuracy, baselines |
| 4 | Agent loop with tools | `app/agent.py`, `app/tools.py`: allow-listed `search_history` and `read_section` tools, stop on `final_answer` or after 5 steps |
| 6 | Served interface | `server.py` + `static/index.html`: web page and `POST /ask` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # class key
export OPENAI_BASE_URL=...       # class proxy URL
```

The corpus is already in the repo (`data/processed/sections.jsonl`). To download it again:

```bash
python -m scripts.fetch_articles
```

The articles are from English Wikipedia under CC BY-SA 4.0; exact revisions are listed in `data/ATTRIBUTION.md`.

## Run

```bash
uvicorn server:app --reload          # website at http://127.0.0.1:8000, API docs at /docs
python cli.py "When was the Emancipation Proclamation issued?"
python cli.py --mode agent "Why did Black Americans move north during the Great Migration?"
python -m pytest -q                  # offline tests, no API key needed
```

## Evaluation

```bash
python -m scripts.check_questions                       # check each expected answer is really in its articles
python -m eval.eval_retrieval                           # no LLM needed
python -m eval.eval_answers --mode none                 # baseline: no retrieval
python -m eval.eval_answers --mode rag --min-score 0    # RAG, no score floor
python -m eval.eval_answers --mode rag                  # RAG, min_score from config
python -m eval.eval_answers --mode agent                # agent
python -m eval.eval_answers --summary
python -m eval.show_failures > results/failures.txt
```

<!-- TODO: describe your test set, paste the retrieval and answer tables, and explain your min_score choice. -->

## Limitations

<!-- TODO: in your own words, after looking at the failures. -->
