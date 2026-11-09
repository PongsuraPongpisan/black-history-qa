# Black History Q&A

I built this app for students and anyone curious about Black history who wants quick answers they can check. General chatbots answer history questions fluently but give no sources, and they sometimes state wrong details with full confidence. This app answers only from 22 Wikipedia articles and footnotes every fact, so the reader can open the exact section it came from.

A question-answering website about the history of Black people, from the Atlantic slave trade to the civil rights movement and the end of apartheid. Answers come only from 22 English Wikipedia articles, and every fact is footnoted with a link to the section it came from. The app refuses to give opinions or make judgments about groups of people.

All LLM calls go to the class proxy: `OpenAI()` with `model="gemma-4-E4B-it"`.

**Live app:** <https://black-history-qa.onrender.com>

## Course capability families covered

| # | Family | Where |
| --- | --- | --- |
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

`eval/questions.jsonl` has 42 questions, each checked against the downloaded articles with `scripts/check_questions.py`:

- 25 single-fact questions about well-known events and people
- 10 detail questions about specific facts inside the articles (names, numbers, dates)
- 3 hard questions worded differently from the article text, or needing two articles
- 4 questions the app must refuse: two that ask for judgments about racial groups, a prediction and an opinion

### Retrieval

| config | recall@1 | recall@3 | MRR |
| --- | --- | --- | --- |
| word overlap + section chunks (baseline) | 0.679 | 0.893 | 0.795 |
| BM25 + fixed 200-word chunks (naive default) | 0.786 | 1.000 | 0.887 |
| **BM25 + section chunks (used)** | **0.857** | 0.964 | **0.914** |

BM25 beats the word-overlap baseline (MRR 0.914 vs 0.795), and chunking by Wikipedia section beats fixed 200-word windows on recall@1 (0.857 vs 0.786). Section chunks keep a heading's text together, and the heading itself is indexed, which helps questions that name a topic.

### Answers

| run | overall | answerable | detail | hard | refusal | citation | sec/q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no retrieval (baseline) | 0.952 | 0.947 | 0.900 | 1.000 | 1.000 | 0.000 | 0.5 |
| **RAG, min_score 0 (used)** | **0.905** | 0.895 | **1.000** | 0.667 | 1.000 | **0.789** | 1.0 |
| RAG, min_score 10 | 0.881 | 0.895 | 1.000 | 0.667 | 0.750 | 0.789 | 0.9 |
| agent, min_score 10 | 0.619 | 0.605 | 0.800 | 0.333 | 0.750 | 0.526 | 5.3 |

- **The model already knows this topic.** Without retrieval, Gemma answers 95% of questions correctly, because famous dates and names are all over its training data. So RAG does not raise overall accuracy here.
- **RAG wins where memory fails.** On detail questions RAG scores 1.000 vs 0.900. Without retrieval, the model said the Combahee Ferry raid freed "approximately 75" people; the article says over 700.
- **Every RAG answer is checkable.** 79% of RAG answers cite a section from the right article; the baseline cites nothing.
- **min_score choice.** The retrieval sweep showed 10 was the highest floor that kept every answerable question, so I tested it. End to end it did not help (refusal 0.750 vs 1.000), so I kept 0.
- **The agent did worse than plain RAG.** It is about 5 times slower and loses accuracy, so the website uses plain RAG by default.
- With 42 questions, one question is about 2.4 percentage points, so small differences are within noise.

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

## Limitations

- **Wrong section, right article:** for "When did the Haitian Revolution begin?" and "When did Frederick Douglass escape?", retrieval found the right article but not the section with the date, so RAG refused while the no-retrieval model simply remembered.
- **Lenient retrieval metric:** a chunk counts as relevant if it comes from the right article, even if it is the wrong section, so recall looks better than the real experience.
- **Keyword search:** BM25 cannot match different wording; embeddings or query rewriting would help.
- **Strict string scoring:** "began December 1955, lasted 385 days" is correct but was marked wrong because it never says "1956".
- **The agent:** it often searches repeatedly instead of answering, which makes it slower and less accurate than plain RAG.
- **Scope:** 22 Wikipedia articles, mostly U.S. history, so many parts of Black history around the world are not covered.
- **Next steps:** hybrid keyword and embedding search, section-level gold labels for a stricter metric, and more detail questions.
