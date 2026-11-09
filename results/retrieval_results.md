# Retrieval results

38 answerable questions, 4 unanswerable. A retrieved chunk counts as relevant if it comes from one of the question's gold articles.

## 1. Retriever / chunking comparison

| config | chunks | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| overlap + section (baseline) | 1537 | 0.711 | 0.921 | 0.974 | 0.818 |
| bm25 + fixed 200 words (naive) | 1629 | 0.763 | 1.000 | 1.000 | 0.877 |
| bm25 + section | 1537 | 0.842 | 0.974 | 1.000 | 0.906 |

## 2. min_score sweep (bm25 + section, top-3)

kept = answerable questions with a gold chunk still in the top-3 after the floor; blocked = unanswerable questions with no chunk above the floor.

| min_score | kept (answerable) | blocked (unanswerable) |
|---|---|---|
| 0 | 0.974 | 0.000 |
| 2 | 0.974 | 0.000 |
| 4 | 0.974 | 0.000 |
| 6 | 0.974 | 0.000 |
| 8 | 0.974 | 0.250 |
| 10 | 0.974 | 0.500 |
| 12 | 0.947 | 0.750 |
| 14 | 0.921 | 1.000 |
| 16 | 0.816 | 1.000 |
