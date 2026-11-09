"""Help verify the test set: for each answerable question, show where its expected answer appears
in its gold articles. A question whose answer can't be found there needs fixing.

    python -m scripts.check_questions
"""
from app.corpus import load_sections
from eval.metrics import contains, load_questions

sections = load_sections()
for q in load_questions():
    if q["type"] == "unanswerable":
        continue
    gold = [s for s in sections if s.article in q["gold_articles"]]
    missing = [m for m in q["must_include"] if not any(contains(s.text, m) for s in gold)]
    where = [s.id for s in gold if all(contains(s.text, m) for m in q["must_include"])][:3]
    status = "OK     " if not missing and where else "CHECK  "
    print(f"{status}{q['id']} {q['question']}")
    if missing:
        print(f"         not found in gold articles: {missing}")
    elif where:
        print(f"         found in: {', '.join(where)}")
