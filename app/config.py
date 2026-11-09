"""All configuration comes from environment variables. No secrets live in the code.

The OpenAI client itself reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS_JSONL = ROOT / "data" / "processed" / "sections.jsonl"
ATTRIBUTION_MD = ROOT / "data" / "ATTRIBUTION.md"

# The corpus: English Wikipedia articles (CC BY-SA 4.0), downloaded once by scripts/fetch_articles.py
# and committed as a snapshot so every evaluation number can be reproduced.
ARTICLES = [
    "Atlantic slave trade",
    "Slavery in the United States",
    "Haitian Revolution",
    "Abolitionism in the United States",
    "Underground Railroad",
    "Harriet Tubman",
    "Frederick Douglass",
    "Emancipation Proclamation",
    "Reconstruction era",
    "Jim Crow laws",
    "Great Migration (African American)",
    "Harlem Renaissance",
    "Civil rights movement",
    "Brown v. Board of Education",
    "Montgomery bus boycott",
    "Rosa Parks",
    "Martin Luther King Jr.",
    "March on Washington for Jobs and Freedom",
    "Civil Rights Act of 1964",
    "Voting Rights Act of 1965",
    "Apartheid",
    "Nelson Mandela",
]

MODEL = os.getenv("LLM_MODEL", "gemma-4-E4B-it")
CHUNKING = os.getenv("CHUNKING", "section")
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0"))  # TODO: set from the sweep in results/retrieval_results.md
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "5"))

REFUSAL = "I couldn't find this in the source articles."

# Public deployment: protect the class API key from heavy use.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "6"))   # per visitor IP
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "300"))                      # whole app, per UTC day
