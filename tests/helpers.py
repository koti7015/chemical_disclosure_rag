"""Fakes for the LLM and the embedder, so the tests run without internet or an API key."""
import hashlib
import re
from pathlib import Path

import numpy as np

from app.agents.extractor import EXTRACT_PROMPT
from app.agents.planner import PLAN_PROMPT

SAMPLE = Path(__file__).parent.parent / "data" / "sample_100.xlsx"


class FakeEmbedder:
    """Bag-of-words hashing: similar words give similar vectors. Good enough to test the flow."""

    def encode(self, texts, normalize_embeddings=True, batch_size=64):
        out = np.zeros((len(texts), 128))
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 128] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


# What the "LLM" extracts for each question (or sub-question)
EXTRACTIONS = {
    "Which products contain CAS 14807-96-6?": {"cas": "14807-96-6"},
    "Which products were discontinued in 2017?": {"discontinued_year": 2017},
    "Which products are still on sale?": {"discontinued": False},
    "Show products with Titanium dioxid": {"chemical": "Titanium dioxid"},
    "Products from Avon": {"company": "New Avon"},
    "Products from Beauty": {"company": "Beauty"},
    "Which products contain lead?": {"chemical": "lead"},
    "Which products contain CAS 99999-99-9?": {"cas": "99999-99-9"},
    "Any baby products?": {"category": "Baby Products"},
    "Anything with vitamin A?": {},
    "Which products contain vitamin A?": {"chemical": "vitamin A"},
    "Which products contain cocamide DEA?": {"chemical": "cocamide DEA"},
    "Products with Vitamin A palmitate": {"chemical": "Vitamin A palmitate"},
    "What chemicals are in Eye Shadow products?": {"subcategory": "Eye Shadow"},
    "Products first reported between 2013 and 2014": {"reported_from": 2013, "reported_to": 2014},
    "Lip products in makeup": {"category": "Makeup Products (non-permanent)", "topic": "lip color lipstick"},
    "Summarize reporting trends over time": {},
    "Any conflicting or missing data?": {},
    "Products from New Avon LLC": {"company": "New Avon LLC"},
    "Products from Coty": {"company": "Coty"},
}

PLANS = {
    "Compare New Avon LLC and Coty": {
        "intent": "compare",
        "sub_questions": [{"question": "Products from New Avon LLC", "intent": "summarize"},
                          {"question": "Products from Coty", "intent": "summarize"}]},
    "Summarize reporting trends over time": {"intent": "trend"},
    "Any conflicting or missing data?": {"intent": "data_quality"},
}


def fake_llm(system, user):
    if system == PLAN_PROMPT:
        return PLANS.get(user, {"intent": "list", "sub_questions": [{"question": user}]})
    if system == EXTRACT_PROMPT:
        return EXTRACTIONS.get(user, {})
    tags = re.findall(r"\[\d+:\d+\]", user)
    if not tags:
        return {"short_answer": "The rows do not answer this.", "details": ""}
    return {"short_answer": f"Found {len(tags)} rows.", "details": "Examples: " + " ".join(tags[:3])}


def ask(client, question, **extra):
    r = client.post("/ask", json={"question": question, **extra})
    assert r.status_code == 200, r.text
    return r.json()
