"""Planner agent: what kind of question is this, does it need splitting, and which tool answers it."""
import re

from app.core import config, llm

INTENTS = ("lookup", "list", "compare", "summarize", "trend", "data_quality")

PLAN_PROMPT = (
    "You plan answers to questions about chemicals reported in cosmetic products. "
    "Classify the question into one intent: lookup (one fact, e.g. which chemical is in product X), "
    "list (records matching filters), compare (two or more companies/brands/chemicals side by side), "
    "summarize (overview of a company, brand or category), trend (changes over time / by year), "
    "data_quality (missing, conflicting or suspicious data). "
    "If the question asks about several separate things, split it into sub_questions, one per thing, "
    "each one complete on its own (for compare: one sub-question per item being compared). "
    "Otherwise sub_questions has just the original question. At most 3 sub-questions. "
    'Return JSON: {"intent": "...", "sub_questions": [{"question": "...", "intent": "..."}]}'
)

# Used when the LLM reply can't be read. Order matters: first hit wins.
KEYWORDS = [
    ("data_quality", r"conflict|inconsisten|data quality|missing|suspicious|wrong date"),
    ("trend", r"trend|over time|per year|by year|each year|over the years"),
    ("compare", r"\bcompare|\bvs\.?\b|versus|difference between"),
    ("summarize", r"summari[sz]e|overview|summary"),
    ("list", r"\blist\b|\bshow\b|\bwhich\b|how many|\bany\b"),
]


def guess_intent(question: str) -> str:
    q = question.lower()
    for intent, pattern in KEYWORDS:
        if re.search(pattern, q):
            return intent
    return "lookup"


class PlannerAgent:
    name = "planner"

    def plan(self, question: str) -> tuple[dict, list[str]]:
        """Returns ({"intent", "sub_questions"}, warnings)."""
        warnings = []
        try:
            reply = llm.ask_llm(PLAN_PROMPT, question)
        except ValueError:
            reply = {}
            warnings.append("Could not read the plan from the LLM, used keyword rules instead")

        intent = reply.get("intent")
        if intent not in INTENTS:
            intent = guess_intent(question)

        subs = []
        for item in reply.get("sub_questions") or []:
            if isinstance(item, str):
                item = {"question": item}
            text = str(item.get("question") or "").strip()
            if len(text) >= 3:
                sub_intent = item.get("intent") if item.get("intent") in INTENTS else intent
                subs.append({"question": text, "intent": sub_intent})
        if not subs:
            subs = [{"question": question, "intent": intent}]
        if len(subs) > config.MAX_SUB_QUESTIONS:
            warnings.append(f"Question was split into {len(subs)} parts, only the first "
                            f"{config.MAX_SUB_QUESTIONS} were answered")
            subs = subs[:config.MAX_SUB_QUESTIONS]
        # a compare sub-question is really a list/summary of one side
        for s in subs:
            if len(subs) > 1 and s["intent"] == "compare":
                s["intent"] = "summarize"
        return {"intent": intent, "sub_questions": subs}, warnings

    def choose_route(self, intent: str, filters: dict, topic: str | None) -> str:
        """sql = exact filters only, vector = meaning only, hybrid = filter in SQL then rank by meaning,
        sql_aggregate = group-by counts (trends, data quality)."""
        if intent in ("trend", "data_quality"):
            return "sql_aggregate"
        has_filters = any(v not in (None, [], "") for v in filters.values())
        if has_filters and topic:
            return "hybrid"
        if has_filters:
            return "sql"
        return "vector"
