"""Orchestrator: runs the agents in order and keeps a trace of what each one did.

    planner      -> intent + sub-questions
    extractor    -> filters for each sub-question, matched to real values
    planner      -> route: sql / vector / hybrid / sql_aggregate
    sql / vector -> rows (with RowIds for citations)
    synthesizer  -> short answer + details, citations checked
"""
from app.agents import (AnswerSynthesizerAgent, ExtractionAgent, PlannerAgent, SemanticRetrievalAgent,
                        StructuredQueryAgent)
from app.agents.sql_agent import date_issues
from app.agents.synthesizer import score_confidence
from app.core import config


class Orchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.extractor = ExtractionAgent()
        self.sql = StructuredQueryAgent()
        self.vector = SemanticRetrievalAgent()
        self.synthesizer = AnswerSynthesizerAgent()

    def run(self, question: str, top_k: int = 10) -> dict:
        steps, warnings, assumptions = [], [], []

        plan, w = self.planner.plan(question)
        warnings += w
        steps.append({"agent": "planner", "action": "classify and split",
                      "intent": plan["intent"], "sub_questions": plan["sub_questions"]})

        parts, sub_tasks = [], []
        fuzzy_used = False
        for sub in plan["sub_questions"]:
            ext = self.extractor.run(sub["question"])
            warnings += ext.warnings
            assumptions += [a for a in ext.assumptions if a not in assumptions]
            fuzzy_used = fuzzy_used or ext.fuzzy_used
            steps.append({"agent": "extractor", "question": sub["question"], "llm_output": ext.raw,
                          "filters": ext.filters, "topic": ext.topic})

            if ext.clarify:                              # ambiguous name: ask, don't pick one at random
                return self._clarify(question, plan, ext, steps, warnings, assumptions)

            task = {"question": sub["question"], "intent": sub["intent"], "filters": ext.filters,
                    "topic": ext.topic}
            if ext.not_found:
                note = "No matching records found: " + ", ".join(ext.not_found) + " is not in the data."
                parts.append({"question": sub["question"], "note": note, "rows": [], "total": 0})
                sub_tasks.append({**task, "route": "none", "not_found": ext.not_found})
                continue

            route = self.planner.choose_route(sub["intent"], ext.filters, ext.topic)
            part = self._execute(sub, route, ext, top_k, steps)
            parts.append(part)
            sub_tasks.append({**task, "route": route, "rows_returned": len(part["rows"]),
                              "total_matching_rows": part["total"]})

        query_plan = {"intent": plan["intent"], "sub_tasks": sub_tasks, "steps": steps,
                      "assumptions": assumptions}
        # keep the single-question shape easy to read
        if len(sub_tasks) == 1:
            query_plan.update({k: sub_tasks[0].get(k) for k in
                               ("route", "filters", "rows_returned", "total_matching_rows")})

        if all(not p["rows"] and not p.get("table") for p in parts):
            notes = [p["note"] for p in parts if p.get("note")]
            answer = " ".join(notes) or "No matching records found."
            return {"short_answer": answer, "answer": answer, "evidence": [], "query_plan": query_plan,
                    "warnings": warnings, "confidence": 0.9 if notes else 0.5,
                    "confidence_label": "high" if notes else "medium", "clarification": None}

        # conflicting dates in the rows we are about to use
        conflicts = [i for p in parts if p.get("route") != "sql_aggregate"
                     for r in p["rows"] for i in date_issues(r)]
        for i in conflicts[:3]:
            warnings.append(f"Conflicting dates in row [{i['CDPHId']}:{i['RowId']}]: {i['detail']}")

        synth = self.synthesizer.run(question, parts)
        warnings += synth["warnings"]
        steps.append({"agent": "synthesizer", "cited": len(synth["evidence"]) if synth["cited_any"] else 0,
                      "citations_ok": synth["citations_ok"]})

        sims = [s for p in parts for s in p.get("similarity", [])]
        score, label = score_confidence([t["route"] for t in sub_tasks if t["route"] != "none"], fuzzy_used,
                                        synth, max(sims) if sims else None, len(conflicts))
        if sims and max(sims) < config.WEAK_SIMILARITY:
            warnings.append("The closest matches by meaning are weak, the answer may not be relevant")
        return {"short_answer": synth["short_answer"], "answer": synth["answer"], "evidence": synth["evidence"],
                "query_plan": query_plan, "warnings": warnings, "confidence": score,
                "confidence_label": label, "clarification": None}

    def _execute(self, sub: dict, route: str, ext, top_k: int, steps: list) -> dict:
        part = {"question": sub["question"], "route": route, "rows": [], "total": 0}
        if route == "sql_aggregate":
            res = self.sql.data_quality(ext.filters) if sub["intent"] == "data_quality" \
                else self.sql.trend(ext.filters)
            part.update(rows=res.rows, total=res.total, table=res.table, issues=res.issues)
            steps.append({"agent": "sql", "action": sub["intent"], "queries": res.queries,
                          "table": res.table})
        elif route == "sql":
            res = self.sql.fetch(ext.filters)
            part.update(rows=res.rows, total=res.total)
            steps.append({"agent": "sql", "action": "filter", "queries": res.queries, "total": res.total})
        elif route == "hybrid":
            res = self.sql.fetch(ext.filters)
            rows, sims = self.vector.rerank(ext.topic, [r["RowId"] for r in res.rows], top_k)
            part.update(rows=rows, total=res.total, similarity=sims)
            steps.append({"agent": "sql", "action": "filter candidates", "queries": res.queries,
                          "total": res.total})
            steps.append({"agent": "vector", "action": "rerank by meaning", "text": ext.topic,
                          "kept": len(rows), "top_similarity": sims[:3]})
        else:
            text = ext.topic or sub["question"]
            rows, sims = self.vector.search(text, top_k)
            part.update(rows=rows, total=len(rows), similarity=sims)
            steps.append({"agent": "vector", "action": "search", "text": text, "top_k": top_k,
                          "top_similarity": sims[:3]})
        return part

    @staticmethod
    def _clarify(question, plan, ext, steps, warnings, assumptions) -> dict:
        lines = [f"'{c['asked']}' matches more than one {c['field']}: {', '.join(c['options'])}"
                 for c in ext.clarify]
        answer = " ".join(lines) + ". Which one did you mean? Ask again with the full name."
        query_plan = {"intent": plan["intent"], "sub_tasks": [], "steps": steps, "assumptions": assumptions,
                      "route": "clarify"}
        return {"short_answer": "Your question matches more than one value, please pick one.", "answer": answer,
                "evidence": [], "query_plan": query_plan, "warnings": warnings, "confidence": 0.0,
                "confidence_label": "low", "clarification": ext.clarify}
