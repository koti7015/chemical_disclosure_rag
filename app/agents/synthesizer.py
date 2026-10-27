"""Answer synthesizer agent: turns the retrieved rows into a short answer + details, with citations.

The LLM only sees the rows we retrieved. Afterwards every citation is checked against those rows.
"""
import re

from app.core import config, llm

ANSWER_PROMPT = (
    "Answer the question using ONLY the data provided. "
    "Cite every claim about a record with its tag, written exactly like [123:456]. "
    "Numbers taken from a SUMMARY TABLE do not need a tag. "
    "If the data does not answer the question, say so plainly. Do not use outside knowledge. "
    "If there are several parts, answer each one; for comparisons point out the differences. "
    'Return JSON: {"short_answer": "one sentence", '
    '"details": "a few short sentences or lines, with citations"}'
)

TAG_RE = re.compile(r"\[(\d+):(\d+)\]")


def tag(r: dict) -> str:
    return f"[{r['CDPHId']}:{r['RowId']}]"


def row_line(r: dict) -> str:
    return (f"{tag(r)} {r['ProductName']} | brand {r['BrandName']} | {r['CompanyName']} | "
            f"{r['PrimaryCategory']} > {r['SubCategory']} | chemical {r['ChemicalName']} (CAS {r['CasNumber']}) | "
            f"reported {r['InitialDateReported']} | discontinued {r['DiscontinuedDate']} | "
            f"chemical removed {r['ChemicalDateRemoved']}")


def table_lines(table: list[dict]) -> list[str]:
    if not table:
        return []
    cols = list(table[0])
    return [" | ".join(cols)] + [" | ".join(str(t[c]) for c in cols) for t in table]


class AnswerSynthesizerAgent:
    name = "synthesizer"

    def run(self, question: str, parts: list[dict]) -> dict:
        """parts = one dict per sub-question: question, intent, route, rows, total, table, issues, note."""
        warnings = []
        truncated = False
        shown = {}                                   # (CDPHId, RowId) -> row, everything the LLM saw
        blocks = []
        per_part = max(5, config.MAX_ROWS_TO_LLM // max(len(parts), 1))
        for n, p in enumerate(parts, start=1):
            lines = [f"PART {n}: {p['question']}"]
            if p.get("note"):
                lines.append(f"NOTE: {p['note']}")
            if p.get("table"):
                lines.append("SUMMARY TABLE (from SQL):")
                lines += table_lines(p["table"])
            for issue in p.get("issues", [])[:per_part]:
                lines.append(f"ISSUE [{issue['CDPHId']}:{issue['RowId']}] {issue['type']}: {issue['detail']}")
            rows = p.get("rows", [])[:per_part]
            if rows:
                lines.append("ROWS:")
                lines += [row_line(r) for r in rows]
                shown.update({(r["CDPHId"], r["RowId"]): r for r in rows})
            if p.get("total", 0) > len(rows) and not p.get("table"):
                lines.append(f"({p['total']} rows match in total; only {len(rows)} are shown)")
                warnings.append(f"The answer is based on {len(rows)} of {p['total']} matching rows "
                                f"({p['question']})")
                truncated = True
            blocks.append("\n".join(lines))

        prompt = f"Question: {question}\n\n" + "\n\n".join(blocks)
        try:
            reply = llm.ask_llm(ANSWER_PROMPT, prompt)
            short = str(reply.get("short_answer") or "").strip()
            details = str(reply.get("details") or reply.get("answer") or "").strip()
            if not short and not details:
                raise ValueError("empty answer")
        except ValueError:
            warnings.append("The LLM answer could not be read, showing a plain summary of the rows instead")
            short, details = self._fallback(parts)

        answer = f"{short}\n\n{details}".strip() if details and details != short else short
        cited = {(int(a), int(b)) for a, b in TAG_RE.findall(answer)}
        bad = cited - shown.keys()
        if bad:
            warnings.append("The answer cites a record that was not retrieved, treat it with caution")
        needs_tags = any(p.get("rows") and not p.get("table") for p in parts)
        if not cited and needs_tags:
            warnings.append("The answer has no citations")

        evidence = [shown[k] for k in shown if k in cited] or list(shown.values())[:10]
        return {"short_answer": short or details.split("\n")[0], "answer": answer, "evidence": evidence,
                "warnings": warnings, "citations_ok": not bad, "cited_any": bool(cited), "truncated": truncated}

    @staticmethod
    def _fallback(parts: list[dict]) -> tuple[str, str]:
        """No LLM: list what we found, so the user still gets a grounded answer."""
        total = sum(p.get("total", 0) for p in parts)
        lines = []
        for p in parts:
            if p.get("note"):
                lines.append(p["note"])
            lines += table_lines(p.get("table", []))
            for r in p.get("rows", [])[:10]:
                lines.append(f"{tag(r)} {r['ProductName']} ({r['BrandName']}) - {r['ChemicalName']}")
        return f"Found {total} matching records.", "\n".join(lines)


def score_confidence(routes: list[str], fuzzy_used: bool, synth: dict,
                     top_similarity: float | None, conflicts: int) -> tuple[float, str]:
    """Simple rule-based score. Not a probability, just a hint for the reader."""
    base = {"sql": 0.9, "sql_aggregate": 0.9, "hybrid": 0.8, "vector": 0.6}
    score = min(base.get(r, 0.6) for r in routes) if routes else 0.5
    if fuzzy_used:
        score -= 0.1
    if synth["truncated"]:
        score -= 0.05
    if not synth["citations_ok"]:
        score -= 0.3
    if not synth["cited_any"] and "sql_aggregate" not in routes:
        score -= 0.2
    if top_similarity is not None and top_similarity < config.WEAK_SIMILARITY:
        score -= 0.15
    if conflicts:
        score -= 0.05
    score = round(max(0.05, min(score, 1.0)), 2)
    label = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
    return score, label
