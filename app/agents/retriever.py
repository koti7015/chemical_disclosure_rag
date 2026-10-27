"""Semantic retrieval agent: finds rows by meaning (partial names, synonyms, descriptive questions)."""
from app.core import config
from app.storage import database, vector_store


class SemanticRetrievalAgent:
    name = "vector"

    def search(self, text: str, k: int) -> tuple[list[dict], list[float]]:
        """Closest rows in the whole dataset."""
        hits = vector_store.search(text, k)
        return self._rows(hits)

    def rerank(self, text: str, row_ids: list[int], k: int) -> tuple[list[dict], list[float]]:
        """Hybrid: SQL already picked the candidate rows, order them by meaning and keep the best k."""
        hits = vector_store.rerank(text, row_ids[:config.MAX_RERANK], k)
        return self._rows(hits)

    @staticmethod
    def _rows(hits: list[tuple[int, float]]) -> tuple[list[dict], list[float]]:
        rows = database.get_records([row_id for row_id, _ in hits])
        score = dict(hits)
        return rows, [round(score[r["RowId"]], 3) for r in rows]
