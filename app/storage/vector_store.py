"""Vector DB (Chroma) + embedding model. Both are created on first use, not at import."""
import chromadb
import numpy as np
import pandas as pd

from app.core import config

_state: dict = {}
BATCH = 500


def get_embedder():
    if "embedder" not in _state:
        from sentence_transformers import SentenceTransformer  # downloads the model on first run
        _state["embedder"] = SentenceTransformer(config.EMBED_MODEL)
    return _state["embedder"]


def get_collection():
    if "collection" not in _state:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        _state["collection"] = client.get_or_create_collection(
            "chem_records", metadata={"hnsw:space": "cosine"})
    return _state["collection"]


def _embed(texts: list[str]) -> list[list[float]]:
    return np.asarray(get_embedder().encode(texts, normalize_embeddings=True, batch_size=64)).tolist()


def _text(value) -> str:
    return "" if pd.isna(value) else str(value)


def row_to_text(r) -> str:
    """The sentence we embed for one row. Dates are left out on purpose: SQL handles dates."""
    return (f"Product: {_text(r.ProductName)}. Brand: {_text(r.BrandName)}. Company: {_text(r.CompanyName)}. "
            f"Category: {_text(r.PrimaryCategory)} > {_text(r.SubCategory)}. "
            f"Chemical: {_text(r.ChemicalName)} (CAS {_text(r.CasNumber)}).")


def add_records(df: pd.DataFrame, row_ids: list[int]) -> None:
    """Embed every row and store it. The vector id is the SQL RowId."""
    df = df.assign(RowId=row_ids)
    col = get_collection()
    for start in range(0, len(df), BATCH):
        rows = list(df.iloc[start:start + BATCH].itertuples())
        texts = [row_to_text(r) for r in rows]
        col.upsert(                                  # upsert: running twice never duplicates
            ids=[str(r.RowId) for r in rows],
            documents=texts,
            embeddings=_embed(texts),
            metadatas=[{"cdph_id": int(r.CDPHId)} for r in rows],
        )


def search(question: str, k: int = 10) -> list[tuple[int, float]]:
    """(RowId, similarity) of the k closest rows, best first."""
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=_embed([question]), n_results=min(k, col.count()))
    # cosine distance -> similarity
    return [(int(i), 1 - float(d)) for i, d in zip(res["ids"][0], res["distances"][0])]


def rerank(question: str, row_ids: list[int], k: int = 10) -> list[tuple[int, float]]:
    """Order rows that SQL already picked by how close they are to the question."""
    if not row_ids:
        return []
    got = get_collection().get(ids=[str(i) for i in row_ids], include=["embeddings"])
    if len(got["ids"]) == 0:
        return []
    vectors = np.asarray(got["embeddings"])
    query = np.asarray(_embed([question])[0])
    scores = vectors @ query                         # both are normalized, so this is cosine
    best = np.argsort(-scores)[:k]
    return [(int(got["ids"][i]), float(scores[i])) for i in best]


def count() -> int:
    return get_collection().count()
