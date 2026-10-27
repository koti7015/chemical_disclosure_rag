"""Loading a file into SQL + the vector DB, and small text helpers shared by the agents."""
import hashlib
import io
import re

import pandas as pd

from app.storage import database, vector_store

# The CAS numbers in questions are often typed or pasted with other dash characters (the case study itself uses
# U+2011), so turn all of them into a plain "-" before matching.
DASHES = "‐‑‒–—−"
CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")


def normalize_dashes(text: str) -> str:
    for d in DASHES:
        text = text.replace(d, "-")
    return text


def normalize_cas(text: str) -> str:
    return re.sub(r"\s+", "", normalize_dashes(str(text)))


def find_cas(question: str) -> str | None:
    """CAS number written in the question, if any. Regex is more reliable than the LLM for this."""
    m = CAS_RE.search(normalize_dashes(question))
    return m.group(0) if m else None


def load_file(content: bytes, filename: str) -> tuple[pd.DataFrame, int]:
    """Read an Excel/CSV upload. Returns the clean frame and how many exact duplicates were dropped."""
    buffer = io.BytesIO(content)
    df = pd.read_csv(buffer, low_memory=False) if filename.lower().endswith(".csv") \
        else pd.read_excel(buffer, engine="openpyxl")
    missing = [c for c in database.REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    for c in database.RECORD_COLS:
        if c not in df.columns:                        # optional columns, e.g. ChemicalCreatedAt
            df[c] = None
    df = df[database.RECORD_COLS].copy()

    for c in database.DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in database.INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in database.TEXT_COLS:
        # the source file has broken characters (U+FFFD) at the end of some product names
        df[c] = df[c].astype("string").str.replace("�", "", regex=False).str.strip()
        df[c] = df[c].replace("", pd.NA)
    df["CasNumber"] = df["CasNumber"].map(lambda v: normalize_cas(v) if pd.notna(v) else v)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


def ingest_file(content: bytes, filename: str) -> dict:
    """Load a file once. Used by both the API and the CLI."""
    digest = hashlib.sha256(content).hexdigest()
    already = database.dataset_rows(digest)
    if already is not None:                            # same file was loaded before
        return {"status": "already_loaded", "rows": already, "duplicates_dropped": 0}

    df, dropped = load_file(content, filename)
    row_ids = database.save_records(df, filename, digest)
    try:
        vector_store.add_records(df, row_ids)
    except Exception:
        database.delete_dataset(digest)                # keep SQL and vector DB in step
        raise
    return {"status": "loaded", "rows": len(df), "duplicates_dropped": dropped}
