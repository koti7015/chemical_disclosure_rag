"""SQLite storage: the records (source of truth), loaded datasets, and analysis history.

The query building for questions lives in app/agents/sql_agent.py. This file only knows about tables.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from app.core import config

INT_COLS = ["CDPHId", "CSFId", "CompanyId", "PrimaryCategoryId", "SubCategoryId", "CasId",
            "ChemicalId", "ChemicalCount"]
DATE_COLS = ["InitialDateReported", "MostRecentDateReported", "DiscontinuedDate",
             "ChemicalCreatedAt", "ChemicalUpdatedAt", "ChemicalDateRemoved"]
TEXT_COLS = ["CasNumber", "ChemicalName", "ProductName", "CSF", "CompanyName", "BrandName",
             "PrimaryCategory", "SubCategory"]
RECORD_COLS = INT_COLS + TEXT_COLS + DATE_COLS

# The file must have these. The rest are filled with NULL if missing.
REQUIRED_COLS = ["CDPHId", "CSFId", "ChemicalId", "ChemicalCount", "CasNumber", "ChemicalName",
                 "ProductName", "CompanyName", "BrandName", "PrimaryCategory", "SubCategory",
                 "InitialDateReported", "MostRecentDateReported", "DiscontinuedDate", "ChemicalDateRemoved"]


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    col_defs = ", ".join(
        f"{c} INTEGER" if c in INT_COLS else f"{c} TEXT" for c in RECORD_COLS)
    with connect() as c:
        c.executescript(f"""
        CREATE TABLE IF NOT EXISTS datasets (
            DatasetId INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT, sha256 TEXT UNIQUE, row_count INTEGER, loaded_at TEXT);
        -- RowId is our own key: the file has no column that is unique per row
        CREATE TABLE IF NOT EXISTS records (
            RowId INTEGER PRIMARY KEY AUTOINCREMENT, DatasetId INTEGER, {col_defs});
        CREATE INDEX IF NOT EXISTS idx_records_cdph ON records (CDPHId);
        CREATE INDEX IF NOT EXISTS idx_records_cas ON records (CasNumber);
        CREATE INDEX IF NOT EXISTS idx_records_chem ON records (ChemicalName);
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, short_answer TEXT, answer TEXT,
            query_plan TEXT, evidence TEXT, warnings TEXT, confidence REAL, created_at TEXT);
        """)
        # app.db files made by the first version don't have these two columns
        existing = {r["name"] for r in c.execute("PRAGMA table_info(analyses)")}
        for col, kind in (("short_answer", "TEXT"), ("confidence", "REAL")):
            if col not in existing:
                c.execute(f"ALTER TABLE analyses ADD COLUMN {col} {kind}")


def _to_python(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):          # numpy scalar -> python
        return value.item()
    return value


# ---------- datasets / records ----------

def dataset_rows(sha256: str):
    """Row count of an already loaded dataset, or None if this file was never loaded."""
    with connect() as c:
        r = c.execute("SELECT row_count FROM datasets WHERE sha256=?", (sha256,)).fetchone()
    return r["row_count"] if r else None


def save_records(df: pd.DataFrame, filename: str, sha256: str) -> list[int]:
    """Insert the rows and return their new RowIds, in the same order as df."""
    now = datetime.now(timezone.utc).isoformat()
    with connect() as c:
        cur = c.execute(
            "INSERT INTO datasets (filename, sha256, row_count, loaded_at) VALUES (?,?,?,?)",
            (filename, sha256, len(df), now))
        dataset_id = cur.lastrowid
        rows = [(dataset_id, *[_to_python(v) for v in r])
                for r in df[RECORD_COLS].itertuples(index=False, name=None)]
        marks = ",".join("?" * (len(RECORD_COLS) + 1))
        c.executemany(f"INSERT INTO records (DatasetId, {', '.join(RECORD_COLS)}) VALUES ({marks})", rows)
        ids = c.execute("SELECT RowId FROM records WHERE DatasetId=? ORDER BY RowId", (dataset_id,))
        return [r["RowId"] for r in ids]


def delete_dataset(sha256: str) -> None:
    """Undo a load (used if the vector DB step fails after the SQL step worked)."""
    with connect() as c:
        row = c.execute("SELECT DatasetId FROM datasets WHERE sha256=?", (sha256,)).fetchone()
        if row:
            c.execute("DELETE FROM records WHERE DatasetId=?", (row["DatasetId"],))
            c.execute("DELETE FROM datasets WHERE DatasetId=?", (row["DatasetId"],))


def count_records() -> int:
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM records").fetchone()[0]


def get_records(row_ids: list[int]) -> list[dict]:
    """Full rows for the given RowIds, keeping the order we were given (search ranking)."""
    if not row_ids:
        return []
    found = {}
    with connect() as c:
        for start in range(0, len(row_ids), 900):        # SQLite limit on '?' per query
            part = row_ids[start:start + 900]
            marks = ",".join("?" * len(part))
            for r in c.execute(f"SELECT * FROM records WHERE RowId IN ({marks})", part):
                found[r["RowId"]] = dict(r)
    return [found[i] for i in row_ids if i in found]


def records_for_product(cdph_id: int) -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM records WHERE CDPHId=? ORDER BY RowId", (cdph_id,))]


def distinct_values(column: str) -> list[str]:
    assert column in TEXT_COLS          # column names can't be query parameters, so whitelist
    with connect() as c:
        return [r[0] for r in c.execute(
            f"SELECT DISTINCT {column} FROM records WHERE {column} IS NOT NULL")]


def chemicals_for_cas(cas: str) -> list[str]:
    with connect() as c:
        return [r[0] for r in c.execute(
            "SELECT DISTINCT ChemicalName FROM records WHERE CasNumber=? ORDER BY ChemicalName", (cas,))]


def cas_for_chemicals(names: list[str]) -> list[str]:
    """Real CAS numbers used by these chemical names ('0' is how the file marks trade secrets)."""
    if not names:
        return []
    marks = ",".join("?" * len(names))
    with connect() as c:
        return [r[0] for r in c.execute(
            f"SELECT DISTINCT CasNumber FROM records WHERE ChemicalName IN ({marks}) "
            "AND CasNumber IS NOT NULL AND CasNumber != '0'", names)]


# ---------- analysis history ----------

def save_analysis(question: str, result: dict) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO analyses (question, short_answer, answer, query_plan, evidence, warnings, "
            "confidence, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (question, result["short_answer"], result["answer"], json.dumps(result["query_plan"]),
             json.dumps(result["evidence"]), json.dumps(result["warnings"]), result["confidence"],
             datetime.now(timezone.utc).isoformat()))
        return cur.lastrowid


def list_history() -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, question, answer, created_at FROM analyses ORDER BY id DESC")]


def get_analysis(analysis_id: int):
    with connect() as c:
        r = c.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    if not r:
        return None
    out = dict(r)
    for key in ("query_plan", "evidence", "warnings"):
        out[key] = json.loads(out[key]) if out[key] else ([] if key != "query_plan" else {})
    out["short_answer"] = out["short_answer"] or ""
    return out
