import pandas as pd
from helpers import SAMPLE


def test_ask_before_ingest_is_rejected(client):
    assert client.post("/ask", json={"question": "Any baby products?"}).status_code == 409


def test_ingest_loads_once(client):
    with open(SAMPLE, "rb") as f:
        first = client.post("/ingest", files={"file": ("sample_100.xlsx", f)}).json()
    assert first["status"] == "loaded" and first["rows"] == 96
    with open(SAMPLE, "rb") as f:
        again = client.post("/ingest", files={"file": ("sample_100.xlsx", f)}).json()
    assert again == {"status": "already_loaded", "rows": 96, "duplicates_dropped": 0}
    assert client.get("/health").json() == {"sql_records": 96, "vector_records": 96}


def test_ingest_rejects_wrong_file_type_and_missing_columns(client):
    assert client.post("/ingest", files={"file": ("a.txt", b"hi")}).status_code == 400
    r = client.post("/ingest", files={"file": ("bad.csv", b"a,b\n1,2\n")})
    assert r.status_code == 400 and "Missing columns" in r.json()["detail"]


def test_ingest_drops_exact_duplicates(client):
    df = pd.read_excel(SAMPLE).head(5)
    csv = pd.concat([df, df.head(1)]).to_csv(index=False).encode()
    r = client.post("/ingest", files={"file": ("dup.csv", csv)}).json()
    assert r["rows"] == 5 and r["duplicates_dropped"] == 1
