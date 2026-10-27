import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import pytest
from fastapi.testclient import TestClient
from helpers import SAMPLE, FakeEmbedder, fake_llm

from app.core import config, llm
from app.main import app
from app.storage import vector_store


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Empty app: its own SQLite file and Chroma folder, fake LLM and embedder."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    vector_store._state.clear()
    monkeypatch.setattr(vector_store, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(llm, "ask_llm", fake_llm)
    with TestClient(app) as c:
        yield c
    vector_store._state.clear()


@pytest.fixture
def loaded(client):
    """App with the 96-row sample already loaded."""
    with open(SAMPLE, "rb") as f:
        client.post("/ingest", files={"file": ("sample_100.xlsx", f)})
    return client
