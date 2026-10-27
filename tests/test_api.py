"""History, records, error handling and the CLI."""
from helpers import ask, fake_llm

from app import cli
from app.agents.extractor import EXTRACT_PROMPT
from app.agents.planner import PLAN_PROMPT
from app.core import llm


def test_history_and_records(loaded):
    first = ask(loaded, "Any baby products?")
    second = ask(loaded, "Which products were discontinued in 2017?")

    history = loaded.get("/history").json()
    assert [h["id"] for h in history] == [second["id"], first["id"]]      # newest first
    detail = loaded.get(f"/history/{first['id']}").json()
    assert detail["query_plan"]["filters"]["category"] == "Baby Products"
    assert detail["confidence"] == first["confidence"]
    assert loaded.get("/history/9999").status_code == 404

    cdph = first["evidence"][0]["CDPHId"]
    rows = loaded.get(f"/records/{cdph}").json()
    assert rows and all(r["CDPHId"] == cdph for r in rows)
    assert loaded.get("/records/1").status_code == 404


def test_unreadable_llm_answer_falls_back(loaded, monkeypatch):
    def half_broken(system, user):
        if system in (PLAN_PROMPT, EXTRACT_PROMPT):
            return fake_llm(system, user)
        raise ValueError("not json")

    monkeypatch.setattr(llm, "ask_llm", half_broken)
    body = ask(loaded, "Any baby products?")
    assert body["short_answer"] == "Found 5 matching records."
    assert any("could not be read" in w for w in body["warnings"])


def test_llm_down_returns_503(loaded, monkeypatch):
    def broken(system, user):
        raise llm.LLMUnavailable("GROQ_API_KEY is not set")

    monkeypatch.setattr(llm, "ask_llm", broken)
    r = loaded.post("/ask", json={"question": "Any baby products?"})
    assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["detail"]


def test_cli(loaded, capsys):
    assert cli.main(["ask", "Which products contain CAS 14807-96-6?"]) == 0
    out = capsys.readouterr().out
    assert "Evidence:" in out and "SQL:" in out and "Confidence:" in out
