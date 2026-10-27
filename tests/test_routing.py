"""Which route each kind of question takes, and what the SQL / vector agents return."""
import re

from helpers import ask

from app.agents.planner import guess_intent


def test_cas_lookup_uses_exact_sql_filter(loaded):
    body = ask(loaded, "Which products contain CAS 14807-96-6?")
    plan = body["query_plan"]
    assert plan["route"] == "sql" and plan["filters"]["cas"] == "14807-96-6"
    assert plan["total_matching_rows"] == 2
    assert all(e["ChemicalName"] == "Talc" for e in body["evidence"])
    assert re.search(r"\[\d+:\d+\]", body["answer"])
    sql = [q["sql"] for s in plan["steps"] for q in s.get("queries", [])]
    assert any("CasNumber = ?" in q for q in sql)                 # the real SQL is in the plan


def test_year_filter(loaded):
    body = ask(loaded, "Which products were discontinued in 2017?")
    assert body["query_plan"]["total_matching_rows"] == 6
    assert body["query_plan"]["rows_returned"] == 6
    assert all(e["DiscontinuedDate"].startswith("2017") for e in body["evidence"])


def test_active_only_filter_is_kept(loaded):
    # first version turned discontinued=False into None and returned everything
    body = ask(loaded, "Which products are still on sale?")
    assert body["query_plan"]["filters"]["discontinued"] is False
    assert body["query_plan"]["total_matching_rows"] == 96 - 28


def test_subcategory_and_date_range(loaded):
    assert ask(loaded, "What chemicals are in Eye Shadow products?")["query_plan"]["total_matching_rows"] == 8
    body = ask(loaded, "Products first reported between 2013 and 2014")
    assert body["query_plan"]["total_matching_rows"] == 19
    assert any("InitialDateReported" in a for a in body["query_plan"]["assumptions"])


def test_category_filter(loaded):
    body = ask(loaded, "Any baby products?")
    assert body["query_plan"]["filters"]["category"] == "Baby Products"
    assert body["query_plan"]["total_matching_rows"] == 5


def test_semantic_only_question(loaded):
    body = ask(loaded, "Anything with vitamin A?")
    assert body["query_plan"]["route"] == "vector"
    assert len(body["evidence"]) > 0


def test_hybrid_filters_then_ranks_by_meaning(loaded):
    body = ask(loaded, "Lip products in makeup", top_k=5)
    plan = body["query_plan"]
    assert plan["route"] == "hybrid"
    assert plan["total_matching_rows"] == 28 and plan["rows_returned"] == 5
    assert all(e["PrimaryCategory"] == "Makeup Products (non-permanent)" for e in body["evidence"])


def test_trend_uses_group_by(loaded):
    body = ask(loaded, "Summarize reporting trends over time")
    plan = body["query_plan"]
    assert plan["route"] == "sql_aggregate"
    table = next(s["table"] for s in plan["steps"] if s.get("action") == "trend")
    assert sum(t["chemical_reports"] for t in table) == 96
    by_year = {t["year"]: t for t in table}
    assert by_year["2009"]["chemical_reports"] == 32


def test_data_quality_finds_conflicts(loaded):
    body = ask(loaded, "Any conflicting or missing data?")
    table = next(s["table"] for s in body["query_plan"]["steps"] if s.get("action") == "data_quality")
    found = {t["issue"]: t["rows"] for t in table}
    assert found["date_conflict"] == 9                  # discontinued before first reported
    assert found["trade_secret"] == 5
    assert found["cas_many_names"] == 1                 # 79-81-2


def test_compare_is_split_into_sub_tasks(loaded):
    body = ask(loaded, "Compare New Avon LLC and Coty")
    tasks = body["query_plan"]["sub_tasks"]
    assert [t["filters"]["company"] for t in tasks] == ["New Avon LLC", "Coty"]
    assert all(t["route"] == "sql" for t in tasks)


def test_planner_keyword_fallback():
    assert guess_intent("Show the trend by year for Coty") == "trend"
    assert guess_intent("Compare Avon vs Coty") == "compare"
    assert guess_intent("Which products contain talc?") == "list"
