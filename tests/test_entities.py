"""Entity resolution: CAS, typos, synonyms, partial and ambiguous names."""
from helpers import ask


def test_cas_with_unicode_dashes(loaded):
    # the case study writes CAS numbers with non-breaking hyphens (U+2011)
    body = ask(loaded, "Which products contain CAS 14807‑96‑6?")
    assert body["query_plan"]["filters"]["cas"] == "14807-96-6"
    assert body["query_plan"]["total_matching_rows"] == 2


def test_typo_is_matched_with_warning(loaded):
    body = ask(loaded, "Show products with Titanium dioxid")
    assert body["query_plan"]["filters"]["chemical"] == ["Titanium dioxide"]
    assert any("Matched chemical" in w for w in body["warnings"])
    assert body["query_plan"]["total_matching_rows"] == 55
    assert any("30 of 55" in w for w in body["warnings"])       # LLM saw 30, the count is still exact


def test_synonym_list(loaded):
    body = ask(loaded, "Which products contain cocamide DEA?")
    assert set(body["query_plan"]["filters"]["chemical"]) == {"Cocamide DEA", "Cocamide diethanolamine"}
    assert body["query_plan"]["total_matching_rows"] == 7


def test_same_cas_number_means_same_chemical(loaded):
    body = ask(loaded, "Products with Vitamin A palmitate")
    names = body["query_plan"]["filters"]["chemical"]
    assert "Vitamin A palmitate" in names and any(n.startswith("Retinol/retinyl") for n in names)
    assert any("same CAS number" in a for a in body["query_plan"]["assumptions"])


def test_vitamin_a_family(loaded):
    body = ask(loaded, "Which products contain vitamin A?")
    assert len(body["query_plan"]["filters"]["chemical"]) == 3
    assert body["query_plan"]["total_matching_rows"] == 13


def test_company_partial_name(loaded):
    body = ask(loaded, "Products from Avon")
    assert body["query_plan"]["filters"]["company"] == "New Avon LLC"


def test_ambiguous_company_asks_back(loaded):
    body = ask(loaded, "Products from Beauty")
    assert body["clarification"][0]["field"] == "company"
    assert len(body["clarification"][0]["options"]) > 1
    assert body["evidence"] == [] and body["confidence_label"] == "low"


def test_unknown_entities_return_no_match_not_a_guess(loaded):
    for q in ("Which products contain lead?", "Which products contain CAS 99999-99-9?"):
        body = ask(loaded, q)
        assert body["answer"].startswith("No matching records found")
        assert body["evidence"] == []
