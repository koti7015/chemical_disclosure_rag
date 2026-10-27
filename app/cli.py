"""Command line version of the API.

    python -m app.cli ingest data/sample_100.xlsx
    python -m app.cli ask "Which products contain CAS 14807-96-6?"
    python -m app.cli ask "Compare New Avon LLC and Coty" --json
"""
import argparse
import json
import sys
from pathlib import Path

from app.core.llm import LLMUnavailable
from app.ingestion import loader
from app.orchestrator import Orchestrator
from app.storage import database


def print_result(result: dict) -> None:
    print(result["answer"])
    print(f"\nConfidence: {result['confidence']} ({result['confidence_label']})")
    if result["evidence"]:
        print("\nEvidence:")
        for e in result["evidence"]:
            print(f"  [{e['CDPHId']}:{e['RowId']}] {e['ProductName']} | {e['CompanyName']} | {e['ChemicalName']}"
                  f" | CSFId {e['CSFId']} | ChemicalId {e['ChemicalId']}")
    plan = result["query_plan"]
    print("\nHow it was answered:")
    print(f"  intent: {plan['intent']}")
    for t in plan.get("sub_tasks", []):
        print(f"  - {t['question']}  ->  route {t['route']}, filters {t.get('filters')}")
    for step in plan["steps"]:
        for q in step.get("queries", []):
            print(f"    SQL: {q['sql']}  {q['params']}")
    for a in plan.get("assumptions", []):
        print(f"  assumption: {a}")
    for w in result["warnings"]:
        print(f"  warning: {w}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chemical disclosure Q&A")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="load an .xlsx or .csv file")
    p_ingest.add_argument("path")
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=10)
    p_ask.add_argument("--json", action="store_true", help="print the raw result as JSON")
    args = parser.parse_args(argv)

    database.init_db()
    if args.command == "ingest":
        path = Path(args.path)
        try:
            print(loader.ingest_file(path.read_bytes(), path.name))
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0

    if database.count_records() == 0:
        print("No data loaded yet. Run: python -m app.cli ingest <file>", file=sys.stderr)
        return 1
    try:
        result = Orchestrator().run(args.question, args.top_k)
    except LLMUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    result["id"] = database.save_analysis(args.question, result)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
