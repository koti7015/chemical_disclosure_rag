# Chemical Disclosure RAG

A small multi-agent assistant over the California "Chemicals in Cosmetics" data. You load the Excel/CSV once, ask a
question in plain English, and get an answer with row-level citations, the query plan (with the SQL that ran) and a
confidence score.

Stack: FastAPI, SQLite, Chroma, Groq (LLM), `bge-small-en-v1.5` embeddings (run locally).

## Setup

I used a conda env and installed the packages with uv.

```bash
conda create -n agentic-rag python=3.11 -y
conda activate agentic-rag
pip install uv
uv pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq key:

```
GROQ_API_KEY=your_key
GROQ_MODEL=openai/gpt-oss-20b
```

If the embedding model download fails with an SSL / "No such file or directory" error, also add your certifi path
(get it with `python -c "import certifi; print(certifi.where())"`):

```
CA_BUNDLE=C:\Users\<you>\anaconda3\envs\agentic-rag\Lib\site-packages\certifi\cacert.pem
```

## Run

From the project folder:

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs and call `POST /ingest` with `data/sample_100.xlsx`. The first ingest downloads the
embedding model (~130 MB). Data is saved in `app.db` and `chroma_db/`, so you don't need to ingest again after a
restart.

There is also a CLI:

```bash
python -m app.cli ingest data/sample_100.xlsx
python -m app.cli ask "Which products contain CAS 14807-96-6?"
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| POST | /ingest | load an .xlsx or .csv (once) |
| POST | /ask | `{"question": "...", "top_k": 10}` → answer, evidence, query plan, warnings, confidence |
| GET | /history | past questions, newest first |
| GET | /history/{id} | one past answer in full |
| GET | /records/{cdph_id} | all rows of a product, to check a citation |
| GET | /health | row counts in SQL and Chroma |

## How it works

```
question
  -> planner      intent (lookup, list, compare, summarize, trend, data_quality), splits multi-part questions
  -> extractor    chemical, CAS, company, brand, category, dates... matched to real values in the data
  -> route        sql (filters) / vector (meaning) / hybrid (both) / sql_aggregate (trends, data quality)
  -> synthesizer  answer from the retrieved rows only, citations checked
```

The agents are in `app/agents/`, and `app/orchestrator.py` runs them. Citations look like `[CDPHId:RowId]`. RowId is
added at load time because no column in the file is unique per row.

A few things it handles:

- CAS numbers are matched exactly, even with odd dash characters
- typos, partial names and a few synonyms (e.g. "Cocamide DEA"); names with the same CAS are treated as one chemical
- if a company name matches more than one company, it asks back instead of guessing
- if a chemical/company isn't in the data, it says so ("No matching records found")
- data quality: dates that conflict, missing CAS numbers, one CAS with different names

## Testing

Unit tests (fake LLM, no key or internet needed):

```bash
pytest
```

API tests: import `postman/chemical-disclosure-rag.postman_collection.json` into Postman. It has every request
with the expected result in its description. Or use Swagger at `/docs`. Ingest the sample first. Numbers below are
for the 96-row sample.

| Question | Expected |
|---|---|
| Which products contain CAS 14807-96-6? | sql, 2 rows (Talc) |
| Show products discontinued in 2017 | 6 rows |
| Which products are still on sale and not discontinued? | 68 rows |
| What chemicals are reported in Eye Shadow products? | 8 rows |
| Products first reported between 2013 and 2014 | 19 rows |
| Show products with Titanium dioxid | typo fixed, 55 rows, warning |
| Which products contain cocamide DEA? | finds both names, 7 rows |
| What chemicals does Avon report? | matched to New Avon LLC |
| What chemicals does Beauty report? | asks which company |
| Which products contain lead? | No matching records found |
| Which makeup products are for lips? | hybrid, lip products first |
| Compare New Avon LLC and Coty | two sub-tasks |
| Summarize reporting trends over time | counts per year |
| Are there any conflicting or missing data in the records? | 9 date conflicts, 5 trade secrets |

The LLM decides the filters, so the route can change now and then. Check `query_plan` to see what it did.

## Project layout

```
app/
  main.py, cli.py, orchestrator.py
  api/        routes.py, schemas.py
  agents/     planner, extractor, sql_agent, retriever, synthesizer
  core/       config.py, llm.py
  storage/    database.py, vector_store.py
  ingestion/  loader.py
data/         sample_100.xlsx
tests/        pytest tests
postman/      API test collection
scripts/      make_sample.py (builds the sample from the full CSV)
```

## Known limits

- One vector per row. Fine for the sample; for the full 114k rows I'd index one document per product.
- The LLM sees at most 30 rows per answer. Counts always come from SQL, so they are exact.
- Synonyms are partly a hand-written list. PubChem would be a better source.
- The confidence score is simple rules, not a real probability.
- No auth, SQLite only.
