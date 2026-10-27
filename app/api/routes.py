"""HTTP endpoints. The real work is done by the orchestrator and the agents."""
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import AskRequest, AskResponse, HistoryDetail, HistoryItem, IngestResponse
from app.core.llm import LLMUnavailable
from app.ingestion import loader
from app.orchestrator import Orchestrator
from app.storage import database, vector_store

router = APIRouter()
orchestrator = Orchestrator()


@router.get("/health")
def health():
    return {"sql_records": database.count_records(), "vector_records": vector_store.count()}


@router.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    """Load an Excel (.xlsx) or .csv file once into SQL and the vector DB."""
    if not file.filename.lower().endswith((".xlsx", ".csv")):
        raise HTTPException(400, "Upload an .xlsx or .csv file")
    try:
        return loader.ingest_file(file.file.read(), file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if database.count_records() == 0:
        raise HTTPException(409, "No data loaded yet. Upload a file with POST /ingest first")
    try:
        result = orchestrator.run(req.question, req.top_k)
    except LLMUnavailable as e:
        raise HTTPException(503, str(e))
    analysis_id = database.save_analysis(req.question, result)
    return {"id": analysis_id, **result}


@router.get("/history", response_model=list[HistoryItem])
def history():
    """Every question answered so far, newest first."""
    return database.list_history()


@router.get("/history/{analysis_id}", response_model=HistoryDetail)
def history_detail(analysis_id: int):
    item = database.get_analysis(analysis_id)
    if item is None:
        raise HTTPException(404, "Analysis not found")
    return item


@router.get("/records/{cdph_id}")
def records(cdph_id: int):
    """All rows for one product, so a citation can be checked."""
    rows = database.records_for_product(cdph_id)
    if not rows:
        raise HTTPException(404, "Product not found")
    return rows
