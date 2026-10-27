"""Pydantic models for requests and responses."""
from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=10, ge=1, le=50)


class Evidence(BaseModel):
    RowId: int
    CDPHId: int
    CSFId: Optional[int] = None
    ChemicalId: Optional[int] = None
    ProductName: Optional[str] = None
    BrandName: Optional[str] = None
    CompanyName: Optional[str] = None
    PrimaryCategory: Optional[str] = None
    SubCategory: Optional[str] = None
    ChemicalName: Optional[str] = None
    CasNumber: Optional[str] = None
    InitialDateReported: Optional[str] = None
    DiscontinuedDate: Optional[str] = None
    ChemicalDateRemoved: Optional[str] = None


class Clarification(BaseModel):
    field: str
    asked: str
    options: list[str]


class AskResponse(BaseModel):
    id: int                       # id of the saved analysis (see /history/{id})
    short_answer: str
    answer: str                   # short answer + details
    evidence: list[Evidence]
    query_plan: dict              # intent, sub-tasks, route, filters, SQL run, assumptions
    warnings: list[str]
    confidence: float
    confidence_label: str         # high / medium / low
    clarification: Optional[list[Clarification]] = None


class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    created_at: str


class HistoryDetail(HistoryItem):
    short_answer: str = ""
    query_plan: dict
    evidence: list[dict]
    warnings: list[str]
    confidence: Optional[float] = None


class IngestResponse(BaseModel):
    status: str                   # "loaded" or "already_loaded"
    rows: int
    duplicates_dropped: int = 0
