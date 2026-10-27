"""FastAPI app. Run with:  uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.storage import database


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(title="Chemical Disclosure RAG", lifespan=lifespan)
app.include_router(router)
