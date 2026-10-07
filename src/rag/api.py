"""FastAPI service exposing the RAG pipeline."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .pipeline import RagPipeline

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="evaluated-rag-assistant", version=__version__)


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, examples=["How many hours of sleep do adults need?"])


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    return RagPipeline()


@app.get("/health")
def health() -> dict:
    pipe = get_pipeline()
    return {"status": "ok", "chunks": len(pipe.by_id), "generator": pipe.generator.name, "version": __version__}


@app.post("/ask")
def ask(body: Question) -> dict:
    try:
        return get_pipeline().ask(body.question).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
