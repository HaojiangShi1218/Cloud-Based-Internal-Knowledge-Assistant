from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from typing import Literal, Optional

from app.config import settings
from app.rag import retrieve
from app.formatter import format_answer
from app.llm import synthesize_answer

app = FastAPI(title=settings.APP_NAME)

NO_HITS = (
    "I couldn’t find anything relevant in the current knowledge base. "
    "Add docs related to this topic and re-run ingestion."
)

@app.on_event("startup")
def on_startup():
    logger.info("Application startup complete")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "environment": settings.ENV
    }

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: Literal["extract", "llm"] = "extract"

@app.post("/ask")
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required")

    hits = retrieve(q, top_k=req.top_k)

    if not hits:
        return {"question": q, "answer": NO_HITS, "citations": []}

    citations = [
        {
            "rank": h["rank"],
            "source": h["source"],
            "page_num": h.get("page_num"),
            "page_end": h.get("page_end"),
            "chunk_index": h["chunk_index"],
            "score": h["score"],
        }
        for h in hits
    ]

    if req.mode == "llm":
        # hard cap to prevent accidental spend (server-side only)
        max_toks = int(getattr(settings, "MAX_ANSWER_TOKENS", 250))
        max_toks = max(32, min(max_toks, 400))  # clamp
        answer = synthesize_answer(q, hits, max_tokens=max_toks)
    else:
        answer = format_answer(hits, question=q)

    return {"question": q, "answer": answer, "citations": citations}
