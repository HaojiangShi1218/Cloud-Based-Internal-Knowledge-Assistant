from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, root_validator
from loguru import logger
from typing import Literal, Optional
import json
import time
from hashlib import sha256

from app.config import settings
from app.rag import retrieve
from app.formatter import format_answer
from app.cache import clear_cache
from app.llm import select_llm_hits, synthesize_answer

app = FastAPI(
    title=settings.APP_NAME,
    root_path="/api",
)

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

@app.post("/debug/cache/clear")
def debug_clear_cache():
    cleared = clear_cache()
    return {"cleared": cleared}

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(5, alias="top-k")
    mode: Literal["extract", "llm"] = "extract"
    query_rewrite_enabled: Optional[bool] = Field(None, alias="query-rewrite-enabled")

    @root_validator(pre=True)
    def _normalize_hyphen_keys(cls, values):
        if not isinstance(values, dict):
            return values
        mappings = {
            "top-k": "top_k",
            "query-rewrite-enabled": "query_rewrite_enabled",
        }
        for src, dst in mappings.items():
            if src in values and dst not in values:
                values[dst] = values[src]
        return values

    class Config:
        allow_population_by_field_name = True

@app.post("/ask")
def ask(req: AskRequest, request: Request):
    t0 = time.perf_counter()
    ok = False
    hit_count = 0

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        hits = retrieve(
            q,
            top_k=req.top_k,
            expand_lists=True,
            query_rewrite_enabled=req.query_rewrite_enabled,
        )
        hit_count = len(hits or [])

        if not hits:
            ok = True
            return {"question": q, "answer": NO_HITS, "citations": []}

        if req.mode == "llm":
            # hard cap to prevent accidental spend (server-side only)
            max_toks = int(getattr(settings, "MAX_ANSWER_TOKENS", 250))
            max_toks = max(32, min(max_toks, 400))  # clamp
            llm_hits = select_llm_hits(hits, max_hits=req.top_k)
            answer, used_hits = synthesize_answer(
                q,
                llm_hits,
                max_tokens=max_toks,
            )
            citations_src = used_hits
        else:
            llm_hits = select_llm_hits(hits, max_hits=req.top_k)
            answer = format_answer(llm_hits, question=q)
            citations_src = llm_hits

        citations = [
            {
                "rank": h["rank"],
                "source": h["source"],
                "page_num": h.get("page_num"),
                "page_end": h.get("page_end"),
                "chunk_index": h["chunk_index"],
                "score": h["score"],
            }
            for h in (citations_src or [])
        ]

        ok = True
        return {"question": q, "answer": answer, "citations": citations}
    finally:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_obj = {
            "event": "ask_request",
            "ok": ok,
            "latency_ms": latency_ms,
            "mode": req.mode,
            "top_k": req.top_k,
            "query_rewrite_enabled": req.query_rewrite_enabled,
            "hit_count": hit_count,
            "client_ip": request.client.host if request.client else None,
            # privacy-safe question logging
            "question_len": len(q),
            "question_sha256": sha256(q.encode("utf-8")).hexdigest(),
            "question_preview": q[:80],
        }
        logger.info(json.dumps(log_obj, ensure_ascii=False))
