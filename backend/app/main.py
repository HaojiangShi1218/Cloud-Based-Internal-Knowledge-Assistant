from fastapi import FastAPI, HTTPException, Request, Header, Depends, BackgroundTasks, UploadFile, File, Query
from pydantic import BaseModel, Field, root_validator
from loguru import logger
from typing import Literal, Optional, List
import json
import time
import secrets
from hashlib import sha256

from app.config import settings
from app.admin_auth import require_admin_token
from app.admin_jobs import create_job_for_documents, delete_document_assets, run_ingestion_job
from app.admin_store import (
    delete_document,
    get_document,
    init_admin_store,
    get_documents,
    get_ingestion_job,
    get_job_document_ids,
    get_running_job,
    list_documents,
)
from app.admin_uploads import validate_upload_files, save_upload_files
from app.rag import retrieve
from app.formatter import format_answer, format_answer_with_evidence
from app.cache import clear_cache
from app.embeddings import get_model
from app.llm import select_llm_hits, synthesize_answer
from app.utils.timing import span
import logging

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
    init_admin_store()
    # Warm local embedding model at startup to avoid first-request latency spikes.
    get_model()
    logger.info("Application startup complete")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "environment": settings.ENV
    }

@app.post(
    "/debug/cache/clear",
    responses={403: {"description": "Forbidden (missing/invalid x-debug-token)"}},
)
def debug_clear_cache(
    x_debug_token: Optional[str] = Header(None, alias="x-debug-token"),
):
    token = (settings.DEBUG_CACHE_CLEAR_TOKEN or "").strip()
    if not token:
        raise HTTPException(status_code=403, detail="Cache clear endpoint is disabled")

    provided = (x_debug_token or "").strip()
    if not secrets.compare_digest(provided, token):
        raise HTTPException(status_code=403, detail="Forbidden")

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


class CreateIngestionJobRequest(BaseModel):
    document_ids: List[int]


@app.post("/admin/uploads/validate")
async def admin_validate_uploads(
    files: List[UploadFile] = File(...),
    _admin: None = Depends(require_admin_token),
):
    results = await validate_upload_files(files)
    return {
        "valid": all(r["valid"] for r in results),
        "files": results,
    }


@app.post("/admin/uploads")
async def admin_upload_files(
    files: List[UploadFile] = File(...),
    _admin: None = Depends(require_admin_token),
):
    results = await save_upload_files(files)
    uploaded = [r for r in results if r.get("status") == "uploaded"]
    duplicates = [r for r in results if r.get("status") == "duplicate"]
    rejected = [r for r in results if r.get("status") == "rejected"]
    return {
        "uploaded_count": len(uploaded),
        "duplicate_count": len(duplicates),
        "rejected_count": len(rejected),
        "files": results,
    }


@app.post("/admin/ingestion/jobs")
def admin_create_ingestion_job(
    req: CreateIngestionJobRequest,
    background_tasks: BackgroundTasks,
    _admin: None = Depends(require_admin_token),
):
    document_ids = sorted(set(req.document_ids))
    if not document_ids:
        raise HTTPException(status_code=400, detail="document_ids is required")

    docs = get_documents(document_ids)
    if len(docs) != len(document_ids):
        raise HTTPException(status_code=404, detail="One or more document IDs were not found")

    invalid = [doc["id"] for doc in docs if doc.get("status") != "uploaded"]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Only documents with status 'uploaded' can be ingested in this phase: {invalid}",
        )

    try:
        job = create_job_for_documents(document_ids)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    background_tasks.add_task(run_ingestion_job, int(job["id"]), document_ids)
    return {
        "job": job,
        "document_ids": document_ids,
    }


@app.post("/admin/documents/{document_id}/reingest")
def admin_reingest_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    _admin: None = Depends(require_admin_token),
):
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    stored_path = doc.get("stored_path")
    if not stored_path:
        raise HTTPException(status_code=400, detail="Document has no stored file path")

    from pathlib import Path
    if not Path(stored_path).exists():
        raise HTTPException(status_code=400, detail="Stored file is missing on disk")

    try:
        job = create_job_for_documents([document_id], message="Re-ingest queued")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    background_tasks.add_task(run_ingestion_job, int(job["id"]), [document_id])
    return {
        "job": job,
        "document_ids": [document_id],
    }


@app.get("/admin/ingestion/jobs/{job_id}")
def admin_get_ingestion_job(
    job_id: int,
    _admin: None = Depends(require_admin_token),
):
    job = get_ingestion_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    document_ids = get_job_document_ids(job_id)
    documents = get_documents(document_ids)
    return {
        "job": job,
        "documents": documents,
    }


@app.get("/admin/documents")
def admin_list_documents(
    limit: int = Query(200, ge=1, le=1000),
    _admin: None = Depends(require_admin_token),
):
    return {
        "documents": list_documents(limit=limit),
    }


@app.delete("/admin/documents/{document_id}")
def admin_delete_document(
    document_id: int,
    _admin: None = Depends(require_admin_token),
):
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    running = get_running_job()
    if running:
        linked_ids = set(get_job_document_ids(int(running["id"])))
        if document_id in linked_ids:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete document while ingestion job {running['id']} is active",
            )

    cleanup = delete_document_assets(document_id)
    deleted = delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "deleted": True,
        "document_id": document_id,
        "file_deleted": cleanup["file_deleted"],
        "deleted_chunks": cleanup["deleted_chunks"],
    }

@app.post("/ask")
def ask(req: AskRequest, request: Request):
    t0 = time.perf_counter()
    ok = False
    hit_count = 0
    spans = {}

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        query_rewrite_enabled = req.query_rewrite_enabled

        with span("retrieve", spans):
            hits = retrieve(
                q,
                top_k=req.top_k,
                expand_lists=True,
                query_rewrite_enabled=query_rewrite_enabled,
            )
        hit_count = len(hits or [])

        if not hits:
            ok = True
            return {"question": q, "answer": NO_HITS, "citations": []}

        with span("select_llm_hits", spans):
            llm_hits = select_llm_hits(hits, max_hits=req.top_k)

        if req.mode == "llm":
            max_toks = int(getattr(settings, "MAX_ANSWER_TOKENS", 250))
            max_toks = max(32, min(max_toks, 400))

            with span("synthesize_answer", spans):
                synthesized_answer, used_hits = synthesize_answer(
                    q,
                    llm_hits,
                    max_tokens=max_toks,
                )
            answer = format_answer_with_evidence(
                synthesized_answer,
                used_hits,
                question=q,
                use_all_hits=True,
            ) if synthesized_answer and used_hits else synthesized_answer
            citations_src = used_hits
        else:
            with span("format_answer", spans):
                answer = format_answer(llm_hits, question=q)
            citations_src = llm_hits

        with span("build_citations", spans):
            citations = [
                {
                    "rank": h["rank"],
                    "source": h["source"],
                    "page_num": h.get("page_num"),
                    "page_end": h.get("page_end"),
                    "chunk_index": h["chunk_index"],
                    "semantic_score": h.get("semantic_score", h.get("score")),
                    "final_score": h.get("final_score"),
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
            "query_rewrite_enabled": query_rewrite_enabled,
            "hit_count": hit_count,
            "client_ip": request.client.host if request.client else None,
            "question_len": len(q),
            "question_sha256": sha256(q.encode("utf-8")).hexdigest(),
            "question_preview": q[:80],
            "spans_ms": spans,          # <-- the money line
        }
        logger.info("ASK_SPANS {}", log_obj)
