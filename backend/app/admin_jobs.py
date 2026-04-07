import os
import threading
from typing import Any, Dict, List

from loguru import logger

from app.admin_store import (
    create_ingestion_job,
    get_document,
    get_documents,
    get_running_job,
    update_document,
    update_ingestion_job,
    utc_now_iso,
)
from app.ingest import ingest_paths
from app.vectorstores.opensearch_store import delete_chunks_by_doc_id


_JOB_CREATE_LOCK = threading.Lock()


def create_job_for_documents(document_ids: List[int], message: str = "Job queued") -> Dict[str, Any]:
    with _JOB_CREATE_LOCK:
        running = get_running_job()
        if running:
            raise RuntimeError(f"Ingestion job already running: {running['id']}")
        return create_ingestion_job(
            document_ids=document_ids,
            total_files=len(document_ids),
            status="queued",
            message=message,
        )


def run_ingestion_job(job_id: int, document_ids: List[int]) -> None:
    docs = get_documents(document_ids)
    if not docs:
        update_ingestion_job(
            job_id,
            status="failed",
            message="No documents found for job",
            error_detail="Selected document IDs were not found",
            started_at=utc_now_iso(),
            finished_at=utc_now_iso(),
        )
        return

    update_ingestion_job(
        job_id,
        status="running",
        message="Ingestion in progress",
        started_at=utc_now_iso(),
    )

    processed = 0
    successful = 0
    failed = 0

    for doc in docs:
        doc_id = int(doc["id"])
        stored_path = doc["stored_path"]
        update_document(doc_id, status="ingesting", validation_error=None)
        current_doc_id = doc.get("doc_id")
        if current_doc_id:
            delete_chunks_by_doc_id(current_doc_id)

        try:
            results = ingest_paths([stored_path], source_names={stored_path: doc.get("filename") or os.path.basename(stored_path)})
            result = results[0] if results else None
        except Exception as exc:
            logger.exception("Background ingestion failed for document {}", doc_id)
            result = {
                "status": "failed",
                "error": str(exc),
                "doc_id": None,
            }

        processed += 1
        status = result.get("status") if result else "failed"
        error = result.get("error") if result else "Unknown ingestion failure"

        if status == "indexed":
            successful += 1
            update_document(
                doc_id,
                doc_id=result.get("doc_id"),
                status="indexed",
                validation_error=None,
                last_ingested_at=utc_now_iso(),
            )
        else:
            failed += 1
            update_document(
                doc_id,
                doc_id=result.get("doc_id"),
                status="failed",
                validation_error=error or f"Ingestion ended with status: {status}",
            )

        update_ingestion_job(
            job_id,
            processed_files=processed,
            successful_files=successful,
            failed_files=failed,
            message=f"Processed {processed} of {len(docs)} files",
        )

    final_status = "completed" if failed == 0 else ("failed" if successful == 0 else "completed_with_errors")
    update_ingestion_job(
        job_id,
        status=final_status,
        processed_files=processed,
        successful_files=successful,
        failed_files=failed,
        message=f"Finished processing {processed} files",
        finished_at=utc_now_iso(),
    )


def delete_document_assets(document_id: int) -> Dict[str, Any]:
    doc = get_document(document_id)
    if not doc:
        raise ValueError(f"Document not found: {document_id}")

    file_deleted = False
    stored_path = doc.get("stored_path")
    if stored_path:
        try:
            if os.path.exists(stored_path):
                os.remove(stored_path)
                file_deleted = True
        except OSError:
            file_deleted = False

    deleted_chunks = 0
    if doc.get("doc_id"):
        deleted_chunks = delete_chunks_by_doc_id(doc["doc_id"])

    return {
        "document": doc,
        "file_deleted": file_deleted,
        "deleted_chunks": deleted_chunks,
    }
