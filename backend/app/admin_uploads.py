import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import UploadFile

from app.admin_store import (
    create_document,
    find_document_by_hash,
    find_latest_document_by_filename,
    update_document,
)
from app.config import settings
from app.vectorstores.opensearch_store import delete_chunks_by_doc_id


def _max_upload_bytes() -> int:
    return int(settings.MAX_UPLOAD_MB) * 1024 * 1024


async def _read_upload_bytes(upload: UploadFile) -> bytes:
    data = await upload.read()
    await upload.close()
    return data


def _validate_file_bytes(filename: str, data: bytes) -> Tuple[bool, str]:
    ext = Path(filename or "").suffix.lower()
    if ext != ".pdf":
        return False, "Only PDF files are allowed"
    if len(data) > _max_upload_bytes():
        return False, f"File exceeds MAX_UPLOAD_MB ({settings.MAX_UPLOAD_MB} MB)"
    if not data:
        return False, "File is empty"
    return True, ""


async def validate_upload_files(files: List[UploadFile]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen_hashes = set()
    for upload in files:
        data = await _read_upload_bytes(upload)
        filename = upload.filename or "unnamed.pdf"
        valid, error = _validate_file_bytes(filename, data)
        content_sha256 = hashlib.sha256(data).hexdigest() if data else None
        duplicate = False
        if content_sha256:
            duplicate = content_sha256 in seen_hashes or bool(find_document_by_hash(content_sha256))
            seen_hashes.add(content_sha256)
        results.append(
            {
                "filename": filename,
                "size_bytes": len(data),
                "content_sha256": content_sha256,
                "valid": valid,
                "error": error or None,
                "duplicate": duplicate,
            }
        )
    return results


def _stored_upload_path(content_sha256: str, filename: str) -> Path:
    ext = Path(filename).suffix.lower() or ".pdf"
    stored_name = f"{content_sha256}{ext}"
    return Path(settings.UPLOADS_DIR) / stored_name


async def save_upload_files(files: List[UploadFile]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    uploads_dir = Path(settings.UPLOADS_DIR)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        data = await _read_upload_bytes(upload)
        filename = upload.filename or "unnamed.pdf"
        valid, error = _validate_file_bytes(filename, data)
        content_sha256 = hashlib.sha256(data).hexdigest() if data else None

        if not valid:
            results.append(
                {
                    "filename": filename,
                    "size_bytes": len(data),
                    "content_sha256": content_sha256,
                    "status": "rejected",
                    "error": error,
                }
            )
            continue

        existing = find_document_by_hash(content_sha256)
        if existing:
            updated = update_document(existing["id"], status=existing["status"], validation_error=None)
            results.append(
                {
                    "document": updated,
                    "duplicate": True,
                    "status": "duplicate",
                    "error": None,
                }
            )
            continue

        stored_path = _stored_upload_path(content_sha256, filename)
        with open(stored_path, "wb") as f:
            f.write(data)

        existing_by_name = find_latest_document_by_filename(filename)
        if existing_by_name and existing_by_name["content_sha256"] != content_sha256:
            old_path = existing_by_name.get("stored_path")
            old_doc_id = existing_by_name.get("doc_id")
            if old_doc_id:
                delete_chunks_by_doc_id(old_doc_id)
            if old_path and old_path != str(stored_path):
                try:
                    Path(old_path).unlink(missing_ok=True)
                except OSError:
                    pass

            document = update_document(
                int(existing_by_name["id"]),
                doc_id=None,
                stored_path=str(stored_path),
                file_size_bytes=len(data),
                content_sha256=content_sha256,
                status="uploaded",
                validation_error=None,
                last_ingested_at=None,
            )
            results.append(
                {
                    "document": document,
                    "duplicate": False,
                    "replaced": True,
                    "status": "uploaded",
                    "error": None,
                }
            )
            continue

        document = create_document(
            filename=filename,
            stored_path=str(stored_path),
            file_size_bytes=len(data),
            content_sha256=content_sha256,
            status="uploaded",
            validation_error=None,
        )
        results.append(
            {
                "document": document,
                "duplicate": False,
                "replaced": False,
                "status": "uploaded",
                "error": None,
            }
        )

    return results
