import hashlib
from io import BytesIO
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
from app.document_storage import (
    delete_s3_object,
    generate_s3_object_key,
    s3_uri,
    storage_backend,
    upload_fileobj_to_s3,
)
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


def _mime_type(upload: UploadFile) -> str:
    return upload.content_type or "application/pdf"


def _persist_upload(
    *,
    filename: str,
    data: bytes,
    content_sha256: str,
    mime_type: str,
) -> Dict[str, Any]:
    backend = storage_backend()
    if backend == "local":
        stored_path = _stored_upload_path(content_sha256, filename)
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stored_path, "wb") as f:
            f.write(data)
        return {
            "storage_backend": "local",
            "stored_path": str(stored_path),
            "s3_bucket": None,
            "s3_key": None,
            "original_filename": filename,
            "mime_type": mime_type,
        }

    object_key = generate_s3_object_key(filename, content_sha256)
    upload_fileobj_to_s3(BytesIO(data), object_key, content_type=mime_type)
    return {
        "storage_backend": "s3",
        "stored_path": s3_uri(object_key),
        "s3_bucket": settings.S3_BUCKET_NAME,
        "s3_key": object_key,
        "original_filename": filename,
        "mime_type": mime_type,
    }


def _cleanup_replaced_storage(existing_doc: Dict[str, Any], new_stored_path: str, new_s3_key: str | None) -> None:
    backend = (existing_doc.get("storage_backend") or "local").lower()
    if backend == "s3":
        old_s3_key = existing_doc.get("s3_key")
        if old_s3_key and old_s3_key != new_s3_key:
            try:
                delete_s3_object(old_s3_key, bucket_name=existing_doc.get("s3_bucket"))
            except Exception:
                pass
        return

    old_path = existing_doc.get("stored_path")
    if old_path and old_path != new_stored_path:
        try:
            Path(old_path).unlink(missing_ok=True)
        except OSError:
            pass


async def save_upload_files(files: List[UploadFile]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if storage_backend() == "local":
        uploads_dir = Path(settings.UPLOADS_DIR)
        uploads_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        data = await _read_upload_bytes(upload)
        filename = upload.filename or "unnamed.pdf"
        mime_type = _mime_type(upload)
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

        storage_meta = _persist_upload(
            filename=filename,
            data=data,
            content_sha256=content_sha256,
            mime_type=mime_type,
        )

        existing_by_name = find_latest_document_by_filename(filename)
        if existing_by_name and existing_by_name["content_sha256"] != content_sha256:
            old_doc_id = existing_by_name.get("doc_id")
            if old_doc_id:
                delete_chunks_by_doc_id(old_doc_id)
            _cleanup_replaced_storage(
                existing_by_name,
                new_stored_path=storage_meta["stored_path"],
                new_s3_key=storage_meta.get("s3_key"),
            )

            document = update_document(
                int(existing_by_name["id"]),
                doc_id=None,
                stored_path=storage_meta["stored_path"],
                storage_backend=storage_meta["storage_backend"],
                s3_bucket=storage_meta["s3_bucket"],
                s3_key=storage_meta["s3_key"],
                original_filename=storage_meta["original_filename"],
                mime_type=storage_meta["mime_type"],
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
            stored_path=storage_meta["stored_path"],
            storage_backend=storage_meta["storage_backend"],
            s3_bucket=storage_meta["s3_bucket"],
            s3_key=storage_meta["s3_key"],
            original_filename=storage_meta["original_filename"],
            mime_type=storage_meta["mime_type"],
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
