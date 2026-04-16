import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional

import boto3
from botocore.exceptions import ClientError

from app.config import settings


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def storage_backend() -> str:
    backend = (settings.DOCUMENT_STORAGE_BACKEND or "local").strip().lower()
    if backend not in {"local", "s3"}:
        raise ValueError("DOCUMENT_STORAGE_BACKEND must be 'local' or 's3'")
    return backend


def _s3_client():
    return boto3.client("s3", region_name=settings.AWS_REGION)


def _require_bucket(bucket_name: Optional[str] = None) -> str:
    bucket = (bucket_name or settings.S3_BUCKET_NAME or "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME is required when using S3 document storage")
    return bucket


def _safe_filename(filename: str) -> str:
    name = Path(filename or "document.pdf").name
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("._")
    return cleaned or "document.pdf"


def generate_s3_object_key(filename: str, content_sha256: str) -> str:
    """
    Build a deterministic object key for an uploaded source document.

    The content hash keeps objects stable across duplicate uploads, while the
    original filename keeps the S3 path readable for operators.
    """
    digest = (content_sha256 or "").strip()
    if not digest:
        raise ValueError("content_sha256 is required to generate an S3 object key")

    prefix = (settings.S3_PREFIX or "").strip().strip("/")
    safe_name = _safe_filename(filename)
    key = f"{digest}/{safe_name}"
    return f"{prefix}/{key}" if prefix else key


def upload_fileobj_to_s3(
    fileobj: BinaryIO,
    object_key: str,
    content_type: Optional[str] = None,
) -> str:
    bucket = _require_bucket()
    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    if hasattr(fileobj, "seek"):
        fileobj.seek(0)

    kwargs = {"ExtraArgs": extra_args} if extra_args else {}
    _s3_client().upload_fileobj(fileobj, bucket, object_key, **kwargs)
    return object_key


def s3_uri(object_key: str, bucket_name: Optional[str] = None) -> str:
    return f"s3://{_require_bucket(bucket_name)}/{object_key}"


def s3_object_exists(object_key: str, bucket_name: Optional[str] = None) -> bool:
    bucket = _require_bucket(bucket_name)
    try:
        _s3_client().head_object(Bucket=bucket, Key=object_key)
        return True
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404:
            return False
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def delete_s3_object(object_key: str, bucket_name: Optional[str] = None) -> bool:
    bucket = _require_bucket(bucket_name)
    existed = s3_object_exists(object_key, bucket_name=bucket)
    _s3_client().delete_object(Bucket=bucket, Key=object_key)
    return existed


def download_s3_object_to_tempfile(
    object_key: str,
    suffix: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> str:
    bucket = _require_bucket(bucket_name)
    suffix = suffix if suffix is not None else Path(object_key).suffix
    fd, temp_path = tempfile.mkstemp(prefix="ika_doc_", suffix=suffix)
    os.close(fd)
    try:
        _s3_client().download_file(bucket, object_key, temp_path)
        return temp_path
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def generate_presigned_s3_url(
    object_key: str,
    expires_in: Optional[int] = None,
    bucket_name: Optional[str] = None,
) -> str:
    bucket = _require_bucket(bucket_name)
    expiry = int(expires_in or settings.S3_PRESIGN_EXPIRY_SECONDS)
    return _s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=expiry,
    )
