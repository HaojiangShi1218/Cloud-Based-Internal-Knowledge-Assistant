import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Any, Dict, List, Optional

from app.config import settings


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    _ensure_parent_dir(settings.SQLITE_DB_PATH)
    conn = sqlite3.connect(settings.SQLITE_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> None:
    if column_name not in _table_columns(conn, table_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


def _migrate_documents_table(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "documents", "storage_backend", "TEXT")
    _ensure_column(conn, "documents", "s3_bucket", "TEXT")
    _ensure_column(conn, "documents", "s3_key", "TEXT")
    _ensure_column(conn, "documents", "original_filename", "TEXT")
    _ensure_column(conn, "documents", "mime_type", "TEXT")

    conn.execute(
        """
        UPDATE documents
        SET storage_backend = 'local'
        WHERE storage_backend IS NULL OR storage_backend = ''
        """
    )
    conn.execute(
        """
        UPDATE documents
        SET original_filename = filename
        WHERE original_filename IS NULL OR original_filename = ''
        """
    )
    conn.execute(
        """
        UPDATE documents
        SET mime_type = 'application/pdf'
        WHERE (mime_type IS NULL OR mime_type = '')
          AND lower(filename) LIKE '%.pdf'
        """
    )


def init_admin_store() -> None:
    _ensure_parent_dir(settings.SQLITE_DB_PATH)
    os.makedirs(settings.UPLOADS_DIR, exist_ok=True)

    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                storage_backend TEXT,
                s3_bucket TEXT,
                s3_key TEXT,
                original_filename TEXT,
                mime_type TEXT,
                file_size_bytes INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                validation_error TEXT,
                last_ingested_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);
            CREATE INDEX IF NOT EXISTS idx_documents_content_sha256 ON documents(content_sha256);
            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
            CREATE INDEX IF NOT EXISTS idx_documents_storage_backend ON documents(storage_backend);
            CREATE INDEX IF NOT EXISTS idx_documents_s3_key ON documents(s3_key);

            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                total_files INTEGER NOT NULL DEFAULT 0,
                processed_files INTEGER NOT NULL DEFAULT 0,
                successful_files INTEGER NOT NULL DEFAULT 0,
                failed_files INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                error_detail TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created_at ON ingestion_jobs(created_at);

            CREATE TABLE IF NOT EXISTS job_documents (
                job_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                PRIMARY KEY (job_id, document_id),
                FOREIGN KEY (job_id) REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_job_documents_document_id ON job_documents(document_id);
            """
        )
        _migrate_documents_table(conn)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


def find_document_by_hash(content_sha256: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM documents
            WHERE content_sha256 = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (content_sha256,),
        ).fetchone()
    return _row_to_dict(row)


def find_latest_document_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM documents
            WHERE filename = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (filename,),
        ).fetchone()
    return _row_to_dict(row)


def create_document(
    *,
    filename: str,
    stored_path: str,
    file_size_bytes: int,
    content_sha256: str,
    status: str,
    validation_error: Optional[str] = None,
    doc_id: Optional[str] = None,
    storage_backend: str = "local",
    s3_bucket: Optional[str] = None,
    s3_key: Optional[str] = None,
    original_filename: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> Dict[str, Any]:
    now = utc_now_iso()
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO documents (
                doc_id,
                filename,
                stored_path,
                storage_backend,
                s3_bucket,
                s3_key,
                original_filename,
                mime_type,
                file_size_bytes,
                content_sha256,
                status,
                validation_error,
                last_ingested_at,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                filename,
                stored_path,
                storage_backend,
                s3_bucket,
                s3_key,
                original_filename or filename,
                mime_type,
                file_size_bytes,
                content_sha256,
                status,
                validation_error,
                None,
                now,
                now,
            ),
        )
        row_id = cur.lastrowid
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (row_id,)).fetchone()
    return dict(row)


def update_document(document_id: int, **fields: Any) -> Dict[str, Any]:
    allowed = {
        "doc_id",
        "filename",
        "stored_path",
        "storage_backend",
        "s3_bucket",
        "s3_key",
        "original_filename",
        "mime_type",
        "file_size_bytes",
        "content_sha256",
        "status",
        "validation_error",
        "last_ingested_at",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    updates["updated_at"] = utc_now_iso()
    if not updates:
        raise ValueError("No updatable document fields provided")

    assignments = ", ".join(f"{key} = ?" for key in updates.keys())
    values = list(updates.values()) + [document_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE documents SET {assignments} WHERE id = ?", values)
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if row is None:
        raise ValueError(f"Document not found: {document_id}")
    return dict(row)


def get_document(document_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return _row_to_dict(row)


def delete_document(document_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    return cur.rowcount > 0


def get_documents(document_ids: List[int]) -> List[Dict[str, Any]]:
    if not document_ids:
        return []
    placeholders = ", ".join("?" for _ in document_ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM documents WHERE id IN ({placeholders}) ORDER BY id ASC",
            document_ids,
        ).fetchall()
    return [dict(row) for row in rows]


def list_documents(limit: int = 200) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM documents
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_running_job() -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM ingestion_jobs
            WHERE status IN ('queued', 'running')
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    return _row_to_dict(row)


def create_ingestion_job(
    *,
    document_ids: List[int],
    total_files: int,
    status: str = "queued",
    message: Optional[str] = None,
) -> Dict[str, Any]:
    now = utc_now_iso()
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO ingestion_jobs (
                status,
                total_files,
                processed_files,
                successful_files,
                failed_files,
                message,
                error_detail,
                created_at,
                started_at,
                finished_at
            ) VALUES (?, ?, 0, 0, 0, ?, NULL, ?, NULL, NULL)
            """,
            (status, total_files, message, now),
        )
        job_id = cur.lastrowid
        for document_id in document_ids:
            conn.execute(
                "INSERT INTO job_documents (job_id, document_id) VALUES (?, ?)",
                (job_id, document_id),
            )
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row)


def update_ingestion_job(job_id: int, **fields: Any) -> Dict[str, Any]:
    allowed = {
        "status",
        "total_files",
        "processed_files",
        "successful_files",
        "failed_files",
        "message",
        "error_detail",
        "started_at",
        "finished_at",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        raise ValueError("No updatable ingestion job fields provided")

    assignments = ", ".join(f"{key} = ?" for key in updates.keys())
    values = list(updates.values()) + [job_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE ingestion_jobs SET {assignments} WHERE id = ?", values)
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise ValueError(f"Ingestion job not found: {job_id}")
    return dict(row)


def get_ingestion_job(job_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


def get_job_document_ids(job_id: int) -> List[int]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT document_id
            FROM job_documents
            WHERE job_id = ?
            ORDER BY document_id ASC
            """,
            (job_id,),
        ).fetchall()
    return [int(row["document_id"]) for row in rows]
