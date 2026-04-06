import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

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
