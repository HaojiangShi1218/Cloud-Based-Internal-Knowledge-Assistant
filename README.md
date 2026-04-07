# Internal Knowledge Assistant (IKA)

Internal RAG assistant for document-grounded Q&A with citation traceability.

## Current Status

- Backend and frontend are implemented and running (`FastAPI` + `Streamlit`)
- Retrieval is OpenSearch k-NN + lexical hybrid reranking (FAISS removed)
- Embeddings are local (`sentence-transformers/all-MiniLM-L6-v2`, 384 dims)
- Ingestion pipeline indexes chunk text + metadata + embeddings into OpenSearch
- Two answer modes are supported:
  - `extract`: evidence-first output
  - `llm`: synthesized answer with citations

## What It Does

- Parses PDFs and text files from `DOCS_DIR`
- Builds sliding-window page chunks and chunk-level embeddings
- Supports admin-managed PDF upload, validation, ingestion, re-ingestion, and hard delete
- Retrieves candidate chunks using:
  - semantic k-NN (OpenSearch `knn_vector`)
  - lexical score + BM25 + phrase/proximity/between boosts
- Supports optional query rewrite (`query-rewrite-enabled`)
- Returns citations with both `semantic_score` and `final_score`
- Logs request timing with high-level and retrieval sub-spans
- Supports clearing in-memory LLM cache via debug endpoint

## API (Current)

- `POST /api/ask`
- `POST /api/debug/cache/clear`
- `GET /api/health`
- `POST /api/admin/uploads/validate`
- `POST /api/admin/uploads`
- `POST /api/admin/ingestion/jobs`
- `GET /api/admin/ingestion/jobs/{job_id}`
- `GET /api/admin/documents`
- `POST /api/admin/documents/{document_id}/reingest`
- `DELETE /api/admin/documents/{document_id}`

`/debug/cache/clear` requires header `x-debug-token` matching `DEBUG_CACHE_CLEAR_TOKEN`.

All `/api/admin/*` endpoints require header `X-Admin-Token` matching `ADMIN_TOKEN`.

`/ask` response citations include:
- `rank`
- `source`
- `page_num`
- `page_end`
- `chunk_index`
- `semantic_score`
- `final_score`

## Tech Stack

- Backend: FastAPI, Uvicorn, Loguru
- UI: Streamlit
- Embeddings: sentence-transformers + PyTorch (local model)
- Vector store: Amazon OpenSearch k-NN (`ika_chunks_v1`)
- LLM: OpenAI Chat Completions (`gpt-4o-mini`)
- Ingestion: `pypdf` + OpenSearch bulk indexing
- Deployment: Docker Compose (api/ui/nginx/fluent-bit/opensearch)

## Runbook (Current)

From repo root:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Ingest documents:

```bash
docker compose -f docker-compose.prod.yml exec api python -m app.ingest
```

UI is served through nginx (`/`), API through `/api/*`.

## Admin Document Management

### Configure Admin Token

Set these values in `.env.prod` for the API and UI:

```env
ADMIN_TOKEN=your-admin-token
UPLOADS_DIR=data/uploads
MAX_UPLOAD_MB=50
SQLITE_DB_PATH=data/admin_metadata.db
DISPLAY_TIMEZONE=America/New_York
```

- `ADMIN_TOKEN` protects all admin document management endpoints
- `UPLOADS_DIR` stores uploaded PDFs on local disk
- `MAX_UPLOAD_MB` is enforced by the backend upload validation
- `SQLITE_DB_PATH` stores document/job metadata
- `DISPLAY_TIMEZONE` controls admin UI timestamp rendering

### How Uploads Work

1. Admin opens the Streamlit `Admin Documents` page.
2. Admin enters and saves the admin token.
3. PDFs are selected and optionally validated before upload.
4. Validation checks:
   - PDF-only
   - max file size
   - duplicate content hash
5. Uploaded files are saved to `UPLOADS_DIR` on local disk.
6. SQLite metadata rows are created/updated for each uploaded document.

Notes:
- Admin upload flow currently supports PDF only.
- CLI ingestion still supports `.pdf`, `.txt`, and `.md` from `DOCS_DIR`.

### How Ingestion Jobs Work

- Uploading a file does not automatically index it.
- Admin starts ingestion from the UI or by calling `POST /api/admin/ingestion/jobs`.
- Jobs run in the background using FastAPI background tasks.
- Only one ingestion job is allowed at a time.
- Job metadata is stored in SQLite:
  - queued/running/completed/failed status
  - file counters
  - timestamps
- Ingestion reuses the same chunking and OpenSearch indexing path as CLI ingestion.

### Delete and Re-ingest Behavior

- Delete is a hard delete.
- Deleting a document removes:
  - the uploaded file from local disk
  - indexed OpenSearch chunks for that `doc_id`
  - the SQLite document row
  - related job-document links
- Re-ingest uses the existing stored file in `UPLOADS_DIR`.
- Re-ingest creates a new background job and re-runs indexing for that file.
- If the stored file is missing, re-ingest fails with a clear error.

### Version Replacement Behavior

- Documents are indexed using a content-hash-based `doc_id`.
- If the same logical file is replaced with different content, the previous indexed chunks for the old `doc_id` are removed during replacement/re-ingestion cleanup.
- This prevents stale chunks from older document versions from remaining in OpenSearch.

## Notes and Limitations

- Retrieval quality is strong for most definition/list questions, but some nuanced tradeoff questions still need tuning.
- Query rewrite improves recall but can introduce ranking drift for some questions.
- LLM latency often dominates total latency; rewrite latency can also be material.
- Cache is in-memory per API process (not shared across replicas).
- Admin document metadata uses local SQLite and uploaded files use local disk; this is single-node oriented.
- Admin auth is a shared token, not user-level auth or RBAC.
- Streamlit admin UI depends on backend/API readiness and does not yet have a richer readiness handshake.
- Only one ingestion job can run at a time.
- Upload management is PDF-only in the admin flow.

## Known Future Improvements

- Replace shared admin token with stronger auth/RBAC.
- Move uploaded file storage and metadata to a more scalable multi-node design (for example S3 + managed DB).
- Add richer admin readiness/health signals in the UI.
- Add document replace/version history controls in the UI.
- Add batch job history and better ingestion observability.
- Add document-level access control if the project expands beyond trusted internal use.

## Project Docs

- Requirements: `requirements.md`
- Architecture: `architecture.md`
