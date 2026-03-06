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

`/debug/cache/clear` requires header `x-debug-token` matching `DEBUG_CACHE_CLEAR_TOKEN`.

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

## Notes and Limitations

- Retrieval quality is strong for most definition/list questions, but some nuanced tradeoff questions still need tuning.
- Query rewrite improves recall but can introduce ranking drift for some questions.
- LLM latency often dominates total latency; rewrite latency can also be material.
- Cache is in-memory per API process (not shared across replicas).

## Project Docs

- Requirements: `requirements.md`
- Architecture: `architecture.md`
