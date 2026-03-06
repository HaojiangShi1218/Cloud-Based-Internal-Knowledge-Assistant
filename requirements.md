# Requirements Specification — Internal Knowledge Assistant

## 1) Business Objectives

- Reduce time spent searching internal documentation
- Improve consistency and traceability of answers
- Keep proprietary knowledge out of model training/fine-tuning

## 2) Stakeholders

| Stakeholder | Role |
|---|---|
| Business users | Ask questions and consume answers |
| Engineering | Maintain ingestion, API, UI, deployment |
| Security | Review data handling and access posture |
| Product owner | Prioritize scope and quality goals |

## 3) Functional Requirements and Implementation Status

| ID | Requirement | Priority | Current status |
|---|---|---|---|
| FR-01 | Users can submit natural-language questions | High | Implemented (`/ask`, Streamlit input) |
| FR-02 | System retrieves relevant document content (top-k) | High | Implemented (OpenSearch k-NN + hybrid rerank) |
| FR-03 | System generates answers grounded in retrieved docs | High | Implemented (`extract` and `llm` modes) |
| FR-04 | System displays source citations | High | Implemented (rank/source/page/page_end/chunk + scores) |
| FR-05 | Admin can upload documents for ingestion | Medium | **Not implemented as upload UI/API**; ingestion is CLI/module based (`python -m app.ingest`) |
| FR-06 | System logs queries and response latency | Medium | Implemented (`ASK_SPANS`, retrieval sub-spans) |
| FR-07 | System handles concurrent requests | Medium | Partially addressed by FastAPI/Uvicorn deployment; no formal load test evidence in repo |
| FR-08 | Users can choose mode: evidence-only vs synthesis | Medium | Implemented (Streamlit toggle + API `mode`) |
| FR-09 | Users can toggle query rewrite | Medium | Implemented (`query-rewrite-enabled`) |
| FR-10 | Admin can clear LLM cache (debug) | Low | Implemented (`POST /debug/cache/clear`) |

## 4) Non-Functional Requirements and Current Reality

### Performance
- Target: p95 < 3s
- Current: variable by mode/question; retrieval can be sub-second to ~3s+, and LLM synthesis commonly adds ~1.5–4s.

### Scalability
- Architecture is modular and containerized.
- OpenSearch-based retrieval supports scale-up/scale-out better than prior local FAISS path.
- Horizontal scaling is possible, but cache is currently in-memory per API process.

### Security
- No fine-tuning on proprietary data (implemented).
- HTTPS/TLS termination is **not configured in current compose stack** (nginx serves port 80).
- Fine-grained document access control / auth is **not implemented**.

### Reliability / Observability
- Graceful LLM fallback path exists (`I don’t know based on the provided documents.`).
- Request-level timing and logging are implemented.
- Deterministic behavior is limited by LLM generation/caching characteristics; not guaranteed strictly across runs.

## 5) Out of Scope (MVP)

- Voice interfaces
- External customer access
- Real-time collaborative document editing
- Advanced role-based access and auth workflows

## 6) Risks and Mitigations

| Risk | Current mitigation |
|---|---|
| Hallucination | Retrieval-grounded prompting + citation constraints |
| Retrieval drift on nuanced queries | Hybrid scoring + query rewrite toggle + evaluation set testing |
| Latency spikes | Embedding model warm-up at startup, caching, retrieval instrumentation |
| Operational complexity | Containerized deployment and centralized service boundaries |
