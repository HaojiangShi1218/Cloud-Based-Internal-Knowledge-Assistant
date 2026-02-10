# System Architecture — Internal Knowledge Assistant

## 1. Architectural Goals

- Accuracy through document-grounded responses
- Security of internal data
- Scalability and maintainability
- Clear separation of concerns

---

## 2. High-Level Components

### User Interface
- Streamlit web UI for submitting questions
- Displays answers and citations
- Toggles: mode (extract vs synthesis), top‑k, query rewrite
- Debug: cache clear endpoint

### Backend API
- Orchestrates request flow
- Handles retrieval, prompt construction, and response delivery
- Exposes REST endpoints:
  - `POST /ask`
  - `POST /debug/cache/clear`
  - `GET /health`

### Document Ingestion Service
- Parses PDFs into page text
- Sliding-window chunking across pages
- Generates embeddings
- Stores metadata: source, page range, chunk index, doc chunk sequence

### Vector Database
- Stores document embeddings
- Enables semantic similarity search (FAISS)
- Corpus stats for BM25

### LLM Service
- Generates natural-language responses (synthesis mode)
- Query rewrite for implicit questions
- Uses retrieved context only (RAG)
- Citation alignment via overlap checks

### Monitoring & Logging
- Tracks request volume
- Measures latency and errors
- Supports future observability tooling

---

## 3. Data Flow

1. Admin uploads documents
2. Documents are chunked and embedded
3. Embeddings stored in vector database
4. User submits question
5. System retrieves top‑k relevant chunks (semantic + lexical reranking)
6. Prompt constructed with context
7. LLM generates grounded response (synthesis mode) or evidence-only output
8. Answer and citations returned to user

---

## 4. Deployment Model (Target)

| Layer | Technology |
|-----|-----------|
| UI | Streamlit (local), static hosting (future) |
| API | FastAPI on EC2 / Container |
| Storage | Local filesystem (MVP), S3 (future) |
| Vector DB | FAISS (MVP), OpenSearch (future) |
| AI | OpenAI / Azure OpenAI |
| Monitoring | CloudWatch |

---

## 5. Key Design Decisions

- Retrieval-Augmented Generation instead of fine-tuning
- Managed cloud services to reduce operational overhead
- Modular architecture to enable future expansion
- Evidence fidelity over aggressive synthesis
- Local-first prototype to validate retrieval quality

---

## 6. Future Enhancements

- Role-based access control
- Authentication & authorization
- Feedback-driven answer improvement
- Cost optimization and caching
- Document-structure aware retrieval (headers/TOC)
- Multi-region deployment
