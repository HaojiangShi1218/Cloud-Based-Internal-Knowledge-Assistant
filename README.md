# Internal Knowledge Assistant (IKA)

A local-first internal AI assistant designed to provide accurate, document-grounded answers to organizational knowledge while preserving evidence fidelity and enabling future cloud scaling.

This project demonstrates systems thinking, requirements analysis, and cloud-native architecture design aligned with Technical Business Analyst, Solutions Engineer, and future AI Product Manager roles.

---

## Problem Statement

Employees often spend excessive time searching across fragmented internal documentation (policies, onboarding materials, system docs), resulting in productivity loss and inconsistent information retrieval.

---

## Solution Overview

The Internal Knowledge Assistant (IKA) is a RAG system that:
- Retrieves relevant document chunks with hybrid semantic + lexical signals
- Generates grounded responses or evidence-only extracts
- Provides citations for traceability
- Is designed for scalability, security, and reliability (cloud-ready)

---

## Key Features (Current)

- Natural-language question answering
- Document ingestion (PDF parsing, sliding-window chunking, metadata)
- Vector-based semantic retrieval with hybrid reranking
- LLM-generated answers with citations + evidence-only mode
- Query rewrite (LLM) for implicit questions
- Basic logging and cache controls

---

## High-Level Architecture
![System Architecture Diagram](diagrams/SysArchitecture.drawio.png)

---

## Tech Stack (Current)

- Backend: Python, FastAPI
- UI: Streamlit
- Embeddings: sentence-transformers (all-MiniLM-L6-v2, local)
- Vector DB: FAISS (local)
- Retrieval: hybrid semantic + BM25 + sentence rerank
- LLM: OpenAI API (answer synthesis + query rewrite)
- Storage: Local filesystem (S3 planned)
- Cloud Target: AWS (future deployment)

---

## Learning & Career Objectives

This project is designed to demonstrate:
- Translation of business requirements into technical systems
- Cloud and AI system design trade-offs
- Understanding of modern DevOps, scalability, and reliability concepts
- Clear technical communication for stakeholders

---

## Project Status

- [x] Requirements definition
- [x] Architecture design
- [x] Backend API implementation
- [x] Document ingestion pipeline
- [x] Retrieval-augmented QA
- [ ] Cloud deployment

---

## License

