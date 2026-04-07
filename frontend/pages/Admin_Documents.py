import os
import time
from typing import Any, Dict, List

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://api:8000")


def _ensure_state() -> None:
    st.session_state.setdefault("api_url", API_URL)
    st.session_state.setdefault("admin_token", "")
    st.session_state.setdefault("admin_validation_results", None)
    st.session_state.setdefault("admin_upload_results", None)
    st.session_state.setdefault("admin_last_job_id", None)
    st.session_state.setdefault("admin_documents_cache", None)


def _admin_headers() -> Dict[str, str]:
    token = (st.session_state.get("admin_token") or "").strip()
    if not token:
        raise RuntimeError("Enter and save an admin token first.")
    return {
        "X-Admin-Token": token,
        "accept": "application/json",
    }


def _api_base() -> str:
    return st.session_state.get("api_url", API_URL).rstrip("/")


def _format_error(resp: requests.Response) -> str:
    try:
        payload = resp.json()
        detail = payload.get("detail")
        if detail:
            return f"{resp.status_code} {resp.reason}: {detail}"
    except Exception:
        pass
    return f"{resp.status_code} {resp.reason}: {resp.text}"


def _admin_request(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    url = _api_base() + path
    headers = dict(kwargs.pop("headers", {}) or {})
    headers.update(_admin_headers())
    resp = requests.request(method, url, headers=headers, timeout=kwargs.pop("timeout", 120), **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(_format_error(resp))
    if not resp.content:
        return {}
    return resp.json()


def _multipart_files(files: List[Any]) -> List[Any]:
    parts = []
    for file in files:
        parts.append(
            (
                "files",
                (
                    file.name,
                    file.getvalue(),
                    file.type or "application/pdf",
                ),
            )
        )
    return parts


def _load_documents(force: bool = False) -> List[Dict[str, Any]]:
    if not force and st.session_state.get("admin_documents_cache") is not None:
        return st.session_state["admin_documents_cache"]
    data = _admin_request("GET", "/admin/documents")
    docs = data.get("documents", []) or []
    st.session_state["admin_documents_cache"] = docs
    return docs


def _load_job(job_id: int) -> Dict[str, Any]:
    return _admin_request("GET", f"/admin/ingestion/jobs/{job_id}")


def _document_label(doc: Dict[str, Any]) -> str:
    return f"#{doc['id']} | {doc['filename']} | {doc['status']}"


def _human_size(size_bytes: Any) -> str:
    if not isinstance(size_bytes, (int, float)):
        return "-"
    if size_bytes < 1024:
        return f"{int(size_bytes)} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


_ensure_state()

st.set_page_config(page_title="Admin Documents", page_icon="🗂", layout="wide")

st.title("Admin Document Management")
st.caption("Validate, upload, ingest, re-ingest, and delete documents through the admin backend.")

with st.sidebar:
    st.header("Admin Access")
    st.text_input("Backend URL", value=st.session_state.get("api_url", API_URL), key="api_url")
    token_input = st.text_input(
        "Admin Token",
        value=st.session_state.get("admin_token", ""),
        type="password",
        key="admin_token_input",
    )
    col_save, col_clear = st.columns(2)
    with col_save:
        if st.button("Save Token", use_container_width=True):
            st.session_state["admin_token"] = token_input.strip()
            st.success("Admin token saved in session state.")
    with col_clear:
        if st.button("Clear Token", use_container_width=True):
            st.session_state["admin_token"] = ""
            st.session_state["admin_token_input"] = ""
            st.info("Admin token cleared.")

    token_saved = bool((st.session_state.get("admin_token") or "").strip())
    st.caption("Token status: saved" if token_saved else "Token status: not saved")

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh active job", value=True)
    if st.button("Refresh Data", use_container_width=True):
        st.session_state["admin_documents_cache"] = None
        st.rerun()

if not (st.session_state.get("admin_token") or "").strip():
    st.warning("Enter and save an admin token in the sidebar to use admin actions.")

st.subheader("Upload")
uploaded_files = st.file_uploader(
    "Select PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="admin_file_uploader",
)

upload_col1, upload_col2 = st.columns(2)
with upload_col1:
    if st.button("Validate Files", use_container_width=True, disabled=not uploaded_files):
        try:
            result = _admin_request("POST", "/admin/uploads/validate", files=_multipart_files(uploaded_files))
            st.session_state["admin_validation_results"] = result
            st.success("Validation completed.")
        except Exception as exc:
            st.error(str(exc))

with upload_col2:
    if st.button("Upload Files", use_container_width=True, disabled=not uploaded_files):
        try:
            result = _admin_request("POST", "/admin/uploads", files=_multipart_files(uploaded_files))
            st.session_state["admin_upload_results"] = result
            st.session_state["admin_documents_cache"] = None
            st.success("Upload completed.")
        except Exception as exc:
            st.error(str(exc))

validation_results = st.session_state.get("admin_validation_results")
if validation_results:
    st.markdown("Validation results")
    st.dataframe(validation_results.get("files", []), use_container_width=True)

upload_results = st.session_state.get("admin_upload_results")
if upload_results:
    st.markdown("Upload results")
    st.json(upload_results)

st.divider()
st.subheader("Ingestion")
docs: List[Dict[str, Any]] = []
try:
    docs = _load_documents(force=False)
except Exception as exc:
    st.error(str(exc))

ingest_candidates = [doc for doc in docs if doc.get("status") in {"uploaded", "failed"}]
selected_labels = st.multiselect(
    "Select uploaded or failed documents to ingest",
    options=[_document_label(doc) for doc in ingest_candidates],
    key="admin_ingest_selection",
)
selected_ids = [
    doc["id"] for doc in ingest_candidates
    if _document_label(doc) in selected_labels
]

if st.button("Start Ingestion Job", use_container_width=True, disabled=not selected_ids):
    try:
        data = _admin_request("POST", "/admin/ingestion/jobs", json={"document_ids": selected_ids})
        job = data.get("job", {})
        st.session_state["admin_last_job_id"] = job.get("id")
        st.session_state["admin_documents_cache"] = None
        st.success(f"Started ingestion job #{job.get('id')}.")
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.subheader("Job Status")
last_job_id = st.session_state.get("admin_last_job_id")
if last_job_id:
    try:
        job_payload = _load_job(int(last_job_id))
        job = job_payload.get("job", {})
        total = int(job.get("total_files") or 0)
        processed = int(job.get("processed_files") or 0)
        progress = (processed / total) if total else 0.0
        st.caption(f"Last job: #{job.get('id')} | status: {job.get('status')}")
        st.progress(progress)
        st.write(
            {
                "processed_files": processed,
                "total_files": total,
                "successful_files": job.get("successful_files"),
                "failed_files": job.get("failed_files"),
                "message": job.get("message"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
            }
        )
        if auto_refresh and job.get("status") in {"queued", "running"}:
            time.sleep(2)
            st.rerun()
    except Exception as exc:
        st.error(str(exc))
else:
    st.info("No ingestion job started from this UI session yet.")

st.divider()
st.subheader("Document Management")
try:
    docs = _load_documents(force=True)
except Exception as exc:
    docs = []
    st.error(str(exc))

if not docs:
    st.info("No uploaded documents found.")
else:
    table_rows = [
        {
            "id": doc["id"],
            "filename": doc["filename"],
            "status": doc["status"],
            "size": _human_size(doc.get("file_size_bytes")),
            "updated_at": doc.get("updated_at"),
            "last_ingested_at": doc.get("last_ingested_at"),
        }
        for doc in docs
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.markdown("Actions")
    for doc in docs:
        with st.container():
            col_info, col_reingest, col_delete = st.columns([6, 2, 2])
            with col_info:
                st.write(
                    f"#{doc['id']} | {doc['filename']} | status={doc['status']} | "
                    f"size={_human_size(doc.get('file_size_bytes'))}"
                )
                st.caption(
                    f"Updated: {doc.get('updated_at')} | Last ingested: {doc.get('last_ingested_at')}"
                )
            with col_reingest:
                can_reingest = doc.get("status") in {"indexed", "failed", "uploaded"}
                if st.button("Re-ingest", key=f"reingest_{doc['id']}", use_container_width=True, disabled=not can_reingest):
                    try:
                        payload = _admin_request("POST", f"/admin/documents/{doc['id']}/reingest")
                        job = payload.get("job", {})
                        st.session_state["admin_last_job_id"] = job.get("id")
                        st.session_state["admin_documents_cache"] = None
                        st.success(f"Queued re-ingest job #{job.get('id')} for document #{doc['id']}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            with col_delete:
                if st.button("Delete", key=f"delete_{doc['id']}", use_container_width=True):
                    try:
                        _admin_request("DELETE", f"/admin/documents/{doc['id']}")
                        st.session_state["admin_documents_cache"] = None
                        st.success(f"Deleted document #{doc['id']}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
