import os
import time
from html import escape
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://api:8000")


def _ensure_state() -> None:
    st.session_state.setdefault("api_url", API_URL)
    st.session_state.setdefault("admin_token", "")
    st.session_state.setdefault("admin_validation_results", None)
    st.session_state.setdefault("admin_upload_results", None)
    st.session_state.setdefault("admin_upload_message", None)
    st.session_state.setdefault("admin_upload_message_kind", "info")
    st.session_state.setdefault("admin_last_job_id", None)
    st.session_state.setdefault("admin_documents_cache", None)
    st.session_state.setdefault("admin_file_uploader_key", "admin_file_uploader_0")
    st.session_state.setdefault("admin_delete_confirm_id", None)


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


def _format_display_time(ts: Any) -> str:
    if not ts:
        return "—"
    if not isinstance(ts, str):
        return "—"
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        tz_name = os.getenv("DISPLAY_TIMEZONE") or os.getenv("TZ") or "America/New_York"
        local_dt = parsed.astimezone(ZoneInfo(tz_name))
        display = local_dt.strftime("%Y-%m-%d %I:%M %p")
        return display.replace(" 0", " ")
    except Exception:
        return ts


def _status_badge_html(status: Any) -> str:
    value = str(status or "unknown").strip().lower()
    colors = {
        "indexed": ("#166534", "#dcfce7", "#bbf7d0"),
        "completed": ("#166534", "#dcfce7", "#bbf7d0"),
        "uploaded": ("#1d4ed8", "#dbeafe", "#bfdbfe"),
        "queued": ("#1d4ed8", "#dbeafe", "#bfdbfe"),
        "running": ("#92400e", "#fef3c7", "#fde68a"),
        "ingesting": ("#92400e", "#fef3c7", "#fde68a"),
        "failed": ("#991b1b", "#fee2e2", "#fecaca"),
    }
    fg, bg, border = colors.get(value, ("#374151", "#f3f4f6", "#d1d5db"))
    label = escape(value.replace("_", " ").title())
    return (
        f"<span style='display:inline-block;padding:0.15rem 0.55rem;border-radius:999px;"
        f"font-size:0.82rem;font-weight:600;color:{fg};background:{bg};"
        f"border:1px solid {border};'>{label}</span>"
    )


def _render_document_table(docs: List[Dict[str, Any]]) -> None:
    rows = []
    for doc in docs:
        rows.append(
            "<tr>"
            f"<td>{doc['id']}</td>"
            f"<td>{escape(str(doc['filename']))}</td>"
            f"<td>{_status_badge_html(doc.get('status'))}</td>"
            f"<td>{escape(_human_size(doc.get('file_size_bytes')))}</td>"
            f"<td>{escape(_format_display_time(doc.get('created_at')))}</td>"
            f"<td>{escape(_format_display_time(doc.get('updated_at')))}</td>"
            f"<td>{escape(_format_display_time(doc.get('last_ingested_at')))}</td>"
            "</tr>"
        )
    table_html = (
        "<table style='width:100%; border-collapse:collapse; font-size:0.95rem;'>"
        "<thead>"
        "<tr>"
        "<th style='text-align:left; padding:0.55rem; border-bottom:1px solid #e5e7eb;'>ID</th>"
        "<th style='text-align:left; padding:0.55rem; border-bottom:1px solid #e5e7eb;'>Filename</th>"
        "<th style='text-align:left; padding:0.55rem; border-bottom:1px solid #e5e7eb;'>Status</th>"
        "<th style='text-align:left; padding:0.55rem; border-bottom:1px solid #e5e7eb;'>Size</th>"
        "<th style='text-align:left; padding:0.55rem; border-bottom:1px solid #e5e7eb;'>Created</th>"
        "<th style='text-align:left; padding:0.55rem; border-bottom:1px solid #e5e7eb;'>Updated</th>"
        "<th style='text-align:left; padding:0.55rem; border-bottom:1px solid #e5e7eb;'>Last Ingested</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _friendly_exception_message(exc: Exception) -> str:
    text = str(exc)
    if "HTTPConnectionPool" in text and "api" in text and "Connection refused" in text:
        return (
            "The admin backend is not reachable at the configured backend URL. "
            "Confirm the API container is running, then check the Backend URL in Advanced settings."
        )
    return text


_ensure_state()

st.set_page_config(page_title="Admin Documents", page_icon="🗂", layout="wide")

st.title("Admin Document Management")
st.caption("Follow the steps below to connect, upload PDFs, run ingestion, monitor progress, and manage documents.")

with st.sidebar:
    st.header("Step 1: Admin Access")
    st.caption("Connect once, then use the workflow on the main page.")
    token_input = st.text_input(
        "Admin Token",
        value=st.session_state.get("admin_token", ""),
        type="password",
        key="admin_token_input",
    )
    if st.button("Save Token", use_container_width=True):
        st.session_state["admin_token"] = token_input.strip()
        st.success("Admin token saved in session state.")

    token_saved = bool((st.session_state.get("admin_token") or "").strip())
    token_label = "Saved" if token_saved else "Not saved"
    token_fg = "#166534" if token_saved else "#92400e"
    token_bg = "#dcfce7" if token_saved else "#fef3c7"
    token_border = "#bbf7d0" if token_saved else "#fde68a"
    st.markdown(
        (
            "<div style='margin-top:0.25rem;'>"
            "<span style='font-size:0.85rem;color:#6b7280;margin-right:0.45rem;'>Token</span>"
            f"<span style='display:inline-block;padding:0.1rem 0.5rem;border-radius:999px;"
            f"font-size:0.8rem;font-weight:600;color:{token_fg};background:{token_bg};"
            f"border:1px solid {token_border};'>{token_label}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.expander("Advanced", expanded=False):
        st.text_input("Backend URL", value=st.session_state.get("api_url", API_URL), key="api_url")
        auto_refresh = st.checkbox("Auto-refresh job status", value=True)
        if st.button("Refresh Data", use_container_width=True):
            st.session_state["admin_documents_cache"] = None
            st.rerun()

if not (st.session_state.get("admin_token") or "").strip():
    st.warning("Enter and save an admin token in the sidebar to use admin actions.")

st.subheader("Step 2: Validate & Upload Files")
st.caption("Validate PDFs first, then upload the accepted files into managed storage.")
uploaded_files = st.file_uploader(
    "Select PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key=st.session_state["admin_file_uploader_key"],
)
st.caption("PDF only. Max 50 MB per file. If a file gets stuck in an error state, use Clear Selected Files.")

upload_col1, upload_col2, upload_col3 = st.columns(3)
with upload_col1:
    if st.button("Validate Files", use_container_width=True, disabled=not uploaded_files):
        try:
            result = _admin_request("POST", "/admin/uploads/validate", files=_multipart_files(uploaded_files))
            st.session_state["admin_validation_results"] = result
            st.session_state["admin_upload_message"] = "Validation completed."
            st.session_state["admin_upload_message_kind"] = "success"
        except Exception as exc:
            st.session_state["admin_upload_message"] = str(exc)
            st.session_state["admin_upload_message_kind"] = "error"

with upload_col2:
    if st.button("Upload Files", use_container_width=True, disabled=not uploaded_files):
        try:
            result = _admin_request("POST", "/admin/uploads", files=_multipart_files(uploaded_files))
            st.session_state["admin_upload_results"] = result
            st.session_state["admin_documents_cache"] = None
            st.session_state["admin_upload_message"] = "Upload completed."
            st.session_state["admin_upload_message_kind"] = "success"
        except Exception as exc:
            st.session_state["admin_upload_message"] = str(exc)
            st.session_state["admin_upload_message_kind"] = "error"

with upload_col3:
    if st.button("Clear Selected Files", use_container_width=True):
        current_key = st.session_state["admin_file_uploader_key"]
        prefix, _, suffix = current_key.rpartition("_")
        next_index = int(suffix) + 1 if suffix.isdigit() else 1
        st.session_state["admin_file_uploader_key"] = f"{prefix}_{next_index}" if prefix else f"admin_file_uploader_{next_index}"
        st.session_state["admin_validation_results"] = None
        st.session_state["admin_upload_results"] = None
        st.session_state["admin_upload_message"] = "Selected files cleared."
        st.session_state["admin_upload_message_kind"] = "info"
        st.rerun()

upload_message = st.session_state.get("admin_upload_message")
if upload_message:
    kind = st.session_state.get("admin_upload_message_kind", "info")
    if kind == "success":
        st.success(upload_message)
    elif kind == "error":
        st.error(upload_message)
    else:
        st.info(upload_message)

validation_results = st.session_state.get("admin_validation_results")
if validation_results:
    validation_files = validation_results.get("files", []) or []
    valid_count = sum(1 for item in validation_files if item.get("valid"))
    invalid_count = len(validation_files) - valid_count
    duplicate_count = sum(1 for item in validation_files if item.get("duplicate"))
    st.markdown(
        f"Validation results: {valid_count} valid, {invalid_count} invalid, {duplicate_count} duplicates"
    )
    with st.expander("View validation details"):
        st.dataframe(validation_files, use_container_width=True)

upload_results = st.session_state.get("admin_upload_results")
if upload_results:
    uploaded_count = int(upload_results.get("uploaded_count") or 0)
    rejected_count = int(upload_results.get("rejected_count") or 0)
    duplicate_count = int(upload_results.get("duplicate_count") or 0)
    st.markdown(
        f"Upload results: {uploaded_count} files uploaded successfully, "
        f"{rejected_count} rejected, {duplicate_count} duplicates"
    )
    with st.expander("View upload details"):
        st.json(upload_results)

st.divider()
st.subheader("Step 3: Start Ingestion")
st.caption("Select uploaded or failed documents and queue a background ingestion job.")
docs: List[Dict[str, Any]] = []
documents_error = None
try:
    docs = _load_documents(force=True)
except Exception as exc:
    documents_error = _friendly_exception_message(exc)

if documents_error:
    st.warning(documents_error)

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
st.subheader("Step 4: Monitor Job Status")
st.caption("Track the latest job started from this page and watch progress until completion.")
last_job_id = st.session_state.get("admin_last_job_id")
if last_job_id:
    try:
        job_payload = _load_job(int(last_job_id))
        job = job_payload.get("job", {})
        total = int(job.get("total_files") or 0)
        processed = int(job.get("processed_files") or 0)
        progress = (processed / total) if total else 0.0
        st.markdown(
            f"Last job: #{job.get('id')} {_status_badge_html(job.get('status'))}",
            unsafe_allow_html=True,
        )
        st.progress(progress)
        successful = int(job.get("successful_files") or 0)
        failed = int(job.get("failed_files") or 0)
        st.markdown(
            f"Processed {processed}/{total} files, {successful} successful, {failed} failed"
        )
        if job.get("message"):
            st.caption(str(job.get("message")))
        st.caption(
            f"Started: {_format_display_time(job.get('started_at'))} | "
            f"Finished: {_format_display_time(job.get('finished_at'))}"
        )
        with st.expander("View job details"):
            st.write(
                {
                    "processed_files": processed,
                    "total_files": total,
                    "successful_files": successful,
                    "failed_files": failed,
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
st.subheader("Step 5: Manage Documents")
st.caption("Review stored documents, check status, re-ingest when needed, or permanently delete files.")
if documents_error:
    st.info("Document management will be available after the admin backend becomes reachable.")
elif not docs:
    st.info("No uploaded documents found.")
else:
    _render_document_table(docs)

    st.markdown("Selected document actions")
    doc_options = { _document_label(doc): doc for doc in docs }
    selected_doc_label = st.selectbox(
        "Choose a document to manage",
        options=list(doc_options.keys()),
        key="admin_selected_document",
    )
    selected_doc = doc_options[selected_doc_label]
    selected_doc_id = selected_doc["id"]

    if st.session_state.get("admin_delete_confirm_id") not in {None, selected_doc_id}:
        st.session_state["admin_delete_confirm_id"] = None

    st.write(f"#{selected_doc['id']} | {selected_doc['filename']} | size={_human_size(selected_doc.get('file_size_bytes'))}")
    st.markdown(f"Status: {_status_badge_html(selected_doc.get('status'))}", unsafe_allow_html=True)
    st.caption(
        f"Created: {_format_display_time(selected_doc.get('created_at'))} | "
        f"Updated: {_format_display_time(selected_doc.get('updated_at'))} | "
        f"Last ingested: {_format_display_time(selected_doc.get('last_ingested_at'))}"
    )

    action_col1, action_col2 = st.columns(2)
    can_reingest = selected_doc.get("status") in {"indexed", "failed", "uploaded"}
    with action_col1:
        if st.button("Re-ingest Selected Document", use_container_width=True, disabled=not can_reingest):
            try:
                payload = _admin_request("POST", f"/admin/documents/{selected_doc['id']}/reingest")
                job = payload.get("job", {})
                st.session_state["admin_last_job_id"] = job.get("id")
                st.session_state["admin_documents_cache"] = None
                st.success(f"Queued re-ingest job #{job.get('id')} for document #{selected_doc['id']}.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    with action_col2:
        if st.session_state.get("admin_delete_confirm_id") == selected_doc_id:
            st.warning(
                "This permanently deletes the uploaded file and indexed chunks. "
                "To restore it later, you must upload it again."
            )
            st.caption(f"Ready to delete: {selected_doc['filename']}")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Confirm Permanent Delete", use_container_width=True, key=f"confirm_delete_{selected_doc_id}"):
                    try:
                        _admin_request("DELETE", f"/admin/documents/{selected_doc_id}")
                        st.session_state["admin_documents_cache"] = None
                        st.session_state["admin_delete_confirm_id"] = None
                        st.success(f"Deleted document #{selected_doc_id}.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
            with cancel_col:
                if st.button("Cancel Delete", use_container_width=True, key=f"cancel_delete_{selected_doc_id}"):
                    st.session_state["admin_delete_confirm_id"] = None
                    st.rerun()
        else:
            if st.button("Delete Selected Document", use_container_width=True):
                st.session_state["admin_delete_confirm_id"] = selected_doc_id
                st.rerun()
