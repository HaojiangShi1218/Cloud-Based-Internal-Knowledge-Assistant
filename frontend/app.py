import os
import requests
import streamlit as st

# --- Config ---
# BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = os.getenv("API_URL", "http://api:8000")
DEBUG_CACHE_CLEAR_TOKEN = os.getenv("DEBUG_CACHE_CLEAR_TOKEN", "")

if "api_url" not in st.session_state:
    st.session_state["api_url"] = API_URL

st.set_page_config(page_title="Knowledge Assistant", page_icon="🔎", layout="centered")

st.title("Cloud Knowledge Assistant")
st.caption("Ask questions over your internal docs. Answers include citations for traceability.")
st.caption("Admin document management is available from the Streamlit page navigation.")

with st.sidebar:
    st.header("Settings")
    mode_label = st.radio("Mode", ["Evidence Mode", "Synthesis Mode"], horizontal=True)
    mode = "extract" if mode_label == "Evidence Mode" else "llm"
    # st.divider()
    st.markdown(
        "- **Evidence Mode**: returns evidence-focused answer\n"
        "- **Synthesis Mode**: synthesizes answer with AI\n"
    )
    top_k = st.slider("Top K (citations)", min_value=1, max_value=10, value=5, step=1)
    query_rewrite_enabled = st.checkbox("Query-rewrite", value=True)
    st.caption(
        "Query-rewrite generates a few alternative versions of your question to help retrieval. "
        "It can improve answers to implicit or tricky questions."
    )
    st.text_input("Backend URL", value=st.session_state.get("api_url", API_URL), key="api_url")
    if st.button("Clear LLM Cache", use_container_width=True):
        try:
            base = st.session_state.get("api_url", API_URL).rstrip("/")
            headers = {}
            if DEBUG_CACHE_CLEAR_TOKEN:
                headers["x-debug-token"] = DEBUG_CACHE_CLEAR_TOKEN
            resp = requests.post(f"{base}/debug/cache/clear", headers=headers, timeout=10)
            if resp.status_code == 200:
                cleared = resp.json().get("cleared", 0)
                st.success(f"Cleared {cleared} cached items.")
            else:
                st.error(f"{resp.status_code} {resp.reason}: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Backend request failed: {e}")

# Apply clear before widget instantiation.
if st.session_state.get("clear_question"):
    st.session_state["question_input"] = ""
    st.session_state["clear_question"] = False

question = st.text_area(
    "Your question",
    placeholder="",
    height=90,
    key="question_input",
)

col1, col2 = st.columns([1, 1])
with col1:
    ask_btn = st.button("Ask", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state["clear_question"] = True
    st.rerun()

def call_backend(q: str, mode: str, top_k: int, query_rewrite_enabled: bool):
    payload = {
        "question": q,
        "mode": mode,
        "top-k": int(top_k),
        "query-rewrite-enabled": bool(query_rewrite_enabled),
    }
    url = st.session_state.get("api_url", API_URL).rstrip("/") + "/ask"
    r = requests.post(url, json=payload, timeout=60)

    if r.status_code != 200:
        # Show FastAPI's detailed validation error
        raise RuntimeError(f"{r.status_code} {r.reason}: {r.text}")

    return r.json()


if ask_btn:
    q = (question or "").strip()
    if not q:
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching docs and generating answer..."):
        try:
            data = call_backend(q, mode=mode, top_k=top_k, query_rewrite_enabled=query_rewrite_enabled)
        except requests.exceptions.RequestException as e:
            st.error(f"Backend request failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(data.get("answer", ""))

    citations = data.get("citations", []) or []
    st.subheader(f"Citations ({len(citations)})")

    if not citations:
        st.info("No citations returned.")
    else:
        # show quick summary list
        for c in citations:
            rank = c.get("rank", "?")
            src = c.get("source", "Unknown")
            page = c.get("page_num", None)
            semantic_score = c.get("semantic_score", None)
            final_score = c.get("final_score", None)
            chunk = c.get("chunk_index", None)

            label_bits = [f"#{rank}", src]
            if page is not None:
                label_bits.append(f"p.{page}")
            if chunk is not None:
                label_bits.append(f"chunk {chunk}")
            if final_score is not None:
                label_bits.append(
                    f"final {final_score:.3f}" if isinstance(final_score, (int, float)) else f"final {final_score}"
                )
            if semantic_score is not None:
                label_bits.append(
                    f"semantic {semantic_score:.3f}"
                    if isinstance(semantic_score, (int, float))
                    else f"semantic {semantic_score}"
                )

            with st.expander(" • ".join(label_bits)):
                st.json(c)

    st.divider()
    st.caption(
        "Tips: Top‑K controls how many citations are considered. Lower values tighten results "
        "but can miss key passages. Higher values give the model more to choose from "
        "(often helpful for implicit questions), but may add noise."
    )
