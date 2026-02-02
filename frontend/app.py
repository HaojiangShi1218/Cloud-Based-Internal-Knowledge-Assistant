import os
import requests
import streamlit as st

# --- Config ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Knowledge Assistant", page_icon="🔎", layout="centered")

st.title("Cloud Knowledge Assistant")
st.caption("Ask questions over your internal docs. Answers include citations for traceability.")

with st.sidebar:
    st.header("Settings")
    mode_label = st.radio("Mode", ["Evidence Mode", "Synthesis Mode"], horizontal=True)
    mode = "extract" if mode_label == "Evidence Mode" else "llm"
    top_k = st.slider("Top K (citations)", min_value=1, max_value=10, value=5, step=1)
    st.text_input("Backend URL", value=BACKEND_URL, key="backend_url")
    st.divider()
    st.markdown(
        "- **Evidence Mode**: returns evidence-focused answer\n"
        "- **Synthesis Mode**: synthesizes answer with AI\n"
    )

question = st.text_area(
    "Your question",
    placeholder="",
    height=90,
)

col1, col2 = st.columns([1, 1])
with col1:
    ask_btn = st.button("Ask", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.clear()
    st.rerun()

def call_backend(q: str, mode: str, top_k: int):
    payload = {"question": q, "mode": mode, "top_k": int(top_k)}
    url = st.session_state.get("backend_url", BACKEND_URL).rstrip("/") + "/ask"
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
            data = call_backend(q, mode=mode, top_k=top_k)
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
            score = c.get("score", None)
            chunk = c.get("chunk_index", None)

            label_bits = [f"#{rank}", src]
            if page is not None:
                label_bits.append(f"p.{page}")
            if chunk is not None:
                label_bits.append(f"chunk {chunk}")
            if score is not None:
                label_bits.append(f"score {score:.3f}" if isinstance(score, (int, float)) else f"score {score}")

            with st.expander(" • ".join(label_bits)):
                st.json(c)

    st.divider()
    st.caption("Tip: if citations look off, reduce Top K or tighten chunking/ingestion settings.")
