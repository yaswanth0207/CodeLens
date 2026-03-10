from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import requests
import streamlit as st


API_BASE = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")


def _api_url(path: str) -> str:
    return f"{API_BASE.rstrip('/')}{path}"


def _file_display_name(full_path: str) -> str:
    """Show only filename in UI; full path is still the value."""
    return Path(full_path).name


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {"role": "user"/"assistant", "content": str, "sources": [...]}
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "chunks_indexed" not in st.session_state:
        st.session_state.chunks_indexed = 0


def sidebar_controls() -> None:
    st.sidebar.header("Codebase")

    github_url = st.sidebar.text_input("GitHub URL")
    uploaded_files = st.sidebar.file_uploader(
        "Upload code files",
        type=["py", "js", "ts", "jsx", "tsx"],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Index Codebase", use_container_width=True):
        if uploaded_files:
            # Prefer file upload when files are present (even if GitHub URL is filled).
            with st.spinner("Uploading and indexing files..."):
                files_param = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files]
                resp = requests.post(_api_url("/upload"), files=files_param)
        elif github_url and github_url.strip():
            with st.spinner("Cloning and indexing repository..."):
                resp = requests.post(_api_url("/github"), json={"url": github_url.strip()})
        else:
            st.sidebar.error("Provide a GitHub URL or upload code files.")
            return

        if resp.status_code != 200:
            st.sidebar.error(f"Indexing failed: {resp.text}")
        else:
            data = resp.json()
            st.session_state.indexed_files = data.get("files", [])
            st.session_state.chunks_indexed = data.get("chunks_indexed", 0)
            st.sidebar.success(f"Indexed {len(st.session_state.indexed_files)} files, {st.session_state.chunks_indexed} chunks.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Indexed Files")
    if st.session_state.indexed_files:
        for f in st.session_state.indexed_files:
            icon = "🐍" if f.endswith(".py") else "📄"
            st.sidebar.markdown(f"{icon} `{Path(f).name}`")
        if st.sidebar.button("Clear Index", use_container_width=True, type="secondary"):
            resp = requests.delete(_api_url("/index"))
            if resp.status_code == 200:
                st.session_state.indexed_files = []
                st.session_state.chunks_indexed = 0
                st.sidebar.success("Index cleared.")
                st.rerun()
            else:
                st.sidebar.error(f"Failed to clear index: {resp.text}")
    else:
        st.sidebar.caption("No files indexed yet.")


MODE_DESCRIPTIONS = {
    "Explain": "Understand what code does",
    "Find": "Locate patterns across files",
    "Generate Docs": "Create docstrings",
    "Summarize": "High-level module overview",
}


def tab_ask_questions() -> None:
    st.subheader("Ask Questions About Your Code")

    mode_map = {
        "Explain": "explain",
        "Find": "find",
        "Generate Docs": "docstring",
        "Summarize": "summarize",
    }
    mode_label = st.selectbox(
        "Mode",
        list(mode_map.keys()),
        format_func=lambda k: f"{k} — {MODE_DESCRIPTIONS.get(k, '')}",
    )
    mode = mode_map[mode_label]

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    chat_container = st.container()

    NO_RESULTS_KEY = "couldn't find anything relevant"

    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                content = msg["content"] or ""
                if NO_RESULTS_KEY in content.lower():
                    st.warning(content)
                else:
                    st.markdown(f"**Assistant:** {content}")
                sources = msg.get("sources") or []
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            # Prefer raw semantic score for match % when available
                            raw = s.get("semantic_score")
                            pct = (raw if raw is not None else s.get("score", 0) or 0) * 100
                            st.markdown(
                                f"- `{s['file']}`:{s['start_line']}-{s['end_line']} "
                                f"**{pct:.0f}%** match"
                            )

    st.caption("Try: \"What does ingestor.py do?\", \"Summarize this ingestor.py\", or \"Find all API calls\"")
    question = st.text_input(
        "Ask a question about the codebase",
        placeholder="e.g. What does the retrieval function do?",
    )
    if st.button("Send", use_container_width=True) and question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Querying backend..."):
            resp = requests.post(
                _api_url("/query"),
                json={"question": question, "mode": mode},
                stream=False,
            )
        if resp.status_code != 200:
            answer = f"Error from backend: {resp.text}"
            sources: List[Dict] = []
        else:
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()


def tab_dependency_graph() -> None:
    st.subheader("Dependency Graph")
    if not st.session_state.indexed_files:
        st.info("Index a codebase first.")
        return

    file_choice = st.selectbox(
        "File",
        st.session_state.indexed_files,
        format_func=_file_display_name,
        help="Full path of the selected file is shown below.",
    )
    st.caption(f"Full path: `{file_choice}`")
    if st.button("Show Graph"):
        with st.spinner("Rendering dependency graph..."):
            resp = requests.get(_api_url(f"/graph/{file_choice}"))
        if resp.status_code != 200:
            st.error(f"Failed to fetch graph: {resp.text}")
        else:
            html = resp.text
            st.components.v1.html(html, height=600, scrolling=True)


def tab_auto_generate_docs() -> None:
    st.subheader("Auto-Generate Docstrings")
    if not st.session_state.indexed_files:
        st.info("Index a codebase first.")
        return

    file_choice = st.selectbox(
        "File",
        st.session_state.indexed_files,
        key="docs_file",
        format_func=_file_display_name,
        help="Full path of the selected file is shown below.",
    )
    st.caption(f"Full path: `{file_choice}`")

    # Load "Before" content from backend so we show actual file, not a placeholder.
    before_content: str | None = None
    try:
        r = requests.get(_api_url("/file_content"), params={"path": file_choice}, timeout=10)
        if r.status_code == 200:
            before_content = r.json().get("content", "")
    except Exception:
        before_content = None

    if st.button("Generate Docstrings for All Functions"):
        # Use the query API in 'docstring' mode to ask for docstrings for the whole file.
        question = f"Generate detailed docstrings for all functions and methods in {file_choice}."
        with st.spinner("Generating docstrings..."):
            resp = requests.post(
                _api_url("/query"),
                json={"question": question, "mode": "docstring"},
            )
        if resp.status_code != 200:
            st.error(f"Docstring generation failed: {resp.text}")
            return

        data = resp.json()
        answer = data.get("answer", "")

        before = before_content if before_content is not None else f"# Original file: {file_choice}\n\n(File content not available for display.)"
        after = answer or "(No docstrings generated.)"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Before")
            st.code(before, language="python")
        with col2:
            st.markdown("#### After (Suggested)")
            # Use text_area for long docstrings so the full content is scrollable and visible.
            st.text_area(
                "Generated docstrings (scroll to see full content)",
                value=after,
                height=400,
                disabled=True,
                label_visibility="collapsed",
            )

        st.download_button(
            "Download Suggested Docstrings",
            data=after.encode("utf-8"),
            file_name=f"{Path(file_choice).stem}_docstrings.txt",
            mime="text/plain",
        )


def main() -> None:
    st.set_page_config(
        page_title="CodeLens",
        layout="wide",
        page_icon="📚",
    )

    st.title("CodeLens")
    st.caption("RAG-powered assistant for exploring and documenting codebases.")

    _init_session_state()
    sidebar_controls()

    tab1, tab2, tab3 = st.tabs(["Ask Questions", "Dependency Graph", "Auto-Generate Docs"])

    with tab1:
        tab_ask_questions()
    with tab2:
        tab_dependency_graph()
    with tab3:
        tab_auto_generate_docs()


if __name__ == "__main__":
    main()

