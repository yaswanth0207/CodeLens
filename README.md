## CodeLens

RAG-powered web app for exploring and documenting codebases. Users can upload code files or point the app at a GitHub repository, then ask questions, generate documentation, and visualise dependencies.

### Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **RAG**: LlamaIndex + ChromaDB + `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: ChromaDB (local persistent store)
- **LLM**: Ollama (local) with `codellama` or `llama3`
- **Parsing**: Tree-sitter (Python / JS / TS)
- **Graphs**: `networkx` + `pyvis`

---

### 1. Prerequisites

- Python 3.10+ installed and on your `PATH`.
- [Ollama](https://ollama.com) installed and running locally.
- Git (for cloning GitHub repos from inside the app).

Pull at least one supported model (for example `codellama`):

```bash
ollama pull codellama
```

You can also pull `llama3` and switch via `.env`.

---

### 2. Installation

From the project root (`code-doc-assistant/`):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Configuration

Copy the example environment file and adjust values if needed:

```bash
cp .env.example .env
```

Key settings:

- **`OLLAMA_BASE_URL`**: typically `http://localhost:11434`.
- **`OLLAMA_MODEL`**: `codellama` or `llama3`.
- **`CHROMA_PERSIST_DIR`**: directory for ChromaDB data (default `./chroma_db`).
- **`MAX_CHUNKS_RETURNED`**: number of context chunks to send to the LLM.
- **`SUPPORTED_EXTENSIONS`**: file types to parse and index.

---

### 4. Run the backend (FastAPI)

From the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

This starts the REST API consumed by the Streamlit frontend at `http://localhost:8000`.

If you run on a different port, set `BACKEND_BASE_URL` in your shell or `.env` (for the frontend) accordingly, for example:

```bash
export BACKEND_BASE_URL=http://localhost:8000
```

---

### 5. Run the frontend (Streamlit)

In a **second terminal**, keep your virtualenv activated and run:

```bash
streamlit run frontend/app.py
```

This opens the UI in your browser (by default `http://localhost:8501`).

---

### 6. Using the app

- **Sidebar**
  - Enter a **GitHub URL** _or_ upload one or more code files (`.py`, `.js`, `.ts`, `.jsx`, `.tsx`).
  - Click **Index Codebase**.
  - Watch the status and list of indexed files (with language icons).

- **Ask Questions tab**
  - Choose a mode: **Explain**, **Find**, **Generate Docs**, or **Summarize**.
  - Ask questions like:
    - “What does the retrieval function do?”
    - “Find all database queries.”
  - Answers always include source citations (`file:line-start–line-end`), based only on indexed code.

- **Dependency Graph tab**
  - Select a file and click **Show Graph**.
  - Explore imports and call relationships in an interactive pyvis graph.

- **Auto-Generate Docs tab**
  - Select a file and click **Generate Docstrings for All Functions**.
  - View suggested docstrings and download them as a text file.

---

### 7. Demo flow (end-to-end)

1. Start **Ollama**, the **FastAPI backend**, and the **Streamlit frontend** as described above.
2. In the Streamlit sidebar, paste a GitHub repo URL (e.g. `https://github.com/some/repo`) or upload files.
3. Click **Index Codebase** and wait for indexing to complete (sidebar shows file and chunk counts).
4. In **Ask Questions**, ask _“What does the retrieval function do?”_ and see an explanation plus file/line citations.
5. Ask _“Find all database queries”_ to list DB-related calls with precise locations.
6. Open **Dependency Graph** to inspect module and call relationships.
7. Open **Auto-Generate Docs** to generate docstring suggestions and download them.

---

### 8. Troubleshooting

- **Ollama errors or timeouts**
  - Ensure `ollama serve` is running.
  - Confirm the model in `OLLAMA_MODEL` is pulled (`ollama pull codellama`).
  - Check `OLLAMA_BASE_URL` matches your Ollama host/port.

- **No results / “Not found in indexed codebase”**
  - Confirm you have successfully indexed a repo or files (sidebar shows indexed files).
  - Make sure your question refers to functions or patterns that actually exist in the indexed code.

- **ChromaDB issues**
  - If you see index/collection errors, stop the backend, delete the `CHROMA_PERSIST_DIR` directory (e.g. `rm -rf chroma_db`), and restart.

