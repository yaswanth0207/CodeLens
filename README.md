## CodeLens

RAG-powered web app for exploring and documenting codebases. Users can upload code files or point the app at a GitHub repository, then:

- Ask questions about the code (“What does the RAGAnswerer class do?”)
- Explore imports and call relationships via an interactive dependency graph
- Auto-generate docstrings for functions and classes

The project lives at `https://github.com/yaswanth0207/CodeLens`.

### Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **RAG**: LlamaIndex + ChromaDB + `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: ChromaDB (local persistent store)
- **LLM**: Ollama (local) with `llama3.2:3b` (or `llama3`, configurable)
- **Parsing**: Tree-sitter (Python / JS / TS)
- **Graphs**: `networkx` + `pyvis`

For a deeper architectural view, see `ARCHITECTURE.md`.

---

### 1. Prerequisites

- Python 3.10+ installed and on your `PATH`.
- [Ollama](https://ollama.com) installed and running locally.
-- Git (for cloning GitHub repos from inside the app).

Pull at least one supported model (for example `llama3.2:3b`):

```bash
ollama pull llama3.2:3b
# or a smaller variant, e.g.:
# ollama pull llama3
```

You can also pull `llama3` and switch via `.env`.

---

### 2. Installation

From the project root (`CodeLens/`):

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
- **`OLLAMA_MODEL`**: `llama3.2:3b` (or any other locally available model, e.g. `llama3`).
- **`CHROMA_PERSIST_DIR`**: directory for ChromaDB data (default `./chroma_db`).
- **`MAX_CHUNKS_RETURNED`**: number of context chunks to send to the LLM.
- **`SUPPORTED_EXTENSIONS`**: file types to parse and index.
- **`SIMILARITY_THRESHOLD`**: optional lower bound on raw similarity scores. For code with Chroma, the default of `0` disables filtering so hybrid ranking always has candidates.
- **`MIN_BEST_SCORE`**: optional gate on the best semantic score. For code, the default of `0` disables this check.

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

#### Sidebar

- Enter a **GitHub URL** _or_ upload one or more code files (`.py`, `.js`, `.ts`, `.jsx`, `.tsx`).
- Click **Index Codebase** to build a fresh index.
- See indexed files listed with language icons.
- Use **Clear Index** to wipe the current Chroma index and start over.

#### Ask Questions tab

- Choose a mode:
  - **Explain** — Understand what code does.
  - **Find** — Locate patterns across files.
  - **Generate Docs** — Create docstrings.
  - **Summarize** — High-level module overview.
- Ask questions like:
  - “What does the RAGAnswerer class do?”
  - “Find all database queries.”
  - “Summarize this ingestor.py.”
- The backend uses **hybrid retrieval** (semantic + keyword) over indexed chunks and returns:
  - An answer from the local LLM, constrained to the indexed code.
  - A list of **sources** with `file:start_line–end_line` and a match percentage.
- Use **Clear Chat** to reset the conversation.

#### Dependency Graph tab

- Select a file from the indexed list.
- Click **Show Graph** to render an interactive pyvis graph:
  - Nodes for modules/functions.
  - Edges for imports and key relationships.

#### Auto-Generate Docs tab

- Select a file from the indexed list.
- The left pane shows the **original file** (fetched from the backend).
- Click **Generate Docstrings for All Functions**:
  - The backend runs a `docstring`-mode query over the current file’s chunks.
  - The right pane shows suggested docstrings in a scrollable view.
- Download the suggestions as a text file if you want to apply them manually.

---

### 7. Demo flow (end-to-end)

1. Start **Ollama**, the **FastAPI backend**, and the **Streamlit frontend** as described above.
2. In the Streamlit sidebar, paste a GitHub repo URL (e.g. `https://github.com/some/repo`) or upload files.
3. Click **Index Codebase** and wait for indexing to complete (sidebar shows file and chunk counts).
4. In **Ask Questions**, ask _“What does the RAGAnswerer class do?”_ and see an explanation plus file/line citations.
5. Ask _“Find all database queries”_ to list DB-related calls with precise locations.
6. Ask _“Summarize this ingestor.py”_ to get a high-level overview.
7. Open **Dependency Graph** to inspect module and call relationships.
8. Open **Auto-Generate Docs** to generate docstring suggestions and download them.

---

### 8. Troubleshooting

- **Ollama errors or timeouts**
  - Ensure `ollama serve` is running.
  - Confirm the model in `OLLAMA_MODEL` is pulled (`ollama pull codellama`).
  - Check `OLLAMA_BASE_URL` matches your Ollama host/port.

-- **No results / “Not found in indexed codebase”**
  - Confirm you have successfully indexed a repo or files (sidebar shows indexed files).
  - Make sure your question refers to functions or patterns that actually exist in the indexed code.
  - For general questions (“Explain HTTP”), the system will intentionally return *Not found in indexed codebase* if the concept is not present in the code.

- **ChromaDB issues**
  - If you see index/collection errors, stop the backend, delete the `CHROMA_PERSIST_DIR` directory (e.g. `rm -rf chroma_db`), and restart.
  - If retrieval logs show zero nodes for queries, ensure you have re-indexed after clearing the index.

