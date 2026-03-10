## Architecture Overview

CodeLens is a small, single‑service RAG application for exploring and documenting source code. It has:

- **Frontend**: a Streamlit UI (`frontend/app.py`)
- **Backend API**: a FastAPI app (`backend/main.py`)
- **RAG pipeline**: syntax‑aware indexing + hybrid retrieval + local LLM via Ollama
- **Storage**: local ChromaDB for vectors, local disk for uploaded/cloned repos

All components run on a single machine; there are no background workers or separate services.

---

## High‑Level Flow

1. **User selects codebase**
   - Uploads files through the sidebar, or
   - Provides a GitHub URL to clone.
2. **Backend indexes code**
   - Files are saved under `uploads/` (for uploads) or a cloned repo root (for GitHub).
   - `CodebaseIndexer` parses files with Tree‑sitter and builds chunks around functions/classes.
   - Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and written to ChromaDB via LlamaIndex.
3. **User queries**
   - The Ask Questions tab sends `question` + `mode` to `POST /query`.
4. **Hybrid retrieval**
   - `HybridCodeRetriever` runs vector search on Chroma (semantic similarity).
   - It computes a lightweight keyword score per node (question tokens vs code + filename + symbol).
   - Scores are combined (semantic + keyword) and the top‑k chunks are returned.
5. **LLM generation (Ollama)**
   - `OllamaGenerator` builds a system + user prompt from the retrieved chunks and query mode.
   - It calls `POST {OLLAMA_BASE_URL}/api/chat` with the configured local model (e.g. `llama3.2:3b`).
   - The response text is returned to the frontend along with the source chunks (file + line ranges).
6. **Frontend rendering**
   - Ask Questions shows the answer and expandable source citations.
   - Dependency Graph tab renders a pyvis HTML graph for the selected file.
   - Auto‑Generate Docs tab shows the original file on the left and generated docstrings on the right.

---

## Backend Components

### FastAPI application (`backend/main.py`)

- Configures CORS and loads environment via `python-dotenv`.
- Constructs long‑lived singletons:
  - `CodebaseIndexer` – controls Chroma client and vector index.
  - `HybridCodeRetriever` – wraps the LlamaIndex `VectorStoreIndex` for hybrid retrieval.
  - `OllamaGenerator` – talks to the local Ollama HTTP API.
  - `DependencyGraph` (optional) – built lazily when a codebase is indexed.
- Restores previously indexed files from Chroma on startup (`get_indexed_file_paths()`).

**Key endpoints**

- `POST /upload`
  - Accepts multiple `UploadFile`s.
  - Clears the `uploads/` directory and any existing index.
  - Persists files, filters by extension, and calls `indexer.index_codebase(UPLOAD_DIR)`.
  - Rebuilds the dependency graph over the newly indexed files.
- `POST /github`
  - Clones a remote repo (via `load_github_repo`), filters supported files, then indexes that root.
  - Rebuilds the dependency graph over the repo root.
- `POST /query`
  - Accepts `question` and `mode` (`explain`, `find`, `docstring`, `summarize`).
  - Optionally rewrites summarize‑style questions into a more retrieval‑friendly form.
  - Uses `HybridCodeRetriever` to get `RetrievedChunk`s from Chroma.
  - Enforces only a minimal gate on “no chunks at all”; otherwise passes chunks through.
  - Calls `OllamaGenerator.generate(...)` and returns `{ answer, sources }`.
- `GET /graph/{file_path}`
  - Delegates to `DependencyGraph` to render an HTML graph for a single file.
- `GET /files`
  - Returns the list of indexed file paths (used to populate dropdowns in the UI).
- `GET /file_content`
  - Returns raw file contents for an indexed path (used by Auto‑Generate Docs “Before” pane).
- `DELETE /index`
  - Clears the Chroma collection and in‑memory metadata and resets the dependency graph.

### Indexing layer (`backend/rag/indexer.py`)

Responsible for turning a file tree into a Chroma‑backed LlamaIndex:

- Uses `chromadb.PersistentClient` with `CHROMA_PERSIST_DIR`.
- Wraps the collection with `ChromaVectorStore` and `StorageContext`.
- Uses `SentenceTransformer` + `HuggingFaceEmbedding` for embeddings.
- Holds a `SyntaxAwareChunker` that:
  - Parses Python / JS / TS using Tree‑sitter.
  - Produces `ChunkWithMetadata` objects (code, file path, symbol, language, line ranges, docstring).
- Converts chunks to LlamaIndex `Document`s with stable `doc_id`s based on metadata.
- Offers:
  - `index_codebase(root_path)` – index all supported files under a directory.
  - `add_file(path)` – incrementally add a single file.
  - `clear_index()` – drop and recreate the Chroma collection.
  - `get_indexed_file_paths()` – discover which files are currently in the vector store.

On startup, if the Chroma collection already has documents, it rebuilds the `VectorStoreIndex`
from the vector store so the RAG pipeline survives backend restarts.

### Hybrid retrieval (`backend/rag/retriever.py`)

`HybridCodeRetriever` combines semantic and lexical signals:

- Wraps the `VectorStoreIndex` from the indexer and exposes a `query(question, top_k)` API.
- Uses `index.as_retriever(similarity_top_k=...)` to pull a generous set of candidates.
- Optionally filters by raw similarity (configurable via `SIMILARITY_THRESHOLD`, default `0`).
- If filtering would drop everything, it falls back to the top‑k retrieved nodes.
- Computes a simple keyword score per node:
  - Tokenises the question into meaningful identifiers.
  - Counts occurrences across node text, filename, and symbol name.
- Normalises semantic and keyword scores and combines them into a final rank.
- Returns a list of `RetrievedChunk` objects with:
  - `code`, `file`, `symbol`, `language`, `start_line`, `end_line`,
  - `score` (combined rank) and `semantic_score` (raw vector similarity),
  - serialisable metadata for the API and UI.

This hybrid design improves robustness to phrasing (“what does the RAGAnswerer class do?”) while still
benefiting from semantic similarity.

### Generation (`backend/rag/generator.py`)

`OllamaGenerator` is a thin wrapper over Ollama’s chat API:

- Encodes the current mode via a `QueryMode` enum:
  - `explain`, `find`, `docstring`, `summarize`.
- Builds a **system prompt** that strictly limits the model to the provided snippets and enforces
  “Not found in indexed codebase.” for out‑of‑scope questions.
- Builds a **user prompt** that:
  - Includes the original natural‑language question.
  - Serialises each `RetrievedChunk` as a “Snippet N” block with file and line range.
  - Adds mode‑specific instructions (e.g. explain behaviour vs list matches vs generate docstrings).
- Normalises `OLLAMA_BASE_URL` and constructs the `/api/chat` URL robustly.
- Supports non‑streaming (current UI) and streaming responses.

The generator does not know how retrieval works; it only consumes `RetrievedChunk`s and the selected mode.

---

## Frontend (Streamlit)

The Streamlit app in `frontend/app.py` is a thin UI over the REST API.

### Layout

- **Sidebar**
  - GitHub URL input + code file upload (`Upload code files`).
  - **Index Codebase** button to call `/upload` or `/github`.
  - **Indexed Files** list (filename only, with full path in the backend).
  - **Clear Index** button to call `DELETE /index`.
- **Tabs**
  - **Ask Questions**
    - Mode selector with descriptions (Explain, Find, Generate Docs, Summarize).
    - **Clear Chat** button (resets in‑memory conversation).
    - Chat history (`You` / `Assistant`) with warnings when nothing relevant is found.
    - Expandable **Sources** section listing file:line ranges and match percentages.
  - **Dependency Graph**
    - File dropdown bound to `/files`.
    - **Show Graph** button that calls `/graph/{file}` and embeds the returned HTML.
  - **Auto‑Generate Docs**
    - File dropdown bound to `/files`.
    - Calls `/file_content` to show the original file (“Before” pane).
    - Calls `/query` in `docstring` mode to generate suggested docstrings (“After” pane).
    - Download button for saving suggested docstrings as a text file.

The frontend is intentionally kept stateless beyond `st.session_state` to make it easy to run
locally without extra infrastructure.

---

## Configuration & Environment

Configuration is driven by `.env` / environment variables:

- `OLLAMA_BASE_URL` – Ollama server URL (e.g. `http://localhost:11434`).
- `OLLAMA_MODEL` – model name/tag (e.g. `llama3.2:3b`, `codellama`).
- `CHROMA_PERSIST_DIR` – directory for ChromaDB data (`./chroma_db` by default).
- `MAX_CHUNKS_RETURNED` – default `top_k` for retrieval.
- `SIMILARITY_THRESHOLD` – optional similarity gate for nodes (default `0` = disabled).
- `SUPPORTED_EXTENSIONS` – comma‑separated list of file extensions to index.

All services are local‑only and can be run via:

- `uvicorn backend.main:app --reload --port 8000`
- `streamlit run frontend/app.py`

Ollama must be running (`ollama serve`) with the configured model pulled.

---

## Future Extensions

Potential architectural extensions that fit this design:

- **Language coverage**: add more Tree‑sitter grammars and chunking rules.
- **Search UX**: dedicated “Search” tab that surfaces hybrid retrieval results explicitly.
- **Security & quality analysis**: add static‑analysis passes that feed into the same RAG flow.
- **Multi‑user persistence**: move uploads and Chroma to shared storage if deployed as a service.

