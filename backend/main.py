from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from backend.graph.dependency_graph import DependencyGraph
from backend.rag.generator import NO_RELEVANT_CHUNKS_MESSAGE, OllamaGenerator
from backend.rag.indexer import CodebaseIndexer
from backend.rag.retriever import HybridCodeRetriever
from backend.utils.file_handler import clear_upload_dir, filter_supported_files, save_uploaded_files
from backend.utils.github_loader import load_github_repo


load_dotenv()

app = FastAPI(title="CodeLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"

indexer = CodebaseIndexer()
retriever = HybridCodeRetriever(indexer=indexer)
generator = OllamaGenerator()

_dependency_graph: Optional[DependencyGraph] = None
# Restore indexed files list from Chroma on startup (survives backend restarts)
_indexed_files: List[str] = indexer.get_indexed_file_paths()

# Restore dependency graph when indexed files are under UPLOAD_DIR (file-upload case)
_upload_paths = [p for p in _indexed_files if p.startswith(str(UPLOAD_DIR)) and Path(p).exists()]
if _upload_paths:
    try:
        _dependency_graph = DependencyGraph(root=UPLOAD_DIR)
        _dependency_graph.build([Path(p) for p in _upload_paths])
    except Exception:
        pass


class QueryRequest(BaseModel):
    question: str
    mode: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


@app.post("/upload", response_model=Dict[str, Any])
async def upload_files(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Accept code files, store them, index them, and return the indexed file list.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    filenames = [f.filename or "unnamed" for f in files]
    contents = [await f.read() for f in files]

    clear_upload_dir(UPLOAD_DIR)  # Only index the files from this upload, not previous ones
    saved_paths = save_uploaded_files(UPLOAD_DIR, filenames, contents)
    supported_paths = filter_supported_files(saved_paths)

    if not supported_paths:
        raise HTTPException(status_code=400, detail="No supported file types found in upload")

    indexer.clear_index()  # Replace previous index so results match the current codebase
    chunks = indexer.index_codebase(UPLOAD_DIR)

    global _dependency_graph, _indexed_files
    _indexed_files = sorted({str(c.metadata.file_path) for c in chunks})
    _dependency_graph = DependencyGraph(root=UPLOAD_DIR)
    _dependency_graph.build([Path(p) for p in _indexed_files])

    return {
        "files": _indexed_files,
        "chunks_indexed": len(chunks),
    }


class GithubRequest(BaseModel):
    url: str


@app.post("/github", response_model=Dict[str, Any])
async def github_repo(req: GithubRequest) -> Dict[str, Any]:
    """
    Accept a GitHub URL, clone + index the repo, and return a repo summary.
    """
    try:
        repo_root, files = load_github_repo(req.url)
    except Exception as exc:  # pragma: no cover - network/git runtime
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {exc}") from exc

    if not files:
        raise HTTPException(status_code=400, detail="No supported files found in repository")

    indexer.clear_index()  # Replace previous index so results match the current codebase
    chunks = indexer.index_codebase(repo_root)

    global _dependency_graph, _indexed_files
    _indexed_files = sorted({str(c.metadata.file_path) for c in chunks})
    _dependency_graph = DependencyGraph(root=repo_root)
    _dependency_graph.build([Path(p) for p in _indexed_files])

    return {
        "repo_root": str(repo_root),
        "file_count": len(_indexed_files),
        "chunks_indexed": len(chunks),
        "files": _indexed_files,
    }


# Disabled (0.0): Chroma uses exp(-distance) so absolute score gates filter out valid code matches.
MIN_BEST_SCORE = float(os.getenv("MIN_BEST_SCORE", "0.0"))


def _retrieval_query(question: str, mode: str) -> str:
    """
    For Summarize mode, if the user says "summarize X" or "summarize this X",
    use a retrieval-friendly phrasing so the retriever finds chunks for that file.
    """
    if mode != "summarize":
        return question
    match = re.search(
        r"summarize\s+(?:this\s+)?(?:the\s+)?(.+?)(?:\s*$|\s+please)",
        question.strip(),
        re.IGNORECASE,
    )
    if not match:
        return question
    target = match.group(1).strip()
    if not target:
        return question
    # Looks like a file or module (has extension or path, or is a single word like "ingestor")
    if target.endswith((".py", ".js", ".ts", ".jsx", ".tsx")) or "/" in target or "\\" in target:
        return f"what does {target} do, main functions and entry points"
    if re.match(r"^[\w.-]+$", target):
        return f"what does {target} do, main functions and entry points"
    return question


@app.post("/query", response_model=QueryResponse)
async def query_codebase(req: QueryRequest) -> QueryResponse:
    """
    Answer a question about the indexed codebase using RAG + Ollama.
    """
    retrieval_question = _retrieval_query(req.question, req.mode)
    try:
        chunks = retriever.query(retrieval_question)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        return QueryResponse(answer=NO_RELEVANT_CHUNKS_MESSAGE, sources=[])

    if MIN_BEST_SCORE > 0:
        best_semantic = max(c.semantic_score for c in chunks)
        if best_semantic < MIN_BEST_SCORE:
            return QueryResponse(answer=NO_RELEVANT_CHUNKS_MESSAGE, sources=[])

    sources = [
        {
            "file": c.file,
            "symbol": c.symbol,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "score": c.score,
            "semantic_score": c.semantic_score,
        }
        for c in chunks
    ]

    answer = generator.generate(req.question, req.mode, chunks, stream=False)
    return QueryResponse(answer=str(answer), sources=sources)


@app.get("/graph/{file_path:path}", response_class=HTMLResponse)
async def get_graph(file_path: str) -> HTMLResponse:
    """
    Return dependency graph HTML for a given file.
    """
    if _dependency_graph is None:
        raise HTTPException(status_code=400, detail="No dependency graph available. Index a codebase first.")

    html = _dependency_graph.get_graph(Path(file_path))
    return HTMLResponse(content=html)


@app.get("/files", response_model=List[str])
async def list_files() -> List[str]:
    """
    List all indexed files.
    """
    return _indexed_files


@app.get("/file_content")
async def get_file_content(path: str) -> Dict[str, Any]:
    """
    Return raw content of an indexed file (for "Before" in Auto-Generate Docs).
    Only allows paths that are in the current indexed files list.
    """
    if path not in _indexed_files:
        raise HTTPException(status_code=404, detail="File not in indexed files or index empty.")
    p = Path(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="File no longer exists on disk.")
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Cannot read file: {e}") from e
    return {"path": path, "content": content}


@app.delete("/index")
async def clear_index() -> JSONResponse:
    """
    Clear the ChromaDB index and in-memory metadata.
    """
    global _dependency_graph, _indexed_files
    indexer.clear_index()
    _indexed_files = []
    _dependency_graph = None
    return JSONResponse(content={"status": "ok"})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )

