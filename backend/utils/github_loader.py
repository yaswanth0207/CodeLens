from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

from git import Repo


DEFAULT_SUPPORTED_EXTS = (".py", ".js", ".ts", ".jsx", ".tsx")


def _get_supported_extensions() -> Sequence[str]:
    raw = os.getenv("SUPPORTED_EXTENSIONS")
    if not raw:
        return DEFAULT_SUPPORTED_EXTS
    return [ext.strip() for ext in raw.split(",") if ext.strip()]


def clone_github_repo(url: str) -> Path:
    """
    Clone a GitHub repository URL into a temporary directory.

    Returns the path to the cloned repository root.
    """
    tmp_dir = tempfile.mkdtemp(prefix="code_doc_repo_")
    Repo.clone_from(url, tmp_dir)
    return Path(tmp_dir)


def list_supported_files(root: Path, exts: Sequence[str] | None = None) -> List[Path]:
    """
    Recursively list all files under `root` that have supported extensions.
    """
    exts = exts or _get_supported_extensions()
    exts_lower = {e.lower() for e in exts}
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts_lower:
            files.append(path)
    return files


def load_github_repo(url: str) -> tuple[Path, List[Path]]:
    """
    High-level helper: clone a GitHub repo and return (root, supported file paths).
    """
    repo_root = clone_github_repo(url)
    files = list_supported_files(repo_root)
    return repo_root, files


__all__ = ["clone_github_repo", "list_supported_files", "load_github_repo"]

