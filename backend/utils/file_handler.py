from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_SUPPORTED_EXTS = (".py", ".js", ".ts", ".jsx", ".tsx")


def _get_supported_extensions() -> Sequence[str]:
    raw = os.getenv("SUPPORTED_EXTENSIONS")
    if not raw:
        return DEFAULT_SUPPORTED_EXTS
    return [ext.strip() for ext in raw.split(",") if ext.strip()]


def ensure_storage_dir(base_dir: Path) -> Path:
    """
    Ensure that a storage directory exists for uploaded files.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def clear_upload_dir(upload_dir: Path) -> None:
    """
    Remove all files and subdirectories under upload_dir so only the next
    upload batch is present when indexing (avoids indexing old uploads).
    """
    if not upload_dir.exists():
        return
    for p in upload_dir.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


def save_uploaded_files(upload_dir: Path, filenames: Iterable[str], contents: Iterable[bytes]) -> List[Path]:
    """
    Persist uploaded files to disk and return their paths.

    This function is storage-agnostic; FastAPI-specific request parsing happens
    in the API layer.
    """
    upload_dir = ensure_storage_dir(upload_dir)
    paths: List[Path] = []
    for name, data in zip(filenames, contents):
        path = upload_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        paths.append(path)
    return paths


def filter_supported_files(paths: Iterable[Path]) -> List[Path]:
    """
    Filter a list of paths to only those with supported extensions.
    """
    exts = {e.lower() for e in _get_supported_extensions()}
    out: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


__all__ = ["clear_upload_dir", "ensure_storage_dir", "save_uploaded_files", "filter_supported_files"]

