from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional

from .tree_sitter_parser import CodeChunk, TreeSitterCodeParser


ChunkMode = Literal["file", "project"]


@dataclass
class ChunkMetadata:
    """Metadata attached to each syntax-aware chunk."""

    id: str
    language: str
    file_path: str
    symbol: str
    type: str
    start_line: int
    end_line: int


@dataclass
class ChunkWithMetadata:
    """Container for a code chunk and its metadata."""

    chunk: CodeChunk
    metadata: ChunkMetadata


class SyntaxAwareChunker:
    """
    Split code by function and class boundaries into self-contained units.

    This class builds on `TreeSitterCodeParser` and enriches each `CodeChunk`
    with language and location metadata suitable for indexing in a vector store.
    """

    def __init__(self, parser: Optional[TreeSitterCodeParser] = None) -> None:
        self._parser = parser or TreeSitterCodeParser()

    def _detect_language(self, path: Path) -> Optional[str]:
        """Proxy to the parser's language detection logic."""
        return TreeSitterCodeParser._detect_language(path)  # type: ignore[attr-defined]

    def chunk_file(self, path: Path) -> List[ChunkWithMetadata]:
        """
        Chunk a single source file.

        Each returned unit corresponds to a complete function, method, or class
        definition and carries metadata required for RAG indexing and citation.
        """
        language = self._detect_language(path)
        if language is None:
            raise ValueError(f"Unsupported file type for chunking: {path}")

        chunks = self._parser.parse_file(path)
        return [self._attach_metadata(chunk, language) for chunk in chunks]

    def chunk_paths(self, paths: Iterable[Path]) -> List[ChunkWithMetadata]:
        """
        Chunk multiple files, skipping unsupported extensions.

        This is intended as the main entry point when indexing a codebase.
        """
        all_chunks: List[ChunkWithMetadata] = []
        for path in paths:
            language = self._detect_language(path)
            if language is None:
                continue
            all_chunks.extend(self.chunk_file(path))
        return all_chunks

    def _attach_metadata(self, chunk: CodeChunk, language: str) -> ChunkWithMetadata:
        """
        Build a stable identifier and metadata envelope for a chunk.

        The identifier encodes file path and line range so it can be used to
        de-duplicate or cross-reference chunks across index runs.
        """
        chunk_id = f"{chunk.file}:{chunk.start_line}-{chunk.end_line}"
        metadata = ChunkMetadata(
            id=chunk_id,
            language=language,
            file_path=chunk.file,
            symbol=chunk.name,
            type=chunk.type,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )
        return ChunkWithMetadata(chunk=chunk, metadata=metadata)


__all__ = ["ChunkMetadata", "ChunkWithMetadata", "SyntaxAwareChunker"]

