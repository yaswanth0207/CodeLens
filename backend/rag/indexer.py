from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from sentence_transformers import SentenceTransformer

from backend.parser.chunker import ChunkWithMetadata, SyntaxAwareChunker


load_dotenv()


def _get_supported_extensions() -> Sequence[str]:
    raw = os.getenv("SUPPORTED_EXTENSIONS", ".py,.js,.ts,.jsx,.tsx")
    return [ext.strip() for ext in raw.split(",") if ext.strip()]


class CodebaseIndexer:
    """
    Build and maintain a Chroma-backed LlamaIndex over code chunks.

    This class is responsible purely for indexing; retrieval is handled
    separately by the retriever module.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "code_chunks",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(self.collection_name)

        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index: Optional[VectorStoreIndex] = None

        # Local embedding model (HuggingFace) to avoid OpenAI.
        # We keep the SentenceTransformer instance for potential direct use,
        # but the LlamaIndex integration is provided via HuggingFaceEmbedding.
        self._sentence_model = SentenceTransformer(self.embedding_model_name)
        self._embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)

        self._chunker = SyntaxAwareChunker()
        self._supported_exts = set(_get_supported_extensions())

        # Restore index from persisted Chroma if it has documents (survives backend restarts)
        self._restore_index_if_exists()

    def _restore_index_if_exists(self) -> None:
        """Load VectorStoreIndex from Chroma if the collection already has documents."""
        try:
            count = self._collection.count()
            if count and count > 0:
                self._index = VectorStoreIndex.from_vector_store(
                    self._vector_store,
                    embed_model=self._embed_model,
                )
        except Exception:
            self._index = None

    def get_indexed_file_paths(self) -> List[str]:
        """Return unique file paths currently in the Chroma collection (for restoring state)."""
        try:
            result = self._collection.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            seen: set[str] = set()
            paths: List[str] = []
            for m in metadatas:
                if not isinstance(m, dict):
                    continue
                path = m.get("file") or m.get("file_path") or ""
                if path and path not in seen:
                    seen.add(path)
                    paths.append(path)
            return sorted(paths)
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def index_codebase(self, root_path: Path) -> List[ChunkWithMetadata]:
        """
        Index all supported files under a directory.

        Returns the list of chunk+metadata pairs that were indexed.
        """
        files = list(self._iter_supported_files(root_path))
        chunks_with_meta = self._chunker.chunk_paths(files)

        documents = [self._chunk_to_document(cwm) for cwm in chunks_with_meta]
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self._storage_context,
            embed_model=self._embed_model,
            show_progress=True,
        )
        return chunks_with_meta

    def add_file(self, file_path: Path) -> List[ChunkWithMetadata]:
        """
        Add a single file to the existing index.

        If no index exists yet, a new one is created.
        """
        if not self._is_supported(file_path):
            raise ValueError(f"Unsupported file type: {file_path}")

        chunks_with_meta = self._chunker.chunk_file(file_path)
        documents = [self._chunk_to_document(cwm) for cwm in chunks_with_meta]

        if self._index is None:
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self._storage_context,
                embed_model=self._embed_model,
                show_progress=True,
            )
        else:
            for doc in documents:
                self._index.insert(doc)

        return chunks_with_meta

    def clear_index(self) -> None:
        """
        Clear the underlying Chroma collection and reset the index.
        """
        # Delete and recreate the collection for a clean slate.
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(self.collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = None

    @property
    def index(self) -> Optional[VectorStoreIndex]:
        """Expose the underlying VectorStoreIndex for retrieval components."""
        return self._index

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self._supported_exts

    def _iter_supported_files(self, root: Path) -> Iterable[Path]:
        if root.is_file():
            if self._is_supported(root):
                yield root
            return

        for path in root.rglob("*"):
            if path.is_file() and self._is_supported(path):
                yield path

    def _chunk_to_document(self, cwm: ChunkWithMetadata) -> Document:
        """
        Convert a chunk with metadata into a LlamaIndex Document.

        The `doc_id` is stable across runs as long as file path and line
        numbers remain unchanged.
        """
        chunk = cwm.chunk
        meta = cwm.metadata

        metadata_dict = asdict(meta)
        metadata_dict.update(
            {
                "file": chunk.file,
                "symbol": chunk.name,
                "docstring": chunk.docstring,
            }
        )

        return Document(
            doc_id=meta.id,
            text=chunk.code,
            metadata=metadata_dict,
        )


__all__ = ["CodebaseIndexer"]

