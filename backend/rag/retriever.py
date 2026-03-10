from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .indexer import CodebaseIndexer


load_dotenv()


@dataclass
class RetrievedChunk:
    """A chunk of code returned from hybrid retrieval."""

    id: str
    code: str
    file: str
    symbol: str
    language: str
    start_line: int
    end_line: int
    score: float  # combined score for ranking/display
    semantic_score: float  # raw vector similarity (used for relevance threshold)
    metadata: Dict


class HybridCodeRetriever:
    """
    Combine semantic vector search with lightweight keyword scoring.

    This component sits on top of `CodebaseIndexer` and is responsible for
    returning the most relevant chunks together with their file and line
    locations so the generator can answer questions and cite sources.
    """

    def __init__(
        self,
        indexer: CodebaseIndexer,
        default_top_k: Optional[int] = None,
    ) -> None:
        self._indexer = indexer
        self._default_top_k = default_top_k or int(os.getenv("MAX_CHUNKS_RETURNED", "5"))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def query(self, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Run a hybrid retrieval for a natural-language query.

        Returns up to `top_k` chunks ordered by combined semantic + keyword
        relevance, each containing file name and line numbers for citation.
        """
        index = self._get_index_or_raise()

        k = top_k or self._default_top_k
        # Request enough candidates for hybrid ranking; Chroma often returns tiny similarity scores.
        similarity_top_k = max(k * 5, 20)
        retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        all_retrieved_chunks: List[NodeWithScore] = retriever.retrieve(question)

        if not all_retrieved_chunks:
            return []

        # Optional filter by similarity; for code, Chroma uses exp(-distance) so scores are often < 0.2.
        # Default 0.0 = no filter; use all retrieved nodes. Always fall back to top_k if none pass.
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
        filtered_chunks = [nws for nws in all_retrieved_chunks if (nws.score or 0) >= threshold]
        if not filtered_chunks:
            filtered_chunks = all_retrieved_chunks[:k]

        keyword_scores = self._compute_keyword_scores(question, filtered_chunks)
        combined = self._combine_scores(filtered_chunks, keyword_scores)
        combined.sort(key=lambda rc: rc.score, reverse=True)
        result = combined[:k]

        # Debug: print raw semantic and combined scores to console.
        for rc in result:
            print(f"[retriever] semantic={rc.semantic_score:.3f} combined={rc.score:.3f} file={rc.file!r} symbol={rc.symbol!r}")
        print(f"[retriever] threshold={threshold} returned={len(result)}")

        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_index_or_raise(self) -> VectorStoreIndex:
        index = self._indexer.index
        if index is None:
            raise RuntimeError("No index is available. Index a codebase before querying.")
        return index

    @staticmethod
    def _compute_keyword_scores(
        question: str,
        nodes_with_scores: List[NodeWithScore],
    ) -> Dict[str, float]:
        """
        Compute a simple keyword relevance score for each node.

        We approximate BM25-like behaviour by counting term occurrences for
        meaningful tokens across the node's text and key metadata fields.
        """
        # Extract relatively meaningful lowercase tokens from the question.
        tokens = [t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower()) if len(t) > 2]
        if not tokens:
            return {node.node.node_id: 0.0 for node in nodes_with_scores}

        scores: Dict[str, float] = {}
        for nws in nodes_with_scores:
            node = nws.node
            text = (node.get_content() or "").lower()
            meta = (node.metadata or {})  # type: ignore[assignment]

            # Incorporate filename and symbol name into keyword matching.
            file_name = str(meta.get("file", "")).lower()
            symbol_name = str(meta.get("symbol", "")).lower()
            haystack = " ".join([text, file_name, symbol_name])

            score = 0.0
            for tok in tokens:
                # Simple frequency-based score.
                occurrences = haystack.count(tok)
                if occurrences:
                    score += occurrences
            scores[node.node_id] = score

        return scores

    @staticmethod
    def _combine_scores(
        nodes_with_scores: List[NodeWithScore],
        keyword_scores: Dict[str, float],
    ) -> List[RetrievedChunk]:
        """
        Combine semantic similarity and keyword scores into a final ranking.
        """
        # Collect raw semantic scores; fall back to 0.0 if not provided.
        semantic_raw: Dict[str, float] = {}
        for nws in nodes_with_scores:
            node_id = nws.node.node_id
            # In most vector stores, higher score is better; if None, use 0.0.
            semantic_raw[node_id] = float(nws.score or 0.0)

        # Normalise scores to [0, 1] within this candidate set.
        def _normalise(score_map: Dict[str, float]) -> Dict[str, float]:
            if not score_map:
                return {}
            values = list(score_map.values())
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                # Avoid division by zero; treat all scores as equal.
                return {k: 0.5 for k in score_map.keys()}
            return {k: (v - min_v) / (max_v - min_v) for k, v in score_map.items()}

        semantic_norm = _normalise(semantic_raw)
        keyword_norm = _normalise(keyword_scores)

        combined_chunks: List[RetrievedChunk] = []
        alpha = 0.7  # weight for semantic similarity
        beta = 0.3   # weight for keyword relevance

        for nws in nodes_with_scores:
            node = nws.node
            meta = dict(node.metadata or {})  # type: ignore[arg-type]
            node_id = node.node_id

            final_score = alpha * semantic_norm.get(node_id, 0.0) + beta * keyword_norm.get(node_id, 0.0)
            raw_semantic = semantic_raw.get(node_id, 0.0)

            file_path = str(meta.get("file", ""))
            symbol = str(meta.get("symbol", ""))
            language = str(meta.get("language", meta.get("language_", "")))  # be robust to key naming
            start_line = int(meta.get("start_line", 0))
            end_line = int(meta.get("end_line", 0))

            rc = RetrievedChunk(
                id=node_id,
                code=node.get_content() or "",
                file=file_path,
                symbol=symbol,
                language=language,
                start_line=start_line,
                end_line=end_line,
                score=final_score,
                semantic_score=raw_semantic,
                metadata=asdict_meta(meta),
            )
            combined_chunks.append(rc)

        return combined_chunks


def asdict_meta(meta: Dict) -> Dict:
    """
    Lightweight helper to ensure metadata is JSON-serialisable.

    Currently just returns a shallow copy, but is a dedicated function so it
    can be extended later if needed.
    """
    return dict(meta)


__all__ = ["RetrievedChunk", "HybridCodeRetriever"]

