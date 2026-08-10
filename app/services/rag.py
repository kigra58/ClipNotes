"""Retrieval-Augmented Generation: chunking, indexing and retrieval."""

import logging
from typing import Any, Iterable

import numpy as np

from app.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format a seconds value as ``M:SS`` (e.g. ``3:07``)."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _make_chunk(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Join consecutive segments into a single chunk with a time span."""
    return {
        "start": min(seg["start"] for seg in segments),
        "end": max(seg["end"] for seg in segments),
        "text": " ".join(seg["text"].strip() for seg in segments if seg["text"].strip()),
    }


def _tail(segments: list[dict[str, Any]], overlap_chars: int) -> list[dict[str, Any]]:
    """Return the trailing segments covering at least ``overlap_chars``."""
    tail: list[dict[str, Any]] = []
    total = 0
    for seg in reversed(segments):
        tail.append(seg)
        total += len(seg["text"])
        if total >= overlap_chars:
            break
    return list(reversed(tail))


def chunk_segments(
    segments: list[dict[str, Any]], *, chunk_chars: int, overlap_chars: int
) -> list[dict[str, Any]]:
    """Group timestamped segments into overlapping text chunks.

    Args:
        segments: List of ``{"start", "end", "text"}`` dicts.
        chunk_chars: Approximate maximum characters per chunk.
        overlap_chars: Approximate characters shared between adjacent chunks.

    Returns:
        A list of ``{"start", "end", "text"}`` chunk dicts.
    """
    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_len = 0

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        if current and current_len + len(text) > chunk_chars:
            chunks.append(_make_chunk(current))
            current = _tail(current, overlap_chars)
            current_len = sum(len(s["text"]) for s in current)

        current.append(seg)
        current_len += len(text)

    if current:
        chunks.append(_make_chunk(current))

    return chunks


class RAGService:
    """Builds the vector index for a transcript and retrieves relevant chunks."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        *,
        top_k: int,
        chunk_chars: int,
        chunk_overlap: int,
    ) -> None:
        """Initialize the RAG service.

        Args:
            embeddings: The embedding service used to encode chunks.
            top_k: Number of chunks to retrieve per query.
            chunk_chars: Approximate chunk size in characters.
            chunk_overlap: Approximate overlap between chunks in characters.
        """
        self.embeddings = embeddings
        self.top_k = top_k
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap

    def build_chunks(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split transcript segments into embedded chunks.

        Args:
            segments: List of ``{"start", "end", "text"}`` dicts.

        Returns:
            Chunks with an ``embedding`` (``numpy.ndarray``) key.
        """
        chunks = chunk_segments(
            segments, chunk_chars=self.chunk_chars, overlap_chars=self.chunk_overlap
        )
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.embeddings.embed(texts)
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector
        return chunks

    def retrieve(
        self, chunks: Iterable[dict[str, Any]], query: str, *, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Find the chunks most similar to the query by cosine similarity.

        Args:
            chunks: Stored chunks, each with an ``embedding`` array.
            query: The user question.
            top_k: Number of results to return; defaults to ``self.top_k``.

        Returns:
            The top-k chunks with a ``score`` key, sorted best first.
        """
        chunk_list = [c for c in chunks if c.get("embedding") is not None]
        if not chunk_list:
            return []

        query_vector = self.embeddings.embed([query]).reshape(1, -1)
        matrix = np.stack([c["embedding"] for c in chunk_list])
        matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        scores = (matrix @ query_vector.T / (matrix_norm * query_norm + 1e-9)).flatten()

        limit = top_k if top_k is not None else self.top_k
        order = np.argsort(scores)[::-1][:limit]
        results = []
        for index in order:
            chunk = dict(chunk_list[int(index)])
            chunk["score"] = float(scores[int(index)])
            chunk.pop("embedding", None)
            results.append(chunk)
        return results

    @staticmethod
    def build_context(results: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a compact context block."""
        lines = []
        for result in results:
            span = (
                f"[{format_timestamp(result['start'])} - "
                f"{format_timestamp(result['end'])}]"
            )
            lines.append(f"{span} {result['text']}")
        return "\n\n".join(lines)
