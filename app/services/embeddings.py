"""Sentence-Transformers embedding service for the RAG pipeline."""

import logging
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps a SentenceTransformer model to embed text locally."""

    def __init__(self, model_name: str) -> None:
        """Initialize the embedding service.

        Args:
            model_name: HuggingFace name of the SentenceTransformer model.
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def load_model(self) -> None:
        """Download (on first use) and load the embedding model into memory."""
        if self._model is None:
            logger.info("Loading embedding model %r", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model %r ready", self.model_name)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: Iterable of strings to embed.

        Returns:
            A 2-D ``numpy.ndarray`` of shape ``(len(texts), dim)``.
        """
        if self._model is None:
            raise RuntimeError("Embedding model has not been loaded yet.")
        batch = [text for text in texts if text.strip()]
        if not batch:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(self._model.encode(batch, convert_to_numpy=True), dtype=np.float32)
