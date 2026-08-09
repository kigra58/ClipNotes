"""Chat orchestration: retrieves context and streams answers."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from app.services.database import Database
from app.services.gemini import GeminiService
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions about a YouTube video "
    "transcript. Use ONLY the provided transcript excerpts to answer. If the "
    "answer cannot be found in the excerpts, say that you don't know rather "
    "than guessing. When quoting, cite the timestamp span like [M:SS]. Be "
    "concise, accurate and answer directly.\n\n"
    "Video title: {title}\n\n"
    "Transcript excerpts:\n{context}"
)


class ChatService:
    """Combines RAG retrieval with streaming Gemini answers."""

    def __init__(
        self,
        database: Database,
        rag: RAGService,
        gemini: GeminiService,
    ) -> None:
        """Initialize the chat service with its dependencies.

        Args:
            database: Transcript persistence layer.
            rag: Retrieval service.
            gemini: Streaming Gemini client.
        """
        self.database = database
        self.rag = rag
        self.gemini = gemini

    @property
    def available(self) -> bool:
        """Whether chat is functional (Gemini API key configured)."""
        return self.gemini.available

    async def stream_answer(
        self, transcript_id: int, question: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Retrieve context and stream a chat answer as events.

        Args:
            transcript_id: Primary key of the transcript to query.
            question: The user's question.

        Yields:
            Event dicts with a ``type`` key:
            ``sources`` (retrieved chunks), ``token`` (answer text fragment),
            ``error`` (a failure message) and ``done`` (stream finished).
        """
        transcript = self.database.get_transcript(transcript_id)
        if transcript is None:
            yield {"type": "error", "data": "Transcript not found."}
            return

        chunks = self.database.get_chunks(transcript_id)
        results = self.rag.retrieve(chunks, question)
        context = self.rag.build_context(results)
        sources = [
            {
                "start": result["start"],
                "end": result["end"],
                "text": result["text"],
                "score": result["score"],
            }
            for result in results
        ]
        yield {"type": "sources", "data": sources}

        system_prompt = SYSTEM_INSTRUCTION.format(
            title=transcript["title"],
            context=context,
        )
        try:
            async for token in self.gemini.stream_answer(system_prompt, question):
                yield {"type": "token", "data": token}
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("Gemini streaming failed for transcript %s", transcript_id)
            yield {"type": "error", "data": f"Answer generation failed: {exc}"}
            return

        yield {"type": "done", "data": ""}
