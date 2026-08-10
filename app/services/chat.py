"""Chat orchestration: retrieves context and streams answers."""

import logging
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from app.services.database import Database
from app.services.gemini import GeminiService
from app.services.okf import OKFService
from app.services.rag import RAGService, format_timestamp

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

LIBRARY_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions across a user's library "
    "of YouTube video transcripts. Use ONLY the provided transcript excerpts to "
    "answer. Each excerpt is prefixed with its video title in double brackets "
    "like [[Video Title]], followed by a timestamp span like [M:SS]. If the "
    "answer cannot be found in the excerpts, say that you don't know rather than "
    "guessing. When you draw from a specific video, name it and cite the "
    "timestamp span like [M:SS]. Be concise, accurate and answer directly.\n\n"
    "Transcript excerpts:\n{context}"
)


class ChatService:
    """Combines RAG retrieval with streaming Gemini answers."""

    def __init__(
        self,
        database: Database,
        rag: RAGService,
        gemini: GeminiService,
        okf: OKFService | None = None,
        library_max_videos: int = 5,
        library_top_k: int = 8,
    ) -> None:
        """Initialize the chat service with its dependencies.

        Args:
            database: Persistence layer.
            rag: Retrieval service.
            gemini: Streaming Gemini client.
            okf: Optional OKF knowledge layer used to pick candidate videos
                for library-level questions.
            library_max_videos: Maximum videos considered for a library chat.
            library_top_k: Chunks retrieved across the selected videos.
        """
        self.database = database
        self.rag = rag
        self.gemini = gemini
        self.okf = okf
        self.library_max_videos = library_max_videos
        self.library_top_k = library_top_k

    @property
    def available(self) -> bool:
        """Whether chat is functional (Gemini API key configured)."""
        return self.gemini.available

    async def stream_answer(
        self, conversation_id: int, question: str, user_id: int
    ) -> AsyncIterator[dict[str, Any]]:
        """Retrieve context and stream a chat answer, persisting messages.

        The user's question is saved immediately; the assistant answer is
        saved (together with its sources) once generation completes.

        Args:
            conversation_id: Primary key of the conversation to append to.
            question: The user's question.
            user_id: The owning user (used for ownership checks).

        Yields:
            Event dicts with a ``type`` key:
            ``sources`` (retrieved chunks), ``token`` (answer text fragment),
            ``error`` (a failure message) and ``done`` (stream finished).
        """
        conversation = self.database.get_conversation(conversation_id, user_id)
        if conversation is None:
            yield {"type": "error", "data": "Conversation not found."}
            return

        video_id = conversation["video_id"]
        video = self.database.get_video(video_id, user_id)
        if video is None:
            yield {"type": "error", "data": "Video not found."}
            return

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in self.database.list_messages(conversation_id)
        ]

        chunks = self.database.get_chunks(video_id)
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
            title=video["title"],
            context=context,
        )

        self.database.save_message(
            conversation_id=conversation_id, role="user", content=question
        )
        try:
            async for token in self.gemini.stream_answer(
                system_prompt, question, history=history
            ):
                yield {"type": "token", "data": token}
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("Gemini streaming failed for video %s", video_id)
            yield {"type": "error", "data": f"Answer generation failed: {exc}"}
            return

        yield {"type": "done", "data": ""}

    async def stream_library_answer(
        self, conversation_id: int, question: str, user_id: int
    ) -> AsyncIterator[dict[str, Any]]:
        """Answer a question across all of a user's stored videos.

        Candidate videos are first narrowed with the OKF knowledge layer (the
        cheap ``index.md`` catalog, scored against the question by embedding),
        then exact timestamped excerpts are retrieved with cross-video RAG and
        labeled by video title in the prompt. If no OKF index exists the most
        recent videos are used as candidates instead.

        Args:
            conversation_id: Primary key of a library-level conversation.
            question: The user's question.
            user_id: The owning user.

        Yields:
            Event dicts with a ``type`` key: ``sources``, ``token``, ``error``
            and ``done``.
        """
        conversation = self.database.get_conversation(conversation_id, user_id)
        if conversation is None:
            yield {"type": "error", "data": "Conversation not found."}
            return
        if conversation["video_id"] is not None:
            yield {"type": "error", "data": "This is not a library conversation."}
            return

        videos = self.database.list_videos(user_id)
        if not videos:
            yield {"type": "error", "data": "Your library is empty."}
            return

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in self.database.list_messages(conversation_id)
        ]

        selected = self._select_library_videos(user_id, question, videos)
        chunks: list[dict[str, Any]] = []
        for video in selected:
            for chunk in self.database.get_chunks(video["id"]):
                chunk["video_id"] = video["id"]
                chunk["video_title"] = video["title"]
                chunks.append(chunk)

        if not chunks:
            yield {
                "type": "error",
                "data": "No searchable content found in the selected videos.",
            }
            return

        results = self.rag.retrieve(chunks, question, top_k=self.library_top_k)
        context_lines = []
        for result in results:
            span = (
                f"[{format_timestamp(result['start'])} - "
                f"{format_timestamp(result['end'])}]"
            )
            context_lines.append(
                f"[[{result['video_title']}]] {span} {result['text']}"
            )
        context = "\n\n".join(context_lines)
        sources = [
            {
                "video_id": result["video_id"],
                "video_title": result["video_title"],
                "start": result["start"],
                "end": result["end"],
                "text": result["text"],
                "score": result["score"],
            }
            for result in results
        ]
        yield {"type": "sources", "data": sources}

        system_prompt = LIBRARY_SYSTEM_INSTRUCTION.format(context=context)

        self.database.save_message(
            conversation_id=conversation_id, role="user", content=question
        )
        try:
            async for token in self.gemini.stream_answer(
                system_prompt, question, history=history
            ):
                yield {"type": "token", "data": token}
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("Gemini library streaming failed for user %d", user_id)
            yield {"type": "error", "data": f"Answer generation failed: {exc}"}
            return

        yield {"type": "done", "data": ""}

    def _select_library_videos(
        self, user_id: int, question: str, videos: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Narrow ``videos`` to the candidates most relevant to ``question``.

        Uses the OKF concept catalog when present: the question embedding is
        matched against each concept's title + description. Otherwise the most
        recent ``library_max_videos`` are used.
        """
        candidates: list[tuple[float, dict[str, Any]]] = []
        concepts = self.okf.list_concepts(user_id) if self.okf is not None else []
        if concepts:
            by_yt = {c["youtube_id"]: c for c in concepts}
            concepts = [
                by_yt[video["youtube_id"]]
                for video in videos
                if video["youtube_id"] in by_yt
            ]
        if concepts:
            try:
                texts = [
                    f"{c['title']}: {c['description']}" for c in concepts
                ]
                matrix = self.rag.embeddings.embed(texts)
                query = self.rag.embeddings.embed([question]).reshape(1, -1)
                matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
                query_norm = np.linalg.norm(query)
                if matrix_norm.size and query_norm:
                    scores = (
                        matrix @ query.T / (matrix_norm * query_norm + 1e-9)
                    ).flatten()
                else:
                    scores = np.zeros(len(concepts))
                by_yt = {video["youtube_id"]: video for video in videos}
                candidates = [
                    (float(score), by_yt[c["youtube_id"]])
                    for score, c in zip(scores, concepts)
                    if c["youtube_id"] in by_yt
                ]
                candidates.sort(key=lambda pair: pair[0], reverse=True)
            except Exception:  # noqa: BLE001 - fall back to recency ordering
                logger.exception("OKF concept scoring failed; using recent videos")
        if not candidates:
            candidates = [(0.0, video) for video in videos]
        return [video for _, video in candidates[: self.library_max_videos]]
