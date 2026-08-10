"""Streaming Gemini API client."""

import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def _parse_json_object(text: str) -> Any:
    """Parse a JSON object out of a model response, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


class GeminiService:
    """Streams answers from the Gemini API using the google-genai SDK."""

    def __init__(
        self, *, api_key: str, model: str, max_tokens: int, summary_chars: int = 40000
    ) -> None:
        """Initialize the Gemini service.

        Args:
            api_key: Google Gemini API key.
            model: Gemini model name (e.g. ``gemini-2.0-flash``).
            max_tokens: Maximum number of output tokens per answer.
            summary_chars: Transcript characters fed to concept summarization.
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.summary_chars = summary_chars
        self._client: genai.Client | None = None

    @property
    def available(self) -> bool:
        """Whether the service can make requests (an API key is configured)."""
        return bool(self.api_key)

    def _client_instance(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def stream_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Sequence[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a model response for the given prompts.

        Args:
            system_prompt: System-level instructions including RAG context.
            user_prompt: The user's question.
            history: Optional prior turns as ``{"role": ..., "content": ...}``
                dicts (``"user"`` or ``"assistant"``), oldest first.

        Yields:
            Text fragments of the generated answer.

        Raises:
            RuntimeError: If no API key is configured.
        """
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        client = self._client_instance()
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=self.max_tokens,
        )
        contents = [*self._to_contents(history or [])]
        contents.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))
        stream = await client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def generate_concept_summary(
        self, *, title: str, uploader: str | None, duration: float, transcript: str
    ) -> dict[str, Any]:
        """Summarize a transcript into the fields of an OKF concept document.

        Args:
            title: Video title.
            uploader: Channel name, when available.
            duration: Video duration in seconds.
            transcript: Full transcript text (truncated internally).

        Returns:
            A dict with ``description``, ``tags``, ``summary``, ``topics``
            and ``entities`` string-list keys.

        Raises:
            RuntimeError: If no API key is configured.
            ValueError: If the model does not return a usable JSON object.
        """
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        client = self._client_instance()
        prompt = (
            "Summarize this YouTube video transcript for building a personal "
            "knowledge base. Return ONLY a JSON object, no markdown, no code "
            "fences, with exactly these keys:\n"
            '{"description": "<one-sentence summary naming the main topics>", '
            '"tags": ["<3-6 lowercase keywords>"], '
            '"summary": "<3-5 sentence overview of the whole video>", '
            '"topics": ["<the main topics discussed>"], '
            '"entities": ["<people, companies, technologies, or concepts mentioned>"]}\n\n'
            f"Video title: {title}\n"
            f"Uploader: {uploader or 'unknown'}\n"
            f"Duration (seconds): {duration}\n\n"
            f"Transcript:\n{transcript[: self.summary_chars]}\n"
        )
        config = types.GenerateContentConfig(temperature=0.3, max_output_tokens=1024)
        response = await client.aio.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        data = _parse_json_object(response.text or "")
        if not isinstance(data, dict):
            raise ValueError("Gemini did not return a JSON object for the concept summary.")
        return {
            "description": str(data.get("description") or "").strip(),
            "tags": [str(tag) for tag in (data.get("tags") or [])],
            "summary": str(data.get("summary") or "").strip(),
            "topics": [str(topic) for topic in (data.get("topics") or [])],
            "entities": [str(entity) for entity in (data.get("entities") or [])],
        }

    @staticmethod
    def _to_contents(history: Sequence[dict[str, str]]) -> list[types.Content]:
        """Convert ``{"role", "content"}`` turns into Gemini content parts."""
        contents: list[types.Content] = []
        for turn in history:
            role = "model" if turn.get("role") == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=turn["content"])])
            )
        return contents

