"""Streaming Gemini API client."""

import logging
from collections.abc import AsyncIterator, Sequence

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiService:
    """Streams answers from the Gemini API using the google-genai SDK."""

    def __init__(self, *, api_key: str, model: str, max_tokens: int) -> None:
        """Initialize the Gemini service.

        Args:
            api_key: Google Gemini API key.
            model: Gemini model name (e.g. ``gemini-3.5-flash``).
            max_tokens: Maximum number of output tokens per answer.
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
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

