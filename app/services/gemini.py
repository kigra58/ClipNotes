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
            model: Gemini model name (e.g. ``gemini-3.5-flash``).
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

    async def generate_video_notes(
        self,
        *,
        title: str,
        uploader: str | None,
        duration: float,
        segments: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate TL;DR, key takeaways and chapters for a transcript.

        A single Gemini call turns the timestamped segments into structured
        study notes: a short TL;DR, 3-6 takeaway bullets, and a table of
        contents whose chapter start times snap to real segment boundaries
        (snapping happens in :mod:`app.services.summary`).

        Args:
            title: Video title.
            uploader: Channel name, when available.
            duration: Video duration in seconds.
            segments: List of ``{"start", "end", "text"}`` dicts.

        Returns:
            A dict with ``tldr`` (str), ``takeaways`` (list[str]) and
            ``chapters`` (list of ``{"title", "start"}`` dicts).

        Raises:
            RuntimeError: If no API key is configured.
            ValueError: If the model does not return a usable JSON object.
        """
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        client = self._client_instance()
        prompt = self.build_video_notes_prompt(
            title=title,
            uploader=uploader,
            duration=duration,
            segments=segments,
            summary_chars=self.summary_chars,
        )
        config = types.GenerateContentConfig(temperature=0.3, max_output_tokens=2048)
        response = await client.aio.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        data = _parse_json_object(response.text or "")
        if not isinstance(data, dict):
            raise ValueError("Gemini did not return a JSON object for the video notes.")

        tldr = str(data.get("tldr") or "").strip()
        takeaways = [
            str(item).strip()
            for item in (data.get("takeaways") or [])
            if str(item).strip()
        ]
        chapters: list[dict[str, Any]] = []
        for chapter in data.get("chapters") or []:
            if not isinstance(chapter, dict):
                continue
            chapter_title = str(chapter.get("title") or "").strip()
            try:
                start = float(chapter.get("start"))
            except (TypeError, ValueError):
                continue
            if chapter_title and start >= 0:
                chapters.append({"title": chapter_title, "start": start})
        return {"tldr": tldr, "takeaways": takeaways, "chapters": chapters}

    async def generate_quiz(
        self,
        *,
        title: str,
        uploader: str | None,
        duration: float,
        segments: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a multiple-choice quiz from a transcript.

        A single Gemini call turns the timestamped segments into a set of
        5-8 multiple-choice questions, each with four options, the index of
        the correct option, and a one-sentence explanation.

        Args:
            title: Video title.
            uploader: Channel name, when available.
            duration: Video duration in seconds.
            segments: List of ``{"start", "end", "text"}`` dicts.

        Returns:
            A dict with a ``questions`` key holding a list of
            ``{"question", "options", "correct_index", "explanation"}`` dicts.

        Raises:
            RuntimeError: If no API key is configured.
            ValueError: If the model does not return a usable JSON object.
        """
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        client = self._client_instance()
        prompt = self.build_quiz_prompt(
            title=title,
            uploader=uploader,
            duration=duration,
            segments=segments,
            summary_chars=self.summary_chars,
        )
        config = types.GenerateContentConfig(temperature=0.4, max_output_tokens=4096)
        response = await client.aio.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        data = _parse_json_object(response.text or "")
        if not isinstance(data, dict):
            raise ValueError("Gemini did not return a JSON object for the quiz.")

        questions: list[dict[str, Any]] = []
        for question in data.get("questions") or []:
            if not isinstance(question, dict):
                continue
            text = str(question.get("question") or "").strip()
            options = [
                str(option).strip()
                for option in (question.get("options") or [])
                if str(option).strip()
            ]
            try:
                correct_index = int(question.get("correct_index"))
            except (TypeError, ValueError):
                correct_index = -1
            explanation = str(question.get("explanation") or "").strip()
            if text and len(options) >= 2 and 0 <= correct_index < len(options):
                questions.append(
                    {
                        "question": text,
                        "options": options,
                        "correct_index": correct_index,
                        "explanation": explanation,
                    }
                )
        return {"questions": questions}

    async def generate_social_post(
        self,
        *,
        title: str,
        uploader: str | None,
        duration: float,
        segments: Sequence[dict[str, Any]],
        summary_chars: int = 40000,
    ) -> dict[str, Any]:
        """Generate a ready-to-publish social media post with hashtags.

        A single Gemini call turns the video title and transcript into an
        engaging caption (with a hook and a call to action) plus 5-8 relevant
        hashtags. The caption always ends with the hashtags on their own lines,
        ready to paste. The result is not persisted; callers render it into an
        editable editor.

        Args:
            title: Video title.
            uploader: Channel name, when available.
            duration: Video duration in seconds.
            segments: List of ``{"start", "end", "text"}`` dicts.
            summary_chars: Maximum transcript characters sent to the model.

        Returns:
            A dict with ``post`` (str, hashtags included) and ``hashtags``
            (list[str]).

        Raises:
            RuntimeError: If no API key is configured.
            ValueError: If the model returns an empty response.
        """
        if not self.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        client = self._client_instance()
        prompt = self.build_social_post_prompt(
            title=title,
            uploader=uploader,
            duration=duration,
            segments=segments,
            summary_chars=summary_chars,
        )
        config = types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024)
        response = await client.aio.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        raw = self._clean_response_text(response.text or "")
        if not raw:
            raise ValueError("Gemini returned an empty response for the social post.")

        data = _parse_json_object(raw)
        if isinstance(data, dict):
            post = str(data.get("post") or "").strip()
            tags = [
                str(tag).strip()
                for tag in (data.get("hashtags") or [])
                if str(tag).strip()
            ]
            hashtags = tags or GeminiService._extract_hashtags(post)
            if not post:
                post = raw
        else:
            match = (
                re.search(r'"post"\s*:\s*"(.*?)",\s*"hashtags"', raw, re.DOTALL)
                if raw.lstrip().startswith("{")
                else None
            )
            post = match.group(1) if match else raw
            hashtags = GeminiService._extract_hashtags(raw)

        if hashtags and not any(tag in post for tag in hashtags):
            post = post.rstrip() + "\n\n" + "\n".join(hashtags)
        return {"post": post, "hashtags": hashtags}

    @staticmethod
    def _extract_hashtags(text: str) -> list[str]:
        """Pull unique ``#tag`` words out of a text block, in order."""
        return list(dict.fromkeys(re.findall(r"#[\w-]+", text)))

    @staticmethod
    def _clean_response_text(text: str) -> str:
        """Strip markdown code fences and surrounding whitespace from a response."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def build_social_post_prompt(
        *,
        title: str,
        uploader: str | None,
        duration: float,
        segments: Sequence[dict[str, Any]],
        summary_chars: int,
    ) -> str:
        """Build the prompt used to generate a social media post.

        Args:
            title: Video title.
            uploader: Channel name, when available.
            duration: Video duration in seconds.
            segments: List of ``{"start", "end", "text"}`` dicts.
            summary_chars: Maximum transcript characters sent to the model.

        Returns:
            The full user prompt string.
        """
        transcript = GeminiService._segment_lines(segments)[:summary_chars]
        return (
            "You are a social media content writer. Write an engaging social "
            "media post about the following YouTube video, based on its title "
            "and transcript.\n"
            "Write ONLY the post text itself — no JSON, no markdown, no code "
            "fences, no extra commentary or headings.\n"
            "Make it 2-4 short paragraphs or bullet points, starting with a "
            "hook and ending with a call to action, in a friendly, human tone "
            "as the video creator.\n"
            "Base the post only on the actual content of the transcript; do "
            "not invent facts.\n"
            "End the post with 5-8 relevant hashtags, each starting with '#', "
            "on their own line(s) at the very end of the post.\n\n"
            f"Video title: {title}\n"
            f"Uploader: {uploader or 'unknown'}\n"
            f"Duration (seconds): {duration}\n\n"
            f"Transcript:\n{transcript}\n"
        )

    @staticmethod
    def build_video_notes_prompt(
        *,
        title: str,
        uploader: str | None,
        duration: float,
        segments: Sequence[dict[str, Any]],
        summary_chars: int,
    ) -> str:
        """Build the prompt used to generate structured video notes.

        Segments are rendered as ``[M:SS] text`` lines (truncated to
        ``summary_chars``) so the model can both summarize the content and
        anchor chapter boundaries to real timestamps.

        Args:
            title: Video title.
            uploader: Channel name, when available.
            duration: Video duration in seconds.
            segments: List of ``{"start", "end", "text"}`` dicts.
            summary_chars: Maximum transcript characters sent to the model.

        Returns:
            The full user prompt string.
        """
        timestamped = GeminiService._segment_lines(segments)
        timestamped = timestamped[:summary_chars]
        return (
            "Create concise study notes for this YouTube video transcript: a "
            "TL;DR, key takeaways, and an auto-generated table of contents "
            "(chapters) with timestamps.\n"
            "Return ONLY a JSON object, no markdown, no code fences, with "
            "exactly these keys:\n"
            '{\n  "tldr": "<2-3 sentence summary of the whole video>",\n'
            '  "takeaways": ["<4-6 key takeaways, each a complete sentence>"],\n'
            '  "chapters": [{"title": "<2-5 word chapter name>", '
            '"start": <float seconds>}]\n}\n'
            "Rules:\n"
            "- Chapters must be in chronological order and cover the whole video; "
            "aim for 3-8 chapters.\n"
            '- Each chapter "start" must be a timestamp that appears in the '
            "timestamped transcript below (in seconds).\n"
            "- Chapter titles must be concise and descriptive.\n"
            "- Takeaway bullets must be self-contained sentences.\n\n"
            f"Video title: {title}\n"
            f"Uploader: {uploader or 'unknown'}\n"
            f"Duration (seconds): {duration}\n\n"
            f"Timestamped transcript:\n{timestamped}\n"
        )

    @staticmethod
    def build_quiz_prompt(
        *,
        title: str,
        uploader: str | None,
        duration: float,
        segments: Sequence[dict[str, Any]],
        summary_chars: int,
    ) -> str:
        """Build the prompt used to generate a multiple-choice quiz.

        Segments are rendered as ``[M:SS] text`` lines (truncated to
        ``summary_chars``) so the model can ask questions anchored to the
        actual content of the video.

        Args:
            title: Video title.
            uploader: Channel name, when available.
            duration: Video duration in seconds.
            segments: List of ``{"start", "end", "text"}`` dicts.
            summary_chars: Maximum transcript characters sent to the model.

        Returns:
            The full user prompt string.
        """
        timestamped = GeminiService._segment_lines(segments)
        timestamped = timestamped[:summary_chars]
        return (
            "Create a multiple-choice quiz to test understanding of this "
            "YouTube video, based on its title and transcript.\n"
            "Return ONLY a JSON object, no markdown, no code fences, with "
            "exactly this shape:\n"
            '{\n  "questions": [\n'
            '    {"question": "<question text>", '
            '"options": ["<option A>", "<option B>", "<option C>", "<option D>"], '
            '"correct_index": <0-based index of the correct option>, '
            '"explanation": "<one sentence why>"}\n'
            "  ]\n}\n"
            "Rules:\n"
            "- Generate 5-8 questions covering the main points of the video.\n"
            "- Every question must have exactly 4 options, in a consistent "
            "order, with exactly one correct answer.\n"
            "- \"correct_index\" is the 0-based position of the correct option "
            "inside \"options\".\n"
            "- Questions and options must be based only on the actual content "
            "of the transcript; do not invent facts.\n"
            "- Vary question types: recall, definitions, and "
            '"why/relationship" questions.\n'
            "- Explanations must be a single concise sentence.\n\n"
            f"Video title: {title}\n"
            f"Uploader: {uploader or 'unknown'}\n"
            f"Duration (seconds): {duration}\n\n"
            f"Timestamped transcript:\n{timestamped}\n"
        )

    @staticmethod
    def _segment_lines(segments: Sequence[dict[str, Any]]) -> str:
        """Render segments as ``[M:SS] text`` lines, skipping blank text."""
        lines: list[str] = []
        for segment in segments:
            text = " ".join(str(segment.get("text") or "").split())
            if not text:
                continue
            lines.append(f"[{GeminiService._time_label(segment.get('start', 0))}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _time_label(seconds: float) -> str:
        """Format seconds as ``M:SS`` (e.g. ``3:07``)."""
        total = max(0, int(round(seconds)))
        minutes, secs = divmod(total, 60)
        return f"{minutes}:{secs:02d}"

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

