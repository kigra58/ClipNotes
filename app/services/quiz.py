"""AI multiple-choice quiz generation with persistence.

``QuizService`` turns a video title and transcript into a multiple-choice
quiz with explanations, persists the result in the ``quizzes`` table, and
lets callers fetch the stored version so the video page renders the last
generated quiz (and a Regenerate control) without a fresh API call.
"""

import logging
from typing import Any

from app.services.database import Database
from app.services.gemini import GeminiService

logger = logging.getLogger(__name__)


def normalize_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop malformed questions so the quiz always renders cleanly.

    A question is kept only when it has non-blank text, at least two
    non-blank options, and a ``correct_index`` pointing at one of those
    options. The question is truncated to four options so the multiple-choice
    UI stays consistent.

    Args:
        questions: Candidate ``{"question", "options", "correct_index",
            "explanation"}`` dicts.

    Returns:
        A cleaned list of ``{"question", "options", "correct_index",
        "explanation"}`` dicts.
    """
    cleaned: list[dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        text = str(question.get("question") or "").strip()
        options = [
            str(option).strip()
            for option in (question.get("options") or [])
            if str(option).strip()
        ][:4]
        try:
            correct_index = int(question.get("correct_index"))
        except (TypeError, ValueError):
            correct_index = -1
        if not text or len(options) < 2 or not 0 <= correct_index < len(options):
            continue
        cleaned.append(
            {
                "question": text,
                "options": options,
                "correct_index": correct_index,
                "explanation": str(question.get("explanation") or "").strip(),
            }
        )
    return cleaned


class QuizService:
    """Generates and persists AI quizzes for a user's videos."""

    def __init__(self, database: Database, gemini: GeminiService) -> None:
        """Initialize the quiz service.

        Args:
            database: Persistence layer.
            gemini: Gemini client used for the single generation call.
        """
        self.database = database
        self.gemini = gemini

    @property
    def available(self) -> bool:
        """Whether quizzes can be generated (Gemini API key configured)."""
        return self.gemini.available

    @property
    def model(self) -> str:
        """The Gemini model name used for generation."""
        return self.gemini.model

    def get(self, video_id: int, user_id: int) -> dict[str, Any] | None:
        """Return the stored quiz for a video the user owns.

        Args:
            video_id: The video's primary key.
            user_id: The owning user.

        Returns:
            The stored quiz dict, or ``None`` when the video does not belong
            to the user or has no quiz yet.
        """
        video = self.database.get_video(video_id, user_id)
        if video is None:
            return None
        return self.database.get_quiz(video_id)

    async def generate(self, video_id: int, user_id: int) -> dict[str, Any]:
        """Generate and store a quiz for a video.

        Args:
            video_id: The video's primary key.
            user_id: The owning user.

        Returns:
            The stored quiz dict.

        Raises:
            ValueError: If the video does not exist or belongs to another user.
            RuntimeError: If no Gemini API key is configured.
        """
        video = self.database.get_video(video_id, user_id)
        if video is None:
            raise ValueError("Video not found.")
        if not self.gemini.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        result = await self.gemini.generate_quiz(
            title=video["title"],
            uploader=video["uploader"],
            duration=video["duration"],
            segments=video["segments"],
        )
        questions = normalize_questions(result["questions"])
        if not questions:
            raise ValueError("Gemini returned no valid quiz questions.")
        self.database.save_quiz(
            video_id=video_id,
            questions=questions,
            model=self.model,
        )
        stored = self.database.get_quiz(video_id)
        if stored is None:
            raise ValueError("Quiz could not be stored.")
        return stored
