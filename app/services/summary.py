"""AI study notes: TL;DR, key takeaways and auto-generated chapters.

``SummaryService`` orchestrates one Gemini call per transcript, normalizes the
model's proposed chapter boundaries onto real segment timestamps, and persists
the result in the ``summaries`` table so the video page renders notes without a
fresh API call.
"""

import logging
from typing import Any

from app.services.database import Database
from app.services.gemini import GeminiService

logger = logging.getLogger(__name__)


def normalize_chapters(
    chapters: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    duration: float,
) -> list[dict[str, Any]]:
    """Snap proposed chapters onto real segment boundaries.

    Gemini is asked to anchor each chapter start to a timestamp from the
    transcript, but a model may drift by a fraction of a second or invent a
    boundary. Every chapter is snapped to the nearest segment start, duplicates
    are dropped, chapters are sorted chronologically, and each end is derived
    from the next chapter's start (or the last segment end) so the chapters
    tile the timeline without gaps.

    Args:
        chapters: Candidate ``{"title", "start"}`` dicts.
        segments: List of ``{"start", "end", "text"}`` dicts.
        duration: Video duration in seconds (fallback for the final end).

    Returns:
        A list of ``{"title", "start", "end"}`` dicts sorted by start time.
    """
    boundaries = sorted(
        {float(s["start"]) for s in segments if s.get("start") is not None}
    )
    if not boundaries:
        return []

    cleaned: list[dict[str, Any]] = []
    seen: set[float] = set()
    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        title = str(chapter.get("title") or "").strip()
        try:
            start = float(chapter.get("start"))
        except (TypeError, ValueError):
            continue
        snapped = min(boundaries, key=lambda boundary: abs(boundary - start))
        if snapped in seen or not title:
            continue
        seen.add(snapped)
        cleaned.append({"title": title, "start": snapped})
    cleaned.sort(key=lambda chapter: chapter["start"])

    if segments:
        last_end = max(float(s.get("end") or 0) for s in segments)
    else:
        last_end = float(duration or 0)

    result: list[dict[str, Any]] = []
    for index, chapter in enumerate(cleaned):
        end = cleaned[index + 1]["start"] if index + 1 < len(cleaned) else last_end
        end = max(chapter["start"], end)
        result.append({**chapter, "end": end})
    return result


class SummaryService:
    """Generates and persists AI study notes for a user's videos."""

    def __init__(self, database: Database, gemini: GeminiService) -> None:
        """Initialize the summary service.

        Args:
            database: Persistence layer.
            gemini: Gemini client used for the single generation call.
        """
        self.database = database
        self.gemini = gemini

    @property
    def available(self) -> bool:
        """Whether summaries can be generated (Gemini API key configured)."""
        return self.gemini.available

    @property
    def model(self) -> str:
        """The Gemini model name used for generation."""
        return self.gemini.model

    def get(self, video_id: int, user_id: int) -> dict[str, Any] | None:
        """Return the stored summary for a video the user owns.

        Args:
            video_id: The video's primary key.
            user_id: The owning user.

        Returns:
            The parsed summary dict, or ``None`` when the video does not belong
            to the user or has no summary yet.
        """
        video = self.database.get_video(video_id, user_id)
        if video is None:
            return None
        return self.database.get_summary(video_id)

    async def generate(self, video_id: int, user_id: int) -> dict[str, Any]:
        """Generate and store AI study notes for a video.

        Args:
            video_id: The video's primary key.
            user_id: The owning user.

        Returns:
            The stored summary dict.

        Raises:
            ValueError: If the video does not exist or belongs to another user.
            RuntimeError: If no Gemini API key is configured.
        """
        video = self.database.get_video(video_id, user_id)
        if video is None:
            raise ValueError("Video not found.")
        if not self.gemini.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        notes = await self.gemini.generate_video_notes(
            title=video["title"],
            uploader=video["uploader"],
            duration=video["duration"],
            segments=video["segments"],
        )
        chapters = normalize_chapters(
            notes["chapters"],
            video["segments"],
            video["duration"],
        )
        self.database.save_summary(
            video_id=video_id,
            tldr=notes["tldr"],
            takeaways=notes["takeaways"],
            chapters=chapters,
            model=self.model,
        )
        summary = self.database.get_summary(video_id)
        if summary is None:
            raise ValueError("Summary could not be stored.")
        return summary
