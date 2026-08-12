"""AI social media post generation with persistence.

``SocialPostService`` turns a video title and transcript into an editable
social media post with hashtags, persists the result in the ``social_posts``
table, and lets callers fetch the stored version so the page renders the last
generated post (and a Regenerate control) without a fresh API call.
"""

import logging
from typing import Any

from app.services.database import Database
from app.services.gemini import GeminiService

logger = logging.getLogger(__name__)


class SocialPostService:
    """Generates and persists editable social posts for a user's videos."""

    def __init__(self, database: Database, gemini: GeminiService) -> None:
        """Initialize the social post service.

        Args:
            database: Persistence layer.
            gemini: Gemini client used for the single generation call.
        """
        self.database = database
        self.gemini = gemini

    @property
    def available(self) -> bool:
        """Whether social posts can be generated (Gemini API key configured)."""
        return self.gemini.available

    @property
    def model(self) -> str:
        """The Gemini model name used for generation."""
        return self.gemini.model

    def get(self, video_id: int, user_id: int) -> dict[str, Any] | None:
        """Return the stored social post for a video the user owns.

        Args:
            video_id: The video's primary key.
            user_id: The owning user.

        Returns:
            The stored post dict, or ``None`` when the video does not belong to
            the user or has no post yet.
        """
        video = self.database.get_video(video_id, user_id)
        if video is None:
            return None
        return self.database.get_social_post(video_id)

    async def generate(self, video_id: int, user_id: int) -> dict[str, Any]:
        """Generate and store a social post for a video.

        Args:
            video_id: The video's primary key.
            user_id: The owning user.

        Returns:
            The stored post dict.

        Raises:
            ValueError: If the video does not exist or belongs to another user.
            RuntimeError: If no Gemini API key is configured.
        """
        video = self.database.get_video(video_id, user_id)
        if video is None:
            raise ValueError("Video not found.")
        if not self.gemini.available:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        result = await self.gemini.generate_social_post(
            title=video["title"],
            uploader=video["uploader"],
            duration=video["duration"],
            segments=video["segments"],
        )
        self.database.save_social_post(
            video_id=video_id,
            post=result["post"],
            hashtags=result["hashtags"],
            model=self.model,
        )
        stored = self.database.get_social_post(video_id)
        if stored is None:
            raise ValueError("Social post could not be stored.")
        return stored
