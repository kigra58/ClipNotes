"""Pydantic request and response schemas for the transcript API."""

from pydantic import BaseModel, Field, field_validator

from app.utils.youtube import extract_video_id


class TranscribeRequest(BaseModel):
    """Request body for the transcription endpoint."""

    youtube_url: str = Field(
        description="A valid YouTube video URL.",
        examples=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
    )

    @field_validator("youtube_url")
    @classmethod
    def validate_youtube_url(cls, value: str) -> str:
        """Validate the URL and return the raw (unmodified) URL.

        Args:
            value: The raw URL submitted by the client.

        Returns:
            The validated raw URL.

        Raises:
            InvalidURLError: If the URL is not a valid YouTube URL.
        """
        extract_video_id(value)
        return value


class TranscriptSegment(BaseModel):
    """A timestamped piece of the transcript."""

    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")
    text: str = Field(description="Segment text.")


class TranscribeResponse(BaseModel):
    """Full transcription result returned to the client."""

    video_id: str = Field(description="YouTube video ID.")
    youtube_url: str = Field(description="Canonical watch URL.")
    title: str = Field(description="Video title.")
    uploader: str | None = Field(default=None, description="Channel name, when available.")
    language: str = Field(description="Detected spoken language code.")
    language_probability: float = Field(description="Confidence of the detected language.")
    duration: float = Field(description="Video duration in seconds.")
    transcript: str = Field(description="Full joined transcript text.")
    segments: list[TranscriptSegment] = Field(description="Timestamped transcript segments.")
