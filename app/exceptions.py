"""Application-level exceptions mapped to HTTP error responses."""


class AppError(Exception):
    """Base class for all application-level errors."""

    http_status: int = 500
    detail: str = "Unexpected error."

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.detail)
        self.detail = detail or self.detail


class InvalidURLError(AppError):
    """Raised when the provided URL is not a valid YouTube URL."""

    http_status = 400
    detail = "Invalid YouTube URL."


class VideoUnavailableError(AppError):
    """Raised when the YouTube video cannot be found or is unavailable."""

    http_status = 404
    detail = "YouTube video is unavailable."


class VideoTooLongError(AppError):
    """Raised when the video exceeds the configured maximum duration."""

    http_status = 400
    detail = "Video duration exceeds the maximum allowed duration."


class DownloadError(AppError):
    """Raised when audio download or conversion fails."""

    http_status = 502
    detail = "Failed to download video audio."


class TranscriptionError(AppError):
    """Raised when audio transcription fails."""

    http_status = 500
    detail = "Failed to transcribe audio."
