"""Shared transcription pipeline used by the web action and the JSON API."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.exceptions import AppError
from app.utils.youtube import extract_video_id, normalize_youtube_url

logger = logging.getLogger(__name__)


async def run_transcription(
    request: Any, youtube_url: str, user_id: int
) -> AsyncIterator[dict[str, Any]]:
    """Transcribe a YouTube video and store it for the given user.

    Yields event dicts so callers can stream progress to the client:
    ``{"type": "progress", "message": "..."}`` followed by a final
    ``{"type": "result", "video_id": int}`` or
    ``{"type": "error", "detail": str, "status": int}``.

    Args:
        request: The request (provides application services).
        youtube_url: The YouTube URL to transcribe.
        user_id: The user who owns the resulting video.

    Yields:
        Progress and terminal result/error events.
    """
    youtube_service = request.app.state.youtube_service
    transcription_service = request.app.state.transcription_service
    database = request.app.state.database
    rag = request.app.state.rag

    video_id = extract_video_id(youtube_url)
    logger.info("Transcribing video %s for user %d", video_id, user_id)

    try:
        yield {"type": "progress", "message": "Fetching video metadata…"}
        metadata = await asyncio.to_thread(youtube_service.get_metadata, youtube_url)

        yield {"type": "progress", "message": f"Downloading audio of {metadata['title']}…"}
        result = await asyncio.to_thread(youtube_service.download_audio, youtube_url)

        yield {"type": "progress", "message": "Transcribing audio with Whisper…"}
        transcription = await asyncio.to_thread(
            transcription_service.transcribe, result["audio_path"]
        )

        yield {"type": "progress", "message": "Building searchable chunks…"}
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in transcription["segments"]
        ]
        chunks = rag.build_chunks(segments)

        stored_id = database.save_video(
            user_id=user_id,
            youtube_id=result["video_id"],
            youtube_url=normalize_youtube_url(youtube_url),
            title=result["title"],
            uploader=result["uploader"],
            language=transcription["language"],
            language_probability=transcription["language_probability"],
            duration=result["duration"],
            transcript=transcription["transcript"],
            segments=segments,
            chunks=chunks,
        )
        logger.info("Stored video %d for user %d", stored_id, user_id)
        yield {"type": "result", "video_id": stored_id}
    except AppError as exc:
        logger.error("Transcription failed for video %s: %s", video_id, exc.detail)
        yield {"type": "error", "detail": exc.detail, "status": exc.http_status}
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures
        logger.exception("Unexpected transcription failure for video %s", video_id)
        yield {
            "type": "error",
            "detail": f"Unexpected server error: {exc}",
            "status": 500,
        }
    finally:
        youtube_service.cleanup(video_id)


async def background_transcription(
    request: Any, youtube_url: str, user_id: int, video_id: int
) -> None:
    """Run :func:`run_transcription` as a background task.

    The pipeline saves the finished video (marking it ``ready``) on success.
    On failure the pending row created by the caller is marked ``error`` so the
    dashboard can show why instead of leaving the video stuck ``processing``.

    Args:
        request: The request (provides application services).
        youtube_url: The YouTube URL to transcribe.
        user_id: The user who owns the resulting video.
        video_id: The pending video row to mark on failure.
    """
    database = request.app.state.database
    error: str | None = None
    try:
        async for event in run_transcription(request, youtube_url, user_id):
            if event["type"] == "error":
                error = event["detail"]
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures
        logger.exception("Background transcription failed for video %d", video_id)
        error = f"Unexpected server error: {exc}"
    finally:
        if error is not None:
            database.mark_video_status(video_id=video_id, status="error", error=error)
            logger.error("Background transcription of video %d failed: %s", video_id, error)
