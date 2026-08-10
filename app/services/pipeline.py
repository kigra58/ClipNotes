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

        okf = getattr(request.app.state, "okf", None)
        if okf is not None:
            try:
                await okf.upsert_video(
                    user_id=user_id,
                    youtube_id=result["video_id"],
                    youtube_url=normalize_youtube_url(youtube_url),
                    title=result["title"],
                    uploader=result["uploader"],
                    duration=result["duration"],
                    transcript=transcription["transcript"],
                )
            except Exception:  # noqa: BLE001 - knowledge layer must not break ingest
                logger.exception("OKF concept update failed for video %s", result["video_id"])

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
