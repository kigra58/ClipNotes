"""Shared transcription pipeline used by the web action and the JSON API."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.config import settings
from app.exceptions import AppError
from app.utils.youtube import extract_video_id, normalize_youtube_url

logger = logging.getLogger(__name__)


async def run_transcription(
    request: Any, youtube_url: str, user_id: int
) -> AsyncIterator[dict[str, Any]]:
    """Transcribe a YouTube video and store it for the given user.

    Yields event dicts so callers can stream progress to the client:
    ``{"type": "progress", "message": "...", "percent": 0-100}`` followed by a
    final ``{"type": "result", "video_id": int}`` or
    ``{"type": "error", "detail": str, "status": int}``.

    The Whisper pass runs in a worker thread and reports per-chunk progress,
    which is forwarded to the client as a steady stream of progress events.

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
        yield {
            "type": "progress",
            "message": "Fetching video metadata…",
            "percent": 1,
        }
        metadata = await asyncio.to_thread(youtube_service.get_metadata, youtube_url)

        yield {
            "type": "progress",
            "message": f"Downloading audio of {metadata['title']}…",
            "percent": 4,
        }
        result = await asyncio.to_thread(youtube_service.download_audio, youtube_url)

        yield {
            "type": "progress",
            "message": "Preparing audio for transcription…",
            "percent": 5,
        }
        transcription: dict[str, Any] | None = None
        async for event in _transcribe_with_progress(
            transcription_service, result["audio_path"]
        ):
            if event["type"] == "transcription":
                transcription = event["transcription"]
            else:
                yield event
        assert transcription is not None

        yield {
            "type": "progress",
            "message": "Building searchable chunks…",
            "percent": 92,
        }
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in transcription["segments"]
        ]
        chunks = rag.build_chunks(segments)

        yield {"type": "progress", "message": "Saving video…", "percent": 97}
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


async def _transcribe_with_progress(
    transcription_service: Any, audio_path: str
) -> AsyncIterator[dict[str, Any]]:
    """Run chunked Whisper transcription, streaming progress as it goes.

    The heavy work runs in a worker thread and pushes ``(message, percent)``
    tuples into an ``asyncio.Queue``; this async generator drains the queue and
    yields them as progress events, then yields one final
    ``{"type": "transcription", "transcription": {...}}`` event with the
    result.

    Args:
        transcription_service: The transcription service.
        audio_path: Path to the downloaded audio file.

    Yields:
        Progress events followed by the final transcription result event.
    """
    chunk_seconds = settings.whisper_chunk_seconds
    overlap_seconds = settings.whisper_chunk_overlap_seconds

    queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()

    def _on_progress(message: str, percent: float) -> None:
        queue.put_nowait((message, percent))

    task = asyncio.create_task(
        asyncio.to_thread(
            transcription_service.transcribe_stream,
            audio_path,
            _on_progress,
            chunk_seconds,
            overlap_seconds,
        )
    )

    while not task.done() or not queue.empty():
        try:
            message, percent = queue.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.05)
            continue
        yield {"type": "progress", "message": message, "percent": percent}

    yield {"type": "transcription", "transcription": task.result()}


async def background_transcription(
    request: Any, youtube_url: str, user_id: int, video_id: int
) -> None:
    """Run :func:`run_transcription` as a background task.

    Each progress event is persisted to the pending video row so the
    dashboard's 2s poller can render a live progress bar. The pipeline saves
    the finished video (marking it ``ready``) on success; on failure the
    pending row is marked ``error`` so the dashboard can show why instead of
    leaving the video stuck ``processing``.

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
            if event["type"] == "progress" and "percent" in event:
                database.update_video_progress(
                    video_id=video_id,
                    percent=round(event["percent"], 1),
                    message=event.get("message") or "",
                )
            elif event["type"] == "error":
                error = event["detail"]
    except Exception as exc:  # noqa: BLE001 - surface unexpected failures
        logger.exception("Background transcription failed for video %d", video_id)
        error = f"Unexpected server error: {exc}"
    finally:
        if error is not None:
            database.mark_video_status(video_id=video_id, status="error", error=error)
            logger.error("Background transcription of video %d failed: %s", video_id, error)
