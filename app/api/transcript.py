"""API routes for video transcription."""

import logging
from typing import Any

from fastapi import APIRouter, Request

from app.schemas.transcript import TranscribeRequest, TranscribeResponse, TranscriptSegment
from app.utils.youtube import extract_video_id, normalize_youtube_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["transcription"])


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe a YouTube video",
    description=(
        "Downloads the audio of a YouTube video, converts it to MP3 and "
        "transcribes it using a Whisper model. Returns the detected language, "
        "the full transcript and timestamped segments."
    ),
)
async def transcribe(request: Request, body: TranscribeRequest) -> TranscribeResponse:
    """Transcribe the audio of the given YouTube video.

    Args:
        request: The incoming request (provides application services).
        body: The validated request body containing the YouTube URL.

    Returns:
        A :class:`TranscribeResponse` with the transcript and segments.
    """
    logger.info("Processing YouTube URL")

    youtube_service: Any = request.app.state.youtube_service
    transcription_service: Any = request.app.state.transcription_service
    database: Any = request.app.state.database
    rag: Any = request.app.state.rag

    video_id = extract_video_id(body.youtube_url)
    logger.info("Video ID: %s", video_id)

    try:
        logger.info("Fetching video metadata")
        metadata = youtube_service.get_metadata(body.youtube_url)
        logger.info("Video metadata: title=%r duration=%.1fs", metadata["title"], metadata["duration"])

        logger.info("Downloading audio")
        result = youtube_service.download_audio(body.youtube_url)
        logger.info("Audio downloaded")

        logger.info("Starting transcription")
        transcription = transcription_service.transcribe(result["audio_path"])
    finally:
        youtube_service.cleanup(video_id)
        logger.info("Temporary file removed")

    segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in transcription["segments"]]
    chunks = rag.build_chunks(segments)
    transcript_id = database.save_transcript(
        video_id=result["video_id"],
        youtube_url=normalize_youtube_url(body.youtube_url),
        title=result["title"],
        uploader=result["uploader"],
        language=transcription["language"],
        language_probability=transcription["language_probability"],
        duration=result["duration"],
        transcript=transcription["transcript"],
        segments=segments,
        chunks=chunks,
    )

    return TranscribeResponse(
        transcript_id=transcript_id,
        video_id=result["video_id"],
        youtube_url=normalize_youtube_url(body.youtube_url),
        title=result["title"],
        uploader=result["uploader"],
        language=transcription["language"],
        language_probability=transcription["language_probability"],
        duration=result["duration"],
        transcript=transcription["transcript"],
        segments=[TranscriptSegment(**segment) for segment in segments],
    )
