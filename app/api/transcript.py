"""API routes for video transcription (JWT-authenticated)."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.auth import request_user_id
from app.schemas.transcript import TranscribeRequest, TranscribeResponse, TranscriptSegment
from app.services.pipeline import run_transcription

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["transcription"])


def _require_user(request: Request) -> int:
    """Return the authenticated user id or raise 401."""
    user_id = request_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user_id


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe a YouTube video",
    description=(
        "Downloads the audio of a YouTube video, converts it to MP3 and "
        "transcribes it using a Whisper model. Requires authentication via "
        "the access_token cookie or an Authorization: Bearer header. The "
        "resulting video is stored in the caller's library."
    ),
)
async def transcribe(request: Request, body: TranscribeRequest) -> TranscribeResponse:
    """Transcribe the audio of the given YouTube video for the user.

    Args:
        request: The incoming request (provides application services).
        body: The validated request body containing the YouTube URL.

    Returns:
        A :class:`TranscribeResponse` with the transcript and segments.

    Raises:
        HTTPException: If the request is unauthenticated.
    """
    user_id = _require_user(request)
    logger.info("Processing YouTube URL for user %d", user_id)

    result: dict[str, Any] | None = None
    async for event in run_transcription(request, body.youtube_url, user_id):
        if event["type"] == "result":
            result = event
        elif event["type"] == "error":
            raise HTTPException(status_code=event["status"], detail=event["detail"])
    if result is None:
        raise HTTPException(status_code=500, detail="Transcription failed.")

    video_id = result["video_id"]
    database: Any = request.app.state.database
    video = database.get_video(video_id, user_id)
    if video is None:
        raise HTTPException(status_code=500, detail="Stored video could not be loaded.")

    return TranscribeResponse(
        transcript_id=video_id,
        video_id=video["youtube_id"],
        youtube_url=video["youtube_url"],
        title=video["title"],
        uploader=video["uploader"],
        language=video["language"],
        language_probability=video["language_probability"],
        duration=video["duration"],
        transcript=video["transcript"],
        segments=[TranscriptSegment(**segment) for segment in video["segments"]],
    )
