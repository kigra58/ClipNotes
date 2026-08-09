"""Web UI pages and the streaming WebSocket chat endpoint."""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse

from app.services.rag import format_timestamp

logger = logging.getLogger(__name__)

router = APIRouter(tags=["web"])


def _template(request: Request, name: str, context: dict[str, Any]) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(request, name, context)


@router.get("/", response_class=HTMLResponse, summary="Home page")
async def home(request: Request) -> HTMLResponse:
    """Render the home page with the transcribe form and transcript list."""
    database = request.app.state.database
    chat = request.app.state.chat
    return _template(
        request,
        "home.html",
        {
            "transcripts": database.list_transcripts(),
            "chat_available": chat.available,
        },
    )


@router.post("/", response_class=RedirectResponse, summary="Transcribe from the web")
async def home_transcribe(request: Request) -> RedirectResponse:
    """Accept a YouTube URL from the home form and redirect to its transcript."""
    form = await request.form()
    youtube_url = (form.get("youtube_url") or "").strip()
    if not youtube_url:
        return RedirectResponse(url="/", status_code=303)

    http_client = request.app.state.http_client
    response = await http_client.post(
        "/api/v1/transcribe",
        json={"youtube_url": youtube_url},
    )
    if response.status_code != 200:
        logger.error(
            "Transcription failed with status %s: %s",
            response.status_code,
            response.text,
        )
        return RedirectResponse(url="/?error=1", status_code=303)

    transcript_id = response.json().get("transcript_id")
    return RedirectResponse(url=f"/transcripts/{transcript_id}", status_code=303)


@router.get("/transcripts/{transcript_id}", response_class=HTMLResponse, summary="Transcript page")
async def transcript_page(request: Request, transcript_id: int) -> HTMLResponse:
    """Render a single stored transcript with its timestamped segments."""
    database = request.app.state.database
    transcript = database.get_transcript(transcript_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found.")

    segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "start_label": format_timestamp(segment["start"]),
            "end_label": format_timestamp(segment["end"]),
        }
        for segment in transcript["segments"]
    ]
    return _template(
        request,
        "transcript.html",
        {
            "transcript": {**transcript, "segments": segments},
            "duration_label": format_timestamp(transcript["duration"]),
            "chat_available": request.app.state.chat.available,
        },
    )


@router.get("/chat/{transcript_id}", response_class=HTMLResponse, summary="Chat page")
async def chat_page(request: Request, transcript_id: int) -> HTMLResponse:
    """Render the chat interface for a stored transcript."""
    database = request.app.state.database
    transcript = database.get_transcript(transcript_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found.")

    return _template(
        request,
        "chat.html",
        {
            "transcript_id": transcript_id,
            "transcript": transcript,
            "chat_available": request.app.state.chat.available,
        },
    )


@router.websocket("/ws/chat/{transcript_id}")
async def chat_websocket(websocket: WebSocket, transcript_id: int) -> None:
    """Stream chat answers for a transcript over a WebSocket.

    The client sends ``{"message": "..."}``; the server replies with events
    of type ``sources``, ``token``, ``done`` or ``error``.
    """
    await websocket.accept()
    chat = websocket.app.state.chat

    if not chat.available:
        await websocket.send_json(
            {"type": "error", "data": "Gemini API key is not configured. Add GEMINI_API_KEY to .env and restart."}
        )
        await websocket.close()
        return

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "data": "Invalid JSON message."})
                continue

            question = (payload.get("message") or "").strip()
            if not question:
                continue

            async for event in chat.stream_answer(transcript_id, question):
                await websocket.send_json(event)
                if event["type"] in ("done", "error"):
                    break
    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected (transcript %s)", transcript_id)
    except Exception:  # noqa: BLE001 - keep the socket alive across failures
        logger.exception("Unexpected WebSocket error (transcript %s)", transcript_id)
        try:
            await websocket.send_json({"type": "error", "data": "Unexpected server error."})
        except Exception:  # noqa: BLE001 - socket may already be closed
            pass
