"""Web UI, Datastar actions and JSON API for text-to-speech (Piper).

The web side is a single ``/speak`` page where a user pastes a transcript,
sees a quick analysis (word count, estimated reading time) and plays the
synthesized audio. The JSON API exposes the same capability programmatically:
``POST /api/v1/tts/analyze`` returns statistics and ``POST /api/v1/tts``
returns a WAV file.
"""

import asyncio
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from datastar_py.fastapi import read_signals
from datastar_py.sse import ServerSentEventGenerator as SSE
from datastar_py.starlette import DatastarResponse
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from app.auth import request_user_id
from app.datastar import datastar_action
from app.schemas.tts import AnalyzeResponse, TextInput
from app.services.tts import analyze_text
from app.web import current_user, login_page_events, render_page

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tts"])

_AUDIO_NAME_RE = re.compile(r"^(\d+)_[0-9a-f]{32}\.wav$")


def _login_redirect(request: Request) -> RedirectResponse | DatastarResponse:
    """Redirect or swap to the login page for an anonymous visitor."""
    if request.headers.get("datastar-request"):
        return DatastarResponse(tuple(login_page_events(request)))
    return RedirectResponse(url="/login", status_code=303)


def _require_user(request: Request) -> int:
    """Return the authenticated user id or raise 401."""
    user_id = request_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user_id


def _find_audio_path(request: Request, name: str) -> Path:
    """Resolve a cached WAV clip, checking ownership of the requesting user."""
    user_id = _require_user(request)
    match = _AUDIO_NAME_RE.fullmatch(name)
    if match is None or int(match.group(1)) != user_id:
        raise HTTPException(status_code=404, detail="Audio not found.")

    path = request.app.state.tts_service.cache_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found.")
    return path


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@router.get("/speak", response_class=HTMLResponse, summary="Speak a transcript")
async def speak_page(request: Request) -> HTMLResponse:
    """Render the speak page, optionally prefilled with a stored transcript."""
    user = current_user(request)
    if user is None:
        return _login_redirect(request)

    transcript = ""
    video_id = request.query_params.get("video_id")
    try:
        video_id = int(video_id) if video_id else None
    except ValueError:
        video_id = None
    if video_id is not None:
        database = request.app.state.database
        video = database.get_video(video_id, user["id"])
        if video is not None:
            transcript = video["transcript"] or ""

    voice_clone = getattr(request.app.state, "voice_clone_service", None)
    cloned_voices = (
        request.app.state.database.list_voice_profiles(user["id"])
        if voice_clone is not None and voice_clone.available
        else []
    )
    registered_voices = (
        voice_clone.list_registered_voices()
        if voice_clone is not None and voice_clone.available
        else []
    )
    preselected_voice = request.query_params.get("voice")

    return render_page(
        request,
        "speak.html",
        {
            "user": user,
            "transcript": transcript,
            "video_id": video_id,
            "max_chars": request.app.state.tts_service.max_chars,
            "tts_available": request.app.state.tts_service.available,
            "voices": request.app.state.tts_service.list_voices(),
            "default_voice": request.app.state.tts_service.voice,
            "voices_dir": str(request.app.state.tts_service.voices_dir),
            "cloned_voices": cloned_voices,
            "registered_voices": registered_voices,
            "voice_clone_available": bool(
                voice_clone is not None and voice_clone.available
            ),
            "preselected_voice": preselected_voice,
        },
    )


# ---------------------------------------------------------------------------
# Datastar action
# ---------------------------------------------------------------------------


@router.post("/speak", summary="Analyze and speak a transcript (Datastar)")
@datastar_action
async def speak_action(request: Request):
    """Analyze the submitted text, synthesize it and reveal the audio player."""
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    tts = request.app.state.tts_service
    if not tts.available:
        return SSE.patch_signals(
            {
                "tts_status": "",
                "tts_error": "Speech synthesis is not available. Check the TTS voice model.",
            }
        )

    signals = await read_signals(request) or {}
    text = (signals.get("tts_text") or "").strip()
    if not text:
        return SSE.patch_signals(
            {"tts_status": "", "tts_error": "Enter some text to speak first."}
        )

    tts = request.app.state.tts_service
    voice = str(signals.get("voice") or "").strip()
    profile: dict[str, Any] | None = None
    registered_voice: str | None = None
    voice_clone = getattr(request.app.state, "voice_clone_service", None)
    if voice.startswith("clone_"):
        try:
            profile_id = int(voice.split("_", 1)[1])
        except ValueError:
            profile_id = None
        profile = (
            request.app.state.database.get_voice_profile(
                profile_id=profile_id, user_id=user["id"]
            )
            if profile_id is not None
            else None
        )
        if profile is None:
            return SSE.patch_signals(
                {"tts_status": "", "tts_error": "Voice profile not found."}
            )
        if voice_clone is None or not voice_clone.available:
            return SSE.patch_signals(
                {
                    "tts_status": "",
                    "tts_error": "Voice cloning is not available on this server.",
                }
            )
    elif voice.startswith("registered_"):
        registered_voice = voice.split("_", 1)[1]
        if voice_clone is None or not voice_clone.available:
            return SSE.patch_signals(
                {
                    "tts_status": "",
                    "tts_error": "Voice cloning is not available on this server.",
                }
            )
        if not registered_voice:
            return SSE.patch_signals(
                {"tts_status": "", "tts_error": "Registered voice not found."}
            )
    elif not tts.available:
        return SSE.patch_signals(
            {
                "tts_status": "",
                "tts_error": "Speech synthesis is not available. Check the TTS voice model.",
            }
        )
    elif voice:
        try:
            tts.set_voice(voice)
        except ValueError as exc:
            return SSE.patch_signals({"tts_status": "", "tts_error": str(exc)})

    synthesizer = voice_clone if profile is not None or registered_voice else tts
    timeout = synthesizer.synthesis_timeout_seconds

    async def stream() -> Any:
        voice_label = profile["name"] if profile is not None else (registered_voice or tts.voice)
        yield SSE.patch_signals(
            {
                "tts_url": "",
                "tts_mp3_url": "",
                "tts_status": "Synthesizing speech… this can take a moment for long transcripts.",
                "tts_error": "",
                "tts_voice": voice_label,
            }
        )
        try:
            if profile is not None:
                audio_path = await asyncio.wait_for(
                    asyncio.to_thread(
                        voice_clone.cache_audio,
                        text,
                        profile,
                        user["id"],
                        tts.cache_dir,
                    ),
                    timeout=timeout,
                )
            elif registered_voice is not None:
                audio_path = await asyncio.wait_for(
                    asyncio.to_thread(
                        voice_clone.cache_registered_audio,
                        text,
                        registered_voice,
                        user["id"],
                        tts.cache_dir,
                    ),
                    timeout=timeout,
                )
            else:
                audio_path = await asyncio.wait_for(
                    asyncio.to_thread(tts.cache_audio, text, user["id"]),
                    timeout=timeout,
                )
        except asyncio.TimeoutError:
            logger.error("Speech synthesis timed out for user %d", user["id"])
            yield SSE.patch_signals(
                {
                    "tts_status": "",
                    "tts_error": (
                        "Speech synthesis timed out. Try a shorter transcript "
                        "or split it into smaller pieces."
                    ),
                }
            )
            return
        except ValueError as exc:
            yield SSE.patch_signals({"tts_status": "", "tts_error": str(exc)})
            return
        except Exception:  # noqa: BLE001 - surface as a friendly error
            logger.exception("Speech synthesis failed for user %d", user["id"])
            yield SSE.patch_signals(
                {"tts_status": "", "tts_error": "Speech synthesis failed. Try again."}
            )
            return

        stats = analyze_text(text)
        yield SSE.patch_signals(
            {
                "tts_stats": stats,
                "tts_url": f"/tts/audio/{audio_path.name}",
                "tts_mp3_url": f"/tts/audio/{audio_path.name}/mp3",
                "tts_status": "Ready — press play to listen.",
                "tts_error": "",
            }
        )

    return stream()


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/tts/analyze",
    response_model=AnalyzeResponse,
    tags=["api"],
    summary="Analyze text",
    description=(
        "Returns statistics for a body of text (character/word/sentence counts "
        "and an estimated reading time). Requires authentication."
    ),
)
async def analyze(request: Request, body: TextInput) -> AnalyzeResponse:
    """Analyze the submitted text without synthesizing audio."""
    _require_user(request)
    return AnalyzeResponse(**analyze_text(body.text.strip()))


@router.post(
    "/api/v1/tts",
    tags=["api"],
    summary="Speak text",
    description=(
        "Synthesizes the submitted text with the Piper voice and returns a "
        "16-bit mono WAV file. Requires authentication."
    ),
)
async def speak(request: Request, body: TextInput) -> Response:
    """Synthesize the submitted text and return the audio as WAV bytes."""
    _require_user(request)
    tts = request.app.state.tts_service
    if not tts.available:
        raise HTTPException(status_code=503, detail="Speech synthesis is not available.")

    try:
        wav_bytes, duration = await asyncio.wait_for(
            asyncio.to_thread(tts.synthesize_bytes, body.text),
            timeout=tts.synthesis_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        logger.error("Speech synthesis timed out for API request")
        raise HTTPException(status_code=504, detail="Speech synthesis timed out.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Speech synthesis failed for API request")
        raise HTTPException(status_code=500, detail="Speech synthesis failed.") from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Duration-Seconds": str(duration)},
    )


@router.get(
    "/tts/audio/{name}",
    summary="Stream a synthesized audio clip",
    description="Serves a cached WAV clip; only the user who requested it can fetch it.",
)
async def stream_audio(request: Request, name: str) -> FileResponse:
    """Stream a cached WAV clip, checking ownership of the requesting user."""
    path = _find_audio_path(request, name)
    return FileResponse(path, media_type="audio/wav")


@router.get(
    "/tts/audio/{name}/mp3",
    summary="Download a synthesized clip as MP3",
    description=(
        "Converts the cached WAV clip to MP3 (via ffmpeg) and downloads it; "
        "only the user who requested it can fetch it."
    ),
)
async def download_mp3(request: Request, name: str) -> FileResponse:
    """Convert a cached WAV clip to MP3 and return it as a download."""
    wav_path = _find_audio_path(request, name)
    try:
        mp3_path = await asyncio.to_thread(request.app.state.tts_service.to_mp3, wav_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        logger.error("ffmpeg MP3 conversion failed for %s: %s", name, stderr)
        raise HTTPException(status_code=500, detail="MP3 conversion failed.") from exc
    return FileResponse(mp3_path, media_type="audio/mpeg", filename="transcript.mp3")
