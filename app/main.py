"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.transcript import router as transcript_router
from app.api.tts import router as tts_router
from app.config import settings
from app.exceptions import AppError
from app.services.chat import ChatService
from app.services.database import Database
from app.services.email import EmailService
from app.services.embeddings import EmbeddingService
from app.services.gemini import GeminiService
from app.services.okf import OKFService
from app.services.pipeline import run_transcription
from app.services.quiz import QuizService
from app.services.rag import RAGService, format_timestamp
from app.services.social_post import SocialPostService
from app.services.summary import SummaryService
from app.services.transcription import TranscriptionService
from app.services.tts import TTSService
from app.services.voice_clone import VoiceCloneService
from app.services.youtube import YouTubeService
from app.web import render_markdown

BASE_DIR: Path = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reject oversized request bodies early. Normal API payloads are a small JSON
# object containing a single URL; the chat speech-to-text endpoint additionally
# accepts recorded microphone audio, which can be several megabytes.
MAX_REQUEST_BODY_BYTES = 64 * 1024
MAX_STT_BODY_BYTES = 16 * 1024 * 1024
STT_ROUTE_PATH = "/api/v1/chat/stt"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown lifecycle.

    Builds the service layer and loads the Whisper and embedding models
    exactly once.
    """
    app.state.youtube_service = YouTubeService(
        temp_dir=settings.temp_dir,
        max_video_duration=settings.max_video_duration,
    )
    transcription_service = TranscriptionService(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    transcription_service.load_model()
    app.state.transcription_service = transcription_service

    database = Database(settings.database_path)
    embeddings = EmbeddingService(settings.embedding_model)
    embeddings.load_model()
    rag = RAGService(
        embeddings,
        top_k=settings.rag_top_k,
        chunk_chars=settings.rag_chunk_chars,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    gemini = GeminiService(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        max_tokens=settings.gemini_max_tokens,
        summary_chars=settings.summary_chars,
    )
    okf = OKFService(
        root_dir=settings.okf_dir,
        gemini=gemini,
        summary_chars=settings.okf_summary_chars,
    )
    summary = SummaryService(database, gemini)
    social_post = SocialPostService(database, gemini)
    quiz = QuizService(database, gemini)
    chat = ChatService(
        database,
        rag,
        gemini,
        okf=okf,
        library_max_videos=settings.library_max_videos,
        library_top_k=settings.library_top_k,
    )

    tts = TTSService(
        voice_model=settings.tts_voice_model,
        voices_dir=settings.tts_voices_dir,
        cache_dir=settings.tts_cache_dir,
        max_chars=settings.tts_max_chars,
        synthesis_timeout_seconds=settings.tts_synthesis_timeout_seconds,
    )
    stale_removed = tts.cleanup_stale()
    if stale_removed:
        logger.info("Removed %d stale TTS audio files", stale_removed)
    if tts.available:
        logger.info("TTS voice model: %s", settings.tts_voice_model.name)
    else:
        logger.warning(
            "TTS voice model not found at %s; the Speak feature is disabled.",
            settings.tts_voice_model,
        )

    app.state.database = database
    app.state.rag = rag
    app.state.gemini = gemini
    app.state.okf = okf
    app.state.chat = chat
    app.state.summary_service = summary
    app.state.social_post_service = social_post
    app.state.quiz_service = quiz
    app.state.tts_service = tts
    app.state.voice_clone_service = VoiceCloneService(
        model_name=settings.voice_clone_model,
        cache_dir=settings.voice_clone_dir,
        device=settings.voice_clone_device,
        max_chars=settings.voice_clone_max_chars,
        synthesis_timeout_seconds=settings.voice_clone_timeout_seconds,
    )
    if app.state.voice_clone_service.available:
        logger.info("Voice cloning enabled (XTTS model: %s)", settings.voice_clone_model)
    else:
        logger.warning(
            "Coqui TTS (coqui-tts) is not installed; the voice-clone feature is disabled."
        )

    email_service = EmailService.from_settings()
    app.state.email_service = email_service
    if email_service.available:
        logger.info("Email verification enabled via %s", email_service.host)
    else:
        logger.warning(
            "SMTP_HOST/SMTP_USERNAME not configured; email verification is disabled "
            "and new accounts are auto-verified."
        )

    app.state.pipeline = run_transcription
    app.state.background_tasks: set[asyncio.Task] = set()
    app.state.active_transcriptions: set[tuple[int, str]] = set()
    app.state.http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))

    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Application started: %s v%s", settings.app_name, settings.app_version)
    if not gemini.available:
        logger.warning("GEMINI_API_KEY is not set; chat features will be disabled.")
    yield

    await app.state.http_client.aclose()
    for task in list(app.state.background_tasks):
        task.cancel()
    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns:
        A fully configured :class:`FastAPI` instance.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Converts YouTube videos into timestamped transcripts using Whisper.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=settings.cors_origin_list != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")
    templates.env.globals["app_title"] = settings.app_name
    templates.env.globals["app_version"] = settings.app_version
    templates.env.filters["markdown"] = render_markdown
    templates.env.filters["format_timestamp"] = format_timestamp
    app.state.templates = templates
    app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")

    @app.middleware("http")
    async def limit_request_body(request: Request, call_next):
        """Reject oversized request bodies early."""
        limit = (
            MAX_STT_BODY_BYTES if request.url.path == STT_ROUTE_PATH else MAX_REQUEST_BODY_BYTES
        )
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > limit:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large."},
                )
        return await call_next(request)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        """Convert application exceptions into JSON error responses."""
        logger.error("Error handling %s: %s", request.url.path, exc.detail)
        return JSONResponse(status_code=exc.http_status, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return a generic response for unexpected errors (no stack traces)."""
        logger.exception("Unhandled error while processing %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Unexpected server error."})

    @app.get(
        "/health",
        tags=["health"],
        summary="Health check",
        description="Returns the API health status.",
        response_model=dict,
    )
    async def health() -> dict:
        """Report service health.

        Returns:
            A dict containing ``{"status": "ok"}``.
        """
        return {"status": "ok"}

    app.include_router(auth_router)
    app.include_router(transcript_router)
    app.include_router(chat_router)
    app.include_router(tts_router)

    return app


app = create_app()
