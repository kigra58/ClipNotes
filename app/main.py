"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.transcript import router as transcript_router
from app.config import settings
from app.exceptions import AppError
from app.services.transcription import TranscriptionService
from app.services.youtube import YouTubeService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reject request bodies larger than 64 KB. The only payload accepted by this
# API is a small JSON object containing a single URL.
MAX_REQUEST_BODY_BYTES = 64 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup/shutdown lifecycle.

    Builds the service layer and loads the Whisper model exactly once.
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

    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Application started: %s v%s", settings.app_name, settings.app_version)
    yield
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

    @app.middleware("http")
    async def limit_request_body(request: Request, call_next):
        """Reject oversized request bodies early."""
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
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

    app.include_router(transcript_router)

    return app


app = create_app()
