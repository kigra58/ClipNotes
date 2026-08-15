"""Web UI pages and Datastar actions (dashboard, videos, categories, chat).

All interactive behavior is driven by Datastar: forms submit via
``data-on:submit="@post(...)"`` and the server responds with SSE events
(:class:`datastar_py.sse.ServerSentEventGenerator`).
"""

import asyncio
import html
import logging
import uuid
from pathlib import Path
from typing import Any

from datastar_py.fastapi import read_signals
from datastar_py.sse import ServerSentEventGenerator as SSE
from datastar_py.starlette import DatastarResponse
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.config import settings
from app.datastar import datastar_action
from app.exceptions import AppError
from app.services.pipeline import background_transcription
from app.services.voice_clone import make_reference_clip, pick_sample_window
from app.utils.youtube import extract_video_id, normalize_youtube_url
from app.web import (
    chat_context,
    current_user,
    home_context,
    is_datastar_request,
    library_context,
    login_page_events,
    page_events,
    render_fragment,
    render_page,
    search_context,
    video_context,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["web"])


def _login_redirect(request: Request) -> RedirectResponse | DatastarResponse:
    if is_datastar_request(request):
        return DatastarResponse(tuple(login_page_events(request)))
    return RedirectResponse(url="/login", status_code=303)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse, summary="Dashboard")
async def home(request: Request) -> HTMLResponse:
    """Render the user's dashboard: transcribe form, categories and videos."""
    user = current_user(request)
    if user is None:
        return _login_redirect(request)

    return render_page(request, "home.html", home_context(request, user))


@router.get("/videos/{video_id}", response_class=HTMLResponse, summary="Transcript page")
async def video_page(request: Request, video_id: int) -> HTMLResponse:
    """Render a single stored video with its timestamped segments."""
    user = current_user(request)
    if user is None:
        return _login_redirect(request)

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found.")

    return render_page(request, "video.html", video_context(request, user, video_id))


@router.get("/videos/{video_id}/chat", response_class=HTMLResponse, summary="Chat page")
async def chat_page(request: Request, video_id: int) -> HTMLResponse:
    """Render the chat interface for a stored video."""
    user = current_user(request)
    if user is None:
        return _login_redirect(request)

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found.")

    conversation_id = request.query_params.get("conversation")
    try:
        conversation_id = int(conversation_id) if conversation_id else None
    except ValueError:
        conversation_id = None

    return render_page(
        request,
        "chat.html",
        chat_context(request, user, video_id, conversation_id),
    )


@router.get("/library", response_class=HTMLResponse, summary="Library chat page")
async def library_page(request: Request) -> HTMLResponse:
    """Render the chat interface that spans all of the user's videos."""
    user = current_user(request)
    if user is None:
        return _login_redirect(request)

    conversation_id = request.query_params.get("conversation")
    try:
        conversation_id = int(conversation_id) if conversation_id else None
    except ValueError:
        conversation_id = None

    return render_page(
        request,
        "library.html",
        library_context(request, user, conversation_id),
    )


@router.get("/search", response_class=HTMLResponse, summary="Search transcripts")
async def search_page(request: Request) -> HTMLResponse:
    """Render the library-wide full-text search page.

    The query is read from the ``q`` query parameter (plain form GET) or from
    the Datastar ``datastar`` signal payload (header/live search @get).
    """
    user = current_user(request)
    if user is None:
        return _login_redirect(request)

    signals = await read_signals(request) or {}
    q = (signals.get("q") or request.query_params.get("q") or "").strip()
    return render_page(request, "search.html", search_context(request, user, q))


# ---------------------------------------------------------------------------
# Datastar actions
#
# Each action is a plain async function: signals are read eagerly (reading the
# request body after a streaming response has started is not safe) and the
# returned events/generators are wrapped in a Datastar SSE response by
# ``@datastar_action``.
# ---------------------------------------------------------------------------


@router.post("/transcribe", summary="Transcribe a YouTube video (Datastar)")
@datastar_action
async def transcribe_action(request: Request):
    """Queue a transcription in the background and re-render the dashboard.

    The heavy pipeline runs as an ``asyncio`` task so the user can keep using
    the app. A pending video card is shown immediately and the client polls
    :func:`videos_panel_fragment` until the card flips to ``ready``.
    """
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    signals = await read_signals(request) or {}
    youtube_url = (signals.get("youtube_url") or "").strip()
    if not youtube_url:
        return SSE.patch_elements(
            elements='<p class="status-error">Enter a YouTube URL first.</p>',
            selector="#transcribe-status",
            mode="inner",
        )

    try:
        youtube_id = extract_video_id(youtube_url)
    except AppError as exc:
        return SSE.patch_elements(
            elements=f"<p class='status-error'>{html.escape(exc.detail)}</p>",
            selector="#transcribe-status",
            mode="inner",
        )

    database = request.app.state.database
    existing = database.find_video_by_youtube_id(
        user_id=user["id"], youtube_id=youtube_id
    )
    if existing is not None and existing["status"] == "ready":
        return tuple(
            page_events(
                request,
                "video.html",
                video_context(request, user, existing["id"]),
                url=f"/videos/{existing['id']}",
            )
        )

    active = getattr(request.app.state, "active_transcriptions", None)
    if active is None:
        active = set()
        request.app.state.active_transcriptions = active
    key = (user["id"], youtube_id)
    if key in active:
        return SSE.patch_elements(
            elements=(
                "<p class='status-progress'>Already transcribing this video "
                "in the background.</p>"
            ),
            selector="#transcribe-status",
            mode="inner",
        )

    active.add(key)
    video_id = database.create_pending_video(
        user_id=user["id"],
        youtube_id=youtube_id,
        youtube_url=normalize_youtube_url(youtube_url),
        title="Transcribing…",
    )

    tasks = getattr(request.app.state, "background_tasks", None)
    if tasks is None:
        tasks = set()
        request.app.state.background_tasks = tasks

    async def _run() -> None:
        try:
            await background_transcription(request, youtube_url, user["id"], video_id)
        except Exception:  # noqa: BLE001 - mark the row failed
            logger.exception("Unexpected background transcription failure")
            database.mark_video_status(
                video_id=video_id, status="error", error="Unexpected server error."
            )
        finally:
            active.discard(key)

    task = asyncio.create_task(_run())
    tasks.add(task)
    task.add_done_callback(tasks.discard)

    context = home_context(request, user)
    return (
        SSE.patch_elements(
            elements=(
                "<p class='status-progress'>Transcribing in the background. "
                "You can keep using the app.</p>"
            ),
            selector="#transcribe-status",
            mode="inner",
        ),
        SSE.patch_elements(
            elements=render_fragment(request, "_videos.html", **context),
            selector="#videos-panel",
            mode="outer",
        ),
        SSE.patch_signals({"youtube_url": "", "transcribing": False}),
    )


@router.get(
    "/fragments/videos-panel",
    response_class=HTMLResponse,
    summary="Videos panel fragment (background poller)",
)
async def videos_panel_fragment(request: Request) -> HTMLResponse:
    """Re-render the dashboard videos panel for the background poller.

    Plain requests (browser, debugging) receive the raw HTML fragment; Datastar
    requests receive an SSE patch that morphs ``#videos-panel`` in place.
    """
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")

    context = home_context(request, user)
    fragment = render_fragment(request, "_videos.html", **context)
    if is_datastar_request(request):
        return DatastarResponse(
            (SSE.patch_elements(elements=fragment, selector="#videos-panel", mode="outer"),)
        )
    return HTMLResponse(fragment)


@router.get(
    "/fragments/search-results",
    response_class=HTMLResponse,
    summary="Search results fragment (live search)",
)
async def search_results_fragment(request: Request) -> HTMLResponse:
    """Re-render the library search results for the live-search input.

    Plain requests (browser, debugging) receive the raw HTML fragment; Datastar
    requests receive an SSE patch that morphs ``#search-results`` in place.
    """
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")

    signals = await read_signals(request) or {}
    q = (signals.get("q") or request.query_params.get("q") or "").strip()
    fragment = render_fragment(request, "_search_results.html", **search_context(request, user, q))
    if is_datastar_request(request):
        return DatastarResponse(
            (SSE.patch_elements(elements=fragment, selector="#search-results", mode="outer"),)
        )
    return HTMLResponse(fragment)


@router.post("/categories", summary="Create a category (Datastar)")
@datastar_action
async def create_category_action(request: Request):
    """Create a category and re-render the category list."""
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    signals = await read_signals(request) or {}
    name = (signals.get("category_name") or "").strip()
    database = request.app.state.database
    if name:
        try:
            database.create_category(user_id=user["id"], name=name)
        except Exception:  # noqa: BLE001 - duplicate name
            pass

    categories = database.list_categories(user["id"])
    return (
        SSE.patch_elements(
            elements=render_fragment(request, "_categories.html", categories=categories),
            selector="#category-list",
            mode="outer",
        ),
        SSE.patch_signals({"category_name": ""}),
    )


@router.post("/categories/{category_id}/delete", summary="Delete a category (Datastar)")
@datastar_action
async def delete_category_action(request: Request, category_id: int):
    """Delete a category and re-render the category list."""
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    database.delete_category(category_id=category_id, user_id=user["id"])
    categories = database.list_categories(user["id"])
    return SSE.patch_elements(
        elements=render_fragment(request, "_categories.html", categories=categories),
        selector="#category-list",
        mode="outer",
    )


@router.post("/categories/{category_id}/rename", summary="Rename a category (Datastar)")
@datastar_action
async def rename_category_action(request: Request, category_id: int):
    """Rename a category and re-render the category list."""
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    signals = await read_signals(request) or {}
    name = (signals.get(f"rename_name_{category_id}") or "").strip()
    database = request.app.state.database
    if name:
        try:
            database.rename_category(category_id=category_id, user_id=user["id"], name=name)
        except Exception:  # noqa: BLE001 - duplicate name
            pass

    categories = database.list_categories(user["id"])
    return SSE.patch_elements(
        elements=render_fragment(request, "_categories.html", categories=categories),
        selector="#category-list",
        mode="outer",
    )


@router.post("/videos/{video_id}/category", summary="Assign a category (Datastar)")
@datastar_action
async def set_video_category_action(request: Request, video_id: int):
    """Assign, clear or create-and-assign a video's category.

    Accepts a ``category_id`` signal (existing category or ``0`` to clear) and
    an optional ``new_category_name`` signal. When a new name is given it is
    created (reusing an existing category with the same name) and assigned.
    When the ``modal`` signal is set (dashboard card modal) the videos panel
    and category list are re-rendered; otherwise the details page shows a
    confirmation note.
    """
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    if video is None:
        return tuple(page_events(request, "home.html", home_context(request, user), url="/"))

    signals = await read_signals(request) or {}
    category_id = signals.get("category_id")
    try:
        category_id = int(category_id) if category_id not in (None, "", 0) else None
    except (TypeError, ValueError):
        category_id = None

    new_name = (signals.get("new_category_name") or "").strip()
    if new_name:
        try:
            category_id = database.create_category(user_id=user["id"], name=new_name)
        except Exception:  # noqa: BLE001 - duplicate name; reuse the existing row
            existing = next(
                (
                    c
                    for c in database.list_categories(user["id"])
                    if c["name"].lower() == new_name.lower()
                ),
                None,
            )
            if existing is not None:
                category_id = existing["id"]

    database.set_video_category(video_id=video_id, user_id=user["id"], category_id=category_id)

    if signals.get("modal"):
        context = home_context(request, user)
        return (
            SSE.patch_elements(
                elements=render_fragment(request, "_videos.html", **context),
                selector="#videos-panel",
                mode="outer",
            ),
            SSE.patch_elements(
                elements=render_fragment(
                    request, "_categories.html", categories=context["categories"]
                ),
                selector="#category-list",
                mode="outer",
            ),
        )

    return (
        SSE.patch_elements(
            elements=render_fragment(
                request,
                "_video_category_select.html",
                video=database.get_video(video_id, user["id"]),
                categories=database.list_categories(user["id"]),
            ),
            selector="#video-category-select",
            mode="outer",
        ),
        SSE.patch_elements(
            elements="<span class='saved-note'>Saved</span>",
            selector="#category-saved",
            mode="inner",
        ),
        SSE.patch_signals({"new_category_name": ""}),
    )


@router.post("/videos/{video_id}/summary", summary="Generate AI summary (Datastar)")
@datastar_action
async def generate_summary_action(request: Request, video_id: int):
    """Generate (or regenerate) a TL;DR, key takeaways and chapters.

    Runs a single Gemini call, stores the result in the ``summaries`` table and
    swaps the summary card in place. If a summary already exists it is simply
    re-rendered, so the button doubles as a regenerate control.
    """
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    if video is None:
        return tuple(page_events(request, "home.html", home_context(request, user), url="/"))

    summary_service = getattr(request.app.state, "summary_service", None)
    if summary_service is None or not summary_service.available:
        return SSE.patch_elements(
            elements=(
                "<p class='status-error'>AI summaries are unavailable because "
                "no Gemini API key is configured.</p>"
            ),
            selector="#summary-status",
            mode="inner",
        )

    existing = summary_service.get(video_id, user["id"])
    if existing is not None:
        return SSE.patch_elements(
            elements=render_fragment(
                request, "_summary.html", video=video, summary=existing
            ),
            selector="#summary-card",
            mode="outer",
        )

    async def stream() -> Any:
        yield SSE.patch_elements(
            elements="<p class='status-progress'>Analyzing the transcript with Gemini…</p>",
            selector="#summary-status",
            mode="inner",
        )
        try:
            summary = await summary_service.generate(video_id, user["id"])
        except Exception as exc:  # noqa: BLE001 - surface as a friendly error
            logger.exception("Summary generation failed for video %s", video_id)
            yield SSE.patch_elements(
                elements=(
                    "<p class='status-error'>Summary generation failed. "
                    "Please try again.</p>"
                ),
                selector="#summary-status",
                mode="inner",
            )
            return
        yield SSE.patch_elements(
            elements=render_fragment(
                request, "_summary.html", video=video, summary=summary
            ),
            selector="#summary-card",
            mode="outer",
        )

    return stream()


@router.post("/videos/{video_id}/social-post", summary="Generate a social post (Datastar)")
@datastar_action
async def generate_social_post_action(request: Request, video_id: int):
    """Generate (or regenerate) an editable social post with hashtags.

    Uses Gemini to write a post based on the video title and transcript, stores
    the result in the ``social_posts`` table, then swaps the whole card so the
    editor shows the new post and the button flips to ``Regenerate``. A later
    call (or page load) reads the stored post instead of regenerating.
    """
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    if video is None:
        return tuple(page_events(request, "home.html", home_context(request, user), url="/"))

    service = getattr(request.app.state, "social_post_service", None)
    if service is None or not service.available:
        return SSE.patch_elements(
            elements=(
                "<p class='status-error'>Social post generation is unavailable "
                "because no Gemini API key is configured.</p>"
            ),
            selector="#social-post-status",
            mode="inner",
        )

    async def stream() -> Any:
        yield SSE.patch_elements(
            elements="<p class='status-progress'>Writing your social post…</p>",
            selector="#social-post-status",
            mode="inner",
        )
        try:
            social_post = await service.generate(video_id, user["id"])
        except Exception as exc:  # noqa: BLE001 - surface as a friendly error
            logger.exception("Social post generation failed for video %s", video_id)
            yield SSE.patch_elements(
                elements=(
                    "<p class='status-error'>Social post generation failed. "
                    "Please try again.</p>"
                ),
                selector="#social-post-status",
                mode="inner",
            )
            return
        yield SSE.patch_elements(
            elements=render_fragment(
                request,
                "_social_post.html",
                video=video,
                social_post=social_post,
            ),
            selector="#social-post-card",
            mode="outer",
        )

    return stream()


@router.post("/videos/{video_id}/quiz", summary="Generate an AI quiz (Datastar)")
@datastar_action
async def generate_quiz_action(request: Request, video_id: int):
    """Generate (or regenerate) a multiple-choice quiz from the transcript.

    Uses Gemini to write a quiz based on the video title and transcript, stores
    the result in the ``quizzes`` table, then swaps the whole card so the quiz
    is shown and the button flips to ``Regenerate``. A later call (or page
    load) reads the stored quiz instead of regenerating.
    """
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    if video is None:
        return tuple(page_events(request, "home.html", home_context(request, user), url="/"))

    service = getattr(request.app.state, "quiz_service", None)
    if service is None or not service.available:
        return SSE.patch_elements(
            elements=(
                "<p class='status-error'>Quiz generation is unavailable "
                "because no Gemini API key is configured.</p>"
            ),
            selector="#quiz-status",
            mode="inner",
        )

    async def stream() -> Any:
        yield SSE.patch_elements(
            elements="<p class='status-progress'>Writing your quiz…</p>",
            selector="#quiz-status",
            mode="inner",
        )
        try:
            quiz = await service.generate(video_id, user["id"])
        except Exception as exc:  # noqa: BLE001 - surface as a friendly error
            logger.exception("Quiz generation failed for video %s", video_id)
            yield SSE.patch_elements(
                elements=(
                    "<p class='status-error'>Quiz generation failed. "
                    "Please try again.</p>"
                ),
                selector="#quiz-status",
                mode="inner",
            )
            return
        yield SSE.patch_elements(
            elements=render_fragment(
                request,
                "_quiz.html",
                video=video,
                quiz=quiz,
                quiz_available=True,
            ),
            selector="#quiz-card",
            mode="outer",
        )

    return stream()


@router.post("/videos/{video_id}/delete", summary="Delete a video (Datastar)")
@datastar_action
async def delete_video_action(request: Request, video_id: int):
    """Delete a video (cascades to transcript, chunks and chats).

    When triggered from a dashboard card (``from_dashboard`` signal) the videos
    panel and category counts are re-rendered in place; otherwise the page
    navigates back to the dashboard.
    """
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    database.delete_video(video_id=video_id, user_id=user["id"])

    okf = getattr(request.app.state, "okf", None)
    if okf is not None and video is not None:
        try:
            okf.delete_video(user["id"], video["youtube_id"])
        except Exception:  # noqa: BLE001 - knowledge layer must not block deletion
            logger.exception("OKF concept cleanup failed for video %s", video_id)

    signals = await read_signals(request) or {}
    if signals.get("from_dashboard"):
        context = home_context(request, user)
        return (
            SSE.patch_elements(
                elements=render_fragment(request, "_videos.html", **context),
                selector="#videos-panel",
                mode="outer",
            ),
            SSE.patch_elements(
                elements=render_fragment(
                    request, "_categories.html", categories=context["categories"]
                ),
                selector="#category-list",
                mode="outer",
            ),
        )

    return tuple(page_events(request, "home.html", home_context(request, user), url="/"))


def _voice_clone_context(request: Request, video_id: int, user_id: int) -> dict[str, Any]:
    """Build the template context for the voice-clone panel."""
    database = request.app.state.database
    voice_clone = getattr(request.app.state, "voice_clone_service", None)
    video = database.get_video(video_id, user_id) or {}
    return {
        "video": {"id": video_id, "youtube_id": video.get("youtube_id")},
        "voice_profile": database.get_voice_profile_for_video(
            video_id=video_id, user_id=user_id
        ),
        "voice_clone_available": bool(
            voice_clone is not None and voice_clone.available
        ),
    }


@router.post("/videos/{video_id}/clone-voice", summary="Clone a video's voice (Datastar)")
@datastar_action
async def clone_voice_action(request: Request, video_id: int):
    """Download a clean sample of the video's audio and save a voice profile.

    Streams progress as it picks a speech-dense window, downloads that slice,
    extracts a clean reference clip and stores a reusable voice profile.
    """
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    if video is None:
        return SSE.patch_signals({"cloning": False, "clone_error": "Video not found."})

    voice_clone = getattr(request.app.state, "voice_clone_service", None)
    if voice_clone is None or not voice_clone.available:
        return SSE.patch_signals(
            {
                "cloning": False,
                "clone_error": "Voice cloning is not available on this server.",
            }
        )
    if not video.get("youtube_id"):
        return SSE.patch_signals(
            {
                "cloning": False,
                "clone_error": "This video has no source audio to clone from.",
            }
        )

    youtube_service = request.app.state.youtube_service

    async def stream() -> Any:
        yield SSE.patch_signals({"cloning": True, "clone_status": "", "clone_error": ""})
        try:
            start, end, reference_text = await asyncio.to_thread(
                pick_sample_window, video["segments"], settings.voice_clone_sample_seconds
            )
            if end <= start:
                start, end = 0.0, min(
                    settings.voice_clone_sample_seconds, video["duration"] or 15.0
                )

            yield SSE.patch_signals({"clone_status": "Downloading a short audio sample…"})
            result = await asyncio.to_thread(
                youtube_service.download_audio_section,
                video["youtube_url"],
                start,
                end,
            )
            yield SSE.patch_signals({"clone_status": "Extracting the speaker's voice…"})
            _, ref_name = await asyncio.to_thread(
                make_reference_clip,
                Path(result["audio_path"]),
                settings.voice_clone_dir,
            )

            old_profile = database.get_voice_profile_for_video(
                video_id=video_id, user_id=user["id"]
            )
            profile_id = database.create_voice_profile(
                user_id=user["id"],
                video_id=video_id,
                name=f"{video.get('uploader') or 'Channel'}'s voice",
                language=video.get("language") or "en",
                reference_wav=ref_name,
                reference_text=reference_text,
            )
            if old_profile is not None and old_profile["reference_wav"] != ref_name:
                (Path(settings.voice_clone_dir) / old_profile["reference_wav"]).unlink(
                    missing_ok=True
                )

            yield SSE.patch_elements(
                elements=render_fragment(
                    request,
                    "_voice_clone.html",
                    **_voice_clone_context(request, video_id, user["id"]),
                ),
                selector="#voice-clone-panel",
                mode="outer",
            )
            yield SSE.patch_signals({"cloning": False, "clone_status": "", "clone_error": ""})
        except Exception:  # noqa: BLE001 - surface as a friendly error
            logger.exception("Voice clone failed for video %d", video_id)
            yield SSE.patch_signals(
                {
                    "cloning": False,
                    "clone_status": "",
                    "clone_error": "Voice cloning failed. Try again.",
                }
            )
        finally:
            try:
                youtube_service.cleanup(video["youtube_id"])
            except Exception:  # noqa: BLE001 - best-effort temp cleanup
                logger.warning(
                    "Could not clean up voice-clone temp files for video %d", video_id
                )

    return stream()


@router.post("/voice-profiles/{profile_id}/delete", summary="Delete a voice profile (Datastar)")
@datastar_action
async def delete_voice_profile_action(request: Request, profile_id: int):
    """Delete a cloned voice profile and its reference clip."""
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    profile = database.get_voice_profile(profile_id=profile_id, user_id=user["id"])
    if profile is None:
        return SSE.patch_signals({"clone_error": "Voice profile not found."})
    if database.delete_voice_profile(profile_id=profile_id, user_id=user["id"]):
        (Path(settings.voice_clone_dir) / profile["reference_wav"]).unlink(missing_ok=True)
    return SSE.patch_elements(
        elements=render_fragment(
            request,
            "_voice_clone.html",
            **_voice_clone_context(request, profile["video_id"], user["id"]),
        ),
        selector="#voice-clone-panel",
        mode="outer",
    )


@router.post("/videos/{video_id}/chat/new", summary="Start a conversation (Datastar)")
@datastar_action
async def new_conversation_action(request: Request, video_id: int):
    """Create a new conversation and navigate to it."""
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    if video is None:
        return tuple(page_events(request, "home.html", home_context(request, user), url="/"))

    conversation_id = database.create_conversation(
        user_id=user["id"], video_id=video_id, title="New chat"
    )
    url = f"/videos/{video_id}/chat?conversation={conversation_id}"
    return tuple(page_events(request, "chat.html", chat_context(request, user, video_id, conversation_id), url=url))


@router.post("/library/chat/new", summary="Start a library conversation (Datastar)")
@datastar_action
async def new_library_conversation_action(request: Request):
    """Create a library-level conversation and navigate to it."""
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    conversation_id = database.create_conversation(
        user_id=user["id"], video_id=None, title="Library chat"
    )
    url = f"/library?conversation={conversation_id}"
    return tuple(
        page_events(
            request,
            "library.html",
            library_context(request, user, conversation_id),
            url=url,
        )
    )


@router.post("/library/chat/send", summary="Send a library chat message (Datastar SSE)")
@datastar_action
async def send_library_message_action(request: Request):
    """Append a user message and stream a cross-video answer into the DOM."""
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    signals = await read_signals(request) or {}
    question = (signals.get("message") or "").strip()
    if not question:
        return None

    conversation_id = signals.get("conversation_id")
    try:
        conversation_id = (
            int(conversation_id) if conversation_id not in (None, "", 0) else None
        )
    except (TypeError, ValueError):
        conversation_id = None

    created = False
    if conversation_id is None:
        conversation_id = database.create_conversation(
            user_id=user["id"], video_id=None, title=question[:60] or "Library chat"
        )
        created = True
    else:
        conversation = database.get_conversation(conversation_id, user["id"])
        if conversation is None:
            return tuple(page_events(request, "home.html", home_context(request, user), url="/"))

    chat = request.app.state.chat
    if not chat.available:
        return (
            SSE.patch_elements(
                elements=render_fragment(
                    request,
                    "_library_messages.html",
                    messages=database.list_messages(conversation_id),
                ),
                selector="#messages",
                mode="outer",
            ),
            SSE.patch_signals({"message": "", "conversation_id": conversation_id}),
        )

    user_bubble = render_fragment(
        request,
        "_library_message.html",
        message={"role": "user", "content": question, "sources": []},
    )
    answer_slot = (
        '<div class="message assistant" id="answer-slot">'
        '<div class="bubble assistant">'
        '<span class="thinking">'
        '<span class="dots" aria-hidden="true">'
        '<span></span><span></span><span></span>'
        '</span>'
        '<span class="thinking-label">Thinking…</span>'
        '</span>'
        '</div></div>'
    )

    async def stream() -> Any:
        yield SSE.patch_elements(
            elements=f"{user_bubble}{answer_slot}",
            selector="#messages",
            mode="append",
        )
        yield SSE.patch_signals({"message": "", "conversation_id": conversation_id})

        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        error: str | None = None

        async for event in chat.stream_library_answer(conversation_id, question, user["id"]):
            if event["type"] == "sources":
                sources = event["data"]
            elif event["type"] == "token":
                answer_parts.append(event["data"])
                text = html.escape("".join(answer_parts)).replace("\n", "<br>")
                yield SSE.patch_elements(
                    elements=f"<div class='bubble assistant'>{text}</div>",
                    selector="#answer-slot .bubble",
                    mode="outer",
                )
            elif event["type"] == "error":
                error = event["data"]
            elif event["type"] == "done":
                break

        answer = "".join(answer_parts)
        if error or not answer:
            content = error or "Sorry, I couldn't generate an answer."
            final_bubble = (
                f'<div class="message assistant" id="answer-slot">'
                f'<div class="bubble assistant error">{html.escape(content)}</div></div>'
            )
        else:
            database.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                sources=sources or None,
            )
            final_bubble = render_fragment(
                request,
                "_library_message.html",
                message={"role": "assistant", "content": answer, "sources": sources},
            )

        yield SSE.patch_elements(
            elements=final_bubble,
            selector="#answer-slot",
            mode="outer",
        )

        conversations = database.list_library_conversations(user["id"])
        yield SSE.patch_elements(
            elements=render_fragment(
                request, "_library_conversations.html", conversations=conversations, active=conversation_id
            ),
            selector="#conversation-list",
            mode="outer",
        )
        if created:
            yield SSE.patch_signals({"conversation_id": conversation_id})

    return stream()


@router.post("/chats/{conversation_id}/delete", summary="Delete a conversation (Datastar)")
@datastar_action
async def delete_conversation_action(request: Request, conversation_id: int):
    """Delete a conversation and navigate back to the chat page."""
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    conversation = database.get_conversation(conversation_id, user["id"])
    database.delete_conversation(conversation_id=conversation_id, user_id=user["id"])
    if conversation is None:
        return tuple(page_events(request, "home.html", home_context(request, user), url="/"))
    if conversation["video_id"] is None:
        return tuple(
            page_events(request, "library.html", library_context(request, user), url="/library")
        )
    url = f"/videos/{conversation['video_id']}/chat"
    return tuple(
        page_events(
            request,
            "chat.html",
            chat_context(request, user, conversation["video_id"]),
            url=url,
        )
    )


@router.post("/videos/{video_id}/chat/send", summary="Send a chat message (Datastar SSE)")
@datastar_action
async def send_message_action(request: Request, video_id: int):
    """Append a user message and stream the assistant's answer into the DOM.

    Reads ``message`` (and optionally ``conversation_id``) from the Datastar
    signals, creates a conversation when needed, and patches the messages
    container as tokens arrive.
    """
    user = current_user(request)
    if user is None:
        return tuple(login_page_events(request))

    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    if video is None:
        return tuple(page_events(request, "home.html", home_context(request, user), url="/"))

    signals = await read_signals(request) or {}
    question = (signals.get("message") or "").strip()
    if not question:
        return None

    conversation_id = signals.get("conversation_id")
    try:
        conversation_id = (
            int(conversation_id) if conversation_id not in (None, "", 0) else None
        )
    except (TypeError, ValueError):
        conversation_id = None

    created = False
    if conversation_id is None:
        conversation_id = database.create_conversation(
            user_id=user["id"], video_id=video_id, title=question[:60] or "New chat"
        )
        created = True
    else:
        conversation = database.get_conversation(conversation_id, user["id"])
        if conversation is None:
            return tuple(page_events(request, "home.html", home_context(request, user), url="/"))

    chat = request.app.state.chat
    if not chat.available:
        return (
            SSE.patch_elements(
                elements=render_fragment(
                    request,
                    "_messages.html",
                    messages=database.list_messages(conversation_id),
                    video_id=video_id,
                ),
                selector="#messages",
                mode="outer",
            ),
            SSE.patch_signals({"message": "", "conversation_id": conversation_id}),
        )

    # Render the user bubble and an empty assistant slot.
    user_bubble = render_fragment(
        request,
        "_message.html",
        message={"role": "user", "content": question, "sources": []},
        video_id=video_id,
    )
    answer_slot = (
        '<div class="message assistant" id="answer-slot">'
        '<div class="bubble assistant">'
        '<span class="thinking">'
        '<span class="dots" aria-hidden="true">'
        '<span></span><span></span><span></span>'
        '</span>'
        '<span class="thinking-label">Thinking…</span>'
        '</span>'
        '</div></div>'
    )

    async def stream() -> Any:
        yield SSE.patch_elements(
            elements=f"{user_bubble}{answer_slot}",
            selector="#messages",
            mode="append",
        )
        yield SSE.patch_signals({"message": "", "conversation_id": conversation_id})

        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        error: str | None = None

        async for event in chat.stream_answer(conversation_id, question, user["id"]):
            if event["type"] == "sources":
                sources = event["data"]
            elif event["type"] == "token":
                answer_parts.append(event["data"])
                text = html.escape("".join(answer_parts)).replace("\n", "<br>")
                yield SSE.patch_elements(
                    elements=f"<div class='bubble assistant'>{text}</div>",
                    selector="#answer-slot .bubble",
                    mode="outer",
                )
            elif event["type"] == "error":
                error = event["data"]
            elif event["type"] == "done":
                break

        answer = "".join(answer_parts)
        if error or not answer:
            content = error or "Sorry, I couldn't generate an answer."
            final_bubble = (
                f'<div class="message assistant" id="answer-slot">'
                f'<div class="bubble assistant error">{html.escape(content)}</div></div>'
            )
        else:
            database.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                sources=sources or None,
            )
            final_bubble = render_fragment(
                request,
                "_message.html",
                message={"role": "assistant", "content": answer, "sources": sources},
                video_id=video_id,
            )

        yield SSE.patch_elements(
            elements=final_bubble,
            selector="#answer-slot",
            mode="outer",
        )

        # Refresh the conversation list (title/history may have changed).
        conversations = database.list_conversations(video_id=video_id, user_id=user["id"])
        yield SSE.patch_elements(
            elements=render_fragment(
                request, "_conversations.html", conversations=conversations
            ),
            selector="#conversation-list",
            mode="outer",
        )
        if created:
            yield SSE.patch_signals({"conversation_id": conversation_id})

    return stream()


# ---------------------------------------------------------------------------
# Speech-to-text (microphone dictation)
# ---------------------------------------------------------------------------


@router.post("/api/v1/chat/stt", tags=["api"], summary="Speech-to-text (chat dictation)")
async def speech_to_text_action(request: Request):
    """Transcribe recorded microphone audio and return the spoken text.

    The chat client records a short clip with the MediaRecorder API and POSTs
    the raw audio bytes here. The existing Whisper model is reused to
    transcribe the clip, and the recognised text is returned as JSON so the
    caller can drop it straight into the chat message box.

    Requires authentication; the request body is the raw audio bytes.
    """
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")

    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="No audio data received.")

    content_type = request.headers.get("content-type", "")
    suffix = ".webm" if "webm" in content_type else ".m4a" if "mp4" in content_type else ".bin"
    audio_path = settings.temp_dir / f"stt_{user['id']}_{uuid.uuid4().hex}{suffix}"
    try:
        audio_path.write_bytes(audio)
        transcription_service = request.app.state.transcription_service
        result = await asyncio.to_thread(transcription_service.transcribe, str(audio_path))
    except AppError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.detail) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Speech-to-text failed for user %d", user["id"])
        raise HTTPException(status_code=500, detail="Speech-to-text failed.") from exc
    finally:
        audio_path.unlink(missing_ok=True)

    return JSONResponse({"text": result["transcript"]})
