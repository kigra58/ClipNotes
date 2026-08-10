"""Web UI pages and Datastar actions (dashboard, videos, categories, chat).

All interactive behavior is driven by Datastar: forms submit via
``data-on:submit="@post(...)"`` and the server responds with SSE events
(:class:`datastar_py.sse.ServerSentEventGenerator`).
"""

import asyncio
import html
import logging
import uuid
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
from app.utils.youtube import extract_video_id, normalize_youtube_url
from app.web import (
    chat_context,
    current_user,
    home_context,
    is_datastar_request,
    login_page_events,
    page_events,
    render_fragment,
    render_page,
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


@router.post("/videos/{video_id}/category", summary="Assign a category (Datastar)")
@datastar_action
async def set_video_category_action(request: Request, video_id: int):
    """Assign or clear a video's category and show a confirmation."""
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
        category_id = (
            int(category_id) if category_id not in (None, "", 0) else None
        )
    except (TypeError, ValueError):
        category_id = None
    database.set_video_category(video_id=video_id, user_id=user["id"], category_id=category_id)

    return SSE.patch_elements(
        elements="<span class='saved-note'>Saved</span>",
        selector="#category-saved",
        mode="inner",
    )


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
    database.delete_video(video_id=video_id, user_id=user["id"])

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
    if conversation is not None:
        url = f"/videos/{conversation['video_id']}/chat"
        return tuple(
            page_events(
                request,
                "chat.html",
                chat_context(request, user, conversation["video_id"]),
                url=url,
            )
        )
    return tuple(page_events(request, "home.html", home_context(request, user), url="/"))


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
