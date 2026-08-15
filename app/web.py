"""Shared helpers for the web UI (template rendering and auth)."""

import html
import json
import re
from typing import Any

import markdown as md
from datastar_py.sse import ServerSentEventGenerator as SSE
from datastar_py.starlette import DatastarResponse
from fastapi import Request

from app.auth import request_user_id
from app.services.rag import format_timestamp

_HEAD_RE = re.compile(r"<head[^>]*>(.*?)</head>", re.DOTALL)
_BODY_RE = re.compile(r"<body[^>]*>(.*?)</body>", re.DOTALL)


def render_markdown(text: str) -> str:
    """Render plain text as safe Markdown HTML (input is HTML-escaped first)."""
    return md.markdown(html.escape(text), extensions=["extra"])


def is_datastar_request(request: Request) -> bool:
    """Return ``True`` when the request was issued by a Datastar action.

    Datastar's ``@get``/``@post`` fetches send a ``Datastar-Request`` header
    and advertise ``text/event-stream`` in ``Accept``.
    """
    return bool(
        request.headers.get("datastar-request")
        or "text/event-stream" in request.headers.get("accept", "")
    )


def render_page(request: Request, name: str, context: dict[str, Any]):
    """Render a page, or an SSE swap for Datastar SPA navigation.

    Plain requests (browser navigation, no-JS fallback) receive the full HTML
    document. Requests issued via a Datastar ``@get`` action receive an SSE
    stream that morphs the current ``<head>`` and ``<body>`` in place, so
    navigating between pages never triggers a full page reload.
    """
    templates = request.app.state.templates
    if not is_datastar_request(request):
        return templates.TemplateResponse(request, name, context)
    return DatastarResponse(tuple(page_events(request, name, context)))


def page_events(
    request: Request,
    name: str,
    context: dict[str, Any],
    url: str | None = None,
) -> list[Any]:
    """Return SSE events that morph the current page into ``name``.

    Swaps ``<head>`` and ``<body>`` in place (true SPA navigation) and, when
    ``url`` is given, syncs the address bar with ``history.pushState`` so the
    browser never issues a full page load.
    """
    templates = request.app.state.templates
    page = templates.get_template(name).render(request=request, **context)
    events: list[Any] = []
    head = _HEAD_RE.search(page)
    if head is not None:
        events.append(
            SSE.patch_elements(elements=head.group(1), selector="head", mode="inner")
        )
    body = _BODY_RE.search(page)
    if body is not None:
        events.append(
            SSE.patch_elements(elements=body.group(1), selector="body", mode="inner")
        )
    if url:
        events.append(
            SSE.execute_script(
                "history.pushState({}, '', %s); window.scrollTo(0, 0);"
                % json.dumps(url)
            )
        )
    return events


def login_page_events(request: Request, url: str = "/login") -> list[Any]:
    """SSE events that navigate to the login page (session expired)."""
    return page_events(request, "login.html", {"error": None}, url=url)


def home_context(request: Request, user: dict[str, Any]) -> dict[str, Any]:
    """Build the dashboard template context for ``user``."""
    database = request.app.state.database
    raw_category = (request.query_params.get("category") or "").strip()
    category_id: int | None = None
    uncategorized = False
    if raw_category in ("0", "uncategorized"):
        uncategorized = True
    elif raw_category:
        try:
            category_id = int(raw_category)
        except ValueError:
            category_id = None

    videos = database.list_videos(
        user["id"], category_id=category_id, uncategorized=uncategorized
    )
    categories = database.list_categories(user["id"])
    active_category = next((c for c in categories if c["id"] == category_id), None)

    return {
        "user": user,
        "videos": videos,
        "categories": categories,
        "active_category": active_category,
        "category_id": category_id,
        "uncategorized": uncategorized,
        "videos_processing": any(v["status"] == "processing" for v in videos),
        "chat_available": request.app.state.chat.available,
    }


def video_context(
    request: Request, user: dict[str, Any], video_id: int
) -> dict[str, Any]:
    """Build the transcript page template context for ``video_id``."""
    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "start_label": format_timestamp(segment["start"]),
            "end_label": format_timestamp(segment["end"]),
        }
        for segment in video["segments"]
    ]
    summary_service = getattr(request.app.state, "summary_service", None)
    social_post_service = getattr(request.app.state, "social_post_service", None)
    return {
        "user": user,
        "video": {**video, "segments": segments},
        "duration_label": format_timestamp(video["duration"]),
        "categories": database.list_categories(user["id"]),
        "chat_available": request.app.state.chat.available,
        "summary": database.get_summary(video_id),
        "summary_available": bool(summary_service is not None and summary_service.available),
        "social_post": (
            social_post_service.get(video_id, user["id"])
            if social_post_service is not None
            else None
        ),
    }


def search_context(request: Request, user: dict[str, Any], q: str) -> dict[str, Any]:
    """Build the search page template context for query ``q``."""
    database = request.app.state.database
    results = database.search_library(user_id=user["id"], query=q)
    for segment in results["segments"]:
        segment["start_label"] = format_timestamp(segment["start"])
        segment["end_label"] = format_timestamp(segment["end"])
    return {
        "user": user,
        "q": q,
        "videos": results["videos"],
        "segments": results["segments"],
    }


def chat_context(
    request: Request,
    user: dict[str, Any],
    video_id: int,
    conversation_id: int | None = None,
) -> dict[str, Any]:
    """Build the chat page template context for ``video_id``."""
    database = request.app.state.database
    video = database.get_video(video_id, user["id"])
    conversations = database.list_conversations(video_id=video_id, user_id=user["id"])

    active = None
    messages: list[dict[str, Any]] = []
    if conversation_id is not None:
        active = database.get_conversation(conversation_id, user["id"])
        if active is not None:
            messages = database.list_messages(conversation_id)
    elif conversations:
        active = conversations[0]
        messages = database.list_messages(active["id"])

    return {
        "user": user,
        "video": video,
        "conversations": conversations,
        "active": active,
        "messages": messages,
        "chat_available": request.app.state.chat.available,
    }


def library_context(
    request: Request,
    user: dict[str, Any],
    conversation_id: int | None = None,
) -> dict[str, Any]:
    """Build the library chat page template context for ``user``."""
    database = request.app.state.database
    conversations = database.list_library_conversations(user["id"])

    active = None
    messages: list[dict[str, Any]] = []
    if conversation_id is not None:
        active = database.get_conversation(conversation_id, user["id"])
        if active is not None:
            messages = database.list_messages(conversation_id)
    elif conversations:
        active = conversations[0]
        messages = database.list_messages(active["id"])

    return {
        "user": user,
        "conversations": conversations,
        "active": active,
        "messages": messages,
        "chat_available": request.app.state.chat.available,
    }


def current_user(request: Request) -> dict[str, Any] | None:
    """Return the authenticated user dict, or ``None``.

    Args:
        request: The incoming request.

    Returns:
        The user row from the database, or ``None``.
    """
    user_id = request_user_id(request)
    if user_id is None:
        return None
    return request.app.state.database.get_user_by_id(user_id)


def require_user(request: Request) -> dict[str, Any] | None:
    """Return the authenticated user, or ``None`` (caller redirects).

    Kept as a thin wrapper so route handlers can check once and act.
    """
    return current_user(request)


def render_fragment(request: Request, name: str, **context: Any) -> str:
    """Render a template fragment to a string for Datastar patches.

    Args:
        request: The incoming request.
        name: Template file name (e.g. ``_messages.html``).
        **context: Values passed to the template.

    Returns:
        The rendered HTML string.
    """
    templates = request.app.state.templates
    return templates.get_template(name).render(request=request, **context)
