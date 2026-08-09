"""Authentication routes: signup, login and logout."""

import logging

from datastar_py.fastapi import read_signals
from datastar_py.starlette import DatastarResponse
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth import create_access_token, hash_password, verify_password
from app.config import settings
from app.web import home_context, is_datastar_request, page_events, render_page

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


def _template(request: Request, name: str, context: dict) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(request, name, context)


def _auth_error(request: Request, template: str, message: str):
    """Re-render an auth form with an error, natively or via SSE."""
    context = {"error": message}
    if is_datastar_request(request):
        return DatastarResponse(tuple(page_events(request, template, context)))
    return _template(request, template, context)


def _set_auth_cookie(response, token: str) -> None:
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        max_age=settings.jwt_expire_minutes * 60,
        httponly=True,
        samesite="lax",
        path="/",
    )


def _clear_auth_cookie(response) -> None:
    response.delete_cookie(key=settings.cookie_name, path="/")


def _success_response(
    request: Request, user: dict, token: str, clear: bool = False
):
    """Swap to the dashboard after a successful signup/login, or to login on logout."""
    response = DatastarResponse(
        tuple(
            page_events(
                request,
                "login.html" if clear else "home.html",
                {"error": None} if clear else home_context(request, user),
                url="/login" if clear else "/",
            )
        )
    )
    if clear:
        _clear_auth_cookie(response)
    else:
        _set_auth_cookie(response, token)
    return response


async def _credentials(request: Request) -> tuple[str, str, str]:
    """Read email/password/confirm from Datastar signals or a native form."""
    if is_datastar_request(request):
        signals = await read_signals(request) or {}
        return (
            (signals.get("email") or "").strip(),
            signals.get("password") or "",
            signals.get("confirm") or "",
        )
    form = await request.form()
    return (
        (form.get("email") or "").strip(),
        form.get("password") or "",
        form.get("confirm") or "",
    )


@router.get("/login", response_class=HTMLResponse, summary="Login page")
async def login_page(request: Request) -> HTMLResponse:
    """Render the login form."""
    return render_page(request, "login.html", {"error": None})


@router.get("/signup", response_class=HTMLResponse, summary="Signup page")
async def signup_page(request: Request) -> HTMLResponse:
    """Render the signup form."""
    return render_page(request, "signup.html", {"error": None})


@router.post("/auth/login", summary="Log in")
async def login(request: Request):
    """Validate credentials and set the JWT cookie.

    On success the user lands on their dashboard (SPA swap for Datastar
    requests, redirect otherwise); on failure the form is re-rendered with an
    error message.
    """
    email, password, _ = await _credentials(request)

    database = request.app.state.database
    user = database.get_user_by_email(email) if email else None
    if user is None or not verify_password(password, user["password_hash"]):
        logger.info("Login failed for %r", email)
        return _auth_error(request, "login.html", "Invalid email or password.")

    token = create_access_token(user["id"])
    logger.info("User %s logged in", user["email"])

    if is_datastar_request(request):
        return _success_response(request, user, token)

    response = RedirectResponse(url="/", status_code=303)
    _set_auth_cookie(response, token)
    return response


@router.post("/auth/signup", summary="Create an account")
async def signup(request: Request):
    """Create a user account and log them in.

    On success the user lands on their dashboard (SPA swap for Datastar
    requests, redirect otherwise); on failure the form is re-rendered with an
    error message.
    """
    email, password, confirm = await _credentials(request)

    if not email or "@" not in email:
        return _auth_error(request, "signup.html", "Please enter a valid email.")
    if len(password) < 8:
        return _auth_error(
            request, "signup.html", "Password must be at least 8 characters."
        )
    if password != confirm:
        return _auth_error(request, "signup.html", "Passwords do not match.")

    database = request.app.state.database
    if database.get_user_by_email(email) is not None:
        return _auth_error(
            request, "signup.html", "An account with this email already exists."
        )

    user_id = database.create_user(email=email, password_hash=hash_password(password))
    user = database.get_user_by_id(user_id)
    token = create_access_token(user_id)
    logger.info("User %s signed up", email)

    if is_datastar_request(request):
        return _success_response(request, user, token)

    response = RedirectResponse(url="/", status_code=303)
    _set_auth_cookie(response, token)
    return response


@router.post("/auth/logout", summary="Log out")
async def logout(request: Request):
    """Clear the JWT cookie and return to the login page."""
    if is_datastar_request(request):
        return _success_response(request, {}, "", clear=True)

    response = RedirectResponse(url="/login", status_code=303)
    _clear_auth_cookie(response)
    return response
