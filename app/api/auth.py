"""Authentication routes: signup, login, logout and email verification."""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlencode

from datastar_py.fastapi import read_signals
from datastar_py.starlette import DatastarResponse
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth import create_access_token, hash_password, verify_password
from app.config import settings
from app.services.email import EmailService
from app.web import home_context, is_datastar_request, page_events, render_page

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


def _template(request: Request, name: str, context: dict) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(request, name, context)


def _auth_error(request: Request, template: str, message: str, **extra):
    """Re-render an auth form with an error, natively or via SSE."""
    context = {"error": message, **extra}
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


def _email_service(request: Request) -> EmailService:
    """Return the app's email service (or a settings-built fallback in tests)."""
    service = getattr(request.app.state, "email_service", None)
    if service is None:
        service = EmailService.from_settings()
    return service


def _new_verification_token() -> tuple[str, str]:
    """Generate a fresh verification token and its expiry timestamp."""
    token = secrets.token_urlsafe(32)
    expires = (
        datetime.now(timezone.utc) + timedelta(minutes=settings.email_verify_token_minutes)
    ).isoformat()
    return token, expires


def _is_expired(expires_at: str | None) -> bool:
    """Return ``True`` when an ISO expiry timestamp is in the past."""
    if not expires_at:
        return False
    try:
        return datetime.fromisoformat(expires_at) < datetime.now(timezone.utc)
    except ValueError:
        return True


_MAX_RESET_ATTEMPTS = 5


def _generate_otp() -> str:
    """Generate a 6-digit one-time password (zero-padded string)."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_otp(otp: str) -> str:
    """Hash a 6-digit OTP so a leaked database cannot be used directly."""
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()


def _verify_response(request: Request, email: str, notice: str | None = None):
    """Show the "check your inbox" page after signup/resend.

    Datastar requests swap to the page via SSE; native form posts use a
    Post/Redirect/Get redirect so a refresh never resubmits the form.
    """
    context = {"email": email, "error": None, "notice": notice}
    params = {"email": email}
    if notice:
        params["notice"] = notice
    url = f"/verify-email?{urlencode(params)}"
    if is_datastar_request(request):
        return DatastarResponse(
            tuple(page_events(request, "verify_email.html", context, url=url))
        )
    return RedirectResponse(url=url, status_code=303)


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


async def _reset_credentials(request: Request) -> tuple[str, str, str, str]:
    """Read email/otp/password/confirm from Datastar signals or a native form."""
    if is_datastar_request(request):
        signals = await read_signals(request) or {}
        return (
            (signals.get("email") or "").strip(),
            (signals.get("otp") or "").strip(),
            signals.get("password") or "",
            signals.get("confirm") or "",
        )
    form = await request.form()
    return (
        (form.get("email") or "").strip(),
        (form.get("otp") or "").strip(),
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
    if not user["is_verified"]:
        logger.info("Login blocked for unverified user %r", email)
        return _auth_error(
            request,
            "login.html",
            "Please verify your email before logging in. Check your inbox (and spam) for the link.",
        )

    token = create_access_token(user["id"])
    logger.info("User %s logged in", user["email"])

    if is_datastar_request(request):
        return _success_response(request, user, token)

    response = RedirectResponse(url="/", status_code=303)
    _set_auth_cookie(response, token)
    return response


@router.post("/auth/signup", summary="Create an account")
async def signup(request: Request):
    """Create a user account and require email verification.

    When SMTP is configured the user is not logged in; instead a verification
    email is sent and the "check your inbox" page is shown. When SMTP is not
    configured (local development) the account is auto-verified and the user
    is logged in immediately, preserving the pre-verification behavior.
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

    email_service = _email_service(request)

    token = None
    expires_at = None
    if email_service.available:
        token, expires_at = _new_verification_token()

    user_id = database.create_user(
        email=email,
        password_hash=hash_password(password),
        verification_token=token,
        verification_token_expires_at=expires_at,
    )

    if email_service.available:
        try:
            email_service.send_verification_email(email, token)
        except Exception:
            logger.exception("Failed to send verification email to %s", email)
            database.delete_user(user_id)
            return _auth_error(
                request,
                "signup.html",
                "We couldn't send a verification email. Please try again.",
            )
        logger.info("User %s signed up; verification email sent", email)
        return _verify_response(request, email)

    # SMTP not configured (dev mode): verify immediately and log in.
    database.verify_user(user_id)
    user = database.get_user_by_id(user_id)
    token = create_access_token(user_id)
    logger.info("User %s signed up (email verification disabled)", email)

    if is_datastar_request(request):
        return _success_response(request, user, token)

    response = RedirectResponse(url="/", status_code=303)
    _set_auth_cookie(response, token)
    return response


@router.get("/verify-email", response_class=HTMLResponse, summary="Verify prompt page")
async def verify_email_page(request: Request) -> HTMLResponse:
    """Render the "check your inbox" page after signup."""
    email = request.query_params.get("email", "")
    notice = request.query_params.get("notice")
    return render_page(
        request,
        "verify_email.html",
        {"email": email, "error": None, "notice": notice},
    )


@router.get(
    "/auth/verify-email",
    response_class=HTMLResponse,
    summary="Verify an email address",
)
async def verify_email(request: Request) -> HTMLResponse:
    """Validate a verification token and mark the user's email as verified."""
    token = (request.query_params.get("token") or "").strip()
    database = request.app.state.database
    user = database.get_user_by_verification_token(token) if token else None

    ok = False
    message = "This verification link is invalid or has expired."
    if user is not None:
        if not _is_expired(user.get("verification_token_expires_at")):
            database.verify_user(user["id"])
            ok = True
            message = f"Your email {user['email']} is verified. You can now log in."
            logger.info("User %s verified their email", user["email"])

    return render_page(
        request,
        "verification.html",
        {"ok": ok, "message": message, "email": user["email"] if user else None},
    )


@router.post("/auth/verify/resend", summary="Resend a verification email")
async def resend_verification(request: Request):
    """Issue a new verification link for an unverified account."""
    email, _, _ = await _credentials(request)
    email = email.strip().lower()
    database = request.app.state.database
    email_service = _email_service(request)

    if not email_service.available:
        return _auth_error(
            request, "verify_email.html", "Email sending is not configured on this server."
        )

    user = database.get_user_by_email(email) if email else None
    if user is None or user["is_verified"]:
        # Never reveal whether the account exists; show a neutral message.
        return _verify_response(
            request,
            email,
            notice="If that address has an unverified account, a new link is on its way.",
        )

    token, expires_at = _new_verification_token()
    database.set_verification_token(user_id=user["id"], token=token, expires_at=expires_at)
    try:
        email_service.send_verification_email(email, token)
    except Exception:
        logger.exception("Failed to resend verification email to %s", email)
        return _auth_error(
            request, "verify_email.html", "We couldn't send the email right now. Please try again."
        )
    logger.info("Resent verification email to %s", email)
    return _verify_response(request, email, notice="A new verification link has been sent.")


# ---------------------------------------------------------------------------
# Password reset (email OTP)
# ---------------------------------------------------------------------------

_RESET_NOTICE = "If an account exists for that email, a 6-digit code is on its way."


@router.get("/forgot-password", response_class=HTMLResponse, summary="Forgot password page")
async def forgot_password_page(request: Request) -> HTMLResponse:
    """Render the email form, or the OTP form once an email has been submitted."""
    email = request.query_params.get("email", "")
    return render_page(
        request,
        "forgot_password.html",
        {
            "email": email,
            "error": None,
            "notice": _RESET_NOTICE if email else None,
        },
    )


@router.post("/auth/forgot-password", summary="Request a password reset OTP")
async def forgot_password(request: Request):
    """Email a 6-digit reset OTP for a verified account.

    The response is deliberately neutral so the endpoint never reveals whether
    an email address is registered.
    """
    email, _, _ = await _credentials(request)
    email = email.strip().lower()
    database = request.app.state.database
    email_service = _email_service(request)

    if not email_service.available:
        logger.info("Password reset requested for %r (SMTP disabled)", email)
    else:
        user = database.get_user_by_email(email) if email else None
        if user is None or not user["is_verified"]:
            logger.info("Password reset requested for unknown/unverified %r", email)
        else:
            otp = _generate_otp()
            expires = (
                datetime.now(timezone.utc)
                + timedelta(minutes=settings.password_reset_token_minutes)
            ).isoformat()
            database.set_reset_otp(user_id=user["id"], otp_hash=_hash_otp(otp), expires_at=expires)
            try:
                email_service.send_password_reset_email(email, otp)
            except Exception:
                logger.exception("Failed to send password reset OTP to %s", email)
                return _auth_error(
                    request,
                    "forgot_password.html",
                    "We couldn't send the email right now. Please try again.",
                )
            logger.info("Sent password reset OTP to %s", email)

    url = f"/forgot-password?email={quote(email)}"
    context = {"email": email, "error": None, "notice": _RESET_NOTICE}
    if is_datastar_request(request):
        return DatastarResponse(tuple(page_events(request, "forgot_password.html", context, url=url)))
    return RedirectResponse(url=url, status_code=303)


@router.post("/auth/reset-password", summary="Set a new password with an OTP")
async def reset_password(request: Request):
    """Validate a 6-digit OTP and update the user's password."""
    email, otp, password, confirm = await _reset_credentials(request)
    email = email.strip().lower()
    otp = otp.strip()

    if not email or "@" not in email:
        return _auth_error(
            request, "forgot_password.html", "Please enter a valid email.", email=email
        )
    if not otp or not otp.isdigit() or len(otp) != 6:
        return _auth_error(
            request,
            "forgot_password.html",
            "Please enter the 6-digit code from the email.",
            email=email,
        )
    if len(password) < 8:
        return _auth_error(
            request,
            "forgot_password.html",
            "Password must be at least 8 characters.",
            email=email,
        )
    if password != confirm:
        return _auth_error(
            request, "forgot_password.html", "Passwords do not match.", email=email
        )

    database = request.app.state.database
    user = database.get_reset_otp_user(email)
    if user is None or user["reset_token"] is None:
        # No pending reset for this address: same message, nothing to reveal.
        return _auth_error(
            request,
            "forgot_password.html",
            "That code is incorrect. Check the email and try again.",
            email=email,
        )
    if _is_expired(user.get("reset_token_expires_at")):
        return _auth_error(
            request,
            "forgot_password.html",
            "This code has expired. Request a new one.",
            email=email,
        )
    if user["reset_token_attempts"] >= _MAX_RESET_ATTEMPTS:
        return _auth_error(
            request,
            "forgot_password.html",
            "Too many incorrect attempts. Request a new code.",
            email=email,
        )
    if not secrets.compare_digest(user["reset_token"], _hash_otp(otp)):
        attempts = database.increment_reset_attempts(user["id"])
        if attempts >= _MAX_RESET_ATTEMPTS:
            message = "Too many incorrect attempts. Request a new code."
        else:
            message = "That code is incorrect. Check the email and try again."
        return _auth_error(request, "forgot_password.html", message, email=email)

    database.update_password(user_id=user["id"], password_hash=hash_password(password))
    logger.info("Password reset for user %s", user["email"])

    return render_page(
        request,
        "verification.html",
        {
            "ok": True,
            "message": "Your password has been updated. You can now log in with your new password.",
            "email": None,
        },
    )


@router.post("/auth/logout", summary="Log out")
async def logout(request: Request):
    """Clear the JWT cookie and return to the login page."""
    if is_datastar_request(request):
        return _success_response(request, {}, "", clear=True)

    response = RedirectResponse(url="/login", status_code=303)
    _clear_auth_cookie(response)
    return response
