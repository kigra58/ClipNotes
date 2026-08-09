"""Authentication: PBKDF2 password hashing and JWT token helpers.

Uses the standard library for password hashing and PyJWT for signed tokens.
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from app.config import settings

_ALGORITHM = "HS256"
_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-SHA256 and a random salt.

    Args:
        password: The plaintext password.

    Returns:
        A self-describing string: ``pbkdf2$<iterations>$<salt>$<hash>``.
    """
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERATIONS
    ).hex()
    return f"pbkdf2${_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash.

    Args:
        password: The plaintext password to check.
        stored: The stored hash string produced by :func:`hash_password`.

    Returns:
        ``True`` if the password matches, ``False`` otherwise.
    """
    try:
        algorithm, iterations, salt, expected = stored.split("$", 3)
        if algorithm != "pbkdf2" or not iterations.isdigit():
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        ).hex()
        return hmac.compare_digest(digest, expected)
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int) -> str:
    """Create a signed JWT for the given user id.

    Args:
        user_id: The user id to embed in the token.

    Returns:
        A signed JWT string.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> int | None:
    """Decode and validate a JWT, returning the embedded user id.

    Args:
        token: The JWT string.

    Returns:
        The user id, or ``None`` if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        return None


def bearer_token(request: Any) -> str | None:
    """Extract a token from the ``Authorization`` header, if present.

    Args:
        request: The incoming Starlette/FastAPI request.

    Returns:
        The bearer token, or ``None``.
    """
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def request_user_id(request: Any) -> int | None:
    """Resolve the authenticated user id from the request.

    Checks the auth cookie first, then the ``Authorization`` header.

    Args:
        request: The incoming Starlette/FastAPI request.

    Returns:
        The user id, or ``None`` if the request is unauthenticated.
    """
    token = request.cookies.get(settings.cookie_name) or bearer_token(request)
    if not token:
        return None
    return decode_access_token(token)
