"""Email delivery via SMTP, used for signup email verification.

Uses the standard library (``smtplib``) so no extra dependency is needed.
Compatible with Hostinger's SMTP server (``smtp.hostinger.com``) using either
SSL on port 465 or STARTTLS on port 587.
"""

import logging
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
from typing import Any
from urllib.parse import quote

from app.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Send application emails over SMTP."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        from_email: str = "",
        from_name: str = "",
        use_ssl: bool = True,
        app_name: str = "VidNotes",
        base_url: str = "http://localhost:8000",
        token_minutes: int = 1440,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_email = from_email or username
        self.from_name = (from_name or app_name).strip()
        self.use_ssl = use_ssl
        self.app_name = app_name
        self.base_url = base_url.rstrip("/")
        self.token_minutes = token_minutes

    @classmethod
    def from_settings(cls) -> "EmailService":
        """Build an instance from the application settings."""
        return cls(
            host=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            from_email=settings.smtp_from_email,
            from_name=settings.smtp_from_name,
            use_ssl=settings.smtp_use_ssl,
            app_name=settings.app_name,
            base_url=settings.app_base_url,
            token_minutes=settings.email_verify_token_minutes,
        )

    @property
    def available(self) -> bool:
        """``True`` when SMTP credentials are configured and email can be sent."""
        return bool(self.host and self.username)

    def verification_link(self, token: str) -> str:
        """Build the absolute email-verification URL for ``token``."""
        return f"{self.base_url}/auth/verify-email?token={quote(token)}"

    def _send(self, message: EmailMessage) -> None:
        """Deliver ``message`` over SMTP (SSL or STARTTLS)."""
        if self.use_ssl:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=30) as server:
                server.login(self.username, self.password)
                server.send_message(message)
            return
        with smtplib.SMTP(self.host, self.port, timeout=30) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(message)

    def send_verification_email(self, to_email: str, token: str) -> None:
        """Send a signup verification email containing ``token``'s link."""
        link = self.verification_link(token)
        hours = max(1, self.token_minutes // 60)
        subject = f"Verify your {self.app_name} account"
        body = (
            f"Hi,\n\n"
            f"Thanks for signing up for {self.app_name}!\n\n"
            f"Please verify your email address by clicking the link below "
            f"(valid for {hours} hours):\n\n"
            f"{link}\n\n"
            f"If you didn't create this account, you can safely ignore this email.\n\n"
            f"— {self.app_name}"
        )
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = formataddr((self.from_name, self.from_email))
        message["To"] = to_email
        message.set_content(body)
        self._send(message)
        logger.info("Sent verification email to %s", to_email)

    def send_password_reset_email(self, to_email: str, otp: str) -> None:
        """Send a password-reset email containing the 6-digit OTP."""
        minutes = max(5, self.token_minutes)
        subject = f"Your {self.app_name} password reset code"
        body = (
            f"Hi,\n\n"
            f"We received a request to reset the password for {to_email}.\n\n"
            f"Your one-time password reset code is:\n\n"
            f"{otp}\n\n"
            f"Enter this code on the password reset page to choose a new password. "
            f"It expires in {minutes} minutes.\n\n"
            f"If you didn't request this, you can safely ignore this email.\n\n"
            f"— {self.app_name}"
        )
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = formataddr((self.from_name, self.from_email))
        message["To"] = to_email
        message.set_content(body)
        self._send(message)
        logger.info("Sent password reset OTP to %s", to_email)
