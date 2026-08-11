"""Tests for auth, health, and the transcription flow (web + JSON API)."""

import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.exceptions import DownloadError
from app.main import app
from app.schemas.transcript import TranscribeResponse
from app.services.database import Database
from app.services.email import EmailService
from app.services.pipeline import run_transcription

VIDEO_ID = "dQw4w9WgXcQ"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
EMAIL = "tester@example.com"
PASSWORD = "test-password-123"


class FakeYouTubeService:
    """In-memory stand-in for YouTubeService to avoid real downloads."""

    def __init__(self) -> None:
        self.cleaned_up: list[str] = []

    def get_metadata(self, youtube_url: str) -> dict:
        return {"video_id": VIDEO_ID, "title": "Fake Video", "duration": 60.0, "uploader": "Fake Channel"}

    def download_audio(self, youtube_url: str) -> dict:
        return {
            "video_id": VIDEO_ID,
            "title": "Fake Video",
            "duration": 60.0,
            "uploader": "Fake Channel",
            "audio_path": "temp/fake.mp3",
        }

    def cleanup(self, video_id: str) -> None:
        self.cleaned_up.append(video_id)


class FakeTranscriptionService:
    """In-memory stand-in for TranscriptionService."""

    def transcribe(self, audio_path: str) -> dict:
        return {
            "language": "en",
            "language_probability": 0.98,
            "transcript": "Hello everyone.",
            "segments": [{"start": 0.0, "end": 4.5, "text": "Hello everyone."}],
        }


class FakeRAG:
    """In-memory stand-in for the chunking service."""

    def build_chunks(self, segments: list) -> list:
        return []


class FakeChat:
    """In-memory stand-in for ChatService with chat features disabled."""

    available = False

    async def stream_answer(self, *args, **kwargs):  # pragma: no cover
        yield None


class FakeEmailService:
    """In-memory stand-in for EmailService that records outbound mail."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.sent: list[tuple[str, str]] = []

    def send_verification_email(self, to_email: str, token: str) -> None:
        self.sent.append((to_email, token))

    def send_password_reset_email(self, to_email: str, token: str) -> None:
        self.sent.append((to_email, token))


class FakeStreamingChat(FakeChat):
    """A chat stand-in that streams a short answer for send tests."""

    available = True

    async def stream_answer(self, conversation_id, question, user_id):  # noqa: D401
        yield {"type": "sources", "data": []}
        yield {"type": "token", "data": "Hello"}
        yield {"type": "done", "data": None}


@pytest.fixture()
def client() -> TestClient:
    """A test client wired with fake heavy services and a temp database."""
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    fake_youtube = FakeYouTubeService()
    app.state.database = db
    app.state.youtube_service = fake_youtube
    app.state.transcription_service = FakeTranscriptionService()
    app.state.rag = FakeRAG()
    app.state.chat = FakeChat()
    app.state.email_service = EmailService(host="", port=465, username="", password="")
    app.state.pipeline = run_transcription
    app.state.background_tasks = set()
    app.state.active_transcriptions = set()
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cancel_background_tasks() -> None:
    """Stop any background transcription tasks left over from a test."""
    yield
    for task in list(getattr(app.state, "background_tasks", set())):
        task.cancel()


def signup(client: TestClient) -> None:
    """Register a fresh user; the client cookie jar now holds the JWT."""
    response = client.post(
        "/auth/signup",
        data={"email": EMAIL, "password": PASSWORD, "confirm": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers.get("location") == "/"
    assert "access_token" in response.headers.get("set-cookie", "")


def test_health_returns_ok(client: TestClient) -> None:
    """GET /health must return 200 with {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dashboard_redirects_anon_to_login(client: TestClient) -> None:
    """Anonymous visitors are redirected to the login page."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers.get("location") == "/login"


def test_signup_then_dashboard(client: TestClient) -> None:
    """After signup the dashboard renders with the user's email."""
    signup(client)
    response = client.get("/")
    assert response.status_code == 200
    assert EMAIL in response.text
    assert "No videos yet" in response.text


def test_transcribe_button_disables_during_transcription(client: TestClient) -> None:
    """The transcribe form carries the indicator so the button disables while running."""
    signup(client)
    response = client.get("/")
    assert response.status_code == 200
    assert "data-indicator=\"transcribing\"" in response.text
    assert "data-attr:disabled=\"$transcribing\"" in response.text
    assert "'transcribing': false" in response.text


def test_speak_link_in_header(client: TestClient) -> None:
    """The Speak page is reachable from the header nav link."""
    signup(client)
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/speak"' in response.text
    assert "@get('/speak')" in response.text


def test_login_bad_password_shows_error(client: TestClient) -> None:
    """Wrong credentials re-render the login form with an error."""
    signup(client)
    response = client.post(
        "/auth/logout",
        data={},
        follow_redirects=False,
    )
    assert response.status_code == 303
    response = client.post("/auth/login", data={"email": EMAIL, "password": "wrong-password"})
    assert response.status_code == 200
    assert "Invalid email or password" in response.text


def test_datastar_signup_swaps_to_dashboard(client: TestClient) -> None:
    """A Datastar signup swaps the page to the dashboard and sets the cookie."""
    response = client.post(
        "/auth/signup",
        headers={"Datastar-Request": "true"},
        json={"email": EMAIL, "password": PASSWORD, "confirm": PASSWORD},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "No videos yet" in response.text
    assert "history.pushState({}, '', \"/\")" in response.text
    assert "access_token" in response.headers.get("set-cookie", "")


def test_datastar_login_swaps_to_dashboard(client: TestClient) -> None:
    """A Datastar login swaps to the dashboard without a reload."""
    signup(client)
    client.post("/auth/logout", data={})
    response = client.post(
        "/auth/login",
        headers={"Datastar-Request": "true"},
        json={"email": EMAIL, "password": PASSWORD},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "No videos yet" in response.text
    assert "history.pushState({}, '', \"/\")" in response.text
    assert "access_token" in response.headers.get("set-cookie", "")


def test_datastar_login_error_rerenders_form(client: TestClient) -> None:
    """A Datastar login failure swaps the login form back with an error."""
    signup(client)
    client.post("/auth/logout", data={})
    response = client.post(
        "/auth/login",
        headers={"Datastar-Request": "true"},
        json={"email": EMAIL, "password": "wrong-password"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Invalid email or password" in response.text


def test_datastar_logout_swaps_to_login(client: TestClient) -> None:
    """A Datastar logout swaps to the login page and clears the cookie."""
    signup(client)
    response = client.post(
        "/auth/logout",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Log in" in response.text
    assert "history.pushState({}, '', \"/login\")" in response.text
    assert "access_token" in response.headers.get("set-cookie", "")


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------


def _token_for(client: TestClient) -> str:
    """Return the verification token stored for the test user."""
    user = app.state.database.get_user_by_email(EMAIL)
    with app.state.database._connect() as conn:
        row = conn.execute(
            "SELECT verification_token FROM users WHERE id = ?", (user["id"],)
        ).fetchone()
    return row[0]


def test_signup_sends_email_and_blocks_login(client: TestClient) -> None:
    """With SMTP enabled, signup sends mail, does not log in, and login is blocked."""
    mailer = FakeEmailService(available=True)
    app.state.email_service = mailer

    response = client.post(
        "/auth/signup",
        data={"email": EMAIL, "password": PASSWORD, "confirm": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers.get("location") == "/verify-email?email=tester%40example.com"
    assert "access_token" not in response.headers.get("set-cookie", "")
    assert len(mailer.sent) == 1
    assert mailer.sent[0][0] == EMAIL

    user = app.state.database.get_user_by_email(EMAIL)
    assert user["is_verified"] == 0

    response = client.post("/auth/login", data={"email": EMAIL, "password": PASSWORD})
    assert response.status_code == 200
    assert "verify your email" in response.text.lower()

def test_verify_email_token_marks_user_verified(client: TestClient) -> None:
    """Following the emailed link verifies the account and unlocks login."""
    mailer = FakeEmailService(available=True)
    app.state.email_service = mailer
    client.post(
        "/auth/signup",
        data={"email": EMAIL, "password": PASSWORD, "confirm": PASSWORD},
    )
    token = _token_for(client)
    assert token

    response = client.get(f"/auth/verify-email?token={token}")
    assert response.status_code == 200
    assert "is verified" in response.text

    assert app.state.database.get_user_by_email(EMAIL)["is_verified"] == 1

    response = client.post(
        "/auth/login",
        data={"email": EMAIL, "password": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers.get("location") == "/"
    assert "access_token" in response.headers.get("set-cookie", "")


def test_verify_email_invalid_token_shows_failure(client: TestClient) -> None:
    """A bogus or expired token renders the failure state."""
    response = client.get("/auth/verify-email?token=not-a-real-token")
    assert response.status_code == 200
    assert "invalid or has expired" in response.text


def test_resend_verification_issues_new_token(client: TestClient) -> None:
    """Resending replaces the token and sends a fresh email."""
    mailer = FakeEmailService(available=True)
    app.state.email_service = mailer
    client.post(
        "/auth/signup",
        data={"email": EMAIL, "password": PASSWORD, "confirm": PASSWORD},
    )
    old_token = _token_for(client)

    response = client.post(
        "/auth/verify/resend", data={"email": EMAIL}, follow_redirects=False
    )
    assert response.status_code == 303
    assert len(mailer.sent) == 2
    assert mailer.sent[1][0] == EMAIL
    assert mailer.sent[1][0] == EMAIL

    new_token = _token_for(client)
    assert new_token and new_token != old_token

    response = client.get(f"/auth/verify-email?token={new_token}")
    assert "is verified" in response.text


def test_datastar_signup_with_verification_swaps_to_verify_page(client: TestClient) -> None:
    """A Datastar signup swaps to the 'check your inbox' page without a cookie."""
    mailer = FakeEmailService(available=True)
    app.state.email_service = mailer

    response = client.post(
        "/auth/signup",
        headers={"Datastar-Request": "true"},
        json={"email": EMAIL, "password": PASSWORD, "confirm": PASSWORD},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Check your inbox" in response.text
    assert "history.pushState({}, '', \"/verify-email?email=" in response.text
    assert "access_token" not in response.headers.get("set-cookie", "")


# ---------------------------------------------------------------------------
# Password reset (OTP)
# ---------------------------------------------------------------------------


def _reset_otp(mailer: FakeEmailService) -> str:
    """Return the 6-digit OTP captured by the fake mailer."""
    return mailer.sent[-1][1]


def test_forgot_password_sends_email_without_leaking_account(client: TestClient) -> None:
    """A verified account gets a 6-digit OTP; unknown emails stay silent."""
    signup(client)
    mailer = FakeEmailService(available=True)
    app.state.email_service = mailer

    response = client.post(
        "/auth/forgot-password", data={"email": EMAIL}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers.get("location") == "/forgot-password?email=tester%40example.com"
    assert len(mailer.sent) == 1
    assert mailer.sent[0][0] == EMAIL
    assert len(_reset_otp(mailer)) == 6
    assert _reset_otp(mailer).isdigit()

    response = client.post(
        "/auth/forgot-password",
        data={"email": "nobody@example.com"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert len(mailer.sent) == 1


def test_reset_password_updates_credentials(client: TestClient) -> None:
    """A valid OTP sets a new password that works for login."""
    signup(client)
    mailer = FakeEmailService(available=True)
    app.state.email_service = mailer
    client.post("/auth/forgot-password", data={"email": EMAIL})
    otp = _reset_otp(mailer)
    assert otp

    page = client.get("/forgot-password?email=tester%40example.com")
    assert page.status_code == 200
    assert "Enter the code" in page.text

    new_password = "new-password-456"
    response = client.post(
        "/auth/reset-password",
        data={"email": EMAIL, "otp": otp, "password": new_password, "confirm": new_password},
        follow_redirects=False,
    )
    assert response.status_code == 200
    assert "password has been updated" in response.text

    old_login = client.post(
        "/auth/login",
        data={"email": EMAIL, "password": PASSWORD},
        follow_redirects=False,
    )
    assert old_login.status_code == 200
    assert "Invalid email or password" in old_login.text

    new_login = client.post(
        "/auth/login",
        data={"email": EMAIL, "password": new_password},
        follow_redirects=False,
    )
    assert new_login.status_code == 303
    assert new_login.headers.get("location") == "/"
    assert "access_token" in new_login.headers.get("set-cookie", "")


def test_reset_password_wrong_otp_shows_error_and_counts_attempts(client: TestClient) -> None:
    """A wrong OTP is rejected and failed attempts are tracked."""
    signup(client)
    mailer = FakeEmailService(available=True)
    app.state.email_service = mailer
    client.post("/auth/forgot-password", data={"email": EMAIL})

    for _ in range(3):
        response = client.post(
            "/auth/reset-password",
            data={
                "email": EMAIL,
                "otp": "000000",
                "password": "new-password-456",
                "confirm": "new-password-456",
            },
        )
        assert response.status_code == 200
        assert "That code is incorrect" in response.text

    user = app.state.database.get_user_by_email(EMAIL)
    with app.state.database._connect() as conn:
        attempts = conn.execute(
            "SELECT reset_token_attempts FROM users WHERE id = ?", (user["id"],)
        ).fetchone()[0]
    assert attempts == 3


def test_reset_password_requires_matching_password(client: TestClient) -> None:
    """Mismatched or short passwords re-render the form with an error."""
    signup(client)
    mailer = FakeEmailService(available=True)
    app.state.email_service = mailer
    client.post("/auth/forgot-password", data={"email": EMAIL})
    otp = _reset_otp(mailer)

    response = client.post(
        "/auth/reset-password",
        data={"email": EMAIL, "otp": otp, "password": "new-password-456", "confirm": "different-123"},
    )
    assert response.status_code == 200
    assert "Passwords do not match" in response.text

    response = client.post(
        "/auth/reset-password",
        data={"email": EMAIL, "otp": otp, "password": "short", "confirm": "short"},
    )
    assert response.status_code == 200
    assert "at least 8 characters" in response.text


def test_api_transcribe_requires_auth(client: TestClient) -> None:
    """POST /api/v1/transcribe without a session returns 401."""
    response = client.post("/api/v1/transcribe", json={"youtube_url": WATCH_URL})
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required."}


def test_api_transcribe_full_flow_with_mocked_services(client: TestClient, monkeypatch) -> None:
    """Authenticated transcribe returns a valid response and cleans up."""
    signup(client)
    response = client.post("/api/v1/transcribe", json={"youtube_url": WATCH_URL})

    assert response.status_code == 200
    payload = TranscribeResponse(**response.json())
    assert payload.video_id == VIDEO_ID
    assert payload.title == "Fake Video"
    assert payload.uploader == "Fake Channel"
    assert payload.language == "en"
    assert payload.transcript == "Hello everyone."
    assert len(payload.segments) == 1
    assert payload.segments[0].text == "Hello everyone."


def test_api_transcribe_invalid_url_returns_400(client: TestClient) -> None:
    """A non-YouTube URL must be rejected with 400."""
    signup(client)
    response = client.post("/api/v1/transcribe", json={"youtube_url": "https://google.com"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid YouTube URL."}


def test_api_transcribe_empty_url_rejected(client: TestClient) -> None:
    """An empty URL must be rejected with 400."""
    signup(client)
    response = client.post("/api/v1/transcribe", json={"youtube_url": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid YouTube URL."}


def test_chat_stt_requires_auth(client: TestClient) -> None:
    """POST /api/v1/chat/stt without a session returns 401."""
    response = client.post("/api/v1/chat/stt", content=b"audio-bytes")
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required."}


def test_chat_stt_transcribes_audio(client: TestClient) -> None:
    """Authenticated dictation returns the transcribed text as JSON."""
    signup(client)
    response = client.post(
        "/api/v1/chat/stt",
        headers={"Content-Type": "audio/webm"},
        content=b"\x1a\x45\xdf\xa3fake-webm-bytes",
    )
    assert response.status_code == 200
    assert response.json() == {"text": "Hello everyone."}


def test_chat_stt_empty_body_rejected(client: TestClient) -> None:
    """A request with no audio bytes must be rejected with 400."""
    signup(client)
    response = client.post("/api/v1/chat/stt", content=b"")
    assert response.status_code == 400
    assert response.json() == {"detail": "No audio data received."}


def test_chat_stt_allows_large_audio(client: TestClient) -> None:
    """The STT route bypasses the small-body limit for multi-MB audio."""
    signup(client)
    response = client.post(
        "/api/v1/chat/stt",
        headers={"Content-Type": "audio/webm"},
        content=b"x" * (100 * 1024),
    )
    assert response.status_code == 200
    assert response.json() == {"text": "Hello everyone."}


def test_datastar_create_category_patches_list(client: TestClient) -> None:
    """A Datastar POST streams an SSE patch that re-renders the category list."""
    signup(client)
    response = client.post(
        "/categories",
        headers={"Datastar-Request": "true"},
        json={"category_name": "Tech"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "Tech" in body
    assert "event: datastar-patch-elements" in body
    assert "event: datastar-patch-signals" in body


def test_datastar_create_category_requires_auth(client: TestClient) -> None:
    """Anonymous Datastar actions swap to the login page via SSE (no reload)."""
    response = client.post(
        "/categories",
        headers={"Datastar-Request": "true"},
        json={"category_name": "Tech"},
    )
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/login\")" in response.text
    assert "Log in" in response.text


def test_datastar_delete_category_uses_icon(client: TestClient) -> None:
    """Category rows render a delete SVG icon and deleting removes the row."""
    signup(client)
    client.post(
        "/categories",
        headers={"Datastar-Request": "true"},
        json={"category_name": "Tech"},
    )
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert 'aria-label="Add category"' in dashboard.text
    assert 'aria-label="Delete category"' in dashboard.text
    assert 'viewBox="0 0 24 24"' in dashboard.text
    assert "@setAll(true, {include: /^confirm_delete_" in dashboard.text
    assert "@set({" not in dashboard.text

    category_id = app.state.database.list_categories(
        app.state.database.get_user_by_email(EMAIL)["id"]
    )[0]["id"]
    response = client.post(
        f"/categories/{category_id}/delete",
        headers={"Datastar-Request": "true"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "id: category-list" in body or "category-list" in body
    assert "Tech" not in body

    database = app.state.database
    assert database.list_categories(database.get_user_by_email(EMAIL)["id"]) == []


def test_chat_send_streams_bouncing_balls(client: TestClient) -> None:
    """While the assistant answers, the stream shows a bouncing-dots bubble."""
    signup(client)
    client.post("/api/v1/transcribe", json={"youtube_url": WATCH_URL})
    video_id = app.state.database.list_videos(
        app.state.database.get_user_by_email(EMAIL)["id"]
    )[0]["id"]
    app.state.chat = FakeStreamingChat()

    response = client.post(
        f"/videos/{video_id}/chat/send",
        headers={"Datastar-Request": "true"},
        json={"message": "hi"},
    )
    assert response.status_code == 200
    body = response.text
    assert "answer-slot" in body
    assert 'class="dots"' in body
    assert "Thinking…" in body
    assert "Hello" in body


def test_video_page_after_transcribe(client: TestClient) -> None:
    """The stored video shows on the dashboard and its transcript page."""
    signup(client)
    client.post("/api/v1/transcribe", json={"youtube_url": WATCH_URL})

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Fake Video" in dashboard.text
    assert 'data-tooltip="Chat with this video"' in dashboard.text

    video_id = app.state.database.list_videos(
        app.state.database.get_user_by_email(EMAIL)["id"]
    )[0]["id"]
    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "Hello everyone." in page.text

def test_datastar_delete_video_from_dashboard(client: TestClient) -> None:
    """Deleting a video card removes the video and its transcript rows."""
    signup(client)
    client.post("/api/v1/transcribe", json={"youtube_url": WATCH_URL})
    video_id = app.state.database.list_videos(
        app.state.database.get_user_by_email(EMAIL)["id"]
    )[0]["id"]

    response = client.post(
        f"/videos/{video_id}/delete",
        headers={"Datastar-Request": "true"},
        json={"from_dashboard": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "id: videos-panel" in body or "videos-panel" in body
    assert "No videos yet" in body
    assert "Fake Video" not in body
    assert "history.pushState" not in body
    assert "@set({" not in body

    database = app.state.database
    assert database.list_videos(database.get_user_by_email(EMAIL)["id"]) == []
    with database._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM segments WHERE video_id = ?", (video_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE video_id = ?", (video_id,)
        ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Background transcription (web action + dashboard poller)
# ---------------------------------------------------------------------------


def _user_id() -> int:
    """Return the id of the registered test user."""
    return app.state.database.get_user_by_email(EMAIL)["id"]


def _wait_for_status(database: Database, video_id: int, expected: str, timeout: float = 5.0) -> dict:
    """Poll the database until the video reaches ``expected`` status."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        video = database.get_video(video_id, _user_id())
        if video is not None and video["status"] == expected:
            return video
        time.sleep(0.05)
    raise AssertionError(f"video {video_id} never reached status {expected!r}")


def test_transcribe_duplicate_submission_is_rejected(client: TestClient) -> None:
    """A second submission for a video already being transcribed is refused."""
    signup(client)
    app.state.active_transcriptions.add((_user_id(), VIDEO_ID))
    response = client.post(
        "/transcribe",
        headers={"Datastar-Request": "true"},
        json={"youtube_url": WATCH_URL},
    )
    assert response.status_code == 200
    assert "Already transcribing this video" in response.text
    assert app.state.database.list_videos(_user_id()) == []
    assert app.state.background_tasks == set()


def test_transcribe_runs_in_background_and_completes(client: TestClient) -> None:
    """The web action queues the pipeline and the card flips to ready."""
    signup(client)
    response = client.post(
        "/transcribe",
        headers={"Datastar-Request": "true"},
        json={"youtube_url": WATCH_URL},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "Transcribing in the background" in body
    assert 'data-status="processing"' in body
    assert "data-on-interval__duration.2s" in body
    assert "@get('/fragments/videos-panel')" in body

    video_id = app.state.database.list_videos(_user_id())[0]["id"]
    video = _wait_for_status(app.state.database, video_id, "ready")
    assert video["title"] == "Fake Video"
    assert video["transcript"] == "Hello everyone."
    assert video["error"] is None

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert 'data-status="ready"' in dashboard.text
    assert "Transcribing in the background" not in dashboard.text


def test_transcribe_background_error_marks_row(client: TestClient) -> None:
    """A failing pipeline marks the pending row as error with a message."""
    signup(client)

    class FailingYouTube(FakeYouTubeService):
        def get_metadata(self, youtube_url: str) -> dict:
            raise DownloadError()

    app.state.youtube_service = FailingYouTube()
    response = client.post(
        "/transcribe",
        headers={"Datastar-Request": "true"},
        json={"youtube_url": WATCH_URL},
    )
    assert response.status_code == 200
    video_id = app.state.database.list_videos(_user_id())[0]["id"]
    video = _wait_for_status(app.state.database, video_id, "error")
    assert video["error"] == DownloadError.detail


def test_transcribe_action_rejects_invalid_url(client: TestClient) -> None:
    """A non-YouTube URL is rejected without creating a pending row."""
    signup(client)
    response = client.post(
        "/transcribe",
        headers={"Datastar-Request": "true"},
        json={"youtube_url": "https://google.com"},
    )
    assert response.status_code == 200
    assert "Invalid YouTube URL" in response.text
    assert app.state.database.list_videos(_user_id()) == []


def test_transcribe_action_rejects_empty_url(client: TestClient) -> None:
    """An empty URL shows a hint and creates no video row."""
    signup(client)
    response = client.post(
        "/transcribe",
        headers={"Datastar-Request": "true"},
        json={"youtube_url": ""},
    )
    assert response.status_code == 200
    assert "Enter a YouTube URL first" in response.text
    assert app.state.database.list_videos(_user_id()) == []


def test_dashboard_shows_processing_card_and_poller(client: TestClient) -> None:
    """A processing row renders a spinner card plus the interval poller."""
    signup(client)
    app.state.database.create_pending_video(
        user_id=_user_id(), youtube_id=VIDEO_ID, youtube_url=WATCH_URL, title="Transcribing…"
    )
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert 'data-status="processing"' in body
    assert "Transcribing in the background" in body
    assert "data-on-interval__duration.2s" in body
    assert "@get('/fragments/videos-panel')" in body


def test_create_pending_video_deduplicates_processing(client: TestClient) -> None:
    """Submitting the same URL twice reuses the same processing row."""
    signup(client)
    database = app.state.database
    first = database.create_pending_video(
        user_id=_user_id(), youtube_id=VIDEO_ID, youtube_url=WATCH_URL, title="Transcribing…"
    )
    second = database.create_pending_video(
        user_id=_user_id(), youtube_id=VIDEO_ID, youtube_url=WATCH_URL, title="Transcribing…"
    )
    assert first == second
    assert len(database.list_videos(_user_id())) == 1


def test_videos_panel_fragment_requires_auth(client: TestClient) -> None:
    """The fragment endpoint refuses anonymous requests."""
    response = client.get("/fragments/videos-panel")
    assert response.status_code == 401


def test_videos_panel_fragment_renders(client: TestClient) -> None:
    """The fragment returns plain HTML and an SSE patch for Datastar."""
    signup(client)
    plain = client.get("/fragments/videos-panel")
    assert plain.status_code == 200
    assert "videos-panel" in plain.text
    assert "No videos yet" in plain.text

    sse = client.get("/fragments/videos-panel", headers={"Datastar-Request": "true"})
    assert sse.status_code == 200
    assert sse.headers["content-type"].startswith("text/event-stream")
    assert "event: datastar-patch-elements" in sse.text
    assert "videos-panel" in sse.text


def test_video_page_shows_processing_state(client: TestClient) -> None:
    """Navigating to a still-processing video renders a status page."""
    signup(client)
    video_id = app.state.database.create_pending_video(
        user_id=_user_id(), youtube_id=VIDEO_ID, youtube_url=WATCH_URL, title="Transcribing…"
    )
    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "Transcribing in the background" in page.text
    assert "Hello everyone." not in page.text


def test_schema_v2_migrates_to_v4(tmp_path: Path) -> None:
    """A v2 database file is upgraded in place with status/error columns."""
    import sqlite3

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            youtube_id TEXT NOT NULL,
            youtube_url TEXT NOT NULL,
            title TEXT NOT NULL,
            uploader TEXT,
            language TEXT NOT NULL,
            language_probability REAL NOT NULL,
            duration REAL NOT NULL,
            transcript TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (user_id, youtube_id)
        );
        """
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    Database(db_path)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        video_columns = {
            name
            for name, _ in conn.execute(
                "SELECT name, type FROM pragma_table_info('videos') "
                "WHERE name IN ('status', 'error')"
            ).fetchall()
        }
        assert video_columns == {"status", "error"}
        user_columns = {
            name
            for name, _ in conn.execute(
                "SELECT name, type FROM pragma_table_info('users') "
                "WHERE name IN ('is_verified', 'verification_token', "
                "'verification_token_expires_at')"
            ).fetchall()
        }
        assert user_columns == {"is_verified", "verification_token", "verification_token_expires_at"}


def test_schema_v3_migrates_to_v4_verifies_existing_users(tmp_path: Path) -> None:
    """A v3 database gains the verification columns and existing users stay verified."""
    import sqlite3

    db_path = tmp_path / "legacy-v3.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            youtube_id TEXT NOT NULL,
            youtube_url TEXT NOT NULL,
            title TEXT NOT NULL,
            uploader TEXT,
            language TEXT NOT NULL,
            language_probability REAL NOT NULL,
            duration REAL NOT NULL,
            transcript TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            error TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (user_id, youtube_id)
        );
        """
    )
    conn.execute(
        "INSERT INTO users (email, password_hash) VALUES ('old@example.com', 'hash')"
    )
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()

    db = Database(db_path)

    user = db.get_user_by_email("old@example.com")
    assert user is not None
    assert user["is_verified"] == 1


def test_schema_v3_heals_missing_status_columns(tmp_path: Path) -> None:
    """A v3 database without the status columns is repaired on startup."""
    import sqlite3

    db_path = tmp_path / "half-migrated.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            youtube_id TEXT NOT NULL,
            youtube_url TEXT NOT NULL,
            title TEXT NOT NULL,
            uploader TEXT,
            language TEXT NOT NULL,
            language_probability REAL NOT NULL,
            duration REAL NOT NULL,
            transcript TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (user_id, youtube_id)
        );
        """
    )
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()

    Database(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {
            name
            for name, _ in conn.execute(
                "SELECT name, type FROM pragma_table_info('videos') "
                "WHERE name IN ('status', 'error')"
            ).fetchall()
        }
        assert columns == {"status", "error"}
