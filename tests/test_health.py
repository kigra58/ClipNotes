"""Tests for auth, health, and the transcription flow (web + JSON API)."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.transcript import TranscribeResponse
from app.services.database import Database
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
    app.state.pipeline = run_transcription
    return TestClient(app)


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


def test_video_page_after_transcribe(client: TestClient) -> None:
    """The stored video shows on the dashboard and its transcript page."""
    signup(client)
    client.post("/api/v1/transcribe", json={"youtube_url": WATCH_URL})

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Fake Video" in dashboard.text

    video_id = app.state.database.list_videos(
        app.state.database.get_user_by_email(EMAIL)["id"]
    )[0]["id"]
    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "Hello everyone." in page.text
