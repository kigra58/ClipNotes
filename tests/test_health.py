"""Tests for the health endpoint and API error handling."""

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.transcript import TranscribeResponse

VIDEO_ID = "dQw4w9WgXcQ"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class FakeYouTubeService:
    """In-memory stand-in for YouTubeService to avoid real downloads."""

    def __init__(self) -> None:
        self.cleaned_up: list[str] = []

    def get_metadata(self, youtube_url: str) -> dict:
        return {"video_id": VIDEO_ID, "title": "Fake Video", "duration": 60.0, "uploader": "Fake Channel"}

    def download_audio(self, youtube_url: str) -> dict:
        return {"video_id": VIDEO_ID, "title": "Fake Video", "duration": 60.0, "uploader": "Fake Channel", "audio_path": "temp/fake.mp3"}

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


class FakeDatabase:
    """In-memory stand-in for the persistence layer."""

    def save_transcript(self, **kwargs) -> int:
        return 1


class FakeRAG:
    """In-memory stand-in for the chunking service."""

    def build_chunks(self, segments: list) -> list:
        return []


def build_client() -> tuple[TestClient, FakeYouTubeService]:
    """Create a test client wired with fake services (no lifespan, no model).

    Returns:
        The test client and the fake YouTube service.
    """
    fake_youtube = FakeYouTubeService()
    app.state.youtube_service = fake_youtube
    app.state.transcription_service = FakeTranscriptionService()
    app.state.database = FakeDatabase()
    app.state.rag = FakeRAG()
    return TestClient(app), fake_youtube


def test_health_returns_ok() -> None:
    """GET /health must return 200 with {"status": "ok"}."""
    client, _ = build_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_full_flow_with_mocked_services() -> None:
    """The transcribe endpoint returns a valid response and cleans up."""
    client, fake_youtube = build_client()
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
    assert fake_youtube.cleaned_up == [VIDEO_ID]


def test_transcribe_invalid_url_returns_400() -> None:
    """A non-YouTube URL must be rejected with 400."""
    client, _ = build_client()
    response = client.post(
        "/api/v1/transcribe",
        json={"youtube_url": "https://google.com"},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid YouTube URL."}


def test_transcribe_empty_url_rejected() -> None:
    """An empty URL must be rejected with 400."""
    client, _ = build_client()
    response = client.post("/api/v1/transcribe", json={"youtube_url": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid YouTube URL."}
