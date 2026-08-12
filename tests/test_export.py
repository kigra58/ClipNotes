"""Tests for transcript export (SRT / VTT / TXT / Markdown / PDF)."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.database import Database
from app.services.email import EmailService
from app.services.exporter import (
    EXPORT_FORMATS,
    export,
    export_markdown,
    export_pdf,
    export_srt,
    export_txt,
    export_vtt,
    slugify,
    srt_timestamp,
    vtt_timestamp,
)

VIDEO_ID = "dQw4w9WgXcQ"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
EMAIL = "export@example.com"
OTHER_EMAIL = "other-export@example.com"
PASSWORD = "test-password-123"

SEGMENTS = [
    {"start": 0.0, "end": 4.5, "text": "Hello everyone."},
    {"start": 4.5, "end": 8.0, "text": "This is a test."},
    {"start": 8.0, "end": 9.0, "text": "   "},
]

VIDEO = {
    "title": "Fake Video",
    "uploader": "Fake Channel",
    "youtube_url": WATCH_URL,
    "duration": 60.0,
    "segments": SEGMENTS,
}


class FakeYouTubeService:
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
        pass


class FakeTranscriptionService:
    def transcribe(self, audio_path: str) -> dict:
        return {
            "language": "en",
            "language_probability": 0.98,
            "transcript": "Hello everyone. This is a test.",
            "segments": [s for s in SEGMENTS if s["text"].strip()],
        }


class FakeRAG:
    def build_chunks(self, segments: list) -> list:
        return []


class FakeChat:
    available = False


@pytest.fixture()
def client() -> TestClient:
    """A test client wired with fake heavy services and a temp database."""
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    app.state.database = db
    app.state.youtube_service = FakeYouTubeService()
    app.state.transcription_service = FakeTranscriptionService()
    app.state.rag = FakeRAG()
    app.state.chat = FakeChat()
    app.state.email_service = EmailService(host="", port=465, username="", password="")
    return TestClient(app)


def signup(client: TestClient, email: str = EMAIL) -> None:
    response = client.post(
        "/auth/signup",
        data={"email": email, "password": PASSWORD, "confirm": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 303


def transcribe_video(client: TestClient) -> int:
    response = client.post("/api/v1/transcribe", json={"youtube_url": WATCH_URL})
    assert response.status_code == 200
    return response.json()["transcript_id"]


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (65.5, "00:01:05,500"),
        (3723.0, "01:02:03,000"),
        (0.4, "00:00:00,400"),
    ],
)
def test_srt_timestamp(seconds: float, expected: str) -> None:
    assert srt_timestamp(seconds) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00.000"),
        (65.5, "00:01:05.500"),
        (3723.0, "01:02:03.000"),
    ],
)
def test_vtt_timestamp(seconds: float, expected: str) -> None:
    assert vtt_timestamp(seconds) == expected


# ---------------------------------------------------------------------------
# Pure format functions
# ---------------------------------------------------------------------------


def test_export_srt_skips_blank_segments() -> None:
    content = export_srt(SEGMENTS).decode("utf-8")
    assert content == (
        "1\n"
        "00:00:00,000 --> 00:00:04,500\n"
        "Hello everyone.\n"
        "\n"
        "2\n"
        "00:00:04,500 --> 00:00:08,000\n"
        "This is a test.\n"
    )


def test_export_vtt_skips_blank_segments() -> None:
    content = export_vtt(SEGMENTS).decode("utf-8")
    assert content == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:04.500\n"
        "Hello everyone.\n"
        "\n"
        "00:00:04.500 --> 00:00:08.000\n"
        "This is a test.\n"
    )


def test_export_txt_skips_blank_segments() -> None:
    content = export_txt(SEGMENTS).decode("utf-8")
    assert content == (
        "[0:00-0:04] Hello everyone.\n"
        "[0:04-0:08] This is a test.\n"
    )


def test_export_markdown_includes_metadata() -> None:
    content = export_markdown(**VIDEO).decode("utf-8")
    assert content.startswith("# Fake Video")
    assert "- **Channel:** Fake Channel" in content
    assert f"- **Source:** {WATCH_URL}" in content
    assert "- **Duration:** 1:00" in content
    assert "## Transcript" in content
    assert "**0:00**–**0:04**" in content
    assert "Hello everyone." in content
    assert "   " not in content


def test_export_markdown_without_metadata() -> None:
    content = export_markdown(
        title="", uploader=None, youtube_url="", duration=0.0, segments=SEGMENTS
    ).decode("utf-8")
    assert content.startswith("# Transcript")
    assert "**Channel:**" not in content
    assert "**Source:**" not in content


def test_export_pdf_produces_valid_document() -> None:
    content = export_pdf(**VIDEO)
    assert content.startswith(b"%PDF")
    assert content.rstrip().endswith(b"%%EOF")


def test_export_returns_filename_media_type_and_content() -> None:
    expected = {
        "srt": ("fake-video.srt", "application/x-subrip"),
        "vtt": ("fake-video.vtt", "text/vtt"),
        "txt": ("fake-video.txt", "text/plain; charset=utf-8"),
        "md": ("fake-video.md", "text/markdown; charset=utf-8"),
        "pdf": ("fake-video.pdf", "application/pdf"),
    }
    for fmt in EXPORT_FORMATS:
        filename, media_type, content = export(fmt, **VIDEO)
        assert filename == expected[fmt][0]
        assert media_type == expected[fmt][1]
        assert content


def test_export_unsupported_format_raises() -> None:
    with pytest.raises(ValueError):
        export("docx", **VIDEO)


def test_slugify() -> None:
    assert slugify("Fake Video! (Part 1)") == "fake-video-part-1"
    assert slugify("") == ""
    assert slugify("  only spaces  ") == "only-spaces"


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


def test_export_requires_auth(client: TestClient) -> None:
    response = client.get(f"/api/v1/videos/1/export?format=srt")
    assert response.status_code == 401


def test_export_unknown_video_returns_404(client: TestClient) -> None:
    signup(client)
    response = client.get("/api/v1/videos/999/export?format=srt")
    assert response.status_code == 404


def test_export_srt_download(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/api/v1/videos/{video_id}/export?format=srt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-subrip")
    assert response.headers["content-disposition"] == 'attachment; filename="fake-video.srt"'
    assert "Hello everyone." in response.text


def test_export_vtt_download(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/api/v1/videos/{video_id}/export?format=vtt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/vtt")
    assert response.text.startswith("WEBVTT")


def test_export_txt_download(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/api/v1/videos/{video_id}/export?format=txt")
    assert response.status_code == 200
    assert "[0:00-0:04] Hello everyone." in response.text


def test_export_markdown_download(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/api/v1/videos/{video_id}/export?format=md")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "# Fake Video" in response.text


def test_export_pdf_download(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/api/v1/videos/{video_id}/export?format=pdf")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")
    assert response.content.rstrip().endswith(b"%%EOF")


def test_export_defaults_to_srt(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/api/v1/videos/{video_id}/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-subrip")


def test_export_unsupported_format_returns_400(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/api/v1/videos/{video_id}/export?format=docx")
    assert response.status_code == 400
    assert "Unsupported export format" in response.json()["detail"]


def test_export_rejects_other_users_video(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)

    other = TestClient(app)
    signup(other, email=OTHER_EMAIL)
    response = other.get(f"/api/v1/videos/{video_id}/export?format=srt")
    assert response.status_code == 404
