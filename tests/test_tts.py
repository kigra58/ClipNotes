"""Tests for the text-to-speech feature (web page, Datastar action, JSON API)."""

import ast
import io
import tempfile
import wave
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.database import Database
from app.services.pipeline import run_transcription

VIDEO_ID = "dQw4w9WgXcQ"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
EMAIL = "tts@example.com"
PASSWORD = "test-password-123"
OTHER_EMAIL = "other@example.com"


def _write_wav(path_or_buffer: str | io.BytesIO) -> None:
    """Write a minimal valid 16-bit mono WAV file."""
    with wave.open(path_or_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"\x00\x00" * 1000)


class FakeYouTubeService:
    """In-memory stand-in for YouTubeService to avoid real downloads."""

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


class FakeTTSService:
    """In-memory stand-in for TTSService that writes minimal WAV files."""

    def __init__(self) -> None:
        self.cache_dir = Path(tempfile.mkdtemp())
        self.max_chars = 10000
        self.synthesis_timeout_seconds = 300
        self.available = True
        self.synthesized: list[str] = []

    def _check(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("Text is empty.")
        if len(cleaned) > self.max_chars:
            raise ValueError(f"Text exceeds the {self.max_chars} character limit.")
        return cleaned

    def synthesize_to_file(self, text: str, out_path: Path) -> float:
        self.synthesized.append(self._check(text))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(str(out_path))
        return 0.25

    def synthesize_bytes(self, text: str) -> tuple[bytes, float]:
        self.synthesized.append(self._check(text))
        buffer = io.BytesIO()
        _write_wav(buffer)
        return buffer.getvalue(), 0.25

    def cache_audio(self, text: str, user_id: int) -> Path:
        self.synthesized.append(self._check(text))
        out_path = self.cache_dir / f"{user_id}_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav"
        _write_wav(str(out_path))
        return out_path

    def cleanup_stale(self) -> int:
        return 0


@pytest.fixture()
def client() -> TestClient:
    """A test client wired with fake heavy services and a temp database."""
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    app.state.database = db
    app.state.youtube_service = FakeYouTubeService()
    app.state.transcription_service = FakeTranscriptionService()
    app.state.rag = FakeRAG()
    app.state.chat = FakeChat()
    app.state.pipeline = run_transcription
    app.state.tts_service = FakeTTSService()
    return TestClient(app)


def signup(client: TestClient, email: str = EMAIL) -> None:
    """Register a user; the client cookie jar now holds the JWT."""
    response = client.post(
        "/auth/signup",
        data={"email": email, "password": PASSWORD, "confirm": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 303


def transcribe_video(client: TestClient) -> int:
    """Create a stored video via the JSON API and return its id."""
    response = client.post("/api/v1/transcribe", json={"youtube_url": WATCH_URL})
    assert response.status_code == 200
    return response.json()["transcript_id"]


class _SignalsParser(HTMLParser):
    """Extract the form's ``data-signals`` attribute (entities decoded)."""

    def __init__(self) -> None:
        super().__init__()
        self.value = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "data-signals":
                self.value = value


def parse_signals_attribute(html: str) -> dict:
    """Parse the form's data-signals attribute as a dict."""
    parser = _SignalsParser()
    parser.feed(html)
    assert parser.value is not None, "data-signals attribute not found"
    return ast.literal_eval(
        parser.value.replace("null", "None").replace("false", "False").replace("true", "True")
    )


# ---------------------------------------------------------------------------
# Speak page
# ---------------------------------------------------------------------------


def test_speak_page_redirects_anon(client: TestClient) -> None:
    """Anonymous visitors are redirected to the login page."""
    response = client.get("/speak", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers.get("location") == "/login"


def test_speak_page_renders(client: TestClient) -> None:
    """The speak page renders with the input form."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    assert "Speak a transcript" in response.text
    assert "tts_text" in response.text


def test_speak_page_spinner_tied_to_speaking_signal(client: TestClient) -> None:
    """The spinner shows only while synthesizing, never next to the Ready message."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    assert 'data-show="$speaking && !$tts_error"' in response.text
    assert 'data-show="$tts_status && !$tts_error"' not in response.text


def test_speak_page_prefills_from_video(client: TestClient) -> None:
    """?video_id prefills the textarea with the stored transcript."""
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/speak?video_id={video_id}")
    assert response.status_code == 200
    assert "Hello everyone." in response.text


def test_speak_page_ignores_unknown_video(client: TestClient) -> None:
    """An unknown or foreign video id leaves the form empty."""
    signup(client)
    response = client.get("/speak?video_id=999")
    assert response.status_code == 200
    assert "Hello everyone." not in response.text


def test_speak_page_does_not_leak_other_users_transcript(client: TestClient) -> None:
    """A user cannot prefill another user's transcript via ?video_id."""
    signup(client)
    video_id = transcribe_video(client)
    signup(client, email=OTHER_EMAIL)
    response = client.get(f"/speak?video_id={video_id}")
    assert response.status_code == 200
    assert "Hello everyone." not in response.text
    assert "tts_text" in response.text


def test_speak_page_signals_attribute_parses(client: TestClient) -> None:
    """The data-signals attribute survives HTML parsing as valid signal values."""
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/speak?video_id={video_id}")
    assert response.status_code == 200
    signals = parse_signals_attribute(response.text)
    assert signals["tts_text"] == "Hello everyone."
    assert signals["speaking"] is False
    assert signals["tts_error"] == ""
    assert signals["tts_stats"] == {
        "characters": None,
        "words": None,
        "sentences": None,
        "estimated_duration_seconds": None,
    }


def test_speak_page_stats_subscribe_to_nested_paths(client: TestClient) -> None:
    """The stat cells must read nested signal paths so Datastar updates them."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    for path in (
        "$tts_stats.characters",
        "$tts_stats.words",
        "$tts_stats.sentences",
        "$tts_stats.estimated_duration_seconds",
    ):
        assert path in response.text
    assert "($tts_stats || {})" not in response.text


def test_speak_page_signals_escape_quotes_and_newlines(client: TestClient) -> None:
    """Transcripts with double quotes and newlines stay valid inside the attribute."""
    signup(client)
    user = app.state.database.get_user_by_email(EMAIL)
    tricky = 'He said "hi everyone."\nAnd then: "bye".'
    video_id = app.state.database.save_video(
        user_id=user["id"],
        youtube_id="tricky01",
        youtube_url=f"https://www.youtube.com/watch?v=tricky01",
        title="Tricky",
        uploader="Channel",
        language="en",
        language_probability=0.99,
        duration=60.0,
        transcript=tricky,
        segments=[{"start": 0.0, "end": 1.0, "text": tricky}],
        chunks=[],
    )
    response = client.get(f"/speak?video_id={video_id}")
    assert response.status_code == 200
    signals = parse_signals_attribute(response.text)
    assert signals["tts_text"] == tricky


def test_speak_page_global_helper_defined_before_datastar(client: TestClient) -> None:
    """speak.js must load before datastar.js so its global is set when Datastar evaluates."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    speak_pos = response.text.index("static/speak.js")
    datastar_pos = response.text.index("static/datastar.js")
    assert speak_pos < datastar_pos


# ---------------------------------------------------------------------------
# Datastar action
# ---------------------------------------------------------------------------


def test_speak_action_synthesizes(client: TestClient) -> None:
    """A Datastar POST synthesizes audio and patches the stats + player URL."""
    signup(client)
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello world."},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: datastar-patch-signals" in body
    assert "tts_url" in body
    assert "tts_stats" in body
    assert "/tts/audio/" in body
    assert '"tts_stats":null' not in body


def test_speak_action_does_not_reset_stats_to_null(client: TestClient) -> None:
    """The synthesis-reset patch must not null out tts_stats (breaks signal updates)."""
    signup(client)
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello world."},
    )
    assert response.status_code == 200
    events = [e for e in response.text.split("\n\n") if e.strip()]
    assert len(events) == 2
    assert "tts_stats" not in events[0]
    assert '"characters":12' in events[1]


def test_speak_action_rejects_empty_text(client: TestClient) -> None:
    """An empty transcript produces a friendly error patch."""
    signup(client)
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "   "},
    )
    assert response.status_code == 200
    assert "Enter some text to speak first." in response.text


def test_speak_action_requires_auth(client: TestClient) -> None:
    """Anonymous Datastar actions swap to the login page via SSE (no reload)."""
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello"},
    )
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/login\")" in response.text


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


def test_api_tts_analyze_requires_auth(client: TestClient) -> None:
    """POST /api/v1/tts/analyze without a session returns 401."""
    response = client.post("/api/v1/tts/analyze", json={"text": "Hello."})
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required."}


def test_api_tts_analyze_returns_stats(client: TestClient) -> None:
    """The analyze endpoint reports counts and an estimated reading time."""
    signup(client)
    response = client.post("/api/v1/tts/analyze", json={"text": "Hello world. This is a test."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["characters"] == len("Hello world. This is a test.")
    assert payload["words"] == 6
    assert payload["sentences"] == 2
    assert payload["estimated_duration_seconds"] == pytest.approx(6 * 60.0 / 200)


def test_api_tts_speak_requires_auth(client: TestClient) -> None:
    """POST /api/v1/tts without a session returns 401."""
    response = client.post("/api/v1/tts", json={"text": "Hello."})
    assert response.status_code == 401


def test_api_tts_speak_returns_wav(client: TestClient) -> None:
    """Authenticated speak returns WAV audio with a duration header."""
    signup(client)
    response = client.post("/api/v1/tts", json={"text": "Hello world."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert response.headers["x-duration-seconds"] == "0.25"


def test_api_tts_speak_rejects_too_long(client: TestClient) -> None:
    """Text over the configured limit is rejected with 400."""
    signup(client)
    response = client.post("/api/v1/tts", json={"text": "x" * 10001})
    assert response.status_code == 400
    assert "character limit" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Audio streaming
# ---------------------------------------------------------------------------


def test_audio_stream_requires_auth(client: TestClient) -> None:
    """GET /tts/audio/{name} without a session returns 401."""
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav")
    assert response.status_code == 401


def test_audio_stream_serves_own_audio(client: TestClient) -> None:
    """A user can stream their own synthesized clip."""
    signup(client)
    client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello"},
    )
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")


def test_audio_stream_rejects_other_users(client: TestClient) -> None:
    """A user cannot stream another user's clip."""
    signup(client)
    client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello"},
    )
    signup(client, email=OTHER_EMAIL)
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav")
    assert response.status_code == 404


def test_audio_stream_rejects_invalid_names(client: TestClient) -> None:
    """Names that do not match the cache pattern are rejected."""
    signup(client)
    for name in ("../../etc/passwd", "1_bad.wav", "1_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz.wav"):
        response = client.get(f"/tts/audio/{name}")
        assert response.status_code == 404
