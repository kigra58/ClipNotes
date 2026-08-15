"""Tests for the YouTube voice-clone feature.

The real Coqui XTTS stack is heavy, so the Datastar flows are exercised with
fake services: a fake YouTube section download returning a small WAV, a fake
voice-clone service that reports ``available``, and a temp database.
"""

import io
import tempfile
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.database import Database
from app.services.email import EmailService
from app.services.pipeline import run_transcription
from app.services.voice_clone import pick_sample_window

VIDEO_ID = "dQw4w9WgXcQ"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
EMAIL = "clone@example.com"
PASSWORD = "test-password-123"


def _write_wav(path: str | io.BytesIO) -> None:
    """Write a minimal valid 16-bit mono WAV file."""
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 1000)


class FakeYouTubeService:
    """In-memory stand-in for YouTubeService with section downloads."""

    def __init__(self) -> None:
        self.section_path = None

    def get_metadata(self, youtube_url: str) -> dict:
        return {
            "video_id": VIDEO_ID,
            "title": "Fake Video",
            "duration": 60.0,
            "uploader": "Fake Channel",
        }

    def download_audio(self, youtube_url: str) -> dict:
        return {
            "video_id": VIDEO_ID,
            "title": "Fake Video",
            "duration": 60.0,
            "uploader": "Fake Channel",
            "audio_path": "temp/fake.mp3",
        }

    def download_audio_section(self, youtube_url: str, start: float, end: float) -> dict:
        assert self.section_path is not None
        return {
            "video_id": VIDEO_ID,
            "title": "Fake Video",
            "duration": 60.0,
            "uploader": "Fake Channel",
            "audio_path": str(self.section_path),
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
        self.voices_dir = Path(tempfile.mkdtemp())
        self.max_chars = 10000
        self.synthesis_timeout_seconds = 300
        self.available = True

    @property
    def voice(self) -> str:
        return "en_US-lessac-medium"

    def list_voices(self) -> list[dict]:
        return [
            {
                "name": "en_US-lessac-medium",
                "label": "English (US) · Lessac Medium (default)",
                "default": True,
                "lang": "en",
            }
        ]

    def set_voice(self, name: str) -> None:
        raise ValueError(f"Unknown voice '{name}'.")

    def cache_audio(self, text: str, user_id: int) -> Path:
        out_path = self.cache_dir / f"{user_id}_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav"
        _write_wav(str(out_path))
        return out_path

    def cleanup_stale(self) -> int:
        return 0

    def to_mp3(self, wav_path: Path) -> Path:
        return wav_path


class FakeVoiceCloneService:
    """In-memory stand-in for VoiceCloneService."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.synthesis_timeout_seconds = 300
        self.max_chars = 5000
        self.synthesized: list[str] = []
        self.registered_voices: list[str] = ["Now_click_save_Oops"]
        self.registered_synthesized: list[str] = []

    def cache_audio(
        self, text: str, profile: dict, user_id: int, cache_dir: Path | None = None
    ) -> Path:
        self.synthesized.append(text)
        out_path = Path(cache_dir or tempfile.mkdtemp()) / f"{user_id}_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.wav"
        _write_wav(str(out_path))
        return out_path

    def list_registered_voices(self) -> list[str]:
        return list(self.registered_voices) if self.available else []

    def cache_registered_audio(
        self, text: str, voice_name: str, user_id: int, cache_dir: Path | None = None
    ) -> Path:
        self.registered_synthesized.append(f"{voice_name}: {text}")
        out_path = Path(cache_dir or tempfile.mkdtemp()) / f"{user_id}_cccccccccccccccccccccccccccccccc.wav"
        _write_wav(str(out_path))
        return out_path


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """A test client wired with fake heavy services and a temp database."""
    db = Database(tmp_path / "test.db")
    fake_youtube = FakeYouTubeService()
    app.state.database = db
    app.state.youtube_service = fake_youtube
    app.state.transcription_service = FakeTranscriptionService()
    app.state.rag = FakeRAG()
    app.state.chat = FakeChat()
    app.state.email_service = EmailService(host="", port=465, username="", password="")
    app.state.pipeline = run_transcription
    app.state.tts_service = FakeTTSService()
    app.state.voice_clone_service = FakeVoiceCloneService()
    return TestClient(app)


def signup(client: TestClient) -> None:
    """Register a user; the client cookie jar now holds the JWT."""
    response = client.post(
        "/auth/signup",
        data={"email": EMAIL, "password": PASSWORD, "confirm": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 303


def _save_video(database: Database, user_id: int) -> int:
    """Insert a ready video row directly for voice-clone tests."""
    return database.save_video(
        user_id=user_id,
        youtube_id=VIDEO_ID,
        youtube_url=WATCH_URL,
        title="Fake Video",
        uploader="Fake Channel",
        language="en",
        language_probability=0.98,
        duration=60.0,
        transcript="Hello everyone. This is a longer transcript for testing.",
        segments=[
            {"start": 0.0, "end": 4.5, "text": "Hello everyone."},
            {"start": 4.5, "end": 10.0, "text": "This is a longer transcript for testing."},
        ],
        chunks=[],
    )


def _user_id(database: Database) -> int:
    return database.get_user_by_email(EMAIL)["id"]


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------


def test_pick_sample_window_extends_to_target() -> None:
    """A speech-dense window is chosen around the longest segment."""
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Short."},
        {"start": 2.0, "end": 20.0, "text": "A much longer run of speech here for cloning."},
        {"start": 20.0, "end": 22.0, "text": "Trailing."},
    ]
    start, end, text = pick_sample_window(segments, target_seconds=12.0)
    assert 0.0 <= start < end <= 22.0
    assert end - start <= 12.0
    assert "longer run of speech" in text


def test_pick_sample_window_empty() -> None:
    """No usable segments yields an empty window."""
    assert pick_sample_window([], target_seconds=12.0) == (0.0, 0.0, "")
    assert pick_sample_window([{"start": 0.0, "end": 0.2, "text": ""}]) == (0.0, 0.0, "")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def test_database_voice_profile_crud(client: TestClient) -> None:
    """Profiles can be created, fetched, listed, replaced and deleted."""
    signup(client)
    database = app.state.database
    user_id = _user_id(database)
    video_id = _save_video(database, user_id)

    profile_id = database.create_voice_profile(
        user_id=user_id,
        video_id=video_id,
        name="Fake Channel's voice",
        language="en",
        reference_wav="ref_abc.wav",
        reference_text="Hello.",
    )
    profile = database.get_voice_profile(profile_id=profile_id, user_id=user_id)
    assert profile is not None
    assert profile["name"] == "Fake Channel's voice"
    assert profile["video_title"] == "Fake Video"

    assert (
        database.get_voice_profile_for_video(video_id=video_id, user_id=user_id)["id"]
        == profile_id
    )
    assert [p["id"] for p in database.list_voice_profiles(user_id)] == [profile_id]

    re_id = database.create_voice_profile(
        user_id=user_id,
        video_id=video_id,
        name="Renamed voice",
        language="hi",
        reference_wav="ref_def.wav",
        reference_text="",
    )
    assert re_id == profile_id
    assert (
        database.get_voice_profile(profile_id=profile_id, user_id=user_id)["name"]
        == "Renamed voice"
    )

    assert database.get_voice_profile(profile_id=profile_id, user_id=user_id + 999) is None
    assert database.delete_voice_profile(profile_id=profile_id, user_id=user_id)
    assert database.get_voice_profile(profile_id=profile_id, user_id=user_id) is None


# ---------------------------------------------------------------------------
# Web flow
# ---------------------------------------------------------------------------


def test_video_page_shows_clone_notice_without_service(client: TestClient) -> None:
    """The video page degrades to a notice when cloning is unavailable."""
    app.state.voice_clone_service = FakeVoiceCloneService(available=False)
    signup(client)
    database = app.state.database
    video_id = _save_video(database, _user_id(database))
    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "Voice clone" in page.text
    assert "disabled" in page.text


def test_datastar_clone_voice_creates_profile(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    """POST /videos/{id}/clone-voice downloads a sample and saves a profile."""
    monkeypatch.setattr(settings, "voice_clone_dir", tmp_path / "clones")
    signup(client)
    database = app.state.database
    video_id = _save_video(database, _user_id(database))

    source = tmp_path / "section.mp3"
    _write_wav(str(source))
    app.state.youtube_service.section_path = source

    response = client.post(
        f"/videos/{video_id}/clone-voice",
        headers={"Datastar-Request": "true"},
    )
    assert response.status_code == 200
    body = response.text
    assert 'id="voice-clone-panel"' in body
    assert "Fake Channel&#39;s voice" in body

    profile = database.get_voice_profile_for_video(video_id=video_id, user_id=_user_id(database))
    assert profile is not None
    assert profile["name"] == "Fake Channel's voice"
    assert profile["reference_text"]
    assert (tmp_path / "clones" / profile["reference_wav"]).is_file()


def test_datastar_clone_voice_unavailable(client: TestClient) -> None:
    """Cloning reports a friendly error when the XTTS stack is missing."""
    app.state.voice_clone_service = FakeVoiceCloneService(available=False)
    signup(client)
    database = app.state.database
    video_id = _save_video(database, _user_id(database))
    response = client.post(
        f"/videos/{video_id}/clone-voice",
        headers={"Datastar-Request": "true"},
    )
    assert response.status_code == 200
    assert "not available" in response.text
    assert database.get_voice_profile_for_video(video_id=video_id, user_id=_user_id(database)) is None


def test_speak_page_lists_cloned_voice_and_synthesizes(client: TestClient) -> None:
    """The Speak page shows the cloned voice and routes synthesis to it."""
    signup(client)
    database = app.state.database
    user_id = _user_id(database)
    video_id = _save_video(database, user_id)
    profile_id = database.create_voice_profile(
        user_id=user_id,
        video_id=video_id,
        name="Fake Channel's voice",
        language="en",
        reference_wav="ref_abc.wav",
        reference_text="Hello.",
    )

    page = client.get(f"/speak?video_id={video_id}&voice=clone_{profile_id}")
    assert page.status_code == 200
    assert f'value="clone_{profile_id}"' in page.text
    assert "data-cloned=\"true\"" in page.text
    assert "Fake Channel&#39;s voice" in page.text
    assert "Cloned voices" in page.text

    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={
            "tts_text": "Read this aloud in the cloned voice.",
            "voice": f"clone_{profile_id}",
        },
    )
    assert response.status_code == 200
    assert "tts_url" in response.text
    assert response.text.count("event: datastar-patch-signals") >= 2
    assert app.state.voice_clone_service.synthesized == ["Read this aloud in the cloned voice."]


def test_speak_page_lists_registered_voice_and_synthesizes(client: TestClient) -> None:
    """The Speak page shows registered voices and routes synthesis to them."""
    signup(client)
    database = app.state.database
    user_id = _user_id(database)
    video_id = _save_video(database, user_id)

    page = client.get(f"/speak?video_id={video_id}&voice=registered_Now_click_save_Oops")
    assert page.status_code == 200
    assert 'value="registered_Now_click_save_Oops"' in page.text
    assert "Registered voices" in page.text
    assert "Now_click_save_Oops" in page.text

    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={
            "tts_text": "Read this with the registered voice.",
            "voice": "registered_Now_click_save_Oops",
        },
    )
    assert response.status_code == 200
    assert "tts_url" in response.text
    assert response.text.count("event: datastar-patch-signals") >= 2
    assert app.state.voice_clone_service.registered_synthesized == [
        "Now_click_save_Oops: Read this with the registered voice."
    ]


def test_speak_page_registered_voice_unavailable(client: TestClient) -> None:
    """Registered voices are hidden and rejected when cloning is unavailable."""
    app.state.voice_clone_service = FakeVoiceCloneService(available=False)
    signup(client)
    database = app.state.database
    video_id = _save_video(database, _user_id(database))

    page = client.get(f"/speak?video_id={video_id}")
    assert page.status_code == 200
    assert "Registered voices" not in page.text

    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hi.", "voice": "registered_Now_click_save_Oops"},
    )
    assert response.status_code == 200
    assert "not available" in response.text
    assert app.state.voice_clone_service.registered_synthesized == []


def test_voice_clone_service_lists_registered_voices(tmp_path: Path, monkeypatch) -> None:
    """Registered voice names come from the model's voices directory."""
    import TTS.utils.manage as manage

    from app.services.voice_clone import VoiceCloneService

    voices_dir = (
        tmp_path
        / "tts"
        / "tts_models--multilingual--multi-dataset--xtts_v2"
        / "voices"
    )
    voices_dir.mkdir(parents=True)
    (voices_dir / "Demo_Voice.pth").write_bytes(b"x")
    (voices_dir / "Another.pth").write_bytes(b"x")
    (voices_dir / "ignore.txt").write_bytes(b"x")
    monkeypatch.setattr(
        manage, "get_user_data_dir", lambda appname: tmp_path / "tts"
    )

    service = VoiceCloneService(
        "tts_models/multilingual/multi-dataset/xtts_v2",
        cache_dir=tmp_path / "cache",
    )
    assert service.list_registered_voices() == ["Another", "Demo_Voice"]
