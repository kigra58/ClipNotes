"""Tests for the AI summary feature (TL;DR, takeaways, chapters)."""

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.database import Database
from app.services.email import EmailService
from app.services.gemini import GeminiService
from app.services.social_post import SocialPostService
from app.services.summary import SummaryService, normalize_chapters

VIDEO_ID = "dQw4w9WgXcQ"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
EMAIL = "summary@example.com"
OTHER_EMAIL = "other-summary@example.com"
PASSWORD = "test-password-123"

SEGMENTS = [
    {"start": 0.0, "end": 4.5, "text": "Hello everyone."},
    {"start": 4.5, "end": 8.0, "text": "This is a test."},
    {"start": 8.0, "end": 12.0, "text": "More content."},
]

NOTES = {
    "tldr": "A concise overview of the video.",
    "takeaways": ["First takeaway.", "Second takeaway."],
    "chapters": [
        {"title": "Intro", "start": 0.0},
        {"title": "Main part", "start": 4.5},
    ],
}


# ---------------------------------------------------------------------------
# Chapter normalization
# ---------------------------------------------------------------------------


def test_normalize_chapters_snaps_sorts_and_derives_ends() -> None:
    chapters = [
        {"title": "Intro", "start": 0.2},
        {"title": "End", "start": 7.9},
        {"title": "Body", "start": 4.6},
    ]
    result = normalize_chapters(chapters, SEGMENTS, duration=12.0)
    assert result == [
        {"title": "Intro", "start": 0.0, "end": 4.5},
        {"title": "Body", "start": 4.5, "end": 8.0},
        {"title": "End", "start": 8.0, "end": 12.0},
    ]


def test_normalize_chapters_drops_duplicate_boundaries() -> None:
    chapters = [
        {"title": "Intro", "start": 0.1},
        {"title": "Intro again", "start": 0.0},
        {"title": "Body", "start": 4.5},
    ]
    result = normalize_chapters(chapters, SEGMENTS, duration=12.0)
    assert [c["title"] for c in result] == ["Intro", "Body"]


def test_normalize_chapters_drops_invalid_entries() -> None:
    chapters = [
        {"title": "Intro", "start": 0.0},
        {"title": "", "start": 4.5},
        {"title": "No number", "start": "abc"},
        {"title": "Negative", "start": -5.0},
        "not a dict",
        {"title": "Missing start"},
    ]
    result = normalize_chapters(chapters, SEGMENTS, duration=12.0)
    assert [c["title"] for c in result] == ["Intro"]


def test_normalize_chapters_without_segments() -> None:
    assert normalize_chapters([{"title": "Intro", "start": 0.0}], [], duration=10.0) == []


def test_normalize_chapters_uses_last_segment_end() -> None:
    chapters = [{"title": "Intro", "start": 0.0}, {"title": "Body", "start": 4.5}]
    result = normalize_chapters(chapters, SEGMENTS, duration=999.0)
    assert result[-1]["end"] == 12.0


# ---------------------------------------------------------------------------
# Gemini prompt + notes parsing
# ---------------------------------------------------------------------------


def test_build_video_notes_prompt_includes_timestamped_transcript() -> None:
    prompt = GeminiService.build_video_notes_prompt(
        title="My Video",
        uploader="My Channel",
        duration=60.0,
        segments=SEGMENTS,
        summary_chars=10000,
    )
    assert "Video title: My Video" in prompt
    assert "Uploader: My Channel" in prompt
    assert "Duration (seconds): 60.0" in prompt
    assert "[0:00] Hello everyone." in prompt
    assert "[0:04] This is a test." in prompt
    assert "[0:08] More content." in prompt
    assert "chapters" in prompt.lower()


def test_build_video_notes_prompt_truncates_long_transcripts() -> None:
    long_segments = [{"start": 0.0, "end": 1.0, "text": "word " * 500}]
    prompt = GeminiService.build_video_notes_prompt(
        title="T",
        uploader=None,
        duration=1.0,
        segments=long_segments,
        summary_chars=100,
    )
    assert len(prompt) < 1000


class _FakeModel:
    def __init__(self, text: str) -> None:
        self.text = text

    async def generate_content(self, model: str, contents: list, config: object) -> _FakeModel:
        return self


class _FakeAio:
    def __init__(self, text: str) -> None:
        self.models = _FakeModel(text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.aio = _FakeAio(text)


def test_generate_video_notes_parses_json_and_filters_bad_chapters() -> None:
    service = GeminiService(api_key="test", model="fake-model", max_tokens=1024)
    service._client = _FakeClient(
        '{"tldr": "tl", "takeaways": ["a", "b"], '
        '"chapters": [{"title": "C", "start": 1.2}, {"title": "Bad", "start": "x"}]}'
    )
    notes = asyncio.run(
        service.generate_video_notes(
            title="T", uploader="U", duration=60.0, segments=SEGMENTS
        )
    )
    assert notes == {
        "tldr": "tl",
        "takeaways": ["a", "b"],
        "chapters": [{"title": "C", "start": 1.2}],
    }


def test_generate_video_notes_requires_api_key() -> None:
    service = GeminiService(api_key="", model="fake-model", max_tokens=1024)
    with pytest.raises(RuntimeError):
        asyncio.run(
            service.generate_video_notes(
                title="T", uploader="U", duration=60.0, segments=SEGMENTS
            )
        )


# ---------------------------------------------------------------------------
# SummaryService
# ---------------------------------------------------------------------------


class FakeGemini:
    available = True
    model = "fake-model"

    def __init__(self, notes=None, error: Exception | None = None) -> None:
        self.notes = notes or dict(NOTES)
        self.error = error
        self.calls: list[dict] = []

    async def generate_video_notes(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return dict(self.notes)


class FakeUnavailableGemini:
    available = False
    model = "fake-model"


def _make_service() -> tuple[Database, FakeGemini, SummaryService]:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    gemini = FakeGemini()
    return db, gemini, SummaryService(db, gemini)


def test_service_generates_and_persists_summary() -> None:
    db, gemini, service = _make_service()
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    video_id = db.save_video(
        user_id=user_id,
        youtube_id=VIDEO_ID,
        youtube_url=WATCH_URL,
        title="Fake Video",
        uploader="Fake Channel",
        language="en",
        language_probability=0.98,
        duration=12.0,
        transcript="Hello everyone. This is a test. More content.",
        segments=SEGMENTS,
        chunks=[],
    )

    summary = asyncio.run(service.generate(video_id, user_id))

    assert summary["tldr"] == NOTES["tldr"]
    assert summary["takeaways"] == NOTES["takeaways"]
    assert summary["chapters"] == [
        {"title": "Intro", "start": 0.0, "end": 4.5},
        {"title": "Main part", "start": 4.5, "end": 12.0},
    ]
    assert summary["model"] == "fake-model"
    assert gemini.calls and gemini.calls[0]["title"] == "Fake Video"
    assert db.get_summary(video_id) is not None


def test_service_generate_unknown_video_raises() -> None:
    db, _, service = _make_service()
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    with pytest.raises(ValueError):
        asyncio.run(service.generate(999, user_id))


def test_service_get_returns_none_without_summary() -> None:
    db, _, service = _make_service()
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    video_id = db.save_video(
        user_id=user_id,
        youtube_id=VIDEO_ID,
        youtube_url=WATCH_URL,
        title="T",
        uploader=None,
        language="en",
        language_probability=0.98,
        duration=12.0,
        transcript="x",
        segments=SEGMENTS,
        chunks=[],
    )
    assert service.get(video_id, user_id) is None


def test_service_available_follows_gemini() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    assert SummaryService(db, FakeGemini()).available is True
    assert SummaryService(db, FakeUnavailableGemini()).available is False


# ---------------------------------------------------------------------------
# Database round-trip
# ---------------------------------------------------------------------------


def test_save_and_get_summary_round_trips_and_updates() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    video_id = db.save_video(
        user_id=user_id,
        youtube_id=VIDEO_ID,
        youtube_url=WATCH_URL,
        title="T",
        uploader=None,
        language="en",
        language_probability=0.98,
        duration=12.0,
        transcript="x",
        segments=SEGMENTS,
        chunks=[],
    )

    db.save_summary(
        video_id=video_id,
        tldr="first",
        takeaways=["a"],
        chapters=[{"title": "C", "start": 0.0, "end": 4.5}],
        model="m1",
    )
    summary = db.get_summary(video_id)
    assert summary["tldr"] == "first"
    assert summary["takeaways"] == ["a"]
    assert summary["model"] == "m1"

    db.save_summary(
        video_id=video_id,
        tldr="second",
        takeaways=["a", "b"],
        chapters=[],
        model="m2",
    )
    summary = db.get_summary(video_id)
    assert summary["tldr"] == "second"
    assert summary["takeaways"] == ["a", "b"]
    assert summary["chapters"] == []
    assert summary["model"] == "m2"


def test_schema_v4_migrates_to_v5(tmp_path: Path) -> None:
    """An existing v4 database gains the summaries table on startup."""
    import sqlite3

    from app.services.database import SCHEMA_VERSION

    db_path = tmp_path / "legacy-v4.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            category_id INTEGER,
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
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()

    Database(db_path)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        summaries = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'summaries'"
        ).fetchone()
        assert summaries is not None


# ---------------------------------------------------------------------------
# API flow
# ---------------------------------------------------------------------------


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
    available = True


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
    app.state.summary_service = SummaryService(db, FakeGemini())
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


def test_summary_action_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/videos/1/summary",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/login\")" in response.text


def test_summary_action_unknown_video_navigates_home(client: TestClient) -> None:
    signup(client)
    response = client.post(
        "/videos/999/summary",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/\")" in response.text


def test_summary_action_generates_and_swaps_card(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)

    response = client.post(
        f"/videos/{video_id}/summary",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "AI summary" in body
    assert "A concise overview of the video." in body
    assert "First takeaway." in body
    assert "Chapters" in body
    assert "Intro" in body
    assert "history.pushState" not in body

    stored = app.state.database.get_summary(video_id)
    assert stored is not None
    assert stored["tldr"] == "A concise overview of the video."


def test_summary_action_regenerates_existing(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    client.post(
        f"/videos/{video_id}/summary",
        headers={"Datastar-Request": "true"},
        json={},
    )

    response = client.post(
        f"/videos/{video_id}/summary",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "Regenerate" in response.text
    assert "A concise overview of the video." in response.text


def test_summary_action_reports_generation_failure(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.summary_service = SummaryService(
        app.state.database, FakeGemini(error=RuntimeError("boom"))
    )

    response = client.post(
        f"/videos/{video_id}/summary",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "Summary generation failed" in response.text


def test_summary_action_unavailable_without_gemini(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.summary_service = SummaryService(app.state.database, FakeUnavailableGemini())

    response = client.post(
        f"/videos/{video_id}/summary",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "AI summaries are unavailable" in response.text


def test_video_page_shows_summary_when_present(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.database.save_summary(
        video_id=video_id,
        tldr="A stored TL;DR.",
        takeaways=["A takeaway."],
        chapters=[{"title": "Chapter one", "start": 0.0, "end": 4.5}],
        model="fake-model",
    )

    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "A stored TL;DR." in page.text
    assert "A takeaway." in page.text
    assert "Chapter one" in page.text
    assert 'href="#t-0.0"' in page.text


def test_video_page_shows_generate_button_without_summary(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)

    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "Generate summary" in page.text
    assert "@post('/videos/%d/summary')" % video_id in page.text


def test_video_page_renders_without_summary_service(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    del app.state.summary_service

    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "AI summaries are unavailable" in page.text


# ---------------------------------------------------------------------------
# Social post feature
# ---------------------------------------------------------------------------


def test_build_social_post_prompt_includes_title_and_transcript() -> None:
    prompt = GeminiService.build_social_post_prompt(
        title="My Video",
        uploader="My Channel",
        duration=60.0,
        segments=SEGMENTS,
        summary_chars=10000,
    )
    assert "Video title: My Video" in prompt
    assert "Uploader: My Channel" in prompt
    assert "[0:00] Hello everyone." in prompt
    assert "hashtags" in prompt.lower()


def test_generate_social_post_parses_json() -> None:
    service = GeminiService(api_key="test", model="fake-model", max_tokens=1024)
    service._client = _FakeClient(
        '{"post": "An engaging post.", "hashtags": ["#tech", "#learn"]}'
    )
    result = asyncio.run(
        service.generate_social_post(
            title="T", uploader="U", duration=60.0, segments=SEGMENTS
        )
    )
    assert result == {
        "post": "An engaging post.\n\n#tech\n#learn",
        "hashtags": ["#tech", "#learn"],
    }


def test_generate_social_post_filters_blank_hashtags() -> None:
    service = GeminiService(api_key="test", model="fake-model", max_tokens=1024)
    service._client = _FakeClient(
        '{"post": "Post text.", "hashtags": ["#a", "", "   ", "#b"]}'
    )
    result = asyncio.run(
        service.generate_social_post(
            title="T", uploader=None, duration=60.0, segments=SEGMENTS
        )
    )
    assert result == {"post": "Post text.\n\n#a\n#b", "hashtags": ["#a", "#b"]}


def test_generate_social_post_keeps_hashtags_in_plain_text() -> None:
    service = GeminiService(api_key="test", model="fake-model", max_tokens=1024)
    service._client = _FakeClient(
        "Here's a great post about this video!\n\n#tech #tutorial #learn"
    )
    result = asyncio.run(
        service.generate_social_post(
            title="T", uploader="U", duration=60.0, segments=SEGMENTS
        )
    )
    assert result == {
        "post": "Here's a great post about this video!\n\n#tech #tutorial #learn",
        "hashtags": ["#tech", "#tutorial", "#learn"],
    }


def test_generate_social_post_from_code_fence() -> None:
    service = GeminiService(api_key="test", model="fake-model", max_tokens=1024)
    service._client = _FakeClient(
        "```text\nCheck out this video!\n\n#one #two\n```"
    )
    result = asyncio.run(
        service.generate_social_post(
            title="T", uploader=None, duration=60.0, segments=SEGMENTS
        )
    )
    assert result == {
        "post": "Check out this video!\n\n#one #two",
        "hashtags": ["#one", "#two"],
    }


def test_generate_social_post_salvages_broken_json() -> None:
    service = GeminiService(api_key="test", model="fake-model", max_tokens=1024)
    service._client = _FakeClient(
        '{"post": "Great post about "quotes" inside.", "hashtags": ["#tech"]}'
    )
    result = asyncio.run(
        service.generate_social_post(
            title="T", uploader="U", duration=60.0, segments=SEGMENTS
        )
    )
    assert result["post"] == "Great post about \"quotes\" inside.\n\n#tech"
    assert result["hashtags"] == ["#tech"]


def test_generate_social_post_requires_api_key() -> None:
    service = GeminiService(api_key="", model="fake-model", max_tokens=1024)
    with pytest.raises(RuntimeError):
        asyncio.run(
            service.generate_social_post(
                title="T", uploader="U", duration=60.0, segments=SEGMENTS
            )
        )


class FakeSocialGemini:
    available = True
    model = "fake-model"

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or {
            "post": "Check out this video!\n\n#tech\n#tutorial",
            "hashtags": ["#tech", "#tutorial"],
        }
        self.error = error

    async def generate_social_post(self, **kwargs) -> dict:
        if self.error is not None:
            raise self.error
        return dict(self.result)


def test_social_post_action_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/videos/1/social-post",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/login\")" in response.text


def test_social_post_action_unknown_video_navigates_home(client: TestClient) -> None:
    signup(client)
    response = client.post(
        "/videos/999/social-post",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/\")" in response.text


def test_social_post_action_generates_and_swaps_editor(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.social_post_service = SocialPostService(
        app.state.database, FakeSocialGemini()
    )

    response = client.post(
        f"/videos/{video_id}/social-post",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "Writing your social post" in body
    assert "Check out this video!" in body
    assert "#tech" in body
    assert "#tutorial" in body
    assert "data-copy-social-post" in body
    assert "Regenerate" in body
    assert "history.pushState" not in body

    stored = app.state.database.get_social_post(video_id)
    assert stored is not None
    assert stored["post"] == "Check out this video!\n\n#tech\n#tutorial"
    assert stored["hashtags"] == ["#tech", "#tutorial"]


def test_social_post_action_reports_generation_failure(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.social_post_service = SocialPostService(
        app.state.database, FakeSocialGemini(error=RuntimeError("boom"))
    )

    response = client.post(
        f"/videos/{video_id}/social-post",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "Social post generation failed" in response.text
    assert app.state.database.get_social_post(video_id) is None


def test_social_post_action_unavailable_without_gemini(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.social_post_service = SocialPostService(
        app.state.database, FakeUnavailableGemini()
    )

    response = client.post(
        f"/videos/{video_id}/social-post",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "Social post generation is unavailable" in response.text


def test_video_page_renders_stored_social_post(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.social_post_service = SocialPostService(
        app.state.database, FakeSocialGemini()
    )

    client.post(
        f"/videos/{video_id}/social-post",
        headers={"Datastar-Request": "true"},
        json={},
    )

    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "Check out this video!" in page.text
    assert "Regenerate" in page.text
    assert "Generate social post" not in page.text


def test_video_page_shows_social_post_controls(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.social_post_service = SocialPostService(
        app.state.database, FakeSocialGemini()
    )

    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "Social post" in page.text
    assert "Generate social post" in page.text
    assert "@post('/videos/%d/social-post')" % video_id in page.text
