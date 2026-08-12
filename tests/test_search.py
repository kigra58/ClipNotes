"""Tests for full-text search across the library (SQLite FTS5)."""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.database import Database, _fts_query
from app.services.email import EmailService

VIDEO_ID = "dQw4w9WgXcQ"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
EMAIL = "search@example.com"
OTHER_EMAIL = "other-search@example.com"
PASSWORD = "test-password-123"

SEGMENTS = [
    {"start": 0.0, "end": 4.5, "text": "Welcome to the quantum computing course."},
    {"start": 4.5, "end": 8.0, "text": "Qubits are the building blocks of quantum circuits."},
    {"start": 8.0, "end": 12.0, "text": "Entanglement links two distant qubits."},
]


def _save_video(
    db: Database,
    user_id: int,
    *,
    title: str,
    segments: list | None = None,
    youtube_id: str = VIDEO_ID,
) -> int:
    return db.save_video(
        user_id=user_id,
        youtube_id=youtube_id,
        youtube_url=WATCH_URL,
        title=title,
        uploader="Fake Channel",
        language="en",
        language_probability=0.98,
        duration=12.0,
        transcript=" ".join(s["text"] for s in segments or SEGMENTS),
        segments=segments or SEGMENTS,
        chunks=[],
    )


# ---------------------------------------------------------------------------
# FTS5 query builder
# ---------------------------------------------------------------------------


def test_fts_query_quotes_terms_and_escapes_quotes() -> None:
    assert _fts_query("") == ""
    assert _fts_query("   ") == ""
    assert _fts_query("hello") == '"hello"'
    assert _fts_query("hello world") == '"hello" AND "world"'
    assert _fts_query('say "hi"') == '"say" AND """hi"""'


def test_fts_query_tolerates_operators_and_punctuation() -> None:
    # Operators like OR/AND and punctuation must be inert, not break MATCH.
    query = _fts_query("OR AND NOT ( ) *")
    assert query  # non-empty and safe to pass to MATCH


def test_fts_query_works_end_to_end() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    _save_video(db, user_id, title="A Video")
    results = db.search_library(user_id=user_id, query='welcome "quantum" course')
    assert results["segments"], "query with embedded quotes must match"


# ---------------------------------------------------------------------------
# Database search behavior
# ---------------------------------------------------------------------------


def test_search_finds_title_match() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    video_id = _save_video(db, user_id, title="Quantum Computing Explained")
    other_id = _save_video(
        db, user_id, title="Cooking Basics", youtube_id="aaaaaaaaaaa"
    )

    results = db.search_library(user_id=user_id, query="explained")
    assert [v["video_id"] for v in results["videos"]] == [video_id]
    assert other_id not in [v["video_id"] for v in results["videos"]]
    assert results["segments"] == []


def test_search_finds_segment_match_with_timestamps() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    video_id = _save_video(db, user_id, title="Quantum Computing")

    results = db.search_library(user_id=user_id, query="entanglement")
    assert len(results["segments"]) == 1
    segment = results["segments"][0]
    assert segment["video_id"] == video_id
    assert segment["start"] == 8.0
    assert segment["end"] == 12.0
    assert "Entanglement" in segment["text"]


def test_search_scopes_results_to_owner() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    alice = db.create_user(email="a@b.com", password_hash="hash")
    bob = db.create_user(email="b@b.com", password_hash="hash")
    _save_video(db, alice, title="Private Quantum Notes")
    _save_video(db, bob, title="Quantum For Everyone")

    alice_results = db.search_library(user_id=alice, query="quantum")
    bob_results = db.search_library(user_id=bob, query="quantum")

    assert [v["title"] for v in alice_results["videos"]] == ["Private Quantum Notes"]
    assert [v["title"] for v in bob_results["videos"]] == ["Quantum For Everyone"]


def test_search_multi_term_requires_all_terms() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    _save_video(db, user_id, title="Quantum Computing")

    assert db.search_library(user_id=user_id, query="quantum circuits")["segments"]
    assert db.search_library(user_id=user_id, query="quantum skateboards") == {
        "videos": [],
        "segments": [],
    }


def test_search_empty_and_blank_query() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    _save_video(db, user_id, title="Quantum")
    assert db.search_library(user_id=user_id, query="") == {"videos": [], "segments": []}
    assert db.search_library(user_id=user_id, query="   ") == {
        "videos": [],
        "segments": [],
    }


def test_search_after_delete_returns_nothing() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    video_id = _save_video(db, user_id, title="Quantum")
    assert db.search_library(user_id=user_id, query="quantum")["videos"]

    db.delete_video(video_id=video_id, user_id=user_id)
    assert db.search_library(user_id=user_id, query="quantum") == {
        "videos": [],
        "segments": [],
    }


def test_search_updates_when_video_is_retranscribed() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    video_id = _save_video(db, user_id, title="Quantum")

    _save_video(
        db,
        user_id,
        title="Quantum Reloaded",
        segments=[
            {"start": 0.0, "end": 3.0, "text": "Now we talk about rocket science."}
        ],
    )

    results = db.search_library(user_id=user_id, query="rocket")
    assert results["segments"][0]["video_id"] == video_id
    assert results["segments"][0]["text"] == "Now we talk about rocket science."
    assert db.search_library(user_id=user_id, query="entanglement") == {
        "videos": [],
        "segments": [],
    }


def test_search_does_not_include_processing_or_error_videos() -> None:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    ready_id = _save_video(db, user_id, title="Ready Quantum")
    processing_id = db.create_pending_video(
        user_id=user_id,
        youtube_id="bbbbbbbbbbb",
        youtube_url=WATCH_URL,
        title="Processing Quantum",
    )
    db.mark_video_status(video_id=processing_id, status="error")

    results = db.search_library(user_id=user_id, query="quantum")
    assert [v["video_id"] for v in results["videos"]] == [ready_id]


# ---------------------------------------------------------------------------
# Migration: v5 database gains FTS tables with a backfill
# ---------------------------------------------------------------------------


def test_schema_v5_migrates_to_v6_and_backfills_search_index(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-v5.db"
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
        CREATE TABLE segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            start REAL NOT NULL,
            end REAL NOT NULL,
            text TEXT NOT NULL
        );
        INSERT INTO users (id, email, password_hash) VALUES (1, 'legacy@example.com', 'hash');
        INSERT INTO videos (id, user_id, youtube_id, youtube_url, title, uploader, language, language_probability, duration, transcript)
            VALUES (1, 1, 'ccccccccccc', 'https://www.youtube.com/watch?v=ccccccccccc', 'Legacy Quantum Talk', 'Old Channel', 'en', 0.99, 10.0, 'Legacy transcript about qubits.');
        INSERT INTO segments (video_id, start, end, text) VALUES (1, 0.0, 5.0, 'Legacy transcript about qubits.');
        """
    )
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    conn.close()

    db = Database(db_path)
    assert db.search_library(user_id=1, query="qubits")["segments"]
    assert db.search_library(user_id=1, query="legacy")["videos"]


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
            "transcript": "Welcome to the quantum computing course.",
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


def test_search_page_requires_auth(client: TestClient) -> None:
    response = client.get("/search", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_search_page_requires_auth_datastar(client: TestClient) -> None:
    response = client.get("/search", headers={"Datastar-Request": "true"})
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/login\")" in response.text


def test_search_page_empty_state(client: TestClient) -> None:
    signup(client)
    page = client.get("/search")
    assert page.status_code == 200
    assert "Search your library" in page.text
    assert "Search your transcripts and jump straight to the moment" in page.text


def test_search_page_finds_title_and_segment(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)

    page = client.get("/search", params={"q": "quantum"})
    assert page.status_code == 200
    assert 'href="/videos/%d#t-0.0"' % video_id in page.text
    assert "quantum computing course" in page.text


def test_search_page_finds_title_match_only(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)

    page = client.get("/search", params={"q": "fake"})
    assert page.status_code == 200
    assert 'href="/videos/%d"' % video_id in page.text


def test_search_page_jump_links_carry_timestamp(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)

    page = client.get("/search", params={"q": "building blocks"})
    assert page.status_code == 200
    assert 'href="/videos/%d#t-4.5"' % video_id in page.text
    assert "0:04" in page.text


def test_search_results_fragment_patches_container(client: TestClient) -> None:
    signup(client)
    transcribe_video(client)

    response = client.get(
        "/fragments/search-results",
        params={"datastar": '{"q": "entanglement"}'},
        headers={"Datastar-Request": "true"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: selector #search-results" in response.text
    assert "Entanglement links two distant qubits." in response.text


def test_search_results_fragment_no_matches(client: TestClient) -> None:
    signup(client)
    transcribe_video(client)

    response = client.get("/fragments/search-results", params={"q": "zebra"})
    assert response.status_code == 200
    assert "No matches" in response.text


def test_search_results_fragment_requires_auth(client: TestClient) -> None:
    response = client.get("/fragments/search-results", params={"q": "x"})
    assert response.status_code == 401


def test_search_results_scoped_to_user(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)

    other = TestClient(app)
    signup(other, email=OTHER_EMAIL)
    page = other.get("/search", params={"q": "quantum"})
    assert page.status_code == 200
    assert 'href="/videos/%d"' % video_id not in page.text
    assert "No matches" in page.text


def test_header_omits_search_form_when_logged_in(client: TestClient) -> None:
    signup(client)
    page = client.get("/")
    assert page.status_code == 200
    assert 'class="header-search"' not in page.text
