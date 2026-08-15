"""Tests for the AI quiz generation feature (multiple-choice questions)."""

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.database import Database
from app.services.email import EmailService
from app.services.gemini import GeminiService
from app.services.quiz import QuizService, normalize_questions

VIDEO_ID = "dQw4w9WgXcQ"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
EMAIL = "quiz@example.com"
PASSWORD = "test-password-123"

SEGMENTS = [
    {"start": 0.0, "end": 4.5, "text": "Hello everyone."},
    {"start": 4.5, "end": 8.0, "text": "This is a test."},
    {"start": 8.0, "end": 12.0, "text": "More content."},
]

QUIZ = {
    "questions": [
        {
            "question": "What is the first line?",
            "options": ["Hello everyone.", "Goodbye.", "This is a test.", "More content."],
            "correct_index": 0,
            "explanation": "The transcript begins with 'Hello everyone.'",
        },
        {
            "question": "What happens at 0:04?",
            "options": ["Nothing.", "A test begins.", "The video ends.", "Music plays."],
            "correct_index": 1,
            "explanation": "At 0:04 the narrator says 'This is a test.'",
        },
    ]
}


# ---------------------------------------------------------------------------
# Question normalization
# ---------------------------------------------------------------------------


def test_normalize_questions_keeps_valid_entries() -> None:
    result = normalize_questions(QUIZ["questions"])
    assert result == QUIZ["questions"]


def test_normalize_questions_drops_invalid_entries() -> None:
    questions = [
        {
            "question": "Valid",
            "options": ["A", "B"],
            "correct_index": 1,
            "explanation": "ok",
        },
        {"question": "No options", "options": [], "correct_index": 0},
        {"question": "Missing answer", "options": ["A", "B"], "correct_index": 5},
        {"question": "", "options": ["A", "B"], "correct_index": 0},
        {"question": "Bad index", "options": ["A", "B"], "correct_index": "x"},
        "not a dict",
        {"question": "Blank option", "options": ["A", ""], "correct_index": 0},
    ]
    result = normalize_questions(questions)
    assert [q["question"] for q in result] == ["Valid"]


def test_normalize_questions_trims_to_four_options() -> None:
    result = normalize_questions(
        [
            {
                "question": "Q",
                "options": ["A", "B", "C", "D", "E"],
                "correct_index": 3,
                "explanation": "",
            }
        ]
    )
    assert result[0]["options"] == ["A", "B", "C", "D"]
    assert result[0]["correct_index"] == 3


# ---------------------------------------------------------------------------
# Gemini prompt + parsing
# ---------------------------------------------------------------------------


def test_build_quiz_prompt_includes_title_and_transcript() -> None:
    prompt = GeminiService.build_quiz_prompt(
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
    assert "correct_index" in prompt
    assert "questions" in prompt.lower()


def test_build_quiz_prompt_truncates_long_transcripts() -> None:
    long_segments = [{"start": 0.0, "end": 1.0, "text": "word " * 500}]
    prompt = GeminiService.build_quiz_prompt(
        title="T",
        uploader=None,
        duration=1.0,
        segments=long_segments,
        summary_chars=100,
    )
    assert prompt.count("word") < 100


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


def test_generate_quiz_parses_json_and_filters_bad_questions() -> None:
    service = GeminiService(api_key="test", model="fake-model", max_tokens=1024)
    service._client = _FakeClient(
        '{"questions": [{"question": "Q1", "options": ["A", "B"], '
        '"correct_index": 1, "explanation": "e"}, '
        '{"question": "Q2", "options": ["A"], "correct_index": 0}]}'
    )
    result = asyncio.run(
        service.generate_quiz(title="T", uploader="U", duration=60.0, segments=SEGMENTS)
    )
    assert result == {
        "questions": [
            {
                "question": "Q1",
                "options": ["A", "B"],
                "correct_index": 1,
                "explanation": "e",
            }
        ]
    }


def test_generate_quiz_requires_api_key() -> None:
    service = GeminiService(api_key="", model="fake-model", max_tokens=1024)
    with pytest.raises(RuntimeError):
        asyncio.run(
            service.generate_quiz(title="T", uploader="U", duration=60.0, segments=SEGMENTS)
        )


# ---------------------------------------------------------------------------
# QuizService
# ---------------------------------------------------------------------------


class FakeQuizGemini:
    available = True
    model = "fake-model"

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or dict(QUIZ)
        self.error = error
        self.calls: list[dict] = []

    async def generate_quiz(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return dict(self.result)


class FakeUnavailableGemini:
    available = False
    model = "fake-model"


def _make_service() -> tuple[Database, FakeQuizGemini, QuizService]:
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    gemini = FakeQuizGemini()
    return db, gemini, QuizService(db, gemini)


def test_service_generates_and_persists_quiz() -> None:
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

    quiz = asyncio.run(service.generate(video_id, user_id))

    assert quiz["questions"] == QUIZ["questions"]
    assert quiz["model"] == "fake-model"
    assert gemini.calls and gemini.calls[0]["title"] == "Fake Video"
    assert db.get_quiz(video_id) is not None


def test_service_generate_unknown_video_raises() -> None:
    db, _, service = _make_service()
    user_id = db.create_user(email="a@b.com", password_hash="hash")
    with pytest.raises(ValueError):
        asyncio.run(service.generate(999, user_id))


def test_service_get_returns_none_without_quiz() -> None:
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
    assert QuizService(db, FakeQuizGemini()).available is True
    assert QuizService(db, FakeUnavailableGemini()).available is False


def test_service_rejects_empty_quiz() -> None:
    db, gemini, service = _make_service()
    gemini.result = {"questions": []}
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
    with pytest.raises(ValueError):
        asyncio.run(service.generate(video_id, user_id))


# ---------------------------------------------------------------------------
# Database round-trip
# ---------------------------------------------------------------------------


def test_save_and_get_quiz_round_trips_and_updates() -> None:
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

    db.save_quiz(video_id=video_id, questions=QUIZ["questions"], model="m1")
    quiz = db.get_quiz(video_id)
    assert quiz["questions"] == QUIZ["questions"]
    assert quiz["model"] == "m1"

    db.save_quiz(video_id=video_id, questions=[], model="m2")
    quiz = db.get_quiz(video_id)
    assert quiz["questions"] == []
    assert quiz["model"] == "m2"


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

    def transcribe_stream(
        self, audio_path: str, on_progress, chunk_seconds: float, overlap_seconds: float
    ) -> dict:
        on_progress("Transcribing…", 50)
        return self.transcribe(audio_path)


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
    app.state.quiz_service = QuizService(db, FakeQuizGemini())
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


def test_quiz_action_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/videos/1/quiz",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/login\")" in response.text


def test_quiz_action_unknown_video_navigates_home(client: TestClient) -> None:
    signup(client)
    response = client.post(
        "/videos/999/quiz",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/\")" in response.text


def test_quiz_action_generates_and_swaps_card(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)

    response = client.post(
        f"/videos/{video_id}/quiz",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "Writing your quiz" in body
    assert "Quiz" in body
    assert "What is the first line?" in body
    assert "data-quiz-option" in body
    assert "Regenerate" in body
    assert "history.pushState" not in body

    stored = app.state.database.get_quiz(video_id)
    assert stored is not None
    assert stored["questions"] == QUIZ["questions"]


def test_quiz_action_reports_generation_failure(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.quiz_service = QuizService(
        app.state.database, FakeQuizGemini(error=RuntimeError("boom"))
    )

    response = client.post(
        f"/videos/{video_id}/quiz",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "Quiz generation failed" in response.text
    assert app.state.database.get_quiz(video_id) is None


def test_quiz_action_unavailable_without_gemini(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.quiz_service = QuizService(app.state.database, FakeUnavailableGemini())

    response = client.post(
        f"/videos/{video_id}/quiz",
        headers={"Datastar-Request": "true"},
        json={},
    )
    assert response.status_code == 200
    assert "Quiz generation is unavailable" in response.text


def test_video_page_renders_stored_quiz(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.quiz_service = QuizService(app.state.database, FakeQuizGemini())

    client.post(
        f"/videos/{video_id}/quiz",
        headers={"Datastar-Request": "true"},
        json={},
    )

    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "What is the first line?" in page.text
    assert "data-quiz-option" in page.text
    assert "Regenerate" in page.text
    assert "Generate quiz" not in page.text


def test_video_page_shows_quiz_controls(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    app.state.quiz_service = QuizService(app.state.database, FakeQuizGemini())

    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "Quiz" in page.text
    assert "Generate quiz" in page.text
    assert "@post('/videos/%d/quiz')" % video_id in page.text


def test_video_page_renders_without_quiz_service(client: TestClient) -> None:
    signup(client)
    video_id = transcribe_video(client)
    del app.state.quiz_service

    page = client.get(f"/videos/{video_id}")
    assert page.status_code == 200
    assert "AI quizzes are unavailable" in page.text
