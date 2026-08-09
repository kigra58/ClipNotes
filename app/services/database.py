"""SQLite persistence layer for transcripts, segments and RAG chunks."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id             TEXT NOT NULL UNIQUE,
    youtube_url          TEXT NOT NULL,
    title                TEXT NOT NULL,
    uploader             TEXT,
    language             TEXT NOT NULL,
    language_probability REAL NOT NULL,
    duration             REAL NOT NULL,
    transcript           TEXT NOT NULL,
    created_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    start         REAL NOT NULL,
    end           REAL NOT NULL,
    text          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_transcript
    ON segments(transcript_id);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    start         REAL NOT NULL,
    end           REAL NOT NULL,
    text          TEXT NOT NULL,
    embedding     BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_transcript
    ON chunks(transcript_id);
"""


class Database:
    """Minimal SQLite wrapper used by the application services."""

    def __init__(self, path: Path) -> None:
        """Initialize the database at the given path.

        Args:
            path: Filesystem path of the SQLite database file.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create the database schema if it does not exist yet."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.info("Database ready at %s", self.path)

    def save_transcript(
        self,
        *,
        video_id: str,
        youtube_url: str,
        title: str,
        uploader: str | None,
        language: str,
        language_probability: float,
        duration: float,
        transcript: str,
        segments: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> int:
        """Insert (or replace) a transcript with its segments and RAG chunks.

        Args:
            video_id: Unique YouTube video ID.
            youtube_url: Canonical watch URL.
            title: Video title.
            uploader: Channel name, when available.
            language: Detected spoken language code.
            language_probability: Confidence of the detected language.
            duration: Video duration in seconds.
            transcript: Full joined transcript text.
            segments: List of ``{"start", "end", "text"}`` dicts.
            chunks: List of ``{"start", "end", "text", "embedding"}`` dicts,
                where ``embedding`` is a 1-D ``numpy.ndarray``.

        Returns:
            The primary key of the stored transcript row.
        """
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM transcripts WHERE video_id = ?", (video_id,)
            ).fetchone()
            if existing is not None:
                conn.execute("DELETE FROM transcripts WHERE id = ?", (existing["id"],))

            cursor = conn.execute(
                """
                INSERT INTO transcripts (
                    video_id, youtube_url, title, uploader, language,
                    language_probability, duration, transcript
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    youtube_url,
                    title,
                    uploader,
                    language,
                    language_probability,
                    duration,
                    transcript,
                ),
            )
            transcript_id = cursor.lastrowid

            conn.executemany(
                "INSERT INTO segments (transcript_id, start, end, text) VALUES (?, ?, ?, ?)",
                [(transcript_id, s["start"], s["end"], s["text"]) for s in segments],
            )

            conn.executemany(
                "INSERT INTO chunks (transcript_id, start, end, text, embedding) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        transcript_id,
                        c["start"],
                        c["end"],
                        c["text"],
                        c["embedding"].astype(np.float32).tobytes(),
                    )
                    for c in chunks
                ],
            )

        logger.info("Saved transcript %d for video %s", transcript_id, video_id)
        return transcript_id

    def get_transcript(self, transcript_id: int) -> dict[str, Any] | None:
        """Fetch a transcript together with its segments.

        Args:
            transcript_id: Primary key of the transcript.

        Returns:
            A dict with transcript metadata plus ``segments``, or ``None``.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM transcripts WHERE id = ?", (transcript_id,)
            ).fetchone()
            if row is None:
                return None
            segments = [
                dict(segment)
                for segment in conn.execute(
                    "SELECT start, end, text FROM segments WHERE transcript_id = ? ORDER BY start",
                    (transcript_id,),
                ).fetchall()
            ]
        return {**dict(row), "segments": segments}

    def get_transcript_by_video_id(self, video_id: str) -> dict[str, Any] | None:
        """Fetch a transcript by its YouTube video ID.

        Args:
            video_id: Unique YouTube video ID.

        Returns:
            A dict with transcript metadata plus ``segments``, or ``None``.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM transcripts WHERE video_id = ?", (video_id,)
            ).fetchone()
        if row is None:
            return None
        return self.get_transcript(row["id"])

    def list_transcripts(self) -> list[dict[str, Any]]:
        """List all stored transcripts ordered by most recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, video_id, youtube_url, title, uploader, duration, created_at
                FROM transcripts
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chunks(self, transcript_id: int) -> list[dict[str, Any]]:
        """Fetch RAG chunks for a transcript with their embeddings.

        Args:
            transcript_id: Primary key of the transcript.

        Returns:
            A list of ``{"start", "end", "text", "embedding"}`` dicts where
            ``embedding`` is restored as a ``numpy.ndarray``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT start, end, text, embedding
                FROM chunks
                WHERE transcript_id = ? AND embedding IS NOT NULL
                ORDER BY id
                """,
                (transcript_id,),
            ).fetchall()
        return [
            {
                "start": row["start"],
                "end": row["end"],
                "text": row["text"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32),
            }
            for row in rows
        ]
