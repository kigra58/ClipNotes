"""SQLite persistence layer for users, videos, chats and messages.

Schema version 2: per-user video management with categories, per-video
conversations and persisted messages.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS categories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, name)
);

CREATE TABLE IF NOT EXISTS videos (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id              INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category_id          INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    youtube_id           TEXT NOT NULL,
    youtube_url          TEXT NOT NULL,
    title                TEXT NOT NULL,
    uploader             TEXT,
    language             TEXT NOT NULL,
    language_probability REAL NOT NULL,
    duration             REAL NOT NULL,
    transcript           TEXT NOT NULL,
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id, youtube_id)
);

CREATE INDEX IF NOT EXISTS idx_videos_user ON videos(user_id);
CREATE INDEX IF NOT EXISTS idx_videos_category ON videos(category_id);

CREATE TABLE IF NOT EXISTS segments (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start    REAL NOT NULL,
    end      REAL NOT NULL,
    text     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);

CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id  INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start     REAL NOT NULL,
    end       REAL NOT NULL,
    text      TEXT NOT NULL,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    video_id   INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    title      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_conversations_video ON conversations(video_id);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    sources_json    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
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
        """Create the database schema if it does not exist yet.

        A file that predates schema versioning (``user_version = 0``) may still
        contain tables from an older schema; those are dropped and rebuilt so
        that stale ``CREATE INDEX`` statements never run against a mismatched
        table layout.
        """
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == SCHEMA_VERSION:
                return
            if version != 0:
                raise RuntimeError(
                    f"Unsupported database schema version {version}; "
                    f"expected {SCHEMA_VERSION}. Delete the database file to reset."
                )
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
            ).fetchone() is not None:
                conn.executescript("DROP TABLE IF EXISTS transcripts;")
                conn.executescript("DROP TABLE IF EXISTS messages;")
                conn.executescript("DROP TABLE IF EXISTS conversations;")
                conn.executescript("DROP TABLE IF EXISTS chunks;")
                conn.executescript("DROP TABLE IF EXISTS segments;")
                conn.executescript("DROP TABLE IF EXISTS videos;")
                conn.executescript("DROP TABLE IF EXISTS categories;")
                conn.executescript("DROP TABLE IF EXISTS users;")
                logger.info(
                    "Reset stale database file %s (user_version 0 with existing tables)",
                    self.path,
                )
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        logger.info("Database ready at %s (schema v%d)", self.path, SCHEMA_VERSION)

    # ---------- Users ----------

    def create_user(self, *, email: str, password_hash: str) -> int:
        """Create a new user and return their id.

        Args:
            email: The user's email address (stored lowercased).
            password_hash: The PBKDF2 hash of the user's password.

        Returns:
            The new user's primary key.

        Raises:
            sqlite3.IntegrityError: If the email is already registered.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (email.strip().lower(), password_hash),
            )
            return int(cursor.lastrowid)

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        """Fetch a user by email address.

        Args:
            email: The user's email address.

        Returns:
            The user row, or ``None`` if no such user exists.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        """Fetch a user by primary key.

        Args:
            user_id: The user's primary key.

        Returns:
            The user row (without the password hash), or ``None``.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, created_at FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    # ---------- Categories ----------

    def create_category(self, *, user_id: int, name: str) -> int:
        """Create a category owned by the user.

        Args:
            user_id: The owning user.
            name: The category name.

        Returns:
            The new category's primary key.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO categories (user_id, name) VALUES (?, ?)",
                (user_id, name.strip()),
            )
            return int(cursor.lastrowid)

    def list_categories(self, user_id: int) -> list[dict[str, Any]]:
        """List the user's categories ordered by name.

        Args:
            user_id: The owning user.

        Returns:
            A list of category rows with a ``video_count`` field.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.name,
                       (SELECT COUNT(*) FROM videos v WHERE v.category_id = c.id)
                           AS video_count
                FROM categories c
                WHERE c.user_id = ?
                ORDER BY c.name COLLATE NOCASE
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_category(self, *, category_id: int, user_id: int) -> bool:
        """Delete a category, leaving its videos uncategorized.

        Args:
            category_id: The category to delete.
            user_id: The owning user.

        Returns:
            ``True`` if a category was deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM categories WHERE id = ? AND user_id = ?",
                (category_id, user_id),
            )
            return cursor.rowcount > 0

    # ---------- Videos ----------

    def save_video(
        self,
        *,
        user_id: int,
        youtube_id: str,
        youtube_url: str,
        title: str,
        uploader: str | None,
        language: str,
        language_probability: float,
        duration: float,
        transcript: str,
        segments: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        category_id: int | None = None,
    ) -> int:
        """Insert (or replace) a video with its segments and RAG chunks.

        Args:
            user_id: The owning user.
            youtube_id: Unique YouTube video ID.
            youtube_url: Canonical watch URL.
            title: Video title.
            uploader: Channel name, when available.
            language: Detected spoken language code.
            language_probability: Confidence of the detected language.
            duration: Video duration in seconds.
            transcript: Full joined transcript text.
            segments: List of ``{"start", "end", "text"}`` dicts.
            chunks: List of ``{"start", "end", "text", "embedding"}`` dicts.
            category_id: Optional category to assign.

        Returns:
            The primary key of the stored video row.
        """
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM videos WHERE user_id = ? AND youtube_id = ?",
                (user_id, youtube_id),
            ).fetchone()
            if existing is not None:
                conn.execute("DELETE FROM videos WHERE id = ?", (existing["id"],))

            cursor = conn.execute(
                """
                INSERT INTO videos (
                    user_id, category_id, youtube_id, youtube_url, title, uploader,
                    language, language_probability, duration, transcript
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    category_id,
                    youtube_id,
                    youtube_url,
                    title,
                    uploader,
                    language,
                    language_probability,
                    duration,
                    transcript,
                ),
            )
            video_id = int(cursor.lastrowid)

            conn.executemany(
                "INSERT INTO segments (video_id, start, end, text) VALUES (?, ?, ?, ?)",
                [(video_id, s["start"], s["end"], s["text"]) for s in segments],
            )

            conn.executemany(
                "INSERT INTO chunks (video_id, start, end, text, embedding) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        video_id,
                        c["start"],
                        c["end"],
                        c["text"],
                        c["embedding"].astype(np.float32).tobytes(),
                    )
                    for c in chunks
                ],
            )

        logger.info("Saved video %d for user %d (video %s)", video_id, user_id, youtube_id)
        return video_id

    def get_video(self, video_id: int, user_id: int) -> dict[str, Any] | None:
        """Fetch a user's video together with its segments.

        Args:
            video_id: Primary key of the video.
            user_id: The owning user.

        Returns:
            A dict with video metadata plus ``segments`` and ``category``,
            or ``None`` if the video does not belong to the user.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE id = ? AND user_id = ?",
                (video_id, user_id),
            ).fetchone()
            if row is None:
                return None
            video = dict(row)
            video["segments"] = [
                dict(segment)
                for segment in conn.execute(
                    "SELECT start, end, text FROM segments WHERE video_id = ? ORDER BY start",
                    (video_id,),
                ).fetchall()
            ]
            category = conn.execute(
                "SELECT id, name FROM categories WHERE id = ?", (video["category_id"],)
            ).fetchone()
            video["category"] = dict(category) if category else None
        return video

    def list_videos(self, user_id: int, *, category_id: int | None = None) -> list[dict[str, Any]]:
        """List the user's videos, most recent first.

        Args:
            user_id: The owning user.
            category_id: Optional category filter.

        Returns:
            A list of video rows.
        """
        query = (
            "SELECT v.*, c.name AS category_name FROM videos v "
            "LEFT JOIN categories c ON c.id = v.category_id "
            "WHERE v.user_id = ?"
        )
        params: list[Any] = [user_id]
        if category_id is not None:
            query += " AND v.category_id = ?"
            params.append(category_id)
        query += " ORDER BY v.created_at DESC, v.id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def set_video_category(self, *, video_id: int, user_id: int, category_id: int | None) -> bool:
        """Assign or clear a video's category.

        Args:
            video_id: The video to update.
            user_id: The owning user.
            category_id: The category to assign, or ``None`` to clear.

        Returns:
            ``True`` if the video was updated.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE videos SET category_id = ? WHERE id = ? AND user_id = ?",
                (category_id, video_id, user_id),
            )
            return cursor.rowcount > 0

    def delete_video(self, *, video_id: int, user_id: int) -> bool:
        """Delete a user's video (cascades to segments, chunks and chats).

        Args:
            video_id: The video to delete.
            user_id: The owning user.

        Returns:
            ``True`` if a video was deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM videos WHERE id = ? AND user_id = ?", (video_id, user_id)
            )
            return cursor.rowcount > 0

    def get_chunks(self, video_id: int) -> list[dict[str, Any]]:
        """Fetch RAG chunks for a video with their embeddings.

        Args:
            video_id: Primary key of the video.

        Returns:
            A list of ``{"start", "end", "text", "embedding"}`` dicts.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT start, end, text, embedding
                FROM chunks
                WHERE video_id = ? AND embedding IS NOT NULL
                ORDER BY id
                """,
                (video_id,),
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

    # ---------- Conversations & messages ----------

    def create_conversation(self, *, user_id: int, video_id: int, title: str | None = None) -> int:
        """Create a conversation for a user on a video.

        Args:
            user_id: The owning user.
            video_id: The video being discussed.
            title: Optional conversation title.

        Returns:
            The new conversation's primary key.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (user_id, video_id, title) VALUES (?, ?, ?)",
                (user_id, video_id, title),
            )
            return int(cursor.lastrowid)

    def list_conversations(self, *, video_id: int, user_id: int) -> list[dict[str, Any]]:
        """List a user's conversations for a video, most recently active first.

        Args:
            video_id: The video being discussed.
            user_id: The owning user.

        Returns:
            A list of conversation rows.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE video_id = ? AND user_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (video_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: int, user_id: int) -> dict[str, Any] | None:
        """Fetch a conversation by id if it belongs to the user.

        Args:
            conversation_id: The conversation's primary key.
            user_id: The owning user.

        Returns:
            The conversation row, or ``None``.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def touch_conversation(self, conversation_id: int) -> None:
        """Update a conversation's ``updated_at`` timestamp."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )

    def delete_conversation(self, *, conversation_id: int, user_id: int) -> bool:
        """Delete a user's conversation.

        Args:
            conversation_id: The conversation to delete.
            user_id: The owning user.

        Returns:
            ``True`` if a conversation was deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            return cursor.rowcount > 0

    def save_message(
        self,
        *,
        conversation_id: int,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> int:
        """Append a message to a conversation.

        Args:
            conversation_id: The conversation to append to.
            role: ``"user"`` or ``"assistant"``.
            content: The message text.
            sources: Optional retrieved source chunks.

        Returns:
            The new message's primary key.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, sources_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(sources) if sources else None,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )
            return int(cursor.lastrowid)

    def list_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        """List all messages in a conversation, oldest first.

        Args:
            conversation_id: The conversation's primary key.

        Returns:
            A list of message rows with a parsed ``sources`` field.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, sources_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id
                """,
                (conversation_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw = item.pop("sources_json")
            try:
                item["sources"] = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                item["sources"] = []
            result.append(item)
        return result
