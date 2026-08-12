"""SQLite persistence layer for users, videos, chats and messages.

Schema version 6: per-user video management with categories, per-video
conversations, persisted messages, background transcription status, email
verification state on users, per-video AI summaries, and FTS5 full-text
search over video titles and timestamped transcript segments.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 6

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    email                         TEXT NOT NULL UNIQUE,
    password_hash                 TEXT NOT NULL,
    is_verified                   INTEGER NOT NULL DEFAULT 0,
    verification_token            TEXT,
    verification_token_expires_at TEXT,
    reset_token                   TEXT,
    reset_token_expires_at        TEXT,
    reset_token_attempts          INTEGER NOT NULL DEFAULT 0,
    created_at                    TEXT NOT NULL DEFAULT (datetime('now'))
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
    status               TEXT NOT NULL DEFAULT 'ready',
    error                TEXT,
    progress_percent     REAL NOT NULL DEFAULT 0,
    progress_message     TEXT,
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

CREATE TABLE IF NOT EXISTS summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   INTEGER NOT NULL UNIQUE REFERENCES videos(id) ON DELETE CASCADE,
    tldr       TEXT NOT NULL,
    takeaways  TEXT NOT NULL,
    chapters   TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS social_posts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   INTEGER NOT NULL UNIQUE REFERENCES videos(id) ON DELETE CASCADE,
    post       TEXT NOT NULL,
    hashtags   TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

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
    video_id   INTEGER REFERENCES videos(id) ON DELETE CASCADE,
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

-- FTS5 full-text search indexes. Content is stored in the FTS tables and kept
-- in sync manually by save_video()/delete_video(); user_id is UNINDEXED so a
-- user can only ever match against their own library.
CREATE VIRTUAL TABLE IF NOT EXISTS videos_fts USING fts5(
    user_id UNINDEXED,
    title,
    uploader
);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    video_id UNINDEXED,
    user_id UNINDEXED,
    start UNINDEXED,
    end UNINDEXED,
    text
);
"""


def _fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from free-text user input.

    Each whitespace-separated term is quoted as an FTS5 phrase (so punctuation
    and operators are inert) and AND-joined, so a multi-word query matches
    transcripts containing every term. Embedded double quotes are doubled,
    which is the FTS5 escape for a literal quote inside a phrase.
    """
    terms = [term for term in query.split() if term]
    return " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)


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
        table layout. Schema v2/v3 files are migrated in place to v4 by adding
        the ``videos.status``/``videos.error`` and ``users`` email-verification
        columns. Existing users are marked verified during migration so they
        can keep logging in. Schema v5/v6 files gain the ``summaries`` table
        and the FTS5 full-text search indexes. The migration is idempotent:
        any database that is missing a v4 column (e.g. a partial migration)
        has it added before startup completes.
        """
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {version} is newer than "
                    f"supported version {SCHEMA_VERSION}. Delete the database file to reset."
                )
            has_tables = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
                ).fetchone()
                is not None
            )
            if not has_tables:
                conn.executescript(_SCHEMA)
            elif version == 0:
                conn.executescript("DROP TABLE IF EXISTS transcripts;")
                conn.executescript("DROP TABLE IF EXISTS messages;")
                conn.executescript("DROP TABLE IF EXISTS conversations;")
                conn.executescript("DROP TABLE IF EXISTS chunks;")
                conn.executescript("DROP TABLE IF EXISTS segments;")
                conn.executescript("DROP TABLE IF EXISTS videos;")
                conn.executescript("DROP TABLE IF EXISTS categories;")
                conn.executescript("DROP TABLE IF EXISTS users;")
                conn.executescript(_SCHEMA)
                logger.info(
                    "Reset stale database file %s (user_version 0 with existing tables)",
                    self.path,
                )
            elif version in (2, 3, 4, 5):
                logger.info(
                    "Migrating %s from schema v%d to v%d", self.path, version, SCHEMA_VERSION
                )
            elif version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported database schema version {version}; "
                    f"expected {SCHEMA_VERSION}. Delete the database file to reset."
                )
            self._ensure_v3_video_columns(conn)
            self._ensure_progress_columns(conn)
            migrated_users = self._ensure_v4_user_columns(conn)
            if migrated_users and version in (2, 3):
                # Accounts created before email verification was introduced stay
                # verified so existing users are not locked out.
                conn.execute("UPDATE users SET is_verified = 1 WHERE is_verified = 0")
            self._ensure_summaries_table(conn)
            self._ensure_social_posts_table(conn)
            self._ensure_fts_tables(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        logger.info("Database ready at %s (schema v%d)", self.path, SCHEMA_VERSION)

    def _ensure_v3_video_columns(self, conn: sqlite3.Connection) -> None:
        """Add the v3 ``videos`` columns if they are missing (idempotent).

        ``ALTER TABLE ... ADD COLUMN`` cannot be wrapped in ``IF NOT EXISTS``,
        so the columns are probed via ``PRAGMA table_info`` first. This heals
        databases whose ``user_version`` is already 3 but whose table predates
        the status columns (e.g. an interrupted migration).
        """
        columns = {row[1] for row in conn.execute("PRAGMA table_info(videos)")}
        if "status" not in columns:
            conn.execute(
                "ALTER TABLE videos ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'"
            )
        if "error" not in columns:
            conn.execute("ALTER TABLE videos ADD COLUMN error TEXT")

    def _ensure_progress_columns(self, conn: sqlite3.Connection) -> None:
        """Add the transcription progress columns if they are missing (idempotent).

        Progress is written by the background transcription pipeline so the
        dashboard and video page can show a live progress bar and message.
        """
        columns = {row[1] for row in conn.execute("PRAGMA table_info(videos)")}
        if "progress_percent" not in columns:
            conn.execute(
                "ALTER TABLE videos ADD COLUMN progress_percent REAL NOT NULL DEFAULT 0"
            )
        if "progress_message" not in columns:
            conn.execute("ALTER TABLE videos ADD COLUMN progress_message TEXT")

    def _ensure_v4_user_columns(self, conn: sqlite3.Connection) -> bool:
        """Add the v4 email-verification/reset ``users`` columns (idempotent).

        Returns:
            ``True`` when the ``is_verified`` column was newly added (i.e. the
            database was migrated from a schema older than v4).
        """
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        added = False
        if "is_verified" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN is_verified INTEGER NOT NULL DEFAULT 0"
            )
            added = True
        if "verification_token" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN verification_token TEXT")
        if "verification_token_expires_at" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN verification_token_expires_at TEXT")
        if "reset_token" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
        if "reset_token_expires_at" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN reset_token_expires_at TEXT")
        if "reset_token_attempts" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN reset_token_attempts INTEGER NOT NULL DEFAULT 0"
            )
        return added

    def _ensure_summaries_table(self, conn: sqlite3.Connection) -> None:
        """Create the v5 ``summaries`` table if it is missing (idempotent).

        ``CREATE TABLE IF NOT EXISTS`` keeps databases created before v5 (or a
        fresh schema) from tripping over an already-present table.
        """
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id   INTEGER NOT NULL UNIQUE REFERENCES videos(id) ON DELETE CASCADE,
                tldr       TEXT NOT NULL,
                takeaways  TEXT NOT NULL,
                chapters   TEXT NOT NULL,
                model      TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

    def _ensure_social_posts_table(self, conn: sqlite3.Connection) -> None:
        """Create the ``social_posts`` table if it is missing (idempotent)."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS social_posts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id   INTEGER NOT NULL UNIQUE REFERENCES videos(id) ON DELETE CASCADE,
                post       TEXT NOT NULL,
                hashtags   TEXT NOT NULL,
                model      TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

    def _ensure_fts_tables(self, conn: sqlite3.Connection) -> None:
        """Create the v6 FTS5 search tables and backfill them if empty.

        ``CREATE VIRTUAL TABLE IF NOT EXISTS`` covers fresh databases (schema
        v6) and databases migrated from earlier versions. A database that
        already has videos but whose FTS tables are empty (created before the
        search feature landed, or a partially applied migration) is backfilled
        so every stored transcript becomes searchable immediately. The check
        is idempotent: once indexed, the tables are never re-populated.
        """
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS videos_fts USING fts5("
            "user_id UNINDEXED, title, uploader)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5("
            "video_id UNINDEXED, user_id UNINDEXED, start UNINDEXED, "
            "end UNINDEXED, text)"
        )
        indexed = conn.execute("SELECT COUNT(*) FROM videos_fts").fetchone()[0]
        if indexed:
            return
        video_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        if not video_count:
            return
        conn.execute(
            """
            INSERT INTO videos_fts(rowid, user_id, title, uploader)
            SELECT id, user_id, title, uploader FROM videos
            """
        )
        conn.execute(
            """
            INSERT INTO segments_fts(rowid, video_id, user_id, start, end, text)
            SELECT s.id, s.video_id, v.user_id, s.start, s.end, s.text
            FROM segments s JOIN videos v ON v.id = s.video_id
            """
        )
        logger.info("Backfilled full-text search index from existing library")

    # ---------- Users ----------

    def create_user(
        self,
        *,
        email: str,
        password_hash: str,
        verification_token: str | None = None,
        verification_token_expires_at: str | None = None,
    ) -> int:
        """Create a new user and return their id.

        Args:
            email: The user's email address (stored lowercased).
            password_hash: The PBKDF2 hash of the user's password.
            verification_token: Optional email-verification token.
            verification_token_expires_at: ISO timestamp when the token expires.

        Returns:
            The new user's primary key.

        Raises:
            sqlite3.IntegrityError: If the email is already registered.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (
                    email, password_hash, verification_token, verification_token_expires_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    email.strip().lower(),
                    password_hash,
                    verification_token,
                    verification_token_expires_at,
                ),
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
                "SELECT id, email, password_hash, is_verified, created_at FROM users WHERE email = ?",
                (email.strip().lower(),),
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_verification_token(self, token: str) -> dict[str, Any] | None:
        """Fetch a user by their email-verification token.

        Args:
            token: The verification token.

        Returns:
            The user row (with ``verification_token_expires_at``), or ``None``.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, email, is_verified, verification_token_expires_at
                FROM users
                WHERE verification_token = ?
                """,
                (token,),
            ).fetchone()
        return dict(row) if row else None

    def verify_user(self, user_id: int) -> None:
        """Mark a user's email as verified and clear their token.

        Args:
            user_id: The user's primary key.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET is_verified = 1, verification_token = NULL, verification_token_expires_at = NULL
                WHERE id = ?
                """,
                (user_id,),
            )

    def set_verification_token(
        self, *, user_id: int, token: str, expires_at: str
    ) -> None:
        """Store a fresh verification token for a user (used on resend).

        Args:
            user_id: The user's primary key.
            token: The new verification token.
            expires_at: ISO timestamp when the token expires.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET verification_token = ?, verification_token_expires_at = ?
                WHERE id = ?
                """,
                (token, expires_at, user_id),
            )

    def get_reset_otp_user(self, email: str) -> dict[str, Any] | None:
        """Fetch a user's pending password-reset OTP info by email.

        Args:
            email: The user's email address.

        Returns:
            The user row (with ``reset_token``, ``reset_token_expires_at`` and
            ``reset_token_attempts``), or ``None``.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, email, reset_token, reset_token_expires_at, reset_token_attempts
                FROM users
                WHERE email = ?
                """,
                (email.strip().lower(),),
            ).fetchone()
        return dict(row) if row else None

    def set_reset_otp(self, *, user_id: int, otp_hash: str, expires_at: str) -> None:
        """Store a password-reset OTP hash for a user.

        Args:
            user_id: The user's primary key.
            otp_hash: SHA-256 hex of the 6-digit OTP.
            expires_at: ISO timestamp when the OTP expires.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET reset_token = ?, reset_token_expires_at = ?, reset_token_attempts = 0
                WHERE id = ?
                """,
                (otp_hash, expires_at, user_id),
            )

    def increment_reset_attempts(self, user_id: int) -> int:
        """Record a failed OTP attempt and return the new attempt count.

        Args:
            user_id: The user's primary key.

        Returns:
            The number of failed attempts so far.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET reset_token_attempts = reset_token_attempts + 1 WHERE id = ?",
                (user_id,),
            )
            row = conn.execute(
                "SELECT reset_token_attempts FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return int(row[0])

    def update_password(self, *, user_id: int, password_hash: str) -> None:
        """Replace a user's password hash and clear any reset OTP.

        Args:
            user_id: The user's primary key.
            password_hash: The new PBKDF2 hash.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, reset_token = NULL,
                    reset_token_expires_at = NULL, reset_token_attempts = 0
                WHERE id = ?
                """,
                (password_hash, user_id),
            )

    def delete_user(self, user_id: int) -> bool:
        """Delete a user (cascades to their videos, chats, etc.).

        Args:
            user_id: The user's primary key.

        Returns:
            ``True`` if a user was deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cursor.rowcount > 0

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

    def create_pending_video(
        self,
        *,
        user_id: int,
        youtube_id: str,
        youtube_url: str,
        title: str,
    ) -> int:
        """Create (or reset) a video row for an in-progress transcription.

        A row that is already ``processing`` for the same video is returned
        untouched so duplicate submissions share one background task. A row in
        any other state (``ready`` or ``error``) is reset to ``processing``
        with a placeholder title; the finished transcript later overwrites it
        in place via :meth:`save_video`, keeping the primary key stable.

        Args:
            user_id: The owning user.
            youtube_id: Unique YouTube video ID.
            youtube_url: Canonical watch URL.
            title: Placeholder title shown while transcribing.

        Returns:
            The primary key of the pending video row.
        """
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, status FROM videos WHERE user_id = ? AND youtube_id = ?",
                (user_id, youtube_id),
            ).fetchone()
            if existing is not None:
                video_id = int(existing["id"])
                if existing["status"] == "processing":
                    return video_id
                conn.execute(
                    """
                    UPDATE videos SET
                        category_id = NULL, youtube_url = ?, title = ?, uploader = NULL,
                        language = '', language_probability = 0, duration = 0,
                        transcript = '', status = 'processing', error = NULL,
                        progress_percent = 0, progress_message = NULL
                    WHERE id = ?
                    """,
                    (youtube_url, title, video_id),
                )
                return video_id
            cursor = conn.execute(
                """
                INSERT INTO videos (
                    user_id, youtube_id, youtube_url, title,
                    language, language_probability, duration, transcript, status,
                    progress_percent, progress_message
                ) VALUES (?, ?, ?, ?, '', 0, 0, '', 'processing', 0, NULL)
                """,
                (user_id, youtube_id, youtube_url, title),
            )
            return int(cursor.lastrowid)

    def mark_video_status(
        self, *, video_id: int, status: str, error: str | None = None
    ) -> None:
        """Update a video's background-transcription status.

        Args:
            video_id: The video to update.
            status: ``"processing"``, ``"ready"`` or ``"error"``.
            error: Optional failure detail shown on the dashboard card.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE videos SET status = ?, error = ?, progress_message = NULL WHERE id = ?",
                (status, error, video_id),
            )

    def update_video_progress(
        self, *, video_id: int, percent: float, message: str
    ) -> None:
        """Persist live transcription progress for a pending video row.

        Args:
            video_id: The video being transcribed.
            percent: Overall progress, 0-100.
            message: A short human-readable status message.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE videos SET progress_percent = ?, progress_message = ? WHERE id = ?",
                (percent, message, video_id),
            )

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
        """Insert (or update in place) a video with its segments and RAG chunks.

        When a row already exists for ``(user_id, youtube_id)`` (e.g. a
        pending row created by :meth:`create_pending_video`), it is updated in
        place with the finished transcript and marked ``ready``, preserving the
        primary key and any conversations attached to it.

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
                video_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE videos SET
                        category_id = ?, youtube_url = ?, title = ?, uploader = ?,
                        language = ?, language_probability = ?, duration = ?,
                        transcript = ?, status = 'ready', error = NULL,
                        progress_percent = 100, progress_message = NULL
                    WHERE id = ?
                    """,
                    (
                        category_id,
                        youtube_url,
                        title,
                        uploader,
                        language,
                        language_probability,
                        duration,
                        transcript,
                        video_id,
                    ),
                )
                conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
                conn.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO videos (
                        user_id, category_id, youtube_id, youtube_url, title, uploader,
                        language, language_probability, duration, transcript, status,
                        progress_percent, progress_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', 100, NULL)
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

            # Keep the full-text search index in sync with the stored transcript.
            conn.execute("DELETE FROM videos_fts WHERE rowid = ?", (video_id,))
            conn.execute(
                "INSERT INTO videos_fts(rowid, user_id, title, uploader) "
                "VALUES (?, ?, ?, ?)",
                (video_id, user_id, title, uploader),
            )
            conn.execute("DELETE FROM segments_fts WHERE video_id = ?", (video_id,))
            segment_ids = [
                int(row[0])
                for row in conn.execute(
                    "SELECT id FROM segments WHERE video_id = ? ORDER BY id",
                    (video_id,),
                ).fetchall()
            ]
            conn.executemany(
                "INSERT INTO segments_fts(rowid, video_id, user_id, start, end, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (seg_id, video_id, user_id, s["start"], s["end"], s["text"])
                    for seg_id, s in zip(segment_ids, segments)
                ],
            )

        logger.info("Saved video %d for user %d (video %s)", video_id, user_id, youtube_id)
        return video_id

    def find_video_by_youtube_id(
        self, *, user_id: int, youtube_id: str
    ) -> dict[str, Any] | None:
        """Return ``{"id", "status"}`` for a user's video by YouTube ID.

        Args:
            user_id: The owning user.
            youtube_id: Unique YouTube video ID.

        Returns:
            A dict with the row's ``id`` and ``status``, or ``None`` when the
            user has no video for this YouTube ID.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, status FROM videos WHERE user_id = ? AND youtube_id = ?",
                (user_id, youtube_id),
            ).fetchone()
        return dict(row) if row is not None else None

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
            if cursor.rowcount:
                conn.execute("DELETE FROM videos_fts WHERE rowid = ?", (video_id,))
                conn.execute("DELETE FROM segments_fts WHERE video_id = ?", (video_id,))
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

    def search_library(
        self, *, user_id: int, query: str, limit: int = 20
    ) -> dict[str, list[dict[str, Any]]]:
        """Full-text search across a user's library.

        Searches video titles/uploaders in ``videos_fts`` and timestamped
        transcript text in ``segments_fts``, both ranked by FTS5 relevance and
        scoped to the owning user. Only ready videos are returned.

        Args:
            user_id: The owning user.
            query: Free-text search terms (title and/or transcript).
            limit: Maximum results returned per section.

        Returns:
            A dict with ``"videos"`` (title/uploader matches, carrying video
            metadata) and ``"segments"`` (timestamped transcript matches,
            carrying ``start``/``end`` so the UI can jump to the moment).
        """
        fts = _fts_query(query)
        if not fts:
            return {"videos": [], "segments": []}
        with self._connect() as conn:
            videos = conn.execute(
                """
                SELECT v.id AS video_id, v.youtube_id, v.youtube_url, v.title,
                       v.uploader, v.duration, bm25(videos_fts) AS rank
                FROM videos_fts f
                JOIN videos v ON v.id = f.rowid
                WHERE videos_fts MATCH ? AND f.user_id = ? AND v.status = 'ready'
                ORDER BY rank
                LIMIT ?
                """,
                (fts, user_id, limit),
            ).fetchall()
            segments = conn.execute(
                """
                SELECT v.id AS video_id, v.youtube_id, v.youtube_url, v.title,
                       v.uploader, s.start, s.end, s.text,
                       bm25(segments_fts) AS rank
                FROM segments_fts s
                JOIN videos v ON v.id = s.video_id
                WHERE segments_fts MATCH ? AND s.user_id = ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts, user_id, limit),
            ).fetchall()
        return {
            "videos": [dict(row) for row in videos],
            "segments": [dict(row) for row in segments],
        }

    # ---------- Summaries ----------

    def save_summary(
        self,
        *,
        video_id: int,
        tldr: str,
        takeaways: list[str],
        chapters: list[dict[str, Any]],
        model: str,
    ) -> None:
        """Store (or replace) the AI summary for a video.

        Args:
            video_id: The video being summarized.
            tldr: One-paragraph TL;DR.
            takeaways: List of key takeaway strings.
            chapters: List of ``{"title", "start", "end"}`` dicts.
            model: The Gemini model used to generate the summary.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summaries (video_id, tldr, takeaways, chapters, model)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    tldr = excluded.tldr,
                    takeaways = excluded.takeaways,
                    chapters = excluded.chapters,
                    model = excluded.model,
                    created_at = datetime('now')
                """,
                (
                    video_id,
                    tldr,
                    json.dumps(takeaways),
                    json.dumps(chapters),
                    model,
                ),
            )

    def get_summary(self, video_id: int) -> dict[str, Any] | None:
        """Fetch the stored AI summary for a video.

        Args:
            video_id: The video's primary key.

        Returns:
            A dict with ``tldr``, ``takeaways``, ``chapters``, ``model`` and
            ``created_at``, or ``None`` if no summary exists.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM summaries WHERE video_id = ?", (video_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        try:
            item["takeaways"] = json.loads(item["takeaways"])
        except json.JSONDecodeError:
            item["takeaways"] = []
        try:
            item["chapters"] = json.loads(item["chapters"])
        except json.JSONDecodeError:
            item["chapters"] = []
        return item

    # ---------- Social posts ----------

    def save_social_post(
        self,
        *,
        video_id: int,
        post: str,
        hashtags: list[str],
        model: str,
    ) -> None:
        """Store (or replace) the generated social post for a video.

        Args:
            video_id: The video being posted about.
            post: The editable post body.
            hashtags: List of hashtag strings.
            model: The Gemini model used to generate the post.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO social_posts (video_id, post, hashtags, model)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    post = excluded.post,
                    hashtags = excluded.hashtags,
                    model = excluded.model,
                    created_at = datetime('now')
                """,
                (video_id, post, json.dumps(hashtags), model),
            )

    def get_social_post(self, video_id: int) -> dict[str, Any] | None:
        """Fetch the stored social post for a video.

        Args:
            video_id: The video's primary key.

        Returns:
            A dict with ``post``, ``hashtags``, ``model`` and ``created_at``,
            or ``None`` if no post has been generated yet.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM social_posts WHERE video_id = ?", (video_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        try:
            item["hashtags"] = json.loads(item["hashtags"])
        except json.JSONDecodeError:
            item["hashtags"] = []
        return item

    # ---------- Conversations & messages ----------

    def create_conversation(
        self, *, user_id: int, video_id: int | None = None, title: str | None = None
    ) -> int:
        """Create a conversation for a user.

        Args:
            user_id: The owning user.
            video_id: The video being discussed, or ``None`` for a library-level
                conversation that spans all of the user's videos.
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

    def list_library_conversations(self, user_id: int) -> list[dict[str, Any]]:
        """List a user's library-level conversations, most recently active first.

        Args:
            user_id: The owning user.

        Returns:
            A list of conversation rows whose ``video_id`` is NULL.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE video_id IS NULL AND user_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (user_id,),
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
