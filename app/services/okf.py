"""Open Knowledge Format (OKF) bundle management.

The OKF knowledge layer sits between raw transcript sources and the per-video
RAG index. Each user owns a bundle directory of markdown concept documents
under ``<root>/<user_id>/``. One concept document (``videos/<youtube_id>.md``)
is maintained per stored video: YAML frontmatter (``type``, ``title``,
``description``, ``resource``, ``tags``, provenance/trust fields) plus a body
with the summary, key topics and entities.

``index.md`` files give an agent or human a cheap catalog for progressive
disclosure (coarse-grain selection), ``log.md`` records the timeline of
changes, and markdown links express relationships — the LLM-wiki pattern
formalized by the Open Knowledge Format spec.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONCEPT_TYPE = "Video Transcript"
ACTOR_PROCESS = "process:whisper-pipeline"
ACTOR_GENERATOR = "okf-service"
OKF_VERSION = "0.2"

_INDEX_LINE_RE = re.compile(
    r"^\* \[(?P<title>.*?)\]\(videos/(?P<youtube_id>[\w-]+)\.md\)\s*-\s*(?P<description>.*)$"
)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _now() -> str:
    """Return the current UTC time as an ISO 8601 ``...Z`` string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _today() -> str:
    """Return the current UTC date as ``YYYY-MM-DD``."""
    return datetime.now(timezone.utc).date().isoformat()


def _yaml_scalar(value: str) -> str:
    """Return a YAML-safe single-line scalar for ``value``."""
    text = " ".join(str(value).split())
    if (
        not text
        or text in {"null", "true", "false", "~"}
        or ":" in text
        or any(char in text for char in "\"'#,[]{}&*!|>%@`")
    ):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _sanitize_tag(tag: str) -> str:
    """Normalize a free-form tag into a lowercase slug."""
    return re.sub(r"[^a-z0-9_-]", "", str(tag).lower().strip())[:24]


def _truncate_words(text: str, limit: int) -> str:
    """Return the first ``limit`` words of ``text`` with an ellipsis."""
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]) + "…"


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Best-effort extraction of the single-line frontmatter fields."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}
    block = match.group(1)
    fields: dict[str, str] = {}
    for key in ("type", "title", "description", "resource", "tags"):
        line = re.search(rf"^{key}:\s*(.*)$", block, re.MULTILINE)
        if line is not None:
            fields[key] = line.group(1).strip().strip('"')
    return fields


class OKFService:
    """Maintains the OKF knowledge bundle for each user."""

    def __init__(
        self, root_dir: Path, gemini: Any | None = None, summary_chars: int = 40000
    ) -> None:
        """Initialize the OKF service.

        Args:
            root_dir: Directory that holds one bundle per user.
            gemini: Optional Gemini service used to produce concept summaries.
            summary_chars: Transcript characters fed to summarization.
        """
        self.root_dir = Path(root_dir)
        self.gemini = gemini
        self.summary_chars = summary_chars

    # ---------- Bundle layout ----------

    def bundle_root(self, user_id: int) -> Path:
        """Return the bundle directory for ``user_id``."""
        return self.root_dir / str(user_id)

    def concept_path(self, user_id: int, youtube_id: str) -> Path:
        """Return the concept document path for a video."""
        return self.bundle_root(user_id) / "videos" / f"{youtube_id}.md"

    # ---------- Ingest ----------

    async def upsert_video(
        self,
        user_id: int,
        *,
        youtube_id: str,
        youtube_url: str,
        title: str,
        uploader: str | None,
        duration: float,
        transcript: str,
    ) -> Path:
        """Write (or update) the concept document for a stored video.

        The concept document is the coarse-grain knowledge layer: it is what a
        cross-video query reads first to decide which videos are relevant, then
        per-video RAG supplies exact timestamped excerpts.

        Args:
            user_id: The owning user.
            youtube_id: YouTube video ID (used as the concept identity).
            youtube_url: Canonical watch URL (frontmatter ``resource``).
            title: Video title.
            uploader: Channel name, when available.
            duration: Video duration in seconds.
            transcript: Full transcript text.

        Returns:
            The path of the written concept document.
        """
        summary = await self._build_summary(
            title=title, uploader=uploader, duration=duration, transcript=transcript
        )
        document = self._render_concept_document(
            youtube_url=youtube_url,
            title=title,
            transcript=transcript,
            summary=summary,
        )
        path = self.concept_path(user_id, youtube_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
        self._refresh_videos_index(user_id)
        self._refresh_root_index(user_id)
        self._append_log(user_id, f"**Update**: Added video [{title}](videos/{youtube_id}.md)")
        logger.info("OKF concept %s updated for user %d", path, user_id)
        return path

    async def _build_summary(
        self, *, title: str, uploader: str | None, duration: float, transcript: str
    ) -> dict[str, Any] | None:
        """Produce a concept summary via Gemini, or ``None`` as a fallback."""
        if self.gemini is None or not self.gemini.available:
            return None
        try:
            return await self.gemini.generate_concept_summary(
                title=title, uploader=uploader, duration=duration, transcript=transcript
            )
        except Exception:  # noqa: BLE001 - summarization must never block ingest
            logger.exception("Concept summary failed for %r; using fallback", title)
            return None

    def _render_concept_document(
        self,
        *,
        youtube_url: str,
        title: str,
        transcript: str,
        summary: dict[str, Any] | None,
    ) -> str:
        """Render a conformant OKF concept document (frontmatter + body)."""
        now = _now()
        description = (
            (summary or {}).get("description")
            or _truncate_words(transcript, 40)
            or "Video transcript summary"
        )
        tags = [_sanitize_tag(tag) for tag in (summary or {}).get("tags") or []]
        tags = [tag for tag in tags if tag]
        frontmatter = "\n".join(
            [
                "---",
                f"type: {CONCEPT_TYPE}",
                f"title: {_yaml_scalar(title)}",
                f"description: {_yaml_scalar(description)}",
                f"resource: {_yaml_scalar(youtube_url)}",
                f"tags: [{', '.join(tags)}]",
                f"generated: {{ by: {ACTOR_GENERATOR}, at: {now} }}",
                f"verified: {{ by: {ACTOR_PROCESS}, at: {now} }}",
                "status: stable",
                "---",
                "",
            ]
        )
        return frontmatter + self._render_body(transcript=transcript, summary=summary)

    def _render_body(self, *, transcript: str, summary: dict[str, Any] | None) -> str:
        """Render the markdown body of a concept document."""
        lines = ["# Summary"]
        if summary and (summary.get("summary") or "").strip():
            lines.append(" ".join(str(summary["summary"]).split()))
        else:
            lines.append(_truncate_words(transcript, 90) or "(no transcript content)")
        lines.append("")
        lines.append("# Key topics")
        topics = (summary or {}).get("topics") or []
        if topics:
            lines.extend(f"- {topic}" for topic in topics if str(topic).strip())
        elif summary is None:
            lines.append("- (key topics not generated)")
        lines.append("")
        lines.append("# Entities")
        entities = (summary or {}).get("entities") or []
        if entities:
            lines.extend(f"- {entity}" for entity in entities if str(entity).strip())
        elif summary is None:
            lines.append("- (entities not generated)")
        return "\n".join(lines) + "\n"

    # ---------- Delete ----------

    def delete_video(self, user_id: int, youtube_id: str) -> None:
        """Remove a video's concept document and refresh the bundle indexes."""
        path = self.concept_path(user_id, youtube_id)
        removed = path.exists()
        if removed:
            path.unlink()
        self._refresh_videos_index(user_id)
        self._refresh_root_index(user_id)
        self._append_log(user_id, f"**Deletion**: Removed video concept `{youtube_id}`")
        if removed:
            logger.info("OKF concept %s removed for user %d", path, user_id)

    # ---------- Consumption ----------

    def list_concepts(self, user_id: int) -> list[dict[str, str]]:
        """Return the video concept catalog from the bundle's ``index.md``.

        The index is the progressive-disclosure surface: cross-video queries
        score candidate videos against these cheap entries before any RAG work.

        Args:
            user_id: The owning user.

        Returns:
            A list of ``{"youtube_id", "path", "title", "description"}`` dicts.
        """
        index = self.bundle_root(user_id) / "videos" / "index.md"
        if not index.exists():
            return []
        entries: list[dict[str, str]] = []
        for line in index.read_text(encoding="utf-8").splitlines():
            match = _INDEX_LINE_RE.match(line.strip())
            if match is not None:
                entries.append(
                    {
                        "youtube_id": match.group("youtube_id"),
                        "path": f"videos/{match.group('youtube_id')}.md",
                        "title": match.group("title").strip(),
                        "description": match.group("description").strip(),
                    }
                )
        return entries

    # ---------- Index & log maintenance ----------

    def _refresh_videos_index(self, user_id: int) -> None:
        """Regenerate ``videos/index.md`` from the existing concept documents."""
        videos_dir = self.bundle_root(user_id) / "videos"
        entries: list[tuple[str, str, str]] = []
        if videos_dir.exists():
            for path in sorted(videos_dir.glob("*.md")):
                if path.name == "index.md":
                    continue
                fields = _parse_frontmatter(path.read_text(encoding="utf-8"))
                entries.append(
                    (path.stem, fields.get("title") or path.stem, fields.get("description") or "")
                )
        lines = ["# Videos", ""]
        lines.extend(f"* [{title}](videos/{youtube_id}.md) - {description}" for youtube_id, title, description in entries)
        videos_dir.mkdir(parents=True, exist_ok=True)
        (videos_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _refresh_root_index(self, user_id: int) -> None:
        """Regenerate the bundle-root ``index.md``."""
        root = self.bundle_root(user_id)
        root.mkdir(parents=True, exist_ok=True)
        (root / "index.md").write_text(
            "# Knowledge Bundle\n\n"
            "* [Videos](videos/) - Video transcript concepts\n",
            encoding="utf-8",
        )

    def _append_log(self, user_id: int, entry: str) -> None:
        """Append a date-grouped, newest-first entry to the bundle ``log.md``."""
        log_path = self.bundle_root(user_id) / "log.md"
        header = f"## {_today()}"
        if not log_path.exists():
            log_path.write_text(f"# Update Log\n\n{header}\n* {entry}\n", encoding="utf-8")
            return
        lines = log_path.read_text(encoding="utf-8").splitlines()
        if lines and lines[0] == header:
            lines.insert(1, f"* {entry}")
        else:
            lines[0:0] = [header, f"* {entry}"]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
