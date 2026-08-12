"""Transcript export: SRT, VTT, TXT, Markdown and PDF.

Segments are stored as ``{"start", "end", "text"}`` dicts with times in
seconds; every exporter builds on a small set of pure string functions so the
formats stay consistent (timestamps, word wrapping, escaping) and testable
without a database.
"""

import logging
import re
import unicodedata
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

logger = logging.getLogger(__name__)

EXPORT_FORMATS = ("srt", "vtt", "txt", "md", "pdf")

_MEDIA_TYPES = {
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "pdf": "application/pdf",
}

_FILE_EXTENSIONS = {
    "srt": "srt",
    "vtt": "vtt",
    "txt": "txt",
    "md": "md",
    "pdf": "pdf",
}

_PUNCTUATION_RE = re.compile(r"[\t ]+")


def _clean(text: str) -> str:
    """Collapse whitespace while preserving inner punctuation."""
    return _PUNCTUATION_RE.sub(" ", (text or "").strip())


def srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp ``HH:MM:SS,mmm``."""
    seconds = max(0.0, float(seconds))
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def vtt_timestamp(seconds: float) -> str:
    """Format seconds as a VTT timestamp ``HH:MM:SS.mmm``."""
    return srt_timestamp(seconds).replace(",", ".")


def export_srt(
    segments: list[dict[str, Any]],
) -> bytes:
    """Render segments as SubRip (SRT) subtitles."""
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = _clean(segment["text"])
        if not text:
            continue
        blocks.append(
            f"{index}\n"
            f"{srt_timestamp(segment['start'])} --> {srt_timestamp(segment['end'])}\n"
            f"{text}"
        )
    return ("\n\n".join(blocks) + "\n").encode("utf-8")


def export_vtt(
    segments: list[dict[str, Any]],
) -> bytes:
    """Render segments as WebVTT subtitles."""
    blocks = ["WEBVTT"]
    for segment in segments:
        text = _clean(segment["text"])
        if not text:
            continue
        blocks.append(
            f"{vtt_timestamp(segment['start'])} --> {vtt_timestamp(segment['end'])}\n{text}"
        )
    return ("\n\n".join(blocks) + "\n").encode("utf-8")


def export_txt(
    segments: list[dict[str, Any]],
) -> bytes:
    """Render segments as plain text with ``[M:SS]`` time markers."""
    lines = []
    for segment in segments:
        text = _clean(segment["text"])
        if not text:
            continue
        start = _minutes_colon_seconds(segment["start"])
        end = _minutes_colon_seconds(segment["end"])
        lines.append(f"[{start}-{end}] {text}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def export_markdown(
    *,
    title: str,
    uploader: str | None,
    youtube_url: str,
    duration: float,
    segments: list[dict[str, Any]],
) -> bytes:
    """Render the transcript as a Markdown document."""
    lines = [
        f"# {title or 'Transcript'}",
        "",
    ]
    meta: list[str] = []
    if uploader:
        meta.append(f"- **Channel:** {uploader}")
    if youtube_url:
        meta.append(f"- **Source:** {youtube_url}")
    if duration:
        meta.append(f"- **Duration:** {_minutes_colon_seconds(duration)}")
    if meta:
        lines.extend(meta)
        lines.append("")
    lines.append("## Transcript")
    lines.append("")
    for segment in segments:
        text = _clean(segment["text"])
        if not text:
            continue
        start = _minutes_colon_seconds(segment["start"])
        end = _minutes_colon_seconds(segment["end"])
        lines.append(f"**{start}**–**{end}**  ")
        lines.append(text)
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def export_pdf(
    *,
    title: str,
    uploader: str | None,
    youtube_url: str,
    duration: float,
    segments: list[dict[str, Any]],
) -> bytes:
    """Render the transcript as a paginated PDF via ReportLab."""
    styles = getSampleStyleSheet()
    heading = ParagraphStyle(
        "ExportTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=6 * mm,
    )
    meta = ParagraphStyle(
        "ExportMeta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        spaceAfter=2 * mm,
    )
    cell = ParagraphStyle(
        "ExportCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    )
    time_cell = ParagraphStyle(
        "ExportTime",
        parent=cell,
        fontName="Courier-Bold",
        fontSize=8,
        leading=11,
    )

    flowables: list[Any] = [Paragraph(_clean(title) or "Transcript", heading)]

    if uploader:
        flowables.append(Paragraph(f"Channel: {_clean(uploader)}", meta))
    if youtube_url:
        flowables.append(Paragraph(f"Source: {_clean(youtube_url)}", meta))
    if duration:
        flowables.append(
            Paragraph(f"Duration: {_minutes_colon_seconds(duration)}", meta)
        )
    flowables.append(Spacer(1, 4 * mm))

    rows: list[list[Any]] = []
    for segment in segments:
        text = _clean(segment["text"])
        if not text:
            continue
        time_label = (
            f"{_minutes_colon_seconds(segment['start'])}–"
            f"{_minutes_colon_seconds(segment['end'])}"
        )
        rows.append(
            [
                Paragraph(time_label, time_cell),
                Paragraph(_escape_pdf(text), cell),
            ]
        )

    if rows:
        table = Table(rows, colWidths=[28 * mm, 160 * mm], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (0, -1), 0),
                    ("LEFTPADDING", (1, 0), (1, -1), 6),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e0e0e0")),
                ]
            )
        )
        flowables.append(table)
    else:
        flowables.append(Paragraph("No transcript segments.", cell))

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=_clean(title) or "Transcript",
        author=_clean(uploader) if uploader else None,
    )
    document.build(flowables)
    return buffer.getvalue()


def _escape_pdf(text: str) -> str:
    """Escape XML/HTML entities so ReportLab Paragraphs render text literally."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _minutes_colon_seconds(seconds: float) -> str:
    """Format seconds as ``M:SS`` (e.g. ``3:07``)."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def export(
    export_format: str,
    *,
    title: str,
    uploader: str | None,
    youtube_url: str,
    duration: float,
    segments: list[dict[str, Any]],
) -> tuple[str, str, bytes]:
    """Render a transcript in the requested format.

    Args:
        export_format: One of ``"srt"``, ``"vtt"``, ``"txt"``, ``"md"`` or
            ``"pdf"``.
        title: Video title.
        uploader: Channel name, when available.
        youtube_url: Canonical watch URL.
        duration: Video duration in seconds.
        segments: List of ``{"start", "end", "text"}`` dicts.

    Returns:
        A ``(filename, media_type, content)`` tuple ready to serve.

    Raises:
        ValueError: If ``export_format`` is not a supported format.
    """
    fmt = export_format.lower()
    if fmt not in EXPORT_FORMATS:
        raise ValueError(
            f"Unsupported export format '{export_format}'. "
            f"Choose one of: {', '.join(EXPORT_FORMATS)}."
        )
    kwargs = {
        "title": title,
        "uploader": uploader,
        "youtube_url": youtube_url,
        "duration": duration,
        "segments": segments,
    }
    if fmt == "srt":
        content = export_srt(segments)
    elif fmt == "vtt":
        content = export_vtt(segments)
    elif fmt == "txt":
        content = export_txt(segments)
    elif fmt == "md":
        content = export_markdown(**kwargs)
    else:
        content = export_pdf(**kwargs)

    stem = slugify(title) or "transcript"
    filename = f"{stem}.{_FILE_EXTENSIONS[fmt]}"
    return filename, _MEDIA_TYPES[fmt], content


def slugify(value: str) -> str:
    """Convert a title into a URL/file safe slug (keeps ASCII letters/numbers)."""
    normalized = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore")
    text = normalized.decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:80]
