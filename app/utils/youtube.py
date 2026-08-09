"""YouTube URL validation and video ID extraction utilities."""

import re
from urllib.parse import parse_qs, urlparse

from app.exceptions import InvalidURLError

YOUTUBE_HOSTS: set[str] = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com",
}

# Query key that carries the video ID.
WATCH_QUERY_KEY = "v"
# Path segment after the host for shorthand URLs.
_ID_PATH_SEGMENT = 1

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,11}$")


def extract_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from any supported URL.

    Supported forms:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID

    Args:
        url: The raw YouTube URL provided by the client.

    Returns:
        The 11-character YouTube video ID.

    Raises:
        InvalidURLError: If the URL is empty, not a valid URL,
            not a YouTube domain, or contains no valid video ID.
    """
    parsed = _parse_youtube_url(url)

    video_id = _extract_id_from_parsed(parsed)
    if not _is_valid_id(video_id):
        raise InvalidURLError()

    return video_id


def normalize_youtube_url(url: str) -> str:
    """Return a canonical ``https://www.youtube.com/watch?v=<id>`` URL.

    Args:
        url: The raw YouTube URL provided by the client.

    Returns:
        The normalized watch URL.

    Raises:
        InvalidURLError: If the URL is invalid.
    """
    video_id = extract_video_id(url)
    return f"https://www.youtube.com/watch?v={video_id}"


def _parse_youtube_url(url: str) -> urlparse:
    """Validate scheme/host and return the parsed URL components.

    Args:
        url: The raw URL string.

    Returns:
        The parsed URL.

    Raises:
        InvalidURLError: If the URL cannot be parsed or is not a YouTube URL.
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise InvalidURLError()

    url = url.strip()

    # Accept bare video IDs for convenience? No - the spec requires a URL,
    # so require a proper scheme here.
    if not url.startswith(("http://", "https://")):
        raise InvalidURLError()

    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise InvalidURLError() from exc

    if parsed.scheme not in {"http", "https"}:
        raise InvalidURLError()

    if parsed.netloc.lower().rstrip(".") not in YOUTUBE_HOSTS:
        raise InvalidURLError()

    return parsed


def _extract_id_from_parsed(parsed: urlparse) -> str | None:
    """Extract a video ID from parsed URL components.

    Args:
        parsed: The parsed YouTube URL.

    Returns:
        The video ID or ``None`` if it cannot be determined.
    """
    host = parsed.netloc.lower().rstrip(".")

    if host == "youtu.be":
        return _path_segment(parsed)

    if parsed.path == "/watch" or parsed.path == "":
        query = parse_qs(parsed.query)
        raw = query.get(WATCH_QUERY_KEY, [None])[0]
        return raw or None

    # Handles /shorts/, /embed/ and /v/ paths.
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) >= 2 and segments[0] in {"shorts", "embed", "v"}:
        return segments[_ID_PATH_SEGMENT]

    return None


def _path_segment(parsed: urlparse) -> str | None:
    """Return the first non-empty path segment for youtu.be URLs.

    Args:
        parsed: The parsed youtu.be URL.

    Returns:
        The first path segment or ``None``.
    """
    segments = [segment for segment in parsed.path.split("/") if segment]
    return segments[0] if segments else None


def _is_valid_id(video_id: str | None) -> bool:
    """Check whether a candidate matches the YouTube ID format.

    Args:
        video_id: The candidate video ID.

    Returns:
        ``True`` if the ID matches the 11-char slug pattern.
    """
    return bool(video_id) and bool(_VIDEO_ID_PATTERN.fullmatch(video_id))
