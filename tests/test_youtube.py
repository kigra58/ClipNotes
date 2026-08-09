"""Tests for YouTube URL validation and video ID extraction."""

import pytest

from app.exceptions import InvalidURLError
from app.utils.youtube import extract_video_id, normalize_youtube_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=abc123", "abc123"),
        ("https://youtu.be/abc123", "abc123"),
        ("https://www.youtube.com/shorts/abc123", "abc123"),
        ("https://www.youtube.com/embed/abc123", "abc123"),
        ("https://www.youtube.com/watch?v=abc123&t=120", "abc123"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=abc123", "abc123"),
        ("http://www.youtube.com/watch?v=abc123", "abc123"),
    ],
)
def test_extract_video_id_accepts_valid_urls(url: str, expected: str) -> None:
    """Valid YouTube URLs must yield their 11-character video ID."""
    assert extract_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://google.com",
        "invalid-url",
        "",
        "   ",
        "https://www.youtube.com/",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=",
        "https://youtu.be/",
        "http://localhost:8000/some/path",
        "http://127.0.0.1/some/path",
        "file:///etc/passwd",
        "https://www.youtube.com/watch?v=thisismuchmorethan11chars",
    ],
)
def test_extract_video_id_rejects_invalid_urls(url: str) -> None:
    """Invalid URLs must raise InvalidURLError."""
    with pytest.raises(InvalidURLError):
        extract_video_id(url)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://youtu.be/abc123", "https://www.youtube.com/watch?v=abc123"),
        ("https://www.youtube.com/watch?v=abc123&t=120", "https://www.youtube.com/watch?v=abc123"),
    ],
)
def test_normalize_youtube_url(url: str, expected: str) -> None:
    """normalize_youtube_url must return a canonical watch URL."""
    assert normalize_youtube_url(url) == expected


def test_extract_video_id_rejects_playlists() -> None:
    """Playlist URLs without a video ID must be rejected."""
    with pytest.raises(InvalidURLError):
        extract_video_id("https://www.youtube.com/playlist?list=PLxxxxxxxxxxxx")
