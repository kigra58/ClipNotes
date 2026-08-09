"""YouTube download service built on top of yt-dlp."""

import logging
import shutil
from pathlib import Path

import yt_dlp

from app.exceptions import DownloadError, VideoUnavailableError, VideoTooLongError
from app.utils.youtube import extract_video_id, normalize_youtube_url

logger = logging.getLogger(__name__)


class YouTubeService:
    """Downloads and converts YouTube audio into temporary MP3 files."""

    def __init__(self, temp_dir: Path, max_video_duration: float) -> None:
        """Initialize the service with its configuration.

        Args:
            temp_dir: Directory where downloaded audio files are stored.
            max_video_duration: Maximum allowed video duration in seconds.
        """
        self.temp_dir = temp_dir
        self.max_video_duration = max_video_duration

    def get_metadata(self, youtube_url: str) -> dict:
        """Fetch metadata for a YouTube video without downloading it.

        Args:
            youtube_url: The raw YouTube URL.

        Returns:
            A dict with ``video_id``, ``title``, ``duration`` and
            ``uploader`` keys.

        Raises:
            InvalidURLError: If the URL is invalid.
            VideoUnavailableError: If the video cannot be found.
            VideoTooLongError: If the video is longer than the maximum.
        """
        video_id = extract_video_id(youtube_url)

        with yt_dlp.YoutubeDL(self._dl_options(download=False)) as ydl:
            try:
                info = ydl.extract_info(normalize_youtube_url(youtube_url), download=False)
            except yt_dlp.utils.DownloadError as exc:
                logger.error("Metadata fetch failed for video %s: %s", video_id, exc)
                raise self._classify_download_error(exc) from exc

        if info is None:
            raise VideoUnavailableError()

        duration = float(info.get("duration") or 0.0)
        if duration > self.max_video_duration:
            logger.warning("Video %s exceeds max duration (%.1fs > %.1fs)", video_id, duration, self.max_video_duration)
            raise VideoTooLongError()

        return {
            "video_id": video_id,
            "title": info.get("title") or "Unknown title",
            "duration": duration,
            "uploader": info.get("uploader") or info.get("channel"),
        }

    def download_audio(self, youtube_url: str) -> dict:
        """Download a video's audio and convert it to MP3 via FFmpeg.

        Args:
            youtube_url: The raw YouTube URL.

        Returns:
            Metadata plus the absolute path to the downloaded MP3 file.

        Raises:
            InvalidURLError: If the URL is invalid.
            VideoUnavailableError: If the video cannot be found.
            VideoTooLongError: If the video is too long.
            DownloadError: If download or FFmpeg conversion fails.
        """
        video_id = extract_video_id(youtube_url)
        metadata = self.get_metadata(youtube_url)

        logger.info("Downloading audio for video %s", video_id)
        try:
            with yt_dlp.YoutubeDL(self._dl_options(download=True)) as ydl:
                ydl.extract_info(normalize_youtube_url(youtube_url), download=True)
        except yt_dlp.utils.DownloadError as exc:
            logger.error("Download failed for video %s: %s", video_id, exc)
            raise self._classify_download_error(exc) from exc

        audio_path = self.temp_dir / f"{video_id}.mp3"
        if not audio_path.is_file():
            logger.error("Converted audio file not found for video %s", video_id)
            raise DownloadError()

        logger.info("Audio downloaded for video %s", video_id)
        return {**metadata, "audio_path": str(audio_path)}

    def cleanup(self, video_id: str) -> None:
        """Delete all temporary files created for the given video ID.

        Args:
            video_id: The YouTube video ID whose temp files should be removed.
        """
        for path in self.temp_dir.glob(f"{video_id}.*"):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove temporary file %s: %s", path, exc)

    def _dl_options(self, *, download: bool) -> dict:
        """Build the shared yt-dlp options dict.

        Args:
            download: Whether the extractor should download the media.

        Returns:
            The options dict passed to ``YoutubeDL``.
        """
        options: dict = {
            "format": "bestaudio/best",
            "outtmpl": str(self.temp_dir / "%(id)s.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "skip_download": not download,
            "logger": logger,
        }

        ffmpeg_dir = self._find_ffmpeg_dir()
        if ffmpeg_dir is not None:
            options["ffmpeg_location"] = ffmpeg_dir

        return options

    @staticmethod
    def _find_ffmpeg_dir() -> str | None:
        """Locate a directory containing both ffmpeg and ffprobe executables.

        Returns:
            The directory as a string, or ``None`` if not found.
        """
        exe = shutil.which("ffmpeg")
        if exe:
            parent = Path(exe).resolve().parent
            if (parent / "ffprobe.exe").is_file():
                return str(parent)

        packages_dir = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        if packages_dir.is_dir():
            for pkg in packages_dir.glob("Gyan.FFmpeg*"):
                for bin_dir in pkg.glob("**/bin"):
                    if (bin_dir / "ffmpeg.exe").is_file() and (bin_dir / "ffprobe.exe").is_file():
                        return str(bin_dir)

        return None

    def _classify_download_error(self, exc: yt_dlp.utils.DownloadError) -> DownloadError:
        """Map a yt-dlp error to the appropriate application exception.

        Args:
            exc: The yt-dlp download error.

        Returns:
            A :class:`VideoUnavailableError` for missing videos or a
            :class:`DownloadError` otherwise.
        """
        message = str(exc).lower()
        unavailable_markers = (
            "video unavailable",
            "isn't available",
            "not available",
            "doesn't exist",
            "does not exist",
            "unsupported url",
            "not a video",
            "404",
        )
        if any(marker in message for marker in unavailable_markers):
            return VideoUnavailableError()
        if "ffmpeg" in message:
            return DownloadError(detail="FFmpeg is required but was not found on the system.")
        return DownloadError()
