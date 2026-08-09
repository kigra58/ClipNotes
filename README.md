# YouTube Transcript API

A backend-only REST API that converts a YouTube video into a timestamped transcript.

The API validates the YouTube URL, downloads the video audio with `yt-dlp`, converts it to MP3 with FFmpeg, transcribes it with `faster-whisper` (OpenAI Whisper), detects the spoken language and returns both the full transcript and timestamped segments. All temporary audio files are deleted after processing.

## Features

- YouTube URL validation (normal, `youtu.be`, `/shorts/`, `/embed/`)
- Audio extraction using `yt-dlp`
- FFmpeg conversion to MP3
- Whisper transcription (`faster-whisper`)
- Automatic spoken-language detection
- Timestamped transcript segments
- Video duration limit (default 2 hours)
- Automatic temporary-file cleanup
- Clean REST API with Swagger documentation (`/docs`)
- Playlist downloads are disabled (`noplaylist`)

## Requirements

- Python 3.12+
- FFmpeg (installed separately and available in `PATH`)

> FFmpeg is not a Python package. On Windows you can install it via
> `winget install ffmpeg` or download it from <https://ffmpeg.org/download.html>.
> On Linux/macOS use your package manager (e.g. `sudo apt install ffmpeg`).

## Installation

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Copy the sample configuration and adjust if needed:

```bash
cp .env.example .env
```

| Variable               | Description                                    | Default |
|------------------------|------------------------------------------------|---------|
| `APP_NAME`             | Application display name                       | `YouTube Transcript API` |
| `APP_VERSION`          | Application version                            | `1.0.0` |
| `WHISPER_MODEL`        | Whisper model size (tiny/base/small/medium/large) | `small` |
| `WHISPER_DEVICE`       | Inference device (`cpu` or `cuda`)             | `cpu` |
| `WHISPER_COMPUTE_TYPE` | Model precision (`int8`, `float16`, ...)       | `int8` |
| `TEMP_DIR`             | Directory for temporary audio files            | `temp` |
| `MAX_VIDEO_DURATION`   | Max video length in seconds (default 2 hours)  | `7200` |
| `CORS_ORIGINS`         | Comma-separated allowed origins (`*` for all)  | `*` |

The Whisper model is downloaded on first use and kept in memory for the lifetime of the process.

## Start the server

```bash
python run.py
```

This is equivalent to:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API endpoints

| Method | Path                | Description                          |
|--------|---------------------|--------------------------------------|
| `GET`  | `/health`           | Health check                         |
| `POST` | `/api/v1/transcribe`| Transcribe a YouTube video           |
| `GET`  | `/docs`             | Swagger UI                           |
| `GET`  | `/redoc`            | ReDoc                                |

## Example request

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

### Example response

```json
{
  "video_id": "dQw4w9WgXcQ",
  "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "title": "Example Video",
  "uploader": "Example Channel",
  "language": "en",
  "language_probability": 0.98,
  "duration": 245.5,
  "transcript": "Hello everyone, welcome to this example video.",
  "segments": [
    {
      "start": 0.0,
      "end": 4.5,
      "text": "Hello everyone, welcome to this example video."
    }
  ]
}
```

## Error responses

| Scenario                    | Status | Body                                  |
|-----------------------------|--------|---------------------------------------|
| Invalid / non-YouTube URL   | `400`  | `{"detail": "Invalid YouTube URL."}`  |
| Video unavailable           | `404`  | `{"detail": "YouTube video is unavailable."}` |
| Video longer than limit     | `400`  | `{"detail": "Video duration exceeds the maximum allowed duration."}` |
| Download / FFmpeg failure   | `502`  | `{"detail": "Failed to download video audio."}` |
| Transcription failure       | `500`  | `{"detail": "Failed to transcribe audio."}` |
| Unexpected error            | `500`  | `{"detail": "Unexpected server error."}` |

Internal errors are logged server-side; stack traces are never exposed to clients.

## Running tests

```bash
pytest -v
```

The tests cover the health endpoint, URL validation/ID extraction and the API
error paths. Real YouTube downloads and Whisper model downloads are not
triggered during tests (services are mocked).

## Project structure

```text
youtube-transcript-api/
│
├── app/
│   ├── main.py              # FastAPI app, lifespan, exception handlers
│   ├── config.py            # pydantic-settings configuration
│   ├── exceptions.py        # application-level exceptions
│   ├── api/
│   │   └── transcript.py    # POST /api/v1/transcribe
│   ├── schemas/
│   │   └── transcript.py    # request / response models
│   ├── services/
│   │   ├── youtube.py       # yt-dlp download service
│   │   └── transcription.py # faster-whisper service
│   └── utils/
│       └── youtube.py       # URL parsing / video ID extraction
├── temp/                    # temporary audio files (auto-cleaned)
├── tests/
│   ├── test_health.py
│   └── test_youtube.py
├── run.py
├── requirements.txt
├── .env.example
└── README.md
```

## Known limitations

- The MVP processes requests synchronously; very long videos take a long time.
- The Whisper model is downloaded once on first startup (hundreds of MB for `small`).
- FFmpeg must be installed separately on the host.
- Only local processing is implemented. Redis/Celery/PostgreSQL worker
  architecture can be layered on top of the existing `services` layer later.
