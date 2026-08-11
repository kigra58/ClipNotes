# VidNotes

A backend API and web UI that converts a YouTube video into a timestamped transcript, stores it in SQLite, builds a local RAG (retrieval-augmented generation) index over it, and lets you ask questions about the video from a streaming chat.

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
- **Persistent storage** of transcripts and segments in SQLite
- **RAG pipeline**: local embeddings (`sentence-transformers`) over timestamped chunks with cosine-similarity retrieval
- **Streaming chat** over WebSocket (`/ws/chat/{id}`) powered by the Gemini API
- **Jinja2 web UI**: home, transcript and chat pages

## Requirements

- Python 3.12+
- FFmpeg (installed separately and available in `PATH`)

> FFmpeg is not a Python package. On Windows you can install it via
> `winget install ffmpeg` or download it from <https://ffmpeg.org/download.html>.
> On Linux/macOS use your package manager (e.g. `sudo apt install ffmpeg`).

- A [Google Gemini API key](https://aistudio.google.com/apikey) (for the chat feature)

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
| `APP_NAME`             | Application display name                       | `VidNotes` |
| `APP_VERSION`          | Application version                            | `1.0.0` |
| `WHISPER_MODEL`        | Whisper model size (tiny/base/small/medium/large) | `small` |
| `WHISPER_DEVICE`       | Inference device (`cpu` or `cuda`)             | `cpu` |
| `WHISPER_COMPUTE_TYPE` | Model precision (`int8`, `float16`, ...)       | `int8` |
| `TEMP_DIR`             | Directory for temporary audio files            | `temp` |
| `MAX_VIDEO_DURATION`   | Max video length in seconds (default 2 hours)  | `7200` |
| `CORS_ORIGINS`         | Comma-separated allowed origins (`*` for all)  | `*` |
| `DATABASE_PATH`        | SQLite database file for stored transcripts    | `transcripts.db` |
| `EMBEDDING_MODEL`      | Sentence-Transformers model for embeddings     | `all-MiniLM-L6-v2` |
| `RAG_TOP_K`            | Chunks retrieved per chat question             | `5` |
| `RAG_CHUNK_CHARS`      | Approximate chunk size in characters           | `600` |
| `RAG_CHUNK_OVERLAP`    | Overlap between adjacent chunks in characters  | `60` |
| `GEMINI_API_KEY`       | Google Gemini API key (enables chat)           | *(empty)* |
| `GEMINI_MODEL`         | Gemini model name                              | `gemini-3.5-flash` |
| `GEMINI_MAX_TOKENS`    | Max output tokens per chat answer              | `1024` |

The Whisper model and the embedding model are downloaded on first use and kept
in memory for the lifetime of the process.

## Start the server

```bash
python run.py
```

This is equivalent to:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Web UI

| Page                    | Path                  | Description                                    |
|-------------------------|-----------------------|------------------------------------------------|
| Home                    | `/`                   | Transcribe a video, list saved transcripts     |
| Transcript              | `/transcripts/{id}`   | Read a transcript and its segments             |
| Chat                    | `/chat/{id}`          | Ask questions about the transcript (streams)   |

Open <http://localhost:8000> in your browser. Paste a YouTube URL, wait for the
transcription to finish, then open the chat page to ask questions. Answers are
streamed token-by-token over a WebSocket and cite the transcript timestamps used
as sources.

## API endpoints

| Method | Path                | Description                          |
|--------|---------------------|--------------------------------------|
| `GET`  | `/health`           | Health check                         |
| `POST` | `/api/v1/transcribe`| Transcribe a YouTube video           |
| `GET`  | `/docs`             | Swagger UI                           |
| `GET`  | `/redoc`            | ReDoc                                |
| `WS`   | `/ws/chat/{id}`     | Streaming chat for a transcript      |

### WebSocket chat protocol

Client sends:

```json
{"message": "What types of tools are discussed?"}
```

Server replies with JSON events:

| Type      | Data                              | Meaning                              |
|-----------|-----------------------------------|--------------------------------------|
| `sources` | list of `{start, end, text, score}` | Retrieved chunks used as context   |
| `token`   | string                            | A fragment of the generated answer   |
| `done`    | `""`                              | Answer complete                      |
| `error`   | string                            | An error message                     |

## Example request

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

The response includes a `transcript_id` that can be used with the transcript and
chat pages.

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
error paths. Real YouTube downloads, Whisper model downloads and embedding
model downloads are not triggered during tests (services are mocked).

## Project structure

```text
youtube-transcript-api/
│
├── app/
│   ├── main.py              # FastAPI app, lifespan, exception handlers
│   ├── config.py            # pydantic-settings configuration
│   ├── exceptions.py        # application-level exceptions
│   ├── api/
│   │   ├── transcript.py    # POST /api/v1/transcribe
│   │   └── chat.py          # web pages + WebSocket chat
│   ├── schemas/
│   │   └── transcript.py    # request / response models
│   ├── services/
│   │   ├── youtube.py       # yt-dlp download service
│   │   ├── transcription.py # faster-whisper service
│   │   ├── database.py      # SQLite persistence (transcripts, chunks)
│   │   ├── embeddings.py    # sentence-transformers embeddings
│   │   ├── rag.py           # chunking + cosine-similarity retrieval
│   │   ├── gemini.py        # streaming Gemini API client
│   │   └── chat.py          # RAG + Gemini orchestration
│   ├── templates/           # Jinja2 pages (base, home, transcript, chat)
│   ├── static/              # CSS + chat/transcribe JS
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
- The embedding model is downloaded once on first startup (~90 MB).
- FFmpeg must be installed separately on the host.
- Chat requires a `GEMINI_API_KEY`.
- Only local processing is implemented. Redis/Celery/PostgreSQL worker
  architecture can be layered on top of the existing `services` layer later.
