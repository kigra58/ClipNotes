"""Application configuration loaded from environment variables via pydantic-settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration for the application, sourced from the .env file."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="VidNotes", alias="APP_NAME")
    app_version: str = Field(default="2.0.1", alias="APP_VERSION")

    # Auth configuration.
    jwt_secret: str = Field(default="dev-secret-change-me", alias="JWT_SECRET")
    jwt_expire_minutes: int = Field(default=1440, alias="JWT_EXPIRE_MINUTES")
    cookie_name: str = Field(default="access_token", alias="COOKIE_NAME")

    # Public base URL used to build absolute links (e.g. email verification).
    app_base_url: str = Field(default="http://localhost:8000", alias="APP_BASE_URL")

    # Email verification via SMTP (Hostinger: host smtp.hostinger.com).
    smtp_host: str = Field(default="", alias="SMTP_HOST")
    smtp_port: int = Field(default=465, alias="SMTP_PORT")
    smtp_username: str = Field(default="", alias="SMTP_USERNAME")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_from_email: str = Field(default="", alias="SMTP_FROM_EMAIL")
    smtp_from_name: str = Field(default="", alias="SMTP_FROM_NAME")
    smtp_use_ssl: bool = Field(default=True, alias="SMTP_USE_SSL")
    email_verify_token_minutes: int = Field(default=1440, alias="EMAIL_VERIFY_TOKEN_MINUTES")
    password_reset_token_minutes: int = Field(default=60, alias="PASSWORD_RESET_TOKEN_MINUTES")

    whisper_model: str = Field(default="small", alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")

    temp_dir: Path = Field(default=BASE_DIR / "temp", alias="TEMP_DIR")

    max_video_duration: float = Field(default=7200.0, alias="MAX_VIDEO_DURATION")

    # Text-to-speech (Piper) configuration.
    tts_voice_model: Path = Field(
        default=BASE_DIR / "models" / "tts" / "en_US-lessac-medium.onnx",
        alias="TTS_VOICE_MODEL",
    )
    tts_voices_dir: Path = Field(
        default=BASE_DIR / "models" / "tts",
        alias="TTS_VOICES_DIR",
    )
    tts_max_chars: int = Field(default=10000, alias="TTS_MAX_CHARS")
    tts_synthesis_timeout_seconds: int = Field(default=300, alias="TTS_SYNTHESIS_TIMEOUT_SECONDS")
    tts_cache_dir: Path = Field(default=BASE_DIR / "temp" / "tts_cache", alias="TTS_CACHE_DIR")

    # Comma-separated list of allowed origins. Empty disables CORS.
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # RAG / chat configuration.
    database_path: Path = Field(default=BASE_DIR / "transcripts.db", alias="DATABASE_PATH")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_chunk_chars: int = Field(default=600, alias="RAG_CHUNK_CHARS")
    rag_chunk_overlap: int = Field(default=60, alias="RAG_CHUNK_OVERLAP")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_MODEL")
    gemini_max_tokens: int = Field(default=1024, alias="GEMINI_MAX_TOKENS")

    @property
    def cors_origin_list(self) -> list[str]:
        """Return the configured CORS origins as a list."""
        if self.cors_origins.strip() == "":
            return []
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
