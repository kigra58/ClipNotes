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

    app_name: str = Field(default="YouTube Transcript API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")

    whisper_model: str = Field(default="small", alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")

    temp_dir: Path = Field(default=BASE_DIR / "temp", alias="TEMP_DIR")

    max_video_duration: float = Field(default=7200.0, alias="MAX_VIDEO_DURATION")

    # Comma-separated list of allowed origins. Empty disables CORS.
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # RAG / chat configuration.
    database_path: Path = Field(default=BASE_DIR / "transcripts.db", alias="DATABASE_PATH")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_chunk_chars: int = Field(default=600, alias="RAG_CHUNK_CHARS")
    rag_chunk_overlap: int = Field(default=60, alias="RAG_CHUNK_OVERLAP")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    gemini_max_tokens: int = Field(default=1024, alias="GEMINI_MAX_TOKENS")

    @property
    def cors_origin_list(self) -> list[str]:
        """Return the configured CORS origins as a list."""
        if self.cors_origins.strip() == "":
            return []
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
