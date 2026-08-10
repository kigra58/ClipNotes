"""Pydantic models for the text-to-speech API."""

from pydantic import BaseModel, Field


class TextInput(BaseModel):
    """A body of text to analyze or speak."""

    text: str = Field(..., min_length=1, description="The text to process.")


class AnalyzeResponse(BaseModel):
    """Statistics computed from the input text."""

    characters: int = Field(..., description="Total number of characters.")
    words: int = Field(..., description="Total number of whitespace-separated words.")
    sentences: int = Field(..., description="Estimated number of sentences.")
    estimated_duration_seconds: float = Field(
        ..., description="Estimated reading time at 200 wpm."
    )
