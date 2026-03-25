from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AudioPrefs(BaseModel):
    voice_id: str
    speaking_style: str
    pace: Literal["slow", "normal", "fast"] = "normal"


class ConversationTurn(BaseModel):
    turn_id: int = Field(..., ge=1)
    speaker: Literal["patient", "doctor"]
    utterance: str = Field(..., min_length=1)
    intent: str
    audio_prefs: AudioPrefs
    estimated_duration_sec: float | None = Field(default=None, ge=0)
    pause_after_sec: float | None = Field(default=0.5, ge=0)


class TTSScriptTurn(BaseModel):
    turn_id: int = Field(..., ge=1)
    speaker: Literal["patient", "doctor"]
    text: str
    voice_id: str
    style: str
    pace: Literal["slow", "normal", "fast"] = "normal"
    pause_after_sec: float = Field(default=0.5, ge=0)