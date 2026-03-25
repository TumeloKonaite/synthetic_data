from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    turn_id: int = Field(..., ge=1)
    speaker: Literal["patient", "doctor"]
    utterance: str = Field(..., min_length=1)
    intent: str = Field(..., min_length=1)


class TTSScriptTurn(BaseModel):
    turn_id: int = Field(..., ge=1)
    speaker: Literal["patient", "doctor"]
    text: str = Field(..., min_length=1)
    voice_id: str = Field(..., min_length=1)
    style: str = "neutral"
    pace: Literal["slow", "normal", "fast"] = "normal"
    pause_after_sec: float = Field(default=0.5, ge=0)
