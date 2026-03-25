from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AudioPaths(BaseModel):
    full_audio: str
    turn_audios: list[str] = Field(default_factory=list)


class AudioLabels(BaseModel):
    background_noise_level: Literal["none", "low", "medium", "high"] = "none"
    channel_type: Literal["clinic_mic", "phone", "telehealth_mic"] = "clinic_mic"
    speaker_overlap: bool = False


class AudioManifest(BaseModel):
    audio_ready: bool = False
    output_format: Literal["wav", "mp3", "flac"] = "wav"
    sample_rate: int = Field(default=16000, ge=8000)
    channels: int = Field(default=1, ge=1)
    speaker_mode: Literal["single_speaker", "multi_speaker"] = "multi_speaker"
    paths: AudioPaths
    audio_labels: AudioLabels = Field(default_factory=AudioLabels)
