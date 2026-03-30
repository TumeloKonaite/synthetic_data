
from pydantic import BaseModel
from typing import Literal

class TTSTurn(BaseModel):
    turn_id: int
    speaker: Literal["doctor", "patient"]
    text: str
    voice_id: str
    style: str = "neutral"
    pace: str = "normal"
    pause_after_sec: float = 0.6

class TTSScript(BaseModel):
    consultation_id: str
    locale: str = "en-ZA"
    model: str = "gpt-4o-mini-tts"
    format: str = "wav"
    sample_rate: int = 16000
    turns: list[TTSTurn]
    synthetic_version: str
    tts_config_version: str


class TurnAudioArtifact(BaseModel):
    turn_id: int
    speaker: str
    voice_id: str
    text: str
    path: str
    format: str = "wav"


class FullAudioArtifact(BaseModel):
    consultation_id: str
    path: str
    turn_paths: list[str]
    format: str = "wav"


class AudioManifest(BaseModel):
    consultation_id: str
    canonical_record_path: str
    tts_script_path: str
    turn_audio_paths: list[str]
    full_audio_path: str
    transcript_reference_path: str
    model: str
    synthetic_version: str
    tts_config_version: str
    audio_pipeline_version: str