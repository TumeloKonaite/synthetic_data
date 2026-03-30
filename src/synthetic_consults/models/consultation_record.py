from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from synthetic_consults.models.audio import AudioManifest
from synthetic_consults.models.clinical_outputs import ClinicalOutputs, QualityLabels
from synthetic_consults.models.conversation import ConversationTurn, TTSScriptTurn
from synthetic_consults.models.scenario import Scenario
from synthetic_consults.models.transcript import TranscriptReference


class GeneratorInfo(BaseModel):
    scenario_model: str
    conversation_model: str
    extraction_model: str
    tts_provider: str
    tts_config_version: str


class LocaleInfo(BaseModel):
    language: str
    region: str
    accent_target: str


class ConsultationRecord(BaseModel):
    conversation_id: str
    split: str = "train"
    source: str = "synthetic"
    synthetic_version: str = "v1"
    created_at: datetime
    generator: GeneratorInfo
    locale: LocaleInfo
    scenario: Scenario
    conversation: list[ConversationTurn] = Field(default_factory=list)
    tts_script: list[TTSScriptTurn] = Field(default_factory=list)
    audio_manifest: AudioManifest
    transcript_reference: TranscriptReference
    clinical_outputs: ClinicalOutputs
    quality_labels: QualityLabels
    tags: list[str] = Field(default_factory=list)
