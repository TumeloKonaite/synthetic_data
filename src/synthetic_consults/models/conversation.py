from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PatientPersona(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: Literal["male", "female", "other"]
    occupation: str
    language_style: Literal["plain", "detailed", "anxious", "reserved"]
    health_literacy: Literal["low", "medium", "high"]
    emotional_state: str


class ClinicalContext(BaseModel):
    chief_complaint: str
    symptom_duration: str
    known_conditions: list[str] = Field(default_factory=list)
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    social_history: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    red_flags_expected: bool = False


class AudioProfile(BaseModel):
    environment: Literal["quiet_clinic", "telehealth", "hospital_room", "phone_call"]
    background_noise_level: Literal["none", "low", "medium", "high"]
    channel_type: Literal["clinic_mic", "phone", "telehealth_mic"]
    doctor_voice_profile: str
    patient_voice_profile: str


class Scenario(BaseModel):
    consultation_type: Literal[
        "routine", "follow_up", "urgent", "chronic_review", "medication_review"
    ]
    specialty: str
    care_setting: Literal["clinic", "telehealth", "hospital_outpatient"]
    urgency: Literal["low", "medium", "high"]
    patient_persona: PatientPersona
    clinical_context: ClinicalContext
    audio_profile: AudioProfile