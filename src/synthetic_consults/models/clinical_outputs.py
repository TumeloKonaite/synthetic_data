from __future__ import annotations

from pydantic import BaseModel, Field


class MedicationDiscussed(BaseModel):
    name: str
    action: str
    reason: str


class ClinicalSummary(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class StructuredOutputs(BaseModel):
    primary_impression: str
    differential_diagnoses: list[str] = Field(default_factory=list)
    doctor_actions: list[str] = Field(default_factory=list)
    patient_next_steps: list[str] = Field(default_factory=list)
    follow_up_required: bool = False
    follow_up_timing: str = ""
    red_flags_identified: list[str] = Field(default_factory=list)
    tests_ordered: list[str] = Field(default_factory=list)
    medications_discussed: list[MedicationDiscussed] = Field(default_factory=list)
    referral_needed: bool = False


class DerivedArtifacts(BaseModel):
    doctor_summary: str
    patient_friendly_summary: str
    draft_email_to_patient: str


class ClinicalOutputs(BaseModel):
    clinical_summary: ClinicalSummary
    structured_outputs: StructuredOutputs
    derived_artifacts: DerivedArtifacts


class QualityLabels(BaseModel):
    realism_score: float = 0.0
    coherence_score: float = 0.0
    safety_score: float = 0.0
    completeness_score: float = 0.0
    contains_follow_up: bool = False
    contains_red_flag_screening: bool = False
    passed_qc: bool = False
