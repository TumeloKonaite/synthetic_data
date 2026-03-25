from __future__ import annotations

from synthetic_consults.models.clinical_outputs import (
    ClinicalOutputs,
    ClinicalSummary,
    DerivedArtifacts,
    StructuredOutputs,
)


def extract_stub_outputs() -> ClinicalOutputs:
    return ClinicalOutputs(
        clinical_summary=ClinicalSummary(
            subjective="Patient reports 10-day cough with mild nighttime wheeze and asthma history.",
            objective="No physical examination available in synthetic workflow.",
            assessment="Likely viral cough with mild asthma irritation.",
            plan="Supportive care, continue inhaler, return if worsening.",
        ),
        structured_outputs=StructuredOutputs(
            primary_impression="viral_cough_with_mild_asthma_irritation",
            differential_diagnoses=["post_viral_cough", "mild_asthma_exacerbation"],
            doctor_actions=[
                "clarified cough type",
                "screened for red flags",
                "reviewed asthma history",
            ],
            patient_next_steps=[
                "continue inhaler as prescribed",
                "monitor symptoms",
                "seek review if breathing worsens",
            ],
            follow_up_required=True,
            follow_up_timing="3-5 days if not improving",
            red_flags_identified=[],
            tests_ordered=[],
            medications_discussed=[],
            referral_needed=False,
        ),
        derived_artifacts=DerivedArtifacts(
            doctor_summary="Patient with 10-day cough and mild wheeze, likely viral illness with asthma irritation.",
            patient_friendly_summary="Your cough is likely from a viral illness and may also be irritating your asthma.",
            draft_email_to_patient="Dear Patient, thank you for today's consultation. Please continue your inhaler and seek care if symptoms worsen.",
        ),
    )