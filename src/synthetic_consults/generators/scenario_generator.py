from __future__ import annotations

from pydantic import BaseModel

from synthetic_consults.generators.base import StructuredGenerator
from synthetic_consults.models.scenario import Scenario


class ScenarioGenerationRequest(BaseModel):
    consultation_type: str = "routine"
    specialty: str = "general_practice"
    care_setting: str = "clinic"
    urgency: str = "low"
    language: str = "en"
    region: str = "ZA"
    accent_target: str = "South African English"


def build_scenario_system_prompt() -> str:
    return """
You generate synthetic medical consultation scenarios for dataset creation.

Rules:
- Produce fictional, non-identifiable scenarios only.
- Keep scenarios medically plausible.
- Prefer common outpatient and telehealth presentations.
- Avoid dangerous or reckless content.
- Do not include real patient names, addresses, hospitals, or identifiers.
- Ensure the patient sounds like a normal person, not a textbook.
- Return only data matching the schema.
""".strip()


def build_scenario_user_prompt(request: ScenarioGenerationRequest) -> str:
    return f"""
Generate one synthetic consultation scenario with these constraints:

- consultation_type: {request.consultation_type}
- specialty: {request.specialty}
- care_setting: {request.care_setting}
- urgency: {request.urgency}
- language: {request.language}
- region: {request.region}
- accent target: {request.accent_target}

Requirements:
- Include patient persona, clinical context, and audio profile.
- Keep it realistic for a doctor-patient consultation.
- Prefer common chief complaints like cough, headache, rash, fatigue, abdominal pain, medication review, or blood pressure follow-up.
- The audio profile should be suitable for later TTS generation.
""".strip()


class ScenarioGenerator:
    def __init__(self, llm: StructuredGenerator[Scenario]) -> None:
        self.llm = llm

    def generate(self, request: ScenarioGenerationRequest | None = None) -> Scenario:
        request = request or ScenarioGenerationRequest()
        return self.llm.generate(
            system_prompt=build_scenario_system_prompt(),
            user_prompt=build_scenario_user_prompt(request),
            response_model=Scenario,
            temperature=0.9,
        )
