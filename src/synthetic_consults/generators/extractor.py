from __future__ import annotations

import json

from synthetic_consults.generators.base import StructuredGenerator
from synthetic_consults.models.clinical_outputs import ClinicalOutputs
from synthetic_consults.models.conversation import ConversationTurn
from synthetic_consults.models.scenario import Scenario


def build_extractor_system_prompt() -> str:
    return """
You extract structured clinical workflow outputs from a synthetic consultation.

Rules:
- Stay grounded in the conversation and scenario.
- Do not invent physical exam findings or test results unless stated.
- Use cautious clinical wording when needed.
- Produce concise but useful summaries.
- Return only data matching the schema.
""".strip()


def build_extractor_user_prompt(
    scenario: Scenario,
    conversation: list[ConversationTurn],
) -> str:
    scenario_json = json.dumps(scenario.model_dump(mode="json"), ensure_ascii=False, indent=2)
    conversation_json = json.dumps(
        [turn.model_dump(mode="json") for turn in conversation],
        ensure_ascii=False,
        indent=2,
    )

    return f"""
Extract structured outputs for this synthetic consultation.

Scenario:
{scenario_json}

Conversation:
{conversation_json}

Requirements:
- Fill clinical_summary, structured_outputs, and derived_artifacts.
- Keep outputs useful for downstream summarization and patient communication tasks.
- Do not overstate diagnostic certainty.
""".strip()


class ClinicalExtractor:
    def __init__(self, llm: StructuredGenerator[ClinicalOutputs]) -> None:
        self.llm = llm

    def generate(
        self,
        *,
        scenario: Scenario,
        conversation: list[ConversationTurn],
    ) -> ClinicalOutputs:
        return self.llm.generate(
            system_prompt=build_extractor_system_prompt(),
            user_prompt=build_extractor_user_prompt(scenario, conversation),
            response_model=ClinicalOutputs,
            temperature=0.3,
        )