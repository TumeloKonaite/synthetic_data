from __future__ import annotations

import json

from synthetic_consults.generators.base import StructuredGenerator
from synthetic_consults.generators.conversation_generator import ConversationBundle
from synthetic_consults.models.clinical_outputs import QualityLabels
from synthetic_consults.models.conversation import ConversationTurn
from synthetic_consults.models.scenario import Scenario


def build_revision_system_prompt() -> str:
    return """
You revise synthetic doctor-patient consultations.

Rules:
- Preserve the underlying scenario.
- Improve realism, coherence, safety, and completeness.
- Keep the conversation natural and audio-friendly.
- Ensure the doctor gathers history logically and closes with a clear plan.
- Return only data matching the schema.
""".strip()


def build_revision_user_prompt(
    scenario: Scenario,
    conversation: list[ConversationTurn],
    quality_labels: QualityLabels,
) -> str:
    scenario_json = json.dumps(scenario.model_dump(mode="json"), ensure_ascii=False, indent=2)
    conversation_json = json.dumps(
        [turn.model_dump(mode="json") for turn in conversation],
        ensure_ascii=False,
        indent=2,
    )
    quality_json = json.dumps(quality_labels.model_dump(mode="json"), ensure_ascii=False, indent=2)

    return f"""
Revise this synthetic consultation using the quality review.

Scenario:
{scenario_json}

Original conversation:
{conversation_json}

Quality review:
{quality_json}

Requirements:
- Keep 10 to 24 turns.
- Improve weak or robotic utterances.
- Fix unsafe or unsupported content.
- Preserve the case details while making the dialogue more natural.
""".strip()


class ConversationReviser:
    def __init__(self, llm: StructuredGenerator[ConversationBundle]) -> None:
        self.llm = llm

    def generate(
        self,
        *,
        scenario: Scenario,
        conversation: list[ConversationTurn],
        quality_labels: QualityLabels,
    ) -> list[ConversationTurn]:
        bundle = self.llm.generate(
            system_prompt=build_revision_system_prompt(),
            user_prompt=build_revision_user_prompt(
                scenario=scenario,
                conversation=conversation,
                quality_labels=quality_labels,
            ),
            response_model=ConversationBundle,
            temperature=0.6,
        )
        return bundle.conversation