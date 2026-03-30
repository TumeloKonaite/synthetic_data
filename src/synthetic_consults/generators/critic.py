from __future__ import annotations

import json

from synthetic_consults.generators.base import StructuredGenerator
from synthetic_consults.models.clinical_outputs import QualityLabels
from synthetic_consults.models.conversation import ConversationTurn
from synthetic_consults.models.scenario import Scenario


def build_critic_system_prompt() -> str:
    return """
You review synthetic doctor-patient consultations for dataset quality.

Score the sample on:
- realism
- coherence
- safety
- completeness

Rules:
- Be strict but fair.
- Penalize robotic dialogue, unsupported claims, unsafe advice, weak plans, or poor clinical flow.
- Return only data matching the schema.
""".strip()


def build_critic_user_prompt(
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
Review this synthetic consultation and score it.

Scenario:
{scenario_json}

Conversation:
{conversation_json}

Guidance:
- Set passed_qc=true only if the dialogue is realistic, coherent, safe, and complete enough for dataset use.
- contains_follow_up should be true if the consultation has a clear follow-up or return-plan.
- contains_red_flag_screening should be true if the doctor screened for warning signs when relevant.
""".strip()


class ConversationCritic:
    def __init__(self, llm: StructuredGenerator[QualityLabels]) -> None:
        self.llm = llm

    def generate(
        self,
        *,
        scenario: Scenario,
        conversation: list[ConversationTurn],
    ) -> QualityLabels:
        return self.llm.generate(
            system_prompt=build_critic_system_prompt(),
            user_prompt=build_critic_user_prompt(scenario, conversation),
            response_model=QualityLabels,
            temperature=0.2,
        )
