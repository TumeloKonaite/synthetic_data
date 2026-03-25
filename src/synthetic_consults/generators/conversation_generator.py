from __future__ import annotations

import json

from pydantic import BaseModel, Field

from synthetic_consults.generators.base import StructuredGenerator
from synthetic_consults.models.conversation import ConversationTurn
from synthetic_consults.models.scenario import Scenario


class ConversationBundle(BaseModel):
    conversation: list[ConversationTurn] = Field(default_factory=list)


def build_conversation_system_prompt() -> str:
    return """
You generate synthetic doctor-patient consultation dialogues.

Rules:
- Produce a natural multi-turn conversation.
- The patient should sound human: sometimes vague, concerned, uncertain, or brief.
- The doctor should sound professional, empathetic, and structured.
- The doctor should gather history before concluding.
- Include appropriate red-flag screening when relevant.
- The final doctor turn must include a cautious impression and a next-step plan.
- Keep utterances suitable for later text-to-speech.
- Return only data matching the schema.
""".strip()


def build_conversation_user_prompt(scenario: Scenario) -> str:
    scenario_json = json.dumps(scenario.model_dump(mode="json"), ensure_ascii=False, indent=2)

    return f"""
Given the scenario below, generate a realistic synthetic consultation.

Constraints:
- Between 10 and 24 turns.
- Natural turn-taking between patient and doctor.
- Most utterances should be short enough to sound natural when spoken aloud.
- The first turn should usually come from the patient.
- Use the voice preferences implied by the scenario audio profiles.
- The final doctor turn should summarize likely impression, next steps, and follow-up/safety-net advice.

Scenario:
{scenario_json}
""".strip()


class ConversationGenerator:
    def __init__(self, llm: StructuredGenerator[ConversationBundle]) -> None:
        self.llm = llm

    def generate(self, scenario: Scenario) -> list[ConversationTurn]:
        bundle = self.llm.generate(
            system_prompt=build_conversation_system_prompt(),
            user_prompt=build_conversation_user_prompt(scenario),
            response_model=ConversationBundle,
            temperature=0.8,
        )
        return bundle.conversation