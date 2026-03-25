from __future__ import annotations

from synthetic_consults.models.conversation import ConversationTurn, TTSScriptTurn
from synthetic_consults.models.scenario import Scenario


def build_tts_script(
    scenario: Scenario,
    conversation: list[ConversationTurn],
) -> list[TTSScriptTurn]:
    patient_voice_id = scenario.audio_profile.patient_voice_profile
    doctor_voice_id = scenario.audio_profile.doctor_voice_profile

    return [
        TTSScriptTurn(
            turn_id=turn.turn_id,
            speaker=turn.speaker,
            text=turn.utterance,
            voice_id=patient_voice_id if turn.speaker == "patient" else doctor_voice_id,
            style="neutral",
            pace="normal",
            pause_after_sec=0.5,
        )
        for turn in conversation
    ]
