from __future__ import annotations

from synthetic_consults.models.conversation import AudioPrefs, ConversationTurn
from synthetic_consults.models.scenario import Scenario


def generate_stub_conversation(scenario: Scenario) -> list[ConversationTurn]:
    patient_voice = scenario.audio_profile.patient_voice_profile
    doctor_voice = scenario.audio_profile.doctor_voice_profile

    turns = [
        ("patient", "Hi doctor, I've had a cough for about ten days now.", "present_complaint"),
        ("doctor", "Sorry to hear that. Is it dry or are you coughing up mucus?", "clarify_symptom"),
        ("patient", "Mostly dry, but sometimes a little mucus in the morning.", "provide_history"),
        ("doctor", "Have you had fever, chest pain, or shortness of breath?", "screen_red_flags"),
        ("patient", "No fever or chest pain. I do get a bit wheezy at night.", "provide_history"),
        ("doctor", "Do you have any history of asthma or allergies?", "ask_history"),
        ("patient", "Yes, I have asthma and use an inhaler sometimes.", "provide_history"),
        ("doctor", "Have you needed the inhaler more often than usual?", "discuss_medication"),
        ("patient", "Yes, a little more this past week.", "provide_history"),
        ("doctor", "This could be a viral cough irritating your asthma. Keep using your inhaler as prescribed, rest, and come back if your breathing worsens or you develop fever.", "summarize_assessment"),
    ]

    conversation: list[ConversationTurn] = []
    for idx, (speaker, utterance, intent) in enumerate(turns, start=1):
        voice_id = patient_voice if speaker == "patient" else doctor_voice
        style = "concerned" if speaker == "patient" else "calm_professional"

        conversation.append(
            ConversationTurn(
                turn_id=idx,
                speaker=speaker,
                utterance=utterance,
                intent=intent,
                audio_prefs=AudioPrefs(
                    voice_id=voice_id,
                    speaking_style=style,
                    pace="normal",
                ),
                pause_after_sec=0.6,
            )
        )

    return conversation