from __future__ import annotations

from synthetic_consults.models.consultation_record import ConsultationRecord


class DialogueValidationError(ValueError):
    pass


def validate_dialogue(record: ConsultationRecord) -> None:
    turns = record.conversation

    if len(turns) < 10:
        raise DialogueValidationError("Conversation must contain at least 10 turns.")

    if turns[0].speaker != "patient":
        raise DialogueValidationError("Conversation should start with the patient.")

    doctor_turns = [t for t in turns if t.speaker == "doctor"]
    patient_turns = [t for t in turns if t.speaker == "patient"]

    if not doctor_turns or not patient_turns:
        raise DialogueValidationError("Conversation must include both patient and doctor.")

    if not any(
        "plan" in t.intent or "assessment" in t.intent or "close" in t.intent for t in doctor_turns
    ):
        raise DialogueValidationError(
            "Conversation must include assessment/plan/closing from doctor."
        )

    for turn in turns:
        if len(turn.utterance.split()) > 60:
            raise DialogueValidationError(
                f"Turn {turn.turn_id} is too long for natural spoken dialogue."
            )
