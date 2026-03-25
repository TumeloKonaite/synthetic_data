from __future__ import annotations

from synthetic_consults.models.clinical_outputs import QualityLabels
from synthetic_consults.models.consultation_record import ConsultationRecord


def score_record(record: ConsultationRecord) -> QualityLabels:
    contains_follow_up = record.clinical_outputs.structured_outputs.follow_up_required
    contains_red_flag_screening = any(
        turn.intent == "screen_red_flags" for turn in record.conversation
    )

    return QualityLabels(
        realism_score=4.2,
        coherence_score=4.4,
        safety_score=4.8,
        completeness_score=4.3,
        contains_follow_up=contains_follow_up,
        contains_red_flag_screening=contains_red_flag_screening,
        passed_qc=True,
    )