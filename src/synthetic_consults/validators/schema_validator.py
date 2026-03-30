from __future__ import annotations

from synthetic_consults.models.consultation_record import ConsultationRecord


def validate_schema(data: dict) -> ConsultationRecord:
    return ConsultationRecord.model_validate(data)
