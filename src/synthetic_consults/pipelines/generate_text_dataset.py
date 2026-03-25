from __future__ import annotations

from datetime import datetime, UTC

from synthetic_consults.generators.critic import score_record
from synthetic_consults.generators.conversation_generator import generate_stub_conversation
from synthetic_consults.generators.extractor import extract_stub_outputs
from synthetic_consults.generators.scenario_generator import generate_stub_scenario
from synthetic_consults.audio.tts_script_builder import build_tts_script
from synthetic_consults.models.audio import AudioLabels, AudioManifest, AudioPaths
from synthetic_consults.models.consultation_record import (
    ConsultationRecord,
    GeneratorInfo,
    LocaleInfo,
)
from synthetic_consults.models.transcript import TranscriptArtifact, TranscriptReference
from synthetic_consults.validators.dialogue_validator import validate_dialogue


def build_record(conversation_id: str = "consult_000001") -> ConsultationRecord:
    scenario = generate_stub_scenario()
    conversation = generate_stub_conversation(scenario)
    outputs = extract_stub_outputs()
    tts_script = build_tts_script(scenario, conversation)

    record = ConsultationRecord(
        conversation_id=conversation_id,
        created_at=datetime.now(UTC),
        generator=GeneratorInfo(
            scenario_model="stub",
            conversation_model="stub",
            extraction_model="stub",
            tts_provider="stub",
            tts_config_version="v1",
        ),
        locale=LocaleInfo(
            language="en",
            region="ZA",
            accent_target="South African English",
        ),
        scenario=scenario,
        conversation=conversation,
        tts_script=tts_script,
        audio_manifest=AudioManifest(
            audio_ready=False,
            output_format="wav",
            sample_rate=16000,
            channels=1,
            speaker_mode="multi_speaker",
            paths=AudioPaths(
                full_audio=f"data/audio/{conversation_id}/full.wav",
                turn_audios=[
                    f"data/audio/{conversation_id}/turn_{turn.turn_id:03d}.wav"
                    for turn in conversation
                ],
            ),
            audio_labels=AudioLabels(),
        ),
        transcript_reference=TranscriptReference(
            gold_verbatim=TranscriptArtifact(
                path=f"data/transcripts/gold/{conversation_id}_verbatim.json",
                text=" ".join(f"{t.speaker.capitalize()}: {t.utterance}" for t in conversation),
            ),
            gold_normalized=TranscriptArtifact(
                path=f"data/transcripts/gold/{conversation_id}_normalized.json",
                text=outputs.clinical_summary.subjective,
            ),
        ),
        clinical_outputs=outputs,
        quality_labels=score_record,
        tags=["synthetic", "doctor-patient", "audio-ready"],
    )

    validate_dialogue(record)
    record.quality_labels = score_record(record)
    return record