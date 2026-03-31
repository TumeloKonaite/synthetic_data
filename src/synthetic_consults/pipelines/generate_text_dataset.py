from __future__ import annotations

from datetime import UTC, datetime

from synthetic_consults.audio.tts_script_builder import build_tts_script
from synthetic_consults.generators import (
    ClinicalExtractor,
    ConversationCritic,
    ConversationGenerator,
    ConversationReviser,
    OpenAIChatGenerator,
    ScenarioGenerator,
)
from synthetic_consults.generators.conversation_generator import ConversationBundle
from synthetic_consults.models.audio import AudioLabels, AudioManifest, AudioPaths
from synthetic_consults.models.consultation_record import (
    ConsultationRecord,
    GeneratorInfo,
    LocaleInfo,
)
from synthetic_consults.models.scenario import Scenario
from synthetic_consults.models.transcript import TranscriptArtifact, TranscriptReference
from synthetic_consults.validators.dialogue_validator import validate_dialogue_turns


def build_record(conversation_id: str) -> ConsultationRecord:
    """
    End-to-end generation of one synthetic consultation record.
    """

    # --- LLM Clients ---
    scenario_llm = OpenAIChatGenerator[Scenario](model="gpt-5.4-mini")
    conversation_llm = OpenAIChatGenerator[ConversationBundle](model="gpt-5.4-mini")
    extractor_llm = OpenAIChatGenerator(model="gpt-5.4-mini")
    critic_llm = OpenAIChatGenerator(model="gpt-5.4-mini")

    # --- Generators ---
    scenario_generator = ScenarioGenerator(scenario_llm)
    conversation_generator = ConversationGenerator(conversation_llm)
    extractor = ClinicalExtractor(extractor_llm)
    critic = ConversationCritic(critic_llm)

    # --- 1. Scenario ---
    scenario = scenario_generator.generate()

    # --- 2. Conversation ---
    conversation = conversation_generator.generate(scenario)

    # --- 3. Validate early ---
    validate_dialogue_turns(conversation)

    # --- 4. Critic ---
    quality = critic.generate(
        scenario=scenario,
        conversation=conversation,
    )

    # --- 5. Revise if needed ---
    if not quality.passed_qc:
        reviser_llm = OpenAIChatGenerator[ConversationBundle](model="gpt-5.4-mini")
        reviser = ConversationReviser(reviser_llm)

        conversation = reviser.generate(
            scenario=scenario,
            conversation=conversation,
            quality_labels=quality,
        )

        quality = critic.generate(
            scenario=scenario,
            conversation=conversation,
        )

    # --- 6. Extract structured outputs ---
    clinical_outputs = extractor.generate(
        scenario=scenario,
        conversation=conversation,
    )

    # --- 7. Build TTS script ---
    tts_script = build_tts_script(scenario, conversation)

    # --- 8. Build record ---
    record = ConsultationRecord(
        conversation_id=conversation_id,
        created_at=datetime.now(UTC),
        generator=GeneratorInfo(
            scenario_model="gpt-5.4-mini",
            conversation_model="gpt-5.4-mini",
            extraction_model="gpt-5.4-mini",
            tts_provider="openai",
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
                    f"data/audio/{conversation_id}/turn_{t.turn_id:03d}.wav" for t in conversation
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
                text=clinical_outputs.clinical_summary.subjective,
            ),
        ),
        clinical_outputs=clinical_outputs,
        quality_labels=quality,
        tags=["synthetic", "doctor-patient", "audio-ready"],
    )

    return record
