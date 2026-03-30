# src/pipelines/generate_audio_dataset.py
import json
from pathlib import Path

from src.audio.audio_manifest import save_audio_manifest, save_json
from src.audio.providers.openai_tts import OpenAITTSProvider
from src.audio.stitcher import stitch_turn_audio
from src.audio.synthesizer import synthesize_turns
from src.audio.tts_script_builder import build_tts_script
from src.audio.voice_mapper import assign_voices
from src.models.tts import AudioManifest
from src.validators.audio_validator import validate_record_for_audio


def load_record(record_path: str):
    with open(record_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_audio_pipeline(record_obj, record_path: str, tts_config: dict, output_root: str):
    validate_record_for_audio(record_obj)

    consultation_id = record_obj.consultation_id
    consultation_dir = Path(output_root) / consultation_id
    consultation_dir.mkdir(parents=True, exist_ok=True)

    voice_map = assign_voices(record_obj, tts_config)
    tts_script = build_tts_script(record_obj, voice_map, tts_config)

    tts_script_path = consultation_dir / "tts_script.json"
    tts_script_path.write_text(tts_script.model_dump_json(indent=2), encoding="utf-8")

    provider = OpenAITTSProvider(model=tts_config.get("model", "gpt-4o-mini-tts"))

    turn_artifacts = synthesize_turns(
        tts_script=tts_script,
        provider=provider,
        output_dir=str(consultation_dir),
    )

    full_audio = stitch_turn_audio(
        consultation_id=consultation_id,
        turn_artifacts=turn_artifacts,
        tts_script=tts_script,
        output_dir=str(consultation_dir),
    )

    transcript_reference = {
        "consultation_id": consultation_id,
        "turns": [
            {"turn_id": t.turn_id, "speaker": t.speaker, "text": t.text} for t in tts_script.turns
        ],
        "full_transcript": " ".join([f"{t.speaker}: {t.text}" for t in tts_script.turns]),
    }

    transcript_reference_path = consultation_dir / "transcript_reference.json"
    save_json(transcript_reference, str(transcript_reference_path))

    manifest = AudioManifest(
        consultation_id=consultation_id,
        canonical_record_path=record_path,
        tts_script_path=str(tts_script_path),
        turn_audio_paths=[a.path for a in turn_artifacts],
        full_audio_path=full_audio.path,
        transcript_reference_path=str(transcript_reference_path),
        model=tts_script.model,
        synthetic_version=tts_script.synthetic_version,
        tts_config_version=tts_script.tts_config_version,
        audio_pipeline_version="v1",
    )

    manifest_path = consultation_dir / "audio_manifest.json"
    save_audio_manifest(manifest, str(manifest_path))

    return manifest
