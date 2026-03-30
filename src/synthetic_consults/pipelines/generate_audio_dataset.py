from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import yaml

from synthetic_consults.audio.audio_manifest import save_audio_manifest, save_json
from synthetic_consults.audio.providers.openai_tts import OpenAITTSProvider
from synthetic_consults.audio.stitcher import stitch_turn_audio
from synthetic_consults.audio.synthesizer import synthesize_turns
from synthetic_consults.models.consultation_record import ConsultationRecord
from synthetic_consults.models.tts import AudioManifest, TTSScript, TTSTurn
from synthetic_consults.validators.audio_validator import validate_record_for_audio

VALID_OPENAI_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
}

MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u00a6": "...",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate turn-level and full consultation audio from processed records."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--record",
        type=Path,
        help="Path to a single processed JSON or JSONL record file.",
    )
    source_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing processed JSONL or JSON record files.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern for record files when using --input-dir. Default: *.jsonl",
    )
    parser.add_argument(
        "--tts-config",
        type=Path,
        required=True,
        help="Path to tts.yaml config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root output directory for generated audio artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records to process.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failure.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra progress information.",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Keep only full.wav after stitching and remove intermediate turn audio files.",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"TTS config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"TTS config must be a mapping/dict: {path}")

    return data


def normalize_text(text: str) -> str:
    fixed = text
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        fixed = fixed.replace(bad, good)
    return fixed.strip()


def _iter_payloads_from_file(path: Path) -> Iterable[tuple[str, dict]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                yield f"{path}:{line_no}", json.loads(line)
        return

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        for idx, item in enumerate(payload, start=1):
            yield f"{path}:{idx}", item
        return

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported record payload in {path}")

    yield str(path), payload


def iter_record_payloads(
    record: Path | None,
    input_dir: Path | None,
    pattern: str,
    limit: int | None,
) -> Iterable[tuple[str, dict]]:
    remaining = limit

    if record is not None:
        if not record.exists():
            raise FileNotFoundError(f"Record file not found: {record}")
        for item in _iter_payloads_from_file(record):
            yield item
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    return
        return

    assert input_dir is not None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {input_dir}")

    paths = sorted(p for p in input_dir.rglob(pattern) if p.is_file())

    for path in paths:
        for item in _iter_payloads_from_file(path):
            yield item
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    return


def load_consultation_record(payload: dict) -> ConsultationRecord:
    return ConsultationRecord.model_validate(payload)


def _default_voice_for_speaker(speaker: str, tts_config: dict) -> str:
    voices = tts_config.get("voices", {})
    if speaker == "doctor":
        return voices.get("doctor_default", "alloy")
    return voices.get("patient_default", "verse")


def build_runtime_tts_script(record_obj: ConsultationRecord, tts_config: dict) -> TTSScript:
    source_turns = record_obj.tts_script or record_obj.conversation
    turns: list[TTSTurn] = []

    for turn in source_turns:
        raw_text = getattr(turn, "text", None) or getattr(turn, "utterance", "")
        voice_hint = getattr(turn, "voice_id", "") or ""
        chosen_voice = (
            voice_hint
            if voice_hint in VALID_OPENAI_VOICES
            else _default_voice_for_speaker(turn.speaker, tts_config)
        )

        style_parts: list[str] = []
        if voice_hint and voice_hint not in VALID_OPENAI_VOICES:
            style_parts.append(f"Voice profile: {normalize_text(voice_hint)}.")
        turn_style = getattr(turn, "style", None)
        if turn_style:
            style_parts.append(f"Style: {turn_style}.")
        pace = getattr(turn, "pace", "normal")

        turns.append(
            TTSTurn(
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                text=normalize_text(raw_text),
                voice_id=chosen_voice,
                style=" ".join(style_parts).strip() or "neutral",
                pace=pace,
                pause_after_sec=getattr(
                    turn,
                    "pause_after_sec",
                    tts_config.get("default_pause_after_sec", 0.6),
                ),
            )
        )

    locale = tts_config.get(
        "locale",
        f"{record_obj.locale.language}-{record_obj.locale.region}",
    )

    return TTSScript(
        consultation_id=record_obj.conversation_id,
        locale=locale,
        model=tts_config.get("model", "gpt-4o-mini-tts"),
        format=tts_config.get("format", "wav"),
        sample_rate=tts_config.get("sample_rate", 16000),
        turns=turns,
        synthetic_version=record_obj.synthetic_version,
        tts_config_version=tts_config.get(
            "tts_config_version",
            record_obj.generator.tts_config_version,
        ),
    )


def run_audio_pipeline(
    *,
    record_obj: ConsultationRecord,
    record_path: str,
    tts_config: dict,
    output_root: str,
    retain_turn_audio: bool = True,
) -> AudioManifest:
    validate_record_for_audio(record_obj)

    tts_script = build_runtime_tts_script(record_obj, tts_config)
    consultation_dir = Path(output_root) / tts_script.consultation_id
    consultation_dir.mkdir(parents=True, exist_ok=True)

    tts_script_path = consultation_dir / "tts_script.json"
    tts_script_path.write_text(tts_script.model_dump_json(indent=2), encoding="utf-8")

    provider = OpenAITTSProvider(model=tts_script.model)
    turn_artifacts = synthesize_turns(
        tts_script=tts_script,
        provider=provider,
        output_dir=str(consultation_dir),
    )
    full_audio = stitch_turn_audio(
        consultation_id=tts_script.consultation_id,
        turn_artifacts=turn_artifacts,
        tts_script=tts_script,
        output_dir=str(consultation_dir),
    )

    transcript_reference = {
        "consultation_id": tts_script.consultation_id,
        "source_record": record_path,
        "turns": [
            {"turn_id": turn.turn_id, "speaker": turn.speaker, "text": turn.text}
            for turn in tts_script.turns
        ],
        "full_transcript": " ".join(f"{turn.speaker}: {turn.text}" for turn in tts_script.turns),
    }

    transcript_reference_path = consultation_dir / "transcript_reference.json"
    save_json(transcript_reference, str(transcript_reference_path))

    turn_audio_paths = [artifact.path for artifact in turn_artifacts]
    if not retain_turn_audio:
        for path_str in turn_audio_paths:
            path = Path(path_str)
            if path.exists():
                path.unlink()
        turns_dir = consultation_dir / "turns"
        if turns_dir.exists():
            shutil.rmtree(turns_dir)
        turn_audio_paths = []

    manifest = AudioManifest(
        consultation_id=tts_script.consultation_id,
        canonical_record_path=record_path,
        tts_script_path=str(tts_script_path),
        turn_audio_paths=turn_audio_paths,
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


def main() -> int:
    args = parse_args()

    try:
        tts_config = load_yaml(args.tts_config)
    except Exception as exc:
        print(f"[ERROR] Failed to load TTS config: {exc}", file=sys.stderr)
        return 2

    try:
        record_entries = list(
            iter_record_payloads(
                record=args.record,
                input_dir=args.input_dir,
                pattern=args.pattern,
                limit=args.limit,
            )
        )
    except Exception as exc:
        print(f"[ERROR] Failed to resolve input records: {exc}", file=sys.stderr)
        return 2

    if not record_entries:
        print("[ERROR] No input record files found.", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    successes: list[dict] = []
    failures: list[dict] = []

    total = len(record_entries)
    print(f"[INFO] Found {total} record(s) to process.")

    for idx, (record_ref, payload) in enumerate(record_entries, start=1):
        try:
            if args.verbose:
                print(f"[INFO] ({idx}/{total}) Loading {record_ref}")

            record_obj = load_consultation_record(payload)
            manifest = run_audio_pipeline(
                record_obj=record_obj,
                record_path=record_ref,
                tts_config=tts_config,
                output_root=str(args.output_dir),
                retain_turn_audio=not args.full_only,
            )

            successes.append(
                {
                    "record_path": record_ref,
                    "consultation_id": manifest.consultation_id,
                    "audio_manifest_path": str(
                        Path(args.output_dir) / manifest.consultation_id / "audio_manifest.json"
                    ),
                    "full_audio_path": manifest.full_audio_path,
                }
            )

            print(
                f"[OK] ({idx}/{total}) {record_obj.conversation_id} -> {manifest.full_audio_path}"
            )

        except Exception as exc:
            failures.append({"record_path": record_ref, "error": str(exc)})
            print(f"[FAIL] ({idx}/{total}) {record_ref}: {exc}", file=sys.stderr)

            if args.fail_fast:
                break

    summary = {
        "total": total,
        "succeeded": len(successes),
        "failed": len(failures),
        "successes": successes,
        "failures": failures,
    }

    summary_path = args.output_dir / "audio_generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Wrote summary to {summary_path}")
    print(f"[INFO] Completed: {len(successes)} succeeded, {len(failures)} failed.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
