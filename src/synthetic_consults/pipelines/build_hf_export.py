from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from synthetic_consults.io.jsonl_writer import write_jsonl
from synthetic_consults.models.consultation_record import ConsultationRecord
from synthetic_consults.pipelines.generate_audio_dataset import iter_record_payloads

ExportMode = Literal["text", "audio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Hugging Face-ready export bundle from generated consultation data."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--record", type=Path, help="Path to a single JSON or JSONL record file.")
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
    parser.add_argument("--output-dir", type=Path, required=True, help="Export bundle output directory.")
    parser.add_argument(
        "--mode",
        choices=("text", "audio"),
        default="text",
        help="Export mode. 'audio' includes only records with completed audio artifacts.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("artifacts/audio"),
        help="Root directory containing generated audio artifacts. Default: artifacts/audio",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on records to inspect.")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the built export directory to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Target Hugging Face dataset repo id, for example org/dataset-name.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the remote dataset repo as private when --push is used.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Update synthetic consultation dataset export",
        help="Commit message to use when uploading to the Hub.",
    )
    return parser.parse_args()


def load_consultation_record(payload: dict[str, Any]) -> ConsultationRecord:
    return ConsultationRecord.model_validate(payload)


def _build_full_transcript(record_obj: ConsultationRecord) -> str:
    source_turns = record_obj.tts_script or record_obj.conversation
    return " ".join(
        f"{turn.speaker}: {getattr(turn, 'text', None) or getattr(turn, 'utterance', '')}"
        for turn in source_turns
    )


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _normalize_record_ref(record_ref: str) -> str:
    file_part, separator, line_part = record_ref.rpartition(":")
    if not separator or not line_part.isdigit():
        file_part = record_ref
        line_part = ""

    normalized_path = Path(file_part)
    normalized_file_part = (
        normalized_path.name if normalized_path.is_absolute() else normalized_path.as_posix()
    )
    return f"{normalized_file_part}:{line_part}" if line_part else normalized_file_part


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_audio_files(consultation_id: str, audio_root: Path) -> tuple[Path, Path, Path] | None:
    consultation_dir = audio_root / consultation_id
    full_audio_path = consultation_dir / "full.wav"
    audio_manifest_path = consultation_dir / "audio_manifest.json"
    transcript_reference_path = consultation_dir / "transcript_reference.json"

    if not (
        full_audio_path.exists()
        and audio_manifest_path.exists()
        and transcript_reference_path.exists()
    ):
        return None

    return full_audio_path, audio_manifest_path, transcript_reference_path


def _copy_audio_files(
    *,
    consultation_id: str,
    audio_files: tuple[Path, Path, Path],
    output_dir: Path,
) -> dict[str, Any]:
    full_audio_path, audio_manifest_path, transcript_reference_path = audio_files

    audio_dir = output_dir / "audio" / consultation_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied_full_audio = audio_dir / full_audio_path.name
    shutil.copy2(full_audio_path, copied_full_audio)

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    copied_manifest = manifests_dir / f"{consultation_id}.json"
    shutil.copy2(audio_manifest_path, copied_manifest)

    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    copied_transcript = transcripts_dir / f"{consultation_id}.json"
    shutil.copy2(transcript_reference_path, copied_transcript)

    return {
        "included": True,
        "format": copied_full_audio.suffix.lstrip(".").lower() or "wav",
        "full_audio": _relative_posix(copied_full_audio, output_dir),
        "audio_manifest": _relative_posix(copied_manifest, output_dir),
        "transcript_reference": _relative_posix(copied_transcript, output_dir),
    }


def _build_export_row(
    *,
    record_obj: ConsultationRecord,
    record_ref: str,
    audio_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "consultation_id": record_obj.conversation_id,
        "split": record_obj.split,
        "source": record_obj.source,
        "synthetic_version": record_obj.synthetic_version,
        "created_at": record_obj.created_at.isoformat(),
        "source_record": _normalize_record_ref(record_ref),
        "generator": record_obj.generator.model_dump(mode="json"),
        "locale": record_obj.locale.model_dump(mode="json"),
        "scenario": record_obj.scenario.model_dump(mode="json"),
        "conversation": [turn.model_dump(mode="json") for turn in record_obj.conversation],
        "tts_script": [turn.model_dump(mode="json") for turn in record_obj.tts_script],
        "transcript_reference": record_obj.transcript_reference.model_dump(mode="json"),
        "clinical_outputs": record_obj.clinical_outputs.model_dump(mode="json"),
        "quality_labels": record_obj.quality_labels.model_dump(mode="json"),
        "tags": list(record_obj.tags),
        "full_transcript": _build_full_transcript(record_obj),
        "audio_available": audio_entry is not None,
        "audio": audio_entry
        or {
            "included": False,
            "format": None,
            "full_audio": None,
            "audio_manifest": None,
            "transcript_reference": None,
        },
    }


def _reset_export_dir(output_dir: Path) -> None:
    managed_paths = [
        output_dir / "data",
        output_dir / "audio",
        output_dir / "manifests",
        output_dir / "transcripts",
        output_dir / "README.md",
        output_dir / "export_summary.json",
        output_dir / ".gitattributes",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in managed_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _write_dataset_card(
    *,
    output_dir: Path,
    mode: ExportMode,
    exported: int,
    skipped_missing_audio: int,
    splits: dict[str, int],
    repo_id: str | None,
) -> None:
    split_lines = "\n".join(f"- `{split}`: {count} record(s)" for split, count in sorted(splits.items()))
    content = (
        "# Synthetic Patient DR Data\n\n"
        "Hugging Face-ready export bundle for synthetic doctor-patient consultations.\n\n"
        "## Export Metadata\n\n"
        f"- Mode: `{mode}`\n"
        f"- Repo target: `{repo_id or 'Not specified'}`\n"
        f"- Exported records: {exported}\n"
        f"- Skipped due to missing audio: {skipped_missing_audio}\n\n"
        "## Splits\n\n"
        f"{split_lines or '- None'}\n\n"
        "## Files\n\n"
        "- `data/*.jsonl`: normalized dataset rows by split\n"
        "- `audio/`: copied full-consultation audio assets when exporting in audio mode\n"
        "- `manifests/`: copied generated audio manifests when exporting in audio mode\n"
        "- `transcripts/`: copied transcript reference files when exporting in audio mode\n"
        "- `export_summary.json`: machine-readable export summary\n"
    )
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def build_hf_export(
    *,
    record: Path | None,
    input_dir: Path | None,
    pattern: str,
    output_dir: Path,
    mode: ExportMode,
    audio_root: Path,
    limit: int | None = None,
    repo_id: str | None = None,
) -> dict[str, Any]:
    record_entries = list(
        iter_record_payloads(
            record=record,
            input_dir=input_dir,
            pattern=pattern,
            limit=limit,
        )
    )
    if not record_entries:
        raise ValueError("No input records found for export.")

    _reset_export_dir(output_dir)

    split_rows: dict[str, list[dict[str, Any]]] = {}
    skipped_missing_audio = 0

    for record_ref, payload in record_entries:
        record_obj = load_consultation_record(payload)

        audio_entry: dict[str, Any] | None = None
        if mode == "audio":
            audio_files = _resolve_audio_files(record_obj.conversation_id, audio_root)
            if audio_files is None:
                skipped_missing_audio += 1
                continue
            audio_entry = _copy_audio_files(
                consultation_id=record_obj.conversation_id,
                audio_files=audio_files,
                output_dir=output_dir,
            )

        row = _build_export_row(
            record_obj=record_obj,
            record_ref=record_ref,
            audio_entry=audio_entry,
        )
        split_rows.setdefault(record_obj.split, []).append(row)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}
    for split, rows in sorted(split_rows.items()):
        write_jsonl(data_dir / f"{split}.jsonl", rows)
        split_counts[split] = len(rows)

    exported = sum(split_counts.values())
    summary = {
        "mode": mode,
        "total_input_records": len(record_entries),
        "exported": exported,
        "skipped_missing_audio": skipped_missing_audio,
        "splits": split_counts,
        "repo_id": repo_id,
    }

    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_dataset_card(
        output_dir=output_dir,
        mode=mode,
        exported=exported,
        skipped_missing_audio=skipped_missing_audio,
        splits=split_counts,
        repo_id=repo_id,
    )
    if mode == "audio":
        (output_dir / ".gitattributes").write_text(
            "*.wav filter=lfs diff=lfs merge=lfs -text\n",
            encoding="utf-8",
        )

    return summary


def push_export_to_hub(
    *,
    output_dir: Path,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Update synthetic consultation dataset export",
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --push. Install it before uploading."
        ) from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )


def main() -> int:
    args = parse_args()

    if args.push and not args.repo_id:
        print("[ERROR] --repo-id is required when using --push.", file=sys.stderr)
        return 2

    try:
        summary = build_hf_export(
            record=args.record,
            input_dir=args.input_dir,
            pattern=args.pattern,
            output_dir=args.output_dir,
            mode=args.mode,
            audio_root=args.audio_root,
            limit=args.limit,
            repo_id=args.repo_id,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to build HF export: {exc}", file=sys.stderr)
        return 2

    print(
        "[INFO] Export complete: "
        f"{summary['exported']} exported, "
        f"{summary['skipped_missing_audio']} skipped for missing audio."
    )
    print(f"[INFO] Split counts: {summary['splits']}")
    print(f"[INFO] Wrote export bundle to {args.output_dir}")

    if not args.push:
        return 0

    try:
        push_export_to_hub(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to upload export bundle: {exc}", file=sys.stderr)
        return 2

    print(f"[INFO] Uploaded dataset bundle to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
