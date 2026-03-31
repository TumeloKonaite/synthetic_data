from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from synthetic_consults.pipelines.build_hf_export import build_hf_export, push_export_to_hub


def _load_sample_payload() -> dict:
    first_line = Path("data/processed/train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    return json.loads(first_line)


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for payload in payloads:
            f.write(json.dumps(payload) + "\n")


def _write_complete_audio_artifacts(audio_root: Path, consultation_id: str) -> None:
    consultation_dir = audio_root / consultation_id
    consultation_dir.mkdir(parents=True, exist_ok=True)
    (consultation_dir / "full.wav").write_bytes(b"RIFF")
    (consultation_dir / "audio_manifest.json").write_text(
        json.dumps(
            {
                "consultation_id": consultation_id,
                "full_audio_path": f"artifacts/audio/{consultation_id}/full.wav",
                "transcript_reference_path": (
                    f"artifacts/audio/{consultation_id}/transcript_reference.json"
                ),
            }
        ),
        encoding="utf-8",
    )
    (consultation_dir / "transcript_reference.json").write_text(
        json.dumps(
            {
                "consultation_id": consultation_id,
                "turns": [{"turn_id": 1, "speaker": "patient", "text": "Hello"}],
                "full_transcript": "patient: Hello",
            }
        ),
        encoding="utf-8",
    )


def test_build_hf_text_export_writes_normalized_jsonl(tmp_path: Path) -> None:
    payload = _load_sample_payload()
    record_path = tmp_path / "records.jsonl"
    _write_jsonl(record_path, [payload])

    output_dir = tmp_path / "hf_export"
    summary = build_hf_export(
        record=record_path,
        input_dir=None,
        pattern="*.jsonl",
        output_dir=output_dir,
        mode="text",
        audio_root=tmp_path / "audio",
        repo_id="test-org/synthetic-consults",
    )

    assert summary["exported"] == 1
    assert summary["skipped_missing_audio"] == 0

    train_jsonl = output_dir / "data" / "train.jsonl"
    assert train_jsonl.exists()
    lines = train_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    exported_row = json.loads(lines[0])
    assert exported_row["consultation_id"] == payload["conversation_id"]
    assert exported_row["audio_available"] is False
    assert exported_row["audio"]["full_audio"] is None
    assert exported_row["source_record"] == "records.jsonl:1"

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\npretty_name: Synthetic Patient DR Data\n")
    assert "Mode: `text`" in readme
    assert "All consultations are synthetic" in readme
    assert "GitHub repository: https://github.com/TumeloKonaite/synthetic_data" in readme
    assert not (output_dir / "rows").exists()


def test_build_hf_audio_export_copies_completed_assets_only(tmp_path: Path) -> None:
    payload_complete = _load_sample_payload()
    payload_incomplete = _load_sample_payload()
    payload_incomplete["conversation_id"] = "consult_999999"

    record_path = tmp_path / "records.jsonl"
    _write_jsonl(record_path, [payload_complete, payload_incomplete])

    audio_root = tmp_path / "audio"
    _write_complete_audio_artifacts(audio_root, payload_complete["conversation_id"])
    incomplete_dir = audio_root / payload_incomplete["conversation_id"]
    incomplete_dir.mkdir(parents=True, exist_ok=True)
    (incomplete_dir / "full.wav").write_bytes(b"RIFF")

    output_dir = tmp_path / "hf_export_audio"
    summary = build_hf_export(
        record=record_path,
        input_dir=None,
        pattern="*.jsonl",
        output_dir=output_dir,
        mode="audio",
        audio_root=audio_root,
    )

    assert summary["exported"] == 1
    assert summary["skipped_missing_audio"] == 1

    train_jsonl = output_dir / "data" / "train.jsonl"
    rows = [json.loads(line) for line in train_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["consultation_id"] for row in rows] == [payload_complete["conversation_id"]]
    assert rows[0]["audio_available"] is True
    assert rows[0]["audio"]["full_audio"] == f"audio/{payload_complete['conversation_id']}/full.wav"

    assert (output_dir / rows[0]["audio"]["full_audio"]).exists()
    assert (output_dir / rows[0]["audio"]["audio_manifest"]).exists()
    assert (output_dir / rows[0]["audio"]["transcript_reference"]).exists()
    assert (output_dir / ".gitattributes").exists()
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "- text-to-speech" in readme
    assert "retained artifact: full consultation audio only" in readme


def test_build_hf_export_rebuilds_clean_bundle(tmp_path: Path) -> None:
    payload = _load_sample_payload()
    record_path = tmp_path / "records.jsonl"
    _write_jsonl(record_path, [payload])

    output_dir = tmp_path / "hf_export"
    build_hf_export(
        record=record_path,
        input_dir=None,
        pattern="*.jsonl",
        output_dir=output_dir,
        mode="text",
        audio_root=tmp_path / "audio",
    )

    stale_audio_dir = output_dir / "audio" / payload["conversation_id"]
    stale_audio_dir.mkdir(parents=True, exist_ok=True)
    (stale_audio_dir / "full.wav").write_bytes(b"stale")

    summary = build_hf_export(
        record=record_path,
        input_dir=None,
        pattern="*.jsonl",
        output_dir=output_dir,
        mode="text",
        audio_root=tmp_path / "audio",
    )

    assert summary["exported"] == 1
    assert summary["skipped_missing_audio"] == 0
    assert not stale_audio_dir.exists()

    exported_row = json.loads((output_dir / "data" / "train.jsonl").read_text(encoding="utf-8"))
    assert exported_row["consultation_id"] == payload["conversation_id"]


def test_push_export_to_hub_uses_hf_api(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeHfApi:
        def create_repo(self, **kwargs) -> None:
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs) -> None:
            calls.append(("upload_folder", kwargs))

    fake_module = types.SimpleNamespace(HfApi=FakeHfApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    push_export_to_hub(
        output_dir=tmp_path,
        repo_id="test-org/synthetic-consults",
        private=True,
        commit_message="Ship dataset",
    )

    assert calls == [
        (
            "create_repo",
            {
                "repo_id": "test-org/synthetic-consults",
                "repo_type": "dataset",
                "private": True,
                "exist_ok": True,
            },
        ),
        (
            "upload_folder",
            {
                "folder_path": str(tmp_path),
                "repo_id": "test-org/synthetic-consults",
                "repo_type": "dataset",
                "commit_message": "Ship dataset",
            },
        ),
    ]
