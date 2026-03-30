from __future__ import annotations

import json
import wave
from pathlib import Path

from synthetic_consults.audio.stitcher import stitch_turn_audio
from synthetic_consults.models.consultation_record import ConsultationRecord
from synthetic_consults.pipelines.generate_audio_dataset import (
    build_runtime_tts_script,
    iter_record_payloads,
    run_audio_pipeline,
)


def _load_sample_record() -> ConsultationRecord:
    first_line = Path("data/processed/train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    return ConsultationRecord.model_validate(json.loads(first_line))


def test_iter_record_payloads_reads_jsonl(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample.jsonl"
    sample_path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    payloads = list(
        iter_record_payloads(
            record=sample_path,
            input_dir=None,
            pattern="*.jsonl",
            limit=None,
        )
    )

    assert payloads == [
        (f"{sample_path}:1", {"id": 1}),
        (f"{sample_path}:2", {"id": 2}),
    ]


def test_build_runtime_tts_script_maps_profiles_to_configured_voices() -> None:
    record = _load_sample_record()
    tts_script = build_runtime_tts_script(
        record,
        {
            "model": "gpt-4o-mini-tts",
            "format": "wav",
            "sample_rate": 16000,
            "tts_config_version": "v1",
            "voices": {"doctor_default": "alloy", "patient_default": "verse"},
        },
    )

    assert tts_script.consultation_id == record.conversation_id
    assert {turn.voice_id for turn in tts_script.turns} <= {"alloy", "verse"}
    assert "Voice profile:" in tts_script.turns[0].style
    assert "I\u2019m" in tts_script.turns[0].text


def test_run_audio_pipeline_writes_manifest_and_audio(tmp_path: Path, monkeypatch) -> None:
    record = _load_sample_record()

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

        def synthesize_to_file(
            self,
            *,
            text: str,
            voice_id: str,
            output_path: str,
            format: str = "wav",
            instructions: str | None = None,
        ) -> str:
            assert text
            assert voice_id in {"alloy", "verse"}
            assert format == "wav"
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 160)
            return str(output)

    monkeypatch.setattr(
        "synthetic_consults.pipelines.generate_audio_dataset.OpenAITTSProvider",
        FakeProvider,
    )

    manifest = run_audio_pipeline(
        record_obj=record,
        record_path="data/processed/train.jsonl:1",
        tts_config={
            "model": "gpt-4o-mini-tts",
            "format": "wav",
            "sample_rate": 16000,
            "tts_config_version": "v1",
            "default_pause_after_sec": 0.5,
            "voices": {"doctor_default": "alloy", "patient_default": "verse"},
        },
        output_root=str(tmp_path / "audio"),
    )

    manifest_path = tmp_path / "audio" / record.conversation_id / "audio_manifest.json"
    transcript_path = tmp_path / "audio" / record.conversation_id / "transcript_reference.json"
    full_audio_path = Path(manifest.full_audio_path)

    assert manifest_path.exists()
    assert transcript_path.exists()
    assert full_audio_path.exists()
    assert len(manifest.turn_audio_paths) == len(record.tts_script)


def test_run_audio_pipeline_full_only_removes_turn_files(tmp_path: Path, monkeypatch) -> None:
    record = _load_sample_record()

    class FakeProvider:
        def __init__(self, model: str) -> None:
            self.model = model

        def synthesize_to_file(
            self,
            *,
            text: str,
            voice_id: str,
            output_path: str,
            format: str = "wav",
            instructions: str | None = None,
        ) -> str:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 160)
            return str(output)

    monkeypatch.setattr(
        "synthetic_consults.pipelines.generate_audio_dataset.OpenAITTSProvider",
        FakeProvider,
    )

    manifest = run_audio_pipeline(
        record_obj=record,
        record_path="data/processed/train.jsonl:1",
        tts_config={
            "model": "gpt-4o-mini-tts",
            "format": "wav",
            "sample_rate": 16000,
            "tts_config_version": "v1",
            "default_pause_after_sec": 0.5,
            "voices": {"doctor_default": "alloy", "patient_default": "verse"},
        },
        output_root=str(tmp_path / "audio"),
        retain_turn_audio=False,
    )

    turns_dir = tmp_path / "audio" / record.conversation_id / "turns"
    assert Path(manifest.full_audio_path).exists()
    assert manifest.turn_audio_paths == []
    assert not turns_dir.exists()


def test_stitch_turn_audio_ignores_bogus_source_nframes(tmp_path: Path) -> None:
    turn_path = tmp_path / "turn.wav"
    with wave.open(str(turn_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)

    turn_artifact = type(
        "Artifact",
        (),
        {
            "turn_id": 1,
            "path": str(turn_path),
        },
    )()
    tts_turn = type("Turn", (), {"turn_id": 1, "pause_after_sec": 0.0})()
    tts_script = type("Script", (), {"format": "wav", "turns": [tts_turn]})()

    original_wave_open = wave.open

    class WaveReaderWrapper:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def getparams(self):
            params = self._wrapped.getparams()
            return wave._wave_params(
                params.nchannels,
                params.sampwidth,
                params.framerate,
                2147483647,
                params.comptype,
                params.compname,
            )

        def __enter__(self):
            self._wrapped.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._wrapped.__exit__(exc_type, exc, tb)

    def fake_wave_open(file, mode=None):
        opened = original_wave_open(file, mode)
        if "rb" in mode:
            return WaveReaderWrapper(opened)
        return opened

    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr("synthetic_consults.audio.stitcher.wave.open", fake_wave_open)
    try:
        full_audio = stitch_turn_audio(
            consultation_id="consult_test",
            turn_artifacts=[turn_artifact],
            tts_script=tts_script,
            output_dir=str(tmp_path),
        )
    finally:
        monkeypatch.undo()

    assert Path(full_audio.path).exists()
    with wave.open(full_audio.path, "rb") as stitched:
        assert stitched.getnframes() == 160
