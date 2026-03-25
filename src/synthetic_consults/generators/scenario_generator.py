from __future__ import annotations

from pathlib import Path


def consultation_dir(base_dir: str | Path, conversation_id: str) -> Path:
    return Path(base_dir) / conversation_id


def full_audio_path(base_dir: str | Path, conversation_id: str, ext: str = "wav") -> str:
    return str(consultation_dir(base_dir, conversation_id) / f"full.{ext}")


def turn_audio_path(base_dir: str | Path, conversation_id: str, turn_id: int, ext: str = "wav") -> str:
    return str(consultation_dir(base_dir, conversation_id) / f"turn_{turn_id:03d}.{ext}")