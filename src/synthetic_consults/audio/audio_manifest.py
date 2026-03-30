import json
from pathlib import Path

from synthetic_consults.models.tts import AudioManifest


def save_audio_manifest(manifest: AudioManifest, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return str(path)


def save_json(data: dict, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)
