from __future__ import annotations

import wave
from pathlib import Path

from synthetic_consults.models.tts import FullAudioArtifact, TTSScript, TurnAudioArtifact


def _ensure_matching_params(reference: wave._wave_params, candidate: wave._wave_params) -> None:
    if (
        reference.nchannels != candidate.nchannels
        or reference.sampwidth != candidate.sampwidth
        or reference.framerate != candidate.framerate
        or reference.comptype != candidate.comptype
    ):
        raise ValueError("Turn audio files do not share the same WAV parameters.")


def stitch_turn_audio(
    *,
    consultation_id: str,
    turn_artifacts: list[TurnAudioArtifact],
    tts_script: TTSScript,
    output_dir: str,
) -> FullAudioArtifact:
    if tts_script.format != "wav":
        raise ValueError("Only WAV output is currently supported for stitched audio.")
    if not turn_artifacts:
        raise ValueError("No turn audio artifacts were provided for stitching.")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    full_audio_path = output_root / f"full.{tts_script.format}"
    pause_lookup = {turn.turn_id: turn.pause_after_sec for turn in tts_script.turns}
    reference_params: wave._wave_params | None = None

    with wave.open(str(full_audio_path), "wb") as dst:
        for idx, artifact in enumerate(turn_artifacts):
            with wave.open(artifact.path, "rb") as src:
                params = src.getparams()
                if reference_params is None:
                    reference_params = params
                    dst.setnchannels(params.nchannels)
                    dst.setsampwidth(params.sampwidth)
                    dst.setframerate(params.framerate)
                    dst.setcomptype(params.comptype, params.compname)
                else:
                    _ensure_matching_params(reference_params, params)

                dst.writeframesraw(src.readframes(src.getnframes()))

                pause_after = pause_lookup.get(artifact.turn_id, 0.0)
                if idx < len(turn_artifacts) - 1 and pause_after > 0:
                    silent_frames = int(params.framerate * pause_after)
                    silence = b"\x00" * silent_frames * params.nchannels * params.sampwidth
                    dst.writeframesraw(silence)

    return FullAudioArtifact(
        consultation_id=consultation_id,
        path=str(full_audio_path),
        turn_paths=[artifact.path for artifact in turn_artifacts],
        format=tts_script.format,
    )
