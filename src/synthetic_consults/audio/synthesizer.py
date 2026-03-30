from pathlib import Path

from synthetic_consults.models.tts import TTSScript, TurnAudioArtifact


def _build_instructions(style: str, pace: str) -> str | None:
    parts: list[str] = []
    if style and style != "neutral":
        parts.append(style)
    if pace and pace != "normal":
        parts.append(f"Pace: {pace}.")
    if not parts:
        return None
    return " ".join(parts)


def synthesize_turns(tts_script: TTSScript, provider, output_dir: str) -> list[TurnAudioArtifact]:
    output_root = Path(output_dir)
    artifacts: list[TurnAudioArtifact] = []

    for turn in tts_script.turns:
        filename = f"{turn.turn_id:04d}_{turn.speaker}.{tts_script.format}"
        path = output_root / "turns" / filename

        provider.synthesize_to_file(
            text=turn.text,
            voice_id=turn.voice_id,
            output_path=str(path),
            format=tts_script.format,
            instructions=_build_instructions(turn.style, turn.pace),
        )

        artifacts.append(
            TurnAudioArtifact(
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                voice_id=turn.voice_id,
                text=turn.text,
                path=str(path),
                format=tts_script.format,
            )
        )

    return artifacts
