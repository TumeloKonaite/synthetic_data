import re

from synthetic_consults.models.tts import TTSScript, TTSTurn

MAX_WORDS_PER_UTTERANCE = 60


def normalize_tts_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[ ]+([,?.!])", r"\1", text)
    return text


def estimate_pause(text: str, default_pause: float) -> float:
    if text.endswith("?"):
        return default_pause + 0.2
    if text.endswith("."):
        return default_pause + 0.1
    return default_pause


def build_tts_script(record, voice_map: dict[str, str], tts_config: dict) -> TTSScript:
    turns = []

    for turn in record.conversation:
        text = normalize_tts_text(turn.utterance)
        if len(text.split()) > MAX_WORDS_PER_UTTERANCE:
            raise ValueError(f"Turn {turn.turn_id} too long for natural TTS")

        turns.append(
            TTSTurn(
                turn_id=turn.turn_id,
                speaker=turn.speaker,
                text=text,
                voice_id=voice_map[turn.speaker],
                style="neutral",
                pace="normal",
                pause_after_sec=estimate_pause(
                    text, tts_config.get("default_pause_after_sec", 0.6)
                ),
            )
        )

    return TTSScript(
        consultation_id=record.conversation_id,
        locale=tts_config.get("locale", "en-ZA"),
        model=tts_config.get("model", "gpt-4o-mini-tts"),
        format=tts_config.get("format", "wav"),
        sample_rate=tts_config.get("sample_rate", 16000),
        turns=turns,
        synthetic_version=record.synthetic_version,
        tts_config_version=tts_config.get("tts_config_version", "v1"),
    )
