def validate_record_for_audio(record) -> None:
    turns = getattr(record, "tts_script", None) or getattr(record, "conversation", None)
    if not turns:
        raise ValueError("Record has no usable turns for audio generation.")

    if len(turns) < 10:
        raise ValueError("Record has fewer than 10 turns.")

    for turn in turns:
        text = getattr(turn, "text", None) or getattr(turn, "utterance", None)
        if not text or not text.strip():
            raise ValueError(f"Turn {turn.turn_id} is empty.")
        if turn.speaker not in {"doctor", "patient"}:
            raise ValueError(f"Invalid speaker in turn {turn.turn_id}: {turn.speaker}")
