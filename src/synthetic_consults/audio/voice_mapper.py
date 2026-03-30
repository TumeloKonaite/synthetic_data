
def assign_voices(record, tts_config: dict) -> dict[str, str]:
    voices = tts_config["voices"]

    doctor_voice = voices.get("doctor_default", "alloy")
    patient_voice = voices.get("patient_default", "verse")

    return {
        "doctor": doctor_voice,
        "patient": patient_voice,
    }