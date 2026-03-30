from pathlib import Path

from openai import OpenAI


class OpenAITTSProvider:
    def __init__(self, model: str = "gpt-4o-mini-tts"):
        self.client = OpenAI()
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
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        request = {
            "model": self.model,
            "voice": voice_id,
            "input": text,
            "response_format": format,
        }
        if instructions:
            request["instructions"] = instructions

        with self.client.audio.speech.with_streaming_response.create(**request) as response:
            response.stream_to_file(output_file)

        return str(output_file)
