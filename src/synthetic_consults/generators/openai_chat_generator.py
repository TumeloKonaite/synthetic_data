from __future__ import annotations

import json
import logging
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from synthetic_consults.generators.base import GenerationError, StructuredGenerator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIChatGenerator(StructuredGenerator[T]):
    """
    Structured generator backed by the OpenAI Responses API.

    Notes:
    - Uses direct OpenAI calls, not the Agents SDK.
    - Requests JSON Schema constrained output.
    - Validates again with Pydantic before returning.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5.4-mini",
        client: OpenAI | None = None,
        max_output_tokens: int = 4000,
    ) -> None:
        self.model = model
        self.client = client or OpenAI()
        self.max_output_tokens = max_output_tokens

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.7,
    ) -> T:
        schema = response_model.model_json_schema()

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_output_tokens=self.max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAI request failed.")
            raise GenerationError(f"OpenAI request failed: {exc}") from exc

        try:
            content = getattr(response, "output_text", None)
            if not content:
                raise GenerationError("Model returned empty output_text.")

            payload = json.loads(content)
            return response_model.model_validate(payload)

        except (json.JSONDecodeError, ValidationError) as exc:
            logger.exception("Failed to parse/validate structured model output.")
            raise GenerationError(
                f"Invalid structured output for {response_model.__name__}: {exc}"
            ) from exc