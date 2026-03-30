from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class GenerationError(RuntimeError):
    """Raised when model generation fails or returns invalid output."""


class StructuredGenerator(ABC, Generic[T]):
    """Interface for schema-constrained LLM generation."""

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.7,
    ) -> T:
        raise NotImplementedError
