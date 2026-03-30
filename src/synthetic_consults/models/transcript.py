from __future__ import annotations

from pydantic import BaseModel


class TranscriptArtifact(BaseModel):
    path: str
    text: str


class TranscriptReference(BaseModel):
    gold_verbatim: TranscriptArtifact
    gold_normalized: TranscriptArtifact
