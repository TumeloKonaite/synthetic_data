from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")