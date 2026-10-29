"""Read and write JSONL (JSON Lines: one JSON object per line), streaming."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO, TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


def read(path: str | Path, model: type[M]) -> Iterator[M]:
    """Yield one validated record per non-empty line; "-" reads stdin.

    A bad line fails with its line number, so hand-written files are easy to fix.
    Fields the model doesn't know are kept on the record (passed through) and listed
    once on stderr, so a typo in an optional field is visible without breaking files
    written by other tools."""
    stream = sys.stdin if str(path) == "-" else open(path, encoding="utf-8")
    unknown: set[str] = set()
    try:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = model.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(f"{path}:{lineno}: invalid {model.__name__}:\n{exc}") from exc
            unknown |= set(record.model_extra or {}) - getattr(model, "also_known", frozenset())
            yield record
    finally:
        if stream is not sys.stdin:
            stream.close()
        if unknown:
            print(f"{path}: passing through fields {model.__name__} doesn't use: {', '.join(sorted(unknown))}",
                  file=sys.stderr)


def write(record: BaseModel, out: IO[str]) -> None:
    """Write one record as one line and flush, so partial results are visible at once."""
    out.write(json.dumps(record.model_dump(mode="json", by_alias=True), ensure_ascii=False) + "\n")
    out.flush()
