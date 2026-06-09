"""PaperQA-compatible parsed document types (no paper-qa dependency required)."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field


class ImpossibleParsingError(Exception):
    """Raised when a document cannot be parsed reliably."""


class ParsedMetadata(BaseModel):
    parsing_libraries: list[str] = Field(
        description="Libraries used to generate the parsing."
    )
    total_parsed_text_length: int = Field(ge=0)
    count_parsed_media: int = Field(default=0, ge=0)
    name: str | None = None
    paperqa_version: str | None = Field(
        default=None,
        description="Optional PaperQA version when integrated with paper-qa.",
    )


class ParsedMedia(BaseModel):
    index: int
    data: bytes = b""
    text: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: str | os.PathLike) -> None:
        from pathlib import Path

        Path(path).write_bytes(self.data)


class ParsedText(BaseModel):
    content: (
        dict[str, str]
        | str
        | list[str]
        | dict[str, tuple[str, list[ParsedMedia]]]
    )
    metadata: ParsedMetadata

    def reduce_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n\n".join(self.content)
        return "\n\n".join(
            x[0] if not isinstance(x, str) else x for x in self.content.values()
        )
