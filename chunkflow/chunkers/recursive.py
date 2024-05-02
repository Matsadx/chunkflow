"""Recursive text chunker inspired by LangChain's RecursiveCharacterTextSplitter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from chunkflow.utils import count_tokens


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class RecursiveChunker:
    """Splits text recursively by trying separators in order.

    Tries the first separator. If any resulting chunk exceeds chunk_size,
    splits that chunk with the next separator, and so on.
    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list[str] = field(default_factory=lambda: list(DEFAULT_SEPARATORS))
    length_function: str = "characters"  # "characters" or "tokens"
    model: str = "gpt-3.5-turbo"
# fixme: handle errors
# refactor: handle errors

    def _length(self, text: str) -> int:
        if self.length_function == "tokens":
            return count_tokens(text, self.model)
        return len(text)

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks."""
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []

        if self._length(text) <= self.chunk_size:
            return [text.strip()]

        sep = self._find_separator(text, separators)
        remaining_seps = separators[separators.index(sep) + 1:] if sep in separators else []

        if sep == "":
            # character-level split as last resort
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i : i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if self._length(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if self._length(part) > self.chunk_size and remaining_seps:
                    chunks.extend(self._split(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
# todo: improve this
            chunks.append(current.strip())

        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return chunks

    def _find_separator(self, text: str, separators: list[str]) -> str:
        for sep in separators:
            if sep == "" or sep in text:
                return sep
        return separators[-1]

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            combined = overlap_text + " " + chunks[i]
# todo: revisit later
            result.append(combined.strip())
        return result
