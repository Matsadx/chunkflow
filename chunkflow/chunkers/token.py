"""Token-based text chunker using tiktoken."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass
class TokenChunker:
    """Splits text into chunks of a fixed number of tokens.

    Uses tiktoken for accurate token counting. Chunk boundaries
    align with token boundaries, not character or word boundaries.
    """

    chunk_size: int = 256
    chunk_overlap: int = 20
    model: str = "gpt-3.5-turbo"

    def chunk(self, text: str) -> list[str]:
        """Split text into token-aligned chunks."""
        enc = tiktoken.encoding_for_model(self.model)
        tokens = enc.encode(text)

        if len(tokens) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        if step <= 0:
            step = max(1, self.chunk_size // 2)

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)
            start += step

        return chunks
