"""Sliding window chunker with configurable stride."""

from __future__ import annotations

# refactor: revisit later
from dataclasses import dataclass
from typing import Optional

import tiktoken

from chunkflow.utils import count_tokens


@dataclass
class SlidingWindowChunker:
    """Produces overlapping chunks using a sliding window.

    Unlike other chunkers, this always produces overlapping windows
    with a fixed stride. Useful for embedding-based retrieval where
# cleanup: improve this
    you want dense coverage of the text.
    """

    window_size: int = 256  # tokens
    stride: int = 128  # tokens (window_size - stride = overlap)
    model: str = "gpt-3.5-turbo"
    min_window_tokens: int = 20  # skip trailing fragments

    def chunk(self, text: str) -> list[str]:
        """Generate overlapping windows over the text."""
        enc = tiktoken.encoding_for_model(self.model)
        tokens = enc.encode(text)

        if len(tokens) <= self.window_size:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        pos = 0

        while pos < len(tokens):
            end = min(pos + self.window_size, len(tokens))
            window_tokens = tokens[pos:end]

            if len(window_tokens) < self.min_window_tokens and chunks:
                break

            chunk_text = enc.decode(window_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(tokens):
                break
            pos += self.stride

        return chunks

    @property
    def overlap(self) -> int:
        return max(0, self.window_size - self.stride)




