"""Composable chunking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol


class Chunker(Protocol):
    def chunk(self, text: str) -> list[str]: ...


@dataclass
class ChunkResult:
    """Result of a chunking operation."""

    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"ChunkResult(index={self.index}, text={preview!r})"


class ChunkPipeline:
    """Chains chunking with optional pre/post processing.

    Usage:
        pipeline = (
            ChunkPipeline(RecursiveChunker(chunk_size=500))
            .pre_process(clean_html)
            .post_process(strip_whitespace)
            .with_metadata(source="doc.txt")
        )
        results = pipeline.run(text)
    """

    def __init__(self, chunker: Chunker):
        self._chunker = chunker
        self._pre: list[Callable[[str], str]] = []
        self._post: list[Callable[[str], str]] = []
        self._filters: list[Callable[[str], bool]] = []
        self._metadata: dict[str, Any] = {}

    def pre_process(self, fn: Callable[[str], str]) -> "ChunkPipeline":
        """Add a pre-processing step applied to input text."""
        self._pre.append(fn)
        return self

    def post_process(self, fn: Callable[[str], str]) -> "ChunkPipeline":
        """Add a post-processing step applied to each chunk."""
        self._post.append(fn)
        return self

    def filter(self, fn: Callable[[str], bool]) -> "ChunkPipeline":
        """Add a filter. Chunks where fn returns False are dropped."""
        self._filters.append(fn)
        return self

    def with_metadata(self, **kwargs) -> "ChunkPipeline":
        """Attach metadata to all chunk results."""
        self._metadata.update(kwargs)
        return self

    def run(self, text: str) -> list[ChunkResult]:
        """Execute the pipeline on input text."""
        processed = text
        for fn in self._pre:
            processed = fn(processed)

        raw_chunks = self._chunker.chunk(processed)

        results: list[ChunkResult] = []
        idx = 0
        for chunk in raw_chunks:
            for fn in self._post:
                chunk = fn(chunk)

            if not chunk.strip():
                continue

            skip = False
            for filt in self._filters:
                if not filt(chunk):
                    skip = True
                    break
            if skip:
                continue

            results.append(ChunkResult(
                text=chunk,
                index=idx,
                metadata=dict(self._metadata),
            ))
            idx += 1

        return results
