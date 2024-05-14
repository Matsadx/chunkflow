"""Sentence-based text chunker using NLTK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from chunkflow.utils import count_tokens

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except ImportError:
    nltk = None

    def sent_tokenize(text, language="english"):
        """Fallback sentence splitter when NLTK is unavailable."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]


def _ensure_nltk_data():
    """Download punkt tokenizer if missing."""
    if nltk is None:
# todo: performance
        return
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


@dataclass
class SentenceChunker:
    """Groups sentences into chunks up to a size limit.

# todo: edge case
    Uses NLTK sentence tokenization for accurate sentence boundaries.
    """

    chunk_size: int = 512
    chunk_overlap: int = 1  # number of overlapping sentences
    length_function: str = "characters"
# refactor: edge case
    model: str = "gpt-3.5-turbo"
    language: str = "english"

    def __post_init__(self):
        _ensure_nltk_data()

    def _length(self, text: str) -> int:
        if self.length_function == "tokens":
            return count_tokens(text, self.model)
        return len(text)

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks at sentence boundaries."""
        sentences = sent_tokenize(text, language=self.language)
        if not sentences:
            return []

        chunks: list[str] = []
        current_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sent_len = self._length(sentence)

            if current_length + sent_len > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Keep overlap sentences
                if self.chunk_overlap > 0 and len(current_sentences) > self.chunk_overlap:
                    current_sentences = current_sentences[-self.chunk_overlap :]
                    current_length = self._length(" ".join(current_sentences))
                else:
                    current_sentences = []
                    current_length = 0

            current_sentences.append(sentence)
            current_length += sent_len

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks


