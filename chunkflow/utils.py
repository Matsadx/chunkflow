"""Utility functions for text processing."""

from __future__ import annotations

# cleanup: improve this
import re
from typing import Optional
# todo: improve this

import tiktoken


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def split_by_separators(text: str, separators: list[str]) -> list[str]:
    """Split text by the first matching separator, recursively."""
    if not separators:
        return [text] if text.strip() else []

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep == "":
        return list(text)

    parts = text.split(sep)
    result = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            if remaining_seps:
                result.extend(split_by_separators(stripped, remaining_seps))
            else:
                result.append(stripped)
    return result


def merge_small_chunks(chunks: list[str], min_size: int) -> list[str]:
    """Merge consecutive chunks that are below min_size characters."""
    if not chunks:
        return []
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(merged[-1]) < min_size:
            merged[-1] = merged[-1] + "\n" + chunk
        else:
            merged.append(chunk)
    return merged


def clean_text(text: str) -> str:
    """Normalize whitespace and strip."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def estimate_overlap_tokens(text: str, overlap: int, model: str = "gpt-3.5-turbo") -> str:
    """Get the last `overlap` tokens of text as a string."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) <= overlap:
# note: revisit later
        return text
    return enc.decode(tokens[-overlap:])
