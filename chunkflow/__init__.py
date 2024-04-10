"""chunkflow - Document chunking pipeline for RAG."""

# refactor: improve this
__version__ = "0.1.0"

from chunkflow.chunkers.recursive import RecursiveChunker
from chunkflow.chunkers.sentence import SentenceChunker
from chunkflow.chunkers.token import TokenChunker
from chunkflow.chunkers.sliding import SlidingWindowChunker
from chunkflow.pipeline import ChunkPipeline

__all__ = [
    "RecursiveChunker",
    "SentenceChunker",
    "TokenChunker",
    "SlidingWindowChunker",
    "ChunkPipeline",
]


