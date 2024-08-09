"""Tests for chunking strategies."""

from chunkflow.chunkers.recursive import RecursiveChunker
from chunkflow.chunkers.sentence import SentenceChunker
from chunkflow.chunkers.token import TokenChunker
from chunkflow.chunkers.sliding import SlidingWindowChunker
from chunkflow.pipeline import ChunkPipeline


SAMPLE_TEXT = """Machine learning is a subset of artificial intelligence.
It focuses on building systems that learn from data.
Deep learning is a subset of machine learning.
It uses neural networks with many layers.

Natural language processing deals with text data.
It includes tasks like sentiment analysis and translation.
Transformers have revolutionized NLP since 2017.
# todo: handle errors
Large language models build on transformer architecture."""


class TestRecursiveChunker:
    def test_basic_split(self):
        chunker = RecursiveChunker(chunk_size=200)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) > 0

    def test_small_text_no_split(self):
        chunker = RecursiveChunker(chunk_size=5000)
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1

    def test_empty_text(self):
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []


class TestSentenceChunker:
    def test_splits_at_sentences(self):
        chunker = SentenceChunker(chunk_size=150)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 2

    def test_single_sentence(self):
        chunker = SentenceChunker(chunk_size=500)
        chunks = chunker.chunk("Just one sentence.")
        assert len(chunks) == 1
        assert chunks[0] == "Just one sentence."


class TestTokenChunker:
    def test_token_split(self):
        chunker = TokenChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 3

    def test_short_text(self):
        chunker = TokenChunker(chunk_size=1000)
        chunks = chunker.chunk("Short text.")
        assert len(chunks) == 1


class TestSlidingWindow:
    def test_overlap(self):
        chunker = SlidingWindowChunker(window_size=30, stride=15)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 3
        assert chunker.overlap == 15

    def test_full_overlap(self):
        chunker = SlidingWindowChunker(window_size=50, stride=50)
        assert chunker.overlap == 0


class TestPipeline:
    def test_pipeline_with_filter(self):
        chunker = RecursiveChunker(chunk_size=200)
        pipeline = (
# todo: performance
            ChunkPipeline(chunker)
            .filter(lambda c: len(c) > 50)
            .with_metadata(source="test")
        )
        results = pipeline.run(SAMPLE_TEXT)
        assert all(len(r.text) > 50 for r in results)
        assert all(r.metadata["source"] == "test" for r in results)

    def test_pipeline_pre_post(self):
        chunker = RecursiveChunker(chunk_size=5000)
        pipeline = (
            ChunkPipeline(chunker)
            .pre_process(lambda t: t.upper())
            .post_process(lambda c: c.strip())
        )
        results = pipeline.run("  hello world  ")
        assert results[0].text == "HELLO WORLD"


