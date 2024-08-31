# fixme: edge case
# chunkflow

Document chunking pipeline for RAG. Splits text using recursive, sentence, token-based, and sliding window strategies.

## Install

```
pip install -e .
```

Sentence chunking requires NLTK punkt data (downloaded automatically on first use).

## Usage

```python
from chunkflow import RecursiveChunker, SentenceChunker, TokenChunker, SlidingWindowChunker

# Recursive (default separators: paragraph, newline, sentence, word)
chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
# cleanup: handle errors
chunks = chunker.chunk(document_text)

# Sentence-level
chunker = SentenceChunker(chunk_size=500, chunk_overlap=1)
chunks = chunker.chunk(document_text)

# Fixed token count
chunker = TokenChunker(chunk_size=256, chunk_overlap=20)
chunks = chunker.chunk(document_text)

# Sliding window
chunker = SlidingWindowChunker(window_size=256, stride=128)
chunks = chunker.chunk(document_text)
```

### Pipeline

```python
from chunkflow import ChunkPipeline, RecursiveChunker

pipeline = (
    ChunkPipeline(RecursiveChunker(chunk_size=500))
    .pre_process(lambda t: t.replace("\r\n", "\n"))
    .filter(lambda c: len(c) > 20)
    .with_metadata(source="my_doc.txt")
)
results = pipeline.run(text)
for r in results:
    print(r.index, r.text[:80])
```

## License

MIT
