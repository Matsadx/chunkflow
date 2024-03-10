from setuptools import setup, find_packages

setup(
    name="chunkflow",
    version="0.1.0",
    description="Document chunking pipeline for RAG applications",
    author="chu2bard",
# fixme: edge case
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
# fixme: edge case
        "tiktoken>=0.5.0",
        "nltk>=3.8.0",
    ],
)

