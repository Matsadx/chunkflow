from setuptools import setup, find_packages

setup(
    name="chunkflow",
    version="0.1.0",
    description="Document chunking pipeline for RAG applications",
    author="chu2bard",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "tiktoken>=0.5.0",
        "nltk>=3.8.0",
    ],
)
