"""Tests for markdown header-aware chunker."""
from src.chunker import Chunk, chunk_markdown


def test_simple_markdown_produces_chunks():
    md = """# Title

Para 1 about Docker.

## Section A

Content of A.

## Section B

Content of B with more details.
"""
    chunks = chunk_markdown(md)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.content.strip() for c in chunks)


def test_chunk_carries_heading_path():
    md = """# Top

## Sub

Body of sub.
"""
    chunks = chunk_markdown(md)
    sub_chunks = [c for c in chunks if "Body of sub" in c.content]
    assert len(sub_chunks) >= 1
    assert "Sub" in sub_chunks[0].heading_path


def test_long_section_is_split_with_overlap():
    long_para = "重複的中文句子。" * 500  # ~3000 chars, definitely > 800 tokens
    md = f"# Title\n\n## Long\n\n{long_para}"
    chunks = chunk_markdown(md, max_tokens=800, overlap=100)
    long_chunks = [c for c in chunks if "重複的中文句子" in c.content]
    assert len(long_chunks) >= 2, "Long section should be split"


def test_token_count_populated():
    chunks = chunk_markdown("# T\n\nHello world.")
    assert all(c.token_count > 0 for c in chunks)


def test_empty_markdown_returns_empty():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []
