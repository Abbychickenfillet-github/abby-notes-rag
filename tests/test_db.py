"""Smoke test for db module. Requires running pgvector container + applied schema."""
import numpy as np
import pytest

from src.db import Database


@pytest.fixture
def db():
    d = Database()
    yield d
    # Clean up test rows
    d.execute("DELETE FROM chunks WHERE file_path LIKE 'TEST_%'")
    d.close()


def test_insert_and_count(db):
    emb = np.random.rand(1024).astype(np.float32)
    db.insert_chunk(
        file_path="TEST_smoke.md",
        file_hash="abc123",
        chunk_index=0,
        heading_path="# T",
        content="hello",
        token_count=2,
        embedding=emb,
    )
    rows = db.fetchall("SELECT COUNT(*) FROM chunks WHERE file_path='TEST_smoke.md'")
    assert rows[0][0] == 1


def test_get_file_hashes_returns_dict(db):
    emb = np.random.rand(1024).astype(np.float32)
    db.insert_chunk("TEST_hash.md", "deadbeef", 0, "# X", "x", 1, emb)
    hashes = db.get_file_hashes()
    assert hashes.get("TEST_hash.md") == "deadbeef"


def test_delete_file_chunks(db):
    emb = np.random.rand(1024).astype(np.float32)
    db.insert_chunk("TEST_del.md", "h", 0, "# A", "a", 1, emb)
    db.insert_chunk("TEST_del.md", "h", 1, "# A", "b", 1, emb)
    db.delete_file_chunks("TEST_del.md")
    rows = db.fetchall("SELECT COUNT(*) FROM chunks WHERE file_path='TEST_del.md'")
    assert rows[0][0] == 0


def test_search_returns_top_k(db):
    rng = np.random.default_rng(42)
    target = rng.random(1024, dtype=np.float32)
    target /= np.linalg.norm(target)

    for i in range(5):
        e = rng.random(1024, dtype=np.float32)
        e /= np.linalg.norm(e)
        db.insert_chunk(f"TEST_search_{i}.md", "h", 0, "# A", f"text {i}", 1, e)

    # Insert one chunk that IS the target
    db.insert_chunk("TEST_search_target.md", "h", 0, "# A", "target text", 1, target)

    results = db.search(target, top_k=3)
    assert len(results) == 3
    assert results[0]["file_path"] == "TEST_search_target.md"
    assert results[0]["similarity"] > 0.99
