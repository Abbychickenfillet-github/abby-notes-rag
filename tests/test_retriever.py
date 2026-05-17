"""End-to-end retrieval test against the populated DB."""
from src.retriever import Retriever


def test_retrieve_docker_automigrate():
    r = Retriever()
    results = r.search("Docker AutoMigrate 失敗怎麼處理", top_k=5)
    assert len(results) == 5
    # Expect at least one Docker / migration / backend related path in top 5.
    # MEMORY.md lives in user's ~/.claude, not in the ingested Abby-notes corpus.
    paths_lower = [hit["file_path"].lower() for hit in results]
    assert any(
        "docker" in p or "backend" in p or "migrate" in p or "automigrate" in p
        for p in paths_lower
    ), f"Expected Docker/migrate-related file in top 5, got {paths_lower}"


def test_filter_by_path():
    r = Retriever()
    results = r.search("pgvector setup", top_k=5, filter_path_prefix="RAG/")
    assert len(results) >= 1
    assert all(hit["file_path"].startswith("RAG/") for hit in results)


def test_similarity_descending():
    r = Retriever()
    results = r.search("goroutine", top_k=5)
    sims = [h["similarity"] for h in results]
    assert sims == sorted(sims, reverse=True)
