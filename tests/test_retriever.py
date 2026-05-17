"""End-to-end retrieval test against the populated DB."""
from src.retriever import Retriever


def test_retrieve_docker_automigrate():
    r = Retriever()
    results = r.search("Docker AutoMigrate 失敗怎麼處理", top_k=5)
    assert len(results) == 5
    # Expect MEMORY.md or backend-related notes in top 5
    paths = [hit["file_path"] for hit in results]
    assert any("MEMORY" in p or "backend" in p.lower() or "AutoMigrate" in p for p in paths), \
        f"Expected AutoMigrate-related file in top 5, got {paths}"


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
