"""Retriever: query string -> embedding -> cosine search -> top-k chunks."""
from typing import Optional

from src.db import Database
from src.embedder import Embedder


class Retriever:
    """Glue between Embedder and Database.search()."""

    def __init__(self, embedder: Optional[Embedder] = None, db: Optional[Database] = None):
        self.embedder = embedder or Embedder()
        self.db = db or Database()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_path_prefix: Optional[str] = None,
    ) -> list[dict]:
        """Embed query and return top-k chunks ordered by cosine similarity."""
        qvec = self.embedder.encode([query])[0]
        return self.db.search(qvec, top_k=top_k, filter_path_prefix=filter_path_prefix)

    def close(self) -> None:
        self.db.close()
