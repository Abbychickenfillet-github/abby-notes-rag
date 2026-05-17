"""High-level search API combining embedder + db."""
from typing import Any, Dict, List, Optional

from src.db import Database
from src.embedder import Embedder


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.db = Database()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_path_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Embed the query and return the top_k closest chunks."""
        q_vec = self.embedder.encode_one(query)
        return self.db.search(q_vec, top_k=top_k, filter_path_prefix=filter_path_prefix)

    def close(self):
        self.db.close()
