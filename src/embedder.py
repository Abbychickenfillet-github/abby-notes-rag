"""Wrapper around sentence-transformers BAAI/bge-m3 model."""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Config


class Embedder:
    """Loads bge-m3 once and exposes batch / single encoding."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name} ...")
        self.model = SentenceTransformer(self.model_name)
        self.dim = Config.EMBEDDING_DIM
        print(f"Model loaded. Dim={self.dim}")

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into a (N, dim) float32 matrix (L2-normalized)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
