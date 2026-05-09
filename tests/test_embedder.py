"""Smoke test for bge-m3 wrapper. Downloads model on first run (~2.3GB)."""
import numpy as np
import pytest

from src.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_encode_single_returns_1024_dim(embedder):
    v = embedder.encode_one("Docker AutoMigrate 失敗怎麼處理")
    assert isinstance(v, np.ndarray)
    assert v.shape == (1024,)
    assert v.dtype == np.float32


def test_encode_batch_returns_matrix(embedder):
    texts = ["Hello world", "你好世界", "Goroutine 的用法"]
    matrix = embedder.encode(texts)
    assert matrix.shape == (3, 1024)


def test_similar_texts_have_high_cosine(embedder):
    v1 = embedder.encode_one("Docker 容器啟動失敗")
    v2 = embedder.encode_one("Docker container 無法啟動")
    v3 = embedder.encode_one("烤蛋糕食譜")
    # Cosine sim (vectors are normalized by bge-m3)
    sim_close = float(np.dot(v1, v2))
    sim_far = float(np.dot(v1, v3))
    assert sim_close > sim_far
    assert sim_close > 0.6  # Sanity threshold for clearly related sentences
