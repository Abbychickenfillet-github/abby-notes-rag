# ▶ 執行順序 [函式庫 L3／5]：被 ingest.py（批次）與 retriever.py（單句 query）載入，文字→1024 維向量。
"""Wrapper around sentence-transformers BAAI/bge-m3 model."""

# typing.List 是 Python 內建型別提示工具，讓 List[str] 等寫法被靜態檢查器看懂。
# Python 3.9+ 之後 list[str] 也能用，但這裡為了相容性顯式 import。
from typing import List

# numpy 是科學計算套件，這裡用它的 ndarray 來裝向量。
# 別名習慣寫成 np（社群共識）。
import numpy as np

# sentence-transformers 是 HuggingFace 上包好的「句子嵌入」高階 API，
# 內建很多預訓練模型（包含 BAAI/bge-m3）。
from sentence_transformers import SentenceTransformer

# Config 是專案自己的設定類別（src/config.py），
# 把環境變數（model 名稱、向量維度等）集中管理。
from src.config import Config


class Embedder:
    """Loads bge-m3 once and exposes batch / single encoding."""

    # __init__ = 建構子，建立 Embedder() 物件時自動跑。
    # model_name 是可選參數（傳 None 就用 Config 預設）。
    # str | None 是 Python 3.10+ 的 union type 寫法，等同 Optional[str]。
    def __init__(self, model_name: str | None = None):
        # 「or」在這裡做「預設值」：傳進來的 model_name 若為 None 或空字串，
        # 就 fallback 到 Config.EMBEDDING_MODEL（即 "BAAI/bge-m3"）。
        self.model_name = model_name or Config.EMBEDDING_MODEL

        # 先印出來，讓人看到正在載入哪個 model（這個 model ~2.3GB，要等）。
        print(f"Loading embedding model: {self.model_name} ...")

        # SentenceTransformer(...) 真正把 model 從硬碟 / HuggingFace 載入記憶體。
        # 第一次跑會下載到 ~/.cache/huggingface/，之後從快取讀。
        # 載完之後 self.model 就能呼叫 .encode() 把文字轉向量。
        self.model = SentenceTransformer(self.model_name)

        # bge-m3 的固定維度是 1024，從 Config 拿來存著方便外部讀。
        self.dim = Config.EMBEDDING_DIM

        # 載入完成的訊號（含維度確認）。
        print(f"Model loaded. Dim={self.dim}")

    # encode = 主力方法：把一串文字批次轉成向量矩陣。
    # texts: List[str]    一次傳一群句子（chunk）
    # batch_size: int = 32  一次塞給 GPU/CPU 32 句，太大會 OOM(out-of-memory)需要的batch大過於硬體實際能給的，太小會慢
    # 回傳 np.ndarray：形狀 (N, 1024) 的 float32 矩陣
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into a (N, dim) float32 matrix (L2-normalized)."""
        """批次大小Batch Size:是機器學習與深度學習中控制單次訓練傳遞給模型樣本數量的超參數"""
        # 真正的推論發生在這裡：model.encode 內部跑 forward pass。
        embeddings = self.model.encode(
            texts,
            # batch_size 控制一次塞幾句進 GPU/CPU。
            batch_size=batch_size,
            # normalize_embeddings=True → 輸出向量會被 L2-normalize（長度=1）。
            # 這樣 cosine similarity == dot product，後面查詢更快。
            normalize_embeddings=True,
            # 句子超過 50 句才顯示進度條，少量不要刷一堆 log。
            show_progress_bar=len(texts) > 50,
        )

        # bge-m3 預設回 float32，但保險起見明確 cast，避免後續 pgvector 型別錯誤。
        return embeddings.astype(np.float32)

    # 便利方法：只有一句話時不想自己包 [text] 又解包，這裡幫你做。
    # 回傳 shape (1024,) 的 1D array（不是 (1, 1024)）。
    def encode_one(self, text: str) -> np.ndarray:
        # 包成 list 跑 batch encoding → 拿第 0 個（也是唯一一個）結果。
        return self.encode([text])[0]
