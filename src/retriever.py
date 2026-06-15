# ▶ 執行順序 [函式庫 L5／5]：被 search.py / ask.py / validate.py 載入，黏合 embedder＋db。
"""High-level search API combining embedder + db."""

# typing 的 4 個工具：
#   Any     → 任意型別（dict value 是混雜型別時用）
#   Dict    → 字典，會寫成 Dict[key_type, value_type]
#   List    → 列表，List[element_type]
#   Optional → 「這個值可能是 None」，Optional[str] 等同 str | None
from typing import Any, Dict, List, Optional

# Database 是 src/db.py 的薄包裝（負責 pgvector CRUD + cosine search）
from src.db import Database

# Embedder 是 src/embedder.py 的 bge-m3 包裝（負責文字 → 1024 維向量）
from src.embedder import Embedder


# Retriever = 「把 embedder + db 黏起來」的高階 API。
# 對外只暴露一個 .search() 方法，使用者不用知道內部是先 embed 再查 DB。
class Retriever:
    # 建構子：建立 Retriever() 時自動跑。
    # 這裡會載入 bge-m3 model（~10 秒）+ 連線 pgvector，所以「建立物件」很重。
    # 在 CLI / Jupyter 用法是「建一次、查很多次」，不要每次查詢都 new 一個。
    def __init__(self):
        # 載入 embedding model 進記憶體（2.3 GB），準備好 .encode_one()
        self.embedder = Embedder()
        # 建立 psycopg2 連線到 pgvector，準備好 .search()
        self.db = Database()

    # search = 主力方法：給一個自然語言查詢，回傳最相關的 top_k 個 chunks。
    #
    # 參數：
    #   query: str                            使用者打的問題（中英文都行）
    #   top_k: int = 5                        要回幾筆（預設 5）
    #   filter_path_prefix: Optional[str]     可選的路徑前綴過濾
    #                                          （例：filter_path_prefix="RAG/" 只搜 RAG 資料夾）
    #
    # 回傳：
    #   List[Dict[str, Any]]    每個 dict 有 4 個 key：
    #     - file_path     筆記檔的相對路徑
    #     - heading_path  chunk 所在的標題階層（e.g. "Postgres > WAL > 是什麼"）
    #     - content       chunk 的原文
    #     - similarity    跟 query 的 cosine 相似度（0~1，越高越像）
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_path_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Embed the query and return the top_k closest chunks."""

        # Step 1：把查詢文字變成 1024 維向量（跟 ingest 時用同一個 model，這很重要）。
        # 若用不同 model 算 query 跟 chunk，cosine 結果會是垃圾（不同向量空間）。
        q_vec = self.embedder.encode_one(query)

        # Step 2：把 query vector 丟給 db.search()，DB 端做 cosine ANN 搜尋。
        # pgvector 用 `embedding <=> q_vec` 算 cosine distance，
        # 再 1 - distance 變 similarity，回傳 top_k 結果。
        return self.db.search(q_vec, top_k=top_k, filter_path_prefix=filter_path_prefix)

    # 顯式關閉 DB 連線。CLI 跑完 / Jupyter cell 結束時呼叫一下會比較乾淨。
    # （psycopg2 不主動釋放連線可能會撐到 process 結束才釋放）
    def close(self):
        self.db.close()
