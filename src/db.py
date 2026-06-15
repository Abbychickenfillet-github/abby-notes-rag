# ▶ 執行順序 [函式庫 L4／5]：被 ingest.py（寫入）與 retriever.py（查詢）載入，pgvector CRUD＋cosine 檢索。
"""Thin wrapper over psycopg2 + pgvector for the chunks table."""
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from src.config import Config


class Database:
    # ===== 連線層：管 connection 生命週期 (Infrastructure) =====
    def __init__(self):
        self._connect()

    def _connect(self):
        # Docker Desktop on Windows kills idle TCP after embed batches; we
        # additionally retry-on-OperationalError below.
        self.conn = psycopg2.connect(
            Config.dsn(),
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
        )
        self.conn.autocommit = True
        register_vector(self.conn)

    def close(self):
        self.conn.close()

    # ===== 下層：泛用 SQL 包裝 (Generic) =====
    # 只懂「SQL 字串 + cursor」，不懂 chunks 表長怎樣。
    # 其他所有方法都靠這兩個跑 SQL，是最薄的一層。
    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.conn.cursor() as cur:
            # Pass None when no params so psycopg2 skips %-substitution
            # (otherwise literal % chars in the SQL trigger IndexError)
            cur.execute(sql, params if params else None)

    def fetchall(self, sql: str, params: tuple = ()) -> List[tuple]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params if params else None)
            return cur.fetchall()

    # ===== 中層：chunks 表業務邏輯 (Chunks business) =====
    # 知道 chunks 表有哪些欄位、UPSERT 怎麼做。
    # 對外提供「寫一筆 / 寫一批 / 查 hash / 刪檔 / 清空」這些動作。
    def insert_chunk(
        self,
        file_path: str,
        file_hash: str,
        chunk_index: int,
        heading_path: str,
        content: str,
        token_count: int,
        embedding: np.ndarray,
    ) -> None:
        # ON CONFLICT ... DO UPDATE = UPSERT（撞到 unique key 就改寫，不報錯）。
        # EXCLUDED 是 PostgreSQL 內建假表，代表「這次想 INSERT 但被擋下的新資料那一列」。
        #   EXCLUDED.file_hash = 你這次要寫的新值；不加前綴的欄位 = 表裡現存的舊值。
        self.execute(
            """
            INSERT INTO chunks (file_path, file_hash, chunk_index, heading_path,
                                content, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path, chunk_index) DO UPDATE
            SET file_hash = EXCLUDED.file_hash,
                heading_path = EXCLUDED.heading_path,
                content = EXCLUDED.content,
                token_count = EXCLUDED.token_count,
                embedding = EXCLUDED.embedding
            """,
            (file_path, file_hash, chunk_index, heading_path, content, token_count, embedding),
        )

    def insert_chunks_batch(self, rows: List[tuple]) -> None:
        """rows = list of (file_path, file_hash, chunk_index, heading_path, content, token_count, embedding)"""
        sql = """
            INSERT INTO chunks (file_path, file_hash, chunk_index, heading_path,
                                content, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path, chunk_index) DO UPDATE
            SET file_hash = EXCLUDED.file_hash,
                heading_path = EXCLUDED.heading_path,
                content = EXCLUDED.content,
                token_count = EXCLUDED.token_count,
                embedding = EXCLUDED.embedding
        """
        try:
            with self.conn.cursor() as cur:
                cur.executemany(sql, rows)
        except psycopg2.OperationalError:
            # Connection died (Windows TCP reset after long embed); reconnect once.
            try:
                self.conn.close()
            except Exception:
                pass
            self._connect()
            with self.conn.cursor() as cur:
                cur.executemany(sql, rows)

    def get_file_hashes(self) -> Dict[str, str]:
        """Return {file_path: file_hash} for all currently-stored files."""
        rows = self.fetchall(
            "SELECT file_path, MIN(file_hash) FROM chunks GROUP BY file_path"
        )
        # 回 dict 而非 list：ingest 要用 file_path 當 key 做 O(1) 查找
        # （existing_hashes.get(rel) == new_hash），dict 查找比掃整個 list 快。
        return {fp: h for fp, h in rows}

    def delete_file_chunks(self, file_path: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE file_path = %s", (file_path,))
            return cur.rowcount

    def truncate_all(self) -> None:
        self.execute("TRUNCATE TABLE chunks RESTART IDENTITY")

    # ===== 上層：向量檢索 (Query / Search) =====
    # 知道 pgvector 的 <=> 算子與 cosine similarity 的數學。retriever.py 只呼叫這個。
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        # 參數名稱（query_embedding / top_k / filter_path_prefix）都是「自己取的」，
        # 改名不會壞，只是呼叫端要跟著改；語意清楚即可。
        # 對比：SQL 裡的 file_path / heading_path / content / embedding 是
        # 「資料庫欄位名」，必須跟 init_db.sql 的 CREATE TABLE 一字不差，不能亂改。
        filter_path_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine-similarity search. Returns list of dicts with keys: file_path, heading_path, content, similarity."""
        if filter_path_prefix:
            sql = """
                SELECT file_path, heading_path, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE file_path LIKE %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = (query_embedding, f"{filter_path_prefix}%", query_embedding, top_k)
        else:
            sql = """
                SELECT file_path, heading_path, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = (query_embedding, query_embedding, top_k)

        # <=> 是 pgvector 的「餘弦距離」(0=一樣, 2=相反)；
        # 1 - (embedding <=> q) 才翻成「餘弦相似度」(1=最像, 0=無關)。
        rows = self.fetchall(sql, params)
        # 為何轉成 list[dict]：
        #   list → 有「多筆」結果且已按相似度排序，index 0 就是最相關。
        #   dict → 呼叫端用 r["file_path"] 取值，不必記 tuple 的欄位順序 r[0]/r[1]。
        return [
            {"file_path": r[0], "heading_path": r[1], "content": r[2], "similarity": float(r[3])}
            for r in rows
        ]
