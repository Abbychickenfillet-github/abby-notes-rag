"""Thin wrapper over psycopg2 + pgvector for the chunks table."""
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from src.config import Config


class Database:
    def __init__(self):
        self.conn = psycopg2.connect(Config.dsn())
        self.conn.autocommit = True
        register_vector(self.conn)

    def close(self):
        self.conn.close()

    # ---- generic helpers ----
    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.conn.cursor() as cur:
            # Pass None when no params so psycopg2 skips %-substitution
            # (otherwise literal % chars in the SQL trigger IndexError)
            cur.execute(sql, params if params else None)

    def fetchall(self, sql: str, params: tuple = ()) -> List[tuple]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params if params else None)
            return cur.fetchall()

    # ---- chunks-specific ----
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
        with self.conn.cursor() as cur:
            cur.executemany(
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
                rows,
            )

    def get_file_hashes(self) -> Dict[str, str]:
        """Return {file_path: file_hash} for all currently-stored files."""
        rows = self.fetchall(
            "SELECT file_path, MIN(file_hash) FROM chunks GROUP BY file_path"
        )
        return {fp: h for fp, h in rows}

    def delete_file_chunks(self, file_path: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE file_path = %s", (file_path,))
            return cur.rowcount

    def truncate_all(self) -> None:
        self.execute("TRUNCATE TABLE chunks RESTART IDENTITY")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
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

        rows = self.fetchall(sql, params)
        return [
            {"file_path": r[0], "heading_path": r[1], "content": r[2], "similarity": float(r[3])}
            for r in rows
        ]
