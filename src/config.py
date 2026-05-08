"""Centralized configuration loaded from .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # DB
    POSTGRES_HOST = os.environ["POSTGRES_HOST"]
    POSTGRES_PORT = int(os.environ["POSTGRES_PORT"])
    POSTGRES_USER = os.environ["POSTGRES_USER"]
    POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
    POSTGRES_DB = os.environ["POSTGRES_DB"]

    # Notes source
    NOTES_ROOT = Path(os.environ["NOTES_ROOT"])

    # Embedding
    EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
    EMBEDDING_DIM = int(os.environ["EMBEDDING_DIM"])

    # Chunking
    CHUNK_MAX_TOKENS = int(os.environ["CHUNK_MAX_TOKENS"])
    CHUNK_OVERLAP_TOKENS = int(os.environ["CHUNK_OVERLAP_TOKENS"])
    CHUNK_MIN_TOKENS = int(os.environ["CHUNK_MIN_TOKENS"])

    @classmethod
    def dsn(cls) -> str:
        return (
            f"host={cls.POSTGRES_HOST} port={cls.POSTGRES_PORT} "
            f"user={cls.POSTGRES_USER} password={cls.POSTGRES_PASSWORD} "
            f"dbname={cls.POSTGRES_DB}"
        )
