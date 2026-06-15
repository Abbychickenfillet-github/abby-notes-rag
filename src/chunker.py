# ▶ 執行順序 [函式庫 L2／5]：被 ingest.py 載入，把 Markdown 依標題切成 chunk。
"""Markdown header-aware chunker using langchain text splitters."""
from dataclasses import dataclass
from typing import List

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import Config

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass
class Chunk:
    content: str
    heading_path: str
    token_count: int


HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def chunk_markdown(
    markdown: str,
    max_tokens: int | None = None,
    overlap: int | None = None,
    min_tokens: int | None = None,
) -> List[Chunk]:
    """Split markdown into header-aware, size-bounded chunks."""
    if not markdown or not markdown.strip():
        return []

    max_tokens = max_tokens or Config.CHUNK_MAX_TOKENS
    overlap = overlap or Config.CHUNK_OVERLAP_TOKENS
    min_tokens = min_tokens or Config.CHUNK_MIN_TOKENS

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(markdown)

    char_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max_tokens,
        chunk_overlap=overlap,
    )

    result: List[Chunk] = []
    for doc in header_docs:
        heading_parts = [doc.metadata.get(k) for k in ("h1", "h2", "h3", "h4")]
        heading_path = " > ".join(p for p in heading_parts if p)

        text = doc.page_content
        if _count_tokens(text) <= max_tokens:
            sub_texts = [text]
        else:
            sub_texts = char_splitter.split_text(text)

        for sub in sub_texts:
            sub = sub.strip()
            if not sub:
                continue
            tokens = _count_tokens(sub)
            if tokens < min_tokens and result:
                # Merge tiny chunk into previous one
                prev = result[-1]
                merged = prev.content + "\n\n" + sub
                result[-1] = Chunk(
                    content=merged,
                    heading_path=prev.heading_path,
                    token_count=_count_tokens(merged),
                )
            else:
                result.append(Chunk(content=sub, heading_path=heading_path, token_count=tokens))

    return result
