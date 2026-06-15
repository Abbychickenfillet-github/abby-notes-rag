# Abby-notes RAG

Personal knowledge RAG system over `../Abby-notes/`. See [design doc](../Abby-notes/RAG/2026-05-07-rag-system-design.md).

## Quick start

```powershell
# 1. Start pgvector
docker compose up -d

# 2. Activate venv + install deps
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Initialize DB schema
python scripts\init_db.py

# 4. Ingest notes (~15-30 min first time, downloads bge-m3 ~2.3GB)
python scripts\ingest.py --full

# 5. Search（純檢索，回傳最相關的筆記片段）
python scripts\search.py "Docker AutoMigrate 失敗怎麼處理"

# 6. Ask（檢索 + LLM 生成自然語言答案）
python scripts\ask.py "我學過哪些程式語言？" --provider ollama
```

## 架構與執行順序

### `src/` 模組依賴（由底層往上）

| 順序 | 模組 | 職責 | 被誰呼叫 |
|------|------|------|----------|
| 1 | `config.py` | 讀 `.env`，集中所有設定（DB、model、chunk 參數） | 幾乎所有模組都 import 它 |
| 2 | `chunker.py` | 用 langchain text splitter 把 Markdown 依標題切成大小受限的 chunk | `ingest.py` |
| 3 | `embedder.py` | 包裝 bge-m3，把文字轉成 1024 維向量 | `ingest.py`（批次）、`retriever.py`（單句 query） |
| 4 | `db.py` | psycopg2 + pgvector：chunks 表的 CRUD 與 cosine 檢索 | `ingest.py`（寫入）、`retriever.py`（查詢） |
| 5 | `retriever.py` | 把 `embedder` + `db` 黏起來，對外只暴露 `.search()` | `scripts/search.py`、`scripts/ask.py` |

> 依賴方向：`config` → `chunker` / `embedder` / `db` → `retriever` → scripts。
> 上層只認識下一層，不直接碰更底層細節（例如 `retriever` 不自己寫 SQL，交給 `db`）。

### 兩條主流程

**A. 建立索引（離線，`ingest.py`，第一次或更新筆記時跑）**

```
scripts/ingest.py
  → collect_md_files()           # 掃 NOTES_ROOT 底下所有 .md
  → src/chunker.chunk_markdown() # 切 chunk
  → src/embedder.encode()        # chunk 文字 → 向量（批次）
  → src/db.insert_chunks_batch() # 寫進 pgvector（UPSERT）
```

**B. 查詢 / 問答（線上，`search.py` / `ask.py`）**

```
scripts/search.py 或 ask.py
  → src/retriever.Retriever()
       → src/embedder.encode_one(query)  # 問題 → 向量（同一個 model！）
       → src/db.search()                 # pgvector cosine 找最相近的 top-k
  → (只有 ask.py) 把檢索到的 chunks 塞進 prompt
       → LLM 生成答案（--provider ollama / gemini / claude 三選一）
```

> 關鍵：query 與 chunk 必須用**同一個 embedding model**（bge-m3），
> 否則兩者落在不同向量空間，cosine 相似度會失準。