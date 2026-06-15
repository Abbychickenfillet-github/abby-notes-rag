# Abby-notes RAG

Personal knowledge RAG system over `../Abby-notes/`. See [design doc](../Abby-notes/RAG/2026-05-07-rag-system-design.md).

## Quick start

```powershell
# 1. Start pgvector
docker compose up -d

# 2. Activate venv + install deps
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Initialize DB schema（只有第一次！見下方說明，重跑會清空資料）
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

### 觀念：函式庫（library）vs 腳本（script）

這個專案的檔案分成兩種角色，每個檔頂端都標了 `# 📚 角色` 或 `# 🏃 角色`：

| | 函式庫 / 模組（library/module） | 腳本（script） |
|---|----------|------|
| 放哪 | `src/` 資料夾 | `scripts/` 資料夾 |
| 怎麼用 | **被別的檔 `import`** 來重複使用 | **直接用 `python` 執行** |
| 像什麼 | 工具箱裡的工具（鎚子、螺絲起子） | 拿工具來做事的「一份作業流程」 |
| 範例 | `src/embedder.py`（提供「文字→向量」功能） | `scripts/ingest.py`（呼叫 embedder 把筆記建索引） |
| 自己會跑嗎 | 不會，等別人呼叫 | 會，是你下指令的「進入點」 |

- **函式庫**：寫好一段可重複使用的功能（class / function），自己不會主動執行，
  要等腳本 `from src.embedder import Embedder` 把它載進去用。
- **腳本**：你實際在終端機打 `python scripts/ingest.py` 跑的那種檔，
  它負責「把流程串起來」，過程中會 import `src/` 裡的函式庫來幹活。

> 為什麼要分開？把「功能」(src/) 和「怎麼用功能」(scripts/) 拆開，
> 同一個 `embedder` 函式庫就能同時被 `ingest`、`search`、`ask` 三個腳本重複使用，不用複製貼上。
>
> 技術細節：腳本檔常見的 `if __name__ == "__main__": main()` 就是在說
> 「只有**被直接執行**時才跑 `main()`；若被別人 import 就不要自動跑」——這正是區分兩者的開關。

### 11 個檔的執行順序總表

每個檔頂端都有 `# ▶ 執行順序 [...]` 標註。分兩類：
**L = 函式庫**（被 `import`，不直接執行，依依賴載入）、**R = 進入點**（你打指令直接跑）。

| 標記 | 檔案 | 類型 | 說明 |
|------|------|------|------|
| L1 | `src/config.py` | 函式庫 | 最底層，讀 `.env`，最先被載入 |
| L2 | `src/chunker.py` | 函式庫 | 切 Markdown chunk（被 ingest 用） |
| L3 | `src/embedder.py` | 函式庫 | 文字→向量（被 ingest／retriever 用） |
| L4 | `src/db.py` | 函式庫 | pgvector CRUD＋檢索（被 ingest／retriever 用） |
| L5 | `src/retriever.py` | 函式庫 | 黏 embedder＋db（被 search／ask／validate 用） |
| R1 | `scripts/ingest.py` | 進入點 | **建索引——查詢前必先跑** |
| R2 | `scripts/search.py` | 進入點 | 純檢索（需先 R1） |
| R3 | `scripts/ask.py` | 進入點 | 檢索＋LLM 問答（需先 R1） |
| R4 | `scripts/validate.py` | 進入點 | 選用：驗證檢索命中率（需先 R1） |
| R5 | `scripts/semantic_switch.py` | 進入點 | 獨立工具：意圖分類（不需 DB／ingest） |
| — | `scripts/init_db.py` | 設定 | **只有第一次**建 schema，平時不重跑（會清空） |

> 一句話：第一次照 `init_db.py（設定）→ R1 ingest → R2/R3 查詢`；之後每次只要 `R2/R3`。
> L1–L5 不用自己跑，是被進入點 `import` 進去的。

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

## 常見問題

### `init_db.py` 要每次容器 `up` 之後都跑嗎？

**不用，只有第一次（或資料被清掉時）才跑。**

原因：`docker-compose.yml` 把資料庫掛在本地資料夾（bind mount）：

```yaml
volumes:
  - ./data:/var/lib/postgresql/data
```

所以資料**持久保存在 `./data/`**，跨 `docker compose up` / `down` / `restart` 都還在。
容器重開後，之前 ingest 進去的 chunks 都還在，直接 `search.py` / `ask.py` 即可。

⚠️ **重跑 `init_db.py` 會清空資料！** `init_db.sql` 裡有 `DROP TABLE IF EXISTS chunks CASCADE`，
重跑等於砍掉重建整張表 → 之前 embed 好的內容全沒了 → 必須重新 `ingest.py --full`。

| 情況 | 要跑 `init_db.py` 嗎？ |
|------|------------------------|
| 第一次架設（`data/` 還是空的） | ✅ 要 |
| 平常 `docker compose up`（`data/` 已有資料） | ❌ 不用 |
| 容器 restart / 電腦重開 | ❌ 不用 |
| 手動刪掉了 `data/` 資料夾 | ✅ 要（等於重來） |
| 改了 schema（例如換 model 導致 `EMBEDDING_DIM` 變動） | ✅ 要，並重新 ingest |

> 小提醒：`init_db.py`（初始化資料庫）和 `__init__.py`（Python 套件標記檔）是**完全不同**的東西，
> 別搞混。`__init__.py` 只是讓資料夾變成可 import 的套件，不需要、也不會去「執行」它。