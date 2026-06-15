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

# 5. Search
python scripts\search.py "Docker AutoMigrate 失敗怎麼處理"
```
cli是