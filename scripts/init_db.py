# 🏃 角色：腳本（script）——直接用 python 執行的「進入點」，內部會 import src/ 的函式庫來用。
# ▶ 執行順序 [設定階段・只有第一次]：建立 DB schema；平時不要重跑（DROP TABLE 會清空，需重新 ingest）。
"""Apply init_db.sql to the pgvector container."""
from pathlib import Path
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

SQL_FILE = Path(__file__).parent / "init_db.sql"


def main():
    sql = SQL_FILE.read_text(encoding="utf-8")

    conn = psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.close()
    print("Schema applied successfully.")


if __name__ == "__main__":
    main()
