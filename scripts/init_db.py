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
