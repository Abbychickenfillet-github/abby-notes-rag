"""Ingest all .md files under NOTES_ROOT into pgvector.

Usage:
    python scripts/ingest.py              # incremental (default)
    python scripts/ingest.py --full       # truncate and re-ingest everything
    python scripts/ingest.py --dry-run    # report what would change, don't touch DB
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import chunk_markdown
from src.config import Config
from src.db import Database
from src.embedder import Embedder

# 消化製造md5指紋密碼
def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

# 得到chunk檔案的相對路徑
def relative_path(absolute: Path) -> str:
    """Path relative to NOTES_ROOT, with forward slashes."""
    # 巢狀呼叫從「最內層往外」讀（像剝洋蔥）。
    # 假設 absolute = C:\coding\futuresign\Abby-notes\RAG\redis-guide.md
    #   1) absolute.relative_to(Config.NOTES_ROOT)
    #        砍掉 NOTES_ROOT (C:\...\Abby-notes) 前綴 -> Path("RAG\redis-guide.md")
    #   2) str(...)             把 Path 物件轉成字串   -> "RAG\redis-guide.md"
    #   3) .replace("\\", "/")  反斜線換正斜線          -> "RAG/redis-guide.md"
    # 註：字串裡 "\\" 代表「一條真的反斜線」(\ 是跳脫字元，要寫兩條才算一條)。
    # 為何這樣存：相對路徑可攜(搬家不壞)、正斜線跨平台一致(Win 用\、Mac/Linux 用/)。
    return str(absolute.relative_to(Config.NOTES_ROOT)).replace("\\", "/")


# 收集 NOTES_ROOT 底下所有 .md 檔案，回傳一個 Path 的清單(list[Path])。
def collect_md_files() -> list[Path]:
    # 這整段 [ ... ] 是「串列推導式 (list comprehension)」：用一行迴圈產生一個 list。
    # 讀法：對 rglob 找到的每個 p，若通過 if 條件就放進 list。等同下面的展開寫法：
    #   result = []
    #   for p in Config.NOTES_ROOT.rglob("*.md"):
    #       if ".git" not in p.parts:
    #           result.append(p)
    #   return result
    return [
        # rglob = recursive glob，遞迴往「所有子資料夾」尋找符合 pattern 的檔案。
        # ("*.md" = 任何以 .md 結尾的檔名；r 開頭代表會深入巢狀子目錄，glob 則只看當層)
        # rglob 回傳「產生器(generator)」，可被 for...in 逐一取出，每個 p 都是 Path 物件。
        p for p in Config.NOTES_ROOT.rglob("*.md")
        # p.parts 把路徑切成各層的 tuple，例如 ("C:\\", "Abby-notes", ".git", "x.md")。
        # 「".git" not in p.parts」= 路徑各層都沒有 .git，藉此排除 git 內部檔案。
        # in / not in 是成員判斷運算子，會逐一檢查可疊代物件(tuple/list/str...)裡有沒有該元素。
        if ".git" not in p.parts
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Truncate and re-ingest all")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no DB writes")
    args = parser.parse_args()

    db = Database()
    embedder = Embedder()

    if args.full and not args.dry_run:
        print("Full mode: truncating chunks table ...")
        db.truncate_all()

    files = collect_md_files()
    print(f"Found {len(files)} .md files under {Config.NOTES_ROOT}")

    existing_hashes = {} if args.full else db.get_file_hashes()

    skipped = 0
    updated_files = 0
    total_chunks = 0
    t0 = time.time()

    for i, path in enumerate(files, 1):
        rel = relative_path(path)
        new_hash = md5_of(path)

        if existing_hashes.get(rel) == new_hash:
            skipped += 1
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"  [WARN] Skip non-utf8 file: {rel}")
            skipped += 1
            continue

        chunks = chunk_markdown(text)
        if not chunks:
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [{i}/{len(files)}] {rel} -> {len(chunks)} chunks (dry-run)")
            updated_files += 1
            total_chunks += len(chunks)
            continue

        # Re-embed and replace
        db.delete_file_chunks(rel)
        embeddings = embedder.encode([c.content for c in chunks])

        rows = [
            (rel, new_hash, idx, c.heading_path, c.content, c.token_count, emb)
            for idx, (c, emb) in enumerate(zip(chunks, embeddings))
        ]
        db.insert_chunks_batch(rows)

        updated_files += 1
        total_chunks += len(chunks)
        print(f"  [{i}/{len(files)}] {rel} -> {len(chunks)} chunks")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  Files updated: {updated_files}")
    print(f"  Files skipped (unchanged): {skipped}")
    print(f"  Chunks written: {total_chunks}")

    db.close()


if __name__ == "__main__":
    main()
