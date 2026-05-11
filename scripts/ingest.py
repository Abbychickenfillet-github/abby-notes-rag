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


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def relative_path(absolute: Path) -> str:
    """Path relative to NOTES_ROOT, with forward slashes."""
    return str(absolute.relative_to(Config.NOTES_ROOT)).replace("\\", "/")


def collect_md_files() -> list[Path]:
    return [
        p for p in Config.NOTES_ROOT.rglob("*.md")
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
