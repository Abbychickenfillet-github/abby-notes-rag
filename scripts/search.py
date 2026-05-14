"""Semantic search CLI over the ingested Abby-notes chunks.

Usage:
    python scripts/search.py "Docker AutoMigrate 失敗怎麼處理"
    python scripts/search.py "Chakra Popover" --top-k 10
    python scripts/search.py "Booth 訂單規則" --filter "工作日誌/"
    python scripts/search.py "React Hooks" --no-content
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import Retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("--top-k", type=int, default=5, help="number of results (default: 5)")
    parser.add_argument(
        "--filter",
        default=None,
        help="restrict to file_path starting with this prefix (e.g. 'RAG/')",
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="only list file path + heading, skip content preview",
    )
    args = parser.parse_args()

    retriever = Retriever()
    results = retriever.search(args.query, top_k=args.top_k, filter_path_prefix=args.filter)

    print(f"Query: {args.query}")
    if args.filter:
        print(f"Filter: file_path LIKE '{args.filter}%'")
    print(f"Top {args.top_k} results:")
    print()

    if not results:
        print("(no results)")
        retriever.close()
        return

    for rank, r in enumerate(results, 1):
        print(f"[{rank}] sim={r['similarity']:.3f}  {r['file_path']}")
        if r["heading_path"]:
            print(f"    Heading: {r['heading_path']}")
        if not args.no_content:
            preview = r["content"][:200].replace("\n", " ")
            suffix = " ..." if len(r["content"]) > 200 else ""
            print(f"    Content: {preview}{suffix}")
        print()

    retriever.close()


if __name__ == "__main__":
    main()
