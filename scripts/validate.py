# 🏃 角色：腳本（script）——直接用 python 執行的「進入點」，內部會 import src/ 的函式庫來用。
# ▶ 執行順序 [進入點 R4]：選用——跑驗證題庫檢查檢索命中率（需先完成 R1 ingest）。
"""Run validation queries and print hit/miss summary.

Usage:
    python scripts/validate.py

Reads tests/test_queries.yaml, runs each query through the Retriever
(top_k=5), and reports which queries hit (any expected substring is found
in any returned file_path, case-insensitive). Prints overall hit rate;
PASS if >= 80%, otherwise FAIL.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import Retriever


def main():
    yaml_path = Path(__file__).parent.parent / "tests" / "test_queries.yaml"
    cases = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["queries"]

    # Instantiate Retriever once so the embedding model is loaded a single time.
    r = Retriever()
    hits = 0

    for case in cases:
        q = case["q"]
        expected = case["expect_any_of"]
        results = r.search(q, top_k=5)
        paths = [hit["file_path"] for hit in results]

        hit = any(any(exp.lower() in p.lower() for p in paths) for exp in expected)
        marker = "[OK]" if hit else "[FAIL]"
        if hit:
            hits += 1

        print(f"{marker} {q}")
        if not hit:
            print(f"    expected any of: {expected}")
            print(f"    got top-5:")
            for i, h in enumerate(results, 1):
                print(f"      {i}. sim={h['similarity']:.3f}  {h['file_path']}")

    rate = hits / len(cases) * 100
    print(f"\nHit rate: {hits}/{len(cases)} ({rate:.0f}%)")
    print("PASS" if rate >= 80 else "FAIL - below 80% target")
    r.close()


if __name__ == "__main__":
    main()
