"""Ask a question over the Abby-notes RAG corpus (retrieval + LLM generation).

Phase 3 CLI: retrieve top-k chunks, stuff them into a prompt, let an LLM
answer in natural language. Three generation providers, switch with --provider:
  - ollama  本機免費，需先啟動 Ollama 並 `ollama pull qwen2.5:7b`
  - gemini  Google 免費額度，需設環境變數 GEMINI_API_KEY（aistudio.google.com 申請）
  - claude  Anthropic 付費，需設環境變數 ANTHROPIC_API_KEY（console.anthropic.com，需綁卡）

Usage:
    python scripts/ask.py "我學過哪些程式語言？"                          # 預設 ollama
    python scripts/ask.py "Docker 失敗怎麼辦" --provider gemini
    python scripts/ask.py "Booth 訂單規則" --provider claude --show-sources
    python scripts/ask.py "pgvector 設定" --provider ollama --model qwen2.5:3b
    python scripts/ask.py "退款流程" --filter "backend/" --top-k 8
"""
import argparse
import os
import sys
from pathlib import Path

import requests  # 只有 ollama provider 用得到

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import Retriever

OLLAMA_URL = "http://localhost:11434/api/generate"

# 各 provider 的預設 model（--model 留空時用這個）
DEFAULT_MODELS = {
    "ollama": "qwen2.5:3b",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-opus-4-7",
}

SYSTEM_PROMPT = """你是 Abby 的個人筆記問答助理。請只根據下面提供的「筆記片段」回答問題。

規則：
- 只用筆記片段裡的資訊回答，不要編造筆記裡沒有的內容。
- 如果筆記片段不足以回答，就直說「筆記裡找不到相關資訊」。
- 回答用繁體中文。
- 適當時可引用是哪個檔案來的。"""


# ---------- prompt 組裝（三家共用） ----------
def build_context(chunks: list[dict]) -> str:
    """把檢索到的 chunks 排成「【片段 N】來源: ... \n 內容」區塊。"""
    blocks = []
    for i, c in enumerate(chunks, 1):
        heading = f" > {c['heading_path']}" if c["heading_path"] else ""
        blocks.append(f"【片段 {i}】來源: {c['file_path']}{heading}\n{c['content']}")
    return "\n\n".join(blocks)


def build_user_content(query: str, chunks: list[dict]) -> str:
    """gemini / claude 用：context + 問題（SYSTEM_PROMPT 另外用 system 欄位傳）。"""
    context = build_context(chunks)
    return f"===== 筆記片段 =====\n{context}\n\n===== 問題 =====\n{query}"


def build_ollama_prompt(query: str, chunks: list[dict]) -> str:
    """ollama 的 /api/generate 只吃一條 prompt，所以把 system 也塞進同一條。"""
    return f"{SYSTEM_PROMPT}\n\n{build_user_content(query, chunks)}\n\n===== 回答 ====="


# ---------- 三個 provider 各自的生成函式 ----------
def call_ollama(query: str, chunks: list[dict], model: str) -> str:
    prompt = build_ollama_prompt(query, chunks)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300,
        )
    except requests.exceptions.ConnectionError:
        sys.exit(
            "ERROR: 連不上 Ollama (http://localhost:11434)。\n"
            "請先安裝並啟動 Ollama，然後 `ollama pull qwen2.5:7b`。"
        )
    if resp.status_code == 404:
        sys.exit(f"ERROR: Ollama 找不到模型 '{model}'。先跑 `ollama pull {model}`。")
    resp.raise_for_status()
    return resp.json()["response"].strip()


def call_gemini(query: str, chunks: list[dict], model: str) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("ERROR: 沒裝 google-genai。請跑 `pip install google-genai`。")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: 沒設 GEMINI_API_KEY 環境變數。\n"
            "到 https://aistudio.google.com 申請免費 key，再 setx GEMINI_API_KEY \"你的key\"。"
        )

    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=build_user_content(query, chunks),
            # system_instruction = Gemini 的「系統提示」欄位，等同 Claude 的 system
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
    except Exception as e:
        sys.exit(f"ERROR: Gemini 呼叫失敗：{e}")
    return (resp.text or "").strip()


def call_claude(query: str, chunks: list[dict], model: str) -> str:
    try:
        import anthropic
    except ImportError:
        sys.exit("ERROR: 沒裝 anthropic。請跑 `pip install anthropic`。")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ERROR: 沒設 ANTHROPIC_API_KEY 環境變數。\n"
            "到 https://console.anthropic.com 申請（需綁卡），再 setx ANTHROPIC_API_KEY \"你的key\"。"
        )

    client = anthropic.Anthropic()  # 自動讀 ANTHROPIC_API_KEY
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_content(query, chunks)}],
        )
    except anthropic.AuthenticationError:
        sys.exit("ERROR: ANTHROPIC_API_KEY 無效或已失效。")
    except anthropic.APIError as e:
        sys.exit(f"ERROR: Claude 呼叫失敗：{e}")
    # response.content 是 block 串列，挑出 text block 拼起來
    return "".join(b.text for b in resp.content if b.type == "text").strip()


GENERATORS = {
    "ollama": call_ollama,
    "gemini": call_gemini,
    "claude": call_claude,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="natural-language question")
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini", "claude"],
        default="ollama",
        help="生成用哪家 LLM（預設 ollama，免費免金鑰）",
    )
    parser.add_argument("--top-k", type=int, default=5, help="chunks to retrieve (default: 5)")
    parser.add_argument(
        "--filter",
        default=None,
        help="restrict to file_path starting with this prefix (e.g. 'backend/')",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="覆寫該 provider 的預設 model（留空用各家預設）",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="print the retrieved chunks before the answer",
    )
    args = parser.parse_args()

    # 留空就用該 provider 的預設 model
    model = args.model or DEFAULT_MODELS[args.provider]

    retriever = Retriever()
    chunks = retriever.search(args.query, top_k=args.top_k, filter_path_prefix=args.filter)
    retriever.close()

    if not chunks:
        print("(檢索不到任何片段，無法回答)")
        return

    if args.show_sources:
        print("=" * 80)
        print(f"檢索到 {len(chunks)} 個片段：")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] sim={c['similarity']:.3f}  {c['file_path']}")
        print("=" * 80)
        print()

    print(f"思考中（provider={args.provider}, model={model}）...\n")
    answer = GENERATORS[args.provider](args.query, chunks, model)

    print("=" * 80)
    print(f"問題: {args.query}")
    print("=" * 80)
    print(answer)
    print()


if __name__ == "__main__":
    main()
