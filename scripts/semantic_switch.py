# 🏃 角色：腳本（script）——直接用 python 執行的「進入點」，內部會 import src/ 的函式庫來用。
# ▶ 執行順序 [進入點 R5]：獨立工具——意圖分類，與 RAG 主線平行（只需 embedder，不需 DB／ingest）。
"""語意 switch：用 embedding 相似度,把使用者的自由文字「分類」到預先定義的 case。

這跟 RAG 的差別：
  RAG  = 在「一大堆筆記 chunks」裡撈最相關的片段（檢索 / retrieval）
  本檔 = 把輸入「分類」到「我寫好的幾個 case」其中一個（意圖辨識 / classification）

底層數學一樣（都是 bge-m3 向量 + cosine 相似度），所以直接重用 src/embedder.py 的
Embedder；但 case 只有幾個,不需要 pgvector,在記憶體算就好。

跑法：
    python scripts/semantic_switch.py                 # 互動模式,一直輸入一直判斷
    python scripts/semantic_switch.py "我好累想找人聊聊"   # 單句判斷
"""
import sys
from pathlib import Path

import numpy as np

# 讓 "import src.embedder" 找得到（把專案根目錄加進 import 路徑）
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embedder import Embedder  # ← 直接重用你寫好的 bge-m3 包裝


# ===== 1. 定義 case =====================================================
# 關鍵升級：每個 case 不再只給「一句」描述,而是給「多句不同講法」(examples)。
# 為什麼？因為同一個意圖,人會用很多種說法。多給幾句,涵蓋面就大很多,
# 判斷時取「最像的那一句」當分數（見下方 max 的註解）。
CASES = [
    {
        "label": "去圖書館",
        "examples": [
            "想要有人傾聽、在家太窒息,想黏在一起但又不想耍廢",
            "我好累,好想找個人說說話",
            "待在家好悶,想出門但又想跟你在一起",
            "想找個地方一起待著,順便做點正事",
        ],
        "action": lambda: print("→ 跟著阿弘去圖書館 📚"),
    },
    {
        "label": "待在家",
        "examples": [
            "在外面壓力太大,想跟阿弘待在家享受兩個人的空間",
            "外面好煩,只想兩個人窩在家",
            "今天不想見任何人,只想在家放鬆",
            "想耍廢、想抱抱,哪都不去",
        ],
        "action": lambda: print("→ 跟阿弘待在家享受兩個人的空間 🏠"),
    },
]


# ===== 2. 語意 switch 本體 ===============================================
class SemanticSwitch:
    # threshold：最高分要 >= 這個值才算「有命中」,否則走 default（哪都不去）。
    # margin   ：第 1 名要比第 2 名「至少領先這麼多」,否則算「兩個都很像→不確定」。
    #            （這就是傳統 switch 做不到的「模糊地帶處理」）
    def __init__(self, cases, threshold=0.45, margin=0.05):
        self.embedder = Embedder()  # 載入 bge-m3 進記憶體（只做一次,很重）
        self.cases = cases
        self.threshold = threshold
        self.margin = margin

        # 把每個 case 的所有 examples 攤平成一個大 list 一次 encode（比較有效率）,
        # 同時記住「第 i 個向量屬於哪個 case」,等下才知道分數要歸給誰。
        all_texts, self.owner = [], []
        for ci, c in enumerate(cases):
            for ex in c["examples"]:
                all_texts.append(ex)
                self.owner.append(ci)
        # shape: (所有範例句數, 1024),且已 L2 正規化（Embedder 內設定好了）
        self.example_vecs = self.embedder.encode(all_texts)
        self.owner = np.array(self.owner)

    def _score_each_case(self, user_input: str) -> np.ndarray:
        """回傳每個 case 的分數（= 該 case 範例中「最像的那一句」的相似度）。"""
        q = self.embedder.encode_one(user_input)  # 用戶輸入 → 向量
        # 向量都正規化過 → cosine 相似度 = 點積。一次算出對「每一句範例」的相似度。
        sims = self.example_vecs @ q  # shape: (所有範例句數,)

        # 把句子層級的分數,收斂成 case 層級的分數。
        # 用 max（取最像的一句）而非 mean（平均）：因為只要使用者的話「像其中一種講法」
        # 就該命中,不該被同 case 其他不像的講法拉低平均。
        scores = np.full(len(self.cases), -1.0)
        for ci in range(len(self.cases)):
            scores[ci] = sims[self.owner == ci].max()
        return scores

    def route(self, user_input: str):
        scores = self._score_each_case(user_input)
        order = np.argsort(scores)[::-1]  # 由高到低排序的 case 索引
        best, second = order[0], order[1] if len(order) > 1 else None

        # 印出每個 case 的分數,方便你調 threshold / margin（debug 用,正式可拿掉）
        debug = "  ".join(f"{self.cases[i]['label']}={scores[i]:.2f}" for i in order)
        print(f"  〔分數〕{debug}")

        # 判斷一：最高分都不夠 → default
        if scores[best] < self.threshold:
            print("→ 哪都不去 🤷（沒有夠接近的 case）")
            return None

        # 判斷二：第 1、2 名太接近 → 不確定,不要硬猜（傳統 switch 沒有這層保護）
        if second is not None and (scores[best] - scores[second]) < self.margin:
            print(
                f"→ 不太確定 🤔（「{self.cases[best]['label']}」和"
                f"「{self.cases[second]['label']}」太接近,要不要講清楚一點？）"
            )
            return None

        # 命中：執行對應動作
        self.cases[best]["action"]()
        return self.cases[best]["label"]


# ===== 3. 入口：互動模式 or 單句模式 ======================================
def main():
    sw = SemanticSwitch(CASES)

    if len(sys.argv) > 1:  # 有帶參數 → 只判斷這一句
        sw.route(" ".join(sys.argv[1:]))
        return

    print("\n語意 switch 已就緒（輸入空白或 q 離開）\n")
    while True:
        try:
            text = input("你說> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() == "q":
            break
        sw.route(text)


if __name__ == "__main__":
    main()
