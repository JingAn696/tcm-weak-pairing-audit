"""
KG 关键词命中率验证（splits_v2 train.csv，N=35,345）
======================================================

输入：已校准的 models/kg_keywords.py
输出：每个可匹配节点 × 每个关键词的 阳性命中数 / 否定拦截数 / 例句
用法：python data/gen_kg_calibration_sheet.py
"""

import sys
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.kg_keywords import MATCHABLE_KEYWORDS, MATCHABLE_NODES, _is_negated


def count_kw(text: str, kw: str):
    """统计词 kw 在 text 中：阳性命中数 / 否定拦截数（与 _is_negated 同源）"""
    pos, neg = 0, 0
    i = 0
    while True:
        i = text.find(kw, i)
        if i < 0:
            break
        if _is_negated(text, i):
            neg += 1
        else:
            pos += 1
        i += 1
    return pos, neg


def best_examples(texts, kw, top_n=2):
    """返回最相关的若干例句（含上下文 15 字）"""
    out = []
    for t in texts:
        for i in range(len(t) - len(kw) + 1):
            if t[i:i+len(kw)] == kw and not _is_negated(t, i):
                lo, hi = max(0, i - 15), min(len(t), i + len(kw) + 15)
                ctx = t[lo:hi].replace("\n", " ")
                out.append(ctx)
                break
        if len(out) >= top_n:
            break
    return out


def main():
    csv_path = _PROJECT_ROOT / "data" / "splits_v2" / "train.csv"
    df = pd.read_csv(csv_path).fillna("")
    # 拼接样本文本（与训练一致）
    parts = [df["chief_complaint"], df["description"], df["detection"]]
    texts = [f"{a}。{b}。{c}".strip("。") for a, b, c in zip(*parts)]
    n = len(texts)
    print(f"splits_v2/train.csv: N={n}")

    # 统计每节点的总命中样本数（任一关键词阳性 = 命中）
    samples_hit_per_node = {node: 0 for node in MATCHABLE_NODES}
    total_hits_per_node = {node: 0 for node in MATCHABLE_NODES}
    total_intercepted_per_node = {node: 0 for node in MATCHABLE_NODES}
    kw_stats = {node: {} for node in MATCHABLE_NODES}

    for node in MATCHABLE_NODES:
        for kw in MATCHABLE_KEYWORDS[node]:
            p, k = 0, 0
            for t in texts:
                pos, neg = count_kw(t, kw)
                p += pos
                k += neg
            kw_stats[node][kw] = (p, k)

    for t in texts:
        for node in MATCHABLE_NODES:
            hit, intercepted = False, False
            for kw in MATCHABLE_KEYWORDS[node]:
                pos, neg = count_kw(t, kw)
                if pos > 0:
                    hit = True
                if neg > 0:
                    intercepted = True
            if hit:
                samples_hit_per_node[node] += 1
                total_hits_per_node[node] += 1  # 每节点只记一次命中样本（粗略）
            if intercepted:
                total_intercepted_per_node[node] += 1

    # 输出汇总
    print(f"\n{'节点':25s} {'命中率':>8s} {'命中样本':>10s} {'否定拦截':>10s}")
    print("-" * 60)
    rows = sorted(samples_hit_per_node.items(), key=lambda x: -x[1])
    for node, cnt in rows:
        rate = cnt / n * 100
        inter = total_intercepted_per_node[node]
        marker = "📈" if rate >= 5 else ("📊" if rate >= 1 else ("⚠️ " if rate >= 0.05 else "☠️"))
        print(f"{marker} {node:23s} {rate:7.3f}% {cnt:10d} {inter:10d}")

    print(f"\n{'='*60}")
    print(f"全局：平均命中率 = {sum(samples_hit_per_node.values()) / (n * len(MATCHABLE_NODES)) * 100:.3f}%")
    print(f"     命中 ≥1 节点的样本数 = {sum(1 for t in texts if any(count_kw(t, kw)[0] > 0 for n in MATCHABLE_NODES for kw in MATCHABLE_KEYWORDS[n]))} / {n}")


if __name__ == "__main__":
    main()
