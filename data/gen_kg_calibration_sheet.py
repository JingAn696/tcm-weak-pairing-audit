"""
KG 关键词校准工作表生成器（一次性工具）
========================================
从 splits_v2/train.csv 统计每个节点、每个关键词的：
  - 命中样本数 / 命中率
  - 被否定检测拦截的次数（阴性描述占比）
  - 3 条真实命中例句（截取关键词上下文 ±15 字）

输出：notes/kg-keyword-calibration.md（领域专家逐节点审校用）
"""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from collections import defaultdict

from models.kg_keywords import (
    MATCHABLE_KEYWORDS, CHIEF_COMPLAINT_KEYWORDS, OBSERVATION_KEYWORDS,
    _is_negated,
)


def context(text: str, kw: str, width: int = 15) -> str:
    i = text.find(kw)
    if i < 0:
        return ""
    lo, hi = max(0, i - width), min(len(text), i + len(kw) + width)
    return ("…" if lo > 0 else "") + text[lo:hi] + ("…" if hi < len(text) else "")


def main():
    df = pd.read_csv(_PROJECT_ROOT / "data" / "splits_v2" / "train.csv").fillna("")
    texts = (df["chief_complaint"] + "。" + df["description"] + "。" + df["detection"]).tolist()
    n = len(texts)
    print(f"train 样本数: {n}")

    lines = []
    lines.append("# KG 关键词校准工作表（领域专家审校用）\n")
    lines.append(f"> 数据源：splits_v2/train.csv（{n} 条）。生成时间：2026-09-11\n")
    lines.append("> 审校尺度：**宁可漏报，不可误报**（漏了有文本分支兜底，误报直接污染 KG 特征）\n")
    lines.append("> 标注方式：在每节点后的「领域专家意见」处写 保留/删词/加词/改节点\n")

    for group_name, kw_dict in [("一、主诉节点（20 个）", CHIEF_COMPLAINT_KEYWORDS),
                                ("二、舌象观察节点（14 个）", OBSERVATION_KEYWORDS)]:
        lines.append(f"\n## {group_name}\n")
        for node, kws in kw_dict.items():
            node_hits = 0
            kw_stats = []
            examples = []
            for kw in kws:
                pos, neg, ex = 0, 0, []
                for t in texts:
                    start = 0
                    found_pos = False
                    while True:
                        i = t.find(kw, start)
                        if i < 0:
                            break
                        if _is_negated(t, i):
                            neg += 1
                        else:
                            if not found_pos:
                                pos += 1
                                found_pos = True
                                if len(ex) < 2:
                                    ex.append(context(t, kw))
                            break  # 每样本只计一次阳性命中
                        start = i + 1
                kw_stats.append((kw, pos, neg, ex))
                node_hits += pos
                examples.extend(ex)

            lines.append(f"\n### `{node}` — 命中 {node_hits} 样本（{node_hits/n*100:.1f}%）\n")
            lines.append("| 关键词 | 阳性命中 | 否定拦截 | 例句 |")
            lines.append("|---|---|---|---|")
            for kw, pos, neg, ex in kw_stats:
                ex_str = "<br>".join(ex) if ex else "—"
                lines.append(f"| {kw} | {pos} | {neg} | {ex_str} |")
            lines.append("\n**领域专家意见**：\n")

    out = _PROJECT_ROOT.parent / "notes" / "kg-keyword-calibration.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
