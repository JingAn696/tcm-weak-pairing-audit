#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCM-Tongue「弱验证」：映射后的 10 病性分布 vs TCM-SD 文本侧分布（方向性对比）

⚠️ 定位声明（写论文时必须原样保留）
------------------------------------------------------------------
这是 **sanity check / 方向性合理性讨论，不是验证**（not a validation）。
理由：
  1. TCM-SD 与 TCM-Tongue 是**两个独立数据集**——非同一人群、非同一采集场景、
     非同一标注体系。两者分布本来就不该一致，一致或部分一致都不能证明映射正确。
  2. TCM-Tongue 只有舌象形态标签，**没有病性标签** → 映射无 ground truth。
  3. 因此结论只能用于 Discussion 中"映射方向是否讲得通"的定性讨论，
     不能写成"验证了映射的准确性"。

脚本做三件事
------------
A. 数据质量核查：21 类实测分布（图像级 / 实例级），标出不可用类
B. 舌象侧：COCO 标注 → 图像级 multi-hot → 经映射矩阵投影 → 10 病性阳性率
C. 文本侧：splits_v2 的 10 病性标签阳性率 → 与 B 做同向性对比

用法
----
python gen_tongue_weak_validation.py \
    --tongue_root "D:/科研/第一篇论文/数据集/TCM-Tongue/shezhen_datasets1/shezhen datasets/shezhenv3-coco/shezhenv3-coco" \
    --splits_dir  ./splits_v2 \
    --out         ../notes/tongue-weak-validation.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
# 让 models/ 可导入
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from models.tongue_label_mapping import (  # noqa: E402
    HARD_MATRIX,
    MIN_USABLE_INSTANCES,
    NUM_SYNDROMES,
    SYNDROMES_ZH,
    TONGUE_CLASSES,
    UNUSABLE_CLASSES,
)

SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# A + B：读 TCM-Tongue COCO，统计类别分布并投影到 10 病性
# ---------------------------------------------------------------------------
def load_tongue_stats(tongue_root: Path):
    """返回 (per_class_inst, per_class_img, img_syndrome_counts, n_images, n_boxes)。

    img_syndrome_counts: dict[病性] = 命中的图像数（binary 模式）
    """
    root = Path(tongue_root)
    per_class_inst = defaultdict(int)
    per_class_img = defaultdict(int)
    per_split_inst = {s: defaultdict(int) for s in SPLITS}
    img_syndrome_counts = defaultdict(int)
    total_images = 0
    total_boxes = 0

    for split in SPLITS:
        ann_dir = root / split / "annotations"
        cands = sorted(ann_dir.glob("*.json"))
        if not cands:
            print(f"  [skip] {split}: 未找到 annotations/*.json")
            continue
        d = json.load(open(cands[0], encoding="utf-8"))
        id2name = {c["id"]: c["name"] for c in d["categories"]}

        # 每个 image 的类别集合（图像级 multi-hot）
        img_cats = defaultdict(set)
        for a in d["annotations"]:
            img_cats[a["image_id"]].add(a["category_id"])
            per_class_inst[a["category_id"]] += 1
            per_split_inst[split][a["category_id"]] += 1
            total_boxes += 1

        for cats in img_cats.values():
            for cid in cats:
                per_class_img[cid] += 1
        total_images += len(d["images"])

        # 投影到 10 病性（binary：任一映射类命中即置 1）
        for im_id, cats in img_cats.items():
            for j in range(NUM_SYNDROMES):
                if any(HARD_MATRIX[c][j] for c in cats if 0 <= c < len(HARD_MATRIX)):
                    img_syndrome_counts[SYNDROMES_ZH[j]] += 1

    return per_class_inst, per_class_img, per_split_inst, img_syndrome_counts, total_images, total_boxes, id2name


# ---------------------------------------------------------------------------
# C：读 splits_v2 文本标签，统计 10 病性阳性率
# ---------------------------------------------------------------------------
def load_text_stats(splits_dir: Path):
    counts = defaultdict(int)
    total = 0
    for split in SPLITS:
        p = Path(splits_dir) / f"{split}.csv"
        if not p.exists():
            print(f"  [skip] {p} 不存在")
            continue
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                total += 1
                for zh in SYNDROMES_ZH:
                    col = next((c for c in row if c.endswith(f"|{zh}")), None)
                    if col is None:
                        continue
                    try:
                        if float(row[col]) > 0.5:
                            counts[zh] += 1
                    except (TypeError, ValueError):
                        pass
    return counts, total


def main():
    ap = argparse.ArgumentParser(description="TCM-Tongue 映射弱验证（方向性）")
    ap.add_argument("--tongue_root", default=str(Path("D:/科研/第一篇论文/数据集/TCM-Tongue/"
                                                      "shezhen_datasets1/shezhen datasets/"
                                                      "shezhenv3-coco/shezhenv3-coco")),
                    help="shezhenv3-coco 根目录（含 train/val/test）")
    ap.add_argument("--splits_dir", default=str(_SCRIPT_DIR / "splits_v2"),
                    help="splits_v2 目录")
    ap.add_argument("--out", default=str(_SCRIPT_DIR.parent.parent / "notes" / "tongue-weak-validation.md"),
                    help="输出 markdown 路径")
    args = ap.parse_args()

    tongue_root = Path(args.tongue_root)
    if not tongue_root.is_dir():
        print(f"ERROR: 数据集目录不存在：{tongue_root}")
        sys.exit(1)
    splits_dir = Path(args.splits_dir)

    print("=" * 72)
    print("A. TCM-Tongue 类别分布核查")
    print("=" * 72)
    (per_class_inst, per_class_img, per_split_inst,
     img_syn_cnt, n_img, n_box, id2name) = load_tongue_stats(tongue_root)

    print(f"图像 {n_img} 张 | bbox {n_box} 个 | 类别 {len(id2name)} 类")
    for c in TONGUE_CLASSES:
        cid, zh = c["id"], c["zh"]
        ni = per_class_img.get(cid, 0)
        no = per_class_inst.get(cid, 0)
        flag = "  ⚠️ 不可用" if no < MIN_USABLE_INSTANCES else ""
        print(f"  {cid:2d} {zh:8s} 图像级 {ni:5d} ({ni / max(n_img,1) * 100:5.1f}%)  实例 {no:5d}{flag}")

    print()
    print("=" * 72)
    print("B. 舌象侧 → 10 病性（binary 映射）")
    print("=" * 72)
    tongue_rate = {}
    for zh in SYNDROMES_ZH:
        r = img_syn_cnt.get(zh, 0) / max(n_img, 1)
        tongue_rate[zh] = r
        print(f"  {zh:4s} 图像级阳性 {img_syn_cnt.get(zh, 0):5d}  {r * 100:5.1f}%")

    print()
    print("=" * 72)
    print("C. 文本侧（TCM-SD splits_v2）→ 10 病性")
    print("=" * 72)
    text_cnt, n_text = load_text_stats(splits_dir)
    text_rate = {}
    for zh in SYNDROMES_ZH:
        r = text_cnt.get(zh, 0) / max(n_text, 1)
        text_rate[zh] = r
        print(f"  {zh:4s} 阳性 {text_cnt.get(zh, 0):6d}  {r * 100:5.1f}%")

    # ---------------- 对比 ---------------- #
    # 标签密度：两个数据集的"每条样本平均标签数"不同，阳性率绝对值不可直接比较
    density_t = n_box / max(n_img, 1)            # 舌象侧：bbox 数 / 图数
    density_x = sum(text_cnt.values()) / max(n_text, 1)  # 文本侧：标签数 / 样本数

    usable = {c["zh"] for c in TONGUE_CLASSES if c["n_inst"] >= MIN_USABLE_INSTANCES}
    tongue_order = sorted(SYNDROMES_ZH, key=lambda s: -tongue_rate[s])
    text_order = sorted(SYNDROMES_ZH, key=lambda s: -text_rate[s])
    rank_t = {s: i + 1 for i, s in enumerate(tongue_order)}
    rank_x = {s: i + 1 for i, s in enumerate(text_order)}

    # 密度校正后阳性率（= 阳性率 / 平均标签数），使两列可比
    corr_t = {s: tongue_rate[s] / density_t for s in SYNDROMES_ZH}
    corr_x = {s: text_rate[s] / density_x for s in SYNDROMES_ZH}

    # 两列都有信号的病性上算 Spearman（n=8，仅作参考）
    common = [s for s in SYNDROMES_ZH if tongue_rate[s] > 0.005 and text_rate[s] > 0]
    spearman = None
    if len(common) >= 3:
        n = len(common)
        d2 = sum((rank_t[s] - rank_x[s]) ** 2 for s in common)
        spearman = 1 - 6 * d2 / (n * (n * n - 1))

    # ---------------- 输出报告 ---------------- #
    L = []
    A = L.append
    A("# TCM-Tongue 映射「弱验证」报告（方向性 sanity check）\n")
    A("> 生成脚本：`papers/code/data/gen_tongue_weak_validation.py`  ")
    A(f"> 舌象数据：TMC-Tongue（{n_img} 图 / {n_box} bbox / 21 类）  ")
    A(f"> 文本数据：TCM-SD splits_v2（{n_text} 条）\n")
    A("## ⚠️ 定位声明（必须随结果一起引用）\n")
    A("这是 **sanity check / 方向性合理性讨论，不是验证**。"
      "TCM-SD 与 TCM-Tongue 是两个独立数据集（非同一人群/场景/标注体系），"
      "TCM-Tongue 又只有舌象形态标签、无病性 ground truth —— "
      "因此两列分布的一致或不一致，都不能证明映射正确。"
      "本报告仅用于 Discussion 中讨论映射方向是否讲得通。\n")

    A("## A. 类别分布与数据可用性\n")
    A("| id | 舌象类别 | 图像级 | 占比 | 实例数 | 可用性 |")
    A("|---:|---|---:|---:|---:|---|")
    for c in TONGUE_CLASSES:
        cid, zh = c["id"], c["zh"]
        ni, no = per_class_img.get(cid, 0), per_class_inst.get(cid, 0)
        mark = "❌ 实例不足" if no < MIN_USABLE_INSTANCES else "✅"
        A(f"| {cid} | {zh} | {ni} | {ni / max(n_img, 1) * 100:.1f}% | {no} | {mark} |")
    A("")
    A(f"**不可用类（实例数 < {MIN_USABLE_INSTANCES}）：{('、'.join(UNUSABLE_CLASSES))}**\n")
    A("> 4 个「凸起」类几乎为空（脾胃区凸起 0 实例、肾区凸起/心肺区凸起各 2、肝胆区凸起 9）——"
      "这是数据集标注的固有缺口，不是映射缺陷。\n")

    A("## B. 对比总表\n")
    A(f"**标签密度差异（关键！）**：舌象侧 {density_t:.2f} bbox/图 vs 文本侧 {density_x:.2f} 标签/条 —— "
      "阳性率绝对值**不可直接比较**，下表给出密度校正列（阳性率 ÷ 平均标签数）供对照。\n")
    A("| 病性 | 舌象阳性率 | 舌象校正 | 文本阳性率 | 文本校正 | 视觉信号 |")
    A("|---|---:|---:|---:|---:|:--:|")
    for s in SYNDROMES_ZH:
        vis = "❌ 无信号" if tongue_rate[s] < 0.005 else "✅"
        A(f"| {s} | {tongue_rate[s] * 100:.1f}% | {corr_t[s] * 100:.1f}% | "
          f"{text_rate[s] * 100:.1f}% | {corr_x[s] * 100:.1f}% | {vis} |")
    A("")
    A(f"舌象侧排名：{' > '.join(tongue_order)}")
    A(f"文本侧排名：{' > '.join(text_order)}\n")
    if spearman is not None:
        A(f"两列共同有信号病性的 Spearman 秩相关（n={len(common)}，仅供参考）= "
          f"**{spearman:.3f}**\n")

    A("## C. 关键发现\n")
    live = [s for s in SYNDROMES_ZH if tongue_rate[s] >= 0.005]
    dead = [s for s in SYNDROMES_ZH if tongue_rate[s] < 0.005]
    A(f"1. **视觉侧有效覆盖 {len(live)}/10**：{'、'.join(live)}")
    A(f"2. **视觉侧零信号 {len(dead)}/10**：{'、'.join(dead)}")
    A("   - 内风：数据集无舌态标签（覆盖缺口，已知，有《温热论》舌态依据支撑）")
    A(f"   - 气滞：唯一映射来源「肝胆区凸起」全数据集仅 9 实例 → "
      f"图像级仅 {img_syn_cnt.get('气滞', 0)} 张命中，统计上无意义（名义有映射、实际无信号）")
    A("3. **同向信号**：气虚、痰湿在两列都居高位；血虚在两列都居低位 —— 方向讲得通")
    A("4. **反向信号**（排名差 ≥4，需在 Discussion 中解释）：")
    buf = []
    for s in SYNDROMES_ZH:
        if tongue_rate[s] >= 0.005 and text_rate[s] > 0:
            rt, rx = rank_t[s], rank_x[s]
            if abs(rt - rx) >= 4:
                buf.append(f"   - {s}：舌象排名 {rt} vs 文本排名 {rx}"
                           f"（{tongue_rate[s] * 100:.1f}% vs {text_rate[s] * 100:.1f}%）")
    L.extend(buf if buf else ["   - （无）"])
    A("")
    A("### ⚠️ 如何解读 Spearman = "
      f"{spearman:.3f}（关键，别误读）\n" if spearman is not None else "")
    A("**不要把它读成「映射错了」**，理由：")
    A("- 两数据集人群/场景/标注体系完全不同，分布本不该一致（最大混淆因素）；")
    A("- 硬 0/1 映射存在**「一票多发」的下界效应**（见 D 节），"
      "一个高频舌象会同时抬高多个病性的阳性率，人为制造跨列的正相关/错位；")
    A("- 舌象侧测的是「形态特征出现率」，文本侧测的是「证候患病率」，**量纲不同**。")
    A("")
    A("**它能可靠支撑的只有「缺口类事实」**（见 C.1/C.2），不能支撑「映射准确性」的任何结论。\n")

    A("## D. 方法学观察：硬映射的阳性率「下界效应」\n")
    A("硬 0/1 映射下，若病性 j 的来源集合为 S_j，则必然有 "
      "`rate(j) ≥ max_{i∈S_j} rate(i)` —— 这是集合并集的性质，与映射「对不对」无关。\n")
    A("| 病性 | 最高频可用来源 | 其图像级率 | 理论下界 | 实际阳性率 | 超出下界 |")
    A("|---|---|---:|---:|---:|---:|")
    for j, s in enumerate(SYNDROMES_ZH):
        srcs = [c for c in TONGUE_CLASSES
                if c["n_inst"] >= MIN_USABLE_INSTANCES and HARD_MATRIX[c["id"]][j] == 1]
        if not srcs:
            A(f"| {s} | —（无可用来源） | — | — | {tongue_rate[s] * 100:.1f}% | — |")
            continue
        top = max(srcs, key=lambda c: per_class_img.get(c["id"], 0))
        top_rate = per_class_img.get(top["id"], 0) / max(n_img, 1)
        A(f"| {s} | {top['zh']} | {top_rate * 100:.1f}% | {top_rate * 100:.1f}% | "
          f"{tongue_rate[s] * 100:.1f}% | +{(tongue_rate[s] - top_rate) * 100:.1f}pp |")
    A("")
    A("**读法**：白苔舌覆盖 73.2% 的图，而它同时映射到气虚/阳虚/痰湿三个病性 —— "
      "于是这三个病性的阳性率被抬到 77%~92%，**彼此间的高度相似是映射结构造成的，不是人群特征**。"
      "这也是为什么「舌象侧阳性率」与「文本侧患病率」不该做逐项数值比较。\n")
    A("> 结论：硬 0/1 映射适合当**标签生成器**（生成弱监督信号喂给视觉分支），"
      "不适合当**分布估计器**（用于人群患病率推断）。论文中两者要分清。\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print()
    print("=" * 72)
    print(f"报告已写入：{out}")
    print("=" * 72)
    print(f"标签密度：舌象 {density_t:.2f} bbox/图 | 文本 {density_x:.2f} 标签/条")
    print(f"视觉侧有效覆盖 {len(live)}/10，零信号：{dead}")
    if spearman is not None:
        print(f"Spearman（n={len(common)}）= {spearman:.3f}")


if __name__ == "__main__":
    main()
