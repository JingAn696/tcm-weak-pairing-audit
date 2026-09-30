"""
TCM-Tongue 标注结构诊断（bbox 尺寸 + 裁剪保留率 + 框间重合 + 唯一区域）
=======================================================================

回答四个决定"视觉分支该怎么做"的问题：

  [1] 每类 bbox 有多大？（相对原图面积）
  [2] 同一病征在 整图 / union 裁剪 / 逐框 三种输入下各占几个 ViT patch？
  [3] 不同类的框之间有多重合？（决定"能不能逐框单标签"）
  [4] 去掉重复框后到底有多少个**唯一区域**？每个区域该带几个标签？

背景（2026-09-14 实测）
----------------------
本数据集的 bbox 呈**双峰**，两类量级差 3 个数量级：

  · 局部病变框：红点舌 0.02%（12×14 px）、裂纹舌 1.8%、齿痕舌 2.1%、各舌面分区框
                标注的是**单个病征**（一张图里有好几个框）
  · 整舌框    ：胖大舌 45%、黄苔舌 44%、白苔舌 34%、红舌 36%
                标注的是**整个舌体**

⚠️ 关键结构事实：**不同类的整舌框往往是同一个矩形**（跨类框对 IoU 中位数 = 1.00）。
   即数据集实质上是"**区域级多标签**"标注 —— 一个区域（整舌 / 某个分区 / 某个病变点）
   同时具有若干舌象特征，作者把它按类拆成了多份相同的框。
   这决定了逐框训练**不能**用单标签，必须用"区域覆盖"式的多标签目标。

指标定义
--------
  a = 某类 bbox 中位相对面积；u = 其所在图 union 中位相对面积（原图面积 = 1）
  retention  = a / u                  该病征在 union 裁剪图中的面积占比
  patch_crop = 14 * sqrt(retention)   union 裁剪 resize 到 224 后覆盖的 patch 边长
  patch_full = 14 * sqrt(a)           整图 resize 到 224 后覆盖的 patch 边长
  逐框裁剪恒为 14（每框铺满 224×224）        （224/16 = 14）

标签准则（[4]）——「区域覆盖」
------------------------------
裁剪区域 r 的标签集 = { c : IoA(box_c ∩ r) / area(box_c) >= 0.8 }
即"该裁剪把类 c 的标注区域基本覆盖住了"。

  · r = union（当前做法）→ 所有框都在其中 ⇒ 标签集 = 图像级 multi-hot
    ★ 与现有实现**完全等价**，因此新方案是它的严格推广（向后兼容）
  · r = 某个局部病变框 → 只有落在它内部的小框被覆盖 ⇒ 标签集小而纯
  · r = 某个整舌框 → 其内含的所有小框都被覆盖 ⇒ 标签集 = 全图标签

用法
----
    python data/gen_bbox_structure_diag.py
    python data/gen_bbox_structure_diag.py --out ../notes/bbox-structure-diag.md
    python data/gen_bbox_structure_diag.py --samples 5      # 多打几个样例
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics as st
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data.tongue_coco_loader import DEFAULT_TONGUE_ROOT  # noqa: E402
from models.tongue_label_mapping import (  # noqa: E402
    TONGUE_CLASSES,
    TONGUE_PINYIN_TO_ID,
)

SPLITS = ("train", "val", "test")
IMG_SIZE = 224
PATCH = 16
GRID = IMG_SIZE // PATCH                 # 14
IOU_MERGE = 0.8                          # IoU >= 该值视为"同一个区域的重复框"
COVER_TAU = 0.8                          # 区域覆盖阈值
WHOLE_THRESH = 0.15                      # 相对面积 >= 该值视为"整舌级框"

PINYIN_ZH = {c["pinyin"]: c["zh"] for c in TONGUE_CLASSES}
CATEGORY_ID_TO_PINYIN = {v: k for k, v in TONGUE_PINYIN_TO_ID.items()}


def load_coco(split_dir: Path):
    cands = sorted((split_dir / "annotations").glob("*.json"),
                   key=lambda p: p.stat().st_size, reverse=True)
    if not cands:
        raise FileNotFoundError(f"{split_dir / 'annotations'} 下没有 json")
    with open(cands[0], encoding="utf-8") as f:
        return json.load(f), cands[0].name


def inter_area(b1, b2):
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[0] + b1[2], b2[0] + b2[2])
    y1 = min(b1[1] + b1[3], b2[1] + b2[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def iou(b1, b2):
    inter = inter_area(b1, b2)
    if inter <= 0:
        return 0.0
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def region_match(b1, b2):
    """一方基本覆盖另一方（用于 [3] 的重合分析）。"""
    inter = inter_area(b1, b2)
    if inter <= 0:
        return 0.0
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    if a1 <= 0 or a2 <= 0:
        return 0.0
    return max(inter / a1, inter / a2)


def cover_ioa(crop, box):
    """IoA：box 有多大比例落在 crop 内（用于 [4] 的标签准则）。"""
    a = box[2] * box[3]
    return inter_area(crop, box) / a if a > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="TCM-Tongue 标注结构诊断")
    ap.add_argument("--tongue_root", default=DEFAULT_TONGUE_ROOT)
    ap.add_argument("--out", default="", help="可选：输出 markdown 报告路径")
    ap.add_argument("--samples", type=int, default=3, help="样例图 dump 数量")
    args = ap.parse_args()
    root = Path(args.tongue_root)
    print(f"数据根目录：{root}  (exists={root.is_dir()})")

    acc: dict = {}
    n_img_total = 0
    n_box_total = 0
    global_union = []
    diff_pair_iou = []
    diff_pair_rm = []
    # [4]
    n_region_total = 0
    n_region_by_split = {}
    region_lab_all, region_lab_whole, region_lab_local = [], [], []
    region_w_all, region_h_all = [], []
    sample_bank = []

    for split in SPLITS:
        sd = root / split
        if not sd.is_dir():
            print(f"  [skip] {sd} 不存在")
            continue
        coco, fname = load_coco(sd)
        imgs = {im["id"]: im for im in coco["images"]}
        print(f"[{split}] {fname} | 图 {len(imgs)} | 标注 {len(coco['annotations'])}")

        per_img: dict = {}
        for a in coco["annotations"]:
            im = imgs.get(a["image_id"])
            if im is None:
                continue
            iw, ih = im.get("width"), im.get("height")
            if not iw or not ih:
                continue
            pid = CATEGORY_ID_TO_PINYIN.get(a["category_id"])
            if pid is None:
                continue                        # 越界 id（如 test 中 7 个 category_id=21）
            x, y, w, h = a["bbox"]
            rel = (w * h) / float(iw * ih)
            d = acc.setdefault(pid, {"box_rel": [], "w": [], "h": [], "n": 0})
            d["box_rel"].append(rel)
            d["w"].append(w)
            d["h"].append(h)
            d["n"] += 1
            n_box_total += 1
            rec = per_img.setdefault(a["image_id"], {"wh": (iw, ih), "boxes": []})
            rec["boxes"].append((x, y, w, h, pid))

        for iid, rec in per_img.items():
            boxes = rec["boxes"]
            iw, ih = rec["wh"]
            if not boxes:
                continue
            xs0 = min(b[0] for b in boxes)
            ys0 = min(b[1] for b in boxes)
            xs1 = max(b[0] + b[2] for b in boxes)
            ys1 = max(b[1] + b[3] for b in boxes)
            union_box = (xs0, ys0, xs1 - xs0, ys1 - ys0)
            u = (union_box[2] * union_box[3]) / float(iw * ih)
            global_union.append(u)
            n_img_total += 1
            img_area = float(iw * ih)
            n = len(boxes)

            # --- [3] 不同类框对的重合 ---
            for i in range(n):
                for j in range(i + 1, n):
                    if boxes[i][4] == boxes[j][4]:
                        continue
                    diff_pair_iou.append(iou(boxes[i][:4], boxes[j][:4]))
                    diff_pair_rm.append(region_match(boxes[i][:4], boxes[j][:4]))

            # --- [4] 去重：IoU 聚类成"唯一区域" ---
            parent = list(range(n))

            def find(x, p=parent):
                while p[x] != x:
                    p[x] = p[p[x]]
                    x = p[x]
                return x

            for i in range(n):
                for j in range(i + 1, n):
                    if iou(boxes[i][:4], boxes[j][:4]) >= IOU_MERGE:
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
            groups: dict = {}
            for i in range(n):
                groups.setdefault(find(i), []).append(i)

            n_region_total += len(groups)
            n_region_by_split[split] = n_region_by_split.get(split, 0) + len(groups)

            # 每个区域的标签集：按「区域覆盖」准则
            for idxs in groups.values():
                # 区域代表框 = 该组框的并集（通常就是同一个框）
                gx0 = min(boxes[k][0] for k in idxs)
                gy0 = min(boxes[k][1] for k in idxs)
                gx1 = max(boxes[k][0] + boxes[k][2] for k in idxs)
                gy1 = max(boxes[k][1] + boxes[k][3] for k in idxs)
                crop = (gx0, gy0, gx1 - gx0, gy1 - gy0)
                labels = {boxes[k][4] for k in range(n)
                          if cover_ioa(crop, boxes[k][:4]) >= COVER_TAU}
                rel = (crop[2] * crop[3]) / img_area
                region_lab_all.append(len(labels))
                region_w_all.append(crop[2])
                region_h_all.append(crop[3])
                if rel >= WHOLE_THRESH:
                    region_lab_whole.append(len(labels))
                else:
                    region_lab_local.append(len(labels))
                if len(sample_bank) < args.samples * 40:
                    sample_bank.append({
                        "split": split, "iid": iid, "wh": (iw, ih),
                        "crop": crop, "rel": rel,
                        "labels": sorted(labels), "n_box_in_img": n,
                    })

    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    # ---------------- [1] ----------------
    log("")
    log("=" * 104)
    log("[1] 每类 bbox 的相对面积（bbox 面积 / 原图面积）")
    log("    读法：中位面积越小 → 该病征标注在局部小区域 → 越需要高分辨率")
    log("=" * 104)
    log(f"{'舌象类别':<12s} {'框数':>6s} {'p10':>8s} {'median':>8s} {'p90':>8s} "
        f"{'中位框px':>12s} {'规模':>10s}")
    log("-" * 104)
    rows = []
    for pid, d in acc.items():
        if not d["box_rel"]:
            continue
        v = sorted(d["box_rel"])
        k = len(v)
        rows.append({
            "pinyin": pid, "zh": PINYIN_ZH.get(pid, pid), "n": d["n"],
            "p10": v[max(0, int(0.10 * k) - 1)], "med": st.median(v),
            "p90": v[min(k - 1, int(0.90 * k))],
            "w": st.median(d["w"]), "h": st.median(d["h"]),
        })
    for r in sorted(rows, key=lambda x: x["med"]):
        scale = "整舌级" if r["med"] >= WHOLE_THRESH else "局部病变"
        log(f"{r['zh']:<12s} {r['n']:6d} {r['p10']:8.4f} {r['med']:8.4f} {r['p90']:8.4f} "
            f"{r['w']:5.0f}x{r['h']:<6.0f} {scale:>10s}")

    # ---------------- [2] ----------------
    u_med = st.median(global_union)
    log("")
    log("=" * 104)
    log("[2] 同一病征在三种输入下占几个 ViT patch（14×14 网格）")
    log("=" * 104)
    log(f"全库 {n_img_total} 图 / {n_box_total} 框 | 平均 {n_box_total/max(n_img_total,1):.2f} 框每图")
    log(f"union（所有框并集）相对面积 median = {u_med:.4f}")
    log(f"  → union 裁剪 resize 到 224 后，原图 1% 的面积才 ≈ {GRID*math.sqrt(0.01/u_med):.2f} 个 patch 边长")
    log("")
    log(f"{'舌象类别':<12s} {'框数':>6s} {'中位框面积':>10s} {'保留率':>8s} "
        f"{'整图patch':>9s} {'裁剪patch':>9s} {'逐框patch':>9s}")
    log("-" * 104)
    for r in rows:
        r["ret"] = r["med"] / u_med if u_med > 0 else float("nan")
        r["p_full"] = GRID * math.sqrt(r["med"])
        r["p_crop"] = GRID * math.sqrt(r["ret"])
    for r in sorted(rows, key=lambda x: x["p_crop"]):
        log(f"{r['zh']:<12s} {r['n']:6d} {r['med']:10.4f} {r['ret']:8.4f} "
            f"{r['p_full']:9.2f} {r['p_crop']:9.2f} {GRID:9.0f}")

    hopeless = [r for r in rows if r["p_crop"] < 2.0]
    marginal = [r for r in rows if 2.0 <= r["p_crop"] < 6.0]
    ok = [r for r in rows if r["p_crop"] >= 6.0]
    log("")
    log("【判据速览】")
    log(f"  ✗ 裁剪后仍 <2 patch（crop 消融对此类天然无效） : {[r['zh'] for r in hopeless]}")
    log(f"  ~ 裁剪后 2~6 patch（勉强）                     : {[r['zh'] for r in marginal]}")
    log(f"  ✓ 裁剪后 >=6 patch（crop 已够用）              : {[r['zh'] for r in ok]}")

    # ---------------- [3] ----------------
    log("")
    log("=" * 104)
    log("[3] 不同类框之间的重合 —— 数据集实为「区域级多标签」")
    log("=" * 104)
    if diff_pair_iou:
        iv = sorted(diff_pair_iou)
        rv = sorted(diff_pair_rm)
        k = len(iv)
        log(f"跨类框对共 {k} 对：")
        log(f"  IoU（交集/并集）      median={st.median(iv):.4f}  p90={iv[int(0.9*k)]:.4f}  "
            f"= 1.0 的占比 {sum(1 for x in iv if x >= 0.999)/k:.1%}")
        log(f"  region_match（覆盖）  median={st.median(rv):.4f}  p90={rv[int(0.9*k)]:.4f}  "
            f">= {COVER_TAU} 的占比 {sum(1 for x in rv if x >= COVER_TAU)/k:.1%}")
        log("")
        log("  ⇒ IoU = 1.00 占多数，说明【不同类的整舌框是同一个矩形】：")
        log("     白苔舌 / 红舌 / 胖大舌 / 黄苔舌 … 共用同一个整舌框，只是按类重复标注。")
        log("     ⇒ 本数据集本质是 **区域级多标签**（一个区域带若干特征标签），")
        log("       而不是「每类一块各自的区域」。")
        log("     ⇒ 因此逐框训练**不能**用单标签（同图两个类会给同一张裁剪图互斥标签），")
        log("       必须用「区域覆盖」式的多标签目标。")

    # ---------------- [4] ----------------
    log("")
    log("=" * 104)
    log("[4] 去重后的唯一区域 & 每个区域的标签集")
    log("=" * 104)
    by_split = "  ".join(f"{s}={n_region_by_split.get(s, 0)}" for s in SPLITS)
    log(f"原始框数            ：{n_box_total}")
    log(f"去重后唯一区域数    ：{n_region_total}   （{by_split}）")
    log(f"  压缩比            ：{n_box_total/max(n_region_total,1):.2f} 框 → 1 区域；"
        f"样本量相对整图训练 {n_region_total/max(n_img_total,1):.2f}×")
    log("")
    log("按「区域覆盖」准则（IoA >= 0.8）的标签集大小：")
    for tag, pool in (("全部区域", region_lab_all), ("整舌级区域", region_lab_whole),
                      ("局部病变区域", region_lab_local)):
        if not pool:
            continue
        mu = sum(pool) / len(pool)
        single = sum(1 for x in pool if x == 1) / len(pool)
        log(f"  {tag:<10s} n={len(pool):6d} | 平均标签数 {mu:.2f} | 中位 {st.median(pool):.0f} "
            f"| max {max(pool)} | 单标签占比 {single:.1%}")
    log(f"\n区域中位尺寸：{st.median(region_w_all):.0f}×{st.median(region_h_all):.0f} px")

    # ---------------- 样例 ----------------
    log("")
    log("=" * 104)
    log("[5] 样例核对（人工确认「同框多标签」这个判断）")
    log("=" * 104)
    seen_kinds = set()
    shown = 0
    for s in sample_bank:
        kinds = "whole" if s["rel"] >= WHOLE_THRESH else "local"
        if s["rel"] < 0.02:
            kinds = "tiny"
        if kinds in seen_kinds and shown >= args.samples:
            continue
        zh = "、".join(PINYIN_ZH.get(p, p) for p in s["labels"])
        log(f"  [{s['split']} img {s['iid']}] 原图 {s['wh'][0]}×{s['wh'][1]} | "
            f"该图共 {s['n_box_in_img']} 框 | 区域 {s['crop'][2]:.0f}×{s['crop'][3]:.0f} "
            f"(占图 {s['rel']:.2%})")
        log(f"        标签集({len(s['labels'])}): {zh}")
        seen_kinds.add(kinds)
        shown += 1
        if shown >= args.samples * 3:
            break

    # ---------------- 结论 ----------------
    log("")
    log("=" * 104)
    log("【论文可直接引用的结论】")
    log("=" * 104)
    log(f"  1. 本数据集 bbox 呈双峰：局部病变框（红点最小 {min(r['med'] for r in rows):.2%}，"
        f"12×14 px）与整舌框（胖大最大 {max(r['med'] for r in rows):.0%}）。")
    log(f"  2. 不同类的整舌框共用同一矩形（跨类 IoU 中位 1.00）⇒ 实为区域级多标签标注。")
    log(f"  3. 「union 裁剪」这一档消融存在硬上限：union 中位面积 {u_med:.1%} 被整舌框撑满，")
    log(f"     局部病变框在裁剪图中只从 {min(r['p_full'] for r in rows):.2f} patch 提到 "
        f"{min(r['p_crop'] for r in rows):.2f} patch —— 这解释了裁剪只涨 0.0208。")
    log(f"  4. 正确粒度 = **区域级弱监督**：{n_img_total} 图 → {n_region_total} 个唯一区域"
        f"（{n_region_total/max(n_img_total,1):.2f}×），")
    log(f"     每区域铺满 224×224（14×14 patch），标签按「区域覆盖」继承。")
    log(f"     当区域取 union 时与现有实现完全等价 ⇒ 向后兼容的严格推广。")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with io.open(outp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n报告已保存：{outp}")


if __name__ == "__main__":
    main()
