"""
TCM-Tongue bbox 尺寸诊断
=========================

目的：判断「按 bbox 逐框训练（detection-level supervision）」是否值得做。

背景：当前视觉分支把每张图的多个 bbox 合并成一个整舌裁剪 + 图像级
multi-hot 标签（5,594 训练样本）。若各类 bbox 本身是"局部小框"，
改成"每框一个样本"能把有效样本数提到 14,677，且每框都是该病征的
紧致裁剪 → 稀有类（齿痕/紫舌/分区类）可能受益。

判据：
  · 某类 bbox 的 median 相对面积 << 全图 → 该框是"局部框"，逐框训练有意义
  · 某类 bbox 的 median 相对面积 ≈ 全身/整舌 → 逐框训练≈整图训练，无意义

用法：
    python data/gen_bbox_size_diag.py --tongue_root "D:/.../shezhenv3-coco"
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

# 复用加载器的路径探测（单一事实来源，避免两处各写一份路径）
_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
from data.tongue_coco_loader import DEFAULT_TONGUE_ROOT  # noqa: E402

# 三个 split 的 classes.txt 顺序不一致（train 把 xinfeitu/xinfeiao 写反），
# 所以一律按 COCO json 里的 categories 名称来索引，不用 classes.txt。
SPLITS = ("train", "val", "test")


def load_coco(split_dir: Path):
    ann_dir = split_dir / "annotations"
    cands = sorted(ann_dir.glob("*.json"))
    if not cands:
        raise FileNotFoundError(f"{ann_dir} 下没有 json")
    # 取名字里含 instances / train / val / test 的那个，否则取最大的
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    with open(cands[0], encoding="utf-8") as f:
        return json.load(f), cands[0].name


# 默认路径来自 tongue_coco_loader.resolve_tongue_root()（多候选自动探测）。
# 2026-09-14：净安本机数据集已恢复；注意中间目录是 shezhenv3_coco（下划线）。


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tongue_root", default=DEFAULT_TONGUE_ROOT)
    args = ap.parse_args()
    root = Path(args.tongue_root)
    print(f"数据根目录：{root}  (exists={root.is_dir()})")

    # 汇总容器：cat_name -> list[相对面积]
    rel_area = {}
    # 图像级：union 相对面积
    cov = []
    n_ann_per_img = []
    cat_total = {}

    for split in SPLITS:
        sd = root / split
        if not sd.exists():
            print(f"  [skip] {sd} 不存在")
            continue
        coco, fname = load_coco(sd)
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {im["id"]: im for im in coco["images"]}
        print(f"[{split}] {fname} | 图 {len(imgs)} | 标注 {len(coco['annotations'])} | "
              f"类别 {len(cats)}")

        per_img = {}
        for a in coco["annotations"]:
            im = imgs.get(a["image_id"])
            if im is None:
                continue
            iw, ih = im.get("width"), im.get("height")
            if not iw or not ih:
                continue
            x, y, w, h = a["bbox"]
            r = (w * h) / float(iw * ih)
            name = cats.get(a["category_id"], f"id{a['category_id']}")
            rel_area.setdefault(name, []).append(r)
            cat_total[name] = cat_total.get(name, 0) + 1
            per_img.setdefault(a["image_id"], []).append((x, y, w, h, iw, ih))

        for iid, boxes in per_img.items():
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[0] + b[2] for b in boxes)
            y1 = max(b[1] + b[3] for b in boxes)
            iw, ih = boxes[0][4], boxes[0][5]
            cov.append(((x1 - x0) * (y1 - y0)) / float(iw * ih))
            n_ann_per_img.append(len(boxes))

    print("\n" + "=" * 88)
    print("每类 bbox 的相对面积（bbox 面积 / 原图面积）")
    print("  读法：median 越小 → 该病征标注在局部小区域 → 逐框训练越有意义")
    print("=" * 88)
    print(f"{'类别':<12s} {'标注数':>7s} {'p10':>8s} {'median':>8s} {'p90':>8s} {'max':>8s}")
    print("-" * 88)
    rows = []
    for name, vals in rel_area.items():
        vals_s = sorted(vals)
        n = len(vals_s)
        p10 = vals_s[max(0, int(0.10 * n) - 1)]
        med = st.median(vals_s)
        p90 = vals_s[min(n - 1, int(0.90 * n))]
        rows.append((med, name, n, p10, p90, vals_s[-1]))
    # 按 median 面积升序（越局部越靠前）
    for med, name, n, p10, p90, mx in sorted(rows):
        print(f"{name:<12s} {n:7d} {p10:8.4f} {med:8.4f} {p90:8.4f} {mx:8.4f}")

    print("\n" + "=" * 88)
    print("图像级：所有 bbox 并集的相对面积（= 当前 --crop_bbox 实际裁到的范围）")
    print("=" * 88)
    cov_s = sorted(cov)
    n = len(cov_s)
    print(f"  图像数 {n} | 平均 bbox 数/图 {sum(n_ann_per_img)/max(n,1):.2f}")
    print(f"  union 相对面积：p10={cov_s[int(0.10*n)]:.4f}  median={st.median(cov_s):.4f}  "
          f"p90={cov_s[int(0.90*n)]:.4f}")

    print("\n【判据速览】")
    small = [r for r in sorted(rows) if r[0] < 0.15]
    mid = [r for r in sorted(rows) if 0.15 <= r[0] < 0.45]
    big = [r for r in sorted(rows) if r[0] >= 0.45]
    print(f"  median 面积 < 15%  （局部小框，逐框训练最受益）: "
          f"{[r[1] for r in small]}")
    print(f"  median 面积 15~45%（中等）                    : {[r[1] for r in mid]}")
    print(f"  median 面积 >= 45%（接近整舌/全身，逐框≈整图）: {[r[1] for r in big]}")


if __name__ == "__main__":
    main()
