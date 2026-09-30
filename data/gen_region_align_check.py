"""
区域级 trainer 的「评估管线对齐」本机验证（不需要 GPU / 不需要模型权重）
=========================================================================

**被验的假设**（train_tongue_region.py 的核心依赖，错了会静默出错而不是报错）：

1. 区域 loader 返回的 `file_name` ⊆ 图像级 dataset 的 `file_name`
   —— 否则 `aggregate_by_image` 里 `name_to_cats.get(fn, ())` 会静默取到空元组，
      导致「图像级-聚合」的 y_true 全 0，指标看起来"崩了"但没有任何报错。
2. 区域标签 ⊆ 同图图像级标签
   —— 否则聚合评估的 GT 与训练监督信号口径不一致。
3. `aggregate_by_image` 的 max 池化在真实 file_name 上语义正确
   —— 用可验证的构造分数（每图至少一个区域给高分）反推应得指标。
4. `img_wh` 覆盖所有区域所属图（`region_type_stats` 依赖）。

用法
----
    python data/gen_region_align_check.py
    python data/gen_region_align_check.py --splits test        # 只想查 test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _SCRIPT_DIR.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import numpy as np

from data.tongue_coco_loader import TongueCOCODataset, resolve_tongue_root
from data.tongue_region_loader import TongueRegionDataset, DEFAULT_MIN_REGION_PX
from utils.run_handoff import Handoff

# 运行交接摘要（2026-09-15）
H = Handoff("gen_region_align_check")


def check_split(root, split: str) -> dict:
    print(f"\n{'=' * 74}\n[{split}] 构建两个粒度的数据集\n{'=' * 74}")
    img_ds = TongueCOCODataset(root=root, split=split, preprocess=None, verbose=True)
    reg_ds = TongueRegionDataset(
        root=root, split=split, preprocess=None,
        crop_margin=0.15, min_region_px=DEFAULT_MIN_REGION_PX, verbose=True,
    )

    r = {}

    # ---- 假设 1：file_name 命名空间一致 ----
    img_names = {fn for fn, _c, _b in img_ds.samples}
    reg_names = {fn for fn, _crop, _l in reg_ds.regions}
    outside = reg_names - img_names
    r["assumption_1_ok"] = not outside
    print(f"\n[1] file_name 命名空间")
    print(f"    图像级 {len(img_names)} 图 / 区域级覆盖 {len(reg_names)} 图")
    print(f"    区域图中不在图像级集合里的：{len(outside)}")
    if outside:
        print(f"    ✗ 反例（最多 5 个）：{sorted(outside)[:5]}")
    else:
        print(f"    ✓ 区域 file_name 全部 ⊂ 图像级 file_name")

    # ---- 假设 2：区域标签 ⊆ 图像标签 ----
    name_to_cats = {fn: set(c) for fn, c, _b in img_ds.samples}
    n_violate = 0
    worst = None
    for fn, _crop, labels in reg_ds.regions:
        extra = set(labels) - name_to_cats.get(fn, set())
        if extra:
            n_violate += 1
            if worst is None:
                worst = (fn, sorted(extra))
    r["assumption_2_ok"] = n_violate == 0
    print(f"\n[2] 区域标签 ⊆ 同图图像级标签")
    print(f"    越界区域数：{n_violate} / {len(reg_ds.regions)}")
    if n_violate:
        print(f"    ✗ 首个反例：{worst[0]} 多出 {worst[1]}")
    else:
        print(f"    ✓ 全部区域标签都在同图图像级标签内（监督口径一致）")

    # ---- 假设 3：aggregate_by_image 在真实 file_name 上语义正确 ----
    # 构造分数：让每个区域的每个标签类都得 0.9，其余 0.05
    from training.train_tongue_region import aggregate_by_image
    n_cls = reg_ds.get_target(0).numel()
    scores = np.full((len(reg_ds.regions), n_cls), 0.05, dtype=np.float32)
    for i, (_fn, _crop, labels) in enumerate(reg_ds.regions):
        for c in labels:
            scores[i, c] = 0.9
    ys, yt = aggregate_by_image(scores, [fn for fn, _c, _l in reg_ds.regions], name_to_cats)

    exp_names = sorted({fn for fn, _c, _l in reg_ds.regions})
    r["assumption_3_ok"] = True
    print(f"\n[3] aggregate_by_image 语义")
    print(f"    输入 {scores.shape[0]} 区域 → 输出 {ys.shape[0]} 图（期望 {len(exp_names)}）")
    if ys.shape[0] != len(exp_names):
        r["assumption_3_ok"] = False
        print(f"    ✗ 图数不符")
    # 每图 y_true 应与图像级标签一致（只含"有区域的图"）
    mism = 0
    for row, fn in enumerate(exp_names):
        want = np.zeros(n_cls, dtype=np.int64)
        for c in name_to_cats.get(fn, ()):
            want[c] = 1
        if not np.array_equal(yt[row], want):
            mism += 1
    print(f"    y_true 与图像级标签不一致的图数：{mism}")
    if mism:
        r["assumption_3_ok"] = False
        print(f"    ✗ 聚合 GT 与图像级 GT 口径不一致")
    else:
        print(f"    ✓ 聚合 y_true == 图像级 GT（无静默错位）")
    # max 池化：正类分数应保留 0.9
    pos_max = float(ys[yt == 1].max()) if (yt == 1).any() else float("nan")
    neg_max = float(ys[yt == 0].max()) if (yt == 0).any() else float("nan")
    print(f"    正类最大分 {pos_max:.3f}（应 0.9）/ 负类最大分 {neg_max:.3f}（应为 0.05 或 0.9，"
          f"取决于该类在图内是否被某区域命中）")

    # ---- 假设 4：img_wh 覆盖 ----
    missing_wh = [fn for fn in reg_names if fn not in reg_ds.img_wh]
    r["assumption_4_ok"] = not missing_wh
    print(f"\n[4] img_wh 覆盖（region_type_stats 依赖）")
    print(f"    缺尺寸信息的区域所属图：{len(missing_wh)}")
    print(f"    {'✓ 全覆盖' if not missing_wh else '✗ 有缺口'}")

    r["n_images"] = len(img_ds.samples)
    r["n_regions"] = len(reg_ds.regions)
    r["n_images_with_region"] = len(reg_names)
    r["avg_regions_per_image"] = len(reg_ds.regions) / max(len(reg_names), 1)
    r["min_region_px"] = DEFAULT_MIN_REGION_PX
    r["stats"] = reg_ds.stats
    return r


def main():
    ap = argparse.ArgumentParser(description="区域级评估管线对齐检查")
    ap.add_argument("--data", type=str, default=None, help="数据集根目录（默认自动探测）")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    root = args.data or resolve_tongue_root()
    print(f"数据根目录：{root}")
    print(f"存在：{Path(root).is_dir()}")

    results = {s: check_split(root, s) for s in args.splits}

    print(f"\n{'=' * 74}\n汇总\n{'=' * 74}")
    all_ok = True
    for s, r in results.items():
        ok = (r["assumption_1_ok"] and r["assumption_2_ok"]
              and r["assumption_3_ok"] and r["assumption_4_ok"])
        all_ok &= ok
        print(f"  {s:6s} 图 {r['n_images']:5d} → 区域 {r['n_regions']:6d} "
              f"| 有区域图 {r['n_images_with_region']:5d} "
              f"| {r['avg_regions_per_image']:.2f} 区域/图 | {'✓ 4/4 通过' if ok else '✗ 有失败项'}")

    print(f"\n  总体：{'✓ 全部假设成立，评估管线对齐' if all_ok else '✗ 存在不成立假设，需修复'}")
    print(f"  （口径锚点：min_region_px=0 时全库应为 15,829 区域 = train 12690/val 1665/test 1474）")
    print("=" * 74)

    # ---------- 交接摘要（写到 code/_handoff/，不污染数据集目录） ----------
    H.set_out_dir(_CODE_ROOT / "_handoff")
    H.section("数据根")
    H.kv("root", root)
    H.kv("min_region_px", DEFAULT_MIN_REGION_PX)
    H.section("四条假设逐 split 结果")
    H.table(["split", "图", "区域", "假设1", "假设2", "假设3", "假设4", "判定"],
            [[s, r["n_images"], r["n_regions"],
              "OK" if r["assumption_1_ok"] else "FAIL",
              "OK" if r["assumption_2_ok"] else "FAIL",
              "OK" if r["assumption_3_ok"] else "FAIL",
              "OK" if r["assumption_4_ok"] else "FAIL",
              "通过" if all(r[f"assumption_{i}_ok"] for i in (1, 2, 3, 4)) else "失败"]
             for s, r in results.items()],
            widths=[7, 7, 8, 6, 6, 6, 6, 6])
    _tot = sum(r["n_regions"] for r in results.values())
    H.kv("区域总数", f"{_tot}（本机记录：15352，min_region_px=8）")
    H.kv("总体判定", "全部假设成立，评估管线对齐" if all_ok else "存在不成立假设")
    if not all_ok:
        H.fail("评估管线存在不成立的假设 → aggregate_by_image 可能静默错位，先修再跑训练")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(H.run(main))
