"""
TCM-Tongue 区域级弱监督加载器（Stage B · B-1b）
================================================

把「图像级 21 类多标签」升级为「**区域级多标签**」：

    一张图
      → 按 IoU ≥ 0.8 把重复框合并 → 若干**唯一区域**
      → 每个区域裁成 224×224（14×14 个 ViT patch，铺满）
      → 区域标签 = { c : IoA(区域, box_c) ≥ 0.8 }   ← 「区域覆盖」准则

为什么需要它（依据：2026-09-14 标注结构诊断，见 notes/bbox-structure-diag.md）
------------------------------------------------------------------------------
1. **bbox 双峰**：红点舌中位相对面积 0.0002（约 12×14 px）↔ 胖大舌 0.4453。
2. **union 裁剪已到上限**：所有框的并集占原图中位 **37.5%**，被整舌框撑满，
   局部病变框在 union 裁剪图里只从 0.21 patch 提到 **0.34 patch** —— 等于没裁。
   ⇒ 这解释了 `--crop_bbox` 只涨 0.0208（齿痕 +0.009 / 胖大 +0.008，纹丝不动）。
3. **不同类的整舌框共用同一矩形**（跨类框对 IoU 中位 = 1.00）
   ⇒ 数据集实质是「区域级多标签」，**逐框训练绝不能用单标签**
     （同图两个类会给同一张裁剪图互斥标签）。
4. 去重后共 **15,829 个唯一区域**（= 2.42× 样本量）；
   **局部病变区域占 10,057 个，其中 91.3% 是单标签** ← 监督干净。

与现有实现的关系（重要）
------------------------
· **向后兼容的严格推广**：当区域取「全图所有框的并集」时，标签集恰好等于图像级
  multi-hot ⇒ 与 `tongue_coco_loader` 的现有行为**完全等价**。
· **下游产物口径不变**：图像级 21 维预测 + 512 维特征 + `tongue_prototypes.pt`
  ⇒ **B-2 一行都不用改**。

口径对齐（必须与诊断脚本一致）
------------------------------
· `IOU_MERGE = 0.8`  → 合并重复框的阈值（同 `gen_bbox_structure_diag.py`）
· `COVER_TAU = 0.8`  → 区域覆盖的 IoA 阈值
· 标签用**未外扩的原始区域框**判定（与诊断一致）；实际裁剪才做 margin 外扩 + 正方形化。
  因为外扩只会让 IoA 更大（可能多出标签），用原始框判定是**更严**的口径。

用法
----
    python data/tongue_region_loader.py            # 自测（不需要 torch，只需数据集）
    # 期望：train=12690 val=1665 test=1474，合计 15829
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import DataLoader, Sampler
except ImportError:  # 本机无 torch 时也能 import（只做标注解析的自测）
    torch = None
    DataLoader = None
    Sampler = object  # type: ignore

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data.tongue_coco_loader import (  # noqa: E402
    DEFAULT_TONGUE_ROOT,
    TongueCOCODataset,
    resolve_tongue_root,
)
from models.tongue_label_mapping import (  # noqa: E402
    NUM_TONGUE_CLASSES,
    TONGUE_CLASSES,
)

# ---- 口径常量（与 gen_bbox_structure_diag.py 保持一致，勿单独修改）----
IOU_MERGE = 0.8          # IoU ≥ 该值 → 视为"同一个区域的重复框"
COVER_TAU = 0.8          # 区域覆盖阈值（IoA）
WHOLE_THRESH = 0.15      # 区域相对面积 ≥ 该值 → "整舌级区域"（仅用于统计/诊断）
DEFAULT_MIN_REGION_PX = 8  # 区域边长下限（与 _crop_to_bbox_union 的 8px 保护一致）

Box = Tuple[float, float, float, float]          # (x, y, w, h)
BoxWithCls = Tuple[float, float, float, float, float]  # (x, y, w, h, cls_id)


# ============================================================
# 几何工具（口径与诊断脚本一致）
# ============================================================

def _inter_area(b1: Box, b2: Box) -> float:
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[0] + b1[2], b2[0] + b2[2])
    y1 = min(b1[1] + b1[3], b2[1] + b2[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _iou(b1: Box, b2: Box) -> float:
    inter = _inter_area(b1, b2)
    if inter <= 0:
        return 0.0
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _ioa(container: Box, inner: Box) -> float:
    """IoA：inner 有多大比例落在 container 内。"""
    a = inner[2] * inner[3]
    return _inter_area(container, inner) / a if a > 0 else 0.0


def merge_duplicate_boxes(boxes: List[BoxWithCls],
                          iou_merge: float = IOU_MERGE) -> List[List[int]]:
    """把 IoU ≥ iou_merge 的框并成一组（并查集），返回每组的框下标列表。"""
    n = len(boxes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _iou(boxes[i][:4], boxes[j][:4]) >= iou_merge:  # type: ignore[arg-type]
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def union_box(boxes: List[BoxWithCls], idxs: List[int]) -> Box:
    """一组框的并集（该组通常就是同一个矩形，故并集 = 原框）。"""
    x0 = min(boxes[k][0] for k in idxs)
    y0 = min(boxes[k][1] for k in idxs)
    x1 = max(boxes[k][0] + boxes[k][2] for k in idxs)
    y1 = max(boxes[k][1] + boxes[k][3] for k in idxs)
    return (x0, y0, x1 - x0, y1 - y0)


def labels_by_coverage(crop: Box, boxes: List[BoxWithCls],
                       tau: float = COVER_TAU) -> Tuple[int, ...]:
    """「区域覆盖」准则：区域 crop 把哪些类别的框基本盖住了。

    注意：这里对**全图所有框**判定（不只是本组），因为一个整舌级区域会盖住
    图内所有局部病变框 —— 这正是"同区域多标签"的来源。
    """
    out = set()
    for b in boxes:
        if _ioa(crop, b[:4]) >= tau:  # type: ignore[arg-type]
            out.add(int(b[4]))
    return tuple(sorted(out))


# ============================================================
# 区域级数据集
# ============================================================

class TongueRegionDataset(TongueCOCODataset):
    """区域级多标签数据集：每个样本是一个「唯一区域」的裁剪图 + 标签集。

    继承 `TongueCOCODataset` 复用 COCO 解析与裁剪逻辑，只把样本粒度
    从「图」换成「区域」。
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_TONGUE_ROOT,
        split: str = "train",
        preprocess=None,
        crop_margin: float = 0.15,
        min_region_px: int = DEFAULT_MIN_REGION_PX,
        max_regions: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__(
            root=root, split=split, preprocess=preprocess,
            crop_bbox=False, crop_margin=crop_margin, verbose=False,
        )

        self.min_region_px = min_region_px

        # ---- 逐图提取唯一区域 ----
        regions: List[Tuple[str, Box, Tuple[int, ...]]] = []
        n_dropped_small = 0
        n_img_no_box = 0
        n_empty_label = 0
        n_box_used = 0

        for file_name, _cats, boxes in self.samples:
            if not boxes:
                n_img_no_box += 1
                continue
            n_box_used += len(boxes)
            for idxs in merge_duplicate_boxes(boxes):
                crop = union_box(boxes, idxs)
                if crop[2] < min_region_px or crop[3] < min_region_px:
                    n_dropped_small += 1
                    continue
                labels = labels_by_coverage(crop, boxes)
                if not labels:  # 理论上不可能：本组成员框的 IoA 恒为 1.0
                    n_empty_label += 1
                    continue
                regions.append((file_name, crop, labels))

        # ---- 截断标记（2026-09-15 修）----
        # 遍历统计（框数 / 丢弃数）覆盖整个 split，而 regions 可能被 max_regions 截断，
        # 直接相除会得到「14677 框 → 压缩 305.77 框/区域」这类分子全量、分母截断的
        # 误导值（冒烟实测踩到）。用 truncated 标记，并把 compression 换成
        # 「实际产出这批区域的图」的框数，保证分子分母同口径。
        n_all_regions = len(regions)
        if max_regions is not None:
            regions = regions[:max_regions]
        self.regions = regions
        truncated = n_all_regions > len(regions)

        # ---- 采样局部性：按图分组（供 ImageGroupedSampler 用）----
        by_img: Dict[str, List[int]] = {}
        for i, (fn, _c, _l) in enumerate(regions):
            by_img.setdefault(fn, []).append(i)
        self.by_image: List[List[int]] = list(by_img.values())
        self.region_index_by_image: Dict[str, List[int]] = by_img

        # ---- 图片缓存（1 项。配合按图分组的采样器 → 每图只解码一次）----
        self._cache_name: Optional[str] = None
        self._cache_img = None

        # ---- 统计 ----
        # 口径（2026-09-15 修）：
        #   n_images / n_boxes / n_regions_dropped_small / n_regions_full
        #       = 遍历整个 split 的累计值，描述数据集本身，与截断无关
        #   n_boxes_effective / compression
        #       = 只统计「实际产出这批 regions 的图」，与分母 len(regions) 同口径
        used_names = {fn for fn, _c, _l in regions}
        n_box_effective = sum(len(b) for fn, _c, b in self.samples if fn in used_names)
        self.stats = {
            "split": split,
            "n_images": len(self.samples),
            "n_images_no_label": sum(1 for _, c, _ in self.samples if not c),
            "n_boxes": int(sum(len(b) for _, _, b in self.samples)),
            "n_regions": len(regions),
            "n_regions_full": n_all_regions,
            "truncated": truncated,
            "n_images_with_region": len(self.by_image),
            "n_images_no_region": n_img_no_box,
            "n_regions_dropped_small": n_dropped_small,
            "n_regions_empty_label": n_empty_label,
            "n_boxes_effective": n_box_effective,
            "avg_labels_per_region": (
                sum(len(l) for _, _, l in regions) / max(len(regions), 1)
            ),
            "avg_regions_per_image": len(regions) / max(len(self.by_image), 1),
            "compression": n_box_effective / max(len(regions), 1),
        }

        if verbose:
            s = self.stats
            note = ""
            if s["truncated"]:
                note = (f" | ⚠️ max_regions 截断：全量应为 {s['n_regions_full']} 区域"
                        f"（「无框」与「丢弃过小」两项是全 split 统计，非本批）")
            print(f"  [{split}] {s['n_images']} 图 → {s['n_regions']} 区域 "
                  f"（{s['n_images_with_region']} 图有区域，无框 {n_img_no_box} 图）| "
                  f"{s['n_boxes_effective']} 框 → 压缩 {s['compression']:.2f} 框/区域 | "
                  f"平均 {s['avg_labels_per_region']:.2f} 标签/区域"
                  + (f" | 丢弃过小 {n_dropped_small}" if n_dropped_small else "")
                  + note)

    # ---------- 接口 ----------

    def __len__(self) -> int:
        return len(self.regions)

    def class_counts(self) -> List[int]:
        """每类（21 维）的**区域级**正样本数。"""
        counts = [0] * NUM_TONGUE_CLASSES
        for _fn, _crop, labels in self.regions:
            for c in labels:
                counts[c] += 1
        return counts

    def get_target(self, idx: int) -> "torch.Tensor":
        if torch is None:
            raise ImportError("需要 torch。")
        _fn, _crop, labels = self.regions[idx]
        v = torch.zeros(NUM_TONGUE_CLASSES, dtype=torch.float32)
        for c in labels:
            v[c] = 1.0
        return v

    def region_type_stats(self) -> Dict[str, Dict[str, float]]:
        """按「整舌级 / 局部病变」分组统计标签集大小（与诊断报告口径一致）。

        判据：区域相对面积（原始区域框 / 原图面积）≥ WHOLE_THRESH 视为整舌级。
        """
        pools: Dict[str, List[int]] = {"whole": [], "local": []}
        for file_name, crop, labels in self.regions:
            w, h = self.img_wh.get(file_name, (0, 0))
            if w <= 0 or h <= 0:
                continue
            rel = (crop[2] * crop[3]) / float(w * h)
            pools["whole" if rel >= WHOLE_THRESH else "local"].append(len(labels))

        out: Dict[str, Dict[str, float]] = {}
        for tag, pool in pools.items():
            if not pool:
                out[tag] = {"n": 0, "avg_labels": 0.0, "single_ratio": 0.0}
                continue
            out[tag] = {
                "n": len(pool),
                "avg_labels": sum(pool) / len(pool),
                "single_ratio": sum(1 for x in pool if x == 1) / len(pool),
            }
        return out

    # ---------- 图像读取（带 1 项缓存）----------

    def _load_region_image(self, file_name: str):
        if self._cache_name == file_name and self._cache_img is not None:
            return self._cache_img
        img = self._load_image(file_name)
        self._cache_name, self._cache_img = file_name, img
        return img

    def __getitem__(self, idx: int):
        file_name, region, _labels = self.regions[idx]

        img = self._load_region_image(file_name)
        # 复用父类的裁剪逻辑：单个区域的框 → margin 外扩 → 正方形化 → 贴边裁切
        # （与 union 裁剪同样的几何处理，保证两档消融可比）
        img = self._crop_to_bbox_union(img, [region])

        if self.preprocess is not None:
            img = self.preprocess(img)

        return img, self.get_target(idx), file_name


# ============================================================
# 采样器：按图像分组打乱（省掉重复解码）
# ============================================================

class ImageGroupedSampler(Sampler):  # type: ignore[misc]
    """图像分组采样器。

    区域模式下同一张图产生多个样本（平均 2.42 个）。若完全随机打乱，
    同一张图会被反复解码 —— **JPEG 解码量 ×2.42，成为 CPU 瓶颈**
    （单图最大 5136×2608）。

    本采样器把打乱粒度放在**图像级**：
      · 图像顺序随机（每个 epoch 不同）
      · 图像内部区域顺序也随机
      · 同一张图的区域**连续出现** ⇒ 配合 Dataset 的 1 项缓存 =
        **每图每 epoch 只解码一次**

    代价：一个 batch 内的样本可能来自约 13 张图（而非 32 张），样本相关性略增。
    需要严格独立采样时，用 `make_region_loader(..., shuffle_images=False)`。
    """

    def __init__(self, by_image: List[List[int]], seed: int = 0,
                 shuffle_images: bool = True, shuffle_within: bool = True):
        super().__init__()
        self.by_image = [list(v) for v in by_image]
        self.seed = seed
        self.shuffle_images = shuffle_images
        self.shuffle_within = shuffle_within
        self._epoch = 0
        self._len = sum(len(v) for v in self.by_image)

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self._epoch)
        self._epoch += 1
        order = list(range(len(self.by_image)))
        if self.shuffle_images:
            rng.shuffle(order)
        out: List[int] = []
        for i in order:
            idxs = list(self.by_image[i])
            if self.shuffle_within:
                rng.shuffle(idxs)
            out.extend(idxs)
        return iter(out)

    def __len__(self) -> int:
        return self._len


# ============================================================
# 构建
# ============================================================

def build_region_datasets(
    root: str | Path,
    preprocess=None,
    crop_margin: float = 0.15,
    min_region_px: int = DEFAULT_MIN_REGION_PX,
    max_regions: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, TongueRegionDataset]:
    """一次性构建 train/val/test 三个区域级数据集。

    min_region_px=0 时不做任何过滤，**区域数与 bbox-structure-diag 报告完全一致**
    （15,829 = train 12690 / val 1665 / test 1474），可作口径回归检查。
    """
    out: Dict[str, TongueRegionDataset] = {}
    for split in ("train", "val", "test"):
        out[split] = TongueRegionDataset(
            root=root, split=split, preprocess=preprocess,
            crop_margin=crop_margin, min_region_px=min_region_px,
            max_regions=max_regions, verbose=verbose,
        )
    return out


def make_region_loader(
    ds: TongueRegionDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_images: bool = True,
    seed: int = 0,
    drop_last: bool = False,
):
    """给区域级数据集配 DataLoader（默认用图像分组采样器）。

    train 用分组采样器；val/test 只需顺序遍历，但也用分组采样器以便缓存命中。
    """
    if DataLoader is None:
        raise ImportError("需要 torch 才能构建 DataLoader。")
    sampler = ImageGroupedSampler(
        ds.by_image, seed=seed, shuffle_images=shuffle_images,
    )
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last,
    )


# ============================================================
# 自测（不需要 torch，不解码图片即可验证标注解析）
# ============================================================

if __name__ == "__main__":
    print("=" * 78)
    print("TCM-Tongue 区域级加载器自测（只解析标注，不解码图片）")
    print("=" * 78)

    root = resolve_tongue_root()
    print(f"数据根目录：{root}  (exists={Path(root).is_dir()})")

    if not Path(root).is_dir():
        print("⚠️  数据集目录不存在 → 请用 --tongue_root 指定后重跑。")
        raise SystemExit(0)

    # ============ A. 口径核对：不做任何过滤 → 必须与诊断报告完全一致 ============
    print("\n[A] 口径核对（min_region_px=0，不做任何过滤）")
    ds_all = build_region_datasets(root, preprocess=None, min_region_px=0, verbose=False)
    n_all = {s: ds_all[s].stats["n_regions"] for s in ("train", "val", "test")}
    print(f"  区域数：train={n_all['train']}  val={n_all['val']}  "
          f"test={n_all['test']}  合计={sum(n_all.values())}")
    assert n_all["train"] == 12690, f"train 区域数 {n_all['train']} ≠ 12690（诊断报告值）"
    assert n_all["val"] == 1665, f"val 区域数 {n_all['val']} ≠ 1665"
    assert n_all["test"] == 1474, f"test 区域数 {n_all['test']} ≠ 1474"
    print("  ✓ 与 bbox-structure-diag 报告的 15,829（12690/1665/1474）完全一致")

    print("  标签集大小（与诊断的「整舌级 2.18 / 局部病变 1.14」对比）：")
    agg: Dict[str, Dict[str, float]] = {}
    for ds in ds_all.values():
        for tag, d in ds.region_type_stats().items():
            a = agg.setdefault(tag, {"n": 0.0, "w": 0.0, "single": 0.0})
            a["n"] += d["n"]
            a["w"] += d["avg_labels"] * d["n"]
            a["single"] += d["single_ratio"] * d["n"]
    for tag, zh, expect in (("whole", "整舌级", 2.18), ("local", "局部病变", 1.14)):
        a = agg[tag]
        avg = a["w"] / max(a["n"], 1)
        single = a["single"] / max(a["n"], 1)
        print(f"    {zh:<5s} n={int(a['n']):6d} | 平均标签数 {avg:.2f}（诊断 {expect}）"
              f" | 单标签占比 {single:.1%}")
        assert abs(avg - expect) < 0.06, f"{zh} 平均标签数 {avg:.3f} 偏离诊断值 {expect}"
    print("  ✓ 去重与标签口径与诊断一致")

    # ============ B. 默认配置：丢弃边长 <8px 的退化区域 ============
    print(f"\n[B] 默认配置（min_region_px={DEFAULT_MIN_REGION_PX}）")
    datasets = build_region_datasets(root, preprocess=None)
    total_regions = sum(ds.stats["n_regions"] for ds in datasets.values())
    n_drop = sum(ds.stats["n_regions_dropped_small"] for ds in datasets.values())
    print(f"  区域总数：{total_regions}"
          f"（丢弃 {n_drop} 个边长 <{DEFAULT_MIN_REGION_PX}px 的退化区域，"
          f"占 {n_drop/max(sum(n_all.values()),1):.1%}）")
    assert total_regions + n_drop == sum(n_all.values()), "过滤前后区域数不自洽"
    print("  丢弃理由：这些框边长不足 8px，裁剪会触发 _crop_to_bbox_union 的 8px 保护")
    print("            → 退化成「整图 + 单标签」，是有害的错误监督样本")

    # ---- 每个区域都必须有标签 ----
    n_empty = sum(ds.stats["n_regions_empty_label"] for ds in datasets.values())
    assert n_empty == 0, f"有 {n_empty} 个区域标签为空"
    print("  ✓ 所有区域标签非空")

    # ---- 区域级类别分布（21 类）----
    counts = [0] * NUM_TONGUE_CLASSES
    for ds in datasets.values():
        for i, v in enumerate(ds.class_counts()):
            counts[i] += v
    print("\n区域级类别分布（21 类，正样本区域数）：")
    for i, c in enumerate(TONGUE_CLASSES):
        mark = " ❌ 无区域" if counts[i] == 0 else ""
        print(f"  {i:2d} {c['zh']:8s} {counts[i]:6d}{mark}")

    # ---- 采样器：必须是 0..N-1 的一个排列（不重不漏）----
    tr = datasets["train"]
    if Sampler is not object:
        smp = ImageGroupedSampler(tr.by_image, seed=0)
        order = list(iter(smp))
        assert len(order) == len(tr), f"采样长度 {len(order)} ≠ {len(tr)}"
        assert sorted(order) == list(range(len(tr))), "采样序列不是 0..N-1 的排列"
        second = list(iter(smp))
        assert sorted(second) == list(range(len(tr))), "第二个 epoch 的采样序列不是排列"
        assert second != order, "两个 epoch 的采样顺序完全相同（打乱失效）"
        print(f"  ✓ 采样器：长度/覆盖正确，且跨 epoch 顺序不同")
        # 局部性检查：同一图的区域应连续出现
        img_of = [tr.regions[i][0] for i in range(len(tr))]
        seen_contiguous = sum(
            1 for i in range(1, len(order))
            if img_of[order[i]] == img_of[order[i - 1]]
        )
        print(f"  ✓ 采样局部性：相邻样本同图比例 "
              f"{seen_contiguous / max(len(order) - 1, 1):.1%}"
              f"（≈1 - 1/平均区域数；缓存命中率即此值）")

    # ---- 裁剪路径：真解一张图 + 对一个区域裁剪（不需要 torch）----
    fn, region, labels = tr.regions[0]
    img = tr._load_region_image(fn)
    print(f"\n样例区域 0：{fn} | 区域框 {region[2]:.0f}×{region[3]:.0f} px | "
          f"原图 {img.size[0]}×{img.size[1]}")
    crop = tr._crop_to_bbox_union(img, [region])
    print(f"  裁剪后：{crop.size[0]}×{crop.size[1]} px | 标签 "
          f"{[TONGUE_CLASSES[c]['zh'] for c in labels]}")
    assert crop.size[0] > 0 and crop.size[1] > 0
    # 缓存应命中（同一张图第二次取）
    assert tr._load_region_image(fn) is img, "图像缓存未命中"
    print("  ✓ 裁剪与图像缓存正常")

    print("\n" + "=" * 78)
    print("区域级加载器自测通过 ✓")
    print("=" * 78)
