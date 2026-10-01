"""
TCM-Tongue COCO 数据加载器（Stage B 视觉分支）
==============================================

把 TCM-Tongue 的目标检测标注（bbox）聚合成**图像级 21 类 multi-hot**，
供视觉分支做多标签分类训练。

数据来源（本机路径）
------------------------
    <root>/
      train/{images, annotations/train.json, labels, classes.txt}
      val/{...}
      test/{...}

    root 默认：自动探测（见 TONGUE_ROOT_CANDIDATES，覆盖本机下划线版 /
               工作区副本 / AutoDL 数据盘三种落点）。
               显式指定：--tongue_root "<含 train/val/test 的目录>"

⚠️ 注意目录命名：本机解压版的中间目录是 shezhenv3_coco（**下划线**），
   数据集最内层才是 shezhenv3-coco（连字符）。两者容易混。

⚠️ 三个必须知道的坑（2026-09-14 实测）
--------------------------------------
1. **必须按 pinyin 名索引，不能按 categories 的 id 索引**
   —— 三个 split 的 classes.txt 顺序不一致（train 把 xinfeiao/xinfeitu 写反）。
   COCO json 的 id↔name 映射三个 split 一致（已核实与映射表 0-20 完全对齐），
   但为防将来换数据源，本模块统一走 PINYIN_TO_ID 转换。
2. **原图是 2048×1365 大图**，bbox 平均只占画面一部分。
   整图 resize 到 224 后舌体可能只剩几十像素（ViT patch 16 → 仅几个 patch 覆盖），
   信息损失大。因此提供 crop_bbox 模式（裁剪所有 bbox 的并集 + margin）。
3. **4 个凸起类近乎空**（piweitu 0 / shenqutu 2 / xinfeitu 2 / gandantu 9 实例），
   训练时须对零正样本类屏蔽损失（见 train_tongue_branch.py 的 --min_pos）。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object  # 本机无 torch 时也能 import

from PIL import Image

import sys

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from models.tongue_label_mapping import (  # noqa: E402
    NUM_TONGUE_CLASSES,
    TONGUE_CLASSES,
    TONGUE_PINYIN_TO_ID,
)

# ============================================================
# 数据集根目录：多候选自动探测（2026-09-14 改为自动）
# ============================================================
# 背景：这套数据在本机/服务器上出现过三种不同的落点与命名，硬编码单一路径
# 已两次造成故障（脚本跑不起来 / 指到不存在的目录）：
#   ① 本机解压版的中间目录名是 **shezhenv3_coco（下划线）**，不是 shezhenv3-coco；
#   ② AutoDL 上放的是连字符版；
#   ③ 工作区内还留了一份全量副本（纯 ASCII 路径，本机跑诊断最省事）。
# 因此改为：显式 --tongue_root 最优先，否则按候选列表取第一个真实存在的目录。
TONGUE_ROOT_CANDIDATES = (
    # ① 本机（2026-09-14 恢复；注意中间目录是下划线 shezhenv3_coco）
    r"path/to/TCM-Tongue/shezhen_datasets1"
    r"/shezhen datasets/shezhenv3_coco/shezhenv3-coco",
    # ①' 兼容连字符命名（AutoDL / 早期解压版本）
    r"path/to/TCM-Tongue/shezhen_datasets1"
    r"/shezhen datasets/shezhenv3-coco/shezhenv3-coco",
    # ② 工作区内副本（相对本文件；纯 ASCII，本机调试首选）
    str(Path(__file__).resolve().parent / "TCM-Tongue" / "shezhen_datasets1"
        / "shezhen datasets" / "shezhenv3_coco" / "shezhenv3-coco"),
    # ③ AutoDL 服务器常见位置
    "/root/autodl-tmp/TCM-Tongue/shezhen_datasets1/shezhen datasets/shezhenv3-coco/shezhenv3-coco",
    "/root/autodl-tmp/TCM-Tongue/shezhen_datasets1/shezhen datasets/shezhenv3_coco/shezhenv3-coco",
)


def resolve_tongue_root(explicit: str = "") -> str:
    """显式路径优先；否则返回候选列表中第一个真实存在的；都不存在则返回首个候选。

    这样即便数据集换位置/换命名，也不用改代码——失败时报错信息会带出全部候选，
    便于一眼定位（见 tong_coco_loader 的 FileNotFoundError 处理）。
    """
    if explicit:
        return explicit
    for cand in TONGUE_ROOT_CANDIDATES:
        try:
            if Path(cand).is_dir():
                return cand
        except OSError:
            continue
    return TONGUE_ROOT_CANDIDATES[0]


DEFAULT_TONGUE_ROOT = resolve_tongue_root()


class TongueCOCODataset(Dataset):
    """
    TCM-Tongue 图像级多标签数据集（21 类）。

    每个样本返回 (preprocessed_image, target_21, path)。
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_TONGUE_ROOT,
        split: str = "train",
        preprocess=None,
        crop_bbox: bool = False,
        crop_margin: float = 0.15,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        参数
        ----
        root        : shezhenv3-coco 目录（含 train/val/test）
        split       : "train" / "val" / "test"
        preprocess  : torchvision transform（open_clip 的 preprocess）
        crop_bbox   : True → 裁剪"所有 bbox 的并集 + margin"，而非整图 resize
        crop_margin : 裁剪时向外扩的比例（0.15 = 四周各扩 15%）
        max_samples : 只取前 N 个样本（调试用）
        """
        self.root = Path(root)
        self.split = split
        self.preprocess = preprocess
        self.crop_bbox = crop_bbox
        self.crop_margin = crop_margin

        split_dir = self.root / split
        ann_path = split_dir / "annotations" / f"{split}.json"
        if not ann_path.exists():
            cands = "\n".join(f"      - {c}" for c in TONGUE_ROOT_CANDIDATES)
            raise FileNotFoundError(
                f"找不到 COCO 标注：{ann_path}\n"
                f"  当前 root：{self.root}\n"
                f"  已探测的候选路径（第一个存在的会被自动采用）：\n{cands}\n"
                f"  请用 --tongue_root 显式指定含 train/val/test 的目录。\n"
                f"  ⚠️ 注意中间目录可能是 shezhenv3_coco（下划线）而非 shezhenv3-coco。"
            )
        self.img_dir = split_dir / "images"

        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        # ---- category_id (COCO) → 我们的 21 类索引（按 pinyin，防 classes.txt 顺序坑）----
        self.cat_map: Dict[int, int] = {}
        unknown = []
        for c in coco["categories"]:
            key = str(c["name"]).strip().lower()
            if key in TONGUE_PINYIN_TO_ID:
                self.cat_map[c["id"]] = TONGUE_PINYIN_TO_ID[key]
            else:
                unknown.append(c["name"])
        if unknown:
            raise ValueError(
                f"COCO 里出现未知舌象类别名（未在 tongue_label_mapping 中定义）：{unknown}"
            )
        if len(self.cat_map) != NUM_TONGUE_CLASSES:
            raise ValueError(
                f"COCO 类别数 {len(self.cat_map)} ≠ 映射表 {NUM_TONGUE_CLASSES} 类，"
                f"请检查数据源是否一致"
            )

        # ---- 图片 → 类别集合 / bbox 列表 ----
        # box 统一记为 5 元组 (x, y, w, h, cls_id)：
        # 前 4 项与 COCO 一致，第 5 项是**我们自己的 21 类索引**。
        # 之所以带上类别，是因为区域级加载器（tongue_region_loader）要按
        # "区域覆盖"准则给每个区域定标签；旧代码只取前 4 项，故此为**向后兼容**扩展。
        cats_of_img: Dict[int, set] = defaultdict(set)
        boxes_of_img: Dict[int, List[List[float]]] = defaultdict(list)
        for a in coco["annotations"]:
            img_id = a["image_id"]
            our_id = self.cat_map.get(a["category_id"])
            if our_id is None:
                continue
            cats_of_img[img_id].add(our_id)
            bbox = a.get("bbox")
            if bbox:
                boxes_of_img[img_id].append(list(bbox) + [float(our_id)])

        # ---- 样本表 ----
        # samples[i] = (file_name, (类别索引...), ((x, y, w, h, cls_id), ...))
        self.samples: List[Tuple[str, Tuple[int, ...], Tuple[List[float], ...]]] = []
        self.img_wh: Dict[str, Tuple[int, int]] = {}
        n_no_ann = 0
        for im in coco["images"]:
            img_id = im["id"]
            cats = tuple(sorted(cats_of_img.get(img_id, ())))
            if not cats:
                n_no_ann += 1
            boxes = tuple(boxes_of_img.get(img_id, ()))
            self.samples.append((im["file_name"], cats, boxes))
            self.img_wh[im["file_name"]] = (im.get("width", 0), im.get("height", 0))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # ---- 统计 ----
        self.n_missing_file = 0
        n_img_with_box = sum(1 for _, _, b in self.samples if b)
        self.stats = {
            "split": split,
            "n_images": len(self.samples),
            "n_images_no_label": n_no_ann,
            "n_images_with_bbox": n_img_with_box,
            "n_boxes": int(sum(len(b) for _, _, b in self.samples)),
            "avg_labels_per_image": (
                sum(len(c) for _, c, _ in self.samples) / max(len(self.samples), 1)
            ),
        }

        if verbose:
            print(f"  [{split}] {self.stats['n_images']} 图 | "
                  f"{self.stats['n_boxes']} bbox | "
                  f"平均 {self.stats['avg_labels_per_image']:.2f} 类/图 | "
                  f"无标注 {n_no_ann} 图")

    # ---------- 基础 ----------

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self) -> List[int]:
        """每类（21 维）的**图像级**正样本数。"""
        counts = [0] * NUM_TONGUE_CLASSES
        for _, cats, _ in self.samples:
            for c in cats:
                counts[c] += 1
        return counts

    def bbox_coverage(self) -> List[float]:
        """
        每张图"bbox 并集面积 / 原图面积"。

        **这是决定「整图输入 vs bbox 裁剪」的关键诊断**：
        若中位数只有 30-40%，说明整图 resize 到 224 后舌体只剩 ~1/3 边长
        （ViT patch16 下仅 8 个 patch 覆盖），细粒度特征（齿痕、裂纹）会糊掉，
        此时裁剪模式的信息量约为整图的 2-3 倍。
        """
        out = []
        for file_name, _, boxes in self.samples:
            if not boxes:
                continue
            w, h = self.img_wh.get(file_name, (0, 0))
            if w <= 0 or h <= 0:
                continue
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[0] + b[2] for b in boxes)
            y2 = max(b[1] + b[3] for b in boxes)
            out.append(max(0.0, (x2 - x1) * (y2 - y1)) / (w * h))
        return out

    def get_target(self, idx: int) -> "torch.Tensor":
        """返回第 idx 个样本的 21 维 multi-hot 标签。"""
        if torch is None:
            raise ImportError("需要 torch。")
        _, cats, _ = self.samples[idx]
        v = torch.zeros(NUM_TONGUE_CLASSES, dtype=torch.float32)
        for c in cats:
            v[c] = 1.0
        return v

    # ---------- 图像读取 ----------

    def _load_image(self, file_name: str) -> Image.Image:
        path = self.img_dir / file_name
        if not path.exists():
            self.n_missing_file += 1
            return Image.new("RGB", (224, 224), (128, 128, 128))
        img = Image.open(path)
        return img.convert("RGB")

    def _crop_to_bbox_union(self, img: Image.Image, boxes) -> Image.Image:
        """裁到所有 bbox 的并集 + margin（保持长宽比，后续由 preprocess 统一 resize）。"""
        if not boxes:
            return img
        W, H = img.size
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        # 外扩 margin（相对 bbox 尺寸）
        bw, bh = x2 - x1, y2 - y1
        x1 -= bw * self.crop_margin
        x2 += bw * self.crop_margin
        y1 -= bh * self.crop_margin
        y2 += bh * self.crop_margin
        # 扩成正方形（避免 resize 时舌体形变；ViT 输入本就是方形）
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1)
        x1, x2 = cx - side / 2, cx + side / 2
        y1, y2 = cy - side / 2, cy + side / 2
        box = (
            int(max(0, x1)), int(max(0, y1)),
            int(min(W, x2)), int(min(H, y2)),
        )
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return img
        return img.crop(box)

    # ---------- Dataset 接口 ----------

    def __getitem__(self, idx: int):
        file_name, cats, boxes = self.samples[idx]

        if self.crop_bbox:
            img = self._load_image(file_name)
            img = self._crop_to_bbox_union(img, boxes)
        else:
            img = self._load_image(file_name)

        if self.preprocess is not None:
            img = self.preprocess(img)

        target = self.get_target(idx)
        return img, target, file_name


def build_datasets(
    root: str | Path,
    preprocess,
    crop_bbox: bool = False,
    crop_margin: float = 0.15,
    max_samples: Optional[int] = None,
) -> Dict[str, TongueCOCODataset]:
    """一次性构建 train/val/test 三个 split。"""
    out = {}
    for split in ("train", "val", "test"):
        out[split] = TongueCOCODataset(
            root=root, split=split, preprocess=preprocess,
            crop_bbox=crop_bbox, crop_margin=crop_margin,
            max_samples=max_samples, verbose=True,
        )
    return out


# ============================================================
# 自测（不需要 torch / 不需要图片解码，只验证标注解析）
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("TCM-Tongue COCO 加载器自测（只解析标注，不解码图片）")
    print("=" * 72)

    root = DEFAULT_TONGUE_ROOT
    if not Path(root).exists():
        print(f"⚠️  数据集目录不存在：{root}")
        print("   请用 --tongue_root 指定正确路径后重跑。")
    else:
        datasets = build_datasets(root, preprocess=None, crop_bbox=False)

        # 汇总三类 split 的图像级类别计数
        total = [0] * NUM_TONGUE_CLASSES
        for ds in datasets.values():
            cnt = ds.class_counts()
            for i, v in enumerate(cnt):
                total[i] += v

        print("\n全数据集图像级类别分布（21 类）：")
        grand = sum(ds.stats["n_images"] for ds in datasets.values())
        for i, c in enumerate(TONGUE_CLASSES):
            n = total[i]
            flag = " ❌ 实例不足" if c["n_inst"] < 30 else ""
            print(f"  {i:2d} {c['zh']:8s} {n:5d} 图 ({n/grand*100:5.1f}%){flag}")

        # 断言：解析出的类数应与 tongue_label_mapping 的记录一致
        by_zh = {c["zh"]: total[i] for i, c in enumerate(TONGUE_CLASSES)}
        print("\n关键一致性检查：")
        for zh in ("脾胃区凸起", "肾区凸起", "心肺区凸起", "肝胆区凸起"):
            print(f"  {zh}: {by_zh[zh]} 图（tongue_label_mapping 记录 {[c['n_inst'] for c in TONGUE_CLASSES if c['zh']==zh][0]} 实例）")

        assert by_zh["脾胃区凸起"] == 0, "脾胃区凸起应为 0 图（数据集缺口）"
        assert len(datasets["train"].samples) == 5594
        assert len(datasets["val"].samples) == 572
        assert len(datasets["test"].samples) == 553

    print("\n" + "=" * 72)
    print("自测通过 ✓")
    print("=" * 72)
