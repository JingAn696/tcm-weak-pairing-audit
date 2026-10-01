"""
Stage B 端到端冒烟测试
=======================

在没有 GPU / 没有 open_clip 的机器上，验证 Stage B 完整 pipeline 是否跑得通：

    真实舌象图 → COCO 加载器 → preprocess → TongueVisionBranch(假 backbone)
    → 21 类 BCE → 反传 → 评估指标 → 原型提取

目的：把"只有在 AutoDL 上第一次运行时才会暴露"的问题（形状不匹配、collate、
优化器参数组、数据路径、图像解码）在本机提前捕获。

用法（本机 CPU）：
    python smoke_test_stage_b.py
    # 指定数据集路径
    python smoke_test_stage_b.py --tongue_root "path/to/shezhenv3-coco"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from data.tongue_coco_loader import DEFAULT_TONGUE_ROOT, TongueCOCODataset
from models.tongue_label_mapping import NUM_SYNDROMES, NUM_TONGUE_CLASSES, SYNDROMES_ZH
from models.tongue_vision_branch import (
    TongueVisionBranch,
    configure_finetune,
    extract_syndrome_prototypes,
)

_OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
_OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]


class SimplePreprocess:
    """模拟 open_clip 的 preprocess（本机无 torchvision，手写等价实现）。"""

    def __init__(self, size: int = 224):
        self.size = size
        self._mean = torch.tensor(_OPENAI_MEAN).view(3, 1, 1)
        self._std = torch.tensor(_OPENAI_STD).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        w, h = img.size
        s = self.size / min(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        w, h = img.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))
        arr = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0).permute(2, 0, 1)
        return (arr - self._mean) / self._std


class DummyViT(nn.Module):
    """模拟 BiomedCLIP 视觉塔：参数名含 blocks.N. 以覆盖解冻逻辑。"""

    def __init__(self, raw_dim: int = 768, n_blocks: int = 4):
        super().__init__()
        self.patch_embed = nn.Sequential(nn.Conv2d(3, 64, 16, 16), nn.Flatten(2))
        self.proj_in = nn.Linear(64, raw_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(raw_dim, raw_dim), nn.GELU())
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(raw_dim)

    def forward(self, x):
        h = self.patch_embed(x).transpose(1, 2)   # (B, 196, 64)
        h = self.proj_in(h).mean(dim=1)           # (B, raw_dim) pooled
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tongue_root", type=str, default=DEFAULT_TONGUE_ROOT)
    ap.add_argument("--n_train", type=int, default=24)
    ap.add_argument("--n_eval", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    print("=" * 74)
    print("Stage B 端到端冒烟测试（真实舌象图 + 假 backbone，CPU）")
    print("=" * 74)

    root = Path(args.tongue_root)
    if not root.exists():
        print(f"\n⚠️  数据集不存在：{root}")
        print("   请用 --tongue_root 指定后再跑。")
        return 1

    preprocess = SimplePreprocess(224)

    # ---------- 1. 数据 ----------
    print(f"\n[1] 加载真实数据（各取少量样本）")
    train_ds = TongueCOCODataset(root, "train", preprocess, max_samples=args.n_train)
    val_ds = TongueCOCODataset(root, "val", preprocess, max_samples=args.n_eval)
    print(f"  训练用 {len(train_ds)} 张，评估用 {len(val_ds)} 张")

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    print("\n[2] 取一个 batch 验证形状")
    imgs, targets, names = next(iter(loader))
    print(f"  images {tuple(imgs.shape)} | targets {tuple(targets.shape)}")
    print(f"  图像取值范围 [{imgs.min():.2f}, {imgs.max():.2f}]（应约在 ±2.5 内）")
    assert imgs.shape[1:] == (3, 224, 224), f"图像形状异常：{imgs.shape}"
    assert targets.shape[1] == NUM_TONGUE_CLASSES
    assert targets.sum() > 0, "该 batch 全为阴性，说明标签解析有问题"
    print(f"  标签示例：{names[0]} → {[i for i in range(NUM_TONGUE_CLASSES) if targets[0, i] > 0]}")

    # ---------- 3. 模型 ----------
    print("\n[3] 构建模型（假 backbone，raw_dim=768）")
    backbone = DummyViT(raw_dim=768, n_blocks=4)
    configure_finetune(backbone, n_unfreeze_blocks=2)
    model = TongueVisionBranch(backbone, raw_dim=768, feat_dim=512)
    print(f"  {model.param_stats()}")

    print("\n[4] 前向 + 反传 + 优化（固定小批，验证 loss 能下降）")
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.") and p.requires_grad]
    print(f"  参数组：head {len(head_params)} 个 tensor / backbone {len(backbone_params)} 个 tensor")
    assert len(backbone_params) > 0, "解冻失败：backbone 无可训参数"
    assert len(head_params) > 0, "分类头无可训参数"

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": 1e-3}, {"params": backbone_params, "lr": 1e-4}],
        weight_decay=0.01,
    )
    mask = torch.ones(NUM_TONGUE_CLASSES)
    for i, c in enumerate(range(NUM_TONGUE_CLASSES)):
        if train_ds.class_counts()[i] == 0:
            mask[i] = 0.0

    model.train()
    losses = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, z21, _, _ = model(imgs)
        loss, parts = model.compute_loss(z21, targets, class_mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(parts["bce"])
        if step % 5 == 0 or step == 1:
            print(f"    step {step:2d} | bce={parts['bce']:.4f}")
    assert losses[-1] < losses[0], f"loss 未下降：{losses[0]:.4f} → {losses[-1]:.4f}"
    print(f"  ✓ loss 从 {losses[0]:.4f} 降到 {losses[-1]:.4f}（过拟合小批成功）")

    print("\n[5] 评估通路（无梯度）")
    model.eval()
    with torch.no_grad():
        scores, trues = [], []
        for im, tg, _ in val_loader:
            _, z21v, _, _ = model(im)
            scores.append(torch.sigmoid(z21v).numpy())
            trues.append(tg.numpy())
    y_score = np.concatenate(scores)
    y_true = np.concatenate(trues).astype(int)
    y_pred = (y_score >= 0.5).astype(int)
    from sklearn.metrics import f1_score
    print(f"  y_score {y_score.shape} | macro_f1(21 类) = "
          f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    assert y_score.shape == y_true.shape

    print("\n[6] 原型提取通路")
    with torch.no_grad():
        feats, tgts = [], []
        for im, tg, _ in loader:
            feats.append(model.encode(im))
            tgts.append(tg)
    F = torch.cat(feats, 0)
    T = torch.cat(tgts, 0)
    assert F.size(0) == T.size(0), f"特征数 {F.size(0)} 与标签数 {T.size(0)} 不一致"
    protos, meta = extract_syndrome_prototypes(model, F, T)
    print(f"  原型 {tuple(protos.shape)}（应为 (10, 512)），{F.size(0)} 张图参与统计")
    assert protos.shape == (NUM_SYNDROMES, 512)
    print(f"  各病性支撑图数：{dict(zip(SYNDROMES_ZH, meta['n_images']))}")

    print("\n[7] crop_bbox 模式（读取真实大图 + bbox 裁剪）")
    crop_ds = TongueCOCODataset(root, "val", preprocess, crop_bbox=True,
                                crop_margin=0.15, max_samples=4)
    crop_loader = DataLoader(crop_ds, batch_size=4, shuffle=False, num_workers=0)
    c_imgs, c_tg, _ = next(iter(crop_loader))
    print(f"  裁剪后 images {tuple(c_imgs.shape)} | 取值范围 [{c_imgs.min():.2f}, {c_imgs.max():.2f}]")
    assert c_imgs.shape[1:] == (3, 224, 224)
    assert not torch.allclose(c_imgs, imgs[:4]), "裁剪后的图与整图完全相同 → 裁剪未生效"

    print("\n[8] bbox 覆盖率诊断（整图 vs 裁剪的信息量依据）")
    cov = train_ds.bbox_coverage()
    assert len(cov) > 0, "未能计算出任何 bbox 覆盖率"
    import statistics
    med = statistics.median(cov)
    side = (med ** 0.5) * 224
    print(f"  训练集 {len(cov)} 张有标注图 | bbox 并集占比 中位数 {med*100:.1f}% "
          f"（均值 {statistics.mean(cov)*100:.1f}%）")
    print(f"  整图 resize 到 224 时舌体约占 {side:.0f}×{side:.0f} 像素 "
          f"= ViT patch16 下 {side/16:.1f}×{side/16:.1f} 个 patch")
    print(f"  裁剪模式信息量约为整图的 {1/max(med, 0.01):.1f} 倍")

    print("\n" + "=" * 74)
    print("冒烟测试全部通过 ✓  —— Stage B pipeline 可上 AutoDL")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
