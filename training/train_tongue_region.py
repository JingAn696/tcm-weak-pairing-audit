"""
区域级弱监督视觉分支训练脚本（Stage B · B-1b / 选项③）
=======================================================

**为什么需要它（论文第 4 条贡献「标注粒度决定可学习性」的对照实验）**

B-1 两轮已证明（2026-09-14）：
  · 整图训练  macro F1 = 0.3097
  · union 裁剪 macro F1 = 0.3305（+0.0208，且增益 85% 来自支持数 <30 的类）
bbox 结构诊断给出根因：局部病变框（红点舌中位 12×14px）在整图里只占 0.21 个
ViT patch，union 裁剪也只提到 0.34 patch —— **粒度不匹配，不是模型不行**。

本脚本把监督粒度从「图像级」换成「区域级」（tongue_region_loader）：
  · 每个唯一区域裁成 224×224（局部病变铺满 14×14 patch）
  · 区域标签 = 「区域覆盖」准则（IoA ≥ 0.8），91.3% 局部区域是单标签（监督干净）
  · 样本量 15,829 区域 ≈ 2.42× 图像数

**消融的干净性（论文写法）**：模型结构 / 推理管线 / 原型提取与 B-1 **完全一致**，
唯一变量 = 训练监督粒度。⇒ 任何指标差都可归因于粒度本身。

推理与原型提取走**图像级**（与 B-1 同管线，B-2 零改动）：
  · 指标报三套：区域级（训练原生粒度）、图像级-聚合（区域分数按类 max 池化）、
    图像级-整图（整图前向，与 B-1 的 0.3097/0.3305 直接可比 ← 头条数字）
  · 原型用整图前向特征 + 图像级 GT + 新矩阵（2026-09-15）+ B6 判据

用法（AutoDL，4090）
--------------------
    cd /root/autodl-tmp/papers/code
    python training/train_tongue_region.py \
        --tongue_root "/root/autodl-tmp/TCM-Tongue/shezhen_datasets1/shezhen datasets/shezhenv3-coco/shezhenv3-coco"

输出（runs/tongue_region/）：
    best_model.pt            区域级 val macro-F1 最优权重
    tongue_prototypes.pt     10 病性视觉原型（整图特征 + B6，B-2 直接可用）
    test_report.json         三套粒度指标 + 原型诊断完整报告

预计耗时（4090）：12 epoch × 15.8k 区域 ≈ 1–1.5 小时（B-1 的 ~2.4 倍样本量）。
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.tongue_coco_loader import DEFAULT_TONGUE_ROOT, build_datasets
from data.tongue_region_loader import build_region_datasets, make_region_loader
from models.tongue_label_mapping import (
    MATRIX_VERSION,
    NUM_TONGUE_CLASSES,
    SYNDROMES_ZH,
    TONGUE_CLASSES,
)
from models.tongue_vision_branch import (
    FEAT_DIM,
    TongueVisionBranch,
    configure_finetune,
    extract_syndrome_prototypes,
    load_biomedclip_visual,
)
from training.train_tongue_branch import collect_scores, compute_tongue_metrics
from utils.run_handoff import Handoff  # noqa: E402

# 运行交接摘要（2026-09-15）：收尾时把关键信息同时打印 + 落盘 handoff_*.txt
H = Handoff("train_tongue_region")

# B-1 两轮头条数字（2026-09-14，论文对照锚点）
B1_BASELINE = {"whole": 0.3097, "crop": 0.3305}


# ============================================================
# 工具
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_scores_named(model, loader, device, use_amp: bool):
    """与 collect_scores 相同，但额外返回逐样本 file_name（区域 → 图 聚合用）。"""
    model.eval()
    scores, labels, names = [], [], []
    for imgs, targets, fnames in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _feat, z21, _, _ = model(imgs)
        scores.append(torch.sigmoid(z21.float()).cpu().numpy())
        labels.append(targets.numpy())
        names.extend(list(fnames))
    return np.concatenate(scores), np.concatenate(labels).astype(int), names


def aggregate_by_image(scores: np.ndarray, names: list[str], name_to_cats: dict):
    """区域分数 → 图像级（逐类 max 池化），GT = 图像级标签。

    返回 (y_score_img, y_true_img)，按 name 字典序对齐（确定性）。
    """
    by_img = defaultdict(list)
    for i, fn in enumerate(names):
        by_img[fn].append(i)
    img_names = sorted(by_img)
    y_score = np.zeros((len(img_names), scores.shape[1]), dtype=np.float32)
    y_true = np.zeros((len(img_names), scores.shape[1]), dtype=np.int64)
    for r, fn in enumerate(img_names):
        y_score[r] = scores[by_img[fn]].max(axis=0)
        for c in name_to_cats.get(fn, ()):
            y_true[r, c] = 1
    return y_score, y_true


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage B · 区域级弱监督视觉分支训练（B-1b）")
    parser.add_argument("--tongue_root", type=str, default=DEFAULT_TONGUE_ROOT)
    parser.add_argument("--output_dir", type=str,
                        default=str(_PROJECT_ROOT / "runs" / "tongue_region"))
    parser.add_argument("--model_name", type=str,
                        default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--feat_dim", type=int, default=FEAT_DIM)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--unfreeze_blocks", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--crop_margin", type=float, default=0.15)
    parser.add_argument("--min_region_px", type=int, default=8,
                        help="区域边长下限（与 tongue_region_loader 默认一致）")

    parser.add_argument("--min_pos", type=int, default=5,
                        help="训练集区域级正样本少于此数的类，屏蔽其损失贡献")
    parser.add_argument("--max_pos_weight", type=float, default=50.0)
    parser.add_argument("--eval_min_pos", type=int, default=5,
                        help="val 集正样本少于此数的类，不计入主指标")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--max_regions", type=int, default=None, help="调试：每 split 只取前 N 个区域")
    parser.add_argument("--proto_strategy", choices=["mean", "centered", "centered_idf"],
                        default="centered")
    parser.add_argument("--no_prune_white", dest="prune_normal_white", action="store_false",
                        help="关闭 B6 判据（默认开启，净安 2026-09-15 定稿）")
    parser.set_defaults(prune_normal_white=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not args.no_amp) and torch.cuda.is_available()

    # ---- 交接摘要：环境与配置 ----
    H.section("环境")
    H.kv("device", device)
    if torch.cuda.is_available():
        H.kv("gpu", torch.cuda.get_device_name(0))
    H.kv("amp(bf16)", use_amp)
    H.kv("torch", torch.__version__)
    H.kv("MATRIX_VERSION", MATRIX_VERSION)
    H.section("配置")
    for _k in ("tongue_root", "epochs", "batch_size", "lr_head", "lr_backbone",
               "unfreeze_blocks", "crop_margin", "min_region_px", "min_pos",
               "eval_min_pos", "seed", "proto_strategy"):
        H.kv(_k, getattr(args, _k, "-"))
    H.kv("prune_normal_white(B6)", args.prune_normal_white)
    H.kv("max_regions", args.max_regions if args.max_regions else "不限（正式）")
    if args.max_regions:
        H.warn(f"--max_regions={args.max_regions} 是调试截断，指标无意义，勿用于论文")

    print("=" * 74)
    print("Stage B · 区域级弱监督视觉分支（B-1b：唯一变量 = 监督粒度）")
    print(f"  device={device} | amp(bf16)={use_amp} | seed={args.seed} | "
          f"unfreeze_blocks={args.unfreeze_blocks}")
    print(f"  对照锚点（B-1, 2026-09-14）：整图 {B1_BASELINE['whole']} / 裁剪 {B1_BASELINE['crop']}")
    print("=" * 74)

    # ---------- 1. 视觉塔 ----------
    print(f"\n[1] 加载 BiomedCLIP 视觉塔（{args.model_name.split('/')[-1]}）...")
    visual, preprocess, raw_dim, img_size = load_biomedclip_visual(
        model_name=args.model_name, device=device
    )
    finetune_stats = configure_finetune(visual, n_unfreeze_blocks=args.unfreeze_blocks)

    # ---------- 2. 区域级数据 ----------
    print(f"\n[2] 构建区域级数据集（{args.tongue_root}）...")
    datasets = build_region_datasets(
        root=args.tongue_root, preprocess=preprocess,
        crop_margin=args.crop_margin, min_region_px=args.min_region_px,
        max_regions=args.max_regions,
    )
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    loaders = {
        "train": make_region_loader(train_ds, batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle_images=True, seed=args.seed, drop_last=True),
        "val": make_region_loader(val_ds, batch_size=args.batch_size * 2,
                                  num_workers=args.num_workers,
                                  shuffle_images=False, seed=args.seed),
        "test": make_region_loader(test_ds, batch_size=args.batch_size * 2,
                                   num_workers=args.num_workers,
                                   shuffle_images=False, seed=args.seed),
    }
    print(f"  {len(loaders['train'])} batch/epoch（区域级）")

    # 图像级标签字典（聚合评估 + 图像级 GT 用）
    name_to_cats = {
        split: {fn: cats for fn, cats, _ in datasets[split].samples}
        for split in ("train", "val", "test")
    }

    # ---------- 3. 类别掩码 / 正样本权重（区域级计数）----------
    train_counts = train_ds.class_counts()
    val_counts = val_ds.class_counts()
    class_mask = torch.tensor(
        [1.0 if c >= args.min_pos else 0.0 for c in train_counts], dtype=torch.float32
    )
    eval_classes = [i for i in range(NUM_TONGUE_CLASSES) if val_counts[i] >= args.eval_min_pos]

    print(f"\n[3] 类别可用性（区域级正样本数，括号内为图像级）")
    img_val_counts = {c["id"]: 0 for c in TONGUE_CLASSES}
    for cats in name_to_cats["val"].values():
        for c in cats:
            img_val_counts[c] += 1
    for i, c in enumerate(TONGUE_CLASSES):
        marks = []
        if class_mask[i] == 0:
            marks.append("损失屏蔽")
        if i not in eval_classes:
            marks.append("评估排除")
        print(f"  {i:2d} {c['zh']:8s} train={train_counts[i]:5d} "
              f"val={val_counts[i]:4d}（图 {img_val_counts[i]:4d}）  {' / '.join(marks)}")
    print(f"  → 参与训练 {int(class_mask.sum())}/{NUM_TONGUE_CLASSES} 类，"
          f"参与评估 {len(eval_classes)}/{NUM_TONGUE_CLASSES} 类")

    pw = []
    n_total = len(train_ds)
    for i in range(NUM_TONGUE_CLASSES):
        n_pos = train_counts[i]
        if n_pos < args.min_pos:
            pw.append(1.0)
        else:
            pw.append(min(float(n_total - n_pos) / max(n_pos, 1), args.max_pos_weight))
    pos_weight = torch.tensor(pw, dtype=torch.float32).to(device)

    # ---------- 4. 模型（与 B-1 完全相同）----------
    model = TongueVisionBranch(
        visual_backbone=visual, raw_dim=raw_dim, feat_dim=args.feat_dim
    ).to(device)
    ps = model.param_stats()
    print(f"\n[4] 模型：总可训 {ps['trainable']/1e6:.2f}M ({ps['trainable_ratio']*100:.1f}%)"
          f"（结构与 B-1 一致，唯一变量 = 监督粒度）")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.") and p.requires_grad]
    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ---------- 5. 训练（区域级 BCE；模型选择 = 区域级 val macro F1）----------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    best_val = -1.0
    H.set_out_dir(output_dir)

    print(f"\n[5] 训练开始（epochs={args.epochs}, batch={args.batch_size}，区域级）...")
    t0 = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n_seen = 0.0, 0
        for step, (imgs, targets, _) in enumerate(loaders["train"], 1):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, z21, _, _ = model(imgs)
                loss, parts = model.compute_loss(
                    z21, targets, class_mask=class_mask.to(device), pos_weight=pos_weight
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = targets.size(0)
            run_loss += parts["bce"] * bs
            n_seen += bs
            if step % 100 == 0 or step == len(loaders["train"]):
                print(f"  epoch {epoch} | batch {step}/{len(loaders['train'])} | "
                      f"bce={run_loss/n_seen:.4f}")

        ys, yt, _ = collect_scores(model, loaders["val"], device, use_amp)
        vm = compute_tongue_metrics(yt, ys, eval_classes)
        print(f"  epoch {epoch} | VAL(区域级) macro_f1(valid)={vm['macro_f1_valid']:.4f} "
              f"subset_acc={vm['subset_accuracy']:.4f} auc={vm['auc_valid']:.4f}")
        history.append({"epoch": epoch, "train_bce": run_loss / max(n_seen, 1),
                        "val_region_macro_f1_valid": vm["macro_f1_valid"]})

        if vm["macro_f1_valid"] > best_val:
            best_val = vm["macro_f1_valid"]
            torch.save({"model_state": model.state_dict(),
                        "args": vars(args),
                        "granularity": "region",
                        "finetune_stats": finetune_stats},
                       best_path)
            print(f"  ✓ 保存 best（val 区域级 macro_f1_valid={best_val:.4f}）→ {best_path}")

    print(f"\n  训练完成，用时 {(time.time()-t0)/60:.1f} 分钟，best={best_val:.4f}")

    # ---------- 6. test 三套粒度评估 ----------
    print(f"\n[6] 加载 best，test 三套粒度评估...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # (a) 区域级（训练原生粒度）
    rs, rl, rnames = collect_scores_named(model, loaders["test"], device, use_amp)
    tm_region = compute_tongue_metrics(rl, rs, eval_classes)
    # (b) 图像级-聚合（区域分数逐类 max 池化）
    agg_s, agg_t = aggregate_by_image(rs, rnames, name_to_cats["test"])
    img_eval_classes = [i for i in range(NUM_TONGUE_CLASSES)
                        if img_val_counts[i] >= args.eval_min_pos]
    tm_agg = compute_tongue_metrics(agg_t, agg_s, img_eval_classes)
    # (c) 图像级-整图（与 B-1 同管线，直接可比）
    img_datasets = build_datasets(root=args.tongue_root, preprocess=preprocess,
                                  crop_bbox=False, crop_margin=args.crop_margin)
    img_loaders = {
        s: DataLoader(img_datasets[s], batch_size=args.batch_size * 2, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)
        for s in ("val", "test")
    }
    ws, wt, _ = collect_scores(model, img_loaders["test"], device, use_amp)
    tm_whole = compute_tongue_metrics(wt, ws, img_eval_classes)

    print(f"\n📊 区域级模型 test 三套粒度指标（N区域={len(test_ds)} / N图={agg_t.shape[0]}）")
    print(f"  {'粒度':<16s} {'macro_f1(valid)':>14s} {'micro_f1':>9s} {'auc':>8s}")
    print(f"  {'区域级(原生)':<16s} {tm_region['macro_f1_valid']:>14.4f} "
          f"{tm_region['micro_f1']:>9.4f} {tm_region['auc_valid']:>8.4f}")
    print(f"  {'图像级-聚合':<16s} {tm_agg['macro_f1_valid']:>14.4f} "
          f"{tm_agg['micro_f1']:>9.4f} {tm_agg['auc_valid']:>8.4f}")
    print(f"  {'图像级-整图':<16s} {tm_whole['macro_f1_valid']:>14.4f} "
          f"{tm_whole['micro_f1']:>9.4f} {tm_whole['auc_valid']:>8.4f}")
    print(f"\n  对照（B-1 图像级-整图）：整图训练 {B1_BASELINE['whole']} / "
          f"裁剪训练 {B1_BASELINE['crop']}  →  Δ = "
          f"{tm_whole['macro_f1_valid'] - B1_BASELINE['whole']:+.4f} / "
          f"{tm_whole['macro_f1_valid'] - B1_BASELINE['crop']:+.4f}")
    print(f"\n  图像级-整图 per-class F1（对照 B-1 逐类看粒度效应）：")
    for i, c in enumerate(TONGUE_CLASSES):
        zh = c["zh"]
        f1 = tm_whole["per_class_f1"][zh]
        sup = tm_whole["per_class_support"][zh]
        tag = "" if i in img_eval_classes else "  (未计入主指标)"
        print(f"    {zh:8s} F1={f1:.4f} n={sup:4d} {'█' * int(f1 * 30)}{tag}")

    # ---------- 7. 提取 10 病性视觉原型（整图特征 + B6，与 B-1 同管线）----------
    print(f"\n[7] 提取 10 病性视觉原型（整图前向 | strategy={args.proto_strategy}"
          f" | B6={args.prune_normal_white} | 矩阵 {MATRIX_VERSION}）...")
    img_all = build_datasets(root=args.tongue_root, preprocess=preprocess,
                             crop_bbox=False, crop_margin=args.crop_margin)
    all_feats, all_gt = [], []
    for split in ("train", "val", "test"):
        ld = DataLoader(img_all[split], batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        ys, yt, ft = collect_scores(model, ld, device, use_amp)
        all_feats.append(ft)
        all_gt.append(torch.tensor(yt, dtype=torch.float32))
    feats = torch.cat(all_feats, 0)
    det_labels = torch.cat(all_gt, 0)

    protos, proto_meta = extract_syndrome_prototypes(
        model, feats, det_labels, use_gt=True, l2_normalize=True,
        strategy=args.proto_strategy, prune_normal_white=args.prune_normal_white,
    )
    if args.prune_normal_white:
        print(f"  [B6] 剔除孤立白苔舌 {proto_meta.get('n_pruned_white', 0)} 张图（正常薄白）")
    proto_path = output_dir / "tongue_prototypes.pt"
    proto_blob = {
        "prototypes": protos,
        "syndromes": list(SYNDROMES_ZH),
        "strategy": args.proto_strategy,
        "meta": proto_meta,
        "source": "gt",
        "feat_dim": args.feat_dim,
        "matrix_version": MATRIX_VERSION,
        "prune_normal_white": bool(args.prune_normal_white),
        "granularity": "region_trained__image_infer",
        "config": {"unfreeze_blocks": args.unfreeze_blocks, "epochs": args.epochs,
                   "batch_size": args.batch_size, "lr_head": args.lr_head,
                   "lr_backbone": args.lr_backbone, "min_region_px": args.min_region_px,
                   "seed": args.seed},
    }
    torch.save(proto_blob, proto_path)
    torch.save(proto_blob, output_dir / f"tongue_prototypes_{args.proto_strategy}.pt")
    print(f"  ✓ 原型已保存：{proto_path}")
    print(f"  {'病性':8s} {'支撑图数':>8s} {'专属信号强度':>12s}")
    _pre = proto_meta.get("pre_norm") or [1.0] * 10
    for j, s in enumerate(SYNDROMES_ZH):
        print(f"  {s:8s} {proto_meta['n_images'][j]:8d}  {_pre[j]:12.4f}")

    # ---------- 8. 报告 ----------
    report = {
        "stage": "B-1b_region_weak_supervision",
        "purpose": "标注粒度消融：唯一变量 = 训练监督粒度（区域级 vs 图像级）",
        "config": vars(args),
        "matrix_version": MATRIX_VERSION,
        "finetune_stats": finetune_stats,
        "data": {k: v.stats for k, v in datasets.items()},
        "region_train_counts": {TONGUE_CLASSES[i]["zh"]: train_counts[i]
                                for i in range(NUM_TONGUE_CLASSES)},
        "eval_classes_region": [TONGUE_CLASSES[i]["zh"] for i in eval_classes],
        "eval_classes_image": [TONGUE_CLASSES[i]["zh"] for i in img_eval_classes],
        "history": history,
        "best_val_region_macro_f1_valid": best_val,
        "test": {
            "region_level": tm_region,
            "image_level_aggregated": tm_agg,
            "image_level_whole": tm_whole,
        },
        "b1_baseline_whole_image": B1_BASELINE,
        "delta_vs_b1": {
            "vs_whole": tm_whole["macro_f1_valid"] - B1_BASELINE["whole"],
            "vs_crop": tm_whole["macro_f1_valid"] - B1_BASELINE["crop"],
        },
        "prototypes": {
            "path": str(proto_path),
            "strategy": args.proto_strategy,
            "prune_normal_white": bool(args.prune_normal_white),
            "n_pruned_white": proto_meta.get("n_pruned_white", 0),
            "n_images": dict(zip(SYNDROMES_ZH, proto_meta["n_images"])),
            "pre_norm": dict(zip(SYNDROMES_ZH, proto_meta.get("pre_norm") or [])),
        },
    }
    report_path = output_dir / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存：{report_path}")
    print("=" * 74)

    # ---------- 交接摘要（专供复制给 Buddy） ----------
    H.section("三套粒度 test 指标（头条 = 图像级-整图，对照 B-1 0.3097/0.3305）")
    H.table(["粒度", "macro_f1(valid)", "micro_f1", "auc"],
            [["区域级(原生)", f"{tm_region['macro_f1_valid']:.4f}",
              f"{tm_region['micro_f1']:.4f}", f"{tm_region['auc_valid']:.4f}"],
             ["图像级-聚合", f"{tm_agg['macro_f1_valid']:.4f}",
              f"{tm_agg['micro_f1']:.4f}", f"{tm_agg['auc_valid']:.4f}"],
             ["图像级-整图", f"{tm_whole['macro_f1_valid']:.4f}",
              f"{tm_whole['micro_f1']:.4f}", f"{tm_whole['auc_valid']:.4f}"]],
            widths=[16, 16, 10, 9])
    _dw = tm_whole["macro_f1_valid"] - B1_BASELINE["whole"]
    _dc = tm_whole["macro_f1_valid"] - B1_BASELINE["crop"]
    H.kv("Δ vs B-1 整图(0.3097)", f"{_dw:+.4f}")
    H.kv("Δ vs B-1 裁剪(0.3305)", f"{_dc:+.4f}")
    H.kv("best_val(区域级 macro_f1_valid)", f"{best_val:.4f}")
    H.kv("区域数", f"train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}")

    H.section("原型（整图特征 + 新矩阵 + B6）")
    H.kv("n_pruned_white(B6)", proto_meta.get("n_pruned_white", 0))
    H.kv_dict(dict(zip(SYNDROMES_ZH, proto_meta["n_images"])))

    H.section("产物")
    H.artifact(proto_path)
    H.artifact(output_dir / f"tongue_prototypes_{args.proto_strategy}.pt")
    H.artifact(best_path)
    H.artifact(report_path)

    H.line("")
    H.line("下一步：用上面的原型重跑 B-2 —— ")
    H.line(f"  python training/train_full_model.py --visual_mode proto_attn --freeze_text \\")
    H.line(f"      --epochs 12 --proto_path {proto_path} --seed 42 \\")
    H.line(f"      --output_dir runs/full_model_v2_b2_region_e12_s42")


if __name__ == "__main__":
    raise SystemExit(H.run(main))
