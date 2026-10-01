"""
舌象视觉分支训练脚本（Stage B · B-1）
=====================================

在 TCM-Tongue 上训练 BiomedCLIP 视觉塔 + 21 类舌象分类头，
训练完成后**自动提取 10 病性视觉原型**（B-2 接回完整模型 v2 的输入）。

用法（AutoDL，4090）：
    python training/train_tongue_branch.py \\
        --tongue_root "path/to/shezhenv3-coco"

    # 消融：bbox 裁剪 vs 整图
    python training/train_tongue_branch.py --tongue_root ... --crop_bbox

    # 消融：全冻结 vs 解冻后 N 层
    python training/train_tongue_branch.py --tongue_root ... --unfreeze_blocks 0

输出：
    runs/tongue_branch/best_model.pt            最佳 val macro-F1 权重
    runs/tongue_branch/tongue_prototypes.pt     10 病性视觉原型（B-2 用）
    runs/tongue_branch/test_report.json        21 类 + 原型诊断完整报告

预计耗时（4090）：12 epoch × 5594 图 ≈ 30-60 分钟。

关键设计（对应领域专家 2026-09-14 的三个拍板）
------------------------------------------
1. 图像输入：整图 resize（默认）；`--crop_bbox` 为消融（bbox 并集 + 15% margin）
2. 视觉塔：分类头 + 视觉塔**后 4 层**小 lr 微调（`--unfreeze_blocks`）
   —— TCM 舌象域差大，Stage A 已证"冻结学不动"
3. 输出接法：21 类舌象头 → **固定映射矩阵 M（不训练）** → 10 病性视觉证据
   M 是领域先验注入，不是可学参数（论文 Method 要写清）
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import random
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.tongue_coco_loader import DEFAULT_TONGUE_ROOT, build_datasets
from models.tongue_label_mapping import (
    MATRIX_VERSION,
    NUM_SYNDROMES,
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
from utils.run_handoff import Handoff  # noqa: E402

# 交接摘要（与 train_tongue_region.py / train_full_model.py 同一套写法）
H = Handoff("train_tongue_branch")


# ============================================================
# 随机性
# ============================================================

def set_seed(seed: int) -> None:
    """与 train_tongue_region.py 逐行一致的四件套（仓库统一口径）。

    2026-09-16 补：本脚本此前**完全没有 seed 控制**（既无 --seed 也无 manual_seed），
    导致视觉分支不可复现、无法多 seed 重复 —— 这是"从头做实验"必须先补的方法论缺口。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 评估
# ============================================================

@torch.no_grad()
def collect_scores(model, loader, device, use_amp: bool):
    """返回 (y_score, y_true, feats) —— 均为 numpy / tensor。"""
    model.eval()
    scores, labels, feats = [], [], []
    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            feat, z21, _, _ = model(imgs)
        scores.append(torch.sigmoid(z21.float()).cpu().numpy())
        labels.append(targets.numpy())
        feats.append(feat.float().cpu())
    return (np.concatenate(scores), np.concatenate(labels).astype(int),
            torch.cat(feats, dim=0))


def compute_tongue_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    eval_classes: list[int],
    threshold: float = 0.5,
) -> dict:
    """21 类多标签指标；macro 指标分"全类"与"有效类"两套。"""
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    y_pred = (y_score >= threshold).astype(int)

    m = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_all21": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }
    # 有效类（正样本足够的类）上的 macro —— 论文主指标
    if eval_classes:
        m["macro_f1_valid"] = float(
            f1_score(y_true[:, eval_classes], y_pred[:, eval_classes],
                     average="macro", zero_division=0)
        )
        try:
            m["auc_valid"] = float(
                roc_auc_score(y_true[:, eval_classes], y_score[:, eval_classes],
                              average="macro")
            )
        except ValueError:
            m["auc_valid"] = 0.0
    else:
        m["macro_f1_valid"] = 0.0
        m["auc_valid"] = 0.0

    per = f1_score(y_true, y_pred, average=None, zero_division=0)
    m["per_class_f1"] = {TONGUE_CLASSES[i]["zh"]: float(per[i]) for i in range(NUM_TONGUE_CLASSES)}
    m["per_class_support"] = {TONGUE_CLASSES[i]["zh"]: int(y_true[:, i].sum()) for i in range(NUM_TONGUE_CLASSES)}
    m["eval_classes"] = [TONGUE_CLASSES[i]["zh"] for i in eval_classes]

    # 病性级投影诊断（GT 与预测各自的 10 病性阳性率）
    from models.tongue_label_mapping import get_matrix_torch
    Mt = get_matrix_torch().numpy()  # (21, 10)
    m["syndrome_rate_from_gt"] = {
        SYNDROMES_ZH[j]: float(((y_true @ Mt)[:, j] > 0).mean()) for j in range(NUM_SYNDROMES)
    }
    m["syndrome_rate_from_pred"] = {
        SYNDROMES_ZH[j]: float(((y_pred @ Mt)[:, j] > 0).mean()) for j in range(NUM_SYNDROMES)
    }
    return m


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stage B 舌象视觉分支训练")
    parser.add_argument("--tongue_root", type=str, default=DEFAULT_TONGUE_ROOT)
    parser.add_argument("--output_dir", type=str,
                        default=str(_PROJECT_ROOT / "runs" / "tongue_branch"))
    parser.add_argument("--model_name", type=str,
                        default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--feat_dim", type=int, default=FEAT_DIM)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_head", type=float, default=1e-3, help="分类头 / 投影层学习率")
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                        help="视觉塔解冻层学习率（小 lr 防漂移）")
    parser.add_argument("--unfreeze_blocks", type=int, default=4,
                        help="解冻视觉塔最后 N 个 transformer block（0 = 全冻结）")
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--crop_bbox", action="store_true",
                        help="裁到 bbox 并集 + margin（默认用整图）")
    parser.add_argument("--crop_margin", type=float, default=0.15)
    parser.add_argument("--drop_unlabeled", action="store_true",
                        help="丢弃无任何标注的图（默认保留作全阴样本）")

    parser.add_argument("--min_pos", type=int, default=5,
                        help="训练集正样本少于此数的类，屏蔽其损失贡献")
    parser.add_argument("--use_pos_weight", action="store_true", default=True)
    parser.add_argument("--no_pos_weight", dest="use_pos_weight", action="store_false")
    parser.add_argument("--max_pos_weight", type=float, default=50.0)
    parser.add_argument("--eval_min_pos", type=int, default=5,
                        help="val 集正样本少于此数的类，不计入主指标")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="调试：只取前 N 张")
    parser.add_argument("--proto_source", choices=["gt", "pred"], default="gt",
                        help="原型权重来源：gt（默认，干净）或 pred（模型预测）")
    parser.add_argument("--proto_strategy", choices=["mean", "centered", "centered_idf"],
                        default="centered",
                        help="原型构造策略。默认 centered —— 2026-09-14 实测：mean 会让原型互相重合"
                             "（气虚×痰湿 0.994，B-2 注意力失效）；centered 降至 0.71 且不过度放大差类")
    parser.add_argument("--no_prune_white", dest="prune_normal_white", action="store_false",
                        help="关闭 B6 判据（原型提取时剔除孤立白苔舌）。默认开启"
                             "（领域专家 2026-09-15 定稿）；敏感性分析用旧口径时才加此开关")
    parser.set_defaults(prune_normal_white=True)

    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（2026-09-16 新增）。本脚本此前无 seed 控制 → 不可复现、"
                             "无法多 seed 重复；现与 train_tongue_region.py 对齐")
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not args.no_amp) and torch.cuda.is_available()

    print("=" * 74)
    print("Stage B · 舌象视觉分支（BiomedCLIP + 21 类头 + 映射投影）")
    print(f"  device={device} | amp(bf16)={use_amp} | crop_bbox={args.crop_bbox} | "
          f"unfreeze_blocks={args.unfreeze_blocks} | seed={args.seed}")
    print(f"  MATRIX_VERSION={MATRIX_VERSION} | prune_normal_white(B6)={args.prune_normal_white}")
    print("=" * 74)

    # ---- 交接摘要：环境 ----
    H.section("环境")
    H.kv("torch", torch.__version__)
    H.kv("device", device)
    H.kv("MATRIX_VERSION", MATRIX_VERSION)
    H.section("配置")
    for _k in ("tongue_root", "epochs", "batch_size", "lr_head", "lr_backbone",
               "unfreeze_blocks", "crop_bbox", "crop_margin", "min_pos",
               "proto_source", "proto_strategy", "seed"):
        H.kv(_k, getattr(args, _k, "-"))
    H.kv("prune_normal_white(B6)", args.prune_normal_white)
    if args.max_samples:
        H.warn(f"--max_samples={args.max_samples} 是调试截断，指标无意义，勿用于论文")

    # ---------- 1. 视觉塔 ----------
    print(f"\n[1] 加载 BiomedCLIP 视觉塔（{args.model_name.split('/')[-1]}）...")
    visual, preprocess, raw_dim, img_size = load_biomedclip_visual(
        model_name=args.model_name, device=device
    )
    print(f"  ✓ 输入 {img_size}×{img_size}，视觉塔原始输出 {raw_dim} 维")
    finetune_stats = configure_finetune(visual, n_unfreeze_blocks=args.unfreeze_blocks)

    # ---------- 2. 数据 ----------
    print(f"\n[2] 加载 TCM-Tongue（{args.tongue_root}）...")
    datasets = build_datasets(
        root=args.tongue_root, preprocess=preprocess,
        crop_bbox=args.crop_bbox, crop_margin=args.crop_margin,
        max_samples=args.max_samples,
    )

    if args.drop_unlabeled:
        for split, ds in datasets.items():
            before = len(ds.samples)
            ds.samples = [s for s in ds.samples if s[1]]
            print(f"  [{split}] 丢弃无标注图 {before - len(ds.samples)} 张 → {len(ds.samples)}")

    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    # ---- 分辨率诊断：整图 vs 裁剪的信息量差（论文 Appendix + 决策依据）----
    cov = train_ds.bbox_coverage()
    if cov:
        import statistics
        med = statistics.median(cov)
        side = (med ** 0.5) * img_size
        n_patch = side / 16
        print(f"  [诊断] bbox 并集占原图面积中位数 {med*100:.1f}% → "
              f"整图模式下舌体约占 {side:.0f}×{side:.0f} 像素 "
              f"（ViT patch16 → {n_patch:.1f}×{n_patch:.1f} patch）")
        print(f"  [诊断] 裁剪模式的信息量约为整图的 {1/max(med, 0.01):.1f} 倍"
              f"（当前 = {'裁剪' if args.crop_bbox else '整图'}）")

    # ---------- 3. 类别掩码 / 正样本权重 ----------
    train_counts = train_ds.class_counts()
    val_counts = val_ds.class_counts()
    class_mask = torch.tensor(
        [1.0 if c >= args.min_pos else 0.0 for c in train_counts], dtype=torch.float32
    )
    eval_classes = [i for i in range(NUM_TONGUE_CLASSES) if val_counts[i] >= args.eval_min_pos]

    print(f"\n[3] 类别可用性（训练集图像级正样本数）")
    for i, c in enumerate(TONGUE_CLASSES):
        n_tr, n_va = train_counts[i], val_counts[i]
        marks = []
        if class_mask[i] == 0:
            marks.append("损失屏蔽")
        if i not in eval_classes:
            marks.append("评估排除")
        print(f"  {i:2d} {c['zh']:8s} train={n_tr:5d} val={n_va:4d}  {' / '.join(marks)}")
    print(f"  → 参与训练 {int(class_mask.sum())}/{NUM_TONGUE_CLASSES} 类，"
          f"参与评估 {len(eval_classes)}/{NUM_TONGUE_CLASSES} 类")

    pos_weight = None
    if args.use_pos_weight:
        pw = []
        for i in range(NUM_TONGUE_CLASSES):
            n_pos = train_counts[i]
            n_neg = len(train_ds) - n_pos
            if n_pos < args.min_pos:
                pw.append(1.0)
            else:
                pw.append(min(float(n_neg) / max(n_pos, 1), args.max_pos_weight))
        pos_weight = torch.tensor(pw, dtype=torch.float32).to(device)
        print(f"  → pos_weight（上限 {args.max_pos_weight}）："
              f"最大 {pos_weight.max().item():.1f}（{TONGUE_CLASSES[int(pos_weight.argmax())]['zh']}）")

    # ---------- 4. 模型 ----------
    model = TongueVisionBranch(
        visual_backbone=visual, raw_dim=raw_dim, feat_dim=args.feat_dim
    ).to(device)
    ps = model.param_stats()
    print(f"\n[4] 模型：backbone {ps['backbone_total']/1e6:.1f}M"
          f"（可训 {ps['backbone_trainable']/1e6:.2f}M）+ head {ps['head']/1e6:.2f}M | "
          f"总可训 {ps['trainable']/1e6:.2f}M ({ps['trainable_ratio']*100:.1f}%)")

    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True),
        "val": DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True),
    }
    print(f"  {len(loaders['train'])} batch/epoch")

    # ---------- 5. 优化器 ----------
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.") and p.requires_grad]
    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    print(f"\n[5] 优化器：head {sum(p.numel() for p in head_params)/1e6:.2f}M @ {args.lr_head}"
          f" + backbone {sum(p.numel() for p in backbone_params)/1e6:.2f}M @ {args.lr_backbone}")

    # ---------- 6. 训练 ----------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    best_val = -1.0

    print(f"\n[6] 训练开始（epochs={args.epochs}, batch={args.batch_size}）...")
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
            if step % 50 == 0 or step == len(loaders["train"]):
                print(f"  epoch {epoch} | batch {step}/{len(loaders['train'])} | "
                      f"bce={run_loss/n_seen:.4f}")

        y_score, y_true, _ = collect_scores(model, loaders["val"], device, use_amp)
        vm = compute_tongue_metrics(y_true, y_score, eval_classes)
        print(f"  epoch {epoch} | VAL macro_f1(valid)={vm['macro_f1_valid']:.4f} "
              f"macro_f1(all21)={vm['macro_f1_all21']:.4f} "
              f"subset_acc={vm['subset_accuracy']:.4f} auc={vm['auc_valid']:.4f}")
        history.append({"epoch": epoch, "train_bce": run_loss / max(n_seen, 1),
                        "val_macro_f1_valid": vm["macro_f1_valid"],
                        "val_macro_f1_all21": vm["macro_f1_all21"]})

        if vm["macro_f1_valid"] > best_val:
            best_val = vm["macro_f1_valid"]
            torch.save({"model_state": model.state_dict(),
                        "args": vars(args),
                        "finetune_stats": finetune_stats},
                       best_path)
            print(f"  ✓ 保存 best（val macro_f1_valid={best_val:.4f}）→ {best_path}")

    print(f"\n  训练完成，用时 {(time.time()-t0)/60:.1f} 分钟，best={best_val:.4f}")

    # ---------- 7. test 评估 ----------
    print(f"\n[7] 加载 best，test 评估...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    y_score, y_true, _ = collect_scores(model, loaders["test"], device, use_amp)
    tm = compute_tongue_metrics(y_true, y_score, eval_classes)

    print(f"\n📊 Stage B 视觉分支 test 指标（N={len(test_ds)}）")
    print(f"  Macro F1 (有效类 {len(eval_classes)}) : {tm['macro_f1_valid']:.4f}")
    print(f"  Macro F1 (全 21 类)      : {tm['macro_f1_all21']:.4f}")
    print(f"  Micro F1                 : {tm['micro_f1']:.4f}")
    print(f"  Subset Accuracy          : {tm['subset_accuracy']:.4f}")
    print(f"  AUC (有效类)             : {tm['auc_valid']:.4f}")
    print(f"\n  Per-class F1（支持数 = test 集该类的图像级正样本数）：")
    for i, c in enumerate(TONGUE_CLASSES):
        zh = c["zh"]
        f1 = tm["per_class_f1"][zh]
        sup = tm["per_class_support"][zh]
        bar = "█" * int(f1 * 30)
        tag = "" if i in eval_classes else "  (未计入主指标)"
        print(f"    {zh:8s} F1={f1:.4f} n={sup:4d} {bar}{tag}")

    # ---------- 8. 提取 10 病性视觉原型 ----------
    print(f"\n[8] 提取 10 病性视觉原型（source={args.proto_source} | strategy={args.proto_strategy}"
          f" | B6={args.prune_normal_white}）...")
    # 🔴 2026-09-16 修：原型提取**必须用确定性的全量 loader**，不能复用 loaders["train"]。
    #    loaders["train"] 是为训练建的（shuffle=True + drop_last=True），
    #    拿它做一次性全库遍历会**随机丢掉最后一个不满 batch**
    #    （batch=32 时 5594 % 32 = 26 张），且丢哪 26 张取决于该 seed 的 shuffle 顺序
    #    ⇒ n_images / n_pruned_white 这两个**纯 GT 统计量**变成 seed 的函数。
    #    09-16 实测后果：三 seed 的 n_pruned_white = 933 / 932 / 936（应恒为 937），
    #    血虚 = 188（应为 189），气虚/阳虚/痰湿 各 −19。
    #    修法 = 对齐仓库内既有正确范式 train_tongue_region.py:355-359,389-391：
    #    shuffle=False + 不 drop_last ⇒ 逐图全覆盖、顺序确定。
    proto_loaders = {
        sp: DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
        for sp, ds in datasets.items()
    }
    all_feats, all_gt, all_pred = [], [], []
    n_proto_img = 0
    for split in ("train", "val", "test"):
        ds = datasets[split]
        ys, yt, ft = collect_scores(model, proto_loaders[split], device, use_amp)
        assert len(yt) == len(ds), (
            f"原型语料覆盖不全：{split} 只遍历到 {len(yt)} 张，数据集有 {len(ds)} 张"
            f"（loader 被 drop_last / 数据集被截断？）")
        n_proto_img += len(yt)
        all_feats.append(ft)
        all_gt.append(torch.tensor(yt, dtype=torch.float32))
        all_pred.append(torch.tensor(ys, dtype=torch.float32))
    n_ds_img = sum(len(d) for d in datasets.values())
    assert n_proto_img == n_ds_img, f"原型语料 {n_proto_img} != 数据集总数 {n_ds_img}"
    print(f"  原型语料：train+val+test = {n_proto_img} 张（全量、无截断）")
    feats = torch.cat(all_feats, 0)
    det_labels = torch.cat(all_gt, 0) if args.proto_source == "gt" else torch.cat(all_pred, 0)

    protos, proto_meta = extract_syndrome_prototypes(
        model, feats, det_labels, use_gt=(args.proto_source == "gt"), l2_normalize=True,
        strategy=args.proto_strategy, prune_normal_white=args.prune_normal_white,
    )
    if args.prune_normal_white:
        print(f"  [B6] 剔除孤立白苔舌 {proto_meta.get('n_pruned_white', 0)} 张图（正常薄白）")
    proto_path = output_dir / "tongue_prototypes.pt"
    proto_blob = {
        "prototypes": protos,                  # (10, 512)，顺序 = SYNDROMES_ZH
        "syndromes": list(SYNDROMES_ZH),
        "strategy": args.proto_strategy,
        "meta": proto_meta,
        "source": args.proto_source,
        "feat_dim": args.feat_dim,
        "matrix_version": MATRIX_VERSION,
        "prune_normal_white": bool(args.prune_normal_white),
        "config": {"crop_bbox": args.crop_bbox, "unfreeze_blocks": args.unfreeze_blocks,
                   "epochs": args.epochs, "batch_size": args.batch_size,
                   "lr_head": args.lr_head, "lr_backbone": args.lr_backbone},
    }
    torch.save(proto_blob, proto_path)
    # 同时按策略名另存一份，便于 B-2 直接按策略取用（tongue_prototypes_{strategy}.pt）
    torch.save(proto_blob, output_dir / f"tongue_prototypes_{args.proto_strategy}.pt")
    print(f"  ✓ 原型已保存：{proto_path}")
    print(f"  {'病性':8s} {'支撑图数':>8s} {'权重和':>10s}  {'专属信号强度':>12s}")
    _pre = proto_meta.get("pre_norm") or [1.0] * NUM_SYNDROMES
    for j, s in enumerate(SYNDROMES_ZH):
        print(f"  {s:8s} {proto_meta['n_images'][j]:8d} {proto_meta['total_weight'][j]:10.1f}"
              f"  {_pre[j]:12.4f}")
    print("  （专属信号强度 = L2 归一化前模长，仅 centered 策略有意义；"
          "越接近 0 表示该病性缺乏可区分的舌象特征）")

    # ---------- 9. 报告 ----------
    report = {
        "stage": "B-1_tongue_vision_branch",
        "config": vars(args),
        "finetune_stats": finetune_stats,
        "data": {k: v.stats for k, v in datasets.items()},
        "train_counts": {TONGUE_CLASSES[i]["zh"]: train_counts[i] for i in range(NUM_TONGUE_CLASSES)},
        "class_mask": {TONGUE_CLASSES[i]["zh"]: int(class_mask[i]) for i in range(NUM_TONGUE_CLASSES)},
        "eval_classes": [TONGUE_CLASSES[i]["zh"] for i in eval_classes],
        "history": history,
        "best_val_macro_f1_valid": best_val,
        "test": tm,
        "prototypes": {
            "path": str(proto_path),
            "source": args.proto_source,
            "strategy": args.proto_strategy,
            "matrix_version": MATRIX_VERSION,
            "prune_normal_white": bool(args.prune_normal_white),
            "n_pruned_white": proto_meta.get("n_pruned_white", 0),
            "n_images": dict(zip(SYNDROMES_ZH, proto_meta["n_images"])),
            "total_weight": dict(zip(SYNDROMES_ZH, proto_meta["total_weight"])),
            "pre_norm": dict(zip(SYNDROMES_ZH, proto_meta.get("pre_norm") or [])),
        },
        "known_data_issues": [
            "test split 有 7 个 bbox 的 category_id=21 越界（categories 只定义 0-20），已忽略",
            "三个 split 的 classes.txt 顺序不一致（train 把 xinfeiao/xinfeitu 写反）→ 按 pinyin 索引规避",
            "4 个凸起类实例极少（piweitu 0 / shenqutu 2 / xinfeitu 2 / gandantu 9）",
            f"训练集无标注图 {datasets['train'].stats['n_images_no_label']} 张（默认保留为全阴样本）",
        ],
    }
    report_path = output_dir / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存：{report_path}")

    # ---- 交接摘要：结果与产物 ----
    H.section("结果")
    H.kv("seed", args.seed)
    H.kv("best_val_macro_f1_valid", best_val)
    H.kv("test_macro_f1", tm.get("macro_f1", "-"))
    H.section("产物")
    H.artifact(str(best_path))
    H.artifact(str(proto_path))
    H.artifact(str(report_path))
    H.line("")
    H.line("下一步（B-2 上游）：")
    H.line(f"  python eval/gen_random_prototypes.py --proto {proto_path} \\")
    H.line(f"      --mode shuffle  --seed {args.seed} --output {proto_path.parent}/tongue_prototypes_centered_shuffle.pt")
    H.line(f"  python eval/gen_random_prototypes.py --proto {proto_path} \\")
    H.line(f"      --mode gaussian --seed {args.seed} --output {proto_path.parent}/tongue_prototypes_centered_gaussian.pt")
    print("=" * 74)


if __name__ == "__main__":
    raise SystemExit(H.run(main))
