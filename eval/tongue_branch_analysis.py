"""
Stage B · B-1 结果后处理与诊断（阈值调优 + 原型质量）
====================================================

只读 B-1 的产物，**不重新训练**，做两件事：

1. 阈值调优
   B-1 训练用了 pos_weight 抗类别不平衡（白苔舌 73% vs 紫舌 3.4%），
   代价是模型输出概率**整体偏高**。直接按 0.5 阈值算 F1 会严重低估模型 ——
   典型症状就是「AUC 不低、F1 很低」。本脚本在 **val** 上扫阈值选最优，
   再回到 test 报告。

   ⚠️ 学术合规：阈值只能在 val 上定，在 test 上调属数据泄漏。
      论文 Method 里要写明"分类阈值在验证集上选取"。

2. 原型质量诊断
   B-1 产出的 10×512 病性原型是 B-2 的输入。**支撑图数达标 ≠ 原型可用**：
   若两个原型余弦相似度 > 0.95，说明它们实际没分开，B-2 的文本查询注意力会失效。
   （例如痰湿 6123 张支撑图，远多于其他，很可能被高频类主导而趋同于气虚/阳虚）

用法（AutoDL）
--------------
    cd /root/autodl-tmp/<repo root>
    python eval/tongue_branch_analysis.py --ckpt runs/tongue_branch/best_model.pt

    # 额外做每类独立阈值（val 正样本少的类会过拟合，谨慎使用）
    python eval/tongue_branch_analysis.py --ckpt runs/tongue_branch/best_model.pt --per_class
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent          # .../code/eval
_CODE_ROOT = _SCRIPT_DIR.parent                        # .../code
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data.tongue_coco_loader import DEFAULT_TONGUE_ROOT, build_datasets  # noqa: E402
from models.tongue_label_mapping import (  # noqa: E402
    NUM_SYNDROMES,
    NUM_TONGUE_CLASSES,
    SYNDROMES_ZH,
    TONGUE_CLASSES,
)
from models.tongue_vision_branch import (  # noqa: E402
    FEAT_DIM,
    TongueVisionBranch,
    load_biomedclip_visual,
)
from training.train_tongue_branch import (  # noqa: E402
    collect_scores,
    compute_tongue_metrics,
)

GRID = np.round(np.arange(0.05, 0.951, 0.01), 3)


# ============================================================
# 阈值扫描
# ============================================================

def sweep_global(y_true: np.ndarray, y_score: np.ndarray, classes: list[int]):
    """在给定类集合上扫全局阈值，返回 (最优阈值, 最优 macroF1, 整条曲线)。"""
    from sklearn.metrics import f1_score

    curve, best_t, best_f1 = [], 0.5, -1.0
    for t in GRID:
        pred = (y_score[:, classes] >= t).astype(int)
        f1 = float(f1_score(y_true[:, classes], pred, average="macro", zero_division=0))
        curve.append({"threshold": float(t), "macro_f1": f1})
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1, curve


def sweep_per_class(y_true: np.ndarray, y_score: np.ndarray, classes: list[int]):
    """每个类单独选阈值（val 正样本少的类会过拟合）。"""
    from sklearn.metrics import f1_score

    out = {}
    for i in classes:
        bt, bf = 0.5, -1.0
        for t in GRID:
            p = (y_score[:, i] >= t).astype(int)
            f = float(f1_score(y_true[:, i], p, average="binary", zero_division=0))
            if f > bf:
                bf, bt = f, float(t)
        out[i] = {"threshold": bt, "f1_on_val": bf, "val_pos": int(y_true[:, i].sum())}
    return out


# ============================================================
# 原型诊断
# ============================================================

def diagnose_prototypes(proto_path: Path):
    """10×10 余弦相似度矩阵 + 高相似对清单。"""
    data = torch.load(proto_path, map_location="cpu", weights_only=False)
    P = data["prototypes"].float()                       # (10, 512)
    norms = P.norm(dim=1)
    valid = norms > 1e-6                                 # 内风为零向量，排除

    idx_valid = [j for j in range(NUM_SYNDROMES) if bool(valid[j])]
    Pn = P[idx_valid] / norms[idx_valid].unsqueeze(1)
    S = (Pn @ Pn.T).numpy()

    pairs = []
    for a in range(len(idx_valid)):
        for b in range(a + 1, len(idx_valid)):
            sim = float(S[a, b])
            pairs.append({
                "a": SYNDROMES_ZH[idx_valid[a]],
                "b": SYNDROMES_ZH[idx_valid[b]],
                "cosine": sim,
            })
    pairs.sort(key=lambda d: -d["cosine"])

    return {
        "zero_prototypes": [SYNDROMES_ZH[j] for j in range(NUM_SYNDROMES) if not bool(valid[j])],
        "cosine_matrix": {
            SYNDROMES_ZH[idx_valid[a]]: {
                SYNDROMES_ZH[idx_valid[b]]: float(S[a, b]) for b in range(len(idx_valid))
            } for a in range(len(idx_valid))
        },
        "top_similar_pairs": pairs[:8],
        "meta": data.get("meta", {}),
    }


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Stage B B-1 结果后处理：阈值调优 + 原型诊断")
    ap.add_argument("--ckpt", type=str, required=True, help="runs/tongue_branch/best_model.pt")
    ap.add_argument("--tongue_root", type=str, default=None,
                    help="默认取 ckpt 里记录的路径")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="默认取 ckpt 所在目录")
    ap.add_argument("--per_class", action="store_true",
                    help="额外做每类独立阈值（val 样本少的类会过拟合）")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent

    print("=" * 74)
    print("Stage B · B-1 结果后处理（阈值调优 + 原型质量诊断）")
    print("=" * 74)

    # ---------- 1. 读 ckpt 与配置 ----------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {}) or {}
    crop_bbox = bool(cfg.get("crop_bbox", False))
    crop_margin = float(cfg.get("crop_margin", 0.15))
    model_name = cfg.get("model_name", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    feat_dim = int(cfg.get("feat_dim", FEAT_DIM))
    eval_min_pos = int(cfg.get("eval_min_pos", 5))
    batch_size = int(cfg.get("batch_size", 32))

    tongue_root = args.tongue_root or cfg.get("tongue_root") or DEFAULT_TONGUE_ROOT

    print(f"\n[1] 载入 B-1 权重")
    print(f"  ckpt        : {ckpt_path}")
    print(f"  数据        : {tongue_root}")
    print(f"  配置        : crop_bbox={crop_bbox} | feat_dim={feat_dim} | "
          f"eval_min_pos={eval_min_pos}")

    # ---------- 2. 数据 ----------
    visual, preprocess, raw_dim, img_size = load_biomedclip_visual(
        model_name=model_name, device=device
    )
    datasets = build_datasets(
        root=tongue_root, preprocess=preprocess,
        crop_bbox=crop_bbox, crop_margin=crop_margin,
    )
    val_ds, test_ds = datasets["val"], datasets["test"]

    val_counts = val_ds.class_counts()
    eval_classes = [i for i in range(NUM_TONGUE_CLASSES) if val_counts[i] >= eval_min_pos]
    print(f"  参与评估 {len(eval_classes)}/{NUM_TONGUE_CLASSES} 类（val 正样本 ≥{eval_min_pos}）")

    loaders = {
        "val": DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True),
    }

    # ---------- 3. 模型 ----------
    model = TongueVisionBranch(
        visual_backbone=visual, raw_dim=raw_dim, feat_dim=feat_dim
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  ✓ 权重载入完成（{sum(p.numel() for p in model.parameters())/1e6:.1f}M 参数）")

    # ---------- 4. 收集分数（用 bf16 加速）----------
    use_amp = torch.cuda.is_available()
    print(f"\n[2] 推理 val / test ...")
    yv_score, yv_true, _ = collect_scores(model, loaders["val"], device, use_amp)
    yt_score, yt_true, _ = collect_scores(model, loaders["test"], device, use_amp)
    print(f"  val  {yv_score.shape} | test {yt_score.shape}")

    # ---------- 5. val 上扫阈值 ----------
    print(f"\n[3] 在 val 上扫阈值（{GRID[0]:.2f} ~ {GRID[-1]:.2f}）...")
    t_best, f1_val_best, curve = sweep_global(yv_true, yv_score, eval_classes)
    f1_val_at_05 = compute_tongue_metrics(yv_true, yv_score, eval_classes, threshold=0.5)["macro_f1_valid"]
    print(f"  val macro F1 @0.50 = {f1_val_at_05:.4f}")
    print(f"  val macro F1 @{t_best:.2f} = {f1_val_best:.4f}   ← 最优阈值")

    # 曲线上的几个参考点
    shown = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print("  阈值-曲线采样：" + "  ".join(
        f"{p['threshold']:.2f}→{p['macro_f1']:.3f}" for p in curve
        if round(p["threshold"], 2) in shown
    ))

    # ---------- 6. test 上报告（两种阈值对比）----------
    tm05 = compute_tongue_metrics(yt_true, yt_score, eval_classes, threshold=0.5)
    tmopt = compute_tongue_metrics(yt_true, yt_score, eval_classes, threshold=t_best)

    print(f"\n[4] test 指标对比（阈值只在 val 上选，未用 test 调参）")
    print(f"  {'指标':<26s} {'@0.50':>10s} {'@' + f'{t_best:.2f}':>10s}   Δ")
    rows = [
        ("Macro F1 (有效类)", "macro_f1_valid"),
        ("Macro F1 (全 21 类)", "macro_f1_all21"),
        ("Micro F1", "micro_f1"),
        ("AUC (有效类)", "auc_valid"),
    ]
    print(f"  {'-'*62}")
    for label, key in rows:
        a, b = tm05[key], tmopt[key]
        print(f"  {label:<26s} {a:>10.4f} {b:>10.4f}   {b-a:+.4f}")

    # per-class 对比
    print(f"\n  Per-class F1（阈值 {t_best:.2f}）：")
    print(f"  {'舌象':<10s} {'@0.50':>8s} {'@opt':>8s} {'test 支持':>9s}")
    per = []
    for i, c in enumerate(TONGUE_CLASSES):
        zh = c["zh"]
        a = tm05["per_class_f1"][zh]
        b = tmopt["per_class_f1"][zh]
        sup = tmopt["per_class_support"][zh]
        tag = "" if i in eval_classes else "  (未计入)"
        print(f"  {zh:<10s} {a:>8.4f} {b:>8.4f} {sup:>9d}{tag}")
        per.append({"class": zh, "f1_at_0.50": a, "f1_at_opt": b, "test_support": sup,
                    "in_eval": i in eval_classes})

    improve = sorted(per, key=lambda d: -(d["f1_at_opt"] - d["f1_at_0.50"]))[:5]
    print(f"\n  阈值调优受益最大的 5 个类：")
    for d in improve:
        print(f"    {d['class']:<10s} {d['f1_at_0.50']:.4f} → {d['f1_at_opt']:.4f} "
              f"({d['f1_at_opt']-d['f1_at_0.50']:+.4f})")

    # ---------- 6b. val vs test 泛化 gap 诊断 ----------
    vm_opt = compute_tongue_metrics(yv_true, yv_score, eval_classes, threshold=t_best)
    print(f"\n  val vs test 逐类 F1（同一阈值 {t_best:.2f}）—— 定位泛化 gap 来源：")
    print(f"  {'舌象':<10s} {'val_F1':>8s} {'test_F1':>8s} {'Δ':>9s} {'val_n':>6s} {'test_n':>7s}")
    print(f"  {'-'*56}")
    gap_rows = []
    for i in eval_classes:
        zh = TONGUE_CLASSES[i]["zh"]
        v = vm_opt["per_class_f1"][zh]
        tt = tmopt["per_class_f1"][zh]
        vn = int(yv_true[:, i].sum())
        tn = int(yt_true[:, i].sum())
        print(f"  {zh:<10s} {v:>8.4f} {tt:>8.4f} {tt-v:>+9.4f} {vn:>6d} {tn:>7d}")
        gap_rows.append({"class": zh, "val_f1": v, "test_f1": tt,
                         "delta": tt - v, "val_n": vn, "test_n": tn})
    print(f"  {'-'*56}")
    print(f"  整体：val macro F1 {vm_opt['macro_f1_valid']:.4f} → "
          f"test {tmopt['macro_f1_valid']:.4f} "
          f"(gap {tmopt['macro_f1_valid']-vm_opt['macro_f1_valid']:+.4f})")

    # ---------- 7. 可选：每类阈值 ----------
    per_class_thr = None
    if args.per_class:
        print(f"\n[5] 每类独立阈值（⚠️ val 正样本少的类会过拟合）")
        per_class_thr = sweep_per_class(yv_true, yv_score, eval_classes)
        for i in eval_classes:
            d = per_class_thr[i]
            warn = "  ⚠️ val 样本过少，不可信" if d["val_pos"] < 20 else ""
            print(f"  {TONGUE_CLASSES[i]['zh']:<10s} thr={d['threshold']:.2f} "
                  f"val_f1={d['f1_on_val']:.3f} val_pos={d['val_pos']:4d}{warn}")

    # ---------- 8. 原型质量诊断 ----------
    print(f"\n[6] 10 病性原型质量诊断")
    proto_path = out_dir / "tongue_prototypes.pt"
    proto_diag = None
    if proto_path.exists():
        proto_diag = diagnose_prototypes(proto_path)
        meta = proto_diag["meta"]
        if proto_diag["zero_prototypes"]:
            print(f"  零向量原型（无舌象来源）：{', '.join(proto_diag['zero_prototypes'])}")
        print(f"\n  原型两两余弦相似度最高的 8 对：")
        for d in proto_diag["top_similar_pairs"]:
            flag = "  ← 几乎重合，B-2 难区分" if d["cosine"] > 0.95 else (
                "  ← 偏高" if d["cosine"] > 0.85 else "")
            print(f"    {d['a']:<6s} × {d['b']:<6s} {d['cosine']:.4f}{flag}")
        if "n_images" in meta:
            print(f"\n  支撑图数/权重和：")
            for j, s in enumerate(SYNDROMES_ZH):
                print(f"    {s:<6s} 图 {meta['n_images'][j]:5d} | 权重 {meta['total_weight'][j]:8.1f}")
    else:
        print(f"  ⚠️ 未找到 {proto_path}，跳过")

    # ---------- 9. 保存 ----------
    report = {
        "stage": "B-1_postprocess",
        "ckpt": str(ckpt_path),
        "config": {"crop_bbox": crop_bbox, "feat_dim": feat_dim,
                   "eval_min_pos": eval_min_pos, "n_eval_classes": len(eval_classes)},
        "threshold": {
            "selected_on": "val",
            "best": t_best,
            "val_macro_f1_at_0.50": f1_val_at_05,
            "val_macro_f1_at_best": f1_val_best,
            "curve": curve,
        },
        "test_at_0.50": tm05,
        "test_at_best": tmopt,
        "per_class_comparison": per,
        "val_vs_test": {
            "threshold": t_best,
            "val_macro_f1": vm_opt["macro_f1_valid"],
            "test_macro_f1": tmopt["macro_f1_valid"],
            "per_class": gap_rows,
        },
        "per_class_thresholds": (
            {TONGUE_CLASSES[i]["zh"]: per_class_thr[i] for i in per_class_thr}
            if per_class_thr else None
        ),
        "prototype_diagnosis": proto_diag,
        "note": "阈值在 val 上选取后固定，用于 test 报告；未使用 test 信息调参。",
    }
    out_path = out_dir / "threshold_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*74}")
    print(f"结论：test Macro F1(有效类) {tm05['macro_f1_valid']:.4f} → {tmopt['macro_f1_valid']:.4f}"
          f"（阈值 0.50 → {t_best:.2f}）")
    print(f"报告已保存：{out_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
