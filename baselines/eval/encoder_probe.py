"""
视觉编码器横向对比（frozen-feature linear probe）
==================================================

要回答的问题
------------
B-1 的 21 类舌象分类 macro F1 只有 ~0.33，且"苔/色类学得好、细粒度形态学不好"。
一个关键假设是：**BiomedCLIP 的视觉塔在舌象上不匹配** ——
它是拿 1500 万条英文生物医学图文对（放射/病理/科研插图）训出来的，
而舌象是"人体部位的自然照片"，域差很大。

本脚本用最干净的方式验证：**冻结特征 + 线性探针**（不微调 backbone，
排除训练策略干扰），横向比较三个编码器在同一任务上的上限。

    biomedclip  : microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    openai_clip : open_clip ViT-B-16 / openai（通用域）
    dinov2      : facebook/dinov2-base（自监督，无文本，擅长细粒度形态）

判据
----
· 若通用域编码器（clip/dinov2）显著优于 biomedclip
  → 论文可写"生物医学域 CLIP 在舌象上存在域失配"，并据此换编码器重做 B-1/B-2
· 若三者接近
  → 瓶颈在数据/标注（标签噪声、稀有类样本不足），而非编码器
  → 应转向"按 bbox 逐框训练"或接受该上限

成本：每编码器特征提取约 1-2 分钟（GPU），线性探针数秒。全跑约 5-10 分钟。

用法
----
    python eval/encoder_probe.py --tongue_root "/root/autodl-tmp/.../shezhenv3-coco"
    python eval/encoder_probe.py --crop_bbox            # 裁剪输入下的同一对比
    python eval/encoder_probe.py --encoders dinov2      # 只跑一个
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.tongue_coco_loader import DEFAULT_TONGUE_ROOT, build_datasets
from models.tongue_label_mapping import NUM_TONGUE_CLASSES, TONGUE_CLASSES

PROBE_VERSION = "2026-09-14c"

# 只在这些类上算"有效类 macro F1"（与 B-1 口径一致：正样本数 ≥ 阈值）
EVAL_MIN_POS = 5


# ============================================================
# 编码器注册表
# ============================================================

def _build_biomedclip(device: str):
    import open_clip
    tag = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, preprocess = open_clip.create_model_from_pretrained(tag)
    model = model.to(device).eval()

    def encode(x):
        return model.encode_image(x)

    with torch.no_grad():
        dim = int(encode(torch.zeros(1, 3, 224, 224, device=device)).shape[-1])
    return encode, preprocess, dim, "BiomedCLIP ViT-B/16 (biomedical)"


def _build_openai_clip(device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model = model.to(device).eval()

    def encode(x):
        return model.encode_image(x)

    with torch.no_grad():
        dim = int(encode(torch.zeros(1, 3, 224, 224, device=device)).shape[-1])
    return encode, preprocess, dim, "OpenAI CLIP ViT-B/16 (general)"


def _build_dinov2(device: str):
    from torchvision import transforms as T
    from transformers import AutoModel

    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    preprocess = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    def encode(x):
        out = model(pixel_values=x)
        return out.last_hidden_state[:, 0]          # CLS token

    with torch.no_grad():
        dim = int(encode(torch.zeros(1, 3, 224, 224, device=device)).shape[-1])
    return encode, preprocess, dim, "DINOv2 ViT-B/14 (self-supervised)"


ENCODERS = {
    "biomedclip": _build_biomedclip,
    "openai_clip": _build_openai_clip,
    "dinov2": _build_dinov2,
}


# ============================================================
# 特征提取
# ============================================================

@torch.no_grad()
def extract_features(datasets, encode, device: str, batch_size: int = 64):
    from torch.utils.data import DataLoader

    out = {}
    for split in ("train", "val", "test"):
        ds = datasets[split]
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        feats, labels = [], []
        t0 = time.time()
        for imgs, target, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                f = encode(imgs)
            feats.append(f.float().cpu())
            labels.append(target)
        F_ = torch.cat(feats)
        Y = torch.cat(labels)
        out[split] = (F_, Y)
        print(f"    [{split}] {tuple(F_.shape)}  ({time.time()-t0:.0f}s)")
    return out


# ============================================================
# 线性探针
# ============================================================

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list[int]) -> float:
    f1s = []
    for c in classes:
        tp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 1)).sum())
        fp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 0)).sum())
        fn = int(((y_pred[:, c] == 0) & (y_true[:, c] == 1)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_f1(y_true, y_pred):
    out = {}
    for c in range(y_true.shape[1]):
        tp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 1)).sum())
        fp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 0)).sum())
        fn = int(((y_pred[:, c] == 0) & (y_true[:, c] == 1)).sum())
        out[c] = 0.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return out


def auc_macro(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    scores = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].min() == y_true[:, c].max():
            continue
        scores.append(roc_auc_score(y_true[:, c], y_score[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def train_probe(feats, device: str, epochs: int = 300, lr: float = 1e-3,
                l2_norm: bool = True, verbose: bool = True):
    """在缓存特征上训一个线性多头（多标签 BCE + pos_weight）。"""
    Ftr, Ytr = feats["train"]
    Fva, Yva = feats["val"]
    Fte, Yte = feats["test"]

    if l2_norm:
        Ftr, Fva, Fte = (F.normalize(x, p=2, dim=-1) for x in (Ftr, Fva, Fte))

    Ftr, Fva, Fte = Ftr.to(device), Fva.to(device), Fte.to(device)
    Ytr_t = Ytr.to(device)

    pos = Ytr_t.sum(dim=0).clamp(min=1.0)
    neg = Ytr_t.size(0) - pos
    pos_weight = (neg / pos).clamp(max=50.0)

    head = nn.Linear(Ftr.size(1), NUM_TONGUE_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    valid_cls = [c for c in range(NUM_TONGUE_CLASSES)
                 if int(Yva[:, c].sum()) >= EVAL_MIN_POS]

    Yva_np = Yva.numpy().astype(int)

    def _select_score(pv_np: np.ndarray) -> float:
        """优先用 AUC；sklearn 缺失时退化为 macro F1@0.5。"""
        a = auc_macro(Yva.numpy(), pv_np)
        if a == a:  # not NaN
            return a
        return macro_f1(Yva_np, (pv_np >= 0.5).astype(int), valid_cls)

    best = {"score": -1.0, "state": None, "epoch": -1}
    for ep in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        loss = loss_fn(head(Ftr), Ytr_t)
        loss.backward()
        opt.step()
        if ep % 25 == 0 or ep == epochs:
            head.eval()
            with torch.no_grad():
                pv = torch.sigmoid(head(Fva)).cpu().numpy()
            sc = _select_score(pv)
            if sc > best["score"]:
                best = {"score": sc, "state": {k: v.clone() for k, v in head.state_dict().items()},
                        "epoch": ep}
    if best["state"] is not None:
        head.load_state_dict(best["state"])

    head.eval()
    with torch.no_grad():
        pv = torch.sigmoid(head(Fva)).cpu().numpy()
        pt = torch.sigmoid(head(Fte)).cpu().numpy()

    # 阈值在 val 上选（0.05~0.95），再报 test —— 与 B-1 后处理口径一致
    best_thr, best_val_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        f1 = macro_f1(Yva.numpy().astype(int), (pv >= thr).astype(int), valid_cls)
        if f1 > best_val_f1:
            best_val_f1, best_thr = f1, float(thr)

    Yte_np = Yte.numpy().astype(int)
    pred = (pt >= best_thr).astype(int)
    res = {
        "val_macro_f1": float(best_val_f1),
        "val_auc": float(auc_macro(Yva.numpy(), pv)),
        "test_macro_f1_valid": float(macro_f1(Yte_np, pred, valid_cls)),
        "test_auc": float(auc_macro(Yte_np, pt)),
        "best_threshold": best_thr,
        "per_class_f1": {TONGUE_CLASSES[c]["zh"]: float(v)
                         for c, v in per_class_f1(Yte_np, pred).items()},
        "valid_classes": [TONGUE_CLASSES[c]["zh"] for c in valid_cls],
    }
    if verbose:
        print(f"    val macroF1={res['val_macro_f1']:.4f} (thr={best_thr:.2f}) | "
              f"test macroF1={res['test_macro_f1_valid']:.4f} | test AUC={res['test_auc']:.4f}")
    return res


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="视觉编码器 frozen-feature 对比")
    ap.add_argument("--tongue_root", default=DEFAULT_TONGUE_ROOT)
    ap.add_argument("--encoders", default="biomedclip,openai_clip,dinov2",
                    help="逗号分隔，可选：biomedclip / openai_clip / dinov2")
    ap.add_argument("--crop_bbox", action="store_true")
    ap.add_argument("--crop_margin", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=None, help="调试用")
    ap.add_argument("--out", default=str(_PROJECT_ROOT / "runs" / "encoder_probe.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wanted = [e.strip() for e in args.encoders.split(",") if e.strip()]
    print("=" * 76)
    print(f"视觉编码器 frozen-feature 对比（{PROBE_VERSION}）")
    print(f"  device={device} | crop_bbox={args.crop_bbox} | 编码器={wanted}")
    print("=" * 76)

    results = {}
    for name in wanted:
        if name not in ENCODERS:
            print(f"\n[{name}] 未知编码器，跳过（可选 {list(ENCODERS)}）")
            continue
        print(f"\n[{name}] 加载...")
        try:
            encode, preprocess, dim, desc = ENCODERS[name](device)
        except Exception as e:  # 下载失败 / 环境缺包不致命
            print(f"  ✗ 加载失败：{type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        print(f"  ✓ {desc} | 特征维={dim}")

        datasets = build_datasets(
            args.tongue_root, preprocess=preprocess,
            crop_bbox=args.crop_bbox, crop_margin=args.crop_margin,
            max_samples=args.max_samples,
        )
        print("  提取冻结特征...")
        feats = extract_features(datasets, encode, device, args.batch_size)
        print("  训练线性探针（val 选阈值 → test 报告）...")
        res = train_probe(feats, device)
        res.update({"desc": desc, "feat_dim": dim})
        results[name] = res
        del feats
        torch.cuda.empty_cache()

    # ---------- 汇总 ----------
    ok = {k: v for k, v in results.items() if "error" not in v}
    if ok:
        print("\n" + "=" * 76)
        print("汇总（test macro F1，有效类 / val 阈値）")
        print("=" * 76)
        print(f"{'编码器':<16s} {'特征维':>7s} {'test macroF1':>13s} {'test AUC':>10s} {'thr':>6s}")
        print("-" * 76)
        for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["test_macro_f1_valid"]):
            print(f"{k:<16s} {v['feat_dim']:>7d} {v['test_macro_f1_valid']:>13.4f} "
                  f"{v['test_auc']:>10.4f} {v['best_threshold']:>6.2f}")

        best = max(ok.items(), key=lambda kv: kv[1]["test_macro_f1_valid"])
        print(f"\n最佳编码器：{best[0]}（{best[1]['desc']}）"
              f"  test macroF1={best[1]['test_macro_f1_valid']:.4f}")

        # 关注类对比（判断"细粒度形态"是否是域差问题）
        watch = ["齿痕舌", "胖大舌", "紫舌", "红点舌", "裂纹舌", "白苔舌",
                 "肝胆区凹陷", "脾胃区凹陷"]
        print(f"\n关注类逐类对比（test F1）：")
        header = f"{'舌象':<10s}" + "".join(f"{k:>14s}" for k in ok)
        print(header)
        print("-" * len(header))
        for w in watch:
            row = f"{w:<10s}" + "".join(
                f"{ok[k]['per_class_f1'].get(w, float('nan')):>14.4f}" for k in ok)
            print(row)

        # 结论提示
        if "biomedclip" in ok:
            b = ok["biomedclip"]["test_macro_f1_valid"]
            better = [k for k in ok if k != "biomedclip"
                      and ok[k]["test_macro_f1_valid"] > b + 0.03]
            print()
            print("【判据】以 biomedclip 为基准（无微调上限）：")
            if better:
                print(f"  → {better} 明显更好（>+0.03）⇒ 编码器存在域失配，"
                      f"建议换编码器重做 B-1/B-2")
            else:
                print("  → 无明显更优者 ⇒ 瓶颈在数据/标注（稀有类样本、标签一致性），"
                      "不是在编码器；应转向按 bbox 逐框训练或接受该上限")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"version": PROBE_VERSION, "config": vars(args),
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存：{out}")
    print("=" * 76)


if __name__ == "__main__":
    main()
