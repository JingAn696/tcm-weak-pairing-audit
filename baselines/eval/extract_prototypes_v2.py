"""
Stage B · 病性视觉原型重新提取（三种构造策略对比）
==================================================

**为什么需要这个脚本**

B-1 首轮用 "mean" 策略产出的 10 个病性原型，8 对余弦相似度 > 0.95、前 3 对 > 0.99：
    气虚×痰湿 0.9943 / 气虚×阳虚 0.9938 / 阳虚×痰湿 0.9909 / 湿热×实热 0.9834 ...
即这些原型**在实际意义上几乎无法区分**，B-2 的"文本查询原型"注意力会完全失去区分度。

根因：`mean` 策略 = 该病性所有来源图的特征平均。而映射来源高度重叠 ——
白苔舌一个类就覆盖 73% 的图，且同时映射到气虚/阳虚/痰湿，
于是这三者的"平均值"几乎是同一个东西。

**本脚本重新提取并对比三种策略**

| 策略 | 做法 | 预期 |
|---|---|---|
| `mean` | 病性来源图的特征平均 | 基线（已知失效） |
| `centered` | 先算 21 个舌象类均值 μ_i，减去全局均值 μ_0 得 δ_i，病性原型 = Σ M[i,j]·δ_i | 高频共享来源自动降权（μ_白苔 ≈ μ_0 → δ_白苔 ≈ 0） |
| `centered_idf` | 在 centered 上再乘独占性权重 1/(该舌象映射到的病性数) | 只映射血虚的剥苔舌权重高；映射 3 个病性的白苔舌权重低 |

用法（AutoDL）
--------------
    cd /root/autodl-tmp/<repo root>
    python eval/extract_prototypes_v2.py --ckpt runs/tongue_branch/best_model.pt

产出
----
    runs/tongue_branch/tongue_prototypes_{strategy}.pt   各策略原型
    runs/tongue_branch/prototype_strategy_report.json    对比报告
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _SCRIPT_DIR.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from data.tongue_coco_loader import DEFAULT_TONGUE_ROOT, build_datasets  # noqa: E402
from models.tongue_label_mapping import MATRIX_VERSION, SYNDROMES_ZH  # noqa: E402
from utils.run_handoff import Handoff  # noqa: E402

# 运行交接摘要（2026-09-15）：收尾时把关键信息同时打印 + 落盘 handoff_*.txt
H = Handoff("extract_prototypes_v2")

# ---- 依赖版本自检（2026-09-14 加入）----------------------------------------
# 教训：本脚本依赖新版 tongue_vision_branch.py（含 strategy 参数与 prototype_similarity）。
# 若服务器上是旧文件，原生 ImportError 只报"cannot import name ..."，
# 看不出"该传哪个文件、传哪一版"。这里把它变成一句能直接照做的提示。
try:
    from models.tongue_vision_branch import (  # noqa: E402
        FEAT_DIM,
        PROTO_VERSION,
        TongueVisionBranch,
        extract_syndrome_prototypes,
        load_biomedclip_visual,
        prototype_similarity,
    )
except ImportError as _exc:
    import models.tongue_vision_branch as _mv  # 模块本体已在 sys.modules 里，可安全窥探

    _REQUIRED = {
        "extract_syndrome_prototypes": "带 strategy 参数（mean/centered/centered_idf）的新版",
        "prototype_similarity": "原型两两余弦相似度诊断（新版新增）",
    }
    _missing = [n for n in _REQUIRED if not hasattr(_mv, n)]
    print("=" * 78)
    print("✗ 导入失败：服务器上的 models/tongue_vision_branch.py 不是最新版")
    print(f"  实际加载文件：{getattr(_mv, '__file__', '?')}")
    if _missing:
        print(f"  缺失函数：{', '.join(_missing)}")
        for _n in _missing:
            print(f"    - {_n}()：{_REQUIRED[_n]}")
    else:
        print("  （函数齐全，可能是其它导入错误，见下方原始异常）")
    print("\n  修复步骤：")
    print("    1) 把本机 models/tongue_vision_branch.py 重新上传覆盖")
    print("    2) 清掉可能存在的旧字节码缓存：rm -rf models/__pycache__")
    print("    3) 验证版本：python -c \"from models.tongue_vision_branch import PROTO_VERSION; print(PROTO_VERSION)\"")
    print("       期望输出：2026-09-15")
    print("=" * 78)
    # 模块级 import 失败时 H.run() 还没机会执行 → 这里手动出摘要（跑崩时最需要）
    H.fail(f"依赖版本不匹配：{type(_exc).__name__}: {_exc}")
    H.kv("加载的文件", getattr(_mv, "__file__", "?"))
    H.kv("缺失函数", ", ".join(_missing) if _missing else "（无，可能是其它导入错误）")
    H.line("修复：重传 models/tongue_vision_branch.py → rm -rf models/__pycache__ → 重新运行")
    H.flush(exit_code=2)
    raise SystemExit(2) from _exc

from training.train_tongue_branch import collect_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="病性视觉原型重新提取（三策略对比）")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tongue_root", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--proto_source", choices=["gt", "pred"], default="gt",
                    help="gt = 用检测标签（干净）；pred = 用模型预测（消融）")
    ap.add_argument("--no_prune_white", dest="prune_normal_white", action="store_false",
                    help="关闭 B6 判据（剔除孤立白苔舌）。默认开启（领域专家 2026-09-15 定稿）；"
                         "敏感性分析用旧口径时才加此开关")
    ap.set_defaults(prune_normal_white=True)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent

    print("=" * 78)
    print("Stage B · 病性视觉原型重新提取（mean / centered / centered_idf 对比）")
    print("=" * 78)

    # ---------- 配置 ----------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {}) or {}
    crop_bbox = bool(cfg.get("crop_bbox", False))
    crop_margin = float(cfg.get("crop_margin", 0.15))
    model_name = cfg.get("model_name", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    feat_dim = int(cfg.get("feat_dim", FEAT_DIM))
    batch_size = int(cfg.get("batch_size", 32))
    tongue_root = args.tongue_root or cfg.get("tongue_root") or DEFAULT_TONGUE_ROOT

    print(f"\n[1] 载入 B-1 权重")
    print(f"  ckpt   : {ckpt_path}")
    print(f"  数据   : {tongue_root}")
    print(f"  配置   : crop_bbox={crop_bbox} | feat_dim={feat_dim} | source={args.proto_source}")
    print(f"  B6 判据: prune_normal_white={args.prune_normal_white}"
          f"（剔除孤立白苔舌 → 正常薄白，领域专家 2026-09-15 定稿）")
    print(f"  模块版本: {PROTO_VERSION}（应为 2026-09-15）")

    # ---- 交接摘要：配置段 ----
    H.set_out_dir(out_dir)
    H.section("配置")
    H.kv("ckpt", ckpt_path)
    H.kv("tongue_root", tongue_root)
    H.kv("crop_bbox", crop_bbox)
    H.kv("proto_source", args.proto_source)
    H.kv("prune_normal_white(B6)", args.prune_normal_white)
    H.kv("PROTO_VERSION", PROTO_VERSION)
    H.kv("MATRIX_VERSION", MATRIX_VERSION)

    # ---------- 数据 / 模型 ----------
    visual, preprocess, raw_dim, img_size = load_biomedclip_visual(
        model_name=model_name, device=device
    )
    datasets = build_datasets(
        root=tongue_root, preprocess=preprocess,
        crop_bbox=crop_bbox, crop_margin=crop_margin,
    )
    loaders = {
        s: DataLoader(ds, batch_size=batch_size * 2, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)
        for s, ds in datasets.items()
    }

    model = TongueVisionBranch(
        visual_backbone=visual, raw_dim=raw_dim, feat_dim=feat_dim
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---------- 全库提特征 ----------
    print(f"\n[2] 全库推理提特征（train+val+test）...")
    all_feats, all_gt, all_pred = [], [], []
    for split in ("train", "val", "test"):
        ys, yt, ft = collect_scores(model, loaders[split], device, use_amp)
        all_feats.append(ft)
        all_gt.append(torch.tensor(yt, dtype=torch.float32))
        all_pred.append(torch.tensor(ys, dtype=torch.float32))
        print(f"  [{split}] {ft.shape[0]} 图 → {tuple(ft.shape)}")

    feats = torch.cat(all_feats, 0)
    det_labels = torch.cat(all_gt, 0) if args.proto_source == "gt" else torch.cat(all_pred, 0)
    print(f"  合计 {feats.shape[0]} 图 × {feats.shape[1]} 维")

    # ---------- 三策略对比 ----------
    print(f"\n[3] 三种策略提取原型 + 余弦相似度诊断")
    print(f"  判定：>0.95 几乎重合（B-2 失效）｜0.85–0.95 偏高｜<0.85 可分\n")

    header = f"  {'策略':<15s} {'最大相似对':<22s} {'均值相似':>9s} {'>0.95':>7s} {'>0.85':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    results, protos_by_strategy = {}, {}
    for strat in ("mean", "centered", "centered_idf"):
        protos, meta = extract_syndrome_prototypes(
            model, feats, det_labels,
            use_gt=(args.proto_source == "gt"), l2_normalize=True, strategy=strat,
            prune_normal_white=args.prune_normal_white,
        )
        if strat == "mean" and args.prune_normal_white:
            print(f"  [B6] 剔除孤立白苔舌 {meta.get('n_pruned_white', 0)} 张图（正常薄白）")
        sim = prototype_similarity(protos)
        mp = sim["max_pair"]
        pair_str = f"{mp['a']}×{mp['b']} {mp['cosine']:.4f}" if mp else "—"
        print(f"  {strat:<15s} {pair_str:<22s} {sim['mean_pairwise']:>9.4f} "
              f"{sim['n_over_095']:>7d} {sim['n_over_085']:>7d}")

        protos_by_strategy[strat] = protos
        results[strat] = {
            "prototype_similarity": {k: v for k, v in sim.items() if k != "pairs"},
            "top_pairs": sim["pairs"][:8],
            "meta": meta,
        }

    # ---------- 详细展示最佳策略 ----------
    best_strat = min(results, key=lambda s: results[s]["prototype_similarity"]["max_pair"]["cosine"]
                     if results[s]["prototype_similarity"]["max_pair"] else 9.9)
    print(f"\n[4] 最佳策略 = {best_strat}，其相似度最高的 5 对：")
    for d in results[best_strat]["top_pairs"][:5]:
        flag = "  ← 仍重合" if d["cosine"] > 0.95 else ("  ← 偏高" if d["cosine"] > 0.85 else "  ✓ 可分")
        print(f"    {d['a']:<6s} × {d['b']:<6s} {d['cosine']:.4f}{flag}")

    # ---------- 每个病性的支撑与"专属信号强度"（判断原型可信度） ----------
    print(f"\n[5] {best_strat} 策略下各病性的支撑与专属信号强度：")
    print(f"  {'病性':<6s} {'支撑图数':>8s} {'权重和':>9s}  {'专属信号强度':>12s}")
    _meta = results[best_strat]["meta"]
    _pre = _meta.get("pre_norm") or [float("nan")] * len(SYNDROMES_ZH)
    for s, n, w, p in zip(_meta["syndromes"], _meta["n_images"], _meta["total_weight"], _pre):
        tag = "  ← 无来源（视觉零信号）" if not n else ""
        print(f"  {s:<6s} {n:>8d} {w:>9.2f}  {p:>12.4f}{tag}")
    print("  （专属信号强度 = L2 归一化前模长；越接近 0 表示该病性越缺乏可区分的舌象特征。"
          "跨病性相对比较才有意义）")

    # ---------- 策略选择的提醒（避免只看"相似度最小"就选 idf） ----------
    print("\n  ⚠️ 选策略别只看相似度：centered_idf 会把「独占映射」的来源放大，")
    print("     而本数据集里独占映射的多是舌面分区类（模型学得最差，F1 0.00~0.19），")
    print("     故选型时要一并看 [5] 表的支撑图数与专属信号强度。")

    # ---------- 保存 ----------
    for strat, p in protos_by_strategy.items():
        path = out_dir / f"tongue_prototypes_{strat}.pt"
        torch.save({
            "prototypes": p,
            "syndromes": list(SYNDROMES_ZH),
            "strategy": strat,
            "meta": results[strat]["meta"],
            "source": args.proto_source,
            "feat_dim": feat_dim,
            "matrix_version": MATRIX_VERSION,
            "prune_normal_white": bool(args.prune_normal_white),
            "config": {"crop_bbox": crop_bbox, "unfreeze_blocks": cfg.get("unfreeze_blocks", 4)},
        }, path)
        print(f"\n  ✓ {strat:<15s} → {path}")

    report = {
        "stage": "B-1_prototype_strategies",
        "ckpt": str(ckpt_path),
        "proto_source": args.proto_source,
        "matrix_version": MATRIX_VERSION,
        "prune_normal_white": bool(args.prune_normal_white),
        "n_images_total": int(feats.shape[0]),
        "comparison": results,
        "best_strategy": best_strat,
        "note": ("mean 为 B-1 首轮所用，实测原型重合（8 对 >0.95）；"
                 "centered / centered_idf 通过扣除全局共同成分来恢复区分度。"),
    }
    rp = out_dir / "prototype_strategy_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*78}")
    print(f"结论：最佳策略 {best_strat}，最大原型相似度 "
          f"{results[best_strat]['prototype_similarity']['max_pair']['cosine']:.4f}"
          f"（mean 为 0.9943）")
    print(f"报告已保存：{rp}")
    print("=" * 78)

    # ---------- 交接摘要（与上面的结论同源） ----------
    _meta = results[best_strat]["meta"]
    H.section("结果")
    for strat in ("mean", "centered", "centered_idf"):
        if strat not in results:
            continue
        mp = results[strat]["prototype_similarity"].get("max_pair")
        H.kv(f"max_cosine[{strat}]",
             f"{mp['a']} × {mp['b']} = {mp['cosine']:.4f}" if mp else "无")
    H.kv("best_strategy", best_strat)

    H.section("各病性支撑图数 n_images（血虚应 ≈189）")
    H.kv("口径", _meta.get("n_images_mode",
                           "legacy（来源类实例数之和，可重复计数 → 多来源病性虚高）"))
    H.kv_dict(dict(zip(SYNDROMES_ZH, _meta["n_images"])))

    _pre = _meta.get("pre_norm") or []
    if _pre:
        H.section("各病性专属信号强度 pre_norm（跨病性相对比较才有意义）")
        H.kv_dict({k: round(float(v), 3) for k, v in zip(SYNDROMES_ZH, _pre)})

    if "n_pruned_white" in _meta:
        H.kv("B6_剔除孤立白苔舌", _meta["n_pruned_white"])
    H.kv("n_images_total", report["n_images_total"])

    H.section("产物")
    for strat in protos_by_strategy:
        H.artifact(out_dir / f"tongue_prototypes_{strat}.pt")
    H.artifact(rp)

    if any(int(n) == 0 for n in _meta["n_images"]):
        H.warn("存在支撑图数为 0 的病性（视觉零信号）→ 该列为纯先验，非证据")


if __name__ == "__main__":
    raise SystemExit(H.run(main))
