"""
BiomedCLIP Zero-Shot Baseline
=============================

冻结双编码器，纯推理：用 10 个证型 prompt 与样本计算相似度。
不训练任何参数，直接出 baseline 数字，作为后续 LoRA / 完整模型的对比基线。

输入：data/splits/{test,val}.csv（多标签，10 维）
输出：runs/zero_shot/{test,val}_report.json

用法：
  python baselines/biomedclip_zero_shot.py --split test --top_k 2
"""

from __future__ import annotations  # 让所有 type hint 延迟求值（本机无 numpy 也能 import）

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# === 路径常量：基于脚本位置计算，不依赖调用方的工作目录 ===
#   baselines/biomedclip_zero_shot.py → 父目录为 baselines/ → 父目录的父目录是 papers/code/（项目根）
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data")
_DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "runs" / "zero_shot")

try:
    import numpy as np
except ImportError:  # 本机无 numpy 也能 import（仅做结构检查）
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None

# 把项目根目录加入 import 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    torch = None

from models.biomedclip_wrapper import (
    BiomedCLIPWrapper,
    SYNDROME_TEXT_TEMPLATES_EN,
    SYNDROME_TEXT_TEMPLATES_ZH,
    build_zero_shot_prompts,
)
from eval.metrics import (
    compute_metrics,
    SYNDROME_NAMES_EN,
    SYNDROME_NAMES_ZH,
)


# 与 data/tcm_sd_loader.py 的 LABEL_HEADERS 完全一致（10 病性，格式：英文key|中文名）
#   与 data/splits/{train,val,test}.csv 的实际列名严格匹配
#   SYNDROME_NAMES_EN/ZH（来自 metrics）用于显示 & 指标，不用于这里
LABEL_COLS = [
    "qi_deficiency|气虚",         # 0
    "blood_deficiency|血虚",        # 1
    "yin_deficiency|阴虚",         # 2
    "yang_deficiency|阳虚",         # 3
    "qi_stagnation|气滞",          # 4
    "blood_stasis|血瘀",           # 5
    "phlegm_dampness|痰湿",         # 6
    "damp_heat|湿热",             # 7
    "wind|内风",                # 8
    "excess_heat|实热",            # 9
]


def load_test_data(csv_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    读 test/val CSV，返回 (texts, labels)。

    texts: 拼接 chief_complaint + description + detection
    labels: [N, 10] 0/1 多标签向量（按 SYNDROME_NAMES_EN 顺序）
    """
    df = pd.read_csv(csv_path)

    # 拼接问诊文本
    texts = (
        df["chief_complaint"].fillna("").astype(str)
        + "。 "
        + df["description"].fillna("").astype(str)
        + "。 "
        + df["detection"].fillna("").astype(str)
    ).tolist()

    # 10 维 label 向量（按 LABEL_COLS 顺序）
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺 label 列：{missing}。期望列：{LABEL_COLS}")
    labels = df[LABEL_COLS].values.astype(int)

    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="BiomedCLIP Zero-Shot Baseline（多标签）")
    parser.add_argument("--data_root", type=str, default=_DEFAULT_DATA_ROOT,
                       help="数据根目录（含 splits_v2/ 子目录）")
    parser.add_argument("--splits_dir", type=str, default="splits_v2",
                       help="split 子目录名（默认 splits_v2：去重+按 user_id 分组的干净版；"
                            "传 splits 用旧版做对比）")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=2,
                       help="多标签预测 top-k（每样本预测 k 个证型）")
    parser.add_argument("--template_lang", type=str, default="en", choices=["en", "zh"],
                       help="prompt 模板语言（推荐 en，BiomedCLIP 在 PubMed 英文训练）")
    parser.add_argument("--output_dir", type=str, default=_DEFAULT_OUTPUT_DIR,
                       help="报告输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  BiomedCLIP Zero-Shot Baseline（{args.split} split）")
    print("=" * 70)

    # 1. 加载 BiomedCLIP
    print("\n[1] 加载 BiomedCLIP 模型（首次会从 HF 下载 ~1GB）...")
    wrapper = BiomedCLIPWrapper(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    wrapper.load()
    print(f"  ✓ 模型加载完成（device={wrapper._device}）")

    # 2. 加载数据
    print(f"\n[2] 加载 {args.split} 数据...")
    csv_path = Path(args.data_root) / args.splits_dir / f"{args.split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在：{csv_path}")
    texts, labels = load_test_data(csv_path)
    print(f"  ✓ 加载 {len(texts)} 个样本，{labels.shape[1]} 维标签")

    # 3. 构建 zero-shot prompt ensemble
    print(f"\n[3] 构建 10 证型 prompt ensemble（lang={args.template_lang}）...")
    templates = SYNDROME_TEXT_TEMPLATES_EN if args.template_lang == "en" else SYNDROME_TEXT_TEMPLATES_ZH
    prompt_groups = build_zero_shot_prompts(templates)

    # 4. 预编码所有 prompts（prompt ensemble 均值）
    print("[4] 预编码 10 证型 prompt（每个取 3 条模板均值）...")
    prompt_features = []
    for i, (key, prompts) in enumerate(prompt_groups.items()):
        feat = wrapper.encode_text(prompts)            # [3, 512] 已 L2 归一化
        feat = feat.mean(dim=0, keepdim=True)          # [1, 512]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        prompt_features.append(feat)
        print(f"    {i + 1:2d}. {key:20s} 编码完成")
    prompt_features = torch.cat(prompt_features, dim=0)  # [10, 512]

    # 5. 编码问诊文本（按 batch）+ 计算相似度
    print(f"\n[5] 编码问诊文本（batch_size={args.batch_size}）...")
    all_logits = []
    n_batches = (len(texts) - 1) // args.batch_size + 1

    with torch.no_grad():
        for batch_idx in range(n_batches):
            i = batch_idx * args.batch_size
            batch_texts = texts[i:i + args.batch_size]
            tokens = wrapper._tokenizer(batch_texts).to(wrapper._device)
            text_features = wrapper._model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = (text_features @ prompt_features.T).cpu()  # [B, 10]
            all_logits.append(logits)

            if batch_idx % 20 == 0:
                print(f"    batch {batch_idx + 1}/{n_batches}")

    logits = torch.cat(all_logits).numpy()  # [N, 10]

    # 6. 多标签预测（top-k 阈值化）
    print(f"\n[6] 多标签预测（top-{args.top_k} 阈值化）...")
    y_pred = np.zeros_like(labels)
    top_k_indices = np.argsort(-logits, axis=1)[:, :args.top_k]
    for i, idx in enumerate(top_k_indices):
        y_pred[i, idx] = 1

    # 7. 算指标
    print(f"\n[7] 计算多标签指标...")
    metrics = compute_metrics(labels, y_pred, y_score=logits)

    # 8. 打印报告
    print(f"\n{'=' * 70}")
    print(f"  📊 Zero-Shot Baseline 结果（{args.split}, N={len(texts)}）")
    print(f"{'=' * 70}")
    print(f"  Subset Accuracy : {metrics['subset_accuracy']:.4f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Micro F1        : {metrics['micro_f1']:.4f}")
    print(f"  Weighted F1     : {metrics['weighted_f1']:.4f}")
    print(f"  Samples F1      : {metrics['samples_f1']:.4f}")
    print(f"  Hamming Loss    : {metrics['hamming_loss']:.4f}")
    print(f"  AUC OvR Macro   : {metrics['auc_ovr_macro']:.4f}")
    print(f"  AUC OvR Micro   : {metrics['auc_ovr_micro']:.4f}")
    print()
    print(f"  Per-class F1：")
    for syn_zh in SYNDROME_NAMES_ZH:
        print(f"    {syn_zh:6s} F1 = {metrics[f'f1_{syn_zh}']:.4f}")

    # 9. 保存报告
    report_path = output_dir / f"{args.split}_report.json"
    report = {
        "config": vars(args),
        "metrics": metrics,
        "n_samples": int(len(texts)),
        "n_classes": int(labels.shape[1]),
        "class_distribution": labels.sum(axis=0).tolist(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ 报告已保存：{report_path}")


if __name__ == "__main__":
    main()