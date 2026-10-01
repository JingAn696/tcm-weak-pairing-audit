"""
TCM-SD 文本 Baseline（eval-only 模式，baseline 1 配套）
========================================================

用途：baseline 1 训练完崩溃后，从 best_model checkpoint 重新加载 + 出完整多标签报告。
避免重新训练 3 小时。

用法：
  python baselines/eval_tcm_sd_text.py --checkpoint runs/tcm_text_macbert/checkpoint-2472
  # checkpoint-XXXX 是 baseline 1 训练时 Trainer 自动存的路径
"""

from __future__ import annotations

# === 国内下载模型：自动设 HF 镜像（与 baseline 0/1 一致）===
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import sys
from pathlib import Path

# === 让 eval/ 能被 import ===
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === 复用 baseline 1 的 Dataset 和 LABEL_COLS ===
from baselines.tcm_sd_text_macbert import TCMTextDataset, LABEL_COLS


def main():
    parser = argparse.ArgumentParser(description="从 best_model 重新评估 TCM-SD 文本 baseline")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="best model checkpoint 路径（含 pytorch_model.bin）")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base",
                       help="原模型架构（chinese-macbert-base）")
    parser.add_argument("--data_root", type=str, default=str(_PROJECT_ROOT / "data"))
    parser.add_argument("--splits_dir", type=str, default="splits_v2",
                       help="split 子目录名（默认 splits_v2 干净版）")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str,
                       default=str(_PROJECT_ROOT / "runs" / "tcm_text_macbert" / "test_report.json"))
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"],
                       help="评估哪个 split")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  TCM-SD 文本 Baseline Eval-Only（{args.split}, checkpoint={args.checkpoint}）")
    print("=" * 70)

    # 1. 加载 tokenizer（从原模型 repo）+ 模型（从 checkpoint）
    #   Trainer 默认 save_strategy="epoch" 只存 model weights，不存 tokenizer
    #   所以 tokenizer 走原模型 repo（HF 缓存），model 走 checkpoint
    print(f"\n[1] 加载 tokenizer（来自 {args.model_name}）+ 模型（来自 {args.checkpoint}）...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"  ✓ 模型加载完成（device={device}, 参数量={sum(p.numel() for p in model.parameters())/1e6:.1f}M）")

    # 2. 加载 test 数据
    print(f"\n[2] 加载 {args.split} 数据...")
    csv_path = Path(args.data_root) / args.splits_dir / f"{args.split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在：{csv_path}")
    dataset = TCMTextDataset(csv_path, tokenizer, args.max_length)
    print(f"  ✓ 加载 {len(dataset)} 个样本")

    # 3. 跑预测（手写 DataLoader，避免再走 Trainer 依赖链）
    print(f"\n[3] 跑预测（batch_size={args.batch_size}）...")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(out.logits.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    print(f"  ✓ 预测完成：logits shape={logits.shape}")

    # 4. 多标签完整指标
    print(f"\n[4] 生成多标签完整指标...")
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)
    y_true = dataset.labels.astype(int)

    from eval.metrics import compute_metrics
    final_metrics = compute_metrics(y_true, y_pred, y_score=probs)

    print(f"\n📊 Final Multi-Label Metrics ({args.split}):")
    print(f"  Subset Accuracy : {final_metrics['subset_accuracy']:.4f}")
    print(f"  Macro F1        : {final_metrics['macro_f1']:.4f}")
    print(f"  Micro F1        : {final_metrics['micro_f1']:.4f}")
    print(f"  Weighted F1     : {final_metrics['weighted_f1']:.4f}")
    print(f"  Samples F1      : {final_metrics['samples_f1']:.4f}")
    print(f"  Hamming Loss    : {final_metrics['hamming_loss']:.4f}")
    print(f"  AUC OvR Macro   : {final_metrics['auc_ovr_macro']:.4f}")
    print(f"  AUC OvR Micro   : {final_metrics['auc_ovr_macro']:.4f}")
    print(f"\nPer-class F1:")
    for name, f1 in final_metrics.items():
        if name.startswith("f1_"):
            syn_zh = name[3:]
            bar = "█" * int(f1 * 30)
            print(f"  {syn_zh:6s}: {f1:.4f} {bar}")

    # 5. 保存报告
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存：{output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()