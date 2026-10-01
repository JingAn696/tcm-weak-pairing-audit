"""
TCM-SD 文本-only Baseline（MacBERT 微调，baseline 1）
=====================================================

目的：建立"纯文本多标签分类"基线，对比 zero-shot / 多模态完整模型。
模型：chinese-macbert-base（哈工大讯飞联合实验室，中文场景 SOTA）
任务：多标签（10 病性），BCEWithLogitsLoss（transformers 自动处理）

输入：data/splits_v2/{train,val,test}.csv
输出：<output_dir>/test_report.json + best_model/

用法：
  # 单次（历史口径，seed 默认 42，与论文里 baseline 1 的数字同口径）
  python baselines/tcm_sd_text_macbert.py --epochs 3 --batch_size 16

  # 论文方差估计：3 seed 各跑一次，⚠️ 必须输出到不同目录
  python baselines/tcm_sd_text_macbert.py --seed 42 --output_dir runs/tcm_text_macbert_s42
  python baselines/tcm_sd_text_macbert.py --seed 43 --output_dir runs/tcm_text_macbert_s43
  python baselines/tcm_sd_text_macbert.py --seed 44 --output_dir runs/tcm_text_macbert_s44
  # 批量版（推荐）：bash run_baseline_seeds.sh

🔴 切勿把补 seed 的结果写回默认目录 runs/tcm_text_macbert/ —— 那里有
   Stage B 正在依赖的 checkpoint-6630，覆盖会污染实验。脚本已默认拒绝
   写入含 checkpoint-* 的目录（要强行覆盖需显式 --allow_overwrite）。

变更记录：
  2026-09-16  新增 --seed / --allow_overwrite / Handoff 摘要（论文需 baseline 方差）
"""

from __future__ import annotations

# === 国内下载模型：自动设 HF 镜像（与 biomedclip_wrapper 一致）===
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# === 标准库 + 三方库 ===
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

# === 让 baseline 1 能 import eval/ 和 models/（兄弟包）===
#   baselines/tcm_sd_text_macbert.py → 父目录的父目录是 
#   在 sys.path 里加项目根，让 `from eval.metrics import ...` 能工作
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# === 路径常量：基于脚本位置，不依赖工作目录 ===
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# === 运行交接摘要（2026-09-16 新增）===
#   与 training/*.py、eval/*.py 用同一套机制：收尾时打印 + 落盘 handoff_*.txt，
#   自带 argv / seed / 指标 / 主脚本 sha1 指纹 → 只需 cat 一个文件即可。
#   动机：论文里所有 Δ 都要有共同参照，baseline 1 必须补 3 seed 做方差估计。
from utils.run_handoff import Handoff    # noqa: E402

H = Handoff("baseline_text_macbert")

# === 业务模块 ===
try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object   # 占位：本机无 torch 时 class 仍能定义，运行时才报错
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
    )
except ImportError:
    AutoTokenizer = AutoModelForSequenceClassification = TrainingArguments = Trainer = None
try:
    from sklearn.metrics import f1_score
except ImportError:
    f1_score = None


# === CSV 列名（与 tcm_sd_loader 的 LABEL_HEADERS 一致）===
LABEL_COLS = [
    "qi_deficiency|气虚",       # 0
    "blood_deficiency|血虚",     # 1
    "yin_deficiency|阴虚",       # 2
    "yang_deficiency|阳虚",      # 3
    "qi_stagnation|气滞",        # 4
    "blood_stasis|血瘀",         # 5
    "phlegm_dampness|痰湿",       # 6
    "damp_heat|湿热",            # 7
    "wind|内风",                # 8
    "excess_heat|实热",          # 9
]


# ---------- 数据集封装 ----------
class TCMTextDataset(Dataset):
    """
    把 TCM-SD 的 3 列文本拼成单句，tokenize 后产出 (input_ids, attention_mask, labels)。
    """

    def __init__(self, csv_path: Path, tokenizer, max_length: int = 512):
        df = pd.read_csv(csv_path)

        # 拼接问诊文本（与 baseline 0 一致，便于对比）
        self.texts = (
            df["chief_complaint"].fillna("").astype(str)
            + "。"
            + df["description"].fillna("").astype(str)
            + "。"
            + df["detection"].fillna("").astype(str)
        ).tolist()

        # 校验 label 列存在
        missing = [c for c in LABEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺 label 列：{missing}")

        # float 类型：BCEWithLogits 要求 labels 是 float
        self.labels = df[LABEL_COLS].values.astype("float32")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ---------- 多标签评估指标 ----------
def compute_metrics(eval_pred):
    """transformers Trainer 调用的评估函数：logits + labels → 指标字典"""
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))           # sigmoid
    preds = (probs >= 0.5).astype(int)                # 阈值 0.5
    labels_int = labels.astype(int)

    # 注意：多标签 macro_f1 用 average='macro'（每类单独算再平均）
    macro_f1 = float(f1_score(labels_int, preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(labels_int, preds, average="micro", zero_division=0))
    subset_acc = float((preds == labels_int).all(axis=1).mean())

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "subset_accuracy": subset_acc,
    }


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="TCM-SD 文本-only Baseline（MacBERT 多标签微调）")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base",
                       help="HuggingFace 模型 ID（默认 chinese-macbert-base，~100MB）")
    parser.add_argument("--batch_size", type=int, default=16, help="train batch size（4090 24GB 可开 32）")
    parser.add_argument("--epochs", type=int, default=3, help="训练 epoch 数（3 通常足够）")
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 微调学习率（标准 2e-5~5e-5）")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度（中文 BERT 标准 512）")
    parser.add_argument("--data_root", type=str, default=str(_PROJECT_ROOT / "data"),
                       help="数据根目录（含 splits_v2/ 子目录）")
    parser.add_argument("--splits_dir", type=str, default="splits_v2",
                       help="split 子目录名（默认 splits_v2：去重+按 user_id 分组切分的干净版；"
                            "传 splits 用旧版，仅做对比，正式实验一律 splits_v2）")
    parser.add_argument("--output_dir", type=str,
                       default=str(_PROJECT_ROOT / "runs" / "tcm_text_macbert"),
                       help="输出目录（存 best_model + 报告）。⚠️ 默认值里有 Stage B 依赖的 "
                            "checkpoint-6630，补 seed 时务必换成新目录（见 --allow_overwrite）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子（论文方差估计用；默认 42 = transformers 默认值，"
                            "与历史 baseline 1 数字同口径）")
    parser.add_argument("--allow_overwrite", action="store_true",
                       help="允许写入已存在 checkpoint-* 的目录（默认拒绝，防止覆盖 "
                            "Stage B 正在用的文本塔 checkpoint）")
    parser.add_argument("--no_fp16", action="store_true",
                       help="禁用 FP16 混合精度（默认开 FP16，4090 支持）")
    args = parser.parse_args()
    H.set_out_dir(args.output_dir)

    # ---------- 1. tokenizer ----------
    print("=" * 70)
    print(f"  TCM-SD 文本-only Baseline（MacBERT 微调）")
    print("=" * 70)
    print(f"\n[1] 加载 tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"  ✓ tokenizer 加载完成（vocab_size={tokenizer.vocab_size}）")

    # ---------- 2. 数据 ----------
    data_root = Path(args.data_root)
    splits_dir = data_root / args.splits_dir
    print(f"\n[2] 加载 3 个 split（来自 {splits_dir}）...")
    train_ds = TCMTextDataset(splits_dir / "train.csv", tokenizer, args.max_length)
    val_ds   = TCMTextDataset(splits_dir / "val.csv",   tokenizer, args.max_length)
    test_ds  = TCMTextDataset(splits_dir / "test.csv",  tokenizer, args.max_length)
    print(f"  ✓ train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"  ✓ 每样本最多 {args.max_length} token（chief_complaint+description+detection）")

    # ---------- 3. 模型 ----------
    print(f"\n[3] 加载模型: {args.model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=10,                                    # 10 病性
        problem_type="multi_label_classification",        # 多标签：BCEWithLogitsLoss
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 模型加载完成（参数量={n_params/1e6:.1f}M）")

    # ---------- 4. 训练 ----------
    output_dir = Path(args.output_dir)

    # 🔴 覆盖保护（2026-09-16）：runs/tcm_text_macbert/ 里有 Stage B 依赖的 checkpoint-6630。
    #    补 seed 时若误用默认 output_dir，会把正在用的文本塔覆盖掉 → 默认直接拒绝。
    if not args.allow_overwrite:
        existing = sorted(output_dir.glob("checkpoint-*"))
        if existing:
            print(f"\n[FATAL] 输出目录已含 {len(existing)} 个 checkpoint：{output_dir}")
            print("        这很可能是 Stage B 正在依赖的文本塔目录（checkpoint-6630）。")
            print("        覆盖会污染实验 —— 补 seed 请换新目录，例如：")
            print(f"            --output_dir {output_dir}_s{args.seed}")
            print("        确认无碍时再加 --allow_overwrite。")
            H.fail(f"拒绝覆盖已有 checkpoint 的目录：{output_dir}")
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 摘要：配置（贴回时能自证用的什么配置）----
    H.kv("seed", args.seed)
    H.kv("model_name", args.model_name)
    H.kv("splits_dir", args.splits_dir)
    H.kv("epochs/batch/lr", f"{args.epochs} / {args.batch_size} / {args.lr}")
    H.kv("max_length", args.max_length)
    H.kv("output_dir", output_dir)

    print(f"\n[4] 训练中（epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}）...")
    print(f"  预计 ~{len(train_ds) * args.epochs // args.batch_size // 100} 分钟/epoch")
    print(f"  best_model 路径：{output_dir}")

    # === TrainingArguments 兼容处理 ===
    #   transformers 4.46+：eval_strategy + warmup_ratio 都是标准参数
    #   transformers 4.41-4.45：eval_strategy 替代了 evaluation_strategy
    #   transformers < 4.41：只能用 evaluation_strategy
    #   transformers 极老版本（< 4.0）：可能 warmup_ratio 也不存在（罕见）
    #   我们的目标版本：4.46+（兼容最新 PyTorch 2.6 的安全检查）
    try:
        # 新版：eval_strategy
        targs = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",
            fp16=not args.no_fp16,
            dataloader_num_workers=2,
            seed=args.seed,
        )
    except TypeError:
        # 旧版回退（万一 transformers 是 4.41 之前的版本）
        targs = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",
            fp16=not args.no_fp16,
            dataloader_num_workers=2,
            seed=args.seed,
        )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(f"  ✓ 训练完成，best model 已加载")

    # ---------- 5. 测试集最终评估 ----------
    print(f"\n[5] 测试集最终评估...")
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(f"  test 指标: {test_results}")

    # ---------- 6. 多标签完整指标 ----------
    print(f"\n[6] 生成多标签完整指标...")
    preds = trainer.predict(test_ds)
    logits = preds.predictions
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)
    y_true = test_ds.labels.astype(int)

    from eval.metrics import compute_metrics as ml_compute_metrics
    final_metrics = ml_compute_metrics(y_true, y_pred, y_score=probs)

    print(f"\n📊 Final Multi-Label Metrics (test):")
    print(f"  Subset Accuracy: {final_metrics['subset_accuracy']:.4f}")
    print(f"  Macro F1       : {final_metrics['macro_f1']:.4f}")
    print(f"  Micro F1       : {final_metrics['micro_f1']:.4f}")
    print(f"  Hamming Loss   : {final_metrics['hamming_loss']:.4f}")
    print(f"  AUC OvR Macro  : {final_metrics['auc_ovr_macro']:.4f}")
    print(f"\nPer-class F1:")
    for name, f1 in final_metrics.items():
        if name.startswith("f1_"):
            syn_zh = name[3:]
            bar = "█" * int(f1 * 30)
            print(f"  {syn_zh:6s}: {f1:.4f} {bar}")

    # ---------- 7. 保存报告 ----------
    report_path = output_dir / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存：{report_path}")
    print("=" * 70)

    # ---------- 8. 交接摘要（打印 + 落盘）----------
    H.section("核心指标（test）")
    H.kv("macro_f1", f"{final_metrics['macro_f1']:.4f}")
    H.kv("auc_ovr_macro", f"{final_metrics['auc_ovr_macro']:.4f}")
    H.kv("subset_accuracy", f"{final_metrics['subset_accuracy']:.4f}")
    H.kv("hamming_loss", f"{final_metrics['hamming_loss']:.4f}")
    H.section("逐类 F1")
    H.kv_dict({k[3:]: round(float(v), 4)
               for k, v in final_metrics.items() if k.startswith("f1_")})
    H.artifact(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(H.run(main))