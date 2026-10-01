"""
评估指标模块（多标签版本，10 病性）
================================

实现论文里需要的全部评估指标（多标签分类）：
1. Subset Accuracy（多标签版的整体准确率）
2. Macro-F1 / Micro-F1 / Weighted-F1 / Samples-F1（多视角）
3. Hamming Loss（每标签独立错误率，越低越好）
4. AUC（多标签 OvR 策略，macro / micro）
5. Per-class F1（每个证型单独算 F1，论文 Table 用）
6. 混淆矩阵（10×10）+ 可视化（论文 Figure 用）

输入约定（多标签）：
- y_true: [N, 10] 0/1 向量（10 病性多标签）
- y_pred: [N, 10] 0/1 向量（top-k 阈值化后的硬预测）
- y_score: [N, 10] 连续分数（logits / 相似度 / 概率），可选
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

try:
    import numpy as np
except ImportError:  # 本机无 numpy 也能 import（仅做结构检查）
    np = None


# 10 个病性要素（必须与 training/config.py + data/tcm_sd_loader.py 的列顺序一致）
# 顺序对齐 models/soft_cl_loss.py 的 SYNDROME_ORDER：
#   气虚 血虚 阴虚 阳虚 气滞 血瘀 痰湿 湿热 内风 实热
SYNDROME_NAMES_ZH = ["气虚", "血虚", "阴虚", "阳虚", "气滞", "血瘀", "痰湿", "湿热", "内风", "实热"]
SYNDROME_NAMES_EN = ["qi_def", "blood_def", "yin_def", "yang_def", "qi_stag",
                     "blood_stasis", "phlegm", "damp_heat", "internal_wind", "excess_heat"]


def compute_metrics(
    y_true: np.ndarray,        # [N, 10] 多标签真实标签
    y_pred: np.ndarray,        # [N, 10] 多标签预测标签（top-k 阈值化）
    y_score: Optional[np.ndarray] = None,  # [N, 10] 连续预测分数（用于 AUC）
) -> Dict[str, float]:
    """
    计算所有评估指标（多标签版本）。

    Args:
        y_true: [N, 10] 0/1 真实标签
        y_pred: [N, 10] 0/1 预测标签
        y_score: [N, 10] 连续分数（logits / 相似度），用于算 AUC

    Returns:
        dict 包含所有指标的字典
    """
    from sklearn.metrics import (
        f1_score, roc_auc_score, hamming_loss,
        accuracy_score
    )

    # 确保是 numpy 数组
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_score is not None:
        y_score = np.asarray(y_score)

    metrics = {}

    # 1. Subset Accuracy（多标签版的"完全匹配"准确率）
    metrics["subset_accuracy"] = float(accuracy_score(y_true, y_pred))

    # 2. Macro / Micro / Weighted / Samples F1（多视角 F1）
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["samples_f1"] = float(f1_score(y_true, y_pred, average="samples", zero_division=0))

    # 3. Hamming Loss（每标签独立算错误率，越低越好）
    metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))

    # 4. AUC（多标签 OvR 策略）
    if y_score is not None:
        try:
            metrics["auc_ovr_macro"] = float(roc_auc_score(
                y_true, y_score,
                multi_class="ovr", average="macro"
            ))
        except ValueError:
            # 某个病性完全没样本，roc_auc 会报错
            metrics["auc_ovr_macro"] = 0.0
        try:
            metrics["auc_ovr_micro"] = float(roc_auc_score(
                y_true, y_score,
                multi_class="ovr", average="micro"
            ))
        except ValueError:
            metrics["auc_ovr_micro"] = 0.0

    # 5. Per-class F1（每个证型单独算 F1，论文 Table 用）
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, syn_zh in enumerate(SYNDROME_NAMES_ZH):
        metrics[f"f1_{syn_zh}"] = float(per_class_f1[i])

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    normalize: bool = True,
    title: str = "Confusion Matrix (10 Syndromes, Multi-label)",
):
    """
    画混淆矩阵图（论文 Figure 用）。

    多标签场景：按"每样本取第一个阳性标签"压缩成单标签再画 10×10 混淆矩阵。
    仅作为可视化辅助，主要看 Per-class F1 表更准确。
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("⚠️  matplotlib/seaborn 未安装，跳过可视化")
        return

    # 多标签 → 单标签（每行取第一个阳性标签，无阳性的样本跳过）
    y_true_single = []
    y_pred_single = []
    for t, p in zip(y_true, y_pred):
        t_idx = np.where(t > 0)[0]
        p_idx = np.where(p > 0)[0]
        if len(t_idx) > 0 and len(p_idx) > 0:
            y_true_single.append(int(t_idx[0]))
            y_pred_single.append(int(p_idx[0]))

    if len(y_true_single) == 0:
        print("⚠️  无可用样本画混淆矩阵")
        return

    y_true_arr = np.array(y_true_single)
    y_pred_arr = np.array(y_pred_single)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(SYNDROME_NAMES_ZH))))
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else "d",
        xticklabels=SYNDROME_NAMES_ZH,
        yticklabels=SYNDROME_NAMES_ZH,
        cmap="Blues", cbar=True,
    )
    plt.xlabel("Predicted (Top-1)")
    plt.ylabel("True (Top-1)")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 混淆矩阵已保存到：{output_path}")


def format_metrics_table(metrics_dict: Dict[str, float]) -> str:
    """把指标字典格式化成 markdown 表格字符串（论文用）。"""
    lines = ["| Metric | Value |", "|---|---|"]
    for key, value in metrics_dict.items():
        lines.append(f"| {key} | {value:.4f} |")
    return "\n".join(lines)


# ---------- 单元测试 ----------

if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        print("⚠️  numpy 未安装，跳过单元测试")
    else:
        # 模拟多标签数据：N=200 样本，10 病性，平均每样本 1-3 个阳性
        np.random.seed(42)
        N = 200
        n_classes = 10

        y_true = np.zeros((N, n_classes), dtype=int)
        for i in range(N):
            n_positive = np.random.randint(1, 4)
            pos_indices = np.random.choice(n_classes, n_positive, replace=False)
            y_true[i, pos_indices] = 1

        # 预测分数：真实标签 + 噪声
        y_score = y_true.astype(float) + 0.3 * np.random.rand(N, n_classes)

        # top-2 阈值化预测
        k = 2
        y_pred = np.zeros_like(y_true)
        top_k_indices = np.argsort(-y_score, axis=1)[:, :k]
        for i, idx in enumerate(top_k_indices):
            y_pred[i, idx] = 1

        metrics = compute_metrics(y_true, y_pred, y_score)
        print("📊 评估指标（多标签模拟数据）：")
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