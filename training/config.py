"""
训练超参数配置
==============

所有超参数集中管理，避免在代码里硬编码。
修改此文件即可改变实验设置，无需动训练代码。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """数据相关配置。"""
    data_root: str = "papers/code/data"
    tcm_sd_path: str = "papers/code/data/tcm-sd"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    image_size: int = 224
    text_max_length: int = 256


@dataclass
class ModelConfig:
    """模型架构配置。"""
    biomedclip_model: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    projection_dim: int = 512
    kg_embedding_dim: int = 64
    freeze_backbone: bool = True
    lora_rank: int = 8                  # 仅 baseline 2 用
    cross_attention_heads: int = 8
    cross_attention_layers: int = 2


@dataclass
class TrainingConfig:
    """训练超参数。"""
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 20
    warmup_epochs: int = 2
    gradient_clipping: float = 1.0

    # 多任务损失权重
    lambda_ce: float = 1.0       # 分类交叉熵权重
    lambda_softcl: float = 0.5   # Soft Contrastive Loss 权重

    # SoftCL 温度参数
    softcl_temperature: float = 0.07
    learnable_temperature: bool = False

    # 早停
    early_stop_patience: int = 5

    # 日志
    log_interval: int = 10
    eval_interval: int = 1     # 每 N 个 epoch 评估一次
    save_top_k: int = 3         # 保存最好的 K 个 checkpoint


@dataclass
class ExperimentConfig:
    """实验元配置（包含上述全部）。"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # 实验标识
    experiment_name: str = "tcm_mkg_baseline"
    notes: str = ""

    # 输出目录
    output_dir: str = "papers/code/runs"


# ---------- 不同 baseline 的预设配置 ----------

BASELINE_CONFIGS = {
    "biomedclip_zero_shot": ExperimentConfig(
        experiment_name="baseline_1_zero_shot",
        notes="BiomedCLIP zero-shot（无任何训练）",
    ),
    "biomedclip_lora": ExperimentConfig(
        experiment_name="baseline_2_lora",
        notes="BiomedCLIP + LoRA 微调",
    ),
    "ydyolo": ExperimentConfig(
        experiment_name="baseline_3_ydyolo",
        notes="YD-YOLO 类单模态 baseline",
    ),
    "tcm_sd_text": ExperimentConfig(
        experiment_name="baseline_4_text_only",
        notes="TCM-SD 文本-only baseline",
    ),
    "tcm_mkg_full": ExperimentConfig(
        experiment_name="full_model",
        notes="完整模型：BiomedCLIP + Cross-Attn + KG + SoftCL",
    ),
}


# ---------- 单元测试 ----------

if __name__ == "__main__":
    config = ExperimentConfig()
    print(f"Experiment: {config.experiment_name}")
    print(f"Notes: {config.notes}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Lambda SoftCL: {config.training.lambda_softcl}")

    print("\n所有 baseline 配置：")
    for name, cfg in BASELINE_CONFIGS.items():
        print(f"  {name}: {cfg.notes}")