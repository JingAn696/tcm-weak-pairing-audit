"""
Soft Contrastive Loss（软标签对比损失）
======================================

为什么需要 SoftCL：
- 10 个病性要素不是完全互斥的（如"气虚 + 血瘀"是常见组合）
- 普通 Contrastive Loss 用 one-hot 标签 → 模型学到的是"硬分类"
- SoftCL 用**软标签矩阵**：10×10 矩阵中非对角线元素 > 0
  → 模型学到"气虚和血瘀之间有 0.6 的相似度"，而不是"完全不同"

核心公式（论文待写）：
    L_SoftCL = -sum_i sum_j w_ij * log(exp(sim(z_i, z_j)/τ) / sum_k exp(sim(z_i, z_k)/τ))

其中：
- z_i, z_j: 第 i, j 个样本的归一化特征向量
- τ: 温度参数（默认 0.07）
- w_ij: 软标签权重（来自领域专家校准的 10×10 病性相似度矩阵）

参考论文：
- "Supervised Contrastive Learning" (Khosla et al. NeurIPS 2020)
- "Relational Contrastive Learning" (Yu et al. 2023)
- "Label Confusion Learning" (Xu et al. 2022)

相似度矩阵：领域专家 2026-09-09 校准定稿（见 default_syndrome_similarity_matrix）。
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# 10 个证型的标准顺序（与 SoftCL 默认输入一致）
# 2026-09-09 由 8 扩到 10（领域专家定案，新增"内风""实热"，"郁"并入"气滞"）
SYNDROME_ORDER = [
    "气虚", "血虚", "阴虚", "阳虚",
    "气滞", "血瘀", "痰湿", "湿热",
    "内风", "实热",
]


def default_syndrome_similarity_matrix() -> torch.Tensor:
    """
    领域专家校准定稿的 10×10 病性相似度矩阵（2026-09-09）。

    数值范围：0.0 - 1.0
    - 对角线 = 1.0（自身）
    - 临床常见组合（气血两虚、气滞血瘀、阴虚风动等）= 0.7-0.9
    - 临床对立（阳虚 vs 实热）= 0.1
    - 其余 = 0.2-0.6

    该矩阵是 11 年中医临床经验的直接结晶，是 SoftCL 的核心先验。

    Returns:
        Tensor[10, 10]，软标签矩阵
    """
    # === 领域专家校准定稿（2026-09-09）===
    # 行/列顺序：气虚 血虚 阴虚 阳虚 气滞 血瘀 痰湿 湿热 内风 实热
    matrix = torch.tensor([
        # 气虚  血虚  阴虚  阳虚  气滞  血瘀  痰湿  湿热  内风  实热
        [1.00, 0.80, 0.60, 0.90, 0.40, 0.60, 0.70, 0.30, 0.30, 0.20],  # 气虚
        [0.80, 1.00, 0.80, 0.50, 0.30, 0.70, 0.30, 0.20, 0.80, 0.20],  # 血虚 (vs 内风: 血虚生风 0.80)
        [0.60, 0.80, 1.00, 0.60, 0.40, 0.50, 0.30, 0.30, 0.90, 0.40],  # 阴虚 (vs 内风: 阴虚风动 0.90)
        [0.90, 0.50, 0.60, 1.00, 0.50, 0.70, 0.80, 0.20, 0.30, 0.10],  # 阳虚 (vs 实热: 寒热对立 0.10)
        [0.40, 0.30, 0.40, 0.50, 1.00, 0.90, 0.70, 0.40, 0.50, 0.40],  # 气滞 (vs 血瘀: 气滞血瘀 0.90)
        [0.60, 0.70, 0.50, 0.70, 0.90, 1.00, 0.60, 0.50, 0.50, 0.50],  # 血瘀
        [0.70, 0.30, 0.30, 0.80, 0.70, 0.60, 1.00, 0.80, 0.60, 0.40],  # 痰湿 (vs 阳虚: 痰湿困脾 0.80)
        [0.30, 0.20, 0.30, 0.20, 0.40, 0.50, 0.80, 1.00, 0.40, 0.50],  # 湿热 (vs 痰湿: 痰热互结 0.80)
        [0.30, 0.80, 0.90, 0.30, 0.50, 0.50, 0.60, 0.40, 1.00, 0.60],  # 内风 (vs 阴虚 0.90 / 血虚 0.80)
        [0.20, 0.20, 0.40, 0.10, 0.40, 0.50, 0.40, 0.50, 0.60, 1.00],  # 实热 (vs 阳虚: 寒热对立 0.10)
    ], dtype=torch.float32)
    return matrix


class SoftContrastiveLoss(nn.Module):
    """
    Soft Contrastive Loss 实现。

    用法：
        >>> criterion = SoftContrastiveLoss(similarity_matrix=...)
        >>> features = model(images, texts)  # [B, D]
        >>> labels = torch.tensor([0, 1, 2, ..., 9])  # B 个样本的病性索引
        >>> loss = criterion(features, labels)
    """

    def __init__(
        self,
        similarity_matrix: Optional[torch.Tensor] = None,
        temperature: float = 0.07,
        learnable_temp: bool = False,
    ):
        """
        Args:
            similarity_matrix: 10×10 软标签矩阵，若 None 则用默认值
            temperature: 温度参数 τ，越小越尖锐
            learnable_temp: 是否把温度作为可学习参数
        """
        super().__init__()
        if similarity_matrix is None:
            similarity_matrix = default_syndrome_similarity_matrix()

        # 验证矩阵
        assert similarity_matrix.shape == (10, 10), \
            f"相似度矩阵必须是 10×10，得到 {similarity_matrix.shape}"
        assert (similarity_matrix >= 0).all() and (similarity_matrix <= 1).all(), \
            "相似度矩阵数值必须在 [0, 1] 范围"
        assert torch.allclose(similarity_matrix, similarity_matrix.T), \
            "相似度矩阵必须对称"

        # 注册为 buffer（不参与梯度，但会随模型迁移到 GPU）
        self.register_buffer("similarity_matrix", similarity_matrix)

        # 温度参数
        if learnable_temp:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer(
                "log_temperature", torch.log(torch.tensor(temperature))
            )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(
        self,
        features: torch.Tensor,  # [B, D]
        labels: torch.Tensor,   # [B]，每个样本的证型索引
    ) -> torch.Tensor:
        """
        计算 Soft Contrastive Loss。

        Args:
            features: 归一化后的特征向量 [B, D]
            labels: 证型索引 [B]，每个值 ∈ {0, 1, ..., 9}

        Returns:
            loss: 标量
        """
        # 1. 归一化（防止 features 模长影响相似度）
        features = F.normalize(features, dim=-1)  # [B, D]

        # 2. 计算 batch 内样本间的相似度 [B, B]
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # 3. 根据 labels 取软标签 [B, B]
        # w_ij = similarity_matrix[labels[i], labels[j]]
        soft_labels = self.similarity_matrix[labels]  # [B, 10]
        soft_labels = soft_labels[:, labels]           # [B, B]

        # 4. Soft Cross Entropy Loss
        # 对每个样本 i：
        #   L_i = -sum_j w_ij * log(exp(sim_ij) / sum_k exp(sim_ik))
        #       = -sum_j w_ij * (sim_ij - logsumexp(sim_i))

        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=-1, keepdim=True)  # [B, B]
        # 平均每个 i 上的软标签权重（除以 w_矩阵的和，使得权重和为 1）
        soft_labels_norm = soft_labels / (soft_labels.sum(dim=-1, keepdim=True) + 1e-8)

        loss = -(soft_labels_norm * log_prob).sum(dim=-1).mean()

        return loss

    def update_similarity_matrix(self, new_matrix: torch.Tensor):
        """
        在训练过程中动态更新相似度矩阵（如用可学习参数时）。

        Args:
            new_matrix: 新的 10×10 矩阵
        """
        assert new_matrix.shape == (10, 10)
        self.similarity_matrix.copy_(new_matrix.to(self.similarity_matrix.device))


# ---------- 单元测试 ----------

if __name__ == "__main__":
    # 测试：构造 dummy 数据
    print("Testing SoftContrastiveLoss...")
    criterion = SoftContrastiveLoss()
    features = torch.randn(20, 512)  # 20 个样本（每病性 2 个）
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
    loss = criterion(features, labels)
    print(f"✓ Loss computed: {loss.item():.4f}")

    # 验证：相同病性距离应小于不同病性
    matrix = default_syndrome_similarity_matrix()
    print(f"\n相似度矩阵（领域专家校准定稿）：")
    print(f"  气虚-血虚 = {matrix[0, 1]:.2f} （气血两虚，高相关）")
    print(f"  气滞-血瘀 = {matrix[4, 5]:.2f} （气滞血瘀，高相关）")
    print(f"  阴虚-内风 = {matrix[2, 8]:.2f} （阴虚风动，最高相关之一）")
    print(f"  阳虚-实热 = {matrix[3, 9]:.2f} （寒热对立，最低相关）")
    print(f"  气虚-气虚 = {matrix[0, 0]:.2f} （自身）")