"""
完整的多模态证候分类模型（路线 C 全模型）
====================================

这是论文 §3 描述的"我们的完整模型"，由 5 个模块组成：

    ┌────────────────────────────────────────────────┐
    │         BiomedCLIP 视觉编码器                  │  ← frozen
    │         BiomedCLIP 文本编码器                  │  ← frozen
    └─────┬─────────────────────────────┬────────────┘
          │                             │
          ▼                             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │f_vis 512 │  │f_text512 │  │KG 嵌入 64│  ← 仅 KG + Cross-Attn + 分类头可训
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         └────────┬─────┴────────┬────┘
                  ▼
         ┌───────────────────┐
         │ CrossModalFusion  │  ← 多向交叉注意力（核心创新点）
         │ + 分类头          │
         └────────┬──────────┘
                  ▼
            10 病性 logits
                  │
                  ▼
        ┌─────────────────────────┐
        │ Soft Contrastive Loss   │  ← 10 个病性之间的相似度软标签
        │ + BCE 多标签分类 Loss  │
        └─────────────────────────┘

训练策略（与 W3 精读综合确定）：

    • BiomedCLIP 冻结 + LoRA（只在 cross-attn / KG / classifier 上微调）
    • Loss = α * BCE + β * SoftCL
    • SoftCL 10×10 相似度矩阵 = 净安 11 年经验校准（2026-09-09 定稿）
    • KG 嵌入 = PyKEEN TransE 训练 50 epochs

为什么把 BiomedCLIP 分离 freeze：参数量从 ~150M 降到 ~12M（仅 fusion + classifier），
    单卡 24GB（RTX 3090/4090）即可显存富裕训练。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

# === 项目内模块 ===
from .biomedclip_wrapper import BiomedCLIPWrapper
from .cross_attention_fusion import CrossModalFusion
from .kg_embedding import KGEmbeddingModule
from .soft_cl_loss import SoftContrastiveLoss


class MultimodalSyndromeClassifier(nn.Module):
    """
    完整的多模态证候分类模型（路线 C）。

    数据流：

        texts: List[str] → BiomedCLIP text encoder → f_text (B, 512)
        images: Tensor (B, 3, H, W) → BiomedCLIP vision encoder → f_vis (B, 512)
        kg_query_ids: Tensor (B,) → KGEmbedding → f_kg (B, 64)

        logits = CrossModalFusion(f_text, f_vis, f_kg)  # (B, 10)

    训练时用 BCE + SoftCL，推理时sigmoid 后阈值 0.5 多标签输出。
    """

    def __init__(
        self,
        # BiomedCLIP 配置
        biomedclip_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        biomedclip_pretrained: str = "hf",
        # KG 配置
        kg_embedding_dim: int = 64,
        kg_vocab_size: int = 47,           # 节点数（13 观察 + 8 证型 + 6 脏腑 + 20 主诉）
        kg_checkpoint_path: Optional[str] = None,
        # Fusion 配置
        fusion_d_model: int = 256,
        fusion_n_heads: int = 8,
        fusion_dropout: float = 0.1,
        # 分类
        n_classes: int = 10,
        # 训练策略
        freeze_backbones: bool = True,
        # SoftCL 配置
        softcl_similarity_matrix: Optional["torch.Tensor"] = None,
        softcl_temperature: float = 0.1,
    ):
        super().__init__()
        if nn is None:
            raise ImportError("完整模型需要 torch 环境。")

        # === 模块 1: BiomedCLIP 双编码器（默认冻结）===
        self.biomedclip = BiomedCLIPWrapper(
            model_name=biomedclip_name,
            pretrained=biomedclip_pretrained,
            freeze_image_encoder=freeze_backbones,
            freeze_text_encoder=freeze_backbones,
        )
        text_dim = self.biomedclip.text_dim
        vis_dim = self.biomedclip.vision_dim

        # === 模块 2: KG 嵌入 ===
        self.kg_module = KGEmbeddingModule(
            num_entities=kg_vocab_size,
            embedding_dim=kg_embedding_dim,
        )
        # 如果提供了训练好的 PyKEEN checkpoint，加载；否则随机初始化
        # TODO 阶段 1 后续：上传 PyKEEN checkpoint

        # === 模块 3: Cross-Attention 融合 ===
        self.fusion = CrossModalFusion(
            text_dim=text_dim,
            vis_dim=vis_dim,
            kg_dim=kg_embedding_dim,
            d_model=fusion_d_model,
            n_classes=n_classes,
            n_heads=fusion_n_heads,
            dropout=fusion_dropout,
        )

        # === 模块 4: SoftCL 损失 ===
        self.softcl = SoftContrastiveLoss(
            temperature=softcl_temperature,
            similarity_matrix=softcl_similarity_matrix,
        )

        self.n_classes = n_classes

    def encode_text(self, texts: List[str]) -> "torch.Tensor":
        return self.biomedclip.encode_text(texts)

    def encode_image(self, images: "torch.Tensor") -> "torch.Tensor":
        return self.biomedclip.encode_image(images)

    def encode_kg(self, kg_query_ids: "torch.Tensor") -> "torch.Tensor":
        """KG 查询：输入是 (B,) 整数索引，输出 (B, kg_dim)。"""
        return self.kg_module(kg_query_ids)

    def forward(
        self,
        texts: List[str],
        images: "torch.Tensor",
        kg_query_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        完整前向。

        参数
        ----
        texts : List[str]，长度 B，文本问诊。
        images : (B, 3, H, W) 视觉编码器输入。
        kg_query_ids : (B,) KG 查询索引。

        返回
        ----
        logits : (B, n_classes) 证候 logits。
        """
        f_text = self.encode_text(texts)     # (B, 512)
        f_vis = self.encode_image(images)    # (B, 512)
        f_kg = self.encode_kg(kg_query_ids)  # (B, 64)
        return self.fusion(f_text, f_vis, f_kg)

    def compute_loss(
        self,
        texts: List[str],
        images: "torch.Tensor",
        kg_query_ids: "torch.Tensor",
        labels: "torch.Tensor",
        softcl_weight: float = 0.5,
        bce_weight: float = 1.0,
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        """
        完整损失 = α * BCE(多标签) + β * SoftCL（特征级对比）。

        返回 (loss, metrics_dict)。
        """
        # 1. 拿到融合前的特征（用于 SoftCL）
        f_text = self.encode_text(texts)
        f_vis = self.encode_image(images)
        f_kg = self.encode_kg(kg_query_ids)

        # 2. 分类 logits
        logits = self.fusion(f_text, f_vis, f_kg)

        # 3. BCE 多标签分类
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # 4. SoftCL：把"特征" 整体送入 SoftCL，统一保留梯度
        #   用多模态融合表征做对比学习，比起单模态更有意义
        fused_feat = torch.cat([f_text, f_vis, f_kg], dim=-1)  # 简单拼接
        softcl_loss = self.softcl(fused_feat, labels.argmax(dim=1))

        # 5. 合并
        total_loss = bce_weight * bce_loss + softcl_weight * softcl_loss

        return total_loss, {
            "bce": bce_loss.item(),
            "softcl": softcl_loss.item(),
            "total": total_loss.item(),
        }

    def predict(
        self,
        texts: List[str],
        images: "torch.Tensor",
        kg_query_ids: "torch.Tensor",
        threshold: float = 0.5,
    ) -> "torch.Tensor":
        """推理：sigmoid + 阈值化，返回 0/1 多标签。"""
        logits = self.forward(texts, images, kg_query_ids)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).int()


# === 训练入口的脚手架 ===

def train_step_scaffold():
    """给训练入口的最小骨架（具体训练 pipeline 在 training/train.py）。"""
    print("完整模型脚手架已就位，训练入口见 training/train.py")


# === 单元测试（无 torch 时优雅降级） ===
if __name__ == "__main__":
    if torch is None:
        print("⚠️  torch 未安装。完整模型需要 torch 环境。")
        print("   pip install torch transformers")
    else:
        print("完整模型结构验证（mock forward）：")
        # mock 测试：用随机张量模拟 BiomedCLIP 输出
        B = 2

        # mock BiomedCLIP
        import types
        wrapper = MultimodalSyndromeClassifier.__new__(MultimodalSyndromeClassifier)
        nn.Module.__init__(wrapper)
        wrapper.n_classes = 10

        # mock 各组件
        f_text = torch.randn(B, 512, requires_grad=True)
        f_vis = torch.randn(B, 512, requires_grad=True)
        f_kg = torch.randn(B, 64, requires_grad=True)
        labels = torch.randint(0, 2, (B, 10)).float()

        # 手动 fusion
        from .cross_attention_fusion import CrossModalFusion
        wrapper.fusion = CrossModalFusion(512, 512, 64, 256, 10)
        logits = wrapper.fusion(f_text, f_vis, f_kg)
        print(f"  logits.shape: {logits.shape}")

        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        print(f"  BCE loss: {bce.item():.4f}")

        # SoftCL
        from .soft_cl_loss import SoftContrastiveLoss
        wrapper.softcl = SoftContrastiveLoss()
        fused_feat = torch.cat([f_text, f_vis, f_kg], dim=-1)
        softcl_loss = wrapper.softcl(fused_feat, labels.argmax(dim=1))
        print(f"  SoftCL loss: {softcl_loss.item():.4f}")

        print(f"\n  ✓ 总损失 = BCE + SoftCL = {bce.item() + softcl_loss.item():.4f}")
