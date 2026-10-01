"""
完整模型 v2（路线 C · Stage A：文本 + KG + SoftCL）
=================================================

与 v1（full_model.py，BiomedCLIP 双塔）的区别——2026-09-11 架构调整：

    v1 问题：BiomedCLIP 文本塔是 PubMedBERT（英文）。用它编码中文问诊文本
    得到的是乱码特征——zero-shot baseline AUC 0.49（≈随机）已经证明了这一点。
    而 MacBERT 中文微调拿到 Macro F1 87%（baseline 1）。

    v2 方案（领域专家 2026-09-11 确认的路线 A）：
        文本分支 = MacBERT 热启动（从 baseline 1 checkpoint 加载，768 维）
        视觉分支 = BiomedCLIP 视觉塔（Stage B 接入舌象原型，512 维）
        KG 分支   = 50 节点嵌入（每样本 multi-hot 加权平均，64 维）
        融合      = 三向 Cross-Attention（复用 CrossModalFusion）
        损失      = BCE + β * SoftCL（领域专家校准的 10×10 相似度矩阵）

    Stage A（visual_mode="placeholder"）：TCM-SD 没有逐样本舌象图片，
    视觉输入用可学习的"缺失舌象"共享向量占位——保持三路融合结构完整。

    Stage B · B-2（visual_mode="proto_attn"，2026-09-14 接入）：
    视觉分支换成"文本查询 10 个冻结舌象原型"的注意力
    （TCM-Tongue 经 BiomedCLIP 视觉塔 + 21 类头预编码，按 21×10 映射
    矩阵聚合成 10 个病性原型，centered 策略，见 syndrome_proto_attention.py）。
    气滞/内风因无可用支撑图被 mask。

数据流（Stage A）：
    text ──MacBERT──► f_text (B, 768) ─┐
    multi-hot ──KG嵌入平均──► f_kg (B, 64) ─┼─► CrossModalFusion ─► 10 病性 logits
    [可学习占位] ──► f_vis (B, 512) ─┘              │
                                                BCE + SoftCL(fused)

KG multi-hot 的意义（论文 Method 素材）：
    20 主诉 + 14 舌象观察节点的关键词命中，把文本中隐式的舌象证据
    （detection 字段的"舌淡""齿痕"等）显式路由到 KG 通道。
    血虚是文本分支的软肋（F1 66%），而舌象恰是血虚最强证据——
    Stage B 真实舌象接入后，血虚改善是核心实验假设。
"""

from __future__ import annotations

import os

# 国内下载 HuggingFace 模型必需（setdefault：用户已 export 时不覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # 本机无 torch 也能 import（仅做结构检查）
    torch = None
    nn = None
    F = None

from .cross_attention_fusion import CrossModalFusion
from .kg_embedding import KGEmbeddingModule
from .kg_keywords import ALL_NODES
from .soft_cl_loss import SoftContrastiveLoss
from .syndrome_proto_attention import (
    B2_VERSION,
    DEFAULT_MIN_SUPPORT,
    SyndromeProtoAttention,
)
from .tongue_label_mapping import SYNDROMES_ZH

# Stage A 视觉占位维度 = BiomedCLIP 视觉塔输出维（Stage B 原型同维直接替换）
VIS_PLACEHOLDER_DIM = 512

# 视觉分支两种模式：
#   placeholder → Stage A：可学习的"缺失舌象"共享向量（TCM-SD 无图）
#   proto_attn  → Stage B：文本查询 10 个冻结舌象原型（弱配对桥接，见 syndrome_proto_attention.py）
VISUAL_MODES = ("placeholder", "proto_attn")


class MultimodalSyndromeClassifierV2(nn.Module):
    """
    完整模型 v2（Stage A）。

    参数
    ----
    text_encoder : transformers 的 BertModel 实例（MacBERT，热启动）
    n_classes : 病性数（10）
    kg_embedding_dim : KG 嵌入维（64）
    softcl_temperature / softcl_weight : SoftCL 超参
    visual_mode : "placeholder"（Stage A）或 "proto_attn"（Stage B · B-2）
    proto_path  : proto_attn 模式下 B-1 产出的 tongue_prototypes.pt 路径
    proto_min_support : 支撑图数低于此值的病性原型会被 mask（气滞=5、内风=0）
    proto_include_sim : 视觉表征是否拼接"逐病性相似度向量"（512 → 522 维）
    """

    def __init__(
        self,
        text_encoder,
        n_classes: int = 10,
        kg_embedding_dim: int = 64,
        fusion_d_model: int = 256,
        fusion_n_heads: int = 8,
        fusion_dropout: float = 0.1,
        softcl_temperature: float = 0.07,
        softcl_weight: float = 0.5,
        visual_mode: str = "placeholder",
        proto_path: Optional[str] = None,
        proto_min_support: float = DEFAULT_MIN_SUPPORT,
        proto_include_sim: bool = True,
    ):
        super().__init__()
        if nn is None:
            raise ImportError("完整模型 v2 需要 torch 环境。")
        if visual_mode not in VISUAL_MODES:
            raise ValueError(f"visual_mode 应为 {VISUAL_MODES}，收到 {visual_mode!r}")

        self.text_encoder = text_encoder
        self.text_dim = text_encoder.config.hidden_size  # MacBERT-base = 768
        self.n_classes = n_classes
        self.softcl_weight = softcl_weight
        self.visual_mode = visual_mode

        # === 模块 1: KG 嵌入（50 节点，随机初始化端到端训练；PyKEEN 版本留消融） ===
        self.kg = KGEmbeddingModule(
            num_entities=len(ALL_NODES),  # 50
            embedding_dim=kg_embedding_dim,
        )

        # === 模块 2: 视觉分支 ===
        # Stage A：缺失舌象的可学习共享向量（始终创建，保证老 checkpoint 仍可加载）
        self.missing_vis = nn.Parameter(torch.randn(VIS_PLACEHOLDER_DIM) * 0.02)

        # Stage B · B-2：文本查询 10 个冻结舌象原型的注意力
        self.proto_summary: Dict[str, object] = {}
        if visual_mode == "proto_attn":
            if not proto_path:
                raise ValueError("visual_mode='proto_attn' 时必须提供 proto_path")
            self.proto_attn = SyndromeProtoAttention(
                text_dim=self.text_dim,
                proto_dim=VIS_PLACEHOLDER_DIM,
                n_syndromes=n_classes,
                include_sim=proto_include_sim,
                dropout=fusion_dropout,
            )
            self.proto_summary = self.proto_attn.load_prototypes_from_file(
                proto_path, min_support=proto_min_support, verbose=True
            )
            vis_dim = self.proto_attn.out_dim
        else:
            self.proto_attn = None
            vis_dim = VIS_PLACEHOLDER_DIM

        # === 模块 3: 三向 Cross-Attention 融合（复用路线 C 核心） ===
        self.fusion = CrossModalFusion(
            text_dim=self.text_dim,       # 768
            vis_dim=vis_dim,              # 512（占位）/ 522（原型注意力）
            kg_dim=kg_embedding_dim,      # 64
            d_model=fusion_d_model,
            n_classes=n_classes,
            n_heads=fusion_n_heads,
            dropout=fusion_dropout,
        )

        # === 模块 4: SoftCL（领域专家校准的 10×10 矩阵，默认参数） ===
        self.softcl = SoftContrastiveLoss(temperature=softcl_temperature)

    # ---------- 各分支编码 ----------
    def encode_text(self, input_ids, attention_mask) -> "torch.Tensor":
        """MacBERT CLS 表征 (B, 768)。"""
        out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return out.last_hidden_state[:, 0]

    def encode_kg(self, node_multi_hot: "torch.Tensor") -> "torch.Tensor":
        """
        KG 分支：multi-hot (B, 50) → 命中节点嵌入的平均 (B, 64)。
        无命中样本（multi-hot 全 0）→ 零向量（clamp 保证不除 0）。
        """
        emb = self.kg.embeddings.weight                       # (50, kg_dim)
        counts = node_multi_hot.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return (node_multi_hot @ emb) / counts

    def encode_vis_placeholder(self, batch_size: int) -> "torch.Tensor":
        """Stage A 视觉占位：(512,) 广播到 (B, 512)。"""
        return self.missing_vis.unsqueeze(0).expand(batch_size, -1)

    def encode_vis(self, f_text: "torch.Tensor"):
        """
        Stage B · B-2：文本查询舌象原型 → (f_vis, attn, sim)。
        Stage A：占位向量，attn/sim 为 None。
        """
        if self.visual_mode == "proto_attn":
            return self.proto_attn(f_text)
        return self.encode_vis_placeholder(f_text.size(0)), None, None

    # ---------- 前向 ----------
    def forward(
        self,
        input_ids,
        attention_mask,
        node_multi_hot: "torch.Tensor",
        return_features: bool = False,
        return_attn: bool = False,
    ):
        """
        input_ids / attention_mask : MacBERT tokenizer 输出
        node_multi_hot : (B, 50) float，每样本 KG 节点命中
        return_features : True 时额外返回融合表征（SoftCL 用）
        return_attn : True 时额外返回 (attn, sim)（B-2 的论文可视化 / 诊断用）

        返回
        ----
        logits (B, 10)
          | (logits, fused)                      —— return_features=True
          | (logits, fused, attn, sim)           —— return_features 且 return_attn
          | (logits, attn, sim)                  —— 仅 return_attn
        """
        f_text = self.encode_text(input_ids, attention_mask)   # (B, 768)
        f_vis, attn, sim = self.encode_vis(f_text)             # (B, 512/522)
        f_kg = self.encode_kg(node_multi_hot)                  # (B, 64)
        out = self.fusion(f_text, f_vis, f_kg, return_features=return_features)
        if return_attn:
            if return_features:
                return out[0], out[1], attn, sim
            return out, attn, sim
        return out

    # ---------- 损失 ----------
    def compute_loss(
        self,
        logits: "torch.Tensor",
        fused: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        """
        总损失 = BCE(多标签) + softcl_weight * SoftCL(fused, 主病性)。

        SoftCL 标签说明：多标签样本取 argmax 作"主病性"索引。
        这是 SoftCL 在多标签场景的已知简化（与 v1 相同），
        论文 Method 需注明。
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        primary = labels.argmax(dim=1)
        softcl_loss = self.softcl(fused, primary)
        total = bce_loss + self.softcl_weight * softcl_loss
        return total, {
            "bce": bce_loss.item(),
            "softcl": softcl_loss.item(),
            "total": total.item(),
        }

    # ---------- 参数统计（论文用） ----------
    def param_stats(self) -> Dict[str, int]:
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        kg_params = sum(p.numel() for p in self.kg.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        vis_params = self.missing_vis.numel()
        proto_params = 0
        if self.proto_attn is not None:
            proto_params = sum(p.numel() for p in self.proto_attn.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            "text_encoder": text_params,
            "kg": kg_params,
            "fusion": fusion_params,
            "vis_placeholder": vis_params,
            "proto_attn": proto_params,
            "proto_out_dim": (self.proto_attn.out_dim if self.proto_attn is not None
                              else VIS_PLACEHOLDER_DIM),
            "total": text_params + kg_params + fusion_params + vis_params + proto_params,
            "trainable": trainable,
        }
