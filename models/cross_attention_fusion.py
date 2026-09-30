"""
Cross-Attention Multimodal Fusion Module（路线 C 核心）
=====================================================

将 3 路输入对齐到统一的多模态表征空间：

  1) 文本分支特征  f_text ∈ R^(B, 512)              BiomedCLIP 文本编码器
  2) 视觉分支特征  f_vis  ∈ R^(B, 512)              BiomedCLIP 视觉编码器
  3) KG 嵌入特征  f_kg   ∈ R^(B, 64)               PyKEEN TransE 训练得到

设计哲学（为什么需要 cross-attention）：

  • 简单的 concat + MLP 不能建模模态间的"软对齐"
    —— 文本里"心慌"和舌象里"舌尖红"哪个权重高？concat 给不出这个答案。
  • Cross-attention 让"文本 token"和"视觉 patch"互相查询，
    让 KG 节点作为"桥接键"，把语料和舌象拉到共同语义空间。
  • 论文里这是相对 KG-only baseline 的关键创新点（呼应 Discussion §x.x）。

结构（与论文 Figure 4 对齐）：

    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ f_text   │   │ f_vis    │   │ f_kg     │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         ▼              ▼              ▼
    ┌────────┐    ┌────────┐    ┌────────┐
    │Linear  │    │Linear  │    │Linear  │  ← 各投到 d_model=256
    │ 512→256│    │ 512→256│    │ 64→256 │
    └────┬───┘    └────┬───┘    └────┬───┘
         │              │              │
    ┌────┴──────────────┴──────────────┴────┐
    │         Cross-Attention 层           │  ← 多头注意力
    │   (text query → vis+kg as K/V)        │
    │   (vis  query → text+kg as K/V)        │
    │   (kg   query → text+vis as K/V)        │
    └────┬──────────────┬──────────────┬────┘
         │              │              │
         ▼              ▼              ▼
    ┌────────┐    ┌────────┐    ┌────────┐
    │  ff_文本 │    │  ff_视觉 │    │  ff_kg  │  ← 各自前馈
    └────┬───┘    └────┬───┘    └────┬───┘
         │              │              │
         └──────────────┼──────────────┘
                        ▼
                  Concat & 分类头
                  （10 病性 logits）

为什么用 3 路交叉而不是"文本作为 query、视觉作为 key"那样单向：
    • 真实中医辨证里"舌头提示诊断方向，文本反向验证舌象"，双向信号更强。
    • KG 作为共同信息源可以独立 query，避免被单一模态主导。
    • BiGRU/BiLSTM 也可以叠加（论文里可以加一句 future work）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ProjectionHead(nn.Module):
    """模态投影头：把不同维度的输入特征统一到 d_model。"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _CrossAttentionBlock(nn.Module):
    """
    单向 Cross-Attention 块（pre-norm + residual）：

        q_input = ln(q_input)
        out = MHA(query=q_input, key=k_input, value=v_input) + q_input
        out = ln(out)
        out = FFN(out) + out
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """q: (B, 1, D)；kv: (B, K, D)；返回 (B, 1, D)。"""
        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(kv)
        attn_out, _ = self.attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = q + attn_out
        x = x + self.ffn(self.ln_ffn(x))
        return x


class CrossModalFusion(nn.Module):
    """
    路线 C 论文 §3.3 完整的多模态融合模块。

    三路输入 + 三向交叉注意力 + 各自前馈 + 拼接分类。

    参数
    ----
    text_dim : int
        BiomedCLIP 文本特征维度（默认 512）。
    vis_dim : int
        BiomedCLIP 视觉特征维度（默认 512）。
    kg_dim : int
        KG 嵌入维度（默认 64，受 PyKEEN 配置）。
    d_model : int
        融合维度（论文里 256）。
    n_classes : int
        最终分类数（10 病性，多标签二分类）。
    n_heads : int
        多头注意力头数。
    dropout : float
        dropout 概率。
    """

    def __init__(
        self,
        text_dim: int = 512,
        vis_dim: int = 512,
        kg_dim: int = 64,
        d_model: int = 256,
        n_classes: int = 10,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # === 各模态投影到 d_model ===
        self.text_proj = _ProjectionHead(text_dim, d_model, dropout)
        self.vis_proj = _ProjectionHead(vis_dim, d_model, dropout)
        self.kg_proj = _ProjectionHead(kg_dim, d_model, dropout)

        # === 三向 cross-attention ===
        # text 视角：以 text 为 query，看 vis+kg
        self.cross_text = _CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
        # vis 视角：以 vis 为 query，看 text+kg
        self.cross_vis = _CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)
        # kg 视角：以 kg 为 query，看 text+vis
        self.cross_kg = _CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout)

        # === 各路前馈（与 cross-attn 配合的双层结构用 add+norm，本模块已包含） ===
        # 保留的辅助结构：让每路表征再过一次前馈增强非线性
        self.enhance_text = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.enhance_vis = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.enhance_kg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # === 拼接 + 分类头 ===
        # 三路表征，每路 d_model；concat → 3*d_model
        self.classifier = nn.Sequential(
            nn.Linear(3 * d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
            # 注意：多标签二分类，用 BCEWithLogitsLoss 在外部处理
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化，避免 deep attention 训练初期震荡。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        f_text: torch.Tensor,
        f_vis: torch.Tensor,
        f_kg: torch.Tensor,
        return_features: bool = False,
    ):
        """
        参数
        ----
        f_text : (B, text_dim)  文本编码器 pooled feature
        f_vis  : (B, vis_dim)   视觉编码器 pooled feature
        f_kg   : (B, kg_dim)    KG 嵌入特征
        return_features : 是否同时返回融合表征（SoftCL 需要，默认 False 兼容旧调用）

        返回
        ----
        logits : (B, n_classes)  10 病性 logits（多标签二分类）
        （若 return_features=True，返回 (logits, fused)，
          fused : (B, 3*d_model) 分类头之前的融合表征）
        """
        # 1. 投影到 d_model，并加成 sequence 维度 (B, 1, D)
        text_emb = self.text_proj(f_text).unsqueeze(1)  # (B, 1, D)
        vis_emb = self.vis_proj(f_vis).unsqueeze(1)     # (B, 1, D)
        kg_emb = self.kg_proj(f_kg).unsqueeze(1)        # (B, 1, D)

        # 2. 把三路合并成"K/V 候选"
        kv_text_vis = torch.cat([text_emb, vis_emb], dim=1)  # (B, 2, D)
        kv_text_kg = torch.cat([text_emb, kg_emb], dim=1)    # (B, 2, D)
        kv_vis_kg = torch.cat([vis_emb, kg_emb], dim=1)     # (B, 2, D)

        # 3. 三向 cross-attention
        #   text 视角: kv = vis + kg
        #   vis  视角: kv = text + kg
        #   kg   视角: kv = text + vis
        text_out = self.cross_text(text_emb, kv_vis_kg).squeeze(1)
        vis_out = self.cross_vis(vis_emb, kv_text_kg).squeeze(1)
        kg_out = self.cross_kg(kg_emb, kv_text_vis).squeeze(1)

        # 4. 各自增强（额外 FFN）
        text_out = text_out + self.enhance_text(text_out)
        vis_out = vis_out + self.enhance_vis(vis_out)
        kg_out = kg_out + self.enhance_kg(kg_out)

        # 5. 拼接 + 分类
        fused = torch.cat([text_out, vis_out, kg_out], dim=-1)  # (B, 3D)
        logits = self.classifier(fused)  # (B, n_classes)
        if return_features:
            return logits, fused
        return logits


# === 单元测试（不依赖 torch 安装也能 import 本模块） ===
if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        print("⚠️  torch 未安装。Phase 2 上 GPU 时镜像会预装。")
        print("   本模块可在有 torch 后执行：python -m models.cross_attention_fusion")
    else:
        B = 4
        f_text = torch.randn(B, 512)
        f_vis = torch.randn(B, 512)
        f_kg = torch.randn(B, 64)

        model = CrossModalFusion(
            text_dim=512, vis_dim=512, kg_dim=64,
            d_model=256, n_classes=10, n_heads=8,
        )

        # 参数量统计
        n_params = sum(p.numel() for p in model.parameters())
        print(f"CrossModalFusion 参数量: {n_params:,}")
        print(f"输入维度: text={f_text.shape}, vis={f_vis.shape}, kg={f_kg.shape}")

        logits = model(f_text, f_vis, f_kg)
        print(f"输出 logits 形状: {logits.shape}（应为 (4, 10)）")
        print(f"输出 logits 数值范围: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

        # 反向传播测试
        labels = torch.randint(0, 2, (B, 10)).float()  # 多标签 0/1
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        loss.backward()
        print(f"\n✓ 反向传播通过；loss = {loss.item():.4f}")
