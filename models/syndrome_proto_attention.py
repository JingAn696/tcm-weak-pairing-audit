"""
病性视觉原型注意力（Stage B · B-2）
====================================

把 Stage B-1 训好的「10 个病性视觉原型」接进完整模型 v2 的视觉分支，
替换 Stage A 的"可学习缺失舌象占位向量"。

核心机制（论文 Method §3.4 素材）
--------------------------------
TCM-SD 是纯文本语料，没有逐样本舌象图片 —— 这是"弱配对（weak pairing）"。
做法不是"给文本配一张图"，而是把舌象知识当作**先验查询表**：

    文本表征 f_text (768)
        │  W_q（可学习投影，这是唯一的对齐参数）
        ▼
    查询向量 q (512, L2 归一化)
        │  与 10 个冻结的舌象原型做余弦相似度
        ▼
    sim (B, 10) ──mask 无效原型──► softmax(sim / τ) ──► attn (B, 10)
        │                                                   │
        │                                                   ▼
        │                                     f_vis = Σ_j attn_j · p_j (B, 512)
        ▼
    f_vis = [f_vis ; sim] (B, 522)  ──► CrossModalFusion

三个设计决定（都有理由，写论文时可直接引用）
------------------------------------------
1. **原型冻结**（registered buffer，非 nn.Parameter）
   与 10×10 SoftCL 矩阵、21×10 映射矩阵一致：这是**领域先验注入**，不是可学参数。
   若让原型可学，它会被文本交叉熵拉向"能降低文本损失"的方向，
   失去舌象语义 —— 那就退化成又一个 MLP 了。

2. **mask 掉无支撑原型**
   气滞（5 图）、内风（0 图）的原型方向不可信。若不 mask，softmax 会把
   注意力分给它们，产生无意义的视觉向量。阈值 `--proto_min_support`（默认 30）。

3. **同时输出"聚合向量"和"相似度向量"**
   f_vis 汇总了"最相关的舌象长什么样"；sim 保留了"每种病性各自的视觉证据强度"。
   后者在分类头上可能是更直接的特征（尤其对血虚这种文本弱类）。
   消融开关：`include_sim=False`（只用聚合向量，vis_dim=512）。

⚠️ 学术诚实（必须写进 Discussion）
   弱配对下这个视觉分支**不是真正的第二模态** —— 它没有看到任何新图像，
   而是"用文本激活的先验舌象模板"。它能提供的增量信息，全部来自
   TCM-Tongue 数据集里学到的那 10 个原型方向。若 B-2 无提升，
   结论应是"弱配对桥接在此数据规模下不足以带来增益"，而不是"多模态没用"。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # 本机无 torch 也能 import（仅做结构检查）
    torch = None
    nn = None
    F = None


if nn is None:
    # 无 torch 时的占位基类：保证模块可 import、可做结构检查；
    # 真去实例化会立刻报出可读的错误（而不是 AttributeError: NoneType）。
    class _NoTorchBase:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SyndromeProtoAttention 需要 torch 环境（本机未安装 torch）。"
                "请在 AutoDL 实例上运行。"
            )

    _ModuleBase = _NoTorchBase
    _no_grad = lambda f: f  # noqa: E731  无 torch 时装饰器直通
else:
    _ModuleBase = nn.Module
    _no_grad = torch.no_grad


from .tongue_label_mapping import NUM_SYNDROMES, SYNDROMES_ZH

# ------------------------------------------------------------
# 版本标记（与 PROTO_VERSION 同样的用途：确认服务器上是新版）
# 服务端自查：
#   python -c "from models.syndrome_proto_attention import B2_VERSION; print(B2_VERSION)"
# 历史：
#   2026-09-14c  B-2 初版：文本查询原型注意力 + mask + 可选 sim 拼接
# ------------------------------------------------------------
B2_VERSION = "2026-09-14c"

# 判定"原型可用"的默认支撑图数下界（气滞=5、内风=0 会被 mask）
DEFAULT_MIN_SUPPORT = 30.0


class SyndromeProtoAttention(_ModuleBase):
    """文本查询舌象原型 → 视觉表征（弱配对桥接层）。"""

    def __init__(
        self,
        text_dim: int = 768,
        proto_dim: int = 512,
        n_syndromes: int = NUM_SYNDROMES,
        init_temp: float = 0.1,
        include_sim: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if nn is None:
            raise ImportError("需要 torch 环境。")

        self.text_dim = text_dim
        self.proto_dim = proto_dim
        self.n_syndromes = n_syndromes
        self.include_sim = include_sim

        # 唯一的可学习模块：把文本空间对齐到舌象原型空间
        self.query_proj = nn.Sequential(
            nn.Linear(text_dim, proto_dim),
            nn.LayerNorm(proto_dim),
            nn.Dropout(dropout),
        )

        # 冻结原型（buffer，会进 state_dict 但无梯度）
        self.register_buffer("prototypes", torch.zeros(n_syndromes, proto_dim))
        self.register_buffer("proto_valid", torch.zeros(n_syndromes, dtype=torch.bool))
        self.register_buffer("proto_support", torch.zeros(n_syndromes))

        # 可学习温度（log 空间参数化，数值稳定）；init_temp=0.1 → logits=10*cos
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

        self.out_dim = proto_dim + (n_syndromes if include_sim else 0)

    # ---------- 原型装载 ----------

    def load_prototypes_from_file(
        self,
        path: str | Path,
        min_support: float = DEFAULT_MIN_SUPPORT,
        verbose: bool = True,
    ) -> Dict[str, object]:
        """
        从 B-1 产出的 tongue_prototypes.pt 装载原型。

        该文件由 train_tongue_branch.py 保存，含：
            prototypes : (10, 512)  已 L2 归一化
            syndromes  : list[str]
            meta       : {"n_images": [...], "pre_norm": [...], "strategy": ...}

        返回装载摘要（含被 mask 的病性名单，训练脚本会打印）。
        """
        if torch is None:
            raise ImportError("需要 torch。")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"原型文件不存在：{path}\n"
                f"请先跑 B-1：python training/train_tongue_branch.py --tongue_root ..."
                f"（产出 runs/tongue_branch*/tongue_prototypes.pt）"
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        P = ckpt["prototypes"].float()
        if P.shape != (self.n_syndromes, self.proto_dim):
            raise ValueError(
                f"原型形状 {tuple(P.shape)} 与模型期望 "
                f"({self.n_syndromes}, {self.proto_dim}) 不符"
            )

        meta = ckpt.get("meta", {}) or {}
        n_img = meta.get("n_images") or [0] * self.n_syndromes
        n_img = [float(x) for x in n_img]

        norms = P.norm(dim=1)
        support = torch.tensor(n_img, dtype=torch.float32)
        valid = (norms > 1e-6) & (support >= float(min_support))

        self.prototypes.copy_(P)
        self.proto_valid.copy_(valid)
        self.proto_support.copy_(support)

        masked = [SYNDROMES_ZH[j] for j in range(self.n_syndromes) if not bool(valid[j])]
        summary = {
            "path": str(path),
            "strategy": meta.get("strategy", "?"),
            "min_support": float(min_support),
            "n_valid": int(valid.sum().item()),
            "masked": masked,
            "support": {SYNDROMES_ZH[j]: n_img[j] for j in range(self.n_syndromes)},
            "pre_norm": meta.get("pre_norm"),
        }
        if verbose:
            print(f"  ✓ 装载原型：{path}")
            print(f"    策略={summary['strategy']} | 有效 {summary['n_valid']}/{self.n_syndromes} | "
                  f"mask 掉：{masked if masked else '无'}")
            if meta.get("pre_norm"):
                pn = meta["pre_norm"]
                print("    专属信号强度(pre_norm)：" + "  ".join(
                    f"{SYNDROMES_ZH[j]}={pn[j]:.2f}" for j in range(self.n_syndromes)))
        return summary

    # ---------- 前向 ----------

    def forward(self, f_text: "torch.Tensor"):
        """
        f_text : (B, text_dim)

        返回
        ----
        f_vis : (B, out_dim)  拼接了 sim 的视觉表征
        attn  : (B, 10)       文本对各病性原型的注意力（论文可视化用）
        sim   : (B, 10)       余弦相似度（无效原型已置 0）
        """
        q = F.normalize(self.query_proj(f_text), p=2, dim=-1)      # (B, 512)
        P = F.normalize(self.prototypes.float(), p=2, dim=-1)      # (10, 512)
        sim = q @ P.t()                                            # (B, 10)

        valid = self.proto_valid.to(sim.device)                    # (10,)
        sim_masked = sim.masked_fill(~valid.unsqueeze(0), 0.0)     # 无效位清零

        tau = self.log_temp.exp().clamp(min=1e-3)
        logits = sim_masked / tau
        logits = logits.masked_fill(~valid.unsqueeze(0), -1e4)     # softmax 前压掉
        attn = torch.softmax(logits, dim=-1)                       # (B, 10)

        pooled = attn @ P                                          # (B, 512)
        if self.include_sim:
            f_vis = torch.cat([pooled, sim_masked], dim=-1)        # (B, 522)
        else:
            f_vis = pooled
        return f_vis, attn, sim_masked

    # ---------- 诊断 ----------

    def extra_repr(self) -> str:
        return (f"text_dim={self.text_dim}, proto_dim={self.proto_dim}, "
                f"include_sim={self.include_sim}, out_dim={self.out_dim}")

    @_no_grad
    def param_stats(self) -> Dict[str, int]:
        return {
            "query_proj": sum(p.numel() for p in self.query_proj.parameters()),
            "temperature": 1,
            "prototypes_frozen": self.prototypes.numel(),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


# ============================================================
# 自测（无 torch 时降级；有 torch 时构造假原型跑通前向）
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print(f"SyndromeProtoAttention 自测  (B2_VERSION={B2_VERSION})")
    print("=" * 72)

    if torch is None:
        print("⚠️  torch 未安装，跳过前向测试（结构检查已完成）。")
    else:
        torch.manual_seed(0)
        B, T, D = 4, 768, 512
        m = SyndromeProtoAttention(text_dim=T, proto_dim=D, include_sim=True)
        assert m.out_dim == D + NUM_SYNDROMES, m.out_dim

        # 造 10 个单位原型，其中 2 个无效（模拟 气滞/内风）
        P = F.normalize(torch.randn(NUM_SYNDROMES, D), dim=1)
        m.prototypes.copy_(P)
        valid = torch.ones(NUM_SYNDROMES, dtype=torch.bool)
        valid[4] = valid[8] = False          # 气滞(4)、内风(8)
        m.proto_valid.copy_(valid)
        m.proto_support.copy_(torch.tensor([100.] * NUM_SYNDROMES))

        f_text = torch.randn(B, T)
        f_vis, attn, sim = m(f_text)
        print(f"[1] 输出形状 f_vis={tuple(f_vis.shape)}（应 ({B}, {D+NUM_SYNDROMES})）")
        assert f_vis.shape == (B, D + NUM_SYNDROMES)
        print(f"[2] attn={tuple(attn.shape)}  sim={tuple(sim.shape)}")
        assert attn.shape == (B, NUM_SYNDROMES)

        s = attn.sum(dim=1)
        print(f"[3] 注意力每行和 = {s.tolist()[0]:.6f}（应 = 1）")
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)

        print(f"[4] mask 位注意力 = {attn[:, [4, 8]].max().item():.3e}（应 ≈ 0）")
        assert attn[:, [4, 8]].max().item() < 1e-6
        print(f"[5] mask 位 sim     = {sim[:, [4, 8]].abs().max().item():.3e}（应 = 0）")
        assert sim[:, [4, 8]].abs().max().item() < 1e-9

        # 梯度：query_proj 有梯度，prototypes 无梯度
        loss = f_vis.pow(2).mean()
        loss.backward()
        g = m.query_proj[0].weight.grad
        print(f"[6] query_proj 梯度范数 = {g.norm().item():.4f}（应 > 0）")
        assert g is not None and g.norm().item() > 0
        print(f"[7] prototypes 是否 buffer（无梯度）: {not m.prototypes.requires_grad}（应 True）")
        assert not m.prototypes.requires_grad
        # query_proj = Linear(T→D) 权重 T*D + 偏置 D；LayerNorm(D) = 2*D；温度 1
        n_expect = T * D + D + 2 * D + 1
        print(f"[8] 可训练参数 = {m.param_stats()['trainable']:,}"
              f"（应为 {n_expect:,} = {T}*{D} + 偏置 {D} + LayerNorm {2*D} + 温度 1）")
        assert m.param_stats()["trainable"] == n_expect, m.param_stats()["trainable"]

        # include_sim=False
        m2 = SyndromeProtoAttention(text_dim=T, proto_dim=D, include_sim=False)
        m2.prototypes.copy_(P)
        m2.proto_valid.copy_(valid)
        out2, _, _ = m2(torch.randn(B, T))
        print(f"[9] include_sim=False → f_vis={tuple(out2.shape)}（应 ({B}, {D})）")
        assert out2.shape == (B, D)

    print("\n自测通过 ✓")
    print("=" * 72)
