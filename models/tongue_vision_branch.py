"""
舌象视觉分支（Stage B · B-1）
==============================

把 TCM-Tongue 的舌象图像编码成"病性级视觉证据"，供完整模型 v2 的视觉分支使用。

为什么不能直接"图像 → 10 病性"
--------------------------------
TCM-Tongue 只有舌象形态标签（21 类，无病性标签），没有 10 病性 ground truth。
所以视觉分支走**两跳**：

    舌象图 ──BiomedCLIP 视觉塔──► 512 维表征
                                    ├─► 21 类舌象头（有 GT，可监督训练）
                                    └─► 经映射矩阵 M 投影 ──► 10 病性视觉证据

M（21×10，净安 2026-09-14 定稿）是**固定 0/1 先验、不参与训练** ——
它是"领域知识注入"，不是"可学参数"。这一点在论文 Method 里要写清楚。

与完整模型 v2 的衔接（B-2 阶段）
--------------------------------
TCM-SD（文本数据集）**没有逐样本舌象图片**，所以推理时不可能真的"看图"。
Stage B 的桥接方式是**病性视觉原型（prototype）**：

    1. 用本模块训练好的编码器遍历 TCM-Tongue 全库，提 512 维特征
    2. 每个病性 j 收集所有映射到 j 的图片，加权平均 → proto_j ∈ R^512
    3. 完整模型里：视觉分支 = Attention(文本表征 → {proto_j}_{j=1..10})

这样推理时"虽然没图，但文本可以查询到该病性对应的舌象知识"，
即所谓的**弱配对（weakly-paired）跨模态桥接**。

数据可用性警告（2026-09-14 实测）
--------------------------------
4 个「凸起」类近乎空（piweitu 0 / shenqutu 2 / xinfeitu 2 / gandantu 9 实例），
→ 视觉侧**实际覆盖 8/10 病性**，气滞与内风无信号（数据集覆盖缺口，论文 limitation）。
训练时对训练集里零正样本的类别要屏蔽其损失贡献（见 loss 的 class_mask）。

与 biomedclip_wrapper.py 的区别
-------------------------------
- wrapper：推理用，`encode_image` 带 `torch.no_grad()`，默认冻结全部
- 本模块：**训练用**，可解冻视觉塔后 N 层做小 lr 微调（TCM 舌象域差大，
  纯冻结大概率学不动 —— Stage A 的教训）

依赖：torch + open_clip_torch（本机无 GPU 时仅做结构检查，不加载权重）
"""

from __future__ import annotations

import os

# 国内下载 HuggingFace 模型必需（setdefault：用户已 export 时不覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import re
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # 本机无 torch 也能 import（仅做结构检查）
    torch = None
    nn = None
    F = None

from .tongue_label_mapping import (
    HARD_MATRIX,
    NUM_SYNDROMES,
    NUM_TONGUE_CLASSES,
    SYNDROMES_ZH,
    count_isolated_white_coating,
    prune_normal_white_coating_batch,
)

DEFAULT_MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
FEAT_DIM = 512  # 与完整模型 v2 的视觉占位维度对齐（VIS_PLACEHOLDER_DIM）

# ------------------------------------------------------------
# 版本标记（2026-09-14 加入）
# 用途：确认服务器上这份文件是否为最新版，避免"传漏了但报错莫名其妙"。
# 服务端自查：python -c "from models.tongue_vision_branch import PROTO_VERSION; print(PROTO_VERSION)"
# 历史：
#   2026-09-14a  初版（mean 策略 only）
#   2026-09-14b  新增 strategy=centered/centered_idf + prototype_similarity()
#   2026-09-15   映射矩阵升级到 2026-09-15 版（tongue_label_mapping.py 同步更新）；
#                extract_syndrome_prototypes 新增 prune_normal_white 参数（B6 判据）
# ------------------------------------------------------------
PROTO_VERSION = "2026-09-15"


# ============================================================
# 1. 加载 BiomedCLIP 视觉塔（可微调版）
# ============================================================

def load_biomedclip_visual(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    probe_dim: bool = True,
) -> Tuple["nn.Module", object, int, int]:
    """
    加载 BiomedCLIP 的视觉塔（open_clip），返回可微调的 visual 模块。

    返回
    ----
    (visual, preprocess, raw_dim, image_size)
        visual     : open_clip 的 visual 子模块（ViT-B/16）
        preprocess : torchvision transform（Resize + CenterCrop + Normalize）
        raw_dim    : visual 输出维度（Vit-B 为 768；若框架带 proj 则为 512）
        image_size : 输入边长（224）
    """
    if torch is None:
        raise ImportError("需要 torch 环境才能加载 BiomedCLIP。")

    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "需要 open_clip_torch。AutoDL 上执行：pip install open_clip_torch"
        ) from e

    name = model_name
    if "/" not in name and not name.startswith("hf-hub:"):
        name = f"microsoft/{name}"
    tag = f"hf-hub:{name}"

    model, preprocess = open_clip.create_model_from_pretrained(tag, cache_dir=cache_dir)
    visual = model.visual
    visual.to(device)

    image_size = getattr(visual, "image_size", 224)
    if isinstance(image_size, (tuple, list)):
        image_size = int(image_size[0])
    image_size = int(image_size)

    raw_dim = FEAT_DIM
    if probe_dim:
        was_training = visual.training
        visual.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=device)
            raw_dim = int(visual(dummy).shape[-1])
        visual.train(was_training)

    return visual, preprocess, raw_dim, image_size


def configure_finetune(
    visual: "nn.Module",
    n_unfreeze_blocks: int = 4,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    按 transformer block 索引冻结/解冻视觉塔（PEFT 式小 lr 微调）。

    策略（净安 2026-09-14 拍板"分类头 + 视觉塔后几层小 lr 微调"）：
      - patch_embed / pos_embed  → 冻结（底层纹理特征，不该动）
      - blocks.0 .. blocks.(L-N-1) → 冻结
      - blocks.(L-N) .. blocks.(L-1) → 可训
      - 其余（final norm、proj 等）  → 可训

    参数
    ----
    n_unfreeze_blocks : 解冻最后 N 个 transformer block（0 = 全冻结）

    返回
    ----
    {"n_blocks": L, "unfrozen_blocks": N, "trainable": ..., "frozen": ..., "trainable_ratio": ...}
    """
    block_ids = set()
    for name, _ in visual.named_parameters():
        m = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
        if m:
            block_ids.add(int(m.group(1)))

    if not block_ids:
        raise RuntimeError(
            "未能在视觉塔参数名中找到 `blocks.N.` 结构，无法按层解冻。"
            "请检查 open_clip 版本，或改用 n_unfreeze_blocks=0 之外的手工冻结。"
        )

    keep = set(sorted(block_ids)[-n_unfreeze_blocks:]) if n_unfreeze_blocks > 0 else set()
    freeze_names = ("patch_embed", "pos_embed", "class_embed", "cls_token")

    n_train = 0
    n_frozen = 0
    for name, p in visual.named_parameters():
        m = re.search(r"(?:^|\.)blocks\.(\d+)\.", name)
        if any(k in name for k in freeze_names):
            p.requires_grad = False
        elif m:
            p.requires_grad = int(m.group(1)) in keep
        else:
            p.requires_grad = True  # final norm / proj
        if p.requires_grad:
            n_train += p.numel()
        else:
            n_frozen += p.numel()

    stats = {
        "n_blocks": len(block_ids),
        "unfrozen_blocks": len(keep),
        "trainable": n_train,
        "frozen": n_frozen,
        "trainable_ratio": n_train / max(n_train + n_frozen, 1),
    }
    if verbose:
        print(f"  视觉塔：{stats['n_blocks']} blocks，解冻最后 {stats['unfrozen_blocks']} 个 | "
              f"可训 {n_train/1e6:.2f}M / 冻结 {n_frozen/1e6:.2f}M "
              f"({stats['trainable_ratio']*100:.1f}%)")
    return stats


# ============================================================
# 2. 视觉分支模型
# ============================================================

class TongueVisionBranch(nn.Module):
    """
    BiomedCLIP 视觉塔 + 512 维投影 + 21 类舌象头 + 映射投影到 10 病性。

    前向输出
    --------
    feat   : (B, 512)  舌象表征（供原型提取 / 接回完整模型）
    z21    : (B, 21)   21 类舌象 logits（有 GT，主监督）
    z10    : (B, 10)   10 病性视觉证据 logits（经固定映射 M 聚合，供诊断/消融）
    p10    : (B, 10)   10 病性视觉证据概率（∈[0,1]）
    """

    def __init__(
        self,
        visual_backbone: "nn.Module",
        raw_dim: int,
        feat_dim: int = FEAT_DIM,
        n_tongue: int = NUM_TONGUE_CLASSES,
        n_syndrome: int = NUM_SYNDROMES,
        dropout: float = 0.1,
        mapping_mode: str = "binary",
    ):
        super().__init__()
        if nn is None:
            raise ImportError("TongueVisionBranch 需要 torch 环境。")

        self.backbone = visual_backbone
        self.raw_dim = raw_dim
        self.feat_dim = feat_dim
        self.n_tongue = n_tongue
        self.n_syndrome = n_syndrome

        # 视觉塔输出 → 统一 512 维（与完整模型 v2 的 vis_dim 对齐）
        self.feat_norm = nn.LayerNorm(raw_dim)
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 21 类舌象分类头（主监督目标）
        self.tongue_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_tongue),
        )

        # 固定映射矩阵 M（21×10）—— **不训练**
        # 列归一化：每个病性的证据 = 其所有舌象来源概率的**平均**
        # （否则来源多的病性天然分数高，如痰湿 6 个来源 vs 血瘀 2 个）
        m = torch.tensor(HARD_MATRIX, dtype=torch.float32) if mapping_mode == "binary" \
            else torch.tensor([[v / (sum(r) or 1) for v in r] for r in HARD_MATRIX], dtype=torch.float32)
        col_sum = m.sum(dim=0).clamp(min=1.0)
        m_norm = m / col_sum.unsqueeze(0)  # (21, 10)，每列和为 1
        self.register_buffer("mapping_matrix", m, persistent=False)
        self.register_buffer("mapping_matrix_colnorm", m_norm, persistent=False)

    # ---------- 编码 ----------

    def encode(self, images: "torch.Tensor") -> "torch.Tensor":
        """图像 → 512 维舌象表征。"""
        raw = self.backbone(images)          # (B, raw_dim)
        if raw.dim() > 2:                    # 个别版本返回 token 序列
            raw = raw[:, 0]
        return self.proj(self.feat_norm(raw))

    def forward(self, images: "torch.Tensor", return_dict: bool = False):
        """
        参数
        ----
        images : (B, 3, S, S) 已 preprocess 的整张舌象图
        return_dict : True → 返回 dict；False → 返回 (feat, z21, z10, p10)

        返回
        ----
        feat (B,512), z21 (B,21), z10 (B,10), p10 (B,10)
        """
        feat = self.encode(images)             # (B, 512)
        z21 = self.tongue_head(feat)           # (B, 21)

        # 21 类概率 → 经固定映射聚合到 10 病性（概率域，保持可解释性）
        p21 = torch.sigmoid(z21)               # (B, 21)
        p10 = p21 @ self.mapping_matrix_colnorm  # (B, 21) @ (21, 10) = (B, 10)
        p10 = p10.clamp(1e-6, 1 - 1e-6)
        z10 = torch.log(p10 / (1 - p10))       # logit 变换，尺度和文本分支一致

        if return_dict:
            return {"feat": feat, "z21": z21, "z10": z10, "p10": p10}
        return feat, z21, z10, p10

    # ---------- 损失 ----------

    def compute_loss(
        self,
        z21: "torch.Tensor",
        targets: "torch.Tensor",
        class_mask: Optional["torch.Tensor"] = None,
        pos_weight: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Dict[str, float]]:
        """
        21 类舌象多标签 BCE。

        参数
        ----
        z21        : (B, 21) 预测 logits
        targets    : (B, 21) 0/1 检测标签（图像级聚合）
        class_mask : (21,) 1/0，训练集里零正样本的类别置 0 → 屏蔽其损失
                     （脾胃区凸起 0 实例，若不屏蔽，BCE 会一直把它往负类压，污染共享特征）
        pos_weight : (21,) 正样本权重（处理类别极不平衡，白苔 73% vs 紫舌 3.4%）

        返回 (loss, {"bce": float})
        """
        loss = F.binary_cross_entropy_with_logits(
            z21, targets.float(), pos_weight=pos_weight, reduction="none"
        )  # (B, 21)
        if class_mask is not None:
            loss = loss * class_mask.unsqueeze(0)
            denom = (class_mask.unsqueeze(0) * torch.ones_like(loss)).sum().clamp(min=1.0)
            loss = loss.sum() / denom
        else:
            loss = loss.mean()
        return loss, {"bce": float(loss.item())}

    # ---------- 参数统计（论文用） ----------

    def param_stats(self) -> Dict[str, float]:
        backbone_all = sum(p.numel() for p in self.backbone.parameters())
        backbone_train = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.tongue_head.parameters()) + \
            sum(p.numel() for p in self.proj.parameters()) + \
            sum(p.numel() for p in self.feat_norm.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone_total": backbone_all,
            "backbone_trainable": backbone_train,
            "head": head,
            "trainable": trainable,
            "trainable_ratio": trainable / max(backbone_all + head, 1),
        }


# ============================================================
# 3. 病性视觉原型提取（B-2 的输入）
# ============================================================

@torch.no_grad()
def extract_syndrome_prototypes(
    model: "TongueVisionBranch",
    feats: "torch.Tensor",
    det_labels: "torch.Tensor",
    use_gt: bool = True,
    min_weight: float = 1e-6,
    l2_normalize: bool = True,
    strategy: str = "mean",
    prune_normal_white: bool = False,
) -> Tuple["torch.Tensor", Dict[str, object]]:
    """
    从舌象特征 + 检测标签构建 10 个病性视觉原型。

    参数
    ----
    model      : 训练好的 TongueVisionBranch（只用来取 mapping_matrix）
    feats      : (N, 512) 全库舌象特征（由 model.encode 提取）
    det_labels : (N, 21) 0/1 检测标签（GT 或模型预测，见 use_gt）
    use_gt     : True → det_labels 是 0/1 GT，直接作权重（干净，推荐）
                 False → det_labels 是模型预测概率，先按 0.5 阈值过滤再作权重
    l2_normalize : 原型是否 L2 归一化（后续做余弦注意力时需要）
    strategy   : 原型构造策略（三种，论文里可做消融）
        - "mean"         : 病性所有来源图的**特征平均**
                           ⚠️ 2026-09-14 实测：来源高度重叠时原型会重合
                           （气虚×痰湿 0.994、气虚×阳虚 0.994），不推荐单独使用
        - "centered"     : 先算 21 个舌象类的平均特征 μ_i，减去全局均值 μ_0 得 δ_i，
                           病性原型 = Σ_i M[i,j]·δ_i
                           —— **高频共享来源被自动降权**（白苔舌覆盖 73%，
                           μ_白苔 ≈ μ_全局 → δ_白苔 ≈ 0），保留区分性来源
        - "centered_idf" : 在 centered 基础上，再乘"独占性权重" 1/(该舌象映射到的病性数)
                           —— 只映射到血虚的剥苔舌权重大，映射到 3 个病性的白苔舌权重小
    prune_normal_white : B6 判据（净安 2026-09-15 定，tongue_label_mapping.py §3b）。
        True → 先把「孤立白苔舌」（除健康舌外只有白苔舌一个标签的图）的白苔舌标签清零，
        这些图被视为"正常薄白苔"、不再向任何病性原型贡献特征。
        实测影响（2026-09-15，gen_mapping_revision_impact.py）：约 937 张图被剔除。
        对 use_gt=False（预测概率）模式基本无影响（概率稠密，没有"孤立"行）。

    返回
    ----
    protos : (10, 512)  每个病性的原型向量（顺序 = SYNDROMES_ZH）
    meta   : 诊断信息
             - n_images     : 支撑图数
             - total_weight : 有效权重和（centered_idf 下已计 IDF）
             - pre_norm     : L2 归一化前的模长；centered 策略下 = "专属视觉信号强度"
                              （越小说明该病性越缺乏可区分的舌象特征，方向接近噪声）
             - strategy     : 本次使用的策略
             - prune_normal_white / n_pruned_white : B6 判据开关与被剔除的图数
    """
    if torch is None:
        raise ImportError("需要 torch 环境。")

    m = model.mapping_matrix.to(feats.device)          # (21, 10) 硬 0/1
    w_det = det_labels.float()

    n_pruned_white = 0
    if prune_normal_white:
        n_pruned_white = count_isolated_white_coating(w_det)
        w_det = prune_normal_white_coating_batch(w_det)

    if not use_gt:
        # 预测概率模式：低于阈值的置零，避免"什么都低分"的图贡献噪声
        w_det = w_det * (w_det >= 0.5).float()

    feat_dim = feats.size(1)
    protos = torch.zeros(NUM_SYNDROMES, feat_dim, device=feats.device)
    meta = {"n_images": [], "total_weight": [], "syndromes": list(SYNDROMES_ZH),
            "strategy": strategy,
            "prune_normal_white": bool(prune_normal_white),
            "n_pruned_white": int(n_pruned_white),
            # n_images 的口径标记（2026-09-15b 修）：统一为「唯一图数」=
            # 该图中至少有一个舌象标签映射到该病性。旧版 centered 分支误写成
            # 「来源舌象类实例数之和」，一张图有 胖大舌+齿痕舌 会被计两次 → 虚高。
            "n_images_mode": "unique"}

    # 唯一图数矩阵 (N, 10) bool：两种策略共用，保证 n_images 口径一致。
    # ⚠️ 只在 use_gt=True（0/1 标签）或已过 0.5 阈值过滤（use_gt=False）时才唯一。
    W_unique = (w_det @ m) > 0

    if strategy == "mean":
        W = w_det @ m                                   # (N, 10) 每张图对每病性的证据
        for j in range(NUM_SYNDROMES):
            w = W[:, j]
            total = w.sum()
            if total < min_weight:
                meta["n_images"].append(0)
                meta["total_weight"].append(0.0)
                continue
            w_norm = w / total
            protos[j] = (w_norm.unsqueeze(1) * feats).sum(dim=0)
            meta["n_images"].append(int((w > 0).sum().item()))
            meta["total_weight"].append(float(total.item()))

    elif strategy in ("centered", "centered_idf"):
        # 1) 21 个舌象类的平均特征 μ_i（按 GT 标签加权）
        counts_i = w_det.sum(dim=0)                     # (21,)
        valid_i = counts_i >= 1
        mu_i = torch.zeros(NUM_TONGUE_CLASSES, feat_dim, device=feats.device)
        mu_i[valid_i] = (w_det.t()[valid_i] @ feats) / counts_i[valid_i].unsqueeze(1)
        # 2) 中心化：δ_i = μ_i − μ_全局
        mu_0 = feats.mean(dim=0)                        # (512,)
        delta = mu_i - mu_0
        # 3) 独占性权重（映射到的病性越少 → 区分度越高）
        if strategy == "centered_idf":
            n_syn = m.sum(dim=1).clamp(min=1.0)         # (21,)
            idf = 1.0 / n_syn
        else:
            idf = torch.ones(NUM_TONGUE_CLASSES, device=feats.device)
        Wmat = m * idf.unsqueeze(1)                     # (21, 10)
        denom = Wmat.sum(dim=0)                         # (10,)
        acc = Wmat.t() @ delta                          # (10, 512)

        for j in range(NUM_SYNDROMES):
            src_idx = (m[:, j] > 0).nonzero().flatten().tolist()
            if not src_idx or denom[j] < 1e-9:
                meta["n_images"].append(0)
                meta["total_weight"].append(0.0)
                continue
            protos[j] = acc[j] / denom[j].clamp(min=1e-9)
            # 唯一图数（2026-09-15b 修）：旧实现 = sum(counts_i[i] for i in src_idx)
            # 即「来源舌象类实例数之和」，一张图同时有 胖大舌+齿痕舌 会被计两次，
            # 使多来源病性（气虚/阳虚/痰湿）虚高；单来源病性（血虚/血瘀/气滞）不受影响。
            n_img = int(W_unique[:, j].sum().item())
            meta["n_images"].append(n_img)
            meta["total_weight"].append(float(denom[j].item()))

    else:
        raise ValueError(f"未知 strategy：{strategy}（应为 mean / centered / centered_idf）")

    # L2 归一化前的模长。对 centered / centered_idf 而言这是**该病性的"专属视觉信号强度"**：
    # 值越小 → 该病性的来源舌象与全局平均越接近（或来源太少/太差）→ 原型方向基本是噪声。
    # 例：气滞（来源舌象几乎 0 实例）、内风（无来源）应接近 0。
    meta["pre_norm"] = [float(x) for x in protos.norm(dim=1).tolist()]

    if l2_normalize:
        protos = F.normalize(protos, p=2, dim=1)
    return protos, meta


@torch.no_grad()
def prototype_similarity(protos: "torch.Tensor") -> Dict[str, object]:
    """
    原型质量的快速诊断：两两余弦相似度。

    判定标准（2026-09-14 定）：
        > 0.95 → 几乎重合，B-2 的文本查询注意力无区分度
        0.85–0.95 → 偏高，需留意
        < 0.85 → 可分

    返回 {"max_pair": ..., "n_over_095": ..., "pairs": [(a,b,sim), ...]（按相似度降序）}
    """
    P = protos.float()
    norms = P.norm(dim=1)
    keep = [j for j in range(P.size(0)) if norms[j] > 1e-6]
    if len(keep) < 2:
        return {"max_pair": None, "n_over_095": 0, "pairs": [], "zero": len(keep) == 0}
    Pn = P[keep] / norms[keep].unsqueeze(1)
    S = (Pn @ Pn.T)
    pairs = []
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            pairs.append({
                "a": SYNDROMES_ZH[keep[a]] if keep[a] < len(SYNDROMES_ZH) else str(keep[a]),
                "b": SYNDROMES_ZH[keep[b]] if keep[b] < len(SYNDROMES_ZH) else str(keep[b]),
                "cosine": float(S[a, b]),
            })
    pairs.sort(key=lambda d: -d["cosine"])
    return {
        "max_pair": pairs[0] if pairs else None,
        "mean_pairwise": float(sum(p["cosine"] for p in pairs) / max(len(pairs), 1)),
        "n_over_095": sum(1 for p in pairs if p["cosine"] > 0.95),
        "n_over_085": sum(1 for p in pairs if p["cosine"] > 0.85),
        "pairs": pairs,
    }


# ============================================================
# 4. 自测（无 torch 时只跑映射矩阵的纯逻辑检查）
# ============================================================

if __name__ == "__main__":
    print("=" * 68)
    print("舌象视觉分支（Stage B）自测")
    print("=" * 68)

    if torch is None:
        print("⚠️  torch 未安装，仅做映射矩阵结构检查（AutoDL 上有 GPU 时跑完整自测）")
    else:
        # ---- 用一个假的 backbone 验证形状与映射逻辑（不下载 BiomedCLIP）----
        class _DummyVisual(nn.Module):
            """模拟 ViT：输出 raw_dim 维向量，参数名含 blocks.N. 以测解冻逻辑。"""

            def __init__(self, raw_dim=768, n_blocks=4):
                super().__init__()
                self.patch_embed = nn.Linear(3 * 8 * 8, raw_dim)
                self.blocks = nn.ModuleList([
                    nn.Sequential(nn.Linear(raw_dim, raw_dim), nn.GELU())
                    for _ in range(n_blocks)
                ])
                self.norm = nn.LayerNorm(raw_dim)
                self.out_dim = raw_dim

            def forward(self, x):
                b = x.size(0)
                h = self.patch_embed(x.flatten(1)[:, : 3 * 8 * 8])
                for blk in self.blocks:
                    h = blk(h)
                return self.norm(h)

        dummy = _DummyVisual()
        print("\n[1] 解冻策略")
        stats = configure_finetune(dummy, n_unfreeze_blocks=2)
        assert stats["n_blocks"] == 4
        assert stats["unfrozen_blocks"] == 2
        # patch_embed 必须冻结
        assert not dummy.patch_embed.weight.requires_grad, "patch_embed 应冻结"
        # blocks.2 / blocks.3 可训，blocks.0 / blocks.1 冻结
        for i in range(4):
            p = dummy.blocks[i][0].weight
            should = i >= 2
            assert p.requires_grad == should, f"blocks.{i} 应为 {'可训' if should else '冻结'}"
        print("    ✓ patch_embed 冻结 / 后 2 个 block 可训 / 前 2 个冻结")

        print("\n[2] 前向形状")
        model = TongueVisionBranch(dummy, raw_dim=768, feat_dim=512)
        B = 4
        imgs = torch.randn(B, 3, 224, 224) * 0.1 + 0.5
        feat, z21, z10, p10 = model(imgs)
        print(f"    feat={tuple(feat.shape)} z21={tuple(z21.shape)} "
              f"z10={tuple(z10.shape)} p10={tuple(p10.shape)}")
        assert feat.shape == (B, 512)
        assert z21.shape == (B, NUM_TONGUE_CLASSES)
        assert z10.shape == (B, NUM_SYNDROMES)
        assert ((p10 >= 0) & (p10 <= 1)).all(), "p10 应在 [0,1]"

        print("\n[3] 映射矩阵固定性（不可训练）")
        assert not model.mapping_matrix.requires_grad
        assert not model.mapping_matrix_colnorm.requires_grad
        col_sum = model.mapping_matrix_colnorm.sum(dim=0)
        for j in range(NUM_SYNDROMES):
            n_src = sum(HARD_MATRIX[i][j] for i in range(NUM_TONGUE_CLASSES))
            if n_src > 0:
                assert abs(col_sum[j].item() - 1.0) < 1e-5, \
                    f"{SYNDROMES_ZH[j]} 列和应为 1，得到 {col_sum[j].item()}"
            else:
                assert col_sum[j].item() == 0.0, f"{SYNDROMES_ZH[j]} 无来源，列和应为 0"
        print("    ✓ 映射矩阵 requires_grad=False；有来源的列归一化为 1，内风列（无来源）为 0")

        print("\n[4] 映射投影正确性：只点亮「紫舌」应让 p10 只在 血瘀+阳虚 上非零")
        with torch.no_grad():
            # 直接把 p21 设为 one-hot 紫舌，验证投影
            # 2026-09-15 修订矩阵：紫舌 → 血瘀 + 阳虚（寒凝血瘀，净安 B4 结论）
            p21 = torch.zeros(1, NUM_TONGUE_CLASSES)
            p21[0, 3] = 1.0  # 3 = 紫舌
            proj = p21 @ model.mapping_matrix_colnorm
            nz = (proj[0] > 0).nonzero().flatten().tolist()
            got = {SYNDROMES_ZH[j] for j in nz}
            assert got == {"血瘀", "阳虚"}, f"紫舌应只点亮 血瘀+阳虚，得到 {got}"
            # 列归一化语义：血瘀 2 个来源（紫舌、心肺区凸起）→ 单来源贡献 1/2
            j_stasis = SYNDROMES_ZH.index("血瘀")
            n_src = sum(HARD_MATRIX[i][j_stasis] for i in range(NUM_TONGUE_CLASSES))
            expect = 1.0 / n_src
            assert abs(proj[0, j_stasis].item() - expect) < 1e-5, \
                f"期望 {expect}，得到 {proj[0, j_stasis].item()}"
            # 阳虚 6 个来源（紫舌/胖大舌/齿痕舌/白苔舌/黑苔舌/滑苔舌）→ 1/6
            j_yang = SYNDROMES_ZH.index("阳虚")
            n_src_yang = sum(HARD_MATRIX[i][j_yang] for i in range(NUM_TONGUE_CLASSES))
            assert abs(proj[0, j_yang].item() - 1.0 / n_src_yang) < 1e-5
        print(f"    ✓ 紫舌 → 血瘀 {proj[0, j_stasis].item():.3f}（{n_src} 源之一）"
              f" + 阳虚 {proj[0, j_yang].item():.3f}（{n_src_yang} 源之一）")

        print("\n[5] 损失：class_mask 屏蔽零正样本类")
        targets = torch.zeros(3, NUM_TONGUE_CLASSES)
        targets[0, 9] = 1.0   # 白苔舌
        targets[1, 2] = 1.0   # 红舌
        targets[2, 7] = 1.0   # 裂纹舌
        mask = torch.ones(NUM_TONGUE_CLASSES)
        mask[18] = 0.0        # 脾胃区凸起（训练集 0 实例）→ 屏蔽
        mask[14] = 0.0        # 肾区凸起（2 实例）
        _, z21b, _, _ = model(torch.randn(3, 3, 224, 224) * 0.1 + 0.5)
        loss, parts = model.compute_loss(z21b, targets, class_mask=mask)
        assert torch.isfinite(loss), "loss 应为有限值"
        print(f"    ✓ 带 mask 的 BCE = {parts['bce']:.4f}（屏蔽 2 个无效类）")

        print("\n[6] 原型提取")
        N = 200
        feats = torch.randn(N, 512)
        det = torch.zeros(N, NUM_TONGUE_CLASSES)
        det[:80, 3] = 1.0     # 紫舌 → 血瘀
        det[80:140, 9] = 1.0  # 白苔舌 → 气虚/阳虚/痰湿
        det[140:, 15] = 1.0   # 肝胆区凹陷 → 血虚
        protos, meta = extract_syndrome_prototypes(model, feats, det, use_gt=True)
        print(f"    原型形状 {tuple(protos.shape)}")
        assert protos.shape == (NUM_SYNDROMES, 512)
        # 血瘀应该由 80 张紫舌图撑起
        j_stasis = SYNDROMES_ZH.index("血瘀")
        assert meta["n_images"][j_stasis] == 80, f"血瘀原型应由 80 张图撑起，得到 {meta['n_images'][j_stasis]}"
        # 内风没有来源 → 零向量
        j_wind = SYNDROMES_ZH.index("内风")
        assert protos[j_wind].abs().sum() == 0, "内风原型应为零向量"
        print(f"    ✓ 血瘀原型 {meta['n_images'][j_stasis]} 张图支撑；内风原型零向量（无来源）")
        print(f"    各病性支撑图数：{dict(zip(SYNDROMES_ZH, meta['n_images']))}")

        # B6 判据（2026-09-15）：本例 60 张"孤立白苔舌"图应被剔除，不再支撑任何病性
        protos_b6, meta_b6 = extract_syndrome_prototypes(
            model, feats, det, use_gt=True, prune_normal_white=True
        )
        assert meta_b6["prune_normal_white"] is True
        assert meta_b6["n_pruned_white"] == 60, \
            f"60 张孤立白苔舌图应被剔除，得到 {meta_b6['n_pruned_white']}"
        j_qi = SYNDROMES_ZH.index("气虚")
        assert meta_b6["n_images"][j_qi] == 0, "剔除孤立白苔后气虚应无支撑图"
        # 血瘀（紫舌图）不受影响
        assert meta_b6["n_images"][j_stasis] == 80
        j_yang = SYNDROMES_ZH.index("阳虚")
        print(f"    ✓ B6 判据：60 张孤立白苔舌被剔除 → 气虚/痰湿支撑归零，"
              f"阳虚 {meta['n_images'][j_yang]} → {meta_b6['n_images'][j_yang]}（紫舌图保留），"
              f"血瘀不受影响")

        # n_images 口径回归测试（2026-09-15b 修）
        # 旧实现（centered 分支）= sum(counts_i[i] for i in 来源类)，一张图若同时有
        # 两个映射到同一病性的来源类，会被计两次。这里用「同图双来源」用例锁死新口径。
        m_np = model.mapping_matrix.detach().cpu().numpy()
        src = [i for i in range(NUM_TONGUE_CLASSES) if m_np[i, j_qi] > 0]
        assert len(src) >= 2, "自测前提：气虚应至少有 2 个来源舌象类"
        det_dup = torch.zeros(5, NUM_TONGUE_CLASSES)
        det_dup[:, src[0]] = 1.0
        det_dup[0, src[1]] = 1.0            # 第 0 张图同时带两个来源类
        _, meta_dup = extract_syndrome_prototypes(
            model, torch.randn(5, 512), det_dup, use_gt=True, strategy="centered"
        )
        got = meta_dup["n_images"][j_qi]
        assert got == 5, (f"同图重复来源类不应重复计数：期望 5 张唯一图，得到 {got}"
                          f"（旧 centered 口径会得到 6）")
        assert meta_dup.get("n_images_mode") == "unique"
        print(f"    ✓ n_images 口径：同图两个来源类只计 1 张（centered 得 {got}，旧口径 6）")

        print("\n[7] 参数统计")
        ps = model.param_stats()
        print(f"    backbone {ps['backbone_total']/1e6:.2f}M（可训 {ps['backbone_trainable']/1e6:.2f}M）"
              f" + head {ps['head']/1e6:.2f}M | 总可训 {ps['trainable']/1e6:.2f}M "
              f"({ps['trainable_ratio']*100:.1f}%)")

    print("\n" + "=" * 68)
    print("全部自测通过 ✓")
    print("=" * 68)
