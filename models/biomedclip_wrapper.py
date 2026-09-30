"""
BiomedCLIP 模型包装器
=====================

封装 Microsoft BiomedCLIP（PubMedBERT + ViT-base），提供：
1. 图像编码接口 encode_image
2. 文本编码接口 encode_text
3. 图像-文本相似度接口 encode_similarity（zero-shot 用）
4. 冻结 / 解冻控制（freeze_backbones / unfreeze_backbones / count_trainable_params）

加载方式（open_clip 原生框架，与官方 README 一致）：
    from open_clip import create_model_from_pretrained, get_tokenizer
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

参考：
- https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- 论文：Zhang et al. 2023, "Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language Processing"
  (Boecking et al. 2022 的同组后续，BiomedCLIP 本体)

注意（国内网络）：
- HuggingFace 直连较慢/不稳，AutoDL 上建议先 export HF_ENDPOINT=https://hf-mirror.com
- open_clip 走 huggingface_hub，会自动读取 HF_ENDPOINT 环境变量。

设计要点：
- 默认 lazy_load=True：构造时不下载模型，首次 forward 时才触发加载，
  这样本机（无 GPU / 无 open_clip）也能 import 本模块做语法/结构检查。
- 默认冻结视觉 + 文本 backbone（PEFT 策略：只训 fusion / KG / classifier）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 自动设 HuggingFace 国内镜像（国内网络下不设会直连 huggingface.co → Network is unreachable）
# setdefault 语义：用户已设的不覆盖，调用方可自定义
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    import torch
except ImportError:  # 本机无 torch 也能 import（仅做结构检查）
    torch = None


DEFAULT_MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


@dataclass
class BiomedCLIPConfig:
    """BiomedCLIP 配置参数（独立使用时可用）。"""
    model_name: str = DEFAULT_MODEL_NAME
    image_size: int = 224
    text_max_length: int = 256
    projection_dim: int = 512          # 视觉-文本共享向量空间维度（CLIP embed_dim）
    text_hidden_dim: int = 768         # PubMedBERT hidden size
    vision_hidden_dim: int = 768       # ViT-base hidden size
    cache_dir: Optional[str] = None    # 模型权重本地缓存目录
    freeze_image_encoder: bool = True
    freeze_text_encoder: bool = True


class BiomedCLIPWrapper:
    """
    BiomedCLIP 模型包装器（lazy load）。

    构造后不会立即下载模型；首次调用 encode_* 时才通过 open_clip 加载权重。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = "hf",
        freeze_image_encoder: bool = True,
        freeze_text_encoder: bool = True,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        lazy_load: bool = True,
    ):
        """
        Args:
            model_name: open_clip 模型标识（HF hub 上的 repo 名，不含 hf-hub: 前缀）。
            pretrained: "hf" 从 HuggingFace Hub 加载；Phase 2 可扩展为本地权重路径。
            freeze_image_encoder: 是否冻结视觉 backbone。
            freeze_text_encoder: 是否冻结文本 backbone。
            device: "cuda"/"cpu"，None 则自动检测。
            cache_dir: 模型缓存目录（HF 缓存），None 用默认。
            lazy_load: True 则延迟到首次 forward 才加载。
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_text_encoder = freeze_text_encoder
        self.device = device
        self.cache_dir = cache_dir

        # BiomedCLIP 的 CLIP 投影维度固定 512（projection_dim）
        self.text_dim = 512
        self.vision_dim = 512

        self._model = None
        self._tokenizer = None
        self.preprocess = None
        self._device = None

        if not lazy_load:
            self.load()

    # ---------- 加载 ----------

    def load(self):
        """立即加载模型（open_clip）。幂等，重复调用不重复下载。"""
        if self._model is not None:
            return self
        if torch is None:
            raise ImportError("BiomedCLIP 需要 torch 环境。AutoDL 镜像已预装。")
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "BiomedCLIP 需要 open_clip_torch。AutoDL 上执行：pip install open_clip_torch"
            ) from e

        # 规范化：允许传入 "BiomedCLIP-..."（无 org），自动补 "microsoft/"
        name = self.model_name
        if "/" not in name and not name.startswith("hf-hub:"):
            name = f"microsoft/{name}"
        tag = f"hf-hub:{name}"
        self._model, self.preprocess = open_clip.create_model_from_pretrained(
            tag, cache_dir=self.cache_dir
        )
        self._tokenizer = open_clip.get_tokenizer(tag)
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self._apply_freeze()
        return self

    def _ensure_loaded(self):
        if self._model is None:
            self.load()
        return self

    def _apply_freeze(self):
        if self._model is None:
            return
        visual = getattr(self._model, "visual", None)
        text = getattr(self._model, "text", None)
        if visual is not None:
            for p in visual.parameters():
                p.requires_grad = not self.freeze_image_encoder
        if text is not None:
            for p in text.parameters():
                p.requires_grad = not self.freeze_text_encoder

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    # ---------- 编码接口 ----------

    def encode_image(self, images) -> "torch.Tensor":
        """
        编码舌象图像 → 512 维归一化向量。

        Args:
            images: Tensor[B, 3, 224, 224]，已用 self.preprocess 预处理后的图像批次

        Returns:
            Tensor[B, 512]（open_clip 已 L2 归一化）
        """
        self._ensure_loaded()
        images = images.to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(images)
        return features

    def encode_text(self, texts: List[str]) -> "torch.Tensor":
        """
        编码文本（问诊 / 证候描述）→ 512 维归一化向量。

        Args:
            texts: List[str]，长度 B

        Returns:
            Tensor[B, 512]（open_clip 已 L2 归一化）
        """
        self._ensure_loaded()
        tokens = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        return features

    def encode_similarity(self, images, texts: List[str]) -> "torch.Tensor":
        """
        计算图像-文本相似度（zero-shot 用）。

        Args:
            images: Tensor[B, 3, 224, 224]
            texts: List[str]，长度 C

        Returns:
            Tensor[B, C]，每张图对每条文本的余弦相似度（内积，因已归一化）
        """
        image_features = self.encode_image(images)       # (B, 512)
        text_features = self.encode_text(texts)          # (C, 512)
        return image_features @ text_features.T          # (B, C)

    # ---------- 训练控制 ----------

    def freeze_backbones(self):
        """冻结视觉 + 文本 backbone。"""
        self.freeze_image_encoder = True
        self.freeze_text_encoder = True
        self._apply_freeze()
        return self

    def unfreeze_backbones(self):
        """解冻视觉 + 文本 backbone（如需整体微调）。"""
        self.freeze_image_encoder = False
        self.freeze_text_encoder = False
        self._apply_freeze()
        return self

    def count_trainable_params(self) -> Dict[str, float]:
        """统计可训练参数（论文里报"我们只训练 X% 参数"）。"""
        self._ensure_loaded()
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return {
            "total": int(total),
            "trainable": int(trainable),
            "ratio": float(trainable / total) if total else 0.0,
        }


# ---------- 预定义的证候描述模板（10 病性要素） ----------
# 顺序与 models/soft_cl_loss.py 的 SYNDROME_ORDER 对齐：
#   气虚 血虚 阴虚 阳虚 气滞 血瘀 痰湿 湿热 内风 实热

SYNDROME_TEXT_TEMPLATES_ZH = {
    "气虚":   "舌淡白胖大有齿痕，乏力懒言，自汗，活动后气喘",
    "血虚":   "舌淡白瘦薄，面色苍白，头晕眼花，心悸失眠",
    "阴虚":   "舌红少苔，盗汗潮热，五心烦热，口干咽燥",
    "阳虚":   "舌淡白胖嫩，畏寒肢冷，小便清长，大便溏泄",
    "气滞":   "舌淡红或暗，胀痛走窜，情志不舒，胸胁满闷",
    "血瘀":   "舌紫暗有瘀斑，刺痛固定，夜间加重",
    "痰湿":   "舌淡白胖大有齿痕，苔白腻，体形肥胖，胸闷痰多",
    "湿热":   "舌红苔黄腻，口苦黏腻，小便黄，大便黏滞",
    "内风":   "舌红少苔或无苔，舌体颤动或歪斜，头晕目眩，肢体麻木，震颤抽搐",
    "实热":   "舌红绛苔黄燥，口渴喜冷饮，面红目赤，烦躁易怒，大便秘结",
}

SYNDROME_TEXT_TEMPLATES_EN = {
    "qi_deficiency":     "pale swollen tongue with teeth marks, fatigue, weak voice, sweating on exertion",
    "blood_deficiency":  "pale thin tongue, pale complexion, dizziness, palpitations, insomnia",
    "yin_deficiency":    "red tongue with little coating, night sweats, hot palms and soles, dry mouth",
    "yang_deficiency":   "pale swollen tongue, cold limbs, clear urine, loose stools",
    "qi_stagnation":     "dark or normal tongue, distending pain, emotional distress, chest oppression",
    "blood_stasis":      "purple dark tongue with petechiae, fixed sharp pain, worse at night",
    "phlegm_dampness":   "pale swollen tongue with white greasy coating, obesity, chest oppression, phlegm",
    "damp_heat":         "red tongue with yellow greasy coating, bitter sticky mouth, yellow urine, sticky stools",
    "internal_wind":     "red tongue with little or no coating, trembling or deviated tongue, dizziness, limb numbness, tremor or convulsion",
    "excess_heat":       "deep red tongue with yellow dry coating, thirst for cold drinks, flushed face, irritability, constipation",
}


def build_zero_shot_prompts(syndrome_text_templates: Dict[str, str]) -> Dict[str, List[str]]:
    """
    为 zero-shot baseline 构建 prompt ensemble（医学领域常用）。

    Args:
        syndrome_text_templates: {证型: 描述}

    Returns:
        {证型: [prompt1, prompt2, ...]}，每个证型 3 条模板
    """
    templates = [
        "a photo of a tongue of a patient with {d}",
        "tongue image showing {d}",
        "this tongue indicates {d}",
    ]
    prompts: Dict[str, List[str]] = {}
    for key, desc in syndrome_text_templates.items():
        prompts[key] = [t.format(d=desc) for t in templates]
    return prompts


# ---------- 单元测试 ----------

if __name__ == "__main__":
    if torch is None:
        print("⚠️  torch 未安装，本机跳过加载测试（仅验证模板完整性）。")
    else:
        print("BiomedCLIPWrapper 加载测试：")
        wrapper = BiomedCLIPWrapper()  # lazy_load，不会下载
        print(f"  模型 ID: {wrapper.model_name}")
        print(f"  text_dim={wrapper.text_dim}, vision_dim={wrapper.vision_dim}")
        print(f"  是否已加载: {wrapper.is_loaded}")

        # 验证 zero-shot prompt 构建
        prompts = build_zero_shot_prompts(SYNDROME_TEXT_TEMPLATES_EN)
        print(f"\n  zero-shot prompt 模板数: {len(prompts)} 个证型")
        for k in list(prompts)[:2]:
            print(f"    {k}: {prompts[k][0][:60]}...")

        # 验证 10 病性模板齐全
        assert len(SYNDROME_TEXT_TEMPLATES_ZH) == 10, "ZH 模板应为 10 病性"
        assert len(SYNDROME_TEXT_TEMPLATES_EN) == 10, "EN 模板应为 10 病性"
        print("\n✓ 10 病性模板齐全（内风 / 实热 已补齐）")
