"""
Grad-CAM 可视化模块（论文可解释性的关键证据）
============================================

Grad-CAM（Gradient-weighted Class Activation Mapping, Selvaraju et al. 2017）让模型
"说出为什么这样分类"——对一张输入图像，输出每个空间位置的"重要性热力图"。

对路线 C 论文 §3.4 的意义：

    1) 中医舌诊的临床意义：
       模型预测"血瘀"时，应该重点关注舌尖/舌两侧（血瘀在中医辨证里常伴舌下静脉迂曲）。
       如果模型聚焦在舌根部、苔色上，那就是"它学错了"——这是 Discussion 的灵魂。

    2) 与中医舌面分区的桥接：
       Grad-CAM 输出 7x7 热力图后可放大到 224x224，与中医舌面 4 区
       （舌尖/舌中/舌根/舌两侧）映射，提供
       "模型关注舌尖+舌侧 → 与血瘀证（舌下络脉迂曲在舌下/舌侧）一致" 的解释。

    3) 论文 Figure X 准备数据：
       多张测试样本的 Grad-CAM 叠加到原图，做成 4 行 3 列的案例分析图。

实现方式（采用 torch hooks，避免修改 BiomedCLIP 源码）：

    • 选 target layer：BiomedCLIP 视觉编码器的最后 norm 前的层（视觉主干最后一层）
    • 前向：保存该层激活特征图
    • 反向：保存该层梯度
    • 计算权重 = 梯度在通道维度的均值
    • 热力图 = ReLU(sum(α_k * A_k))

限制：
    • 必须在已加载 BiomedCLIP 模型的环境下运行（torch ≥ 2.0）
    • 默认 target layer 是 BiomedCLIP ViT-B/16 的最终 norm 前层；
      若换 backbone 需要重新指定。

参考：
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError:
    np = None
    torch = None
    F = None

try:
    from PIL import Image
except ImportError:
    Image = None


# 舌头 4 区的归一化坐标（基于 224x224 输入）
# 与 Table 1 / 9 维 + 4 区对照表的舌面分区一致
TONGUE_REGIONS = {
    "舌尖": (0.0, 0.0, 1.0, 0.30),     # 上 30%
    "舌中": (0.0, 0.30, 1.0, 0.70),    # 中 40%
    "舌根": (0.0, 0.70, 1.0, 1.0),     # 下 30%
    "舌两侧": (0.0, 0.0, 0.20, 1.0),   # 左 20%（实际使用时左右对称应用）
}


class GradCAM:
    """
    Grad-CAM 包装器。target_layer 通常是视觉编码器的最后一个 norm 层。

    使用方式：

        cam = GradCAM(model.vision_encoder, target_layer)
        cam.register(image_tensor)
        heatmap = cam.generate(class_idx=target_label)
        # heatmap: (H, W) numpy, 0-1, 可与原图叠加
    """

    def __init__(self, model: "torch.nn.Module", target_layer: "torch.nn.Module"):
        if torch is None:
            raise ImportError("GradCAM 需要 torch 环境。Phase 2 上 GPU 时可用。")
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []

    def register(self) -> None:
        """注册 forward + backward hook。"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove(self) -> None:
        """清除所有 hook。"""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def generate(
        self,
        class_idx: int,
        input_tensor: "torch.Tensor",
        image_size: Tuple[int, int] = (224, 224),
    ) -> "np.ndarray":
        """
        生成一张图像对应某类别索引的 Grad-CAM 热力图。

        参数
        ----
        class_idx : int
            要解释的类别索引（这里是 10 病性中的某一个）。
        input_tensor : (B, C, H, W) torch.Tensor
            图像输入张量。
        image_size : (H, W)
            输出热力图的目标尺寸。

        返回
        ----
        heatmap : (H, W) numpy 数组，值域 0~1。
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # 前向传播
        logits = self.model(input_tensor)

        # 反向传播：对目标类别回传梯度
        # 多标签二分类：用 sigmoid(logits) 的某个分位
        if logits.dim() == 2:
            scalar = logits[0, class_idx].sigmoid()
        else:
            scalar = logits[0, class_idx]
        scalar.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("未注册 hook 或未触发反向。")

        # 通道权重：梯度在空间维度的均值
        weights = self.gradients.mean(dim=(-1, -2), keepdim=True)  # (B, C, 1, 1)

        # 加权激活：sum(α * A)，再 ReLU
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 上采样到目标尺寸
        cam = F.interpolate(
            cam,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # 归一化到 0~1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def generate_with_breakdown(
        self,
        class_idx: int,
        input_tensor: "torch.Tensor",
        image_size: Tuple[int, int] = (224, 224),
    ) -> dict:
        """
        生成热力图 + 按 4 区聚合的注意力强度（用于论文案例分析表格）。

        返回
        ----
        {
            "heatmap": (H, W) ndarray,
            "region_attention": {"舌尖": float, "舌中": float, "舌根": float, "舌两侧": float},
            "max_region": str,  # 最受关注的舌区
        }
        """
        heatmap = self.generate(class_idx, input_tensor, image_size)
        h, w = heatmap.shape

        region_scores = {}
        for region_name, (x1, y1, x2, y2) in TONGUE_REGIONS.items():
            px1 = int(x1 * w)
            px2 = int(x2 * w) if region_name != "舌两侧" else int(x2 * w)  # noqa
            py1 = int(y1 * h)
            py2 = int(y2 * h)
            sub = heatmap[py1:py2, px1:px2]
            region_scores[region_name] = float(sub.mean())

        max_region = max(region_scores, key=region_scores.get)

        return {
            "heatmap": heatmap,
            "region_attention": region_scores,
            "max_region": max_region,
        }


# === 视觉化叠加工具（论文 Figure 用） ===

def overlay_heatmap_on_image(
    image_path: str,
    heatmap: "np.ndarray",
    alpha: float = 0.4,
    output_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    将 Grad-CAM 热力图叠加在原图上（论文 Figure 用）。

    参数
    ----
    image_path : str
        原图路径。
    heatmap : (H, W) ndarray, 0~1
        Grad-CAM 热力图。
    alpha : float
        热力图透明度（0=全原图，1=全热力图）。
    output_path : str, optional
        输出 PNG 路径，None 则返回 PIL Image 不保存。

    返回
    ----
    PIL.Image or None
    """
    if Image is None:
        raise ImportError("Pillow 未安装。pip install pillow")

    from PIL import Image as PILImage

    img = PILImage.open(image_path).convert("RGB")
    img = img.resize((heatmap.shape[1], heatmap.shape[0]))
    img_arr = np.asarray(img, dtype=np.float32) / 255.0

    # jet colormap 简易实现
    heatmap_color = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    heatmap_color[..., 0] = np.clip(2 * heatmap, 0, 1)               # R
    heatmap_color[..., 1] = np.clip(2 * (heatmap - 0.5), 0, 1)       # G
    heatmap_color[..., 2] = np.clip(2 * (1 - heatmap), 0, 1)         # B

    overlay = (1 - alpha) * img_arr + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 1)
    overlay_img = PILImage.fromarray((overlay * 255).astype(np.uint8))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        overlay_img.save(output_path)
        return None
    return overlay_img


# === 论文 §3.4 案例分析报告生成器 ===

def generate_case_report(
    sample_id: str,
    syndrome_zh: str,
    region_attention: dict,
    output_path: Optional[str] = None,
) -> str:
    """
    为 Grad-CAM 案例生成中医解读文本（论文 §4.X 用）。

    参数
    ----
    sample_id : str
        样本 ID（如 "user_123"）。
    syndrome_zh : str
        中医证名（如 "血瘀"）。
    region_attention : dict
        4 区注意力强度。
    output_path : str, optional
        若指定，追加写入指定文件（论文准备好的报告里）。

    返回
    ----
    report : str
        一段可直接放进 Discussion 章节的中医解读。
    """
    max_region = max(region_attention, key=region_attention.get)
    sorted_regions = sorted(region_attention.items(), key=lambda x: -x[1])
    top2 = ", ".join(f"{r[0]}({r[1]:.2f})" for r in sorted_regions[:2])

    # 中医对应表（与 Table 1 的 4 区对照一致）
    region_clinical_mapping = {
        "血瘀": "舌下络脉迂曲（舌下/舌侧）",
        "气虚": "舌淡胖、齿痕（舌中/舌侧）",
        "阳虚": "舌淡胖、苔白滑（舌中/舌尖）",
        "阴虚": "舌红少苔（舌面）",
        "气滞": "舌色或暗（舌两侧）",
        "痰湿": "苔腻（舌中）",
        "湿热": "苔黄腻（舌中/舌根）",
        "内风": "舌颤（舌面/舌尖）",
        "实热": "舌红苔黄（舌面）",
        "血虚": "舌淡（舌面）",
    }
    expected_region = region_clinical_mapping.get(syndrome_zh, "舌面")

    report = (
        f"样本 {sample_id}，模型预测{syndrome_zh}证，"
        f"Grad-CAM 显示模型关注集中在{max_region}区（{top2}）。\n"
        f"中医临床预期：{syndrome_zh}证典型舌象出现在{expected_region}区。"
        f"\n\n一致性评估：{'**高度一致**' if expected_region in region_attention and region_attention[max_region] > 0.5 else '**部分一致**，可能存在其他证型兼夹'}\n"
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(report)
            f.write("\n" + "=" * 60 + "\n\n")

    return report


# === 单元测试（无 torch 时优雅降级） ===
if __name__ == "__main__":
    if torch is None:
        print("⚠️  torch 未安装。Grad-CAM 需要 torch 环境。")
        print("   pip install torch torchvision")
    else:
        print("GradCAM 模块结构验证：")
        print(f"  TONGUE_REGIONS: {list(TONGUE_REGIONS.keys())}")
        print(f"  区域坐标范围: {TONGUE_REGIONS}")

        # 模拟一个视觉编码器
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.AdaptiveAvgPool2d(7),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 7 * 7, 10),
        )

        # 找到视觉编码器的"最后一层 norm 前"层
        cam = GradCAM(model, target_layer=model[-3])  # AdaptiveAvgPool2d
        print(f"  target_layer: {type(cam.target_layer).__name__}")

        # 模拟一次完整 forward + backward
        img = torch.randn(1, 3, 224, 224)
        with cam:
            out = model(img)
            out[0, 3].backward()

        print("✓ Hook 注册 / 反向 / 激活保存链路通过")
