"""
环境冒烟测试（Smoke Test）
=========================

在 AutoDL 实例上跑通全链路前的分层验证脚本。
每步独立 try-except，失败不中断，最后汇总哪几步通过。

用法（在 papers/code 目录下）：
    python smoke_test.py

验证层次：
    Step 0 环境检查        → torch / CUDA / GPU
    Step 1 数据管道        → CSV 能读、行数对
    Step 2 方法模块 import → soft_cl_loss / kg_embedding / cross_attention_fusion
    Step 3 SoftCL 数值冒烟 → dummy 数据算 loss
    Step 4 KG 三元组构建    → 节点数 / 三元组数
    Step 5 BiomedCLIP 包装器 → 10 模板 + prompt + wrapper 结构
    Step 6 完整模型集成    → MultimodalSyndromeClassifier 实例化
"""

import sys
from pathlib import Path

# 保证能从 papers/code 目录 import data/ 和 models/
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = []


def check(name: str, fn):
    """执行一步检查，记录结果，异常不中断。"""
    try:
        msg = fn()
        RESULTS.append((name, True, msg))
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))


def main():
    print("=" * 70)
    print("  环境冒烟测试 Smoke Test")
    print("=" * 70)

    # ---------- Step 0 环境检查 ----------
    def step0():
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        return (f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
                f"GPU可用={cuda_ok} | 型号={gpu} | Python {sys.version.split()[0]}")

    check("Step 0 环境检查", step0)

    # ---------- Step 1 数据管道 ----------
    def step1():
        import pandas as pd
        from pathlib import Path
        data_dir = Path(__file__).resolve().parent / "data"
        splits = data_dir / "splits"
        for name in ["train.csv", "val.csv", "test.csv"]:
            p = splits / name
            if not p.exists():
                return f"✗ 缺 {name}"
        df = pd.read_csv(splits / "train.csv", nrows=5)
        n_cols = len(df.columns)
        return f"train/val/test 存在，train 表头 {n_cols} 列，前 5 列：{list(df.columns[:5])}"

    check("Step 1 数据管道", step1)

    # ---------- Step 2 方法模块 import ----------
    def step2():
        from models.soft_cl_loss import SoftContrastiveLoss, default_syndrome_similarity_matrix
        from models.kg_embedding import ALL_NODES, build_triples_from_syndromes
        from models.cross_attention_fusion import CrossModalFusion
        m = default_syndrome_similarity_matrix()
        return f"3 模块 import 成功，SoftCL 矩阵 {tuple(m.shape)}，KG 节点 {len(ALL_NODES)}"

    check("Step 2 方法模块 import", step2)

    # ---------- Step 3 SoftCL 数值冒烟 ----------
    def step3():
        import torch
        from models.soft_cl_loss import SoftContrastiveLoss
        criterion = SoftContrastiveLoss()
        features = torch.randn(20, 512)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                               5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        loss = criterion(features, labels)
        return f"SoftCL loss = {loss.item():.4f}（应是一个正数）"

    check("Step 3 SoftCL 数值冒烟", step3)

    # ---------- Step 4 KG 三元组构建 ----------
    def step4():
        from pathlib import Path
        from models.kg_embedding import build_triples_from_syndromes
        code_dir = Path(__file__).resolve().parent
        triples = build_triples_from_syndromes(code_dir)
        return f"构造 {len(triples)} 条三元组"

    check("Step 4 KG 三元组构建", step4)

    # ---------- Step 5 BiomedCLIP 包装器结构检查 ----------
    def step5():
        from models.biomedclip_wrapper import (
            BiomedCLIPWrapper,
            SYNDROME_TEXT_TEMPLATES_ZH,
            SYNDROME_TEXT_TEMPLATES_EN,
            build_zero_shot_prompts,
        )
        assert len(SYNDROME_TEXT_TEMPLATES_ZH) == 10, "ZH 模板应为 10 病性"
        assert len(SYNDROME_TEXT_TEMPLATES_EN) == 10, "EN 模板应为 10 病性"
        prompts = build_zero_shot_prompts(SYNDROME_TEXT_TEMPLATES_EN)
        wrapper = BiomedCLIPWrapper()  # lazy_load，不触发下载
        return (f"10 病性模板齐全，zero-shot prompt {len(prompts)} 组，"
                f"wrapper 结构 OK（text_dim={wrapper.text_dim}，未加载模型）")

    check("Step 5 BiomedCLIP 包装器", step5)

    # ---------- Step 6 完整模型集成 ----------
    def step6():
        import torch
        from models.full_model import MultimodalSyndromeClassifier
        model = MultimodalSyndromeClassifier(
            biomedclip_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            kg_vocab_size=50,
            n_classes=10,
            freeze_backbones=True,
        )
        n = sum(p.numel() for p in model.parameters())
        n_classes = model.n_classes
        return (f"完整模型实例化成功，可训参数 {n:,}（BiomedCLIP 未加载，lazy），"
                f"分类头 {n_classes} 类")

    check("Step 6 完整模型集成", step6)

    # ---------- 汇总 ----------
    print()
    print("-" * 70)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, msg in RESULTS:
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name:28s} {msg}")
    print("-" * 70)
    print(f"  结果：{n_pass}/{len(RESULTS)} 步通过")
    if n_pass < 6:
        print("  ⚠️  Step 0-6 应至少全部通过。若有失败，把输出贴给 Buddy。")
    else:
        print("  ✅  基础链路就绪，可以开始实现/运行 baseline。")
    print("=" * 70)


if __name__ == "__main__":
    main()
