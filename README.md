# W4 代码仓库总览

## 目录结构

```
papers/code/
├── README.md                          # 本文件
├── w4-roadmap.md                      # W4 阶段路线图
│
├── data/                              # 数据相关
│   ├── tcm-sd/                        # TCM-SD 数据集（Phase 2 下载）
│   ├── splits/                        # train/val/test 划分
│   ├── inquiries.csv                  # 8 个 JSON 转换后的结构化问诊表
│   ├── DOWNLOAD-GUIDE.md              # 数据集下载指南
│   ├── DATA-STRUCTURE.md              # 字段结构文档
│   └── inquiries_converter.py         # 8 JSON → CSV 转换脚本
│
├── models/                            # 模型核心
│   ├── biomedclip_wrapper.py          # BiomedCLIP 加载 + tokenizer 封装
│   ├── cross_attention_fusion.py      # 视觉-文本跨模态融合
│   ├── kg_embedding.py                # KG 嵌入模块（PyKEEN TransE）
│   ├── soft_cl_loss.py                # Soft Contrastive Loss
│   └── full_model.py                  # 完整模型（KG + SoftCL + Cross-Attn）
│
├── baselines/                         # 4 个 baseline 实现
│   ├── zero_shot_biomedclip.py        # Baseline 1：BiomedCLIP zero-shot
│   ├── lora_biomedclip.py             # Baseline 2：BiomedCLIP + LoRA
│   ├── ydyolo_tongue.py               # Baseline 3：YDYOLO 类（单模态 SOTA）
│   └── tcm_sd_text.py                 # Baseline 4：TCM-SD 文本-only baseline
│
├── training/                          # 训练 pipeline
│   ├── train.py                       # 统一训练入口
│   ├── config.py                      # 超参数配置
│   └── utils.py                       # 训练工具（日志、checkpoint）
│
└── eval/                              # 评估
    ├── metrics.py                     # Accuracy / Macro-F1 / AUC / 混淆矩阵
    └── gradcam.py                     # Grad-CAM 可解释性可视化
```

## 代码风格规范

- **类型标注**：所有 public 函数必须有 type hints
- **文档字符串**：Google 风格
- **日志**：使用 `logging` 模块，不 print
- **配置**：所有超参数放 `training/config.py`，不在代码里硬编码
- **可复现**：随机种子固定（`torch.manual_seed(42)`）

---

## 依赖（Phase 2 时安装）

```bash
pip install torch torchvision transformers open_clip_torch peft
pip install pykeen pandas scikit-learn matplotlib seaborn
pip install wandb  # 实验追踪
```

本机不需要全部安装——只装写代码用的（typing / pytest / black）。
