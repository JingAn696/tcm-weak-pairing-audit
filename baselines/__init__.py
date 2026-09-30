"""
Baselines 包
============

包含论文里 4 个 baseline：
1. biomedclip_zero_shot  — BiomedCLIP 零样本（无需训练）
2. biomedclip_lora       — BiomedCLIP + LoRA 微调（计划中）
3. ydyolo                — 舌象单模态 baseline（计划中）
4. tcm_sd_text           — TCM-SD 文本-only baseline（计划中）

所有 baseline 都用 10 病性多标签场景，对齐 papers/code/eval/metrics.py 的
compute_metrics 输出格式（subset_accuracy / macro_f1 / micro_f1 / auc_ovr 等）。
"""