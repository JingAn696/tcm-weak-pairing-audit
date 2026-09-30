# -*- coding: utf-8 -*-
"""
Stage B · B-2 端到端冒烟测试
=============================

用**假 MacBERT（StubBert）+ 假原型文件**把 B-2 的完整链路跑一遍，
不需要数据集 / 不需要下载模型，10~20 秒完成。

覆盖
----
[A] Stage A（placeholder）路径回归 —— 确认没被 B-2 改动破坏
[B] B-2（proto_attn）主路径：形状 / 注意力归一化 / mask 生效 / 原型冻结 /
    state_dict 往返 / 损失可反传
[C] `--no_proto_sim`（vis_dim=512）变体
[D] `--proto_min_support 0` 变体（验证"支撑数 + 零向量"双重保护）
[E] 错误路径：缺 proto_path / 文件不存在 时的可读报错

设备（2026-09-14 补）
----
默认 `--device auto`：**有 CUDA 就用 GPU**。在 AutoDL 上跑会真实走
`.to("cuda")` + 反向传播，等于把训练脚本的设备路径也验证了；
本机无 GPU 时自动退回 CPU，行为与旧版一致。

用法
----
    python smoke_test_b2.py                     # 自动选设备（AutoDL 上=GPU）
    python smoke_test_b2.py --device cpu        # 强制 CPU
    python -m models.syndrome_proto_attention   # 模块级自测（9 条断言）
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.full_model_v2 import MultimodalSyndromeClassifierV2
from models.kg_keywords import ALL_NODES
from models.tongue_label_mapping import SYNDROMES_ZH


# ------------------------------------------------------------
# 设备（与训练脚本同一套逻辑：auto = 有 CUDA 就用 GPU）
# ------------------------------------------------------------
_ap = argparse.ArgumentParser(description="Stage B · B-2 端到端冒烟测试")
_ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                 help="auto=有 CUDA 就用 GPU（默认）；cpu=强制 CPU")
_args = _ap.parse_args()

if _args.device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = _args.device
if DEVICE == "cuda" and not torch.cuda.is_available():
    raise SystemExit("✗ --device cuda 但 torch.cuda.is_available()=False（未挂载 GPU）")


class StubBert(nn.Module):
    """冒充 MacBERT：有 config.hidden_size，forward 返回 last_hidden_state。"""

    def __init__(self, hidden: int = 768):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden)
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None):
        B, L = input_ids.shape
        # 必须跟随输入设备，否则在 GPU 上会 device mismatch
        h = torch.randn(B, L, self.config.hidden_size,
                        device=input_ids.device, dtype=self.dummy.dtype) + self.dummy
        return types.SimpleNamespace(last_hidden_state=h)


proto_file = Path("runs/_tmp_proto.pt")
proto_file.parent.mkdir(parents=True, exist_ok=True)
P = F.normalize(torch.randn(10, 512), dim=1)
P[8] = 0.0  # 内风：真实数据里无可用来源 → 原型是零向量（L2 后仍为 0）
torch.save(
    {
        "prototypes": P,
        "syndromes": list(SYNDROMES_ZH),
        "meta": {
            "n_images": [7174, 580, 4228, 5743, 5, 231, 7978, 3214, 0, 3874],
            "pre_norm": [3.17, 6.97, 3.46, 4.48, 6.58, 7.05, 2.92, 2.51, 0.0, 3.90],
            "strategy": "centered",
        },
    },
    proto_file,
)

ids = torch.randint(0, 100, (3, 16), device=DEVICE)
am = torch.ones_like(ids)
mh = torch.zeros(3, len(ALL_NODES), device=DEVICE)
mh[:, 0] = 1.0
labels = torch.zeros(3, 10, device=DEVICE)
labels[:, 1] = 1.0

print("=" * 70)
print("B-2 集成自测")
if DEVICE == "cuda":
    print(f"  device=cuda ({torch.cuda.get_device_name(0)})")
else:
    print("  device=cpu（注意：这只验证代码逻辑，不验证 GPU 路径）")
print("=" * 70)

# ---------- A) Stage A 路径（回归：不能破坏原行为） ----------
mA = MultimodalSyndromeClassifierV2(
    text_encoder=StubBert(), visual_mode="placeholder").to(DEVICE)
outA = mA(ids, am, mh, return_features=True)
print(f"[A] placeholder 前向：logits={tuple(outA[0].shape)} fused={tuple(outA[1].shape)}")
assert outA[0].shape == (3, 10) and outA[1].shape == (3, 3 * 256)
lossA, partsA = mA.compute_loss(outA[0], outA[1], labels)
lossA.backward()
print(f"[A] 损失可反传：total={partsA['total']:.4f}  ✓")

# ---------- B) B-2 路径（含 sim） ----------
mB = MultimodalSyndromeClassifierV2(
    text_encoder=StubBert(), visual_mode="proto_attn", proto_path=str(proto_file)
).to(DEVICE)
logits, fused, attn, sim = mB(ids, am, mh, return_features=True, return_attn=True)
print(f"[B] proto_attn 前向：logits={tuple(logits.shape)} "
      f"fused={tuple(fused.shape)} attn={tuple(attn.shape)} sim={tuple(sim.shape)}")
assert logits.shape == (3, 10)
assert fused.shape == (3, 3 * 256)
assert attn.shape == (3, 10) and sim.shape == (3, 10)

sums = attn.sum(dim=1)
print(f"[B] attn 行和 = {sums.tolist()}（应全 1）")
assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
masked = [4, 8]  # 气滞(5 图) / 内风(0 图)
print(f"[B] mask 位（气滞/内风）attn = {attn[:, masked].max().item():.3e}（应≈0）")
assert attn[:, masked].max().item() < 1e-6
print(f"[B] mask 位 sim = {sim[:, masked].abs().max().item():.3e}（应=0）")
assert sim[:, masked].abs().max().item() < 1e-9

# 有效原型 mask 正确性
print(f"[B] proto_valid = {mB.proto_attn.proto_valid.tolist()}")
assert not bool(mB.proto_attn.proto_valid[4]) and not bool(mB.proto_attn.proto_valid[8])
assert int(mB.proto_attn.proto_valid.sum()) == 8

# 原型必须冻结
n_train_proto = sum(p.numel() for p in mB.proto_attn.parameters() if p.requires_grad)
print(f"[B] proto_attn 可训练参数 = {n_train_proto:,}（原型不参与，应为 394,753）")
assert n_train_proto == 768 * 512 + 512 + 2 * 512 + 1

lossB, partsB = mB.compute_loss(logits, fused, labels)
lossB.backward()
print(f"[B] 损失可反传：total={partsB['total']:.4f}  ✓")
g = mB.proto_attn.query_proj[0].weight.grad
print(f"[B] query_proj 梯度范数 = {g.norm().item():.4f}（应>0）")
assert g is not None and g.norm().item() > 0
print(f"[B] 原型的 grad = {mB.proto_attn.prototypes.grad}（应 None）")
assert mB.proto_attn.prototypes.grad is None

# state_dict 可存取（含原型 buffer）
sd = mB.state_dict()
print(f"[B] state_dict 含原型 buffer: {'proto_attn.prototypes' in sd}")
assert "proto_attn.prototypes" in sd
mB.load_state_dict(sd)
print("[B] state_dict 往返 ✓")

stats = mB.param_stats()
print(f"[B] param_stats：proto_out_dim={stats['proto_out_dim']} "
      f"proto_attn={stats['proto_attn']:,} trainable={stats['trainable']:,}")

# ---------- C) 无 sim 变体（vis_dim=512） ----------
mC = MultimodalSyndromeClassifierV2(
    text_encoder=StubBert(), visual_mode="proto_attn",
    proto_path=str(proto_file), proto_include_sim=False,
).to(DEVICE)
outC = mC(ids, am, mh, return_features=True)
print(f"[C] --no_proto_sim：logits={tuple(outC[0].shape)}（应 (3,10)）")
assert outC[0].shape == (3, 10)

# ---------- D) min_support=0 → 气滞/内风不再被 mask（只有内风仍为 0 向量被 mask） ----------
mD = MultimodalSyndromeClassifierV2(
    text_encoder=StubBert(), visual_mode="proto_attn",
    proto_path=str(proto_file), proto_min_support=0.0,
).to(DEVICE)
print(f"[D] min_support=0：proto_valid={mD.proto_attn.proto_valid.tolist()}"
      f"（气滞 5 图应转为 True，内风 0 图仍 False）")
assert bool(mD.proto_attn.proto_valid[4]) and not bool(mD.proto_attn.proto_valid[8])

# ---------- E) 错误路径：缺 proto_path 应给出可读报错 ----------
try:
    MultimodalSyndromeClassifierV2(text_encoder=StubBert(), visual_mode="proto_attn")
    raise SystemExit("✗ 缺 proto_path 竟然没报错")
except ValueError as e:
    print(f"[E] 缺 proto_path 正确报错：{e}")

try:
    MultimodalSyndromeClassifierV2(
        text_encoder=StubBert(), visual_mode="proto_attn", proto_path="runs/不存在.pt")
    raise SystemExit("✗ 文件不存在竟然没报错")
except FileNotFoundError as e:
    print(f"[E] 文件不存在正确报错（含修复指引）✓")

proto_file.unlink(missing_ok=True)
print("\n集成自测通过 ✓")
print("=" * 70)
