"""
完整模型 v2 训练脚本（Stage A：文本 + KG + SoftCL）
==================================================

用法（AutoDL）：
    python training/train_full_model.py \
        --text_checkpoint runs/tcm_text_macbert/checkpoint-XXXX

    # checkpoint-XXXX = baseline 1 的 best checkpoint
    # （ls runs/tcm_text_macbert/ 后选 eval-only 用的那个）

Stage B · B-2（视觉分支接舌象原型）：
    python training/train_full_model.py \
        --text_checkpoint runs/tcm_text_macbert/checkpoint-XXXX \
        --visual_mode proto_attn \
        --proto_path runs/tongue_branch_crop/tongue_prototypes.pt \
        --output_dir runs/full_model_v2_b2

消融实验 flag（论文 Table 用）：
    --freeze_text        冻结 MacBERT，只训 fusion + KG（PEFT 对照）
    --softcl_weight 0    去掉 SoftCL
    --no_kg              KG multi-hot 置零（KG 贡献消融）
    --no_proto_sim       视觉表征不拼相似度向量（只用 512 维聚合向量）
    --proto_min_support  原型支撑图数下界（默认 30；调低会把气滞拉进来）
    --seed               随机种子（默认 42）；seed 重复实验：
                         `for s in 42 43 44; do ... --seed $s --output_dir runs/xxx_s$s; done`
                         （论文的"有/无提升"结论必须有 ≥3 seed 支撑）
    --deterministic      更严格复现（关 cudnn 自动调优，较慢）

输出：
    runs/full_model_v2/best_model.pt      最佳 val macro_f1 的权重
    runs/full_model_v2/test_report.json    test 完整多标签报告

预计耗时（4090）：3 epoch × 35,345 样本 ≈ 1-1.5 小时（与 baseline 1 相当）。
"""

from __future__ import annotations

import os

# 国内下载模型必需（setdefault：用户已 export 时不覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import random
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from eval.metrics import compute_metrics, SYNDROME_NAMES_ZH
from models.full_model_v2 import MultimodalSyndromeClassifierV2
from models.kg_keywords import ALL_NODES, match_kg_nodes
from models.syndrome_proto_attention import B2_VERSION, DEFAULT_MIN_SUPPORT
from models.tongue_label_mapping import SYNDROMES_ZH
from transformers import AutoModel, AutoTokenizer
from utils.run_handoff import Handoff

# 运行交接摘要（2026-09-15）：收尾时把关键信息同时打印 + 落盘 handoff_*.txt
H = Handoff("train_full_model")

# 关注的核心软肋：文本基线 F1 最低的病性（Stage B 成败判据）
WATCH_SYNDROME = "血虚"
# baseline 1（纯文本 MacBERT）的血虚 F1，作为 B-2 判据的参照点（论文协议数字）
TEXT_BASELINE_F1 = 0.6122

# 与 data/splits_v2 CSV 实际列名一致（tcm_sd_loader 的 LABEL_HEADERS，10 病性顺序）
LABEL_COLS = [
    "qi_deficiency|气虚", "blood_deficiency|血虚",
    "yin_deficiency|阴虚", "yang_deficiency|阳虚",
    "qi_stagnation|气滞", "blood_stasis|血瘀",
    "phlegm_dampness|痰湿", "damp_heat|湿热",
    "wind|内风", "excess_heat|实热",
]


# ============================================================
# 数据
# ============================================================

class TCMFullDataset(Dataset):
    """文本 + 10 病性多标签 + KG 节点命中。"""

    def __init__(self, csv_path: Path, use_kg: bool = True):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        missing = [c for c in LABEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺 label 列：{missing}")

        for col in ("chief_complaint", "description", "detection"):
            if col in df.columns:
                df[col] = df[col].fillna("")
            else:
                df[col] = ""

        self.texts = (
            df["chief_complaint"] + "。" + df["description"] + "。" + df["detection"]
        ).tolist()
        self.labels = df[LABEL_COLS].values.astype("float32")

        # KG 节点命中（一次性预计算，纯字符串匹配，秒级）
        if use_kg:
            self.node_hits = [match_kg_nodes(t) for t in self.texts]
        else:
            self.node_hits = [[] for _ in self.texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {"text": self.texts[i], "label": self.labels[i], "nodes": self.node_hits[i]}

    def hit_stats(self):
        """命中率统计（训练前打印一次，确认 KG 通道有信号）。"""
        n = len(self.node_hits)
        n_hit = sum(1 for h in self.node_hits if h)
        avg_hits = sum(len(h) for h in self.node_hits) / max(n, 1)
        return {"samples": n, "hit_rate": n_hit / max(n, 1), "avg_nodes_per_sample": avg_hits}


def make_collate(tokenizer, max_length: int):
    node_index = {n: i for i, n in enumerate(ALL_NODES)}

    def collate(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        labels = torch.tensor(np.stack([b["label"] for b in batch]))
        multi_hot = torch.zeros(len(batch), len(ALL_NODES))
        for i, item in enumerate(batch):
            for n in item["nodes"]:
                j = node_index.get(n)
                if j is not None:
                    multi_hot[i, j] = 1.0
        return enc["input_ids"], enc["attention_mask"], multi_hot, labels

    return collate


# ============================================================
# 评估
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool):
    """返回 compute_metrics 的完整 dict。"""
    model.eval()
    probs_all, labels_all = [], []
    for input_ids, attention_mask, multi_hot, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        multi_hot = multi_hot.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(input_ids, attention_mask, multi_hot)
        probs_all.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels_all.append(labels.numpy())

    y_score = np.concatenate(probs_all)
    y_true = np.concatenate(labels_all).astype(int)
    y_pred = (y_score >= 0.5).astype(int)
    return compute_metrics(y_true, y_pred, y_score=y_score)


@torch.no_grad()
def collect_attention(model, loader, device, use_amp: bool):
    """
    B-2 专属诊断：test 集上"文本 → 舌象原型"的平均注意力与平均相似度。

    论文用途：这张表/图直接展示"模型认为每种病性分别对应多强的舌象证据"，
    是弱配对桥接可解释性的核心证据（也是可写进 Discussion 的定性材料）。
    """
    model.eval()
    attn_sum = np.zeros(len(SYNDROMES_ZH))
    sim_sum = np.zeros(len(SYNDROMES_ZH))
    n = 0
    for input_ids, attention_mask, multi_hot, _ in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        multi_hot = multi_hot.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            _, attn, sim = model(input_ids, attention_mask, multi_hot, return_attn=True)
        bs = input_ids.size(0)
        attn_sum += attn.float().sum(dim=0).cpu().numpy()
        sim_sum += sim.float().sum(dim=0).cpu().numpy()
        n += bs
    return {
        "mean_attn": (attn_sum / max(n, 1)).tolist(),
        "mean_sim": (sim_sum / max(n, 1)).tolist(),
        "syndromes": list(SYNDROMES_ZH),
    }


# ============================================================
# 主流程
# ============================================================

def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定随机性，用于 seed 重复实验（论文的方差估计）。

    - random / numpy / torch(CPU+所有 GPU) 三处都设
    - deterministic=True 时额外关掉 cudnn 的算法自动选择（更严格但更慢）
    - 注意：即便 deterministic=True，跨 GPU 型号仍可能有微小差异；
      本实验的 seed 重复意义是"同机型内的运行方差"，不是跨平台位级复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 非 deterministic：保留 cudnn.benchmark 加速，但算法选择仍受 seed 影响
        torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="完整模型 v2（Stage A）训练")
    parser.add_argument("--text_checkpoint", type=str, required=True,
                        help="baseline 1 的 best checkpoint 路径（MacBERT 热启动）")
    parser.add_argument("--text_model_name", type=str, default="hfl/chinese-macbert-base",
                        help="tokenizer 来源（Trainer 的 checkpoint 只存权重不存 tokenizer，"
                             "tokenizer 从原模型 repo 加载，与 eval 脚本一致）")
    parser.add_argument("--data_root", type=str, default=str(_PROJECT_ROOT / "data"))
    parser.add_argument("--splits_dir", type=str, default="splits_v2")
    parser.add_argument("--output_dir", type=str, default=str(_PROJECT_ROOT / "runs" / "full_model_v2"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="新模块（fusion/KG）学习率")
    parser.add_argument("--text_lr", type=float, default=1e-5,
                        help="MacBERT 微调学习率（热启动后用小 lr 防 catastrophic forgetting）")
    parser.add_argument("--softcl_weight", type=float, default=0.5)
    parser.add_argument("--freeze_text", action="store_true", help="冻结 MacBERT（PEFT 消融）")
    parser.add_argument("--no_kg", action="store_true", help="KG 置零（KG 消融）")

    # ---- Stage B · B-2：视觉分支接舌象原型（弱配对桥接） ----
    parser.add_argument("--visual_mode", choices=["placeholder", "proto_attn"],
                        default="placeholder",
                        help="placeholder=Stage A 占位（默认）；proto_attn=B-2 文本查询舌象原型")
    parser.add_argument("--proto_path", type=str, default=None,
                        help="B-1 产出的 tongue_prototypes.pt（visual_mode=proto_attn 时必填）")
    parser.add_argument("--proto_min_support", type=float, default=DEFAULT_MIN_SUPPORT,
                        help="支撑图数低于此值的原型被 mask（气滞=5、内风=0）")
    parser.add_argument("--no_proto_sim", action="store_true",
                        help="视觉表征不拼接逐病性相似度向量（512 维而非 522 维，消融用）")
    parser.add_argument("--select_metric", choices=["macro_f1", "auc"], default="macro_f1",
                        help="best checkpoint 选择指标（默认 macro_f1@0.5，与 Stage A 一致；"
                             "auc 更稳健但会引入协议差异）")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    # ---- seed 重复实验（论文方差估计：单次运行的 ±Δ 不能当结论） ----
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（seed 重复实验用；默认 42，与 Stage A/B-2 首轮一致）")
    parser.add_argument("--deterministic", action="store_true",
                        help="关闭 cudnn 算法自动选择（更严格的可复现，代价是变慢）")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    use_amp = (not args.no_fp16) and torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = "Stage A（视觉占位）" if args.visual_mode == "placeholder" else "Stage B · B-2（文本查询舌象原型）"
    print("=" * 70)
    print(f"完整模型 v2 · {tag}")
    print(f"  device={device} | amp={use_amp} | freeze_text={args.freeze_text} | "
          f"no_kg={args.no_kg} | softcl_weight={args.softcl_weight}")
    print(f"  visual_mode={args.visual_mode} | proto_attn 版本={B2_VERSION} | "
          f"select_metric={args.select_metric}")
    print(f"  seed={args.seed} | deterministic={args.deterministic}")
    print("=" * 70)
    if args.visual_mode == "proto_attn" and not args.proto_path:
        raise SystemExit("✗ --visual_mode proto_attn 必须同时给 --proto_path")

    # ---------- 1. tokenizer + 文本编码器（热启动） ----------
    print(f"\n[1] 加载 MacBERT（热启动自 {args.text_checkpoint}）...")
    # tokenizer：baseline 1 的 Trainer 没把 tokenizer 存进 checkpoint（只存权重），
    # 所以先试 checkpoint、失败则回退到原模型 repo（HF 缓存已有，不会重新下载）
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.text_checkpoint)
        print(f"  ✓ tokenizer 来自 checkpoint")
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
        print(f"  ✓ checkpoint 无 tokenizer，改从 {args.text_model_name} 加载（vocab 一致，安全）")
    text_encoder = AutoModel.from_pretrained(args.text_checkpoint)
    if args.freeze_text:
        for p in text_encoder.parameters():
            p.requires_grad = False
    print(f"  ✓ hidden_size={text_encoder.config.hidden_size}")

    # ---------- 2. 模型 ----------
    model = MultimodalSyndromeClassifierV2(
        text_encoder=text_encoder,
        softcl_weight=args.softcl_weight,
        visual_mode=args.visual_mode,
        proto_path=args.proto_path,
        proto_min_support=args.proto_min_support,
        proto_include_sim=not args.no_proto_sim,
    ).to(device)
    stats = model.param_stats()
    print(f"[2] 模型构建完成：总参数 {stats['total']/1e6:.1f}M，"
          f"可训练 {stats['trainable']/1e6:.1f}M"
          f"（text {stats['text_encoder']/1e6:.1f}M / "
          f"fusion {stats['fusion']/1e6:.2f}M / kg {stats['kg']/1e6:.3f}M / "
          f"proto_attn {stats['proto_attn']/1e6:.3f}M）")
    if model.proto_attn is not None:
        print(f"  视觉分支输出维 = {stats['proto_out_dim']}"
              f"（512 原型聚合 + {stats['proto_out_dim']-512} 相似度）"
              f"，原型冻结（不参与训练）")

    # ---------- 3. 数据 ----------
    splits = Path(args.data_root) / args.splits_dir
    print(f"\n[3] 加载 3 个 split（{splits}）...")
    train_ds = TCMFullDataset(splits / "train.csv", use_kg=not args.no_kg)
    val_ds = TCMFullDataset(splits / "val.csv", use_kg=not args.no_kg)
    test_ds = TCMFullDataset(splits / "test.csv", use_kg=not args.no_kg)
    print(f"  ✓ train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    ts = train_ds.hit_stats()
    print(f"  ✓ KG 命中率：{ts['hit_rate']*100:.1f}% 样本命中 ≥1 节点，"
          f"平均 {ts['avg_nodes_per_sample']:.2f} 节点/样本")
    if ts["hit_rate"] < 0.3:
        print("  ⚠️  命中率偏低——关键词表可能需要净安校准（models/kg_keywords.py）")

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    # ---------- 4. 优化器（参数分组） ----------
    text_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    other_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("text_encoder.") and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": text_params, "lr": args.text_lr},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=0.01,
    )
    n_batches = len(train_loader)
    print(f"\n[4] 优化器：text {sum(p.numel() for p in text_params)/1e6:.1f}M @ {args.text_lr}"
          f" + 其他 {sum(p.numel() for p in other_params)/1e6:.2f}M @ {args.lr}"
          f" | {n_batches} batch/epoch")

    # AMP GradScaler（torch 2.6 API，旧版兜底）
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---------- 5. 训练循环 ----------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_path = output_dir / "best_model.pt"

    print(f"\n[5] 训练开始（epochs={args.epochs}, batch={args.batch_size}）...")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_bce = run_softcl = run_total = 0.0
        n_seen = 0
        for step, (input_ids, attention_mask, multi_hot, labels) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            multi_hot = multi_hot.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, fused = model(
                    input_ids, attention_mask, multi_hot, return_features=True
                )
                loss, parts = model.compute_loss(logits, fused, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            run_bce += parts["bce"] * bs
            run_softcl += parts["softcl"] * bs
            run_total += parts["total"] * bs
            n_seen += bs

            if step % 100 == 0 or step == n_batches:
                print(f"  epoch {epoch} | batch {step}/{n_batches} | "
                      f"bce={run_bce/n_seen:.4f} softcl={run_softcl/n_seen:.4f} "
                      f"total={run_total/n_seen:.4f}")

        # ---- 每 epoch 评估 val ----
        val_metrics = evaluate(model, val_loader, device, use_amp)
        val_f1 = val_metrics["macro_f1"]
        val_score = val_f1 if args.select_metric == "macro_f1" else val_metrics["auc_ovr_macro"]
        print(f"  epoch {epoch} | VAL macro_f1={val_f1:.4f} "
              f"auc={val_metrics['auc_ovr_macro']:.4f} "
              f"micro_f1={val_metrics['micro_f1']:.4f} "
              f"subset_acc={val_metrics['subset_accuracy']:.4f}"
              f"{'' if args.select_metric=='macro_f1' else f' | 选择分={val_score:.4f}'}")

        if val_score > best_val_f1:
            best_val_f1 = val_score
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ 保存 best（val {args.select_metric}={best_val_f1:.4f}）→ {best_path}")

    print(f"\n  训练完成，用时 {(time.time()-t0)/60:.1f} 分钟，"
          f"best val {args.select_metric}={best_val_f1:.4f}")

    # ---------- 6. test 最终评估（加载 best） ----------
    print(f"\n[6] 加载 best 权重，test 最终评估...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, use_amp)

    print(f"\n📊 完整模型 v2 · {tag} test 指标（N={len(test_ds)}）:")
    print(f"  Subset Accuracy : {test_metrics['subset_accuracy']:.4f}")
    print(f"  Macro F1        : {test_metrics['macro_f1']:.4f}")
    print(f"  Micro F1        : {test_metrics['micro_f1']:.4f}")
    print(f"  Hamming Loss    : {test_metrics['hamming_loss']:.4f}")
    print(f"  AUC OvR Macro   : {test_metrics['auc_ovr_macro']:.4f}")
    watch_f1 = test_metrics.get(f"f1_{WATCH_SYNDROME}")
    if watch_f1 is not None:
        d_watch = watch_f1 - TEXT_BASELINE_F1
        print(f"\n★ 关注指标 · {WATCH_SYNDROME} F1 = {watch_f1:.4f}"
              f"（文本基线 {TEXT_BASELINE_F1:.4f}，Δ = {d_watch:+.4f}）")
        if d_watch > 0.02:
            print("  → 视觉分支带来了实质提升 ✓")
        elif d_watch >= -0.02:
            print("  → 与文本基线持平（|Δ| < 0.02，落在单次运行的噪声范围内；"
                  "要下结论需 ≥3 seed）")
        else:
            print("  → 低于文本基线（视觉分支负贡献）")

    print(f"\nPer-class F1:")
    for name, f1 in test_metrics.items():
        if name.startswith("f1_"):
            syn_zh = name[3:]
            bar = "█" * int(f1 * 30)
            mark = "  ← 关注" if syn_zh == WATCH_SYNDROME else ""
            print(f"  {syn_zh:6s}: {f1:.4f} {bar}{mark}")

    # ---------- 6.5 B-2 专属：文本→舌象原型注意力（论文可视化素材） ----------
    attn_report = None
    if model.proto_attn is not None:
        attn_report = collect_attention(model, test_loader, device, use_amp)
        print(f"\n🔍 文本 → 舌象原型 注意力（test 集平均，N={len(test_ds)}）")
        print(f"  {'病性':<6s} {'平均注意力':>10s} {'平均相似度':>10s} {'支撑图数':>10s}")
        print("  " + "-" * 42)
        for j, s in enumerate(SYNDROMES_ZH):
            sup = float(model.proto_attn.proto_support[j].item())
            flag = "  (mask)" if not bool(model.proto_attn.proto_valid[j]) else ""
            bar = "█" * int(attn_report["mean_attn"][j] * 200)
            print(f"  {s:<6s} {attn_report['mean_attn'][j]:10.4f} "
                  f"{attn_report['mean_sim'][j]:10.4f} {sup:10.0f}  {bar}{flag}")

    # ---------- 7. 保存报告 ----------
    report = {
        "model": ("full_model_v2_stage_b2" if args.visual_mode == "proto_attn"
                  else "full_model_v2_stage_a"),
        "b2_version": B2_VERSION,
        "config": {
            "text_checkpoint": args.text_checkpoint,
            "splits_dir": args.splits_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "text_lr": args.text_lr,
            "softcl_weight": args.softcl_weight,
            "freeze_text": args.freeze_text,
            "no_kg": args.no_kg,
            "max_length": args.max_length,
            "visual_mode": args.visual_mode,
            "proto_path": args.proto_path,
            "proto_min_support": args.proto_min_support,
            "proto_include_sim": not args.no_proto_sim,
            "select_metric": args.select_metric,
            "seed": args.seed,
            "deterministic": args.deterministic,
        },
        "proto_summary": model.proto_summary or None,
        "attention": attn_report,
        "watch_syndrome": {"name": WATCH_SYNDROME,
                           "f1": test_metrics.get(f"f1_{WATCH_SYNDROME}"),
                           "text_baseline_f1": TEXT_BASELINE_F1},
        "best_val_score": best_val_f1,
        "test": test_metrics,
    }
    report_path = output_dir / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存：{report_path}")
    print("=" * 70)

    # ---------- 交接摘要（专供复制给 Buddy；与上面控制台结论同源） ----------
    H.set_out_dir(output_dir)
    H.section("配置")
    for _k in ("visual_mode", "freeze_text", "no_kg", "epochs", "batch_size",
               "lr", "text_lr", "softcl_weight", "seed", "deterministic",
               "proto_path", "proto_min_support", "select_metric"):
        H.kv(_k, getattr(args, _k, "-"))
    H.kv("B2_VERSION", B2_VERSION)
    H.kv("best_val_score", f"{best_val_f1:.4f}")

    H.section("test 指标")
    H.kv("subset_accuracy", f"{test_metrics['subset_accuracy']:.4f}")
    H.kv("macro_f1", f"{test_metrics['macro_f1']:.4f}")
    H.kv("micro_f1", f"{test_metrics['micro_f1']:.4f}")
    H.kv("hamming_loss", f"{test_metrics['hamming_loss']:.4f}")
    H.kv("auc_ovr_macro", f"{test_metrics['auc_ovr_macro']:.4f}")

    if watch_f1 is not None:
        _d = watch_f1 - TEXT_BASELINE_F1
        H.section(f"关注病性 · {WATCH_SYNDROME}")
        H.kv("F1", f"{watch_f1:.4f}")
        H.kv("文本基线 F1", f"{TEXT_BASELINE_F1:.4f}")
        H.kv("Δ(相对文本基线)", f"{_d:+.4f}")
        if abs(_d) < 0.019:
            H.warn(f"|Δ|={abs(_d):.4f} < 单次运行噪声下界 0.019 → 论文口径只能写"
                   f"“未观察到提升”，不能写“显著下降/提升”")
        elif _d > 0:
            H.line(f"Δ 超过噪声下界，需 ≥3 seed 重复确认后才可下结论")

    H.section("逐类 F1")
    H.kv_dict({k[3:]: round(float(v), 4)
               for k, v in test_metrics.items() if k.startswith("f1_")})

    if attn_report is not None:
        H.section("文本→舌象原型 注意力（test 平均）")
        H.table(["病性", "attn", "sim", "support"],
                [[SYNDROMES_ZH[j],
                  f"{attn_report['mean_attn'][j]:.4f}",
                  f"{attn_report['mean_sim'][j]:.4f}",
                  int(model.proto_attn.proto_support[j].item())]
                 for j in range(len(SYNDROMES_ZH))],
                widths=[8, 10, 10, 9])
        _masked = [SYNDROMES_ZH[j] for j in range(len(SYNDROMES_ZH))
                   if not bool(model.proto_attn.proto_valid[j])]
        if _masked:
            H.kv("被 mask 的病性", ", ".join(_masked))

    H.section("产物")
    H.artifact(report_path)
    H.artifact(best_path)


if __name__ == "__main__":
    raise SystemExit(H.run(main))
