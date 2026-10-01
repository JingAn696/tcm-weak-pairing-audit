"""
区域级弱监督 trainer 的**服务器端全链路冒烟**（GPU）
======================================================

**为什么需要它**

本机（无 GPU / 无 open_clip / 无 HF 缓存）只能验证：语法、纯数值逻辑、
stub 模型结构、真实标注解析。以下这些**只有上服务器才知道**：

  · open_clip 是否可用、BiomedCLIP 权重能否从镜像拉到
  · 区域 DataLoader 吐出的 batch 形状/dtype 是否符合模型期望
  · bf16 autocast 下的前向 + 反向是否数值正常（不出现 NaN/Inf）
  · 显存是否装得下（区域级 224² batch 32）
  · 原型提取的 B6 参数路径是否真的走进去了
  · `aggregate_by_image` 接真实 loader 的 file_name 是否对得上

本脚本用**极小样本**把上面每一条都真实跑一遍，5 分钟内给出结论，
避免"正式训练跑了 20 分钟才崩在第 300 个 batch"。

用法（AutoDL / 4090）
---------------------
    cd /root/autodl-tmp/<repo root>

    # 默认：48 个区域、1 个 train step、1 个 val batch
    python smoke_test_region_e2e.py \
        --tongue_root "/root/autodl-tmp/TCM-Tongue/shezhen_datasets1/shezhen datasets/shezhenv3-coco/shezhenv3-coco"

    # 更保守（显存紧时）
    python smoke_test_region_e2e.py --max_regions 16 --batch_size 4

    # 想顺带测真实显存占用（放大到正式配置的一个 batch）
    python smoke_test_region_e2e.py --max_regions 256 --batch_size 32

退出码 0 = 全链路 OK，可以正式开跑；非 0 = 有段落失败（输出里标 ✗）。
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import sys
import time
import traceback
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch

from data.tongue_coco_loader import DEFAULT_TONGUE_ROOT, build_datasets, resolve_tongue_root
from data.tongue_region_loader import build_region_datasets, make_region_loader
from models.tongue_label_mapping import MATRIX_VERSION, NUM_TONGUE_CLASSES, SYNDROMES_ZH
from models.tongue_vision_branch import (
    FEAT_DIM,
    PROTO_VERSION,
    TongueVisionBranch,
    configure_finetune,
    extract_syndrome_prototypes,
    load_biomedclip_visual,
)
from training.train_tongue_branch import collect_scores, compute_tongue_metrics
from training.train_tongue_region import aggregate_by_image, collect_scores_named
from utils.run_handoff import Handoff

# 运行交接摘要（2026-09-15）
H = Handoff("smoke_test_region_e2e")

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))


def mem_report(tag: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
    reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    return f"{tag}: peak_alloc={alloc:.0f}MB peak_reserved={reserved:.0f}MB"


def _main_impl():
    ap = argparse.ArgumentParser(description="区域级 trainer 全链路 GPU 冒烟")
    ap.add_argument("--tongue_root", type=str, default=DEFAULT_TONGUE_ROOT)
    ap.add_argument("--max_regions", type=int, default=48,
                    help="每 split 取前 N 个区域（冒烟用；正式训练不要设）")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3, help="跑几个 train step")
    ap.add_argument("--unfreeze_blocks", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--skip_proto", action="store_true", help="跳过原型提取段")
    # 容错参数（2026-09-15 加）：冒烟脚本没有 epoch 概念，用 --steps 控制步数。
    # 但正式 trainer 是 --epochs，容易顺手复制过来。这里显式接受并忽略，
    # 避免 argparse 直接 SystemExit(2) 让整轮冒烟白跑。
    ap.add_argument("--epochs", type=int, default=None,
                    help="【忽略】冒烟脚本按 step 计，不按 epoch；此参数仅为兼容习惯，传了无效果")
    args, unknown = ap.parse_known_args()
    if args.epochs is not None:
        print(f"[提示] --epochs {args.epochs} 在本冒烟脚本中无效（按 step 计，用 --steps 控制），已忽略。")
    if unknown:
        print(f"[提示] 忽略未识别参数：{' '.join(unknown)}（冒烟脚本参数见 --help）")

    print("=" * 76)
    print("区域级弱监督 trainer · 服务器端全链路冒烟")
    print("=" * 76)

    # ---------- [0] 环境 ----------
    print("\n[0] 环境")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not args.no_amp) and torch.cuda.is_available()
    print(f"  python      : {sys.version.split()[0]}")
    print(f"  torch       : {torch.__version__}")
    print(f"  device      : {device}")
    if torch.cuda.is_available():
        print(f"  gpu         : {torch.cuda.get_device_name(0)}")
        print(f"  cc          : {torch.cuda.get_device_capability(0)}"
              f"（bf16 需 ≥8.0；4090 = 8.9）")
    print(f"  amp(bf16)   : {use_amp}")
    print(f"  模块版本    : PROTO_VERSION={PROTO_VERSION} | MATRIX_VERSION={MATRIX_VERSION}")
    record("环境", device == "cuda", f"device={device}（GPU 上应为 cuda）")

    root = args.tongue_root or resolve_tongue_root()
    print(f"  数据根      : {root}")
    record("数据集路径存在", Path(root).is_dir(), str(root))
    if not Path(root).is_dir():
        return 1

    # ---------- [1] 视觉塔 ----------
    print("\n[1] 加载 BiomedCLIP 视觉塔（open_clip + HF 镜像）")
    t0 = time.time()
    try:
        visual, preprocess, raw_dim, img_size = load_biomedclip_visual(device=device)
        stats = configure_finetune(visual, n_unfreeze_blocks=args.unfreeze_blocks)
        record("视觉塔加载", True,
               f"raw_dim={raw_dim} img_size={img_size} 用时 {time.time()-t0:.1f}s")
        print(f"    微调统计：{stats}")
    except Exception as e:
        record("视觉塔加载", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # ---------- [2] 区域数据集 ----------
    print(f"\n[2] 区域级数据集（max_regions={args.max_regions}，冒烟截断）")
    try:
        t0 = time.time()
        reg_ds = build_region_datasets(
            root=root, preprocess=preprocess, crop_margin=0.15,
            max_regions=args.max_regions, verbose=True,
        )
        record("区域数据集构建", True, f"用时 {time.time()-t0:.1f}s")
    except Exception as e:
        record("区域数据集构建", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    train_ds, val_ds, test_ds = reg_ds["train"], reg_ds["val"], reg_ds["test"]
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        record("区域数据集非空", False,
               f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        return 1
    record("区域数据集非空", True,
           f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    name_to_cats = {s: {fn: cats for fn, cats, _ in reg_ds[s].samples}
                    for s in ("train", "val", "test")}

    # ---------- [3] DataLoader 出 batch ----------
    print("\n[3] 区域 DataLoader 出 batch")
    loaders = {
        "train": make_region_loader(train_ds, batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle_images=True, seed=42, drop_last=False),
        "val": make_region_loader(val_ds, batch_size=args.batch_size * 2,
                                  num_workers=args.num_workers,
                                  shuffle_images=False, seed=42),
    }
    try:
        imgs, targets, fnames = next(iter(loaders["train"]))
        ok = (imgs.dim() == 4 and targets.shape[1] == NUM_TONGUE_CLASSES
              and len(fnames) == imgs.size(0))
        record("batch 形状", ok,
               f"imgs={tuple(imgs.shape)} targets={tuple(targets.shape)} "
               f"names[{len(fnames)}] dtype={imgs.dtype}")
        if imgs.dim() != 4:
            return 1
        side = tuple(imgs.shape[-2:])
        record("图像边长 = 224", side == (224, 224), f"{side}")
    except Exception as e:
        record("batch 形状", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # ---------- [4] 模型 + 一次真实反传 ----------
    print(f"\n[4] 模型前向 + 反向（bf16 autocast={use_amp}，{args.steps} step）")
    try:
        model = TongueVisionBranch(
            visual_backbone=visual, raw_dim=raw_dim, feat_dim=FEAT_DIM
        ).to(device)
        ps = model.param_stats()
        print(f"    可训参数 {ps['trainable']/1e6:.2f}M ({ps['trainable_ratio']*100:.1f}%)")

        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters()
                       if not n.startswith("backbone.") and p.requires_grad]
        pg = [{"params": head_params, "lr": 1e-3}]
        if backbone_params:
            pg.append({"params": backbone_params, "lr": 1e-5})
        opt = torch.optim.AdamW(pg, weight_decay=0.01)

        class_mask = torch.ones(NUM_TONGUE_CLASSES, dtype=torch.float32).to(device)
        pos_weight = torch.ones(NUM_TONGUE_CLASSES, dtype=torch.float32).to(device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        step_done, losses = 0, []
        for imgs, targets, _ in loaders["train"]:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                feat, z21, _, _ = model(imgs)
                loss, parts = model.compute_loss(
                    z21, targets, class_mask=class_mask, pos_weight=pos_weight
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            lv = float(loss.detach())
            losses.append(lv)
            step_done += 1
            print(f"    step {step_done}: loss={lv:.4f} bce={parts['bce']:.4f} "
                  f"feat={tuple(feat.shape)} z21={tuple(z21.shape)}")
            if step_done >= args.steps:
                break

        finite = all(np.isfinite(losses))
        record("前向+反向", step_done > 0 and finite,
               f"{step_done} step，loss {'有限' if finite else '**含 NaN/Inf**'}")
        # 梯度有限性
        g_bad = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
        record("梯度有限", not g_bad, f"异常梯度参数 {len(g_bad)} 个" +
               (f"：{g_bad[:3]}" if g_bad else ""))
        print(f"    {mem_report('显存')}")
    except Exception as e:
        record("前向+反向", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # ---------- [5] collect_scores / 聚合 ----------
    print("\n[5] 评估收集 + 图像级聚合")
    try:
        vs, vt, _ = collect_scores(model, loaders["val"], device, use_amp)
        record("collect_scores(val)", True, f"scores={vs.shape} labels={vt.shape}")
        rs, rl, rnames = collect_scores_named(model, loaders["val"], device, use_amp)
        record("collect_scores_named(val)", True, f"scores={rs.shape} names[{len(rnames)}]")

        agg_s, agg_t = aggregate_by_image(rs, rnames, name_to_cats["val"])
        record("aggregate_by_image", agg_t.shape[0] == len(set(rnames)),
               f"区域 {rs.shape[0]} → 图 {agg_t.shape[0]}（唯一图 {len(set(rnames))}）")

        eval_classes = [i for i in range(NUM_TONGUE_CLASSES)
                        if int(vt[:, i].sum()) >= 1]
        m = compute_tongue_metrics(vt, vs, eval_classes)
        record("compute_tongue_metrics(区域)", True,
               f"macro_f1={m['macro_f1_valid']:.4f} 参与 {len(eval_classes)} 类")
        m2 = compute_tongue_metrics(agg_t, agg_s, eval_classes)
        record("compute_tongue_metrics(聚合)", True, f"macro_f1={m2['macro_f1_valid']:.4f}")
    except Exception as e:
        record("评估收集/聚合", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()

    # ---------- [6] 原型提取（B6 路径）----------
    if args.skip_proto:
        print("\n[6] 原型提取 — 已跳过（--skip_proto）")
    else:
        print("\n[6] 原型提取（整图特征 + B6 判据）")
        try:
            img_ds = build_datasets(root=root, preprocess=preprocess,
                                    crop_bbox=False, crop_margin=0.15)
            from torch.utils.data import DataLoader
            ld = DataLoader(img_ds["val"], batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers)
            _fs, _ft, feats = collect_scores(model, ld, device, use_amp)
            det = torch.tensor(_ft, dtype=torch.float32)
            protos, meta = extract_syndrome_prototypes(
                model, feats, det, use_gt=True, l2_normalize=True,
                strategy="centered", prune_normal_white=True,
            )
            ok = tuple(protos.shape) == (10, FEAT_DIM)
            record("原型提取(B6)", ok,
                   f"protos={tuple(protos.shape)} pruned_white={meta.get('n_pruned_white', '?')}")
            print(f"    n_images={meta['n_images']}")
            print(f"    pre_norm={[round(float(x), 3) for x in (meta.get('pre_norm') or [])]}")
            n_val = len(img_ds["val"])
            print(f"    ⚠️ 本段只为验证流程跑得通：**只用了 val split（{n_val} 图）**，")
            print(f"       不是全量（train+val+test = 6719 图）。所以 n_images / pruned_white")
            print(f"       都会显著偏小，**不能拿来验收**（血虚会显示成个位数，那是正常的）。")
            print(f"       全量验收看第 3 步：血虚 ≈189、pruned_white ≈937。")
            record("原型形状 (10,512)", ok, f"{tuple(protos.shape)}")
        except Exception as e:
            record("原型提取(B6)", False, f"{type(e).__name__}: {e}")
            traceback.print_exc()

    # ---------- 汇总 ----------
    print("\n" + "=" * 76)
    print("冒烟结果汇总")
    print("=" * 76)
    for name, ok, detail in RESULTS:
        print(f"  {'✓' if ok else '✗'} {name:<26s} {detail}")
    n_bad = sum(1 for _n, ok, _d in RESULTS if not ok)
    print(f"\n  {len(RESULTS)} 项检查，{len(RESULTS)-n_bad} 通过，{n_bad} 失败")
    if torch.cuda.is_available():
        print(f"  {mem_report('显存峰值')}")
    print("=" * 76)
    if n_bad == 0:
        print("✅ 全链路 OK —— 可以正式跑 `python training/train_tongue_region.py`")
        print("   建议正式命令加 --max_regions 200 --epochs 1 先跑一轮拿到真实显存/耗时估计")
        return 0
    print("❌ 有失败项，先修再跑正式训练")
    return 1


def main() -> int:
    """跑实现 + 无论成败都输出交接摘要。"""
    code = _main_impl()
    H.section("冒烟结论")
    H.kv("退出码", code)
    H.kv("结论", "全链路 OK，可正式训练" if code == 0 else "有失败项，先修")
    H.section("逐项结果")
    for name, ok, detail in RESULTS:
        H.line(f"  [{'OK' if ok else 'FAIL'}] {name:<26s} {detail}")
    if torch.cuda.is_available():
        H.section("显存")
        H.kv("峰值", mem_report("peak"))
        H.kv("GPU", torch.cuda.get_device_name(0))
    if code == 0:
        H.line("")
        H.line("下一步：正式训练（建议先 --max_regions 200 --epochs 1 估时）")
        H.line("  python training/train_tongue_region.py \\")
        H.line('      --tongue_root "<数据根>"')
    return code


if __name__ == "__main__":
    raise SystemExit(H.run(main))
