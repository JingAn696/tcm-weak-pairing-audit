"""
随机原型对照生成器（Stage B · B-2 的阴性对照）
================================================

**为什么需要它（论文消融设计）**

B-2 的原型注意力若真的"用到了舌象知识"，那么把原型换成**错的/随机的**向量，
增益应当消失。否则"提升"只说明模型多了一层可参数量化的随机扰动，
与舌象无关 —— 这正是审稿人会问的第一个问题。

两种对照（生成与 extract_prototypes_v2.py 完全同格式的 .pt，
直接喂给 `train_full_model.py --proto_path`，B-2 一行不用改）：

| 模式 | 做法 | 检验的问题 |
|---|---|---|
| `shuffle`  | 把**活跃**原型向量做**错位置换**（derangement：没有任何病性拿到自己的原型）。活跃 = `norm>0 且 support>=min_support`；**非活跃列原样不动**。meta 不动 | 同样的向量集合、错的对应关系 → 增益是否来自"方向正确" |
| `gaussian` | 10 个标准高斯随机向量，逐行 L2 归一化（与真实原型同范数分布），meta 不动 | 任意 10×512 归一化向量 → 注意力层是否"给什么都能涨" |

**关键设计 1**：meta（n_images 等）保持原样。
⇒ `proto_min_support` 屏蔽的病性集合与真实原型**完全一致**，
  对照的唯一变量是"原型向量的方向"，实验解读最干净。

**关键设计 2（2026-09-15c 修，两轮）**：shuffle **只在「活跃列」之间置换，其余列原地不动**。

**活跃列** = 真正进入注意力的列 = 同时满足 `norm > 0` **且** `support >= min_support`。
`syndrome_proto_attention.py:176` 的判据就是这两个条件的**合取**：

    valid = (norms > 1e-6) & (support >= min_support)      # 默认 min_support = 30

本项目里两条都会踩到：**内风的原型是零向量**（0 支撑图），**气滞只有 5 图**（< 30 被 mask），
所以活跃列只有 8 个。若让这两列参与置换，会同时破坏两条不变式：

| 破坏的不变式 | 后果 |
|---|---|
| `valid` 集合两臂不等 | 零向量被换进活跃槽 → 该病性 `norm==0` → **被 mask**。实测正好落在**血虚**（论文最关注类），对照臂有效原型 8→7 |
| **活跃向量多重集**两臂不等 | 真实原型被换进已 mask 的槽位＝**白丢**，同时把低支撑向量提进活跃槽。实测 血瘀 被换进气滞槽，气滞 被提进血瘀槽 |

⇒ 只有"活跃列内部错位置换"才真的满足"**同样的向量集合、错的对应关系**"这句承诺。
生成后脚本会自查这两条，任一不通过会打印醒目 WARN 并在摘要里标 ❌。

用法（AutoDL）
--------------
    cd /root/autodl-tmp/papers/code
    python eval/gen_random_prototypes.py \
        --proto runs/tongue_branch/tongue_prototypes_centered.pt \
        --mode shuffle --seed 42
    python eval/gen_random_prototypes.py \
        --proto runs/tongue_branch/tongue_prototypes_centered.pt \
        --mode gaussian --seed 42

`--min_support` 默认取 `syndrome_proto_attention.DEFAULT_MIN_SUPPORT`（30），
**必须与训练时传给 `train_full_model.py --proto_min_support` 的值一致**，
否则"活跃列"口径与训练时不一致，不变式检查会失效。

产出（与输入同目录）
--------------------
    tongue_prototypes_centered_shuffle_s42.pt
    tongue_prototypes_centered_gaussian_s42.pt

自测
----
    python eval/gen_random_prototypes.py --selftest   （不需要真实原型文件）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _SCRIPT_DIR.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch  # noqa: E402

from utils.run_handoff import Handoff  # noqa: E402

# 运行交接摘要（2026-09-15）
H = Handoff("gen_random_prototypes")


def derangement_perm(n: int, seed: int) -> list[int]:
    """生成长度 n 的错位置换（无不动点）。

    n ≤ 1 时无错位置换 → 抛错（本场景 n=10，不会触发）。
    """
    if n < 2:
        raise ValueError("n<2 不存在错位置换")
    g = torch.Generator().manual_seed(seed)
    for _ in range(10000):
        perm = torch.randperm(n, generator=g).tolist()
        if all(perm[i] != i for i in range(n)):
            return perm
    raise RuntimeError("未能生成错位置换（概率极小，请换 seed）")


def make_control_prototypes(
    protos: torch.Tensor,
    mode: str,
    seed: int,
    support: torch.Tensor | None = None,
    min_support: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """由真实原型生成对照原型。返回 (P_control, control_info)。

    shuffle 只置换**活跃列**（既非零范数、又满足 support >= min_support 的列），
    其余列（零范列 = 内风；低支撑列 = 气滞）**原样保留**。两条不变式由此保证：

    1. `valid = (norm>0) & (support>=min_support)` 集合两臂**逐位相等**
       （否则零向量被换进有效槽位 → 该病性被 mask 掉，对照被削弱）
    2. 进入注意力的**向量多重集**两臂相等
       （否则真实原型被换进已 mask 的槽位＝白丢，同时把低支撑向量提进活跃槽）

    未传 support/min_support 时退化为"仅按非零范数划分池"（自测用）。
    """
    n, d = protos.shape
    g = torch.Generator().manual_seed(seed)

    if mode == "shuffle":
        norms_src = protos.norm(dim=1)
        usable = norms_src > 1e-6
        if support is not None and min_support is not None:
            usable = usable & (support >= float(min_support))
        pool = [i for i in range(n) if bool(usable[i])]        # 活跃列：可参与置换
        fixed = [i for i in range(n) if i not in pool]          # 固定列：原地不动
        zero_cols = [i for i in range(n) if float(norms_src[i]) <= 1e-6]
        low_sup_cols = [i for i in fixed if i not in zero_cols]
        if len(pool) < 2:
            raise ValueError(f"活跃原型列只有 {len(pool)} 个，无法构造错位置换")
        sub = derangement_perm(len(pool), seed)                 # 池内无不动点
        perm = list(range(n))
        for k, i in enumerate(pool):
            perm[i] = pool[sub[k]]
        P = protos[perm].clone()
        info = {"perm": perm, "pool": pool, "fixed": fixed,
                "zero_cols": zero_cols, "low_support_cols": low_sup_cols,
                "check": (f"活跃列 {len(pool)} 个内部无不动点；"
                          f"固定列 {len(fixed)} 个原样保留"
                          f"（零范 {len(zero_cols)} + 低支撑 {len(low_sup_cols)}）")}
    elif mode == "gaussian":
        P = torch.randn(n, d, generator=g)
        # 与真实原型同范数分布：真实原型已 L2 归一 → 随机向量也归一
        P = torch.nn.functional.normalize(P, p=2, dim=1)
        info = {"check": "逐行 L2 归一，范数分布与真实原型一致"}
    else:
        raise ValueError(f"未知 mode：{mode}（应为 shuffle / gaussian）")

    return P, info


def main():
    ap = argparse.ArgumentParser(description="随机原型对照生成器（B-2 阴性对照）")
    ap.add_argument("--proto", type=str, default=None,
                    help="真实原型文件（extract_prototypes_v2.py / train_tongue_branch.py 产出）")
    ap.add_argument("--mode", choices=["shuffle", "gaussian"], default="shuffle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None,
                    help="输出路径（默认与输入同目录，按模式+seed 自动命名）")
    ap.add_argument("--min_support", type=float, default=None,
                    help="活跃原型列下界；默认取 models.syndrome_proto_attention.DEFAULT_MIN_SUPPORT（30）。"
                         "必须与训练时传给 train_full_model.py 的值一致，否则不变式口径不符")
    ap.add_argument("--selftest", action="store_true", help="运行自测（不需要真实原型）")
    args = ap.parse_args()

    if args.selftest:
        H.enabled = False          # 自测不出摘要
        _selftest()
        return

    if not args.proto:
        ap.error("必须提供 --proto（真实原型文件），或用 --selftest 跑自测")

    src = Path(args.proto)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    protos = ckpt["prototypes"].float()
    assert protos.dim() == 2, f"prototypes 应为二维，得到 {tuple(protos.shape)}"

    # ---- 先取出 support / min_support，供"活跃列"判定使用 ----
    # 判据与 syndrome_proto_attention.load_prototypes_from_file 逐字一致
    from models.syndrome_proto_attention import DEFAULT_MIN_SUPPORT
    from models.tongue_label_mapping import SYNDROMES_ZH
    _meta_in = ckpt.get("meta", {}) or {}
    _n_img_in = _meta_in.get("n_images") or [0] * protos.shape[0]
    support = torch.tensor([float(x) for x in _n_img_in])
    min_support = float(args.min_support) if args.min_support is not None else float(DEFAULT_MIN_SUPPORT)
    names = ckpt.get("syndromes") or SYNDROMES_ZH

    P, info = make_control_prototypes(protos, args.mode, args.seed,
                                      support=support, min_support=min_support)

    # ---- 两条不变式（对照公平性的硬指标）----
    valid_real = (protos.norm(dim=1) > 1e-6) & (support >= min_support)
    valid_ctrl = (P.norm(dim=1) > 1e-6) & (support >= min_support)
    n_valid_real, n_valid_ctrl = int(valid_real.sum()), int(valid_ctrl.sum())
    valid_identical = bool(torch.equal(valid_real, valid_ctrl))
    if not valid_identical:
        _lost = [names[j] for j in range(len(names))
                 if bool(valid_real[j]) and not bool(valid_ctrl[j])]
        _gained = [names[j] for j in range(len(names))
                   if bool(valid_ctrl[j]) and not bool(valid_real[j])]
        print(f"  [WARN] 有效原型集合被改变！丢失={_lost} 新增={_gained}")
        print("  [WARN] 该对照不公平，请检查生成逻辑（零范列应原样保留）")

    # 不变式 2：进入注意力的向量多重集不变（仅 shuffle 有意义）
    active = [j for j in range(len(names)) if bool(valid_real[j])]
    multiset_ok = True
    if args.mode == "shuffle":
        perm_chk = info["perm"]
        multiset_ok = (sorted(perm_chk[j] for j in active) == active)
        if not multiset_ok:
            _wasted = [names[j] for j in active
                       if perm_chk[j] not in active]
            _promoted = [names[perm_chk[j]] for j in active
                         if perm_chk[j] not in active]
            print(f"  [WARN] 活跃向量多重集被改变！活跃槽拿到非活跃原型：{list(zip(_wasted, _promoted))}")
            print("  [WARN] 有真实原型被换进已 mask 的槽位＝白丢，该对照不公平")

    # ---- 保存：与真实原型同格式（B-2 的 load_prototypes_from_file 只读这些键）----
    blob = {
        "prototypes": P,
        "syndromes": ckpt.get("syndromes"),
        "strategy": f"{ckpt.get('strategy', '?')}_{args.mode}",
        # meta 原样保留 ⇒ proto_min_support 屏蔽集合与真实原型一致（公平对照）
        "meta": ckpt.get("meta", {}),
        "source": ckpt.get("source", "?"),
        "feat_dim": ckpt.get("feat_dim", protos.shape[1]),
        "matrix_version": ckpt.get("matrix_version"),
        "prune_normal_white": ckpt.get("prune_normal_white"),
        # 对照标记
        "control": True,
        "control_mode": args.mode,
        "control_seed": args.seed,
        "control_of": str(src),
        "control_info": info,
    }

    if args.output:
        out = Path(args.output)
    else:
        stem = src.stem  # e.g. tongue_prototypes_centered
        out = src.with_name(f"{stem}_{args.mode}_s{args.seed}.pt")
    torch.save(blob, out)

    print("=" * 70)
    print("随机原型对照已生成")
    print("=" * 70)
    print(f"  真实原型：{src}")
    print(f"  模式    ：{args.mode}（seed={args.seed}）")
    if args.mode == "shuffle":
        perm = info["perm"]
        _fixlab = {j: "零范列·原地" for j in info["zero_cols"]}
        _fixlab.update({j: "低支撑·原地" for j in info["low_support_cols"]})
        print("  置换    ：" + ", ".join(
            f"{names[i]}→{names[perm[i]]}" + (f"（{_fixlab[i]}）" if i in _fixlab else "")
            for i in range(len(perm))))
    print(f"  输出    ：{out}")
    _norm_note = ("（min=0 只应出现在零支撑病性【如内风】，属预期）"
                  if args.mode == "shuffle" else "（全部行 L2 归一，应为 1）")
    print(f"  范数检查：min={P.norm(dim=1).min():.4f} max={P.norm(dim=1).max():.4f}{_norm_note}")
    print(f"  有效原型：真实 {n_valid_real} 个 / 对照 {n_valid_ctrl} 个"
          f" → {'✅ 逐位一致' if valid_identical else '❌ 不一致'}"
          f"（min_support={min_support:g}；masked={[names[j] for j in range(len(names)) if not bool(valid_real[j])]}）")
    if args.mode == "shuffle":
        print(f"  置换范围：{len(info['pool'])} 个活跃列（{', '.join(names[j] for j in info['pool'])}）")
        _fixed_desc = []
        if info["zero_cols"]:
            _fixed_desc.append(f"零范列 {len(info['zero_cols'])} 个（{', '.join(names[j] for j in info['zero_cols'])}）")
        if info["low_support_cols"]:
            _fixed_desc.append(f"低支撑列 {len(info['low_support_cols'])} 个"
                               f"（{', '.join(names[j] for j in info['low_support_cols'])}）")
        print(f"  原地不动：{'；'.join(_fixed_desc) or '无'}")
        print(f"  活跃向量多重集：" + ("✅ 不变（无真实原型被换进 masked 槽位）"
                                      if multiset_ok else "❌ 被改变"))
    print()
    print("  B-2 对照运行：")
    print(f"    python training/train_full_model.py ... --visual_mode proto_attn \\")
    print(f"        --proto_path {out} --output_dir runs/full_model_v2_b2_{args.mode}_s{args.seed}")
    print("=" * 70)

    # ---------- 交接摘要 ----------
    H.set_out_dir(src.parent)
    H.section("配置")
    H.kv("source_prototypes", src)
    H.kv("mode", args.mode)
    H.kv("seed", args.seed)
    H.kv("source_strategy", ckpt.get("strategy", "?"))
    H.kv("source_matrix_version", ckpt.get("matrix_version", "?"))
    H.section("对照生成结果")
    H.kv("output", out)
    H.kv("norm_min_max", f"{P.norm(dim=1).min():.4f} / {P.norm(dim=1).max():.4f}"
                         + ("（min=0 只应来自零支撑病性【如内风】，属预期）"
                            if args.mode == "shuffle" else "（全部行 L2 归一，应为 1）"))
    # 对照公平性硬指标：有效原型集合必须逐位相等（否则对照被人为削弱）
    H.kv("有效原型个数", f"真实 {n_valid_real} / 对照 {n_valid_ctrl}"
                         f"（min_support={min_support:g}）")
    H.kv("有效集合不变检查", "通过（valid 逐位一致，两臂 masked 集合相同）" if valid_identical
                              else "❌ 未通过：对照的有效原型集合与真实不一致，该 run 不可用")
    H.kv("masked 病性", ", ".join(names[j] for j in range(len(names))
                                  if not bool(valid_real[j])) or "无")
    if args.mode == "shuffle":
        perm = info["perm"]
        _fixlab = {j: "零范列·原地" for j in info["zero_cols"]}
        _fixlab.update({j: "低支撑·原地" for j in info["low_support_cols"]})
        H.kv("同时满足两条件才算活跃原型", "norm>0 且 support>=min_support（活跃 = 真正进入注意力的列）")
        H.kv("置换语义", "左边槽位 ← 右边病性的原型（如 血虚→内风 表示血虚槽放内风的原型）")
        H.kv("permutation", ", ".join(
            f"{names[i]}→{names[perm[i]]}" + (f"({_fixlab[i]})" if i in _fixlab else "")
            for i in range(len(perm))))
        H.kv("置换范围", f"仅 {len(info['pool'])} 个活跃列参与：{', '.join(names[j] for j in info['pool'])}")
        H.kv("原地不动", (f"零范列 {len(info['zero_cols'])} 个："
                        f"{', '.join(names[j] for j in info['zero_cols']) or '无'}"
                        f"（防止零向量被换进活跃槽把该病性 mask 掉）；"
                        f"低支撑列 {len(info['low_support_cols'])} 个："
                        f"{', '.join(names[j] for j in info['low_support_cols']) or '无'}"
                        f"（它们本就被 mask，若参与置换会把真实原型换进无效槽＝白丢）"))
        H.kv("活跃向量多重集不变检查", "通过（活跃槽拿到的原型仍全部来自活跃列 → 向量集合逐位保持）"
                                      if multiset_ok else
                                      "❌ 未通过：有真实原型被换进 masked 槽位，该对照不公平")
        H.kv("无不动点检查", "通过（活跃列内部 perm[i]!=i 全部成立）")
    # meta 原样保留 = 屏蔽集合与真实原型一致（公平对照的关键）
    _meta = ckpt.get("meta", {}) or {}
    H.kv("meta.n_images 已原样保留", "是（proto_min_support 屏蔽集合与真实原型一致）")
    if _meta.get("n_images") is not None:
        H.kv("n_images", _meta["n_images"])
    H.artifact(out)
    H.line("")
    H.line("下一步（人工确认）：把上面 output 的 .pt 作为 --proto_path 传给 train_full_model.py，")
    H.line("--output_dir 用 runs/full_model_v2_b2_<mode>_s<seed>")


def _selftest():
    print("=" * 70)
    print("随机原型对照生成器自测")
    print("=" * 70)
    from models.tongue_label_mapping import SYNDROMES_ZH

    n, d = len(SYNDROMES_ZH), 512
    g = torch.Generator().manual_seed(0)
    protos = torch.nn.functional.normalize(torch.randn(n, d, generator=g), p=2, dim=1)

    # shuffle：无不动点 + 同向量集合
    P_s, info_s = make_control_prototypes(protos, "shuffle", seed=42)
    perm = info_s["perm"]
    assert all(perm[i] != i for i in range(n)), "shuffle 必须无不动点"
    assert sorted(perm) == list(range(n)), "shuffle 必须是一个置换"
    # 向量集合相同（排序后逐行一致）
    a = protos[perm]
    assert torch.allclose(P_s, a), "shuffle 应只是行重排"
    # 每行都拿到了"别人的原型"
    for i in range(n):
        assert not torch.allclose(P_s[i], protos[i]), f"第 {i} 行不应是自己的原型"
    print(f"OK shuffle：置换 {perm} 无不动点，向量集合不变")

    # ---- 回归测试（2026-09-15c，两轮）：非活跃列必须原地不动，两条不变式都不得被破坏 ----
    # 复刻真实场景：内风 0 支撑 → 零向量；气滞 5 图 → 低于 min_support（两条都会踩到）
    from models.syndrome_proto_attention import DEFAULT_MIN_SUPPORT
    j_qt = SYNDROMES_ZH.index("气滞")
    j_wf = SYNDROMES_ZH.index("内风")
    protos_zero = protos.clone()
    protos_zero[j_wf] = 0.0
    support = torch.full((n,), 999.0)
    support[j_qt] = 5.0
    support[j_wf] = 0.0
    valid_before = (protos_zero.norm(dim=1) > 1e-6) & (support >= DEFAULT_MIN_SUPPORT)
    active = [j for j in range(n) if bool(valid_before[j])]

    ok_any_seed = True
    for s in range(200):
        P_c, info_c = make_control_prototypes(protos_zero, "shuffle", seed=s,
                                             support=support, min_support=DEFAULT_MIN_SUPPORT)
        # 不变式 1：valid 集合逐位相等
        valid_after = (P_c.norm(dim=1) > 1e-6) & (support >= DEFAULT_MIN_SUPPORT)
        if not torch.equal(valid_before, valid_after):
            ok_any_seed = False
            lost = [SYNDROMES_ZH[j] for j in range(n)
                    if bool(valid_before[j]) and not bool(valid_after[j])]
            print(f"  ✗ seed={s} 有效集合被改变，丢失：{lost}")
            break
        # 不变式 2：活跃向量多重集不变（活跃槽只能拿活跃列的原型）
        pc = info_c["perm"]
        if sorted(pc[j] for j in active) != active:
            ok_any_seed = False
            print(f"  ✗ seed={s} 活跃向量多重集被改变：perm[active]={[pc[j] for j in active]}")
            break
        # 非活跃列必须原样保留（内风仍为零向量；气滞仍拿自己的原型）
        for j in (j_wf, j_qt):
            if not torch.allclose(P_c[j], protos_zero[j]):
                ok_any_seed = False
                print(f"  ✗ seed={s} 非活跃列 {SYNDROMES_ZH[j]} 被移动")
                break
    assert ok_any_seed, "非活跃列必须原地不动，且 200 个 seed 下两条不变式都不得被破坏"
    n_v = int(valid_before.sum())
    print(f"OK 非活跃列保护：200 个 seed 下有效原型集合恒为 {n_v} 个、活跃向量多重集不变")
    print(f"   （内风＝零范列、气滞＝低支撑列均未被换出；旧实现①会把零向量换进活跃槽 → 少 1 个有效原型，"
          f"②会把真实原型换进已 mask 槽 → 白丢 1 个向量）")

    # gaussian：形状/范数/确定性
    P_g1, _ = make_control_prototypes(protos, "gaussian", seed=42)
    P_g2, _ = make_control_prototypes(protos, "gaussian", seed=42)
    P_g3, _ = make_control_prototypes(protos, "gaussian", seed=43)
    assert P_g1.shape == (n, d)
    assert torch.allclose(P_g1.norm(dim=1), torch.ones(n), atol=1e-5), "gaussian 应逐行归一"
    assert torch.allclose(P_g1, P_g2), "同 seed 应可复现"
    assert not torch.allclose(P_g1, P_g3), "不同 seed 应不同"
    print("OK gaussian：形状 (10,512)、逐行范数=1、同 seed 可复现、异 seed 不同")

    # 与真实原型的相关性应接近 0（高维随机向量近似正交）
    sim = (P_g1 @ protos.T).abs().max()
    print(f"OK gaussian 与真实原型最大 |余弦| = {sim:.3f}（高维下应 ≪ 0.5）")
    assert sim < 0.5, "随机向量与真实原型不应高度相关"

    # 错位置换生成器本身的正确性
    for s in range(50):
        p = derangement_perm(10, seed=s)
        assert all(p[i] != i for i in range(10)) and sorted(p) == list(range(10))
    print("OK derangement_perm：50 个 seed 全部生成合法错位置换")

    print("=" * 70)
    print("全部自测通过 ✓")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(H.run(main))
