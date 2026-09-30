#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多 run 逐类 F1 对比 —— 消融结果表一键生成

背景
----
每次训练都会在 <output_dir>/test_report.json 落一份完整报告（含 10 类逐类 F1）。
做消融时手工抄逐类数字既慢又易错 —— 本脚本把 N 份报告读进来直接打表：

  1) 汇总表：macro F1 / AUC / 关键 config（一眼看出"只改了哪一个变量"）
  2) 逐类 F1 表 + Δ（相对 baseline 列），关注病性高亮

用法
----
    # 参数可以是 runs/<name> 目录，也可以是直接的 xxx/test_report.json
    # 列序 = 参数顺序
    python eval/compare_runs.py runs/full_model_v2_freeze runs/full_model_v2_b2

    # 指定 baseline 列（默认第 1 列）：Δ = 该列 − baseline
    python eval/compare_runs.py runs/tcm_text_macbert runs/full_model_v2_freeze runs/full_model_v2_b2

    # 换关注病性（默认 血虚，用于高亮并单列 Δ）
    python eval/compare_runs.py ... --watch 阴虚

    # seed 重复实验：按 run 名剥掉 seed 后缀（_s42 / _seed42）分组，输出 mean±std 与组间判定
    python eval/compare_runs.py runs/full_model_v2_freeze_e12_s{42,43,44} \
        runs/full_model_v2_b2_e12_s{42,43,44} --agg

纯标准库，无需 torch / sklearn。输出即为可贴进论文或日志的文本表。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.run_handoff import Handoff  # noqa: E402

# 运行交接摘要（2026-09-15）：本脚本的输出本身就是"要看的东西"，
# 所以开启 tee —— 控制台照常显示，同时完整落盘，可直接 cat 给 Buddy。
H = Handoff("compare_runs")
H.tee = True

COMPARE_VERSION = "2026-09-14b"

# 与 eval/metrics.SYNDROME_NAMES_ZH 保持一致（此处硬编码以避免引入 sklearn 依赖）
SYNDROMES_ZH = ["气虚", "血虚", "阴虚", "阳虚", "气滞", "血瘀", "痰湿", "湿热", "内风", "实热"]

CONFIG_KEYS = ["epochs", "batch_size", "lr", "text_lr", "softcl_weight",
               "freeze_text", "no_kg", "visual_mode", "select_metric", "seed"]

# --agg 分组：剥掉 run 名末尾的 seed 后缀（_s42 / _seed42 / -s42），同组聚合
_SEED_SUFFIX_RE = r"[-_](?:seed)?s?\d+$"


# ---------------------------------------------------------------- 宽度处理
def _dw(s) -> int:
    """显示宽度：中日韩等全角字符算 2（用于列对齐）。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in str(s))


def pad(s, width: int, align: str = "<") -> str:
    s = str(s)
    n = max(0, width - _dw(s))
    return s + " " * n if align == "<" else " " * n + s


def trunc(s, width: int) -> str:
    s = str(s)
    if _dw(s) <= width:
        return s
    out, w = "", 0
    for c in s:
        cw = 2 if ord(c) > 0x2E7F else 1
        if w + cw > width - 1:
            break
        out += c
        w += cw
    return out + "…"


# ---------------------------------------------------------------- 读取
def load_report(arg: str):
    """返回 (report_dict | None, 实际使用的路径)。"""
    p = Path(arg)
    if p.is_dir():
        p = p / "test_report.json"
    if not p.is_file():
        return None, p
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f), p
    except Exception as e:                                    # noqa: BLE001
        print(f"  ⚠️  解析失败 {p}: {e}")
        return None, p


def pick_metrics(r: dict) -> dict:
    """兼容三种报告结构：{"test":{...}} / {"test_metrics":{...}} / 扁平。"""
    for key in ("test", "test_metrics", "metrics"):
        v = r.get(key)
        if isinstance(v, dict) and any(k.startswith("f1_") for k in v):
            return v
    if any(k.startswith("f1_") for k in r):
        return r
    return {}


def first_key(d: dict, names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def fmt_cfg(v):
    """config 列专用：小学习率用科学计数法，避免 1e-05 被打成 0.0000 误读为 0。"""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float) and v != 0 and abs(v) < 1e-3:
        return f"{v:.0e}"
    return fmt(v)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="多 run 逐类 F1 对比")
    ap.add_argument("runs", nargs="+", help="runs/<name> 或 */test_report.json，列序即参数序")
    ap.add_argument("--baseline", type=int, default=1, help="baseline 列号（1-based，默认 1）")
    ap.add_argument("--watch", type=str, default="血虚", help="高亮病性（默认 血虚）")
    ap.add_argument("--agg", action="store_true",
                    help="按 run 名剥掉 seed 后缀分组，输出 mean±std（seed 重复实验用）")
    args = ap.parse_args()

    rows = []
    for a in args.runs:
        r, p = load_report(a)
        if r is None:
            print(f"✗ 找不到报告：{p}")
            continue
        rows.append({"arg": a, "path": p, "report": r, "metrics": pick_metrics(r)})

    if not rows:
        raise SystemExit("✗ 没有任何可用报告")
    if not 1 <= args.baseline <= len(rows):
        raise SystemExit(f"✗ --baseline 超出范围（1..{len(rows)}）")
    base_i = args.baseline - 1

    print("=" * 78)
    print(f"多 run 对比（compare_runs {COMPARE_VERSION}）  baseline = 第 {args.baseline} 列")
    print("=" * 78)

    # ---------- 汇总表 ----------
    w_run = max(14, max(_dw(Path(x["path"]).parent.name) for x in rows) + 1)
    w_mod = max(20, max(_dw(x["report"].get("model", "-")) for x in rows) + 1)
    print(f"\n{pad('#', 3)}{pad('run', w_run)}{pad('model', w_mod)}"
          f"{pad('macroF1', 9, '>')}{pad('AUC', 9, '>')}{pad('bestVal', 9, '>')}")
    print("-" * (3 + w_run + w_mod + 27))
    for i, x in enumerate(rows, 1):
        m = x["metrics"]
        name = Path(x["path"]).parent.name
        if name in ("runs", "."):                              # 直接给了某 runs/ 下的 json
            name = Path(x["path"]).stem
        print(f"{pad(i, 3)}{pad(trunc(name, w_run - 1), w_run)}"
              f"{pad(trunc(x['report'].get('model') or name, w_mod - 1), w_mod)}"
              f"{pad(fmt(first_key(m, ['macro_f1'])), 9, '>')}"
              f"{pad(fmt(first_key(m, ['auc_ovr_macro', 'auc'])), 9, '>')}"
              f"{pad(fmt(x['report'].get('best_val_score')), 9, '>')}")

    # ---------- config 差异（看"只改了哪个变量"） ----------
    cfgs = [x["report"].get("config") or {} for x in rows]
    if any(cfgs):
        print("\n关键 config：")
        print(f"  {pad('run', w_run)}" + "".join(pad(k[:11], 12) for k in CONFIG_KEYS))
        for i, (x, c) in enumerate(zip(rows, cfgs), 1):
            name = trunc(Path(x["path"]).parent.name, w_run - 1)
            vals = "".join(pad(trunc(fmt_cfg(c.get(k)), 11), 12) for k in CONFIG_KEYS)
            print(f"  {pad(name, w_run)}{vals}")

    # ---------- 逐类 F1 + Δ ----------
    w_cell = 9
    print(f"\n逐类 F1（Δ = 该列 − baseline 第 {args.baseline} 列）")
    print(f"{pad('病性', 8)}" + "".join(f"{pad('#' + str(i), w_cell, '>')}" for i in range(1, len(rows) + 1))
          + "".join(f"{pad('Δ' + str(i), w_cell, '>')}" for i in range(1, len(rows) + 1) if i - 1 != base_i))
    print("-" * (8 + w_cell * (2 * len(rows) - 1)))

    base_m = rows[base_i]["metrics"]
    for syn in SYNDROMES_ZH:
        line = pad(syn + ("  ←" if syn == args.watch else ""), 8)
        for x in rows:
            line += f"{pad(fmt(x['metrics'].get(f'f1_{syn}')), w_cell, '>')}"
        for i, x in enumerate(rows, 1):
            if i - 1 == base_i:
                continue
            a = x["metrics"].get(f"f1_{syn}")
            b = base_m.get(f"f1_{syn}")
            if a is None or b is None:
                line += f"{pad('-', w_cell, '>')}"
            else:
                d = a - b
                line += f"{pad(('+' if d >= 0 else '') + f'{d:.4f}', w_cell, '>')}"
        print(line)

    # ---------- 汇总行 ----------
    line = pad("macro", 8)
    for x in rows:
        line += f"{pad(fmt(first_key(x['metrics'], ['macro_f1'])), w_cell, '>')}"
    for i, x in enumerate(rows, 1):
        if i - 1 == base_i:
            continue
        a = first_key(x["metrics"], ["macro_f1"])
        b = first_key(base_m, ["macro_f1"])
        line += f"{pad(('+' if (a or 0) >= (b or 0) else '') + f'{(a or 0) - (b or 0):.4f}', w_cell, '>')}"
    print(line)

    # ---------- 关注病性单独结论 ----------
    w = args.watch
    print(f"\n关注病性「{w}」：")
    for i, x in enumerate(rows, 1):
        tag = "（baseline）" if i - 1 == base_i else ""
        print(f"  #{i} {fmt(x['metrics'].get(f'f1_{w}'))} {tag}")

    # ---------- seed 聚合（--agg，seed 重复实验用） ----------
    if args.agg:
        groups: dict[str, list] = {}
        for x in rows:
            nm = Path(x["path"]).parent.name
            groups.setdefault(re.sub(_SEED_SUFFIX_RE, "", nm) or nm, []).append(x)

        def _vals(g, key):
            out = []
            for x in g:
                v = first_key(x["metrics"], [key])
                if v is not None:
                    out.append(float(v))
            return out

        def _ms(v):
            if not v:
                return "-"
            if len(v) == 1:
                return f"{v[0]:.4f}(n=1)"
            return f"{statistics.mean(v):.4f}±{statistics.stdev(v):.4f}"

        print(f"\n{'=' * 78}")
        print(f"seed 聚合（run 名去 seed 后缀分组）  关注病性 = {args.watch}")
        print(f"{'=' * 78}")
        print(f"{pad('group', 26)}{pad('n', 4, '>')}{pad('macroF1 mean±std', 24, '>')}"
              f"{pad(args.watch + ' F1 mean±std', 24, '>')}")
        print("-" * 78)
        for k, g in groups.items():
            print(f"{pad(trunc(k, 25), 26)}{pad(len(g), 4, '>')}"
                  f"{pad(_ms(_vals(g, 'macro_f1')), 24, '>')}"
                  f"{pad(_ms(_vals(g, f'f1_{args.watch}')), 24, '>')}")

        if len(groups) == 2:
            (k1, g1), (k2, g2) = list(groups.items())
            print("\n组间差值（两种合并方差判定）：")
            for metric, label in (("macro_f1", "macroF1"),
                                  (f"f1_{args.watch}", f"{args.watch} F1")):
                a, b = _vals(g1, metric), _vals(g2, metric)
                if len(a) < 2 or len(b) < 2:
                    print(f"  {label}: 每组 n<2，无法估计方差（至少各组 2 seed）")
                    continue
                d = statistics.mean(b) - statistics.mean(a)
                pooled = ((statistics.stdev(a) ** 2 + statistics.stdev(b) ** 2) / 2) ** 0.5
                ratio = abs(d) / pooled if pooled > 0 else float("inf")
                verdict = ("可分辨（|Δ| > 2×std）" if ratio > 2
                           else "不可分辨（|Δ| ≤ 2×std，落在运行噪声内）")
                print(f"  {label}: {k2} − {k1} = {d:+.4f}"
                      f" | 合并 std={pooled:.4f} | |Δ|/std={ratio:.2f} → {verdict}")

    print("\n提示：文本基线逐类 F1 见 runs/tcm_text_macbert/test_report.json；")
    print("      同 epoch 的严格对照是 runs/full_model_v2_freeze_e12（B-2 对照为 runs/full_model_v2_b2_e12）。")

    # ---------- 交接摘要的结构化字段（完整表格见下方"完整控制台输出"） ----------
    H.section("本次对比")
    H.kv("对比 run 数", len(rows))
    _names = [Path(x["path"]).parent.name for x in rows]
    H.line("  runs: " + ", ".join(_names))
    if 0 <= base_i < len(_names):
        H.kv("基线(baseline)", _names[base_i])
    H.kv("关注病性", args.watch)
    H.kv("--agg", args.agg)


if __name__ == "__main__":
    raise SystemExit(H.run(main))
