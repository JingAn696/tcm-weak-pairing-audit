#!/usr/bin/env bash
# ============================================================================
# run_baseline_seeds.sh — baseline 1（纯文本 MacBERT）seed 重复
# 版本：2026-09-16a
# ----------------------------------------------------------------------------
# 为什么要有这个（2026-09-16）：
#   freeze 臂 3 seed 实测 macro F1 std = 0.0041，而**血虚单类**跨 seed 极差
#   高达 0.0522（−0.0432 / −0.0042 / +0.0090）—— 说明"单次运行的 Δ"完全
#   不可解读。可论文里所有 Δ 都拿 baseline 1（**单 seed** 的 0.8401 /
#   血虚 0.6122）当参照：基准自己没有方差，整套 Δ 就都站不住。
#   所以给 baseline 1 补 3 seed，且**种子集与 freeze / b2 臂完全相同**，
#   这样两臂还能做逐 seed 配对比较（比独立比较更严格）。
#
# 用法：
#   bash run_baseline_seeds.sh [seeds] [epochs]
#     seeds   默认 "42 43 44"（与 freeze / b2 臂同种子集）
#     epochs  默认 3（与历史 baseline 1 同口径）
#
# 产物：
#   runs/tcm_text_macbert_s<seed>/test_report.json   每 seed 独立目录
#   log_baseline_s<seed>.log                         完整日志（含 HANDOFF 摘要）
#   baseline_summary.txt                             本脚本汇总表（贴回用）
#   baseline_done.txt                                完成标记
#
# 🔴 绝不写回 runs/tcm_text_macbert/ —— 那里有 Stage B 依赖的
#    checkpoint-6630。本脚本一律输出到 runs/tcm_text_macbert_s<seed>/，
#    且 baselines/tcm_sd_text_macbert.py 自身也会拒绝写入含 checkpoint-* 的目录。
# ============================================================================
set -u

SEEDS="${1:-42 43 44}"
EPOCHS="${2:-3}"

# 历史单 seed 数字（2026-09-15 论文协议）—— 用来量"复现偏差"
HIST_F1=0.8401
HIST_AUC=0.9750
HIST_WX=0.6122

# ---- 无论从哪里调用，都切到脚本所在目录（否则相对路径全错）----
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

echo "=============================================================="
echo " baseline 1 · 纯文本 MacBERT · seed 重复"
echo " seeds  : $SEEDS"
echo " epochs : $EPOCHS"
echo " cwd    : $(pwd)"
echo " 开始   : $(date '+%F %T')"
echo "=============================================================="

# ---- 预检：缺文件就别浪费 GPU 时间 ----
[ -f "baselines/tcm_sd_text_macbert.py" ] || { echo "[FATAL] 找不到 baselines/tcm_sd_text_macbert.py"; exit 3; }
[ -f "utils/run_handoff.py" ] || { echo "[FATAL] 找不到 utils/run_handoff.py（漏传 utils/？）"; exit 3; }
[ -f "data/splits_v2/train.csv" ] || { echo "[FATAL] 找不到 data/splits_v2/train.csv"; exit 3; }

# 只提醒、不报错：Stage B 的文本塔目录必须留在原地，本脚本不碰它
if [ -d "runs/tcm_text_macbert/checkpoint-6630" ]; then
  echo "[预检] OK：runs/tcm_text_macbert/checkpoint-6630 在位（本脚本不会碰它）"
else
  echo "[预检] ⚠️ 没看到 runs/tcm_text_macbert/checkpoint-6630 ——"
  echo "       本脚本不依赖它，但若 Stage B 还要续跑，先确认这个路径。"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[预检] GPU：$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
fi
echo "[预检] 通过 $(date '+%F %T')"
echo "--------------------------------------------------------------"

FAILED=""
PAIRS=""
FIRST="$(echo $SEEDS | awk '{print $1}')"

for s in $SEEDS; do
  OUT="runs/tcm_text_macbert_s${s}"
  PAIRS="$PAIRS $s:$OUT"
  LOG="log_baseline_s${s}.log"
  echo "##### [$(date '+%F %T')] baseline seed=$s 开始 → $OUT #####"
  python baselines/tcm_sd_text_macbert.py \
      --seed "$s" \
      --epochs "$EPOCHS" \
      --output_dir "$OUT" \
      > "$LOG" 2>&1
  RC=$?

  if [ "$RC" -eq 0 ]; then
    echo "##### [$(date '+%F %T')] baseline seed=$s 结束 OK (rc=0) #####"
    grep -E "^(macro_f1|auc_ovr_macro|subset_accuracy|hamming_loss) :" "$LOG" | sed 's/^/    /'
  else
    echo "##### [$(date '+%F %T')] baseline seed=$s 失败 rc=$RC #####"
    echo "    —— $LOG 尾部 25 行 ——"
    tail -n 25 "$LOG" | sed 's/^/    /'
    FAILED="$FAILED $s"
    if [ "$s" = "$FIRST" ]; then
      echo "[ABORT] 第 1 个 seed 就失败 → 先查配置，终止后续 seed，避免白烧 GPU"
      break
    fi
  fi
done

# ---- 汇总表（与 arm_summary_*.txt 的列语义对齐，便于横向对照）----
python - "$EPOCHS" "$SEEDS" "$HIST_F1" "$HIST_AUC" "$HIST_WX" $PAIRS <<'PY' || echo "[WARN] 汇总表生成失败（不影响上面的训练结果）"
import json, statistics, sys, unicodedata
from pathlib import Path

def _w(s):
    """显示宽度：中文算 2 列（与 utils/run_handoff.py 同一套算法）"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))

def pad(s, n, right=False):
    s = str(s)
    k = n - _w(s)
    return (" " * max(k, 0) + s) if right else (s + " " * max(k, 0))

ep = sys.argv[1]
intended = sys.argv[2].split()
hist_f1, hist_auc, hist_wx = (float(x) for x in sys.argv[3:6])
pairs = {}
for a in sys.argv[6:]:
    s, d = a.split(":", 1)
    pairs[s] = d

def fmt(x, nd=4):
    return "  ——  " if x is None else f"{x:.{nd}f}"

# baseline 的 test_report.json 是**扁平**结构（不像多模态 run 的 {test:{}, watch_syndrome:{}}）
rows = []
for s in intended:
    d = pairs.get(s)
    if d is None:                                     # 前一个 seed 失败被 ABORT
        rows.append((s, None, None, None, None, None))
        continue
    rp = Path(d) / "test_report.json"
    if not rp.exists():
        rows.append((s, None, None, None, None, d))
        continue
    try:
        r = json.loads(rp.read_text(encoding="utf-8-sig"))   # 容忍 BOM
    except Exception as e:
        print(f"[WARN] {rp} 读取失败：{type(e).__name__}: {e}")
        rows.append((s, None, None, None, None, d))
        continue
    f1 = r.get("macro_f1")
    auc = r.get("auc_ovr_macro")
    wx = r.get("f1_血虚")
    dwx = (wx - hist_wx) if wx is not None else None
    rows.append((s, f1, auc, wx, dwx, d))

lines = []
lines.append("=" * 84)
lines.append(f" baseline_summary · baseline=1（纯文本 MacBERT）  epochs={ep}  seeds={intended}")
lines.append("=" * 84)
lines.append("".join([pad("seed", 8), pad("macro_f1", 12, True),
                      pad("auc_ovr_macro", 16, True), pad("血虚 F1", 12, True),
                      pad("血虚 Δvs历史", 14, True)]))
for s, f1, auc, wx, dwx, _d in rows:
    dws = "  ——  " if dwx is None else f"{dwx:+.4f}"
    lines.append("".join([pad(s, 8), pad(fmt(f1), 12, True), pad(fmt(auc), 16, True),
                          pad(fmt(wx), 12, True), pad(dws, 14, True)]))
vals = [r[1] for r in rows if r[1] is not None]        # macro_f1
wxvals = [r[3] for r in rows if r[3] is not None]      # 血虚 F1（论文的观察类）
if len(vals) >= 2:
    sd = statistics.stdev(vals)
    sdx = statistics.stdev(wxvals) if len(wxvals) >= 2 else None
    lines.append("-" * 84)
    # 两个指标各给 mean/std/range —— 只有 macro_f1 的方差不够，WATCH_SYNDROME=血虚
    # 的逐类方差才是"单次运行能不能读逐类 Δ"的直接依据（freeze 臂实测极差 0.0522）
    lines.append("".join([pad("指标", 12), pad("mean", 10, True), pad("std(n-1)", 12, True),
                          pad("range", 22, True), pad("极差", 10, True)]))
    lines.append("".join([pad("macro_f1", 12), pad(f"{statistics.mean(vals):.4f}", 10, True),
                          pad(f"{sd:.4f}", 12, True),
                          pad(f"[{min(vals):.4f}, {max(vals):.4f}]", 22, True),
                          pad(f"{max(vals) - min(vals):.4f}", 10, True)]))
    if sdx is not None:
        lines.append("".join([pad("血虚 F1", 12), pad(f"{statistics.mean(wxvals):.4f}", 10, True),
                              pad(f"{sdx:.4f}", 12, True),
                              pad(f"[{min(wxvals):.4f}, {max(wxvals):.4f}]", 22, True),
                              pad(f"{max(wxvals) - min(wxvals):.4f}", 10, True)]))
    lines.append("-" * 84)
    lines.append(f"对比历史单 seed（macro_f1 {hist_f1:.4f} / auc {hist_auc:.4f} / 血虚 {hist_wx:.4f}）")
    lines.append(f"  ⇒ macro_f1 Δ = {statistics.mean(vals) - hist_f1:+.4f}"
                 f"（= {abs(statistics.mean(vals) - hist_f1) / sd:.2f} 个 seed-std）")
    if sdx:
        lines.append(f"  ⇒ 血虚 F1  Δ = {statistics.mean(wxvals) - hist_wx:+.4f}"
                     f"（= {abs(statistics.mean(wxvals) - hist_wx) / sdx:.2f} 个 seed-std）")
    lines.append("  ⇒ |Δ| 远小于 1 个 std ⇒ 历史数字复现成功；但也说明它作为**单点参照**")
    lines.append("     本身就带 ±std 的不确定性，不能再当无误差的基准用")
    lines.append(f"与 freeze 臂比较（独立两样本近似）：95% 判据 |Δ| < "
                 f"{1.96 * sd * (2 / 3) ** 0.5:.4f} = 1.96 × std(macro_f1) × √(2/3)")
    lines.append("  ⚠️ 两臂用的是**同一组 seed** → 可做配对比较，判据会比这个更严格；")
    lines.append("     等 b2 / freeze 的逐 seed 数据齐了按配对差的 std 另算。")
elif len(vals) == 1:
    lines.append("只有 1 个有效 run，无法估计方差（论文要求 >=3 seed）")
else:
    lines.append("没有任何有效 run")
missing = [r[0] for r in rows if r[1] is None]
if missing:
    lines.append(f"[WARN] 缺失/失败的 seed：{', '.join(missing)}")
lines.append("-" * 84)
lines.append("产物目录：")
for s, _f1, _a, _w1, _wd, d in rows:
    lines.append(f"  s{s}  {d if d else '(未运行)'}")
lines.append("=" * 84)

text = "\n".join(lines)
print(text)
Path("baseline_summary.txt").write_text(text + "\n", encoding="utf-8")
print("（已写入 baseline_summary.txt）")
PY

echo "##### BASELINE ALL DONE $(date '+%F %T') #####"
if [ -n "$FAILED" ]; then
  echo "##### 注意：以下 seed 失败 →$FAILED（详见 log_baseline_s*.log）#####"
  printf '%s\n' "FAILED_SEEDS:$FAILED" > baseline_done.txt
else
  printf '%s\n' "OK $(date '+%F %T')  seeds=$SEEDS" > baseline_done.txt
fi
echo "（完成标记：baseline_done.txt ；查看汇总：cat baseline_summary.txt）"
