#!/usr/bin/env bash
# ============================================================================
# run_seed_arm.sh — Stage B：seed 重复实验批量跑（第 5 / 6 步通用）
# 版本：2026-09-15d
# ----------------------------------------------------------------------------
# 为什么要有这个文件（2026-09-15 教训）：
#   之前把"创建脚本"和"运行脚本"分成两条消息给，中间被打断 →
#   服务器上根本没有 .sh，`nohup bash run_freeze_seeds.sh ...` 直接
#   exit 127（command not found：文件不存在）。现在把它当仓库文件管理：
#   上传一次，以后每个臂都只用一条命令。
#
# 用法：
#   bash run_seed_arm.sh <arm> <text_ckpt_tpl> [seeds] [epochs] [proto_dir_tpl]
#
#   arm            freeze | b2 | shuffle | gaussian
#   text_ckpt_tpl  MacBERT 文本塔，**可含 @SEED@ 占位符**
#                  例：runs/tcm_text_macbert_s@SEED@/checkpoint-6630  ← 全链路配对用法
#                  不含占位符时全部 seed 共用同一个（旧用法，仍兼容）
#   seeds          默认 "42 43 44"
#   epochs         默认 12
#   proto_dir_tpl  原型目录，**可含 @SEED@**，默认 runs/tongue_branch_s@SEED@
#                  仅 b2 / shuffle / gaussian 用；freeze 不需要
#
# 【占位符为什么是 @SEED@ 而不是 {seed}】
#   Git-Bash/MSYS 的路径转换层会吞掉 `{`，导致模板在传参途中被破坏（2026-09-16 实测）。
#   @SEED@ 不含任何 shell 特殊字符 → 粘贴时**连引号都可以不加**，也不会被路径转换干扰。
#   `{seed}` 仍兼容（Linux 服务器上等效），但不推荐。
#
# 【2026-09-16 · 全链路 seed 配对】把占位符写进两个模板后，
#   seed s 的那一次运行 = 文本塔 s + 视觉原型 s + 下游 s，是**一次完整的独立重复**，
#   而不是"固定上游、只换下游随机种子"。四条臂都这么做，臂间才能逐 seed 配对比较。
#   原型文件按约定命名（同目录）：
#     真原型     tongue_prototypes_centered.pt
#     shuffle    tongue_prototypes_centered_shuffle.pt
#     gaussian   tongue_prototypes_centered_gaussian.pt
#
# 四个臂的唯一变量是 --visual_mode：
#   freeze    对照臂，--visual_mode placeholder，零视觉信息，不读原型文件
#   b2        真原型臂，--visual_mode proto_attn + 真实原型（tongue_prototypes_centered.pt）
#   shuffle   阴性对照，原型「向量集合相同、对应关系打乱」
#   gaussian  阴性对照，高斯随机原型（可选；shuffle 已够回答审稿人）
#   ⚠️ 四个臂全部带 --freeze_text（冻结 MacBERT）。这是 Stage B 的既定协议，
#      与 runbook §14 / 上机手册第 5、6 步逐字一致；漏掉就不再可比。
#
# 例（后台跑，freeze 臂约 3 小时）：
#   nohup bash run_seed_arm.sh freeze runs/tcm_text_macbert/checkpoint-6630 \
#         > log_freeze_all.log 2>&1 &
#
# 产物：
#   runs/full_model_v2_freeze_e<ep>_s<seed>/      每 seed 一个目录（含 test_report.json）
#   runs/full_model_v2_b2_e<ep>_s<seed>/          命名与手册第 5 步一致
#   runs/full_model_v2_b2_shuffle_s<seed>/        命名与手册第 6 步一致
#   runs/full_model_v2_b2_gaussian_s<seed>/
#   log_<arm>_s<seed>.log                         每 seed 完整日志（含 HANDOFF 摘要）
#   arm_summary_<arm>.txt                         本脚本汇总的 seed 表（贴回给 Buddy 用）
#   arm_done_<arm>.txt                            完成标记
#
# 怎么看进度（Ctrl+C 安全，不会杀掉后台任务）：
#   tail -f log_freeze_all.log     总进度
#   tail -n 30 log_freeze_s42.log  当前 seed 尾部
#   nvidia-smi                     另开窗口看显存
# ============================================================================
set -u

ARM="${1:-}"
CKPT_TPL="${2:-}"
SEEDS="${3:-42 43 44}"
EPOCHS="${4:-12}"
PROTO_DIR_TPL="${5:-runs/tongue_branch_s@SEED@}"

# 展开 seed 占位符（bash 内建替换，不依赖 sed/awk）
expand_seed() {
  local tpl="$1" s="$2"
  tpl="${tpl//@SEED@/$s}"
  tpl="${tpl//\{seed\}/$s}"
  printf '%s' "$tpl"
}

PROTO_NAME_REAL="tongue_prototypes_centered.pt"
PROTO_NAME_SHUFFLE="tongue_prototypes_centered_shuffle.pt"
PROTO_NAME_GAUSS="tongue_prototypes_centered_gaussian.pt"

# ---- 无论从哪里调用，都切到脚本所在目录（否则相对路径全错）----
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

if [ -z "$ARM" ] || [ -z "$CKPT_TPL" ]; then
  echo "用法：bash run_seed_arm.sh <freeze|b2|shuffle|gaussian> <text_ckpt_tpl> [seeds] [epochs] [proto_dir_tpl]"
  echo "例  ：bash run_seed_arm.sh b2 runs/tcm_text_macbert_s@SEED@/checkpoint-6630 '42 43 44' 12"
  exit 2
fi

# ---- 臂定义：四个臂都带 --freeze_text，只有 visual_mode / proto 不同 ----
# proto 的**完整路径按 seed 解析**（在循环里展开），这里只决定用哪个原型文件名
case "$ARM" in
  freeze)   VMODE="placeholder"; PROTO_NAME="" ;;
  b2)       VMODE="proto_attn";  PROTO_NAME="$PROTO_NAME_REAL" ;;
  shuffle)  VMODE="proto_attn";  PROTO_NAME="$PROTO_NAME_SHUFFLE" ;;
  gaussian) VMODE="proto_attn";  PROTO_NAME="$PROTO_NAME_GAUSS" ;;
  *) echo "未知 arm：$ARM（应为 freeze|b2|shuffle|gaussian）"; exit 2 ;;
esac

# ---- 输出目录命名：唯一来源，避免 shell 与汇总表两处写法不一致 ----
out_dir_for() {   # $1=seed
  case "$ARM" in
    freeze)   echo "runs/full_model_v2_freeze_e${EPOCHS}_s$1" ;;
    b2)       echo "runs/full_model_v2_b2_e${EPOCHS}_s$1" ;;
    shuffle)  echo "runs/full_model_v2_b2_shuffle_s$1" ;;
    gaussian) echo "runs/full_model_v2_b2_gaussian_s$1" ;;
  esac
}

FIRST_SEED="$(printf '%s' "$SEEDS" | awk '{print $1}')"
FIRST_CKPT="$(expand_seed "$CKPT_TPL" "$FIRST_SEED")"
FIRST_PDIR="$(expand_seed "$PROTO_DIR_TPL" "$FIRST_SEED")"

echo "=============================================================="
echo " arm           : $ARM"
echo " seeds         : $SEEDS"
echo " epochs        : $EPOCHS"
echo " text_ckpt_tpl : $CKPT_TPL"
echo " proto_dir_tpl : $PROTO_DIR_TPL${PROTO_NAME:+   （原型 $PROTO_NAME）}"
echo " visual_mode   : $VMODE"
echo " 展开示例 s$FIRST_SEED : $FIRST_CKPT"
echo " cwd           : $(pwd)"
echo " 开始          : $(date '+%F %T')"
echo "=============================================================="

# ---- 预检：缺文件就别浪费 GPU 时间（按第一个 seed 展开后检查）----
[ -e "$FIRST_CKPT" ] || { echo "[FATAL] 找不到 text checkpoint：$FIRST_CKPT（模板 $CKPT_TPL）"; exit 3; }
[ -f "training/train_full_model.py" ] || { echo "[FATAL] 找不到 training/train_full_model.py（cwd 不对？）"; exit 3; }
[ -f "utils/run_handoff.py" ] || { echo "[FATAL] 找不到 utils/run_handoff.py（漏传 utils/ 目录？）"; exit 3; }
[ -f "data/splits_v2/train.csv" ] || { echo "[FATAL] 找不到 data/splits_v2/train.csv"; exit 3; }

if [ -n "$PROTO_NAME" ]; then
  _P="$FIRST_PDIR/$PROTO_NAME"
  [ -f "$_P" ] || { echo "[FATAL] 找不到原型文件：$_P"; exit 3; }
  echo "[预检] 原型文件存在（s$FIRST_SEED）：$_P"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[预检] GPU：$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
fi
echo "[预检] 通过 $(date '+%F %T')"
echo "--------------------------------------------------------------"

FAILED=""
PAIRS=""
FIRST="$FIRST_SEED"

for s in $SEEDS; do
  OUT="$(out_dir_for "$s")"
  PAIRS="$PAIRS $s:$OUT"
  LOG="log_${ARM}_s${s}.log"

  # ---- 按 seed 解析这一次运行的文本塔与原型（全链路配对的关键）----
  CKPT_S="$(expand_seed "$CKPT_TPL" "$s")"
  PROTO_S=""
  EXTRA="--freeze_text"
  if [ -n "$PROTO_NAME" ]; then
    PROTO_S="$(expand_seed "$PROTO_DIR_TPL" "$s")/$PROTO_NAME"
    EXTRA="--freeze_text --proto_path $PROTO_S"
  fi

  echo "##### [$(date '+%F %T')] $ARM seed=$s 开始 → $OUT #####"
  echo "      text  = $CKPT_S"
  [ -z "$PROTO_S" ] || echo "      proto = $PROTO_S"

  # 逐 seed 再查一次：上游只要缺某个 seed 的产物，立刻停手（别跑到一半才发现）
  [ -e "$CKPT_S" ] || { echo "[FATAL] seed=$s 的 text ckpt 不存在：$CKPT_S"; FAILED="$FAILED $s"; break; }
  if [ -n "$PROTO_S" ] && [ ! -f "$PROTO_S" ]; then
    echo "[FATAL] seed=$s 的原型不存在：$PROTO_S"; FAILED="$FAILED $s"; break
  fi

  python training/train_full_model.py \
      --text_checkpoint "$CKPT_S" \
      --visual_mode "$VMODE" $EXTRA \
      --epochs "$EPOCHS" \
      --seed "$s" \
      --output_dir "$OUT" \
      > "$LOG" 2>&1
  RC=$?

  if [ "$RC" -eq 0 ]; then
    echo "##### [$(date '+%F %T')] $ARM seed=$s 结束 OK (rc=0) #####"
    grep -E "^(macro_f1|auc_ovr_macro|subset_accuracy|hamming_loss) :" "$LOG" | sed 's/^/    /'
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "    GPU：$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    fi
  else
    echo "##### [$(date '+%F %T')] $ARM seed=$s 失败 rc=$RC #####"
    echo "    —— $LOG 尾部 25 行 ——"
    tail -n 25 "$LOG" | sed 's/^/    /'
    FAILED="$FAILED $s"
    if [ "$s" = "$FIRST" ]; then
      echo "[ABORT] 第 1 个 seed 就失败 → 配置有问题，终止后续 seed，避免白烧 GPU"
      break
    fi
  fi
done

# ---- 汇总表（与 eval/compare_runs.py --agg 互为交叉校验）----
python - "$ARM" "$EPOCHS" "$SEEDS" $PAIRS <<'PY' || echo "[WARN] 汇总表生成失败（不影响上面的训练结果）"
import json, statistics, sys, unicodedata
from pathlib import Path

def _w(s):
    """显示宽度：中文算 2 列（与 utils/run_handoff.py 同一套算法）"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))

def pad(s, n, right=False):
    s = str(s)
    k = n - _w(s)
    return (" " * max(k, 0) + s) if right else (s + " " * max(k, 0))

arm, ep = sys.argv[1], sys.argv[2]
intended = sys.argv[3].split()
pairs = {}
for a in sys.argv[4:]:
    s, d = a.split(":", 1)
    pairs[s] = d

def fmt(x, nd=4):
    return "  ——  " if x is None else f"{x:.{nd}f}"

rows = []
for s in intended:
    d = pairs.get(s)
    if d is None:                                   # 前一个 seed 失败被 ABORT，这个没跑
        rows.append((s, None, None, None, None, None))
        continue
    rp = Path(d) / "test_report.json"
    if not rp.exists():
        rows.append((s, None, None, None, None, d))
        continue
    try:
        # utf-8-sig：容忍 BOM（Windows 编辑器往返会带 BOM，json 会直接报错）
        r = json.loads(rp.read_text(encoding="utf-8-sig"))
    except Exception as e:                          # 单个报告坏了不该毁掉整张表
        print(f"[WARN] {rp} 读取失败：{type(e).__name__}: {e}")
        rows.append((s, None, None, None, None, d))
        continue
    t = r.get("test", {}) or {}
    w = r.get("watch_syndrome", {}) or {}
    wf1, wb = w.get("f1"), w.get("text_baseline_f1")
    rows.append((s, t.get("macro_f1"), t.get("auc_ovr_macro"), wf1,
                 (wf1 - wb) if (wf1 is not None and wb is not None) else None, d))

lines = []
lines.append("=" * 84)
lines.append(f" arm_summary · arm={arm}  epochs={ep}  seeds={intended}")
lines.append("=" * 84)
lines.append("".join([pad("seed", 8), pad("macro_f1", 12, True),
                      pad("auc_ovr_macro", 16, True), pad("血虚 F1", 12, True),
                      pad("血虚 Δ", 12, True)]))
for s, f1, auc, wf1, wd, _d in rows:
    wds = "  ——  " if wd is None else f"{wd:+.4f}"
    lines.append("".join([pad(s, 8), pad(fmt(f1), 12, True), pad(fmt(auc), 16, True),
                          pad(fmt(wf1), 12, True), pad(wds, 12, True)]))
vals = [r[1] for r in rows if r[1] is not None]
if len(vals) >= 2:
    lines.append("-" * 84)
    lines.append(f"mean     {statistics.mean(vals):.4f}    "
                 f"std(sample,n-1) {statistics.stdev(vals):.4f}    "
                 f"range [{min(vals):.4f}, {max(vals):.4f}]")
    lines.append(f"seed 离散度：极差 {max(vals) - min(vals):.4f}"
                 f"（若 >=0.019 说明单次运行的 Δ 不可解读，必须多 seed）")
elif len(vals) == 1:
    lines.append("只有 1 个有效 run，无法估计方差（论文要求 >=3 seed）")
else:
    lines.append("没有任何有效 run")
missing = [r[0] for r in rows if r[1] is None]
if missing:
    lines.append(f"[WARN] 缺失/失败的 seed：{', '.join(missing)}")
lines.append("-" * 84)
lines.append("产物目录（可直接作为 eval/compare_runs.py 的参数）：")
for s, _f1, _a, _w1, _wd, d in rows:
    lines.append(f"  s{s}  {d if d else '(未运行)'}")
lines.append("=" * 84)

text = "\n".join(lines)
print(text)
out = Path(f"arm_summary_{arm}.txt")
out.write_text(text + "\n", encoding="utf-8")
print(f"（已写入 {out}）")
PY

echo "##### ARM=$ARM ALL DONE $(date '+%F %T') #####"
if [ -n "$FAILED" ]; then
  echo "##### 注意：以下 seed 失败 →$FAILED（详见对应 log_${ARM}_s*.log）#####"
  printf '%s\n' "FAILED_SEEDS:$FAILED" > "arm_done_${ARM}.txt"
else
  printf '%s\n' "OK $(date '+%F %T')  seeds=$SEEDS" > "arm_done_${ARM}.txt"
fi
echo "（完成标记：arm_done_${ARM}.txt ；贴回给我：cat arm_summary_${ARM}.txt）"
