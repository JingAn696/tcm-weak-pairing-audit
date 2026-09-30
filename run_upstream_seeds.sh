#!/usr/bin/env bash
# ============================================================================
# run_upstream_seeds.sh — Stage B 上游三段（从头重跑版）
# 版本：2026-09-16
# ----------------------------------------------------------------------------
# 目的：把「文本塔 / 视觉分支 / 对照原型」三段按**同一个 seed 集**串起来，
#       使 seed s 的一条链完整：文本塔 s → 视觉原型 s → 对照原型 s → 下游 s。
#       这是"全链路 seed 配对"的上半段；下半段是 run_seed_arm.sh。
#
# 用法：
#   nohup bash run_upstream_seeds.sh "42 43 44" > log_upstream_all.log 2>&1 &
#
#   第 2 参 STAGES（默认 "123"）= 只跑哪几段，用于**部分重跑**：
#     "123"  三段全跑（默认）
#     "23"   只重跑 视觉分支 + 对照原型，文本塔复用已有产物
#   2026-09-16 新增：修了 train_tongue_branch.py 的原型提取 bug（原型语料被
#   训练 loader 的 drop_last 随机丢 26 张）后，文本塔不受影响、无需重跑，
#   只需 `STAGES=23` 重跑 ②③。跳过的段会**复用已有产物**并在日志里显式标注。
#
# 产物：
#   runs/tcm_text_macbert_s@SEED@/    文本塔（HF checkpoint-*）+ test_report.json
#   runs/tongue_branch_s@SEED@/       视觉分支 best_model.pt
#                                     + tongue_prototypes_centered.pt（真原型）
#                                     + ..._shuffle.pt / ..._gaussian.pt（对照原型）
#   upstream_manifest.txt             实际产物清单（含**自动探测**到的 text checkpoint 路径）
#   upstream_done.txt                 完成标记
#
# 为什么不用硬编码 checkpoint-6630：
#   6630 是"3 epoch × 2210 步"推出来的，参数一变编号就变。这里用
#   `ls -d checkpoint-* | sort -V | tail -1` 取最新，缺了就回落到 best_model/，
#   两者都没有才报错 —— 避免把"路径猜错"伪装成"训练失败"。
#
# 阶段耗时（4090，**均未实测**，跑完请按实际值更新文档）：
#   文本塔   3 epoch × 2210 步        ≈ 20–40 min/seed
#   视觉分支 12 epoch（--crop_bbox）  ≈ 30–60 min/seed
# ============================================================================
set -u

SEEDS="${1:-42 43 44}"
# 2026-09-16 新增：可选第 2 参，只跑指定段（默认 "123" = 全跑）
STAGES="${2:-123}"
case "$STAGES" in
  *[!123]*|"") echo "[FATAL] STAGES 只能是 1/2/3 的组合（如 123 / 23），得到：$STAGES"; exit 3 ;;
esac

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

MANIFEST="upstream_manifest.txt"
: > "$MANIFEST"

echo "=============================================================="
echo " Stage B 上游重跑：文本塔 → 视觉分支 → 对照原型"
echo " seeds : $SEEDS"
echo " stages: $STAGES   （1=文本塔 2=视觉分支 3=对照原型；未含的段跳过并复用已有产物）"
echo " cwd   : $(pwd)"
echo " 开始  : $(date '+%F %T')"
echo "=============================================================="

# ---- 预检 ----
MISSING=""
for f in baselines/tcm_sd_text_macbert.py \
         training/train_tongue_branch.py \
         eval/gen_random_prototypes.py \
         utils/run_handoff.py \
         data/splits_v2/train.csv; do
  [ -f "$f" ] || MISSING="$MISSING $f"
done
if [ -n "$MISSING" ]; then
  echo "[FATAL] 缺文件：$MISSING"
  echo "        （漏传？先确认已在本目录解压整包并 sha1sum -c 通过）"
  exit 3
fi
echo "[预检] 脚本与数据齐备"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[预检] GPU：$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
fi
echo "--------------------------------------------------------------"

FAILED=""

for s in $SEEDS; do
  TDIR="runs/tcm_text_macbert_s$s"
  VDIR="runs/tongue_branch_s$s"

  # ---------- ① 文本塔（同时是论文的纯文本基线）----------
  echo ""
  if [[ "$STAGES" == *1* ]]; then
    echo "##### [$(date '+%F %T')] s$s ① 文本塔 → $TDIR #####"
    python baselines/tcm_sd_text_macbert.py --seed "$s" --output_dir "$TDIR" \
        > "log_text_s${s}.log" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "##### s$s ① 文本塔失败 rc=$RC #####"
      tail -n 25 "log_text_s${s}.log" | sed 's/^/    /'
      FAILED="$FAILED text_s$s"
      break
    fi
  else
    echo "##### [$(date '+%F %T')] s$s ① 文本塔：STAGES=$STAGES 未含 1 → 跳过（复用已有产物）#####"
  fi
  # ckpt 探测**无条件执行**：跳过 ① 时也要拿到路径写 manifest / 打印下游命令
  CK="$(ls -d "$TDIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  if [ -z "$CK" ]; then
    if [ -d "$TDIR/best_model" ]; then
      CK="$TDIR/best_model"
    else
      echo "[FATAL] s$s 找不到文本塔产物（既无 checkpoint-* 也无 best_model/）"
      FAILED="$FAILED textckpt_s$s"
      break
    fi
  fi
  echo "     text_ckpt = $CK"
  [ -n "${FIRST_CK_BASE:-}" ] || FIRST_CK_BASE="$(basename "$CK")"
  if [ -f "log_text_s${s}.log" ]; then
    grep -E "^(macro_f1|auc_ovr_macro|subset_accuracy|hamming_loss) :" \
        "log_text_s${s}.log" | sed 's/^/    /'
  else
    echo "    （无 log_text_s${s}.log，跳过文本指标回显；不影响结果）"
  fi

  # ---------- ② 视觉分支（B-1 判别力 + 真原型来源）----------
  if [[ "$STAGES" == *2* ]]; then
    echo "##### [$(date '+%F %T')] s$s ② 视觉分支 → $VDIR #####"
    python training/train_tongue_branch.py --crop_bbox --seed "$s" --output_dir "$VDIR" \
        > "log_branch_s${s}.log" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "##### s$s ② 视觉分支失败 rc=$RC #####"
      tail -n 25 "log_branch_s${s}.log" | sed 's/^/    /'
      FAILED="$FAILED branch_s$s"
      break
    fi
  else
    echo "##### [$(date '+%F %T')] s$s ② 视觉分支：STAGES=$STAGES 未含 2 → 跳过（复用已有原型）#####"
  fi
  PROTO="$VDIR/tongue_prototypes_centered.pt"
  if [ ! -f "$PROTO" ]; then
    echo "[FATAL] s$s 视觉分支跑完但没产出原型：$PROTO"
    FAILED="$FAILED proto_s$s"
    break
  fi
  echo "     proto = $PROTO"

  # ---------- ③ 对照原型（shuffle / gaussian）----------
  if [[ "$STAGES" != *3* ]]; then
    echo "##### [$(date '+%F %T')] s$s ③ 对照原型：STAGES=$STAGES 未含 3 → 跳过 #####"
  else
    echo "##### [$(date '+%F %T')] s$s ③ 对照原型 #####"
    for M in shuffle gaussian; do
      python eval/gen_random_prototypes.py \
          --proto "$PROTO" --mode "$M" --seed "$s" \
          --output "$VDIR/tongue_prototypes_centered_${M}.pt" \
          > "log_proto_${M}_s${s}.log" 2>&1
      RC=$?
      if [ "$RC" -ne 0 ]; then
        echo "##### s$s $M 原型失败 rc=$RC #####"
        tail -n 20 "log_proto_${M}_s${s}.log" | sed 's/^/    /'
        FAILED="$FAILED ${M}_s$s"
        break 2
      fi
      echo "     $M → $VDIR/tongue_prototypes_centered_${M}.pt"
      grep -E "有效原型个数|不变检查|置换范围" "log_proto_${M}_s${s}.log" | sed 's/^/       /'
    done
  fi

  echo "seed=$s  text_ckpt=$CK  proto=$PROTO" >> "$MANIFEST"
  echo "##### [$(date '+%F %T')] s$s ① ② ③ 全部完成 #####"
done

echo ""
echo "=============================================================="
if [ -n "$FAILED" ]; then
  printf 'FAILED_SEEDS:%s\n' "$FAILED" > upstream_done.txt
  echo "有失败：$FAILED"
  echo "⚠️ 上游不完整 → **不要**开下游训练（run_seed_arm.sh 的预检也会拦住）"
  echo "   先看对应 log_text_/log_branch_/log_proto_ 日志定位"
else
  printf 'OK %s  seeds=%s\n' "$(date '+%F %T')" "$SEEDS" > upstream_done.txt
  echo "上游全部完成 $(date '+%F %T')"
fi
echo "--------------------------------------------------------------"
echo "产物清单（$MANIFEST）："
cat "$MANIFEST"
echo "=============================================================="
echo "下一步：下游四条臂"
if [ -n "${FIRST_CK_BASE:-}" ]; then
  TPL="runs/tcm_text_macbert_s@SEED@/$FIRST_CK_BASE"
  echo "  探测到的 checkpoint 目录名：$FIRST_CK_BASE"
  echo "  TEXT_CKPT_TPL=$TPL"
  echo ""
  echo "  一条命令跑完四臂 × 3 seed（≈4 小时）："
  echo "  nohup bash -c \"bash run_seed_arm.sh freeze   '$TPL' '42 43 44' && \\"
  echo "                 bash run_seed_arm.sh b2       '$TPL' '42 43 44' && \\"
  echo "                 bash run_seed_arm.sh shuffle  '$TPL' '42 43 44' && \\"
  echo "                 bash run_seed_arm.sh gaussian '$TPL' '42 43 44'\" > log_arms_all.log 2>&1 &"
else
  echo "  ⚠️ 上游没有成功产物，无法给出下游命令"
fi
echo "=============================================================="
