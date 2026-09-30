"""
重切分 + 去重脚本（splits → splits_v2）
=========================================

背景（2026-09-11 泄漏诊断）：
- test 有 9.90%（522条）样本与 train 完全相同全文 → 模型背答案
- 其中 507/522 来自不同 user_id（跨用户复制样本，TCM-SD 数据集本身的重复）
- 504/522 标签与 train 一致 → 确认是同文本复制
- val 有 14.71%（305条）完全重复，user_id 泄漏也有（train∩val 6.32%）

修复（两步）：
1. 全局文本去重：完全相同 (chief_complaint+description+detection) 只保留一条
2. 按 user_id 分组切分：同一 user 的所有样本必须落在同一集合

输出：data/splits_v2/{train,val,test}.csv
验证：去重切分后
  - 三集之间 0 条完全重复文本
  - 三集之间 0 个重叠 user_id
  - 各证型标签占比与全量分布偏差 < 2 个百分点
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

# ---------- 配置 ----------
SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.85, 0.05   # test = 1 - 0.85 - 0.05 = 0.10
LABEL_COLS = [
    "qi_deficiency|气虚", "blood_deficiency|血虚", "yin_deficiency|阴虚",
    "yang_deficiency|阳虚", "qi_stagnation|气滞", "blood_stasis|血瘀",
    "phlegm_dampness|痰湿", "damp_heat|湿热", "wind|内风", "excess_heat|实热",
]
TEXT_FIELDS = ["chief_complaint", "description", "detection"]

# ---------- 路径 ----------
_SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS_DIR = _SCRIPT_DIR / "splits"
OUT_DIR = _SCRIPT_DIR / "splits_v2"


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    random.seed(SEED)

    # ---------- 1. 读取全部数据 ----------
    rows = []
    for name in ["train", "val", "test"]:
        p = SPLITS_DIR / f"{name}.csv"
        part = read_rows(p)
        rows.extend(part)
        print(f"  读 {name}: {len(part)} 行")
    print(f"  合计: {len(rows)} 行")

    # ---------- 2. 全局文本去重 ----------
    # key = (chief_complaint, description, detection) 完全相同 → 保留第一条
    # 标签冲突的重复（同文本不同标签）也只保留第一条，避免把冲突标签扩散
    seen = set()
    deduped = []
    n_dup = 0
    n_conflict = 0
    for r in rows:
        key = tuple(r.get(f, "") for f in TEXT_FIELDS)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        deduped.append(r)
    print(f"\n[去重] 删除完全重复文本 {n_dup} 条 → 保留 {len(deduped)} 条")

    # 重复里标签冲突统计（供论文报告）
    text2labels = defaultdict(set)
    for r in rows:
        key = tuple(r.get(f, "") for f in TEXT_FIELDS)
        text2labels[key].add(tuple(r[c] for c in LABEL_COLS))
    n_conflict = sum(1 for v in text2labels.values() if len(v) > 1)
    print(f"[去重] 不同标签的重复文本组: {n_conflict} 组（已按保留首条处理）")

    # ---------- 3. 按 user_id 分组切分 ----------
    # 同一 user 的全部样本 → 同一集合
    uid2rows = defaultdict(list)
    for r in deduped:
        uid2rows[r.get("user_id", "")].append(r)
    uids = sorted(uid2rows.keys())
    random.shuffle(uids)
    n_uid = len(uids)
    print(f"\n[切分] 去重后 user_id 数: {n_uid}")

    n_train_uid = int(n_uid * TRAIN_RATIO)
    n_val_uid = int(n_uid * VAL_RATIO)
    train_uids = set(uids[:n_train_uid])
    val_uids = set(uids[n_train_uid:n_train_uid + n_val_uid])
    test_uids = set(uids[n_train_uid + n_val_uid:])

    train_rows = [r for u in train_uids for r in uid2rows[u]]
    val_rows = [r for u in val_uids for r in uid2rows[u]]
    test_rows = [r for u in test_uids for r in uid2rows[u]]
    print(f"[切分] train={len(train_rows)}({len(train_uids)} users), "
          f"val={len(val_rows)}({len(val_uids)} users), test={len(test_rows)}({len(test_uids)} users)")

    # ---------- 4. 泄漏验证 ----------
    print(f"\n[验证] === 切分后泄漏检查 ===")

    # 4.1 完全重复文本
    def texts_of(rs):
        return set(tuple(r.get(f, "") for f in TEXT_FIELDS) for r in rs)
    tr_t, va_t, te_t = texts_of(train_rows), texts_of(val_rows), texts_of(test_rows)
    print(f"  train∩val 重复文本: {len(tr_t & va_t)}")
    print(f"  train∩test 重复文本: {len(tr_t & te_t)}")
    print(f"  val∩test 重复文本: {len(va_t & te_t)}")

    # 4.2 user_id 重叠
    print(f"  train∩val  user_id: {len(train_uids & val_uids)}")
    print(f"  train∩test user_id: {len(train_uids & test_uids)}")
    print(f"  val∩test   user_id: {len(val_uids & test_uids)}")

    # 4.3 标签分布对比（与全量比偏差）
    def label_ratio(rs):
        n = len(rs)
        return {c: sum(int(r[c]) for r in rs) / n for c in LABEL_COLS}
    all_ratio = label_ratio(deduped)
    print(f"\n[验证] 标签占比（vs 全量，偏差应 < 2pp）:")
    print(f"  {'病性':8s} {'全量':>8s} {'train':>8s} {'val':>8s} {'test':>8s}")
    bad = 0
    for name, tr_r, va_r, te_r in [
        ("train", label_ratio(train_rows), None, None),
        ("val", None, label_ratio(val_rows), None),
        ("test", None, None, label_ratio(test_rows)),
    ]:
        pass
    tr_ratio, va_ratio, te_ratio = label_ratio(train_rows), label_ratio(val_rows), label_ratio(test_rows)
    for c in LABEL_COLS:
        zh = c.split("|")[1]
        a, b, cc, d = all_ratio[c], tr_ratio[c], va_ratio[c], te_ratio[c]
        flag = ""
        if max(abs(b-a), abs(cc-a), abs(d-a)) * 100 > 2.0:
            flag = "  ⚠️ 偏差>2pp"
            bad += 1
        print(f"  {zh:6s} {a*100:7.2f}% {b*100:7.2f}% {cc*100:7.2f}% {d*100:7.2f}%{flag}")
    if bad == 0:
        print("  ✓ 全部病性分布偏差 < 2pp")

    # ---------- 5. 写出 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for name, rs in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        out = OUT_DIR / f"{name}.csv"
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rs)
        print(f"\n[写出] {out}（{len(rs)} 行）")

    print("\n" + "=" * 60)
    print(f"✅ 重切分完成：去重 {n_dup} 条 + 按 user_id 分组切分")
    print(f"   输出目录: {OUT_DIR}")
    print(f"   下一步: AutoDL 上传 splits_v2 → 重跑 baseline 1 验证真实数字")
    print("=" * 60)


if __name__ == "__main__":
    main()
