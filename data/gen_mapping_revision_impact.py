"""
映射矩阵 2026-09-15 修订的影响量化（旧版 2026-09-14 vs 新版 2026-09-15）
=====================================================================

目的
----
领域专家 2026-09-15 逐格复核后修订了 6 个格子（见 notes/mapping-open-questions.md）。
视觉分支的"原型"是**冻结 buffer**，矩阵一变、每张图的 10 病性标签就变、原型随之变，
因此必须先把**影响面**量出来，再决定是否值得重跑 B-2。

本脚本做三件事（纯标注解析，不需要 torch / PIL / 图像解码）：
  1. 逐 split 统计每个病性在**图像级**的正样本数：旧版 vs 新版(不含B6) vs 新版(含B6)
  2. 量化 B6 判据（剔除"孤立白苔舌"）的影响：多少张图被判为正常薄白
  3. 输出每格改动带来的 Δ（找出受影响最大的病性）

数据源：TCM-Tongue COCO json（自动探测路径，见 tongue_coco_loader.TONGUE_ROOT_CANDIDATES）
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from models.tongue_label_mapping import (  # noqa: E402
    HARD_MATRIX,
    MATRIX_VERSION,
    NUM_SYNDROMES,
    SYNDROMES_ZH,
    TONGUE_CLASSES,
    TONGUE_PINYIN_TO_ID,
    prune_normal_white_coating,
)

# ---------------------------------------------------------------------------
# 2026-09-14 旧版矩阵（硬编码快照，用于对比；勿改）
#   行 = 21 类舌象（COCO category id 顺序），列 = [气虚,血虚,阴虚,阳虚,气滞,血瘀,痰湿,湿热,内风,实热]
# ---------------------------------------------------------------------------
OLD_MATRIX_20260914 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0  健康舌
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 1  剥苔舌
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],  # 2  红舌
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 3  紫舌
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 4  胖大舌
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 5  瘦舌
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 6  红点舌
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 7  裂纹舌
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 8  齿痕舌
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # 9  白苔舌
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # 10 黄苔舌
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # 11 黑苔舌
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # 12 滑苔舌
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # 13 肾区凹陷
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 14 肾区凸起
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 15 肝胆区凹陷
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 16 肝胆区凸起
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 17 脾胃区凹陷
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 18 脾胃区凸起
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 19 心肺区凹陷
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 20 心肺区凸起
]

SPLITS = ("train", "val", "test")

TONGUE_ROOT_CANDIDATES = (
    str(Path(__file__).resolve().parent / "TCM-Tongue" / "shezhen_datasets1"
        / "shezhen datasets" / "shezhenv3_coco" / "shezhenv3-coco"),
    r"path/to/TCM-Tongue/shezhen_datasets1"
    r"/shezhen datasets/shezhenv3_coco/shezhenv3-coco",
    "/root/autodl-tmp/TCM-Tongue/shezhen_datasets1/shezhen datasets/shezhenv3-coco/shezhenv3-coco",
)


def resolve_root() -> Path:
    for c in TONGUE_ROOT_CANDIDATES:
        p = Path(c)
        if p.is_dir():
            return p
    return Path(TONGUE_ROOT_CANDIDATES[0])


def load_split_cats(root: Path, split: str):
    """返回 [frozenset(我们的 21 类索引), ...]，逐图一个。纯 json 解析。"""
    ann = root / split / "annotations" / f"{split}.json"
    with open(ann, encoding="utf-8") as f:
        coco = json.load(f)

    cat_map = {}
    for c in coco["categories"]:
        key = str(c["name"]).strip().lower()
        if key not in TONGUE_PINYIN_TO_ID:
            raise ValueError(f"未知类别名：{c['name']}")
        cat_map[c["id"]] = TONGUE_PINYIN_TO_ID[key]

    per_img = defaultdict(set)
    for a in coco["annotations"]:
        oid = cat_map.get(a["category_id"])
        if oid is not None:
            per_img[a["image_id"]].add(oid)

    out = []
    for im in coco["images"]:
        out.append(frozenset(per_img.get(im["id"], ())))
    return out


def vec_from_cats(cats, n=len(TONGUE_CLASSES)):
    v = [0.0] * n
    for c in cats:
        v[c] = 1.0
    return v


def syndrome_pos(det_vec, matrix):
    """21 维 0/1 → 10 维 0/1（硬映射）。"""
    out = [0] * NUM_SYNDROMES
    for i, present in enumerate(det_vec):
        if not present:
            continue
        for j in range(NUM_SYNDROMES):
            if matrix[i][j]:
                out[j] = 1
    return out


def main():
    root = resolve_root()
    print("=" * 78)
    print("映射矩阵 2026-09-15 修订影响量化（旧版 2026-09-14 vs 新版 %s）" % MATRIX_VERSION)
    print("=" * 78)
    print(f"数据根目录：{root}")
    print(f"存在性：{root.is_dir()}")
    if not root.is_dir():
        print("❌ 数据集不存在，无法量化。请用 --root 指定或检查候选路径。")
        return

    # 计数容器：三种口径
    cols = ["old", "new", "new_b6"]
    counts = {c: [0] * NUM_SYNDROMES for c in cols}
    n_img = {c: 0 for c in cols}
    n_img_total = 0
    n_isolated_white = 0
    n_white_total = 0
    # 空标签图（新版+B6 后被完全清空）
    n_empty_new_b6 = 0
    n_empty_old = 0

    for split in SPLITS:
        cats_list = load_split_cats(root, split)
        print(f"  [{split}] {len(cats_list)} 图")
        for cats in cats_list:
            n_img_total += 1
            v = vec_from_cats(cats)

            # 旧版
            p_old = syndrome_pos(v, OLD_MATRIX_20260914)
            # 新版（不含 B6）
            p_new = syndrome_pos(v, HARD_MATRIX)
            # 新版（含 B6：先剔除孤立白苔舌）
            v_b6 = vec_from_cats(prune_normal_white_coating(cats))
            p_b6 = syndrome_pos(v_b6, HARD_MATRIX)

            for j in range(NUM_SYNDROMES):
                if p_old[j]:
                    counts["old"][j] += 1
                if p_new[j]:
                    counts["new"][j] += 1
                if p_b6[j]:
                    counts["new_b6"][j] += 1

            if not any(p_new):
                n_img["new"] += 1
            if not any(p_b6):
                n_img["new_b6"] += 1
            if not any(p_old):
                n_img["old"] += 1

            # 2026-09-15b 修：旧条件 len(v_b6) != len(v) 恒为 False
            #（vec_from_cats 两种输入都返回定长 21 维），导致该计数永远为 0。
            # 正确判定：含白苔舌标签，且剔除后白苔舌标签已消失。
            if (TONGUE_PINYIN_TO_ID["baitaishe"] in cats
                    and TONGUE_PINYIN_TO_ID["baitaishe"]
                    not in prune_normal_white_coating(cats)):
                n_isolated_white += 1
            if TONGUE_PINYIN_TO_ID["baitaishe"] in cats:
                n_white_total += 1

    print("-" * 78)
    print("① 每病性「图像级正样本数」对比（全数据集 %d 图）" % n_img_total)
    print("-" * 78)
    hdr = f"{'病性':<6}{'旧版0914':>10}{'新版0915':>10}{'Δ':>8}{'新版+B6':>10}{'Δ(含B6)':>10}"
    print(hdr)
    print("-" * 78)
    for j, s in enumerate(SYNDROMES_ZH):
        o, n, nb = counts["old"][j], counts["new"][j], counts["new_b6"][j]
        print(f"{s:<6}{o:>10}{n:>10}{n - o:>+8}{nb:>10}{nb - o:>+10}")
    print("-" * 78)
    print(f"{'合计':<6}{sum(counts['old']):>10}{sum(counts['new']):>10}"
          f"{sum(counts['new']) - sum(counts['old']):>+8}{sum(counts['new_b6']):>10}"
          f"{sum(counts['new_b6']) - sum(counts['old']):>+10}")

    print()
    print("-" * 78)
    print("② B6 判据（剔除孤立白苔舌）的影响")
    print("-" * 78)
    print(f"  含白苔舌标签的图          ：{n_white_total}  ({n_white_total / n_img_total * 100:.1f}%)")
    print(f"  其中被判为「孤立白苔舌」  ：{n_isolated_white}  "
          f"({n_isolated_white / max(n_white_total, 1) * 100:.1f}% of 白苔舌图)")
    print(f"  即：白苔舌图中有 {n_isolated_white} 张被判为**正常薄白**、不再计入任何病性")

    print()
    print("-" * 78)
    print("③ 标签清空情况（某图在 10 病性上全阴性）")
    print("-" * 78)
    print(f"  旧版 2026-09-14          ：{n_img['old']:>6} 图")
    print(f"  新版 2026-09-15（不含 B6）：{n_img['new']:>6} 图")
    print(f"  新版 2026-09-15（含 B6）  ：{n_img['new_b6']:>6} 图")
    print("  （全阴性图不参与任何病性的正样本统计，但仍是训练样本；B6 会显著增多这类图）")

    print()
    print("=" * 78)
    print("解读提示")
    print("=" * 78)
    print("""
  · 「旧版0914」= 领域专家 2026-09-14 定稿矩阵；「新版0915」= 本次逐格复核后修订
  · Δ 为负的病性 = 视觉支撑被削弱；Δ 为正 = 被加强
  · 重点看 **血虚**：A2 决定删去「剥苔舌→血虚」后，血虚的视觉来源只剩「肝胆区凹陷」，
    若 Δ 为大幅负值，Stage B 对血虚的可支持性将显著下降（血虚恰是文本基线最弱类）
""")


if __name__ == "__main__":
    main()
