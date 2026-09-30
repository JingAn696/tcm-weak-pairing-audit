"""
10 个证型 JSON → CSV 转换器
===========================

将净安校准的 10 个病性要素 JSON 文件合并为结构化 CSV，供训练时的"自构问诊增强"使用。

运行：
    cd papers/code/data
    python inquiries_converter.py

输出：
    - inquiries.csv          每个问诊条目一行
    - inquiries_long.csv     每个选项展开成一行（适合文本 baseline 训练）
    - syndrome_summary.csv   每个证型一行
    - tongue_expectation.csv  每个证型的舌象期望（13 维）
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import ast

# 10 个证型的中英文映射（2026-09-09 由 8 扩到 10，新增"内风""实热"）
SYNDROME_EN = {
    "气虚": "qi_deficiency",
    "血虚": "blood_deficiency",
    "阴虚": "yin_deficiency",
    "阳虚": "yang_deficiency",
    "气滞": "qi_stagnation",
    "血瘀": "blood_stasis",
    "痰湿": "phlegm_dampness",
    "湿热": "damp_heat",
    "内风": "internal_wind",
    "实热": "excess_heat",
}

# 文件名映射（中文 → 英文）
SYNDROME_FILE = {
    "气虚": "syndrome-qi-deficiency.json",
    "血虚": "syndrome-blood-deficiency.json",
    "阴虚": "syndrome-yin-deficiency.json",
    "阳虚": "syndrome-yang-deficiency.json",
    "气滞": "syndrome-qi-stagnation.json",
    "血瘀": "syndrome-blood-stasis.json",
    "痰湿": "syndrome-phlegm-dampness.json",
    "湿热": "syndrome-damp-heat.json",
    "内风": "syndrome-internal-wind.json",
    "实热": "syndrome-excess-heat.json",
}


def load_all_syndromes(data_dir: Path) -> List[Dict[str, Any]]:
    """加载 10 个 JSON 文件，返回统一格式的列表。"""
    syndromes = []
    for cn_name, filename in SYNDROME_FILE.items():
        path = data_dir / filename
        if not path.exists():
            print(f"⚠️  跳过（文件不存在）：{filename}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 补全英文键（如果缺失）
        data.setdefault("english", SYNDROME_EN.get(cn_name, cn_name))
        data["_filename"] = filename
        syndromes.append(data)
        print(f"✓ 已加载：{cn_name} ({filename})")
    return syndromes


def write_inquiries_csv(syndromes: List[Dict], out_path: Path) -> int:
    """
    每个问诊条目一行。
    返回写入的行数。
    """
    rows = []
    for syn in syndromes:
        syndrome_cn = syn["syndrome"]
        syndrome_en = syn["english"]
        for inq in syn.get("inquiries", []):
            rows.append({
                "syndrome_cn": syndrome_cn,
                "syndrome_en": syndrome_en,
                "inquiry_id": inq["id"],
                "inquiry_type": inq["type"],   # positive / negative / comorbid
                "anchor": inq["anchor"],        # ✓气虚 / →湿困 / →血虚
                "question": inq["question"],
                "options_str": " | ".join(inq["options"]),  # 用 | 分隔便于解析
                "n_options": len(inq["options"]),
                # 关键：把选项还原成 list 存为字符串（后续可解析）
                "options_list": str(inq["options"]),
            })
    fieldnames = [
        "syndrome_cn", "syndrome_en", "inquiry_id", "inquiry_type",
        "anchor", "question", "options_str", "n_options", "options_list"
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_inquiries_long_csv(syndromes: List[Dict], out_path: Path) -> int:
    """
    每个选项展开成一行（适合文本 baseline 训练）。
    例：气虚-第1问-选项"经常" → 一行
    """
    rows = []
    for syn in syndromes:
        syndrome_cn = syn["syndrome"]
        syndrome_en = syn["english"]
        for inq in syn.get("inquiries", []):
            for opt_idx, option in enumerate(inq["options"]):
                rows.append({
                    "syndrome_cn": syndrome_cn,
                    "syndrome_en": syndrome_en,
                    "inquiry_id": inq["id"],
                    "inquiry_type": inq["type"],
                    "anchor": inq["anchor"],
                    "question": inq["question"],
                    "option_idx": opt_idx,
                    "option_text": option,
                    # 给 baseline 一个二值标签：positive 类型 → 阳性答案
                    # 但这里我们没有"用户实际选择"，所以标签是候选项本身
                    "is_positive_anchor": "✓" in inq["anchor"],
                    "is_differential_probe": "→" in inq["anchor"],
                })
    fieldnames = [
        "syndrome_cn", "syndrome_en", "inquiry_id", "inquiry_type", "anchor",
        "question", "option_idx", "option_text",
        "is_positive_anchor", "is_differential_probe"
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_syndrome_summary_csv(syndromes: List[Dict], out_path: Path) -> int:
    """每个证型一行（含 pathogenesis、ratio、tongue_summary 等）。"""
    rows = []
    for syn in syndromes:
        rows.append({
            "syndrome_cn": syn["syndrome"],
            "syndrome_en": syn["english"],
            "version": syn.get("version", "v?"),
            "filled_by": syn.get("filled_by", "?"),
            "last_revised": syn.get("last_revised", "?"),
            "pathogenesis": syn.get("pathogenesis", ""),
            "tongue_high_region": syn.get("tongue_high_region", ""),
            "tongue_summary": syn.get("tongue_summary", ""),
            "dataset_expected_ratio": syn.get("dataset_expected_ratio", 0.0),
            "dataset_ratio_basis": syn.get("dataset_ratio_basis", ""),
            "confusable_syndromes": " | ".join(syn.get("confusable_syndromes", [])),
            "n_inquiries": len(syn.get("inquiries", [])),
            "n_tongue_features": len(syn.get("tongue_expectation", [])),
            "n_differential": len(syn.get("differential", [])),
        })
    fieldnames = [
        "syndrome_cn", "syndrome_en", "version", "filled_by", "last_revised",
        "pathogenesis", "tongue_high_region", "tongue_summary",
        "dataset_expected_ratio", "dataset_ratio_basis",
        "confusable_syndromes", "n_inquiries", "n_tongue_features", "n_differential"
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_tongue_expectation_csv(syndromes: List[Dict], out_path: Path) -> int:
    """每个证型的 13 维舌象期望拆成行。"""
    rows = []
    for syn in syndromes:
        for te in syn.get("tongue_expectation", []):
            rows.append({
                "syndrome_cn": syn["syndrome"],
                "syndrome_en": syn["english"],
                "dimension": te["dimension"],
                "value": te["value"],
                "confidence": te["confidence"],
            })
    fieldnames = ["syndrome_cn", "syndrome_en", "dimension", "value", "confidence"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    # 脚本所在目录 = papers/code/data/
    data_dir = Path(__file__).parent
    # JSON 在 papers/code/ 目录里（上一级）
    json_dir = data_dir.parent

    print(f"📂 JSON 目录：{json_dir}")
    print(f"📂 CSV 输出目录：{data_dir}\n")

    syndromes = load_all_syndromes(json_dir)
    if not syndromes:
        print("❌ 未加载到任何证型 JSON，请检查目录")
        return

    print(f"\n✅ 共加载 {len(syndromes)} 个证型\n")

    # 写 4 个 CSV
    n1 = write_inquiries_csv(syndromes, data_dir / "inquiries.csv")
    print(f"✓ inquiries.csv        ：{n1} 行")

    n2 = write_inquiries_long_csv(syndromes, data_dir / "inquiries_long.csv")
    print(f"✓ inquiries_long.csv   ：{n2} 行")

    n3 = write_syndrome_summary_csv(syndromes, data_dir / "syndrome_summary.csv")
    print(f"✓ syndrome_summary.csv ：{n3} 行")

    n4 = write_tongue_expectation_csv(syndromes, data_dir / "tongue_expectation.csv")
    print(f"✓ tongue_expectation.csv：{n4} 行")

    print(f"\n🎉 全部转换完成！")
    print(f"   净安校准的 10 病性要素 → 已就绪供训练 pipeline 使用。")


if __name__ == "__main__":
    main()