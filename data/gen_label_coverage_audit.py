# -*- coding: utf-8 -*-
"""标签空间覆盖性审计：TCM-Tongue 21 类 vs 专家文献的舌象描述维度。

用途
----
检验"视觉模态在中医证候分类上无增益"是否只是融合方法问题，
还是**基准数据的标签空间本身存在结构性缺口**。

方法
----
以《中医舌诊临床图解》（许家佗 主编，化学工业出版社，2017，ISBN 978-7-122-30552-7）
第三章第三节「常见基础证候的典型舌象特征」+ 第二章「舌诊的内容」+ 第三章第四节病例
为参照语料，抽取全部"舌象特征/常见舌象"描述段（锚点窗口），
按 5 个舌诊维度做关键词统计，并逐项标注 TCM-Tongue 21 类是否可表达。

⚠️ 口径警告（重要）
------------------
输出的是**关键词词频加权**比例，不是严格标签覆盖率。
"厚/薄/燥"等程度修饰占不可表达部分很大比重，但其严重性远低于"淡白舌"。
⇒ 论文中**不要直接引用这个百分比**，应按维度报告（舌色 2/5、舌形 5/5、舌态 0/1 …）。
详见 papers/notes/tongue-label-coverage-audit.md §7。

用法
----
    python data/gen_label_coverage_audit.py --pdf "D:/.../中医舌诊临床图解.pdf"
    python data/gen_label_coverage_audit.py --pdf <pdf> --out runs/coverage_audit.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

AUDIT_VERSION = "2026-09-15a"

# 锚点：三处章节用了不同的字段名
ANCHORS = ["舌象特征：", "常见舌象：", "【舌象特征】"]

# (维度标签, 关键词, 21 类对应情况)
GROUPS = [
    ("舌色 · 淡白", ["淡白", "枯白", "淡舌"], "无对应类"),
    ("舌色 · 淡红", ["淡红"], "无对应类"),
    ("舌色 · 红", ["舌红", "红舌", "偏红", "质红", "较红"], "红舌"),
    ("舌色 · 绛", ["红绛", "绛舌", "舌绛"], "无对应类(红舌近似)"),
    ("舌色 · 青紫/淡紫", ["青紫", "淡紫", "紫暗", "紫舌"], "紫舌"),
    ("舌色 · 局部红(尖/边)", ["舌尖红", "尖红", "边尖红", "舌边红"], "无对应类"),
    ("舌形 · 胖", ["胖"], "胖大舌"),
    ("舌形 · 瘦", ["瘦"], "瘦舌"),
    ("舌形 · 嫩", ["嫩"], "无对应类"),
    ("舌形 · 老", ["老"], "无对应类"),
    ("舌形 · 齿痕", ["齿痕"], "齿痕舌"),
    ("舌形 · 裂纹", ["裂纹"], "裂纹舌"),
    ("舌形 · 瘀斑", ["瘀斑"], "无对应类"),
    ("舌形 · 瘀点", ["瘀点"], "无对应类"),
    ("舌形 · 红点/点刺", ["红点", "点刺", "芒刺"], "红点舌"),
    ("舌态 · 歪斜/僵硬/痿软/震颤", ["歪斜", "僵硬", "痿软", "短缩", "吐弄", "震颤"], "无对应类"),
    ("舌下络脉", ["舌下络脉", "舌脉", "络脉"], "无对应类"),
    ("苔质 · 剥/少苔/无苔", ["剥苔", "苔剥", "少苔", "无苔", "光剥", "镜面"], "剥苔舌"),
    ("苔质 · 滑", ["滑"], "滑苔舌"),
    ("苔质 · 腻", ["腻"], "无对应类"),
    ("苔质 · 厚", ["厚"], "无对应类"),
    ("苔质 · 薄", ["薄"], "无对应类"),
    ("苔质 · 燥/干", ["燥", "干"], "无对应类"),
    ("苔色 · 白", ["苔白", "白苔", "白腻", "白厚", "白滑", "薄白", "白而"], "白苔舌"),
    ("苔色 · 黄", ["苔黄", "黄苔", "黄腻", "薄黄", "黄燥", "黄厚"], "黄苔舌"),
    ("苔色 · 黑", ["苔黑", "黑苔", "紫黑"], "黑苔舌"),
    ("苔色 · 灰", ["灰苔", "苔灰", "灰腻"], "无对应类"),
]

# 关键词匹配的已知噪声（论文引用时需说明）
CAVEATS = [
    "词频加权口径，非严格覆盖率",
    "'厚/薄/燥'等程度修饰约占不可表达部分的 44%，严重性低于'淡白舌'",
    "本书为典型舌象教科书，非随机临床分布，不能用于估计人群频次",
    "单本文献不足以代表领域，需再交叉 1-2 部权威文献",
]


def audit(pdf_path: str, sample_len: int = 300):
    import pymupdf  # 延迟导入，保持模块可被无 pymupdf 环境 import

    doc = pymupdf.open(pdf_path)
    pages = [(i + 1, p.get_text()) for i, p in enumerate(doc)]
    n_pages = doc.page_count
    doc.close()

    # 抽取锚点窗口（一个描述段 ≈ 锚点后 sample_len 字符）
    windows = []
    for pno, t in pages:
        for a in ANCHORS:
            k = t.find(a)
            while k >= 0:
                windows.append((pno, t[k:k + sample_len]))
                k = t.find(a, k + len(a))

    rows = []
    cover = miss = 0
    for name, kws, mapping in GROUPS:
        n = sum(1 for _, txt in windows if any(k in txt for k in kws))
        if n == 0:
            continue
        ok = not mapping.startswith("无对应类")
        cover, miss = (cover + n, miss) if ok else (cover, miss + n)
        rows.append((name, n, mapping, ok))

    # 病例规模（第三章第四节）
    diag = sum(t.count("西医诊断") + t.count("中医诊断")
               for pno, t in pages if 96 <= pno <= 171)
    case_analysis = sum(t.count("舌象分析") for _, t in pages)

    return {
        "n_pages": n_pages,
        "n_windows": len(windows),
        "rows": rows,
        "cover": cover,
        "miss": miss,
        "diag_hits": diag,
        "case_analysis": case_analysis,
        "window_pages": sorted({p for p, _ in windows}),
    }


def render(res, lines):
    w = lines.append
    w("=" * 78)
    w(f"舌象标签空间覆盖性审计（{AUDIT_VERSION}）")
    w("参照：许家佗 主编《中医舌诊临床图解》化学工业出版社 2017 / ISBN 978-7-122-30552-7")
    w("=" * 78)
    w(f"PDF 页数: {res['n_pages']} | 舌象描述段（锚点窗口）: {res['n_windows']}")
    w("")
    w(f"{'维度与特征':34s}{'出现次数':>8s}   21 类可否表达")
    w("-" * 78)
    for name, n, mapping, ok in res["rows"]:
        mark = "✅" if ok else "❌"
        w(f"{name:34s}{n:8d}   {mark} {mapping}")
    w("-" * 78)
    tot = res["cover"] + res["miss"]
    if tot:
        w(f"可表达 {res['cover']} / 不可表达 {res['miss']} / 合计 {tot}"
          f"  →  加权比例 {res['cover'] / tot * 100:.1f}%")
    w("")
    w("缺口按频次排序:")
    for name, n, mapping, ok in sorted(res["rows"], key=lambda x: -x[1]):
        if not ok:
            w(f"  {name:34s} {n:4d} 次")
    w("")
    w("⚠️ 方法学限制:")
    for c in CAVEATS:
        w(f"  - {c}")
    w("")
    w("### 配对病例规模（第三章第四节 p96-171）")
    w(f"'西医诊断'+'中医诊断' 出现次数: {res['diag_hits']}  → 估计 80-150 个配对病例")
    w(f"第三章第二节'舌象分析'出现次数: {res['case_analysis']}")
    w("")
    w("详见 papers/notes/tongue-label-coverage-audit.md")


def main():
    ap = argparse.ArgumentParser(description="舌象标签空间覆盖性审计")
    ap.add_argument("--pdf", required=True, help="《中医舌诊临床图解》PDF 路径")
    ap.add_argument("--out", default=None, help="输出 txt 路径（默认打印到 stdout）")
    args = ap.parse_args()

    res = audit(args.pdf)
    lines = []
    render(res, lines)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print("WROTE", args.out)
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
