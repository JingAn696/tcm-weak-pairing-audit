# -*- coding: utf-8 -*-
"""
舌诊古籍/图谱 PDF 结构探测 + 关键词命中图

用途
----
为论文的「标签空间覆盖性审计」寻找**独立外部参照**。给定若干舌诊 PDF，
逐本输出：
  1. 结构：页数 / 可提取文字页数 / 图片数 / 目录（TOC）
  2. 关键词命中图：我们关心的视觉维度与证候名，分别出现在哪些页、各多少处

依赖
----
    pymupdf   （隔离环境：C:\\Users\\think\\.workbuddy\\binaries\\python\\envs\\pdfenv）

用法
----
    python data/gen_tongue_book_probe.py                      # 默认处理内置书单
    python data/gen_tongue_book_probe.py --pdf a.pdf --pdf b.pdf
    python data/gen_tongue_book_probe.py --out report.txt

设计说明
--------
- **路径内置优先**：PowerShell 传中文参数会乱码，故默认书单硬编码在 DEFAULT_BOOKS。
- **不做 OCR**：只统计可提取文字。扫描版（无文字层）会明确标出 TEXT_LAYER = NO。
- 命中图输出「页号:次数」，便于直接翻页核对。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

try:
    import fitz  # pymupdf
except ImportError:  # pragma: no cover
    print("[FATAL] 需要 pymupdf。请用隔离环境运行：")
    print(r'  "C:\Users\think\.workbuddy\binaries\python\envs\pdfenv\Scripts\python.exe" '
          r'data/gen_tongue_book_probe.py')
    sys.exit(2)


BASE = r"D:\BaiduNetdiskDownload\舌诊\古代中医舌诊十大名著"

DEFAULT_BOOKS = [
    os.path.join(BASE, r"8、望舌识病图谱\《望舌识病图谱》第2版 费兆馥 顾亦棣著 人民卫生出版社2006.pdf"),
    os.path.join(BASE, r"8、望舌识病图谱\《舌诊图谱：观舌知健康》臧俊岐主编 江西科学技术出版社2018.pdf"),
    os.path.join(BASE, r"6、舌诊源鉴\《舌诊源鉴》王季藜 杨拴成主编 人民卫生出版社2001.pdf"),
    os.path.join(BASE, r"10、临床实用舌象图谱\《临床实用舌象图谱》王彦晖主编 化学工业出版社2012.pdf"),
]

# ---------------------------------------------------------------
# 关键词分组：围绕当前论文的 3 个待核实项 + 4 个维度缺口
# ---------------------------------------------------------------
KEYWORD_GROUPS = {
    "A_舌色维度": ["淡白舌", "淡白", "淡红舌", "红舌", "绛舌", "青紫舌", "紫舌"],
    "B_苔质维度": ["腻苔", "厚腻", "腐苔", "剥苔", "剥落苔", "花剥", "类剥", "光剥", "无苔", "少苔"],
    "C_舌形维度": ["胖大", "胖嫩", "齿痕", "裂纹", "点刺", "芒刺", "红点", "瘀点", "瘦薄"],
    "D_舌态维度": ["舌歪", "歪斜", "舌强", "强硬", "舌颤", "颤动", "痿软", "吐弄", "短缩", "舌纵"],
    "E_舌下络脉": ["舌下", "络脉", "舌脉", "舌底", "舌腹"],
    "F_证候名": ["气虚证", "血虚证", "阴虚证", "阳虚证", "气滞证", "血瘀证",
                 "痰湿证", "湿热证", "实热证", "津液亏虚证", "实寒证"],
    "G_内风相关": ["中风", "内风", "肝风", "风痰", "眩晕", "震颤"],
    "H_核心待核实": ["剥苔", "胖大舌", "淡白舌"],
}


def probe_book(path: str) -> dict:
    doc = fitz.open(path)
    n_pages = doc.page_count
    text_pages = 0
    img_count = 0
    page_texts = []
    for i in range(n_pages):
        p = doc[i]
        t = p.get_text() or ""
        page_texts.append(t)
        if len(t.strip()) > 20:
            text_pages += 1
        try:
            img_count += len(p.get_images(full=True))
        except Exception:
            pass
    toc = doc.get_toc() or []
    doc.close()
    return {
        "path": path,
        "name": os.path.basename(path),
        "n_pages": n_pages,
        "text_pages": text_pages,
        "n_images": img_count,
        "toc": toc,
        "page_texts": page_texts,
    }


def keyword_map(page_texts, keywords):
    hits = defaultdict(list)
    for i, t in enumerate(page_texts):
        if not t:
            continue
        for kw in keywords:
            c = t.count(kw)
            if c:
                hits[kw].append((i + 1, c))  # 1-based 页码
    return hits


def main():
    ap = argparse.ArgumentParser(description="舌诊书 PDF 结构探测 + 关键词命中图")
    ap.add_argument("--pdf", action="append", default=None, help="可重复；不传则用内置书单")
    ap.add_argument("--out", default=None, help="输出文本文件")
    args = ap.parse_args()

    books = args.pdf or DEFAULT_BOOKS
    lines = []
    lines.append("=" * 78)
    lines.append("舌诊书 PDF 结构探测 + 关键词命中图  (gen_tongue_book_probe.py)")
    lines.append("=" * 78)

    for bi, path in enumerate(books, 1):
        lines.append("")
        lines.append("-" * 78)
        lines.append(f"[{bi}] {os.path.basename(path)}")
        lines.append("-" * 78)
        if not os.path.isfile(path):
            lines.append("  [MISSING] 文件不存在")
            continue
        info = probe_book(path)
        lines.append(f"  体积      : {os.path.getsize(path)/1024/1024:.1f} MB")
        lines.append(f"  页数      : {info['n_pages']}")
        lines.append(f"  图片数    : {info['n_images']}")
        lines.append(f"  可提文字页: {info['text_pages']} / {info['n_pages']}"
                     f"   → TEXT_LAYER = {'YES' if info['text_pages'] > info['n_pages'] * 0.3 else 'NO(疑扫描版)'}")

        toc = info["toc"]
        lines.append(f"  目录条目  : {len(toc)}")
        if toc:
            lines.append("  --- TOC (前 40 条) ---")
            for lvl, title, pg in toc[:40]:
                lines.append(f"    {'  ' * (lvl - 1)}L{lvl} p{pg:<5} {title}")

        lines.append("  --- 关键词命中（页码:次数） ---")
        for gname, kws in KEYWORD_GROUPS.items():
            lines.append(f"  [{gname}]")
            for kw in kws:
                h = keyword_map(info["page_texts"], [kw])[kw]
                if not h:
                    continue
                total = sum(c for _, c in h)
                pages = ",".join(f"{p}:{c}" for p, c in h[:25])
                more = f" ...(+{len(h)-25}页)" if len(h) > 25 else ""
                lines.append(f"    {kw:<8} 总{total:>4}处  页 {pages}{more}")

    lines.append("")
    lines.append("=" * 78)
    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] 已写出 {args.out}  ({len(text)} chars)")
    else:
        print(text)


if __name__ == "__main__":
    main()
