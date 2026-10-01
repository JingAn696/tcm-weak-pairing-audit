# -*- coding: utf-8 -*-
"""
把扫描版舌诊书的指定页渲染为 PNG，供多模态阅读（内置书单，避免中文参数乱码）。

用法
----
    python data/gen_tongue_book_render.py --plan toc         # 渲染各书目录页
    python data/gen_tongue_book_render.py --book 3 --pages 10-15
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import fitz
except ImportError:
    print("[FATAL] 需要 pymupdf（隔离环境 pdfenv）")
    sys.exit(2)

BASE = r"path/to/book-pdfs"  # 改成你自己的古籍 PDF 目录
BOOKS = {
    1: os.path.join(BASE, r"8、望舌识病图谱\《望舌识病图谱》第2版 费兆馥 顾亦棣著 人民卫生出版社2006.pdf"),
    2: os.path.join(BASE, r"8、望舌识病图谱\《舌诊图谱：观舌知健康》臧俊岐主编 江西科学技术出版社2018.pdf"),
    3: os.path.join(BASE, r"6、舌诊源鉴\《舌诊源鉴》王季藜 杨拴成主编 人民卫生出版社2001.pdf"),
    4: os.path.join(BASE, r"10、临床实用舌象图谱\《临床实用舌象图谱》王彦晖主编 化学工业出版社2012.pdf"),
}

OUT_ROOT = r"path/to/output"  # 改成你自己的输出目录

# 要渲染的页（1-based，含端点）——先按各书 TOC 声明的位置猜
PLANS = {
    "toc": {
        1: (8, 12),
        2: (6, 12),
        3: (10, 21),
        4: (3, 8),
    },
}


def parse_pages(spec: str):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None)
    ap.add_argument("--book", type=int, default=None)
    ap.add_argument("--pages", default=None)
    ap.add_argument("--zoom", type=float, default=2.0)
    args = ap.parse_args()

    jobs = {}
    if args.plan:
        jobs = dict(PLANS[args.plan])
    elif args.book and args.pages:
        pages = parse_pages(args.pages)
        jobs = {args.book: (min(pages), max(pages))}
    else:
        ap.error("需要 --plan 或 (--book + --pages)")

    os.makedirs(OUT_ROOT, exist_ok=True)
    made = []
    for bi, (a, b) in jobs.items():
        path = BOOKS[bi]
        if not os.path.isfile(path):
            print(f"[MISSING] book {bi}")
            continue
        doc = fitz.open(path)
        for pno in range(a, b + 1):
            if pno < 1 or pno > doc.page_count:
                continue
            pix = doc[pno - 1].get_pixmap(matrix=fitz.Matrix(args.zoom, args.zoom))
            fn = os.path.join(OUT_ROOT, f"b{bi}_p{pno:03d}.png")
            pix.save(fn)
            made.append(fn)
        doc.close()
    print(f"[OK] rendered {len(made)} pages -> {OUT_ROOT}")
    for m in made:
        print("   ", m)


if __name__ == "__main__":
    main()
