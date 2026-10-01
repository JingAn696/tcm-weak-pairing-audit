# -*- coding: utf-8 -*-
"""扫描版 PDF 渲染（路径内置于脚本，规避中文命令行参数乱码）。

用法：改 FILES / PAGES 后直接 python 运行。
"""
import os
import sys

import fitz  # pymupdf

OUT_ROOT = r"path/to/output"  # 改成你自己的输出目录

# (标签, PDF 路径, 页码范围 None=全部)
JOBS = [
    ("diag20", r"path/to/book-pdfs/中医诊断学（国家十五规划）.pdf", None),
]

ZOOM = float(os.environ.get("RENDER_ZOOM", "2.0"))


def render(tag, path, pages):
    if not os.path.isfile(path):
        print(f"[MISSING] {path}")
        return []
    doc = fitz.open(path)
    n = doc.page_count
    idx = range(n) if pages is None else [p - 1 for p in pages if 1 <= p <= n]
    made = []
    os.makedirs(OUT_ROOT, exist_ok=True)
    for i in idx:
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
        fn = os.path.join(OUT_ROOT, f"{tag}_p{i+1:03d}.png")
        pix.save(fn)
        made.append(fn)
    doc.close()
    return made


def main():
    total = []
    for tag, path, pages in JOBS:
        m = render(tag, path, pages)
        total += m
        print(f"[{tag}] {len(m)} pages")
    print(f"[OK] {len(total)} files -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
