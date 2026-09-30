# -*- coding: utf-8 -*-
"""扫描版/文本版 PDF 结构探测：页数、文字层、图片数、目录关键词。

用法
----
    python data/gen_tongue_book_probe_diag.py --pdf "<路径>"
    python data/gen_tongue_book_probe_diag.py --pdf "<路径>" --grep 舌诊,舌色
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


def probe(path: str, keywords, max_hits: int = 40):
    print("=" * 78)
    print(f"FILE: {os.path.basename(path)}")
    print(f"PATH: {path}")
    if not os.path.isfile(path):
        print("[MISSING]")
        return
    print(f"SIZE: {os.path.getsize(path)/1024/1024:.1f} MB")
    doc = fitz.open(path)
    n = doc.page_count
    print(f"PAGES: {n}")

    # 抽样测文字层覆盖率
    step = max(1, n // 12)
    sampled = list(range(0, n, step))[:12]
    chars, imgs = 0, 0
    for i in sampled:
        pg = doc[i]
        chars += len(pg.get_text().strip())
        imgs += len(pg.get_images(full=True))
    print(f"TEXT LAYER: 抽样 {len(sampled)} 页共 {chars} 字符 "
          f"→ {'文本型' if chars > 200 * len(sampled) else '扫描版（无文字层）'}")
    print(f"IMAGES: 抽样 {len(sampled)} 页共 {imgs} 张")

    # 逐页图片统计（前 30 页）
    head = []
    for i in range(min(n, 30)):
        c = len(doc[i].get_images(full=True))
        head.append(f"p{i+1}:{c}")
    print("PER-PAGE IMG(前30): " + " ".join(head))

    if keywords:
        print("-" * 78)
        print(f"KEYWORD HITS: {keywords}")
        hits = []
        for i in range(n):
            t = doc[i].get_text()
            if not t:
                continue
            for kw in keywords:
                if kw in t:
                    hits.append((i + 1, kw))
                    break
            if len(hits) >= max_hits:
                break
        if hits:
            for pno, kw in hits:
                print(f"   p{pno:>4}  {kw}")
        else:
            print("   （无命中）")
    doc.close()
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--grep", default=None, help="逗号分隔关键词")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kws = [s.strip() for s in args.grep.split(",")] if args.grep else None

    if args.out:
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            probe(args.pdf, kws)
        finally:
            sys.stdout = old
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print(f"[OK] written {args.out}")
    else:
        probe(args.pdf, kws)


if __name__ == "__main__":
    main()
