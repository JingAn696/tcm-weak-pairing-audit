"""Batch OCR for 当当云阅读 (com.dangdang.reader) screenshots of a book.

Usage:
    python data/gen_ocr_book_screenshots.py --in_dir "path/to/book" --out_txt runs/book_ocr.txt

Outputs one text block per image in chronological (filename) order.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR


def main():
    ap = argparse.ArgumentParser(description="OCR 手机阅读器截图并输出纯文本")
    ap.add_argument("--in_dir", required=True, help="截图所在目录")
    ap.add_argument("--out_txt", required=True, help="输出 txt 路径")
    ap.add_argument("--ext", default=".jpg,.jpeg,.png", help="图片扩展名（逗号分隔）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 张（0=全部）")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    exts = tuple(e.strip().lower() for e in args.ext.split(","))
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts],
                   key=lambda p: p.stem)

    if args.limit:
        files = files[:args.limit]

    print(f"[OCR] 发现 {len(files)} 张图片，目录：{in_dir.resolve()}")
    if not files:
        sys.exit(0)

    # init once
    engine = RapidOCR()

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write(f"# OCR 输出 | 目录: {in_dir.resolve()}\n")
        fw.write(f"# 图片数: {len(files)}\n\n")

        for idx, p in enumerate(files, 1):
            t0 = time.time()
            try:
                res, _ = engine(p)
            except Exception as e:
                fw.write(f"\n## [{idx:04d}] {p.name}  OCR_FAILED: {e}\n\n")
                continue

            # res is list of tuples: (box, text, score)
            if res is None:
                text = ""
            else:
                lines = []
                for item in res:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        txt = str(item[1])
                        lines.append(txt)
                    else:
                        lines.append(str(item))
                text = "\n".join(lines)

            # clean: collapse multiple empty lines
            text = re.sub(r"\n{3,}", "\n\n", text)

            header = f"\n## [{idx:04d}] {p.name}\n"
            fw.write(header)
            fw.write(text.strip() if text.strip() else "(无文字)")
            fw.write("\n")

            elapsed = time.time() - t0
            if idx % 10 == 0 or idx <= 3:
                print(f"  [{idx:04d}/{len(files):04d}] {p.name} ({elapsed:.2f}s)")

    print(f"[OCR] 完成，输出：{out_path.resolve()}")


if __name__ == "__main__":
    main()
