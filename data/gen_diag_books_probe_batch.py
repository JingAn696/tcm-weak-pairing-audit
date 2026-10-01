# -*- coding: utf-8 -*-
"""批量探测临时文件夹里几本"中医诊断学"的结构。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根

from data.gen_tongue_book_probe_diag import probe

BASE = r"path/to/book-pdfs"  # 改成你自己的 PDF 目录
FILES = [
    r"中医诊断学（国家十五规划）.pdf",
    r"10、【赠】中医诊断学（国家十五规划）.pdf",
    r"05中医诊断学.pdf",
    r"04、中医诊断学讲堂实录_李灿东.pdf",
]
KWS = ["舌诊", "舌色", "淡白舌", "舌态", "舌下络脉", "剥苔", "腻苔", "胖大"]

for fn in FILES:
    probe(f"{BASE}/{fn}", KWS, max_hits=15)
