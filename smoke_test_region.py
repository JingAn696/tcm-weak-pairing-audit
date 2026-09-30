"""Smoke test for train_tongue_region.py helpers (no dataset / no model / no GPU).

服务器部署后快速验证：
    cd /root/autodl-tmp/papers/code
    python smoke_test_region.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from training.train_tongue_region import aggregate_by_image, set_seed

# --- aggregate_by_image ---
# 2 images, 3 regions: img_b has 2 regions, img_a has 1
scores = np.array([
    [0.9, 0.1, 0.0],   # region 0 of img_b
    [0.2, 0.8, 0.0],   # region 1 of img_b
    [0.1, 0.2, 0.7],   # region 0 of img_a
], dtype=np.float32)
names = ["img_b.jpg", "img_b.jpg", "img_a.jpg"]
name_to_cats = {"img_a.jpg": (2,), "img_b.jpg": (0, 1)}

ys, yt = aggregate_by_image(scores, names, name_to_cats)

# sorted by name: img_a first, img_b second
assert ys.shape == (2, 3) and yt.shape == (2, 3)
# img_a: single region -> scores as-is; GT class 2
assert np.allclose(ys[0], [0.1, 0.2, 0.7]), ys[0]
assert yt[0].tolist() == [0, 0, 1]
# img_b: max-pool over 2 regions -> [max(0.9,0.2), max(0.1,0.8), 0] = [0.9, 0.8, 0]
assert np.allclose(ys[1], [0.9, 0.8, 0.0]), ys[1]
assert yt[1].tolist() == [1, 1, 0]
print("OK aggregate_by_image: max-pool + GT union + name-sorted alignment")

# determinism
ys2, yt2 = aggregate_by_image(scores, names, name_to_cats)
assert np.allclose(ys, ys2) and (yt == yt2).all()
print("OK aggregate_by_image: deterministic across calls")

# --- set_seed reproducibility ---
set_seed(42)
a = torch.randn(5)
set_seed(42)
b = torch.randn(5)
assert torch.equal(a, b)
print("OK set_seed: reproducible")

# --- module import surface ---
import training.train_tongue_region as m
assert hasattr(m, "main") and hasattr(m, "collect_scores_named")
print("OK module imports: main / collect_scores_named present")

print("ALL STRUCTURAL TESTS PASSED")
