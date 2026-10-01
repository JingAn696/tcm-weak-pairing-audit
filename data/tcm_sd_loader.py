"""
tcm_sd_loader.py
================
TCM-SD 数据加载器 - 把原始 JSONL 转成训练可用的 CSV

流程：
1. 读取 TCM-SD 的 train.json / dev.json
2. 用 syndrome_mapping.map_syndrome 映射到 10 病性
3. 对待手动拆解的样本（manual_needed），用 auto_disambiguate 拆解
4. 数据切分：原 train 切 5% 为新 val，原 dev 作为新 test
5. 输出：train.csv / val.csv / test.csv（每个含 10 维 label 向量）

切分方案（领域专家 2026-09-08 拍板）：
  原作者：train (43,180) + dev (5,486) + test (5,486, 隐藏)
  我们的方案：
    新 train = 原 train 切 95%  → ~41,021
    新 val   = 原 train 切 5%   → ~2,159
    新 test  = 原 dev 整集       → 5,486
  完全在原作者数据范围内，遵循 ZY-BERT 原始 train/dev 划分。
"""

import json
import csv
import sys
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.syndrome_mapping import (
    map_syndrome, SYNDROMES_10, SYNDROME_CN
)
from data.auto_disambiguate import infer_labels as disambiguate


# === 1. 数据读取 ===

def read_jsonl(path: Path) -> List[dict]:
    """读取 JSONL 文件，每行一个 JSON 对象"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# === 2. 病性映射 ===

def get_labels(sample: dict) -> Tuple[List[str], str]:
    """
    对一条样本映射病性标签。
    返回：(labels, status)
    status:
      - 'auto'                    : 自动映射成功
      - 'manual_disambiguated'    : 待手动拆解 → 自动拆解成功
      - 'manual_no_evidence'      : 待手动拆解 → 无明确证据
      - 'external_excluded'       : 外感病（排除）
      - 'unmapped'                : 边角样本（无任何映射）
    """
    syndrome = sample.get('syndrome', '')
    r = map_syndrome(syndrome)

    if r['status'] == 'external_excluded':
        return [], 'external_excluded'

    if r['status'] == 'unmapped':
        return [], 'unmapped'

    if r['status'] == 'auto':
        return r['labels'], 'auto'

    if r['status'] == 'manual_needed':
        labels, _ = disambiguate(sample)
        if labels:
            return labels, 'manual_disambiguated'
        return [], 'manual_no_evidence'

    return [], 'unknown'


# === 3. 数据切分 ===

def split_train_val(train_data: List[dict],
                    val_ratio: float = 0.05,
                    seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """
    把原 train 切成 train + val（默认 95/5）。
    遵循 ZY-BERT 原始 train 划分（领域专家 2026-09-08 拍板）。
    """
    random.seed(seed)
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    n_val = int(len(train_data) * val_ratio)

    val_set = set(indices[:n_val])
    new_train = [train_data[i] for i in range(len(train_data)) if i not in val_set]
    new_val = [train_data[i] for i in range(len(train_data)) if i in val_set]

    return new_train, new_val


# === 4. CSV 输出 ===

LABEL_HEADERS = SYNDROMES_10  # 10 病性要素的英文 key
LABEL_HEADERS_CN = [SYNDROME_CN[l] for l in SYNDROMES_10]


def labels_to_vector(labels: List[str]) -> List[int]:
    """把 labels 列表转成 10 维 0/1 向量（按 SYNDROMES_10 顺序）"""
    vec = [0] * len(SYNDROMES_10)
    for l in labels:
        if l in SYNDROMES_10:
            idx = SYNDROMES_10.index(l)
            vec[idx] = 1
    return vec


def write_csv(samples: List[dict], labels_list: List[List[str]],
              status_list: List[str], output_path: Path):
    """写出 CSV：每行一个样本，含 10 维 label 向量 + 病性中文名"""
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        # 表头
        label_cols = [f'{h}|{cn}' for h, cn in zip(LABEL_HEADERS, LABEL_HEADERS_CN)]
        writer.writerow(
            ['user_id', 'source', 'syndrome_original', 'status',
             'chief_complaint', 'description', 'detection'] +
            label_cols +
            ['n_labels', 'labels_cn']
        )

        for sample, labels, status in zip(samples, labels_list, status_list):
            vec = labels_to_vector(labels)
            labels_cn = '|'.join([SYNDROME_CN.get(l, l) for l in labels])
            writer.writerow([
                sample.get('user_id', ''),
                sample.get('source', ''),
                sample.get('syndrome', ''),
                status,
                sample.get('chief_complaint', ''),
                sample.get('description', ''),
                sample.get('detection', ''),
            ] + vec + [len(labels), labels_cn])


def process_split(data: List[dict], source: str) -> Tuple[List[dict], List[List[str]], List[str]]:
    """处理一个 split 的所有样本"""
    samples_kept = []
    labels_list = []
    status_list = []

    skip_status = {'external_excluded', 'unmapped', 'manual_no_evidence'}

    for sample in data:
        labels, status = get_labels(sample)
        if status in skip_status:
            continue
        if not labels:
            continue

        sample_with_source = dict(sample, source=source)
        samples_kept.append(sample_with_source)
        labels_list.append(labels)
        status_list.append(status)

    return samples_kept, labels_list, status_list


# === 5. 主流程 ===

def main():
    # 数据路径基于脚本所在目录
    data_dir = Path(__file__).resolve().parent  # data
    tcm_sd_dir = data_dir / 'tcm-sd' / 'TCM_SD_train_dev'

    print('='*70)
    print('  TCM-SD 数据加载器')
    print('='*70)
    print(f'读取: {tcm_sd_dir}')

    train_data = read_jsonl(tcm_sd_dir / 'train.json')
    dev_data = read_jsonl(tcm_sd_dir / 'dev.json')
    print(f'原 train: {len(train_data):,} 条')
    print(f'原 dev  : {len(dev_data):,} 条')
    print()

    # 数据切分
    new_train, new_val = split_train_val(train_data, val_ratio=0.05, seed=42)
    print('--- 切分方案 ---')
    print(f'  新 train (95% 原 train): {len(new_train):,}')
    print(f'  新 val   (5% 原 train):  {len(new_val):,}')
    print(f'  新 test  (原 dev 整集):  {len(dev_data):,}')
    print()

    # 处理每个 split
    train_samples, train_labels, train_status = process_split(new_train, 'train')
    val_samples, val_labels, val_status = process_split(new_val, 'val')
    test_samples, test_labels, test_status = process_split(dev_data, 'test')

    print('--- 处理后保留样本 ---')
    print(f'  train: {len(train_samples):,}')
    print(f'  val  : {len(val_samples):,}')
    print(f'  test : {len(test_samples):,}')

    # 统计 status 分布
    status_counter = Counter()
    for s in train_status + val_status + test_status:
        status_counter[s] += 1
    print()
    print('--- 标签来源分布 ---')
    for status, n in sorted(status_counter.items(), key=lambda x: -x[1]):
        print(f'  {status:25s}: {status_counter[status]:,}')

    # 输出
    output_dir = data_dir / 'splits'
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(train_samples, train_labels, train_status, output_dir / 'train.csv')
    write_csv(val_samples, val_labels, val_status, output_dir / 'val.csv')
    write_csv(test_samples, test_labels, test_status, output_dir / 'test.csv')

    print()
    print('--- 输出文件 ---')
    for name in ['train.csv', 'val.csv', 'test.csv']:
        path = output_dir / name
        size_mb = path.stat().st_size / 1024 / 1024
        print(f'  {path} ({size_mb:.1f} MB)')

    # 10 病性分布
    print()
    print('='*70)
    print('  10 病性样本分布（train+val+test，多标签各计一次）')
    print('='*70)
    label_counter = Counter()
    for labels in train_labels + val_labels + test_labels:
        for l in labels:
            label_counter[l] += 1

    total = sum(label_counter.values())
    for label in SYNDROMES_10:
        cn = SYNDROME_CN[label]
        n = label_counter[label]
        print(f'  {cn:6s} ({label:20s}): {n:7,} ({n/total*100:5.1f}%)')

    # 多标签样本占比
    print()
    n_total = len(train_labels) + len(val_labels) + len(test_labels)
    n_multi = sum(1 for labels in train_labels + val_labels + test_labels if len(labels) >= 2)
    print(f'  多标签样本（≥2 个病性）: {n_multi:,} ({n_multi/n_total*100:.1f}%)')


if __name__ == '__main__':
    main()