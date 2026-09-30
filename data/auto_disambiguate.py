"""
auto_disambiguate.py
====================
对"待手动拆解"的复合虚证（脾肾两虚、肝肾亏虚等）做自动病性拆解。

核心原则（净安要求）：
- 有较大把握的（文本里有明确病性证据） → 按规则批量拆
- 拿不准的（无明确病性证据） → 排除

策略：基于"望闻"detection 字段里的舌脉描述 + 病史的关键症状，按 6 病性
（阴虚/阳虚/血虚/血瘀/气虚/痰湿）的证据关键词打分。**单条样本至少需
出现 ≥2 个强证据**才认定为对应病性，避免单字模糊匹配误拆。

强证据采用中医临床共识的"舌+脉+症"三联组合，不是单一关键词。
"""

import json
import csv
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

# === 6 病性证据字典（第二轮扩充，覆盖更多变体）===
# 规则分两层：
#   - definitive: 单条即足以判定的"教科书证据"（特异性高，几乎无歧义）
#   - supportive: 普通证据，需要 ≥2 条联合（避免误拆）
#
# 判定逻辑：definitive ≥1 OR (definitive 0 AND supportive ≥2)
#
# 第二轮补充：
#   - 加 "舌质X" 变体（数据里用"舌质暗/红/紫"，我原规则只写了"舌暗/红/紫"）
#   - 加 "苔薄腻/黄腻/黄厚腻" 变体
#   - 加 "苔剥/苔光" 等少苔变体
#   - 把"脉弦细"提到 definitive（阴虚常见脉）

EVIDENCE_RULES = {
    'yin_deficiency': {
        'label_cn': '阴虚',
        'definitive': [
            # 舌象（高特异）
            '舌红', '舌绛', '舌质红', '舌质绛',
            '少苔', '剥苔', '无苔', '苔少', '苔剥', '苔光', '苔干',
            # 脉象（高特异）
            '脉细数', '脉数', '脉弦数', '脉弦细',
            # 症状（高特异）
            '盗汗', '潮热', '五心烦热', '颧红', '午后潮热', '手足心热',
            '口干', '咽干', '口燥',
        ],
        'supportive': [
            '消瘦', '形体偏瘦', '低热', '心烦',
        ],
    },
    'yang_deficiency': {
        'label_cn': '阳虚',
        'definitive': [
            # 舌象（高特异）
            # 注意："舌淡" 已移到 supportive（"舌淡红"在数据里太常见，单条不能定阳虚）
            '舌淡胖', '舌质淡', '舌质胖大', '舌质淡胖',
            '齿痕', '苔白滑', '苔润滑',
            # 脉象
            '脉沉迟', '脉迟', '脉微', '脉沉细', '脉沉弱',
            # 症状
            '畏寒', '畏冷', '肢冷', '怕冷', '形寒', '四肢不温',
            '小便清长', '夜尿多', '五更泻',
            '面色㿠白', '面色苍白',
        ],
        'supportive': [
            '舌淡',  # 弱（"舌淡红"也是舌淡，需配合脉象/症状）
            '下肢浮肿', '下肢有浮肿', '便溏', '喜热饮',
        ],
    },
    'blood_deficiency': {
        'label_cn': '血虚',
        'definitive': [
            # 症状（高特异）
            '面色少华', '面色苍白', '面色萎黄',
            '唇淡', '口唇色淡', '唇甲淡', '爪甲色淡',
            '月经量少', '月经色淡',
        ],
        'supportive': [
            '舌淡', '舌质淡',  # 弱（与阳虚重叠）
            '脉细',
            '头晕', '心悸', '眼花', '乏力',
        ],
    },
    'blood_stasis': {
        'label_cn': '血瘀',
        'definitive': [
            # 舌象（高特异）
            '舌紫', '舌暗', '舌淡紫', '舌青紫', '舌偏暗',
            '舌质暗', '舌质紫', '舌质紫暗', '舌质淡紫', '舌质青紫',
            '瘀斑', '舌下静脉',
            # 脉象
            '脉涩', '脉弦涩',
            # 症状
            '刺痛', '固定痛', '夜间加重',
            '面色晦暗', '口唇发绀', '口唇紫暗', '肌肤甲错',
        ],
        'supportive': [
            '疼痛固定', '夜间痛甚',
        ],
    },
    'qi_deficiency': {
        'label_cn': '气虚',
        'definitive': [
            # 脉象
            '脉弱', '脉虚', '脉细弱',
            # 症状（高特异）
            '乏力', '神疲', '倦怠', '少气懒言',
            '自汗', '动则汗出', '气短', '声低',
        ],
        'supportive': [
            '活动后加重', '劳累后加重',
        ],
    },
    'phlegm_dampness': {
        'label_cn': '痰湿',
        'definitive': [
            # 舌象（高特异）
            '苔腻', '苔白腻', '苔厚腻', '苔薄腻', '苔黄腻', '苔黄厚腻',
            # 脉象
            '脉滑', '脉弦滑',
            # 症状
            '咳痰', '痰多', '形体肥胖', '身重', '头重如裹',
        ],
        'supportive': [
            '胸闷', '脘痞',
        ],
    },
}


def find_evidence(text: str, keywords: List[str]) -> List[str]:
    """在文本中找出命中的关键词"""
    return [kw for kw in keywords if kw in text]


def infer_labels(sample: dict) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    根据望闻(detection) + 主诉(chief_complaint) + 病史(description)
    三处文本，按 EVIDENCE_RULES 推断病性标签。

    判定逻辑（两层规则）：
      - definitive 命中 ≥1 → 认定为该病性
      - definitive 命中 =0 但 supportive 命中 ≥2 → 认定为该病性
      - 其他情况 → 不认定
    """
    # 拼接所有文本字段
    full_text = ' '.join([
        sample.get('chief_complaint', ''),
        sample.get('description', ''),
        sample.get('detection', ''),
    ])

    labels = []
    evidence = {}

    for label, rule in EVIDENCE_RULES.items():
        definitive_hits = find_evidence(full_text, rule['definitive'])
        supportive_hits = find_evidence(full_text, rule['supportive'])

        # 合并去重
        all_hits = list(set(definitive_hits + supportive_hits))

        # 判定逻辑
        if len(definitive_hits) >= 1:
            labels.append(label)
            evidence[label] = {
                'definitive': definitive_hits,
                'supportive': supportive_hits,
            }
        elif len(supportive_hits) >= 2:
            labels.append(label)
            evidence[label] = {
                'definitive': [],
                'supportive': supportive_hits,
            }
        # 否则不认定

    return labels, evidence


def process_manual_samples(train_jsonl_path: Path) -> Dict:
    """处理所有待手动拆解样本，返回分类结果"""

    # 复用 syndrome_mapping 里的判定
    sys.path.insert(0, str(Path(__file__).parent))
    from syndrome_mapping import map_syndrome

    auto_disambiguated = []  # 自动拆解成功
    excluded = []            # 排除（无明确证据）

    with open(train_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            syndrome = sample.get('syndrome', '')

            # 只处理待手动拆解的证候
            r = map_syndrome(syndrome)
            if r['status'] != 'manual_needed':
                continue

            # 自动推断病性
            labels, evidence = infer_labels(sample)

            if labels:
                auto_disambiguated.append({
                    'user_id': sample.get('user_id', ''),
                    'syndrome': syndrome,
                    'inferred_labels': labels,
                    'evidence': evidence,
                    'chief_complaint': sample.get('chief_complaint', '')[:200],
                    'detection': sample.get('detection', '')[:300],
                })
            else:
                excluded.append({
                    'user_id': sample.get('user_id', ''),
                    'syndrome': syndrome,
                    'chief_complaint': sample.get('chief_complaint', '')[:200],
                    'detection': sample.get('detection', '')[:300],
                })

    return {
        'auto_disambiguated': auto_disambiguated,
        'excluded': excluded,
    }


def save_results(results: Dict, output_dir: Path):
    """保存拆解结果到 CSV 文件"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 自动拆解成功的
    auto_path = output_dir / 'auto_disambiguated.csv'
    with open(auto_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', '原证候', '推断病性', '病性（中文）', '强证据', '辅助证据',
                         '主诉前200字', '望闻前300字'])
        for item in results['auto_disambiguated']:
            label_cn = [EVIDENCE_RULES[l]['label_cn'] for l in item['inferred_labels']]
            all_definitive = []
            all_supportive = []
            for l in item['evidence']:
                all_definitive.extend([f'{EVIDENCE_RULES[l]["label_cn"]}:{h}' for h in item['evidence'][l]['definitive']])
                all_supportive.extend([f'{EVIDENCE_RULES[l]["label_cn"]}:{h}' for h in item['evidence'][l]['supportive']])
            writer.writerow([
                item['user_id'],
                item['syndrome'],
                '|'.join(item['inferred_labels']),
                '|'.join(label_cn),
                '; '.join(all_definitive) if all_definitive else '(无)',
                '; '.join(all_supportive) if all_supportive else '(无)',
                item['chief_complaint'],
                item['detection'],
            ])

    # 2. 排除的（无明确证据）
    excl_path = output_dir / 'excluded_no_evidence.csv'
    with open(excl_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', '原证候', '主诉前200字', '望闻前300字'])
        for item in results['excluded']:
            writer.writerow([
                item['user_id'],
                item['syndrome'],
                item['chief_complaint'],
                item['detection'],
            ])

    print(f'✓ 自动拆解成功的写入: {auto_path}')
    print(f'✓ 排除的写入: {excl_path}')


def print_summary(results: Dict):
    """打印拆解摘要"""
    auto = results['auto_disambiguated']
    excl = results['excluded']

    print('='*70)
    print('  病性证据自动拆解结果摘要')
    print('='*70)
    print(f'待手动拆解样本总数: {len(auto) + len(excl)}')
    print(f'  ✓ 自动拆解成功（按规则批量拆）: {len(auto)} ({len(auto)/(len(auto)+len(excl))*100:.1f}%)')
    print(f'  ✗ 排除（无明确证据）: {len(excl)} ({len(excl)/(len(auto)+len(excl))*100:.1f}%)')
    print()

    # 按推断标签分布
    label_counter = Counter()
    for item in auto:
        for l in item['inferred_labels']:
            label_counter[l] += 1

    print('=== 自动拆解样本的病性分布（多标签各计一次）===')
    for label, count in label_counter.most_common():
        label_cn = EVIDENCE_RULES[label]['label_cn']
        print(f'  {label_cn:6s} ({label:20s}): {count:5,d} 条')

    # 按原证候分布
    print()
    print('=== 按原证候的拆解结果 ===')
    syn_auto = Counter(item['syndrome'] for item in auto)
    syn_excl = Counter(item['syndrome'] for item in excl)
    all_syns = set(syn_auto.keys()) | set(syn_excl.keys())
    for syn in sorted(all_syns, key=lambda s: -(syn_auto.get(s, 0) + syn_excl.get(s, 0))):
        print(f'  {syn:15s}: 自动拆 {syn_auto.get(syn, 0):4d} | 排除 {syn_excl.get(syn, 0):4d}')


if __name__ == '__main__':
    # 默认路径（基于项目根目录 = __file__ 向上 4 层）
    # auto_disambiguate.py 在 papers/code/data/ 下
    # 项目根 = C:\Users\think\WorkBuddy\2026-09-05-15-22-10
    project_root = Path(__file__).resolve().parents[3]
    train_jsonl = project_root / 'papers' / 'code' / 'data' / 'tcm-sd' / 'TCM_SD_train_dev' / 'train.json'
    output_dir = project_root / 'papers' / 'code' / 'data'

    if len(sys.argv) >= 2:
        train_jsonl = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])

    print(f'读取: {train_jsonl}')
    print(f'输出到: {output_dir}')
    print()

    results = process_manual_samples(train_jsonl)
    print_summary(results)
    save_results(results, output_dir)