# syndrome_mapping.py
"""
10 病性要素映射核心 - TCM-SD 148 证候 → 10 病性多标签

8+2 框架（领域专家 2026-09-09 拍板）：
  1. 气虚     qi_deficiency
  2. 血虚     blood_deficiency
  3. 阴虚     yin_deficiency
  4. 阳虚     yang_deficiency
  5. 气滞     qi_stagnation     （含"郁"并入）
  6. 血瘀     blood_stasis
  7. 痰湿     phlegm_dampness
  8. 湿热     damp_heat
  9. 内风     wind              （仅内风，外感排除，领域专家 2026-09-09 由"风"改名）
 10. 实热     excess_heat       （湿热/虚热/阴虚以外）

学术诚信声明（论文 Method 写）：
  - "气血亏虚"等病性明确的复合证，按中医辨证学常识拆解为多标签
  - 病性不明的复合证（脾肾两虚等）由中医专家（第一作者）按临床经验判定
  - TCM-SD 部分样本标注质量参差，我们做了规则化重标注，重标注规则详见附录 A
"""

from pathlib import Path
from collections import Counter
import json
import csv


# === 1. 10 病性要素定义 ===
SYNDROMES_10 = [
    'qi_deficiency', 'blood_deficiency', 'yin_deficiency', 'yang_deficiency',
    'qi_stagnation', 'blood_stasis', 'phlegm_dampness', 'damp_heat',
    'wind', 'excess_heat',
]

SYNDROME_CN = {
    'qi_deficiency': '气虚',
    'blood_deficiency': '血虚',
    'yin_deficiency': '阴虚',
    'yang_deficiency': '阳虚',
    'qi_stagnation': '气滞',
    'blood_stasis': '血瘀',
    'phlegm_dampness': '痰湿',
    'damp_heat': '湿热',
    'wind': '内风',
    'excess_heat': '实热',
}

# 颜色用于论文图（用项目蓝白主题）
SYNDROME_COLOR = {
    'qi_deficiency': '#0E3F8C',
    'blood_deficiency': '#C84B4B',
    'yin_deficiency': '#3B7DD8',
    'yang_deficiency': '#FF8C42',
    'qi_stagnation': '#7A5CC4',
    'blood_stasis': '#8B2C2C',
    'phlegm_dampness': '#5B8C5A',
    'damp_heat': '#D4A017',
    'wind': '#4A90A4',
    'excess_heat': '#D64545',
}


# === 2. 关键词自动匹配（基础层）===
# 格式：(病性标签, 关键词)
# 注意：
#   - "痰" 单独出现也会匹配（痰瘀互结=痰+血瘀）
#   - "风" 在本层不匹配，留给复合证精确拆解（区分内风/外风）
KEYWORD_RULES = [
    ('qi_deficiency', '气虚'),
    ('blood_deficiency', '血虚'),
    ('yin_deficiency', '阴虚'),
    ('yang_deficiency', '阳虚'),
    ('qi_stagnation', '气滞'),
    ('blood_stasis', '血瘀'),
    ('blood_stasis', '瘀血'),
    ('phlegm_dampness', '痰湿'),
    ('phlegm_dampness', '痰浊'),
    ('phlegm_dampness', '痰'),       # 弱匹配：痰瘀互结/痰湿互结
    ('damp_heat', '湿热'),
    # 实热关键词
    ('excess_heat', '郁热'),
    ('excess_heat', '热毒'),
]


# === 3. 复合证精确拆解（病性明确型）===
# 优先级高于关键词匹配
COMPOUND_RULES = {
    # 气血相关 → 都拆 [气虚, 血虚]（领域专家 2026-09-09 拍板：方案 1 全自动）
    '气血两虚证': ['qi_deficiency', 'blood_deficiency'],
    '气血不足证': ['qi_deficiency', 'blood_deficiency'],
    '气血亏虚证': ['qi_deficiency', 'blood_deficiency'],
    '气血失调证': ['qi_deficiency', 'blood_deficiency'],
    '气血两虚': ['qi_deficiency', 'blood_deficiency'],
    '气血不足': ['qi_deficiency', 'blood_deficiency'],
    '气血瘀滞证': ['qi_deficiency', 'blood_stasis'],
    '气血瘀滞': ['qi_deficiency', 'blood_stasis'],

    # 气阴两虚
    '气阴两虚证': ['qi_deficiency', 'yin_deficiency'],
    '气阴亏虚证': ['qi_deficiency', 'yin_deficiency'],

    # 阴阳两虚
    '阴阳两虚证': ['yin_deficiency', 'yang_deficiency'],
    '阴阳两虚': ['yin_deficiency', 'yang_deficiency'],

    # 阳虚类（病性明确）
    '脾肾阳虚证': ['yang_deficiency'],
    '肾阳虚证': ['yang_deficiency'],
    '阳气虚证': ['yang_deficiency'],

    # 阴虚类
    '肝肾阴虚证': ['yin_deficiency'],
    '肾阴虚证': ['yin_deficiency'],

    # 脏腑气虚
    '肺肾气虚证': ['qi_deficiency', 'yang_deficiency'],
    '脾肾气虚证': ['qi_deficiency', 'yang_deficiency'],
    '心肺气虚证': ['qi_deficiency'],
    '肾气虚证': ['qi_deficiency', 'yang_deficiency'],
    '心脾两虚证': ['qi_deficiency', 'blood_deficiency'],

    # 郁→气滞（领域专家拍板"郁"并入气滞）
    '肝气郁结证': ['qi_stagnation'],
    '肝气郁结': ['qi_stagnation'],
    '肝气郁滞证': ['qi_stagnation'],
    '肝气郁滞': ['qi_stagnation'],
    '肝郁气滞证': ['qi_stagnation'],
    '肝郁气结证': ['qi_stagnation'],
    '肝胃气滞证': ['qi_stagnation'],
    '肝胃不和证': ['qi_stagnation'],
    '肝胃郁热证': ['qi_stagnation', 'excess_heat'],
    '脾胃不和证': ['qi_stagnation'],
    '肝郁脾虚证': ['qi_stagnation', 'qi_deficiency'],
    '肝郁血瘀证': ['qi_stagnation', 'blood_stasis'],
    '气郁痰阻证': ['qi_stagnation', 'phlegm_dampness'],
    '气郁痰凝证': ['qi_stagnation', 'phlegm_dampness'],
    '痰热郁肺证': ['phlegm_dampness', 'excess_heat'],

    # 风痰类（内风 + 痰湿）
    '风痰阻络证': ['wind', 'phlegm_dampness'],
    '风痰上扰证': ['wind', 'phlegm_dampness'],
    '风痰入络证': ['wind', 'phlegm_dampness'],
    '风痰瘀阻证': ['wind', 'phlegm_dampness', 'blood_stasis'],
    '风痰瘀阻': ['wind', 'phlegm_dampness', 'blood_stasis'],

    # 痰瘀类
    '痰瘀互结证': ['phlegm_dampness', 'blood_stasis'],
    '痰瘀痹阻证': ['phlegm_dampness', 'blood_stasis'],

    # 寒/毒/虚（内伤部分）
    '脾胃虚寒证': ['yang_deficiency'],
    '肾虚寒凝证': ['yang_deficiency'],
    '肾虚督寒证': ['yang_deficiency'],
    '寒凝血瘀证': ['blood_stasis'],
    '寒湿阻络证': ['phlegm_dampness', 'yang_deficiency'],
    '寒湿痹阻证': ['phlegm_dampness', 'yang_deficiency'],
    '寒滞经络证': ['yang_deficiency'],

    # 正虚
    '正虚瘀结证': ['qi_deficiency', 'blood_stasis'],
    '正虚瘀结': ['qi_deficiency', 'blood_stasis'],
    '正虚毒瘀证': ['qi_deficiency', 'blood_stasis'],
    '正虚毒结证': ['qi_deficiency', 'blood_stasis'],
    '正虚毒恋证': ['qi_deficiency'],
    '气虚毒滞证': ['qi_deficiency'],
    '气虚不摄证': ['qi_deficiency'],

    # 血虚类（已含血虚，但有些带风/燥）
    '血虚风燥证': ['blood_deficiency', 'wind'],
    '血虚风燥': ['blood_deficiency', 'wind'],
    '血虚生风证': ['blood_deficiency', 'wind'],
    '血虚生风': ['blood_deficiency', 'wind'],
    '血热证': ['excess_heat'],
    '血热': ['excess_heat'],

    # 阳亢（本质阴虚）
    '肝阳上亢证': ['yin_deficiency'],
    '肝阳上亢': ['yin_deficiency'],
    '肝风内动证': ['wind', 'yin_deficiency'],
    '肝风内动': ['wind', 'yin_deficiency'],

    # 热毒（实热）
    '热毒蕴结证': ['excess_heat'],
    '热毒炽盛证': ['excess_heat'],
    '热毒蕴肤证': ['excess_heat'],
    '热毒壅结证': ['excess_heat'],
    '湿热蕴毒证': ['damp_heat', 'excess_heat'],
    '湿毒蕴结证': ['damp_heat', 'excess_heat'],
    '湿毒蕴肤证': ['damp_heat', 'excess_heat'],

    # 杂证（已确认的）
    '痰结毒滞证': ['phlegm_dampness', 'excess_heat'],
    '脓毒侵袭证': ['excess_heat'],
    '毒邪流窜证': ['excess_heat'],

    # === 第二轮扩充（2026-09-09）：明显可自动拆的杂证 ===
    # 心系
    '心血不足证': ['blood_deficiency'],
    '心血亏虚证': ['blood_deficiency'],
    '心气亏虚证': ['qi_deficiency'],
    '心阳不振证': ['yang_deficiency'],
    '心脉瘀阻证': ['blood_stasis'],
    '瘀阻心脉证': ['blood_stasis'],
    '瘀阻脉络证': ['blood_stasis'],
    '瘀滞胞宫证': ['blood_stasis'],
    '心虚胆怯证': ['qi_deficiency', 'qi_stagnation'],
    # 脾胃系
    '脾胃虚弱证': ['qi_deficiency'],
    '脾胃亏虚证': ['qi_deficiency'],
    '脾虚证': ['qi_deficiency'],
    '脾虚湿盛证': ['qi_deficiency', 'phlegm_dampness'],
    '脾虚湿蕴证': ['qi_deficiency', 'phlegm_dampness'],
    '脾肾亏虚证': ['qi_deficiency', 'yang_deficiency'],  # 中医常拆法
    '中气下陷证': ['qi_deficiency'],
    '中气不足证': ['qi_deficiency'],
    '中气不中证': ['qi_deficiency'],   # 错别字兼容（"中气不足"）
    '脾不统血证': ['qi_deficiency', 'blood_deficiency'],
    '饮食积滞证': ['qi_stagnation', 'phlegm_dampness'],
    '胃热滞脾证': ['excess_heat', 'qi_stagnation'],
    '脾胃积热证': ['excess_heat'],
    # 肝系
    '肝阴不足证': ['yin_deficiency'],
    '肝气犯胃证': ['qi_stagnation'],
    '肝阳上扰证': ['yin_deficiency'],
    '气机阻滞证': ['qi_stagnation'],
    '肝脾两虚证': ['qi_stagnation', 'qi_deficiency'],   # 中医常拆为"肝郁脾虚"
    '肝脾亏虚证': ['qi_stagnation', 'qi_deficiency'],
    # 肺系
    '肺卫不固证': ['qi_deficiency'],
    '肺阴亏耗证': ['yin_deficiency'],
    '燥热伤肺证': ['excess_heat'],
    # 肾系
    '肾气不足证': ['qi_deficiency', 'yang_deficiency'],
    '肾不纳气证': ['qi_deficiency', 'yang_deficiency'],
    '肾虚水泛证': ['yang_deficiency', 'phlegm_dampness'],  # 水泛=痰湿类
    '肾虚肝亢证': ['yin_deficiency', 'yang_deficiency'],
    '髓海不足证': ['yin_deficiency', 'yang_deficiency'],  # 肾精不足
    '肾虚寒凝证': ['yang_deficiency'],
    '肾虚督寒证': ['yang_deficiency'],
    # 气血津液
    '气不摄血证': ['qi_deficiency', 'blood_deficiency'],
    '津亏热结证': ['yin_deficiency', 'excess_heat'],
    '肠燥津伤证': ['yin_deficiency'],
    '热结肠燥证': ['excess_heat'],
    # 复合（病性相对明确）
    '心肾不交证': ['yin_deficiency', 'yang_deficiency'],  # 中医常拆
    '心肾不交': ['yin_deficiency', 'yang_deficiency'],
    '肾虚证': ['qi_deficiency', 'yang_deficiency'],
    '肾虚': ['qi_deficiency', 'yang_deficiency'],
    # 无"证"字兼容
    '气阴两虚': ['qi_deficiency', 'yin_deficiency'],
    '气血两虚': ['qi_deficiency', 'blood_deficiency'],
    '气血不足': ['qi_deficiency', 'blood_deficiency'],
}


# === 4. 外感关键词（命中即排除，不参与内伤映射）===
EXTERNAL_KEYWORDS = [
    '风寒', '风热', '风湿', '外感', '寒湿', '暑湿', '燥邪', '风邪',
    '风寒袭', '风寒外', '风寒湿', '风寒阻', '风寒痹', '风寒束',
    '风湿蕴', '风热犯', '风热外', '风热袭', '风热毒',  # 留着观察
]


# === 5. 待领域专家手动拆解的证候（病性不明的复合证）===
# 这些无法从证名直接判断病性，需要 11 年中医临床经验
MANUAL_DISAMBIGUATION = [
    # 病性不明型（领域专家需逐条审核）：
    '脾肾两虚证',          # 病性不明
    '肝肾两虚证',          # 病性不明
    '肝肾亏虚证',          # 病性不明
    '肝肾亏损证',          # 病性不明
    '肝肾不足证',          # 病性不明
    '外伤损络证',          # 病位+病性不明
    '寒证',                # 范围过宽，无法自动归类
    '虚证',                # 同上
]


# === 6. 主映射函数 ===
def map_syndrome(syndrome_name: str) -> dict:
    """
    148 证候 → 10 病性多标签映射

    Returns:
        {
            'labels': set of 病性英文标签,
            'status': 'auto' | 'manual_needed' | 'external_excluded' | 'unmapped',
            'reason': str,
        }
    """
    syn = syndrome_name.strip()

    # 1. 外感排除
    if any(kw in syn for kw in EXTERNAL_KEYWORDS):
        return {'labels': set(), 'status': 'external_excluded',
                'reason': f'外感证候，已排除: {syn}'}

    # 2. 复合证精确拆解（最高优先级）
    if syn in COMPOUND_RULES:
        return {'labels': set(COMPOUND_RULES[syn]), 'status': 'auto',
                'reason': f'复合证拆解: {syn} → {COMPOUND_RULES[syn]}'}

    # 3. 关键词匹配
    labels = set()
    matched_kws = []
    for label, kw in KEYWORD_RULES:
        if kw in syn:
            labels.add(label)
            matched_kws.append(f'{kw}→{label}')

    if labels:
        return {'labels': labels, 'status': 'auto',
                'reason': f'关键词匹配: {matched_kws}'}

    # 4. 待手动拆解（病性不明）
    if syn in MANUAL_DISAMBIGUATION:
        return {'labels': set(), 'status': 'manual_needed',
                'reason': f'病性不明，需领域专家手动标注: {syn}'}

    # 5. 真正无法映射
    return {'labels': set(), 'status': 'unmapped',
            'reason': f'未匹配，需检查: {syn}'}


# === 7. 数据集分析 ===
DATA_DIR = Path(__file__).parent  # data/
CODE_DIR = DATA_DIR.parent        # code/

def analyze_dataset(jsonl_path: str = None) -> dict:
    """分析整个数据集，统计 10 病性的样本分布"""
    if jsonl_path is None:
        jsonl_path = CODE_DIR / 'data' / 'tcm-sd' / 'TCM_SD_train_dev' / 'train.json'
    jsonl_path = Path(jsonl_path)

    label_counter = Counter()
    status_counter = Counter()
    syndrome_labels = {}  # syndrome_name -> set of labels
    sample_labels = []    # 每条样本的标签集
    total = 0
    multi_label = 0
    unmapped_samples = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            total += 1
            result = map_syndrome(obj['syndrome'])
            status_counter[result['status']] += 1
            syndrome_labels.setdefault(obj['syndrome'], result['labels'])
            sample_labels.append(result['labels'])
            if len(result['labels']) > 1:
                multi_label += 1
            for l in result['labels']:
                label_counter[l] += 1
            if result['status'] == 'unmapped':
                unmapped_samples.append((total, obj['syndrome']))

    return {
        'total': total,
        'multi_label_samples': multi_label,
        'label_counter': label_counter,
        'status_counter': status_counter,
        'syndrome_labels': syndrome_labels,
        'sample_labels': sample_labels,
        'unmapped_samples': unmapped_samples,
    }


# === 8. 待拆解清单生成器 ===
def generate_manual_list(jsonl_path: str = None, output_csv: str = None) -> list:
    """
    扫描数据集，找出所有"待手动拆解"的样本，
    输出每条的 user_id / syndrome / chief_complaint / 检测 摘要，
    给领域专家逐条审核。
    """
    if jsonl_path is None:
        jsonl_path = CODE_DIR / 'data' / 'tcm-sd' / 'TCM_SD_train_dev' / 'train.json'
    jsonl_path = Path(jsonl_path)
    if output_csv is None:
        output_csv = CODE_DIR / 'data' / 'manual_disambiguation_list.csv'
    output_csv = Path(output_csv)

    rows = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            result = map_syndrome(obj['syndrome'])
            if result['status'] in ('manual_needed', 'unmapped'):
                rows.append({
                    'user_id': obj.get('user_id', ''),
                    'lcd_name': obj.get('lcd_name', ''),
                    'syndrome': obj['syndrome'],
                    'status': result['status'],
                    'reason': result['reason'],
                    'chief_complaint': obj.get('chief_complaint', '')[:200],
                    'detection_snippet': obj.get('detection', '')[:300],
                })

    # 写 CSV
    if rows:
        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


# === 9. 单元测试 ===
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    print('='*70)
    print('TCM-SD 148 证候 → 10 病性 映射统计')
    print('='*70)

    stats = analyze_dataset()
    total = stats['total']

    print(f'\n总样本数: {total:,}')
    print(f'多标签样本数: {stats["multi_label_samples"]:,} ({stats["multi_label_samples"]/total*100:.1f}%)')
    print(f'\n映射状态分布:')
    for status, c in stats['status_counter'].most_common():
        print(f'  {status:25s}: {c:6,} ({c/total*100:5.1f}%)')

    print(f'\n10 病性要素样本分布（多标签各计一次）:')
    print(f'  (注：因多标签重叠，合计 > 总样本数)')
    for label, c in stats['label_counter'].most_common():
        cn = SYNDROME_CN[label]
        print(f'  {cn:6s} ({label:18s}): {c:6,} ({c/total*100:5.1f}%)')

    print(f'\n=== 待领域专家手动拆解样本清单 ===')
    manual_list = generate_manual_list()
    print(f'共 {len(manual_list)} 条，已写入 manual_disambiguation_list.csv')

    # 按证候分组统计
    syn_counter = Counter(r['syndrome'] for r in manual_list)
    print(f'\n按证候统计:')
    for syn, c in syn_counter.most_common():
        print(f'  {c:4d}  {syn}')

    if stats['unmapped_samples']:
        print(f'\n=== ⚠️ 真正无法映射的样本（需检查规则）===')
        syn_unmapped = Counter(s for _, s in stats['unmapped_samples'])
        for syn, c in syn_unmapped.most_common():
            print(f'  {c:4d}  {syn}')
