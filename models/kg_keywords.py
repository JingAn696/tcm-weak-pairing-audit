"""
KG 关键词匹配模块（torch-free，本机可独立测试）
==============================================

功能：把样本文本（chief_complaint + description + detection）映射到 KG 的
34 个"可匹配"节点，生成每样本的 multi-hot 特征，供完整模型 v2 的 KG 分支使用。

节点构成（50 个，唯一权威定义在本模块，kg_embedding.py 从这里 import）：
    14 舌象观察 + 10 病性要素 + 6 脏腑 + 20 主诉

匹配范围限制：
    - 只匹配「20 主诉 + 14 舌象观察」共 34 个节点
    - **不匹配 10 个病性节点**（如"气虚"字样直接标病性 = 标签泄漏）
    - **不匹配 6 个脏腑节点**（单字"心/肝/脾"误报率太高）

⚠️ 关键词表 2026-09-14 已由领域专家逐节点校准定稿（splits_v2 实测命中率驱动）。
   匹配逻辑是保守的字符串包含：宁可漏报（模型还有文本分支兜底），
   不可误报（噪声直接污染 KG 特征）。

舌象观察的关键词主要会命中 detection 字段（TCM-SD 的检测描述），
这正是设计意图：文本里的舌象证据走 KG 通道显式提取，
Stage B 接入真实舌象图片后走同一条 KG 通道替换。
"""

from typing import Dict, List


# ---------- 节点定义（唯一权威来源，与领域专家校准的 KG 一致） ----------
# 2026-09-09 定稿：10 病性（8 经典 + 内风 + 实热）

NODES_OBSERVATION = [
    # 9 维舌象观察要素
    "tongue_color", "tongue_shape", "tongue_moisture", "teeth_marks",
    "coating_color", "coating_thickness", "coating_moisture", "coating_texture",
    "tongue_posture",
    # 4 区域（舌尖/舌中/舌根/舌侧）
    "region_tip", "region_center", "region_root", "region_side",
    # 舌下络脉
    "sublingual_vein",
]

NODES_SYNDROME = [
    "qi_deficiency", "blood_deficiency", "yin_deficiency", "yang_deficiency",
    "qi_stagnation", "blood_stasis", "phlegm_dampness", "damp_heat",
    "internal_wind", "excess_heat",
]

NODES_ORGAN = [
    "spleen", "lung", "heart", "liver", "kidney", "stomach",
]

NODES_CHIEF_COMPLAINT = [
    "fatigue", "palpitation", "insomnia", "dizziness",
    "sweating", "chest_pain", "abdominal_pain", "diarrhea",
    "constipation", "poor_appetite", "cold_intolerance", "heat_intolerance",
    "headache", "back_pain", "limb_numbness", "menstrual_disorder",
    "cough", "phlegm", "edema", "bitter_mouth",
]

ALL_NODES = NODES_OBSERVATION + NODES_SYNDROME + NODES_ORGAN + NODES_CHIEF_COMPLAINT

# KG 节点 → 索引
NODE_TO_INDEX = {n: i for i, n in enumerate(ALL_NODES)}


# ---------- 关键词表（2026-09-14 领域专家校准定稿） ----------
# 主诉节点关键词（症状描述常见说法）
CHIEF_COMPLAINT_KEYWORDS: Dict[str, List[str]] = {
    "fatigue":          ["乏力", "疲倦", "疲乏", "神疲", "倦怠", "懒言", "体倦"],
    "palpitation":      ["心悸", "心慌", "怔忡"],
    "insomnia":         ["失眠", "不寐", "入睡困难", "易醒", "多梦", "眠浅", "不得眠"],
    "dizziness":        ["头晕", "眩晕", "头眩", "晕眩"],
    "sweating":         ["自汗", "盗汗", "出汗", "汗多", "大汗",
                         "黄汗", "但头汗出", "汗出"],
    "chest_pain":       ["胸闷", "胸痛",
                         "胸中闷", "胸痞", "胸满", "胸中满", "胸中满闷"],
    # 注：原"胁痛/胁胀/胸胁"删除——前者属胁症属肝胆范畴，后者纯部位词无症
    "abdominal_pain":   ["腹痛", "腹胀",
                         # 加词（领域专家校准）
                         "胃脘痛", "胃脘胀满", "胃脘痞闷", "脘痞",
                         "脘腹胀满", "脘腹痞塞", "脘腹痞闷", "脘腹不适",
                         "心下痞", "心下痞满", "心下痞硬",
                         "心下急", "心下支结",
                         "嘈杂", "胃脘嘈杂",
                         "胃脘冷痛", "胃脘灼痛", "胃脘隐痛",
                         "胃脘刺痛", "胃脘坠胀",
                         "腹满", "少腹冷痛", "小腹胀痛",
                         "绕脐痛", "肠鸣"],
    "diarrhea":         ["腹泻", "便溏", "泄泻", "大便稀",
                         "下利", "大便稀溏", "水样便", "水泻",
                         "久泻", "下利清谷"],
    "constipation":     ["便秘", "便干", "大便干", "大便秘",
                         "大便干结", "大便难", "大便燥结"],
    "poor_appetite":    ["纳差", "食欲不振", "不思饮食", "纳呆", "食少"],
    "cold_intolerance": ["畏寒", "怕冷", "肢冷", "四肢不温",
                         "畏寒", "畏冷", "形寒", "手足不温", "厥冷"],
    # 注：原"恶寒"删除——外感表证特有词，非阳虚畏寒
    "heat_intolerance": ["烦热", "潮热",
                         "壮热", "五心烦热", "身热", "烘热"],
    # 注：原"发热/怕热"删除——前者多为外感，后者术语不严谨
    "headache":         ["头痛", "偏头痛", "头胀",
                         "头疼", "巅顶痛", "后头痛", "前额痛",
                         "全头痛", "头项强痛"],
    "back_pain":        ["腰痛", "腰酸", "腰膝酸软",
                         "腰疼", "腰酸痛", "腰胀痛", "腰刺痛",
                         "腰隐痛", "腰冷痛", "腰重痛", "腰脊痛",
                         "腰背痛", "腰腿痛", "腰骶痛"],
    # 注：原单字"腰背"删除（无具体痛症）
    "limb_numbness":    ["麻木", "肢麻"],
    "menstrual_disorder": ["月经", "经期", "经量", "痛经", "闭经",
                           "崩漏"],
    "cough":            ["咳嗽", "干咳", "咳喘",
                         "咳痰", "咯痰", "久咳", "暴咳", "阵咳"],
    "phlegm":           ["痰",
                         "痰白", "痰黄", "痰稀", "痰稠", "痰黏", "痰中带血"],
    "edema":            ["水肿", "浮肿"],
    "bitter_mouth":     ["口苦"],
}

# 舌象观察节点关键词（主要命中 detection 字段）
OBSERVATION_KEYWORDS: Dict[str, List[str]] = {
    "tongue_color":     ["舌红", "舌绛", "舌紫", "舌青",
                         "舌质淡", "舌质红", "舌质暗",
                         "淡紫", "舌暗",
                         # 加词（领域专家校准，删"舌淡、淡白"）
                         "舌淡红", "淡红舌", "淡白舌",
                         "舌淡紫", "舌紫暗", "舌青紫",
                         "舌暗红", "舌尖红", "舌边红", "舌淡白"],
    # 注：原"舌淡/淡白"删除——单字歧义大
    "tongue_shape":     ["胖大", "舌瘦", "裂纹", "舌胖",
                         "芒刺舌", "舌生芒刺", "草莓舌"],
    "tongue_moisture":  ["少津",
                         # 加词（领域专家校准，删"干燥"——多为大便/皮肤干）
                         "舌润", "舌面润", "舌滑",
                         "舌燥", "舌干", "舌少津", "舌枯"],
    "teeth_marks":      ["齿痕"],
    "coating_color":    ["苔白", "白苔", "苔黄", "黄苔",
                         "薄白", "薄黄", "黄腻", "白腻",
                         "白滑", "黄燥"],
    "coating_thickness": ["苔薄", "苔厚", "少苔", "无苔", "剥苔"],
    "coating_moisture": ["苔润", "苔燥", "水滑", "苔滑",
                         "苔湿润", "苔干", "舌干"],
    "coating_texture":  ["苔腻", "腻苔", "腐苔", "剥落",
                         "花剥苔", "地图舌",
                         "光剥苔", "镜面舌", "积粉苔"],
    "tongue_posture":   ["舌歪",
                         "吐舌", "伸舌", "缩舌",
                         "舌纵", "舌硬", "舌颤", "短缩"],
    # 注：原"颤动"删除——会误命中"心房颤动"
    "region_tip":       ["舌尖"],
    "region_center":    ["舌中"],
    "region_root":      ["舌根"],
    "region_side":      ["舌边", "舌侧", "舌两侧"],
    "sublingual_vein":  ["舌下络脉", "络脉迂曲", "瘀点", "瘀斑"],
}

# 合并可匹配关键词表（34 个节点）
MATCHABLE_KEYWORDS: Dict[str, List[str]] = {**CHIEF_COMPLAINT_KEYWORDS, **OBSERVATION_KEYWORDS}
MATCHABLE_NODES = list(MATCHABLE_KEYWORDS.keys())


# ---------- 否定检测（2026-09-11 实测数据后加的） ----------
# TCM-SD 的 detection 字段是标准化病历模板，充满阴性描述：
# "双下肢无浮肿"（94% 的水肿命中其实是阴性！）、"无斑疹"、"皮肤正常"等。
# 不做否定检测，KG 分支一半是噪声。
#
# 规则：从关键词起点向前扫描（最多 5 字），遇到标点停止（"无汗，咳嗽"
# 中咳嗽不被误杀），遇到否定词（无/未/不）则该次命中作废。

_NEGATION_MARKERS = ("无", "未", "不")
_PUNCT = "，。；、！？,.;:：（）()\"'“”‘’ \t"
_NEG_SCAN_BACK = 5


def _is_negated(text: str, start: int) -> bool:
    """判断 text 中位置 start 处的关键词是否被否定词修饰。"""
    lo = max(0, start - _NEG_SCAN_BACK)
    for i in range(start - 1, lo - 1, -1):
        ch = text[i]
        if ch in _PUNCT:
            return False  # 标点隔断 → 否定词属于前一个短语
        if ch in _NEGATION_MARKERS:
            return True
    return False


def match_kg_nodes(text: str) -> List[str]:
    """
    返回文本命中的 KG 节点名列表（去重、按定义顺序）。

    命中规则：节点任一关键词存在**至少一次非否定**出现即命中。
    （"无浮肿"不算，"双下肢浮肿"才算）

    已知局限（待领域专家校准时评估）：
    - 不处理"正常/缓解/消失"等其他否定表述
    - "不"作否定词时"不寐"类关键词本身含"不"不受影响（只看关键词之前）
    """
    if not text:
        return []
    hits = []
    for node in MATCHABLE_NODES:
        matched = False
        for kw in MATCHABLE_KEYWORDS[node]:
            start = 0
            while True:
                i = text.find(kw, start)
                if i < 0:
                    break
                if not _is_negated(text, i):
                    matched = True
                    break
                start = i + 1
            if matched:
                break
        if matched:
            hits.append(node)
    return hits


def build_multi_hot(node_names: List[str], num_entities: int = None) -> List[float]:
    """
    节点名列表 → 长度 50 的 multi-hot 向量（顺序与 ALL_NODES 一致）。
    """
    if num_entities is None:
        num_entities = len(ALL_NODES)
    vec = [0.0] * num_entities
    for n in node_names:
        idx = NODE_TO_INDEX.get(n)
        if idx is not None:
            vec[idx] = 1.0
    return vec


# ---------- 单元测试 ----------
if __name__ == "__main__":
    print("Testing kg_keywords...")

    # 1. 节点总数
    assert len(ALL_NODES) == 50, f"节点数应为 50，实际 {len(ALL_NODES)}"
    assert len(NODES_OBSERVATION) == 14
    assert len(NODES_SYNDROME) == 10
    assert len(NODES_ORGAN) == 6
    assert len(NODES_CHIEF_COMPLAINT) == 20
    print(f"OK 节点总数 = {len(ALL_NODES)}（14 观察 + 10 病性 + 6 脏腑 + 20 主诉）")

    # 2. 可匹配节点数
    assert len(MATCHABLE_NODES) == 34
    assert not any(n in NODES_SYNDROME for n in MATCHABLE_NODES), "病性节点不可文本匹配"
    assert not any(n in NODES_ORGAN for n in MATCHABLE_NODES), "脏腑节点不可文本匹配"
    print(f"OK 可匹配节点 = {len(MATCHABLE_NODES)}（20 主诉 + 14 观察）")

    # 3. 匹配测试
    t1 = "患者神疲乏力半年，心悸失眠，舌淡红，苔白，边有齿痕。"
    h1 = match_kg_nodes(t1)
    print(f"   文本: {t1}")
    print(f"   命中: {h1}")
    assert "fatigue" in h1 and "palpitation" in h1 and "insomnia" in h1
    assert "tongue_color" in h1 and "teeth_marks" in h1 and "coating_color" in h1

    t2 = "头晕耳鸣，腰膝酸软，五心烦热，舌红少苔。"
    h2 = match_kg_nodes(t2)
    print(f"   文本: {t2}")
    print(f"   命中: {h2}")
    assert "dizziness" in h2 and "back_pain" in h2
    assert "coating_thickness" in h2  # 少苔

    # 5. 否定检测（真实病历模板的阴性描述）
    t3 = "神志清晰，双下肢无浮肿，舌质淡紫，苔薄白，脉虚细涩。"
    h3 = match_kg_nodes(t3)
    print(f"   文本: {t3}")
    print(f"   命中: {h3}")
    assert "edema" not in h3, "『无浮肿』不应命中 edema"
    assert "tongue_color" in h3 and "coating_thickness" in h3

    t4 = "咳嗽咳痰，无汗，双下肢浮肿，舌淡。"
    h4 = match_kg_nodes(t4)
    print(f"   文本: {t4}")
    print(f"   命中: {h4}")
    assert "cough" in h4, "『无汗，咳嗽』中咳嗽不应被否定误杀"
    assert "sweating" not in h4, "『无汗』不应命中 sweating"
    assert "edema" in h4, "『双下肢浮肿』应命中 edema"

    # 6. multi-hot
    mh = build_multi_hot(h1)
    assert len(mh) == 50 and sum(mh) == len(h1)
    print(f"OK multi-hot 长度 50，命中 {int(sum(mh))} 个节点")

    # 7. 领域专家校准后新增回归测试（验证删词/加词生效）
    # "心房颤动" 不应再误中 tongue_posture 的"颤动"（已删）
    t5 = "心电图示心房颤动，舌淡红。"
    h5 = match_kg_nodes(t5)
    assert "tongue_posture" not in h5, "『心房颤动』不应再误中 tongue_posture"
    print(f"OK 领域专家校准回归：『心房颤动』不再误命中 tongue_posture")

    # "黄腻" 应命中 coating_color(黄腻)；"腻苔" 应命中 coating_texture(腻苔)
    t6a = "舌红，苔黄腻。"
    h6a = match_kg_nodes(t6a)
    assert "coating_color" in h6a, f"『苔黄腻』应命中 coating_color，实际 {h6a}"
    print(f"OK 领域专家校准回归：苔黄腻 → {h6a}")

    t6b = "舌红，苔腻厚。"
    h6b = match_kg_nodes(t6b)
    assert "coating_texture" in h6b, f"『苔腻厚』应命中 coating_texture，实际 {h6b}"
    print(f"OK 领域专家校准回归：苔腻厚 → {h6b}")

    # "腐苔/花剥苔/地图舌/光剥苔/镜面舌" 应全部命中 coating_texture
    t6c = "舌红光剥，花剥苔如地图，镜面舌无苔。"
    h6c = match_kg_nodes(t6c)
    for w in ["光剥苔", "花剥苔", "地图舌", "镜面舌"]:
        assert "coating_texture" in h6c, f"『{w}』应命中 coating_texture"
    print(f"OK 领域专家校准回归：剥苔类关键词 → {h6c}")

    # "心下痞" / "脘腹胀满" 等腹部词都应命中 abdominal_pain
    t7 = "胃脘胀满，心下痞满，绕脐痛，肠鸣漉漉。"
    h7 = match_kg_nodes(t7)
    assert "abdominal_pain" in h7, f"腹部词应命中 abdominal_pain，实际 {h7}"
    print(f"OK 领域专家校准回归：腹部加词 → {h7}")

    print("\n=== kg_keywords 全部自测通过 ===")
