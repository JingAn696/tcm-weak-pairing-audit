"""
TCM-Tongue 21 类舌象标签 → 10 病性  映射矩阵（Stage B 视觉分支的"翻译层"）

依据：notes/tcm-tongue-label-mapping.md（领域专家 2026-09-14 定稿）
数据源：TMC-Tongue (Dryad 10.5061/dryad.1c59zw48r)，**21 类**目标检测标签，编号 0-20

⚠️ 类别数澄清（2026-09-14 实测纠错）
------------------------------------------------
原以为"20 类（论文写 21 类是笔误）"，**实测为 21 类**：
  - 三个 split 的 COCO json 均为 categories id=0..20，共 21 类
  - 21 类是 `piweitu`（脾胃区凸起），README 的类别列表漏写了它（README 才是笔误）
  - 三个 split 的 classes.txt 也存在顺序不一致：train 把 xinfeitu/xinfeiao 写反
    （test/val 为 ...piweiao|piweitu|xinfeiao|xinfeitu）
  → 因此**必须按 pinyin 名称索引，绝不能按行号/id 索引**（本模块提供 PINYIN_TO_ID）

数据可用性（实测实例数，全数据集 6719 图 / 18223 bbox）
------------------------------------------------
4 个凸起类近乎空：piweitu 0 / shenqutu 2 / xinfeitu 2 / gandantu 9
→ B 组（脏腑分区）实际只有 4 个"凹陷"类有可用信号；
  气滞的两个映射来源（gandantu 9 实例、piweitu 0 实例）全部不可用
  → **气滞在视觉侧等同于无信号**（与内风并列，均为数据集覆盖缺口）。

设计要点
--------
1. 行索引 0-20 与数据集 COCO category id 严格对齐，不要随意调整顺序。
2. 矩阵为领域专家定稿的**硬 0/1**；软权重模式作为可选增强（见 tongue_to_syndrome 的 mode 参数）。
3. 本模块不依赖 torch/numpy，纯 Python；需要张量时用 get_matrix_torch()。
4. 内风列全 0 —— 数据集无舌态标签，属覆盖缺口（非映射缺陷），论文需写成 limitation。
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# 1. 病性维度定义（与全项目保持一致：10 病性）
# ----------------------------------------------------------------------------
SYNDROMES_ZH = ["气虚", "血虚", "阴虚", "阳虚", "气滞", "血瘀", "痰湿", "湿热", "内风", "实热"]
SYNDROME_KEYS = [
    "qi_deficiency",     # 气虚
    "blood_deficiency",  # 血虚
    "yin_deficiency",    # 阴虚
    "yang_deficiency",   # 阳虚
    "qi_stagnation",     # 气滞
    "blood_stasis",      # 血瘀
    "phlegm_dampness",   # 痰湿
    "damp_heat",         # 湿热
    "internal_wind",     # 内风
    "excess_heat",       # 实热
]
NUM_SYNDROMES = len(SYNDROMES_ZH)  # 10

# ----------------------------------------------------------------------------
# 2. 21 类舌象标签（索引 = 数据集 COCO category id，0-20）
#    n_inst = 全数据集实例数（实测），供数据可用性判断参考
# ----------------------------------------------------------------------------
TONGUE_CLASSES = [
    {"id": 0,  "pinyin": "jiankangshe", "zh": "健康舌",     "en": "Healthy Tongue",           "scope": "global", "n_inst": 21},
    {"id": 1,  "pinyin": "botaishe",    "zh": "剥苔舌",     "en": "Peeling Coating",          "scope": "global", "n_inst": 581},
    {"id": 2,  "pinyin": "hongshe",     "zh": "红舌",       "en": "Red Tongue",               "scope": "global", "n_inst": 1466},
    {"id": 3,  "pinyin": "zishe",       "zh": "紫舌",       "en": "Purple Tongue",            "scope": "global", "n_inst": 231},
    {"id": 4,  "pinyin": "pangdashe",   "zh": "胖大舌",     "en": "Chubby Tongue",            "scope": "global", "n_inst": 689},
    {"id": 5,  "pinyin": "shoushe",     "zh": "瘦舌",       "en": "Thin Tongue",              "scope": "global", "n_inst": 285},
    {"id": 6,  "pinyin": "hongdianshe", "zh": "红点舌",     "en": "Red Dot Tongue",           "scope": "global", "n_inst": 3653},
    {"id": 7,  "pinyin": "liewenshe",   "zh": "裂纹舌",     "en": "Cracked Tongue",           "scope": "global", "n_inst": 1886},
    {"id": 8,  "pinyin": "chihenshe",   "zh": "齿痕舌",     "en": "Dentate Tongue",           "scope": "global", "n_inst": 1482},
    {"id": 9,  "pinyin": "baitaishe",   "zh": "白苔舌",     "en": "White Coating",            "scope": "global", "n_inst": 5040},
    {"id": 10, "pinyin": "huangtaishe", "zh": "黄苔舌",     "en": "Yellow Coating",           "scope": "global", "n_inst": 1119},
    {"id": 11, "pinyin": "heitaishe",   "zh": "黑苔舌",     "en": "Black Coating",            "scope": "global", "n_inst": 122},
    {"id": 12, "pinyin": "huataishe",   "zh": "滑苔舌",     "en": "Smooth Coating",           "scope": "global", "n_inst": 283},
    {"id": 13, "pinyin": "shenquao",    "zh": "肾区凹陷",   "en": "Renal Depression",         "scope": "local",  "n_inst": 495},
    {"id": 14, "pinyin": "shenqutu",    "zh": "肾区凸起",   "en": "Renal Protrusion",         "scope": "local",  "n_inst": 2},
    {"id": 15, "pinyin": "gandanao",    "zh": "肝胆区凹陷", "en": "Hepatobiliary Depression", "scope": "local",  "n_inst": 292},
    {"id": 16, "pinyin": "gandantu",    "zh": "肝胆区凸起", "en": "Hepatobiliary Protrusion", "scope": "local",  "n_inst": 9},
    {"id": 17, "pinyin": "piweiao",     "zh": "脾胃区凹陷", "en": "Spleen-Stomach Depression","scope": "local",  "n_inst": 186},
    {"id": 18, "pinyin": "piweitu",     "zh": "脾胃区凸起", "en": "Spleen-Stomach Protrusion","scope": "local",  "n_inst": 0},
    {"id": 19, "pinyin": "xinfeiao",    "zh": "心肺区凹陷", "en": "Heart-Lung Depression",    "scope": "local",  "n_inst": 372},
    {"id": 20, "pinyin": "xinfeitu",    "zh": "心肺区凸起", "en": "Heart-Lung Protrusion",    "scope": "local",  "n_inst": 2},
]
NUM_TONGUE_CLASSES = len(TONGUE_CLASSES)  # 21

# 便捷索引
TONGUE_ZH_TO_ID = {c["zh"]: c["id"] for c in TONGUE_CLASSES}
TONGUE_PINYIN_TO_ID = {c["pinyin"]: c["id"] for c in TONGUE_CLASSES}
TONGUE_ID_TO_ZH = {c["id"]: c["zh"] for c in TONGUE_CLASSES}

# 数据可用性阈值：实例数少于此值的类，视觉分支基本学不到（用于报告/裁剪）
MIN_USABLE_INSTANCES = 30
UNUSABLE_CLASSES = [c["zh"] for c in TONGUE_CLASSES if c["n_inst"] < MIN_USABLE_INSTANCES]

# ----------------------------------------------------------------------------
# 3. 映射矩阵 M（21×10）—— 硬 0/1
#    列顺序严格等于 SYNDROMES_ZH：
#    [气虚, 血虚, 阴虚, 阳虚, 气滞, 血瘀, 痰湿, 湿热, 内风, 实热]
#
#    MATRIX_VERSION 沿革（每次改格必须升版，并在论文中报告用的是哪一版）：
#      2026-09-14  领域专家定稿版（34 个映射）
#      2026-09-15  修订版（领域专家逐格复核 + 四本舌诊文献交叉验证后的结论，见
#                  notes/mapping-open-questions.md 的结论表）
#                  · 剥苔舌   ：删「血虚」（A2=②）
#                  · 紫舌     ：加「阳虚」（B4）
#                  · 胖大舌   ：删「湿热」、加「阳虚」（A1=(c)乙，不拆红胖/淡胖）
#                  · 齿痕舌   ：加「阳虚」（B3）
#                  · 黄苔舌   ：删「痰湿」（B5）
#                  · 肾区凹陷 ：由「阴虚+阳虚」改为「气虚」（C 组结论）
#                  · 白苔舌   ：矩阵不变，但引入 B6 共现判据剔除"孤立白苔舌"（正常薄白）
#                  · 维持不变 ：红舌、瘦舌、裂纹舌、黑苔舌、滑苔舌、红点舌（A3=③维持并写 limitation）、
#                              及全部分区类除肾区凹陷外
# ----------------------------------------------------------------------------
MATRIX_VERSION = "2026-09-15"
HARD_MATRIX = [
    # 0  健康舌      不映射任何病性
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # 1  剥苔舌      阴虚
    #    ← 2026-09-15 修订：删「血虚」（领域专家 A2=②）。理由：血虚的视觉主指征是"淡白舌"，
    #      李灿东《中医诊断学》p82 亦要求"舌淡+剥苔"才指向血虚，而数据集无舌色类、无法判定
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # 2  红舌        阴虚、湿热、实热
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    # 3  紫舌        血瘀、阳虚
    #    ← 2026-09-15 修订：加「阳虚」（领域专家 B4）。李灿东：淡紫或青紫而湿润者，多属寒凝血瘀
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    # 4  胖大舌      气虚、痰湿、阳虚
    #    ← 2026-09-15 修订：删「湿热」、加「阳虚」（领域专家选 (c) 乙 —— 不拆红胖/淡胖类）。
    #      依据：李灿东「舌淡胖大者，多为脾肾阳虚」/ 王彦晖 p41 / 费兆馥 p20「淡胖舌属气虚、阳虚水肿」
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    # 5  瘦舌        阴虚
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # 6  红点舌      实热
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    # 7  裂纹舌      阴虚
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # 8  齿痕舌      气虚、痰湿、阳虚
    #    ← 2026-09-15 修订：加「阳虚」（领域专家 B3）。齿痕主脾虚水湿，脾肾阳虚水湿泛溢亦见齿痕
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    # 9  白苔舌      气虚、阳虚、痰湿
    #    ← 2026-09-15 追加 B6 判据（领域专家选 (a) 共现判据）：本行矩阵不变，
    #      但"孤立的"白苔舌（该图除健康舌外只有白苔舌一个标签）视为**正常薄白**，
    #      在进入映射前先剔除 → 见文件末 prune_normal_white_coating()
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    # 10 黄苔舌      湿热、实热
    #    ← 2026-09-15 修订：删「痰湿」（领域专家 B5）。痰湿的典型苔是**白腻**（李灿东 p82、张坚 p56/p80），
    #      黄苔主热，归痰湿属越界；且"痰湿+湿热"双计会稀释湿热的判别信号
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    # 11 黑苔舌      阳虚、实热
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    # 12 滑苔舌      阳虚、痰湿
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    # 13 肾区凹陷    气虚
    #    ← 2026-09-15 修订：由「阴虚、阳虚」改为「气虚（肾气虚）」（领域专家 C 组结论）。
    #      注意：本类 495 实例（可用），改动会动原型与指标；阴虚 5→4 源、阳虚 4→3 源、气虚 5→6 源
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # 14 肾区凸起    实热  ⚠️ 全数据集仅 2 实例，视觉分支实际不可用
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    # 15 肝胆区凹陷  血虚
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    # 16 肝胆区凸起  气滞  ⚠️ 全数据集仅 9 实例，视觉分支实际不可用 → 气滞在视觉侧等同无信号
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    # 17 脾胃区凹陷  气虚
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # 18 脾胃区凸起  气滞  ← 领域专家 2026-09-14 定：≠ 痰湿（痰湿更典型的是"苔厚腻"，非单纯凸起），
    #                       取"中焦气机壅滞"之意 → 气滞。⚠️ 数据 0 实例，仅补完整映射表
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    # 19 心肺区凹陷  气虚
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # 20 心肺区凸起  血瘀、痰湿  ⚠️ 全数据集仅 2 实例，视觉分支实际不可用
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
]

# 行归一化的软权重矩阵（可选增强）：每行映射的 1 → 1/该行映射数
def _row_normalized(matrix):
    out = []
    for row in matrix:
        s = sum(row)
        out.append([v / s for v in row] if s > 0 else list(row))
    return out


SOFT_MATRIX = _row_normalized(HARD_MATRIX)


# ----------------------------------------------------------------------------
# 3b. B6 判据：剔除"正常薄白苔"（领域专家 2026-09-15 选 (a) 共现判据）
#
#   背景：白苔舌覆盖全数据集 **75% 的图**，而李灿东《中医诊断学》p89 表 3-1 首行
#   明确「淡红舌、薄白苔 = 健康人」——薄白苔本身是正常舌象，白厚腻苔才主寒湿痰饮。
#   若不剔除，"白苔舌 → 气虚+阳虚+痰湿"会把这三列一并抬到 77~92%，
#   造成弱验证报告里观测到的**「下界效应」**（rate(j) ≥ max_{i∈S_j} rate(i)），
#   与真实人群特征无关。
#
#   判据 (a) 共现判据：若某图的标签集合（去掉"健康舌"后）**只有**白苔舌，
#   则该图判为"正常薄白"，白苔舌不计入映射；只要同图还有**任何**其他异常标签，
#   白苔舌就照常计入。
#
#   ⚠️ 该判据是对"苔厚薄"的**代理**（数据集只有 bbox，没有苔质标注），
#      论文中必须写明是代理判据、并做敏感性分析（可用 prune_normal_white=False 复现旧口径）。
# ----------------------------------------------------------------------------
B6_VERSION = "2026-09-15"

WHITE_COATING_ID = TONGUE_ZH_TO_ID["白苔舌"]     # 9
HEALTHY_TONGUE_ID = TONGUE_ZH_TO_ID["健康舌"]    # 0


def is_isolated_white_coating(labels) -> bool:
    """一组 21 类标签（类别 id 的可迭代对象）是否为"孤立白苔舌"。

    「孤立」= 去掉"健康舌"之后只剩"白苔舌"一个标签 → 依 B6 判据视为正常薄白。
    """
    s = set(int(x) for x in labels)
    s.discard(HEALTHY_TONGUE_ID)
    return s == {WHITE_COATING_ID}


def prune_normal_white_coating(labels) -> tuple:
    """类别 id 序列版：孤立白苔舌 → 剔除该标签；否则原样返回（tuple）。"""
    s = tuple(int(x) for x in labels)
    if is_isolated_white_coating(s):
        return tuple(x for x in s if x != WHITE_COATING_ID)
    return s


def prune_normal_white_coating_vec(det_vec) -> list:
    """21 维向量版：若只有"白苔舌"（允许同时有"健康舌"）非零，则把白苔舌清零。"""
    v = list(det_vec)
    nz = {i for i, x in enumerate(v) if x}
    if nz - {HEALTHY_TONGUE_ID} == {WHITE_COATING_ID}:
        v[WHITE_COATING_ID] = 0.0
    return v


def prune_normal_white_coating_batch(det_mat):
    """(N, 21) 批量版 B6 判据。

    对每一行做与 prune_normal_white_coating_vec 相同的判定：
    非零列集合 ⊆ {健康舌, 白苔舌} 且含白苔舌 → 该行白苔舌清零。

    参数
    ----
    det_mat : (N, 21) 的 list-of-list / numpy 数组 / torch 张量

    返回
    ----
    与输入同类型的新对象（不原地修改）。
    """
    # torch 张量：向量化实现（原型提取的主路径）
    try:
        import torch as _t
        if isinstance(det_mat, _t.Tensor):
            out = det_mat.clone()
            nz = out != 0
            white = nz[:, WHITE_COATING_ID]
            others = nz.clone()
            others[:, WHITE_COATING_ID] = False
            others[:, HEALTHY_TONGUE_ID] = False
            isolated = white & (~others.any(dim=1))
            out[isolated, WHITE_COATING_ID] = 0.0
            return out
    except ImportError:
        pass

    # numpy 数组
    try:
        import numpy as _np
        if isinstance(det_mat, _np.ndarray):
            out = det_mat.copy()
            nz = out != 0
            white = nz[:, WHITE_COATING_ID]
            others = nz.copy()
            others[:, WHITE_COATING_ID] = False
            others[:, HEALTHY_TONGUE_ID] = False
            isolated = white & (~others.any(axis=1))
            out[isolated, WHITE_COATING_ID] = 0.0
            return out
    except ImportError:
        pass

    # 纯 list 兜底
    return [prune_normal_white_coating_vec(row) for row in det_mat]


def count_isolated_white_coating(det_mat) -> int:
    """统计 (N, 21) 标签矩阵里有多少行是「孤立白苔舌」（供报告用）。"""
    try:
        import torch as _t
        if isinstance(det_mat, _t.Tensor):
            nz = det_mat != 0
            white = nz[:, WHITE_COATING_ID]
            others = nz.clone()
            others[:, WHITE_COATING_ID] = False
            others[:, HEALTHY_TONGUE_ID] = False
            return int((white & (~others.any(dim=1))).sum().item())
    except ImportError:
        pass
    return sum(
        1 for row in det_mat
        if ({i for i, x in enumerate(row) if x} - {HEALTHY_TONGUE_ID}) == {WHITE_COATING_ID}
    )


# ----------------------------------------------------------------------------
# 4. 核心函数
# ----------------------------------------------------------------------------
def tongue_to_syndrome(det_vec, mode="binary", matrix=None, normalize=False,
                       prune_normal_white=False):
    """把 21 维舌象检测向量投影到 10 病性空间。

    参数
    ----
    det_vec : 长度 21 的 0/1 向量（或任意非负权重，如检测置信度）
              顺序必须 = TONGUE_CLASSES 的 id 顺序（0-20，即 COCO category id）
    mode    : "binary"  → 硬映射（任一命中即置 1）
              "soft"    → 行归一化软权重（1/映射数），再按 det 加权求和
    matrix  : 自定义 21×10 矩阵；None 则按 mode 选 HARD_MATRIX / SOFT_MATRIX
    normalize : soft 模式下是否把结果归一到 [0,1]（按其最大可能值）
    prune_normal_white : True → 先按 B6 共现判据剔除"孤立白苔舌"（正常薄白苔）
              默认 **False**（保持纯矩阵语义，便于做敏感性分析）；
              实验流程（数据加载 / 原型提取）请显式传 True。

    返回
    ----
    长度 10 的实值向量（顺序 = SYNDROMES_ZH）
    """
    if len(det_vec) != NUM_TONGUE_CLASSES:
        raise ValueError(f"det_vec 长度应为 {NUM_TONGUE_CLASSES}，得到 {len(det_vec)}")

    if prune_normal_white:
        det_vec = prune_normal_white_coating_vec(det_vec)

    if matrix is None:
        matrix = HARD_MATRIX if mode == "binary" else SOFT_MATRIX

    out = [0.0] * NUM_SYNDROMES
    for i, present in enumerate(det_vec):
        if not present:
            continue
        w = float(present)  # 支持置信度加权
        for j in range(NUM_SYNDROMES):
            out[j] += w * matrix[i][j]

    if mode == "binary":
        # 硬模式：取值 >0 即视为命中，输出 0/1（保留多标签语义）
        out = [1.0 if v > 0 else 0.0 for v in out]
    elif normalize:
        # 归一：除以"若全部舌象都命中"的理论最大值
        denom = [sum(row[j] for row in matrix) for j in range(NUM_SYNDROMES)]
        out = [out[j] / denom[j] if denom[j] > 0 else 0.0 for j in range(NUM_SYNDROMES)]
    return out


def get_matrix_torch(dtype=None, device=None, mode="binary"):
    """返回 torch 张量形式的映射矩阵（供视觉分支直接矩阵乘用）。"""
    import torch  # 延迟导入，保持本模块可无 torch 运行

    m = HARD_MATRIX if mode == "binary" else SOFT_MATRIX
    t = torch.tensor(m, dtype=dtype or torch.float32)
    return t.to(device) if device is not None else t


def mapping_stats():
    """返回覆盖统计：每个病性的来源数 + 每个舌象的映射数。"""
    per_syndrome = {
        SYNDROMES_ZH[j]: sum(row[j] for row in HARD_MATRIX)
        for j in range(NUM_SYNDROMES)
    }
    per_class = {
        TONGUE_CLASSES[i]["zh"]: int(sum(HARD_MATRIX[i]))
        for i in range(NUM_TONGUE_CLASSES)
    }
    return per_syndrome, per_class


# ----------------------------------------------------------------------------
# 5. 自测
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TCM-Tongue 映射矩阵自测")
    print(f"  MATRIX_VERSION = {MATRIX_VERSION} | B6_VERSION = {B6_VERSION}")
    print("=" * 60)

    # 维度校验
    assert len(HARD_MATRIX) == NUM_TONGUE_CLASSES, "矩阵行数应为 21"
    assert all(len(r) == NUM_SYNDROMES for r in HARD_MATRIX), "矩阵列数应为 10"
    assert [c["id"] for c in TONGUE_CLASSES] == list(range(21)), "舌象 id 必须 0-20 连续（对齐 COCO category id）"
    print(f"OK 维度：{NUM_TONGUE_CLASSES} 类舌象 × {NUM_SYNDROMES} 病性")

    # 健康舌：不映射任何病性
    v = [0] * 21; v[0] = 1
    r = tongue_to_syndrome(v)
    assert sum(r) == 0, "健康舌不应映射任何病性"
    print("OK 健康舌 → 全阴性")

    # 紫舌：血瘀 + 阳虚（2026-09-15 修订：加阳虚 —— 淡紫/青紫而润属寒凝血瘀）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["紫舌"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"血瘀", "阳虚"}, f"紫舌应为 血瘀+阳虚，得到 {got}"
    print(f"OK 紫舌 → {sorted(got)}")

    # 剥苔舌：仅阴虚（2026-09-15 修订：删血虚 —— 血虚视觉主指征是淡白舌，数据集无舌色类）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["剥苔舌"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"阴虚"}, f"剥苔舌应为 阴虚，得到 {got}"
    print(f"OK 剥苔舌 → {sorted(got)}")

    # 胖大舌：气虚 + 痰湿 + 阳虚（2026-09-15 修订：删湿热、加阳虚）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["胖大舌"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"气虚", "痰湿", "阳虚"}, f"胖大舌应为 气虚+痰湿+阳虚，得到 {got}"
    print(f"OK 胖大舌 → {sorted(got)}")

    # 齿痕舌：气虚 + 痰湿 + 阳虚（2026-09-15 修订：加阳虚）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["齿痕舌"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"气虚", "痰湿", "阳虚"}, f"齿痕舌应为 气虚+痰湿+阳虚，得到 {got}"
    print(f"OK 齿痕舌 → {sorted(got)}")

    # 黄苔舌：湿热 + 实热（2026-09-15 修订：删痰湿）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["黄苔舌"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"湿热", "实热"}, f"黄苔舌应为 湿热+实热，得到 {got}"
    print(f"OK 黄苔舌 → {sorted(got)}")

    # 肾区凹陷：仅气虚（2026-09-15 修订：原 阴虚+阳虚 → 改为 肾气虚）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["肾区凹陷"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"气虚"}, f"肾区凹陷应为 气虚，得到 {got}"
    print(f"OK 肾区凹陷 → {sorted(got)}（肾气虚）")

    # 红舌：阴虚 + 湿热 + 实热（领域专家新增湿热）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["红舌"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"阴虚", "湿热", "实热"}, f"红舌应为 阴虚+湿热+实热，得到 {got}"
    print(f"OK 红舌 → {sorted(got)}")

    # 心肺区凸起：血瘀 + 痰湿（领域专家把"气滞、实热"改为"血瘀、痰湿"）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["心肺区凸起"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"血瘀", "痰湿"}, f"心肺区凸起应为 血瘀+痰湿，得到 {got}"
    print(f"OK 心肺区凸起 → {sorted(got)}")

    # 脾胃区凸起 → 气滞（领域专家 2026-09-14 定：≠ 痰湿）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["脾胃区凸起"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"气滞"}, f"脾胃区凸起应为 气滞，得到 {got}"
    print(f"OK 脾胃区凸起 → {sorted(got)}（领域专家定：非痰湿）")

    # 内风列必须全 0（数据集无舌态标签）
    assert all(row[SYNDROMES_ZH.index("内风")] == 0 for row in HARD_MATRIX), "内风列应全 0"
    print("OK 内风列全 0（数据集覆盖缺口）")

    # 软模式：紫舌（2 个映射）→ 血瘀/阳虚 各 0.5
    v = [0] * 21; v[TONGUE_ZH_TO_ID["紫舌"]] = 1
    r = tongue_to_syndrome(v, mode="soft")
    assert abs(r[SYNDROMES_ZH.index("血瘀")] - 0.5) < 1e-6, "紫舌软模式：血瘀应 0.5（2 个映射）"
    print("OK 软模式：紫舌 → 血瘀/阳虚 各 0.5")

    # 软模式：白苔舌（3 个映射）→ 各 1/3
    v = [0] * 21; v[TONGUE_ZH_TO_ID["白苔舌"]] = 1
    r = tongue_to_syndrome(v, mode="soft")
    assert abs(r[SYNDROMES_ZH.index("气虚")] - 1 / 3) < 1e-6
    print("OK 软模式：白苔舌 → 气虚/阳虚/痰湿 各 1/3")

    # 多标签组合：白苔舌 + 黄苔舌
    v = [0] * 21; v[TONGUE_ZH_TO_ID["白苔舌"]] = 1; v[TONGUE_ZH_TO_ID["黄苔舌"]] = 1
    r = tongue_to_syndrome(v)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"气虚", "阳虚", "痰湿", "湿热", "实热"}, f"组合映射异常：{got}"
    print(f"OK 多标签（白苔+黄苔）→ {sorted(got)}")

    # ---- B6 判据：剔除"正常薄白苔" ----
    print("-" * 60)
    # ① 孤立白苔舌（除健康舌外只有白苔舌）→ 判为正常薄白，剔除后不映射任何病性
    v = [0] * 21; v[TONGUE_ZH_TO_ID["白苔舌"]] = 1
    assert is_isolated_white_coating([WHITE_COATING_ID]) is True
    r = tongue_to_syndrome(v, prune_normal_white=True)
    assert sum(r) == 0, "孤立白苔舌（正常薄白）应不映射任何病性"
    print("OK B6：孤立白苔舌 → 正常薄白，不映射")

    # ② 白苔舌 + 健康舌 仍算孤立
    v = [0] * 21
    v[TONGUE_ZH_TO_ID["白苔舌"]] = 1; v[TONGUE_ZH_TO_ID["健康舌"]] = 1
    assert is_isolated_white_coating([WHITE_COATING_ID, HEALTHY_TONGUE_ID]) is True
    assert sum(tongue_to_syndrome(v, prune_normal_white=True)) == 0
    print("OK B6：白苔舌 + 健康舌 → 仍视为正常薄白")

    # ③ 白苔舌与任一异常类共现 → 照常映射
    v = [0] * 21
    v[TONGUE_ZH_TO_ID["白苔舌"]] = 1; v[TONGUE_ZH_TO_ID["齿痕舌"]] = 1
    assert is_isolated_white_coating([WHITE_COATING_ID, TONGUE_ZH_TO_ID["齿痕舌"]]) is False
    r = tongue_to_syndrome(v, prune_normal_white=True)
    got = {SYNDROMES_ZH[j] for j in range(10) if r[j]}
    assert got == {"气虚", "阳虚", "痰湿"}, f"白苔+齿痕 应仍映射，得到 {got}"
    print(f"OK B6：白苔舌+齿痕舌 → 照常映射 {sorted(got)}")

    # ④ 默认不开 B6（保持纯矩阵语义，便于敏感性分析）
    v = [0] * 21; v[TONGUE_ZH_TO_ID["白苔舌"]] = 1
    assert sum(tongue_to_syndrome(v)) == 3, "默认 prune_normal_white=False 时应按原矩阵映射 3 个病性"
    print("OK B6：默认关闭（prune_normal_white=False），可用于敏感性分析")

    # ⑤ 批量版：与逐行版结果一致 + 计数正确
    batch = [
        [0] * 21,  # 全阴
        [1 if i == WHITE_COATING_ID else 0 for i in range(21)],          # 孤立白苔 → 清零
        [1 if i in (WHITE_COATING_ID, HEALTHY_TONGUE_ID) else 0 for i in range(21)],  # 白苔+健康 → 清零
        [1 if i in (WHITE_COATING_ID, TONGUE_ZH_TO_ID["齿痕舌"]) else 0 for i in range(21)],  # 共现 → 保留
    ]
    pruned = prune_normal_white_coating_batch(batch)
    assert pruned[1][WHITE_COATING_ID] == 0 and pruned[2][WHITE_COATING_ID] == 0
    assert pruned[3][WHITE_COATING_ID] == 1, "共现白苔不应被清"
    assert count_isolated_white_coating(batch) == 2
    assert count_isolated_white_coating(pruned) == 0
    print("OK B6：批量版与计数函数正确（list 路径）")
    try:
        import torch as _t
        tb = _t.tensor(batch, dtype=_t.float32)
        tp = prune_normal_white_coating_batch(tb)
        assert isinstance(tp, _t.Tensor) and tp.shape == tb.shape
        assert tp[1, WHITE_COATING_ID].item() == 0 and tp[3, WHITE_COATING_ID].item() == 1
        assert count_isolated_white_coating(tb) == 2
        print("OK B6：批量版与计数函数正确（torch 路径）")
    except ImportError:
        print("-- B6：torch 不可用，跳过 torch 路径自测")

    # 覆盖统计
    per_syn, per_cls = mapping_stats()
    print("-" * 60)
    print("每病性来源数：", {k: v for k, v in per_syn.items()})
    print("总映射数：", sum(per_syn.values()))

    # 数据可用性（关键：凸起类近乎空 → 部分病性在视觉侧实际无信号）
    print("-" * 60)
    print(f"⚠️ 视觉分支不可用类（实例数 < {MIN_USABLE_INSTANCES}）：")
    for c in TONGUE_CLASSES:
        if c["n_inst"] < MIN_USABLE_INSTANCES:
            print(f"    id={c['id']:2d} {c['zh']:8s} 仅 {c['n_inst']:4d} 实例")
    usable_rows = [i for i, c in enumerate(TONGUE_CLASSES) if c["n_inst"] >= MIN_USABLE_INSTANCES]
    live = [SYNDROMES_ZH[j] for j in range(NUM_SYNDROMES)
            if any(HARD_MATRIX[i][j] for i in usable_rows)]
    dead = [s for s in SYNDROMES_ZH if s not in live]
    print(f"✅ 视觉侧有信号的病性（{len(live)}/10）：{live}")
    print(f"❌ 视觉侧无信号的病性（{len(dead)}/10）：{dead}")
    print("=" * 60)
    print("全部自测通过 ✓")
