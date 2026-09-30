"""
KG（知识图谱）嵌入模块
======================

基于净安校准的 TCM 知识图谱：
- 节点类型：14 观察要素 + 10 病性要素 + 6 脏腑 + 20 主诉（共 50 节点）
- 关系类型：
    - 舌象 → 证候（is_indicator_of）
    - 证候 → 证候（similar_to / differential_with）
    - 证候 → 脏腑（maps_to）
    - 主诉 → 证候（suggests）
    - 舌区 → 脏腑（maps_to_region）

使用 PyKEEN 训练 TransE 嵌入。

参考：
- PyKEEN 文档：https://pykeen.readthedocs.io/
- Bordes et al. 2013, "Translating Embeddings for Modeling Multi-relational Data" (TransE)
- 净安校准的 KG 节点见 papers/templates/tcm-knowledge-graph.md
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import torch
import torch.nn as nn


# ---------- 节点定义（唯一权威来源在 kg_keywords.py，这里仅 re-export） ----------

try:
    from .kg_keywords import (
        NODES_OBSERVATION, NODES_SYNDROME, NODES_ORGAN,
        NODES_CHIEF_COMPLAINT, ALL_NODES, NODE_TO_INDEX,
    )
except ImportError:  # 作为脚本直接运行时的兜底
    from kg_keywords import (
        NODES_OBSERVATION, NODES_SYNDROME, NODES_ORGAN,
        NODES_CHIEF_COMPLAINT, ALL_NODES, NODE_TO_INDEX,
    )


# ---------- 关系定义 ----------

RELATION_INDICATES = "is_indicator_of"     # 观察 → 证候
RELATION_SIMILAR = "similar_to"             # 证候 → 证候
RELATION_DIFFERENTIAL = "differential_with" # 证候 → 证候
RELATION_MAPS_ORGAN = "maps_to_organ"       # 证候 → 脏腑
RELATION_SUGGESTS = "suggests"              # 主诉 → 证候
RELATION_MAPS_REGION = "maps_to_region"     # 区域 → 脏腑


# ---------- 边构造（基于 8 个 JSON + 维度映射） ----------

def build_triples_from_syndromes(data_dir: Path) -> List[Tuple[str, str, str]]:
    """
    从 10 个证型 JSON 自动构建三元组 (head, relation, tail)。

    Args:
        data_dir: papers/code/ 目录

    Returns:
        List of (head, relation, tail)
    """
    triples = []

    # 1. 观察要素 → 证候（从 tongue_expectation 提取）
    SYNDROME_FILE = {
        "气虚": "syndrome-qi-deficiency.json",
        "血虚": "syndrome-blood-deficiency.json",
        "阴虚": "syndrome-yin-deficiency.json",
        "阳虚": "syndrome-yang-deficiency.json",
        "气滞": "syndrome-qi-stagnation.json",
        "血瘀": "syndrome-blood-stasis.json",
        "痰湿": "syndrome-phlegm-dampness.json",
        "湿热": "syndrome-damp-heat.json",
        "内风": "syndrome-internal-wind.json",
        "实热": "syndrome-excess-heat.json",
    }
    SYNDROME_EN = {
        "气虚": "qi_deficiency", "血虚": "blood_deficiency",
        "阴虚": "yin_deficiency", "阳虚": "yang_deficiency",
        "气滞": "qi_stagnation", "血瘀": "blood_stasis",
        "痰湿": "phlegm_dampness", "湿热": "damp_heat",
        "内风": "internal_wind", "实热": "excess_heat",
    }
    # 维度映射：中文 → 英文节点名
    DIM_TO_NODE = {
        "舌色": "tongue_color",
        "舌形": "tongue_shape",
        "齿痕": "teeth_marks",
        "舌质": "tongue_moisture",
        "舌态": "tongue_posture",
        "苔色": "coating_color",
        "苔厚薄": "coating_thickness",
        "苔润燥": "coating_moisture",
        "苔腻腐剥落": "coating_texture",
        "舌尖": "region_tip",
        "舌中": "region_center",
        "舌根": "region_root",
        "舌两侧": "region_side",
        "舌下络脉": "sublingual_vein",
    }

    for cn_name, filename in SYNDROME_FILE.items():
        json_path = data_dir / filename
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        syndrome_en = SYNDROME_EN[cn_name]

        # 观察要素 → 证候
        for te in data.get("tongue_expectation", []):
            dim = te["dimension"]
            if dim in DIM_TO_NODE:
                head = DIM_TO_NODE[dim]
                # 只保留 confidence = "高" 的边，避免噪声
                if te.get("confidence") == "高":
                    triples.append((head, RELATION_INDICATES, syndrome_en))

        # 证候 → 脏腑（基于 pathogenesis 文本推断）
        # === Buddy 草稿，待净安校准 ===
        organ_map = {
            "气虚":   ["spleen", "lung"],
            "血虚":   ["spleen", "heart", "liver"],
            "阴虚":   ["kidney", "lung", "heart"],
            "阳虚":   ["kidney", "spleen"],
            "气滞":   ["liver"],
            "血瘀":   ["heart", "liver"],
            "痰湿":   ["spleen", "lung"],
            "湿热":   ["spleen", "liver", "stomach"],
            "内风":   ["liver", "kidney"],
            "实热":   ["heart", "liver", "stomach"],
        }
        for organ in organ_map.get(cn_name, []):
            triples.append((syndrome_en, RELATION_MAPS_ORGAN, organ))

    # 2. 区域 → 脏腑（中医舌诊经典分区）
    REGION_ORGAN = [
        ("region_tip", RELATION_MAPS_REGION, "heart"),
        ("region_center", RELATION_MAPS_REGION, "spleen"),
        ("region_root", RELATION_MAPS_REGION, "kidney"),
        ("region_side", RELATION_MAPS_REGION, "liver"),
    ]
    triples.extend(REGION_ORGAN)

    # 3. 证候相似度（基于净安校准定稿的 10×10 矩阵，取 ≥ 0.3 的为 similar_to）
    try:
        from .soft_cl_loss import default_syndrome_similarity_matrix
    except ImportError:  # 作为脚本直接运行时的兜底
        from soft_cl_loss import default_syndrome_similarity_matrix
    sim_matrix = default_syndrome_similarity_matrix()
    n = len(NODES_SYNDROME)
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j].item()
            if sim >= 0.3:  # 阈值：相似度 ≥ 0.3 视为"similar_to"
                triples.append((NODES_SYNDROME[i], RELATION_SIMILAR, NODES_SYNDROME[j]))

    # 4. 主诉 → 证候（基于 10 JSON 的 inquiries 字段提取）
    # === 待 Phase 2 实现：解析 anchor 字段（如 ✓气虚 → 主诉 suggests 气虚）===

    return triples


def train_kg_embeddings(
    triples: List[Tuple[str, str, str]],
    embedding_dim: int = 64,
    epochs: int = 100,
    output_path: Optional[Path] = None,
):
    """
    用 PyKEEN 训练 TransE 嵌入。

    Args:
        triples: 三元组列表
        embedding_dim: 嵌入维度（论文中应尝试 32/64/128）
        epochs: 训练轮数
        output_path: 嵌入保存路径（默认 papers/code/models/kg_embeddings.pt）

    Returns:
        entity_to_idx: {实体名: 索引}
        relation_to_idx: {关系名: 索引}
        embeddings: Tensor[num_entities, embedding_dim]
    """
    try:
        from pykeen.models import TransE
        from pykeen.pipeline import pipeline
        from pykeen.triples import TriplesFactory
    except ImportError:
        print("⚠️  PyKEEN 未安装，本机阶段无需安装，Phase 2 上 GPU 时再装。")
        print("   pip install pykeen")
        return None, None, None

    # === Phase 2 实现 ===
    # tf = TriplesFactory.from_labeled_triples(triples_array, ...)
    # result = pipeline(
    #     training=tf_training,
    #     model='TransE',
    #     model_kwargs={'embedding_dim': embedding_dim},
    #     epochs=epochs,
    # )
    # embeddings = result.model.entity_representations[0]().detach().cpu()
    raise NotImplementedError("Phase 2 实现（PyKEEN 训练）")


class KGEmbeddingModule(nn.Module):
    """
    训练时使用的 KG 嵌入模块。

    功能：
    1. 加载预训练的 KG 嵌入
    2. 提供 lookup 接口（forward 按索引查 / lookup_by_name 按实体名查）
    3. 提供"知识增强"接口（把 KG 嵌入拼接到模型中间层）
    """

    def __init__(
        self,
        num_entities: Optional[int] = None,
        embedding_dim: int = 64,
        embeddings_path: Optional[Path] = None,
    ):
        """
        Args:
            num_entities: 实体总数（默认 = len(ALL_NODES) = 50）。
            embedding_dim: 嵌入维度（受 PyKEEN 配置，论文尝试 32/64/128）。
            embeddings_path: 预训练嵌入路径（来自 train_kg_embeddings），
                None 则随机初始化（Phase 1 不实际训练）。
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities if num_entities is not None else len(ALL_NODES)
        self.entity_to_idx = {n: i for i, n in enumerate(ALL_NODES)}

        if embeddings_path is not None and Path(embeddings_path).exists():
            # === Phase 2 实现：从文件加载 ===
            # state = torch.load(embeddings_path)
            # self.embeddings = nn.Embedding.from_pretrained(state['embeddings'])
            # self.entity_to_idx = state['entity_to_idx']
            raise NotImplementedError("Phase 2 实现：从 PyKEEN checkpoint 加载")
        else:
            # 占位：随机初始化（Phase 1 不实际训练）
            self.embeddings = nn.Embedding(self.num_entities, embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        查询实体嵌入（按整数索引）。

        Args:
            indices: LongTensor[B]，实体索引 ∈ [0, num_entities)

        Returns:
            Tensor[B, embedding_dim]
        """
        return self.embeddings(indices.to(self.embeddings.weight.device))

    def lookup_by_name(self, entity_names: List[str]) -> torch.Tensor:
        """按实体名查嵌入（便捷接口，供 KG 桥接 / 调试用）。"""
        indices = torch.tensor([self.entity_to_idx[n] for n in entity_names])
        return self.embeddings(indices)


# ---------- 单元测试 ----------

if __name__ == "__main__":
    print("Testing KGEmbeddingModule...")

    # 1. 构造三元组
    data_dir = Path(__file__).parent.parent
    triples = build_triples_from_syndromes(data_dir)
    print(f"\n✓ 构造了 {len(triples)} 条三元组")

    # 2. 显示前 10 条
    print("\n样例三元组（前 10 条）：")
    for h, r, t in triples[:10]:
        print(f"  ({h}, {r}, {t})")

    # 3. 统计
    from collections import Counter
    rel_counter = Counter(r for _, r, _ in triples)
    print(f"\n关系类型分布：")
    for rel, count in rel_counter.most_common():
        print(f"  {rel}: {count}")