"""层规划与 43 层实例绑定（工作包 B）：四组 2/1/20/20 + 全部实例差异。

:func:`derive_layer_groups`（rules）给四组划分与组内模板状态；
本模块在其上加：
- 每实例绑定：layer 4 绑定已构建的 csa_topk 模板（build_reference_graph
  通过才允许 extracted）；其余 42 层绑定同组模板但状态 pending_extraction
  ——分组正确性已由 compress_ratios + n_hash_layers 推导（derive_layer_groups
  具名校验），模板图未建不冒充完整结构。
- 参数与 state_id 逐实例唯一：params 带 layer_id，state_ids 带层命名空间。
- 40/41/42 的 DSpark 外部读出关系（model.py:917-921：target_layer_ids 命中
  层的 hc 流均值写入 main_hiddens，DSpark forward_embed 消费，model.py:851-853
  ——只记录外部关系，DSpark 内部图属 V2）。
- 未覆盖清单：其余三类模板 pending_extraction；kernel 内部数值与外部
  fast_hadamard_transform 为 uncovered 边界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from structure_reader.extract.deepseek_v4_reference import (
    REPRESENTATIVE_LAYER,
    TEMPLATE_ID,
    build_reference_graph,
)
from structure_reader.rules.deepseek_v4_flash_0731 import (
    LayerGroup,
    ModelValues,
    derive_layer_groups,
    load_model_values,
)

__all__ = [
    "LayerPlanError",
    "LayerPlan",
    "LayerInstanceBinding",
    "DSparkExternalReadout",
    "TemplateCoverage",
    "build_layer_plan",
    "assert_dspark_readout_binding",
]

REPRESENTATIVE_GROUP = "csa_topk"


class LayerPlanError(ValueError):
    """层规划失败：提取失败或绑定不一致（具名拒绝）。"""


@dataclass(frozen=True)
class LayerInstanceBinding:
    """一个主干层的实例绑定：模板 ID、逐实例参数与状态命名空间。"""

    layer_id: int
    template_id: str | None  # None = 模板尚未提取
    template_status: str  # extracted | pending_extraction
    params: dict[str, int]
    state_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "template_id": self.template_id,
            "template_status": self.template_status,
            "params": dict(self.params),
            "state_ids": list(self.state_ids),
        }


@dataclass(frozen=True)
class DSparkExternalReadout:
    """DSpark 对主干层的外部读出关系（不含 DSpark 内部图）。

    实际语义（B4 集成后；不再一一对应 draft stage）：三个读出层按主干
    遍历顺序（40→41→42）有序读出后 concat 进 stage0（mtp.0）唯一的
    main_proj/main_norm，产出唯一 main_x 再共享分发三 stage（model.py:
    917-921 收集、926 concat、832 main_proj、931-932 透传）。
    """

    readout_layers: tuple[int, ...]   # 有序（主干遍历顺序 40,41,42）
    consumer_stage_id: int            # concat/proj/norm 所在实例（恒 0）
    shared_consumers: tuple[int, ...]  # main_x 共享消费的全部 stage 实例
    evidence_ref: str

    def to_dict(self) -> dict:
        return {
            "readout_layers": list(self.readout_layers),
            "consumer_stage_id": self.consumer_stage_id,
            "shared_consumers": list(self.shared_consumers),
            "evidence_ref": self.evidence_ref,
            "concat_note": (
                f"三个有序 readout（{', '.join(map(str, self.readout_layers))}）"
                f"共同 concat 进 stage{self.consumer_stage_id} 唯一 "
                f"main_proj/main_norm，产出唯一 main_x 后共享分发三 "
                f"stage{list(self.shared_consumers)}（不是每 stage 一个"
                f" readout 的一对一映射）"),
        }


@dataclass(frozen=True)
class TemplateCoverage:
    """模板覆盖清单：extracted / pending_extraction / uncovered 三态齐备。"""

    extracted: tuple[str, ...]
    pending_extraction: tuple[str, ...]
    uncovered: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "extracted": list(self.extracted),
            "pending_extraction": list(self.pending_extraction),
            "uncovered": list(self.uncovered),
        }


@dataclass(frozen=True)
class LayerPlan:
    """层规划：四组、43 实例绑定、DSpark 外部读出与覆盖清单。"""

    groups: tuple[LayerGroup, ...]
    bindings: tuple[LayerInstanceBinding, ...]
    dspark_readouts: tuple[DSparkExternalReadout, ...]
    coverage: TemplateCoverage

    def to_dict(self) -> dict:
        return {
            "groups": [
                {
                    "group_id": g.group_id,
                    "attention_kind": g.attention_kind,
                    "gate_kind": g.gate_kind,
                    "compress_ratio": g.compress_ratio,
                    "has_indexer": g.has_indexer,
                    "member_layers": list(g.member_layers),
                    "template_status": g.template_status,
                    "note": g.note,
                }
                for g in self.groups
            ],
            "bindings": [bnd.to_dict() for bnd in self.bindings],
            "dspark_readouts": [r.to_dict() for r in self.dspark_readouts],
            "coverage": self.coverage.to_dict(),
        }


def _instance_params(values: ModelValues, group: LayerGroup,
                     layer_id: int) -> dict[str, int]:
    """逐实例参数：与组模板共享的全部常量 + layer_id 个体差异。"""
    return {
        "layer_id": layer_id,
        "compress_ratio": group.compress_ratio,
        "gate_kind_is_hash": 1 if layer_id < values.n_hash_layers else 0,
        "H": values.dim, "N_hc": values.hc_mult, "Hq": values.n_heads,
        "Hkv": 1, "D": values.head_dim, "D_rope": values.rope_head_dim,
        "D_nope": values.head_dim - values.rope_head_dim,
        "Rq": values.q_lora_rank, "Ro": values.o_lora_rank,
        "G": values.o_groups, "W": values.window_size,
        "Ki": values.index_topk, "Hi": values.index_n_heads,
        "Di": values.index_head_dim, "E": values.n_routed_experts,
        "K": values.n_activated_experts, "I": values.moe_inter_dim,
        "Mcap": values.max_seq_len,
    }


def _instance_state_ids(group: LayerGroup, layer_id: int) -> tuple[str, ...]:
    """逐实例状态命名空间：与模板内 state_id 同构、层前缀区分。"""
    if group.attention_kind == "swa":
        return (f"layer_{layer_id}:attn_kv_cache",)
    if group.attention_kind == "hca":
        return (
            f"layer_{layer_id}:attn_kv_cache",
            f"layer_{layer_id}:compress_kv_state",
            f"layer_{layer_id}:compress_score_state",
        )
    # CSA（含学习式 Indexer）
    return (
        f"layer_{layer_id}:attn_kv_cache",
        f"layer_{layer_id}:compress_kv_state",
        f"layer_{layer_id}:compress_score_state",
        f"layer_{layer_id}:indexer_kv_cache",
        f"layer_{layer_id}:indexer_compress_kv_state",
        f"layer_{layer_id}:indexer_compress_score_state",
    )


def build_layer_plan(
    workspace_root: Path,
    *,
    values: ModelValues | None = None,
    build_graph: bool = True,
) -> LayerPlan:
    """构建层规划；build_graph=True 时真实构建代表图（失败则整体失败）。

    唯一标 extracted 的路径：csa_topk 的 build_reference_graph 成功（模板图
    已通过通用引用检查）。其余分组保持 pending_extraction——即使其成员层
    的分组归属已确定，模板图未建就不冒充完整结构。
    """
    workspace_root = Path(workspace_root)
    values = values or load_model_values(workspace_root)

    extracted: tuple[str, ...] = ()
    if build_graph:
        # V2：四类变体全部真实构建（任一失败即整体失败——extracted 只标
        # 真实提取成功的模板）；csa_topk 委托 V1 提取器
        from structure_reader.extract.deepseek_v4_variants import (
            build_all_variant_graphs, VARIANT_SPECS)
        build_all_variant_graphs(workspace_root, values=values)
        extracted = tuple(
            spec.template_id for spec in VARIANT_SPECS.values())
        # D1：真实提取成功后组记录必须与提取事实一致（rules 的
        # extracted_groups 参数；build_graph=False 仍全 pending）
        groups = derive_layer_groups(values,
                                     extracted_groups=set(VARIANT_SPECS))
    else:
        groups = derive_layer_groups(values)

    bindings: list[LayerInstanceBinding] = []
    group_by_layer: dict[int, LayerGroup] = {}
    for g in groups:
        for layer_id in g.member_layers:
            group_by_layer[layer_id] = g

    # V2：四类代表层（各组的代表层 extracted；其余实例 pending——模板
    # 覆盖不等于逐实例数值验证）
    from structure_reader.extract.deepseek_v4_variants import VARIANT_SPECS
    representative_by_group = {
        spec.key: (spec.representative_layer, spec.template_id)
        for spec in VARIANT_SPECS.values()
    }
    for layer_id in range(values.n_layers):
        group = group_by_layer[layer_id]
        rep_layer, rep_template = representative_by_group.get(
            group.group_id, (None, None))
        is_representative = (
            extracted and rep_layer is not None and layer_id == rep_layer)
        status = "extracted" if is_representative else "pending_extraction"
        bindings.append(LayerInstanceBinding(
            layer_id=layer_id,
            template_id=rep_template if is_representative else None,
            template_status=status,
            params=_instance_params(values, group, layer_id),
            state_ids=_instance_state_ids(group, layer_id),
        ))

    # 三个读出层有序 concat 进 stage0 唯一 main_proj/main_norm，唯一
    # main_x 共享分发三 stage（B4 集成实际接线；断言见
    # assert_dspark_readout_binding）——不是旧一对一 readout→stage 映射
    readouts = (
        DSparkExternalReadout(
            readout_layers=tuple(values.dspark_target_layer_ids),
            consumer_stage_id=0,
            shared_consumers=tuple(
                range(len(values.dspark_target_layer_ids))),
            evidence_ref="model.py:917-921,851-853,832,931-932",
        ),
    )

    coverage = TemplateCoverage(
        extracted=extracted,
        pending_extraction=(),
        uncovered=(
            "kernel.py: sparse_attn_kernel / hc_split_sinkhorn_kernel 的"
            "数值实现（不透明 kernel 边界，仅接口签名已核对）",
            "外部库 fast_hadamard_transform（Hadamard 旋转数值）",
            "每专家实际 token 数 N_e（数据相关，静态分析未知）",
            "Indexer 实际可见条目数（依赖历史调用序列，C unknown）",
        ),
    )
    return LayerPlan(groups=groups, bindings=tuple(bindings),
                     dspark_readouts=readouts, coverage=coverage)


def assert_dspark_readout_binding(plan: LayerPlan, dspark_graph) -> None:
    """层规划 DSpark 读出关系与实际集成图连接一致断言（防旧一对一映射
    回潮：规划声明与图真实连接任一侧漂移即具名拒绝）。

    核对项（与 dspark_integration 真实接线同源）：
    - 规划恰一条 readout 记录，readout_layers 有序等于 run config
      target 40/41/42；
    - 集成图 entry.readout_concat 的 s0/s1/s2 端口依序消费
      readout_mean_40/41/42（有序 concat）；
    - 集成图 main_x 由 entry.main_norm 唯一生产（经 stage_feed 转发）
      分发三 stage——与 shared_consumers = (0,1,2) 一致。
    """
    from structure_reader.checks.dspark_contracts import (
        _edges_into, _tensor_producers)

    if len(plan.dspark_readouts) != 1:
        raise LayerPlanError(
            f"DSpark readout 规划必须恰一条记录（三读出共同 concat），"
            f"得到 {len(plan.dspark_readouts)} 条")
    rec = plan.dspark_readouts[0]
    edges = {e.stable_id: e for e in dspark_graph.edges}
    # 有序 concat：s0/s1/s2 端口 → readout_mean_40/41/42
    for port, mean_node, want_tensor in (
            ("s0", "entry.readout_mean_40", "tensor:entry:mean_h_40"),
            ("s1", "entry.readout_mean_41", "tensor:entry:mean_h_41"),
            ("s2", "entry.readout_mean_42", "tensor:entry:mean_h_42")):
        got = _edges_into(dspark_graph, "entry.readout_concat", port)
        if len(got) != 1 or got[0].src_node_id != mean_node \
                or got[0].tensor_id != want_tensor:
            raise LayerPlanError(
                f"层规划/集成图读出不一致：concat 端口 {port} 必须消费 "
                f"{mean_node}（{want_tensor}），得到 "
                f"{[(e.src_node_id, e.tensor_id) for e in got]}")
    # main_x 唯一生产 + shared_consumers 数量一致
    mx = _tensor_producers(dspark_graph, "tensor:entry:main_x")
    fwd = [e for e in mx if e.src_port == "main_x_out"]
    produced = [e for e in mx if e.stable_id not in {f.stable_id for f in fwd}]
    if [e.src_node_id for e in produced] != ["entry.main_norm"] * len(rec.shared_consumers):
        raise LayerPlanError(
            "层规划/集成图读出不一致：main_x 必须由 entry.main_norm 唯一"
            f"生产，得到 {[(e.src_node_id, e.dst_node_id) for e in produced]}")
    if len(fwd) != len(rec.shared_consumers):
        raise LayerPlanError(
            f"main_x 转发边数（{len(fwd)}）与规划共享消费 stage 数"
            f"（{len(rec.shared_consumers)}）不一致")
    edges_ref = [e for e in produced]
    _ = edges  # 明确 edges 供未来扩展；当前核对经 _edges_into/_tensor_producers
    _ = edges_ref
