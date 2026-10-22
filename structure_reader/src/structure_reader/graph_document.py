"""统一语义图文档（SemanticGraph）与规范化 canonical JSON——小单01 01a 探针增量。

本文件当前实际实现（仅以下内容）：
- SemanticGraph 标准 frozen dataclass：显式 schema（semantic-graph/v1）、
  template_id、world_size 分析假设（首版固定 1，仅记录假设，不声称支持多 rank
  局部维度）与各实体类型的有序集合：named_axes、tensors、nodes、edges、
  state_versions、state_transitions、guards、guard_activations、alias_relations、
  fold_groups。端口（Port）依附于 Node 的 input_ports/output_ports，不设独立
  集合。
- graph_to_json(graph) -> str：规范化序列化——实体集合按稳定 ID 排序
  （GuardActivation 按 (guard_id, scenario_id)），仅排序实体集合，不改变单个
  张量的轴/布局顺序、节点端口顺序、证据顺序、折叠映射顺序等有语义的内部
  顺序；同一有效图无论集合录入顺序如何，产出逐字节相同的 canonical JSON。
- graph_from_json(text) -> SemanticGraph：反序列化——text 必须是字符串（非字符串
  受控拒绝）；校验顶层 JSON 对象、schema、template_id、world_size **显式存在**、
  全部十个必需集合字段为 JSON 数组（缺失集合全部点名，不静默补空）；实体经各自
  from_dict 重建（构造错误带集合名与索引前缀）；**反序列化阶段**即拒绝重复
  stable_id / 重复 (guard_id, scenario_id) 组合键（报集合、第二次索引与重复键），
  不等 graph_to_json 才发现；合法输入读回的各集合为规范化排序序。组合键为真实
  二元 tuple (guard_id, scenario_id)（排序与去重同键，不用拼串，避免内容含分隔符
  时碰撞）。不抛裸 KeyError/TypeError。
- SemanticGraph 构造器（容器入口）校验各集合元素实体类型（非本类型对象具名拒绝，
  定位集合与索引），不把任意对象误收为合法图实体；构造器保留默认空集合，仅供
  合成图方便，反序列化路径不使用该默认。
- 未实现，本文件不声称：图级连接语义核验（端点/轴/版本/alias/guard 引用
  一致性、方向匹配、无版本环等，属 01b 检查器）、真实 DeepSeek 结构提取、
  template/instance 实例绑定（当前仅 template_id 字段）、完整来源证据对象
  （各实体现以 origin/evidence_ref/evidence_refs 字符串引用承载，证据等级
  字段未建模）。文档可序列化不代表图正确；S1 不批准。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from structure_reader.graph_ir import (
    SCHEMA,
    AliasRelation,
    Edge,
    FoldBoundaryMapping,
    FoldGroup,
    Guard,
    GuardActivation,
    NamedAxis,
    Node,
    Port,
    StateTransition,
    StateVersion,
    Tensor,
)

_TEMPLATE_WORLD_SIZE = 1


@dataclass(frozen=True)
class SemanticGraph:
    """统一语义图文档：01a 已交付实体的有序集合容器。

    schema 固定 semantic-graph/v1；template_id 为非空字符串；world_size 记录
    分析假设（本单固定 1，不声称支持多 rank 局部维度）。各集合有序；规范化
    序列化时按稳定 ID 排序，图对象本身的集合顺序仅为录入顺序。
    Port 依附于 Node（input_ports/output_ports），不设独立集合。
    实例绑定（instance_binding）与完整来源证据对象尚未建模，本单不发明。
    """

    template_id: str
    named_axes: Sequence[NamedAxis] = ()
    tensors: Sequence[Tensor] = ()
    nodes: Sequence[Node] = ()
    edges: Sequence[Edge] = ()
    state_versions: Sequence[StateVersion] = ()
    state_transitions: Sequence[StateTransition] = ()
    guards: Sequence[Guard] = ()
    guard_activations: Sequence[GuardActivation] = ()
    alias_relations: Sequence[AliasRelation] = ()
    fold_groups: Sequence[FoldGroup] = ()
    schema: str = SCHEMA
    world_size: int = _TEMPLATE_WORLD_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not self.template_id.strip():
            raise ValueError("template_id 必须是非空字符串")
        if self.template_id != self.template_id.strip():
            raise ValueError("template_id 不允许首尾空白")
        if self.schema != SCHEMA:
            raise ValueError(f"schema 必须是 {SCHEMA}，得到 {self.schema!r}")
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size != _TEMPLATE_WORLD_SIZE
        ):
            raise ValueError(
                f"world_size 当前固定为 {_TEMPLATE_WORLD_SIZE}（首版分析假设，"
                f"不支持多 rank 局部维度），得到 {self.world_size!r}"
            )
        for name in ("named_axes", "tensors", "nodes", "edges", "state_versions",
                     "state_transitions", "guards", "guard_activations",
                     "alias_relations", "fold_groups"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{name} 必须是 list/tuple 序列")
            object.__setattr__(self, name, tuple(value))
        _check_entity_types(self)


_ENTITY_TYPES: dict[str, type] = {
    "named_axes": NamedAxis,
    "tensors": Tensor,
    "nodes": Node,
    "edges": Edge,
    "state_versions": StateVersion,
    "state_transitions": StateTransition,
    "guards": Guard,
    "guard_activations": GuardActivation,
    "alias_relations": AliasRelation,
    "fold_groups": FoldGroup,
}

_ENTITY_FIELDS: dict[str, tuple[str, Any]] = {
    name: (name, entity_type) for name, entity_type in _ENTITY_TYPES.items()
}


def _check_entity_types(graph: SemanticGraph) -> None:
    """容器入口实体类型闸门：各集合元素必须是声明类型，定位集合与索引。"""
    for collection, entity_type in _ENTITY_TYPES.items():
        for index, item in enumerate(getattr(graph, collection)):
            if not isinstance(item, entity_type):
                raise ValueError(
                    f"{collection}[{index}] 必须是 {entity_type.__name__}，"
                    f"得到 {type(item).__name__}"
                )


def _sorted_by_stable_id(items: Sequence[Any], collection: str) -> list[Any]:
    """按 stable_id 升序复制排序；重复 ID 受控拒绝（报第二次索引与重复键）。"""
    seen: dict[str, Any] = {}
    for index, item in enumerate(items):
        key = item.stable_id
        if key in seen:
            raise ValueError(
                f"{collection}[{index}] stable_id 重复: {key}"
                f"（与 {collection} 中已有条目冲突）"
            )
        seen[key] = item
    return [seen[key] for key in sorted(seen)]


def _sorted_activations(
    items: Sequence[GuardActivation], collection: str
) -> list[GuardActivation]:
    """按真实二元 tuple (guard_id, scenario_id) 排序去重——不用拼串，避免碰撞。"""
    seen: dict[tuple[str, str], GuardActivation] = {}
    for index, item in enumerate(items):
        key = (item.guard_id, item.scenario_id)
        if key in seen:
            raise ValueError(
                f"{collection}[{index}] (guard_id, scenario_id) 组合键重复: {key}"
            )
        seen[key] = item
    return [seen[key] for key in sorted(seen)]


def graph_to_json(graph: SemanticGraph) -> str:
    """规范化序列化：实体集合按稳定 ID 排序，语义内部顺序一律不重排。

    同一有效图在集合录入顺序不同的情况下产出逐字节相同的 JSON；
    重复 stable_id / 重复 (guard_id, scenario_id) 组合键受控拒绝。
    """
    if not isinstance(graph, SemanticGraph):
        raise ValueError("graph_to_json 需要 SemanticGraph 实例")
    envelope = {
        "schema": graph.schema,
        "template_id": graph.template_id,
        "world_size": graph.world_size,
        "named_axes": [axis.to_dict() for axis in _sorted_by_stable_id(graph.named_axes, "named_axes")],
        "tensors": [t.to_dict() for t in _sorted_by_stable_id(graph.tensors, "tensors")],
        "nodes": [n.to_dict() for n in _sorted_by_stable_id(graph.nodes, "nodes")],
        "edges": [e.to_dict() for e in _sorted_by_stable_id(graph.edges, "edges")],
        "state_versions": [sv.to_dict() for sv in _sorted_by_stable_id(graph.state_versions, "state_versions")],
        "state_transitions": [st.to_dict() for st in _sorted_by_stable_id(graph.state_transitions, "state_transitions")],
        "guards": [g.to_dict() for g in _sorted_by_stable_id(graph.guards, "guards")],
        "guard_activations": [
            ga.to_dict() for ga in _sorted_activations(graph.guard_activations, "guard_activations")
        ],
        "alias_relations": [
            ar.to_dict() for ar in _sorted_by_stable_id(graph.alias_relations, "alias_relations")
        ],
        "fold_groups": [fg.to_dict() for fg in _sorted_by_stable_id(graph.fold_groups, "fold_groups")],
    }
    return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_ENTITY_FIELDS: dict[str, tuple[str, Any]] = {
    "named_axes": ("named_axes", NamedAxis),
    "tensors": ("tensors", Tensor),
    "nodes": ("nodes", Node),
    "edges": ("edges", Edge),
    "state_versions": ("state_versions", StateVersion),
    "state_transitions": ("state_transitions", StateTransition),
    "guards": ("guards", Guard),
    "guard_activations": ("guard_activations", GuardActivation),
    "alias_relations": ("alias_relations", AliasRelation),
    "fold_groups": ("fold_groups", FoldGroup),
}


def graph_from_json(text: str) -> SemanticGraph:
    """反序列化：必需字段显式存在、实体重建、反序列化阶段拒绝重复键。

    text 必须是字符串；schema/template_id/world_size/全部十个必需集合字段
    缺失时全部点名拒绝（不静默补空）；实体构造错误定位集合[索引]；重复
    stable_id 与重复 (guard_id, scenario_id) 组合键在反序列化阶段即受控拒绝
    （报集合、第二次索引与重复键）。合法输入读回的各集合为规范化排序序。
    """
    if not isinstance(text, str):
        raise ValueError(
            f"图文档输入必须是 JSON 字符串，得到 {type(text).__name__}"
        )
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"图文档不是合法 JSON: {exc}") from exc
    if not isinstance(envelope, dict):
        raise ValueError("图文档顶层必须是 JSON 对象")
    if envelope.get("schema") != SCHEMA:
        raise ValueError(f"schema 不符，期望 {SCHEMA}，得到 {envelope.get('schema')!r}")
    template_id = envelope.get("template_id")
    if not isinstance(template_id, str) or not template_id.strip():
        raise ValueError("template_id 必须是非空字符串")
    if "world_size" not in envelope:
        raise ValueError("world_size 必须显式记录（首版分析假设），不允许缺省")
    world_size = envelope["world_size"]
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size != _TEMPLATE_WORLD_SIZE
    ):
        raise ValueError(
            f"world_size 当前固定为 {_TEMPLATE_WORLD_SIZE}（首版分析假设，"
            f"不支持多 rank 局部维度），得到 {world_size!r}"
        )
    missing_collections = [
        collection for collection in _ENTITY_TYPES if collection not in envelope
    ]
    if missing_collections:
        raise ValueError(
            f"图文档缺少必需集合字段: {', '.join(missing_collections)}"
        )
    kwargs: dict[str, Any] = {"template_id": template_id, "world_size": world_size}
    for collection, (arg_name, entity_type) in _ENTITY_FIELDS.items():
        raw_items = envelope[collection]
        if not isinstance(raw_items, list):
            raise ValueError(f"{collection} 必须是 JSON 数组，得到 {type(raw_items).__name__}")
        entities = []
        for index, item in enumerate(raw_items):
            try:
                entities.append(entity_type.from_dict(item))
            except ValueError as exc:
                raise ValueError(f"{collection}[{index}]: {exc}") from exc
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"{collection}[{index}]: {type(exc).__name__}: {exc}"
                ) from exc
        if collection == "guard_activations":
            entities = _sorted_activations(entities, collection)
        else:
            entities = _sorted_by_stable_id(entities, collection)
        kwargs[arg_name] = entities
    return SemanticGraph(**kwargs)
