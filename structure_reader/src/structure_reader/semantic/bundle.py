"""语义阶段图 bundle：复用既有 semantic-graph IR，加注释/来源/旧节点映射与校验。

- :class:`SemanticBundle` = ``graph_document.SemanticGraph``（具名轴/张量/节点/边/
  状态版本/状态转移/guard/alias）+ 节点注释（显示名、算子类型、模块路径、结构化
  来源、旧事实映射、条件）+ 张量注释（角色/显示名）+ 全称轴目录 + 元数据。
- :class:`GraphBuilder`：确定性 ID 的建图助手；端口/边的 tensor 三方一致由既有
  ``graph_ir``/``checks.graph_contracts`` 校验，本模块补齐注释完整性、全称轴与
  旧事实映射。
- ``validate_bundle``：模型无关校验接口（复用图级引用检查器，不重写一套）。
- ``bundle_to_json`` / ``bundle_from_json``：确定性序列化（图部分走既有
  canonical ``graph_to_json``）。

本模块不解析模型源码、不执行模型、不读权重；DeepSeek 专属事实由适配器声明。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from structure_reader.checks.graph_contracts import GraphIssue, validate_graph_references
from structure_reader.graph_document import SemanticGraph, graph_from_json, graph_to_json
from structure_reader.graph_ir import (
    AliasRelation,
    Edge,
    Guard,
    NamedAxis,
    Node,
    Port,
    StateTransition,
    StateVersion,
    Tensor,
)
from structure_reader.semantic.axes import AxisCatalog, AxisError

SCHEMA = "semantic-stage-graph/v1"


class BundleError(ValueError):
    """语义 bundle 构造/校验失败（具名拒绝）。"""


# ---------------------------------------------------------------------------
# 来源与注释
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRef:
    """一条结构化源码来源：仓库身份 + 仓库相对路径 + 行区间。"""

    repo_key: str
    path: str
    line: int
    end_line: int | None = None

    def __post_init__(self) -> None:
        if not self.repo_key.strip() or not self.path.strip():
            raise BundleError("SourceRef 的 repo_key/path 不能为空")
        if self.line <= 0 or (self.end_line is not None and self.end_line < self.line):
            raise BundleError(f"SourceRef 行区间非法: {self.path}:{self.line}-{self.end_line}")

    @property
    def location(self) -> str:
        if self.end_line is None or self.end_line == self.line:
            return f"{self.path}:{self.line}"
        return f"{self.path}:{self.line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_key": self.repo_key,
            "path": self.path,
            "line": self.line,
            "end_line": self.end_line,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceRef":
        return cls(
            repo_key=data["repo_key"],
            path=data["path"],
            line=int(data["line"]),
            end_line=None if data.get("end_line") is None else int(data["end_line"]),
        )


def sources_text(sources: Sequence[SourceRef]) -> tuple[str, ...]:
    return tuple(source.location for source in sources)


@dataclass(frozen=True)
class NodeMeta:
    """一个语义节点的注释：显示名、算子类型、模块身份、来源、旧节点映射。

    - label：主图显示名（如 ``kv_proj · Linear``）；display_name/op_label 分列实例名
      与操作类型（Linear 默认同时显示）。
    - kind：语义类别（external/linear/rmsnorm/view/transpose/reshape/rope/matmul/
      scale/add/concat/softmax/broadcast/cache_update/composite/state_in/state_out…）。
    - module_path：真实模块路径（如 ``model.layers.4.self_attn.kv_proj``）。
    - legacy_refs：旧事实映射（``op:kv_proj``、``tensor:kv_proj_out``）。
    - sources：结构化源码来源；condition 为条件路径说明（如 ``layer_type !=
      sliding_attention``）。
    - attributes：规则需要的附加语义（如 ``rhs_transposed_axes``）。
    """

    node_id: str
    label: str
    kind: str
    module_path: str
    sources: tuple[SourceRef, ...]
    legacy_refs: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    note: str = ""
    condition: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("node_id", "label", "kind", "module_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise BundleError(f"NodeMeta.{name} 必须是非空字符串")
        if not self.sources:
            raise BundleError(f"节点 {self.node_id} 必须至少一条来源")
        for source in self.sources:
            if not isinstance(source, SourceRef):
                raise BundleError(f"节点 {self.node_id} 的来源必须是 SourceRef")
        for ref in self.legacy_refs:
            if not isinstance(ref, str) or not ref.strip():
                raise BundleError(f"节点 {self.node_id} 的旧事实映射不能为空")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "legacy_refs", tuple(self.legacy_refs))
        object.__setattr__(self, "params", tuple(self.params))
        object.__setattr__(self, "attributes", dict(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "kind": self.kind,
            "module_path": self.module_path,
            "sources": [source.to_dict() for source in self.sources],
            "legacy_refs": list(self.legacy_refs),
            "params": list(self.params),
            "note": self.note,
            "condition": self.condition,
            "attributes": dict(sorted(self.attributes.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodeMeta":
        return cls(
            node_id=data["node_id"],
            label=data["label"],
            kind=data["kind"],
            module_path=data["module_path"],
            sources=tuple(SourceRef.from_dict(item) for item in data["sources"]),
            legacy_refs=tuple(data.get("legacy_refs", ())),
            params=tuple(data.get("params", ())),
            note=data.get("note", ""),
            condition=data.get("condition"),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class TensorMeta:
    """一个语义张量的注释：角色（activation/mask/index/cache）与显示名/说明。"""

    tensor_id: str
    role: str
    label: str
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("tensor_id", "role", "label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise BundleError(f"TensorMeta.{name} 必须是非空字符串")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "role": self.role,
            "label": self.label,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorMeta":
        return cls(
            tensor_id=data["tensor_id"],
            role=data["role"],
            label=data["label"],
            note=data.get("note", ""),
        )


# ---------------------------------------------------------------------------
# 建图助手
# ---------------------------------------------------------------------------


class GraphBuilder:
    """确定性语义图构建器：张量/节点/边/状态/alias 与注释一起登记。"""

    def __init__(self, template_id: str, axes: AxisCatalog, world_size: int = 1) -> None:
        if not template_id.strip():
            raise BundleError("template_id 不能为空")
        self.template_id = template_id
        self.axes = axes
        self.world_size = world_size
        self._tensors: dict[str, Tensor] = {}
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._state_versions: dict[str, StateVersion] = {}
        self._state_transitions: dict[str, StateTransition] = {}
        self._guards: dict[str, Guard] = {}
        self._aliases: dict[str, AliasRelation] = {}
        self._node_meta: dict[str, NodeMeta] = {}
        self._tensor_meta: dict[str, TensorMeta] = {}
        self._output_ports: dict[str, dict[str, str]] = {}
        self._input_ports: dict[str, dict[str, str]] = {}
        self._tensor_ids: dict[str, str] = {}

    def _resolve_tensor(self, value: str) -> str:
        """作者可用短名（``kv``）或稳定 ID（``tensor:kv``），统一成稳定 ID。"""
        if value.startswith("tensor:"):
            return value
        if value not in self._tensor_ids:
            raise BundleError(f"张量未登记: {value!r}")
        return self._tensor_ids[value]

    # -- 张量 ---------------------------------------------------------------
    def tensor(
        self,
        tensor_id: str,
        symbols: Sequence[str],
        *,
        dtype: str = "bfloat16",
        role: str = "activation",
        label: str,
        note: str = "",
        physical_layout: Sequence[str] | None = None,
    ) -> str:
        if tensor_id in self._tensor_ids:
            raise BundleError(f"张量重复登记: {tensor_id}")
        axes = [self.axes.named_axis(symbol) for symbol in symbols]
        layout = tuple(symbols)
        physical = tuple(physical_layout) if physical_layout is not None else layout
        tensor = Tensor(
            name=tensor_id,
            dtype=dtype,
            axes=axes,
            logical_layout=layout,
            physical_layout=physical,
        )
        self._tensor_ids[tensor_id] = tensor.stable_id
        self._tensors[tensor.stable_id] = tensor
        self._tensor_meta[tensor.stable_id] = TensorMeta(
            tensor_id=tensor.stable_id, role=role, label=label, note=note
        )
        return tensor.stable_id

    # -- 节点 ---------------------------------------------------------------
    def node(
        self,
        node_id: str,
        *,
        op_type: str,
        granularity: str,
        label: str,
        kind: str,
        module_path: str,
        sources: Sequence[SourceRef],
        inputs: Sequence[tuple[str, str]] = (),
        outputs: Sequence[tuple[str, str]] = (),
        legacy_refs: Sequence[str] = (),
        params: Sequence[str] = (),
        note: str = "",
        condition: str | None = None,
        attributes: Mapping[str, str] | None = None,
    ) -> str:
        if node_id in self._nodes:
            raise BundleError(f"节点重复登记: {node_id}")
        resolved_inputs = [
            (name, self._resolve_tensor(tid)) for name, tid in inputs
        ]
        resolved_outputs = [
            (name, self._resolve_tensor(tid)) for name, tid in outputs
        ]
        input_ports = [
            Port(name=name, direction="input", tensor_id=tid) for name, tid in resolved_inputs
        ]
        output_ports = [
            Port(name=name, direction="output", tensor_id=tid)
            for name, tid in resolved_outputs
        ]
        for name, tid in resolved_inputs:
            if tid not in self._tensors:
                raise BundleError(f"节点 {node_id} 的输入张量未登记: {tid}")
        for name, tid in resolved_outputs:
            if tid not in self._tensors:
                raise BundleError(f"节点 {node_id} 的输出张量未登记: {tid}")
        self._nodes[node_id] = Node(
            stable_id=node_id,
            op_type=op_type,
            granularity=granularity,
            input_ports=input_ports,
            output_ports=output_ports,
            evidence_refs=sources_text(sources),
        )
        self._node_meta[node_id] = NodeMeta(
            node_id=node_id,
            label=label,
            kind=kind,
            module_path=module_path,
            sources=tuple(sources),
            legacy_refs=tuple(legacy_refs),
            params=tuple(params),
            note=note,
            condition=condition,
            attributes=dict(attributes or {}),
        )
        self._input_ports[node_id] = {name: tid for name, tid in resolved_inputs}
        self._output_ports[node_id] = {name: tid for name, tid in resolved_outputs}
        return node_id

    # -- 边 -----------------------------------------------------------------
    def edge(
        self,
        edge_id: str,
        src_node: str,
        src_port: str,
        dst_node: str,
        dst_port: str,
        *,
        kind: str = "data",
        guard_ref: str | None = None,
        transition_ref: str | None = None,
        sources: Sequence[SourceRef] = (),
    ) -> str:
        if edge_id in self._edges:
            raise BundleError(f"边重复登记: {edge_id}")
        try:
            tensor_id = self._output_ports[src_node][src_port]
        except KeyError as error:
            raise BundleError(
                f"边 {edge_id}: 节点 {src_node} 没有输出端口 {src_port!r}"
            ) from error
        try:
            consumed = self._input_ports[dst_node][dst_port]
        except KeyError as error:
            raise BundleError(
                f"边 {edge_id}: 节点 {dst_node} 没有输入端口 {dst_port!r}"
            ) from error
        if consumed != tensor_id:
            raise BundleError(
                f"边 {edge_id}: 端口张量不一致（{src_node}.{src_port}={tensor_id} vs "
                f"{dst_node}.{dst_port}={consumed}）"
            )
        self._edges[edge_id] = Edge(
            stable_id=edge_id,
            src_node_id=src_node,
            src_port=src_port,
            dst_node_id=dst_node,
            dst_port=dst_port,
            tensor_id=tensor_id,
            kind=kind,
            evidence_refs=sources_text(sources),
            guard_ref=guard_ref,
            state_transition_ref=transition_ref,
        )
        return edge_id

    # -- 状态 / guard / alias ----------------------------------------------
    def guard(self, guard_id: str, predicate_expr: str, sources: Sequence[SourceRef]) -> str:
        self._guards[guard_id] = Guard(
            stable_id=guard_id,
            predicate_expr=predicate_expr,
            evidence_refs=sources_text(sources),
        )
        return guard_id

    def state_transition(
        self, transition_id: str, state_id: str, before: str, after: str
    ) -> str:
        before_id = f"sv.{before}"
        after_id = f"sv.{after}"
        self._state_versions[before_id] = StateVersion(
            stable_id=before_id, state_id=state_id, version_label="before"
        )
        self._state_versions[after_id] = StateVersion(
            stable_id=after_id, state_id=state_id, version_label="after"
        )
        self._state_transitions[transition_id] = StateTransition(
            stable_id=transition_id,
            state_id=state_id,
            before_version_id=before_id,
            after_version_id=after_id,
        )
        return transition_id

    def alias(self, alias_id: str, tensor_id: str, role: str) -> str:
        tensor_id = self._resolve_tensor(tensor_id)
        if tensor_id not in self._tensors:
            raise BundleError(f"alias {alias_id}: 张量未登记 {tensor_id}")
        self._aliases[alias_id] = AliasRelation(
            stable_id=alias_id,
            base_tensor_id=tensor_id,
            alias_tensor_id=tensor_id,
            relation_kind="same_tensor",
            alias_role=role,
        )
        return alias_id

    # -- 构建 ---------------------------------------------------------------
    def build(self, metadata: Mapping[str, Any]) -> "SemanticBundle":
        graph = SemanticGraph(
            template_id=self.template_id,
            named_axes=tuple(
                self.axes.named_axis(symbol) for symbol in self.axes.symbols()
            ),
            tensors=tuple(self._tensors.values()),
            nodes=tuple(self._nodes.values()),
            edges=tuple(self._edges.values()),
            state_versions=tuple(self._state_versions.values()),
            state_transitions=tuple(self._state_transitions.values()),
            guards=tuple(self._guards.values()),
            alias_relations=tuple(self._aliases.values()),
            world_size=self.world_size,
        )
        return SemanticBundle(
            graph=graph,
            node_meta=dict(self._node_meta),
            tensor_meta=dict(self._tensor_meta),
            axes=self.axes,
            metadata=dict(metadata),
        )


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticBundle:
    """语义阶段图 + 注释 + 轴目录 + 元数据（模型无关）。"""

    graph: SemanticGraph
    node_meta: Mapping[str, NodeMeta]
    tensor_meta: Mapping[str, TensorMeta]
    axes: AxisCatalog
    metadata: Mapping[str, Any]

    # -- 查询 ---------------------------------------------------------------
    @property
    def nodes(self) -> dict[str, Node]:
        return {node.stable_id: node for node in self.graph.nodes}

    @property
    def tensors(self) -> dict[str, Tensor]:
        return {tensor.stable_id: tensor for tensor in self.graph.tensors}

    @property
    def edges(self) -> dict[str, Edge]:
        return {edge.stable_id: edge for edge in self.graph.edges}

    def shape_text(self, tensor_id: str) -> str:
        parts: list[str] = []
        for axis in self.tensors[tensor_id].axes:
            definition = self.axes.definition(axis.symbol)
            if definition.display_override is not None:
                parts.append(definition.display_override)
            elif axis.value is not None and definition.display_value:
                parts.append(str(axis.value))
            elif axis.expression is not None:
                parts.append(axis.expression)
            else:
                parts.append(axis.name)
        return "[" + ", ".join(parts) + "]"

    def axis_symbols(self, tensor_id: str) -> tuple[str, ...]:
        return tuple(axis.symbol for axis in self.tensors[tensor_id].axes)

    def node_label(self, node_id: str) -> str:
        return self.node_meta[node_id].label

    def replace(self, *, graph: SemanticGraph, node_meta=None, tensor_meta=None) -> "SemanticBundle":
        return SemanticBundle(
            graph=graph,
            node_meta=dict(node_meta if node_meta is not None else self.node_meta),
            tensor_meta=dict(tensor_meta if tensor_meta is not None else self.tensor_meta),
            axes=self.axes,
            metadata=dict(self.metadata),
        )

    # -- 序列化 -------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "metadata": dict(self.metadata),
            "axes": self.axes.to_json(),
            "graph": json.loads(graph_to_json(self.graph)),
            "node_meta": {
                node_id: meta.to_dict() for node_id, meta in sorted(self.node_meta.items())
            },
            "tensor_meta": {
                tensor_id: meta.to_dict()
                for tensor_id, meta in sorted(self.tensor_meta.items())
            },
        }

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> "SemanticBundle":
        if document.get("schema") != SCHEMA:
            raise BundleError(f"schema 不符，期望 {SCHEMA}")
        graph = graph_from_json(
            json.dumps(document["graph"], ensure_ascii=False, sort_keys=True)
        )
        return cls(
            graph=graph,
            node_meta={
                key: NodeMeta.from_dict(value)
                for key, value in document["node_meta"].items()
            },
            tensor_meta={
                key: TensorMeta.from_dict(value)
                for key, value in document["tensor_meta"].items()
            },
            axes=AxisCatalog.from_json(document["axes"]),
            metadata=dict(document["metadata"]),
        )


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def _issue(issues: list[BundleIssue], code: str, path: str, message: str) -> None:
    issues.append(BundleIssue(code=code, path=path, message=message))


@dataclass(frozen=True)
class BundleIssue:
    """一条 bundle 校验发现：稳定代码、精确路径、易懂信息。"""

    code: str
    path: str
    message: str


def validate_bundle(bundle: SemanticBundle) -> tuple[BundleIssue, ...]:
    """模型无关校验：复用图级引用检查器，补注释/全称轴/旧事实映射/状态覆盖。"""
    issues: list[BundleIssue] = [
        BundleIssue(code=f"graph:{issue.code}", path=issue.path, message=issue.message)
        for issue in validate_graph_references(bundle.graph)
    ]

    nodes = bundle.nodes
    tensors = bundle.tensors
    for node_id in nodes:
        meta = bundle.node_meta.get(node_id)
        if meta is None:
            _issue(issues, "node-meta-missing", f"nodes[{node_id}]", "节点缺少注释")
            continue
        if not meta.label.strip():
            _issue(issues, "node-label-empty", f"nodes[{node_id}]", "节点显示名为空")
        if not meta.sources:
            _issue(issues, "node-source-missing", f"nodes[{node_id}]", "节点缺少来源")
        if not meta.legacy_refs:
            _issue(
                issues,
                "node-legacy-missing",
                f"nodes[{node_id}]",
                "节点缺少旧事实映射（无法定位原始算子/张量）",
            )
    for node_id in bundle.node_meta:
        if node_id not in nodes:
            _issue(issues, "node-meta-orphan", f"node_meta[{node_id}]", "注释指向不存在的节点")

    for tensor_id in tensors:
        meta = bundle.tensor_meta.get(tensor_id)
        if meta is None:
            _issue(issues, "tensor-meta-missing", f"tensors[{tensor_id}]", "张量缺少注释")
        for axis in tensors[tensor_id].axes:
            try:
                bundle.axes.definition(axis.symbol)
            except AxisError as error:
                _issue(
                    issues,
                    "axis-not-full-name",
                    f"tensors[{tensor_id}]",
                    str(error),
                )
    for tensor_id in bundle.tensor_meta:
        if tensor_id not in tensors:
            _issue(
                issues,
                "tensor-meta-orphan",
                f"tensor_meta[{tensor_id}]",
                "注释指向不存在的张量",
            )

    # 状态语义：每条 StateTransition 至少被一条 state 边引用；state 边引用的转移必须存在
    referenced = {
        edge.state_transition_ref
        for edge in bundle.graph.edges
        if edge.state_transition_ref is not None
    }
    for transition in bundle.graph.state_transitions:
        if transition.stable_id not in referenced:
            _issue(
                issues,
                "state-transition-unused",
                f"state_transitions[{transition.stable_id}]",
                "状态转移没有被任何 state 边引用（缓存副作用被隐藏）",
            )
    for edge in bundle.graph.edges:
        if edge.kind == "state" and edge.state_transition_ref is None:
            _issue(
                issues,
                "state-edge-without-transition",
                f"edges[{edge.stable_id}]",
                "state 边缺少状态转移引用",
            )

    for key in ("model_key", "layer_key", "layer_group", "template_id", "source_revision"):
        if not str(bundle.metadata.get(key, "")).strip():
            _issue(issues, "metadata-missing", f"metadata[{key}]", f"元数据缺少 {key}")
    if bundle.metadata.get("template_id") != bundle.graph.template_id:
        _issue(
            issues,
            "template-id-mismatch",
            "metadata[template_id]",
            "元数据 template_id 与图身份不一致",
        )
    return tuple(issues)


def graph_issue_to_bundle_issue(issue: GraphIssue) -> BundleIssue:
    return BundleIssue(code=f"graph:{issue.code}", path=issue.path, message=issue.message)
