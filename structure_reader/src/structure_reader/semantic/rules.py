"""模型无关的可追溯等价化简：保守白名单规则、前提、应用/拒绝审计与保持性校验。

规则集（T12 白名单，逐条对应 12.md / 调整结论 §5）：

- R1 ``shape-expression-simplify``：形状表达式白名单化简（常量折叠、``x+0``/
  ``x*1``/``x*0``/``x-x``/``x//1``/``min(x,x)``，以及带 ``divisible`` 前提的
  ``a*(b//a)==b``）。保留动态变量，不把未知规模变成示例常量。
- R2 ``identity-eliminate``：目标轴线顺序与物理布局完全相同的 View/Reshape/
  Transpose（或恒等广播）消除；共享输出消费者全部改接输入，不删其它分支。
- R3 ``transpose-pair-cancel``：相邻两个 Transpose 复合为恒等时抵消；要求中间
  张量只有一个消费者。
- R4 ``reshape-merge``：相邻 Reshape/View 合并为一次等价 Reshape；中间结果不得
  有其它使用者，不能跨越真实重排/切分。
- R5 ``unit-axis-reorder``：View + Transpose 只在单位轴之间重排时化为一次 4D
  Reshape（保留三维→四维变化）；非单位轴（如 Q 的 64 头）拒绝。
- R6 ``split-concat-cancel``：完整 Split 后同轴同序原样拼回时抵消；分支被消费或
  拼接来源不同则保留。
- R7 ``transpose-into-matmul``：只交换最后两个矩阵轴的 Transpose 并入矩阵乘语义
  （``Q @ Kᵀ``），保留轴置换信息与两个算子的来源，不声称设备融合。

每条规则都记录成立条件、应用明细（改接前后的节点/张量/来源）或拒绝原因；
``validate_normalization`` 复用既有图级检查器并补：映射覆盖、来源保留、状态
转移保留、anchor 节点间数据可达关系不变。本模块不执行模型、不折叠权重数值、
不改 dtype。
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from structure_reader.graph_ir import (
    AliasRelation,
    Edge,
    Guard,
    Node,
    Port,
    StateTransition,
    StateVersion,
    Tensor,
)
from structure_reader.semantic.bundle import (
    BundleIssue,
    NodeMeta,
    SemanticBundle,
    SourceRef,
    TensorMeta,
    validate_bundle,
)
from structure_reader.semantic import layout_probe, shape_expr
from structure_reader.semantic.axes import AxisCatalog
from structure_reader.graph_ir import NamedAxis

RULES: tuple[str, ...] = (
    "R1-shape-expression-simplify",
    "R2-identity-eliminate",
    "R3-transpose-pair-cancel",
    "R4-reshape-merge",
    "R5-unit-axis-reorder",
    "R6-split-concat-cancel",
    "R7-transpose-into-matmul",
)

_LAYOUT_KINDS = frozenset({"view", "reshape", "transpose"})
_PROBE_REQUIRED = frozenset(
    {
        "R2-identity-eliminate",
        "R3-transpose-pair-cancel",
        "R4-reshape-merge",
        "R5-unit-axis-reorder",
        "R7-transpose-into-matmul",
    }
)


@dataclass(frozen=True)
class RuleApplication:
    """一条规则的一次实际应用（可追溯到具体节点/张量/来源）。"""

    rule_id: str
    target_nodes: tuple[str, ...]
    removed_nodes: tuple[str, ...]
    added_nodes: tuple[str, ...]
    removed_tensors: tuple[str, ...]
    added_tensors: tuple[str, ...]
    rewired: tuple[tuple[str, str], ...]
    condition: str
    reason: str
    sources: tuple[str, ...]
    probe: str = ""
    guards: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "target_nodes": list(self.target_nodes),
            "removed_nodes": list(self.removed_nodes),
            "added_nodes": list(self.added_nodes),
            "removed_tensors": list(self.removed_tensors),
            "added_tensors": list(self.added_tensors),
            "rewired": [list(item) for item in self.rewired],
            "condition": self.condition,
            "reason": self.reason,
            "sources": list(self.sources),
            "probe": self.probe,
            "guards": list(self.guards),
        }


@dataclass(frozen=True)
class RuleReport:
    """一条规则的结论：应用明细或拒绝原因（每条规则都要有其一）。"""

    rule_id: str
    title: str
    precondition: str
    status: str
    reason: str
    applications: tuple[RuleApplication, ...] = ()
    rejected_candidates: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "precondition": self.precondition,
            "status": self.status,
            "reason": self.reason,
            "applications": [item.to_dict() for item in self.applications],
            "rejected_candidates": [dict(item) for item in self.rejected_candidates],
        }


@dataclass(frozen=True)
class NormalizationResult:
    """原始图 → 化简图的结果：新图 + 规则审计 + 双向节点/张量映射。"""

    raw: SemanticBundle
    normalized: SemanticBundle
    reports: tuple[RuleReport, ...]
    node_mapping: Mapping[str, tuple[str, ...]]
    tensor_mapping: Mapping[str, str]
    removed_nodes: Mapping[str, dict[str, str]]
    removed_tensors: Mapping[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            "rules": [report.to_dict() for report in self.reports],
            "node_mapping": {key: list(value) for key, value in sorted(self.node_mapping.items())},
            "tensor_mapping": dict(sorted(self.tensor_mapping.items())),
            "removed_nodes": dict(sorted(self.removed_nodes.items())),
            "removed_tensors": dict(sorted(self.removed_tensors.items())),
            "counts": {
                "raw_nodes": len(self.raw.graph.nodes),
                "normalized_nodes": len(self.normalized.graph.nodes),
                "raw_tensors": len(self.raw.graph.tensors),
                "normalized_tensors": len(self.normalized.graph.tensors),
            },
        }


# ---------------------------------------------------------------------------
# 可变异步图
# ---------------------------------------------------------------------------


class _Mutable:
    def __init__(self, bundle: SemanticBundle) -> None:
        self.bundle = bundle
        self.axes: AxisCatalog = bundle.axes
        self.template_id = bundle.graph.template_id
        self.world_size = bundle.graph.world_size
        self.nodes: dict[str, Node] = {node.stable_id: node for node in bundle.graph.nodes}
        self.edges: dict[str, Edge] = {edge.stable_id: edge for edge in bundle.graph.edges}
        self.tensors: dict[str, Tensor] = {
            tensor.stable_id: tensor for tensor in bundle.graph.tensors
        }
        self.state_versions: dict[str, StateVersion] = {
            item.stable_id: item for item in bundle.graph.state_versions
        }
        self.state_transitions: dict[str, StateTransition] = {
            item.stable_id: item for item in bundle.graph.state_transitions
        }
        self.guards: dict[str, Guard] = {
            item.stable_id: item for item in bundle.graph.guards
        }
        self.aliases: dict[str, AliasRelation] = {
            item.stable_id: item for item in bundle.graph.alias_relations
        }
        self.node_meta: dict[str, NodeMeta] = dict(bundle.node_meta)
        self.tensor_meta: dict[str, TensorMeta] = dict(bundle.tensor_meta)
        self.metadata = dict(bundle.metadata)

    # -- 事务快照 -----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "nodes": dict(self.nodes),
            "edges": dict(self.edges),
            "tensors": dict(self.tensors),
            "state_versions": dict(self.state_versions),
            "state_transitions": dict(self.state_transitions),
            "guards": dict(self.guards),
            "aliases": dict(self.aliases),
            "node_meta": dict(self.node_meta),
            "tensor_meta": dict(self.tensor_meta),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.nodes = dict(snapshot["nodes"])
        self.edges = dict(snapshot["edges"])
        self.tensors = dict(snapshot["tensors"])
        self.state_versions = dict(snapshot["state_versions"])
        self.state_transitions = dict(snapshot["state_transitions"])
        self.guards = dict(snapshot["guards"])
        self.aliases = dict(snapshot["aliases"])
        self.node_meta = dict(snapshot["node_meta"])
        self.tensor_meta = dict(snapshot["tensor_meta"])

    # -- 查询 ---------------------------------------------------------------
    def edges_in(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges.values() if edge.dst_node_id == node_id]

    def edges_out(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges.values() if edge.src_node_id == node_id]

    def tensor_consumers(self, tensor_id: str) -> list[Edge]:
        return [edge for edge in self.edges.values() if edge.tensor_id == tensor_id]

    def input_tensor(self, node_id: str, port: str) -> str | None:
        for item in self.nodes[node_id].input_ports:
            if item.name == port:
                return item.tensor_id
        return None

    def output_tensor(self, node_id: str, port: str) -> str | None:
        for item in self.nodes[node_id].output_ports:
            if item.name == port:
                return item.tensor_id
        return None

    def data_inputs(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges_in(node_id) if edge.kind == "data"]

    def data_outputs(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges_out(node_id) if edge.kind == "data"]

    def single_data_input(self, node_id: str) -> Edge | None:
        edges = self.data_inputs(node_id)
        return edges[0] if len(edges) == 1 else None

    def single_output_edge(self, node_id: str) -> Edge | None:
        edges = self.edges_out(node_id)
        return edges[0] if len(edges) == 1 else None

    def single_output_tensor(self, node_id: str) -> str | None:
        ports = self.nodes[node_id].output_ports
        return ports[0].tensor_id if len(ports) == 1 else None

    def output_tensor_of(self, node_id: str, port: str) -> str | None:
        for item in self.nodes[node_id].output_ports:
            if item.name == port:
                return item.tensor_id
        return None

    # -- 修改 ---------------------------------------------------------------
    def set_input_tensor(self, node_id: str, port_name: str, tensor_id: str) -> None:
        node = self.nodes[node_id]
        ports = [
            Port(name=item.name, direction=item.direction, tensor_id=tensor_id)
            if item.name == port_name
            else item
            for item in node.input_ports
        ]
        self.nodes[node_id] = replace(node, input_ports=tuple(ports))

    def rewire_tensor(self, old_tensor: str, new_tensor: str) -> list[tuple[str, str]]:
        """把所有消费 old_tensor 的输入端口改接 new_tensor；返回改接明细。"""
        rewired: list[tuple[str, str]] = []
        for edge in list(self.edges.values()):
            if edge.tensor_id != old_tensor:
                continue
            self.set_input_tensor(edge.dst_node_id, edge.dst_port, new_tensor)
            self.edges[edge.stable_id] = replace(edge, tensor_id=new_tensor)
            rewired.append((edge.dst_node_id, edge.dst_port))
        return rewired

    def drop_node(self, node_id: str) -> None:
        self.nodes.pop(node_id)
        self.node_meta.pop(node_id, None)
        for edge_id, edge in list(self.edges.items()):
            if edge.src_node_id == node_id or edge.dst_node_id == node_id:
                del self.edges[edge_id]

    def drop_tensor(self, tensor_id: str) -> None:
        self.tensors.pop(tensor_id)
        self.tensor_meta.pop(tensor_id, None)

    def add_node(
        self,
        node_id: str,
        *,
        op_type: str,
        granularity: str,
        label: str,
        kind: str,
        module_path: str,
        sources: Sequence[SourceRef],
        inputs: Sequence[tuple[str, str]],
        outputs: Sequence[tuple[str, str]],
        legacy_refs: Sequence[str],
        params: Sequence[str] = (),
        note: str = "",
        condition: str | None = None,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        input_ports = [Port(name=name, direction="input", tensor_id=tid) for name, tid in inputs]
        output_ports = [
            Port(name=name, direction="output", tensor_id=tid) for name, tid in outputs
        ]
        self.nodes[node_id] = Node(
            stable_id=node_id,
            op_type=op_type,
            granularity=granularity,
            input_ports=tuple(input_ports),
            output_ports=tuple(output_ports),
            evidence_refs=tuple(source.location for source in sources),
        )
        self.node_meta[node_id] = NodeMeta(
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

    def add_edge(self, edge_id: str, src_node: str, src_port: str, dst_node: str, dst_port: str) -> None:
        tensor_id = self.output_tensor(src_node, src_port)
        if tensor_id is None:
            raise ValueError(f"{edge_id}: {src_node}.{src_port} 无输出张量")
        if self.input_tensor(dst_node, dst_port) != tensor_id:
            raise ValueError(f"{edge_id}: 端口张量不一致，拒绝连接")
        self.edges[edge_id] = Edge(
            stable_id=edge_id,
            src_node_id=src_node,
            src_port=src_port,
            dst_node_id=dst_node,
            dst_port=dst_port,
            tensor_id=tensor_id,
            kind="data",
        )

    def build(self) -> SemanticBundle:
        # named_axes 由全部张量轴重建，保证化简后的轴定义与张量一致。
        axes_by_symbol: dict[str, NamedAxis] = {}
        for tensor_id in sorted(self.tensors):
            for axis in self.tensors[tensor_id].axes:
                previous = axes_by_symbol.get(axis.symbol)
                if previous is not None and previous.to_dict() != axis.to_dict():
                    raise ValueError(f"轴 {axis.symbol} 在化简后定义不一致")
                axes_by_symbol[axis.symbol] = axis
        from structure_reader.graph_document import SemanticGraph

        graph = SemanticGraph(
            template_id=self.template_id,
            named_axes=tuple(axes_by_symbol[symbol] for symbol in sorted(axes_by_symbol)),
            tensors=tuple(self.tensors[tensor_id] for tensor_id in sorted(self.tensors)),
            nodes=tuple(self.nodes[node_id] for node_id in sorted(self.nodes)),
            edges=tuple(self.edges[edge_id] for edge_id in sorted(self.edges)),
            state_versions=tuple(
                self.state_versions[key] for key in sorted(self.state_versions)
            ),
            state_transitions=tuple(
                self.state_transitions[key] for key in sorted(self.state_transitions)
            ),
            guards=tuple(self.guards[key] for key in sorted(self.guards)),
            alias_relations=tuple(self.aliases[key] for key in sorted(self.aliases)),
            world_size=self.world_size,
        )
        return SemanticBundle(
            graph=graph,
            node_meta=dict(self.node_meta),
            tensor_meta=dict(self.tensor_meta),
            axes=self.axes,
            metadata=dict(self.metadata),
        )


# ---------------------------------------------------------------------------
# 规则实现
# ---------------------------------------------------------------------------

_PRECONDITIONS = {
    "R1-shape-expression-simplify": (
        "表达式只含白名单整数代数；divisible 前提必须由调用方显式给出；未知规模保持符号"
    ),
    "R2-identity-eliminate": (
        "节点输出轴线顺序与物理布局与输入完全相同（恒等映射），且该节点无状态副作用"
    ),
    "R3-transpose-pair-cancel": (
        "两个 Transpose 相邻、复合为恒等置换，且中间张量没有其它消费者"
    ),
    "R4-reshape-merge": (
        "两个相邻 Reshape/View 之间无任何计算，中间张量没有其它消费者，不跨越真实重排/切分"
    ),
    "R5-unit-axis-reorder": (
        "View 引入单位轴后只由 Transpose 在单位轴之间重排（非单位轴顺序不变），可化为一次 4D Reshape"
    ),
    "R6-split-concat-cancel": (
        "Split 完整切分、Concat 同轴同序原样拼回、所有分支只进入该 Concat、无其它消费者"
    ),
    "R7-transpose-into-matmul": (
        "Transpose 只交换最后两个矩阵轴、其输出只被该 MatMul 消费；保留轴置换信息与来源"
    ),
}


def _axis_order(tensor: Tensor) -> tuple[str, ...]:
    return tuple(axis.symbol for axis in tensor.axes)


def _is_unit(axis: NamedAxis) -> bool:
    return axis.value == 1


def _non_unit_order(tensor: Tensor) -> tuple[str, ...]:
    return tuple(axis.symbol for axis in tensor.axes if not _is_unit(axis))


def _layout_identical(left: Tensor, right: Tensor) -> bool:
    return (
        _axis_order(left) == _axis_order(right)
        and tuple(left.physical_layout) == tuple(right.physical_layout)
        and left.dtype == right.dtype
    )


class _RewriteRejected(ValueError):
    """改写无法保持执行条件或语义（规则必须保守拒绝）。"""


def _axes_map(tensors: Iterable[Tensor]) -> dict[str, NamedAxis]:
    result: dict[str, NamedAxis] = {}
    for tensor in tensors:
        for axis in tensor.axes:
            result.setdefault(axis.symbol, axis)
    return result


def _probe_extents(mutable: _Mutable, *orders: Sequence[str]) -> dict[str, int]:
    symbols: list[str] = []
    for order in orders:
        symbols.extend(order)
    return layout_probe.extents_for(
        tuple(dict.fromkeys(symbols)), _axes_map(mutable.tensors.values())
    )


def _last_two_swap(left: Tensor, right: Tensor) -> bool:
    left_symbols = _axis_order(left)
    right_symbols = _axis_order(right)
    if len(left_symbols) < 2 or len(right_symbols) != len(left_symbols):
        return False
    expected = left_symbols[:-2] + (left_symbols[-1], left_symbols[-2])
    return right_symbols == expected


def _rewrite(
    mutable: _Mutable,
    *,
    removed_nodes: Sequence[str],
    removed_tensors: Sequence[str],
    new_node: dict[str, Any] | None = None,
    forward: Sequence[tuple[str, str, str]] = (),
    connect: Sequence[tuple[str, str, str, str]] = (),
    probe: str = "",
) -> RuleApplication:
    """通用改写：先完整预检，再事务提交；任何拒绝/异常都保持原图不变。

    ``forward`` = ``(old_tensor, producer_node, producer_port)``。被改写链的输入边
    若带 guard，则没有自身 guard 的消费边继承该条件（条件转移）；消费边 guard 与
    链输入 guard 不同（一条边无法同时表达两个条件）时保守拒绝。预检在删除任何边
    之前完成；提交阶段用快照保护，异常时回滚，拒绝不会留下部分修改。
    """
    incoming_guards = {
        edge.guard_ref
        for edge in mutable.edges.values()
        if edge.dst_node_id in removed_nodes and edge.kind == "data" and edge.guard_ref
    }
    if len(incoming_guards) > 1:
        raise _RewriteRejected(
            f"改写链有多个不同执行条件 {sorted(incoming_guards)}，无法在一条新边上保持"
        )
    connect_guard = next(iter(incoming_guards), None)

    # 预检：只读地收集全部待改接消费者，任何冲突在修改前抛出。
    pending: list[tuple[str, str, str, str, str | None, str]] = []
    for old_tensor, producer_node, producer_port in forward:
        for edge in mutable.edges.values():
            if edge.tensor_id != old_tensor:
                continue
            if edge.guard_ref and connect_guard and edge.guard_ref != connect_guard:
                raise _RewriteRejected(
                    f"消费边条件 {edge.guard_ref} 与链输入条件 {connect_guard} 不同，"
                    "一条新边无法同时保持两个条件"
                )
            pending.append(
                (
                    producer_node,
                    producer_port,
                    edge.dst_node_id,
                    edge.dst_port,
                    edge.guard_ref or connect_guard,
                    edge.stable_id,
                )
            )
    for node_id in removed_nodes:
        if node_id not in mutable.nodes:
            raise _RewriteRejected(f"待删除节点 {node_id} 不存在")
    for tensor_id in removed_tensors:
        if tensor_id not in mutable.tensors:
            raise _RewriteRejected(f"待删除张量 {tensor_id} 不存在")
    snapshot = mutable.snapshot()
    try:
        for node_id in removed_nodes:
            mutable.drop_node(node_id)
        for edge_id in {item[5] for item in pending}:
            mutable.edges.pop(edge_id, None)
        for tensor_id in removed_tensors:
            mutable.drop_tensor(tensor_id)
        added_nodes: list[str] = []
        added_tensors: tuple[str, ...] = ()
        if new_node is not None:
            mutable.add_node(**new_node)
            added_nodes.append(new_node["node_id"])
            added_tensors = tuple(tensor_id for _port, tensor_id in new_node["outputs"])
        guards: list[str] = []
        for index, (src_node, src_port, dst_node, dst_port, guard, _edge_id) in enumerate(pending):
            tensor_id = mutable.output_tensor_of(src_node, src_port)
            if tensor_id is None:
                raise ValueError(f"改写失败：{src_node}.{src_port} 不存在输出张量")
            mutable.set_input_tensor(dst_node, dst_port, tensor_id)
            edge_id = f"e.rw.{index}.{dst_node}.{dst_port}"
            mutable.add_edge(edge_id, src_node, src_port, dst_node, dst_port)
            if guard:
                mutable.edges[edge_id] = replace(mutable.edges[edge_id], guard_ref=guard)
                guards.append(guard)
        for index, (src_node, src_port, dst_node, dst_port) in enumerate(connect):
            tensor_id = mutable.output_tensor_of(src_node, src_port)
            if tensor_id is None:
                raise ValueError(f"改写失败：{src_node}.{src_port} 不存在输出张量")
            mutable.set_input_tensor(dst_node, dst_port, tensor_id)
            edge_id = f"e.rw.c{index}.{dst_node}.{dst_port}"
            mutable.add_edge(edge_id, src_node, src_port, dst_node, dst_port)
            if connect_guard:
                mutable.edges[edge_id] = replace(
                    mutable.edges[edge_id], guard_ref=connect_guard
                )
                guards.append(connect_guard)
    except Exception:
        mutable.restore(snapshot)
        raise
    rewired = tuple((dst_node, dst_port) for _s, _p, dst_node, dst_port, _g, _e in pending)
    return RuleApplication(
        rule_id="",
        target_nodes=tuple(removed_nodes) + tuple(added_nodes),
        removed_nodes=tuple(removed_nodes),
        added_nodes=tuple(added_nodes),
        removed_tensors=tuple(removed_tensors),
        added_tensors=added_tensors,
        rewired=rewired,
        condition="",
        reason="",
        sources=(),
        probe=probe,
        guards=tuple(sorted(set(guards))),
    )


def _apply_r1(mutable: _Mutable, facts: frozenset[tuple]) -> RuleReport:
    applications: list[RuleApplication] = []
    rejected: list[dict[str, str]] = []
    changed = 0
    for tensor_id in sorted(mutable.tensors):
        tensor = mutable.tensors[tensor_id]
        new_axes: list[NamedAxis] = []
        tensor_changed = False
        for axis in tensor.axes:
            if axis.expression is None:
                new_axes.append(axis)
                continue
            try:
                original = shape_expr.parse(axis.expression)
                simplified = shape_expr.simplify(original, facts)
            except shape_expr.ShapeExprError as error:
                rejected.append(
                    {"tensor": tensor_id, "axis": axis.symbol, "reason": f"表达式不可化简: {error}"}
                )
                new_axes.append(axis)
                continue
            if simplified == original:
                new_axes.append(axis)
                continue
            new_axes.append(
                replace(axis, expression=shape_expr.to_text(simplified))
            )
            tensor_changed = True
            changed += 1
            applications.append(
                RuleApplication(
                    rule_id="R1-shape-expression-simplify",
                    target_nodes=(),
                    removed_nodes=(),
                    added_nodes=(),
                    removed_tensors=(),
                    added_tensors=(),
                    rewired=(),
                    condition="表达式的白名单代数恒等"
                    + (
                        "（使用了 divisible 前提）"
                        if any(fact[0] == "divisible" for fact in facts)
                        else ""
                    ),
                    reason=f"{tensor_id}.{axis.symbol}: {axis.expression} → "
                    f"{shape_expr.to_text(simplified)}",
                    sources=(axis.evidence_ref,),
                )
            )
        if tensor_changed:
            mutable.tensors[tensor_id] = replace(tensor, axes=tuple(new_axes))
    if applications:
        return RuleReport(
            rule_id="R1-shape-expression-simplify",
            title="形状表达式化简",
            precondition=_PRECONDITIONS["R1-shape-expression-simplify"],
            status="applied",
            reason=f"化简 {changed} 处形状表达式",
            applications=tuple(applications),
            rejected_candidates=tuple(rejected),
        )
    return RuleReport(
        rule_id="R1-shape-expression-simplify",
        title="形状表达式化简",
        precondition=_PRECONDITIONS["R1-shape-expression-simplify"],
        status="rejected",
        reason="没有可证明的白名单恒等（动态变量保持符号，未使用示例常量）"
        if not rejected
        else "候选表达式均不可化简",
        rejected_candidates=tuple(rejected),
    )


def _apply_r2(mutable: _Mutable) -> RuleReport:
    applications: list[RuleApplication] = []
    rejected: list[dict[str, str]] = []
    for node_id in sorted(mutable.nodes):
        if node_id not in mutable.nodes:
            continue
        meta = mutable.node_meta[node_id]
        if meta.kind not in _LAYOUT_KINDS | {"broadcast"}:
            continue
        edges_in = mutable.data_inputs(node_id)
        output_tensor = mutable.single_output_tensor(node_id)
        if len(edges_in) != 1 or output_tensor is None:
            rejected.append(
                {"node": node_id, "reason": "输入/输出不是单一数据依赖，无法确认恒等且不误删分支"}
            )
            continue
        edge_in = edges_in[0]
        source = mutable.tensors[edge_in.tensor_id]
        target = mutable.tensors[output_tensor]
        if not _layout_identical(source, target):
            rejected.append(
                {
                    "node": node_id,
                    "reason": f"轴/布局不恒等：{_axis_order(source)} → {_axis_order(target)}"
                    f"（物理 {tuple(source.physical_layout)} → {tuple(target.physical_layout)}）",
                }
            )
            continue
        if meta.condition is not None:
            rejected.append({"node": node_id, "reason": "条件路径节点，不按恒等删除"})
            continue
        probe_kind = "transpose" if meta.kind == "transpose" else "reshape"
        extents = _probe_extents(mutable, _axis_order(source), _axis_order(target))
        probe_ok, probe_detail = layout_probe.identity_chain(
            _axis_order(source), [(probe_kind, _axis_order(target))], extents
        )
        if not probe_ok:
            rejected.append({"node": node_id, "reason": probe_detail})
            continue
        old_output = output_tensor
        try:
            application = _rewrite(
                mutable,
                removed_nodes=(node_id,),
                removed_tensors=(old_output,),
                forward=((old_output, edge_in.src_node_id, edge_in.src_port),),
                probe=probe_detail,
            )
        except _RewriteRejected as error:
            rejected.append({"node": node_id, "reason": str(error)})
            continue
        application = replace(
            application,
            rule_id="R2-identity-eliminate",
            condition=f"输出轴线顺序/物理布局与输入一致（{_axis_order(source)}）",
            reason=f"恒等 {meta.kind} 节点 {node_id} 消除，消费者改接 {edge_in.tensor_id}",
            sources=tuple(source_ref.location for source_ref in meta.sources),
        )
        applications.append(application)
    if applications:
        return RuleReport(
            rule_id="R2-identity-eliminate",
            title="恒等变换消除",
            precondition=_PRECONDITIONS["R2-identity-eliminate"],
            status="applied",
            reason=f"消除 {len(applications)} 个恒等布局节点",
            applications=tuple(applications),
            rejected_candidates=tuple(rejected),
        )
    return RuleReport(
        rule_id="R2-identity-eliminate",
        title="恒等变换消除",
        precondition=_PRECONDITIONS["R2-identity-eliminate"],
        status="rejected",
        reason="本图没有可证明的恒等布局节点（候选见拒绝原因）",
        rejected_candidates=tuple(rejected),
    )


def _apply_r3(mutable: _Mutable) -> RuleReport:
    applications: list[RuleApplication] = []
    rejected: list[dict[str, str]] = []
    for node_id in sorted(mutable.nodes):
        if node_id not in mutable.nodes:
            continue
        meta = mutable.node_meta[node_id]
        if meta.kind != "transpose":
            continue
        inner = mutable.single_data_input(node_id)
        if inner is None:
            rejected.append({"node": node_id, "reason": "Transpose 的输入不是单一数据依赖"})
            continue
        producer = inner.src_node_id
        if mutable.node_meta[producer].kind != "transpose":
            rejected.append({"node": node_id, "reason": "相邻上游不是 Transpose"})
            continue
        if len(mutable.nodes[producer].output_ports) != 1:
            rejected.append({"node": node_id, "reason": "上游 Transpose 有多个输出/跳线"})
            continue
        if len(mutable.tensor_consumers(inner.tensor_id)) != 1:
            rejected.append(
                {
                    "node": node_id,
                    "reason": "中间张量存在其它消费者，保留中间分支",
                }
            )
            continue
        outer_in = mutable.single_data_input(producer)
        if outer_in is None:
            rejected.append({"node": node_id, "reason": "上游 Transpose 输入不是单一数据依赖"})
            continue
        output_tensor = mutable.single_output_tensor(node_id)
        if output_tensor is None:
            rejected.append({"node": node_id, "reason": "Transpose 输出不是单一张量"})
            continue
        source = mutable.tensors[outer_in.tensor_id]
        intermediate = mutable.tensors[inner.tensor_id]
        target = mutable.tensors[output_tensor]
        composed = _axis_order(source) == _axis_order(target)
        if not composed:
            rejected.append(
                {
                    "node": node_id,
                    "reason": "两个 Transpose 复合不是恒等置换："
                    f"{_axis_order(source)} → {_axis_order(intermediate)} → {_axis_order(target)}",
                }
            )
            continue
        extents = _probe_extents(
            mutable, _axis_order(source), _axis_order(intermediate), _axis_order(target)
        )
        probe_ok, probe_detail = layout_probe.identity_chain(
            _axis_order(source),
            [
                ("transpose", _axis_order(intermediate)),
                ("transpose", _axis_order(target)),
            ],
            extents,
        )
        if not probe_ok:
            rejected.append({"node": node_id, "reason": probe_detail})
            continue
        old_output = output_tensor
        try:
            application = _rewrite(
                mutable,
                removed_nodes=(producer, node_id),
                removed_tensors=(inner.tensor_id, old_output),
                forward=((old_output, outer_in.src_node_id, outer_in.src_port),),
                probe=probe_detail,
            )
        except _RewriteRejected as error:
            rejected.append({"node": node_id, "reason": str(error)})
            continue
        application = replace(
            application,
            rule_id="R3-transpose-pair-cancel",
            condition=(
                f"Transpose({producer}) ∘ Transpose({node_id}) = 恒等；"
                "中间张量只有一个消费者"
            ),
            reason=(
                f"抵消相邻 Transpose 对，消费者改接 {outer_in.tensor_id}；"
                f"轴顺序 {_axis_order(source)} 保持不变"
            ),
            sources=tuple(
                sorted(
                    {
                        source_ref.location
                        for source_ref in (
                            mutable.bundle.node_meta[producer].sources
                            + mutable.bundle.node_meta[node_id].sources
                        )
                    }
                )
            ),
        )
        applications.append(application)
    if applications:
        return RuleReport(
            rule_id="R3-transpose-pair-cancel",
            title="连续 Transpose 复合/抵消",
            precondition=_PRECONDITIONS["R3-transpose-pair-cancel"],
            status="applied",
            reason=f"抵消 {len(applications)} 组相邻 Transpose",
            applications=tuple(applications),
            rejected_candidates=tuple(rejected),
        )
    return RuleReport(
        rule_id="R3-transpose-pair-cancel",
        title="连续 Transpose 复合/抵消",
        precondition=_PRECONDITIONS["R3-transpose-pair-cancel"],
        status="rejected",
        reason="没有相邻且复合为恒等的 Transpose 对",
        rejected_candidates=tuple(rejected),
    )


def _apply_r4(mutable: _Mutable) -> RuleReport:
    applications: list[RuleApplication] = []
    rejected: list[dict[str, str]] = []
    for node_id in sorted(mutable.nodes):
        if node_id not in mutable.nodes:
            continue
        meta = mutable.node_meta[node_id]
        if meta.kind not in {"reshape", "view"}:
            continue
        inner = mutable.single_data_input(node_id)
        if inner is None:
            rejected.append({"node": node_id, "reason": "Reshape/View 输入不是单一数据依赖"})
            continue
        producer = inner.src_node_id
        if mutable.node_meta[producer].kind not in {"reshape", "view"}:
            rejected.append({"node": node_id, "reason": "相邻上游不是 Reshape/View"})
            continue
        if len(mutable.nodes[producer].output_ports) != 1 or len(
            mutable.tensor_consumers(inner.tensor_id)
        ) != 1:
            rejected.append(
                {"node": node_id, "reason": "中间张量存在其它使用者，不能跨越合并"}
            )
            continue
        outer_in = mutable.single_data_input(producer)
        if outer_in is None:
            rejected.append({"node": node_id, "reason": "上游 Reshape 输入不是单一数据依赖"})
            continue
        output_tensor = mutable.single_output_tensor(node_id)
        if output_tensor is None:
            rejected.append({"node": node_id, "reason": "下游 Reshape 输出不是单一数据依赖"})
            continue
        source_tensor = outer_in.tensor_id
        new_id = f"rw.{producer}+{node_id}"
        sources = _merge_sources(
            mutable.node_meta[producer].sources, mutable.node_meta[node_id].sources
        )
        legacy = tuple(
            dict.fromkeys(
                mutable.node_meta[producer].legacy_refs
                + mutable.node_meta[node_id].legacy_refs
            )
        )
        extents = _probe_extents(
            mutable,
            _axis_order(mutable.tensors[source_tensor]),
            _axis_order(mutable.tensors[inner.tensor_id]),
            _axis_order(mutable.tensors[output_tensor]),
        )
        probe_ok, probe_detail = layout_probe.identity_chain(
            _axis_order(mutable.tensors[source_tensor]),
            [
                ("reshape", _axis_order(mutable.tensors[inner.tensor_id])),
                ("reshape", _axis_order(mutable.tensors[output_tensor])),
            ],
            extents,
        )
        if not probe_ok:
            rejected.append({"node": node_id, "reason": probe_detail})
            continue
        new_node = {
            "node_id": new_id,
            "op_type": mutable.nodes[node_id].op_type,
            "granularity": "reshape-merged",
            "label": "Reshape（连续 View/Reshape 合并）",
            "kind": "reshape",
            "module_path": mutable.node_meta[node_id].module_path,
            "sources": sources,
            "inputs": (("x", source_tensor),),
            "outputs": (("y", output_tensor),),
            "legacy_refs": legacy,
            "params": (),
            "note": "无中间计算的连续 Reshape/View 合并；元素顺序与合法输入域不变",
        }
        try:
            application = _rewrite(
                mutable,
                removed_nodes=(producer, node_id),
                removed_tensors=(inner.tensor_id,),
                new_node=new_node,
                forward=((output_tensor, new_id, "y"),),
                connect=((outer_in.src_node_id, outer_in.src_port, new_id, "x"),),
                probe=probe_detail,
            )
        except _RewriteRejected as error:
            rejected.append({"node": node_id, "reason": str(error)})
            continue
        application = replace(
            application,
            rule_id="R4-reshape-merge",
            condition="两个 Reshape/View 相邻且中间张量无其它消费者",
            reason=f"{producer} + {node_id} 合并为 {new_id}（一次 Reshape）",
            sources=tuple(source.location for source in sources),
        )
        applications.append(application)
    if applications:
        return RuleReport(
            rule_id="R4-reshape-merge",
            title="连续 Reshape/View 合并",
            precondition=_PRECONDITIONS["R4-reshape-merge"],
            status="applied",
            reason=f"合并 {len(applications)} 组连续 Reshape/View",
            applications=tuple(applications),
            rejected_candidates=tuple(rejected),
        )
    return RuleReport(
        rule_id="R4-reshape-merge",
        title="连续 Reshape/View 合并",
        precondition=_PRECONDITIONS["R4-reshape-merge"],
        status="rejected",
        reason="本图没有相邻的 Reshape/View 对（View 之后是 Transpose/计算，不是 Reshape）",
        rejected_candidates=tuple(rejected),
    )


def _apply_r5(mutable: _Mutable) -> RuleReport:
    applications: list[RuleApplication] = []
    rejected: list[dict[str, str]] = []
    for node_id in sorted(mutable.nodes):
        if node_id not in mutable.nodes:
            continue
        meta = mutable.node_meta[node_id]
        if meta.kind != "transpose":
            continue
        inner = mutable.single_data_input(node_id)
        if inner is None:
            rejected.append({"node": node_id, "reason": "Transpose 输入不是单一数据依赖"})
            continue
        view_id = inner.src_node_id
        if mutable.node_meta[view_id].kind != "view":
            continue
        if len(mutable.nodes[view_id].output_ports) != 1 or len(
            mutable.tensor_consumers(inner.tensor_id)
        ) != 1:
            rejected.append({"node": node_id, "reason": "View 输出存在其它消费者，保留分支"})
            continue
        view_in = mutable.single_data_input(view_id)
        if view_in is None:
            rejected.append({"node": view_id, "reason": "View 输入不是单一数据依赖"})
            continue
        output_tensor = mutable.single_output_tensor(node_id)
        if output_tensor is None:
            rejected.append({"node": node_id, "reason": "Transpose 输出不是单一数据依赖"})
            continue
        source = mutable.tensors[view_in.tensor_id]
        viewed = mutable.tensors[inner.tensor_id]
        target = mutable.tensors[output_tensor]
        if _non_unit_order(viewed) != _non_unit_order(target):
            rejected.append(
                {
                    "node": node_id,
                    "reason": (
                        "Transpose 改变了非单位轴的相对顺序（比较 View 之后与 Transpose "
                        f"之后）：{_non_unit_order(viewed)} → {_non_unit_order(target)}；"
                        "不能化为一次 Reshape"
                    ),
                }
            )
            continue
        if len(_axis_order(target)) == len(_axis_order(source)):
            rejected.append({"node": node_id, "reason": "目标秩未变化，不构成本规则前提"})
            continue
        extents = _probe_extents(
            mutable, _axis_order(source), _axis_order(viewed), _axis_order(target)
        )
        probe_ok, probe_detail = layout_probe.identity_chain(
            _axis_order(source),
            [
                ("view", _axis_order(viewed)),
                ("transpose", _axis_order(target)),
            ],
            extents,
        )
        if not probe_ok:
            rejected.append({"node": node_id, "reason": probe_detail})
            continue
        new_id = f"rw.{view_id}+{node_id}"
        sources = _merge_sources(
            mutable.node_meta[view_id].sources, mutable.node_meta[node_id].sources
        )
        legacy = tuple(
            dict.fromkeys(
                mutable.node_meta[view_id].legacy_refs + mutable.node_meta[node_id].legacy_refs
            )
        )
        new_node = {
            "node_id": new_id,
            "op_type": "reshape",
            "granularity": "unit-axis-reshape",
            "label": "Reshape（单 KV 头：View+Transpose 合并）",
            "kind": "reshape",
            "module_path": mutable.node_meta[node_id].module_path,
            "sources": sources,
            "inputs": (("x", view_in.tensor_id),),
            "outputs": (("y", output_tensor),),
            "legacy_refs": legacy,
            "params": ("num_key_value_heads",),
            "note": (
                "num_key_value_heads=1 时 [batch_size, seqlen, 1, head_dim] 的 "
                "Transpose(1,2) 是纯单位轴重排，等价于一次 4D Reshape；"
                "三维→四维的秩变化保留"
            ),
            "attributes": {"unit_axis": "num_key_value_heads", "rank_change": "3->4"},
        }
        try:
            application = _rewrite(
                mutable,
                removed_nodes=(view_id, node_id),
                removed_tensors=(inner.tensor_id,),
                new_node=new_node,
                forward=((output_tensor, new_id, "y"),),
                connect=((view_in.src_node_id, view_in.src_port, new_id, "x"),),
                probe=probe_detail,
            )
        except _RewriteRejected as error:
            rejected.append({"node": node_id, "reason": str(error)})
            continue
        application = replace(
            application,
            rule_id="R5-unit-axis-reorder",
            condition=(
                "Transpose 只在单位轴之间重排（比较 View 之后与 Transpose 之后）："
                f"{_non_unit_order(viewed)} == {_non_unit_order(target)}；"
                "num_key_value_heads=1"
            ),
            reason=(
                f"{view_id} + {node_id} 化为一次 4D Reshape {new_id}；"
                f"保留单位 KV 头轴（{_axis_order(viewed)}）与三维→四维变化"
            ),
            sources=tuple(source.location for source in sources),
        )
        applications.append(application)
    if applications:
        return RuleReport(
            rule_id="R5-unit-axis-reorder",
            title="可证明的单位轴重排",
            precondition=_PRECONDITIONS["R5-unit-axis-reorder"],
            status="applied",
            reason=f"合并 {len(applications)} 组单位轴 View+Transpose",
            applications=tuple(applications),
            rejected_candidates=tuple(rejected),
        )
    return RuleReport(
        rule_id="R5-unit-axis-reorder",
        title="可证明的单位轴重排",
        precondition=_PRECONDITIONS["R5-unit-axis-reorder"],
        status="rejected",
        reason="没有满足单位轴前提的 View+Transpose" if not rejected else "候选不满足单位轴前提",
        rejected_candidates=tuple(rejected),
    )


def _apply_r6(mutable: _Mutable) -> RuleReport:
    applications: list[RuleApplication] = []
    rejected: list[dict[str, str]] = []
    for node_id in sorted(mutable.nodes):
        if node_id not in mutable.nodes:
            continue
        meta = mutable.node_meta[node_id]
        if meta.kind != "concat":
            continue
        inputs = mutable.data_inputs(node_id)
        if len(inputs) < 2:
            rejected.append({"node": node_id, "reason": "Concat 输入少于 2 个"})
            continue
        producers = {edge.src_node_id for edge in inputs}
        if len(producers) != 1:
            rejected.append(
                {
                    "node": node_id,
                    "reason": "拼接输入来自不同生产者，不是完整 Split 后原样拼回",
                }
            )
            continue
        producer = producers.pop()
        if mutable.node_meta[producer].kind != "split":
            rejected.append({"node": node_id, "reason": "上游不是 Split"})
            continue
        split_edges = mutable.edges_out(producer)
        if len(split_edges) != len(inputs):
            rejected.append({"node": node_id, "reason": "Split 分支数与 Concat 输入数不一致"})
            continue
        if any(len(mutable.tensor_consumers(edge.tensor_id)) != 1 for edge in split_edges):
            rejected.append({"node": node_id, "reason": "Split 分支被其它计算消费，保留"})
            continue
        by_dst_port = {edge.dst_port: edge for edge in inputs}
        concat_order = tuple(
            by_dst_port[port.name].src_port
            for port in mutable.nodes[node_id].input_ports
            if port.name in by_dst_port
        )
        split_order = tuple(port.name for port in mutable.nodes[producer].output_ports)
        if concat_order != split_order:
            rejected.append(
                {
                    "node": node_id,
                    "reason": f"Concat 顺序 {concat_order} 与 Split 顺序 {split_order} 不同",
                }
            )
            continue
        split_in = mutable.single_data_input(producer)
        output_tensor = mutable.single_output_tensor(node_id)
        if split_in is None or output_tensor is None:
            rejected.append({"node": node_id, "reason": "Split/Concat 输入输出不是单一数据依赖"})
            continue
        source_axes = _axis_order(mutable.tensors[split_in.tensor_id])
        output_axes = _axis_order(mutable.tensors[output_tensor])
        if source_axes != output_axes:
            rejected.append({"node": node_id, "reason": "拼回形状与切分前不一致"})
            continue
        old_output = output_tensor
        try:
            application = _rewrite(
                mutable,
                removed_nodes=(producer, node_id),
                removed_tensors=tuple(edge.tensor_id for edge in split_edges) + (old_output,),
                forward=((old_output, split_in.src_node_id, split_in.src_port),),
                probe="structural: 完整 Split + 同轴同序原样 Concat（元素顺序不变）",
            )
        except _RewriteRejected as error:
            rejected.append({"node": node_id, "reason": str(error)})
            continue
        application = replace(
            application,
            rule_id="R6-split-concat-cancel",
            condition="完整 Split、同轴同序原样 Concat、无其它消费者",
            reason=f"抵消 {producer} + {node_id}，消费者改接 {split_in.tensor_id}",
            sources=tuple(
                sorted(
                    {
                        source_ref.location
                        for source_ref in (
                            mutable.bundle.node_meta[producer].sources
                            + mutable.bundle.node_meta[node_id].sources
                        )
                    }
                )
            ),
        )
        applications.append(application)
    if applications:
        return RuleReport(
            rule_id="R6-split-concat-cancel",
            title="完整原样 Split/Concat 抵消",
            precondition=_PRECONDITIONS["R6-split-concat-cancel"],
            status="applied",
            reason=f"抵消 {len(applications)} 组 Split/Concat",
            applications=tuple(applications),
            rejected_candidates=tuple(rejected),
        )
    return RuleReport(
        rule_id="R6-split-concat-cancel",
        title="完整原样 Split/Concat 抵消",
        precondition=_PRECONDITIONS["R6-split-concat-cancel"],
        status="rejected",
        reason="本图没有完整 Split 后原样拼回的候选"
        if not rejected
        else "候选不满足抵消前提",
        rejected_candidates=tuple(rejected),
    )


def _apply_r7_fusion(
    mutable: _Mutable,
    *,
    node_id: str,
    producer: str,
    edge_in: Edge,
    transpose_in: Edge,
    intermediate: Tensor,
    guard: str | None,
) -> None:
    """R7 提交阶段的实际变更（由调用方用快照保护）。"""
    mutable.set_input_tensor(node_id, edge_in.dst_port, transpose_in.tensor_id)
    del mutable.edges[edge_in.stable_id]
    mutable.drop_node(producer)
    mutable.drop_tensor(intermediate.stable_id)


def _apply_r7(mutable: _Mutable) -> RuleReport:
    applications: list[RuleApplication] = []
    rejected: list[dict[str, str]] = []
    for node_id in sorted(mutable.nodes):
        if node_id not in mutable.nodes:
            continue
        meta = mutable.node_meta[node_id]
        if meta.kind != "matmul":
            continue
        attrs = dict(meta.attributes)
        port_names = [port.name for port in mutable.nodes[node_id].input_ports]
        lhs_port = attrs.get("lhs_port") or ("lhs" if "lhs" in port_names else None)
        rhs_port = attrs.get("rhs_port") or ("rhs" if "rhs" in port_names else None)
        if lhs_port is None or rhs_port is None or lhs_port == rhs_port:
            rejected.append(
                {
                    "node": node_id,
                    "reason": "未声明矩阵操作数角色（lhs_port/rhs_port），不能确定转置语义",
                }
            )
            continue
        lhs_symbol = attrs.get("lhs_symbol", "A")
        rhs_symbol = attrs.get("rhs_symbol", "B")
        data_inputs = mutable.data_inputs(node_id)
        transpose_edges = [
            edge
            for edge in data_inputs
            if mutable.node_meta[edge.src_node_id].kind == "transpose"
        ]
        if len(transpose_edges) > 1:
            rejected.append(
                {
                    "node": node_id,
                    "reason": "两侧矩阵操作数同时带 Transpose，不在本轮批准范围内，保留原结构",
                }
            )
            continue
        for edge_in in transpose_edges:
            producer = edge_in.src_node_id
            if edge_in.dst_port == rhs_port:
                operand = "rhs"
            elif edge_in.dst_port == lhs_port:
                operand = "lhs"
            else:
                rejected.append(
                    {
                        "node": node_id,
                        "reason": (
                            f"Transpose 不在矩阵操作数端口 {sorted([lhs_port, rhs_port])} 上"
                            f"（端口 {edge_in.dst_port!r}），不改变计算解释"
                        ),
                    }
                )
                continue
            intermediate = mutable.tensors[edge_in.tensor_id]
            if len(mutable.tensor_consumers(edge_in.tensor_id)) != 1:
                rejected.append(
                    {
                        "node": node_id,
                        "reason": "Transpose 结果还有其它消费者，保留分支",
                    }
                )
                continue
            transpose_in = mutable.single_data_input(producer)
            if transpose_in is None:
                rejected.append(
                    {"node": node_id, "reason": "Transpose 输入不是单一数据依赖"}
                )
                continue
            source = mutable.tensors[transpose_in.tensor_id]
            extents = _probe_extents(
                mutable, _axis_order(source), _axis_order(intermediate)
            )
            probe_ok, probe_detail = layout_probe.last_two_swap_chain(
                _axis_order(source), _axis_order(intermediate), extents
            )
            if not probe_ok:
                rejected.append({"node": node_id, "reason": probe_detail})
                continue
            guards = {
                edge.guard_ref
                for edge in (edge_in, transpose_in)
                if edge.guard_ref
            }
            if len(guards) > 1:
                rejected.append(
                    {
                        "node": node_id,
                        "reason": f"转置前后执行条件不同 {sorted(guards)}，无法在合并后保持",
                    }
                )
                continue
            guard = next(iter(guards), None)
            sources = _merge_sources(
                mutable.node_meta[node_id].sources, mutable.node_meta[producer].sources
            )
            legacy = tuple(
                dict.fromkeys(
                    mutable.node_meta[node_id].legacy_refs
                    + mutable.node_meta[producer].legacy_refs
                )
            )
            # 改接 MatMul 的输入端口到转置前张量：删旧边/节点，再从转置前的生产者重建一条边。
            # 事务保护：任何异常都回滚，拒绝/失败不留下部分修改。
            snapshot = mutable.snapshot()
            try:
                _apply_r7_fusion(
                    mutable,
                    node_id=node_id,
                    producer=producer,
                    edge_in=edge_in,
                    transpose_in=transpose_in,
                    intermediate=intermediate,
                    guard=guard,
                )
            except Exception:
                mutable.restore(snapshot)
                raise
            edge_id = f"e.rw.k{node_id}.{edge_in.dst_port}"
            mutable.add_edge(
                edge_id,
                transpose_in.src_node_id,
                transpose_in.src_port,
                node_id,
                edge_in.dst_port,
            )
            if guard:
                mutable.edges[edge_id] = replace(
                    mutable.edges[edge_id], guard_ref=guard
                )
            label = (
                f"{lhs_symbol} @ {rhs_symbol}ᵀ"
                if operand == "rhs"
                else f"{lhs_symbol}ᵀ @ {rhs_symbol}"
            )
            updated_attrs = {**attrs, "transposed_operand": operand}
            if operand == "rhs":
                updated_attrs["rhs_transposed_axes"] = "last_two_axes"
                updated_attrs["rhs_transpose_source"] = producer
                interpretation = f"右矩阵 {rhs_symbol} 的 Transpose 并入矩阵乘语义（{label}）"
            else:
                updated_attrs["lhs_transposed_axes"] = "last_two_axes"
                updated_attrs["lhs_transpose_source"] = producer
                interpretation = f"左矩阵 {lhs_symbol} 的 Transpose 并入矩阵乘语义（{label}）"
            previous = mutable.node_meta[node_id]
            mutable.node_meta[node_id] = replace(
                previous,
                label=f"{label} · MatMul",
                sources=sources,
                legacy_refs=legacy,
                attributes=updated_attrs,
                note=(
                    f"{interpretation}：只交换最后两个矩阵轴，"
                    "batch/head/seqlen 轴未被吞掉；逻辑计算等价不代表设备已融合"
                ),
            )
            mutable.nodes[node_id] = replace(
                mutable.nodes[node_id],
                evidence_refs=tuple(source_ref.location for source_ref in sources),
            )
            applications.append(
                RuleApplication(
                    rule_id="R7-transpose-into-matmul",
                    target_nodes=(producer, node_id),
                    removed_nodes=(producer,),
                    added_nodes=(),
                    removed_tensors=(intermediate.stable_id,),
                    added_tensors=(),
                    rewired=((node_id, edge_in.dst_port),),
                    condition=(
                        f"操作数角色={operand}；Transpose 只交换最后两个矩阵轴、"
                        "结果只被该 MatMul 消费："
                        f"{_axis_order(source)} → {_axis_order(intermediate)}"
                    ),
                    reason=(
                        f"{node_id} 的 {operand} 输入改为 {transpose_in.tensor_id}，"
                        f"标注 {label}；保留 {producer} 的来源"
                    ),
                    sources=tuple(source_ref.location for source_ref in sources),
                    probe=probe_detail,
                    guards=(guard,) if guard else (),
                )
            )
    if applications:
        return RuleReport(
            rule_id="R7-transpose-into-matmul",
            title="Transpose 并入矩阵乘语义",
            precondition=_PRECONDITIONS["R7-transpose-into-matmul"],
            status="applied",
            reason=f"并入 {len(applications)} 处矩阵乘转置",
            applications=tuple(applications),
            rejected_candidates=tuple(rejected),
        )
    return RuleReport(
        rule_id="R7-transpose-into-matmul",
        title="Transpose 并入矩阵乘语义",
        precondition=_PRECONDITIONS["R7-transpose-into-matmul"],
        status="rejected",
        reason="没有只交换最后两个矩阵轴的相邻 Transpose" if not rejected else "候选不满足前提",
        rejected_candidates=tuple(rejected),
    )


def _merge_sources(*groups: Sequence[SourceRef]) -> tuple[SourceRef, ...]:
    merged: list[SourceRef] = []
    seen: set[tuple] = set()
    for group in groups:
        for source in group:
            key = (source.repo_key, source.path, source.line, source.end_line)
            if key not in seen:
                seen.add(key)
                merged.append(source)
    return tuple(merged)


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------


def normalize(
    bundle: SemanticBundle,
    *,
    facts: Iterable[tuple] = (),
    max_passes: int = 4,
) -> NormalizationResult:
    """按固定规则顺序跑到不动点（最多 ``max_passes`` 轮），返回审计与映射。"""
    fact_set = frozenset(facts)
    mutable = _Mutable(bundle)
    raw_node_ids = sorted(mutable.nodes)
    raw_tensor_ids = sorted(mutable.tensors)
    by_rule: dict[str, RuleReport] = {}
    removed_nodes: dict[str, dict[str, str]] = {}
    removed_tensors: dict[str, str] = {}
    for pass_index in range(max_passes):
        changed = False
        for rule in (_apply_r1, _apply_r2, _apply_r3, _apply_r4, _apply_r5, _apply_r6, _apply_r7):
            if rule is _apply_r1:
                report = rule(mutable, fact_set)
            else:
                report = rule(mutable)
            previous = by_rule.get(report.rule_id)
            if previous is None or previous.status != "applied":
                merged_applications = (
                    previous.applications if previous is not None else ()
                ) + report.applications
                merged_rejected = (
                    previous.rejected_candidates if previous is not None else ()
                ) + report.rejected_candidates
                by_rule[report.rule_id] = RuleReport(
                    rule_id=report.rule_id,
                    title=report.title,
                    precondition=report.precondition,
                    status="applied" if merged_applications else "rejected",
                    reason=report.reason if report.applications else (previous.reason if previous else report.reason),
                    applications=merged_applications,
                    rejected_candidates=merged_rejected,
                )
            else:
                by_rule[report.rule_id] = RuleReport(
                    rule_id=previous.rule_id,
                    title=previous.title,
                    precondition=previous.precondition,
                    status="applied",
                    reason=previous.reason,
                    applications=previous.applications + report.applications,
                    rejected_candidates=previous.rejected_candidates
                    + report.rejected_candidates,
                )
            if report.applications:
                changed = True
                for application in report.applications:
                    for node_id in application.removed_nodes:
                        removed_nodes.setdefault(
                            node_id,
                            {"rule_id": application.rule_id, "reason": application.reason},
                        )
                    for tensor_id in application.removed_tensors:
                        removed_tensors.setdefault(tensor_id, application.rule_id)
        if not changed or pass_index == max_passes - 1:
            break
    reports = [by_rule[rule_id] for rule_id in RULES if rule_id in by_rule]

    normalized = mutable.build()
    node_mapping = _build_node_mapping(bundle, normalized, removed_nodes)
    tensor_mapping = _build_tensor_mapping(bundle, normalized, removed_tensors)
    return NormalizationResult(
        raw=bundle,
        normalized=normalized,
        reports=tuple(reports),
        node_mapping=node_mapping,
        tensor_mapping=tensor_mapping,
        removed_nodes=removed_nodes,
        removed_tensors=removed_tensors,
    )


def _build_node_mapping(
    raw: SemanticBundle,
    normalized: SemanticBundle,
    removed: Mapping[str, dict[str, str]],
) -> dict[str, tuple[str, ...]]:
    normalized_ids = {node.stable_id for node in normalized.graph.nodes}
    adjacency_out: dict[str, set[str]] = {}
    adjacency_in: dict[str, set[str]] = {}
    for edge in normalized.graph.edges:
        if edge.kind != "data":
            continue
        adjacency_out.setdefault(edge.src_node_id, set()).add(edge.dst_node_id)
        adjacency_in.setdefault(edge.dst_node_id, set()).add(edge.src_node_id)
    mapping: dict[str, tuple[str, ...]] = {}
    for node in raw.graph.nodes:
        node_id = node.stable_id
        if node_id in normalized_ids and node_id not in removed:
            mapping[node_id] = (node_id,)
            continue
        # 被移除的布局节点：映射到它在化简图中的直接上游与下游节点（定位用）
        neighbors: set[str] = set()
        for edge in raw.graph.edges:
            if edge.src_node_id == node_id:
                neighbors.update(adjacency_in.get(edge.dst_node_id, set()))
                neighbors.add(edge.dst_node_id)
            if edge.dst_node_id == node_id:
                neighbors.update(adjacency_out.get(edge.src_node_id, set()))
                neighbors.add(edge.src_node_id)
        mapping[node_id] = tuple(sorted(item for item in neighbors if item in normalized_ids))
    return mapping


def _build_tensor_mapping(
    raw: SemanticBundle,
    normalized: SemanticBundle,
    removed: Mapping[str, str],
) -> dict[str, str]:
    normalized_ids = {tensor.stable_id for tensor in normalized.graph.tensors}
    mapping: dict[str, str] = {}
    for tensor in raw.graph.tensors:
        tensor_id = tensor.stable_id
        if tensor_id in normalized_ids:
            mapping[tensor_id] = tensor_id
            continue
        # 被移除的张量：找化简图中同名轴/布局的最近张量（定位用）
        candidates = [
            candidate.stable_id
            for candidate in normalized.graph.tensors
            if _axis_order(candidate) == _axis_order(tensor)
        ]
        mapping[tensor_id] = candidates[0] if candidates else ""
    return mapping


# ---------------------------------------------------------------------------
# 保持性校验
# ---------------------------------------------------------------------------


def _node_adjacency(bundle: SemanticBundle) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for edge in bundle.graph.edges:
        if edge.kind in ("data", "state"):
            adjacency.setdefault(edge.src_node_id, set()).add(edge.dst_node_id)
    return adjacency


def _reachable(adjacency: Mapping[str, set[str]], start: str) -> set[str]:
    seen: set[str] = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in adjacency.get(current, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def validate_normalization(result: NormalizationResult) -> tuple[BundleIssue, ...]:
    """化简保持性校验：引用有效、映射覆盖、来源/状态保留、anchor 可达关系不变。"""
    issues: list[BundleIssue] = list(validate_bundle(result.normalized))
    for report in result.reports:
        if not report.reason.strip():
            issues.append(
                BundleIssue(
                    code="rule-reason-missing",
                    path=f"rules[{report.rule_id}]",
                    message="规则结论缺少适用/拒绝原因",
                )
            )
        if report.status == "applied" and not report.applications:
            issues.append(
                BundleIssue(
                    code="rule-application-missing",
                    path=f"rules[{report.rule_id}]",
                    message="规则标记 applied 但没有应用明细",
                )
            )

    normalized_nodes = {node.stable_id for node in result.normalized.graph.nodes}
    for raw_node_id, mapped in result.node_mapping.items():
        if raw_node_id not in result.removed_nodes and raw_node_id in normalized_nodes:
            if mapped != (raw_node_id,):
                issues.append(
                    BundleIssue(
                        code="mapping-inconsistent",
                        path=f"node_mapping[{raw_node_id}]",
                        message=f"保留节点映射应为自身，得到 {mapped}",
                    )
                )
        if not mapped:
            issues.append(
                BundleIssue(
                    code="mapping-unresolved",
                    path=f"node_mapping[{raw_node_id}]",
                    message="原始节点在化简图中没有可定位的对应节点",
                )
            )

    raw_tensor_ids = {tensor.stable_id for tensor in result.raw.graph.tensors}
    normalized_tensor_ids = {tensor.stable_id for tensor in result.normalized.graph.tensors}
    for tensor_id in raw_tensor_ids:
        if tensor_id not in normalized_tensor_ids and tensor_id not in result.removed_tensors:
            issues.append(
                BundleIssue(
                    code="tensor-disappeared",
                    path=f"tensors[{tensor_id}]",
                    message="原始张量既不在化简图中，也没有删除记录",
                )
            )

    # 来源保全：化简节点的来源必须覆盖被它替代的原始节点来源（规则应用里已合并）
    for application in (item for report in result.reports for item in report.applications):
        if application.removed_nodes and not application.sources:
            issues.append(
                BundleIssue(
                    code="source-not-preserved",
                    path=f"rules[{application.rule_id}]",
                    message="改写删除节点但没有记录来源",
                )
            )

    # 执行条件保全：guard 不得丢失或凭空新增（改接必须把条件带到新边）
    raw_guards = Counter(
        edge.guard_ref
        for edge in result.raw.graph.edges
        if edge.kind == "data" and edge.guard_ref
    )
    normalized_guards = Counter(
        edge.guard_ref
        for edge in result.normalized.graph.edges
        if edge.kind == "data" and edge.guard_ref
    )
    for guard in raw_guards:
        if normalized_guards.get(guard, 0) == 0:
            issues.append(
                BundleIssue(
                    code="guard-lost",
                    path=f"guards[{guard}]",
                    message="化简后没有数据边再携带该执行条件（条件被静默丢弃）",
                )
            )
    for guard in normalized_guards:
        if guard not in raw_guards:
            issues.append(
                BundleIssue(
                    code="guard-introduced",
                    path=f"guards[{guard}]",
                    message="化简后出现原始图没有的执行条件",
                )
            )
    applications = [item for report in result.reports for item in report.applications]
    for application in applications:
        for guard in application.guards:
            if guard not in raw_guards:
                issues.append(
                    BundleIssue(
                        code="guard-unknown",
                        path=f"rules[{application.rule_id}]",
                        message=f"改写记录的执行条件 {guard} 不在原始图中",
                    )
                )

    # 布局改写必须有确定性索引映射探针（结构规则需给出结构性理由）
    for application in applications:
        if application.rule_id in _PROBE_REQUIRED and not application.probe:
            issues.append(
                BundleIssue(
                    code="rewrite-without-probe",
                    path=f"rules[{application.rule_id}]",
                    message="布局改写没有索引映射探针记录，无法证明值/顺序保持",
                )
            )
        if application.rule_id == "R6-split-concat-cancel" and not application.probe:
            issues.append(
                BundleIssue(
                    code="rewrite-without-probe",
                    path="rules[R6-split-concat-cancel]",
                    message="Split/Concat 抵消没有结构性理由记录",
                )
            )

    # 状态语义：状态转移与 state 边必须保留
    raw_transitions = {item.stable_id for item in result.raw.graph.state_transitions}
    normalized_transitions = {
        item.stable_id for item in result.normalized.graph.state_transitions
    }
    if raw_transitions != normalized_transitions:
        issues.append(
            BundleIssue(
                code="state-transition-changed",
                path="state_transitions",
                message=f"状态转移集合变化: {sorted(raw_transitions ^ normalized_transitions)}",
            )
        )
    for transition in raw_transitions:
        if not any(
            edge.state_transition_ref == transition
            for edge in result.normalized.graph.edges
        ):
            issues.append(
                BundleIssue(
                    code="state-edge-lost",
                    path=f"state_transitions[{transition}]",
                    message="化简后没有 state 边引用该状态转移（缓存副作用被删除）",
                )
            )

    # dtype 不被改变
    raw_tensors = {tensor.stable_id: tensor for tensor in result.raw.graph.tensors}
    for tensor in result.normalized.graph.tensors:
        original = raw_tensors.get(tensor.stable_id)
        if original is not None and original.dtype != tensor.dtype:
            issues.append(
                BundleIssue(
                    code="dtype-changed",
                    path=f"tensors[{tensor.stable_id}]",
                    message=f"dtype 被改变: {original.dtype} → {tensor.dtype}",
                )
            )

    # anchor 节点间数据/状态可达关系不变
    raw_adjacency = _node_adjacency(result.raw)
    normalized_adjacency = _node_adjacency(result.normalized)
    anchors = [
        node.stable_id
        for node in result.raw.graph.nodes
        if node.stable_id in normalized_nodes
    ]
    for source in anchors:
        raw_reach = _reachable(raw_adjacency, source) & set(anchors)
        normalized_reach = _reachable(normalized_adjacency, source) & set(anchors)
        missing = raw_reach - normalized_reach
        added = normalized_reach - raw_reach
        if missing or added:
            issues.append(
                BundleIssue(
                    code="reachability-changed",
                    path=f"nodes[{source}]",
                    message=(
                        "anchor 可达关系变化：缺失 "
                        f"{sorted(missing)}，新增 {sorted(added)}"
                    ),
                )
            )
    return tuple(issues)
