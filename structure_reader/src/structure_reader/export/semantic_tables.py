"""语义图同源 CSV 导出（工作包 C）：connections.csv 与 tensor_axes.csv。

数据完全从 SemanticGraph 实体派生（同源，不手写结构事实）：
- connections.csv：每条边一行——边/源/目标节点与端口、张量 ID（联查键）、
  kind、guard 引用与该场景激活、state transition 引用、证据引用。
- tensor_axes.csv：每个张量的每条轴一行——张量/轴 symbol、轴名/含义/来源/
  证据、value/expression/unknown_reason、dtype、logical/physical 布局位置。
ID 与 semantic_graph.json 一致，可联查。
"""

from __future__ import annotations

import csv
import io

from structure_reader.graph_document import SemanticGraph

__all__ = ["export_connections_csv", "export_tensor_axes_csv"]


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def export_connections_csv(graph: SemanticGraph) -> bytes:
    """全部边的同源导出；ID 与图 JSON 一致。"""
    guard_defs = {g.stable_id: g.predicate_expr for g in graph.guards}
    header = [
        "edge_id", "kind", "src_node", "src_port", "dst_node", "dst_port",
        "tensor_id", "guard_ref", "guard_predicate",
        "state_transition_ref", "evidence_refs",
    ]
    rows = []
    for e in sorted(graph.edges, key=lambda x: x.stable_id):
        rows.append([
            e.stable_id, e.kind, e.src_node_id, e.src_port,
            e.dst_node_id, e.dst_port, e.tensor_id,
            e.guard_ref or "", guard_defs.get(e.guard_ref, "") if e.guard_ref else "",
            e.state_transition_ref or "",
            "; ".join(e.evidence_refs),
        ])
    return _csv_bytes(header, rows)


def export_tensor_axes_csv(graph: SemanticGraph) -> bytes:
    """每个张量的每条轴一行（同源派生；轴定义与 named_axes 一致）。"""
    header = [
        "tensor_id", "dtype", "axis_index", "axis_symbol", "axis_name",
        "axis_meaning", "axis_origin", "axis_evidence_ref", "value",
        "expression", "unknown_reason", "logical_position", "physical_position",
    ]
    rows = []
    for t in sorted(graph.tensors, key=lambda x: x.stable_id):
        symbols = [a.symbol for a in t.axes]
        for i, axis in enumerate(t.axes):
            phys = t.physical_layout.index(axis.symbol)
            rows.append([
                t.stable_id, t.dtype, str(i), axis.symbol, axis.name,
                axis.meaning, axis.origin, axis.evidence_ref,
                "" if axis.value is None else str(axis.value),
                axis.expression or "", axis.unknown_reason or "",
                str(i + 1), str(phys + 1),
            ])
    return _csv_bytes(header, rows)
