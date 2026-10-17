"""语义图引用与连接检查器（S1 01b 首步）——只读、确定性。

本文件当前实际实现（仅以下内容）：
- GraphIssue frozen dataclass：稳定 code、精确 path（如 edges[3].src_port）、
  易懂 message。
- validate_graph_references(graph: SemanticGraph) -> tuple[GraphIssue, ...]：
  基于 SemanticGraph 实体核验引用与连接不变量，收集全部问题后按
  (path, code, message) 固定顺序返回——不因第一个问题退出。有效图返回
  空元组。不执行模型源码、不加载权重、不求值任何表达式。

已覆盖不变量（V1–V5）：
- V1 集合唯一性：十类实体 stable_id 不重复；GuardActivation
  (guard_id, scenario_id) 组合键不重复（直接构造的图也报，不等 JSON 层）。
- V2 张量轴引用：Tensor.axes 每个轴按 stable_id 与完整轴定义核对必须存在于
  named_axes（同数值不同 symbol 不合并）；Node 端口的 tensor_id 必须存在于
  tensors。
- V3 边连接：全部四类 kind 的 Edge（含 state）核对 src/dst 节点存在；按方向
  匹配 output/input 端口（input/output 同名不合并）；两端端口的 tensor_id 与
  Edge.tensor_id 三方一致；该 tensor 存在。缺节点、缺端口、方向不符、张量
  不一致分别具名报出；所需方向缺失而反方向同名端口存在时报方向不符，该端口
  tensor 不参与三方比较。
- V4 状态版本：StateTransition 的 before/after 版本引用存在且属于该
  transition 的 state_id；有向版本更新图（节点=version，边=transition）
  无环（仅状态版本有向图，不触及 data 边 DAG，不误判合法循环）；
  state 类 Edge 的 state_transition_ref 必须指向已定义 StateTransition。
- V5 悬空引用：Edge.guard_ref 与 GuardActivation.guard_id 指向已定义 Guard；
  AliasRelation 的 base/alias tensor 指向已定义 Tensor（view/slice 的
  transform_expr 只验引用，绝不作为 Python 执行）。

未实现，本文件不声称：折叠前后连通性等价、跨组边界冲突、模板实例绑定、
完整来源证据等级、场景表达式求值、缓存容量/有效长度、真实 DeepSeek 提取
（留待后续小单）；端口/边 data 相等不证明源码语义或 shape 数值兼容；
01b 未完成，S1 不批准。
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any

from structure_reader.graph_document import SemanticGraph
from structure_reader.graph_ir import (
    _EDGE_KIND_STATE,
    Edge,
    GuardActivation,
    NamedAxis,
    Port,
    StateTransition,
    Tensor,
)


@dataclass(frozen=True)
class GraphIssue:
    """一条检查器发现：稳定代码、精确路径、易懂信息。"""

    code: str
    path: str
    message: str


def _issue(issues: list[GraphIssue], code: str, path: str, message: str) -> None:
    issues.append(GraphIssue(code=code, path=path, message=message))


def _canonical_json(entity: Any) -> str:
    """实体 canonical JSON 字节：重复 ID 歧义代表的确定选择依据。

    用规范化序列化（键排序）+ canonical 轴/端口/证据顺序外的原样字段。
    注意：轴/端口/证据的**列表顺序不改**（任务单要求不因去重改变内部顺序），
    但键序与浮点格式是确定的，故同一多重集合无论录入顺序如何，同一实体的
    canonical 字节相同，最小者唯一。
    """
    return json.dumps(
        entity.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _check_unique_ids(
    graph: SemanticGraph, issues: list[GraphIssue]
) -> dict[str, dict[str, Any]]:
    """V1：各集合 stable_id 唯一；GuardActivation 组合键唯一。

    返回各实体的查找表：{"axis": {stable_id: axis}, "tensor": …,
    "node": {stable_id: node}, "edge": …, "state_version": …,
    "state_transition": …, "guard": …}；"node_ports" 按方向分表：
    {node_id: {"input": {port_name: Port}, "output": {port_name: Port}}}，
    input/output 同名不合并（Node 构造已保证同一方向内端口名唯一）。
    重复 stable_id 时不先到先得：以 canonical JSON 字节最小者为代表
    （V1 照报，图仍无效），保证同一多重集合不同录入顺序下代表唯一。
    """
    registries: dict[str, dict[str, Any]] = {}
    spec = (
        ("named_axes", "axis", lambda x: x.stable_id),
        ("tensors", "tensor", lambda x: x.stable_id),
        ("nodes", "node", lambda x: x.stable_id),
        ("edges", "edge", lambda x: x.stable_id),
        ("state_versions", "state_version", lambda x: x.stable_id),
        ("state_transitions", "state_transition", lambda x: x.stable_id),
        ("guards", "guard", lambda x: x.stable_id),
        ("alias_relations", "alias", lambda x: x.stable_id),
        ("fold_groups", "fold_group", lambda x: x.stable_id),
    )
    for collection, key_name, id_of in spec:
        registry: dict[str, Any] = {}
        canonical: dict[str, str] = {}
        for item in getattr(graph, collection):
            stable_id = id_of(item)
            if stable_id in registry:
                _issue(
                    issues,
                    "V1_DUPLICATE_ID",
                    f"{collection}[{stable_id}]",
                    f"stable_id 重复: {stable_id}",
                )
                existing_canonical = canonical[stable_id]
                candidate_canonical = _canonical_json(item)
                if candidate_canonical < existing_canonical:
                    registry[stable_id] = item
                    canonical[stable_id] = candidate_canonical
            else:
                registry[stable_id] = item
                canonical[stable_id] = _canonical_json(item)
        registries[key_name] = registry
    activations: dict[tuple[str, str], GuardActivation] = {}
    for ga in graph.guard_activations:
        key = (ga.guard_id, ga.scenario_id)
        if key in activations:
            _issue(
                issues,
                "V1_DUPLICATE_ACTIVATION",
                f"guard_activations[{key}]",
                f"(guard_id, scenario_id) 组合键重复: {key}",
            )
        else:
            activations[key] = ga
    registries["activation"] = activations
    node_ports: dict[str, dict[str, dict[str, Port]]] = {}
    for node_id, node in registries["node"].items():
        ports: dict[str, dict[str, Port]] = {"input": {}, "output": {}}
        for port in node.input_ports:
            ports["input"].setdefault(port.name, port)
        for port in node.output_ports:
            ports["output"].setdefault(port.name, port)
        node_ports[node_id] = ports
    registries["node_ports"] = node_ports
    return registries


def _check_tensor_axes_and_port_refs(
    graph: SemanticGraph, registries: dict[str, dict[str, Any]], issues: list[GraphIssue]
) -> None:
    """V2：张量轴存在于 named_axes（完整轴定义核对）；端口 tensor_id 存在。"""
    axis_registry: dict[str, NamedAxis] = registries["axis"]
    tensor_ids = set(registries["tensor"])
    for tensor in graph.tensors:
        for a_index, axis in enumerate(tensor.axes):
            known = axis_registry.get(axis.stable_id)
            if known is None:
                _issue(
                    issues,
                    "V2_AXIS_NOT_DEFINED",
                    f"tensors[{tensor.stable_id}].axes[{a_index}]",
                    f"轴 {axis.stable_id} 不在 named_axes 中",
                )
            elif known is not axis:
                if known != axis:
                    _issue(
                        issues,
                        "V2_AXIS_DEFINITION_MISMATCH",
                        f"tensors[{tensor.stable_id}].axes[{a_index}]",
                        f"轴 {axis.stable_id} 的完整定义与 named_axes 中的同名轴不一致"
                        "（同 ID 必须同定义）",
                    )
    for node in graph.nodes:
        for direction, ports in (
            ("input_ports", node.input_ports),
            ("output_ports", node.output_ports),
        ):
            for p_index, port in enumerate(ports):
                if port.tensor_id not in tensor_ids:
                    _issue(
                        issues,
                        "V2_PORT_TENSOR_NOT_DEFINED",
                        f"nodes[{node.stable_id}].{direction}[{p_index}].tensor_id",
                        f"端口 {port.name!r} 引用的张量 {port.tensor_id} 不在 tensors 中",
                    )


def _check_edges(
    graph: SemanticGraph, registries: dict[str, dict[str, Any]], issues: list[GraphIssue]
) -> None:
    """V3：边端点存在、方向正确、三方 tensor_id 一致、张量存在。

    全部四类 kind（data/state/select/control）一律核对节点、端口与 tensor，
    state 边不豁免——其 transition 引用由 V4 另行核验，合法 transition
    引用不得掩盖幽灵端点。端口按方向匹配：src 只取 output、dst 只取
    input，input/output 同名不合并；所需方向缺失而反方向同名端口存在时
    具名报方向不符，该反方向端口的 tensor 不参与三方比较（不制造伪
    tensor 不一致）。
    path 按实体 stable_id 定位（如 edges[edge:ok].src_port），不随录入索引变化。
    """
    nodes = registries["node"]
    node_ports = registries["node_ports"]
    tensor_ids = set(registries["tensor"])
    for edge in graph.edges:
        base = f"edges[{edge.stable_id}]"
        src_port_obj = None
        if edge.src_node_id not in nodes:
            _issue(
                issues,
                "V3_SRC_NODE_NOT_DEFINED",
                f"{base}.src_node_id",
                f"生产节点 {edge.src_node_id!r} 不在 nodes 中",
            )
        else:
            direction_ports = node_ports[edge.src_node_id]
            src_port_obj = direction_ports["output"].get(edge.src_port)
            if src_port_obj is None:
                opposite = direction_ports["input"].get(edge.src_port)
                if opposite is not None:
                    _issue(
                        issues,
                        "V3_SRC_DIRECTION_MISMATCH",
                        f"{base}.src_port",
                        f"{edge.src_node_id!r}.{edge.src_port!r} 不是 output 端口"
                        f"（direction={opposite.direction!r}）",
                    )
                else:
                    _issue(
                        issues,
                        "V3_SRC_PORT_NOT_FOUND",
                        f"{base}.src_port",
                        f"生产节点 {edge.src_node_id!r} 没有名为 {edge.src_port!r} 的 output 端口",
                    )
        dst_port_obj = None
        if edge.dst_node_id not in nodes:
            _issue(
                issues,
                "V3_DST_NODE_NOT_DEFINED",
                f"{base}.dst_node_id",
                f"消费节点 {edge.dst_node_id!r} 不在 nodes 中",
            )
        else:
            direction_ports = node_ports[edge.dst_node_id]
            dst_port_obj = direction_ports["input"].get(edge.dst_port)
            if dst_port_obj is None:
                opposite = direction_ports["output"].get(edge.dst_port)
                if opposite is not None:
                    _issue(
                        issues,
                        "V3_DST_DIRECTION_MISMATCH",
                        f"{base}.dst_port",
                        f"{edge.dst_node_id!r}.{edge.dst_port!r} 不是 input 端口"
                        f"（direction={opposite.direction!r}）",
                    )
                else:
                    _issue(
                        issues,
                        "V3_DST_PORT_NOT_FOUND",
                        f"{base}.dst_port",
                        f"消费节点 {edge.dst_node_id!r} 没有名为 {edge.dst_port!r} 的 input 端口",
                    )
        if edge.tensor_id not in tensor_ids:
            _issue(
                issues,
                "V3_TENSOR_NOT_DEFINED",
                f"{base}.tensor_id",
                f"边引用的张量 {edge.tensor_id!r} 不在 tensors 中",
            )
        # 三方比较只用匹配到所需方向的真实端口；所需方向缺失时该端不参与，
        # 反方向同名端口的 tensor 绝不代入。
        participants = {
            "edge": edge.tensor_id,
            "src_port": src_port_obj.tensor_id if src_port_obj is not None else None,
            "dst_port": dst_port_obj.tensor_id if dst_port_obj is not None else None,
        }
        known = {k: v for k, v in participants.items() if v is not None}
        if len(set(known.values())) > 1:
            detail = ", ".join(f"{k}={v!r}" for k, v in sorted(known.items()))
            _issue(
                issues,
                "V3_TENSOR_INCONSISTENT",
                f"{base}.tensor_id",
                f"边与两端端口的张量引用不一致（{detail}）",
            )


def _check_state_transitions(
    graph: SemanticGraph, registries: dict[str, dict[str, Any]], issues: list[GraphIssue]
) -> None:
    """V4：版本引用存在且同 state_id；版本有向图无环；state 边引用存在。"""
    versions = registries["state_version"]
    transition_ids = set(registries["state_transition"])
    for transition in graph.state_transitions:
        base = f"state_transitions[{transition.stable_id}]"
        before = versions.get(transition.before_version_id)
        if before is None:
            _issue(
                issues,
                "V4_VERSION_NOT_DEFINED",
                f"{base}.before_version_id",
                f"版本 {transition.before_version_id!r} 不在 state_versions 中",
            )
        elif before.state_id != transition.state_id:
            _issue(
                issues,
                "V4_VERSION_STATE_MISMATCH",
                f"{base}.before_version_id",
                f"版本 {transition.before_version_id!r} 属于状态 "
                f"{before.state_id!r}，与 transition 的 {transition.state_id!r} 不一致",
            )
        after = versions.get(transition.after_version_id)
        if after is None:
            _issue(
                issues,
                "V4_VERSION_NOT_DEFINED",
                f"{base}.after_version_id",
                f"版本 {transition.after_version_id!r} 不在 state_versions 中",
            )
        elif after.state_id != transition.state_id:
            _issue(
                issues,
                "V4_VERSION_STATE_MISMATCH",
                f"{base}.after_version_id",
                f"版本 {transition.after_version_id!r} 属于状态 "
                f"{after.state_id!r}，与 transition 的 {transition.state_id!r} 不一致",
            )
    for edge in graph.edges:
        if edge.kind == _EDGE_KIND_STATE and edge.state_transition_ref is not None:
            if edge.state_transition_ref not in transition_ids:
                _issue(
                    issues,
                    "V4_STATE_EDGE_REF_NOT_DEFINED",
                    f"edges[{edge.stable_id}].state_transition_ref",
                    f"state 边引用的更新 {edge.state_transition_ref!r} "
                    "不在 state_transitions 中",
                )
    # 有向版本更新图无环：节点=version stable_id，边=transition（before→after）。
    # 仅在两端版本都已定义且与 transition 同 state_id 时计边，避免悬空引用噪声。
    # 诊断确定性：邻接表按目标 id 排序、Tarjan 迭代式强连通分量（SCC）——
    # 录入顺序不影响 SCC 划分；只有 size>1 的 SCC（或自环）才是环，
    # 通往环的非环入口路径不进入任何 SCC，绝不误报。环消息以 SCC 内
    # stable_id 升序最小者为起点、沿升序后继走完一圈，跨录入顺序完全一致。
    adjacency: dict[str, list[str]] = {}
    for transition in graph.state_transitions:
        if (
            transition.before_version_id in versions
            and transition.after_version_id in versions
            and versions[transition.before_version_id].state_id == transition.state_id
            and versions[transition.after_version_id].state_id == transition.state_id
        ):
            adjacency.setdefault(transition.before_version_id, []).append(
                transition.after_version_id
            )
    for successors in adjacency.values():
        successors.sort()

    # 迭代式 Tarjan（免递归深度限制）；adjacency.get 覆盖合法链的 sink 版本
    # （after 末端无出边，不在 adjacency 字典中）。
    index_counter = 0
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    sccs: list[list[str]] = []
    for root in sorted(adjacency):
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            node, succ_i = work[-1]
            if succ_i == 0:
                index[node] = lowlink[node] = index_counter
                index_counter += 1
                stack.append(node)
                on_stack.add(node)
            descended = False
            successors = adjacency.get(node, ())
            for i in range(succ_i, len(successors)):
                nxt = successors[i]
                if nxt not in index:
                    work[-1] = (node, i + 1)
                    work.append((nxt, 0))
                    descended = True
                    break
                if nxt in on_stack:
                    lowlink[node] = min(lowlink[node], index[nxt])
            if descended:
                continue
            if lowlink[node] == index[node]:
                scc: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    scc.append(member)
                    if member == node:
                        break
                sccs.append(sorted(scc))
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    for scc in sorted(sccs):
        is_self_loop = len(scc) == 1 and scc[0] in adjacency.get(scc[0], ())
        if len(scc) == 1 and not is_self_loop:
            continue
        start = scc[0]
        if len(scc) == 1:
            walk = [start]
        else:
            # 在 SCC 内从 start 做 BFS（后继已排序、每节点至多入队一次——有界，
            # 必然终止），构造一条从 start 回到 start 的真实环；SCC 强连通保证
            # 可达。绝不把 SCC 成员按任意顺序画成路径；防御分支（理论不可达）
            # 输出成员集合而非伪路径。
            scc_set = set(scc)
            parent: dict[str, str] = {}
            queue = deque([start])
            enqueued = {start}
            found = False
            while queue and not found:
                current = queue.popleft()
                for nxt in adjacency.get(current, ()):
                    if nxt == start:
                        parent[nxt] = current
                        found = True
                        break
                    if nxt in scc_set and nxt not in enqueued:
                        enqueued.add(nxt)
                        parent[nxt] = current
                        queue.append(nxt)
            if found:
                # parent 指向 BFS 来路：从 parent[start] 沿 parent 链回溯到
                # start 的前一节点，反接后即 start→…→start 的真实环
                # （每条 v_i -> v_{i+1} 都是图中真实存在的 transition 边）。
                tail: list[str] = []
                back = parent[start]
                while back != start:
                    tail.append(back)
                    back = parent[back]
                tail.reverse()
                walk = [start, *tail]
            else:
                members = ", ".join(scc)
                _issue(
                    issues,
                    "V4_VERSION_CYCLE",
                    "state_transitions",
                    f"状态版本有向更新存在环（强连通分量成员: {members}）",
                )
                continue
        _issue(
            issues,
            "V4_VERSION_CYCLE",
            "state_transitions",
            "状态版本有向更新存在环: " + " -> ".join(walk + [start]),
        )


def _check_dangling_refs(
    graph: SemanticGraph, registries: dict[str, dict[str, Any]], issues: list[GraphIssue]
) -> None:
    """V5：guard/alias 悬空引用。"""
    guard_ids = set(registries["guard"])
    tensor_ids = set(registries["tensor"])
    for edge in graph.edges:
        if edge.guard_ref is not None and edge.guard_ref not in guard_ids:
            _issue(
                issues,
                "V5_GUARD_REF_NOT_DEFINED",
                f"edges[{edge.stable_id}].guard_ref",
                f"边引用的 guard {edge.guard_ref!r} 不在 guards 中",
            )
    for activation in graph.guard_activations:
        if activation.guard_id not in guard_ids:
            _issue(
                issues,
                "V5_GUARD_REF_NOT_DEFINED",
                f"guard_activations[({activation.guard_id}, {activation.scenario_id})].guard_id",
                f"激活引用的 guard {activation.guard_id!r} 不在 guards 中",
            )
    for alias in graph.alias_relations:
        if alias.base_tensor_id not in tensor_ids:
            _issue(
                issues,
                "V5_ALIAS_TENSOR_NOT_DEFINED",
                f"alias_relations[{alias.stable_id}].base_tensor_id",
                f"别名关系的 base 张量 {alias.base_tensor_id!r} 不在 tensors 中",
            )
        if alias.alias_tensor_id not in tensor_ids:
            _issue(
                issues,
                "V5_ALIAS_TENSOR_NOT_DEFINED",
                f"alias_relations[{alias.stable_id}].alias_tensor_id",
                f"别名关系的 alias 张量 {alias.alias_tensor_id!r} 不在 tensors 中",
            )


def validate_graph_references(graph: SemanticGraph) -> tuple[GraphIssue, ...]:
    """核验语义图引用与连接不变量，收集全部问题后按固定顺序返回。

    有效图返回空元组；返回顺序按 (path, code, message) 升序——确定性，
    不因发现顺序或录入顺序改变。只读：不修改图，不执行模型源码，不加载
    权重，不求值表达式（view/slice 的 transform_expr 仅按引用核验，不执行）。
    """
    issues: list[GraphIssue] = []
    registries = _check_unique_ids(graph, issues)
    _check_tensor_axes_and_port_refs(graph, registries, issues)
    _check_edges(graph, registries, issues)
    _check_state_transitions(graph, registries, issues)
    _check_dangling_refs(graph, registries, issues)
    return tuple(sorted(issues, key=lambda i: (i.path, i.code, i.message)))
