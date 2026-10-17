"""S1 01b 首步引用检查器独立验证：V1–V5 引用/连接不变量、确定性与只读性。

任务单：`outputs/_coordination/s1/2026_10_12_round_01/dispatch/verification_01b_graph_reference_checker.md`
契约依据：`dispatch/development_01b_graph_reference_checker.md`（不是把当前产品行为当正确答案），
缺陷依据：`evidence/codex_01b_initial_defect_probe.json` + `audit/retry_2026_10_13_01b.md`。

被测接口（开发报告声明的实际调用法）：
    from structure_reader.checks.graph_contracts import GraphIssue, validate_graph_references
    issues: tuple[GraphIssue, ...] = validate_graph_references(graph)

本测试独立构造**合成最小图**（synthetic contract fixture，非 DeepSeek 结构），
期望值来自任务单声明的不变量，不取自产品输出，不复制产品输出为 golden。

覆盖：
- 有效图（含合法 state 边）返回空 tuple；
- V1 各集合 stable_id 唯一、GuardActivation 组合键唯一（直接构造的图也须报）；
- V2 轴按 stable_id + 完整定义核对、同数值异 symbol 不合并、端口 tensor 存在；
- V3 全部四种 kind 的边：端点节点/端口/方向/三方 tensor 一致；
- V4 版本存在/同 state/版本环/state 边 transition 引用；合法 data 环不误报；
- V5 guard/alias 悬空引用；transform_expr/predicate_expr 不被执行；
- 多错误一次收集、返回按 (path, code, message) 固定排序、只读（不改图）、可重复。

针对 Codex 三项实测缺陷的独立复现（本文件断言契约，而非当前实现）：
1. state 边不得被 V3 静默豁免——幽灵 src/dst 节点/端口/tensor 必须具名报出，
   即使 state_transition_ref 合法；合法 state 边仍为空结果。
2. Node input/output 跨方向同名不得按名字 input 优先合并——合法 output 不得被
   误判为 input，也不得因此得出错误的 tensor 不一致；所需方向缺失而反方向同名时
   须具名报告方向问题，且不得用反方向端口参与三方 tensor 比较。
3. 状态版本环诊断须在 transition/版本录入次序变化时完全稳定；重复 ID 且定义不同的
   无效图其 issue 序列也须与录入顺序无关。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from structure_reader.checks.graph_contracts import (  # noqa: E402
    GraphIssue,
    validate_graph_references,
)
from structure_reader.graph_document import SemanticGraph  # noqa: E402
from structure_reader.graph_ir import (  # noqa: E402
    AliasRelation,
    Edge,
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

# ---------------------------------------------------------------------------
# 合成构造器
# ---------------------------------------------------------------------------


def axis(symbol: str, value: int = 4096, meaning: str = "hidden") -> NamedAxis:
    return NamedAxis(
        symbol=symbol,
        name=f"{symbol} 轴",
        meaning=meaning,
        origin="synthetic",
        evidence_ref="synthetic:1",
        value=value,
    )


def unknown_axis(symbol: str) -> NamedAxis:
    return NamedAxis(
        symbol=symbol,
        name=f"{symbol} 轴",
        meaning="size",
        origin="synthetic",
        evidence_ref="synthetic:1",
        unknown_reason="合成未知轴",
    )


H = axis("H", 4096)
D = axis("D", 512)


def tensor(name: str, axes=(H, D)) -> Tensor:
    symbols = [a.symbol for a in axes]
    return Tensor(
        name=name,
        dtype="float32",
        axes=list(axes),
        logical_layout=list(symbols),
        physical_layout=list(symbols),
    )


def node(nid: str, inputs=(), outputs=()) -> Node:
    return Node(
        stable_id=nid,
        op_type="op",
        granularity="layer",
        input_ports=[Port(name, "input", tid) for name, tid in inputs],
        output_ports=[Port(name, "output", tid) for name, tid in outputs],
    )


def data_edge(
    eid="e:data",
    src="n:src",
    src_port="out",
    dst="n:dst",
    dst_port="in",
    tid="tensor:a",
    kind="data",
    **kw,
) -> Edge:
    return Edge(
        stable_id=eid,
        src_node_id=src,
        src_port=src_port,
        dst_node_id=dst,
        dst_port=dst_port,
        tensor_id=tid,
        kind=kind,
        **kw,
    )


def minimal_graph(**overrides) -> SemanticGraph:
    """一个无任何违规的最小图：H/D 轴、tensor:a/b、src→dst 的 data 边。"""
    params = dict(
        template_id="tpl:min",
        named_axes=[H, D],
        tensors=[tensor("a"), tensor("b")],
        nodes=[
            node("n:src", outputs=[("out", "tensor:a")]),
            node("n:dst", inputs=[("in", "tensor:a")]),
        ],
        edges=[data_edge()],
    )
    params.update(overrides)
    return SemanticGraph(**params)


def stateful_graph(**overrides) -> SemanticGraph:
    """在最小图之上加入合法 state 版本/更新与一条合法 state 边。"""
    params = dict(
        template_id="tpl:min",
        named_axes=[H, D],
        tensors=[tensor("a"), tensor("b")],
        nodes=[
            node("n:src", outputs=[("out", "tensor:a")]),
            node("n:dst", inputs=[("in", "tensor:a")]),
        ],
        edges=[
            data_edge(),
            Edge(
                stable_id="e:state",
                src_node_id="n:src",
                src_port="out",
                dst_node_id="n:dst",
                dst_port="in",
                tensor_id="tensor:a",
                kind="state",
                state_transition_ref="st:0",
            ),
        ],
        state_versions=[
            StateVersion(stable_id="sv:0", state_id="s:k", version_label="v0"),
            StateVersion(stable_id="sv:1", state_id="s:k", version_label="v1"),
        ],
        state_transitions=[
            StateTransition(
                stable_id="st:0",
                state_id="s:k",
                before_version_id="sv:0",
                after_version_id="sv:1",
            )
        ],
    )
    params.update(overrides)
    return SemanticGraph(**params)


def pairs(issues) -> list[tuple[str, str]]:
    return [(issue.code, issue.path) for issue in issues]


def codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def signatures(issues) -> list[tuple[str, str, str]]:
    """完整 issue 签名 (code, path, message)，用于跨录入顺序稳定性比较。"""
    return [(issue.code, issue.path, issue.message) for issue in issues]


def fingerprint(graph: SemanticGraph) -> str:
    """产品只读断言：把图全部实体序列化为规范化文本指纹。"""
    payload = {
        name: [item.to_dict() for item in getattr(graph, name)]
        for name in (
            "named_axes",
            "tensors",
            "nodes",
            "edges",
            "state_versions",
            "state_transitions",
            "guards",
            "guard_activations",
            "alias_relations",
            "fold_groups",
        )
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# 有效图与返回契约
# ---------------------------------------------------------------------------


class ValidGraphAndContractTest(unittest.TestCase):
    def test_valid_graph_returns_empty_tuple(self):
        result = validate_graph_references(minimal_graph())
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, ())

    def test_valid_stateful_graph_returns_empty_tuple(self):
        """合法 state 边（端点/方向/tensor 一致且 transition 引用合法）不得误报。"""
        self.assertEqual(validate_graph_references(stateful_graph()), ())

    def test_returns_graph_issue_instances_with_stable_fields(self):
        issues = validate_graph_references(
            minimal_graph(edges=[data_edge(dst="n:ghost")])
        )
        self.assertTrue(issues)
        for issue in issues:
            self.assertIsInstance(issue, GraphIssue)
            self.assertIsInstance(issue.code, str)
            self.assertIsInstance(issue.path, str)
            self.assertIsInstance(issue.message, str)
            self.assertTrue(issue.code.strip())
            self.assertTrue(issue.path.strip())
            self.assertTrue(issue.message.strip())

    def test_result_sorted_by_path_code_message(self):
        issues = validate_graph_references(multi_violation_graph())
        self.assertEqual(
            list(issues),
            sorted(issues, key=lambda i: (i.path, i.code, i.message)),
        )

    def test_multiple_issues_collected_in_one_call(self):
        issues = validate_graph_references(multi_violation_graph())
        expected = {
            ("V2_AXIS_NOT_DEFINED", "tensors[tensor:u].axes[0]"),
            ("V3_DST_NODE_NOT_DEFINED", "edges[e2].dst_node_id"),
            ("V3_TENSOR_INCONSISTENT", "edges[e1].tensor_id"),
            ("V5_GUARD_REF_NOT_DEFINED", "edges[e1].guard_ref"),
            ("V5_ALIAS_TENSOR_NOT_DEFINED", "alias_relations[al].base_tensor_id"),
        }
        got = set(pairs(issues))
        self.assertTrue(
            expected <= got, f"缺少问题: {sorted(expected - got)}，实际: {sorted(got)}"
        )
        self.assertGreaterEqual(len(issues), len(expected))

    def test_checker_is_read_only(self):
        graph = multi_violation_graph()
        before = fingerprint(graph)
        validate_graph_references(graph)
        self.assertEqual(before, fingerprint(graph))

    def test_repeated_calls_deterministic(self):
        graph = multi_violation_graph()
        self.assertEqual(
            validate_graph_references(graph), validate_graph_references(graph)
        )


def multi_violation_graph() -> SemanticGraph:
    """一个包含多个彼此独立问题的非环图（用于收集/排序/只读断言）。"""
    return minimal_graph(
        named_axes=[H, D],
        tensors=[tensor("a"), tensor("b"), tensor("u", axes=(unknown_axis("H2"), D))],
        nodes=[
            node("n:src", outputs=[("out", "tensor:a")]),
            node("n:dst", inputs=[("in", "tensor:b")]),
        ],
        edges=[
            data_edge("e1", tid="tensor:a", guard_ref="g:ghost"),
            data_edge("e2", dst="n:ghost"),
        ],
        guard_activations=[
            GuardActivation(guard_id="g:ghost2", scenario_id="s1", active_for_scenario="true")
        ],
        alias_relations=[
            AliasRelation(
                stable_id="al",
                base_tensor_id="tensor:ghost",
                alias_tensor_id="tensor:a",
                relation_kind="view",
                transform_expr="a[0]",
            )
        ],
    )


# ---------------------------------------------------------------------------
# V1 唯一性
# ---------------------------------------------------------------------------


class V1UniquenessTest(unittest.TestCase):
    def test_duplicate_ids_reported_per_collection(self):
        cases = {
            "named_axes": (
                {"named_axes": [H, H, D]},
                ("V1_DUPLICATE_ID", "axis:H"),
            ),
            "tensors": (
                {"tensors": [tensor("a"), tensor("a")]},
                ("V1_DUPLICATE_ID", "tensor:a"),
            ),
            "nodes": (
                {"nodes": [node("n:dup", outputs=[("out", "tensor:a")])] * 2
                 + [node("n:dst", inputs=[("in", "tensor:a")])]},
                ("V1_DUPLICATE_ID", "n:dup"),
            ),
            "edges": (
                {"edges": [data_edge("e:dup"), data_edge("e:dup")]},
                ("V1_DUPLICATE_ID", "e:dup"),
            ),
            "state_versions": (
                {"state_versions": [
                    StateVersion("sv:dup", "s:k", "v0"),
                    StateVersion("sv:dup", "s:k", "v1"),
                ]},
                ("V1_DUPLICATE_ID", "sv:dup"),
            ),
            "state_transitions": (
                {"state_transitions": [
                    StateTransition("st:dup", "s:k", "sv:0", "sv:1"),
                    StateTransition("st:dup", "s:k", "sv:1", "sv:0"),
                ],
                 "state_versions": [
                    StateVersion("sv:0", "s:k", "v0"),
                    StateVersion("sv:1", "s:k", "v1"),
                 ]},
                ("V1_DUPLICATE_ID", "st:dup"),
            ),
            "guards": (
                {"guards": [Guard("g:dup", "x > 0"), Guard("g:dup", "x < 0")]},
                ("V1_DUPLICATE_ID", "g:dup"),
            ),
            "alias_relations": (
                {"alias_relations": [
                    AliasRelation("al:dup", "tensor:a", "tensor:b", "view",
                                  transform_expr="a[0]"),
                    AliasRelation("al:dup", "tensor:a", "tensor:b", "view",
                                  transform_expr="a[1]"),
                ]},
                ("V1_DUPLICATE_ID", "al:dup"),
            ),
            "fold_groups": (
                {"fold_groups": [FoldGroup("fg:dup", [], []), FoldGroup("fg:dup", [], [])]},
                ("V1_DUPLICATE_ID", "fg:dup"),
            ),
        }
        for name, (overrides, expected) in cases.items():
            with self.subTest(collection=name):
                issues = validate_graph_references(minimal_graph(**overrides))
                matching = [
                    (c, p) for c, p in pairs(issues) if c == expected[0]
                ]
                self.assertTrue(
                    matching,
                    f"{name}: 未报告 {expected[0]}，实际: {pairs(issues)}",
                )
                self.assertTrue(
                    any(expected[1] in p for _, p in matching),
                    f"{name}: path 未定位到 {expected[1]}，实际: {matching}",
                )

    def test_duplicate_activation_key_reported(self):
        graph = minimal_graph(
            guards=[Guard("g:1", "x > 0")],
            guard_activations=[
                GuardActivation("g:1", "s1", "true"),
                GuardActivation("g:1", "s1", "false"),
            ],
        )
        got = codes(validate_graph_references(graph))
        self.assertIn("V1_DUPLICATE_ACTIVATION", got)

    def test_distinct_activation_keys_not_reported(self):
        graph = minimal_graph(
            guards=[Guard("g:1", "x > 0"), Guard("g:2", "x < 0")],
            guard_activations=[
                GuardActivation("g:1", "s1", "true"),
                GuardActivation("g:1", "s2", "true"),
                GuardActivation("g:2", "s1", "true"),
            ],
        )
        self.assertEqual(validate_graph_references(graph), ())

    def test_collection_order_does_not_change_issues(self):
        """无重复 ID 的图，集合逆序录入不得改变 issue 序列。"""
        forward = multi_violation_graph()
        reverse = minimal_graph(
            named_axes=list(reversed(forward.named_axes)),
            tensors=list(reversed(forward.tensors)),
            nodes=list(reversed(forward.nodes)),
            edges=list(reversed(forward.edges)),
            guards=list(reversed(forward.guards)),
            guard_activations=list(reversed(forward.guard_activations)),
            alias_relations=list(reversed(forward.alias_relations)),
        )
        self.assertEqual(
            validate_graph_references(forward), validate_graph_references(reverse)
        )

    def test_duplicate_id_with_differing_definition_is_order_invariant(self):
        """重复 node ID 且定义不同：issue 序列不得随录入顺序变化（Codex 缺陷 4）。"""

        def build(order: str) -> SemanticGraph:
            dup_x = node("n:dup", outputs=[("x", "tensor:a")])
            dup_y = node("n:dup", outputs=[("y", "tensor:a")])
            dup_nodes = [dup_x, dup_y] if order == "x_first" else [dup_y, dup_x]
            return minimal_graph(
                nodes=dup_nodes + [node("n:dst", inputs=[("in", "tensor:a")])],
                edges=[
                    data_edge(
                        "e:data",
                        src="n:dup",
                        src_port="x",
                        dst="n:dst",
                        dst_port="in",
                        tid="tensor:a",
                    )
                ],
            )

        first = validate_graph_references(build("x_first"))
        second = validate_graph_references(build("y_first"))
        self.assertIn("V1_DUPLICATE_ID", codes(first))
        self.assertIn("V1_DUPLICATE_ID", codes(second))
        self.assertEqual(
            pairs(first), pairs(second),
            "重复 ID 且定义不同时，issue 随录入顺序变化（先到先得的不稳定诊断）",
        )


# ---------------------------------------------------------------------------
# V2 轴与端口张量引用
# ---------------------------------------------------------------------------


class V2AxesAndTensorRefsTest(unittest.TestCase):
    def test_axis_not_in_named_axes_reported(self):
        graph = minimal_graph(
            tensors=[tensor("a", axes=(unknown_axis("H3"), D)), tensor("b")],
        )
        self.assertIn(
            ("V2_AXIS_NOT_DEFINED", "tensors[tensor:a].axes[0]"),
            pairs(validate_graph_references(graph)),
        )

    def test_same_value_different_symbol_not_merged(self):
        """H=4096 与未登记的 H2=4096 数值相同也不同轴，必须报未登记轴。"""
        graph = minimal_graph(
            tensors=[tensor("a", axes=(axis("H2", 4096), D)), tensor("b")],
        )
        got = validate_graph_references(graph)
        self.assertIn(
            ("V2_AXIS_NOT_DEFINED", "tensors[tensor:a].axes[0]"), pairs(got)
        )

    def test_same_stable_id_different_definition_reported(self):
        """同一 symbol（stable_id）但完整轴定义不同必须报定义不一致。"""
        other_h = axis("H", 4096, meaning="另一个含义")
        graph = minimal_graph(
            named_axes=[other_h, D],
            tensors=[tensor("a", axes=(H, D)), tensor("b")],
        )
        self.assertIn(
            ("V2_AXIS_DEFINITION_MISMATCH", "tensors[tensor:a].axes[0]"),
            pairs(validate_graph_references(graph)),
        )

    def test_port_tensor_not_defined_reported_both_directions(self):
        graph = minimal_graph(
            nodes=[
                node("n:src", outputs=[("out", "tensor:ghost")]),
                node("n:dst", inputs=[("in", "tensor:ghost2")]),
            ],
            edges=[],
        )
        got = set(pairs(validate_graph_references(graph)))
        self.assertIn(
            ("V2_PORT_TENSOR_NOT_DEFINED", "nodes[n:src].output_ports[0].tensor_id"),
            got,
        )
        self.assertIn(
            ("V2_PORT_TENSOR_NOT_DEFINED", "nodes[n:dst].input_ports[0].tensor_id"),
            got,
        )

    def test_valid_axes_and_ports_not_reported(self):
        self.assertEqual(validate_graph_references(minimal_graph()), ())


# ---------------------------------------------------------------------------
# V3 边连接（含 state 边不豁免、同名端口方向）
# ---------------------------------------------------------------------------


class V3EdgeConnectivityTest(unittest.TestCase):
    def test_missing_src_node(self):
        graph = minimal_graph(edges=[data_edge(src="n:ghost")])
        self.assertIn(
            ("V3_SRC_NODE_NOT_DEFINED", "edges[e:data].src_node_id"),
            pairs(validate_graph_references(graph)),
        )

    def test_missing_dst_node(self):
        graph = minimal_graph(edges=[data_edge(dst="n:ghost")])
        self.assertIn(
            ("V3_DST_NODE_NOT_DEFINED", "edges[e:data].dst_node_id"),
            pairs(validate_graph_references(graph)),
        )

    def test_missing_src_port(self):
        graph = minimal_graph(edges=[data_edge(src_port="nope")])
        self.assertIn(
            ("V3_SRC_PORT_NOT_FOUND", "edges[e:data].src_port"),
            pairs(validate_graph_references(graph)),
        )

    def test_missing_dst_port(self):
        graph = minimal_graph(edges=[data_edge(dst_port="nope")])
        self.assertIn(
            ("V3_DST_PORT_NOT_FOUND", "edges[e:data].dst_port"),
            pairs(validate_graph_references(graph)),
        )

    def test_src_direction_mismatch(self):
        graph = minimal_graph(
            nodes=[
                node("n:src", inputs=[("p", "tensor:a")]),
                node("n:dst", inputs=[("in", "tensor:a")]),
            ],
            edges=[data_edge(src_port="p")],
        )
        self.assertIn(
            ("V3_SRC_DIRECTION_MISMATCH", "edges[e:data].src_port"),
            pairs(validate_graph_references(graph)),
        )

    def test_dst_direction_mismatch(self):
        graph = minimal_graph(
            nodes=[
                node("n:src", outputs=[("out", "tensor:a")]),
                node("n:dst", inputs=[("other", "tensor:a")], outputs=[("in", "tensor:a")]),
            ],
            edges=[data_edge(dst_port="in")],
        )
        self.assertIn(
            ("V3_DST_DIRECTION_MISMATCH", "edges[e:data].dst_port"),
            pairs(validate_graph_references(graph)),
        )

    def test_edge_tensor_not_defined(self):
        graph = minimal_graph(edges=[data_edge(tid="tensor:ghost")])
        self.assertIn(
            ("V3_TENSOR_NOT_DEFINED", "edges[e:data].tensor_id"),
            pairs(validate_graph_references(graph)),
        )

    def test_three_way_tensor_inconsistent(self):
        graph = minimal_graph(
            nodes=[
                node("n:src", outputs=[("out", "tensor:a")]),
                node("n:dst", inputs=[("in", "tensor:b")]),
            ],
            edges=[data_edge(tid="tensor:a")],
        )
        self.assertIn(
            ("V3_TENSOR_INCONSISTENT", "edges[e:data].tensor_id"),
            pairs(validate_graph_references(graph)),
        )

    def test_non_state_edge_kinds_checked_for_endpoints_ports_tensors(self):
        """data/select/control 三类边都要核对端点、端口与三方 tensor。"""
        for kind in ("data", "select", "control"):
            with self.subTest(kind=kind, case="ghost_node"):
                graph = minimal_graph(edges=[data_edge(kind=kind, dst="n:ghost")])
                self.assertIn(
                    ("V3_DST_NODE_NOT_DEFINED", "edges[e:data].dst_node_id"),
                    pairs(validate_graph_references(graph)),
                )
            with self.subTest(kind=kind, case="ghost_port_tensor"):
                graph = minimal_graph(
                    edges=[data_edge(kind=kind, dst_port="nope", tid="tensor:ghost")]
                )
                got = set(pairs(validate_graph_references(graph)))
                self.assertIn(("V3_DST_PORT_NOT_FOUND", "edges[e:data].dst_port"), got)
                self.assertIn(("V3_TENSOR_NOT_DEFINED", "edges[e:data].tensor_id"), got)
            with self.subTest(kind=kind, case="tensor_inconsistent"):
                graph = minimal_graph(
                    nodes=[
                        node("n:src", outputs=[("out", "tensor:a")]),
                        node("n:dst", inputs=[("in", "tensor:b")]),
                    ],
                    edges=[data_edge(kind=kind, tid="tensor:a")],
                )
                self.assertIn(
                    ("V3_TENSOR_INCONSISTENT", "edges[e:data].tensor_id"),
                    pairs(validate_graph_references(graph)),
                )

    def test_valid_graph_with_all_four_edge_kinds_is_empty(self):
        """四类边都合法（含合法 state 边）时仍须返回空 tuple。"""
        edges = [
            data_edge("e:data", kind="data"),
            data_edge("e:select", kind="select"),
            data_edge("e:control", kind="control"),
            Edge(
                stable_id="e:state",
                src_node_id="n:src",
                src_port="out",
                dst_node_id="n:dst",
                dst_port="in",
                tensor_id="tensor:a",
                kind="state",
                state_transition_ref="st:0",
            ),
        ]
        graph = stateful_graph(edges=edges)
        self.assertEqual(validate_graph_references(graph), ())

    def test_state_edge_ghost_nodes_not_masked_by_valid_transition_ref(self):
        """Codex 缺陷 1：state 边幽灵 src/dst 节点不得被合法 transition 引用掩盖。"""
        graph = stateful_graph(
            edges=[
                Edge(
                    stable_id="e:state",
                    src_node_id="n:missing_src",
                    src_port="ghost_out",
                    dst_node_id="n:missing_dst",
                    dst_port="ghost_in",
                    tensor_id="tensor:ghost",
                    kind="state",
                    state_transition_ref="st:0",
                )
            ]
        )
        got = pairs(validate_graph_references(graph))
        for expected in (
            ("V3_SRC_NODE_NOT_DEFINED", "edges[e:state].src_node_id"),
            ("V3_DST_NODE_NOT_DEFINED", "edges[e:state].dst_node_id"),
            ("V3_TENSOR_NOT_DEFINED", "edges[e:state].tensor_id"),
        ):
            self.assertIn(expected, got, f"state 边漏检: {expected}，实际 {got}")
        self.assertEqual(
            {c for c, _ in got if c.startswith("V4")},
            set(),
            "合法 transition 引用不应产生 V4 问题",
        )

    def test_state_edge_ghost_ports_and_tensor_not_masked(self):
        """Codex 缺陷 1 续：节点存在但 state 边端口/tensor 幽灵仍须报出。"""
        graph = stateful_graph(
            edges=[
                Edge(
                    stable_id="e:state",
                    src_node_id="n:src",
                    src_port="ghost_out",
                    dst_node_id="n:dst",
                    dst_port="ghost_in",
                    tensor_id="tensor:ghost",
                    kind="state",
                    state_transition_ref="st:0",
                )
            ]
        )
        got = pairs(validate_graph_references(graph))
        for expected in (
            ("V3_SRC_PORT_NOT_FOUND", "edges[e:state].src_port"),
            ("V3_DST_PORT_NOT_FOUND", "edges[e:state].dst_port"),
            ("V3_TENSOR_NOT_DEFINED", "edges[e:state].tensor_id"),
        ):
            self.assertIn(expected, got, f"state 边漏检: {expected}，实际 {got}")

    def test_same_name_ports_across_directions_are_legal_both_ends(self):
        """Codex 缺陷 2：合法 output 不得因反方向同名 input 被误判或错 tensor。"""
        graph = minimal_graph(
            nodes=[
                node("n:src", inputs=[("x", "tensor:b")], outputs=[("x", "tensor:a")]),
                node("n:dst", inputs=[("x", "tensor:a")], outputs=[("x", "tensor:b")]),
            ],
            edges=[
                data_edge(
                    "e:data", src="n:src", src_port="x", dst="n:dst", dst_port="x",
                    tid="tensor:a",
                )
            ],
        )
        self.assertEqual(
            validate_graph_references(graph),
            (),
            "src/dst 双向同名端口的合法图被误报",
        )

    def test_same_name_ports_src_side_only(self):
        graph = minimal_graph(
            nodes=[
                node("n:src", inputs=[("x", "tensor:b")], outputs=[("x", "tensor:a")]),
                node("n:dst", inputs=[("in", "tensor:a")]),
            ],
            edges=[
                data_edge("e:data", src="n:src", src_port="x", dst="n:dst",
                          dst_port="in", tid="tensor:a")
            ],
        )
        self.assertEqual(validate_graph_references(graph), ())

    def test_same_name_ports_dst_side_only(self):
        graph = minimal_graph(
            nodes=[
                node("n:src", outputs=[("out", "tensor:a")]),
                node("n:dst", inputs=[("x", "tensor:a")], outputs=[("x", "tensor:b")]),
            ],
            edges=[
                data_edge("e:data", src="n:src", src_port="out", dst="n:dst",
                          dst_port="x", tid="tensor:a")
            ],
        )
        self.assertEqual(validate_graph_references(graph), ())

    def test_required_direction_missing_reports_named_direction_issue(self):
        """所需 input 缺失、只有同方向缺失的反向同名 output：须具名报方向问题，
        且不得把反方向端口的 tensor 当作三方比较参与者。"""
        graph = minimal_graph(
            tensors=[tensor("a"), tensor("b"), tensor("z")],
            nodes=[
                node("n:src", outputs=[("out", "tensor:a")]),
                node("n:dst", inputs=[("other", "tensor:a")], outputs=[("x", "tensor:z")]),
            ],
            edges=[
                data_edge("e:data", src="n:src", src_port="out", dst="n:dst",
                          dst_port="x", tid="tensor:a")
            ],
        )
        got = pairs(validate_graph_references(graph))
        self.assertTrue(
            any(
                c in ("V3_DST_DIRECTION_MISMATCH", "V3_DST_PORT_NOT_FOUND")
                and p == "edges[e:data].dst_port"
                for c, p in got
            ),
            f"未具名报告 dst 端口方向/缺失问题：{got}",
        )
        self.assertNotIn(
            "V3_TENSOR_INCONSISTENT",
            {c for c, _ in got},
            "用反方向同名端口参与了三方 tensor 比较，产生错误的不一致报告",
        )


# ---------------------------------------------------------------------------
# V4 状态版本
# ---------------------------------------------------------------------------


class V4StateVersionsTest(unittest.TestCase):
    def test_before_version_not_defined(self):
        graph = stateful_graph(
            state_transitions=[
                StateTransition("st:0", "s:k", "sv:ghost", "sv:1")
            ]
        )
        self.assertIn(
            ("V4_VERSION_NOT_DEFINED", "state_transitions[st:0].before_version_id"),
            pairs(validate_graph_references(graph)),
        )

    def test_after_version_not_defined(self):
        graph = stateful_graph(
            state_transitions=[
                StateTransition("st:0", "s:k", "sv:0", "sv:ghost")
            ]
        )
        self.assertIn(
            ("V4_VERSION_NOT_DEFINED", "state_transitions[st:0].after_version_id"),
            pairs(validate_graph_references(graph)),
        )

    def test_version_of_other_state_reported(self):
        graph = stateful_graph(
            state_versions=[
                StateVersion("sv:0", "s:other", "v0"),
                StateVersion("sv:1", "s:k", "v1"),
            ],
            state_transitions=[
                StateTransition("st:0", "s:k", "sv:0", "sv:1")
            ],
        )
        self.assertIn(
            ("V4_VERSION_STATE_MISMATCH", "state_transitions[st:0].before_version_id"),
            pairs(validate_graph_references(graph)),
        )

    def test_state_edge_transition_ref_not_defined(self):
        graph = stateful_graph(
            edges=[
                data_edge("e:data"),
                Edge(
                    stable_id="e:state",
                    src_node_id="n:src",
                    src_port="out",
                    dst_node_id="n:dst",
                    dst_port="in",
                    tensor_id="tensor:a",
                    kind="state",
                    state_transition_ref="st:ghost",
                ),
            ]
        )
        self.assertIn(
            ("V4_STATE_EDGE_REF_NOT_DEFINED", "edges[e:state].state_transition_ref"),
            pairs(validate_graph_references(graph)),
        )

    def test_version_cycle_detected(self):
        issues = validate_graph_references(cycle_graph("forward"))
        cycle = [i for i in issues if i.code == "V4_VERSION_CYCLE"]
        self.assertTrue(cycle)
        self.assertTrue(
            all(i.path.startswith("state_transitions") for i in cycle),
            f"版本环 path 未定位到 state_transitions: {pairs(issues)}",
        )
        # 语义校验（非字节稳定）：箭头路径须首尾同一版本、每步都是真实有向边。
        for issue in cycle:
            ok, why = _cycle_message_shape(
                issue.message, SIMPLE_CYCLE_EDGES, SIMPLE_CYCLE_MEMBERS
            )
            self.assertTrue(
                ok, f"环诊断不是真实有限环或完整成员集合（{why}）: {issue.message!r}"
            )

    def test_version_cycle_diagnosis_stable_across_insertion_order(self):
        """Codex 缺陷 3：同一版本环逆序录入后 issue 序列必须完全相等。"""
        forward = validate_graph_references(cycle_graph("forward"))
        reverse = validate_graph_references(cycle_graph("reverse"))
        self.assertIn("V4_VERSION_CYCLE", codes(forward))
        self.assertEqual(
            pairs(forward), pairs(reverse),
            "版本环诊断随 transition 录入顺序变化（DFS 依赖录入顺序）",
        )
        self.assertEqual(
            [i.message for i in forward], [i.message for i in reverse],
            "版本环消息随录入顺序变化",
        )
        # 稳定性之外仍须语义真实：两种录入的箭头路径都要首尾闭合且每步真实。
        for issue in forward + reverse:
            if issue.code != "V4_VERSION_CYCLE":
                continue
            ok, why = _cycle_message_shape(
                issue.message, SIMPLE_CYCLE_EDGES, SIMPLE_CYCLE_MEMBERS
            )
            self.assertTrue(
                ok, f"环诊断不是真实有限环或完整成员集合（{why}）: {issue.message!r}"
            )

    def test_two_disjoint_cycles_both_discoverable(self):
        graph = minimal_graph(
            state_versions=[
                StateVersion("a:0", "s:a", "v0"),
                StateVersion("a:1", "s:a", "v1"),
                StateVersion("b:0", "s:b", "v0"),
                StateVersion("b:1", "s:b", "v1"),
            ],
            state_transitions=[
                StateTransition("t:a0", "s:a", "a:0", "a:1"),
                StateTransition("t:a1", "s:a", "a:1", "a:0"),
                StateTransition("t:b0", "s:b", "b:0", "b:1"),
                StateTransition("t:b1", "s:b", "b:1", "b:0"),
            ],
        )
        issues = validate_graph_references(graph)
        cycle_messages = " ".join(
            i.message for i in issues if i.code == "V4_VERSION_CYCLE"
        )
        self.assertTrue(cycle_messages, "两个独立环均未被发现")
        for version_id in ("a:0", "a:1", "b:0", "b:1"):
            self.assertIn(
                version_id, cycle_messages,
                f"独立环中的版本 {version_id} 未被任何环诊断提及",
            )

    def test_legal_data_cycle_not_reported_as_version_cycle(self):
        graph = minimal_graph(
            nodes=[
                node("n1", inputs=[("i", "tensor:a")], outputs=[("o", "tensor:a")]),
                node("n2", inputs=[("i", "tensor:a")], outputs=[("o", "tensor:a")]),
            ],
            edges=[
                data_edge("e:12", src="n1", src_port="o", dst="n2", dst_port="i"),
                data_edge("e:21", src="n2", src_port="o", dst="n1", dst_port="i"),
            ],
        )
        self.assertNotIn(
            "V4_VERSION_CYCLE",
            codes(validate_graph_references(graph)),
            "data 边 DAG 中的合法循环被误判为状态版本环",
        )
        self.assertEqual(validate_graph_references(graph), ())


SIMPLE_CYCLE_EDGES = {("sv:0", "sv:1"), ("sv:1", "sv:2"), ("sv:2", "sv:0")}
SIMPLE_CYCLE_MEMBERS = {"sv:0", "sv:1", "sv:2"}


def cycle_graph(order: str) -> SemanticGraph:
    versions = [
        StateVersion("sv:0", "s:k", "v0"),
        StateVersion("sv:1", "s:k", "v1"),
        StateVersion("sv:2", "s:k", "v2"),
    ]
    forward = [
        StateTransition("st:0", "s:k", "sv:0", "sv:1"),
        StateTransition("st:1", "s:k", "sv:1", "sv:2"),
        StateTransition("st:2", "s:k", "sv:2", "sv:0"),
    ]
    transitions = forward if order == "forward" else list(reversed(forward))
    return minimal_graph(state_versions=versions, state_transitions=transitions)


# ---------------------------------------------------------------------------
# V5 悬空引用与不执行表达式
# ---------------------------------------------------------------------------


class V5DanglingRefsTest(unittest.TestCase):
    def test_edge_guard_ref_not_defined(self):
        graph = minimal_graph(edges=[data_edge(guard_ref="g:ghost")])
        self.assertIn(
            ("V5_GUARD_REF_NOT_DEFINED", "edges[e:data].guard_ref"),
            pairs(validate_graph_references(graph)),
        )

    def test_activation_guard_ref_not_defined(self):
        graph = minimal_graph(
            guard_activations=[
                GuardActivation("g:ghost", "s1", "true"),
            ]
        )
        got = pairs(validate_graph_references(graph))
        self.assertTrue(
            any(c == "V5_GUARD_REF_NOT_DEFINED" and "guard_activations" in p
                for c, p in got),
            f"激活的悬空 guard 未报出：{got}",
        )

    def test_alias_base_and_alias_tensor_refs(self):
        graph = minimal_graph(
            alias_relations=[
                AliasRelation("al:1", "tensor:ghost_b", "tensor:a", "view",
                              transform_expr="a[0]"),
                AliasRelation("al:2", "tensor:a", "tensor:ghost_a", "view",
                              transform_expr="a[0]"),
            ]
        )
        got = set(pairs(validate_graph_references(graph)))
        self.assertIn(
            ("V5_ALIAS_TENSOR_NOT_DEFINED", "alias_relations[al:1].base_tensor_id"),
            got,
        )
        self.assertIn(
            ("V5_ALIAS_TENSOR_NOT_DEFINED", "alias_relations[al:2].alias_tensor_id"),
            got,
        )

    def test_valid_guard_and_alias_refs_not_reported(self):
        graph = minimal_graph(
            guards=[Guard("g:1", "x > 0")],
            guard_activations=[GuardActivation("g:1", "s1", "true")],
            edges=[data_edge(guard_ref="g:1")],
            alias_relations=[
                AliasRelation("al:1", "tensor:a", "tensor:b", "view",
                              transform_expr="a[0]"),
            ],
        )
        self.assertEqual(validate_graph_references(graph), ())

    def test_transform_expr_and_predicate_expr_are_not_executed(self):
        sentinel = Path(tempfile.gettempdir()) / "dsh_01b_expr_must_not_run.txt"
        if sentinel.exists():
            sentinel.unlink()
        expr = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('ran')"
        graph = minimal_graph(
            guards=[Guard("g:1", expr)],
            guard_activations=[GuardActivation("g:1", "s1", "true")],
            edges=[data_edge(guard_ref="g:1")],
            alias_relations=[
                AliasRelation("al:1", "tensor:a", "tensor:b", "view",
                              transform_expr=expr),
            ],
        )
        self.assertEqual(validate_graph_references(graph), ())
        self.assertFalse(
            sentinel.exists(),
            "检查器执行了 transform_expr/predicate_expr（副作用哨兵文件被创建）",
        )


class DiagnosticStabilityBoundaryTest(unittest.TestCase):
    """返修第二小步（诊断稳定性）的补充边界：环入口、独立环、重复定义排列、内部顺序。

    断言基于**代码语义结果**（完整 issue 签名与环成员），不使用字符串排序作为通过证据。
    """

    def test_version_cycle_with_non_cycle_predecessor_locates_real_cycle(self):
        """带非环前驱入口时，诊断须定位真正环，不得把通往环的前缀当成环的一部分。"""

        def build(t_order: str, v_order: str) -> SemanticGraph:
            versions = {
                "entry": StateVersion("sv:entry", "s:k", "v-entry"),
                "c0": StateVersion("sv:c0", "s:k", "v0"),
                "c1": StateVersion("sv:c1", "s:k", "v1"),
                "c2": StateVersion("sv:c2", "s:k", "v2"),
            }
            v_keys = ["entry", "c0", "c1", "c2"]
            if v_order == "reverse":
                v_keys = ["c2", "c1", "c0", "entry"]
            transitions = [
                StateTransition("t:entry", "s:k", "sv:entry", "sv:c0"),
                StateTransition("t:c0", "s:k", "sv:c0", "sv:c1"),
                StateTransition("t:c1", "s:k", "sv:c1", "sv:c2"),
                StateTransition("t:c2", "s:k", "sv:c2", "sv:c0"),
            ]
            if t_order == "reverse":
                transitions = list(reversed(transitions))
            return minimal_graph(
                state_versions=[versions[k] for k in v_keys],
                state_transitions=transitions,
            )

        sigs = []
        for t_order in ("as_is", "reverse"):
            for v_order in ("as_is", "reverse"):
                issues = validate_graph_references(build(t_order, v_order))
                cycle = [i for i in issues if i.code == "V4_VERSION_CYCLE"]
                self.assertTrue(
                    cycle, f"真实环未发现（transitions={t_order}, versions={v_order}）"
                )
                message = " ".join(i.message for i in cycle)
                for member in ("sv:c0", "sv:c1", "sv:c2"):
                    self.assertIn(
                        member, message,
                        f"环成员 {member} 未被定位（transitions={t_order}, versions={v_order}）",
                    )
                self.assertNotIn(
                    "sv:entry", message,
                    "把通往环的非环前缀 sv:entry 当成了环的一部分",
                )
                sigs.append(signatures(issues))
        for sig in sigs[1:]:
            self.assertEqual(sigs[0], sig, "带非环入口的环诊断随录入顺序变化")

    def test_two_disjoint_cycles_order_invariant(self):
        """两个独立环：均须可检出，且诊断在 transition 三种排列下完全一致。"""

        def build(order: str) -> SemanticGraph:
            versions = [
                StateVersion("a:0", "s:a", "v0"),
                StateVersion("a:1", "s:a", "v1"),
                StateVersion("b:0", "s:b", "v0"),
                StateVersion("b:1", "s:b", "v1"),
            ]
            transitions = [
                StateTransition("t:a0", "s:a", "a:0", "a:1"),
                StateTransition("t:a1", "s:a", "a:1", "a:0"),
                StateTransition("t:b0", "s:b", "b:0", "b:1"),
                StateTransition("t:b1", "s:b", "b:1", "b:0"),
            ]
            if order == "reverse":
                transitions = list(reversed(transitions))
            elif order == "interleaved":
                transitions = [transitions[0], transitions[2],
                               transitions[1], transitions[3]]
            return minimal_graph(state_versions=versions, state_transitions=transitions)

        results = {order: validate_graph_references(build(order))
                   for order in ("as_is", "reverse", "interleaved")}
        for order, issues in results.items():
            message = " ".join(
                i.message for i in issues if i.code == "V4_VERSION_CYCLE"
            )
            for member in ("a:0", "a:1", "b:0", "b:1"):
                self.assertIn(member, message, f"{order}: 独立环成员 {member} 未被提及")
        base = signatures(results["as_is"])
        for order in ("reverse", "interleaved"):
            self.assertEqual(base, signatures(results[order]),
                             f"{order}: 双环诊断随录入顺序变化")

    def test_duplicate_definitions_order_invariant_across_entity_kinds(self):
        """重复 stable_id 且定义不同：多种录入排列下 issue 序列稳定，重复仍被具名报出。"""

        other_h = axis("H", 8192, meaning="同名不同定义轴")

        def axis_graph(swap: bool) -> SemanticGraph:
            axes = [other_h, H, D] if swap else [H, other_h, D]
            return minimal_graph(named_axes=axes)

        def node_graph(swap: bool) -> SemanticGraph:
            dup_x = node("n:dup", outputs=[("x", "tensor:a")])
            dup_y = node("n:dup", outputs=[("y", "tensor:a")])
            dup_nodes = [dup_y, dup_x] if swap else [dup_x, dup_y]
            return minimal_graph(
                nodes=dup_nodes + [node("n:dst", inputs=[("in", "tensor:a")])],
                edges=[data_edge("e:data", src="n:dup", src_port="x",
                                 dst="n:dst", dst_port="in", tid="tensor:a")],
            )

        def state_version_graph(swap: bool) -> SemanticGraph:
            dup_a = StateVersion("sv:dup", "s:a", "v0")
            dup_b = StateVersion("sv:dup", "s:b", "v1")
            versions = [dup_b, dup_a] if swap else [dup_a, dup_b]
            return minimal_graph(
                state_versions=versions + [StateVersion("sv:1", "s:a", "v1")],
                state_transitions=[StateTransition("st:0", "s:a", "sv:dup", "sv:1")],
            )

        def state_transition_graph(swap: bool) -> SemanticGraph:
            dup_ok = StateTransition("st:dup", "s:a", "sv:a0", "sv:a1")
            dup_bad = StateTransition("st:dup", "s:a", "sv:a0", "sv:ghost")
            transitions = [dup_bad, dup_ok] if swap else [dup_ok, dup_bad]
            return minimal_graph(
                state_versions=[
                    StateVersion("sv:a0", "s:a", "v0"),
                    StateVersion("sv:a1", "s:a", "v1"),
                ],
                state_transitions=transitions,
            )

        cases = {
            "axis": axis_graph,
            "node": node_graph,
            "state_version": state_version_graph,
            "state_transition": state_transition_graph,
        }
        for entity, build in cases.items():
            with self.subTest(entity=entity):
                first = validate_graph_references(build(False))
                second = validate_graph_references(build(True))
                self.assertIn(
                    "V1_DUPLICATE_ID", codes(first),
                    f"{entity}: 重复定义被静默吞掉",
                )
                self.assertIn("V1_DUPLICATE_ID", codes(second),
                              f"{entity}: 重复定义被静默吞掉")
                self.assertEqual(
                    signatures(first), signatures(second),
                    f"{entity}: 重复定义诊断随录入顺序变化",
                )

    def test_duplicate_with_permuted_internal_port_order_still_reported(self):
        """同名节点仅端口顺序不同仍是重复且不得被规范化合并；issue 跨排列稳定。"""
        node_ab = node("n:dup", outputs=[("p1", "tensor:a"), ("p2", "tensor:b")])
        node_ba = node("n:dup", outputs=[("p2", "tensor:b"), ("p1", "tensor:a")])

        def build(swap: bool) -> SemanticGraph:
            dups = [node_ba, node_ab] if swap else [node_ab, node_ba]
            return minimal_graph(
                nodes=dups + [node("n:dst", inputs=[("in", "tensor:a")])],
                edges=[data_edge("e:data", src="n:dup", src_port="p1",
                                 dst="n:dst", dst_port="in", tid="tensor:a")],
            )

        first = validate_graph_references(build(False))
        second = validate_graph_references(build(True))
        self.assertIn("V1_DUPLICATE_ID", codes(first))
        self.assertIn("V1_DUPLICATE_ID", codes(second))
        self.assertEqual(
            signatures(first), signatures(second),
            "端口顺序不同的重复节点导致 issue 随录入顺序变化（内部顺序被错误规范化）",
        )

    def test_valid_tensor_with_non_sorted_axis_order_is_not_normalized(self):
        """合法张量轴按录入顺序语义保留（D,H）不得被视为需排序规范化。"""
        tensor_dh = tensor("a", axes=(D, H))
        graph = minimal_graph(
            tensors=[tensor_dh, tensor("b")],
            nodes=[
                node("n:src", outputs=[("out", "tensor:a")]),
                node("n:dst", inputs=[("in", "tensor:a")]),
            ],
            edges=[data_edge(tid="tensor:a")],
        )
        self.assertEqual(validate_graph_references(graph), ())
        self.assertEqual(tensor_dh.logical_layout, ("D", "H"))
        self.assertEqual(tensor_dh.axes[0].symbol, "D")

    def test_valid_node_with_non_sorted_port_order_is_not_normalized(self):
        """合法节点端口按录入顺序保留（非字典序）不得误报或错配 tensor。"""
        graph = minimal_graph(
            nodes=[
                node("n:src", outputs=[("z", "tensor:a"), ("a", "tensor:b")]),
                node("n:dst", inputs=[("b", "tensor:b"), ("i", "tensor:a")]),
            ],
            edges=[
                data_edge("e:z", src="n:src", src_port="z", dst="n:dst",
                          dst_port="i", tid="tensor:a"),
                data_edge("e:a", src="n:src", src_port="a", dst="n:dst",
                          dst_port="b", tid="tensor:b"),
            ],
        )
        self.assertEqual(validate_graph_references(graph), ())


class SccBoundarySubprocessTest(unittest.TestCase):
    """SCC/环诊断的有界子进程测试（防挂起）。

    任务单 review_01b_scc_in_progress.md：合法链 after 末端（sink）可能裸 KeyError；
    分支 SCC v0→v1、v1→v2、v2→v1、v2→v0 中"取最小后继直到回到最小 start"的走法
    可能永不返回（无限循环）。因此这里**一律在带超时的子进程**中调用产品：
    产品若挂起，超时会终止子进程并让本测试明确失败，而不会挂住整个测试进程。
    环形输出必须是**真实有限环**（首尾同一版本、每步都是真实 transition 边）
    或明确的**强连通分量成员集合**；不接受把 SCC 成员按任意顺序画成路径。
    """

    TIMEOUT_SECONDS = 15

    def _run_subprocess(self, body: str) -> dict:
        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.checks.graph_contracts import validate_graph_references\n"
            "from structure_reader.graph_document import SemanticGraph\n"
            "from structure_reader.graph_ir import (\n"
            "    AliasRelation, Edge, FoldBoundaryMapping, FoldGroup, Guard,\n"
            "    GuardActivation, NamedAxis, Node, Port, StateTransition,\n"
            "    StateVersion, Tensor,\n"
            ")\n"
            f"{body}\n"
            "print('DSH_JSON_BEGIN')\n"
            "print(json.dumps(result, ensure_ascii=False))\n"
            "print('DSH_JSON_END')\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            self.fail(
                f"检查器在 {self.TIMEOUT_SECONDS}s 内未返回：疑似后继选择/环遍历无限循环（已终止子进程）"
            )
        self.assertEqual(
            proc.returncode, 0,
            f"检查器子进程异常退出（非受控 issue）: {proc.stderr.strip()[-600:]}",
        )
        self.assertIn("DSH_JSON_BEGIN", proc.stdout)
        payload = proc.stdout.split("DSH_JSON_BEGIN", 1)[1]
        payload = payload.split("DSH_JSON_END", 1)[0].strip()
        return json.loads(payload)

    def test_legal_chain_terminal_sink_no_crash_no_cycle(self):
        """合法链 v0→v1→v2 的 sink（sv:2 无出边、不在 adjacency）不得裸 KeyError。"""
        body = (
            "versions = [StateVersion(f'sv:{i}', 's:k', f'v{i}') for i in range(3)]\n"
            "transitions = [\n"
            "    StateTransition('t:0', 's:k', 'sv:0', 'sv:1'),\n"
            "    StateTransition('t:1', 's:k', 'sv:1', 'sv:2'),\n"
            "]\n"
            "graph = SemanticGraph(template_id='tpl:scc', state_versions=versions,\n"
            "                      state_transitions=transitions)\n"
            "issues = validate_graph_references(graph)\n"
            "result = {'issues': [{'code': i.code, 'path': i.path, 'message': i.message}\n"
            "                     for i in issues]}\n"
        )
        payload = self._run_subprocess(body)
        cycle = [i for i in payload["issues"] if i["code"] == "V4_VERSION_CYCLE"]
        self.assertEqual(cycle, [], f"无环合法链被误报为版本环: {cycle}")

    def test_simple_cycle_path_is_real_closed_walk(self):
        """Codex 第3点用例 v0→v1→v2→v0：箭头路径须闭合、每步真实，禁止 `v1->v2->v0->v0` 式发明自环。"""
        body = (
            "versions = [StateVersion(f'sv:{i}', 's:k', f'v{i}') for i in range(3)]\n"
            "transitions = [\n"
            "    StateTransition('t:0', 's:k', 'sv:0', 'sv:1'),\n"
            "    StateTransition('t:1', 's:k', 'sv:1', 'sv:2'),\n"
            "    StateTransition('t:2', 's:k', 'sv:2', 'sv:0'),\n"
            "]\n"
            "graph = SemanticGraph(template_id='tpl:scc', state_versions=versions,\n"
            "                      state_transitions=transitions)\n"
            "issues = validate_graph_references(graph)\n"
            "result = {'cycle_messages': [i.message for i in issues\n"
            "                             if i.code == 'V4_VERSION_CYCLE']}\n"
        )
        payload = self._run_subprocess(body)
        self.assertTrue(payload["cycle_messages"], "简单环未被检出")
        edges = {("sv:0", "sv:1"), ("sv:1", "sv:2"), ("sv:2", "sv:0")}
        members = {"sv:0", "sv:1", "sv:2"}
        for message in payload["cycle_messages"]:
            ok, why = _cycle_message_shape(message, edges, members)
            self.assertTrue(
                ok, f"简单环诊断不是真实闭合路径或完整成员集合（{why}）: {message!r}"
            )
            if "->" in message:
                ids = re.findall(r"sv:\d+", message)
                self.assertEqual(ids[0], ids[-1], f"箭头路径首尾不闭合: {message!r}")
                self.assertFalse(
                    any(a == b for a, b in zip(ids, ids[1:])),
                    f"箭头路径发明了不存在的自环边: {message!r}",
                )

    def test_branching_scc_terminates_and_reports_real_cycle(self):
        """分支 SCC v0→v1,v1→v2,v2→v1,v2→v0：须有界结束，环输出须真实（非任意排序）。"""
        body = (
            "versions = [StateVersion(f'sv:{i}', 's:k', f'v{i}') for i in range(3)]\n"
            "triples = [\n"
            "    ('t:01', 'sv:0', 'sv:1'),\n"
            "    ('t:12', 'sv:1', 'sv:2'),\n"
            "    ('t:21', 'sv:2', 'sv:1'),\n"
            "    ('t:20', 'sv:2', 'sv:0'),\n"
            "]\n"
            "transitions = [StateTransition(t, 's:k', b, a) for t, b, a in triples]\n"
            "graph = SemanticGraph(template_id='tpl:scc', state_versions=versions,\n"
            "                      state_transitions=transitions)\n"
            "issues = validate_graph_references(graph)\n"
            "result = {'cycle_messages': [i.message for i in issues\n"
            "                             if i.code == 'V4_VERSION_CYCLE'],\n"
            "          'all_codes': sorted(i.code for i in issues)}\n"
        )
        payload = self._run_subprocess(body)
        self.assertTrue(
            payload["cycle_messages"],
            f"分支 SCC 未被检出为环（codes={payload['all_codes']}）",
        )
        edges = {("sv:0", "sv:1"), ("sv:1", "sv:2"), ("sv:2", "sv:1"), ("sv:2", "sv:0")}
        members = {"sv:0", "sv:1", "sv:2"}
        for message in payload["cycle_messages"]:
            ok, why = _cycle_message_shape(message, edges, members)
            self.assertTrue(ok, f"环输出不是真实有限环/成员集合（{why}）: {message!r}")

    def test_branching_scc_message_order_invariant(self):
        """分支 SCC 的环输出在 transition 三种录入排列下完全一致（不得随顺序变化）。"""
        body = (
            "versions = [StateVersion(f'sv:{i}', 's:k', f'v{i}') for i in range(3)]\n"
            "orders = {\n"
            "    'as_is': [('t:01','sv:0','sv:1'), ('t:12','sv:1','sv:2'),\n"
            "              ('t:21','sv:2','sv:1'), ('t:20','sv:2','sv:0')],\n"
            "    'reverse': [('t:20','sv:2','sv:0'), ('t:21','sv:2','sv:1'),\n"
            "                ('t:12','sv:1','sv:2'), ('t:01','sv:0','sv:1')],\n"
            "    'interleaved': [('t:12','sv:1','sv:2'), ('t:20','sv:2','sv:0'),\n"
            "                    ('t:01','sv:0','sv:1'), ('t:21','sv:2','sv:1')],\n"
            "}\n"
            "out = {}\n"
            "for name, triples in orders.items():\n"
            "    transitions = [StateTransition(t, 's:k', b, a) for t, b, a in triples]\n"
            "    graph = SemanticGraph(template_id='tpl:scc', state_versions=versions,\n"
            "                          state_transitions=transitions)\n"
            "    issues = validate_graph_references(graph)\n"
            "    out[name] = [i.message for i in issues if i.code == 'V4_VERSION_CYCLE']\n"
            "result = out\n"
        )
        payload = self._run_subprocess(body)
        base = payload["as_is"]
        self.assertTrue(base, "分支 SCC 未被检出为环")
        for name in ("reverse", "interleaved"):
            self.assertEqual(base, payload[name], f"{name}: 环输出随录入顺序变化")


def _cycle_message_shape(message: str, edges: set, members: set) -> tuple[bool, str]:
    """校验环消息形态（只认语义形态，不认"含成员名字"或字节稳定）。

    箭头路径：必须首尾同一版本，且每一相邻有向边都在真实 transition 边集合内——
    因此既禁止"发明自环"（如 v1->v2->v0->v0），也禁止把 SCC 成员按任意顺序
    拼成路径，还禁止提到 SCC 外版本。
    成员集合（无箭头）：必须恰好给出该 SCC 的全部成员。
    """
    mentioned = re.findall(r"sv:\d+", message)
    if "->" in message:
        if len(mentioned) < 2 or mentioned[0] != mentioned[-1]:
            return False, "箭头路径未闭合（首尾版本不同）"
        if not set(mentioned) <= members:
            return False, "箭头路径提及了 SCC 外版本"
        for left, right in zip(mentioned, mentioned[1:]):
            if (left, right) not in edges:
                return False, f"箭头路径含非真实有向边 {left}->{right}"
        return True, "real-closed-path"
    if not mentioned:
        return False, "既非箭头路径也未给出任何版本"
    if set(mentioned) != members:
        return False, "成员集合未恰好覆盖该 SCC 全部成员"
    return True, "member-set"


if __name__ == "__main__":
    unittest.main(verbosity=2)
