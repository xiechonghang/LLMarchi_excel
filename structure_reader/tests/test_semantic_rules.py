"""语义图规则/校验的正反例与确定性张量/索引映射核对（T12 单元测试）。

期望值由本文件独立计算（显式索引/嵌套循环），不由被测生成器产出；不使用模型
实例或权重。运行：cd structure_reader && python3 tests/test_semantic_rules.py
"""

from __future__ import annotations

import itertools
import json
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from structure_reader.semantic.axes import AxisCatalog, AxisDef
from structure_reader.semantic.bundle import GraphBuilder, SourceRef, validate_bundle
from structure_reader.semantic.rules import normalize, validate_normalization

MODEL_METADATA = {
    "model_key": "test_model",
    "layer_key": "layer.0",
    "layer_group": "test",
    "template_id": "test:template",
    "source_revision": "0" * 40,
}


def _catalog() -> AxisCatalog:
    catalog = AxisCatalog()
    catalog.register(
        AxisDef("batch_size", "批大小", "test", "test:1", aliases=("B",))
    )
    catalog.register(
        AxisDef("seqlen", "本次输入 token 数", "test", "test:2", aliases=("S", "T"))
    )
    catalog.register(
        AxisDef("hidden_size", "隐藏维", "test", "test:3", value=4096, aliases=("H",))
    )
    catalog.register(
        AxisDef("head_dim", "头维", "test", "test:4", value=512, aliases=("Dh",))
    )
    catalog.register(
        AxisDef(
            "num_attention_heads",
            "注意力头数",
            "test",
            "test:5",
            value=64,
            aliases=("Nh",),
        )
    )
    catalog.register(
        AxisDef(
            "num_key_value_heads",
            "KV 头数",
            "test",
            "test:6",
            value=1,
            display_value=True,
            aliases=("Nkv",),
        )
    )
    catalog.register(
        AxisDef(
            "qk_rope_head_dim",
            "RoPE 维",
            "test",
            "test:7",
            value=64,
            aliases=("Rk",),
        )
    )
    catalog.register(
        AxisDef("key_length", "键长度", "test", "test:13")
    )
    catalog.register(
        AxisDef(
            "attention_heads_total",
            "num_attention_heads * head_dim",
            "test",
            "test:8",
            expression="num_attention_heads * head_dim",
        )
    )
    return catalog


def _source(line: int) -> SourceRef:
    return SourceRef(repo_key="test", path="model.py", line=line, end_line=line + 2)


def _build(spec: dict) -> "object":
    """按声明构建测试图：tensors/nodes/edges/state。"""
    catalog = _catalog()
    builder = GraphBuilder("test:template", catalog)
    for tensor in spec["tensors"]:
        builder.tensor(
            tensor["id"],
            tensor["axes"],
            label=tensor["id"],
            dtype=tensor.get("dtype", "bfloat16"),
            role=tensor.get("role", "activation"),
            physical_layout=tensor.get("physical_layout"),
        )
    for node in spec["nodes"]:
        builder.node(
            node["id"],
            op_type=node.get("op_type", node["kind"]),
            granularity=node.get("granularity", node["kind"]),
            label=node["id"],
            kind=node["kind"],
            module_path="model",
            sources=[_source(index + 10) for index in range(len(node.get("legacy", ("x",))))],
            inputs=node.get("inputs", ()),
            outputs=node.get("outputs", ()),
            legacy_refs=node.get("legacy", ("op:" + node["id"],)),
            condition=node.get("condition"),
        )
    for edge in spec.get("edges", ()):
        builder.edge(*edge)
    return builder.build(dict(MODEL_METADATA))


def _edge(src: str, src_port: str, dst: str, dst_port: str) -> tuple:
    return (f"e.{src}.{src_port}.{dst}.{dst_port}", src, src_port, dst, dst_port)


def _linear_bundle(transpose_pair: bool, shared: bool = False) -> "object":
    """x [B,H,S,D] → T1(1,2) → [B,S,H,D] → T2(1,2) → [B,H,S,D] → out。"""
    nodes = [
        {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
        {"id": "t1", "kind": "transpose", "inputs": (("x", "x"),), "outputs": (("y", "t1out"),)},
        {
            "id": "t2",
            "kind": "transpose",
            "inputs": (("x", "t1out"),),
            "outputs": (("y", "t2out"),),
        },
        {"id": "out", "kind": "external_out", "inputs": (("x", "t2out"),)},
    ]
    tensors = [
        {"id": "x", "axes": ("batch_size", "num_attention_heads", "seqlen", "head_dim")},
        {"id": "t1out", "axes": ("batch_size", "seqlen", "num_attention_heads", "head_dim")},
        {"id": "t2out", "axes": ("batch_size", "num_attention_heads", "seqlen", "head_dim")},
    ]
    edges = [
        _edge("input", "x", "t1", "x"),
        _edge("t1", "y", "t2", "x"),
        _edge("t2", "y", "out", "x"),
    ]
    if shared:
        nodes.append({"id": "side", "kind": "external_out", "inputs": (("x", "t1out"),)})
        edges.append(_edge("t1", "y", "side", "x"))
    return _build({"tensors": tensors, "nodes": nodes, "edges": edges})


class TestRules(unittest.TestCase):
    def test_transpose_pair_cancelled_and_preserved(self):
        bundle = _linear_bundle(transpose_pair=True)
        self.assertEqual(validate_bundle(bundle), ())
        result = normalize(bundle)
        normalized = result.normalized
        self.assertNotIn("t1", normalized.nodes)
        self.assertNotIn("t2", normalized.nodes)
        report = {item.rule_id: item for item in result.reports}["R3-transpose-pair-cancel"]
        self.assertEqual(report.status, "applied")
        self.assertEqual(len(report.applications), 1)
        # 消费者仍挂在原始张量上，且可达关系不变
        self.assertIn("out", normalized.nodes)
        self.assertEqual(validate_normalization(result), ())

    def test_non_inverse_transpose_pair_rejected(self):
        nodes = [
            {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
            {
                "id": "t1",
                "kind": "transpose",
                "inputs": (("x", "x"),),
                "outputs": (("y", "t1out"),),
            },
            {
                "id": "t2",
                "kind": "transpose",
                "inputs": (("x", "t1out"),),
                "outputs": (("y", "t2out"),),
            },
            {"id": "out", "kind": "external_out", "inputs": (("x", "t2out"),)},
        ]
        tensors = [
            {"id": "x", "axes": ("batch_size", "num_attention_heads", "seqlen", "head_dim")},
            {"id": "t1out", "axes": ("batch_size", "seqlen", "num_attention_heads", "head_dim")},
            {"id": "t2out", "axes": ("batch_size", "seqlen", "head_dim", "num_attention_heads")},
        ]
        edges = [
            _edge("input", "x", "t1", "x"),
            _edge("t1", "y", "t2", "x"),
            _edge("t2", "y", "out", "x"),
        ]
        result = normalize(_build({"tensors": tensors, "nodes": nodes, "edges": edges}))
        report = {item.rule_id: item for item in result.reports}["R3-transpose-pair-cancel"]
        self.assertEqual(report.status, "rejected")
        self.assertTrue(report.rejected_candidates)
        self.assertIn("t1", result.normalized.nodes)
        self.assertIn("t2", result.normalized.nodes)

    def test_transpose_pair_with_shared_intermediate_rejected(self):
        bundle = _linear_bundle(transpose_pair=True, shared=True)
        result = normalize(bundle)
        report = {item.rule_id: item for item in result.reports}["R3-transpose-pair-cancel"]
        self.assertEqual(report.status, "rejected")
        self.assertIn("t1", result.normalized.nodes)

    def test_unit_axis_view_transpose_becomes_reshape(self):
        nodes = [
            {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
            {"id": "view", "kind": "view", "inputs": (("x", "x"),), "outputs": (("y", "y"),)},
            {
                "id": "transpose",
                "kind": "transpose",
                "inputs": (("x", "y"),),
                "outputs": (("y", "z"),),
            },
            {"id": "out", "kind": "external_out", "inputs": (("x", "z"),)},
        ]
        tensors = [
            {"id": "x", "axes": ("batch_size", "seqlen", "head_dim")},
            {"id": "y", "axes": ("batch_size", "seqlen", "num_key_value_heads", "head_dim")},
            {"id": "z", "axes": ("batch_size", "num_key_value_heads", "seqlen", "head_dim")},
        ]
        edges = [
            _edge("input", "x", "view", "x"),
            _edge("view", "y", "transpose", "x"),
            _edge("transpose", "y", "out", "x"),
        ]
        result = normalize(_build({"tensors": tensors, "nodes": nodes, "edges": edges}))
        report = {item.rule_id: item for item in result.reports}["R5-unit-axis-reorder"]
        self.assertEqual(report.status, "applied")
        normalized = result.normalized
        self.assertNotIn("view", normalized.nodes)
        self.assertNotIn("transpose", normalized.nodes)
        reshape_nodes = [
            node
            for node in normalized.graph.nodes
            if normalized.node_meta[node.stable_id].kind == "reshape"
        ]
        self.assertEqual(len(reshape_nodes), 1)
        tensor = normalized.tensors[reshape_nodes[0].output_ports[0].tensor_id]
        self.assertEqual(len(tensor.axes), 4)  # 三维→四维变化保留
        self.assertEqual(validate_normalization(result), ())

    def test_multi_head_view_transpose_not_reduced(self):
        """Q 路径 64 头：非单位轴顺序改变，不能化为 Reshape。"""
        nodes = [
            {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
            {"id": "view", "kind": "view", "inputs": (("x", "x"),), "outputs": (("y", "y"),)},
            {
                "id": "transpose",
                "kind": "transpose",
                "inputs": (("x", "y"),),
                "outputs": (("y", "z"),),
            },
            {"id": "out", "kind": "external_out", "inputs": (("x", "z"),)},
        ]
        tensors = [
            {"id": "x", "axes": ("batch_size", "seqlen", "attention_heads_total")},
            {"id": "y", "axes": ("batch_size", "seqlen", "num_attention_heads", "head_dim")},
            {"id": "z", "axes": ("batch_size", "num_attention_heads", "seqlen", "head_dim")},
        ]
        edges = [
            _edge("input", "x", "view", "x"),
            _edge("view", "y", "transpose", "x"),
            _edge("transpose", "y", "out", "x"),
        ]
        result = normalize(_build({"tensors": tensors, "nodes": nodes, "edges": edges}))
        report = {item.rule_id: item for item in result.reports}["R5-unit-axis-reorder"]
        self.assertEqual(report.status, "rejected")
        reasons = " ".join(item["reason"] for item in report.rejected_candidates)
        self.assertIn("改变了非单位轴的相对顺序", reasons)
        self.assertIn("transpose", result.normalized.nodes)

    def test_transpose_into_matmul_merges_last_two_axes(self):
        nodes = [
            {"id": "q", "kind": "external", "outputs": (("x", "q"),)},
            {"id": "k", "kind": "external", "outputs": (("x", "k"),)},
            {
                "id": "kt",
                "kind": "transpose",
                "inputs": (("x", "k"),),
                "outputs": (("y", "kt"),),
            },
            {
                "id": "mm",
                "kind": "matmul",
                "inputs": (("lhs", "q"), ("rhs", "kt")),
                "outputs": (("y", "logits"),),
            },
            {"id": "out", "kind": "external_out", "inputs": (("x", "logits"),)},
        ]
        tensors = [
            {"id": "q", "axes": ("batch_size", "num_attention_heads", "seqlen", "head_dim")},
            {"id": "k", "axes": ("batch_size", "num_attention_heads", "seqlen", "head_dim")},
            {"id": "kt", "axes": ("batch_size", "num_attention_heads", "head_dim", "seqlen")},
            {
                "id": "logits",
                "axes": ("batch_size", "num_attention_heads", "seqlen", "key_length"),
            },
        ]
        edges = [
            _edge("q", "x", "mm", "lhs"),
            _edge("k", "x", "kt", "x"),
            _edge("kt", "y", "mm", "rhs"),
            _edge("mm", "y", "out", "x"),
        ]
        result = normalize(_build({"tensors": tensors, "nodes": nodes, "edges": edges}))
        report = {item.rule_id: item for item in result.reports}["R7-transpose-into-matmul"]
        self.assertEqual(report.status, "applied")
        normalized = result.normalized
        self.assertNotIn("kt", normalized.nodes)
        meta = normalized.node_meta["mm"]
        self.assertTrue(meta.label.endswith("ᵀ · MatMul"), meta.label)
        self.assertEqual(meta.attributes["transposed_operand"], "rhs")
        self.assertEqual(meta.attributes["rhs_transposed_axes"], "last_two_axes")
        self.assertIn("kt", meta.attributes["rhs_transpose_source"])
        self.assertEqual(validate_normalization(result), ())

    def test_transpose_swapping_head_axes_not_merged(self):
        nodes = [
            {"id": "q", "kind": "external", "outputs": (("x", "q"),)},
            {"id": "k", "kind": "external", "outputs": (("x", "k"),)},
            {
                "id": "kt",
                "kind": "transpose",
                "inputs": (("x", "k"),),
                "outputs": (("y", "kt"),),
            },
            {
                "id": "mm",
                "kind": "matmul",
                "inputs": (("lhs", "q"), ("rhs", "kt")),
                "outputs": (("y", "logits"),),
            },
            {"id": "out", "kind": "external_out", "inputs": (("x", "logits"),)},
        ]
        tensors = [
            {"id": "q", "axes": ("batch_size", "num_attention_heads", "seqlen", "head_dim")},
            {"id": "k", "axes": ("batch_size", "num_attention_heads", "seqlen", "head_dim")},
            {"id": "kt", "axes": ("batch_size", "seqlen", "num_attention_heads", "head_dim")},
            {
                "id": "logits",
                "axes": ("batch_size", "num_attention_heads", "seqlen", "key_length"),
            },
        ]
        edges = [
            _edge("q", "x", "mm", "lhs"),
            _edge("k", "x", "kt", "x"),
            _edge("kt", "y", "mm", "rhs"),
            _edge("mm", "y", "out", "x"),
        ]
        result = normalize(_build({"tensors": tensors, "nodes": nodes, "edges": edges}))
        report = {item.rule_id: item for item in result.reports}["R7-transpose-into-matmul"]
        self.assertEqual(report.status, "rejected")
        self.assertIn("kt", result.normalized.nodes)

    def test_split_concat_cancel_positive_and_branch_negative(self):
        def bundle(branch: bool):
            nodes = [
                {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
                {
                    "id": "split",
                    "kind": "split",
                    "inputs": (("x", "x"),),
                    "outputs": (("a", "ya"), ("b", "yb")),
                },
                {
                    "id": "cat",
                    "kind": "concat",
                    "inputs": (("a", "ya"), ("b", "yb")),
                    "outputs": (("y", "z"),),
                },
                {"id": "out", "kind": "external_out", "inputs": (("x", "z"),)},
            ]
            tensors = [
                {"id": "x", "axes": ("batch_size", "seqlen", "head_dim")},
                {"id": "ya", "axes": ("batch_size", "seqlen", "qk_rope_head_dim")},
                {"id": "yb", "axes": ("batch_size", "seqlen", "qk_rope_head_dim")},
                {"id": "z", "axes": ("batch_size", "seqlen", "head_dim")},
            ]
            edges = [
                _edge("input", "x", "split", "x"),
                _edge("split", "a", "cat", "a"),
                _edge("split", "b", "cat", "b"),
                _edge("cat", "y", "out", "x"),
            ]
            if branch:
                nodes.append({"id": "side", "kind": "external_out", "inputs": (("x", "ya"),)})
                edges.append(_edge("split", "a", "side", "x"))
            return _build({"tensors": tensors, "nodes": nodes, "edges": edges})

        applied = normalize(bundle(branch=False))
        report = {item.rule_id: item for item in applied.reports}["R6-split-concat-cancel"]
        self.assertEqual(report.status, "applied")
        self.assertNotIn("split", applied.normalized.nodes)
        self.assertNotIn("cat", applied.normalized.nodes)
        self.assertEqual(validate_normalization(applied), ())

        rejected = normalize(bundle(branch=True))
        report = {item.rule_id: item for item in rejected.reports}["R6-split-concat-cancel"]
        self.assertEqual(report.status, "rejected")
        self.assertIn("split", rejected.normalized.nodes)
        self.assertIn("cat", rejected.normalized.nodes)

    def test_identity_reshape_removed_with_shared_consumers(self):
        nodes = [
            {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
            {
                "id": "reshape",
                "kind": "reshape",
                "inputs": (("x", "x"),),
                "outputs": (("y", "y"),),
            },
            {"id": "out1", "kind": "external_out", "inputs": (("x", "y"),)},
            {"id": "out2", "kind": "external_out", "inputs": (("x", "y"),)},
        ]
        tensors = [
            {"id": "x", "axes": ("batch_size", "seqlen", "head_dim")},
            {"id": "y", "axes": ("batch_size", "seqlen", "head_dim")},
        ]
        edges = [
            _edge("input", "x", "reshape", "x"),
            _edge("reshape", "y", "out1", "x"),
            _edge("reshape", "y", "out2", "x"),
        ]
        result = normalize(_build({"tensors": tensors, "nodes": nodes, "edges": edges}))
        report = {item.rule_id: item for item in result.reports}["R2-identity-eliminate"]
        self.assertEqual(report.status, "applied")
        self.assertNotIn("reshape", result.normalized.nodes)
        self.assertEqual(validate_normalization(result), ())
        consumers = [
            edge.dst_node_id
            for edge in result.normalized.graph.edges
            if edge.src_node_id == "input"
        ]
        self.assertEqual(sorted(consumers), ["out1", "out2"])

    def test_consecutive_reshape_merged(self):
        nodes = [
            {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
            {"id": "v1", "kind": "view", "inputs": (("x", "x"),), "outputs": (("y", "y1"),)},
            {"id": "v2", "kind": "reshape", "inputs": (("x", "y1"),), "outputs": (("y", "z"),)},
            {"id": "out", "kind": "external_out", "inputs": (("x", "z"),)},
        ]
        tensors = [
            {"id": "x", "axes": ("batch_size", "seqlen", "head_dim")},
            {"id": "y1", "axes": ("batch_size", "seqlen", "num_key_value_heads", "head_dim")},
            {
                "id": "z",
                "axes": ("batch_size", "num_key_value_heads", "seqlen", "head_dim"),
            },
        ]
        edges = [
            _edge("input", "x", "v1", "x"),
            _edge("v1", "y", "v2", "x"),
            _edge("v2", "y", "out", "x"),
        ]
        result = normalize(_build({"tensors": tensors, "nodes": nodes, "edges": edges}))
        report = {item.rule_id: item for item in result.reports}["R4-reshape-merge"]
        self.assertEqual(report.status, "applied")
        self.assertEqual(validate_normalization(result), ())
        kinds = [
            result.normalized.node_meta[node.stable_id].kind
            for node in result.normalized.graph.nodes
        ]
        self.assertEqual(kinds.count("reshape"), 1)

    def test_shape_expression_divisibility_needs_explicit_fact(self):
        nodes = [
            {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
            {
                "id": "broadcast",
                "kind": "broadcast",
                "inputs": (("x", "x"),),
                "outputs": (("y", "y"),),
            },
            {"id": "out", "kind": "external_out", "inputs": (("x", "y"),)},
        ]
        tensors = [
            {"id": "x", "axes": ("batch_size", "num_key_value_heads", "seqlen", "head_dim")},
            {"id": "y", "axes": ("batch_size", "expanded_heads", "seqlen", "head_dim")},
        ]
        # 直接给 y 的 expanded_heads 轴登记表达式
        bundle = _build({"tensors": tensors[:1], "nodes": [], "edges": []})
        catalog = bundle.axes
        catalog.register(
            AxisDef(
                "expanded_heads",
                "广播后的头数",
                "test",
                "test:9",
                expression="num_key_value_heads * (num_attention_heads // num_key_value_heads)",
            )
        )
        builder = GraphBuilder("test:template", catalog)
        for tensor in tensors:
            builder.tensor(tensor["id"], tensor["axes"], label=tensor["id"])
        builder.node(
            "input",
            op_type="external",
            granularity="external",
            label="input",
            kind="external",
            module_path="model",
            sources=[_source(1)],
            outputs=(("x", "x"),),
            legacy_refs=("op:input",),
        )
        builder.node(
            "broadcast",
            op_type="broadcast",
            granularity="broadcast",
            label="broadcast",
            kind="broadcast",
            module_path="model",
            sources=[_source(2)],
            inputs=(("x", "x"),),
            outputs=(("y", "y"),),
            legacy_refs=("op:broadcast",),
        )
        builder.node(
            "out",
            op_type="external_out",
            granularity="external_out",
            label="out",
            kind="external_out",
            module_path="model",
            sources=[_source(3)],
            inputs=(("x", "y"),),
            legacy_refs=("op:out",),
        )
        builder.edge("e1", "input", "x", "broadcast", "x")
        builder.edge("e2", "broadcast", "y", "out", "x")
        graph_bundle = builder.build(dict(MODEL_METADATA))

        without_fact = normalize(graph_bundle)
        report = {item.rule_id: item for item in without_fact.reports}[
            "R1-shape-expression-simplify"
        ]
        self.assertEqual(report.status, "rejected")
        axis = without_fact.normalized.tensors["tensor:y"].axes[1]
        self.assertEqual(
            axis.expression,
            "num_key_value_heads * (num_attention_heads // num_key_value_heads)",
        )

        with_fact = normalize(
            graph_bundle,
            facts=(("divisible", "num_attention_heads", "num_key_value_heads"),),
        )
        report = {item.rule_id: item for item in with_fact.reports}[
            "R1-shape-expression-simplify"
        ]
        self.assertEqual(report.status, "applied")
        axis = with_fact.normalized.tensors["tensor:y"].axes[1]
        self.assertEqual(axis.expression, "num_attention_heads")

    def test_dynamic_length_is_not_specialized(self):
        catalog = _catalog()
        catalog.register(
            AxisDef(
                "kv_length",
                "KV 逻辑长度",
                "test",
                "test:10",
                expression="min(kvcachelen, sliding_window - 1) + seqlen",
            )
        )
        catalog.register(
            AxisDef("kvcachelen", "历史长度", "test", "test:11")
        )
        catalog.register(
            AxisDef("sliding_window", "滑窗", "test", "test:12", value=128, aliases=("W",))
        )
        builder = GraphBuilder("test:template", catalog)
        builder.tensor("x", ("batch_size", "seqlen", "head_dim"), label="x")
        builder.tensor("y", ("batch_size", "kv_length", "head_dim"), label="y")
        builder.node(
            "input",
            op_type="external",
            granularity="external",
            label="input",
            kind="external",
            module_path="model",
            sources=[_source(1)],
            outputs=(("x", "x"),),
            legacy_refs=("op:input",),
        )
        builder.node(
            "cache",
            op_type="cache_update",
            granularity="cache_update",
            label="cache",
            kind="cache_update",
            module_path="model",
            sources=[_source(2)],
            inputs=(("x", "x"),),
            outputs=(("y", "y"),),
            legacy_refs=("op:cache",),
        )
        builder.edge("e1", "input", "x", "cache", "x")
        result = normalize(builder.build(dict(MODEL_METADATA)))
        axis = result.normalized.tensors["tensor:y"].axes[1]
        self.assertIn("kvcachelen", axis.expression)
        self.assertIn("seqlen", axis.expression)


# ---------------------------------------------------------------------------
# 确定性张量/索引映射核对（独立于被测生成器）
# ---------------------------------------------------------------------------


def _tensor(shape, fn):
    """按行主序构造嵌套列表张量，值由 fn(index) 给出。"""
    indices = list(itertools.product(*[range(dimension) for dimension in shape]))

    def build(prefix, dims):
        if not dims:
            return fn(prefix)
        return [build(prefix + (index,), dims[1:]) for index in range(dims[0])]

    return build((), shape)


def _flatten(tensor):
    if isinstance(tensor, list):
        return [item for child in tensor for item in _flatten(child)]
    return [tensor]


def _transpose(tensor, axis_a, axis_b):
    """纯 Python 轴交换（显式索引映射）。"""
    shape = []

    def shape_of(value):
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0]

    shape_of(tensor)
    order = list(range(len(shape)))
    order[axis_a], order[axis_b] = order[axis_b], order[axis_a]

    def get(prefix):
        # prefix 是输出索引；输入索引按 order 映射
        return prefix

    def build(dims, prefix):
        if not dims:
            return None
        return [build(dims[1:], prefix + (index,)) for index in range(dims[0])]

    output = build([shape[index] for index in order], ())

    def fill(node, prefix):
        if isinstance(node, list):
            return [fill(child, prefix + (index,)) for index, child in enumerate(node)]
        input_index = [0] * len(prefix)
        for output_position, input_axis in enumerate(order):
            input_index[input_axis] = prefix[output_position]
        value = tensor
        for index in input_index:
            value = value[index]
        return value

    return fill(output, ())


def _reshape(tensor, shape):
    flat = _flatten(tensor)

    def build(dims, offset):
        if not dims:
            return None
        if len(dims) == 1:
            return flat[offset: offset + dims[0]]
        stride = 1
        for dimension in dims[1:]:
            stride *= dimension
        return [
            build(dims[1:], offset + index * stride) for index in range(dims[0])
        ]

    return build(list(shape), 0)


class TestDeterministicMappings(unittest.TestCase):
    def test_double_transpose_is_identity(self):
        shape = (2, 2, 3, 2)
        data = _tensor(shape, lambda idx: idx[0] * 1000 + idx[1] * 100 + idx[2] * 10 + idx[3])
        twice = _transpose(_transpose(data, 1, 2), 1, 2)
        self.assertEqual(_flatten(twice), _flatten(data))

    def test_unit_axis_transpose_equals_reshape(self):
        data = _tensor((2, 3, 1, 4), lambda idx: idx[0] * 100 + idx[1] * 10 + idx[3])
        transposed = _transpose(data, 1, 2)  # [2,1,3,4]
        reshaped = _reshape(data, (2, 1, 3, 4))
        self.assertEqual(_flatten(transposed), _flatten(reshaped))

    def test_multi_head_transpose_differs_from_reshape(self):
        data = _tensor((2, 3, 2, 4), lambda idx: idx[0] * 1000 + idx[1] * 100 + idx[2] * 10 + idx[3])
        transposed = _transpose(data, 1, 2)  # [2,2,3,4]
        reshaped = _reshape(data, (2, 2, 3, 4))
        self.assertNotEqual(_flatten(transposed), _flatten(reshaped))

    def test_qk_transpose_matmul_matches_explicit_loop(self):
        q = [[[[1, 2, 3], [4, 5, 6]]]]  # [1,1,2,3]
        k = [[[[1, 0, 1], [0, 1, 1], [1, 1, 1], [2, 0, 0]]]]  # [1,1,4,3]
        kt = _transpose(k, 2, 3)  # [1,1,3,4]
        queries, keys, dims = 2, 4, 3
        expected = [
            [
                [
                    [sum(q[0][0][s][d] * k[0][0][t][d] for d in range(dims)) for t in range(keys)]
                    for s in range(queries)
                ]
            ]
        ]
        got = [
            [
                [
                    [sum(q[0][0][s][d] * kt[0][0][d][t] for d in range(dims)) for t in range(keys)]
                    for s in range(queries)
                ]
            ]
        ]
        self.assertEqual(got, expected)

    def test_split_concat_roundtrip(self):
        data = _tensor((2, 4, 3), lambda idx: idx[0] * 100 + idx[1] * 10 + idx[2])
        parts = [[row[:2] for row in batch] for batch in data], [[row[2:] for row in batch] for batch in data]
        rebuilt = [
            [a + b for a, b in zip(left, right)]
            for left, right in zip(parts[0], parts[1])
        ]
        self.assertEqual(_flatten(rebuilt), _flatten(data))




# ---------------------------------------------------------------------------
# 返修回归：Codex round-1 反例（条件保持 / R5 真实重排 / R7 操作数角色 / 探针）
# ---------------------------------------------------------------------------


class TestRoundOneCounterexamples(unittest.TestCase):
    def _guarded_identity(self):
        bundle = _build(
            {
                "tensors": [
                    {"id": "x", "axes": ("batch_size", "seqlen", "head_dim")},
                    {"id": "y", "axes": ("batch_size", "seqlen", "head_dim")},
                ],
                "nodes": [
                    {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
                    {"id": "view", "kind": "view", "inputs": (("x", "x"),), "outputs": (("y", "y"),)},
                    {"id": "out", "kind": "external_out", "inputs": (("x", "y"),)},
                ],
                "edges": [_edge("input", "x", "view", "x"), _edge("view", "y", "out", "x")],
            }
        )
        from dataclasses import replace
        from structure_reader.graph_ir import Guard

        bundle = replace(
            bundle,
            graph=replace(
                bundle.graph,
                guards=(Guard("g", "kvcachelen > 0", ("model.py:1",)),),
                edges=tuple(
                    replace(edge, guard_ref="g") if edge.dst_node_id == "out" else edge
                    for edge in bundle.graph.edges
                ),
            ),
        )
        return bundle

    def test_guard_is_preserved_when_identity_is_removed(self):
        result = normalize(self._guarded_identity())
        edges = [
            (edge.src_node_id, edge.dst_node_id, edge.guard_ref)
            for edge in result.normalized.graph.edges
        ]
        self.assertIn(("input", "out", "g"), edges)
        self.assertEqual(validate_normalization(result), ())

    def test_conflicting_guards_reject_rewrite(self):
        from dataclasses import replace
        from structure_reader.graph_ir import Guard

        bundle = self._guarded_identity()
        bundle = replace(
            bundle,
            graph=replace(
                bundle.graph,
                guards=(Guard("g", "kvcachelen > 0", ("model.py:1",)), Guard("h", "seqlen == 1", ("model.py:2",))),
                edges=tuple(
                    replace(edge, guard_ref="h")
                    if edge.src_node_id == "input"
                    else edge
                    for edge in bundle.graph.edges
                ),
            ),
        )
        result = normalize(bundle)
        report = {item.rule_id: item for item in result.reports}["R2-identity-eliminate"]
        self.assertEqual(report.status, "rejected")
        self.assertIn("view", result.normalized.nodes)
        self.assertIn("无法同时保持两个条件", json.dumps(report.rejected_candidates, ensure_ascii=False))

    def test_unit_axis_rule_rejects_real_non_unit_reorder(self):
        """Codex 反例：View 改变了非单位轴相对顺序时，Transpose 不能化为 Reshape。"""
        bundle = _build(
            {
                "tensors": [
                    {"id": "x", "axes": ("batch_size", "seqlen", "head_dim")},
                    {"id": "v", "axes": ("batch_size", "head_dim", "seqlen", "num_key_value_heads")},
                    {"id": "z", "axes": ("batch_size", "num_key_value_heads", "seqlen", "head_dim")},
                ],
                "nodes": [
                    {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
                    {"id": "view", "kind": "view", "inputs": (("x", "x"),), "outputs": (("y", "v"),)},
                    {"id": "trans", "kind": "transpose", "inputs": (("x", "v"),), "outputs": (("y", "z"),)},
                    {"id": "out", "kind": "external_out", "inputs": (("x", "z"),)},
                ],
                "edges": [
                    _edge("input", "x", "view", "x"),
                    _edge("view", "y", "trans", "x"),
                    _edge("trans", "y", "out", "x"),
                ],
            }
        )
        result = normalize(bundle)
        report = {item.rule_id: item for item in result.reports}["R5-unit-axis-reorder"]
        self.assertEqual(report.status, "rejected")
        self.assertIn("view", result.normalized.nodes)
        self.assertIn("trans", result.normalized.nodes)
        self.assertEqual(validate_normalization(result), ())
        # 独立索引映射：真实链不是恒等
        from structure_reader.semantic import layout_probe

        axes_by_symbol = {
            axis.symbol: axis
            for tensor in bundle.graph.tensors
            for axis in tensor.axes
        }
        extents = layout_probe.extents_for(
            ("batch_size", "seqlen", "head_dim", "num_key_value_heads"),
            axes_by_symbol,
        )
        ok, detail = layout_probe.identity_chain(
            ("batch_size", "seqlen", "head_dim"),
            [
                ("view", ("batch_size", "head_dim", "seqlen", "num_key_value_heads")),
                ("transpose", ("batch_size", "num_key_value_heads", "seqlen", "head_dim")),
            ],
            extents,
        )
        self.assertFalse(ok, detail)

    def test_r7_left_operand_keeps_transpose_role(self):
        bundle = _build(
            {
                "tensors": [
                    {"id": "a", "axes": ("batch_size", "head_dim", "seqlen")},
                    {"id": "at", "axes": ("batch_size", "seqlen", "head_dim")},
                    {"id": "b", "axes": ("batch_size", "head_dim", "key_length")},
                    {"id": "c", "axes": ("batch_size", "seqlen", "key_length")},
                ],
                "nodes": [
                    {"id": "a", "kind": "external", "outputs": (("x", "a"),)},
                    {"id": "b", "kind": "external", "outputs": (("x", "b"),)},
                    {"id": "at", "kind": "transpose", "inputs": (("x", "a"),), "outputs": (("y", "at"),)},
                    {
                        "id": "mm",
                        "kind": "matmul",
                        "inputs": (("lhs", "at"), ("rhs", "b")),
                        "outputs": (("y", "c"),),
                    },
                    {"id": "out", "kind": "external_out", "inputs": (("x", "c"),)},
                ],
                "edges": [
                    _edge("a", "x", "at", "x"),
                    _edge("at", "y", "mm", "lhs"),
                    _edge("b", "x", "mm", "rhs"),
                    _edge("mm", "y", "out", "x"),
                ],
            }
        )
        result = normalize(bundle)
        meta = result.normalized.node_meta["mm"]
        self.assertEqual(meta.attributes.get("transposed_operand"), "lhs")
        self.assertEqual(meta.attributes.get("lhs_transposed_axes"), "last_two_axes")
        self.assertNotIn("rhs_transposed_axes", meta.attributes)
        self.assertIn("Aᵀ @ B", meta.label)
        self.assertNotIn("Q @ Kᵀ", meta.label)

    def test_r7_rejects_unsupported_transpose_port(self):
        bundle = _build(
            {
                "tensors": [
                    {"id": "a", "axes": ("batch_size", "head_dim", "seqlen")},
                    {"id": "at", "axes": ("batch_size", "seqlen", "head_dim")},
                    {"id": "b", "axes": ("batch_size", "head_dim", "key_length")},
                    {"id": "c", "axes": ("batch_size", "seqlen", "key_length")},
                ],
                "nodes": [
                    {"id": "a", "kind": "external", "outputs": (("x", "a"),)},
                    {"id": "b", "kind": "external", "outputs": (("x", "b"),)},
                    {"id": "at", "kind": "transpose", "inputs": (("x", "a"),), "outputs": (("y", "at"),)},
                    {
                        "id": "mm",
                        "kind": "matmul",
                        "inputs": (("lhs", "b"), ("rhs", "b"), ("aux", "at")),
                        "outputs": (("y", "c"),),
                    },
                    {"id": "out", "kind": "external_out", "inputs": (("x", "c"),)},
                ],
                "edges": [
                    _edge("a", "x", "at", "x"),
                    _edge("at", "y", "mm", "aux"),
                    _edge("b", "x", "mm", "lhs"),
                    _edge("mm", "y", "out", "x"),
                ],
            }
        )
        result = normalize(bundle)
        report = {item.rule_id: item for item in result.reports}["R7-transpose-into-matmul"]
        self.assertEqual(report.status, "rejected")
        self.assertIn("矩阵操作数端口", json.dumps(report.rejected_candidates, ensure_ascii=False))

    def test_layout_rewrites_record_index_probe(self):
        result = normalize(_linear_bundle(transpose_pair=False))
        for report in result.reports:
            if report.status == "applied":
                self.assertTrue(report.applications)
                for application in report.applications:
                    self.assertTrue(application.probe, report.rule_id)
        self.assertEqual(validate_normalization(result), ())




class TestRejectedRewriteLeavesGraphUnchanged(unittest.TestCase):
    def test_multi_consumer_first_ok_second_conflict_keeps_original(self):
        """Codex round-2 反例：先可接受、后冲突的拒绝不得留下部分修改。"""
        from dataclasses import replace
        from structure_reader.graph_ir import Guard

        bundle = _build(
            {
                "tensors": [
                    {"id": "x", "axes": ("batch_size", "seqlen", "head_dim")},
                    {"id": "y", "axes": ("batch_size", "seqlen", "head_dim")},
                ],
                "nodes": [
                    {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
                    {"id": "view", "kind": "view", "inputs": (("x", "x"),), "outputs": (("y", "y"),)},
                    {"id": "out1", "kind": "external_out", "inputs": (("x", "y"),)},
                    {"id": "out2", "kind": "external_out", "inputs": (("x", "y"),)},
                ],
                "edges": [
                    _edge("input", "x", "view", "x"),
                    _edge("view", "y", "out1", "x"),
                    _edge("view", "y", "out2", "x"),
                ],
            }
        )
        bundle = replace(
            bundle,
            graph=replace(
                bundle.graph,
                guards=(Guard("g", "kvcachelen > 0", ("model.py:1",)), Guard("h", "seqlen == 1", ("model.py:2",))),
                edges=tuple(
                    replace(edge, guard_ref="g")
                    if edge.dst_node_id == "view"
                    else replace(edge, guard_ref="h")
                    if edge.dst_node_id == "out2"
                    else edge
                    for edge in bundle.graph.edges
                ),
            ),
        )
        before_edges = sorted(
            (edge.stable_id, edge.src_node_id, edge.src_port, edge.dst_node_id, edge.dst_port, edge.tensor_id, edge.guard_ref)
            for edge in bundle.graph.edges
        )
        before_nodes = sorted(
            (node.stable_id, tuple((p.name, p.direction, p.tensor_id) for p in node.input_ports), tuple((p.name, p.direction, p.tensor_id) for p in node.output_ports))
            for node in bundle.graph.nodes
        )
        result = normalize(bundle)
        report = {item.rule_id: item for item in result.reports}["R2-identity-eliminate"]
        self.assertEqual(report.status, "rejected")
        after_edges = sorted(
            (edge.stable_id, edge.src_node_id, edge.src_port, edge.dst_node_id, edge.dst_port, edge.tensor_id, edge.guard_ref)
            for edge in result.normalized.graph.edges
        )
        after_nodes = sorted(
            (node.stable_id, tuple((p.name, p.direction, p.tensor_id) for p in node.input_ports), tuple((p.name, p.direction, p.tensor_id) for p in node.output_ports))
            for node in result.normalized.graph.nodes
        )
        self.assertEqual(before_edges, after_edges)
        self.assertEqual(before_nodes, after_nodes)
        self.assertEqual(validate_normalization(result), ())

    def test_rejected_rule_keeps_normalized_graph_identical_to_raw(self):
        from dataclasses import replace
        from structure_reader.graph_ir import Guard

        bundle = _build(
            {
                "tensors": [
                    {"id": "x", "axes": ("batch_size", "seqlen", "head_dim")},
                    {"id": "y", "axes": ("batch_size", "seqlen", "head_dim")},
                ],
                "nodes": [
                    {"id": "input", "kind": "external", "outputs": (("x", "x"),)},
                    {"id": "view", "kind": "view", "inputs": (("x", "x"),), "outputs": (("y", "y"),)},
                    {"id": "out", "kind": "external_out", "inputs": (("x", "y"),)},
                ],
                "edges": [_edge("input", "x", "view", "x"), _edge("view", "y", "out", "x")],
            }
        )
        bundle = replace(
            bundle,
            graph=replace(
                bundle.graph,
                guards=(Guard("g", "a > 0", ("m:1",)), Guard("h", "b > 0", ("m:2",))),
                edges=tuple(
                    replace(edge, guard_ref="g")
                    if edge.dst_node_id == "view"
                    else replace(edge, guard_ref="h")
                    if edge.dst_node_id == "out"
                    else edge
                    for edge in bundle.graph.edges
                ),
            ),
        )
        result = normalize(bundle)
        self.assertIn("view", result.normalized.nodes)
        self.assertEqual(
            sorted(json.dumps(edge.to_dict(), sort_keys=True) for edge in result.raw.graph.edges),
            sorted(json.dumps(edge.to_dict(), sort_keys=True) for edge in result.normalized.graph.edges),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
