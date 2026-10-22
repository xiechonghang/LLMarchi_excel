"""gate_repairs/task02 红测：G3 两套 Compressor 状态链核验 + 已知轴缺口。"""

from __future__ import annotations

import sys
import unittest
from dataclasses import fields as dc_fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.model_registry import find_workspace_root

ROOT = find_workspace_root()


def build():
    from structure_reader.extract.deepseek_v4_reference import (
        build_reference_graph)
    return build_reference_graph(ROOT)


def rebuild(graph, **overrides):
    from structure_reader.graph_document import SemanticGraph
    kwargs = {f.name: getattr(graph, f.name) for f in dc_fields(SemanticGraph)}
    kwargs.update(overrides)
    return SemanticGraph(**kwargs)


def drop_edges(graph, predicate):
    kept = [e for e in graph.edges if not predicate(e)]
    return rebuild(graph, edges=kept)


def drop_nodes(graph, node_ids):
    kept_nodes = [n for n in graph.nodes if n.stable_id not in node_ids]
    kept_edges = [e for e in graph.edges
                  if e.src_node_id not in node_ids
                  and e.dst_node_id not in node_ids]
    return rebuild(graph, nodes=kept_nodes, edges=kept_edges)


def reguard_edge(graph, edge_id, new_guard):
    from structure_reader.graph_ir import Edge
    new_edges = [
        Edge(stable_id=e.stable_id, src_node_id=e.src_node_id,
             src_port=e.src_port, dst_node_id=e.dst_node_id,
             dst_port=e.dst_port, tensor_id=e.tensor_id, kind=e.kind,
             evidence_refs=e.evidence_refs, guard_ref=new_guard,
             state_transition_ref=e.state_transition_ref)
        if e.stable_id == edge_id else e
        for e in graph.edges
    ]
    return rebuild(graph, edges=new_edges)


def retable_axis(graph, tensor_name, new_symbol):
    from structure_reader.graph_ir import Tensor
    axis_by = {a.symbol: a for a in graph.named_axes}
    new_tensors = []
    for t in graph.tensors:
        if t.name == tensor_name:
            symbols = [a.symbol for a in t.axes]
            symbols[-1] = new_symbol
            axes = [axis_by[s] for s in symbols]
            t = Tensor(name=t.name, dtype=t.dtype, axes=axes,
                       logical_layout=symbols, physical_layout=symbols)
        new_tensors.append(t)
    return rebuild(graph, tensors=new_tensors)


class TestG3BothCompressors(unittest.TestCase):
    """G3：C12/C13/C14 状态链检查必须覆盖两套 compressor（非硬编码第一套）。"""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.reg, cls.vals = build()
        cls.validate = _validate()

    def test_second_compressor_all_checks_active(self):
        # 正常图两套都零问题（本测试同时确认第二套节点存在）
        nodes = {n.stable_id for n in self.graph.nodes}
        for required in (
            "attn.indexer.compressor.state_update",
            "attn.indexer.compressor.pool_shift",
            "attn.indexer.compressor.publish_final",
            "attn.indexer.compressor.pool_decode",
        ):
            self.assertIn(required, nodes, required)

    def test_idx_delete_shift_publish_detected(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        mutated = drop_edges(
            self.graph,
            lambda e: e.stable_id == "attn.indexer.compressor.shift_publish")
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertIn("C13_FINAL_PUBLISH_SOURCE", codes,
                      "第二套删 shift_publish 必须检出")

    def test_idx_delete_updated_publish_detected(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        mutated = drop_edges(
            self.graph,
            lambda e: e.stable_id == "attn.indexer.compressor.updated_publish")
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertIn("C13_FINAL_PUBLISH_SOURCE", codes,
                      "第二套删 updated_publish 必须检出")

    def test_idx_drop_phase_edge_detected(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        mutated = drop_edges(
            self.graph,
            lambda e: e.stable_id == "attn.indexer.compressor.updated_publish_dec")
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertIn("C13_FINAL_PUBLISH_SOURCE", codes,
                      "第二套删相位边必须检出")

    def test_idx_shift_output_dropped_detected(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        mutated = drop_edges(
            self.graph,
            lambda e: e.src_node_id == "attn.indexer.compressor.pool_shift")
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertTrue(
            codes & {"C12_SHIFT_UNCONNECTED", "C13_FINAL_PUBLISH_SOURCE"},
            f"第二套 shift 输出断开必须检出，得到 {codes}")

    def test_idx_pool_control_dependency_dropped(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        mutated = drop_edges(
            self.graph,
            lambda e: e.stable_id == "attn.indexer.compressor.pool_done_shift")
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertIn("C13_SHIFT_NO_POOL_ORDER", codes,
                      "第二套删 pool 控制依赖必须检出")

    def test_idx_publish_source_wrong_detected(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        # 把第二套 final 端口来源改成 state_update（错误来源）
        def swap_src(e):
            if e.stable_id == "attn.indexer.compressor.shift_publish":
                return type(e)(stable_id=e.stable_id,
                               src_node_id="attn.indexer.compressor.state_update",
                               src_port=e.src_port, dst_node_id=e.dst_node_id,
                               dst_port=e.dst_port, tensor_id=e.tensor_id,
                               kind=e.kind, evidence_refs=e.evidence_refs,
                               guard_ref=e.guard_ref,
                               state_transition_ref=e.state_transition_ref)
            return e
        from structure_reader.graph_ir import Edge
        new_edges = [swap_src(e) for e in self.graph.edges]
        mutated = rebuild(self.graph, edges=new_edges)
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertIn("C13_FINAL_PUBLISH_SOURCE", codes,
                      "第二套发布来源错误必须检出")

    def test_second_compressor_dims_not_confused(self):
        # 两套维度/量化差异保留：idx 侧量化 op 与张量前缀不混淆
        idx_quant = next(n for n in self.graph.nodes
                         if n.stable_id == "attn.indexer.compressor.quant")
        self.assertEqual(idx_quant.op_type, "hadamard_fp4_quant")
        idx_edges = [e for e in self.graph.edges
                     if e.src_node_id.startswith("attn.indexer.compressor.")]
        for e in idx_edges:
            if "comp_" in e.tensor_id and "idx_comp" not in e.tensor_id:
                # idx 侧不得消费 attention 的 comp_* 中间张量
                self.fail(f"{e.stable_id} 消费 {e.tensor_id}")


class TestG2ResidualAxes(unittest.TestCase):
    """G2-residual：kv_concat/x_attn_normed/gate_indices/APE 切片轴契约。"""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.reg, cls.vals = build()

    def test_expected_axes_current(self):
        tensors = {t.name: t for t in self.graph.tensors}
        self.assertEqual(tensors["csa_topk:kv_concat"].axes[-1].symbol, "D")
        self.assertEqual(tensors["csa_topk:x_attn_normed"].axes[-1].symbol, "H")
        self.assertEqual(tensors["csa_topk:gate_indices"].axes[-1].symbol, "K")
        self.assertEqual(len(tensors["csa_topk:comp_ape"].axes), 2)
        self.assertEqual(tensors["csa_topk:comp_ape_tail_rows"].axes[0].symbol,
                         "N_rem")
        self.assertEqual(tensors["csa_topk:comp_ape_dec_row"].axes[0].symbol,
                         "N_one")

    def test_kv_concat_axis_mutated_detected(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        mutated = retable_axis(self.graph, "csa_topk:kv_concat", "D2")
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertTrue(any(c.startswith("C17_") for c in codes),
                        f"kv_concat 末轴错改未检出，得到 {codes}")

    def test_gate_indices_axis_mutated_detected(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        mutated = retable_axis(self.graph, "csa_topk:gate_indices", "E")
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertTrue(any(c.startswith("C17_") for c in codes),
                        f"gate_indices 末轴错改未检出，得到 {codes}")

    def test_ape_row_slice_axis_mutated_detected(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        mutated = retable_axis(self.graph, "csa_topk:comp_ape_tail_rows", "r")
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertTrue(any(c.startswith("C17_") for c in codes),
                        f"APE 行切片轴错改未检出，得到 {codes}")

    def test_unknown_axes_preserved(self):
        # N_e / C 保持 unknown（不得以固定 shape 假装全部轴相等）
        for sym in ("N_e", "C"):
            a = next(a for a in self.graph.named_axes if a.symbol == sym)
            self.assertIsNone(a.value, sym)
            self.assertIsNone(a.expression, sym)
            self.assertTrue(a.unknown_reason and a.unknown_reason.strip())


def _validate():
    from structure_reader.checks.deepseek_v4_contracts import (
        validate_model_contracts)
    return validate_model_contracts


if __name__ == "__main__":
    unittest.main()
