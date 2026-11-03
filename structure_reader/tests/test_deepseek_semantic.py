"""DeepSeek 四组层语义投影测试：来源、全名轴、旧事实覆盖、raw/归一化结果。

运行：cd structure_reader && python3 tests/test_deepseek_semantic.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODELING = (
    REPO_ROOT
    / "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py"
)


def _ready() -> bool:
    return (REPO_ROOT / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json").is_file() and MODELING.is_file()


@unittest.skipUnless(_ready(), "缺少只读 HF 输入；如实跳过")
class TestSemanticProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_layer_structure
        from structure_reader.extract.deepseek_v4_semantic import (
            build_all_layer_group_bundles,
            legacy_coverage,
        )
        from structure_reader.semantic.bundle import validate_bundle
        from structure_reader.semantic.rules import normalize, validate_normalization

        cls.build_layer_structure = staticmethod(build_layer_structure)
        cls.bundles = build_all_layer_group_bundles(REPO_ROOT)
        cls.results = {
            variant: normalize(
                bundle,
                facts=[tuple(fact) for fact in bundle.metadata.get("shape_facts", [])],
            )
            for variant, bundle in cls.bundles.items()
        }
        cls.validate_bundle = staticmethod(validate_bundle)
        cls.validate_normalization = staticmethod(validate_normalization)
        cls.legacy_coverage = staticmethod(legacy_coverage)

    def test_four_layer_groups_built(self):
        self.assertEqual(
            sorted(self.bundles), ["csa-hash", "csa-topk", "hca-topk", "swa-hash"]
        )
        self.assertEqual(
            [self.bundles[key].metadata["layer_index"] for key in sorted(self.bundles)],
            [2, 4, 3, 0],
        )

    def test_raw_bundle_and_normalization_validate(self):
        for variant, bundle in self.bundles.items():
            block = self.build_layer_structure(
                REPO_ROOT, bundle.metadata["layer_index"]
            )["block"]
            self.assertEqual(self.validate_bundle(bundle), (), variant)
            self.assertEqual(self.legacy_coverage(bundle, block), (), variant)
            result = self.results[variant]
            self.assertEqual(self.validate_normalization(result), (), variant)

    def test_every_old_operator_is_mapped(self):
        for variant, bundle in self.bundles.items():
            block = self.build_layer_structure(
                REPO_ROOT, bundle.metadata["layer_index"]
            )["block"]
            covered = {
                ref
                for meta in bundle.node_meta.values()
                for ref in meta.legacy_refs
                if ref.startswith("op:")
            }
            for operator in block["operators"]:
                self.assertIn(f"op:{operator['id']}", covered, f"{variant}:{operator['id']}")

    def test_kv_proj_and_rmsnorm_are_separate_nodes(self):
        for variant, bundle in self.bundles.items():
            kinds = {
                node_id: meta.kind for node_id, meta in bundle.node_meta.items()
            }
            self.assertEqual(kinds.get("attn.kv_proj"), "linear", variant)
            self.assertEqual(kinds.get("attn.kv_norm"), "rmsnorm", variant)
            self.assertTrue(
                {"tensor:kv_weight", "tensor:kv_weight_attn"}
                & set(bundle.node_meta["attn.kv_proj"].legacy_refs)
            )
            self.assertIn("tensor:kv_norm_weight", bundle.node_meta["attn.kv_norm"].legacy_refs)

    def test_view_and_transpose_are_distinct_nodes(self):
        bundle = self.bundles["csa-topk"]
        self.assertEqual(bundle.node_meta["attn.q_view"].kind, "view")
        self.assertEqual(bundle.node_meta["attn.q_transpose"].kind, "transpose")
        self.assertEqual(bundle.node_meta["attn.kv_view"].kind, "view")
        self.assertEqual(bundle.node_meta["attn.kv_transpose"].kind, "transpose")
        self.assertEqual(bundle.node_meta["attn.q_rope"].kind, "rope")

    def test_axes_use_full_names_and_no_alias_symbols(self):
        bundle = self.bundles["csa-topk"]
        aliases = set(bundle.axes.aliases())
        for tensor in bundle.graph.tensors:
            for axis in tensor.axes:
                self.assertNotIn(axis.symbol, aliases, axis.symbol)
                self.assertIn(axis.symbol, bundle.axes.symbols(), axis.symbol)

    def test_kv_preview_source_uses_seqlen_not_kvcachelen(self):
        bundle = self.bundles["csa-topk"]
        result = self.results["csa-topk"]
        raw_kv_shape = bundle.shape_text("tensor:kv_rope")
        normalized_kv_shape = result.normalized.shape_text("tensor:kv_rope")
        self.assertIn("seqlen", raw_kv_shape)
        self.assertNotIn("kvcachelen", raw_kv_shape)
        self.assertIn("seqlen", normalized_kv_shape)
        self.assertNotIn("kvcachelen", normalized_kv_shape)

    def test_history_length_only_in_cache_and_attention_key_axis(self):
        bundle = self.bundles["csa-topk"]
        cache_axis = {axis.symbol: axis for axis in bundle.tensors["tensor:cache_keys"].axes}["kv_cache_len"]
        self.assertIn("kvcachelen", cache_axis.expression)
        full_axis = {axis.symbol: axis for axis in bundle.tensors["tensor:kv_full"].axes}["kv_length"]
        self.assertIn("kvcachelen", full_axis.expression)
        self.assertIn("seqlen", full_axis.expression)
        # 主图显示使用定义过的逻辑长度名；正文/卡片给出 kvcachelen 公式
        self.assertIn("kv_cache_len", bundle.shape_text("tensor:cache_keys"))

    def test_single_kv_head_displayed_as_one(self):
        bundle = self.bundles["csa-topk"]
        self.assertIn("[batch_size, 1, seqlen, head_dim]", bundle.shape_text("tensor:kv_head"))

    def test_rules_applied_and_rejected_with_reasons(self):
        bundle = self.bundles["csa-topk"]
        result = self.results["csa-topk"]
        reports = {report.rule_id: report for report in result.reports}
        for rule_id in (
            "R1-shape-expression-simplify",
            "R3-transpose-pair-cancel",
            "R5-unit-axis-reorder",
            "R7-transpose-into-matmul",
        ):
            self.assertEqual(reports[rule_id].status, "applied", rule_id)
            self.assertTrue(reports[rule_id].applications, rule_id)
        for rule_id in (
            "R2-identity-eliminate",
            "R4-reshape-merge",
            "R6-split-concat-cancel",
        ):
            self.assertEqual(reports[rule_id].status, "rejected", rule_id)
            self.assertTrue(reports[rule_id].reason, rule_id)
            self.assertTrue(
                reports[rule_id].rejected_candidates
                or "没有" in reports[rule_id].reason,
                rule_id,
            )
        # R5 必须保留三维→四维变化（属性落在化简后的节点上）
        r5 = reports["R5-unit-axis-reorder"].applications[0]
        new_node = r5.added_nodes[0]
        self.assertEqual(
            result.normalized.node_meta[new_node].attributes.get("rank_change"),
            "3->4",
        )
        # R7 保留转置轴语义在 MatMul 节点上，并保留被并入 Transpose 的来源
        r7 = reports["R7-transpose-into-matmul"].applications[0]
        meta = result.normalized.node_meta["attn.qk_matmul"]
        self.assertEqual(meta.attributes.get("rhs_transposed_axes"), "last_two_axes")
        self.assertIn("attn.kv_transposed", meta.attributes.get("rhs_transpose_source", ""))
        self.assertTrue(r7.sources)

    def test_q_multi_head_is_not_reduced_to_reshape(self):
        bundle = self.bundles["csa-topk"]
        result = self.results["csa-topk"]
        report = {
            item.rule_id: item for item in result.reports
        }["R5-unit-axis-reorder"]
        reasons = json.dumps([item for item in report.rejected_candidates], ensure_ascii=False)
        self.assertIn("改变了非单位轴的相对顺序", reasons)
        self.assertIn("attn.q_view", result.normalized.nodes)
        self.assertIn("attn.q_transpose", result.normalized.nodes)

    def test_normalized_keeps_shared_consumers_and_state(self):
        result = self.results["csa-topk"]
        normalized = result.normalized
        # K=V：同一 kv 张量同时被 QK 与 AV 消费（通过复用入口在图中标出）
        repeat = [edge for edge in normalized.graph.edges if edge.src_node_id == "attn.repeat_kv"]
        self.assertEqual(len(repeat), 2)
        # 三个状态转移都保留且都有 state 边引用
        transitions = {item.stable_id for item in normalized.graph.state_transitions}
        referenced = {
            edge.state_transition_ref
            for edge in normalized.graph.edges
            if edge.state_transition_ref
        }
        self.assertEqual(transitions, referenced)

    def test_has_compressor_flags_match_layer_type(self):
        self.assertFalse(self.bundles["swa-hash"].metadata["has_compressor"])
        self.assertFalse(self.bundles["swa-hash"].metadata["has_indexer"])
        self.assertTrue(self.bundles["hca-topk"].metadata["has_compressor"])
        self.assertFalse(self.bundles["hca-topk"].metadata["has_indexer"])
        for variant in ("csa-hash", "csa-topk"):
            self.assertTrue(self.bundles[variant].metadata["has_indexer"])

    def test_sources_point_into_modeling_file_within_range(self):
        bundle = self.bundles["csa-topk"]
        total_lines = len(MODELING.read_text(encoding="utf-8").splitlines())
        for node_id, meta in bundle.node_meta.items():
            for source in meta.sources:
                self.assertTrue(source.path.startswith("src/transformers/"), node_id)
                self.assertGreaterEqual(source.line, 1, node_id)
                self.assertLessEqual(source.line, total_lines + 1, node_id)
                if source.end_line is not None:
                    self.assertGreaterEqual(source.end_line, source.line, node_id)

    def test_semantic_bundle_json_roundtrip(self):
        from structure_reader.semantic.bundle import SemanticBundle

        bundle = self.bundles["csa-topk"]
        restored = SemanticBundle.from_json(bundle.to_json())
        self.assertEqual(
            json.dumps(restored.to_json(), sort_keys=True, ensure_ascii=False),
            json.dumps(bundle.to_json(), sort_keys=True, ensure_ascii=False),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
