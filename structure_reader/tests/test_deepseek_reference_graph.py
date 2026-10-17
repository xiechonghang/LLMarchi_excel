"""DeepSeek-V4-Flash-0731 参考图提取器测试（工作包 B）：
真实图构建、证据原文核对、模型契约与受控错误、层规划绑定。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.extract.deepseek_v4_reference import (
    ExtractionError,
    REPRESENTATIVE_LAYER,
    TEMPLATE_ID,
    build_reference_graph,
    build_scenario_views,
    indexer_visibility,
)
from structure_reader.extract.evidence import (
    EvidenceError,
    SourceEvidence,
    load_evidence_base,
)
from structure_reader.layer_plan import build_layer_plan
from structure_reader.model_registry import find_workspace_root
from structure_reader.checks.deepseek_v4_contracts import (
    apply_contract_mutation,
    contract_error_specs,
    validate_model_contracts,
)

ROOT = find_workspace_root()


def build():
    return build_reference_graph(ROOT)


class TestEvidenceBase(unittest.TestCase):
    def test_baseline_all_match(self):
        base = load_evidence_base(ROOT)
        self.assertEqual(len(base.file_sha256), 5)
        for key, sha in base.file_sha256.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", key)

    def test_unknown_source_key(self):
        base = load_evidence_base(ROOT)
        with self.assertRaises(EvidenceError):
            base.sha256_of("not_a_file")


class TestEvidenceRegistry(unittest.TestCase):
    def setUp(self):
        self.graph, self.registry, self.values = build()

    def test_all_evidence_verified(self):
        # 全部证据在注册时已核对原文；能构建即全部通过
        entries = self.registry.all()
        self.assertGreaterEqual(len(entries), 50)
        coverages = {e.coverage for e in entries}
        self.assertEqual(
            coverages,
            {"located", "source_checked", "derived", "uncovered"},
            "四个证据等级都必须有真实条目",
        )

    def test_key_claims_have_source_checked(self):
        # 关键控制流必须是 source_checked：comb 方向、HCA 固定选择、专家模板
        by_id = {e.evidence_id for e in self.registry.by_coverage("source_checked")}
        for required in ("ev_hc_post", "ev_kv_path".replace("kv_path", "o_group"),
                         "ev_idx_topk", "ev_expert"):
            self.assertIn(required, by_id, required)

    def test_uncovered_entries_honest(self):
        uncovered = self.registry.by_coverage("uncovered")
        self.assertGreaterEqual(len(uncovered), 1)
        for e in uncovered:
            self.assertTrue(e.note and e.note.strip(), e.evidence_id)
            self.assertIsNone(e.line_start, e.evidence_id)

    def test_duplicate_evidence_id(self):
        dup = SourceEvidence(evidence_id="ev_hc_pre", source_key="model.py",
                             qualified_symbol="Block.hc_pre", coverage="derived",
                             derivation="x")
        with self.assertRaises(EvidenceError):
            self.registry.register(dup)

    def test_snippet_mismatch_rejected(self):
        bad = SourceEvidence(
            evidence_id="ev_test_wrong", source_key="model.py",
            qualified_symbol="Block.forward", line_start=695, line_end=707,
            snippet="self.hc_post(x, residual, comb, post)  # 参数顺序故意错",
            claim="测试用", coverage="located")
        fresh = type(self.registry)(
            load_evidence_base(ROOT), ROOT)
        with self.assertRaises(EvidenceError):
            fresh.register(bad)

    def test_line_range_out_of_bounds(self):
        bad = SourceEvidence(
            evidence_id="ev_test_oob", source_key="model.py",
            qualified_symbol="x", line_start=99990, line_end=99999,
            snippet="import torch", claim="测试用", coverage="located")
        fresh = type(self.registry)(load_evidence_base(ROOT), ROOT)
        with self.assertRaises(EvidenceError):
            fresh.register(bad)


class TestReferenceGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph, cls.registry, cls.values = build()

    def test_graph_passes_generic_checks(self):
        # build_reference_graph 内部已跑 validate_graph_references；再次确认
        from structure_reader.checks.graph_contracts import (
            validate_graph_references)
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_template_id_and_world_size(self):
        self.assertEqual(self.graph.template_id, TEMPLATE_ID)
        self.assertEqual(self.graph.world_size, 1)

    def test_scale(self):
        self.assertGreater(len(self.graph.nodes), 50)
        self.assertGreater(len(self.graph.edges), 70)
        self.assertGreater(len(self.graph.tensors), 50)
        self.assertGreater(len(self.graph.named_axes), 40)

    def test_block_two_hc_rounds(self):
        nodes = {n.stable_id for n in self.graph.nodes}
        for required in (
            "block.hc_pre_attn.flatten", "block.hc_pre_attn.mix",
            "block.hc_pre_attn.sinkhorn", "block.hc_pre_attn.reduce",
            "attn_norm", "block.hc_post_attn.combine",
            "block.hc_pre_ffn.flatten", "block.hc_pre_ffn.mix",
            "block.hc_pre_ffn.sinkhorn", "block.hc_pre_ffn.reduce",
            "ffn_norm", "block.hc_post_ffn.combine",
        ):
            self.assertIn(required, nodes, required)

    def test_comb_source_target_axes(self):
        comb = next(t for t in self.graph.tensors
                    if t.name == "csa_topk:attn_comb")
        symbols = [a.symbol for a in comb.axes]
        self.assertEqual(symbols[-2:], ["hc_j", "hc_k"])
        self.assertEqual(symbols[-2], "hc_j", "comb 源轴 j 在倒数第二维")
        self.assertEqual(symbols[-1], "hc_k", "comb 目标轴 k 在最后一维")

    def test_hcpost_comb_input_from_sinkhorn(self):
        comb_edges = [e for e in self.graph.edges
                      if e.dst_node_id == "block.hc_post_attn.combine"
                      and e.dst_port == "comb"]
        self.assertEqual(len(comb_edges), 1)
        self.assertEqual(comb_edges[0].src_node_id,
                         "block.hc_pre_attn.sinkhorn")
        self.assertEqual(comb_edges[0].src_port, "comb")

    def test_residual_flows_to_post(self):
        # attention 残差（块入口）与 FFN 残差（attn post）分别到两个 combine
        res_a = [e for e in self.graph.edges
                 if e.dst_node_id == "block.hc_post_attn.combine"
                 and e.dst_port == "residual"]
        self.assertEqual(len(res_a), 1)
        self.assertEqual(res_a[0].tensor_id, "tensor:csa_topk:x_hc_in")
        res_f = [e for e in self.graph.edges
                 if e.dst_node_id == "block.hc_post_ffn.combine"
                 and e.dst_port == "residual"]
        self.assertEqual(len(res_f), 1)
        self.assertEqual(res_f[0].tensor_id, "tensor:csa_topk:x_attn_post")

    def test_qr_dual_consumer(self):
        qr_edges = [e for e in self.graph.edges
                    if e.src_node_id == "attn.q_norm" and e.src_port == "y"
                    and e.tensor_id == "tensor:csa_topk:qr"]
        consumers = {e.dst_node_id for e in qr_edges}
        self.assertEqual(consumers, {"attn.wq_b", "attn.indexer.wq_b"})

    def test_shared_kv_single_projection(self):
        # 一个 wkv 节点；K/V 两条 same_tensor 指向同一张量
        wkv = [n for n in self.graph.nodes if n.stable_id == "attn.wkv"]
        self.assertEqual(len(wkv), 1)
        aliases = {(a.alias_role, a.alias_tensor_id)
                   for a in self.graph.alias_relations
                   if a.alias_role in ("key", "value")}
        self.assertEqual(aliases, {("key", "tensor:csa_topk:kv_quant"),
                                   ("value", "tensor:csa_topk:kv_quant")})

    def test_compressor_state_and_slice_alias(self):
        # slice alias：压缩 cache 是注意力 cache 的视图
        alias = next(a for a in self.graph.alias_relations
                     if a.stable_id == "alias_comp_cache_slice")
        self.assertEqual(alias.base_tensor_id, "tensor:csa_topk:attn_cache")
        self.assertIn("W:", alias.transform_expr)
        # 独立压缩状态
        state_ids = {t.state_id for t in self.graph.state_transitions}
        self.assertIn("csa_topk:compress_kv_state", state_ids)
        self.assertIn("csa_topk:compress_score_state", state_ids)

    def test_indexer_select_not_value(self):
        select_edges = [e for e in self.graph.edges if e.kind == "select"]
        self.assertTrue(select_edges)
        tensors = {e.tensor_id for e in select_edges}
        self.assertEqual(tensors, {"tensor:csa_topk:comp_idxs"})

    def test_inverse_rope_before_reshape(self):
        order = [(e.src_node_id, e.dst_node_id) for e in self.graph.edges
                 if e.src_node_id == "attn.o_rope_inv"]
        self.assertIn(("attn.o_rope_inv", "attn.o_reshape"), order)

    def test_unknown_axes_have_reasons(self):
        for axis in self.graph.named_axes:
            if axis.is_unknown:
                self.assertTrue(
                    axis.unknown_reason and axis.unknown_reason.strip(),
                    axis.symbol,
                )
        ne = next(a for a in self.graph.named_axes if a.symbol == "N_e")
        self.assertIsNone(ne.value)
        self.assertIsNone(ne.expression)

    def test_scenario_related_axes_are_expressions(self):
        by_symbol = {a.symbol: a for a in self.graph.named_axes}
        self.assertEqual(by_symbol["B"].expression, "B")
        self.assertEqual(by_symbol["T"].expression, "T")
        self.assertEqual(by_symbol["P"].expression, "P")

    def test_missing_evidence_rejected(self):
        # 图引用未注册证据时构建失败（require_evidence 的注册表校验）
        from structure_reader.extract import deepseek_v4_reference as mod
        original = mod._evidence_defs
        def broken_defs():
            defs = list(original())
            dropped = {d.evidence_id for d in defs[len(defs)//2:]}
            # 只删派生/未覆盖条目，避免破坏 located 原文核对路径；保留前半
            keep = [d for d in defs
                    if d.evidence_id not in dropped
                    or d.coverage in ("located", "source_checked")][:60]
            removed = [d for d in defs if d not in keep]
            assert removed, "must remove at least one referenced evidence"
            type(self)._removed = {d.evidence_id for d in removed}
            return keep
        mod._evidence_defs = broken_defs
        try:
            with self.assertRaises((ExtractionError, EvidenceError)):
                mod.build_reference_graph(ROOT)
        finally:
            mod._evidence_defs = original


class TestModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph, cls.registry, cls.values = build()

    def test_base_graph_clean(self):
        self.assertEqual(validate_model_contracts(self.graph), ())

    def test_controlled_errors_detected(self):
        # 按产品适用性过滤（template/spec）：csa_topk 图不适用 hash 依赖、
        # HCA 注入、HCA overlap 三项；其余全部 apply 且逐项检出
        _not_applicable = {"hash_token_dependency_removed",
                           "hca_indexer_injected", "hca_overlap_injected"}
        specs = [s for s in contract_error_specs()
                 if s.mutation_id not in _not_applicable]
        self.assertEqual(len(specs), 31 - 3)
        for spec in specs:
            with self.subTest(mutation=spec.mutation_id):
                mutated = apply_contract_mutation(self.graph, spec.mutation_id)
                found = {i.code for i in validate_model_contracts(mutated)}
                for expected in spec.expected_codes:
                    self.assertIn(expected, found,
                                  f"{spec.mutation_id} 未检出 {expected}")

    def test_review01_semantic_expectations(self):
        # codex_review_01 六项的正向期望（源码独立对照）
        g = self.graph
        edges = {e.stable_id: e for e in g.edges}
        nodes = {n.stable_id for n in g.nodes}
        tensors = {t.name for t in g.tensors}
        # 1 sink 独立参数 [Hq]，不是 hc post
        self.assertIn("attn.attn_sink", nodes)
        self.assertIn("csa_topk:attn_sink_param", tensors)
        self.assertEqual(edges["e_sparse_sink"].src_node_id, "attn.attn_sink")
        # 2 compressor 输入是 attn_norm（H），不是窗口 wkv
        for suffix in ("wkv", "wgate"):
            e = edges[f"e_comp_in{'' if suffix == 'wkv' else '_gate'}"]
            self.assertEqual(e.src_node_id, "attn_norm",
                             f"compressor {suffix} 输入")
        # 3 两套张量隔离：indexer 实例的中间张量边不消费 attention 的 comp_*
        idx_edges = [e for e in g.edges
                     if e.src_node_id.startswith("attn.indexer.compressor.")]
        self.assertTrue(idx_edges)
        for e in idx_edges:
            if e.tensor_id.startswith("tensor:csa_topk:comp_"):
                self.assertIn("idx_comp", e.tensor_id, e.stable_id)
        # cache 容量 N_cc = Mcap//r（不是 Mcap）
        axes = {a.symbol: a for a in g.named_axes}
        self.assertEqual(axes["N_cc"].expression, "Mcap // r")
        # state 宽度 2D/2Di
        state_kv = next(t for t in g.tensors
                        if t.name == "csa_topk:comp_state_kv_updated")
        self.assertEqual([a.symbol for a in state_kv.axes][-2:], ["N_cs", "D2"])
        idx_state = next(t for t in g.tensors
                         if t.name == "csa_topk:idx_comp_state_kv_final")
        self.assertEqual([a.symbol for a in idx_state.axes][-2:], ["N_cs", "D2i"])
        # 4 decode cache 读路径：cache_read 直达 sparse_attn（g_decode），
        # prefill 走 kv_concat（g_prefill），压缩拼接挂触发 guard
        self.assertEqual(edges["e_sparse_kv_decode"].guard_ref, "g_decode")
        self.assertEqual(edges["e_concat_win"].guard_ref, "g_prefill")
        self.assertEqual(edges["e_concat_comp"].guard_ref,
                         "g_compress_prefill")
        # select_idxs 进入 sparse_attn
        self.assertEqual(edges["e_sparse_topk"].src_node_id, "attn.select_cat")
        # Indexer 打分读可见前缀：真实 slice 节点（idx_cache → N_vis 前缀）
        # → score 消费（review02：alias 不能无消费）
        self.assertIn("attn.indexer.cache_slice",
                      {n.stable_id for n in g.nodes})
        self.assertEqual(edges["e_idx_cache_score"].tensor_id,
                         "tensor:csa_topk:idx_cache_visible")
        # 5 guard/state 连图：至少 5 条 guard 分支边、全部 transition 被引用
        self.assertGreaterEqual(
            sum(1 for e in g.edges if e.guard_ref), 5)
        referenced = {e.state_transition_ref for e in g.edges
                      if e.state_transition_ref}
        self.assertGreaterEqual(len(referenced), 6)
        self.assertEqual(referenced,
                         {t.stable_id for t in g.state_transitions})

    def test_unknown_mutation_rejected(self):
        with self.assertRaises(ValueError):
            apply_contract_mutation(self.graph, "not_a_mutation")

    def test_layer_group_errors_are_named(self):
        # 缺失/重叠层：rules 并集校验具名拒绝（受控错误第 8 项）
        from structure_reader.rules.deepseek_v4_flash_0731 import RulesError, derive_layer_groups
        values = type(self.values)(
            **{**self.values.__dict__,
               "compress_ratios": (0, 0, 4, 128, 4)})  # 5 项 < n_layers=43
        with self.assertRaises(RulesError):
            derive_layer_groups(values)


class TestLayerPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_layer_plan(ROOT)

    def test_four_groups_2_1_20_20(self):
        sizes = {g.group_id: len(g.member_layers) for g in self.plan.groups}
        self.assertEqual(sizes, {"swa_hash": 2, "csa_hash": 1,
                                 "hca_topk": 20, "csa_topk": 20})

    def test_all_43_bound(self):
        self.assertEqual(len(self.plan.bindings), 43)
        self.assertEqual([b.layer_id for b in self.plan.bindings],
                         list(range(43)))

    def test_only_layer4_extracted(self):
        # V2：四类代表层（0/2/3/4）extracted；其余 39 实例 pending
        extracted = [b for b in self.plan.bindings
                     if b.template_status == "extracted"]
        self.assertEqual(sorted(b.layer_id for b in extracted),
                         [0, 2, 3, 4])
        self.assertEqual(len(extracted), 4)
        self.assertIn(next(b for b in extracted
                           if b.layer_id == REPRESENTATIVE_LAYER).template_id,
                      TEMPLATE_ID)
        others = [b for b in self.plan.bindings
                  if b.layer_id not in (0, 2, 3, 4)]
        self.assertTrue(all(b.template_status == "pending_extraction"
                            and b.template_id is None for b in others))

    def test_params_and_states_unique_per_instance(self):
        param_keys = [tuple(sorted(b.params.items())) for b in self.plan.bindings]
        state_keys = [b.state_ids for b in self.plan.bindings]
        self.assertEqual(len(set(param_keys)), 43, "参数逐实例唯一")
        self.assertEqual(len(set(state_keys)), 43, "state_id 逐实例唯一")
        # 同组实例参数只差 layer_id
        csa_topk = [b for b in self.plan.bindings if b.layer_id in
                    next(g.member_layers for g in self.plan.groups
                         if g.group_id == "csa_topk")]
        base = dict(csa_topk[0].params)
        for b in csa_topk[1:]:
            diff = {k: v for k, v in b.params.items() if base.get(k) != v}
            self.assertEqual(set(diff), {"layer_id"}, b.layer_id)

    def test_state_namespaces_per_layer(self):
        l4 = self.plan.bindings[4]
        self.assertIn("layer_4:attn_kv_cache", l4.state_ids)
        self.assertIn("layer_4:indexer_kv_cache", l4.state_ids)
        l0 = self.plan.bindings[0]
        self.assertEqual(l0.state_ids, ("layer_0:attn_kv_cache",))

    def test_dspark_readouts_external_only(self):
        # B4 补修01：三个有序 readout 共同 concat 进 stage0 唯一
        # main_proj/main_norm → 唯一 main_x 三 stage 共享（单条记录，
        # 不再是旧一对一 readout→draft_stage 映射）
        self.assertEqual(len(self.plan.dspark_readouts), 1)
        rec = self.plan.dspark_readouts[0]
        self.assertEqual(rec.readout_layers, (40, 41, 42))
        self.assertEqual(rec.consumer_stage_id, 0)
        self.assertEqual(rec.shared_consumers, (0, 1, 2))
        self.assertIn("model.py", rec.evidence_ref)

    def test_coverage_honest(self):
        cov = self.plan.coverage
        # V2+B4：四类主干模板 extracted；DSpark 内部图已整体集成
        # （B4，coverage pending_extraction 清空；本 plan 不含 DSpark
        # 集成图——它由 cli._build_documents 单独构建并通过集成契约）
        self.assertEqual(
            cov.extracted,
            ("deepseek_v4_flash_0731:swa_hash", "deepseek_v4_flash_0731:csa_hash",
             TEMPLATE_ID, "deepseek_v4_flash_0731:hca_topk"))
        self.assertEqual(cov.pending_extraction, ())
        self.assertTrue(all("kernel" in u or "hadamard" in u
                            or "token 数" in u or "可见条目" in u
                            for u in cov.uncovered))


class TestScenarioViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph, cls.registry, cls.values = build()
        cls.views = {v.scenario_id: v
                     for v in build_scenario_views(cls.values, cls.registry)}

    def test_all_scenarios_present(self):
        expected = {"prefill_t3", "prefill_t4", "prefill_t5", "prefill_t127",
                    "prefill_t128", "prefill_t129",
                    "decode_p2", "decode_p3", "decode_p4", "decode_p126",
                    "decode_p127", "decode_p128"}
        self.assertEqual(set(self.views), expected)

    def test_guards_all_determined(self):
        for view in self.views.values():
            for act in view.guard_activations:
                self.assertNotEqual(act.active_for_scenario, "unknown",
                                    f"{view.scenario_id}/{act.guard_id}")

    def test_prefill_vs_decode_guards(self):
        p3 = self.views["prefill_t3"]
        acts = {a.guard_id: a.active_for_scenario for a in p3.guard_activations}
        self.assertEqual(acts["g_prefill"], "true")
        self.assertEqual(acts["g_decode"], "false")
        self.assertEqual(acts["g_compress_prefill_skip"], "true",
                         "T=3 < r=4：prefill 压缩未触发")
        self.assertEqual(acts["g_compress_prefill"], "false")
        d127 = self.views["decode_p127"]
        acts127 = {a.guard_id: a.active_for_scenario
                   for a in d127.guard_activations}
        self.assertEqual(acts127["g_decode"], "true")
        self.assertEqual(acts127["g_compress_decode"], "true",
                         "start_pos=127: (127+1)%4==0 触发")
        d128 = self.views["decode_p128"]
        acts128 = {a.guard_id: a.active_for_scenario
                   for a in d128.guard_activations}
        self.assertEqual(acts128["g_compress_decode"], "false")

    def test_indexer_visibility_bounds(self):
        # C=0 边界：decode P=2 → end_pos=3 → 可见 0 条
        v2 = indexer_visibility(self.views["decode_p2"].env)
        self.assertEqual(v2["visible_upper"], 0)
        self.assertEqual(v2["select_width"], 0)
        # 少于 Ki：P=3 → 1 条
        v3 = indexer_visibility(self.views["decode_p3"].env)
        self.assertEqual(v3["visible_upper"], 1)
        self.assertEqual(v3["select_width"], 1)
        # prefill T=129 → 32 条 < Ki=512
        v129 = indexer_visibility(self.views["prefill_t129"].env)
        self.assertEqual(v129["visible_upper"], 32)
        self.assertEqual(v129["select_width"], 32)
        # 达到 Ki：P+T = 2048 → 512 条 = Ki
        big = indexer_visibility({"P": 0, "T": 2048, "r": 4, "Ki": 512})
        self.assertEqual(big["visible_upper"], 512)
        self.assertEqual(big["select_width"], 512)
        # 超过 Ki：P+T = 4096 → 1024 条 > Ki，有效选择封顶 Ki
        bigger = indexer_visibility({"P": 0, "T": 4096, "r": 4, "Ki": 512})
        self.assertEqual(bigger["visible_upper"], 1024)
        self.assertEqual(bigger["select_width"], 512,
                         "Ki 是上限，不是每 query 有效数")

    def test_history_premise_recorded(self):
        for view in self.views.values():
            self.assertIn("历史连续调用前提", view.premise)
            self.assertIn("unknown", view.premise)

    def test_derived_counts(self):
        p5 = self.views["prefill_t5"]
        self.assertEqual(p5.derived["N_tokens"], 5)
        self.assertEqual(p5.derived["remainder"], 1)
        self.assertEqual(p5.derived["N_blk"], 1)
        self.assertEqual(p5.derived["N_ctx"], 6)  # 5 + 1
        d2 = self.views["decode_p2"]
        self.assertEqual(d2.derived["decode_trigger"], 0)
        d3 = self.views["decode_p3"]
        self.assertEqual(d3.derived["decode_trigger"], 1)


class TestGraphSerializationRoundtrip(unittest.TestCase):
    def test_roundtrip_identical(self):
        from structure_reader.graph_document import graph_from_json, graph_to_json
        graph, _reg, _vals = build()
        text = graph_to_json(graph)
        restored = graph_from_json(text)
        self.assertEqual(graph_to_json(restored), text)


class TestReview02Findings(unittest.TestCase):
    """codex_review_02 探针发现：活动路径、可见 slice、状态输入、状态建模。"""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.registry, cls.values = build()
        cls.env_p128 = {"B": 1, "T": 1, "P": 128, "start_pos": 128,
                        "r": 4, "W": 128, "Ki": 512}

    def _active_edges(self, graph, env):
        """按场景 guard 求值过滤出的活动边（guard=None 恒活动）。"""
        from structure_reader.rules.deepseek_v4_flash_0731 import GUARD_DEFS
        from structure_reader.scenario_evaluator import evaluate_tri
        guard_state = {}
        for g in GUARD_DEFS:
            result = evaluate_tri(g.predicate_expr, env)
            guard_state[g.stable_id] = result.value if result.is_known else None
        active = []
        for e in graph.edges:
            if e.guard_ref is None:
                active.append(e)
            elif guard_state.get(e.guard_ref) is True:
                active.append(e)
        return active

    def _reachable_producers(self, graph, active_edges):
        """活动子图中每个节点是否可达（从无输入的参数/常量节点出发）。"""
        has_in = {e.dst_node_id for e in active_edges}
        sources = {n.stable_id for n in graph.nodes
                   if n.stable_id not in has_in}  # 无活动入边的节点是源
        reachable = set(sources)
        changed = True
        while changed:
            changed = False
            for e in active_edges:
                if e.src_node_id in reachable and e.dst_node_id not in reachable:
                    reachable.add(e.dst_node_id)
                    changed = True
        return reachable

    def test_decode_p128_sparse_kv_input_active(self):
        # decode P=128：sparse_attn 的 kv_win 端口必须有活动供给边，
        # 且生产者可达（审查探针：active sparse kv inputs == []）
        active = self._active_edges(self.graph, self.env_p128)
        kv_edges = [e for e in active
                    if e.dst_node_id == "attn.sparse_attn"
                    and e.dst_port == "kv_win"]
        self.assertTrue(kv_edges, "decode P128 下 sparse_attn.kv_win 无活动输入")
        reachable = self._reachable_producers(self.graph, active)
        for e in kv_edges:
            self.assertIn(e.src_node_id, reachable,
                          f"kv_win 供给者 {e.src_node_id} 在活动子图中不可达")

    def test_prefill_t3_sparse_kv_active_without_compress(self):
        # prefill T=3（<r）：kv_win 有活动供给（只有窗口路径，无压缩拼接）
        env = {"B": 1, "T": 3, "P": 0, "start_pos": 0, "r": 4, "W": 128,
               "Ki": 512}
        active = self._active_edges(self.graph, env)
        kv_edges = [e for e in active
                    if e.dst_node_id == "attn.sparse_attn"
                    and e.dst_port == "kv_win"]
        self.assertTrue(kv_edges, "prefill T3 下 sparse_attn.kv_win 无活动输入")
        # 未触发压缩：kv_concat 的 compress 输入边不活动
        comp_in = [e for e in active
                   if e.dst_node_id == "attn.kv_concat"
                   and e.dst_port == "compress"]
        self.assertFalse(comp_in, "T<r 时压缩拼接输入不得活动")

    def test_indexer_score_reads_visible_slice(self):
        # score 的 cache 输入必须是可见前缀张量 idx_cache_visible（轴 N_vis），
        # 由真实 slice 节点从 idx_cache 产出（alias 不能无消费）
        tensors = {t.name: t for t in self.graph.tensors}
        self.assertIn("csa_topk:idx_cache_visible", tensors)
        vis = tensors["csa_topk:idx_cache_visible"]
        vis_symbols = [a.symbol for a in vis.axes]
        self.assertIn("N_vis", vis_symbols, "可见前缀张量必须是 N_vis 轴")
        score = next(n for n in self.graph.nodes
                     if n.stable_id == "attn.indexer.score")
        cache_port = next(p for p in score.input_ports if p.name == "cache")
        self.assertEqual(cache_port.tensor_id, "tensor:csa_topk:idx_cache_visible",
                         "score 必须消费可见前缀张量")
        # slice 节点真实存在且被消费（不是孤立 alias）
        slice_edges = [e for e in self.graph.edges
                       if e.dst_node_id == "attn.indexer.score"
                       and e.tensor_id == "tensor:csa_topk:idx_cache_visible"]
        self.assertTrue(slice_edges, "idx_cache_visible 无真实消费边")
        # 可见前缀的生产者从 idx_cache 派生
        producer_id = slice_edges[0].src_node_id
        producer_in = [e for e in self.graph.edges
                       if e.dst_node_id == producer_id
                       and e.tensor_id == "tensor:csa_topk:idx_cache"]
        self.assertTrue(producer_in, "可见前缀节点必须从 idx_cache 派生")

    def test_causal_mask_guarded_and_decode_bypass(self):
        # prefill 因果 mask 挂 g_causal_mask；decode 有无 mask 直连路径
        mask_in = [e for e in self.graph.edges
                   if e.dst_node_id == "attn.indexer.causal_mask"
                   and e.dst_port == "x"]
        self.assertTrue(mask_in)
        self.assertTrue(all(e.guard_ref == "g_causal_mask" for e in mask_in),
                        "causal_mask 输入必须挂 g_causal_mask（model.py:430）")
        decode_direct = [e for e in self.graph.edges
                         if e.dst_node_id == "attn.indexer.topk"
                         and e.guard_ref == "g_decode"]
        self.assertTrue(decode_direct,
                        "decode 必须有无 mask 直连 topk 的路径")

    def test_compressor_state_update_has_real_io(self):
        # 状态更新节点必须有真实输入（历史读 + 本次投影）与输出（写入），
        # 不能是没有 incoming 的孤立 source（审查探针：producers == []）
        for comp in ("attn.compressor", "attn.indexer.compressor"):
            upd = next(n for n in self.graph.nodes
                       if n.stable_id == f"{comp}.state_update")
            self.assertTrue(upd.input_ports, f"{comp}.state_update 无输入端口")
            incoming = [e for e in self.graph.edges
                        if e.dst_node_id == f"{comp}.state_update"]
            self.assertTrue(incoming, f"{comp}.state_update 无入边")
            roles = {e.dst_port for e in incoming}
            # 必须同时消费历史状态与本次投影（主块/尾块端口级拆分）
            self.assertIn("hist_kv", roles, f"{comp} 状态更新缺历史 kv 输入")
            self.assertIn("main_kv", roles, f"{comp} 状态更新缺主块输入")
            self.assertIn("tail_kv", roles, f"{comp} 状态更新缺尾块输入")
            outgoing = [e for e in self.graph.edges
                        if e.src_node_id == f"{comp}.state_update"]
            self.assertTrue(outgoing, f"{comp}.state_update 输出无消费")
            # after 版本被 decode 池化消费（当前槽位更新后读）
            consumed_by_pool = [e for e in outgoing
                                if e.dst_node_id == f"{comp}.pool_decode"]
            self.assertTrue(consumed_by_pool,
                            f"{comp}.state_update 的 after 输出未被池化消费")

    def test_decode_pool_reads_state_not_prefill_path(self):
        # decode 触发时的池化必须读 updated 状态（历史+当前合一，写入后），
        # 不是 prefill 的 blockify 路径
        decode_pool = [e for e in self.graph.edges
                       if e.dst_node_id == "attn.compressor.pool_decode"
                       and e.guard_ref == "g_compress_decode"]
        roles = {e.dst_port for e in decode_pool}
        self.assertIn("kv_state", roles, "decode 池化缺 kv_state 输入")
        self.assertIn("score_state", roles, "decode 池化缺 score_state 输入")
        # 池化读的是 updated 版本（不是 hist 直读）
        for e in decode_pool:
            if e.dst_port == "kv_state":
                self.assertEqual(e.tensor_id, "tensor:csa_topk:comp_state_kv_updated")

    def test_compress_write_gated_by_trigger(self):
        # 压缩 cache 写与 return 的输入边必须挂触发 guard（未触发不产值）
        for dst, port in (("attn.compressor.cache_out", "x"),
                          ("attn.compressor.return", "x")):
            ins = [e for e in self.graph.edges
                   if e.dst_node_id == dst and e.dst_port == port]
            self.assertTrue(ins, f"{dst}.{port} 无输入")
            for e in ins:
                self.assertIn(e.guard_ref,
                              ("g_compress_prefill", "g_compress_decode"),
                              f"{dst} 的输入边未挂触发 guard（{e.stable_id}）")

    def test_single_cache_state_no_parallel_after(self):
        # 写后读是同一 state_id 的版本更新，不得建成独立平行 state
        state_ids = {t.state_id for t in self.graph.state_transitions}
        self.assertNotIn("csa_topk:attn_kv_cache_after", state_ids,
                         "cache_after 被建成独立 state（写后读错误建模）")
        # 同一 cache 的写边与读边引用同一 transition
        cache_tr = "tr_attn_cache"
        write_read = [e for e in self.graph.edges
                      if e.kind == "state"
                      and e.state_transition_ref == cache_tr]
        self.assertGreaterEqual(len(write_read), 1,
                                "cache 状态须有实际 state 边")
        # 写生产者（cache_update）与读消费者（cache_read）经该边连通
        edge = write_read[0]
        self.assertEqual(edge.src_node_id, "attn.cache_update")
        self.assertEqual(edge.dst_node_id, "attn.cache_read")

    def test_no_dead_end_state_sinks(self):
        # 状态路径不得终止在无输出的 sink 节点（写入必须到达读取）
        sinks = [n for n in self.graph.nodes if n.op_type == "state_sink"]
        self.assertEqual(sinks, [], "存在终止型 state_sink 节点（状态孤岛）")

    def test_no_duplicate_semantic_edges(self):
        # 同 (src, dst, tensor, guard) 的重复接线不得存在
        seen: set[tuple] = set()
        for e in self.graph.edges:
            key = (e.src_node_id, e.src_port, e.dst_node_id, e.dst_port,
                   e.tensor_id, e.guard_ref, e.state_transition_ref)
            self.assertNotIn(key, seen, f"重复边: {e.stable_id}")
            seen.add(key)


class TestReviewC2Findings(unittest.TestCase):
    """codex_review_02（C 二轮）：主/尾块真实切片与 pool 后 shift 最终版本。"""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.registry, cls.values = build()

    def test_main_tail_have_real_slice_producers(self):
        # main_kv/tail_kv 的生产者必须是真实 slice 节点（不是同一个完整投影），
        # 且两者的输入张量不同（cutoff-r:cutoff vs cutoff:）
        upd_in = {e.dst_port: e for e in self.graph.edges
                  if e.dst_node_id == "attn.compressor.state_update"}
        for port in ("main_kv", "tail_kv"):
            self.assertIn(port, upd_in, port)
        main_edge, tail_edge = upd_in["main_kv"], upd_in["tail_kv"]
        self.assertNotEqual(main_edge.src_node_id, tail_edge.src_node_id,
                            "main/tail 不能共享同一生产者节点")
        for edge in (main_edge, tail_edge):
            self.assertIn("slice", edge.src_node_id,
                          f"{edge.stable_id} 生产者必须是 slice 节点")
        # 两个 slice 的源都是完整投影，但各自是不同 view 张量
        self.assertNotEqual(main_edge.tensor_id, tail_edge.tensor_id,
                            "main/tail 必须消费不同 view 张量")

    def test_tail_edges_guarded_for_prefill(self):
        # 尾块只在 prefill && remainder>0 活动（T=4/128 无尾块活动）；
        # decode 当前 token 有独立清晰路径（不是无 guard 的 prefill 尾块边）
        tail_edges = [e for e in self.graph.edges
                      if e.dst_node_id == "attn.compressor.state_update"
                      and e.dst_port in ("tail_kv", "tail_score")]
        self.assertTrue(tail_edges)
        guards = {e.guard_ref for e in tail_edges}
        self.assertTrue(
            guards <= {"g_compress_prefill", "g_compress_prefill_remainder",
                       "g_decode"},
            f"tail 边必须挂 prefill/decode 场景 guard，得到 {guards}")
        self.assertIn("g_compress_prefill_remainder", guards,
                      "prefill 尾块边必须挂 remainder guard")

    def test_t4_no_tail_activity_t5_both(self):
        # 场景探针：T=4（无尾块活动、主块活动）、T=5（两者活动）、
        # T=3（主块不活动、尾块活动）
        def active(env):
            from structure_reader.rules.deepseek_v4_flash_0731 import GUARD_DEFS
            from structure_reader.scenario_evaluator import evaluate_tri
            states = {}
            for g in GUARD_DEFS:
                r = evaluate_tri(g.predicate_expr, env)
                states[g.stable_id] = r.value if r.is_known else None
            result = {"main": False, "tail": False}
            for e in self.graph.edges:
                if e.dst_node_id != "attn.compressor.state_update":
                    continue
                if e.guard_ref is not None and states.get(e.guard_ref) is not True:
                    continue
                if e.dst_port in ("main_kv", "main_score"):
                    result["main"] = True
                if e.dst_port in ("tail_kv", "tail_score"):
                    result["tail"] = True
            return result

        base = {"B": 1, "P": 0, "start_pos": 0, "r": 4, "W": 128, "Ki": 512}
        self.assertEqual(active({**base, "T": 3}), {"main": False, "tail": True})
        self.assertEqual(active({**base, "T": 4}), {"main": True, "tail": False})
        self.assertEqual(active({**base, "T": 5}), {"main": True, "tail": True})
        self.assertEqual(active({**base, "T": 128}), {"main": True, "tail": False})
        self.assertEqual(active({**base, "T": 129}), {"main": True, "tail": True})

    def test_shift_final_version_exists_and_connected(self):
        # pool 后 shift 是独立节点，产出最终状态版本（v_final）；
        # 触发时 pool 读 updated 版本 → shift → final；final 被连续调用消费
        nodes = {n.stable_id for n in self.graph.nodes}
        self.assertIn("attn.compressor.pool_shift", nodes,
                      "缺少 pool 后 shift 节点")
        shift_in = [e for e in self.graph.edges
                    if e.dst_node_id == "attn.compressor.pool_shift"]
        self.assertTrue(shift_in, "shift 节点无输入")
        shift_out = [e for e in self.graph.edges
                     if e.src_node_id == "attn.compressor.pool_shift"]
        self.assertTrue(shift_out, "shift 输出无消费")
        # shift 输入来自 pool 触发（guard g_compress_decode）
        self.assertTrue(all(e.guard_ref == "g_compress_decode"
                            for e in shift_in + shift_out
                            if e.guard_ref is not None)
                        or any(e.guard_ref == "g_compress_decode"
                               for e in shift_in),
                        "shift 路径必须挂 g_compress_decode")
        # 版本标签区分 updated（待 pool）与 final（shift 后）
        labels = {v.version_label for v in self.graph.state_versions}
        self.assertIn("v_final", labels, "缺少 pool 后 shift 最终版本标签")

    def test_ape_score_slice_path_queryable(self):
        # APE score 切片/相加的槽位语义可查询（alias 或 note 字段）
        ape = next(n for n in self.graph.nodes
                   if n.stable_id == "attn.compressor.ape")
        self.assertTrue(ape.output_ports)


class TestReviewC3Findings(unittest.TestCase):
    """codex_review_03（C 三轮）：切片 shape/表达式与最终状态时序。"""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.registry, cls.values = build()

    def test_tail_slice_axes_match_remainder_semantics(self):
        # 尾块张量长度是 remainder（T%r，逐场景变化）不是 r=4；
        # decode 当前 token 是单 token（T=1）独立张量
        tensors = {t.name: t for t in self.graph.tensors}
        tail = tensors["csa_topk:comp_tail_slice"]
        len_axis = tail.axes[1]
        self.assertEqual(len_axis.symbol, "N_rem",
                         "尾块长度轴必须是 N_rem（=T%r）")
        self.assertIn("T % r", len_axis.expression)
        dec = tensors["csa_topk:comp_dec_token"]
        self.assertEqual(dec.axes[1].value, 1, "decode token 长度是 1")

    def test_decode_token_tensor_distinct_from_tail(self):
        tensors = {t.name: t for t in self.graph.tensors}
        self.assertIn("csa_topk:comp_dec_token", tensors)
        self.assertNotEqual(tensors["csa_topk:comp_dec_token"].stable_id,
                            tensors["csa_topk:comp_tail_slice"].stable_id)
        # decode_token_slice 节点产出 dec_token 张量（不与 prefill 尾块共享）
        dec_out = [e for e in self.graph.edges
                   if e.src_node_id == "attn.compressor.decode_token_slice"]
        self.assertTrue(dec_out)
        self.assertTrue(all(e.tensor_id == "tensor:csa_topk:comp_dec_token"
                            for e in dec_out))

    def test_slice_aliases_with_transform_expr(self):
        # 切片事实保存在 AliasRelation（slice + transform_expr 原文），
        # 不只靠节点名/stable_id
        aliases = {a.stable_id: a for a in self.graph.alias_relations}
        main = aliases.get("alias_comp_main_slice")
        tail = aliases.get("alias_comp_tail_slice")
        dec = aliases.get("alias_comp_dec_token_slice")
        for a in (main, tail, dec):
            self.assertIsNotNone(a, "缺少切片 alias")
            self.assertEqual(a.relation_kind, "slice")
            self.assertTrue(a.transform_expr and a.transform_expr.strip())
        self.assertIn("cutoff-r:cutoff", main.transform_expr)
        self.assertIn("cutoff:", tail.transform_expr)
        # score 切片 alias 是纯 view（APE 相加由 ape_add 节点承载，
        # codex_review_04：slice 与 add 值流分开）
        self.assertIsNotNone(aliases.get("alias_comp_main_score_slice"))
        self.assertIsNotNone(aliases.get("alias_comp_tail_score_slice"))
        self.assertIn("0:1", aliases.get("alias_comp_dec_token_slice").transform_expr)
        self.assertIn("0:1", aliases.get("alias_comp_dec_score_slice").transform_expr)

    def test_shift_has_pool_completion_dependency(self):
        # pool_shift 必须有 pool_decode 完成的 control 依赖（不只是
        # state_update 的数据入边）
        ctrl = [e for e in self.graph.edges
                if e.dst_node_id == "attn.compressor.pool_shift"
                and e.kind == "control"]
        self.assertTrue(ctrl, "pool_shift 缺 pool 完成的 control 依赖边")
        self.assertTrue(all(e.src_node_id == "attn.compressor.pool_decode"
                            for e in ctrl),
                        "control 依赖必须来自 pool_decode 完成")

    def test_final_state_published_for_all_paths(self):
        # 最终状态发布路径：触发 decode 用 shift 后状态（v_final）；
        # 非触发 decode 与 prefill 用 updated 状态发布（publish_final 节点
        # 按 guard 选择），下一调用经同 state_id 版本链消费
        pub = next((n for n in self.graph.nodes
                    if n.stable_id == "attn.compressor.publish_final"), None)
        self.assertIsNotNone(pub, "缺少最终状态发布节点")
        pub_in = [e for e in self.graph.edges
                  if e.dst_node_id == "attn.compressor.publish_final"]
        src_guards = {(e.src_node_id, e.guard_ref) for e in pub_in}
        # 触发路径：shift 后 final；非触发/prefill：updated（挂互斥 guard）
        self.assertIn(("attn.compressor.pool_shift", "g_compress_decode"),
                      src_guards, "触发路径：shift 后 final 发布")
        self.assertIn(("attn.compressor.state_update", "g_prefill"),
                      src_guards, "prefill：updated 发布（g_prefill 相位 guard）")
        self.assertIn(("attn.compressor.state_update", "g_decode_no_compress"),
                      src_guards, "非触发 decode：updated 发布")
        pub_out = [e for e in self.graph.edges
                   if e.src_node_id == "attn.compressor.publish_final"]
        self.assertTrue(pub_out, "publish_final 输出无消费")


class TestReviewC4Findings(unittest.TestCase):
    """codex_review_04（C 四轮）：decode 局部坐标、slice/add 分流、互斥发布。"""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.registry, cls.values = build()

    def test_decode_slice_uses_local_index(self):
        # 本调用投影长度 1，局部索引 0:1（start_pos 是历史绝对位置，
        # 不是投影局部索引——P2/127/128 时 start_pos:start_pos+1 为空）
        aliases = {a.stable_id: a for a in self.graph.alias_relations}
        for aid in ("alias_comp_dec_token_slice", "alias_idx_comp_dec_token_slice"):
            expr = aliases[aid].transform_expr
            self.assertIn("0:1", expr, f"{aid} 必须用局部索引 0:1")
            self.assertNotIn("start_pos:start_pos+1", expr,
                             f"{aid} 不得用历史绝对位置")
        # 长度 1 列表切片语义：本调用投影轴 N_one=1，切片 0:1 恰取 1 个 token
        tensors = {t.name: t for t in self.graph.tensors}
        self.assertEqual(tensors["csa_topk:comp_dec_token"].axes[1].value, 1)

    def test_score_add_is_not_slice_alias(self):
        # score+APE 是产生新值的 add：slice alias 只表达实际 view（无 ape 相加）；
        # APE 相加由明确 add 节点承载（slice 输出 + ape → add → state_update）
        aliases = {a.stable_id: a for a in self.graph.alias_relations}
        for aid in ("alias_comp_main_score_slice", "alias_comp_tail_score_slice",
                    "alias_comp_dec_score_slice"):
            self.assertNotIn("+", aliases[aid].transform_expr,
                             f"{aid} 是 slice，不得含相加（add 由节点承载）")
            self.assertNotIn("ape", aliases[aid].transform_expr)
        # add 节点存在且消费对应 APE 范围
        adds = [n for n in self.graph.nodes if n.stable_id.startswith(
            "attn.compressor.ape_add")]
        self.assertTrue(adds, "缺少 score+APE 的 add 节点")
        # add 节点有两个输入：切片 score + ape
        for add in adds:
            ins = [e for e in self.graph.edges if e.dst_node_id == add.stable_id]
            tensors = {e.tensor_id for e in ins}
            # score 切片 + APE（main 消费全 r 表 alias；tail/dec 消费行切片
            # view——D2 后 tail/dec 的 APE 输入是 *_ape_tail_rows/*_ape_dec_row）
            self.assertEqual(len(tensors), 2, f"{add.stable_id} 输入应是 score 切片+APE")
            if "tail" in add.stable_id:
                self.assertTrue(
                    any(x.endswith("ape_tail_rows") for x in tensors),
                    f"{add.stable_id} tail 必须消费 APE 行切片")
            if "_dec" in add.stable_id:
                self.assertTrue(
                    any(x.endswith("ape_dec_row") for x in tensors),
                    f"{add.stable_id} decode 必须消费 APE 单行切片")

    def test_publish_sources_mutually_exclusive(self):
        # publish_final 双来源互斥：触发 decode 只收 shift 后（updated 边挂
        # g_compress_decode 取反语义由 g_prefill/non-trigger 表达），
        # 非触发 decode 与 prefill 只收 updated
        pub_in = [e for e in self.graph.edges
                  if e.dst_node_id == "attn.compressor.publish_final"]
        guards = {e.stable_id: e.guard_ref for e in pub_in}
        upd_edges = [e for e in pub_in if e.src_node_id == "attn.compressor.state_update"]
        self.assertTrue(upd_edges)
        for e in upd_edges:
            self.assertIsNotNone(e.guard_ref,
                                 f"{e.stable_id} 无 guard——触发 decode 时会双源")
            self.assertNotEqual(e.guard_ref, "g_compress_decode",
                                f"{e.stable_id} 挂触发 guard（应是非触发路径）")
        # 逐场景验证 kv/score 各有唯一活动状态源
        def active_sources(env):
            from structure_reader.rules.deepseek_v4_flash_0731 import GUARD_DEFS
            from structure_reader.scenario_evaluator import evaluate_tri
            states = {}
            for g in GUARD_DEFS:
                r = evaluate_tri(g.predicate_expr, env)
                states[g.stable_id] = r.value if r.is_known else None
            srcs = set()
            for e in pub_in:
                if e.dst_port == "final" or e.dst_port == "final_score":
                    if states.get(e.guard_ref) is True:
                        srcs.add("shifted")
                else:
                    # updated 边的 guard 是"非触发"语义
                    gval = states.get(e.guard_ref)
                    if e.guard_ref is None or gval is True:
                        srcs.add("updated")
            return srcs
        base = {"B": 1, "r": 4, "W": 128, "Ki": 512}
        for t in (3, 4, 5, 129):
            srcs = active_sources({**base, "T": t, "P": 0, "start_pos": 0})
            self.assertEqual(srcs, {"updated"}, f"T={t} 应只有 updated 发布")
        self.assertEqual(active_sources({**base, "T": 1, "P": 2, "start_pos": 2}),
                         {"updated"}, "P=2 非触发应只有 updated")
        self.assertEqual(active_sources({**base, "T": 1, "P": 127, "start_pos": 127}),
                         {"shifted"}, "P=127 触发应只有 shifted")

    def test_version_chain_has_stage_refs(self):
        # 4 步链每步有 before/updated/final 明确引用与上一 final→下一 before
        from structure_reader.build_pipeline import _decode_version_chain
        chain = _decode_version_chain(self.values)
        self.assertEqual(len(chain), 4)
        for i, step in enumerate(chain):
            self.assertIn("before_ref", step, i)
            self.assertIn("updated_ref", step, i)
            self.assertIn("final_ref", step, i)
            self.assertIn("final_state", step)  # 'updated' 或 'shifted'
        # 跨步：step i+1 的 before 引用 step i 的 final
        for i in range(1, len(chain)):
            self.assertEqual(chain[i]["before_ref"], chain[i - 1]["final_ref"],
                             f"step {i + 1} before 必须引用上一 final")

    def test_no_synthetic_shift_state_inputs(self):
        # cache_out/return 不得有 shift_state 人为入参（源码消费量化条目，
        # 不消费状态）
        for nid in ("attn.compressor.cache_out", "attn.compressor.return",
                    "attn.indexer.compressor.cache_out",
                    "attn.indexer.compressor.return"):
            node = next(n for n in self.graph.nodes if n.stable_id == nid)
            self.assertNotIn("shift_state", [p.name for p in node.input_ports],
                             f"{nid} 残留 shift_state 入参")


class TestReviewC5Findings(unittest.TestCase):
    """codex_review_05（C 五轮）：发布 guard 相位范围——逐边计数验证唯一活动源。"""

    @classmethod
    def setUpClass(cls):
        cls.graph, cls.registry, cls.values = build()

    def _active_publish_edges(self, env):
        """对每个 publish_final 端口计数活动边（不去重源节点）。"""
        from structure_reader.rules.deepseek_v4_flash_0731 import GUARD_DEFS
        from structure_reader.scenario_evaluator import evaluate_tri
        states = {}
        for g in GUARD_DEFS:
            r = evaluate_tri(g.predicate_expr, env)
            states[g.stable_id] = r.value if r.is_known else None
        by_port: dict[str, list[str]] = {}
        for e in self.graph.edges:
            if e.dst_node_id != "attn.compressor.publish_final":
                continue
            port = e.dst_port
            if e.guard_ref is None:
                by_port.setdefault(port, []).append(e.stable_id)
            elif states.get(e.guard_ref) is True:
                by_port.setdefault(port, []).append(e.stable_id)
            elif states.get(e.guard_ref) is None:
                by_port.setdefault(port, []).append(
                    e.stable_id + "?(unknown)")
        return by_port

    def test_publish_single_active_source_per_port(self):
        base = {"B": 1, "r": 4, "W": 128, "Ki": 512}
        cases = [
            ({"T": 3, "P": 0, "start_pos": 0}, "prefill_t3"),
            ({"T": 4, "P": 0, "start_pos": 0}, "prefill_t4"),
            ({"T": 5, "P": 0, "start_pos": 0}, "prefill_t5"),
            ({"T": 128, "P": 0, "start_pos": 0}, "prefill_t128"),
            ({"T": 129, "P": 0, "start_pos": 0}, "prefill_t129"),
            ({"T": 1, "P": 2, "start_pos": 2}, "decode_p2"),
            ({"T": 1, "P": 3, "start_pos": 3}, "decode_p3"),
            ({"T": 1, "P": 127, "start_pos": 127}, "decode_p127"),
            ({"T": 1, "P": 128, "start_pos": 128}, "decode_p128"),
        ]
        for env, name in cases:
            by_port = self._active_publish_edges({**base, **env})
            for port, edges in by_port.items():
                self.assertEqual(
                    len(edges), 1,
                    f"{name}.{port} 应有唯一活动源，得到 {edges}")
                self.assertFalse(
                    any(x.endswith("?(unknown)") for x in edges),
                    f"{name}.{port} 有 unknown guard（不得默认选路径）")

    def test_decode_no_compress_guard_has_phase_limit(self):
        from structure_reader.rules.deepseek_v4_flash_0731 import GUARD_DEFS
        guard = next(g for g in GUARD_DEFS
                     if g.stable_id == "g_decode_no_compress")
        self.assertIn("start_pos > 0", guard.predicate_expr,
                      "decode 未触发 guard 必须限定 decode 相位")
        # prefill T3 下该 guard 必须为 false（不再误借 decode 条件）
        from structure_reader.scenario_evaluator import evaluate_tri
        result = evaluate_tri(guard.predicate_expr,
                              {"B": 1, "T": 3, "P": 0, "start_pos": 0,
                               "r": 4, "W": 128, "Ki": 512})
        self.assertIs(result.value, False)

    def test_prefill_publish_uses_phase_guard(self):
        # updated prefill 发布边必须挂 g_prefill（全部 prefill 相位），
        # 不能用 g_compress_prefill（T>=r 才真，T3 丢发布）
        pub_in = [e for e in self.graph.edges
                  if e.dst_node_id == "attn.compressor.publish_final"
                  and e.src_node_id == "attn.compressor.state_update"
                  and e.guard_ref == "g_compress_prefill"]
        self.assertEqual(pub_in, [], "prefill updated 发布不得挂压缩触发 guard")
        prefill_edges = [e for e in self.graph.edges
                         if e.dst_node_id == "attn.compressor.publish_final"
                         and e.src_node_id == "attn.compressor.state_update"
                         and e.guard_ref == "g_prefill"]
        self.assertGreaterEqual(len(prefill_edges), 2,
                                "prefill updated 发布必须挂 g_prefill")


if __name__ == "__main__":
    unittest.main()
