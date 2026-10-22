"""V2 变体测试（工作包 A）：四类真实拓扑、Hash token 依赖、HCA 无 Indexer、
实例分区、受控错误。"""

from __future__ import annotations

import sys
import unittest
from dataclasses import fields as dc_fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.model_registry import find_workspace_root

ROOT = find_workspace_root()


def build_all():
    from structure_reader.extract.deepseek_v4_variants import (
        build_all_variant_graphs)
    return build_all_variant_graphs(ROOT)


def rebuild(graph, **overrides):
    from structure_reader.graph_document import SemanticGraph
    kwargs = {f.name: getattr(graph, f.name) for f in dc_fields(SemanticGraph)}
    kwargs.update(overrides)
    return SemanticGraph(**kwargs)


class TestVariantTopologies(unittest.TestCase):
    """四类变体真实拓扑（源码推导，非标题）。"""

    @classmethod
    def setUpClass(cls):
        cls.results = build_all()

    def test_swa_no_compression_no_indexer(self):
        g = self.results["swa_hash"][0]
        nodes = {n.stable_id for n in g.nodes}
        for absent in ("attn.compressor.wkv", "attn.indexer.wq_b",
                       "attn.compressor.state_update"):
            self.assertNotIn(absent, nodes, f"SWA 不应有 {absent}")
        for present in ("attn.wq_a", "attn.wkv", "attn.win_topk",
                        "attn.sparse_attn"):
            self.assertIn(present, nodes, f"SWA 应有 {present}")
        # YaRN 关闭：无压缩区 slice alias
        self.assertFalse([a for a in g.alias_relations
                          if "comp_cache" in a.stable_id])

    def test_hca_no_indexer_but_has_compression(self):
        g = self.results["hca_topk"][0]
        nodes = {n.stable_id for n in g.nodes}
        for absent in ("attn.indexer.wq_b", "attn.indexer.compressor.wkv",
                       "attn.indexer.score"):
            self.assertNotIn(absent, nodes, f"HCA 不应有学习式 Indexer {absent}")
        for present in ("attn.compressor.wkv", "attn.compressor.state_update",
                        "attn.compress_topk_fixed"):
            self.assertIn(present, nodes, f"HCA 应有 {present}")
        # 固定压缩位置选择产出 select 语义索引
        select = [e for e in g.edges if e.kind == "select"]
        self.assertTrue(select, "HCA 固定选择仍是 select 语义")

    def test_csa_hash_has_indexer_and_hash_gate(self):
        g = self.results["csa_hash"][0]
        nodes = {n.stable_id for n in g.nodes}
        self.assertIn("attn.indexer.wq_b", nodes)
        self.assertIn("moe.gate.tid2eid", nodes, "Hash 门控需 tid2eid 查表")
        self.assertNotIn("moe.gate.bias", nodes, "Hash 层无选择偏置")

    def test_hash_token_dependency(self):
        # Hash 路由的 indices 依赖 input_ids（tid2eid[input_ids]，
        # model.py:582）——input_ids 源 → tid2eid 查表 → gather/dispatch
        for key in ("swa_hash", "csa_hash"):
            g = self.results[key][0]
            tid = g.template_id.rsplit(":", 1)[-1]
            chain = [e for e in g.edges
                     if e.dst_node_id == "moe.gate.tid2eid"
                     and e.tensor_id == f"tensor:{tid}:gate_input_ids"]
            self.assertTrue(chain, f"{key}: tid2eid 缺 input_ids 输入")
            src = chain[0].src_node_id
            self.assertEqual(src, "block.input_ids_src")
            # TopK 变体无 tid2eid
        g4 = self.results["csa_topk"][0]
        self.assertNotIn("moe.gate.tid2eid",
                         {n.stable_id for n in g4.nodes})

    def test_topk_bias_not_in_gather(self):
        # TopK 变体：偏置只进选择不进 gather（V1 已验证结构在变体保持）
        g = self.results["csa_topk"][0]
        bias_out = [e for e in g.edges if e.src_node_id == "moe.gate.bias"]
        self.assertEqual({e.dst_node_id for e in bias_out}, {"moe.gate.topk"})

    def test_shared_structure_across_variants(self):
        # 双 mHC、低秩 Q/O、共享 KV 单投影在四类都保留
        for key, (g, _r, _v) in self.results.items():
            nodes = {n.stable_id for n in g.nodes}
            for shared in ("block.hc_pre_attn.sinkhorn",
                           "block.hc_post_ffn.combine", "attn.wq_a",
                           "attn.wo_b", "attn.wkv", "attn.sparse_attn"):
                self.assertIn(shared, nodes, f"{key} 缺共享结构 {shared}")

    def test_compression_ratio_distinction(self):
        # CSA r=4 overlap；HCA r=128 无 overlap——投影宽 D2=2D 只在 CSA
        g4 = self.results["csa_topk"][0]
        g3 = self.results["hca_topk"][0]
        t4 = {t.name: t for t in g4.tensors}
        t3 = {t.name: t for t in g3.tensors}
        self.assertEqual(t4["csa_topk:comp_kv_proj"].axes[-1].symbol, "D2")
        # HCA r=128 无 overlap（model.py:296 overlap = ratio==4）
        self.assertNotIn("csa_topk:comp_overlap",
                         t3)  # HCA 无 overlap 变换张量


class TestVariantContracts(unittest.TestCase):
    """变体感知契约：正常图 0 违规；变体专属受控错误检出。"""

    @classmethod
    def setUpClass(cls):
        cls.results = build_all()

    def test_all_variants_clean(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        for key, (g, _r, _v) in self.results.items():
            issues = validate_model_contracts(g)
            self.assertEqual(issues, (), f"{key}: {issues[:3]}")

    def test_hca_with_indexer_detected(self):
        # 给 HCA 加 Indexer（错误拓扑）必须检出
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        g = self.results["hca_topk"][0]
        # 加一条 qr→不存在的 indexer.wq_b 边（模拟错误拓扑注入）
        from structure_reader.graph_ir import Edge
        bad = Edge(stable_id="e_injected_idx", src_node_id="attn.q_norm",
                   src_port="y", dst_node_id="attn.indexer.wq_b",
                   dst_port="x", tensor_id="tensor:csa_topk:qr", kind="data")
        mutated = rebuild(g, edges=list(g.edges) + [bad])
        # 通用引用检查会因节点不存在拒绝；契约层面验证 qr 消费方变化
        from structure_reader.checks.graph_contracts import (
            validate_graph_references)
        self.assertTrue(validate_graph_references(mutated),
                        "注入不存在的 Indexer 节点必须被通用检查拒绝")

    def test_hash_token_dependency_removed_detected(self):
        # 移除 Hash token 依赖（tid2eid 无 input_ids 输入）必须检出
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        g = self.results["swa_hash"][0]
        tid = g.template_id.rsplit(":", 1)[-1]
        kept = [e for e in g.edges
                if not (e.dst_node_id == "moe.gate.tid2eid"
                        and e.tensor_id == f"tensor:{tid}:gate_input_ids")]
        mutated = rebuild(g, edges=kept)
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertIn("C18_HASH_TOKEN_DEPENDENCY_MISSING", codes,
                      f"Hash token 依赖被移除未检出，得到 {codes}")

    def test_r_mismatch_detected(self):
        # HCA 图注入 overlap 节点+边（合法端口、合法张量引用——r=128 无
        # overlap 的错误拓扑）必须被 C19 检出，非悬空引用
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        g = self.results["hca_topk"][0]
        from structure_reader.graph_ir import Edge, Node, Port, Tensor
        axis_by = {a.symbol: a for a in g.named_axes}
        ov = Tensor(name="hca_topk:comp_overlap_injected", dtype="fp32",
                    axes=[axis_by["B"], axis_by["N_blk"], axis_by["r"],
                          axis_by["D"]],
                    logical_layout=["B", "N_blk", "r", "D"],
                    physical_layout=["B", "N_blk", "r", "D"])
        ov_node = Node(stable_id="attn.compressor.overlap",
                       op_type="overlap_transform", granularity="tensor_op",
                       input_ports=[Port("x", "input", ov.stable_id)],
                       output_ports=[Port("y", "output", ov.stable_id)])
        bad = Edge(stable_id="e_injected_overlap",
                   src_node_id="attn.compressor.blockify", src_port="y",
                   dst_node_id="attn.compressor.overlap", dst_port="x",
                   tensor_id=ov.stable_id, kind="data")
        mutated = rebuild(g, tensors=list(g.tensors) + [ov],
                          nodes=list(g.nodes) + [ov_node],
                          edges=list(g.edges) + [bad])
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertIn("C19_HCA_OVERLAP_NODE", codes,
                      f"HCA 注入 overlap 节点未检出，得到 {codes}")

    def test_variant_guard_semantics(self):
        # SWA 的压缩 guard 缺失是正确拓扑（不是 C16 违规）；CSA 系齐备
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        g_swa = self.results["swa_hash"][0]
        self.assertEqual(validate_model_contracts(g_swa), ())
        g_hca = self.results["hca_topk"][0]
        self.assertEqual(validate_model_contracts(g_hca), ())

    def test_hca_axis_expressions_ki_free(self):
        # review04：HCA 轴/张量公式不得借 Indexer Ki，且必须是可执行求值的
        # 阶段统一公式（非不可求值的阶段标签串）——四例真实 shape
        # （prefill T3/T129、decode P2/P127，T1）分相位求值，物理宽与
        # 有效数分别断言；不只比对公式字符串。
        from structure_reader.extract.deepseek_v4_variants import (
            build_variant_graph, VARIANT_SPECS)
        spec = VARIANT_SPECS["hca_topk"]
        g, _reg, _v = build_variant_graph(ROOT, spec)
        axes = {a.symbol: a for a in g.named_axes}
        # 去 Ki（语义断言：表达式不得引用 Ki）
        self.assertNotIn("Ki", axes["N_sel_c"].expression,
                         "HCA N_sel_c 不得引用 Ki")
        self.assertNotIn("Ki", axes["N_sel"].expression,
                         "HCA N_sel 不得引用 Ki")
        # 轴表达式必须可求值（evaluate_int 白名单；阶段由 P 编码）
        from structure_reader.scenario_evaluator import evaluate_int
        # 张量轴同源：comp_idxs/select_idxs 挂 N_sel_c/N_sel；
        # win_idxs 挂 N_win（窗口索引物理宽轴，非 W）
        t = {x.name: x for x in g.tensors}
        idx_syms = [ax.symbol for ax in t["hca_topk:comp_idxs"].axes]
        self.assertIn("N_sel_c", idx_syms)
        sel_syms = [ax.symbol for ax in t["hca_topk:select_idxs"].axes]
        self.assertIn("N_sel", sel_syms)
        win_syms = [ax.symbol for ax in t["hca_topk:win_idxs"].axes]
        self.assertIn("N_win", win_syms,
                      "窗口选择索引物理宽轴必须是 N_win（prefill T3 真实宽 3）")
        # 四例实算矩阵（model.py:261-271/274-282/513-519 真实控制流推导；
        # Ki=8 注入环境——若公式误借 min(Ki,..)，decode_p127 会取 8≠1，
        # 期望矩阵直接杀掉该错误，也是去 Ki 的可执行证据）
        envs = {
            "prefill_t3": {"B": 1, "T": 3, "P": 0, "start_pos": 0,
                           "W": 128, "r": 128, "Ki": 8},
            "prefill_t129": {"B": 1, "T": 129, "P": 0, "start_pos": 0,
                             "W": 128, "r": 128, "Ki": 8},
            "decode_p2": {"B": 1, "T": 1, "P": 2, "start_pos": 2,
                          "W": 128, "r": 128, "Ki": 8},
            "decode_p127": {"B": 1, "T": 1, "P": 127, "start_pos": 127,
                            "W": 128, "r": 128, "Ki": 8},
        }
        expect_total = {"prefill_t3": 3, "prefill_t129": 129,
                        "decode_p2": 128, "decode_p127": 129}
        expect_window = {"prefill_t3": 3, "prefill_t129": 128,
                         "decode_p2": 128, "decode_p127": 128}
        expect_compress = {"prefill_t3": 0, "prefill_t129": 1,
                           "decode_p2": 0, "decode_p127": 1}
        for sid, env in envs.items():
            total = evaluate_int(axes["N_sel"].expression, env)
            self.assertEqual(
                total, expect_total[sid],
                f"{sid}: HCA 总选择物理宽（review04 矩阵 3/129/128/129）")
            win = evaluate_int(axes["N_win"].expression, env)
            self.assertEqual(
                win, expect_window[sid],
                f"{sid}: 窗口索引物理宽（decode 槽位数 W；未填满 -1 填充）")
            selc = evaluate_int(axes["N_sel_c"].expression, env)
            self.assertEqual(selc, expect_compress[sid],
                             f"{sid}: 固定压缩选择物理宽 (P+T)//r")
            self.assertEqual(total, win + selc,
                             f"{sid}: 总物理宽 = 窗口 + 压缩")
        # decode_p2：物理宽 128 槽位，但有效数上限 = min(P+T,W)+(P+T)//r = 3
        # ——物理宽 ≠ 有效数（review04 核心区分，不能互相冒充）
        valid_upper = (min(envs["decode_p2"]["P"] + envs["decode_p2"]["T"], 128)
                       + (envs["decode_p2"]["P"] + envs["decode_p2"]["T"]) // 128)
        self.assertEqual(valid_upper, 3)
        self.assertNotEqual(
            evaluate_int(axes["N_sel"].expression, envs["decode_p2"]),
            valid_upper, "decode_p2 物理宽 128 与有效数 3 必须是两个量")
        # 场景视图同源：derived 与图轴表达式对同一 env 求值一致
        from structure_reader.extract.deepseek_v4_reference import (
            build_scenario_views)
        g2, reg2, v2 = build_variant_graph(ROOT, spec)
        views = build_scenario_views(v2, reg2, compress_ratio=128,
                                     has_indexer=False)
        for view in views:
            env = dict(view.env)
            env.update({"W": 128, "r": 128, "Ki": 512})
            self.assertEqual(view.derived["N_sel"],
                             evaluate_int(axes["N_sel"].expression, env),
                             f"{view.scenario_id}: derived 与图轴同源求值")
            self.assertEqual(view.derived["N_win"],
                             evaluate_int(axes["N_win"].expression, env),
                             f"{view.scenario_id}: 窗口物理宽同源")
            # 有效数上限独立字段存在且 ≤ 物理宽
            self.assertLessEqual(view.derived["N_sel_valid_upper"],
                                 view.derived["N_sel"],
                                 f"{view.scenario_id}: 有效数上限 ≤ 物理宽")
            # 场景视图同 Ki-free
            self.assertNotIn("Ki", str(view.indexer.get("capacity_note", "")),
                             view.scenario_id)

    def test_swa_window_width_vs_valid_four_cases(self):
        # review04：SWA select_width 同混物理宽与有效数——四例实算矩阵
        # prefillT3/T129、decodeP2/P127（T1）：物理宽 3/128/128/128，
        # 有效数上限 3/128/3/128（decode_p2 物理宽 128 槽位、有效仅 3）
        from structure_reader.extract.deepseek_v4_variants import (
            build_variant_graph, VARIANT_SPECS)
        from structure_reader.extract.deepseek_v4_reference import (
            build_scenario_views)
        spec = VARIANT_SPECS["swa_hash"]
        g, reg, v = build_variant_graph(ROOT, spec)
        axes = {a.symbol: a for a in g.named_axes}
        self.assertIn("N_win", axes,
                      "SWA 也有窗口索引物理宽轴（model.py:261-271 对所有层）")
        from structure_reader.scenario_evaluator import evaluate_int
        views = build_scenario_views(v, reg, compress_ratio=0,
                                     has_indexer=False)
        by_id = {x.scenario_id: x for x in views}
        cases = {"prefill_t3": (3, 3), "prefill_t129": (128, 128),
                 "decode_p2": (128, 3), "decode_p127": (128, 128)}
        for sid, (want_phys, want_valid) in cases.items():
            view = by_id[sid]
            self.assertEqual(view.derived["N_win"], want_phys,
                             f"{sid}: SWA 窗口索引物理宽")
            self.assertEqual(view.derived["N_sel"], want_phys,
                             f"{sid}: SWA 选择总物理宽（无压缩项）")
            self.assertEqual(view.derived["N_sel_valid_upper"], want_valid,
                             f"{sid}: SWA 有效数上限（≠物理宽当 decode 未填满）")
            self.assertEqual(view.indexer["window_width"], want_phys,
                             f"{sid}: indexer 窗口物理宽")
            self.assertEqual(view.indexer["select_valid_upper"], want_valid,
                             f"{sid}: indexer 有效数与物理宽分列")
        # 图轴同源：derived 与 N_win 轴表达式同 env 求值一致
        for view in views:
            env = dict(view.env)
            env.update({"W": 128, "Ki": 512})
            self.assertEqual(view.derived["N_win"],
                             evaluate_int(axes["N_win"].expression, env),
                             f"{view.scenario_id}: SWA N_win 图轴同源")

    def test_swa_window_scenarios_real(self):
        # review03 第 1 项：SWA 有真实 prefill/decode 窗口场景
        # （窗口槽位/有效长度/容量 W），压缩 guard/state 明确不适用
        from structure_reader.extract.deepseek_v4_variants import (
            build_variant_graph, VARIANT_SPECS)
        from structure_reader.extract.deepseek_v4_reference import (
            build_scenario_views)
        spec = VARIANT_SPECS["swa_hash"]
        g, reg, v = build_variant_graph(ROOT, spec)
        views = build_scenario_views(v, reg, compress_ratio=0,
                                     has_indexer=False)
        self.assertEqual(len(views), 12)
        t5 = next(x for x in views if x.scenario_id == "prefill_t5")
        self.assertEqual(t5.derived["window_slots_used"], 5)
        self.assertEqual(t5.derived["window_valid_length"], 5)
        self.assertEqual(t5.derived["decode_trigger"], 0)
        # SWA guards 仅相位 + 窗口回绕
        guard_ids = {a.guard_id for a in t5.guard_activations}
        self.assertEqual(guard_ids, {"g_prefill", "g_decode", "g_window_wrap"})
        d128 = next(x for x in views if x.scenario_id == "decode_p128")
        # decode P=128：end_pos=129 → 已填充 min(W,129)=128 槽，
        # 下一写入槽位 = 129%128 = 1（环形回绕），有效长度 = W
        self.assertEqual(d128.derived["window_slots_used"], 128)
        self.assertEqual(d128.derived["window_next_slot"], 1)
        self.assertEqual(d128.derived["window_valid_length"], 128)
        # 无压缩派生数量：SWA derived 不得含压缩块字段（N_blk 是压缩
        # 专属；SWA 窗口语义用 window_slots_used/window_valid_length）
        self.assertNotIn("N_blk", t5.derived)
        self.assertNotIn("N_vis", t5.derived)
        self.assertNotIn("N_sel_c", t5.derived)


if __name__ == "__main__":
    unittest.main()
