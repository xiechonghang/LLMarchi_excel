"""DSparkAttention 真实子图提取器测试（工作包 B · 步骤 B1/B2）：
真实图构建、证据原文核对、canonical JSON 往返、prefill/decode 相位边界、
KV 单投影共享、草稿位置 offset 与 cache 环形槽、共享参数与相位边、
位置/槽位坐标的图中可验证表达（B2 修正：bundle 级 CoordinateExpression
显式记录 + 公开 JSON 往返携带，坐标值非数据张量 view；从实际 bundle/
序列化结果读取求值，篡改可检出；虚假坐标 view alias 已删断言）。
只覆盖 B1/B2 范围（Attention 子图 + 坐标表示修正）；公共 stage/三实例
绑定在 test_dspark_stage.py；Markov/Confidence/全量集成由后续步骤续接，
本测试不声称覆盖。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_dspark import (
    EXPR_KV_CONCAT_WIDTH,
    EXPR_SEL_WIDTH,
    TEMPLATE_ID,
    build_dspark_attention_graph,
    bundle_from_json,
    bundle_to_json,
)
from structure_reader.extract.evidence import EvidenceError, SourceEvidence
from structure_reader.graph_document import graph_from_json, graph_to_json
from structure_reader.model_registry import find_workspace_root
from structure_reader.scenario_evaluator import (
    check_expression,
    evaluate_int,
    evaluate_tri,
)

ROOT = find_workspace_root()


def build():
    return build_dspark_attention_graph(ROOT)


def _node_map(graph):
    return {n.stable_id: n for n in graph.nodes}


def _edges_by_id(graph):
    return {e.stable_id: e for e in graph.edges}


def _axis_map(graph):
    return {a.symbol: a for a in graph.named_axes}


def _tensor_map(graph):
    return {t.name: t for t in graph.tensors}


def _alias_expr_for(bundle, coordinate_id):
    """从实际 bundle 读出 coordinate_id 的坐标表达式原文（B2：坐标值以
    bundle 级 CoordinateExpression 显式记录，非 view alias）。"""
    for c in bundle.coordinate_expressions:
        if c.coordinate_id == coordinate_id and c.expression:
            return c.expression
    raise AssertionError(f"未找到 {coordinate_id} 的坐标表达式记录")


def _alias_expr_from_doc(doc, coordinate_id):
    """从序列化结果（bundle 公开 JSON 解析回的 doc）读坐标表达式原文。"""
    for c in doc["coordinate_expressions"]:
        if c["coordinate_id"] == coordinate_id and c["expression"]:
            return c["expression"]
    raise AssertionError(f"序列化结果未找到 {coordinate_id} 的坐标表达式记录")


def _assert_slot_coords(expr):
    """写槽坐标断言（真实图与篡改检出共用）：本次写槽 = P%W
    （decode T=1，model.py:783），必须与下一次写槽 (P+T)%W 区分。"""
    for p, expect in ((2, 2), (126, 126), (127, 127), (128, 0), (255, 127)):
        got = evaluate_int(expr, {"P": p, "T": 1, "W": 128})
        if got != expect:
            raise AssertionError(f"P={p}: 写槽求值 {got} != {expect}")
    got = evaluate_int(expr, {"P": 255, "T": 1, "W": 128})
    next_slot = evaluate_int("(P + T) % W", {"P": 255, "T": 1, "W": 128})
    if got == next_slot:
        raise AssertionError(
            f"写槽表达式把本次写槽与下一次写槽混淆（{got} == {next_slot}）")


def _assert_pos0_coords(expr):
    """草稿 RoPE 起点断言（真实图与篡改检出共用）：起点 = P+T（772），
    覆盖 P=2/126/127/128、T=1 与 prefill 短/长分支求值。"""
    for env, expect in (
        ({"P": 2, "T": 1, "W": 128, "N_draft": 5}, 3),      # decode 早期
        ({"P": 126, "T": 1, "W": 128, "N_draft": 5}, 127),  # decode 边界
        ({"P": 127, "T": 1, "W": 128, "N_draft": 5}, 128),
        ({"P": 128, "T": 1, "W": 128, "N_draft": 5}, 129),
        ({"P": 255, "T": 1, "W": 128, "N_draft": 5}, 256),
        ({"P": 0, "T": 3, "W": 128, "N_draft": 5}, 3),      # prefill 短
        ({"P": 0, "T": 129, "W": 128, "N_draft": 5}, 129),  # prefill 长
    ):
        got = evaluate_int(expr, env)
        if got != expect:
            raise AssertionError(f"{env}: 草稿起点求值 {got} != {expect}")


class TestEvidence(unittest.TestCase):
    def setUp(self):
        self.bundle = build()

    def test_all_evidence_verified(self):
        # 能构建即全部证据注册时已对原文核对通过
        entries = self.bundle.registry.all()
        self.assertGreaterEqual(len(entries), 14)
        coverages = {e.coverage for e in entries}
        self.assertEqual(
            coverages,
            {"located", "source_checked", "derived", "uncovered"},
            "四个证据等级都必须有真实条目",
        )

    def test_key_control_flow_source_checked(self):
        by_id = {e.evidence_id for e in self.bundle.registry.by_coverage(
            "source_checked")}
        for required in ("ev_ds_prefill", "ev_ds_main_kv", "ev_ds_draft_kv",
                         "ev_ds_decode", "ev_ds_topk"):
            self.assertIn(required, by_id, required)

    def test_uncovered_honest(self):
        for e in self.bundle.registry.by_coverage("uncovered"):
            self.assertTrue(e.note and e.note.strip(), e.evidence_id)
            self.assertIsNone(e.line_start, e.evidence_id)

    def test_derived_capacity_no_lines(self):
        cap = self.bundle.registry.get("ev_ds_capacity")
        self.assertEqual(cap.coverage, "derived")
        self.assertIsNone(cap.line_start)
        self.assertTrue(cap.derivation and "479" in cap.derivation)

    def test_snippet_mismatch_rejected(self):
        bad = SourceEvidence(
            evidence_id="ev_ds_test_wrong", source_key="model.py",
            qualified_symbol="DSparkAttention.forward", line_start=763,
            line_end=769, snippet="self.kv_cache[:bsz, :seqlen] = kv  # 故意错",
            claim="测试用", coverage="located")
        fresh = type(self.bundle.registry)(
            __import__("structure_reader.extract.evidence",
                       fromlist=["load_evidence_base"]).load_evidence_base(ROOT),
            ROOT)
        with self.assertRaises(EvidenceError):
            fresh.register(bad)


class TestGraphStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph

    def test_template_id(self):
        self.assertEqual(self.graph.template_id, TEMPLATE_ID)

    def test_zero_reference_issues(self):
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_canonical_json_roundtrip(self):
        text = graph_to_json(self.graph)
        rebuilt = graph_from_json(text)
        self.assertEqual(graph_to_json(rebuilt), text)
        self.assertEqual(rebuilt.template_id, TEMPLATE_ID)

    def test_entity_counts(self):
        g = self.graph
        self.assertGreaterEqual(len(g.named_axes), 18)
        self.assertGreaterEqual(len(g.tensors), 24)
        self.assertGreaterEqual(len(g.nodes), 27)
        self.assertGreaterEqual(len(g.edges), 29)
        self.assertEqual(len(g.state_versions), 2)
        self.assertEqual(len(g.state_transitions), 1)
        self.assertEqual(len(g.guards), 4)
        # B2 修正：虚假坐标 view alias 已删，唯一别名是真实 kv_cache[:bsz]
        # batch view
        self.assertEqual(len(g.alias_relations), 1)

    def test_no_fake_coordinate_view_alias(self):
        # B2 修正核心断言：坐标标量（int64）不是任何 bf16 数据张量的
        # view/别名——alias_relations 只剩真实 batch view，且没有任何
        # alias 端指向 slot_write/draft_pos0
        for a in self.graph.alias_relations:
            self.assertNotIn(
                a.alias_tensor_id,
                ("tensor:dspark:slot_write", "tensor:dspark:draft_pos0"),
                a.stable_id)
            self.assertNotIn(
                a.base_tensor_id,
                ("tensor:dspark:slot_write", "tensor:dspark:draft_pos0"),
                a.stable_id)
        aliases = {a.stable_id for a in self.graph.alias_relations}
        self.assertNotIn("alias_ds_slot_expr", aliases)
        self.assertNotIn("alias_ds_pos0_expr", aliases)
        # 真实存储 view 保留：784 kv_cache[:bsz]
        self.assertIn("alias_ds_cache_read_view", aliases)

    def test_coordinate_expressions_record_and_roundtrip(self):
        # bundle 级 CoordinateExpression：表达式原文可求值、消费端口真实、
        # 公开 JSON 往返逐字节一致；标量 int64 张量存在且零轴（非数据
        # 张量的 view）
        from structure_reader.extract.deepseek_v4_dspark import (
            bundle_from_json,
            bundle_to_json,
        )
        coords = {c.coordinate_id: c for c in self.bundle.coordinate_expressions}
        self.assertEqual(set(coords),
                         {"dspark:slot_write", "dspark:draft_pos0"})
        tm = _tensor_map(self.graph)
        for cid, c in coords.items():
            check_expression(c.expression)
            t = tm[c.scalar_tensor_id.removeprefix("tensor:")]
            self.assertEqual(t.dtype, "int64", cid)
            self.assertEqual(len(t.axes), 0, cid)
            node_map = {n.stable_id: n for n in self.graph.nodes}
            n = node_map[c.consumer_node_id]
            ports = {p.name: p.tensor_id for p in n.input_ports}
            self.assertEqual(ports.get(c.consumer_port),
                             c.scalar_tensor_id, cid)
            self.assertTrue(c.claim.strip(), cid)
        text = bundle_to_json(self.bundle)
        rebuilt = bundle_from_json(text)
        self.assertEqual(bundle_to_json(rebuilt), text)
        self.assertEqual(
            {c.coordinate_id: c.expression
             for c in rebuilt.coordinate_expressions},
            {cid: c.expression for cid, c in coords.items()})

    def test_coordinate_tamper_detected_in_json(self):
        # 篡改 bundle JSON 中坐标记录的表达式 → 读取检出（与图内篡改
        # 检测互补：公开 JSON 通道上的篡改同样可检出）
        from structure_reader.extract.deepseek_v4_dspark import bundle_to_json
        text = bundle_to_json(build())
        doc = json.loads(text)
        for c in doc["coordinate_expressions"]:
            if c["coordinate_id"] == "dspark:slot_write":
                c["expression"] = "P // W"
        tampered = _alias_expr_from_doc(doc, "dspark:slot_write")
        with self.assertRaises(AssertionError):
            _assert_slot_coords(tampered)

    def test_no_compressor_indexer(self):
        # DSparkAttention assert compress_ratio==0：无压缩路径——
        # 无 compressor/indexer 节点、无压缩张量、无压缩轴
        for n in self.graph.nodes:
            self.assertNotIn("compressor", n.stable_id, n.stable_id)
            self.assertNotIn("indexer", n.stable_id, n.stable_id)
        for t in self.graph.tensors:
            self.assertFalse(t.name.startswith("dspark:comp_"), t.name)
            self.assertFalse(t.name.startswith("dspark:idx_comp"), t.name)
        symbols = {a.symbol for a in self.graph.named_axes}
        for banned in ("r", "N_cap", "N_cs", "N_vis", "N_sel_c", "N_cc",
                       "N_ctx", "N_blk", "D2", "Ki"):
            self.assertNotIn(banned, symbols, banned)


class TestPrefillPhase(unittest.TestCase):
    """prefill：仅 cache 构建 return x——无 Q/索引/sparse_attn。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = _node_map(cls.graph)
        cls.edges = _edges_by_id(cls.graph)

    def test_prefill_return_passthrough(self):
        n = self.nodes["dspark.prefill_return"]
        self.assertEqual(n.op_type, "passthrough")
        in_tensors = [p.tensor_id for p in n.input_ports]
        out_tensors = [p.tensor_id for p in n.output_ports]
        self.assertEqual(in_tensors, ["tensor:dspark:x_draft_hc"])
        self.assertEqual(out_tensors, ["tensor:dspark:x_draft_hc"])
        self.assertIn("ev_ds_prefill", n.evidence_refs)

    def test_prefill_input_edge_guarded_prefill_only(self):
        # 图输入 x_draft_hc 经显式源节点消费，passthrough 消费边挂
        # g_prefill——prefill passthrough 只能 prefill
        e = self.edges["e_ds_hc_in_pref"]
        self.assertEqual(e.src_node_id, "dspark.x_draft_hc_src")
        self.assertEqual(e.dst_node_id, "dspark.prefill_return")
        self.assertEqual(e.tensor_id, "tensor:dspark:x_draft_hc")
        self.assertEqual(e.guard_ref, "g_prefill")

    def test_prefill_input_shape_hc_stream(self):
        # prefill 分支 attn 的 x 是 hc 展开流 [B, N_draft, N_hc, H]
        # （DSparkBlock.forward_embed unsqueeze+repeat，model.py:858-861）
        t = _tensor_map(self.graph)["dspark:x_draft_hc"]
        self.assertEqual([a.symbol for a in t.axes], ["B", "N_draft", "N_hc", "H"])

    def test_cache_write_branch_guards(self):
        # 三条真实写分支：prefill T<=W 顺序 / prefill T>W 环形 / decode 单槽
        branches = {
            "e_ds_cachew_inwin": "g_dspark_prefill_inwin",
            "e_ds_cachew_wrap": "g_window_wrap",
            "e_ds_cachew_dec": "g_decode",
        }
        guard_ids = {g.stable_id for g in self.graph.guards}
        for eid, guard in branches.items():
            self.assertIn(guard, guard_ids, guard)
            self.assertEqual(self.edges[eid].guard_ref, guard, eid)
            self.assertEqual(self.edges[eid].src_node_id, "dspark.main_quant", eid)

    def test_no_q_or_attention_in_prefill(self):
        # sparse_attn 全部入边（含静态 sink 参数的消费边）挂 g_decode：
        # prefill 在 769 return，不产生 Q/索引/注意力计算
        q_edges = ["e_ds_sparse_q", "e_ds_cat_sparse", "e_ds_idx_select",
                   "e_ds_sparse_sink"]
        for eid in q_edges:
            self.assertEqual(self.edges[eid].guard_ref, "g_decode", eid)


class TestDecodePhase(unittest.TestCase):
    """decode：草稿 Q 低秩/归一化/RoPE、位置 offset、环形槽、索引与注意力。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = _node_map(cls.graph)
        cls.edges = _edges_by_id(cls.graph)

    def test_draft_input_shape_single_stream(self):
        # decode 的 x 是 Block.forward hc_pre 后单流 [B, N_draft, H]
        t = _tensor_map(self.graph)["dspark:x_draft_single"]
        self.assertEqual([a.symbol for a in t.axes], ["B", "N_draft", "H"])

    def test_draft_q_path(self):
        # 774-777：wq_a→q_norm→wq_b→unflatten→RMS 缩放→RoPE
        chain = [("dspark.wq_a", "linear"),
                 ("dspark.q_norm", "rms_norm"),
                 ("dspark.wq_b", "linear"),
                 ("dspark.q_unflatten", "unflatten"),
                 ("dspark.q_rms_scale", "rms_scale"),
                 ("dspark.q_rope", "rope_partial")]
        for sid, op in chain:
            self.assertEqual(self.nodes[sid].op_type, op, sid)
        # 全链证据锚定草稿 Q 源码段
        for sid in ("dspark.wq_a", "dspark.q_rms_scale"):
            self.assertIn("ev_ds_draft_q", self.nodes[sid].evidence_refs, sid)
        # RoPE 消费草稿位置 offset 证据（772）
        self.assertIn("ev_ds_draft_pos", self.nodes["dspark.q_rope"].evidence_refs)

    def test_draft_position_offset_axis(self):
        # 草稿位置 = start_pos + seqlen 起（772）：张量轴是草稿局部
        # N_draft，位置坐标由具名标量 + alias transform_expr 表达
        # （test_draft_pos0_expression_from_graph）
        t = _tensor_map(self.graph)["dspark:q_rope"]
        self.assertEqual([a.symbol for a in t.axes], ["B", "N_draft", "Hq", "D"])

    def test_draft_pos0_expression_from_graph(self):
        # 从实际 bundle 读取草稿 RoPE 位置起点坐标：q_rope.pos 端口消费的
        # 标量 draft_pos0，值语义是 CoordinateExpression 原文（P + T）——
        # 读取实际表达式求值（P=2/126/127/128、T=1；prefill 短/长分支），
        # 不是复制实现常量断言
        coord_id = "dspark:draft_pos0"
        expr = _alias_expr_for(self.bundle, coord_id)
        check_expression(expr)  # 与图同源求值：非白名单表达式拒绝
        _assert_pos0_coords(expr)
        pos_tensor = "tensor:dspark:draft_pos0"
        rope_users = {"dspark.q_rope", "dspark.kv_rope",
                      "dspark.o_rope_inv"}
        # 实际消费端口存在：三个 RoPE 算子都在 pos 上消费该标量
        for sid in rope_users:
            ports = {p.name: p.tensor_id for p in self.nodes[sid].input_ports}
            self.assertEqual(ports.get("pos"), pos_tensor, sid)
        for n in self.graph.nodes:
            pos_ports = [p for p in n.input_ports
                         if p.name == "pos" and p.tensor_id == pos_tensor]
            if pos_ports:
                self.assertIn(n.stable_id, rope_users, n.stable_id)
        # 记录声称的消费端口与图一致（记录可被图自身证伪）
        rec = next(c for c in self.bundle.coordinate_expressions
                   if c.coordinate_id == coord_id)
        self.assertEqual(rec.consumer_node_id, "dspark.q_rope")
        self.assertEqual(rec.consumer_port, "pos")
        self.assertEqual(rec.scalar_tensor_id, pos_tensor)
        self.assertIn("非 shape 轴", rec.claim)
        self.assertIn("view", rec.claim)

    def test_pos0_tamper_detected(self):
        # 故意篡改：把草稿起点 P+T 改成 P——从 bundle 序列化结果读取
        # 篡改后的表达式，针对性断言必须失败/被检出（不是复制实现常量）
        bundle = build()
        text = bundle_to_json(bundle)
        doc = json.loads(text)
        expr = _alias_expr_from_doc(doc, "dspark:draft_pos0")
        _assert_pos0_coords(expr)  # 真实图通过
        for c in doc["coordinate_expressions"]:
            if c["coordinate_id"] == "dspark:draft_pos0":
                c["expression"] = "P"
        tampered = _alias_expr_from_doc(doc, "dspark:draft_pos0")
        with self.assertRaises(AssertionError):
            _assert_pos0_coords(tampered)

    def test_select_width_no_minus1_padding(self):
        # get_dspark_topk_idxs：宽 = min(W,P+1) + N_draft，无 -1 填充；
        # P=2（早期 decode）→ 3+5=8；P>=127 → W+5=133
        env = {"W": 128, "N_draft": 5}
        for p, expect in ((2, 8), (3, 9), (126, 132), (127, 133), (200, 133)):
            self.assertEqual(evaluate_int(EXPR_SEL_WIDTH, {**env, "P": p}),
                             expect, f"P={p}")
        axis = _axis_map(self.graph)["N_sel_ds"]
        self.assertEqual(axis.expression, EXPR_SEL_WIDTH)
        self.assertIn("无 -1", axis.meaning)

    def test_select_edge_kind(self):
        # 索引输出是选择语义（select），不是 Value
        e = self.edges["e_ds_idx_select"]
        self.assertEqual(e.kind, "select")
        self.assertEqual(e.dst_node_id, "dspark.sparse_attn")
        self.assertIn("ev_ds_topk", self.nodes["dspark.idx_build"].evidence_refs)

    def test_cache_ring_slot_and_state(self):
        # 主 cache 槽 start_pos%W（783）：decode 写边挂 g_decode；
        # 写→读同 state_id 版本对；state 边引用 tr_dspark_cache；
        # 写后读 state 边挂 g_decode（cache_read 只在 decode 实例化）
        self.assertEqual(self.edges["e_ds_cachew_dec"].guard_ref, "g_decode")
        state_edge = self.edges["e_ds_cache_state"]
        self.assertEqual(state_edge.kind, "state")
        self.assertEqual(state_edge.state_transition_ref, "tr_dspark_cache")
        self.assertEqual(state_edge.guard_ref, "g_decode")
        transitions = {t.stable_id: t for t in self.graph.state_transitions}
        tr = transitions["tr_dspark_cache"]
        versions = {v.stable_id: v for v in self.graph.state_versions}
        self.assertEqual(versions[tr.before_version_id].state_id,
                         "dspark:win_cache")
        self.assertEqual(versions[tr.before_version_id].version_label, "v_hist")
        self.assertEqual(versions[tr.after_version_id].version_label, "v_after")

    def test_cache_write_slot_expression_from_graph(self):
        # 从实际 bundle 读取 decode 写槽坐标：cache_update.slot 端口消费的
        # 标量 slot_write，值语义是 CoordinateExpression 原文（P % W）——
        # 读取实际表达式求值（P=2/126/127/128/255、T=1），不是复制实现常量
        coord_id = "dspark:slot_write"
        expr = _alias_expr_for(self.bundle, coord_id)
        check_expression(expr)
        _assert_slot_coords(expr)
        slot_tensor = "tensor:dspark:slot_write"
        upd = self.nodes["dspark.cache_update"]
        in_ports = {p.name: p.tensor_id for p in upd.input_ports}
        self.assertEqual(in_ports.get("slot"), slot_tensor)
        slot_edge = self.edges["e_slot_cachew"]
        self.assertEqual(slot_edge.src_node_id, "dspark.slot_src")
        self.assertEqual(slot_edge.dst_port, "slot")
        self.assertEqual(slot_edge.tensor_id, slot_tensor)
        self.assertEqual(slot_edge.guard_ref, "g_decode")
        # 记录声称的消费端口与图一致（记录可被图自身证伪）
        rec = next(c for c in self.bundle.coordinate_expressions
                   if c.coordinate_id == coord_id)
        self.assertEqual(rec.consumer_node_id, "dspark.cache_update")
        self.assertEqual(rec.consumer_port, "slot")
        self.assertEqual(rec.scalar_tensor_id, slot_tensor)
        self.assertIn("非 shape 轴", rec.claim)

    def test_slot_tamper_detected(self):
        # 故意篡改：把图内写槽表达式 P%W 改成 (P+T)%W——从 bundle 序列化
        # 结果读取篡改后的表达式，针对性断言必须失败/被检出
        bundle = build()
        text = bundle_to_json(bundle)
        doc = json.loads(text)
        expr = _alias_expr_from_doc(doc, "dspark:slot_write")
        _assert_slot_coords(expr)  # 真实图通过
        for c in doc["coordinate_expressions"]:
            if c["coordinate_id"] == "dspark:slot_write":
                c["expression"] = "(P + T) % W"
        tampered = _alias_expr_from_doc(doc, "dspark:slot_write")
        with self.assertRaises(AssertionError):
            _assert_slot_coords(tampered)

    def test_cache_read_batch_view_alias(self):
        # 784 读 self.kv_cache[:bsz]：batch 前缀 view（非独立拷贝）
        aliases = {a.stable_id: a for a in self.graph.alias_relations}
        a = aliases["alias_ds_cache_read_view"]
        self.assertEqual(a.relation_kind, "view")
        self.assertEqual(a.base_tensor_id, "tensor:dspark:win_cache")
        self.assertEqual(a.alias_tensor_id, "tensor:dspark:kv_win_read")
        self.assertIn("kv_cache[:bsz]", a.transform_expr)

    def test_kv_concat_width(self):
        # 784：物理宽 = W + N_draft（窗口 cache 槽 + 草稿 KV）
        self.assertEqual(EXPR_KV_CONCAT_WIDTH, "W + N_draft")
        env = {"W": 128, "N_draft": 5}
        for p in (2, 127, 128):
            self.assertEqual(
                evaluate_int(EXPR_KV_CONCAT_WIDTH, {**env, "P": p}), 133, f"P={p}")
        axis = _axis_map(self.graph)["N_kvcat"]
        self.assertEqual(axis.expression, EXPR_KV_CONCAT_WIDTH)
        # concat 节点两输入：窗口读 + 草稿 KV
        cat = self.nodes["dspark.kv_cat"]
        in_ports = {p.name: p.tensor_id for p in cat.input_ports}
        self.assertEqual(in_ports["win"], "tensor:dspark:kv_win_read")
        self.assertEqual(in_ports["draft"], "tensor:dspark:draft_kv_quant")
        self.assertEqual(
            [a.symbol for a in _tensor_map(self.graph)["dspark:kv_concat"].axes],
            ["B", "N_kvcat", "D"])

    def test_sparse_attn_and_o_chain(self):
        # 785-791：sparse_attn（不透明 kernel）→ inverse RoPE → reshape →
        # 分组低秩 einsum → flatten → wo_b
        sa = self.nodes["dspark.sparse_attn"]
        self.assertEqual(sa.op_type, "sparse_attention")
        self.assertEqual(sa.granularity, "opaque_kernel")
        in_ports = {p.name: p.tensor_id for p in sa.input_ports}
        self.assertEqual(set(in_ports),
                         {"q", "kv", "topk_idxs", "sink"})
        chain = [("dspark.o_rope_inv", "rope_partial_inverse"),
                 ("dspark.o_reshape", "reshape"),
                 ("dspark.o_einsum", "grouped_lowrank_einsum"),
                 ("dspark.o_flatten", "flatten"),
                 ("dspark.wo_b", "linear")]
        for sid, op in chain:
            self.assertEqual(self.nodes[sid].op_type, op, sid)
        self.assertIn("ev_ds_sparse", self.nodes["dspark.o_rope_inv"].evidence_refs)
        self.assertIn("ev_ds_ogroup", self.nodes["dspark.wo_b"].evidence_refs)

    def test_output_shape(self):
        t = _tensor_map(self.graph)["dspark:attn_out"]
        self.assertEqual([a.symbol for a in t.axes], ["B", "N_draft", "H"])


class TestSharedKVProjection(unittest.TestCase):
    """KV 单投影共享：wkv/kv_norm 同一实例两次调用（main 759 / draft 778）。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = _node_map(cls.graph)
        cls.edges = _edges_by_id(cls.graph)

    def test_shared_weight_fanout(self):
        # 权重节点扇出两条边：main 与 draft 调用点消费同一权重张量；
        # main 消费边两相位共享，draft 消费边挂 g_decode（静态参数身份
        # 两相位可见，不因此把 decode 计算边标为 prefill 运行）
        out_w = [e for e in self.graph.edges
                 if e.src_node_id == "dspark.wkv_w"]
        self.assertEqual(len(out_w), 2, "wkv_w 必须恰好扇出到 main/draft 两处")
        self.assertEqual({e.dst_node_id for e in out_w},
                         {"dspark.wkv_main", "dspark.wkv_draft"})
        w_tensor = {e.tensor_id for e in out_w}
        self.assertEqual(w_tensor, {"tensor:dspark:wkv_w"})
        out_n = [e for e in self.graph.edges
                 if e.src_node_id == "dspark.kv_norm_w"]
        self.assertEqual({e.dst_node_id for e in out_n},
                         {"dspark.kv_norm_main", "dspark.kv_norm_draft"})
        guards_by_id = {e.stable_id: e.guard_ref for e in self.graph.edges}
        self.assertIsNone(guards_by_id["e_ds_wkv_w_main"])
        self.assertIsNone(guards_by_id["e_ds_kvnorm_w_main"])
        self.assertEqual(guards_by_id["e_ds_wkv_w_draft"], "g_decode")
        self.assertEqual(guards_by_id["e_ds_kvnorm_w_draft"], "g_decode")

    def test_call_sites_consume_shared_weight(self):
        for sid in ("dspark.wkv_main", "dspark.wkv_draft"):
            ports = {p.name: p.tensor_id for p in self.nodes[sid].input_ports}
            self.assertEqual(ports.get("w"), "tensor:dspark:wkv_w", sid)
        for sid in ("dspark.kv_norm_main", "dspark.kv_norm_draft"):
            ports = {p.name: p.tensor_id for p in self.nodes[sid].input_ports}
            self.assertEqual(ports.get("w"), "tensor:dspark:kv_norm_w", sid)

    def test_same_wkv_evidence_both_paths(self):
        # main（759）与 draft（778）证据都锚定 self.kv_norm(self.wkv(...))
        self.assertIn("ev_ds_main_kv", self.nodes["dspark.wkv_main"].evidence_refs)
        self.assertIn("ev_ds_draft_kv", self.nodes["dspark.wkv_draft"].evidence_refs)

    def test_draft_kv_same_shape_as_main_projection(self):
        # 同一 wkv（Linear(H, D)）：main [B,T,D]、draft [B,N_draft,D]
        tm = _tensor_map(self.graph)
        self.assertEqual([a.symbol for a in tm["dspark:main_kv_proj"].axes],
                         ["B", "T", "D"])
        self.assertEqual([a.symbol for a in tm["dspark:draft_kv_proj"].axes],
                         ["B", "N_draft", "D"])
        w = tm["dspark:wkv_w"]
        self.assertEqual([a.symbol for a in w.axes], ["D", "H"])


class TestSharedParamsAndPhaseEdges(unittest.TestCase):
    """共享参数结构 + 相位边完整性。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = _node_map(cls.graph)
        cls.edges = _edges_by_id(cls.graph)

    def test_main_kv_path_phase_free(self):
        # main 路径两相位共享（758-761 在相位分支之前）：wkv_main→kv_norm_main
        # →rope→quant 的链边与 main 调用点的参数消费边不挂相位 guard
        for eid in ("e_ds_main_wkv", "e_ds_main_norm_rope",
                    "e_ds_main_rope_quant", "e_ds_wkv_w_main",
                    "e_ds_kvnorm_w_main"):
            self.assertIsNone(self.edges[eid].guard_ref, eid)
        # 图输入 x_main_in 经显式源节点消费（两相位共享，不挂 guard）
        e = self.edges["e_ds_main_in_wkv"]
        self.assertEqual(e.src_node_id, "dspark.x_main_in_src")
        self.assertEqual(e.guard_ref, None, "main 输入消费边两相位共享")

    def test_quant_inplace_single_node(self):
        # 761/780 act_quant in-place（第四参 True）：节点自环
        # （输入=输出同一张量），非第二套量化
        for sid in ("dspark.main_quant", "dspark.kv_quant"):
            n = self.nodes[sid]
            self.assertEqual(n.op_type, "act_quant_fp8", sid)
            in_t = [p.tensor_id for p in n.input_ports]
            out_t = [p.tensor_id for p in n.output_ports]
            self.assertEqual(in_t, out_t, sid)

    def test_sink_param_shape(self):
        # attn_sink [Hq] fp32（model.py:462），非 hc sinkhorn post
        t = _tensor_map(self.graph)["dspark:attn_sink_param"]
        self.assertEqual([a.symbol for a in t.axes], ["Hq"])
        self.assertEqual(t.dtype, "fp32")

    def test_guard_predicates_expressions(self):
        preds = {g.stable_id: g.predicate_expr for g in self.graph.guards}
        self.assertEqual(preds["g_prefill"], "start_pos == 0")
        self.assertEqual(preds["g_decode"], "start_pos > 0")
        self.assertEqual(preds["g_dspark_prefill_inwin"],
                         "start_pos == 0 and T <= W")
        # 本图 g_window_wrap 限定 prefill：decode 主 seqlen=1 是实际调用
        # 前提，非法长 decode 不激活 prefill 回绕分支
        self.assertEqual(preds["g_window_wrap"], "start_pos == 0 and T > W")

    def test_guard_activation_tri_state(self):
        # 相位边界三态求值：prefill 短/长、decode；表达式与图 guard 同源。
        # g_window_wrap 在 decode P=0 长序列 env 下必须为 False（相位限定）
        preds = {g.stable_id: g.predicate_expr for g in self.graph.guards}
        W = 128
        cases = [
            ({"start_pos": 0, "T": 3, "W": W},
             {"g_prefill": True, "g_decode": False,
              "g_dspark_prefill_inwin": True, "g_window_wrap": False}),
            ({"start_pos": 0, "T": 129, "W": W},
             {"g_prefill": True, "g_decode": False,
              "g_dspark_prefill_inwin": False, "g_window_wrap": True}),
            ({"start_pos": 2, "T": 1, "W": W},
             {"g_prefill": False, "g_decode": True,
              "g_dspark_prefill_inwin": False, "g_window_wrap": False}),
            ({"start_pos": 128, "T": 1, "W": W},
             {"g_prefill": False, "g_decode": True,
              "g_dspark_prefill_inwin": False, "g_window_wrap": False}),
        ]
        for env, expected in cases:
            for gid, want in expected.items():
                result = evaluate_tri(preds[gid], env)
                self.assertTrue(result.is_known, (gid, env))
                self.assertEqual(result.value, want, (gid, env))
        # 非法长 decode（T=129 而非 seqlen=1）不激活 prefill 回绕：
        # decode 相位下 g_window_wrap 恒 False
        bad_decode = {"start_pos": 5, "T": 129, "W": W}
        self.assertFalse(evaluate_tri(preds["g_window_wrap"], bad_decode).value)
        self.assertTrue(evaluate_tri(preds["g_decode"], bad_decode).value)

    def test_decode_runtime_chain_guard_eval(self):
        # 验证实际序列化图的 guard 求值：prefill env 屏蔽全部草稿计算链、
        # decode env 保留真实草稿链。逐边从序列化图读取 guard 谓词求值，
        # 不复制实现里的边清单。
        text = graph_to_json(self.graph)
        doc = json.loads(text)
        preds = {g["stable_id"]: g["predicate_expr"] for g in doc["guards"]}
        prefill_env = {"start_pos": 0, "T": 200, "P": 0, "W": 128,
                       "N_draft": 5}
        decode_env = {"start_pos": 7, "T": 1, "P": 7, "W": 128,
                      "N_draft": 5}
        draft_edges = []
        for e in doc["edges"]:
            ref = e["guard_ref"]
            if ref is None:
                continue
            active = evaluate_tri(preds[ref], decode_env)
            masked = evaluate_tri(preds[ref], prefill_env)
            self.assertTrue(active.is_known and masked.is_known, e["stable_id"])
            if active.value and not masked.value:
                draft_edges.append(e["stable_id"])
        # decode 激活而 prefill 屏蔽的边 = 真实草稿链（含 O 链），
        # 必须覆盖全部关键运行时算子的入边
        self.assertGreaterEqual(len(draft_edges), 20)
        for must in ("e_ds_xin_wqa", "e_ds_xin_wkvd", "e_ds_qa_qnorm",
                     "e_ds_qnorm_wqb",
                     "e_ds_wqb_unfl", "e_ds_unfl_scale", "e_ds_scale_rope",
                     "e_ds_draft_wkv", "e_ds_draft_norm_rope",
                     "e_ds_draft_rope_quant", "e_ds_read_cat",
                     "e_ds_draft_cat", "e_ds_sparse_q", "e_ds_cat_sparse",
                     "e_ds_idx_select", "e_ds_sparse_sink",
                     "e_ds_sparse_o", "e_ds_oinv_reshape",
                     "e_ds_reshape_einsum", "e_ds_einsum_flat",
                     "e_ds_flat_wob", "e_ds_cachew_dec", "e_ds_cache_state",
                     "e_slot_cachew", "e_pos_qrope", "e_pos_kvrope",
                     "e_pos_oinv", "e_ds_wkv_w_draft", "e_ds_kvnorm_w_draft"):
            self.assertIn(must, draft_edges, must)
        # prefill env（T=200>W 回绕场景）下激活 guard 的边只能是：
        # prefill passthrough 消费边与 prefill 两条 cache 写分支
        prefill_allowed = {"e_ds_hc_in_pref", "e_ds_cachew_inwin",
                           "e_ds_cachew_wrap"}
        for e in doc["edges"]:
            ref = e["guard_ref"]
            if ref is None:
                self.assertIn(e["stable_id"],
                              {"e_ds_main_wkv", "e_ds_main_norm_rope",
                               "e_ds_main_rope_quant", "e_ds_wkv_w_main",
                               "e_ds_kvnorm_w_main",
                               "e_ds_main_in_wkv"})
                continue
            masked = evaluate_tri(preds[ref], prefill_env)
            if masked.value:
                self.assertIn(e["stable_id"], prefill_allowed,
                              e["stable_id"])

    def test_cache_slot_ring_expression(self):
        # 环形槽位：图内 decode 写槽坐标表达式（test_cache_write_slot_
        # expression_from_graph 逐点求值 P%W）；此处只验证测试自算的
        # (P+T)%W 事实与本次写槽的关系——逐 decode 步后下一槽 = (P+1)%W，
        # 而本次写槽 = P%W（T=1 时二者相差一步，不混淆）
        W = 128
        for p, slot_now, slot_next in ((2, 2, 3), (126, 126, 127),
                                       (127, 127, 0), (128, 0, 1),
                                       (255, 127, 0)):
            self.assertEqual(p % W, slot_now, (p, "本次写槽"))
            self.assertEqual((p + 1) % W, slot_next, (p, "下一步写槽"))
        # prefill T=129：cutoff = T%W，回绕写前 cutoff 槽 + 后 W-cutoff 槽
        t_len = 129
        self.assertEqual(t_len % W, 1, "T=129 → cutoff=1")

    def test_window_capacity_is_w(self):
        # r=0：cache 容量 = W（无压缩区项）——win_cache 轴 [B, W, D]
        t = _tensor_map(self.graph)["dspark:win_cache"]
        self.assertEqual([a.symbol for a in t.axes], ["B", "W", "D"])
        # values 来自 run config：MTP 层 compress_ratios 全 0
        values = self.bundle.values
        tail = values.compress_ratios[values.n_layers:]
        self.assertEqual(len(tail), values.n_mtp_layers)
        self.assertTrue(all(v == 0 for v in tail))
        self.assertEqual(values.dspark_block_size, 5)
        self.assertEqual(values.dspark_target_layer_ids, (40, 41, 42))

    def test_rebuild_deterministic(self):
        # 重复构建 canonical JSON 逐字节一致（无隐藏状态）
        text1 = graph_to_json(build().graph)
        text2 = graph_to_json(build().graph)
        self.assertEqual(text1, text2)


class TestAllEdgesPhaseEval(unittest.TestCase):
    """全边三相位求值（B2 补修 01 统一标准）：None=真、逐边可求值性、
    短 prefill/长 prefill/decode 分别求值——不跳过任何边（与 stage 图
    test_all_edges_evaluated_three_phases 同一标准，B1 图同验）。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.doc = json.loads(graph_to_json(cls.graph))
        cls.preds = {g["stable_id"]: g["predicate_expr"]
                     for g in cls.doc["guards"]}

    def _active(self, env):
        active = set()
        for e in self.doc["edges"]:
            ref = e["guard_ref"]
            if ref is None:
                active.add(e["stable_id"])
                continue
            r = evaluate_tri(self.preds[ref], env)
            if not r.is_known:
                raise AssertionError(f"{e['stable_id']}: guard {ref!r} 不可求值")
            if r.value:
                active.add(e["stable_id"])
        return active

    def test_three_phase_eval_all_edges(self):
        active_ps = self._active({"start_pos": 0, "T": 3, "P": 0, "W": 128,
                                  "N_draft": 5})
        active_pl = self._active({"start_pos": 0, "T": 200, "P": 0, "W": 128,
                                  "N_draft": 5})
        active_d = self._active({"start_pos": 7, "T": 1, "P": 7, "W": 128,
                                 "N_draft": 5})
        # 短/长 prefill 差异只在两条 cache 写分支之间互换
        self.assertEqual(active_ps - active_pl, {"e_ds_cachew_inwin"})
        self.assertEqual(active_pl - active_ps, {"e_ds_cachew_wrap"})
        # prefill 激活边 = prefill 分支（passthrough/cache 写）+ main 链
        # （两相位共享）；无草稿链。短 prefill 激活 inwin 分支、长 prefill
        # 激活 wrap 分支（真实分支结构）
        main_chain = {"e_ds_main_in_wkv", "e_ds_main_wkv",
                      "e_ds_main_norm_rope", "e_ds_main_rope_quant",
                      "e_ds_wkv_w_main", "e_ds_kvnorm_w_main"}
        self.assertEqual(active_ps, main_chain | {"e_ds_hc_in_pref",
                                                  "e_ds_cachew_inwin"},
                         "prefill 短")
        self.assertEqual(active_pl, main_chain | {"e_ds_hc_in_pref",
                                                  "e_ds_cachew_wrap"},
                         "prefill 长")
        # decode 包含 main 链 + 完整草稿链（含 O 链与 cache 写后读）
        self.assertTrue(active_d >= main_chain)
        for must in ("e_ds_xin_wqa", "e_ds_draft_wkv", "e_ds_sparse_q",
                     "e_ds_flat_wob", "e_ds_cache_state", "e_slot_cachew"):
            self.assertIn(must, active_d, must)
        self.assertNotIn("e_ds_hc_in_pref", active_d,
                         "prefill passthrough 不在 decode 激活")

    def test_none_guard_edges_are_main_chain_only(self):
        # None（无条件）边全部属于两相位共享 main 链（main_x 源→wkv→norm
        # →rope→quant + 参数到 main 调用点扇出）——审查要求 None=真逐边
        # 实算，白名单断言保证无隐藏的 decode/prefill 专用无条件边
        none_edges = {e["stable_id"] for e in self.doc["edges"]
                      if e["guard_ref"] is None}
        self.assertEqual(
            none_edges,
            {"e_ds_main_in_wkv", "e_ds_main_wkv", "e_ds_main_norm_rope",
             "e_ds_main_rope_quant", "e_ds_wkv_w_main",
             "e_ds_kvnorm_w_main"})

    def test_stage_graph_guard_eval_consistency(self):
        # stage 图 attention 内部成员边与 B1 图同 ID 边相位求值一致
        # （同源单一实现的求值不变量；经 stage bundle 的序列化图核对）
        from structure_reader.extract.dspark_stage import (
            build_dspark_stage_template,
        )
        stage = build_dspark_stage_template(ROOT)
        sdoc = json.loads(graph_to_json(stage.graph))
        spreds = {g["stable_id"]: g["predicate_expr"] for g in sdoc["guards"]}
        decode_env = {"start_pos": 7, "T": 1, "P": 7, "W": 128,
                      "N_draft": 5}
        checked = 0
        for e in self.doc["edges"]:
            sid = e["stable_id"]
            if e["guard_ref"] is None:
                continue
            s_edges = {x["stable_id"]: x for x in sdoc["edges"]}
            if sid not in s_edges:
                continue
            b1_on = evaluate_tri(self.preds[e["guard_ref"]], decode_env).value
            s_on = evaluate_tri(spreds[s_edges[sid]["guard_ref"]],
                                decode_env).value
            self.assertEqual(bool(b1_on), bool(s_on), sid)
            checked += 1
        self.assertGreaterEqual(checked, 28)


if __name__ == "__main__":
    unittest.main()
