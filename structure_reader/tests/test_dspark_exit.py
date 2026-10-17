"""DSpark 末出口 head 链最小测试（B3b 限定恢复02，resume_b3b02_head.md）。

本轮只覆盖 head 链（hc_head→norm→共享 head full_logits）：真实图构建、
guard 机器可检、canonical 往返、重复构建一致、角色映射 canonical 往返。
Markov/Confidence 未实现，本测试不声称覆盖。

负例（删边/一致重接/共享 head 复制/prefill 误启/错 last 角色）在
TestExitTamper。fix01 补修负例：gate 算子突变（sigmoid_gate→sinkhorn）、
必要节点粒度/端口面突变、同步改轴值（V 129280→129281、伪造 B 数值）、
n_stages 顶层篡改/类型突变、单stage first&last 合成往返（n_stages 核对
非硬编码 3 的证明）。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_reference import ExtractionError
from structure_reader.extract.dspark_exit import (
    EXIT_SCHEMA,
    EXIT_TEMPLATE_ID,
    G_DECODE_EXPR,
    G_DECODE_ID,
    build_dspark_exit_graph,
    build_exit_role_map,
    deserialize_exit_role_map,
    serialize_exit_role_map,
    verify_exit_graph,
)
from structure_reader.graph_document import graph_from_json, graph_to_json
from structure_reader.model_registry import find_workspace_root
from structure_reader.scenario_evaluator import evaluate_bool

ROOT = find_workspace_root()


def build():
    return build_dspark_exit_graph(ROOT)


class TestExitGraph(unittest.TestCase):
    """末出口 head 链图结构：真实节点/边、guard 机器可检、canonical
    往返、重复构建一致。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = {n.stable_id: n for n in cls.graph.nodes}
        cls.edges = {e.stable_id: e for e in cls.graph.edges}
        cls.doc = json.loads(graph_to_json(cls.graph))

    def test_template_id(self):
        self.assertEqual(self.graph.template_id, EXIT_TEMPLATE_ID)

    def test_zero_reference_issues(self):
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_chain_real_nodes_and_edges(self):
        """图有真实节点/边：h_out→hc_head(flatten/mix/gate/reduce)→
        norm→head→logits_out，参数身份/端口可联查。"""
        n = self.nodes
        e = self.edges
        # hc_head 参数身份节点与消费边
        for pid, port in (("exit.hc_head_fn_w", "w"),
                          ("exit.hc_head_scale_w", "scale"),
                          ("exit.hc_head_base_w", "base")):
            self.assertIn(pid, n)
        self.assertIn(("exit.hc_head_fn_w", "exit.hc_head.mix"),
                      [(s, d) for s, d in
                       ((x.src_node_id, x.dst_node_id) for x in e.values())])
        # 主链节点
        for sid in ("exit.hc_head.flatten", "exit.hc_head.mix",
                    "exit.hc_head.gate", "exit.hc_head.reduce",
                    "exit.norm", "exit.head", "exit.logits_out"):
            self.assertIn(sid, n)
        # 必要链边存在
        for eid in ("e_ex_hin_flat", "e_ex_fnw_mix", "e_ex_scale_gate",
                    "e_ex_base_gate", "e_ex_gate_reduce",
                    "e_ex_prenorm_norm", "e_ex_norm_head",
                    "e_ex_headw_head", "e_ex_logits_out"):
            self.assertIn(eid, e)

    def test_param_identity_shapes_and_dtype(self):
        """参数身份/shape/dtype：hc 参数 fp32（scale (C1,) 非标量）、
        head_w fp32 [V,H]、logits fp32 [B,K,V]、h_out bf16
        [B,K,N_hc,H]。"""
        tensors = {t.name: t for t in self.graph.tensors}
        cases = {
            "exit:h_out": ("bf16", ("B", "N_draft", "N_hc", "H")),
            "exit:hc_head_fn_w": ("fp32", ("N_hc", "N_hcH")),
            "exit:hc_head_scale_w": ("fp32", ("C1",)),
            "exit:hc_head_base_w": ("fp32", ("N_hc",)),
            "exit:norm_w": ("fp32", ("H",)),
            "exit:head_w": ("fp32", ("V", "H")),
            "exit:logits": ("fp32", ("B", "N_draft", "V")),
        }
        for name, (dtype, syms) in cases.items():
            t_ = tensors[name]
            self.assertEqual(t_.dtype, dtype, name)
            self.assertEqual(tuple(t_.logical_layout), syms, name)

    def test_shared_head_single_identity(self):
        """共享 head 不可复制：head_w 恰一、唯一消费边 exit.head.w。"""
        head_w = [t for t in self.graph.tensors if t.name == "exit:head_w"]
        self.assertEqual(len(head_w), 1)
        consumers = [e for e in self.graph.edges
                     if e.tensor_id == head_w[0].stable_id]
        self.assertEqual(len(consumers), 1)
        self.assertEqual(consumers[0].dst_node_id, "exit.head")
        self.assertEqual(consumers[0].dst_port, "w")

    def test_decode_only_guard_on_every_edge(self):
        """整条 head 链 decode 活动、prefill 不活动：唯一 g_decode guard
        （start_pos > 0 机器求值）且每条边都挂它。"""
        guards = {g.stable_id: g for g in self.graph.guards}
        self.assertEqual(set(guards), {G_DECODE_ID})
        self.assertEqual(guards[G_DECODE_ID].predicate_expr, G_DECODE_EXPR)
        self.assertFalse(evaluate_bool(G_DECODE_EXPR, {"start_pos": 0}))
        self.assertTrue(evaluate_bool(G_DECODE_EXPR, {"start_pos": 1}))
        for e in self.graph.edges:
            self.assertEqual(e.guard_ref, G_DECODE_ID, e.stable_id)

    def test_no_markov_confidence_placeholder(self):
        """不把 Markov/Confidence 画成已实现占位链：无 markov/confidence/
        sample/output_ids 相关节点/张量。"""
        for n in self.graph.nodes:
            self.assertNotIn("markov", n.stable_id)
            self.assertNotIn("confid", n.stable_id)
            self.assertNotIn("sample", n.stable_id)
        for t_ in self.graph.tensors:
            self.assertNotIn("markov", t_.name)
            self.assertNotIn("confid", t_.name)
            self.assertNotIn("output", t_.name)

    def test_canonical_json_roundtrip(self):
        text = graph_to_json(self.graph)
        rebuilt = graph_from_json(text)
        self.assertEqual(graph_to_json(rebuilt), text)
        verify_exit_graph(rebuilt, self.bundle.values)  # 读回仍过语义核对

    def test_deterministic_rebuild(self):
        g1 = graph_to_json(build().graph)
        g2 = graph_to_json(build().graph)
        self.assertEqual(g1, g2)

    def test_exit_role_stage2_only(self):
        """实际 3 stage 仅 last（stage2）有出口角色（B2 推导复用）。"""
        roles = self.bundle.roles
        self.assertEqual(len(roles), 3)
        self.assertTrue(roles[-1].is_last)
        self.assertFalse(roles[0].is_last)
        self.assertFalse(roles[1].is_last)
        self.assertIn("forward_head", roles[-1].role_modules)
        for r in roles[:2]:
            self.assertNotIn("forward_head", r.role_modules)


class TestExitRoleMap(unittest.TestCase):
    """最小可检角色/参数映射：canonical 往返、guard 机器求值核对、
    篡改具名拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.values = cls.bundle.values
        cls.text = serialize_exit_role_map(build_exit_role_map(cls.values))

    def test_schema_and_guard_machine_check(self):
        self.assertIn(EXIT_SCHEMA, self.text)
        doc = json.loads(self.text)
        self.assertEqual(doc["guard"]["expr"], "start_pos > 0")
        self.assertFalse(doc["guard"]["prefill_active"])
        self.assertTrue(doc["guard"]["decode_active"])

    def test_canonical_roundtrip(self):
        doc = deserialize_exit_role_map(self.text, self.values)
        self.assertEqual(serialize_exit_role_map(doc), self.text)

class TestExitTamper(unittest.TestCase):
    """对真实序列化图文档的引用合法语义篡改：具名拒绝，非悬空引用兜底。
    每例先断言 validate_graph_references 为 0（篡改后引用合法），再断言
    verify_exit_graph 具名拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.doc = json.loads(graph_to_json(cls.bundle.graph))
        cls.values = cls.bundle.values

    def _tamper(self, mutate):
        doc = copy.deepcopy(self.doc)
        mutate(doc)
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        return graph_from_json(text)

    def test_tamper_delete_mandatory_edge_prenorm_norm(self):
        """删必要 head 边 e_ex_prenorm_norm（hc_head→norm 断链）。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_ex_prenorm_norm"]
            for n in doc["nodes"]:
                if n["stable_id"] == "exit.norm":
                    n["input_ports"] = [p for p in n["input_ports"]
                                        if p["name"] != "x"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        self.assertIn("e_ex_prenorm_norm", str(cm.exception))

    def test_tamper_delete_mandatory_edge_norm_head(self):
        """删必要 head 边 e_ex_norm_head（norm→head 断链）。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_ex_norm_head"]
            for n in doc["nodes"]:
                if n["stable_id"] == "exit.head":
                    n["input_ports"] = [p for p in n["input_ports"]
                                        if p["name"] != "x"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        self.assertIn("e_ex_norm_head", str(cm.exception))

    def test_tamper_consistent_rewire_norm_head(self):
        """保边 ID 一致重接：e_ex_norm_head 源改为 exit.hc_head.reduce.y
        （跳过 norm 的第二通路，端口/张量三方同步、引用合法）——norm 被
        绕过具名拒绝。"""
        def mutate(doc):
            for e in doc["edges"]:
                if e["stable_id"] == "e_ex_norm_head":
                    e["src_node_id"] = "exit.hc_head.reduce"
                    e["tensor_id"] = "tensor:exit:prenorm_x"
            for n in doc["nodes"]:
                if n["stable_id"] == "exit.head":
                    for p in n["input_ports"]:
                        if p["name"] == "x":
                            p["tensor_id"] = "tensor:exit:prenorm_x"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("e_ex_norm_head", msg)

    def test_tamper_shared_head_double_bind(self):
        """共享 head 身份复制：为 head_w 注入第二条消费边（同一身份双
        消费通路，引用合法）——唯一消费边核对具名拒绝。"""
        def mutate(doc):
            doc["edges"].append({
                "stable_id": "e_ex_headw_head_dup",
                "src_node_id": "exit.head_w", "src_port": "y",
                "dst_node_id": "exit.head", "dst_port": "w",
                "tensor_id": "tensor:exit:head_w", "kind": "data",
                "guard_ref": "g_decode", "evidence_refs": [],
            })
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("head", msg)

    def test_tamper_prefill_guard_removed(self):
        """prefill 误启：去掉一条边的 g_decode（head 链 prefill 可达）
        引用合法——逐边 guard 核对具名拒绝。"""
        def mutate(doc):
            for e in doc["edges"]:
                if e["stable_id"] == "e_ex_norm_head":
                    e["guard_ref"] = None
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("e_ex_norm_head", msg)
        self.assertIn("g_decode", msg)

    def test_tamper_guard_expr_weakened(self):
        """guard 表达式被弱化（start_pos >= 0：prefill 也为真，机器求值
        与 prefill 不活动矛盾）具名拒绝。"""
        def mutate(doc):
            for g_ in doc["guards"]:
                if g_["stable_id"] == "g_decode":
                    g_["predicate_expr"] = "start_pos >= 0"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("start_pos", msg)

    # ===== fix01 ①：必要节点算子/粒度/端口面闭世界负例 =====

    def test_fix01_tamper_gate_op_sigmoid_to_sinkhorn(self):
        """Codex 复现：exit.hc_head.gate op_type sigmoid_gate→sinkhorn
        （端口/证据/粒度不动，引用合法）——hc_head 是 sigmoid 门控加权
        汇聚非 Sinkhorn（model.py:714），op_type 闭世界具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "exit.hc_head.gate":
                    n["op_type"] = "sinkhorn"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("exit.hc_head.gate", msg)
        self.assertIn("sigmoid_gate", msg)

    def test_fix01_tamper_node_granularity(self):
        """必要节点 granularity 被改（tensor_op→parameter_op）具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "exit.hc_head.flatten":
                    n["granularity"] = "parameter_op"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        self.assertIn("exit.hc_head.flatten", str(cm.exception))

    def test_fix01_tamper_node_port_face(self):
        """必要节点端口面被改：为 exit.head 注入未连线的多余输出端口
        （张量存在、无悬空引用，引用检查不查端口面）——端口面闭世界
        具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "exit.head":
                    n["output_ports"] = n["output_ports"] + [
                        {"name": "extra", "direction": "output",
                         "tensor_id": "tensor:exit:logits"}]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        self.assertIn("exit.head", str(cm.exception))

    # ===== fix01 ②：消费轴可信值/表达式负例 =====

    def test_fix01_tamper_vocab_axis_value_synced(self):
        """Codex 复现：V 轴 value 129280→129281 在 named_axes 与 head_w/
        logits 两张量 axes 同步一致改（引用检查 V2 查两处互相同一致，仍
        为 0）——对照 rules+ModelValues 派生的可信值具名拒绝。"""
        def mutate(doc):
            for a in doc["named_axes"]:
                if a["symbol"] == "V":
                    a["value"] = 129281
            for t_ in doc["tensors"]:
                for axis in t_["axes"]:
                    if axis["symbol"] == "V":
                        axis["value"] = 129281
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("V", msg)
        self.assertIn("129280", msg)

    def test_fix01_tamper_named_axes_only_value(self):
        """只改 named_axes 一份 V 值（制造悬空不一致）具名拒绝——先于
        可信值核对被 V2 定义不一致拒绝亦可，语义层均不通过。"""
        def mutate(doc):
            for a in doc["named_axes"]:
                if a["symbol"] == "V":
                    a["value"] = 129281
        g = self._tamper(mutate)
        issues = validate_graph_references(g)
        if not issues:
            with self.assertRaises(ExtractionError) as cm:
                verify_exit_graph(g, self.values)
            self.assertIn("129280", str(cm.exception))
        else:
            self.assertIn("V2_AXIS_DEFINITION_MISMATCH",
                          sorted(i.code for i in issues))

    def test_fix01_tamper_expression_axis_forged_value(self):
        """表达式轴被伪造数值：B 从场景动态表达式（value=None）改成
        value=7——保持动态语义不伪造数值，具名拒绝。"""
        def mutate(doc):
            for a in doc["named_axes"]:
                if a["symbol"] == "B":
                    a["expression"] = None
                    a["value"] = 7
            for t_ in doc["tensors"]:
                for axis in t_["axes"]:
                    if axis["symbol"] == "B":
                        axis["expression"] = None
                        axis["value"] = 7
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_exit_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("B", msg)

    def test_fix01_axis_closes_world(self):
        """轴闭世界：全部张量 axes 只引用可信消费轴（V/H/N_hc/N_draft/
        N_hcH/B/C1），且真实图上 verify 通过——闭世界核对不误伤。"""
        trusted = {"V", "H", "N_hc", "N_draft", "N_hcH", "B", "C1"}
        used = {a.symbol for t_ in self.bundle.graph.tensors
                for a in t_.axes}
        self.assertTrue(used <= trusted)
        verify_exit_graph(self.bundle.graph, self.values)


class TestExitRoleMapTamper(unittest.TestCase):
    """角色映射级篡改（canonical JSON 文档，非图文档）：错误 last 角色 /
    prefill 误启 / 删 scope pending——读回全量核对具名拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.values = cls.bundle.values
        cls.text = serialize_exit_role_map(build_exit_role_map(cls.values))

    def test_tamper_exit_stage_param_misbind(self):
        """错误 last 角色绑定：exit_stage.param_path 绑到 mtp.0（stage0
        无出口角色）——B2 重推导逐字段核对具名拒绝。"""
        doc = json.loads(self.text)
        doc["exit_stage"]["param_path"] = "mtp.0"
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_exit_role_map(text, self.values)
        self.assertIn("exit_stage", str(cm.exception))

    def test_tamper_wrong_last_role(self):
        """错误 last 角色（stage1 出口）具名拒绝——先证引用合法前提。"""
        doc = json.loads(self.text)
        for s in doc["stages"]:
            s["exit_role"] = (s["stage_id"] == 1)
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_exit_role_map(text, self.values)
        self.assertIn("exit_role", str(cm.exception))

    def test_tamper_guard_prefill_enabled(self):
        """prefill 误启（head 链 prefill 活动）具名拒绝。"""
        doc = json.loads(self.text)
        doc["guard"]["prefill_active"] = True
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_exit_role_map(text, self.values)
        self.assertIn("prefill", str(cm.exception))

    def test_tamper_scope_pending_removed(self):
        """删 scope.pending（把未实现 Markov/Confidence 声称为无边界）
        具名拒绝。"""
        doc = json.loads(self.text)
        doc["scope"]["pending"] = []
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_exit_role_map(text, self.values)
        self.assertIn("scope.pending", str(cm.exception))

    # ===== fix01 ③：n_stages 类型与可信长度核对负例 =====

    def test_fix01_tamper_top_n_stages_99(self):
        """Codex 复现：顶层 n_stages 3→99、stages/exit_stage 保持原值
        （此前无核对放行）——对照 B2 派生长度 3 具名拒绝。"""
        doc = json.loads(self.text)
        self.assertEqual(doc["n_stages"], 3)
        doc["n_stages"] = 99
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_exit_role_map(text, self.values)
        msg = str(cm.exception)
        self.assertIn("n_stages", msg)
        self.assertIn("99", msg)

    def test_fix01_tamper_n_stages_type(self):
        """n_stages 类型突变（99.0/true/"3"）均具名拒绝——bool 是 int
        子类须显式排除。"""
        for bad in (99.0, True, "3"):
            doc = json.loads(self.text)
            doc["n_stages"] = bad
            text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"))
            with self.assertRaises(ExtractionError) as cm:
                deserialize_exit_role_map(text, self.values)
            self.assertIn("n_stages", str(cm.exception))

    def test_fix01_n_stages_derived_not_hardcoded(self):
        """单 stage first&last 合成（n_mtp_layers==1，B2 纯角色推导，
        不修改固定输入）：映射 n_stages=1、同一实例 first&last、
        canonical 往返——证明 n_stages 核对派生自 B2 而非硬编码 3。"""
        single_values = dataclasses.replace(self.values, n_mtp_layers=1)
        doc = build_exit_role_map(single_values)
        self.assertEqual(doc["n_stages"], 1)
        only = doc["stages"][0]
        self.assertEqual(len(doc["stages"]), 1)
        self.assertTrue(only["is_first"] and only["is_last"])
        self.assertTrue(only["exit_role"])
        text = serialize_exit_role_map(doc)
        back = deserialize_exit_role_map(text, single_values)
        self.assertEqual(serialize_exit_role_map(back), text)
        # 同一映射对真实 3 stage 输入不合法（n_stages=1 ≠ 派生 3）。
        with self.assertRaises(ExtractionError) as cm:
            deserialize_exit_role_map(text, self.values)
        self.assertIn("n_stages", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
