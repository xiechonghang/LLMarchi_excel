"""B4 整体集成测试：集成图装配/跨图连接/相位/场景/契约/幂等（B4）。

范围（task_b4_integration01.md 交付要求）：
- 集成图真实装配：六组件合并、276 节点/364 边、canonical 往返一致、
  重复构建确定；
- 跨图连接：14 条连接目录与真实边一一对应（连接 ID/端点/张量/kind/
  guard 逐条可联查；目录与图漂移即失败）；
- 闭世界：graph_input 恰 4 / graph_output 恰 3；
- 相位：decode 三元组可达、prefill 草稿链阻断、prefill passthrough 链
  可达、回绕分支（T>W）guard 活动逐场景求值；
- 身份：main_x 唯一生产/三 stage 共享、win_cache state_id 逐实例唯一、
  Markov logits/output_ids 各 6 版本、三实例模板同构；
- 场景状态：prefill T<=W 顺序/T>W 回绕/decode 单槽 P%W、选择索引宽
  min(W,P+1)+N_draft、草稿位置 P+T——逐场景对真实 env 求值核对；
- 契约与受控错误：原图零违规 + 全部 mutation 检出（引用合法）；
- 负例：轴未声明冲突拒绝、连接目录篡改可从图自身读出检出。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from structure_reader.checks.dspark_contracts import (
    apply_dspark_mutation,
    build_dspark_validation_report,
    dspark_integration_error_specs,
    validate_dspark_integration_contracts,
)
from structure_reader.extract.deepseek_v4_reference import ExtractionError
from structure_reader.extract.dspark_integration import (
    TEMPLATE_ID,
    build_dspark_integration_graph,
    dspark_scenario_views,
    integration_document,
)
from structure_reader.extract.evidence import EvidenceError
from structure_reader.graph_document import graph_from_json, graph_to_json
from structure_reader.model_registry import find_workspace_root
from structure_reader.rules.deepseek_v4_flash_0731 import (
    SCENARIOS,
    load_model_values,
    scenario_base_env,
)
from structure_reader.scenario_evaluator import evaluate_bool


class TestIntegrationBuild(unittest.TestCase):
    """集成图真实装配与身份。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.graph = cls.bundle.graph

    def test_template_and_size(self):
        self.assertEqual(self.graph.template_id, TEMPLATE_ID)
        self.assertEqual(len(self.graph.nodes), 276)
        self.assertEqual(len(self.graph.edges), 364)
        # 六组件合并：entry 20 + stage×3 66 + exit 14 + markov 49 + conf 8
        # − 移除边界 13 = 276
        self.assertEqual(len(self.bundle.registry.all()), 122)

    def test_zero_reference_issues_and_canonical(self):
        from structure_reader.checks.graph_contracts import (
            validate_graph_references)
        self.assertEqual(validate_graph_references(self.graph), ())
        text = graph_to_json(self.graph)
        back = graph_from_json(text)
        self.assertEqual(graph_to_json(back), text)

    def test_rebuild_deterministic(self):
        b2 = build_dspark_integration_graph()
        self.assertEqual(graph_to_json(b2.graph), graph_to_json(self.graph))
        self.assertEqual(integration_document(b2), integration_document(self.bundle))

    def test_closed_world_boundaries(self):
        gi = sorted(n.stable_id for n in self.graph.nodes
                    if n.op_type == "graph_input")
        self.assertEqual(gi, ["entry.input_h_40", "entry.input_h_41",
                              "entry.input_h_42", "entry.seed_src"])
        go = sorted(n.stable_id for n in self.graph.nodes
                    if n.op_type == "graph_output")
        self.assertEqual(go, ["cf.confidence_out", "mk.logits_out",
                              "mk.output_ids_out"])
        # 三元组张量形状（B,K+1 / B,K,V / B,K）
        shapes = {}
        for t in self.graph.tensors:
            if t.stable_id in ("tensor:mk:output_ids", "tensor:mk:logits",
                               "tensor:cf:confidence"):
                shapes[t.stable_id] = [a.symbol for a in t.axes]
        self.assertEqual(shapes["tensor:mk:output_ids"], ["B", "N_draft_plus_1"])
        self.assertEqual(shapes["tensor:mk:logits"],
                         ["B", "N_draft", "V"])
        self.assertEqual(shapes["tensor:cf:confidence"], ["B", "N_draft"])

    def test_main_x_single_identity_shared(self):
        tids = []
        for i in range(3):
            n = next(x for x in self.graph.nodes
                     if x.stable_id == f"mtp{i}.dspark.x_main_in_src")
            tids.append(n.input_ports[0].tensor_id)
        self.assertEqual(len(set(tids)), 1)
        self.assertEqual(tids[0], "tensor:entry:main_x")
        producers = [e for e in self.graph.edges
                     if e.tensor_id == "tensor:entry:main_x"
                     and e.src_node_id == "entry.main_norm"]
        self.assertEqual(len(producers), 3)

    def test_stage_instances_unique_state(self):
        states = sorted({t.state_id for t in self.graph.state_transitions
                         if t.state_id.startswith("dspark:")})
        self.assertEqual(states, ["dspark:win_cache:mtp.0",
                                  "dspark:win_cache:mtp.1",
                                  "dspark:win_cache:mtp.2"])
        # 三实例模板同构（op_type/granularity/端口签名多重集一致）
        sigs = []
        for i in range(3):
            sig = sorted(
                (n.op_type, n.granularity,
                 tuple((p.direction, p.name) for p in n.input_ports),
                 tuple((p.direction, p.name) for p in n.output_ports))
                for n in self.graph.nodes
                if n.stable_id.startswith(f"mtp{i}.")
                and not n.stable_id.startswith(f"mtp{i}.dspark.x_"))
            sigs.append(sig)
        self.assertEqual(sigs[0], sigs[1])
        self.assertEqual(sigs[1], sigs[2])

    def test_embed_head_shared_single_entity(self):
        embed_nodes = [n.stable_id for n in self.graph.nodes
                       if n.op_type == "embedding_weight"]
        self.assertEqual(embed_nodes, ["entry.embed_w"])
        head_w = [n.stable_id for n in self.graph.nodes
                  if n.op_type == "linear_weight"
                  and any(p.tensor_id == "tensor:exit:head_w"
                          for p in n.output_ports)]
        self.assertEqual(head_w, ["exit.head_w"])

    def test_markov_version_chain_complete(self):
        lv = sorted(v.stable_id for v in self.graph.state_versions
                    if v.state_id == "mk:logits")
        self.assertEqual(lv, [f"state:mk:logits:v{i}" for i in range(6)])
        ov = sorted(v.stable_id for v in self.graph.state_versions
                    if v.state_id == "mk:output_ids")
        self.assertEqual(ov, [f"state:mk:output_ids:v{i}" for i in range(6)])


class TestIntegrationConnections(unittest.TestCase):
    """14 条跨图连接：目录与真实边一一对应。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.graph = cls.bundle.graph
        cls.edges = {e.stable_id: e for e in cls.graph.edges}
        cls.nodes = {n.stable_id: n for n in cls.graph.nodes}
        cls.doc = integration_document(cls.bundle)

    def test_connection_catalog_matches_graph(self):
        """目录 14 条逐条：边存在、端点/张量/kind/guard 与图一致。"""
        self.assertEqual(len(self.bundle.connections), 14)
        for c in self.bundle.connections:
            d = c.to_dict()
            e = self.edges.get(d["edge_id"])
            self.assertIsNotNone(e, d["connection_id"])
            self.assertEqual(e.src_node_id, d["src"]["node"],
                             d["connection_id"])
            self.assertEqual(e.src_port, d["src"]["port"])
            self.assertEqual(e.dst_node_id, d["dst"]["node"])
            self.assertEqual(e.dst_port, d["dst"]["port"])
            self.assertEqual(e.tensor_id, d["src"]["tensor"])
            self.assertEqual(d["src"]["tensor"], d["dst"]["tensor"])
            self.assertEqual(e.kind, d["kind"])
            self.assertEqual(e.guard_ref, d["guard_ref"])
            # 端点真实存在且端口张量一致（可由图自身证伪）
            for side in ("src", "dst"):
                node = self.nodes[d[side]["node"]]
                port_name = d[side]["port"]
                ports = {p.name: p for p in node.output_ports} \
                    if side == "src" else {p.name: p for p in node.input_ports}
                self.assertIn(port_name, ports, d["connection_id"])
                self.assertEqual(ports[port_name].tensor_id,
                                 d[side]["tensor"])

    def test_doc_connections_equal_bundle(self):
        self.assertEqual(self.doc["connections"],
                         [c.to_dict() for c in self.bundle.connections])

    def test_dropped_boundaries_absent(self):
        for nid in ("mtp0.stage.main_x_src", "mtp1.stage.main_x_src",
                    "mtp2.stage.main_x_src", "mtp0.stage.x_hc_in_src",
                    "mtp1.stage.x_hc_in_src", "mtp2.stage.x_hc_in_src",
                    "exit.h_out_src", "exit.logits_out", "mk.seed_src",
                    "mk.init_logits_src", "mk.embed_out", "cf.hidden_src",
                    "cf.embed_src"):
            self.assertNotIn(nid, self.nodes, nid)
        self.assertNotIn("e_ex_logits_out", self.edges)
        self.assertNotIn("e_mk_stack_out", self.edges)

    def test_prefill_last_h_out_unconsumed(self):
        """prefill 末 stage h_out 无下游消费（933-934 提前 return），
        不画假边。"""
        out_edges = [e for e in self.graph.edges
                     if e.src_node_id == "mtp2.stage.prefill_passthrough"]
        self.assertEqual(out_edges, [])


class TestIntegrationPhases(unittest.TestCase):
    """相位可达性（真实 guard 逐边求值）。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.graph = cls.bundle.graph
        cls.values = load_model_values(find_workspace_root())

    def _reach(self, scenario_id):
        env = scenario_base_env(self.values,
                                next(s for s in SCENARIOS
                                     if s.scenario_id == scenario_id))
        active = {}
        for e in self.graph.edges:
            if e.guard_ref is None:
                active[e.stable_id] = True
            else:
                g = next(x for x in self.graph.guards
                         if x.stable_id == e.guard_ref)
                active[e.stable_id] = evaluate_bool(g.predicate_expr, env)
        sources = [n.stable_id for n in self.graph.nodes
                   if n.op_type == "graph_input"]
        seen = set(sources)
        frontier = list(sources)
        out_by_node = {}
        for e in self.graph.edges:
            if active[e.stable_id]:
                out_by_node.setdefault(e.src_node_id, []).append(
                    e.dst_node_id)
        while frontier:
            n = frontier.pop()
            for m in out_by_node.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    frontier.append(m)
        return seen

    def test_decode_reaches_triple(self):
        r = self._reach("decode_p128")
        for target in ("mk.output_ids_out", "mk.logits_out",
                       "cf.confidence_out", "exit.hc_head.flatten",
                       "mtp2.stage.hc_post_ffn.combine",
                       "mtp1.stage.hc_pre_attn.flatten"):
            self.assertIn(target, r, target)

    def test_prefill_blocks_draft_chain(self):
        r = self._reach("prefill_t128")
        for target in ("mk.output_ids_out", "mk.logits_out",
                       "cf.confidence_out", "exit.hc_head.flatten",
                       "mtp0.stage.hc_pre_attn.flatten"):
            self.assertNotIn(target, r, target)
        self.assertIn("mtp2.stage.prefill_passthrough", r)

    def test_wrap_prefill_reaches_passthrough(self):
        r = self._reach("prefill_t129")
        self.assertIn("mtp2.stage.prefill_passthrough", r)


class TestScenarioViews(unittest.TestCase):
    """DSpark 场景状态视图（真实 env 求值）。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.values = load_model_values(find_workspace_root())
        cls.views = dspark_scenario_views(cls.bundle, [
            dict(scenario_base_env(cls.values, s), scenario_id=s.scenario_id)
            for s in SCENARIOS
        ])
        cls.by_id = {v["scenario_id"]: v for v in cls.views}

    def test_all_scenarios(self):
        self.assertEqual({v["scenario_id"] for v in self.views},
                         {s.scenario_id for s in SCENARIOS})

    def test_prefill_sequential_and_wrap(self):
        p127 = self.by_id["prefill_t127"]["window_cache"]
        self.assertEqual(p127["branch"], "sequential")
        self.assertEqual(p127["written_slots_this_call"], list(range(127)))
        p128 = self.by_id["prefill_t128"]["window_cache"]
        self.assertEqual(p128["branch"], "sequential")
        self.assertEqual(p128["written_slots_this_call"], list(range(128)))
        p129 = self.by_id["prefill_t129"]["window_cache"]
        self.assertEqual(p129["branch"], "wraparound")
        self.assertEqual(p129["written_slots_this_call"], list(range(128)))
        self.assertIn("model.py:767-768", p129["wrap_note"])
        # prefill 无草稿（769 早退）
        self.assertFalse(self.by_id["prefill_t129"]["draft_kv"]["active"])
        self.assertEqual(
            self.by_id["prefill_t129"]["guard_activations"]["g_window_wrap"]
            ["state"], "true")
        self.assertEqual(
            self.by_id["prefill_t128"]["guard_activations"]["g_window_wrap"]
            ["state"], "false")

    def test_decode_slots_and_positions(self):
        d128 = self.by_id["decode_p128"]
        self.assertEqual(d128["window_cache"]["written_slots_this_call"], [0])
        self.assertEqual(d128["positions"]["slot_write"], "P % W = 128 % 128 = 0")
        self.assertEqual(d128["draft_kv"]["select_width_value"], 133)
        self.assertEqual(d128["draft_kv"]["k_steps"], 5)
        self.assertTrue(d128["draft_kv"]["active"])
        self.assertEqual(d128["positions"]["draft_rope"],
                         "P+T..P+T+N_draft-1（129..133）")
        self.assertEqual(
            d128["guard_activations"]["g_decode"]["state"], "true")
        self.assertEqual(
            d128["guard_activations"]["g_prefill"]["state"], "false")
        d2 = self.by_id["decode_p2"]
        self.assertEqual(d2["draft_kv"]["select_width_value"], 8)
        self.assertEqual(d2["window_cache"]["written_slots_this_call"], [2])
        self.assertIn("unknown", d2["window_cache"]["valid_length"])

    def test_capacity_vs_valid_length_distinct(self):
        """capacity 恒 W 槽；有效项数数值与前提分开（前提未满足=unknown，
        不当作空/满）；物理 concat 宽与选择有效宽分开。"""
        for sid in ("decode_p2", "decode_p128", "prefill_t127"):
            wc = self.by_id[sid]["window_cache"]
            self.assertEqual(wc["capacity_slots"], 128)
            self.assertEqual(wc["compress_zone_slots"], 0)
        d2 = self.by_id["decode_p2"]["window_cache"]
        # P=2 < W：数量阈值未满足 → 数值=after 确定可数槽（3=P+1），
        # state=partial，不冒充满窗（连续历史为显式假设）
        self.assertEqual(d2["valid_slot_count"], 3)
        self.assertEqual(d2["valid_length_state"], "partial")
        self.assertIn("未满足", d2["valid_length_premise"])
        self.assertIn("显式假设", d2["valid_length_premise"])
        self.assertIn("unknown", d2["valid_length"])
        d128 = self.by_id["decode_p128"]["window_cache"]
        # P=128：P+1=129 ≥ W → 满窗；文字按 P+1 口径（不是 P≥W）且
        # 声明是显式假设不由数量证明
        self.assertEqual(d128["valid_slot_count"], 128)
        self.assertEqual(d128["valid_length_state"], "full_premise")
        self.assertIn("P+1", d128["valid_length_premise"])
        self.assertIn("显式假设", d128["valid_length_premise"])
        d127 = self.by_id["decode_p127"]["window_cache"]
        # P=127 边界：P+1=128=W 恰满窗（覆盖判定按 P+1，不是 P≥W）
        self.assertEqual(d127["valid_slot_count"], 128)
        self.assertEqual(d127["valid_length_state"], "full_premise")
        # 物理宽 W+N_draft=133 与选择有效宽 min(W,P+1)+N_draft=133 分开
        # 字段（P=128 时数值相同但口径不同字段）
        self.assertEqual(d128["physical_concat_width"], 133)
        self.assertEqual(d128["select_valid_width"], 133)
        self.assertNotEqual(d128["physical_concat_note"],
                            d128["select_valid_note"])
        # prefill：无历史前提 → 数值=本次写入槽数、state=unknown
        t3 = self.by_id["prefill_t3"]["window_cache"]
        self.assertEqual(t3["valid_slot_count"], 3)
        self.assertEqual(t3["valid_length_state"], "unknown")

    def test_stage_state_version_bindings(self):
        """每实例 cache state_id/版本 ID/写分支与发布后读取关系（B4 补修
        01）；版本 ID 必须与集成图真实 state_versions/transitions 一致。"""
        g = build_dspark_integration_graph().graph
        version_ids = {v.stable_id for v in g.state_versions}
        tr_ids = {t.stable_id for t in g.state_transitions}
        for sid in ("decode_p128", "decode_p2", "prefill_t3"):
            view = self.by_id[sid]
            self.assertEqual(len(view["stage_states"]), 3)
            for i, st in enumerate(view["stage_states"]):
                self.assertEqual(st["stage_id"], i)
                self.assertEqual(st["param_path"], f"mtp.{i}")
                self.assertEqual(st["state_id"], f"dspark:win_cache:mtp.{i}")
                self.assertIn(st["before_version_id"], version_ids)
                self.assertIn(st["after_version_id"], version_ids)
                self.assertEqual(st["published_version_id"],
                                 st["after_version_id"])
                self.assertIn(st["transition_id"], tr_ids)
                self.assertIn("cache_tensor", st)
                self.assertTrue(st["write_branch"])
                self.assertTrue(st["read_relation"])
        # before 三态 unknown（数量阈值不转 bool——review02 口径）；
        # after 有效槽数分字段（覆盖数量口径 min(P+1, W)）
        d2_s0 = self.by_id["decode_p2"]["stage_states"][0]
        self.assertEqual(d2_s0["before_state_known"], "unknown")
        self.assertIn("unknown", d2_s0["before_state_note"])
        self.assertEqual(d2_s0["after_slot_count"], 3)
        d128_s0 = self.by_id["decode_p128"]["stage_states"][0]
        self.assertEqual(d128_s0["before_state_known"], "unknown")
        self.assertEqual(d128_s0["after_slot_count"], 128)
        self.assertIn("显式假设", d128_s0["before_state_note"])


class TestIntegrationContracts(unittest.TestCase):
    """集成契约与受控错误。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.report = build_dspark_validation_report(cls.bundle.graph)

    def test_original_zero_issues(self):
        self.assertEqual(
            validate_dspark_integration_contracts(self.bundle.graph), ())

    def test_all_mutations_detected(self):
        ce = self.report["controlled_errors"]
        self.assertEqual(ce["catalog_count"],
                         len(dspark_integration_error_specs()))
        self.assertTrue(ce["all_detected"])
        for r in ce["results"]:
            self.assertTrue(r["detected"], r["mutation_id"])

    def test_mutations_reference_valid(self):
        """每个 mutation 注入后引用合法（受控错误语义错而非引用错）。"""
        from structure_reader.checks.graph_contracts import (
            validate_graph_references)
        for spec in dspark_integration_error_specs():
            mutated = apply_dspark_mutation(self.bundle.graph,
                                            spec.mutation_id)
            self.assertEqual(validate_graph_references(mutated), (),
                             spec.mutation_id)

    def test_unknown_mutation_rejected(self):
        with self.assertRaises(ExtractionError):
            apply_dspark_mutation(self.bundle.graph, "no_such_mutation")


class TestIntegrationDoc(unittest.TestCase):
    """集成文档：实例/边界/轴限定/HF named diff/组件映射。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.doc = integration_document(cls.bundle)

    def test_instances_roles(self):
        inst = {i["param_path"]: i for i in self.doc["instances"]}
        self.assertEqual(len(inst), 3)
        self.assertTrue(inst["mtp.0"]["is_first"])
        self.assertFalse(inst["mtp.0"]["is_last"])
        self.assertTrue(inst["mtp.2"]["is_last"])
        self.assertEqual([i["actual_layer_id"] for i in self.doc["instances"]],
                         [43, 44, 45])
        self.assertIn("main_proj", inst["mtp.0"]["role_modules"])
        self.assertIn("forward_head", inst["mtp.2"]["role_modules"])

    def test_axis_qualifications_recorded(self):
        syms = {a.symbol for a in self.bundle.graph.named_axes}
        # C1（stage 模板 hc_pre keepdim 单例）保留原名；exit/conf 的
        # C1 按图限定改名（同 symbol 异定义合并的唯一声明冲突）
        self.assertIn("C1", syms)
        self.assertIn("C1_exit", syms)
        self.assertIn("C1_conf", syms)
        self.assertEqual(len(syms), 37)

    def test_hf_difference_named(self):
        hf = self.doc["hf_difference"]
        self.assertIn("num_nextn_predict_layers", hf["named"][1])
        self.assertIn("无 DSpark", hf["named"][2])
        self.assertIn("不暗示", hf["conclusion"])

    def test_unknowns_kept(self):
        self.assertTrue(any("kernel" in u for u in self.doc["unknowns"]))
        self.assertTrue(any("数值未知" in u or "unknown" in u
                            for u in self.doc["unknowns"]))

    def test_ratios_consumers(self):
        rc = self.doc["ratios_consumers"]
        self.assertEqual(rc["total_len"], 46)
        self.assertEqual(rc["draft_tail_values"], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
