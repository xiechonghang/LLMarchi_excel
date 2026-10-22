"""DSpark Markov 递推图最小测试（B3b 限定恢复01，resume_b3b_markov01.md）。

正例：真实图构建（49 节点/64 边/37 张量/12 版本/10 transition）、decode-only
guard 机器可检、canonical 往返、重复构建一致、logits/output_ids 版本链
（slice 消费 v{i} / sample 消费本步 updated / logits_out 消费 v5）、w1/w2
共享参数单一身份（恰一节点/张量、恰 5 条消费边）、sample 恰一 logits 输入
（temperature 标量绑定不进端口）、step 映射 canonical 往返 + 单 stage
first&last 合成往返。

负例（引用合法语义篡改，先证 validate_graph_references 为 0 再断言
verify_markov_graph 具名拒绝）：删必要边/一致重接（slice 链、write_ids
跨步链）、w1/w2 复制身份、seed rank2（extra 列写第二通路）、算子突变
（gumbel_max_sample→argmax_sample、bias_add→覆盖写）、端口面突变、同步
改轴值、step 映射顶层/类型/pending 篡改。Confidence 未实现不画占位。

定点补修01（codex_b3b_markov_fix01.md）：deserialize 全量逐叶子闭世界
核对补漏——Codex 四份原始复现 JSON（seed rank2 shape [B,1]/final_logits
dtype bf16/markov_embed note 反转/gumbel_max 分支文本偷换）从磁盘只读
回放拒绝；boundaries/sample/temperature/phases/steps/params 叶子字段
篡改与 int/bool 精确类型负例；正常映射往返仍逐字节一致。
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
from structure_reader.extract.dspark_markov import (
    G_DECODE_EXPR,
    G_DECODE_ID,
    MARKOV_SCHEMA,
    MARKOV_STEP_MAP_EXPECTED_TOP,
    MARKOV_TEMPLATE_ID,
    N_STEPS,
    build_dspark_markov_graph,
    build_markov_step_map,
    deserialize_markov_step_map,
    serialize_markov_step_map,
    verify_markov_graph,
)
from structure_reader.graph_document import graph_from_json, graph_to_json
from structure_reader.model_registry import find_workspace_root
from structure_reader.scenario_evaluator import evaluate_bool

ROOT = find_workspace_root()


def build():
    return build_dspark_markov_graph(ROOT)


class TestMarkovGraph(unittest.TestCase):
    """Markov 递推图结构：真实节点/边、guard 机器可检、canonical 往返、
    重复构建一致。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = {n.stable_id: n for n in cls.graph.nodes}
        cls.edges = {e.stable_id: e for e in cls.graph.edges}
        cls.values = cls.bundle.values

    def test_template_id(self):
        self.assertEqual(self.graph.template_id, MARKOV_TEMPLATE_ID)

    def test_zero_reference_issues(self):
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_world_counts(self):
        """闭世界计数：49 节点/64 边/37 张量/12 版本/10 transition。"""
        g = self.graph
        self.assertEqual(len(g.nodes), 49)
        self.assertEqual(len(g.edges), 64)
        self.assertEqual(len(g.tensors), 37)
        self.assertEqual(len(g.state_versions), 12)
        self.assertEqual(len(g.state_transitions), 10)

    def test_recurrence_chain_real(self):
        """五步递推链真实存在：每步 read_col→embed(w1_w)→bias(w2_w)→
        slice→add→logits_write→sample→write_col，跨步 write_ids 与
        slice_read 版本链边齐全。"""
        n = self.nodes
        e = self.edges
        for i in range(N_STEPS):
            for sid in (f"mk.step{i}.read_col", f"mk.step{i}.embed",
                        f"mk.step{i}.bias", f"mk.step{i}.slice",
                        f"mk.step{i}.add", f"mk.step{i}.logits_write",
                        f"mk.step{i}.sample", f"mk.step{i}.write_col"):
                self.assertIn(sid, n, sid)
        for eid in ("e_mk_init_col0", "e_mk_seed_init", "e_mk_stack_out",
                    "e_mk_logits_out"):
            self.assertIn(eid, e)
        # 跨步链：step{i}.write_col → step{i+1}.read_col；step4 → 输出
        for i in range(N_STEPS - 1):
            self.assertIn(f"e_mk_s{i}_write_ids", e)
            self.assertEqual(e[f"e_mk_s{i}_write_ids"].src_node_id,
                             f"mk.step{i}.write_col")
            self.assertEqual(e[f"e_mk_s{i}_write_ids"].dst_node_id,
                             f"mk.step{i + 1}.read_col")
        self.assertIn("e_mk_s4_write_ids", e)

    def test_shared_param_single_identity(self):
        """w1/w2 共享参数单一身份：恰一节点/张量，恰 5 条消费边到各步
        embed.w/bias.w，w1 bf16 / w2 fp32。"""
        tensors = {t.name: t for t in self.graph.tensors}
        self.assertEqual({n for n in self.nodes if "w1_param" in n},
                         {"mk.w1_param"})
        self.assertEqual({n for n in self.nodes if "w2_param" in n},
                         {"mk.w2_param"})
        for name, want_dtype, dsts in (
                ("mk:w1_w", "bf16",
                 {f"mk.step{i}.embed" for i in range(N_STEPS)}),
                ("mk:w2_w", "fp32",
                 {f"mk.step{i}.bias" for i in range(N_STEPS)})):
            t_ = tensors[name]
            self.assertEqual(t_.dtype, want_dtype)
            self.assertEqual(tuple(t_.logical_layout), ("V", "Rm"))
            consumers = [e for e in self.graph.edges
                         if e.tensor_id == t_.stable_id]
            self.assertEqual(len(consumers), N_STEPS)
            self.assertEqual({e.dst_node_id for e in consumers}, dsts)
            self.assertTrue(all(e.dst_port == "w" for e in consumers))

    def test_decode_only_guard_on_every_edge(self):
        """整条 Markov 链 decode 活动、prefill 不活动：唯一 g_decode guard
        （start_pos > 0 机器求值）且每条边都挂它。"""
        guards = {g.stable_id: g for g in self.graph.guards}
        self.assertEqual(set(guards), {G_DECODE_ID})
        self.assertEqual(guards[G_DECODE_ID].predicate_expr, G_DECODE_EXPR)
        self.assertFalse(evaluate_bool(G_DECODE_EXPR, {"start_pos": 0}))
        self.assertTrue(evaluate_bool(G_DECODE_EXPR, {"start_pos": 1}))
        for e in self.graph.edges:
            self.assertEqual(e.guard_ref, G_DECODE_ID, e.stable_id)

    def test_logits_versions_and_boundary(self):
        """logits 版本链：slice 消费 v{i}（v0=外部 init_logits 边界，
        i≥1=存储张量）、sample 消费本步 updated、logits_out 消费 v5 终版
        （不接 v0）。"""
        e = self.edges
        self.assertEqual(e["e_mk_s0_slice_read"].tensor_id,
                         "tensor:mk:init_logits")
        for i in range(1, N_STEPS):
            self.assertEqual(e[f"e_mk_s{i}_slice_read"].tensor_id,
                             "tensor:mk:logits")
        for i in range(N_STEPS):
            self.assertEqual(e[f"e_mk_s{i}_add_sample"].tensor_id,
                             f"tensor:mk:step{i}_logits_updated")
        self.assertEqual(e["e_mk_logits_out"].src_node_id,
                         "mk.step4.logits_write")
        self.assertEqual(e["e_mk_logits_out"].tensor_id, "tensor:mk:logits")

    def test_output_ids_versions_and_seed(self):
        """output_ids 版本链：第 0 步读 v0（init 后 seed 列）、第 i 步读
        v{i}（第 i-1 步写后）、最终输出消费 v5；seed [B] rank-1 进 col0
        初始化。"""
        e = self.edges
        self.assertEqual(e["e_mk_init_col0"].src_node_id,
                         "mk.output_ids_init")
        for i in range(1, N_STEPS):
            self.assertEqual(e[f"e_mk_s{i - 1}_write_ids"].src_node_id,
                             f"mk.step{i - 1}.write_col")
            self.assertEqual(e[f"e_mk_s{i - 1}_write_ids"].dst_node_id,
                             f"mk.step{i}.read_col")
        self.assertEqual(e["e_mk_s4_write_ids"].dst_node_id,
                         "mk.output_ids_out")
        tensors = {t.name: t for t in self.graph.tensors}
        self.assertEqual(tuple(tensors["mk:seed_ids"].logical_layout), ("B",))
        self.assertEqual(
            tuple(tensors["mk:output_ids"].logical_layout),
            ("B", "N_draft_plus_1"))

    def test_sample_single_logits_input(self):
        """sample 端口面恰一 logits 输入（871 sample(logits[:, i],
        temperature)——temperature 是标量绑定非张量输入；无源码签名外的
        token_ids 数据依赖）。"""
        for i in range(N_STEPS):
            n = self.nodes[f"mk.step{i}.sample"]
            self.assertEqual(n.op_type, "gumbel_max_sample")
            self.assertEqual([(p.name, p.tensor_id) for p in n.input_ports],
                             [("logits",
                               f"tensor:mk:step{i}_logits_updated")])
            self.assertEqual([p.name for p in n.output_ports], ["token"])

    def test_no_confidence_placeholder(self):
        """Confidence 未实现：图内无 confidence 节点/张量（873 不画占位）。"""
        for n in self.graph.nodes:
            self.assertNotIn("confid", n.stable_id)
        for t_ in self.graph.tensors:
            self.assertNotIn("confid", t_.name)

    def test_canonical_json_roundtrip(self):
        text = graph_to_json(self.graph)
        rebuilt = graph_from_json(text)
        self.assertEqual(graph_to_json(rebuilt), text)
        verify_markov_graph(rebuilt, self.values)  # 读回仍过语义核对

    def test_deterministic_rebuild(self):
        g1 = graph_to_json(build().graph)
        g2 = graph_to_json(build().graph)
        self.assertEqual(g1, g2)


class TestMarkovTamper(unittest.TestCase):
    """对真实序列化图文档的引用合法语义篡改：具名拒绝，非悬空引用兜底。
    每例先断言 validate_graph_references 为 0（篡改后引用合法），再断言
    verify_markov_graph 具名拒绝。"""

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

    def test_tamper_delete_mandatory_edge_slice_read(self):
        """删第 2 步 slice_read（logits 版本链断链）。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_mk_s1_slice_read"]
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.step1.slice":
                    n["input_ports"] = [p for p in n["input_ports"]
                                        if p["name"] != "logits"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        self.assertIn("e_mk_s1_slice_read", str(cm.exception))

    def test_tamper_delete_mandatory_edge_write_ids(self):
        """删跨步 ids 链边 e_mk_s2_write_ids（第 3 步读不到第 2 步写回）。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_mk_s2_write_ids"]
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.step3.read_col":
                    n["input_ports"] = [p for p in n["input_ports"]
                                        if p["name"] != "ids"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        self.assertIn("e_mk_s2_write_ids", str(cm.exception))

    def test_tamper_consistent_rewire_sample_to_preadd(self):
        """一致重接：step2.sample.logits 从 add.y 重接到 slice.slice（读
        更新前 logits——871 语义改变，端口/张量三方同步、引用合法）。"""
        def mutate(doc):
            for e in doc["edges"]:
                if e["stable_id"] == "e_mk_s2_add_sample":
                    e["src_node_id"] = "mk.step2.slice"
                    e["src_port"] = "slice"
                    e["tensor_id"] = "tensor:mk:step2_logits_slice"
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.step2.sample":
                    n["input_ports"] = [
                        {"name": "logits", "direction": "input",
                         "tensor_id": "tensor:mk:step2_logits_slice"}]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("mk.step2.sample", msg)

    def test_tamper_old_version_logits_out(self):
        """logits_out 一致重接到 step0.add（旧版本输出，936 应为 v5 终版）
        ——端口/张量三方同步、引用合法，版本表具名拒绝。"""
        def mutate(doc):
            for e in doc["edges"]:
                if e["stable_id"] == "e_mk_logits_out":
                    e["src_node_id"] = "mk.step0.add"
                    e["tensor_id"] = "tensor:mk:step0_logits_updated"
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.logits_out":
                    n["input_ports"] = [
                        {"name": "x", "direction": "input",
                         "tensor_id": "tensor:mk:step0_logits_updated"}]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("mk.logits_out", msg)
        self.assertIn("mk.step4.logits_write", msg)

    def test_tamper_w1_copy_identity(self):
        """w1 复制身份：新增 step1 专有 w1_w 张量+参数节点+消费边（第二
        Markov 参数身份，引用合法）——节点/张量/边闭世界具名拒绝。"""
        def mutate(doc):
            doc["tensors"].append({
                "name": "mk:step1_w1_w", "dtype": "bf16",
                "stable_id": "tensor:mk:step1_w1_w",
                "global_rank": None, "local_rank": None,
                "axes": [
                    {"symbol": "V", "name": "vocab_size",
                     "meaning": "词表大小", "origin": "run config vocab_size",
                     "evidence_ref": "run_config.json:2",
                     "stable_id": "axis:V", "expression": None,
                     "unknown_reason": None, "value": 129280},
                    {"symbol": "Rm", "name": "dspark_markov_rank",
                     "meaning": "Markov 低秩秩",
                     "origin": "run config dspark_markov_rank",
                     "evidence_ref": "run_config.json:11",
                     "stable_id": "axis:Rm", "expression": None,
                     "unknown_reason": None, "value": 256}],
                "logical_layout": ["V", "Rm"],
                "physical_layout": ["V", "Rm"],
            })
            doc["nodes"].append({
                "stable_id": "mk.step1.w1_w", "op_type": "markov_w1_param",
                "granularity": "parameter_op", "input_ports": [],
                "output_ports": [
                    {"name": "y", "direction": "output",
                     "tensor_id": "tensor:mk:step1_w1_w"}],
                "evidence_refs": [],
            })
            # 把 step1.embed 的 w 端口重接到复制身份（三方一致）
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.step1.embed":
                    for p in n["input_ports"]:
                        if p["name"] == "w":
                            p["tensor_id"] = "tensor:mk:step1_w1_w"
            for e in doc["edges"]:
                if e["stable_id"] == "e_mk_s1_w1_embed":
                    e["src_node_id"] = "mk.step1.w1_w"
                    e["tensor_id"] = "tensor:mk:step1_w1_w"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("mk.step1.w1_w", msg)

    def test_tamper_w2_missing_step_consumer(self):
        """w2 消费边缺步：删 e_mk_s3_w2_bias（step3 bias 无 w 来源，共享
        身份只覆盖 4 步）——恰 5 条消费边核对具名拒绝。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_mk_s3_w2_bias"]
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.step3.bias":
                    n["input_ports"] = [p for p in n["input_ports"]
                                        if p["name"] != "w"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("mk:w2_w", msg)

    def test_tamper_seed_rank2_extra_col_write(self):
        """seed rank2 第二通路：注入 output_ids 之外的 output_ids2 [B,2]
        列写张量+write 节点+边（K+1 六列之外的第七列写入，引用合法）——
        闭世界具名拒绝。"""
        def mutate(doc):
            doc["tensors"].append({
                "name": "mk:output_ids2", "dtype": "int64",
                "stable_id": "tensor:mk:output_ids2",
                "global_rank": None, "local_rank": None,
                "axes": [
                    {"symbol": "B",
                     "name": "batch_size", "meaning": "批大小",
                     "origin": "场景输入", "evidence_ref": "scenarios",
                     "stable_id": "axis:B", "expression": "B",
                     "unknown_reason": None, "value": None},
                    {"symbol": "N_draft_plus_1",
                     "name": "dspark_block_size_plus_1",
                     "meaning": "output_ids 列宽 K+1（草稿步数+种子列）",
                     "origin": "推导 model.py:864",
                     "evidence_ref": "ev_mk_loop",
                     "stable_id": "axis:N_draft_plus_1", "expression": None,
                     "unknown_reason": None, "value": 6}],
                "logical_layout": ["B", "N_draft_plus_1"],
                "physical_layout": ["B", "N_draft_plus_1"],
            })
            doc["nodes"].append({
                "stable_id": "mk.step0.write_col2", "op_type": "col_write",
                "granularity": "index_op",
                "input_ports": [
                    {"name": "seed", "direction": "input",
                     "tensor_id": "tensor:mk:output_ids"}],
                "output_ports": [
                    {"name": "y", "direction": "output",
                     "tensor_id": "tensor:mk:output_ids2"}],
                "evidence_refs": [],
            })
            doc["edges"].append({
                "stable_id": "e_mk_s0_write_ids2",
                "src_node_id": "mk.step0.write_col", "src_port": "y",
                "dst_node_id": "mk.step0.write_col2", "dst_port": "seed",
                "tensor_id": "tensor:mk:output_ids", "kind": "data",
                "guard_ref": "g_decode", "evidence_refs": [],
            })
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("mk.step0.write_col2", msg)

    def test_tamper_sample_op_argmax(self):
        """算子突变：sample 的 gumbel_max_sample→argmax_sample（939-946
        Gumbel-max 语义被偷换，端口/证据/粒度不动，引用合法）——op_type
        闭世界具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.step2.sample":
                    n["op_type"] = "argmax_sample"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("mk.step2.sample", msg)
        self.assertIn("gumbel_max_sample", msg)

    def test_tamper_add_op_overwrite(self):
        """算子突变：step1.add 的 bias_add→覆盖写（869 语义改变）具名
        拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.step1.add":
                    n["op_type"] = "overwrite"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("mk.step1.add", msg)

    def test_tamper_node_port_face_extra(self):
        """端口面突变：为 step2.slice 注入未连线多余输出端口（张量存在、
        无悬空引用）——端口面闭世界具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.step2.slice":
                    n["output_ports"] = n["output_ports"] + [
                        {"name": "extra", "direction": "output",
                         "tensor_id": "tensor:mk:step2_logits_slice"}]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        self.assertIn("mk.step2.slice", str(cm.exception))

    def test_tamper_sample_extra_token_ids_input(self):
        """sample 端口面突变：加回源码签名外的 token_ids 输入端口（虚假
        数据依赖，引用合法）——端口面闭世界具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "mk.step0.sample":
                    n["input_ports"] = [
                        {"name": "logits", "direction": "input",
                         "tensor_id": "tensor:mk:step0_logits_updated"},
                        {"name": "token_ids", "direction": "input",
                         "tensor_id": "tensor:mk:step0_col_read"}]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        self.assertIn("mk.step0.sample", str(cm.exception))

    def test_tamper_vocab_axis_value_synced(self):
        """V 轴 value 129280→129281 同步一致改（named_axes 与张量 axes 两
        处同改，引用检查仍 0）——对照 rules+ModelValues 可信值具名拒绝。"""
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
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("V", msg)
        self.assertIn("129280", msg)

    def test_tamper_rm_axis_value_synced(self):
        """Rm 轴 value 256→257 同步一致改——Rm 本模块可信常量交叉核对
        具名拒绝。"""
        def mutate(doc):
            for a in doc["named_axes"]:
                if a["symbol"] == "Rm":
                    a["value"] = 257
            for t_ in doc["tensors"]:
                for axis in t_["axes"]:
                    if axis["symbol"] == "Rm":
                        axis["value"] = 257
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("Rm", msg)
        self.assertIn("256", msg)

    def test_tamper_version_transition_direction(self):
        """transition 方向被改：tr_mk_logits_s1 的 before/after 互换
        （in-place 更新方向 869 被改，V4 无环仍过）——方向核对具名拒绝。"""
        def mutate(doc):
            for tr in doc["state_transitions"]:
                if tr["stable_id"] == "tr_mk_logits_s1":
                    tr["before_version_id"] = "state:mk:logits:v2"
                    tr["after_version_id"] = "state:mk:logits:v1"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_markov_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("tr_mk_logits_s1", msg)


class TestMarkovStepMap(unittest.TestCase):
    """step 映射：canonical 往返、guard 机器求值核对、单 stage 合成。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.values = cls.bundle.values
        cls.text = serialize_markov_step_map(build_markov_step_map(cls.values))

    def test_schema_and_top_keys(self):
        self.assertIn('"schema":"%s"' % MARKOV_SCHEMA, self.text)
        doc = json.loads(self.text)
        self.assertEqual(set(doc), set(MARKOV_STEP_MAP_EXPECTED_TOP))

    def test_canonical_roundtrip(self):
        back = deserialize_markov_step_map(self.text, self.values)
        self.assertEqual(serialize_markov_step_map(back), self.text)

    def test_single_stage_synthesis(self):
        """单 stage first&last 合成（n_mtp_layers==1，B2 纯角色推导）：
        映射 markov_stage 绑定 mtp.0、canonical 往返；同一映射对真实
        3 stage 输入不合法。"""
        single_values = dataclasses.replace(self.values, n_mtp_layers=1)
        doc = build_markov_step_map(single_values)
        self.assertEqual(doc["markov_stage"]["param_path"], "mtp.0")
        text = serialize_markov_step_map(doc)
        back = deserialize_markov_step_map(text, single_values)
        self.assertEqual(serialize_markov_step_map(back), text)
        # 同一映射对真实 3 stage 输入不合法（n_stages=1 ≠ 派生 3）
        with self.assertRaises(ExtractionError) as cm:
            deserialize_markov_step_map(text, self.values)
        self.assertIn("n_stages", str(cm.exception))


class TestMarkovStepMapTamper(unittest.TestCase):
    """step 映射级篡改：n_stages 类型/顶层键/pending 删除/guard prefill
    误启——读回全量核对具名拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.values = build().values
        cls.text = serialize_markov_step_map(build_markov_step_map(cls.values))

    def _roundtrip_tampered(self, doc):
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        deserialize_markov_step_map(text, self.values)

    def test_tamper_top_key_added(self):
        doc = json.loads(self.text)
        doc["extra"] = 1
        with self.assertRaises(ExtractionError) as cm:
            self._roundtrip_tampered(doc)
        self.assertIn("顶层键", str(cm.exception))

    def test_tamper_n_stages_type(self):
        for bad in (99.0, True, "3"):
            doc = json.loads(self.text)
            doc["n_stages"] = bad
            with self.assertRaises(ExtractionError) as cm:
                self._roundtrip_tampered(doc)
            self.assertIn("n_stages", str(cm.exception))

    def test_tamper_guard_prefill_enabled(self):
        doc = json.loads(self.text)
        doc["guard"]["prefill_active"] = True
        with self.assertRaises(ExtractionError) as cm:
            self._roundtrip_tampered(doc)
        self.assertIn("prefill", str(cm.exception))

    def test_tamper_scope_pending_removed(self):
        doc = json.loads(self.text)
        doc["scope"]["pending"] = []
        with self.assertRaises(ExtractionError) as cm:
            self._roundtrip_tampered(doc)
        self.assertIn("scope.pending", str(cm.exception))

    def test_tamper_step_version_link(self):
        """steps[2].logits_consume_version 改成 v0（版本链断裂映射）具名
        拒绝。"""
        doc = json.loads(self.text)
        doc["steps"][2]["logits_consume_version"] = "state:mk:logits:v0"
        with self.assertRaises(ExtractionError) as cm:
            self._roundtrip_tampered(doc)
        self.assertIn("steps[2]", str(cm.exception))


# B4 自包含受控 fixtures（tests/fixtures/markov_mutant_*.json，只读）：
# 由真实序列化映射按 Codex 复核同款逐叶子篡改生成（rank2 伪造 seed /
# final_logits dtype / 当前 token embed 换成新采样 token / Gumbel-max 分支
# 换成直接 argmax），替代旧 coordination review JSON 强依赖——最终产品/
# 门禁不再读 .work/coordination（旧 JSON 只读保留在原处，不删除）。
# 文件名 → 期望具名拒绝应命中的叶子路径片段。
CODEX_MUTANTS = (
    ("markov_mutant_rank2_seed.json",
     ("boundaries.external_inputs.seed_input_ids.shape", "[B,1]", "[B]")),
    ("markov_mutant_final_logits_dtype.json",
     ("boundaries.outputs.final_logits.dtype", "bf16", "fp32")),
    ("markov_mutant_wrong_current_embed.json",
     ("boundaries.outputs.markov_embed.note", "新采样token", "当前 token")),
    ("markov_mutant_sample_branch_swap.json",
     ("sample.branches.gumbel_max", "直接 argmax", "Gumbel-max")),
)

CODEX_MUTANT_DIR = Path(__file__).resolve().parent / "fixtures"


class TestMarkovStepMapLeafFieldTamper(unittest.TestCase):
    """定点补修01：deserialize 全量逐叶子闭世界核对。

    - Codex 四份原始复现 JSON（只读回放）全部具名拒绝（修复前基线为
      错误放行，见 logs/b3b_markov_fix01_pre_fix_replay_*.txt）；
    - 正常映射往返仍逐字节一致（构建与读回共用同一可信期望构造）；
    - boundaries/sample/temperature/phases/steps/params 叶子字段篡改
      具名拒绝（从实际构建映射只改一个叶子字段）；
    - int/bool/float 精确类型负例（False==0、2.0==2 类型篡改拒绝）；
    - 必要链正确仍过（正常映射读回返回原 doc）。"""

    @classmethod
    def setUpClass(cls):
        cls.values = build().values
        cls.text = serialize_markov_step_map(build_markov_step_map(cls.values))

    def test_good_map_still_roundtrips_and_returns_doc(self):
        """正常映射读回仍过且返回内容一致（必要链正确仍过）。"""
        back = deserialize_markov_step_map(self.text, self.values)
        self.assertEqual(serialize_markov_step_map(back), self.text)

    def test_codex_mutant_files_replayed_and_rejected(self):
        """Codex 四份原始复现 JSON 从磁盘只读回放：全部具名拒绝，消息
        命中对应叶子路径与得到/期望文本（真实篡改件，非自构负例）。"""
        for name, (path, got_frag, want_frag) in CODEX_MUTANTS:
            with self.subTest(mutant=name):
                body = (CODEX_MUTANT_DIR / name).read_text(encoding="utf-8")
                with self.assertRaises(ExtractionError) as cm:
                    deserialize_markov_step_map(body, self.values)
                msg = str(cm.exception)
                self.assertIn(path, msg)
                self.assertIn(got_frag, msg)
                self.assertIn(want_frag, msg)

    def _tamper_one_field(self, mutate):
        doc = json.loads(self.text)
        mutate(doc)
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        return text

    def test_tamper_boundaries_leaf_fields(self):
        """boundaries 每个被检 boundary 的关键叶子篡改（从实际映射只改
        一个字段，与 Codex 变异法一致）具名拒绝并命中叶子路径。"""
        cases = (
            ("seed rank", lambda d: d["boundaries"]["external_inputs"]
             ["seed_input_ids"].__setitem__("shape", "[B,1]"),
             "seed_input_ids.shape"),
            ("init_logits dtype", lambda d: d["boundaries"]
             ["external_inputs"]["init_head_logits"].__setitem__(
                 "dtype", "bf16"),
             "init_head_logits.dtype"),
            ("init_logits source", lambda d: d["boundaries"]
             ["external_inputs"]["init_head_logits"].__setitem__(
                 "source", "未知来源"),
             "init_head_logits.source"),
            ("output_ids shape", lambda d: d["boundaries"]["outputs"]
             ["output_ids_final"].__setitem__("shape", "[B,K]"),
             "output_ids_final.shape"),
            ("output_ids token_values",
             lambda d: d["boundaries"]["outputs"]["output_ids_final"]
             .__setitem__("token_values", "运行时可算"),
             "output_ids_final.token_values"),
            ("final_logits version", lambda d: d["boundaries"]["outputs"]
             ["final_logits"].__setitem__("version", "state:mk:logits:v0"),
             "final_logits.version"),
            ("embed dtype", lambda d: d["boundaries"]["outputs"]
             ["markov_embed"].__setitem__("dtype", "fp32"),
             "markov_embed.dtype"),
            ("embed note reversal", lambda d: d["boundaries"]["outputs"]
             ["markov_embed"].__setitem__("note",
                                          "使用每步新采样token的embed"),
             "markov_embed.note"),
            ("boundaries extra key", lambda d: d["boundaries"]["outputs"]
             ["final_logits"].__setitem__("extra", 1),
             "final_logits"),
        )
        for label, mutate, path in cases:
            with self.subTest(case=label):
                text = self._tamper_one_field(mutate)
                with self.assertRaises(ExtractionError) as cm:
                    deserialize_markov_step_map(text, self.values)
                self.assertIn(path, str(cm.exception))

    def test_tamper_sample_leaf_fields(self):
        """sample 段叶子篡改：分支文本偷换（非零温度直接 argmax）、
        runtime_values、static_branch_source_checked、module 具名拒绝。"""
        cases = (
            ("gumbel branch text swap", lambda d: d["sample"]["branches"]
             .__setitem__("gumbel_max",
                          "temperature!=0 直接 argmax，跳过softmax/随机噪声"),
             "sample.branches.gumbel_max"),
            ("argmax branch text", lambda d: d["sample"]["branches"]
             .__setitem__("argmax_direct", "任意温度都 argmax"),
             "sample.branches.argmax_direct"),
            ("branches extra key", lambda d: d["sample"]["branches"]
             .__setitem__("topk", "…"),
             "sample.branches"),
            ("runtime_values claim", lambda d: d["sample"]
             .__setitem__("runtime_values", "输出 token 静态可算"),
             "sample.runtime_values"),
            ("source_checked false", lambda d: d["sample"]
             .__setitem__("static_branch_source_checked", False),
             "sample"),
            ("module text", lambda d: d["sample"]
             .__setitem__("module", "model.py:1-10"),
             "sample.module"),
        )
        for label, mutate, path in cases:
            with self.subTest(case=label):
                text = self._tamper_one_field(mutate)
                with self.assertRaises(ExtractionError) as cm:
                    deserialize_markov_step_map(text, self.values)
                self.assertIn(path, str(cm.exception))

    def test_tamper_temperature_and_phases_leaves(self):
        """temperature.value 与 phases 活动布尔叶子篡改具名拒绝。"""
        cases = (
            ("temperature value", lambda d: d["temperature"]
             .__setitem__("value", 0),
             "temperature.value"),
            ("temperature source text", lambda d: d["temperature"]
             .__setitem__("source", "run config 有 temperature 字段"),
             "temperature.source"),
            ("prefill active", lambda d: d["phases"]["prefill"]
             .__setitem__("markov_chain_active", True),
             "phases.prefill.markov_chain_active"),
            ("decode inactive", lambda d: d["phases"]["decode"]
             .__setitem__("markov_chain_active", False),
             "phases.decode.markov_chain_active"),
            ("decode reason", lambda d: d["phases"]["decode"]
             .__setitem__("reason", "prefill 也可达"),
             "phases.decode.reason"),
        )
        for label, mutate, path in cases:
            with self.subTest(case=label):
                text = self._tamper_one_field(mutate)
                with self.assertRaises(ExtractionError) as cm:
                    deserialize_markov_step_map(text, self.values)
                self.assertIn(path, str(cm.exception))

    def test_tamper_steps_and_params_leaves(self):
        """steps/params 叶子篡改（read_col 偏移、w1 dtype、身份串、绑定
        路径、params shape/dtype/op）具名拒绝并命中叶子路径。"""
        cases = (
            ("step read_col shift", lambda d: d["steps"][3]
             .__setitem__("read_col", 4),
             "steps[3].read_col"),
            ("step w1 dtype", lambda d: d["steps"][1]["w1"]
             .__setitem__("dtype", "fp32"),
             "steps[1].w1.dtype"),
            ("step w2 identity", lambda d: d["steps"][2]["w2"]
             .__setitem__("identity", "mk_w2_step2"),
             "steps[2].w2.identity"),
            ("step w1 bind", lambda d: d["steps"][4]["w1"]
             .__setitem__("bind", "mtp.2.markov_head.markov_w2.weight"),
             "steps[4].w1.bind"),
            ("step embed port", lambda d: d["steps"][0]
             .__setitem__("embed_out_port", "out0"),
             "steps[0].embed_out_port"),
            ("params w1 shape", lambda d: d["params"]["markov_w1"]
             .__setitem__("shape", [129280, 128]),
             "params.markov_w1.shape"),
            ("params w2 dtype", lambda d: d["params"]["markov_w2"]
             .__setitem__("dtype", "bf16"),
             "params.markov_w2.dtype"),
            ("params w1 op", lambda d: d["params"]["markov_w1"]
             .__setitem__("op", "ParallelHead"),
             "params.markov_w1.op"),
            ("params w2 checkpoint_dtype", lambda d: d["params"]["markov_w2"]
             .__setitem__("checkpoint_dtype", "fp32"),
             "params.markov_w2.checkpoint_dtype"),
            ("markov_stage param_path", lambda d: d["markov_stage"]
             .__setitem__("param_path", "mtp.1"),
             "markov_stage.param_path"),
        )
        for label, mutate, path in cases:
            with self.subTest(case=label):
                text = self._tamper_one_field(mutate)
                with self.assertRaises(ExtractionError) as cm:
                    deserialize_markov_step_map(text, self.values)
                self.assertIn(path, str(cm.exception))

    def test_exact_type_rejections(self):
        """精确类型负例（Python 宽松相等曾会放行）：False==0、2.0==2、
        True==1、0==False 在标称严格映射中不再通过——逐字段具名拒绝。"""
        cases = (
            ("n_stages 3.0", lambda d: d.__setitem__("n_stages", 3.0),
             "n_stages"),
            ("steps[0].step 0.0", lambda d: d["steps"][0]
             .__setitem__("step", 0.0),
             "steps[0].step"),
            ("steps[0].read_col False", lambda d: d["steps"][0]
             .__setitem__("read_col", False),
             "steps[0].read_col"),
            ("steps[1].write_col 2.0", lambda d: d["steps"][1]
             .__setitem__("write_col", 2.0),
             "steps[1].write_col"),
            ("steps[4].logits_pos True", lambda d: d["steps"][4]
             .__setitem__("logits_pos", True),
             "steps[4].logits_pos"),
            ("markov_stage.block_size 5.0", lambda d: d["markov_stage"]
             .__setitem__("block_size", 5.0),
             "markov_stage.block_size"),
            ("temperature.value 1.0", lambda d: d["temperature"]
             .__setitem__("value", 1.0),
             "temperature.value"),
            ("temperature.value True", lambda d: d["temperature"]
             .__setitem__("value", True),
             "temperature.value"),
            ("prefill active 0", lambda d: d["phases"]["prefill"]
             .__setitem__("markov_chain_active", 0),
             "phases.prefill.markov_chain_active"),
            ("decode active 1", lambda d: d["phases"]["decode"]
             .__setitem__("markov_chain_active", 1),
             "phases.decode.markov_chain_active"),
            ("params shape element 256.0", lambda d: d["params"]
             ["markov_w1"].__setitem__("shape", [129280, 256.0]),
             "params.markov_w1.shape"),
            ("static_branch_source_checked 1", lambda d: d["sample"]
             .__setitem__("static_branch_source_checked", 1),
             "sample"),
        )
        for label, mutate, path in cases:
            with self.subTest(case=label):
                text = self._tamper_one_field(mutate)
                with self.assertRaises(ExtractionError) as cm:
                    deserialize_markov_step_map(text, self.values)
                self.assertIn(path, str(cm.exception))


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
