"""DSpark Confidence 前向链最小测试（B3b 限定恢复01，resume_b3b_confidence01.md；
定点补修01 后追加公开 builder 隔离与必要 data 边 kind 负例，codex_b3b_confidence_fix01.md）。

正例：真实图构建（8 节点/7 边/7 张量）、decode-only guard 机器可检、
canonical 往返、重复构建一致、实质链（concat hidden 在前→float→proj
→squeeze）、proj.weight 单一参数身份 fp32 [C1,H+Rm] 无 bias、映射
canonical 往返 + single-stage first&last 合成往返、公开 builder 返回
对象与内部可信基准递归隔离（逐段变异后 fresh build 仍 canonical）。

负例（引用合法语义篡改，先证 validate_graph_references 为 0 再断言
verify_confidence_graph 具名拒绝）：删每条必要边（7/7）、必要边
data→control/select/state 逐 kind 换、必要边挂 state_transition_ref、
hidden 边界身份换成 post-norm 张量、concat 端口序交换、float 绕过
（proj 消费 bf16）/float 算子偷换、权重转置/dtype 混淆（runtime fp32
vs checkpoint bf16）、squeeze 错维/输出秩错、参数复制身份、prefill 误启
（去 guard/改表达式）、同步改 named_axes+tensor.axes 轴值；映射读回来
源 producer 换 post-norm/虚构 embed 节点、shape/dtype 篡改、链路算子与
拼接序、参数形状转置/bias/stage、guard/相位类型、pending 删除；映射
公开返回对象直接变异（不经 json.loads 往返）坏 map 拒绝 + fresh build
不受污染。
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
from structure_reader.extract.dspark_confidence import (
    CONFIDENCE_SCHEMA,
    CONFIDENCE_TEMPLATE_ID,
    build_confidence_map,
    build_dspark_confidence_graph,
    deserialize_confidence_map,
    serialize_confidence_map,
    verify_confidence_graph,
)
from structure_reader.graph_document import graph_from_json, graph_to_json
from structure_reader.model_registry import find_workspace_root
from structure_reader.scenario_evaluator import evaluate_bool

ROOT = find_workspace_root()


def build():
    return build_dspark_confidence_graph(ROOT)


class TestConfidenceGraph(unittest.TestCase):
    """Confidence 前向链结构：真实节点/边、guard 机器可检、canonical
    往返、重复构建一致。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = {n.stable_id: n for n in cls.graph.nodes}
        cls.edges = {e.stable_id: e for e in cls.graph.edges}
        cls.values = cls.bundle.values

    def test_template_id(self):
        self.assertEqual(self.graph.template_id, CONFIDENCE_TEMPLATE_ID)

    def test_zero_reference_issues(self):
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_world_counts(self):
        """闭世界计数：8 节点/7 边/7 张量/0 版本/0 transition。"""
        g = self.graph
        self.assertEqual(len(g.nodes), 8)
        self.assertEqual(len(g.edges), 7)
        self.assertEqual(len(g.tensors), 7)
        self.assertEqual(len(g.state_versions), 0)
        self.assertEqual(len(g.state_transitions), 0)

    def test_chain_real(self):
        """实质数据链存在：hidden/embed 两外部端口 → concat（hidden 在
        前）→ float_cast → proj（w 端口）→ squeeze → 唯一输出。"""
        e = self.edges
        n = self.nodes
        for sid in ("cf.hidden_src", "cf.embed_src", "cf.concat",
                    "cf.to_float", "cf.proj_w", "cf.proj", "cf.squeeze",
                    "cf.confidence_out"):
            self.assertIn(sid, n, sid)
        for eid in ("e_cf_hidden_in", "e_cf_embed_in", "e_cf_concat_float",
                    "e_cf_float_proj", "e_cf_w_proj", "e_cf_proj_squeeze",
                    "e_cf_squeeze_out"):
            self.assertIn(eid, e)
        # concat 端口序即 814 拼接序：hidden 在前、markov_embed 在后
        concat = n["cf.concat"]
        self.assertEqual([p.name for p in concat.input_ports],
                         ["hidden", "embed"])
        # proj 输出 [B,K,1] 经 squeeze(-1) 到 [B,K]
        self.assertEqual(e["e_cf_proj_squeeze"].dst_node_id, "cf.squeeze")
        self.assertEqual(e["e_cf_squeeze_out"].dst_node_id,
                         "cf.confidence_out")

    def test_mandatory_edges_are_data(self):
        """正例：7 条必要链边全部 kind=data（真实张量消费），无
        state_transition_ref（前向链无跨调用状态版本）。"""
        for e in self.graph.edges:
            self.assertEqual(e.kind, "data", e.stable_id)
            self.assertIsNone(e.state_transition_ref, e.stable_id)

    def test_decode_only_guard_on_every_edge(self):
        """整条链 decode 活动、prefill 不活动：唯一 g_decode guard
        （start_pos > 0 机器求值）且每条边都挂它。"""
        guards = {g.stable_id: g for g in self.graph.guards}
        self.assertEqual(set(guards), {"g_decode"})
        self.assertFalse(evaluate_bool("start_pos > 0", {"start_pos": 0}))
        self.assertTrue(evaluate_bool("start_pos > 0", {"start_pos": 1}))
        for e in self.graph.edges:
            self.assertEqual(e.guard_ref, "g_decode")

    def test_param_single_identity_no_bias(self):
        """proj.weight 单一参数身份 fp32 [C1, H_plus_Rm]，恰一条消费边；
        无 bias 张量/节点。"""
        tensors = {t.name: t for t in self.graph.tensors}
        self.assertIn("cf:proj_w", tensors)
        w = tensors["cf:proj_w"]
        self.assertEqual(w.dtype, "fp32")
        self.assertEqual(tuple(w.logical_layout), ("C1", "H_plus_Rm"))
        consumers = [e for e in self.graph.edges
                     if e.tensor_id == w.stable_id]
        self.assertEqual(len(consumers), 1)
        self.assertEqual(consumers[0].dst_node_id, "cf.proj")
        self.assertEqual(consumers[0].dst_port, "w")
        for t_ in self.graph.tensors:
            self.assertNotIn("bias", t_.name)
        for nd in self.graph.nodes:
            self.assertNotIn("bias", nd.stable_id)

    def test_trusted_values_cross_check(self):
        """派生轴与受控 values 交叉核对：H=4096/Rm=256/K=5/H+Rm=4352，
        B 动态（value=None 表达式轴）。"""
        axes = {a.symbol: a for a in self.graph.named_axes}
        self.assertEqual(axes["H"].value, 4096)
        self.assertEqual(axes["Rm"].value, 256)
        self.assertEqual(axes["N_draft"].value, 5)
        self.assertEqual(axes["H_plus_Rm"].value, 4352)
        self.assertEqual(axes["C1"].value, 1)
        self.assertIsNone(axes["B"].value)
        self.assertIsNotNone(axes["B"].expression)

    def test_canonical_json_roundtrip(self):
        text = graph_to_json(self.graph)
        rebuilt = graph_from_json(text)
        self.assertEqual(graph_to_json(rebuilt), text)
        verify_confidence_graph(rebuilt, self.values)  # 读回仍过语义核对

    def test_deterministic_rebuild(self):
        g1 = graph_to_json(build().graph)
        g2 = graph_to_json(build().graph)
        self.assertEqual(g1, g2)


class TestConfidenceTamper(unittest.TestCase):
    """对真实序列化图文档的引用合法语义篡改：具名拒绝，非悬空引用兜底。
    每例先断言 validate_graph_references 为 0（篡改后引用合法），再断言
    verify_confidence_graph 具名拒绝。"""

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

    # 删必要边：边 ID → (dst 节点, dst 端口)——删边同时删端口保持引用
    # 合法（与 Markov 同口径）。
    _EDGES = {
        "e_cf_hidden_in": ("cf.concat", "hidden"),
        "e_cf_embed_in": ("cf.concat", "embed"),
        "e_cf_concat_float": ("cf.to_float", "x"),
        "e_cf_float_proj": ("cf.proj", "x"),
        "e_cf_w_proj": ("cf.proj", "w"),
        "e_cf_proj_squeeze": ("cf.squeeze", "x"),
        "e_cf_squeeze_out": ("cf.confidence_out", "x"),
    }

    def test_delete_each_mandatory_edge(self):
        """删每条必要边（7/7）：链断链逐条具名拒绝。"""
        for eid, (dst, port) in self._EDGES.items():
            with self.subTest(edge=eid):
                def mutate(doc):
                    doc["edges"] = [e for e in doc["edges"]
                                    if e["stable_id"] != eid]
                    for n in doc["nodes"]:
                        if n["stable_id"] == dst:
                            n["input_ports"] = [
                                p for p in n["input_ports"]
                                if p["name"] != port]
                g = self._tamper(mutate)
                self.assertEqual(validate_graph_references(g), ())
                with self.assertRaises(ExtractionError) as cm:
                    verify_confidence_graph(g, self.values)
                self.assertIn(eid, str(cm.exception))

    def test_mandatory_edge_kind_demoted(self):
        """必要 data 边 kind 篡改：逐边 data→control（e_cf_float_proj 是
        815 concat_f32 fp32 进 proj 的唯一真实数据边，改 control 即不再
        表达实际消费、断开真实图链）与 data→select——两种变异均引用
        合法（V3 对四类 kind 一律只核端点/张量），逐边具名拒绝。"""
        for eid in self._EDGES:
            for bad_kind in ("control", "select"):
                with self.subTest(edge=eid, kind=bad_kind):
                    def mutate(doc):
                        for e in doc["edges"]:
                            if e["stable_id"] == eid:
                                e["kind"] = bad_kind
                    g = self._tamper(mutate)
                    self.assertEqual(validate_graph_references(g), ())
                    with self.assertRaises(ExtractionError) as cm:
                        verify_confidence_graph(g, self.values)
                    msg = str(cm.exception)
                    self.assertIn(eid, msg)
                    self.assertIn(bad_kind, msg)

    def test_mandatory_edge_state_kind_rejected_at_ir(self):
        """必要边 data→state：state 边必须携带 state_transition_ref
        （graph_ir Edge 构造约束），不带 ref 的 state 变异在反序列化层
        即受控拒绝（edges[i] 定位），到不了语义核对层——guard/端口保持
        原样，仅 kind 与 ref 字段变异。"""
        doc = copy.deepcopy(self.doc)
        for e in doc["edges"]:
            if e["stable_id"] == "e_cf_float_proj":
                e["kind"] = "state"
        with self.assertRaises(ValueError) as cm:
            graph_from_json(json.dumps(doc, ensure_ascii=False,
                                       sort_keys=True, separators=(",", ":")))
        self.assertIn("state", str(cm.exception))

    def test_mandatory_edge_attaches_state_transition_ref(self):
        """必要 data 边挂 state_transition_ref（state 身份不适用前向
        链）：data 边携带该引用在 graph_ir Edge 构造层即拒绝（仅 state
        边可用），反序列化受控拒绝。"""
        doc = copy.deepcopy(self.doc)
        for e in doc["edges"]:
            if e["stable_id"] == "e_cf_float_proj":
                e["state_transition_ref"] = "st_confidence"
        with self.assertRaises(ValueError) as cm:
            graph_from_json(json.dumps(doc, ensure_ascii=False,
                                       sort_keys=True, separators=(",", ":")))
        self.assertIn("state_transition_ref", str(cm.exception))

    def test_rewire_hidden_boundary_to_postnorm_identity(self):
        """hidden 边界身份换成 post-norm 张量（exit:norm_x 同形 [B,K,H]
        bf16——shape/dtype 核对不可分辨，张量身份闭世界可分辨）：两个
        变体均具名拒绝。变体 A 一致换名 → 原身份缺失；变体 B 保留原
        身份注入 impostor → 多余张量具名拒绝。"""
        def rewire(doc):
            """hidden_src/concat.hidden/e_cf_hidden_in 全引用同步到
            tensor:cf:norm_x_in。"""
            for n in doc["nodes"]:
                if n["stable_id"] == "cf.hidden_src":
                    n["output_ports"][0]["tensor_id"] = "tensor:cf:norm_x_in"
                if n["stable_id"] == "cf.concat":
                    for p in n["input_ports"]:
                        if p["name"] == "hidden":
                            p["tensor_id"] = "tensor:cf:norm_x_in"
            for e in doc["edges"]:
                if e["stable_id"] == "e_cf_hidden_in":
                    e["tensor_id"] = "tensor:cf:norm_x_in"

        # 变体 A：一致换名（cf:hidden_in → cf:norm_x_in）
        def mutate_a(doc):
            for t in doc["tensors"]:
                if t["name"] == "cf:hidden_in":
                    t["name"] = "cf:norm_x_in"
                    t["stable_id"] = "tensor:cf:norm_x_in"
            rewire(doc)
        g = self._tamper(mutate_a)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        self.assertIn("cf:hidden_in", str(cm.exception))

        # 变体 B：原身份保留、impostor 张量注入（同形同 dtype，仅名字
        # 不同）并全引用重接——多余张量闭世界具名拒绝。
        def mutate_b(doc):
            for t in doc["tensors"]:
                if t["name"] == "cf:hidden_in":
                    impostor = copy.deepcopy(t)
                    impostor["name"] = "cf:norm_x_in"
                    impostor["stable_id"] = "tensor:cf:norm_x_in"
                    doc["tensors"].append(impostor)
            rewire(doc)
        g = self._tamper(mutate_b)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        self.assertIn("cf:norm_x_in", str(cm.exception))

    def test_concat_order_swap(self):
        """concat 拼接序交换：hidden 端口改消费 embed 张量、embed 端口改
        消费 hidden 张量（三方一致、引用合法）——814 [hidden, markov_
        embed] 序被换，端口面闭世界具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "cf.concat":
                    n["input_ports"] = [
                        {"name": "hidden", "direction": "input",
                         "tensor_id": "tensor:cf:markov_embed_in"},
                        {"name": "embed", "direction": "input",
                         "tensor_id": "tensor:cf:hidden_in"},
                    ]
            for e in doc["edges"]:
                if e["stable_id"] == "e_cf_hidden_in":
                    e["dst_port"] = "embed"
                if e["stable_id"] == "e_cf_embed_in":
                    e["dst_port"] = "hidden"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("cf.concat", msg)
        self.assertIn("端口", msg)

    def test_float_bypass_proj_consumes_bf16(self):
        """绕过 float：proj.x 一致重接到 concat.y（bf16）——815 独立显式
        float 转换被跳过，边闭世界具名拒绝。"""
        def mutate(doc):
            for e in doc["edges"]:
                if e["stable_id"] == "e_cf_float_proj":
                    e["src_node_id"] = "cf.concat"
                    e["tensor_id"] = "tensor:cf:concat"
            for n in doc["nodes"]:
                if n["stable_id"] == "cf.proj":
                    for p in n["input_ports"]:
                        if p["name"] == "x":
                            p["tensor_id"] = "tensor:cf:concat"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        self.assertIn("e_cf_float_proj", str(cm.exception))

    def test_float_op_replaced_by_identity(self):
        """float 算子偷换：float_cast→identity（转换语义被绕）——算子
        闭世界具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "cf.to_float":
                    n["op_type"] = "identity"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("cf.to_float", msg)
        self.assertIn("float_cast", msg)

    def test_weight_transposed(self):
        """权重转置：proj_w [C1,H+Rm]→[H+Rm,C1]（轴序同步换）——shape
        闭世界具名拒绝。"""
        def mutate(doc):
            for t in doc["tensors"]:
                if t["name"] == "cf:proj_w":
                    t["axes"] = [t["axes"][1], t["axes"][0]]
                    t["logical_layout"] = ["H_plus_Rm", "C1"]
                    t["physical_layout"] = ["H_plus_Rm", "C1"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("cf:proj_w", msg)

    def test_weight_dtype_runtime_checkpoint_confusion(self):
        """dtype 混淆：runtime proj_w fp32→bf16（checkpoint 存储 dtype
        泄漏进运行参数，810 注释两态分开）——具名拒绝。"""
        def mutate(doc):
            for t in doc["tensors"]:
                if t["name"] == "cf:proj_w":
                    t["dtype"] = "bf16"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("cf:proj_w", msg)

    def test_concat_f32_dtype_demoted(self):
        """float 产物 dtype 降级：concat_f32 fp32→bf16（显式转换失效）——
        具名拒绝。"""
        def mutate(doc):
            for t in doc["tensors"]:
                if t["name"] == "cf:concat_f32":
                    t["dtype"] = "bf16"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        self.assertIn("cf:concat_f32", str(cm.exception))

    def test_squeeze_wrong_dim(self):
        """squeeze 错维：squeeze_last→squeeze_dim0（815 squeeze(-1) 末维
        语义被换）——算子闭世界具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "cf.squeeze":
                    n["op_type"] = "squeeze_dim0"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("cf.squeeze", msg)
        self.assertIn("squeeze_last", msg)

    def test_confidence_keeps_rank3(self):
        """输出秩错：confidence 保持 [B,K,1]（squeeze 未生效）——shape
        闭世界具名拒绝。"""
        def mutate(doc):
            # C1 轴取 proj_out 的完整定义（与 named_axes 一致，保证引用
            # 合法——语义拒绝由 shape 闭世界给出，非悬空/定义不一致兜底）
            c1 = next(a for t in doc["tensors"] if t["name"] == "cf:proj_out"
                      for a in t["axes"] if a["symbol"] == "C1")
            for t in doc["tensors"]:
                if t["name"] == "cf:confidence":
                    t["axes"] = t["axes"] + [copy.deepcopy(c1)]
                    t["logical_layout"] = ["B", "N_draft", "C1"]
                    t["physical_layout"] = ["B", "N_draft", "C1"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        self.assertIn("cf:confidence", str(cm.exception))

    def test_param_copy_identity(self):
        """参数复制身份：新增 proj_w2 张量+参数节点，cf.proj.w 重接到
        复制身份（三方一致、引用合法）——节点/张量/边闭世界具名拒绝。"""
        def mutate(doc):
            for t in doc["tensors"]:
                if t["name"] == "cf:proj_w":
                    t2 = copy.deepcopy(t)
                    t2["name"] = "cf:proj_w2"
                    t2["stable_id"] = "tensor:cf:proj_w2"
                    doc["tensors"].append(t2)
            for n in doc["nodes"]:
                if n["stable_id"] == "cf.proj_w":
                    n2 = copy.deepcopy(n)
                    n2["stable_id"] = "cf.proj_w2"
                    n2["output_ports"][0]["tensor_id"] = "tensor:cf:proj_w2"
                    doc["nodes"].append(n2)
                if n["stable_id"] == "cf.proj":
                    for p in n["input_ports"]:
                        if p["name"] == "w":
                            p["tensor_id"] = "tensor:cf:proj_w2"
            for e in doc["edges"]:
                if e["stable_id"] == "e_cf_w_proj":
                    e["src_node_id"] = "cf.proj_w2"
                    e["tensor_id"] = "tensor:cf:proj_w2"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("cf.proj_w2", msg)

    def test_guard_removed_from_edge(self):
        """prefill 误启：删一条边的 guard（prefill 可达）——逐边 guard
        核对具名拒绝。"""
        def mutate(doc):
            for e in doc["edges"]:
                if e["stable_id"] == "e_cf_hidden_in":
                    e["guard_ref"] = None
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("e_cf_hidden_in", msg)
        self.assertIn("g_decode", msg)

    def test_guard_expr_tampered(self):
        """guard 表达式篡改：start_pos > 0 → start_pos >= 0（prefill
        误启）——表达式原文 + 机器求值具名拒绝。"""
        def mutate(doc):
            for g_ in doc["guards"]:
                if g_["stable_id"] == "g_decode":
                    g_["predicate_expr"] = "start_pos >= 0"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("start_pos >= 0", msg)

    def test_sync_axis_value_tamper(self):
        """同步改轴值：named_axes H 4096→4097 且全部 tensor.axes H 同步
        （引用合法）——可信轴值（H=dim=4096）具名拒绝。"""
        def mutate(doc):
            for a in doc["named_axes"]:
                if a["symbol"] == "H":
                    a["value"] = 4097
            for t in doc["tensors"]:
                for a in t["axes"]:
                    if a["symbol"] == "H":
                        a["value"] = 4097
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_confidence_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("H", msg)
        self.assertIn("4096", msg)


class TestConfidenceMapSmoke(unittest.TestCase):
    """映射正例：构造、schema、来源 producer 四元组、canonical 往返、
    single-stage 合成。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.values = cls.bundle.values
        cls.text = serialize_confidence_map(build_confidence_map(cls.values))

    def test_schema_and_top(self):
        doc = json.loads(self.text)
        self.assertEqual(doc["schema"], CONFIDENCE_SCHEMA)
        self.assertEqual(doc["confidence_stage"]["param_path"], "mtp.2")
        # 来源映射 producer 四元组（prenorm/markov_embed 各一）
        self.assertEqual(doc["inputs"]["hidden"]["producer_template"],
                         "deepseek_v4_flash_0731:dspark_exit")
        self.assertEqual(doc["inputs"]["hidden"]["producer_node"],
                         "exit.hc_head.reduce")
        self.assertEqual(doc["inputs"]["hidden"]["producer_port"], "y")
        self.assertEqual(doc["inputs"]["hidden"]["producer_tensor"],
                         "exit:prenorm_x")
        self.assertEqual(doc["inputs"]["markov_embed"]["producer_template"],
                         "deepseek_v4_flash_0731:dspark_markov")
        self.assertEqual(doc["inputs"]["markov_embed"]["producer_node"],
                         "mk.stack_embed")
        self.assertEqual(doc["inputs"]["markov_embed"]["producer_port"],
                         "stacked")
        self.assertEqual(doc["inputs"]["markov_embed"]["producer_tensor"],
                         "mk:markov_embed")
        # 参数形状 [1, 4352] 不转置，无 bias，runtime/checkpoint 分开
        p = doc["params"]["confidence_proj"]
        self.assertEqual(p["shape"], [1, 4352])
        self.assertEqual(p["dtype"], "fp32")
        self.assertEqual(p["checkpoint_dtype"], "bf16")
        self.assertIsNone(p["bias"])
        self.assertEqual(p["bind"], "mtp.2.confidence_head.proj.weight")

    def test_roundtrip(self):
        back = deserialize_confidence_map(self.text, self.values)
        self.assertEqual(serialize_confidence_map(back), self.text)

    def test_single_stage_synthesis(self):
        """单 stage first&last 合成（n_mtp_layers==1，B2 纯角色推导）：
        映射 confidence_stage 绑定 mtp.0、canonical 往返；同一映射对
        真实 3 stage 输入不合法。"""
        single_values = dataclasses.replace(self.values, n_mtp_layers=1)
        doc = build_confidence_map(single_values)
        self.assertEqual(doc["confidence_stage"]["param_path"], "mtp.0")
        text = serialize_confidence_map(doc)
        back = deserialize_confidence_map(text, single_values)
        self.assertEqual(serialize_confidence_map(back), text)
        with self.assertRaises(ExtractionError) as cm:
            deserialize_confidence_map(text, self.values)
        self.assertIn("n_stages", str(cm.exception))


class TestConfidenceMapIsolation(unittest.TestCase):
    """公开 builder 返回对象与内部可信基准递归隔离（定点补修01）。

    负例直接变异真实公开 build_confidence_map 返回对象（不经
    json.loads 往返、不触碰任何私有全局），覆盖 inputs/chain（嵌套
    list）/phases/boundaries 四段；断言坏 map 被读回具名拒绝、变异后
    fresh build 仍是原 canonical、变异前已构建的另一映射不受影响。
    全程不通过恢复/重载私有全局来"清理"污染。"""

    @classmethod
    def setUpClass(cls):
        cls.values = build().values
        # baseline 先于任何变异捕获；other 是变异前已构建的另一映射
        cls.baseline = serialize_confidence_map(build_confidence_map(
            cls.values))
        cls.other = build_confidence_map(cls.values)

    def _mutated_map(self, mutate):
        """拿真实公开返回对象，直接变异并断言变异确实生效（隔离靠深
        拷贝基准，不靠只读代理/拒绝变异）。"""
        m = build_confidence_map(self.values)
        mutate(m)
        return m

    def test_inputs_mutation_isolated(self):
        """inputs 段（Codex public_builder_alias_probe 同复现）：
        producer 换 post-norm——坏 map 具名拒绝，fresh/other 不受污染。"""
        def mutate(m):
            m["inputs"]["hidden"]["producer_node"] = "exit.norm"
            m["inputs"]["hidden"]["producer_tensor"] = "exit:norm_x"
        m = self._mutated_map(mutate)
        self.assertEqual(m["inputs"]["hidden"]["producer_node"],
                         "exit.norm")  # 变异对公开对象生效
        with self.assertRaises(ExtractionError) as cm:
            deserialize_confidence_map(serialize_confidence_map(m),
                                       self.values)
        self.assertIn("inputs.hidden.producer_node", str(cm.exception))
        self.assertEqual(serialize_confidence_map(
            build_confidence_map(self.values)), self.baseline)
        self.assertEqual(serialize_confidence_map(self.other),
                         self.baseline)

    def test_chain_nested_list_mutation_isolated(self):
        """chain 段嵌套 list（814 拼接序）与算子偷换分别变异：各自坏
        map 具名拒绝，基准不受污染。"""
        def mutate_order(m):
            m["chain"]["concat"]["inputs"] = ["markov_embed", "hidden"]

        def mutate_op(m):
            m["chain"]["float_cast"]["op"] = "identity"

        for mutate, path in ((mutate_order, "chain.concat.inputs"),
                             (mutate_op, "chain.float_cast.op")):
            m = self._mutated_map(mutate)
            with self.assertRaises(ExtractionError) as cm:
                deserialize_confidence_map(serialize_confidence_map(m),
                                           self.values)
            self.assertIn(path, str(cm.exception))
        self.assertEqual(serialize_confidence_map(
            build_confidence_map(self.values)), self.baseline)
        self.assertEqual(serialize_confidence_map(self.other),
                         self.baseline)

    def test_phases_mutation_isolated(self):
        """phases 段（嵌套 dict）：decode.confidence_active true→False
        ——坏 map 具名拒绝，基准不受污染。"""
        def mutate(m):
            m["phases"]["decode"]["confidence_active"] = False
        m = self._mutated_map(mutate)
        self.assertIs(m["phases"]["decode"]["confidence_active"], False)
        with self.assertRaises(ExtractionError) as cm:
            deserialize_confidence_map(serialize_confidence_map(m),
                                       self.values)
        self.assertIn("phases.decode.confidence_active",
                      str(cm.exception))
        self.assertEqual(serialize_confidence_map(
            build_confidence_map(self.values)), self.baseline)
        self.assertEqual(serialize_confidence_map(self.other),
                         self.baseline)

    def test_boundaries_mutation_isolated(self):
        """boundaries 段：输出 dtype fp32→bf16（runtime/checkpoint 混淆
        语义）——坏 map 具名拒绝，基准不受污染。"""
        def mutate(m):
            m["boundaries"]["output_confidence"]["dtype"] = "bf16"
        m = self._mutated_map(mutate)
        self.assertEqual(m["boundaries"]["output_confidence"]["dtype"],
                         "bf16")
        with self.assertRaises(ExtractionError) as cm:
            deserialize_confidence_map(serialize_confidence_map(m),
                                       self.values)
        self.assertIn("boundaries.output_confidence.dtype",
                      str(cm.exception))
        self.assertEqual(serialize_confidence_map(
            build_confidence_map(self.values)), self.baseline)
        self.assertEqual(serialize_confidence_map(self.other),
                         self.baseline)

    def test_mutation_between_two_public_maps_no_crosstalk(self):
        """交叉污染负例：对 map A 的 inputs 变异后，map B（后构建）与
        先前 deserialize 读回的正常文本都不受影响——别名只在单次返回
        值内部，无跨调用共享可变状态。"""
        a = build_confidence_map(self.values)
        a["inputs"]["markov_embed"]["producer_node"] = "mk.sample_lookup"
        b = build_confidence_map(self.values)
        self.assertEqual(b["inputs"]["markov_embed"]["producer_node"],
                         "mk.stack_embed")
        back = deserialize_confidence_map(self.baseline, self.values)
        self.assertEqual(serialize_confidence_map(back), self.baseline)


class TestConfidenceMapTamper(unittest.TestCase):
    """映射读回篡改：来源 producer/shape/dtype、链路算子/拼接序、参数
    形状转置/bias/stage、guard/相位类型、pending 删除——读回全量核对
    具名拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.values = build().values
        cls.text = serialize_confidence_map(
            build_confidence_map(cls.values))

    def _tampered(self, mutate, substring):
        doc = json.loads(self.text)
        mutate(doc)
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_confidence_map(text, self.values)
        self.assertIn(substring, str(cm.exception))

    def test_hidden_producer_postnorm(self):
        """来源换 post-norm：producer exit.hc_head.reduce/exit:prenorm_x
        → exit.norm/exit:norm_x——producer 四元组逐字段具名拒绝。"""
        def mutate(doc):
            doc["inputs"]["hidden"]["producer_node"] = "exit.norm"
            doc["inputs"]["hidden"]["producer_tensor"] = "exit:norm_x"
        self._tampered(mutate, "inputs.hidden.producer_node")

    def test_embed_producer_fictional(self):
        """embed 来源偷换：872 当前 token embed stack → 虚构采样后查表
        节点——具名拒绝。"""
        def mutate(doc):
            doc["inputs"]["markov_embed"]["producer_node"] = \
                "mk.sample_lookup"
        self._tampered(mutate, "inputs.markov_embed.producer_node")

    def test_hidden_shape_swap(self):
        """hidden shape 篡改：[B,K,H]→[B,K,Rm]——具名拒绝。"""
        def mutate(doc):
            doc["inputs"]["hidden"]["shape"] = "[B,K,Rm]"
        self._tampered(mutate, "inputs.hidden.shape")

    def test_hidden_dtype_swap(self):
        """hidden dtype 篡改：bf16→fp32——具名拒绝。"""
        def mutate(doc):
            doc["inputs"]["hidden"]["dtype"] = "fp32"
        self._tampered(mutate, "inputs.hidden.dtype")

    def test_chain_concat_order_swapped(self):
        """链路拼接序文案偷换：hidden 在前 → embed 在前——具名拒绝。"""
        def mutate(doc):
            doc["chain"]["concat"]["inputs"] = ["markov_embed", "hidden"]
        self._tampered(mutate, "chain.concat.inputs")

    def test_chain_float_removed(self):
        """链路 float 节点偷换：float_cast→identity（独立显式转换声明被
        绕过）——具名拒绝。"""
        def mutate(doc):
            doc["chain"]["float_cast"]["op"] = "identity"
        self._tampered(mutate, "chain.float_cast.op")

    def test_param_shape_transposed(self):
        """参数形状转置：[1,4352]→[4352,1]——具名拒绝。"""
        def mutate(doc):
            doc["params"]["confidence_proj"]["shape"] = [4352, 1]
        self._tampered(mutate, "params.confidence_proj.shape")

    def test_param_dtype_confusion(self):
        """runtime/checkpoint dtype 混淆：dtype fp32→bf16 且
        checkpoint_dtype bf16→fp32——两者都具名拒绝。"""
        def mutate(doc):
            doc["params"]["confidence_proj"]["dtype"] = "bf16"
        self._tampered(mutate, "params.confidence_proj.dtype")

        def mutate2(doc):
            doc["params"]["confidence_proj"]["checkpoint_dtype"] = "fp32"
        self._tampered(mutate2, "params.confidence_proj.checkpoint_dtype")

    def test_param_bias_false(self):
        """bias None→False（虚构无 bias 语义的替身值）——is 判别具名
        拒绝。"""
        def mutate(doc):
            doc["params"]["confidence_proj"]["bias"] = False
        self._tampered(mutate, "params.confidence_proj.bias")

    def test_param_bind_wrong_stage(self):
        """参数错 stage：mtp.2→mtp.1（confidence_head 仅 last 拥有）——
        具名拒绝。"""
        def mutate(doc):
            doc["params"]["confidence_proj"]["bind"] = \
                "mtp.1.confidence_head.proj.weight"
        self._tampered(mutate, "params.confidence_proj.bind")

    def test_guard_prefill_activated(self):
        """prefill 误启：guard.prefill_active false→true——具名拒绝。"""
        def mutate(doc):
            doc["guard"]["prefill_active"] = True
        self._tampered(mutate, "guard")

    def test_guard_decode_type(self):
        """相位类型篡改：decode_active true→1（int 冒充 bool）——is 判别
        具名拒绝。"""
        def mutate(doc):
            doc["guard"]["decode_active"] = 1
        self._tampered(mutate, "guard")

    def test_n_stages_float(self):
        """n_stages 3→3.0（float 冒充 int）——类型篡改具名拒绝。"""
        def mutate(doc):
            doc["n_stages"] = 3.0
        self._tampered(mutate, "n_stages")

    def test_phases_prefill_active(self):
        """phases.prefill.confidence_active false→true（与 guard 机器
        求值矛盾）——具名拒绝。"""
        def mutate(doc):
            doc["phases"]["prefill"]["confidence_active"] = True
        self._tampered(mutate, "phases.prefill.confidence_active")

    def test_scope_pending_removed(self):
        """pending 删除（把未实现整体拼接/门禁声称为已实现）——scope
        边界具名拒绝。"""
        def mutate(doc):
            doc["scope"]["pending"] = doc["scope"]["pending"][:1]
        self._tampered(mutate, "scope.pending")

    def test_boundary_dtype(self):
        """输出 boundary dtype 篡改：fp32→bf16——具名拒绝。"""
        def mutate(doc):
            doc["boundaries"]["output_confidence"]["dtype"] = "bf16"
        self._tampered(mutate, "boundaries.output_confidence.dtype")


if __name__ == "__main__":
    unittest.main()
