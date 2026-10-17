"""DSpark 首入口图与端口绑定测试（工作包 B · 步骤 B3a 补修01）：

首入口 SemanticGraph（主干 readout 边界 917-925 + forward_embed
851-858 + forward_spec 929-936 首入口部分）结构断言——readout 真实
链（三外部 h [B,T,N_hc,H] → 各自 mean(dim=2) → 有序 concat →
main_proj→main_norm→唯一 main_x）、seed rank-1 [B]（hc_head 折合 +
ParallelHead x[:,-1] → sample [B]，855 首列 [B] 槽写入，rank-2 伪造
拒绝）、noise 填充 + 首列写 + 共享 embed 查表 + HC 展开 repeat、三
stage main_x 分发与草稿 h 仅 stage0、两相位无条件（prefill
start_pos=0 仍执行首入口）、T_main 与 draft K 轴不混；必要生产/消费
链闭世界核对（缺边/一致重接/多余通路具名拒绝）；端口/角色绑定规范
JSON 的 canonical 往返、重复构建一致、读回全量核对；对真实序列化图
文档与绑定文档的引用合法语义篡改（删必要边、一致重接、readout concat
端口交换、seed rank2、注入第二投影路径、共享 embed 复制、stage0 错绑、
guard 停用入口、noise/temperature 伪造）具名拒绝——负例先证
validate_graph_references 为 0 再断言语义拒绝，非悬空引用兜底。

只覆盖 B3a 范围；末出口/Markov/Confidence（B3b）、全量集成（B4）pending，
本测试不声称覆盖。
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_reference import ExtractionError
from structure_reader.extract.dspark_endpoints import (
    ENTRY_SCHEMA,
    ENTRY_TEMPLATE_ID,
    NOISE_TOKEN_ID,
    PREMISE_KEYS,
    build_dspark_entry_graph,
    deserialize_entry_bindings,
    serialize_entry_bindings,
    verify_entry_graph,
)
from structure_reader.extract.dspark_stage import EXPR_MAIN_PROJ_IN
from structure_reader.graph_document import graph_from_json, graph_to_json
from structure_reader.model_registry import find_workspace_root

ROOT = find_workspace_root()


def build():
    return build_dspark_entry_graph(ROOT)


def _node_map(graph):
    return {n.stable_id: n for n in graph.nodes}


def _edges_by_id(graph):
    return {e.stable_id: e for e in graph.edges}


def _tensor_map(graph):
    return {t.name: t for t in graph.tensors}


class TestEntryGraph(unittest.TestCase):
    """首入口图结构：唯一 main_x 投影路径、noise/首列/embed/HC 展开、
    三 stage 分发、两相位无条件、T/K 不混。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = _node_map(cls.graph)
        cls.edges = _edges_by_id(cls.graph)
        cls.doc = json.loads(graph_to_json(cls.graph))

    def test_template_id(self):
        self.assertEqual(self.graph.template_id, ENTRY_TEMPLATE_ID)

    def test_zero_reference_issues(self):
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_canonical_json_roundtrip(self):
        text = graph_to_json(self.graph)
        rebuilt = graph_from_json(text)
        self.assertEqual(graph_to_json(rebuilt), text)

    def test_deterministic_rebuild(self):
        text1 = graph_to_json(build().graph)
        text2 = graph_to_json(build().graph)
        self.assertEqual(text1, text2)

    def test_no_guards_two_phase_unconditional(self):
        """首入口两相位无条件：无 guard 定义、无边挂 guard——prefill
        start_pos=0 仍执行 forward_embed + stage loop（930-932），933-934
        prefill 在 loop 后才 return，不能把入口打 decode-only。"""
        self.assertEqual(len(self.graph.guards), 0)
        for e in self.graph.edges:
            self.assertIsNone(e.guard_ref,
                              f"{e.stable_id} 挂相位 guard {e.guard_ref!r}")

    def test_no_cache_states(self):
        self.assertEqual(len(self.graph.state_versions), 0)
        self.assertEqual(len(self.graph.state_transitions), 0)
        for e in self.graph.edges:
            self.assertIsNone(e.state_transition_ref)

    def test_readout_order_anchored(self):
        """readout 真实链：三外部 h 输入 [B,T,N_hc,H] → reduce_mean →
        唯一 concat 端口 s0/s1/s2 依次消费 mean_h_40/41/42（主干遍历
        顺序）；main_hidden 宽 = H×N_tgt；绑定 readout_order 与固定
        config 一致。"""
        values = self.bundle.values
        self.assertEqual(values.dspark_target_layer_ids, (40, 41, 42))
        tmap = _tensor_map(self.graph)
        for lid in (40, 41, 42):
            self.assertEqual(tmap[f"entry:src_h_{lid}"].logical_layout,
                             ("B", "T", "N_hc", "H"))
            self.assertEqual(tmap[f"entry:mean_h_{lid}"].logical_layout,
                             ("B", "T", "H"))
        mh = tmap["entry:main_hidden"]
        self.assertEqual(mh.logical_layout, ("B", "T", "N_tgtH"))
        means = sorted(n.stable_id for n in self.graph.nodes
                       if n.op_type == "reduce_mean")
        self.assertEqual(means, ["entry.readout_mean_40",
                                 "entry.readout_mean_41",
                                 "entry.readout_mean_42"])
        cats = [n.stable_id for n in self.graph.nodes
                if n.op_type == "concat"]
        self.assertEqual(cats, ["entry.readout_concat"])
        cat = self.nodes["entry.readout_concat"]
        self.assertEqual(
            {p.name: p.tensor_id for p in cat.input_ports},
            {port: tmap[f"entry:mean_h_{lid}"].stable_id
             for lid, port in zip((40, 41, 42), ("s0", "s1", "s2"))})
        edges = _edges_by_id(self.graph)
        for lid, port in zip((40, 41, 42), ("s0", "s1", "s2")):
            e = edges[f"e_ep_mean{lid}_cat"]
            self.assertEqual((e.src_node_id, e.dst_node_id, e.dst_port),
                             (f"entry.readout_mean_{lid}",
                              "entry.readout_concat", port))
        axmap = {a.symbol: a for a in self.graph.named_axes}
        self.assertEqual(axmap["N_tgt"].value, 3)
        self.assertEqual(axmap["N_tgtH"].expression, EXPR_MAIN_PROJ_IN)
        self.assertEqual(self.bundle.bindings.main_x.readout_order,
                         (40, 41, 42))

    def test_seed_rank_one(self):
        """seed input_ids rank-1 [B]：主干 hc_head 折合 + ParallelHead
        x[:,-1] → sample [B]（922-924、939-946），855 首列 [B] 槽精确
        写入；first_col_write.src 消费该 [B] 张量。"""
        tmap = _tensor_map(self.graph)
        seed = tmap["entry:seed_ids"]
        self.assertEqual(seed.logical_layout, ("B",))
        self.assertEqual(seed.dtype, "int64")
        axmap = {a.symbol: a for a in self.graph.named_axes}
        self.assertEqual(axmap["T_in"].value, 1)
        self.assertIn("rank-1", axmap["T_in"].meaning)
        fcw_in = {p.name: p.tensor_id
                  for p in self.nodes["entry.first_col_write"].input_ports}
        self.assertEqual(fcw_in["src"], seed.stable_id)
        edges = _edges_by_id(self.graph)
        e = edges["e_ep_seed_firstcol"]
        self.assertEqual((e.src_node_id, e.dst_node_id, e.dst_port),
                         ("entry.seed_src", "entry.first_col_write", "src"))
        self.assertIn("rank-1 [B]",
                      self.bundle.bindings.call_premises["input_ids_width"])
        self.assertIn("无广播",
                      self.bundle.bindings.call_premises["input_ids_width"])

    def test_unique_main_x_projection_chain(self):
        """唯一 main_x：恰一条 linear（main_proj）+ 恰一条 rms_norm
        （main_norm）投影链；main_x 出边全部来自 main_norm.y；无第二
        投影路径。"""
        tmap = _tensor_map(self.graph)
        mx = tmap["entry:main_x"]
        self.assertEqual(mx.logical_layout, ("B", "T", "H"))
        linear_nodes = [n for n in self.graph.nodes if n.op_type == "linear"]
        self.assertEqual([n.stable_id for n in linear_nodes],
                         ["entry.main_proj"])
        norm_nodes = [n for n in self.graph.nodes if n.op_type == "rms_norm"]
        self.assertEqual([n.stable_id for n in norm_nodes],
                         ["entry.main_norm"])
        # 链条：readout concat → main_proj → main_norm → main_x
        chain = {e.stable_id: e for e in self.graph.edges}
        e1 = chain["e_ep_cat_proj"]
        self.assertEqual(e1.tensor_id, tmap["entry:main_hidden"].stable_id)
        self.assertEqual((e1.src_node_id, e1.dst_node_id),
                         ("entry.readout_concat", "entry.main_proj"))
        e2 = chain["e_ep_proj_norm"]
        self.assertEqual((e2.src_node_id, e2.dst_node_id),
                         ("entry.main_proj", "entry.main_norm"))
        e3 = chain["e_ep_norm_feed0"]
        self.assertEqual((e3.src_node_id, e3.src_port),
                         ("entry.main_norm", "y"))
        # main_proj 权重 fp8 / main_norm weight fp32（真实 dtype 分开）
        self.assertEqual(tmap["entry:main_proj_w"].dtype, "fp8_e4m3fn")
        self.assertEqual(tmap["entry:main_norm_w"].dtype, "fp32")

    def test_main_x_shared_distribution_three_stages(self):
        """同一 main_x 向三 stage 分发：三条分发边共享同一 main_x 张量
        身份，feed_0/1/2 的 x 端口三方一致。"""
        tmap = _tensor_map(self.graph)
        mx_id = tmap["entry:main_x"].stable_id
        feed_edges = [e for e in self.graph.edges
                      if e.dst_node_id in ("entry.stage_feed_0",
                                           "entry.stage_feed_1",
                                           "entry.stage_feed_2")
                      and e.dst_port == "x"]
        self.assertEqual(len(feed_edges), 3)
        for e in feed_edges:
            self.assertEqual(e.tensor_id, mx_id)
            self.assertEqual(e.src_node_id, "entry.main_norm")
        feeds = [n for n in self.graph.nodes
                 if n.op_type == "stage_dispatch"]
        self.assertEqual(sorted(n.stable_id for n in feeds),
                         ["entry.stage_feed_0", "entry.stage_feed_1",
                          "entry.stage_feed_2"])

    def test_draft_h_only_stage0(self):
        """草稿 h 只进 stage0：entry_h 到 stage feed 的边恰为 feed_0.h；
        stage1/2 无 h 端口（h_in 来自前一 stage h_out——B2 链路）。"""
        tmap = _tensor_map(self.graph)
        eh_id = tmap["entry:entry_h"].stable_id
        self.assertEqual(tmap["entry:entry_h"].logical_layout,
                         ("B", "N_draft", "N_hc", "H"))
        h_edges = [e for e in self.graph.edges if e.tensor_id == eh_id]
        to_stage = [(e.dst_node_id, e.dst_port) for e in h_edges
                    if e.dst_node_id.startswith("entry.stage_feed")]
        self.assertEqual(to_stage, [("entry.stage_feed_0", "h")])
        for i in (1, 2):
            ports = [p.name for p in self.nodes[f"entry.stage_feed_{i}"]
                     .input_ports]
            self.assertNotIn("h", ports)

    def test_draft_token_chain(self):
        """noise 填充 → 首列写 rank-1 seed → 共享 embed 查表 →
        unsqueeze/repeat HC 展开：节点与端口逐一核对（model.py:
        854-858）。"""
        nodes = self.nodes
        self.assertEqual(nodes["entry.noise_fill"].op_type, "fill_const")
        self.assertEqual(nodes["entry.first_col_write"].op_type,
                         "index_write")
        fcw_in = {p.name: p.tensor_id
                  for p in nodes["entry.first_col_write"].input_ports}
        tmap = _tensor_map(self.graph)
        self.assertEqual(fcw_in,
                         {"dst": tmap["entry:draft_noise"].stable_id,
                          "src": tmap["entry:seed_ids"].stable_id})
        # draft_input_ids 唯一生产、唯一消费 embed_lookup.x
        diid = tmap["entry:draft_input_ids"]
        di_edges = [e for e in self.graph.edges if e.tensor_id == diid.stable_id]
        self.assertEqual(
            [(e.src_node_id, e.dst_node_id, e.dst_port) for e in di_edges],
            [("entry.first_col_write", "entry.embed_lookup", "x")])
        self.assertEqual(nodes["entry.embed_lookup"].op_type, "embedding")
        self.assertEqual(nodes["entry.hc_expand"].op_type,
                         "unsqueeze_repeat")
        # seed 轴 (B,) rank-1、noise/draft 轴 (B,N_draft)
        self.assertEqual(tmap["entry:seed_ids"].logical_layout, ("B",))
        self.assertEqual(tmap["entry:draft_noise"].logical_layout,
                         ("B", "N_draft"))
        self.assertEqual(tmap["entry:draft_input_ids"].logical_layout,
                         ("B", "N_draft"))

    def test_noise_token_id_anchored(self):
        """noise 常量 128799 与 run config 原文证据锚定（ModelValues 无
        noise_token_id 字段——以常量+ev_ep_noise_cfg 表达；dspark_block_size
        是 ModelValues 实有字段，不误称缺失）。"""
        self.assertEqual(NOISE_TOKEN_ID, 128799)
        self.assertEqual(self.bundle.bindings.draft_tokens.noise_token_id,
                         128799)
        reg = self.bundle.registry
        ev = reg.get("ev_ep_noise_cfg")
        self.assertEqual(ev.line_start, 9)
        self.assertIn("dspark_noise_token_id", ev.snippet)
        self.assertIn("ModelValues", ev.claim)

    def test_shared_embed_single_identity(self):
        """共享 embed 单份身份：恰一个 embedding_weight 节点 + 恰一个
        embedding 查表节点；embed_w 张量恰一；绑定表 copied=false、
        instance_paths = mtp.{0,1,2}.embed（903 同一对象绑定）。"""
        nodes = self.graph.nodes
        emb_w = [n for n in nodes if n.op_type == "embedding_weight"]
        self.assertEqual([n.stable_id for n in emb_w], ["entry.embed_w"])
        lookup = [n for n in nodes if n.op_type == "embedding"]
        self.assertEqual([n.stable_id for n in lookup],
                         ["entry.embed_lookup"])
        tmap = _tensor_map(self.graph)
        ew = [n for n in tmap if "embed_w" in n]
        self.assertEqual(ew, ["entry:embed_w"])
        self.assertEqual(tmap["entry:embed_w"].dtype, "bf16")
        self.assertEqual(tmap["entry:embed_w"].logical_layout, ("V", "H"))
        se = self.bundle.bindings.shared_embed
        self.assertFalse(se.copied)
        self.assertEqual(se.transformer_object, "transformer.embed")
        self.assertEqual(se.instance_paths,
                         ("mtp.0.embed", "mtp.1.embed", "mtp.2.embed"))

    def test_t_main_vs_draft_k_axes(self):
        """T_main 与 draft K 不混：main 链含 T 无 N_draft；draft 链含
        N_draft 无 T；main_proj 不改 T、embed 链不见 T。"""
        tmap = _tensor_map(self.graph)
        for name in ("entry:main_hidden", "entry:main_proj_y",
                     "entry:main_x"):
            syms = tmap[name].logical_layout
            self.assertIn("T", syms, name)
            self.assertNotIn("N_draft", syms, name)
        for name in ("entry:draft_noise", "entry:draft_input_ids",
                     "entry:embed_y", "entry:entry_h"):
            syms = tmap[name].logical_layout
            self.assertIn("N_draft", syms, name)
            self.assertNotIn("T", syms, name)
        axmap = {a.symbol: a for a in self.graph.named_axes}
        self.assertEqual(axmap["T_in"].value, 1)

    def test_verify_entry_graph_passes_and_rejects(self):
        """verify_entry_graph 对合法图通过；对换 template_id 的图拒绝。"""
        verify_entry_graph(self.graph, self.bundle.values)  # 不抛
        tampered = copy.deepcopy(self.doc)
        tampered["template_id"] = "deepseek_v4_flash_0731:dspark_stage"
        text = json.dumps(tampered, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        g2 = graph_from_json(text)
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g2, self.bundle.values)
        self.assertIn("template_id", str(cm.exception))


class TestEntryTamper(unittest.TestCase):
    """对真实序列化图文档的引用合法语义篡改：具名拒绝，非悬空引用兜底。"""

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

    def test_tamper_swap_readout_axis_value(self):
        """交换 readout：绑定文档 readout_order 42/41/40（引用合法）——
        绑定读回以固定 config 拒绝。"""
        doc = copy.deepcopy(json.loads(
            serialize_entry_bindings(self.bundle.bindings)))
        doc["main_x"]["readout_order"] = [42, 41, 40]
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_entry_bindings(text, None, ROOT)
        self.assertIn("readout_order", str(cm.exception))

    def test_tamper_concat_port_swap(self):
        """readout concat 端口交换：mean_h_40 与 mean_h_41 的 concat
        来源交换（节点端口/边三方一致、引用合法）——主干遍历顺序被改
        具名拒绝（不能只改绑定 metadata 作负例）。"""
        def mutate(doc):
            id40 = "tensor:entry:mean_h_40"
            id41 = "tensor:entry:mean_h_41"
            for n in doc["nodes"]:
                if n["stable_id"] == "entry.readout_concat":
                    for p in n["input_ports"]:
                        if p["name"] == "s0":
                            p["tensor_id"] = id41
                        elif p["name"] == "s1":
                            p["tensor_id"] = id40
            for e in doc["edges"]:
                if e["stable_id"] == "e_ep_mean40_cat":
                    e["dst_port"] = "s1"
                elif e["stable_id"] == "e_ep_mean41_cat":
                    e["dst_port"] = "s0"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("readout_concat", msg)

    def test_tamper_concat_mean_rewire_consistent(self):
        """保边 ID 一致重接：e_ep_mean40_cat 的源从 readout_mean_40
        改为 readout_mean_42（端口/张量三方同步、引用合法）——必要链
        源端被重接具名拒绝。"""
        def mutate(doc):
            for e in doc["edges"]:
                if e["stable_id"] == "e_ep_mean40_cat":
                    e["src_node_id"] = "entry.readout_mean_42"
                    e["tensor_id"] = "tensor:entry:mean_h_42"
            for n in doc["nodes"]:
                if n["stable_id"] == "entry.readout_concat":
                    for p in n["input_ports"]:
                        if p["name"] == "s0":
                            p["tensor_id"] = "tensor:entry:mean_h_42"
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("e_ep_mean40_cat", msg)
        self.assertIn("重接", msg)

    def test_tamper_delete_mandatory_edge_mainhid(self):
        """Codex 原始漏检①：删除 e_ep_cat_proj（readout_concat →
        main_proj.x，原 e_ep_mainhid_proj）——引用 0 但必要生产/消费链
        断裂，具名拒绝。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_ep_cat_proj"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("e_ep_cat_proj", msg)
        self.assertIn("缺失", msg)

    def test_tamper_delete_mandatory_edge_proj_norm(self):
        """Codex 原始漏检②：删除 e_ep_proj_norm（main_proj.y →
        main_norm.x）——引用 0 但 main 投影链断裂，具名拒绝。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_ep_proj_norm"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("e_ep_proj_norm", msg)
        self.assertIn("缺失", msg)

    def test_tamper_delete_mandatory_edge_embed_hcexp(self):
        """Codex 原始漏检③：删除 e_ep_embed_hcexp（embed_lookup.y →
        hc_expand.x）——引用 0 但 draft 链断裂，具名拒绝。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_ep_embed_hcexp"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("e_ep_embed_hcexp", msg)
        self.assertIn("缺失", msg)

    def test_tamper_delete_shared_param_consume_edge(self):
        """共享参数消费边丢失：删 e_ep_embedw_embed（embed_w →
        embed_lookup.w）——共享 embed 参数不再被消费，具名拒绝。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_ep_embedw_embed"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("e_ep_embedw_embed", msg)

    def test_tamper_seed_rank2_forgery(self):
        """seed rank-2 伪造：seed_ids 轴改 (B,T_in)（axes 同步替换、引用
        合法）——与主干 sample [B] 不符具名拒绝。"""
        def mutate(doc):
            for t in doc["tensors"]:
                if t["name"] == "entry:seed_ids":
                    t_in = [a for a in doc["named_axes"]
                            if a["symbol"] in ("B", "T_in")]
                    t["axes"] = t_in
                    t["logical_layout"] = ["B", "T_in"]
                    t["physical_layout"] = ["B", "T_in"]
        g = self._tamper(mutate)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("seed", msg)
        self.assertIn("rank-1", msg)

    def test_tamper_second_projection_independent_main_x(self):
        """注入第二投影路径（引用完全合法：新节点/新张量/新边自成体系），
        产生第二份独立 main_x——具名拒绝而非悬空引用兜底。"""
        def mutate(doc):
            doc["tensors"].append({
                "stable_id": "tensor:entry:main_x2", "name": "entry:main_x2",
                "dtype": "bf16",
                "axes": [a for a in doc["named_axes"]
                         if a["symbol"] in ("B", "T", "H")],
                "logical_layout": ["B", "T", "H"],
                "physical_layout": ["B", "T", "H"],
                "global_rank": None, "local_rank": None})
            doc["nodes"].append({
                "stable_id": "entry.main_proj2", "op_type": "linear",
                "granularity": "parameter_op",
                "input_ports": [
                    {"name": "x", "direction": "input",
                     "tensor_id": "tensor:entry:main_hidden"},
                    {"name": "w", "direction": "input",
                     "tensor_id": "tensor:entry:main_proj_w"}],
                "output_ports": [
                    {"name": "y", "direction": "output",
                     "tensor_id": "tensor:entry:main_x2"}],
                "evidence_refs": []})
            doc["nodes"].append({
                "stable_id": "entry.stage_feed_1x", "op_type": "stage_dispatch",
                "granularity": "boundary_op",
                "input_ports": [
                    {"name": "x", "direction": "input",
                     "tensor_id": "tensor:entry:main_x2"}],
                "output_ports": [
                    {"name": "main_x_out", "direction": "output",
                     "tensor_id": "tensor:entry:main_x2"}],
                "evidence_refs": []})
            doc["edges"].append({
                "stable_id": "e_ep_cat_proj2",
                "src_node_id": "entry.readout_concat", "src_port": "y",
                "dst_node_id": "entry.main_proj2", "dst_port": "x",
                "tensor_id": "tensor:entry:main_hidden", "kind": "data",
                "guard_ref": None, "state_transition_ref": None,
                "evidence_refs": []})
            # stage_feed_1 改吃第二投影产物
            for e in doc["edges"]:
                if e["stable_id"] == "e_ep_norm_feed1":
                    e["dst_node_id"] = "entry.stage_feed_1x"
                    e["dst_port"] = "x"
                    e["tensor_id"] = "tensor:entry:main_x2"
        g = self._tamper(mutate)
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        # 闭世界核对直接点名注入的伪造实体（main_proj2/stage_feed_1x/
        # main_x2）为"必要链之外"——比旧"分发节点不唯一"更精准的具名拒绝
        self.assertTrue(
            ("main_proj2" in msg) or ("stage_feed_1x" in msg)
            or ("main_x2" in msg) or ("stage 分发节点" in msg)
            or ("main_x" in msg),
            f"拒绝信息应点名注入的第二投影实体: {msg}")

    def test_tamper_remove_distribution_edge(self):
        """删除 stage2 的 main_x 分发边（引用合法的删边）——分发缺失
        具名拒绝（stage_dispatch 节点缺 x 端口入边）。"""
        def mutate(doc):
            doc["edges"] = [e for e in doc["edges"]
                            if e["stable_id"] != "e_ep_norm_feed2"]
        g = self._tamper(mutate)
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        self.assertIn("stage_feed_2", str(cm.exception))

    def test_tamper_prefill_entry_guard_disabled(self):
        """prefill 入口停用：给首入口边挂 g_decode（引用合法——guard
        定义也一并注入）——两相位无条件被破坏具名拒绝。"""
        def mutate(doc):
            doc["guards"].append({
                "stable_id": "g_decode", "predicate_expr": "start_pos > 0",
                "evidence_refs": []})
            for e in doc["edges"]:
                if e["stable_id"] == "e_ep_cat_proj":
                    e["guard_ref"] = "g_decode"
        g = self._tamper(mutate)
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertIn("guard", msg)
        self.assertIn("e_ep_cat_proj", msg)

    def test_tamper_first_col_write_port_swap(self):
        """首列写端口颠倒：节点端口与边张量三方一致地交换（dst=seed /
        src=noise，引用完全合法）——forward_embed 854-855 语义被改
        具名拒绝。"""
        def mutate(doc):
            noise_id = "tensor:entry:draft_noise"
            seed_id = "tensor:entry:seed_ids"
            for n in doc["nodes"]:
                if n["stable_id"] == "entry.first_col_write":
                    for p in n["input_ports"]:
                        if p["name"] == "dst":
                            p["tensor_id"] = seed_id
                        elif p["name"] == "src":
                            p["tensor_id"] = noise_id
            for e in doc["edges"]:
                if e["stable_id"] == "e_ep_noise_firstcol":
                    e["dst_port"] = "src"
                elif e["stable_id"] == "e_ep_seed_firstcol":
                    e["dst_port"] = "dst"
        g = self._tamper(mutate)
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        self.assertIn("first_col_write", str(cm.exception))

    def test_tamper_draft_h_to_stage1(self):
        """草稿 h 误分发 stage1：新增 feed_1.h 入边（张量三方一致、引用
        合法）——与源码不符（stage1 h_in 来自 stage0 h_out）具名拒绝。"""
        def mutate(doc):
            for n in doc["nodes"]:
                if n["stable_id"] == "entry.stage_feed_1":
                    n["input_ports"].append(
                        {"name": "h", "direction": "input",
                         "tensor_id": "tensor:entry:entry_h"})
            doc["edges"].append({
                "stable_id": "e_ep_hcexp_feed1",
                "src_node_id": "entry.hc_expand", "src_port": "y",
                "dst_node_id": "entry.stage_feed_1", "dst_port": "h",
                "tensor_id": "tensor:entry:entry_h", "kind": "data",
                "guard_ref": None, "state_transition_ref": None,
                "evidence_refs": []})
        g = self._tamper(mutate)
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertTrue(("stage_feed_0" in msg) or ("h" in msg),
                        f"拒绝信息应点名草稿 h 消费: {msg}")

    def test_tamper_t_draft_axis_mix(self):
        """T_main 与 draft K 混用：main_x 轴改 (B,N_draft,H)（axes 列表
        同步替换，引用合法通过反序列化）——verify 层具名拒绝。"""
        def mutate(doc):
            for t in doc["tensors"]:
                if t["name"] == "entry:main_x":
                    draft_axes = [a for a in doc["named_axes"]
                                  if a["symbol"] in ("B", "N_draft", "H")]
                    t["axes"] = draft_axes
                    t["logical_layout"] = ["B", "N_draft", "H"]
                    t["physical_layout"] = ["B", "N_draft", "H"]
        g = self._tamper(mutate)
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        self.assertTrue(("main_x" in msg) or ("N_draft" in msg),
                        f"拒绝信息应点名轴混用: {msg}")


class TestEntryBindings(unittest.TestCase):
    """绑定规范 JSON：canonical 往返、确定性、读回全量核对、篡改拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.text = serialize_entry_bindings(cls.bundle.bindings)
        cls.values = cls.bundle.values
        cls.doc = json.loads(graph_to_json(cls.bundle.graph))

    def test_schema(self):
        doc = json.loads(self.text)
        self.assertEqual(doc["schema"], ENTRY_SCHEMA)
        self.assertEqual(doc["template_id"], ENTRY_TEMPLATE_ID)
        self.assertEqual(doc["n_stages"], 3)
        self.assertEqual(doc["n_main_layers"], 43)

    def test_canonical_roundtrip(self):
        b = deserialize_entry_bindings(self.text, None, ROOT)
        self.assertEqual(serialize_entry_bindings(b), self.text)

    def test_deterministic(self):
        t2 = serialize_entry_bindings(build().bindings)
        self.assertEqual(self.text, t2)

    def test_stage_ports_role(self):
        doc = json.loads(self.text)
        ports = doc["stage_ports"]
        self.assertEqual([p["stage_id"] for p in ports], [0, 1, 2])
        self.assertEqual([p["param_path"] for p in ports],
                         ["mtp.0", "mtp.1", "mtp.2"])
        for p in ports:
            self.assertTrue(p["main_x_in"])
        self.assertTrue(ports[0]["draft_h_in"])
        self.assertFalse(ports[1]["draft_h_in"])
        self.assertFalse(ports[2]["draft_h_in"])
        self.assertEqual(ports[0]["draft_h_src"], "mtp.0.forward_embed")
        self.assertEqual(ports[1]["draft_h_src"], "mtp.0.h_out")
        self.assertEqual(ports[2]["draft_h_src"], "mtp.1.h_out")

    def test_call_premises_keys_and_unknown_boundary(self):
        doc = json.loads(self.text)
        self.assertEqual(set(doc["call_premises"].keys()), set(PREMISE_KEYS))
        caller = doc["call_premises"]["forward_spec_caller"]
        self.assertIn("generate.py", caller)
        self.assertIn("不声称", caller)
        self.assertIn("model.py:39",
                      doc["call_premises"]["temperature_source"])
        self.assertIn("128799",
                      doc["call_premises"]["noise_token_id_source"])
        # seed rank-1 前提（922-924 折合 + x[:,-1] → sample [B]，855 无广播）
        width = doc["call_premises"]["input_ids_width"]
        self.assertIn("rank-1 [B]", width)
        self.assertIn("无广播", width)
        self.assertNotIn("[B,1]", width)
        # 事实边界：ModelValues 有 dspark_block_size（rules 178/260），
        # 真正缺的字段只有 noise_token_id/temperature——不把已有字段
        # 误称为缺失
        self.assertIn("ModelValues.dspark_block_size",
                      doc["draft_tokens"]["block_size_source"])
        # temperature：来源记录不冒充字段、不属于首入口
        self.assertIsNone(doc["temperature"]["run_config_override"])
        self.assertFalse(doc["temperature"]["in_entry"])

    def test_tamper_stage0_role_misbind(self):
        """stage0 角色错绑：main_x 生产者改 mtp.1（引用合法）——具名
        拒绝。"""
        doc = json.loads(self.text)
        doc["main_x"]["producer"] = "mtp.1.forward_embed"
        doc["main_x"]["param_path"] = "mtp.1"
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_entry_bindings(text, None, ROOT)
        msg = str(cm.exception)
        self.assertIn("mtp.0", msg)

    def test_tamper_shared_embed_copied(self):
        """共享 embed 复制身份：copied=true / 改 transformer_object——
        具名拒绝。"""
        doc = json.loads(self.text)
        doc["shared_embed"]["copied"] = True
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_entry_bindings(text, None, ROOT)
        self.assertIn("copied", str(cm.exception))

    def test_tamper_shared_embed_paths(self):
        """共享 embed 实例路径改名（mtp.0.embed → embed.shared）——与
        903 逐实例绑定不符具名拒绝。"""
        doc = json.loads(self.text)
        doc["shared_embed"]["instance_paths"] = ["embed.shared"] * 3
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_entry_bindings(text, None, ROOT)
        self.assertIn("instance_paths", str(cm.exception))

    def test_tamper_noise_token_id(self):
        """噪声 token 伪造（128799 → 0）——与固定 run config 原文证据
        不一致具名拒绝。"""
        doc = json.loads(self.text)
        doc["draft_tokens"]["noise_token_id"] = 0
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_entry_bindings(text, None, ROOT)
        self.assertIn("noise_token_id", str(cm.exception))

    def test_tamper_temperature_source(self):
        """temperature 来源伪造：声称 run config 覆盖 / in_entry=true——
        具名拒绝。"""
        doc = json.loads(self.text)
        doc["temperature"]["run_config_override"] = 0.7
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_entry_bindings(text, None, ROOT)
        self.assertIn("run_config_override", str(cm.exception))
        doc = json.loads(self.text)
        doc["temperature"]["in_entry"] = True
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_entry_bindings(text, None, ROOT)
        self.assertIn("in_entry", str(cm.exception))

    def test_tamper_premises_keys(self):
        """调用前提键集篡改：删 forward_spec_caller 键——未知边界声明
        缺失具名拒绝。"""
        doc = json.loads(self.text)
        del doc["call_premises"]["forward_spec_caller"]
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_entry_bindings(text, None, ROOT)
        self.assertIn("call_premises", str(cm.exception))

    def test_tamper_spec_rewrite_caller_claim(self):
        """调用前提声明改写：声称 generate.py 实际调用 forward_spec——
        具名拒绝（unknown 边界不可翻转）。"""
        doc = json.loads(self.text)
        doc["call_premises"]["forward_spec_caller"] = \
            "generate.py 实际调用 forward_spec（已核实）"
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with self.assertRaises(ExtractionError) as cm:
            deserialize_entry_bindings(text, None, ROOT)
        self.assertIn("forward_spec_caller", str(cm.exception))

    def test_tamper_schema(self):
        with self.assertRaises(ExtractionError):
            deserialize_entry_bindings("{\"schema\": \"other/v1\"}", None,
                                       ROOT)
        with self.assertRaises(ExtractionError):
            deserialize_entry_bindings("not json", None, ROOT)

    def test_stage0_misbind_graph_level(self):
        """stage0 角色错绑（图级）：注入第二 rms_norm 生产者直接产
        main_x 并喂 feed_2（节点/张量/边自成体系、引用合法）——闭世界
        核对具名拒绝（点名伪造生产者 entry.main_x_forge）。"""
        doc = copy.deepcopy(self.doc)
        # 注入一个新参数节点直接产 main_x（第二生产者）
        doc["nodes"].append({
            "stable_id": "entry.main_x_forge", "op_type": "rms_norm",
            "granularity": "parameter_op",
            "input_ports": [{"name": "x", "direction": "input",
                             "tensor_id": "tensor:entry:main_proj_y"}],
            "output_ports": [{"name": "y", "direction": "output",
                              "tensor_id": "tensor:entry:main_x"}],
            "evidence_refs": []})
        doc["edges"].append({
            "stable_id": "e_ep_forge_feed2",
            "src_node_id": "entry.main_x_forge", "src_port": "y",
            "dst_node_id": "entry.stage_feed_2", "dst_port": "x",
            "tensor_id": "tensor:entry:main_x", "kind": "data",
            "guard_ref": None, "state_transition_ref": None,
            "evidence_refs": []})
        text = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        g = graph_from_json(text)
        self.assertEqual(validate_graph_references(g), ())
        with self.assertRaises(ExtractionError) as cm:
            verify_entry_graph(g, self.values)
        msg = str(cm.exception)
        # 闭世界核对点名伪造生产者节点/多余边（stage0 角色错绑图级负例）
        self.assertTrue(
            ("main_x_forge" in msg) or ("e_ep_forge_feed2" in msg)
            or ("main_norm" in msg),
            f"拒绝信息应点名第二 main_x 生产者: {msg}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
