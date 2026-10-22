"""DSpark 公共 stage 与三实例绑定测试（工作包 B · 步骤 B2/B2 补修 01）：
公共 stage 图（真实 h 输入消费 + Block 双 hc 骨架 + 归一化因子链 + 真实
DSparkAttention 子图 fold 边界 + TopK MoE 路由因果）、全边三相位求值
（prefill 短/长/decode 逐边、None=真，断言 prefill 无双 mHC/FFN 消费链、
decode 从 h 输入到 h 输出完整可达）、fold 成员当前图真实性、参数/状态
展开绑定表（模板→实例、shape/dtype/层号/唯一性）与定向篡改拒绝、三实
例绑定与首末角色、跨 stage main_x/h 链路图（显式 phase/guard）、坐标值
非存储 view、公开 JSON 往返与确定性。
只覆盖 B2 范围；first/last 完整内部图、Markov/Confidence、全量集成
pending，本测试不声称覆盖。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.dspark_stage import (
    BINDING_SCHEMA,
    CHAIN_SCHEMA,
    EXPANSION_SCHEMA,
    EXPR_MAIN_PROJ_IN,
    STAGE_TEMPLATE_ID,
    build_chain_map,
    build_dspark_stage_template,
    build_template_expansion,
    chain_map_from_json,
    chain_map_to_json,
    deserialize_bindings,
    deserialize_expansion,
    derive_stage_bindings,
    serialize_bindings,
    serialize_expansion,
)
from structure_reader.extract.deepseek_v4_reference import ExtractionError
from structure_reader.graph_document import graph_from_json, graph_to_json
from structure_reader.model_registry import find_workspace_root
from structure_reader.scenario_evaluator import evaluate_int, evaluate_tri

ROOT = find_workspace_root()


def build():
    return build_dspark_stage_template(ROOT)


def _node_map(graph):
    return {n.stable_id: n for n in graph.nodes}


def _edges_by_id(graph):
    return {e.stable_id: e for e in graph.edges}


def _tensor_map(graph):
    return {t.name: t for t in graph.tensors}


DECODE_ENV = {"start_pos": 7, "T": 1, "P": 7, "W": 128, "N_draft": 5}
PREFILL_SHORT_ENV = {"start_pos": 0, "T": 3, "P": 0, "W": 128, "N_draft": 5}
PREFILL_LONG_ENV = {"start_pos": 0, "T": 200, "P": 0, "W": 128, "N_draft": 5}


def _active_edges(doc, preds, env):
    """全边相位求值（None=无条件真）：返回激活边集合。逐边求值断言可
    求值性（不允许 unknown），不跳过任何边。"""
    active = set()
    for e in doc["edges"]:
        ref = e["guard_ref"]
        if ref is None:
            active.add(e["stable_id"])
            continue
        r = evaluate_tri(preds[ref], env)
        self_known = r.is_known
        if not self_known:
            raise AssertionError(f"{e['stable_id']}: guard {ref!r} 不可求值")
        if r.value:
            active.add(e["stable_id"])
    return active


def _reachability(adj, start):
    seen = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for n in frontier:
            for m in adj.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    nxt.append(m)
        frontier = nxt
    return seen


class TestStageGraph(unittest.TestCase):
    """公共 stage 图结构：真实 h 输入、Block 骨架、Attention 子图边界、
    TopK MoE 路由因果。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = _node_map(cls.graph)
        cls.edges = _edges_by_id(cls.graph)

    def test_template_id(self):
        self.assertEqual(self.graph.template_id, STAGE_TEMPLATE_ID)

    def test_zero_reference_issues(self):
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_canonical_json_roundtrip(self):
        text = graph_to_json(self.graph)
        rebuilt = graph_from_json(text)
        self.assertEqual(graph_to_json(rebuilt), text)

    def test_rebuild_deterministic(self):
        # 重复构建 canonical JSON 逐字节一致（无隐藏状态）
        text1 = graph_to_json(build().graph)
        text2 = graph_to_json(build().graph)
        self.assertEqual(text1, text2)

    def test_h_input_consumed_by_first_hc_pre(self):
        # 真实输入消费：stage.x_hc_in_src → stage.hc_pre_attn.flatten.x
        # （decode，Block.forward 695-696 residual/hc_pre）；prefill 走
        # Block 级 passthrough——输入不再悬空
        e = self.edges["e_st_hcin_hcpre"]
        self.assertEqual(e.src_node_id, "stage.x_hc_in_src")
        self.assertEqual(e.dst_node_id, "stage.hc_pre_attn.flatten")
        self.assertEqual(e.dst_port, "x")
        self.assertEqual(e.guard_ref, "g_decode")
        flatten = self.nodes["stage.hc_pre_attn.flatten"]
        in_ports = {p.name for p in flatten.input_ports}
        self.assertIn("x", in_ports)

    def test_hc_pre_norm_factor_chain(self):
        # model.py:683-685：flatten → rsqrt 因子 → linear(x, hc_fn)*factor
        # → sinkhorn——归一化因子是真实因果步骤，非只 linear→sinkhorn
        for phase, factor_role in (("attn", "attn_norm_factor"),
                                   ("ffn", "ffn_norm_factor")):
            pre = f"stage.hc_pre_{phase}"
            factor = self.nodes[f"{pre}.factor"]
            self.assertEqual(factor.op_type, "rms_factor", pre)
            mix = self.nodes[f"{pre}.mix"]
            in_ports = {p.name: p.tensor_id for p in mix.input_ports}
            self.assertIn("factor", in_ports, pre)
            self.assertEqual(
                in_ports["factor"], f"tensor:stage:{factor_role}", pre)
            self.assertIn("w", in_ports, pre)
            # mix→sinkhorn 消费归一化后的 mixes
            e_mix = self.edges[f"e_st_hcpre_{phase[0]}_mix"]
            self.assertEqual(e_mix.src_node_id, f"{pre}.mix")
            self.assertEqual(e_mix.dst_node_id, f"{pre}.sinkhorn")
            # 因子自身从 flatten 的 x_flat 派生
            e_f = self.edges[f"e_st_hcpre_{phase[0]}_factor"]
            self.assertEqual(e_f.src_node_id, f"{pre}.flatten")
            self.assertEqual(e_f.dst_node_id, f"{pre}.factor")

    def test_block_skeleton_double_hc(self):
        # Block.forward 695-707：两次 hc_pre→norm→module→hc_post；
        # residual/post/comb 因果边完整（attention 侧 + ffn 侧）
        for phase in ("attn", "ffn"):
            pre = self.nodes[f"stage.hc_pre_{phase}.sinkhorn"]
            self.assertEqual(pre.op_type, "sinkhorn_split")
            post = self.nodes[f"stage.hc_post_{phase}.combine"]
            self.assertEqual(post.op_type, "hc_combine")
            in_ports = {p.name: p.tensor_id
                        for p in post.input_ports}
            self.assertIn("post", in_ports)
            self.assertIn("comb", in_ports)
            self.assertIn("residual", in_ports)
            self.assertIn("x", in_ports)
        # residual 边：hc_pre flatten residual → hc_post combine residual
        for phase, tensor in (("a", "x_hc_in"), ("f", "x_attn_post")):
            eid = f"e_st_hcpost_{phase}_residual"
            e = self.edges[eid]
            self.assertEqual(e.tensor_id, f"tensor:stage:{tensor}", eid)
        # post/comb 从 sinkhorn 到 combine 的边真实存在
        self.assertEqual(
            self.edges["e_st_hcpost_a_post"].src_node_id,
            "stage.hc_pre_attn.sinkhorn")
        self.assertEqual(
            self.edges["e_st_hcpost_f_comb"].dst_node_id,
            "stage.hc_post_ffn.combine")

    def test_hc_and_norm_params_consumed(self):
        # hc_fn/scale/base、attn_norm/ffn_norm 参数节点真实存在且被算子
        # 参数端口消费（不能仅通过引用 ev_hc_params 代替）
        param_nodes = {
            "stage.hc_attn_fn_w": ("e_st_hcpre_a_fn_w",
                                   ("stage.hc_pre_attn.mix", "w")),
            "stage.hc_attn_scale_w": ("e_st_hcpre_a_scale_w",
                                      ("stage.hc_pre_attn.sinkhorn",
                                       "scale")),
            "stage.hc_attn_base_w": ("e_st_hcpre_a_base_w",
                                     ("stage.hc_pre_attn.sinkhorn", "base")),
            "stage.hc_ffn_fn_w": ("e_st_hcpre_f_fn_w",
                                  ("stage.hc_pre_ffn.mix", "w")),
            "stage.hc_ffn_scale_w": ("e_st_hcpre_f_scale_w",
                                     ("stage.hc_pre_ffn.sinkhorn", "scale")),
            "stage.hc_ffn_base_w": ("e_st_hcpre_f_base_w",
                                    ("stage.hc_pre_ffn.sinkhorn", "base")),
            "stage.attn_norm_w": ("e_st_attnnorm_w",
                                  ("stage.attn_norm", "w")),
            "stage.ffn_norm_w": ("e_st_ffnnorm_w", ("stage.ffn_norm", "w")),
        }
        tm = _tensor_map(self.graph)
        for sid, (eid, (dst, port)) in param_nodes.items():
            self.assertIn(sid, self.nodes, sid)
            e = self.edges[eid]
            self.assertEqual((e.src_node_id, e.dst_node_id, e.dst_port),
                             (sid, dst, port), eid)
        # 参数张量形状/dtype（ev_st_hc_params/ev_st_norm_params 锚定）
        n_mix, n_hch = 24, 16384  # (2+4)*4, 4*4096
        checks = [
            ("stage:hc_attn_fn_w", ["N_mix", "N_hcH"], "fp32"),
            ("stage:hc_attn_scale_w", ["HC3"], "fp32"),
            ("stage:hc_attn_base_w", ["N_mix"], "fp32"),
            ("stage:hc_ffn_fn_w", ["N_mix", "N_hcH"], "fp32"),
            ("stage:attn_norm_w", ["H"], "fp32"),
            ("stage:ffn_norm_w", ["H"], "fp32"),
        ]
        for name, syms, dtype in checks:
            t = tm[name]
            self.assertEqual([a.symbol for a in t.axes], syms, name)
            self.assertEqual(t.dtype, dtype, name)

    def test_gate_params_consumed(self):
        # gate weight/bias 参数真实消费（model.py:562/567 → 570）
        score = self.nodes["stage.moe.gate.score"]
        in_ports = {p.name: p.tensor_id for p in score.input_ports}
        self.assertEqual(in_ports.get("w"), "tensor:stage:gate_weight_w")
        bias = self.nodes["stage.moe.gate.bias"]
        in_ports = {p.name: p.tensor_id for p in bias.input_ports}
        self.assertEqual(in_ports.get("b"), "tensor:stage:gate_bias_w")
        tm = _tensor_map(self.graph)
        self.assertEqual([a.symbol for a in tm["stage:gate_weight_w"].axes],
                         ["E", "H"])
        self.assertEqual(tm["stage:gate_bias_w"].dtype, "fp32")
        self.assertEqual([a.symbol for a in tm["stage:gate_bias_w"].axes],
                         ["E"])

    def test_gate_weight_storage_dtype_bf16(self):
        # gate.weight 存储 dtype = 参考入口 torch 默认 dtype bf16（B2 补修
        # 02）：model.py:562 torch.empty 无显式 dtype → torch 默认 dtype；
        # 950 set_default_dtype(bfloat16)；884 default_dtype 是模块变量仅
        # 供自定义 Linear（136 dtype=dtype or default_dtype）消费，Gate
        # 不经该机制；570 weight.float() 是 fp32 求值非存储证据——存储/
        # 计算 dtype 分离。真实量化 Linear 权重（wq/wkv/wo_b fp8、wo_a
        # bf16、RMSNorm fp32）保持源码语义不改动。
        tm = _tensor_map(self.graph)
        self.assertEqual(tm["stage:gate_weight_w"].dtype, "bf16")
        specs = {s.spec_id: s for s in self.bundle.expansion.param_specs}
        gate = specs["stage.gate_weight_w"]
        self.assertEqual(gate.dtype, "bf16")
        self.assertEqual(gate.shape, (256, 4096))
        # 证据同时覆盖：构造（551-567）、参考入口默认 dtype（949-950）、
        # Linear 默认机制（129-136）、模块变量 default_dtype（884）
        self.assertIn("ev_st_gate_params", gate.evidence)
        self.assertIn("ev_st_entry_default_dtype", gate.evidence)
        self.assertIn("ev_st_linear_dtype_mech", gate.evidence)
        self.assertIn("ev_st_module_default_dtype", gate.evidence)
        # 说明不把 fp8 存储或 fp32 求值错当存储 dtype；量化 Linear 权重
        # 不被本修正波及（fp8 机制证据独立锚定）
        self.assertIn("bf16", gate.note)
        wqa = specs["stage.wq_a_w"]
        self.assertEqual(wqa.dtype, "fp8_e4m3fn")
        woa = specs["stage.wo_a_w"]
        self.assertEqual(woa.dtype, "bf16")
        norm = specs["stage.attn_norm_w"]
        self.assertEqual(norm.dtype, "fp32")

    def test_moe_topk_structure(self):
        # TopK MoE：score→bias→topk→gather（偏置不进 gather）→dispatch→
        # 专家模板→shared expert→accumulate；无 hash 查表（stage 层
        # 43/44/45 ≥ n_hash_layers=3）
        chain = [("stage.moe.gate.score", "moe_gate_score"),
                 ("stage.moe.gate.bias", "add_bias"),
                 ("stage.moe.gate.topk", "topk_select"),
                 ("stage.moe.gate.gather", "gather_weights"),
                 ("stage.moe.dispatch", "token_dispatch"),
                 ("stage.moe.expert_template", "swiglu_expert"),
                 ("stage.moe.shared_experts", "swiglu_expert"),
                 ("stage.moe.out", "accumulate_add")]
        for sid, op in chain:
            self.assertEqual(self.nodes[sid].op_type, op, sid)
        # gather 的 original 输入直接来自 score（original_scores，非 biased）
        self.assertEqual(self.edges["e_st_orig_gather"].src_node_id,
                         "stage.moe.gate.score")
        self.assertEqual(self.edges["e_st_orig_gather"].src_port, "scores")
        # dispatch 消费 topk indices
        self.assertEqual(self.edges["e_st_idx_dispatch"].src_node_id,
                         "stage.moe.gate.topk")
        # 无 CSA 压缩路径节点
        for n in self.graph.nodes:
            self.assertNotIn("compressor", n.stable_id, n.stable_id)

    def test_moe_route_causal(self):
        # 路由因果（model.py:639-645）：dispatch 产出 counts（空专家跳过
        # 门控，control 边）与 routing=(idx,top)（token 子集+权重槽位，
        # select 边）——counts 不再无消费，expert 二者真实消费
        dispatch = self.nodes["stage.moe.dispatch"]
        out_ports = {p.name: p.tensor_id for p in dispatch.output_ports}
        self.assertEqual(out_ports.get("counts"), "tensor:stage:n_e")
        self.assertEqual(out_ports.get("routing"),
                         "tensor:stage:route_select")
        expert = self.nodes["stage.moe.expert_template"]
        in_ports = {p.name: p.tensor_id for p in expert.input_ports}
        self.assertEqual(in_ports.get("gate_counts"), "tensor:stage:n_e")
        self.assertEqual(in_ports.get("routing"),
                         "tensor:stage:route_select")
        e_sel = self.edges["e_st_dispatch_select"]
        self.assertEqual(e_sel.kind, "select")
        self.assertEqual((e_sel.src_node_id, e_sel.dst_node_id),
                         ("stage.moe.dispatch", "stage.moe.expert_template"))
        e_gate = self.edges["e_st_dispatch_gate"]
        self.assertEqual(e_gate.kind, "control")
        # N_e 数据相关未知轴：无编造数值
        symbols = {a.symbol: a for a in self.graph.named_axes}
        self.assertIsNone(symbols["N_e"].value)
        self.assertTrue(symbols["N_e"].unknown_reason)

    def test_no_hash_gate_branch(self):
        # hash 查表（tid2eid）不出现在 stage 图（层 43-45 全 TopK）
        for n in self.graph.nodes:
            self.assertNotIn("hash", n.op_type.lower(), n.stable_id)
            self.assertNotIn("tid2eid", n.stable_id, n.stable_id)

    def test_main_x_single_input_axis(self):
        # main_x [B,T,H]（同一张量分发三 stage，模型事实 B/T 与草稿
        # N_draft 不同长度）
        t = _tensor_map(self.graph)["stage:main_x"]
        self.assertEqual([a.symbol for a in t.axes], ["B", "T", "H"])

    def test_decode_input_hc_stream(self):
        # decode 输入 HC 流 [B,N_draft,N_hc,H]（forward_spec 的 h）
        t = _tensor_map(self.graph)["stage:x_hc_in"]
        self.assertEqual([a.symbol for a in t.axes],
                         ["B", "N_draft", "N_hc", "H"])


class TestStagePhaseGuards(unittest.TestCase):
    """prefill 屏蔽 / decode 全链：对实际序列化图逐边求值（None=真，
    不跳过任何边）+ 可达性断言 + 定向篡改负例。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = _node_map(cls.graph)
        cls.edges = _edges_by_id(cls.graph)
        cls.doc = json.loads(graph_to_json(cls.graph))
        cls.preds = {g["stable_id"]: g["predicate_expr"]
                     for g in cls.doc["guards"]}

    def _active(self, env):
        return _active_edges(self.doc, self.preds, env)

    def _adjacency(self, active, doc):
        adj = {}
        for e in doc["edges"]:
            if e["stable_id"] in active:
                adj.setdefault(e["src_node_id"], []).append(e["dst_node_id"])
        return adj

    def test_guard_predicates(self):
        self.assertEqual(self.preds["g_prefill"], "start_pos == 0")
        self.assertEqual(self.preds["g_decode"], "start_pos > 0")
        self.assertEqual(self.preds["g_dspark_prefill_inwin"],
                         "start_pos == 0 and T <= W")
        self.assertEqual(self.preds["g_window_wrap"],
                         "start_pos == 0 and T > W")

    def test_prefill_only_passthrough(self):
        # prefill：Block 级 passthrough（DSparkBlock.forward 849）消费边
        # 挂 g_prefill——prefill 只有 cache 构建与 passthrough
        e = self.edges["e_st_prefill_in"]
        self.assertEqual(e.guard_ref, "g_prefill")
        self.assertEqual(e.src_node_id, "stage.x_hc_in_src")
        self.assertEqual(e.dst_node_id, "stage.prefill_passthrough")
        node = self.nodes["stage.prefill_passthrough"]
        self.assertEqual(node.op_type, "passthrough")
        self.assertIn("ev_ds_block_prefill", node.evidence_refs)

    def test_decode_skeleton_edges_guarded(self):
        # 全部 decode 专用骨架运行时消费边挂可求值 g_decode（含入边、
        # post/comb/residual、FFN/MoE）——不从图删边实现屏蔽
        skeleton = (
            "e_st_hcin_hcpre", "e_st_hcpre_a_flat", "e_st_hcpre_a_mix",
            "e_st_hcpre_a_pre", "e_st_hcpre_a_x", "e_st_hcpre_a_out",
            "e_st_norm_xsrc", "e_st_attn_out_post", "e_st_hcpost_a_post",
            "e_st_hcpost_a_comb", "e_st_hcpost_a_residual",
            "e_st_hcpost_a_out", "e_st_hcpre_f_flat", "e_st_hcpre_f_mix",
            "e_st_hcpre_f_pre", "e_st_hcpre_f_x", "e_st_hcpre_f_out",
            "e_st_ffnnorm_gate", "e_st_moe_post", "e_st_hcpost_f_post",
            "e_st_hcpost_f_comb", "e_st_hcpost_f_residual",
            "e_st_hcpre_a_fn_w", "e_st_hcpre_a_scale_w",
            "e_st_hcpre_a_base_w", "e_st_hcpre_f_fn_w",
            "e_st_hcpre_f_scale_w", "e_st_hcpre_f_base_w",
            "e_st_attnnorm_w", "e_st_ffnnorm_w", "e_st_gatew_score",
            "e_st_gateb_bias", "e_st_score_bias", "e_st_bias_topk",
            "e_st_orig_gather", "e_st_idx_gather", "e_st_idx_dispatch",
            "e_st_gate_expert", "e_st_w_expert", "e_st_dispatch_select",
            "e_st_dispatch_gate", "e_st_gate_shared", "e_st_expert_acc",
            "e_st_shared_acc",
        )
        for eid in skeleton:
            self.assertEqual(self.edges[eid].guard_ref, "g_decode", eid)

    def test_phase_free_edges(self):
        # 两相位共享：main_x→共享 mainKV（e_st_mainx_xsrc）；prefill
        # passthrough（g_prefill）——除此之外骨架边全部有相位 guard
        self.assertIsNone(self.edges["e_st_mainx_xsrc"].guard_ref)
        self.assertEqual(self.edges["e_st_prefill_in"].guard_ref,
                         "g_prefill")

    def test_all_edges_evaluated_three_phases(self):
        # 全边三相位求值（None=真、逐边可求值性）：prefill 短/长唯一差异
        # 是 cache 写分支（T<=W 顺序 / T>W 回绕，两条真实分支）；其余
        # 激活集一致。prefill 无任何双 mHC/FFN/MoE 消费边。
        active_ps = self._active(PREFILL_SHORT_ENV)
        active_pl = self._active(PREFILL_LONG_ENV)
        active_d = self._active(DECODE_ENV)
        # 短/长差异只在两条 prefill cache 写分支之间互换
        self.assertEqual(active_ps - active_pl, {"e_ds_cachew_inwin"})
        self.assertEqual(active_pl - active_ps, {"e_ds_cachew_wrap"})
        # decode 包含 prefill 除 prefill-only 分支外的全部激活边（骨架+
        # 草稿链激活；passthrough 消费边与 cache 写分支相位互斥属真实
        # 分支结构）
        prefill_only_branches = {"e_ds_cachew_inwin", "e_ds_cachew_wrap",
                                 "e_st_prefill_in"}
        self.assertTrue(active_d >= active_ps - prefill_only_branches)
        # prefill 激活的 stage 骨架边只允许 main_x→mainKV（两相位共享）
        # 与 prefill passthrough 消费边（g_prefill）；无任何双 mHC/FFN/
        # MoE 消费边
        for eid in active_ps:
            if eid.startswith("e_st_"):
                self.assertIn(eid, ("e_st_mainx_xsrc", "e_st_prefill_in"),
                              eid)
        # prefill 激活的 dspark.* 边只允许 main KV 链与 cache 写分支
        allowed_ds = {
            "e_ds_main_in_wkv", "e_ds_main_wkv", "e_ds_main_norm_rope",
            "e_ds_main_rope_quant", "e_ds_wkv_w_main", "e_ds_kvnorm_w_main",
            "e_ds_cachew_inwin", "e_ds_cachew_wrap",
        }
        for eid in active_ps:
            if eid.startswith("e_ds_"):
                self.assertIn(eid, allowed_ds, eid)
        # decode 全链关键边激活
        for must in ("e_st_hcin_hcpre", "e_st_norm_xsrc", "e_ds_xin_wqa",
                     "e_ds_draft_wkv", "e_ds_idx_select", "e_ds_sparse_q",
                     "e_ds_flat_wob", "e_st_ffnnorm_gate", "e_st_moe_post",
                     "e_st_dispatch_select", "e_st_dispatch_gate",
                     "e_st_hcpost_f_residual"):
            self.assertIn(must, active_d, must)

    def test_prefill_reachability_no_skeleton(self):
        # prefill env 下从 h 输入出发的可达节点：只有 passthrough——
        # 无任何双 mHC/FFN/MoE 消费链（Codex 指出 81 绿未覆盖的运行时
        # 语义，逐边实算后断言）
        for env, name in ((PREFILL_SHORT_ENV, "short"),
                          (PREFILL_LONG_ENV, "long")):
            active = self._active(env)
            adj = self._adjacency(active, self.doc)
            reach = _reachability(adj, "stage.x_hc_in_src")
            self.assertEqual(
                reach, {"stage.x_hc_in_src", "stage.prefill_passthrough"},
                f"prefill {name}: h 可达集必须只有 passthrough，得到 {reach}")
            # main_x 可达 main KV 与 cache（两相位共享路径真实存在）
            reach_main = _reachability(adj, "stage.main_x_src")
            self.assertIn("dspark.cache_update", reach_main, name)

    def test_decode_reachability_full_chain(self):
        # decode env：从 h 输入到 h 输出完整路径 + main_x 到 cache 且
        # Attention 相连（稀疏注意力消费 concat 的窗口 cache 读）
        active = self._active(DECODE_ENV)
        adj = self._adjacency(active, self.doc)
        reach_h = _reachability(adj, "stage.x_hc_in_src")
        for must in ("stage.hc_pre_attn.flatten", "stage.attn_norm",
                     "dspark.sparse_attn", "dspark.wo_b",
                     "stage.hc_post_attn.combine", "stage.hc_pre_ffn.flatten",
                     "stage.moe.out", "stage.hc_post_ffn.combine"):
            self.assertIn(must, reach_h, must)
        # 终点：hc_post_ffn 输出张量（h_out）
        out_tensors = {p.tensor_id
                       for p in self.nodes["stage.hc_post_ffn.combine"
                                           ].output_ports}
        self.assertEqual(out_tensors, {"tensor:stage:x_hc_out"})
        reach_main = _reachability(adj, "stage.main_x_src")
        self.assertIn("dspark.cache_update", reach_main)
        self.assertIn("dspark.sparse_attn", reach_main)

    def test_skeleton_guard_tamper_detected(self):
        # 定向篡改负例 1：删除 h 输入消费边 → decode 可达性断裂，测试
        # 必须真实检出（不被旧断言的跳过逻辑掩盖）
        doc = json.loads(graph_to_json(self.graph))
        doc["edges"] = [e for e in doc["edges"]
                        if e["stable_id"] != "e_st_hcin_hcpre"]
        active = _active_edges(doc, self.preds, DECODE_ENV)
        adj = self._adjacency(active, doc)
        reach = _reachability(adj, "stage.x_hc_in_src")
        self.assertNotIn("stage.hc_pre_attn.flatten", reach,
                         "篡改后 h 仍可达首 hc_pre——负例未检出，测试无效")

    def test_skeleton_guard_clear_tamper_detected(self):
        # 定向篡改负例 2：清空骨架边 guard（g_decode→None，即无条件）→
        # prefill env 立即激活双 mHC 消费链，prefill 屏蔽断言必须检出
        doc = json.loads(graph_to_json(self.graph))
        cleared = []
        for e in doc["edges"]:
            if e["guard_ref"] == "g_decode" and e["stable_id"].startswith(
                    "e_st_"):
                e["guard_ref"] = None
                cleared.append(e["stable_id"])
        self.assertGreaterEqual(len(cleared), 30)
        active = _active_edges(doc, self.preds, PREFILL_LONG_ENV)
        bad = [eid for eid in active
               if eid.startswith("e_st_") and eid != "e_st_mainx_xsrc"]
        self.assertTrue(bad,
                        "清空骨架 guard 后 prefill 仍无骨架消费边——负例未"
                        "检出，prefill 屏蔽断言无效")


class TestStageFold(unittest.TestCase):
    """fold 成员 = 当前 stage _add_attn_flow 实际返回集合，逐成员/边界
    映射在当前图验证（missing 具名拒绝，不允许 continue 跳过）。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph
        cls.nodes = _node_map(cls.graph)

    def test_fold_members_exist_in_current_graph(self):
        fg = next(fg for fg in self.graph.fold_groups
                  if fg.stable_id == "fg_stage_attn")
        self.assertGreaterEqual(len(fg.member_node_ids), 30)
        for m in fg.member_node_ids:
            self.assertIn(m, self.nodes, f"fold 成员不属于当前 stage 图: {m}")
        self.assertTrue(all(m.startswith("dspark.")
                            for m in fg.member_node_ids))
        # 当前 stage 图不含 prefill return 子图（include_prefill=False，
        # prefill 由 Block 级 passthrough 承载）——B1 的两个 prefill 成员
        # 不应出现
        self.assertNotIn("dspark.prefill_return", fg.member_node_ids)
        self.assertNotIn("dspark.x_draft_hc_src", fg.member_node_ids)

    def test_boundary_mappings_members_and_ports(self):
        # 每条边界映射：内部节点是 fold 成员、在当前图真实存在、端口
        # 方向/存在正确
        fg = next(fg for fg in self.graph.fold_groups
                  if fg.stable_id == "fg_stage_attn")
        by_port = {}
        for m in fg.boundary_mappings:
            self.assertIn(m.internal_node_id, self.nodes,
                          f"映射节点不在当前图: {m.internal_node_id}")
            self.assertIn(m.internal_node_id, fg.member_node_ids,
                          f"映射节点不是 fold 成员: {m.internal_node_id}")
            n = self.nodes[m.internal_node_id]
            ports = n.output_ports if m.direction == "output" \
                else n.input_ports
            self.assertTrue(
                any(p.name == m.internal_port for p in ports),
                f"映射端口不存在: {m.internal_node_id}.{m.internal_port}")
            by_port[(m.direction, m.external_port, m.edge_kind)] = m
        # 四条边界完整
        mx = by_port[("input", "x", "data")]
        self.assertEqual(mx.internal_node_id, "dspark.x_draft_single_src")
        mm = by_port[("input", "main_x", "data")]
        self.assertEqual(mm.internal_node_id, "dspark.x_main_in_src")
        my = by_port[("output", "y", "data")]
        self.assertEqual(my.internal_node_id, "dspark.wo_b")
        mc = by_port[("output", "cache", "state")]
        self.assertEqual(mc.internal_node_id, "dspark.cache_update")
        self.assertEqual(len(self.bundle.boundary_mappings),
                         len(fg.boundary_mappings))

    def test_fold_missing_member_rejected(self):
        # 定向篡改：fold 成员/映射引用当前图不存在的节点 → 构建期核对
        # 具名拒绝（_verify_fold_members——build 与测试共用同一实现，
        # 序列化图上直接驱动）
        from structure_reader.graph_document import graph_from_json
        from structure_reader.extract.dspark_stage import (
            _verify_fold_members,
        )
        fg = next(fg for fg in self.graph.fold_groups
                  if fg.stable_id == "fg_stage_attn")
        doc = json.loads(graph_to_json(self.graph))
        doc["fold_groups"][0]["member_node_ids"] = list(
            fg.member_node_ids) + ["dspark.prefill_return",
                                   "dspark.x_draft_hc_src"]
        tampered = graph_from_json(json.dumps(doc))
        with self.assertRaises(ExtractionError) as ctx:
            _verify_fold_members(
                tampered,
                tuple(tampered.fold_groups[0].member_node_ids),
                tuple(tampered.fold_groups[0].boundary_mappings))
        self.assertIn("dspark.prefill_return", str(ctx.exception))

    def test_boundary_missing_port_rejected(self):
        # 定向篡改：边界映射指向不存在端口 → 核对具名拒绝
        from structure_reader.graph_document import graph_from_json
        from structure_reader.graph_ir import FoldBoundaryMapping
        from structure_reader.extract.dspark_stage import (
            _verify_fold_members,
        )
        fg = next(fg for fg in self.graph.fold_groups
                  if fg.stable_id == "fg_stage_attn")
        bad_mapping = FoldBoundaryMapping(
            direction="input", external_port="x", edge_kind="data",
            internal_node_id="dspark.x_draft_single_src",
            internal_port="no_such_port")
        with self.assertRaises(ExtractionError):
            _verify_fold_members(self.graph, fg.member_node_ids,
                                 (bad_mapping,))

    def test_attention_internal_wiring_same_origin_as_b1(self):
        # 成员节点与 B1 同源（同 ID 同 op；源节点 op 两图设计不同：
        # B1 graph_input / stage subgraph_input——边界喂入语义，比较时
        # 排除）；stage 图内部连线真实存在（fold 组成员携带完整连线，
        # 不是 opaque 名字）
        from structure_reader.extract.deepseek_v4_dspark import (
            build_dspark_attention_graph,
        )
        b1 = build_dspark_attention_graph(ROOT)
        b1_nodes = {n.stable_id: n.op_type for n in b1.graph.nodes}
        fg = next(fg for fg in self.graph.fold_groups
                  if fg.stable_id == "fg_stage_attn")
        for m in fg.member_node_ids:
            self.assertIn(m, b1_nodes, m)
            if m in ("dspark.x_draft_single_src", "dspark.x_main_in_src"):
                # 输入源节点：stage 以 subgraph_input 喂入（fed_sources），
                # B1 是图输入源——ID/端口/连线同源，op 设计差异
                self.assertEqual(self.nodes[m].op_type, "subgraph_input", m)
                self.assertEqual(b1_nodes[m], "graph_input", m)
            else:
                self.assertEqual(self.nodes[m].op_type, b1_nodes[m], m)
        # 关键内部链在当前 stage 图真实存在
        chain = [("dspark.o_rope_inv", "rope_partial_inverse"),
                 ("dspark.o_reshape", "reshape"),
                 ("dspark.o_einsum", "grouped_lowrank_einsum"),
                 ("dspark.o_flatten", "flatten"),
                 ("dspark.wo_b", "linear")]
        for sid, op in chain:
            self.assertEqual(self.nodes[sid].op_type, op, sid)
            self.assertIn(sid, fg.member_node_ids, sid)


class TestStageBindings(unittest.TestCase):
    """三实例绑定：角色纯逻辑、actual layer_id、独立 state、共享身份、
    读回实例身份一致性（复制/交换身份具名拒绝）。"""

    @classmethod
    def setUpClass(cls):
        cls.b3 = derive_stage_bindings(3, 43)
        cls.cm = build_chain_map(ROOT)

    def test_three_bindings(self):
        self.assertEqual(len(self.b3), 3)
        for i, b in enumerate(self.b3):
            self.assertEqual(b.stage_id, i)
            self.assertEqual(b.actual_layer_id, 43 + i)
            self.assertEqual(b.param_path, f"mtp.{i}")
            self.assertEqual(b.state_id, f"dspark:win_cache:mtp.{i}")
            self.assertEqual(b.template_id, STAGE_TEMPLATE_ID)

    def test_main_layers_not_46(self):
        # 主干仍 43 层：actual_layer_id 是 constructor 编号（n_layers+i），
        # 不算 46 个 main layers——绑定记录不声称 n_layers 改变
        values = self.cm.ratios_consumers
        self.assertEqual(values["main_range"], [0, 42])
        self.assertEqual(values["draft_range"], [43, 45])
        self.assertEqual(values["total_len"], 46)

    def test_first_middle_last_roles(self):
        b0, b1, b2 = self.b3
        self.assertTrue(b0.is_first and not b0.is_last)
        self.assertFalse(b1.is_first or b1.is_last)
        self.assertFalse(b2.is_first)
        self.assertTrue(b2.is_last)
        # 角色专有模块按 model.py:830-841 真实分支
        self.assertIn("main_proj", b0.role_modules)
        self.assertIn("main_norm", b0.role_modules)
        self.assertIn("forward_embed", b0.role_modules)
        self.assertNotIn("main_proj", b2.role_modules)
        self.assertNotIn("main_proj", b1.role_modules)
        self.assertNotIn("norm", b0.role_modules)
        self.assertNotIn("markov_head", b0.role_modules)
        self.assertNotIn("markov_head", b1.role_modules)
        for last_only in ("norm", "markov_head", "confidence_head",
                          "forward_head", "hc_head_fn"):
            self.assertIn(last_only, b2.role_modules)
        # middle：无角色专有模块（只有共享引用）
        self.assertEqual(set(b1.role_modules),
                         {"shared.embed", "shared.head"})

    def test_single_stage_first_and_last_true(self):
        # 纯角色逻辑：n_stages=1 → 同一实例 first 与 last 同时 true
        b1 = derive_stage_bindings(1, 43)
        self.assertEqual(len(b1), 1)
        self.assertTrue(b1[0].is_first)
        self.assertTrue(b1[0].is_last)
        # single 角色模块并集
        self.assertIn("main_proj", b1[0].role_modules)
        self.assertIn("markov_head", b1[0].role_modules)
        self.assertTrue(b1[0].pending_subgraphs)

    def test_shared_embed_head_not_copied(self):
        # 三个实例都只有 shared.embed/shared.head 引用（Transformer 同一
        # 对象，model.py:903-904），不是三份独立权重记录
        for b in self.b3:
            self.assertIn("shared.embed", b.role_modules)
            self.assertIn("shared.head", b.role_modules)
        for b in self.b3:
            for rm in b.role_modules:
                self.assertFalse(rm.startswith("own."), rm)

    def test_bindings_json_roundtrip(self):
        text = serialize_bindings(self.b3)
        doc = json.loads(text)
        self.assertEqual(doc["schema"], BINDING_SCHEMA)
        self.assertEqual(doc["n_stages"], 3)
        rebuilt = deserialize_bindings(text)
        self.assertEqual(serialize_bindings(rebuilt), text)
        # 篡改检出：角色与序号矛盾被拒
        bad = json.loads(text)
        bad["bindings"][1]["is_first"] = True
        with self.assertRaises(ExtractionError):
            deserialize_bindings(json.dumps(bad))
        bad2 = json.loads(text)
        bad2["n_stages"] = 5
        with self.assertRaises(ExtractionError):
            deserialize_bindings(json.dumps(bad2))
        bad3 = json.loads(text)
        bad3["bindings"][0]["role_modules"] = "not-a-list"
        with self.assertRaises(ExtractionError):
            deserialize_bindings(json.dumps(bad3))

    def test_bindings_identity_tamper_rejected(self):
        # Codex 实测篡改复现并拒绝：第 1 实例携带第 0 实例 param_path/
        # state_id 且 layer=43 → deserialize_bindings 具名拒绝，不再接受
        text = serialize_bindings(self.b3)
        bad = json.loads(text)
        bad["bindings"][1]["param_path"] = "mtp.0"
        bad["bindings"][1]["state_id"] = "dspark:win_cache:mtp.0"
        bad["bindings"][1]["actual_layer_id"] = 43
        with self.assertRaises(ExtractionError) as ctx:
            deserialize_bindings(json.dumps(bad))
        msg = str(ctx.exception)
        self.assertTrue(
            ("param_path" in msg) or ("state_id" in msg)
            or ("actual_layer_id" in msg) or ("实例" in msg),
            f"拒绝信息未指明身份不一致: {msg}")
        # 交换身份同样拒绝
        bad2 = json.loads(text)
        bad2["bindings"][0]["param_path"], bad2["bindings"][1]["param_path"] \
            = "mtp.1", "mtp.0"
        with self.assertRaises(ExtractionError):
            deserialize_bindings(json.dumps(bad2))

    def test_pending_first_last_recorded(self):
        # 角色专有模块记录为 pending，不冒称已提取首末图
        b0, _, b2 = self.b3
        self.assertTrue(any("first 入口完整图" in p
                            for p in b0.pending_subgraphs))
        self.assertTrue(any("last 出口完整图" in p
                            for p in b2.pending_subgraphs))
        self.assertEqual(self.b3[1].pending_subgraphs, ())


class TestStageExpansion(unittest.TestCase):
    """参数/状态展开绑定表：模板 spec → 实例展开，shape/dtype/层号/
    唯一性/角色规则读回复检，篡改具名拒绝。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.exp = cls.bundle.expansion

    def test_schema_and_template(self):
        self.assertEqual(self.exp.to_dict()["schema"], EXPANSION_SCHEMA)
        self.assertEqual(self.exp.template_id, STAGE_TEMPLATE_ID)
        self.assertEqual(self.exp.n_stages, 3)
        self.assertEqual(self.exp.n_main_layers, 43)

    def test_spec_shapes_dtypes_real(self):
        # 真实 shape/dtype（ModelValues 派生 + 源码锚定）；gate.weight
        # 存储 bf16（B2 补修 02：torch.empty 无显式 dtype + 参考入口
        # set_default_dtype(bfloat16)，非 fp8）
        specs = {s.spec_id: s for s in self.exp.param_specs}
        checks = [
            ("stage.hc_attn_fn_w", (24, 16384), "fp32"),
            ("stage.hc_ffn_fn_w", (24, 16384), "fp32"),
            ("stage.hc_attn_scale_w", (3,), "fp32"),
            ("stage.attn_norm_w", (4096,), "fp32"),
            ("stage.wkv_w", (512, 4096), "fp8_e4m3fn"),
            ("stage.q_norm_w", (1024,), "fp32"),
            ("stage.wo_a_w", (8 * 1024, 64 * 512 // 8), "bf16"),
            ("stage.gate_weight_w", (256, 4096), "bf16"),
            ("stage.gate_bias_w", (256,), "fp32"),
            ("stage.expert_w1_w", (2048, 4096), "fp4_e2m1fn_x2"),
            ("stage.shared_w1_w", (2048, 4096), "fp8_e4m3fn"),
            ("stage.main_proj_w", (4096, 4096 * 3), "fp8_e4m3fn"),
            ("stage.markov_w1_w", (129280, 256), "bf16"),
            ("stage.confidence_proj_w", (1, 4096 + 256), "fp32"),
        ]
        for sid, shape, dtype in checks:
            s = specs[sid]
            self.assertEqual(s.shape, shape, sid)
            self.assertEqual(s.dtype, dtype, sid)
            self.assertTrue(s.evidence, sid)
            self.assertTrue(s.note.strip(), sid)
        # 全部 spec 都有消费点/语义说明（不能只给名字）
        for s in self.exp.param_specs:
            self.assertTrue(s.note.strip(), s.spec_id)

    def test_bindings_cover_common_template(self):
        # 公共参数（双 mHC/norm/Attention/MoE）每实例独立展开
        # mtp.{i}.*；角色专有只在对应实例
        by_stage = {}
        for b in self.exp.param_bindings:
            by_stage.setdefault(b.stage_id, set()).add(b.spec_id)
        common = ("stage.hc_attn_fn_w", "stage.hc_ffn_fn_w",
                  "stage.attn_norm_w", "stage.ffn_norm_w", "stage.wkv_w",
                  "stage.gate_weight_w", "stage.gate_bias_w",
                  "stage.expert_w1_w", "stage.shared_w1_w")
        for i in range(3):
            for sid in common:
                self.assertIn(sid, by_stage[i], f"stage {i} 缺 {sid}")
        self.assertIn("stage.main_proj_w", by_stage[0])
        self.assertNotIn("stage.main_proj_w", by_stage[1])
        self.assertNotIn("stage.main_proj_w", by_stage[2])
        self.assertIn("stage.norm_w", by_stage[2])
        self.assertNotIn("stage.norm_w", by_stage[0])
        self.assertNotIn("stage.norm_w", by_stage[1])
        # instance_path 前缀与实例对应
        for b in self.exp.param_bindings:
            self.assertTrue(b.instance_path.startswith(f"mtp.{b.stage_id}."),
                            b.instance_path)
        # spec × 实例绑定总数：33 specs = 24 公共×3 + 2 first×1 + 7
        # last×1 → 81 绑定（上轮摘要误写 26 公共，本轮更正为 24）
        self.assertEqual(len(self.exp.param_specs), 33)
        common_count = sum(
            1 for s in self.exp.param_specs
            if s.spec_id not in ("stage.main_proj_w", "stage.main_norm_w")
            and s.spec_id not in {
                "stage.norm_w", "stage.hc_head_fn_w", "stage.hc_head_base_w",
                "stage.hc_head_scale_w", "stage.markov_w1_w",
                "stage.markov_w2_w", "stage.confidence_proj_w"})
        self.assertEqual(common_count, 24)
        self.assertEqual(len(self.exp.param_bindings), 81)

    def test_state_bindings_independent_identity(self):
        # cache 状态：模板 dspark:win_cache → 三实例独立 state_id、版本对
        # v_hist→v_after、transition tr_dspark_cache、layer 43+i
        self.assertEqual(len(self.exp.state_bindings), 3)
        seen = set()
        for k, sb in enumerate(self.exp.state_bindings):
            self.assertEqual(sb.stage_id, k)
            self.assertEqual(sb.state_id, f"dspark:win_cache:mtp.{k}")
            self.assertEqual(sb.template_state_id, "dspark:win_cache")
            self.assertEqual(sb.cache_tensor, "stage:win_cache")
            self.assertEqual(sb.version_labels, ("v_hist", "v_after"))
            self.assertEqual(sb.transition_id, "tr_dspark_cache")
            self.assertEqual(sb.layer_id, 43 + k)
            self.assertNotIn(sb.state_id, seen)
            seen.add(sb.state_id)

    def test_shared_refs_not_expanded(self):
        # 共享 embed/head 只引用 Transformer 身份（同一对象），不复制
        # 三份实例路径
        self.assertEqual(self.exp.shared_refs, ("shared.embed",
                                                "shared.head"))
        paths = [b.instance_path for b in self.exp.param_bindings]
        self.assertNotIn("mtp.0.embed.weight", paths)
        self.assertNotIn("mtp.0.head.weight", paths)

    def test_expansion_json_roundtrip(self):
        text = serialize_expansion(self.exp)
        doc = json.loads(text)
        self.assertEqual(doc["schema"], EXPANSION_SCHEMA)
        rebuilt = deserialize_expansion(text)
        self.assertEqual(serialize_expansion(rebuilt), text)
        # 独立构建一致（无隐藏状态）
        text2 = serialize_expansion(build_template_expansion(
            self.bundle.values))
        self.assertEqual(text2, text)

    def test_expansion_tamper_rejected(self):
        # 定向篡改负例：实例间复制参数路径（第 1 实例 → mtp.0 前缀）、
        # 复制 state、层号矛盾、shape/dtype 不符 → 具名拒绝
        text = serialize_expansion(self.exp)

        def tamper(mutate):
            doc = json.loads(text)
            mutate(doc)
            with self.assertRaises(ExtractionError):
                deserialize_expansion(json.dumps(doc))

        def copy_param(doc):
            for pb in doc["param_bindings"]:
                if pb["stage_id"] == 1:
                    pb["instance_path"] = ("mtp.0."
                                           + pb["instance_path"].split(
                                               ".", 1)[1])
                    break

        def copy_state(doc):
            doc["state_bindings"][1]["state_id"] = "dspark:win_cache:mtp.0"

        def bad_layer(doc):
            doc["state_bindings"][2]["layer_id"] = 43

        def bad_shape(doc):
            for pb in doc["param_bindings"]:
                if pb["spec_id"] == "stage.wkv_w":
                    pb["shape"] = [256, 4096]
                    break

        def bad_dtype(doc):
            for pb in doc["param_bindings"]:
                if pb["spec_id"] == "stage.hc_attn_fn_w":
                    pb["dtype"] = "bf16"
                    break

        def drop_common(doc):
            doc["param_bindings"] = [
                b for b in doc["param_bindings"]
                if not (b["spec_id"] == "stage.wkv_w"
                        and b["stage_id"] == 2)]

        def wrong_role(doc):
            for pb in doc["param_bindings"]:
                if pb["spec_id"] == "stage.main_proj_w":
                    pb["stage_id"] = 2
                    pb["instance_path"] = "mtp.2.main_proj.weight"
                    break

        for mutate in (copy_param, copy_state, bad_layer, bad_shape,
                       bad_dtype, drop_common, wrong_role):
            tamper(mutate)

    def test_expansion_rename_path_rejected(self):
        # B2 补修 02 负例 1（Codex 实测篡改）：合法前缀 mtp.0. 但槽位不
        # 对应 spec.param_slot → 读回具名拒绝（完整路径恰为
        # mtp.{stage_id}.{param_slot}，非 startswith 前缀）
        text = serialize_expansion(self.exp)
        bad = json.loads(text)
        bad["param_bindings"][0]["instance_path"] = (
            "mtp.0.nonexistent_wrong_weight")
        with self.assertRaises(ExtractionError) as ctx:
            deserialize_expansion(json.dumps(bad))
        msg = str(ctx.exception)
        self.assertIn("mtp.0.nonexistent_wrong_weight", msg)
        self.assertIn("不恰为可信映射", msg)
        # 通配槽同样按完整路径核对（含 *）：改成错误通配层数被拒
        bad2 = json.loads(text)
        for pb in bad2["param_bindings"]:
            if pb["spec_id"] == "stage.expert_w1_w" and pb["stage_id"] == 1:
                pb["instance_path"] = "mtp.1.ffn.experts.*.w3.weight"
                break
        with self.assertRaises(ExtractionError):
            deserialize_expansion(json.dumps(bad2))

    def test_expansion_drop_spec_and_bindings_rejected(self):
        # B2 补修 02 负例 2（Codex 实测篡改）：删一个公共 spec 及其全部
        # bindings → 期望目录来自固定 ModelValues（非被检文档自身），整槽
        # 消失具名拒绝
        text = serialize_expansion(self.exp)
        bad = json.loads(text)
        sid = "stage.wkv_w"
        bad["param_specs"] = [s for s in bad["param_specs"]
                              if s["spec_id"] != sid]
        bad["param_bindings"] = [b for b in bad["param_bindings"]
                                 if b["spec_id"] != sid]
        with self.assertRaises(ExtractionError) as ctx:
            deserialize_expansion(json.dumps(bad))
        self.assertIn(sid, str(ctx.exception))
        self.assertIn("可信目录", str(ctx.exception))
        # spec 与全部 bindings 共同篡改 shape（两边一致地错）同样拒绝
        bad2 = json.loads(text)
        sid2 = "stage.attn_norm_w"
        for s in bad2["param_specs"]:
            if s["spec_id"] == sid2:
                s["shape"] = [123]
        for b in bad2["param_bindings"]:
            if b["spec_id"] == sid2:
                b["shape"] = [123]
        with self.assertRaises(ExtractionError) as ctx2:
            deserialize_expansion(json.dumps(bad2))
        self.assertIn("共同篡改 shape", str(ctx2.exception))
        # spec 与全部 bindings 共同篡改 dtype 同样拒绝
        bad3 = json.loads(text)
        sid3 = "stage.gate_weight_w"
        for s in bad3["param_specs"]:
            if s["spec_id"] == sid3:
                s["dtype"] = "fp8_e4m3fn"
        for b in bad3["param_bindings"]:
            if b["spec_id"] == sid3:
                b["dtype"] = "fp8_e4m3fn"
        with self.assertRaises(ExtractionError) as ctx3:
            deserialize_expansion(json.dumps(bad3))
        self.assertIn("共同篡改 dtype", str(ctx3.exception))
        # n_stages/n_main_layers 与固定 config 不一致也被拒（期望目录
        # 维度以固定来源为准）
        bad4 = json.loads(text)
        bad4["n_stages"] = 4
        with self.assertRaises(ExtractionError):
            deserialize_expansion(json.dumps(bad4))

    def test_expansion_wrong_template_state_rejected(self):
        # B2 补修 02 负例 3（Codex 实测篡改）：template_state_id 改成
        # nonexistent_state、state_id 同步改拼、cache_tensor 改
        # not_a_cache_tensor → 合法 state 以固定源码模板常量（B1 _states
        # 单一实现）为准，不从待检字段拼期望——具名拒绝
        text = serialize_expansion(self.exp)
        bad = json.loads(text)
        bad["state_bindings"][0]["template_state_id"] = "nonexistent_state"
        bad["state_bindings"][0]["state_id"] = "nonexistent_state:mtp.0"
        bad["state_bindings"][0]["cache_tensor"] = "not_a_cache_tensor"
        with self.assertRaises(ExtractionError) as ctx:
            deserialize_expansion(json.dumps(bad))
        self.assertIn("nonexistent_state", str(ctx.exception))
        self.assertIn("dspark:win_cache", str(ctx.exception))
        # state_id 拼接路径单独篡改（template 合法、state_id 不恰为
        # {template}:mtp.{stage_id}）被拒
        bad2 = json.loads(text)
        bad2["state_bindings"][1]["state_id"] = "dspark:win_cache:mtp.9"
        with self.assertRaises(ExtractionError):
            deserialize_expansion(json.dumps(bad2))
        # cache_tensor 单独篡改被拒
        bad3 = json.loads(text)
        bad3["state_bindings"][2]["cache_tensor"] = "not_a_cache_tensor"
        with self.assertRaises(ExtractionError) as ctx3:
            deserialize_expansion(json.dumps(bad3))
        self.assertIn("not_a_cache_tensor", str(ctx3.exception))
        # 版本对/transition 单独篡改被拒
        bad4 = json.loads(text)
        bad4["state_bindings"][0]["version_labels"] = ["v0", "v1"]
        with self.assertRaises(ExtractionError):
            deserialize_expansion(json.dumps(bad4))
        bad5 = json.loads(text)
        bad5["state_bindings"][1]["transition_id"] = "tr_fake"
        with self.assertRaises(ExtractionError):
            deserialize_expansion(json.dumps(bad5))

    def test_expansion_state_facts_match_real_stage_graph(self):
        # 合法 state/version/transition 映射与真实 stage 图对应（B2 补修
        # 02）：常量经构建期 _verify_state_facts 对照当前图核对——版本
        # v_hist/v_after、transition tr_dspark_cache 属 dspark:win_cache、
        # cache 张量 tensor:stage:win_cache（cache_tensor 字段语义 = 图内
        # name，stable ID = tensor:+name）、state 边引用该 transition
        from structure_reader.extract.dspark_stage import (
            TEMPLATE_CACHE_TENSOR_NAME,
            TEMPLATE_STATE_ID,
            TEMPLATE_STATE_TRANSITION,
            TEMPLATE_STATE_VERSIONS,
            _verify_state_facts,
        )
        g = self.bundle.graph
        versions = {(v.state_id, v.version_label) for v in g.state_versions}
        for label in TEMPLATE_STATE_VERSIONS:
            self.assertIn((TEMPLATE_STATE_ID, label), versions)
        trans = {t.stable_id: t.state_id for t in g.state_transitions}
        self.assertEqual(trans[TEMPLATE_STATE_TRANSITION], TEMPLATE_STATE_ID)
        tensor_ids = {t.stable_id for t in g.tensors}
        self.assertIn(f"tensor:{TEMPLATE_CACHE_TENSOR_NAME}", tensor_ids)
        self.assertIn("tensor:stage:win_cache", tensor_ids)
        state_edges = [e for e in g.edges if e.kind == "state"]
        self.assertTrue(any(e.state_transition_ref == TEMPLATE_STATE_TRANSITION
                            for e in state_edges))
        _verify_state_facts(g)  # 构建期同一实现再驱动
        sb = self.exp.state_bindings[0]
        self.assertEqual(sb.cache_tensor, "stage:win_cache")
        self.assertNotEqual(sb.cache_tensor, "tensor:stage:win_cache")

    def test_state_facts_tamper_detected(self):
        # 定向篡改负例：常量与真实图漂移（伪造版本/transition/张量名）
        # → _verify_state_facts 具名拒绝，不静默通过
        from structure_reader.extract.dspark_stage import _verify_state_facts
        from structure_reader.graph_document import SemanticGraph
        from dataclasses import replace
        g = self.bundle.graph
        fake_versions = tuple(
            replace(v, version_label="v_bogus") if
            (v.state_id == "dspark:win_cache" and v.version_label == "v_hist")
            else v for v in g.state_versions)
        g2 = SemanticGraph(
            template_id=g.template_id, named_axes=g.named_axes,
            tensors=g.tensors, nodes=g.nodes, edges=g.edges,
            state_versions=fake_versions,
            state_transitions=g.state_transitions, guards=g.guards,
            alias_relations=g.alias_relations, fold_groups=g.fold_groups)
        with self.assertRaises(ExtractionError):
            _verify_state_facts(g2)
        g3 = SemanticGraph(
            template_id=g.template_id, named_axes=g.named_axes,
            tensors=g.tensors, nodes=g.nodes, edges=g.edges,
            state_versions=g.state_versions,
            state_transitions=tuple(
                replace(t, state_id="dspark:other") if
                t.stable_id == "tr_dspark_cache" else t
                for t in g.state_transitions),
            guards=g.guards, alias_relations=g.alias_relations,
            fold_groups=g.fold_groups)
        with self.assertRaises(ExtractionError):
            _verify_state_facts(g3)


class TestChainMap(unittest.TestCase):
    """三 stage 链路：同一 main_x 分发、h 串接（显式 phase/guard）、
    readout 汇聚、ratios、embed_h 来源诚实。"""

    @classmethod
    def setUpClass(cls):
        cls.cm = build_chain_map(ROOT)
        cls.text = chain_map_to_json(cls.cm)

    def test_main_x_dispatched_to_all_three(self):
        # 同一 main_x 生产点（mtp.0.forward_embed）分发三 stage——
        # 不是各 stage 消费 40/41/42 各自 readout
        mainx = [e for e in self.cm.chain_edges
                 if e["edge_id"].startswith("chain.main_x_stage")]
        self.assertEqual(len(mainx), 3)
        self.assertEqual({e["dst"] for e in mainx}, {"mtp.0", "mtp.1",
                                                     "mtp.2"})
        self.assertTrue(all(e["src"] == "mtp.0.forward_embed"
                            for e in mainx))

    def test_chain_edges_carry_phase_guard(self):
        # 链路边显式 phase/guard 字段（条件不靠 note 暗示）；h 计算串接
        # decode-only，prefill passthrough 是独立边
        for e in self.cm.chain_edges:
            self.assertIn(e["phase"], ("both", "decode", "prefill"),
                          e["edge_id"])
            self.assertTrue(e["guard"] is None
                            or isinstance(e["guard"], str), e["edge_id"])
        h01 = next(e for e in self.cm.chain_edges
                   if e["edge_id"] == "chain.h_0_1")
        self.assertEqual(h01["phase"], "decode")
        self.assertEqual(h01["guard"], "start_pos > 0")
        pp = next(e for e in self.cm.chain_edges
                  if e["edge_id"] == "chain.prefill_h_passthrough")
        self.assertEqual(pp["phase"], "prefill")
        self.assertEqual(pp["guard"], "start_pos == 0")
        for mid in ("chain.main_x_stage0", "chain.main_x_stage1",
                    "chain.main_x_stage2", "chain.embed_h"):
            e = next(e for e in self.cm.chain_edges
                     if e["edge_id"] == mid)
            self.assertEqual(e["phase"], "both")
            self.assertIsNone(e["guard"], mid)

    def test_embed_h_source_is_forward_embed_hc_stream(self):
        # chain.embed_h 来源是 forward_embed 产出的 HC 展开流（embed 查
        # 表→unsqueeze→repeat），不是原始 transformer.embed.y 直接当 HC 流
        eh = next(e for e in self.cm.chain_edges
                  if e["edge_id"] == "chain.embed_h")
        self.assertEqual(eh["src"], "mtp.0.forward_embed")
        self.assertIn("HC 展开", eh["note"])
        self.assertNotIn("transformer.embed", eh["src"])

    def test_draft_h_chain(self):
        # h 串接 stage0→1→2，末 stage 输出留待 head（pending）
        h01 = next(e for e in self.cm.chain_edges
                   if e["edge_id"] == "chain.h_0_1")
        h12 = next(e for e in self.cm.chain_edges
                   if e["edge_id"] == "chain.h_1_2")
        self.assertEqual((h01["src"], h01["dst"]), ("mtp.0", "mtp.1"))
        self.assertEqual((h12["src"], h12["dst"]), ("mtp.1", "mtp.2"))
        hh = next(e for e in self.cm.chain_edges
                  if e["edge_id"] == "chain.h_head_pending")
        self.assertIn("pending", hh["note"])

    def test_prefill_passthrough_honest(self):
        # prefill：forward_spec for 循环仍执行并传递 h（931-934），各
        # stage prefill 分支 passthrough 返回原 h（不算草稿双 mHC 计算）；
        # 三 cache 构建无 head——说明链路诚实，不声称"无 h 串接消费"
        pp = next(e for e in self.cm.chain_edges
                  if e["edge_id"] == "chain.prefill_h_passthrough")
        self.assertIn("passthrough", pp["note"])
        self.assertIn("845-849", pp["note"])
        self.assertIn("for 循环", pp["note"])
        self.assertIn("ev_ds_block_prefill", pp["evidence"])

    def test_readout_binding_to_stage0_main_proj(self):
        # 40/41/42 → mean(hc) → 有序 concat → stage0 main_proj/main_norm；
        # 不把各 readout 对应到各 stage
        rb = self.cm.readout_binding
        self.assertEqual([c["layer"] for c in rb["consumers"]],
                         [40, 41, 42])
        self.assertEqual([c["order"] for c in rb["consumers"]], [0, 1, 2])
        self.assertEqual(rb["consumer_binding"]["param_path"], "mtp.0")
        self.assertEqual(rb["consumer_binding"]["module"], "main_proj")
        self.assertEqual(rb["consumer_binding"]["in_width_expr"],
                         EXPR_MAIN_PROJ_IN)
        self.assertIn("不把各 readout 对应到各 stage", rb["note"])

    def test_main_proj_width_expression(self):
        # H * N_tgt = 4096 * 3 = 12288（model.py:832）
        self.assertEqual(EXPR_MAIN_PROJ_IN, "H * N_tgt")
        env = {"H": 4096, "N_tgt": 3}
        self.assertEqual(evaluate_int(EXPR_MAIN_PROJ_IN, env), 12288)

    def test_ratios_consumers(self):
        # 46 项 = 前 43 主干 + 后 3 draft（全 0）
        rc = self.cm.ratios_consumers
        self.assertEqual(rc["total_len"], 46)
        self.assertEqual(rc["draft_tail_values"], [0, 0, 0])
        self.assertIn("model.py:753", rc["draft_consumers"])

    def test_nextn_note(self):
        # root config num_nextn_predict_layers=1（checkpoint 布局）≠
        # run config n_mtp_layers=3（推理图实例数）
        nn = self.cm.nextn_note
        self.assertEqual(nn["root_config_value"], 1)
        self.assertEqual(nn["run_config_value"], 3)
        self.assertIn("checkpoint", nn["consumer"])

    def test_chain_json_roundtrip(self):
        doc = json.loads(self.text)
        self.assertEqual(doc["schema"], CHAIN_SCHEMA)
        rebuilt = chain_map_from_json(self.text)
        self.assertEqual(chain_map_to_json(rebuilt), self.text)
        # 篡改检出：schema 不符被拒
        bad = json.loads(self.text)
        bad["schema"] = "other/v1"
        with self.assertRaises(ExtractionError):
            chain_map_from_json(json.dumps(bad))

    def test_chain_bindings_tamper_rejected(self):
        # ChainMap 读回同样不能绕过绑定一致性验证：复制实例身份被拒
        bad = json.loads(self.text)
        bad["stage_bindings"][2]["param_path"] = "mtp.0"
        bad["stage_bindings"][2]["state_id"] = "dspark:win_cache:mtp.0"
        bad["stage_bindings"][2]["actual_layer_id"] = 43
        with self.assertRaises(ExtractionError):
            chain_map_from_json(json.dumps(bad))

    def test_chain_bindings_match_derived(self):
        # 链路图携带的绑定与 derive_stage_bindings(3, 43) 同源一致
        derived = serialize_bindings(derive_stage_bindings(3, 43))
        in_chain = serialize_bindings(self.cm.stage_bindings)
        self.assertEqual(derived, in_chain)

    def test_fixed_inputs_untouched(self):
        # 固定输入（config.json）未被本模块修改：build_chain_map 两次构建
        # 逐字节一致且数值来自 config（43/3/(40,41,42)/46 项）
        cm2 = build_chain_map(ROOT)
        self.assertEqual(chain_map_to_json(cm2), self.text)


class TestStageCoordinates(unittest.TestCase):
    """坐标值显式记录：坐标 int64 非存储 view；消费端口可证伪。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build()
        cls.graph = cls.bundle.graph

    def test_coords_recorded(self):
        ids = {c.coordinate_id for c in self.bundle.coordinate_expressions}
        self.assertEqual(ids, {"dspark:slot_write", "dspark:draft_pos0"})

    def test_scalar_not_a_view_of_data_tensor(self):
        # 标量坐标张量零轴 int64；图内唯一 alias 是真实 batch view，
        # 不存在坐标 → bf16 数据张量的 view
        tm = _tensor_map(self.graph)
        for c in self.bundle.coordinate_expressions:
            t = tm[c.scalar_tensor_id.removeprefix("tensor:")]
            self.assertEqual(t.dtype, "int64")
            self.assertEqual(len(t.axes), 0)
        for a in self.graph.alias_relations:
            self.assertNotIn("slot_write", a.alias_tensor_id)
            self.assertNotIn("draft_pos0", a.alias_tensor_id)

    def test_coords_eval(self):
        # 表达式与 B1 同源：P%W 本次写槽、P+T 草稿起点
        coords = {c.coordinate_id: c
                  for c in self.bundle.coordinate_expressions}
        for p in (2, 127, 128, 255):
            self.assertEqual(evaluate_int(
                coords["dspark:slot_write"].expression,
                {"P": p, "T": 1, "W": 128}), p % 128)
        self.assertEqual(evaluate_int(
            coords["dspark:draft_pos0"].expression,
            {"P": 7, "T": 1, "W": 128}), 8)


if __name__ == "__main__":
    unittest.main()
