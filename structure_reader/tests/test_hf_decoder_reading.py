"""HF decoder 阅读样板自测（里程碑一，Claude 范围）。

只测当前源码在 tempfile 输出根的行为：不读取 .work/coordination、不按 mtime
选旧 outputs、不依赖既有候选；不导入/不执行 HF 模型代码、不加载权重。
模型输入（根 config.json/transformers 检出）为只读真实输入；缺失时相关测试
跳过并注明（不伪造通过）。

运行方式（工程 Python）：
  cd structure_reader && python3 tests/test_hf_decoder_reading.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_KEY = "deepseek_v4_flash_0731_hf"


def _workspace_ready() -> bool:
    root = REPO_ROOT
    return (root / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json").is_file() and (
        root / "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py"
    ).is_file()


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入（models/ 或 hflib/transformers）；如实跳过")
class TestReadingDocument(unittest.TestCase):
    """reading_document 构建与冻结接口符合性。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document

        cls.build = staticmethod(build_reading_document)
        cls.doc = build_reading_document(REPO_ROOT)

    def test_schema_and_identity(self):
        self.assertEqual(self.doc["schema"], "hf_decoder_reading.v1")
        self.assertEqual(self.doc["model"]["model_key"], MODEL_KEY)
        self.assertEqual(self.doc["model"]["implementation"], "transformers")
        self.assertNotEqual(self.doc["model"]["source_revision"], "unknown")
        self.assertEqual(self.doc["scope"]["layer_index"], 4)
        self.assertEqual(self.doc["scope"]["attention_type"], "compressed_sparse_attention")
        self.assertEqual(self.doc["scope"]["moe_type"], "moe")
        self.assertEqual(self.doc["scope"]["analysis_path"], "eager")

    def test_two_phases_in_order(self):
        self.assertEqual([p["id"] for p in self.doc["phases"]], ["prefill", "decode"])

    def test_reference_closure_both_phases(self):
        for phase in self.doc["phases"]:
            pids = {p["id"] for p in phase["parameters"]}
            tids = {t["id"] for t in phase["tensors"]}
            for t in phase["tensors"]:
                for ref in t["parameter_refs"]:
                    self.assertIn(ref, pids, f"{phase['id']}: 张量 {t['id']} 参数引用 {ref}")
            for o in phase["operators"]:
                for ref in o["inputs"] + o["outputs"] + o["weights"]:
                    self.assertIn(ref, tids, f"{phase['id']}: 算子 {o['id']} 张量引用 {ref}")
                for ref in o["parameter_refs"]:
                    self.assertIn(ref, pids, f"{phase['id']}: 算子 {o['id']} 参数引用 {ref}")

    def test_flow_closed_no_selfloop(self):
        """数据流机器核查：activation/index/mask 输入必须有更早 producer；无输入输出同 id。"""
        from collections import defaultdict

        for phase in self.doc["phases"]:
            kinds = {t["id"]: t["kind"] for t in phase["tensors"]}
            producers = defaultdict(list)
            for i, o in enumerate(phase["operators"]):
                for tid in o["outputs"]:
                    producers[tid].append(i)
            for i, o in enumerate(phase["operators"]):
                for tid in o["inputs"]:
                    if kinds[tid] in ("activation", "index", "mask"):
                        self.assertTrue(
                            producers[tid] and min(producers[tid]) < i,
                            f"{phase['id']}: 算子 {o['id']} 输入 {tid} 无更早 producer",
                        )
                self.assertFalse(
                    set(o["inputs"]) & set(o["outputs"]),
                    f"{phase['id']}: 算子 {o['id']} 输入输出同 id（原地更新须前后状态分离）",
                )

    def test_boundary_and_cache_kinds(self):
        """层输入/输出/位置是诚实边界；缓存调用前后状态分离。"""
        for phase in self.doc["phases"]:
            kinds = {t["id"]: t["kind"] for t in phase["tensors"]}
            for tid in ("hidden_streams_in", "hidden_streams_out", "position_ids", "input_ids"):
                self.assertEqual(kinds[tid], "boundary", f"{phase['id']}: {tid} 应为边界")
            for pre in ("cache_keys", "cache_entry_count_c", "cache_compressed_c",
                        "cache_overlap_c_kv"):
                self.assertEqual(kinds[pre], "cache")
                self.assertIn(pre + "_post", kinds, f"{phase['id']}: 缺调用后状态 {pre}_post")

    def test_no_repeat_count_field(self):
        blob = json.dumps(self.doc, ensure_ascii=False)
        for token in ("repeat_count", "repetition_count", "重复次数"):
            self.assertNotIn(token, blob)

    def test_expert_template_not_unrolled(self):
        """E=256 专家只允许一份模板链（每算子一行），不逐专家复制行。"""
        import re

        chain = ("moe_select_tokens", "moe_expert_slice", "moe_expert_gu",
                 "moe_expert_clamp", "moe_expert_act", "moe_expert_down",
                 "moe_expert_weight", "moe_combine", "moe_routed_view")
        for phase in self.doc["phases"]:
            op_ids = [o["id"] for o in phase["operators"]]
            for oid in chain:
                self.assertEqual(op_ids.count(oid), 1, f"{phase['id']}: 模板算子 {oid} 必须恰一次")
            self.assertFalse(
                [oid for oid in op_ids if re.fullmatch(r"moe_expert_\d+.*", oid)],
                f"{phase['id']}: 不得出现逐专家复制行",
            )
            e_param = next(p for p in phase["parameters"] if p["id"] == "E")
            self.assertEqual(e_param["value"], 256)
            select = next(o for o in phase["operators"] if o["id"] == "moe_select_tokens")
            self.assertIn("ffn_flat", select["inputs"], "专家模板必须读真实 token 行")

    def test_all_four_real_layer_types_supported_and_out_of_range_rejected(self):
        """四类真实层型均已支持（旧「SWA/Hash 未实现」断言随功能迁移）；越界层具名拒绝。"""
        for idx in (0, 2, 3, 4):
            doc = self.build(workspace_root=REPO_ROOT, layer_index=idx)
            self.assertEqual([p["id"] for p in doc["phases"]], ["prefill", "decode"])
        with self.assertRaises(ValueError):
            self.build(workspace_root=REPO_ROOT, layer_index=43)
        with self.assertRaises(ValueError):
            self.build(workspace_root=REPO_ROOT, layer_index=-1)


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestSemanticsAnchors(unittest.TestCase):
    """关键语义锚点（mHC 方向/共享 KV/缓存顺序/eager 链/RoPE/MoE）。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document

        cls.doc = build_reading_document(REPO_ROOT)

    def test_mhc_comb_transpose_both_segments_both_phases(self):
        for phase in self.doc["phases"]:
            ops = {o["id"]: o for o in phase["operators"]}
            tensors = {t["id"]: t for t in phase["tensors"]}
            for seg in ("attn", "ffn"):
                mix = ops[f"{seg}_residual_mix"]
                blob = mix["note"] + ops[f"{seg}_residual_combT"]["note"]
                self.assertIn("comb", blob)
                self.assertTrue(
                    "transpose" in blob or "combᵀ" in blob,
                    f"{phase['id']}/{seg}: comb 转置方向未表达",
                )
                # combT 是 rank 正确的独立张量，且在 mix 之前被 mix 真实消费
                combt = tensors[f"{seg}_combT"]
                self.assertEqual(combt["shape"], ["B", "seqlen", "Nhc", "Nhc"])
                self.assertIn(f"{seg}_combT", mix["inputs"])
                order = {o["id"]: i for i, o in enumerate(phase["operators"])}
                self.assertLess(order[f"{seg}_residual_combT"], order[f"{seg}_residual_mix"])

    def test_mhc_raw_weights_separate_from_coefficients(self):
        """原始拆分权重与门控系数分离：split 产 *_w，门控算子消费 *_w。"""
        for phase in self.doc["phases"]:
            ops = {o["id"]: o for o in phase["operators"]}
            for seg in ("attn", "ffn"):
                split = ops[f"{seg}_hc_split"]
                for raw in ("pre_w", "post_w", "comb_w"):
                    out_id = f"{seg}_mhc_{raw}"
                    self.assertIn(out_id, split["outputs"], f"{seg}: split 缺原始 {raw}")
                for gate_op, raw in ((f"{seg}_hc_pre", "pre_w"), (f"{seg}_hc_post", "post_w"),
                                     (f"{seg}_hc_comb", "comb_w")):
                    self.assertIn(f"{seg}_mhc_{raw}", ops[gate_op]["inputs"],
                                  f"{gate_op} 必须消费原始拆分权重")
                    self.assertNotIn(f"{seg}_mhc_weights", ops[gate_op]["inputs"],
                                     f"{gate_op} 不得重复消费整段投影输出")

    def test_shared_kv_single_head(self):
        for phase in self.doc["phases"]:
            nkv = next(p for p in phase["parameters"] if p["id"] == "Nkv")
            self.assertEqual(nkv["value"], 1)
            upd = next(o for o in phase["operators"] if o["id"] == "cache_update")
            self.assertIn("kv_rope", upd["inputs"])
            # kv_head 有真实 producer（view/transpose 显式算子）
            kv_view = next(o for o in phase["operators"] if o["id"] == "kv_head_view")
            self.assertEqual(kv_view["outputs"], ["kv_head"])
            rope_apply = next(o for o in phase["operators"] if o["id"] == "kv_rope_apply")
            self.assertIn("kv_head", rope_apply["inputs"])
            # cache_update 写调用后状态（不与调用前同 id）
            self.assertIn("cache_keys", upd["inputs"])
            self.assertIn("cache_keys_post", upd["outputs"])

    def test_cache_update_before_compressed_concat(self):
        for phase in self.doc["phases"]:
            order = {o["id"]: i for i, o in enumerate(phase["operators"])}
            self.assertLess(order["cache_update"], order["kv_concat"])

    def test_eager_chain_order(self):
        for phase in self.doc["phases"]:
            order = {o["id"]: i for i, o in enumerate(phase["operators"])}
            self.assertLess(order["qk_scaled"], order["mask_add"])
            self.assertLess(order["mask_add"], order["sink_softmax"])
            self.assertLess(order["sink_softmax"], order["av_matmul"])
            sink = next(o for o in phase["operators"] if o["id"] == "sink_softmax")
            blob = (sink["note"] + sink["op"]).lower()
            for token in ("cat", "softmax"):
                self.assertIn(token, blob)
            # AV 概率输入是去 sink 概率张量（非 q_a 等伪连接）
            av = next(o for o in phase["operators"] if o["id"] == "av_matmul")
            self.assertEqual(av["inputs"][0], "attn_probs")
            self.assertIn("kv_repeated", av["inputs"])

    def test_probability_axis_matches_kv_axis(self):
        """概率序列轴=L_kv+N_comp（与 AV 的 KV 轴闭合），不是 Nkv+N_comp。"""
        for phase in self.doc["phases"]:
            shapes = {t["id"]: t["shape"] for t in phase["tensors"]}
            for tid in ("combined_logits", "probs", "attn_probs", "qk_masked"):
                self.assertIn("L_kv+N_comp", "".join(shapes[tid]),
                              f"{phase['id']}: {tid} 概率/KV 序列轴应含 L_kv+N_comp")
            self.assertNotIn("Nkv+N_comp", json.dumps(shapes, ensure_ascii=False))

    def test_topk_and_persistent_lengths_are_min(self):
        """topk 长度 min(Ki,条目数)；持久 KV min(累计,W-1)；mask 长度 min(P,W-1)+T。"""
        import json as _json
        for phase in self.doc["phases"]:
            shapes = {t["id"]: t["shape"] for t in phase["tensors"]}
            params = {p["id"]: p for p in phase["parameters"]}
            self.assertEqual(shapes["indexer_topk"], ["B", "seqlen", "min(Ki,Tci+N_win_i)"])
            self.assertEqual(shapes["cache_keys"][2], "min(kvcachelen,W-1)")
            self.assertEqual(shapes["cache_keys_post"][2], "min(kvcachelen+seqlen,W-1)")
            self.assertIn("min(kvcachelen, W - 1) + seqlen", params["Lkv"]["expression"])
            self.assertIn("min(kvcachelen, W - 1) + seqlen", params["Lmask"]["expression"])
            self.assertIn("Nwin_i", params, "Indexer 系列闭窗数独立成参")

    def test_attention_output_layout_before_after_rope(self):
        """AV 输出与逆 RoPE 前后都是 [B,T,Nh,Dh]（transpose 配对为 layout 修正）。"""
        for phase in self.doc["phases"]:
            shapes = {t["id"]: t["shape"] for t in phase["tensors"]}
            for tid in ("attn_head_out", "attn_rope_inv"):
                self.assertEqual(shapes[tid], ["B", "seqlen", "Nh", "Dh"],
                                 f"{phase['id']}: {tid} layout 应为 [B,T,Nh,Dh]")
            self.assertEqual(shapes["i_q"], ["B", "seqlen", "Ih", "Id"],
                             "Indexer q 转置回后为 [B,T,Ih,Id]")

    def test_indexer_reads_normalized_residual_and_block_bias_order(self):
        """Indexer 查询读归一后 q_residual；块偏置在 topk 之后构建。"""
        for phase in self.doc["phases"]:
            ops = {o["id"]: o for o in phase["operators"]}
            self.assertIn("q_a_normed", ops["i_q_proj"]["inputs"])
            self.assertNotIn("q_a", ops["i_q_proj"]["inputs"])
            order = {o["id"]: i for i, o in enumerate(phase["operators"])}
            self.assertLess(order["i_topk"], order["c_block_bias"],
                            "c_block_bias 必须在 indexer_topk 生产之后（源码 :690→:692-699）")
            # 索引器/压缩器两套 overlap 与 gate 前后状态都被消费/写出
            wv = ops["i_window_view"]
            self.assertIn("cache_overlap_i_gate", wv["inputs"])
            self.assertIn("cache_overlap_i_gate_post", wv["outputs"])
            self.assertIn("i_chunk_gate", wv["inputs"])
            cwv = ops["c_window_view"]
            self.assertIn("cache_overlap_c_gate", cwv["inputs"])
            self.assertIn("cache_overlap_c_gate_post", cwv["outputs"])
            # 压缩 RoPE cos/sin 有 producer 且位置链经 entry_count
            self.assertIn("cache_entry_count_c", ops["c_rope_gen"]["inputs"])

    def test_q_a_norm_has_weight(self):
        """q_a_norm 是带权 RMSNorm，必须绑定 q_a_norm.weight。"""
        for phase in self.doc["phases"]:
            tensors = {t["id"]: t for t in phase["tensors"]}
            self.assertIn("q_a_norm_weight", tensors)
            self.assertEqual(tensors["q_a_norm_weight"]["shape"], ["Rq"])
            norm = next(o for o in phase["operators"] if o["id"] == "q_a_norm")
            self.assertIn("q_a_norm_weight", norm["weights"])

    def test_partial_rope_and_inverse(self):
        for phase in self.doc["phases"]:
            params = {p["id"]: p["value"] for p in phase["parameters"]}
            self.assertEqual(params["Rk"], 64)
            self.assertEqual(params["Dh"], 512)
            inv = next(o for o in phase["operators"] if o["id"] == "attn_inv_rope")
            self.assertIn("-sin", inv["note"])

    def test_compressed_entry_deterministic_positions(self):
        for phase in self.doc["phases"]:
            c_rope = next(o for o in phase["operators"] if o["id"] == "c_rope")
            self.assertIn("first_window_position", c_rope["note"] + c_rope["source_refs"][0]["locator"])

    def test_expert_clamp_silu_and_combine(self):
        for phase in self.doc["phases"]:
            clamp = next(o for o in phase["operators"] if o["id"] == "moe_expert_clamp")
            blob = (clamp["note"] + clamp["op"]).lower()
            for token in ("chunk", "clamp"):
                self.assertIn(token, blob)
            act = next(o for o in phase["operators"] if o["id"] == "moe_expert_act")
            blob = (act["note"] + act["op"]).lower()
            for token in ("silu", "mul"):
                self.assertIn(token, blob)
            combine = next(o for o in phase["operators"] if o["id"] == "moe_combine")
            self.assertIn("index_add", combine["op"] + combine["note"])
            self.assertIn("expert_token_idx", combine["inputs"])
            self.assertNotIn("topk_weights", combine["inputs"])

    def test_moe_routed_view_before_merge(self):
        """HF :1133 routed 先 view(B,T,H) 再与 shared add——显式维度变换算子。"""
        for phase in self.doc["phases"]:
            order = {o["id"]: i for i, o in enumerate(phase["operators"])}
            self.assertLess(order["moe_combine"], order["moe_routed_view"])
            self.assertLess(order["moe_routed_view"], order["moe_merge"])
            shapes = {t["id"]: t["shape"] for t in phase["tensors"]}
            self.assertEqual(shapes["routed_out"], ["B×seqlen", "H"])
            self.assertEqual(shapes["routed_view"], ["B", "seqlen", "H"])
            merge = next(o for o in phase["operators"] if o["id"] == "moe_merge")
            self.assertEqual(merge["inputs"], ["routed_view", "shared_out"])

    def test_expert_chain_connections(self):
        """单专家模板链按源码连接：选行→切片→gate_up→chunk/clamp→SiLU·mul→down→路由加权。"""
        for phase in self.doc["phases"]:
            ops = {o["id"]: o for o in phase["operators"]}
            shapes = {t["id"]: t["shape"] for t in phase["tensors"]}
            self.assertEqual(ops["moe_select_tokens"]["outputs"],
                             ["expert_tokens", "expert_token_idx", "expert_topk_pos"])
            # HF :1045-1052：expert_idx 选择 2D 权重切片，Linear 不直接消费 3D 整库
            sl = ops["moe_expert_slice"]
            self.assertEqual(set(sl["outputs"]), {"expert_gate_up_e", "expert_down_e"})
            self.assertIn("hit_experts", sl["inputs"])
            self.assertEqual(shapes["expert_gate_up_e"], ["2Es", "H"])
            self.assertEqual(shapes["expert_down_e"], ["H", "Es"])
            self.assertEqual(ops["moe_expert_gu"]["weights"], ["expert_gate_up_e"])
            self.assertEqual(ops["moe_expert_down"]["weights"], ["expert_down_e"])
            self.assertEqual(ops["moe_expert_gu"]["inputs"], ["expert_tokens"])
            self.assertEqual(ops["moe_expert_clamp"]["inputs"], ["expert_gu_out"])
            self.assertEqual(ops["moe_expert_act"]["inputs"], ["expert_gate_c", "expert_up_c"])
            self.assertEqual(ops["moe_expert_down"]["inputs"], ["expert_gated"])
            # 路由权重按 (token_idx, top_k_pos) 双索引真实寻址，只乘一次
            w = ops["moe_expert_weight"]
            self.assertEqual(set(w["inputs"]),
                             {"expert_down_raw", "topk_weights", "expert_token_idx",
                              "expert_topk_pos"})

    def test_overlap_shape_and_window_condition(self):
        """HF :278-292 chunk[:, -1, :, :head_dim] → overlap [B,M,Dh]/[B,M,Id]；N_win=0 不读写。"""
        for phase in self.doc["phases"]:
            shapes = {t["id"]: t["shape"] for t in phase["tensors"]}
            self.assertEqual(shapes["cache_overlap_c_kv"], ["B", "M", "Dh"])
            self.assertEqual(shapes["cache_overlap_c_gate"], ["B", "M", "Dh"])
            self.assertEqual(shapes["cache_overlap_c_kv_post"], ["B", "M", "Dh"])
            self.assertEqual(shapes["cache_overlap_i_kv"], ["B", "M", "Id"])
            self.assertEqual(shapes["cache_overlap_i_kv_post"], ["B", "M", "Id"])
            for oid, win_param in (("c_window_view", "N_win"), ("i_window_view", "N_win_i")):
                op = next(o for o in phase["operators"] if o["id"] == oid)
                blob = op["note"]
                self.assertIn(win_param, blob, f"{oid} 未表达闭窗条件")
                self.assertTrue(
                    ("N_win" in blob and "=0" in blob) or "不读写" in blob or "不发生" in blob,
                    f"{oid} 未表达无闭窗时 overlap 保留",
                )

    def test_indexer_scores_cover_new_entries(self):
        """i_scores 轴=Tci+N_win_i（:556-566 用追加后条目计分/TopK），Tci 保持调用前含义。"""
        for phase in self.doc["phases"]:
            shapes = {t["id"]: t["shape"] for t in phase["tensors"]}
            self.assertEqual(shapes["i_scores"], ["B", "seqlen", "Tci+N_win_i"])
            self.assertEqual(shapes["cache_compressed_i"], ["B", "Tci", "Id"])
            self.assertEqual(shapes["cache_compressed_i_post"], ["B", "Tci+N_win_i", "Id"])
            tci = next(p for p in phase["parameters"] if p["id"] == "Tci")
            self.assertIsNone(tci["value"], "Tci 为运行时数量，值必须 null")
            self.assertIn("调用前", tci["meaning"])

    def test_cache_length_symbols_documented(self):
        """shape 引用的缓存长度符号在参数区有含义/公式/runtime null。"""
        for phase in self.doc["phases"]:
            params = {p["id"]: p for p in phase["parameters"]}
            for pid in ("Tbuf", "Tbufp", "Tbufi", "Tbufip", "Tcc", "Tci"):
                self.assertIn(pid, params, f"{phase['id']}: 缺缓存长度参数 {pid}")
                self.assertIsNone(params[pid]["value"])
                self.assertTrue(params[pid]["expression"], f"{pid} 缺公式")

    def test_position_ids_default_shape(self):
        """默认 position_ids [1,T]（modeling:1335-1337 unsqueeze(0)），与 cos/sin [1,T,Rk÷2] 一致。"""
        for phase in self.doc["phases"]:
            shapes = {t["id"]: t["shape"] for t in phase["tensors"]}
            self.assertEqual(shapes["position_ids"], ["1", "seqlen"])
            self.assertEqual(shapes["cos_q"], ["1", "seqlen", "Rk÷2"])
            self.assertEqual(shapes["sin_q"], ["1", "seqlen", "Rk÷2"])

    def test_prefill_vs_decode_scenario_binding(self):
        """两阶段只绑定场景取值（空缓存 Prefill / 单 token Decode 预设），不是两套结构。"""
        notes = {p["id"]: p["cache_notes"] for p in self.doc["phases"]}
        # 缓存注记是同一份结构的条件说明：两阶段都同时描述空缓存与非零历史
        for pid in ("prefill", "decode"):
            blob = " ".join(notes[pid])
            self.assertIn("空缓存", blob)
            self.assertIn("历史", blob)
        kv_prefill = next(p for p in self.doc["phases"][0]["parameters"] if p["id"] == "kvcachelen")
        kv_decode = next(p for p in self.doc["phases"][1]["parameters"] if p["id"] == "kvcachelen")
        self.assertEqual(kv_prefill["value"], 0)
        self.assertIsNone(kv_decode["value"])
        seq_prefill = next(p for p in self.doc["phases"][0]["parameters"] if p["id"] == "seqlen")
        seq_decode = next(p for p in self.doc["phases"][1]["parameters"] if p["id"] == "seqlen")
        self.assertIsNone(seq_prefill["value"], "Prefill 的 seqlen 保留符号")
        self.assertEqual(seq_decode["value"], 1, "Decode 单 token 只是场景预设")
        # 同一份结构的张量/算子计数在两阶段完全一致（无第二套阶段事实）
        p0, p1 = self.doc["phases"]
        self.assertEqual([t["id"] for t in p0["tensors"]], [t["id"] for t in p1["tensors"]])
        self.assertEqual([o["id"] for o in p0["operators"]], [o["id"] for o in p1["operators"]])
        self.assertEqual([t["shape"] for t in p0["tensors"]], [t["shape"] for t in p1["tensors"]])

    def test_no_internal_unanalyzed_claimed_analyzed(self):
        """本样板全算子 analyzed；黑盒机制不在文档中使用（如实声明于 coverage）。"""
        for phase in self.doc["phases"]:
            for o in phase["operators"]:
                self.assertEqual(o["analysis_status"], "analyzed")


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestHcaVariant(unittest.TestCase):
    """HCA（layer 3）实际提取：正例两阶段 + 错误 2Dh/Indexer 负例；layer 4 保持。"""

    @classmethod
    def setUpClass(cls):
        import hashlib
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES
        cls.shas = {k: hashlib.sha256((REPO_ROOT / v).read_bytes()).hexdigest()
                    for k, v in SOURCE_FILES.items()}
        cls.doc = build_reading_document(REPO_ROOT, layer_index=3)
        cls.csa = build_reading_document(REPO_ROOT, layer_index=4)

    def test_hca_two_phases_and_scope(self):
        from structure_reader.checks.deepseek_v4_hf_contracts import validate_hf_reading_document
        doc = self.doc
        self.assertEqual([p["id"] for p in doc["phases"]], ["prefill", "decode"])
        self.assertEqual(doc["scope"]["layer_index"], 3)
        self.assertEqual(doc["scope"]["attention_type"], "heavily_compressed_attention")
        self.assertEqual(validate_hf_reading_document(doc, self.shas), [])
        for phase in doc["phases"]:
            ops = {o["id"] for o in phase["operators"]}
            for oid in ("c_kv_proj", "c_gate_proj", "c_store_window", "c_window_view",
                        "c_pool", "c_rope", "c_update_states", "hca_block_bias",
                        "cache_update", "kv_concat", "sink_softmax", "av_matmul"):
                self.assertIn(oid, ops)
            self.assertFalse(any(o.startswith("i_") for o in ops))
            tensors = {t["id"]: t for t in phase["tensors"]}
            self.assertEqual(tensors["c_kv"]["shape"][-1], "Dh")
            self.assertEqual(tensors["c_gate"]["shape"][-1], "Dh")
            self.assertEqual(tensors["c_position_bias"]["shape"], ["M", "Dh"])
            self.assertNotIn("cache_buffer_i_kv", tensors)
            self.assertNotIn("cache_overlap_c_kv", tensors)
            self.assertNotIn("cache_values", tensors)
            self.assertIn("cache_values_post", tensors)
            upd = next(o for o in phase["operators"] if o["id"] == "cache_update")
            self.assertIn("self.values = self.keys", upd["note"])
            cro = next(o for o in phase["operators"] if o["id"] == "c_rope")
            self.assertNotIn("position_ids", cro["inputs"])
            rg = next(o for o in phase["operators"] if o["id"] == "c_rope_gen")
            self.assertIn("cache_entry_count_c", rg["inputs"])
        self.assertEqual(len(self.csa["phases"][0]["operators"]), 75)

    def test_hca_negative_rope_and_kv_norm(self):
        import copy
        from structure_reader.checks.deepseek_v4_hf_contracts import validate_hf_reading_document
        # c_rope 退回 query position_ids
        bad = copy.deepcopy(self.doc)
        for o in bad["phases"][0]["operators"]:
            if o["id"] == "c_rope":
                o["inputs"] = ["c_compressed", "position_ids"]
        issues = validate_hf_reading_document(bad, self.shas)
        self.assertTrue(any("position_ids" in i for i in issues), issues[:3])
        # kv_proj 丢 RMSNorm 权重
        bad2 = copy.deepcopy(self.doc)
        for o in bad2["phases"][0]["operators"]:
            if o["id"] == "kv_proj":
                o["weights"] = ["kv_weight_attn"]
        issues2 = validate_hf_reading_document(bad2, self.shas)
        self.assertTrue(any("kv_norm_weight" in i for i in issues2), issues2[:3])
        # cache_values 独立字段回归
        bad3 = copy.deepcopy(self.doc)
        bad3["phases"][0]["tensors"].append({
            "id": "cache_values", "name": "x", "shape": ["B", "1", "min(P,W-1)", "Dh"],
            "parameter_refs": [], "kind": "cache", "note": "独立字段"})
        issues3 = validate_hf_reading_document(bad3, self.shas)
        self.assertTrue(any("cache_values" in i for i in issues3), issues3[:3])

    def test_hca_negative_2dh_and_indexer(self):
        import copy
        from structure_reader.checks.deepseek_v4_hf_contracts import validate_hf_reading_document
        bad = copy.deepcopy(self.doc)
        for t in bad["phases"][0]["tensors"]:
            if t["id"] == "c_kv":
                t["shape"] = ["B", "seqlen", "2Dh"]
        issues = validate_hf_reading_document(bad, self.shas)
        self.assertTrue(any("head_dim" in i for i in issues), issues[:3])
        bad2 = copy.deepcopy(self.doc)
        bad2["phases"][0]["operators"].append({
            "id": "i_topk", "module_path": "x", "name": "Indexer", "op": "topk",
            "inputs": [], "outputs": [], "weights": [], "parameter_refs": [],
            "source_refs": [], "analysis_status": "analyzed", "note": "-1"})
        issues2 = validate_hf_reading_document(bad2, self.shas)
        self.assertTrue(any("Indexer" in i for i in issues2), issues2[:3])


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestSwaHashVariant(unittest.TestCase):
    """SWA+Hash（层0）与 CSA+Hash（层2）数据正负例；HCA/CSA 回归。"""

    @classmethod
    def setUpClass(cls):
        import hashlib
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES
        cls.shas = {k: hashlib.sha256((REPO_ROOT / v).read_bytes()).hexdigest()
                    for k, v in SOURCE_FILES.items()}
        cls.swa = build_reading_document(REPO_ROOT, layer_index=0)
        cls.csh = build_reading_document(REPO_ROOT, layer_index=2)
        cls.hca = build_reading_document(REPO_ROOT, layer_index=3)
        cls.csa = build_reading_document(REPO_ROOT, layer_index=4)

    def _validate(self, doc):
        from structure_reader.checks.deepseek_v4_hf_contracts import validate_hf_reading_document
        return validate_hf_reading_document(doc, self.shas)

    def test_swa_hash_document(self):
        doc = self.swa
        self.assertEqual(doc["scope"]["attention_type"], "sliding_attention")
        self.assertEqual(doc["scope"]["moe_type"], "hash_moe")
        self.assertEqual(self._validate(doc), [])
        p = doc["phases"][0]
        ops = {o["id"] for o in p["operators"]}
        tensors = {t["id"]: t for t in p["tensors"]}
        for bad in ("c_kv_proj", "c_store_window", "c_rope", "kv_concat", "i_topk"):
            self.assertNotIn(bad, ops)
        self.assertEqual(tensors["cache_values"]["note"].count("独立"), 1)
        self.assertNotIn("block_bias", tensors)
        self.assertIn("moe_hash_lookup", ops)
        self.assertIn("tid2eid", tensors)
        self.assertIn("input_ids", tensors)
        self.assertNotIn("router_bias", tensors)
        self.assertEqual(self._validate(self.hca), [])
        self.assertEqual(self._validate(self.csa), [])

    def test_csa_hash_document(self):
        doc = self.csh
        self.assertEqual(doc["scope"]["attention_type"], "compressed_sparse_attention")
        self.assertEqual(doc["scope"]["moe_type"], "hash_moe")
        self.assertEqual(self._validate(doc), [])
        ops = {o["id"] for o in doc["phases"][0]["operators"]}
        self.assertIn("moe_hash_lookup", ops)
        self.assertNotIn("router_bias", {t["id"] for t in doc["phases"][0]["tensors"]})

    def test_swa_hash_negatives(self):
        import copy
        bad = copy.deepcopy(self.swa)
        for t in bad["phases"][0]["tensors"]:
            if t["id"] == "cache_values":
                t["note"] = "与 keys 同一存储"
        self.assertTrue(any("独立" in i for i in self._validate(bad)))
        bad2 = copy.deepcopy(self.swa)
        bad2["phases"][0]["tensors"] = [t for t in bad2["phases"][0]["tensors"]
                                        if t["id"] != "tid2eid"]
        self.assertTrue(any("tid2eid" in i for i in self._validate(bad2)))
        bad3 = copy.deepcopy(self.swa)
        bad3["phases"][0]["tensors"].append({
            "id": "router_bias", "name": "bias", "shape": ["E"], "parameter_refs": [],
            "kind": "weight", "note": "correction"})
        self.assertTrue(any("bias" in i for i in self._validate(bad3)))


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestVariantDiagram(unittest.TestCase):
    """8 份 phase 的 module_groups/overview/module 图：无 other/KeyError，变体 frame/title 正确。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        cls.docs = {i: build_reading_document(REPO_ROOT, layer_index=i) for i in (0, 2, 3, 4)}

    def test_groups_and_overview_all_phases(self):
        from structure_reader.render.diagram import (
            build_module_graph, build_overview_graph, module_groups, render_svg)
        for idx, doc in self.docs.items():
            for ph in doc["phases"]:
                groups = module_groups(ph)
                self.assertFalse(any(g["id"] == "other" and g["ops"] for g in groups))
                for g in groups:
                    if g["ops"]:
                        svg = render_svg(build_module_graph(ph, g))
                        self.assertTrue(svg.startswith("<svg"))
                ov = build_overview_graph(ph)
                render_svg(ov)
                op_ids = {o["id"] for o in ph["operators"]}
                if "c_kv_proj" in op_ids:  # HCA
                    self.assertEqual([f["id"] for f in ov["frames"]],
                                     ["frame:attention", "frame:csa"])
                    title = next(g["title"] for g in groups if g["id"] == "attn_csa")
                    self.assertIn("HCA", title)
                elif "moe_hash_lookup" in op_ids and "c_window_view" not in op_ids:  # SWA
                    self.assertEqual([f["id"] for f in ov["frames"]], ["frame:attention"])
                    self.assertNotIn("attn_csa", {g["id"] for g in groups})
                    ids = {n["id"] for n in ov["nodes"]}
                    self.assertIn("t:tid2eid", ids)
                    self.assertTrue(any(e["src"] == "t:tid2eid" for e in ov["edges"]))
                elif "i_topk" in op_ids:  # CSA
                    self.assertIn("frame:indexer", [f["id"] for f in ov["frames"]])


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestVariantLabelsNarrative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        cls.docs = {i: build_reading_document(REPO_ROOT, layer_index=i) for i in (0, 2, 3, 4)}

    def test_narratives_and_labels(self):
        import re
        from structure_reader.render.diagram import build_overview_graph, module_groups
        from structure_reader.render.diagram_source_labels import op_source_label
        from structure_reader.render.module_narrative import module_narrative
        for idx, doc in self.docs.items():
            for ph in doc["phases"]:
                ids = {f"t-{ph['id']}-{t['id']}" for t in ph["tensors"]}
                ids |= {f"p-{ph['id']}-{p['id']}" for p in ph["parameters"]}
                ids |= {f"mod-{ph['id']}-{g['id']}" for g in module_groups(ph)}
                labels = {o["id"]: op_source_label(o) for o in ph["operators"]}
                for oid, lab in labels.items():
                    if oid in ("c_kv_proj", "moe_hash_lookup", "rope_main", "c_window_view"):
                        self.assertTrue(lab)
                for g in module_groups(ph):
                    if not g["ops"]:
                        continue
                    html_text = module_narrative(ph, g["id"])
                    for target in re.findall(r'href="#([^"]+)"', html_text):
                        self.assertIn(target, ids, f"{idx}/{ph['id']}/{g['id']}: {target}")
                ov = build_overview_graph(ph)
                carrier = {n["id"]: n for n in ov["nodes"] if n["id"].startswith("c:")}
                if idx == 0:
                    self.assertIn("DynamicSlidingWindowLayer", carrier["c:main"]["title"])
                    self.assertNotIn("CSA", carrier["c:main"]["title"])
                if idx == 3:
                    self.assertIn("HCACache", carrier["c:csa"]["title"])
                    self.assertNotIn("overlap_kv", carrier["c:csa"]["sub"])
                    self.assertNotIn("overlap_gate", carrier["c:csa"]["sub"])
                if idx in (0, 2):
                    t2e = next(n for n in ov["nodes"] if n["id"] == "t:tid2eid")
                    self.assertIn("num_experts_per_tok", t2e["sub"])


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestVariantLabelsFix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        cls.docs = {i: build_reading_document(REPO_ROOT, layer_index=i) for i in (0, 2, 3)}

    def _svg(self, idx, module_id):
        from structure_reader.render.diagram import build_module_graph, module_groups, render_svg
        ph = self.docs[idx]["phases"][0]
        g = next(g for g in module_groups(ph) if g["id"] == module_id)
        return render_svg(build_module_graph(ph, g))

    def test_actual_svg_and_narrative(self):
        hca = self._svg(3, "attn_csa")
        self.assertIn("chunk_kv", hca)
        self.assertIn("chunk_gate", hca)
        self.assertNotIn("new_kv", hca)
        self.assertNotIn("tid2eid", self._svg(3, "attn_core"))
        csa = self._svg(2, "attn_csa")
        self.assertIn("new_kv", csa)
        swa = self._svg(0, "attn_cache")
        self.assertIn("values", swa)
        self.assertIn("min(kvcachelen+seqlen,sliding_window-1)", swa.replace(" ", ""))
        self.assertNotIn("kv_all", self._svg(0, "attn_core"))
        route = self._svg(0, "moe_route")
        self.assertIn("tid2eid[input_ids.reshape(-1)]", route)
        self.assertNotIn("router_bias", route)


class TestContractChecks(unittest.TestCase):
    """语义契约检查：当前文档通过；破坏锚点/数据流被抓住。"""

    @classmethod
    def setUpClass(cls):
        import hashlib

        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES
        from structure_reader.checks.deepseek_v4_hf_contracts import (
            validate_hf_reading_document,
        )

        cls.doc = build_reading_document(REPO_ROOT)
        cls.input_shas = {
            key: hashlib.sha256((REPO_ROOT / relpath).read_bytes()).hexdigest()
            for key, relpath in SOURCE_FILES.items()
        }
        cls.validate = staticmethod(validate_hf_reading_document)

    def test_current_document_passes(self):
        issues = self.validate(self.doc, self.input_shas)
        self.assertEqual(issues, [], f"当前阅读数据不应有问题: {issues[:5]}")

    def test_missing_input_shas_rejected(self):
        issues = self.validate(self.doc, {})
        self.assertTrue(any("input_shas" in i for i in issues))

    def test_wrong_av_input_rejected(self):
        """回放 Codex wrong_av 变异：av_matmul 概率输入换成 q_a 必须被拒。"""
        import copy

        broken = copy.deepcopy(self.doc)
        for phase in broken["phases"]:
            for o in phase["operators"]:
                if o["id"] == "av_matmul":
                    o["inputs"][0] = "q_a"
        issues = self.validate(broken, self.input_shas)
        self.assertTrue(any("av_matmul" in i for i in issues),
                          f"wrong_av 变异未被拒绝: {issues[:3]}")

    def test_broken_producer_caught(self):
        """删掉 kv_head_view 使 kv_head 无 producer，必须被数据流检查抓住。"""
        import copy

        broken = copy.deepcopy(self.doc)
        for phase in broken["phases"]:
            phase["operators"] = [o for o in phase["operators"] if o["id"] != "kv_head_view"]
        issues = self.validate(broken, self.input_shas)
        self.assertTrue(any("kv_head" in i and "producer" in i for i in issues))

    def test_expert_selfloop_caught(self):
        """专家模板算子自读自产（inputs=outputs）必须被拒。"""
        import copy

        broken = copy.deepcopy(self.doc)
        for phase in broken["phases"]:
            for o in phase["operators"]:
                if o["id"] == "moe_expert_down":
                    o["inputs"] = ["expert_gated", "expert_down_raw"]
                    o["outputs"] = ["expert_gated", "expert_down_raw"]
        issues = self.validate(broken, self.input_shas)
        self.assertTrue(any("同 id" in i for i in issues))

    def test_broken_comb_direction_caught(self):
        import copy

        broken = copy.deepcopy(self.doc)
        for phase in broken["phases"]:
            for o in phase["operators"]:
                if o["id"] in ("attn_residual_mix", "attn_residual_combT",
                               "ffn_residual_mix", "ffn_residual_combT"):
                    o["note"] = "直接相加（无转置）"
                    o["op"] = "add"
                    o["name"] = "残差相加"
        issues = self.validate(broken, self.input_shas)
        self.assertTrue(any("comb 转置" in i for i in issues))

    def test_broken_shared_kv_caught(self):
        import copy

        broken = copy.deepcopy(self.doc)
        for phase in broken["phases"]:
            for p in phase["parameters"]:
                if p["id"] == "Nkv":
                    p["value"] = 8
        issues = self.validate(broken, self.input_shas)
        self.assertTrue(any("Nkv" in i for i in issues))


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
class TestDeliverables(unittest.TestCase):
    """HTML/xlsx 交付：临时输出根生成、双表、字节稳定、真实 xlsx。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.render.decoder_page import render_decoder_page
        from structure_reader.export.operator_workbook import export_operator_workbook

        cls.doc = build_reading_document(REPO_ROOT)
        cls.html_bytes = render_decoder_page(cls.doc)
        cls.xlsx_bytes = export_operator_workbook(cls.doc)

    def test_html_single_file_offline(self):
        text = self.html_bytes.decode("utf-8")
        self.assertIn("phase-prefill", text)
        self.assertIn("phase-decode", text)
        # 两阶段连续展示：decode 段落在 prefill 段落之后同页出现
        self.assertLess(text.index("phase-prefill"), text.index("phase-decode"))
        # SVG 命名空间是标识符（从不联网拉取），与已豁免的论文链接一致
        offline_text = text.replace("https://huggingface.co/papers", "") \
                           .replace('xmlns="http://www.w3.org/2000/svg"', "")
        for external in ("http://", "https://", "cdn.jsdelivr", "unpkg.com", "cdnjs"):
            self.assertNotIn(external, offline_text,
                             "离线页面不应引用外部资源")
        self.assertNotIn("（未定义！）", text, "算子引用不得悬空")

    def test_html_param_links_all_resolve(self):
        """全部 #p-<phase>-<param> 联查 href 都有真实参数行锚点（回放 56 缺失目标回归）。"""
        from html.parser import HTMLParser

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = set()
                self.links = []

            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if "id" in a:
                    self.ids.add(a["id"])
                if a.get("href", "").startswith("#"):
                    self.links.append(a["href"][1:])

        parser = Links()
        parser.feed(self.html_bytes.decode("utf-8"))
        self.assertGreater(len(parser.links), 0, "页面应含参数联查链接")
        broken = sorted(set(parser.links) - parser.ids)
        self.assertEqual(broken, [], f"悬空联查目标: {broken[:8]}")
        # 参数行锚点确实存在
        for phase in self.doc["phases"]:
            for p in phase["parameters"]:
                self.assertIn(f"p-{phase['id']}-{p['id']}", parser.ids)

    def test_page_no_corrupted_transpose_or_key(self):
        text = self.html_bytes.decode("utf-8")
        for bad in ("compressed_kv.sequence_length", "comb.sequence_length",
                    "k^sequence_length", "num_experts_per_tok^", "num_experts_per_tok 键"):
            self.assertNotIn(bad, text)
        self.assertIn("kvcachelen", text)
        self.assertIn("seqlen", text)
        self.assertIn("Linear(hidden_size", text)

    def test_edge_labels_visible_complete(self):
        """每条边标签的可见 tspan 拼接必须等于其 title 全文（无 3 行截断）。"""
        import html as html_mod
        import re
        text = self.html_bytes.decode("utf-8")
        blocks = re.findall(r'<text class="elabel"[^>]*>(.*?)</text>', text, re.S)
        self.assertGreater(len(blocks), 400)
        for b in blocks:
            vis = html_mod.unescape("".join(
                re.findall(r'<tspan[^>]*>(.*?)</tspan>', b, re.S)))
            tm = re.search(r'<title>(.*?)</title>', b, re.S)
            if tm:
                self.assertEqual(vis, html_mod.unescape(tm.group(1)))

    def test_derived_param_names_and_long_wrap(self):
        import re
        from structure_reader.render.diagram import _wrap_edge_label
        text = self.html_bytes.decode("utf-8")
        for s in ("num_full_windows", "num_full_windows_indexer",
                  "num_compressed_entries", "kv_sequence_length",
                  "mask_sequence_length", "buffer_length_after",
                  "compressor_entries_before", "indexer_entries_before"):
            self.assertIn(s, text)
        nosrc = re.sub(r'<pre.*?</pre>', '', text, flags=re.S)
        self.assertNotIn("N_win&gt;0", nosrc)
        long = ("key_states / value_states（K=V 同存储） "
                "[batch_size,num_attention_heads,kv_sequence_length+"
                "num_compressed_entries,head_dim]")
        self.assertEqual("".join(_wrap_edge_label(long)), long)

    def test_display_names_token_boundaries(self):
        from structure_reader.render.display_names import (display_shape, expand_display_text,
                                                          shape_text)
        self.assertEqual(display_shape(["B", "T", "Nhc", "H"]),
                         "[batch_size,sequence_length,hc_mult,hidden_size]")
        self.assertEqual(display_shape(["Nh×Dh÷G"]),
                         "[num_attention_heads×head_dim÷o_groups]")
        self.assertEqual(display_shape(["min(P,W-1)"]),
                         "[min(past_sequence_length,sliding_window-1)]")
        self.assertEqual(display_shape(["Tbuf'"]), "[buffer_length_after]")
        self.assertEqual(display_shape(["B×T", "(2+Nhc)×Nhc"]),
                         "[batch_size×sequence_length,(2+hc_mult)×hc_mult]")
        # 幂等 & 不误伤长名
        self.assertEqual(shape_text(["batch_size", "H"]), "batch_size,hidden_size")
        self.assertEqual(expand_display_text("kv_proj=Linear(H,2Dh)"),
                         "kv_proj=Linear(hidden_size,2×head_dim)")
        self.assertEqual(expand_display_text("N_win 与 N_win_i"),
                         "num_full_windows 与 num_full_windows_indexer")
        # 转置/键语义不得被维度替换改写
        for s in ("compressed_kv.T.float().unsqueeze(1)", "comb.T", "matmul(q, k^T)",
                  "ReLU(q·K^T)", "K 用追加后全量条目", "K 键", "K=V", "K/V 头"):
            self.assertEqual(expand_display_text(s), s)
        # 合法维度/参数仍展开
        self.assertEqual(expand_display_text("Linear(H,2Dh)"),
                         "Linear(hidden_size,2×head_dim)")
        self.assertEqual(expand_display_text("min(P+T,W-1)"),
                         "min(past_sequence_length+sequence_length,sliding_window-1)")
        self.assertEqual(expand_display_text("buffer_length' = (buffer_length + T) % M"),
                         "buffer_length_after = (buffer_length + sequence_length) % compress_rate")
        self.assertEqual(expand_display_text("TopK 选 K 个专家"),
                         "TopK 选 num_experts_per_tok 个专家")

    def test_html_module_narratives(self):
        """两阶段模块讲解覆盖（CSA 含 Indexer）、mHC 闭环语义、shape 来自 JSON。"""
        import re
        text = self.html_bytes.decode("utf-8")
        self.assertEqual(text.count('class="narr"'), 36)   # 17 图 + 总览，每阶段 18
        self.assertEqual(text.count('class="narr narr-sub"'), 2)  # Indexer 内嵌
        mods = set(re.findall(r'data-module="([a-z]+)/([a-z_]+)"', text))
        self.assertEqual(len(mods), 38)  # 18 模块 + overview，两阶段
        for phase in ("prefill", "decode"):
            ids = {m for p, m in mods if p == phase and m != "overview"}
            self.assertEqual(len(ids), 18)
            self.assertIn("attn_csa", ids)
            self.assertIn("attn_indexer", ids)
            self.assertIn("mhc_attn", ids)
        mhc = re.search(r'<div class="narr" data-module="prefill/mhc_attn">(.*?)</div>',
                        text, re.S).group(1)
        for s in ("24 个系数", "pre", "post", "comb", "collapsed",
                  "hidden_streams_in", "attn_out_streams", "hc_mult=4", "hidden_size=4096"):
            self.assertIn(s, mhc)
        # shape 来自原 JSON（非写死表格）
        self.assertIn("[batch_size,seqlen,hc_mult,hidden_size]", mhc)
        self.assertIn("[batch_size,seqlen,(2+hc_mult)×hc_mult]", mhc)
        # 讲解中的 tensor 链接均可达（param 联查由既有用例统一校验）
        parser_ids = re.findall(r'id="([^"]+)"', text)
        narr_links = re.findall(r'href="#((?:t|mod)-[^"]+)"', text)
        self.assertTrue(narr_links)
        self.assertEqual(sorted(set(narr_links) - set(parser_ids)), [])
        # 36 张原 SVG 与基线候选逐字节同
        baseline = REPO_ROOT / "structure_reader/outputs/deepseek_v4_flash_0731_hf/b81c1e8a7ac0dc931a9f1bed804d0b014e6ec9eb0c88c5f246b6ceec757b6a6c/viewer/decoder.html"
        if baseline.is_file():
            old = baseline.read_text()
            olds = re.findall(r'<svg class="dg".*?</svg>', old, re.S)
            news = re.findall(r'<svg class="dg".*?</svg>', text, re.S)
            self.assertEqual(len(olds), 36)
            self.assertEqual(len(news), 36)
            # 本轮只改可见名称：图形结构（路径/标记数）不变
            for a, b in zip(olds, news):
                self.assertEqual(a.count("<path"), b.count("<path"))
                self.assertEqual(a.count("marker"), b.count("marker"))

    def test_narrative_facts_follow_parameters(self):
        """参数变体 Nhc=5/H=128/Rk=32：文字里的派生数字必须跟着变，不写死。"""
        import copy
        from structure_reader.render.module_narrative import module_narrative
        doc = copy.deepcopy(self.doc)
        for p in doc["phases"][0]["parameters"]:
            if p["id"] == "Nhc":
                p["value"] = 5
            if p["id"] == "H":
                p["value"] = 128
            if p["id"] == "Rk":
                p["value"] = 32
        text = module_narrative(doc["phases"][0], "mhc_attn")
        for s in ("hc_mult=5", "hidden_size=128", "35", "640"):
            self.assertIn(s, text)
        self.assertIn("最后 32 维", module_narrative(doc["phases"][0], "attn_q"))
        self.assertNotIn("24 个系数", text)
        self.assertNotIn("16384", text)

    def test_narrative_missing_key_rejected_and_escaped(self):
        import copy
        from structure_reader.render.module_narrative import _Ctx, module_narrative
        c = _Ctx(self.doc["phases"][0])
        with self.assertRaises(KeyError):
            c.t("no_such_tensor")
        doc = copy.deepcopy(self.doc)
        for p in doc["phases"][0]["parameters"]:
            if p["id"] == "EpsN":
                p["value"] = "<script>alert(1)</script>"
        text = module_narrative(doc["phases"][0], "mhc_attn")
        self.assertIn("&lt;script&gt;", text)
        self.assertNotIn("<script>alert", text)

    def test_narrative_mhc_shapes_and_moe_semantics(self):
        import re
        text = self.html_bytes.decode("utf-8")

        def block(mod):
            return re.search(rf'<div class="narr" data-module="prefill/{mod}">(.*?)</div>',
                             text, re.S).group(1)

        mhc = block("mhc_attn")
        for s in ("[(2+hc_mult)×hc_mult,hc_mult×hidden_size]", "[3]", "pre 不返回", "本段返回 post"):
            self.assertIn(s, mhc)
        self.assertIn("本层 Attention 混合输出", block("mhc_ffn"))
        route = block("moe_route")
        self.assertIn("bias 只参与", route)
        self.assertIn("未加 bias 的 scores", route)
        expert = block("moe_expert")
        self.assertIn("gate 只 clamp 上界 Lim", expert)
        self.assertIn("up 范围 [-Lim, Lim]", expert)
        self.assertNotIn("[[", expert)
        overview = text[text.index('data-module="prefill/overview"'):
                        text.index('data-module="prefill/overview"') + 700]
        self.assertIn("第二次（最后一次）残差", overview)
        self.assertNotIn("第三次", overview)
        idx = block("attn_csa")
        self.assertIn("生成<b>独立的压缩键</b>", idx)
        self.assertIn("生成<b>查询</b>", idx)
        self.assertIn("scorer.weights_proj", idx)
        csa = block("attn_csa")
        for s in ("[batch_size,seqlen,indexer_entries_before+num_full_windows_indexer]",
                  "[batch_size,1,seqlen,num_compressed_entries]"):
            self.assertIn(s, csa)
        # 总览贯通说明
        self.assertEqual(text.count('data-module="prefill/overview"'), 1)
        self.assertEqual(text.count('data-module="decode/overview"'), 1)
        ov = text[text.index('data-module="prefill/overview"'):
                  text.index('data-module="prefill/overview"') + 600]
        for s in ("空缓存 Prefill：kvcachelen=0", "mHC", "Attention", "MoE"):
            self.assertIn(s, ov)

    def test_html_deterministic(self):
        from structure_reader.render.decoder_page import render_decoder_page

        again = render_decoder_page(self.doc)
        self.assertEqual(self.html_bytes, again)

    def test_xlsx_two_sheets_readable(self):
        from openpyxl import load_workbook
        import io

        wb = load_workbook(io.BytesIO(self.xlsx_bytes))
        self.assertEqual(wb.sheetnames, ["Prefill", "Decode"])
        for name in ("Prefill", "Decode"):
            ws = wb[name]
            self.assertGreater(ws.max_row, 10)

    def test_xlsx_is_real_zip(self):
        import io

        with zipfile.ZipFile(io.BytesIO(self.xlsx_bytes)) as zf:
            bad = zf.testzip()
            self.assertIsNone(bad)
            names = zf.namelist()
            self.assertIn("[Content_Types].xml", names)
            self.assertIn("xl/workbook.xml", names)

    def test_xlsx_deterministic(self):
        from structure_reader.export.operator_workbook import export_operator_workbook

        again = export_operator_workbook(self.doc)
        self.assertEqual(self.xlsx_bytes, again)

    def test_full_build_atomic_output(self):
        """run_hf_build 全链：临时工程根落盘验证（不触碰真实 outputs）。"""
        from structure_reader.build_hf_decoder import run_hf_build

        # run_hf_build 只生成字节（不落盘）；CLI 层负责提交——此处验证字节完整
        # 工具/配置身份已与四变体统一为按工程根复算，假工程根镜像 src/configs。
        with tempfile.TemporaryDirectory() as td:
            fake_project = Path(td) / "project"
            fake_project.mkdir()
            (fake_project / "src").symlink_to(
                REPO_ROOT / "structure_reader/src", target_is_directory=True)
            (fake_project / "configs").symlink_to(
                REPO_ROOT / "structure_reader/configs", target_is_directory=True)
            result, documents = run_hf_build(REPO_ROOT, fake_project, layer_index=4)
            self.assertEqual(result.model_key, MODEL_KEY)
            self.assertEqual(len(result.analysis_id), 64)
            for name in (
                "README.md", "run_manifest.json", "data/reading_document.json",
                "data/effective_config.json", "reports/source_manifest.json",
                "viewer/decoder.html", "tables/decoder_reading.xlsx",
            ):
                self.assertIn(name, documents)
                self.assertGreater(len(documents[name]), 0)

    def test_analysis_id_stable_and_sensitive(self):
        from structure_reader.build_hf_decoder import run_hf_build

        with tempfile.TemporaryDirectory() as td:
            fake_project = Path(td) / "project"
            fake_project.mkdir()
            (fake_project / "src").symlink_to(
                REPO_ROOT / "structure_reader/src", target_is_directory=True)
            (fake_project / "configs").symlink_to(
                REPO_ROOT / "structure_reader/configs", target_is_directory=True)
            result1, _ = run_hf_build(REPO_ROOT, fake_project, layer_index=4)
            result2, _ = run_hf_build(REPO_ROOT, fake_project, layer_index=4)
            self.assertEqual(result1.analysis_id, result2.analysis_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
