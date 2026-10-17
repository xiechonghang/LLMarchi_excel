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

    def test_wrong_layer_rejected(self):
        """非 CSA+TopK 组合具名拒绝（不拼造组合）。"""
        with self.assertRaises(ValueError):
            self.build(workspace_root=REPO_ROOT, layer_index=0)  # SWA+hash
        with self.assertRaises(ValueError):
            self.build(workspace_root=REPO_ROOT, layer_index=3)  # HCA+moe


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
                self.assertEqual(combt["shape"], ["B", "T", "Nhc", "Nhc"])
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
            self.assertEqual(shapes["indexer_topk"], ["B", "T", "min(Ki,Tci+N_win_i)"])
            self.assertEqual(shapes["cache_keys"][2], "min(P,W-1)")
            self.assertEqual(shapes["cache_keys_post"][2], "min(P+T,W-1)")
            self.assertIn("min(P, W-1) + T", params["Lkv"]["expression"])
            self.assertIn("min(P, W-1) + T", params["Lmask"]["expression"])
            self.assertIn("Nwin_i", params, "Indexer 系列闭窗数独立成参")

    def test_attention_output_layout_before_after_rope(self):
        """AV 输出与逆 RoPE 前后都是 [B,T,Nh,Dh]（transpose 配对为 layout 修正）。"""
        for phase in self.doc["phases"]:
            shapes = {t["id"]: t["shape"] for t in phase["tensors"]}
            for tid in ("attn_head_out", "attn_rope_inv"):
                self.assertEqual(shapes[tid], ["B", "T", "Nh", "Dh"],
                                 f"{phase['id']}: {tid} layout 应为 [B,T,Nh,Dh]")
            self.assertEqual(shapes["i_q"], ["B", "T", "Ih", "Id"],
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
            self.assertEqual(shapes["routed_out"], ["B×T", "H"])
            self.assertEqual(shapes["routed_view"], ["B", "T", "H"])
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
            self.assertEqual(shapes["i_scores"], ["B", "T", "Tci+N_win_i"])
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
            self.assertEqual(shapes["position_ids"], ["1", "T"])
            self.assertEqual(shapes["cos_q"], ["1", "T", "Rk÷2"])
            self.assertEqual(shapes["sin_q"], ["1", "T", "Rk÷2"])

    def test_prefill_vs_decode_cache_premise(self):
        notes = {p["id"]: p["cache_notes"] for p in self.doc["phases"]}
        prefill_blob = " ".join(notes["prefill"])
        decode_blob = " ".join(notes["decode"])
        self.assertIn("空", prefill_blob)
        self.assertIn("历史", decode_blob)
        t_prefill = next(p for p in self.doc["phases"][0]["parameters"] if p["id"] == "P")
        t_decode = next(p for p in self.doc["phases"][1]["parameters"] if p["id"] == "P")
        self.assertEqual(t_prefill["value"], 0)
        self.assertIsNone(t_decode["value"])
        t1 = next(p for p in self.doc["phases"][1]["parameters"] if p["id"] == "T")
        self.assertEqual(t1["value"], 1)

    def test_no_internal_unanalyzed_claimed_analyzed(self):
        """本样板全算子 analyzed；黑盒机制不在文档中使用（如实声明于 coverage）。"""
        for phase in self.doc["phases"]:
            for o in phase["operators"]:
                self.assertEqual(o["analysis_status"], "analyzed")


@unittest.skipUnless(_workspace_ready(), "缺少只读 HF 输入；如实跳过")
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
        for external in ("http://", "https://", "cdn.jsdelivr", "unpkg.com", "cdnjs"):
            self.assertNotIn(external, text.replace("https://huggingface.co/papers", ""),
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
        with tempfile.TemporaryDirectory() as td:
            fake_project = Path(td)
            (fake_project / "configs" / "models").mkdir(parents=True)
            (fake_project / "configs" / "models" / f"{MODEL_KEY}.json").write_bytes(
                (REPO_ROOT / "structure_reader/configs/models" / f"{MODEL_KEY}.json").read_bytes()
            )
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
            fake_project = Path(td)
            (fake_project / "configs" / "models").mkdir(parents=True)
            (fake_project / "configs" / "models" / f"{MODEL_KEY}.json").write_bytes(
                (REPO_ROOT / "structure_reader/configs/models" / f"{MODEL_KEY}.json").read_bytes()
            )
            result1, _ = run_hf_build(REPO_ROOT, fake_project, layer_index=4)
            result2, _ = run_hf_build(REPO_ROOT, fake_project, layer_index=4)
            self.assertEqual(result1.analysis_id, result2.analysis_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
