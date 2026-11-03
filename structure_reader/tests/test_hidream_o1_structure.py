"""HiDream-O1-Image 单次 forward 的结构、场景、契约与产物回归。

只测当前源码行为，不读旧 outputs、不依赖既有候选、不导入/运行 HF 模型、不加载
权重。测试要发现的是「把其它模型事实套进来」或「把外部采样混进单次网络」这类
真实错误，不是照实现复制断言：

- 身份：官方 HF revision / model_type / architectures、官方 GitHub commit；
- 层组：decoder 36 层（0..35）与 ViT 27 blocks（0..26）是两个独立索引空间；
- 统一主干：GQA 32/8、逐头 QK-norm、MRoPE 只作用 q/k、两趟注意力 + index_copy、
  两个 RMSNorm 站点 + 标量残差、SwiGLU 12288、条件 DeepStack、无 KV cache/MoE/adaLN；
- 视觉塔：LayerNorm、qkv 合并投影（bias）、2D rotary、非因果分图注意力、GELU-tanh
  4304、block 8/16/24 的 DeepStack tap；
- 整网：终点 x_pred（不是 logits）、无 lm_head/VAE/独立文本编码器/采样调度、
  视觉支路条件执行、时间步经 tms token 嵌入替换；
- 场景：t2i 1024/2048、单参考图编辑、符号场景的独立期望值；
- 契约：改动结构/网络后检查必须拒绝；
- 产物：真实 HTML 与单 sheet 工作簿（两组、活公式、成本待提供）。

运行：
  cd structure_reader && PYTHONPATH=src python3 tests/test_hidream_o1_structure.py
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openpyxl import load_workbook  # noqa: E402

from structure_reader.build_hf_decoder import run_hf_build  # noqa: E402
from structure_reader.checks.hidream_o1_hf_contracts import (  # noqa: E402
    validate_hf_layer_structure,
    validate_hf_network_overview,
    validate_hf_reading_document,
)
from structure_reader.extract.hf_layer_structures import build_structures_container  # noqa: E402
from structure_reader.extract.hidream_o1_hf import (  # noqa: E402
    build_block_structure,
    build_layer_documents,
    group_by_id,
)
from structure_reader.extract.hidream_o1_network import build_network_overview  # noqa: E402
from structure_reader.export.hidream_o1_workbook import (  # noqa: E402
    build_structures_workbook_spec,
    export_structures_workbook,
)
from structure_reader.render.hf_model_profile import (  # noqa: E402
    HIDREAM_O1_PAGE_PROFILE,
    QWEN38FLASH_PAGE_PROFILE,
    profile_for,
)
from structure_reader.render.hidream_o1_source_labels import MODELING_FILE  # noqa: E402
from structure_reader.rules import hidream_o1_hf as rules  # noqa: E402
from structure_reader.rules.hidream_o1_hf import (  # noqa: E402
    HIDREAM_BIND_KEYS,
    HIDREAM_SCENARIOS,
    MODEL_REPOSITORY,
    SOURCE_FILES,
    layer_group_assignments,
    load_hf_model_values,
    verify_source_fingerprints,
)

MODEL_KEY = "hidream_o1_image_hf"


def _values():
    return load_hf_model_values(REPO_ROOT, model_key=MODEL_KEY)


def _structure(group_id: str) -> dict:
    structure, _ = build_layer_documents(REPO_ROOT, group_id=group_id, model_key=MODEL_KEY)
    return structure


def _document(group_id: str) -> dict:
    _, document = build_layer_documents(REPO_ROOT, group_id=group_id, model_key=MODEL_KEY)
    return document


def _network() -> dict:
    return build_network_overview(REPO_ROOT, model_key=MODEL_KEY)


def _container() -> dict:
    return build_structures_container(
        REPO_ROOT, model_key=MODEL_KEY,
        values_loader=lambda root, key: load_hf_model_values(root, model_key=key),
        groups_fn=layer_group_assignments,
        build_variant_fn=lambda root, group, key: build_layer_documents(
            root, group_id=group["id"], model_key=key))


def _ops(block: dict) -> dict:
    return {op["id"]: op for op in block["operators"]}


def _tensors(block: dict) -> dict:
    return {t["id"]: t for t in block["tensors"]}


def _params(block: dict) -> dict:
    return {p["id"]: p for p in block["parameters"]}


def _shas() -> dict:
    return verify_source_fingerprints(REPO_ROOT)


class IdentityTest(unittest.TestCase):
    def test_official_repository_identity(self):
        self.assertEqual(MODEL_REPOSITORY["repo_id"], "HiDream-ai/HiDream-O1-Image")
        self.assertEqual(MODEL_REPOSITORY["revision"],
                         "0b0901d99f200389e138c61946af1185f5f49a13")
        self.assertEqual(MODEL_REPOSITORY["model_type"], "qwen3_vl")
        self.assertEqual(MODEL_REPOSITORY["architectures"],
                         ["Qwen3VLForConditionalGeneration"])
        self.assertEqual(MODEL_REPOSITORY["code_revision"],
                         "2c2d29ff729e48f33e41f49edfdbd81d5ac103b4")

    def test_checkpoint_version_and_code_revision(self):
        values = _values()
        self.assertEqual(values.checkpoint_transformers_version, "4.57.0.dev0")
        self.assertEqual(values.source_revision,
                         "2c2d29ff729e48f33e41f49edfdbd81d5ac103b4")
        self.assertEqual(values.implementation, "transformers")
        self.assertIn("HiDream", values.display_name)

    def test_text_config_facts(self):
        values = _values()
        self.assertEqual((values.hidden_size, values.num_hidden_layers,
                          values.num_attention_heads, values.num_key_value_heads,
                          values.head_dim, values.intermediate_size),
                         (4096, 36, 32, 8, 128, 12288))
        self.assertEqual(values.mrope_section, (24, 20, 20))
        self.assertTrue(values.mrope_interleaved)
        self.assertEqual(values.rope_theta, 5000000)
        self.assertEqual(values.vocab_size, 151936)
        self.assertFalse(values.tie_word_embeddings)
        self.assertFalse(values.attention_bias)

    def test_vision_config_facts(self):
        values = _values()
        self.assertEqual((values.vision_depth, values.vision_hidden_size,
                          values.vision_heads, values.vision_head_dim,
                          values.vision_intermediate_size, values.vision_patch_size,
                          values.vision_temporal_patch_size, values.spatial_merge_size,
                          values.vision_out_hidden_size),
                         (27, 1152, 16, 72, 4304, 16, 2, 2, 4096))
        self.assertEqual(values.deepstack_visual_indexes, (8, 16, 24))
        self.assertEqual(values.num_position_embeddings, 2304)
        self.assertEqual(values.vision_hidden_act, "gelu_pytorch_tanh")

    def test_pixel_patch_and_timestep_facts(self):
        values = _values()
        self.assertEqual((values.pixel_patch_size, values.pixel_in_dim,
                          values.pixel_pca_dim, values.pixel_out_dim),
                         (32, 3072, 1024, 3072))
        self.assertEqual(values.timestep_frequency_embedding_size, 256)
        self.assertEqual(values.timestep_token_num, 1)
        self.assertEqual(values.tms_token_id, 151673)
        self.assertEqual((values.image_token_id, values.video_token_id,
                          values.vision_start_token_id), (151655, 151656, 151652))

    def test_external_sampling_boundary_recorded_only(self):
        values = _values()
        self.assertEqual((values.external_steps_full, values.external_steps_dev),
                         (50, 28))
        self.assertEqual(values.external_guidance_full, 5.0)
        self.assertEqual(values.external_shift_full, 3.0)

    def test_source_fingerprints_verify(self):
        shas = _shas()
        self.assertEqual(set(shas), set(SOURCE_FILES))
        for value in shas.values():
            self.assertEqual(len(value), 64)

    def test_reject_wrong_model_type(self):
        raw = json.loads((REPO_ROOT / SOURCE_FILES["config.json"]).read_text(encoding="utf-8"))
        raw["model_type"] = "hidream_i1"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models/HiDream/HiDream-O1-Image").mkdir(parents=True)
            (root / "models/HiDream/HiDream-O1-Image/config.json").write_text(
                json.dumps(raw), encoding="utf-8")
            with self.assertRaises(rules.HfConfigError):
                load_hf_model_values(root, model_key=MODEL_KEY)

    def test_reject_other_architecture(self):
        raw = json.loads((REPO_ROOT / SOURCE_FILES["config.json"]).read_text(encoding="utf-8"))
        raw["architectures"] = ["HiDreamO1ForConditionalGeneration"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models/HiDream/HiDream-O1-Image").mkdir(parents=True)
            (root / "models/HiDream/HiDream-O1-Image/config.json").write_text(
                json.dumps(raw), encoding="utf-8")
            with self.assertRaises(rules.HfConfigError):
                load_hf_model_values(root, model_key=MODEL_KEY)


class GroupTest(unittest.TestCase):
    def test_two_groups_with_independent_index_spaces(self):
        groups = {g["id"]: g for g in layer_group_assignments(_values())}
        self.assertEqual(set(groups), {"uit-decoder", "vit-tower"})
        self.assertEqual(groups["uit-decoder"]["layer_indices"], list(range(36)))
        self.assertEqual(groups["vit-tower"]["layer_indices"], list(range(27)))
        # 不把 ViT block 当 decoder 的第 36..62 层
        self.assertNotIn(36, groups["vit-tower"]["layer_indices"])
        self.assertEqual(groups["vit-tower"]["representative_layer_index"], 0)

    def test_group_attention_and_mlp_types(self):
        groups = {g["id"]: g for g in layer_group_assignments(_values())}
        self.assertEqual(groups["uit-decoder"]["attention_type"], "hybrid_causal_full")
        self.assertEqual(groups["uit-decoder"]["moe_type"], "dense_swiglu")
        self.assertEqual(groups["vit-tower"]["attention_type"], "bidirectional_vision")
        self.assertEqual(groups["vit-tower"]["moe_type"], "dense_gelu_mlp")

    def test_layer_index_alone_is_ambiguous(self):
        with self.assertRaises(ValueError):
            build_layer_documents(REPO_ROOT, layer_index=0, model_key=MODEL_KEY)

    def test_unknown_group_rejected(self):
        with self.assertRaises(ValueError):
            group_by_id(_values(), "decoder")
        with self.assertRaises(ValueError):
            build_block_structure(_values(), "hidream-i1-dit")


class DecoderBlockTest(unittest.TestCase):
    def setUp(self):
        self.structure = _structure("uit-decoder")
        self.block = self.structure["block"]
        self.ops = _ops(self.block)
        self.tensors = _tensors(self.block)
        self.params = _params(self.block)
        self.derivations = {d["id"]: d for d in self.block["derivations"]}

    def test_block_kind_and_layer(self):
        self.assertEqual(self.block["kind"], "uit_decoder")
        self.assertEqual(self.block["layer_index"], 0)
        self.assertEqual(self.block["attention_type"], "hybrid_causal_full")

    def test_gqa_projections_and_qk_norm(self):
        for oid, out in (("d_q_proj", "d_q_lin"), ("d_k_proj", "d_k_lin"),
                         ("d_v_proj", "d_v_lin")):
            self.assertEqual(self.ops[oid]["outputs"], [out])
        self.assertEqual(self.tensors["d_w_q"]["shape"], ["Hq", "H"])
        self.assertEqual(self.tensors["d_w_k"]["shape"], ["Hkv", "H"])
        self.assertEqual(self.tensors["d_w_v"]["shape"], ["Hkv", "H"])
        self.assertEqual(self.ops["d_q_norm"]["op"], "rmsnorm")
        self.assertEqual(self.ops["d_k_norm"]["op"], "rmsnorm")
        # V 没有 norm、没有 RoPE
        self.assertFalse(any(op["op"] == "rmsnorm" and "d_v" in op["inputs"]
                             for op in self.block["operators"]))
        self.assertNotIn("v_rope", self.ops)

    def test_rope_applies_to_q_and_k_only(self):
        self.assertEqual(self.ops["d_q_rope"]["outputs"], ["d_q"])
        self.assertEqual(self.ops["d_k_rope"]["outputs"], ["d_k"])
        for oid in ("d_q_rope", "d_k_rope"):
            self.assertIn("d_pos_cos", self.ops[oid]["inputs"])
            self.assertIn("d_pos_sin", self.ops[oid]["inputs"])

    def test_two_pass_attention_shapes_and_merge(self):
        self.assertEqual(self.tensors["d_score_ar"]["shape"], ["B", "Nh", "Nar", "Nar"])
        self.assertEqual(self.tensors["d_score_full"]["shape"], ["B", "Nh", "Stot", "Stot"])
        self.assertEqual(self.ops["d_merge_ar"]["op"], "index-copy")
        self.assertEqual(self.ops["d_merge_ar"]["outputs"], ["d_ctx"])
        self.assertIn("4D", self.ops["d_merge_ar"]["note"])
        self.assertIn("causal=True", self.ops["d_qk_ar"]["source_refs"][0]["note"])
        self.assertIn("causal=False", self.ops["d_qk_full"]["source_refs"][0]["note"])

    def test_two_norm_sites_and_scalar_residuals(self):
        self.assertEqual(self.ops["d_in_norm"]["op"], "rmsnorm")
        self.assertEqual(self.ops["d_post_norm"]["op"], "rmsnorm")
        self.assertEqual(self.ops["d_res_attn"]["inputs"], ["d_hidden_in", "d_attn"])
        self.assertEqual(self.ops["d_res_ffn"]["inputs"], ["d_h1", "d_ffn"])
        self.assertEqual(self.ops["d_res_ffn"]["op"], "add")

    def test_swiglu_dense_mlp(self):
        self.assertEqual(self.tensors["d_w_gate"]["shape"], ["Fi", "H"])
        self.assertEqual(self.tensors["d_w_down"]["shape"], ["H", "Fi"])
        self.assertEqual(self.ops["d_silu"]["op"], "silu")
        self.assertEqual(self.ops["d_swiglu_mul"]["op"], "mul")
        self.assertIn("没有 clamp", " ".join(
            op["note"] for op in self.block["operators"]))

    def test_no_cache_no_moe_no_adaln(self):
        self.assertFalse(any(t["kind"] == "cache" for t in self.block["tensors"]))
        ids = " ".join(self.ops)
        for token in ("cache", "moe", "expert", "router", "adaln", "timestep_add"):
            self.assertNotIn(token, ids)
        text = " ".join(op["note"] for op in self.block["operators"])
        self.assertIn("没有 KV cache", " ".join(self.block["cache_notes"]))
        self.assertIn("时间步不以 adaLN", " ".join(self.block["scope_notes"]))

    def test_deepstack_is_conditional(self):
        op = self.ops["d_deepstack_add"]
        self.assertEqual(op["inputs"][1:], ["d_visual_mask", "d_ds_embeds"])
        self.assertIn("Kref>0", op["note"])
        self.assertIn("0,1,2", op["note"])
        self.assertEqual(self.tensors["d_ds_embeds"]["shape"], ["Nvis", "H"])
        self.assertIsNone(self.params["Nvis"]["value"])
        derivation = {d["id"]: d for d in self.block["derivations"]}["N_vis"]
        self.assertEqual(derivation["expression"], "Svis // (Mv * Mv)")

    def test_derived_length_chain(self):
        self.assertEqual(self.derivations["S_pix"]["expression"],
                         "(Himg // Ppix) * (Wimg // Ppix)")
        self.assertEqual(self.derivations["S_gen"]["expression"], "S_pix + Sref")
        self.assertEqual(self.derivations["S_tot"]["expression"], "Ptxt + S_gen")
        self.assertEqual(self.derivations["N_ar"]["expression"], "Ptxt - 1")
        self.assertEqual(self.derivations["N_gt"]["expression"], "S_gen + 1")

    def test_input_domains(self):
        self.assertEqual(self.params["Himg"]["domain"], {"kind": "integer", "min": 32})
        self.assertEqual(self.params["Ptxt"]["domain"], {"kind": "integer", "min": 1})
        self.assertEqual(self.params["B"]["domain"], {"kind": "integer", "min": 1})


class VisionBlockTest(unittest.TestCase):
    def setUp(self):
        self.structure = _structure("vit-tower")
        self.block = self.structure["block"]
        self.ops = _ops(self.block)
        self.tensors = _tensors(self.block)
        self.params = _params(self.block)

    def test_layernorm_not_rmsnorm(self):
        self.assertEqual(self.ops["v_norm1"]["op"], "layernorm")
        self.assertEqual(self.ops["v_norm2"]["op"], "layernorm")
        self.assertFalse(any(op["op"] == "rmsnorm" for op in self.block["operators"]))

    def test_merged_qkv_with_bias(self):
        op = self.ops["v_qkv_proj"]
        self.assertEqual(op["weights"], ["v_w_qkv", "v_b_qkv"])
        self.assertEqual(self.tensors["v_w_qkv"]["shape"], ["Hv3", "Hv"])
        self.assertEqual(self.ops["v_qkv_split"]["outputs"], ["v_q", "v_k", "v_v"])
        self.assertEqual(self.params["Hv3"]["value"], 3456)

    def test_2d_rotary_and_non_causal_attention(self):
        self.assertEqual(self.ops["v_q_rope"]["op"], "rope")
        self.assertEqual(self.ops["v_k_rope"]["op"], "rope")
        self.assertNotIn("v_v_rope", self.ops)
        self.assertIn("非因果", self.ops["v_qk"]["note"])
        self.assertIn("cu_seqlens", self.ops["v_seg_split"]["note"])
        # 逐图分段：局部 shape，不是总长平方
        self.assertEqual(self.tensors["v_scores"]["shape"], ["Nhv", "S_i", "S_i"])
        self.assertEqual(self.tensors["v_probs"]["shape"], ["Nhv", "S_i", "S_i"])
        for tid in ("v_q_seg", "v_k_seg", "v_v_seg"):
            self.assertEqual(self.tensors[tid]["shape"], ["Nhv", "S_i", "Dhv"])
        self.assertEqual(self.tensors["v_ctx"]["shape"], ["Svis", "Nhv", "Dhv"])
        self.assertIn("v_cu_seqlens", self.tensors)
        # 一维 cu_seqlens：cumsum 后左 pad 0 → [Kref+1]（Kref=2 时 [3]）
        self.assertEqual(self.tensors["v_cu_seqlens"]["shape"], ["Kref+1"])
        self.assertEqual(self.ops["v_seg_split"]["outputs"],
                         ["v_q_seg", "v_k_seg", "v_v_seg"])
        self.assertEqual(self.ops["v_seg_concat"]["inputs"], ["v_ctx_seg"])
        self.assertEqual(self.ops["v_seg_concat"]["outputs"], ["v_ctx"])

    def test_attention_scale_is_sum_of_per_image_squares(self):
        # 两张各 576：真总量 10,616,832；若按总长 1152 平方会多一倍（21,233,664）
        attn = self.params["Sattn"]
        self.assertEqual(attn["expression"], "S_attn = Nhv * Kref * S_i * S_i")
        derivation = {d["id"]: d for d in self.block["derivations"]}["S_attn_elems"]
        self.assertEqual(derivation["expression"], "Nhv * Kref * S_i * S_i")
        self.assertEqual(derivation["guard"],
                         "(Kref * S_i == Svis) and (S_i % (Mv * Mv) == 0)")
        s_i = self.params["S_i"]
        self.assertIn("Kref * S_i == Svis", s_i["consistency"])
        self.assertIn("S_i % (Mv * Mv) == 0", s_i["consistency"])
        self.assertIn("Svis % (Mv * Mv) == 0", self.params["Svis"]["consistency"])
        self.assertEqual(self.ops["v_qk"]["parameter_refs"], ["Nhv", "S_i", "Dhv"])

    def test_gelu_tanh_mlp(self):
        self.assertEqual(self.ops["v_gelu"]["op"], "gelu-tanh")
        self.assertEqual(self.tensors["v_w_fc1"]["shape"], ["Fv", "Hv"])
        self.assertEqual(self.tensors["v_w_fc2"]["shape"], ["Hv", "Fv"])
        self.assertEqual(self.params["Fv"]["value"], 4304)

    def test_deepstack_tap_blocks(self):
        self.assertIn("8,16,24", self.ops["v_deepstack_tap"]["note"])
        self.assertEqual(self.ops["v_deepstack_tap"]["outputs"], ["v_tap"])
        self.assertEqual(self.ops["v_deepstack_tap"]["op"], "identity-read")

    def test_no_cache_and_conditional_execution(self):
        self.assertFalse(any(t["kind"] == "cache" for t in self.block["tensors"]))
        self.assertIn("Kref>0", " ".join(self.block["scope_notes"]))
        self.assertEqual(self.params["Nv"]["value"], 27)


class NetworkTest(unittest.TestCase):
    def setUp(self):
        self.network = _network()
        self.path = self.network["path"]
        self.ops = _ops(self.path)
        self.tensors = _tensors(self.path)
        self.params = _params(self.path)

    def test_schema_and_scope(self):
        self.assertEqual(self.network["schema"], "hf_network_overview.v1")
        scope = self.network["scope"]
        self.assertEqual(scope["entry_class"], "Qwen3VLForConditionalGeneration")
        self.assertEqual(scope["layer_count"], 36)
        self.assertEqual(scope["vision_block_count"], 27)
        self.assertTrue(scope["single_forward"])
        self.assertEqual(scope["deepstack_visual_indexes"], [8, 16, 24])

    def test_endpoint_is_x_pred_not_logits(self):
        self.assertIn("n_x_pred", self.tensors)
        self.assertEqual(self.ops["net_select_gen"]["outputs"], ["n_x_pred"])
        self.assertNotIn("net_lm_head", self.ops)
        self.assertNotIn("net_logits", self.ops)
        for op in self.path["operators"]:
            self.assertNotIn("lm_head", op["module_path"])

    def test_no_vae_or_text_encoder(self):
        modules = " ".join(op["module_path"] for op in self.path["operators"]).lower()
        for token in ("vae", "clip", "text_encoder"):
            self.assertNotIn(token, modules)
        self.assertIn("没有 VAE", " ".join(self.path["coverage_notes"]))
        self.assertIn("没有独立文本编码器", " ".join(self.path["coverage_notes"]))

    def test_external_sampling_boundary_not_executed(self):
        for token in ("scheduler", "guidance", "cfg", "sampling_step"):
            self.assertNotIn(token, " ".join(self.ops))
        boundary = self.network["scope"]["external_boundary"]
        self.assertEqual(boundary["sampling_steps_full"], 50)
        self.assertEqual(boundary["sampling_steps_dev"], 28)
        self.assertIn("不分析", boundary["note"])

    def test_vision_branch_is_conditional(self):
        for oid in ("net_vision_patch", "net_vision_pos", "net_vision_rope",
                    "net_vision_loop", "net_vmerge_norm", "net_vmerge_view",
                    "net_vmerge_fc1", "net_vmerge_gelu", "net_vmerge_fc2",
                    "net_ds_view", "net_ds_norm", "net_ds_fc1", "net_ds_gelu",
                    "net_ds_fc2", "net_cond_scatter"):
            self.assertIn("Kref>0", self.ops[oid]["note"], oid)

    def test_mergers_are_detailed_not_blackbox(self):
        for oid in ("net_vision_merge", "net_deepstack_merge"):
            self.assertNotIn(oid, self.ops)
        # 主线：Norm(1152) 在 view 前；DeepStack：view 在 Norm(4608) 后
        self.assertEqual(self.ops["net_vmerge_norm"]["op"], "layernorm")
        self.assertEqual(self.ops["net_vmerge_view"]["inputs"], ["n_vmerge_normed"])
        self.assertEqual(self.ops["net_vmerge_fc1"]["op"], "linear")
        self.assertEqual(self.ops["net_vmerge_gelu"]["op"], "gelu")
        self.assertEqual(self.ops["net_vmerge_fc2"]["op"], "linear")
        self.assertEqual(self.ops["net_ds_view"]["inputs"], ["n_vtaps"])
        self.assertEqual(self.ops["net_ds_norm"]["inputs"], ["n_ds_units"])
        self.assertEqual(self.ops["net_ds_norm"]["op"], "layernorm")
        for tid in ("n_vmerge_units", "n_vmerge_h", "n_vmerge_act",
                    "n_ds_units", "n_ds_normed", "n_ds_h", "n_ds_act"):
            self.assertIn(tid, self.tensors)
        self.assertEqual(self.tensors["n_vmerge_units"]["shape"], ["Svism", "Hv4"])
        self.assertEqual(self.tensors["n_ds_normed"]["shape"], ["Ndh", "Svism", "Hv4"])
        self.assertIn("×3", self.ops["net_ds_norm"]["note"])
        self.assertIn("8/16/24", self.ops["net_ds_fc2"]["note"])

    def test_timestep_enters_through_tms_token(self):
        op = self.ops["net_tms_substitute"]
        self.assertEqual(op["inputs"], ["n_text_cond", "n_t_emb", "n_input_ids"])
        self.assertEqual(op["op"], "where")
        self.assertIn("tms", op["note"])
        self.assertEqual(self.tensors["n_timestep"]["shape"], ["B"])
        self.assertEqual(self.params["Fq"]["value"], 256)

    def test_position_ids_three_axes(self):
        self.assertEqual(self.tensors["n_position_ids"]["shape"], [3, "B", "Stot"])
        self.assertEqual(self.tensors["n_token_types"]["shape"], ["B", "Stot"])
        self.assertEqual(self.ops["net_attn_partition"]["outputs"], ["n_ar_index"])
        self.assertEqual(self.tensors["n_ar_index"]["shape"], ["Nar"])

    def test_layer_order_is_36_decoder_layers(self):
        order = self.network["layer_order"]
        self.assertEqual([e["index"] for e in order], list(range(36)))
        self.assertTrue(all(e["group_id"] == "uit-decoder" for e in order))
        groups = {g["id"]: g for g in self.network["groups"]}
        self.assertEqual(groups["uit-decoder"]["count"], 36)
        self.assertEqual(groups["vit-tower"]["count"], 27)
        self.assertIn("Kref>0", groups["vit-tower"]["execution"])

    def test_visual_tower_consumes_27_blocks(self):
        self.assertEqual(self.ops["net_vision_loop"]["op"], "layer-loop")
        self.assertEqual(self.ops["net_vision_loop"]["outputs"], ["n_vtower", "n_vtaps"])
        self.assertEqual(self.tensors["n_vtaps"]["shape"], ["Ndh", "Svis", "Hv"])


class ScenarioTest(unittest.TestCase):
    def setUp(self):
        self.structure = _structure("uit-decoder")
        self.vision = _structure("vit-tower")

    def _values_of(self, structure, scenario):
        return {d["id"]: d for d in structure["scenario_derivations"][scenario]}

    def _determined(self, structure, scenario):
        return {k: v.get("value") for k, v in self._values_of(structure, scenario).items()}

    def test_bind_keys_are_image_axes_not_prefill_decode(self):
        self.assertEqual(HIDREAM_BIND_KEYS,
                         ("Himg", "Wimg", "Ptxt", "Kref", "Sref", "Svis", "S_i"))
        self.assertEqual([s["id"] for s in HIDREAM_SCENARIOS],
                         ["t2i-1024", "t2i-2048", "edit-1ref", "edit-2ref-equal",
                          "edit-2ref-unequal", "edit-4ref-nondivisible", "t2i-symbolic"])
        self.assertEqual(self.structure["scope"]["default_scenario"], "t2i-1024")

    def test_t2i_1024_values(self):
        values = self._determined(self.structure, "t2i-1024")
        self.assertEqual(values["S_pix"], 1024)
        self.assertEqual(values["S_gen"], 1024)
        self.assertEqual(values["S_tot"], 1152)
        self.assertEqual(values["N_ar"], 127)
        self.assertEqual(values["N_gt"], 1025)

    def test_t2i_2048_values(self):
        values = self._determined(self.structure, "t2i-2048")
        self.assertEqual(values["S_pix"], 4096)
        self.assertEqual(values["S_gen"], 4096)
        self.assertEqual(values["S_tot"], 4224)
        self.assertEqual(values["N_ar"], 127)

    def test_edit_with_data_dependent_ref_is_unknown(self):
        values = self._values_of(self.structure, "edit-1ref")
        self.assertEqual(values["S_pix"]["state"], "determined")
        self.assertEqual(values["S_pix"]["value"], 1024)
        for key in ("S_gen", "S_tot", "N_gt"):
            self.assertEqual(values[key]["state"], "unknown", key)
            self.assertTrue(values[key]["unknown_reason"], key)
        vision = self._values_of(self.vision, "edit-1ref")
        self.assertEqual(vision["S_vis_m"]["state"], "unknown")
        self.assertIn("Svis", vision["S_vis_m"]["unknown_reason"])

    def test_symbolic_scenario_keeps_unknown(self):
        values = self._values_of(self.structure, "t2i-symbolic")
        for key in ("S_pix", "S_gen", "S_tot", "N_ar", "N_gt"):
            self.assertEqual(values[key]["state"], "unknown", key)
            self.assertTrue(values[key]["unknown_reason"], key)

    def test_two_equal_ref_images_attention_total(self):
        values = self._determined(self.vision, "edit-2ref-equal")
        self.assertEqual(values["S_vis_m"], 288)
        self.assertEqual(values["S_attn_elems"], 10616832)

    def test_two_unequal_ref_images_attention_unknown(self):
        values = self._values_of(self.vision, "edit-2ref-unequal")
        # 总 merge 后 token 数仍可确定（300），但逐段/总注意力规模未知
        self.assertEqual(values["S_vis_m"]["state"], "determined")
        self.assertEqual(values["S_vis_m"]["value"], 300)
        self.assertEqual(values["S_attn_elems"]["state"], "unknown")
        self.assertIn("Kref * S_i == Svis", values["S_attn_elems"]["unknown_reason"])

    def test_nondivisible_per_image_tokens_unknown(self):
        # 总长 2308 可被 4 整除、4×577=2308 也一致，但每图 577 不是 2×2 merge 单元
        values = self._values_of(self.vision, "edit-4ref-nondivisible")
        self.assertEqual(values["S_attn_elems"]["state"], "unknown")
        self.assertIn("S_i % (Mv * Mv) == 0", values["S_attn_elems"]["unknown_reason"])

    def test_missing_segmentation_keeps_attention_unknown(self):
        values = self._values_of(self.vision, "edit-1ref")
        self.assertEqual(values["S_attn_elems"]["state"], "unknown")

    def test_no_kvcachelen_in_scenario(self):
        for scenario in self.structure["scope"]["scenarios"]:
            self.assertNotIn("kvcachelen", scenario)
            self.assertNotIn("seqlen", scenario)


class ContractTest(unittest.TestCase):
    def test_validators_accept_built_documents(self):
        shas = _shas()
        for group_id in ("uit-decoder", "vit-tower"):
            self.assertEqual(
                validate_hf_reading_document(_document(group_id), shas), [], group_id)
            self.assertEqual(
                validate_hf_layer_structure(_structure(group_id)), [], group_id)
        self.assertEqual(validate_hf_network_overview(_network(), shas), [])

    def _reject(self, structure, fragment=None):
        issues = validate_hf_layer_structure(structure)
        self.assertTrue(issues)
        if fragment:
            self.assertTrue(any(fragment in issue for issue in issues), issues)

    def test_reject_cache_tensor(self):
        structure = copy.deepcopy(_structure("uit-decoder"))
        structure["block"]["tensors"].append({
            "id": "d_cache", "name": "KV cache", "shape": ["B", "Nkv", "Stot", "Dh"],
            "kind": "cache", "note": "伪造缓存", "parameter_refs": ["B", "Nkv", "Stot", "Dh"]})
        self._reject(structure, "cache")

    def test_reject_missing_two_pass_merge(self):
        structure = copy.deepcopy(_structure("uit-decoder"))
        block = structure["block"]
        block["operators"] = [op for op in block["operators"] if op["id"] != "d_merge_ar"]
        self._reject(structure, "index_copy")

    def test_reject_ar_score_replaced_by_full_length(self):
        structure = copy.deepcopy(_structure("uit-decoder"))
        for tensor in structure["block"]["tensors"]:
            if tensor["id"] == "d_score_ar":
                tensor["shape"] = ["B", "Nh", "Stot", "Stot"]
        self._reject(structure, "Nar")

    def test_reject_vision_layernorm_to_rmsnorm(self):
        structure = copy.deepcopy(_structure("vit-tower"))
        for op in structure["block"]["operators"]:
            if op["id"] == "v_norm1":
                op["op"] = "rmsnorm"
        self._reject(structure, "LayerNorm")

    def test_reject_vision_score_total_square(self):
        structure = copy.deepcopy(_structure("vit-tower"))
        for tensor in structure["block"]["tensors"]:
            if tensor["id"] in ("v_scores", "v_probs"):
                tensor["shape"] = ["Nhv", "Svis", "Svis"]
        self._reject(structure, "总长平方")

    def test_reject_s_i_without_merge_divisibility(self):
        structure = copy.deepcopy(_structure("vit-tower"))
        for param in structure["block"]["parameters"]:
            if param["id"] == "S_i":
                param["consistency"] = ("((Kref > 0) or (S_i == 0)) and "
                                        "((Kref == 0) or (S_i > 0)) and "
                                        "(Kref * S_i == Svis)")
        self._reject(structure, "merge 单元整除")

    def test_reject_cu_seqlens_wrong_shape(self):
        structure = copy.deepcopy(_structure("vit-tower"))
        for tensor in structure["block"]["tensors"]:
            if tensor["id"] == "v_cu_seqlens":
                tensor["shape"] = ["Kref", 1]
        self._reject(structure, "cu_seqlens")

    def test_reject_vision_segment_split_removed(self):
        structure = copy.deepcopy(_structure("vit-tower"))
        block = structure["block"]
        block["operators"] = [op for op in block["operators"] if op["id"] != "v_seg_split"]
        self._reject(structure, "逐段切分")

    def test_reject_vision_qkv_bias_removed(self):
        structure = copy.deepcopy(_structure("vit-tower"))
        for op in structure["block"]["operators"]:
            if op["id"] == "v_qkv_proj":
                op["weights"] = ["v_w_qkv"]
        self._reject(structure, "bias")

    def test_reject_forbidden_mechanism(self):
        structure = copy.deepcopy(_structure("uit-decoder"))
        structure["block"]["operators"][0]["id"] = "moe_router_probe"
        self._reject(structure, "moe")

    def test_reject_document_with_prefill_decode_phases(self):
        document = copy.deepcopy(_document("uit-decoder"))
        document["scope"]["phases"] = ["prefill", "decode"]
        issues = validate_hf_reading_document(document, _shas())
        self.assertTrue(any("generation" in issue for issue in issues))

    def test_reject_network_with_logits(self):
        network = copy.deepcopy(_network())
        network["path"]["operators"].append({
            "id": "net_lm_head", "module_path": "Qwen3VLForConditionalGeneration.lm_head",
            "name": "LM Head", "op": "linear", "inputs": ["n_hidden_normed"],
            "outputs": ["n_hidden_normed"], "weights": [], "parameter_refs": [],
            "source_refs": [], "analysis_status": "analyzed", "note": "logits"})
        issues = validate_hf_network_overview(network, _shas())
        self.assertTrue(any("lm_head" in issue or "logits" in issue for issue in issues))

    def test_reject_network_composite_merger(self):
        network = copy.deepcopy(_network())
        network["path"]["operators"].append({
            "id": "net_vision_merge", "module_path": "Qwen3VLVisionPatchMerger",
            "name": "merger", "op": "norm-mlp-gelu", "inputs": ["n_vtower"],
            "outputs": ["n_vimg"], "weights": ["w_vmerge1"], "parameter_refs": [],
            "source_refs": [], "analysis_status": "analyzed", "note": "条件：K>0。"})
        issues = validate_hf_network_overview(network, _shas())
        self.assertTrue(any("net_vision_merge" in issue for issue in issues))

    def test_reject_vision_group_always_executed(self):
        network = copy.deepcopy(_network())
        for group in network["groups"]:
            if group["id"] == "vit-tower":
                group["execution"] = "always"
        issues = validate_hf_network_overview(network, _shas())
        self.assertTrue(any("K>0" in issue for issue in issues))

    def test_reject_wrong_tms_token(self):
        network = copy.deepcopy(_network())
        network["scope"]["tms_token_id"] = 1
        issues = validate_hf_network_overview(network, _shas())
        self.assertTrue(any("tms_token_id" in issue for issue in issues))


class WorkbookTest(unittest.TestCase):
    def test_single_sheet_spec(self):
        container = _container()
        spec = build_structures_workbook_spec(container, _network())
        self.assertEqual([s["name"] for s in spec["sheets"]], ["Generation"])
        self.assertEqual(spec["sheets"][0]["id"], "generation")
        sheet = spec["sheets"][0]
        self.assertEqual([g["id"] for g in sheet["groups"]],
                         ["network-ops", "uit-decoder", "vit-tower"])
        counts = {g["id"]: len(g["operators"]) for g in sheet["groups"]}
        # 28 个 network_overview 一次性任务 + 1 个声明式整网 DeepStack 行：
        # 该行替代 decoder/vit 模板里按 36/27 次重复计的条件注入（见性能规则）。
        self.assertEqual(counts["network-ops"], 29)
        self.assertEqual(counts["uit-decoder"], 30)
        self.assertEqual(counts["vit-tower"], 19)
        self.assertEqual(sum(counts.values()), 78)
        self.assertEqual(len(sheet["inputs"]), 13)
        self.assertIn("CUBE", spec["performance"]["status"])
        self.assertEqual(spec["performance"]["engine"]["vec_default"], 64)
        network_ops = {op["id"] for op in sheet["groups"][0]["operators"]}
        self.assertNotIn("net_layer_loop", network_ops)
        self.assertNotIn("net_vision_loop", network_ops)
        self.assertIn("net_deepstack_inject", network_ops)
        # 视觉塔是条件层组：Kref=0 时整段贡献 0，不依赖该分支专有尺寸
        vit = next(g for g in sheet["groups"] if g["id"] == "vit-tower")
        self.assertEqual(vit["execution_condition"]["when"], "Kref >= 1")
        for required in ("net_t_lin1", "net_patch_lin1", "net_vision_patch",
                         "net_vmerge_norm", "net_ds_norm", "net_final_norm",
                         "net_final_lin"):
            self.assertIn(required, network_ops)

    def test_workbook_bytes_and_formulas(self):
        data = export_structures_workbook(_container(), _network())
        wb = load_workbook(io.BytesIO(data), data_only=False)
        self.assertEqual(wb.sheetnames, ["Generation"])
        ws = wb["Generation"]
        values = {}
        for row in ws.iter_rows(min_col=1, max_col=6):
            if row[0].value:
                values[str(row[0].value)] = row[3].value
        for pid in ("Spix", "Sgen", "Stot", "Nar", "Ngt", "Hq", "Hkv", "Svism",
                    "Dhv", "Hv3", "S_i", "Sattn"):
            self.assertTrue(str(values.get(pid, "")).startswith("="), pid)
        self.assertTrue(str(values.get("Nvis", "")).startswith("="))
        self.assertEqual(export_structures_workbook(_container(), _network()), data)

    def test_operator_actual_shape_uses_param_guards(self):
        """算子 Q/R/S 列必须用参数区的完整守卫（不是顶部输入的非负整数简化版）。"""
        data = export_structures_workbook(_container(), _network())
        ws = load_workbook(io.BytesIO(data), data_only=False)["Generation"]
        input_row = {str(ws.cell(row=r, column=1).value): r
                     for r in range(1, ws.max_row + 1)}
        svis_ref = f"$D${input_row['cond_vision_tokens']}"
        rows = {str(ws.cell(row=r, column=2).value): r
                for r in range(1, ws.max_row + 1)}
        qk_r = str(ws.cell(row=rows["v_qk"], column=18).value)
        # v_qk 输出 [Nhv,S_i,S_i]：R 列守卫必须引用 Svis 单元格并含一致性 OR
        self.assertIn(svis_ref, qk_r)
        self.assertIn("OR(", qk_r)
        qkv_r = str(ws.cell(row=rows["v_qkv_proj"], column=18).value)
        # v_qkv 输出 [Svis,Hv3]：R 列守卫必须含 merge 整除的 MOD
        self.assertIn("MOD(", qkv_r)
        # cu_seqlens 一维 shape 在 v_seg_split 的实际输入列（Q）里：Kref+1
        seg_q = str(ws.cell(row=rows["v_seg_split"], column=17).value).replace(" ", "")
        kref_ref = f"$D${input_row['ref_images']}"
        self.assertIn("cu_seqlens（每图边界，一维）=", seg_q)
        self.assertIn(f"{kref_ref}+1", seg_q)
        self.assertNotIn(f"{kref_ref},1]", seg_q)

    def test_input_consistency_guards_in_formulas(self):
        data = export_structures_workbook(_container(), _network())
        wb = load_workbook(io.BytesIO(data), data_only=False)
        ws = wb["Generation"]
        values = {}
        for row in ws.iter_rows(min_col=1, max_col=6):
            if row[0].value:
                values[str(row[0].value)] = row[3].value
        # 输入一致性被编译进单元格公式：patch 整除、merge 整除、K/Sref/Svis/S_i 一致
        self.assertIn("MOD(", str(values["Himg"]))
        self.assertIn("MOD(", str(values["Svis"]))
        # S_i 与 Sattn 的守卫同时引用 Kref/Svis 单元格与 merge 整除
        self.assertIn("MOD(", str(values["S_i"]))
        self.assertIn("*$D$", str(values["S_i"]))
        self.assertIn("MOD(", str(values["Sattn"]))
        # 源结构里参数带 consistency 约束
        decoder = {p["id"]: p for p in _structure("uit-decoder")["block"]["parameters"]}
        self.assertIn("Himg % Ppix == 0", decoder["Himg"]["consistency"])
        self.assertIn("Svis % (Mv * Mv) == 0", decoder["Svis"]["consistency"])
        self.assertIn("Kref * S_i == Svis",
                      {p["id"]: p for p in _structure("vit-tower")["block"]["parameters"]}
                      ["S_i"]["consistency"])

    def test_cost_cells_are_user_rules_not_pending(self):
        data = export_structures_workbook(_container(), _network())
        wb = load_workbook(io.BytesIO(data), data_only=False)
        ws = wb["Generation"]
        cost_col = 16
        formula_costs = 0
        for row in range(1, ws.max_row + 1):
            value = ws.cell(row, cost_col).value
            if isinstance(value, str) and value.startswith("="):
                formula_costs += 1
        # 77 个结构算子 + 1 个整网 DeepStack 行：计费行都写活公式（不是“待提供”）。
        self.assertGreaterEqual(formula_costs, 77)
        texts = [str(c.value) for row in ws.iter_rows() for c in row if c.value is not None]
        self.assertNotIn("成本待提供", " ".join(texts))
        # 小计/总计用 COUNT+SUM 守卫；成本列不得用忽略文本的裸 SUM。
        subtotals = [t for t in texts if t.upper().startswith("=IF(COUNT(")]
        self.assertTrue(subtotals)
        self.assertTrue(all("SUM(" in t for t in subtotals))

    def test_network_section_does_not_sum_two_stacks(self):
        data = export_structures_workbook(_container(), _network())
        wb = load_workbook(io.BytesIO(data), data_only=False)
        ws = wb["Generation"]
        texts = [str(c.value) for row in ws.iter_rows() for c in row if c.value is not None]
        joined = " ".join(texts)
        self.assertIn("不把两者相加成 63 层", joined)
        self.assertNotIn("合计 63", joined)


class ArtifactTest(unittest.TestCase):
    def _build(self):
        return run_hf_build(REPO_ROOT, REPO_ROOT / "structure_reader",
                            model_key=MODEL_KEY, all_variants=True)

    def test_build_candidate_files(self):
        result, files = self._build()
        self.assertTrue(result.all_variants)
        self.assertTrue(result.analysis_id)
        for rel in ("viewer/structure.html", "tables/decoder_variants.xlsx",
                    "data/layer_structures.json", "data/network_overview.json",
                    "data/effective_config.json", "reports/source_manifest.json",
                    "run_manifest.json", "README.md"):
            self.assertIn(rel, files)
        manifest = json.loads(files["run_manifest.json"])
        for rel, sha in manifest["deliverables"].items():
            self.assertEqual(hashlib.sha256(files[rel]).hexdigest(), sha, rel)
        self.assertEqual(manifest["structure"]["single_forward"], True)
        self.assertEqual(manifest["structure"]["sheets"], ["generation"])
        self.assertEqual(manifest["structure"]["layer_count"], 36)
        self.assertEqual(manifest["structure"]["vision_block_count"], 27)

    def test_html_layers_and_navigation_targets(self):
        _, files = self._build()
        html = files["viewer/structure.html"].decode("utf-8")
        self.assertIn("HiDream-O1-Image", html)
        self.assertEqual(len(re.findall(r'data-lv="', html)), 7)
        ids = re.findall(r'\sid="([^"]+)"', html)
        self.assertEqual(len(ids), len(set(ids)), "HTML id 必须唯一")
        for gid in ("uit-decoder", "vit-tower"):
            for href in (f"#/group/{gid}", f"#/group/{gid}/block",
                         f"#/group/{gid}/ops"):
                self.assertIn(href, html)
        self.assertIn("一次 forward", html)
        self.assertIn("输出 <b>x_pred</b>", html)
        self.assertIn("图像尺寸 / prompt / 参考图", html)
        self.assertIn("t2i-1024", html)

    def test_html_has_no_other_model_mechanics_in_prose(self):
        _, files = self._build()
        html = files["viewer/structure.html"].decode("utf-8")
        # 去掉源码快照面板（官方源码里可能出现别的模型名）后再查页面事实
        prose = re.sub(r'<div id="srcpanel-.*?</div>', "", html, flags=re.S)
        for token in ("KDA", "mHC", "Sinkhorn", "Gated DeltaNet", "Qwen Sparse Attention",
                      "GLM", "DeepSeek", "Qwen3.8", "Qwen4", "FiLM", "adaLN 调制"):
            self.assertNotIn(token, prose, token)

    def test_build_is_deterministic(self):
        first = self._build()[1]
        second = self._build()[1]
        self.assertEqual(first, second)

    def test_single_layer_and_onnx_are_refused(self):
        from structure_reader.build_hf_decoder import HfBuildError
        with self.assertRaises(HfBuildError):
            run_hf_build(REPO_ROOT, REPO_ROOT / "structure_reader", model_key=MODEL_KEY)
        with self.assertRaises(HfBuildError):
            run_hf_build(REPO_ROOT, REPO_ROOT / "structure_reader", model_key=MODEL_KEY,
                         all_variants=True, onnx_modules=True)

    def test_profile_and_source_table_isolated(self):
        self.assertIs(profile_for(MODEL_KEY), HIDREAM_O1_PAGE_PROFILE)
        self.assertIs(profile_for("qwen_3_8_flash_next_hf"), QWEN38FLASH_PAGE_PROFILE)
        table = HIDREAM_O1_PAGE_PROFILE.source_table([], {"path": {}})
        self.assertEqual(table["modeling_file"], MODELING_FILE)
        self.assertIn("HiDream/HiDream-O1-Image", table["modeling_file"])
        self.assertNotIn("qwen4_exp", table["modeling_file"])
        self.assertNotEqual(QWEN38FLASH_PAGE_PROFILE.scenario_label,
                            HIDREAM_O1_PAGE_PROFILE.scenario_label)

    def test_same_process_build_does_not_pollute_qwen(self):
        _, hidream = self._build()
        _, qwen = run_hf_build(REPO_ROOT, REPO_ROOT / "structure_reader",
                               model_key="qwen_3_8_flash_next_hf", all_variants=True)
        qwen_html = qwen["viewer/structure.html"].decode("utf-8")
        self.assertIn("query_width", qwen_html)   # Qwen 的 Dq 显示名仍在
        self.assertNotIn("HiDream", qwen_html)
        hidream_html = hidream["viewer/structure.html"].decode("utf-8")
        self.assertNotIn("query_width", hidream_html)
        # 再构建一次 HiDream 与第一次逐字节相同
        self.assertEqual(hidream, self._build()[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
