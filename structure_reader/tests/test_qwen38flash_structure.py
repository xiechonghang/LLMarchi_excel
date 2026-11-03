"""Qwen3.8-Flash-Next（Qwen/Qwen3.8-Flash-Next, qwen4_exp）结构、场景、契约与产物回归。

只测当前源码行为，不读旧 outputs、不依赖既有候选、不导入/运行 HF 模型、不加载
权重。测试想发现的是「把其它 Qwen/GLM/DeepSeek 事实套进来」或「状态/缓存/层归属/
掩码/残差写错」这类真实错误，不是照实现复制断言：

- 身份：官方 config revision/model_type/architectures、text_config、5.8 vs 本地 5.19；
- 层归属：三组（gdn-moe 35 / gdn-ple-moe 1 / qsa-moe 12）恰好覆盖 48 层，
  QSA 层 3,7,…,47、PLE 层 0-based 1；
- GDN：q/k/v 合并投影、因果短卷积+固定 conv state、beta/decay、delta-rule 扫描+
  固定 recurrent state、Gated RMSNorm；没有 RoPE/indexer/attention；
- QSA：q/gate、q/k RMSNorm、partial RoPE 64/256、2 头 KV+repeat、K/V+indexer
  key 缓存、micro-block 均值、relu(q·k).sum/√128、块 top-k+tail+pad 2051、
  True=可见 mask、eager QK/softmax/AV、输出 sigmoid 门；
- Gated Residual：读门逐元素 [4,H]、写门每支路标量 [4]、加权 mean、原 4 路流加回；
  不是普通残差，也没有 mHC/Sinkhorn；
- PLE：仅 layer 1；16 头×160、上下文 [B,2]、dilation-3 短卷积 state 9；
- MoE：softmax 512 选 10、top-10 归一；没有 correction/scale/clamp；共享专家
  有 sigmoid 标量门；单份专家模板；
- 场景：空缓存/非零历史/超过 2048/多 token 的独立期望值；
- 契约：改动结构/网络后检查必须拒绝（非空问题串）；
- 产物：真实 HTML 与两页工作簿（三组、活公式、成本待提供、无其它模型字段）。

运行：
  cd structure_reader && PYTHONPATH=src python3 tests/test_qwen38flash_structure.py
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openpyxl import load_workbook  # noqa: E402

from structure_reader.build_hf_decoder import run_hf_build  # noqa: E402
from structure_reader.checks.qwen38flash_hf_contracts import (  # noqa: E402
    validate_hf_layer_structure,
    validate_hf_network_overview,
    validate_hf_reading_document,
)
from structure_reader.extract.hf_layer_structures import build_structures_container  # noqa: E402
from structure_reader.extract.qwen38flash_hf import (  # noqa: E402
    build_block_structure,
    build_layer_documents,
    build_layer_structure,
)
from structure_reader.extract.qwen38flash_network import build_network_overview  # noqa: E402
from structure_reader.export.qwen38flash_workbook import (  # noqa: E402
    build_structures_workbook_spec,
    export_structures_workbook,
)
from structure_reader.render.hf_model_profile import (  # noqa: E402
    GLM53FLASH_PAGE_PROFILE,
    QWEN38FLASH_PAGE_PROFILE,
    profile_for,
)
from structure_reader.render.qwen38flash_source_labels import MODELING_FILE  # noqa: E402
from structure_reader.rules import qwen38flash_hf as qwen_rules  # noqa: E402
from structure_reader.rules.qwen38flash_hf import (  # noqa: E402
    MODEL_REPOSITORY,
    QWEN38FLASH_SCENARIOS,
    SOURCE_FILES,
    gdn_layer_indices,
    layer_group_assignments,
    load_hf_model_values,
    ple_layer_indices,
    qsa_layer_indices,
    verify_source_fingerprints,
)

MODEL_KEY = "qwen_3_8_flash_next_hf"
GROUP_COUNTS = {"gdn-moe": 35, "gdn-ple-moe": 1, "qsa-moe": 12}


def _shas() -> dict:
    return {k: hashlib.sha256((REPO_ROOT / v).read_bytes()).hexdigest()
            for k, v in SOURCE_FILES.items()}


def _values():
    return load_hf_model_values(REPO_ROOT, model_key=MODEL_KEY)


def _block(layer_index: int) -> dict:
    block = build_block_structure(_values(), layer_index)
    block["id"] = f"block-{layer_index}"
    block["scope_notes"] = ["test"]
    return block


def _container():
    def docs(root, layer_index, model_key):
        return build_layer_documents(root, layer_index=layer_index, model_key=model_key)

    return build_structures_container(
        REPO_ROOT, model_key=MODEL_KEY,
        values_loader=lambda root, key: load_hf_model_values(root, model_key=key),
        groups_fn=layer_group_assignments, build_documents_fn=docs)


class IdentityTest(unittest.TestCase):
    def test_official_repository_identity(self):
        self.assertEqual(MODEL_REPOSITORY["repo_id"], "Qwen/Qwen3.8-Flash-Next")
        self.assertEqual(MODEL_REPOSITORY["revision"],
                         "de4b8e4d43b917e7706784d8bb445c9af86a3540")
        self.assertEqual(MODEL_REPOSITORY["model_type"], "qwen4_exp")
        self.assertEqual(MODEL_REPOSITORY["architectures"],
                         ["Qwen4ExpForConditionalGeneration"])
        raw = json.loads((REPO_ROOT / SOURCE_FILES["config.json"]).read_text())
        self.assertEqual(raw["model_type"], "qwen4_exp")
        self.assertEqual(raw["architectures"], ["Qwen4ExpForConditionalGeneration"])
        self.assertEqual(raw["text_config"]["model_type"], "qwen4_exp_text")

    def test_checkpoint_version_and_local_implementation(self):
        values = _values()
        self.assertEqual(values.checkpoint_transformers_version, "5.8.0.dev0")
        self.assertEqual(values.source_revision,
                         "536ecc007387a50e77603bb5d92100e9b07514cc")
        self.assertTrue((REPO_ROOT / SOURCE_FILES["modeling_qwen4_exp.py"]).is_file())

    def test_text_config_facts(self):
        v = _values()
        self.assertEqual(v.hidden_size, 2560)
        self.assertEqual(v.num_hidden_layers, 48)
        self.assertEqual((v.num_attention_heads, v.num_key_value_heads), (24, 2))
        self.assertEqual(v.head_dim, 256)
        self.assertEqual((v.rotary_dim, v.partial_rotary_factor), (64, 0.25))
        self.assertEqual(v.mrope_section, (11, 11, 10))
        self.assertEqual(v.vocab_size, 248320)
        self.assertEqual(v.rms_norm_eps, 1e-6)
        self.assertFalse(v.tie_word_embeddings)

    def test_gdn_config_facts(self):
        v = _values()
        self.assertEqual((v.linear_num_key_heads, v.linear_num_value_heads),
                         (16, 48))
        self.assertEqual((v.linear_key_head_dim, v.linear_value_head_dim),
                         (128, 128))
        self.assertEqual(v.linear_conv_kernel_dim, 4)
        self.assertEqual(v.output_gate_type, "sigmoid")

    def test_qsa_and_indexer_config_facts(self):
        v = _values()
        self.assertEqual((v.indexer_n_heads, v.indexer_kv_heads, v.indexer_head_dim),
                         (4, 1, 128))
        self.assertEqual((v.indexer_budget, v.indexer_compress_ratio),
                         (2048, 4))
        self.assertEqual(v.indexer_block_topk, 512)
        self.assertEqual(v.attention_scaling, 256 ** -0.5)

    def test_gated_residual_and_ple_config_facts(self):
        v = _values()
        self.assertEqual((v.hc_count, v.hc_lowrank), (4, 320))
        self.assertEqual(v.ple_layer_ids, [2])
        self.assertEqual((v.ngram_size, v.heads_per_ngram), (3, 8))
        self.assertEqual((v.ngram_heads, v.ngram_head_dim), (16, 160))
        self.assertEqual(v.ple_embed_dim, 2560)
        self.assertEqual((v.ple_conv_kernel_size, v.ple_conv_dilation,
                          v.ple_short_conv_state_len), (4, 3, 9))
        self.assertEqual(v.ngram_context_len, 2)
        self.assertEqual(v.eos_token_id, 248044)
        self.assertEqual(v.number_of_conv_states, 3)

    def test_moe_config_facts(self):
        v = _values()
        self.assertEqual((v.num_experts, v.num_experts_per_tok), (512, 10))
        self.assertEqual((v.moe_intermediate_size, v.shared_expert_intermediate_size),
                         (640, 640))
        self.assertTrue(v.norm_topk_prob)

    def test_source_fingerprints_verify(self):
        verify_source_fingerprints(REPO_ROOT)
        self.assertEqual(len(_shas()), len(SOURCE_FILES))


class LayerAttributionTest(unittest.TestCase):
    def test_48_layers_and_hybrid_pattern(self):
        v = _values()
        self.assertEqual(len(v.layer_types), 48)
        self.assertEqual(len(gdn_layer_indices(v)), 36)
        self.assertEqual(qsa_layer_indices(v),
                         [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47])
        self.assertEqual(ple_layer_indices(v), [1])

    def test_three_groups_cover_48_layers(self):
        groups = layer_group_assignments(_values())
        counts = {g["id"]: g["count"] for g in groups}
        self.assertEqual(counts, GROUP_COUNTS)
        covered = sorted(i for g in groups for i in g["layer_indices"])
        self.assertEqual(covered, list(range(48)))
        self.assertEqual(sum(counts.values()), 48)

    def test_group_attention_types_and_ple(self):
        groups = {g["id"]: g for g in layer_group_assignments(_values())}
        self.assertEqual(groups["gdn-moe"]["attention_type"], "linear_attention")
        self.assertFalse(groups["gdn-moe"]["ple"])
        self.assertEqual(groups["gdn-ple-moe"]["attention_type"], "linear_attention")
        self.assertTrue(groups["gdn-ple-moe"]["ple"])
        self.assertEqual(groups["gdn-ple-moe"]["layer_indices"], [1])
        self.assertEqual(groups["qsa-moe"]["attention_type"], "indexed_attention")
        self.assertFalse(groups["qsa-moe"]["ple"])

    def test_layer_order_matches_groups(self):
        net = build_network_overview(REPO_ROOT, model_key=MODEL_KEY)
        group_of = {i: g["id"] for g in net["groups"] for i in g["layer_indices"]}
        for entry in net["layer_order"]:
            self.assertEqual(entry["group_id"], group_of[entry["index"]])
            self.assertEqual(entry["attention_type"],
                             "indexed_attention" if entry["index"] % 4 == 3
                             else "linear_attention")
            self.assertEqual(entry["ple"], entry["index"] == 1)


class GdnBlockTest(unittest.TestCase):
    def test_params(self):
        params = {p["id"]: p for p in _block(0)["parameters"]}
        expected = {"H": 2560, "Hc": 4, "Rhc": 320, "NkH": 16, "NvH": 48,
                    "Dk": 128, "DvH": 128, "Kconv": 4, "Dqkv": 10240,
                    "Dz": 6144, "E": 512, "K": 10, "Es": 640, "Ess": 640}
        for pid, value in expected.items():
            self.assertEqual(params[pid]["value"], value, pid)

    def test_short_conv_and_fixed_states(self):
        block = _block(0)
        tensors = {t["id"]: t for t in block["tensors"]}
        self.assertEqual(tensors["g_gdn_conv_state_pre"]["shape"],
                         ["B", "Dqkv", "Kconv"])
        self.assertEqual(tensors["g_gdn_state_pre"]["shape"],
                         ["B", "NvH", "Dk", "DvH"])
        deriv = {d["id"]: d for d in block["derivations"]}
        self.assertEqual(deriv["S_rec"]["expression"], "NvH * Dk * DvH")
        self.assertEqual(deriv["S_conv"]["expression"], "Dqkv * Kconv")

    def test_scan_is_delta_rule_not_attention(self):
        ops = {o["id"]: o for o in _block(0)["operators"]}
        note = ops["g_gdn_scan"]["note"]
        for token in ("A = S", "A^T k", "delta", "chunk", "recurrent"):
            self.assertIn(token, note)
        self.assertNotIn("softmax", note.lower())

    def test_no_rope_or_indexer_or_plain_attention(self):
        block = _block(0)
        ids = [o["id"] for o in block["operators"]]
        self.assertFalse([x for x in ids if x.startswith("g_qsa_")])
        self.assertFalse([x for x in ids if "rope" in x.lower()])
        self.assertFalse([x for x in ids if "indexer" in x.lower()])

    def test_gated_norm_uses_sigmoid_output_gate(self):
        ops = {o["id"]: o for o in _block(0)["operators"]}
        note = ops["g_gdn_o_norm"]["note"]
        self.assertIn("sigmoid", note)
        self.assertIn("RMSNorm", note)


class QsaBlockTest(unittest.TestCase):
    def test_q_gate_and_partial_rope(self):
        block = _block(3)
        ops = {o["id"]: o for o in block["operators"]}
        tensors = {t["id"]: t for t in block["tensors"]}
        self.assertIn("g_qsa_q_gate_chunk", ops)
        self.assertEqual(tensors["g_qsa_q_rope"]["shape"][-1], "Dh")
        self.assertIn("64", ops["g_qsa_q_rope"]["note"])
        self.assertIn("partial", ops["g_qsa_q_rope"]["name"].lower())

    def test_two_kv_heads_and_cache(self):
        block = _block(3)
        tensors = {t["id"]: t for t in block["tensors"]}
        self.assertEqual(tensors["g_qsa_k_pre"]["shape"], ["B", "Nkv", "kvcachelen", "Dh"])
        ops = {o["id"]: o for o in block["operators"]}
        self.assertIn("Nrep", ops["g_qsa_repeat_kv"]["parameter_refs"])

    def test_indexer_micro_block_and_score(self):
        block = _block(3)
        ops = {o["id"]: o for o in block["operators"]}
        self.assertIn("均值", ops["g_qsa_ix_pool"]["name"])
        note = ops["g_qsa_ix_scores"]["note"]
        self.assertIn("relu", note.lower())
        self.assertIn("sum", note.lower())
        self.assertIn("sqrt", note.lower())
        self.assertIn("不是 softmax", note.lower())

    def test_indexer_topk_tail_and_mask(self):
        block = _block(3)
        ops = {o["id"]: o for o in block["operators"]}
        self.assertIn("512", ops["g_qsa_ix_block_topk"]["note"])
        self.assertIn("tail", ops["g_qsa_ix_tail"]["name"].lower())
        self.assertIn("2051", ops["g_qsa_ix_pad"]["note"])
        self.assertIn("True=可见", ops["g_qsa_ix_mask_bool"]["note"])
        self.assertIn("0.0", ops["g_qsa_ix_mask_float"]["note"])
        self.assertIn("g_qsa_ix_mask_float", ops["g_qsa_mask_combine"]["inputs"])
        self.assertIn("合并", ops["g_qsa_mask_combine"]["name"])

    def test_eager_attention_and_output_gate(self):
        ops = {o["id"]: o for o in _block(3)["operators"]}
        for oid in ("g_qsa_qk", "g_qsa_mask_add", "g_qsa_softmax",
                    "g_qsa_av", "g_qsa_gate_sigmoid", "g_qsa_o_proj"):
            self.assertIn(oid, ops)
        self.assertIn("softmax", ops["g_qsa_softmax"]["name"].lower())


class GatedResidualAndPleTest(unittest.TestCase):
    def test_read_gate_elementwise_write_gate_scalar(self):
        block = _block(0)
        tensors = {t["id"]: t for t in block["tensors"]}
        self.assertEqual(tensors["g_hc_attn_read"]["shape"],
                         ["B", "seqlen", "Hc", "H"])
        self.assertEqual(tensors["g_hc_attn_inject"]["shape"],
                         ["B", "seqlen", "Hc"])
        ops = {o["id"]: o for o in block["operators"]}
        self.assertIn("mean", ops["g_hc_attn_mix"]["name"].lower())
        self.assertIn("相加", ops["g_hc_attn_add"]["name"])

    def test_no_mhc_sinkhorn_or_plain_residual(self):
        block = _block(0)
        ids = [o["id"] + t["id"] for o in block["operators"]
               for t in block["tensors"]]
        joined = " ".join(ids).lower()
        for token in ("sinkhorn", "mhc", "pre_logits", "comb_logits"):
            self.assertNotIn(token, joined)

    def test_ple_only_layer_one(self):
        self.assertFalse(_block(0)["ple"])
        self.assertTrue(_block(1)["ple"])
        self.assertFalse([o for o in _block(0)["operators"]
                          if o["id"].startswith("g_ple_")])
        self.assertTrue([o for o in _block(1)["operators"]
                         if o["id"].startswith("g_ple_")])

    def test_ple_context_and_dilated_conv(self):
        block = _block(1)
        tensors = {t["id"]: t for t in block["tensors"]}
        self.assertEqual(tensors["g_ple_context_pre"]["shape"], ["B", "Lctx"])
        self.assertEqual(tensors["g_ple_conv_state_pre"]["shape"],
                         ["B", "Dqkv", "Lpconv"])
        ops = {o["id"]: o for o in block["operators"]}
        self.assertIn("dilation=3", ops["g_ple_conv"]["note"])
        self.assertIn("16", ops["g_ple_gather"]["note"])


class MoeTest(unittest.TestCase):
    def test_softmax_top10_and_norm(self):
        block = _block(0)
        ops = {o["id"]: o for o in block["operators"]}
        self.assertIn("softmax", ops["g_moe_softmax"]["name"].lower())
        self.assertIn("topk", ops["g_moe_topk"]["name"].lower())
        self.assertIn("归一", ops["g_moe_weight_norm"]["name"])

    def test_no_bias_scaling_or_clamp(self):
        ids = {o["id"] for o in _block(0)["operators"]}
        for oid in ("g_moe_bias_add", "g_moe_weight_scale", "g_moe_expert_clamp"):
            self.assertNotIn(oid, ids)

    def test_shared_expert_scalar_gate(self):
        ops = {o["id"]: o for o in _block(0)["operators"]}
        self.assertIn("g_moe_shared_gate_scalar", ops)
        self.assertIn("sigmoid", ops["g_moe_shared_gate_scalar"]["op"].lower())
        self.assertIn("g_moe_merge", ops)

    def test_single_expert_template(self):
        block = _block(0)
        expert_ops = [o for o in block["operators"] if o["id"].startswith("g_moe_expert_")]
        self.assertLess(len(expert_ops), 20)
        note = next(o["note"] for o in block["operators"] if o["id"] == "g_moe_expert_hit")
        self.assertIn("N_hit", note)


class ScenarioAndContractTest(unittest.TestCase):
    def test_qsa_scenarios(self):
        structure = build_layer_structure(REPO_ROOT, layer_index=3, model_key=MODEL_KEY)
        expected = {
            "prefill-empty": (8, 8, 2, 2, 8, 2051),
            "decode-nonzero": (4097, 4097, 1024, 512, 2049, 2051),
            "decode-long": (131073, 131073, 32768, 512, 2049, 2051),
            "decode-multi": (4096, 4096, 1024, 512, 2048, 2051),
        }
        for sid, values in expected.items():
            got = {e["id"]: e.get("value")
                   for e in structure["scenario_derivations"][sid]}
            self.assertEqual(
                (got["L_kv"], got["L_idx"], got["N_pool"], got["B_sel"],
                 got["K_sel_last"], got["K_out"]), values, sid)
        symbolic = {e["id"]: e for e in
                    structure["scenario_derivations"]["prefill-symbolic"]}
        for did in ("L_kv", "L_idx", "N_pool", "V_q", "K_sel", "K_sel_last"):
            self.assertNotEqual(symbolic[did]["state"], "determined", did)

    def test_gdn_fixed_state_scenarios(self):
        structure = build_layer_structure(REPO_ROOT, layer_index=1, model_key=MODEL_KEY)
        for sid in ("prefill-empty", "decode-nonzero", "decode-long", "decode-multi"):
            got = {e["id"]: e.get("value")
                   for e in structure["scenario_derivations"][sid]}
            self.assertEqual(got["S_rec"], 786432)
            self.assertEqual(got["S_conv"], 40960)
            self.assertEqual(got["S_ple_conv"], 92160)
            self.assertEqual(got["L_ngram_ctx"], 2)

    def test_validators_accept_built_documents(self):
        shas = _shas()
        for layer_index in (0, 1, 3):
            structure, reading = build_layer_documents(
                REPO_ROOT, layer_index=layer_index, model_key=MODEL_KEY)
            self.assertEqual(validate_hf_layer_structure(structure), [])
            self.assertEqual(validate_hf_reading_document(reading, shas), [])
        net = build_network_overview(REPO_ROOT, model_key=MODEL_KEY)
        self.assertEqual(validate_hf_network_overview(net, shas), [])

    def test_no_per_token_indexer_k_norm(self):
        ids = {o["id"] for o in _block(3)["operators"]}
        self.assertNotIn("g_qsa_ix_k_norm", ids)
        self.assertIn("g_qsa_ix_pool_norm", ids)

    def test_per_query_shapes_use_nq_bq(self):
        tensors = {t["id"]: t for t in _block(3)["tensors"]}
        for tid in ("g_qsa_ix_block_tokens", "g_qsa_ix_pooled_keys",
                    "g_qsa_ix_pooled_normed", "g_qsa_ix_scores"):
            self.assertEqual(tensors[tid]["shape"][2], "Nq", tid)
        self.assertEqual(tensors["g_qsa_ix_block_idx"]["shape"],
                         ["B", "seqlen", "Bq"])
        self.assertEqual(tensors["g_qsa_ix_tail"]["shape"],
                         ["B", "seqlen", "Rtail"])
        params = {p["id"]: p for p in _block(3)["parameters"]}
        self.assertIn("Nq", params)
        self.assertIn("Bq", params)
        self.assertIn("Rtail", params)

    def test_flat_residual_and_ple_layout(self):
        gdn = _block(0)
        tensors = {t["id"]: t for t in gdn["tensors"]}
        self.assertEqual(tensors["g_hidden_streams_in"]["shape"],
                         ["B", "seqlen", "Hc×H"])
        self.assertEqual(tensors["g_hc_attn_expanded_flat"]["shape"],
                         ["B", "seqlen", "Hc×H"])
        ops = {o["id"]: o for o in gdn["operators"]}
        self.assertEqual(ops["g_hc_attn_add"]["inputs"][1],
                         "g_hc_attn_expanded_flat")
        ple = _block(1)
        pt = {t["id"]: t for t in ple["tensors"]}
        self.assertEqual(pt["g_ple_out"]["shape"], ["B", "seqlen", "Dqkv"])
        self.assertEqual(pt["g_hidden_streams_ple"]["shape"],
                         ["B", "seqlen", "Hc×H"])
        pops = {o["id"]: o for o in ple["operators"]}
        self.assertEqual(pops["g_ple_add"]["inputs"],
                         ["g_ple_gated_value_masked_flat", "g_ple_conv_out"])

    def test_reject_per_query_shape_replaced_by_full_length(self):
        block = copy.deepcopy(_block(3))
        for tensor in block["tensors"]:
            if tensor["id"] == "g_qsa_ix_block_tokens":
                tensor["shape"] = ["B", "seqlen", "Npool", "Kc"]
        structure = build_layer_structure(REPO_ROOT, layer_index=3, model_key=MODEL_KEY)
        issues = validate_hf_layer_structure({
            "schema": "hf_layer_structure.v1",
            "model": structure["model"],
            "scope": structure["scope"],
            "block": block,
            "scenario_derivations": structure["scenario_derivations"],
        })
        self.assertTrue(issues)

    def test_reject_4d_residual_added_without_flatten(self):
        block = copy.deepcopy(_block(0))
        for op in block["operators"]:
            if op["id"] == "g_hc_attn_add":
                op["inputs"] = ["g_hidden_streams_in", "g_hc_attn_expanded"]
        structure = build_layer_structure(REPO_ROOT, layer_index=0, model_key=MODEL_KEY)
        issues = validate_hf_layer_structure({
            "schema": "hf_layer_structure.v1",
            "model": structure["model"],
            "scope": structure["scope"],
            "block": block,
            "scenario_derivations": structure["scenario_derivations"],
        })
        self.assertTrue(issues)

    def test_reject_direct_bool_mask_add(self):
        block = copy.deepcopy(_block(3))
        for op in block["operators"]:
            if op["id"] == "g_qsa_mask_combine":
                op["inputs"] = ["g_causal_mask", "g_qsa_ix_mask_bool"]
        structure = build_layer_structure(REPO_ROOT, layer_index=3, model_key=MODEL_KEY)
        issues = validate_hf_layer_structure({
            "schema": "hf_layer_structure.v1",
            "model": structure["model"],
            "scope": structure["scope"],
            "block": block,
            "scenario_derivations": structure["scenario_derivations"],
        })
        self.assertTrue(issues)

    def test_qwen_did_not_override_glm_dq_display(self):
        from structure_reader.render.display_names import PARAM_NAMES, full_name
        self.assertEqual(PARAM_NAMES["Dq"], "qk_head_dim")
        self.assertEqual(full_name("Dq"), "qk_head_dim")
        self.assertEqual(full_name("Dqw"), "query_width")
        self.assertEqual(full_name("Nmtp"), "Nmtp")

    def test_reject_wrong_k_sel_formula(self):
        structure = build_layer_structure(REPO_ROOT, layer_index=3, model_key=MODEL_KEY)
        block = copy.deepcopy(structure["block"])
        for deriv in block["derivations"]:
            if deriv["id"] == "K_sel":
                deriv["expression"] = "min(L_idx, Ki + (L_idx % Kc))"
        mutated = dict(structure)
        mutated["block"] = block
        issues = validate_hf_layer_structure(mutated)
        self.assertTrue(issues)

    def test_reject_missing_indexer_mask(self):
        block = copy.deepcopy(_block(3))
        block["operators"] = [o for o in block["operators"]
                              if o["id"] != "g_qsa_ix_mask_bool"]
        issues = validate_hf_layer_structure({
            "schema": "hf_layer_structure.v1",
            "model": {"model_key": MODEL_KEY, "display_name": "x",
                      "implementation": "transformers",
                      "source_revision": "536ecc007387a50e77603bb5d92100e9b07514cc",
                      "config_path": SOURCE_FILES["config.json"],
                      "model_revision": MODEL_REPOSITORY["revision"]},
            "scope": {"layer_index": 3, "analysis_path": "eager",
                      "scenarios": [dict(s) for s in QWEN38FLASH_SCENARIOS],
                      "default_scenario": "prefill-empty"},
            "block": block, "scenario_derivations": {},
        })
        self.assertTrue(issues)

    def test_reject_scalar_read_gate(self):
        block = copy.deepcopy(_block(0))
        for tensor in block["tensors"]:
            if tensor["id"] == "g_hc_attn_read":
                tensor["shape"] = ["B", "seqlen", "Hc"]
        structure = build_layer_structure(REPO_ROOT, layer_index=0, model_key=MODEL_KEY)
        issues = validate_hf_layer_structure({
            "schema": "hf_layer_structure.v1",
            "model": structure["model"],
            "scope": structure["scope"],
            "block": block,
            "scenario_derivations": structure["scenario_derivations"],
        })
        self.assertTrue(issues)

    def test_reject_sigmoid_router(self):
        block = copy.deepcopy(_block(0))
        for op in block["operators"]:
            if op["id"] == "g_moe_router":
                op["note"] = "sigmoid 打分"
        structure = build_layer_structure(REPO_ROOT, layer_index=0, model_key=MODEL_KEY)
        issues = validate_hf_layer_structure({
            "schema": "hf_layer_structure.v1",
            "model": structure["model"],
            "scope": structure["scope"],
            "block": block,
            "scenario_derivations": structure["scenario_derivations"],
        })
        self.assertTrue(issues)


class WorkbookAndArtifactTest(unittest.TestCase):
    def test_workbook_spec(self):
        container = _container()
        net = build_network_overview(REPO_ROOT, model_key=MODEL_KEY)
        spec = build_structures_workbook_spec(container, net)
        self.assertEqual([s["name"] for s in spec["sheets"]], ["Prefill", "Decode"])
        for sheet in spec["sheets"]:
            # 三个真实层组 + 整网外围块（T11 起接入性能区，外围也要逐算子计成本）
            self.assertEqual([g["id"] for g in sheet["groups"]],
                             ["gdn-moe", "gdn-ple-moe", "qsa-moe", "net-periphery"])
            self.assertEqual(sum(len(g["operators"]) for g in sheet["groups"]
                                 if g["id"] != "net-periphery"), 268)
        self.assertIn("CUBE", spec["performance"]["status"])
        self.assertEqual(spec["performance"]["engine"]["cube_default"], 4096)
        rows = [p for g in spec["sheets"][0]["groups"] for p in g["parameters"]]
        j = next(p for p in rows if p["id"] == "j")
        self.assertEqual(j["domain"], {"kind": "integer", "min": 0,
                                       "max_symbol": "seqlen"})

    def test_workbook_bytes_and_formulas(self):
        container = _container()
        net = build_network_overview(REPO_ROOT, model_key=MODEL_KEY)
        data = export_structures_workbook(container, net)
        wb = load_workbook(io.BytesIO(data), data_only=False)
        self.assertEqual(wb.sheetnames, ["Prefill", "Decode"])
        ws = wb["Prefill"]
        values = {}
        for row in ws.iter_rows(min_col=1, max_col=6):
            if row[0].value:
                values[str(row[0].value)] = row[3].value
        for pid in ("Srec", "Sconv", "Sple", "Lkv", "Lidx", "Npool",
                    "Bsel", "Ksel", "Ksel_last", "Kout"):
            self.assertTrue(str(values.get(pid, "")).startswith("="), pid)

    def test_build_candidate_files(self):
        result, files = run_hf_build(
            REPO_ROOT, REPO_ROOT / "structure_reader",
            model_key=MODEL_KEY, all_variants=True)
        self.assertTrue(result.all_variants)
        self.assertTrue(result.analysis_id)
        for rel in ("viewer/structure.html", "tables/decoder_variants.xlsx",
                    "data/layer_structures.json", "data/network_overview.json",
                    "reports/source_manifest.json", "run_manifest.json"):
            self.assertIn(rel, files)
        html = files["viewer/structure.html"].decode("utf-8")
        self.assertIn("Qwen3.8-Flash-Next", html)
        self.assertIn("gdn-ple-moe", html)
        self.assertNotIn("KDA", html)
        self.assertNotIn("mHC", html)
        self.assertNotIn("DeepSeek", html)
        ids = re.findall(r'\sid="([^"]+)"', html)
        self.assertEqual(len(ids), len(set(ids)))
        manifest = json.loads(files["run_manifest.json"])
        for rel, sha in manifest["deliverables"].items():
            self.assertEqual(hashlib.sha256(files[rel]).hexdigest(), sha, rel)

    def test_same_process_display_not_polluted(self):
        _, qwen_first = run_hf_build(
            REPO_ROOT, REPO_ROOT / "structure_reader",
            model_key=MODEL_KEY, all_variants=True)
        _, deep = run_hf_build(
            REPO_ROOT, REPO_ROOT / "structure_reader",
            model_key="deepseek_v4_flash_0731_hf", all_variants=True)
        deep_html = deep["viewer/structure.html"].decode("utf-8")
        self.assertIn("sum_j comb[j,k]·residual[j,d]", deep_html)
        self.assertNotIn("query_index", deep_html)
        _, glm = run_hf_build(
            REPO_ROOT, REPO_ROOT / "structure_reader",
            model_key="glm_5_3_hf", all_variants=True)
        self.assertNotIn("query_index", glm["viewer/structure.html"].decode("utf-8"))
        _, qwen_again = run_hf_build(
            REPO_ROOT, REPO_ROOT / "structure_reader",
            model_key=MODEL_KEY, all_variants=True)
        self.assertEqual(qwen_first, qwen_again)

    def test_cross_model_profile_and_source_table_isolated(self):
        self.assertIs(profile_for(MODEL_KEY), QWEN38FLASH_PAGE_PROFILE)
        self.assertIs(profile_for("glm_5_3_flash_hf"), GLM53FLASH_PAGE_PROFILE)
        table = QWEN38FLASH_PAGE_PROFILE.source_table([], {"path": {}})
        self.assertEqual(table["modeling_file"], MODELING_FILE)
        self.assertIn("qwen4_exp", table["modeling_file"])
        self.assertNotIn("glm5_next", table["modeling_file"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
