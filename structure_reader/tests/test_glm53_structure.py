"""GLM-5.3（zai-org/GLM-5.3, glm_moe_dsa）结构、场景、契约与产物回归。

只测当前源码行为，不读旧 outputs、不依赖既有候选、不导入/运行 HF 模型、不加载
权重。测试想发现的是「把别的模型事实套进来」或「维度/缓存/层归属写错」这类真实
错误，不是照实现复制断言：

- 身份：官方 config revision/model_type/architectures；head_dim=192 被覆盖为 64；
- 层归属：三组（full-dense 3 / shared-moe 57 / full-moe 18）恰好覆盖 78 层，
  full indexer 21 层、dense 层 0/1/2；
- Attention：Q 低秩链顺序、KV 只缓存 512+64 压缩 latent、kv_b 在缓存之后展开、
  交错部分 RoPE 只 64 维、scaling=1/16；
- DSA：full 层 indexer 算子/键缓存/带 bias LayerNorm；shared 层无 indexer、
  消费 prev_topk_indices；
- MoE：sigmoid + 修正偏置 + top-8 + 归一 + ×2.5；单份专家模板；共享专家并行相加；
  dense 层没有 MoE 算子；
- 残差：两处普通 add；没有 mHC/压缩器/滑窗/hash 等其它模型机制；
- 场景：空缓存/非零历史/超过 index_topk 的独立期望值；
- 契约：改动结构/网络后检查必须拒绝（非空问题串）；
- 产物：真实 HTML 与两页工作簿（三组、活公式、成本待提供、无 DeepSeek 字段）。

运行：
  cd structure_reader && PYTHONPATH=src python3 tests/test_glm53_structure.py
"""

from __future__ import annotations

import copy
import io
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openpyxl import load_workbook  # noqa: E402

from structure_reader.build_hf_decoder import HfBuildError, run_hf_build  # noqa: E402
from structure_reader.checks.glm53_hf_contracts import (  # noqa: E402
    validate_hf_layer_structure,
    validate_hf_network_overview,
    validate_hf_reading_document,
)
from structure_reader.extract.glm53_hf import (  # noqa: E402
    build_layer_documents,
    build_layer_structure,
)
from structure_reader.extract.glm53_network import build_network_overview  # noqa: E402
from structure_reader.extract.hf_layer_structures import (  # noqa: E402
    build_structures_container,
)
from structure_reader.export.glm53_workbook import (  # noqa: E402
    build_structures_workbook_spec,
)
from structure_reader.export.operator_workbook import export_workbook  # noqa: E402
from structure_reader.render.hf_model_profile import GLM53_PAGE_PROFILE  # noqa: E402
from structure_reader.rules.glm53_hf import (  # noqa: E402
    GLM53_SCENARIOS,
    MODEL_REPOSITORY,
    SOURCE_FILES,
    derive_indexer_types,
    effective_indexer_types,
    layer_group_assignments,
    load_hf_model_values,
)

MODEL_KEY = "glm_5_3_hf"
EXPECTED_PARAMS = {
    "H": 6144, "Nh": 64, "Nkv": 64, "Dq": 256, "Dn": 192, "Rk": 64, "Dv": 256,
    "Rq": 2048, "Rkv": 512, "Ih": 32, "Id": 128, "Ki": 2048, "E": 256, "K": 8,
    "Es": 2048, "Is": 12288, "Scl": 2.5, "Ng": 1, "Ktg": 1,
    "ThetaM": 8000000.0, "Vocab": 154880,
}


def _workspace_ready() -> bool:
    return (REPO_ROOT / SOURCE_FILES["config.json"]).is_file() and (
        REPO_ROOT / SOURCE_FILES["modeling_glm_moe_dsa.py"]).is_file()


@unittest.skipUnless(_workspace_ready(), "缺少 GLM-5.3 config / glm_moe_dsa 源码；如实跳过")
class Glm53TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = load_hf_model_values(REPO_ROOT, model_key=MODEL_KEY)
        cls.groups = layer_group_assignments(cls.values)
        cls.block_dense = build_layer_structure(REPO_ROOT, 0, MODEL_KEY)["block"]
        cls.block_shared = build_layer_structure(REPO_ROOT, 3, MODEL_KEY)["block"]
        cls.block_full = build_layer_structure(REPO_ROOT, 6, MODEL_KEY)["block"]
        cls.network = build_network_overview(REPO_ROOT, model_key=MODEL_KEY)

    @staticmethod
    def ops(block):
        return {o["id"]: o for o in block["operators"]}

    @staticmethod
    def tensors(block):
        return {t["id"]: t for t in block["tensors"]}

    @staticmethod
    def params(block):
        return {p["id"]: p for p in block["parameters"]}


class TestIdentity(Glm53TestBase):
    def test_official_repository_identity(self):
        raw = load_hf_model_values(REPO_ROOT, model_key=MODEL_KEY)
        self.assertEqual(MODEL_REPOSITORY["repo_id"], "zai-org/GLM-5.3")
        self.assertEqual(MODEL_REPOSITORY["revision"],
                         "ba32f8c19c8738e2c57c71798c601d477a7bf844")
        self.assertEqual(raw.model_revision, MODEL_REPOSITORY["revision"])
        self.assertTrue(raw.source_revision)
        self.assertEqual(raw.config_path, "models/zai-org/GLM-5.3/config.json")

    def test_head_dim_override_is_64_not_192(self):
        # 根配置保存 head_dim=192；__post_init__ 把运行期 head_dim 指向 rope slice 64。
        self.assertEqual(self.values.config_head_dim_raw, 192)
        self.assertEqual(self.values.head_dim, 64)
        self.assertEqual(self.values.qk_head_dim, 256)
        self.assertEqual(self.values.qk_nope_head_dim + self.values.qk_rope_head_dim,
                         self.values.qk_head_dim)
        self.assertAlmostEqual(self.values.attention_scaling, 1 / 16, places=12)

    def test_indexer_types_explicit_equals_freq_offset_derivation(self):
        derived = derive_indexer_types(self.values.index_topk_freq,
                                       self.values.index_skip_topk_offset,
                                       self.values.num_hidden_layers)
        self.assertEqual(self.values.indexer_types, derived)
        with self.assertRaises(ValueError):
            effective_indexer_types({
                "num_hidden_layers": 4,
                "indexer_types": ["full", "shared", "full", "shared"],
                "index_topk_freq": 4,
                "index_skip_topk_offset": 3,
            })
        with self.assertRaises(ValueError):
            effective_indexer_types({"num_hidden_layers": 2, "indexer_types": ["full"]})
        with self.assertRaises(ValueError):
            effective_indexer_types({"num_hidden_layers": 2, "indexer_types": ["full", "bogus"]})


class TestLayerGroups(Glm53TestBase):
    def test_three_real_groups_cover_78_layers_exactly(self):
        self.assertEqual([g["id"] for g in self.groups],
                         ["full-dense", "shared-moe", "full-moe"])
        self.assertEqual([g["count"] for g in self.groups], [3, 57, 18])
        self.assertEqual([g["representative_layer_index"] for g in self.groups], [0, 3, 6])
        covered = [index for group in self.groups for index in group["layer_indices"]]
        self.assertEqual(sorted(covered), list(range(78)))
        self.assertEqual(len(set(covered)), len(covered))
        firsts = [min(group["layer_indices"]) for group in self.groups]
        self.assertEqual(firsts, sorted(firsts))
        for group in self.groups:
            self.assertIn(group["representative_layer_index"], group["layer_indices"])
            self.assertEqual(group["attention_type"], "indexed_attention")

    def test_indexer_and_dense_attribution(self):
        full = [g for g in self.groups if g["indexer_type"] == "full"]
        shared = [g for g in self.groups if g["indexer_type"] == "shared"]
        self.assertEqual(sum(g["count"] for g in full), 21)
        self.assertEqual(sum(g["count"] for g in shared), 57)
        dense = [g for g in self.groups if g["moe_type"] == "dense"]
        self.assertEqual([g["layer_indices"] for g in dense], [[0, 1, 2]])
        # 配置里 dense 层全是 full indexer：不存在 shared+dense 组合，不能虚构第四组
        self.assertEqual(self.values.indexer_type(0), "full")
        self.assertEqual(self.values.indexer_type(3), "shared")
        self.assertEqual(self.values.mlp_layer_type(2), "dense")
        self.assertEqual(self.values.mlp_layer_type(3), "sparse")


class TestAttentionSemantics(Glm53TestBase):
    def test_q_low_rank_chain_order_and_interleaved_rope(self):
        block = self.block_full
        ops = self.ops(block)
        order = {oid: i for i, oid in enumerate(
            [o["id"] for o in block["operators"]])}
        for first, second in (("g_q_a_proj", "g_q_a_norm"),
                              ("g_q_a_norm", "g_q_b_proj"),
                              ("g_q_b_proj", "g_q_view"),
                              ("g_q_view", "g_q_split"),
                              ("g_q_split", "g_q_rope"),
                              ("g_q_rope", "g_q_cat")):
            self.assertLess(order[first], order[second], (first, second))
        self.assertEqual(ops["g_q_rope"]["op"], "rope-interleaved")
        self.assertEqual(ops["g_q_rope"]["inputs"], ["g_q_rot", "g_cos", "g_sin"])
        tensors = self.tensors(block)
        self.assertEqual(tensors["g_cos"]["shape"], ["1", "seqlen", "Rk"])
        # 只有 rope 段被旋转；nope 段原样拼回
        self.assertEqual(ops["g_q_cat"]["inputs"], ["g_q_pass", "g_q_rot_emb"])
        self.assertNotEqual(tensors["g_q_pass"]["shape"][-1], "Rk")

    def test_cache_stores_compressed_latents_before_expansion(self):
        block = self.block_full
        ops = self.ops(block)
        tensors = self.tensors(block)
        order = {oid: i for i, oid in enumerate([o["id"] for o in block["operators"]])}
        cache = ops["g_cache_update"]
        self.assertEqual(cache["op"], "cache_update")
        self.assertEqual(sorted(cache["outputs"]),
                         ["g_cache_k_pass_post", "g_cache_k_rot_post"])
        self.assertEqual(tensors["g_cache_k_pass_pre"]["shape"],
                         ["B", "1", "kvcachelen", "Rkv"])
        self.assertEqual(tensors["g_cache_k_rot_pre"]["shape"],
                         ["B", "1", "kvcachelen", "Rk"])
        for tid in ("g_cache_k_pass_pre", "g_cache_k_rot_pre"):
            self.assertEqual(tensors[tid]["kind"], "cache")
            self.assertNotIn("Nh", tensors[tid]["shape"])
        self.assertLess(order["g_kv_a_norm"], order["g_cache_update"])
        self.assertLess(order["g_cache_update"], order["g_kv_b_proj"])
        self.assertEqual(ops["g_cache_update"]["inputs"],
                         ["g_cache_k_pass_pre", "g_cache_k_rot_pre",
                          "g_k_pass_4d", "g_k_rot_emb"])
        self.assertEqual(ops["g_k_pass_view"]["inputs"], ["g_k_pass_normed"])
        self.assertEqual(tensors["g_k_pass_normed"]["shape"], ["B", "seqlen", "Rkv"])
        self.assertEqual(tensors["g_k_pass_4d"]["shape"], ["B", "1", "seqlen", "Rkv"])
        self.assertLess(order["g_kv_a_norm"], order["g_k_pass_view"])
        self.assertLess(order["g_k_pass_view"], order["g_cache_update"])
        self.assertEqual(ops["g_kv_b_proj"]["inputs"], ["g_cache_k_pass_post"])
        self.assertEqual(tensors["g_kv_nope_flat"]["shape"][-1], "Nh×(Dn+Dv)")
        self.assertEqual(ops["g_key_cat"]["inputs"], ["g_k_nope", "g_k_rot_expanded"])
        self.assertEqual(tensors["g_v"]["shape"][-1], "Dv")

    def test_attention_core_scaling_and_mask(self):
        block = self.block_full
        ops = self.ops(block)
        order = {oid: i for i, oid in enumerate([o["id"] for o in block["operators"]])}
        self.assertEqual(ops["g_qk"]["parameter_refs"], ["Nh", "seqlen", "Lkv", "Dq"])
        self.assertIn("1/16", ops["g_qk"]["note"])
        self.assertEqual(ops["g_mask_add"]["inputs"], ["g_attn_logits", "g_sparse_mask"])
        self.assertEqual(ops["g_softmax"]["inputs"], ["g_attn_masked"])
        self.assertEqual(ops["g_av"]["inputs"], ["g_attn_probs", "g_v"])
        self.assertLess(order["g_softmax"], order["g_av"])
        self.assertEqual(ops["g_av"]["op"], "matmul")
        self.assertEqual(ops["g_attn_transpose"]["op"], "transpose+contiguous")
        self.assertEqual(ops["g_attn_reshape"]["inputs"], ["g_attn_heads_seq"])
        self.assertEqual(self.tensors(block)["g_attn_heads_seq"]["shape"],
                         ["B", "seqlen", "Nh", "Dv"])
        self.assertLess(order["g_av"], order["g_attn_transpose"])
        self.assertLess(order["g_attn_transpose"], order["g_attn_reshape"])
        self.assertEqual(self.tensors(block)["g_attn_flat"]["shape"][-1], "Nh×Dv")


class TestDsaIndexer(Glm53TestBase):
    def test_full_indexer_has_projections_cache_and_bias_layernorm(self):
        block = self.block_full
        ops = self.ops(block)
        ix_ops = sorted(oid for oid in ops if oid.startswith("g_ix_"))
        self.assertEqual(len(ix_ops), 17)
        self.assertEqual(ops["g_ix_q_proj"]["inputs"], ["g_q_residual"])
        self.assertEqual(ops["g_ix_k_proj"]["inputs"], ["g_attn_normed"])
        self.assertEqual(ops["g_ix_k_norm"]["op"], "layernorm")
        self.assertEqual(set(ops["g_ix_k_norm"]["weights"]),
                         {"g_ix_k_norm_bias", "g_ix_k_norm_weight"})
        self.assertIn("g_cache_indexer_keys_post", self.tensors(block))
        self.assertEqual(self.tensors(block)["g_cache_indexer_keys_post"]["shape"],
                         ["B", "Lidx", "Id"])
        self.assertEqual(ops["g_ix_topk"]["inputs"], ["g_ix_masked_scores"])
        self.assertEqual(self.tensors(block)["g_ix_topk_indices"]["shape"][-1], "Ksel")
        # indexer 打分前必须先写键缓存（否则选的是旧集合）
        order = {oid: i for i, oid in enumerate([o["id"] for o in block["operators"]])}
        self.assertLess(order["g_ix_cache_update"], order["g_ix_scores"])

    def test_shared_layer_has_no_indexer_and_reuses_previous_topk(self):
        block = self.block_shared
        ops = self.ops(block)
        tensors = self.tensors(block)
        self.assertFalse([oid for oid in ops if oid.startswith("g_ix_")])
        self.assertNotIn("g_cache_indexer_keys_post", tensors)
        self.assertEqual(ops["g_index_mask_scatter"]["inputs"],
                         ["g_prev_topk_indices"])
        note = tensors["g_prev_topk_indices"]["note"]
        self.assertIn("上一", note)
        self.assertEqual(tensors["g_prev_topk_indices"]["kind"], "boundary")


class TestSparseMask(Glm53TestBase):
    def test_index_mask_true_means_masked_and_is_4d(self):
        block = self.block_full
        ops = self.ops(block)
        tensors = self.tensors(block)
        mask = tensors["g_index_mask"]
        self.assertEqual(mask["shape"], ["B", "1", "seqlen", "Lidx"])
        self.assertIn("True", mask["note"])
        self.assertIn("屏蔽", mask["note"])
        self.assertIn("False", mask["note"])
        self.assertIn("选中", mask["note"])
        self.assertEqual(ops["g_index_mask_scatter"]["op"], "scatter+unsqueeze")
        self.assertEqual(ops["g_sparse_mask_apply"]["op"], "masked_fill")
        self.assertEqual(ops["g_sparse_mask_apply"]["inputs"],
                         ["g_attention_mask", "g_index_mask"])
        self.assertEqual(tensors["g_sparse_mask"]["shape"],
                         ["B", "1", "seqlen", "Lidx"])

    def test_shared_layer_uses_same_mask_polarity(self):
        mask = self.tensors(self.block_shared)["g_index_mask"]
        self.assertEqual(mask["shape"], ["B", "1", "seqlen", "Lidx"])
        self.assertIn("True", mask["note"])
        self.assertIn("屏蔽", mask["note"])


class TestMoeAndDense(Glm53TestBase):
    def test_dense_layer_has_no_moe_operators(self):
        ops = self.ops(self.block_dense)
        self.assertFalse([oid for oid in ops if oid.startswith("g_moe_")])
        for oid in ("g_dense_gate", "g_dense_up", "g_dense_act", "g_dense_down"):
            self.assertIn(oid, ops)
        self.assertEqual(self.tensors(self.block_dense)["g_dense_gate"]["shape"][-1],
                         "Is")

    def test_moe_router_and_single_expert_template(self):
        block = self.block_full
        ops = self.ops(block)
        tensors = self.tensors(block)
        self.assertEqual(ops["g_moe_sigmoid"]["op"], "sigmoid")
        order = {oid: i for i, oid in enumerate([o["id"] for o in block["operators"]])}
        self.assertLess(order["g_moe_bias_add"], order["g_moe_topk"])
        self.assertLess(order["g_moe_weight_norm"], order["g_moe_weight_scale"])
        self.assertEqual(ops["g_moe_topk"]["parameter_refs"], ["K", "E"])
        # 组掩码：scatter → expand → reshape → [Ttok,E] → masked_fill
        self.assertEqual(tensors["g_group_mask"]["shape"], ["Ttok", "Ng"])
        self.assertEqual(tensors["g_group_mask_expanded"]["shape"],
                         ["Ttok", "Ng", "E÷Ng"])
        self.assertEqual(tensors["g_score_mask"]["shape"], ["Ttok", "E"])
        self.assertLess(order["g_moe_group_scatter"], order["g_moe_group_expand"])
        self.assertLess(order["g_moe_group_expand"], order["g_moe_group_reshape"])
        self.assertLess(order["g_moe_group_reshape"],
                        order["g_moe_group_masked_fill"])
        self.assertEqual(ops["g_moe_group_masked_fill"]["inputs"],
                         ["g_scores_for_choice", "g_score_mask"])
        self.assertEqual(tensors["g_moe_experts_gate_up"]["shape"], ["E", "2Es", "H"])
        self.assertEqual(tensors["g_moe_experts_down"]["shape"], ["E", "H", "Es"])
        self.assertEqual(ops["g_moe_merge"]["inputs"], ["g_routed", "g_shared_out"])
        # token 路由的真实依赖：完整分配矩阵 + (top_k_pos, token_idx) 贯通
        self.assertEqual(tensors["g_expert_mask"]["shape"], ["E+1", "K", "Ttok"])
        self.assertEqual(ops["g_moe_expert_mask"]["op"], "one_hot+permute")
        self.assertEqual(ops["g_moe_expert_hit"]["op"], "sum+nonzero")
        gather = ops["g_moe_expert_gather_indices"]
        self.assertEqual(gather["inputs"], ["g_expert_mask", "g_expert_hit"])
        self.assertEqual(gather["outputs"],
                         ["g_expert_topk_pos", "g_expert_token_idx"])
        self.assertEqual(ops["g_moe_select_tokens"]["inputs"],
                         ["g_moe_flat", "g_expert_token_idx"])
        self.assertIn("g_expert_token_idx",
                      ops["g_moe_expert_weight"]["inputs"])
        self.assertIn("g_expert_topk_pos",
                      ops["g_moe_expert_weight"]["inputs"])
        self.assertEqual(ops["g_moe_index_add"]["inputs"],
                         ["g_routed_init", "g_expert_weighted", "g_expert_token_idx"])
        self.assertLess(order["g_moe_expert_gather_indices"],
                        order["g_moe_select_tokens"])
        self.assertLess(order["g_moe_expert_gather_indices"],
                        order["g_moe_expert_weight"])
        self.assertLess(order["g_moe_expert_weight"], order["g_moe_index_add"])
        # 共享专家是并行支路：消费 routed 之前的 residuals，不是 routed 输出
        self.assertEqual(ops["g_moe_shared_gate"]["inputs"], ["g_ffn_in"])
        self.assertFalse(any("hash" in oid for oid in ops))

    def test_moe_scale_and_group_facts(self):
        params = self.params(self.block_full)
        self.assertEqual(params["Scl"]["value"], 2.5)
        self.assertEqual(params["K"]["value"], 8)
        self.assertEqual(params["E"]["value"], 256)
        self.assertEqual(params["Es"]["value"], 2048)
        self.assertEqual(params["Ng"]["value"], 1)
        self.assertEqual(params["N_hit"]["value"], None)
        self.assertEqual(params["N_e"]["value"], None)


class TestNoForeignModelFacts(Glm53TestBase):
    def test_two_plain_residuals_without_mhc_or_compression(self):
        block = self.block_full
        add_ids = [o["id"] for o in block["operators"] if o["op"] == "add"]
        self.assertEqual(add_ids, ["g_res_attn", "g_moe_bias_add",
                                   "g_moe_merge", "g_res_ffn"])
        joined = " ".join([o["id"] for o in block["operators"]] +
                          [t["id"] for t in block["tensors"]]).lower()
        for token in ("mhc", "sink", "compress_rate", "sliding", "csa", "hca",
                      "kv_full", "tid2eid", "block_bias", "overlap"):
            self.assertNotIn(token, joined, token)

    def test_all_ids_use_model_namespace(self):
        for block in (self.block_dense, self.block_shared, self.block_full):
            for item in block["operators"] + block["tensors"]:
                self.assertTrue(str(item["id"]).startswith("g_"), item["id"])


class TestScenarios(Glm53TestBase):
    def test_scenario_values_are_independent_expectations(self):
        structure = build_layer_structure(REPO_ROOT, 6, MODEL_KEY)
        by_scenario = structure["scenario_derivations"]
        self.assertEqual([s["id"] for s in structure["scope"]["scenarios"]],
                         [s["id"] for s in GLM53_SCENARIOS])
        expect = {
            "prefill-empty": {"L_kv": 8, "L_idx": 8, "L_mask": 8, "K_sel": 8},
            "decode-nonzero": {"L_kv": 4097, "K_sel": 2048},
            "decode-long": {"L_kv": 131073, "K_sel": 2048},
            "decode-multi": {"L_kv": 4096, "K_sel": 2048},
        }
        for sid, values in expect.items():
            rows = {row["id"]: row for row in by_scenario[sid]}
            for did, value in values.items():
                self.assertEqual(rows[did].get("value"), value, (sid, did))
        symbolic = {row["id"]: row for row in by_scenario["prefill-symbolic"]}
        self.assertEqual(symbolic["L_kv"]["state"], "unknown")
        # 公式不含滑窗/压缩语义
        derivations = {d["id"]: d["expression"] for d in structure["block"]["derivations"]}
        self.assertEqual(derivations["L_kv"].replace(" ", ""), "kvcachelen+seqlen")
        self.assertEqual(derivations["K_sel"], "min(Ki, L_idx)")
        self.assertNotIn("%", derivations["L_kv"])


class TestNetworkOverview(Glm53TestBase):
    def test_layer_order_groups_and_mtp_boundary(self):
        network = self.network
        self.assertEqual(network["scope"]["entry_class"], "GlmMoeDsaForCausalLM")
        self.assertEqual(network["scope"]["layer_count"], 78)
        self.assertEqual([entry["index"] for entry in network["layer_order"]],
                         list(range(78)))
        group_of = {index: group["id"] for group in network["groups"]
                    for index in group["layer_indices"]}
        for entry in network["layer_order"]:
            self.assertEqual(entry["group_id"], group_of[entry["index"]])
            self.assertEqual(entry["attention_type"], "indexed_attention")
        self.assertEqual(network["scope"]["dense_layers"], [0, 1, 2])
        self.assertEqual(network["scope"]["indexer_full_layers"],
                         [0, 1, 2] + list(range(6, 78, 4)))
        ops = {o["id"]: o for o in network["path"]["operators"]}
        self.assertEqual(list(ops), [
            "net_embed", "net_position_ids", "net_mask_build", "net_rope",
            "net_layer_loop", "net_final_norm", "net_logits_slice", "net_lm_head"])
        self.assertEqual(ops["net_layer_loop"]["parameter_refs"], ["Nlay", "B", "seqlen"])
        for required in ("cos_rope", "sin_rope", "causal_mask", "past_key_values"):
            self.assertIn(required, ops["net_layer_loop"]["inputs"])
        self.assertNotIn("hidden_streams", " ".join(ops["net_layer_loop"]["inputs"]))
        params = {p["id"]: p for p in network["path"]["parameters"]}
        self.assertEqual(params["Nmtp"]["value"], 1)
        self.assertIn("不", params["Nmtp"]["note"])
        self.assertIn("MTP", " ".join(network["scope"]["notes"]))
        self.assertIn("视觉", " ".join(network["scope"]["notes"]))
        self.assertFalse(any("vision" in oid or "image" in oid for oid in ops))


class TestContractsRejectMutations(Glm53TestBase):
    def test_valid_documents_pass(self):
        structure = build_layer_structure(REPO_ROOT, 6, MODEL_KEY)
        self.assertEqual(validate_hf_layer_structure(structure), [])
        import hashlib
        shas = {k: hashlib.sha256((REPO_ROOT / v).read_bytes()).hexdigest()
                for k, v in SOURCE_FILES.items()}
        _, document = build_layer_documents(REPO_ROOT, 6, MODEL_KEY)
        self.assertEqual(validate_hf_reading_document(document, shas), [])
        self.assertEqual(validate_hf_network_overview(self.network, shas), [])

    def test_mutated_attention_and_indexer_are_rejected(self):
        structure = build_layer_structure(REPO_ROOT, 6, MODEL_KEY)
        # 1) 把缓存写成展开后的多头 K/V
        broken = copy.deepcopy(structure)
        for tensor in broken["block"]["tensors"]:
            if tensor["id"] == "g_cache_k_pass_pre":
                tensor["shape"] = ["B", "Nh", "kvcachelen", "Rkv"]
        issues = validate_hf_layer_structure(broken)
        self.assertTrue(any("Nh" in issue or "缓存压缩" in issue for issue in issues), issues)
        # 2) full 层删掉 indexer 键缓存
        broken = copy.deepcopy(structure)
        broken["block"]["tensors"] = [t for t in broken["block"]["tensors"]
                                      if t["id"] != "g_cache_indexer_keys_post"]
        broken["block"]["operators"] = [o for o in broken["block"]["operators"]
                                        if o["id"] != "g_ix_cache_update"]
        issues = validate_hf_layer_structure(broken)
        self.assertTrue(issues)

    def test_mutated_scaling_and_moe_rejected(self):
        structure = build_layer_structure(REPO_ROOT, 6, MODEL_KEY)
        broken = copy.deepcopy(structure)
        for param in broken["block"]["parameters"]:
            if param["id"] == "Dq":
                param["value"] = 192  # 按根配置 head_dim 误读
        issues = validate_hf_layer_structure(broken)
        self.assertTrue(any("Dq" in issue for issue in issues), issues)
        broken = copy.deepcopy(structure)
        for op in broken["block"]["operators"]:
            if op["id"] == "g_moe_sigmoid":
                op["op"] = "softmax"
        issues = validate_hf_layer_structure(broken)
        self.assertTrue(any("sigmoid" in issue for issue in issues), issues)

    def test_mutated_network_group_coverage_rejected(self):
        broken = copy.deepcopy(self.network)
        broken["groups"][0]["layer_indices"] = broken["groups"][0]["layer_indices"][:-1]
        broken["groups"][0]["count"] -= 1
        import hashlib
        shas = {k: hashlib.sha256((REPO_ROOT / v).read_bytes()).hexdigest()
                for k, v in SOURCE_FILES.items()}
        issues = validate_hf_network_overview(broken, shas)
        self.assertTrue(any("覆盖" in issue or "count" in issue for issue in issues), issues)


class TestBuildPipelineAndArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _workspace_ready():
            raise unittest.SkipTest("缺少 GLM-5.3 输入")
        import tempfile
        cls.tmp = tempfile.TemporaryDirectory(prefix="glm53-build-")
        cls.project = Path(cls.tmp.name) / "structure_reader"
        cls.project.mkdir(parents=True)
        # 工具源按工程根复算：假工程根镜像真实 src/configs（不触碰真实 outputs）
        (cls.project / "src").symlink_to(SRC_DIR, target_is_directory=True)
        (cls.project / "configs").symlink_to(
            TESTS_DIR.parent / "configs", target_is_directory=True)
        cls.result, cls.documents = run_hf_build(
            REPO_ROOT, cls.project, model_key=MODEL_KEY, all_variants=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_capability_limits_are_named(self):
        with self.assertRaises(HfBuildError) as ctx:
            run_hf_build(REPO_ROOT, self.project, model_key=MODEL_KEY, all_variants=True,
                         onnx_modules=True)
        self.assertIn("ONNX", str(ctx.exception))
        with self.assertRaises(HfBuildError) as ctx:
            run_hf_build(REPO_ROOT, self.project, model_key=MODEL_KEY)
        self.assertIn("--all-variants", str(ctx.exception))
        with self.assertRaises(HfBuildError) as ctx:
            run_hf_build(REPO_ROOT, self.project, model_key="not_a_model",
                         all_variants=True)
        self.assertIn("未知模型键", str(ctx.exception))

    def test_documents_and_html(self):
        docs = self.documents
        self.assertEqual(set(docs), {
            "README.md", "run_manifest.json", "data/layer_structures.json",
            "data/network_overview.json", "data/effective_config.json",
            "reports/source_manifest.json", "tables/decoder_variants.xlsx",
            "viewer/structure.html"})
        html = docs["viewer/structure.html"].decode("utf-8")
        for gid in ("full-dense", "shared-moe", "full-moe"):
            for suffix in ("", "/block", "/ops"):
                self.assertIn(f"#/group/{gid}{suffix}", html)
        self.assertIn('id="scn-data"', html)
        self.assertIn("decode-long", html)
        self.assertNotIn("swa-hash", html)
        self.assertNotIn("DeepseekV4", html)
        self.assertNotIn("mHC 权重", html)
        self.assertIn("压缩 latent", html)
        self.assertGreaterEqual(html.count("<svg"), 40)
        manifest = docs["run_manifest.json"].decode("utf-8")
        self.assertIn(self.result.analysis_id, manifest)

    def test_workbook_two_pages_three_groups_and_live_formulas(self):
        data = self.documents["tables/decoder_variants.xlsx"]
        book = load_workbook(io.BytesIO(data))
        self.assertEqual(book.sheetnames, ["Prefill", "Decode"])
        for name, (seq, kv) in (("Prefill", (8, 0)), ("Decode", (1, 4096))):
            ws = book[name]
            banner = next(r for r in range(1, ws.max_row + 1)
                          if str(ws.cell(r, 1).value).startswith("性能配置与场景输入"))
            input_rows = {}
            r = banner + 2
            while r <= ws.max_row and ws.cell(r, 1).value:
                input_rows[ws.cell(r, 1).value] = r
                r += 1
            self.assertEqual(ws.cell(input_rows["seqlen"], 4).value, seq)
            self.assertEqual(ws.cell(input_rows["kvcachelen"], 4).value, kv)
            self.assertEqual(ws.cell(input_rows["batch_size"], 4).value, 1)
            # 性能区启用后有两个可改吞吐输入
            self.assertEqual(ws.cell(input_rows["cube_mac_per_ms"], 4).value, 4096)
            self.assertEqual(ws.cell(input_rows["vec_elem_per_ms"], 4).value, 64)
            group_rows = {gid: r for gid in ("full-dense", "shared-moe", "full-moe")
                          for r in range(1, ws.max_row + 1)
                          if isinstance(ws.cell(r, 1).value, str)
                          and f"（{gid}）" in ws.cell(r, 1).value}
            self.assertEqual(len(group_rows), 3, group_rows)
            self.assertEqual(sorted(group_rows), ["full-dense", "full-moe", "shared-moe"])
            for gid, count in (("full-dense", 3), ("shared-moe", 57), ("full-moe", 18)):
                facts = {ws.cell(r, 1).value: ws.cell(r, 2).value
                         for r in range(group_rows[gid], group_rows[gid] + 9)}
                self.assertIn(f"× {count}", str(facts["循环归属"]))
                self.assertEqual(str(facts["真实层索引"]).count(",") + 1, count)
            param_rows = {}
            for r in range(1, ws.max_row + 1):
                key = ws.cell(r, 1).value
                if isinstance(key, str):
                    param_rows.setdefault(key, r)
            for pid in ("Lkv", "Lidx", "Lmask", "Ksel", "Ttok"):
                self.assertTrue(str(ws.cell(param_rows[pid], 4).value).startswith("="))
            self.assertIn("MIN", str(ws.cell(param_rows["Ksel"], 4).value))
            # N_hit 参数行引用本表可改输入（默认未知），不再有“参数区未知、输入区有值”
            # 两套口径；输入空白时公式在 Excel 里显示未知。
            self.assertTrue(str(ws.cell(param_rows["N_hit"], 4).value).startswith("="))
            self.assertIn(f"$D${input_rows['expert_hit']}",
                          str(ws.cell(param_rows["N_hit"], 4).value))
            self.assertTrue(str(ws.cell(input_rows["expert_hit"], 4).value).startswith("未知"))
            # Lkv 公式引用本表 seqlen 的可改单元格（不写死行号）
            self.assertIn(f"$D${input_rows['seqlen']}",
                          str(ws.cell(param_rows["Lkv"], 4).value))
            op_headers = [r for r in range(1, ws.max_row + 1)
                          if ws.cell(r, 1).value == "黑盒标记"
                          and ws.cell(r, 2).value == "算子id"]
            # 三个层组的算子区 + 整网外围块（T11 起外围也逐算子计成本）
            self.assertEqual(len(op_headers), 4)
            op_count = 0
            cost_rows = 0
            for header in op_headers:
                r = header + 1
                while ws.cell(r, 1).value in ("已展开", "黑盒待补", "↳ 同算子分项"):
                    value = ws.cell(r, 16).value
                    if isinstance(value, str) and value.startswith("="):
                        cost_rows += 1
                    elif isinstance(value, str):
                        self.assertTrue(value.startswith("未知") or
                                        "整网一次性项" in value or value == "—", value)
                    op_count += 1
                    r += 1
            # 3 个层组 193 个算子 + 整网外围 8 个算子
            self.assertEqual(op_count, 201)
            # 性能区：成本列是活公式（或明确的未知/引用），不再是整列“待提供”。
            self.assertGreater(cost_rows, 150)

    def test_all_normal_anchors_resolve_and_ids_are_unique(self):
        """P2 导航回归：普通 # 锚点目标必须存在、id 唯一；讲解/引用用真实组作用域。"""
        import re

        html = self.documents["viewer/structure.html"].decode("utf-8")
        ids = re.findall(r'\sid="([^"]+)"', html)
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        self.assertEqual(duplicates, [], f"重复 id: {duplicates[:8]}")
        id_set = set(ids)
        hrefs = re.findall(r'href="#([^"]+)"', html)
        normal = [value for value in hrefs if not value.startswith("/")]
        missing = sorted({value for value in normal if value not in id_set})
        self.assertEqual(missing, [], f"普通锚点缺少目标: {missing[:8]}")
        self.assertNotIn('href="#mod-block-glm_', html)
        self.assertNotIn('href="#t-block-g_', html)
        self.assertNotIn('href="#p-block-E"', html)
        for gid, module in (("full-dense", "glm_attn_q"),
                            ("shared-moe", "glm_moe_expert"),
                            ("full-moe", "glm_moe_expert")):
            for anchor in (f"mod-block-{gid}-{module}",
                           f"p-block-{gid}-Rk",
                           f"t-block-{gid}-g_hidden_in",
                           f"op-block-{gid}-g_input_norm"):
                self.assertIn(f'id="{anchor}"', html, anchor)
                self.assertIn(f'href="#{anchor}"', html, anchor)

    def test_workbook_spec_rejects_bad_attribution(self):
        container = build_structures_container(
            REPO_ROOT, model_key=MODEL_KEY,
            values_loader=lambda root, key: load_hf_model_values(root, model_key=key),
            groups_fn=layer_group_assignments,
            build_documents_fn=lambda root, layer_index, model_key: build_layer_documents(
                root, layer_index=layer_index, model_key=model_key))
        broken = copy.deepcopy(container)
        broken["variants"][0]["representative_layer_index"] = 5
        with self.assertRaises(ValueError):
            build_structures_workbook_spec(broken, None)
        bad_indexer = copy.deepcopy(container)
        bad_indexer["variants"][0].pop("indexer_type")
        with self.assertRaises(ValueError):
            build_structures_workbook_spec(bad_indexer)


class TestPageProfileIsModelSpecific(Glm53TestBase):
    def test_profile_uses_glm_modules_and_labels(self):
        groups = GLM53_PAGE_PROFILE.module_groups(self.block_full)
        ids = [g["id"] for g in groups]
        self.assertIn("glm_attn_indexer", ids)
        self.assertIn("glm_moe_route", ids)
        self.assertNotIn("attn_csa", ids)
        self.assertEqual(sum(len(g["ops"]) for g in groups), 79)
        overview = GLM53_PAGE_PROFILE.overview_graph(self.block_full)
        node_ids = {node["id"] for node in overview["nodes"]}
        self.assertIn("m:glm_attn_indexer", node_ids)
        self.assertIn("m:glm_attn_cache", node_ids)
        narrative = GLM53_PAGE_PROFILE.module_narrative(self.block_full, "glm_attn_cache")
        self.assertIn("压缩 latent", narrative)
        self.assertIn("glm_attn_cache", narrative)

    def test_shared_profile_has_no_indexer_module(self):
        groups = GLM53_PAGE_PROFILE.module_groups(self.block_shared)
        self.assertNotIn("glm_attn_indexer", [g["id"] for g in groups])


if __name__ == "__main__":
    unittest.main(verbosity=2)
