"""GLM-5.3-Flash（zai-org/GLM-5.3-Flash, glm5_next）结构、场景、契约与产物回归。

只测当前源码行为，不读旧 outputs、不依赖既有候选、不导入/运行 HF 模型、不加载
权重。测试想发现的是「把 GLM-5.3/DeepSeek 事实套进来」或「状态/缓存/层归属/
掩码写错」这类真实错误，不是照实现复制断言：

- 身份：官方 Flash config revision/model_type/architectures、text_config、
  checkpoint 5.16 vs 本地 5.19、vision 边界；
- 层归属：三组（kda-dense 3 / dsa-moe 11 / kda-moe 31）恰好覆盖 45 层，
  DSA 层 3,7,…,43、dense 层 0/1/2、indexer 全 full；
- KDA：q/k/v、因果短卷积+固定 conv state、forget gate/beta、delta-rule 扫描+
  固定 recurrent state、gated RMSNorm；没有 RoPE/indexer/普通残差；
- MLA/DSA：Q 低秩+NoPE、压缩 latent 缓存、kv_b 在缓存后展开、packed 缓存、
  k-pool 池 top-k + tail、counts 稀疏 mask（True=可见）、2051 输出宽度；
- mHC：两处 4 路流坍缩/pre/post/comb(Sinkhorn)/展开/混合；不是普通 add；
- MoE：sigmoid + 修正偏置 + 288 选 8 + 归一 + ×2.5；单份专家模板；共享专家并行；
  SwiGLU clamp；dense 层没有 MoE；
- 场景：空缓存/非零历史/超过 index_topk/多 token 的独立期望值；
- 契约：改动结构/网络后检查必须拒绝（非空问题串）；
- 产物：真实 HTML 与两页工作簿（三组、活公式、成本待提供、无其它模型字段）。

运行：
  cd structure_reader && PYTHONPATH=src python3 tests/test_glm53flash_structure.py
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

from structure_reader.build_hf_decoder import HfBuildError, run_hf_build  # noqa: E402
from structure_reader.checks.glm53flash_hf_contracts import (  # noqa: E402
    validate_hf_layer_structure,
    validate_hf_network_overview,
    validate_hf_reading_document,
)
from structure_reader.extract.glm53flash_hf import (  # noqa: E402
    build_layer_documents,
    build_layer_structure,
)
from structure_reader.extract.glm53flash_network import build_network_overview  # noqa: E402
from structure_reader.extract.hf_layer_structures import build_structures_container  # noqa: E402
from structure_reader.export.glm53flash_workbook import (  # noqa: E402
    build_structures_workbook_spec,
    export_structures_workbook,
)
from structure_reader.render.hf_model_profile import GLM53FLASH_PAGE_PROFILE  # noqa: E402
from structure_reader.rules import glm53flash_hf as flash_rules  # noqa: E402
from structure_reader.rules.glm53flash_hf import (  # noqa: E402
    GLM53FLASH_SCENARIOS,
    MODEL_REPOSITORY,
    SOURCE_FILES,
    dsa_layer_indices,
    kda_layer_indices,
    layer_group_assignments,
    load_hf_model_values,
    verify_source_fingerprints,
)

MODEL_KEY = "glm_5_3_flash_hf"
GROUP_IDS = ("kda-dense", "dsa-moe", "kda-moe")
GROUP_COUNTS = {"kda-dense": 3, "dsa-moe": 11, "kda-moe": 31}


def _workspace_ready() -> bool:
    return (REPO_ROOT / SOURCE_FILES["config.json"]).is_file() and (
        REPO_ROOT / SOURCE_FILES["modeling_glm5_next.py"]).is_file()


def _shas() -> dict:
    return {k: hashlib.sha256((REPO_ROOT / v).read_bytes()).hexdigest()
            for k, v in SOURCE_FILES.items()}


def _kda_step_correct(state, q, k, v, g, beta):
    """T06 复核用的小型代数：A=Spre*exp(g), mem=A^T k, delta=beta*(v-mem),
    S=A+k delta^T, o=S^T q（全部纯 Python，不导入 HF/torch）。"""
    import math
    A = [[state[i][j] * math.exp(g[i]) for j in range(len(v))]
         for i in range(len(k))]
    mem = [sum(A[i][j] * k[i] for i in range(len(k))) for j in range(len(v))]
    delta = [beta * (v[j] - mem[j]) for j in range(len(v))]
    S = [[A[i][j] + k[i] * delta[j] for j in range(len(v))]
         for i in range(len(k))]
    o = [sum(S[i][j] * q[i] for i in range(len(k))) for j in range(len(v))]
    return o, S


def _kda_step_wrong_undecayed(state, q, k, v, g, beta):
    """故意错误：误差项用未衰减 Sprev，而不是 A。"""
    import math
    A = [[state[i][j] * math.exp(g[i]) for j in range(len(v))]
         for i in range(len(k))]
    mem = [sum(state[i][j] * k[i] for i in range(len(k))) for j in range(len(v))]
    delta = [beta * (v[j] - mem[j]) for j in range(len(v))]
    S = [[A[i][j] + k[i] * delta[j] for j in range(len(v))]
         for i in range(len(k))]
    o = [sum(S[i][j] * q[i] for i in range(len(k))) for j in range(len(v))]
    return o, S


def _kda_step_wrong_direction(state, q, k, v, g, beta):
    """故意错误：mem 按 V 维而不是 K 维（方向错误）。"""
    import math
    A = [[state[i][j] * math.exp(g[i]) for j in range(len(v))]
         for i in range(len(k))]
    mem = [sum(A[i][j] * k[j] for j in range(len(v))) for i in range(len(k))]
    delta = [beta * (v[j] - mem[j]) for j in range(len(v))]
    S = [[A[i][j] + k[i] * delta[j] for j in range(len(v))]
         for i in range(len(k))]
    o = [sum(S[i][j] * q[i] for i in range(len(k))) for j in range(len(v))]
    return o, S


@unittest.skipUnless(_workspace_ready(), "缺少 Flash config / glm5_next 源码；如实跳过")
class FlashTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = load_hf_model_values(REPO_ROOT, model_key=MODEL_KEY)
        cls.groups = layer_group_assignments(cls.values)
        cls.block_dense = build_layer_structure(REPO_ROOT, 0, MODEL_KEY)["block"]
        cls.block_dsa = build_layer_structure(REPO_ROOT, 3, MODEL_KEY)["block"]
        cls.block_moe = build_layer_structure(REPO_ROOT, 4, MODEL_KEY)["block"]
        cls.network = build_network_overview(REPO_ROOT, model_key=MODEL_KEY)
        cls.shas = _shas()
        # 三类文档 + 容器 + 产物只构建一次
        cls.structure_dense, cls.doc_dense = build_layer_documents(REPO_ROOT, 0, MODEL_KEY)
        cls.structure_dsa, cls.doc_dsa = build_layer_documents(REPO_ROOT, 3, MODEL_KEY)
        cls.container = build_structures_container(
            REPO_ROOT, MODEL_KEY,
            values_loader=lambda root, key: load_hf_model_values(root, model_key=key),
            groups_fn=layer_group_assignments,
            build_documents_fn=lambda root, layer_index, model_key: build_layer_documents(
                root, layer_index=layer_index, model_key=model_key))
        cls.workbook_bytes = export_structures_workbook(cls.container, cls.network)
        cls.result, cls.documents = run_hf_build(
            REPO_ROOT, REPO_ROOT / "structure_reader",
            model_key=MODEL_KEY, all_variants=True)

    @staticmethod
    def ops(block):
        return {o["id"]: o for o in block["operators"]}

    @staticmethod
    def tensors(block):
        return {t["id"]: t for t in block["tensors"]}

    @staticmethod
    def params(block):
        return {p["id"]: p for p in block["parameters"]}


class TestIdentity(FlashTestBase):
    def test_official_repository_identity(self):
        self.assertEqual(MODEL_REPOSITORY["repo_id"], "zai-org/GLM-5.3-Flash")
        self.assertEqual(MODEL_REPOSITORY["revision"],
                         "46a050bec13002ad29ee795c16e61cf4fa3c4a9c")
        self.assertEqual(MODEL_REPOSITORY["model_type"], "glm5_next")
        self.assertEqual(MODEL_REPOSITORY["architectures"],
                         ["Glm5NextForConditionalGeneration"])
        self.assertEqual(MODEL_REPOSITORY["checkpoint_transformers_version"], "5.16.0")
        self.assertEqual(self.values.model_revision, MODEL_REPOSITORY["revision"])
        self.assertTrue(self.values.source_revision)
        self.assertEqual(self.values.config_path, "models/zai-org/GLM-5.3-Flash/config.json")

    def test_text_config_and_vision_boundary(self):
        self.assertEqual(self.values.vision_depth, 24)
        self.assertEqual(self.values.image_token_id, 154854)
        self.assertEqual(self.values.dtype, "bfloat16")
        self.assertEqual(self.values.pad_token_id, 154820)

    def test_no_rope_and_head_dim(self):
        self.assertEqual(self.values.qk_rope_head_dim, 0)
        self.assertEqual(self.values.head_dim, 0)
        self.assertEqual(self.values.config_head_dim_raw, 0)
        self.assertEqual(self.values.qk_head_dim, 256)
        self.assertEqual(self.values.qk_nope_head_dim, 256)
        self.assertTrue(self.values.no_rope)
        self.assertAlmostEqual(self.values.attention_scaling, 1 / 16)

    def test_45_layers_and_hybrid_pattern(self):
        self.assertEqual(self.values.num_hidden_layers, 45)
        self.assertEqual(dsa_layer_indices(self.values), list(range(3, 44, 4)))
        self.assertEqual(len(dsa_layer_indices(self.values)), 11)
        self.assertEqual(len(kda_layer_indices(self.values)), 34)
        self.assertEqual(self.values.layer_types[4], "linear_attention")
        self.assertEqual(self.values.layer_types[44], "linear_attention")
        self.assertEqual(self.values.mlp_layer_types[:4],
                         ["dense", "dense", "dense", "sparse"])
        self.assertEqual(self.values.mlp_layer_types.count("sparse"), 42)

    def test_indexer_and_kpool(self):
        self.assertEqual(set(self.values.indexer_types), {"full"})
        self.assertEqual(self.values.index_kpool, 4)
        self.assertTrue(self.values.index_kpool_always_select_tail)
        self.assertEqual(self.values.index_topk, 2048)
        self.assertEqual(self.values.index_n_heads, 32)
        self.assertEqual(self.values.index_head_dim, 128)

    def test_kda_and_mhc_params(self):
        self.assertEqual(self.values.linear_num_heads, 64)
        self.assertEqual(self.values.linear_head_dim, 128)
        self.assertEqual(self.values.linear_conv_kernel_dim, 4)
        self.assertEqual(self.values.linear_lower_bound, -5.0)
        self.assertEqual(self.values.hc_mult, 4)
        self.assertEqual(self.values.hc_sinkhorn_iters, 20)
        self.assertAlmostEqual(self.values.hc_eps, 1e-6)
        self.assertAlmostEqual(self.values.swiglu_limit, 10.0)

    def test_moe_params(self):
        self.assertEqual(self.values.n_routed_experts, 288)
        self.assertEqual(self.values.num_experts_per_tok, 8)
        self.assertEqual(self.values.n_shared_experts, 1)
        self.assertEqual(self.values.moe_intermediate_size, 2048)
        self.assertEqual(self.values.intermediate_size, 12288)
        self.assertEqual(self.values.routed_scaling_factor, 2.5)
        self.assertEqual(self.values.scoring_func, "sigmoid")
        self.assertTrue(self.values.norm_topk_prob)

    def test_eps_and_mtp(self):
        self.assertAlmostEqual(self.values.rms_norm_eps, 1e-5)
        self.assertAlmostEqual(self.values.sub_norm_eps, 1e-6)
        self.assertEqual(self.values.num_nextn_predict_layers, 1)
        self.assertEqual(self.values.max_position_embeddings, 1048576)
        self.assertFalse(self.values.tie_word_embeddings)

    def test_global_display_does_not_rename_flow_index_j(self):
        from structure_reader.render.display_names import expand_display_text
        text = "sum_j comb[j,k]·residual[j,d]"
        expanded = expand_display_text(text)
        self.assertNotIn("query_index", expanded)
        self.assertIn("comb[j,k]", expanded)
        self.assertIn("residual[j,d]", expanded)

    def test_source_fingerprints_verify(self):
        verify_source_fingerprints(REPO_ROOT)
        self.assertEqual(set(flash_rules.ANALYZED_SOURCE_FINGERPRINTS), set(SOURCE_FILES))


class TestGroups(FlashTestBase):
    def test_three_groups_cover_45_layers(self):
        self.assertEqual([g["id"] for g in self.groups], list(GROUP_IDS))
        covered = [i for g in self.groups for i in g["layer_indices"]]
        self.assertEqual(sorted(covered), list(range(45)))
        self.assertEqual([g["count"] for g in self.groups], [3, 11, 31])
        self.assertEqual([g["representative_layer_index"] for g in self.groups], [0, 3, 4])

    def test_group_attention_and_indexer_types(self):
        by_id = {g["id"]: g for g in self.groups}
        self.assertEqual(by_id["kda-dense"]["attention_type"], "linear_attention")
        self.assertEqual(by_id["kda-moe"]["attention_type"], "linear_attention")
        self.assertEqual(by_id["dsa-moe"]["attention_type"], "indexed_attention")
        for gid in ("kda-dense", "kda-moe"):
            self.assertEqual(by_id[gid]["indexer_type"], "none")
        self.assertEqual(by_id["dsa-moe"]["indexer_type"], "full")

    def test_layer_order_matches_groups(self):
        order = self.network["layer_order"]
        self.assertEqual([e["index"] for e in order], list(range(45)))
        group_of = {i: g["id"] for g in self.groups for i in g["layer_indices"]}
        for entry in order:
            self.assertEqual(entry["group_id"], group_of[entry["index"]])
            if entry["attention_type"] == "indexed_attention":
                self.assertEqual(entry["indexer_type"], "full")
            else:
                self.assertEqual(entry["indexer_type"], "none")


class TestKdaBlock(FlashTestBase):
    def test_params(self):
        p = self.params(self.block_dense)
        for pid, value in (("H", 4096), ("Hc", 4), ("NhL", 64), ("DhL", 128),
                           ("Kconv", 4), ("EpsN", 1e-5), ("GateLB", -5.0),
                           ("Sit", 20), ("Swi", 10.0), ("Mhc", 24)):
            self.assertEqual(p[pid]["value"], value, pid)
        self.assertNotIn("Lkv", p)
        self.assertNotIn("Ksel", p)
        self.assertIsNone(p["seqlen"]["value"])

    def test_short_conv_and_fixed_states(self):
        t = self.tensors(self.block_dense)
        self.assertEqual(t["g_kda_conv_state_pre"]["shape"],
                         ["B", "3×NhL×DhL", "Kconv"])
        self.assertEqual(t["g_kda_state_pre"]["shape"],
                         ["B", "NhL", "DhL", "DhL"])
        ops = self.ops(self.block_dense)
        self.assertEqual(ops["g_kda_conv1d"]["op"], "conv1d")
        self.assertEqual(ops["g_kda_conv_state_update"]["op"], "cache_update")
        self.assertIn("g_kda_conv_weight", ops["g_kda_conv1d"]["weights"])

    def test_scan_is_delta_rule_not_attention(self):
        ops = self.ops(self.block_dense)
        scan = ops["g_kda_scan"]
        self.assertEqual(scan["op"], "scan")
        for required in ("g_kda_state_pre", "g_kda_g", "g_kda_beta",
                         "g_kda_k_hat", "g_kda_v", "g_kda_q_scaled"):
            self.assertIn(required, scan["inputs"])
        self.assertEqual(set(scan["outputs"]),
                         {"g_kda_core_out", "g_kda_state_new"})
        for token in ("delta-rule", "recurrent", "chunk", "S_t"):
            self.assertIn(token, scan["note"])
        self.assertEqual(ops["g_kda_state_write"]["op"], "cache_update")

    def test_forget_gate_and_gated_norm(self):
        ops = self.ops(self.block_dense)
        self.assertEqual(ops["g_kda_decay"]["op"], "sigmoid+scale")
        self.assertIn("GateLB", ops["g_kda_decay"]["parameter_refs"])
        self.assertEqual(ops["g_kda_q_l2"]["op"], "l2norm")
        self.assertEqual(ops["g_kda_k_l2"]["op"], "l2norm")
        self.assertEqual(ops["g_kda_o_norm"]["op"], "rmsnorm-gated")
        self.assertIn("g_kda_gate", ops["g_kda_o_norm"]["inputs"])
        self.assertIn("-5", ops["g_kda_decay"]["note"])

    def test_no_dsa_or_rope_or_plain_residual(self):
        ops = self.ops(self.block_dense)
        self.assertFalse([oid for oid in ops if oid.startswith("g_ix_") or oid.startswith("g_dsa_")])
        self.assertFalse([oid for oid in ops if "rope" in oid])
        self.assertNotIn("g_res_attn", ops)
        self.assertNotIn("g_res_ffn", ops)
        self.assertNotIn("g_qk", ops)
        self.assertNotIn("g_index_mask_scatter", ops)

    def test_dense_mlp_has_swiglu_clamp(self):
        ops = self.ops(self.block_dense)
        self.assertEqual(ops["g_dense_clamp"]["op"], "clamp")
        self.assertIn("Swi", ops["g_dense_clamp"]["parameter_refs"])
        self.assertEqual(ops["g_dense_act"]["op"], "silu*mul")
        self.assertFalse([oid for oid in ops if oid.startswith("g_moe_")])


class TestDsaBlock(FlashTestBase):
    def test_params(self):
        p = self.params(self.block_dsa)
        for pid, value in (("Nh", 64), ("Dn", 256), ("Dv", 256), ("Dq", 256),
                           ("Rk", 0), ("Rq", 1536), ("Rkv", 512), ("Ih", 32),
                           ("Id", 128), ("Ipack", 257), ("Ki", 2048),
                           ("Kpool", 4), ("EpsA", 1e-6)):
            self.assertEqual(p[pid]["value"], value, pid)
        self.assertIsNone(p["kvcachelen"]["value"])
        for pid in ("Lkv", "Lidx", "Npool", "SelP", "Ksel", "Kout"):
            self.assertIsNone(p[pid]["value"], pid)

    def test_nope_no_rope_or_split(self):
        ops = self.ops(self.block_dsa)
        for forbidden in ("g_q_split", "g_q_rope", "g_q_cat", "g_kv_rope",
                          "g_ix_rope", "g_ix_q_split", "g_ix_k_split"):
            self.assertNotIn(forbidden, ops)
        t = self.tensors(self.block_dsa)
        self.assertEqual(t["g_dsa_q"]["shape"], ["B", "Nh", "seqlen", "Dq"])
        self.assertEqual(t["g_dsa_key"]["shape"], ["B", "Nh", "Lkv", "Dq"])
        self.assertEqual(t["g_dsa_k_rot"]["shape"], ["B", "seqlen", "Rk"])

    def test_compressed_latent_cache_and_expand_after(self):
        t = self.tensors(self.block_dsa)
        ops = self.ops(self.block_dsa)
        self.assertEqual(t["g_cache_kv_pre"]["shape"], ["B", "1", "kvcachelen", "Rkv"])
        self.assertEqual(t["g_cache_kv_post"]["shape"], ["B", "1", "Lkv", "Rkv"])
        self.assertEqual(t["g_cache_krot_pre"]["shape"], ["B", "1", "kvcachelen", "Rk"])
        self.assertNotIn("Nh", t["g_cache_kv_pre"]["shape"])
        self.assertIn("g_dsa_k_pass_4d", ops["g_dsa_cache_kv_update"]["inputs"])
        self.assertIn("g_cache_kv_post", ops["g_dsa_kv_b_proj"]["inputs"])
        order = {o["id"]: i for i, o in enumerate(self.block_dsa["operators"])}
        self.assertLess(order["g_dsa_cache_kv_update"], order["g_dsa_kv_b_proj"])

    def test_indexer_packed_and_pool(self):
        t = self.tensors(self.block_dsa)
        ops = self.ops(self.block_dsa)
        self.assertEqual(t["g_dsa_ix_packed"]["shape"], ["B", "seqlen", "Ipack"])
        self.assertEqual(t["g_dsa_ix_cache_pre"]["shape"],
                         ["B", "kvcachelen", "Ipack"])
        self.assertEqual(ops["g_dsa_ix_pool"]["op"], "pool+softmax")
        self.assertEqual(t["g_dsa_ix_pool_keys"]["shape"], ["B", "Npool", "Id"])
        self.assertEqual(t["g_dsa_ix_pool_indices"]["shape"],
                         ["B", "Npool", "Kpool"])
        self.assertEqual(t["g_dsa_ix_tail_indices"]["shape"],
                         ["B", "seqlen", "Kpool-1"])
        self.assertEqual(t["g_dsa_topk_indices"]["shape"], ["B", "seqlen", "Kout"])
        self.assertEqual(ops["g_dsa_ix_k_norm"]["op"], "layernorm")
        self.assertEqual(sorted(ops["g_dsa_ix_k_norm"]["weights"]),
                         ["g_dsa_ix_k_norm_bias", "g_dsa_ix_k_norm_weight"])

    def test_mask_true_visible_and_counts(self):
        t = self.tensors(self.block_dsa)
        ops = self.ops(self.block_dsa)
        self.assertEqual(t["g_dsa_sparse_mask"]["shape"],
                         ["B", "1", "seqlen", "Lkv"])
        self.assertEqual(ops["g_dsa_mask_counts"]["op"], "scatter+add")
        self.assertEqual(ops["g_dsa_sparse_mask"]["op"], "ne+unsqueeze")
        self.assertIn("True=可见", t["g_dsa_sparse_mask"]["note"])
        self.assertIn("g_dsa_eager_mask", ops["g_dsa_mask_add"]["inputs"])
        self.assertEqual(ops["g_dsa_qk"]["op"], "matmul+scale")
        self.assertIn("1/16", ops["g_dsa_qk"]["note"])
        self.assertEqual(t["g_dsa_attn_flat"]["shape"],
                         ["B", "seqlen", "Nh×Dv"])

    def test_moe_288_swiglu_clamp_and_shared(self):
        p = self.params(self.block_dsa)
        self.assertEqual(p["E"]["value"], 288)
        self.assertEqual(p["K"]["value"], 8)
        ops = self.ops(self.block_dsa)
        t = self.tensors(self.block_dsa)
        self.assertEqual(ops["g_moe_sigmoid"]["op"], "sigmoid")
        self.assertEqual(t["g_expert_mask"]["shape"], ["E+1", "K", "Ttok"])
        self.assertEqual(ops["g_moe_expert_mask"]["op"], "one_hot+permute")
        self.assertEqual(ops["g_moe_expert_gather_indices"]["op"], "where")
        self.assertEqual(ops["g_moe_expert_clamp"]["op"], "clamp")
        self.assertIn("Swi", ops["g_moe_expert_clamp"]["parameter_refs"])
        self.assertIn("Swi", ops["g_moe_shared_clamp"]["parameter_refs"])
        self.assertIn("g_routed", ops["g_moe_merge"]["inputs"])
        self.assertIn("g_shared_out", ops["g_moe_merge"]["inputs"])
        self.assertEqual(t["g_moe_experts_gate_up"]["shape"], ["E", "2Es", "H"])

    def test_dsa_has_no_kda_state(self):
        t = self.tensors(self.block_dsa)
        ops = self.ops(self.block_dsa)
        self.assertNotIn("g_kda_state_pre", t)
        self.assertFalse([oid for oid in ops if oid.startswith("g_kda_")])


class TestMhcAndContract(FlashTestBase):
    def test_mhc_two_sites(self):
        for block in (self.block_dense, self.block_dsa):
            t = self.tensors(block)
            ops = self.ops(block)
            for site in ("attn", "ffn"):
                for name in ("in_norm", "fn", "split", "gate", "sinkhorn",
                             "collapse", "expand", "comb_mix", "add"):
                    self.assertIn(f"g_mhc_{site}_{name}", ops, (block["kind"], site, name))
                self.assertEqual(t[f"g_mhc_{site}_comb"]["shape"],
                                 ["B", "seqlen", "Hc", "Hc"])
                self.assertEqual(t[f"g_mhc_{site}_fn_weight"]["shape"],
                                 ["Mhc", "Hc×H"])
                self.assertEqual(t[f"g_mhc_{site}_base"]["shape"], ["Mhc"])
                self.assertEqual(t[f"g_mhc_{site}_mixes"]["shape"],
                                 ["B", "seqlen", "Mhc"])
                self.assertEqual(t[f"g_mhc_{site}_pre"]["shape"],
                                 ["B", "seqlen", "Hc"])
                self.assertEqual(ops[f"g_mhc_{site}_in_norm"]["op"],
                                 "rmsnorm-unweighted")
                self.assertEqual(ops[f"g_mhc_{site}_sinkhorn"]["op"], "sinkhorn")
            self.assertEqual(t["g_hidden_streams_out"]["shape"],
                             ["B", "seqlen", "Hc", "H"])
            self.assertNotIn("g_res_attn", ops)
            self.assertNotIn("g_res_ffn", ops)

    def test_mhc_coefficients_are_24(self):
        for block in (self.block_dense, self.block_dsa):
            params = self.params(block)
            tensors = self.tensors(block)
            self.assertEqual(params["Mhc"]["value"], 24)
            for site in ("attn", "ffn"):
                self.assertEqual(tensors[f"g_mhc_{site}_fn_weight"]["shape"],
                                 ["Mhc", "Hc×H"])
                self.assertEqual(tensors[f"g_mhc_{site}_base"]["shape"], ["Mhc"])
                self.assertEqual(tensors[f"g_mhc_{site}_mixes"]["shape"],
                                 ["B", "seqlen", "Mhc"])
                split = self.ops(block)[f"g_mhc_{site}_split"]
                self.assertEqual(split["outputs"],
                                 [f"g_mhc_{site}_pre_logits",
                                  f"g_mhc_{site}_post_logits",
                                  f"g_mhc_{site}_comb_logits"])

    def test_contract_rejects_mhc_20(self):
        structure = copy.deepcopy(self.structure_dense)
        for param in structure["block"]["parameters"]:
            if param["id"] == "Mhc":
                param["value"] = 20
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_dsa_ape_and_pool_dependency(self):
        ops = self.ops(self.block_dsa)
        tensors = self.tensors(self.block_dsa)
        pool = ops["g_dsa_ix_pool"]
        self.assertIn("g_dsa_ix_ape", pool["weights"])
        self.assertNotIn("g_dsa_ix_visible", pool["inputs"])
        self.assertIn("g_dsa_ix_cache_post", pool["inputs"])
        self.assertEqual(ops["g_dsa_ix_pool_end"]["inputs"],
                         ["g_dsa_ix_pool_indices"])
        self.assertEqual(tensors["g_dsa_ix_pool_end"]["shape"], ["B", "Npool"])
        self.assertEqual(sorted(ops["g_dsa_ix_valid_candidates"]["inputs"]),
                         ["g_dsa_ix_pool_end", "g_dsa_ix_pool_valid",
                          "g_dsa_ix_visible"])
        self.assertEqual(sorted(ops["g_dsa_ix_mask"]["inputs"]),
                         ["g_dsa_ix_index_scores", "g_dsa_ix_valid_candidates"])
        self.assertEqual(sorted(ops["g_dsa_ix_selected_valid"]["inputs"]),
                         ["g_dsa_ix_selected", "g_dsa_ix_valid_candidates"])
        self.assertIn("g_dsa_ix_selected_valid", ops["g_dsa_ix_invalid"]["inputs"])
        self.assertEqual(ops["g_dsa_ix_pad"]["outputs"],
                         ["g_dsa_ix_padded_indices"])
        self.assertEqual(sorted(ops["g_dsa_ix_query_mask"]["inputs"]),
                         ["g_attention_mask", "g_dsa_ix_padded_indices"])
        self.assertEqual(ops["g_dsa_ix_query_mask"]["outputs"],
                         ["g_dsa_topk_indices"])

    def test_j_has_integer_domain_guard_metadata(self):
        params = self.params(self.block_dsa)
        self.assertEqual(params["j"]["domain"],
                         {"kind": "integer", "min": 0, "max_symbol": "seqlen"})
        self.assertIsNone(params["Vq"]["value"])
        self.assertIsNone(params["Ksel"]["value"])
        self.assertIsNone(params["Ksel_last"]["value"])

    def test_contract_rejects_missing_j_domain(self):
        structure = copy.deepcopy(self.structure_dsa)
        for param in structure["block"]["parameters"]:
            if param["id"] == "j":
                param.pop("domain", None)
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_missing_ape(self):
        structure = copy.deepcopy(self.structure_dsa)
        for op in structure["block"]["operators"]:
            if op["id"] == "g_dsa_ix_pool":
                op["weights"] = []
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_missing_valid_candidates(self):
        structure = copy.deepcopy(self.structure_dsa)
        block = structure["block"]
        block["operators"] = [op for op in block["operators"]
                              if op["id"] != "g_dsa_ix_valid_candidates"]
        block["tensors"] = [t for t in block["tensors"]
                            if t["id"] != "g_dsa_ix_valid_candidates"]
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_missing_selected_valid(self):
        structure = copy.deepcopy(self.structure_dsa)
        block = structure["block"]
        block["operators"] = [op for op in block["operators"]
                              if op["id"] != "g_dsa_ix_selected_valid"]
        block["tensors"] = [t for t in block["tensors"]
                            if t["id"] != "g_dsa_ix_selected_valid"]
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_missing_query_mask(self):
        structure = copy.deepcopy(self.structure_dsa)
        structure["block"]["operators"] = [
            op for op in structure["block"]["operators"]
            if op["id"] != "g_dsa_ix_query_mask"]
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_kda_recurrence_algebra_uses_decayed_state(self):
        state = [[1.0, 2.0], [3.0, 4.0]]
        q, k, v, g, beta = [0.5, 0.25], [0.5, 1.0], [1.0, -1.0], [0.2, -0.1], 0.7
        out, new_state = _kda_step_correct(state, q, k, v, g, beta)
        wrong_prev, _ = _kda_step_wrong_undecayed(state, q, k, v, g, beta)
        wrong_dir, _ = _kda_step_wrong_direction(state, q, k, v, g, beta)
        self.assertAlmostEqual(out[0], 0.47550467099126736, places=12)
        self.assertAlmostEqual(out[1], 0.08197682558972685, places=12)
        self.assertAlmostEqual(new_state[0][0], 0.4075779865443826, places=12)
        self.assertNotAlmostEqual(out[0], wrong_prev[0], places=6)
        self.assertNotAlmostEqual(out[0], wrong_dir[0], places=6)

    def test_contract_rejects_wrong_kda_formula(self):
        structure = copy.deepcopy(self.structure_dense)
        for op in structure["block"]["operators"]:
            if op["id"] == "g_kda_scan":
                op["note"] = ("真实语义（delta-rule）：S_t = S_{t-1}*exp(g_t) + "
                              "k_t(v_t - S_{t-1}k_t)^T*beta_t；输出 o_t = S_t^T q_t。"
                              "chunk / recurrent。")
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_full_length_ksel(self):
        structure = copy.deepcopy(self.structure_dsa)
        for item in structure["block"]["derivations"]:
            if item["id"] == "K_sel":
                item["expression"] = "min(L_idx, Ki + (L_idx % Kpool))"
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_layer_contracts_pass(self):
        for structure in (self.structure_dense, self.structure_dsa):
            self.assertEqual(validate_hf_layer_structure(structure), [])
        for doc in (self.doc_dense, self.doc_dsa):
            self.assertEqual(validate_hf_reading_document(doc, self.shas), [])
        self.assertEqual(validate_hf_network_overview(self.network, self.shas), [])

    def test_contract_rejects_removed_scan(self):
        structure = copy.deepcopy(self.structure_dense)
        structure["block"]["operators"] = [
            op for op in structure["block"]["operators"] if op["id"] != "g_kda_scan"]
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_wrong_ksel_formula(self):
        structure = copy.deepcopy(self.structure_dsa)
        for item in structure["block"]["derivations"]:
            if item["id"] == "K_sel":
                item["expression"] = "min(Ki, L_idx)"
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_missing_sinkhorn(self):
        structure = copy.deepcopy(self.structure_dense)
        structure["block"]["operators"] = [
            op for op in structure["block"]["operators"]
            if op["id"] != "g_mhc_attn_sinkhorn"]
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_rope_injection(self):
        structure = copy.deepcopy(self.structure_dsa)
        block = structure["block"]
        block["tensors"].append({
            "id": "g_ix_rope", "name": "rope", "shape": ["B", "seqlen"],
            "parameter_refs": ["B", "seqlen"], "kind": "activation", "note": "注入"})
        block["operators"].append({
            "id": "g_ix_rope", "module_path": "x", "name": "rope", "op": "rope",
            "inputs": ["g_dsa_ix_q"], "outputs": ["g_ix_rope"], "weights": [],
            "parameter_refs": ["B"], "source_refs": [
                {"path": "x", "locator": "1", "kind": "source", "note": "x"}],
            "analysis_status": "analyzed", "note": "注入"})
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_expanded_cache(self):
        structure = copy.deepcopy(self.structure_dsa)
        for tensor in structure["block"]["tensors"]:
            if tensor["id"] == "g_cache_kv_pre":
                tensor["shape"] = ["B", "Nh", "kvcachelen", "Rkv"]
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_contract_rejects_missing_clamp(self):
        structure = copy.deepcopy(self.structure_dsa)
        structure["block"]["operators"] = [
            op for op in structure["block"]["operators"]
            if op["id"] != "g_moe_expert_clamp"]
        self.assertTrue(validate_hf_layer_structure(structure))

    def test_network_contract_rejects_wrong_layer_count(self):
        network = copy.deepcopy(self.network)
        network["scope"]["layer_count"] = 78
        self.assertTrue(validate_hf_network_overview(network, self.shas))

    def test_network_contract_rejects_rope(self):
        network = copy.deepcopy(self.network)
        network["path"]["operators"].append({
            "id": "net_rope", "module_path": "x", "name": "rope", "op": "cos/sin",
            "inputs": ["inputs_embeds"], "outputs": ["position_ids"], "weights": [],
            "parameter_refs": [], "source_refs": [
                {"path": "x", "locator": "1", "kind": "source", "note": "x"}],
            "analysis_status": "analyzed", "note": "注入"})
        self.assertTrue(validate_hf_network_overview(network, self.shas))


class TestScenarios(FlashTestBase):
    def test_scenario_ids(self):
        self.assertEqual([s["id"] for s in GLM53FLASH_SCENARIOS],
                         ["prefill-empty", "prefill-symbolic", "decode-nonzero",
                          "decode-long", "decode-multi"])

    def test_kda_scenarios(self):
        evaluated = self.structure_dense["scenario_derivations"]
        for sid, srec, sconv in (("prefill-empty", 1048576, 98304),
                                 ("decode-nonzero", 1048576, 98304),
                                 ("decode-long", 1048576, 98304),
                                 ("decode-multi", 1048576, 98304)):
            rows = {r["id"]: r for r in evaluated[sid]}
            self.assertEqual(rows["S_rec"]["value"], srec, sid)
            self.assertEqual(rows["S_conv"]["value"], sconv, sid)
        symbolic = {r["id"]: r for r in evaluated["prefill-symbolic"]}
        self.assertEqual(symbolic["T_tok"]["state"], "unknown")

    def test_dsa_scenarios(self):
        evaluated = self.structure_dsa["scenario_derivations"]
        expected = {
            "prefill-empty": {"L_kv": 8, "L_idx": 8, "N_pool": 2, "SelP": 2,
                              "K_sel_last": 8, "K_out": 2051},
            "decode-nonzero": {"L_kv": 4097, "L_idx": 4097, "N_pool": 1024,
                               "SelP": 512, "K_sel_last": 2049, "K_out": 2051},
            "decode-long": {"L_kv": 131073, "L_idx": 131073, "N_pool": 32768,
                            "SelP": 512, "K_sel_last": 2049, "K_out": 2051},
            "decode-multi": {"L_kv": 4096, "L_idx": 4096, "N_pool": 1024,
                             "SelP": 512, "K_sel_last": 2048, "K_out": 2051},
        }
        for sid, values in expected.items():
            rows = {r["id"]: r for r in evaluated[sid]}
            for did, value in values.items():
                self.assertEqual(rows[did]["state"], "determined", (sid, did))
                self.assertEqual(rows[did]["value"], value, (sid, did))
            for did in ("V_q", "K_sel"):
                self.assertEqual(rows[did]["state"], "unknown", (sid, did))
                self.assertTrue(rows[did].get("unknown_reason"))

    def test_ksel_is_per_query_not_full_length(self):
        # 多 token 场景：S=4、P=4092，逐 query Ksel=2049/2050/2051/2048，
        # 而 Ksel_last 只有最后 query 的 2048；全长统一值会与 j=0..2 不符。
        values = []
        for j in range(4):
            v = 4092 + j + 1
            values.append(min(v, 2048 + v % 4))
        self.assertEqual(values, [2049, 2050, 2051, 2048])
        rows = {r["id"]: r for r in
                self.structure_dsa["scenario_derivations"]["decode-multi"]}
        self.assertEqual(rows["K_sel_last"]["value"], 2048)
        self.assertEqual(rows["K_sel"]["state"], "unknown")
        derivations = {d["id"]: d["expression"]
                       for d in self.structure_dsa["block"]["derivations"]}
        self.assertEqual(derivations["K_sel"], "min(V_q, Ki + (V_q % Kpool))")
        self.assertEqual(derivations["K_sel_last"],
                         "min(L_idx, Ki + (L_idx % Kpool))")

    def test_symbolic_unknown(self):
        rows = {r["id"]: r for r in self.structure_dsa["scenario_derivations"]["prefill-symbolic"]}
        for did in ("L_kv", "L_idx", "N_pool", "V_q", "K_sel", "K_sel_last"):
            self.assertEqual(rows[did]["state"], "unknown", did)
            self.assertTrue(rows[did].get("unknown_reason"))


class TestArtifacts(FlashTestBase):
    def test_analysis_id_and_files(self):
        self.assertEqual(len(self.result.analysis_id), 64)
        self.assertEqual(set(self.documents), {
            "README.md", "data/effective_config.json", "data/layer_structures.json",
            "data/network_overview.json", "reports/source_manifest.json",
            "run_manifest.json", "tables/decoder_variants.xlsx", "viewer/structure.html"})

    def test_effective_config(self):
        config = json.loads(self.documents["data/effective_config.json"])
        values = config["values"]
        self.assertEqual(values["num_hidden_layers"], 45)
        self.assertEqual(values["hc_mult"], 4)
        self.assertEqual(values["index_kpool"], 4)
        self.assertEqual(values["qk_rope_head_dim"], 0)
        self.assertEqual(values["swiglu_limit"], 10.0)
        self.assertTrue(values["no_rope"])
        self.assertEqual(values["topk_method"], "noaux_tc")
        self.assertEqual(config["layer_types"], self.values.layer_types)
        self.assertEqual(len(config["parameter_records"]), 44)

    def test_html_structure_and_anchors(self):
        html = self.documents["viewer/structure.html"].decode("utf-8")
        self.assertEqual(html.count('data-lv="network"'), 1)
        for gid in GROUP_IDS:
            self.assertIn(f'id="lv-group-{gid}"', html)
            self.assertIn(f'id="lv-block-{gid}"', html)
            self.assertIn(f'id="lv-ops-{gid}"', html)
        self.assertIn("KDA", html)
        self.assertIn("DSA", html)
        self.assertIn("mHC", html)
        self.assertIn("NoPE", html)
        self.assertNotIn("四类层组", html)
        self.assertNotIn("DeepseekV4", html)
        self.assertNotIn("shared DSA indexer", html)
        self.assertNotIn("glm_moe_dsa", html)
        ids = re.findall(r'\bid="([^"]+)"', html)
        self.assertEqual(len(ids), len(set(ids)), "页面 id 必须唯一")
        id_set = set(ids)
        hrefs = re.findall(r'href="#([^"]+)"', html)
        missing = sorted({h for h in hrefs if h not in id_set and h != "/network"})
        self.assertEqual(missing, [], f"普通锚点必须有目标: {missing[:5]}")
        for scenario in GLM53FLASH_SCENARIOS:
            self.assertIn(f'"id": "{scenario["id"]}"', html)

    def test_workbook_structure(self):
        wb = load_workbook(io.BytesIO(self.workbook_bytes))
        self.assertEqual(wb.sheetnames, ["Prefill", "Decode"])
        for ws in wb.worksheets:
            text = "\n".join(str(c.value) for row in ws.iter_rows() for c in row
                             if c.value is not None)
            for gid in GROUP_IDS:
                self.assertIn(gid, text)
            self.assertIn("CUBE", text)
            self.assertIn("kvcachelen", text)
            self.assertIn("未知", text)
            # 成本小计/串行总计用 COUNT+SUM 守卫；不允许忽略文本的裸 SUM。
            for row in ws.iter_rows():
                for cell in row:
                    # 串行区的“已知部分小计”在贡献列（H）且标注“不是整网总计”；
                    # 成本/工作量列（16/25）不允许出现忽略文本的裸 SUM。
                    if isinstance(cell.value, str) and cell.column in (16, 25):
                        self.assertFalse(cell.value.upper().startswith("=SUM("),
                                         (ws.title, cell.coordinate, cell.value))
            self.assertNotIn("full-dense", text)
            self.assertNotIn("shared-moe", text)
            self.assertNotIn("full-moe", text)
        # 可填输入与派生公式
        ws = wb["Prefill"]
        cells = {c.coordinate: c.value for row in ws.iter_rows() for c in row}
        values = [v for v in cells.values() if isinstance(v, str)]
        self.assertTrue(any("=IF(" in v for v in values), "应有带守卫的活公式")
        self.assertTrue(any("MOD(" in v for v in values), "K_sel 的 % 应编译为 MOD")
        self.assertTrue(any("K_sel" in v or "index_output" in v or "Kout" in v
                            for v in values))

    def test_workbook_spec_rejects_wrong_attribution(self):
        container = copy.deepcopy(self.container)
        container["variants"][0]["attention_type"] = "indexed_attention"
        container["variants"][0]["indexer_type"] = "full"
        with self.assertRaises(ValueError):
            build_structures_workbook_spec(container, self.network)

    def test_single_and_onnx_rejected(self):
        with self.assertRaises(HfBuildError):
            run_hf_build(REPO_ROOT, REPO_ROOT / "structure_reader",
                         model_key=MODEL_KEY, all_variants=False)
        with self.assertRaises(HfBuildError):
            run_hf_build(REPO_ROOT, REPO_ROOT / "structure_reader",
                         model_key=MODEL_KEY, all_variants=True, onnx_modules=True)

    def test_cross_model_flash_deepseek_glm_flash_no_pollution(self):
        flash_first = self.documents
        _, deep = run_hf_build(REPO_ROOT, REPO_ROOT / "structure_reader",
                               model_key="deepseek_v4_flash_0731_hf",
                               all_variants=True)
        deep_html = deep["viewer/structure.html"].decode("utf-8")
        self.assertIn("sum_j comb[j,k]·residual[j,d]", deep_html)
        self.assertNotIn("query_index", deep_html)
        self.assertIn("concat + pad", deep_html)
        self.assertNotIn("concat + Pad", deep_html)
        _, glm = run_hf_build(REPO_ROOT, REPO_ROOT / "structure_reader",
                              model_key="glm_5_3_hf", all_variants=True)
        self.assertNotIn("query_index", glm["viewer/structure.html"].decode("utf-8"))
        _, flash_again = run_hf_build(REPO_ROOT, REPO_ROOT / "structure_reader",
                                      model_key=MODEL_KEY, all_variants=True)
        self.assertEqual(flash_again, flash_first)

    def test_build_deterministic(self):
        result2, docs2 = run_hf_build(
            REPO_ROOT, REPO_ROOT / "structure_reader",
            model_key=MODEL_KEY, all_variants=True)
        self.assertEqual(result2.analysis_id, self.result.analysis_id)
        self.assertEqual(docs2, self.documents)

    def test_profile_module_coverage(self):
        for block in (self.block_dense, self.block_dsa):
            groups = GLM53FLASH_PAGE_PROFILE.module_groups(block)
            covered = [op["id"] for g in groups for op in g["ops"]]
            self.assertEqual(sorted(covered),
                             sorted(op["id"] for op in block["operators"]))
            self.assertNotIn("other", {g["id"] for g in groups})
            graph = GLM53FLASH_PAGE_PROFILE.overview_graph(block)
            self.assertTrue(graph["frames"])
            self.assertTrue(all(n["id"] in graph["positions"] for n in graph["nodes"]))


if __name__ == "__main__":
    unittest.main()
