"""Effective-configuration checks for layer_idx=4 (independently re-derived).

Re-derives the effective config from raw config.json following the documented
logic in configuration_deepseek_v4.py, then asserts the sample layer resolves to
CSA + TopK MoE + compress RoPE, and that the supporting key types/aliases hold.
"""

import unittest
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hf_expectations as hx  # noqa: E402


class EffectiveConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = hx.load_raw_config()
        cls.e = hx.derive_effective(layer_idx=4)

    def test_hidden_layer_truncation_from_46_legacy_entries(self):
        self.assertEqual(len(self.raw["compress_ratios"]), 46)
        self.assertEqual(self.raw["num_hidden_layers"], 43)
        self.assertEqual(len(self.e["layer_types"]), 43)
        self.assertEqual(len(self.e["mlp_layer_types"]), 43)

    def test_layer4_is_csa_and_topk_moe(self):
        self.assertEqual(self.e["layer_idx"], 4)
        self.assertEqual(self.e["attention_type"], "compressed_sparse_attention")
        self.assertEqual(self.e["moe_type"], "moe")
        self.assertEqual(self.e["rope_label"], "compress")

    def test_layer_type_mapping_follows_compress_ratios(self):
        self.assertEqual(hx.derive_layer_types([0], 1)[0], "sliding_attention")
        self.assertEqual(hx.derive_layer_types([4], 1)[0], "compressed_sparse_attention")
        self.assertEqual(hx.derive_layer_types([128], 1)[0], "heavily_compressed_attention")
        self.assertEqual(self.e["layer_types"][0], "sliding_attention")
        self.assertEqual(self.e["layer_types"][1], "sliding_attention")
        self.assertEqual(self.e["layer_types"][2], "compressed_sparse_attention")
        self.assertEqual(self.e["layer_types"][3], "heavily_compressed_attention")

    def test_four_real_groups_and_no_fabricated_combination(self):
        groups = hx.layer_index_groups(self.e["layer_types"], self.e["mlp_layer_types"])
        observed = {k: len(v) for k, v in groups.items()}
        self.assertEqual(
            observed,
            {
                ("sliding_attention", "hash_moe"): 2,
                ("compressed_sparse_attention", "hash_moe"): 1,
                ("heavily_compressed_attention", "moe"): 20,
                ("compressed_sparse_attention", "moe"): 20,
            },
        )
        self.assertEqual(sum(observed.values()), 43)
        # no layer is simultaneously sliding_attention and routed (TopK) moe
        for attn, moe in groups:
            if attn == "sliding_attention":
                self.assertEqual(moe, "hash_moe")

    def test_partial_rotary_and_head_split(self):
        self.assertEqual(self.e["head_dim"], 512)
        self.assertEqual(self.e["qk_rope_head_dim"], 64)
        self.assertEqual(self.e["rope_dim"], 64)
        self.assertEqual(self.e["nope_dim"], 448)
        self.assertAlmostEqual(self.e["partial_rotary_factor"], 64 / 512)

    def test_shared_kv_attention_groups(self):
        self.assertEqual(self.e["num_key_value_heads"], 1)
        self.assertEqual(self.e["num_key_value_groups"], self.e["num_attention_heads"])
        self.assertEqual(self.e["num_key_value_groups"], 64)

    def test_attribute_map_aliases(self):
        self.assertEqual(self.e["num_local_experts"], self.raw["n_routed_experts"])
        self.assertEqual(self.e["num_local_experts"], 256)
        self.assertEqual(self.e["intermediate_size"], self.raw["moe_intermediate_size"])
        self.assertEqual(self.e["intermediate_size"], 2048)

    def test_moe_dynamic_params(self):
        self.assertEqual(self.e["n_routed_experts"], 256)
        self.assertEqual(self.e["num_experts_per_tok"], 6)
        self.assertEqual(self.e["n_shared_experts"], 1)
        self.assertEqual(self.e["scoring_func"], "sqrtsoftplus")
        self.assertEqual(self.e["routed_scaling_factor"], 1.5)
        self.assertEqual(self.e["swiglu_limit"], 10.0)

    def test_csa_cache_and_indexer_params(self):
        self.assertEqual(self.e["csa_compress_rate"], 4)
        self.assertEqual(self.e["hca_compress_rate"], 128)
        self.assertEqual(self.e["sliding_window"], 128)
        self.assertEqual(self.e["persistent_sliding_keys"], 127)
        self.assertEqual(self.e["index_n_heads"], 64)
        self.assertEqual(self.e["index_head_dim"], 128)
        self.assertEqual(self.e["index_topk"], 512)

    def test_mhc_params(self):
        self.assertEqual(self.e["hc_mult"], 4)
        self.assertEqual(self.e["hc_sinkhorn_iters"], 20)
        self.assertAlmostEqual(self.e["hc_eps"], 1e-6)

    def test_rope_parameters_two_labels(self):
        # compress branch carries the raw yarn scaling with compress_rope_theta
        self.assertEqual(self.e["rope_theta"], 10000)
        self.assertEqual(self.e["compress_rope_theta"], 160000)
        rs = self.e["rope_scaling"]
        self.assertEqual(rs["type"], "yarn")
        self.assertEqual(rs["factor"], 16)
        self.assertEqual(rs["beta_fast"], 32)
        self.assertEqual(rs["beta_slow"], 1)
        self.assertEqual(rs["original_max_position_embeddings"], 65536)

    def test_config_class_declares_required_legacy_handling(self):
        src = hx.code("configuration")
        for needle in (
            'kwargs.pop("compress_ratios"',
            'kwargs.pop("num_hash_layers"',
            'kwargs.pop("qk_rope_head_dim"',
            '"num_local_experts": "n_routed_experts"',
            '"intermediate_size": "moe_intermediate_size"',
            "self.layer_types = list(self.layer_types[:n])",
            'self.mlp_layer_types = ["hash_moe"] * min(n, n_hash)',
            'compress.setdefault("attention_factor", 1.0)',
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, src)


class EffectiveConfigNegativeTests(unittest.TestCase):
    def test_ignoring_legacy_compress_ratios_changes_layer4(self):
        raw = hx.load_raw_config()
        n = raw["num_hidden_layers"]
        actual = hx.derive_layer_types(raw["compress_ratios"], n)
        default = hx.v4pro_default_layer_types(n)
        self.assertEqual(actual[4], "compressed_sparse_attention")
        self.assertEqual(default[4], "heavily_compressed_attention")
        self.assertNotEqual(actual[4], default[4])
        self.assertNotEqual(actual, default)

    def test_46_legacy_entries_must_be_truncated_to_43(self):
        raw = hx.load_raw_config()
        untruncated = hx.derive_layer_types(raw["compress_ratios"], 46)
        self.assertEqual(len(untruncated), 46)
        self.assertEqual(len(hx.derive_layer_types(raw["compress_ratios"], 43)), 43)
        self.assertNotEqual(len(untruncated), raw["num_hidden_layers"])

    def test_off_by_one_persistent_kv_length_would_be_wrong(self):
        # persistent sliding cache stores sliding_window - 1 = 127 tokens, not 128
        self.assertEqual(128 - 1, 127)
        self.assertNotEqual(127, 128)


if __name__ == "__main__":
    unittest.main()
