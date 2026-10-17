"""CSA / HCA KVCache structure checks for layer_idx=4.

Separates the cache tensor read by attention this call ("full", returned by
``update``) from the persistently stored sliding-window tail
(``sliding_window - 1`` tokens), and covers the CSA compression buffer, the
overlap (Ca) state, the indexer top-k sentinel and the per-query ``block_bias``.
"""

import unittest
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hf_expectations as hx  # noqa: E402


class CacheClassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modeling = hx.code("modeling")
        cls.cache = hx.code("cache_utils")

    def test_csa_and_hca_cache_layer_types(self):
        self.assertIn("class DeepseekV4HCACache(DynamicSlidingWindowLayer):", self.modeling)
        self.assertIn('_layer_type = "heavily_compressed_attention"', self.modeling)
        self.assertIn("class DeepseekV4CSACache(DeepseekV4HCACache):", self.modeling)
        self.assertIn('_layer_type = "compressed_sparse_attention"', self.modeling)

    def test_cache_layer_registration_mechanism(self):
        self.assertIn("DYNAMIC_LAYER_TYPE_MAPPING[cls._layer_type] = cls", self.cache)
        # CSA/HCA cache kwargs need the config (for compress rates)
        self.assertIn('elif layer_type in ("heavily_compressed_attention", "compressed_sparse_attention"):', self.cache)
        self.assertIn('layer_kwargs["config"] = layer_config', self.cache)

    def test_compress_rates_follow_config(self):
        self.assertIn('self.compress_rate = config.compress_rates["heavily_compressed_attention"]', self.modeling)
        self.assertIn('self.compress_rate = config.compress_rates["compressed_sparse_attention"]', self.modeling)

    def test_hca_has_no_indexer_or_overlap(self):
        self.assertIn('self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}', self.modeling)
        self.assertNotIn("overlap_kv", self.modeling.split("class DeepseekV4CSACache")[0].split("class DeepseekV4HCACache")[1])

    def test_csa_adds_indexer_and_overlap_entries(self):
        csa = self.modeling.split("class DeepseekV4CSACache", 1)[1]
        for needle in (
            'self.buffer_kv["indexer"] = None',
            'self.buffer_gate["indexer"] = None',
            'self.compressed_kv["indexer"] = None',
            'self.entry_count["indexer"] = 0',
            'self.overlap_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}',
            'self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}',
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, csa)


class CacheUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modeling = hx.code("modeling")

    def test_update_returns_full_but_persists_window_minus_one(self):
        self.assertIn("self.cumulative_length += key_states.shape[-2]", self.modeling)
        self.assertIn("full = torch.cat([self.keys, key_states], dim=-2)", self.modeling)
        self.assertIn("self.keys = full[:, :, -self.sliding_window + 1 :, :]", self.modeling)
        self.assertIn("self.values = self.keys", self.modeling)
        self.assertIn("return full, full", self.modeling)

    def test_persistent_length_is_127_not_128(self):
        e = hx.derive_effective(4)
        self.assertEqual(e["sliding_window"], 128)
        self.assertEqual(e["persistent_sliding_keys"], e["sliding_window"] - 1)
        self.assertEqual(e["persistent_sliding_keys"], 127)

    def test_store_compression_weights_peels_window_aligned_prefix(self):
        for needle in (
            "first_window_position = self.entry_count[name] * self.compress_rate",
            "usable = (kv.shape[1] // self.compress_rate) * self.compress_rate",
            "self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]",
            "return kv[:, :usable], gate[:, :usable], first_window_position",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.modeling)

    def test_update_compressor_states_counts_entries(self):
        for needle in (
            "self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)",
            "self.entry_count[name] += compressed.shape[1]",
            "return self.compressed_kv[name]",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.modeling)

    def test_overlap_state_stores_only_ca_slice_of_last_window(self):
        for needle in (
            'self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()',
            'self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()',
            "return prior_kv, prior_gate",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.modeling)

    def test_single_token_none_block_bias_is_hca_only_not_csa(self):
        # HCA compressor short-circuits single-token / empty-compression calls with (kv, None)
        hca = self.modeling.split("class DeepseekV4HCACompressor", 1)[1].split("class DeepseekV4IndexerScorer", 1)[0]
        self.assertIn("if seq_len == 1 or compressed_len == 0:", hca)
        self.assertIn("return compressed_kv, None", hca)
        # CSA compressor has NO such early return: even for a single decode token it
        # builds a real block_bias from the indexer top-k.
        csa = self.modeling.split("class DeepseekV4CSACompressor", 1)[1].split("class repeat_kv", 1)[0]
        self.assertNotIn("seq_len == 1 or compressed_len == 0", csa)
        self.assertIn("block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)", csa)
        self.assertIn("return compressed_kv, block_bias[..., :compressed_len]", csa)

    def test_persistent_length_is_min_cumulative_and_cap(self):
        # persistent keys = last W-1 positions of the running (prior_stored + new) tensor,
        # i.e. min(cumulative_seen, W-1); the cap is W-1=127, not the read-view length.
        self.assertIn("self.keys = full[:, :, -self.sliding_window + 1 :, :]", self.modeling)
        self.assertEqual(min(7, 128 - 1), 7)
        self.assertEqual(min(500, 128 - 1), 127)


class CompressionAndMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modeling = hx.code("modeling")
        cls.masking = hx.code("masking_utils")

    def test_csa_two_series_layout_width_2m(self):
        for needle in (
            "new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))",
            "new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]",
            "new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.modeling)

    def test_hca_single_series_layout(self):
        self.assertIn("chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)", self.modeling)
        self.assertIn(
            "compressed = self.kv_norm( (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32)"
            ".to(chunk_kv.dtype)).sum(dim=2) )",
            self.modeling,
        )

    def test_hca_causal_block_bias_rule(self):
        self.assertIn("causal_threshold = (position_ids + 1) // self.compress_rate", self.modeling)
        self.assertIn("entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1)", self.modeling)

    def test_indexer_topk_sentinel_and_threshold(self):
        self.assertIn("top_k = min(self.index_topk, compressed_len)", self.modeling)
        self.assertIn("top_k_indices = index_scores.topk(top_k, dim=-1).indices", self.modeling)
        self.assertIn("invalid = top_k_indices >= causal_threshold.unsqueeze(-1)", self.modeling)
        self.assertIn("return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)", self.modeling)

    def test_csa_block_bias_scatter_from_indexer(self):
        for needle in (
            "valid = top_k_indices >= 0",
            "safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))",
            "block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)",
            "return compressed_kv, block_bias[..., :compressed_len]",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.modeling)

    def test_block_bias_is_concatenated_into_mask(self):
        for needle in (
            "if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:",
            "attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.modeling)

    def test_all_layer_types_share_sliding_window_mask(self):
        self.assertIn('"sliding_attention": create_sliding_window_causal_mask', self.masking)
        self.assertIn('"compressed_sparse_attention": create_sliding_window_causal_mask', self.masking)
        self.assertIn('"heavily_compressed_attention": create_sliding_window_causal_mask', self.masking)
        self.assertIn("return kv_idx > q_idx - sliding_window", self.masking)

    def test_attention_uses_mask_sizes_from_cache(self):
        self.assertIn('if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:', self.masking)
        self.assertIn("kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)", self.masking)


if __name__ == "__main__":
    unittest.main()
