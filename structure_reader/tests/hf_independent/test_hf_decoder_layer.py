"""Decoder-layer assembly and analysis-boundary checks for layer_idx=4.

Verifies the two mHC sites wrap attention and MoE respectively, that the eager
path is the selected one, and that the analysis endpoint is logits (no sampling
or generation loop).
"""

import unittest
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hf_expectations as hx  # noqa: E402


class DecoderLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = hx.code("modeling")

    def test_decoder_layer_submodules(self):
        for needle in (
            "self.self_attn = DeepseekV4Attention(config, layer_idx)",
            "self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)",
            "self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)",
            "self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)",
            "self.attn_hc = DeepseekV4HyperConnection(config)",
            "self.ffn_hc = DeepseekV4HyperConnection(config)",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.src)

    def test_two_mhc_sites_order(self):
        order = [
            "post, comb, collapsed = self.attn_hc(hidden_states)",
            "attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)",
            "post, comb, collapsed = self.ffn_hc(hidden_states)",
            "mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)",
        ]
        positions = [self.src.index(n) for n in order]
        self.assertEqual(positions, sorted(positions))

    def test_hidden_streams_dtype_restored_after_mhc(self):
        # mHC weights come out in fp32; both mixing sites cast back to the input dtype
        self.assertIn("dtype = hidden_states.dtype", self.src)
        self.assertIn("post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2)", self.src)
        self.assertIn("post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2)", self.src)

    def test_moe_receives_input_ids_for_hash_layers(self):
        # input_ids must be threaded through even though layer 4 is not a hash layer
        self.assertIn("mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)", self.src)

    def test_eager_path_is_selected(self):
        for needle in (
            "_supports_flash_attn = False",
            "_supports_sdpa = False",
            "_supports_flex_attn = False",
            "_can_compile_fullgraph = False",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.src)
        self.assertIn('ALL_ATTENTION_FUNCTIONS.get_interface( self.config._attn_implementation, eager_attention_forward )', self.src)

    def test_analysis_endpoint_is_logits(self):
        self.assertIn("logits = self.lm_head(hidden_states[:, slice_indices, :])", self.src)
        self.assertIn("hidden_states = self.norm(self.hc_head(hidden_states))", self.src)
        # generation/sampling is out of analysis scope
        self.assertNotIn("def generate(", self.src)

    def test_model_builds_43_layers_and_two_rope_labels(self):
        e = hx.derive_effective(4)
        self.assertEqual(e["num_hidden_layers"], 43)
        self.assertIn(
            "[DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]", self.src
        )
        self.assertIn('"main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main")', self.src)
        self.assertIn('"compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress")', self.src)


class BoundaryNegativeTests(unittest.TestCase):
    def test_layer4_kv_cache_persists_only_window_minus_one(self):
        e = hx.derive_effective(4)
        # read view may reach the sliding window; persistent storage is one shorter
        self.assertNotEqual(e["persistent_sliding_keys"], e["sliding_window"])

    def test_attention_mask_extension_does_not_zero_pad_compressed_slots(self):
        src = hx.code("modeling")
        # when a compressor contributes block_bias it must be used, not zero padding
        self.assertIn("attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)", src)
        self.assertIn("attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)", src)
        cat_pos = src.index("attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)")
        pad_pos = src.index("attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)")
        self.assertLess(cat_pos, pad_pos)


if __name__ == "__main__":
    unittest.main()
