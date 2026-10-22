"""CSA Attention structure checks for layer_idx=4.

Covers: low-rank Q projection, shared-KV MQA (single KV head broadcast to all
query heads), partial/interleaved RoPE on the trailing 64 channels, eager
QK->scale/mask/sink->softmax->AV, inverse RoPE on the attention output, and the
grouped low-rank output projection.
"""

import unittest
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hf_expectations as hx  # noqa: E402


class AttentionShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = hx.derive_effective(4)
        cls.shapes = hx.expected_attention_shapes(cls.e)

    def test_projection_weight_shapes(self):
        s = self.shapes
        self.assertEqual(s["q_a_proj"], [1024, 4096])
        self.assertEqual(s["q_a_norm"], [1024])
        self.assertEqual(s["q_b_proj"], [32768, 1024])
        self.assertEqual(s["kv_proj"], [512, 4096])
        self.assertEqual(s["kv_norm"], [512])
        self.assertEqual(s["sinks"], [64])

    def test_grouped_output_projection_shapes(self):
        s = self.shapes
        self.assertEqual(self.e["o_groups"], 8)
        self.assertEqual(self.e["o_lora_rank"], 1024)
        self.assertEqual(s["o_a_proj"], [8192, 4096])
        self.assertEqual(s["o_b_proj"], [4096, 8192])

    def test_single_kv_head_broadcast(self):
        self.assertEqual(self.e["num_key_value_heads"], 1)
        self.assertEqual(self.shapes["kv_heads_axis"], ["B", "S", 1, 512])
        self.assertEqual(self.shapes["q_heads_axis"], ["B", "S", 64, 512])

    def test_constructor_shapes_match_source_arguments(self):
        src = hx.code("modeling")
        for needle in (
            "self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)",
            "self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)",
            "self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)",
            "self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)",
            "self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)",
            "self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)",
            "self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)",
            "self.sinks = nn.Parameter(torch.empty(self.num_heads))",
            "self.num_key_value_groups = config.num_attention_heads",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, src)

    def test_grouped_linear_constructor(self):
        src = hx.code("modeling")
        self.assertIn(
            "self.o_a_proj = DeepseekV4GroupedLinear( self.num_heads * self.head_dim // config.o_groups, "
            "config.o_groups * config.o_lora_rank, config.o_groups )",
            src,
        )

    def test_grouped_linear_forward_is_block_diagonal(self):
        src = hx.code("modeling")
        self.assertIn("w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)", src)
        self.assertIn("x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)", src)
        self.assertIn("y = torch.bmm(x, w).transpose(0, 1)", src)
        self.assertIn("return y.reshape(*input_shape, self.n_groups, -1)", src)


class PartialRopeTests(unittest.TestCase):
    def test_rotary_embedding_partial_dim(self):
        src = hx.code("modeling")
        self.assertIn('partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)', src)
        self.assertIn("dim = int(head_dim * partial_rotary_factor)", src)
        self.assertIn("inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))", src)

    def test_interleaved_rope_uses_repeat_interleave(self):
        src = hx.code("modeling")
        self.assertIn("cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)", src)
        self.assertIn("sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)", src)
        self.assertIn("x1 = x[..., 0::2]", src)
        self.assertIn("x2 = x[..., 1::2]", src)
        self.assertIn("return torch.stack((-x2, x1), dim=-1).flatten(-2)", src)

    def test_rope_applied_to_trailing_slice_only(self):
        src = hx.code("modeling")
        self.assertIn("nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]", src)
        self.assertIn("return torch.cat([nope, rotated], dim=-1)", src)
        self.assertIn("rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)", src)

    def test_rope_label_selection_per_layer(self):
        src = hx.code("modeling")
        self.assertIn(
            'self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"', src
        )
        self.assertIn('cos, sin = position_embeddings[self.rope_layer_type]', src)

    def test_compress_rope_is_yarn_with_forced_attention_factor(self):
        cfg = hx.code("configuration")
        self.assertIn('if compress["rope_type"] == "yarn":', cfg)
        self.assertIn('compress.setdefault("attention_factor", 1.0)', cfg)


class AttentionForwardChainTests(unittest.TestCase):
    def test_forward_operator_order(self):
        src = hx.code("modeling")
        order = [
            "q_residual = self.q_a_norm(self.q_a_proj(hidden_states))",
            "q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)",
            "q = self.q_b_norm(q)",
            "q = apply_rotary_pos_emb(q, cos, sin)",
            "kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)",
            "kv = apply_rotary_pos_emb(kv, cos, sin)",
            "kv = past_key_values.update(kv, kv, self.layer_idx)[0]",
            "kv = torch.cat([kv, compressed_kv], dim=2)",
            "attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)",
            "grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)",
            "grouped = self.o_a_proj(grouped).flatten(2)",
            "output = self.o_b_proj(grouped)",
        ]
        positions = [src.index(needle) for needle in order]
        self.assertEqual(positions, sorted(positions), "attention operators are out of source order")

    def test_shared_kv_update_passes_same_tensor(self):
        src = hx.code("modeling")
        self.assertIn("if past_key_values is not None:", src)
        self.assertIn("kv = past_key_values.update(kv, kv, self.layer_idx)[0]", src)

    def test_inverse_rope_on_output_uses_negative_sin(self):
        src = hx.code("modeling")
        self.assertIn(
            "attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)", src
        )

    def test_eager_attention_operator_order(self):
        src = hx.code("modeling")
        order = [
            "key_states = repeat_kv(key, module.num_key_value_groups)",
            "value_states = repeat_kv(value, module.num_key_value_groups)",
            "attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling",
            "attn_weights = attn_weights + attention_mask",
            "sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)",
            "combined_logits = torch.cat([attn_weights, sinks], dim=-1)",
            "combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values",
            "probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)",
            "scores = probs[..., :-1]",
            "attn_output = torch.matmul(attn_weights, value_states)",
        ]
        positions = [src.index(needle) for needle in order]
        self.assertEqual(positions, sorted(positions), "eager attention operators are out of source order")

    def test_repeat_kv_expands_single_head(self):
        src = hx.code("modeling")
        self.assertIn(
            "hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)",
            src,
        )
        self.assertIn("return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)", src)

    def test_compressor_selected_by_layer_type(self):
        src = hx.code("modeling")
        self.assertIn(
            "self.compressor = ( COMPRESSOR_CLASSES[self.layer_type](config) "
            "if self.layer_type != \"sliding_attention\" else None )",
            src,
        )
        self.assertIn('"compressed_sparse_attention": DeepseekV4CSACompressor', src)
        self.assertIn('"heavily_compressed_attention": DeepseekV4HCACompressor', src)


class AttentionNegativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = hx.code("modeling")

    def test_no_separate_value_projection(self):
        self.assertNotIn("self.v_proj", self.src)
        self.assertNotIn("self.v_b_proj", self.src)

    def test_kv_projection_is_single_head_not_double(self):
        attention = self.src.split("class DeepseekV4Attention", 1)[1].split("class DeepseekV4HyperConnection", 1)[0]
        self.assertIn("self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)", attention)
        self.assertNotIn("2 * self.head_dim", attention)

    def test_q_b_proj_is_not_full_rank(self):
        # low-rank Q: q_lora_rank (1024) is the input dim, not hidden_size (4096)
        self.assertNotIn("self.q_b_proj = nn.Linear(config.hidden_size, self.num_heads", self.src)

    def test_rope_is_not_applied_to_leading_slice(self):
        self.assertIn("x[..., -rope_dim:]", self.src)
        self.assertNotIn("nope, rope = x[..., rope_dim:], x[..., :rope_dim]", self.src)

    def test_output_projection_is_grouped_not_plain(self):
        self.assertNotIn(
            "self.o_a_proj = nn.Linear(self.num_heads * self.head_dim, config.o_groups * config.o_lora_rank",
            self.src,
        )


if __name__ == "__main__":
    unittest.main()
