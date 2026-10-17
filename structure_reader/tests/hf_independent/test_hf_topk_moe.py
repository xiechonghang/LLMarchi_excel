"""TopK MoE structure checks for layer_idx=4 (mlp_layer_types[4] == "moe").

Covers routing scores (sqrt(softplus)), top-k selection with a correction bias,
weight normalisation + routed scaling, merged gate_up/down expert weights,
clamp + SiLU + mul, weighted accumulation, the shared expert, and the dynamic
token dimension.
"""

import unittest
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hf_expectations as hx  # noqa: E402


class MoeShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = hx.derive_effective(4)
        cls.shapes = hx.expected_moe_shapes(cls.e)

    def test_router_and_expert_shapes(self):
        s = self.shapes
        self.assertEqual(s["router.weight"], [256, 4096])
        self.assertEqual(s["router.e_score_correction_bias"], [256])
        self.assertEqual(s["experts.gate_up_proj"], [256, 4096, 4096])
        self.assertEqual(s["experts.down_proj"], [256, 4096, 2048])
        self.assertEqual(s["shared_experts.gate_proj"], [2048, 4096])
        self.assertEqual(s["shared_experts.up_proj"], [2048, 4096])
        self.assertEqual(s["shared_experts.down_proj"], [4096, 2048])

    def test_layer4_uses_topk_router_not_hash(self):
        self.assertEqual(self.e["moe_type"], "moe")
        src = hx.code("modeling")
        self.assertIn('self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)', src)

    def test_router_constructor_params(self):
        for needle in (
            "self.top_k = config.num_experts_per_tok",
            "self.num_experts = config.num_local_experts",
            "self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))",
            "self.score_fn = ACT2FN[config.scoring_func]",
            "self.routed_scaling_factor = config.routed_scaling_factor",
            "self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts))",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, hx.code("modeling"))

    def test_experts_constructor_is_merged_3d(self):
        for needle in (
            "self.num_experts = config.num_local_experts",
            "self.intermediate_dim = config.intermediate_size",
            "self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))",
            "self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, hx.code("modeling"))

    def test_shared_expert_is_mlp_at_moe_width(self):
        src = hx.code("modeling")
        self.assertIn("self.shared_experts = DeepseekV4MLP(config)", src)
        self.assertIn("self.intermediate_size = config.intermediate_size", src)
        # attribute_map routes intermediate_size -> moe_intermediate_size (2048)
        self.assertEqual(self.e["intermediate_size"], 2048)


class RouterForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = hx.code("modeling")

    def test_router_forward_order(self):
        order = [
            "flat = hidden_states.reshape(-1, self.hidden_dim)",
            "logits = F.linear(flat, self.weight)",
            "scores = self.score_fn(logits)",
            "indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices",
            "weights = scores.gather(1, indices)",
            "weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)",
            "return logits, weights * self.routed_scaling_factor, indices",
        ]
        positions = [self.src.index(n) for n in order]
        self.assertEqual(positions, sorted(positions))

    def test_topk_selection_uses_bias_but_weights_do_not(self):
        self.assertIn("indices = torch.topk(scores + self.e_score_correction_bias, self.top_k", self.src)
        self.assertIn("weights = scores.gather(1, indices)", self.src)

    def test_scoring_func_is_sqrtsoftplus(self):
        act = hx.code("activations")
        self.assertIn("class SqrtSoftplusActivation(nn.Module):", act)
        self.assertIn("return nn.functional.softplus(input).sqrt()", act)
        self.assertIn('"sqrtsoftplus": SqrtSoftplusActivation', act)


class ExpertComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = hx.code("modeling")

    def test_experts_forward_merges_gate_up_then_down(self):
        for needle in (
            "mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)",
            "hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()",
            "top_k_pos, token_idx = torch.where(mask[expert_idx])",
            "current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))",
            "current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]",
            "final.index_add_(0, token_idx, current.to(final.dtype))",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.src)

    def test_apply_gate_clamps_and_multiplies(self):
        for needle in (
            "gate, up = gate_up.chunk(2, dim=-1)",
            "gate = gate.clamp(max=self.limit)",
            "up = up.clamp(min=-self.limit, max=self.limit)",
            "return self.act_fn(gate) * up",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.src)

    def test_swiglu_limit_value(self):
        e = hx.derive_effective(4)
        self.assertEqual(e["swiglu_limit"], 10.0)
        self.assertEqual(e["hidden_act"], "silu")

    def test_dynamic_token_dimension(self):
        # token axis is flattened at runtime and kept symbolic
        self.assertIn("flat = hidden_states.view(-1, hidden_dim)", self.src)
        self.assertIn("routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)", self.src)
        self.assertIn("batch, seq_len, hidden_dim = hidden_states.shape", self.src)

    def test_shared_expert_added_to_routed(self):
        self.assertIn("residual = hidden_states", self.src)
        self.assertIn("return routed + self.shared_experts(residual)", self.src)


class MoeNegativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = hx.code("modeling")

    def test_layer4_is_not_hash_moe(self):
        e = hx.derive_effective(4)
        self.assertNotEqual(e["moe_type"], "hash_moe")
        self.assertEqual(e["mlp_layer_types"][2], "hash_moe")

    def test_num_experts_and_top_k_are_distinct(self):
        e = hx.derive_effective(4)
        self.assertEqual(e["n_routed_experts"], 256)
        self.assertEqual(e["num_experts_per_tok"], 6)
        self.assertNotEqual(e["n_routed_experts"], e["num_experts_per_tok"])

    def test_experts_do_not_hold_separate_gate_and_up_parameters(self):
        experts = self.src.split("class DeepseekV4Experts", 1)[1].split("class DeepseekV4TopKRouter", 1)[0]
        self.assertIn("self.gate_up_proj", experts)
        self.assertNotIn("self.gate_proj = nn.Parameter", experts)
        self.assertNotIn("self.up_proj = nn.Parameter", experts)

    def test_gate_clamp_is_one_sided_up_clamp_is_two_sided(self):
        self.assertIn("gate = gate.clamp(max=self.limit)", self.src)
        self.assertIn("up = up.clamp(min=-self.limit, max=self.limit)", self.src)
        self.assertNotIn("gate = gate.clamp(min=-self.limit, max=self.limit)", self.src)

    def test_routed_scaling_applied_after_normalisation(self):
        norm_pos = self.src.index("weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)")
        scale_pos = self.src.index("return logits, weights * self.routed_scaling_factor, indices")
        self.assertLess(norm_pos, scale_pos)


if __name__ == "__main__":
    unittest.main()
