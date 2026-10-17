"""mHC (Manifold-Constrained Hyper-Connection) structure checks for layer_idx=4.

Each decoder layer applies mHC twice (attn_hc before attention, ffn_hc before
the MoE block). The residual mixing direction is the high-risk fact: the source
consumes ``comb`` transposed (``comb.T @ residual``, i.e. ``sum_j comb[j,k]*h[j,d]``),
which is NOT the same as ``comb @ residual`` for a non-symmetric Sinkhorn matrix.
"""

import unittest
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hf_expectations as hx  # noqa: E402


def _matmul(a, b):
    n, m, p = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def _transpose(a):
    return [list(row) for row in zip(*a)]


class MhcShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = hx.derive_effective(4)
        cls.shapes = hx.expected_mhc_shapes(cls.e)

    def test_weight_shapes(self):
        s = self.shapes
        self.assertEqual(self.e["hc_mult"], 4)
        self.assertEqual(self.e["hidden_size"], 4096)
        self.assertEqual(s["fn"], [24, 16384])
        self.assertEqual(s["base"], [24])
        self.assertEqual(s["scale"], [3])
        self.assertEqual(s["cws"], (2 + self.e["hc_mult"]) * self.e["hc_mult"])

    def test_stream_and_weight_axes(self):
        s = self.shapes
        self.assertEqual(s["streams"], ["B", "S", 4, 4096])
        self.assertEqual(s["collapsed"], ["B", "S", 4096])
        self.assertEqual(s["pre"], ["B", "S", 4])
        self.assertEqual(s["post"], ["B", "S", 4])
        self.assertEqual(s["comb"], ["B", "S", 4, 4])

    def test_mhc_constructor_matches_expected_shapes(self):
        src = hx.code("modeling")
        for needle in (
            "concatenated_weights_size = (2 + self.hc_mult) * self.hc_mult",
            "self.fn = nn.Parameter(torch.empty(concatenated_weights_size, self.hc_mult * config.hidden_size))",
            "self.base = nn.Parameter(torch.empty(concatenated_weights_size))",
            "self.scale = nn.Parameter(torch.empty(3))",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, src)

    def test_weight_formulas(self):
        self.assertIn("flattened.split([hc, hc, hc * hc], dim=-1)", hx.code("modeling"))
        self.assertIn("pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps", hx.code("modeling"))
        self.assertIn("post = 2 * torch.sigmoid(post_w * post_scale + post_b)", hx.code("modeling"))
        self.assertIn("comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps", hx.code("modeling"))

    def test_sinkhorn_order_and_iterations(self):
        src = hx.code("modeling")
        self.assertIn("comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)", src)
        self.assertIn("for _ in range(self.hc_sinkhorn_iters - 1):", src)
        self.assertIn("comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)", src)
        self.assertIn("comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)", src)
        # first column normalisation appears before the alternating row/col loop
        self.assertLess(src.index("comb.sum(dim=-2"), src.index("for _ in range(self.hc_sinkhorn_iters - 1)"))

    def test_pre_is_consumed_as_collapse(self):
        src = hx.code("modeling")
        self.assertIn("collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)", src)
        self.assertIn("return post, comb, collapsed", src)


class MhcDirectionTests(unittest.TestCase):
    def test_decoder_uses_comb_transposed(self):
        src = hx.code("modeling")
        attn_mix = (
            "hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) "
            "+ torch.matmul( comb.to(dtype).transpose(-1, -2), hidden_states )"
        )
        ffn_mix = (
            "return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) "
            "+ torch.matmul( comb.to(dtype).transpose(-1, -2), hidden_states )"
        )
        self.assertIn(attn_mix, src)
        self.assertIn(ffn_mix, src)

    def test_post_expands_output_before_mix(self):
        src = hx.code("modeling")
        self.assertIn("post, comb, collapsed = self.attn_hc(hidden_states)", src)
        self.assertIn("post, comb, collapsed = self.ffn_hc(hidden_states)", src)

    def test_hidden_streams_are_expanded_at_model_entry(self):
        src = hx.code("modeling")
        self.assertIn(
            "hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()",
            src,
        )

    def test_direction_matters_for_non_symmetric_comb(self):
        # doubly-stochastic but NOT symmetric (rows and columns each sum to 1)
        comb = [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ]
        residual = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
        for axis in (0, 1):
            self.assertAlmostEqual(sum(comb[axis]), 1.0)
            self.assertAlmostEqual(sum(_transpose(comb)[axis]), 1.0)
        # independent expectation: mixed[k,d] = sum_j comb[j,k] * residual[j,d]
        expected = _matmul(_transpose(comb), residual)
        wrong = _matmul(comb, residual)
        self.assertEqual(expected, [[3.0, 4.0], [2.0, 3.0], [4.0, 5.0]])
        self.assertNotEqual(expected, wrong)
        # and the source implements the `expected` orientation
        with_transpose = "torch.matmul( comb.to(dtype).transpose(-1, -2), hidden_states )"
        without = "torch.matmul( comb.to(dtype), hidden_states )"
        src = hx.code("modeling")
        self.assertIn(with_transpose, src)
        self.assertNotIn(without, src)


if __name__ == "__main__":
    unittest.main()
