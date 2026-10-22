"""Self-contained tests for round02 direction + block-occlusion checks.

Synthetic fixtures only; no reads of the real candidate/outputs/coordination.
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import parse_svg as ps  # noqa: E402
import graph_expect as gx  # noqa: E402
import run_round02 as rr  # noqa: E402


def _t(tid, kind="activation"):
    return {"id": tid, "kind": kind, "shape": ["B", "T", "H"]}


def _o(oid, inputs, outputs, weights=()):
    return {"id": oid, "inputs": list(inputs), "outputs": list(outputs), "weights": list(weights)}


def _phase():
    return {"id": "prefill", "tensors": [_t("x", "boundary"), _t("y"), _t("z")],
            "operators": [_o("op_a", ["x"], ["y"]), _o("op_b", ["y"], ["z"])], "parameters": []}


def _fig(nodes, edges, labels=(), w=800.0, h=400.0):
    return ps.Figure("mod-p-g", "p", "module", "g", w, h, list(nodes), list(edges), list(labels), "")


def _node(href, kind, x, y, w=172.0, h=44.0):
    return ps.Node(href, kind, x, y, w, h, ["n"])


def _edge(start, end, kind="flow", tid=None, href=None):
    x1, y1 = start
    xe, ye = end
    return ps.Edge(x1, y1, (x1 + xe) / 2, y1, (x1 + xe) / 2, ye, xe - 3.0, ye, kind, href, tid, False)


class DirectionTests(unittest.TestCase):
    def setUp(self):
        self.a = _node("#op-p-op_a", "op", 16, 16)
        self.b = _node("#op-p-op_b", "op", 292, 16)
        self.expected = gx.expected_edges(_phase(), {"op_a", "op_b"})
        self.assertIn(("op:op_a", "op:op_b", "y", "flow"), self.expected)

    def _geo(self, edges):
        fig = _fig([self.a, self.b], edges)
        res = rr.resolve_edge_endpoints(fig)
        return {(r["src"], r["dst"], r["tid"], r["edge"].kind) for r in res}

    def test_correct_direction(self):
        geo = self._geo([_edge(self.a.out_anchor, self.b.in_anchor, tid="y")])
        self.assertIn(("op:op_a", "op:op_b", "y", "flow"), geo)
        self.assertEqual(geo & self.expected, {("op:op_a", "op:op_b", "y", "flow")})

    def test_reversed_edge_detected(self):
        # edge drawn B -> A (wrong direction) for tensor y
        geo = self._geo([_edge(self.b.out_anchor, self.a.in_anchor, tid="y")])
        missing = self.expected - geo
        extra = geo - self.expected
        reversed_edges = [m for m in missing if (m[1], m[0], m[2], m[3]) in extra]
        self.assertTrue(reversed_edges, "reversed edge must be detectable")

    def test_wrong_endpoint_unresolved(self):
        fig = _fig([self.a, self.b], [_edge(self.a.out_anchor, (999.0, 999.0), tid="y")])
        res = rr.resolve_edge_endpoints(fig)
        self.assertIsNone(res[0]["dst"], "endpoint not at any node anchor must be unresolved")

    def test_dropped_edge(self):
        geo = self._geo([])  # no edges rendered
        self.assertEqual(self.expected - geo, self.expected)


class BlockOcclusionTests(unittest.TestCase):
    def test_block_overlap_counted(self):
        node = _node("#op-p-op_a", "op", 0, 0, 172, 44)
        L = ps.Label(120.0, 20.0, ["追加 N_win_i 条（可为 0）"], "追加 N_win_i 条（可为 0）", True)
        fig = _fig([node], [], [L])
        issues, report = [], {}
        rr.geometry([fig], issues, report)
        self.assertGreater(report["geometry"]["label_block_overlap"], 0)

    def test_robust_tail_occlusion_flagged(self):
        # node to the right of the centre; a wide label whose tail reaches well inside the node
        node = _node("#op-p-op_a", "op", 200, 0, 172, 44)
        L = ps.Label(190.0, 20.0, ["追加 N_win_i 条（可为 0）"], "追加 N_win_i 条（可为 0）", True)
        fig = _fig([node], [], [L])
        issues, report = [], {}
        rr.geometry([fig], issues, report)
        self.assertEqual(report["geometry"]["anchor_in_node"], 0)  # centre outside
        self.assertTrue(any(c == "label_block_overlap" for _s, c, _m in issues))

    def test_anchor_only_not_flagged_as_anchor(self):
        # centre outside node -> anchor_in_node 0 even if block overlaps
        node = _node("#op-p-op_a", "op", 0, 0, 60, 44)
        L = ps.Label(120.0, 20.0, ["x" * 40], "x" * 40, False)
        fig = _fig([node], [], [L])
        issues, report = [], {}
        rr.geometry([fig], issues, report)
        self.assertEqual(report["geometry"]["anchor_in_node"], 0)
        self.assertGreater(report["geometry"]["label_block_overlap"], 0)

    def test_clean_label_no_overlap(self):
        node = _node("#op-p-op_a", "op", 0, 0, 60, 44)
        L = ps.Label(400.0, 200.0, ["short"], "short", False)
        fig = _fig([node], [], [L])
        issues, report = [], {}
        rr.geometry([fig], issues, report)
        self.assertEqual(report["geometry"]["label_block_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
