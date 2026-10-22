"""Self-contained tests for the ONNX-style view adapter (positives/negatives).

Synthetic SVG only; no candidate/outputs/coordination reads.
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import onnx_view_check as ov  # noqa: E402


def _svg(extra="", edges=""):
    return (
        '<h4 id="mod-p-prefill-a">x</h4>'
        '<svg class="dg" viewBox="0 0 600 220" role="img" xmlns="http://www.w3.org/2000/svg">'
        '<defs><marker id="dg-arr-flow-p-a"/></defs>'
        f"{edges}"
        '<a href="#op-p-op_a"><g class="dg-node op" transform="translate(103,16)"><rect width="172" height="49" rx="7"/>'
        '<text class="nlabel" x="86" y="14" text-anchor="middle">A</text>'
        '<text class="nsub" x="86" y="43" text-anchor="middle">Linear</text></g></a>'
        '<a href="#t-p-x"><g class="dg-dot port" transform="translate(26,40)"><circle r="5"/></g></a>'
        '<a href="#t-p-h"><g class="dg-port io" transform="translate(16,150)"><rect width="172" height="24" rx="12"/>'
        '<text class="portlabel" x="86" y="16" text-anchor="middle">hidden_states（层输入）</text></g></a>'
        f"{extra}"
        "</svg>"
    )


def _edge(x1, y1, x2, y2, kind="flow", tid="x"):
    return (f'<a href="#t-p-{tid}"><path class="dg-edge {kind}" d="M{x1},{y1} C{(x1+x2)/2},{y1} {(x1+x2)/2},{y2} {x2},{y2}" '
            'marker-end="url(#dg-arr-flow-p-a)"/></a>')


class ParseTests(unittest.TestCase):
    def test_classification(self):
        figs = ov.parse(_svg(edges=_edge(32, 40, 100, 40.5)))
        f = figs[0]
        kinds = sorted(e.kind for e in f.elems)
        self.assertIn("compute", kinds)
        self.assertIn("io", kinds)
        self.assertIn("dot", kinds)
        io = [e for e in f.elems if e.kind == "io"][0]
        self.assertAlmostEqual(io.h, 24.0)          # IO identifier 24px
        comp = [e for e in f.elems if e.kind == "compute"][0]
        self.assertAlmostEqual(comp.w, 172.0)
        self.assertGreater(comp.h, 24.0)            # compute box distinct from IO
        dot = [e for e in f.elems if e.kind == "dot"][0]
        self.assertEqual(dot.node_id, "t:x")

    def test_resolve_correct_direction(self):
        figs = ov.parse(_svg(edges=_edge(32, 40, 100, 40.5)))
        res = ov.resolve_edges(figs[0])
        self.assertEqual(res, [("t:x", "op:op_a", "x", "flow", True)])

    def test_reversed_edge_detected(self):
        # op -> dot (wrong direction for a dot->op expectation)
        figs = ov.parse(_svg(edges=_edge(275, 40.5, 20, 40)))
        res = ov.resolve_edges(figs[0])
        self.assertEqual(res[0][:4], ("op:op_a", "t:x", "x", "flow"))
        expected = {("t:x", "op:op_a", "x", "flow")}
        geo = {(r[0], r[1], r[2], r[3]) for r in res if r[4]}
        missing = expected - geo
        rev = [m for m in missing if (m[1], m[0], m[2], m[3]) in geo]
        self.assertTrue(rev, "reversed edge must be detectable")

    def test_wrong_endpoint_unresolved(self):
        figs = ov.parse(_svg(edges=_edge(32, 40, 999, 999)))
        res = ov.resolve_edges(figs[0])
        self.assertFalse(res[0][4])  # endpoint not at any node anchor -> unresolved

    def test_old_162px_dangling_dot(self):
        # dot centre (26,119) but edge starts at (188,131.5): ~162px away -> not connected
        dot = '<a href="#t-p-y"><g class="dg-dot weight" transform="translate(26,119)"><circle r="5"/></g></a>'
        figs = ov.parse(_svg(extra=dot, edges=_edge(188, 131.5, 275, 40.5, kind="weight", tid="y")))
        res = ov.resolve_edges(figs[0])
        self.assertFalse(res[0][4], "dangling dot edge must not resolve to the far dot")

    def test_io_anchor(self):
        # edge out of the IO identifier: out anchor = (x+w, y+h/2) = (188,162)
        figs = ov.parse(_svg(edges=_edge(188, 162, 275, 40.5)))
        res = ov.resolve_edges(figs[0])
        self.assertEqual(res[0][0], "t:h")


class ColourTests(unittest.TestCase):
    def test_contrast_monotonic(self):
        self.assertGreater(ov.contrast("#1f2933", "#fcfdfe"), 10)
        self.assertLess(ov.contrast("#fcfdfe", "#ffffff"), 1.2)


if __name__ == "__main__":
    unittest.main()
