"""Self-contained tests for the diagram gate expectations and geometry checks.

Uses small synthetic fixtures (clearly not the real candidate); no reads of the
real candidate, outputs or coordination state.
"""

import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import parse_html as ph  # noqa: E402
import graph_expect as ge  # noqa: E402
import run_diagram_gate as rg  # noqa: E402


def _t(tid, kind="activation", shape=None):
    return {"id": tid, "kind": kind, "shape": shape or ["B", "T", "H"]}


def _o(oid, inputs, outputs, weights=()):
    return {"id": oid, "inputs": list(inputs), "outputs": list(outputs), "weights": list(weights)}


def _phase():
    return {
        "id": "prefill",
        "tensors": [
            _t("x", "boundary"), _t("y"), _t("w", "weight", ["H", "H"]),
            _t("c_in", "cache"), _t("c_out", "cache"), _t("c_hist", "cache"),
            _t("seed"),
        ],
        "operators": [
            _o("op_a", ["x"], ["y"]),
            _o("op_b", ["y", "w"], ["z"]),
            _o("op_cache", ["seed", "c_hist"], ["c_out", "c_hist2"]),
        ],
        "parameters": [],
    }


class GraphExpectTests(unittest.TestCase):
    def test_module_edges(self):
        counts = ge.expected_module_edges(_phase(), {"op_a", "op_b"})
        # op_a: input x external port; output y internal to op_b
        # op_b: input y internal (skip); weight w external; output z output port
        self.assertEqual(counts[("x", "flow")], 1)
        self.assertEqual(counts[("y", "flow")], 1)  # internal op_a->op_b
        self.assertEqual(counts[("w", "weight")], 1)
        self.assertEqual(counts[("z", "flow")], 1)
        self.assertEqual(sum(counts.values()), 4)

    def test_dropped_edge_detected(self):
        # removing a module op changes the expected edge set (used to flag a dropped edge)
        before = ge.expected_module_edges(_phase(), {"op_a", "op_b"})
        after = ge.expected_module_edges(_phase(), {"op_a"})
        self.assertNotEqual(before, after)
        # op_a output y no longer has an in-set downstream -> becomes an output port, but the
        # internal op_a->op_b edge count is gone from the pair view
        self.assertEqual(after[("w", "weight")], 0)

    def test_cache_roles(self):
        p = _phase()
        roles = ge.cache_roles(p, {"c_in", "c_out", "c_hist", "c_hist2"}, {"op_cache"})
        # c_in has no producer -> prior state; c_out produced by op_cache, no outside consumer -> write-back
        self.assertEqual(roles["c_in"], "调用前状态")
        self.assertEqual(roles["c_out"], "调用后回写")
        # make c_out consumed outside
        p2 = copy.deepcopy(p)
        p2["operators"].append(_o("op_use", ["c_out"], ["q"]))
        roles2 = ge.cache_roles(p2, {"c_in", "c_out", "c_hist", "c_hist2"}, {"op_cache"})
        self.assertEqual(roles2["c_out"], "本次读视图")

    def test_downstream_edges(self):
        p = _phase()
        p["operators"].append(_o("op_use", ["c_out"], ["q"]))
        ds = ge.downstream_edges(p, ["c_out"], {"op_cache"})
        self.assertEqual(ds, [("c_out", "op_use")])


class ParseTests(unittest.TestCase):
    SVG = (
        '<svg class="dg" viewBox="0 0 400 100" role="img" xmlns="http://www.w3.org/2000/svg">'
        '<defs><marker id="dg-arr-flow-p-g"/></defs>'
        '<a href="#t-p-a"><path class="dg-edge flow" d="M0,0" marker-end="url(#dg-arr-flow-p-g)"/></a>'
        '<path class="dg-edge cache" d="M0,0" marker-end="url(#dg-arr-cache-p-g)"/>'
        '<a href="#op-p-op_a"><g class="dg-node op" transform="translate(10,10)"><rect width="172" height="44" rx="7"/>'
        '<text class="nlabel" x="86" y="14" text-anchor="middle">A</text></g></a>'
        '<a href="#op-p-op_use"><g class="dg-node downstream" transform="translate(10,60)"><rect width="172" height="44" rx="7"/>'
        '<text class="nlabel" x="86" y="14" text-anchor="middle">下游算子 op_use</text></g></a>'
        "</svg>"
    )

    def test_parse_edges_nodes(self):
        figs = ph.parse("<h4 id=\"mod-p-prefill-a\">x</h4>" + self.SVG)
        self.assertEqual(len(figs), 1)
        f = figs[0]
        self.assertEqual(f.kind, "module")
        self.assertEqual(f.edge_hrefs, [("a", "flow")])
        self.assertEqual(f.op_ids(), ["op_a"])
        self.assertEqual(f.downstream_ids(), ["op_use"])
        total = len(ph.re.findall(r'<path class="dg-edge ', f.raw))
        self.assertEqual(total - len(f.edge_hrefs), 1)


def _fig(nodes, labels, w=400.0, h=200.0):
    return ph.Figure("mod-p-g", "p", "module", "g", w, h, [], [], [], nodes, labels, "")


class GeometryTests(unittest.TestCase):
    def test_anchor_in_node(self):
        n = ph.Node("#op-p-a", "op", 0, 0, 100, 50, ["A"], "")
        L = ph.EdgeLabel(50, 25, ["label"], "label", False)
        f = rg.geometry_findings([_fig([n], [L])])
        self.assertEqual(len(f["label_anchor_in_node"]), 1)

    def test_label_out_of_canvas(self):
        n = ph.Node("#op-p-a", "op", 0, 0, 100, 50, ["A"], "")
        L = ph.EdgeLabel(300, 300, ["a", "b", "c"], "abc", False)
        f = rg.geometry_findings([_fig([n], [L])])
        self.assertEqual(len(f["label_out_of_canvas"]), 1)

    def test_spacing_violation(self):
        L1 = ph.EdgeLabel(50, 100, ["a"], "a", False)
        L2 = ph.EdgeLabel(50, 105, ["b"], "b", False)
        f = rg.geometry_findings([_fig([], [L1, L2])])
        self.assertEqual(len(f["label_spacing_violations"]), 1)

    def test_node_overlap(self):
        n1 = ph.Node("#op-p-a", "op", 0, 0, 100, 50, ["A"], "")
        n2 = ph.Node("#op-p-b", "op", 50, 25, 100, 50, ["B"], "")
        f = rg.geometry_findings([_fig([n1, n2], [])])
        self.assertEqual(len(f["node_overlap"]), 1)

    def test_node_out_of_canvas(self):
        n = ph.Node("#op-p-a", "op", 350, 0, 100, 50, ["A"], "")
        f = rg.geometry_findings([_fig([n], [], w=400, h=200)])
        self.assertEqual(len(f["node_out_of_canvas"]), 1)


if __name__ == "__main__":
    unittest.main()
