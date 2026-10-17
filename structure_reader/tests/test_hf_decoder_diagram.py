"""HF decoder 阅读页连接图自测（图示修复01；自包含 fixture，不读 coordination/outputs）。

核查原则：图边必须对应 fixture/operators 的真实 producer→consumer 张量引用，
不得把 builder 输出当正确期望；缓存角色来自 producer/consumer 而非命名后缀猜测；
渲染确定性；SVG 为结构化断言（去重/标签错位/marker id/节点容纳），非字符串碰瓷。

运行（工程 Python）：
  python3 structure_reader/tests/test_hf_decoder_diagram.py
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from structure_reader.render.diagram import (  # noqa: E402
    build_cache_graphs,
    build_module_graph,
    build_overview_graph,
    diagram_css,
    legend_html,
    module_groups,
    render_svg,
)


def _op(oid, inputs, outputs, weights=()):
    return {
        "id": oid, "module_path": "Fixture", "name": oid, "op": "fixture",
        "inputs": list(inputs), "outputs": list(outputs), "weights": list(weights),
        "parameter_refs": [], "source_refs": [], "analysis_status": "analyzed",
        "note": "",
    }


def _tensor(tid, kind, shape):
    return {"id": tid, "name": tid, "shape": list(shape), "parameter_refs": [],
            "kind": kind, "note": ""}


def fixture_phase() -> dict:
    """手工小图：分叉（a→op2/op3）、汇合（op4 双输入）、外部边界输入、
    共享权重引用（w→op1/op2）、缓存调用前/后 + 本次读视图 + 兼保存 + 条件写 +
    缓存算子间 activation 链（op5→op7）与族外下游（op6）。"""
    return {
        "id": "prefill",
        "title": "fixture",
        "parameters": [
            {"id": "H", "symbol": "H", "meaning": "hidden", "value": 4,
             "expression": None, "note": "", "source_refs": []},
            {"id": "Nwin", "symbol": "N_win", "meaning": "windows", "value": None,
             "expression": "floor(.../M)", "note": "", "source_refs": []},
        ],
        "tensors": [
            _tensor("x_in", "boundary", ["B", "T", "H"]),
            _tensor("w", "weight", ["H", "H"]),
            _tensor("a", "activation", ["B", "T", "H"]),
            _tensor("b", "activation", ["B", "T", "H"]),
            _tensor("c", "activation", ["B", "T", "H"]),
            _tensor("y", "activation", ["B", "T", "H"]),
            _tensor("new_kv", "activation", ["B", "T", "Dh"]),
            _tensor("cache_pre", "cache", ["B", "M", "Dh"]),
            _tensor("cache_post", "cache", ["B", "M", "Dh"]),
            _tensor("view", "cache", ["B", "L", "Dh"]),
            _tensor("accum_post", "cache", ["B", "Tacc", "Dh"]),
        ],
        "operators": [
            _op("op1", ["x_in", "w"], ["a"]),
            _op("op2", ["a", "w"], ["b"]),
            _op("op3", ["a"], ["c"]),
            _op("op4", ["b", "c"], ["y"]),
            _op("op_new", ["x_in"], ["new_kv"]),
            _op("op5", ["cache_pre", "new_kv"], ["cache_post", "view"]),
            _op("op7", ["view"], ["accum_post"]),
            _op("op6", ["accum_post", "view"], []),
        ],
        "cache_notes": ["x"], "coverage_notes": ["x"],
    }


FIXTURE_RULES = (
    ("g1", "组一", ("op1", "op2")),
    ("g2", "组二", ("op3",)),
    ("g3", "组三", ("op4",)),
    ("gnew", "新输入", ("op_new",)),
    ("gcache", "缓存组", ("op5", "op7")),
    ("gdown", "下游", ("op6",)),
)

FIXTURE_CACHE_FAMILIES = {"fam": ("cache_", "view", "accum_")}
FIXTURE_CACHE_OPS = {"op5", "op7"}


def _edge_set(graph):
    return {(e["src"], e["dst"], e["kind"]) for e in graph["edges"]}


class TestModuleGraphConnections(unittest.TestCase):
    """模块图边 = 真实 producer→consumer；分叉/汇合/边界/共享权重/缓存可见；无重复边。"""

    def setUp(self):
        self.phase = fixture_phase()
        self.groups = {g["id"]: g for g in module_groups(self.phase, FIXTURE_RULES)}
        self.g1 = build_module_graph(self.phase, self.groups["g1"])
        self.g3 = build_module_graph(self.phase, self.groups["g3"])
        self.gcache = build_module_graph(self.phase, self.groups["gcache"])

    def test_fork_and_join(self):
        # 分叉（组内可见）：a 由 op1 产出，被 op2 消费
        e1 = _edge_set(self.g1)
        self.assertIn(("op:op1", "op:op2", "flow"), e1)
        # 汇合（跨组以端口呈现）：op4 两条数据流入边（b、c 均为组外产出端口）
        e3 = _edge_set(self.g3)
        self.assertIn(("t:b", "op:op4", "flow"), e3)
        self.assertIn(("t:c", "op:op4", "flow"), e3)
        # 组内链：gcache 中 op5→op7（经 view）直连
        self.assertIn(("op:op5", "op:op7", "cache"), _edge_set(self.gcache))

    def test_no_duplicate_edges(self):
        """同一路径/张量/类型/条件的边只出现一次（input/output 双循环去重）。"""
        for g in (self.g1, self.g3, self.gcache):
            keys = [(e["src"], e["dst"], e["kind"], e["title"], e["cond"]) for e in g["edges"]]
            self.assertEqual(len(keys), len(set(keys)), f"{g['id']}: 存在重复边")

    def test_external_boundary_input_port(self):
        # x_in 无 producer → 边界端口节点 → op1
        self.assertIn(("t:x_in", "op:op1", "flow"), _edge_set(self.g1))
        node = next(n for n in self.g1["nodes"] if n["id"] == "t:x_in")
        self.assertEqual(node["kind"], "boundary")
        self.assertEqual(node["href"], "#t-prefill-x_in")

    def test_shared_weight_reference(self):
        # w 被两个算子共享引用 → 两条 weight 边（引用，不改变数据流）
        e1 = _edge_set(self.g1)
        self.assertIn(("t:w", "op:op1", "weight"), e1)
        self.assertIn(("t:w", "op:op2", "weight"), e1)
        kinds = {n["id"]: n["kind"] for n in self.g1["nodes"]}
        self.assertEqual(kinds["t:w"], "weight")

    def test_edges_reference_real_tensors_only(self):
        """每条边的 title（张量 id）必须存在于文档 tensors——不捏造连接。"""
        tids = {t["id"] for t in self.phase["tensors"]}
        for graph in (self.g1, self.g3, self.gcache):
            for e in graph["edges"]:
                self.assertIn(e["title"], tids, f"边 {e['src']}->{e['dst']} 引用未知张量")

    def test_group_coverage_complete(self):
        """分组覆盖所有算子恰一次；聚合边可追溯真实张量。"""
        ops = module_groups(self.phase, FIXTURE_RULES)
        covered = [o["id"] for g in ops for o in g["ops"]]
        self.assertEqual(sorted(covered), sorted(o["id"] for o in self.phase["operators"]))
        ov = build_overview_graph(self.phase, FIXTURE_RULES)
        producers = {}
        consumers = {}
        for i, op in enumerate(self.phase["operators"]):
            for tid in op["outputs"]:
                producers.setdefault(tid, []).append(i)
            for tid in op["inputs"]:
                consumers.setdefault(tid, []).append(i)
        op_group = {}
        for g in ops:
            for o in g["ops"]:
                op_group[o["id"]] = g["id"]
        for e in ov["edges"]:
            if e["src"].startswith("m:") and e["dst"].startswith("m:"):
                supported = any(
                    op_group.get(self.phase["operators"][p]["id"]) == e["src"][2:]
                    and op_group.get(self.phase["operators"][c]["id"]) == e["dst"][2:]
                    for tid in e["title"].split("，")
                    for p in producers.get(tid, [])
                    for c in consumers.get(tid, [])
                )
                self.assertTrue(supported, f"聚合边 {e['src']}->{e['dst']} 无真实张量支撑")


class TestCacheGraphFixture(unittest.TestCase):
    """缓存图：真实下游 chip、角色来自引用（非后缀）、本次新输入与算子间数据路径。"""

    def setUp(self):
        self.phase = fixture_phase()
        self.graph = build_cache_graphs(
            self.phase, family_rules=FIXTURE_CACHE_FAMILIES,
            cache_op_ids=FIXTURE_CACHE_OPS)[0]
        self.nodes = {n["id"]: n for n in self.graph["nodes"]}
        self.edges = self.graph["edges"]

    def test_downstream_chips_from_real_consumers(self):
        """下游 chip 来自真实族外消费者（op6 消费 accum_post/view），非 producers。"""
        chips = {e["dst"] for e in self.edges if e["dst"].startswith("d:")}
        self.assertEqual(chips, {"d:op6"})
        chip = self.nodes["d:op6"]
        self.assertEqual(chip["href"], "#op-prefill-op6")

    def test_roles_from_producer_consumer_not_suffix(self):
        """角色断言：调用前（无 producer）/回写（无族外消费）/读视图（有族外消费，
        无后缀）/读视图·兼保存（有族外消费且 _post）。"""
        roles = {tid: self.nodes[f"t:{tid}"]["sub"].split(" · ")[0]
                 for tid in ("cache_pre", "cache_post", "view", "accum_post")}
        self.assertEqual(roles["cache_pre"], "调用前状态")      # 无 producer
        self.assertEqual(roles["cache_post"], "调用后回写")     # 有 producer、无族外消费
        self.assertEqual(roles["view"], "本次读视图")           # 有 producer+族外消费，无 _post
        self.assertEqual(roles["accum_post"], "本次读视图·兼调用后保存")

    def test_this_call_inputs_and_internal_activation_path(self):
        """缓存图含本次新输入端口（new_kv）与缓存算子间真实 activation 边（op5→op7）。"""
        self.assertIn(("t:new_kv", "op:op5", "flow"), _edge_set(self.graph))
        self.assertIn(("op:op5", "op:op7", "cache"), _edge_set(self.graph))
        node = self.nodes["t:new_kv"]
        self.assertEqual(node["kind"], "port")

    def test_cond_label_on_conditional_write(self):
        """条件写边保留条件标注（条件算子+参数齐备才渲染）。"""
        self.phase["operators"][5]["id"] = "c_window_view"
        graph = build_cache_graphs(self.phase, family_rules=FIXTURE_CACHE_FAMILIES,
                                   cache_op_ids={"c_window_view", "op7"})[0]
        conds = [e["cond"] for e in graph["edges"] if e["cond"]]
        self.assertIn("仅 N_win>0", conds)

    def test_edges_truthful(self):
        """每条边 src 真实产出/目标真实消费（对 fixture 逐一核对）。"""
        producers, consumers = {}, {}
        for op in self.phase["operators"]:
            for tid in op["outputs"]:
                producers.setdefault(tid, set()).add(op["id"])
            for tid in op["inputs"]:
                consumers.setdefault(tid, set()).add(op["id"])
        out_index = {op["id"]: set(op["outputs"]) for op in self.phase["operators"]}
        for e in self.edges:
            src, dst = e["src"], e["dst"]
            if src.startswith("op:"):
                self.assertIn(e["title"], out_index[src[3:]],
                              f"{src} 不产出 {e['title']}")
            if dst.startswith("op:"):
                self.assertIn(dst[3:], consumers.get(e["title"], set()),
                              f"{dst} 不消费 {e['title']}")


class TestSvgRendering(unittest.TestCase):
    """SVG 结构化输出：节点/边/锚点/去重/标签错位/marker 唯一/节点容纳/转义/确定性。"""

    def setUp(self):
        self.phase = fixture_phase()
        self.graph = build_module_graph(
            self.phase, module_groups(self.phase, FIXTURE_RULES)[0])

    def test_svg_structure_and_anchors(self):
        svg = render_svg(self.graph)
        self.assertIn("<svg", svg)
        self.assertIn('class="dg-node op"', svg)
        self.assertIn('<path class="dg-edge flow"', svg)
        self.assertIn('<path class="dg-edge weight"', svg)
        self.assertIn('href="#op-prefill-op1"', svg)
        self.assertIn('href="#t-prefill-x_in"', svg)
        self.assertIn('href="#t-prefill-w"', svg)

    def test_marker_ids_unique_per_graph_and_referenced(self):
        """marker id 带图唯一 uid；两张图 id 不冲突且引用与定义一致。"""
        svg1 = render_svg(self.graph)
        g2 = build_module_graph(self.phase, module_groups(self.phase, FIXTURE_RULES)[2])
        svg2 = render_svg(g2)
        ids1 = set(re.findall(r'marker id="([^"]+)"', svg1))
        ids2 = set(re.findall(r'marker id="([^"]+)"', svg2))
        self.assertEqual(len(ids1), 3)
        self.assertEqual(len(ids2), 3)
        self.assertFalse(ids1 & ids2, "不同图的 marker id 不得重名")
        for svg, ids in ((svg1, ids1), (svg2, ids2)):
            refs = set(re.findall(r'marker-end="url\(#([^)]+)\)"', svg))
            self.assertTrue(refs <= ids, "箭头引用必须指向本图已定义 marker")

    def test_no_duplicate_edges_in_svg_paths(self):
        """渲染层不含同 (src,dst,张量,类型) 重复路径。"""
        svg = render_svg(self.graph)
        paths = re.findall(r'<path class="dg-edge [^"]*" d="([^"]+)"', svg)
        self.assertEqual(len(paths), len(set(paths)))

    def test_parallel_edge_labels_offset(self):
        """同起终点的平行张量边标签 y 错位（不叠在同一点）。"""
        phase = self.phase
        phase["operators"][0]["outputs"] = ["a"]
        phase["operators"][2]["inputs"] = ["a", "a"]  # 同张量重复引用为同一 id，改用双张量
        phase["operators"][2]["inputs"] = ["a", "b"]
        phase["operators"][1]["outputs"] = ["b"]
        g = build_module_graph(phase, {"id": "par", "title": "p",
                                       "ops": [phase["operators"][0],
                                               phase["operators"][2]]})
        svg = render_svg(g)
        ys = re.findall(r'<text class="elabel" x="[\d.-]+" y="([\d.-]+)"', svg)
        self.assertGreaterEqual(len(ys), 2)
        self.assertNotEqual(ys[0], ys[1], "平行边标签应错位")

    def test_multiline_label_blocks_do_not_overlap(self):
        """自包含两行回放：多行标签按实际占用排布——同列相邻标签块
        （首行基线到末行基线）间距 ≥13px，而非仅首行坐标不同。"""
        phase = self.phase
        # 构造同列两条两行标签：一对节点间两条平行长名张量边（共享路径中点）
        op_a = _op("opLA", ["x_in"], ["la", "lb"])
        op_b = _op("opLB", ["la", "lb"], ["lc"])
        op_c = _op("opLC", ["lc"], ["y"])
        phase["operators"] += [op_a, op_b, op_c]
        phase["tensors"] += [
            _tensor("la", "activation", ["B", "T", "H"]),
            _tensor("lb", "activation", ["B", "T", "H"]),
            _tensor("lc", "activation", ["B", "T", "H"]),
        ]
        for t in phase["tensors"]:
            if t["id"] in ("la", "lb"):
                t["name"] = f"超长张量名称{t['id']}超长张量名称{t['id']}"
        g = build_module_graph(phase, {"id": "ml", "title": "m",
                                       "ops": [op_a, op_b, op_c]})
        svg = render_svg(g)
        blocks = self._label_blocks(svg)
        self.assertTrue(any(n >= 2 for _, _, n in blocks), "回放应含两行标签")
        by_x = {}
        for x, y0, n in blocks:
            by_x.setdefault(x, []).append((y0, n))
        worst = None
        for items in by_x.values():
            items.sort()
            prev_bottom = None
            for y0, n in items:
                bottom = y0 + 11.0 * (n - 1)
                if prev_bottom is not None:
                    gap = y0 - prev_bottom
                    if worst is None or gap < worst:
                        worst = gap
                prev_bottom = bottom
        self.assertIsNotNone(worst)
        self.assertGreaterEqual(worst, 12.9,
                                f"多行标签块间距 {worst} < 13px（末行→首行）")
        # 画布高度覆盖推让后的标签（不越出 SVG 高度）
        view_h = float(re.search(r'viewBox="0 0 [\d.]+ ([\d.]+)"', svg).group(1))
        max_bottom = max(y0 + 11.0 * (n - 1) for _, y0, n in blocks)
        self.assertLessEqual(max_bottom + 4, view_h, "标签被推让后越出画布")

    @staticmethod
    def _label_blocks(svg: str):
        blocks = []
        for m in re.finditer(
                r'<text class="elabel[^"]*" x="([\d.-]+)" y="([\d.-]+)"[^>]*>'
                r"((?:<tspan[^>]*>[^<]*</tspan>)+)", svg):
            x = float(m.group(1))
            y0 = float(m.group(2))
            n = m.group(3).count("<tspan")
            blocks.append((x, y0, n))
        return blocks

    def test_cross_column_label_occlusion_replay(self):
        """自包含跨列遮挡回放：长跨列边标签原中点落在中间节点矩形内时，
        标签沿所属边滑移——锚点不再进入任何节点矩形，标签文本保留。"""
        phase = self.phase
        long_name = "跨列长张量名称甲跨列长张量名称甲超长超长"
        ops = [
            _op("colA", ["x_in"], ["m1", "far"]),
            _op("colB", ["m1"], ["m2"]),
            _op("colC", ["m2"], ["m3"]),
            _op("colD", ["far", "m3"], ["y"]),
        ]
        phase["operators"] += ops
        phase["tensors"] += [
            _tensor("m1", "activation", ["B", "T", "H"]),
            _tensor("m2", "activation", ["B", "T", "H"]),
            _tensor("m3", "activation", ["B", "T", "H"]),
            _tensor("far", "activation", ["B", "T", "H"]),
        ]
        for t in phase["tensors"]:
            if t["id"] == "far":
                t["name"] = long_name
        g = build_module_graph(phase, {"id": "occ", "title": "o", "ops": ops})
        svg = render_svg(g)
        # 节点矩形与标签锚点
        rects = [(float(a), float(b), float(c)) for a, b, c in re.findall(
            r'<g class="dg-node [^"]*" transform="translate\(([\d.-]+),([\d.-]+)\)">'
            r'<rect width="172" height="([\d.]+)"', svg)]
        self.assertGreaterEqual(len(rects), len(ops))
        anchors = [(float(a), float(b)) for a, b in re.findall(
            r'<text class="elabel[^"]*" x="([\d.-]+)" y="([\d.-]+)"', svg)]
        self.assertTrue(anchors)
        for lx, ly in anchors:
            for rx, ry, rh in rects:
                self.assertFalse(rx < lx < rx + 172 and ry < ly < ry + rh,
                                 f"标签锚点 ({lx},{ly}) 落入节点矩形内")
        self.assertIn("跨列长张量名称甲", svg, "滑移后标签文本必须保留（对应真实边）")

    def test_stacked_column_label_occlusion_replay(self):
        """自包含同列堆叠回放：边跨过中间状态节点时锚点滑到行间隙。"""
        phase = self.phase
        ops = [
            _op("st1", ["x_in"], ["s1", "s2"]),
            _op("st2", ["s1"], ["s2"]),
            _op("st3", ["s2"], ["y"]),
        ]
        phase["operators"] += ops
        phase["tensors"] += [
            _tensor("s1", "activation", ["B", "T", "H"]),
            _tensor("s2", "activation", ["B", "T", "H"]),
        ]
        g = build_module_graph(phase, {"id": "stk", "title": "s", "ops": ops})
        svg = render_svg(g)
        rects = [(float(a), float(b), float(c)) for a, b, c in re.findall(
            r'<g class="dg-node [^"]*" transform="translate\(([\d.-]+),([\d.-]+)\)">'
            r'<rect width="172" height="([\d.]+)"', svg)]
        for lx, ly in [(float(a), float(b)) for a, b in re.findall(
                r'<text class="elabel[^"]*" x="([\d.-]+)" y="([\d.-]+)"', svg)]:
            for rx, ry, rh in rects:
                self.assertFalse(rx < lx < rx + 172 and ry < ly < ry + rh,
                                 f"锚点 ({lx},{ly}) 落入节点矩形")

    def test_wide_cond_label_full_block_clear(self):
        """自包含「宽度超过默认间隙」回放：宽条件标签（如追加 N_win_i 条（可为 0））
        整块（含条件尾部）不与任何节点矩形相交；画布覆盖；节点标题仍在。"""
        phase = {
            "id": "prefill", "title": "w",
            "parameters": [
                {"id": "Nwin_i", "symbol": "N_win_i", "meaning": "windows",
                 "value": None, "expression": "floor(...)", "note": "", "source_refs": []},
            ],
            "tensors": [
                _tensor("st_pre", "cache", ["B", "M", "Id"]),
                _tensor("st_post", "cache", ["B", "M", "Id"]),
                _tensor("mid", "activation", ["N_e", "Es"]),
                _tensor("out", "activation", ["B", "T", "Id"]),
            ],
            "operators": [
                _op("i_pool_rope", ["st_pre"], ["mid"]),
                _op("i_update_states", ["mid"], ["st_post", "out"]),
                _op("i_scorer", ["out"], []),
            ],
            "cache_notes": ["x"], "coverage_notes": ["x"],
        }
        g = build_module_graph(phase, {"id": "wide", "title": "w",
                                       "ops": phase["operators"]})
        svg = render_svg(g)
        # 条件尾文完整存在（未截断/未隐藏）
        self.assertIn("追加 N_win_i 条（可为 0）", svg)
        rects = [(float(a), float(b), float(c)) for a, b, c in re.findall(
            r'<g class="dg-node [^"]*" transform="translate\(([\d.-]+),([\d.-]+)\)">'
            r'<rect width="172" height="([\d.]+)"', svg)]
        self.assertGreaterEqual(len(rects), 3)
        view_w = float(re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).group(1))
        view_h = float(re.search(r'viewBox="0 0 [\d.]+ ([\d.]+)"', svg).group(1))
        found = False
        for m in re.finditer(
                r'<text class="elabel[^"]*" x="([\d.-]+)" y="([\d.-]+)"[^>]*>'
                r"((?:<tspan[^>]*>[^<]*</tspan>)+)", svg):
            lines = [re.sub(r"<[^>]+>", "", seg)
                     for seg in re.findall(r"<tspan[^>]*>([^<]*)</tspan>", m.group(3))]
            if "追加" not in "".join(lines):
                continue
            found = True
            lx, ly = float(m.group(1)), float(m.group(2))
            n = len(lines)
            w = max(sum(10.0 if ord(ch) > 0x2E80 else 5.5 for ch in ln)
                    for ln in lines)
            left, right = lx - w / 2, lx + w / 2
            top, bot = ly - 8, ly + 11 * (n - 1) + 2
            for rx, ry, rh in rects:
                ox = min(right, rx + 172) - max(left, rx)
                oy = min(bot, ry + rh) - max(top, ry)
                self.assertLessEqual(ox * oy, 0,
                                     f"条件标签块与节点矩形相交 {ox}x{oy}")
            self.assertGreaterEqual(left, 0)
            self.assertLessEqual(right, view_w)
            self.assertLessEqual(bot + 4, view_h)
        self.assertTrue(found, "未找到条件标签")
        # 节点标题文字未被遮挡（节点在标签后绘制但标签不进矩形）
        self.assertIn("i_scorer", svg)

    def test_real_page_no_label_anchor_in_nodes(self):
        """真实文档全部 SVG：边标签锚点不落在任何节点矩形内
        （回放 diagram_gate round01 的 32 处遮挡）。"""
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.render.decoder_page import render_decoder_page

        doc = build_reading_document(REPO_ROOT)
        html_text = render_decoder_page(doc).decode("utf-8")
        svgs = re.findall(r"<svg class=\"dg\".*?</svg>", html_text, re.S)
        self.assertGreater(len(svgs), 0)
        bad = 0
        for svg in svgs:
            rects = [(float(a), float(b), float(c)) for a, b, c in re.findall(
                r'<g class="dg-node [^"]*" transform="translate\(([\d.-]+),([\d.-]+)\)">'
                r'<rect width="172" height="([\d.]+)"', svg)]
            for m in re.finditer(
                    r'<text class="elabel[^"]*" x="([\d.-]+)" y="([\d.-]+)"', svg):
                lx, ly = float(m.group(1)), float(m.group(2))
                for rx, ry, rh in rects:
                    if rx < lx < rx + 172 and ry < ly < ry + rh:
                        bad += 1
                        break
        self.assertEqual(bad, 0, f"{bad} 个标签锚点仍在节点矩形内")

    def test_real_page_multiline_blocks_gap(self):
        """真实文档全部 SVG：同列标签块按实际占用排布（回放 label_geometry 缺陷）。"""
        from structure_reader.extract.deepseek_v4_hf import build_reading_document

        doc = build_reading_document(REPO_ROOT)
        from structure_reader.render.decoder_page import render_decoder_page

        html_text = render_decoder_page(doc).decode("utf-8")
        svgs = re.findall(r"<svg class=\"dg\".*?</svg>", html_text, re.S)
        self.assertGreater(len(svgs), 0)
        worst = 999.0
        for svg in svgs:
            view_h = float(re.search(r'viewBox="0 0 [\d.]+ ([\d.]+)"', svg).group(1))
            by_x = {}
            for x, y0, n in self._label_blocks(svg):
                by_x.setdefault(x, []).append((y0, n))
                self.assertLessEqual(y0 + 11.0 * (n - 1) + 4, view_h,
                                     "标签越出画布高度")
            for items in by_x.values():
                items.sort()
                prev_bottom = None
                for y0, n in items:
                    bottom = y0 + 11.0 * (n - 1)
                    if prev_bottom is not None:
                        worst = min(worst, y0 - prev_bottom)
                    prev_bottom = bottom
        self.assertGreaterEqual(worst, 12.9,
                                f"真实页最坏标签块间距 {worst} < 13px")

    def test_node_height_contains_text(self):
        """长标签节点高度大于短标签节点（文字不被裁切/重叠）。"""
        phase = self.phase
        long_op = _op("opL", [], ["y"])
        long_op["name"] = "超长算子名称测试超长算子名称测试超长算子名称测试超长算子"
        short_op = _op("opS", ["y"], [])
        short_op["name"] = "短"
        phase["operators"] += [long_op, short_op]
        g = build_module_graph(phase, {"id": "h", "title": "h", "ops": [long_op, short_op]})
        svg = render_svg(g)
        hs = [float(h) for h in re.findall(r'<rect width="172" height="([\d.]+)"', svg)]
        self.assertEqual(len(hs), 2)
        self.assertGreater(max(hs), min(hs), "长标签节点应更高")
        self.assertGreaterEqual(min(hs), 44.0)

    def test_escape(self):
        phase = self.phase
        phase["operators"][0]["name"] = "a<b>&c"
        phase["tensors"][1]["name"] = "w&<x>"
        svg = render_svg(build_module_graph(phase, module_groups(phase, FIXTURE_RULES)[0]))
        self.assertNotIn("a<b>&c", svg)
        self.assertIn("a&lt;b&gt;&amp;c", svg)
        self.assertNotIn("w&<x>", svg)

    def test_deterministic(self):
        self.assertEqual(render_svg(self.graph), render_svg(self.graph))

    def test_legend_and_css(self):
        legend = legend_html()
        self.assertIn("数据流", legend)
        self.assertIn("权重", legend)
        self.assertIn("缓存", legend)
        css = diagram_css()
        for cls in (".dg-edge.flow", ".dg-edge.weight", ".dg-edge.cache", ".dg-edge.cond",
                    ".dg-node.boundary", ".dg-node.downstream", ".diagram-scroll"):
            self.assertIn(cls, css)


class TestRealDocumentClosure(unittest.TestCase):
    """真实文档（只读输入；缺失如实跳过）：图与文档闭合，回放 review facts 期望。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document

        cls.doc = build_reading_document(REPO_ROOT)

    def test_groups_cover_all_ops_and_edges_reference_real_ids(self):
        for phase in self.doc["phases"]:
            groups = module_groups(phase)
            covered = [o["id"] for g in groups for o in g["ops"]]
            self.assertEqual(
                sorted(covered), sorted(o["id"] for o in phase["operators"]),
                f"{phase['id']}: 分组必须覆盖全部算子恰一次")
            tids = {t["id"] for t in phase["tensors"]}
            oids = {o["id"] for o in phase["operators"]}
            for g in groups:
                graph = build_module_graph(phase, g)
                keys = [(e["src"], e["dst"], e["kind"], e["title"], e["cond"])
                        for e in graph["edges"]]
                self.assertEqual(len(keys), len(set(keys)), f"{g['id']}: 存在重复边")
                for e in graph["edges"]:
                    self.assertIn(e["title"], tids)
                for n in graph["nodes"]:
                    if n["id"].startswith("op:"):
                        self.assertIn(n["id"][3:], oids)
                    elif n["id"].startswith("t:"):
                        self.assertIn(n["id"][2:], tids)

    def test_cache_graphs_replay_review_facts(self):
        """回放 codex_diagram_review01/facts.json 期望：下游 chip 与缓存角色。"""
        for phase in self.doc["phases"]:
            graphs = {g["id"]: g for g in build_cache_graphs(phase)}
            self.assertEqual(set(graphs), {"main", "csa", "indexer"})
            expected_chips = {"main": {"kv_concat"},
                              "csa": {"kv_concat", "c_block_bias"},
                              "indexer": {"i_scorer", "i_topk"}}
            for fam, expected in expected_chips.items():
                chips = {e["dst"][2:] for e in graphs[fam]["edges"]
                         if e["dst"].startswith("d:")}
                self.assertEqual(chips, expected, f"{phase['id']}/{fam} 下游 chip 与 facts 不符")
            # 角色：kv_full 本次产出+被消费 → 本次读视图（非"调用前状态"）
            roles = {n["id"]: n["sub"].split(" · ")[0]
                     for g in graphs.values() for n in g["nodes"] if n["kind"] == "cache"}
            self.assertEqual(roles["t:kv_full"], "本次读视图")
            self.assertEqual(roles["t:cache_keys"], "调用前状态")
            self.assertEqual(roles["t:cache_keys_post"], "调用后回写")
            self.assertEqual(roles["t:cache_compressed_c_post"], "本次读视图·兼调用后保存")
            # 本次新输入与算子间 activation 路径存在（非纯 cache 类型边）
            for fam, port_tid, pair in (("main", "t:kv_rope", ("op:cache_update", "op:kv_concat")),
                                        ("csa", "t:c_kv", None),
                                        ("indexer", "t:i_kv", None)):
                edge_set = _edge_set(graphs[fam])
                self.assertIn((port_tid, f"op:{'cache_update' if fam == 'main' else
                                         ('c_store_window' if fam == 'csa' else 'i_store_window')}",
                               "flow"), edge_set, f"{fam} 缺本次新输入端口边")
            self.assertIn(("op:c_rope_gen", "op:c_rope", "flow"), _edge_set(graphs["csa"]),
                          "csa 缓存算子间 activation 链（c_rope_gen→c_rope）缺失")
            self.assertIn(("op:c_rope", "op:c_update_states", "flow"),
                          _edge_set(graphs["csa"]),
                          "csa 缓存算子间 activation 链（c_rope→c_update_states）缺失")

    def test_overview_reference_real_tensors(self):
        from collections import defaultdict

        for phase in self.doc["phases"]:
            ov = build_overview_graph(phase)
            producers = defaultdict(list)
            consumers = defaultdict(list)
            for i, op in enumerate(phase["operators"]):
                for tid in op["outputs"]:
                    producers[tid].append(i)
                for tid in op["inputs"]:
                    consumers[tid].append(i)
            for e in ov["edges"]:
                self.assertTrue(e["title"], "总览边需可追溯张量清单")
                for tid in e["title"].split("，"):
                    self.assertIn(tid, {t["id"] for t in phase["tensors"]})

    def test_cache_condition_labels_both_series(self):
        for phase in self.doc["phases"]:
            graphs = {g["id"]: g for g in build_cache_graphs(phase)}
            conds = [e["cond"] for g in graphs.values() for e in g["edges"] if e["cond"]]
            self.assertTrue(any(c == "仅 N_win>0" for c in conds))
            self.assertTrue(any(c == "仅 N_win_i>0" for c in conds))


if __name__ == "__main__":
    unittest.main(verbosity=2)
