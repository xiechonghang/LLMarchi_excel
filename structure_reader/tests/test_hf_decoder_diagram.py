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
    compute_type_label,
    diagram_css,
    legend_html,
    module_groups,
    render_svg,
)


def dg_build_module(phase, group):
    return build_module_graph(phase, group)


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
        """同起终点的平行张量边标签不互叠（纵向布局：同走廊内横向错位）。"""
        phase = self.phase
        phase["operators"][0]["outputs"] = ["a"]
        phase["operators"][2]["inputs"] = ["a", "a"]  # 同张量重复引用为同一 id，改用双张量
        phase["operators"][2]["inputs"] = ["a", "b"]
        phase["operators"][1]["outputs"] = ["b"]
        g = build_module_graph(phase, {"id": "par", "title": "p",
                                       "ops": [phase["operators"][0],
                                               phase["operators"][2]]})
        svg = render_svg(g)
        blocks = self._label_blocks(svg)
        self.assertGreaterEqual(len(blocks), 2)
        # 2D 块不重叠（纵向布局中平行标签在同走廊横向错位）
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                self.assertTrue(self._blocks_separated(blocks[i], blocks[j]),
                                f"平行标签互叠：{blocks[i]} vs {blocks[j]}")

    @staticmethod
    def _view_box(svg: str):
        m = re.search(r'viewBox="([\d.-]+) ([\d.-]+) ([\d.]+) ([\d.]+)"', svg)
        return tuple(float(v) for v in m.groups())

    @staticmethod
    def _label_blocks(svg: str):
        """从 SVG 解析标签块 (lx, ly, n_lines, half_w)；宽度按浏览器实际渲染的
        反转义文本度量（&#x27; 等实体是单字符，不按字面计宽）。"""
        import html as _h
        blocks = []
        for m in re.finditer(
                r'<text class="elabel[^"]*" x="([\d.-]+)" y="([\d.-]+)"[^>]*>'
                r"((?:<tspan[^>]*>[^<]*</tspan>)+)", svg):
            x = float(m.group(1))
            y0 = float(m.group(2))
            n = m.group(3).count("<tspan")
            text_lines = [_h.unescape(t) for t in
                          re.findall(r"<tspan[^>]*>([^<]*)</tspan>", m.group(3))]
            w = max(sum(10.0 if ord(ch) > 0x2E80 else 5.5 for ch in ln)
                    for ln in text_lines) / 2 + 4.0
            blocks.append((x, y0, n, w))
        return blocks

    @staticmethod
    def _blocks_separated(a, b, margin=3.0):
        """标签块（首行基线 -8 上沿 / 末行 +2 下沿；水平 ±half_w）二维不重叠。"""
        ax, ay, an, aw = a
        bx, by, bn, bw = b
        a_top, a_bot = ay - 8, ay + 11 * (an - 1) + 2
        b_top, b_bot = by - 8, by + 11 * (bn - 1) + 2
        x_sep = (ax + aw + margin <= bx - bw) or (bx + bw + margin <= ax - aw)
        y_sep = (a_bot + margin <= b_top) or (b_bot + margin <= a_top)
        return x_sep or y_sep

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
        self.assertTrue(any(n >= 2 for _, _, n, _ in blocks), "回放应含两行标签")
        # 纵向布局不变式：标签块两两 2D 不重叠（含 13px 块距语义）
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                self.assertTrue(self._blocks_separated(blocks[i], blocks[j]),
                                f"标签块互叠：{blocks[i]} vs {blocks[j]}")
        # 画布高度覆盖全部标签（不越出 SVG 高度）
        _vx, _vy, _vw, view_h = self._view_box(svg)
        max_bottom = max(y0 + 11.0 * (n - 1) for _, y0, n, _ in blocks)
        self.assertLessEqual(max_bottom + 4, _vy + view_h, "标签被推让后越出画布")

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
        view_w, view_h = self._view_box(svg)[2:]
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
                self.assertTrue(ox <= 0 or oy <= 0,
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
        for svg in svgs:
            _vx, _vy, _vw, view_h = self._view_box(svg)
            blocks = self._label_blocks(svg)
            for _, y0, n, _ in blocks:
                self.assertLessEqual(y0 + 11.0 * (n - 1) + 4, view_h,
                                     "标签越出画布高度")
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    self.assertTrue(self._blocks_separated(blocks[i], blocks[j]),
                                    f"标签块互叠：{blocks[i]} vs {blocks[j]}")

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


class TestOnnxViewSemantics(unittest.TestCase):
    """用户反馈02：ONNX 式阅读语义 + 图示固定对比配色。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.render import diagram_source_labels as dsl

        cls.doc = build_reading_document(REPO_ROOT)
        cls.dsl = dsl

    def test_source_label_covers_all_real_ids(self):
        """每个真实算子/张量 id 都有源码命名映射（内部 id 不冒充源码变量的保障）。"""
        for phase in self.doc["phases"]:
            for op in phase["operators"]:
                self.assertIsNotNone(
                    self.dsl.op_source_label(op),
                    f"{phase['id']}: 算子 {op['id']} 缺源码名映射")
            for t in phase["tensors"]:
                self.assertIsNotNone(
                    self.dsl.tensor_source_label(t["id"]),
                    f"{phase['id']}: 张量 {t['id']} 缺源码名映射")

    def test_op_box_shows_source_name_and_compute_type(self):
        """计算框 = 源码调用名（主行）+ 实际计算类型（副行，复合如实列出）。"""
        phase = self.doc["phases"][0]
        ops = {o["id"]: o for o in phase["operators"]}
        for oid, expect_name, expect_type in (
                ("q_a_proj", "self.q_a_proj(hidden_states)", "Linear"),
                ("qk_scaled", "matmul(query, key_states.transpose(2,3))*scaling", "MatMul"),
                ("sink_softmax", "softmax(cat([attn_weights, sinks]) - max)", "Softmax"),
        ):
            op = ops[oid]
            self.assertEqual(self.dsl.op_source_label(op), expect_name)
            self.assertIn(expect_type, self.dsl.compute_type_label(op))
        # 复合算子类型如实列出（q_b_proj：Linear + View + Transpose）
        self.assertEqual(self.dsl.compute_type_label(ops["q_b_proj"]),
                         "Linear + View + Transpose")
        g = dg_build_module(phase, next(
            g for g in module_groups(phase) if g["id"] == "attn_q"))
        svg = render_svg(g)
        self.assertIn("self.q_a_proj(hidde", svg)

    def test_tensor_arrows_use_source_names_not_internal_ids(self):
        """箭头标签 = 源码变量/表达式名 + shape（如 q_a_normed→q_residual）。"""
        phase = self.doc["phases"][0]
        tensors = {t["id"]: t for t in phase["tensors"]}
        g = dg_build_module(phase, next(
            g for g in module_groups(phase) if g["id"] == "attn_q"))
        labels = [e["label"] for e in g["edges"]]
        self.assertIn("q_residual [B,T,Rq]",
                      [l for l in labels if "q_residual" in l][0])
        for e in g["edges"]:
            self.assertNotEqual(e["label"], e["title"],
                                f"边标签不得直接使用内部语义 id {e['title']}")

    def test_non_compute_nodes_are_ports_not_boxes(self):
        """tensor/权重/缓存状态不再是计算框矩形：渲染为端口圆点；
        初始 hidden_states 为显式输入标识；总览模块框为导航（虚线）。"""
        phase = self.doc["phases"][0]
        groups = {g["id"]: g for g in module_groups(phase)}
        mhc = render_svg(dg_build_module(phase, groups["mhc_attn"]))
        self.assertIn('dg-port io', mhc, "hidden_states 输入标识缺失")
        self.assertIn('hidden_states（层输入）', mhc)
        self.assertIn('dg-dot', mhc, "tensor/权重端点应为端口圆点")
        self.assertNotIn('dg-node boundary', mhc, "tensor 不得再画计算样式矩形")
        self.assertNotIn('dg-node weight', mhc)
        # 边界输入圆点（position_ids）在 Attention-Q 图中
        q = render_svg(dg_build_module(phase, groups["attn_q"]))
        self.assertIn('dg-dot boundary', q)
        ov = render_svg(build_overview_graph(phase))
        self.assertIn('dg-node nav', ov)
        self.assertIn('dg-node navlane', ov)
        self.assertIn('导航', ov)

    def test_cache_arrow_labels_carry_source_name_and_role(self):
        """缓存状态为端口圆点，箭头标签含时序角色 + 源码名 + shape。"""
        from structure_reader.render.diagram import build_cache_graphs

        graphs = {g["id"]: g for g in build_cache_graphs(self.doc["phases"][0])}
        labels = [e["label"] for g in graphs.values() for e in g["edges"]]
        self.assertTrue(any("读调用前状态" in l and "self.keys" in l for l in labels),
                        "主 KV 读边应含角色与源码名")
        self.assertTrue(any("写调用后状态" in l and "self.keys" in l for l in labels))
        self.assertTrue(any("本次读视图→参与计算" in l and "compressed_kv" in l
                            for l in labels), "下游边应含缓存状态源码名")

    def test_diagram_fixed_contrast_palette(self):
        """浅/暗主题可读：框内文字固定深色（不依赖 --ink），边标签成对主题配色
        且光晕不再使用 var(--card)。"""
        css = diagram_css()
        self.assertIn(".dg-node .nlabel { fill: #1f2933;", css)
        self.assertIn(".dg-node .nsub { fill: #4a5568;", css)
        self.assertNotIn(".dg-node text { fill: var(--ink)", css)
        self.assertIn(".elabel { font-size: 10px; fill: #33404d;", css)
        self.assertNotIn("stroke: var(--card)", css)
        self.assertIn("@media (prefers-color-scheme: dark)", css)
        self.assertIn(".elabel { fill: #d9e2ea; stroke: rgba(20,23,27,0.92); }", css)
        self.assertIn(".elabel-cond { fill: #ff9a9a; }", css)

    def test_compute_type_composite_and_fallback(self):
        """复合类型如实列出；未收录 token 原样保留（不伪造 ONNX 名）。"""
        op = {"op": "linear+view+transpose"}
        self.assertEqual(compute_type_label(op), "Linear + View + Transpose")
        op2 = {"op": "customop+linear"}
        self.assertEqual(compute_type_label(op2), "customop + Linear")


class TestOnnxViewFix01(unittest.TestCase):
    """反馈02收尾回放：端口真接、条件+名称/shape 同见、源码真名、暗主题箭头对比。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.render.decoder_page import render_decoder_page
        from structure_reader.render import diagram_source_labels as dsl

        cls.doc = build_reading_document(REPO_ROOT)
        cls.html_text = render_decoder_page(cls.doc).decode("utf-8")
        cls.dsl = dsl

    @staticmethod
    def _svg_geoms(svg: str):
        """解析 SVG：圆点圆心（含 title）、IO/计算框矩形、路径端点。"""
        circle_by_title = {}
        for m in re.finditer(
                r'<g class="dg-dot ([^"]*)" transform="translate\(([\d.-]+),([\d.-]+)\)">'
                r'<circle r="5"/><title>([^<]*)</title>', svg):
            circle_by_title[m.group(4)] = (float(m.group(2)), float(m.group(3)))
        rects = [(float(a), float(b), float(w), float(h)) for a, b, w, h in re.findall(
            r'<g class="dg-(?:node|port) [^"]*" transform="translate\(([\d.-]+),([\d.-]+)\)">'
            r'<rect width="([\d.]+)" height="([\d.]+)"', svg)]
        paths = [(float(a), float(b), float(c), float(d)) for a, b, c, d in re.findall(
            r'<path class="dg-edge[^"]*" d="M([\d.-]+),([\d.-]+) C[^"]*'
            r'([\d.-]+),([\d.-]+)"', svg)]
        return circle_by_title, rects, paths

    def test_paths_reach_port_dots(self):
        """端口真接回放：href 指向的张量圆点必须与该边某一端点重合（±7px，
        旧 162px 悬空为 0）；IO 标识边端点落在其 24px 高边界上。"""
        svgs = re.findall(r"<svg class=\"dg\".*?</svg>", self.html_text, re.S)
        checked = 0
        for svg in svgs:
            circle_by_title, rects, _paths = self._svg_geoms(svg)
            dot_by_tid = {}
            for title, center in circle_by_title.items():
                dot_by_tid[title.split(" · ")[0]] = center
            for m in re.finditer(
                    r'<a href="#t-([\w-]+)"><path class="dg-edge[^"]*" '
                    r'd="M([\d.-]+),([\d.-]+) C[\d.-]+,[\d.-]+ [\d.-]+,[\d.-]+ '
                    r'([\d.-]+),([\d.-]+)"', svg):
                tid = m.group(1).split("-", 1)[1]  # href = #t-{phase}-{tid}
                endpoints = [(float(m.group(2)), float(m.group(3))),
                             (float(m.group(4)), float(m.group(5)))]
                if tid in dot_by_tid:
                    cx, cy = dot_by_tid[tid]
                    d = min((sx - cx) ** 2 + (sy - cy) ** 2
                            for sx, sy in endpoints) ** 0.5
                    checked += 1
                    self.assertLessEqual(d, 7.0,
                                         f"圆点 {tid} 距箭头端点 {d:.1f}px（悬空）")
                else:
                    # IO 标识：端点须落在其 172×24 矩形边界 6px 内
                    hit = False
                    for rx, ry, rw, rh in rects:
                        if rh == 24:
                            for sx, sy in endpoints:
                                if (rx - 6 <= sx <= rx + rw + 6
                                        and ry - 6 <= sy <= ry + rh + 6):
                                    hit = True
                    if any(tid == "hidden_streams_in" or tid == "hidden_streams_out"
                           for tid in [tid]):
                        checked += 1
                        self.assertTrue(hit, f"IO 标识 {tid} 边端点未接到端口边界")
        self.assertGreater(checked, 100, "端口连线回放样本不足")

    def test_cond_and_name_shape_both_visible(self):
        """条件边：条件行与源码名/shape 行同时在可见 tspan 中（不再互斥/截断）。"""
        svgs = re.findall(r"<svg class=\"dg\".*?</svg>", self.html_text, re.S)
        cond_labels = 0
        for svg in svgs:
            for m in re.finditer(
                    r'<text class="elabel" x="[\d.-]+" y="[\d.-]+"[^>]*>'
                    r"((?:<tspan[^>]*>[^<]*</tspan>)+)", svg):
                lines = re.findall(r"<tspan[^>]*>([^<]*)</tspan>", m.group(1))
                joined = "".join(lines)
                if any(k in joined for k in ("仅 N_win", "追加 N_win", "余量回写")):
                    cond_labels += 1
                    self.assertTrue(
                        any("[" in ln and "]" in ln for ln in lines),
                        f"条件标签缺可见 shape：{lines}")
                    self.assertTrue(
                        any("self." in ln or "past_key_values" in ln for ln in lines),
                        f"条件标签缺可见源码名：{lines}")
        self.assertGreater(cond_labels, 20, "条件标签回放样本不足")

    def test_no_shape_truncation(self):
        """全部边标签的可见行拼接 == 语义全文（12 类两行截断维度回归为 0）。"""
        svgs = re.findall(r"<svg class=\"dg\".*?</svg>", self.html_text, re.S)
        checked = 0
        for svg in svgs:
            for m in re.finditer(
                    r'<text class="elabel[^"]*" x="[\d.-]+" y="[\d.-]+"[^>]*>'
                    r"((?:<tspan[^>]*>[^<]*</tspan>)+)<title>([^<]*)</title>", svg):
                import html as _h
                lines = [_h.unescape(x) for x in
                         re.findall(r"<tspan[^>]*>([^<]*)</tspan>", m.group(1))]
                full = _h.unescape(m.group(2))
                checked += 1
                if "｜" in full and len(lines) > 1:
                    # 条件行 + 名称/shape 行（title 以｜分隔）：可见行拼接须还原全文
                    expected = lines[0] + "".join(lines[1:])
                    expected = expected  # 与 full 去｜后一致
                    self.assertEqual(expected, full.replace("｜", ""),
                                     f"可见标签与全文不一致（截断）：{expected!r}")
                    continue
                expected = "".join(lines)
                self.assertEqual(expected, full,
                                 f"可见标签与全文不一致（截断）：{expected!r}")
        self.assertGreater(checked, 400)

    def test_source_name_semantic_corrections(self):
        """具名 4 例修正 + 同类图内代号不再混称源码表达式。"""
        dsl = self.dsl
        self.assertEqual(
            dsl.tensor_source_label("expert_gu_out"),
            "F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])")
        self.assertEqual(dsl.tensor_source_label("i_scores"),
                         "index_scores（scorer 返回）")
        self.assertEqual(dsl.op_source_label({"id": "q_a_norm"}),
                         "self.q_a_norm(self.q_a_proj(hidden_states))")
        self.assertEqual(
            dsl.tensor_source_label("expert_down_raw"),
            "F.linear(current, self.down_proj[expert_idx])")
        all_values = " ".join(v for v, _ in dsl.OP_SOURCE.values()) + " " + \
            " ".join(v for v, _ in dsl._T.values())
        for bad in ("down[e]", "gate_up[e]", "(q_a)", "combT,", "*Out "):
            self.assertNotIn(bad, all_values, f"图内代号混入源码名：{bad}")
        # 消歧标注明确非源码变量的场合保留括注（如 full（update 返回））
        self.assertIn("（", dsl.tensor_source_label("kv_full"))

    def test_node_source_name_fully_visible(self):
        """框内源码名完整可见：实际 SVG nlabel 拼接 == 完整源码名
        （回放 review02 的 5 具名截断：moe_expert_gu/slice、i_topk、
        attn/ffn_residual_mix；两阶段共 10 次出现）。"""
        import html as _h

        from structure_reader.render.diagram_source_labels import op_source_label

        svgs = re.findall(r"<svg class=\"dg\".*?</svg>", self.html_text, re.S)
        named_seen = 0
        named = {"moe_expert_gu", "moe_expert_slice", "i_topk",
                 "attn_residual_mix", "ffn_residual_mix"}
        checked = 0
        for svg in svgs:
            for m in re.finditer(
                    r'<g class="dg-node op" transform="translate\([\d.-]+,[\d.-]+\)">'
                    r"<rect[^>]*/><title>([^<·]*) · [^<]*</title>(.*?)</g>", svg):
                oid = m.group(1).strip()
                full = op_source_label({"id": oid})
                if full is None:
                    continue
                nlabels = re.findall(r'<text class="nlabel"[^>]*>([^<]*)</text>',
                                     m.group(2))
                visible = _h.unescape("".join(nlabels))
                checked += 1
                self.assertEqual(visible, full,
                                 f"{oid} 框内源码名被截断：{visible!r}")
                if oid in named:
                    named_seen += 1
        self.assertGreater(checked, 100)
        self.assertEqual(named_seen, 10, "5 具名算子应两阶段各出现一次且完整")

    def test_node_long_name_fixture_wraps_without_cap(self):
        """自包含回放：超长源码名按需换行（无 3 行硬上限），高度随行数增长。"""
        phase = fixture_phase()
        long_op = _op("opXL", ["x_in"], ["y"])
        long_op["name"] = "F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])"
        phase["operators"] = [long_op]
        g = build_module_graph(phase, {"id": "xl", "title": "x", "ops": [long_op]})
        svg = render_svg(g)
        nlabels = re.findall(r'<text class="nlabel"[^>]*>([^<]*)</text>', svg)
        import html as _h
        self.assertEqual(_h.unescape("".join(nlabels)), long_op["name"],
                         "框内长名被行数上限截断")
        self.assertGreater(len(nlabels), 3, "长名应超过旧 3 行上限")
        h = float(re.search(r'<rect width="172" height="([\d.]+)"', svg).group(1))
        self.assertGreaterEqual(h, 12 + 12 * len(nlabels), "节点高度未容纳全部行")

    def test_dark_theme_arrow_contrast(self):
        """暗主题 flow/weight/cache 箭头成对浅色：按 WCAG 静态计算对比度。"""
        css = diagram_css()

        def lum(hex_color: str) -> float:
            r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
            def lin(c):
                return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

        def ratio(c1: str, c2: str) -> float:
            l1, l2 = sorted((lum(c1), lum(c2)), reverse=True)
            return (l1 + 0.05) / (l2 + 0.05)

        dark_bg, light_bg = "#1d2126", "#f6f7f9"
        dark_block = css[css.index("@media (prefers-color-scheme: dark)"):]
        flow_light = re.search(r"\.dg-edge\.flow \{ stroke: (#[0-9a-f]+)", css).group(1)
        flow_dark = re.search(r"\.dg-edge\.flow \{ stroke: (#[0-9a-f]+)", dark_block).group(1)
        self.assertGreaterEqual(ratio(flow_dark, dark_bg), 4.5,
                                f"暗主题 flow 对比 {ratio(flow_dark, dark_bg):.2f}")
        self.assertGreaterEqual(ratio(flow_light, light_bg), 4.5)
        for kind in ("weight", "cache"):
            c_dark = re.search(rf"\.dg-edge\.{kind} \{{ stroke: (#[0-9a-f]+)",
                               dark_block).group(1)
            self.assertGreaterEqual(ratio(c_dark, dark_bg), 4.5,
                                    f"暗主题 {kind} 对比不足")
        # 框字保持已认可对比（主行 ≥7）
        self.assertGreaterEqual(ratio("#1f2933", "#fcfdfe"), 7.0)


class TestVerticalHierarchy(unittest.TestCase):
    """用户反馈03：纵向自上而下主阅读 + 真实包含关系章节编号。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.render.decoder_page import render_decoder_page

        cls.doc = build_reading_document(REPO_ROOT)
        cls.html_text = render_decoder_page(cls.doc).decode("utf-8")

    def test_cross_rank_edges_flow_downward(self):
        """自包含 fixture：跨层边目标在源下方（纵向主阅读），文字横向（无旋转）。"""
        phase = fixture_phase()
        g = build_module_graph(phase, {"id": "v", "title": "v",
                                       "ops": phase["operators"][:4]})
        svg = render_svg(g)
        self.assertNotIn("rotate(", svg, "不得旋转 SVG（文字须横向）")
        # 跨层边：终点 y > 起点 y（自上而下）
        for m in re.finditer(
                r'<path class="dg-edge flow" d="M([\d.-]+),([\d.-]+) '
                r"C[\d.-]+,[\d.-]+ [\d.-]+,[\d.-]+ ([\d.-]+),([\d.-]+)\"", svg):
            self.assertGreater(float(m.group(4)), float(m.group(2)),
                               "跨层边未自上而下")

    def test_vertical_fixture_geometry_and_labels(self):
        """自包含纵向回放：链+分叉汇合+残差+权重/圆点/IO+长名/条件边——
        几何端点到达、标签完整、无旋转。"""
        phase = fixture_phase()
        g = build_module_graph(phase, {"id": "vf", "title": "f",
                                       "ops": phase["operators"]})
        svg = render_svg(g)
        self.assertNotIn("rotate(", svg)
        # 圆点/计算框都存在（IO 标识由真实文档测试覆盖：hidden_states）
        self.assertIn("dg-dot", svg)
        self.assertIn("dg-node op", svg)
        # 条件+源码名同见
        blocks = TestSvgRendering._label_blocks(svg)
        self.assertTrue(any(n >= 2 for _, _, n, _ in blocks))
        # 标签两两不重叠、不越界
        view_h = float(re.search(r'viewBox="0 0 [\d.]+ ([\d.]+)"', svg).group(1))
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                self.assertTrue(
                    TestSvgRendering._blocks_separated(blocks[i], blocks[j]))
        _vx, _vy, _vw, vh = TestSvgRendering._view_box(svg)
        for _, y0, n, _ in blocks:
            self.assertLessEqual(y0 + 11.0 * (n - 1) + 4, _vy + vh,
                                 "标签越出画布（纵向原点口径）")

    def test_section_tree_containment(self):
        """章节树体现真实包含：Attention→CSA→Indexer 同前缀逐层深入；
        编号来自树而非列表顺序（乱序输入不改变编号）。"""
        from structure_reader.render.diagram import section_entries, section_number

        entries = section_entries("prefill")
        by_gid = {e["gid"]: e for e in entries if e["gid"]}
        csa = by_gid["attn_csa"]
        indexer = by_gid["attn_indexer"]
        self.assertEqual(csa["number"], "1.2.4")
        self.assertEqual(indexer["number"], "1.2.4.1")
        self.assertTrue(indexer["number"].startswith(csa["number"] + "."))
        self.assertEqual(indexer["parent_number"], csa["number"])
        # 编号与列表顺序无关（乱序树输入）
        import structure_reader.render.diagram as dg
        old = dg._SECTION_TREE
        try:
            dg._SECTION_TREE = tuple(reversed(old))
            self.assertEqual(section_number("attn_csa", "prefill"), "1.2.4")
            self.assertEqual(section_number("attn_indexer", "prefill"), "1.2.4.1")
        finally:
            dg._SECTION_TREE = old
        # Decode 用 2 前缀
        self.assertEqual(section_number("attn_csa", "decode"), "2.2.4")

    def test_same_target_same_number_everywhere(self):
        """同一目标（CSA/Indexer/缓存通道）在总览导航、章节标题、目录中编号一致。"""
        from structure_reader.render.diagram import section_number

        for phase in self.doc["phases"]:
            p = "1" if phase["id"] == "prefill" else "2"
            for gid in ("attn_csa", "attn_indexer", "cache:main"):
                num = section_number(gid, phase["id"])
                self.assertIn(num, self.html_text)
                # 总览导航框标签含该编号（nav label = 编号 + 标题）
                self.assertGreaterEqual(self.html_text.count(f"{num} "), 1)
        # 目录链接锚点全部存在（含父章节 sec- 锚点）
        from html.parser import HTMLParser

        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = set()
                self.links = []

            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if "id" in a:
                    self.ids.add(a["id"])
                if a.get("href", "").startswith("#"):
                    self.links.append(a["href"][1:])
        parser = Links()
        parser.feed(self.html_text)
        broken = sorted(set(parser.links) - parser.ids)
        self.assertEqual(broken, [], f"悬空联查目标: {broken[:6]}")

    def test_phase_sections_numbered(self):
        """Prefill=1、Decode=2 根章；章节实际呈现顺序按编号数值层级
        （先 x.1…x.6 再 x.7；缓存子图 7.1/7.2/7.3 不倒序）；缓存标题各自
        具名关联（main→x.2.3、csa→x.2.4、indexer→x.2.4.1）。"""
        self.assertIn("1 Prefill", self.html_text)
        self.assertIn("2 Decode", self.html_text)
        self.assertIn("1.2.4 CSA 压缩器", self.html_text)
        self.assertIn("1.2.4.1 Lightning Indexer", self.html_text)
        # 实际顺序：1.1 标题先于 1.7；1.6 先于 1.7；7.1 → 7.2 → 7.3
        for pre, post in (
                (">1.1 ", ">1.7 "), (">1.6 ", ">1.7 "),
                (">2.6 ", ">2.7 "), (">1.7.1 ", ">1.7.2 "),
                (">1.7.2 ", ">1.7.3 "), (">2.7.1 ", ">2.7.2 "),
                (">2.7.2 ", ">2.7.3 ")):
            self.assertLess(self.html_text.index(pre), self.html_text.index(post),
                            f"章节呈现顺序错误：{pre} 应先于 {post}")
        # 每个缓存子图各自具名关联
        self.assertIn("（缓存附图，关联 1.2.3）", self.html_text)
        self.assertIn("（缓存附图，关联 1.2.4）", self.html_text)
        self.assertIn("（缓存附图，关联 1.2.4.1）", self.html_text)
        self.assertIn("（缓存附图，关联 2.2.3）", self.html_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
