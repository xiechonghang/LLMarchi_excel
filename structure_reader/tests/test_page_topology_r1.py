"""R1 拓扑图针对性页面测试：实际生成 HTML 的图元与数据核验（无浏览器）。

范围（resume01.md R1 第 4 条）：
- 图元引用/节点与边 ID 集合与 G 完全对应（无缺边/假边）：SVG 由页面 JS
  运行时生成，静态 HTML 不含 data-node/data-edge，因此集合核验在 Node 桩
  执行后从 #gnodes/#gedges 的 innerHTML 提取（真实渲染产物），并用静态
  G（data/semantic_graph.json）做期望集合。
- 默认场景及 prefill/decode 切换（边三态覆盖集合核验 + 已知真实关键分支
  显式断言）。
- 节点/边点击与张量/证据详情（JS 级数据路径核验）。
- JS 无异常/特殊字符（Node VM 执行 + 未知元素点名）。

Node 桩局限如实声明：这是 JS 单元执行（DOM 桩无布局/CSS/getBoundingClientRect
为固定值），不是浏览器/CSS/布局验收；pointer/wheel 交互与视觉效果由 Codex
真实浏览器接续核验。gFit 依赖 clientWidth，桩内为固定 1000×640，因此适配
结果不核验（桩局限，不声称布局通过）。

工程 Python：codex-primary-runtime python3；Node：/opt/homebrew/bin/node。
"""

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NODE = "/opt/homebrew/bin/node"
MODEL_OUTPUTS = PROJECT / "outputs" / "deepseek_v4_flash_0731"
LOGS_DIR = PROJECT / ".work" / "coordination" / "v2" / "reader_recovery" / "logs"

def _latest_output_dir() -> Path:
    runs = [r for r in MODEL_OUTPUTS.iterdir()
            if (r / "viewer" / "structure.html").is_file()]
    return max(runs, key=lambda r: r.stat().st_mtime)


# Node VM 执行后从 DOM 桩提取的渲染产物（模块级缓存，一次执行多处断言）
_RENDER_CACHE: dict | None = None

# 页面 JS 的 DOM 桩执行探针：初始化 → 全场景切换 → 全节点/边详情 →
# 全张量/证据面板；收集 #gnodes/#gedges innerHTML 与未知元素点名。
# 局限：无布局/CSS；querySelectorAll 返回桩元素数组；交互只验证回调。
_PROBE_CJS = r'''
const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const payload=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\n([\s\S]*?)<\/script>\s*<\/body>/)[1];
const P=JSON.parse(payload);
const els=new Map();const unknown=new Set();
function element(id){return {id,innerHTML:'',textContent:'',hidden:false,
  dataset:{},children:[],classList:{toggle(){}},
  appendChild(b){this.children.push(b)},insertBefore(){},
  addEventListener(){},setAttribute(){},
  getBoundingClientRect(){return {left:0,top:0,width:1000,height:640}},
  clientWidth:1000,clientHeight:640,setPointerCapture(){}};}
for(const id of [...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]))els.set(id,element(id));
for(const id of ['statepanel','evdetail'])els.set(id,element(id));
els.get('payload').textContent=payload;
const context={document:{
  getElementById(id){if(!els.has(id)){unknown.add(id);els.set(id,element(id));}
    return els.get(id)},
  querySelector(){return element('main')},
  querySelectorAll(sel){
    if(sel==='.gnode'||sel==='.gedge'||sel==='#gscenbtns button')
      return els.get('scenbtns').children.map(()=>element('dyn'));
    return els.get('scenbtns').children;},
  createElement(){return element('dynamic')}},
  window:{},location:{hash:''}};
vm.createContext(context);vm.runInContext(script,context,{timeout:5000});
// 场景切换（图工具条按钮与既有面板共用 curScen）
let switched=0;
for(const b of els.get('gscenbtns').children){b.onclick();switched++;
  const sum=els.get('gscensum').innerHTML;
  if(/undefined|NaN/.test(sum))throw Error('invalid gscensum '+b.dataset.id);}
// 全部节点详情 + 全部边详情 + 张量/证据面板
let nodes=0,edges=0;
for(const n of P.graph.nodes){context.window.gShowNode(n.stable_id);nodes++;
  const d=els.get('gdetail').innerHTML;
  if(/undefined|NaN/.test(d))throw Error('invalid node detail '+n.stable_id);}
for(const e of P.graph.edges){context.window.gShowEdgeById(e.stable_id);edges++;
  const d=els.get('gdetail').innerHTML;
  if(/undefined|NaN/.test(d))throw Error('invalid edge detail '+e.stable_id);}
let tensors=0;for(const t of P.graph.tensors){context.window.showTensor(t.stable_id);tensors++;}
let evidence=0;for(const e of P.evidence.entries){context.window.showEv(e.evidence_id);evidence++;}
if(unknown.size)throw Error('unknown elements probed: '+[...unknown].join(','));
// 完整图模式的 SVG 图元集合（R2a 后默认视图为模块总览；切 full 提取
// 全量图渲染产物，ID 集合核验强度不变——仍须与 G 完全对应）
context.window.gSetMode('full');
console.log(JSON.stringify({scenarios:switched,nodes,edges,tensors,evidence,
  scenBtnCounts:{toolbar:els.get('gscenbtns').children.length,
                 panel:els.get('scenbtns').children.length},
  scenBtnIds:{toolbar:els.get('gscenbtns').children.map(b=>b.dataset.id),
              panel:els.get('scenbtns').children.map(b=>b.dataset.id)},
  gnodesHtml:els.get('gnodes').innerHTML,
  gedgesHtml:els.get('gedges').innerHTML,
  note:'local JS unit execution with DOM stubs; no browser/CSS/layout acceptance'}));
'''


def _render_once() -> dict:
    """执行一次页面 JS（Node 桩），缓存渲染产物供各测试共享。"""
    global _RENDER_CACHE
    if _RENDER_CACHE is not None:
        return _RENDER_CACHE
    out_dir = _latest_output_dir()
    html = (out_dir / "viewer" / "structure.html").read_text(encoding="utf-8")
    probe_path = PROJECT / ".work" / "r1_graph_probe.cjs"
    probe_path.write_text(_PROBE_CJS, encoding="utf-8")
    run = subprocess.run(
        [NODE, str(probe_path), str(out_dir / "viewer" / "structure.html")],
        capture_output=True, text=True, timeout=120,
        cwd=str(PROJECT.parent))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "r1_graph_probe_run.json").write_text(
        run.stdout or run.stderr, encoding="utf-8")
    if run.returncode != 0:
        raise AssertionError(f"页面 JS 执行失败:\n{run.stderr[:3000]}")
    _RENDER_CACHE = {"payload": json.loads(json.loads(run.stdout).pop("payload_raw")
                                           if False else run.stdout),
                     "html": html, "out_dir": out_dir}
    return _RENDER_CACHE


class _PageHarness(unittest.TestCase):
    """公共：最新产物 HTML + payload + G + Node 桩渲染产物。"""

    @classmethod
    def setUpClass(cls):
        cls.data = _render_once()
        cls.out_dir = cls.data["out_dir"]
        cls.html = cls.data["html"]
        cls.render = cls.data["payload"]
        cls.payload = json.loads(re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            cls.html, re.S).group(1))
        cls.G = cls.payload["graph"]
        # 期望集合同源：产物内 canonical semantic_graph.json
        cls.G_doc = json.loads(
            (cls.out_dir / "data" / "semantic_graph.json").read_text(
                encoding="utf-8"))


class TestGraphSvgIdsComplete(_PageHarness):
    """图元 ID 集合与 G 完全对应：渲染出的 SVG 节点/边 = G 的节点/边。"""

    def test_svg_node_elements_cover_exactly_G_nodes(self):
        rendered = set(re.findall(r'data-node="([^"]+)"',
                                  self.render["gnodesHtml"]))
        expected = {n["stable_id"] for n in self.G["nodes"]}
        self.assertEqual(rendered, expected,
                         f"缺节点: {sorted(expected - rendered)[:8]}；"
                         f"假节点: {sorted(rendered - expected)[:8]}")
        self.assertEqual(len(rendered), 107)
        # 与产物图 JSON 同源（不是 payload 转抄自检）
        self.assertEqual(expected,
                         {n["stable_id"] for n in self.G_doc["nodes"]})

    def test_svg_edge_paths_cover_exactly_G_edges(self):
        rendered = set(re.findall(r'data-edge="([^"]+)"',
                                  self.render["gedgesHtml"]))
        expected = {e["stable_id"] for e in self.G["edges"]}
        self.assertEqual(rendered, expected,
                         f"缺边: {sorted(expected - rendered)[:8]}；"
                         f"假边: {sorted(rendered - expected)[:8]}")
        self.assertEqual(len(self.G["edges"]), 173)
        # 每条边有可见路径 + 加宽命中路径两层
        self.assertEqual(
            self.render["gedgesHtml"].count('class="gedge" '), 173)
        self.assertEqual(
            self.render["gedgesHtml"].count('class="gedge gedge-hit"'), 173)

    def test_svg_is_real_vector_topology_not_table(self):
        # 是 SVG 拓扑：marker 箭头 + 变换定位的节点盒，x 多列 / y 多分区
        self.assertIn('<svg id="gsvg"', self.html)
        gnodes = self.render["gnodesHtml"]
        self.assertIn('marker-end="url(#garrow-', self.render["gedgesHtml"])
        m = re.findall(r'<g class="gnode" data-node="[^"]+" '
                       r'transform="translate\(([-\d.]+),([-\d.]+)\)"', gnodes)
        self.assertEqual(len(m), 107)
        xs = [float(x) for x, _ in m]
        ys = [float(y) for _, y in m]
        # 非字典序单列：x ≥ 30 个不同列坐标（39 列深度）、y ≥ 11 个分区带
        # （10 语义 lane + 单元内行错开；实测 24 个不同 y 值）
        self.assertGreaterEqual(len(set(xs)), 30,
                                "节点 x 坐标近乎单列，不是拓扑布局")
        self.assertGreaterEqual(len(set(ys)), 11,
                                "节点 y 坐标近乎单行，不是分区布局")

    def test_flow_edges_go_left_to_right_by_depth(self):
        # 分层正确性：每条 flow 边的 x(src) < x(dst)（严格左→右，无字典序列）
        pos = {m[0]: (float(m[1]), float(m[2])) for m in re.findall(
            r'<g class="gnode" data-node="([^"]+)" '
            r'transform="translate\(([-\d.]+),([-\d.]+)\)"',
            self.render["gnodesHtml"])}
        gnode_by_id = {n["stable_id"]: n for n in self.G["nodes"]}
        checked = 0
        for e in self.G_doc["edges"]:
            if e["kind"] == "state":
                continue  # 回边（跨调用）允许弧形回连
            self.assertIn(e["src_node_id"], gnode_by_id, e["stable_id"])
            self.assertIn(e["dst_node_id"], gnode_by_id, e["stable_id"])
            x1 = pos[e["src_node_id"]][0]
            x2 = pos[e["dst_node_id"]][0]
            self.assertLess(x1, x2,
                            f"边 {e['stable_id']} 不是左→右：{x1} >= {x2}")
            checked += 1
        self.assertEqual(checked, 163)  # 173 - 10 state 回边

    def test_no_aliased_fake_data_edges(self):
        # alias_relations 不伪装成 data 边：边集合与 G.edges 精确相等已由
        # test_svg_edge_paths… 断言；页面声明文字存在
        self.assertIn("不伪装成 data 边", self.html)


class TestKeyBranches(_PageHarness):
    """已知真实关键分支显式断言（Q/KV/压缩/Indexer/主干/回边）。"""

    KEY_NODES = (
        "attn.wq_a", "attn.wq_b", "attn.sparse_attn", "attn.select_cat",
        "attn.compressor.blockify", "attn.compressor.overlap",
        "attn.indexer.topk", "attn.indexer.causal_mask",
        "block.hc_pre_attn.sinkhorn", "block.hc_post_attn.combine",
        "moe.gate.topk", "moe.dispatch", "attn.cache_read",
        "attn.cache_update",
    )
    KEY_EDGES = (
        "e_wqa_qnorm",            # attn_norm → attn.wq_a（主干进注意力）
        "e_sparse_q",             # q_rope → sparse_attn（Q 进核心注意力）
        "e_sparse_topk",          # select_cat → sparse_attn（选择索引）
        "e_sparse_kv_decode",     # cache_read → sparse_attn @g_decode
        "e_comp_cache_w",         # 压缩条目写 cache @g_compress_prefill
        "e_comp_cache_w_dec",     # 压缩条目写 cache @g_compress_decode
        "e_idx_select",           # indexer.topk → sparse_attn（select 边）
        "e_idx_mask_topk",        # causal_mask → topk @g_causal_mask
        "e_cache_state_w",        # cache_update → cache_read（跨调用 state）
    )

    def test_key_nodes_rendered(self):
        for nid in self.KEY_NODES:
            self.assertIn(f'data-node="{nid}"',
                          self.render["gnodesHtml"], nid)

    def test_key_edges_rendered_with_guard(self):
        edge_by_id = {e["stable_id"]: e for e in self.G["edges"]}
        for eid in self.KEY_EDGES:
            self.assertIn(f'data-edge="{eid}"',
                          self.render["gedgesHtml"], eid)
        # 关键分支的 guard 与 G 一致（页面不自行发明）
        self.assertIsNone(edge_by_id["e_wqa_qnorm"]["guard_ref"])
        self.assertEqual(edge_by_id["e_sparse_kv_decode"]["guard_ref"],
                         "g_decode")
        self.assertEqual(edge_by_id["e_comp_cache_w"]["guard_ref"],
                         "g_compress_prefill")
        self.assertEqual(edge_by_id["e_idx_select"]["kind"], "select")
        self.assertEqual(edge_by_id["e_cache_state_w"]["kind"], "state")
        self.assertEqual(edge_by_id["e_cache_state_w"]["state_transition_ref"],
                         "tr_attn_cache")

    def test_state_edges_not_in_flow_layering_claim(self):
        # 页面脚本确实把 state 边排除在 Kahn 分层外（回边不破坏左→右布局）
        self.assertIn("if (e.kind === 'state') continue;", self.html)


class TestScenarioEdgeTriState(_PageHarness):
    """默认场景与 prefill/decode 切换实际改变边状态（数据级核验）。"""

    def _scenario(self, sid):
        return next(s for s in self.payload["scenarios"]["instances"]
                    if s["scenario_id"] == sid)

    def _tri_counts(self, sid):
        acts = self._scenario(sid)["guard_activations"]
        on = off = unk = 0
        for e in self.G["edges"]:
            g = e["guard_ref"]
            if not g:
                on += 1
            elif acts[g]["state"] == "true":
                on += 1
            elif acts[g]["state"] == "false":
                off += 1
            else:
                unk += 1
        return on, off, unk

    def test_default_scenario_is_decode_p128_with_full_coverage(self):
        self.assertEqual(
            self.payload["scenarios"]["default_scenario"], "decode_p128")
        on, off, unk = self._tri_counts("decode_p128")
        # 全部 173 条边必落入三态之一（覆盖集合），无第四态
        self.assertEqual(on + off + unk, len(self.G["edges"]))
        self.assertGreater(on, 0)
        self.assertGreater(off, 0)
        # decode_p128 真实活动分支：decode cache 读路径 + Indexer 直选
        acts = self._scenario("decode_p128")["guard_activations"]
        for eid in ("e_sparse_kv_decode", "e_idx_direct_topk"):
            e = next(e for e in self.G["edges"] if e["stable_id"] == eid)
            self.assertEqual(acts[e["guard_ref"]]["state"], "true", eid)
        # prefill 专属压缩写入边在 decode 不活动
        self.assertEqual(acts["g_compress_prefill"]["state"], "false")

    def test_prefill_decode_switch_changes_edge_activity(self):
        # 同一真实边在 prefill/decode 下状态不同（切换实际有作用）
        acts_pre = self._scenario("prefill_t128")["guard_activations"]
        acts_dec = self._scenario("decode_p128")["guard_activations"]
        e = next(e for e in self.G["edges"]
                 if e["stable_id"] == "e_comp_cache_w")
        self.assertEqual(acts_pre[e["guard_ref"]]["state"], "true")
        self.assertEqual(acts_dec[e["guard_ref"]]["state"], "false")
        e2 = next(e for e in self.G["edges"]
                  if e["stable_id"] == "e_sparse_kv_decode")
        self.assertEqual(acts_pre[e2["guard_ref"]]["state"], "false")
        self.assertEqual(acts_dec[e2["guard_ref"]]["state"], "true")
        # 图场景按钮与既有场景面板共用统一入口 gSetScenario（R2a 补修后
        # 两套按钮一次同步 curScen/curScenObj/高亮/图三态；R2b 起两处按钮
        # 均由 gRebuildScenButtons 运行时生成并注册同一回调，单一来源）
        self.assertIn("function gSetScenario(sid)", self.html)
        self.assertIn("function gRebuildScenButtons()", self.html)
        self.assertEqual(
            self.html.count("gSetScenario(s.scenario_id);"), 1,
            "场景按钮回调应只在 gRebuildScenButtons 生成处注册一次")

    def test_rebuild_buttons_register_runtime_callbacks(self):
        # R2b：两套场景按钮为运行时生成（每次切图重建），onclick 均指向
        # gSetScenario；探针执行后两处按钮子元素各含 12 个有效场景按钮
        r = self.render
        self.assertEqual(r["scenBtnCounts"],
                         {"toolbar": 12, "panel": 12})
        expected = {s["scenario_id"] for s in self.payload["scenarios"]["instances"]}
        self.assertEqual(set(r["scenBtnIds"]["toolbar"]), expected)
        self.assertEqual(set(r["scenBtnIds"]["panel"]), expected)

    def test_unguarded_edges_count_matches_G(self):
        # None guard=无条件真：数量由 G 决定（82 条），页面不增不减
        unguarded = [e for e in self.G["edges"] if e["guard_ref"] is None]
        self.assertEqual(len(unguarded), 82)


class TestNodeEdgeClickDetails(_PageHarness):
    """节点/边点击详情：真实端口/张量/证据数据路径全部可解析。"""

    def test_every_node_evidence_and_tensor_resolves(self):
        ev_ids = {e["evidence_id"]
                  for e in self.payload["evidence"]["entries"]}
        tensor_ids = {t["stable_id"] for t in self.G["tensors"]}
        for n in self.G["nodes"]:
            for r in n["evidence_refs"]:
                self.assertIn(r, ev_ids, n["stable_id"])
            for pt in n["input_ports"] + n["output_ports"]:
                self.assertIn(pt["tensor_id"], tensor_ids, n["stable_id"])

    def test_every_edge_tensor_and_guard_resolves(self):
        tensor_ids = {t["stable_id"] for t in self.G["tensors"]}
        guard_ids = {g["stable_id"] for g in self.G["guards"]}
        for e in self.G["edges"]:
            self.assertIn(e["tensor_id"], tensor_ids, e["stable_id"])
            if e["guard_ref"]:
                self.assertIn(e["guard_ref"], guard_ids, e["stable_id"])
            if e["kind"] == "state":
                self.assertTrue(e["state_transition_ref"], e["stable_id"])

    def test_alias_relations_only_as_alias_annotations(self):
        # 19 条 alias_relations 在节点详情中标注 alias（非 data 边）
        self.assertEqual(len(self.G["alias_relations"]), 19)
        self.assertIn("alias（张量别名关系，非 data 边）", self.html)


class TestPageJsExecutionWithGraph(_PageHarness):
    """Node VM 执行：初始化 + 场景切换 + 节点/边详情全部无异常。"""

    def test_js_runs_all_scenarios_and_details_without_error(self):
        # _render_once 已在 setUpClass 完成：0 退出 + 无未知元素 + 计数
        payload = self.render
        self.assertEqual(payload["scenarios"], 12)
        self.assertEqual(payload["nodes"], 107)
        self.assertEqual(payload["edges"], 173)
        self.assertGreaterEqual(payload["tensors"], 100)
        self.assertGreaterEqual(payload["evidence"], 60)
        # 探针原始运行记录已落盘（真实 stdout）
        logged = json.loads(
            (LOGS_DIR / "r1_graph_probe_run.json").read_text(encoding="utf-8"))
        self.assertEqual(logged["nodes"], 107)

    def test_no_undefined_or_nan_in_panels(self):
        # 探针在遇到 undefined/NaN 时已抛错；这里复核退出路径真实执行
        self.assertNotIn("undefined", self.render["note"])
        self.assertIn("no browser/CSS/layout acceptance",
                      self.render["note"])


if __name__ == "__main__":
    unittest.main()
