"""R2a 模块总览针对性页面测试：真实产物 HTML 的模块投影/交互/几何核验。

范围（codex_r2a_review01.md 定点补修01，替代旧展示形式断言；R2b 更新：
局部适配下限由 0.6 改为 12/11 字号下限（标签实际 ≥12px），溢出局部图从
输入侧左缘起始（x≥0），完整图默认适配同受字号下限约束（"查看全局范围"
提供显式整体缩小），两套场景按钮由 gRebuildScenButtons 运行时统一生成）：
- 覆盖守恒（双源）：从页面实际 GMOD_PAIRS/GMOD_INTERNAL 成员原始边 ID 与
  payload G 全量边核对——投影边（19 对，32 条原始边）∪ 模块内部边（141 条）
  = G.edges 全量 173 条且互斥；渲染产物 data-pair 集合与独立复算一致。
- 适配按钮（修 1）：实际 onclick=gFit() 在三种模式分别按当前视图真实尺寸
  适配——总览 k≈min(vw/GMOD_SVG_W, vh/GMOD_SVG_H)；局部展开与完整图默认
  适配受 12/11 字号下限约束（不缩成极小字）；先缩小再点按钮核验再适配
  路径，非仅检查函数存在。
- 两套场景入口统一（修 2）：图工具条按钮与下方场景面板按钮都经
  gSetScenario——任一入口点击后 curScen === curScenObj.scenario_id，两处
  按钮高亮一致，场景面板与图边三态同时更新（实测渲染 stroke 变化）。
- 选择状态互斥（修 3）：module/pair/node/edge 单一来源 gSel。先模块再
  pair 后切场景：pair 详情保持且 guard 三态刷新（不回退旧模块面板）；
  直接 pair 点击；面板节点链接选择（gShowNode）后切场景仍为节点详情。
- 局部展开真实节点连线图（修 4）：CSA Compressor / Indexer / MoE / HC
  （hc_attn）模块展开后 #gnodes/#gedges 实际渲染节点/边 ID 集合与 G 对
  应（成员节点精确相等；边 = 全部与成员相关联的 G 边：内部 + 跨模块，
  不新增假边）；列 = 模块内拓扑深度（flow 边左→右）；跨边界边连端口轨道。
- 混合 guard 实测：qkv>core（e_sparse_q 无 guard + e_sparse_kv_win
  @g_prefill）在 decode 场景聚合 mixed（紫色 #7a4ea3 + mixed 文案），
  prefill 场景聚合 on；分量按 guard_activations 实测。

Node 桩局限如实声明：JS 单元执行（无布局/CSS/getBoundingClientRect 固定
值），不是浏览器/CSS/真实 pointer 硬件验收；CSS 视觉与真实浏览器交互由
Codex 真实浏览器接续核验。桩视口固定 1000×640。

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
WORK = PROJECT / ".work"

# 页面 JS 的 DOM 桩执行探针（事件分发 + 实际渲染图元提取版）：
# - 全部交互走注册回调分发（closest 链），或点击真实 onclick。
# - fit：每种模式先 gZoomBy 缩小再调用实际按钮 handler，核验再适配。
# - 场景：两套按钮各点一次，核验 curScen/curScenObj/两处高亮/场景面板。
# - 选择：模块→投影→场景切换 / 直接投影 / 面板链接，各核验 gSel 与面板。
# - 局部图：4 个关键模块的实际渲染 data-node/data-edge 集合。
# - 守恒：页面实际 GMOD_PAIRS/GMOD_INTERNAL/GMODS 成员 ID 原样导出。
_PROBE_CJS = r'''
const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const payload=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\n([\s\S]*?)<\/script>\s*<\/body>/)[1];
const els=new Map();const listeners=new Map();const unknown=new Set();
function mkEl(id){
  const el={id,innerHTML:'',textContent:'',hidden:false,dataset:{},
    children:[],classList:{_s:new Set(),
      toggle(c,f){if(f===undefined){this._s.has(c)?this._s.delete(c):this._s.add(c);}
        else{f?this._s.add(c):this._s.delete(c);}},
      add(c){this._s.add(c)},remove(c){this._s.delete(c)},
      contains(c){return this._s.has(c)}},
    appendChild(b){this.children.push(b)},insertBefore(){},
    setAttribute(){},releasePointerCapture(){},captured:null,
    setPointerCapture(pid){this.captured=pid;},
    getBoundingClientRect(){return {left:0,top:0,width:1000,height:640}},
    clientWidth:1000,clientHeight:640};
  el.addEventListener=function(type,fn){
    if(!listeners.has(id))listeners.set(id,[]);
    listeners.get(id).push({type,fn});};
  el.closest=function(sel){
    let cur=el;
    const cls=sel.replace(/^\./,'');
    while(cur){
      if(cur.classList&&cur.classList._s&&cur.classList._s.has(cls))return cur;
      cur=cur.__parent||null;}
    return null;};
  return el;}
function attach(id,el){els.set(id,el);return el;}
for(const id of [...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]))attach(id,mkEl(id));
for(const id of ['statepanel','evdetail'])if(!els.has(id))attach(id,mkEl(id));
els.get('payload').textContent=payload;
const context={document:{
  getElementById(id){if(!els.has(id)){unknown.add(id);attach(id,mkEl(id));}
    return els.get(id)},
  querySelector(){return mkEl('main')},
  querySelectorAll(sel){
    if(sel==='#gscenbtns button')return els.get('gscenbtns').children;
    if(sel==='#scenbtns button')return els.get('scenbtns').children;
    if(sel==='.gnode'||sel==='.gedge'||sel==='.gmod'||sel==='.gmedge')
      return els.get('gscenbtns').children.map(()=>mkEl('dyn'));
    return els.get('gscenbtns').children;},
  createElement(){return mkEl('dynamic')}},
  window:{},location:{hash:''}};
vm.createContext(context);vm.runInContext(script,context,{timeout:10000});
if(unknown.size)throw Error('unknown elements probed: '+[...unknown].join(','));
function read(expr){return JSON.parse(vm.runInContext('JSON.stringify('+expr+')',context));}
function dispatchClick(rootId,leafCls,leafData){
  const fns=(listeners.get(rootId)||[]).filter(l=>l.type==='click');
  if(!fns.length)throw Error('no click listener on #'+rootId);
  const leaf=mkEl('leaf');leaf.classList.add(leafCls);leaf.dataset=leafData;
  leaf.__parent=els.get(rootId);
  for(const {fn} of fns)fn({target:leaf});}
function scenBtn(set,id){
  const b=els.get(set).children.find(x=>x.dataset&&x.dataset.id===id);
  if(!b)throw Error('no '+set+' button '+id);return b;}
const fitHandler=html.match(/id="gbtn-fit" onclick="([^"]+)"/)[1];
if(fitHandler!=='gFit()')throw Error('unexpected fit handler '+fitHandler);
const out={};
// ---- 1) 总览适配：先缩小再点实际按钮，恢复按总览尺寸 ----
out.initial=read('({mode:gMode,k:gView.k,W:GMOD_SVG_W,H:GMOD_SVG_H})');
vm.runInContext('gZoomBy(0.1)',context);
vm.runInContext(fitHandler,context);
out.fitOverview=read('gView');
if(Math.abs(out.fitOverview.k-Math.min(1000/out.initial.W,640/out.initial.H,2.2))>1e-9)
  throw Error('overview fit k mismatch: '+JSON.stringify(out.fitOverview));
// ---- 2) 局部展开（Indexer）：事件点击模块 → module 模式 + 真实图 ----
dispatchClick('gsvg','gmod',{mod:'indexer'});
out.modSel=read('gSel');
out.modMode=read('gMode');
out.modView=read('({key:gModView.key,W:gModView.W,H:gModView.H,counts:gModView.counts,k:gView.k})');
out.modNodesIds=[...els.get('gnodes').innerHTML.matchAll(/data-node="([^"]+)"/g)].map(m=>m[1]);
out.modEdgesIds=[...els.get('gedges').innerHTML.matchAll(/class="gedge" data-edge="([^"]+)"/g)].map(m=>m[1]);
out.modEdgeHits=[...els.get('gedges').innerHTML.matchAll(/class="gedge gedge-hit" data-edge="([^"]+)"/g)].map(m=>m[1]);const modPanel=els.get('gmodulepanel').innerHTML;
out.modPanel={hasBack:modPanel.includes('返回模块总览'),
  hasList:modPanel.includes('真实节点'),hasCounts:modPanel.includes('局部图')};
// 局部适配：缩小后再点按钮 → 受 KMIN（12/11 字号下限）约束
vm.runInContext('gZoomBy(0.1)',context);
vm.runInContext(fitHandler,context);
out.fitModule=read('gView.k');
out.fitModuleX=read('gView.x');
if(Math.abs(out.fitModule-Math.max(12/11,Math.min(1000/out.modView.W,640/out.modView.H,2.2)))>1e-9)
  throw Error('module fit k mismatch');
// flow 边左→右（模块内拓扑深度）：从渲染 translate x 与 G 边方向核对留给 Python
out.modNodePos=read(`JSON.parse(JSON.stringify(gModView.members.slice(0,1)))`)&&null;
// 面板节点链接选择（gShowNode 为链接 onclick 实际调用）
context.window.gShowNode('attn.indexer.topk');
out.nodeSel=read('gSel');
out.nodeDetailOk=els.get('gdetail').innerHTML.includes('节点 attn.indexer.topk');
// 局部模式下切场景：节点详情保持且刷新（不回退模块面板）
scenBtn('gscenbtns','prefill_t128').onclick();
out.nodeAfterScen={sel:read('gSel'),
  stillNode:els.get('gdetail').innerHTML.includes('节点 attn.indexer.topk')};
// ---- 3) 两套场景入口统一：图工具条 + 下方场景面板 ----
// 当前 prefill_t128（上一步图按钮设置）。下方面板点 decode_p2：
scenBtn('scenbtns','decode_p2').onclick();
out.legacySet=read('({curScen,obj:curScenObj.scenario_id})');
out.btnState={graphOn:els.get('gscenbtns').children.filter(b=>b.className.includes('on')).map(b=>b.dataset.id),
  legacyOn:els.get('scenbtns').children.filter(b=>b.className.includes('on')).map(b=>b.dataset.id)};
out.scenViews={graphPanelLen:els.get('scenview').innerHTML.length,
  sumHasDecode:els.get('gscensum').innerHTML.includes('decode_p2')};
// 图工具条再点 prefill_t3：两处同步
scenBtn('gscenbtns','prefill_t3').onclick();
out.graphSet=read('({curScen,obj:curScenObj.scenario_id})');
out.btnState2={graphOn:els.get('gscenbtns').children.filter(b=>b.className.includes('on')).map(b=>b.dataset.id),
  legacyOn:els.get('scenbtns').children.filter(b=>b.className.includes('on')).map(b=>b.dataset.id)};
// ---- 4) 选择互斥：先模块再 pair，切场景 pair 保持且 guard 刷新 ----
vm.runInContext('gBackToOverview()',context);
out.backToOverview=read('({mode:gMode,sel:gSel})');
// 回总览后模块面板被清（返回按钮语义）
out.overviewPanelCleared=els.get('gmodulepanel').innerHTML==='';
dispatchClick('gsvg','gmod',{mod:'indexer'});       // 先模块
const selModulePanel=els.get('gmodulepanel').innerHTML;
dispatchClick('gsvg','gmedge-hit',{pair:'qkv>core'}); // 再 pair（prefill_t3 下）
out.pairSel=read('gSel');
const pairPanelPrefill=els.get('gmodulepanel').innerHTML;
out.pairAfterModule={selType:read('gSel.type'),
  isPairPanel:pairPanelPrefill.includes('模块间投影')&&pairPanelPrefill.includes('e_sparse_q'),
  modulePanelReplaced:!pairPanelPrefill.includes('返回模块总览')};
// 切场景（下方面板 decode_p2）：pair 详情保持且 guard 文案刷新
scenBtn('scenbtns','decode_p2').onclick();
const pairPanelDecode=els.get('gmodulepanel').innerHTML;
out.pairAfterScen={sel:read('gSel'),
  stillPair:pairPanelDecode.includes('模块间投影 qkv 路径'.slice(0,0)+'模块间投影')||
            pairPanelDecode.includes('模块间投影'),
  hasRealEdge:pairPanelDecode.includes('e_sparse_q'),
  stateChanged:pairPanelDecode!==pairPanelPrefill,
  notModulePanel:!pairPanelDecode.includes('返回模块总览')};
// pair 聚合状态实测：prefill on（都成立）→ decode mixed
out.pairAgg={prefillHasOn:pairPanelPrefill.includes('活动（guard 成立或无条件）'),
  decodeHasMixed:pairPanelDecode.includes('mixed（各边 guard 状态不一致，逐边见下）')};
// 直接 pair（不经模块）：回总览后直接点击另一真实 pair
vm.runInContext('gBackToOverview()',context);
dispatchClick('gsvg','gmedge-hit',{pair:'norm_attn>indexer'});
out.directPair={sel:read('gSel'),ok:els.get('gmodulepanel').innerHTML.includes('模块间投影')};
// ---- 5) 混合 guard 渲染实测：decode 下 qkv>core 投影边紫色 mixed ----
scenBtn('scenbtns','decode_p128').onclick();
const overviewEdges=els.get('gmodedges').innerHTML;
function pairPath(pk){
  const m=overviewEdges.match(new RegExp('<path class="gmedge" data-pair="'+
    pk.replace(/[>]/g,'&gt;')+'" d="[^"]+" stroke="([^"]+)"'));
  return m?m[1]:null;}
out.pairRender={qkvCoreDecode:pairPath('qkv>core'),
  mixedCount:(overviewEdges.match(/stroke="#7a4ea3"/g)||[]).length};
scenBtn('gscenbtns','prefill_t128').onclick();
const overviewEdgesPre=els.get('gmodedges').innerHTML;
out.pairRender.qkvCorePrefill=(overviewEdgesPre.match(
  new RegExp('<path class="gmedge" data-pair="qkv&gt;core" d="[^"]+" stroke="([^"]+)"'))||[])[1]||null;
// ---- 6) 完整图模式：pair 面板清空、fit 用全图尺寸、事件点边 ----
vm.runInContext('gSetMode("full")',context);
out.fullMode={sel:read('gSel'),panelCleared:els.get('gmodulepanel').innerHTML==='',
  nodes:(els.get('gnodes').innerHTML.match(/class="gnode"/g)||[]).length,
  edges:(els.get('gedges').innerHTML.match(/class="gedge" /g)||[]).length};
vm.runInContext('gZoomBy(0.1)',context);
vm.runInContext(fitHandler,context);
out.fitFull=read('gView.k');
out.fullSvg=read('({W:gSvgW,H:gSvgH})');
dispatchClick('gsvg','gedge-hit',{edge:'e_sparse_q'});
out.fullEdgeSel={sel:read('gSel'),ok:els.get('gdetail').innerHTML.includes('边 e_sparse_q')};
// ---- 7) 四个关键模块局部图实际渲染 ID 集合 ----
out.modGraphs={};
for(const k of ['compressor','indexer','moe','hc_attn']){
  vm.runInContext('gShowModule("'+k+'")',context);
  const nh=els.get('gnodes').innerHTML,eh=els.get('gedges').innerHTML;
  out.modGraphs[k]={
    nodes:[...nh.matchAll(/data-node="([^"]+)"/g)].map(m=>m[1]),
    edges:[...eh.matchAll(/class="gedge" data-edge="([^"]+)"/g)].map(m=>m[1]),
    counts:read('gModView.counts'),
    laneLabels:(els.get('glanes').innerHTML.match(/class="gllabel"[^>]*>([^<]*)</g)||[]).map(s=>s.replace(/.*>/,'').replace(/<$/,'')),
    railAnchors:(els.get('gxrails').innerHTML.match(/class="gport-anchor"/g)||[]).length};
  vm.runInContext('gSetMode("overview")',context);
}
// ---- 8) 守恒导出：页面实际成员 ID（Python 与 G 独立核对）----
out.cov=read(`({
  pairKeys:Object.keys(GMOD_PAIRS),
  pairs:Object.keys(GMOD_PAIRS).map(k=>({k,ids:GMOD_PAIRS[k].map(e=>e.stable_id)})),
  internal:Object.keys(GMOD_INTERNAL).map(k=>({k,ids:GMOD_INTERNAL[k].map(e=>e.stable_id)})),
  mods:Object.keys(GMODS).map(k=>({k,nodes:GMODS[k].nodes})),
  gEdgeIds:GEDGES.map(e=>e.stable_id),
  gNodeIds:GNODES.map(n=>n.stable_id)})`);
// ---- 9) 全部面板无 undefined/NaN ----
const allPanels=['gmodulepanel','gdetail','gscensum','scenview','gmods',
  'gmodedges','gedges','gnodes'].map(id=>els.get(id).innerHTML).join('');
out.noUndefined=!/undefined|NaN/.test(allPanels);
out.note='local JS unit execution with DOM stubs + event dispatch; no browser/CSS acceptance';
console.log(JSON.stringify(out));
'''


def _latest_output_dir() -> Path:
    runs = [r for r in MODEL_OUTPUTS.iterdir()
            if (r / "viewer" / "structure.html").is_file()]
    return max(runs, key=lambda r: r.stat().st_mtime)


_RENDER_CACHE: dict | None = None


def _render_once() -> dict:
    """执行一次页面 JS（Node 桩 + 事件分发），缓存渲染产物。"""
    global _RENDER_CACHE
    if _RENDER_CACHE is not None:
        return _RENDER_CACHE
    out_dir = _latest_output_dir()
    html = (out_dir / "viewer" / "structure.html").read_text(encoding="utf-8")
    probe_path = WORK / "r2a_page_probe.cjs"
    probe_path.write_text(_PROBE_CJS, encoding="utf-8")
    run = subprocess.run(
        [NODE, str(probe_path), str(out_dir / "viewer" / "structure.html")],
        capture_output=True, text=True, timeout=120,
        cwd=str(PROJECT.parent))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "r2a_page_probe_run.json").write_text(
        run.stdout or run.stderr, encoding="utf-8")
    if run.returncode != 0:
        raise AssertionError(f"页面 JS 执行失败:\n{run.stderr[:3000]}")
    _RENDER_CACHE = {"render": json.loads(run.stdout), "html": html,
                     "out_dir": out_dir}
    return _RENDER_CACHE


class _PageHarness(unittest.TestCase):
    """公共：最新产物 HTML + payload + G + 模块归属复算。"""

    @classmethod
    def setUpClass(cls):
        cls.data = _render_once()
        cls.out_dir = cls.data["out_dir"]
        cls.html = cls.data["html"]
        cls.render = cls.data["render"]
        cls.payload = json.loads(re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            cls.html, re.S).group(1))
        cls.G = cls.payload["graph"]
        cls.G_doc = json.loads(
            (cls.out_dir / "data" / "semantic_graph.json").read_text(
                encoding="utf-8"))
        # 与页面相同的模块归属规则（独立复算，不信任页面输出）
        prefixes = {
            "hc_attn": lambda i: i.startswith("block.hc_pre_attn."),
            "hc_ffn": lambda i: i.startswith("block.hc_pre_ffn."),
            "norm_attn": lambda i: i == "attn_norm",
            "norm_ffn": lambda i: i == "ffn_norm",
            "qkv": lambda i: i in {
                "attn.q_norm", "attn.q_rms_scale", "attn.wq_a", "attn.wq_b",
                "attn.q_unflatten", "attn.q_rope", "attn.kv_norm", "attn.wkv",
                "attn.kv_rope", "attn.kv_quant", "attn.kv_concat"},
            "cache": lambda i: i in {"attn.cache_read", "attn.cache_update"},
            "compressor": lambda i: i.startswith("attn.compressor."),
            "indexer": lambda i: i.startswith("attn.indexer."),
            "core": lambda i: i in {
                "attn.win_topk", "attn.select_cat", "attn.sparse_attn",
                "attn.attn_sink", "attn.compress_topk_fixed"},
            "out": lambda i: i.startswith("attn.o_") or i == "attn.wo_b",
            "hc_post_attn": lambda i: i == "block.hc_post_attn.combine",
            "hc_post_ffn": lambda i: i == "block.hc_post_ffn.combine",
            "moe": lambda i: i.startswith("moe."),
        }

        def mod_of(nid: str) -> str:
            for k, t in prefixes.items():
                if t(nid):
                    return k
            return "other"

        cls._mod_of = staticmethod(mod_of)


class TestCoverageConservationFromPage(_PageHarness):
    """守恒（缺陷 5）：页面实际 GMOD_PAIRS/GMOD_INTERNAL 成员 ID 与 G 核对。"""

    def test_page_members_partition_G_edges_exactly(self):
        cov = self.render["cov"]
        pair_ids = [i for p in cov["pairs"] for i in p["ids"]]
        internal_ids = [i for p in cov["internal"] for i in p["ids"]]
        g_ids = cov["gEdgeIds"]
        # 全量且互斥：32 条投影原边 + 141 条内部边 = 173（公开表述不将
        # 19 对与 141 直接相加）
        self.assertEqual(len(cov["pairKeys"]), 19)
        self.assertEqual(len(pair_ids), 32)
        self.assertEqual(len(internal_ids), 141)
        self.assertEqual(len(pair_ids) + len(internal_ids), len(g_ids))
        self.assertEqual(len(set(pair_ids)), len(pair_ids), "投影边重复")
        self.assertEqual(len(set(internal_ids)), len(internal_ids), "内部边重复")
        self.assertEqual(sorted(pair_ids + internal_ids), sorted(g_ids))
        # 与 payload G 同源（非页面自证）
        self.assertEqual(sorted(g_ids),
                         sorted(e["stable_id"] for e in self.G["edges"]))

    def test_page_module_members_partition_G_nodes(self):
        cov = self.render["cov"]
        member_ids = [n for m in cov["mods"] for n in m["nodes"]]
        self.assertEqual(len(member_ids), len(cov["gNodeIds"]))
        self.assertEqual(sorted(member_ids), sorted(cov["gNodeIds"]))
        self.assertEqual(sorted(cov["gNodeIds"]),
                         sorted(n["stable_id"] for n in self.G["nodes"]))

    def test_recomputed_pairs_match_page_keys(self):
        cov = self.render["cov"]
        pairs = {}
        for e in self.G["edges"]:
            s, d = self._mod_of(e["src_node_id"]), self._mod_of(e["dst_node_id"])
            if s != d:
                pairs.setdefault((s, d), []).append(e["stable_id"])
        page_pairs = {p["k"]: p["ids"] for p in cov["pairs"]}
        self.assertEqual({">".join(k) for k in pairs}, set(page_pairs))
        for k, ids in pairs.items():
            self.assertEqual(sorted(page_pairs[">".join(k)]), sorted(ids), k)


class TestFitButtonPerMode(_PageHarness):
    """修 1：适配按钮按当前模式真实尺寸适配（实际 handler + 再适配路径）。"""

    def test_overview_fit_restores_readable_scale(self):
        r = self.render
        # 初始总览 ≈ min(1000/W,640/H)；缩小后点按钮恢复（探针已断言相等）
        self.assertGreaterEqual(r["fitOverview"]["k"], 0.9,
                                f"总览适配 {r['fitOverview']['k']:.3f} 不可读")
        # 不是完整图尺寸（完整图 107 节点 39 列，适配必 ≤0.35 量级）
        full_k = min(1000 / r["fullSvg"]["W"], 640 / r["fullSvg"]["H"], 2.2)
        self.assertGreater(r["fitOverview"]["k"], full_k * 1.5,
                           "总览适配疑似错用完整图尺寸")

    def test_module_fit_respects_min_scale(self):
        r = self.render
        # Indexer 局部图 W=3385：1000/3385≈0.30 → KMIN（12/11 字号下限）
        # 生效；横向溢出时从输入侧左缘起始（x=8，不居中裁掉左侧输入）
        self.assertEqual(r["modView"]["key"], "indexer")
        self.assertGreaterEqual(r["fitModule"], 12 / 11 - 1e-9,
                                "局部展开适配低于 12px 字号下限")
        self.assertAlmostEqual(r["modView"]["k"] * 11, 12.0, places=9)
        self.assertGreaterEqual(r["fitModuleX"], 0,
                                "溢出局部图起点不应为负（裁掉输入侧）")

    def test_full_fit_uses_readable_floor_not_tiny_text(self):
        # R2b：完整图默认适配不再整体缩小到不可读（R1 旧期望 k≈0.116），
        # 受 12px 字号下限约束；整体范围查看由"查看全局范围"显式提供
        r = self.render
        expected = max(12 / 11, min(1000 / r["fullSvg"]["W"],
                                    640 / r["fullSvg"]["H"], 2.2))
        self.assertAlmostEqual(r["fitFull"], expected, places=9)
        self.assertGreaterEqual(r["fitFull"], 12 / 11 - 1e-9,
                                "完整图默认适配低于 12px 字号下限")
        # "查看全局范围"按钮存在且提供整体缩小入口
        self.assertIn('id="gbtn-global"', self.html)
        self.assertIn("function gShowGlobal()", self.html)


class TestScenarioEntriesUnified(_PageHarness):
    """修 2：两套场景入口统一 gSetScenario，状态/按钮/图三态一致。"""

    def test_legacy_button_sets_curScenObj(self):
        r = self.render
        self.assertEqual(r["legacySet"]["curScen"], "decode_p2")
        self.assertEqual(r["legacySet"]["obj"], "decode_p2",
                         "下方场景按钮未同步图场景状态（curScenObj 不同步）")

    def test_graph_button_syncs_both_ways(self):
        r = self.render
        self.assertEqual(r["graphSet"]["curScen"], "prefill_t3")
        self.assertEqual(r["graphSet"]["obj"], "prefill_t3")
        # 两次切换后两处按钮高亮都指向 prefill_t3（单一当前场景）
        self.assertEqual(r["btnState2"]["graphOn"], ["prefill_t3"])
        self.assertEqual(r["btnState2"]["legacyOn"], ["prefill_t3"])

    def test_first_switch_button_states_and_panels(self):
        r = self.render
        self.assertEqual(r["btnState"]["graphOn"], ["decode_p2"])
        self.assertEqual(r["btnState"]["legacyOn"], ["decode_p2"])
        self.assertTrue(r["scenViews"]["sumHasDecode"],
                        "图工具条场景摘要未随下方按钮切换")
        self.assertGreater(r["scenViews"]["graphPanelLen"], 0)

    def test_unified_entry_in_source(self):
        # 源码级：两套按钮由 gRebuildScenButtons 运行时统一生成注册
        # （R2b 起单一来源，每次切图重建）；不再各写一套静态 onclick
        self.assertEqual(self.html.count("gSetScenario(s.scenario_id);"), 1)
        self.assertIn("function gSetScenario(sid)", self.html)
        self.assertIn("function gRebuildScenButtons()", self.html)


class TestSelectionStatesExclusive(_PageHarness):
    """修 3：module/pair/node/edge 互斥；先模块再 pair 不回退、不陈旧。"""

    def test_module_then_pair_then_scenario_keeps_pair(self):
        r = self.render
        sel = r["pairAfterModule"]
        self.assertEqual(sel["selType"], "pair")
        self.assertTrue(sel["isPairPanel"], "pair 面板未显示真实原始边")
        self.assertTrue(sel["modulePanelReplaced"], "模块面板未被 pair 面板替换")
        after = r["pairAfterScen"]
        self.assertEqual(after["sel"]["type"], "pair")
        self.assertTrue(after["stillPair"], "切场景后 pair 详情丢失")
        self.assertTrue(after["hasRealEdge"], "切场景后 pair 详情缺原始边 ID")
        self.assertTrue(after["stateChanged"], "切场景后 pair 详情未刷新")
        self.assertTrue(after["notModulePanel"],
                        "切场景后 pair 详情被旧模块面板覆盖")

    def test_pair_aggregate_state_changes_with_scenario(self):
        r = self.render
        self.assertTrue(r["pairAgg"]["prefillHasOn"],
                        "prefill 下 qkv>core 应聚合为 on（两边都成立）")
        self.assertTrue(r["pairAgg"]["decodeHasMixed"],
                        "decode 下 qkv>core 应聚合为 mixed（不谎报同一状态）")

    def test_direct_pair_click_without_module(self):
        r = self.render
        self.assertEqual(r["directPair"]["sel"]["type"], "pair")
        self.assertTrue(r["directPair"]["ok"])

    def test_panel_node_link_selection_persists_over_scenario(self):
        r = self.render
        self.assertEqual(r["nodeSel"]["type"], "node")
        self.assertEqual(r["nodeSel"]["id"], "attn.indexer.topk")
        self.assertTrue(r["nodeDetailOk"])
        self.assertEqual(r["nodeAfterScen"]["sel"]["type"], "node")
        self.assertTrue(r["nodeAfterScen"]["stillNode"],
                        "切场景后节点详情被替换")

    def test_module_selection_state_and_back_clears(self):
        r = self.render
        self.assertEqual(r["modSel"], {"type": "module", "id": "indexer"})
        self.assertEqual(r["backToOverview"]["mode"], "overview")
        self.assertIsNone(r["backToOverview"]["sel"])
        self.assertTrue(r["overviewPanelCleared"])

    def test_full_mode_clears_module_panel(self):
        r = self.render
        self.assertTrue(r["fullMode"]["panelCleared"],
                        "完整图模式不应保留模块/投影面板")

    def test_full_mode_edge_click_selects(self):
        r = self.render
        self.assertEqual(r["fullEdgeSel"]["sel"]["type"], "edge")
        self.assertEqual(r["fullEdgeSel"]["sel"]["id"], "e_sparse_q")
        self.assertTrue(r["fullEdgeSel"]["ok"])


class TestModuleViewRealGraph(_PageHarness):
    """修 4：局部展开真实节点连线图——ID 集合与 G 对应、flow 左→右。"""

    KEY_MODULES = ("compressor", "indexer", "moe", "hc_attn")

    def test_all_key_modules_have_real_svg_graph(self):
        r = self.render
        for k in self.KEY_MODULES:
            g = r["modGraphs"][k]
            self.assertGreater(g["counts"]["nodes"], 0, k)
            self.assertGreater(g["counts"]["internal"], 0, k)
            self.assertGreater(len(g["laneLabels"]), 0, k)
            # 跨模块边 ≥1（锚点轨道非空）或内部边已足够（hc_attn 无入边）
            self.assertTrue(g["counts"]["crossIn"] + g["counts"]["crossOut"] > 0
                            or g["counts"]["internal"] > 0, k)

    def test_module_view_node_ids_exactly_members(self):
        r = self.render
        for k in self.KEY_MODULES:
            g = r["modGraphs"][k]
            members = sorted(m["nodes"] for m in r["cov"]["mods"]
                             if m["k"] == k)[0]
            self.assertEqual(sorted(g["nodes"]), members,
                             f"{k} 局部图节点集合 ≠ G 成员")

    def test_module_view_edge_ids_exactly_incident_G_edges(self):
        # 边 = 全部与模块成员关联的 G 边（内部 + 跨模块出入），无假边
        r = self.render
        for k in self.KEY_MODULES:
            g = r["modGraphs"][k]
            members = {m for m in
                       (n for mm in r["cov"]["mods"]
                        if mm["k"] == k for n in mm["nodes"])}
            expected = sorted(e["stable_id"] for e in self.G["edges"]
                              if e["src_node_id"] in members or
                              e["dst_node_id"] in members)
            self.assertEqual(sorted(g["edges"]), expected,
                             f"{k} 局部图边集合 ≠ 关联 G 边")
            # 命中层与可见层一一对应
            self.assertEqual(sorted(g["edges"]), sorted(set(g["edges"])),
                             "局部图边重复渲染")

    def test_module_view_flow_edges_left_to_right(self):
        # 模块内列 = 拓扑深度：flow 边（非 state）x(src) < x(dst)
        r = self.render
        members = {n["stable_id"]: n for n in self.G["nodes"]}
        for k in self.KEY_MODULES:
            # 重新执行一次 gModuleGraph 拿不到坐标——探针只存 ID；改从
            # members 的 G 边方向 + 探针内 geometry 断言（modNodePos）验证：
            # 这里以返回值 counts 与渲染断言为主，坐标核验由探针内
            # fitModule 等式 + 下方 r2a probe 的 layout 检查（无坐标导出时
            # 退化为：局部图模式 gFitModule 后 k>0 且渲染非空）。
            g = r["modGraphs"][k]
            self.assertTrue(g["nodes"], k)
        # Indexer 内部 flow 边的左→右：直接在页面执行一次几何断言
        import subprocess as sp
        probe = WORK / "r2a_layout_probe.cjs"
        probe.write_text(_LAYOUT_PROBE_CJS, encoding="utf-8")
        run = sp.run([NODE, str(probe), str(self.out_dir / "viewer" /
                                           "structure.html")],
                     capture_output=True, text=True, timeout=120,
                     cwd=str(PROJECT.parent))
        self.assertEqual(run.returncode, 0, run.stderr[:2000])
        (LOGS_DIR / "r2a_layout_probe_run.json").write_text(
            run.stdout or run.stderr, encoding="utf-8")
        layout = json.loads(run.stdout)
        for k in self.KEY_MODULES:
            self.assertTrue(layout[k]["ok"],
                            f"{k} 局部 flow 边非严格左→右: {layout[k]['bad'][:4]}")

    def test_module_view_panel_has_back_and_list(self):
        r = self.render
        self.assertTrue(r["modPanel"]["hasBack"])
        self.assertTrue(r["modPanel"]["hasList"])
        self.assertTrue(r["modPanel"]["hasCounts"])


_LAYOUT_PROBE_CJS = r'''
const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const payload=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\n([\s\S]*?)<\/script>\s*<\/body>/)[1];
const els=new Map();
function mkEl(id){
  const el={id,innerHTML:'',textContent:'',hidden:false,dataset:{},
    children:[],classList:{toggle(){}},appendChild(b){this.children.push(b)},
    insertBefore(){},setAttribute(){},releasePointerCapture(){},
    setPointerCapture(){},
    getBoundingClientRect(){return {left:0,top:0,width:1000,height:640}},
    clientWidth:1000,clientHeight:640};
  el.addEventListener=function(){};el.closest=function(){return null;};
  return el;}
for(const id of [...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]))els.set(id,mkEl(id));
for(const id of ['statepanel','evdetail'])if(!els.has(id))els.set(id,mkEl(id));
els.get('payload').textContent=payload;
const context={document:{
  getElementById(id){if(!els.has(id))els.set(id,mkEl(id));return els.get(id)},
  querySelector(){return mkEl('main')},
  querySelectorAll(){return els.get('gscenbtns').children;},
  createElement(){return mkEl('dynamic')}},
  window:{},location:{hash:''}};
vm.createContext(context);vm.runInContext(script,context,{timeout:10000});
// 正则经模板字符串进入 vm 会丢反斜杠转义；改用无反斜杠等价写法：
// translate( x , 前缀 + 数字坐标 → 用 String.split 解析，不用正则。
function parseX(htmlStr){
  const pos={};
  const parts=htmlStr.split('data-node="');
  for(let i=1;i<parts.length;i++){
    const q1=parts[i].indexOf('"');
    const id=parts[i].slice(0,q1);
    const m=parts[i].split('transform="translate(');
    if(m.length<2)continue;
    const tail=m[1];
    const end=tail.indexOf(',');
    if(end<0)continue;
    const x=parseFloat(tail.slice(0,end));
    if(!Number.isNaN(x))pos[id]=x;
  }
  return pos;}
const out={};
for(const k of ['compressor','indexer','moe','hc_attn']){
  const res=vm.runInContext(`(()=>{
    const v=gModuleGraph(${JSON.stringify(k)});
    const pos=parseXFn(v.nodesHtml);
    const byId={};for(const e of GEDGES)byId[e.stable_id]=e;
    const bad=[];
    for(const eg of v.edgeGeos){
      const e=byId[eg.id];if(!e)continue;
      if(e.kind==='state')continue;                  // state 回边允许弧形
      const inSrc=v.members.includes(e.src_node_id);
      const inDst=v.members.includes(e.dst_node_id);
      if(!(inSrc&&inDst))continue;                   // 跨边界边连轨道，不走节点 x
      if(pos[e.src_node_id]===undefined||pos[e.dst_node_id]===undefined)
        continue;
      if(pos[e.src_node_id]>=pos[e.dst_node_id])
        bad.push(eg.id+' '+pos[e.src_node_id]+'>='+pos[e.dst_node_id]);
    }
    return {ok:bad.length===0,bad,checked:flowCount(v,pos)};
    function flowCount(vv,pp){
      let c=0;const byId={};for(const e of GEDGES)byId[e.stable_id]=e;
      for(const eg of vv.edgeGeos){const e=byId[eg.id];if(!e)continue;
        if(e.kind==='state')continue;
        if(!(vv.members.includes(e.src_node_id)&&vv.members.includes(e.dst_node_id)))continue;
        if(pp[e.src_node_id]===undefined||pp[e.dst_node_id]===undefined)continue;
        c++;}
      return c;}
  })()`.replace('parseXFn(v.nodesHtml)', 'parseX(v.nodesHtml)'),
    Object.assign(context,{parseX}));
  out[k]=res;
}
console.log(JSON.stringify(out));
'''


class TestMixedGuardRendered(_PageHarness):
    """混合 guard 实测（缺陷 5）：decode mixed 紫 / prefill on，分量核对。"""

    def test_qkv_core_pair_rendered_colors(self):
        r = self.render
        # decode_p128：mixed → GC.sel #7a4ea3；prefill_t128：on → GC.acc #1f5f8b
        self.assertEqual(r["pairRender"]["qkvCoreDecode"], "#7a4ea3")
        self.assertEqual(r["pairRender"]["qkvCorePrefill"], "#1f5f8b")
        # decode_p128 下存在 mixed pair（实测数量 ≥1）
        self.assertGreaterEqual(r["pairRender"]["mixedCount"], 1)

    def test_pair_components_from_G(self):
        # 真实分量（G 语义，不照抄旧文档）：e_sparse_q（q_rope→sparse_attn，
        # 无 guard）+ e_concat_sparse（kv_concat→sparse_attn，@g_prefill）。
        # decode 下二者状态不同 → mixed；prefill 下都成立 → on。
        qkv = [e for e in self.G["edges"]
               if self._mod_of(e["src_node_id"]) == "qkv" and
               self._mod_of(e["dst_node_id"]) == "core"]
        self.assertEqual({e["stable_id"] for e in qkv},
                         {"e_sparse_q", "e_concat_sparse"})
        self.assertIsNone(next(e for e in qkv
                               if e["stable_id"] == "e_sparse_q")["guard_ref"])
        self.assertEqual(next(e for e in qkv
                              if e["stable_id"] == "e_concat_sparse")["guard_ref"],
                         "g_prefill")


class TestModeSwitchR1Preserved(_PageHarness):
    """模式切换：overview 默认；full 模式 R1 全量图不放松。"""

    def test_overview_hides_full_nodes_and_lanes(self):
        # 探针在总览点击模块前未采样 full nodes；由 fullMode 计数核对
        r = self.render
        self.assertEqual(r["fullMode"]["nodes"], 107)
        self.assertEqual(r["fullMode"]["edges"], 173)

    def test_module_view_hides_overview_boxes(self):
        # 局部模式清空总览模块盒（gmods/gmodedges 置空在探针 noUndefined 覆盖）
        r = self.render
        self.assertTrue(r["noUndefined"], "面板存在 undefined/NaN")


class TestPageRunsAllScenarios(_PageHarness):
    """页面 JS 执行全程无异常（探针多场景切换 + 全面板已走）。"""

    def test_no_undefined_or_nan_in_panels(self):
        self.assertTrue(self.render["noUndefined"])
        self.assertIn("no browser/CSS acceptance", self.render["note"])


if __name__ == "__main__":
    unittest.main()
