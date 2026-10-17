"""B4 补修01 页面 JS 执行回归（Node vm + DOM 桩；非浏览器验收）。

覆盖 codex_b4_review01 缺口2/3 的页面侧关闭证据：
- 证据点选：head/Markov/Confidence/DSparkAttention 关键证据经 showEv
  真实渲染（固定 SHA 来源、行区间、片段、核对结论入 innerHTML）；
  DSpark 图 ev_ 引用全部可解析（无 showEv 直接 return 的丢证据路径）。
- 角色面板：gSetVariant('dspark') 后 DSpark 实例角色/参数绑定面板渲染
  （mtp.0/1/2、层 43/44/45、first/last 角色、公共模板参数→实例路径、
  唯一 embed/head、last 专有参数 runtime/checkpoint dtype）；切回主图
  面板恢复归属说明（不残留 DSpark 数据）。
- 场景版本绑定：DSpark 场景面板渲染每实例 before/after/published 版本
  与前提分开的有效槽位数（decode_p127 按 P+1 口径）。

局限如实声明：JS 单元执行（DOM 桩），不是浏览器/CSS/交互验收；真实
浏览器验收仍 blocked，由 Codex 接续。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT = Path(__file__).resolve().parents[1]
NODE = "/opt/homebrew/bin/node"
MODEL_OUTPUTS = PROJECT / "outputs" / "deepseek_v4_flash_0731"

# 页面证据点选关键锚（缺口2 显式要求实际点选的四类证据）
_ANCHOR_EV = {
    "head": "ev_ex_parallel_head",
    "markov": "ev_mk_loop",
    "confidence": "ev_cf_conf_head",
    "dspark_attention": "ev_ds_attn_init",
}

_PROBE_CJS = r'''
const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const payload=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\n([\s\S]*?)<\/script>\s*<\/body>/)[1];
const P=JSON.parse(payload);
const els=new Map();
function mkEl(id){
  const el={id,textContent:'',hidden:false,dataset:{},children:[],
    classList:{_s:new Set(),
      toggle(c,f){if(f===undefined){this._s.has(c)?this._s.delete(c):this._s.add(c);}
        else{f?this._s.add(c):this._s.delete(c);}},
      add(c){this._s.add(c)},remove(c){this._s.delete(c)},
      contains(c){return this._s.has(c)}},
    appendChild(b){this.children.push(b)},insertBefore(){},
    setAttribute(){},releasePointerCapture(){},captured:null,
    setPointerCapture(pid){this.captured=pid;},
    getBoundingClientRect(){return {left:0,top:0,width:1000,height:640}},
    clientWidth:1000,clientHeight:640};
  let _html='';
  Object.defineProperty(el,'innerHTML',{get(){return _html;},
    set(v){_html=v;if(v==='')el.children.length=0;}});
  el.addEventListener=function(){};
  return el;}
for(const id of [...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]))els.set(id,mkEl(id));
for(const id of ['statepanel','evdetail','dsparkrolepanel','gdetail'])
  if(!els.has(id))els.set(id,mkEl(id));
els.get('payload').textContent=payload;
const context={document:{
  getElementById(id){if(!els.has(id))els.set(id,mkEl(id));return els.get(id)},
  querySelector(){return mkEl('main')},
  querySelectorAll(sel){
    if(sel==='#gscenbtns button')return els.get('gscenbtns').children;
    if(sel==='#scenbtns button')return els.get('scenbtns').children;
    if(sel==='#gbtn-var')return [];
    return [];},
  createElement(){return mkEl('dynamic')}},
  window:{},location:{hash:''}};
vm.createContext(context);vm.runInContext(script,context,{timeout:30000});
function read(expr){return JSON.parse(vm.runInContext('JSON.stringify('+expr+')',context));}
const out={};
// ---- 1) 关键证据点选（head/Markov/Confidence/DSparkAttention）----
out.anchorEvidence={};
for(const [tag,id] of Object.entries(JSON.parse(process.argv[3]))){
  if(!P.evidence.entries.some(e=>e.evidence_id===id))
    throw Error('anchor evidence missing from payload: '+id);
  context.window.showEv(id);
  const htmlOut=els.get('evdetail').innerHTML;
  if(!(htmlOut.indexOf(id)>=0))throw Error('showEv did not render '+id);
  out.anchorEvidence[tag]={
    rendered:true,
    hasSource:/来源/.test(htmlOut),
    hasLines:/行区间/.test(htmlOut),
    hasSnippet:/<pre>/.test(htmlOut),
    hasClaim:/核对结论/.test(htmlOut)};
}
// ---- 2) DSpark 图 ev_ 引用全量可解析（无 showEv 静默 return）----
vm.runInContext("gSetVariant('dspark')",context);
out.curVar=read('curVar');
const evIds=read('(function(){const s=new Set();for(const n of ctx.G.nodes)for(const r of n.evidence_refs)s.add(r);for(const e of ctx.G.edges)for(const r of e.evidence_refs)s.add(r);for(const g of ctx.G.guards)for(const r of g.evidence_refs)s.add(r);for(const t of ctx.G.state_transitions)for(const r of t.evidence_refs)s.add(r);for(const a of ctx.G.named_axes)if(a.evidence_ref.indexOf("ev_")===0)s.add(a.evidence_ref);for(const t of ctx.G.tensors)for(const a of t.axes)if(a.evidence_ref.indexOf("ev_")===0)s.add(a.evidence_ref);return [...s]})()');
const payloadEv=new Set(P.evidence.entries.map(e=>e.evidence_id));
out.dsparkEvRefs=evIds.length;
out.dsparkEvMissing=evIds.filter(id=>!payloadEv.has(id));
if(out.dsparkEvMissing.length)throw Error('dspark ev refs missing from payload: '+out.dsparkEvMissing.slice(0,5));
// 每条都真实点选渲染
let evRendered=0;
for(const id of evIds){
  context.window.showEv(id);
  if(els.get('evdetail').innerHTML.indexOf(id)<0)
    throw Error('showEv silent return for '+id);
  evRendered++;}
out.dsparkEvRendered=evRendered;
// ---- 3) 角色面板渲染（DSpark 图）----
const rb=P.dsparkView&&P.dsparkView.integration&&P.dsparkView.integration.role_bindings;
if(!rb)throw Error('role_bindings missing from payload');
const roleHtml=els.get('dsparkrolepanel').innerHTML;
out.rolePanel={
  nonEmpty:roleHtml.length>200,
  hasInstances:['mtp.0','mtp.1','mtp.2'].every(s=>roleHtml.indexOf(s)>=0),
  hasLayers:[43,44,45].every(n=>roleHtml.indexOf(String(n))>=0),
  hasFirst:/first（独占入口/.test(roleHtml),
  hasLast:/last（独占出口/.test(roleHtml),
  hasMiddle:/middle/.test(roleHtml),
  hasParamPaths:rb.template_param_bindings.every(pb=>
    pb.instances.every(x=>roleHtml.indexOf(x.instance_path)>=0)),
  has24Specs:rb.template_param_bindings.length>=24,
  hasInlineParamLabels:/内联算子参数/.test(roleHtml),
  hasQnormConsumer:roleHtml.indexOf('dspark.q_norm')>=0,
  hasExpertWildcard:/ffn\.experts\.\*\.w1\.weight/.test(roleHtml)||rb.template_param_bindings.some(pb=>pb.param_slot==='ffn.experts.*.w1.weight'),
  hasSharedEmbed:roleHtml.indexOf('entry.embed_w')>=0,
  hasSharedHead:roleHtml.indexOf('exit.head_w')>=0,
  hasCheckpointDtype:/checkpoint/.test(roleHtml),
  hasRuntimeDtype:/runtime/.test(roleHtml)};
if(!out.rolePanel.nonEmpty)throw Error('role panel empty on dspark graph');
if(!out.rolePanel.hasInstances||!out.rolePanel.hasLayers)throw Error('role panel missing instance/layer ids');
if(!out.rolePanel.hasParamPaths)throw Error('role panel missing param instance paths');
if(!out.rolePanel.has24Specs)throw Error('role panel <24 template param specs');
if(!out.rolePanel.hasInlineParamLabels)throw Error('role panel missing inline_op_param labels');
if(!out.rolePanel.hasQnormConsumer)throw Error('role panel missing q_norm consumer node');
// 张量链接可点开（联查当前图；tensorPanel 标题渲染 name 形式 = 去前缀
// stable_id，同时存在完整 stable_id 防伪——两者都核对）
const firstParamTb=rb.template_param_bindings.filter(pb=>pb.instances[0].graph_tensor)[0];
if(!firstParamTb)throw Error('no param_tensor binding in role panel');
const firstTensor=firstParamTb.instances[0].graph_tensor;
context.window.showTensor(firstTensor);
const tHtml=els.get('evdetail').innerHTML;
const tName=firstTensor.replace(/^tensor:/, '');
if(tHtml.indexOf(tName)<0)
  throw Error('role panel tensor link not clickable in current graph: '+firstTensor+' -> rendered: '+tHtml.slice(0,120));
out.roleTensorLinkWorks=true;
// ---- 4) 场景版本绑定渲染 + before 三态口径（codex_b4_review02 阻断3）----
out.scenarioStates={};
for(const sid of ['decode_p128','decode_p127','decode_p2','prefill_t3','prefill_t4','prefill_t5']){
  context.window.gSetScenario(sid);
  const sv=els.get('scenview').innerHTML;
  if(!/每实例 cache 状态版本绑定/.test(sv))
    throw Error('stage state table missing for '+sid);
  if(!/published/.test(sv))throw Error('published version missing '+sid);
  const checks={published:true,
    pPlusOne:/P\+1/.test(sv),
    isPrefill:/prefill/.test(sv)};
  // before_state_known 必须以三态文本出现（unknown 不转 bool false）
  if(!/unknown/.test(sv))throw Error('before tri-state chip missing '+sid);
  if(!/after 有效槽数/.test(sv))throw Error('after slot count column missing '+sid);
  if(!/显式假设/.test(sv))throw Error('history explicit-assumption wording missing '+sid);
  // 数量阈值不得写成历史已成立的断言（before/after 混淆）
  if(/前提成立/.test(sv))throw Error('stale premise-asserted wording in '+sid);
  if(sid==='decode_p127'){
    // P=127：前提文字必须按 P+1=128 口径，不得写 P=127 ≥ W
    if(/P=127 ≥ W/.test(sv))throw Error('p127 premise written as P>=W');
    checks.fullPremise=/full_premise/.test(sv);
    checks.assumptionWording=/显式假设.*P\+1=128/.test(sv.replace(/\n/g,' '))||/P\+1=128/.test(sv);}
  if(sid==='decode_p2')checks.partial=/partial/.test(sv);
  if(sid==='decode_p128')checks.fullPremise=/full_premise/.test(sv);
  if(sid==='prefill_t3')checks.unknown=/unknown/.test(sv);
  out.scenarioStates[sid]=checks;}
// ---- 5) 切回主图：角色面板恢复归属说明（不残留 DSpark 数据）----
vm.runInContext("gSetVariant('main')",context);
const backHtml=els.get('dsparkrolepanel').innerHTML;
out.switchBack={
  noDsparkData:backHtml.indexOf('mtp.0')<0||/属 DSpark 整体图/.test(backHtml),
  mentionsDspark:backHtml.indexOf('DSpark')>=0};
if(out.switchBack.noDsparkData!==true)throw Error('role panel leaked dspark data after switch back');
console.log(JSON.stringify(out));
'''

_PROBE_NAME = "b4_fix01_page_probe.cjs"


def _build_isolated_page(tmp_root: Path) -> tuple[Path, list[str]]:
    """用当前源码在临时隔离输出根真实 _build_documents 生成本轮测试页
    （codex_b4_review02 阻断4：不再扫真实 outputs 按 mtime 选旧/任意
    candidate——改了源码也必须测到新页；无已有 candidate 时不再
    ValueError）。不写真实 outputs、不需要服务/浏览器。返回 (页面路径,
    build 日志)。"""
    import unittest.mock as mock
    from structure_reader import cli as cli_mod
    from structure_reader.inventory import build_inventory
    from structure_reader.model_registry import find_workspace_root

    workspace_root = find_workspace_root()
    inv = build_inventory("deepseek_v4_flash_0731",
                          workspace_root=workspace_root)
    args = argparse.Namespace(
        model="deepseek_v4_flash_0731", workspace_root=str(workspace_root),
        scenario_set="standard", json=True)
    identity = cli_mod._build_identity_document(
        args.model, inv, workspace_root, args.scenario_set)
    analysis_id = cli_mod._sha256_bytes(cli_mod._canonical_json_bytes(identity))
    build_log: list[str] = []
    with mock.patch.object(cli_mod, "_outputs_root", return_value=tmp_root):
        docs = cli_mod._build_documents(args.model, workspace_root,
                                        args.scenario_set, analysis_id,
                                        inv, build_log)
    page_rel = "viewer/structure.html"
    page_path = tmp_root / page_rel
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.write_bytes(docs[page_rel])
    return page_path, build_log


@unittest.skipUnless(Path(NODE).exists(), "本机无 node；跳过 JS 执行回归")
class TestPageEvidenceClickAndRolePanel(unittest.TestCase):
    """证据点选 + 角色面板 + 场景版本绑定（JS 单元执行，非浏览器）。

    页面来自本测试进程内临时隔离输出根的真实 _build_documents（当前
    源码）；不扫真实 outputs、不依赖已有 candidate。"""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory(prefix="b4-fix02-page-")
        cls.out_dir = Path(cls._tmp.name) / "outputs" / "deepseek_v4_flash_0731"
        cls.html_path, cls.build_log = _build_isolated_page(
            Path(cls._tmp.name) / "outputs_root" / "deepseek_v4_flash_0731")
        probe_path = PROJECT / ".work" / _PROBE_NAME
        probe_path.write_text(_PROBE_CJS, encoding="utf-8")
        run = subprocess.run(
            [NODE, str(probe_path), str(cls.html_path),
             json.dumps(_ANCHOR_EV)],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT.parent))
        cls.probe_run = run
        if run.returncode == 0:
            cls.out = json.loads(run.stdout)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_probe_executes(self):
        self.assertEqual(self.probe_run.returncode, 0,
                         f"探针失败:\n{self.probe_run.stderr[:3000]}")

    def test_anchor_evidence_renders_source_snippet_claim(self):
        for tag, info in self.out["anchorEvidence"].items():
            with self.subTest(anchor=tag):
                self.assertTrue(info["rendered"])
                self.assertTrue(info["hasSource"], tag)
                self.assertTrue(info["hasLines"], tag)
                self.assertTrue(info["hasSnippet"], tag)
                self.assertTrue(info["hasClaim"], tag)

    def test_dspark_graph_evidence_fully_clickable(self):
        # 52 个图引用 ev ID 全部可解析可点选（此前 43 缺失 + showEv 直接
        # return 的丢证据路径已关闭）
        self.assertEqual(self.out["dsparkEvMissing"], [])
        self.assertGreaterEqual(self.out["dsparkEvRefs"], 52)
        self.assertEqual(self.out["dsparkEvRendered"],
                         self.out["dsparkEvRefs"])

    def test_role_panel_renders_bindings(self):
        rp = self.out["rolePanel"]
        self.assertTrue(rp["hasInstances"])
        self.assertTrue(rp["hasLayers"])
        self.assertTrue(rp["hasFirst"])
        self.assertTrue(rp["hasLast"])
        self.assertTrue(rp["hasParamPaths"])
        self.assertTrue(rp["has24Specs"])
        self.assertTrue(rp["hasInlineParamLabels"])
        self.assertTrue(rp["hasQnormConsumer"])
        self.assertTrue(rp["hasSharedEmbed"])
        self.assertTrue(rp["hasSharedHead"])
        self.assertTrue(rp["hasCheckpointDtype"])
        self.assertTrue(rp["hasRuntimeDtype"])
        self.assertTrue(self.out["roleTensorLinkWorks"])

    def test_scenario_stage_states_tri_state(self):
        st = self.out["scenarioStates"]
        for sid in ("decode_p128", "decode_p127", "decode_p2",
                    "prefill_t3", "prefill_t4", "prefill_t5"):
            with self.subTest(scenario=sid):
                self.assertTrue(st[sid]["published"])
                # P+1 数量口径只出现在 decode 场景（prefill 无 P 语义）
                if not st[sid]["isPrefill"]:
                    self.assertTrue(st[sid]["pPlusOne"], sid)
        self.assertTrue(st["decode_p127"]["fullPremise"])
        self.assertTrue(st["decode_p127"]["assumptionWording"])
        self.assertTrue(st["decode_p2"]["partial"])
        self.assertTrue(st["decode_p128"]["fullPremise"])
        self.assertTrue(st["prefill_t3"]["unknown"])

    def test_switch_back_clears_role_panel(self):
        self.assertTrue(self.out["switchBack"]["noDsparkData"])


if __name__ == "__main__":
    unittest.main()
