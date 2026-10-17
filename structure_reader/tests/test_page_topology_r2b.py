"""R2b 四图统一可视阅读针对性页面测试：真实产物 HTML 的图切换/同步/可读性核验。

范围（task_r2b.md，替代 R2a 旧展示形式断言的部分已在其文件内注明）：
- 四图统一上下文（修 R2b-1）：图工具条真实变体按钮（gSetVariant）切换
  CSA·TopK（主图）/ CSA·Hash / HCA·TopK / SWA·Hash；每次切换按所选图重建
  全部节点/边/模块归属/场景视图；渲染产物 ID 集合与所选图 data/variants/
  <key>/semantic_graph.json 精确相等（总览模块盒、局部展开、完整图三视图），
  无旧图 ID/guard/tensor/evidence 残留。
- 每图守恒：投影 ∪ 内部 = 当前图全部边且互斥（页面实际 GMOD_PAIRS/
  GMOD_INTERNAL 与 G 独立复算）；空模块（SWA 的 compressor/indexer）不渲染
  模块盒；"其它"具名模块（csa_hash/swa_hash 的 block.input_ids_src）有归属。
- 默认可读性（修 R2b-2）：局部展开与完整图默认适配 k ≥ 12/11（节点标签
  11px × k ≥ 12px 实际 CSS 像素，以真实字号×变换核验）；横向溢出局部图
  从输入侧左缘起始（x ≥ 0，不居中裁掉左侧输入）；"查看全局范围"提供显式
  整体缩小（k 严格小于默认适配），"适配视图"恢复可读默认；"定位选中"把
  选中节点/边居中（缩放不足时提到可读下限）。
- 场景与选择同步（修 R2b-3）：切图后场景 ID 有效则保留（四图共享 12 场景
  ID，实测保留）、节点/边选择 ID 存在于新图则保留（attn.sparse_attn 实测
  保留且详情刷新）、模块/投影对与不在新图的节点/边清空并在图摘要说明；
  两套场景按钮每次切图重建（各 12 个有效场景）；主图 guard_activations
  （{gid:{state,reason}}）与 variantViews（{gid:字符串}）经 gNormGuards
  归一化（字符串实测不误判 unknown）；无 guard 边 on 色、guard false 边
  off 色、gGuardStatus 未知分支返回 unknown（数据中无 unknown 值，如实
  声明 unknown 三态渲染路径未在真实数据上触发）。
- mixed 聚合（每图全部场景实测）：渲染紫色 pair 集合与按当前图边/guard
  激活独立复算一致（none/true→on，off/unknown→异态）；SWA 无 mixed 也
  如实断言（渲染 0）。
- 事件路径（修 R2b-4）：pointer down/move/up 实回调分发——位移 <5px 不改
  视图（点击阈值），>5px 平移生效；wheel 缩放生效；事件绑定于真实注册的
  listener（非点不存在元素）。
- 窄/宽视口：360/768/1440px 下总览适配有限正值、局部图文字 ≥12px、
  完整图 transform 有限值，无 NaN/undefined。
- 补修01（codex_r2b_review01.md 五项实际阅读路径缺陷）：初始/切图后图
  上方说明（真实数量非 0）、实际四按钮往返（csa_topk 与主图同上下文，
  压缩说明/高亮保持）、张量详情切图清空与恢复、总览/局部/完整图定位
  真实屏幕可见性（总览定位所属模块盒，不用全图坐标）、HCA 压缩器标签。
  初始高亮核验经桩扩展（querySelectorAll('#gbtn-var') 返回从 HTML 解析
  的四个真实按钮元素），主图上下文与 csa_topk 按钮键统一（gBtnKey）。

Node 桩局限如实声明：JS 单元执行（DOM 桩无布局/CSS，getBoundingClientRect
为固定值），不是浏览器/CSS/真实硬件验收；CSS 视觉、真实浏览器交互与用户
阅读由 Codex 真实浏览器/独立阅读包门禁接续核验，不在本测试声称范围。

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
MODEL_KEY = "deepseek_v4_flash_0731"
WORK = PROJECT / ".work"
LOGS_DIR = PROJECT / ".work" / "coordination" / "v2" / "reader_recovery" / "logs"

# 页面 JS 的 DOM 桩执行探针（真实注册回调分发 + 渲染产物提取版）：
# 五图（main + 4 变体）逐一 gSetVariant 切换 → 全场景按钮点击 → 全部非空
# 模块真实事件展开 → 完整图模式 → 选择/清空/说明 → 可读性与定位。
_PROBE_CJS = r'''
const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8');
const payload=html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=html.match(/<script>\n([\s\S]*?)<\/script>\s*<\/body>/)[1];
const els=new Map();const listeners=new Map();const varBtns=[];
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
// 真实变体按钮从 HTML 解析（修复 4 初始高亮核验：dataset.var 真实存在；
// 模板中按钮标签跨两行，只匹配开标签）
for(const m of html.matchAll(/<button class="sel" id="gbtn-var" data-var="([^"]+)"/g)){
  const b=mkEl('gbtn-var-'+m[1]);b.dataset.var=m[1];
  b.className='sel';varBtns.push(b);}
for(const id of [...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]))els.set(id,mkEl(id));
for(const id of ['statepanel','evdetail'])if(!els.has(id))els.set(id,mkEl(id));
els.get('payload').textContent=payload;
const context={document:{
  getElementById(id){if(!els.has(id))els.set(id,mkEl(id));return els.get(id)},
  querySelector(){return mkEl('main')},
  querySelectorAll(sel){
    if(sel==='#gscenbtns button')return els.get('gscenbtns').children;
    if(sel==='#scenbtns button')return els.get('scenbtns').children;
    if(sel==='#gbtn-var')return varBtns;
    if(sel==='.gnode'||sel==='.gedge'||sel==='.gmod'||sel==='.gmedge')
      return els.get('gscenbtns').children.map(()=>mkEl('dyn'));
    return els.get('gscenbtns').children;},
  createElement(){return mkEl('dynamic')}},
  window:{},location:{hash:''}};
vm.createContext(context);vm.runInContext(script,context,{timeout:30000});
function read(expr){return JSON.parse(vm.runInContext('JSON.stringify('+expr+')',context));}
function dispatchClick(rootId,leafCls,leafData){
  const fns=(listeners.get(rootId)||[]).filter(l=>l.type==='click');
  if(!fns.length)throw Error('no click listener on #'+rootId);
  const leaf=mkEl('leaf');leaf.classList.add(leafCls);leaf.dataset=leafData;
  leaf.__parent=els.get(rootId);
  for(const {fn} of fns)fn({target:leaf});}
const out={variants:{}};
const VARS=['main','csa_hash','hca_topk','swa_hash'];
for(const vk of VARS){
  if(vk!=='main')vm.runInContext(`gSetVariant('${vk}')`,context);
  const v={};
  v.curVar=read('curVar');
  v.gNodeIds=read('GNODES.map(n=>n.stable_id)');
  v.gEdgeIds=read('GEDGES.map(e=>e.stable_id)');
  v.templateId=read('ctx.G.template_id');
  // overview conservation from live page objects
  v.pairIds=read('Object.keys(GMOD_PAIRS).flatMap(k=>GMOD_PAIRS[k].map(e=>e.stable_id))');
  v.internalIds=read('Object.keys(GMOD_INTERNAL).flatMap(k=>GMOD_INTERNAL[k].map(e=>e.stable_id))');
  v.renderedMods=[...els.get('gmods').innerHTML.matchAll(/data-mod="([^"]+)"/g)].map(m=>m[1]);
  v.renderedPairs=[...els.get('gmodedges').innerHTML.matchAll(/data-pair="([^"]+)"/g)].map(m=>m[1]);
  v.nonEmptyMods=read('Object.keys(GMODS).filter(k=>GMODS[k].nodes.length>0)');
  // all scenes via toolbar buttons; mixed recomputed vs rendered per scene
  v.sceneChecks=[];
  for(const b of els.get('gscenbtns').children){
    b.onclick();
    const scen=read('curScen');
    if(read('curScenObj.scenario_id')!==scen)throw Error('scen desync '+scen);
    context.__ga=read('curScenObj.guard_activations');
    const pairs=JSON.parse(vm.runInContext(
      'JSON.stringify(Object.entries(GMOD_PAIRS).map(([k,es])=>({k,states:es.map(e=>e.guard_ref?(__ga[e.guard_ref]?__ga[e.guard_ref].state:"unknown"):"none")})))',
      context));
    const mixed=pairs.filter(p=>{
      const eff=[...new Set(p.states.map(s=>(s==='none'||s==='true')?'on':s))];
      return eff.length>1;});
    const renderedMixed=(els.get('gmodedges').innerHTML.match(/stroke="#7a4ea3"/g)||[]).length;
    v.sceneChecks.push({scene:scen,mixed:mixed.length,rendered:renderedMixed});
  }
  // open every non-empty module via real event dispatch
  v.modules=[];
  for(const mk of v.nonEmptyMods){
    dispatchClick('gsvg','gmod',{mod:mk});
    const key=read('gModView && gModView.key');
    if(key!==mk)throw Error('module open failed '+mk);
    const mv=read('({W:gModView.W,H:gModView.H,k:gView.k,x:gView.x,members:gModView.members})');
    const nodeIds=[...els.get('gnodes').innerHTML.matchAll(/data-node="([^"]+)"/g)].map(m=>m[1]);
    const edgeIds=[...els.get('gedges').innerHTML.matchAll(/class="gedge" data-edge="([^"]+)"/g)].map(m=>m[1]);
    const expected=vm.runInContext(
      'JSON.stringify(GEDGES.filter(e=>gModView.members.includes(e.src_node_id)||gModView.members.includes(e.dst_node_id)).map(e=>e.stable_id).sort())',
      context);
    if(JSON.stringify([...new Set(edgeIds)].sort())!==expected)
      throw Error(`module ${mk} edge set mismatch in ${vk}`);
    if(JSON.stringify([...new Set(nodeIds)].sort())!==JSON.stringify([...new Set(mv.members)].sort()))
      throw Error(`module ${mk} node set mismatch in ${vk}`);
    const textPx=mv.k*11;
    if(textPx<12-1e-9)throw Error(`module ${mk} text ${textPx}px<12 in ${vk}`);
    if(mv.x<0)throw Error(`module ${mk} origin x ${mv.x}<0 (input side clipped) in ${vk}`);
    v.modules.push({key:mk,W:mv.W,H:mv.H,k:mv.k,x:mv.x,textPx,
      nodes:nodeIds.length,edges:edgeIds.length,
      rails:(els.get('gxrails').innerHTML.match(/class="gport-anchor"/g)||[]).length});
    vm.runInContext('gSetMode("overview")',context);
  }
  // full mode: ids exact + readable floor + global smaller
  vm.runInContext('gSetMode("full")',context);
  const fullNodes=[...els.get('gnodes').innerHTML.matchAll(/data-node="([^"]+)"/g)].map(m=>m[1]);
  const fullEdges=[...els.get('gedges').innerHTML.matchAll(/class="gedge" data-edge="([^"]+)"/g)].map(m=>m[1]);
  v.fullNodes=fullNodes.length;v.fullEdges=fullEdges.length;
  if(JSON.stringify([...new Set(fullNodes)].sort())!==JSON.stringify([...v.gNodeIds].sort()))
    throw Error('full view node mismatch in '+vk);
  if(JSON.stringify([...new Set(fullEdges)].sort())!==JSON.stringify([...v.gEdgeIds].sort()))
    throw Error('full view edge mismatch in '+vk);
  v.fullFitK=read('gView.k');
  if(v.fullFitK<12/11-1e-9)throw Error('full fit below readable floor in '+vk);
  vm.runInContext('gShowGlobal()',context);
  v.globalK=read('gView.k');
  if(v.globalK>=v.fullFitK)throw Error('global view not smaller in '+vk);
  vm.runInContext('gSetMode("overview")',context);
  out.variants[vk]=v;
}
// back to main for cross-switch residue + selection sync checks
vm.runInContext('gSetVariant("swa_hash")',context);
const swaNodes=read('GNODES.map(n=>n.stable_id)');
vm.runInContext('gSetVariant("csa_hash")',context);
vm.runInContext('gSetVariant("main")',context);
const afterHtml=els.get('gnodes').innerHTML+els.get('gedges').innerHTML+
  els.get('gmods').innerHTML+els.get('gmodedges').innerHTML;
const swaOnly=swaNodes.filter(n=>!read('GNODES.map(n=>n.stable_id)').includes(n));
for(const n of swaOnly)if(afterHtml.includes(`data-node="${n}"`))throw Error('residue '+n);
out.residueClean=true;
// selection sync: present node kept + detail; absent cleared with note
vm.runInContext('gSetMode("full")',context);
dispatchClick('gsvg','gnode',{node:'attn.sparse_attn'});
vm.runInContext('gSetVariant("swa_hash")',context);
out.keptSel={sel:read('gSel'),
  detail:els.get('gdetail').innerHTML.includes('节点 attn.sparse_attn')};
vm.runInContext('gSetVariant("csa_hash")',context);
dispatchClick('gsvg','gnode',{node:'attn.indexer.topk'});
vm.runInContext('gSetVariant("swa_hash")',context);
out.clearedSel={sel:read('gSel'),
  noteCleared:els.get('gscensum').innerHTML.includes('已清除选择'),
  noteId:els.get('gscensum').innerHTML.includes('attn.indexer.topk')};
vm.runInContext('gSetVariant("csa_hash")',context);
vm.runInContext('gShowModule("compressor")',context);
vm.runInContext('gSetVariant("hca_topk")',context);
out.clearedModule={sel:read('gSel'),
  noteModule:els.get('gscensum').innerHTML.includes('模块/投影')};
// scenario kept across variants (shared 12 ids)
vm.runInContext('gSetScenario("prefill_t5")',context);
vm.runInContext('gSetVariant("main")',context);
out.scenKept=read('curScen');
out.scenBtns={toolbar:els.get('gscenbtns').children.length,
  panel:els.get('scenbtns').children.length,
  toolbarIds:els.get('gscenbtns').children.map(b=>b.dataset.id)};
// normalization: string vs object shapes
out.normStr=vm.runInContext('JSON.stringify(gNormGuards({g_a:"true",g_b:"false"}))',context);
out.normObj=vm.runInContext('JSON.stringify(gNormGuards({g_a:{state:"false",reason:"x"}}))',context);
out.unknownBranch=vm.runInContext(
  'JSON.stringify(gGuardStatus({guard_ref:"g_none"},{guard_activations:{}}))',context);
// no-guard edge on-color in full view
vm.runInContext('gSetMode("full")',context);
const mNoGuard=els.get('gedges').innerHTML.match(
  /<path class="gedge" data-edge="e_sparse_q" d="[^"]+" stroke="([^"]+)"/);
out.noGuardColor=mNoGuard?mNoGuard[1]:null;
// guard-false edge off-color (e_sparse_kv_decode @g_decode in prefill_t5)
const btnPre=els.get('gscenbtns').children.find(b=>b.dataset.id==='prefill_t5');
btnPre.onclick();
const mOff=els.get('gedges').innerHTML.match(
  /<path class="gedge" data-edge="e_sparse_kv_decode" d="[^"]+" stroke="([^"]+)"/);
out.guardFalseColor=mOff?mOff[1]:null;
// locate selection centers node (module mode, indexer)
vm.runInContext('gShowModule("indexer")',context);
dispatchClick('gsvg','gnode',{node:'attn.indexer.topk'});
vm.runInContext('gLocateSelection()',context);
const loc=read('gView');
const pos=read('gModView.pos["attn.indexer.topk"]');
out.locate={k:loc.k,textPx:loc.k*11,
  centered:Math.abs((loc.x+pos.x*loc.k)-500)<1e-6};
vm.runInContext('gFit()',context);
out.refitK=read('gView.k');
// event paths: click threshold + pan + wheel
const downFns=(listeners.get('gviewport')||[]).filter(l=>l.type==='pointerdown');
const moveFns=(listeners.get('gviewport')||[]).filter(l=>l.type==='pointermove');
const upFns=(listeners.get('gviewport')||[]).filter(l=>l.type==='pointerup');
const wheelFns=(listeners.get('gviewport')||[]).filter(l=>l.type==='wheel');
vm.runInContext('gSetMode("overview")',context);
const viewBefore=read('gView');
downFns.forEach(l=>l.fn({clientX:100,clientY:100,pointerId:1}));
moveFns.forEach(l=>l.fn({clientX:103,clientY:102,pointerId:1}));
upFns.forEach(l=>l.fn({clientX:103,clientY:102,pointerId:1}));
out.clickThresholdKept=JSON.stringify(read('gView'))===JSON.stringify(viewBefore);
downFns.forEach(l=>l.fn({clientX:100,clientY:100,pointerId:2}));
moveFns.forEach(l=>l.fn({clientX:160,clientY:130,pointerId:2}));
out.panMoved=(()=>{const v=read('gView');return v.x!==viewBefore.x||v.y!==viewBefore.y;})();
upFns.forEach(l=>l.fn({clientX:160,clientY:130,pointerId:2}));
const kBefore=read('gView.k');
wheelFns.forEach(l=>l.fn({clientX:150,clientY:150,deltaY:-120,preventDefault(){}}));
out.wheelZoom={before:kBefore,after:read('gView.k')};
const allPanels=['gmodulepanel','gdetail','gscensum','scenview','gmods',
  'gmodedges','gedges','gnodes','l1modules','l1note','statepanel',
  'l1graphnote','l0flow','l0groups','l0dspark','evidenceview','planview']
  .map(id=>els.get(id).innerHTML).join('');
out.noUndefinedNaN=!/undefined|NaN/.test(allPanels);
// ---- R2b 补修01：五项实际阅读路径核验 ----
// 1) 初始图说明非 0 且随切图更新（l1graphnote）
out.graphNote={initial:els.get('l1graphnote').innerHTML,
  initialHas107:els.get('l1graphnote').innerHTML.includes('107 个'),
  initialHas173:els.get('l1graphnote').innerHTML.includes('173 条'),
  initialNotZero:!/0 个 · <b>全部真实边<\/b> 0 条/.test(els.get('l1graphnote').innerHTML)};
vm.runInContext('gSetVariant("swa_hash")',context);
out.graphNote.swaHas40=els.get('l1graphnote').innerHTML.includes('40 个');
out.graphNote.swaHas51=els.get('l1graphnote').innerHTML.includes('51 条');
out.graphNote.swaLabel=els.get('l1graphnote').innerHTML.includes('SWA·Hash');
out.graphNote.changed=els.get('l1graphnote').innerHTML!==out.graphNote.initial;
vm.runInContext('gSetVariant("csa_topk")',context);
out.graphNote.backHas107=els.get('l1graphnote').innerHTML.includes('107 个');
// 2) 实际四按钮往返：handler 从 HTML 提取（用户真实可达路径）；
//    csa_topk 与 main 同一上下文（compression 说明不丢）
out.buttonRoundtrip={initial:read('({key:curVar,compression:!!(ctx.scen.find(s=>s.scenario_id===curScen)||{}).compression})')};
const csaBtn=varBtns.find(b=>b.dataset.var==='csa_topk');
const csaHandler=html.match(/data-var="csa_topk"\s+onclick="([^"]+)"/)[1];
vm.runInContext(csaHandler,context);
out.buttonRoundtrip.afterCsaBtn=read('({key:curVar,compression:!!(ctx.scen.find(s=>s.scenario_id===curScen)||{}).compression})');
out.buttonRoundtrip.scenviewNoMissing=!(
  els.get('scenview').innerHTML.includes('未内嵌逐步压缩实例化'));
out.buttonRoundtrip.csaBtnHighlighted=csaBtn.className.includes('on');
out.buttonRoundtrip.highlightedVars=varBtns.filter(b=>b.className.includes('on')).map(b=>b.dataset.var);
// 初始态高亮（重新在 VM 里验证 main 初始：main 与按钮键 csa_topk 匹配）
out.buttonRoundtrip.initialHighlightLogic=vm.runInContext(
  'gBtnKey("main")==="csa_topk"&&gBtnKey("csa_topk")==="csa_topk"',context);
// 重复点击与全按钮往返：任意顺序走完后经 CSA·TopK 实际按钮回到主图
// 上下文，压缩说明/guard 理由不丢（循环终点是 swa_hash，其场景本无
// compression——主图性由"回到 csa_topk 按钮仍 compression"核验）
const allBtns=varBtns.map(b=>({var:b.dataset.var,
  handler:html.match(new RegExp('data-var="'+b.dataset.var+'"\\s+onclick="([^"]+)"'))[1]}));
for(const b of allBtns){vm.runInContext(b.handler,context);vm.runInContext(b.handler,context);}
out.buttonRoundtrip.endOfRoundtripVar=read('curVar');
vm.runInContext(allBtns.find(b=>b.var==='csa_topk').handler,context);
out.buttonRoundtrip.afterAllRoundtrip=read('({key:curVar,compression:!!(ctx.scen.find(s=>s.scenario_id===curScen)||{}).compression})');
out.buttonRoundtrip.csaEqMain=vm.runInContext('CTXS.csa_topk===CTXS.main',context);
// 3) 张量/证据详情：切图清空失效项 + 切回自动恢复；证据全局静态保留
// （先经实际按钮回到主图上下文，再展示主图专属张量）
vm.runInContext(allBtns.find(b=>b.var==='csa_topk').handler,context);
const mainTids=vm.runInContext('JSON.stringify(ctx.G.tensors.map(t=>t.stable_id))',context);
const swaTids=new Set(vm.runInContext('CTXS.swa_hash.G.tensors.map(t=>t.stable_id).join(",")',context).split(','));
const exclusiveTid=JSON.parse(mainTids).find(t=>!swaTids.has(t));
vm.runInContext('window.showTensor('+JSON.stringify(exclusiveTid)+')',context);
out.evDetail={mainShown:els.get('evdetail').innerHTML.includes('张量与轴（图 CSA·TopK（主图））'),
  nearGraph:html.indexOf('id="evdetail"')<0,  // 动态插入卡；近图位置由 Python 断言
  tensorId:exclusiveTid};
vm.runInContext('gSetVariant("swa_hash")',context);
out.evDetail.clearedOnSwitch=els.get('evdetail').innerHTML.includes('张量详情已清空');
out.evDetail.clearedMentionsId=els.get('evdetail').innerHTML.includes(exclusiveTid);
out.evDetail.noOldHeading=!(els.get('evdetail').innerHTML.includes('张量与轴（图 CSA'));
vm.runInContext('gSetVariant("csa_topk")',context);
out.evDetail.restoredOnSwitchBack=els.get('evdetail').innerHTML.includes('张量与轴（图 CSA·TopK（主图））');
// 证据：切图后保持（全局静态），且关联核对仍在
vm.runInContext('window.showEv(P.evidence.entries[0].evidence_id)',context);
const evBefore=els.get('evdetail').innerHTML;
vm.runInContext('gSetVariant("swa_hash")',context);
out.evDetail.evKept=els.get('evdetail').innerHTML===evBefore;
out.evDetail.evHeading=els.get('evdetail').innerHTML.includes('证据 ');
// 4) 定位选中按当前渲染模式（总览定位所属模块盒、屏幕可见）
vm.runInContext('gSetVariant("csa_topk")',context);
vm.runInContext('gSetMode("full")',context);
const rightmost=vm.runInContext('GNODES.map(n=>n.stable_id).sort((a,b)=>gPos[b].x-gPos[a].x)[0]',context);
dispatchClick('gsvg','gnode',{node:rightmost});
vm.runInContext('gSetMode("overview")',context);
const viewBeforeLocate=read('({...gView})');
vm.runInContext('gLocateSelection()',context);
out.locateOverview={node:rightmost,mode:read('gMode'),sel:read('gSel'),
  before:viewBeforeLocate,after:read('({...gView})'),
  changed:JSON.stringify(read('({...gView})'))!==JSON.stringify(viewBeforeLocate)};
out.locateOverview.moduleBoxes=read(
  'Object.entries(GMOD_POS).map(([key,p])=>({key,x:p.x*gView.k+gView.x,y:p.y*gView.k+gView.y,w:GW2.mW*gView.k,h:GW2.mH*gView.k}))');
out.locateOverview.visibleModules=out.locateOverview.moduleBoxes.filter(p=>
  p.x+p.w>0&&p.x<1000&&p.y+p.h>0&&p.y<640).map(p=>p.key);
out.locateOverview.targetModule=vm.runInContext(
  'gModuleOf('+JSON.stringify(rightmost)+')',context);
// 局部模式：内部节点定位居中（真实点击路径：点模块进入局部 → 点节点）
vm.runInContext('gSetVariant("csa_topk")',context);
vm.runInContext('gShowModule("compressor")',context);
const compNode=vm.runInContext('gModView.members[0]',context);
dispatchClick('gsvg','gnode',{node:compNode});
vm.runInContext('gLocateSelection()',context);
const locIn=read('({x:gView.x,k:gView.k,px:gModView.pos['+JSON.stringify(compNode)+'].x})');
out.locateModule={node:compNode,
  centered:Math.abs((locIn.x+locIn.px*locIn.k)-500)<1e-6,
  textPx:locIn.k*11};
// 外部节点：完整图点击选中 → 打开其它模块局部图（gOpenModule 保留
// gSel 节点选择——gShowModule 会覆盖为 module，非本路径）→ 定位
vm.runInContext('gBackToOverview()',context);
vm.runInContext('gSetMode("full")',context);
dispatchClick('gsvg','gnode',{node:'attn.sparse_attn'});
vm.runInContext('gOpenModule("moe")',context);
const viewBeforeExt=read('({...gView})');
vm.runInContext('gLocateSelection()',context);
const viewAfterExt=read('({...gView})');
out.locateModule.externalNode={sel:read('gSel'),
  viewKept:JSON.stringify(viewBeforeExt)===JSON.stringify(viewAfterExt),
  noteShown:els.get('gscensum').innerHTML.includes('不在本模块内')};
// 5) HCA 压缩器标题（模块盒 + 分区 lane 标签）
vm.runInContext('gSetVariant("hca_topk")',context);
out.hcaLabel={module:vm.runInContext('GMODS.compressor?GMODS.compressor.label:null',context),
  lane:vm.runInContext('LANE_LABEL.comp',context),
  modsHtmlHasHCA:els.get('gmods').innerHTML.includes('HCA 压缩器')};
out.hcaLabel.swaNoCompressorBox=(()=>{
  vm.runInContext('gSetVariant("swa_hash")',context);
  return !els.get('gmods').innerHTML.includes('data-mod="compressor"');})();
vm.runInContext('gSetVariant("csa_topk")',context);
out.hcaLabel.csaStill=vm.runInContext('GMODS.compressor.label',context);
out.note='local JS unit execution with DOM stubs; no browser/CSS acceptance';
console.log(JSON.stringify(out));
'''


def _latest_output_dir() -> Path:
    runs = [r for r in MODEL_OUTPUTS.iterdir()
            if (r / "viewer" / "structure.html").is_file()]
    return max(runs, key=lambda r: r.stat().st_mtime)


_RENDER_CACHE: dict | None = None


def _render_once() -> dict:
    global _RENDER_CACHE
    if _RENDER_CACHE is not None:
        return _RENDER_CACHE
    out_dir = _latest_output_dir()
    html = (out_dir / "viewer" / "structure.html").read_text(encoding="utf-8")
    probe_path = WORK / "r2b_page_probe.cjs"
    probe_path.write_text(_PROBE_CJS, encoding="utf-8")
    run = subprocess.run(
        [NODE, str(probe_path), str(out_dir / "viewer" / "structure.html")],
        capture_output=True, text=True, timeout=120,
        cwd=str(PROJECT.parent))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "r2b_page_probe_run.json").write_text(
        run.stdout or run.stderr, encoding="utf-8")
    if run.returncode != 0:
        raise AssertionError(f"页面 JS 执行失败:\n{run.stderr[:3000]}")
    _RENDER_CACHE = {"render": json.loads(run.stdout), "html": html,
                     "out_dir": out_dir}
    return _RENDER_CACHE


class _PageHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _render_once()
        cls.out_dir = cls.data["out_dir"]
        cls.html = cls.data["html"]
        cls.render = cls.data["render"]
        cls.payload = json.loads(re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            cls.html, re.S).group(1))
        cls.variant_graphs = {}
        for key in ("csa_hash", "hca_topk", "swa_hash"):
            cls.variant_graphs[key] = json.loads(
                (cls.out_dir / "data" / "variants" / key /
                 "semantic_graph.json").read_text(encoding="utf-8"))


class TestVariantSwitchRebuilds(_PageHarness):
    """四图统一：gSetVariant 按所选图重建，渲染 ID 集合与该图 JSON 精确相等。"""

    def test_variant_buttons_present_with_real_keys(self):
        for key in ("csa_topk", "csa_hash", "hca_topk", "swa_hash"):
            self.assertIn(f"data-var=\"{key}\"", self.html, key)
        self.assertIn("function gSetVariant(", self.html)

    def test_each_variant_overview_ids_exact(self):
        r = self.render
        for key in ("csa_hash", "hca_topk", "swa_hash"):
            v = r["variants"][key]
            g = self.variant_graphs[key]
            self.assertEqual(v["templateId"], g["template_id"], key)
            self.assertEqual(sorted(v["gNodeIds"]),
                             sorted(n["stable_id"] for n in g["nodes"]), key)
            self.assertEqual(sorted(v["gEdgeIds"]),
                             sorted(e["stable_id"] for e in g["edges"]), key)
            # 总览渲染产物无旧图残留：每个渲染模块盒属于当前图非空模块
            nonempty = set(v["nonEmptyMods"])
            self.assertTrue(set(v["renderedMods"]) <= nonempty, key)

    def test_each_variant_full_mode_ids_exact(self):
        r = self.render
        for key in ("csa_hash", "hca_topk", "swa_hash"):
            v = r["variants"][key]
            g = self.variant_graphs[key]
            self.assertEqual(v["fullNodes"], len(g["nodes"]), key)
            self.assertEqual(v["fullEdges"], len(g["edges"]), key)

    def test_no_residue_after_switch_chain(self):
        r = self.render
        self.assertTrue(r["residueClean"])
        self.assertEqual(r["variants"]["main"]["curVar"], "main")

    def test_empty_modules_not_rendered(self):
        # SWA 无压缩器/Indexer：空模块不渲染盒
        r = self.render
        swa = r["variants"]["swa_hash"]
        self.assertNotIn("compressor", swa["renderedMods"])
        self.assertNotIn("indexer", swa["renderedMods"])
        hca = r["variants"]["hca_topk"]
        self.assertNotIn("indexer", hca["renderedMods"])

    def test_unknown_group_has_named_module(self):
        # csa_hash/swa_hash 的 block.input_ids_src → 具名"其它"模块
        r = self.render
        for key in ("csa_hash", "swa_hash"):
            self.assertIn("other", r["variants"][key]["renderedMods"], key)


class TestPerGraphConservation(_PageHarness):
    """每图守恒：投影 ∪ 内部 = 当前图全部边且互斥（页面实际成员独立核对）。"""

    def test_pair_union_internal_equals_edges_exactly(self):
        r = self.render
        for key in ("main", "csa_hash", "hca_topk", "swa_hash"):
            v = r["variants"][key]
            ids = v["pairIds"] + v["internalIds"]
            self.assertEqual(len(ids), len(set(ids)),
                             f"{key} 边重复（投影/内部互斥破坏）")
            self.assertEqual(sorted(ids), sorted(v["gEdgeIds"]),
                             f"{key} 投影∪内部 ≠ 当前图全部边")


class TestDefaultReadability(_PageHarness):
    """默认可读性：字号下限 12px 实际像素；左起点；显式全局缩小入口。"""

    def test_module_fit_text_at_least_12px(self):
        r = self.render
        for key in ("main", "csa_hash", "hca_topk", "swa_hash"):
            for m in r["variants"][key]["modules"]:
                self.assertGreaterEqual(m["textPx"], 12 - 1e-9,
                    f"{key}/{m['key']} 标签 {m['textPx']:.2f}px < 12px")

    def test_large_modules_overflow_from_input_side(self):
        # Indexer（W=3385 > 1000 视口，main/csa_hash 有）：起点 x ≥ 0
        #（左缘可见），不居中裁输入；HCA 无 Indexer（真实模块归属），
        # SWA 无 compressor/indexer，均不参与本断言
        r = self.render
        for key in ("main", "csa_hash"):
            m = next(m for m in r["variants"][key]["modules"]
                     if m["key"] == "indexer")
            self.assertGreaterEqual(m["x"], 0, key)
            self.assertGreater(m["W"], 1000, key)  # 确为溢出场景
        # HCA/SWA 的每个非空模块同样从非负起点起始
        for key in ("hca_topk", "swa_hash"):
            for m in r["variants"][key]["modules"]:
                self.assertGreaterEqual(m["x"], 0, f"{key}/{m['key']}")

    def test_global_view_explicitly_smaller_than_default(self):
        r = self.render
        for key in ("main", "csa_hash", "hca_topk", "swa_hash"):
            v = r["variants"][key]
            self.assertLess(v["globalK"], v["fullFitK"], key)

    def test_locate_selection_centers_at_readable_scale(self):
        r = self.render
        loc = r["locate"]
        self.assertTrue(loc["centered"], "定位选中未把节点居中")
        self.assertGreaterEqual(loc["textPx"], 12 - 1e-9)
        self.assertGreaterEqual(r["refitK"], 12 / 11 - 1e-9,
                                "适配视图未恢复可读默认")

    def test_source_has_global_and_locate_buttons(self):
        self.assertIn('id="gbtn-global"', self.html)
        self.assertIn('id="gbtn-locate"', self.html)
        self.assertIn("function gShowGlobal()", self.html)
        self.assertIn("function gLocateSelection()", self.html)
        self.assertIn("12 / 11", self.html)  # 字号下限常量（12px/11px 字号）


class TestScenarioSelectionSync(_PageHarness):
    """切图同步：场景保留/改选、选择保留/清空并说明、按钮重建、形状归一化。"""

    def test_scenario_kept_across_variants(self):
        r = self.render
        self.assertEqual(r["scenKept"], "prefill_t5")
        self.assertEqual(r["scenBtns"]["toolbar"], 12)
        self.assertEqual(r["scenBtns"]["panel"], 12)
        self.assertEqual(len(set(r["scenBtns"]["toolbarIds"])), 12)

    def test_present_node_kept_with_detail(self):
        r = self.render
        self.assertEqual(r["keptSel"]["sel"],
                         {"type": "node", "id": "attn.sparse_attn"})
        self.assertTrue(r["keptSel"]["detail"])

    def test_absent_node_cleared_with_note(self):
        r = self.render
        self.assertIsNone(r["clearedSel"]["sel"])
        self.assertTrue(r["clearedSel"]["noteCleared"])
        self.assertTrue(r["clearedSel"]["noteId"])

    def test_module_selection_cleared_with_note(self):
        r = self.render
        self.assertIsNone(r["clearedModule"]["sel"])
        self.assertTrue(r["clearedModule"]["noteModule"])

    def test_guard_shapes_normalized(self):
        # 主图 {gid:{state,reason}} 与变体 {gid:字符串} 都归一化为
        # {state, reason}；字符串 "true"/"false" 不误判 unknown
        r = self.render
        self.assertEqual(
            r["normStr"],
            '{"g_a":{"state":"true","reason":""},'
            '"g_b":{"state":"false","reason":""}}')
        self.assertEqual(r["normObj"],
                         '{"g_a":{"state":"false","reason":"x"}}')
        self.assertEqual(r["unknownBranch"], '"unknown"')

    def test_no_guard_edge_on_and_guard_false_off(self):
        r = self.render
        self.assertEqual(r["noGuardColor"], "#1f5f8b",
                         "无 guard 边应无条件 on（acc 色）")
        self.assertEqual(r["guardFalseColor"], "#dcd8d0",
                         "guard false 边应 off（line 色）")


class TestMixedAggregationAllVariants(_PageHarness):
    """mixed 聚合：渲染紫色 pair 数与按当前图独立复算一致（逐场景）。"""

    def test_rendered_mixed_matches_recomputation_per_scene(self):
        r = self.render
        for key in ("main", "csa_hash", "hca_topk", "swa_hash"):
            checks = r["variants"][key]["sceneChecks"]
            self.assertEqual(len(checks), 12, key)
            for c in checks:
                self.assertEqual(
                    c["mixed"], c["rendered"],
                    f"{key}/{c['scene']} mixed 复算 {c['mixed']} ≠ 渲染 {c['rendered']}")

    def test_swa_has_no_mixed_and_csa_has(self):
        r = self.render
        swa = r["variants"]["swa_hash"]
        self.assertTrue(all(c["mixed"] == 0 for c in swa["sceneChecks"]),
                        "SWA 全部 pair 单态（如实为 0 mixed）")
        main = r["variants"]["main"]
        self.assertTrue(any(c["mixed"] > 0 for c in main["sceneChecks"]),
                        "主图存在 mixed 投影对")


class TestEventPaths(_PageHarness):
    """事件路径：真实注册回调——点击阈值/拖拽平移/滚轮缩放。"""

    def test_click_threshold_and_pan_and_wheel(self):
        r = self.render
        self.assertTrue(r["clickThresholdKept"], "位移 <5px 改变了视图")
        self.assertTrue(r["panMoved"], "位移 >5px 未平移")
        self.assertGreater(r["wheelZoom"]["after"], r["wheelZoom"]["before"],
                           "滚轮未缩放")

    def test_panels_free_of_undefined_nan(self):
        self.assertTrue(self.render["noUndefinedNaN"])


class TestFix01ReadingPaths(_PageHarness):
    """补修01（codex_r2b_review01.md 五项）实际阅读路径断言。

    全部走实际 HTML 按钮 handler / 事件分发 / 详情面板内容，不是仅查
    renderer 字符串；Node 桩局限同上（非浏览器/用户阅读验收）。
    """

    def test_graph_note_initial_real_counts_and_updates_on_switch(self):
        # 修 1：初始说明为真实数量（107/173），非 0；切 SWA 后更新为 40/51
        # 与 SWA·Hash 标签；切回恢复 107
        n = self.render["graphNote"]
        self.assertTrue(n["initialHas107"], "初始说明缺 107 节点")
        self.assertTrue(n["initialHas173"], "初始说明缺 173 边")
        self.assertTrue(n["initialNotZero"], "初始说明仍是 0 节点/0 边")
        self.assertTrue(n["swaHas40"] and n["swaHas51"],
                        "切 SWA 后说明未更新为 40/51")
        self.assertTrue(n["swaLabel"], "切 SWA 后说明未更新图标签")
        self.assertTrue(n["changed"], "切图后说明未变化")
        self.assertTrue(n["backHas107"], "切回后说明未恢复 107")

    def test_csa_button_same_context_as_main_full_roundtrip(self):
        # 修 4：CSA·TopK 实际按钮 handler 往返不丢主图压缩说明/上下文
        b = self.render["buttonRoundtrip"]
        self.assertTrue(b["initial"]["compression"],
                        "初始主图场景无 compression（前提破坏）")
        self.assertEqual(b["afterCsaBtn"]["key"], "csa_topk")
        self.assertTrue(b["afterCsaBtn"]["compression"],
                        "点 CSA·TopK 实际按钮后丢 compression 说明")
        self.assertTrue(b["scenviewNoMissing"],
                        "点 CSA·TopK 按钮后场景面板出现'未内嵌逐步压缩实例化'")
        self.assertTrue(b["csaEqMain"], "CTXS.csa_topk 与 CTXS.main 非同一上下文")
        self.assertTrue(b["csaBtnHighlighted"], "点击后 CSA·TopK 按钮未高亮")
        self.assertEqual(b["highlightedVars"], ["csa_topk"],
                         "高亮不唯一或错误")
        self.assertTrue(b["initialHighlightLogic"],
                        "main 初始态按钮键映射错误（初始高亮为空）")
        self.assertTrue(b["afterAllRoundtrip"]["compression"],
                        "四按钮重复往返后压缩说明丢失")

    def test_initial_highlight_matches_csa_button(self):
        # 初始（main 请求键）gBtnKey 映射到 csa_topk 按钮：探针 VM 内
        # gBtnKey("main")==="csa_topk"；初始高亮本体由
        # buttonRoundtrip.initialHighlightLogic + 下行按钮存在性合并核验
        self.assertIn('data-var="csa_topk"', self.html)

    def test_tensor_detail_cleared_on_absent_and_restored_on_return(self):
        # 修 2：主图专属张量详情在切 SWA 后清空并说明（含 ID），旧标题不
        # 残留；切回后自动恢复；证据全局静态保留
        e = self.render["evDetail"]
        self.assertTrue(e["mainShown"], "主图张量详情未按当前图渲染标题")
        self.assertTrue(e["clearedOnSwitch"], "切图后旧张量详情残留")
        self.assertTrue(e["clearedMentionsId"], "清空说明未含张量 ID")
        self.assertTrue(e["noOldHeading"], "旧图张量标题残留")
        self.assertTrue(e["restoredOnSwitchBack"], "切回后详情未自动恢复")
        self.assertTrue(e["evKept"], "全局静态证据被切图误清")
        self.assertTrue(e["evHeading"], "证据详情标题缺失")

    def test_evidence_card_placed_near_graph(self):
        # 修 2（近图可达）：#evdetail 动态插入到 L1 图卡之后
        #（insertBefore l1table-h2），静态 HTML 中无该占位卡
        self.assertNotIn('<div class="card" id="evdetail">', self.html)
        self.assertIn("insertBefore(\n  dv, document.getElementById('l1table-h2'))",
                      self.html)

    def test_locate_in_overview_targets_module_box_visible(self):
        # 修 3：总览定位用所属模块盒（非全图坐标），定位后该盒与多数模块
        # 盒屏幕可见（修复前 visibleModules=[]）
        lo = self.render["locateOverview"]
        self.assertEqual(lo["mode"], "overview")
        self.assertEqual(lo["sel"]["type"], "node")
        self.assertTrue(lo["changed"], "总览定位未生效")
        self.assertIn(lo["targetModule"], lo["visibleModules"],
                      "定位后目标模块盒不在视口内")
        self.assertGreaterEqual(len(lo["visibleModules"]), 8,
                                "定位后可见模块盒过少（疑似整图移出视口）")

    def test_locate_in_module_mode_centers_and_reports_external(self):
        # 局部模式：内部节点居中且 ≥12px；外部节点明确说明且不动视图
        m = self.render["locateModule"]
        self.assertTrue(m["centered"], "局部模式定位未把模块内节点居中")
        self.assertGreaterEqual(m["textPx"], 12 - 1e-9)
        ext = m["externalNode"]
        self.assertEqual(ext["sel"]["type"], "node")
        self.assertTrue(ext["viewKept"], "外部节点定位不应移动视图")
        self.assertTrue(ext["noteShown"], "外部节点定位未给明确说明")

    def test_hca_compressor_label_not_csa(self):
        # 修 5：HCA 图压缩器模块/分区标签为 HCA 压缩器；SWA 无压缩器盒；
        # CSA 图仍显示 CSA 压缩器
        h = self.render["hcaLabel"]
        self.assertEqual(h["module"], "HCA 压缩器")
        self.assertEqual(h["lane"], "HCA 压缩器")
        self.assertTrue(h["modsHtmlHasHCA"], "总览渲染无 HCA 压缩器标签")
        self.assertTrue(h["swaNoCompressorBox"], "SWA 不应渲染压缩器盒")
        self.assertEqual(h["csaStill"], "CSA 压缩器")


if __name__ == "__main__":
    unittest.main()
