"""自包含离线阅读页渲染（工作包 C）：图/规划/场景/证据 → viewer/structure.html。

单文件 HTML，无外网/CDN，file:// 可打开。数据以 JSON 嵌入（同源：图、规划、
场景、证据都来自 build 产物，页面 JS 只做呈现不派生事实）。
- L0：Embedding → HC Expand → 43 层（四组计数/层序）→ HC Head → Final
  RMSNorm → LM Head + DSpark 外部读出；未提取模板明确标"未覆盖"。
- L1（主视图，首屏）：默认 CSA 紧凑模块总览（R2a）——固定语义模块分组
  （真实节点计数，空模块不显示），模块间投影边保留全部原始 edge stable_id，
  模块内部边单列，投影 ∪ 内部 = G.edges 全量且互斥（每图渲染前守恒断言）。
  点击模块进入局部展开视图：该模块内部真实节点/原始边有向图（lane =
  stable_id 去末段的真实族、列 = 模块内拓扑深度，state 边画弧不参与分层），
  跨边界边连到左右端口轨道并标注真实外部端点（保留原边 ID，不新增假边）；
  节点表/边列表保留为补充。"完整图"模式保留 R1 全量拓扑（主图 107 节点/
  173 边，列 = 数据流拓扑深度、行 = 语义分区）。
- R2b 默认可读性：局部展开默认适配使节点标签实际 CSS 像素 ≥ 12px
  （11px 字号 × 缩放下限 12/11，以真实字号×变换核验）；横向溢出时从左侧
  输入端起始（不再居中裁掉输入侧）；工具条提供"定位选中"（节点/边居中）、
  "查看全局范围"（显式整体缩小，允许低于可读字号但不作默认）、"适配视图"
  （恢复可读默认）。长标签截短显示，全名在 title 与常驻详情面板保留。
- R2b 四图统一：CSA·TopK（主图）/ CSA·Hash / HCA·TopK / SWA·Hash 共用
  同一渲染管线，图工具条"图"按钮切换；每次切换按所选图重建全部节点/边/
  张量/端口/guards/场景/模块归属与布局（无跨图残留缓存）。场景与选择跨图
  同步：场景 ID 在当前图有效则保留，无效选择清空并在详情区说明；主图
  guard_activations（{gid:{state,reason}}）与 variantViews（{gid:字符串}）
  两种真实数据形状经 gNormGuards 归一化，字符串不误判为 unknown。
  边三态随场景切换（None guard=无条件，unknown 不伪作 false）；混合 guard
  聚合显示 mixed，不谎报同一状态。两套场景按钮（图工具条/下方场景面板）
  统一入口 gSetScenario。选择状态 module/pair/node/edge 互斥单一来源，场景
  切换按当前选择类型刷新详情。点击节点/边/模块看真实端口/张量/证据（复用
  轴与来源面板）；拖拽平移 + 滚轮缩放（带 5px 点击阈值与按需 pointer capture）。
- R2b 补修01（实际页面 JS 路径五缺陷）：图上方说明在派生数据就绪后生成并
  随切图更新（不再初始 0 节点/0 边、切图不刷新）；CSA·TopK 按钮键与主图
  统一为同一上下文（变体 JSON 只读不改，页面别名合并——往返/重复点击不丢
  主图压缩说明与 guard 理由，默认高亮即按钮键）；张量/证据详情卡近图放置
  （L1 图卡之后）并随切图维护：有效项按当前图重渲染、失效项清空并说明、
  切回包含它的图自动恢复；"定位选中"按当前渲染模式定位——总览定位所选
  节点/边所属模块盒（不用完整图坐标，不把整图移出视口），局部展开只定位
  本模块内对象（外部项明确说明且不动视图），完整图按全图坐标居中；压缩器
  模块/分区标题按当前图真实 attention 类型显示 CSA/HCA/SWA（不再 HCA 沿用
  CSA）。
- guard 三态随场景切换（数据在 scenarios.json，切换只是呈现）；状态前后
  （v_hist → v_after）可读；默认 decode_p128。
- DSpark 整体集成图（B4）：第五图上下文 + 场景状态 + 证据合并发布（主图
  61 条 + DSpark 122 条同 ID 逐字段一致合并进同一 evidence payload 与
  reports/source_evidence.json，全部 ev_ 引用闭合）。
- B4 补修01：DSpark 实例角色/参数绑定面板（消费
  dsparkView.integration.role_bindings——逐实例 mtp.0/1/2、实际层
  43/44/45、first/last 角色、公共模板参数 → mtp.i 路径/张量联查、唯一
  embed/head 共享对象、各实例 cache state_id、last norm/hc_head/Markov/
  Confidence 参数及 runtime/checkpoint dtype；全部从实际绑定数据派生，
  节点/张量链接可点开当前图详情）；场景面板每实例 cache before/after/
  published 版本绑定与前提分开的 valid_length 数值（前提未满足=unknown，
  覆盖判定按 P+1 而非 P——P=127 时 P+1=128 恰满 W=128）；L0/层规划读出
  表述 = 三有序 readout concat → stage0 唯一 main_proj/main_norm → 唯一
  main_x 三 stage 共享（与 plan.dspark_readouts 单条记录同口径）。
- 中文非技术阅读；信息不靠颜色/悬停单独承载（关键状态有文字）。
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

__all__ = ["render_structure_html"]


def render_structure_html(
    *,
    graph_json: str,
    plan_dict: dict,
    scenarios_json: str,
    evidence_json: str,
    model_key: str,
    analysis_id: str,
    display_name: str,
    variant_graphs: dict | None = None,
    variant_views: dict | None = None,
    dspark_graph_json: str | None = None,
    dspark_view_json: str | None = None,
) -> bytes:
    """渲染单文件阅读页；全部数据 JSON 内嵌，JS 只呈现。

    payload 嵌入约定：script[type=application/json] 是 raw text，浏览器
    JSON.parse(textContent) 不做 HTML 实体解码——嵌入的是原始 JSON 序列化
    （非 HTML 转义），仅把 ``</script``、``<!--`` 等终止序列用 JSON 字符串
    转义（<）防提前闭合，符合浏览器与 json.loads 双向可解析。
    dspark_graph_json/dspark_view_json 为 B4 新增：DSpark 整体集成图与
    场景状态视图（缺省 None 时页面显示 DSpark 未提供数据）。
    """
    payload = {
        "modelKey": model_key,
        "analysisId": analysis_id,
        "displayName": display_name,
        "graph": json.loads(graph_json),
        "plan": plan_dict,
        "scenarios": json.loads(scenarios_json),
        "evidence": json.loads(evidence_json),
        "variants": variant_graphs or {},
        "variantViews": variant_views or {},
        "dsparkGraph": (json.loads(dspark_graph_json)
                        if dspark_graph_json else None),
        "dsparkView": (json.loads(dspark_view_json)
                       if dspark_view_json else None),
    }
    data_json = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"),
    ).replace("</", "<\\/")  # 防 </script> 提前终止；JSON 内 \/ 合法等价
    template = _TEMPLATE.replace("__DATA__", data_json)
    return template.encode("utf-8")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>模型结构阅读 — DeepSeek-V4-Flash-0731</title>
<style>
:root { --bg:#faf9f7; --fg:#1a1a18; --mut:#6b675f; --line:#dcd8d0;
        --card:#ffffff; --acc:#1f5f8b; --warn:#8a5a00; --unk:#8a5a00;
        --ok:#2c6e49; --chip:#eef2f5; --code:#f4f2ee; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font: 15px/1.65 -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 24px 20px 80px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 28px 0 10px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 16px; margin: 18px 0 8px; }
.sub { color: var(--mut); font-size: 13px; }
.chip { display:inline-block; background:var(--chip); border:1px solid var(--line);
        border-radius: 4px; padding: 1px 8px; margin: 2px 4px 2px 0; font-size: 12.5px; }
.chip.warn { background:#fdf3e0; border-color:#e5c98f; color:var(--warn); }
.chip.ok { background:#e8f3ec; border-color:#bcd9c6; color:var(--ok); }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding: 14px 16px; margin: 10px 0; }
details { margin: 6px 0; }
summary { cursor: pointer; font-weight: 600; }
summary:hover { color: var(--acc); }
pre { background: var(--code); border:1px solid var(--line); border-radius:6px;
      padding: 10px 12px; overflow-x:auto; font-size: 12.5px; line-height:1.5; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 8px 0; }
th, td { border: 1px solid var(--line); padding: 5px 8px; text-align: left;
         vertical-align: top; }
th { background: var(--chip); }
button.sel { margin: 2px 6px 2px 0; padding: 4px 12px; border:1px solid var(--line);
        background:var(--card); border-radius: 5px; cursor: pointer; font-size: 13px; }
button.sel.on { background: var(--acc); color:#fff; border-color: var(--acc); }
.flow { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:10px 0; }
.flow .step { background:var(--card); border:1px solid var(--line); border-radius:6px;
        padding: 6px 12px; }
.flow .arr { color: var(--mut); }
.unknown { color: var(--unk); font-weight: 600; }
.kv { font-size: 13px; }
.kv b { display:inline-block; min-width: 120px; }
.badge-u { background:#fdf3e0; color:var(--warn); border:1px solid #e5c98f;
       border-radius:4px; padding:0 6px; font-size:12px; }
footer { margin-top:40px; color:var(--mut); font-size:12.5px;
         border-top:1px solid var(--line); padding-top:12px; }

/* ---- L1 拓扑图（R1）---- */
.gtoolbar { display:flex; flex-wrap:wrap; gap:6px; align-items:center;
            margin:8px 0; }
.gtoolbar .sel { padding:2px 10px; font-size:12.5px; }
.gsep { width:1px; height:18px; background:var(--line); margin:0 4px; }
.glegend { display:flex; flex-wrap:wrap; gap:4px 12px; font-size:12px;
           color:var(--mut); }
.gl { display:inline-flex; align-items:center; gap:4px; }
.gsw { display:inline-block; width:18px; height:10px; border-radius:2px; }
.gsw.edge-on { background:var(--acc); height:4px; border-radius:2px; }
.gsw.edge-off { background:var(--line); height:4px; border-radius:2px; }
.gsw.edge-unk { background:repeating-linear-gradient(90deg, var(--warn) 0 4px, transparent 4px 7px); height:4px; border-radius:2px; }
.gsw.eg-state { background:var(--ok); height:4px; border-radius:2px; }
.gsw.eg-select { background:#7a4ea3; height:4px; border-radius:2px; }
.gsw.eg-control { background:var(--mut); height:4px; border-radius:2px;
                  border-top:1px dashed var(--mut); border-bottom:1px dashed var(--mut); }
.gsw.nd-param { background:#f0e6d2; border:1px solid #cbb27f; }
.gsw.nd-tensor { background:#e3edf4; border:1px solid #9db9cc; }
.gsw.nd-state { background:#e8f3ec; border:1px solid #8fbf9f; }
.gsw.nd-quant { background:#fbeaea; border:1px solid #d29a9a; }
.gsw.nd-index { background:#efe4f3; border:1px solid #b48ec7; }
.gsw.nd-opaque { background:#e8e8e6; border:1px solid #9a9a94; }
#gviewport { position:relative; overflow:hidden; background:var(--code);
             border:1px solid var(--line); border-radius:6px;
             cursor:grab; touch-action:none; }
#gviewport.ggrabbing { cursor:grabbing; }
#gsvg { display:block; width:100%; height:640px; }
#gdragtip { position:absolute; right:10px; bottom:8px; color:var(--mut);
            font-size:12px; pointer-events:none; }
.gdetail { margin-top:10px; }
/* R2a 修复：CSS fill 曾覆盖节点 presentation fill 属性，类型色板全白。
   现在 .gnode rect 只设描边宽度，类型颜色由元素 fill/stroke 属性生效。 */
.gnode { cursor:pointer; }
.gnode rect { stroke-width:1.4; }
.gnode:hover rect { stroke-width:2.4; }
.gnode text { font-size:11px; fill:var(--fg);
              font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
.gnode .gtype { font-size:9.5px; fill:var(--mut); }
.gnode.gsel rect { stroke-width:2.6; }
/* R2a 模块总览：语义模块盒 + 模块间投影边 */
.gmod { cursor:pointer; }
.gmod rect.mbox { fill:var(--chip); stroke:#9db9cc; stroke-width:1.4; }
.gmod:hover rect.mbox { stroke-width:2.4; }
.gmod.gsel rect.mbox { stroke:#1f5f8b; stroke-width:2.6; }
.gmod text { font-size:12.5px; fill:var(--fg); font-weight:600;
             font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
.gmod .gsub { font-size:11px; fill:var(--mut); font-weight:400; }
.gmedge { fill:none; }
.gmedge-hit { fill:none; stroke:transparent; stroke-width:12;
              cursor:pointer; pointer-events:stroke; }
.gmlabel { font-size:10px; fill:var(--mut); paint-order:stroke;
           stroke:var(--code); stroke-width:3px;
           font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
.gmodwrap { overflow-x:auto; border:1px solid var(--line); border-radius:6px;
            background:var(--code); }
#gmodulepanel h4 { margin:12px 0 4px; font-size:13.5px; }
#gmodulepanel ul { margin:4px 0 10px; padding-left:18px; font-size:13px; }
/* R2a 局部展开图：跨边界边复用 .gedge（事件/选中共用）；此处为
   端口轨道锚点与族标签样式 */
.gmedge.gsel { stroke-width:2.6; }
.gport-anchor { fill:#e8e8e6; stroke:#9a9a94; }
.gllabel { font-size:10px; fill:var(--mut);
           font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
.gport { font-size:9.5px; fill:var(--mut);
         font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; }
.gmodhead { display:flex; flex-wrap:wrap; gap:6px; align-items:center;
            margin:2px 0 6px; }
@media (max-width: 720px) {
  #gviewport { overflow-x:auto; }
  #gsvg { min-width: 760px; }
}
.gedge { fill:none; cursor:pointer; }
.gedge:hover { stroke-width:2.6; }
.gedge.gsel { stroke-width:2.6; }
.gedge-hit { fill:none; stroke:transparent; stroke-width:12;
             cursor:pointer; pointer-events:stroke; }
.glane { stroke:var(--line); stroke-dasharray:3,5; }
.gcols { display:flex; gap:24px; flex-wrap:wrap; }
.gcols > div { flex:1 1 280px; }
#gdetail h4 { margin:10px 0 4px; font-size:13.5px; }
#gdetail ul { margin:4px 0 10px; padding-left:18px; font-size:13px; }
.garrow { fill:var(--mut); }
</style>
</head>
<body>
<main>
<h1 id="title"></h1>
<p class="sub" id="meta"></p>

<h2 id="l1graph-h2">L1 图 · 真实有向拓扑（四图统一视图，默认 CSA·TopK）</h2>
<div class="card" id="l1graphcard">
  <p class="kv" id="l1graphnote"></p>
  <div class="gtoolbar">
    <span class="sub">图：</span>
    <button class="sel" id="gbtn-var" data-var="csa_topk"
            onclick="gSetVariant('csa_topk')">CSA·TopK</button>
    <button class="sel" id="gbtn-var" data-var="csa_hash"
            onclick="gSetVariant('csa_hash')">CSA·Hash</button>
    <button class="sel" id="gbtn-var" data-var="hca_topk"
            onclick="gSetVariant('hca_topk')">HCA·TopK</button>
    <button class="sel" id="gbtn-var" data-var="swa_hash"
            onclick="gSetVariant('swa_hash')">SWA·Hash</button>
    <button class="sel" id="gbtn-var" data-var="dspark"
            onclick="gSetVariant('dspark')">DSpark</button>
    <span class="gsep"></span>
    <span class="sub">视图：</span>
    <button class="sel on" id="gbtn-mod" onclick="gSetMode('overview')">模块总览</button>
    <button class="sel" id="gbtn-full" onclick="gSetMode('full')">完整图</button>
    <span class="gsep"></span>
    <span class="sub">场景（边状态随场景变化）：</span><span id="gscenbtns"></span>
    <span class="gsep"></span>
    <button class="sel" id="gbtn-fit" onclick="gFit()">适配视图</button>
    <button class="sel" id="gbtn-zin" onclick="gZoomBy(1.25)">放大 +</button>
    <button class="sel" id="gbtn-zout" onclick="gZoomBy(0.8)">缩小 −</button>
    <button class="sel" id="gbtn-locate" onclick="gLocateSelection()">定位选中</button>
    <button class="sel" id="gbtn-global" onclick="gShowGlobal()">查看全局范围</button>
    <span class="sub" id="gzoomlabel"></span>
    <span class="gsep"></span>
    <span class="glegend">
      <span class="gl"><span class="gsw edge-on"></span>活动边</span>
      <span class="gl"><span class="gsw edge-off"></span>guard 不成立</span>
      <span class="gl"><span class="gsw edge-unk"></span>guard 未知</span>
      <span class="gl"><span class="gsw eg-state"></span>state（跨调用）</span>
      <span class="gl"><span class="gsw eg-select"></span>select</span>
      <span class="gl"><span class="gsw eg-control"></span>control</span>
      <span class="gl"><span class="gsw nd-param"></span>参数</span>
      <span class="gl"><span class="gsw nd-tensor"></span>张量算子</span>
      <span class="gl"><span class="gsw nd-state"></span>状态读写</span>
      <span class="gl"><span class="gsw nd-quant"></span>量化</span>
      <span class="gl"><span class="gsw nd-index"></span>索引/选择</span>
      <span class="gl"><span class="gsw nd-opaque"></span>不透明 kernel</span>
    </span>
  </div>
  <div id="gviewport">
    <svg id="gsvg" role="img" aria-label="当前所选变体的真实有向拓扑图">
      <defs id="gdefs"></defs>
      <g id="groot"><g id="glanes"></g><g id="gmods"></g><g id="gmodedges"></g>
        <g id="gedges"></g><g id="gnodes"></g><g id="gxrails"></g></g>
    </svg>
    <div id="gdragtip" hidden>拖拽平移 · 滚轮缩放</div>
  </div>
  <p class="kv" id="gscensum"></p>
  <div id="gmodulepanel" class="gdetail"></div>
  <div id="gdetail" class="gdetail"></div>
  <p class="kv" id="l1graphpending"></p>
</div>

<h2>L0 · 整体层序</h2>
<div class="card">
  <div class="flow" id="l0flow"></div>
  <p class="kv" id="l0groups"></p>
  <p class="kv" id="l0dspark"></p>
</div>

<h2>场景</h2>
<div class="card">
  <p class="kv">场景切换改变 guard 三态与压缩路径细节的显示（数据同源，页面只呈现）。
  默认 decode_p128。下方按钮与 L1 图工具条共用同一场景状态（统一入口
  gSetScenario，两处按钮与图边/详情始终一致）；图工具条切图后场景按当前图
  有效性保留或改选默认，两处按钮同步。</p>
  <div id="scenbtns"></div>
  <div id="scenview"></div>
</div>

<h2 id="l1table-h2">L1 · 代表层节点表（当前图：<span id="l1tablevar">csa_topk</span>）</h2>
<div class="card">
  <p class="kv" id="l1note"></p>
  <div id="l1modules"></div>
</div>

<h2>层规划与实例绑定</h2>
<div class="card" id="planview"></div>

<h2>来源证据与覆盖</h2>
<div class="card" id="evidenceview"></div>

<footer id="foot"></footer>
</main>
<script id="payload" type="application/json">__DATA__</script>
<script>
const P = JSON.parse(document.getElementById('payload').textContent);
const G = P.graph;
const esc = s => String(s==null?'':s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('title').textContent = P.displayName + ' — 模型结构阅读';
document.getElementById('meta').textContent =
  'analysis_id: ' + P.analysisId + ' · template: ' + G.template_id +
  ' · world_size=' + G.world_size + '（首版分析假设）';

// ---- L0（结构事实全部从 plan/graph 数据派生，不手写常量）----
const byGroup = {};
for (const g of P.plan.groups) byGroup[g.group_id] = g;
const nLayers = P.plan.bindings.length;
const hashLayers = P.plan.groups.filter(g => g.gate_kind === 'hash')
  .flatMap(g => g.member_layers).length;
const flow = [['Embedding','词表查表 + HC 展开'],
  ['HC Expand','1 → '+ (G.named_axes.find(a=>a.symbol==='N_hc')||{}).value +' 流'],
  [nLayers + ' 层 × Block','两次 hc_pre→norm→module→hc_post'],
  ['HC Head','4 流加权汇聚'],
  ['Final RMSNorm',''],
  ['LM Head','词表投影']];
document.getElementById('l0flow').innerHTML = flow.map((s,i) =>
  (i?'<span class="arr">→</span>':'') +
  '<span class="step"><b>'+esc(s[0])+'</b><br><span class="sub">'+esc(s[1])+'</span></span>').join('');
document.getElementById('l0groups').innerHTML = '<b>四组层序（0..'+(nLayers-1)+
  '；Hash 门控层 '+hashLayers+' 个）</b><br>' +
  P.plan.groups.map(g => {
    const extractedLayers = (P.plan.bindings||[])
      .filter(b => b.template_status==='extracted' &&
                   g.member_layers.includes(b.layer_id));
    const rep = extractedLayers.length > 0
      ? ' <span class="chip ok">代表模板已提取（层 '+
        extractedLayers.map(b=>b.layer_id).join(',')+'）</span>'
      : '';
    const rest = g.member_layers.length - extractedLayers.length;
    const pend = rest > 0 ? ' <span class="chip warn">其余 '+rest+' 实例未覆盖</span>' : '';
    return '<span class="chip">'+esc(g.group_id)+' ×'+g.member_layers.length+
      '（层 '+g.member_layers.join(',')+'）'+rep+pend+'</span>';
  }).join(' ');
// B4 补修01：读出关系 = 有序 concat → stage0 唯一 main_proj/main_norm →
// 唯一 main_x 三 stage 共享（plan.dspark_readouts 单条记录；不再沿用旧
// 一对一 readout→draft stage 映射表述）。
const readoutRec = (P.plan.dspark_readouts||[])[0] || null;
const readout = readoutRec ? readoutRec.readout_layers.join('→') : '';
// B4：DSpark 整体集成图可用性（L0 与图上下文共用；先于两处使用定义）
const DSPARK_AVAIL = !!(P.dsparkGraph && P.dsparkGraph.nodes);
const dsparkL0 = DSPARK_AVAIL
  ? '整体图 <button class="sel" onclick="gSetVariant(\'dspark\')">打开 DSpark 整体图</button>'
    +'（首入口 40/41/42 readout 有序 concat → stage0 唯一 main_proj/main_norm → '
    +'唯一 main_x 三 stage 分发 → 公共 stage 模板三实例 mtp.0/1/2 → 末出口 '
    +'head/Markov/Confidence → 三元组 output_ids/logits/confidence；'
    +P.dsparkGraph.nodes.length+' 节点/'+P.dsparkGraph.edges.length+' 边，'
    +'14 条跨图连接可联查 data/dspark/integration.json）'
  : '<span class="chip warn">DSpark 内部图未提供（本产物未含集成数据）</span>';
document.getElementById('l0dspark').innerHTML = '<b>DSpark 外部读出</b>：主干层 '+
  readout+' 的 hc 流均值<b>有序 concat</b> 进 stage0（mtp.0）唯一 main_proj/'
  +'main_norm，产出唯一 main_x 后<b>共享分发三 stage</b>（不是每 stage 一个'
  +' readout 的一对一映射；来源证据 model.py:917-921/832/931-932，'+
  '见层规划 data/layer_plan.json）；'+dsparkL0+
  ' 草稿 token 数 N_draft='+(G.named_axes.find(a=>a.symbol==='N_draft')||{}).value+
  '（'+(readoutRec ? '读出记录：'+esc(readoutRec.concat_note||'') : '')+'；'+
  '参考配置 nextn=1 分开解释；HF 实现无对应 DSpark 类，named 差异见 '
  +'data/dspark/integration.json hf_difference）。';
// L1 节点表说明由 renderL1Table() 按当前图生成（图工具条切换后同步）。

// ---- 场景（常量前置：图上下文构建引用 SC/SCEN）----
const SC = P.scenarios;            // 对象：{scenario_set, default_scenario, instances}
const SCEN = SC.instances;         // 数组
let curScen = SC.default_scenario || 'decode_p128';

// ---- 图选择（R2b 四图统一）：CSA·TopK（主图）+ 三个真实变体图 ----
// 每图一份 ctx：G/场景视图/模块归属/布局全部按所选图重建，无跨图缓存。
// 场景归一化：主图 guard_activations = {gid:{state,reason}}，variantViews
// = {gid:'true'|'false'}（build_pipeline 两种真实形状）；gNormGuards 统一
// 成 {state, reason}，字符串不误判 unknown。scenarioIds 为该图有效场景集。
const VAR = P.variants || {};
const VARV = P.variantViews || {};
function gNormGuards(raw) {
  const out = {};
  if (!raw) return out;
  for (const gid of Object.keys(raw)) {
    const v = raw[gid];
    if (v && typeof v === 'object') {
      out[gid] = { state: String(v.state || 'unknown'),
                   reason: v.reason || '' };
    } else {
      out[gid] = { state: String(v), reason: '' };
    }
  }
  return out;
}
function gBuildCtx(vkey) {
  const g = vkey === 'main' ? G : VAR[vkey];
  const scen = vkey === 'main'
    ? SCEN.map(s => ({ scenario_id: s.scenario_id,
        guard_activations: gNormGuards(s.guard_activations),
        env: s.env, derived: s.derived, indexer: s.indexer,
        compression: s.compression || null,
        premise: s.history_premise || '' }))
    : (VARV[vkey] || []).map(v => ({ scenario_id: v.scenario_id,
        guard_activations: gNormGuards(v.guard_activations),
        env: v.env, derived: v.derived, indexer: v.indexer,
        compression: null, premise: v.premise || '' }));
  return { key: vkey, G: g, scen,
    scenIds: new Set(scen.map(s => s.scenario_id)),
    defaultScen: scen.some(s => s.scenario_id === SC.default_scenario)
      ? SC.default_scenario : (scen[0] ? scen[0].scenario_id : null),
    tensorById: (() => { const m = {};
      for (const t of g.tensors) m[t.stable_id] = t; return m; })(),
    isMain: vkey === 'main' };
}
// R2b 补修01：主图与 csa_topk 代表模板是同一张图（template/nodes/edges/
// tensors 全同），页面把两者统一为同一用户可达上下文——CTXS.csa_topk 与
// CTXS.main 是同一对象（场景含 compression 逐步说明与 guard 理由；变体
// JSON 只读不改）。VAR_BTN：请求键 → 按钮键（gBtnKey，按钮高亮匹配用，
// 使 main 初始态也点亮 csa_topk 按钮）。
// B4：DSpark 整体集成图（P.dsparkGraph）为第五个真实图上下文：场景视图
// 来自 dsparkView.views（对集成图 guards 逐场景求值 + 窗口/槽位/草稿
// 状态），不与主干场景混用。
const CTXS = { main: gBuildCtx('main') };
for (const k of Object.keys(VAR).sort())
  if (k !== 'csa_topk') CTXS[k] = gBuildCtx(k);
CTXS.csa_topk = CTXS.main;
if (DSPARK_AVAIL) {
  const dviews = (P.dsparkView && P.dsparkView.views) || [];
  CTXS.dspark = { key: 'dspark', G: P.dsparkGraph,
    scen: dviews.map(v => ({ scenario_id: v.scenario_id,
      guard_activations: gNormGuards(v.guard_activations),
      env: v.env, derived: {}, indexer: {}, compression: null,
      dspark: v, premise: v.history_premise || '' })),
    scenIds: new Set(dviews.map(v => v.scenario_id)),
    defaultScen: dviews.some(v => v.scenario_id === SC.default_scenario)
      ? SC.default_scenario : (dviews[0] ? dviews[0].scenario_id : null),
    tensorById: (() => { const m = {};
      for (const t of P.dsparkGraph.tensors) m[t.stable_id] = t; return m; })(),
    isMain: false, isDspark: true };
}
const VAR_BTN = { main: 'csa_topk' };     // 请求键 main → 按钮 csa_topk
function gBtnKey(vkey) { return VAR_BTN[vkey] || vkey; }
let curVar = 'main';          // 请求键（'main' 与 'csa_topk' 同一上下文）
let ctx = CTXS.main;
const VAR_LABEL = { main: 'CSA·TopK（主图）', csa_topk: 'CSA·TopK（主图）',
  csa_hash: 'CSA·Hash', hca_topk: 'HCA·TopK', swa_hash: 'SWA·Hash',
  dspark: 'DSpark 整体' };
const GUARD_ZH = {
  g_prefill: 'prefill 阶段（整段一次前向）',
  g_decode: 'decode 阶段（单 token 增量）',
  g_compress_csa: '本层启用 KV 压缩（r>0）',
  g_has_indexer: '学习式 Indexer 存在（r==4）',
  g_compress_prefill: 'prefill 压缩触发：本段 ≥ 一个压缩块',
  g_compress_prefill_skip: 'prefill 压缩未触发（不足一块，不产出压缩条目）',
  g_compress_decode: 'decode 压缩触发：(位置+1) 能整除压缩比',
  g_window_wrap: 'prefill 超窗：环形写入',
  g_causal_mask: 'prefill 压缩条目因果掩码',
  g_dspark_prefill_inwin: 'DSpark prefill 未超窗：顺序写 cache',
};
// R2b 补修01：压缩器命名按当前图真实 attention 类型（template_id：csa→
// CSA 压缩器、hca→HCA 压缩器），不手写每图常量；其余（如 swa_hash 无压缩
// 器）用语义准确的通用"KV 压缩器"。模块盒/端口轨道/分区标签共用。
function gCompKind(G0) {
  const t = String((G0 && G0.template_id) || '');
  if (t.indexOf('hca') >= 0) return 'hca';
  if (t.indexOf('csa') >= 0) return 'csa';
  return 'generic';
}
const GCOMP_ZH = { csa: 'CSA 压缩器', hca: 'HCA 压缩器', generic: 'KV 压缩器' };
function gCompLabel(G0) { return GCOMP_ZH[gCompKind(G0)]; }
function zhCompression(c) {
  if (c.phase === 'prefill') {
    let lines = [];
    lines.push('prefill：本段 ' + c.n_blk + ' 个完整压缩块' +
      (c.remainder ? '，另有 ' + c.remainder + ' 个 token 不足一块（尾块）' : '，无尾块'));
    if (c.should_compress) {
      lines.push('压缩触发：完整块做 overlap 变换后加权汇聚，产出 ' +
        c.cache_write_slots.length + ' 条压缩条目，写入压缩区');
      lines.push('主序列末块 → 状态前 ' + 4 + ' 槽（overlap 窗）；' +
        (c.remainder ? '尾块 → 状态第 5..8 槽中的 ' + c.remainder + ' 个（供后续 decode 池化）' : '无尾块写入'));
    } else {
      lines.push('压缩未触发：不足一个完整块，不产出压缩条目；' +
        (c.remainder ? '尾块仍写入状态第 5..8 槽中的 ' + c.remainder + ' 个' : '') +
        '（下次调用可用）');
    }
    lines.push('压缩区容量 ' + c.cache_write_capacity_note.match(/N_cc=\d+/)[0] + ' 条（上限），本调用写 ' + c.cache_write_slots.length + ' 条');
    return lines.map(l => '· ' + l).join('<br>');
  }
  let lines = [];
  const slot = c.state_slots.current_token_write.slots[0];
  lines.push('decode：当前 token 写入状态槽位 第' + (slot + 1) + ' 槽（共 8 槽）');
  if (c.should_compress) {
    lines.push('本步触发压缩：从状态 8 槽（历史+当前）加权汇聚出 1 条压缩条目，' +
      '写入压缩区第 ' + c.cache_write_slots[0] + ' 槽（0 起）');
    lines.push('池化后状态前移：前 4 槽 = 原 5..8 槽（model.py:359-360）');
  } else {
    lines.push('本步不触发压缩：只更新状态，不产出压缩条目、不写压缩区');
  }
  return lines.map(l => '· ' + l).join('<br>');
}
function renderScen() {
  const s = ctx.scen.find(v => v.scenario_id === curScen);
  document.querySelectorAll('#scenbtns button').forEach(b =>
    b.className = 'sel' + (b.dataset.id===curScen?' on':''));
  if (!s) {
    document.getElementById('scenview').innerHTML =
      '<p class="kv">当前场景 '+esc(curScen)+' 不属于图 '+
      esc(VAR_LABEL[curVar]||curVar)+'，已改选默认场景。</p>';
    return;
  }
  const env = s.env;
  let html = '<table><tr><th>变量</th><th>值</th><th>说明</th></tr>';
  const names = {B:'批大小',T:'本调用序列长',P:'历史长度',start_pos:'起始位置',
                 r:'压缩比',W:'窗口',Ki:'Indexer 上限'};
  for (const k of ['B','T','P','start_pos','r','W','Ki'])
    html += '<tr><td>'+k+'</td><td>'+esc(env[k])+'</td><td>'+esc(names[k]||'')+'</td></tr>';
  html += '</table>';
  html += '<h3>Guard 三态</h3><table><tr><th>guard</th><th>状态</th><th>谓词（原文）</th></tr>';
  for (const [gid, act] of Object.entries(s.guard_activations)) {
    const g = ctx.G.guards.find(x => x.stable_id === gid) || {};
    html += '<tr><td>'+esc(GUARD_ZH[gid]||gid)+'</td><td>'+
      (act.state==='true'?'<span class="chip ok">true（成立）</span>':
       act.state==='false'?'<span class="chip">false（不成立）</span>':
       '<span class="chip warn">unknown：'+esc(act.reason||'')+'</span>')+
      '</td><td><code>'+esc(g.predicate_expr)+'</code></td></tr>';
  }
  html += '</table>';
  if (s.compression) {
    html += '<h3>压缩路径（本场景逐步说明）</h3><p class="kv">'+zhCompression(s.compression)+'</p>';
    html += '<details><summary>压缩路径结构化数据（同源 scenarios.json）</summary><pre>'+
      esc(JSON.stringify(s.compression, null, 1))+'</pre></details>';
  } else {
    html += '<p class="kv"><b>压缩路径</b>：该图 '+esc(VAR_LABEL[curVar]||curVar)+
      ' 的场景视图未内嵌逐步压缩实例化（压缩/窗口路径在主图场景面板或 '+
      'data/variants/ 场景 JSON）；适用前提见下。</p>';
  }
  html += '<h3>选择边界</h3><table><tr><th>项</th><th>值</th></tr>';
  const idxZh = {visible_upper:'本调用可见压缩条目上限',
    hist_dependent:'是否依赖历史', select_width:'每 query 实际有效选择数',
    capacity_note:'说明', select_valid_upper:'每 query 有效选择上限',
    selection_mode:'选择方式', window_width:'窗口宽（物理）',
    window_valid_length:'窗口内有效条目', window_slots_used:'窗口已用槽位',
    window_next_slot:'下一写入槽位'};
  for (const [k,v] of Object.entries(s.indexer||{}))
    html += '<tr><td>'+esc(idxZh[k]||k)+'</td><td>'+esc(v)+'</td></tr>';
  html += '</table>';
  // B4：DSpark 整体图场景状态（窗口 cache / 草稿 KV / 位置坐标，同源
  // data/dspark/scenarios.json；由集成图 guards 对场景 env 真实求值）
  if (s.dspark) {
    const d = s.dspark;
    html += '<h3>DSpark 相位与状态（整体图本场景）</h3>';
    html += '<p class="kv"><b>forward_spec 流程</b>：'+esc(d.forward_spec_flow||'')+'</p>';
    const wc = d.window_cache||{}, dk = d.draft_kv||{}, ps = d.positions||{};
    html += '<table><tr><th>项</th><th>值</th></tr>';
    html += '<tr><td>win_cache 容量</td><td>'+esc(wc.capacity_slots)+' 槽'
      +'（压缩区 '+esc(wc.compress_zone_slots)+' 槽；r=0 无压缩区）</td></tr>';
    html += '<tr><td>本调用写入槽位</td><td>'+(Array.isArray(wc.written_slots_this_call)
      ? (wc.written_slots_this_call.length>12
         ? esc(wc.written_slots_this_call.length)+' 个槽（'+esc(wc.written_slots_this_call[0])+'..'+esc(wc.written_slots_this_call[wc.written_slots_this_call.length-1])+'）'
         : esc(JSON.stringify(wc.written_slots_this_call)))
      : '—')+'</td></tr>';
    if (wc.branch) html += '<tr><td>分支</td><td>'+esc(wc.branch)+'（'+esc(wc.branch_evidence||'')+'）</td></tr>';
    if (wc.wrap_note) html += '<tr><td>回绕说明</td><td>'+esc(wc.wrap_note)+'</td></tr>';
    // B4 补修01：数值与前提分开——valid_slot_count 数值 + 前提单独声明；
    // 前提未满足明确 unknown，不冒充满窗也不当作空
    html += '<tr><td>有效槽位数（valid_length 数值）</td><td>'+
      (wc.valid_slot_count != null ? esc(wc.valid_slot_count) : '<span class="unknown">unknown</span>')+
      (wc.valid_length_state ? ' <span class="chip">'+esc(wc.valid_length_state)+'</span>' : '')+'</td></tr>';
    if (wc.valid_length_premise) html += '<tr><td>前提（单独声明，不并入数值）</td><td>'+esc(wc.valid_length_premise)+'</td></tr>';
    if (wc.valid_length) html += '<tr><td>valid_length 文字口径</td><td>'+esc(wc.valid_length)+'</td></tr>';
    // 物理宽（W+N_draft）与选择有效宽（min(W,P+1)+N_draft）分开
    if (wc.physical_concat_width != null) html += '<tr><td>decode KV concat 物理宽</td><td>'+esc(wc.physical_concat_width)+'（W+N_draft；'+esc(wc.physical_concat_note||'')+'）</td></tr>';
    if (wc.select_valid_width != null) html += '<tr><td>每 query 有效选择宽</td><td>'+esc(wc.select_valid_width)+'（'+esc(wc.select_valid_expr||'')+'；'+esc(wc.select_valid_note||'')+'）</td></tr>';
    html += '<tr><td>草稿链</td><td>'+(dk.active
      ? ('激活：K='+esc(dk.k_steps)+' 步递推；选择索引宽 '+esc(dk.select_width_value)
         +'（'+esc(dk.select_width_expr)+'，无 -1 填充）')
      : 'prefill 早退：无草稿 Q/KV、无选择索引、无 sparse_attn（769）')+'</td></tr>';
    html += '<tr><td>main 位置</td><td>'+esc(ps.main_rope||'—')+'</td></tr>';
    if (ps.draft_rope) html += '<tr><td>草稿位置（RoPE 起点 P+T）</td><td>'+esc(ps.draft_rope)+'</td></tr>';
    if (ps.slot_write) html += '<tr><td>decode 写槽坐标</td><td>'+esc(ps.slot_write)+'</td></tr>';
    html += '</table>';
    // B4 补修01：每实例 cache 状态版本绑定（before/after/published +
    // 前提状态；从集成图真实 state_versions/state_transitions 派生）
    if (Array.isArray(d.stage_states) && d.stage_states.length) {
      html += '<h3>每实例 cache 状态版本绑定（mtp.0/1/2；before→after→published）</h3>';
      html += '<table><tr><th>实例</th><th>state_id / 迁移</th><th>写分支</th><th>读关系</th><th>before 内容前提（三态）</th><th>after 有效槽数</th></tr>';
      for (const st of d.stage_states) {
        // before_state_known 三态：known / unknown / 前提声明——unknown
        // 不显示成布尔 false（数量阈值不转 bool 语义，codex_b4_review02）
        const known = st.before_state_known;
        const knownChip = known === 'known'
          ? '<span class="chip ok">known</span>'
          : '<span class="unknown">'+esc(String(known))+'</span>';
        html += '<tr><td><b>mtp.'+st.stage_id+'</b>（'+esc(st.param_path)+'）</td><td class="sub">'+
          esc(st.state_id)+'<br>'+esc(st.before_version_id)+' → '+esc(st.after_version_id)+
          '<br>published = '+esc(st.published_version_id)+'（迁移 '+esc(st.transition_id)+'；张量 '+
          (window.showTensor ? '<a href="#" onclick="showTensor(\''+String(st.cache_tensor).replace(/'/g,'')+'\');return false;">'+esc(String(st.cache_tensor).replace(/^tensor:[^:]*:/, ''))+'</a>' : esc(st.cache_tensor))+
          '）</td><td class="sub">'+esc(st.write_branch||'')+'</td><td class="sub">'+
          esc(st.read_relation||'')+'</td><td class="sub">'+knownChip+' '+
          esc(st.before_state_note||'')+'</td><td>'+
          (st.after_slot_count != null ? esc(st.after_slot_count) : '—')+'</td></tr>';
      }
      html += '</table>';
    }
    html += '<p class="kv"><b>unknown 边界</b>：'+esc(wc.unknown_note||'')+
      '；output_ids/logits/confidence 实际 token 与数值未知（静态结构分析，不执行推理）。</p>';
  }
  html += '<p class="kv"><b>派生数量</b>：'+Object.entries(s.derived||{})
    .map(([k,v])=>k+'='+v).join('，')+'</p>';
  html += '<p class="kv"><b>前提：</b>'+esc(s.premise||'—')+'</p>';
  document.getElementById('scenview').innerHTML = html;
}
function gRebuildScenButtons() {
  // 每图重建两套场景按钮（只保留当前图有效场景；ID 集来自 ctx.scenIds）
  for (const holderId of ['scenbtns', 'gscenbtns']) {
    const sb = document.getElementById(holderId);
    sb.innerHTML = '';
    for (const s of ctx.scen) {
      const b = document.createElement('button');
      b.className = 'sel'; b.dataset.id = s.scenario_id;
      b.textContent = s.scenario_id;
      b.onclick = () => { gSetScenario(s.scenario_id); };
      sb.appendChild(b);
    }
  }
}

// ---- L1 节点表（当前图 ctx；每次切图重建）----
const EVID = P.evidence.entries;
const evById = {}; for (const e of EVID) evById[e.evidence_id] = e;
function renderL1Table() {
  const G0 = ctx.G;
  const tensorById0 = ctx.tensorById;
  const mods = {};
  for (const n of G0.nodes) {
    const top = n.stable_id.split('.').slice(0,2).join('.');
    (mods[top] = mods[top] || []).push(n);
  }
  function axisRow(t, a, i) {
    const phys = t.physical_layout.indexOf(a.symbol);
    const shapeVal = a.value != null ? String(a.value) : (a.expression ? '='+a.expression : '未知');
    return '<tr><td>'+(i+1)+'</td><td><b>'+esc(a.symbol)+'</b></td><td>'+esc(a.name)+
      '</td><td>'+esc(a.meaning)+'</td><td>'+esc(a.origin)+'</td><td>'+esc(shapeVal)+
      (a.unknown_reason?'<br><span class="unknown">未知：'+esc(a.unknown_reason)+'</span>':'')+
      '</td><td>逻辑第 '+(i+1)+' / 物理第 '+(phys+1)+'</td><td>'+
      (evById[a.evidence_ref]?evBtn(a.evidence_ref):esc(a.evidence_ref))+'</td></tr>';
  }
  function tensorPanel(tid) {
    const t = tensorById0[tid];
    if (!t) return '<p class="sub">张量 '+esc(tid)+' 不在本图张量表</p>';
    let html = '<p class="kv"><b>'+esc(t.name)+'</b> · dtype '+esc(t.dtype)+
      ' · 逻辑布局 '+t.logical_layout.join('×')+
      ' · 物理布局 '+t.physical_layout.join('×')+'</p>';
    html += '<table><tr><th>#</th><th>轴</th><th>名称</th><th>含义</th><th>来源</th><th>值/表达式</th><th>布局位置</th><th>证据</th></tr>';
    t.axes.forEach((a, i) => { html += axisRow(t, a, i); });
    html += '</table>';
    return html;
  }
  let l1html = '';
  for (const m of Object.keys(mods).sort()) {
    const members = mods[m];
    l1html += '<details><summary>'+esc(m)+'（'+members.length+' 节点）</summary><table>'+
      '<tr><th>节点</th><th>op</th><th>输入 ←</th><th>→ 输出</th><th>证据</th></tr>';
    for (const n of members) {
      const ins = G0.edges.filter(e => e.dst_node_id===n.stable_id)
        .map(e => esc(e.src_node_id)+'.'+esc(e.src_port)+' → '+esc(e.dst_port)
          + (e.guard_ref?' <span class="chip">guard:'+esc(e.guard_ref)+'</span>':'')
          + (e.kind==='state'?' <span class="chip warn">state</span>':'')
          + (e.kind==='select'?' <span class="chip">select</span>':'')).join('<br>');
      const outs = n.output_ports.map(pt => '<a href="#" onclick="showTensor(\''+
        pt.tensor_id+'\');return false;">'+esc((tensorById0[pt.tensor_id]||{}).name||pt.tensor_id)+
        '</a>').join('<br>');
      const ev = (n.evidence_refs||[]).map(id => {
        const e = evById[id]; return e ? evBtn(id) : esc(id); }).join('<br>');
      l1html += '<tr><td><code>'+esc(n.stable_id)+'</code></td><td>'+esc(n.op_type)+
        '</td><td>'+ins+'</td><td>'+outs+'</td><td>'+ev+'</td></tr>';
    }
    l1html += '</table></details>';
  }
  document.getElementById('l1modules').innerHTML = l1html;
  const rAxis = G0.named_axes.find(a=>a.symbol==='r');
  const isDs = !!ctx.isDspark;
  const repLayers = isDs ? '—' : P.plan.bindings.filter(b=>b.template_id&&G0.template_id.endsWith(
    b.template_id.split(':')[1])).map(b=>b.layer_id).join(',')||'—';
  document.getElementById('l1note').innerHTML =
    '当前图 <b>'+esc(VAR_LABEL[curVar]||curVar)+'</b> 模板 <code>'+esc(G0.template_id)+'</code>'+
    (isDs ? '' : ' · 代表层 '+esc(repLayers)+' · r='+(rAxis ? rAxis.value : '0（无压缩）'))+
    '：'+G0.nodes.length+' 节点 / '+G0.edges.length+' 边（data '+
    G0.edges.filter(e=>e.kind==='data').length+
    ' · state '+G0.edges.filter(e=>e.kind==='state').length+
    ' · select '+G0.edges.filter(e=>e.kind==='select').length+
    '）。展开模块看真实连接；点击节点行的输出张量看完整轴（名称/含义/来源/逻辑物理位置/'+
    'dtype/shape），点击"证据"看行号与原文。';
  document.getElementById('l1tablevar').textContent = VAR_LABEL[curVar]||curVar;
  window._tensorPanel = tensorPanel;
}
function evBtn(id) {
  return '<a href="#" onclick="showEv(\''+id+'\');return false;">'+esc(id)+'</a>';
}
window.showEv = function(id) {
  const e = evById[id]; if (!e) return;
  // 证据条目全局静态（全图同一份来源证据）；记录展示类型供切图后恢复
  gEvSel = { type: 'ev', id };
  gRefreshEvDetail();
  location.hash = 'evdetail';
};
// R2b 补修01：张量详情选择状态 {type:'tensor', id}（图关联）——切图时
// 有效项按当前图重渲染，失效项清空并说明（见 gRefreshEvDetail）。
let gEvSel = null;
function gRefreshEvDetail() {
  const el = document.getElementById('evdetail');
  if (!gEvSel) return;                       // 未展示过内容：保持占位说明
  if (gEvSel.type === 'ev') {
    const e = evById[gEvSel.id];
    if (!e) {
      const gone = gEvSel.id;
      gEvSel = null;
      el.innerHTML = '<h3>证据详情已清空</h3><p class="kv">证据 <code>' +
        esc(gone) + '</code> 不在当前证据表。</p>';
      return;
    }
    renderEvEntry(e);
    return;
  }
  // tensor：必须按当前图重新解析——存在于当前图则重渲染（标题/轴与
  // 当前图一致），不存在则清空旧详情并明确说明；选择上下文保留，
  // 切回包含它的图时自动恢复（不依赖用户再点一次）
  const t = ctx.tensorById[gEvSel.id];
  if (!t) {
    el.innerHTML = '<h3>张量详情已清空</h3><p class="kv">张量 <code>' +
      esc(gEvSel.id) + '</code> 不在图 ' + esc(VAR_LABEL[curVar]||curVar) +
      ' 中（随切图清空，旧详情不保留）；切回包含它的图会自动恢复，' +
      '或在节点表/节点详情中点击当前图的张量名。</p>';
    return;
  }
  renderTensorEntry(t);
}
function renderEvEntry(e) {
  let html = '<h3>证据 ' + esc(e.evidence_id) + '</h3>';
  html += '<p class="kv"><b>来源</b> ' + esc(e.source_key) + '（固定 SHA256 基线） · '+
    '<b>符号</b> <code>'+esc(e.qualified_symbol)+'</code> · '+
    '<b>等级</b> '+evChip(e.coverage)+'</p>';
  if (e.line_start) html += '<p class="kv"><b>行区间</b> '+e.line_start+'-'+e.line_end+'</p>';
  if (e.snippet) html += '<pre>'+esc(e.snippet)+'</pre>';
  if (e.claim) html += '<p class="kv"><b>核对结论</b> '+esc(e.claim)+'</p>';
  if (e.derivation) html += '<p class="kv"><b>推导</b> '+esc(e.derivation)+'</p>';
  if (e.note) html += '<p class="kv"><b>未覆盖说明</b> '+esc(e.note)+'</p>';
  document.getElementById('evdetail').innerHTML = html;
}
function renderTensorEntry(t) {
  const panel = window._tensorPanel || (tid => '');
  const G0 = ctx.G;
  const allMatch = G0.tensors.every(t2 =>
    JSON.stringify(t2.logical_layout) === JSON.stringify(t2.physical_layout));
  const transposed = G0.tensors.filter(t2 =>
    JSON.stringify(t2.logical_layout) !== JSON.stringify(t2.physical_layout));
  const layoutFact = allMatch
    ? '本图实际：全部张量两布局一致（未声明转置）'
    : '本图实际：' + transposed.length + ' 个张量存在布局差异（' +
      esc(transposed.map(t2=>t2.name).slice(0,4).join('，')) + ' 等）；图中 transpose/reshape 节点的轴变换以节点连接为准';
  document.getElementById('evdetail').innerHTML =
    '<h3>张量与轴（图 '+esc(VAR_LABEL[curVar]||curVar)+'）</h3>' +
    panel(t.stable_id) +
    '<p class="sub">逻辑布局 = axes 顺序；物理布局 = 实际存储顺序。' + layoutFact + '。'+
    'shape 中带 =表达式 的轴随场景求值（见场景面板）。</p>';
}
window.showTensor = function(tid) {
  // 按当前图解析：不存在于当前图张量表时明确说明（不展示旧图内容）
  const t = ctx.tensorById[tid];
  if (!t) {
    gEvSel = null;
    document.getElementById('evdetail').innerHTML =
      '<h3>张量详情（图 ' + esc(VAR_LABEL[curVar]||curVar) + '）</h3>' +
      '<p class="kv">张量 <code>' + esc(tid) + '</code> 不在图 ' +
      esc(VAR_LABEL[curVar]||curVar) + ' 的张量表中。</p>';
    location.hash = 'evdetail';
    return;
  }
  gEvSel = { type: 'tensor', id: tid };
  renderTensorEntry(t);
  location.hash = 'evdetail';
};
function evChip(c) {
  return c==='source_checked' ? '<span class="chip ok">source_checked（逐句核对）</span>' :
         c==='located' ? '<span class="chip">located（定位核对）</span>' :
         c==='derived' ? '<span class="chip">derived（声明式推导）</span>' :
         '<span class="chip warn">uncovered（未覆盖）</span>';
}
// R2b 补修01：轴/证据详情卡放在 L1 图卡之后（近图可达，不要求滚过全部
// 长表才能接续读图）；仍为常驻面板，showEv/showTensor 写入，切图时由
// gRefreshEvDetail 维护。
const dv = document.createElement('div'); dv.className='card'; dv.id='evdetail';
document.querySelector('main').insertBefore(
  dv, document.getElementById('l1table-h2'));
dv.innerHTML = '<p class="sub">点击证据链接或张量名查看行号/原文/轴详情。</p>';

// ============================================================
// L1 真实有向拓扑图（R1，R2b 起按当前图重建）：ctx.G 全量 → 内嵌 SVG。
// 布局：列 = 数据流拓扑深度（Kahn 分层，flow 边严格左→右）；
// 行 = 语义分区 lane（hc/q/kv/cache/comp/idx/core/out/tail/moe）。
// state 边（跨调用）不参与分层，单独下弧绘制（本次→下次调用闭环）。
// 场景切换重算边三态：无 guard=无条件真；unknown 不伪作 false。
// 每次切图（gSetVariant）重建：gInitGraphCtx(ctx) 计算该图全部派生数据。
// ============================================================
const gnodeById = {};   // 当前图（重建时刷新）
const gedgeById = {};
let GNODES = [], GEDGES = [];
// 独立 alias_relations：张量别名关系，在节点详情中注明，不伪装成 data 边
let ALIASES = [];

// ---- 节点分类（op_type → 显示类型）----
function gNodeKind(n) {
  const op = n.op_type;
  if (op === 'quant_sim' || op.indexOf('quant') >= 0) return 'quant';
  if (op === 'index_op') return 'index';
  if (op === 'opaque_kernel') return 'opaque';
  if (op === 'state_op') return 'state';
  if (op === 'parameter_op') return 'param';
  return 'tensor';
}
const GKIND_FILL = {
  param: '#f0e6d2', tensor: '#e3edf4', state: '#e8f3ec',
  quant: '#fbeaea', index: '#efe4f3', opaque: '#e8e8e6',
};
const GKIND_STROKE = {
  param: '#cbb27f', tensor: '#9db9cc', state: '#8fbf9f',
  quant: '#d29a9a', index: '#b48ec7', opaque: '#9a9a94',
};
const GKIND_ZH = {
  param: '参数', tensor: '张量算子', state: '状态读写',
  quant: '量化', index: '索引/选择', opaque: '不透明 kernel',
};

// ---- 语义分区（stable_id → lane）----
const LANE_DEFS = [
  ['hc',       'HC（前处理/汇聚）',
    n => n.stable_id.startsWith('block.')],
  ['q',        'Q 路径',
    n => ['attn.q_norm','attn.q_rms_scale','attn.wq_a','attn.wq_b',
          'attn.q_unflatten','attn.q_rope'].includes(n.stable_id)],
  ['kv',       'KV 路径',
    n => ['attn.kv_norm','attn.wkv','attn.kv_rope','attn.kv_quant',
          'attn.kv_concat'].includes(n.stable_id)],
  ['cache',    'KV cache（跨调用）',
    n => ['attn.cache_read','attn.cache_update'].includes(n.stable_id)],
  ['comp',     gCompLabel(G),
    n => n.stable_id.startsWith('attn.compressor')],
  ['idx',      'Indexer',
    n => n.stable_id.startsWith('attn.indexer')],
  ['core',     '注意力核心',
    n => ['attn.win_topk','attn.select_cat','attn.sparse_attn',
          'attn.attn_sink','attn.compress_topk_fixed'].includes(n.stable_id)],
  ['out',      '输出投影',
    n => n.stable_id.startsWith('attn.o_') ||
         n.stable_id === 'attn.wo_b'],
  ['normtail', '归一化尾部',
    n => ['attn_norm','ffn_norm'].includes(n.stable_id)],
  ['moe',      'MoE', n => n.stable_id.startsWith('moe.')],
];
// B4：DSpark 整体图 lane 定义（仅 ctx.isDspark 时使用；node/stage/entry/
// exit/mk/cf 族按真实 stable_id 前缀分 lane，其余进 other 不丢弃）
const DSPARK_LANE_DEFS = [
  ['entry',  '首入口（readout/main_x/embed）', n => n.stable_id.startsWith('entry.')],
  ['stage',  '公共 stage（HC/MoE/骨架）', n => /(^|\.)stage\./.test(n.stable_id)],
  ['attn',   'DSparkAttention', n => /(^|\.)dspark\./.test(n.stable_id)],
  ['exit',   '末出口（hc_head/norm/head）', n => n.stable_id.startsWith('exit.')],
  ['markov', 'Markov 递推（K5）', n => n.stable_id.startsWith('mk.')],
  ['conf',   'Confidence', n => n.stable_id.startsWith('cf.')],
];
const DSPARK_LANE_LABEL = {};
for (const [key, label] of DSPARK_LANE_DEFS) DSPARK_LANE_LABEL[key] = label;
DSPARK_LANE_LABEL.other = '其它';
function gLaneDefsCur() { return ctx && ctx.isDspark ? DSPARK_LANE_DEFS : LANE_DEFS; }
function gLaneLabelMapCur() {
  return ctx && ctx.isDspark ? DSPARK_LANE_LABEL : LANE_LABEL;
}
function gLaneOf(n) {
  for (const [key, , test] of gLaneDefsCur()) if (test(n)) return key;
  return 'other';
}
const LANE_LABEL = {};
for (const [key, label] of LANE_DEFS) LANE_LABEL[key] = label;
LANE_LABEL.other = '其它';

// ---- 当前图派生数据（拓扑深度/几何/位置；gInitGraphCtx 重建）----
let gDepth = {}, gMaxDepth = 0, gPos = {}, gSvgW = 0, gSvgH = 0;
let gLaneOrder = [], gLaneY = {}, gLaneH = {};
const GW = { colW: 220, nodeW: 170, nodeH: 34, cellTop: 26, cellDy: 38,
             padL: 96, padT: 26, padB: 20 };

function gInitGraphCtx() {
  // 从 ctx.G 重建当前图全部派生数据（切图/初始化调用；无跨图残留）
  GNODES = ctx.G.nodes;
  GEDGES = ctx.G.edges;
  ALIASES = ctx.G.alias_relations || [];
  // 压缩器分区标签按当前图真实 attention 类型刷新（HCA/SWA 不沿用 CSA）
  const compLane = gCompLabel(ctx.G);
  const compDef = LANE_DEFS.find(d => d[0] === 'comp');
  if (compDef) compDef[1] = compLane;
  LANE_LABEL.comp = compLane;
  for (const k in gnodeById) delete gnodeById[k];
  for (const k in gedgeById) delete gedgeById[k];
  for (const n of GNODES) gnodeById[n.stable_id] = n;
  for (const e of GEDGES) gedgeById[e.stable_id] = e;
  gDepth = {}; gPos = {};
  const gFlowOut = {}, gInDeg = {};
  for (const n of GNODES) { gFlowOut[n.stable_id] = []; gInDeg[n.stable_id] = 0; }
  for (const e of GEDGES) {
    if (e.kind === 'state') continue;           // 跨调用状态不参与分层
    if (!gnodeById[e.src_node_id] || !gnodeById[e.dst_node_id]) continue;
    gFlowOut[e.src_node_id].push(e.dst_node_id);
    gInDeg[e.dst_node_id]++;
  }
  {
    const ready = GNODES.map(n => n.stable_id)
      .filter(id => gInDeg[id] === 0).sort();
    for (const id of ready) gDepth[id] = 0;
    const remaining = {}; for (const id in gInDeg) remaining[id] = gInDeg[id];
    const queue = ready.slice();
    while (queue.length) {
      const u = queue.shift();
      for (const v of gFlowOut[u].slice().sort()) {
        gDepth[v] = Math.max(gDepth[v] || 0, gDepth[u] + 1);
        if (--remaining[v] === 0) queue.push(v);
      }
    }
    // 分层完成性守卫：流图无环时应覆盖全部节点
    if (Object.keys(gDepth).length !== GNODES.length)
      throw new Error('topology layering incomplete: ' +
        Object.keys(gDepth).length + '/' + GNODES.length + '（流图存在环）');
  }
  gMaxDepth = Math.max.apply(null, GNODES.map(n => gDepth[n.stable_id]));
  // 几何：每 lane 高度按其 (lane,depth) 单元最大占位伸展
  gLaneOrder = gLaneDefsCur().map(d => d[0]);
  {
    const seenLanes = new Set();
    for (const n of GNODES) seenLanes.add(gLaneOf(n));
    for (const n of GNODES)
      if (!seenLanes.has(gLaneOf(n))) throw new Error('unknown lane');
    if (seenLanes.has('other')) gLaneOrder.push('other');
  }
  const gCellCount = {};
  for (const n of GNODES) {
    const k = gLaneOf(n) + '|' + gDepth[n.stable_id];
    gCellCount[k] = (gCellCount[k] || 0) + 1;
  }
  gLaneY = {}; gLaneH = {};
  {
    let y = GW.padT;
    for (const l of gLaneOrder) {
      let maxN = 1;
      for (let d = 0; d <= gMaxDepth; d++)
        maxN = Math.max(maxN, gCellCount[l + '|' + d] || 0);
      gLaneY[l] = y;
      gLaneH[l] = Math.max(96, GW.cellTop + maxN * GW.cellDy + 16);
      y += gLaneH[l];
    }
  }
  const gColX = d => GW.padL + d * GW.colW;
  {
    const idx = {};
    for (let d = 0; d <= gMaxDepth; d++)
      for (const n of GNODES) {
        if (gDepth[n.stable_id] !== d) continue;
        const lane = gLaneOf(n);
        const k = lane + '|' + d;
        const i = idx[k] || 0; idx[k] = i + 1;
        gPos[n.stable_id] = {
          x: gColX(d) + GW.nodeW / 2,
          y: gLaneY[lane] + GW.cellTop + i * GW.cellDy + GW.nodeH / 2,
          lane, depth: d,
        };
      }
  }
  gSvgW = gColX(gMaxDepth) + GW.nodeW / 2 + 60;
  let h = GW.padT + GW.padB;
  for (const l of gLaneOrder) h += gLaneH[l];
  gSvgH = h;
}

// ---- 边三态（按当前场景）----
// SVG presentation attribute 不解析 CSS var()，色值在这里展开
// （与 :root 色板一致：--acc #1f5f8b、--line #dcd8d0、--warn #8a5a00、
//  --ok #2c6e49、--mut #6b675f、select 紫 #7a4ea3）。
const GC = { acc: '#1f5f8b', line: '#dcd8d0', warn: '#8a5a00',
             ok: '#2c6e49', mut: '#6b675f', sel: '#7a4ea3' };
// R2b 可读性：节点标签 11px（.gnode text），默认适配保证实际 CSS 像素
// ≥ 12px → 完整图默认适配下限 = 12/11（想整体缩小用"查看全局范围"）。
const GNODE_TEXT_KMIN = 12 / 11;
function gGuardStatus(e, s) {
  if (!e.guard_ref) return 'on';                       // 无 guard=无条件真
  const act = s && s.guard_activations
    && s.guard_activations[e.guard_ref];
  if (!act) return 'unknown';
  if (act.state === 'true') return 'on';
  if (act.state === 'false') return 'off';
  return 'unknown';                                    // 未知不伪作 false
}
const GEDGE_COLOR = { on: GC.acc, off: GC.line, unknown: GC.warn };
const GKIND_EDGE_COLOR = { state: GC.ok, select: GC.sel, control: GC.mut };
function gEdgeColor(e, st) {
  return GKIND_EDGE_COLOR[e.kind] || GEDGE_COLOR[st];
}
function gEdgeDash(e, st) {
  if (e.kind === 'control') return '2,3';
  if (st === 'unknown' && e.kind === 'data') return '5,4';
  return '';
}
function gEdgeMarker(e, st) {
  if (e.kind === 'state') return 'state';
  if (e.kind === 'select') return 'sel';
  if (e.kind === 'control') return 'off';      // mut 灰，与 control 描边一致
  return st === 'on' ? 'on' : st === 'off' ? 'off' : 'unk';
}

// ---- SVG 生成（innerHTML 字符串；全部动态文本经 esc 转义）----
function gNodeLabel(id) {
  if (id.startsWith('attn.indexer.compressor.')) return 'idx·comp.' +
    id.slice('attn.indexer.compressor.'.length);
  if (id.startsWith('attn.indexer.')) return 'idx.' +
    id.slice('attn.indexer.'.length);
  if (id.startsWith('attn.compressor.')) return 'comp.' +
    id.slice('attn.compressor.'.length);
  return id;
}
function gNodeBox(n) {
  const p = gPos[n.stable_id], kind = gNodeKind(n);
  const x = p.x - GW.nodeW / 2, y = p.y - GW.nodeH / 2;
  let label = gNodeLabel(n.stable_id);
  if (label.length > 24) label = label.slice(0, 23) + '…';
  return '<g class="gnode" data-node="' + esc(n.stable_id) +
    '" transform="translate(' + x + ',' + y + ')">' +
    '<title>' + esc(n.stable_id) + ' · ' + esc(n.op_type) + '</title>' +
    '<rect width="' + GW.nodeW + '" height="' + GW.nodeH + '" rx="7"' +
    ' fill="' + GKIND_FILL[kind] + '" stroke="' + GKIND_STROKE[kind] + '"/>' +
    '<text x="8" y="15">' + esc(label) + '</text>' +
    '<text class="gtype" x="8" y="27">' + esc(n.op_type) + ' · ' +
    GKIND_ZH[kind] + '</text></g>';
}
function gEdgePath(e) {
  const s = gPos[e.src_node_id], t = gPos[e.dst_node_id];
  const x1 = s.x + GW.nodeW / 2, y1 = s.y;
  const x2 = t.x - GW.nodeW / 2 - 5, y2 = t.y;
  if (e.kind === 'state') {
    // 跨调用状态回边：下方大弧（本次写入 → 下次读取）
    let h = 0; for (const c of e.stable_id) h = (h * 31 + c.charCodeAt(0)) % 97;
    const dip = Math.min(120, Math.max(46,
      Math.abs(x1 - x2) * 0.06)) + h % 24;
    return 'M ' + x1 + ' ' + (y1 + GW.nodeH / 2) +
      ' C ' + x1 + ' ' + (y1 + GW.nodeH / 2 + dip) +
      ', ' + x2 + ' ' + (y2 + GW.nodeH / 2 + dip) +
      ', ' + x2 + ' ' + (y2 + GW.nodeH / 2);
  }
  const mx = (x1 + x2) / 2;
  if (Math.abs(y2 - y1) < 2) return 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
  return 'M ' + x1 + ' ' + y1 + ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' + y2 +
    ', ' + x2 + ' ' + y2;
}
function gRenderLanes() {
  let out = '';
  for (const l of gLaneOrder) {
    out += '<rect class="glane" x="8" y="' + (gLaneY[l] + 4) +
      '" width="' + (gSvgW - 16) + '" height="' + (gLaneH[l] - 12) + '"/>';
    out += '<text class="glane-label" x="14" y="' + (gLaneY[l] + 22) +
      '">' + esc(gLaneLabelMapCur()[l]) + '</text>';
  }
  document.getElementById('glanes').innerHTML = out;
}
function gRenderEdges(curSt) {
  let out = '';
  for (const e of GEDGES) {
    const st = gGuardStatus(e, curSt);
    const d = gEdgePath(e);
    out += '<path class="gedge" data-edge="' + esc(e.stable_id) +
      '" d="' + d + '" stroke="' + gEdgeColor(e, st) +
      '" stroke-width="1.4"' +
      (gEdgeDash(e, st) ? ' stroke-dasharray="' + gEdgeDash(e, st) + '"' : '') +
      (st === 'off' ? ' opacity="0.55"' : '') +
      ' marker-end="url(#garrow-' + gEdgeMarker(e, st) + ')"/>';
  }
  // 加宽透明命中路径：1.4px 细线难以点击
  for (const e of GEDGES)
    out += '<path class="gedge gedge-hit" data-edge="' + esc(e.stable_id) +
      '" d="' + gEdgePath(e) + '"/>';
  document.getElementById('gedges').innerHTML = out;
}
function gRenderNodes() {
  let out = '';
  for (const n of GNODES) out += gNodeBox(n);
  document.getElementById('gnodes').innerHTML = out;
}

// ============================================================
// R2a：CSA 紧凑模块总览（默认视图；R2b 起按当前图重建）
// 语义模块 = 固定规则（stable_id 前缀/成员集合，模块表按当前图 G 计数），
// 不手画数据；每个节点唯一归属，未知分组进具名 "其它"，不丢弃。
// 空模块（该图无成员）不显示。覆盖守恒：跨模块投影边（保存全部原始 edge
// stable_id）∪ 模块内部隐藏边
// = G.edges 全量且互斥；运行时守卫在渲染前断言，失败即显式报错。
// 总览布局 = 固定语义网格（模块间存在真实循环数据流：残差/HC 组合回写，
// 模块级不可拓扑分层，网格行列是可读性选择，不声称拓扑深度）。
// ============================================================
const GMOD_DEFS = [
  ['hc_attn', 'HC 展开（attention 前处理）',
    id => id.startsWith('block.hc_pre_attn.')],
  ['hc_ffn', 'HC 展开（FFN 前处理）',
    id => id.startsWith('block.hc_pre_ffn.')],
  ['norm_attn', 'attn_norm',
    id => id === 'attn_norm'],
  ['norm_ffn', 'ffn_norm', id => id === 'ffn_norm'],
  ['qkv', 'Q/KV 投影与旋转',
    id => ['attn.q_norm', 'attn.q_rms_scale', 'attn.wq_a', 'attn.wq_b',
      'attn.q_unflatten', 'attn.q_rope', 'attn.kv_norm', 'attn.wkv',
      'attn.kv_rope', 'attn.kv_quant', 'attn.kv_concat'].includes(id)],
  ['cache', 'KV cache（跨调用）',
    id => ['attn.cache_read', 'attn.cache_update'].includes(id)],
  ['compressor', 'CSA 压缩器',
    id => id.startsWith('attn.compressor.')],
  ['indexer', 'Indexer（学习式选择）',
    id => id.startsWith('attn.indexer.')],
  ['core', '注意力核心',
    id => ['attn.win_topk', 'attn.select_cat', 'attn.sparse_attn',
      'attn.attn_sink', 'attn.compress_topk_fixed'].includes(id)],
  ['out', '输出投影',
    id => id.startsWith('attn.o_') || id === 'attn.wo_b'],
  ['hc_post_attn', 'HC 汇聚（attention 后）',
    id => id === 'block.hc_post_attn.combine'],
  ['hc_post_ffn', 'HC 汇聚（FFN 后）',
    id => id === 'block.hc_post_ffn.combine'],
  ['moe', 'MoE',
    id => id.startsWith('moe.')],
];
// 总览网格：row/col（列 = 主干数据流方向，行 = 语义分组）。
// 布局事实：模块级投影有真实环（如 out→hc_post_attn→hc_ffn→norm_attn→
// compressor→cache→core→out），不能按模块拓扑深度排布；网格位置表达
// 语义分组而非深度，模块间连接方向仍由投影边箭头真实呈现。
// padT 预留上弧高度（回边从盒顶向上绕行，曲线峰值 ≈ 盒顶 - 0.75×lift，
// lift ≤ 56 → 峰值 ≥ 28 > 0，不裁剪）
const GW2 = { mW: 170, mH: 46, mCol: 205, mRow: 86, mPadL: 24, mPadT: 70 };
let gOtherKey = null;
// B4：DSpark 整体图语义模块（读出/入口、三实例 stage、草稿注意力、
// 末出口 head、Markov、Confidence——共享模板/实例差异在模块计数可见：
// stage 模块含 mtp0./mtp1./mtp2. 三实例全部节点，展开后按前缀区分）
const DSPARK_GMOD_DEFS = [
  ['entry', '首入口（readout→main_x→embed）',
    id => id.startsWith('entry.')],
  ['stage0', 'stage mtp.0（first：main_x 生产+embed 展开）',
    id => id.startsWith('mtp0.')],
  ['stage1', 'stage mtp.1（中段实例）', id => id.startsWith('mtp1.')],
  ['stage2', 'stage mtp.2（last：末出口角色）', id => id.startsWith('mtp2.')],
  ['exit', '末出口（hc_head→norm→共享 head）', id => id.startsWith('exit.')],
  ['markov', 'Markov 递推（K5，版本链）', id => id.startsWith('mk.')],
  ['conf', 'Confidence（prenorm_x+markov_embed）', id => id.startsWith('cf.')],
];
function gModDefsCur() { return ctx && ctx.isDspark ? DSPARK_GMOD_DEFS : GMOD_DEFS; }
function gModuleOf(nid) {
  for (const [key, , test] of gModDefsCur())
    if (test(nid)) return key;
  return 'other';
}
// 计算模块投影（按当前图 ctx.G；gInitModuleCtx 在切图/初始化时重建）
let GMODS = {};            // key → {key,label,nodes:[nid],nodeIds:Set}
let GMOD_PAIRS = {};       // 'src>dst' → [{edge,srcMod,dstMod}]
let GMOD_INTERNAL = {};    // modKey → [原始边引用]
let GMOD_GRID = null, GMOD_POS = {}, GMOD_SVG_W = 0, GMOD_SVG_H = 0;
// 总览网格：row/col（列 = 主干数据流方向，行 = 语义分组）。
// 布局事实：模块级投影有真实环（如 out→hc_post_attn→hc_ffn→norm_attn→
// compressor→cache→core→out），不能按模块拓扑深度排布；网格位置表达
// 语义分组而非深度，模块间连接方向仍由投影边箭头真实呈现。
function gInitModuleCtx() {
  GMODS = {}; GMOD_PAIRS = {}; GMOD_INTERNAL = {};
  gOtherKey = null;
  if (ctx.isDspark) {
    GMOD_GRID = {
      entry: [0, 0], stage0: [1, 0], stage1: [1, 1],
      stage2: [1, 2], exit: [1, 3], markov: [2, 3], conf: [2, 4],
    };
  } else {
    GMOD_GRID = {
      hc_attn: [0, 0], norm_attn: [0, 1], qkv: [0, 2], cache: [0, 3],
      compressor: [0, 4],
      hc_ffn: [1, 0], hc_post_ffn: [1, 1], core: [1, 3], indexer: [1, 4],
      norm_ffn: [2, 0], hc_post_attn: [2, 1], moe: [2, 2], out: [2, 3],
    };
  }
  for (const [key, label] of gModDefsCur())
    GMODS[key] = { key, label, nodes: [] };
  for (const n of GNODES) {
    const k = gModuleOf(n.stable_id);
    if (!GMODS[k]) {
      gOtherKey = 'other';
      if (!GMODS.other) GMODS.other = { key: 'other', label: '其它', nodes: [] };
    }
    GMODS[k].nodes.push(n.stable_id);
  }
  // 压缩器模块标签按当前图真实 attention 类型（HCA 图显示 HCA 压缩器），
  // 端口轨道/详情复用 gModLabel → 同步一致；空模块（SWA）仍不渲染
  if (GMODS.compressor) GMODS.compressor.label = gCompLabel(ctx.G);
  if (GMODS.other) GMODS.other.nodes.sort();
  for (const k in GMODS) GMODS[k].nodeIds = new Set(GMODS[k].nodes);
  if (!gOtherKey) gOtherKey = '';
  for (const e of GEDGES) {
    const s = gModuleOf(e.src_node_id), d = gModuleOf(e.dst_node_id);
    if (s === d) (GMOD_INTERNAL[s] = GMOD_INTERNAL[s] || []).push(e);
    else {
      const k = s + '>' + d;
      (GMOD_PAIRS[k] = GMOD_PAIRS[k] || []).push(e);
    }
  }
  // 覆盖守恒守卫：投影 ∪ 内部 = 全量且互斥（渲染前断言，失败显式报错）
  {
    const seen = new Set();
    for (const k in GMOD_PAIRS)
      for (const e of GMOD_PAIRS[k]) {
        if (seen.has(e.stable_id)) throw new Error('模块投影边重复: ' + e.stable_id);
        seen.add(e.stable_id);
      }
    for (const m in GMOD_INTERNAL)
      for (const e of GMOD_INTERNAL[m]) {
        if (seen.has(e.stable_id)) throw new Error('模块内部边重复: ' + e.stable_id);
        seen.add(e.stable_id);
      }
    if (seen.size !== GEDGES.length)
      throw new Error('模块边覆盖不守恒: ' + seen.size + '/' + GEDGES.length);
    const nodeTotal = Object.values(GMODS).reduce((a, m) => a + m.nodes.length, 0);
    if (nodeTotal !== GNODES.length)
      throw new Error('模块节点覆盖不守恒: ' + nodeTotal + '/' + GNODES.length);
  }
  // 总览几何（模块盒位置固定网格；空模块无位置不显示）
  GMOD_POS = {};
  {
    const usedRows = new Set(), usedCols = new Set();
    for (const k in GMOD_GRID) {
      if (!GMODS[k] || !GMODS[k].nodes.length) continue;
      usedRows.add(GMOD_GRID[k][0]); usedCols.add(GMOD_GRID[k][1]);
    }
    // "其它"模块（若有成员）：放在网格右侧外
    if (GMODS.other && GMODS.other.nodes.length) {
      const maxCol = Math.max.apply(null, [...usedCols]);
      GMOD_GRID.other = [0, maxCol + 1];
      usedCols.add(maxCol + 1);
    }
    let maxRow = 0, maxCol = 0;
    for (const k in GMOD_GRID) {
      if (!GMODS[k] || !GMODS[k].nodes.length) continue;
      maxRow = Math.max(maxRow, GMOD_GRID[k][0]);
      maxCol = Math.max(maxCol, GMOD_GRID[k][1]);
    }
    const colX = c => GW2.mPadL + c * GW2.mCol;
    const rowY = r => GW2.mPadT + r * GW2.mRow;
    for (const k in GMOD_GRID)
      if (GMODS[k] && GMODS[k].nodes.length)
        GMOD_POS[k] = { x: colX(GMOD_GRID[k][1]), y: rowY(GMOD_GRID[k][0]) };
    GMOD_SVG_W = colX(maxCol) + GW2.mW + 24;
    GMOD_SVG_H = rowY(maxRow) + GW2.mH + 20;
  }
}
// 混合 guard 聚合：pair 的边 guard 状态不一致时显示 mixed，不谎报同一状态
function gPairGuardStatus(pairEdges, scen) {
  let hasOn = false, hasOff = false, hasUnk = false, hasNone = false;
  for (const e of pairEdges) {
    const st = gGuardStatus(e, scen);
    if (st === 'on') hasOn = true;
    else if (st === 'off') hasOff = true;
    else hasUnk = true;
    if (!e.guard_ref) hasNone = true;
  }
  const states = [hasOn, hasOff, hasUnk].filter(x => x).length;
  if (states === 1 && !hasNone) {
    return hasOn ? 'on' : hasOff ? 'off' : 'unknown';
  }
  if (states === 1 && hasNone) return hasOn ? 'on' : 'mixed';
  if (states > 1) return 'mixed';
  return 'on';  // 全部无 guard
}
const GMOD_PAIR_COLOR = {
  on: GC.acc, off: GC.line, unknown: GC.warn, mixed: GC.sel,
};
// pair 方向性来源标签：真实 src/dst 节点（跨模块端点可能多对）
function gPairEndpoints(pairEdges) {
  const srcs = [...new Set(pairEdges.map(e => e.src_node_id))];
  const dsts = [...new Set(pairEdges.map(e => e.dst_node_id))];
  return { srcs, dsts };
}
// ---- 总览几何见 gInitModuleCtx（按当前图计算 GMOD_POS/GMOD_SVG_W/H）----
function gModBox(k) {
  const m = GMODS[k], p = GMOD_POS[k];
  return '<g class="gmod" data-mod="' + esc(k) + '">' +
    '<title>' + esc(m.label) + '（' + m.nodes.length + ' 真实节点）</title>' +
    '<rect class="mbox" x="' + p.x + '" y="' + p.y + '" width="' + GW2.mW +
    '" height="' + GW2.mH + '" rx="9" fill="#e3edf4" stroke="#9db9cc"/>' +
    '<text x="' + (p.x + 10) + '" y="' + (p.y + 20) + '">' + esc(m.label) +
    '</text>' +
    '<text class="gsub" x="' + (p.x + 10) + '" y="' + (p.y + 37) + '">' +
    m.nodes.length + ' 节点 · ' + (GMOD_INTERNAL[k] ? GMOD_INTERNAL[k].length : 0) +
    ' 内部边</text></g>';
}
function gModPairPath(pairEdges) {
  const { srcs, dsts } = gPairEndpoints(pairEdges);
  const sMod = gModuleOf(srcs[0]), dMod = gModuleOf(dsts[0]);
  const s = GMOD_POS[sMod], t = GMOD_POS[dMod];
  const sC = GMOD_GRID[sMod][1], dC = GMOD_GRID[dMod][1];
  const sR = GMOD_GRID[sMod][0], dR = GMOD_GRID[dMod][0];
  if (sC === dC && sR === dR) return '';
  if (sC === dC && sR !== dR) {         // 垂直：底部出 → 顶部入，侧向鼓出
    const x1 = s.x + GW2.mW / 2, y1 = s.y + GW2.mH;
    const x2 = t.x + GW2.mW / 2, y2 = t.y;
    const mx = x1 + (x2 - x1) / 2 + (x1 <= x2 ? 30 : -30);
    return 'M ' + x1 + ' ' + y1 + ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' +
      y2 + ', ' + x2 + ' ' + y2;
  }
  if (dC <= sC) {                        // 回边（向左）：上方绕行
    const x1 = s.x + GW2.mW / 2, y1 = s.y;
    const x2 = t.x + GW2.mW / 2, y2 = t.y;
    const lift = 34 + (sR === dR ? 18 : 0);
    return 'M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + (y1 - lift) +
      ', ' + x2 + ' ' + (y2 - lift) + ', ' + x2 + ' ' + y2;
  }
  // 前向（dC > sC）：同行水平直连；跨行从右出→左入的 C 曲线
  const x1 = s.x + GW2.mW, y1 = s.y + GW2.mH / 2;
  const x2 = t.x - 5, y2 = t.y + GW2.mH / 2;
  const mx = (x1 + x2) / 2;
  if (Math.abs(y2 - y1) < 2) return 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2;
  return 'M ' + x1 + ' ' + y1 + ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' + y2 +
    ', ' + x2 + ' ' + y2;
}
function gRenderOverview(curSt) {
  let mods = '';
  for (const k in GMODS) if (GMOD_POS[k]) mods += gModBox(k);
  document.getElementById('gmods').innerHTML = mods;
  let edges = '';
  for (const k in GMOD_PAIRS) {
    const pairEdges = GMOD_PAIRS[k];
    const st = gPairGuardStatus(pairEdges, curSt);
    const d = gModPairPath(pairEdges);
    if (!d) continue;
    const dash = st === 'unknown' ? '5,4' : st === 'mixed' ? '2,3' : '';
    edges += '<path class="gmedge" data-pair="' + esc(k) + '" d="' + d +
      '" stroke="' + GMOD_PAIR_COLOR[st] + '" stroke-width="1.6"' +
      (dash ? ' stroke-dasharray="' + dash + '"' : '') +
      (st === 'off' ? ' opacity="0.5"' : '') +
      ' marker-end="url(#garrow-' + (st === 'mixed' ? 'sel' :
        st === 'on' ? 'on' : st === 'off' ? 'off' : 'unk') + ')"/>';
    edges += '<path class="gmedge gmedge-hit" data-pair="' + esc(k) +
      '" d="' + d + '"/>';
  }
  document.getElementById('gmodedges').innerHTML = edges;
}
function gModLabel(k) {
  return GMODS[k] ? GMODS[k].label : k;
}
// ---- 局部展开视图（R2a 补修）：单模块真实节点/边有向图 ----
// 布局：列 = 模块内拓扑深度（Kahn，state 边不参与分层、画下弧）；
// 行 = stable_id 去末段后的真实族（如 attn.indexer.compressor vs
// attn.indexer，不依赖 G 之外的假数据）。跨模块边连到左右端口轨道
// （标注真实外部端点），保留全部原始边 ID，不新增假边。初始适配与
// "适配视图"按钮满足 GMOD_VIEW_KMIN 字号下限（见 gFitModule 定义处）。
let gModView = null;   // 当前局部展开（gModuleGraph 返回值）；定义见 gFitModule 前
function gModuleGraph(k) {
  const m = GMODS[k];
  const sub = new Set(m.nodes);
  const members = m.nodes.slice().sort();
  const modW = 190, modH = 34, modCellDy = 40, modColW = 232;
  const modPadL = 118, modPadT = 24, railW = 128;
  const laneOf = nid => {
    const parts = String(nid).split('.');
    return parts.length <= 1 ? nid : parts.slice(0, parts.length - 1).join('.');
  };
  const laneKeys = [];
  {
    const seen = new Set();
    for (const id of members) {
      const l = laneOf(id);
      if (!seen.has(l)) { seen.add(l); laneKeys.push(l); }
    }
  }
  const laneLabel = {};
  for (const l of laneKeys)
    laneLabel[l] = l.replace(/^(attn|block)\./, '');
  const flow = [], stateEdges = [];
  const fOut = {}, fIn = {};
  for (const id of members) { fOut[id] = []; fIn[id] = 0; }
  for (const e of GEDGES) {
    if (!sub.has(e.src_node_id) || !sub.has(e.dst_node_id)) continue;
    if (e.kind === 'state') { stateEdges.push(e); continue; }
    flow.push(e);
    fOut[e.src_node_id].push(e.dst_node_id);
    fIn[e.dst_node_id]++;
  }
  const mDepth = {};
  {
    const ready = members.filter(id => fIn[id] === 0);
    if (!ready.length && members.length)
      throw new Error('局部图分层失败（无入度 0 节点）: ' + k);
    const rem = {}; for (const id of members) rem[id] = fIn[id];
    for (const id of ready) mDepth[id] = 0;
    const q = ready.slice();
    while (q.length) {
      const u = q.shift();
      for (const v of fOut[u].slice().sort()) {
        mDepth[v] = Math.max(mDepth[v] || 0, mDepth[u] + 1);
        if (--rem[v] === 0) q.push(v);
      }
    }
    if (Object.keys(mDepth).length !== members.length)
      throw new Error('局部拓扑分层不完整: ' + k + ' ' +
        Object.keys(mDepth).length + '/' + members.length);
  }
  const mMaxDepth = Math.max.apply(null, members.map(id => mDepth[id]));
  const cellN = {};
  for (const id of members) {
    const kk = laneOf(id) + '|' + mDepth[id];
    cellN[kk] = (cellN[kk] || 0) + 1;
  }
  const laneY = {}, laneH = {};
  {
    let y = modPadT;
    for (const l of laneKeys) {
      let maxN = 1;
      for (let d = 0; d <= mMaxDepth; d++)
        maxN = Math.max(maxN, cellN[l + '|' + d] || 0);
      laneY[l] = y;
      laneH[l] = Math.max(64, modH + maxN * modCellDy + 14);
      y += laneH[l];
    }
  }
  const colX = d => modPadL + d * modColW;
  const lPos = {};
  {
    const idx = {};
    for (let d = 0; d <= mMaxDepth; d++)
      for (const id of members) {
        if (mDepth[id] !== d) continue;
        const l = laneOf(id);
        const kk = l + '|' + d;
        const i = idx[kk] || 0; idx[kk] = i + 1;
        lPos[id] = { x: colX(d) + modW / 2,
                     y: laneY[l] + modH / 2 + i * modCellDy };
      }
  }
  const svgW = colX(mMaxDepth) + modW / 2 + railW + 28;
  const lastLane = laneKeys[laneKeys.length - 1];
  const svgH = laneY[lastLane] + laneH[lastLane] + 18;
  // 左右端口轨道：跨模块边按入/出分组，锚点 y 在轨道高度内均分
  const inEdges = [], outEdges = [];
  for (const pk in GMOD_PAIRS) {
    if (pk.endsWith('>' + k)) inEdges.push(...GMOD_PAIRS[pk]);
    if (pk.startsWith(k + '>')) outEdges.push(...GMOD_PAIRS[pk]);
  }
  const railY0 = 24, railY1 = Math.max(svgH - 18, railY0 + 40);
  const railYs = n => Array.from({ length: n }, (_, i) =>
    n === 1 ? (railY0 + railY1) / 2
            : railY0 + (railY1 - railY0) * i / (n - 1));
  const inYs = railYs(inEdges.length), outYs = railYs(outEdges.length);
  // ---- 图元 ----
  let nodesHtml = '';
  for (const id of members) {
    const p = lPos[id], n = gnodeById[id], kind = gNodeKind(n);
    let label = gNodeLabel(id);
    if (label.length > 26) label = label.slice(0, 25) + '…';
    nodesHtml += '<g class="gnode" data-node="' + esc(id) +
      '" transform="translate(' + (p.x - modW / 2) + ',' + (p.y - modH / 2) + ')">' +
      '<title>' + esc(id) + ' · ' + esc(n.op_type) + '</title>' +
      '<rect width="' + modW + '" height="' + modH + '" rx="7"' +
      ' fill="' + GKIND_FILL[kind] + '" stroke="' + GKIND_STROKE[kind] + '"/>' +
      '<text x="8" y="15">' + esc(label) + '</text>' +
      '<text class="gtype" x="8" y="27">' + esc(n.op_type) + ' · ' +
      GKIND_ZH[kind] + '</text></g>';
  }
  let laneTags = '';
  for (const l of laneKeys)
    laneTags += '<text class="gllabel" x="14" y="' + (laneY[l] + 14) + '">' +
      esc(laneLabel[l]) + '</text>';
  const anchorX = { in: 40, out: svgW - 40 };
  let railsHtml = '';
  inEdges.forEach((e, i) => {
    const y = inYs[i];
    railsHtml += '<rect class="gport-anchor" x="' + (anchorX.in - 4) +
      '" y="' + (y - 4) + '" width="8" height="8" rx="2"/>' +
      '<text class="gport" x="6" y="' + Math.max(12, y - 7) +
      '">← ' + esc(gModLabel(gModuleOf(e.src_node_id))) + '</text>';
  });
  outEdges.forEach((e, i) => {
    const y = outYs[i];
    railsHtml += '<rect class="gport-anchor" x="' + (anchorX.out - 4) +
      '" y="' + (y - 4) + '" width="8" height="8" rx="2"/>' +
      '<text class="gport" x="' + (svgW - 6) + '" y="' + Math.max(12, y - 7) +
      '" text-anchor="end">→ ' + esc(gModLabel(gModuleOf(e.dst_node_id))) +
      '</text>';
  });
  const extEdges = [];
  inEdges.forEach((e, i) => extEdges.push(
    { e, fixed: 'in', x: anchorX.in, y: inYs[i] }));
  outEdges.forEach((e, i) => extEdges.push(
    { e, fixed: 'out', x: anchorX.out, y: outYs[i] }));
  const edgeGeos = [];
  const extPath = (x1, y1, x2, y2) => {
    const mx = (x1 + x2) / 2;
    return 'M ' + x1 + ' ' + y1 + ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' +
      y2 + ', ' + x2 + ' ' + y2;
  };
  for (const ee of extEdges) {
    const e = ee.e;
    let x1, y1, x2, y2;
    if (ee.fixed === 'in') {       // 入边：左锚点 → 模块内 dst 左缘
      x1 = ee.x; y1 = ee.y;
      x2 = lPos[e.dst_node_id].x - modW / 2 - 5;
      y2 = lPos[e.dst_node_id].y;
    } else {                        // 出边：模块内 src 右缘 → 右锚点
      x1 = lPos[e.src_node_id].x + modW / 2; y1 = lPos[e.src_node_id].y;
      x2 = ee.x; y2 = ee.y;
    }
    edgeGeos.push({ id: e.stable_id, d: extPath(x1, y1, x2, y2),
      kind: e.kind, guard: e.guard_ref || null });
  }
  for (const e of flow) {
    const s = lPos[e.src_node_id], t = lPos[e.dst_node_id];
    const x1 = s.x + modW / 2, y1 = s.y, x2 = t.x - modW / 2 - 5, y2 = t.y;
    const mx = (x1 + x2) / 2;
    edgeGeos.push({ id: e.stable_id,
      d: Math.abs(y2 - y1) < 2
        ? 'M ' + x1 + ' ' + y1 + ' L ' + x2 + ' ' + y2
        : 'M ' + x1 + ' ' + y1 + ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' +
          y2 + ', ' + x2 + ' ' + y2,
      kind: e.kind, guard: e.guard_ref || null });
  }
  for (const e of stateEdges) {    // 跨调用状态：下弧（不参与分层）
    const s = lPos[e.src_node_id], t = lPos[e.dst_node_id];
    const x1 = s.x + modW / 2, y1 = s.y + modH / 2;
    const x2 = t.x - modW / 2 - 5, y2 = t.y + modH / 2;
    let h = 0; for (const c of e.stable_id) h = (h * 31 + c.charCodeAt(0)) % 97;
    const dip = Math.min(120, Math.max(40, Math.abs(x1 - x2) * 0.12)) + h % 20;
    edgeGeos.push({ id: e.stable_id,
      d: 'M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + (y1 + dip) +
        ', ' + x2 + ' ' + (y2 + dip) + ', ' + x2 + ' ' + y2,
      kind: e.kind, guard: e.guard_ref || null });
  }
  // 边锚点（定位选中用）：每条边取路径中段坐标（M 后第一段 C/L 中点近似）
  const edgeAnchorById = {};
  for (const eg of edgeGeos) {
    const pts = eg.d.split(/[MLC, ]+/).filter(s => s !== '').map(Number);
    // 取路径首末控制点的中点（曲线中段近似，足够定位）
    const xs = pts.filter((_, i) => i % 2 === 0);
    const ys = pts.filter((_, i) => i % 2 === 1);
    const mid = Math.floor(xs.length / 2);
    edgeAnchorById[eg.id] = { x: xs[Math.min(mid, xs.length - 1)],
                              y: ys[Math.min(mid, ys.length - 1)] };
  }
  return { key: k, W: svgW, H: svgH, edgeGeos,
    nodesHtml, laneTags, railsHtml,
    members, mDepth, modW, modH,
    pos: lPos, edgeAnchor: edgeAnchorById,
    inEdges, outEdges, railX: anchorX, inYs, outYs,
    counts: { nodes: members.length, internal: flow.length + stateEdges.length,
              crossIn: inEdges.length, crossOut: outEdges.length } };
}
function gRenderModuleView(v) {
  // 局部视图复用完整图图层（glanes=族标签、gnodes/gedges=主图元、
  // gxrails=端口轨道），gmods/gmodedges 清空；跨边界边同样用
  // gedge 类（事件委托/选中标记共用）。三态按当前场景实时求值
  // （gModView 只存几何与 guard 引用，不冻结状态）。
  document.getElementById('glanes').innerHTML = v.laneTags;
  document.getElementById('gnodes').innerHTML = v.nodesHtml;
  let edgesHtml = '';
  const markerOf = (kind, st) => kind === 'state' ? 'state' :
    kind === 'select' ? 'sel' : kind === 'control' ? 'off' :
    st === 'on' ? 'on' : st === 'off' ? 'off' : 'unk';
  const dashOf = (kind, st) => kind === 'control' ? '2,3' :
    kind === 'data' && st === 'unknown' ? '5,4' : '';
  for (const eg of v.edgeGeos) {
    const st = eg.guard ? gGuardStatus({ guard_ref: eg.guard }, curScenObj)
                        : 'on';
    const col = eg.kind === 'state' ? GC.ok : eg.kind === 'select' ? GC.sel :
      eg.kind === 'control' ? GC.mut :
      st === 'on' ? GC.acc : st === 'off' ? GC.line : GC.warn;
    edgesHtml += '<path class="gedge" data-edge="' + esc(eg.id) + '" d="' +
      eg.d + '" stroke="' + col + '" stroke-width="1.4"' +
      (st === 'off' ? ' opacity="0.55"' : '') +
      (dashOf(eg.kind, st) ? ' stroke-dasharray="' + dashOf(eg.kind, st) + '"' : '') +
      ' marker-end="url(#garrow-' + markerOf(eg.kind, st) + ')"/>';
  }
  for (const eg of v.edgeGeos)
    edgesHtml += '<path class="gedge gedge-hit" data-edge="' + esc(eg.id) +
      '" d="' + eg.d + '"/>';
  document.getElementById('gedges').innerHTML = edgesHtml;
  document.getElementById('gmods').innerHTML = '';
  document.getElementById('gmodedges').innerHTML = '';
  document.getElementById('gxrails').innerHTML = v.railsHtml;
}
// ---- R2b 默认可读性：字号约束的局部适配 ----
// 节点标签 11px（.gnode text）；可读下限 12px 实际 CSS 像素 → 缩放下限
// KMIN = 12/11。适配优先满足字号下限（横向溢出可平移），起点从图内容左缘
// 开始（不居中裁掉左侧输入侧）；"查看全局范围"按钮才整体缩小（显式操作）。
const GMOD_VIEW_KMIN = 12 / 11;
function gFitModule(v) {
  const vp = document.getElementById('gviewport');
  const vw = vp.clientWidth || 1000, vh = vp.clientHeight || 640;
  const k = Math.max(GMOD_VIEW_KMIN, Math.min(vw / v.W, vh / v.H, 2.2));
  // 从内容左缘起始：可见输入侧；仅在内容比视口窄时居中
  const x = v.W * k <= vw ? (vw - v.W * k) / 2 : 8;
  const y = v.H * k <= vh ? (vh - v.H * k) / 2 : 8;
  gView = { x, y, k };
  gApplyView();
}
// 定位选中：按当前渲染模式取坐标（R2b 补修01：不得跨模式用坐标——
// 总览用模块盒（节点/边所属模块），局部展开用模块内坐标（外部项不定位、
// 明确说明），完整图用全图坐标；屏幕坐标必须落在视口内，否则视为失败
// 不动视图。缩放不足时提到可读下限。
function gLocateSelection() {
  const vp = document.getElementById('gviewport');
  const vw = vp.clientWidth || 1000, vh = vp.clientHeight || 640;
  const sum = document.getElementById('gscensum');
  const locateNote = msg => {
    if (sum) sum.innerHTML = '<span class="chip">' + esc(msg) + '</span>' +
      sum.innerHTML;
  };
  const apply = c => {
    const kMin = gMode === 'module' ? GMOD_VIEW_KMIN : GNODE_TEXT_KMIN;
    const k = Math.max(gView.k, kMin);
    const x = vw / 2 - c.x * k, y = vh / 2 - c.y * k;
    // 屏幕可见性守卫：定位后目标屏幕坐标必须在本视口内（单点目标以
    // 中心点校验；overview 盒目标由上方分支单独处理），否则不动视图
    const sx = x + c.x * k, sy = y + c.y * k;
    if (!(sx >= 0 && sx <= vw && sy >= 0 && sy <= vh)) {
      locateNote('定位失败：目标不在当前视图范围内（未移动）');
      return;
    }
    gView = { x, y, k };
    gApplyView();
  };
  let c = null;
  if (gMode === 'overview') {
    // 总览：定位节点/边所属的模块盒中心（模块坐标），不用完整图 gPos。
    // 缩放取 max(当前, 可读下限) 但夹到"整盒可入视口"内；盒比视口还小时
    // 用下限即可完整显示。
    if (gSel && (gSel.type === 'node' || gSel.type === 'edge')) {
      const nid = gSel.type === 'node' ? gSel.id
        : (gedgeById[gSel.id] ? gedgeById[gSel.id].src_node_id : null);
      const mk = nid != null ? gModuleOf(nid) : null;
      const p = mk && GMOD_POS[mk] ? GMOD_POS[mk] : null;
      if (p) {
        c = { x: p.x + GW2.mW / 2, y: p.y + GW2.mH / 2,
              w: GW2.mW, h: GW2.mH };
      } else if (mk) {
        locateNote('模块 ' + gModLabel(mk) + ' 在当前图中为空模块，' +
          '无可定位目标（未移动）');
        return;
      }
    } else if (gSel && gSel.type === 'pair') {
      const sMod = gSel.id.split('>')[0];
      const p = GMOD_POS[sMod];
      if (p) c = { x: p.x + GW2.mW / 2, y: p.y + GW2.mH / 2,
                   w: GW2.mW, h: GW2.mH };
    } else if (gSel && gSel.type === 'module' && GMOD_POS[gSel.id]) {
      const p = GMOD_POS[gSel.id];
      c = { x: p.x + GW2.mW / 2, y: p.y + GW2.mH / 2, w: GW2.mW, h: GW2.mH };
    }
    if (!c) { locateNote('当前选择没有总览可定位目标（未移动）'); return; }
    let k = Math.max(gView.k, GNODE_TEXT_KMIN);
    k = Math.min(k, vw / c.w, vh / c.h, 2.2);
    gView = { x: vw / 2 - c.x * k, y: vh / 2 - c.y * k, k };
    gApplyView();
    return;
  }
  if (gMode === 'module') {
    if (!gModView) return;
    if (gSel && gSel.type === 'module' && gSel.id === gModView.key) {
      gFitModule(gModView);          // 所选即当前模块：恢复可读适配
      return;
    }
    if (gSel && gSel.type === 'node' && gModView.pos[gSel.id]) {
      c = gModView.pos[gSel.id];
    } else if (gSel && gSel.type === 'edge' && gModView.edgeAnchor[gSel.id]) {
      c = gModView.edgeAnchor[gSel.id];
    } else {
      locateNote('局部展开视图只包含模块 ' + gModLabel(gModView.key) +
        ' 的成员；当前选择不在本模块内，未移动视图（可用"完整图"模式定位）');
      return;
    }
    apply(c);
    return;
  }
  // 完整图：节点用 gPos；边取两端中点
  if (gSel && gSel.type === 'node' && gPos[gSel.id]) {
    c = gPos[gSel.id];
  } else if (gSel && gSel.type === 'edge' && gedgeById[gSel.id]) {
    const e = gedgeById[gSel.id];
    const s = gPos[e.src_node_id], t = gPos[e.dst_node_id];
    if (s && t) c = { x: (s.x + t.x) / 2, y: (s.y + t.y) / 2 };
  }
  if (!c) { locateNote('当前选择在完整图中无可定位目标（未移动）'); return; }
  apply(c);
}
window.gLocateSelection = gLocateSelection;
// 查看全局范围：显式整体缩小到当前视图全部内容可见（可读性让位于范围，
// 非默认；再点"适配视图"恢复可读默认）
function gShowGlobal() {
  const vp = document.getElementById('gviewport');
  const vw = vp.clientWidth || 1000, vh = vp.clientHeight || 640;
  let W, H;
  if (gMode === 'overview') { W = GMOD_SVG_W; H = GMOD_SVG_H; }
  else if (gMode === 'module' && gModView) { W = gModView.W; H = gModView.H; }
  else { W = gSvgW; H = gSvgH; }
  const k = Math.min(vw / W, vh / H, 2.2);
  gView = { x: (vw - W * k) / 2, y: (vh - H * k) / 2, k };
  gApplyView();
}
window.gShowGlobal = gShowGlobal;
// 打开局部展开视图（真实节点/边图 + 列表补充；gSel 由调用方管理）
function gShowModulePanel(k) {
  const m = GMODS[k]; if (!m) return;
  const v = gModView && gModView.key === k ? gModView : gModuleGraph(k);
  const insList = (GMOD_INTERNAL[k] || []).map(e =>
    '<li>' + gEdgeLink(e.stable_id) + '：' +
    esc(e.src_node_id) + '.' + esc(e.src_port) + ' → ' +
    esc(e.dst_port) + ' <span class="chip">' + esc(e.kind) + '</span>' +
    (e.guard_ref ? ' guard:' + gGuardChip(e) : '') + '</li>').join('');
  const outs = [];
  for (const pk in GMOD_PAIRS) {
    if (!pk.startsWith(k + '>')) continue;
    const pairEdges = GMOD_PAIRS[pk];
    outs.push('<li><b>→ ' + esc(gModLabel(pk.slice(k.length + 1))) + '</b>（' +
      pairEdges.length + ' 条原始边）：' +
      pairEdges.map(e => gEdgeLink(e.stable_id) +
        '（' + esc(e.src_node_id) + '.' + esc(e.src_port) + ' → ' +
        esc(e.dst_node_id) + '.' + esc(e.dst_port) + '）').join('；') + '</li>');
  }
  const insAcross = [];
  for (const pk in GMOD_PAIRS) {
    if (!pk.endsWith('>' + k)) continue;
    const pairEdges = GMOD_PAIRS[pk];
    insAcross.push('<li><b>← ' +
      esc(gModLabel(pk.slice(0, pk.length - k.length - 1))) + '</b>（' +
      pairEdges.length + ' 条原始边）：' +
      pairEdges.map(e => gEdgeLink(e.stable_id) +
        '（' + esc(e.src_node_id) + '.' + esc(e.src_port) +
        ' → ' + esc(e.dst_node_id) + '.' + esc(e.dst_port) + '）').join('；') +
      '</li>');
  }
  const nodeRows = m.nodes.slice().sort().map(nid => {
    const n = gnodeById[nid];
    return '<tr><td><a href="#" onclick="gShowNode(\'' +
      String(nid).replace(/'/g, '') + '\');return false;">' +
      esc(nid) + '</a></td><td>' + esc(n.op_type) + '</td></tr>';
  }).join('');
  document.getElementById('gmodulepanel').innerHTML =
    '<div class="gmodhead">' +
    '<button class="sel" onclick="gBackToOverview()">← 返回模块总览</button>' +
    '<b>' + esc(m.label) + '</b> <span class="sub">局部图：' +
    v.counts.nodes + ' 真实节点 · 内部边 ' + v.counts.internal +
    ' · 跨模块入 ' + v.counts.crossIn + ' / 出 ' + v.counts.crossOut +
    '（列 = 模块内拓扑深度，行 = stable_id 真实族；跨模块边连左右端口轨道，' +
    '标注真实外部端点，保留全部原始边 ID）</span></div>' +
    '<details><summary>节点表 / 边列表（图的补充，点击节点/边 ID 看详情）</summary>' +
    '<h4>真实节点（点击看节点详情）</h4><table><tr><th>节点</th><th>op</th></tr>' +
    nodeRows + '</table>' +
    (insList ? '<h4>模块内部边（' + insList.length + '，含真实端口）</h4><ul>' +
      insList + '</ul>' : '') +
    (outs.length ? '<h4>跨模块出边（真实端口与原始边 ID）</h4><ul>' +
      outs.join('') + '</ul>' : '') +
    (insAcross.length ? '<h4>跨模块入边（真实端口与原始边 ID）</h4><ul>' +
      insAcross.join('') + '</ul>' : '') +
    '</details>';
}
function gOpenModule(k) {
  const m = GMODS[k]; if (!m) return;
  const v = gModuleGraph(k);
  gModView = v;
  gMode = 'module';
  gRenderModuleView(v);
  gFitModule(v);
  gUpdateModeButtons();
  gShowModulePanel(k);
  gMarkSelection(gSel);   // gSel 由调用方设置；此处按其刷新图元高亮
}
window.gOpenModule = gOpenModule;
function gBackToOverview() {
  gSel = null; gModView = null;
  gSetMode('overview');
  document.getElementById('gmodulepanel').innerHTML = '';
}
window.gBackToOverview = gBackToOverview;
// 兼容旧入口：gShowModule 打开局部展开视图（选择状态单一来源 gSel）
function gShowModule(k) {
  gSel = { type: 'module', id: k };
  gMarkSelection(gSel);
  gOpenModule(k);
}
window.gShowModule = gShowModule;
function gClearModulePanel() {
  document.getElementById('gmodulepanel').innerHTML = '';
  document.querySelectorAll('.gmod').forEach(el =>
    el.classList.toggle('gsel', false));
}
window.gClearModulePanel = gClearModulePanel;

const GPEND = 'R2b 范围说明：默认视图为当前图的紧凑模块总览（固定语义模块 + 真实' +
  '节点计数，空模块不显示；跨模块投影边保留全部原始 edge ID，模块内部边单列；' +
  '投影 ∪ 内部 = 全部当前图边，渲染前守恒断言）。点击模块进入局部展开视图：' +
  '该模块内部真实节点/边的有向图（列 = 模块内拓扑深度，行 = stable_id 真实族；' +
  '跨模块边连左右端口轨道并标注真实外部端点，不新增假边），节点表/边列表保留' +
  '为补充；默认适配保证节点标签实际 ≥12px（横向溢出可平移，起点从输入侧起），' +
  '"定位选中"把所选节点/边居中，"查看全局范围"显式整体缩小（非默认），' +
  '"适配视图"恢复可读默认。"完整图"模式保留全节点/边视图与平移缩放适配。' +
  '四图（工具条"图"按钮）共用此渲染管线，切换后按所选图重建；两套场景按钮' +
  '（工具条/场景面板）共用统一入口，状态始终一致。';

// ---- 详情面板（点击节点/边）----
function gTensorLink(tid) {
  const safe = String(tid).replace(/'/g, '');
  return '<a href="#" onclick="showTensor(\'' + safe +
    '\');return false;">' + esc(String(tid).replace(/^tensor:[^:]*:/, '')) +
    '</a>';
}
function gEdgeLink(eid) {
  const safe = String(eid).replace(/'/g, '');
  return '<a href="#" onclick="gShowEdgeById(\'' + safe +
    '\');return false;">' + esc(eid) + '</a>';
}
function gGuardChip(e) {
  const st = gGuardStatus(e, curScenObj);
  if (st === 'on') return '<span class="chip ok">' + esc(e.guard_ref) + '</span>';
  if (st === 'off') return '<span class="chip">' + esc(e.guard_ref) + '</span>';
  return '<span class="chip warn">' + esc(e.guard_ref) + ':unknown</span>';
}
function gShowNode(nid) {
  const n = gnodeById[nid]; if (!n) return;
  const ins = GEDGES.filter(e => e.dst_node_id === nid)
    .map(e => '<li>' + gEdgeLink(e.stable_id) + '：' +
      esc(e.src_node_id) + '.' + esc(e.src_port) + ' → ' +
      esc(e.dst_port) + ' <span class="chip">' + esc(e.kind) + '</span>' +
      (e.guard_ref ? ' guard:' + gGuardChip(e) : '') +
      ' · ' + gTensorLink(e.tensor_id) + '</li>').join('');
  const outs = GEDGES.filter(e => e.src_node_id === nid)
    .map(e => '<li>' + gEdgeLink(e.stable_id) + '：' +
      esc(e.src_port) + ' → ' + esc(e.dst_node_id) + '.' +
      esc(e.dst_port) + ' <span class="chip">' + esc(e.kind) + '</span>' +
      (e.guard_ref ? ' guard:' + gGuardChip(e) : '') +
      ' · ' + gTensorLink(e.tensor_id) + '</li>').join('');
  // 该节点端口张量涉及的别名关系（真实 alias_relations；注明非 data 边）
  const tids = {};
  for (const pt of n.input_ports) tids[pt.tensor_id] = 1;
  for (const pt of n.output_ports) tids[pt.tensor_id] = 1;
  const aliasRows = [];
  for (const a of ALIASES) {
    if (tids[a.base_tensor_id] || tids[a.alias_tensor_id])
      aliasRows.push('<li>' + esc(a.relation_kind) + '：' +
        gTensorLink(a.base_tensor_id) + ' ↔ ' +
        gTensorLink(a.alias_tensor_id) +
        (a.transform_expr ? '（' + esc(a.transform_expr) + '）' : '') +
        ' <span class="chip">alias（张量别名关系，非 data 边）</span> ' +
        ((a.evidence_refs || []).map(id => evBtn(id)).join(' ') || '') +
        '</li>');
  }
  const inPorts = n.input_ports.map(pt =>
    '<li>' + esc(pt.name) + ' ← ' + gTensorLink(pt.tensor_id) + '</li>').join('');
  const outPorts = n.output_ports.map(pt =>
    '<li>' + esc(pt.name) + ' → ' + gTensorLink(pt.tensor_id) + '</li>').join('');
  const ev = (n.evidence_refs || []).map(id => evBtn(id)).join(' ') || '—';
  document.getElementById('gdetail').innerHTML =
    '<details open><summary>节点 ' + esc(n.stable_id) + '</summary>' +
    '<p class="kv"><b>op</b> ' + esc(n.op_type) + ' · <b>类型</b> ' +
    GKIND_ZH[gNodeKind(n)] + ' · <b>granularity</b> ' + esc(n.granularity) +
    ' · <b>位置</b> 第 ' + gPos[nid].depth + ' 列 / ' +
    esc(gLaneLabelMapCur()[gLaneOf(n)]) + '</p>' +
    '<p class="kv"><b>证据</b> ' + ev + '</p>' +
    '<div class="gcols"><div><h4>输入端口</h4><ul>' + inPorts +
    '</ul></div><div><h4>输出端口</h4><ul>' + outPorts +
    '</ul></div></div>' +
    '<h4>入边（' + GEDGES.filter(e => e.dst_node_id === nid).length +
    '）</h4><ul>' + ins + '</ul>' +
    '<h4>出边（' + GEDGES.filter(e => e.src_node_id === nid).length +
    '）</h4><ul>' + outs + '</ul>' +
    (aliasRows.length ? '<h4>张量别名（alias_relations，非 data 边）</h4><ul>' +
      aliasRows.join('') + '</ul>' : '') +
    '</details>';
  gSel = { type: 'node', id: nid };
  gMarkSelection(gSel);
}
function gShowEdge(eid) {
  const e = gedgeById[eid]; if (!e) return;
  const st = gGuardStatus(e, curScenObj);
  const stZh = st === 'on'
    ? '<span class="chip ok">活动（guard 成立或无条件）</span>'
    : st === 'off'
    ? '<span class="chip">本场景不活动（guard 不成立）</span>'
    : '<span class="chip warn">unknown（guard 未知，未伪作 false）</span>';
  const guardExpr = e.guard_ref
    ? ((ctx.G.guards.find(x => x.stable_id === e.guard_ref) || {}).predicate_expr
       || '') : '';
  const t = ctx.tensorById[e.tensor_id];
  const axesSummary = t
    ? '<p class="kv"><b>张量</b> ' + gTensorLink(t.stable_id) + ' · dtype ' +
      esc(t.dtype) + ' · shape(' + esc(t.logical_layout.join(',')) + ')</p>'
    : '<p class="kv"><b>张量</b> ' + esc(e.tensor_id) + '（不在本图张量表）</p>';
  const tr = e.state_transition_ref
    ? '<p class="kv"><b>状态版本</b> ' + esc(e.state_transition_ref) +
      '（before→after，跨调用）</p>' : '';
  const ev = (e.evidence_refs || []).map(id => evBtn(id)).join(' ') || '—';
  document.getElementById('gdetail').innerHTML =
    '<details open><summary>边 ' + esc(e.stable_id) + '</summary>' +
    '<p class="kv"><b>连接</b> ' + esc(e.src_node_id) + '.' +
    esc(e.src_port) + ' → ' + esc(e.dst_node_id) + '.' +
    esc(e.dst_port) + '</p>' +
    '<p class="kv"><b>类别</b> ' + esc(e.kind) + ' · ' + stZh + '</p>' +
    (e.guard_ref ? '<p class="kv"><b>guard</b> ' + esc(e.guard_ref) +
      ' · <code>' + esc(guardExpr) + '</code></p>' : '') +
    axesSummary + tr +
    '<p class="kv"><b>证据</b> ' + ev + '</p></details>';
  gSel = { type: 'edge', id: eid };
  gMarkSelection(gSel);
}
// 模块间投影边详情：该 pair 全部原始边（不虚构单边）；选择状态 type='pair'
function gShowPair(pk) {
  const pairEdges = GMOD_PAIRS[pk] || [];
  const sMod = pk.split('>')[0], dMod = pk.split('>')[1];
  const st = gPairGuardStatus(pairEdges, curScenObj);
  const stZh = st === 'on'
    ? '<span class="chip ok">活动（guard 成立或无条件）</span>'
    : st === 'off'
    ? '<span class="chip">本场景不活动（guard 不成立）</span>'
    : st === 'unknown'
    ? '<span class="chip warn">unknown（guard 未知，未伪作 false）</span>'
    : '<span class="chip" style="color:' + GC.sel + '">mixed（各边 guard 状态不一致，逐边见下）</span>';
  document.getElementById('gmodulepanel').innerHTML =
    '<details open><summary>模块间投影 ' + esc(gModLabel(sMod)) +
    ' → ' + esc(gModLabel(dMod)) + '（' + pairEdges.length +
    ' 条原始边）</summary>' +
    '<p class="kv"><b>聚合状态</b> ' + stZh + '</p><ul>' +
    pairEdges.map(e => '<li>' + gEdgeLink(e.stable_id) + '：' +
      esc(e.src_node_id) + '.' + esc(e.src_port) + ' → ' +
      esc(e.dst_node_id) + '.' + esc(e.dst_port) +
      ' <span class="chip">' + esc(e.kind) + '</span>' +
      (e.guard_ref ? ' guard:' + gGuardChip(e) : '') + '</li>').join('') +
    '</ul></details>';
  gSel = { type: 'pair', id: pk };
  gMarkSelection(gSel);
}
window.gShowPair = gShowPair;
window.gShowNode = gShowNode;
window.gShowEdgeById = function(eid) { gShowEdge(eid); };

// ---- 选择状态（R2a 补修）：module/pair/node/edge 互斥单一来源 ----
let gSel = null;   // {type:'module'|'pair'|'node'|'edge', id} | null
function gMarkSelection(sel) {
  const wantNode = sel && sel.type === 'node' ? sel.id : null;
  const wantEdge = sel && sel.type === 'edge' ? sel.id : null;
  const wantPair = sel && sel.type === 'pair' ? sel.id : null;
  document.querySelectorAll('.gnode').forEach(el =>
    el.classList.toggle('gsel', wantNode != null &&
      el.dataset && el.dataset.node === wantNode));
  document.querySelectorAll('.gedge').forEach(el =>
    el.classList.toggle('gsel', wantEdge != null &&
      el.dataset && el.dataset.edge === wantEdge));
  document.querySelectorAll('.gmod').forEach(el =>
    el.classList.toggle('gsel', sel && sel.type === 'module' &&
      el.dataset && el.dataset.mod === sel.id));
  document.querySelectorAll('.gmedge').forEach(el =>
    el.classList.toggle('gsel', wantPair != null &&
      el.dataset && el.dataset.pair === wantPair));
}
let curScenObj = SCEN.find(v => v.scenario_id === curScen) || SCEN[0];
// 当前视图模式：overview（模块总览，默认）/ module（局部展开）/ full（R1 完整图）
let gMode = 'overview';
// gApplyScenario 按当前选择类型刷新详情（module/pair/node/edge 互斥）。
// module/pair 概念只属总览/局部视图（完整图模式面板清空，不显示旧详情）；
// 模块视图内选中节点/边时同步刷新模块面板（guard 三态不陈旧）。
function gRefreshSelectionDetail() {
  if (!gSel) return;
  if (gSel.type === 'module') {
    if (gMode !== 'full') gShowModulePanel(gSel.id);
  } else if (gSel.type === 'pair') {
    if (gMode !== 'full') gShowPair(gSel.id);
  } else if (gSel.type === 'node') {
    gShowNode(gSel.id);
    if (gMode === 'module' && gModView) gShowModulePanel(gModView.key);
  } else if (gSel.type === 'edge') {
    gShowEdge(gSel.id);
    if (gMode === 'module' && gModView) gShowModulePanel(gModView.key);
  }
}
function gApplyScenario() {
  curScenObj = ctx.scen.find(v => v.scenario_id === curScen)
    || ctx.scen[0] || { scenario_id: curScen, guard_activations: {},
                        env: {}, derived: {}, indexer: {} };
  // 场景切换重渲染当前模式（overview/module/full），guard 三态同步
  if (gMode === 'overview') gRenderOverview(curScenObj);
  else if (gMode === 'module' && gModView) gRenderModuleView(gModView);
  else gRenderEdges(curScenObj);
  document.querySelectorAll('#gscenbtns button').forEach(b =>
    b.className = 'sel' + (b.dataset && b.dataset.id === curScen ? ' on' : ''));
  document.querySelectorAll('#scenbtns button').forEach(b =>
    b.className = 'sel' + (b.dataset && b.dataset.id === curScen ? ' on' : ''));
  const s = curScenObj;
  const act = Object.keys(s.guard_activations)
    .filter(k => s.guard_activations[k].state === 'true').join(', ');
  document.getElementById('gscensum').innerHTML =
    '<b>当前图</b> ' + esc(VAR_LABEL[curVar]||curVar) +
    ' · <b>当前场景</b> ' + esc(s.scenario_id) +
    ' · 边三态：活动 ' + GEDGES.filter(e => gGuardStatus(e, s) === 'on').length +
    ' / 不活动 ' + GEDGES.filter(e => gGuardStatus(e, s) === 'off').length +
    ' / unknown ' + GEDGES.filter(e => gGuardStatus(e, s) === 'unknown').length +
    '（guard 成立：' + esc(act || '—') + '）';
  // 已选中的模块/投影/节点/边详情随场景刷新（guard 三态同步，不沿用旧
  // 状态）；重渲染后按 gSel 恢复图元高亮。
  gRefreshSelectionDetail();
  if (gSel) gMarkSelection(gSel);
}
// 统一场景入口：两套按钮（图工具条 / 下方场景面板）都走这里，
// curScen/curScenObj/两处按钮高亮/场景面板/图边三态/已选详情一次同步。
// 场景 ID 不属于当前图时拒绝（按钮只渲染有效场景，此处为防御）。
function gSetScenario(sid) {
  if (!ctx.scenIds.has(sid)) {
    document.getElementById('gscensum').innerHTML =
      '<b>场景 ' + esc(sid) + ' 不属于图 ' + esc(VAR_LABEL[curVar]||curVar) +
      '</b>（未切换；有效场景按钮见两侧任一场景组）';
    return;
  }
  curScen = sid;
  renderScen();
  gApplyScenario();
}
window.gSetScenario = gSetScenario;

// ---- 模式切换（模块总览 / 局部展开 / 完整图）----
function gUpdateModeButtons() {
  const on = 'sel on', off = 'sel';
  document.getElementById('gbtn-mod').className =
    gMode === 'full' ? off : on;   // 总览与局部展开都点亮"模块总览"主按钮
  document.getElementById('gbtn-full').className =
    gMode === 'full' ? on : off;
}
function gSetMode(m) {
  gMode = m;
  gUpdateModeButtons();
  const lanes = document.getElementById('glanes');
  const edges = document.getElementById('gedges');
  const nodes = document.getElementById('gnodes');
  const mods = document.getElementById('gmods');
  const modedges = document.getElementById('gmodedges');
  if (m === 'overview') {
    lanes.innerHTML = '';
    edges.innerHTML = ''; nodes.innerHTML = '';
    document.getElementById('gxrails').innerHTML = '';
    gModView = null;
    gRenderOverview(curScenObj);
    gFitOverview();
    // 模块选择的列表/高亮由 gApplyScenario → gRefreshSelectionDetail 恢复
  } else {
    mods.innerHTML = ''; modedges.innerHTML = '';
    document.getElementById('gxrails').innerHTML = '';
    document.getElementById('gmodulepanel').innerHTML = '';
    gModView = null;
    if (gSel && gSel.type === 'module') gSel = null;
    gRenderLanes();
    gRenderNodes();
    gRenderEdges(curScenObj);
    gFit();
  }
  gApplyScenario();
}
window.gSetMode = gSetMode;

// ---- 图切换（R2b 四图统一）：按所选图重建全部渲染上下文 ----
// 选择同步规则：模块/投影对选择清空（模块集合随图变化，不可跨图保留）；
// 节点/边选择仅在 ID 存在于新图时保留，否则清空并说明；场景 ID 有效则
// 保留，否则改选该图默认场景，两套按钮随 gRebuildScenButtons 同步重建。
let gVariantNote = '';
function gSetVariant(vkey) {
  if (!CTXS[vkey]) return;
  const prevSel = gSel;
  const prevScen = curScen;
  curVar = vkey;
  ctx = CTXS[vkey];
  gVariantNote = '';
  // 重建当前图派生数据（图/布局/模块归属/场景视图），清残留视图状态
  gInitGraphCtx();
  gInitModuleCtx();
  gModView = null;
  gMode = 'overview';
  gUpdateModeButtons();
  // 场景同步：有效保留，无效改默认（curScenObj 与渲染同步更新）
  if (!ctx.scenIds.has(curScen)) {
    if (ctx.defaultScen) {
      gVariantNote = '场景 ' + esc(prevScen) + ' 不属于图 ' +
        esc(VAR_LABEL[vkey]) + '，已改选默认 ' + esc(ctx.defaultScen) + '。';
      curScen = ctx.defaultScen;
    }
  }
  curScenObj = ctx.scen.find(v => v.scenario_id === curScen)
    || ctx.scen[0] || { scenario_id: curScen, guard_activations: {},
                        env: {}, derived: {}, indexer: {} };
  // 选择同步：节点/边仅 ID 存在时保留；模块/投影对清空
  if (prevSel) {
    if (prevSel.type === 'node' && gnodeById[prevSel.id]) {
      gSel = prevSel;
    } else if (prevSel.type === 'edge' && gedgeById[prevSel.id]) {
      gSel = prevSel;
    } else {
      if (prevSel.type === 'module' || prevSel.type === 'pair')
        gVariantNote += '已清除模块/投影选择（模块归属随图重建）。';
      else
        gVariantNote += '已清除选择：' + esc(prevSel.type) + ' ' +
          esc(prevSel.id) + ' 不在图 ' + esc(VAR_LABEL[vkey]) + ' 中。';
      gSel = null;
    }
  }
  // 重建按钮/表格/图渲染
  gRebuildScenButtons();
  renderL1Table();
  renderStatePanel();
  renderScen();
  gRenderGraphNote();
  gRefreshEvDetail();
  document.querySelectorAll('#gbtn-var').forEach(b =>
    b.className = 'sel' + (b.dataset && b.dataset.var === gBtnKey(vkey)
      ? ' on' : ''));
  // 默认总览渲染（清空各图层避免残留旧图 ID）
  document.getElementById('glanes').innerHTML = '';
  document.getElementById('gedges').innerHTML = '';
  document.getElementById('gnodes').innerHTML = '';
  document.getElementById('gxrails').innerHTML = '';
  document.getElementById('gmodulepanel').innerHTML = '';
  document.getElementById('gdetail').innerHTML = '';
  gRenderOverview(curScenObj);
  gFitOverview();
  gApplyScenario();
  // 变体切换说明放图摘要前（渲染后追加一次）
  if (gVariantNote) {
    const el = document.getElementById('gscensum');
    el.innerHTML = '<span class="chip">' + gVariantNote + '</span>' + el.innerHTML;
  }
  // B4 补修01：实例角色/参数绑定面板随图刷新（DSpark 图渲染真实绑定，
  // 其它图显示归属说明——不残留上一图数据）
  if (typeof renderDsparkRolePanel === 'function') renderDsparkRolePanel();
}
window.gSetVariant = gSetVariant;

// R2b 补修01：图上方说明按当前图派生数据生成（唯一入口；gInit 在
// gInitGraphCtx/gInitModuleCtx 之后调用，gSetVariant 每次切图后调用，
// 不再出现"0 节点/0 边"或切图后不更新的旧说明）。
function gRenderGraphNote() {
  const srcNote = ctx.isDspark
    ? '，ID 集合与 data/dspark/semantic_graph.json 完全一致'
    : '，ID 集合与 semantic_graph.json / data/variants/&lt;key&gt;/semantic_graph.json ' +
      '完全一致';
  document.getElementById('l1graphnote').innerHTML =
    '<b>当前图</b> ' + esc(VAR_LABEL[curVar]||curVar) + '（工具条可切五图） · ' +
    '<b>全部真实节点</b> ' + GNODES.length + ' 个 · <b>全部真实边</b> ' +
    GEDGES.length + ' 条（data ' +
    GEDGES.filter(e => e.kind === 'data').length + ' · state ' +
    GEDGES.filter(e => e.kind === 'state').length + ' · select ' +
    GEDGES.filter(e => e.kind === 'select').length + ' · control ' +
    GEDGES.filter(e => e.kind === 'control').length +
    '）' + srcNote +
    '。默认模块总览：固定语义模块' +
    '（真实节点计数）+ 模块间投影边（' + Object.keys(GMOD_PAIRS).length +
    ' 对，共 ' + Object.values(GMOD_PAIRS).reduce((a, p) => a + p.length, 0) +
    ' 条原始边；模块内部另有 ' +
    Object.values(GMOD_INTERNAL).reduce((a, p) => a + p.length, 0) +
    ' 条内部边，三者对全部边守恒）。点击模块进入局部展开视图，点击' +
    '"完整图"看全部 ' + GNODES.length + ' 节点/' + GEDGES.length +
    ' 边拓扑（列 = 数据流拓扑深度，行 = 语义分区）。拖拽平移、滚轮或按钮缩放。';
}

// ---- 平移 / 缩放 / 适配（三模式共用 viewport；fit 按当前模式真实尺寸）----
let gView = { x: 12, y: 8, k: 0.9 };
function gApplyView() {
  const r = document.getElementById('groot');
  // 防御：非浏览器环境（JS 单元测试 DOM 桩）无 setAttribute 时跳过视图应用
  if (!r.setAttribute) return;
  r.setAttribute('transform',
    'translate(' + gView.x + ',' + gView.y + ') scale(' + gView.k + ')');
  document.getElementById('gzoomlabel').textContent =
    Math.round(gView.k * 100) + '%';
}
// 适配视图按钮入口：按当前模式真实尺寸适配（总览/局部展开/完整图各有
// 自己的 W/H；不得用完整图尺寸缩总览）。完整图默认适配同样受可读字号
// 下限约束（标签实际 ≥12px；想一眼看全图请用"查看全局范围"，显式操作）。
function gFit() {
  if (gMode === 'overview') { gFitOverview(); return; }
  if (gMode === 'module' && gModView) { gFitModule(gModView); return; }
  gFitFull();
}
function gFitFull() {
  const vp = document.getElementById('gviewport');
  const vw = vp.clientWidth || 1000, vh = vp.clientHeight || 640;
  const k = Math.max(GNODE_TEXT_KMIN, Math.min(vw / gSvgW, vh / gSvgH, 2.2));
  const x = gSvgW * k <= vw ? (vw - gSvgW * k) / 2 : 8;
  const y = gSvgH * k <= vh ? (vh - gSvgH * k) / 2 : 8;
  gView = { x, y, k };
  gApplyView();
}
function gFitOverview() {
  const vp = document.getElementById('gviewport');
  const vw = vp.clientWidth || 1000, vh = vp.clientHeight || 640;
  const k = Math.min(vw / GMOD_SVG_W, vh / GMOD_SVG_H, 2.2);
  gView = { x: (vw - GMOD_SVG_W * k) / 2, y: (vh - GMOD_SVG_H * k) / 2, k };
  gApplyView();
}
function gZoomBy(f) {
  const vp = document.getElementById('gviewport');
  const cx = (vp.clientWidth || 1000) / 2, cy = (vp.clientHeight || 640) / 2;
  const k2 = Math.min(8, Math.max(0.08, gView.k * f));
  gView.x = cx - (cx - gView.x) * (k2 / gView.k);
  gView.y = cy - (cy - gView.y) * (k2 / gView.k);
  gView.k = k2;
  gApplyView();
}
window.gFit = gFit; window.gZoomBy = gZoomBy;
(function gBindPanZoom() {
  const vp = document.getElementById('gviewport');
  // 防御：非浏览器环境（JS 单元测试 DOM 桩）无 addEventListener 时
  // 跳过指针/滚轮绑定；真实浏览器始终绑定。
  if (!vp.addEventListener) return;
  // 拖拽阈值：移动超过 5px 才算拖拽；pointerdown 时把 pointer capture
  // 设到 vp 自身（原实现无条件 capture 已使 gsvg 上 click 事件重定向，
  // 现改为仅拖拽成立后 capture，且 click 判定使用位移阈值，避免误选）。
  let drag = null, moved = false;
  vp.addEventListener('pointerdown', ev => {
    drag = { x: ev.clientX, y: ev.clientY, vx: gView.x, vy: gView.y };
    moved = false;
    vp.classList.add('ggrabbing');
  });
  vp.addEventListener('pointermove', ev => {
    if (!drag) return;
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    if (!moved && Math.hypot(dx, dy) < 5) return;   // 阈值内不算拖拽
    if (!moved && vp.setPointerCapture) {
      try { vp.setPointerCapture(ev.pointerId); } catch (e) { /* 已释放 */ }
    }
    moved = true;
    gView.x = drag.vx + dx;
    gView.y = drag.vy + dy;
    gApplyView();
  });
  const end = ev => {
    if (drag && !moved && ev && vp.releasePointerCapture &&
        ev.pointerId != null) {
      try { vp.releasePointerCapture(ev.pointerId); } catch (e) { /* 无捕获 */ }
    }
    drag = null; moved = false; vp.classList.remove('ggrabbing');
  };
  vp.addEventListener('pointerup', end);
  vp.addEventListener('pointercancel', ev => { drag = null; moved = false;
    vp.classList.remove('ggrabbing'); });
  vp.addEventListener('wheel', ev => {
    ev.preventDefault();
    const rect = vp.getBoundingClientRect();
    const cx = ev.clientX - rect.left, cy = ev.clientY - rect.top;
    const f = ev.deltaY < 0 ? 1.15 : 0.87;
    const k2 = Math.min(8, Math.max(0.08, gView.k * f));
    gView.x = cx - (cx - gView.x) * (k2 / gView.k);
    gView.y = cy - (cy - gView.y) * (k2 / gView.k);
    gView.k = k2;
    gApplyView();
  }, { passive: false });
})();

// ---- 事件委托（点击节点/边/模块/投影边；SVG innerHTML 重建后仍有效）----
// 点击判定由位移阈值保证：pointerdown→up 之间移动 <5px 视为点击。
// 选择状态互斥单一来源 gSel：module/pair/node/edge 相互清除。
(function gBindClicks() {
  const svg = document.getElementById('gsvg');
  // 防御：非浏览器环境（JS 单元测试 DOM 桩）无 addEventListener 时跳过
  if (!svg.addEventListener) return;
  svg.addEventListener('click', ev => {
    const t = ev.target;
    const modEl = t.closest ? t.closest('.gmod') : null;
    if (modEl && modEl.dataset && modEl.dataset.mod) {
      gShowModule(modEl.dataset.mod);
      return;
    }
    const pairEl = t.closest ? t.closest('.gmedge-hit') : null;
    if (pairEl && pairEl.dataset && pairEl.dataset.pair) {
      gShowPair(pairEl.dataset.pair);
      return;
    }
    const nodeEl = t.closest ? t.closest('.gnode') : null;
    if (nodeEl && nodeEl.dataset && nodeEl.dataset.node) {
      gSel = { type: 'node', id: nodeEl.dataset.node };
      gShowNode(gSel.id);
      if (gMode === 'module' && gModView) gShowModulePanel(gModView.key);
      return;
    }
    const edgeEl = t.closest ? t.closest('.gedge-hit') : null;
    if (edgeEl && edgeEl.dataset && edgeEl.dataset.edge) {
      gSel = { type: 'edge', id: edgeEl.dataset.edge };
      gShowEdge(gSel.id);
      if (gMode === 'module' && gModView) gShowModulePanel(gModView.key);
      return;
    }
  });
})();

// ---- 初始化 ----
(function gInit() {
  document.getElementById('gdefs').innerHTML =
    '<marker id="garrow-on" viewBox="0 0 10 10" refX="9" refY="5"' +
    ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="' + GC.acc + '"/></marker>' +
    '<marker id="garrow-off" viewBox="0 0 10 10" refX="9" refY="5"' +
    ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="' + GC.mut + '"/></marker>' +
    '<marker id="garrow-unk" viewBox="0 0 10 10" refX="9" refY="5"' +
    ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="' + GC.warn + '"/></marker>' +
    '<marker id="garrow-state" viewBox="0 0 10 10" refX="9" refY="5"' +
    ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="' + GC.ok + '"/></marker>' +
    '<marker id="garrow-sel" viewBox="0 0 10 10" refX="9" refY="5"' +
    ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="' + GC.sel + '"/></marker>';
  // 默认模块总览：先重建派生数据，再生成图说明（R2b 补修01：说明读取
  // 已就绪的 GNODES/GEDGES/GMOD_PAIRS，不再是 0 节点/0 边）并渲染总览
  gInitGraphCtx();
  gInitModuleCtx();
  gRebuildScenButtons();
  renderL1Table();
  gRenderGraphNote();
  curScenObj = ctx.scen.find(v => v.scenario_id === curScen) || ctx.scen[0];
  renderScen();
  document.querySelectorAll('#gbtn-var').forEach(b =>
    b.className = 'sel' + (b.dataset && b.dataset.var === gBtnKey(curVar)
      ? ' on' : ''));
  gRenderLanes();
  gRenderNodes();
  gSetMode('overview');
  document.getElementById('l1graphpending').textContent = GPEND;
})();

// ---- 状态版本面板（当前图 ctx.G；切图后刷新）----
(function(){
  const st = document.createElement('details');
  st.className = 'card';
  st.innerHTML = '<summary>状态与版本（cache / 压缩状态，跨调用；当前图）</summary><div id="statepanel"></div>';
  document.querySelector('main').insertBefore(st, document.getElementById('planview'));
})();
function renderStatePanel() {
  const G0 = ctx.G;
  const byState = {};
  for (const v of G0.state_versions)
    (byState[v.state_id] = byState[v.state_id] || []).push(v.version_label);
  let html = '<p class="kv">当前图 <b>'+esc(VAR_LABEL[curVar]||curVar)+'</b>（'+
    esc(G0.template_id)+'）</p>';
  html += '<table><tr><th>状态</th><th>版本（历史 → 本调用后）</th><th>说明</th></tr>';
  for (const tr of G0.state_transitions) {
    const before = tr.before_version_id.split(':').pop();
    const after = tr.after_version_id.split(':').pop();
    html += '<tr><td><code>'+esc(tr.state_id)+'</code></td><td>'+
      esc(before)+' → '+esc(after)+
      (before==='v_hist'?'（历史累积，内容 unknown）':'')+
      '</td><td class="sub">'+esc(byState[tr.state_id]?'见版本列':'')+'</td></tr>';
  }
  html += '</table>';
  if (ctx.isMain) {
    html += '<p class="kv"><b>decode 连续调用版本链</b>（同一 state_id 的版本号演进，'+
      '来源 instance_bindings.json；主图 csa_topk 压缩状态）：</p><table><tr><th>步</th><th>位置</th><th>写槽</th><th>触发池化</th><th>压缩区写槽</th></tr>';
    for (const c of (P.plan.decode_version_chain||[]))
      html += '<tr><td>v'+c.step+'</td><td>start_pos='+c.start_pos+'</td><td>第'+
        (c.write_slot+1)+' 槽</td><td>'+(c.should_compress?'是（读状态池化）':'否')+
        '</td><td>'+(c.cache_write_slot==null?'—':c.cache_write_slot)+'</td></tr>';
    html += '</table>';
  } else {
    html += '<p class="sub">decode 版本链面板为主图压缩状态（instance_bindings.json）；'+
      '本图状态版本见上表（同源 data/variants/ 语义图）。</p>';
  }
  document.getElementById('statepanel').innerHTML = html;
}
renderStatePanel();

// ---- DSpark 实例角色/参数绑定面板（B4 补修01：消费
// dsparkView.integration.role_bindings——从实际集成图 + 展开绑定表派生，
// 全部联查真实图节点/张量；节点/张量链接可点开当前图详情）----
(function(){
  const holder = document.createElement('details');
  holder.className = 'card';
  holder.innerHTML = '<summary>DSpark 实例角色与参数绑定（mtp.0/1/2 · 层 43/44/45 · first/last 角色 · 公共模板参数 → 实例路径；当前图）</summary><div id="dsparkrolepanel"></div>';
  document.querySelector('main').insertBefore(holder, document.getElementById('planview'));
})();
function renderDsparkRolePanel() {
  const el = document.getElementById('dsparkrolepanel');
  if (!el) return;
  const rb = P.dsparkView && P.dsparkView.integration &&
             P.dsparkView.integration.role_bindings;
  if (!ctx || !ctx.isDspark || !rb) {
    el.innerHTML = '<p class="sub">实例角色/参数绑定面板属 DSpark 整体图'+
      '（工具条"图"按钮 DS 切换后显示；数据缺失时如实说明，不静态编造）。</p>';
    return;
  }
  const tensorLink = window.showTensor ? (tid =>
    '<a href="#" onclick="showTensor(\''+String(tid).replace(/'/g,'')+
    '\');return false;">'+esc(String(tid).replace(/^tensor:[^:]*:/, ''))+'</a>') : (t=>esc(t));
  const evLink = id => (P.evidence.entries.some(e=>e.evidence_id===id)
    ? evBtn(id) : esc(id));
  let html = '<p class="kv">来源：集成图真实节点/张量 + B2 展开绑定表'+
    '（instance_path 联查 mtp.{i}.{slot}；构建期逐一核验锚点存在）。'+
    esc(rb.derivation_note||'')+'</p>';
  // 逐实例角色
  html += '<h3>三实例角色（DSparkBlock stage_id = layer_id − n_layers）</h3>';
  html += '<table><tr><th>实例</th><th>实际层号</th><th>角色</th><th>角色专有模块</th><th>win_cache 状态（当前图状态迁移）</th></tr>';
  for (const inst of rb.instances) {
    const roles = [
      inst.is_first ? '<span class="chip ok">first（独占入口：main_proj/main_norm/forward_embed）</span>' : '',
      inst.is_last ? '<span class="chip ok">last（独占出口：norm/hc_head/共享 head/Markov/Confidence）</span>' : '',
      (!inst.is_first && !inst.is_last) ? '<span class="chip">middle（仅公共模板 + 共享 embed/head 引用）</span>' : '',
    ].filter(Boolean).join(' ');
    const cs = inst.cache_state || {};
    html += '<tr><td><b>mtp.'+inst.stage_id+'</b></td><td>'+inst.actual_layer_id+
      '</td><td>'+roles+'</td><td class="sub">'+esc((inst.role_modules||[]).join('，'))+
      '</td><td class="sub">'+esc(cs.state_id||'')+'<br>'+esc(cs.before_version||'')+
      ' → '+esc(cs.after_version||'')+'（迁移 '+esc(cs.transition_id||'')+'；'+
      '张量 '+tensorLink(cs.cache_tensor||'')+'）</td></tr>';
  }
  html += '</table>';
  // first/last 真实链说明（联查当前图）
  html += '<p class="kv">';
  for (const inst of rb.instances) {
    if (inst.entry_chain_live) html += '<b>mtp.'+inst.stage_id+' 入口链</b>：'+
      esc(inst.entry_chain_note||'')+'（图节点 entry.main_proj → entry.main_norm'+
      ' → 唯一 main_x）<br>';
    if (inst.exit_chain_live) html += '<b>mtp.'+inst.stage_id+' 出口链</b>：'+
      esc(inst.exit_chain_note||'')+'（图节点 exit.hc_head → exit.norm → '+
      'exit.head → mk.*/cf.*）<br>';
  }
  html += '</p>';
  // 公共模板参数 → 实例路径（联查；param_tensor 展示真实参数张量，
  // inline_op_param 明确标注参数内联在算子内并链接消费节点）
  html += '<h3>公共模板参数 → 实例参数路径（24 项；param_tensor=独立参数张量 / inline_op_param=内联算子参数）</h3>';
  html += '<table><tr><th>模板 spec（param_slot）</th><th>dtype / shape</th><th>绑定口径</th><th>实例路径 → 参数实体/消费节点</th></tr>';
  for (const pb of (rb.template_param_bindings||[])) {
    html += '<tr><td><code>'+esc(pb.spec_id)+'</code><br><span class="sub">'+
      esc(pb.param_slot||'')+'</span></td><td>'+esc(pb.dtype)+
      (pb.shape ? '<br><span class="sub">'+esc(JSON.stringify(pb.shape))+'</span>' : '')+
      '</td><td class="sub">'+esc(pb.binding_summary||'')+'</td><td class="sub">'+
      (pb.instances||[]).map(x =>
        '<code>'+esc(x.instance_path)+'</code> → '+
        (x.graph_tensor
          ? tensorLink(x.graph_tensor)
          : '<span class="chip">内联算子参数</span>')+
        (x.consumer_nodes||[]).map(cn => ' · 消费 <code>'+esc(cn)+'</code>').join('')+
        (x.inline_note ? '<br><span class="sub">'+esc(x.inline_note)+'</span>' : '')
      ).join('<br>')+'</td></tr>';
  }
  html += '</table>';
  // 唯一共享对象
  const sh = rb.shared || {};
  html += '<h3>唯一共享对象（三实例同一，非复制）</h3><ul class="kv">';
  if (sh.embed) html += '<li><b>embed</b>：'+esc(sh.embed.checkpoint_ref)+
    '；权重节点 '+esc(sh.embed.weight_node)+' 张量 '+tensorLink(sh.embed.weight_tensor)+
    '；实例引用 '+esc((sh.embed.instance_refs||[]).join('，'))+'</li>';
  if (sh.head) html += '<li><b>head</b>：'+esc(sh.head.checkpoint_ref)+
    '；权重节点 '+esc(sh.head.weight_node)+' 张量 '+tensorLink(sh.head.weight_tensor)+
    '；实例引用 '+esc((sh.head.instance_refs||[]).join('，'))+'</li>';
  html += '</ul>';
  // first/last 专有参数（runtime/checkpoint dtype 分开）
  const lastParams = rb.last_only_params||[], firstParams = rb.first_only_params||[];
  const paramRows = (rows) => rows.map(p =>
    '<tr><td><code>'+esc(p.param_path)+'</code><br><span class="sub">'+
    esc(p.module||'')+'</span></td><td>'+esc(p.graph_node)+'<br>张量 '+
    tensorLink(p.graph_tensor)+'</td><td>消费 '+esc(p.consumer_node)+
    '</td><td>runtime '+esc(p.dtype)+' / checkpoint '+esc(p.checkpoint_dtype)+
    '<br><span class="sub">'+esc(p.dtype_note||'')+'</span></td></tr>').join('');
  html += '<p class="kv">last/first 专有参数消费节点逐字段联查（存在该张量的输入端口且图中有携带边；hc_head_base/scale 真实消费端口是 gate 的 scale/base，fn 才进 mix）。</p>';
  html += '<h3>first 专有参数（仅 mtp.0）</h3><table><tr><th>参数路径</th><th>图节点/张量</th><th>消费节点</th><th>dtype（runtime/checkpoint）</th></tr>'+
    paramRows(firstParams)+'</table>';
  html += '<h3>last 专有参数（仅 mtp.2；norm/hc_head/Markov/Confidence）</h3>'+
    '<table><tr><th>参数路径</th><th>图节点/张量</th><th>消费节点</th><th>dtype（runtime/checkpoint）</th></tr>'+
    paramRows(lastParams)+'</table>';
  html += '<p class="kv"><b>unknown 边界</b>：'+esc(rb.unknown_note||'')+'。</p>';
  el.innerHTML = html;
}
// 切图随图刷新已内置于 gSetVariant（见上）；此处只做初始渲染（初始
// 上下文为主图 → 归属说明；gInit 之后不再有其它入口）。
renderDsparkRolePanel();

// ---- 层规划 ----
(function(){
  const pv = document.getElementById('planview');
  let html = '<table><tr><th>层</th><th>分组</th><th>模板状态</th><th>状态 ID（逐实例）</th></tr>';
  const groupOf = {};
  for (const g of P.plan.groups) for (const l of g.member_layers) groupOf[l] = g.group_id;
  for (const b of P.plan.bindings) {
    html += '<tr><td>'+b.layer_id+'</td><td>'+esc(groupOf[b.layer_id])+'</td><td>'+
      (b.template_status==='extracted'?'<span class="chip ok">extracted</span>':
       '<span class="chip warn">pending_extraction（未覆盖）</span>')+
      '</td><td class="sub">'+esc(b.state_ids.join('，'))+'</td></tr>';
  }
  html += '</table>';
  html += '<p class="kv"><b>覆盖</b> extracted: '+esc(P.plan.coverage.extracted.join('，'))+
    '<br>未提取: '+esc(P.plan.coverage.pending_extraction.join('，'))+
    '<br>未覆盖明细:<br>· '+P.plan.coverage.uncovered.map(esc).join('<br>· ')+'</p>';
  pv.innerHTML = html;
})();

// ---- 证据与覆盖 ----
(function(){
  const cov = {};
  for (const e of EVID) cov[e.coverage] = (cov[e.coverage]||0)+1;
  let html = '<p class="kv">证据 '+EVID.length+' 条：'+
    Object.entries(cov).map(([k,v]) => k+' '+v).join('，')+
    '。等级含义：located=定位+原文核对；source_checked=逐句语义核对；'+
    'derived=声明式推导；uncovered=诚实未覆盖。</p><details><summary>全部证据列表</summary><table>'+
    '<tr><th>ID</th><th>来源</th><th>行</th><th>等级</th><th>结论</th></tr>';
  for (const e of EVID)
    html += '<tr><td>'+evBtn(e.evidence_id)+'</td><td>'+esc(e.source_key)+'</td><td>'+
      (e.line_start? e.line_start+'-'+e.line_end : '—')+'</td><td>'+evChip(e.coverage)+
      '</td><td class="sub">'+esc((e.claim||e.note||'').slice(0,80))+'</td></tr>';
  html += '</table></details>';
  document.getElementById('evidenceview').innerHTML = html;
})();

document.getElementById('foot').textContent =
  '本页由 structure_reader build 生成（analysis_id ' + P.analysisId.slice(0,16) +
  '…）；图/表/面板同源。静态结构分析：不执行模型代码，cache 实际内容与' +
  '每专家 token 数 unknown；kernel 内部数值未核对。';
</script>
</body>
</html>
"""
