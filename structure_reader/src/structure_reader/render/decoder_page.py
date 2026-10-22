"""HF decoder 阅读页渲染（里程碑一样板：单文件离线 HTML）。

从 build_reading_document 产出的 hf_decoder_reading.v1 文档渲染单文件 HTML：
先完整 Prefill Decoder，再完整 Decode Decoder，连续展示、无标签切换隐藏；
结构同构模板一份，数量通过参数表达，无"重复次数"。

页面结构（每阶段）：
- 阶段标题与范围说明（scope.notes / coverage_notes）。
- 参数符号表（id/符号/含义/值/表达式/来源）。
- 缓存状态注记（cache_notes）。
- 算子流：按阅读顺序逐算子卡片（模块/名称/算子类型/输入输出权重 shape/
  参数引用/源码位置/分析状态/说明），黑盒算子醒目标注"内部未分析"。
- 张量索引表（id/name/shape/kind/note）供联查。

离线约束：不引外部资源（无 CDN/字体/JS 库），样式内嵌；结构由 Python 拼
HTML 字符串（html.escape 转义），不执行任何模型代码。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from structure_reader.render.diagram import (
    build_cache_graphs,
    build_csa_composite_graph,
    build_module_graph,
    build_overview_graph,
    diagram_css,
    legend_html,
    module_groups,
    render_svg,
)

__all__ = ["render_decoder_page", "DecoderPageError"]

_PHASE_LABELS = {"prefill": "Prefill", "decode": "Decode"}
_KIND_LABELS = {
    "activation": "激活",
    "weight": "权重",
    "cache": "缓存",
    "mask": "掩码",
    "index": "索引",
    "boundary": "边界",
}


class DecoderPageError(ValueError):
    """渲染失败（文档缺字段/阶段顺序错误），具名拒绝。"""


def render_decoder_page(reading_document: dict,
                        source_snapshot: dict[str, str] | None = None,
                        onnx_bundle: dict | None = None,
                        candidate_id: str | None = None) -> bytes:
    """reading_document → 单文件离线 HTML 字节（UTF-8）。

    source_snapshot: {相对路径: 完整源码文本}，由 build 层从已分析本地 HF 检出
    读取传入（只读既有 SOURCE_FILES 清单内文件）；渲染层不做文件 I/O。
    None 时源码面板如实显示"快照不可用"。

    onnx_bundle: {phase: {module: {display, details, state, base}}}，由 build 层
    在显式 ONNX 选项中从真实 ModelProto 生成；None 时不显示入口（保持旧能力）。
    candidate_id: 实际 analysis_id，用于生成本地 Netron 链接；None 时不出链接。
    """
    if not isinstance(reading_document, dict):
        raise DecoderPageError("reading_document 必须是 dict")
    if reading_document.get("schema") != "hf_decoder_reading.v1":
        raise DecoderPageError(
            f"schema 必须是 'hf_decoder_reading.v1'，实得 {reading_document.get('schema')!r}"
        )
    model = reading_document.get("model")
    scope = reading_document.get("scope")
    phases = reading_document.get("phases")
    if not isinstance(model, dict) or not isinstance(scope, dict):
        raise DecoderPageError("model/scope 必须是对象")
    if not isinstance(phases, list) or [p.get("id") for p in phases] != ["prefill", "decode"]:
        raise DecoderPageError("phases 必须依次为 prefill、decode 两个阶段")

    parts: list[str] = []
    parts.append(_PAGE_HEAD.format(
        title=html.escape(f"{model.get('display_name', model.get('model_key', ''))} Decoder 阅读样板"),
        model_key=html.escape(str(model.get("model_key", ""))),
        display=html.escape(str(model.get("display_name", ""))),
        impl=html.escape(str(model.get("implementation", ""))),
        revision=html.escape(str(model.get("source_revision", ""))),
        config_path=html.escape(str(model.get("config_path", ""))),
        layer_index=html.escape(str(scope.get("layer_index", ""))),
        attn=html.escape(str(scope.get("attention_type", ""))),
        moe=html.escape(str(scope.get("moe_type", ""))),
        path=html.escape(str(scope.get("analysis_path", ""))),
        scope_notes=_notes_list(scope.get("notes", [])),
        diagram_css=diagram_css(),
    ))
    for phase in phases:
        parts.append(_render_phase(phase, source_snapshot=source_snapshot,
                                   onnx_bundle=onnx_bundle, candidate_id=candidate_id))
    parts.append(_PAGE_FOOT)
    payload = "\n".join(parts).encode("utf-8")
    return payload


# ---------------------------------------------------------------------------
# 阶段渲染
# ---------------------------------------------------------------------------


def _source_snapshot_lines(
    path: str, locator: str,
    snapshot: dict[str, str] | None,
) -> str | None:
    """从快照提取 locator 所指的源码行段；不可用时 None（诚实显示）。"""
    if not snapshot or path not in snapshot:
        return None
    text = snapshot[path]
    lines = text.split("\n")
    # locator 解析：支持 "123"、"123-456"、"123/456"、"123:456"、
    # 多段 "1177-1179/1183-1185"（提取所有数字对的最大范围）
    import re as _re
    if not locator:
        return "\n".join(f"{k+1:5d} │ {ln}" for k, ln in enumerate(lines[:80]))
    nums = [int(x) for x in _re.findall(r"\d+", locator)]
    if not nums:
        return "\n".join(f"{k+1:5d} │ {ln}" for k, ln in enumerate(lines[:80]))
    lo = min(nums)
    hi = max(nums)
    start = max(1, lo - 3)
    end = min(len(lines), hi + 5)
    sel = lines[start - 1:end]
    return "\n".join(f"{start + k:5d} │ {ln}" for k, ln in enumerate(sel))


def _render_phase(phase: dict, source_snapshot: dict[str, str] | None = None,
                  onnx_bundle: dict | None = None,
                  candidate_id: str | None = None) -> str:
    pid = phase.get("id", "")
    label = _PHASE_LABELS.get(pid, pid)
    params = phase.get("parameters", [])
    tensors = phase.get("tensors", [])
    operators = phase.get("operators", [])
    tensor_index = {t["id"]: t for t in tensors}
    param_index = {p["id"]: p for p in params}

    out = [f'<section class="phase" id="phase-{html.escape(pid)}">']
    out.append(
        f'<h2><span class="phase-badge {"badge-prefill" if pid == "prefill" else "badge-decode"}">'
        f'{html.escape(label)}</span> {html.escape(str(phase.get("title", "")))}</h2>'
    )

    # 参数区
    out.append('<div class="card"><h3>参数符号区</h3>')
    out.append(
        "<div class=\"table-scroll\"><table class=\"params\">"
        "<thead><tr><th>参数id</th><th>符号</th><th>含义</th>"
        "<th>值</th><th>表达式</th><th>来源</th><th>说明</th></tr></thead><tbody>"
    )
    for p in params:
        value = p.get("value")
        value_text = "null（数据相关）" if value is None else html.escape(
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
        sources = "；".join(
            f'{html.escape(str(r.get("kind", "")))}:{html.escape(str(r.get("path", "")))}'
            f'[{html.escape(str(r.get("locator", "")))}]'
            for r in p.get("source_refs", [])
        ) or "—"
        # 参数行锚点：算子/张量区的 #p-<phase>-<param_id> 联查目标（缺行即断链）
        out.append(
            f'<tr id="p-{html.escape(pid)}-{html.escape(str(p.get("id", "")))}">'
            f'<td class="mono">{html.escape(str(p.get("id", "")))}</td>'
            f'<td class="mono sym">{html.escape(str(p.get("symbol", "")))}</td>'
            f"<td>{html.escape(str(p.get('meaning', '')))}</td>"
            f'<td class="mono">{value_text}</td>'
            f'<td class="mono">{html.escape(str(p.get("expression") or "—"))}</td>'
            f'<td class="src">{sources}</td>'
            f"<td>{html.escape(str(p.get('note', '')))}</td></tr>"
        )
    out.append("</tbody></table></div></div>")
    # ---- 结构图（展示层；边全部来自本文档 operators/tensors 真实引用）----
    # ONNX 式阅读语义：计算框=源码算子名+计算类型；tensor/权重/缓存状态=箭头上
    # 的源码名+shape（端口圆点）；总览为模块导航（虚线框，非计算算子）。
    out.append('<div class="card"><h3>Decoder 结构总览（并列支路横向展开 + 真实包含边界 + 缓存载体）</h3>')
    out.append('<div class="diagram-scroll">' +
               render_svg(build_overview_graph(phase)) + "</div>")
    out.append(legend_html())
    out.append('<p class="dg-note">本图为<b>模块导航</b>（虚线框，非计算算子）：'
               "层=自上而下的真实主数据流，层内横向并列的是真实并行支路（Q / 共享 KV / "
               "CSA→Indexer），不串成顺序；<b>容器边界</b>表示真实包含关系"
               "（Attention ⊃ CSA Compressor ⊃ Lightning Indexer）；"
               "<b>持久状态存储</b>框是同一 past_key_values.layers[layer_idx] 载体，"
               "读前态/写后态接入对应模块；跨多层的残差/来源类长连接用少量"
               "<b>成对同名接口</b>（▸ 名称 / 名称 ▸，各自写明来源与去向），"
               "避免长线互相压盖。点击模块框跳到下方详图；悬停边可看原始张量清单。</p></div>")



    # 配置选择（COMPRESSOR_CLASSES :745-749; :794-795 实例化）
    out.append('<div class="card"><h3>压缩器配置选择（COMPRESSOR_CLASSES[config.layer_types[layer_idx]]）</h3>')
    out.append('<p>源码 :745-749 定义三种映射；:794-795 <code>self.compressor = COMPRESSOR_CLASSES[self.layer_type](config)</code>。'
               '当前 layer_idx=4 的 <code>config.layer_types[4]="compressed_sparse_attention"</code> → 选中 <b>CSA</b>。</p>')
    out.append('<div class="table-scroll"><table class="params"><thead><tr><th>映射键</th><th>压缩器</th><th>Indexer</th><th>本样板状态</th></tr></thead><tbody>'
               '<tr><td class="mono">compressed_sparse_attention</td><td class="mono">DeepseekV4CSACompressor</td>'
               '<td class="mono">有（self.indexer，:618）</td><td><b>✓ 当前生效（layer 4）</b>，下方 CSA / Indexer 模块图 详细展开</td></tr>'
               '<tr><td class="mono">heavily_compressed_attention</td><td class="mono">DeepseekV4HCACompressor</td>'
               '<td>无</td><td>配置可选，本样板未展开</td></tr>'
               '<tr><td class="mono">sliding_attention</td><td>None（无压缩器）</td>'
               '<td>无</td><td>配置可选（layer 0/1 为 SWA），本样板未展开</td></tr>'
               '</tbody></table></div>'
               '<p class="dg-note">选择边界/控制选择是展示注记，不混作实际 tensor 流。'
               '未展开变体的完整通路后续按需提取，本页不伪装已实现。</p></div>')

    groups = module_groups(phase)
    by_id = {g["id"]: g for g in groups}
    out.append('<div class="card"><h3>模块详细计算图（计算框=源码算子+类型；'
               "tensor/权重=箭头上的源码名+shape；容器=真实包含边界）</h3>")
    if onnx_bundle:
        out.append('<p class="onnx-note onnx-global-note">ONNX 说明：点击“用本地 Netron 打开”会'
                   "唤起本机 Netron 并载入该候选的真实简化 ONNX（本地链接，不经网页）；"
                   "图为 <b>FP32 阅读特化图</b>（无真实权重，不是量化模型的数值等价图）。</p>")
    for g in groups:
        if g["id"] == "attn_indexer":
            # Indexer 不再作为平级章节：它已作为 CSA 的嵌套子容器画在下面。
            continue
        if g["id"] == "attn_csa" and "attn_indexer" in by_id:
            graph = build_csa_composite_graph(phase, g, by_id["attn_indexer"])
            heading = "CSA 压缩器（包含 Lightning Indexer）"
            out.append(f'<span class="anchor-mark" id="mod-{html.escape(pid)}-attn_indexer"></span>')
        else:
            graph = build_module_graph(phase, g)
            heading = g["title"]
        out.append(f'<h4 id="mod-{html.escape(pid)}-{html.escape(g["id"])}">'
                   f'{html.escape(heading)}</h4>')
        left_svg = render_svg(graph)
        out.append('<div class="diagram-scroll">' + left_svg + "</div>")
        if onnx_bundle:
            onnx_ids = ([g["id"], "attn_indexer"] if g["id"] == "attn_csa" else [g["id"]])
            out.append(_render_onnx_links(pid, onnx_ids, onnx_bundle, candidate_id))
    out.append(legend_html())
    out.append('<p class="dg-note">计算框=真实算子（源码调用名+计算类型，复合算子如实列出），'
               "点击联查算子卡片；箭头上的 tensor 标注源码变量/表达式名与符号 shape，"
               "点击联查张量索引；权重为虚线引用。<b>容器边界</b>只表示真实包含关系"
               "（Attention ⊃ CSA Compressor ⊃ Lightning Indexer），不是计算算子；"
               "<b>持久状态存储</b>框的字段左缘是读前/写后/视图时态端口，读/写线端点"
               "落在具体字段上（更新后的值不会冒充调用前输入）；"
               "配置选择是<b>控制线</b>（虚点线），与 tensor/cache 实线区分，"
               "未生效的 HCA/SWA 如实标注未展开、不伪装切换。"
               "同名源码变量按图内步骤消歧（HF 源码逐步重赋值，悬停见源码行号）；"
               "专家模板仍只有一份（不复制 E 份）。</p></div>")

    # 缓存注记
    cache_notes = phase.get("cache_notes", [])
    if cache_notes:
        out.append('<div class="card"><h3>缓存状态与读写（本阶段）</h3><ul>')
        out.extend(f"<li>{html.escape(str(n))}</li>" for n in cache_notes)
        out.append("</ul></div>")



    # 算子流
    out.append('<div class="card"><h3>算子流（按真实阅读顺序）</h3><div class="ops">')
    blackbox_count = 0
    for i, op in enumerate(operators):
        blackbox = op.get("analysis_status") == "internal_unanalyzed"
        if blackbox:
            blackbox_count += 1
        status_html = (
            '<span class="blackbox-flag">◆ 内部未分析</span>' if blackbox
            else '<span class="ok-flag">已分析</span>'
        )

        def tensor_cell(tid: str) -> str:
            t = tensor_index.get(tid)
            if t is None:
                return f'<span class="mono err">{html.escape(tid)}（未定义！）</span>'
            shape = "[" + ", ".join(
                html.escape(str(a)) for a in t.get("shape", [])
            ) + "]"
            return (
                f'<span class="t-name">{html.escape(str(t.get("name", tid)))}</span>'
                f'<span class="mono t-shape">{shape}</span>'
                f'<span class="t-kind k-{html.escape(str(t.get("kind", "")))}">'
                f"{_KIND_LABELS.get(str(t.get('kind', '')), t.get('kind', ''))}</span>"
            )

        def weight_cell(tid: str) -> str:
            t = tensor_index.get(tid)
            if t is None:
                return f'<span class="mono err">{html.escape(tid)}（未定义！）</span>'
            shape = "[" + ", ".join(
                html.escape(str(a)) for a in t.get("shape", [])
            ) + "]"
            return (
                f'<span class="t-name">{html.escape(str(t.get("name", tid)))}</span>'
                f'<span class="mono t-shape">{shape}</span>'
            )

        inputs = "<br>".join(tensor_cell(t) for t in op.get("inputs", [])) or "—"
        outputs = "<br>".join(tensor_cell(t) for t in op.get("outputs", [])) or "—"
        weights = "<br>".join(weight_cell(t) for t in op.get("weights", [])) or "—"
        # 参数引用（含所用张量的引用，联查）
        involved = list(op.get("parameter_refs", []))
        for tid in (*op.get("inputs", []), *op.get("outputs", []), *op.get("weights", [])):
            t = tensor_index.get(tid)
            if t:
                for pref in t.get("parameter_refs", []):
                    if pref not in involved:
                        involved.append(pref)
        param_links = "；".join(
            f'<a class="param-link mono" href="#p-{html.escape(pid)}-{html.escape(pr)}" '
            f'title="{html.escape(str(param_index[pr].get("meaning", "")) if pr in param_index else "")}">'
            f"{html.escape(str(param_index[pr].get('symbol', pr)) if pr in param_index else pr)}"
            f"{('=(' + html.escape(_value_repr(param_index[pr].get('value'))) + ')') if pr in param_index and param_index[pr].get('value') is not None else ''}"
            f"</a>"
            for pr in involved
        ) or "—"
        sources = "<br>".join(
            f'<span class="src-item">'
            f'<button type="button" class="param-link src-btn" '
            f'onclick="toggleSource(\'srcpanel-{html.escape(pid)}-{html.escape(str(op.get("id", "")))}-{k}\'); return false;">'
            f'{html.escape(str(r.get("path", "")))}'
            f'<span class="mono">:{html.escape(str(r.get("locator", "")))}</span></button>'
            f'<span class="src-note">{html.escape(str(r.get("note", "")))}</span>'
            f'</span>'
            for k, r in enumerate(op.get("source_refs", []))
        ) or "—"
        # 源码面板（离线快照，由 build 传入 source_snapshot）
        src_panels = ""
        for k, r in enumerate(op.get("source_refs", [])):
            panel_id = f'srcpanel-{html.escape(pid)}-{html.escape(str(op.get("id", "")))}-{k}'
            src_lines = _source_snapshot_lines(
                r.get("path", ""), str(r.get("locator", "")), source_snapshot)
            if src_lines is not None:
                src_panels += (
                    f'<div id="{panel_id}" class="src-panel" style="display:none">'
                    f'<div class="src-panel-head">{html.escape(str(r.get("path","")))}:{html.escape(str(r.get("locator","")))}</div>'
                    f'<pre class="src-code">{html.escape(src_lines)}</pre></div>')
            else:
                src_panels += (
                    f'<div id="{panel_id}" class="src-panel" style="display:none">'
                    f'<div class="src-panel-head">{html.escape(str(r.get("path","")))}:{html.escape(str(r.get("locator","")))}</div>'
                    f'<pre class="src-code">（源码快照不可用：离线单文件未包含该文件内容）</pre></div>')
        # 回图链接：当前算子在主详图中的 figop id
        oid_esc = html.escape(str(op.get("id", "")))
        locate = (f'<a class="param-link" href="#figop-{html.escape(pid)}-{oid_esc}" '
                  f'onclick="event.preventDefault(); locateInDiagram(\'figop-{html.escape(pid)}-{oid_esc}\'); '
                  f'return false;">定位到框图 ▸</a>')
        out.append(
            f'<div class="op {"op-blackbox" if blackbox else ""}" '
            f'id="op-{html.escape(pid)}-{oid_esc}">'
            f'<div class="op-head"><span class="op-no">{i + 1}</span>'
            f'<span class="mono op-id">{html.escape(str(op.get("id", "")))}</span>'
            f'<span class="op-name">{html.escape(str(op.get("name", "")))}</span>'
            f'<span class="op-type mono">{html.escape(str(op.get("op", "")))}</span>'
            f"{status_html}</div>"
            f'<div class="op-module">模块：{html.escape(str(op.get("module_path", "")))}</div>'
            f'<div class="op-io">'
            f'<div><span class="io-label">输入</span>{inputs}</div>'
            f'<div><span class="io-label">输出</span>{outputs}</div>'
            f'<div><span class="io-label">权重</span>{weights}</div>'
            f"</div>"
            f"<div><span class=\"io-label\">参数</span>{param_links}</div>"
            f"<div><span class=\"io-label\">源码</span>{sources}</div>"
            f"{src_panels}"
            f"<div><span class=\"io-label\">回图</span>{locate}</div>"
            f"<div class=\"op-note\">{html.escape(str(op.get('note', '')))}</div>"
            f"</div>"
        )
    out.append("</div></div>")

    # 张量索引
    out.append('<div class="card"><h3>张量索引（联查）</h3>')
    out.append(
        "<div class=\"table-scroll\"><table class=\"tensors\">"
        "<thead><tr><th>张量id</th><th>名称</th><th>shape</th>"
        "<th>类型</th><th>参数引用</th><th>说明</th></tr></thead><tbody>"
    )
    for t in tensors:
        shape = "[" + ", ".join(html.escape(str(a)) for a in t.get("shape", [])) + "]"
        prefs = "；".join(
            f'<a class="param-link mono" href="#p-{html.escape(pid)}-{html.escape(pr)}">'
            f"{html.escape(str(param_index[pr].get('symbol', pr)) if pr in param_index else pr)}</a>"
            for pr in t.get("parameter_refs", [])
        ) or "—"
        out.append(
            f'<tr id="t-{html.escape(pid)}-{html.escape(str(t.get("id", "")))}">'
            f'<td class="mono">{html.escape(str(t.get("id", "")))}</td>'
            f"<td>{html.escape(str(t.get('name', '')))}</td>"
            f'<td class="mono">{shape}</td>'
            f"<td>{_KIND_LABELS.get(str(t.get('kind', '')), str(t.get('kind', '')))}</td>"
            f"<td>{prefs}</td>"
            f"<td>{html.escape(str(t.get('note', '')))}</td></tr>"
        )
    out.append("</tbody></table></div></div>")

    # 覆盖注记
    cov = phase.get("coverage_notes", [])
    if cov:
        out.append('<div class="card coverage"><h3>覆盖与未覆盖（本阶段）</h3><ul>')
        out.extend(f"<li>{html.escape(str(n))}</li>" for n in cov)
        out.append("</ul></div>")

    out.append("</section>")
    return "\n".join(out)




def _render_onnx_links(phase_id: str, module_ids: list[str], bundle: dict | None,
                       candidate_id: str | None) -> str:
    """每个模块一个本地 llmarchi-netron 链接（唤起本机 Netron）+ 下载/映射链接。"""
    if not bundle or not candidate_id:
        return ""
    mods = bundle.get(phase_id, {})
    parts: list[str] = []
    for mid in module_ids:
        entry = mods.get(mid)
        if not entry:
            parts.append(f'<p class="onnx-note">ONNX 未生成：{html.escape(mid)}'
                         "（原因见 reports/onnx_build.json）</p>")
            continue
        st = entry["state"]
        badge = (" 单专家模板；" if mid == "moe_expert" else
                 " 默认因果/滑窗，padding 未覆盖；" if mid == "attn_mask" else "")
        base = "../" + html.escape(entry.get("base", ""))
        label = "CSA" if mid == "attn_csa" else ("Indexer" if mid == "attn_indexer" else mid)
        href = (f"llmarchi-netron://open?candidate={html.escape(candidate_id)}"
                f"&phase={html.escape(phase_id)}&module={html.escape(mid)}&variant=sim")
        parts.append(
            f'<p class="onnx-open">ONNX（{html.escape(label)}）：'
            f'<a class="onnx-open-link" data-module="{html.escape(phase_id)}/{html.escape(mid)}" '
            f'href="{href}">用本地 Netron 打开 '
            f'<span class="mono">{html.escape(phase_id)}_{html.escape(mid)}.sim.onnx</span></a>'
            f'<span class="onnx-note">（需已注册 llmarchi-netron 链接；'
            f'{st.get("node_count_original")}→{st.get("node_count_simplified")} 节点；'
            f'FP32 阅读特化、无真实权重，非量化模型数值等价；{badge}）</span>'
            f'<span class="onnx-note"> 下载：<a href="{base}/model.onnx">原始 onnx</a> · '
            f'<a href="{base}/model.sim.onnx">简化 onnx</a> · '
            f'<a href="{base}/mapping.json">映射</a></span></p>')
    return "".join(parts)


def _value_repr(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _notes_list(notes: list) -> str:
    if not notes:
        return ""
    return "<ul>" + "".join(f"<li>{html.escape(str(n))}</li>" for n in notes) + "</ul>"


# ---------------------------------------------------------------------------
# 页面骨架（内嵌样式，无外部资源）
# ---------------------------------------------------------------------------

_PAGE_HEAD = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #f6f7f9; --card: #ffffff; --ink: #1f2933; --muted: #5f6b7a;
  --line: #d9e0e7; --accent: #1f4e79; --prefill: #215c36; --decode: #7a3b0e;
  --code: #eef1f4; --blackbox-bg: #ffe9e9; --blackbox-ink: #9c0006;
  --ok-ink: #1a6b3c;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.65 -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  padding: 0 16px 64px;
}}
header.page {{
  max-width: 1180px; margin: 0 auto; padding: 28px 0 10px;
}}
h1 {{ font-size: 24px; margin: 0 0 6px; }}
.meta {{ color: var(--muted); font-size: 13px; }}
.meta b {{ color: var(--ink); }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }}
.card {{
  max-width: 1180px; margin: 18px auto; background: var(--card);
  border: 1px solid var(--line); border-radius: 10px; padding: 16px 20px;
}}
h2 {{ font-size: 20px; margin: 34px auto 4px; max-width: 1180px; }}
h3 {{ font-size: 16px; margin: 4px 0 12px; }}
.phase-badge {{
  display: inline-block; color: #fff; border-radius: 6px;
  padding: 2px 10px; font-size: 15px; vertical-align: 2px;
}}
.badge-prefill {{ background: var(--prefill); }}
.badge-decode {{ background: var(--decode); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ border: 1px solid var(--line); padding: 6px 9px; text-align: left; vertical-align: top; }}
/* 反馈10：表格受 card 内容宽约束；长路径/表达式自然折行；必要时表格容器内水平滚动 */
.table-scroll {{ width: 100%; max-width: 100%; overflow-x: auto; overflow-y: visible;
  box-sizing: border-box; }}
.card table {{ width: 100%; max-width: 100%; table-layout: auto; box-sizing: border-box; }}
.card th, .card td {{ overflow-wrap: anywhere; word-break: break-word; white-space: normal; }}
.card td.mono, .card td.src {{ word-break: break-all; }}
th {{ background: var(--code); font-weight: 600; }}
.sym {{ font-weight: 700; }}
.src {{ color: var(--muted); font-size: 12px; }}
.diagram-scroll {{ overflow: auto; }}
.onnx-open {{ font-size: 13px; margin: 6px 0 14px; }}
.onnx-open-link {{ font-weight: 600; }}
.onnx-note {{ font-size: 12px; color: var(--muted); margin: 4px 0; }}
.onnx-details {{ font-size: 12px; margin: 4px 0 2px; }}
.onnx-badge {{ display: inline-block; font-size: 11px; font-weight: 600; color: #7a3b0e;
  background: #fdf1e3; border: 1px solid #e8c9a6; border-radius: 8px; padding: 0 6px; }}
.onnx-global-note {{ background: #eef4fa; border: 1px solid #c9d9ea; border-radius: 8px;
  padding: 6px 10px; }}
.onnx-details table {{ border-collapse: collapse; width: 100%; }}
.onnx-details th, .onnx-details td {{ border: 1px solid var(--line); padding: 2px 6px;
  vertical-align: top; }}
.ops {{ display: flex; flex-direction: column; gap: 10px; }}
.op {{
  border: 1px solid var(--line); border-left: 4px solid var(--accent);
  border-radius: 8px; padding: 10px 14px; background: #fcfdfe;
}}
.op-blackbox {{ border-left-color: var(--blackbox-ink); background: var(--blackbox-bg); }}
.op-head {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
.op-no {{
  background: var(--accent); color: #fff; border-radius: 50%;
  min-width: 22px; height: 22px; display: inline-flex; align-items: center;
  justify-content: center; font-size: 12px;
}}
.op-id {{ color: var(--muted); }}
.op-name {{ font-weight: 700; font-size: 15px; }}
.op-type {{
  background: var(--code); border-radius: 4px; padding: 1px 7px; color: var(--muted);
}}
.blackbox-flag {{
  color: var(--blackbox-ink); font-weight: 800; margin-left: auto;
  border: 1.5px solid var(--blackbox-ink); border-radius: 6px; padding: 1px 8px;
}}
.ok-flag {{ color: var(--ok-ink); margin-left: auto; font-size: 12px; }}
.op-module {{ color: var(--muted); font-size: 12.5px; margin-top: 2px; }}
.op-io {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px; margin-top: 8px; }}
.io-label {{
  display: inline-block; min-width: 40px; color: var(--muted); font-size: 12px;
  font-weight: 600; margin-right: 6px;
}}
.t-name {{ font-weight: 600; margin-right: 4px; }}
.t-shape {{ background: var(--code); border-radius: 4px; padding: 0 6px; margin-right: 6px; }}
.t-kind {{
  font-size: 11px; border-radius: 4px; padding: 0 5px;
  border: 1px solid var(--line); color: var(--muted);
}}
.k-weight {{ border-color: #b08d57; color: #7a5c1e; }}
.k-cache {{ border-color: #4a7bb5; color: #2b5d8f; }}
.k-mask {{ border-color: #9c6bb5; color: #6d3f8f; }}
.k-index {{ border-color: #b54a6b; color: #8f2b4d; }}
.k-boundary {{ border-color: #4ab58d; color: #1e6d4f; }}
.param-link {{ color: var(--accent); text-decoration: none; border-bottom: 1px dotted; margin-right: 4px; }}
.param-link:hover {{ text-decoration: underline; }}
.src-item {{ display: block; font-size: 12px; color: var(--muted); }}
.src-item .mono {{ color: var(--ink); }}
.src-btn {{ font: inherit; cursor: pointer; background: none; padding: 0; border: 0; }}
.anchor-mark {{ display: block; height: 0; overflow: hidden; }}
.src-note {{ color: var(--muted); }}
.op-note {{ margin-top: 6px; font-size: 13.5px; color: #33404d; }}
.err {{ color: var(--blackbox-ink); font-weight: 700; }}
.coverage {{ border-left: 4px solid #b08d57; }}
.coverage h3 {{ color: #7a5c1e; }}
ul {{ margin: 4px 0; padding-left: 22px; }}
li {{ margin: 3px 0; }}
h4 {{ font-size: 14px; margin: 16px 0 6px; color: var(--ink); }}
{diagram_css}
footer.page {{ max-width: 1180px; margin: 26px auto 0; color: var(--muted); font-size: 12.5px; }}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14171b; --card: #1d2126; --ink: #e4e9ee; --muted: #9aa7b4;
    --line: #333b44; --accent: #6ea8dc; --code: #262c33;
    --blackbox-bg: #3a2024; --blackbox-ink: #ff9a9a;
  }}
  .op {{ background: #20252b; }}
  th {{ background: #262c33; }}
}}
</style>
</head>
<body>
<header class="page">
<h1>{display} — HF Decoder 阅读样板（layer_idx={layer_index}）</h1>
<div class="meta">
模型键：<b>{model_key}</b> ｜ 实现：<b>{impl}</b>（eager）｜ 源码版本：
<b class="mono">{revision}</b> ｜ 配置：<b class="mono">{config_path}</b><br>
本层结构：<b>{attn} + {moe}</b>（分析路径 <b>{path}</b>）；
先完整阅读 Prefill，再完整阅读 Decode，两阶段连续展示（无标签切换）。
</div>
{scope_notes}
</header>
"""

_PAGE_FOOT = """<footer class="page">
由 structure_reader HF 主干从本地 Transformers 检出与根 config.json 静态生成；
不加载权重、不执行模型；静态结构一致为目标。黑盒算子以「内部未分析」醒目标注。
</footer>

<script>
(function() {
  "use strict";
  // 7) 定位到框图：滚动+高亮
  window.locateInDiagram = function(figId) {
    var els = document.querySelectorAll('[id$="-' + figId + '"], [id="' + figId + '"]');
    var el = els.length ? els[els.length - 1] : null;
    if (!el) return;
    el.scrollIntoView({behavior: "smooth", block: "center", inline: "center"});
    el.style.filter = "drop-shadow(0 0 6px #e8a740)";
    setTimeout(function() { el.style.filter = ""; }, 2000);
  };
  // 3) 源码面板开关
  window.toggleSource = function(id) {
    var el = document.getElementById(id);
    if (el) el.style.display = (el.style.display === "none") ? "block" : "none";
  };
})();
</script>
</body>
</html>
"""
