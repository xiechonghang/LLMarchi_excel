"""共享详情渲染单元：参数行、张量行、算子卡（含源码面板/图内定位）。

HF 单层阅读页（``decoder_page._render_phase``）与分层浏览页（``hf_structure_page``）
消费同一实现：页面外壳、阶段/整网专属章节与结构事实各自保留，详情单元不再各写
一份长模板。算子卡的“所涉张量参数引用合并”、源码快照面板、回图定位均由本模块
统一生成；本模块不定义结构事实、不做文件 I/O，全部内容来自传入的文档行。
"""

from __future__ import annotations

import html
import json
import re

from structure_reader.render.display_names import (
    display_shape,
    expand_display_text,
    full_name,
)

__all__ = [
    "KIND_LABELS",
    "render_notes_list",
    "render_operator_card",
    "render_parameter_row",
    "render_tensor_row",
    "source_snapshot_lines",
    "value_repr",
]

KIND_LABELS = {
    "activation": "激活",
    "weight": "权重",
    "cache": "缓存",
    "mask": "掩码",
    "index": "索引",
    "boundary": "边界",
}


def value_repr(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def source_snapshot_lines(
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
    if not locator:
        return "\n".join(f"{k+1:5d} │ {ln}" for k, ln in enumerate(lines[:80]))
    nums = [int(x) for x in re.findall(r"\d+", locator)]
    if not nums:
        return "\n".join(f"{k+1:5d} │ {ln}" for k, ln in enumerate(lines[:80]))
    lo = min(nums)
    hi = max(nums)
    start = max(1, lo - 3)
    end = min(len(lines), hi + 5)
    sel = lines[start - 1:end]
    return "\n".join(f"{start + k:5d} │ {ln}" for k, ln in enumerate(sel))


def render_notes_list(notes: list, css_class: str | None = None) -> str:
    """注记列表；css_class 供页面外壳追加专属样式类（不传时与既有页面一致）。"""
    if not notes:
        return ""
    class_attr = f' class="{css_class}"' if css_class else ""
    return f"<ul{class_attr}>" + "".join(
        f"<li>{html.escape(expand_display_text(str(n)))}</li>" for n in notes) + "</ul>"


def render_parameter_row(param: dict, scope_id: str) -> str:
    """参数符号表一行（锚点 ``p-<scope>-<param_id>``，供算子/张量联查）。"""
    value = param.get("value")
    value_text = "null（数据相关）" if value is None else html.escape(
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    )
    sources = "；".join(
        f'{html.escape(str(r.get("kind", "")))}:{html.escape(str(r.get("path", "")))}'
        f'[{html.escape(str(r.get("locator", "")))}]'
        for r in param.get("source_refs", [])
    ) or "—"
    return (
        f'<tr id="p-{html.escape(scope_id)}-{html.escape(str(param.get("id", "")))}">'
        f'<td class="mono sym" title="{html.escape(str(param.get("id", "")))}">'
        f'{html.escape(full_name(str(param.get("id", ""))))}</td>'
        f"<td>{html.escape(expand_display_text(str(param.get('meaning', ''))))}</td>"
        f'<td class="mono">{value_text}</td>'
        f'<td class="mono">{html.escape(expand_display_text(str(param.get("expression") or "—")))}</td>'
        f'<td class="src">{sources}</td>'
        f"<td>{html.escape(expand_display_text(str(param.get('note', ''))))}</td></tr>"
    )


def render_tensor_row(tensor: dict, scope_id: str, anchor: bool = True) -> str:
    """张量索引一行（锚点 ``t-<scope>-<tensor_id>``）。

    anchor=False：同一结构在多个层级重复列出时只渲染内容、不再产生重复 id
    （参数链接仍指向该 scope 的参数行）。
    """
    prefs = "；".join(
        f'<a class="param-link mono" href="#p-{html.escape(scope_id)}-{html.escape(str(pr))}">'
        f"{html.escape(full_name(str(pr)))}</a>"
        for pr in tensor.get("parameter_refs", [])
    ) or "—"
    tr_id = (f' id="t-{html.escape(scope_id)}-{html.escape(str(tensor.get("id", "")))}"'
             if anchor else "")
    return (
        f"<tr{tr_id}>"
        f'<td class="mono">{html.escape(str(tensor.get("id", "")))}</td>'
        f"<td>{html.escape(str(tensor.get('name', '')))}</td>"
        f'<td class="mono">{display_shape(tensor.get("shape", []))}</td>'
        f"<td>{KIND_LABELS.get(str(tensor.get('kind', '')), str(tensor.get('kind', '')))}</td>"
        f"<td>{prefs}</td>"
        f"<td>{html.escape(expand_display_text(str(tensor.get('note', ''))))}</td></tr>"
    )


def _tensor_cell(tensor: dict | None, tid: str) -> str:
    if tensor is None:
        return f'<span class="mono err">{html.escape(tid)}（未定义！）</span>'
    shape = display_shape(tensor.get("shape", []))
    kind = str(tensor.get("kind", ""))
    return (
        f'<span class="t-name">{html.escape(str(tensor.get("name", tid)))}</span>'
        f'<span class="mono t-shape">{shape}</span>'
        f'<span class="t-kind k-{html.escape(kind)}">'
        f"{KIND_LABELS.get(kind, kind)}</span>"
    )


def _weight_cell(tensor: dict | None, tid: str) -> str:
    if tensor is None:
        return f'<span class="mono err">{html.escape(tid)}（未定义！）</span>'
    shape = display_shape(tensor.get("shape", []))
    return (
        f'<span class="t-name">{html.escape(str(tensor.get("name", tid)))}</span>'
        f'<span class="mono t-shape">{shape}</span>'
    )


def render_operator_card(
    op: dict,
    order: int,
    scope_id: str,
    tensor_index: dict[str, dict],
    param_index: dict[str, dict],
    source_snapshot: dict[str, str] | None = None,
    extra_links: dict[str, str] | None = None,
    locate_html: str | None = None,
) -> str:
    """算子卡：输入/输出/权重、所涉张量参数引用、源码面板与回图定位。

    extra_links: 可选 {op_id: HTML}，在回图链接后追加页面专属动作（如整网页的
    “层顺序与四组”入口）；不传时输出与既有详情页一致。
    locate_html: 可选“回图”链接 HTML（正式阅读页没有旧 SVG，改为指向对应
    Archify 图组）；不传时沿用旧页的 ``locateInDiagram`` 定位链接。
    """
    blackbox = op.get("analysis_status") == "internal_unanalyzed"
    status_html = (
        '<span class="blackbox-flag">◆ 内部未分析</span>' if blackbox
        else '<span class="ok-flag">已分析</span>'
    )
    inputs = "<br>".join(
        _tensor_cell(tensor_index.get(tid), tid) for tid in op.get("inputs", [])) or "—"
    outputs = "<br>".join(
        _tensor_cell(tensor_index.get(tid), tid) for tid in op.get("outputs", [])) or "—"
    weights = "<br>".join(
        _weight_cell(tensor_index.get(tid), tid) for tid in op.get("weights", [])) or "—"
    # 参数引用（含所用张量的引用，联查）
    involved = list(op.get("parameter_refs", []))
    for tid in (*op.get("inputs", []), *op.get("outputs", []), *op.get("weights", [])):
        tensor = tensor_index.get(tid)
        if tensor:
            for pref in tensor.get("parameter_refs", []):
                if pref not in involved:
                    involved.append(pref)
    param_links = "；".join(
        f'<a class="param-link mono" href="#p-{html.escape(scope_id)}-{html.escape(str(pr))}" '
        f'title="{html.escape(str(param_index[pr].get("meaning", "")) if pr in param_index else "")}">'
        f"{html.escape(full_name(str(pr)))}"
        f"{('=(' + html.escape(value_repr(param_index[pr].get('value'))) + ')') if pr in param_index and param_index[pr].get('value') is not None else ''}"
        f"</a>"
        for pr in involved
    ) or "—"
    sources = "<br>".join(
        f'<span class="src-item">'
        f'<button type="button" class="param-link src-btn" '
        f'onclick="toggleSource(\'srcpanel-{html.escape(scope_id)}-{html.escape(str(op.get("id", "")))}-{k}\'); return false;">'
        f'{html.escape(str(r.get("path", "")))}'
        f'<span class="mono">:{html.escape(str(r.get("locator", "")))}</span></button>'
        f'<span class="src-note">{html.escape(expand_display_text(str(r.get("note", ""))))}</span>'
        f'</span>'
        for k, r in enumerate(op.get("source_refs", []))
    ) or "—"
    # 源码面板（离线快照，由 build 传入 source_snapshot）
    src_panels = ""
    for k, r in enumerate(op.get("source_refs", [])):
        panel_id = f'srcpanel-{html.escape(scope_id)}-{html.escape(str(op.get("id", "")))}-{k}'
        src_lines = source_snapshot_lines(
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
    locate = locate_html or (
        f'<a class="param-link" href="#figop-{html.escape(scope_id)}-{oid_esc}" '
        f'onclick="event.preventDefault(); locateInDiagram(\'figop-{html.escape(scope_id)}-{oid_esc}\'); '
        f'return false;">定位到框图 ▸</a>')
    extra = (extra_links or {}).get(str(op.get("id", "")))
    if extra:
        locate += " ｜ " + extra
    return (
        f'<div class="op {"op-blackbox" if blackbox else ""}" '
        f'id="op-{html.escape(scope_id)}-{oid_esc}">'
        f'<div class="op-head"><span class="op-no">{order}</span>'
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
        f"<div class=\"op-note\">{html.escape(expand_display_text(str(op.get('note', ''))))}</div>"
        f"</div>"
    )
