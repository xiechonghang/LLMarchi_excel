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


def render_decoder_page(reading_document: dict) -> bytes:
    """reading_document → 单文件离线 HTML 字节（UTF-8）。"""
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
        parts.append(_render_phase(phase))
    parts.append(_PAGE_FOOT)
    payload = "\n".join(parts).encode("utf-8")
    return payload


# ---------------------------------------------------------------------------
# 阶段渲染
# ---------------------------------------------------------------------------


def _render_phase(phase: dict) -> str:
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

    # ---- 结构图（展示层；边全部来自本文档 operators/tensors 真实引用）----
    # ONNX 式阅读语义：计算框=源码算子名+计算类型；tensor/权重/缓存状态=箭头上
    # 的源码名+shape（端口圆点）；总览为模块导航（虚线框，非计算算子）。
    out.append('<div class="card"><h3>Decoder 结构总览（模块导航：分支/残差/汇合 + 缓存通道）</h3>')
    out.append('<div class="diagram-scroll">' +
               render_svg(build_overview_graph(phase)) + "</div>")
    out.append(legend_html())
    out.append('<p class="dg-note">本图为<b>模块导航</b>（虚线框，非计算算子），'
               "点击跳转到下方「模块详细连接图」查看实际计算框；"
               "聚合边可追溯到原始算子/张量（悬停查看张量清单）。</p></div>")

    cache_graphs = build_cache_graphs(phase)
    if cache_graphs:
        out.append('<div class="card"><h3>缓存读写图（调用前历史 → 本次算子读写 → 调用后回写）</h3>')
        for g in cache_graphs:
            out.append(f'<h4 id="{html.escape(g["anchor"])}">{html.escape(g["title"])}</h4>')
            out.append('<div class="diagram-scroll">' + render_svg(g) + "</div>")
        out.append(legend_html())
        out.append('<p class="dg-note">缓存状态为端口圆点（时序角色与源码名/shape 在箭头上）；'
                   "标注「仅 N_win&gt;0」的条件更新只在闭窗调用发生，无闭窗时状态保留"
                   "（与算子卡片 note 一致）；「下游算子」端口联查消费该读视图的原算子卡片。</p></div>")

    module_graphs = [(g, build_module_graph(phase, g)) for g in module_groups(phase)]
    out.append('<div class="card"><h3>模块详细计算图（计算框=源码算子+类型；'
               "tensor/权重=箭头上的源码名+shape）</h3>")
    for g, graph in module_graphs:
        out.append(f'<h4 id="mod-{html.escape(pid)}-{html.escape(g["id"])}">'
                   f'{html.escape(g["title"])}</h4>')
        out.append('<div class="diagram-scroll">' + render_svg(graph) + "</div>")
    out.append(legend_html())
    out.append('<p class="dg-note">计算框=真实算子（源码调用名+计算类型，复合算子如实列出），'
               "点击联查算子卡片；箭头上的 tensor 标注源码变量/表达式名与符号 shape，"
               "点击联查张量索引；权重为虚线引用。同名源码变量按图内步骤消歧"
               "（HF 源码逐步重赋值，悬停见源码行号）；专家模板仍只有一份（不复制 E 份）。</p></div>")

    # 缓存注记
    cache_notes = phase.get("cache_notes", [])
    if cache_notes:
        out.append('<div class="card"><h3>缓存状态与读写（本阶段）</h3><ul>')
        out.extend(f"<li>{html.escape(str(n))}</li>" for n in cache_notes)
        out.append("</ul></div>")

    # 参数区
    out.append('<div class="card"><h3>参数符号区</h3>')
    out.append(
        "<table class=\"params\"><thead><tr><th>参数id</th><th>符号</th><th>含义</th>"
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
    out.append("</tbody></table></div>")

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
            f'<span class="src-item">{html.escape(str(r.get("path", "")))}'
            f'<span class="mono">:{html.escape(str(r.get("locator", "")))}</span>'
            f' <span class="src-note">{html.escape(str(r.get("note", "")))}</span></span>'
            for r in op.get("source_refs", [])
        ) or "—"
        out.append(
            f'<div class="op {"op-blackbox" if blackbox else ""}" '
            f'id="op-{html.escape(pid)}-{html.escape(str(op.get("id", "")))}">'
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
            f"<div class=\"op-note\">{html.escape(str(op.get('note', '')))}</div>"
            f"</div>"
        )
    out.append("</div></div>")

    # 张量索引
    out.append('<div class="card"><h3>张量索引（联查）</h3>')
    out.append(
        "<table class=\"tensors\"><thead><tr><th>张量id</th><th>名称</th><th>shape</th>"
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
    out.append("</tbody></table></div>")

    # 覆盖注记
    cov = phase.get("coverage_notes", [])
    if cov:
        out.append('<div class="card coverage"><h3>覆盖与未覆盖（本阶段）</h3><ul>')
        out.extend(f"<li>{html.escape(str(n))}</li>" for n in cov)
        out.append("</ul></div>")

    out.append("</section>")
    return "\n".join(out)


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
th {{ background: var(--code); font-weight: 600; }}
.sym {{ font-weight: 700; }}
.src {{ color: var(--muted); font-size: 12px; }}
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
</body>
</html>
"""
