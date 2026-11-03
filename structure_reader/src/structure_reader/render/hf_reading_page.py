"""DeepSeek 正式阅读入口页（D01：四层导航 + 仅 Prefill/Decode + 只显示 Archify 图）。

与既有 ``hf_structure_page.render_structure_page`` 分开：
- 旧页并列渲染“旧 SVG 图”和“Archify 子图卡”，本页只内嵌 Archify 原图（iframe，
  ``?embed=1`` 原生 embed 模式，不改生成后的 HTML 字节）；
- 旧页提供 5 类数值场景，本页只有 Prefill / Decode 两个阶段，图中一律保留
  seqlen / kvcachelen 符号，不代入示例长度；
- 层级路由沿用 ``#/network`` → ``#/group/<gid>`` → ``#/group/<gid>/block`` →
  ``#/group/<gid>/ops[/<module>]``，因此 Archify 子图包装页的返回链接仍然有效。

页面只消费构建层传入的同候选事实（network / layer_structures / archify 清单），
不做文件 I/O、不读权重、不实例化模型；离线单文件 + 同目录 Archify HTML 即可打开。
"""

from __future__ import annotations

import html
import json

from structure_reader.render.detail_render import (
    render_notes_list,
    render_operator_card,
)
from structure_reader.render.hf_structure_page import (
    _PAGE_HEAD,
    _page_css,
    _sources_table,
)
from structure_reader.render.hf_model_profile import DEEPSEEK_PAGE_PROFILE, PageProfile

__all__ = ["render_reading_page", "ReadingPageError"]


class ReadingPageError(ValueError):
    """正式阅读页装配失败（文档/清单不一致），具名拒绝。"""


# 只交付两个阶段；图形本身是符号结构，阶段只改变“本阶段约束”说明。
_STAGES = (
    {
        "id": "prefill",
        "label": "Prefill",
        "kvcachelen": "0（空历史约束）",
        "note": (
            "seqlen 是本次新输入的 token 数；Prefill 取空历史约束 kvcachelen = 0。"
            "图与参数保留符号，不代入示例长度。"
        ),
    },
    {
        "id": "decode",
        "label": "Decode",
        "kvcachelen": "kvcachelen（调用前逻辑历史长度，符号）",
        "note": (
            "seqlen 是本次新输入的 token 数；kvcachelen 是调用前已处理的逻辑历史 token 数。"
            "图中两者都保留为符号，不强制 seqlen = 1。"
        ),
    },
)

# 图前精简参数：整网与端点各自只看相关项（完整参数表仍在各层 details）。
_NETWORK_PARAM_IDS = (
    "B", "seqlen", "P", "H", "Vocab", "Nlay", "Nhc", "Lk", "Ssel", "W",
)
_EMBED_PARAM_IDS = ("B", "seqlen", "H", "Vocab", "Nhc")
_HEAD_PARAM_IDS = ("B", "seqlen", "Nhc", "H", "Vocab", "Lk", "Ssel", "EpsN", "Eps")

# 每层图注：本票交付 / 仍未交付的改进（阶段性复用旧子图时明确声明）。
_LAYER_NOTES = {
    "group": (
        "本票（D04）在 D03 基础上交付 MoE 模块关系、Hash/TopK 路由、"
        "单激活专家、共享专家与 routed+shared 汇合；D05 仍未交付。"
    ),
    "block": (
        "模块关系层的 Attention 总览保持 Q/KV 同图、不造顺序依赖；"
        "MoE 关系只到 gate/激活专家/共享专家/加和，内部在算子层；D05 尚未交付。"
    ),
    "ops": (
        "内部算子层包含 Attention 总览/Q·KV·O、KV 新投影、缓存状态、压缩器与"
        "Lightning Indexer；本轮另有 Hash/TopK、单激活专家、共享专家与汇合；"
        "D05 未交付内容不伪造入口。"
    ),
}


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _rel_link(path: str) -> str:
    """候选内路径（viewer/ 或 data/ 相对候选根）→ 相对 viewer/structure.html。"""
    text = str(path or "")
    if not text:
        return ""
    if text.startswith("viewer/"):
        return html.escape(text[len("viewer/"):])
    return html.escape("../" + text)


def _stage_bar() -> str:
    buttons = "".join(
        f'<button type="button" data-stage-btn="{stage["id"]}"'
        + (' class="on" aria-pressed="true"' if index == 0 else ' aria-pressed="false"')
        + f">{_esc(stage['label'])}</button>"
        for index, stage in enumerate(_STAGES)
    )
    panels = "".join(
        f'<div class="rd-stage" data-stage="{stage["id"]}"'
        + ("" if index == 0 else " hidden")
        + f'><p>{_esc(stage["note"])}</p>'
        f'<p class="mono">kvcachelen：{_esc(stage["kvcachelen"])}</p></div>'
        for index, stage in enumerate(_STAGES)
    )
    return (
        '<div class="rd-stagebar" role="group" aria-label="分析阶段">'
        '<span class="rd-stagebar-title">分析阶段：</span>'
        f'<span class="rd-stage-buttons">{buttons}</span>'
        '<span class="rd-symbol-note">只提供 Prefill / Decode；'
        'seqlen、kvcachelen 在图中保持符号，不代入示例长度。</span>'
        f"</div>{panels}"
    )


def _nav(groups: list[dict]) -> str:
    first = str(groups[0]["id"]) if groups else ""
    items = (
        ("network", "01 整网", "#/network"),
        ("group", "02 Decoder", f"#/group/{first}"),
        ("block", "03 模块关系", f"#/group/{first}/block"),
        ("ops", "04 内部算子", f"#/group/{first}/ops"),
    )
    return (
        '<nav class="rd-nav" id="rd-nav" aria-label="阅读层次">'
        + "".join(
            f'<a class="rd-nav-item" data-nav="{key}" href="{_esc(href)}">{_esc(label)}</a>'
            for key, label, href in items
        )
        + "</nav>"
    )


def _crumbs(items: list[tuple[str, str | None]]) -> str:
    parts = []
    for index, (label, href) in enumerate(items):
        last = index == len(items) - 1
        if href and not last:
            parts.append(f'<a href="{_esc(href)}">{_esc(label)}</a>')
        else:
            parts.append(f'<span aria-current="page">{_esc(label)}</span>')
    return '<nav class="crumbs" aria-label="当前位置">' + " › ".join(parts) + "</nav>"


# ---------------------------------------------------------------------------
# 图块：原生 Archify HTML 内嵌 + 单独打开 + 完整文字/关系
# ---------------------------------------------------------------------------


def _canvas_of(entry: dict) -> tuple[float, float]:
    canvas = entry.get("canvas") or {}
    try:
        width = float(canvas.get("width") or 0)
        height = float(canvas.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0.0
    return (width or 900.0, height or 700.0)


def _frame_size(entry: dict, *, cap: int = 1460, floor: int = 1200) -> tuple[int, int]:
    """内嵌高度：按候选画布与显示宽度等比计算，宁留白不裁切。

    画布宽度较小时放大到 1200 的阅读宽度，较大时缩到页面
    最大内容宽度；高度同步换算，再加原生 embed 模式的容器内边距。这样窄而高的
    整网/层组图在章节内嵌时不会被压到 720 宽而让 8px 连线字再次变小。
    """
    width, height = _canvas_of(entry)
    display = min(cap, max(floor, width))
    scaled = height * (display / width) if width else height
    return (int(round(display)), int(round(scaled + 112)))


def _fig_block(entry: dict | None, *, figure_id: str, caption: str,
               scope: str, note: str = "") -> str:
    """一张 Archify 图：题注 + 单独打开/数据入口 + 原生图 iframe + 完整文字折叠。"""
    if not entry:
        return (
            f'<div class="rd-missing" id="{_esc(figure_id)}">'
            f"<b>{_esc(caption)}</b>：本候选未生成该图（不提供空白占位或假链接）。</div>"
        )
    slug = _esc(entry.get("slug"))
    view = _rel_link(entry.get("view") or entry.get("html"))
    native = _rel_link(entry.get("html"))
    candidate = _rel_link(entry.get("candidate"))
    relations = _rel_link(entry.get("relations"))
    width, height = _frame_size(entry)
    nodes = entry.get("nodes", 0)
    rel_count = entry.get("relations_count", entry.get("connections", 0))
    links = [
        f'<a href="{view}" target="_blank" rel="noopener">子图页（带返回）↗</a>',
        f'<a href="{native}" target="_blank" rel="noopener">原生图新标签 ↗</a>',
    ]
    if candidate:
        links.append(f'<a href="{candidate}">图数据 candidate JSON</a>')
    if relations:
        links.append(f'<a href="{relations}">完整关系 JSON（{rel_count} 条）</a>')
    if entry.get("sources"):
        links.append(
            f'<a href="{_rel_link(entry.get("sources"))}">来源映射 JSON</a>')
    note_html = f'<p class="rd-fig-note">{_esc(note)}</p>' if note else ""
    local_params = entry.get("local_parameters") or []
    local_html = ""
    if local_params:
        local_title = entry.get("local_parameters_title") or "mHC 局部参数（全称/来源）"
        local_note = entry.get("local_parameters_note") or (
            "config：hc_mult、hc_sinkhorn_iters、hc_eps；"
            "modeling_deepseek_v4.py:926-938、976-979。")
        local_html = (
            f'<div class="rd-local-params"><b>{_esc(local_title)}</b>'
            f'<p class="rd-fig-note">{_esc(local_note)}</p>'
            "<ul>" + "".join(
                f"<li>{_esc(str(item))}</li>" for item in local_params
            ) + "</ul></div>"
        )
    return (
        f'<figure class="rd-fig" id="{_esc(figure_id)}">'
        '<figcaption class="rd-fig-head">'
        f'<span class="rd-fig-title">{_esc(caption)}</span>'
        f'<span class="rd-fig-meta mono">{nodes} 节点 · {rel_count} 条关系 · '
        f'阶段 <span data-stage-badge>Prefill</span></span>'
        "</figcaption>"
        f'<p class="rd-fig-links">{" ｜ ".join(links)}</p>'
        + note_html
        + f'<iframe class="rd-frame" src="{native}?embed=1&amp;theme=light" '
          f'title="{_esc(caption)}" loading="lazy" '
          f'style="width:{width}px;max-width:100%;height:{height}px"></iframe>'
        + local_html
        + _fulltext(entry)
        + "</figure>"
    )


def _fulltext(entry: dict) -> str:
    """完整文字：节点全名/操作与逐条数据流（不受主图排版宽度影响）。"""
    nodes = entry.get("node_details") or []
    relations = entry.get("relations_all") or []
    node_rows = []
    for node in nodes:
        detail = "；".join(str(item) for item in node.get("details") or () if item)
        source_bits = []
        for source in node.get("sources") or ():
            path = str(source.get("path") or "")
            line = source.get("line")
            end = source.get("end_line")
            if not path:
                continue
            location = f"{path}:{line}" if end in (None, line) else f"{path}:{line}-{end}"
            source_bits.append(location)
        node_rows.append(
            '<p class="rd-full-node">'
            f'<b>{_esc(node.get("label"))}</b><br>'
            f'<code>{_esc(node.get("sublabel"))}</code>'
            + (f'<br><span class="rd-full-detail">{_esc(detail)}</span>' if detail else "")
            + (f'<br><span class="rd-full-tag">{_esc(node.get("tag"))}</span>'
               if node.get("tag") else "")
            + (f'<br><span class="rd-full-src">来源：{_esc("；".join(source_bits))}</span>'
               if source_bits else "")
            + "</p>"
        )
    rel_rows = []
    for record in relations:
        rel_rows.append(
            "<tr>"
            f'<td class="mono">{_esc(record.get("from"))} → {_esc(record.get("to"))}</td>'
            f'<td>{_esc(record.get("tensor"))}</td>'
            f'<td class="mono">{_esc(record.get("shape"))}</td>'
            f'<td>{_esc(record.get("kind"))}'
            + (f' · {_esc(record.get("label"))}' if record.get("label") else "")
            + "</td></tr>"
        )
    out = [
        f'<details class="fold rd-full"><summary>节点完整名称与操作 · '
        f"{len(nodes)} 项（不截断，不靠 tooltip）</summary>"
        + "".join(node_rows)
        + "</details>",
        f'<details class="fold rd-full"><summary>数据流来源与目标 · '
        f"{len(relations)} 条</summary>"
        '<div class="table-scroll"><table class="tensors"><thead><tr>'
        "<th>数据流</th><th>张量</th><th>全称维度</th><th>类型/说明</th>"
        "</tr></thead><tbody>" + "".join(rel_rows) + "</tbody></table></div></details>",
    ]
    return "".join(out)


# ---------------------------------------------------------------------------
# 就近参数
# ---------------------------------------------------------------------------


def _param_cells(params: list[dict], scope: str, ids) -> str:
    wanted = [str(item) for item in ids]
    index = {str(param.get("id")): param for param in params}
    cells = []
    for param_id in wanted:
        param = index.get(param_id)
        if param is None:
            continue
        value = param.get("value")
        value_text = ("null（数据相关）" if value is None else
                      str(value) if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        expression = str(param.get("expression") or "—")
        sources = "；".join(
            f'{r.get("kind", "")}:{r.get("path", "")}[{r.get("locator", "")}]'
            for r in param.get("source_refs", [])
        ) or "—"
        stage_value = ""
        if param_id == "P":
            stage_value = (
                '<div class="rd-param-stage">'
                '<span data-stage-value="prefill">Prefill：0（空历史约束）</span>'
                '<span data-stage-value="decode">Decode：kvcachelen（符号）</span>'
                "</div>"
            )
        cells.append(
            f'<div class="rd-param" data-param="{_esc(param_id)}">'
            f'<div class="rd-param-sym mono">{_esc(param_id)}</div>'
            f'<div class="rd-param-meaning">{_esc(param.get("meaning"))}</div>'
            f'<code class="rd-param-expr">{_esc(expression)}</code>'
            f'<div class="rd-param-value mono">{_esc(value_text)}</div>'
            f"{stage_value}"
            f'<div class="rd-param-src src">{_esc(sources)}</div>'
            "</div>"
        )
    return '<div class="rd-params">' + "".join(cells) + "</div>"


def _param_table(params: list[dict], scope: str, ids=None) -> str:
    """完整参数表（details 内）：全称、含义、值、表达式、来源。"""
    wanted = {str(item) for item in ids} if ids is not None else None
    rows = []
    for param in params:
        param_id = str(param.get("id"))
        if wanted is not None and param_id not in wanted:
            continue
        value = param.get("value")
        value_text = ("null（数据相关）" if value is None else
                      str(value) if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        sources = "；".join(
            f'{r.get("kind", "")}:{r.get("path", "")}[{r.get("locator", "")}]'
            for r in param.get("source_refs", [])
        ) or "—"
        rows.append(
            f'<tr id="p-{_esc(scope)}-{_esc(param_id)}">'
            f'<td class="mono">{_esc(param_id)}</td>'
            f"<td>{_esc(param.get('meaning'))}</td>"
            f'<td class="mono">{_esc(value_text)}</td>'
            f'<td class="mono">{_esc(param.get("expression") or "—")}</td>'
            f'<td class="src">{_esc(sources)}</td></tr>'
        )
    if not rows:
        return ""
    return (
        '<details class="fold"><summary>全部参数符号（含完整名称、表达式与配置来源）</summary>'
        '<div class="table-scroll"><table class="params"><thead><tr>'
        "<th>参数</th><th>含义</th><th>值</th><th>表达式</th><th>来源</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>"
    )


# ---------------------------------------------------------------------------
# 各层 section
# ---------------------------------------------------------------------------


def _network_section(network: dict, archify: dict, source_snapshot, input_shas) -> str:
    path = network.get("path") or {}
    scope = network.get("scope") or {}
    model = network.get("model") or {}
    network_entry = archify.get("network")
    endpoints = archify.get("endpoints") or {}
    out = ['<section class="lv rd-lv" id="lv-network" data-lv="network">']
    out.append(_crumbs([("整网", None)]))
    out.append(
        f'<h2 class="lv-title">整网 · <span class="mono">'
        f'{_esc(scope.get("entry_class") or "DeepseekV4ForCausalLM")}</span>'
        "（token → logits）</h2>"
    )
    out.append(
        '<p class="rd-lead">主路径来自源码/config 同源事实：'
        f'Embedding → mHC 流扩展 → <b>{_esc(scope.get("layer_count", ""))} 层 Decoder 循环</b>'
        "（真实层序见下方层条）→ HyperHead 收拢 → Final RMSNorm → logits 位置切片 → lm_head。"
        "循环次数来自 <code>config.num_hidden_layers</code>，不是模板常量；"
        "同组共用一份 Block 模板不表示共享权重。</p>"
    )
    out.append(_param_cells(path.get("parameters", []), "network", _NETWORK_PARAM_IDS))
    out.append(_fig_block(
        network_entry, figure_id="fig-network", caption="DeepseekV4ForCausalLM · 整网（模块级）",
        scope="network",
        note="标记为模块级的节点在下方 Decoder 层展开；简单的输入/输出端点在本页直接给出逐算子图。",
    ))
    out.append(_param_table(path.get("parameters", []), "network"))
    out.append(_endpoints_block(endpoints, path.get("parameters", [])))

    out.append('<h3>真实层顺序（每格＝该层所属层组，点入 Decoder 层）</h3>')
    layer_order = network.get("layer_order", [])
    layers = []
    for entry in layer_order:
        gid = str(entry.get("group_id", ""))
        short = {
            "sliding_attention": "SWA",
            "compressed_sparse_attention": "CSA",
            "heavily_compressed_attention": "HCA",
        }.get(str(entry.get("attention_type", "")), str(entry.get("attention_type", "")))
        moe = {"hash_moe": "Hash", "moe": "TopK"}.get(
            str(entry.get("moe_type", "")), str(entry.get("moe_type", "")))
        layers.append(
            f'<li class="np-layer np-{_esc(gid)}">'
            f'<a href="#/group/{_esc(gid)}" title="层 {entry.get("index")}：'
            f'{_esc(gid)}（{_esc(entry.get("attention_type", ""))} + '
            f'{_esc(entry.get("moe_type", ""))}）— 进入该层所属 Decoder">'
            f'<span class="np-layer-idx">{entry.get("index")}</span>'
            f'<span class="np-layer-type">{_esc(short)} · {_esc(moe)}</span></a></li>'
        )
    out.append('<ol class="np-layers">' + "".join(layers) + "</ol>")

    out.append('<h3>四类真实层组（Decoder 上下文）</h3>')
    rows = []
    for group in network.get("groups", []):
        gid = str(group["id"])
        indices = ", ".join(str(i) for i in group.get("layer_indices", []))
        rows.append(
            '<tr class="group-row">'
            f'<td><a class="strong-link" href="#/group/{_esc(gid)}">'
            f'{_esc(group.get("label", gid))}</a><br>'
            f'<span class="mono">{_esc(gid)}</span></td>'
            f'<td class="mono">{_esc(group.get("attention_type", ""))} + '
            f'{_esc(group.get("moe_type", ""))}</td>'
            f'<td>{group.get("count", len(group.get("layer_indices", [])))}</td>'
            f'<td class="mono rd-wrap">{_esc(indices)}</td>'
            f'<td>{group.get("representative_layer_index")}</td>'
            f'<td><a class="param-link" href="#/group/{_esc(gid)}">进入 Decoder →</a></td></tr>'
        )
    out.append(
        '<div class="table-scroll"><table class="params"><thead><tr>'
        "<th>层组</th><th>层型 × MLP</th><th>层数</th><th>真实层索引</th>"
        "<th>代表层</th><th>进入</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )

    out.append('<details class="fold"><summary>整网范围、缓存前提与未覆盖项</summary>')
    out.append("<h4>缓存前提</h4>")
    out.append(render_notes_list(path.get("cache_notes", [])))
    out.append("<h4>覆盖与未覆盖</h4>")
    out.append(render_notes_list(path.get("coverage_notes", [])))
    out.append("</details>")
    out.append(
        '<details class="fold"><summary>来源身份核对（不下载、不实例化、不读权重）</summary>'
        + _sources_table(input_shas)
        + '<p class="np-note">本候选只静态读取上述官方源码与根配置；'
        f"实现 <code>{_esc(model.get('implementation', ''))}</code>，"
        f"源码 revision <code>{_esc(model.get('source_revision', ''))}</code>。"
        "工具与交付哈希见同候选 <code>reports/source_manifest.json</code>、"
        "<code>run_manifest.json</code>。</p></details>"
    )
    out.append("</section>")
    return "".join(out)


def _endpoints_block(endpoints: dict, params: list[dict]) -> str:
    embed = endpoints.get("embedding")
    head = endpoints.get("head")
    panels = []
    if embed:
        panels.append(
            '<div class="rd-tabbody" data-tab="embedding">'
            + _param_cells(params, "embedding", _EMBED_PARAM_IDS)
            + _fig_block(
                embed, figure_id="fig-embedding",
                caption="Embedding · input_ids → embed_tokens → mHC 多流（逐算子）",
                scope="embedding",
                note="查表/Gather 与 unsqueeze/expand/contiguous 逐算子展开；"
                     "权重形状就近列出，seqlen 保持符号。",
            )
            + "</div>"
        )
    if head:
        panels.append(
            '<div class="rd-tabbody" data-tab="head" hidden>'
            + _param_cells(params, "head", _HEAD_PARAM_IDS)
            + _fig_block(
                head, figure_id="fig-head",
                caption="HyperHead → Final Norm → logits 切片 → lm_head（逐算子）",
                scope="head",
                note="HyperHead 不冒充单算子：流展平+归一、门控投影、sigmoid 门控、加权求和"
                     "分别具名；logits 位置由 logits_to_keep（Lk）决定，输出序列轴为 Ssel。",
            )
            + "</div>"
        )
    if not panels:
        return ""
    tabs = []
    if embed:
        tabs.append('<button type="button" class="on" data-tab-btn="embedding">'
                    "Embedding / Gather 与多流扩展</button>")
    if head:
        tabs.append('<button type="button" data-tab-btn="head">'
                    "输出端：HyperHead → Norm → lm_head</button>")
    return (
        '<h3>整网层简单端点（算子级）</h3>'
        '<p class="rd-lead">两个端点在整网层就能读到具名算子、符号 shape 与就近参数；'
        "复合模块（HyperHead）已拆开，不当作单算子。</p>"
        '<div class="rd-tabgroup" data-tab-group="network-endpoints">'
        '<div class="rd-tabs">' + "".join(tabs) + "</div>"
        + "".join(panels) + "</div>"
    )


def _group_section(item: dict, archify: dict, network: dict) -> str:
    gid = _esc(item["id"])
    label = _esc(item.get("label", item["id"]))
    group = next(
        (g for g in network.get("groups", []) if str(g.get("id")) == str(item["id"])),
        {},
    )
    indices = ", ".join(str(i) for i in item.get("layer_indices", []))
    out = [f'<section class="lv rd-lv" id="lv-group-{gid}" data-lv="group" data-gid="{gid}">']
    out.append(_crumbs([("整网", "#/network"), (label, None)]))
    out.append(f'<h2 class="lv-title">02 Decoder · {label}</h2>')
    out.append(
        f'<p class="rd-lead">本组是真实层型组合 <code>{_esc(item.get("attention_type", ""))} + '
        f'{_esc(item.get("moe_type", ""))}</code>，代表层 '
        f'<b>{item.get("representative_layer_index")}</b>，真实层索引见下表。'
        "同组共用一份 Block 模板，模板复用不代表共享权重或删除层间连接。</p>"
    )
    out.append(
        '<div class="table-scroll"><table class="params facts"><tbody>'
        f'<tr><th>层型</th><td class="mono">{_esc(item.get("attention_type", ""))}</td>'
        f'<th>MLP</th><td class="mono">{_esc(item.get("moe_type", ""))}</td></tr>'
        f'<tr><th>真实层数</th><td>{len(item.get("layer_indices", []))}</td>'
        f'<th>代表层</th><td>{item.get("representative_layer_index")}</td></tr>'
        f'<tr><th>真实层索引</th><td class="mono rd-wrap" colspan="3">'
        f'{_esc(indices)}</td></tr>'
        "</tbody></table></div>"
    )
    out.append(_fig_block(
        archify.get("decoder"), figure_id=f"fig-decoder-{gid}",
        caption=f"DeepseekV4DecoderLayer · {item.get('label', item['id'])}（模块级）",
        scope=f"decoder-{gid}", note=_LAYER_NOTES["group"],
    ))
    out.append(
        f'<details class="fold"><summary>层组上下文与到达路径（整网层循环 → 本组 → 代表层）'
        "</summary>"
        + _fig_block(
            archify.get("group"), figure_id=f"fig-group-{gid}",
            caption=f"层组 {label} · 真实层与代表层（上下文）",
            scope=f"group-{gid}",
        )
        + "</details>"
    )
    block = item.get("structure", {}).get("block", {})
    out.append(_param_cells(block.get("parameters", []), f"block-{gid}",
                            [p.get("id") for p in block.get("parameters", [])][:10]))
    out.append(_param_table(block.get("parameters", []), f"block-{gid}"))

    out.append('<h3>模块清单（进入模块关系层）</h3>')
    cards = []
    for group in _module_groups(item):
        ops = group.get("ops", [])
        key_ops = "、".join(str(op.get("name", op.get("id"))) for op in ops[:3])
        more = " 等" if len(ops) > 3 else ""
        cards.append(
            '<div class="mod-card">'
            f'<div class="mod-card-head"><a class="strong-link" '
            f'href="#/group/{gid}/block">{_esc(group.get("title", group.get("id")))}</a>'
            f'<span class="chip-count">{len(ops)} 算子</span></div>'
            f'<div class="mod-card-ops mono">{_esc(key_ops)}{more}</div></div>'
        )
    out.append('<div class="mod-cards">' + "".join(cards) + "</div>")
    out.append(
        f'<p class="lv-next"><a class="btn" href="#/group/{gid}/block">进入模块关系 →</a>'
        " ｜ " + f'<a class="param-link" href="#/group/{gid}/ops">直接看内部算子</a>'
        " ｜ " + '<a class="param-link" href="#/network">↑ 返回整网</a></p>'
    )
    out.append("</section>")
    return "".join(out)


def _block_section(item: dict, archify: dict) -> str:
    gid = _esc(item["id"])
    label = _esc(item.get("label", item["id"]))
    block = item.get("structure", {}).get("block", {})
    out = [f'<section class="lv rd-lv" id="lv-block-{gid}" data-lv="block" data-gid="{gid}">']
    out.append(_crumbs([("整网", "#/network"), (label, f"#/group/{gid}"), ("模块关系", None)]))
    out.append(f'<h2 class="lv-title">03 模块关系 · {label}</h2>')
    out.append(
        '<p class="rd-lead">本层只到“模块之间的关系与真实数据流”：mHC 的 pre/post 分支、'
        "Attention 总览（Q/KV 同图）、MoE 路由与共享专家的关系。逐算子与通路在下一层。</p>"
    )
    tabs = [
        ("mhc", "mHC 多流混合与残差"),
        ("attention", "Attention 总览（Q/KV 同图）"),
        ("moe", "MoE 路由与专家关系"),
    ]
    def _single(value):
        if isinstance(value, list):
            return value[0] if value else None
        return value

    all_entries = {key: _single(archify.get(key)) for key, _ in tabs}
    present = [(key, title) for key, title in tabs if all_entries.get(key)]
    if not present:
        out.append('<div class="rd-missing">本候选未生成模块关系子图。</div>')
    else:
        buttons = []
        for index, (key, title) in enumerate(present):
            on = ' class="on"' if index == 0 else ""
            buttons.append(
                f'<button type="button"{on} data-tab-btn="block-{_esc(key)}">'
                f"{_esc(title)}</button>"
            )
        group_parts = [
            f'<div class="rd-tabgroup" data-tab-group="block-{gid}">',
            '<div class="rd-tabs">' + "".join(buttons) + "</div>",
        ]
        for index, (key, title) in enumerate(present):
            entry = all_entries[key]
            group_parts.append(
                f'<div class="rd-tabbody" data-tab="block-{_esc(key)}"'
                + ("" if index == 0 else " hidden") + ">"
                + _fig_block(
                    entry, figure_id=f"fig-{_esc(key)}-{gid}", caption=title,
                    scope=f"{key}-{gid}", note=_LAYER_NOTES["block"],
                )
                + "</div>"
            )
        group_parts.append("</div>")
        out.append("".join(group_parts))
    out.append(_param_cells(
        block.get("parameters", []), f"block-{gid}",
        [p.get("id") for p in block.get("parameters", [])][:10],
    ))
    out.append('<h3>模块清单（进入内部算子层）</h3>')
    cards = []
    for group in _module_groups(item):
        mid = str(group.get("id"))
        cards.append(
            '<div class="mod-card">'
            f'<div class="mod-card-head"><a class="strong-link" '
            f'href="#/group/{gid}/ops/{_esc(mid)}">{_esc(group.get("title", mid))}</a>'
            f'<span class="chip-count">{len(group.get("ops", []))} 算子</span></div>'
            "</div>"
        )
    out.append('<div class="mod-cards">' + "".join(cards) + "</div>")
    boundary = [t for t in block.get("tensors", []) if t.get("kind") == "boundary"]
    cache = [t for t in block.get("tensors", []) if t.get("kind") == "cache"]
    out.append(
        f'<details class="fold"><summary>块边界与缓存载体（边界 {len(boundary)} 项 · '
        f'缓存 {len(cache)} 项；读前/写后分开）</summary>'
        '<div class="table-scroll"><table class="tensors"><thead><tr>'
        "<th>张量id</th><th>名称</th><th>shape</th><th>类型</th><th>说明</th>"
        "</tr></thead><tbody>"
        + "".join(_tensor_row(t) for t in (*boundary, *cache))
        + "</tbody></table></div></details>"
    )
    out.append(
        f'<p class="lv-next"><a class="btn" href="#/group/{gid}/ops">进入内部算子 →</a>'
        f' ｜ <a class="param-link" href="#/group/{gid}">↑ 返回 Decoder</a>'
        ' ｜ <a class="param-link" href="#/network">↑ 返回整网</a></p>'
    )
    out.append("</section>")
    return "".join(out)


def _tensor_row(tensor: dict) -> str:
    shape = ", ".join(str(dim) for dim in tensor.get("shape", []))
    return (
        "<tr>"
        f'<td class="mono">{_esc(tensor.get("id"))}</td>'
        f"<td>{_esc(tensor.get('name'))}</td>"
        f'<td class="mono">{_esc(shape)}</td>'
        f"<td>{_esc(tensor.get('kind'))}</td>"
        f"<td>{_esc(tensor.get('note'))}</td>"
        "</tr>"
    )


def _cache_state_section(block: dict, gid: str) -> str:
    """缓存逐项表：全称符号 shape、调用前/后、真实读写算子来源。"""
    from structure_reader.render.display_names import expand_display_text

    cache = [t for t in block.get("tensors", []) if t.get("kind") == "cache"]
    if not cache:
        return ""
    ops = list(block.get("operators", []))
    producers = {}
    consumers = {}
    for op in ops:
        op_id = str(op.get("id"))
        for tid in op.get("outputs", ()):
            producers.setdefault(str(tid), []).append(op)
        for tid in op.get("inputs", ()):
            consumers.setdefault(str(tid), []).append(op)

    def op_source(records):
        parts = []
        for op in records[:2]:
            refs = op.get("source_refs") or []
            where = "；".join(
                f'{r.get("path", "")}:{r.get("locator", "")}' for r in refs[:1]
            )
            parts.append(str(op.get("id")) + (f"（{where}）" if where else ""))
        return "；".join(parts) or "—"

    rows = []
    for tensor in cache:
        tid = str(tensor.get("id"))
        if tid.startswith(("tid2eid", "kv_full")):
            continue
        phase = "调用后" if tid.endswith("_post") else "调用前"
        shape = expand_display_text(
            ", ".join(str(dim) for dim in tensor.get("shape", [])) or "[]")
        note = tensor.get("name") or ""
        write_source = op_source(producers.get(tid, []))
        if write_source == "—":
            write_source = "上一调用持久状态"
        source = "写: " + write_source + " ｜ 读: " + op_source(consumers.get(tid, []))
        rows.append(
            "<tr>"
            f'<td class="mono">{_esc(tid)}</td><td>{_esc(phase)}</td>'
            f'<td class="mono">{_esc(shape)}</td><td>{_esc(note)}</td><td>{_esc(source)}</td>'
            "</tr>"
        )
    rows.append(
        "<tr>"
        '<td class="mono">cache_cumlen_post（阅读投影）</td><td>调用后</td>'
        '<td class="mono">kvcachelen+seqlen</td>'
        '<td>逻辑历史累计长度（不新增实际 tensor）</td>'
        '<td>写: cache.update 累计长度；读: 下一调用 mask/cache 状态</td>'
        "</tr>"
    )
    new_kv = next((t for t in block.get("tensors", []) if t.get("id") == "kv_rope"), None)
    full = next((t for t in block.get("tensors", []) if t.get("id") == "kv_full"), None)
    for label, tensor in (("本调用新 KV（cache.update 输入）", new_kv),
                          ("本次可读 full（cache.update 输出）", full)):
        if tensor:
            shape = expand_display_text(
                ", ".join(str(dim) for dim in tensor.get("shape", [])) or "[]")
            rows.append(
                "<tr>"
                f'<td class="mono">{_esc(tensor.get("id"))}</td><td>本次</td>'
                f'<td class="mono">{_esc(shape)}</td><td>{_esc(label)}</td><td>—</td>'
                "</tr>"
            )
    params = block.get("parameters", [])
    ids = ["B", "seqlen", "kvcachelen", "Nkv", "Dh", "W", "M", "Tbuf", "Tbufp",
           "Tcc", "Tci", "Tbufi", "Tbufip", "Nwin", "Nwin_i", "Ncomp", "Ki", "Ih", "Id"]
    return (
        '<h3>缓存逐项身份 / 完整符号 shape（调用前/后分开）</h3>'
        '<p class="rd-note">形状按现有全称符号展开；来源来自真实读写算子。'
        '调用后累计长度是阅读投影 kvcachelen+seqlen，不新增底层 tensor。</p>'
        '<div class="table-scroll"><table class="tensors"><thead><tr>'
        "<th>id</th><th>阶段</th><th>完整 shape</th><th>名称/说明</th><th>真实读写来源</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        '<h3>缓存/压缩器/Indexer 局部参数与来源</h3>'
        + _param_cells(params, f"d03-cache-{gid}", ids)
        + _param_table(params, f"d03-cache-{gid}", ids)
        + '<p class="rd-note">cache.update 规则：新 KV 与调用前 keys/values 合并，'
          'full = cat(keys, new_kv)；持久尾部保留 W-1，cumlen 同步推进；'
          '压缩器/Indexer 使用各自独立键与 entry_count，topk 只生成 block_bias 可见性，'
          '不裁剪 KV。</p>'
    )


def _ops_section(item: dict, archify: dict, source_snapshot) -> str:
    gid = _esc(item["id"])
    label = _esc(item.get("label", item["id"]))
    block = item.get("structure", {}).get("block", {})
    scope = f"block-{item['id']}"
    tensor_index = {t["id"]: t for t in block.get("tensors", [])}
    param_index = {p["id"]: p for p in block.get("parameters", [])}
    groups = _module_groups(item)
    out = [f'<section class="lv rd-lv" id="lv-ops-{gid}" data-lv="ops" data-gid="{gid}">']
    out.append(_crumbs([("整网", "#/network"), (label, f"#/group/{gid}"),
                        ("模块关系", f"#/group/{gid}/block"), ("内部算子", None)]))
    out.append(f'<h2 class="lv-title">04 内部算子与通路 · {label}</h2>')
    out.append(
        '<p class="rd-lead">Attention 内部与 KV/跨调用状态以同源语义图给出（raw 与化简后成对），'
        "算子卡按模块折叠：参数、权重与源码快照按需展开，全部离线可用。</p>"
    )
    out.append(f'<p class="rd-note">{_esc(_LAYER_NOTES["ops"])}</p>')
    def _entry_list(value):
        if isinstance(value, list):
            return [item for item in value if item]
        return [value] if value else []

    tabs = [
        ("attention", "Attention 总览 / Q·KV·O（化简后）", _entry_list(archify.get("attention"))),
        ("q", "Q 通路（独立）", _entry_list(archify.get("q"))),
        ("kv", "KV 新投影（化简后）", _entry_list(archify.get("kv"))),
        ("o", "O 通路（独立）", _entry_list(archify.get("o"))),
        ("cache", "SWA/CSA/HCA 缓存状态", _entry_list(archify.get("cache"))),
        ("compressor", "压缩器（CSA/HCA）", _entry_list(archify.get("compressor"))),
        ("indexer", "Lightning Indexer（CSA）", _entry_list(archify.get("indexer"))),
        ("moe-route", "MoE 路由 / 分派", _entry_list(archify.get("moe_route"))),
        ("moe-expert", "单激活专家内部", _entry_list(archify.get("moe_expert"))),
        ("moe-shared", "共享专家内部", _entry_list(archify.get("moe_shared"))),
        ("moe-combine", "routed + shared 汇合", _entry_list(archify.get("moe_combine"))),
    ]
    present = [(key, title, entries) for key, title, entries in tabs if entries]
    if present:
        buttons = []
        for index, (key, title, _entries) in enumerate(present):
            on = ' class="on"' if index == 0 else ""
            buttons.append(
                f'<button type="button"{on} data-tab-btn="ops-{_esc(key)}">'
                f"{_esc(title)}</button>"
            )
        group_parts = [
            f'<div class="rd-tabgroup" data-tab-group="ops-{gid}">',
            '<div class="rd-tabs">' + "".join(buttons) + "</div>",
        ]
        for index, (key, title, entries) in enumerate(present):
            blocks = []
            for entry_index, entry in enumerate(entries):
                form = str(entry.get("form") or "normalized")
                blocks.append(_fig_block(
                    entry,
                    figure_id=f"fig-ops-{key}-{gid}-{form}-{entry_index}",
                    caption=(title if entry_index == 0 else
                             ("Attention 原始语义图（对照）" if form == "raw" else title)),
                    scope=f"{key}-{gid}",
                ))
            group_parts.append(
                f'<div class="rd-tabbody" data-tab="ops-{_esc(key)}"'
                + ("" if index == 0 else " hidden") + ">" + "".join(blocks) + "</div>"
            )
        group_parts.append("</div>")
        out.append("".join(group_parts))
    out.append(_cache_state_section(block, gid))
    out.append('<h3>按模块的算子卡（参数/权重/源码按需展开）</h3>')
    nav = "".join(
        f'<a class="mod-chip" href="#/group/{gid}/ops/{_esc(group.get("id"))}">'
        f'{_esc(group.get("title", group.get("id")))}'
        f'<span class="chip-count">{len(group.get("ops", []))}</span></a>'
        for group in groups
    )
    out.append('<div class="chips mod-nav">' + nav + "</div>")
    for group in groups:
        mid = str(group.get("id"))
        out.append(
            f'<details class="fold mod" data-module="{gid}/{_esc(mid)}" '
            f'id="mod-{gid}-{_esc(mid)}">'
            f'<summary><b>{_esc(group.get("title", mid))}</b>'
            f'<span class="chip-count">{len(group.get("ops", []))} 算子</span></summary>'
            '<div class="ops">'
        )
        for index, op in enumerate(group.get("ops", []), start=1):
            out.append(render_operator_card(
                op, index, scope, tensor_index, param_index, source_snapshot,
                locate_html=(
                    f'<a class="param-link" href="#/group/{gid}/block">'
                    "在模块关系图中查看 ▸</a>"
                ),
            ))
        out.append("</div></details>")

    out.append('<h3>张量索引（联查）</h3>')
    out.append(
        '<div class="table-scroll"><table class="tensors"><thead><tr>'
        "<th>张量id</th><th>名称</th><th>shape</th><th>类型</th><th>参数引用</th>"
        "<th>说明</th></tr></thead><tbody>"
        + "".join(
            '<tr id="t-%s-%s">'
            % (scope, _esc(t.get("id")))
            + f'<td class="mono">{_esc(t.get("id"))}</td>'
            + f"<td>{_esc(t.get('name'))}</td>"
            + f'<td class="mono">{_esc(", ".join(str(d) for d in t.get("shape", [])))}</td>'
            + f"<td>{_esc(t.get('kind'))}</td>"
            + "<td>"
            + ("；".join(
                f'<a class="param-link mono" href="#p-{scope}-{_esc(str(pr))}">{_esc(pr)}</a>'
                for pr in t.get("parameter_refs", [])
            ) or "—")
            + "</td>"
            + f"<td>{_esc(t.get('note'))}</td></tr>"
            for t in block.get("tensors", [])
        )
        + "</tbody></table></div>"
    )
    out.append('<h3>缓存状态与读写</h3>')
    out.append(render_notes_list(block.get("cache_notes", [])))
    out.append('<h3>覆盖与未覆盖</h3>')
    out.append(render_notes_list(block.get("coverage_notes", [])))
    out.append(
        f'<p class="lv-next"><a class="param-link" href="#/group/{gid}/block">'
        "↑ 返回模块关系</a> ｜ "
        f'<a class="param-link" href="#/group/{gid}">↑ 返回 Decoder</a></p>'
    )
    out.append("</section>")
    return "".join(out)


def _module_groups(item: dict) -> list[dict]:
    """模块分组：优先使用页面已缓存的 ``module_groups``，否则按 block 结构回退。"""
    groups = item.get("module_groups")
    if groups:
        return list(groups)
    block = item.get("structure", {}).get("block", {})
    fallback = []
    for op in block.get("operators", []):
        module = str(op.get("module_path") or "block")
        tail = module.replace("（", ".").replace("）", "").rsplit(".", 1)[-1]
        existing = next((g for g in fallback if g["id"] == tail), None)
        if existing is None:
            existing = {"id": tail, "title": tail, "ops": []}
            fallback.append(existing)
        existing["ops"].append(op)
    return fallback


# ---------------------------------------------------------------------------
# 页面装配
# ---------------------------------------------------------------------------


def render_reading_page(network: dict,
                        structures: list[dict],
                        source_snapshot: dict[str, str] | None = None,
                        input_shas: dict[str, str] | None = None,
                        candidate_id: str | None = None,
                        onnx: dict | None = None,
                        profile: PageProfile | None = None,
                        archify: dict | None = None) -> bytes:
    """整网 + Decoder + 模块关系 + 内部算子 → 单文件离线阅读页（只显示 Archify 图）。"""
    if not isinstance(network, dict) or network.get("schema") != "hf_network_overview.v1":
        raise ReadingPageError("network 必须是 hf_network_overview.v1")
    if not isinstance(structures, list) or not structures:
        raise ReadingPageError("structures 必须是非空列表")
    if not archify or not archify.get("network"):
        raise ReadingPageError("正式阅读入口需要同候选 Archify 图清单")
    group_ids = [str(item.get("id", "")) for item in structures]
    network_groups = [str(g["id"]) for g in network.get("groups", [])]
    if sorted(group_ids) != sorted(network_groups):
        raise ReadingPageError(
            f"structures 与整网层组不一致: {group_ids} vs {network_groups}")
    prof = profile or DEEPSEEK_PAGE_PROFILE
    model = network.get("model") or {}
    scope = network.get("scope") or {}

    parts: list[str] = []
    parts.append('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(
        f"<title>{_esc(model.get('display_name', ''))} · 结构阅读"
        "（整网 → Decoder → 模块关系 → 内部算子）</title>"
    )
    parts.append("<style>" + _page_css(prof) + _reading_css() + "</style></head><body>")
    parts.append(
        '<header class="page rd-head">'
        f"<h1>{_esc(model.get('display_name', ''))} · 结构阅读"
        "（整网 → Decoder → 模块关系 → 内部算子）</h1>"
        '<div class="meta">'
        f"模型键：<b>{_esc(model.get('model_key', ''))}</b> ｜ "
        f"实现：<b>{_esc(model.get('implementation', ''))}</b>（eager 静态分析）｜ "
        f"源码版本：<b class=\"mono\">{_esc(model.get('source_revision', ''))}</b> ｜ "
        f"配置：<b class=\"mono\">{_esc(model.get('config_path', ''))}</b><br>"
        f"入口类：<b class=\"mono\">{_esc(scope.get('entry_class', ''))}</b> ｜ "
        f"主干：<b>{_esc(scope.get('layer_count', ''))} 层</b> · "
        f"hidden_size <b>{_esc(scope.get('hidden_size', ''))}</b> · "
        f"vocab_size <b>{_esc(scope.get('vocab_size', ''))}</b> · "
        f"hc_mult <b>{_esc(scope.get('hc_mult', ''))}</b>"
        "</div>"
        + _scope_banner()
        + _stage_bar()
        + _nav(structures)
        + '<nav class="crumbs" id="rd-crumbs" aria-label="当前位置"></nav>'
        "</header>"
    )
    parts.append('<main class="rd-main" id="rd-main">')
    parts.append(_network_section(network, archify, source_snapshot, input_shas))
    for item in structures:
        item_view = _with_module_groups(item, prof)
        gid = str(item["id"])
        group_archify = (archify.get("groups") or {}).get(gid) or {}
        parts.append(_group_section(item_view, group_archify, network))
        parts.append(_block_section(item_view, group_archify))
        parts.append(_ops_section(item_view, group_archify, source_snapshot))
    parts.append("</main>")
    parts.append(_footer(candidate_id, archify))
    parts.append(_router_script(structures))
    parts.append(_reading_foot())
    return "\n".join(parts).encode("utf-8")


def _with_module_groups(item: dict, prof: PageProfile) -> dict:
    """把页面使用的模块分组缓存到结构副本上（不改结构事实，只做展示投影）。"""
    clone = dict(item)
    block = item.get("structure", {}).get("block")
    if block:
        clone["module_groups"] = prof.module_groups(block)
    return clone


def _scope_banner() -> str:
    return (
        '<div class="rd-banner">'
        "<b>本票（D04）交付</b>：第三层 MoE 关系（gate/路由、激活专家、共享专家、"
        "输出加和）与第四层 Hash/TopK、单激活专家、共享专家和汇合；"
        "D01–D03 的两阶段符号、四层导航、Attention/QKV-O/缓存、mHC 分支保留。"
        "<b>D05 内容尚未交付</b>，未展开处不提供空白占位或假链接。"
        "</div>"
    )


def _footer(candidate_id: str | None, archify: dict) -> str:
    tool = archify.get("tool") or {}
    candidate = _esc(candidate_id or "")
    lines = [
        '<footer class="page rd-foot">',
        "<p>本页由 <code>structure_reader</code> 从本地 Transformers 检出与根 "
        "<code>config.json</code> 静态生成：不导入/实例化/运行模型，不读取真实权重。"
        "图全部由 Archify 原生生成器产出并内嵌（<code>?embed=1</code> 原生 embed 模式），"
        "不修改生成后的 HTML 字节。</p>",
    ]
    if candidate:
        lines.append(
            f'<p class="mono">候选 analysis_id：{candidate} ｜ '
            '<a href="../reports/source_manifest.json">来源清单</a> ｜ '
            '<a href="../run_manifest.json">交付清单</a> ｜ '
            '<a href="../data/archify/manifest.json">Archify 清单</a></p>'
        )
    if tool.get("version"):
        lines.append(
            '<p class="mono">Archify '
            f'{_esc(tool.get("version"))}（Node {_esc(tool.get("node"))}；'
            f'CLI SHA256 {_esc(str(tool.get("cli_sha256", ""))[:16])}…）</p>'
        )
    lines.append("</footer>")
    return "".join(lines)


def _reading_css() -> str:
    return """
.rd-head{max-width:1500px;margin:0 auto}
.rd-head h1{font-size:22px;margin:6px 0 6px}
.rd-banner{background:#eaf0f8;border-left:3px solid #7e9bc4;padding:10px 14px;
  color:#3c526d;font-size:13px;margin:10px 0;border-radius:4px}
.rd-stagebar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0 4px;font-size:13px}
.rd-stagebar-title{color:var(--muted);font-weight:600}
.rd-stage-buttons{display:inline-flex;gap:6px}
.rd-stage-buttons button{font:inherit;font-size:13px;border:1px solid var(--line);
  background:var(--card);color:var(--ink);border-radius:16px;padding:4px 16px;cursor:pointer}
.rd-stage-buttons button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.rd-symbol-note{color:var(--muted);font-size:12.5px}
.rd-stage{background:var(--card);border:1px solid var(--line);border-radius:8px;
  padding:8px 14px;margin:6px 0 10px;font-size:13px}
.rd-stage p{margin:4px 0}
.rd-nav{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 4px;border-top:1px solid var(--line);
  padding-top:10px}
.rd-nav-item{border:1px solid var(--line);border-radius:8px;padding:6px 14px;
  text-decoration:none;color:var(--ink);background:var(--card);font-size:13.5px;font-weight:600}
.rd-nav-item:hover{border-color:var(--accent);color:var(--accent)}
.rd-nav-item[aria-current="true"]{background:var(--accent);color:#fff;border-color:var(--accent)}
.rd-main{max-width:1500px;margin:0 auto;padding:0 4px 60px}
.rd-head + .rd-main{padding-top:8px}
.rd-lv{max-width:1480px}
.rd-lead{color:var(--muted);font-size:13.5px;margin:6px 0 12px}
.rd-note{background:var(--card);border:1px solid var(--line);border-radius:8px;
  padding:8px 12px;font-size:12.5px;color:var(--muted);margin:8px 0 14px}
.rd-params{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
  gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;
  overflow:hidden;margin:12px 0}
.rd-param{background:var(--card);padding:9px 12px;min-width:0}
.rd-param-sym{font-weight:700;color:var(--accent);font-size:13px}
.rd-param-meaning{font-size:12.5px;color:var(--muted);margin:2px 0}
.rd-param-expr{display:block;color:var(--accent);font-size:12px;white-space:normal;
  overflow-wrap:anywhere}
.rd-param-value{font-size:12px;color:var(--ink);margin-top:3px;overflow-wrap:anywhere}
.rd-param-stage span{display:block;font-size:12px;color:var(--ink)}
.rd-param-stage span[hidden]{display:none}
.rd-param-src{font-size:11px;color:var(--muted);margin-top:4px;overflow-wrap:anywhere}
.rd-fig{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:0;margin:16px 0;overflow:hidden}
.rd-fig-head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;
  padding:11px 16px;border-bottom:1px solid var(--line);align-items:baseline}
.rd-fig-title{font-weight:700;font-size:14px}
.rd-fig-meta{color:var(--muted);font-size:12px}
.rd-fig-links{font-size:12.5px;padding:8px 16px 0;margin:0;overflow-wrap:anywhere}
.rd-fig-links a{color:var(--accent);text-decoration:none;border-bottom:1px dotted}
.rd-fig-note{font-size:12.5px;color:var(--muted);padding:0 16px;margin:6px 0 0}
.rd-frame{display:block;border:0;margin:10px auto 0;background:#f7fafc;max-width:100%}
.rd-full{border:0;border-top:1px solid var(--line);border-radius:0;margin:12px 0 0;
  padding:10px 16px;background:var(--bg)}
.rd-full>summary{cursor:pointer;font-weight:600;font-size:13px}
.rd-full-node{margin:8px 0;font-size:13px;overflow-wrap:anywhere}
.rd-full-node code{font-size:12px;color:var(--accent);white-space:normal;overflow-wrap:anywhere}
.rd-full-detail{color:var(--muted);font-size:12px}
.rd-full-tag{color:var(--muted);font-size:11.5px}
.rd-full-src{color:var(--muted);font-size:11px;overflow-wrap:anywhere}
.rd-tabs{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 10px}
.rd-tabs button{border:1px solid var(--line);border-radius:8px;padding:6px 14px;
  background:var(--card);color:var(--ink);font:inherit;font-size:12.5px;cursor:pointer}
.rd-tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.rd-missing{border:1px dashed #b45309;background:#fff7ed;color:#7c2d12;
  border-radius:8px;padding:10px 14px;margin:12px 0;font-size:13px}
.rd-wrap{white-space:normal;overflow-wrap:anywhere}
.rd-tabbody[hidden]{display:none}
@media (max-width:900px){
  .rd-params{grid-template-columns:1fr}
  .rd-head h1{font-size:18px}
}
"""


def _reading_foot() -> str:
    """阅读页脚：与四层命名一致，并保留算子卡“源码”面板开关。"""
    return """<footer class="page rd-foot">
<p>本页是 DeepSeek 正式阅读入口：整网 → Decoder → 模块关系 → 内部算子。
层级由普通链接 + hash 路由切换（<code>#/network</code>、
<code>#/group/&lt;层组&gt;</code>、<code>/block</code>、<code>/ops</code>），
静态 HTML 可离线打开；图是同候选内嵌的 Archify 原生 HTML（不改其字节）。</p>
</footer>
<script>
(function () {
  "use strict";
  window.toggleSource = function (id) {
    var el = document.getElementById(id);
    if (el) el.style.display = (el.style.display === "none") ? "block" : "none";
  };
})();
</script>
</body>
</html>
"""


def _router_script(structures: list[dict]) -> str:
    labels = {str(item["id"]): str(item.get("label", item["id"])) for item in structures}
    labels_json = json.dumps(labels, ensure_ascii=False).replace("<", "\\u003c")
    return """<script>
(function () {
  "use strict";
  var GROUP_LABELS = __GROUP_LABELS__;
  var STAGES = ["prefill", "decode"];
  var state = { stage: "prefill" };
  var sections = Array.prototype.slice.call(document.querySelectorAll("[data-lv]"));
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c];
    });
  }
  function levelRoute(raw) {
    if (raw.indexOf("#/group/") === 0) {
      var rest = raw.slice("#/group/".length).split("/");
      var gid = rest[0];
      if (!GROUP_LABELS[gid]) return null;
      if (rest[1] === "block") return { level: "block", gid: gid };
      if (rest[1] === "ops") return { level: "ops", gid: gid, mod: rest[2] || null };
      return { level: "group", gid: gid };
    }
    if (raw === "" || raw === "#" || raw === "#/network") return { level: "network" };
    return null;
  }
  function anchorRoute(id) {
    var el = document.getElementById(id);
    if (el && el.closest) {
      var sec = el.closest("[data-lv]");
      if (sec) {
        return { level: sec.getAttribute("data-lv"),
                 gid: sec.getAttribute("data-gid") || null, anchor: id };
      }
    }
    return null;
  }
  function parseRoute() {
    var raw = window.location.hash || "";
    var route = levelRoute(raw);
    if (route) return route;
    var id = raw.charAt(0) === "#" ? raw.slice(1) : raw;
    if (id) {
      try { id = decodeURIComponent(id); } catch (e) { /* keep raw */ }
      route = anchorRoute(id);
      if (route) return route;
    }
    return { level: "network" };
  }
  // 页签组作用域：每个章节的页签只操作自己的 [data-tab-group]，不能清空其他章节。
  function tabGroupOf(el) {
    if (!el || !el.closest) return null;
    return el.closest("[data-tab-group]");
  }
  function activateTab(btn) {
    var group = tabGroupOf(btn) || document;
    var key = btn.getAttribute("data-tab-btn");
    Array.prototype.forEach.call(group.querySelectorAll("[data-tab]"), function (body) {
      body.hidden = body.getAttribute("data-tab") !== key;
    });
    Array.prototype.forEach.call(group.querySelectorAll("[data-tab-btn]"), function (other) {
      other.classList.toggle("on", other.getAttribute("data-tab-btn") === key);
    });
  }
  // 防御：跨层进入时若某组没有任何可见图（历史状态/手工 hash），恢复该组第一张。
  function ensureActiveTabs(root) {
    if (!root || !root.querySelectorAll) return;
    Array.prototype.forEach.call(root.querySelectorAll("[data-tab-group]"), function (group) {
      var bodies = group.querySelectorAll("[data-tab]");
      var visible = 0;
      Array.prototype.forEach.call(bodies, function (body) {
        if (!body.hidden) visible += 1;
      });
      if (visible === 0 && bodies.length) {
        var first = bodies[0];
        var key = first.getAttribute("data-tab");
        Array.prototype.forEach.call(bodies, function (body) {
          body.hidden = body !== first;
        });
        Array.prototype.forEach.call(group.querySelectorAll("[data-tab-btn]"), function (btn) {
          btn.classList.toggle("on", btn.getAttribute("data-tab-btn") === key);
        });
      }
    });
  }
  function reveal(id) {
    var el = document.getElementById(id);
    if (!el) return false;
    var node = el;
    while (node && node !== document.body) {
      if (node.tagName === "DETAILS") node.open = true;
      node = node.parentElement;
    }
    var tab = el.closest ? el.closest("[data-tab]") : null;
    if (tab && tab.hidden) {
      var key = tab.getAttribute("data-tab");
      var group = tabGroupOf(tab);
      if (group) {
        Array.prototype.forEach.call(group.querySelectorAll("[data-tab]"), function (b) {
          b.hidden = b.getAttribute("data-tab") !== key;
        });
        Array.prototype.forEach.call(group.querySelectorAll("[data-tab-btn]"), function (btn) {
          btn.classList.toggle("on", btn.getAttribute("data-tab-btn") === key);
        });
      }
    }
    if (el.scrollIntoView) el.scrollIntoView({ block: "center" });
    el.classList.add("anchor-target");
    setTimeout(function () { el.classList.remove("anchor-target"); }, 2400);
    return true;
  }
  function renderStage() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-stage]"), function (panel) {
      panel.hidden = panel.getAttribute("data-stage") !== state.stage;
    });
    Array.prototype.forEach.call(document.querySelectorAll("[data-stage-btn]"), function (btn) {
      var on = btn.getAttribute("data-stage-btn") === state.stage;
      btn.classList.toggle("on", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
    });
    Array.prototype.forEach.call(document.querySelectorAll("[data-stage-value]"), function (el) {
      el.hidden = el.getAttribute("data-stage-value") !== state.stage;
    });
    Array.prototype.forEach.call(document.querySelectorAll("[data-stage-badge]"), function (el) {
      el.textContent = state.stage === "decode" ? "Decode" : "Prefill";
    });
  }
  function renderNav(route) {
    var gid = route.gid || Object.keys(GROUP_LABELS)[0];
    var hrefs = {
      network: "#/network",
      group: "#/group/" + gid,
      block: "#/group/" + gid + "/block",
      ops: "#/group/" + gid + "/ops"
    };
    Array.prototype.forEach.call(document.querySelectorAll("[data-nav]"), function (item) {
      var key = item.getAttribute("data-nav");
      item.setAttribute("href", hrefs[key] || "#/network");
      item.setAttribute("aria-current", key === route.level ? "true" : "false");
    });
  }
  function renderCrumbs(route) {
    var el = document.getElementById("rd-crumbs");
    if (!el) return;
    var parts = ['<a href="#/network">整网</a>'];
    if (route.level !== "network" && route.gid) {
      var label = GROUP_LABELS[route.gid] || route.gid;
      var base = "#/group/" + route.gid;
      if (route.level === "group") {
        parts.push('<span aria-current="page">Decoder：' + esc(label) + '</span>');
      } else if (route.level === "block") {
        parts.push('<a href="' + base + '">Decoder：' + esc(label) + '</a>');
        parts.push('<span aria-current="page">模块关系</span>');
      } else {
        parts.push('<a href="' + base + '">Decoder：' + esc(label) + '</a>');
        parts.push('<a href="' + base + '/block">模块关系</a>');
        parts.push('<span aria-current="page">内部算子</span>');
      }
    } else if (route.level === "network") {
      parts.push('<span aria-current="page">整网</span>');
    }
    el.innerHTML = "当前位置：" + parts.join(" › ");
  }
  function show(route) {
    var activeId = "lv-" + route.level + (route.gid ? "-" + route.gid : "");
    sections.forEach(function (sec) {
      sec.hidden = sec.id !== activeId;
      sec.style.display = sec.id === activeId ? "block" : "none";
    });
    document.body.setAttribute("data-level", route.level);
    document.body.setAttribute("data-group", route.gid || "");
    ensureActiveTabs(document.getElementById(activeId));
    renderCrumbs(route);
    renderNav(route);
    renderStage();
    if (route.anchor && reveal(route.anchor)) return;
    if (route.level === "ops" && route.mod) {
      var target = document.querySelector('[data-module="' + route.gid + "/" + route.mod + '"]');
      if (target) {
        target.open = true;
        Array.prototype.forEach.call(document.querySelectorAll("details.mod"), function (d) {
          if (d !== target) d.open = false;
        });
        reveal(target.id);
        return;
      }
    }
    if (!route.anchor) window.scrollTo(0, 0);
  }
  function bindTabs() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-tab-btn]"), function (btn) {
      btn.addEventListener("click", function () { activateTab(btn); });
    });
    Array.prototype.forEach.call(document.querySelectorAll("[data-stage-btn]"), function (btn) {
      btn.addEventListener("click", function () {
        state.stage = btn.getAttribute("data-stage-btn");
        renderStage();
      });
    });
  }
  window.locateInDiagram = function (figureId) {
    var el = document.getElementById(figureId);
    if (el && el.scrollIntoView) el.scrollIntoView({ block: "center" });
  };
  window.addEventListener("hashchange", function () { show(parseRoute()); });
  document.addEventListener("DOMContentLoaded", function () { bindTabs(); show(parseRoute()); });
  if (document.readyState !== "loading") { bindTabs(); show(parseRoute()); }
})();
</script>""".replace("__GROUP_LABELS__", labels_json)
