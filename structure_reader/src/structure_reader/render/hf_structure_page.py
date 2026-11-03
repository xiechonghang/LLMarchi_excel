"""HF 分层结构浏览页（整网 → 层组 → Block → 内部语义算子，单文件离线 HTML）。

一份参数化结构的四层阅读界面：

- **整网**（``#/network``）：token→logits 主路径图、真实层顺序、按 profile 命名的层组入口；
  网络级参数/张量/算子/来源身份放在默认折叠的 ``<details>`` 里按需查看。
- **层组**（``#/group/<gid>``）：该组真实层索引与代表层、同构模板说明、场景派生面板
  （seqlen/kvcachelen 预设的窗口/缓存规模）、Block 模板的模块链。
- **Block**（``#/group/<gid>/block``）：Block 总览图（真实分支/残差/缓存载体）、
  模块卡片、块边界与缓存载体清单、参数表（折叠）。
- **内部算子**（``#/group/<gid>/ops[/<module>]``）：按模块折叠的模块图 + 算子卡
  （参数/权重/源码按需展开）、张量索引、缓存与覆盖注记。

默认只显示当前层级：CSS 默认只展开整网，其余层级由页面脚本按 hash 路由切换；
每层都带静态面包屑，进入/返回都是普通链接。样式与需求内联，无外部资源。
渲染层不做文件 I/O；源码快照由构建层只读读入后传入。

与旧 ``render/structure_page.py``（参考实现模型的自包含页面）分开：本模块只服务
HF 阅读文档，复用 ``diagram`` / ``detail_render`` / ``module_narrative`` 的既有能力，
不移植旧页面的模型结构结论。
"""

from __future__ import annotations

import copy
import html
import json
import re

from structure_reader.render.decoder_page import _PAGE_HEAD
from structure_reader.render.detail_render import (
    render_notes_list,
    render_operator_card,
    render_parameter_row,
    render_tensor_row,
)
from structure_reader.render.diagram import (
    build_module_graph,
    diagram_css,
    legend_html,
    render_svg,
)
from structure_reader.render.diagram_source_labels import use_source_table
from structure_reader.render.hf_model_profile import (
    DEEPSEEK_PAGE_PROFILE,
    PageProfile,
)

__all__ = ["render_structure_page", "StructurePageError"]

_SCOPE_PREFIX = "block"


class StructurePageError(ValueError):
    """分层浏览页渲染失败（文档缺字段/结构不符），具名拒绝。"""


def render_structure_page(network: dict,
                          structures: list[dict],
                          source_snapshot: dict[str, str] | None = None,
                          input_shas: dict[str, str] | None = None,
                          candidate_id: str | None = None,
                          onnx: dict | None = None,
                          profile: PageProfile | None = None,
                          archify: dict | None = None) -> bytes:
    """整网总览 + 各层组参数化 Block 结构 → 单文件分层浏览 HTML 字节。

    ``profile`` 提供模型特有的展示规则；缺省为 DeepSeek（既有输出不变）。源码命名
    表按 profile 临时启用，退出后复原，不污染其它模型的渲染。
    ``archify`` 为可选的底层分层图入口（每层可进入同源 Archify 图并查看完整关系/
    来源）；缺省时页面输出与既有候选完全一致。
    """
    prof = profile or DEEPSEEK_PAGE_PROFILE
    table = prof.source_table(structures, network) if prof.source_table else None
    with use_source_table(table):
        return _render_structure_page(
            network, structures, source_snapshot=source_snapshot,
            input_shas=input_shas, candidate_id=candidate_id, onnx=onnx,
            profile=prof, archify=archify)


def _render_structure_page(network: dict,
                           structures: list[dict],
                           source_snapshot: dict[str, str] | None = None,
                           input_shas: dict[str, str] | None = None,
                           candidate_id: str | None = None,
                           onnx: dict | None = None,
                           profile: PageProfile | None = None,
                           archify: dict | None = None) -> bytes:
    """分层浏览页装配（profile 必填；公共入口负责选择/复原源码命名表）。"""
    prof = profile or DEEPSEEK_PAGE_PROFILE
    if not isinstance(network, dict) or network.get("schema") != "hf_network_overview.v1":
        raise StructurePageError("network 必须是 hf_network_overview.v1")
    if not isinstance(structures, list) or not structures:
        raise StructurePageError("structures 必须是非空列表")
    group_ids = [str(item.get("id", "")) for item in structures]
    if len(group_ids) != len(set(group_ids)) or not all(group_ids):
        raise StructurePageError("层组 id 缺失或重复")
    network_groups = [g["id"] for g in network.get("groups", [])]
    if sorted(group_ids) != sorted(network_groups):
        raise StructurePageError(
            f"structures 与整网层组不一致: {group_ids} vs {network_groups}")
    for item in structures:
        doc = item.get("structure") or {}
        if doc.get("schema") != "hf_layer_structure.v1":
            raise StructurePageError(f"层组 {item.get('id')} 结构 schema 非法")

    model = network["model"]
    scope = network["scope"]
    detail = network["detail"]
    detail_href = str(detail.get("href") or "structure.html")

    parts: list[str] = []
    parts.append("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">")
    parts.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    parts.append(f"<title>{html.escape(str(model.get('display_name', '')))} 分层结构浏览</title>")
    parts.append("<style>" + _page_css(prof) + "</style></head><body>")
    parts.append(
        '<header class="page">'
        f"<h1>{html.escape(str(model.get('display_name', '')))} — 分层结构浏览（整网 → 层组 → Block → 算子）</h1>"
        '<div class="meta">'
        f"模型键：<b>{html.escape(str(model.get('model_key', '')))}</b> ｜ "
        f"实现：<b>{html.escape(str(model.get('implementation', '')))}</b>（eager）｜ "
        f"源码版本：<b class=\"mono\">{html.escape(str(model.get('source_revision', '')))}</b> ｜ "
        f"配置：<b class=\"mono\">{html.escape(str(model.get('config_path', '')))}</b><br>"
        f"入口类：<b class=\"mono\">{html.escape(str(scope.get('entry_class', '')))}</b> ｜ "
        f"主干：<b>{html.escape(str(scope.get('layer_count', '')))} 层</b> · "
        f"hidden_size <b>{html.escape(str(scope.get('hidden_size', '')))}</b> · "
        f"vocab_size <b>{html.escape(str(scope.get('vocab_size', '')))}</b>"
        + prof.scope_meta(scope)
        + "<br>阅读方式：每一层默认只显示当前层级的内容；层组/Block/算子逐级进入，"
        "面包屑与「返回」链接回到上级。参数、权重、源码面板按需展开；"
        + html.escape(profile.header_shape_note)
        + "</div>"
        + _scenario_bar(prof)
        + '<nav class="crumbs" id="crumbs" aria-label="当前位置"></nav>'
        + "</header>"
    )

    parts.append(_network_section(network, detail_href, source_snapshot, input_shas,
                                  profile=prof, archify=archify))
    for item in structures:
        block = _render_block(item)
        gid = str(item.get("id"))
        group_archify = (archify or {}).get("groups", {}).get(gid) if archify else None
        parts.append(_group_section(item, block, detail_href, profile=prof,
                                    archify=group_archify))
        parts.append(_block_section(item, block, detail_href, profile=prof,
                                    archify=group_archify))
        parts.append(_ops_section(item, block, source_snapshot, detail_href,
                                 candidate_id, onnx, profile=prof,
                                 archify=group_archify))
    parts.append(_scenario_data_script(structures))
    parts.append(_router_script(structures, prof))
    parts.append(_PAGE_FOOT)
    return "\n".join(parts).encode("utf-8")


# ---------------------------------------------------------------------------
# 结构准备
# ---------------------------------------------------------------------------


def _render_block(item: dict) -> dict:
    """结构副本：id 作用域化为 block-<gid>，供图/卡片锚点在整页内唯一。"""
    block = copy.deepcopy(item["structure"]["block"])
    block["id"] = f"{_SCOPE_PREFIX}-{item['id']}"
    return block


# ---------------------------------------------------------------------------
# 整网层
# ---------------------------------------------------------------------------


def _scenario_bar(profile: PageProfile) -> str:
    return (
        '<div class="scn-bar" role="group" aria-label="分析场景">'
        f'<span class="scn-title">{html.escape(profile.scenario_label)}：</span>'
        '<span class="scn-buttons" id="scn-buttons"></span>'
        '<span class="scn-current" id="scn-current"></span></div>'
    )


def _network_section(network: dict, detail_href: str,
                     source_snapshot: dict[str, str] | None,
                     input_shas: dict[str, str] | None,
                     *, profile: PageProfile,
                     archify: dict | None = None) -> str:
    scope = network["scope"]
    path = network["path"]
    layer_count = scope.get("layer_count", len(network.get("layer_order", [])))
    out = ['<section class="lv" id="lv-network" data-lv="network">']
    out.append(_crumbs([("整网", None)]))
    out.append(f'<h2 class="lv-title">{html.escape(profile.network_title)}</h2>')
    if profile.network_note is not None:
        out.append(profile.network_note(layer_count))
    else:
        out.append('<p class="dg-note">本层只显示整网主路径与层组入口。节点/边来自按源码逐句记录的'
                   "算子与张量引用；箭头标签是源码变量名与符号 shape，点节点可跳到下方算子卡，"
                   f"点箭头标签可跳到张量索引。层循环体（{layer_count} 层）不在此铺开，"
                   f"{profile.network_groups_label}的层内结构见各自的 Block/算子层。</p>")
    out.append('<div class="diagram-scroll">' + render_svg(
        _network_graph(path, profile=profile)) + "</div>")
    out.append(legend_html(profile.legend))
    out.append(profile.network_extra(layer_count))
    if archify:
        out.append(_archify_panel(
            [archify.get("network")] if archify.get("network") else [],
            heading="整网底层图（Archify，同源生成）",
            note="该图由整网结构文档的程序化投影生成，节点/关系与上方总览同源；"
                 "在新标签打开后本层上下文与返回链接保持不变。",
            back_links=()))

    out.append(f'<h3>{profile.group_heading}</h3>')
    out.append('<p class="dg-note">每组只保留一份 Block 模板，但模板复用不代表共享权重：'
               "同组每层是独立实例。点组名进入该组的层组层。</p>")
    out.append(_layer_strip(network["layer_order"], detail_href, profile=profile))
    out.append(_group_cards(network["groups"], detail_href, profile=profile))

    out.append('<details class="fold" id="net-params"><summary>网络级参数符号（'
               f"{len(path.get('parameters', []))} 项）</summary>")
    out.append('<div class="table-scroll"><table class="params"><thead><tr>'
               "<th>参数（完整名称）</th><th>含义</th><th>值</th><th>表达式</th>"
               "<th>来源</th><th>说明</th></tr></thead><tbody>")
    path_id = str(path.get("id", "network"))
    for param in path.get("parameters", []):
        out.append(render_parameter_row(param, path_id))
    out.append("</tbody></table></div></details>")

    out.append('<details class="fold" id="net-tensors"><summary>网络级张量索引（'
               f"{len(path.get('tensors', []))} 项）</summary>")
    out.append('<div class="table-scroll"><table class="tensors"><thead><tr>'
               "<th>张量id</th><th>名称</th><th>shape</th><th>类型</th><th>参数引用</th>"
               "<th>说明</th></tr></thead><tbody>")
    for tensor in path.get("tensors", []):
        out.append(render_tensor_row(tensor, path_id))
    out.append("</tbody></table></div></details>")

    out.append('<details class="fold" id="net-ops"><summary>网络级算子流（'
               f"{len(path.get('operators', []))} 个算子；源码按需展开）</summary>")
    tensor_index = {t["id"]: t for t in path.get("tensors", [])}
    param_index = {p["id"]: p for p in path.get("parameters", [])}
    out.append('<div class="ops">')
    for index, op in enumerate(path.get("operators", []), start=1):
        out.append(render_operator_card(
            op, index, path_id, tensor_index, param_index, source_snapshot))
    out.append("</div></details>")

    out.append('<details class="fold coverage"><summary>整网缓存前提与覆盖范围</summary>')
    out.append("<h4>缓存前提与读写（整网级）</h4>")
    out.append(render_notes_list(path.get("cache_notes", [])))
    out.append("<h4>覆盖与未覆盖</h4>")
    out.append(render_notes_list(path.get("coverage_notes", [])))
    out.append("</details>")

    out.append('<details class="fold" id="net-sources"><summary>来源身份核对（不下载、不伪造）</summary>')
    out.append(_sources_table(input_shas))
    out.append('<p class="np-note">本页只静态读取上述官方源码与根配置，不导入/实例化/运行 HF 模型、'
               "不下载权重；缺失或漂移时构建具名拒绝（不沿用旧结论）。工具身份与交付 SHA 见同候选 "
               "<code>reports/source_manifest.json</code> 与 <code>run_manifest.json</code>。</p></details>")
    out.append("</section>")
    return "".join(out)


def _network_graph(path: dict, *, profile: PageProfile) -> dict:
    """用 diagram 引擎按真实张量引用建网络级模块图；层循环节点按模型 profile 补文案。"""
    phase = {
        "id": path.get("id", "network"),
        "parameters": path.get("parameters", []),
        "tensors": path.get("tensors", []),
        "operators": path.get("operators", []),
    }
    group = {"id": "network", "title": path.get("title", "整网主路径"),
             "ops": path.get("operators", [])}
    graph = build_module_graph(phase, group)
    loop_count = next(
        (str(p.get("value")) for p in path.get("parameters", []) if p.get("id") == "Nlay"), "N")
    for node in graph["nodes"]:
        if node["id"] == "op:net_layer_loop":
            profile.network_loop_node(node, loop_count)
    return graph


def _layer_strip(layer_order: list[dict], detail_href: str,
                 *, profile: PageProfile) -> str:
    cells = []
    for entry in layer_order:
        gid = str(entry.get("group_id", ""))
        href = f"{html.escape(detail_href)}#/group/{html.escape(gid)}"
        short = profile.layer_cell(entry)
        cells.append(
            f'<li class="np-layer np-{html.escape(gid)}">'
            f'<a href="{href}" title="层 {entry.get("index")}：{html.escape(gid)}'
            f'（{html.escape(str(entry.get("attention_type", "")))} + '
            f'{html.escape(str(entry.get("moe_type", "")))}）— 进入该层所属层组">'
            f'<span class="np-layer-idx">{entry.get("index")}</span>'
            f'<span class="np-layer-type">{html.escape(short)}</span></a></li>'
        )
    return (f'<h4>{len(layer_order)} 层真实顺序（每格＝该层所属组，点入层组）</h4>'
            '<ol class="np-layers">' + "".join(cells) + "</ol>")


def _group_cards(groups: list[dict], detail_href: str, *, profile: PageProfile) -> str:
    rows = []
    for group in groups:
        gid = str(group["id"])
        indices = ", ".join(str(i) for i in group.get("layer_indices", []))
        rows.append(
            '<tr class="group-row">'
            f'<td><a class="strong-link" href="{html.escape(detail_href)}#/group/{html.escape(gid)}">'
            f'{html.escape(str(group.get("label", gid)))}</a><br>'
            f'<span class="mono">{html.escape(gid)}</span></td>'
            f'<td class="mono">{html.escape(str(group.get("attention_type", "")))} + '
            f'{html.escape(str(group.get("moe_type", "")))}</td>'
            f'<td>{group.get("count", len(group.get("layer_indices", [])))}</td>'
            f'<td class="mono">{html.escape(indices)}</td>'
            f'<td>{group.get("representative_layer_index")}</td>'
            f'<td><a class="param-link" href="{html.escape(detail_href)}#/group/{html.escape(gid)}">'
            "进入层组 →</a></td></tr>"
        )
    return ('<div class="table-scroll"><table class="params"><thead><tr>'
            f"<th>层组</th><th>{html.escape(profile.group_type_header)}</th><th>层数</th>"
            "<th>真实层索引</th>"
            "<th>代表层</th><th>进入</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


# ---------------------------------------------------------------------------
# 层组层
# ---------------------------------------------------------------------------


def _group_section(item: dict, block: dict, detail_href: str,
                   *, profile: PageProfile,
                   archify: dict | None = None) -> str:
    gid = html.escape(str(item["id"]))
    label = html.escape(str(item.get("label", item["id"])))
    base = f"{html.escape(detail_href)}#/group/{gid}"
    groups = profile.module_groups(block)
    out = [f'<section class="lv" id="lv-group-{gid}" data-lv="group" data-gid="{gid}">']
    out.append(_crumbs([("整网", f"{html.escape(detail_href)}#/network"), (label, None)]))
    out.append(f'<h2 class="lv-title">{label}：重复层组</h2>')
    out.append('<div class="table-scroll"><table class="params facts"><tbody>'
               f'<tr><th>{html.escape(profile.group_type_label)}</th>'
               f'<td class="mono">{html.escape(str(item.get("attention_type", "")))} + '
               f'{html.escape(str(item.get("moe_type", "")))}</td>'
               f'<th>真实层数</th><td>{len(item.get("layer_indices", []))}</td></tr>'
               f'<tr><th>真实层索引</th><td class="mono" colspan="3">'
               f'{html.escape(", ".join(str(i) for i in item.get("layer_indices", [])))}</td></tr>'
               f'<tr><th>代表层</th><td>{item.get("representative_layer_index")}</td>'
               "<th>结构</th><td>同组共用一份模板；模板复用不代表共享权重或删除层间连接</td></tr>"
               + profile.group_facts_extra(item)
               + "</tbody></table></div>")
    out.append('<div class="scn-slot" data-scn-group="' + gid + '"></div>')
    out.append('<h3>Block 模板（当前层级 → Block）</h3>')
    out.append(profile.group_block_note)
    chips = []
    for group in groups:
        chips.append(
            f'<a class="mod-chip" href="{base}/ops/{html.escape(str(group["id"]))}">'
            f'{html.escape(str(group.get("title", group["id"])))}'
            f'<span class="chip-count">{len(group.get("ops", []))}</span></a>')
    out.append('<div class="chips">' + "".join(chips) + "</div>")
    if archify:
        prefix = f"{html.escape(detail_href)}#/group/{gid}"
        out.append(_archify_panel(
            [archify.get("group")] if archify.get("group") else [],
            heading="层组底层图（Archify，同源生成）",
            note=(str(archify.get("group_note") or "")
                  + "该图把整网层循环、本组真实层、代表层与 Block 模板连成一条进入路径；"
                  "Block 内部张量数据流见下一层的 Block 图。"),
            back_links=(("↑ 返回整网", f"{html.escape(detail_href)}#/network"),
                        ("进入 Block 层", f"{prefix}/block"))))
    out.append(f'<p class="lv-next"><a class="btn" href="{base}/block">'
               f'{html.escape(profile.block_enter_label)}</a></p>')
    out.append("</section>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Block 层
# ---------------------------------------------------------------------------


def _block_section(item: dict, block: dict, detail_href: str,
                   *, profile: PageProfile,
                   archify: dict | None = None) -> str:
    gid = html.escape(str(item["id"]))
    label = html.escape(str(item.get("label", item["id"])))
    base = f"{html.escape(detail_href)}#/group/{gid}"
    groups = profile.module_groups(block)
    out = [f'<section class="lv" id="lv-block-{gid}" data-lv="block" data-gid="{gid}">']
    out.append(_crumbs([("整网", f"{html.escape(detail_href)}#/network"),
                        (label, base), ("Block", None)]))
    out.append(f'<h2 class="lv-title">{label} · {html.escape(profile.block_title)}</h2>')
    out.append(f'<p class="dg-note">{html.escape(str(block.get("title", "")))}</p>')
    out.append(f'<details class="fold" id="block-scenario-{gid}">'
               f"<summary>{html.escape(profile.scenario_panel_summary)}</summary>"
               '<div class="scn-slot" data-scn-group="' + gid + '"></div></details>')
    out.append(profile.overview_narrative(block))
    out.append('<div class="diagram-scroll">' + render_svg(profile.overview_graph(block)) + "</div>")
    out.append(legend_html(profile.legend))
    out.append(profile.block_diagram_note)
    if archify:
        prefix = f"{html.escape(detail_href)}#/group/{gid}"
        if archify.get("subviews"):
            # 五模型通用：子图由模型声明给出（Indexer/PLE/DeepStack 等不硬套 DeepSeek 槽位）。
            sub_views = tuple(
                (str(item.get("label") or ""), list(item.get("entries") or ()))
                for item in archify.get("subviews") or ()
                if item.get("entries")
            )
            note = str(archify.get("block_note") or (
                "模块级张量数据流：跨调用状态以调用前/后分开，辅助输入走短旁枝；"
                "Attention/FFN/MoE/超连接等细节在下方子图。"))
        else:
            sub_views = (
                ("Attention / KV（逐张量语义图）",
                 list(archify.get("attention") or ()) + list(archify.get("kv") or ())),
                ("FFN / MoE 内部计算",
                 [archify["moe"]] if archify.get("moe") else []),
                ("mHC 多流混合与残差",
                 [archify["mhc"]] if archify.get("mhc") else []),
            )
            note = (
                "模块级张量数据流：mHC 的 pre/post/comb 作为真实依赖进入两段残差，"
                "缓存以调用前/后状态表达；Attention 内部与 FFN/MoE/mHC 细节在下方子图。")
        out.append(_archify_panel(
            [archify.get("decoder")] if archify.get("decoder") else [],
            heading="Block/Decoder 底层图（Archify，同源生成）",
            note=note,
            back_links=(("↑ 返回层组", base),
                        ("↑ 返回整网", f"{html.escape(detail_href)}#/network")),
            sub_views=sub_views))

    out.append('<h3>模块清单（进入内部算子层）</h3>')
    cards = []
    for group in groups:
        ops = group.get("ops", [])
        key_ops = "、".join(html.escape(str(op.get("name", op.get("id")))) for op in ops[:3])
        more = "…" if len(ops) > 3 else ""
        cards.append(
            '<div class="mod-card">'
            f'<div class="mod-card-head"><a class="strong-link" '
            f'href="{base}/ops/{html.escape(str(group["id"]))}">'
            f'{html.escape(str(group.get("title", group["id"])))}</a>'
            f'<span class="chip-count">{len(ops)} 算子</span></div>'
            f'<div class="mod-card-ops mono">{key_ops}{more}</div></div>')
    out.append('<div class="mod-cards">' + "".join(cards) + "</div>")

    boundary = [t for t in block.get("tensors", []) if t.get("kind") == "boundary"]
    cache = [t for t in block.get("tensors", []) if t.get("kind") == "cache"]
    out.append(f'<details class="fold" id="block-boundary-{gid}"><summary>'
               "块边界与缓存载体（边界 "
               f"{len(boundary)} 项 · 缓存 {len(cache)} 项；读前/写后分开）</summary>")
    out.append('<div class="table-scroll"><table class="tensors"><thead><tr>'
               "<th>张量id</th><th>名称</th><th>shape</th><th>类型</th><th>说明</th>"
               "</tr></thead><tbody>")
    for tensor in (*boundary, *cache):
        # 同一结构在 L3 另有完整张量索引：此处只列内容，避免重复 id
        out.append(render_tensor_row(tensor, block["id"], anchor=False))
    out.append("</tbody></table></div></details>")

    out.append(f'<details class="fold" id="block-params-{gid}"><summary>'
               "Block 参数符号（"
               f"{len(block.get('parameters', []))} 项；场景参数与配置常量）</summary>")
    out.append('<div class="table-scroll"><table class="params"><thead><tr>'
               "<th>参数（完整名称）</th><th>含义</th><th>值</th><th>表达式</th>"
               "<th>来源</th><th>说明</th></tr></thead><tbody>")
    for param in block.get("parameters", []):
        out.append(render_parameter_row(param, block["id"]))
    out.append("</tbody></table></div></details>")

    out.append(profile.variant_selection(item))
    out.append(f'<p class="lv-next"><a class="btn" href="{base}/ops">进入内部算子 →</a>'
               f' <a class="param-link" href="{base}">↑ 返回层组</a></p>')
    out.append("</section>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 内部算子层
# ---------------------------------------------------------------------------


def _ops_section(item: dict, block: dict, source_snapshot: dict[str, str] | None,
                 detail_href: str, candidate_id: str | None,
                 onnx: dict | None = None, *, profile: PageProfile,
                 archify: dict | None = None) -> str:
    gid = html.escape(str(item["id"]))
    label = html.escape(str(item.get("label", item["id"])))
    base = f"{html.escape(detail_href)}#/group/{gid}"
    groups = profile.module_groups(block)
    by_id = {g["id"]: g for g in groups}
    tensor_index = {t["id"]: t for t in block.get("tensors", [])}
    param_index = {p["id"]: p for p in block.get("parameters", [])}
    out = [f'<section class="lv" id="lv-ops-{gid}" data-lv="ops" data-gid="{gid}">']
    out.append(_crumbs([("整网", f"{html.escape(detail_href)}#/network"),
                        (label, base), ("Block", f"{base}/block"), ("内部算子", None)]))
    out.append(f'<h2 class="lv-title">{label} · 内部语义算子</h2>')
    out.append('<p class="dg-note">按模块折叠：展开模块才显示模块图与算子卡；算子卡的参数/权重/'
               "源码面板再按需展开。全部模块共用同一份 Block 结构，不复制结构事实。</p>")
    nav = "".join(
        f'<a class="mod-chip" href="{base}/ops/{html.escape(str(g["id"]))}">'
        f'{html.escape(str(g.get("title", g["id"])))}'
        f'<span class="chip-count">{len(g.get("ops", []))}</span></a>'
        for g in groups)
    out.append('<div class="chips mod-nav">' + nav + "</div>")

    for group in groups:
        mid = str(group["id"])
        out.append(f'<details class="fold mod" data-module="{gid}/{html.escape(mid)}" '
                   f'id="mod-{html.escape(str(block["id"]))}-{html.escape(mid)}">')
        out.append(f'<summary><b>{html.escape(str(group.get("title", mid)))}</b>'
                   f'<span class="chip-count">{len(group.get("ops", []))} 算子</span></summary>')
        nested_note = profile.nested_module_note(mid, by_id)
        if nested_note:
            # 该模块已作为复合图的嵌套子容器画出：不再重画同批算子（锚点会重复），
            # 只保留说明与算子卡。
            out.append(nested_note)
        else:
            out.append('<div class="diagram-scroll">'
                       + render_svg(profile.module_graph(block, group, by_id)) + "</div>")
        narr = profile.module_narrative(block, mid)
        if narr:
            out.append(narr)
        out.append(_onnx_links(gid, mid, onnx))
        out.append('<div class="ops">')
        for index, op in enumerate(group.get("ops", []), start=1):
            out.append(render_operator_card(
                op, index, block["id"], tensor_index, param_index, source_snapshot))
        out.append("</div></details>")

    out.append('<h3>张量索引（联查）</h3>')
    out.append('<div class="table-scroll"><table class="tensors"><thead><tr>'
               "<th>张量id</th><th>名称</th><th>shape</th><th>类型</th><th>参数引用</th>"
               "<th>说明</th></tr></thead><tbody>")
    for tensor in block.get("tensors", []):
        out.append(render_tensor_row(tensor, block["id"]))
    out.append("</tbody></table></div>")

    out.append('<h3>缓存状态与读写（同一结构，场景绑定取值）</h3>')
    out.append(render_notes_list(block.get("cache_notes", [])))
    out.append('<h3>覆盖与未覆盖</h3>')
    out.append(render_notes_list(block.get("coverage_notes", [])))
    if candidate_id:
        out.append(f'<p class="np-note mono">候选：{html.escape(str(candidate_id))}</p>')
    out.append(f'<p class="lv-next"><a class="param-link" href="{base}/block">↑ 返回 Block</a>'
               f' ｜ <a class="param-link" href="{html.escape(detail_href)}#/network">↑ 返回整网</a></p>')
    out.append("</section>")
    return "".join(out)




def _onnx_links(gid: str, mid: str, onnx: dict | None) -> str:
    """模块级 ONNX/Netron 联查（可选资产；未启用时返回空串，不伪造）。

    ONNX 图是阅读特化投影（按 prefill/decode 各建一份），只作辅助联查，
    不是结构事实来源，也不代表量化执行或数值等价。
    """
    if not onnx:
        return ""
    candidate = str(onnx.get("candidate_id") or "")
    bundles = onnx.get("bundles") or {}
    if not candidate:
        return ""
    parts: list[str] = []
    for phase in ("prefill", "decode"):
        entry = (bundles.get(gid) or {}).get(phase, {}).get(mid)
        if not entry:
            continue
        state = entry.get("state", {})
        base = "../" + html.escape(str(entry.get("base", "")))
        href = (f"llmarchi-netron://open?candidate={html.escape(candidate)}"
                f"&phase={html.escape(phase)}&module={html.escape(mid)}"
                f"&variant=sim&group={html.escape(gid)}")
        parts.append(
            '<p class="onnx-note">ONNX（' + html.escape(phase) + " 投影）："
            f'<a class="param-link" data-module="{html.escape(gid)}/{html.escape(phase)}/{html.escape(mid)}" '
            f'href="{href}">用本地 Netron 打开</a>'
            f'<span class="mono"> onnx/{html.escape(gid)}/{html.escape(phase)}/{html.escape(mid)}.sim.onnx</span>'
            f' · {state.get("node_count_original")}→{state.get("node_count_simplified")} 节点 · '
            f'<a class="param-link" href="{base}/model.onnx">原始 onnx</a> · '
            f'<a class="param-link" href="{base}/model.sim.onnx">简化 onnx</a></p>')
    if not parts:
        return ""
    return ('<div class="onnx-block"><p class="dg-note">ONNX 为 FP32 阅读特化图（无真实权重，'
            "非量化模型数值等价）；prefill/decode 各一份，仅作辅助联查。</p>"
            + "".join(parts) + "</div>")



# ---------------------------------------------------------------------------
# 底层分层图（Archify）：入口、关系清单与返回上下文
# ---------------------------------------------------------------------------


def _viewer_relpath(path: str) -> str:
    """候选内路径 → 相对 viewer/structure.html 的链接。"""
    text = str(path or "")
    if text.startswith("viewer/"):
        return html.escape(text[len("viewer/"):])
    return html.escape(text)


def _data_relpath(path: str) -> str:
    text = str(path or "")
    return html.escape("../" + text) if text else ""


def _gate_summary(entry: dict) -> str:
    """静态说明：构建门禁随构建执行，运行证据留在证据目录。

    页面只含与模型/结构相关的稳定内容，不把某次 Chrome 是否可用写进交付字节，
    否则同一身份在不同环境重建会出现不同产物。
    """
    return "同源生成 · 校验回执见证据目录（validate / deliver / check / browser-check）"


def _archify_card(entry: dict) -> str:
    title = html.escape(str(entry.get("title", entry.get("slug", ""))))
    form = str(entry.get("form") or "normalized")
    form_label = "原始语义图" if form == "raw" else "化简后"
    links = []
    if entry.get("candidate"):
        links.append(
            f'<a class="param-link" href="{_data_relpath(entry["candidate"])}">'
            "图数据 candidate JSON</a>")
    if entry.get("relations"):
        links.append(
            f'<a class="param-link" href="{_data_relpath(entry["relations"])}">'
            f'完整关系 JSON（{entry.get("relations_count", entry.get("connections", 0))} 条）</a>')
    if entry.get("sources"):
        links.append(
            f'<a class="param-link" href="{_data_relpath(entry["sources"])}">'
            "来源映射 JSON（节点→路径/行号）</a>")
    rows = []
    for record in entry.get("links", []) or ():
        shape = html.escape(str(record.get("shape") or "—"))
        label = record.get("label") or record.get("kind") or ""
        rows.append(
            "<tr>"
            f'<td class="mono">{html.escape(str(record.get("from")))} → '
            f'{html.escape(str(record.get("to")))}</td>'
            f'<td>{html.escape(str(record.get("tensor") or ""))}</td>'
            f'<td class="mono">{shape}</td>'
            f"<td>{html.escape(str(label))}</td>"
            "</tr>")
    shown = len(rows)
    table = ""
    if rows:
        table = (
            f'<details class="fold ac-rel"><summary>关系清单（前 {shown} 条；'
            f'全部 {entry.get("relations_count", entry.get("connections", 0))} 条见 JSON）</summary>'
            '<div class="table-scroll"><table class="tensors"><thead><tr>'
            "<th>数据流</th><th>张量</th><th>全称维度</th><th>分支/状态</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>")
    return (
        '<div class="ac-card" data-archify-slug="' + html.escape(str(entry.get("slug", ""))) + '">'
        f'<div class="ac-card-title">{title}'
        f'<span class="ac-form">{html.escape(form_label)}</span></div>'
        f'<div class="ac-meta">{entry.get("nodes", 0)} 节点 · '
        f'{entry.get("relations_count", entry.get("connections", 0))} 条关系记录 · '
        f'{html.escape(_gate_summary(entry))}</div>'
        f'<div class="ac-actions"><a class="btn" '
        f'href="{_viewer_relpath(entry.get("view") or entry.get("html", ""))}">'
        '打开分层子图（可返回本层）→</a> '
        f'<a class="param-link" target="_blank" rel="noopener" '
        f'href="{_viewer_relpath(entry.get("html", ""))}">原始图新标签</a></div>'
        + ('<div class="ac-links">' + " · ".join(links) + "</div>" if links else "")
        + table
        + "</div>"
    )


def _archify_panel(entries, *, heading: str, note: str,
                   back_links=(), sub_views=()) -> str:
    entries = [entry for entry in entries or () if entry]
    if not entries and not sub_views:
        return ""
    out = ['<div class="ac-panel">', f'<h3 class="ac-head">{html.escape(heading)}</h3>',
           f'<p class="ac-note">{html.escape(note)}</p>']
    if entries:
        out.append('<div class="ac-grid">' + "".join(_archify_card(e) for e in entries) + "</div>")
    for label, sub_entries in sub_views or ():
        sub_entries = [entry for entry in sub_entries if entry]
        if not sub_entries:
            continue
        out.append(f'<h4 class="ac-sub">{html.escape(label)}</h4>')
        out.append('<div class="ac-grid">'
                   + "".join(_archify_card(e) for e in sub_entries) + "</div>")
    if back_links:
        out.append('<p class="ac-back">' + " ｜ ".join(
            f'<a class="param-link" href="{href}">{html.escape(text)}</a>'
            for text, href in back_links) + "</p>")
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# 面包屑 / 场景数据 / 脚本 / 样式
# ---------------------------------------------------------------------------


def _crumbs(items: list[tuple[str, str | None]]) -> str:
    parts = []
    for index, (label, href) in enumerate(items):
        last = index == len(items) - 1
        if href and not last:
            parts.append(f'<a href="{html.escape(href)}">{html.escape(label)}</a>')
        else:
            parts.append(f'<span aria-current="page">{html.escape(label)}</span>')
    return '<nav class="crumbs" aria-label="当前位置">' + " › ".join(parts) + "</nav>"


def _scenario_data_script(structures: list[dict]) -> str:
    first = structures[0]["structure"]["scope"]
    payload = {
        "scenarios": first.get("scenarios", []),
        "default": first.get("default_scenario", ""),
        "groups": {
            str(item["id"]): {
                "label": item.get("label", item["id"]),
                "derivations": item["structure"].get("scenario_derivations", {}),
            }
            for item in structures
        },
    }
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return f'<script id="scn-data" type="application/json">{data}</script>'


def _js_single_quoted(text: str) -> str:
    """把文案转义进 JS 单引号字符串（默认文案替换后与既有页面字节一致）。"""
    return (str(text).replace("\\", "\\\\").replace("'", "\\'")
            .replace("\r", "").replace("\n", "\\n"))


def _router_script(structures: list[dict], profile: PageProfile | None = None) -> str:
    labels = {str(item["id"]): str(item.get("label", item["id"])) for item in structures}
    labels_json = json.dumps(labels, ensure_ascii=False).replace("<", "\\u003c")
    panel_title = _js_single_quoted(
        profile.scenario_panel_title if profile else "场景派生（seqlen / kvcachelen 绑定）")
    hint = profile.scenario_panel_hint if profile else None
    panel_hint_js = (
        # 默认整段（含 <p> 标签与换行）与既有页面逐字节一致。
        '\'<p class="dg-note">结构 shape 始终保留符号；下表只是把场景取值代入固定配置与\' +\n'
        '        "「空态起步、连续正常执行」前提后的派生规模。未知项如实标注，不默认成 0。</p>" +'
        if hint is None else
        "\'<p class=\"dg-note\">" + _js_single_quoted(hint) + "</p>\' +"
    )
    return """<script>
(function () {
  "use strict";
  var GROUP_LABELS = __GROUP_LABELS__;
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
    // 既有联查锚点自带作用域：<tag>-block-<gid>-<x> / <tag>-network-<x>。
    // p 行在 Block 参数区，t/op/figop/mod 在内部算子层；按目标所属层进入。
    var tags = ["p", "t", "op", "figop", "mod"];
    for (var gid in GROUP_LABELS) {
      if (!Object.prototype.hasOwnProperty.call(GROUP_LABELS, gid)) continue;
      for (var i = 0; i < tags.length; i++) {
        var tag = tags[i], prefix = tag + "-block-" + gid;
        if (id === prefix || id.indexOf(prefix + "-") === 0) {
          return { level: tag === "p" ? "block" : "ops", gid: gid, anchor: id };
        }
      }
    }
    if (/^(p|t|op|figop)-network-/.test(id) ||
        /^net-(params|tensors|ops|sources)$/.test(id)) {
      return { level: "network", anchor: id };
    }
    // 兜底：按目标实际所在 section 判层级，新增锚点不必改路由表。
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
      try { id = decodeURIComponent(id); } catch (e) { /* 保留原始 hash */ }
      route = anchorRoute(id);
      if (route) return route;
    }
    return { level: "network" };
  }
  function reveal(id) {
    var el = document.getElementById(id);
    if (!el) return false;
    var node = el;
    while (node && node !== document.body) {
      if (node.tagName === "DETAILS") node.open = true;
      node = node.parentElement;
    }
    if (el.scrollIntoView) el.scrollIntoView({ block: "center" });
    el.classList.add("anchor-target");
    if (el.tagName && el.tagName.toLowerCase() === "g") {
      el.style.filter = "drop-shadow(0 0 6px #e8a740)";
      setTimeout(function () { el.style.filter = ""; }, 2000);
    }
    setTimeout(function () { el.classList.remove("anchor-target"); }, 2400);
    return true;
  }
  function show(route) {
    var activeId = "lv-" + route.level + (route.gid ? "-" + route.gid : "");
    sections.forEach(function (sec) {
      sec.hidden = sec.id !== activeId;
      sec.style.display = sec.id === activeId ? "block" : "none";
    });
    document.body.setAttribute("data-level", route.level);
    document.body.setAttribute("data-group", route.gid || "");
    renderCrumbs(route);
    renderScenario(route.gid);
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
    window.scrollTo(0, 0);
  }
  window.locateInDiagram = function (figId) {
    var el = document.getElementById(figId);
    if (!el) {
      var candidates = document.querySelectorAll('[id$="-' + figId + '"]');
      el = candidates.length ? candidates[candidates.length - 1] : null;
    }
    if (!el) return;
    var sec = el.closest ? el.closest("[data-lv]") : null;
    if (sec && (sec.hidden || getComputedStyle(sec).display === "none")) {
      show({ level: sec.getAttribute("data-lv"),
             gid: sec.getAttribute("data-gid") || null });
    }
    reveal(el.id);
  };
  function renderCrumbs(route) {
    var el = document.getElementById("crumbs");
    if (!el) return;
    var parts = ['<a href="#/network">整网</a>'];
    if (route.level === "network") { parts.push('<span aria-current="page">整网</span>'); }
    else {
      var gid = route.gid, label = GROUP_LABELS[gid];
      var base = "#/group/" + gid;
      if (route.level === "group") parts.push('<span aria-current="page">' + esc(label) + '</span>');
      else {
        parts.push('<a href="' + base + '">' + esc(label) + '</a>');
        if (route.level === "block") parts.push('<span aria-current="page">Block</span>');
        else {
          parts.push('<a href="' + base + '/block">Block</a>');
          parts.push('<span aria-current="page">内部算子</span>');
        }
      }
    }
    el.innerHTML = "当前位置：" + parts.join(" › ");
  }
  function renderScenario(gid) {
    var dataEl = document.getElementById("scn-data");
    var buttons = document.getElementById("scn-buttons");
    var current = document.getElementById("scn-current");
    if (!dataEl || !buttons) return;
    var data = JSON.parse(dataEl.textContent);
    var state = window.__scnState || data.default;
    window.__scnState = state;
    buttons.innerHTML = data.scenarios.map(function (s) {
      var on = s.id === state ? ' class="on" aria-pressed="true"' : ' aria-pressed="false"';
      return '<button type="button" data-scn="' + esc(s.id) + '"' + on + '>' + esc(s.label) + "</button>";
    }).join("");
    Array.prototype.forEach.call(buttons.querySelectorAll("button"), function (btn) {
      btn.addEventListener("click", function () {
        window.__scnState = btn.getAttribute("data-scn");
        renderScenario(document.body.getAttribute("data-group"));
      });
    });
    var chosen = data.scenarios.filter(function (s) { return s.id === state; })[0];
    current.textContent = chosen ? ("当前：" + chosen.label + "（" + chosen.note + "）") : "";
    Array.prototype.forEach.call(document.querySelectorAll("[data-scn-group]"), function (slot) {
      var g = slot.getAttribute("data-scn-group");
      var group = data.groups[g];
      if (!group) { slot.innerHTML = ""; return; }
      var rows = (group.derivations[state] || []).map(function (d) {
        var value = d.state === "determined" ? esc(String(d.value))
          : '<span class="unknown">未知（' + esc(d.unknown_reason || "") + "）</span>";
        return "<tr><td><b>" + esc(d.label) + '</b><br><span class="mono">' + esc(d.id) +
          "</span></td>" +
          '<td class="mono">' + esc(d.expression) + "</td><td>" + value + "</td>" +
          '<td class="dg-note">' + esc(d.note) + "</td></tr>";
      }).join("");
      slot.innerHTML = '<div class="scn-panel"><h3>__SCN_PANEL_TITLE__</h3>' +
        __SCN_PANEL_HINT_JS__
        '<div class="table-scroll"><table class="params"><thead><tr><th>派生量</th>' +
        '<th>表达式</th><th>本场景取值</th><th>前提/说明</th></tr></thead><tbody>' +
        rows + "</tbody></table></div></div>";
    });
  }
  window.addEventListener("hashchange", function () { show(parseRoute()); });
  document.addEventListener("DOMContentLoaded", function () { show(parseRoute()); });
  if (document.readyState !== "loading") show(parseRoute());
})();
</script>""".replace("__GROUP_LABELS__", labels_json).replace(
        "__SCN_PANEL_TITLE__", panel_title).replace("__SCN_PANEL_HINT_JS__", panel_hint_js)


def _sources_table(input_shas: dict[str, str] | None) -> str:
    rows = []
    for key, value in (input_shas or {}).items():
        rows.append(
            f'<tr><td class="mono">{html.escape(str(key))}</td>'
            f'<td class="mono">{html.escape(str(value))}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="2">未提供输入 SHA（构建层未传入；不伪造）</td></tr>')
    return ('<div class="table-scroll"><table class="params"><thead><tr>'
            "<th>输入（SOURCE_FILES 键）</th><th>SHA256</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _page_css(profile: PageProfile) -> str:
    match = re.search(r"<style>(.*?)</style>", _PAGE_HEAD, re.S)
    base = match.group(1) if match else ""
    base = base.replace("{{", "{").replace("}}", "}")
    base = base.replace("{diagram_css}", diagram_css())
    return base + """
.lv{display:none;max-width:1180px;margin:0 auto}
#lv-network{display:block}
.lv-title{font-size:20px;margin:18px 0 6px}
.crumbs{margin:8px 0 4px;font-size:13.5px;color:var(--muted)}
.crumbs a{color:var(--accent);text-decoration:none;border-bottom:1px dotted}
.crumbs a:hover{text-decoration:underline}
.scn-bar{margin:10px 0 2px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:13px}
.scn-title{color:var(--muted);font-weight:600}
.scn-buttons{display:inline-flex;flex-wrap:wrap;gap:4px}
.scn-buttons button{font:inherit;font-size:12.5px;border:1px solid var(--line);background:var(--card);
  color:var(--ink);border-radius:14px;padding:2px 10px;cursor:pointer}
.scn-buttons button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.scn-current{color:var(--muted);font-size:12.5px}
.scn-slot{margin:12px 0}
.scn-panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 16px}
.unknown{color:#9c0006;font-weight:700}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 14px}
.mod-chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);
  border-radius:14px;padding:3px 10px;color:var(--ink);text-decoration:none;font-size:12.5px;
  background:var(--card)}
.mod-chip:hover{border-color:var(--accent);color:var(--accent)}
.chip-count{background:var(--code);border-radius:10px;padding:0 7px;font-size:11.5px;color:var(--muted)}
.mod-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;margin:8px 0 16px}
.mod-card{border:1px solid var(--line);border-radius:9px;padding:8px 12px;background:var(--card)}
.mod-card-head{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.mod-card-ops{color:var(--muted);font-size:12px;margin-top:4px}
.strong-link{font-weight:700;color:var(--accent);text-decoration:none}
.strong-link:hover{text-decoration:underline}
.fold{border:1px solid var(--line);border-radius:10px;padding:10px 16px;margin:12px 0;
  background:var(--card)}
.fold>summary{cursor:pointer;font-weight:600}
.fold[open]>summary{margin-bottom:8px}
details.mod>summary{font-size:15px}
.anchor-target{background:rgba(232,167,64,.28)!important;outline:2px solid #e8a740;
  outline-offset:2px;transition:background .2s}
.op,tr[id],details[id]{scroll-margin-top:16px}
.onnx-block{border-left:3px solid #b08d57;background:var(--code);border-radius:6px;
  padding:6px 12px;margin:6px 0 12px}
.onnx-block .onnx-note{font-size:12.5px;color:var(--muted);margin:4px 0}
.lv-next{margin:14px 0 30px}
.btn{display:inline-block;border:1px solid var(--accent);border-radius:8px;padding:5px 14px;
  color:var(--accent);text-decoration:none;font-weight:600;background:var(--card)}
.btn:hover{background:var(--accent);color:#fff}
table.facts th{width:120px}
.np-layers{list-style:none;padding:0;margin:8px 0 18px;display:flex;flex-wrap:wrap;gap:6px}
.np-layer{margin:0}
.np-layer a{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-width:58px;padding:4px 6px;border:1px solid var(--line);border-radius:7px;
  text-decoration:none;color:var(--ink);font-size:11px;line-height:1.25;background:var(--card)}
.np-layer a:hover{border-color:var(--accent)}
.np-layer .np-layer-idx{font-weight:700;font-size:13px}
.np-layer .np-layer-type{color:var(--muted)}
.ac-panel{margin:14px 0 18px;padding:12px 14px;border:1px solid var(--line);border-radius:10px;
  background:var(--card)}
.ac-head{margin:2px 0 4px;font-size:15px}
.ac-note{margin:0 0 10px;color:var(--muted);font-size:13px}
.ac-sub{margin:14px 0 6px;font-size:13.5px;color:var(--muted)}
.ac-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}
.ac-card{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--bg)}
.ac-card-title{font-weight:700;font-size:14px;display:flex;gap:8px;align-items:center}
.ac-form{font-weight:600;font-size:11.5px;color:var(--accent);border:1px solid var(--accent);
  border-radius:10px;padding:1px 8px;white-space:nowrap}
.ac-meta{color:var(--muted);font-size:12.5px;margin:6px 0 8px}
.ac-actions{margin:0 0 6px}
.ac-links{font-size:12.5px}
.ac-rel{margin-top:8px}
.ac-back{margin:10px 0 0;font-size:13px}
{group_css}a:focus-visible,button:focus-visible,summary:focus-visible{
  outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
@media (max-width:1100px){
  .mod-cards{grid-template-columns:1fr}
  body{padding:0 10px 48px}
  .scn-bar{flex-direction:column;align-items:flex-start}
}
@media (max-width:720px){
  .np-layer a{min-width:48px;font-size:10px}
  .fold{padding:8px 10px}
  .crumbs{font-size:12.5px}
}
""".replace("{group_css}", profile.group_css)


_PAGE_FOOT = """<footer class="page">
分层结构浏览页由 structure_reader 从本地 Transformers 检出与根 config.json 静态生成；
不加载权重、不执行模型。整网→层组→Block→算子的层级由页面脚本按 hash 路由切换，
每层都可用普通链接进入/返回；静态 HTML 可离线打开。
</footer>

<script>
(function() {
  "use strict";
  // 源码快照面板开关（算子卡“源码”按钮）。图内定位 locateInDiagram 在路由脚本内
  // 定义，以便跨层级时先切换层级、展开 details 再定位。
  window.toggleSource = function(id) {
    var el = document.getElementById(id);
    if (el) el.style.display = (el.style.display === "none") ? "block" : "none";
  };
})();
</script>
</body>
</html>
"""
