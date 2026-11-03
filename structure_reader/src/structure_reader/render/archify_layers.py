"""产品分层 Archify 图：整网 → 层组 → Block/Decoder → Attention/KV → MoE/mHC。

本模块把**同一份产品事实文档**（``hf_network_overview.v1`` 的 ``path`` 与
``hf_layer_structure.v1`` 的 ``block``）程序化投影成 Archify 图，供分层浏览页在
各层级给出可进入的底层图：

- ``network``：整网 token→logits 主路径（14 个网络级算子，层循环不铺开）；
- ``group``：43 层 → 本层组真实层 → 代表层 → Block 模板（层组归属与进入路径）；
- ``decoder``：Block 的模块级数据流（module_groups 分组，含 mHC post/comb → 残差、
  缓存读写与分支汇合）；
- ``moe``：MoE 内部 Linear/路由/分派/单专家模板/共享专家/汇合（算子级）；
- ``mhc``：attn_hc / ffn_hc 的 pre/post/comb/collapse 与两段残差（算子级）。

Attention/KV 两层复用 T12 的语义图（``render.archify_preview.build_view_model`` /
``build_candidate``），本模块只负责把它们与产品层级、路径、身份绑定。

所有节点/边都由文档中的 ``operators``/``tensors``/``source_refs`` 生成，不手抄
模型结构；形状经 ``render.display_names`` 展开为 config/modeling 全称。本模块不
实例化模型、不读权重；Archify CLI 通过 ``ARCHIFY_CLI``/配置/常规安装位置发现，
路径与工具身份写入 manifest，不硬编码个人路径。
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from structure_reader.render.archify_preview import (
    ArchifyTool,
    PreviewError,
    _canonical_json_bytes,
    build_candidate,
    build_view_model,
    classify_browser_failure,
    discover_archify,
    finalize_candidate,
    package_closure,
    tool_identity,
)
from structure_reader.render.archify_profiles import (
    GraphRepository,
    ModelGraphProfile,
    SegmentSpec,
    graph_profile_for,
)
from structure_reader.render.diagram import module_groups
from structure_reader.render.display_names import expand_display_text, full_name, shape_text

__all__ = [
    "SCHEMA",
    "BUILDER_VERSION",
    "LevelNode",
    "LevelEdge",
    "LevelGraph",
    "ArchifyLayersError",
    "build_network_graph",
    "build_group_graph",
    "build_decoder_graph",
    "build_moe_graph",
    "build_mhc_graph",
    "build_level_candidate",
    "layout_level",
    "validate_graph_shapes",
    "generate_archify_layers",
    "archify_identity",
    "build_network_graph_for",
    "build_group_graph_for",
    "build_decoder_graph_for",
    "build_segment_graph",
    "segment_requests",
    "normalization_audit",
]

SCHEMA = "hf-archify-layers.v1"
BUILDER_VERSION = "t14-archify-layers/5"

# D01 公开图（按层级）：节点按文字预算加宽、主图完整显示，不用省略号换取紧凑。
# 整网/端点/层组是第一轮已启用；返修把 Decoder、mHC、MoE 三个公开层级一并纳入，
# 因为 AC14 适用于本票全部公开范围，完整文字表不能替代主图。
_COMPLETE_TEXT_LEVELS = (
    "network", "embedding", "head", "group", "decoder", "mhc", "moe",
    "cache", "compressor", "indexer", "q", "o",
    "moe", "moe-relation", "moe-route", "moe-expert", "moe-shared", "moe-combine",
)

_REPO_PREFIX = "hflib/transformers/"
_MAIN_X = 258
_MAIN_W = 210
_COL_PITCH = 238
_NODE_H = 84
_STEP_Y = _NODE_H + 56
_Y0 = 70
_LEFT_X = 20
_SIDE_W = 180
_SIDE_H = 72
_MARGIN_X = 60
_MARGIN_Y = 70


@dataclass(frozen=True)
class LevelSizing:
    """分层图列几何。默认值与既有候选逐字节一致；``complete`` 模式按文字预算加宽。

    节点文字（label/sublabel/tag）在 Archify 中是单行、不换行；产品侧要么按预算
    截断（会丢字），要么给足宽度让渲染器按首选字号显示全文。D01 的整网与端点图
    使用 ``complete`` 几何，避免用省略号换取紧凑。
    """

    main_x: float = _MAIN_X
    main_w: float = _MAIN_W
    col_pitch: float = _COL_PITCH
    node_h: float = _NODE_H
    step_y: float = _STEP_Y
    side_w: float = _SIDE_W
    side_h: float = _SIDE_H


def _text_budget(text: str, font_px: float, padding: float) -> float:
    """完整显示一段文字所需的节点宽度（与 ``_fit`` 的预算口径一致）。"""
    return _text_width(text, font_px) + padding


def _complete_sizing(graph: "LevelGraph") -> LevelSizing:
    """按全部节点文字计算列宽：主链/边界节点决定 main_w，旁枝决定 side_w。

    宽度上限防止异常长文案把画布无限放大；超出上限的极长文本仍由 ``_fit``
    兜底并在候选里保留完整文本（页面详情与关系 JSON 另见）。
    """
    main_need = max(
        max(
            _text_budget(node.label, 13.5, 26.0),
            _text_budget(node.sublabel, 10.5, 22.0),
            _MAIN_W,
        )
        for node in graph.nodes
        if node.role not in ("aux", "state", "branch")
    )
    side_nodes = [
        node for node in graph.nodes if node.role in ("aux", "state", "branch")
    ]
    side_need = max(
        max(
            _text_budget(node.label, 13.5, 26.0),
            _text_budget(node.sublabel, 10.5, 22.0),
            _SIDE_W,
        )
        for node in side_nodes
    ) if side_nodes else _SIDE_W
    # 宽度取偶数整数：节点中心 x = main_x + col*pitch + w/2 保持整数，
    # 正交路由的端点坐标才与渲染器算出的中心完全一致（差 0.1px 也会被判不垂直）。
    # 侧栏在 Decoder 层承载缓存/边界张量，sublabel 有长符号 shape；上限放宽到
    # 760，避免为了紧凑把真实张量名/维度截断。主链上限 520 已高于本库最长标签。
    wide_d03 = graph.level in ("cache", "compressor", "indexer")
    if wide_d03:
        main_need *= 1.4
        side_need *= 1.4
    main_w = float(min(int(main_need + 1) // 2 * 2, 2000 if wide_d03 else 520))
    side_w = float(min(int(side_need + 1) // 2 * 2, 2000 if wide_d03 else 760))
    # 侧栏到主链的净距要容纳侧栏连线标签（最宽约 48px，标签中心在 lane+26）。
    # 40px 在侧栏加宽后会让“辅助输入/段输入”压到主链节点，取 64px。
    tight = graph.level == "decoder"
    main_x = float(max(_MAIN_X, int(_LEFT_X + side_w + (40 if tight else 64))))
    return LevelSizing(
        main_x=main_x,
        main_w=main_w,
        col_pitch=main_w + (12.0 if tight else 28.0),
        node_h=_NODE_H,
        step_y=_NODE_H + 56.0 + (80.0 if graph.level == "o" else 0.0),
        side_w=side_w,
        side_h=_SIDE_H,
    )


class ArchifyLayersError(ValueError):
    """分层图生成/排序/交付失败（具名拒绝）。"""


# ---------------------------------------------------------------------------
# 图模型
# ---------------------------------------------------------------------------


@dataclass
class LevelNode:
    node_id: str
    label: str
    sublabel: str
    tag: str
    component_type: str
    sources: tuple[dict[str, Any], ...]
    role: str  # boundary_in | main | aux | state | branch | boundary_out
    anchor: str | None = None
    details: tuple[str, ...] = ()
    op_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "sublabel": self.sublabel,
            "tag": self.tag,
            "component_type": self.component_type,
            "sources": [dict(item) for item in self.sources],
            "role": self.role,
            "anchor": self.anchor,
            "details": list(self.details),
            "op_ids": list(self.op_ids),
        }


@dataclass
class LevelEdge:
    src: str
    dst: str
    tensor_id: str
    tensor_label: str
    shape: str
    kind: str  # data | state | aux
    label: str = ""
    condition: str | None = None
    # 同一模块对之间的全部真实张量（图上合并为一条关系，身份逐条保留）。
    members: tuple[dict[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "from": self.src,
            "to": self.dst,
            "tensor_id": self.tensor_id,
            "tensor_label": self.tensor_label,
            "shape": self.shape,
            "kind": self.kind,
            "label": self.label,
            "members": [dict(item) for item in self.members],
            "condition": self.condition,
        }


@dataclass
class LevelGraph:
    key: str
    level: str
    title: str
    subtitle: str
    nodes: list[LevelNode] = field(default_factory=list)
    edges: list[LevelEdge] = field(default_factory=list)
    notes: tuple[str, ...] = ()
    parameter_items: tuple[dict[str, Any], ...] = ()
    form: str = "normalized"
    variant: str | None = None
    group: str | None = None
    expansions: tuple[dict[str, Any], ...] = ()

    def node(self, node_id: str) -> LevelNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise ArchifyLayersError(f"图 {self.key} 没有节点 {node_id!r}")

    def relations(self) -> list[dict[str, Any]]:
        records = []
        for edge in self.edges:
            record = edge.to_json()
            # 关系 JSON 是产品展示面：tensor 身份表达式同样展开为全称；内部
            # tensor_id 与来源仍保留，原始源码不丢失。
            record["tensor_label"] = expand_display_text(
                str(record.get("tensor_label") or ""))
            record["label"] = expand_display_text(str(record.get("label") or ""))
            records.append(record)
        return records


# ---------------------------------------------------------------------------
# 文档读取助手
# ---------------------------------------------------------------------------


def _tensor_index(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(t.get("id")): t for t in document.get("tensors", [])}


def _op_index(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(op.get("id")): op for op in document.get("operators", [])}


def _shape_of(tensor: Mapping[str, Any] | None) -> str:
    if not tensor:
        return ""
    return shape_text(tensor.get("shape", []))


def _tensor_name(tensor: Mapping[str, Any] | None, fallback: str) -> str:
    if not tensor:
        return fallback
    return str(tensor.get("name") or fallback)


def _source_locators(locator: Any) -> list[tuple[int, int | None]]:
    """源码 locator 可能是 ``a``、``a-b`` 或 ``a/b-c,d``：逐段解析并保留。"""
    parts: list[tuple[int, int | None]] = []
    for chunk in re.split(r"[/,，、;；]", str(locator or "")):
        match = re.match(r"\s*(\d+)(?:\s*[-–]\s*(\d+))?", chunk)
        if match is None:
            continue
        parts.append(
            (int(match.group(1)), int(match.group(2)) if match.group(2) else None)
        )
    return parts


def _sources(
    op: Mapping[str, Any], *, limit: int = 3, prefix: str = _REPO_PREFIX
) -> tuple[dict[str, Any], ...]:
    """产品 source_refs → Archify 来源引用（检出根内的仓库相对路径）。

    ``prefix`` 由模型声明提供：HF transformers 为 ``hflib/transformers/``，
    HiDream 官方实现为 ``models/HiDream/HiDream-O1-Image/``；不把根 HEAD 或
    其它模型的前缀冒充当前来源。
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int | None]] = set()
    for ref in op.get("source_refs", []) or ():
        if ref.get("kind") != "source":
            continue
        path = str(ref.get("path", ""))
        if not path.startswith(prefix):
            continue
        path = path[len(prefix):]
        for line, end in _source_locators(ref.get("locator")):
            key = (path, line, end)
            if key in seen:
                continue
            seen.add(key)
            item: dict[str, Any] = {"path": path, "line": line}
            if end is not None:
                item["end_line"] = end
            out.append(item)
            if len(out) >= limit:
                return tuple(out)
    return tuple(out)


def _module_tail(module_path: str) -> str:
    text = str(module_path or "")
    text = text.replace("（专家模板）", "").replace("（", ".").replace("）", "")
    return text.rsplit(".", 1)[-1] if text else ""


def _node_tag(module_path: Any, fallback: str = "") -> str:
    """组件小标签：取模块路径中最具辨识度的短段，避免长实现路径撑破卡片。"""
    text = str(module_path or "").replace("（", ".").replace("）", "")
    parts = [part.strip() for part in re.split(r"[./]", text) if part.strip()]
    if not parts:
        return fallback
    tag = max(parts, key=len)
    # 标签区域约 200px；按 6px 可读下限截断，完整模块路径仍在节点详情里。
    return _fit(tag, 190, 6) or fallback


def _op_label(op: Mapping[str, Any]) -> str:
    return str(op.get("name") or op.get("id"))


def _op_sublabel(op: Mapping[str, Any]) -> str:
    tail = _module_tail(op.get("module_path", ""))
    kind = str(op.get("op") or "")
    if tail and kind:
        return f"{tail} · {kind}"
    return tail or kind


def _parameter_items(document: Mapping[str, Any], *, limit: int = 24) -> tuple[dict[str, Any], ...]:
    items = []
    for param in document.get("parameters", [])[:limit]:
        items.append(
            {
                "id": param.get("id"),
                "symbol": param.get("symbol"),
                "meaning": param.get("meaning"),
                "value": param.get("value"),
                "expression": param.get("expression"),
            }
        )
    return tuple(items)


def _mhc_parameter_items(block: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """mHC 图只保留本图必要参数，避免 block 通用 Q/KV/MoE 前缀占满参数卡。"""
    wanted = {"B", "seqlen", "H", "Nhc", "Sit", "Eps"}
    return tuple(
        item for item in _parameter_items(block, limit=64)
        if str(item.get("id")) in wanted
    )


def _boundary_tensor_ids(document: Mapping[str, Any]) -> dict[str, str]:
    """块边界/缓存张量 → (tensor_id → role)。"""
    roles: dict[str, str] = {}
    for tensor in document.get("tensors", []):
        kind = str(tensor.get("kind"))
        if kind == "boundary":
            roles[str(tensor["id"])] = "boundary"
        elif kind == "cache":
            roles[str(tensor["id"])] = "state"
    return roles


# ---------------------------------------------------------------------------
# 整网层
# ---------------------------------------------------------------------------


def build_network_graph(network: Mapping[str, Any]) -> LevelGraph:
    """整网：14 个网络级算子按真实张量引用单向排布，层循环不铺开。"""
    path = network.get("path") or {}
    ops = path.get("operators") or []
    if not ops:
        raise ArchifyLayersError("整网文档缺少 path.operators")
    tensors = _tensor_index(path)
    scope = network.get("scope") or {}
    producer = {
        str(tensor_id): str(op["id"])
        for op in ops
        for tensor_id in op.get("outputs", [])
    }
    graph = LevelGraph(
        key="network",
        level="network",
        title="DeepSeek V4 · 整网主路径（token → logits）",
        subtitle=(
            f"入口 {scope.get('entry_class', '')} · {scope.get('layer_count', '')} 层 · "
            f"hidden_size={scope.get('hidden_size', '')} · 层循环不铺开"
        ),
        notes=tuple(path.get("coverage_notes", [])),
        parameter_items=_parameter_items(path),
        form="normalized",
    )
    input_ids = tensors.get("input_ids")
    graph.nodes.append(
        LevelNode(
            node_id="n-input",
            label="input_ids / inputs_embeds",
            sublabel=_shape_of(input_ids),
            tag="输入张量",
            component_type="external",
            sources=_sources(ops[0]),
            role="boundary_in",
        )
    )
    # 网络入口的预计算（position ids / mask / 两组 RoPE）不是 token→logits
    # 主链上的算子，而是层循环消费的辅助输入。把它们放在左侧旁枝、主链保持
    # 单列：既保留全部真实节点/边/来源，又避免四列并排把主链推到最右列、
    # 在 1440 宽度下被 Reader 压到最小宽度导致连线文字过小。
    # mask 构建仍留在主链列（rank 2 的第二列），它的输入/输出 shape
    # 保留在真实连线上；position/RoPE 是纯辅助预计算，放左侧旁枝。
    side_ops = {
        "net_position_ids", "net_rope_main", "net_rope_compress",
    }
    for op in ops:
        is_side = str(op["id"]) in side_ops
        side_anchor = {
            "net_position_ids": "n-input",
            "net_rope_main": "n-net_embed",
            "net_rope_compress": "n-net_embed",
        }.get(str(op["id"]))
        details = [str(op.get("note", ""))]
        if is_side:
            details.append(
                "布局：整网入口预计算，作为辅助输入进入层循环；"
                "真实数据边与来源保留，不并入主链列。"
            )
        graph.nodes.append(
            LevelNode(
                node_id=f"n-{op['id']}",
                label=_op_label(op),
                sublabel=_op_sublabel(op),
                tag=_node_tag(op.get("module_path", ""), "整网"),
                component_type="backend",
                sources=_sources(op),
                role="branch" if is_side else "main",
                anchor=side_anchor if is_side else None,
                details=tuple(details),
                op_ids=(str(op["id"]),),
            )
        )
    graph.nodes.append(
        LevelNode(
            node_id="n-logits",
            label="logits",
            sublabel=_shape_of(tensors.get("logits")),
            tag="输出张量",
            component_type="external",
            sources=_sources(ops[-1]),
            role="boundary_out",
        )
    )
    loop = next((op for op in ops if op["id"] == "net_layer_loop"), None)
    if loop is not None:
        for node_id, label, shape in (
            ("n-cache-in", "past_key_values（调用前）", _shape_of(tensors.get("past_key_values"))),
            ("n-cache-out", "past_key_values（调用后）", _shape_of(tensors.get("past_key_values_post"))),
        ):
            graph.nodes.append(
                LevelNode(
                    node_id=node_id,
                    label=label,
                    sublabel=shape,
                    tag="跨调用状态",
                    component_type="external",
                    sources=_sources(loop),
                    role="state",
                    anchor=f"n-{loop['id']}",
                )
            )
    pos = next((op for op in ops if op["id"] == "net_position_ids"), None)
    if pos is not None:
        graph.nodes.append(
            LevelNode(
                node_id="n-aux-attention-mask",
                label="attention_mask（可选输入）",
                sublabel=_shape_of(tensors.get("attention_mask")),
                tag="辅助输入（同一张量）",
                component_type="external",
                sources=_sources(next(op for op in ops if op["id"] == "net_mask_build")),
                role="aux",
                anchor="n-input",
            )
        )

    # rank 2 的列顺序：hc_expand 先于 mask，使层循环继承主链列（col0），
    # 而不是被 mask 列挤到右侧；mask 作为 rank 2 的第二列分支。
    expand_node = next(
        (node for node in graph.nodes if node.node_id == "n-net_hc_expand"),
        None,
    )
    if expand_node is not None:
        graph.nodes.remove(expand_node)
        mask_index = next(
            index for index, node in enumerate(graph.nodes)
            if node.node_id == "n-net_mask_build"
        )
        graph.nodes.insert(mask_index, expand_node)

    # 辅助输入节点放在最前面（input 之后），使侧栏从上到下按消费顺序排列，
    # 所有数据边在布局中向下，而不是从侧栏向上回指层循环。
    aux_node = next(
        (node for node in graph.nodes if node.node_id == "n-aux-attention-mask"),
        None,
    )
    if aux_node is not None:
        graph.nodes.remove(aux_node)
        input_index = next(
            index for index, node in enumerate(graph.nodes)
            if node.node_id == "n-input"
        )
        graph.nodes.insert(input_index + 1, aux_node)

    for op in ops:
        for tensor_id in op.get("inputs", []):
            src = producer.get(str(tensor_id))
            if src is None:
                continue
            tensor = tensors.get(str(tensor_id))
            graph.edges.append(
                LevelEdge(
                    src=f"n-{src}",
                    dst=f"n-{op['id']}",
                    tensor_id=str(tensor_id),
                    tensor_label=_tensor_name(tensor, str(tensor_id)),
                    shape=_shape_of(tensor),
                    kind="data",
                )
            )
    first = ops[0]
    for tensor_id in first.get("inputs", []):
        if str(tensor_id) in producer:
            continue
        tensor = tensors.get(str(tensor_id))
        graph.edges.append(
            LevelEdge(
                src="n-input",
                dst=f"n-{first['id']}",
                tensor_id=str(tensor_id),
                tensor_label=_tensor_name(tensor, str(tensor_id)),
                shape=_shape_of(tensor),
                kind="data",
            )
        )
    last = ops[-1]
    for tensor_id in last.get("outputs", []):
        if str(tensor_id) == "logits":
            graph.edges.append(
                LevelEdge(
                    src=f"n-{last['id']}",
                    dst="n-logits",
                    tensor_id=str(tensor_id),
                    tensor_label=_tensor_name(tensors.get(str(tensor_id)), str(tensor_id)),
                    shape=_shape_of(tensors.get(str(tensor_id))),
                    kind="data",
                )
            )
    if loop is not None:
        graph.edges.append(
            LevelEdge(
                src="n-cache-in",
                dst=f"n-{loop['id']}",
                tensor_id="past_key_values",
                tensor_label="past_key_values（调用前状态）",
                shape=_shape_of(tensors.get("past_key_values")),
                kind="state",
                label="状态读入",
            )
        )
        graph.edges.append(
            LevelEdge(
                src=f"n-{loop['id']}",
                dst="n-cache-out",
                tensor_id="past_key_values_post",
                tensor_label="past_key_values（调用后状态）",
                shape=_shape_of(tensors.get("past_key_values_post")),
                kind="state",
                label="状态写出",
            )
        )
    graph.edges.append(
        LevelEdge(
            src="n-aux-attention-mask",
            dst="n-net_mask_build",
            tensor_id="attention_mask",
            tensor_label="attention_mask",
            shape=_shape_of(tensors.get("attention_mask")),
            kind="aux",
            label="mask 输入",
        )
    )
    return graph


# ---------------------------------------------------------------------------
# 整网层简单端点的逐算子展开（D01：Embedding / 输出端）
# ---------------------------------------------------------------------------

# 端点图的算子只从整网同源事实（path.operators/tensors）里选取，不重写一份结构。
_ENDPOINT_SPECS: dict[str, dict[str, Any]] = {
    "embedding": {
        "title": "Embedding · input_ids → embed_tokens → mHC 多流（逐算子）",
        "subtitle": (
            "DeepseekV4Model.forward · Gather 查表 + unsqueeze/expand/contiguous；"
            "seqlen 保持符号，权重形状就近列出"
        ),
        "op_ids": ("net_embed", "net_hc_expand"),
        "external_labels": {
            "input_ids": "input_ids（调用入口 token id）",
        },
        "output_labels": {
            "hidden_streams": "hidden_streams（送入层循环）",
        },
        "output_tensors": ("hidden_streams",),
        "param_ids": ("Vocab", "H", "Nhc", "B", "seqlen"),
    },
    "head": {
        "title": "HyperHead → Final Norm → logits 切片 → lm_head（逐算子）",
        "subtitle": (
            "DeepseekV4HyperHead.forward + DeepseekV4ForCausalLM.forward · "
            "flatten/float → unweighted RMSNorm → F.linear → sigmoid 门控 → 加权求和 → "
            "Final RMSNorm → logits_to_keep 切片 → lm_head"
        ),
        "op_ids": (
            "net_hc_head_norm",
            "net_hc_head_project",
            "net_hc_head_pre",
            "net_hc_head_collapse",
            "net_final_norm",
            "net_logits_slice",
            "net_lm_head",
        ),
        "external_labels": {
            "streams_after_layers": "层循环输出 hidden_states（全部层之后）",
        },
        "output_labels": {
            "logits": "logits（未归一化分数）",
        },
        "output_tensors": ("logits",),
        "param_ids": ("Lk", "Ssel", "Nhc", "H", "Vocab", "EpsN", "Eps", "B", "seqlen"),
    },
}


def _endpoint_parameter_items(
    path: Mapping[str, Any], param_ids: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    wanted = {str(item) for item in param_ids}
    items = []
    for param in path.get("parameters", []):
        if str(param.get("id")) not in wanted:
            continue
        items.append(
            {
                "id": param.get("id"),
                "symbol": param.get("symbol"),
                "meaning": param.get("meaning"),
                "value": param.get("value"),
                "expression": param.get("expression"),
            }
        )
    return tuple(items)


def build_endpoint_graph(network: Mapping[str, Any], kind: str) -> LevelGraph:
    """整网简单端点（embedding / head）→ 逐算子图。

    只取整网文档里已有的 operator/tensor 事实：节点是具名算子与就近权重/输入，
    边是真实张量引用（含全称符号 shape）。不引入整网图没有的算子，也不把
    HyperHead 这类复合模块画成单节点。

    依赖与来源口径：
    - 每个 (权重张量, 消费算子) 都产生一条 ``aux`` 边，不因节点已存在而丢边；
    - 端点外张量优先使用其真实生产算子的来源；没有生产算子时使用消费算子
      （权重即其定义/消费位置），并在节点详情里写明“生产/消费算子”。
    """
    if kind not in _ENDPOINT_SPECS:
        raise ArchifyLayersError(f"未知端点图 {kind!r}")
    spec = _ENDPOINT_SPECS[kind]
    path = network.get("path") or {}
    ops = path.get("operators") or []
    if not ops:
        raise ArchifyLayersError("整网文档缺少 path.operators")
    by_id = {str(op.get("id")): op for op in ops}
    missing = [op_id for op_id in spec["op_ids"] if op_id not in by_id]
    if missing:
        raise ArchifyLayersError(
            f"端点图 {kind!r} 需要的整网算子缺失: {', '.join(missing)}"
        )
    selected = [by_id[op_id] for op_id in spec["op_ids"]]
    selected_ids = {str(op["id"]) for op in selected}
    tensors = _tensor_index(path)
    producer_all = {
        str(tensor_id): str(op["id"])
        for op in ops
        for tensor_id in op.get("outputs", [])
    }
    producer = {
        tensor_id: op_id
        for tensor_id, op_id in producer_all.items()
        if op_id in selected_ids
    }
    graph = LevelGraph(
        key=kind,
        level=kind,
        title=spec["title"],
        subtitle=spec["subtitle"],
        notes=tuple(path.get("coverage_notes", [])[:3]),
        parameter_items=_endpoint_parameter_items(path, spec["param_ids"]),
        form="normalized",
    )
    emitted: dict[str, str] = {}

    def _op_ref(op: Mapping[str, Any]) -> str:
        """来源用途说明：模块/算子名 + 稳定 op id，便于在整网文档里回溯。"""
        name = op.get("name", op.get("id", ""))
        return f"{op.get('module_path', '')}.{name}（{op.get('id', '')}）"

    def _ensure_external(
        tensor_id: str,
        *,
        consumer_op: Mapping[str, Any] | None = None,
        anchor: str | None = None,
    ) -> str:
        """端点外的数据/权重张量 → 具名边界或就近参数节点（不静默丢弃）。"""
        if tensor_id in emitted:
            return emitted[tensor_id]
        tensor = tensors.get(tensor_id)
        kind_of = str(tensor.get("kind") if tensor else "")
        node_id = f"e-{tensor_id}"
        producer_id = producer_all.get(tensor_id)
        producer_op = by_id.get(producer_id) if producer_id else None
        if kind_of == "weight":
            label = spec["external_labels"].get(tensor_id) or str(
                (tensor or {}).get("name") or tensor_id
            )
            details = []
            if consumer_op is not None:
                details.append(f"消费算子：{_op_ref(consumer_op)}（权重就近参数）")
            sources = _sources(consumer_op) if consumer_op is not None else ()
            graph.nodes.append(
                LevelNode(
                    node_id=node_id,
                    label=label,
                    sublabel=_shape_of(tensor),
                    tag="权重（就近参数）",
                    component_type="external",
                    sources=sources,
                    role="aux",
                    anchor=anchor,
                    details=tuple(details),
                )
            )
        elif producer_op is not None:
            label = spec["external_labels"].get(tensor_id) or str(
                (tensor or {}).get("name") or tensor_id
            )
            graph.nodes.append(
                LevelNode(
                    node_id=node_id,
                    label=label,
                    sublabel=_shape_of(tensor),
                    tag="上游张量（端点外）",
                    component_type="external",
                    sources=_sources(producer_op),
                    role="boundary_in",
                    details=(f"生产算子：{_op_ref(producer_op)}",),
                )
            )
        else:
            label = spec["external_labels"].get(tensor_id) or str(
                (tensor or {}).get("name") or tensor_id
            )
            details = []
            if consumer_op is not None:
                details.append(f"消费算子：{_op_ref(consumer_op)}")
            graph.nodes.append(
                LevelNode(
                    node_id=node_id,
                    label=label,
                    sublabel=_shape_of(tensor),
                    tag="输入张量" if kind_of else "入口张量",
                    component_type="external",
                    sources=_sources(consumer_op) if consumer_op is not None else (),
                    role="boundary_in",
                    details=tuple(details),
                )
            )
        emitted[tensor_id] = node_id
        return node_id

    edge_keys: set[tuple[str, str, str]] = set()

    def _add_edge(
        src: str,
        dst: str,
        tensor_id: str,
        *,
        kind: str,
        label: str = "",
    ) -> None:
        key = (src, dst, tensor_id)
        if key in edge_keys:
            return
        edge_keys.add(key)
        tensor = tensors.get(tensor_id)
        graph.edges.append(
            LevelEdge(
                src=src,
                dst=dst,
                tensor_id=str(tensor_id),
                tensor_label=_tensor_name(tensor, str(tensor_id)),
                shape=_shape_of(tensor),
                kind=kind,
                label=label,
            )
        )

    # 端点输入：主输入（+ 任何被选中算子消费、但不由选中算子生产的非权重张量）。
    for op in selected:
        for tensor_id in op.get("inputs", []):
            if str(tensor_id) in producer:
                continue
            tensor = tensors.get(str(tensor_id))
            if tensor is not None and str(tensor.get("kind")) == "weight":
                continue  # 权重在算子节点后按 (权重, 消费算子) 就近展开
            _ensure_external(str(tensor_id), consumer_op=op)

    for op in selected:
        graph.nodes.append(
            LevelNode(
                node_id=f"e-{op['id']}",
                label=_op_label(op),
                sublabel=_op_sublabel(op),
                tag=_node_tag(op.get("module_path", ""), kind),
                component_type="backend",
                sources=_sources(op),
                role="main",
                details=(str(op.get("note", "")),),
                op_ids=(str(op["id"]),),
            )
        )
        for tensor_id in op.get("weights", []) or ():
            _ensure_external(str(tensor_id), consumer_op=op, anchor=f"e-{op['id']}")
        for tensor_id in op.get("inputs", []):
            tensor = tensors.get(str(tensor_id))
            if tensor is not None and str(tensor.get("kind")) == "weight":
                _ensure_external(
                    str(tensor_id), consumer_op=op, anchor=f"e-{op['id']}"
                )

    # 端点输出：被端点外消费者使用、或作为本端点终点的张量。
    for tensor_id in spec["output_tensors"]:
        tensor = tensors.get(tensor_id)
        if tensor is None:
            raise ArchifyLayersError(
                f"端点图 {kind!r} 的输出张量 {tensor_id!r} 不在整网文档里"
            )
        producer_op = by_id.get(producer_all.get(tensor_id, ""))
        node_id = f"out-{tensor_id}"
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=spec["output_labels"].get(tensor_id)
                or str(tensor.get("name") or tensor_id),
                sublabel=_shape_of(tensor),
                tag="输出张量",
                component_type="external",
                sources=_sources(producer_op) if producer_op is not None else (),
                role="boundary_out",
                details=(
                    (f"生产算子：{_op_ref(producer_op)}",)
                    if producer_op is not None else ()
                ),
            )
        )

    for op in selected:
        for tensor_id in op.get("inputs", []):
            tensor = tensors.get(str(tensor_id))
            if tensor is not None and str(tensor.get("kind")) == "weight":
                aux_id = _ensure_external(
                    str(tensor_id), consumer_op=op, anchor=f"e-{op['id']}"
                )
                _add_edge(aux_id, f"e-{op['id']}", str(tensor_id),
                          kind="aux", label="权重")
                continue
            src = producer.get(str(tensor_id))
            if src is None:
                external = _ensure_external(str(tensor_id), consumer_op=op)
                _add_edge(external, f"e-{op['id']}", str(tensor_id), kind="data")
                continue
            _add_edge(f"e-{src}", f"e-{op['id']}", str(tensor_id), kind="data")
        for tensor_id in op.get("weights", []) or ():
            aux_id = _ensure_external(
                str(tensor_id), consumer_op=op, anchor=f"e-{op['id']}"
            )
            _add_edge(aux_id, f"e-{op['id']}", str(tensor_id),
                      kind="aux", label="权重")

    for tensor_id in spec["output_tensors"]:
        src = producer.get(tensor_id)
        if src is not None:
            _add_edge(f"e-{src}", f"out-{tensor_id}", tensor_id, kind="data")
    return graph


# ---------------------------------------------------------------------------
# 层组层
# ---------------------------------------------------------------------------


def build_group_graph(
    network: Mapping[str, Any],
    group: Mapping[str, Any],
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
) -> LevelGraph:
    """层组：43 层循环 → 本组真实层 → 代表层 → Block 模板（不铺开全网）。"""
    scope = network.get("scope") or {}
    gid = str(group["id"])
    indices = [int(i) for i in group.get("layer_indices", [])]
    rep = group.get("representative_layer_index")
    op_count = len(block.get("operators", []))
    graph = LevelGraph(
        key=f"group-{gid}",
        level="group",
        title=f"DeepSeek V4 · 层组 {group.get('label', gid)}（{len(indices)} 层）",
        subtitle=(
            f"{group.get('attention_type', '')} + {group.get('moe_type', '')} · "
            f"代表层 {rep} · 同组共用一份 Block 模板，模板复用不代表共享权重"
        ),
        notes=tuple(block.get("scope_notes", [])),
        parameter_items=_parameter_items(block, limit=12),
        variant=gid,
        group=gid,
    )
    graph.nodes.extend(
        [
            LevelNode(
                node_id="g-loop",
                label=f"{scope.get('layer_count', '')} 层循环",
                sublabel="真实层序见整网层条；本图只展开当前层组",
                tag="整网上下文",
                component_type="external",
                sources=(),
                role="boundary_in",
            ),
            LevelNode(
                node_id="g-group",
                label=f"本组 {len(indices)} 层",
                # 完整文字优先：列出全部真实层索引，不压成省略号预览；
                # 节点按文字预算自动加宽（complete_text 层级）。
                sublabel="、".join(str(i) for i in indices),
                tag="层组归属",
                component_type="backend",
                sources=_group_sources(block),
                role="main",
                details=(
                    f"真实层索引：{', '.join(str(i) for i in indices)}",
                    f"attention_type={group.get('attention_type', '')}；"
                    f"moe_type={group.get('moe_type', '')}",
                ),
            ),
            LevelNode(
                node_id="g-rep",
                label=f"代表层 {rep}",
                sublabel=f"DeepseekV4DecoderLayer · 实例 model.layers.{rep}",
                tag="代表层",
                component_type="backend",
                sources=_group_sources(block),
                role="main",
                details=(str(block.get("title", "")),),
            ),
            LevelNode(
                node_id="g-block",
                label=f"Block 模板（{len(modules)} 模块 / {op_count} 算子）",
                sublabel=(
                    f"{group.get('attention_type', '')} + {group.get('moe_type', '')} · "
                    "进入模块关系层看到真实张量数据流"
                ),
                tag="Block 模板",
                component_type="backend",
                sources=_group_sources(block),
                role="main",
                details=tuple(
                    f"{module['title']}（{len(module['ops'])} 算子）" for module in modules
                ),
            ),
        ]
    )
    graph.nodes.append(
        LevelNode(
            node_id="g-other-groups",
            label="其余层组",
            sublabel="、".join(
                f"{item['id']}({item['count']})"
                for item in network.get("groups", [])
                if item["id"] != gid
            ),
            tag="整网上下文",
            component_type="external",
            sources=(),
            role="aux",
            anchor="g-group",
        )
    )
    for src, dst, label in (
        ("g-loop", "g-group", f"{scope.get('layer_count', '')} 层中的本组"),
        ("g-group", "g-rep", "取代表层生成模板"),
        ("g-rep", "g-block", "模块级模板"),
    ):
        tensor = next(
            (
                t
                for t in block.get("tensors", [])
                if t.get("id")
                == ("hidden_streams_in" if dst == "g-block" else "attn_normed")
            ),
            None,
        )
        graph.edges.append(
            LevelEdge(
                src=src,
                dst=dst,
                tensor_id=str(tensor["id"]) if tensor else "layer",
                tensor_label=_tensor_name(tensor, "层输入"),
                shape=_shape_of(tensor),
                kind="data",
                label=label,
            )
        )
    graph.edges.append(
        LevelEdge(
            src="g-other-groups",
            dst="g-group",
            tensor_id="layer_order",
            tensor_label="同构其它层组（不展开）",
            shape="",
            kind="aux",
            label="同构模板",
        )
    )
    return graph


def _group_sources(block: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    for op in block.get("operators", []):
        sources = _sources(op)
        if sources:
            return sources
    return ()


# ---------------------------------------------------------------------------
# Block/Decoder 层：模块级数据流
# ---------------------------------------------------------------------------

def _op_topological_order(block: Mapping[str, Any]) -> list[str]:
    """块内算子的确定性拓扑序（Kahn，同层按原始顺序），用于把模块切成真实阶段。"""
    ops = list(block.get("operators", []))
    order_index = {str(op.get("id")): index for index, op in enumerate(ops)}
    producer = {
        str(tensor_id): str(op.get("id"))
        for op in ops
        for tensor_id in op.get("outputs", [])
    }
    indegree = {str(op.get("id")): 0 for op in ops}
    adjacency: dict[str, list[str]] = {str(op.get("id")): [] for op in ops}
    for op in ops:
        dst = str(op.get("id"))
        for tensor_id in op.get("inputs", []):
            src = producer.get(str(tensor_id))
            if src is None or src == dst or src not in adjacency:
                continue
            adjacency[src].append(dst)
            indegree[dst] += 1
    ready = sorted(
        (op_id for op_id, degree in indegree.items() if degree == 0),
        key=lambda item: order_index[item],
    )
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for dst in adjacency[current]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
                ready.sort(key=lambda item: order_index[item])
    if len(order) != len(ops):
        raise ArchifyLayersError("块内算子图存在环，无法形成单向拓扑阶段")
    return order


def _module_stages(
    block: Mapping[str, Any], modules: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """按拓扑序把每个模块切成连续区间；跨模块互依的模块因此自然拆成前后阶段。

    这样模块级图是真实 DAG：``kv_concat`` 与 ``attn_mask`` 的块偏置、注意力核心
    不再互相回指，而是“KV 拼接 → mask 构建 → 核心计算”的先后阶段。
    """
    order = _op_topological_order(block)
    position = {op_id: index for index, op_id in enumerate(order)}
    module_of_op = {
        str(op.get("id")): module for module in modules for op in module.get("ops", [])
    }
    stages: list[dict[str, Any]] = []
    last_stage_of: dict[str, dict[str, Any]] = {}
    last_position: dict[str, int] = {}
    for op_id in order:
        module = module_of_op.get(op_id)
        if module is None:
            raise ArchifyLayersError(f"算子 {op_id} 不属于任何模块分组")
        current_position = position[op_id]
        previous = last_stage_of.get(str(module["id"]))
        # 只有“紧接上一位置”才算同一连续区间；被其它模块插入时切成新阶段，
        # 这样跨模块的先后依赖不会在同一模块框里形成双向箭头。
        if previous is None or current_position != previous["last"] + 1:
            previous = {
                "stage_id": f"{module['id']}#{len(stages)}",
                "module_id": str(module["id"]),
                "title": str(module.get("title", module["id"])),
                "ops": [],
                "first": current_position,
                "last": current_position,
            }
            stages.append(previous)
            last_stage_of[str(module["id"])] = previous
        else:
            previous["last"] = current_position
        previous["ops"].append(op_id)
        last_position[str(module["id"])] = current_position
    # 同模块被切开的阶段数 → 给标题加前后段标注（保留真实模块名）
    counts: dict[str, int] = {}
    for stage in stages:
        counts[stage["module_id"]] = counts.get(stage["module_id"], 0) + 1
    seen: dict[str, int] = {}
    for stage in stages:
        module_id = stage["module_id"]
        if counts[module_id] > 1:
            seen[module_id] = seen.get(module_id, 0) + 1
            stage["label"] = f"{stage['title']}（阶段 {seen[module_id]}/{counts[module_id]}）"
        else:
            stage["label"] = stage["title"]
    return stages


def build_decoder_graph(
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
    attention_types: Mapping[str, str],
) -> LevelGraph:
    """单个 Decoder 的分支级数据流。

    - mHC 只作为一个 ``mHC.forward`` 节点，返回 post/comb/collapsed；
    - collapsed 经 Norm 进入现有 Attention/MoE 阶段（内部仍由专门页签承接）；
    - Attention/MoE 的 branch_output 交给 Decoder.forward 的 post 展开；
    - comb 在 Decoder.forward 中先转置，再与原始多流、展开后的分支输出相加。
    """
    gid = str(group["id"])
    tensors = _tensor_index(block)
    stages = _module_stages(block, modules)
    graph = LevelGraph(
        key=f"decoder-{gid}",
        level="decoder",
        title=f"DeepSeek V4 · {group.get('label', gid)} Decoder 分支数据流",
        subtitle=(
            f"代表层 {group.get('representative_layer_index')} · "
            f"{group.get('attention_type', '')} + {group.get('moe_type', '')} · "
            f"两段 mHC / {len(stages)} 个模块阶段 / {len(block.get('operators', []))} 算子"
        ),
        notes=tuple(block.get("cache_notes", [])) + tuple(block.get("coverage_notes", [])),
        parameter_items=_parameter_items(block),
        variant=gid,
        group=gid,
    )
    stage_of_op: dict[str, str] = {}
    for stage in stages:
        for op_id in stage["ops"]:
            stage_of_op[op_id] = stage["stage_id"]
    boundary_tensor = next(
        (t for t in block.get("tensors", []) if t.get("id") == "hidden_streams_in"), None
    )
    boundary_out = next(
        (t for t in block.get("tensors", []) if t.get("id") == "hidden_streams_out"), None
    )
    graph.nodes.append(
        LevelNode(
            node_id="d-in",
            label="hidden_streams（层输入）",
            sublabel=_shape_of(boundary_tensor),
            tag="输入张量",
            component_type="external",
            sources=_group_sources(block),
            role="boundary_in",
        )
    )
    op_index = _op_index(block)
    node_of_op: dict[str, str] = {}
    residual_labels = {
        "attn_residual_combT": ("attn 残差：combᵀ 转置", "transpose · comb.T"),
        "attn_residual_expand": ("attn 残差：post 展开 × branch_output", "broadcast-mul"),
        "attn_residual_mix": ("attn 残差：combᵀ 混合 + Add", "matmul+add"),
        "ffn_residual_combT": ("ffn 残差：combᵀ 转置", "transpose · comb.T"),
        "ffn_residual_expand": ("ffn 残差：post 展开 × branch_output", "broadcast-mul"),
        "ffn_residual_mix": ("ffn 残差：combᵀ 混合 + Add", "matmul+add"),
    }
    for stage in stages:
        ops = [op_index[op_id] for op_id in stage["ops"]]
        if stage["module_id"] in ("res_attn", "res_ffn"):
            for op, op_id in zip(ops, stage["ops"]):
                label, sublabel_tail = residual_labels.get(
                    str(op["id"]), (_op_label(op), _op_sublabel(op))
                )
                node_id = f"m-{op['id']}"
                graph.nodes.append(
                    LevelNode(
                        node_id=node_id,
                        label=label,
                        sublabel=f"{sublabel_tail} · Decoder.forward",
                        tag="Decoder.forward 残差（不在 mHC.forward）",
                        component_type="backend",
                        sources=_sources(op),
                        role="main",
                        details=(
                            str(op.get("note", "")),
                            "该残差展开/混合/相加由 Decoder.forward 执行；"
                            "mHC.forward 只返回 post/comb/collapsed。",
                        ),
                        op_ids=(str(op["id"]),),
                    )
                )
                node_of_op[str(op["id"])] = node_id
            continue
        label = stage["label"]
        sublabel = f"{len(ops)} 算子 · " + "、".join(
            _module_tail(op.get("module_path", "")) for op in ops[:2]
        )
        details = tuple(
            f"{_op_label(op)}（{op.get('id')} · {op.get('module_path', '')}）"
            for op in ops
        )
        if stage["module_id"] in ("mhc_attn", "mhc_ffn"):
            label = f"{label}：返回 post / comb / collapsed"
            sublabel = "mHC.forward"
            details = details + (
                "mHC.forward 只返回 post/comb/collapsed；"
                "Attention/MoE 分支调用与残差合并发生在 Decoder.forward。",
                "本节点未展开 mHC 内部；细图入口：mHC 页签。",
            )
        elif stage["module_id"].startswith(("attn_", "norm_attn")):
            details = details + (
                "模块未展开：Q/KV/O、缓存、mask 与核心细节见 "
                "Attention/缓存页签；本节点只表示 Decoder 分支边界。",
            )
        elif stage["module_id"].startswith(("moe_", "norm_ffn")):
            details = details + (
                "模块未展开：Hash 路由、专家与共享专家细节见 MoE 页签；"
                "本节点只表示 Decoder 分支边界。",
            )
        node_id = f"m-{stage['stage_id']}"
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=label,
                sublabel=sublabel,
                tag=stage["module_id"],
                component_type="backend",
                sources=_sources(ops[0]),
                role="main",
                details=details,
                op_ids=tuple(str(op.get("id")) for op in ops),
            )
        )
        for op_id in stage["ops"]:
            node_of_op[op_id] = node_id
    graph.nodes.append(
        LevelNode(
            node_id="d-out",
            label="hidden_streams（层输出）",
            sublabel=_shape_of(boundary_out),
            tag="输出张量",
            component_type="external",
            sources=_group_sources(block),
            role="boundary_out",
        )
    )
    seen: set[tuple[str, str, str]] = set()
    semantic_labels = {
        "attn_mhc_collapsed": "collapsed",
        "ffn_mhc_collapsed": "collapsed",
        "attn_mhc_post": "post",
        "ffn_mhc_post": "post",
        "attn_mhc_comb": "comb",
        "ffn_mhc_comb": "comb",
        "attn_block_output": "branch_output",
        "mlp_output": "branch_output",
        "attn_combT": "combᵀ",
        "ffn_combT": "combᵀ",
        "hidden_streams_in": "原始多流",
        "attn_out_streams": "混合后多流",
    }

    def add_edge(
        src: str,
        dst: str,
        tensor_id: str,
        *,
        kind: str = "data",
        label: str = "",
        condition: str | None = None,
    ) -> None:
        key = (src, dst, tensor_id)
        if key in seen:
            return
        seen.add(key)
        tensor = tensors.get(tensor_id)
        graph.edges.append(
            LevelEdge(
                src=src,
                dst=dst,
                tensor_id=tensor_id,
                tensor_label=_tensor_name(tensor, tensor_id),
                shape=_shape_of(tensor),
                kind=kind,
                label=label or semantic_labels.get(tensor_id, ""),
                condition=condition,
            )
        )

    producer = {
        str(tensor_id): str(op["id"])
        for op in block.get("operators", [])
        for tensor_id in op.get("outputs", [])
    }
    consumed = {
        str(tensor_id)
        for op in block.get("operators", [])
        for tensor_id in op.get("inputs", [])
    }
    weight_tensors = {
        str(tensor.get("id"))
        for tensor in block.get("tensors", [])
        if str(tensor.get("kind")) == "weight"
    }
    for op in block.get("operators", []):
        dst = node_of_op.get(str(op["id"]))
        if dst is None:
            continue
        for tensor_id in op.get("inputs", []):
            tensor_id = str(tensor_id)
            if tensor_id in weight_tensors:
                continue
            src_op = producer.get(tensor_id)
            if src_op is None:
                if tensor_id == "hidden_streams_in":
                    add_edge("d-in", dst, tensor_id)
                continue
            src_node = node_of_op.get(src_op)
            if src_node is None or src_node == dst:
                continue
            add_edge(src_node, dst, tensor_id)
    # 块输出：本块产生、块外消费（或显式层输出）的张量 → 输出边界。
    for op in block.get("operators", []):
        src_node = node_of_op.get(str(op["id"]))
        if src_node is None:
            continue
        for tensor_id in op.get("outputs", []):
            tensor_id = str(tensor_id)
            if tensor_id in weight_tensors:
                continue
            if tensor_id in consumed and tensor_id != "hidden_streams_out":
                continue
            add_edge(src_node, "d-out", tensor_id)
    state_nodes, state_edges = _add_module_state_nodes(
        graph, block, modules, stage_of_op, producer, consumed
    )
    graph.nodes.extend(state_nodes)
    graph.edges.extend(state_edges)
    graph.edges = _aggregate_module_edges(graph.edges)
    _assert_acyclic(graph)
    return graph


def _is_unknown_shape(edge: LevelEdge) -> bool:
    condition = str(edge.condition or "")
    return "未知" in condition or "unknown" in condition.lower()


def validate_graph_shapes(graph: LevelGraph) -> None:
    """校验器：真实数据边必须有全称形状；未知必须显式说明原因。

    只放行数据边（state/aux 关系由状态/辅助输入语义表达）；中间张量若形状无法从
    同源事实派生，必须在 condition 里写明“未知：<原因>”，否则拒绝出图。
    """
    problems = [
        edge
        for edge in graph.edges
        if edge.kind == "data" and not edge.shape and not _is_unknown_shape(edge)
    ]
    if problems:
        detail = "；".join(
            f"{edge.src} → {edge.dst}（{edge.tensor_id}）" for edge in problems[:5]
        )
        raise ArchifyLayersError(
            f"图 {graph.key} 有 {len(problems)} 条内部数据边缺少形状且未标明未知原因：{detail}"
        )


def _assert_acyclic(graph: LevelGraph) -> None:
    """模块/阶段图必须是真实 DAG：不允许用同秩横向箭头掩盖双向回流。"""
    main_ids = {
        node.node_id
        for node in graph.nodes
        if node.role in ("boundary_in", "main", "boundary_out")
    }
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in main_ids}
    for edge in graph.edges:
        if edge.kind == "data" and edge.src in main_ids and edge.dst in main_ids:
            adjacency[edge.src].append(edge.dst)
    components = _strongly_connected(adjacency)
    cyclic = [members for members in components if len(members) > 1]
    if cyclic:
        raise ArchifyLayersError(
            f"图 {graph.key} 仍存在双向数据依赖（SCC={cyclic}）："
            "需要按拓扑把模块拆成真实阶段，不能用同秩/横向箭头掩盖回流"
        )


def _add_module_state_nodes(
    graph: LevelGraph,
    block: Mapping[str, Any],
    modules: Sequence[Mapping[str, Any]],
    stage_of_op: Mapping[str, str],
    producer: Mapping[str, str],
    consumed: Mapping[str, Any],
) -> tuple[list[LevelNode], list[LevelEdge]]:
    """跨调用状态：只画算子最少的缓存模块（主 KV）的前态/后态。"""
    tensors = _tensor_index(block)
    cache_kind = {
        str(tensor["id"])
        for tensor in block.get("tensors", [])
        if tensor.get("kind") == "cache"
    }
    cache_modules = []
    for module in modules:
        pre_ids: list[str] = []
        post_ids: list[str] = []
        for op in module.get("ops", []):
            for tensor_id in op.get("inputs", []):
                tensor_id = str(tensor_id)
                if tensor_id in cache_kind and tensor_id not in producer:
                    pre_ids.append(tensor_id)
            for tensor_id in op.get("outputs", []):
                tensor_id = str(tensor_id)
                if (
                    tensor_id in cache_kind
                    and tensor_id.endswith("_post")
                    and tensor_id not in consumed
                ):
                    post_ids.append(tensor_id)
        if pre_ids or post_ids:
            cache_modules.append(
                (len(module.get("ops", [])), str(module["id"]), module, pre_ids, post_ids)
            )
    primary = (
        min(cache_modules, key=lambda item: (item[0], item[1]))[1]
        if cache_modules
        else None
    )
    nodes: list[LevelNode] = []
    edges: list[LevelEdge] = []
    for _count, module_id, module, pre_ids, post_ids in cache_modules:
        if module_id != primary:
            continue
        stage_ids = sorted(
            {
                stage_of_op[str(op["id"])]
                for op in module.get("ops", [])
                if str(op["id"]) in stage_of_op
            }
        )
        if pre_ids:
            tensor_id = pre_ids[0]
            tensor = tensors.get(tensor_id)
            consumer_op = next(
                (
                    op
                    for op in module.get("ops", [])
                    if tensor_id in [str(item) for item in op.get("inputs", [])]
                ),
                {},
            )
            nodes.append(
                LevelNode(
                    node_id=f"d-cache-{_slug(tensor_id)}",
                    label="主 KV 前态（调用前）",
                    sublabel=f"{_tensor_name(tensor, tensor_id)} {_shape_of(tensor)}",
                    tag="跨调用状态",
                    component_type="external",
                    sources=_sources(consumer_op),
                    role="state",
                    anchor=f"m-{stage_ids[0]}",
                )
            )
            for tensor_id in pre_ids:
                tensor = tensors.get(tensor_id)
                edges.append(
                    LevelEdge(
                        src=f"d-cache-{_slug(pre_ids[0])}",
                        dst=f"m-{stage_ids[0]}",
                        tensor_id=tensor_id,
                        tensor_label=_tensor_name(tensor, tensor_id),
                        shape=_shape_of(tensor),
                        kind="state",
                        label="状态读入",
                    )
                )
        if post_ids:
            tensor_id = post_ids[0]
            tensor = tensors.get(tensor_id)
            production_op = next(
                (
                    op
                    for op in module.get("ops", [])
                    if tensor_id in [str(item) for item in op.get("outputs", [])]
                ),
                module.get("ops", [{}])[0],
            )
            nodes.append(
                LevelNode(
                    node_id=f"d-cache-{_slug(tensor_id)}",
                    label="主 KV 后态（调用后）",
                    sublabel=f"{_tensor_name(tensor, tensor_id)} {_shape_of(tensor)}",
                    tag="跨调用状态",
                    component_type="external",
                    sources=_sources(production_op),
                    role="state",
                    anchor=f"m-{stage_ids[-1]}",
                )
            )
            for tensor_id in post_ids:
                tensor = tensors.get(tensor_id)
                edges.append(
                    LevelEdge(
                        src=f"m-{stage_ids[-1]}",
                        dst=f"d-cache-{_slug(post_ids[0])}",
                        tensor_id=tensor_id,
                        tensor_label=_tensor_name(tensor, tensor_id),
                        shape=_shape_of(tensor),
                        kind="state",
                        label="状态写出",
                    )
                )
    return nodes, edges


def _aggregate_module_edges(edges: Sequence[LevelEdge]) -> list[LevelEdge]:
    """同一阶段/模块对之间的多个真实张量合并为一条关系，成员逐条保留。"""
    merged: dict[tuple[str, str, str], LevelEdge] = {}
    order: list[tuple[str, str, str]] = []
    for edge in edges:
        key = (edge.src, edge.dst, edge.kind)
        if key not in merged:
            merged[key] = LevelEdge(
                src=edge.src,
                dst=edge.dst,
                tensor_id=edge.tensor_id,
                tensor_label=edge.tensor_label,
                shape=edge.shape,
                kind=edge.kind,
                label=edge.label,
                condition=edge.condition,
                members=(),
            )
            order.append(key)
        record = merged[key]
        record.members = record.members + (
            {
                "tensor_id": edge.tensor_id,
                "tensor_label": edge.tensor_label,
                "shape": edge.shape,
            },
        )
    result = []
    for key in order:
        record = merged[key]
        count = len(record.members)
        if count == 1:
            record.label = record.label or record.tensor_label
        elif record.kind == "state":
            record.label = f"{count} 项状态"
        else:
            record.label = " + ".join(
                str(item["tensor_label"]) for item in record.members[:2]
            ) + (f" 等 {count} 张量" if count > 2 else "")
        result.append(record)
    return result


def _first_op_using(
    block: Mapping[str, Any],
    tensor_id: str,
    module_of_op: Mapping[str, Mapping[str, Any]],
    producers: Sequence[str],
) -> str:
    for op in block.get("operators", []):
        if str(op["id"]) in module_of_op and tensor_id in op.get("inputs", []):
            return str(op["id"])
    return ""


def _slug(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "-", text).strip("-").lower()


def _producer_of(block: Mapping[str, Any], tensor_id: str) -> str | None:
    for op in block.get("operators", []):
        if tensor_id in op.get("outputs", []):
            return str(op["id"])
    return None


# ---------------------------------------------------------------------------
# MoE / mHC 层：算子级
# ---------------------------------------------------------------------------


def build_moe_graph(
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
) -> LevelGraph:
    """MoE：路由/分派/单专家模板/共享专家/汇合，共享专家三个 Linear 具名展开。

    FFN 复合算子的展开来自同源语义投影 ``ffn_named_linear_steps``（R-FFN-1/2/3，
    各带成立条件与拒绝原因）；展开只影响本层图，不修改旧算子事实与 Excel 粒度。
    """
    from structure_reader.extract.deepseek_v4_semantic import ffn_named_linear_steps

    gid = str(group["id"])
    wanted = [
        m for m in modules if str(m["id"]).startswith("moe_") or str(m["id"]) == "norm_ffn"
    ]
    ops = [op for module in wanted for op in module.get("ops", [])]
    if not ops:
        raise ArchifyLayersError(f"层组 {gid} 没有 MoE 模块")
    expansions = ffn_named_linear_steps(block)
    applied = {item.op_id: item for item in expansions if item.status == "applied"}
    rejected = [item for item in expansions if item.status == "rejected"]
    tensors = _tensor_index(block)
    graph = LevelGraph(
        key=f"moe-{gid}",
        level="moe",
        title=f"DeepSeek V4 · {group.get('label', gid)} FFN/MoE 内部计算",
        subtitle=(
            f"代表层 {group.get('representative_layer_index')} · moe_type={group.get('moe_type', '')} · "
            f"{len(ops) - len(applied)} 复合算子 + {sum(len(item.steps) for item in applied.values())} "
            "具名步骤（共享专家 gate/up/down Linear、专家模板融合参数）"
        ),
        notes=tuple(block.get("cache_notes", []))[:2]
        + tuple(block.get("coverage_notes", []))[:2],
        parameter_items=_parameter_items(block),
        variant=gid,
        group=gid,
    )
    graph.expansions = tuple(item.to_json() for item in expansions)
    member_ops = [op for op in ops if str(op.get("id")) not in applied]
    member_ids = {str(op.get("id")) for op in member_ops}
    # 被展开的复合算子仍属于本段内部：它的输入/输出不是段外依赖。
    inside_ids = member_ids | set(applied)
    producer = {
        str(tensor_id): str(op["id"])
        for op in block.get("operators", [])
        for tensor_id in op.get("outputs", [])
    }
    consumer = {
        str(tensor_id): str(op["id"])
        for op in block.get("operators", [])
        for tensor_id in op.get("inputs", [])
    }
    entry = tensors.get("ffn_normed")
    external_inputs = sorted(
        {
            str(tensor_id)
            for op in member_ops
            for tensor_id in op.get("inputs", [])
            if producer.get(str(tensor_id)) not in member_ids
            and str(tensor_id) not in {
                out
                for item in applied.values()
                for step in item.steps
                for _port, out in step.outputs
            }
            and str(tensors.get(str(tensor_id), {}).get("kind")) != "weight"
        }
    )
    graph.nodes.append(
        LevelNode(
            node_id="o-in",
            label="段输入（Block 上游张量）",
            sublabel=f"{len(external_inputs)} 项外部激活输入",
            tag="输入张量",
            component_type="external",
            sources=_sources(member_ops[0]),
            role="boundary_in",
            details=tuple(
                f"{_tensor_name(tensors.get(tid), tid)} {_shape_of(tensors.get(tid))}"
                f"（生产者 {producer.get(tid) or '段外/调用方'}）"
                for tid in external_inputs
            ),
        )
    )
    for op in member_ops:
        graph.nodes.append(
            LevelNode(
                node_id=f"o-{op['id']}",
                label=_op_label(op),
                sublabel=_op_sublabel(op),
                tag=str(op.get("module_path", "")).split(".")[-1] or "moe",
                component_type="backend",
                sources=_sources(op),
                role="main",
                details=(str(op.get("note", "")),),
                op_ids=(str(op["id"]),),
            )
        )
    step_producer: dict[str, str] = {}
    semantic_tensors: dict[str, dict[str, Any]] = {
        str(fact["id"]): dict(fact)
        for item in applied.values()
        for step in item.steps
        for fact in step.tensor_facts
    }
    step_node_of: dict[str, LevelNode] = {}
    for op_id, item in applied.items():
        composite = next(op for op in ops if str(op.get("id")) == op_id)
        for step in item.steps:
            node_id = f"o-{step.step_id}"
            shape = ""
            source_dicts = tuple(
                {"path": source.path, "line": source.line, **({"end_line": source.end_line} if source.end_line else {})}
                for source in step.sources[:3]
            )
            out_tensors = [tensor_id for _port, tensor_id in step.outputs]
            if out_tensors:
                shape = _shape_of(tensors.get(out_tensors[0]))
            weight = step.weight or {}
            sublabel = (
                f"{weight.get('name')} [{shape_text(weight.get('shape', ()))}]"
                if weight
                else f"{step.op_type} · {_module_tail(step.module_path)}"
            )
            node = LevelNode(
                node_id=node_id,
                label=step.label,
                sublabel=sublabel,
                tag=_module_tail(step.module_path) or "Linear",
                component_type="backend",
                sources=source_dicts,
                role="main",
                details=(
                    step.note,
                    f"模块路径 {step.module_path}",
                    *(
                        [
                            f"权重 {weight.get('name')} "
                            f"[{shape_text(weight.get('shape', ()))}]"
                        ]
                        if weight
                        else []
                    ),
                    *[
                        f"输出张量 {fact['name']} [{shape_text(fact['shape'])}]"
                        for fact in step.tensor_facts
                    ],
                    f"源码 {', '.join(source.location for source in step.sources[:3])}",
                ),
                op_ids=(op_id,),
            )
            graph.nodes.append(node)
            step_node_of[step.step_id] = node
            for tensor_id in out_tensors:
                step_producer[tensor_id] = node_id
    exit_tensor_doc = tensors.get("mlp_output")
    graph.nodes.append(
        LevelNode(
            node_id="o-out",
            label="段输出（本层接口张量）",
            sublabel=f"{_tensor_name(exit_tensor_doc, 'mlp_output')} {_shape_of(exit_tensor_doc)}",
            tag="输出张量",
            component_type="external",
            sources=_sources(
                next(op for op in member_ops if "mlp_output" in op.get("outputs", ()))
            ),
            role="boundary_out",
        )
    )

    def add(src: str, dst: str, tensor_id: str, *, label: str = "", condition: str | None = None) -> None:
        tensor = tensors.get(tensor_id)
        fact = semantic_tensors.get(tensor_id)
        if tensor is not None:
            tensor_label = _tensor_name(tensor, tensor_id)
            shape = _shape_of(tensor)
        elif fact is not None:
            tensor_label = str(fact.get("name") or tensor_id)
            shape = shape_text(fact.get("shape", ()))
            if not shape and fact.get("reason"):
                condition = condition or f"未知：{fact['reason']}"
        else:
            tensor_label = tensor_id
            shape = ""
            condition = condition or "未知：该张量在既有事实中未登记形状"
        graph.edges.append(
            LevelEdge(
                src=src,
                dst=dst,
                tensor_id=tensor_id,
                tensor_label=tensor_label,
                shape=shape,
                kind="data",
                label=label,
                condition=condition,
            )
        )

    for op in member_ops:
        dst = f"o-{op['id']}"
        for tensor_id in op.get("inputs", []):
            tensor_id = str(tensor_id)
            tensor = tensors.get(tensor_id)
            if str(tensor.get("kind") if tensor else "") == "weight":
                continue
            src_op = producer.get(tensor_id)
            if tensor_id in step_producer:
                add(step_producer[tensor_id], dst, tensor_id)
            elif src_op in member_ids:
                add(f"o-{src_op}", dst, tensor_id)
            else:
                add(
                    "o-in",
                    dst,
                    tensor_id,
                    label="段输入",
                    condition=(
                        f"生产者 {src_op} 在视图中（段外）" if src_op else "段外/调用方输入"
                    ),
                )
    for op_id, item in applied.items():
        for step in item.steps:
            dst = f"o-{step.step_id}"
            for _port, tensor_id in step.inputs:
                tensor_id = str(tensor_id)
                if tensor_id in step_producer:
                    add(step_producer[tensor_id], dst, tensor_id)
                    continue
                src_op = producer.get(tensor_id)
                if src_op in member_ids:
                    add(f"o-{src_op}", dst, tensor_id)
                elif src_op in applied:
                    continue
                else:
                    add(
                        "o-in",
                        dst,
                        tensor_id,
                        label="段输入",
                        condition=(
                            f"生产者 {src_op} 在视图中（段外）"
                            if src_op
                            else "段外/调用方输入"
                        ),
                    )
    exit_producers = [op for op in member_ops if "mlp_output" in op.get("outputs", ())]
    for op in exit_producers:
        add(f"o-{op['id']}", "o-out", "mlp_output")
    # 辅助出口：段内产生、段外消费的非主输出张量（含展开步骤的输出）。
    member_outputs = {
        str(tensor_id): (
            f"o-{producer[str(tensor_id)]}"
            if producer.get(str(tensor_id)) in member_ids
            else step_producer[str(tensor_id)]
        )
        for op in ops
        for tensor_id in op.get("outputs", [])
        if str(tensor_id) in producer or str(tensor_id) in step_producer
    }
    aux_outputs = sorted(
        {
            tensor_id
            for tensor_id, src_node in member_outputs.items()
            if tensor_id != "mlp_output"
            and str(tensors.get(tensor_id, {}).get("kind")) != "weight"
            and any(
                str(candidate["id"]) not in inside_ids
                for candidate in block.get("operators", [])
                if tensor_id in candidate.get("inputs", [])
            )
        }
    )
    for tensor_id in aux_outputs:
        tensor = tensors.get(tensor_id)
        node_id = f"o-aux-{_slug(tensor_id)}"
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=_tensor_name(tensor, tensor_id),
                sublabel=_shape_of(tensor),
                tag="段外消费（同一张量）",
                component_type="external",
                sources=_sources(
                    next(
                        op
                        for op in ops
                        if tensor_id in [str(item) for item in op.get("outputs", [])]
                    )
                ),
                role="branch",
                anchor=member_outputs[tensor_id],
                details=(
                    f"段内产生，段外消费者 {consumer.get(tensor_id, '（块外）')} 读取；不并入主出口",
                ),
            )
        )
        add(member_outputs[tensor_id], node_id, tensor_id, label="段外消费")
    if rejected:
        graph.notes = graph.notes + tuple(
            f"{item.op_id} 未展开：{item.reason}" for item in rejected
        )[:2]
    _assert_acyclic(graph)
    return graph


def _cache_tensor_groups(block: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    tensors = [t for t in block.get("tensors", []) if str(t.get("kind")) == "cache"]
    groups: dict[str, list[Mapping[str, Any]]] = {"main": [], "compressor": [], "indexer": []}
    for tensor in tensors:
        tid = str(tensor.get("id"))
        if tid.startswith(("tid2eid", "kv_full")):
            continue
        if "_buffer_i_" in tid or "_overlap_i_" in tid or "_compressed_i" in tid or "_entry_count_i" in tid:
            groups["indexer"].append(tensor)
        elif "_buffer_c_" in tid or "_overlap_c_" in tid or "_compressed_c" in tid or "_entry_count_c" in tid:
            groups["compressor"].append(tensor)
        else:
            groups["main"].append(tensor)
    return groups


def _cache_summary(items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "无状态"
    kinds = []
    for tensor in items:
        tid = str(tensor.get("id"))
        if "overlap" in tid:
            kinds.append("overlap")
        elif "buffer" in tid:
            kinds.append("buffer")
        elif "compressed" in tid:
            kinds.append("compressed")
        elif "entry_count" in tid:
            kinds.append("entry_count")
        elif "keys" in tid:
            kinds.append("keys")
        elif "cumlen" in tid:
            kinds.append("cumlen")
        else:
            kinds.append("state")
    unique = []
    for kind in kinds:
        if kind not in unique:
            unique.append(kind)
    return f"{len(items)} 项状态"


def build_cache_graph(
    block: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
    part: str = "main",
) -> LevelGraph:
    """缓存状态分图：main / compressor / indexer；每图只放本组状态，避免横向膨胀。"""
    gid = str(group["id"])
    groups = _cache_tensor_groups(block)
    if part == "main":
        groups = {"main": groups["main"], "compressor": [], "indexer": []}
    elif part == "compressor":
        groups = {"main": [], "compressor": groups["compressor"], "indexer": []}
    elif part == "indexer":
        groups = {"main": [], "compressor": [], "indexer": groups["indexer"]}
    else:
        raise ArchifyLayersError(f"未知缓存分图 {part!r}")
    title_part = {"main": "主 KV", "compressor": "压缩器状态", "indexer": "Indexer 状态"}[part]
    graph = LevelGraph(
        key=f"cache-{part}-{gid}",
        level="cache",
        title=f"DeepSeek V4 · {group.get('label', gid)} {title_part}",
        subtitle=f"代表层 {group.get('representative_layer_index')} · 调用前/更新/调用后",
        notes=(
            "逐 tensor 身份/全称 shape 在节点/边与页面缓存表中；kvcachelen 是逻辑历史，"
            "不等于物理保留条目或字节。",
        ),
        parameter_items=_filter_params(block, (
            "B", "seqlen", "kvcachelen", "Nkv", "Dh", "W", "M", "Tbuf", "Tbufp",
            "Tcc", "Tci", "Tbufi", "Tbufip", "Nwin", "Nwin_i", "Ncomp", "Ki", "Ih", "Id",
        )),
        variant=gid, group=gid,
    )
    tensors = _tensor_index(block)
    edges: list[LevelEdge] = []

    def add_state_group(prefix: str, items: list[Mapping[str, Any]], update_label: str,
                        update_ops: tuple[str, ...], note: str) -> None:
        if not items:
            return
        pre = [t for t in items if not str(t.get("id")).endswith("_post")]
        post = [t for t in items if str(t.get("id")).endswith("_post")]
        graph.nodes.append(LevelNode(
            node_id=f"cache-{prefix}-in", label=f"{prefix} 调用前",
            sublabel=_cache_summary(pre), tag="调用前状态",
            component_type="external", sources=_group_sources(block), role="boundary_in",
        ))
        graph.nodes.append(LevelNode(
            node_id=f"cache-{prefix}-update", label=update_label,
            sublabel=note, tag="真实状态更新", component_type="backend",
            sources=_group_sources(block), role="main", op_ids=update_ops,
        ))
        graph.nodes.append(LevelNode(
            node_id=f"cache-{prefix}-out", label=f"{prefix} 调用后",
            sublabel=_cache_summary(post), tag="调用后状态",
            component_type="external", sources=_group_sources(block), role="boundary_out",
        ))

        def edge(src: str, dst: str, seq: Sequence[Mapping[str, Any]], label: str) -> None:
            if not seq:
                return
            first = seq[0]
            edges.append(LevelEdge(
                src=src, dst=dst, tensor_id=str(first.get("id")),
                tensor_label=_tensor_name(first, str(first.get("id"))),
                shape=_shape_of(first) or "标量", kind="data",
                label=f"{label}（{len(seq)} 项）",
                members=tuple(
                    {"tensor_id": str(t.get("id")),
                     "tensor_label": _tensor_name(t, str(t.get("id"))),
                     "shape": _shape_of(t) or "标量"} for t in seq),
            ))
        edge(f"cache-{prefix}-in", f"cache-{prefix}-update", pre, "调用前读取")
        edge(f"cache-{prefix}-update", f"cache-{prefix}-out", post, "调用后写回")

    if part == "main":
        main = groups["main"]
        pre = [t for t in main if not str(t.get("id")).endswith("_post")]
        post = [t for t in main if str(t.get("id")).endswith("_post")]
        graph.nodes.append(LevelNode(
            node_id="cache-main-in", label="主 KV 调用前",
            sublabel=_cache_summary(pre), tag="调用前状态",
            component_type="external", sources=_group_sources(block), role="boundary_in",
        ))
        graph.nodes.append(LevelNode(
            node_id="cache-main-new-kv", label="本调用新 KV",
            sublabel="kv_rope｜" + (_shape_of(tensors.get("kv_rope")) or "标量"),
            tag="本次输入", component_type="external", sources=_group_sources(block), role="boundary_in",
        ))
        graph.nodes.append(LevelNode(
            node_id="cache-main-update", label="cache.update 主 KV",
            sublabel="keys=values；持久尾部+新 KV；写 cumlen",
            tag="cache_update", component_type="backend", sources=_group_sources(block),
            role="main", op_ids=("cache_update",),
        ))
        graph.nodes.append(LevelNode(
            node_id="cache-main-full", label="本次可读 full",
            sublabel="kv_full｜" + (_shape_of(tensors.get("kv_full")) or "标量"),
            tag="本次输出", component_type="external", sources=_group_sources(block), role="branch",
            anchor="cache-main-update",
        ))
        graph.nodes.append(LevelNode(
            node_id="cache-main-consumer", label="repeat_kv 消费",
            sublabel="full→K/V 广播到 Attention",
            tag="本次消费者", component_type="external", sources=_group_sources(block), role="branch",
            anchor="cache-main-update",
        ))
        graph.nodes.append(LevelNode(
            node_id="cache-main-out", label="主 KV 调用后",
            sublabel=_cache_summary(post), tag="调用后状态",
            component_type="external", sources=_group_sources(block), role="boundary_out",
        ))
        graph.nodes.append(LevelNode(
            node_id="cache-main-cumlen-post", label="cumlen 调用后（阅读投影）",
            sublabel="kvcachelen+seqlen", tag="累计长度投影",
            component_type="external", sources=_group_sources(block),
            role="branch", anchor="cache-main-update",
        ))

        def main_edge(src, dst, seq, label):
            if not seq:
                return
            first = seq[0]
            edges.append(LevelEdge(
                src=src, dst=dst, tensor_id=str(first.get("id")),
                tensor_label=_tensor_name(first, str(first.get("id"))),
                shape=_shape_of(first) or "标量", kind="data",
                label=f"{label}（{len(seq)} 项）",
                members=tuple(
                    {"tensor_id": str(t.get("id")),
                     "tensor_label": _tensor_name(t, str(t.get("id"))),
                     "shape": _shape_of(t) or "标量"} for t in seq),
            ))
        main_edge("cache-main-in", "cache-main-update", pre, "调用前读取")
        main_edge("cache-main-new-kv", "cache-main-update", [tensors["kv_rope"]], "本调用新 KV")
        main_edge("cache-main-update", "cache-main-full", [tensors["kv_full"]], "写出 full")
        main_edge("cache-main-full", "cache-main-consumer", [tensors["kv_full"]], "本次可读")
        main_edge("cache-main-update", "cache-main-out", post, "调用后写回")
        edges.append(LevelEdge(
            src="cache-main-update", dst="cache-main-cumlen-post",
            tensor_id="cache_cumlen_post", tensor_label="cumlen（调用后，阅读投影）",
            shape="kvcachelen+seqlen", kind="data",
            label="kvcachelen+seqlen（阅读投影）",
        ))

    if part == "compressor":
        add_state_group("compressor", groups["compressor"],
                        "压缩器状态更新",
                        ("c_store_window", "c_update_states"),
                        "buffer/overlap/compressed/entry_count；CSA Ca/Cb；HCA 非重叠")
    if part == "indexer":
        add_state_group("indexer", groups["indexer"],
                        "Indexer 状态更新",
                        ("i_store_window", "i_update_states"),
                        "独立 buffer/overlap/compressed/entry_count；不与外层压缩 KV 合并")
    graph.edges.extend(edges)
    _assert_acyclic(graph)
    return graph


def _unique_sources(items):
    out = []
    seen = set()
    for item in items:
        key = (item.get("path"), item.get("line"), item.get("end_line"))
        if key in seen:
            continue
        seen.add(key); out.append(item)
    return tuple(out)


def _stage_graph(
    block: Mapping[str, Any],
    *,
    stages: Sequence[tuple[str, str, tuple[str, ...]]],
    key: str,
    level: str,
    title: str,
    subtitle: str,
    entry_tensor: str,
    variant: str,
    group: str,
) -> LevelGraph:
    """按真实 op inputs/outputs 推导阶段边界，不再手写第二套输入/输出。

    - 阶段仍可逻辑分组，但边界、shape、跨阶段边全部由 block facts 推导；
    - 未知非权重 tensor / op 立即具名报错，禁止用标量兜底；
    - 阶段输出被后续阶段消费则连阶段边，否则连段边界输出；
    - 阶段输入未由任何阶段生产则连段边界输入，保留 cache/overlap/entry/cos/sin 等真实依赖。
    """
    ops = {str(op.get("id")): op for op in block.get("operators", [])}
    tensors = _tensor_index(block)
    stage_op_ids = [op for _sid, _label, op_ids in stages for op in op_ids]
    for op_id in stage_op_ids:
        if op_id not in ops:
            raise ArchifyLayersError(f"{key}: 未知 op_id {op_id!r}（阶段事实不存在）")

    # 每个 op 的真实 outputs；每个 tensor 的 producer（阶段内）
    op_outputs = {op_id: tuple(str(t) for t in ops[op_id].get("outputs", ()))
                  for op_id in stage_op_ids}
    tensor_producer: dict[str, str] = {}
    for sid, _label, op_ids in stages:
        for op_id in op_ids:
            for tensor_id in op_outputs[op_id]:
                tensor_producer.setdefault(tensor_id, sid)

    graph = LevelGraph(
        key=key, level=level, title=title, subtitle=subtitle,
        notes=(
            "阶段边界由真实 op inputs/outputs 推导；未知 id 报错，只有事实 shape=[] 才是标量。",
            "cache/overlap/entry_count/cos/sin 等外部依赖与状态写回均保留，不因阶段名省略。",
        ),
        variant=variant, group=group,
    )
    graph.nodes.append(LevelNode(
        node_id="s-in", label="段输入（真实外部依赖）",
        sublabel="逐张量身份/shape 见边与完整关系", tag="段边界",
        component_type="external", sources=_group_sources(block), role="boundary_in",
    ))
    stage_nodes: list[str] = []
    for sid, label, op_ids in stages:
        op_docs = [ops[op] for op in op_ids]
        details = tuple(
            f"{op}: {doc.get('note') or doc.get('op') or op}"
            for op, doc in zip(op_ids, op_docs)
        )
        graph.nodes.append(LevelNode(
            node_id=sid, label=label, sublabel=f"{len(op_ids)} 个真实算子",
            tag=f"{len(op_ids)} 个算子", component_type="backend",
            sources=_unique_sources(
                item for doc in op_docs for item in _sources(doc)
            )[:3],
            role="main", op_ids=tuple(op_ids), details=details,
        ))
        stage_nodes.append(sid)
    graph.nodes.append(LevelNode(
        node_id="s-out", label="段边界输出",
        sublabel="真实输出；未被后续阶段消费的写回/消费者在此列出",
        tag="段边界", component_type="external", sources=_group_sources(block),
        role="boundary_out",
    ))

    def actual_tensor(tensor_id: str) -> Mapping[str, Any]:
        tensor = tensors.get(tensor_id)
        if tensor is not None:
            return tensor
        raise ArchifyLayersError(
            f"{key}: 未知 tensor_id {tensor_id!r}（阶段事实不存在；"
            "不允许以标量/空 shape 兜底）"
        )

    def is_weight(tensor_id: str, tensor: Mapping[str, Any]) -> bool:
        return str(tensor.get("kind") or "") == "weight" or tensor_id.endswith(("_weight", "_w"))

    edges: list[LevelEdge] = []
    for sid, _label, op_ids in stages:
        # 阶段内局部生产，排除内部边；权重不画。
        local = {t for op in op_ids for t in op_outputs[op]}
        input_ids: list[str] = []
        for op in op_ids:
            for raw in ops[op].get("inputs", ()):
                tid = str(raw)
                if tid in local:
                    continue
                if tid not in input_ids:
                    input_ids.append(tid)
        for tid in input_ids:
            tensor = actual_tensor(tid)
            if is_weight(tid, tensor):
                continue
            src = tensor_producer.get(tid)
            if src == sid:
                continue
            edges.append(LevelEdge(
                src=src or "s-in", dst=sid, tensor_id=tid,
                tensor_label=_tensor_name(tensor, tid),
                shape=_shape_of(tensor) or "标量", kind="data",
                label=_edge_display_text(LevelEdge(
                    src="", dst="", tensor_id=tid,
                    tensor_label=_tensor_name(tensor, tid),
                    shape=_shape_of(tensor) or "标量", kind="data",
                    label=_tensor_name(tensor, tid),
                ), graph),
            ))

    # 所有阶段真实 outputs：图内消费与图外消费分别判断。
    # 同一 tensor 可以既被图内下游消费，又被图外真实 op 消费（扇出）；
    # 前者已有阶段边，后者仍必须连 s-out。纯图内中间张量不泄漏为输出。
    stage_ids = {sid for sid, _label, _op_ids in stages}
    stage_input_ops: dict[str, set[str]] = {}
    for sid, _label, op_ids in stages:
        for op in op_ids:
            for raw in ops[op].get("inputs", ()):
                stage_input_ops.setdefault(str(raw), set()).add(sid)
    all_op_inputs: dict[str, set[str]] = {}
    for op_id, op in ops.items():
        for raw in op.get("inputs", ()):
            all_op_inputs.setdefault(str(raw), set()).add(op_id)
    all_stage_ops = {op for _sid, _label, op_ids in stages for op in op_ids}
    for sid, _label, op_ids in stages:
        for op in op_ids:
            for tid in op_outputs[op]:
                tensor = actual_tensor(tid)
                if is_weight(tid, tensor):
                    continue
                outside = sorted(
                    op_id for op_id in all_op_inputs.get(tid, set())
                    if op_id not in all_stage_ops
                )
                internal = bool(stage_input_ops.get(tid))
                if not outside and internal:
                    # 只被本阶段图内消费：真实内部边，不伪造段出口。
                    continue
                if not outside and not internal:
                    # 真实末端只在“持久状态写回”时保留；普通中间产物
                    # （如 first_window_position/c_positions）不冒充段输出。
                    if not (str(tensor.get("kind") or "") == "cache"
                            and (tid.startswith("cache_") or tid.endswith("_post"))):
                        continue
                edges.append(LevelEdge(
                    src=sid, dst="s-out", tensor_id=tid,
                    tensor_label=_tensor_name(tensor, tid),
                    shape=_shape_of(tensor) or "标量", kind="data",
                    label=_edge_display_text(LevelEdge(
                        src="", dst="", tensor_id=tid,
                        tensor_label=_tensor_name(tensor, tid),
                        shape=_shape_of(tensor) or "标量", kind="data",
                        label=_tensor_name(tensor, tid),
                    ), graph),
                    condition=(f"图外消费者: {', '.join(outside)}"
                               if outside else "真实末端（图内无消费者）"),
                ))
    graph.edges.extend(edges)
    _assert_acyclic(graph)
    return graph


def build_compressor_graph(
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
) -> LevelGraph:
    gid = str(group["id"])
    attention_type = str(group.get("attention_type") or block.get("attention_type") or "")
    if attention_type == "heavily_compressed_attention":
        stages = (
            ("s-proj", "kv/gate 投影", ("c_kv_proj", "c_gate_proj")),
            ("s-window", "缓冲/窗口/余量", ("c_store_window", "c_window_view")),
            ("s-pool", "softmax 汇聚+Norm+RoPE", ("c_pool", "c_rope_gen", "c_rope")),
            ("s-state", "条目追加/历史", ("c_update_states",)),
            ("s-bias", "hca_block_bias · HCA（None 分支）", ("hca_block_bias",)),
            ("s-mask", "c_mask_extend · mask 扩展", ("c_mask_extend",)),
        )
        title = "DeepSeek V4 · HCA 压缩器（真实窗口/位置偏置）"
    else:
        stages = (
            ("s-proj", "kv/gate 投影", ("c_kv_gate_proj",)),
            ("s-window", "缓冲/Ca-Cb 重叠窗口", ("c_store_window", "c_window_view")),
            ("s-pool", "softmax 汇聚+Norm+RoPE", ("c_pool", "c_rope_gen", "c_rope")),
            ("s-state", "条目追加/历史", ("c_update_states",)),
            ("s-bias", "c_block_bias · CSA（topk 可见性）", ("c_block_bias",)),
            ("s-mask", "mask_extend · 扩展 mask", ("mask_extend",)),
        )
        title = "DeepSeek V4 · CSA 压缩器（Ca/Cb + Indexer 可见性）"
    graph = _stage_graph(
        block, stages=stages, key=f"compressor-{gid}", level="compressor",
        title=title, subtitle=f"代表层 {group.get('representative_layer_index')} · 6 个逻辑阶段",
        entry_tensor="attn_normed", variant=gid, group=gid,
    )
    graph.parameter_items = _filter_params(block, (
        "B", "seqlen", "kvcachelen", "H", "Nkv", "Dh", "M", "W", "Tbuf", "Tbufp",
        "Tcc", "Nwin", "Ncomp", "Eps", "Rk",
    ))
    return graph


def _filter_params(block: Mapping[str, Any], wanted: Sequence[str]) -> tuple[dict[str, Any], ...]:
    wanted_set = set(wanted)
    return tuple(
        item for item in _parameter_items(block, limit=64)
        if str(item.get("id")) in wanted_set
    )


def build_indexer_graph(
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
) -> LevelGraph:
    gid = str(group["id"])
    stages = (
        ("i-q", "q 投影（q_residual + cos/sin）", ("i_q_proj",)),
        ("i-kv", "KV/gate 投影", ("i_kv_gate_proj",)),
        ("i-window", "独立缓冲/重叠窗口", ("i_store_window", "i_window_view")),
        ("i-pool", "汇聚/自身条目 RoPE", ("i_pool_rope",)),
        ("i-state", "独立条目追加", ("i_update_states",)),
        ("i-score", "scorer（真实输入）", ("i_scorer",)),
        ("i-topk", "future mask + topk（含自身状态）", ("i_topk",)),
    )
    graph = _stage_graph(
        block, stages=stages, key=f"indexer-{gid}", level="indexer",
        title="DeepSeek V4 · Lightning Indexer（自身压缩状态）",
        subtitle=f"代表层 {group.get('representative_layer_index')} · 7 个逻辑阶段",
        entry_tensor="q_a_normed", variant=gid, group=gid,
    )
    graph.parameter_items = _filter_params(block, (
        "B", "seqlen", "kvcachelen", "Ki", "Ih", "Id", "M", "W", "Tbufi",
        "Tbufip", "Tci", "Nwin_i", "Eps", "Rk",
    ))
    return graph


def build_q_graph(
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
) -> LevelGraph:
    gid = str(group["id"])
    stages = (
        ("q-lowrank", "q_a_proj + q_a_norm", ("q_a_proj", "q_a_norm")),
        ("q-b", "q_b_proj → View/Transpose 轴布局", ("q_b_proj",)),
        ("q-norm", "q_b_norm（head_dim RMSNorm）", ("q_b_norm",)),
        ("q-rope", "部分 RoPE（尾部 qk_rope_head_dim）", ("q_rope_apply",)),
    )
    graph = _stage_graph(
        block, stages=stages, key=f"q-{gid}", level="q",
        title="DeepSeek V4 · Q 通路（低秩投影/轴布局/部分 RoPE）",
        subtitle=f"代表层 {group.get('representative_layer_index')} · 独立 Q 入口，返回 Attention 总览",
        entry_tensor="attn_normed", variant=gid, group=gid,
    )
    graph.parameter_items = _filter_params(block, (
        "B", "seqlen", "H", "Nh", "Dh", "Rq", "Nkv", "Eps", "ThetaC", "ThetaM",
    ))
    return graph


def build_o_graph(
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
) -> LevelGraph:
    gid = str(group["id"])
    stages = (
        ("o-core", "AV + 逆 RoPE + 轴布局", ("av_matmul", "attn_inv_rope")),
        ("o-group", "分组 Reshape（分组轴）", ("attn_group_reshape",)),
        ("o-a", "o_a_proj（GroupedLinear）", ("o_a_proj",)),
        ("o-b", "o_b_proj（Linear）", ("o_b_proj",)),
    )
    graph = _stage_graph(
        block, stages=stages, key=f"o-{gid}", level="o",
        title="DeepSeek V4 · O 通路（AV/逆 RoPE/分组输出）",
        subtitle=f"代表层 {group.get('representative_layer_index')} · 独立 O 入口，返回 Attention 总览",
        entry_tensor="attn_probs", variant=gid, group=gid,
    )
    graph.parameter_items = _filter_params(block, (
        "B", "seqlen", "H", "Nh", "Dh", "G", "Ro", "ThetaC", "ThetaM",
    ))
    return graph


def _moe_scope(group: Mapping[str, Any], block: Mapping[str, Any]) -> tuple[str, bool]:
    gid = str(group["id"])
    moe_type = str(group.get("moe_type") or block.get("moe_type") or "")
    return moe_type, moe_type == "hash_moe"


def _moe_edge(graph: LevelGraph, tensors, src: str, dst: str, tid: str,
              condition: str | None = None) -> None:
    tensor = tensors.get(tid)
    if tensor is None:
        raise ArchifyLayersError(f"{graph.key}: 未知 MoE tensor {tid!r}")
    graph.edges.append(LevelEdge(
        src=src, dst=dst, tensor_id=tid,
        tensor_label=_tensor_name(tensor, tid),
        shape=_shape_of(tensor) or "标量", kind="data",
        label=_shape_of(tensor) or "标量", condition=condition,
    ))


def build_moe_relation_graph(
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
) -> LevelGraph:
    """第三层：只显示 gate/路由、激活专家模板、共享专家与加和的关系。"""
    gid = str(group["id"]); tensors = _tensor_index(block)
    moe_type, is_hash = _moe_scope(group, block)
    graph = LevelGraph(
        key=f"moe-{gid}", level="moe-relation",
        title=f"DeepSeek V4 · {group.get('label', gid)} MoE 模块关系",
        subtitle=f"代表层 {group.get('representative_layer_index')} · {moe_type}",
        notes=("本层只画模块关系；路由、单专家、共享专家、汇合内部在下一层。",
               "激活专家与共享专家都从 ffn_flat/ffn_normed 的真实输入派生，不画 gate→共享专家数据依赖。"),
        parameter_items=_filter_params(block, ("B", "seqlen", "H", "E", "K", "Es", "Scl", "Lim")),
        variant=gid, group=gid,
    )
    for nid,label,sub,tag,role in (
        ("r-in","MoE 输入","ffn_normed / ffn_flat","段输入","boundary_in"),
        ("r-route",("Hash 查表路由" if is_hash else "TopK 纠偏选择"),"indices / weights","路由模块","main"),
        ("r-dispatch","token dispatch / 命中专家","expert_mask、hit_experts、token_idx","分派模块","main"),
        ("r-expert","单激活专家模板（未展开）","单专家加权输出 expert_down_out [N_e,H]","激活专家","main"),
        ("r-shared","共享专家（未展开）","独立 gate/up/down + 双侧 Clamp，全 token","共享专家","main"),
        ("r-combine","routed 零初值 index_add + View + shared 加和","expert_down_out+token_idx → routed_out → View → +shared","汇合","main"),
        ("r-out","mlp_output","本层接口","段输出","boundary_out"),
    ):
        graph.nodes.append(LevelNode(
            node_id=nid, label=label, sublabel=sub, tag=tag,
            component_type="external" if role in ("boundary_in","boundary_out") else "backend",
            sources=_group_sources(block), role=role,
            details=((f"moe_type={moe_type}；进入下一层查看真实张量与消费者"),),
        ))
    ffn = "ffn_normed"
    _moe_edge(graph,tensors,"r-in","r-route",ffn if is_hash else "ffn_flat")
    _moe_edge(graph,tensors,"r-in","r-shared",ffn)
    if is_hash:
        _moe_edge(graph,tensors,"r-route","r-dispatch","topk_indices")
    else:
        _moe_edge(graph,tensors,"r-route","r-dispatch","topk_indices")
    _moe_edge(graph,tensors,"r-in","r-dispatch","ffn_flat")
    _moe_edge(graph,tensors,"r-route","r-expert","topk_weights")
    _moe_edge(graph,tensors,"r-dispatch","r-expert","expert_tokens")
    _moe_edge(graph,tensors,"r-dispatch","r-expert","expert_token_idx")
    _moe_edge(graph,tensors,"r-dispatch","r-expert","expert_topk_pos")
    _moe_edge(graph,tensors,"r-expert","r-combine","expert_down_out")
    _moe_edge(graph,tensors,"r-dispatch","r-combine","expert_token_idx")
    _moe_edge(graph,tensors,"r-shared","r-combine","shared_out")
    _moe_edge(graph,tensors,"r-combine","r-out","mlp_output")
    semantic = {
        ("r-in","r-route"): "tokens",
        ("r-in","r-shared"): "全 token",
        ("r-route","r-dispatch"): "indices",
        ("r-in","r-dispatch"): "tokens",
        ("r-route","r-expert"): "weights",
        ("r-dispatch","r-expert"): "tokens/idx",
        ("r-expert","r-combine"): "单专家加权输出",
        ("r-dispatch","r-combine"): "index_add 寻址 token_idx",
        ("r-shared","r-combine"): "shared 全 token",
        ("r-combine","r-out"): "mlp_output",
    }
    for edge in graph.edges:
        if (edge.src, edge.dst) in semantic:
            edge.label = semantic[(edge.src, edge.dst)]
    _assert_acyclic(graph)
    return graph


def build_moe_route_graph(
    block: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
) -> LevelGraph:
    gid=str(group["id"]); moe_type,is_hash=_moe_scope(group,block)
    stages = (
        ("m-lookup","tid2eid[input_ids] 查表",("moe_hash_lookup",)),
        ("m-router","Hash 原始 scores gather/归一/缩放（无 TopK）",("moe_router",)),
        ("m-dispatch","分派/命中专家",("moe_dispatch",)),
        ("m-select","选 token 行/索引",("moe_select_tokens",)),
    ) if is_hash else (
        ("m-router","TopK 选择 + 原始 scores gather/归一/缩放",("moe_router",)),
        ("m-dispatch","分派/命中专家",("moe_dispatch",)),
        ("m-select","选 token 行/索引",("moe_select_tokens",)),
    )
    graph=_stage_graph(block,stages=stages,key=f"moe-route-{gid}",level="moe-route",
        title=("DeepSeek V4 · Hash 路由与分派" if is_hash else "DeepSeek V4 · TopK 路由与分派"),
        subtitle=f"代表层 {group.get('representative_layer_index')} · 原始 scores gather 计权，纠偏只用于选择",
        entry_tensor="ffn_flat",variant=gid,group=gid)
    tensors=_tensor_index(block)
    if not is_hash and "router_bias" in tensors:
        bias=tensors["router_bias"]
        graph.nodes.append(LevelNode(
            node_id="m-router-bias", label="e_score_correction_bias",
            sublabel=f"{_shape_of(bias)}；仅参与 TopK 选择，不参与权重",
            tag="路由权重/buffer", component_type="external",
            sources=_sources(next((op for op in block.get("operators",[])
                                   if op.get("id")=="moe_router"),{})),
            role="aux", anchor="m-router",
        ))
        graph.edges.append(LevelEdge(
            src="m-router-bias", dst="m-router", tensor_id="router_bias",
            tensor_label="e_score_correction_bias", shape=_shape_of(bias) or "标量",
            kind="data", label="仅选择，不参与权重",
        ))
        _assert_acyclic(graph)
    router_op=next((op for op in block.get("operators",[]) if op.get("id")=="moe_router"),{})
    for node in graph.nodes:
        if node.node_id=="m-router":
            node.sublabel=(
                "logits→score_fn=sqrtsoftplus(scores)→+e_score_correction_bias→topk indices"
                "→scores.gather(原始 scores)→/(sum+1e-20)→×Scl"
                if not is_hash else
                "logits→score_fn=sqrtsoftplus(scores)→tid2eid[input_ids] indices"
                "→scores.gather(原始 scores)→/(sum+1e-20)→×Scl")
    graph.parameter_items=_filter_params(block,("B","seqlen","H","E","K","Scl"))
    return graph


def build_moe_expert_graph(
    block: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
) -> LevelGraph:
    gid=str(group["id"])
    stages=(
        ("e-slice","当前命中专家 gate_up/down 切片",("moe_expert_slice",)),
        ("e-gu","gate_up_proj[e] 融合 Linear",("moe_expert_gu",)),
        ("e-clamp","chunk + gate 上界/up 双侧 Clamp",("moe_expert_clamp",)),
        ("e-act","实际激活 × up",("moe_expert_act",)),
        ("e-down","down_proj[e]",("moe_expert_down",)),
        ("e-weight","按 token_idx/top_k_pos 取路由权重并乘",("moe_expert_weight",)),
    )
    graph=_stage_graph(block,stages=stages,key=f"moe-expert-{gid}",level="moe-expert",
        title="DeepSeek V4 · 单激活专家内部（模板一份）",
        subtitle=f"代表层 {group.get('representative_layer_index')} · 仅真实命中专家循环",
        entry_tensor="expert_tokens",variant=gid,group=gid)
    graph.parameter_items=_filter_params(block,("B","seqlen","E","K","H","Es","Lim"))
    # 真实 weights 字段：expert_gate_up_e/expert_down_e 来自 slice 阶段。
    tensors=_tensor_index(block); ops={o["id"]:o for o in block.get("operators",[])}
    for op_id in ("moe_expert_gu","moe_expert_down"):
        for wid in ops[op_id].get("weights",[]):
            t=tensors.get(wid)
            if t is None: continue
            src="e-slice" if wid in ("expert_gate_up_e","expert_down_e") else "s-in"
            dst="e-gu" if op_id=="moe_expert_gu" else "e-down"
            if not any(e.src==src and e.dst==dst and e.tensor_id==wid for e in graph.edges):
                graph.edges.append(LevelEdge(src=src,dst=dst,tensor_id=wid,
                    tensor_label=_tensor_name(t,wid),shape=_shape_of(t) or "标量",kind="data",
                    label=_shape_of(t) or "标量",condition="真实 weights 字段消费（expert slice）"))
    _assert_acyclic(graph)
    return graph


def build_moe_shared_graph(
    block: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
) -> LevelGraph:
    gid=str(group["id"])
    graph=_stage_graph(block,stages=(
        ("sh-gate-up","共享 gate 上界 Clamp / up 双侧 Clamp（独立投影）",("shared_gate_up",)),
        ("sh-act-down","SiLU(gate)×up → down_proj",("shared_act_down",)),
    ),key=f"moe-shared-{gid}",level="moe-shared",
      title="DeepSeek V4 · 共享专家（全 token，gate/up Clamp）",
      subtitle=f"代表层 {group.get('representative_layer_index')} · gate clamp(max=Lim)；up clamp(±Lim)；全 token，不消费 expert_tokens",
      entry_tensor="ffn_normed",variant=gid,group=gid)
    for node in graph.nodes:
        if node.node_id=="sh-gate-up":
            node.sublabel="gate_proj→min(·,Lim) / up_proj→clamp(±Lim)；两个独立 Linear"
        elif node.node_id=="sh-act-down":
            node.sublabel="SiLU(gate)·up → down_proj；输出 shared_out [B,seqlen,H]"
    graph.parameter_items=_filter_params(block,("B","seqlen","H","Es","Lim"))
    return graph


def build_moe_combine_graph(
    block: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
) -> LevelGraph:
    gid=str(group["id"])
    graph=_stage_graph(block,stages=(
        ("c-add","moe_combine · index_add 回填（final=zeros_like 起步）",("moe_combine",)),
        ("c-view","routed_out → View [B,seqlen,H]",("moe_routed_view",)),
        ("c-merge","routed_view + shared_out → mlp_output",("moe_merge",)),
    ),key=f"moe-combine-{gid}",level="moe-combine",
      title="DeepSeek V4 · routed 零初值 index_add、View 与 shared 加和",
      subtitle=f"代表层 {group.get('representative_layer_index')} · expert_down_out+token_idx → routed_out → View → +shared",
      entry_tensor="expert_down_out",variant=gid,group=gid)
    graph.notes=graph.notes+(
        "moe_combine 是真实 index_add：final=zeros_like(ffn_flat) 起步，按 expert_token_idx 对命中专家循环累加 expert_down_out（modeling:1041-1055）。",
        "路由权重已在单专家模板 moe_expert_weight 中乘过一次；combine 不再消费 topk_weights。",
    )
    for node in graph.nodes:
        if node.node_id=="c-add":
            node.sublabel="index_add_(0, token_idx, expert_down_out)；零初值、命中专家累加"
        elif node.node_id=="c-view":
            node.sublabel="routed.view(B,seqlen,H)（:1133）"
        elif node.node_id=="c-merge":
            node.sublabel="routed_view + shared_out（:1134）"
    graph.parameter_items=_filter_params(block,("B","seqlen","H"))
    return graph


def build_mhc_graph(
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
) -> LevelGraph:
    """mHC 权重细图：只展开两段 mHC.forward，不把 Attention/FFN 画进 mHC。

    - 两段各 7 个具名算子：flatten/norm、project、split、pre、post、comb+Sinkhorn、
      collapse；
    - 每段显式返回 collapsed/post/comb，消费者是 Decoder.forward 的边界节点；
    - 分支调用与残差展开/combᵀ/Add 不在本图内，避免虚构 mHC 内部回调。
    """
    gid = str(group["id"])
    tensors = _tensor_index(block)
    segment_modules = {
        "attn": next((m for m in modules if m["id"] == "mhc_attn"), None),
        "ffn": next((m for m in modules if m["id"] == "mhc_ffn"), None),
    }
    segment_modules = {
        name: module for name, module in segment_modules.items() if module is not None
    }
    if not segment_modules:
        raise ArchifyLayersError(f"层组 {gid} 缺少 mhc_attn/mhc_ffn 模块")
    ops = [op for module in segment_modules.values() for op in module.get("ops", [])]
    members = {str(op["id"]) for op in ops}
    producer = {
        str(tensor_id): str(op["id"])
        for op in block.get("operators", [])
        for tensor_id in op.get("outputs", [])
    }
    weight_tensors = {
        str(tensor.get("id"))
        for tensor in block.get("tensors", [])
        if str(tensor.get("kind")) == "weight"
    }
    graph = LevelGraph(
        key=f"mhc-{gid}",
        level="mhc",
        title=f"DeepSeek V4 · {group.get('label', gid)} mHC.forward 权重与门控",
        subtitle=(
            f"代表层 {group.get('representative_layer_index')} · "
            f"两段各 7 算子 · 返回 post/comb/collapsed；"
            "分支调用与残差合并在 Decoder.forward"
        ),
        notes=(
            "mHC.forward 只返回 post/comb/collapsed；本图不包含 Attention/FFN 调用，"
            "也不包含 Decoder.forward 的残差展开、combᵀ 或 Add。",
        ),
        parameter_items=_mhc_parameter_items(block),
        variant=gid,
        group=gid,
    )
    graph.nodes.append(
        LevelNode(
            node_id="o-in",
            label="段输入（两段 mHC 的流输入）",
            sublabel="hidden_streams_in；attn 段混合输出 attn_out_streams",
            tag="输入张量",
            component_type="external",
            sources=_sources(ops[0]),
            role="boundary_in",
            details=(
                "attn 段输入 hidden_streams_in（batch_size,seqlen,hc_mult,hidden_size）",
                "ffn 段输入 attn_out_streams（attn 段残差混合后的多流）",
            ),
        )
    )
    node_ids = {"o-in"}
    for op in ops:
        node_id = f"o-{op['id']}"
        node_ids.add(node_id)
        comb_loop = str(op["id"]).endswith("_hc_comb")
        label = _op_label(op)
        sublabel = _op_sublabel(op)
        details = (
            str(op.get("note", "")),
            "mHC.forward 内部算子；不调用 Attention/FFN。",
        )
        if comb_loop:
            label = "comb 混合（softmax+Sinkhorn）"
            sublabel = (
                "affine+softmax+sinkhorn · 1 次初始列归一 + "
                "(Sit-1) 次 [行归一→列归一]"
            )
            details = details + (
                "Sinkhorn 前向循环：先做 1 次列归一；再重复 "
                "(hc_sinkhorn_iters-1)=19 次 [行归一→列归一]；"
                "全称参数 Sit=hc_sinkhorn_iters=20，Eps=hc_eps=1e-06。",
                "循环在本节点内部前向完成，不复制 20 份节点、无向上回流。",
            )
        # pre 门控留在主链；post/comb 是 split 的两个并列输出，放旁枝竖列，
        # 避免 rank 布局把两段 mHC 的 4 个门控并排拉成超宽画布。
        gate_branch = str(op["id"]).endswith(("_post", "_comb"))
        split_anchor = f"o-{str(op['id']).rsplit('_', 1)[0]}_split"
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=label,
                sublabel=sublabel,
                tag=_node_tag(op.get("module_path", ""), "mhc"),
                component_type="backend",
                sources=_sources(op),
                role="branch" if gate_branch else "main",
                anchor=split_anchor if gate_branch else None,
                details=details,
                op_ids=(str(op["id"]),),
            )
        )
    consumers = {
        "attn": ("o-consumer-attn", "attn_hc_collapse", "Attention"),
        "ffn": ("o-consumer-ffn", "ffn_hc_collapse", "MoE"),
    }
    consumers = {segment: spec for segment, spec in consumers.items()
                 if segment in segment_modules}
    for segment, (node_id, collapse_op, branch_name) in consumers.items():
        collapse = next(
            (op for op in ops if str(op["id"]) == collapse_op), ops[0]
        )
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=f"Decoder.forward：mHC 返回的三路 → {branch_name} 分支/残差",
                sublabel="collapsed → Norm → 分支；post → 展开；comb → combᵀ + Add",
                tag="Decoder.forward 消费（不在 mHC.forward）",
                component_type="external",
                sources=_sources(collapse),
                role="aux",
                anchor=f"o-{collapse_op}",
                details=(
                    f"段 {segment}：collapsed 交 Norm→{branch_name}；"
                    "post 与分支输出展开；comb 由 Decoder.forward 转置后混合原始多流。",
                    "本节点是 Decoder.forward 的消费边界，不是 mHC 内部对分支的调用。",
                ),
            )
        )
    for op in ops:
        for tensor_id in op.get("inputs", []):
            tensor_id = str(tensor_id)
            if tensor_id in weight_tensors:
                continue
            tensor = tensors.get(tensor_id)
            src = producer.get(tensor_id)
            if src in members:
                graph.edges.append(
                    LevelEdge(
                        src=f"o-{src}",
                        dst=f"o-{op['id']}",
                        tensor_id=tensor_id,
                        tensor_label=_tensor_name(tensor, tensor_id),
                        shape=_shape_of(tensor),
                        kind="data",
                    )
                )
            else:
                graph.edges.append(
                    LevelEdge(
                        src="o-in",
                        dst=f"o-{op['id']}",
                        tensor_id=tensor_id,
                        tensor_label=_tensor_name(tensor, tensor_id),
                        shape=_shape_of(tensor),
                        kind="data",
                        label="段输入" if tensor_id in (
                            "hidden_streams_in", "attn_out_streams") else "外部输入",
                        condition=(
                            f"生产者 {producer.get(tensor_id)} 在 mHC 视图外"
                            if tensor_id in producer
                            else "调用方/段外输入"
                        ),
                    )
                )
    output_specs = {
        "attn": (
            ("attn_mhc_collapsed", "collapsed"),
            ("attn_mhc_post", "post"),
            ("attn_mhc_comb", "comb"),
        ),
        "ffn": (
            ("ffn_mhc_collapsed", "collapsed"),
            ("ffn_mhc_post", "post"),
            ("ffn_mhc_comb", "comb"),
        ),
    }
    output_specs = {segment: records for segment, records in output_specs.items()
                    if segment in segment_modules}
    for segment, records in output_specs.items():
        consumer_id = consumers[segment][0]
        for tensor_id, label in records:
            tensor = tensors.get(tensor_id)
            src = producer.get(tensor_id)
            if src not in members:
                continue
            graph.edges.append(
                LevelEdge(
                    src=f"o-{src}",
                    dst=consumer_id,
                    tensor_id=tensor_id,
                    tensor_label=_tensor_name(tensor, tensor_id),
                    shape=_shape_of(tensor),
                    kind="data",
                    label=label,
                    condition=f"mHC.forward 第 {segment} 段返回；消费者 Decoder.forward",
                )
            )
    _assert_acyclic(graph)
    return graph


def _ops_graph(
    block: Mapping[str, Any],
    ops: Sequence[Mapping[str, Any]],
    *,
    key: str,
    level: str,
    title: str,
    subtitle: str,
    variant: str,
    group: str,
    entry_tensor: str,
    exit_tensor: str,
    extra_nodes: Sequence[LevelNode] = (),
    extra_edges: Sequence[LevelEdge] = (),
    member_ops: Sequence[Mapping[str, Any]] | None = None,
    source_prefix: str = _REPO_PREFIX,
    state_cache: bool = False,
) -> LevelGraph:
    """算子级段图：显式入口/出口 + 全部跨边界张量（含视图外生产者）。

    - ``o-in`` 收本段全部**外部激活输入**（由段外算子产生或段外传入），逐条保留
      真实张量身份、全称形状与来源；权重不作为数据流边。
    - ``o-out`` 只接 ``exit_tensor`` 的真实生产者（没有块内消费者也必须连上）。
    - 其它“段内产生、段外消费”的张量各自有辅助边界节点，不冒充主输出。
    """
    tensors = _tensor_index(block)
    graph = LevelGraph(
        key=key,
        level=level,
        title=title,
        subtitle=subtitle,
        notes=tuple(block.get("cache_notes", []))[:2]
        + tuple(block.get("coverage_notes", []))[:2],
        parameter_items=_parameter_items(block),
        variant=variant,
        group=group,
    )
    members = {str(op["id"]) for op in ops}
    member_docs = list(member_ops if member_ops is not None else ops)
    producer = {
        str(tensor_id): str(op["id"])
        for op in block.get("operators", [])
        for tensor_id in op.get("outputs", [])
    }
    consumer = {
        str(tensor_id): str(op["id"])
        for op in block.get("operators", [])
        for tensor_id in op.get("inputs", [])
    }
    entry = tensors.get(entry_tensor)
    external_inputs = sorted(
        {
            str(tensor_id)
            for op in ops
            for tensor_id in op.get("inputs", [])
            if producer.get(str(tensor_id)) not in members
            and str(tensors.get(str(tensor_id), {}).get("kind")) != "weight"
            and not (
                state_cache
                and str(tensors.get(str(tensor_id), {}).get("kind")) == "cache"
            )
        }
    )
    # cos/sin/mask/position 等辅助输入：不再汇入顶部 o-in，而是每个消费点一条
    # 局部短旁枝；同一 tensor 的多个消费点各自有同源引用，真实 tensor id/shape 保留。
    aux_entries: list[tuple[Mapping[str, Any], str]] = []
    aux_index: dict[tuple[str, str], int] = {}
    if state_cache:
        for op in ops:
            for tid in op.get("inputs", []):
                tid = str(tid)
                if tid == entry_tensor or producer.get(tid) in members:
                    continue
                tensor = tensors.get(tid)
                if str(tensor.get("kind") if tensor else "") not in ("boundary", "mask"):
                    continue
                key = (str(op["id"]), tid)
                if key not in aux_index:
                    aux_index[key] = len(aux_entries)
                    aux_entries.append((op, tid))
    aux_tensor_ids = {tid for _op, tid in aux_entries}
    inline_inputs = [tid for tid in external_inputs if tid not in aux_tensor_ids]
    graph.nodes.append(
        LevelNode(
            node_id="o-in",
            label="段输入（Block 上游张量）",
            sublabel=(
                f"入口 {entry_tensor} {_shape_of(entry)}".strip()
                if entry_tensor
                else "段输入"
            ),
            tag="输入张量",
            component_type="external",
            sources=_sources(ops[0] if ops else member_docs[0], prefix=source_prefix),
            role="boundary_in",
            details=tuple(
                f"{_tensor_name(tensors.get(tid), tid)} {_shape_of(tensors.get(tid))}"
                f"（生产者 {producer.get(tid) or '段外/调用方'}）"
                for tid in external_inputs
            ),
        )
    )
    for op in member_docs:
        graph.nodes.append(
            LevelNode(
                node_id=f"o-{op['id']}",
                label=_op_label(op),
                sublabel=_op_sublabel(op),
                tag=_node_tag(op.get("module_path", ""), level),
                component_type="backend",
                sources=_sources(op, prefix=source_prefix),
                role="main",
                details=(str(op.get("note", "")),),
                op_ids=(str(op["id"]),),
            )
        )
    for index, (consumer_op, tid) in enumerate(aux_entries):
        tensor = tensors.get(tid)
        graph.nodes.append(
            LevelNode(
                node_id=f"o-aux-in-{index}",
                label=_tensor_name(tensor, tid),
                sublabel=_shape_of(tensor),
                tag="辅助输入（局部短旁枝）",
                component_type="external",
                sources=_sources(consumer_op, prefix=source_prefix),
                role="aux",
                anchor=f"o-{consumer_op['id']}",
                details=(
                    f"tensor {tid}：同一张量在多个消费点各有一处局部引用；"
                    f"生产者 {producer.get(tid) or '段外/调用方'}",
                ),
            )
        )
    graph.nodes.extend(extra_nodes)
    exit_tensor_doc = tensors.get(exit_tensor)
    graph.nodes.append(
        LevelNode(
            node_id="o-out",
            label="段输出（本层接口张量）",
            sublabel=f"{_tensor_name(exit_tensor_doc, exit_tensor)} {_shape_of(exit_tensor_doc)}",
            tag="输出张量",
            component_type="external",
            sources=_sources(
                next(
                    (op for op in member_docs if exit_tensor in op.get("outputs", ())),
                    member_docs[-1],
                )
            ),
            role="boundary_out",
        )
    )
    # 段内算子之间的边 + 每个跨边界输入一条 o-in 边（不跳过、不合并）。
    for op in ops:
        for tensor_id in op.get("inputs", []):
            tensor_id = str(tensor_id)
            src = producer.get(tensor_id)
            tensor = tensors.get(tensor_id)
            if src in members:
                graph.edges.append(
                    LevelEdge(
                        src=f"o-{src}",
                        dst=f"o-{op['id']}",
                        tensor_id=tensor_id,
                        tensor_label=_tensor_name(tensor, tensor_id),
                        shape=_shape_of(tensor) or "标量",
                        kind="data",
                    )
                )
            elif str(tensor.get("kind") if tensor else "") == "weight":
                continue
            elif state_cache and str(tensor.get("kind") if tensor else "") == "cache":
                continue
            elif (str(op["id"]), tensor_id) in aux_index:
                node_id = f"o-aux-in-{aux_index[(str(op['id']), tensor_id)]}"
                graph.edges.append(
                    LevelEdge(
                        src=node_id,
                        dst=f"o-{op['id']}",
                        tensor_id=tensor_id,
                        tensor_label=_tensor_name(tensor, tensor_id),
                        shape=_shape_of(tensor),
                        kind="aux",
                        label="辅助输入（同源局部引用）",
                        condition=(
                            f"同一 tensor {tensor_id} 的独立消费点；"
                            "生产者见节点详情/来源"
                        ),
                    )
                )
            else:
                graph.edges.append(
                    LevelEdge(
                        src="o-in",
                        dst=f"o-{op['id']}",
                        tensor_id=tensor_id,
                        tensor_label=_tensor_name(tensor, tensor_id),
                        shape=_shape_of(tensor) or "标量",
                        kind="data",
                        label="段输入",
                        condition=(
                            f"生产者 {producer[tensor_id]} 在视图中（段外）"
                            if tensor_id in producer
                            else "段外/调用方输入"
                        ),
                    )
                )
    graph.edges.extend(extra_edges)
    # 主出口：只连 exit_tensor 的真实生产者（即使块内没有消费者）。
    exit_producers = [
        op for op in member_docs if exit_tensor in op.get("outputs", ())
    ]
    if exit_producers:
        for op in exit_producers:
            graph.edges.append(
                LevelEdge(
                    src=f"o-{op['id']}",
                    dst="o-out",
                    tensor_id=exit_tensor,
                    tensor_label=_tensor_name(exit_tensor_doc, exit_tensor),
                    shape=_shape_of(exit_tensor_doc),
                    kind="data",
                )
            )
    # 辅助出口：段内产生、段外消费的中间张量逐个建边界节点（不冒充主输出）。
    aux_outputs = sorted(
        {
            str(tensor_id)
            for op in ops
            for tensor_id in op.get("outputs", [])
            if str(tensor_id) != exit_tensor
            and str(tensors.get(str(tensor_id), {}).get("kind")) != "weight"
            and not (
                state_cache
                and str(tensors.get(str(tensor_id), {}).get("kind")) == "cache"
            )
            and any(
                str(candidate["id"]) not in members
                for candidate in block.get("operators", [])
                if str(tensor_id) in candidate.get("inputs", [])
            )
        }
    )
    for tensor_id in aux_outputs:
        tensor = tensors.get(tensor_id)
        node_id = f"o-aux-{_slug(tensor_id)}"
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=_tensor_name(tensor, tensor_id),
                sublabel=_shape_of(tensor),
                tag="段外消费（同一张量）",
                component_type="external",
                sources=_sources(
                    next(
                        (
                            op
                            for op in member_docs
                            if tensor_id in op.get("outputs", ())
                        ),
                        member_docs[-1],
                    )
                ),
                role="branch",
                anchor=f"o-{producer[tensor_id]}",
                details=(
                    f"段内由 {producer[tensor_id]} 产生，段外消费者 "
                    f"{consumer.get(tensor_id, '（块外）')} 读取；不并入主出口",
                ),
            )
        )
        graph.edges.append(
            LevelEdge(
                src=f"o-{producer[tensor_id]}",
                dst=node_id,
                tensor_id=tensor_id,
                tensor_label=_tensor_name(tensor, tensor_id),
                shape=_shape_of(tensor),
                kind="data",
                label="段外消费",
            )
        )
    if state_cache:
        nodes, edges = _segment_state_nodes(block, ops, source_prefix=source_prefix)
        graph.nodes.extend(nodes)
        graph.edges.extend(edges)
    return graph


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------


def _ranks(graph: LevelGraph) -> dict[str, int]:
    """秩 = 主链节点在**强连通分量缩合图**上的最长路径深度。

    模块级投影可能让两个模块互相依赖（例如 mask 模块读取注意力拼接后的 KV、
    同时向注意力核心提供 mask）。真实依赖不能删，所以把互依模块放进同一 SCC、
    排在同一行（横向连线），其余照常自上而下；这样既不隐藏边，也不制造回流。
    """
    main_ids = {
        node.node_id
        for node in graph.nodes
        if node.role in ("boundary_in", "main", "boundary_out")
    }
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in main_ids}
    for edge in graph.edges:
        if edge.kind == "data" and edge.src in main_ids and edge.dst in main_ids:
            adjacency[edge.src].append(edge.dst)
    components = _strongly_connected(adjacency)
    comp_of = {
        node_id: index
        for index, members in enumerate(components)
        for node_id in members
    }
    comp_edges = {
        (comp_of[src], comp_of[dst])
        for src, targets in adjacency.items()
        for dst in targets
        if comp_of[src] != comp_of[dst]
    }
    rank = {index: 0 for index in range(len(components))}
    for _ in range(len(components) + 1):
        changed = False
        for src, dst in comp_edges:
            if rank[dst] < rank[src] + 1:
                rank[dst] = rank[src] + 1
                changed = True
        if not changed:
            break
    else:  # pragma: no cover - SCC 缩合后必为 DAG
        raise ArchifyLayersError(f"图 {graph.key} 的缩合图仍有环")
    return {node_id: rank[comp_of[node_id]] for node_id in main_ids}


def _strongly_connected(adjacency: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """Tarjan SCC（迭代实现，顺序稳定，便于确定性产物）。"""
    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0
    for root in adjacency:
        if root in index_of:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, position = work.pop()
            if position == 0:
                index_of[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True
            targets = adjacency.get(node, ())
            recursed = False
            for offset in range(position, len(targets)):
                target = targets[offset]
                if target not in index_of:
                    work.append((node, offset + 1))
                    work.append((target, 0))
                    recursed = True
                    break
                if on_stack.get(target):
                    low[node] = min(low[node], index_of[target])
            if recursed:
                continue
            if low[node] == index_of[node]:
                members: list[str] = []
                while True:
                    item = stack.pop()
                    on_stack[item] = False
                    members.append(item)
                    if item == node:
                        break
                components.append(sorted(members))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return components


def layout_level(
    graph: LevelGraph, *, sizing: LevelSizing | None = None
) -> dict[str, tuple[float, float, float, float]]:
    """单向秩布局：主链按秩自上而下，分叉向右、列随父链继承，辅助/状态紧贴消费节点。"""
    size = sizing or LevelSizing()
    rank = _ranks(graph)
    by_rank: dict[int, list[str]] = {}
    for node in graph.nodes:
        if node.node_id in rank:
            by_rank.setdefault(rank[node.node_id], []).append(node.node_id)
    order = {node.node_id: index for index, node in enumerate(graph.nodes)}
    parents: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind != "data" or edge.src not in rank or edge.dst not in rank:
            continue
        if rank[edge.src] < rank[edge.dst]:
            parents.setdefault(edge.dst, []).append(edge.src)
    positions: dict[str, tuple[float, float, float, float]] = {}
    column: dict[str, int] = {}
    for level in sorted(by_rank):
        ids = sorted(by_rank[level], key=lambda item: order[item])
        taken: set[int] = set()
        pending: list[str] = []
        # 先让“父链所在列空着”的节点继承父列，主链因此保持一条直线。
        for node_id in ids:
            inherited = None
            for parent in parents.get(node_id, ()):
                candidate = column.get(parent)
                if candidate is not None and candidate not in taken:
                    inherited = candidate
                    break
            if inherited is None:
                pending.append(node_id)
            else:
                column[node_id] = inherited
                taken.add(inherited)
        cursor = 0
        for node_id in pending:
            while cursor in taken:
                cursor += 1
            column[node_id] = cursor
            taken.add(cursor)
            cursor += 1
        y = _Y0 + level * size.step_y
        for node_id in ids:
            x = size.main_x + column[node_id] * size.col_pitch
            positions[node_id] = (x, y, size.main_w, size.node_h)
    # 辅助/状态节点：贴住其消费节点（同秩左侧），冲突时下移，保持短旁枝。
    side_nodes = [
        node for node in graph.nodes if node.role in ("aux", "state", "branch")
    ]
    used_y: list[tuple[float, float]] = []
    for node in side_nodes:
        anchor = node.anchor or next(
            (
                edge.dst
                for edge in graph.edges
                if edge.src == node.node_id and edge.dst in positions
            ),
            None,
        )
        if anchor is None or anchor not in positions:
            raise ArchifyLayersError(f"旁枝节点 {node.node_id} 没有可定位的锚点")
        anchor_rect = positions[anchor]
        y = anchor_rect[1]
        while any(
            not (y + size.side_h + 8 <= start or y >= start + height + 8)
            for start, height in used_y
        ):
            y += size.side_h + 14
        used_y.append((y, size.side_h))
        positions[node.node_id] = (_LEFT_X, y, size.side_w, size.side_h)
    # 硬约束：数据边只能向下或同秩横向，不得回到更上一个秩。
    for edge in graph.edges:
        if edge.kind != "data":
            continue
        if edge.src in rank and edge.dst in rank and rank[edge.dst] < rank[edge.src]:
            raise ArchifyLayersError(
                f"数据边 {edge.src} → {edge.dst} 在布局中向上回指"
                f"（秩 {rank[edge.src]} → {rank[edge.dst]}）"
            )
    for edge in graph.edges:
        if edge.kind != "state":
            continue
        src = positions.get(edge.src)
        dst = positions.get(edge.dst)
        if src is None or dst is None:
            continue
        if src[0] + src[2] <= size.main_x and dst[0] + dst[2] <= size.main_x:
            if src[1] >= dst[1]:
                raise ArchifyLayersError(
                    f"状态边 {edge.src} → {edge.dst} 在布局中向上回指"
                    f"（源 y={src[1]}, 目标 y={dst[1]}）"
                )
    if not positions:  # pragma: no cover - 空图不可交付
        raise ArchifyLayersError(f"图 {graph.key} 没有可排布节点")
    return positions


def _lane_route(
    positions: Mapping[str, tuple[float, float, float, float]],
    lane_x: float,
    src: tuple[float, float, float, float],
    dst: tuple[float, float, float, float],
) -> list[list[float]]:
    """跨秩关系：右侧竖井 + 目标行上方空隙横穿，避开所有组件。

    路径：源底部 → 下一行空隙右移 → 竖井下移 → 目标行上方横移到目标列 → 进入目标顶部。
    横段只落在行间空隙，不穿过任何组件；竖井在最右列之外。
    """
    src_cx = round(src[0] + src[2] / 2, 1)
    dst_cx = round(dst[0] + dst[2] / 2, 1)
    below_src = round(src[1] + src[3] + 16, 1)
    above_dst = round(dst[1] - 18, 1)
    return [
        [src_cx, below_src],
        [round(lane_x, 1), below_src],
        [round(lane_x, 1), above_dst],
        [dst_cx, above_dst],
    ]


def _sides(
    src: tuple[float, float, float, float],
    dst: tuple[float, float, float, float],
) -> tuple[str, str]:
    src_cx = src[0] + src[2] / 2
    dst_cx = dst[0] + dst[2] / 2
    src_cy = src[1] + src[3] / 2
    dst_cy = dst[1] + dst[3] / 2
    if abs(src_cx - dst_cx) < 40 and dst[1] >= src[1] + src[3]:
        return "bottom", "top"
    if dst[1] + dst[3] <= src[1]:
        return "top", "bottom"
    if abs(src_cy - dst_cy) < 30 and abs(src_cx - dst_cx) >= 40:
        return ("right", "left") if src_cx < dst_cx else ("left", "right")
    return ("bottom", "top") if dst_cy > src_cy else ("top", "bottom")


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


def _component_id(node_id: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z_-]+", "-", node_id).strip("-").lower()
    return f"n-{slug}"[:64]


def _edge_label_box_width(text: str) -> float:
    """Archify 连线标签盒宽度：取实际文字宽度与 textUnits*4.8+10 的较大值。"""
    return max(_edge_label_width(text), _archify_text_units(text) * 4.8 + 10.0)


def _edge_label_width(text: str) -> float:
    """连线标签的实际文字宽度估算（与浏览器探针的字体度量接近）。

    Archify 的 architecture 连线标签固定 8px；CJK 约 14.2px/字，ASCII 约
    4.16px/字。用它做去重叠预算，实际 SVG bbox 可能有 1px 级差异。
    """
    width = 0.0
    for char in str(text or ""):
        width += 14.2 if ord(char) > 0x7F else 4.16
    return width


def _rects_overlap(a, b, margin: float = 0.0) -> bool:
    return not (
        a[0] + a[2] + margin <= b[0]
        or a[0] >= b[0] + b[2] + margin
        or a[1] + a[3] + margin <= b[1]
        or a[1] >= b[1] + b[3] + margin
    )


def _edge_display_text(edge: LevelEdge, graph: LevelGraph) -> str:
    """主图可见文字：Decoder/mHC 边同时给出身份与全称 shape。"""
    identity = str(edge.label or "").strip()
    shape = str(edge.shape or "").strip()
    if graph.level.startswith("moe"):
        tensor_identity = str(edge.tensor_label or "").strip() or identity
        if tensor_identity and shape:
            return f"{tensor_identity}｜{shape}"
        return tensor_identity or shape
    if graph.key.startswith(("decoder-", "mhc-")) and identity and shape:
        if identity not in ("数据依赖", "段输入", "辅助输入"):
            return f"{identity}｜{shape}"
    return shape or identity


def separate_edge_labels(
    connections: list[dict[str, Any]],
    positions: Mapping[str, tuple[float, float, float, float]],
    obstacles: list[tuple[float, float, float, float]] | None = None,
    *,
    max_right: float | None = None,
) -> dict[str, Any]:
    """把同排/平行/汇入的连线标签错开，避免互相遮挡。

    - 两条端点的竖直边：标签只能在该行间隙内上下错层；
    - 跨层/跨列显式 labelAt 的边：可在锚点附近较大竖向范围内错层；
    - 每个候选位置都要避开所有节点矩形与已放置标签；
    - 不删除标签、不缩字、不改文字内容；只改 labelAt（并移除 labelDx/Dy）。
    """
    specs: list[dict[str, Any]] = []
    for conn in connections:
        text = str(conn.get("label") or "")
        if not text:
            continue
        width = _edge_label_box_width(text)
        if "labelAt" in conn:
            x, y = conn["labelAt"]
            mode = "explicit"
        elif conn.get("via"):
            points = [tuple(point) for point in conn.get("via") or () if len(point) == 2]
            if not points:
                continue
            x = (points[0][0] + points[-1][0]) / 2.0
            y = (points[0][1] + points[-1][1]) / 2.0
            mode = "explicit"
        else:
            src = positions.get(conn.get("from"))
            dst = positions.get(conn.get("to"))
            if src is None or dst is None:
                continue
            x = (src[0] + src[2] / 2 + dst[0] + dst[2] / 2) / 2
            y = src[1] + src[3] - 10 + float(conn.get("labelDy") or 0)
            mode = "vertical"
        specs.append(
            {
                "conn": conn,
                "text": text,
                "width": width,
                "height": 14.0,
                "x": float(x),
                "y": float(y),
                "mode": mode,
            }
        )
    # 长标签先放，给它们优先选择空间。
    specs.sort(key=lambda item: (-item["width"], item["y"], item["x"]))
    placed: list[tuple[float, float, float, float]] = []
    unresolved: list[str] = []
    for spec in specs:
        conn = spec["conn"]
        width = spec["width"]
        height = spec["height"]
        base_x, base_y = spec["x"], spec["y"]
        if spec["mode"] == "vertical":
            src = positions.get(conn.get("from"))
            dst = positions.get(conn.get("to"))
            if src is None or dst is None:
                continue
            y_min = src[1] + src[3] + 4
            y_max = dst[1] - 4
            if y_min >= y_max:
                # 同排/辅助短边（cos/sin → RoPE 等）没有行间空隙，允许在
                # 锚点附近上下错层，而不是直接回退到压节点的位置。
                y_min = base_y - 120
                y_max = base_y + 120
        else:
            # 跨层/汇入标签贴着各自路径：允许在路径 bbox 附近上下错层；
            # 只有目标是画布最底部的输出节点时，才允许继续向下堆叠。
            src = positions.get(conn.get("from"))
            dst = positions.get(conn.get("to"))
            if src is not None and dst is not None:
                def _side_point(rect, side):
                    if side == "bottom":
                        return (rect[0] + rect[2] / 2, rect[1] + rect[3])
                    if side == "top":
                        return (rect[0] + rect[2] / 2, rect[1])
                    if side == "left":
                        return (rect[0], rect[1] + rect[3] / 2)
                    return (rect[0] + rect[2], rect[1] + rect[3] / 2)

                path_y = [
                    _side_point(src, conn.get("fromSide") or "bottom")[1],
                    *[float(point[1]) for point in conn.get("via") or () if len(point) == 2],
                    _side_point(dst, conn.get("toSide") or "top")[1],
                ]
                path_min, path_max = min(path_y), max(path_y)
                canvas_bottom = max(rect[1] + rect[3] for rect in positions.values())
                bottom_target = dst[1] + dst[3] >= canvas_bottom - 10
                y_min = min(base_y - 80, path_min - 40)
                y_max = max(
                    base_y + 120,
                    path_max + (180 if bottom_target else 40),
                )
            else:
                y_min = base_y - 160
                y_max = base_y + 260
        x_offsets = (
            0.0, width / 2 + 8, -(width / 2 + 8),
            width / 2 + 20, -(width / 2 + 20),
            width / 2 + 60, -(width / 2 + 60),
            width / 2 + 120, -(width / 2 + 120),
            width / 2 + 200, -(width / 2 + 200),
            width / 2 + 300, -(width / 2 + 300),
        )
        # 先在小范围上下错层（保留与边的近距离），再水平让开节点文字，
        # 最后才用较大的上下偏移。避免为了避开节点标题把标签推到边路径之外。
        placements: list[tuple[float, float]] = []
        # 先做“小范围错层 × 水平列”的网格，让同一汇入点的多条标签
        # 在目标附近排成 2–3 行，而不是被推到路径端点之外。
        for step in (0, 1, 2):
            for sign in (1, -1) if step else (1,):
                for dx in x_offsets:
                    placements.append((dx, base_y + sign * step * 14))
        for step in range(3, 31):
            for sign in (1, -1):
                for dx in x_offsets:
                    placements.append((dx, base_y + sign * step * 14))
        chosen = None
        chosen_rect = None
        best_score = float("inf")
        for dx, y in placements:
            if y < y_min or y > y_max:
                continue
            rect = (
                round(base_x + dx - width / 2, 1),
                round(y - 9, 1),
                round(width, 1),
                height,
            )
            if rect[0] < 2:
                continue
            if max_right is not None and rect[0] + rect[2] > max_right:
                continue
            if any(_rects_overlap(rect, other, 2.0) for other in positions.values()):
                continue
            if obstacles and any(_rects_overlap(rect, other, 2.0) for other in obstacles):
                continue
            if any(_rects_overlap(rect, other, 1.0) for other in placed):
                continue
            score = abs(y - base_y) + abs(dx) * 0.25
            if score < best_score:
                best_score = score
                chosen = (base_x + dx, y)
                chosen_rect = rect
        if chosen is None:
            unresolved.append(str(conn.get("id")))
            chosen = (base_x, base_y)
            chosen_rect = (
                round(base_x - width / 2, 1), round(base_y - 9, 1),
                round(width, 1), height,
            )
        conn["labelAt"] = [round(chosen[0], 1), round(chosen[1], 1)]
        conn.pop("labelDx", None)
        conn.pop("labelDy", None)
        placed.append(chosen_rect)
    return {"labels": len(placed), "unresolved": unresolved}


def build_level_candidate(
    graph: LevelGraph,
    *,
    output_relpath: str,
    repository: Mapping[str, Any] | None,
    tool: Mapping[str, Any] | None,
    quality: str = "standard",
    source_evidence: bool = True,
    sizing: LevelSizing | None = None,
    complete_text: bool = False,
) -> dict[str, Any]:
    """图 → candidate。``complete_text`` 只用加宽节点显示全文，不截断主图文字。"""
    validate_graph_shapes(graph)
    resolved = sizing or (_complete_sizing(graph) if complete_text else LevelSizing())
    positions = layout_level(graph, sizing=resolved)
    label_budget = 4000.0 if graph.level in (
        "cache", "compressor", "indexer", "attention", "kv") else 720.0
    rank = _ranks(graph)
    id_of = {node.node_id: _component_id(node.node_id) for node in graph.nodes}
    right_edge = max(x + w for x, _y, w, _h in positions.values())
    # 跨秩/斜穿关系走右侧竖井：每条约 14px，避免穿过无关组件。
    skip_edges = [
        edge
        for edge in graph.edges
        if edge.kind == "data"
        and edge.src in rank
        and edge.dst in rank
        and rank[edge.dst] - rank[edge.src] > 1
    ]
    # 跨度大的关系走更外侧的竖井，减少竖井之间的交叉。
    skip_edges.sort(key=lambda edge: rank[edge.dst] - rank[edge.src], reverse=True)
    lane_of: dict[int, float] = {}
    for order, edge in enumerate(skip_edges):
        lane_of[id(edge)] = right_edge + 24 + order * 6
    width = right_edge + 24 + max(len(skip_edges) - 1, 0) * 6 + 24
    height = max(y + h for _x, y, _w, h in positions.values()) + _MARGIN_Y
    width = max(width, resolved.main_x + resolved.main_w + _MARGIN_X)
    components = []
    for node in graph.nodes:
        x, y, w, h = positions[node.node_id]
        component: dict[str, Any] = {
            "id": id_of[node.node_id],
            "type": node.component_type,
            "icon": "none",
            "label": _fit(
                expand_display_text(node.label),
                max(w - 26, 4000.0) if graph.level in (
                    "cache", "compressor", "indexer", "attention", "kv") else w - 26,
                13.5, complete=complete_text,
            ),
            # 浏览器检查的可读性下限按投影后像素计算；图越宽缩放越小，
            # sublabel 用 10.5px 保证 2412 宽画布在 1440 视口仍 >=6px。
            "sublabel": _fit(
                expand_display_text(node.sublabel),
                max(w - 22, 4000.0) if graph.level in (
                    "cache", "compressor", "indexer", "attention", "kv") else w - 22,
                10.5, complete=complete_text,
            ),
            "tag": expand_display_text(node.tag),
            "pos": [round(x, 1), round(y, 1)],
            "size": [w, h],
        }
        if node.sources and source_evidence:
            component["sources"] = [dict(item) for item in node.sources]
        components.append(component)
    connections = []
    side_label_rects: list[tuple[float, float, float, float]] = []
    gap_cursor: dict[tuple[float, float], int] = {}
    for index, edge in enumerate(graph.edges):
        if edge.src not in positions or edge.dst not in positions:
            continue
        src, dst = positions[edge.src], positions[edge.dst]
        from_side, to_side = _sides(src, dst)
        vertical = from_side == "bottom" and to_side == "top"
        payload: dict[str, Any] = {
            "id": f"f-{index}-{_slug(edge.src)}-{_slug(edge.dst)}"[:64],
            "from": id_of[edge.src],
            "to": id_of[edge.dst],
            "variant": "dashed" if edge.kind == "state" else "default",
        }
        if id(edge) in lane_of:
            lane_x = lane_of[id(edge)]
            route = _lane_route(positions, lane_x, src, dst)
            payload["fromSide"] = "bottom"
            payload["toSide"] = "top"
            payload["route"] = "orthogonal-v"
            payload["via"] = route
            # 跨秩 lane 路由同样必须带张量/全称 shape；放在目标行上方的
            # 横段中点，避免把标签推到画布右侧空白里。
            label_text = _edge_display_text(edge, graph)
            if label_text:
                payload["label"] = _fit(
                    label_text, label_budget if complete_text else 260, 8,
                    complete=complete_text,
                )
                payload["labelAt"] = [
                    round((lane_x + dst_cx) / 2, 1),
                    round(dst[1] - 10, 1),
                ]
            connections.append(payload)
            continue
        src_cx = src[0] + src[2] / 2
        dst_cx = dst[0] + dst[2] / 2
        src_is_side = src[0] <= _LEFT_X + 1
        dst_is_side = dst[0] <= _LEFT_X + 1
        if src_is_side or dst_is_side:
            # 旁枝（辅助输入/跨调用状态/段外消费）紧贴锚点：从锚点所在**行间空隙**
            # 绕行（不横穿整行），因此锚点在第几列都不会压到别的组件。
            lane = round(_LEFT_X + resolved.side_w + 6, 1)
            if dst_is_side and not src_is_side:
                # 主链节点 → 左侧旁枝：源底部 → 行下空隙左移 → 旁枝右侧进入。
                gap_y = round(src[1] + src[3] + 14, 1)
                payload["fromSide"] = "bottom"
                payload["toSide"] = "right"
                payload["route"] = "orthogonal-v"
                payload["via"] = [
                    [round(src[0] + src[2] / 2, 1), gap_y],
                    [lane, gap_y],
                    [lane, round(dst[1] + dst[3] / 2, 1)],
                ]
                label_at = [round(lane + 26, 1), round(gap_y - 9, 1)]
            elif src_is_side and dst_is_side:
                # 旁枝内部：右侧竖井 → 目标行中心，从右侧进入（不贴着别的旁枝堆叠）。
                payload["fromSide"] = "right"
                payload["toSide"] = "right"
                payload["route"] = "orthogonal-v"
                payload["via"] = [
                    [lane, round(src[1] + src[3] / 2, 1)],
                    [lane, round(dst[1] + dst[3] / 2, 1)],
                ]
                label_at = [round(lane + 26, 1),
                            round((src[1] + dst[1]) / 2 - 9, 1)]
            else:
                # 左侧旁枝 → 主链节点：旁枝右侧 → 行上空隙右移 → 目标顶部进入。
                gap_y = round(dst[1] - 12, 1)
                payload["fromSide"] = "right"
                payload["toSide"] = "top"
                payload["route"] = "orthogonal-v"
                payload["via"] = [
                    [lane, round(src[1] + src[3] / 2, 1)],
                    [lane, gap_y],
                    [round(dst[0] + dst[2] / 2, 1), gap_y],
                ]
                label_at = [round(lane + 26, 1), round(gap_y - 9, 1)]
            # 侧栏空间只有约 60px：默认用短语义标签，全称 shape 在旁枝节点
            # sublabel 与 relations JSON 中；不把长形状串压到主链组件上。
            # 返修例外：整网主链→旁枝的数据边若在侧栏上方空白里放得下，就把
            # 全称 shape 放在 lane 左侧（不遮主链/旁枝节点），让原先因缩小而
            # 不可读的连线文字恢复完整；放不下才退回短标签。
            if graph.key.startswith(("decoder-", "mhc-")) or graph.level.startswith("moe"):
                label_text = _edge_display_text(edge, graph)
            elif edge.tensor_id == "cache_cumlen_post":
                label_text = "cumlen 后"
            else:
                label_text = "辅助输入" if edge.kind == "aux" else "段输入"
            placed_full = False
            if (
                complete_text
                and graph.key == "network"
                and dst_is_side
                and not src_is_side
                and edge.kind == "data"
                and edge.shape
            ):
                full_text = expand_display_text(edge.shape)
                label_width = _archify_text_units(full_text) * 4.8 + 10.0
                label_cx = round(lane - label_width / 2 - 8.0, 1)
                label_y = round(gap_y - 9.0, 1)
                rect = (
                    label_cx - label_width / 2,
                    label_y - 10.0,
                    label_width,
                    14.0,
                )
                fits = rect[0] >= 2.0 and rect[0] + rect[2] <= lane - 4.0
                if fits:
                    for other in positions.values():
                        ox, oy, ow, oh = other
                        if not (
                            rect[0] + rect[2] <= ox
                            or rect[0] >= ox + ow
                            or rect[1] + rect[3] <= oy
                            or rect[1] >= oy + oh
                        ):
                            fits = False
                            break
                if fits:
                    for other in side_label_rects:
                        if not (
                            rect[0] + rect[2] <= other[0]
                            or rect[0] >= other[0] + other[2]
                            or rect[1] + rect[3] <= other[1]
                            or rect[1] >= other[1] + other[3]
                        ):
                            fits = False
                            break
                if fits:
                    side_label_rects.append(rect)
                    payload["label"] = full_text
                    payload["labelAt"] = [label_cx, label_y]
                    payload["labelDx"] = 0
                    placed_full = True
            if not placed_full and label_text and edge.kind != "state":
                if graph.key != "network":
                    full_side = (graph.key.startswith(("decoder-", "mhc-"))
                                 or graph.level.startswith("moe"))
                    payload["label"] = _fit(
                        label_text,
                        label_budget if full_side else 60,
                        8 if full_side else 9,
                        complete=complete_text if full_side else False,
                    )
                    payload["labelAt"] = [round(lane + 26, 1), label_at[1]]
                    connections.append(payload)
                    continue
                short_text = _fit(label_text, 60, 9, complete=complete_text)
                short_width = _archify_text_units(short_text) * 4.8 + 10.0
                short_cx = round(lane + 26, 1)
                short_y = label_at[1]
                # 同一行的多条旁枝边会共用行间空隙；逐条向下错开，避免
                # “段输入”互相叠字。最多尝试 8 个 18px 档位。
                for _attempt in range(8):
                    rect = (
                        short_cx - short_width / 2,
                        short_y - 10.0,
                        short_width,
                        14.0,
                    )
                    collided = False
                    for other in side_label_rects:
                        if not (
                            rect[0] + rect[2] <= other[0]
                            or rect[0] >= other[0] + other[2]
                            or rect[1] + rect[3] <= other[1]
                            or rect[1] >= other[1] + other[3]
                        ):
                            collided = True
                            break
                    if not collided:
                        for ox, oy, ow, oh in positions.values():
                            if not (
                                rect[0] + rect[2] <= ox
                                or rect[0] >= ox + ow
                                or rect[1] + rect[3] <= oy
                                or rect[1] >= oy + oh
                            ):
                                collided = True
                                break
                    if not collided:
                        side_label_rects.append(rect)
                        payload["label"] = short_text
                        payload["labelAt"] = [short_cx, short_y]
                        break
                    short_y = round(short_y + 18.0, 1)
            connections.append(payload)
            continue
        same_or_adjacent = (
            edge.src in rank
            and edge.dst in rank
            and 0 <= rank[edge.dst] - rank[edge.src] <= 1
        )
        if same_or_adjacent and abs(src_cx - dst_cx) >= 40:
            # 同秩/相邻秩但不同列：在行间空隙走水平段，避免穿过同排或下一排组件。
            gap_y = round(src[1] + src[3] + (resolved.step_y - resolved.node_h) / 2, 1)
            # 不得压进目标行；留 2px 以上净空。
            gap_y = min(gap_y, round(dst[1] - 12, 1))
            key = (round(src_cx, 1), gap_y)
            offset = gap_cursor.get(key, 0)
            gap_cursor[key] = offset + 1
            shift = min(offset * 9, max(8, int((dst[1] - (src[1] + src[3])) / 2) - 10))
            shift = min(shift, max(0, int(dst[1] - 12 - gap_y)))
            payload["fromSide"] = "bottom"
            payload["toSide"] = "top"
            payload["route"] = "orthogonal-v"
            payload["via"] = [
                [round(src_cx, 1), round(gap_y + shift, 1)],
                [round(dst_cx, 1), round(gap_y + shift, 1)],
            ]
            # 行间水平段：完整文字模式优先显示真实 shape（同一依赖不能只写
            # “数据依赖”而把张量身份留空）；非完整模式沿用既有短语义。
            if graph.key.startswith(("decoder-", "mhc-")) or graph.level.startswith("moe"):
                label_text = _edge_display_text(edge, graph) or "数据依赖"
            else:
                label_text = (
                    (edge.shape or edge.label or "数据依赖")
                    if complete_text and graph.key == "network"
                    else (edge.label or "数据依赖")
                )
            if label_text:
                payload["label"] = _fit(
                    label_text, label_budget if complete_text else 80, 8,
                    complete=complete_text,
                )
                payload["labelAt"] = [
                    round((src_cx + dst_cx) / 2, 1),
                    round(gap_y + shift - 9, 1),
                ]
            connections.append(payload)
            continue
        if vertical:
            payload["fromSide"] = from_side
            payload["toSide"] = to_side
            # 完整文字模式：纵向行间空隙只放一条 14px 高的标签，宽度预算放大到
            # 720 让全称 shape/表达式完整显示；标签仍在行间空隙，不压组件。
            shape_budget = label_budget if complete_text else 260
            text_budget = label_budget if complete_text else 150
            if graph.key.startswith(("decoder-", "mhc-")) or graph.level.startswith("moe"):
                payload["label"] = _edge_display_text(edge, graph)
                payload["labelDy"] = round((dst[1] - (src[1] + src[3])) / 2 + 10, 1)
            elif edge.shape:
                payload["label"] = _fit(
                    edge.shape, shape_budget, 8, complete=complete_text)
                payload["labelDy"] = round((dst[1] - (src[1] + src[3])) / 2 + 10, 1)
            elif edge.label:
                payload["label"] = _fit(
                    edge.label, text_budget, 8, complete=complete_text)
                payload["labelDy"] = round((dst[1] - (src[1] + src[3])) / 2 + 10, 1)
        connections.append(payload)
    # 统一连线标签去重叠：全部图位（整网/端点/层组/Decoder/mHC/MoE）都走这里。
    # connections 使用组件 id（n-...），positions 使用图节点 id；先建立反向映射。
    component_positions = {
        id_of[node.node_id]: positions[node.node_id] for node in graph.nodes
    }
    # 节点文字可能比节点框更宽（完整文字模式不截断）；把文字溢出部分也作为
    # 障碍，防止连线标签压在节点标题上。
    node_text_obstacles: list[tuple[float, float, float, float]] = []
    for node in graph.nodes:
        rect = positions[node.node_id]
        text_width = max(
            _edge_label_width(node.label) * 11 / 8,
            _edge_label_width(node.sublabel or "") * 9 / 8,
        )
        expand = max(0.0, (text_width - rect[2]) / 2)
        node_text_obstacles.append(
            (rect[0] - expand, rect[1], rect[2] + expand * 2, rect[3])
        )
    canvas_right = max(rect[0] + rect[2] for rect in positions.values())
    label_layout = separate_edge_labels(
        connections, component_positions, node_text_obstacles,
        max_right=(canvas_right + 24)
        if graph.key.startswith("decoder-") else None,
    )

    candidate: dict[str, Any] = {
        "schema_version": 1,
        "diagram_type": "architecture",
        "meta": {
            "title": graph.title,
            "subtitle": graph.subtitle,
            "locale": "zh-CN",
            "quality_profile": quality,
            "output": output_relpath,
            "legend": {
                "entries": {
                    "backend": {"label": {
                        "network": "整网算子", "group": "层组/模板",
                        "decoder": "Block 模块", "moe": "MoE 算子",
                        "mhc": "mHC / 门控残差算子",
                        "attention": "Attention 内部算子",
                        "kv": "缓存/状态算子",
                        "indexer": "Indexer 算子",
                        "ffn": "FFN 算子",
                        "ple": "PLE 算子",
                        "deepstack": "DeepStack 算子",
                    }.get(graph.level, "计算操作")},
                    "external": {"label": "输入/输出、辅助输入与跨调用状态"},
                }
            },
        },
        "components": components,
        "connections": connections,
        "cards": _level_cards(
            graph, tool, repository, source_evidence=source_evidence),
    }
    if repository:
        candidate["meta"]["repository"] = dict(repository)
    return candidate


def _text_width(text: str, font_px: float) -> float:
    """粗略字宽：CJK 全角按 font，ASCII 按 0.56 font（只用于截断预算）。"""
    width = 0.0
    for char in str(text):
        width += font_px if ord(char) > 0x2E7F else font_px * 0.56
    return width


def _archify_text_units(text: str) -> float:
    """Archify ``textUnits`` 的等价口径：全角/中文 2 单位，ASCII 1 单位。"""
    return sum(2.0 if ord(char) > 0x2E7F else 1.0 for char in str(text or ""))


def _fit(text: str, max_px: float, font_px: float, *, complete: bool = False) -> str:
    """按像素预算截断主图文字（完整内容保留在关系 JSON 与详情里）。

    ``complete=True`` 时不为了紧凑截断：只要 Archify 的可读下限（标签 8px /
    副标签 6px）仍能容纳全文就原样保留，让渲染器收缩字号而不是丢字。
    """
    text = str(text or "")
    if complete:
        floor = 8.0 if font_px >= 13.0 else 6.0
        if _archify_text_units(text) * floor * 0.6 <= max_px + 8.0:
            return text
    if _text_width(text, font_px) <= max_px:
        return text
    trimmed = text
    while trimmed and _text_width(trimmed + "…", font_px) > max_px:
        trimmed = trimmed[:-1]
    return (trimmed + "…") if trimmed else ""


def _level_cards(
    graph: LevelGraph,
    tool: Mapping[str, Any] | None,
    repository: Mapping[str, Any] | None,
    *,
    source_evidence: bool = True,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if graph.parameter_items:
        items = []
        for param in graph.parameter_items[:12]:
            symbol = param.get("symbol") or param.get("id")
            name = full_name(str(symbol))
            value = param.get("value")
            expression = param.get("expression")
            shown = (
                f"= {value}"
                if value is not None
                else (f"= {expression}" if expression else "（符号长度）")
            )
            items.append(
                f"{name}：{expand_display_text(str(param.get('meaning') or ''))} {shown}")
        items.append("展示使用 modeling/config 全称；旧别名只用于旧事实映射，不进入主图。")
        if graph.level == "mhc":
            items.append("来源：config[hc_mult]/[hc_sinkhorn_iters]/[hc_eps]；"
                         "modeling 926-938、976-979。")
        cards.append({"dot": "cyan", "title": "参数（全称）", "items": items})
    side = [
        edge
        for edge in graph.relations()
        if edge["kind"] != "data" or not edge["shape"]
    ]
    if side:
        items = [
            f"{edge['from']} → {edge['to']}："
            f"{expand_display_text(str(edge['tensor_label']))} "
            f"{edge['shape']}".strip()
            + (f"（{edge['label']}）" if edge.get("label") else "")
            for edge in side[:18]
        ]
        if len(side) > 18:
            items.append(f"…其余 {len(side) - 18} 条见同目录 relations JSON")
        cards.append(
            {"dot": "rose", "title": "分支/状态关系（张量身份与全称维度）", "items": items}
        )
    node_items = [
        (
            f"{expand_display_text(node.label)}"
            f"（{expand_display_text(node.sublabel)}）"
            if node.sublabel else expand_display_text(node.label)
        )
        for node in graph.nodes
    ]
    if node_items:
        if len(node_items) > 24:
            node_items = node_items[:24] + [f"…其余 {len(graph.nodes) - 24} 个节点见 candidate JSON"]
        cards.append(
            {"dot": "slate", "title": "节点全称（主图截断文字的完整版本）", "items": node_items}
        )
    audit = [item for item in graph.expansions if isinstance(item, dict) and item.get("rule")]
    if audit:
        applied = [item for item in audit if item.get("status") == "applied"]
        rejected = [item for item in audit if item.get("status") == "rejected"]
        items = []
        if rejected and not applied:
            items.append(
                "能力边界：当前实现没有为该模型提供可执行变换；"
                "0 条应用是能力边界，不是逐条评估后拒绝。"
            )
        items.extend([
            f"{item.get('rule')} {item.get('title')}：{item.get('status')} — "
            f"{item.get('reason')}"
            + (f"（候选：{', '.join(item.get('candidates') or ())}）"
               if item.get("candidates") else "")
            for item in audit
        ])
        cards.append(
            {
                "dot": "amber",
                "title": f"化简能力与规则（应用 {len(applied)} / 拒绝 {len(rejected)}，原结构保留）",
                "items": items,
            }
        )
    elif graph.form == "raw":
        cards.append(
            {
                "dot": "amber",
                "title": "原始语义图（未应用任何化简）",
                "items": [
                    "本图是源码逐句算子/张量的原始投影，不经过化简规则。",
                    "同层组的 normalized 版给出逐条规则的候选与拒绝原因；结构未改。",
                ],
            }
        )
    if graph.notes:
        cards.append(
            {
                "dot": "amber",
                "title": "缓存前提与覆盖范围",
                "items": [expand_display_text(str(item)) for item in graph.notes[:6]],
            }
        )
    if not source_evidence:
        refs = [
            f"{expand_display_text(node.label)}：{source['path']}:{source['line']}"
            + (f"-{source.get('end_line')}" if source.get("end_line") else "")
            for node in graph.nodes
            for source in node.sources
        ]
        cards.append(
            {
                "dot": "violet",
                "title": "来源映射（离线核对；不声明 Archify 仓库证据）",
                "items": (
                    [f"本图 {len(refs)} 条来源引用，完整映射见候选 "
                     "`data/archify/sources/` 下的 sources JSON。"]
                    + refs[:18]
                    + ([f"…其余 {len(refs) - 18} 条见 sources JSON"] if len(refs) > 18 else [])
                ),
            }
        )
    cards.append(
        {
            "dot": "emerald",
            "title": "来源、边界与限制",
            "items": [
                (
                    f"源码：{repository.get('url')} @ {repository.get('revision')}"
                    if repository
                    else "源码：模型检出（文件哈希见 reports/source_manifest.json；"
                         "Archify 仓库证据未声明）"
                ),
                f"层级：{graph.level}；层组：{graph.group or '—'}；形态：{graph.form}",
                "节点/边由同候选的结构文档程序生成（operators/tensors/source_refs），"
                "不是手抄或从 SVG 反推。",
                "逻辑计算图不等于设备执行图；未合并精度、未省略缓存副作用。",
                "不实例化模型、不读权重；未知/条件量保留符号。",
            ],
        }
    )
    if tool:
        cards.append(
            {
                "dot": "violet",
                "title": "产物身份",
                "items": [
                    f"Archify CLI：{tool.get('version')}（{str(tool.get('cli_sha256', ''))[:12]}…）",
                    f"Node：{tool.get('node')}；生成器：{BUILDER_VERSION}",
                ],
            }
        )
    return cards


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LevelRequest:
    """一个分层图请求：level 决定构建器，slug 决定产物文件名。"""

    level: str
    variant: str
    form: str = "normalized"


_STABLE_ENTRY_KEYS = (
    "level",
    "variant",
    "form",
    "title",
    "slug",
    "html",
    "html_sha256",
    "view",
    "view_sha256",
    "candidate",
    "candidate_sha256",
    "relations",
    "relations_sha256",
    "sources",
    "sources_sha256",
    "nodes",
    "connections",
    "relations_count",
    "links",
)


def stable_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """交付面白名单：只保留与模型/图本身相关的稳定字段。

    门禁结果、浏览器可用性、回执路径、时间戳等运行证据一律剥离，避免同一身份在
    Chrome 可用/受限两种环境生出不同交付字节。
    """
    stable = {key: entry.get(key) for key in _STABLE_ENTRY_KEYS if key in entry}
    stable["links"] = [dict(item) for item in entry.get("links", ()) or ()]
    return stable


def stable_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """稳定 manifest：白名单 entry + 工具/来源/请求/文件哈希，其余运行字段剥离。"""
    return {
        "schema": manifest.get("schema"),
        "builder": manifest.get("builder"),
        "tool": manifest.get("tool"),
        "repository": manifest.get("repository"),
        "requests": manifest.get("requests"),
        "entries": [stable_entry(entry) for entry in manifest.get("entries", ()) or ()],
        "files": manifest.get("files"),
    }


def archify_identity(workspace_root: str | Path) -> dict[str, Any]:
    """工具身份：CLI/package 闭包/Node 版本 + 生成器版本（进入候选身份）。"""
    tool = discover_archify(workspace_root)
    identity = dict(tool_identity(tool))
    identity["builder"] = BUILDER_VERSION
    identity["schema"] = SCHEMA
    return identity


def generate_archify_layers(
    workspace_root: str | Path,
    *,
    project_root: Path,
    network: Mapping[str, Any],
    structures: Sequence[Mapping[str, Any]],
    raw_bundles: Mapping[str, Any] | None = None,
    normalized_bundles: Mapping[str, Any] | None = None,
    normalization_reports: Mapping[str, Mapping[str, Any]] | None = None,
    requests: Sequence[LevelRequest] | None = None,
    repo_root: Path,
    repository: Mapping[str, Any],
    model_key: str = "deepseek_v4_flash_0731_hf",
    module_groups_fn: Callable[[Mapping[str, Any]], list] | None = None,
    viewer_prefix: str = "viewer/archify",
    data_prefix: str = "data/archify",
    allow_browser_check_unavailable: bool = True,
    quality: str = "standard",
    source_evidence: bool = True,
) -> dict[str, Any]:
    """按请求生成分层图 candidate + HTML + relations，返回 manifest 与需落盘文件。

    产物先写入 ``project_root/.work`` 临时目录（Archify finalize 需要真实文件），
    再由调用方以字节形式并入交付包，保证既有原子提交语义不变。
    """
    root = Path(workspace_root)
    tool = discover_archify(root)
    identity = dict(tool_identity(tool))
    identity["builder"] = BUILDER_VERSION
    identity["schema"] = SCHEMA
    profile = graph_profile_for(model_key)
    is_deepseek = model_key == "deepseek_v4_flash_0731_hf"
    group_module_fn = module_groups_fn or module_groups
    variants = {str(item["id"]): item for item in structures}
    block_of = {
        gid: item["structure"]["block"] for gid, item in variants.items()
    }
    modules_of = {gid: group_module_fn(block) for gid, block in block_of.items()}
    segment_of: dict[tuple[str, str], SegmentSpec] = {}
    pair_levels: set[str] = set()
    for gid, block in block_of.items():
        specs = profile.segments(block, _group_stub(variants[gid], block), modules_of[gid])
        for spec in specs:
            segment_of[(gid, spec.level)] = spec
            if spec.pair_raw:
                pair_levels.add(spec.level)
    # 首次构建时 .work 可能还不存在（CLI 的 _materialize 建目录更晚），先建立父目录。
    work_root = Path(project_root) / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="archify-layers-", dir=str(work_root)))
    files: dict[str, bytes] = {}
    receipts: dict[str, bytes] = {}
    runtime: dict[str, dict[str, Any]] = {}
    node_details: dict[str, list[dict[str, Any]]] = {}
    entries: list[dict[str, Any]] = []
    try:
        for request in requests or ():
            gid = request.variant
            if request.level == "network":
                graph = (
                    build_network_graph(network) if is_deepseek
                    else build_network_graph_for(network, profile)
                )
            elif request.level == "group":
                group = _group_of(network, gid)
                graph = (
                    build_group_graph(
                        network, group, block_of[gid],
                        modules=modules_of[gid],
                    ) if is_deepseek
                    else build_group_graph_for(
                        network, group, block_of[gid],
                        modules=modules_of[gid], profile=profile)
                )
            elif request.level == "decoder":
                group = _group_of(network, gid)
                graph = (
                    build_decoder_graph(
                        block_of[gid], modules=modules_of[gid], group=group,
                        attention_types={},
                    ) if is_deepseek
                    else build_decoder_graph_for(
                        block_of[gid], modules=modules_of[gid], group=group,
                        profile=profile)
                )
            elif not is_deepseek and (gid, request.level) in segment_of:
                group = _group_of(network, gid)
                graph = build_segment_graph(
                    block_of[gid], group=group, modules=modules_of[gid],
                    spec=segment_of[(gid, request.level)], profile=profile,
                    form=request.form,
                )
            elif request.level in ("embedding", "head"):
                if not is_deepseek:
                    raise ArchifyLayersError(
                        f"端点图 {request.level!r} 只用于 DeepSeek 整网简单端点"
                    )
                graph = build_endpoint_graph(network, request.level)
            elif request.level == "moe":
                group = _group_of(network, gid)
                graph = build_moe_relation_graph(
                    block_of[gid], modules=modules_of[gid], group=group
                )
            elif request.level == "moe-route":
                group = _group_of(network, gid)
                graph = build_moe_route_graph(block_of[gid], group=group)
            elif request.level == "moe-expert":
                group = _group_of(network, gid)
                graph = build_moe_expert_graph(block_of[gid], group=group)
            elif request.level == "moe-shared":
                group = _group_of(network, gid)
                graph = build_moe_shared_graph(block_of[gid], group=group)
            elif request.level == "moe-combine":
                group = _group_of(network, gid)
                graph = build_moe_combine_graph(block_of[gid], group=group)
            elif request.level == "mhc":
                group = _group_of(network, gid)
                graph = build_mhc_graph(
                    block_of[gid], modules=modules_of[gid], group=group
                )
            elif request.level in ("cache-main", "cache-compressor", "cache-indexer"):
                group = _group_of(network, gid)
                graph = build_cache_graph(
                    block_of[gid], group=group, part=request.level.split("-", 1)[1])
            elif request.level == "q":
                group = _group_of(network, gid)
                graph = build_q_graph(block_of[gid], modules=modules_of[gid], group=group)
            elif request.level == "o":
                group = _group_of(network, gid)
                graph = build_o_graph(block_of[gid], modules=modules_of[gid], group=group)
            elif request.level == "compressor":
                group = _group_of(network, gid)
                graph = build_compressor_graph(
                    block_of[gid], modules=modules_of[gid], group=group)
            elif request.level == "indexer":
                group = _group_of(network, gid)
                graph = build_indexer_graph(
                    block_of[gid], modules=modules_of[gid], group=group)
            elif request.level in ("attention", "kv"):
                if normalized_bundles is None:
                    raise ArchifyLayersError("语义视图需要 T12 normalized 语义 bundle")
                bundle = (
                    raw_bundles[gid]
                    if request.form == "raw" and raw_bundles is not None
                    else normalized_bundles[gid]
                )
                view = build_view_model(bundle, request.level)
                slug = f"{request.level}-{gid}-{request.form}"
                candidate = build_candidate(
                    bundle,
                    view,
                    output_relpath=f"{viewer_prefix}/{slug}.html",
                    form=request.form,
                    normalization_report=(normalization_reports or {}).get(gid),
                    tool=identity,
                    repository=repository,
                )
                candidate["meta"]["quality_profile"] = quality
                edges_records = view.edge_records
                entry = _finalize_level(
                        root=root,
                        tool=tool,
                        work_dir=work_dir,
                        candidate=candidate,
                        slug=slug,
                        viewer_prefix=viewer_prefix,
                        data_prefix=data_prefix,
                        files=files,
                        repo_root=repo_root,
                        allow_browser_check_unavailable=allow_browser_check_unavailable,
                        level=request.level,
                        variant=gid,
                        form=request.form,
                        nodes=len(view.nodes),
                        connections=len(view.connections),
                        relations=edges_records,
                        title=str(candidate["meta"]["title"]),
                        subtitle=str(candidate["meta"].get("subtitle") or ""),
                        source_evidence=source_evidence,
                    )
                receipts.update(entry.pop("receipts", {}))
                runtime[entry["slug"]] = entry.pop("runtime", {})
                node_details[entry["slug"]] = [
                    {
                        "id": view_node.semantic_id,
                        "label": view_node.label,
                        "sublabel": view_node.sublabel,
                        "tag": view_node.tag,
                        "shape": view_node.shape,
                        "sources": [source.to_dict() for source in view_node.sources],
                    }
                    for view_node in view.nodes
                ]
                entries.append(entry)
                continue
            else:
                raise ArchifyLayersError(f"未知层级 {request.level!r}")
            if request.level == "network":
                slug = "network"
            elif request.level in ("embedding", "head"):
                slug = request.level
            elif not is_deepseek and request.level in pair_levels:
                slug = f"{request.level}-{gid}-{request.form}"
            else:
                slug = f"{request.level}-{gid}"
            graph.form = request.form
            candidate = build_level_candidate(
                graph,
                output_relpath=f"{viewer_prefix}/{slug}.html",
                repository=repository,
                tool=identity,
                quality=quality,
                source_evidence=source_evidence,
                complete_text=graph.level in _COMPLETE_TEXT_LEVELS,
            )
            entry = _finalize_level(
                root=root,
                tool=tool,
                work_dir=work_dir,
                candidate=candidate,
                slug=slug,
                viewer_prefix=viewer_prefix,
                data_prefix=data_prefix,
                files=files,
                repo_root=repo_root,
                allow_browser_check_unavailable=allow_browser_check_unavailable,
                level=request.level,
                variant=gid,
                form=request.form,
                nodes=len(graph.nodes),
                connections=len(graph.edges),
                relations=graph.relations(),
                title=graph.title,
                subtitle=graph.subtitle,
                graph=graph,
                source_evidence=source_evidence,
            )
            receipts.update(entry.pop("receipts", {}))
            runtime[entry["slug"]] = entry.pop("runtime", {})
            node_details[entry["slug"]] = [
                {
                    "id": graph_node.node_id,
                    "label": graph_node.label,
                    "sublabel": graph_node.sublabel,
                    "tag": graph_node.tag,
                    "details": list(graph_node.details),
                    "sources": [dict(item) for item in graph_node.sources],
                }
                for graph_node in graph.nodes
            ]
            entries.append(entry)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    manifest = stable_manifest(
        {
            "schema": SCHEMA,
            "builder": BUILDER_VERSION,
            "tool": identity,
            "repository": dict(repository) if repository else None,
            "requests": [request.__dict__ for request in requests or ()],
            "entries": entries,
            "files": {
                relpath: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for relpath, data in sorted(files.items())
            },
        }
    )
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    files[f"{data_prefix}/manifest.json"] = manifest_bytes
    return {
        "manifest": manifest,
        "files": files,
        "receipts": receipts,
        "runtime": runtime,
        "node_details": node_details,
    }


def _finalize_level(
    *,
    root: Path,
    tool: ArchifyTool,
    work_dir: Path,
    candidate: Mapping[str, Any],
    slug: str,
    viewer_prefix: str,
    data_prefix: str,
    files: dict[str, bytes],
    repo_root: Path,
    allow_browser_check_unavailable: bool,
    level: str,
    variant: str,
    form: str,
    nodes: int,
    connections: int,
    relations: Sequence[Mapping[str, Any]],
    title: str,
    subtitle: str = "",
    graph: LevelGraph | None = None,
    source_evidence: bool = True,
) -> dict[str, Any]:
    """生成一张图：交付字节只含稳定内容，运行证据（门禁/浏览器）单独返回。"""
    candidate_path = work_dir / f"{slug}.json"
    candidate_path.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    html_path = work_dir / f"{slug}.html"
    receipt_dir = work_dir / "receipts" / slug
    run = finalize_candidate(
        tool,
        candidate_path,
        html_path,
        repo_root=repo_root,
        out_dir=receipt_dir,
        workspace_root=root,
        allow_browser_check_unavailable=allow_browser_check_unavailable,
        quality=str(candidate["meta"].get("quality_profile", "standard")),
    )
    receipt = run["receipt"]
    files[f"{data_prefix}/{slug}.json"] = candidate_path.read_bytes()
    relations_bytes = (
        json.dumps(list(relations), ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    files[f"{data_prefix}/edges/{slug}-relations.json"] = relations_bytes
    sources_rel = ""
    sources_bytes = b""
    if graph is not None and not source_evidence:
        records = [
            {
                "node_id": node.node_id,
                "label": node.label,
                "role": node.role,
                "sources": [dict(item) for item in node.sources],
            }
            for node in graph.nodes
            if node.sources
        ]
        sources_rel = f"{data_prefix}/sources/{slug}-sources.json"
        sources_bytes = (
            json.dumps(
                {
                    "schema": "hf-archify-node-sources.v1",
                    "slug": slug,
                    "level": level,
                    "variant": variant,
                    "note": (
                        "该模型源码检出不是本工作区的 git top-level（HiDream 官方实现"
                        "位于嵌套目录），因此不声明 Archify 仓库证据；路径/行号为固定"
                        "官方 revision 的静态来源，文件 SHA256 见 reports/source_manifest.json。"
                    ),
                    "nodes": records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        files[sources_rel] = sources_bytes
    files[f"{viewer_prefix}/{slug}.html"] = html_path.read_bytes()
    view_rel = f"{viewer_prefix}/{slug}.view.html"
    files[view_rel] = _wrapper_page(
        slug=slug,
        title=title,
        subtitle=subtitle,
        level=level,
        variant=variant,
        form=form,
        nodes=nodes,
        relations_count=len(relations),
        back_href=_back_href(level, variant),
        sources=(
            f"{data_prefix}/sources/{slug}-sources.json"
            if graph is not None and not source_evidence else ""
        ),
    )
    receipts: dict[str, bytes] = {}
    for name in ("finalize.json", "finalize-summary.json", "browser-check.json"):
        path = receipt_dir / f"{slug}.{name}"
        if path.is_file():
            receipts[f"{data_prefix}/receipts/{slug}/{name}"] = path.read_bytes()
    return {
        "level": level,
        "variant": variant,
        "form": form,
        "title": title,
        "slug": slug,
        "html": f"{viewer_prefix}/{slug}.html",
        "html_sha256": _sha256_file(html_path),
        "view": view_rel,
        "view_sha256": hashlib.sha256(files[view_rel]).hexdigest(),
        "candidate": f"{data_prefix}/{slug}.json",
        "candidate_sha256": _sha256_file(candidate_path),
        "relations": f"{data_prefix}/edges/{slug}-relations.json",
        "relations_sha256": hashlib.sha256(relations_bytes).hexdigest(),
        "sources": sources_rel,
        "sources_sha256": (
            hashlib.sha256(sources_bytes).hexdigest() if sources_bytes else ""
        ),
        "nodes": nodes,
        "connections": connections,
        "relations_count": len(relations),
        "links": _relations_summary(relations),
        "receipts": receipts,
        "runtime": {
            "gates": receipt.get("gates") or {},
            "browser_check_available": receipt.get("browser_check_available"),
            "browser_check_classification": receipt.get("browser_check_classification"),
        },
    }


def _back_href(level: str, variant: str) -> str:
    """子图返回的分层页位置（写入 wrapper，离线相对链接）。"""
    if level == "network":
        return "../structure.html#/network"
    if level in ("embedding", "head"):
        return "../structure.html#/network"
    if level == "group":
        return f"../structure.html#/group/{variant}"
    return f"../structure.html#/group/{variant}/block"


def _wrapper_page(
    *,
    slug: str,
    title: str,
    subtitle: str,
    level: str,
    variant: str,
    form: str,
    nodes: int,
    relations_count: int,
    back_href: str,
    sources: str = "",
) -> bytes:
    """子图的同源产品包装页：可见的返回路径 + 内嵌真实图 + 图数据入口。

    图本身仍是 Archify 生成的单文件 HTML（不改动其字节/门禁）；包装页只加导航与
    说明，离线可用。用户从分层页点击进入即落到这里，返回即是原层组上下文。
    """
    level_label = {
        "network": "整网",
        "embedding": "Embedding 端点（逐算子）",
        "head": "输出端算子（HyperHead → lm_head）",
        "group": "层组",
        "decoder": "Block/Decoder",
        "attention": "Attention",
        "kv": "KV / 跨调用状态",
        "moe": "FFN/MoE",
        "ffn": "FFN / MoE",
        "mhc": "mHC / 门控残差",
        "indexer": "Indexer",
        "ple": "N-gram PLE",
        "deepstack": "DeepStack",
    }.get(level, level)
    form_label = "原始语义图" if form == "raw" else "化简后"
    back_text = {
        "network": "← 返回整网层",
        "group": f"← 返回层组 {variant}",
    }.get(level, f"← 返回层组 {variant} · Block 层")
    escaped = html.escape
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escaped(title)}</title>
<style>
:root {{ color-scheme: dark light; }}
html,body{{margin:0;padding:0;background:#0f1115;color:#e7e9ee;
  font:13px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC","Segoe UI",sans-serif}}
header{{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;padding:9px 14px;
  border-bottom:1px solid #2a2f3a;background:#151922;position:sticky;top:0;z-index:5}}
header .ttl{{font-weight:700;font-size:14px}}
header .meta{{color:#9aa4b2}}
header .sp{{flex:1}}
header a{{color:#7cc4ff;text-decoration:none;border-bottom:1px dotted currentColor}}
header a.back{{color:#fff;background:#2563eb;border:0;border-radius:6px;padding:4px 12px;font-weight:600}}
main{{height:calc(100vh - 50px)}}
iframe{{width:100%;height:100%;border:0;display:block}}
footer{{padding:8px 14px;color:#9aa4b2;border-top:1px solid #2a2f3a;font-size:12px}}
</style></head><body>
<header>
  <a class="back" href="{escaped(back_href)}">{escaped(back_text)}</a>
  <span class="ttl">{escaped(title)}</span>
  <span class="meta">{escaped(level_label)} · {escaped(form_label)} · {nodes} 节点 · {relations_count} 条关系</span>
  <span class="sp"></span>
  <a href="{escaped(slug)}.html" target="_blank" rel="noopener">原始图（新标签）</a>
  <a href="../../data/archify/{escaped(slug)}.json">图数据 candidate JSON</a>
  <a href="../../data/archify/edges/{escaped(slug)}-relations.json">完整关系 JSON</a>
  {f'<a href="../../{escaped(sources)}">来源映射 JSON</a>' if sources else ''}
  <a href="{escaped(back_href)}">返回分层页</a>
</header>
<main><iframe src="{escaped(slug)}.html" title="{escaped(title)}"></iframe></main>
<footer>{escaped(subtitle) or escaped(title)}。图为 Archify 从同候选结构文档程序生成的单文件离线图；
校验回执与浏览器证据见候选证据目录。</footer>
</body></html>
""".encode("utf-8")


def _relations_summary(
    relations: Sequence[Mapping[str, Any]], *, limit: int = 24
) -> list[dict[str, Any]]:
    items = []
    for record in relations[:limit]:
        shape = str(record.get("shape") or "")
        items.append(
            {
                "from": record.get("from"),
                "to": record.get("to"),
                "tensor": record.get("tensor_label"),
                "shape": shape,
                "kind": record.get("kind"),
                "label": record.get("label"),
            }
        )
    return items


def _group_of(network: Mapping[str, Any], gid: str) -> Mapping[str, Any]:
    for group in network.get("groups", []):
        if str(group.get("id")) == gid:
            return group
    raise ArchifyLayersError(f"整网文档没有层组 {gid!r}")


# ---------------------------------------------------------------------------
# 通用分层图（GLM / Qwen / HiDream）：同一套构建器按模型声明投影
# ---------------------------------------------------------------------------


def _all_ops(block: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(block.get("operators", []))


def _key_tensor(block: Mapping[str, Any], *, kind: str) -> Mapping[str, Any] | None:
    tensors = [t for t in block.get("tensors", []) if str(t.get("kind")) == kind]
    return tensors[0] if tensors else None


def _main_boundary(
    block: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """块的主输入/主输出：边界张量里名字含 hidden 的那一对。

    位置编码/mask/cos/sin/PLE ids/DeepStack 特征等同为 ``kind=boundary`` 的张量，
    显式排除出主链后作为短旁枝；不靠张量顺序猜。
    """
    boundary = [
        t for t in block.get("tensors", []) if str(t.get("kind")) == "boundary"
    ]
    hidden = [
        t for t in boundary
        if "hidden" in str(t.get("id", "")).lower()
        or "hidden" in str(t.get("name", "")).lower()
    ]
    if len(hidden) < 2:
        # 真实边界命名不含 hidden 时，用“无生产者/无消费者”判断，仍不猜数值。
        producer = {
            str(tid): str(op["id"])
            for op in block.get("operators", [])
            for tid in op.get("outputs", [])
        }
        consumed = {
            str(tid)
            for op in block.get("operators", [])
            for tid in op.get("inputs", [])
        }
        inputs = [
            t for t in boundary if str(t.get("id")) not in producer
            and str(t.get("id")) in consumed
        ]
        outputs = [
            t for t in boundary if str(t.get("id")) in producer
            and str(t.get("id")) not in consumed
        ]
        if not outputs:
            # 有些模型（HiDream）块输出是 activation 而不是 boundary；用最后一个
            # 未被块内消费的真实输出，不造空“层输出”节点。
            for op in reversed(block.get("operators", [])):
                for tid in op.get("outputs", []):
                    tensor = next(
                        (t for t in block.get("tensors", [])
                         if str(t.get("id")) == str(tid)), None)
                    if tensor is None:
                        continue
                    if str(tensor.get("kind")) in ("weight", "cache"):
                        continue
                    if str(tid) not in consumed:
                        outputs = [tensor]
                        break
                if outputs:
                    break
        return (inputs[0] if inputs else None, outputs[0] if outputs else None)
    return hidden[0], hidden[1]


def _boundary_aux(block: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    main_in, main_out = _main_boundary(block)
    exclude = {str(t.get("id")) for t in (main_in, main_out) if t}
    return [
        t for t in block.get("tensors", [])
        if str(t.get("kind")) == "boundary" and str(t.get("id")) not in exclude
    ]


def _segment_state_nodes(
    block: Mapping[str, Any],
    ops: Sequence[Mapping[str, Any]],
    *,
    source_prefix: str,
) -> tuple[list[LevelNode], list[LevelEdge]]:
    """子图内的缓存前态/后态：状态节点复用，每个真实消费者保留输入依赖。"""
    tensors = _tensor_index(block)
    producer_all = {
        str(tid): str(op["id"])
        for op in block.get("operators", [])
        for tid in op.get("outputs", [])
    }
    consumer_all = {
        str(tid): str(op["id"])
        for op in block.get("operators", [])
        for tid in op.get("inputs", [])
    }
    nodes: list[LevelNode] = []
    edges: list[LevelEdge] = []
    # 一个状态张量可能有多个真实消费者（例如 conv state 同时供 conv1d 与
    # state_write 读取）；节点复用，但每个消费者都必须有一条状态读入边。
    pre_consumers: dict[str, list[str]] = {}
    post_producers: dict[str, list[str]] = {}
    op_order = {str(op["id"]): index for index, op in enumerate(ops)}
    for op in ops:
        op_id = str(op["id"])
        for tid in op.get("inputs", []):
            tid = str(tid)
            tensor = tensors.get(tid, {})
            if str(tensor.get("kind")) != "cache" or tid in producer_all:
                continue
            consumers = pre_consumers.setdefault(tid, [])
            if op_id not in consumers:
                consumers.append(op_id)
        for tid in op.get("outputs", []):
            tid = str(tid)
            tensor = tensors.get(tid, {})
            if str(tensor.get("kind")) != "cache" or tid in consumer_all:
                continue
            producers = post_producers.setdefault(tid, [])
            if op_id not in producers:
                producers.append(op_id)
    for tid in sorted(pre_consumers):
        consumers = sorted(pre_consumers[tid], key=lambda item: op_order.get(item, 0))
        tensor = tensors[tid]
        first = next(op for op in ops if str(op["id"]) == consumers[0])
        node_id = f"o-cache-{_slug(tid)}-pre"
        nodes.append(
            LevelNode(
                node_id=node_id,
                label=_tensor_name(tensor, tid),
                sublabel=_shape_of(tensor),
                tag="跨调用状态（调用前）",
                component_type="external",
                sources=_sources(first, prefix=source_prefix),
                role="state",
                anchor=f"o-{consumers[0]}",
                details=(
                    f"同一状态张量 {tid} 的真实消费者："
                    + "、".join(f"o-{op_id}" for op_id in consumers),
                ),
            )
        )
        for op_id in consumers:
            edges.append(
                LevelEdge(
                    src=node_id,
                    dst=f"o-{op_id}",
                    tensor_id=tid,
                    tensor_label=_tensor_name(tensor, tid),
                    shape=_shape_of(tensor),
                    kind="state",
                    label="状态读入",
                )
            )
    for tid in sorted(post_producers):
        producers = sorted(post_producers[tid], key=lambda item: op_order.get(item, 0))
        tensor = tensors[tid]
        first = next(op for op in ops if str(op["id"]) == producers[0])
        node_id = f"o-cache-{_slug(tid)}-post"
        nodes.append(
            LevelNode(
                node_id=node_id,
                label=_tensor_name(tensor, tid),
                sublabel=_shape_of(tensor),
                tag="跨调用状态（调用后）",
                component_type="external",
                sources=_sources(first, prefix=source_prefix),
                role="state",
                anchor=f"o-{producers[0]}",
            )
        )
        for op_id in producers:
            edges.append(
                LevelEdge(
                    src=f"o-{op_id}",
                    dst=node_id,
                    tensor_id=tid,
                    tensor_label=_tensor_name(tensor, tid),
                    shape=_shape_of(tensor),
                    kind="state",
                    label="状态写出",
                )
            )
    return nodes, edges


def _segment_exit(
    block: Mapping[str, Any],
    ops: Sequence[Mapping[str, Any]],
    *,
    preferred: str = "",
) -> str:
    """段出口：本段产出、段外消费的最后一个激活张量；退化到块主输出。"""
    tensors = _tensor_index(block)
    member_ids = {str(op["id"]) for op in ops}
    consumers: dict[str, list[str]] = {}
    for op in block.get("operators", []):
        for tid in op.get("inputs", []):
            consumers.setdefault(str(tid), []).append(str(op["id"]))
    candidates: list[str] = []
    for op in ops:
        for tid in op.get("outputs", []):
            tid = str(tid)
            tensor = tensors.get(tid, {})
            if str(tensor.get("kind")) in ("weight", "cache", "mask"):
                continue
            outside = [cid for cid in consumers.get(tid, []) if cid not in member_ids]
            if outside:
                candidates.append(tid)
    if candidates:
        preferred_hits = [tid for tid in candidates if preferred and preferred in tid]
        return preferred_hits[0] if preferred_hits else candidates[-1]
    main_in, main_out = _main_boundary(block)
    if main_out is not None:
        produced = {
            str(tid)
            for op in ops
            for tid in op.get("outputs", [])
        }
        if str(main_out.get("id")) in produced:
            return str(main_out["id"])
    if ops:
        for tid in ops[-1].get("outputs", []):
            if str(tensors.get(str(tid), {}).get("kind")) != "weight":
                return str(tid)
    return ""


def build_segment_graph(
    block: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
    modules: Sequence[Mapping[str, Any]],
    spec: SegmentSpec,
    profile: ModelGraphProfile,
    form: str = "normalized",
) -> LevelGraph:
    """Block 内部子图：真实模块集合 → 算子级单向数据流 + 缓存前态/后态。"""
    available = {str(module["id"]) for module in modules}
    missing = [mid for mid in spec.modules if mid not in available]
    if missing:
        raise ArchifyLayersError(
            f"层组 {group.get('id')} 的子图 {spec.level!r} 缺少模块 {missing}："
            "模型声明与 PageProfile 分组不一致，拒绝生成不完整图"
        )
    wanted = set(spec.modules)
    wanted_ops = {
        str(op.get("id"))
        for module in modules
        if str(module.get("id")) in wanted
        for op in module.get("ops", [])
    }
    ops = [op for op in block.get("operators", []) if str(op.get("id")) in wanted_ops]
    if not ops:
        raise ArchifyLayersError(f"子图 {spec.level!r} 没有算子")
    main_in, _main_out = _main_boundary(block)
    entry = spec.entry or (str(main_in["id"]) if main_in else "")
    exit_tensor = spec.exit or _segment_exit(block, ops, preferred=spec.level)
    graph = _ops_graph(
        block,
        ops,
        key=f"{spec.level}-{group.get('id')}",
        level=spec.level,
        title=spec.title,
        subtitle=(
            f"代表层 {group.get('representative_layer_index')} · {len(ops)} 算子 · "
            f"form={form}"
        ),
        variant=str(group.get("id")),
        group=str(group.get("id")),
        entry_tensor=entry,
        exit_tensor=exit_tensor,
        source_prefix=profile.repository.path_prefix,
        state_cache=True,
    )
    graph.form = form
    graph.notes = graph.notes + ((spec.note,) if spec.note else ())
    if form == "normalized":
        graph.expansions = tuple(normalization_audit(block))
    return graph


def normalization_audit(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    """保守化简审计：逐条规则给出候选与拒绝原因；没有证明就不改动原结构。

    返回记录进入 candidate 的卡片与关系 JSON 旁证，不写入产品结构事实，也不改变
    Excel 粒度。GLM/Qwen/HiDream 的算子已经是源码逐句语义算子；本轮实际应用 0 条。
    """
    ops = _all_ops(block)
    tensors = _tensor_index(block)
    producer = {
        str(tid): str(op["id"]) for op in ops for tid in op.get("outputs", [])
    }
    consumer: dict[str, list[str]] = {}
    for op in ops:
        for tid in op.get("inputs", []):
            consumer.setdefault(str(tid), []).append(str(op["id"]))

    def shapes(op: Mapping[str, Any]) -> tuple[list, list]:
        return (
            [tensors.get(str(tid), {}).get("shape") for tid in op.get("inputs", [])],
            [tensors.get(str(tid), {}).get("shape") for tid in op.get("outputs", [])],
        )

    def op_type(op: Mapping[str, Any]) -> str:
        return str(op.get("op") or "").lower()

    def candidate_ids(pairs, limit=4):
        return [str(item[0]) for item in pairs[:limit]]

    identity_views = []
    unit_views = []
    reshape_chains = []
    transpose_pairs = []
    split_concat = []
    transpose_into_matmul = []
    shape_ops = []
    for op in ops:
        kind = op_type(op)
        ins, outs = shapes(op)
        if kind == "shape":
            shape_ops.append(op.get("id"))
        if kind in ("view", "reshape") and len(ins) == 1 and len(outs) == 1:
            if ins[0] == outs[0]:
                identity_views.append((op.get("id"), ins[0], outs[0]))
            outer = [str(item) for item in (outs[0] or ())]
            inner = [str(item) for item in (ins[0] or ())]
            if (
                len(outer) == len(inner) + 1 and "1" in outer
            ) or (
                len(inner) == len(outer) + 1 and "1" in inner
            ):
                unit_views.append((op.get("id"), ins[0], outs[0]))
            downstream = consumer.get(str(op.get("outputs", [""])[0]), [])
            if len(downstream) == 1:
                nxt = next((c for c in ops if str(c.get("id")) == downstream[0]), None)
                if nxt is not None and op_type(nxt) in ("view", "reshape"):
                    reshape_chains.append((op.get("id"), nxt.get("id")))
        if kind == "transpose":
            for tid in op.get("outputs", []):
                for cid in consumer.get(str(tid), []):
                    nxt = next((c for c in ops if str(c.get("id")) == cid), None)
                    if nxt is None:
                        continue
                    if op_type(nxt) == "transpose":
                        transpose_pairs.append((op.get("id"), cid))
                    if any(token in str(cid).lower() for token in ("qk", "matmul", "score")):
                        transpose_into_matmul.append((op.get("id"), cid))
        if kind == "split":
            for tid in op.get("outputs", []):
                for cid in consumer.get(str(tid), []):
                    nxt = next((c for c in ops if str(c.get("id")) == cid), None)
                    if nxt is not None and op_type(nxt) == "concat":
                        split_concat.append((op.get("id"), cid))

    def record(rule, title, candidates, reason):
        return {
            "rule": rule,
            "title": title,
            "status": "rejected",
            "candidates": [str(item) for item in candidates],
            "reason": reason,
        }

    return [
        record(
            "R1", "形状计算化简", shape_ops,
            "本层组没有把 Shape/Gather/Concat 用于动态 shape 构造的算子；"
            "形状由既有事实直接给出，无需折叠。" if not shape_ops else
            "检测到 shape 构造候选，但本票未证明其只用于 shape 且不参与数据计算，保留原结构。",
        ),
        record(
            "R2", "恒等操作消除（同形 View/Reshape）",
            candidate_ids(identity_views),
            "本层组没有输入/输出形状完全相同的 View/Reshape；"
            "有形状变化的 View 是真实头/单位轴重排，不能删除。"
            if not identity_views else
            "存在同形 View/Reshape，但其输出仍被源码级张量身份引用；删除会改变节点/源码对应，保留。",
        ),
        record(
            "R3", "连续 Transpose 合并/抵消", candidate_ids(transpose_pairs),
            "本层组没有相邻且轴序复合为恒等的 Transpose 对；"
            "相隔真实计算（短卷积/注意力/缓存）的转置对不能抵消。"
            if not transpose_pairs else
            "检测到相邻 Transpose 候选，但轴语义/其它使用者未证明，保留原结构。",
        ),
        record(
            "R4", "连续 Reshape/View 合并", candidate_ids(reshape_chains),
            "本层组没有“View/Reshape 直接接 View/Reshape”的无计算链；"
            "中间存在真实算子时不能跨越合并。" if not reshape_chains else
            "检测到连续 View/Reshape 候选，但元素顺序/形状推导的合法域未证明，保留。",
        ),
        record(
            "R5", "单位轴相关化简（View + 单头轴）", candidate_ids(unit_views),
            "单位轴是真实 KV/头轴：后续按该 4D 形状进入注意力/广播；"
            "改写成单一 Reshape 需同时证明下游索引与广播等价，本票未证明，保留。"
            if unit_views else
            "本层组没有新增/删除单位轴的 View/Reshape 候选。",
        ),
        record(
            "R6", "Split/Concat 抵消", candidate_ids(split_concat),
            "本层组没有“完整切分后原样同序拼回”的直接 Split→Concat；"
            "中间有 RoPE/卷积/扫描等真实计算时不能抵消。" if not split_concat else
            "检测到 Split→Concat 候选，但中间语义/其它使用者未证明，保留。",
        ),
        record(
            "R7", "Transpose 并入矩阵乘语义", candidate_ids(transpose_into_matmul),
            "转置表达的是 head 布局，并与 mask/两趟注意力或缓存共享；"
            "并入 QK^T 会丢失显式轴语义与源码对应，本票未应用。"
            if transpose_into_matmul else
            "本层组没有紧邻 QK/MatMul 且可证明只服务该乘法的 Transpose。",
        ),
    ]


def _build_network_phase_graph(
    network: Mapping[str, Any], profile: ModelGraphProfile
) -> LevelGraph:
    """整网阶段图：把很长的单次 forward 收成入口级阶段；细节逐 op 留在节点详情。

    阶段由模型声明给出（label/tag/真实 op ids），不是按渲染方便任意合并；阶段之间的
    张量关系仍逐条聚合（members 保留每个 tensor id/全称维度）。
    """
    path = network.get("path") or {}
    ops = list(path.get("operators") or [])
    if not ops:
        raise ArchifyLayersError("整网文档缺少 path.operators")
    tensors = _tensor_index(path)
    scope = network.get("scope") or {}
    op_of = {str(op["id"]): op for op in ops}
    phases: list[tuple[str, str, list[Mapping[str, Any]]]] = []
    phase_of: dict[str, int] = {}
    for label, tag, op_ids in profile.network_phases:
        members = [op_of[oid] for oid in op_ids if oid in op_of]
        if not members:
            continue
        index = len(phases)
        phases.append((label, tag, members))
        for op in members:
            phase_of[str(op["id"])] = index
    missing = [str(op["id"]) for op in ops if str(op["id"]) not in phase_of]
    if missing:
        raise ArchifyLayersError(
            f"整网阶段声明未覆盖算子 {missing}：不静默丢节点"
        )
    producer = {
        str(tid): str(op["id"]) for op in ops for tid in op.get("outputs", [])
    }
    consumed = {str(tid) for op in ops for tid in op.get("inputs", [])}
    main_input = next(
        (
            str(tid)
            for op in ops
            for tid in op.get("inputs", [])
            if str(tid) not in producer
            and str(tensors.get(str(tid), {}).get("kind")) != "weight"
            and str(tensors.get(str(tid), {}).get("kind")) != "cache"
        ),
        "",
    )
    outputs = [
        str(tid) for op in ops for tid in op.get("outputs", [])
        if str(tid) not in consumed
    ]
    preferred = [tid for tid in outputs if tid in ("logits", "x_pred", "n_x_pred")]
    output = preferred[-1] if preferred else (outputs[-1] if outputs else "")
    loop = next((op for op in ops if "layer_loop" in str(op["id"])), None)
    graph = LevelGraph(
        key="network",
        level="network",
        title=profile.network_title,
        subtitle=(
            f"入口 {scope.get('entry_class', '')} · {scope.get('layer_count', '')} 层 · "
            f"hidden_size={scope.get('hidden_size', '')} · "
            f"{len(phases)} 个真实阶段（单次 forward，不展开 36 层/27 blocks）"
        ),
        notes=tuple(path.get("coverage_notes", [])),
        parameter_items=_parameter_items(path),
        form="normalized",
    )
    graph.nodes.append(
        LevelNode(
            node_id="n-input",
            label=_tensor_name(tensors.get(main_input), main_input or "input"),
            sublabel=_shape_of(tensors.get(main_input)),
            tag="输入张量",
            component_type="external",
            sources=_sources(ops[0], prefix=profile.repository.path_prefix),
            role="boundary_in",
        )
    )
    for index, (label, tag, members) in enumerate(phases):
        sources: list[dict[str, Any]] = []
        for op in (members[0], members[-1]):
            for source in _sources(op, prefix=profile.repository.path_prefix):
                if source not in sources:
                    sources.append(source)
        graph.nodes.append(
            LevelNode(
                node_id=f"n-phase-{index}",
                label=label,
                sublabel=f"{len(members)} 算子 · " + "、".join(
                    _op_label(op) for op in members[:2]
                ) + ("…" if len(members) > 2 else ""),
                tag=tag,
                component_type="backend",
                sources=tuple(sources[:3]),
                role="main",
                details=tuple(
                    f"{_op_label(op)}（{op['id']} · {op.get('module_path', '')}）"
                    for op in members
                ),
                op_ids=tuple(str(op["id"]) for op in members),
            )
        )
    graph.nodes.append(
        LevelNode(
            node_id="n-out",
            label=_tensor_name(tensors.get(output), output or "output"),
            sublabel=_shape_of(tensors.get(output)),
            tag="输出张量",
            component_type="external",
            sources=_sources(ops[-1], prefix=profile.repository.path_prefix),
            role="boundary_out",
        )
    )
    seen: set[tuple[str, str, str]] = set()
    edge_records: list[LevelEdge] = []

    def add(src: str, dst: str, tensor_id: str, *, kind: str = "data",
            label: str = "") -> None:
        key = (src, dst, tensor_id)
        if key in seen or src == dst:
            return
        seen.add(key)
        tensor = tensors.get(tensor_id)
        edge_records.append(
            LevelEdge(
                src=src, dst=dst, tensor_id=tensor_id,
                tensor_label=_tensor_name(tensor, tensor_id),
                shape=_shape_of(tensor), kind=kind, label=label,
            )
        )

    for op in ops:
        dst_index = phase_of[str(op["id"])]
        dst = f"n-phase-{dst_index}"
        for tid in op.get("inputs", []):
            tid = str(tid)
            tensor = tensors.get(tid)
            kind = str(tensor.get("kind") if tensor else "")
            if kind in ("weight", "cache"):
                continue
            src_op = producer.get(tid)
            if src_op is None:
                if tid == main_input:
                    add("n-input", dst, tid)
                continue
            src_index = phase_of.get(src_op)
            if src_index is None or src_index == dst_index:
                continue
            add(f"n-phase-{src_index}", dst, tid)
    if output and output in producer:
        out_index = phase_of.get(producer[output])
        if out_index is not None:
            add(f"n-phase-{out_index}", "n-out", output)
    # 未由算子产生的辅助输入（timestep / pixel_values / position_ids …）短旁枝。
    aux_inputs = sorted({
        str(tid)
        for op in ops
        for tid in op.get("inputs", [])
        if str(tid) not in producer
        and str(tid) != main_input
        and str(tensors.get(str(tid), {}).get("kind")) not in ("weight", "cache")
    })
    for tid in aux_inputs:
        tensor = tensors.get(tid)
        consumer = next((op for op in ops if tid in op.get("inputs", [])), ops[0])
        node_id = f"n-aux-{_slug(tid)}"
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=_tensor_name(tensor, tid),
                sublabel=_shape_of(tensor),
                tag="辅助输入（短旁枝）",
                component_type="external",
                sources=_sources(consumer, prefix=profile.repository.path_prefix),
                role="aux",
                anchor=f"n-phase-{phase_of[str(consumer['id'])]}",
            )
        )
        add(node_id, f"n-phase-{phase_of[str(consumer['id'])]}", tid,
            kind="aux", label="辅助输入")
    # 跨调用缓存（本网络没有）：仍按真实 cache 张量表达，不虚构。
    cache_pre = sorted(
        str(t["id"]) for t in path.get("tensors", [])
        if str(t.get("kind")) == "cache" and str(t["id"]) not in producer
    )
    cache_post = sorted(
        str(t["id"]) for t in path.get("tensors", [])
        if str(t.get("kind")) == "cache"
        and str(t["id"]) in producer and str(t["id"]) not in consumed
    )
    if loop is not None:
        loop_phase = f"n-phase-{phase_of[str(loop['id'])]}"
        for prefix, ids, label in (
            ("n-cache-in", cache_pre, "状态读入"),
            ("n-cache-out", cache_post, "状态写出"),
        ):
            for tid in ids:
                tensor = tensors.get(tid)
                node_id = f"{prefix}-{_slug(tid)}"
                graph.nodes.append(
                    LevelNode(
                        node_id=node_id,
                        label=_tensor_name(tensor, tid),
                        sublabel=_shape_of(tensor),
                        tag="跨调用状态",
                        component_type="external",
                        sources=_sources(loop, prefix=profile.repository.path_prefix),
                        role="state",
                        anchor=loop_phase,
                    )
                )
                if prefix == "n-cache-in":
                    add(node_id, loop_phase, tid, kind="state", label=label)
                else:
                    add(loop_phase, node_id, tid, kind="state", label=label)
    graph.edges = _aggregate_module_edges(edge_records)
    _assert_acyclic(graph)
    return graph


def build_network_graph_for(
    network: Mapping[str, Any], profile: ModelGraphProfile
) -> LevelGraph:
    """整网：真实 path.operators 单项投影；辅助算子作短旁枝，缓存前/后态分开。"""
    if profile.network_phases:
        return _build_network_phase_graph(network, profile)
    path = network.get("path") or {}
    ops = list(path.get("operators") or [])
    if not ops:
        raise ArchifyLayersError("整网文档缺少 path.operators")
    tensors = _tensor_index(path)
    scope = network.get("scope") or {}
    main_path = [str(item) for item in scope.get("main_path") or ()]
    op_ids = {str(op["id"]) for op in ops}
    side_prefixes = tuple(profile.network_side_prefixes)
    # 辅助前缀优先于 scope.main_path：mask/RoPE/位置编码不应把主链挤到另一列。
    side_ids = [
        str(op["id"]) for op in ops
        if any(str(op["id"]).startswith(prefix) for prefix in side_prefixes)
    ]
    main_ids = [
        oid for oid in main_path if oid in op_ids and oid not in set(side_ids)
    ] or [str(op["id"]) for op in ops if str(op["id"]) not in set(side_ids)]
    main_set = set(main_ids)
    # 既不在 main_path 也不匹配旁枝前缀的算子仍保留为真实节点（不隐藏）。
    extra_ids = [
        str(op["id"]) for op in ops
        if str(op["id"]) not in main_set and str(op["id"]) not in set(side_ids)
    ]
    main_ops = [op for op in ops if str(op["id"]) in main_set]
    side_ops = [op for op in ops if str(op["id"]) in set(side_ids)]
    extra_ops = [op for op in ops if str(op["id"]) in set(extra_ids)]
    producer = {
        str(tid): str(op["id"]) for op in ops for tid in op.get("outputs", [])
    }
    consumed = {
        str(tid) for op in ops for tid in op.get("inputs", [])
    }
    consumer: dict[str, list[str]] = {}
    for op in ops:
        for tid in op.get("inputs", []):
            consumer.setdefault(str(tid), []).append(str(op["id"]))
    main_input = next(
        (
            str(tid)
            for op in main_ops
            for tid in op.get("inputs", [])
            if str(tid) not in producer
            and str(tensors.get(str(tid), {}).get("kind")) != "weight"
        ),
        "",
    )
    output_candidates = [
        str(tid)
        for op in main_ops
        for tid in op.get("outputs", [])
        if str(tid) not in consumed
    ]
    preferred = [tid for tid in output_candidates if tid in ("logits", "x_pred", "n_x_pred")]
    output = preferred[-1] if preferred else (output_candidates[-1] if output_candidates else "")
    loop = next((op for op in ops if "layer_loop" in str(op["id"])), None)
    graph = LevelGraph(
        key="network",
        level="network",
        title=profile.network_title,
        subtitle=(
            f"入口 {scope.get('entry_class', '')} · {scope.get('layer_count', '')} 层 · "
            f"hidden_size={scope.get('hidden_size', '')} · 层循环不铺开"
        ),
        notes=tuple(path.get("coverage_notes", [])),
        parameter_items=_parameter_items(path),
        form="normalized",
    )
    graph.nodes.append(
        LevelNode(
            node_id="n-input",
            label=_tensor_name(tensors.get(main_input), main_input or "input"),
            sublabel=_shape_of(tensors.get(main_input)),
            tag="输入张量",
            component_type="external",
            sources=_sources(ops[0], prefix=profile.repository.path_prefix),
            role="boundary_in",
        )
    )
    for op in main_ops:
        graph.nodes.append(
            LevelNode(
                node_id=f"n-{op['id']}",
                label=_op_label(op),
                sublabel=_op_sublabel(op),
                tag=_node_tag(op.get("module_path", ""), "整网"),
                component_type="backend",
                sources=_sources(op, prefix=profile.repository.path_prefix),
                role="main",
                details=(str(op.get("note", "")),),
                op_ids=(str(op["id"]),),
            )
        )
    graph.nodes.append(
        LevelNode(
            node_id="n-out",
            label=_tensor_name(tensors.get(output), output or "output"),
            sublabel=_shape_of(tensors.get(output)),
            tag="输出张量",
            component_type="external",
            sources=_sources(main_ops[-1], prefix=profile.repository.path_prefix),
            role="boundary_out",
        )
    )
    seen: set[tuple[str, str, str]] = set()

    def add(src: str, dst: str, tensor_id: str, *, kind: str = "data", label: str = "") -> None:
        key = (src, dst, tensor_id)
        if key in seen or src == dst:
            return
        seen.add(key)
        tensor = tensors.get(tensor_id)
        graph.edges.append(
            LevelEdge(
                src=src,
                dst=dst,
                tensor_id=tensor_id,
                tensor_label=_tensor_name(tensor, tensor_id),
                shape=_shape_of(tensor),
                kind=kind,
                label=label,
            )
        )

    for op in ops:
        op_id = str(op["id"])
        for tid in op.get("inputs", []):
            tid = str(tid)
            tensor = tensors.get(tid)
            if str(tensor.get("kind") if tensor else "") == "weight":
                continue
            if str(tensor.get("kind") if tensor else "") == "cache":
                continue
            src = producer.get(tid)
            if src is not None:
                add(f"n-{src}", f"n-{op_id}", tid)
            elif tid == main_input:
                add("n-input", f"n-{op_id}", tid)
        for tid in op.get("outputs", []):
            tid = str(tid)
            if tid == output:
                add(f"n-{op_id}", "n-out", tid)
    # 旁枝算子（位置编码/mask/RoPE 等）：贴住其消费者，局部短旁枝。
    for op in side_ops + extra_ops:
        op_id = str(op["id"])
        consumer_main = next(
            (
                str(candidate["id"])
                for candidate in ops
                if str(candidate["id"]) in main_set
                and any(
                    str(tid) in producer and producer[str(tid)] == op_id
                    for tid in candidate.get("inputs", [])
                )
            ),
            "",
        )
        producer_main = next(
            (
                str(candidate["id"])
                for candidate in ops
                if str(candidate["id"]) in main_set
                and any(
                    str(tid) in consumer and op_id in consumer[str(tid)]
                    for tid in candidate.get("outputs", [])
                )
            ),
            "",
        )
        graph.nodes.append(
            LevelNode(
                node_id=f"n-{op_id}",
                label=_op_label(op),
                sublabel=_op_sublabel(op),
                tag="网络级辅助/分支",
                component_type="backend",
                sources=_sources(op, prefix=profile.repository.path_prefix),
                role="branch",
                anchor=(f"n-{consumer_main}" if consumer_main else
                        (f"n-{producer_main}" if producer_main else None)),
                details=(str(op.get("note", "")),),
                op_ids=(op_id,),
            )
        )
    # 未由任何算子产生的辅助输入（attention_mask / timestep / pixel_values …）。
    aux_inputs = sorted(
        {
            str(tid)
            for op in ops
            for tid in op.get("inputs", [])
            if str(tid) not in producer
            and str(tid) != main_input
            and str(tensors.get(str(tid), {}).get("kind")) != "weight"
            and str(tensors.get(str(tid), {}).get("kind")) != "cache"
        }
    )
    for tid in aux_inputs:
        tensor = tensors.get(tid)
        first_consumer = next(
            (op for op in ops if tid in op.get("inputs", [])), ops[0]
        )
        graph.nodes.append(
            LevelNode(
                node_id=f"n-aux-{_slug(tid)}",
                label=_tensor_name(tensor, tid),
                sublabel=_shape_of(tensor),
                tag="辅助输入（短旁枝）",
                component_type="external",
                sources=_sources(first_consumer, prefix=profile.repository.path_prefix),
                role="aux",
                anchor=f"n-{first_consumer['id']}",
            )
        )
        add(f"n-aux-{_slug(tid)}", f"n-{first_consumer['id']}", tid, kind="aux", label="辅助输入")
    # 跨调用缓存：调用前状态 → loop，loop → 调用后状态（单向）。
    cache_pre = sorted(
        str(t["id"]) for t in path.get("tensors", [])
        if str(t.get("kind")) == "cache" and str(t["id"]) not in producer
    )
    cache_post = sorted(
        str(t["id"]) for t in path.get("tensors", [])
        if str(t.get("kind")) == "cache"
        and str(t["id"]) in producer
        and str(t["id"]) not in consumed
    )
    if loop is not None:
        for prefix, ids, suffix, label in (
            ("n-cache-in", cache_pre, "（调用前）", "状态读入"),
            ("n-cache-out", cache_post, "（调用后）", "状态写出"),
        ):
            for tid in ids:
                tensor = tensors.get(tid)
                graph.nodes.append(
                    LevelNode(
                        node_id=f"{prefix}-{_slug(tid)}",
                        label=_tensor_name(tensor, tid),
                        sublabel=_shape_of(tensor),
                        tag="跨调用状态",
                        component_type="external",
                        sources=_sources(loop, prefix=profile.repository.path_prefix),
                        role="state",
                        anchor=f"n-{loop['id']}",
                    )
                )
                if suffix in ("（调用前）",):
                    graph.edges.append(
                        LevelEdge(
                            src=f"{prefix}-{_slug(tid)}", dst=f"n-{loop['id']}",
                            tensor_id=tid, tensor_label=_tensor_name(tensor, tid),
                            shape=_shape_of(tensor), kind="state", label=label,
                        )
                    )
                else:
                    graph.edges.append(
                        LevelEdge(
                            src=f"n-{loop['id']}", dst=f"{prefix}-{_slug(tid)}",
                            tensor_id=tid, tensor_label=_tensor_name(tensor, tid),
                            shape=_shape_of(tensor), kind="state", label=label,
                        )
                    )
    _assert_acyclic(graph)
    return graph


def build_group_graph_for(
    network: Mapping[str, Any],
    group: Mapping[str, Any],
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    profile: ModelGraphProfile,
) -> LevelGraph:
    """层组：整网层循环 → 本组真实层 → 代表层 → Block 模板。"""
    scope = network.get("scope") or {}
    gid = str(group["id"])
    indices = [int(i) for i in group.get("layer_indices", [])]
    rep = group.get("representative_layer_index")
    op_count = len(block.get("operators", []))
    graphical = profile.display
    is_vision = "vit" in gid or "vision" in str(block.get("kind", ""))
    decoder_class = (
        scope.get("vision_block_class") if is_vision else scope.get("decoder_layer_class")
    ) or ("Qwen3VLVisionBlock" if is_vision else scope.get("decoder_layer_class", "Block"))
    instance = (
        f"实例 model.visual.blocks.{rep}" if is_vision else f"实例 model.layers.{rep}"
    )
    main_in, _ = _main_boundary(block)
    graph = LevelGraph(
        key=f"group-{gid}",
        level="group",
        title=f"{graphical} · 层组 {group.get('label', gid)}（{len(indices)} 层）",
        subtitle=(
            f"{group.get('attention_type', '')} + {group.get('moe_type', '')} · "
            f"代表层 {rep} · 同组共用一份 Block 模板，模板复用不代表共享权重"
        ),
        notes=tuple(block.get("scope_notes", [])),
        parameter_items=_parameter_items(block, limit=12),
        variant=gid,
        group=gid,
    )
    graph.nodes.extend(
        [
            LevelNode(
                node_id="g-loop",
                label=f"{len(indices)} 层循环",
                sublabel=(
                    f"本组真实层数；整网共 {scope.get('layer_count', '')} 层"
                    "（本图只展开当前层组）"
                ),
                tag="整网上下文",
                component_type="external",
                sources=(),
                role="boundary_in",
            ),
            LevelNode(
                node_id="g-group",
                label=f"本组 {len(indices)} 层",
                sublabel="、".join(str(i) for i in indices[:10])
                + ("…" if len(indices) > 10 else ""),
                tag="层组归属",
                component_type="backend",
                sources=_group_sources(block),
                role="main",
                details=(
                    f"真实层索引：{', '.join(str(i) for i in indices)}",
                    f"attention_type={group.get('attention_type', '')}；"
                    f"moe_type={group.get('moe_type', '')}",
                ),
            ),
            LevelNode(
                node_id="g-rep",
                label=f"代表层 {rep}",
                sublabel=f"{decoder_class} · {instance}",
                tag="代表层",
                component_type="backend",
                sources=_group_sources(block),
                role="main",
                details=(str(block.get("title", "")),),
            ),
            LevelNode(
                node_id="g-block",
                label=f"Block 模板（{len(modules)} 模块 / {op_count} 算子）",
                sublabel=(
                    f"{group.get('attention_type', '')} + {group.get('moe_type', '')} · "
                    "进入 Block 层看到真实张量数据流"
                ),
                tag="Block 模板",
                component_type="backend",
                sources=_group_sources(block),
                role="main",
                details=tuple(
                    f"{module['title']}（{len(module['ops'])} 算子）" for module in modules
                ),
            ),
        ]
    )
    graph.nodes.append(
        LevelNode(
            node_id="g-other-groups",
            label="其余层组",
            sublabel="、".join(
                f"{item['id']}({item['count']})"
                for item in network.get("groups", [])
                if item["id"] != gid
            ),
            tag="整网上下文",
            component_type="external",
            sources=(),
            role="aux",
            anchor="g-group",
        )
    )
    for src, dst, label in (
        ("g-loop", "g-group", f"{len(indices)} 层中的本组"),
        ("g-group", "g-rep", "取代表层生成模板"),
        ("g-rep", "g-block", "模块级模板"),
    ):
        tensor = main_in
        graph.edges.append(
            LevelEdge(
                src=src,
                dst=dst,
                tensor_id=str(tensor["id"]) if tensor else "layer",
                tensor_label=_tensor_name(tensor, "层输入"),
                shape=_shape_of(tensor),
                kind="data",
                label=label,
            )
        )
    graph.edges.append(
        LevelEdge(
            src="g-other-groups",
            dst="g-group",
            tensor_id="layer_order",
            tensor_label="同构其它层组（不展开）",
            shape="",
            kind="aux",
            label="同构模板",
        )
    )
    return graph


def build_decoder_graph_for(
    block: Mapping[str, Any],
    *,
    modules: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
    profile: ModelGraphProfile,
) -> LevelGraph:
    """Block 数据流：真实拓扑阶段（跨模块依赖切开）+ 全缓存前/后态 + 辅助短旁枝。"""
    gid = str(group["id"])
    tensors = _tensor_index(block)
    stages = _module_stages(block, modules)
    main_in, main_out = _main_boundary(block)
    graph = LevelGraph(
        key=f"decoder-{gid}",
        level="decoder",
        title=f"{profile.display} · {group.get('label', gid)} Block 数据流（模块阶段级）",
        subtitle=(
            f"代表层 {group.get('representative_layer_index')} · "
            f"{group.get('attention_type', '')} + {group.get('moe_type', '')} · "
            f"{len(modules)} 模块 / {len(stages)} 阶段 / {len(block.get('operators', []))} 算子"
        ),
        notes=tuple(block.get("cache_notes", [])) + tuple(block.get("coverage_notes", [])),
        parameter_items=_parameter_items(block),
        variant=gid,
        group=gid,
    )
    stage_of_op: dict[str, str] = {}
    for stage in stages:
        for op_id in stage["ops"]:
            stage_of_op[op_id] = stage["stage_id"]
    graph.nodes.append(
        LevelNode(
            node_id="d-in",
            label=_tensor_name(main_in, "层输入"),
            sublabel=_shape_of(main_in),
            tag="输入张量",
            component_type="external",
            sources=_group_sources(block),
            role="boundary_in",
        )
    )
    op_index = _op_index(block)
    for stage in stages:
        ops = [op_index[op_id] for op_id in stage["ops"]]
        graph.nodes.append(
            LevelNode(
                node_id=f"m-{stage['stage_id']}",
                label=stage["label"],
                sublabel=f"{len(ops)} 算子 · " + "、".join(
                    _module_tail(op.get("module_path", "")) for op in ops[:2]
                ),
                tag=stage["module_id"],
                component_type="backend",
                sources=_sources(ops[0], prefix=profile.repository.path_prefix),
                role="main",
                details=tuple(
                    f"{_op_label(op)}（{op.get('id')} · {op.get('module_path', '')}）"
                    for op in ops
                ),
                op_ids=tuple(str(op.get("id")) for op in ops),
            )
        )
    graph.nodes.append(
        LevelNode(
            node_id="d-out",
            label=_tensor_name(main_out, "层输出"),
            sublabel=_shape_of(main_out),
            tag="输出张量",
            component_type="external",
            sources=_group_sources(block),
            role="boundary_out",
        )
    )
    seen: set[tuple[str, str, str]] = set()

    def add(src: str, dst: str, tensor_id: str, *, kind: str = "data", label: str = "") -> None:
        key = (src, dst, tensor_id)
        if key in seen or src == dst:
            return
        seen.add(key)
        tensor = tensors.get(tensor_id)
        graph.edges.append(
            LevelEdge(
                src=src, dst=dst, tensor_id=tensor_id,
                tensor_label=_tensor_name(tensor, tensor_id),
                shape=_shape_of(tensor), kind=kind, label=label,
            )
        )

    producer = {
        str(tid): str(op["id"])
        for op in block.get("operators", [])
        for tid in op.get("outputs", [])
    }
    consumed = {
        str(tid)
        for op in block.get("operators", [])
        for tid in op.get("inputs", [])
    }
    for op in block.get("operators", []):
        dst_stage = stage_of_op.get(str(op["id"]))
        if dst_stage is None:
            continue
        dst = f"m-{dst_stage}"
        for tensor_id in op.get("inputs", []):
            tensor_id = str(tensor_id)
            tensor = tensors.get(tensor_id)
            kind = str(tensor.get("kind") if tensor else "")
            if kind in ("weight", "cache"):
                continue
            src_op = producer.get(tensor_id)
            if src_op is None:
                if main_in is not None and tensor_id == str(main_in.get("id")):
                    add("d-in", dst, tensor_id)
                continue
            src_stage = stage_of_op.get(src_op)
            if src_stage is None:
                continue
            add(f"m-{src_stage}", dst, tensor_id)
    if main_out is not None:
        main_out_id = str(main_out.get("id"))
        for op in block.get("operators", []):
            src_stage = stage_of_op.get(str(op["id"]))
            if src_stage is None:
                continue
            if main_out_id in [str(item) for item in op.get("outputs", [])]:
                add(f"m-{src_stage}", "d-out", main_out_id)
    # 辅助边界输入：贴住第一个消费阶段，短旁枝。
    for tensor in _boundary_aux(block):
        tid = str(tensor["id"])
        consumer_op = next(
            (op for op in block.get("operators", []) if tid in op.get("inputs", [])),
            None,
        )
        if consumer_op is None:
            continue
        stage = stage_of_op.get(str(consumer_op["id"]))
        if stage is None:
            continue
        node_id = f"d-aux-{_slug(tid)}"
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=_tensor_name(tensor, tid),
                sublabel=_shape_of(tensor),
                tag="辅助输入（短旁枝）",
                component_type="external",
                sources=_sources(consumer_op, prefix=profile.repository.path_prefix),
                role="aux",
                anchor=f"m-{stage}",
                details=(str(tensor.get("note", "")),),
            )
        )
        add(node_id, f"m-{stage}", tid, kind="aux", label="辅助输入")
    # 全部缓存前态/后态：每个真实缓存张量一条 state 关系。
    cache_pre: set[str] = set()
    cache_post: set[str] = set()
    for tensor in block.get("tensors", []):
        tid = str(tensor["id"])
        if str(tensor.get("kind")) != "cache":
            continue
        if tid not in producer:
            cache_pre.add(tid)
        elif tid not in consumed:
            cache_post.add(tid)
    for tid in sorted(cache_pre):
        tensor = tensors[tid]
        consumer_ops = [
            op for op in block.get("operators", []) if tid in op.get("inputs", [])
        ]
        stages = []
        for consumer_op in consumer_ops:
            stage = stage_of_op.get(str(consumer_op["id"]))
            if stage is not None and stage not in stages:
                stages.append(stage)
        if not stages:
            continue
        node_id = f"d-cache-{_slug(tid)}-pre"
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=_tensor_name(tensor, tid),
                sublabel=_shape_of(tensor),
                tag="跨调用状态（调用前）",
                component_type="external",
                sources=_sources(consumer_ops[0], prefix=profile.repository.path_prefix),
                role="state",
                anchor=f"m-{stages[0]}",
                details=(
                    f"同一状态张量 {tid} 的真实消费阶段："
                    + "、".join(f"m-{stage}" for stage in stages),
                ),
            )
        )
        for stage in stages:
            add(node_id, f"m-{stage}", tid, kind="state", label="状态读入")
    for tid in sorted(cache_post):
        tensor = tensors[tid]
        source_ops = [
            op for op in block.get("operators", []) if tid in op.get("outputs", [])
        ]
        stages = []
        for source_op in source_ops:
            stage = stage_of_op.get(str(source_op["id"]))
            if stage is not None and stage not in stages:
                stages.append(stage)
        if not stages:
            continue
        node_id = f"d-cache-{_slug(tid)}-post"
        graph.nodes.append(
            LevelNode(
                node_id=node_id,
                label=_tensor_name(tensor, tid),
                sublabel=_shape_of(tensor),
                tag="跨调用状态（调用后）",
                component_type="external",
                sources=_sources(source_ops[0], prefix=profile.repository.path_prefix),
                role="state",
                anchor=f"m-{stages[0]}",
            )
        )
        for stage in stages:
            add(f"m-{stage}", node_id, tid, kind="state", label="状态写出")
    graph.edges = _aggregate_module_edges(graph.edges)
    _assert_acyclic(graph)
    return graph


def segment_requests(
    model_key: str,
    structures: Sequence[Mapping[str, Any]],
    module_groups_fn,
) -> list:
    """按模型声明生成图请求：整网 + 每组 group/decoder + 真实子图（含 raw 对照）。"""
    profile = graph_profile_for(model_key)
    if model_key == "deepseek_v4_flash_0731_hf":
        return _deepseek_requests([str(item["id"]) for item in structures],
                                  structures=structures)
    requests: list[LevelRequest] = [LevelRequest("network", "network")]
    for item in structures:
        gid = str(item["id"])
        block = item["structure"]["block"]
        modules = module_groups_fn(block)
        requests.append(LevelRequest("group", gid))
        requests.append(LevelRequest("decoder", gid))
        for spec in profile.segments(block, _group_stub(item, block), modules):
            requests.append(LevelRequest(spec.level, gid, "normalized"))
            if spec.pair_raw:
                requests.append(LevelRequest(spec.level, gid, "raw"))
    return requests


def _group_stub(item: Mapping[str, Any], block: Mapping[str, Any]) -> dict[str, Any]:
    """层组声明：整网 groups 里的条目 + 块级 / 层组字段回退。"""
    scope = block.get("scope") or {}
    out = {
        "id": item.get("id"),
        "label": item.get("label"),
        "layer_indices": item.get("layer_indices"),
        "representative_layer_index": item.get("representative_layer_index"),
        "attention_type": item.get("attention_type") or scope.get("attention_type"),
        "moe_type": item.get("moe_type") or scope.get("moe_type"),
        "indexer_type": scope.get("indexer_type") or block.get("indexer_type"),
    }
    return out


def _deepseek_requests(
    variant_ids: list[str],
    *,
    structures: Sequence[Mapping[str, Any]] | None = None,
):
    # 整网 + 两个简单端点 + 四组分层子图 + D03 缓存/压缩器/Indexer。
    requests = [
        LevelRequest("network", "network"),
        LevelRequest("embedding", "network"),
        LevelRequest("head", "network"),
    ]
    by_id = {str(item.get("id")): item for item in structures or ()}
    for gid in variant_ids:
        item = by_id.get(gid) or {}
        attention_type = str(item.get("attention_type") or "")
        requests.extend([
            LevelRequest("group", gid),
            LevelRequest("decoder", gid),
            LevelRequest("kv", gid, "normalized"),
            LevelRequest("attention", gid, "normalized"),
            LevelRequest("q", gid),
            LevelRequest("o", gid),
            LevelRequest("moe", gid),
            LevelRequest("moe-route", gid),
            LevelRequest("moe-expert", gid),
            LevelRequest("moe-shared", gid),
            LevelRequest("moe-combine", gid),
            LevelRequest("mhc", gid),
            LevelRequest("cache-main", gid),
        ])
        if attention_type != "sliding_attention":
            requests.append(LevelRequest("cache-compressor", gid))
            requests.append(LevelRequest("compressor", gid))
        if attention_type == "compressed_sparse_attention":
            requests.append(LevelRequest("cache-indexer", gid))
            requests.append(LevelRequest("indexer", gid))
    for view in ("kv", "attention"):
        requests.append(LevelRequest(view, "csa-topk", "raw"))
    return requests
