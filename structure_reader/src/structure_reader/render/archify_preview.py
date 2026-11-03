"""Archify 预览生成：从同一语义结果程序生成 KV / 完整 Attention 的可查看图。

- 工具发现：``ARCHIFY_CLI`` 环境变量 → ``structure_reader/configs/tools/archify.json``
  → 常规安装位置（``~/.dsh/skills/archify``、``~/.claude/skills/archify``）。
  不把某台机器的个人绝对路径硬编码进产品；实际解析到的路径、CLI/package 哈希与
  Node 版本写进预览 manifest（工具身份进入产物身份）。
- 图生成：视图（kv / attention）由语义 bundle 的旧算子映射 + 张量边界程序化导出，
  主链纵向单向展开，辅助输入紧贴消费节点，状态读/写以虚线旁枝表达；原始图与
  化简图使用同一生成器（只是输入 bundle 不同），不做手抄。
- 来源：组件 ``sources`` 使用 hflib/transformers 检出（真实 origin + 40 位 commit）
  的仓库相对路径与行区间；``finalize --repo-root`` 再核验。

本模块不执行模型、不读权重；输出 candidate JSON 与调用 Archify CLI 的 receipt。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from structure_reader.semantic.bundle import SemanticBundle, SourceRef

__all__ = [
    "ArchifyToolError",
    "PreviewError",
    "ArchifyTool",
    "discover_archify",
    "tool_identity",
    "build_view_model",
    "build_candidate",
    "finalize_candidate",
    "generate_previews",
]

CONFIG_REL = Path("structure_reader") / "configs" / "tools" / "archify.json"
_TRANSFORMERS_REPO_KEYS = ("transformers",)
_MAX_SOURCES = 3

_KIND_LABELS = {
    "linear": "Linear",
    "grouped_linear": "GroupedLinear",
    "rmsnorm": "RMSNorm",
    "view": "View",
    "transpose": "Transpose",
    "reshape": "Reshape",
    "rope": "RoPE",
    "matmul": "MatMul",
    "scale": "Scale",
    "add": "Add",
    "concat": "Concat",
    "softmax": "Softmax",
    "broadcast": "Broadcast",
    "cache_update": "Cache",
    "composite": "具名模块",
    "state_in": "状态（调用前）",
    "state_out": "状态（调用后）",
    "aux": "辅助输入",
    "external": "输入",
    "external_out": "输出",
}


class ArchifyToolError(ValueError):
    """找不到可用的 Archify CLI，或工具身份不可核对（具名拒绝）。"""


class PreviewError(ValueError):
    """视图/布局/候选生成失败（具名拒绝）。"""


@dataclass(frozen=True)
class ArchifyTool:
    cli: Path
    discovery: str
    version: str | None
    package_json: Path | None


def _candidate_paths(workspace_root: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    env = os.environ.get("ARCHIFY_CLI")
    if env:
        candidates.append((Path(env).expanduser(), "env:ARCHIFY_CLI"))
    config = workspace_root / CONFIG_REL
    if config.is_file():
        data = json.loads(config.read_text(encoding="utf-8"))
        cli = data.get("cli")
        if cli:
            path = Path(cli).expanduser()
            if not path.is_absolute():
                path = (workspace_root / path).resolve()
            candidates.append((path, f"config:{CONFIG_REL.as_posix()}"))
    home = Path.home()
    for base, label in (
        (home / ".dsh" / "skills" / "archify", "convention:~/.dsh/skills/archify"),
        (home / ".claude" / "skills" / "archify", "convention:~/.claude/skills/archify"),
        (home / ".codex" / "skills" / "archify", "convention:~/.codex/skills/archify"),
    ):
        candidates.append((base / "bin" / "archify.mjs", label))
    return candidates


def discover_archify(workspace_root: str | Path) -> ArchifyTool:
    root = Path(workspace_root)
    for path, source in _candidate_paths(root):
        if path.is_file():
            package = path.parent.parent / "package.json"
            version = None
            release = path.parent.parent / "skill-release.json"
            if release.is_file():
                version = json.loads(release.read_text(encoding="utf-8")).get("version")
            return ArchifyTool(
                cli=path,
                discovery=source,
                version=version,
                package_json=package if package.is_file() else None,
            )
    raise ArchifyToolError(
        "找不到 Archify CLI：设置 ARCHIFY_CLI，或写 structure_reader/configs/tools/archify.json"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_ENV_BROWSER_ERROR_CODES = frozenset({"viewer/browser-check-runtime"})


def classify_browser_failure(receipt: Mapping[str, Any]) -> tuple[str, str]:
    """区分浏览器环境失败与真实产物失败。

    只有失败阶段为 browser-check 且全部 error 级诊断都是 Chrome 启动/权限/运行
    环境错误（``viewer/browser-check-runtime``）时才是环境失败；几何、来源、
    证据冲突等任何其它诊断都判为真实失败，不得降级。
    """
    if receipt.get("failedStage") != "browser-check":
        return "real", f"失败阶段不是 browser-check: {receipt.get('failedStage')!r}"
    diagnostics = [
        item for item in receipt.get("diagnostics", []) if item.get("severity") == "error"
    ]
    codes = {item.get("code") for item in diagnostics}
    if not diagnostics or not codes <= _ENV_BROWSER_ERROR_CODES:
        return "real", f"browser-check 失败包含非环境诊断: {sorted(code for code in codes if code)}"
    return "environment", "Chrome 启动/权限/运行环境失败（viewer/browser-check-runtime）"


def package_closure(root: Path) -> dict[str, Any]:
    """实际运行依赖文件闭包：逐文件 SHA256 + 确定性摘要（变更即身份变化）。"""
    excluded = {".git", "node_modules", "__pycache__", ".DS_Store"}
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in excluded for part in path.parts):
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    if not records:
        raise ArchifyToolError(f"Archify 包目录没有文件: {root}")
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "root": str(root),
        "file_count": len(records),
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "files": records,
    }


def tool_identity(tool: ArchifyTool) -> dict[str, Any]:
    """可移植工具身份：只含内容哈希/版本/发现方式，不含安装绝对路径。

    ``cli`` 绝对路径与 ``package.root`` 会随 HOME/技能安装位置变化，进入
    analysis_id 会让同内容换机器得到不同候选；这里只保留 discovery 标签、
    cli/package.json/包闭包的内容哈希与 Node 版本。
    """
    node_version = subprocess.run(
        ["node", "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    closure = package_closure(tool.cli.parent.parent)
    identity: dict[str, Any] = {
        "discovery": tool.discovery,
        "cli_sha256": _sha256_file(tool.cli),
        "version": tool.version,
        "node": node_version,
        "package": {
            "file_count": closure["file_count"],
            "digest": closure["digest"],
        },
    }
    if tool.package_json is not None:
        identity["package_json_sha256"] = _sha256_file(tool.package_json)
    return identity


# ---------------------------------------------------------------------------
# 视图导出
# ---------------------------------------------------------------------------


@dataclass
class ViewNode:
    semantic_id: str
    label: str
    sublabel: str
    tag: str
    component_type: str
    sources: tuple[SourceRef, ...]
    role: str  # boundary_in | main | aux | state | branch | boundary_out
    anchor: str | None = None  # 侧节点锚定的主链节点 id
    shape: str = ""


@dataclass
class ViewConnection:
    src: str
    dst: str
    label: str
    variant: str
    from_side: str | None = None
    to_side: str | None = None


@dataclass
class ViewModel:
    view_key: str
    title: str
    nodes: list[ViewNode] = field(default_factory=list)
    connections: list[ViewConnection] = field(default_factory=list)
    main_order: list[str] = field(default_factory=list)
    boundary_input: str | None = None
    boundary_output: str | None = None
    edge_records: list[dict[str, Any]] = field(default_factory=list)


_STATE_LABELS = {
    "state_in": "调用前状态",
    "state_out": "调用后状态",
}


def _main_order_for_view(bundle: SemanticBundle, member_ids: set[str], view_key: str) -> list[str]:
    """主链顺序：按数据边拓扑序，Q 路径在 KV 路径前，侧枝不在主链。"""
    if view_key == "kv":
        preferred = [
            "attn.kv_proj",
            "attn.kv_norm",
            "attn.kv_view",
            "attn.kv_transpose",
            "attn.kv_rope",
        ]
    else:
        preferred = [
            "attn.q_a_proj",
            "attn.q_a_norm",
            "attn.q_b_proj",
            "attn.q_view",
            "attn.q_transpose",
            "attn.q_b_norm",
            "attn.q_rope",
            "attn.kv_proj",
            "attn.kv_norm",
            "attn.kv_view",
            "attn.kv_transpose",
            "attn.kv_rope",
            "attn.cache_update",
            "attn.kv_concat",
            "attn.repeat_kv",
            "attn.kv_transposed",
            "attn.qk_matmul",
            "attn.scale",
            "attn.mask_extend",
            "attn.mask_add",
            "attn.sink_softmax",
            "attn.av_matmul",
            "attn.out_transpose",
            "attn.rope_in_transpose",
            "attn.inv_rope",
            "attn.rope_out_transpose",
            "attn.group_reshape",
            "attn.o_a_proj",
            "attn.o_b_proj",
        ]
    order: list[str] = []

    def add(node_id: str) -> None:
        if node_id in member_ids and node_id not in order:
            order.append(node_id)

    for node_id in preferred:
        add(node_id)
    # 未列入偏好顺序的成员（如被化简规则替换出的 rw.* 节点）按拓扑序补入，
    # 插入位置取其原始邻居在偏好顺序中的位置。
    remaining = [
        node_id
        for node_id in sorted(member_ids)
        if node_id not in order and bundle.node_meta[node_id].kind not in _SIDE_KINDS
    ]
    for node_id in remaining:
        producers = [
            edge.src_node_id
            for edge in bundle.graph.edges
            if edge.dst_node_id == node_id and edge.kind == "data"
        ]
        consumers = [
            edge.dst_node_id
            for edge in bundle.graph.edges
            if edge.src_node_id == node_id and edge.kind == "data"
        ]
        producer_positions = [order.index(item) for item in producers if item in order]
        consumer_positions = [order.index(item) for item in consumers if item in order]
        if producer_positions:
            order.insert(max(producer_positions) + 1, node_id)
        elif consumer_positions:
            order.insert(min(consumer_positions), node_id)
        else:
            order.append(node_id)
    return order


_SIDE_KINDS = frozenset({"aux", "state_in", "state_out", "composite"})


def _side_nodes(bundle: SemanticBundle, member_ids: set[str], main_order: list[str]) -> list[str]:
    return [node_id for node_id in sorted(member_ids) if node_id not in main_order]


def _short_sublabel(bundle: SemanticBundle, node_id: str) -> str:
    meta = bundle.node_meta[node_id]
    node = bundle.nodes[node_id]
    kind = meta.kind
    def last_axis(tensor_id: str) -> str:
        text = bundle.shape_text(tensor_id)
        return text[1:-1].rsplit(", ", 1)[-1] if text.startswith("[") else text
    input_id = node.input_ports[0].tensor_id if node.input_ports else None
    output_id = node.output_ports[0].tensor_id if node.output_ports else None
    if kind in ("linear", "grouped_linear") and input_id and output_id:
        return f"{last_axis(input_id)} → {last_axis(output_id)}"
    if kind == "rmsnorm" and output_id:
        return f"沿 {last_axis(output_id)} 归一"
    if kind in ("view", "reshape", "transpose") and input_id and output_id:
        return f"{len(bundle.axis_symbols(input_id))}D → {len(bundle.axis_symbols(output_id))}D（layout）"
    if kind == "rope":
        return "尾部 qk_rope_head_dim"
    if kind == "matmul":
        return "Q @ Kᵀ" if meta.attributes.get("rhs_transposed_axes") else "P @ V"
    if kind == "scale":
        return "× head_dim^-0.5"
    if kind == "add":
        return "0 / -inf mask"
    if kind == "softmax":
        return "sink + 稳定 softmax"
    if kind == "concat":
        return "dim=2" if node_id != "attn.mask_extend" else "dim=-1（mask）"
    if kind == "broadcast":
        return "num_key_value_heads → num_attention_heads"
    if kind == "cache_update":
        return "K=V 同存储"
    if kind in ("state_in", "state_out"):
        return "W-1 尾部 / 压缩状态" if node_id.endswith("cache_in") else "下一调用读取"
    if kind == "composite":
        return "具名计算（来源见节点）"
    if kind == "aux":
        return _aux_sublabel(bundle, node_id)
    if kind in ("external", "external_out"):
        return _shape_or_note(bundle, node_id)
    return _shape_or_note(bundle, node_id)


def _shape_or_note(bundle: SemanticBundle, node_id: str) -> str:
    node = bundle.nodes[node_id]
    if node.output_ports:
        return bundle.shape_text(node.output_ports[0].tensor_id)
    if node.input_ports:
        return bundle.shape_text(node.input_ports[0].tensor_id)
    return ""


def _kind_label(meta) -> str:
    label = _KIND_LABELS.get(meta.kind, meta.kind)
    legacy_ops = [ref[3:] for ref in meta.legacy_refs if ref.startswith("op:")]
    if legacy_ops and meta.node_id.startswith("rw."):
        return f"{label} · 原 {legacy_ops[0]}"
    return label


def _component_sources(bundle: SemanticBundle, node_id: str) -> tuple[SourceRef, ...]:
    sources = [
        source
        for source in bundle.node_meta[node_id].sources
        if source.repo_key in _TRANSFORMERS_REPO_KEYS
    ]
    # 去重并保留前 3 条（Archify 每个组件最多 3 条来源）
    unique: list[SourceRef] = []
    seen: set[tuple] = set()
    for source in sources:
        key = (source.path, source.line, source.end_line)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return tuple(unique[:_MAX_SOURCES])


def build_view_model(bundle: SemanticBundle, view_key: str) -> ViewModel:
    spec = bundle.metadata["views"].get(view_key)
    if spec is None:
        raise PreviewError(f"bundle 没有视图 {view_key!r}")
    old_ops = set(spec["old_ops"]) | set(spec.get("extra_old_ops", ()))
    member_ids = {
        node_id
        for node_id, meta in bundle.node_meta.items()
        if any(ref == f"op:{op_id}" for ref in meta.legacy_refs for op_id in old_ops)
        and not meta.node_id.startswith("layer.")
    }
    if not member_ids:
        raise PreviewError(f"视图 {view_key!r} 没有匹配节点（旧算子映射为空）")
    main_order = _main_order_for_view(bundle, member_ids, view_key)
    side_ids = _side_nodes(bundle, member_ids, main_order)
    view = ViewModel(view_key=view_key, title=spec["title"])
    view.main_order = list(main_order)

    start_tensor = f"tensor:{spec['start_tensor']}"
    end_tensor = f"tensor:{spec['end_tensor']}"
    main_index = {node_id: index for index, node_id in enumerate(main_order)}
    main_set = set(main_order)
    members = set(member_ids)
    first = main_order[0]

    def shape_of(tensor_id: str) -> str:
        return bundle.shape_text(tensor_id)

    def tensor_label(tensor_id: str) -> str:
        meta = bundle.tensor_meta.get(tensor_id)
        return meta.label if meta is not None else tensor_id

    # --- 节点 -------------------------------------------------------------
    boundary_in_id = f"view.{view_key}.input"
    view.nodes.append(
        ViewNode(
            semantic_id=boundary_in_id,
            label="hidden_states" if view_key == "kv" else "hidden_states（Attention 输入）",
            sublabel=shape_of(start_tensor),
            tag="输入张量",
            component_type="external",
            sources=_component_sources(bundle, first),
            role="boundary_in",
            shape=shape_of(start_tensor),
        )
    )
    view.boundary_input = boundary_in_id
    for node_id in main_order:
        meta = bundle.node_meta[node_id]
        view.nodes.append(
            ViewNode(
                semantic_id=node_id,
                label=meta.label,
                sublabel=_short_sublabel(bundle, node_id),
                tag=_kind_label(meta),
                component_type="backend",
                sources=_component_sources(bundle, node_id),
                role="main",
                shape=_shape_or_note(bundle, node_id),
            )
        )
    side_ids = _side_nodes(bundle, members, main_order)
    for node_id in side_ids:
        meta = bundle.node_meta[node_id]
        if meta.kind == "aux":
            consumers = list(
                dict.fromkeys(
                    edge.dst_node_id
                    for edge in bundle.graph.edges
                    if edge.kind == "data"
                    and edge.src_node_id == node_id
                    and edge.dst_node_id in main_set
                )
            )
            if not consumers:
                raise PreviewError(f"辅助节点 {node_id} 没有主链消费者，无法定位")
            for dst in consumers:
                view.nodes.append(
                    ViewNode(
                        semantic_id=f"{node_id}@{dst}",
                        label=_aux_label(bundle, node_id),
                        sublabel=_aux_sublabel(bundle, node_id),
                        tag="辅助输入（同一张量）",
                        component_type="external",
                        sources=_component_sources(bundle, node_id),
                        role="aux",
                        anchor=dst,
                        shape=_shape_or_note(bundle, node_id),
                    )
                )
        elif meta.kind in ("state_in", "state_out"):
            view.nodes.append(
                ViewNode(
                    semantic_id=node_id,
                    label=_STATE_LABELS[meta.kind],
                    sublabel=_shape_or_note(bundle, node_id),
                    tag="跨调用状态",
                    component_type="external",
                    sources=_component_sources(bundle, node_id),
                    role="state",
                    anchor="attn.cache_update",
                    shape=_shape_or_note(bundle, node_id),
                )
            )
        elif meta.kind == "composite":
            view.nodes.append(
                ViewNode(
                    semantic_id=node_id,
                    label=meta.label,
                    sublabel=_shape_or_note(bundle, node_id),
                    tag="具名分支",
                    component_type="backend",
                    sources=_component_sources(bundle, node_id),
                    role="branch",
                    anchor=({"attn.block_bias": "attn.mask_extend",
                             "attn.position_ids": "attn.mask_add"}
                            .get(node_id, "attn.kv_concat")),
                    shape=_shape_or_note(bundle, node_id),
                )
            )
        else:
            raise PreviewError(
                f"视图 {view_key!r} 的侧节点 {node_id!r}（kind={meta.kind}）没有布局规则"
            )
    boundary_out_id = f"view.{view_key}.output"
    last = main_order[-1]
    view.nodes.append(
        ViewNode(
            semantic_id=boundary_out_id,
            label="kv（缓存更新前）" if view_key == "kv" else "attn_output",
            sublabel=shape_of(end_tensor),
            tag="输出张量",
            component_type="external",
            sources=_component_sources(bundle, last),
            role="boundary_out",
            shape=shape_of(end_tensor),
        )
    )
    view.boundary_output = boundary_out_id
    view_node_ids = {node.semantic_id for node in view.nodes}

    # --- 连接：真实数据/状态依赖一条都不省 ---------------------------------
    def add_connection(
        src: str,
        dst: str,
        label: str,
        variant: str,
        from_side: str | None = None,
        to_side: str | None = None,
    ) -> None:
        if src not in view_node_ids or dst not in view_node_ids:
            raise PreviewError(f"连接端点不在视图中: {src} -> {dst}")
        payload = ViewConnection(
            src=src,
            dst=dst,
            label=label,
            variant=variant,
            from_side=from_side,
            to_side=to_side,
        )
        if any(
            item.src == src and item.dst == dst and item.label == label
            for item in view.connections
        ):
            return
        view.connections.append(payload)

    def record(edge, src_view: str, dst_view: str) -> None:
        view.edge_records.append(
            {
                "semantic_edge": edge.stable_id,
                "src_node": edge.src_node_id,
                "dst_node": edge.dst_node_id,
                "view_src": src_view,
                "view_dst": dst_view,
                "tensor_id": edge.tensor_id,
                "tensor_label": tensor_label(edge.tensor_id),
                "shape": shape_of(edge.tensor_id),
                "kind": edge.kind,
                "guard_ref": edge.guard_ref,
                "state_transition_ref": edge.state_transition_ref,
            }
        )

    reuse_nodes: set[str] = set()
    for edge in bundle.graph.edges:
        if edge.kind != "data":
            continue
        src, dst = edge.src_node_id, edge.dst_node_id
        if src not in members or dst not in members:
            continue
        src_view = (
            f"{src}@{dst}" if bundle.node_meta[src].kind == "aux" else src
        )
        if src_view not in view_node_ids or dst not in view_node_ids:
            continue
        if src in main_index and dst in main_index:
            distance = main_index[dst] - main_index[src]
            if distance == 1:
                add_connection(src_view, dst, shape_of(edge.tensor_id), "emphasis")
                record(edge, src_view, dst)
            elif distance >= 3:
                reuse_id = f"reuse.{src}.{dst}"
                if reuse_id not in reuse_nodes:
                    reuse_nodes.add(reuse_id)
                    view.nodes.append(
                        ViewNode(
                            semantic_id=reuse_id,
                            label="↳ "
                            + edge.tensor_id.split(":", 1)[-1].split(".")[-1],
                            sublabel="同一张量（跨段复用）",
                            tag="跨段复用（同一张量）",
                            component_type="external",
                            sources=_component_sources(bundle, src),
                            role="reuse",
                            anchor=dst,
                            shape=shape_of(edge.tensor_id),
                        )
                    )
                    view_node_ids.add(reuse_id)
                add_connection(
                    reuse_id,
                    dst,
                    shape_of(edge.tensor_id),
                    "default",
                    from_side="right",
                    to_side="left",
                )
                record(edge, reuse_id, dst)
            else:
                # 间距 2 的真实依赖（如 scale → mask_add）必须画出，不能用排序代替数据流
                add_connection(src_view, dst, shape_of(edge.tensor_id), "emphasis")
                record(edge, src_view, dst)
        else:
            add_connection(
                src_view,
                dst,
                shape_of(edge.tensor_id),
                "default",
            )
            record(edge, src_view, dst)

    # 边界输入：所有消费视图外起始张量的节点都要有入边
    for node_id in sorted(members):
        node = bundle.nodes[node_id]
        for port in node.input_ports:
            tensor_id = port.tensor_id
            has_internal_producer = any(
                edge.kind == "data"
                and edge.src_node_id in members
                and edge.dst_node_id == node_id
                and edge.tensor_id == tensor_id
                for edge in bundle.graph.edges
            )
            if has_internal_producer:
                continue
            has_state_feed = any(
                edge.kind == "state"
                and edge.src_node_id in members
                and edge.dst_node_id == node_id
                and edge.dst_port == port.name
                for edge in bundle.graph.edges
            )
            if has_state_feed:
                continue
            if tensor_id != start_tensor:
                raise PreviewError(
                    f"视图 {view_key!r} 的节点 {node_id!r} 输入 {tensor_id!r} "
                    "没有视图内生产者，也不是视图起点张量（会丢失真实依赖）"
                )
            add_connection(
                boundary_in_id,
                node_id,
                shape_of(tensor_id),
                "emphasis" if node_id in main_index else "default",
            )
            view.edge_records.append(
                {
                    "semantic_edge": "(boundary)",
                    "src_node": "(outside)",
                    "dst_node": node_id,
                    "view_src": boundary_in_id,
                    "view_dst": node_id,
                    "tensor_id": tensor_id,
                    "tensor_label": tensor_label(tensor_id),
                    "shape": shape_of(tensor_id),
                    "kind": "data",
                    "guard_ref": None,
                    "state_transition_ref": None,
                }
            )

    # 边界输出：主链节点产出、被视图外消费的张量（辅助输入是复制的入口，不作为输出）
    for edge in bundle.graph.edges:
        if edge.kind != "data":
            continue
        if edge.src_node_id not in main_index:
            continue
        if edge.dst_node_id in members:
            continue
        add_connection(
            edge.src_node_id,
            boundary_out_id,
            shape_of(edge.tensor_id),
            "emphasis",
        )
        record(edge, edge.src_node_id, boundary_out_id)

    # 状态边：显式保留（虚线 + 状态转移引用）
    for edge in bundle.graph.edges:
        if edge.kind != "state":
            continue
        if edge.src_node_id not in members or edge.dst_node_id not in members:
            # 视图范围外的状态读写（如 KV 新投影预览不含 cache.update）不在本视图展开
            continue
        add_connection(
            edge.src_node_id,
            edge.dst_node_id,
            "状态读入" if edge.src_node_id == "attn.cache_in" else "状态写出",
            "dashed",
        )
        record(edge, edge.src_node_id, edge.dst_node_id)

    return view


def _aux_label(bundle: SemanticBundle, node_id: str) -> str:
    if node_id == "attn.rope_freqs":
        return "cos / sin"
    if node_id == "attn.mask":
        return "mask"
    if node_id == "attn.sinks":
        return "sinks"
    return bundle.node_meta[node_id].label


    return view


def _aux_sublabel(bundle: SemanticBundle, node_id: str) -> str:
    node = bundle.node_meta[node_id]
    if node_id == "attn.rope_freqs":
        return bundle.shape_text("tensor:rope_cos") + "（按头广播）"
    if node_id == "attn.sinks":
        return "[num_attention_heads]（每头一个标量）"
    if node_id == "attn.mask":
        return "4D float mask（0 / -inf）"
    return _shape_or_note(bundle, node_id)


def _side_edge_label(bundle: SemanticBundle, tensor_id: str, src: str, dst: str) -> str:
    if bundle.node_meta[src].kind == "aux":
        if src == "attn.rope_freqs":
            return "cos / sin"
        if src == "attn.mask":
            return "mask"
        if src == "attn.sinks":
            return "sinks"
    if src == "attn.indexer":
        return "topk_selected"
    if src == "attn.compressor" and dst == "attn.kv_concat":
        return "compressed_kv"
    if src == "attn.compressor":
        return "block_bias"
    return bundle.shape_text(tensor_id)


def dashed_route_points(html_text: str) -> dict[str, list[tuple[float, float]]]:
    """从最终 HTML 的 SVG 里读取虚线（state）关系的实际路由折点。"""
    routes: dict[str, list[tuple[float, float]]] = {}
    for match in re.finditer(r"<path[^>]*class=\"a-dashed\"[^>]*>", html_text):
        tag = match.group(0)
        edge = re.search(r'data-edge-id="([^"]+)"', tag)
        points = re.search(r'data-composition-points="([^"]+)"', tag)
        if edge is None or points is None:
            continue
        parsed = []
        for pair in points.group(1).split(";"):
            parts = pair.split(",")
            if len(parts) == 2:
                parsed.append((float(parts[0]), float(parts[1])))
        routes[edge.group(1)] = parsed
    return routes


def state_route_violations(html_text: str) -> list[dict[str, Any]]:
    """最终 SVG 中任何 state（虚线）路由出现向上段即违规；返回逐条证据。"""
    violations: list[dict[str, Any]] = []
    for edge_id, points in dashed_route_points(html_text).items():
        ys = [point[1] for point in points]
        upward = [
            (ys[index], ys[index + 1])
            for index in range(len(ys) - 1)
            if ys[index + 1] < ys[index] - 0.01
        ]
        if upward:
            violations.append(
                {
                    "edge_id": edge_id,
                    "points": points,
                    "upward_segments": upward,
                }
            )
    return violations


# ---------------------------------------------------------------------------
# 布局与候选
# ---------------------------------------------------------------------------

_LAYOUT = {
    "main_x": 380,
    "main_w": 300,
    "main_h": 76,
    "y0": 200,
    "step": 132,
    "left_x": 20,
    "left_w": 240,
    "left_h": 64,
    "right_x": 980,
    "right_w": 260,
    "right_h": 76,
    "canvas_w": 1240,
}


def _component_id(node_id: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z_-]+", "-", node_id).strip("-").lower()
    return f"n-{slug}"[:64]


def _layout(view: ViewModel) -> dict[str, tuple[float, float, float, float]]:
    spec = _LAYOUT
    positions: dict[str, tuple[float, float, float, float]] = {}
    main_index = {node_id: index for index, node_id in enumerate(view.main_order)}
    for node_id in view.main_order:
        y = spec["y0"] + main_index[node_id] * spec["step"]
        positions[node_id] = (spec["main_x"], y, spec["main_w"], spec["main_h"])
    left: list[tuple[float, str, float]] = []
    state_ids: list[tuple[str, float]] = []
    for node in view.nodes:
        if node.semantic_id in positions:
            continue
        anchor = node.anchor or view.main_order[0]
        anchor_y = positions.get(anchor, (0, spec["y0"], 0, 0))[1]
        if node.role == "boundary_in":
            positions[node.semantic_id] = (
                spec["main_x"],
                spec["y0"] - 160,
                spec["main_w"],
                spec["main_h"],
            )
        elif node.role == "boundary_out":
            last_y = positions[view.main_order[-1]][1] + spec["step"]
            positions[node.semantic_id] = (spec["main_x"], last_y, spec["main_w"], spec["main_h"])
        elif node.role == "state":
            state_ids.append((node.semantic_id, spec["left_h"]))
        elif node.role in ("aux", "reuse"):
            left.append((anchor_y, node.semantic_id, spec["left_h"]))
        elif node.role == "branch":
            offset = (
                -spec["step"] * 2.9
                if node.semantic_id == "attn.compressor"
                else -spec["step"] * 4.0
            )
            left.append((anchor_y + offset, node.semantic_id, spec["right_h"]))
    left.sort()
    cursor = -10**9
    for y, node_id, height in left:
        y = max(y, cursor + 18)
        positions[node_id] = (spec["left_x"], y, spec["left_w"], height)
        cursor = y + height

    # 状态节点按“全部真实消费者”的拓扑放置：前态在所有消费者上方，后态在所有生产者下方。
    dashed = [connection for connection in view.connections if connection.variant == "dashed"]

    def _left_nodes() -> list[tuple[str, tuple[float, float, float, float]]]:
        return [
            (node_id, rect)
            for node_id, rect in positions.items()
            if rect[0] == spec["left_x"]
        ]

    for node_id, height in state_ids:
        if node_id.endswith("cache_in"):
            targets = [
                connection.dst
                for connection in dashed
                if connection.src == node_id and connection.dst in positions
            ]
            if not targets:
                raise PreviewError(f"状态节点 {node_id} 没有已定位的目标消费者")
            limit = min(positions[target][1] for target in targets)
            desired = limit - height - 24
            # 冲突时把挡道的旁枝节点上移，保证前态整体位于全部消费者之上
            for _ in range(8):
                conflicts = [
                    (other, rect)
                    for other, rect in _left_nodes()
                    if rect[1] < desired + height and rect[1] + rect[3] > desired
                ]
                if not conflicts:
                    break
                for other, rect in sorted(conflicts, key=lambda item: item[1][1], reverse=True):
                    positions[other] = (
                        rect[0],
                        desired - 18 - rect[3],
                        rect[2],
                        rect[3],
                    )
            positions[node_id] = (spec["left_x"], desired, spec["left_w"], height)
        else:
            sources = [
                connection.src
                for connection in dashed
                if connection.dst == node_id and connection.src in positions
            ]
            if not sources:
                raise PreviewError(f"状态节点 {node_id} 没有已定位的生产者")
            limit = max(
                positions[source][1] + positions[source][3] for source in sources
            )
            desired = limit + 24
            for _ in range(8):
                conflicts = [
                    (other, rect)
                    for other, rect in _left_nodes()
                    if rect[1] < desired + height and rect[1] + rect[3] > desired
                ]
                if not conflicts:
                    break
                for other, rect in sorted(conflicts, key=lambda item: item[1][1]):
                    positions[other] = (
                        rect[0],
                        desired + height + 18,
                        rect[2],
                        rect[3],
                    )
            positions[node_id] = (spec["left_x"], desired, spec["left_w"], height)

    # 硬约束：所有 state 边在最终几何中必须向下（源顶部在上，目标顶部在下）。
    for connection in dashed:
        if connection.src not in positions or connection.dst not in positions:
            continue
        source = positions[connection.src]
        target = positions[connection.dst]
        if source[1] >= target[1]:
            raise PreviewError(
                f"状态边 {connection.src} → {connection.dst} 在布局中向上回指"
                f"（源 y={source[1]}, 目标 y={target[1]}）"
            )
    return positions


def _sides(view: ViewModel, connection: ViewConnection, positions) -> tuple[str, str]:
    if connection.from_side and connection.to_side:
        return connection.from_side, connection.to_side
    if connection.variant == "dashed":
        src = positions.get(connection.src)
        dst = positions.get(connection.dst)
        if (
            src is not None
            and dst is not None
            and src[0] + src[2] <= _LAYOUT["main_x"]
            and dst[0] + dst[2] <= _LAYOUT["main_x"]
        ):
            return "right", "right"
    src = positions.get(connection.src)
    dst = positions.get(connection.dst)
    if src is None or dst is None:
        return "bottom", "top"
    src_cx = src[0] + src[2] / 2
    dst_cx = dst[0] + dst[2] / 2
    src_cy = src[1] + src[3] / 2
    dst_cy = dst[1] + dst[3] / 2
    if abs(src_cx - dst_cx) < 40 and dst[1] >= src[1] + src[3]:
        return "bottom", "top"
    if dst[1] + dst[3] <= src[1]:
        return "top", "bottom"
    return ("right", "left") if src_cx < dst_cx else ("left", "right")


def _routes(
    view: ViewModel, connection: ViewConnection, positions
) -> dict[str, Any]:
    src = positions.get(connection.src)
    dst = positions.get(connection.dst)
    extra: dict[str, Any] = {}
    if src is None or dst is None:
        return extra
    main_x = _LAYOUT["main_x"]
    if (
        connection.variant == "dashed"
        and src[0] + src[2] <= main_x
        and dst[0] + dst[2] <= main_x
    ):
        lane_x = main_x - 30
        extra["via"] = [
            [lane_x, src[1] + src[3] / 2],
            [lane_x, dst[1] + dst[3] / 2],
        ]
    return extra


def build_candidate(
    bundle: SemanticBundle,
    view: ViewModel,
    *,
    output_relpath: str,
    form: str,
    normalization_report: Mapping[str, Any] | None,
    tool: Mapping[str, Any] | None,
    repository: Mapping[str, Any] | None,
) -> dict[str, Any]:
    positions = _layout(view)
    viewbox_h = max(y + h for _x, y, _w, h in positions.values()) + 140
    components = []
    for node in view.nodes:
        x, y, width, height = positions[node.semantic_id]
        sources = [
            {
                "path": source.path,
                **({"line": source.line} if source.line else {}),
                **({"end_line": source.end_line} if source.end_line else {}),
            }
            for source in node.sources
        ]
        component: dict[str, Any] = {
            "id": _component_id(node.semantic_id),
            "type": node.component_type,
            "icon": "none",
            "label": node.label,
            "sublabel": node.sublabel,
            "tag": node.tag,
            "pos": [x, y],
            "size": [width, height],
        }
        if sources:
            component["sources"] = sources
        components.append(component)

    id_by_node = {node.semantic_id: _component_id(node.semantic_id) for node in view.nodes}
    connections = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for index, connection in enumerate(view.connections):
        src, dst = connection.src, connection.dst
        if src not in id_by_node or dst not in id_by_node:
            continue
        pair = (src, dst, connection.label)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        from_side, to_side = _sides(view, connection, positions)
        vertical = from_side == "bottom" and to_side == "top"
        # D03：主图边必须直接显示真实张量身份/全称维度，不能只放在卡片或
        # 边缘 JSON。Archify 的自动标签布局负责避让；不再按方向隐藏标签。
        display_label = connection.label
        payload: dict[str, Any] = {
            "id": f"f-{index}-{re.sub(r'[^0-9a-zA-Z_-]+', '-', src)}-{re.sub(r'[^0-9a-zA-Z_-]+', '-', dst)}"[:64],
            "from": id_by_node[src],
            "to": id_by_node[dst],
            "variant": connection.variant,
            "fromSide": from_side,
            "toSide": to_side,
        }
        if display_label:
            payload["label"] = display_label
        payload.update(_routes(view, connection, positions))
        src_pos = positions.get(src)
        dst_pos = positions.get(dst)
        if (
            from_side == "bottom"
            and to_side == "top"
            and src_pos is not None
            and dst_pos is not None
        ):
            gap = dst_pos[1] - (src_pos[1] + src_pos[3])
            payload["labelDy"] = round(gap / 2 + 10, 1)
        elif src_pos is not None and dst_pos is not None and not vertical:
            payload["labelDy"] = 22
        connections.append(payload)

    # D03：语义图也用同一套共享标签去重叠器；位置以组件 id 为键。
    from structure_reader.render.archify_layers import (
        _edge_label_width,
        separate_edge_labels,
    )
    component_positions = {
        _component_id(node.semantic_id): (
            positions[node.semantic_id][0], positions[node.semantic_id][1],
            positions[node.semantic_id][2], positions[node.semantic_id][3],
        )
        for node in view.nodes
    }
    obstacles = []
    for node in view.nodes:
        rect = positions[node.semantic_id]
        text_width = max(
            _edge_label_width(node.label) * 11 / 8,
            _edge_label_width(node.sublabel or "") * 9 / 8,
        )
        expand = max(0.0, (text_width - rect[2]) / 2.0)
        obstacles.append((rect[0] - expand, rect[1], rect[2] + expand * 2, rect[3]))
    separate_edge_labels(connections, component_positions, obstacles)

    cards = _cards(bundle, view, form, normalization_report, tool, repository)
    title = (
        f"DeepSeek V4 · {view.title}"
        + ("（原始语义图）" if form == "raw" else "（化简后）")
    )
    candidate: dict[str, Any] = {
        "schema_version": 1,
        "diagram_type": "architecture",
        "meta": {
            "title": title,
            "locale": "zh-CN",
            "quality_profile": "standard",
            "output": output_relpath,
            "repository": dict(repository),
            "legend": {
                "entries": {
                    "backend": {"label": "计算与布局操作"},
                    "external": {"label": "输入/输出与跨调用状态"},
                }
            },
        },
        "components": components,
        "connections": connections,
        "cards": cards,
    }
    return candidate


def _short_relation_label(view: ViewModel, connection: ViewConnection) -> str:
    record = next(
        (
            item
            for item in view.edge_records
            if item["view_src"] == connection.src and item["view_dst"] == connection.dst
        ),
        None,
    )
    if record is None:
        return ""
    if record["kind"] == "state":
        return "状态"
    if connection.src.startswith("reuse."):
        return "同一张量"
    tensor_id = record["tensor_id"]
    tail = tensor_id.split(":", 1)[-1].split(".")[-1]
    if tail in ("rope_cos", "rope_sin"):
        return "cos / sin"
    return tail


def _relation_card(bundle: SemanticBundle, view: ViewModel) -> dict[str, Any] | None:
    labels = {node.semantic_id: node.label for node in view.nodes}
    items: list[str] = []
    for record in view.edge_records:
        if record["kind"] == "data" and record["src_node"] in view.main_order and record["dst_node"] in view.main_order:
            continue
        src_label = labels.get(record["view_src"], record["view_src"])
        dst_label = labels.get(record["view_dst"], record["view_dst"])
        detail = f"{src_label} → {dst_label}：{record['tensor_label']} {record['shape']}"
        if record["kind"] == "state":
            detail += "（state 边）"
        if record["guard_ref"]:
            detail += f"（条件 {record['guard_ref']}）"
        items.append(detail)
    if not items:
        return None
    if len(items) > 18:
        items = items[:18] + [f"…其余 {len(items) - 18} 条见 previews edges JSON"]
    return {"dot": "rose", "title": "分支/状态关系（张量身份与全称维度）", "items": items}


def _cards(
    bundle: SemanticBundle,
    view: ViewModel,
    form: str,
    normalization_report: Mapping[str, Any] | None,
    tool: Mapping[str, Any] | None,
    repository: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    metadata = bundle.metadata
    cards: list[dict[str, Any]] = []
    cards.append(
        {
            "dot": "cyan",
            "title": "符号与参数（全称）",
            "items": [
                "张量轴使用 modeling/config 全称：batch_size、seqlen、kvcachelen、hidden_size、"
                "head_dim、num_attention_heads、num_key_value_heads、q_lora_rank、qk_rope_head_dim、"
                "o_groups、o_lora_rank。",
                f"当前配置：hidden_size={4096 if metadata.get('model_key') else '4096'}"
                "（以 config.json 为准）",
                "kv_length = min(kvcachelen, sliding_window - 1) + seqlen（本次 Attention 可读滑窗长度）；"
                "kv_cache_len = min(kvcachelen, sliding_window - 1)（调用前持久态）。",
                "seqlen 是本次新输入 token 数；kvcachelen 是调用前逻辑历史长度，二者不混用。",
                "num_key_value_heads=1 是当前配置固定值，图上显示 1 并在参数中保留名字；"
                "不是可任意改变的头数。",
            ],
        }
    )
    relation_card = _relation_card(bundle, view)
    if relation_card is not None:
        cards.append(relation_card)
    if form == "raw":
        items = [
            "本图是**化简前**的原始语义图：View/Transpose/连续布局操作按源码粒度保留。",
            "同一语义结果经保守白名单化简后的节点见对应的 normalized 预览与 T12 化简审计。",
        ]
        if normalization_report:
            for report in normalization_report.get("rules", []):
                status = "应用" if report["status"] == "applied" else "拒绝"
                items.append(f"{report['rule_id']}：{status}——{report['reason']}")
        cards.append({"dot": "amber", "title": "原始 vs 化简后", "items": items})
    else:
        items = []
        if normalization_report:
            for report in normalization_report.get("rules", []):
                status = "应用" if report["status"] == "applied" else "拒绝"
                detail = report["reason"]
                items.append(f"{report['rule_id']}：{status}——{detail}")
                if report["status"] == "applied":
                    for application in report.get("applications", [])[:3]:
                        removed = "、".join(application["removed_nodes"]) or "—"
                        added = "、".join(application["added_nodes"]) or "—"
                        items.append(f"  · 去掉 {removed} → 得到 {added}；{application['condition']}")
        items.append("化简保持值/形状/轴/类型/状态语义与源码对应；未证明的候选保留并记录拒绝原因。")
        cards.append({"dot": "amber", "title": "化简对照（原始 → 化简后）", "items": items})
    cards.append(
        {
            "dot": "emerald",
            "title": "来源、边界与限制",
            "items": [
                f"源码：{repository.get('url')} @ {repository.get('revision')}"
                if repository
                else "源码：hflib/transformers 检出",
                "层组：" + str(metadata.get("layer_group")) + "；层：" + str(metadata.get("layer_key"))
                + "；RoPE 类型：" + str(metadata.get("rope_type")),
                "K=V 共享同一 KV 张量（单投影）；缓存读/写以 state 边表达，未隐藏状态副作用。",
                "逻辑计算图不等于设备执行图：Q@Kᵀ 的转置合并不代表设备已融合，也未改变精度。",
                "本图由语义数据程序生成，不是手抄讨论稿；数值/权重未参与结构判断。",
            ],
        }
    )
    if tool:
        cards.append(
            {
                "dot": "violet",
                "title": "产物身份",
                "items": [
                    f"Archify CLI：{tool.get('version')}（{tool.get('cli_sha256', '')[:12]}…）",
                    f"Node：{tool.get('node')}；语义规则集 R1–R7；语义数据哈希与 manifest 见 T12 证据目录。",
                ],
            }
        )
    return cards


# ---------------------------------------------------------------------------
# 驱动：candidate → finalize → manifest
# ---------------------------------------------------------------------------


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _run_finalize(
    tool: ArchifyTool,
    candidate_path: Path,
    html_path: Path,
    *,
    repo_root: Path,
    out_dir: Path,
    workspace_root: Path,
    allow_browser_check_unavailable: bool = False,
    quality: str = "standard",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "node",
        str(tool.cli),
        "finalize",
        "architecture",
        str(candidate_path),
        str(html_path),
        "--quality",
        quality,
        "--json",
        "--repo-root",
        str(repo_root),
        "--out-dir",
        str(out_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    receipt: dict[str, Any] = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                receipt = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    gates = receipt.get("gates") or {}
    kind, detail = classify_browser_failure(receipt)
    browser_unavailable = (
        kind == "environment"
        and gates.get("validate") == "pass"
        and gates.get("deliver") == "pass"
        and gates.get("check") == "pass"
    )
    if completed.returncode != 0 or receipt.get("ok") is not True:
        if not (allow_browser_check_unavailable and browser_unavailable):
            raise PreviewError(
                "Archify finalize 未通过："
                f"exit={completed.returncode}; classify={kind} ({detail}); "
                f"stdout={stdout[:2000]}; stderr={completed.stderr[-2000:]}"
            )
        receipt = dict(receipt)
        receipt["browser_check_available"] = False
        receipt["browser_check_classification"] = kind
        receipt["browser_check_error"] = detail
    else:
        receipt = dict(receipt)
        receipt["browser_check_available"] = True
    return {"receipt": receipt, "stdout": stdout, "stderr": completed.stderr}


@dataclass(frozen=True)
class PreviewRequest:
    variant: str
    view: str
    form: str  # raw | normalized


def finalize_candidate(
    tool: ArchifyTool,
    candidate_path: Path,
    html_path: Path,
    *,
    repo_root: Path,
    out_dir: Path,
    workspace_root: Path,
    allow_browser_check_unavailable: bool = False,
    quality: str = "standard",
) -> dict[str, Any]:
    """公开的 finalize 驱动：产品分层图复用同一套 receipt/浏览器失败分类。"""
    return _run_finalize(
        tool,
        candidate_path,
        html_path,
        repo_root=repo_root,
        out_dir=out_dir,
        workspace_root=workspace_root,
        allow_browser_check_unavailable=allow_browser_check_unavailable,
        quality=quality,
    )


def generate_previews(
    workspace_root: str | Path,
    *,
    raw_bundles: Mapping[str, SemanticBundle],
    normalized_bundles: Mapping[str, SemanticBundle],
    normalization_reports: Mapping[str, Mapping[str, Any]],
    requests: Sequence[PreviewRequest],
    output_dir: Path,
    repo_root: Path,
    repository: Mapping[str, Any],
    allow_browser_check_unavailable: bool = False,
) -> dict[str, Any]:
    """按请求生成 candidate + finalize HTML，返回可复现 manifest。

    环境没有可用的 Chrome 时，validate/deliver/check 仍必须通过；browser-check
    失败作为环境限制记录（``browser_check_available=False``），不冒充通过。
    """
    root = Path(workspace_root)
    tool = discover_archify(root)
    identity = tool_identity(tool)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates").mkdir(exist_ok=True)
    entries: list[dict[str, Any]] = []
    for request in requests:
        bundle = (
            raw_bundles[request.variant]
            if request.form == "raw"
            else normalized_bundles[request.variant]
        )
        view = build_view_model(bundle, request.view)
        slug = f"{request.variant}-{request.view}-{request.form}"
        try:
            output_rel = output_dir.resolve().relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise PreviewError(f"预览输出目录必须位于工作区内: {output_dir}") from error
        candidate_rel = f"{output_rel}/{slug}.json"
        html_rel = f"{output_rel}/{slug}.html"
        candidate = build_candidate(
            bundle,
            view,
            output_relpath=html_rel,
            form=request.form,
            normalization_report=normalization_reports.get(request.variant)
            if request.form == "normalized"
            else normalization_reports.get(request.variant),
            tool=identity,
            repository=repository,
        )
        candidate_path = root / candidate_rel
        candidate_path.write_text(
            json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        edges_rel = f"{output_rel}/edges/{slug}-edges.json"
        edges_path = root / edges_rel
        edges_path.parent.mkdir(parents=True, exist_ok=True)
        edges_path.write_text(
            json.dumps(view.edge_records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        html_path = root / html_rel
        run = _run_finalize(
            tool,
            candidate_path,
            html_path,
            repo_root=repo_root,
            out_dir=output_dir / "receipts" / slug,
            workspace_root=root,
            allow_browser_check_unavailable=allow_browser_check_unavailable,
        )
        entries.append(
            {
                "variant": request.variant,
                "view": request.view,
                "form": request.form,
                "candidate": candidate_rel,
                "candidate_sha256": _sha256_file(candidate_path),
                "html": html_rel,
                "html_sha256": _sha256_file(html_path),
                "edges": edges_rel,
                "edges_sha256": _sha256_file(edges_path),
                "edge_records": len(view.edge_records),
                "finalize": run["receipt"],
                "semantic_bundle_sha256": hashlib.sha256(
                    _canonical_json_bytes(bundle.to_json())
                ).hexdigest(),
                "nodes": len(view.nodes),
                "connections": len(view.connections),
            }
        )
    manifest = {
        "schema": "t12-archify-previews.v1",
        "generated_by": "structure_reader.render.archify_preview",
        "tool": identity,
        "repository": dict(repository),
        "requests": [request.__dict__ for request in requests],
        "previews": entries,
    }
    (output_dir / "previews-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
