"""HF decoder 阅读页连接图 helper（展示层，里程碑一图示01）。

从同一 reading_document 的 operators/tensors 生成图结构与内嵌 SVG：
- 边一律来自 operators.inputs/outputs/weights 的真实张量引用
  （producer→consumer；无 producer 的 boundary/weight/cache 作为虚拟端口节点），
  不按算子顺序串线、不引入文档之外的数据流事实、不靠顺序号猜边；
- 分组是展示层规则（按算子 id 的确定性映射），聚合边可逐条追溯到
  原始节点/张量（tests/test_hf_decoder_diagram.py 核查）；
- 缓存图区分调用前状态、本次算子读写与调用后回写；条件标注为展示层注记，
  仅在对应算子与所引用参数同时存在于文档时渲染（如 N_win>0）。

纯函数、确定性输出（同一文档两次渲染字节相同）；不执行模型、不读外部资源、
不修改文档事实。HTML/xlsx 的数据内容不受本模块影响。
"""

from __future__ import annotations

import html

from structure_reader.render.diagram_source_labels import (
    MODELING_FILE,
    compute_type_label,
    op_source_label,
    op_source_locator,
    tensor_source_label,
    tensor_source_locator,
)

__all__ = [
    "module_groups",
    "build_module_graph",
    "build_overview_graph",
    "build_cache_graphs",
    "render_svg",
    "diagram_css",
]

# ---------------------------------------------------------------------------
# 展示层配置：模块分组与缓存分族（确定性映射；算子/张量不存在时不渲染）
# ---------------------------------------------------------------------------

# (group_id, 标题, 精确 id 元组 或 前缀字符串)；先匹配先得，未匹配入 other。
_MODULE_RULES: tuple[tuple[str, str, object], ...] = (
    ("mhc_attn", "mHC 权重与流坍缩（attn_hc）", "attn_hc_"),
    ("norm_attn", "RMSNorm（input_layernorm）", ("attn_norm",)),
    ("attn_q", "Attention — Q 支路（低秩+部分 RoPE）",
     ("rope_compress", "q_a_proj", "q_a_norm", "q_b_proj", "q_b_norm", "q_rope_apply")),
    ("attn_kv", "Attention — 共享 KV 支路（K=V）",
     ("kv_proj", "kv_head_view", "kv_rope_apply")),
    ("attn_cache", "Attention — 主 KV 缓存更新", ("cache_update",)),
    ("attn_csa", "CSA 压缩器（窗口缓冲/Ca-Cb 重叠/条目 RoPE）",
     ("c_kv_gate_proj", "c_store_window", "c_window_view", "c_pool", "c_rope_gen",
      "c_rope", "c_update_states")),
    ("attn_indexer", "Lightning Indexer（打分/topk/块偏置）",
     ("i_kv_gate_proj", "i_store_window", "i_window_view", "i_pool_rope",
      "i_update_states", "i_q_proj", "i_scorer", "i_topk", "c_block_bias")),
    ("attn_mask", "模型级滑动 mask 与压缩条目扩展",
     ("mask_model_build", "mask_extend")),
    ("attn_core", "eager 注意力核心（QK→mask→sink→softmax→AV）",
     ("kv_concat", "repeat_kv", "qk_scaled", "mask_add", "sink_softmax", "av_matmul")),
    ("attn_out", "逆 RoPE 与分组输出投影",
     ("attn_inv_rope", "attn_group_reshape", "o_a_proj", "o_b_proj")),
    ("res_attn", "mHC 残差混合（attn 段）",
     ("attn_residual_combT", "attn_residual_expand", "attn_residual_mix")),
    ("mhc_ffn", "mHC 权重与流坍缩（ffn_hc）", "ffn_hc_"),
    ("norm_ffn", "RMSNorm（post_attention_layernorm）", ("ffn_norm",)),
    ("moe_route", "MoE 输入展平与 TopK 路由",
     ("moe_flatten", "moe_router", "moe_dispatch")),
    ("moe_expert", "MoE 专家模板（一份；不复制 E 份）",
     ("moe_select_tokens", "moe_expert_slice", "moe_expert_gu", "moe_expert_clamp",
      "moe_expert_act", "moe_expert_down", "moe_expert_weight", "moe_combine",
      "moe_routed_view")),
    ("moe_shared", "共享专家（DeepseekV4MLP）", ("shared_gate_up", "shared_act_down")),
    ("moe_merge", "MoE 汇合（routed+shared）", ("moe_merge",)),
    ("res_ffn", "mHC 残差混合（ffn 段）",
     ("ffn_residual_combT", "ffn_residual_expand", "ffn_residual_mix")),
)

# 缓存分族：张量 id → family（机械规则；kv_full 属主 KV 本次读视图）。
_CACHE_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "main": ("cache_keys", "cache_cumlen", "kv_full"),
    "csa": ("cache_buffer_c_", "cache_compressed_c", "cache_entry_count_c",
            "cache_overlap_c_"),
    "indexer": ("cache_buffer_i_", "cache_compressed_i", "cache_entry_count_i",
                "cache_overlap_i_"),
}
_CACHE_FAMILY_TITLES = {
    "main": "主 KV 缓存（K=V 滑动窗口）",
    "csa": "CSA 压缩器缓存（buffer/entries/entry_count/overlap）",
    "indexer": "Indexer 缓存（同机制，'indexer' 键独立）",
}

# 条件标注（展示层注记）：仅在算子存在且所引用参数存在于阶段参数区时渲染。
_CONDITION_LABELS = {
    "c_window_view": ("仅 N_win>0", ("Nwin",)),
    "i_window_view": ("仅 N_win_i>0", ("Nwin_i",)),
    "c_store_window": ("余量回写（每次调用）", ()),
    "i_store_window": ("余量回写（每次调用）", ()),
    "c_update_states": ("追加 N_win 条（可为 0）", ("Nwin",)),
    "i_update_states": ("追加 N_win_i 条（可为 0）", ("Nwin_i",)),
}

# ---------------------------------------------------------------------------
# 缓存存储载体的字段（展示层）：同一 past_key_values.layers[layer_idx] 对象，
# 字段名/键来自 HF 源码事实（:156-294、CSA :626-690）；字段属于载体框内部，
# 读/写线的端点落在具体字段的时态端口（pre=调用前态 / post=调用后态 / view=本次读视图）。
# ---------------------------------------------------------------------------

_CACHE_FIELD_SPECS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "main": (
        ("keys", "self.keys / self.values（K=V 同存储）",
         "past_key_values.layers[layer_idx].keys/values（layer_idx 槽）"),
        ("cumlen", "self.cumulative_length（已累计 token 数）",
         "past_key_values.layers[layer_idx].cumulative_length · HF :196-197"),
        ("full", "full = cat(keys, kv_rope)（update 局部返回，本次读视图，不持久）",
         "HF :197-201 update 内局部变量 full（非 Cache 存储属性）"),
    ),
    "csa": (
        ("buffer_kv", "self.buffer_kv['compressor']（未闭窗余量）",
         "buffer_kv 字典的 'compressor' 键"),
        ("buffer_gate", "self.buffer_gate['compressor']",
         "buffer_gate 字典的 'compressor' 键"),
        ("compressed_kv", "self.compressed_kv['compressor']（已发射条目）",
         "compressed_kv 字典的 'compressor' 键"),
        ("entry_count", "self.entry_count['compressor']（窗口计数）",
         "entry_count 字典的 'compressor' 键"),
        ("overlap_kv", "self.overlap_kv['compressor']（闭窗 Ca 尾）",
         "overlap_kv 字典的 'compressor' 键"),
        ("overlap_gate", "self.overlap_gate['compressor']",
         "overlap_gate 字典的 'compressor' 键"),
    ),
    "indexer": (
        ("buffer_kv", "self.buffer_kv['indexer']（未闭窗余量）",
         "buffer_kv 字典的 'indexer' 键（与 compressor 分开）"),
        ("buffer_gate", "self.buffer_gate['indexer']",
         "buffer_gate 字典的 'indexer' 键"),
        ("compressed_kv", "self.compressed_kv['indexer']（已发射条目）",
         "compressed_kv 字典的 'indexer' 键"),
        ("entry_count", "self.entry_count['indexer']（窗口计数）",
         "entry_count 字典的 'indexer' 键"),
        ("overlap_kv", "self.overlap_kv['indexer']（闭窗 Ca 尾）",
         "overlap_kv 字典的 'indexer' 键"),
        ("overlap_gate", "self.overlap_gate['indexer']",
         "overlap_gate 字典的 'indexer' 键"),
    ),
}

# 张量 id → (载体族, 字段名)；字段名与 _CACHE_FIELD_SPECS 对齐。
_CACHE_TENSOR_FIELD: dict[str, tuple[str, str]] = {
    "cache_keys": ("main", "keys"), "cache_keys_post": ("main", "keys"),
    "cache_cumlen": ("main", "cumlen"), "kv_full": ("main", "full"),
    "cache_buffer_c_kv": ("csa", "buffer_kv"),
    "cache_buffer_c_kv_post": ("csa", "buffer_kv"),
    "cache_buffer_c_gate": ("csa", "buffer_gate"),
    "cache_buffer_c_gate_post": ("csa", "buffer_gate"),
    "cache_compressed_c": ("csa", "compressed_kv"),
    "cache_compressed_c_post": ("csa", "compressed_kv"),
    "cache_entry_count_c": ("csa", "entry_count"),
    "cache_entry_count_c_post": ("csa", "entry_count"),
    "cache_overlap_c_kv": ("csa", "overlap_kv"),
    "cache_overlap_c_kv_post": ("csa", "overlap_kv"),
    "cache_overlap_c_gate": ("csa", "overlap_gate"),
    "cache_overlap_c_gate_post": ("csa", "overlap_gate"),
    "cache_buffer_i_kv": ("indexer", "buffer_kv"),
    "cache_buffer_i_kv_post": ("indexer", "buffer_kv"),
    "cache_buffer_i_gate": ("indexer", "buffer_gate"),
    "cache_buffer_i_gate_post": ("indexer", "buffer_gate"),
    "cache_compressed_i": ("indexer", "compressed_kv"),
    "cache_compressed_i_post": ("indexer", "compressed_kv"),
    "cache_entry_count_i": ("indexer", "entry_count"),
    "cache_entry_count_i_post": ("indexer", "entry_count"),
    "cache_overlap_i_kv": ("indexer", "overlap_kv"),
    "cache_overlap_i_kv_post": ("indexer", "overlap_kv"),
    "cache_overlap_i_gate": ("indexer", "overlap_gate"),
    "cache_overlap_i_gate_post": ("indexer", "overlap_gate"),
}

# 字段时态端口：pre=调用前态（只读），post=调用后态（写/读本次产物），
# view=本次读视图（update 返回，不持久）。
_CACHE_PORT_LABELS = {"pre": "读前", "post": "写后", "view": "视图"}


def _cache_field_tid(tid: str) -> tuple[str, str] | None:
    return _CACHE_TENSOR_FIELD.get(tid)


def _cache_port_of(tid: str) -> str:
    if tid == "kv_full":
        return "view"
    return "post" if tid.endswith("_post") else "pre"


def _match(op_id: str, rule) -> bool:
    if isinstance(rule, str):
        return op_id.startswith(rule)
    return op_id in rule


def module_groups(phase: dict, rules=None) -> list[dict]:
    """把阶段算子按展示层规则分成模块组；每组 {id, title, ops}。

    覆盖完备：每个算子恰属一组（未匹配入 other），不丢不重。
    """
    active_rules = _MODULE_RULES if rules is None else rules
    groups: dict[str, dict] = {}
    order: list[str] = []
    for gid, title, rule in active_rules:
        groups[gid] = {"id": gid, "title": title, "ops": []}
        order.append(gid)
    for op in phase.get("operators", []):
        oid = op.get("id", "")
        for gid, _title, rule in active_rules:
            if _match(oid, rule):
                groups[gid]["ops"].append(op)
                break
        else:
            if "other" not in groups:
                groups["other"] = {"id": "other", "title": "其他", "ops": []}
                order.append("other")
            groups["other"]["ops"].append(op)
    return [groups[gid] for gid in order if gid in groups and groups[gid]["ops"]]


def _cache_family(tid: str) -> str | None:
    for family, prefixes in _CACHE_FAMILY_PREFIXES.items():
        for prefix in prefixes:
            if tid == prefix or tid.startswith(prefix):
                return family
    return None


def _consumers(phase: dict) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = {}
    for i, op in enumerate(phase.get("operators", [])):
        for tid in op.get("inputs", []):
            consumers.setdefault(tid, []).append(i)
    return consumers


def _weight_consumers(phase: dict) -> dict[str, list[int]]:
    weight_consumers: dict[str, list[int]] = {}
    for i, op in enumerate(phase.get("operators", [])):
        for tid in op.get("weights", []):
            weight_consumers.setdefault(tid, []).append(i)
    return weight_consumers


def _producers(phase: dict) -> dict[str, list[int]]:
    producers: dict[str, list[int]] = {}
    for i, op in enumerate(phase.get("operators", [])):
        for tid in op.get("outputs", []):
            producers.setdefault(tid, []).append(i)
    return producers


def _tensor_map(phase: dict) -> dict[str, dict]:
    return {t["id"]: t for t in phase.get("tensors", [])}


def _param_ids(phase: dict) -> set[str]:
    return {p["id"] for p in phase.get("parameters", [])}


def _edge_kind(tid: str, tensors: dict[str, dict]) -> str:
    kind = tensors.get(tid, {}).get("kind", "activation")
    if kind == "weight":
        return "weight"
    if kind == "cache":
        return "cache"
    return "flow"


def _tensor_label(tid: str, tensors: dict[str, dict]) -> str:
    """边上张量标签：源码变量/表达式名 + 符号 shape（ONNX 式：tensor 是箭头上的
    传输信息，不是计算框）。无源码映射时回退文档名并标注（内部 id 不冒充源码）。"""
    t = tensors.get(tid)
    shape = ",".join(str(a) for a in t.get("shape", [])) if t else ""
    src = tensor_source_label(tid)
    if src:
        return f"{src} [{shape}]" if shape else src
    name = t.get("name", tid) if t else tid
    label = f"{name} [{shape}]" if shape else name
    return label + "（语义id）" if t else label


def _cond_label(op_id: str, phase: dict) -> str:
    entry = _CONDITION_LABELS.get(op_id)
    if entry is None:
        return ""
    text, needed = entry
    return text if set(needed) <= _param_ids(phase) else ""


def _op_node(phase: dict, op: dict) -> dict:
    """计算框节点：源码调用名/表达式为主行，实际计算类型为副行（复合如实列出）。
    title 保留算子 id、模块路径与源码行号联查。"""
    src = op_source_label(op)
    locator = op_source_locator(op)
    label = src if src else op.get("name", op["id"])
    sub = compute_type_label(op)
    title_bits = [op["id"], op.get("module_path", "")]
    if locator:
        title_bits.append(f"{MODELING_FILE.split('/')[-1]}:{locator}")
    return {
        "id": f"op:{op['id']}", "kind": "op",
        "label": label, "sub": sub,
        "href": f"#op-{phase['id']}-{op['id']}",
        "title": " · ".join(b for b in title_bits if b),
    }


def _tensor_node(phase: dict, tid: str, tensors: dict[str, dict], kind_hint: str,
                 sub: str = "") -> dict:
    """非计算端点（端口）：渲染为小圆点/输入输出标识，不再画 tensor/权重/缓存矩形。
    文字（源码名+shape+时序角色）走箭头标签；title 保留内部 id 与源码行号联查。"""
    t = tensors.get(tid, {})
    src = tensor_source_label(tid)
    locator = tensor_source_locator(tid)
    if src:
        label = src
    else:
        label = t.get("name", tid)
    title = tid + (f" · {MODELING_FILE.split('/')[-1]}:{locator}" if locator else "")
    return {
        "id": f"t:{tid}", "kind": kind_hint,
        "label": label, "sub": sub or tid,
        "href": f"#t-{phase['id']}-{tid}", "title": title,
    }


def _edge(src: str, dst: str, kind: str, label: str, cond: str, href: str, title: str) -> dict:
    return {"src": src, "dst": dst, "kind": kind, "label": label,
            "cond": cond, "href": href, "title": title}


# ---------------------------------------------------------------------------
# 图构建：模块图 / 总览图 / 缓存图
# ---------------------------------------------------------------------------


def _dedup_edges(edges: list[dict]) -> list[dict]:
    """按 (src,dst,kind,张量,条件,标签) 去重；不同张量/条件/类型保留。"""
    seen: set[tuple] = set()
    out: list[dict] = []
    for e in edges:
        key = (e["src"], e["dst"], e["kind"], e["title"], e["cond"], e["label"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def build_module_graph(phase: dict, group: dict, field_ports: bool = True) -> dict:
    """模块算子图：节点=组内算子+外部端口，边=真实张量引用。

    - 内部边：producer op → consumer op（组内；仅由输出侧生成一次，不做输入/输出
      双循环重复）；
    - 外部输入端口：被组内消费、但 producer 在组外或无 producer 的张量
      （boundary/weight/cache 按其类型着色，其余为普通端口）；
    - 输出端口：组内产出、但无组内下游的张量（缓存回写含条件标注）；
    - 权重引用为虚线边（引用，不改变数据流方向；组内产出的权重切片由
      producer→consumer 直连表达，不重复端口）。
    - field_ports=True 时，缓存张量不再画成圆点，而是接入"持久状态存储"载体的
      具体字段，读/写线端点落在字段的时态端口（pre/post/view）；
      field_ports=False 保留旧的缓存端口行为（缓存图族回放用）。
    边最终按 (src,dst,kind,张量,条件,标签) 去重；不同张量各自成边不丢。
    """
    tensors = _tensor_map(phase)
    producers = _producers(phase)
    consumers = _consumers(phase)
    weight_consumers = _weight_consumers(phase)
    ops = group["ops"]
    op_ids = {o["id"] for o in ops}

    referenced: set[str] = set()
    for op in ops:
        referenced.update(op.get("inputs", []))
        referenced.update(op.get("outputs", []))
    cache_tids = ({tid for tid in referenced
                   if tensors.get(tid, {}).get("kind") == "cache"
                   and _cache_field_tid(tid) is not None}
                  if field_ports else set())

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for op in ops:
        nodes[op["id"]] = _op_node(phase, op)

    for op in ops:
        oid = op["id"]
        for tid in op.get("inputs", []):
            if tid in cache_tids:
                continue  # 缓存状态统一走载体字段端点（下方）
            internal = any(phase["operators"][p]["id"] in op_ids
                           for p in producers.get(tid, []))
            if internal:
                continue  # 组内 producer→consumer 边统一由输出侧生成（防重复）
            kind = _edge_kind(tid, tensors)
            tkind = tensors.get(tid, {}).get("kind", "activation")
            hint = tkind if tkind in ("boundary", "weight", "cache") else "port"
            nodes[f"t:{tid}"] = _tensor_node(phase, tid, tensors, hint)
            edges.append(_edge(
                f"t:{tid}", f"op:{oid}", kind, _tensor_label(tid, tensors),
                "", f"#t-{phase['id']}-{tid}", tid))
        for tid in op.get("weights", []):
            if tid in cache_tids:
                continue
            produced_inside = any(phase["operators"][p]["id"] in op_ids
                                  for p in producers.get(tid, []))
            if produced_inside:
                continue  # 组内产出（如当前专家切片）：由 producer→consumer 直连边表达
            nodes.setdefault(f"t:{tid}", _tensor_node(phase, tid, tensors, "weight"))
            edges.append(_edge(
                f"t:{tid}", f"op:{oid}", "weight", _tensor_label(tid, tensors),
                "", f"#t-{phase['id']}-{tid}", tid))
    # 输出侧：真实下游（inputs+weights 引用）生成内部边，否则输出端口
    for op in ops:
        oid = op["id"]
        for tid in op.get("outputs", []):
            if tid in cache_tids:
                continue
            kind = _edge_kind(tid, tensors)
            downstream = {phase["operators"][c]["id"]
                          for c in consumers.get(tid, []) + weight_consumers.get(tid, [])
                          if phase["operators"][c]["id"] in op_ids
                          and phase["operators"][c]["id"] != oid}
            cond = _cond_label(oid, phase) if kind == "cache" else ""
            if downstream:
                for dst in sorted(downstream):
                    edges.append(_edge(
                        f"op:{oid}", f"op:{dst}", kind, _tensor_label(tid, tensors),
                        cond, f"#t-{phase['id']}-{tid}", tid))
            else:
                tkind = tensors.get(tid, {}).get("kind", "activation")
                hint = "cache" if tkind == "cache" else "port"
                sub = "调用后回写" if tkind == "cache" and tid.endswith("_post") else ""
                nodes[f"t:{tid}"] = _tensor_node(phase, tid, tensors, hint, sub)
                edges.append(_edge(
                    f"op:{oid}", f"t:{tid}", kind, _tensor_label(tid, tensors),
                    cond, f"#t-{phase['id']}-{tid}", tid))

    # 缓存状态 → 载体字段端点：读线接 pre/视图端口，写线接 post 端口；
    # 同一字段的读前态与写后态是两个不同端口，更新后的值不会冒充调用前输入。
    field_ports_used: dict[str, set[str]] = {}
    field_order: list[str] = []
    for tid in sorted(cache_tids):
        fam_field = _cache_field_tid(tid)
        if fam_field is None:
            continue
        fam, fkey = fam_field
        fid = f"f:{fam}:{fkey}"
        producers_in = [p for p in producers.get(tid, [])
                        if phase["operators"][p]["id"] in op_ids]
        consumers_in = [c for c in consumers.get(tid, [])
                        if phase["operators"][c]["id"] in op_ids]
        if not producers_in and not consumers_in:
            continue
        ports = field_ports_used.setdefault(fid, set())
        if fid not in field_order:
            field_order.append(fid)
        label = _tensor_label(tid, tensors)
        href = f"#t-{phase['id']}-{tid}"
        # 时态端口由「阶段内是否已产出该状态」决定：调用前状态才进 pre 端口；
        # 已产出（本组或组外）的调用后/视图值只能进 post/view 端口，绝不冒充前态。
        produced_anywhere = bool(producers.get(tid))
        if produced_anywhere:
            port = _cache_port_of(tid)
            for p in producers_in:
                oid = phase["operators"][p]["id"]
                ports.add(port)
                edges.append(_edge(
                    f"op:{oid}", f"{fid}#{port}", "cache",
                    ("产出本次读视图：" if port == "view" else "写调用后状态：") + label,
                    _cond_label(oid, phase), href, tid))
            for c in consumers_in:
                cid = phase["operators"][c]["id"]
                ports.add(port)
                edges.append(_edge(
                    f"{fid}#{port}", f"op:{cid}", "cache",
                    f"本次读视图：{label}", "", href, tid))
        else:
            ports.add("pre")
            for c in consumers_in:
                cid = phase["operators"][c]["id"]
                edges.append(_edge(
                    f"{fid}#pre", f"op:{cid}", "cache",
                    f"读调用前状态：{label}", "", href, tid))
    for fid in field_order:
        fam, fkey = fid[2:].split(":", 1)
        nodes[fid] = _field_node(fam, fkey, sorted(field_ports_used[fid]))

    graph: dict = {"id": group["id"], "title": group["title"],
                   "nodes": list(nodes.values()), "edges": _dedup_edges(edges),
                   "phase": phase["id"]}
    if field_ports and field_order:
        fams = sorted({fid[2:].split(":", 1)[0] for fid in field_order})
        _attach_store_layout(phase, ops, graph, field_order)
        graph["frames"] = [
            {
                "id": f"frame:store:{fam}",
                "label": "持久状态存储",
                "sub": ("past_key_values.layers[layer_idx] : DeepseekV4CSACache"
                        + (" · HF :156-294" if fam == "main" else " · CSA :626-690")),
                "children": [fid for fid in field_order
                             if fid[2:].split(":", 1)[0] == fam],
                "style": "store",
            }
            for fam in fams
            if any(fid[2:].split(":", 1)[0] == fam for fid in field_order)
        ]
    return graph


def _field_node(family: str, field_key: str, ports: list[str]) -> dict:
    """载体字段行：字段名属于存储框内部；ports 决定该行暴露的时态端口。"""
    spec = next((s for s in _CACHE_FIELD_SPECS[family] if s[0] == field_key), None)
    if spec is None:
        label, sub = field_key, f"{family} 字段"
    else:
        _key, label, sub = spec
    key = "compressor" if family == "csa" else ("indexer" if family == "indexer" else None)
    if family == "main" and field_key == "cumlen":
        # 真实 HF 属性是 self.cumulative_length（modeling_deepseek_v4.py:196-197），
        # 内部展示键 main.cumlen / JSON cache_cumlen 不变。
        owner = "past_key_values.layers[layer_idx].cumulative_length"
    elif family == "main" and field_key == "full":
        # full 是 update 的局部返回变量（:197-201），不是 Cache 的持久字段：
        # 它只以“本次读视图/不持久”身份出现在视图端口，不冒充存储属性。
        owner = ("DeepseekV4CSACache.update 返回的局部 full（本次读视图，不持久；"
                 "不是 past_key_values 的 Cache 字段）")
    else:
        owner = f"past_key_values.layers[layer_idx].{field_key}"
        if key:
            owner += f"['{key}']"
    return {
        "id": f"f:{family}:{field_key}", "kind": "field",
        "label": label, "sub": sub, "w": _W_FIELD, "wrap": 26,
        "ports": ports, "href": "", "title": owner,
    }


_COMPRESSOR_OPTIONS = (
    ("ctl:csa", "compressed_sparse_attention", "DeepseekV4CSACompressor · 当前生效（layer_idx=4）", True),
    ("ctl:hca", "heavily_compressed_attention", "DeepseekV4HCACompressor（无 Indexer）· 未展开", False),
    ("ctl:swa", "sliding_attention", "None（SWA 无压缩器）· 未展开", False),
)


def build_csa_composite_graph(phase: dict, csa_group: dict, indexer_group: dict) -> dict:
    """CSA 详细阅读：真父子嵌套容器（CSA ⊃ Indexer）+ 图内三选一配置 + 字段端口。

    CSA 与 Indexer 是同一模块内的子模块（HF :618 创建 self.indexer、:690 调用），
    不是两个平级章节：CSA 容器包住自身的算子/存储字段，并把 Indexer 子图作为
    嵌套容器放在其内部下方。配置选择是控制流（COMPRESSOR_CLASSES 三选一），
    用控制线/实例化线与 tensor/cache 线区分；未生效的 HCA/SWA 只诚实标注未展开，
    不伪装切换。所有包含边界只是导航/归类，不冒充计算算子。
    """
    csa = build_module_graph(phase, csa_group, field_ports=True)
    idx = build_module_graph(phase, indexer_group, field_ports=True)
    nodes: dict[str, dict] = {n["id"]: dict(n) for n in csa["nodes"]}
    edges: list[dict] = [dict(e) for e in csa["edges"]]
    positions: dict[str, tuple[int, float]] = dict(csa.get("positions", {}))
    k_offset = max((c for c, _ in positions.values()), default=0) + 1
    id_map: dict[str, str] = {}
    for n in idx["nodes"]:
        nid = n["id"]
        new_id = "pi:" + nid[2:] if nid.startswith("t:") else nid
        id_map[nid] = new_id
        if nid.startswith("f:") and new_id in nodes:
            ports = set(nodes[new_id].get("ports") or []) | set(n.get("ports") or [])
            nodes[new_id]["ports"] = sorted(ports)
            continue
        nn = dict(n)
        nn["id"] = new_id
        nodes[new_id] = nn
        pos = idx.get("positions", {}).get(nid)
        if pos is not None:
            layer, row = pos
            # Indexer 子图整体下移到 CSA 之后；与 CSA 共享的存储字段保留
            # CSA 存储列里的原位置（上面已按 id 合并）。
            positions[new_id] = (layer + k_offset, row)

    def _remap(ref: str) -> str:
        base, sep, port = ref.partition("#")
        return id_map.get(base, base) + (sep + port if sep else "")

    for e in idx["edges"]:
        ee = dict(e)
        ee["src"] = _remap(e["src"])
        ee["dst"] = _remap(e["dst"])
        edges.append(ee)

    # ---- 图内配置选择：COMPRESSOR_CLASSES[config.layer_types[layer_idx]] ----
    selector_ids: list[str] = []
    nodes["ctl:config"] = {
        "id": "ctl:config", "kind": "control",
        "label": "config.layer_types[layer_idx]", "wrap": 26,
        "sub": '= "compressed_sparse_attention"（layer 4）', "w": 236, "wrap": 26,
        "href": "", "title": "COMPRESSOR_CLASSES 选择依据 · HF :745-749",
    }
    positions["ctl:config"] = (-3, 1.0)
    selector_ids.append("ctl:config")
    for k, (oid, key, sub, active) in enumerate(_COMPRESSOR_OPTIONS):
        nodes[oid] = {
            "id": oid, "kind": "controlopt", "active": active,
            "label": key, "sub": sub, "w": 208, "wrap": 22,
            "href": "", "title": f"COMPRESSOR_CLASSES['{key}'] · HF :745-749",
        }
        positions[oid] = (-2, float(k))
        selector_ids.append(oid)
        edges.append({"src": "ctl:config", "dst": oid, "kind": "control",
                      "label": "控制选择", "cond": "", "href": "",
                      "title": f"COMPRESSOR_CLASSES['{key}']"})
    edges.append({"src": "ctl:csa", "dst": "frame:csa", "kind": "control",
                  "label": "实例化 self.compressor（控制线，非 tensor）",
                  "cond": "", "href": "", "title": "HF :794-795",
                  "src_side": "bottom", "dst_side": "top", "span_hint": 2})

    csa_ops = [f"op:{o['id']}" for o in csa_group["ops"]]
    idx_ops = [f"op:{o['id']}" for o in indexer_group["ops"]]
    csa_fields = [n["id"] for n in csa["nodes"] if n["id"].startswith("f:csa:")]
    idx_fields = [n["id"] for n in idx["nodes"] if n["id"].startswith("f:indexer:")]
    # 同一持久 Cache 对象的两个键各自成组：字段重新回到明确的存储载体边界内，
    # self.buffer_*/compressed_kv/... 的 owner 是 past_key_values.layers[layer_idx]
    # 的 DeepseekV4CSACache（'compressor' / 'indexer' 键），不是 Compressor/Indexer。
    frames = [
        {"id": "frame:selector", "style": "config",
         "label": "压缩器配置选择（控制流，非 tensor）",
         "sub": "COMPRESSOR_CLASSES[config.layer_types[layer_idx]] · HF :745-749 / :776",
         "children": selector_ids},
        {"id": "frame:store:indexer", "style": "store",
         "label": "持久状态存储 · 'indexer' 键",
         "sub": "past_key_values.layers[layer_idx] : DeepseekV4CSACache（同一对象）",
         "children": idx_fields},
        {"id": "frame:indexer", "style": "indexer",
         "label": "Lightning Indexer（self.indexer）",
         "sub": 'HF :618 创建 / :690 调用 · 键独立 "indexer" · 打分/topk/块偏置',
         "children": idx_ops + ["frame:store:indexer"]},
        {"id": "frame:store:csa", "style": "store",
         "label": "持久状态存储 · 'compressor' 键",
         "sub": "past_key_values.layers[layer_idx] : DeepseekV4CSACache（同一对象）",
         "children": csa_fields},
        {"id": "frame:csa", "style": "csa",
         "label": "CSA Compressor（DeepseekV4CSACompressor）",
         "sub": ":794-795 创建 self.compressor · 包含 Lightning Indexer",
         "children": csa_ops + ["frame:store:csa", "frame:indexer"]},
    ]
    return {"id": "attn_csa_nested", "title": csa_group["title"],
            "nodes": list(nodes.values()), "edges": _dedup_edges(edges),
            "phase": phase["id"], "positions": positions, "frames": frames}


def _op_ranks(phase: dict, ops: list[dict]) -> dict[str, int]:
    """组内算子的纵向层级：真实张量 producer→consumer 最长路径（不按顺序号）。"""
    ids = {o["id"] for o in ops}
    consumers = _consumers(phase)
    layer = {oid: 0 for oid in ids}
    pairs: list[tuple[str, str]] = []
    for op in phase.get("operators", []):
        if op["id"] not in ids:
            continue
        for tid in op.get("outputs", []):
            for c in consumers.get(tid, []):
                cid = phase["operators"][c]["id"]
                if cid in ids and cid != op["id"]:
                    pairs.append((op["id"], cid))
    for _ in range(len(ids) + 1):
        changed = False
        for src, dst in pairs:
            if layer[dst] < layer[src] + 1:
                layer[dst] = layer[src] + 1
                changed = True
        if not changed:
            break
    return layer


def _attach_store_layout(phase: dict, ops: list[dict], graph: dict,
                         field_order: list[str]) -> None:
    """显式布局：算子列在左、外部输入/权重端口在左侧栏、载体字段在右侧存储列。

    存储列是载体框内部，字段逐行排列（按真实写入/读取所在的算子层就近对齐，
    同层字段顺序下移一行），读/写线端点因此落在具体字段行上。
    """
    ranks = _op_ranks(phase, ops)
    positions: dict[str, tuple[int, float]] = {}
    for op in ops:  # 同层多算子按文档序横排（真实并列，不串成顺序）
        layer = ranks[op["id"]]
        row = sum(1 for o in ops
                  if ranks[o["id"]] == layer and
                  ops.index(o) < ops.index(op))
        positions[f"op:{op['id']}"] = (layer, float(row))
    left_used: dict[int, int] = {}
    right_used: dict[int, int] = {}
    for e in graph["edges"]:
        src = e["src"].split("#", 1)[0]
        dst = e["dst"].split("#", 1)[0]
        if src.startswith("t:") and dst.startswith("op:"):
            layer = ranks[dst[3:]]
            if src not in positions:
                k = left_used.get(layer, 0)
                left_used[layer] = k + 1
                positions[src] = (layer, -(1.0 + 0.55 * k))
        elif src.startswith("op:") and dst.startswith("t:"):
            layer = ranks[src[3:]]
            if dst not in positions:
                k = right_used.get(layer, 0)
                right_used[layer] = k + 1
                positions[dst] = (layer, 1.0 + 0.55 * k)
    anchors: dict[str, list[int]] = {}
    for e in graph["edges"]:
        src = e["src"].split("#", 1)[0]
        dst = e["dst"].split("#", 1)[0]
        if src.startswith("op:") and dst.startswith("f:"):
            anchors.setdefault(dst, []).append(ranks[src[3:]])
        elif src.startswith("f:"):
            anchors.setdefault(src, [])
    order = sorted(field_order,
                   key=lambda f: (min(anchors.get(f) or [0]),
                                  field_order.index(f)))
    max_row = max([r for _, r in positions.values()] + [0.0])
    store_col = max_row + 1.0
    last_layer = -1
    used_layers = set()
    for fid in order:
        anchor = min(anchors.get(fid) or [0])
        layer = max(anchor, last_layer + 1)
        while layer in used_layers:
            layer += 1
        positions[fid] = (layer, store_col)
        used_layers.add(layer)
        last_layer = layer
    graph["positions"] = positions


# 总览的显式布局：层 = 自上而下的真实主数据流；列 = 同层真实并列支路
# （Attention 的 Q / 共享 KV / CSA→Indexer 三支并排，缓存载体就近，接入通路）。
_OVERVIEW_POS: dict[str, tuple[int, float]] = {
    "hidden_streams_in": (0, 0.0),
    "position_ids": (0, 1.0),
    "input_ids": (0, 2.0),
    "mhc_attn": (1, 0.0),
    "norm_attn": (2, 0.0),
    "attn_q": (3, 0.0),
    "attn_kv": (3, 1.0),
    "attn_csa": (3, 2.0),
    "attn_mask": (4, 0.0),
    "attn_cache": (4, 1.0),
    "attn_indexer": (4, 2.0),
    "c:main": (6, 0.45),
    "attn_core": (5, 1.0),
    "c:csa": (4, 3.0),
    "c:indexer": (5, 3.0),
    "attn_out": (6, 1.0),
    "res_attn": (7, 0.0),
    "mhc_ffn": (8, 0.0),
    "norm_ffn": (9, 0.0),
    "moe_route": (10, 0.0),
    "moe_shared": (10, 1.0),
    "moe_expert": (11, 0.0),
    "moe_merge": (12, 0.0),
    "res_ffn": (13, 0.0),
    "hidden_streams_out": (14, 0.0),
}

# 仅 Decoder 总览使用的展开留白（不影响 34 张模块详图）
_OVERVIEW_SPACING = {"col_extra": 96.0, "row_extra": 52.0, "frame_extra": 34.0}

_OVERVIEW_FRAMES = (
    ("frame:attention", "Attention（DeepseekV4Attention）",
     ":794-795 创建 self.compressor；包含 CSA → Indexer",
     "attn",
     ("m:attn_q", "m:attn_kv", "m:attn_csa", "m:attn_indexer", "m:attn_cache",
      "m:attn_mask", "m:attn_core", "m:attn_out", "c:main", "c:csa", "c:indexer",
      "frame:csa")),
    ("frame:csa", "CSA Compressor（DeepseekV4CSACompressor）",
     ":618 创建 self.indexer、:690 调用（父模块）", "csa",
     ("m:attn_csa", "frame:indexer")),
    ("frame:indexer", "Lightning Indexer（self.indexer）",
     "打分 / topk / 块偏置（分组键独立）", "indexer", ("m:attn_indexer",)),
)


def build_overview_graph(phase: dict, rules=None) -> dict:
    """Decoder 总览：模块框 + 真实包含容器 + 缓存载体 + 长连接同名接口。

    聚合边逐条由真实张量支撑（producer 属 src、consumer 属 dst），方向不变；
    同层并列支路按列展开（不串成假顺序）；缓存载体作为"持久状态存储"接入
    Attention/CSA/Indexer 计算通路；跨多层的残差/来源类长连接改为少量成对的
    同名接口（来源端 ▸ 名称 / 去向端 名称 ▸），避免长线互相压盖，
    同时保留完整依赖（两条 stub 边各自保留原张量清单与方向）。
    """
    tensors = _tensor_map(phase)
    producers = _producers(phase)
    consumers = _consumers(phase)
    groups = module_groups(phase, rules)
    op_group: dict[str, str] = {}
    for g in groups:
        for op in g["ops"]:
            op_group[op["id"]] = g["id"]

    positions: dict[str, tuple[int, float]] = {}
    nodes: dict[str, dict] = {}
    for g in groups:
        sub_txt = f"导航 · {len(g['ops'])} 算子"
        nodes[f"m:{g['id']}"] = {
            "id": f"m:{g['id']}", "kind": "nav", "label": g["title"],
            "sub": sub_txt,
            "href": f"#mod-{phase['id']}-{g['id']}",
            "title": "，".join(op["id"] for op in g["ops"]),
        }
        if g["id"] in _OVERVIEW_POS:
            positions[f"m:{g['id']}"] = _OVERVIEW_POS[g["id"]]
    for tid in ("hidden_streams_in", "input_ids", "position_ids", "hidden_streams_out"):
        if tid in tensors:
            nodes[f"t:{tid}"] = {
                "id": f"t:{tid}", "kind": "boundary",
                "label": tensor_source_label(tid) or tensors[tid]["name"],
                "sub": tid, "href": f"#t-{phase['id']}-{tid}", "title": tid,
            }
            if tid in _OVERVIEW_POS:
                positions[f"t:{tid}"] = _OVERVIEW_POS[tid]
    cache_families_present = {
        _cache_family(tid) for tid, t in tensors.items()
        if (t.get("kind") == "cache" and _cache_family(tid)) or tid == "kv_full"
    }
    for family in ("main", "csa", "indexer"):
        if family in cache_families_present:
            key = f"c:{family}"
            fields = "，".join(s[1].split("（")[0] for s in _CACHE_FIELD_SPECS[family])
            nodes[key] = {
                "id": key, "kind": "carrier",
                "label": "持久状态存储",
                "sub": f"{_CACHE_FAMILY_TITLES[family]} · {fields}",
                "href": "",
                "title": "past_key_values.layers[layer_idx] : DeepseekV4CSACache · "
                         f"{_CACHE_FAMILY_TITLES[family]}",
            }
            positions[key] = _OVERVIEW_POS[key]

    agg: dict[tuple[str, str], dict] = {}
    for tid, t in tensors.items():
        producer_groups = [op_group[phase["operators"][p]["id"]]
                           for p in producers.get(tid, []) if phase["operators"][p]["id"] in op_group]
        consumer_groups = {op_group[phase["operators"][c]["id"]]
                           for c in consumers.get(tid, []) if phase["operators"][c]["id"] in op_group}
        kind = _edge_kind(tid, tensors)
        if producer_groups:
            src = f"m:{producer_groups[0]}"
            if not consumer_groups:
                if t.get("kind") == "boundary":
                    key = (src, f"t:{tid}")
                    entry = agg.setdefault(key, {"tensors": [], "kinds": set()})
                    entry["tensors"].append(tid)
                    entry["kinds"].add(kind)
                continue
        elif t.get("kind") == "boundary":
            src = f"t:{tid}"
        elif kind == "cache" and _cache_family(tid):
            src = f"c:{_cache_family(tid)}"
        else:
            continue
        for cg in sorted(consumer_groups):
            dst = f"m:{cg}"
            if dst == src:
                continue
            entry = agg.setdefault((src, dst), {"tensors": [], "kinds": set()})
            entry["tensors"].append(tid)
            entry["kinds"].add(kind)
    for tid, t in tensors.items():
        if t.get("kind") != "cache" or not _cache_family(tid):
            continue
        family = _cache_family(tid)
        for p in producers.get(tid, []):
            gsrc = op_group.get(phase["operators"][p]["id"])
            if gsrc:
                entry = agg.setdefault((f"m:{gsrc}", f"c:{family}"),
                                       {"tensors": [], "kinds": set()})
                entry["tensors"].append(tid)
                entry["kinds"].add("cache")

    group_titles = {f"m:{g['id']}": g["title"] for g in groups}
    edges: list[dict] = []
    # 先收集长残差/来源类连接（跨多层的真实聚合依赖），再汇总成少量成对同名接口。
    out_groups: dict[str, list[dict]] = {}
    in_groups: dict[str, list[dict]] = {}
    ordinary: list[tuple[tuple[str, str], dict]] = []
    for (src, dst), entry in sorted(agg.items()):
        if src not in nodes or dst not in nodes:
            continue
        n = len(entry["tensors"])
        title = "，".join(entry["tensors"])
        name = "、".join(
            tensor_source_label(tid) or tensors.get(tid, {}).get("name", tid)
            for tid in entry["tensors"][:2])
        is_cache_set = entry["kinds"] == {"cache"}
        src_pos = positions.get(src, (0, 0.0))
        dst_pos = positions.get(dst, (0, 0.0))
        span = dst_pos[0] - src_pos[0]
        if is_cache_set:
            edge_entry = {"src": src, "dst": dst, "kind": "cache",
                          "label": (f"读前态 {n} 项" if src.startswith("c:")
                                    else f"写后态 {n} 项"),
                          "cond": "", "href": "", "title": title}
            if {src, dst} == {"c:main", "m:attn_cache"}:
                # 主 KV 载体与缓存更新之间的真实读写线：走载体与 attn_core 之间的
                # 窄通道，避开无关的 attn_core 计算框（不改为注记）。
                if src == "c:main":
                    edge_entry.update(src_side="top", dst_side="left",
                                      lane_off=97.0)
                else:
                    edge_entry.update(src_side="left", dst_side="top",
                                      lane_off=-11.0)
            edges.append(edge_entry)
            continue
        if src.startswith("m:") and dst.startswith("m:") and entry["kinds"] == {"cache"}:
            edges.append({"src": src, "dst": dst, "kind": "cache",
                          "label": f"本次读视图 {n} 项", "cond": "", "href": "",
                          "title": title})
            continue
        iface_ok = (src.startswith("m:") and dst.startswith("m:")) or (
            src.startswith("t:") and dst == "m:res_attn")
        if span >= 4 and iface_ok:
            out_groups.setdefault(src, []).append(
                {"dst": dst, "name": name, "title": title})
            in_groups.setdefault(dst, []).append(
                {"src": src, "name": name, "title": title})
            continue
        ordinary.append(((src, dst), entry))

    def _place_iface(layer: int, base_row: float) -> tuple[int, float]:
        # 同一层已有接口 chip（含来源端/去向端）依次左移一档，绝不重叠。
        used = sum(1 for c, r in positions.values()
                   if c == layer and -2.8 < r < base_row + 1e-6)
        # chip 宽 216 > 0.55 列（194px）：间距至少 0.65 列，避免两个 chip 框重叠
        return layer, base_row - 0.65 * used

    for src, items in sorted(out_groups.items()):
        layer, row = _place_iface(int(positions[src][0]),
                                  positions[src][1] - 0.62)
        cid = f"if:{src}:out"
        positions[cid] = (layer, row)
        names = "、".join(dict.fromkeys(i["name"] for i in items))
        nodes[cid] = {
            "id": cid, "kind": "iface", "label": f"▸ {names}",
            "sub": "去向：" + "、".join(dict.fromkeys(
                group_titles.get(i["dst"], i["dst"]) for i in items)),
            "w": 216, "wrap": 17, "href": "",
            "title": "同名接口（来源端）· " + "；".join(i["title"] for i in items),
        }
        for k, item in enumerate(items):
            edges.append({"src": src, "dst": cid, "kind": "flow", "label": "",
                          "cond": "", "href": "", "title": item["title"],
                          "src_side": "left", "dst_side": "right",
                          "span_hint": 0, "iface_pair": f"{src}->{item['dst']}"})
    for dst, items in sorted(in_groups.items()):
        layer, row = _place_iface(int(positions[dst][0]),
                                  positions[dst][1] - 0.62)
        cid = f"if:{dst}:in"
        positions[cid] = (layer, row)
        names = "、".join(dict.fromkeys(i["name"] for i in items))
        nodes[cid] = {
            "id": cid, "kind": "iface", "label": f"{names} ▸",
            "sub": "来源：" + "、".join(dict.fromkeys(
                group_titles.get(i["src"], i["src"]) for i in items)),
            "w": 216, "wrap": 17, "href": "",
            "title": "同名接口（去向端）· " + "；".join(i["title"] for i in items),
        }
        for item in items:
            # 去向端接口可能排在来源端 chip 外侧：在走廊里向下绕行再进入节点，
            # 不横穿同层另一个 chip。
            edges.append({"src": cid, "dst": dst, "kind": "flow", "label": "",
                          "cond": "", "href": "", "title": item["title"],
                          "src_side": "right", "dst_side": "left",
                          "span_hint": 0, "detour": 58.0,
                          "iface_pair": f"{item['src']}->{dst}"})

    for (src, dst), entry in ordinary:
        n = len(entry["tensors"])
        title = "，".join(entry["tensors"])
        name = "、".join(
            (tensor_source_label(tid) or tensors.get(tid, {}).get("name", tid)).split("（")[0]
            for tid in entry["tensors"][:2])
        label = f"{n} 项：{name}" + ("…" if n > 2 else "")
        if dst.startswith("m:"):
            href = f"#mod-{phase['id']}-{dst[2:]}"
        elif dst.startswith("t:"):
            href = f"#t-{phase['id']}-{dst[2:]}"
        else:
            href = ""
        edge_entry = {"src": src, "dst": dst, "kind": "flow", "label": label,
                      "cond": "", "href": href, "title": title}
        if src == "m:attn_indexer" and dst == "m:attn_mask":
            edge_entry["detour"] = 58.0
        if src == "m:attn_q" and dst == "m:attn_out":
            # 从 q 左缘出发，沿 attn_mask 左侧列间隙下行再进入 out，不穿 mask 框
            edge_entry["src_side"] = "left"
            edge_entry["dst_side"] = "top"
            edge_entry["lane_off"] = -140.0
        edges.append(edge_entry)

    # 无边界的孤立输入标识（如 input_ids 在本阶段算子中无引用）不画成无标签圆点
    used_ids = {e["src"].split("#", 1)[0] for e in edges} | \
               {e["dst"].split("#", 1)[0] for e in edges}
    for nid in [nid for nid, n in nodes.items()
                if n["kind"] == "boundary" and nid not in used_ids]:
        nodes.pop(nid, None)
        positions.pop(nid, None)

    frames = [
        {"id": fid, "label": label, "sub": sub, "style": style,
         "children": list(children),
         "extra_top": 20 if fid == "frame:attention" else 0}
        for fid, label, sub, style, children in _OVERVIEW_FRAMES
    ]
    return {"id": "overview", "title": "Decoder 总览", "nodes": list(nodes.values()),
            "edges": edges, "phase": phase["id"], "positions": positions,
            "frames": frames, "spacing": dict(_OVERVIEW_SPACING)}

def _cache_role(tid: str, phase: dict, producers: dict[str, list[int]],
                consumers: dict[str, list[int]], cache_op_ids: set[str]) -> str:
    """缓存张量的时间角色：来自真实 producer/consumer 引用，不靠命名后缀猜测。

    - 无 producer（文档内无人产出）→ 调用前状态（历史/缺省）；
    - 有 producer 且无族外消费者 → 调用后回写（本次产出供下调用）；
    - 有 producer 且被族外算子消费 → 本次读视图（参与本次计算）；
      其 id 以 _post 结尾的（累计列表）补充「兼调用后保存」注记——主角色仍来自引用。
    """
    if not producers.get(tid):
        return "调用前状态"
    outside_consumers = [phase["operators"][c]["id"] for c in consumers.get(tid, [])
                         if phase["operators"][c]["id"] not in cache_op_ids]
    if not outside_consumers:
        return "调用后回写"
    role = "本次读视图"
    if tid.endswith("_post"):
        role += "·兼调用后保存"
    return role


def build_cache_graphs(phase: dict, family_rules=None, cache_op_ids=None) -> list[dict]:
    """缓存图（每族一张）：调用前历史 → 本次新输入与算子读写 → 调用后回写。

    以 build_module_graph 为基底生成缓存算子间的真实 activation 数据路径与
    本次新输入端口（如 kv_rope/c_kv/c_gate），再叠加缓存状态节点（角色来自
    producer/consumer 引用）与「下游算子」chip（真实族外消费者，联查原算子）。
    族与缓存算子集合为展示层配置（可注入以便 fixture 回放）；算子/张量不存在
    时不渲染。
    """
    tensors = _tensor_map(phase)
    producers = _producers(phase)
    consumers = _consumers(phase)
    active_cache_ops = ({"cache_update", "c_store_window", "c_window_view", "c_rope_gen",
                         "c_rope", "c_update_states", "i_store_window", "i_window_view",
                         "i_pool_rope", "i_update_states", "mask_model_build"}
                        if cache_op_ids is None else set(cache_op_ids))
    active_family_rules = _CACHE_FAMILY_PREFIXES if family_rules is None else family_rules
    all_cache_tids = {tid for tid, t in tensors.items() if t.get("kind") == "cache"}

    def family_of(tid: str) -> str | None:
        for family, prefixes in active_family_rules.items():
            for prefix in prefixes:
                if tid == prefix or tid.startswith(prefix):
                    return family
        return None

    fam_tensors: dict[str, list[str]] = {}
    for tid in sorted(all_cache_tids):
        family = family_of(tid)
        if family:
            fam_tensors.setdefault(family, []).append(tid)

    graphs = []
    for family in sorted(fam_tensors):
        tids = fam_tensors[family]
        tid_set = set(tids)
        inner_ops = [op for op in phase["operators"]
                     if (set(op.get("inputs", [])) | set(op.get("outputs", []))) & tid_set
                     and op["id"] in active_cache_ops]
        # 传递闭包：经任意共享张量（含 activation）相连的其他缓存算子并入本族，
        # 使算子间真实数据路径完整（如 c_rope_gen→c_rope→c_update_states）。
        inner_ids = {op["id"] for op in inner_ops}
        cache_op_by_id = {op["id"]: op for op in phase["operators"]
                          if op["id"] in active_cache_ops}
        changed = True
        while changed:
            changed = False
            for oid, op in cache_op_by_id.items():
                if oid in inner_ids:
                    continue
                refs = set(op.get("inputs", [])) | set(op.get("outputs", []))
                for sel_id in list(inner_ids):
                    sel = cache_op_by_id.get(sel_id)
                    if sel and refs & (set(sel.get("inputs", [])) |
                                       set(sel.get("outputs", []))):
                        inner_ids.add(oid)
                        changed = True
                        break
        doc_order = {op["id"]: i for i, op in enumerate(phase["operators"])}
        inner_ops = [cache_op_by_id[i] for i in sorted(inner_ids, key=doc_order.__getitem__)]
        if not inner_ops:
            continue
        inner_ids = {op["id"] for op in inner_ops}
        base = build_module_graph(phase, {"id": family, "title": _CACHE_FAMILY_TITLES.get(
            family, family), "ops": inner_ops}, field_ports=False)
        nodes = {n["id"]: dict(n) for n in base["nodes"]}
        edges = [dict(e) for e in base["edges"]]
        # 缓存状态节点角色重标（覆盖基底端口的默认 sub），并规范缓存读边标签
        for tid in tids:
            role = _cache_role(tid, phase, producers, consumers, inner_ids)
            key = f"t:{tid}"
            if key not in nodes:
                nodes[key] = _tensor_node(phase, tid, tensors, "cache")
            nodes[key]["kind"] = "cache"
            nodes[key]["sub"] = f"{role} · {tid}"
        for e in edges:
            if e["kind"] != "cache":
                continue
            if e["src"].startswith("t:") and e["dst"].startswith("op:"):
                tid = e["src"][2:]
                e["label"] = (("读调用前状态" if not producers.get(tid)
                               else "读本次产出（缓存视图）")
                              + f"：{_tensor_label(tid, tensors)}")
            elif e["src"].startswith("op:") and e["dst"].startswith("t:"):
                tid = e["dst"][2:]
                role = _cache_role(tid, phase, producers, consumers, inner_ids)
                e["label"] = (("产出本次读视图" if role.startswith("本次读视图")
                               else "写调用后状态")
                              + f"：{_tensor_label(tid, tensors)}")
        # 下游 chip：族外真实消费者（inputs 或 weights 引用）
        for tid in tids:
            role = _cache_role(tid, phase, producers, consumers, inner_ids)
            if not role.startswith("本次读视图"):
                continue
            for c in sorted(consumers.get(tid, [])):
                cid = phase["operators"][c]["id"]
                if cid in inner_ids:
                    continue
                nodes.setdefault(f"d:{cid}", {
                    "id": f"d:{cid}", "kind": "downstream",
                    "label": f"下游算子 {cid}", "sub": "参与本次计算",
                    "href": f"#op-{phase['id']}-{cid}", "title": cid,
                })
                edges.append(_edge(f"t:{tid}", f"d:{cid}", "flow",
                                   f"本次读视图→参与计算：{_tensor_label(tid, tensors)}",
                                   "", "", cid))
        graphs.append({
            "id": family, "title": _CACHE_FAMILY_TITLES.get(family, family),
            "anchor": f"cache-{phase['id']}-{family}",
            "nodes": list(nodes.values()), "edges": _dedup_edges(edges),
            "phase": phase["id"],
        })
    return graphs


# ---------------------------------------------------------------------------
# SVG 渲染（确定性；无外部资源；文字转义）
# ---------------------------------------------------------------------------


def _layout(nodes: list[dict], edges: list[dict]) -> dict[str, tuple[int, int]]:
    """最长路径分层（op/module 节点）+ 同层文档序排布；返回 id → (层, 层内序)。

    纵向渲染约定：层 = 自上而下 rank（y），层内序 = 横向位（x）。
    其余节点：仅作为边源（边界/权重/调用前缓存状态）放最上层（层 -1）；
    作为边终点（输出端口/调用后回写/下游 chip）放最下层（层 max+1）；
    孤立节点放最上层——主数据流自上而下，历史→本次读→回写顺向。
    """
    layered_kinds = ("op", "module", "nav", "navlane")
    layered = [n for n in nodes if n["kind"] in layered_kinds]
    others = [n for n in nodes if n["kind"] not in layered_kinds]
    layer = {n["id"]: 0 for n in layered}
    layered_ids = set(layer)
    pairs = [(e["src"], e["dst"]) for e in edges
             if e["src"] in layered_ids and e["dst"] in layered_ids]
    for _ in range(len(layered) + 1):
        changed = False
        for src, dst in pairs:
            if layer[dst] < layer[src] + 1:
                layer[dst] = layer[src] + 1
                changed = True
        if not changed:
            break
    max_layer = max(layer.values(), default=0)
    srcs = {e["src"] for e in edges}
    dsts = {e["dst"] for e in edges}
    positions: dict[str, tuple[int, int]] = {}
    rows: dict[int, int] = {}
    for n in layered:
        c = layer[n["id"]]
        r = rows.get(c, 0)
        rows[c] = r + 1
        positions[n["id"]] = (c, r)
    left_row = 0
    right_row = 0
    for n in others:
        nid = n["id"]
        if nid in srcs and nid not in dsts or nid not in srcs and nid not in dsts:
            positions[nid] = (-1, left_row)
            left_row += 1
        else:
            positions[nid] = (max_layer + 1, right_row)
            right_row += 1
    return positions


_NODE_W = 172.0
_NODE_H_MIN = 44.0
# 缓存字段行（载体框内部）与接口 chip 的显示宽度；字段端点落在字段行左缘。
_W_FIELD = 236.0
_H_FIELD = 48.0
_W_IFACE = 188.0
# 最小列间隙/行走廊；实际值按本图最宽标签块自适应加宽（render_svg）
_COL_GAP_MIN = 104.0
_ROW_GAP_MIN = 22.0
_PAD = 16.0


def _node_width(node: dict) -> float:
    return float(node.get("w", _NODE_W))


def _node_shape(node: dict) -> str:
    """节点几何形状：框（op/nav/控制/字段/接口 chip）、IO 标识、端口圆点。"""
    if node["kind"] in ("op", "nav", "carrier", "field", "iface",
                        "control", "controlopt"):
        return "box"
    if node["id"] in ("t:hidden_streams_in", "t:hidden_streams_out"):
        return "io"
    return "dot"


def _node_lines(node: dict) -> list[str]:
    """框内主行（源码名）：按需换行、无行数上限——用户要求框内源码名完整可见，
    节点高度随行数自适应；不为塞进固定行数而截断真实信息。"""
    return _wrap(node["label"], int(node.get("wrap", 19)), max_lines=None)


def _node_sub_lines(node: dict) -> list[str]:
    """计算类型副行：完整显示不截断（最多 2 行，长者换行）。"""
    sub = node.get("sub", "")
    return _wrap(sub, 30)[:2] if sub else []


def _node_height(node: dict, lines: list[str]) -> float:
    """节点高度容纳实际文字行数与副标题行数，避免重叠/截断。"""
    if node["kind"] == "field":
        return _H_FIELD
    sub_lines = _node_sub_lines(node)
    h = 12.0 + 12.0 * len(lines) + (13.0 * len(sub_lines) if sub_lines else 8.0)
    return max(_NODE_H_MIN, h)


def _wrap(text: str, limit: int, max_lines: int | None = 3) -> list[str]:
    lines: list[str] = []
    current, width = "", 0
    for ch in text:
        w = 2 if ord(ch) > 0x2E80 else 1
        if width + w > limit:
            lines.append(current)
            current, width = ch, w
        else:
            current += ch
            width += w
    if current:
        lines.append(current)
    if max_lines is not None:
        lines = lines[:max_lines]
    return lines or [""]


def _wrap_edge_label(text: str) -> list[str]:
    """边标签有界换行（最多 3 行；全文放 <title> 联查/悬停）——
    源码名与 shape 维度不再因两行截断丢失。"""
    return _wrap(text, 30)[:3]


def _label_block_box(full_label: str) -> tuple[float, float, int]:
    """标签块（宽度 px、行数）；宽度按渲染器自身度量（CJK 10px/其他 5.5px）+ 8px 留白。"""
    lines = _wrap_edge_label(full_label)
    if not lines:
        return 0.0, 0, 0
    width = max(sum(10.0 if ord(ch) > 0x2E80 else 5.5 for ch in line)
                for line in lines) + 8.0
    return width, len(lines), 0


def render_svg(graph: dict, wide: bool = False) -> str:
    """图结构 → 内嵌 SVG 字符串（节点/边可点击联查原详情锚点）。

    布局：优先采用 graph["positions"] 的显式 (层, 列)——层=自上而下的主数据流
    rank，列=同层并列分支的横向位（真实并列展开，不串成顺序）；未给出时按真实
    张量引用的最长路径自动分层。所有端点按真实几何（框边界/端口圆点/载体字段
    的时态端口/容器边界）求，不用隐形矩形糊弄。

    容器：graph["frames"] 声明父子嵌套边界（Attention ⊃ CSA ⊃ Indexer、
    持久状态存储），按子节点/子容器的实际包围盒绘制；容器只表达包含边界，
    不冒充计算算子、不参与数据流方向。

    渲染顺序：容器 → 边 → 边标签 → 节点；标签整块避让全部真实矩形（节点、
    字段行、容器表头），并沿所属边的真实路径反解空位；列间隙/行走廊按本图
    最宽标签块与穿越该走廊的标签堆叠需求自适应。
    """
    uid = html.escape(f"{graph.get('phase', 'p')}-{graph.get('id', 'g')}", quote=True)
    spacing = graph.get("spacing") or {}
    nodes = list(graph["nodes"])
    edges = list(graph["edges"])
    node_by_id = {n["id"]: n for n in nodes}
    frame_defs = {f["id"]: f for f in graph.get("frames", [])}
    explicit = bool(graph.get("positions"))
    if explicit:
        positions = dict(graph["positions"])
        missing = [n for n in nodes if n["id"] not in positions]
        if missing:
            base_c = max((c for c, _ in positions.values()), default=0) + 1
            for k, n in enumerate(missing):
                positions[n["id"]] = (base_c, float(k))
    else:
        positions = _layout(nodes, edges)

    def _edge_label_lines(e: dict) -> list[str]:
        cond = e.get("cond", "")
        label = e.get("label", "")
        if cond and label:
            return [cond] + _wrap_edge_label(label)
        if label:
            return _wrap_edge_label(label)
        return [cond] if cond else []

    max_block_w, max_label_lines = 0.0, 0
    for e in edges:
        lines = _edge_label_lines(e)
        if not lines:
            continue
        w = max(sum(10.0 if ord(ch) > 0x2E80 else 5.5 for ch in ln) for ln in lines) + 8.0
        max_block_w = max(max_block_w, w)
        max_label_lines = max(max_label_lines, len(lines))
    col_gap = max(_COL_GAP_MIN, max_block_w + 10.0) + float(spacing.get("col_extra", 0.0))
    if wide:
        col_gap = max(col_gap, 220.0)  # 兼容旧调用：仅加宽列间隙
    col_w = _NODE_W + col_gap

    lines_by_node = {n["id"]: _node_lines(n) for n in nodes}
    h_by_node = {n["id"]: _node_height(n, lines_by_node[n["id"]]) for n in nodes}
    layers = [c for c, _ in positions.values()]
    min_c, max_c = (min(layers), max(layers)) if layers else (0, 0)
    # 行走廊纵向留白：跨层边标签只落在自己路径中点所在的走廊，按
    # (走廊, 横向桶) 累计该处真正需要的堆叠高度——不把长边需求重复叠加到
    # 它穿过的每条走廊（旧实现会把画布撑到上万像素高）。
    slots = [r for _, r in positions.values()]
    min_r, max_r = (min(slots), max(slots)) if slots else (0, 0)
    corridor_demand: dict[tuple[int, int], float] = {}
    for e in edges:
        src = e["src"].split("#", 1)[0]
        dst = e["dst"].split("#", 1)[0]
        if src not in positions or dst not in positions:
            continue
        c1 = positions[src][0]
        c2 = positions[dst][0]
        if c1 == c2:
            continue  # 同层边标签走在横向巷道（by_x 排列），不占行走廊
        lines = _edge_label_lines(e)
        if not lines:
            continue
        block_h = 10.0 + 11.0 * (len(lines) - 1)
        mid_layer = (c1 + c2) / 2.0
        corr = int(mid_layer)
        mid_row = (positions[src][1] + positions[dst][1]) / 2.0
        mid_x = _PAD + (mid_row - min_r) * col_w
        key = (corr, int(round(mid_x / 200.0)))
        corridor_demand[key] = corridor_demand.get(key, 0.0) + max(13.0, block_h) + 6.0
    base_gap = max(_ROW_GAP_MIN, 8.0 + 11.0 * (max_label_lines - 1) + 2.0 + 5.0)
    corridor_gap: dict[int, float] = {}
    for c in range(int(min_c), int(max_c) + 1):
        need = max([v for (cc, _b), v in corridor_demand.items() if cc == c],
                   default=0.0)
        corridor_gap[c] = max(base_gap, need) + float(spacing.get("row_extra", 0.0))
    # 容器表头预留：容器边界从最上层子节点向上留出表头高度，必须在它上方
    # 的行走廊里预留同等空间，否则表头会压到上一层不属于该容器的模块框。
    def _frame_top_layer(fid: str) -> int | None:
        f = frame_defs.get(fid)
        if not f:
            return None
        vals: list[int] = []
        for cid in f.get("children", []):
            if cid in positions:
                vals.append(int(positions[cid][0]))
            elif cid in frame_defs:
                sub = _frame_top_layer(cid)
                if sub is not None:
                    vals.append(sub)
        return min(vals) if vals else None

    header_by_layer: dict[int, float] = {}
    for fid, f in frame_defs.items():
        top_layer = _frame_top_layer(fid)
        if top_layer is None:
            continue
        sub_lines = _wrap(f.get("sub", ""), 42)[:2]
        header_h = (20.0 + 12.0 * len(sub_lines) + float(f.get("extra_top", 0.0))
                    + float(spacing.get("frame_extra", 0.0)))
        header_by_layer[top_layer] = header_by_layer.get(top_layer, 0.0) + header_h
    for layer, need in header_by_layer.items():
        corr = layer - 1
        if corr in corridor_gap:
            corridor_gap[corr] = max(corridor_gap[corr], need + 10.0)
    row_gap = max(corridor_gap.values(), default=base_gap)
    # 纵向主阅读：层 = 自上而下 rank（y），层内序 = 横向位（x）。
    rank_h: dict[int, float] = {}
    for nid, (c, _) in positions.items():
        rank_h[c] = max(rank_h.get(c, 0.0), h_by_node[nid])
    rank_y: dict[int, float] = {}
    y_cursor = _PAD
    for c in range(int(min_c), int(max_c) + 1):
        rank_y[c] = y_cursor
        y_cursor += rank_h.get(c, _NODE_H_MIN) + corridor_gap.get(c, base_gap)
    node_height = y_cursor - corridor_gap.get(int(max_c), base_gap) + _PAD
    height = node_height
    width = (max_r - min_r + 1) * col_w + 2 * _PAD

    def x_of(r: float) -> float:
        return _PAD + (r - min_r) * col_w

    # 真实节点几何：框（172×h，宽节点按自身宽度居中）、IO 标识（×24）、
    # 端口圆点（r=5）、载体字段行（端口在左缘的 pre/post/view 时态位）。
    node_geom: dict[str, dict] = {}
    for n in nodes:
        nid = n["id"]
        c, r = positions[nid]
        x = x_of(r)
        y = rank_y[c]
        shape = _node_shape(n)
        if shape == "box":
            w = _node_width(n)
            h = h_by_node[nid]
            gx = x + (_NODE_W - w) / 2.0
            geom = {
                "top": (gx + w / 2, y), "bottom": (gx + w / 2, y + h),
                "left": (gx, y + h / 2), "right": (gx + w, y + h / 2),
                "cx": gx + w / 2, "cy": y + h / 2, "shape": "box",
                "x": gx, "y": y, "w": w, "h": h,
            }
            if n["kind"] == "field":
                ports = list(n.get("ports") or [])
                if len(ports) == 1:
                    offs = {ports[0]: h / 2}
                elif len(ports) == 2:
                    offs = {ports[0]: h * 0.33, ports[1]: h * 0.67}
                else:
                    offs = {p: h * (0.26 + 0.24 * k) for k, p in enumerate(ports)}
                for p, off in offs.items():
                    py = y + off
                    geom[f"port:{p}:left"] = (gx, py)
                    geom[f"port:{p}:right"] = (gx + w, py)
                    geom[f"port:{p}:top"] = (gx + w / 2, y)
                    geom[f"port:{p}:bottom"] = (gx + w / 2, y + h)
                geom["ports"] = offs
        elif shape == "io":
            geom = {
                "top": (x + _NODE_W / 2, y), "bottom": (x + _NODE_W / 2, y + 24),
                "left": (x, y + 12), "right": (x + _NODE_W, y + 12),
                "cx": x + _NODE_W / 2, "cy": y + 12, "shape": "io",
                "x": x, "y": y, "w": _NODE_W, "h": 24,
            }
        else:
            cx, cy = x + 10.0, y + 12.0
            geom = {
                "top": (cx, cy - 6.0), "bottom": (cx, cy + 6.0),
                "left": (cx - 6.0, cy), "right": (cx + 6.0, cy),
                "cx": cx, "cy": cy, "shape": "dot",
                "x": cx - 6.0, "y": cy - 6.0, "w": 12.0, "h": 12.0,
            }
        node_geom[nid] = geom

    def anchor(ref: str, side: str) -> tuple[float, float]:
        base, _, port = ref.partition("#")
        geom = node_geom[base]
        if port:
            return geom.get(f"port:{port}:{side}") or geom[f"port:{port}:left"]
        return geom[side]

    # 容器（嵌套边界）：包围盒由子节点/子容器实际几何决定；表头在包围盒上方。
    frame_box: dict[str, tuple[float, float, float, float]] = {}
    frame_header_h: dict[str, float] = {}

    def _frame_box(fid: str) -> tuple[float, float, float, float] | None:
        if fid in frame_box:
            return frame_box[fid]
        f = frame_defs.get(fid)
        if f is None:
            return None
        xs: list[float] = []
        ys: list[float] = []
        xe: list[float] = []
        ye: list[float] = []
        for cid in f.get("children", []):
            if cid in node_geom:
                g = node_geom[cid]
                xs.append(g["x"])
                ys.append(g["y"])
                xe.append(g["x"] + g["w"])
                ye.append(g["y"] + g["h"])
            elif cid in frame_defs:
                b = _frame_box(cid)
                if b:
                    xs.append(b[0])
                    ys.append(b[1])
                    xe.append(b[2])
                    ye.append(b[3])
        if not xs:
            return None
        sub_lines = _wrap(f.get("sub", ""), 42)[:2]
        header = (20.0 + 12.0 * len(sub_lines) + float(f.get("extra_top", 0.0))
                  + float(spacing.get("frame_extra", 0.0)))
        box = (min(xs) - 15.0, min(ys) - header, max(xe) + 15.0, max(ye) + 15.0)
        frame_box[fid] = box
        frame_header_h[fid] = header
        return box

    for fid in frame_defs:
        _frame_box(fid)
        if fid in frame_box:
            x0, y0, x1, y1 = frame_box[fid]
            node_geom[fid] = {
                "top": ((x0 + x1) / 2, y0), "bottom": ((x0 + x1) / 2, y1),
                "left": (x0, (y0 + y1) / 2), "right": (x1, (y0 + y1) / 2),
                "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "shape": "frame",
                "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
            }

    # 横向巷道中心（同层边标签空位）/ 纵向走廊中心（跨层边标签空位）
    lane_centers: list[float] = []
    for r in range(int(min_r) + 1, int(max_r) + 1):
        left = x_of(r - 1) + _NODE_W
        lane_centers.append(left + col_gap / 2)
    corridor_centers: list[float] = []
    for c in range(int(min_c), int(max_c)):
        corridor_centers.append(rank_y[c] + rank_h.get(c, _NODE_H_MIN)
                                + corridor_gap.get(c, base_gap) / 2)
    if min_c <= max_c:
        corridor_centers.append(rank_y[int(min_c)] / 2)
        corridor_centers.append(rank_y[int(max_c)] + rank_h.get(int(max_c), _NODE_H_MIN)
                                + corridor_gap.get(int(max_c), base_gap) / 2)

    pair_count: dict[tuple[str, str], int] = {}
    for e in edges:
        src = e["src"].split("#", 1)[0]
        dst = e["dst"].split("#", 1)[0]
        if src in node_geom and dst in node_geom:
            key = tuple(sorted((src, dst)))
            pair_count[key] = pair_count.get(key, 0) + 1
    pair_idx: dict[tuple[str, str], int] = {}

    edge_parts: list[str] = []
    edge_pts_all: list[tuple] = []
    label_slots: list[dict] = []

    # 标签遮挡检测的矩形：所有真实节点框/端口小方/字段行 + 容器表头条带
    rects = [(node_geom[n["id"]]["x"], node_geom[n["id"]]["y"],
              node_geom[n["id"]]["w"], node_geom[n["id"]]["h"])
             for n in nodes if n["id"] in node_geom]
    for fid, box in frame_box.items():
        rects.append((box[0], box[1], box[2] - box[0], frame_header_h.get(fid, 20.0)))

    def _bezier(pt0, pt1, pt2, pt3, t):
        mt = 1 - t
        x = (mt ** 3 * pt0[0] + 3 * mt * mt * t * pt1[0]
             + 3 * mt * t * t * pt2[0] + t ** 3 * pt3[0])
        y = (mt ** 3 * pt0[1] + 3 * mt * mt * t * pt1[1]
             + 3 * mt * t * t * pt2[1] + t ** 3 * pt3[1])
        return x, y

    def _t_at(target: float, axis: int, pts) -> float | None:
        """在自身贝塞尔上反解坐标=target 的参数 t（扫描+二分；无交则 None）。"""
        prev = None
        for k in range(0, 41):
            t = 0.015 + 0.97 * k / 40
            val = _bezier(*pts, t)[axis]
            if prev is not None and (prev[1] - target) * (val - target) <= 0 \
                    and prev[1] != val:
                a, b = prev[0], t
                fa = prev[1] - target
                for _ in range(26):
                    m = (a + b) / 2
                    fm = _bezier(*pts, m)[axis] - target
                    if fa * fm <= 0:
                        b = m
                    else:
                        a, fa = m, fm
                return (a + b) / 2
            prev = (t, val)
        return None

    def _block_free(cx, cy, n_lines, half_w):
        top, bot = cy - 8.0, cy + 11.0 * (n_lines - 1) + 2.0
        for rx, ry, rw, rh in rects:
            if cx + half_w > rx and cx - half_w < rx + rw and bot > ry and top < ry + rh:
                return False
        return True

    placed: list[dict] = []

    def _free_of_labels(lx, ly, n_lines, half_w):
        top, bot = ly - 8.0, ly + 11.0 * (n_lines - 1) + 2.0
        for p in placed:
            ptop = p["ly"] - 8.0
            pbot = p["ly"] + 11.0 * (len(p["lines"]) - 1) + 2.0
            if (lx + half_w + 3.0 > p["lx"] - p["half_w"]
                    and lx - half_w - 3.0 < p["lx"] + p["half_w"]):
                separated = (top >= pbot + 3.0) or (ptop >= bot + 3.0)
                if not separated:
                    return False
        return True

    for e in edges:
        src = e["src"]
        dst = e["dst"]
        base_src = src.split("#", 1)[0]
        base_dst = dst.split("#", 1)[0]
        if base_src not in node_geom or base_dst not in node_geom:
            continue
        key = tuple(sorted((base_src, base_dst)))
        i = pair_idx.get(key, 0)
        pair_idx[key] = i + 1
        # 层差：两端都显式定位时用真实层级；容器端点用 span_hint。
        if base_src in positions and base_dst in positions:
            span = int(positions[base_dst][0] - positions[base_src][0])
        else:
            span = int(e.get("span_hint", 1))
        side_src = e.get("src_side")
        side_dst = e.get("dst_side")
        r1 = positions[base_src][1] if base_src in positions else 0.0
        r2 = positions[base_dst][1] if base_dst in positions else 0.0
        long_cross = (abs(span) >= 2 and abs(r2 - r1) >= 0.35
                      and node_by_id.get(base_src, {}).get("kind") != "field"
                      and node_by_id.get(base_dst, {}).get("kind") != "field")
        if side_src is None or side_dst is None:
            if long_cross:
                side_src = "bottom" if span > 0 else "top"
                side_dst = "top" if span > 0 else "bottom"
            elif span > 0:
                side_src, side_dst = "bottom", "top"
            elif span < 0:
                side_src, side_dst = "top", "bottom"
            else:
                if r2 >= r1:
                    side_src, side_dst = "right", "left"
                else:
                    side_src, side_dst = "left", "right"
        # 载体字段端点：读/写线一律从字段左缘的时态端口进出（字段行在存储列，
        # 面向算子列），不从字段顶/底穿出，避免穿过同列其它字段行。
        src_is_field = node_by_id.get(base_src, {}).get("kind") == "field"
        dst_is_field = node_by_id.get(base_dst, {}).get("kind") == "field"
        if src_is_field or dst_is_field:
            if src_is_field:
                side_src = "left"
                side_dst = "right" if not dst_is_field else "left"
            if dst_is_field:
                side_dst = "left"
                side_src = "right" if not src_is_field else "right"
        x1, y1 = anchor(src, side_src)
        x2, y2 = anchor(dst, side_dst)
        end_trim = (0.0 if long_cross or src_is_field or dst_is_field
                    else (3.0 if node_geom[base_dst]["shape"] in ("box", "io") else 0.0))
        # 平行边（同一对节点，如读/写）：端点精确不变，用弧线在中段分开。
        pair_off = 0.0
        if pair_count.get(key, 1) > 1:
            n_pair = pair_count[key]
            pair_off = (i - (n_pair - 1) / 2.0) * 24.0
        r1 = positions[base_src][1] if base_src in positions else 0.0
        r2 = positions[base_dst][1] if base_dst in positions else 0.0
        same_column = abs(r2 - r1) < 0.35
        # 长跨层边：显式 lane 走线（左侧分道）；同列长边绕开中间计算框；
        # 其余交替小弧。字段读/写线只在存储列与算子列之间的走廊里走。
        if span != 0:
            mid_y = (y1 + y2) / 2
            end_y = y2 - (end_trim if span > 0 else -end_trim)
            if "lane" in e:
                lane_x = -40.0 - 168.0 * float(e["lane"])
                pts = ((x1, y1), (lane_x, mid_y), (lane_x, mid_y), (x2, end_y))
            elif src_is_field or dst_is_field:
                # 字段读/写：在存储列与算子列之间的走廊里纵向走线
                pts = ((x1, y1), (x1, mid_y), (x2, mid_y), (x2, end_y))
            elif same_column and abs(span) >= 2:
                # 同列长边：沿列侧分道绕开中间计算框
                lane_x = x1 + (128.0 if i % 2 == 0 else -128.0) + pair_off
                pts = ((x1, y1), (lane_x, y1), (lane_x, end_y), (x2, end_y))
            elif long_cross:
                # 跨多层/隔列的边：在源列旁边的分道竖向下行到目标上方的
                # 行走廊，再在走廊里横移到目标正上方进入顶边——不斜穿中间层
                # 的计算框，长边各有分道可区分。
                lane_x = x1 + float(e.get("lane_off",
                                         110.0 if x2 > x1 else -110.0)) + pair_off
                if span > 0:
                    pts = ((x1, y1), (lane_x, y1 + 6.0),
                           (lane_x, end_y - 10.0), (x2, end_y))
                else:
                    pts = ((x1, y1), (lane_x, y1 - 6.0),
                           (lane_x, end_y + 10.0), (x2, end_y))
            else:
                bow = 26.0 if i % 2 == 0 else -26.0
                bow += pair_off
                pts = ((x1, y1), (x1 + bow, mid_y), (x2 + bow, mid_y), (x2, end_y))
            path_d = (f"M{pts[0][0]:g},{pts[0][1]:g} C{pts[1][0]:g},{pts[1][1]:g} "
                      f"{pts[2][0]:g},{pts[2][1]:g} {pts[3][0]:g},{pts[3][1]:g}")
        else:
            mid_x = (x1 + x2) / 2
            end_trim_x = end_trim if x2 >= x1 else -end_trim
            detour = float(e.get("detour", 0.0))
            if detour == 0.0 and not (src_is_field or dst_is_field) \
                    and abs(r2 - r1) >= 1.2:
                # 同层长跨列边：从行带上方向绕开中间计算框（真实同层依赖）
                detour = -(rank_h.get(int(positions[base_src][0]), _NODE_H_MIN) / 2.0 + 34.0)
            # 平行边只在控制点处分开，端点仍精确落在端口/字段时态位上。
            ex = x2 - end_trim_x
            if detour:
                pts = ((x1, y1), (x1, y1 + pair_off + detour),
                       (ex, y2 + pair_off + detour), (ex, y2))
            else:
                pts = ((x1, y1), (mid_x, y1 + pair_off),
                       (mid_x, y2 + pair_off), (ex, y2))
            path_d = (f"M{pts[0][0]:g},{pts[0][1]:g} C{pts[1][0]:g},{pts[1][1]:g} "
                      f"{pts[2][0]:g},{pts[2][1]:g} {pts[3][0]:g},{pts[3][1]:g}")
        cls = (f"dg-edge {e['kind']}"
               + (" cond" if e.get("cond") else "")
               + (" parallel" if pair_count.get(key, 1) > 1 else ""))
        marker_kind = {"flow": "flow", "weight": "weight", "control": "control"}.get(
            e["kind"], "cache")
        path = (f'<path class="{cls}" d="{path_d}" '
                f'marker-end="url(#dg-arr-{marker_kind}-{uid})"/>')
        if e.get("href"):
            path = f'<a href="{html.escape(e["href"])}">{path}</a>'
        edge_parts.append(path)
        edge_pts_all.append(pts)
        label_lines = _edge_label_lines(e)
        has_cond = bool(e.get("cond")) and bool(e.get("label"))
        if label_lines and any(ln for ln in label_lines):
            n_lines = len(label_lines)
            half_w = max(sum(10.0 if ord(ch) > 0x2E80 else 5.5 for ch in line)
                         for line in label_lines) / 2 + 4.0
            ly0 = 3.0 + 5.5 * (n_lines - 1) - 2.0
            mid_lx, mid_ly = _bezier(*pts, 0.5)
            lx, ly = mid_lx, mid_ly - ly0
            if not (_block_free(lx, ly, n_lines, half_w)
                    and _free_of_labels(lx, ly, n_lines, half_w)):
                cands: list[tuple[float, float]] = []
                if span != 0:
                    mid_y0 = (y1 + y2) / 2
                    ys_cand = sorted({round(v, 1) for v in corridor_centers}
                                     | {round(mid_y0 + k * 12.0, 1)
                                        for k in (-2, -1, 0, 1, 2)})
                    for cy_ in ys_cand:
                        if not (min(y1, y2) - 2.0 <= cy_ <= max(y1, y2) + 2.0):
                            continue
                        t = _t_at(cy_, 1, pts)
                        if t is None:
                            continue
                        px, py = _bezier(*pts, t)
                        cands.append((px, py))
                else:
                    mid_x0 = (x1 + x2) / 2
                    xs_cand = sorted({round(v, 1) for v in lane_centers}
                                     | {round(mid_x0 + k * 40.0, 1)
                                        for k in (-2, -1, 0, 1, 2)})
                    for lx_ in xs_cand:
                        if not (min(x1, x2) - 2.0 <= lx_ <= max(x1, x2) + 2.0):
                            continue
                        t = _t_at(lx_, 0, pts)
                        if t is None:
                            continue
                        px, py = _bezier(*pts, t)
                        for dy in (0.0, 25.0, -25.0, 50.0, -50.0):
                            cands.append((px, py + dy))
                for px, py in cands:
                    cand_ly = py - (5.5 * (n_lines - 1) - 3.0)
                    if _block_free(px, cand_ly, n_lines, half_w) and \
                            _free_of_labels(px, cand_ly, n_lines, half_w):
                        lx, ly = px, cand_ly
                        break
            placed.append({"lx": lx, "ly": ly, "lines": label_lines, "half_w": half_w})
            cond_txt = e.get("cond", "")
            label_txt = e.get("label", "")
            full = (cond_txt + "｜" + label_txt) if has_cond else label_txt
            label_slots.append({
                "lx": lx, "ly": ly, "lines": label_lines, "cond": has_cond,
                "full": full, "half_w": half_w, "span": span,
                "pts": pts, "nat_lx": (x1 + x2) / 2,
            })
    # 标签去冲突（纵向布局）：跨层边标签归其走廊，同走廊 x 重叠时纵向堆叠；
    # 同层边标签在横向巷道，纵向 13px 推让。
    by_y: dict[tuple, list[dict]] = {}
    by_x: dict[float, list[dict]] = {}
    for slot in label_slots:
        if slot["span"] != 0:
            near = min(corridor_centers,
                       key=lambda v: abs(v - slot["ly"]), default=None)
            if near is not None and abs(near - slot["ly"]) <= row_gap:
                key = ("C", round(near, 1))
            else:
                key = ("Y", round(slot["ly"], 1))
            by_y.setdefault(key, []).append(slot)
        else:
            near = min(lane_centers, key=lambda v: abs(v - slot["lx"]),
                       default=None)
            if near is not None and abs(near - slot["lx"]) <= col_gap:
                key = ("L", round(near, 1))
            else:
                key = ("X", round(slot["lx"], 1))
            by_x.setdefault(key, []).append(slot)
    max_label_bottom = 0.0
    max_label_right = 0.0
    for key, slots_ in by_y.items():
        if key[0] == "C":
            cc = key[1]
            y_cursor_lbl = cc - sum(11.0 * (len(x["lines"]) - 1) + 10.0 + 3.0
                                    for x in slots_) / 2.0
            slots_.sort(key=lambda s: s["lx"])
            for slot in slots_:
                slot["ly"] = y_cursor_lbl + 8.0
                y_cursor_lbl += 11.0 * (len(slot["lines"]) - 1) + 10.0 + 3.0
        for slot in slots_:
            max_label_bottom = max(max_label_bottom,
                                   slot["ly"] + 11.0 * (len(slot["lines"]) - 1) + 2.0)
            max_label_right = max(max_label_right,
                                  slot["lx"] + slot["half_w"])
    for slots_ in by_x.values():
        slots_.sort(key=lambda s: s["ly"])
        prev_bottom = None
        for slot in slots_:
            if prev_bottom is not None and slot["ly"] - prev_bottom < 16.0:
                new_ly = prev_bottom + 16.0
                if _block_free(slot["lx"], new_ly, len(slot["lines"]),
                               slot["half_w"]):
                    slot["ly"] = new_ly
            prev_bottom = slot["ly"] + 11.0 * (len(slot["lines"]) - 1) + 2.0
            max_label_bottom = max(max_label_bottom, prev_bottom + 2.0)
            max_label_right = max(max_label_right, slot["lx"] + slot["half_w"])
    # 全局去冲突：不同走廊/巷道分组的标签块若二维相交，沿本曲线在最近空位
    # 重新落位（x 仍从自身曲线反解），再做一次贴曲线 snap——标签始终贴自己的边。
    def _slot_conflicts(slot, x, y, others):
        top, bot = y - 8.0, y + 11.0 * (len(slot["lines"]) - 1) + 2.0
        hw = slot["half_w"]
        if not _block_free(x, y, len(slot["lines"]), hw):
            return True
        for o in others:
            otop = o["ly"] - 8.0
            obot = o["ly"] + 11.0 * (len(o["lines"]) - 1) + 2.0
            # 与相邻标签块至少留 3px 间隙（横向同样按 3px 计），与纵向块距一致
            if (x + hw + 3.0 > o["lx"] - o["half_w"]
                    and x - hw - 3.0 < o["lx"] + o["half_w"]):
                if not (top >= obot + 3.0 or otop >= bot + 3.0):
                    return True
        return False

    label_slots.sort(key=lambda s: (s["ly"], s["lx"]))
    for _ in range(4):
        moved = False
        for idx, slot in enumerate(label_slots):
            t = _t_at(slot["ly"], 1, slot["pts"])
            if t is not None:
                slot["lx"] = _bezier(*slot["pts"], t)[0]
            if not _slot_conflicts(slot, slot["lx"], slot["ly"], label_slots[:idx]):
                continue
            best = None
            for step in range(1, 90):
                for cand in (slot["ly"] + 16.0 * step, slot["ly"] - 16.0 * step):
                    t2 = _t_at(cand, 1, slot["pts"])
                    x2 = _bezier(*slot["pts"], t2)[0] if t2 is not None else slot["lx"]
                    if not _slot_conflicts(slot, x2, cand, label_slots[:idx]):
                        best = (x2, cand)
                        break
                if best:
                    break
            if best:
                slot["lx"], slot["ly"] = best
                moved = True
        label_slots.sort(key=lambda s: (s["ly"], s["lx"]))
        if not moved:
            break
    # 最终 snap：标签 ly 确定后，lx 从本条边路径上 y 反解——标签始终贴自己曲线。
    for slot in label_slots:
        t = _t_at(slot["ly"], 1, slot["pts"])
        if t is not None:
            px, _py = _bezier(*slot["pts"], t)
            slot["lx"] = px
    # snap 后终检：极窄间隙里仍可能与相邻标签差 <3px，固定 x 做最后一次纵向
    # 微移（只动 y，不再改变贴线的 x），保证整块文字互不重叠。
    for _ in range(3):
        moved_final = False
        label_slots.sort(key=lambda s: (s["ly"], s["lx"]))
        for idx, slot in enumerate(label_slots):
            if not _slot_conflicts(slot, slot["lx"], slot["ly"], label_slots[:idx]):
                continue
            for step in range(1, 45):
                for cand in (slot["ly"] + 4.0 * step, slot["ly"] - 4.0 * step):
                    if not _slot_conflicts(slot, slot["lx"], cand,
                                           label_slots[:idx]):
                        slot["ly"] = cand
                        moved_final = True
                        break
                if moved_final:
                    break
        if not moved_final:
            break
    height = max(height, max_label_bottom + _PAD)
    width = max(width, max_label_right + _PAD)
    # 四向画布：全部最终可见内容（标签块/节点/字段/容器/路径含弧绕）落在画布内。
    min_x, max_x = 0.0, width
    min_y, max_y = 0.0, height
    for pts in edge_pts_all:
        for px, py in pts:
            min_x = min(min_x, px)
            max_x = max(max_x, px)
            min_y = min(min_y, py)
            max_y = max(max_y, py)
    for slot in label_slots:
        min_x = min(min_x, slot["lx"] - slot["half_w"])
        max_x = max(max_x, slot["lx"] + slot["half_w"])
        min_y = min(min_y, slot["ly"] - 8.0)
        max_y = max(max_y, slot["ly"] + 11.0 * (len(slot["lines"]) - 1) + 2.0)
    for nid, g in node_geom.items():
        min_x = min(min_x, g["x"])
        max_x = max(max_x, g["x"] + g["w"])
        min_y = min(min_y, g["y"])
        max_y = max(max_y, g["y"] + g["h"])
    vb_x, vb_y = min_x - _PAD, min_y - _PAD
    vb_w = (max_x - min_x) + 2 * _PAD
    vb_h = (max_y - min_y) + 2 * _PAD
    parts = [
        f'<svg class="dg" viewBox="{vb_x:g} {vb_y:g} {vb_w:g} {vb_h:g}" '
        f'width="{vb_w:g}" height="{vb_h:g}" role="img" '
        'xmlns="http://www.w3.org/2000/svg">',
        "<defs>"
        f'<marker id="dg-arr-flow-{uid}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" class="dg-arr-flow"/></marker>'
        f'<marker id="dg-arr-weight-{uid}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" class="dg-arr-weight"/></marker>'
        f'<marker id="dg-arr-cache-{uid}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" class="dg-arr-cache"/></marker>'
        f'<marker id="dg-arr-control-{uid}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" class="dg-arr-control"/></marker>'
        "</defs>",
    ]
    # 容器边界先画（最底层），子容器在父容器之后画（嵌套清晰）。
    for fid in sorted(frame_box, key=lambda k: (frame_box[k][2] - frame_box[k][0])
                      * (frame_box[k][3] - frame_box[k][1]), reverse=True):
        f = frame_defs[fid]
        x0, y0, x1, y1 = frame_box[fid]
        sub_lines = _wrap(f.get("sub", ""), 42)[:2]
        fparts = [f'<g class="dg-frame {html.escape(str(f.get("style", "group")))}">',
                  f'<rect x="{x0:g}" y="{y0:g}" width="{x1 - x0:g}" '
                  f'height="{y1 - y0:g}" rx="10"/>',
                  f'<text class="flabel" x="{x0 + 12:g}" y="{y0 + 17:g}">'
                  f'{html.escape(f.get("label", ""))}</text>']
        for k, line in enumerate(sub_lines):
            fparts.append(f'<text class="fsub" x="{x0 + 12:g}" '
                          f'y="{y0 + 31 + k * 12:g}">{html.escape(line)}</text>')
        fparts.append("</g>")
        parts.extend(fparts)
    parts.extend(edge_parts)
    for slot in label_slots:
        spans = ""
        for k, line in enumerate(slot["lines"]):
            cls = ' class="elabel-cond"' if slot["cond"] and k == 0 else ""
            spans += (f'<tspan{cls} x="{slot["lx"]:g}" '
                      f'dy="{0 if k == 0 else 11:g}">{html.escape(line)}</tspan>')
        parts.append(
            f'<text class="elabel" x="{slot["lx"]:g}" y="{slot["ly"]:g}" '
            f'text-anchor="middle">{spans}<title>{html.escape(slot["full"])}</title></text>')
    for n in nodes:
        nid = n["id"]
        if nid not in node_geom:
            continue
        g = node_geom[nid]
        if g["shape"] == "frame":
            continue
        title = f'<title>{html.escape(n.get("title", n["label"]))}</title>'
        href = f'<a href="{html.escape(n["href"])}">' if n.get("href") else ""
        href_end = "</a>" if href else ""
        if n["kind"] == "field":
            w, h = g["w"], g["h"]
            body = ""
            for k, line in enumerate(lines_by_node[nid]):
                body += (f'<text class="flabel" x="{40:g}" y="{15 + k * 12:g}">'
                         f'{html.escape(line)}</text>')
            for port, off in sorted(g.get("ports", {}).items(), key=lambda kv: kv[1]):
                body += (f'<circle class="fport" cx="0" cy="{off:g}" r="4"/>'
                         f'<text class="fportlabel" x="8" y="{off + 3:g}">'
                         f'{html.escape(_CACHE_PORT_LABELS.get(port, port))}</text>')
            node = (f'{href}<g class="dg-field" transform="translate({g["x"]:g},{g["y"]:g})">'
                    f'<rect width="{w:g}" height="{h:g}" rx="6"/>{title}{body}</g>{href_end}')
        elif n["kind"] in ("op", "nav", "navlane", "carrier", "iface",
                           "control", "controlopt"):
            w, h = g["w"], g["h"]
            body = ""
            lines = lines_by_node[nid]
            for k, line in enumerate(lines):
                body += (f'<text class="nlabel" x="{w / 2:g}" y="{14 + k * 12:g}" '
                         f'text-anchor="middle">{html.escape(line)}</text>')
            sub_lines = _node_sub_lines(n)
            for k, sline in enumerate(sub_lines):
                body += (f'<text class="nsub" x="{w / 2:g}" '
                         f'y="{h - 6 - 13 * (len(sub_lines) - 1 - k):g}" '
                         f'text-anchor="middle">{html.escape(sline)}</text>')
            cls = f'dg-node {n["kind"]}'
            if n["kind"] == "controlopt":
                cls += " active" if n.get("active") else " inactive"
            fig_attr = ""
            if n.get("href", "").startswith("#op-") and "cache-" not in uid:
                fig_attr = f' id="figop-{n["href"][4:]}"'
            node = (f'{href}<g class="{cls}"{fig_attr} '
                    f'transform="translate({g["x"]:g},{g["y"]:g})">'
                    f'<rect width="{w:g}" height="{h:g}" rx="7"/>{title}{body}</g>'
                    f'{href_end}')
        elif g["shape"] == "io":
            node = (f'{href}<g class="dg-port io" transform="translate({g["x"]:g},{g["y"]:g})">'
                    f'<rect width="{g["w"]:g}" height="24" rx="12"/>{title}'
                    f'<text class="portlabel" x="{g["w"] / 2:g}" y="16" '
                    f'text-anchor="middle">{html.escape(n["label"])}</text></g>{href_end}')
        else:
            node = (f'{href}<g class="dg-dot {n["kind"]}" '
                    f'transform="translate({g["cx"]:g},{g["cy"]:g})">'
                    f'<circle r="5"/>{title}</g>{href_end}')
        parts.append(node)
    parts.append("</svg>")
    return "".join(parts)

_LEGEND_ITEMS = (
    ("opbox", "计算算子框（源码名 + 实际计算类型）"),
    ("navbox", "导航框（总览模块，虚线——非计算算子）"),
    ("frame", "容器边界（真实包含：Attention ⊃ CSA ⊃ Indexer；非计算算子）"),
    ("field", "持久状态存储字段（past_key_values.layers[layer_idx]；左缘读前/写后/视图端口）"),
    ("dot", "端口圆点（tensor/权重端点；源码名+shape 在箭头上）"),
    ("flow", "数据流（激活/索引/掩码张量，箭头=producer→consumer）"),
    ("weight", "权重/参数引用（虚线；不改变数据流方向）"),
    ("cache", "缓存读写（点线；端点落在载体字段的读前/写后端口）"),
    ("control", "控制选择/实例化（虚点线；非 tensor 流）"),
    ("cond", "条件更新（如仅 N_win>0 闭窗时读写 overlap）"),
)


def legend_html() -> str:
    items = "".join(
        f'<span><i class="dg-swatch {cls}"></i>{html.escape(text)}</span>'
        for cls, text in _LEGEND_ITEMS
    )
    return f'<div class="dg-legend">{items}</div>'


def diagram_css() -> str:
    """图示样式（内嵌进页面 <style>，随页面单文件离线）。

    配色为图示独立固定高对比成对方案：节点/端口底色固定浅色 → 框内文字固定深色，
    两个主题下名称/类型/副标题均清晰；边标签随页面主题成对切换（浅底深字/深底浅字，
    均带同色描边光晕），不再依赖 --ink/--muted 直接叠在固定浅色节点上。
    """
    return """
.diagram-scroll { overflow-x: auto; padding: 4px 0; }
svg.dg { display: block; min-width: 100%; height: auto; font-family: inherit; }
.dg-node.op rect { fill: #fcfdfe; stroke: #1f4e79; stroke-width: 1.4; }
.dg-node.nav rect { fill: #eef3fa; stroke: #1f4e79; stroke-width: 1.6; stroke-dasharray: 7 4; }
.dg-node.carrier rect { fill: #e8f0e8; stroke: #2d7a3a; stroke-width: 1.8; stroke-dasharray: 9 4; }
.dg-node.carrier .nlabel { fill: #2d7a3a; }
.dg-node.carrier .nsub { fill: #2d7a3a; }
.dg-node.boundary rect { fill: #e9f7f0; stroke: #1e6d4f; stroke-width: 1.4; }
.dg-node.weight rect { fill: #fdf6ea; stroke: #b08d57; stroke-width: 1.2; stroke-dasharray: 4 3; }
.dg-node.cache rect { fill: #eaf1fa; stroke: #2b5d8f; stroke-width: 1.4; }
.dg-node.port rect { fill: #ffffff; stroke: #5f6b7a; stroke-width: 1.1; stroke-dasharray: 3 2; }
.dg-node.downstream rect { fill: #f3f0fa; stroke: #6d3f8f; stroke-width: 1.1; }
/* 框内文字固定深色（底色固定浅色，两个主题都可读）；副标题显式覆盖 */
.dg-node text { fill: #1f2933; }
.dg-node .nlabel { fill: #1f2933; font-size: 12px; font-weight: 600; }
.dg-node .nsub { fill: #4a5568; font-size: 10px; }
/* 端口：tensor/权重/缓存状态为小圆点（非计算框）；hidden_states 为输入/输出标识 */
.dg-dot circle { fill: #5f6b7a; stroke: #ffffff; stroke-width: 1.2; }
.dg-dot.boundary circle { fill: #1e6d4f; }
.dg-dot.weight circle { fill: #b08d57; }
.dg-dot.cache circle { fill: #2b5d8f; }
.dg-dot.port circle { fill: #5f6b7a; }
.dg-dot.downstream circle { fill: #6d3f8f; }
.dg-port.io rect { fill: #e9f7f0; stroke: #1e6d4f; stroke-width: 1.3; stroke-dasharray: 5 3; }
.dg-port .portlabel { fill: #1f2933; font-size: 11px; font-weight: 700; }
.dg-edge { fill: none; stroke-width: 1.5; }
.dg-edge.flow { stroke: #1f4e79; }
.dg-edge.weight { stroke: #b08d57; stroke-dasharray: 5 4; }
.dg-edge.cache { stroke: #2b5d8f; stroke-dasharray: 2 3; }
.dg-edge.cond { stroke-dasharray: 7 4; }
.dg-arr-flow { fill: #1f4e79; }
.dg-arr-weight { fill: #b08d57; }
.dg-arr-cache { fill: #2b5d8f; }
/* 容器：真实包含边界（Attention ⊃ CSA ⊃ Indexer）与持久状态存储——只表达
   包含/归类，不是计算算子，不参与数据流方向 */
.dg-frame rect { fill: none; stroke: #6b7a8c; stroke-width: 1.2; stroke-dasharray: 10 6; }
.dg-frame.attn rect { stroke: #1f4e79; stroke-width: 1.7; }
.dg-frame.csa rect { stroke: #7a3b0e; stroke-width: 1.5; }
.dg-frame.indexer rect { stroke: #6d3f8f; stroke-width: 1.3; }
.dg-frame.store rect { fill: #eef6ee; stroke: #2d7a3a; stroke-width: 1.6; stroke-dasharray: 9 4; }
.dg-frame .flabel { font-size: 12px; font-weight: 700; fill: #1f2933; }
.dg-frame .fsub { font-size: 9.5px; fill: #4a5568; }
.dg-frame.store .flabel, .dg-frame.store .fsub { fill: #2d7a3a; }
/* 载体字段行：字段属于存储框内部；左缘 pre/post/view 时态端口是读写线真实端点 */
.dg-field rect { fill: #f4f9f4; stroke: #2d7a3a; stroke-width: 1.1; }
.dg-field .flabel { font-size: 10.5px; fill: #1f2933; font-weight: 600; }
.dg-field .fport { fill: #2d7a3a; stroke: #ffffff; stroke-width: 1; }
.dg-field .fportlabel { font-size: 7.5px; fill: #2d7a3a; }
/* 配置选择：控制线（虚点线）与实际 tensor/实例线区分 */
.dg-node.control rect { fill: #f6f0fa; stroke: #6d3f8f; stroke-width: 1.2; }
.dg-node.controlopt rect { fill: #fbf7fd; stroke: #6d3f8f; stroke-width: 1.3; }
.dg-node.controlopt.active rect { fill: #eaf4ec; stroke: #2d7a3a; stroke-width: 2.4; }
.dg-node.controlopt.inactive rect { stroke-dasharray: 5 4; }
.dg-node.iface rect { fill: #eef3fa; stroke: #1f4e79; stroke-width: 1.2; stroke-dasharray: 6 3; }
.dg-edge.control { stroke: #6d3f8f; stroke-dasharray: 9 4; stroke-width: 1.3; }
.dg-arr-control { fill: #6d3f8f; }
/* 边标签随主题成对：浅色主题深字浅晕 / 深色主题浅字深晕，均不依赖节点底色 */
.elabel { font-size: 10px; fill: #33404d; paint-order: stroke; stroke: rgba(246,247,249,0.92); stroke-width: 3px; }
.elabel-cond { fill: #8f2b4d; font-weight: 700; }
.dg-legend { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 12px; color: var(--muted); margin: 6px 0 2px; }
.dg-legend span { display: inline-flex; align-items: center; gap: 6px; }
.dg-swatch { display: inline-block; width: 26px; height: 0; border-top-width: 2.5px; }
.dg-swatch.opbox { border-top-style: solid; border-top-color: #1f4e79; }
.dg-swatch.navbox { border-top-style: dashed; border-top-color: #1f4e79; }
.dg-swatch.dot { border-top-style: dotted; border-top-color: #5f6b7a; }
.dg-swatch.flow { border-top-style: solid; border-top-color: #1f4e79; }
.dg-swatch.weight { border-top-style: dashed; border-top-color: #b08d57; }
.dg-swatch.cache { border-top-style: dotted; border-top-color: #2b5d8f; }
.dg-swatch.control { border-top-style: dashed; border-top-color: #6d3f8f; }
.dg-swatch.frame { border-top-style: dashed; border-top-color: #6b7a8c; }
.dg-swatch.field { border-top-style: solid; border-top-color: #2d7a3a; }
.dg-swatch.cond { border-top-style: dashed; border-top-color: #8f2b4d; }
.dg-note { font-size: 12px; color: var(--muted); margin: 6px 0 0; }
@media (prefers-color-scheme: dark) {
  /* 箭头/marker/图例随主题成对：固定深蓝在暗底对比不足（~1.87），
     暗主题换浅色画线（对暗卡对比 ≥4.5），浅主题保持原深蓝（≥8） */
  .dg-edge.flow { stroke: #7fa8d9; }
  .dg-arr-flow { fill: #7fa8d9; }
  .dg-edge.weight { stroke: #d4b183; }
  .dg-arr-weight { fill: #d4b183; }
  .dg-edge.cache { stroke: #9fc2e8; }
  .dg-arr-cache { fill: #9fc2e8; }
  .dg-edge.control { stroke: #c39be0; }
  .dg-arr-control { fill: #c39be0; }
  .dg-frame .flabel { fill: #e4e9ee; }
  .dg-frame .fsub { fill: #b9c4cf; }
  /* 存储载体框成对配色：暗主题深底(+深绿描边)配浅绿字，不再浅底浅字（1.63） */
  .dg-frame.store rect { fill: #17301f; stroke: #6cc283; }
  .dg-frame.store .flabel, .dg-frame.store .fsub { fill: #c3eccd; }
  /* 非 store 透明容器在真实暗底上改用成对浅色描边（原深蓝/棕/紫对 #1d2126 不足） */
  .dg-frame rect { stroke: #9aa7b4; }
  .dg-frame.attn rect { stroke: #7fa8d9; }
  .dg-frame.csa rect { stroke: #d4b183; }
  .dg-frame.indexer rect { stroke: #c39be0; }
  .dg-swatch.flow { border-top-color: #7fa8d9; }
  .dg-swatch.weight { border-top-color: #d4b183; }
  .dg-swatch.cache { border-top-color: #9fc2e8; }
  .dg-swatch.control { border-top-color: #c39be0; }
  .dg-swatch.opbox { border-top-color: #7fa8d9; }
  .dg-swatch.navbox { border-top-color: #7fa8d9; }
  .dg-swatch.frame { border-top-color: #b9c4cf; }
  .dg-swatch.field { border-top-color: #8fd0a0; }
  .dg-swatch.dot { border-top-color: #9aa7b4; }
  .elabel { fill: #d9e2ea; stroke: rgba(20,23,27,0.92); }
  .elabel-cond { fill: #ff9a9a; }
}
"""
