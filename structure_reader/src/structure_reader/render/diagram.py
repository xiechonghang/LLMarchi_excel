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


def build_module_graph(phase: dict, group: dict) -> dict:
    """模块算子图：节点=组内算子+外部端口，边=真实张量引用。

    - 内部边：producer op → consumer op（组内；仅由输出侧生成一次，不做输入/输出
      双循环重复）；
    - 外部输入端口：被组内消费、但 producer 在组外或无 producer 的张量
      （boundary/weight/cache 按其类型着色，其余为普通端口）；
    - 输出端口：组内产出、但无组内下游的张量（缓存回写含条件标注）；
    - 权重引用为虚线边（引用，不改变数据流方向；组内产出的权重切片由
      producer→consumer 直连表达，不重复端口）。
    边最终按 (src,dst,kind,张量,条件,标签) 去重；不同张量各自成边不丢。
    """
    tensors = _tensor_map(phase)
    producers = _producers(phase)
    consumers = _consumers(phase)
    weight_consumers = _weight_consumers(phase)
    ops = group["ops"]
    op_ids = {o["id"] for o in ops}

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for op in ops:
        nodes[op["id"]] = _op_node(phase, op)

    for op in ops:
        oid = op["id"]
        for tid in op.get("inputs", []):
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

    return {"id": group["id"], "title": group["title"], "nodes": list(nodes.values()),
            "edges": _dedup_edges(edges), "phase": phase["id"]}


def build_overview_graph(phase: dict, rules=None) -> dict:
    """Decoder 总览：模块框 + 边界/缓存节点；聚合边逐条由真实张量支撑。

    每条聚合边 (src, dst) 都能从文档找到至少一个张量：其 producer 属 src
    （或 src 为边界/缓存节点）且 consumer 属 dst；组内连接留给模块详图。
    """
    tensors = _tensor_map(phase)
    producers = _producers(phase)
    consumers = _consumers(phase)
    groups = module_groups(phase, rules)
    op_group: dict[str, str] = {}
    for g in groups:
        for op in g["ops"]:
            op_group[op["id"]] = g["id"]

    nodes: dict[str, dict] = {}
    for g in groups:
        num = section_number(g["id"], phase["id"])
        label = f"{num} {g['title']}" if num else g["title"]
        nodes[f"m:{g['id']}"] = {
            "id": f"m:{g['id']}", "kind": "nav", "label": label,
            "sub": f"导航 · {len(g['ops'])} 算子",
            "href": f"#mod-{phase['id']}-{g['id']}",
            "title": "，".join(op["id"] for op in g["ops"]),
        }
    for tid in ("hidden_streams_in", "input_ids", "position_ids", "hidden_streams_out"):
        if tid in tensors:
            nodes[f"t:{tid}"] = {
                "id": f"t:{tid}", "kind": "boundary",
                "label": tensor_source_label(tid) or tensors[tid]["name"],
                "sub": tid, "href": f"#t-{phase['id']}-{tid}", "title": tid,
            }
    cache_families_present = {
        _cache_family(tid) for tid, t in tensors.items()
        if (t.get("kind") == "cache" and _cache_family(tid)) or tid == "kv_full"
    }
    for family in ("main", "csa", "indexer"):
        if family in cache_families_present:
            num = section_number(f"cache:{family}", phase["id"])
            nodes[f"c:{family}"] = {
                "id": f"c:{family}", "kind": "navlane",
                "label": f"{num} {_CACHE_FAMILY_TITLES[family]}" if num
                else _CACHE_FAMILY_TITLES[family],
                "sub": "导航 · 调用前→读视图→调用后",
                "href": f"#cache-{phase['id']}-{family}",
                "title": _CACHE_FAMILY_TITLES[family],
            }

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
                # 层输出/末端张量：仅边界输出画到边界节点（其余末端留给模块详图）
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
    # 缓存写回：产出缓存状态的模块 → 缓存族节点
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
    edges: list[dict] = []
    for (src, dst), entry in sorted(agg.items()):
        if src not in nodes or dst not in nodes:
            continue
        is_cache = entry["kinds"] == {"cache"}
        names = "，".join(entry["tensors"][:3]) + ("…" if len(entry["tensors"]) > 3 else "")
        edges.append(_edge(
            src, dst, "cache" if is_cache else "flow",
            (f"读写 {len(entry['tensors'])} 状态" if is_cache
             else f"{len(entry['tensors'])} 张量：{names}"),
            "", "", "，".join(entry["tensors"])))

    return {"id": "overview", "title": "Decoder 总览", "nodes": list(nodes.values()),
            "edges": edges, "phase": phase["id"]}


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
            family, family), "ops": inner_ops})
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
    """最长路径分层（op/nav 节点）+ 同层文档序横向排布；返回 id → (层, 层内序)。

    纵向渲染约定：层 = 自上而下的 rank，层内序 = 横向位。
    其余节点：仅作为边源（边界/权重/调用前缓存）放最上层（层 -1）；
    作为边终点（输出端口/调用后回写/下游 chip）放最下层（层 max+1）；
    孤立节点放最上层——主阅读方向自上而下，历史→本次读→回写顺向。
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
    idxs: dict[int, int] = {}
    for n in layered:
        c = layer[n["id"]]
        i = idxs.get(c, 0)
        idxs[c] = i + 1
        positions[n["id"]] = (c, i)
    top_idx = 0
    bottom_idx = 0
    for n in others:
        nid = n["id"]
        if nid in dsts:
            # 被写入（调用后回写/读视图兼保存/下游 chip）：最下层，
            # 箭头自上而下到达；其向下游的读边同级横向展开
            positions[nid] = (max_layer + 1, bottom_idx)
            bottom_idx += 1
        else:
            # 仅作为源（调用前状态/权重/边界输入）或孤立：最上层
            positions[nid] = (-1, top_idx)
            top_idx += 1
    return positions


_NODE_W = 172.0
_NODE_H_MIN = 44.0
# 最小列间隙/行走廊；实际值按本图最宽标签块自适应加宽（render_svg）
_COL_GAP_MIN = 104.0
_ROW_GAP_MIN = 22.0
_PAD = 16.0


def _node_lines(node: dict) -> list[str]:
    """框内主行（源码名）：按需换行、无行数上限——用户要求框内源码名完整可见，
    节点高度随行数自适应；不为塞进固定行数而截断真实信息。"""
    return _wrap(node["label"], 19, max_lines=None)


def _node_sub_lines(node: dict) -> list[str]:
    """计算类型副行：完整显示不截断（最多 2 行，长者换行）。"""
    sub = node.get("sub", "")
    return _wrap(sub, 30)[:2] if sub else []


def _node_height(node: dict, lines: list[str]) -> float:
    """节点高度容纳实际文字行数与副标题行数，避免重叠/截断。"""
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


# ---------------------------------------------------------------------------
# 章节编号（展示元数据）：体现真实模块包含关系。
# 源码依据（modeling_deepseek_v4.py，536ecc00…）：
#   DeepseekV4Attention :794-795 self.compressor = COMPRESSOR_CLASSES[layer_type](config)
#   DeepseekV4CSACompressor :618 self.indexer = DeepseekV4Indexer(config)；:690 调用
#   DeepseekV4DecoderLayer :1154-1159 self_attn/mlp/attn_hc/ffn_hc/双 RMSNorm
# 编号只属展示层，不改变 reading_document 的算子/边/shape 事实。
# ---------------------------------------------------------------------------

_PHASE_NUMBERS = {"prefill": "1", "decode": "2"}

# (编号模板, group_id 或 "cache:<family>" 或 None=纯父章节, 标题)
_SECTION_TREE: tuple[tuple[str, str | None, str], ...] = (
    ("{p}.1", None, "注意力前段（attn_hc mHC 与 input_layernorm）"),
    ("{p}.1.1", "mhc_attn", "mHC 权重与流坍缩（attn_hc）"),
    ("{p}.1.2", "norm_attn", "RMSNorm（input_layernorm）"),
    ("{p}.2", None, "Attention（self_attn / DeepseekV4Attention）"),
    ("{p}.2.1", "attn_q", "Q 支路（低秩 + 部分 RoPE）"),
    ("{p}.2.2", "attn_kv", "共享 KV 支路（K=V）"),
    ("{p}.2.3", "attn_cache", "主 KV 缓存更新"),
    ("{p}.2.4", "attn_csa", "CSA 压缩器（self.compressor，:794-795）"),
    ("{p}.2.4.1", "attn_indexer", "Lightning Indexer（self.indexer，:618/:690）"),
    ("{p}.2.5", "attn_mask", "模型级 mask 与压缩条目扩展"),
    ("{p}.2.6", "attn_core", "eager 注意力核心"),
    ("{p}.2.7", "attn_out", "逆 RoPE 与分组输出投影"),
    ("{p}.3", "res_attn", "mHC 残差混合（attn 段）"),
    ("{p}.4", None, "MoE 前段（ffn_hc mHC 与 post_attention_layernorm）"),
    ("{p}.4.1", "mhc_ffn", "mHC 权重与流坍缩（ffn_hc）"),
    ("{p}.4.2", "norm_ffn", "RMSNorm（post_attention_layernorm）"),
    ("{p}.5", None, "MoE（DeepseekV4SparseMoeBlock）"),
    ("{p}.5.1", "moe_route", "输入展平与 TopK 路由"),
    ("{p}.5.2", "moe_expert", "专家模板（一份；不复制 E 份）"),
    ("{p}.5.3", "moe_shared", "共享专家（DeepseekV4MLP）"),
    ("{p}.5.4", "moe_merge", "汇合（routed + shared）"),
    ("{p}.6", "res_ffn", "mHC 残差混合（ffn 段）"),
    ("{p}.7", None, "缓存附图（调用前 → 本次读 → 调用后回写）"),
    ("{p}.7.1", "cache:main", "主 KV 缓存（关联 {p}.2.3）"),
    ("{p}.7.2", "cache:csa", "CSA 压缩器缓存（关联 {p}.2.4）"),
    ("{p}.7.3", "cache:indexer", "Indexer 缓存（关联 {p}.2.4.1）"),
)


def section_entries(phase_id: str) -> list[dict]:
    """阶段章节树：[{number, gid, title, anchor, parent, parent_number}]（有序）。"""
    p = _PHASE_NUMBERS.get(phase_id, "1")
    entries = []
    for tpl, gid, title in _SECTION_TREE:
        number = tpl.format(p=p)
        parts = number.split(".")
        parent_number = ".".join(parts[:-1])
        anchor = (f"mod-{phase_id}-{gid}" if gid and not gid.startswith("cache:")
                  else f"cache-{phase_id}-{gid.split(':', 1)[1]}"
                  if gid else f"sec-{phase_id}-{number.replace('.', '-')}")
        title = title.replace("{p}.", f"{p}.")
        entries.append({
            "number": number, "gid": gid, "title": title,
            "anchor": anchor, "parent": None if "." not in parent_number[2:]
            and parent_number.count(".") < 1 else parent_number,
        })
    by_num = {e["number"]: e for e in entries}
    for e in entries:
        e["parent_number"] = e["number"].rsplit(".", 1)[0] \
            if "." in e["number"] else None
        pe = by_num.get(e["parent_number"] or "")
        e["parent"] = pe
    return entries


def section_number(group_key: str, phase_id: str) -> str | None:
    """group_id 或 cache:<family> 的章节编号；总览导航与标题共用同一编号。"""
    for tpl, gid, _title in _SECTION_TREE:
        if gid == group_key:
            return tpl.format(p=_PHASE_NUMBERS.get(phase_id, "1"))
    return None


def section_title(group_key: str, phase_id: str) -> str | None:
    for tpl, gid, title in _SECTION_TREE:
        if gid == group_key:
            return title.replace("{p}.", f"{_PHASE_NUMBERS.get(phase_id, '1')}.")
    return None


def _label_block_box(full_label: str) -> tuple[float, float, int]:
    """标签块（宽度 px、行数）；宽度按渲染器自身度量（CJK 10px/其他 5.5px）+ 8px 留白。"""
    lines = _wrap_edge_label(full_label)
    if not lines:
        return 0.0, 0, 0
    width = max(sum(10.0 if ord(ch) > 0x2E80 else 5.5 for ch in line)
                for line in lines) + 8.0
    return width, len(lines), 0


def render_svg(graph: dict) -> str:
    """图结构 → 内嵌 SVG 字符串（节点/边可点击联查原详情锚点）。

    展示层修正：marker id 按图唯一（uid=phase+graph id，避免多图文档重名）；
    节点高度容纳文字行数；边标签按实际行数占用排布（同列块间最小间距 13px）；
    标签块被节点矩形阻挡时，在**所属边的真实路径上**反解参数移动到列间隙/
    行走廊的空闲位置——整块文字不与任何节点矩形相交（无 anchor-only 降级，
    宽标签不再被后绘节点盖住尾部）；列间隙/行走廊按本图最宽标签块自适应加宽，
    保证空位容纳完整占用；长跨列（≥3 列）边按序号交替上下弧绕行；
    画布高度在排布后按实际占用扩展。
    """
    uid = html.escape(f"{graph.get('phase', 'p')}-{graph.get('id', 'g')}", quote=True)
    # 每图自适应留白：列间隙须容纳本图最宽标签块（否则该图加宽），行走廊同理。
    # 标签块 = 条件行（如有）+ 源码名/shape 换行（最多 3 行）——条件与名称/维度
    # 都必须可见，不再用 cond 替换 tensor 信息。
    def _edge_label_lines(e: dict) -> list[str]:
        cond = e.get("cond", "")
        label = e.get("label", "")
        if cond and label:
            return [cond] + _wrap_edge_label(label)
        if label:
            return _wrap_edge_label(label)
        return [cond] if cond else []

    max_block_w, max_label_lines = 0.0, 0
    for e in graph["edges"]:
        lines = _edge_label_lines(e)
        if not lines:
            continue
        w = max(sum(10.0 if ord(ch) > 0x2E80 else 5.5 for ch in ln) for ln in lines) + 8.0
        max_block_w = max(max_block_w, w)
        max_label_lines = max(max_label_lines, len(lines))
    col_gap = max(_COL_GAP_MIN, max_block_w + 10.0)
    # 行走廊须容纳最高标签块（8 上沿 + 11×(行数-1) + 2 下沿）+ 5px 余量
    row_gap = max(_ROW_GAP_MIN, 8.0 + 11.0 * (max_label_lines - 1) + 2.0 + 5.0)
    col_w = _NODE_W + col_gap

    positions = _layout(graph["nodes"], graph["edges"])
    lines_by_node = {n["id"]: _node_lines(n) for n in graph["nodes"] if n["id"] in positions}
    h_by_node = {n["id"]: _node_height(n, lines_by_node[n["id"]])
                 for n in graph["nodes"] if n["id"] in positions}
    layers = [c for c, _ in positions.values()]
    idxs = [r for _, r in positions.values()]
    min_c, max_c = (min(layers), max(layers)) if layers else (0, 0)
    min_r, max_r = (min(idxs), max(idxs)) if idxs else (0, 0)
    # 纵向主阅读：层 = 自上而下 rank（y），层内序 = 横向位（x）。
    # 每 rank 高度 = 层内最高节点 + 纵向走廊；横向位距容纳节点宽与标签块。
    rank_h: dict[int, float] = {}
    for nid, (c, _) in positions.items():
        rank_h[c] = max(rank_h.get(c, 0.0), h_by_node[nid])
    rank_y: dict[int, float] = {}
    y_cursor = _PAD
    for c in range(min_c, max_c + 1):
        rank_y[c] = y_cursor
        y_cursor += rank_h.get(c, _NODE_H_MIN) + row_gap
    height = y_cursor - row_gap + _PAD
    width = (max_r - min_r + 1) * col_w + 2 * _PAD

    def x_of(r: int) -> float:
        return _PAD + (r - min_r) * col_w

    # 真实节点几何（纵向）：计算/导航框（172×h，文字横向）、IO 标识（172×24）、
    # 端口圆点（r=5，圆心 (x+10, y+12)）——箭头锚点按实际形状边界求：
    # 跨层边 = 源底边 → 目标顶边；同层边 = 横向；方向仍由同 JSON 引用决定。
    node_geom: dict[str, dict] = {}
    for n in graph["nodes"]:
        if n["id"] not in positions:
            continue
        c, r = positions[n["id"]]
        x = x_of(r)
        y = rank_y[c]
        if n["kind"] in ("op", "nav", "navlane"):
            h = h_by_node[n["id"]]
            node_geom[n["id"]] = {
                "top": (x + _NODE_W / 2, y), "bottom": (x + _NODE_W / 2, y + h),
                "left": (x, y + h / 2), "right": (x + _NODE_W, y + h / 2),
                "cx": x + _NODE_W / 2, "cy": y + h / 2, "shape": "box",
            }
        elif n["id"] in ("t:hidden_streams_in", "t:hidden_streams_out"):
            node_geom[n["id"]] = {
                "top": (x + _NODE_W / 2, y), "bottom": (x + _NODE_W / 2, y + 24),
                "left": (x, y + 12), "right": (x + _NODE_W, y + 12),
                "cx": x + _NODE_W / 2, "cy": y + 12, "shape": "io",
            }
        else:
            cx, cy = x + 10.0, y + 12.0
            node_geom[n["id"]] = {
                "top": (cx, cy - 6.0), "bottom": (cx, cy + 6.0),
                "left": (cx - 6.0, cy), "right": (cx + 6.0, cy),
                "cx": cx, "cy": cy, "shape": "dot",
            }

    def anchor(nid: str, side: str) -> tuple[float, float]:
        return node_geom[nid][side]

    # 纵向走廊中心（跨层边标签空位）/ 横向巷道中心（同层边标签空位）
    corridor_centers: list[float] = []
    for c in range(min_c, max_c):
        if c in rank_h and c + 1 in rank_h or (c in rank_y and c + 1 in rank_y):
            corridor_centers.append(rank_y[c] + rank_h.get(c, _NODE_H_MIN) + row_gap / 2)
    if min_c <= max_c:
        corridor_centers.append(rank_y[min_c] / 2)
        corridor_centers.append(rank_y[max_c] + rank_h.get(max_c, _NODE_H_MIN) + row_gap / 2)
    lane_centers: list[float] = []
    for r in range(min_r, max_r):
        lane_centers.append(x_of(r) + _NODE_W + col_gap / 2)
    if min_r <= max_r:
        lane_centers.append(x_of(min_r) / 2)
        lane_centers.append(x_of(max_r) + _NODE_W + col_gap / 2)

    # 平行边（同一对节点，含方向相反的读写对）序号与长跨列边序号（用于错位/绕行）
    pair_count: dict[tuple[str, str], int] = {}
    for e in graph["edges"]:
        if e["src"] in positions and e["dst"] in positions:
            key = tuple(sorted((e["src"], e["dst"])))
            pair_count[key] = pair_count.get(key, 0) + 1
    pair_idx: dict[tuple[str, str], int] = {}

    edge_parts: list[str] = []
    edge_pts_all: list[tuple] = []
    label_slots: list[dict] = []

    # 节点矩形（标签遮挡检测）：绘制顺序为边→标签→节点，标签块不得落在矩形内
    rects = [(x_of(positions[n["id"]][1]),
              rank_y[positions[n["id"]][0]], _NODE_W, h_by_node[n["id"]])
             for n in graph["nodes"] if n["id"] in positions]

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
        """标签块（首行基线 cy，向上约 8px、向下 11×(行数-1)+2px）是否不与任何节点矩形相交。"""
        top, bot = cy - 8.0, cy + 11.0 * (n_lines - 1) + 2.0
        for rx, ry, rw, rh in rects:
            if cx + half_w > rx and cx - half_w < rx + rw and bot > ry and top < ry + rh:
                return False
        return True

    placed: list[dict] = []

    def _free_of_labels(lx, ly, n_lines, half_w):
        """与已放置标签块保持 13px 块间距：x 区间重叠时，两块的文字占用区间
        （基线 -8 上沿 / 末行 +2 下沿）须竖直间隔 ≥3px（等价于
        「下方首行基线 - 上方末行基线 ≥13px」）。避免避让本身引入标签互叠。"""
        top, bot = ly - 8.0, ly + 11.0 * (n_lines - 1) + 2.0
        for p in placed:
            ptop = p["ly"] - 8.0
            pbot = p["ly"] + 11.0 * (len(p["lines"]) - 1) + 2.0
            separated = (lx + half_w + 3.0 <= p["lx"] - p["half_w"]
                         or p["lx"] + p["half_w"] + 3.0 <= lx - half_w
                         or top >= pbot + 3.0 or ptop >= bot + 3.0)
            if not separated:
                return False
        return True

    for e in graph["edges"]:
        if e["src"] not in positions or e["dst"] not in positions:
            continue
        key = tuple(sorted((e["src"], e["dst"])))
        i = pair_idx.get(key, 0)
        pair_idx[key] = i + 1
        c1, _ = positions[e["src"]]
        c2, _ = positions[e["dst"]]
        span = c2 - c1
        # 纵向主阅读：跨层边 = 源底边 → 目标顶边（自上而下）；同层边横向连接；
        # 上行边（如缓存读）同样源→目标方向不变。
        if span > 0:
            x1, y1 = anchor(e["src"], "bottom")
            x2, y2 = anchor(e["dst"], "top")
        elif span < 0:
            x1, y1 = anchor(e["src"], "top")
            x2, y2 = anchor(e["dst"], "bottom")
        else:
            r1, r2 = positions[e["src"]][1], positions[e["dst"]][1]
            if r2 >= r1:
                x1, y1 = anchor(e["src"], "right")
                x2, y2 = anchor(e["dst"], "left")
            else:
                x1, y1 = anchor(e["src"], "left")
                x2, y2 = anchor(e["dst"], "right")
        # marker 箭头头部需留 3px：仅框形终点收缩；圆点端口已按半径偏移
        end_trim = 3.0 if node_geom[e["dst"]]["shape"] in ("box", "io") else 0.0
        # 长跨层边交替左右弧绕行；平行边按序号额外偏移控制点
        bow = 0.0
        if abs(span) >= 3:
            bow = (26.0 if i % 2 == 0 else -26.0) + (8.0 if pair_count[key] > 1 else 0.0)
        if span != 0:
            mid_y = (y1 + y2) / 2
            cy1, cy2 = y1 + (bow if span > 0 else -bow), y2 + (bow if span > 0 else -bow)
            pts = ((x1, y1), (x1 + bow * 0.0 + (x2 - x1) * 0.0 + x1, cy1),
                   (x2, cy2), (x2, y2 - (end_trim if span > 0 else -end_trim)))
            # 纵向贝塞尔：控制点保持端点 x，纵向推进
            ctrl1 = (x1, cy1)
            ctrl2 = (x2, cy2)
            endpt = (x2, y2 - (end_trim if span > 0 else -end_trim))
            pts = ((x1, y1), ctrl1, ctrl2, endpt)
            path_d = (f"M{x1:g},{y1:g} C{ctrl1[0]:g},{ctrl1[1]:g} "
                      f"{ctrl2[0]:g},{ctrl2[1]:g} {endpt[0]:g},{endpt[1]:g}")
        else:
            mid_x = (x1 + x2) / 2
            ctrl1 = (mid_x, y1)
            ctrl2 = (mid_x, y2)
            endpt = (x2 - (end_trim if x2 >= x1 else -end_trim), y2)
            pts = ((x1, y1), ctrl1, ctrl2, endpt)
            edge_pts_all.append(pts)
            path_d = (f"M{x1:g},{y1:g} C{ctrl1[0]:g},{ctrl1[1]:g} "
                      f"{ctrl2[0]:g},{ctrl2[1]:g} {endpt[0]:g},{endpt[1]:g}")
        cls = f"dg-edge {e['kind']}" + (" cond" if e.get("cond") else "")
        marker_kind = {"flow": "flow", "weight": "weight"}.get(e["kind"], "cache")
        path = (f'<path class="{cls}" d="{path_d}" '
                f'marker-end="url(#dg-arr-{marker_kind}-{uid})"/>')
        if e.get("href"):
            path = f'<a href="{html.escape(e["href"])}">{path}</a>'
        edge_parts.append(path)
        # 标签块 = 条件行（如有，条件样式）+ 源码名/shape 换行——两者都可见
        label_lines = _edge_label_lines(e)
        has_cond = bool(e.get("cond")) and bool(e.get("label"))
        if label_lines and any(ln for ln in label_lines):
            n_lines = len(label_lines)
            half_w = max(sum(10.0 if ord(ch) > 0x2E80 else 5.5 for ch in line)
                         for line in label_lines) / 2 + 4.0
            ly0 = (4.0 if n_lines == 1 else 6.0)  # 首行基线相对块中心的近似偏移
            lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - ly0
            leader_from: tuple[float, float] | None = None
            if not (_block_free(lx, ly, n_lines, half_w)
                    and _free_of_labels(lx, ly, n_lines, half_w)):
                # 有界外置避让：先取边上空位（纵向走廊/横向巷道反解），
                # 再取「巷道×走廊」网格点（该格必不在任何节点矩形内），
                # 网格点偏离所属边时以细引线连回边上的穿越点，保持对应。
                # 无 anchor-only 降级、无通用求解器。
                cands: list[tuple[float, float, tuple[float, float] | None]] = []
                nat_x, nat_y = (x1 + x2) / 2, (y1 + y2) / 2
                if span != 0:
                    for cy_ in sorted(corridor_centers, key=lambda v: abs(v - nat_y)):
                        t = _t_at(cy_, 1, pts)
                        if t is None:
                            continue
                        px_path, py_path = _bezier(*pts, t)
                        cands.append((px_path, cy_, None))
                        for lx_ in sorted(lane_centers, key=lambda v: abs(v - px_path)):
                            cands.append((lx_, cy_, (px_path, py_path)))
                else:
                    for lx_ in sorted(lane_centers, key=lambda v: abs(v - nat_x)):
                        t = _t_at(lx_, 0, pts)
                        if t is None:
                            continue
                        px_path, py_path = _bezier(*pts, t)
                        cands.append((px_path, py_path, None))
                        for cy_ in sorted(corridor_centers,
                                          key=lambda v: abs(v - py_path)):
                            cands.append((lx_, cy_, (px_path, py_path)))
                for px, py, leader in cands:
                    cand_ly = py - 3.0 - 5.5 * (n_lines - 1)
                    if _block_free(px, cand_ly, n_lines, half_w) and \
                            _free_of_labels(px, cand_ly, n_lines, half_w):
                        lx, ly = px, cand_ly
                        leader_from = leader
                        break
            placed.append({"lx": lx, "ly": ly, "lines": label_lines, "half_w": half_w})
            cond_txt = e.get("cond", "")
            label_txt = e.get("label", "")
            full = (cond_txt + "｜" + label_txt) if has_cond else label_txt
            label_slots.append({
                "lx": lx, "ly": ly, "lines": label_lines, "cond": has_cond,
                "full": full, "half_w": half_w, "leader": leader_from,
            })
    # 同走廊（同 y）标签横向排布（x 不重叠）；同巷道（同 x）标签纵向按块推让。
    by_y: dict[float, list[dict]] = {}
    by_x: dict[float, list[dict]] = {}
    for slot in label_slots:
        by_y.setdefault(slot["ly"], []).append(slot)
        by_x.setdefault(slot["lx"], []).append(slot)
    max_label_extent = 0.0
    for slots in by_y.values():
        slots.sort(key=lambda s: s["lx"])
        prev_right = None
        for slot in slots:
            if prev_right is not None and slot["lx"] - slot["half_w"] < prev_right + 8.0:
                moved = False
                new_lx = prev_right + 8.0 + slot["half_w"]
                # 横向推让不得落入节点矩形；受阻则改移到同走廊的巷道空位
                if _block_free(new_lx, slot["ly"], len(slot["lines"]), slot["half_w"]):
                    slot["lx"] = new_lx
                    moved = True
                else:
                    for lx_ in sorted(lane_centers, key=lambda v: abs(v - slot["lx"])):
                        if lx_ + slot["half_w"] < slot["lx"] - slot["half_w"] or                                 lx_ - slot["half_w"] > slot["lx"] + slot["half_w"] + 40.0:
                            continue
                        if _block_free(lx_, slot["ly"], len(slot["lines"]),
                                       slot["half_w"]) and                                 _free_of_labels(lx_, slot["ly"], len(slot["lines"]),
                                                slot["half_w"]):
                            slot["lx"] = lx_
                            moved = True
                            break
                if not moved:
                    # 左移尝试（走廊反向空位）
                    new_lx = slot["lx"] - slot["half_w"] * 2 - 8.0
                    if _block_free(new_lx, slot["ly"], len(slot["lines"]),
                                   slot["half_w"]) and                             _free_of_labels(new_lx, slot["ly"], len(slot["lines"]),
                                            slot["half_w"]):
                        slot["lx"] = new_lx
            prev_right = slot["lx"] + slot["half_w"]
            max_label_extent = max(max_label_extent, prev_right)
    for slots in by_x.values():
        slots.sort(key=lambda s: s["ly"])
        prev_bottom = None
        for slot in slots:
            if prev_bottom is not None and slot["ly"] - prev_bottom < 13.0:
                new_ly = prev_bottom + 13.0
                # 仅接受完整块空位（无 anchor-only 降级；推让后落入节点矩形则不推）
                if _block_free(slot["lx"], new_ly, len(slot["lines"]), slot["half_w"]):
                    slot["ly"] = new_ly
            prev_bottom = slot["ly"] + 11.0 * (len(slot["lines"]) - 1)
            max_label_extent = max(max_label_extent,
                                   slot["ly"] + 11.0 * (len(slot["lines"]) - 1) + 2.0)
    height = max(height, max_label_extent + _PAD)
    # 四向边界：全部最终可见内容（节点/圆点/IO/路径含弧绕控制点/细引线/标签块）
    # 必须落在画布内；先收集内容再按实际占用定 viewBox 原点与宽高。
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
        if slot.get("leader"):
            fx, fy = slot["leader"]
            min_x = min(min_x, fx, slot["lx"])
            max_x = max(max_x, fx, slot["lx"])
            min_y = min(min_y, fy, slot["ly"])
            max_y = max(max_y, fy, slot["ly"])
    for n in graph["nodes"]:
        if n["id"] not in positions:
            continue
        c, r = positions[n["id"]]
        nx = x_of(r)
        ny = rank_y[c]
        nh = h_by_node[n["id"]] if n["kind"] in ("op", "nav", "navlane") else 24.0
        min_x = min(min_x, nx, nx + _NODE_W)
        max_x = max(max_x, nx, nx + _NODE_W)
        min_y = min(min_y, ny, ny + nh)
        max_y = max(max_y, ny, ny + nh)
    vb_x = min_x - _PAD
    vb_y = min_y - _PAD
    vb_w = (max_x - min_x) + 2 * _PAD
    vb_h = (max_y - min_y) + 2 * _PAD
    content: list[str] = []
    content.extend(edge_parts)
    for slot in label_slots:
        if slot.get("leader"):
            fx, fy = slot["leader"]
            content.append(f'<path class="dg-leader" d="M{fx:g},{fy:g} '
                         f'L{slot["lx"]:g},{slot["ly"]:g}"/>')
    for slot in label_slots:
        # 条件行用条件色，其余行（源码名/shape）用常规边标签色——两者都可见
        spans = ""
        for k, line in enumerate(slot["lines"]):
            cls = ' class="elabel-cond"' if slot["cond"] and k == 0 else ""
            spans += (f'<tspan{cls} x="{slot["lx"]:g}" '
                      f'dy="{0 if k == 0 else 11:g}">{html.escape(line)}</tspan>')
        content.append(
            f'<text class="elabel" x="{slot["lx"]:g}" y="{slot["ly"]:g}" '
            f'text-anchor="middle">{spans}<title>{html.escape(slot["full"])}</title></text>')
    for n in graph["nodes"]:
        if n["id"] not in positions:
            continue
        c, r = positions[n["id"]]
        x = x_of(r)
        y = rank_y[c]
        title = f'<title>{html.escape(n.get("title", n["label"]))}</title>'
        href = f'<a href="{html.escape(n["href"])}">' if n.get("href") else ""
        href_end = "</a>" if href else ""
        if n["kind"] in ("op", "nav", "navlane"):
            # 计算框（op：源码名 + 实际计算类型）或导航框（总览模块/缓存通道，
            # 虚线，明确不是计算算子）
            h = h_by_node[n["id"]]
            body = ""
            lines = lines_by_node[n["id"]]
            for k, line in enumerate(lines):
                body += (f'<text class="nlabel" x="{_NODE_W / 2:g}" y="{14 + k * 12:g}" '
                         f'text-anchor="middle">{html.escape(line)}</text>')
            sub_lines = _node_sub_lines(n)
            for k, sline in enumerate(sub_lines):
                body += (f'<text class="nsub" x="{_NODE_W / 2:g}" '
                         f'y="{h - 6 - 13 * (len(sub_lines) - 1 - k):g}" '
                         f'text-anchor="middle">{html.escape(sline)}</text>')
            node = (f'{href}<g class="dg-node {n["kind"]}" '
                    f'transform="translate({x:g},{y:g})">'
                    f'<rect width="{_NODE_W:g}" height="{h:g}" rx="7"/>{title}{body}</g>'
                    f'{href_end}')
        elif n["id"] in ("t:hidden_streams_in", "t:hidden_streams_out"):
            # 初始 hidden_states 输入 / 层输出标识：非计算框的显式端口标签
            node = (f'{href}<g class="dg-port io" transform="translate({x:g},{y:g})">'
                    f'<rect width="{_NODE_W:g}" height="24" rx="12"/>{title}'
                    f'<text class="portlabel" x="{_NODE_W / 2:g}" y="16" '
                    f'text-anchor="middle">{html.escape(n["label"])}</text></g>{href_end}')
        else:
            # tensor/权重/缓存状态端点：小圆点端口（源码名+shape 走箭头标签，
            # 不再画与计算框相似的矩形）
            node = (f'{href}<g class="dg-dot {n["kind"]}" '
                    f'transform="translate({x + 10:g},{y + 12:g})">'
                    f'<circle r="5"/>{title}</g>{href_end}')
        content.append(node)
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
        "</defs>",
        *content,
        "</svg>",
    ]
    return "".join(parts)


_LEGEND_ITEMS = (
    ("opbox", "计算算子框（源码名 + 实际计算类型）"),
    ("navbox", "导航框（总览模块/缓存通道，虚线——非计算算子）"),
    ("dot", "端口圆点（tensor/权重/缓存状态端点；源码名+shape 在箭头上）"),
    ("flow", "数据流（激活/索引/掩码张量，箭头=producer→consumer）"),
    ("weight", "权重/参数引用（虚线；不改变数据流方向）"),
    ("cache", "缓存状态（点线；调用前读入算子 / 算子写调用后状态）"),
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
.dg-leader { stroke: #8a94a6; stroke-width: 1; stroke-dasharray: 2 2; fill: none; }
.dg-edge { fill: none; stroke-width: 1.5; }
.dg-edge.flow { stroke: #1f4e79; }
.dg-edge.weight { stroke: #b08d57; stroke-dasharray: 5 4; }
.dg-edge.cache { stroke: #2b5d8f; stroke-dasharray: 2 3; }
.dg-edge.cond { stroke-dasharray: 7 4; }
.dg-arr-flow { fill: #1f4e79; }
.dg-arr-weight { fill: #b08d57; }
.dg-arr-cache { fill: #2b5d8f; }/* 边标签随主题成对：浅色主题深字浅晕 / 深色主题浅字深晕，均不依赖节点底色 */
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
.dg-swatch.cond { border-top-style: dashed; border-top-color: #8f2b4d; }
.dg-note { font-size: 12px; color: var(--muted); margin: 6px 0 0; }
ul.toc { list-style: none; padding-left: 8px; }
ul.toc li { margin: 2px 0; font-size: 13px; }
.crumb { font-size: 12px; color: var(--muted); margin: 2px 0 6px; }
@media (prefers-color-scheme: dark) {
  /* 箭头/marker/图例随主题成对：固定深蓝在暗底对比不足（~1.87），
     暗主题换浅色画线（对暗卡对比 ≥4.5），浅主题保持原深蓝（≥8） */
  .dg-edge.flow { stroke: #7fa8d9; }
  .dg-arr-flow { fill: #7fa8d9; }
  .dg-edge.weight { stroke: #d4b183; }
  .dg-arr-weight { fill: #d4b183; }
  .dg-edge.cache { stroke: #9fc2e8; }
  .dg-arr-cache { fill: #9fc2e8; }
  .dg-swatch.flow { border-top-color: #7fa8d9; }
  .dg-swatch.weight { border-top-color: #d4b183; }
  .dg-swatch.cache { border-top-color: #9fc2e8; }
  .dg-swatch.opbox { border-top-color: #7fa8d9; }
  .dg-swatch.navbox { border-top-color: #7fa8d9; }
  .dg-swatch.dot { border-top-color: #9aa7b4; }
  .elabel { fill: #d9e2ea; stroke: rgba(20,23,27,0.92); }
  .elabel-cond { fill: #ff9a9a; }
}
"""
