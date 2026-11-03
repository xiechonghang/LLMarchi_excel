"""DeepSeek layout-v2：共用模板 + 差异化通路 + 参数层派生维度。

第二轮用户反馈（E01-layout-feedback02）要求：
- A 列整体结构只标一次，取消四层组逐行重复/横幅；
- mHC 前后、Attention Q/KV 前置、O/残差按同构模板只展示一次，成本×真实使用次数；
- SWA/CSA/HCA 的缓存、压缩、Indexer、条件分支保留；
- 复杂维度/检查提升到参数层，主行维度以直接参数引用/短算术为主；
- 主表 12 列，删顶部 revision/实现/草稿说明与表尾长规则。

本模块复用 ``deepseek_perf_layout`` 的既有一切事实/公式/参数/守卫/稳定 ZIP 辅助，
不重建第二套模型事实；v1 旧布局保持独立入口。
"""
from __future__ import annotations

import hashlib
import io
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter

from structure_reader.export import deepseek_perf_layout as v1
from structure_reader.export.hf_workbook import build_structures_workbook_spec
from structure_reader.export.operator_workbook import _normalize_zip
from structure_reader.scenario_evaluator import (
    ExpressionError,
    evaluate_bool,
    evaluate_int,
    excel_formula,
    expression_names,
)

__all__ = ["build_perf_layout_v2_workbook", "export_perf_layout_v2_workbook"]

MAIN_HEADERS = v1.MAIN_HEADERS
MAIN_WIDTHS = (28, 27, 14, 7, 15, 17, 14, 12, 15, 16, 16, 16)
_FMT_TOTAL = "#,##0.0"

# 同构判定中，这些 family 需要按注意力/hash-topk 语义再分作用域：
# 即使成本公式相同，也不把 SWA/CSA/HCA 缓存或 Hash/TopK 路由混成一行。
_CONTEXT_FAMILIES = {"cache_update", "moe_router"}

_QKV_FAMILIES = {
    "q_a_proj", "q_a_norm", "q_b_proj", "q_b_norm", "q_rope_apply",
    "kv_proj", "kv_head_view", "kv_rope_apply",
}
_O_FAMILIES = {"o_a_proj", "o_b_proj", "attn_inv_rope", "attn_group_reshape"}
_CORE_FAMILIES = {
    "cache_update", "repeat_kv", "qk_scaled", "av_matmul", "mask_add",
    "sink_softmax", "mask_extend", "c_mask_extend", "kv_concat",
    "c_block_bias", "hca_block_bias",
}

_BLOCK_ORDER = [
    "整网入口",
    "共用 mHC",
    "层前 RMSNorm",
    "Attention Q/KV 前置",
    "SWA 缓存更新",
    "SWA Attention 核心",
    "CSA 缓存更新",
    "CSA 压缩器/Indexer",
    "CSA Attention 核心",
    "HCA 缓存更新",
    "HCA 压缩器",
    "HCA Attention 核心",
    "O 投影/后处理",
    "共用残差混合",
    "MoE 路由/专家模板",
    "Hash 路由专属",
    "TopK 路由",
    "Hash 专家切片",
    "TopK 专家切片",
    "共享专家",
    "整网出口",
]

_ATTN_CN = {
    "sliding_attention": "SWA",
    "compressed_sparse_attention": "CSA",
    "heavily_compressed_attention": "HCA",
}


@dataclass
class _Template:
    key: str
    family: str
    op: dict
    group: dict
    rule: dict
    groups: tuple[str, ...]
    instances: tuple[tuple[str, str], ...]
    usage: int
    block: str
    order: int
    sig: tuple
    params_sig: tuple = ()
    context: str = ""
    usage_row: int = 0
    main_row: int = 0
    aux_row: int = 0


@dataclass
class _Plan:
    templates: list[_Template]
    blocks: list[tuple[str, int, list[_Template]]]
    instance_map: dict[tuple[str, str], _Template]
    all_instances: list[tuple[str, str]]
    excluded: list[tuple[str, str]]
    container: list[tuple[str, str]]
    coverage: dict
    usage_rows: dict[str, int]
    composite_cells: dict[tuple[str, str], int] = field(default_factory=dict)
    composite_rows: dict[tuple[str, str], list[int]] = field(default_factory=dict)


def _family(op_id: str) -> str:
    for prefix in ("attn_hc_", "ffn_hc_"):
        if op_id.startswith(prefix):
            return "hc_" + op_id[len(prefix):]
    for prefix in ("attn_residual_", "ffn_residual_"):
        if op_id.startswith(prefix):
            return "residual_" + op_id[len(prefix):]
    if op_id in ("attn_norm", "ffn_norm"):
        return "norm"
    return op_id


def _param_sig(group: dict, name: str, stack: tuple[str, ...] = ()) -> tuple:
    if name in ("B", "seqlen", "kvcachelen", "N_hit", "Tcube", "Tvec"):
        return ("input", name)
    param = next((p for p in group["parameters"]
                  if p["id"] == name or p["symbol"] == name), None)
    if param is None:
        return ("missing", name)
    if param.get("formula"):
        deps = tuple(sorted((dep, _param_sig(group, dep, stack + (param["id"],)))
                            for dep in expression_names(param["formula"])))
        return ("formula", param["id"], param["formula"], deps)
    return ("value", param["id"], param.get("value"))


def _signature(group: dict, op: dict) -> tuple:
    rule = op["performance"]
    names: set[str] = set()
    for term in rule.get("terms", []):
        for key in ("b", "m", "k", "n"):
            if term.get(key):
                names.update(expression_names(term[key]))
    if isinstance(rule.get("repeat"), str):
        names.update(expression_names(rule["repeat"]))
    for guard in rule.get("guards", []):
        names.update(expression_names(guard))
    conditional = rule.get("conditional") or {}
    if conditional.get("when"):
        names.update(expression_names(conditional["when"]))
    params = tuple(sorted((name, _param_sig(group, name)) for name in names))
    rule_sig = (
        rule["class"],
        tuple((t.get("b"), t.get("m"), t.get("k"), t.get("n"))
              for t in rule.get("terms", [])),
        rule.get("repeat"),
        conditional.get("when"),
        (rule.get("charge") or {}).get("mode"),
        (rule.get("charge") or {}).get("charged_at"),
        tuple(rule.get("guards", [])),
    )
    return (_family(op["id"]), rule_sig, params)


def _context(group: dict, family: str) -> str:
    if family == "cache_update":
        return group.get("attention_type", "")
    if family == "moe_router":
        return group.get("moe_type", "")
    return ""


def _group_count(group: dict) -> int:
    return len(group["layer_indices"]) if group.get("layer_indices") else 1


def _is_billable(op: dict) -> bool:
    rule = op["performance"]
    return (rule["class"] != "container"
            and (rule.get("charge") or {}).get("mode") != "excluded")


def _block_for(family: str, group: dict, usage: int) -> str:
    attn = _ATTN_CN.get(group.get("attention_type", ""), "Attention")
    if family.startswith("hc_"):
        return "共用 mHC"
    if family == "norm":
        return "层前 RMSNorm"
    if family in _QKV_FAMILIES:
        return "Attention Q/KV 前置"
    if family in _O_FAMILIES:
        return "O 投影/后处理"
    if family.startswith("residual_"):
        return "共用残差混合"
    if family == "moe_hash_lookup":
        return "Hash 路由专属"
    if family == "moe_router":
        return "Hash 路由专属" if group.get("moe_type") == "hash_moe" else "TopK 路由"
    if family == "moe_expert_slice":
        return "Hash 专家切片" if usage <= 3 else "TopK 专家切片"
    if family.startswith("shared_"):
        return "共享专家"
    if family.startswith("moe_"):
        return "MoE 路由/专家模板"
    if family.startswith("net_"):
        return "整网出口" if family in {
            "net_hc_head_norm", "net_hc_head_project", "net_hc_head_pre",
            "net_hc_head_collapse", "net_final_norm", "net_logits_slice",
            "net_lm_head"} else "整网入口"
    if family == "cache_update":
        return f"{attn} 缓存更新"
    if family.startswith("c_") or family.startswith("i_"):
        if attn == "HCA" and family.startswith("c_"):
            return "HCA 压缩器"
        return f"{attn} 压缩器/Indexer"
    if family in _CORE_FAMILIES:
        return f"{attn} Attention 核心"
    return "其他"


def _block_usage(block: str, plan_templates: list[_Template]) -> int:
    return max(t.usage for t in plan_templates)


def _block_derivation(block: str) -> str:
    if block in ("共用 mHC", "层前 RMSNorm", "共用残差混合"):
        return "每层 attn/ffn 各 1 次 = 2×43"
    if block in ("Attention Q/KV 前置", "O 投影/后处理",
                 "MoE 路由/专家模板", "共享专家"):
        return "每层 1 次 = 43"
    if block == "SWA 缓存更新" or block == "SWA Attention 核心":
        return "SWA 真实层 2"
    if block.startswith("CSA"):
        return "CSA-hash 1 + CSA-topk 20 = 21"
    if block.startswith("HCA"):
        return "HCA 真实层 20"
    if block == "Hash 路由专属" or block == "Hash 专家切片":
        return "SWA 2 + CSA-hash 1 = 3"
    if block in ("TopK 路由", "TopK 专家切片"):
        return "HCA 20 + CSA-topk 20 = 40"
    if block in ("整网入口", "整网出口"):
        return "整网一次"
    return ""


def _build_plan(spec: dict) -> _Plan:
    sheet = spec["sheets"][0]
    groups = {g["id"]: g for g in sheet["groups"]}
    serial = ["net-periphery"]
    serial += [item["group"] for item in sheet["network"]["serial_order"]
               if item.get("kind") == "group" and item["group"] != "net-periphery"]
    order_index: dict[str, int] = {}
    all_instances: list[tuple[str, str]] = []
    excluded: list[tuple[str, str]] = []
    container: list[tuple[str, str]] = []
    buckets: dict[tuple, list[tuple[str, dict]]] = defaultdict(list)
    for gid in serial:
        group = groups[gid]
        for op in group["operators"]:
            all_instances.append((gid, op["id"]))
            rule = op["performance"]
            if rule["class"] == "container":
                container.append((gid, op["id"]))
                continue
            if (rule.get("charge") or {}).get("mode") == "excluded":
                excluded.append((gid, op["id"]))
                continue
            family = _family(op["id"])
            order_index.setdefault(family, len(order_index))
            context = _context(group, family)
            full_sig = _signature(group, op)
            key = (full_sig, context)
            buckets[key].append((gid, op))

    templates: list[_Template] = []
    instance_map: dict[tuple[str, str], _Template] = {}
    for key, items in buckets.items():
        families = {_family(op["id"]) for _gid, op in items}
        if len(families) != 1:
            raise ValueError(f"模板 family 不一致：{families}")
        family = next(iter(families))
        groups_used = tuple(sorted({gid for gid, _op in items}))
        usage = sum(_group_count(groups[gid]) for gid, _op in items)
        first_gid, first_op = items[0]
        context = _context(groups[first_gid], family)
        block = _block_for(family, groups[first_gid], usage)
        template = _Template(
            key=f"{family}|{hashlib.sha1(repr((key, groups_used)).encode("utf-8")).hexdigest()[:10]}",
            family=family, op=first_op, group=groups[first_gid],
            rule=first_op["performance"], groups=groups_used,
            instances=tuple((gid, op["id"]) for gid, op in items),
            usage=usage, block=block,
            order=order_index.get(family, 999), sig=key, params_sig=key[0][2],
            context=context,
        )
        templates.append(template)
        for gid, op in items:
            instance_map[(gid, op["id"])] = template

    block_order = {name: index for index, name in enumerate(_BLOCK_ORDER)}
    templates.sort(key=lambda t: (block_order.get(t.block, 999), t.order, t.key))
    grouped: dict[str, list[_Template]] = defaultdict(list)
    for template in templates:
        grouped[template.block].append(template)
    blocks: list[tuple[str, int, list[_Template]]] = []
    for block in sorted(grouped, key=lambda name: block_order.get(name, 999)):
        items = grouped[block]
        usages = {t.usage for t in items}
        if len(usages) != 1:
            raise ValueError(f"模板块 {block} 内使用次数不一致：{sorted(usages)}")
        blocks.append((block, items[0].usage, items))

    # 模板使用行（模板映射表顶部）
    usage_rows: dict[str, int] = {}
    for index, (block, usage, _items) in enumerate(blocks, start=5):
        usage_rows[block] = index
        for template in _items:
            template.usage_row = index

    # 覆盖校验：每个可计费原实例都被映射；使用次数之和等于原实例层数之和。
    expected_usage = sum(_group_count(groups[gid])
                         for gid, oid in all_instances
                         if (gid, oid) in instance_map)
    actual_usage = sum(t.usage for t in templates)
    if expected_usage != actual_usage:
        raise ValueError(f"模板使用次数覆盖不一致：{actual_usage} != {expected_usage}")
    coverage = {
        "original_instances": len(all_instances),
        "billable_instances": len(instance_map),
        "excluded_instances": len(excluded),
        "container_instances": len(container),
        "templates": len(templates),
        "expected_usage": expected_usage,
        "actual_usage": actual_usage,
    }
    composite_cells: dict[tuple[str, str], int] = {}
    composite_rows: dict[tuple[str, str], list[int]] = {}
    aux_row = 5
    for scene in ("Prefill", "Decode"):
        for template in templates:
            if len(template.rule.get("terms", [])) <= 1:
                continue
            rows = list(range(aux_row, aux_row + len(template.rule["terms"])))
            composite_rows[(scene, template.key)] = rows
            composite_cells[(scene, template.key)] = aux_row + len(rows)
            aux_row += len(rows) + 2
    return _Plan(templates=templates, blocks=blocks, instance_map=instance_map,
                 all_instances=all_instances, excluded=excluded, container=container,
                 coverage=coverage, usage_rows=usage_rows,
                 composite_cells=composite_cells, composite_rows=composite_rows)


# ---------------------------------------------------------------- 派生参数

_V2_DERIVED = [
    ("M_attn", "M_attn 有效查询行", "Nh*seqlen", "", "Nh×T"),
    ("M_og", "M_og 分组有效行", "G*seqlen", "", "G×T"),
    ("M_route", "M_route 路由分配行", "seqlen*K", "", "T×K"),
    ("M_res", "M_res 残差混合行", "seqlen*Nhc", "", "T×Nhc"),
    ("K_og", "K_og 分组收缩维", "Nh*Dh//G", "", "Nh·Dh/G"),
    ("K_hc", "K_hc mHC 收缩维", "Nhc*H", "", "Nhc·H"),
    ("N_hc", "N_hc mHC 投影输出", "(2+Nhc)*Nhc", "", "(2+Nhc)·Nhc"),
    ("N_qkv", "N_qkv QKV 输出维", "Nh*Dh", "", "Nh·Dh"),
    ("N_gu", "N_gu 专家 gate/up 输出", "2*Es", "", "2·Es"),
    ("N_slice", "N_slice 专家切片", "3*Es*H", "", "3·Es·H"),
    ("N_comb", "N_comb 残差 comb 矩阵", "Nhc*Nhc", "", "Nhc²"),
    ("N_disp", "N_disp 分派 one-hot", "(E+1)*K", "", "(E+1)·K"),
    ("Nhit_ok_hash", "Nhit_ok 命中数相容（Hash）",
     "(N_hit <= min(E, seqlen*K)) and ((seqlen*K == 0) or (N_hit >= 1))", "", "布尔检查"),
    ("Nhit_ok_topk", "Nhit_ok 命中数相容（TopK）",
     "(N_hit <= min(E, seqlen*K)) and ((seqlen*K == 0) or (N_hit >= K))", "", "布尔检查"),
    ("Kv_vis", "Kv_vis 可见 KV 长度", "Lkv+Ncomp", "CSA", "Lkv+Ncomp"),
    ("Kv_vis", "Kv_vis 可见 KV 长度", "Lkv+Ncomp", "HCA", "Lkv+Ncomp"),
    ("H_bias_on", "HCA block_bias 启用", "(seqlen != 1) and (Ncomp > 0)", "HCA", "布尔条件"),
    ("V_seqlen", "seqlen（校验后的值）", "seqlen", "", "整数 ≥ 0"),
    ("V_kvcach", "kvcachelen（校验后的值）", "kvcachelen", "", "整数 ≥ 0"),
    ("V_Nhit", "N_hit（校验后的值）", "N_hit", "", "整数 ≥ 0；空白=未知"),
]


_V2_DROP_PIDS = {"Nkv", "Sit", "Eps", "EpsN", "Lim", "Scl", "ThetaM",
                  "ThetaC", "Yf", "Nlay", "Lkvp"}
_V2_POSITIVE_PIDS = {"H", "Nh", "Dh", "Rq", "Rk", "G", "Ro", "W", "Nhc",
                     "E", "K", "Es", "Id", "Ih", "Ki", "M", "Vocab"}


def _build_v2_entries(spec: dict):
    entries, by_key, mdep = v1._build_entries(spec)
    entries = [e for e in entries
               if not (e.kind == "param" and e.pid in _V2_DROP_PIDS)]
    by_key = {e.key: e for e in entries}
    # 基础字面量的校验副本；主表维度引用 V_*，不直接引用可被改写的配置格。
    base_values = [e for e in entries
                   if e.kind == "param" and e.formula is None and e.value is not None]
    for base in base_values:
        pid = f"V_{base.pid}"
        key = f"{pid}@{base.scope}" if base.scope else pid
        param = {"id": pid, "symbol": pid, "formula": base.symbol,
                 "meaning": f"{base.label}（校验值）", "axis_aliases": [],
                 "raw_id": base.pid,
                 "raw_kind": ("int" if isinstance(base.value, int) else
                              "float" if isinstance(base.value, float) else "text")}
        entry = v1._Entry(
            key=key, pid=pid, label=f"{base.label}（校验值）", symbol=pid,
            category="校验输入", scope=base.scope, kind="param",
            param=param, formula=base.symbol, value=None,
            note="已校验基础值；非法时未知，主表直接引用",
        )
        entries.append(entry)
        by_key[key] = entry
    for pid, label, formula, scope, meaning in _V2_DERIVED:
        key = f"{pid}@{scope}" if scope else pid
        symbol = pid if not scope else pid
        display = label + (f"（{scope}）" if scope else "")
        param = {
            "id": pid, "symbol": symbol, "formula": formula,
            "meaning": meaning, "axis_aliases": [],
            "raw_id": {"V_seqlen": "seqlen", "V_kvcach": "kvcachelen",
                       "V_Nhit": "N_hit"}.get(pid),
            "raw_kind": "int",
        }
        entry = v1._Entry(
            key=key, pid=pid, label=display, symbol=symbol,
            category="派生维度", scope=scope, kind="param",
            param=param, formula=formula, value=None,
            note=f"派生维度：{meaning}",
        )
        entries.append(entry)
        by_key[key] = entry
    return entries, by_key, mdep


# ---------------------------------------------------------------- 参考成本

def _resolver(group: dict, env: dict):
    params = {p["id"]: p for p in group["parameters"]}
    by_symbol = {p["symbol"]: p for p in group["parameters"]}
    values: dict = {}

    def get(name):
        if name in env:
            value = env[name]
            if value is None:
                raise KeyError(name)
            return value
        if name in values:
            return values[name]
        param = params.get(name) or by_symbol.get(name)
        if param is None:
            raise KeyError(name)
        if param.get("formula"):
            refs = {key: get(key) for key in expression_names(param["formula"])}
            try:
                value = evaluate_int(param["formula"], refs)
            except Exception:
                value = evaluate_bool(param["formula"], refs)
        else:
            value = param.get("value")
        values[param["id"]] = value
        values[param["symbol"]] = value
        for alias in param.get("axis_aliases", []):
            values[alias] = value
        return value
    return get


def _op_cost(group: dict, op: dict, env: dict, cube_t: float, vec_t: float):
    rule = op["performance"]
    get = _resolver(group, env)
    conditional = rule.get("conditional")
    if conditional:
        try:
            refs = {key: get(key) for key in expression_names(conditional["when"])}
            if not evaluate_bool(conditional["when"], refs):
                return 0.0, 0.0
        except Exception:
            return None
    for guard in rule.get("guards", []):
        try:
            refs = {key: get(key) for key in expression_names(guard)}
            if not evaluate_bool(guard, refs):
                return None
        except Exception:
            return None
    repeat = rule.get("repeat", 1)
    repeat = int(repeat) if isinstance(repeat, (int, str)) else 1
    work = 0
    try:
        for term in rule.get("terms", []):
            keys = ("b", "m", "k", "n") if rule["class"] == "cube" else ("m", "n")
            product = repeat
            for key in keys:
                product *= evaluate_int(term[key],
                                        {n: get(n) for n in expression_names(term[key])})
            work += product
    except Exception:
        return None
    if rule["class"] == "cube":
        return work / cube_t, 0.0
    return 0.0, work / vec_t


def _reference_totals(spec: dict, n_hit=6):
    out = {}
    for sheet in spec["sheets"]:
        env = {"B": 1, "seqlen": sheet["inputs"][0]["value"],
               "kvcachelen": sheet["inputs"][1]["value"], "N_hit": n_hit}
        cube_t = sheet["inputs"][7]["value"]
        vec_t = sheet["inputs"][8]["value"]
        totals = {"cube": 0.0, "vec": 0.0, "op": 0.0}
        for group in sheet["groups"]:
            count = len(group["layer_indices"]) if group["layer_indices"] else 1
            for op in group["operators"]:
                if not _is_billable(op):
                    continue
                cost = _op_cost(group, op, env, cube_t, vec_t)
                if cost is None:
                    raise ValueError(f"参考成本未定：{group['id']}/{op['id']}")
                c, v = cost
                totals["cube"] += c * count
                totals["vec"] += v * count
                totals["op"] += max(c, v) * count
        out[sheet["name"]] = totals
    return out


def _template_cost(template: _Template, spec: dict, scene: str, n_hit=6):
    sheet = next(item for item in spec["sheets"] if item["name"] == scene)
    env = {"B": 1, "seqlen": sheet["inputs"][0]["value"],
           "kvcachelen": sheet["inputs"][1]["value"], "N_hit": n_hit}
    cube_t = sheet["inputs"][7]["value"]
    vec_t = sheet["inputs"][8]["value"]
    cost = _op_cost(template.group, template.op, env, cube_t, vec_t)
    if cost is None:
        raise ValueError(f"模板成本未定：{template.key}")
    return cost


def _compare_costs(plan: _Plan, spec: dict, n_hit=6):
    ref = _reference_totals(spec, n_hit=n_hit)
    got = {"Prefill": {"cube": 0.0, "vec": 0.0, "op": 0.0},
           "Decode": {"cube": 0.0, "vec": 0.0, "op": 0.0}}
    for template in plan.templates:
        for name in got:
            c, v = _template_cost(template, spec, name, n_hit=n_hit)
            got[name]["cube"] += c * template.usage
            got[name]["vec"] += v * template.usage
            got[name]["op"] += max(c, v) * template.usage
    for name in got:
        for key in ("cube", "vec", "op"):
            if abs(got[name][key] - ref[name][key]) > max(1e-7, abs(ref[name][key]) * 1e-9):
                raise ValueError(
                    f"模板成本对照不一致 {name}/{key}: {got[name][key]} != {ref[name][key]}")
    return {"reference": ref, "templates": got, "n_hit": n_hit}

# ---------------------------------------------------------------- 参数/维度

_VALIDATED_INPUT = {
    "seqlen": "V_seqlen",
    "kvcachelen": "V_kvcach",
    "N_hit": "V_Nhit",
}

_DERIVED_MAP = {
    "Nh*seqlen": "M_attn",
    "G*seqlen": "M_og",
    "seqlen*K": "M_route",
    "seqlen*Nhc": "M_res",
    "Nh*Dh//G": "K_og",
    "Nhc*H": "K_hc",
    "(2+Nhc)*Nhc": "N_hc",
    "Nh*Dh": "N_qkv",
    "2*Es": "N_gu",
    "3*Es*H": "N_slice",
    "Nhc*Nhc": "N_comb",
    "(E+1)*K": "N_disp",
    "Lkv+Ncomp": "Kv_vis",
}


def _expr_names(expr: str) -> list[str]:
    try:
        return expression_names(expr)
    except ExpressionError:
        return []


def _ref_for(name: str, table: dict, input_table: dict, sheet_prefix: str = "") -> str:
    if name == "B":
        return "1"
    if name in _VALIDATED_INPUT:
        return sheet_prefix + table[_VALIDATED_INPUT[name]][0]
    vname = f"V_{name}"
    if vname in table:
        return sheet_prefix + table[vname][0]
    if name in input_table:
        return sheet_prefix + input_table[name][0]
    if name in table:
        return sheet_prefix + table[name][0]
    raise KeyError(name)


def _expr_formula(expr: str, table: dict, input_table: dict,
                  sheet_prefix: str = "") -> str:
    """把维度表达式编译成“直接参数引用/短算术”，不在这里展开逐项守卫。"""
    if expr == "1":
        return "1"
    if expr in ("B", "batch_size"):
        return "1"
    if expr in _VALIDATED_INPUT:
        return _ref_for(expr, table, input_table, sheet_prefix)
    if expr in ("B*seqlen", "seqlen*B"):
        return _ref_for("seqlen", table, input_table, sheet_prefix)
    if expr in _DERIVED_MAP:
        return _ref_for(_DERIVED_MAP[expr], table, input_table, sheet_prefix)
    if expr in table:
        return _ref_for(expr, table, input_table, sheet_prefix)
    refs = {name: _ref_for(name, table, input_table, sheet_prefix)
            for name in _expr_names(expr)}
    return excel_formula(expr, refs)


def _v2_n_formula(n_expr: str, table: dict, input_table: dict) -> Any:
    if n_expr == "1":
        return 1
    return "=" + _expr_formula(n_expr, table, input_table)


def _v2_m_formula(mp: dict, row: int, table: dict, input_table: dict) -> str:
    b = mp["b"]
    if b in ("B", "B*Nh", "B*Ih"):
        return f'=IFERROR(D{row}*E{row},"未知*")'

    if b == "B*G":
        return "=" + _ref_for("M_og", table, input_table)
    if b == "B*seqlen":
        return "=" + _ref_for("M_res", table, input_table)
    if b == "1" and mp["m"] in ("B*seqlen*K",):
        return "=" + _ref_for("M_route", table, input_table)
    return "=" + _expr_formula(f"({b})*({mp['m']})", table, input_table)


def _v2_term_work_formula(term: dict, table: dict, input_table: dict,
                          sheet_prefix: str = "") -> str:
    keys = ("b", "m", "k", "n") if "b" in term else ("m", "n")
    pieces = []
    for key in keys:
        expr = term[key]
        value = _expr_formula(expr, table, input_table, sheet_prefix)
        pieces.append(f"({value})")
    return "*".join(pieces)


def _v2_term_dims(term: dict, table: dict, input_table: dict,
                  sheet_prefix: str = "") -> tuple[str, str, str]:
    keys = ("b", "m", "k", "n") if "b" in term else ("m", "n")
    if "b" in term:
        m = _expr_formula(f"({term['b']})*({term['m']})", table, input_table, sheet_prefix)
        k = _expr_formula(term["k"], table, input_table, sheet_prefix)
        n = _expr_formula(term["n"], table, input_table, sheet_prefix)
    else:
        m = _expr_formula(term["m"], table, input_table, sheet_prefix)
        k = "—"
        n = _expr_formula(term["n"], table, input_table, sheet_prefix)
    return m, k, n


def _v2_dims(template: _Template, table: dict, input_table: dict, row: int) -> dict:
    rule = template.rule
    if rule["class"] == "cube":
        mp = v1._cube_map(rule["terms"][0])
        n = _v2_n_formula(mp["n"], table, input_table)
        t = "=" + _expr_formula(mp["T"], table, input_table)
        d = "=" + _expr_formula(mp["D"], table, input_table)
        m = _v2_m_formula(mp, row, table, input_table)
        k = f"=F{row}"
        n_out = "=" + _expr_formula(mp["N"], table, input_table)
        return {"n": n, "T": t, "D": d, "M": m, "K": k, "N": n_out}
    mp = v1._vec_map(rule["terms"][0])
    n = _v2_n_formula(mp["n"], table, input_table)
    t = "=" + _expr_formula(mp["T"], table, input_table)
    d = "=" + _expr_formula(mp["D"], table, input_table)
    return {"n": n, "T": t, "D": d, "M": "—", "K": "—", "N": "—"}


def _write_v2_dim(ws, row: int, col: int, value, *, fmt=None) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v1._span_number(ws, row, col, col, value, fmt=fmt)
    else:
        text = str(value)
        if text == "—":
            v1._span_text(ws, row, col, col, "—", align=v1._ALIGN_CENTER)
        else:
            if not text.startswith("="):
                text = "=" + text
            v1._span_formula(ws, row, col, col, text, fmt=fmt)
            ws.cell(row=row, column=col).font = v1._FONT_BODY


def _wrap_bool_guard(cell: str, core: str, false_text: str) -> str:
    # Excel 逻辑值可能是 TRUE/FALSE，也可能被重算器落成 1/0；两种都接受，
    # 文本/空白/其他数值一律保持未知，不把非法条件折成 0。
    return (f'IF(ISLOGICAL({cell}),IF({cell},{core},{false_text}),'
            f'IF(AND(ISNUMBER({cell}),OR({cell}=0,{cell}=1)),'
            f'IF({cell}<>0,{core},{false_text}),"未知*"))')


def _v2_cost_cell(kind: str, row: int, repeat: int,
                  cube_ref: str, vec_ref: str, *,
                  guard_cell: str | None = None,
                  cond_cell: str | None = None,
                  composite_cell: str | None = None) -> tuple[str, int]:
    """返回 (J/K 公式或 0, 另一侧 0)。耗时格只做短行内算术 + 吞吐守卫。"""
    thru = cube_ref if kind == "cube" else vec_ref
    thru_ok = (f"AND(ISNUMBER({thru}),NOT(ISLOGICAL({thru})),{thru}>0)")
    if composite_cell is not None:
        core = (f'IF(AND(ISNUMBER({composite_cell}),{thru_ok}),'
                f'IFERROR(({composite_cell})/{thru},"未知*"),"未知*")')
        return "=" + core, 0
    if kind == "cube":
        dims = f"G{row},H{row},I{row}"
        body = f"G{row}*H{row}*I{row}"
    else:
        dims = f"D{row},E{row},F{row}"
        body = f"D{row}*E{row}*F{row}"
    if repeat != 1:
        body = f"({body})*{repeat}"
    guard = f"AND(AND(ISNUMBER({dims.replace(',', '),ISNUMBER(')})),{thru_ok})"
    core = f'IF({guard},IFERROR(({body})/{thru},"未知*"),"未知*")'
    if guard_cell:
        core = _wrap_bool_guard(guard_cell, core, '"未知*"')
    elif cond_cell:
        core = _wrap_bool_guard(cond_cell, core, "0")
    return "=" + core, 0


def _base_literals(entries: list) -> list:
    return [entry for entry in entries
            if entry.kind == "param" and entry.formula is None
            and entry.value is not None]


def _raw_cells(entries: list) -> dict[str, str]:
    return {entry.key: f"'原始参数'!$D${index}"
            for index, entry in enumerate(_base_literals(entries), start=5)}


def _validated_base_formula(value, raw_ref: str, minimum: int = 0) -> str:
    if isinstance(value, bool):
        return f'=IF(ISLOGICAL({raw_ref}),{raw_ref},"未知*")'
    if isinstance(value, int):
        return (f'=IF(IFERROR(AND(ISNUMBER({raw_ref}),NOT(ISLOGICAL({raw_ref})),'
                f'INT({raw_ref})={raw_ref},{raw_ref}>={minimum}),FALSE()),'
                f'{raw_ref},"未知*")')
    if isinstance(value, float):
        return (f'=IF(IFERROR(AND(ISNUMBER({raw_ref}),'
                f'NOT(ISLOGICAL({raw_ref}))),FALSE()),{raw_ref},"未知*")')
    return (f'=IF(IFERROR(AND(ISTEXT({raw_ref}),LEN({raw_ref})>0),FALSE()),'
            f'{raw_ref},"未知*")')


def _overwrite_base_cells(ws, entries: list, raw_cells: dict[str, str]) -> None:
    for entry in _base_literals(entries):
        raw_ref = raw_cells.get(entry.key)
        if not raw_ref:
            continue
        minimum = 1 if entry.pid in _V2_POSITIVE_PIDS else 0
        cell = ws.cell(row=entry.row, column=entry.col)
        cell.value = _validated_base_formula(entry.value, raw_ref, minimum)
        cell.number_format = "General"


def _validated_formula(raw_ref: str, kind: str, minimum: int = 0) -> str:
    if kind == "int":
        return (f'=IF(IFERROR(AND(ISNUMBER({raw_ref}),NOT(ISLOGICAL({raw_ref})),'
                f'INT({raw_ref})={raw_ref},{raw_ref}>={minimum}),FALSE()),'
                f'{raw_ref},"未知*")')
    if kind == "float":
        return (f'=IF(IFERROR(AND(ISNUMBER({raw_ref}),'
                f'NOT(ISLOGICAL({raw_ref}))),FALSE()),{raw_ref},"未知*")')
    return (f'=IF(IFERROR(AND(ISTEXT({raw_ref}),LEN({raw_ref})>0),FALSE()),'
            f'{raw_ref},"未知*")')


def _overwrite_validated_cells(ws, entries: list, base: dict,
                               scope_tables: dict) -> None:
    for entry in entries:
        param = entry.param or {}
        raw_id = param.get("raw_id")
        if not entry.pid.startswith("V_") or not raw_id:
            continue
        table = scope_tables.get(entry.scope) or base
        raw_ref = table[raw_id][0]
        minimum = 1 if raw_id in _V2_POSITIVE_PIDS else 0
        cell = ws.cell(row=entry.row, column=entry.col)
        cell.value = _validated_formula(raw_ref, param.get("raw_kind", "int"), minimum)
        cell.number_format = "General"
        entry.guard = f"ISNUMBER({entry.cell})"


def _write_raw_sheet(wb, entries: list, raw_cells: dict[str, str]) -> None:
    ws = wb.create_sheet("原始参数")
    headers = ("参数id", "符号", "作用域", "原始值", "说明")
    for index, label in enumerate(headers, start=1):
        c = ws.cell(row=4, column=index, value=label)
        c.font = v1._FONT_HEAD
        c.fill = v1._FILL_HEAD
        c.border = v1._BORDER
    for address, entry in zip(raw_cells.values(), _base_literals(entries)):
        row = int(address.rsplit("$", 1)[1])
        values = (entry.pid, entry.symbol, entry.scope or "通用",
                  entry.value, entry.note or "config/基础参数")
        for index, value in enumerate(values, start=1):
            c = ws.cell(row=row, column=index, value=value)
            c.font = v1._FONT_SMALL
            c.border = v1._BORDER
            c.alignment = v1._ALIGN_WRAP
            if isinstance(value, str):
                c.data_type = "s"
        ws.row_dimensions[row].height = 24
    ws.sheet_state = "hidden"


def _rebuild_validated_table(table: dict, entries: list, scope: str) -> dict:
    out = dict(table)
    for entry in entries:
        param = entry.param or {}
        if not entry.pid.startswith("V_") or not param.get("raw_id"):
            continue
        if entry.scope not in ("", scope):
            continue
        guard = f"ISNUMBER({entry.cell})"
        out[param["raw_id"]] = (entry.cell, guard)
        out[entry.pid] = (entry.cell, guard)
    return out


def _rewrite_validated_formulas(ws, entries: list, base: dict,
                                 scope_tables: dict) -> None:
    for scope in ("", "CSA", "HCA"):
        table = scope_tables.get(scope) or base
        vtable = _rebuild_validated_table(table, entries, scope)
        for entry in entries:
            if entry.scope != scope or entry.kind != "param":
                continue
            if not entry.formula or entry.pid.startswith("V_"):
                continue
            v1._write_param_cell(ws, entry, vtable)


def _write_v2_main_sheet(ws, spec: dict, sheet: dict, plan: _Plan,
                         config: dict, raw_cells: dict) -> None:
    v1._prep_span(ws, 1, 1, 12, v1._FILL_TITLE)
    title = ws.cell(row=1, column=1)
    model = spec.get("model") or {}
    model_label = model.get("display_name") or model.get("model_key") or "DeepSeek"
    title.value = f"{model_label} · {sheet['title']} · layout-v2"
    title.font = v1._FONT_TITLE
    title.alignment = v1._ALIGN_CENTER
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=12)
    ws.row_dimensions[1].height = 26
    v1._span_text(ws, 2, 1, 12,
                  "batch=1 · element/ms · 每行一个共用/差异模板；成本×真实使用次数；"
                  "未知* = 必要输入/吞吐未定或非法（原因见批注/辅助明细）。",
                  font=v1._FONT_SMALL)

    scenario_start = 4
    input_table, scenario_next = _write_v2_scenario(ws, scenario_start, sheet)
    engine_refs, hw_table = v1._write_hardware(ws, scenario_start, sheet)
    input_table.update(hw_table)
    cube_ref, _cube_guard = engine_refs["cube"]
    vec_ref, _vec_guard = engine_refs["vec"]

    param_start = max(scenario_next, scenario_start + 5) + 1
    layer_cells: dict[str, str] = {}
    base, next_row, scope_tables = v1._write_param_table(
        ws, param_start, spec, config["entries"], input_table, layer_cells)
    _overwrite_base_cells(ws, config["entries"], raw_cells)
    _overwrite_validated_cells(ws, config["entries"], base, scope_tables)
    _rewrite_validated_formulas(ws, config["entries"], base, scope_tables)
    config.update({"base": base, "scope_tables": scope_tables,
                   "input": input_table, "layer_cells": layer_cells,
                   "cube_ref": cube_ref, "vec_ref": vec_ref})

    header_row = next_row + 1
    v1._header(ws, header_row)
    row = header_row + 1
    for block, usage, templates in plan.blocks:
        usage_cell = f"'模板映射'!$C${plan.usage_rows[block]}"
        # 区块标题只写 A 列一次；不铺横向横幅。
        v1._span_text(ws, row, 1, 1, f"{block}（×{usage} 次）",
                      font=v1._FONT_BOLD, fill=v1._FILL_SECTION)
        ws.row_dimensions[row].height = 28
        row += 1
        start_row = row
        for template in templates:
            table = (scope_tables.get(v1._scope_of_group(template.group["id"]))
                     if v1._scope_of_group(template.group["id"]) else base)
            table = table or base
            template.main_row = row
            _write_v2_template_row(ws, row, template, table, input_table,
                                   cube_ref, vec_ref, plan)
            row += 1
        # 模板小计 × 使用次数
        refs_j = [f"J{r}" for r in range(start_row, row)]
        v1._span_text(ws, row, 1, 1, f"模板小计 × {usage}", font=v1._FONT_BOLD,
                      fill=v1._FILL_SUB)
        for col, letter in ((10, "J"), (11, "K"), (12, "L")):
            joined = ",".join(f"{letter}{r}" for r in range(start_row, row))
            formula = (f'=IF(COUNT({joined})={len(refs_j)},'
                       f'SUM({joined})*{usage_cell},"未知*")')
            v1._span_formula(ws, row, col, col, formula, fmt=_FMT_TOTAL,
                             fill=v1._FILL_SUB)
        ws.row_dimensions[row].height = 24
        row += 1

    # 整网总计：各块小计再相加；块小计已含使用次数。
    block_total_rows = []
    cursor = header_row + 1
    for block, _usage, templates in plan.blocks:
        cursor += 1
        cursor += len(templates)
        block_total_rows.append(cursor)
        cursor += 1
    v1._span_text(ws, row, 1, 1, "整网总计", font=v1._FONT_BOLD, fill=v1._FILL_TOTAL)
    for col, letter in ((10, "J"), (11, "K"), (12, "L")):
        cells = [f"{letter}{r}" for r in block_total_rows]
        joined = ",".join(cells)
        formula = (f'=IF(COUNT({joined})={len(cells)},SUM({joined}),"未知*")')
        v1._span_formula(ws, row, col, col, formula, fmt=_FMT_TOTAL,
                         fill=v1._FILL_TOTAL)
    ws.row_dimensions[row].height = 24
    config["final_row"] = row
    config["block_total_rows"] = block_total_rows
    ws.freeze_panes = "C1"
    ws.print_title_rows = f"{header_row}:{header_row}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    for index, width in enumerate(MAIN_WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width


def _write_v2_scenario(ws, start_row: int, sheet: dict) -> tuple[dict, int]:
    items = {item["id"]: item for item in sheet["inputs"]}
    v1._span_text(ws, start_row, 1, 2, "场景输入（黄色可改；batch 固定 1）",
                  font=v1._FONT_SECTION, fill=v1._FILL_SECTION)
    v1._span_text(ws, start_row, 3, 4, "值", font=v1._FONT_HEAD, fill=v1._FILL_HEAD)
    v1._span_text(ws, start_row, 5, 7, "说明", font=v1._FONT_HEAD, fill=v1._FILL_HEAD)
    table: dict = {}
    row = start_row + 1
    for iid in ("seqlen", "kvcachelen", "expert_hit"):
        item = items[iid]
        v1._span_text(ws, row, 1, 2, item["label"], font=v1._FONT_BOLD)
        ref = f"$C${row}"
        if iid == "expert_hit" and item.get("value") is None:
            v1._span_text(ws, row, 3, 4, "", fill=v1._FILL_INPUT)
        else:
            # 不套整数格式：布尔输入在部分重算器里会被转成 1，丢失类型。
            v1._span_number(ws, row, 3, 4, item["value"], fill=v1._FILL_INPUT)
        short_note = {
            "seqlen": "示例 8，可改；本调用新 token 数。",
            "kvcachelen": "示例 0，可改；调用前逻辑历史长度。",
            "expert_hit": "留空=未知；填写后启用路由专家切片。",
        }.get(iid, "")
        v1._span_text(ws, row, 5, 7, short_note, font=v1._FONT_SMALL)
        guard = (f"IFERROR(AND(ISNUMBER({ref}),NOT(ISLOGICAL({ref})),"
                 f"INT({ref})={ref},{ref}>=0),FALSE)")
        table[item["id"]] = (ref, guard)
        table[item["symbol"]] = (ref, guard)
        row += 1
    v1._span_text(ws, row, 1, 2, "batch（固定）", font=v1._FONT_BOLD)
    v1._span_text(ws, row, 3, 4, "固定 1", fill=v1._FILL_EXCLUDED)
    v1._span_text(ws, row, 5, 7, "计算源直接使用常量 1，不引用本格。",
                  font=v1._FONT_SMALL)
    table["B"] = ("1", "TRUE")
    table["batch_size"] = ("1", "TRUE")
    row += 1
    return table, row


def _write_v2_template_row(ws, row: int, template: _Template, table: dict,
                           input_table: dict, cube_ref: str, vec_ref: str,
                           plan: _Plan) -> None:
    rule = template.rule
    kind = rule["class"]
    v1._span_text(ws, row, 1, 1, "", font=v1._FONT_SMALL)
    v1._span_text(ws, row, 2, 2, template.op["id"], font=v1._FONT_BOLD)
    comment = (f"{template.op.get('name', '')}\n覆盖：{', '.join(template.groups)}"
               f"\n使用次数：{template.usage}\n代表层组：{template.group['id']}")
    ws.cell(row=row, column=2).comment = Comment(comment, "E01-v2")
    ws.row_dimensions[row].height = 24
    repeat = rule.get("repeat", 1)
    repeat = int(repeat) if isinstance(repeat, (int, str)) else 1
    composite = len(rule.get("terms", [])) > 1
    if composite:
        v1._span_text(ws, row, 3, 3, "CUBE（复合）" if kind == "cube" else "VEC（复合）",
                      font=v1._FONT_SMALL, align=v1._ALIGN_CENTER)
        for col in (4, 5, 6):
            v1._span_text(ws, row, col, col, "复合", align=v1._ALIGN_CENTER)
        for col in (7, 8, 9):
            v1._span_text(ws, row, col, col, "见明细", align=v1._ALIGN_CENTER)
        aux_cell = f"'项分解'!$H${plan.composite_cells[(ws.title, template.key)]}"
        j_formula, _ = _v2_cost_cell("cube", row, repeat, cube_ref, vec_ref,
                                     composite_cell=aux_cell)
        v1._span_formula(ws, row, 10, 10, j_formula, fmt=v1._FMT_MS)
        v1._span_number(ws, row, 11, 11, 0, fmt=v1._FMT_MS)
    else:
        dims = _v2_dims(template, table, input_table, row)
        v1._span_text(ws, row, 3, 3,
                      ("CUBE" if kind == "cube" else "VEC")
                      + (f"（×{repeat}）" if repeat != 1 else ""),
                      font=v1._FONT_SMALL, align=v1._ALIGN_CENTER)
        _write_v2_dim(ws, row, 4, dims["n"], fmt=v1._FMT_INT)
        _write_v2_dim(ws, row, 5, dims["T"], fmt=v1._FMT_INT)
        _write_v2_dim(ws, row, 6, dims["D"], fmt=v1._FMT_INT)
        if kind == "cube":
            _write_v2_dim(ws, row, 7, dims["M"], fmt=v1._FMT_INT)
            _write_v2_dim(ws, row, 8, dims["K"], fmt=v1._FMT_INT)
            _write_v2_dim(ws, row, 9, dims["N"], fmt=v1._FMT_INT)
        else:
            for col in (7, 8, 9):
                v1._span_text(ws, row, col, col, "—", align=v1._ALIGN_CENTER)
        if kind == "cube":
            j_formula, _ = _v2_cost_cell("cube", row, repeat, cube_ref, vec_ref,
                                         guard_cell=_guard_cell(template, table),
                                         cond_cell=_cond_cell(template, table))
            v1._span_formula(ws, row, 10, 10, j_formula, fmt=v1._FMT_MS)
            v1._span_number(ws, row, 11, 11, 0, fmt=v1._FMT_MS)
        else:
            k_formula, _ = _v2_cost_cell("vec", row, repeat, cube_ref, vec_ref,
                                         guard_cell=_guard_cell(template, table),
                                         cond_cell=_cond_cell(template, table))
            v1._span_number(ws, row, 10, 10, 0, fmt=v1._FMT_MS)
            v1._span_formula(ws, row, 11, 11, k_formula, fmt=v1._FMT_MS)
    v1._span_formula(ws, row, 12, 12,
                     f'=IF(AND(ISNUMBER(J{row}),ISNUMBER(K{row})),MAX(J{row},K{row}),"未知*")',
                     fmt=v1._FMT_MS)


def _guard_cell(template: _Template, table: dict) -> str | None:
    if not template.rule.get("guards"):
        return None
    name = "Nhit_ok_hash" if template.usage <= 3 else "Nhit_ok_topk"
    return table[name][0]


def _cond_cell(template: _Template, table: dict) -> str | None:
    if not template.rule.get("conditional"):
        return None
    return table["H_bias_on"][0]

# ---------------------------------------------------------------- 辅助区

def _write_v2_mapping(wb, spec: dict, plan: _Plan, comparison: dict) -> None:
    ws = wb.create_sheet("模板映射")
    widths = (22, 26, 10, 30, 36, 30, 34, 18)
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width
    v1._prep_span(ws, 1, 1, 8, v1._FILL_TITLE)
    cell = ws.cell(row=1, column=1)
    cell.value = "模板映射 / 使用次数 / 覆盖与成本对照（结构来源与限制见本表说明）"
    cell.font = v1._FONT_TITLE
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=8)
    ws.row_dimensions[1].height = 26
    v1._span_text(ws, 2, 1, 8,
                  "结构来自冻结 layer_structures.json / network_overview.json；"
                  "同构判定覆盖算法/规则、维度映射、参数作用域、条件、排除计费与 repeat。"
                  "excluded 层组项只在整网入口收一次；container 不计。"
                  "未知* = 必要输入/吞吐未定或非法。", font=v1._FONT_SMALL)

    v1._span_text(ws, 4, 1, 1, "模板块", font=v1._FONT_HEAD, fill=v1._FILL_HEAD)
    v1._span_text(ws, 4, 2, 2, "模板数", font=v1._FONT_HEAD, fill=v1._FILL_HEAD)
    v1._span_text(ws, 4, 3, 3, "使用次数（派生，不可改）", font=v1._FONT_HEAD, fill=v1._FILL_HEAD)
    v1._span_text(ws, 4, 4, 4, "次数推导", font=v1._FONT_HEAD, fill=v1._FILL_HEAD)
    v1._span_text(ws, 4, 5, 8, "覆盖层型", font=v1._FONT_HEAD, fill=v1._FILL_HEAD)
    for block, usage, templates in plan.blocks:
        row = plan.usage_rows[block]
        v1._span_text(ws, row, 1, 1, block, font=v1._FONT_BOLD)
        v1._span_number(ws, row, 2, 2, len(templates), fmt=v1._FMT_INT)
        v1._span_number(ws, row, 3, 3, usage, fmt=v1._FMT_INT, fill=v1._FILL_SUB)
        v1._span_text(ws, row, 4, 4, _block_derivation(block), font=v1._FONT_SMALL)
        groups = sorted({gid for t in templates for gid in t.groups})
        v1._span_text(ws, row, 5, 8, ", ".join(groups), font=v1._FONT_SMALL)

    row = max(plan.usage_rows.values()) + 2
    headers = ("模板块", "模板 op", "使用次数", "覆盖层组", "原 op 实例",
               "等价性摘要", "说明")
    for index, label in enumerate(headers, start=1):
        c = ws.cell(row=row, column=index, value=label)
        c.font = v1._FONT_HEAD
        c.fill = v1._FILL_HEAD
        c.border = v1._BORDER
        c.alignment = v1._ALIGN_CENTER
    row += 1
    for template in plan.templates:
        param_hash = hashlib.sha1(repr(template.params_sig).encode("utf-8")).hexdigest()[:8]
        summary = (f"{template.family} / {template.rule['class']} / "
                   f"terms={len(template.rule.get('terms', []))} / "
                   f"repeat={template.rule.get('repeat', 1)} / "
                   f"cond={bool(template.rule.get('conditional'))} / "
                   f"param_sig={param_hash}")
        values = (template.block, template.op["id"], template.usage,
                  ", ".join(template.groups),
                  ", ".join(f"{g}:{o}" for g, o in template.instances),
                  summary, "共用一个模板；成本×使用次数")
        for index, value in enumerate(values, start=1):
            c = ws.cell(row=row, column=index, value=value)
            c.font = v1._FONT_SMALL
            c.border = v1._BORDER
            c.alignment = v1._ALIGN_WRAP
            if isinstance(value, str):
                c.data_type = "s"
        ws.row_dimensions[row].height = 26
        row += 1

    # 逐实例映射：完整覆盖 291 个原 op 实例（含 excluded/container）
    row += 1
    v1._span_text(ws, row, 1, 8, "逐实例映射（原 op → 模板/排除/容器）",
                  font=v1._FONT_SECTION, fill=v1._FILL_SECTION)
    row += 1
    instance_headers = ("原 op", "层组", "真实层数", "映射类型",
                        "模板块", "模板 op", "使用次数贡献", "说明")
    for index, label in enumerate(instance_headers, start=1):
        c = ws.cell(row=row, column=index, value=label)
        c.font = v1._FONT_HEAD
        c.fill = v1._FILL_HEAD
        c.border = v1._BORDER
        c.alignment = v1._ALIGN_CENTER
    row += 1
    groups = {g["id"]: g for g in spec["sheets"][0]["groups"]}
    excluded_set = set(plan.excluded)
    container_set = set(plan.container)
    for gid, oid in plan.all_instances:
        count = _group_count(groups[gid])
        if (gid, oid) in plan.instance_map:
            template = plan.instance_map[(gid, oid)]
            kind = "共用/差异模板"
            block = template.block
            mapped = template.op["id"]
            contribution = template.usage
            note = f"同构模板；该实例按 {count} 层计入"
        elif (gid, oid) in excluded_set:
            kind = "excluded"
            block = "整网一次性"
            mapped = "—"
            contribution = 0
            note = "层组只引用；整网入口收一次，不计入模板"
        elif (gid, oid) in container_set:
            kind = "container"
            block = "容器"
            mapped = "—"
            contribution = 0
            note = "容器不计成本"
        else:
            kind = "未映射"
            block = "—"
            mapped = "—"
            contribution = 0
            note = "未映射（应不会出现）"
        values = (oid, gid, count, kind, block, mapped, contribution, note)
        for index, value in enumerate(values, start=1):
            c = ws.cell(row=row, column=index, value=value)
            c.font = v1._FONT_SMALL
            c.border = v1._BORDER
            c.alignment = v1._ALIGN_WRAP
            if isinstance(value, str):
                c.data_type = "s"
        row += 1

    # 覆盖与成本对照
    row += 1
    v1._span_text(ws, row, 1, 8, "覆盖校验与成本对照（N_hit=6，当前场景）",
                  font=v1._FONT_SECTION, fill=v1._FILL_SECTION)
    row += 1
    for label, value in (
            ("原算子实例", plan.coverage["original_instances"]),
            ("可计费实例", plan.coverage["billable_instances"]),
            ("excluded 实例", plan.coverage["excluded_instances"]),
            ("container 实例", plan.coverage["container_instances"]),
            ("模板数", plan.coverage["templates"]),
            ("使用次数合计", plan.coverage["actual_usage"])):
        v1._span_text(ws, row, 1, 2, label, font=v1._FONT_SMALL)
        v1._span_number(ws, row, 3, 3, value, fmt=v1._FMT_INT)
        row += 1
    row += 1
    header = ("场景", "口径", "前轮参考", "layout-v2 模板", "差额")
    for index, label in enumerate(header, start=1):
        c = ws.cell(row=row, column=index, value=label)
        c.font = v1._FONT_HEAD
        c.fill = v1._FILL_HEAD
        c.border = v1._BORDER
        c.alignment = v1._ALIGN_CENTER
    row += 1
    for scene, values in comparison["reference"].items():
        for key in ("cube", "vec", "op"):
            ref = values[key]
            got = comparison["templates"][scene][key]
            v1._span_text(ws, row, 1, 1, scene, font=v1._FONT_SMALL)
            v1._span_text(ws, row, 2, 2, key, font=v1._FONT_SMALL)
            v1._span_number(ws, row, 3, 3, ref, fmt=v1._FMT_MS)
            v1._span_number(ws, row, 4, 4, got, fmt=v1._FMT_MS)
            v1._span_number(ws, row, 5, 5, got - ref, fmt=v1._FMT_MS)
            row += 1
    row += 1
    notes = [
        "前轮参考 = 按原四层组逐算子规则、真实层数、N_hit=6 直接求和；"
        "layout-v2 = 模板单次成本×使用次数；两者 cube/vec/op 必须一致。",
        "mHC、RMSNorm、残差混合按 attn/ffn 同构模板各计每层 2 次；"
        "QKV/O 各计每层 1 次；实际执行次数没有漏算。",
        "SWA/CSA/HCA 的缓存、压缩器/Indexer、Attention 核心按差异拆分；"
        "CSA-hash 与 CSA-topk 的同构核心合为 21 层，HCA 单独 20 层。",
        "Hash/TopK 路由与专家切片按 moe_type/守卫分开；共享专家与 MoE 专家模板计 43 层。",
        "excluded 项（mask_model_build/rope_main/rope_compress）不进入模板；"
        "整网入口的 net_mask_build/net_rope_* 各收一次；container 不计成本。",
    ]
    for text in notes:
        v1._span_text(ws, row, 1, 8, "· " + text, font=v1._FONT_SMALL)
        row += 1


def _write_v2_params(wb, entries: list) -> None:
    ws = wb.create_sheet("参数与来源")
    widths = (18, 16, 12, 46, 70, 18)
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width
    v1._prep_span(ws, 1, 1, 6, v1._FILL_TITLE)
    cell = ws.cell(row=1, column=1)
    cell.value = "参数与来源（基础参数 + 派生维度 + 检查；作用域保留）"
    cell.font = v1._FONT_TITLE
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=6)
    headers = ("参数id", "符号", "作用域", "公式/值", "说明/来源", "类别")
    for index, label in enumerate(headers, start=1):
        c = ws.cell(row=3, column=index, value=label)
        c.font = v1._FONT_HEAD
        c.fill = v1._FILL_HEAD
        c.border = v1._BORDER
        c.alignment = v1._ALIGN_CENTER
    row = 4
    for entry in entries:
        param = entry.param or {}
        formula = entry.formula or param.get("expression") or (
            str(entry.value) if entry.value is not None else "未知（未提供）")
        note = entry.note or param.get("note", "")
        values = (entry.pid, entry.symbol, entry.scope or "通用", formula,
                  note, entry.category)
        for index, value in enumerate(values, start=1):
            c = ws.cell(row=row, column=index, value=value)
            c.font = v1._FONT_SMALL
            c.border = v1._BORDER
            c.alignment = v1._ALIGN_WRAP
            if isinstance(value, str):
                c.data_type = "s"
        ws.row_dimensions[row].height = 28
        row += 1


def _write_v2_terms(wb, spec: dict, plan: _Plan, configs: dict) -> None:
    ws = wb.create_sheet("项分解")
    widths = (10, 24, 22, 14, 28, 22, 26, 20, 8, 46)
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width
    v1._prep_span(ws, 1, 1, 10, v1._FILL_TITLE)
    cell = ws.cell(row=1, column=1)
    cell.value = "项分解：复合矩阵 SUM、同形 repeat、excluded/container 追溯（不改主表成本）"
    cell.font = v1._FONT_TITLE
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=10)
    headers = ("场景", "模板块", "模板 op", "项", "M", "K", "N", "工作量",
               "repeat", "说明")
    for index, label in enumerate(headers, start=1):
        c = ws.cell(row=4, column=index, value=label)
        c.font = v1._FONT_HEAD
        c.fill = v1._FILL_HEAD
        c.border = v1._BORDER
        c.alignment = v1._ALIGN_CENTER

    # 复合项：先写在主表引用到的固定行
    for (scene, key), rows in plan.composite_rows.items():
        template = next(t for t in plan.templates if t.key == key)
        config = configs[scene]
        scope = v1._scope_of_group(template.group["id"])
        table = config["scope_tables"].get(scope) or config["base"]
        input_table = config["input"]
        for offset, term in enumerate(template.rule["terms"]):
            row = rows[offset]
            m, k, n = _v2_term_dims(term, table, input_table, f"{scene}!")
            work = _v2_term_work_formula(term, table, input_table, f"{scene}!")
            values = (scene, template.block, template.op["id"],
                      term.get("label", f"term{offset + 1}"),
                      "=" + m, "=" + k, "=" + n, "=" + work,
                      template.rule.get("repeat", 1), "复合项；主表 J 引用本组总工作量")
            for index, value in enumerate(values, start=1):
                c = ws.cell(row=row, column=index, value=value)
                c.font = v1._FONT_SMALL
                c.border = v1._BORDER
                c.alignment = v1._ALIGN_WRAP
                if isinstance(value, str) and not value.startswith("="):
                    c.data_type = "s"
            ws.row_dimensions[row].height = 26
        total_row = plan.composite_cells[(scene, key)]
        work_refs = [f"H{r}" for r in rows]
        values = (scene, template.block, template.op["id"], "合计", "—", "—", "—",
                  "=IFERROR(" + "+".join(work_refs) + ',"未知*")',
                  template.rule.get("repeat", 1), "供主表 J 直接引用")
        for index, value in enumerate(values, start=1):
            c = ws.cell(row=total_row, column=index, value=value)
            c.font = v1._FONT_BOLD
            c.border = v1._BORDER
            c.alignment = v1._ALIGN_WRAP
            if isinstance(value, str) and not value.startswith("="):
                c.data_type = "s"
        ws.row_dimensions[total_row].height = 24

    # repeat / excluded / container 追溯
    row = max([r for rows in plan.composite_rows.values() for r in rows] + [4]) + 2
    for scene in ("Prefill", "Decode"):
        config = configs[scene]
        for template in plan.templates:
            repeat = template.rule.get("repeat", 1)
            if repeat in (None, 1, "1") or len(template.rule.get("terms", [])) > 1:
                continue
            scope = v1._scope_of_group(template.group["id"])
            table = config["scope_tables"].get(scope) or config["base"]
            term = template.rule["terms"][0]
            m, k, n = _v2_term_dims(term, table, config["input"], f"{scene}!")
            work = _v2_term_work_formula(term, table, config["input"], f"{scene}!")
            values = (scene, template.block, template.op["id"], "同形 repeat",
                      "=" + m, "=" + k, "=" + n, f"=({work})*{repeat}", repeat,
                      "主表 J 行内乘 repeat；本行只追溯")
            for index, value in enumerate(values, start=1):
                c = ws.cell(row=row, column=index, value=value)
                c.font = v1._FONT_SMALL
                c.border = v1._BORDER
                c.alignment = v1._ALIGN_WRAP
                if isinstance(value, str) and not value.startswith("="):
                    c.data_type = "s"
            ws.row_dimensions[row].height = 26
            row += 1
    for gid, oid in plan.excluded + plan.container:
        values = ("Prefill/Decode", "整网一次性/容器", oid,
                  "excluded" if (gid, oid) in plan.excluded else "container",
                  "—", "—", "—", "—", 1,
                  f"{gid}：层组只引用；整网入口收一次或容器不计")
        for index, value in enumerate(values, start=1):
            c = ws.cell(row=row, column=index, value=value)
            c.font = v1._FONT_SMALL
            c.border = v1._BORDER
            c.alignment = v1._ALIGN_WRAP
            if isinstance(value, str):
                c.data_type = "s"
        row += 1


# ---------------------------------------------------------------- 入口

def build_perf_layout_v2_workbook(container: dict, network: dict | None = None) -> bytes:
    spec = build_structures_workbook_spec(container, network)
    plan = _build_plan(spec)
    comparison = _compare_costs(plan, spec, n_hit=6)
    entries, _by_key, _mdep = _build_v2_entries(spec)
    raw_cells = _raw_cells(entries)
    wb = Workbook()
    wb.iso_dates = True
    wb.remove(wb.active)
    configs: dict[str, dict] = {}
    for sheet in spec["sheets"]:
        ws = wb.create_sheet(sheet["name"])
        config = {"entries": entries}
        _write_v2_main_sheet(ws, spec, sheet, plan, config, raw_cells)
        configs[sheet["name"]] = config
    _write_v2_mapping(wb, spec, plan, comparison)
    _write_v2_params(wb, entries)
    _write_v2_terms(wb, spec, plan, configs)
    _write_raw_sheet(wb, entries, raw_cells)
    buffer = io.BytesIO()
    wb.save(buffer)
    return _normalize_zip(buffer.getvalue())


def export_perf_layout_v2_workbook(container: dict, network: dict | None = None) -> bytes:
    return build_perf_layout_v2_workbook(container, network)
