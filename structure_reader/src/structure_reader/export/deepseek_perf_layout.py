"""DeepSeek-V4-Flash 的 12 列 Excel 性能样表（E01 / GitHub #22）。

本模块只做 **Excel 展示布局适配**：消费已有的
``hf_workbook.build_structures_workbook_spec``（真实结构 + 同一套 CUBE/VEC 规则），
把每页重排为：

- 顶部左侧集中参数、右侧独立硬件吞吐；
- 主表固定 12 列：结构/通路、op、op type、n、T、D、M、K、N、cube(ms)、
  vector(ms)、op(ms)；
- 每行一个算子；CUBE 工作量 = M×K×N（head/分组/矩阵 batch 已折入有效 M，
  不再乘 n）；VEC 工作量 = n×T×D；op(ms)=MAX(cube,vector)；
- 复合矩阵项先 SUM / 乘 repeat，再除以 CUBE 吞吐；各项在「辅助明细」展开；
- 未知/非法输入保持“未知（原因）”，不参与 SUM/MAX 被吞成 0；
- 层组单层小计 × 真实层数、整网按 serial_order 串行汇总，excluded/container
  不重复计费。

公式编译、输入/参数守卫和未知文本复用
``export/operator_workbook.py`` 的同一套辅助函数，不引入第二套求值语义。
旧 27 列导出入口（``export_structures_workbook``）保持原行为；本模块是独立入口。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from structure_reader.export.hf_workbook import build_structures_workbook_spec
from structure_reader.export.operator_workbook import (
    _FORMULA_LIMIT,
    _and_guard,
    _formula,
    _input_guard,
    _perf_compile,
    _perf_dim,
    _normalize_zip,
    _perf_unknown_text,
    _text,
    _xl_quote,
)
from structure_reader.scenario_evaluator import ExpressionError, excel_formula, expression_names

__all__ = [
    "LAYOUT_SCHEMA",
    "MAIN_HEADERS",
    "build_perf_layout_workbook",
    "export_perf_layout_workbook",
]

LAYOUT_SCHEMA = "deepseek_perf_layout.v1"

MAIN_HEADERS = (
    "结构/通路",
    "op",
    "op type",
    "n",
    "T seqlen or N",
    "hiddensize or D",
    "M",
    "K",
    "N",
    "cube(ms)",
    "vector(ms)",
    "op(ms)",
)
MAIN_WIDTHS = (17, 27, 14, 7, 15, 17, 14, 12, 15, 14, 13, 13)

# ---------------------------------------------------------------- 样式

_THIN = Side(style="thin", color="FFBFBFBF")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

_FONT_NAME = "宋体"
_FONT_TITLE = Font(name=_FONT_NAME, size=15, bold=True, color="FF1F1F1F")
_FONT_SUB = Font(name=_FONT_NAME, size=10, color="FF404040")
_FONT_SECTION = Font(name=_FONT_NAME, size=11, bold=True, color="FF1F1F1F")
_FONT_HEAD = Font(name=_FONT_NAME, size=11, bold=True, color="FF1F1F1F")
_FONT_BODY = Font(name=_FONT_NAME, size=11, color="FF1F1F1F")
_FONT_SMALL = Font(name=_FONT_NAME, size=9, color="FF595959")
_FONT_BOLD = Font(name=_FONT_NAME, size=11, bold=True, color="FF1F1F1F")

_FILL_TITLE = PatternFill("solid", fgColor="FFEDF2F9")
_FILL_SECTION = PatternFill("solid", fgColor="FFDDE7F3")
_FILL_GROUP = PatternFill("solid", fgColor="FFEAF1F8")
_FILL_HEAD = PatternFill("solid", fgColor="FFF2F5F9")
_FILL_SUB = PatternFill("solid", fgColor="FFF7F7F7")
_FILL_TOTAL = PatternFill("solid", fgColor="FFE2EFDA")
_FILL_INPUT = PatternFill("solid", fgColor="FFFFF2CC")
_FILL_NOTE = PatternFill("solid", fgColor="FFFFF9E6")
_FILL_UNKNOWN = PatternFill("solid", fgColor="FFFCE4D6")
_FILL_EXCLUDED = PatternFill("solid", fgColor="FFF2F2F2")

_ALIGN_WRAP = Alignment(wrap_text=True, vertical="top")
_ALIGN_TOP = Alignment(vertical="top")
_ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
_ALIGN_RIGHT = Alignment(horizontal="right", vertical="top")

_FMT_MS = "#,##0.000000"
_FMT_INT = "#,##0"
_FMT_NUM = "0.000000"

# 参数分类顺序与短标签（只影响展示，不新增/改变模型事实）。
_CATEGORY_ORDER = ("真实层结构", "基础维度", "Attention/Indexer", "MoE/FFN", "mHC/其他", "整网")
_CATEGORY_OF = {
    **{pid: "基础维度" for pid in ("H", "Nh", "Nkv", "Dh", "Rq", "Rk", "G", "Ro", "W", "Nhc")},
    **{pid: "Attention/Indexer" for pid in (
        "Lkv", "Lmask", "Lkvp", "M", "Ki", "Ih", "Id", "Nwin", "Nwin_i", "Ncomp",
        "Ncompi", "Tbuf", "Tbufp", "Tbufi", "Tbufip", "Tcc", "Tci", "Fwp", "Fut",
        "Fqt", "Kisel", "Bbias")},
    **{pid: "MoE/FFN" for pid in ("E", "K", "Es", "Lim", "Scl")},
    **{pid: "mHC/其他" for pid in ("Sit", "Eps", "EpsN")},
    **{pid: "整网" for pid in ("Vocab", "Nlay", "Lk", "Ssel", "ThetaM", "ThetaC", "Yf")},
}
_PARAM_LABEL = {
    "H": "hidden_size",
    "Nh": "num_attention_heads",
    "Nkv": "num_key_value_heads",
    "Dh": "head_dim",
    "Rq": "q_lora_rank",
    "Rk": "qk_rope_head_dim",
    "G": "o_groups",
    "Ro": "o_lora_rank",
    "W": "sliding_window",
    "Nhc": "hc_mult",
    "Sit": "hc_sinkhorn_iters",
    "Eps": "hc_eps",
    "EpsN": "rms_norm_eps",
    "E": "n_routed_experts",
    "K": "num_experts_per_tok",
    "Es": "moe_intermediate_size",
    "Lim": "swiglu_limit",
    "Scl": "routed_scaling_factor",
    "Lkv": "L_kv 本次可读长度",
    "Lmask": "L_mask 模型级 mask 长度",
    "Lkvp": "L_kv_post 调用后持久长度",
    "Ki": "index_topk",
    "Ih": "index_n_heads",
    "Id": "index_head_dim",
    "Nwin": "N_win 新闭窗数",
    "Nwin_i": "N_win_i Indexer 新闭窗数",
    "Ncomp": "N_comp 压缩条目数",
    "Ncompi": "N_comp_i Indexer 条目数",
    "Tbuf": "Tbuf 窗口余量",
    "Tbufp": "Tbuf' 调用后余量",
    "Tbufi": "Tbuf_i Indexer 余量",
    "Tbufip": "Tbuf_i' Indexer 调用后余量",
    "Tcc": "Tcc 已发射条目数",
    "Tci": "Tci Indexer 已发射条目数",
    "Fwp": "first_window_position",
    "Fut": "future_threshold",
    "Fqt": "first_query_threshold",
    "Kisel": "Ki_sel 实际 top-k",
    "Bbias": "block_bias 是否物化",
    "M": "compress_rate 压缩窗宽",
    "ThetaM": "main RoPE theta",
    "ThetaC": "compress RoPE theta",
    "Yf": "YaRN factor",
    "Vocab": "vocab_size",
    "Nlay": "num_hidden_layers",
    "Lk": "logits_to_keep",
    "Ssel": "logits 位置数",
    "B": "batch（固定 1）",
    "seqlen": "seqlen",
    "kvcachelen": "kvcachelen",
    "N_hit": "N_hit 命中专家数",
    "Tcube": "CUBE 吞吐",
    "Tvec": "VEC 吞吐",
}

_SCOPE_CN = {"": "", "CSA": "CSA", "HCA": "HCA"}

# 直接/传递进入 CUBE/VEC 工作量公式的参数；其余仅作追溯展示。
_COST_RELEVANT = {
    "B", "seqlen", "kvcachelen", "N_hit",
    "H", "Nh", "Dh", "Rq", "Rk", "G", "Ro", "W", "Nhc",
    "Id", "Ih", "M", "Nwin", "Nwin_i", "Ncomp", "Tbuf", "Tbufi", "Tci",
    "Lkv", "Lmask", "Lk", "Ssel", "Vocab",
    "E", "K", "Es",
}


# ---------------------------------------------------------------- 映射


def _split_product(expr: str) -> list[str]:
    """按顶层 ``*`` 拆分表达式；括号内的乘号保留。"""
    expr = str(expr).replace(" ", "")
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "*" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        parts.append(cur)
    return parts


_HEADS = {"Nh", "Ih", "Nkv"}


def _head_factor(factors: list[str]) -> str | None:
    for factor in factors:
        if factor in _HEADS:
            return factor
    return None


def _join_product(factors: list[str]) -> str:
    if not factors:
        return "1"
    return "*".join(f"({f})" if ("+" in f or "-" in f) else f for f in factors)


def _cube_map(term: dict) -> dict:
    """CUBE 项 → 主表 n/T/D/M/K/N。

    ``M`` 是有效 M，已包含 b 中的 batch/head/分组因子；n 只展示 head 或 1，
    不再乘进工作量。T=m、D=K=k、N=规则 N。
    """
    b, m, k, n = term["b"], term["m"], term["k"], term["n"]
    bf = _split_product(b)
    head = _head_factor(bf)
    return {
        "n": head or "1",
        "T": m,
        "D": k,
        "M": m if b == "1" else f"({b})*({m})",
        "K": k,
        "N": n,
        "b": b,
        "m": m,
        "k": k,
        "n_raw": n,
        "head": head,
    }


def _vec_map(term: dict) -> dict:
    """VEC 项 → 主表 n/T/D。

    n = m 中独立 head 因子（Nh/Ih/Nkv）或 1；T = m 去掉 B 与 head 后的其余
    因子乘积（VEC 行数）；D = 规则 n（VEC 列数）。因此 n×T×D = m×n。
    """
    m, n = term["m"], term["n"]
    mf = _split_product(m)
    head = _head_factor(mf)
    rest = [f for f in mf if f != "B" and f != head]
    return {
        "n": head or "1",
        "T": _join_product(rest),
        "D": n,
        "m": m,
        "n_raw": n,
        "head": head,
        "rest": rest,
    }


def _scope_of_group(group_id: str) -> str:
    if group_id == "hca-topk":
        return "HCA"
    if "csa" in group_id:
        return "CSA"
    return ""


def _m_dependent_ids(groups: list[dict]) -> set[str]:
    """传递依赖 M 的参数 id（含 M 自身）；这些参数按 CSA/HCA 作用域分行。

    公式里写的是符号（如 ``N_comp``/``N_win``），这里先把符号映射回参数 id，
    否则 Bbias/Ki_sel 这类“依赖派生参数”的项会被漏掉作用域。
    """
    formulas: dict[str, str] = {}
    symbol_to_id: dict[str, str] = {}
    for group in groups:
        for param in group["parameters"]:
            symbol_to_id[param["symbol"]] = param["id"]
            for alias in param.get("axis_aliases", []):
                symbol_to_id[alias] = param["id"]
            if param.get("formula"):
                formulas.setdefault(param["id"], param["formula"])
    need = {"M"}
    changed = True
    while changed:
        changed = False
        for pid, formula in formulas.items():
            if pid in need:
                continue
            deps = {symbol_to_id.get(name, name) for name in expression_names(formula)}
            if deps & need:
                need.add(pid)
                changed = True
    return need


@dataclass
class _Entry:
    key: str
    pid: str
    label: str
    symbol: str
    category: str
    scope: str = ""
    kind: str = "param"          # param | layer | layer_total
    param: dict | None = None
    value: Any = None
    formula: str | None = None   # 白名单表达式（不带 =）
    note: str = ""
    row: int = 0
    col: int = 0
    cell: str = ""
    guard: str = "FALSE"
    layer_group: str = ""
    value_span: int = 3


def _category_for(pid: str) -> str:
    return _CATEGORY_OF.get(pid, "其他")


def _label_for(param: dict) -> str:
    pid = param["id"]
    base = _PARAM_LABEL.get(pid, param.get("meaning") or pid)
    return base


def _build_entries(spec: dict) -> tuple[list[_Entry], dict[str, _Entry], set[str]]:
    sheet = spec["sheets"][0]
    groups = sheet["groups"]
    mdep = _m_dependent_ids(groups)
    entries: dict[str, _Entry] = {}
    for group in groups:
        for param in group["parameters"]:
            pid = param["id"]
            if pid in ("B", "seqlen", "kvcachelen"):
                continue
            scope = _scope_of_group(group["id"]) if pid in mdep else ""
            key = f"{pid}@{scope}" if scope else pid
            if key in entries:
                continue
            label = _label_for(param)
            if scope:
                label = f"{label}（{_SCOPE_CN[scope]}）"
            entries[key] = _Entry(
                key=key, pid=pid, label=label, symbol=param["symbol"],
                category=_category_for(pid), scope=scope, param=param,
                formula=param.get("formula"), value=param.get("value"),
                note=param.get("note", ""),
            )
    # 真实层数：不是模型参数，但汇总公式需要可引用的固定单元格。
    total_layers = 0
    for group in groups:
        if group["id"] == "net-periphery":
            continue
        count = len(group["layer_indices"])
        total_layers += count
        key = f"layer_count@{group['id']}"
        entries[key] = _Entry(
            key=key, pid=key, label=f"{group['label']} 真实层数",
            symbol=f"Nlay_{group['id'].replace('-', '_')}",
            category="真实层结构", kind="layer", value=count,
            layer_group=group["id"],
        )
    entries["layer_count@total"] = _Entry(
        key="layer_count@total", pid="layer_count@total", label="43 层合计",
        symbol="Nlay_total", category="真实层结构", kind="layer_total",
        value=total_layers,
    )
    expected = sheet["network"]["layer_count"] if sheet.get("network") else total_layers
    if total_layers != expected:
        raise ValueError(f"真实层组合计 {total_layers} 与整网 layer_count {expected} 不一致")

    order = {name: i for i, name in enumerate(_CATEGORY_ORDER)}
    layer_order = {"swa-hash": 0, "csa-hash": 1, "hca-topk": 2, "csa-topk": 3}

    def _sort_key(e: _Entry):
        if e.category == "真实层结构":
            return (0, layer_order.get(e.layer_group, 4),
                    0 if e.kind == "layer" else 1, e.key)
        return (order.get(e.category, 99), 1, 0, e.key)

    ordered = sorted(entries.values(), key=_sort_key)
    by_key = {e.key: e for e in ordered}
    return ordered, by_key, mdep


def _add_symbol(table: dict, key: str, cell: str, guard: str) -> None:
    table[key] = (cell, guard)


def _add_entry_symbols(table: dict, entry: _Entry) -> None:
    if entry.kind != "param":
        return
    _add_symbol(table, entry.pid, entry.cell, entry.guard)
    _add_symbol(table, entry.symbol, entry.cell, entry.guard)
    if entry.param:
        for alias in entry.param.get("axis_aliases", []):
            _add_symbol(table, alias, entry.cell, entry.guard)


# ---------------------------------------------------------------- 单元格辅助


def _span(ws, row: int, c1: int, c2: int) -> None:
    if c2 > c1:
        ws.merge_cells(start_row=row, start_column=c1, end_row=row, end_column=c2)


def _prep_span(ws, row: int, c1: int, c2: int, fill=None) -> None:
    for col in range(c1, c2 + 1):
        cell = ws.cell(row=row, column=col)
        cell.border = _BORDER
        if fill is not None:
            cell.fill = fill


def _span_text(ws, row: int, c1: int, c2: int, value, *, font=None,
               fill=None, align=None) -> None:
    _prep_span(ws, row, c1, c2, fill)
    cell = ws.cell(row=row, column=c1)
    cell.value = "" if value is None else str(value)
    cell.data_type = "s"
    cell.font = font or _FONT_BODY
    cell.alignment = align or _ALIGN_WRAP
    _span(ws, row, c1, c2)


def _span_number(ws, row: int, c1: int, c2: int, value, *, fmt=None,
                 fill=None) -> None:
    _prep_span(ws, row, c1, c2, fill)
    cell = ws.cell(row=row, column=c1)
    cell.value = value
    cell.font = _FONT_BODY
    cell.alignment = _ALIGN_TOP
    if fmt:
        cell.number_format = fmt
    _span(ws, row, c1, c2)


def _span_formula(ws, row: int, c1: int, c2: int, formula: str, *, fmt=None,
                  fill=None) -> None:
    _prep_span(ws, row, c1, c2, fill)
    cell = ws.cell(row=row, column=c1)
    cell.value = formula
    cell.font = _FONT_BODY
    cell.alignment = _ALIGN_WRAP
    if fmt:
        cell.number_format = fmt
    _span(ws, row, c1, c2)


def _banner(ws, row: int, text: str, *, fill=_FILL_SECTION, font=_FONT_SECTION,
            height: int = 22) -> None:
    for col in range(1, len(MAIN_HEADERS) + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = fill
        cell.border = _BORDER
        cell.alignment = _ALIGN_CENTER
    ws.cell(row=row, column=1).value = text
    ws.cell(row=row, column=1).font = font
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(MAIN_HEADERS))
    ws.row_dimensions[row].height = height


def _header(ws, row: int) -> None:
    for col, label in enumerate(MAIN_HEADERS, start=1):
        cell = ws.cell(row=row, column=col)
        cell.value = label
        cell.font = _FONT_HEAD
        cell.fill = _FILL_HEAD
        cell.border = _BORDER
        cell.alignment = _ALIGN_CENTER
    ws.row_dimensions[row].height = 34


def _compact_layers(indices: list[int]) -> str:
    if not indices:
        return "—"
    if len(indices) <= 6:
        return ",".join(str(i) for i in indices)
    return f"{indices[0]},{indices[1]},…,{indices[-1]}（{len(indices)} 层）"


def _path_label(group_id: str, op: dict) -> str:
    oid = op["id"]
    mp = op.get("module_path") or ""
    if group_id == "net-periphery":
        return "整网·入口" if oid not in ("net_hc_head_norm", "net_hc_head_project",
                                          "net_hc_head_pre", "net_hc_head_collapse",
                                          "net_final_norm", "net_logits_slice",
                                          "net_lm_head") else "整网·出口"
    if oid.startswith("attn_hc_"):
        return "mHC·注意力前"
    if oid == "attn_norm":
        return "Attention·Norm"
    if oid in ("q_a_proj", "q_a_norm", "q_b_proj", "q_b_norm"):
        return "Attention·Q"
    if oid in ("rope_main", "q_rope_apply", "rope_compress", "kv_rope_apply"):
        return "Attention·RoPE"
    if oid in ("kv_proj", "kv_head_view", "cache_update", "mask_model_build", "repeat_kv",
               "kv_concat"):
        return "Attention·KV/Cache"
    if oid.startswith("c_") or oid.startswith("i_"):
        return "压缩器/Indexer"
    if oid in ("qk_scaled", "mask_add", "sink_softmax", "av_matmul", "mask_extend",
               "c_mask_extend", "c_block_bias", "hca_block_bias"):
        return "Attention·核心"
    if oid in ("attn_inv_rope", "attn_group_reshape", "o_a_proj", "o_b_proj"):
        return "Attention·O"
    if oid.startswith("attn_residual_"):
        return "Attention·残差"
    if oid.startswith("ffn_hc_"):
        return "mHC·MoE 前"
    if oid == "ffn_norm":
        return "MoE·Norm"
    if oid in ("moe_flatten", "moe_hash_lookup", "moe_router", "moe_dispatch"):
        return "MoE·路由"
    if oid.startswith("moe_expert_"):
        return "MoE·路由专家"
    if oid in ("moe_combine", "moe_routed_view", "moe_merge"):
        return "MoE·汇合"
    if oid in ("shared_gate_up", "shared_act_down"):
        return "MoE·共享专家"
    if oid.startswith("ffn_residual_"):
        return "FFN·残差"
    if "Attention" in mp:
        return "Attention·其他"
    if "Moe" in mp or "MLP" in mp or "Experts" in mp:
        return "MoE·其他"
    return "其他"


# ---------------------------------------------------------------- 参数区


@dataclass
class _Config:
    """一页的输入/参数/吞吐单元格与符号表。"""
    input_table: dict = field(default_factory=dict)
    engine_refs: dict = field(default_factory=dict)
    entries: list = field(default_factory=list)
    by_key: dict = field(default_factory=dict)
    base_table: dict = field(default_factory=dict)
    scope_tables: dict = field(default_factory=dict)
    layer_cells: dict = field(default_factory=dict)


def _input_guard_value(item: dict, ref: str) -> str:
    return _input_guard(item, ref)


def _write_scenario(ws, start_row: int, sheet: dict) -> tuple[dict, int]:
    """左侧场景输入；返回输入符号表和下一行。"""
    items = {item["id"]: item for item in sheet["inputs"]}
    _span_text(ws, start_row, 1, 2, "场景与配置（黄色可改；batch=1 固定）",
               font=_FONT_SECTION, fill=_FILL_SECTION)
    _span_text(ws, start_row, 3, 4, "值", font=_FONT_HEAD, fill=_FILL_HEAD)
    _span_text(ws, start_row, 5, 7, "说明", font=_FONT_HEAD, fill=_FILL_HEAD)
    row = start_row + 1
    table: dict = {}
    order = ("seqlen", "kvcachelen", "expert_hit", "compute_precision",
             "weight_storage_bits", "activation_storage_bits")
    for iid in order:
        item = items.get(iid)
        if item is None:
            continue
        label = item["label"]
        _span_text(ws, row, 1, 2, label, font=_FONT_BOLD)
        ref = f"$C${row}"
        if iid == "expert_hit":
            if item.get("value") is None:
                _span_text(ws, row, 3, 4, "", fill=_FILL_INPUT)
            else:
                _span_number(ws, row, 3, 4, item["value"], fmt=_FMT_INT, fill=_FILL_INPUT)
        elif item["kind"] in ("integer", "number"):
            _span_number(ws, row, 3, 4, item["value"], fmt=_FMT_INT, fill=_FILL_INPUT)
        else:
            _span_text(ws, row, 3, 4, item["value"], fill=_FILL_INPUT)
        _span_text(ws, row, 5, 7, item.get("note") or item.get("source") or "",
                   font=_FONT_SMALL)
        guard = _input_guard(item, ref)
        table[item["id"]] = (ref, guard)
        table[item["symbol"]] = (ref, guard)
        row += 1
    # batch 固定 1：计算源直接使用常量 1，不引用本行；本行只作标注。
    _span_text(ws, row, 1, 2, "batch（固定）", font=_FONT_BOLD)
    _span_text(ws, row, 3, 4, "固定 1", fill=_FILL_EXCLUDED)
    _span_text(ws, row, 5, 7,
               "计算源直接使用常量 1，不引用本格；改本格不影响成本。"
               "n 只表示 head 或 1，不承载 batch/层数/专家数。",
               font=_FONT_SMALL)
    table["B"] = ("1", "TRUE")
    table["batch_size"] = ("1", "TRUE")
    row += 1
    return table, row


def _write_hardware(ws, start_row: int, sheet: dict) -> tuple[dict, int]:
    items = {item["id"]: item for item in sheet["inputs"]}
    _span_text(ws, start_row, 8, 12, "硬件吞吐（element/ms；独立可改）",
               font=_FONT_SECTION, fill=_FILL_SECTION)
    _span_text(ws, start_row + 1, 8, 9, "项目", font=_FONT_HEAD, fill=_FILL_HEAD)
    _span_text(ws, start_row + 1, 10, 10, "值", font=_FONT_HEAD, fill=_FILL_HEAD)
    _span_text(ws, start_row + 1, 11, 12, "单位/依据", font=_FONT_HEAD, fill=_FILL_HEAD)
    cube_row = start_row + 2
    vec_row = start_row + 3
    _span_text(ws, cube_row, 8, 9, "CUBE 吞吐", font=_FONT_HEAD, fill=_FILL_HEAD)
    _span_text(ws, vec_row, 8, 9, "VEC 吞吐", font=_FONT_HEAD, fill=_FILL_HEAD)
    _span_text(ws, start_row + 4, 8, 12,
               "两项独立引用：cube(ms) 只除以 CUBE 吞吐，vector(ms) 只除以 VEC 吞吐；"
               "≤0、空白或文本时相关成本显示未知，不按 0 计。",
               font=_FONT_SMALL)

    cube_item = items["cube_mac_per_ms"]
    vec_item = items["vec_elem_per_ms"]
    cube_ref = f"$J${cube_row}"
    vec_ref = f"$J${vec_row}"
    # 不套整数格式：部分重算器会把布尔输入按整数格式转成 1，丢失 ISLOGICAL 类型。
    _span_number(ws, cube_row, 10, 10, cube_item["value"], fill=_FILL_INPUT)
    _span_number(ws, vec_row, 10, 10, vec_item["value"], fill=_FILL_INPUT)
    _span_text(ws, cube_row, 11, 12,
               "element/ms；默认 4096（B1/M1/K64/N64=1 ms）", font=_FONT_SMALL)
    _span_text(ws, vec_row, 11, 12,
               "element/ms；默认 64（M1/N64=1 ms）", font=_FONT_SMALL)
    # 布尔值在部分重算器里会被当成 1；显式用 ISLOGICAL 拒绝，不只靠 ISNUMBER。
    cube_guard = (f"IFERROR(AND(ISNUMBER({cube_ref}),NOT(ISLOGICAL({cube_ref})),"
                  f"{cube_ref}>0),FALSE)")
    vec_guard = (f"IFERROR(AND(ISNUMBER({vec_ref}),NOT(ISLOGICAL({vec_ref})),"
                 f"{vec_ref}>0),FALSE)")
    refs = {
        "cube": (cube_ref, cube_guard),
        "vec": (vec_ref, vec_guard),
    }
    table = {
        "cube_mac_per_ms": (cube_ref, cube_guard),
        "vec_elem_per_ms": (vec_ref, vec_guard),
        "Tcube": (cube_ref, cube_guard),
        "Tvec": (vec_ref, vec_guard),
    }
    return refs, table


def _param_value_guard(cell: str, param: dict | None, kind: str = "param") -> str:
    if kind == "layer":
        return "TRUE"
    if kind == "layer_total":
        return "TRUE"
    if param and param.get("formula") and param["id"] == "Bbias":
        return f"ISLOGICAL({cell})"
    value = None if param is None else param.get("value")
    if isinstance(value, bool):
        return f"ISLOGICAL({cell})"
    if isinstance(value, int) or (param and param.get("formula")):
        return f"IFERROR(AND(ISNUMBER({cell}),INT({cell})={cell},{cell}>=0),FALSE)"
    return f"ISNUMBER({cell})"


def _compile_param_formula(formula: str, table: dict) -> dict | None:
    try:
        names = expression_names(formula)
    except ExpressionError:
        return None
    missing = [name for name in names if name not in table]
    if missing:
        return None
    refs = {name: table[name][0] for name in names}
    try:
        expr = excel_formula(formula, refs)
    except ExpressionError:
        return None
    guard = _and_guard([table[name][1] for name in names])
    return {"expr": expr, "guard": guard, "names": names, "refs": refs}


def _write_param_cell(ws, entry: _Entry, table: dict) -> None:
    """写一个参数的活值/公式；entry.guard 供依赖参数与算子规则引用。"""
    if entry.kind == "layer":
        _span_number(ws, entry.row, entry.col, entry.col + entry.value_span - 1, entry.value, fmt=_FMT_INT)
        entry.guard = "TRUE"
        return
    if entry.kind == "layer_total":
        # 由调用方在 layer_cells 齐备后写 SUM 公式；此处占位。
        return
    param = entry.param or {}
    if entry.formula:
        compiled = _compile_param_formula(entry.formula, table)
        if compiled is None:
            _span_text(ws, entry.row, entry.col, entry.col + entry.value_span - 1,
                       "未知（公式依赖缺失或非法）", fill=_FILL_UNKNOWN)
            entry.guard = "FALSE"
            return
        reason = _perf_unknown_text(compiled["names"])
        fail = "未知（公式求值失败）"
        if param.get("id") == "Bbias":
            formula = (f'=IF({compiled["guard"]},'
                       f'IFERROR(IF({compiled["expr"]},"TRUE","FALSE"),'
                       f'{_xl_quote(fail)}),{_xl_quote(reason)})')
            _span_formula(ws, entry.row, entry.col, entry.col + entry.value_span - 1, formula)
            _shorten_unknown_cell(ws, entry.row, entry.col)
            entry.guard = f"AND(ISTEXT({entry.cell}),LEN({entry.cell})>0)"
            return
        formula = (f'=IF({compiled["guard"]},IFERROR({compiled["expr"]},'
                   f'{_xl_quote(fail)}),{_xl_quote(reason)})')
        _span_formula(ws, entry.row, entry.col, entry.col + entry.value_span - 1, formula, fmt=_FMT_INT)
        _shorten_unknown_cell(ws, entry.row, entry.col)
        entry.guard = f"IFERROR(AND({compiled['guard']},ISNUMBER({entry.cell}),"
        entry.guard += f"INT({entry.cell})={entry.cell},{entry.cell}>=0),FALSE)"
        return
    value = entry.value
    if value is None:
        _span_text(ws, entry.row, entry.col, entry.col + entry.value_span - 1, "未知（未提供）",
                   fill=_FILL_UNKNOWN)
        entry.guard = "FALSE"
        return
    if isinstance(value, bool):
        _span_text(ws, entry.row, entry.col, entry.col + entry.value_span - 1,
                   "TRUE" if value else "FALSE")
        entry.guard = f"ISLOGICAL({entry.cell})"
        return
    if isinstance(value, int):
        _span_number(ws, entry.row, entry.col, entry.col + entry.value_span - 1, value, fmt=_FMT_INT)
        entry.guard = _param_value_guard(entry.cell, param)
        return
    if isinstance(value, float):
        _span_number(ws, entry.row, entry.col, entry.col + entry.value_span - 1, value, fmt=_FMT_NUM)
        entry.guard = f"ISNUMBER({entry.cell})"
        return
    _span_text(ws, entry.row, entry.col, entry.col + entry.value_span - 1, value)
    entry.guard = f"AND(ISTEXT({entry.cell}),LEN({entry.cell})>0)"


def _entry_symbol_map(entries: list[_Entry], scope: str) -> dict[str, _Entry]:
    mapping: dict[str, _Entry] = {}
    for entry in entries:
        if entry.kind != "param":
            continue
        if entry.scope not in ("", scope):
            continue
        mapping[entry.pid] = entry
        mapping[entry.symbol] = entry
        if entry.param:
            for alias in entry.param.get("axis_aliases", []):
                mapping[alias] = entry
    return mapping


def _compute_param_guards(entries: list[_Entry], input_table: dict) -> None:
    """预计算参数守卫。

    派生参数单元格自己的公式在依赖非法时返回“未知”文本，所以守卫只需检查
    该单元格自身是合法整数/数值；不再递归展开依赖守卫（否则公式指数级膨胀）。
    """
    for entry in entries:
        if entry.kind != "param":
            entry.guard = "TRUE"
            continue
        if entry.formula:
            if entry.pid == "Bbias":
                entry.guard = f"AND(ISTEXT({entry.cell}),LEN({entry.cell})>0)"
            else:
                entry.guard = (f"IFERROR(AND(ISNUMBER({entry.cell}),"
                               f"INT({entry.cell})={entry.cell},{entry.cell}>=0),FALSE)")
        else:
            entry.guard = _param_value_guard(entry.cell, entry.param, entry.kind)


def _write_param_table(ws, start_row: int, spec: dict, entries: list[_Entry],
                       input_table: dict, layer_cells: dict) -> tuple[dict, int, dict]:
    """左右两栏集中参数区；返回 base/scope 符号表与下一行。"""
    _span_text(ws, start_row, 1, 6, "集中参数（左）", font=_FONT_SECTION, fill=_FILL_SECTION)
    _span_text(ws, start_row, 7, 12, "集中参数（右）", font=_FONT_SECTION, fill=_FILL_SECTION)
    head = start_row + 1
    for c1, c2, label in ((1, 1, "类别"), (2, 2, "参数（单位/域）"),
                          (3, 3, "符号"), (4, 6, "值/公式"),
                          (7, 7, "类别"), (8, 9, "参数（单位/域）"),
                          (10, 10, "符号"), (11, 12, "值/公式")):
        _span_text(ws, head, c1, c2, label, font=_FONT_HEAD, fill=_FILL_HEAD)
    n = len(entries)
    half = math.ceil(n / 2)
    left = entries[:half]
    right = entries[half:]
    for list_items, is_left in ((left, True), (right, False)):
        row = head + 1
        c_cat, c_label, c_sym, c_val = (1, 2, 3, 4) if is_left else (7, 8, 10, 11)
        for entry in list_items:
            entry.row = row
            entry.col = c_val
            entry.cell = f"${get_column_letter(c_val)}${row}"
            entry.value_span = 3 if is_left else 2
            _span_text(ws, row, c_cat, c_cat, entry.category, font=_FONT_SMALL)
            if is_left:
                _span_text(ws, row, c_label, c_label, entry.label)
            else:
                _span_text(ws, row, c_label, c_label + 1, entry.label)
            _span_text(ws, row, c_sym, c_sym, entry.symbol, font=_FONT_SMALL)
            if entry.kind in ("layer", "layer_total"):
                layer_cells[entry.key] = entry.cell
            ws.row_dimensions[row].height = 34
            row += 1
    _compute_param_guards(entries, input_table)

    # 先建基础表（无作用域参数 + 输入），再按作用域补派生参数。
    base = dict(input_table)
    for entry in entries:
        if entry.scope == "" and entry.kind == "param":
            _add_entry_symbols(base, entry)
    scope_tables: dict[str, dict] = {}
    for scope in ("CSA", "HCA"):
        table = dict(base)
        for entry in entries:
            if entry.scope == scope:
                _add_entry_symbols(table, entry)
        scope_tables[scope] = table

    # 写参数值/公式（layer_total 最后写，需引用层数单元格）。
    for entry in entries:
        if entry.kind == "layer_total":
            continue
        table = scope_tables.get(entry.scope, base)
        _write_param_cell(ws, entry, table)
    total = next(e for e in entries if e.kind == "layer_total")
    layer_refs = [layer_cells[f"layer_count@{gid}"]
                  for gid in ("swa-hash", "csa-hash", "hca-topk", "csa-topk")
                  if f"layer_count@{gid}" in layer_cells]
    total.cell = f"${get_column_letter(total.col)}${total.row}"
    formula = f"=SUM({','.join(layer_refs)})" if layer_refs else f"={total.value}"
    _span_formula(ws, total.row, total.col, total.col + total.value_span - 1, formula, fmt=_FMT_INT,
                  fill=_FILL_TOTAL)
    total.guard = "TRUE"
    layer_cells[total.key] = total.cell
    next_row = start_row + 2 + max(half, n - half)
    return base, next_row, scope_tables


def _group_table(group: dict, base: dict, scope_tables: dict,
                 by_key: dict[str, _Entry], mdep: set[str], input_table: dict) -> dict:
    scope = _scope_of_group(group["id"])
    table = dict(scope_tables.get(scope, base)) if scope else dict(base)
    for param in group["parameters"]:
        pid = param["id"]
        key = f"{pid}@{scope}" if scope and pid in mdep else pid
        entry = by_key.get(key)
        if entry is None:
            continue
        _add_entry_symbols(table, entry)
    table.update(input_table)
    return table


# ---------------------------------------------------------------- 主表


def _shorten_unknown_cell(ws, row: int, col: int) -> None:
    """把公式里的长“未知（原因）”收成“未知*”，完整原因放批注。"""
    cell = ws.cell(row=row, column=col)
    if cell.data_type != "f":
        return
    formula = str(cell.value)
    reasons = re.findall(r'"未知（[^"]*）"', formula)
    if not reasons:
        return
    cell.value = re.sub(r'"未知（[^"]*）"', '"未知*"', formula)
    cell.comment = Comment("显示“未知*”时的完整原因：" + "；".join(reasons), "E01")


def _write_dim(ws, row: int, col: int, expr: str, table: dict, label: str,
               *, condition: dict | None = None):
    compiled = _perf_compile(expr, table, label)
    _perf_dim(ws, row, col, compiled, fill=None, condition=condition)
    cell = ws.cell(row=row, column=col)
    cell.font = _FONT_BODY
    _shorten_unknown_cell(ws, row, col)
    if isinstance(cell.value, str) and cell.value.startswith("未知"):
        reason = cell.value
        cell.value = "未知*"
        cell.comment = Comment("显示“未知*”时的原因：" + reason, "E01")
    return compiled


def _write_n(ws, row: int, n_expr: str, table: dict, label: str, condition=None):
    if n_expr == "1":
        _span_number(ws, row, 4, 4, 1, fmt=_FMT_INT)
        return None
    return _write_dim(ws, row, 4, n_expr, table, label, condition=condition)


def _compile_guard(rule: dict, table: dict) -> tuple[list[str], list[str]]:
    guards: list[str] = []
    names: list[str] = []
    for expr in rule.get("guards", []):
        compiled = _perf_compile(expr, table, "guards")
        guards.append(f"IFERROR(AND({compiled['guard']},{compiled['expr']}),FALSE)")
        names.extend(compiled["names"])
    return guards, sorted(set(names))


def _compile_repeat(rule: dict, table: dict):
    repeat = rule.get("repeat", 1)
    if isinstance(repeat, str):
        compiled = _perf_compile(repeat, table, "repeat")
        guard = (f"IFERROR(AND({compiled['guard']},ISNUMBER({compiled['expr']}),"
                 f"INT({compiled['expr']})={compiled['expr']},{compiled['expr']}>=1),FALSE)")
        return compiled["expr"], guard, compiled["names"]
    return str(repeat), None, []


def _unknown_text(names: list[str], rule: dict) -> str:
    return "未知*"


def _unknown_full(names: list[str], rule: dict) -> str:
    text = _perf_unknown_text(sorted(set(names)))
    guard_text = rule.get("guard_text")
    if guard_text:
        text = text[:-1] + "；" + guard_text + "）"
    return text


def _cost_formula(kind: str, body: str, guard: str, unknown: str,
                  engine_refs: dict, condition: dict | None) -> str:
    thru_ref, thru_guard = engine_refs[kind]
    fail = _xl_quote(unknown)
    # 吞吐必须是正数数值：负数/0/空白/数字文本/普通文本/布尔全部拒绝。
    # 条件明确不执行时仍返回合法 0，不因吞吐非法而改写条件 0。
    guard_all = _and_guard([guard, thru_guard])
    core = f"IF({guard_all},IFERROR(({body})/{thru_ref},{fail}),{fail})"
    if condition is not None:
        cond_reason = _xl_quote("未知*")
        return (f"=IF({condition['guard']},IF({condition['expr']},{core},0),{cond_reason})")
    return "=" + core


def _write_op_row(ws, row: int, group: dict, op: dict, table: dict,
                  engine_refs: dict, sheet_name: str) -> dict:
    rule = op["performance"]
    kind = rule["class"]
    path = _path_label(group["id"], op)
    _span_text(ws, row, 1, 1, path, font=_FONT_SMALL)
    _span_text(ws, row, 2, 2, op["id"], font=_FONT_BOLD)
    comment_text = f"{op.get('name', '')}\n模块：{op.get('module_path', '')}\nop：{op.get('op', '')}"
    if rule.get("note"):
        comment_text += f"\n规则：{rule['note']}"
    ws.cell(row=row, column=2).comment = Comment(comment_text, "E01")
    ws.row_dimensions[row].height = 28
    condition = rule.get("conditional")
    cond_compiled = (_perf_compile(condition["when"], table, "conditional.when")
                     if condition else None)
    guards, guard_names = _compile_guard(rule, table)
    repeat_expr, repeat_guard, repeat_names = _compile_repeat(rule, table)

    if kind == "container":
        ws.row_dimensions[row].height = 34
        _span_text(ws, row, 3, 3, "容器", font=_FONT_SMALL)
        for col in (4, 5, 6, 7, 8, 9):
            _span_text(ws, row, col, col, "—", align=_ALIGN_CENTER)
        _span_text(ws, row, 10, 10, "—（容器）", fill=_FILL_EXCLUDED)
        _span_text(ws, row, 11, 11, "—（容器）", fill=_FILL_EXCLUDED)
        _span_text(ws, row, 12, 12, "容器不计", fill=_FILL_EXCLUDED, font=_FONT_SMALL)
        ws.cell(row=row, column=12).comment = Comment(
            "容器：成本由下面四组层组总计表达，不重复计。", "E01")
        return {"row": row, "op": op, "group": group, "sheet": sheet_name,
                "class": kind, "billable": False, "excluded": False,
                "container": True, "terms": []}

    charge = rule.get("charge") or {}
    excluded = charge.get("mode") == "excluded"
    terms = rule.get("terms", [])
    composite = len(terms) > 1

    # 维度列
    dim_names: list[str] = []
    if kind == "cube":
        maps = [_cube_map(t) for t in terms]
        if composite:
            _span_text(ws, row, 4, 4, "复合", align=_ALIGN_CENTER)
            _span_text(ws, row, 5, 5, "见明细", align=_ALIGN_CENTER)
            _span_text(ws, row, 6, 6, "见明细", align=_ALIGN_CENTER)
            for col in (7, 8, 9):
                _span_text(ws, row, col, col, "见明细", align=_ALIGN_CENTER)
        else:
            mp = maps[0]
            compiled_n = _write_n(ws, row, mp["n"], table, f"{op['id']}/n")
            if compiled_n:
                dim_names.extend(compiled_n["names"])
            for col, expr, tag in ((5, mp["T"], "T"), (6, mp["D"], "D"),
                                   (7, mp["M"], "M"), (8, mp["K"], "K"),
                                   (9, mp["N"], "N")):
                compiled = _write_dim(ws, row, col, expr, table, f"{op['id']}/{tag}",
                                      condition=cond_compiled)
                dim_names.extend(compiled["names"])
            if mp["head"]:
                ws.cell(row=row, column=4).comment = Comment(
                    f"n={mp['head']} 是 head 数；有效 M 已含 head 一次，"
                    "J=M×K×N 不再乘 n。", "E01")
            if mp["b"] not in ("B", "1"):
                detail = ("head" if mp["head"] else
                          "分组 G" if mp["b"] == "B*G" else
                          "batch×T" if mp["b"] == "B*seqlen" else mp["b"])
                ws.cell(row=row, column=7).comment = Comment(
                    f"有效 M=({mp['b']})×({mp['m']})；批量因子（{detail}）只折入 M 一次。",
                    "E01")
    else:
        maps = [_vec_map(t) for t in terms]
        if composite:
            _span_text(ws, row, 4, 4, "复合", align=_ALIGN_CENTER)
            _span_text(ws, row, 5, 5, "见明细", align=_ALIGN_CENTER)
            _span_text(ws, row, 6, 6, "见明细", align=_ALIGN_CENTER)
        else:
            mp = maps[0]
            compiled_n = _write_n(ws, row, mp["n"], table, f"{op['id']}/n")
            if compiled_n:
                dim_names.extend(compiled_n["names"])
            for col, expr, tag in ((5, mp["T"], "T"), (6, mp["D"], "D")):
                compiled = _write_dim(ws, row, col, expr, table, f"{op['id']}/{tag}",
                                      condition=cond_compiled)
                dim_names.extend(compiled["names"])
        for col in (7, 8, 9):
            _span_text(ws, row, col, col, "—", align=_ALIGN_CENTER)
    _span_text(ws, row, 3, 3, "CUBE" if kind == "cube" else "VEC", font=_FONT_SMALL,
               align=_ALIGN_CENTER)
    if composite:
        ws.cell(row=row, column=3).value = (
            "CUBE（复合）" if kind == "cube" else "VEC（复合）")
    if rule.get("repeat") not in (None, 1, "1"):
        ws.cell(row=row, column=3).value = f"{'CUBE' if kind == 'cube' else 'VEC'}（×{rule['repeat']}）"

    if excluded:
        ws.row_dimensions[row].height = 34
        charged_at = charge.get("charged_at", "整网外围")
        _span_text(ws, row, 10, 10, "整网一次", fill=_FILL_EXCLUDED, font=_FONT_SMALL)
        _span_text(ws, row, 11, 11, "整网一次", fill=_FILL_EXCLUDED, font=_FONT_SMALL)
        _span_text(ws, row, 12, 12, "不计本组", fill=_FILL_EXCLUDED, font=_FONT_SMALL)
        ws.cell(row=row, column=10).comment = Comment(
            f"整网一次性项：见 {charged_at}；层组只引用，不计入本组小计。", "E01")
        return {"row": row, "op": op, "group": group, "sheet": sheet_name,
                "class": kind, "billable": False, "excluded": True,
                "container": False, "terms": terms}

    # 工作量与耗时
    unknown_names = list(guard_names) + list(repeat_names)
    if composite:
        term_exprs = []
        for term in terms:
            keys = ("b", "m", "k", "n") if kind == "cube" else ("m", "n")
            pieces = []
            for key in keys:
                compiled = _perf_compile(term[key], table, f"{op['id']}/{key}")
                pieces.append(f"({compiled['expr']})")
                dim_names.extend(compiled["names"])
                guards.append(f"IFERROR(AND({compiled['guard']},ISNUMBER({compiled['expr']})),FALSE)")
            term_exprs.append("*".join(pieces))
        body = "+".join(f"({expr})" for expr in term_exprs)
        if repeat_expr != "1":
            body = f"({body})*({repeat_expr})"
        guard_parts = list(guards)
        if repeat_guard:
            guard_parts.append(repeat_guard)
        guard = _and_guard(guard_parts)
        full_unknown = _unknown_full(unknown_names + dim_names, rule)
        unknown = _unknown_text(unknown_names + dim_names, rule)
        if kind == "cube":
            _span_formula(ws, row, 10, 10,
                          _cost_formula("cube", body, guard, unknown, engine_refs, cond_compiled),
                          fmt=_FMT_MS)
            _span_number(ws, row, 11, 11, 0, fmt=_FMT_MS)
        else:
            _span_number(ws, row, 10, 10, 0, fmt=_FMT_MS)
            _span_formula(ws, row, 11, 11,
                          _cost_formula("vec", body, guard, unknown, engine_refs, cond_compiled),
                          fmt=_FMT_MS)
    else:
        # 单矩阵项/单 VEC 项：维度单元格已带守卫，直接引用。
        if kind == "cube":
            dim_cols = (7, 8, 9)
            body = f"G{row}*H{row}*I{row}"
        else:
            dim_cols = (4, 5, 6)
            body = f"D{row}*E{row}*F{row}"
        if repeat_expr != "1":
            body = f"({body})*({repeat_expr})"
        dim_ok = "AND(" + ",".join(f"ISNUMBER({get_column_letter(c)}{row})"
                                   for c in dim_cols) + ")"
        guard_parts = [dim_ok] + list(guards)
        if repeat_guard:
            guard_parts.append(repeat_guard)
        guard = _and_guard(guard_parts)
        full_unknown = _unknown_full(unknown_names + dim_names, rule)
        unknown = _unknown_text(unknown_names + dim_names, rule)
        if kind == "cube":
            _span_formula(ws, row, 10, 10,
                          _cost_formula("cube", body, guard, unknown, engine_refs, cond_compiled),
                          fmt=_FMT_MS)
            _span_number(ws, row, 11, 11, 0, fmt=_FMT_MS)
        else:
            _span_number(ws, row, 10, 10, 0, fmt=_FMT_MS)
            _span_formula(ws, row, 11, 11,
                          _cost_formula("vec", body, guard, unknown, engine_refs, cond_compiled),
                          fmt=_FMT_MS)
    # op(ms)=MAX；任一未知则保持未知，不被 0 吞掉。
    l_formula = (f'=IF(AND(ISNUMBER(J{row}),ISNUMBER(K{row})),MAX(J{row},K{row}),'
                 f'"未知*")')
    _span_formula(ws, row, 12, 12, l_formula, fmt=_FMT_MS)
    if rule.get("guards") or rule.get("conditional"):
        cost_col = 10 if kind == "cube" else 11
        ws.cell(row=row, column=cost_col).comment = Comment(
            "显示“未知”时的完整原因：" + full_unknown, "E01")
        ws.cell(row=row, column=12).comment = Comment(
            "op=MAX(cube,vector)；任一未知时保持未知，不按 0 计。完整原因：" + full_unknown,
            "E01")
    if composite or rule.get("repeat") not in (None, 1, "1"):
        ws.cell(row=row, column=12).comment = Comment(
            "复合/重复项：CUBE 工作量已按 SUM 或 repeat 合成；逐项分解见「辅助明细」。",
            "E01")
    return {"row": row, "op": op, "group": group, "sheet": sheet_name,
            "class": kind, "billable": True, "excluded": False,
            "container": False, "terms": terms,
            "composite": composite, "repeat": rule.get("repeat"),
            "conditional": condition, "guards": rule.get("guards", []),
            "excluded_reason": charge.get("reason", "")}


def _subtotal_formula(col: str, rows: list[int]) -> str:
    if not rows:
        return "=0"
    refs = [f"{col}{r}" for r in rows]
    joined = ",".join(refs)
    return (f'=IF(COUNT({joined})={len(refs)},SUM({joined}),'
            f'"未知*")')


def _write_subtotal(ws, row: int, label: str, rows: list[dict], *, total_fill=False,
                    note: str = "不含整网一次/容器"):
    fill = _FILL_TOTAL if total_fill else _FILL_SUB
    _span_text(ws, row, 1, 1, label, font=_FONT_BOLD, fill=fill)
    _span_text(ws, row, 2, 2, f"{len(rows)} 个计费算子；{note}" if note
               else f"{len(rows)} 个计费算子", font=_FONT_SMALL, fill=fill)
    _span_text(ws, row, 3, 3, "小计", font=_FONT_SMALL, fill=fill, align=_ALIGN_CENTER)
    for col, letter in ((10, "J"), (11, "K"), (12, "L")):
        _span_formula(ws, row, col, col, _subtotal_formula(letter, [r["row"] for r in rows]),
                      fmt=_FMT_MS, fill=fill)
    ws.row_dimensions[row].height = 30


def _group_total_formula(sub_col: str, sub_row: int, layer_cell: str) -> str:
    return (f'=IFERROR(IF(ISNUMBER({sub_col}{sub_row}),{sub_col}{sub_row}*{layer_cell},'
            f'"未知*"),"未知*")')


def _write_group_total(ws, row: int, sub_row: int, layer_cell: str, count: int):
    _span_text(ws, row, 1, 1, f"层组总计 × {count} 真实层", font=_FONT_BOLD, fill=_FILL_TOTAL)
    _span_text(ws, row, 2, 2, "单层小计 × 层数", font=_FONT_SMALL, fill=_FILL_TOTAL)
    _span_text(ws, row, 3, 3, "汇总", font=_FONT_SMALL, fill=_FILL_TOTAL,
               align=_ALIGN_CENTER)
    for col, letter in ((10, "J"), (11, "K"), (12, "L")):
        _span_formula(ws, row, col, col,
                      _group_total_formula(letter, sub_row, layer_cell),
                      fmt=_FMT_MS, fill=_FILL_TOTAL)
    ws.row_dimensions[row].height = 20


def _write_grand_total(ws, row: int, term_cells: list[str], layer_total_cell: str):
    _span_text(ws, row, 1, 1, "整网串行总计", font=_FONT_BOLD, fill=_FILL_TOTAL)
    _span_text(ws, row, 2, 2, "真实 43 层；入口+4组×层数+出口",
               font=_FONT_SMALL, fill=_FILL_TOTAL)
    _span_text(ws, row, 3, 3, "总计", font=_FONT_SMALL, fill=_FILL_TOTAL,
               align=_ALIGN_CENTER)
    refs = ",".join(term_cells)
    for col in (10, 11, 12):
        letter = get_column_letter(col)
        cells = [f"{letter}{cell}" for cell in term_cells]
        joined = ",".join(cells)
        formula = (f'=IF(COUNT({joined})={len(cells)},SUM({joined}),'
                   f'"未知*")')
        _span_formula(ws, row, col, col, formula, fmt=_FMT_MS, fill=_FILL_TOTAL)
    ws.row_dimensions[row].height = 32


# ---------------------------------------------------------------- 主表装配


def _write_main_sheet(ws, spec: dict, sheet: dict, config: _Config,
                      op_rows: list[dict]) -> None:
    _prep_span(ws, 1, 1, 12, _FILL_TITLE)
    title = ws.cell(row=1, column=1)
    title.value = f"{sheet['title']} · Excel 性能样表（12 列）"
    title.font = _FONT_TITLE
    title.alignment = _ALIGN_CENTER
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=12)
    ws.row_dimensions[1].height = 28
    model_info = spec.get("model") or {}
    model_label = (model_info.get("display_name") or model_info.get("model_key")
                   or str(model_info))
    model_impl = model_info.get("implementation", "")
    revision = model_info.get("source_revision", "")
    _span_text(ws, 2, 1, 12,
               f"模型：{model_label}；实现：{model_impl}；源码 revision：{revision}；"
               f"数据：冻结 layer_structures.json / network_overview.json；batch=1；单位 element/ms；"
               f"每行一个算子；CUBE=M×K×N，VEC=n×T×D，op=MAX(cube,vector)。",
               font=_FONT_SUB)
    _span_text(ws, 3, 1, 12,
               "草稿只作布局参考，其中的模型名、层型、层数、参数与维度均不是事实。"
               "黄色单元格可改；蓝色/公式单元格自动重算；未知/缺失保持“未知（原因）”，不按 0 计。"
               "主表从下方“结构/通路”表头开始；列 A/B 已冻结。",
               font=_FONT_SMALL)

    # 标题行下的输入/吞吐区
    scenario_start = 5
    input_table, scenario_next = _write_scenario(ws, scenario_start, sheet)
    engine_refs, hw_table = _write_hardware(ws, scenario_start, sheet)
    input_table.update(hw_table)

    # 参数区
    param_start = max(scenario_next, scenario_start + 5) + 1
    layer_cells: dict[str, str] = {}
    base, next_row, scope_tables = _write_param_table(
        ws, param_start, spec, config.entries, input_table, layer_cells)

    # 参数写完后，作用域表/基础表需要用最终 cell/guard 重建。
    base = dict(input_table)
    for entry in config.entries:
        if entry.scope == "" and entry.kind == "param":
            _add_entry_symbols(base, entry)
    scope_tables = {}
    for scope in ("CSA", "HCA"):
        table = dict(base)
        for entry in config.entries:
            if entry.scope == scope:
                _add_entry_symbols(table, entry)
        scope_tables[scope] = table
    config.base_table = base
    config.scope_tables = scope_tables
    config.layer_cells = layer_cells

    # 主表
    header_row = next_row + 1
    _banner(ws, header_row - 1,
            "主表：每行一个算子；CUBE 有效 M 已含 head/分组/矩阵 batch 一次；"
            "VEC 为 n×T×D；op(ms)=MAX(cube,vector)")
    _header(ws, header_row)
    row = header_row + 1
    groups = {g["id"]: g for g in sheet["groups"]}
    mdep = _m_dependent_ids(sheet["groups"])

    # 整网入口（serial_order 第一段）
    net = sheet["network"]
    net_group = groups["net-periphery"]
    net_ops = net_group["operators"]
    container_index = next((i for i, op in enumerate(net_ops)
                            if op["id"] == "net_layer_loop"), len(net_ops))
    entry_ops = net_ops[:container_index]
    exit_ops = net_ops[container_index + 1:] if container_index < len(net_ops) else []
    container_op = net_ops[container_index] if container_index < len(net_ops) else None

    _banner(ws, row, "整网入口（serial_order 第一段；不计层数）", fill=_FILL_GROUP)
    row += 1
    entry_table = _group_table(net_group, base, scope_tables, config.by_key,
                               mdep, input_table)
    entry_rows = []
    for op in entry_ops:
        info = _write_op_row(ws, row, net_group, op, entry_table, engine_refs, sheet["name"])
        op_rows.append(info)
        if info["billable"]:
            entry_rows.append(info)
        row += 1
    entry_sub_row = row
    _write_subtotal(ws, row, "入口小计", entry_rows, note="整网一次，不乘层数")
    row += 1

    # 层循环容器
    if container_op is not None:
        info = _write_op_row(ws, row, net_group, container_op, entry_table, engine_refs,
                             sheet["name"])
        op_rows.append(info)
        # 容器行补充 43 层说明（B 列仍是 op id；A 列写短标签）。
        ws.cell(row=row, column=1).value = "43 层循环"
        ws.cell(row=row, column=12).value = "容器不计"
        row += 1

    # 四个真实层组（serial_order 中间段）
    group_serials = []
    for item in net["serial_order"]:
        if item.get("kind") != "group":
            continue
        gid = item["group"]
        if gid == "net-periphery":
            continue
        group = groups[gid]
        layer_count = len(group["layer_indices"])
        layer_cell = layer_cells[f"layer_count@{gid}"]
        _banner(ws, row,
                f"层组 {group['label']}（{gid}） · 真实层 [{_compact_layers(group['layer_indices'])}]"
                f" × {layer_count} · 代表层 {group['representative_layer_index']}",
                fill=_FILL_GROUP)
        row += 1
        table = _group_table(group, base, scope_tables, config.by_key,
                             mdep, input_table)
        rows = []
        for op in group["operators"]:
            info = _write_op_row(ws, row, group, op, table, engine_refs, sheet["name"])
            op_rows.append(info)
            if info["billable"]:
                rows.append(info)
            row += 1
        sub_row = row
        _write_subtotal(ws, row, "单层小计", rows)
        row += 1
        total_row = row
        _write_group_total(ws, row, sub_row, layer_cell, layer_count)
        row += 1
        group_serials.append({"group": gid, "sub": sub_row, "total": total_row,
                              "count": layer_count, "rows": rows})

    # 整网出口（serial_order 最后一段）
    _banner(ws, row, "整网出口（serial_order 最后一段；不计层数）", fill=_FILL_GROUP)
    row += 1
    exit_table = _group_table(net_group, base, scope_tables, config.by_key,
                              mdep, input_table)
    exit_rows = []
    for op in exit_ops:
        info = _write_op_row(ws, row, net_group, op, exit_table, engine_refs, sheet["name"])
        op_rows.append(info)
        if info["billable"]:
            exit_rows.append(info)
        row += 1
    exit_sub_row = row
    _write_subtotal(ws, row, "出口小计", exit_rows, note="整网一次，不乘层数")
    row += 1

    # 整网总计：入口 + 4 组总计 + 出口，各只引用一次。
    term_cells = [f"{entry_sub_row}"]
    term_cells += [str(g["total"]) for g in group_serials]
    term_cells.append(str(exit_sub_row))
    _write_grand_total(ws, row, term_cells, layer_cells["layer_count@total"])
    row += 2

    # 表尾说明
    notes = [
        "读法：CUBE 工作量为 M×K×N（head/分组/矩阵 batch 已折入有效 M，不额外乘 n）；"
        "VEC 工作量为 n×T×D；两项分别除以本表 CUBE/VEC 吞吐；op(ms)=MAX(cube,vector)。",
        "复合算子：同一 op 的多个矩阵项先 SUM、同形重复乘 repeat，再除以 CUBE 吞吐；"
        "逐项 M/K/N 与来源见「辅助明细」，不对多项取 MAX。",
        "未知*：维度/吞吐等必要输入未定或非法；完整原因见单元格批注与「辅助明细」。"
        "SUM/MAX 前先检查数值，不会把未知吞成 0；条件不执行是合法 0（如 Decode 的 HCA block_bias）。",
        "归属：层组只写代表层模板，单层小计×真实层数；mask/RoPE 整网一次性项在层组只引用，"
        "整网入口/出口各收一次；容器不重复计费。",
        "batch 固定 1；n 只表示 head 或 1，不承载层数、执行次数或专家数。",
    ]
    for text in notes:
        _span_text(ws, row, 1, 12, "· " + text, font=_FONT_SMALL)
        row += 1

    # 冻结列 A/B；打印标题为主表表头。
    ws.freeze_panes = "C1"
    ws.print_title_rows = f"{header_row}:{header_row}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    for idx, width in enumerate(MAIN_WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width


# ---------------------------------------------------------------- 辅助明细


def _tensor_label(group: dict, tid: str) -> str:
    tensor = next((t for t in group.get("tensors", []) if t["id"] == tid), None)
    if tensor is None:
        return tid
    axes = ", ".join(str(axis) for axis in tensor["shape"])
    return f"{tensor['name']}[{axes}]"


def _source_text(refs: list) -> str:
    parts = []
    for ref in refs:
        where = f"{ref.get('path', '')}#{ref.get('locator', '')}".strip("#")
        note = ref.get("note", "")
        parts.append(f"[{ref.get('kind', '')}] {where}" + (f"（{note}）" if note else ""))
    return " | ".join(parts) or "（无来源）"


def _write_detail(wb: Workbook, spec: dict, op_rows: list[dict]) -> None:
    ws = wb.create_sheet("辅助明细")
    widths = (10, 14, 12, 18, 16, 22, 28, 12, 30, 34, 30, 30, 30, 24, 8, 14, 14, 14,
              8, 28, 30, 22, 60, 40)
    for idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width

    _prep_span(ws, 1, 1, 12, _FILL_TITLE)
    cell = ws.cell(row=1, column=1)
    cell.value = "辅助明细：参数来源 + 逐算子完整 shape / 源码 / 项分解（只追溯，不改变成本）"
    cell.font = _FONT_TITLE
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=12)
    ws.row_dimensions[1].height = 26
    _span_text(ws, 2, 1, 12,
               "主表每行一个算子；本表按 term 展开复合/重复项，并列出完整张量 shape、"
               "源码定位与参数说明。表格中的数值/公式是主表同一事实的追溯副本，不另计成本。",
               font=_FONT_SMALL)

    # 参数来源表
    header = 4
    param_headers = ("参数id", "符号", "适用层组", "公式/来源", "说明", "影响本粗估成本")
    for idx, label in enumerate(param_headers, start=1):
        c = ws.cell(row=header, column=idx, value=label)
        c.font = _FONT_HEAD
        c.fill = _FILL_HEAD
        c.border = _BORDER
        c.alignment = _ALIGN_CENTER
    row = header + 1
    seen = set()
    mdep = _m_dependent_ids(spec["sheets"][0]["groups"])
    for sheet in spec["sheets"][:1]:
        for group in sheet["groups"]:
            for param in group["parameters"]:
                pid = param["id"]
                if pid in ("B", "seqlen", "kvcachelen"):
                    continue
                scope = ("HCA" if group["id"] == "hca-topk" else "CSA") if pid in mdep else "通用"
                key = (pid, scope, str(param.get("formula")), str(param.get("source")),
                       str(param.get("value")))
                if key in seen:
                    continue
                seen.add(key)
                ws.cell(row=row, column=1, value=pid).font = _FONT_BODY
                ws.cell(row=row, column=2, value=param.get("symbol")).font = _FONT_BODY
                ws.cell(row=row, column=3, value=scope).font = _FONT_BODY
                formula = param.get("formula") or param.get("expression") or (
                    str(param.get("value")) if param.get("value") is not None else "未知（未提供）")
                src = param.get("source")
                if src:
                    formula = f"{formula}；来源：{src}"
                ws.cell(row=row, column=4, value=formula).font = _FONT_BODY
                ws.cell(row=row, column=5, value=param.get("note", "")).font = _FONT_SMALL
                ws.cell(row=row, column=6,
                        value="是" if pid in _COST_RELEVANT else "否（仅追溯）").font = _FONT_BODY
                for col in range(1, 7):
                    cell = ws.cell(row=row, column=col)
                    cell.border = _BORDER
                    cell.alignment = _ALIGN_WRAP
                    if isinstance(cell.value, str):
                        cell.data_type = "s"
                ws.row_dimensions[row].height = 30
                row += 1

    # 逐算子明细
    row += 1
    _prep_span(ws, row, 1, 12, _FILL_SECTION)
    ws.cell(row=row, column=1, value="逐算子明细（按 term 展开）").font = _FONT_SECTION
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=12)
    row += 1
    op_headers = (
        "工作表", "层组", "通路", "op id", "op 名称", "op 类型", "模块路径",
        "源码定位", "输入 shape", "输出 shape", "权重 shape", "参数引用",
        "计费项", "B/批", "M", "K", "N", "repeat", "n", "T", "D",
        "条件/守卫/排除", "说明", "主表行",
    )
    for idx, label in enumerate(op_headers, start=1):
        c = ws.cell(row=row, column=idx, value=label)
        c.font = _FONT_HEAD
        c.fill = _FILL_HEAD
        c.border = _BORDER
        c.alignment = _ALIGN_CENTER
    row += 1
    for info in op_rows:
        group = info.get("group")
        if group is None:
            continue
        op = info["op"]
        rule = op["performance"]
        terms = info.get("terms") or []
        if not terms:
            terms = [{}]
        for index, term in enumerate(terms, start=1):
            values = (
                info["sheet"],
                group["id"],
                _path_label(group["id"], op),
                op["id"],
                op.get("name", ""),
                {"cube": "CUBE", "vec": "VEC", "container": "容器"}.get(
                    rule.get("class"), rule.get("class", "")),
                op.get("module_path", ""),
                _source_text(op.get("source_refs", [])),
                " | ".join(_tensor_label(group, tid) for tid in op.get("inputs", [])),
                " | ".join(_tensor_label(group, tid) for tid in op.get("outputs", [])),
                " | ".join(_tensor_label(group, tid) for tid in op.get("weights", [])),
                ", ".join(op.get("parameter_refs", [])),
                term.get("label", "—") if term else "—",
                term.get("b", "—") if term else "—",
                term.get("m", "—") if term else "—",
                term.get("k", "—") if term else "—",
                term.get("n", "—") if term else "—",
                rule.get("repeat", 1),
                _vec_map(term)["n"] if term and rule.get("class") == "vec" else "—",
                _vec_map(term)["T"] if term and rule.get("class") == "vec" else "—",
                _vec_map(term)["D"] if term and rule.get("class") == "vec" else "—",
                _detail_flags(rule),
                (rule.get("note", "")
                 + ("；" + _m_fold_note(rule) if _m_fold_note(rule) else "")
                 + ("；" + op.get("note", "") if op.get("note") else "")),
                info.get("row", ""),
            )
            for idx, value in enumerate(values, start=1):
                c = ws.cell(row=row, column=idx, value=value)
                c.font = _FONT_SMALL if idx not in (4, 5) else _FONT_BODY
                c.border = _BORDER
                c.alignment = _ALIGN_WRAP
                if isinstance(c.value, str):
                    c.data_type = "s"
            ws.row_dimensions[row].height = 30
            row += 1
    ws.freeze_panes = "A5"
    ws.print_title_rows = f"{header}:{header}"


def _m_fold_note(rule: dict) -> str:
    if rule.get("class") != "cube":
        return ""
    notes = []
    for term in rule.get("terms", []):
        b = term.get("b")
        if b in (None, "B", "1"):
            continue
        if b in ("B*Nh", "B*Ih", "B*Nkv"):
            notes.append(f"M=({b})×({term['m']})，head 只折入 M 一次")
        elif b == "B*G":
            notes.append(f"M=({b})×({term['m']})，分组 G 只折入 M 一次")
        elif b == "B*seqlen":
            notes.append(f"M=({b})×({term['m']})，batch×T 只折入 M 一次")
        else:
            notes.append(f"M=({b})×({term['m']})，批量因子只折入 M 一次")
    return "；".join(notes)


def _detail_flags(rule: dict) -> str:
    flags = []
    if rule.get("class") == "container":
        flags.append("容器：不计成本")
    if (rule.get("charge") or {}).get("mode") == "excluded":
        flags.append("整网一次性：不计入层组小计")
    if rule.get("conditional"):
        flags.append("条件：" + rule["conditional"].get("when_text", ""))
    if rule.get("repeat") not in (None, 1, "1"):
        flags.append(f"repeat={rule['repeat']}")
    if rule.get("guards"):
        flags.append("守卫：" + "；".join(rule["guards"]))
    return "；".join(flags) or "—"


# ---------------------------------------------------------------- 入口


def build_perf_layout_workbook(container: dict, network: dict | None = None) -> bytes:
    """真实结构容器 → 新 12 列样表字节（Prefill/Decode/辅助明细）。"""
    spec = build_structures_workbook_spec(container, network)
    wb = Workbook()
    wb.iso_dates = True
    wb.remove(wb.active)
    entries, by_key, mdep = _build_entries(spec)
    config = _Config(entries=entries, by_key=by_key)
    op_rows: list[dict] = []
    for sheet in spec["sheets"]:
        ws = wb.create_sheet(sheet["name"])
        _write_main_sheet(ws, spec, sheet, config, op_rows)
    _write_detail(wb, spec, op_rows)
    import io
    buffer = io.BytesIO()
    wb.save(buffer)
    # 与唯一导出器一致：固定 ZIP 条目顺序/时间戳/权限，保证相同输入字节一致。
    return _normalize_zip(buffer.getvalue())


def export_perf_layout_workbook(container: dict, network: dict | None = None) -> bytes:
    """兼容命名：与 ``build_perf_layout_workbook`` 相同。"""
    return build_perf_layout_workbook(container, network)
