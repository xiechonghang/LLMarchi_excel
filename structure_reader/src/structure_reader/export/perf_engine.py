"""模型无关的 CUBE/VEC 粗估引擎声明与规则构造器（唯一导出器的公共基座）。

用户 2026-10-29 确认的规则（T10，DeepSeek 首版；T11 覆盖其余四模型）：

- CUBE（含矩阵乘/线性投影/批量矩阵乘/矩阵-向量收缩的算子）：
  ``耗时 ms = B·M·K·N / 吞吐``；基准 ``B1/M1/K64/N64 = 1 ms`` ⇒ 默认 4096 MAC/ms。
- VEC（不含矩阵计算的算子）：``耗时 ms = M·N / 吞吐``；基准 ``M1/N64 = 1 ms``
  ⇒ 默认 64 元素/ms。
- 连续按比例计算，不向 64 取整；不加带宽/并行/融合/通信/精度折扣。

本模块只提供**模型无关**的声明与构造器：默认吞吐、通用文案、``cube()``/``vec()``/
``container()`` 规则构造、规则符号提取、可审查清单、以及“本组每个算子都有规则且
符号都在本组声明”的绑定校验。各模型模块只声明自己的语义（分类/维度/来源），
DeepSeek 的 ``deepseek_perf_rules`` 与本票新增的四个模型规则模块都复用它；
不存在第二套解析或写表流程。

维度口径（各模型 note 里写具体源码依据）：

- Linear（``Y = X·Wᵀ``，权重存 ``[N,K]``）：``B=batch, M=序列/行数, K=in, N=out``，
  batch 只出现一次。
- 注意力打分/加权：头数作矩阵 batch（``B=B·Nh``），``M=查询位置数``，
  ``K``/``N`` 为 head_dim 或可见 KV 轴。
- 卷积：``B_eff=batch·groups, M=输出位置数, K=(Cin/groups)·核体积, N=Cout/groups``。
- 复合算子：矩阵部分计入 CUBE，所有矩阵项都列入 terms；伴随 view/激活等不另收费
  （note 写明）。同一算子的多个矩阵项各占一行（分项）。
- MoE 专家模板：用源码可证明的总路由分配数 ``B·seqlen·K`` 表达专家计算总量；
  依赖运行期命中专家数的项使用本表 ``N_hit`` 输入（默认未知，带相容守卫）。
"""
from __future__ import annotations

import copy

__all__ = [
    "ENGINE",
    "PERFORMANCE_STATUS",
    "PERFORMANCE_NOTE",
    "THROUGHPUT_NOTE",
    "DISCLAIMER",
    "ENGINE_DEFAULTS",
    "cube",
    "vec",
    "cube_terms",
    "vec_terms",
    "container",
    "with_extra",
    "engine_defaults",
    "performance_spec",
    "disclaimer",
    "declared_symbols",
    "attach_performance",
    "rule_symbols",
    "rules_report",
    "periphery_parameters",
    "expert_hit_rule",
]

# 用户确认的默认吞吐（表内可改，见每页输入区的 CUBE/VEC 吞吐行）。
ENGINE = {
    "kind": "cube_vec",
    "cube_throughput_input": "cube_mac_per_ms",
    "vec_throughput_input": "vec_elem_per_ms",
    "cube_default": 4096,
    "vec_default": 64,
    "cube_unit": "MAC/ms",
    "vec_unit": "元素/ms",
    "source": ("用户 2026-10-29 确认：CUBE B1/M1/K64/N64 = 1 ms（4096 MAC/ms）；"
               "VEC M1/N64 = 1 ms（64 元素/ms）；按比例、不向 64 取整"),
}

# 兼容别名（工作簿写入只需要这两个默认值）。
ENGINE_DEFAULTS = {"cube": 4096, "vec": 64}

PERFORMANCE_STATUS = "已按用户 CUBE/VEC 规则接入（2026-10-29 确认）"
PERFORMANCE_NOTE = (
    "CUBE 耗时 = B·M·K·N ÷ 本表 CUBE 吞吐（默认 4096 MAC/ms）；"
    "VEC 耗时 = M·N ÷ 本表 VEC 吞吐（默认 64 元素/ms）；连续按比例、不向 64 取整；"
    "吞吐与 B/seqlen/kvcachelen 均为可改单元格，改后由 Excel 重算；"
    "不计带宽、并行、融合、通信与精度折扣，不等同实测性能")
THROUGHPUT_NOTE = (
    "用户确认基准：CUBE B1/M1/K64/N64 = 1 ms（4096 MAC/ms）；VEC M1/N64 = 1 ms"
    "（64 元素/ms）；连续按比例、不向 64 取整。可改本单元格，公式随改随算；"
    "≤0、空白或文本时相关成本显示未知，不按 0 计")
DISCLAIMER = (
    "本工作簿由 structure_reader 唯一导出链生成：结构、参数、层组归属来自同源事实文档；"
    "CUBE/VEC 分类与维度映射是本模型适配层声明的语义规则，导出器只按规则写公式。"
    "耗时=用户 2026-10-29 确认的粗估规则（CUBE B·M·K·N/吞吐、VEC M·N/吞吐），"
    "不含带宽/并行/融合/通信/精度折扣，不等同实测；未知或非法输入显示“未知（原因）”，"
    "不默认 0，也不用忽略空白的 SUM 冒充汇总。"
)


def engine_defaults() -> dict:
    """一份独立的引擎声明副本（模型适配层不共享可变对象）。"""
    return copy.deepcopy(ENGINE)


def performance_spec(status: str | None = None, note: str | None = None) -> dict:
    return {
        "status": PERFORMANCE_STATUS if status is None else status,
        "note": PERFORMANCE_NOTE if note is None else note,
        "engine": engine_defaults(),
    }


def disclaimer(model_label: str = "") -> str:
    if model_label:
        return DISCLAIMER.replace("本模型适配层", f"{model_label} 适配层")
    return DISCLAIMER


# ---------------------------------------------------------------- 规则构造


def with_extra(rule: dict, repeat=None, conditional=None, charge=None, guards=None,
               guard_text: str = "") -> dict:
    if repeat is not None:
        rule["repeat"] = repeat
    if conditional is not None:
        rule["conditional"] = conditional
    if charge is not None:
        rule["charge"] = charge
    if guards is not None:
        rule["guards"] = list(guards)
    if guard_text:
        rule["guard_text"] = guard_text
    return rule


def cube(b: str, m: str, k: str, n: str, note: str, *, label: str = "矩阵乘",
         repeat=None, conditional=None, charge=None, guards=None,
         guard_text: str = "") -> dict:
    """单矩阵项 CUBE 规则：工作量 = B·M·K·N（×repeat）。"""
    term = {"label": label, "b": b, "m": m, "k": k, "n": n}
    return with_extra({"class": "cube", "terms": [term], "note": note},
                      repeat, conditional, charge, guards, guard_text)


def cube_terms(terms: list[dict], note: str, *, repeat=None, guards=None,
               guard_text: str = "", conditional=None) -> dict:
    """复合算子的多个矩阵项：每项一行、各行工作量分别写公式。"""
    rule = {"class": "cube", "terms": [dict(term) for term in terms], "note": note}
    return with_extra(rule, repeat, conditional, None, guards, guard_text)


def vec(m: str, n: str, note: str, *, label: str = "元素操作", repeat=None,
        conditional=None, charge=None, guards=None, guard_text: str = "") -> dict:
    """VEC 规则：工作量 = M·N（×repeat）。"""
    term = {"label": label, "m": m, "n": n}
    return with_extra({"class": "vec", "terms": [term], "note": note},
                      repeat, conditional, charge, guards, guard_text)


def vec_terms(terms: list[dict], note: str, *, repeat=None, guards=None,
              guard_text: str = "", conditional=None) -> dict:
    """复合 VEC 项的多个逻辑工作张量：每项一行、各行工作量分别写公式。"""
    rule = {"class": "vec", "terms": [dict(term) for term in terms], "note": note}
    return with_extra(rule, repeat, conditional, None, guards, guard_text)


def container(reason: str) -> dict:
    return {"class": "container", "reason": reason,
            "note": "容器节点：不是可单独计成本的算子；其成本由被包含的层组串行汇总表达。"}


# ---------------------------------------------------------------- 校验/清单


def declared_symbols(group: dict) -> set:
    """本组可被性能表达式引用的符号（参数符号/id/轴别名 + 本表可填写输入）。"""
    symbols = {"B", "seqlen", "kvcachelen", "N_hit"}
    for param in group["parameters"]:
        symbols.add(param["symbol"])
        symbols.add(param["id"])
        symbols.update(param.get("axis_aliases", []))
    return symbols


def rule_symbols(rule: dict) -> list[str]:
    """规则引用的全部符号（b/m/k/n/repeat/conditional.when/guards），供构建期校验。"""
    from structure_reader.scenario_evaluator import expression_names

    names: list[str] = []
    for term in rule.get("terms", []):
        for key in ("b", "m", "k", "n"):
            value = term.get(key)
            if value is not None:
                names.extend(expression_names(value))
    repeat = rule.get("repeat")
    if isinstance(repeat, str):
        names.extend(expression_names(repeat))
    conditional = rule.get("conditional") or {}
    if conditional.get("when"):
        names.extend(expression_names(conditional["when"]))
    for guard in rule.get("guards", []):
        names.extend(expression_names(guard))
    return sorted(set(names))


def attach_performance(group: dict, rules: dict, label: str) -> dict:
    """给本组每个算子挂上 CUBE/VEC 规则（复制算子，不修改同源结构输入）。

    同时校验：本组每个算子都有规则、规则引用的符号都在本组声明；
    缺一个就具名失败，避免用字符串前缀命中少数项便声称全覆盖。
    """
    declared = declared_symbols(group)
    operators = []
    for op in group["operators"]:
        rule = rules.get(op["id"])
        if rule is None:
            raise ValueError(f"{label}: 算子 {op['id']!r} 缺少 CUBE/VEC 分类与维度规则")
        missing = [symbol for symbol in rule_symbols(rule) if symbol not in declared]
        if missing:
            raise ValueError(
                f"{label}: 算子 {op['id']!r} 的 CUBE/VEC 规则引用了本组未声明符号 "
                f"{sorted(missing)}（不静默取未知）")
        operators.append({**op, "performance": copy.deepcopy(rule)})
    enriched = dict(group)
    enriched["operators"] = operators
    return enriched


def _report_row(scope: str, op_id: str, rule: dict) -> dict:
    return {
        "scope": scope, "operator_id": op_id,
        "class": rule["class"],
        "terms": [dict(term) for term in rule.get("terms", [])],
        "repeat": rule.get("repeat", 1),
        "conditional": dict(rule.get("conditional") or {}),
        "charge": dict(rule.get("charge") or {"mode": "cell"}),
        "guards": list(rule.get("guards", [])),
        "guard_text": rule.get("guard_text", ""),
        "note": rule.get("note") or rule.get("reason", ""),
        "symbols": rule_symbols(rule),
    }


def rules_report(block_rules_by_group: dict, network_rules: dict) -> list[dict]:
    """规则清单（可审查/可写 JSON 证据）：按层组与整网外围展开每个算子。"""
    report = []
    for group_id, rules in block_rules_by_group.items():
        for op_id, rule in rules.items():
            report.append(_report_row(group_id, op_id, rule))
    for op_id, rule in network_rules.items():
        report.append(_report_row("net-periphery", op_id, rule))
    return report


# ---------------------------------------------------------------- 通用 MoE / 外围


def expert_hit_rule(*, experts: str = "E", top_k: str = "K", tokens: str = "Ttok",
                    slice_rows: str = "K", lower_text: str = "K",
                    note: str = "", label: str = "命中专家扫描") -> dict:
    """依赖运行期命中专家数 N_hit 的 VEC 项（共用相容守卫的统一样式）。

    相容性来自源码可证明的边界：每 token 至多 K 个专家分配，因此命中专家数
    ≤ min(E, B·seqlen·K)；非空调用时 TopK 至少 K 个不同专家。违反即未知，
    不给“完整总计”。单值是“各层统一命中数”假设，不推断实际路由分布。
    """
    return vec(
        f"N_hit*{slice_rows}", tokens,
        note or ("按每个命中专家扫描完整分配矩阵 [E+1,K,Ttok] 的 [K,Ttok] 切片；"
                 "总量只依赖运行期命中专家数 N_hit（本表输入，默认未知）。"
                 "单值是各层统一命中数假设，实际路由分布不由本工具推断。"),
        label=label,
        guards=[
            f"N_hit <= min({experts}, B*seqlen*{top_k})",
            f"(B*seqlen*{top_k} == 0) or (N_hit >= {slice_rows})",
        ],
        guard_text=("N_hit 必须 ≤ min(E, B·seqlen·K)（总路由分配数），"
                    f"非空调用还需 ≥ {lower_text}；单值是各层统一的命中数假设，"
                    "实际路由不由本工具推断"),
    )


def periphery_parameters(network: dict, *, input_ids: dict,
                         derivation_to_param: dict | None = None,
                         extra_param_ids: dict | None = None,
                         derived_source: list[dict] | None = None) -> list[dict]:
    """整网外围参数行（复用同源 network_overview 参数，不另造一份）。

    ``S_sel`` 的展示式在结构文档里带中文/多分支，不能直接编译；这里按源码语义
    写活为 ``min(seqlen,Lk) + seqlen*(1-min(Lk,1))``（Lk=0 保留全部 seqlen）。
    该修正对 GLM-5.3 / GLM-5.3-Flash / Qwen3.8-Flash-Next / DeepSeek 一致。
    """
    from structure_reader.export.hf_workbook_common import parameter_rows

    rows = parameter_rows(
        network["path"], input_ids=input_ids,
        derivation_to_param=derivation_to_param or {},
        extra_param_ids=extra_param_ids or {},
        derived_source=derived_source or [])
    for row in rows:
        if row["id"] == "Ssel":
            row["formula"] = "min(seqlen, Lk) + seqlen*(1 - min(Lk, 1))"
            row["expression"] = (
                "S_sel = min(seqlen, Lk) + seqlen*(1 - min(Lk, 1))"
                "（Lk=0 时即 seqlen；Lk>0 时 min(seqlen,Lk)）")
            row["note"] = (
                "logits 只对切片后的位置做投影；Lk=0（默认）保留全部 seqlen 个位置，"
                "Lk>0 保留末 min(seqlen,Lk) 个。负整数/tensor 索引不在本分析范围。")
        if row["id"] == "Lk":
            row["domain"] = {"kind": "integer", "min": 0}
            row["note"] = ("logits_to_keep 默认 0；本表按非负整数处理（负整数/tensor "
                           "索引不在本分析范围）。")
    return rows
