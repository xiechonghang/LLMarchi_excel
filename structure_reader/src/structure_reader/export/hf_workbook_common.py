"""HF 结构工作簿的模型无关部分（唯一导出实现之上的一层薄公共声明）。

职责：
- 两页场景（Prefill/Decode）与可填写性能配置行；
- 参数行组装（Block 参数 + 场景派生式 → 可填写输入/活公式/未知）；
- 层组声明（真实层归属、循环次数、模板说明）；
- 整网归属区（外围算子/层组/层数/成本待提供）。

不创建 Workbook、不写 sheet、不保存文件：这些仍只在
``export/operator_workbook.export_workbook`` 一处。DeepSeek 与 GLM 的适配模块
只声明各自的映射与文案，避免把某个模型的 43 层/四类/特有参数规则硬塞给另一个。
"""
from __future__ import annotations

__all__ = [
    "SHEET_SCENARIOS",
    "PERFORMANCE",
    "sheet_inputs",
    "throughput_inputs",
    "pure_expression",
    "derived_note",
    "parameter_rows",
    "group_from_block",
    "sheet_notes",
    "network_section",
    "sheet",
]

# 文本模型两页：同一结构的两个可改场景。示例 Prefill(8,0)、Decode(1,4096)
# 只是预设取值，不是用户给定的实际负载。
SHEET_SCENARIOS = (
    ("prefill", "Prefill", "空缓存示例场景", "prefill-empty"),
    ("decode", "Decode", "非零历史示例场景", "decode-nonzero"),
)

PERFORMANCE = {
    "status": "待提供",
    "note": ("用户性能公式与 XPU 能力参数尚未提供：本表不生成猜测耗时、"
             "不把缺失成本记为 0，也不用忽略空白的 SUM 冒充汇总"),
}


def derived_note(note: str) -> str:
    suffix = "派生值由 Excel 公式按本表输入重算；未给定输入保持未知，不默认 0。"
    return f"{note} {suffix}".strip()


def _throughput_inputs(engine: dict) -> list[dict]:
    """用户 CUBE/VEC 粗估的两个可改吞吐（默认 4096 MAC/ms、64 元素/ms）。"""
    return [
        {
            "id": engine["cube_throughput_input"], "symbol": "Tcube",
            "label": "CUBE 吞吐（MAC/ms，可改）",
            "value": engine["cube_default"], "kind": "number",
            "source": "用户 2026-10-29 确认：B1/M1/K64/N64 = 1 ms",
            "note": ("CUBE 耗时 = B·M·K·N ÷ 本单元格；默认 4096 MAC/ms 使 "
                     "1·1·64·64/4096 = 1 ms。按比例、不向 64 取整；"
                     "≤0、空白或文本时相关成本显示未知，不按 0 计。"),
            "editable": True,
        },
        {
            "id": engine["vec_throughput_input"], "symbol": "Tvec",
            "label": "VEC 吞吐（元素/ms，可改）",
            "value": engine["vec_default"], "kind": "number",
            "source": "用户 2026-10-29 确认：M1/N64 = 1 ms",
            "note": ("VEC 耗时 = M·N ÷ 本单元格；默认 64 元素/ms 使 "
                     "1·64/64 = 1 ms。按比例、不向 64 取整；"
                     "≤0、空白或文本时相关成本显示未知，不按 0 计。"),
            "editable": True,
        },
    ]


def throughput_inputs(engine: dict) -> list[dict]:
    """CUBE/VEC 两个可改吞吐输入行（模型无关；HiDream 单页也复用它）。"""
    return _throughput_inputs(engine)


def sheet_inputs(scenario: dict, *, source_suffix: str = "SCENARIO_PRESETS",
                 performance_engine: dict | None = None) -> list[dict]:
    """每页一份可填写配置；同一页所有层组引用同一组输入。"""
    scenario_note = scenario.get("note", "")
    rows = [
        {
            "id": "seqlen", "symbol": "seqlen", "label": "本次输入长度 seqlen",
            "value": scenario.get("seqlen"), "kind": "integer",
            "source": f"场景示例 {scenario['id']}（{source_suffix}）",
            "note": ("示例场景，可改；seqlen 是本次新 token 数，不含历史。"
                     + (f" {scenario_note}" if scenario_note else "")),
            "editable": True,
        },
        {
            "id": "kvcachelen", "symbol": "kvcachelen",
            "label": "调用前逻辑历史长度 kvcachelen",
            "value": scenario.get("kvcachelen"), "kind": "integer",
            "source": f"场景示例 {scenario['id']}（{source_suffix}）",
            "note": ("示例场景，可改；kvcachelen 是逻辑历史 token 总数，"
                     "不是物理保留长度或缓存字节数。"),
            "editable": True,
        },
        {
            "id": "batch_size", "symbol": "B", "label": "batch size",
            "value": 1, "kind": "integer", "source": "示例 batch=1（可改）",
            "note": "示例性能配置，可改；不声称代表用户实际负载。",
            "editable": True,
        },
        {
            "id": "compute_precision", "symbol": "精度", "label": "计算精度",
            "value": "FP32", "kind": "text",
            "source": "本工程 FP32 阅读特化（可改）",
            "note": (("默认来自静态阅读的 FP32 特化，可改为用户实际精度；"
                      "用户 CUBE/VEC 粗估规则不含精度折扣，改本项不改变耗时。")
                     if performance_engine else
                     ("默认来自静态阅读的 FP32 特化，可改为用户实际精度；"
                      "用户性能公式未提供，不据此计算耗时。")),
            "editable": True,
        },
        {
            "id": "weight_storage_bits", "symbol": "Wbits",
            "label": "权重存储位宽（bit）", "value": 32, "kind": "integer",
            "source": "示例 32 bit（可改）",
            "note": (("示例性能配置，可改；用户 CUBE/VEC 粗估规则不含带宽/位宽项，"
                      "改本项不改变耗时。")
                     if performance_engine else
                     "示例性能配置，可改；仅为用户公式准备的输入，不对应已核实硬件能力。"),
            "editable": True,
        },
        {
            "id": "activation_storage_bits", "symbol": "Abits",
            "label": "激活存储位宽（bit）", "value": 32, "kind": "integer",
            "source": "示例 32 bit（可改）",
            "note": (("示例性能配置，可改；用户 CUBE/VEC 粗估规则不含带宽/位宽项，"
                      "改本项不改变耗时。")
                     if performance_engine else
                     "示例性能配置，可改；仅为用户公式准备的输入，不对应已核实硬件能力。"),
            "editable": True,
        },
        {
            "id": "expert_hit", "symbol": "N_hit",
            "label": "路由专家实际命中数（数据相关）",
            "value": None, "kind": "integer",
            "source": "待确定（不根据权重推断）",
            "note": ("数据相关量，尚未给定；保持未知，不默认 0 或 K，"
                     "也不根据权重数值推断 MoE 实际路由。"
                     + (" 本表把 N_hit 当作各层统一的命中数假设，用于专家权重切片"
                        "（每命中专家固定读取量）的成本：必须 ≤ min(E, B·seqlen·K)，"
                        "非空调用还要 ≥ 每 token 命中数（TopK 为 K、Hash 为 1）；"
                        "不满足时相关成本与小计显示未知，不给不可实现的完整总计。"
                        "实际路由分布不由本工具推断。"
                        if performance_engine else "")),
            "editable": True,
        },
    ]
    if performance_engine is not None:
        rows.extend(_throughput_inputs(performance_engine))
    return rows


def pure_expression(expression) -> str | None:
    """从参数表的展示表达式提取**可编译**的纯表达式；提取不到返回 None。

    结构常量（如 ``Rk = QkRopeHeadDim`` 已有核验值 64）的展示表达式不是白名单
    语法：此时保留已核验字面量，不把常量改成未知/公式。
    """
    from structure_reader.scenario_evaluator import ExpressionError, expression_names

    if not isinstance(expression, str) or " = " not in expression:
        return None
    text = expression.split(" = ", 1)[1]
    text = text.split("（", 1)[0].strip()
    if not text:
        return None
    try:
        expression_names(text)
    except ExpressionError:
        return None
    return text


def _extra_parameter(derivation: dict, extra_id: str, derived_source: list[dict]) -> dict:
    return {
        "id": extra_id,
        "symbol": derivation["id"],
        "meaning": derivation.get("label", derivation["id"]),
        "value": None,
        "expression": derivation.get("expression", ""),
        "note": derived_note(derivation.get("note", "")),
        "source_refs": [dict(ref) for ref in derived_source],
        "value_ref": None,
        "formula": derivation.get("expression", ""),
        "unknown_reason": "",
        "axis_aliases": [extra_id] if extra_id != derivation["id"] else [],
    }


def parameter_rows(block: dict, *, input_ids: dict[str, str],
                   derivation_to_param: dict[str, str],
                   extra_param_ids: dict[str, str],
                   derived_source: list[dict]) -> list[dict]:
    """把 Block 参数 + 场景派生式声明成工作簿参数行（不复制第二套事实）。"""
    derivations = {item["id"]: item for item in block.get("derivations", [])}
    # 映射方向：derivation_id → 既有参数行 id（与适配声明一致）；同一行不重复建。
    param_of_derivation = {did: pid for did, pid in derivation_to_param.items()}
    derivation_of_param = {pid: did for did, pid in param_of_derivation.items()}
    used_derivations: set[str] = set()
    rows: list[dict] = []
    for param in block.get("parameters", []):
        row = {
            "id": param["id"],
            "symbol": param["symbol"],
            "meaning": param["meaning"],
            "value": param.get("value"),
            "expression": param.get("expression"),
            "note": param.get("note", ""),
            "source_refs": [dict(ref) for ref in param.get("source_refs", [])],
            "value_ref": None,
            "formula": None,
            "unknown_reason": param.get("unknown_reason", ""),
            "axis_aliases": ([param["id"]] if param["id"] != param["symbol"] else []),
            **({"domain": dict(param["domain"])} if param.get("domain") else {}),
            # 可选一致性约束（例如 K×S_i=Svis、尺寸按 patch 对齐）：导出器把它编译成
            # 守卫，不满足时依赖量未知；旧模型没有该键，行内容不变。
            **({"consistency": param["consistency"]} if param.get("consistency") else {}),
        }
        input_id = input_ids.get(param["id"])
        if input_id is not None:
            row["value"] = None
            row["value_ref"] = input_id
            row["expression"] = f"引用本表性能配置 {input_id}（示例值可改）"
            row["note"] = derived_note(
                f"{row['note']}；取值来自本表上方可填写的 {input_id} 单元格。")
        else:
            mapped = derivation_of_param.get(param["id"])
            if mapped is not None and mapped in derivations:
                derivation = derivations[mapped]
                row["value"] = None
                row["formula"] = derivation.get("expression", "")
                row["expression"] = derivation.get("expression", "")
                row["note"] = derived_note(row["note"])
                used_derivations.add(mapped)
            else:
                pure = pure_expression(param.get("expression"))
                if pure is not None:
                    row["value"] = None
                    row["formula"] = pure
                    row["expression"] = pure
                    row["note"] = derived_note(row["note"])
        rows.append(row)

    for derivation in block.get("derivations", []):
        if derivation["id"] in used_derivations:
            continue
        extra_id = extra_param_ids.get(derivation["id"])
        if extra_id is None:
            continue
        if any(row["id"] == extra_id for row in rows):
            raise ValueError(
                f"派生参数 {derivation['id']} 的目标行 {extra_id!r} 已存在；"
                "不静默生成重复参数行")
        rows.append(_extra_parameter(derivation, extra_id, derived_source))
        used_derivations.add(derivation["id"])
    return rows


def group_from_block(block: dict, *, group_id: str, label: str,
                     layer_indices: list[int], representative: int,
                     loop: dict, template_note: str, scope_note: str,
                     input_ids: dict[str, str],
                     derivation_to_param: dict[str, str],
                     extra_param_ids: dict[str, str],
                     derived_source: list[dict]) -> dict:
    return {
        "id": group_id,
        "label": label,
        "attention_type": block.get("attention_type", ""),
        "moe_type": block.get("moe_type", ""),
        "indexer_type": block.get("indexer_type", ""),
        "layer_indices": list(layer_indices),
        "representative_layer_index": representative,
        "loop": loop,
        "template_note": template_note,
        "scope_note": scope_note,
        "parameters": parameter_rows(
            block, input_ids=input_ids, derivation_to_param=derivation_to_param,
            extra_param_ids=extra_param_ids, derived_source=derived_source),
        "tensors": block.get("tensors", []),
        "operators": block.get("operators", []),
        "cache_notes": block.get("cache_notes", []),
        "coverage_notes": block.get("coverage_notes", []),
    }


def sheet_notes(scenario: dict, performance_engine: dict | None = None) -> list[str]:
    performance_note = (
        "性能按用户 2026-10-29 确认的 CUBE/VEC 粗估：CUBE=B·M·K·N/吞吐、"
        "VEC=M·N/吞吐（默认 4096 MAC/ms、64 元素/ms，表内可改）；"
        "连续按比例、不向 64 取整；未知/非法输入不按 0 计。"
        if performance_engine is not None else
        "性能公式与 XPU 能力待提供：本表不生成猜测耗时，不把缺失成本记为 0。")
    return [
        "示例场景只是可改取值：Prefill(8,0)、Decode(1,4096)；结构 shape 仍保留符号 "
        "seqlen/kvcachelen，不另维护一套图。",
        f"本页场景：{scenario['label']}。" + (
            f" {scenario['note']}" if scenario.get("note") else ""),
        "kvcachelen 是调用前已处理的逻辑历史 token 总数；不是物理保留长度，也不是缓存字节数。",
        performance_note,
    ]


def network_section(network: dict, *, label: str, notes: list[str],
                    count_note: str | None = None,
                    performance_engine: dict | None = None,
                    serial_order: list[dict] | None = None) -> dict:
    scope = network["scope"]
    path = network["path"]
    operators = {item["id"]: item for item in path.get("operators", [])}
    periphery = []
    for oid in scope.get("main_path", []):
        item = operators.get(oid)
        if item is None:
            continue
        periphery.append({
            "id": oid,
            "name": item.get("name", oid),
            "module_path": item.get("module_path", ""),
        })
    groups = [{
        "id": group["id"],
        "label": group["label"],
        "layer_indices": list(group["layer_indices"]),
        "count": group["count"],
    } for group in network.get("groups", [])]
    section = {
        "label": label,
        "periphery": periphery,
        "groups": groups,
        "layer_count": scope["layer_count"],
        "notes": notes,
        "cost_status": ("已按用户 CUBE/VEC 规则串行汇总（口径见性能区；不填猜测量）"
                        if performance_engine is not None else
                        "待提供（用户公式/XPU 未提供；不填 0）"),
        **({"count_note": count_note} if count_note else {}),
    }
    if serial_order is not None:
        section["serial_order"] = [dict(entry) for entry in serial_order]
    return section


def sheet(phase_id: str, name: str, title: str, scenario: dict,
          groups: list[dict], network: dict | None, *, net_label: str,
          net_notes: list[str], input_source_suffix: str = "SCENARIO_PRESETS",
          performance_engine: dict | None = None,
          periphery_group: dict | None = None,
          serial_order: list[dict] | None = None,
          count_note: str | None = None) -> dict:
    blocks = list(groups)
    if periphery_group is not None:
        blocks.append(periphery_group)
    return {
        "id": phase_id,
        "name": name,
        "title": title,
        "notes": sheet_notes(scenario, performance_engine),
        "inputs": sheet_inputs(scenario, source_suffix=input_source_suffix,
                               performance_engine=performance_engine),
        "groups": blocks,
        "network": (network_section(network, label=net_label, notes=net_notes,
                                    count_note=count_note,
                                    performance_engine=performance_engine,
                                    serial_order=serial_order)
                    if network is not None else None),
    }
