"""DeepSeek HF 模型适配：把同一份参数化 Block 结构声明成通用工作簿 spec。

职责边界（T04/T05）：
- **只做模型适配**：固定 Prefill/Decode 两页、DeepSeek 的 scope/层组/场景默认值、
  派生式→参数行的映射都留在这里；不创建 Workbook、不写 sheet、不保存文件。
- 模型无关的公共声明（可填写输入行、参数行组装、整网归属区、两页框架）已提炼到
  ``export/hf_workbook_common.py``，由 DeepSeek / GLM-5.3 适配共享；本模块不含
  第二套导出流程。
- 单层入口与分层入口都构造同一形状的 spec，再交给
  ``export/operator_workbook.export_workbook``（唯一导出实现）。
- 参数值来源：
  - ``seqlen``/``kvcachelen``/``B`` → 引用本表可填写的性能配置单元格；
  - Block 参数里可解析的派生式与 ``derivations`` → 真实 Excel 公式（引用本组参数
    单元格）；公式只使用 ``scenario_evaluator`` 白名单语义；
  - 固定结构常量默认来自已核验配置；
  - 未提供/不可解析的值保持未知，不默认 0。
- 层组循环归属来自真实 ``layer_indices``；模板复用不表示共享权重。成本列统一
  “待提供”，不生成猜测耗时、不做父子重复计数。
"""
from __future__ import annotations

import copy

from structure_reader.export import deepseek_perf_rules as perf_rules
from structure_reader.export import perf_engine
from structure_reader.export.hf_workbook_common import (
    SHEET_SCENARIOS,
    group_from_block as _common_group_from_block,
    sheet as _common_sheet,
)
from structure_reader.extract.hf_scenario import scenario_by_id

__all__ = [
    "SHEET_SCENARIOS",
    "build_structure_workbook_spec",
    "build_structures_workbook_spec",
    "export_single_structure_workbook",
    "export_structures_workbook",
]

# 场景派生 id → 既有参数行 id（复用同一行，不新增重复符号）。
DERIVATION_TO_PARAM = {
    "L_kv": "Lkv",
    "L_mask": "Lmask",
    "L_kv_post": "Lkvp",
    "Tbuf": "Tbuf",
    "Tbuf_post": "Tbufp",
    "N_win": "Nwin",
    "N_comp": "Ncomp",
    "N_win_i": "Nwin_i",
    "N_comp_i": "Ncompi",
    "first_window_position": "Fwp",
    "future_threshold": "Fut",
    "first_query_threshold": "Fqt",
    "Ki_sel": "Kisel",
    "block_bias": "Bbias",
}

# 没有既有参数行、需要补成派生参数行的 id（其余映射复用原行）。
EXTRA_PARAM_IDS = {
    "L_kv_post": "Lkvp",
    "N_comp_i": "Ncompi",
    "first_window_position": "Fwp",
    "future_threshold": "Fut",
    "first_query_threshold": "Fqt",
    "Ki_sel": "Kisel",
    "block_bias": "Bbias",
}

INPUT_IDS = {
    "B": "batch_size",
    "seqlen": "seqlen",
    "kvcachelen": "kvcachelen",
}

_DERIVED_SOURCE = [{
    "path": "structure_reader/src/structure_reader/extract/hf_scenario.py",
    "locator": "evaluate_derivations",
    "kind": "derived",
    "note": "场景派生式；同一份结构按本表输入在 Excel 中重算",
}]


def _group_from_block(block: dict, *, group_id: str, label: str,
                      layer_indices: list[int], representative: int,
                      loop: dict, template_note: str,
                      scope_note: str) -> dict:
    return _common_group_from_block(
        block, group_id=group_id, label=label, layer_indices=layer_indices,
        representative=representative, loop=loop, template_note=template_note,
        scope_note=scope_note, input_ids=INPUT_IDS,
        derivation_to_param=DERIVATION_TO_PARAM,
        extra_param_ids=EXTRA_PARAM_IDS, derived_source=_DERIVED_SOURCE)


def _performance_spec() -> dict:
    """DeepSeek 的性能区声明：用户 CUBE/VEC 规则 + 两个可改吞吐输入。"""
    return {
        "status": perf_rules.PERFORMANCE_STATUS,
        "note": perf_rules.PERFORMANCE_NOTE,
        "engine": copy.deepcopy(perf_rules.ENGINE),
    }


def _attach_performance(group: dict, rules: dict, label: str) -> dict:
    """给本组每个算子挂 CUBE/VEC 规则；公共基座负责全覆盖与符号声明校验。"""
    return perf_engine.attach_performance(group, rules, label)


def _validate_deepseek_attribution(container: dict) -> None:
    """DeepSeek 适配仍严格校验真实层归属；通用导出本身不强制 Decoder 层号。"""
    variants = container.get("variants", [])
    if not variants:
        raise ValueError("DeepSeek 容器没有 variants")
    seen: list[int] = []
    for variant in variants:
        layers = list(variant.get("layer_indices", []))
        if not layers:
            raise ValueError(f"层组 {variant.get('id')} 缺少真实 layer_indices")
        representative = variant.get("representative_layer_index")
        if representative is None or representative not in layers:
            raise ValueError(
                f"层组 {variant.get('id')} 的代表层必须属于真实 layer_indices")
        overlap = set(seen) & set(layers)
        if overlap:
            raise ValueError(f"层组 {variant.get('id')} 与其它组层索引重叠: {sorted(overlap)}")
        seen.extend(layers)
    if sorted(seen) != list(range(len(seen))):
        raise ValueError("DeepSeek 层组未按真实层归属恰好覆盖连续层集合")


def _variant_group(variant: dict, scenario: dict) -> dict:
    structure = variant["structure"]
    block = structure["block"]
    layers = list(variant["layer_indices"])
    group = _group_from_block(
        block,
        group_id=variant["id"],
        label=variant["label"],
        layer_indices=layers,
        representative=variant["representative_layer_index"],
        loop={
            "count": len(layers),
            "label": "真实层组循环（逐层实例化）",
            "condition": "同组真实层按顺序各执行一次；本表只保留代表层结构模板，不复制 43 层行",
            "note": "循环归属来自真实 layer_indices，模板复用不表示共享权重。",
        },
        template_note="同组每层是独立 DeepseekV4DecoderLayer 实例；模板复用不代表共享权重。",
        scope_note=(f"分析代表层 {variant['representative_layer_index']} 的 Block 结构；"
                    f"真实层索引 [{', '.join(str(i) for i in layers)}]，"
                    f"共 {len(layers)} 层，均属本组同构模板。"),
    )
    return _attach_performance(group, perf_rules.block_rules(variant["id"]),
                               f"层组 {variant['id']}")


def _single_group(structure: dict, scenario: dict) -> dict:
    block = structure["block"]
    scope = structure["scope"]
    layer = scope["layer_index"]
    group = _group_from_block(
        block,
        group_id=f"layer-{layer}",
        label=f"单层 layer {layer}（{scope['attention_type']} + {scope['moe_type']}）",
        layer_indices=[layer],
        representative=layer,
        loop={
            "count": 1,
            "label": "单层阅读表",
            "condition": "本表只分析该层一次；整网四类层组循环见分层入口/整网归属",
            "note": "单层入口不汇总整网循环，避免与分层入口重复计数。",
        },
        template_note="单层样板：同一 Block 模板复用不代表共享权重。",
        scope_note=(f"单层范围 layer_index={layer}；attention_type={scope['attention_type']}；"
                    f"moe_type={scope['moe_type']}；整网层组归属见分层入口。"),
    )
    return _attach_performance(group, perf_rules.block_rules(
        perf_rules.group_id_for_block(block)), f"单层 layer {layer}")


def _deepseek_network_notes() -> list[str]:
    return [
        "网络外围（Embedding/位置/mask/RoPE/mHC 扩展/HyperHead/Final Norm/logits slice/"
        "LM head）与 43 层真实层序、四类层组归属同源；层内结构由各层组代表层 Block 表示。",
        "同构模板复用不表示共享权重：同组每层独立实例；tie_word_embeddings=false，"
        "embed_tokens 与 lm_head 权重也独立。",
        "Hash MoE 层（0/1/2）的路由查表消费网络入口 input_ids；T03 已核层组为 2/1/20/20，共 43 层。",
        "串行汇总顺序：入口外围 6 项 → swa-hash(2) → csa-hash(1) → hca-topk(20) → "
        "csa-topk(20) → 出口外围 8 项；层组用“单层小计×真实层数”，父小计与算子行不重复计入。",
        "mask 与 main/compress RoPE 频率在 model.forward 只构建一次并传给全部层"
        "（modeling_deepseek_v4.py:1344-1362）：层组内对应行标为“整网一次性项”的引用，"
        "不计入层组小计，整网只在外围 net_mask_build/net_rope_main/net_rope_compress 收一次；"
        "43 层循环容器行成本为 0，不另收费。",
    ]


def _sheet(phase_id: str, name: str, title: str, scenario: dict,
           groups: list[dict], network: dict | None) -> dict:
    return _common_sheet(
        phase_id, name, title, scenario, groups, network,
        net_label="整网串行汇总（来自 network_overview；不铺开 43 层）",
        net_notes=_deepseek_network_notes(),
        performance_engine=perf_rules.ENGINE,
        periphery_group=(perf_rules.network_group(network)
                         if network is not None else None),
        serial_order=(perf_rules.serial_order() if network is not None else None),
        count_note=None)


def build_structures_workbook_spec(container: dict, network: dict | None = None) -> dict:
    """四层组容器 → 两页工作簿 spec（每页四组，同一结构不同场景）。"""
    _validate_deepseek_attribution(container)
    sheets = []
    for phase_id, name, subtitle, scenario_id in SHEET_SCENARIOS:
        scenario = scenario_by_id(scenario_id)
        groups = [_variant_group(variant, scenario) for variant in container["variants"]]
        sheets.append(_sheet(
            phase_id, name, f"{name}（{subtitle}）", scenario, groups, network))
    return {
        "schema": "operator_workbook.v2",
        "model": container["model"],
        "performance": _performance_spec(),
        "disclaimer": perf_rules.DISCLAIMER,
        "sheets": sheets,
    }


def build_structure_workbook_spec(structure: dict, network: dict | None = None) -> dict:
    """单层结构 → 两页工作簿 spec（每页一组；与分层入口共用同一导出能力）。"""
    sheets = []
    for phase_id, name, subtitle, scenario_id in SHEET_SCENARIOS:
        scenario = scenario_by_id(scenario_id)
        sheets.append(_sheet(
            phase_id, name, f"{name}（{subtitle}，单层样板）", scenario,
            [_single_group(structure, scenario)], network))
    return {
        "schema": "operator_workbook.v2",
        "model": structure["model"],
        "performance": _performance_spec(),
        "disclaimer": perf_rules.DISCLAIMER,
        "sheets": sheets,
    }


def export_structures_workbook(container: dict, network: dict | None = None) -> bytes:
    """四层组工作簿字节；导出器延迟导入，缺 openpyxl 时在调用点具名失败。"""
    from structure_reader.export.operator_workbook import export_workbook

    return export_workbook(build_structures_workbook_spec(container, network))


def export_single_structure_workbook(structure: dict, network: dict | None = None) -> bytes:
    """单层工作簿字节；与四组入口共用 ``export_workbook``。"""
    from structure_reader.export.operator_workbook import export_workbook

    return export_workbook(build_structure_workbook_spec(structure, network))
