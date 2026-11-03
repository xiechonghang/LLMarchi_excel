"""GLM-5.3 模型适配：把同一份参数化 Block 结构声明成通用工作簿 spec。

薄适配，不创建 Workbook、不写 sheet、不保存文件：唯一导出实现仍是
``export/operator_workbook.export_workbook``；模型无关的公共声明在
``export/hf_workbook_common.py``。这里只声明 GLM-5.3 自己的映射与文案：

- 两页 Prefill/Decode、三组真实层型（full-dense / shared-moe / full-moe）；
- 派生式映射：L_kv/L_idx/L_mask/K_sel/T_tok → Lkv/Lidx/Lmask/Ksel/Ttok；
- 层组循环次数来自真实 layer_indices（3/57/18，共 78），不机械复制 78 层；
- 成本列=用户 CUBE/VEC 规则耗时（表内可改吞吐）；未知/非法输入显示未知，不按 0 计。
"""
from __future__ import annotations

from structure_reader.export import glm53_perf_rules as perf_rules
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

# GLM-5.3 的派生 id → Block 参数行 id（复用同一行，不新增重复符号）。
DERIVATION_TO_PARAM = {
    "L_kv": "Lkv",
    "L_idx": "Lidx",
    "L_mask": "Lmask",
    "K_sel": "Ksel",
    "T_tok": "Ttok",
}

# 没有既有参数行、需要补成派生参数行的 id。
EXTRA_PARAM_IDS = {
    "L_idx": "Lidx",
    "K_sel": "Ksel",
    "T_tok": "Ttok",
}

INPUT_IDS = {
    "B": "batch_size",
    "seqlen": "seqlen",
    "kvcachelen": "kvcachelen",
    # 数据相关的命中专家数是本表可改输入（默认未知）；参数行引用同一单元格，
    # 避免“参数区未知、输入区有值”两套口径。
    "N_hit": "expert_hit",
}

_DERIVED_SOURCE = [{
    "path": "structure_reader/src/structure_reader/extract/glm53_hf.py",
    "locator": "build_block_structure._build_derivations",
    "kind": "derived",
    "note": "场景派生式；同一份结构按本表输入在 Excel 中重算",
}]



def _group_from_block(block: dict, *, group_id: str, label: str,
                      layer_indices: list[int], representative: int,
                      loop: dict, template_note: str, scope_note: str) -> dict:
    return _common_group_from_block(
        block, group_id=group_id, label=label, layer_indices=layer_indices,
        representative=representative, loop=loop, template_note=template_note,
        scope_note=scope_note, input_ids=INPUT_IDS,
        derivation_to_param=DERIVATION_TO_PARAM,
        extra_param_ids=EXTRA_PARAM_IDS, derived_source=_DERIVED_SOURCE)


def _validate_glm_attribution(container: dict) -> None:
    """GLM 适配严格校验真实层归属与 indexer 模式（不借用 DeepSeek 四组契约）。"""
    variants = container.get("variants", [])
    if not variants:
        raise ValueError("GLM-5.3 容器没有 variants")
    seen: list[int] = []
    for variant in variants:
        layers = list(variant.get("layer_indices", []))
        if not layers:
            raise ValueError(f"层组 {variant.get('id')} 缺少真实 layer_indices")
        representative = variant.get("representative_layer_index")
        if representative is None or representative not in layers:
            raise ValueError(
                f"层组 {variant.get('id')} 的代表层必须属于真实 layer_indices")
        if variant.get("indexer_type") not in ("full", "shared"):
            raise ValueError(f"层组 {variant.get('id')} 缺少真实 indexer_type")
        overlap = set(seen) & set(layers)
        if overlap:
            raise ValueError(f"层组 {variant.get('id')} 与其它组层索引重叠: {sorted(overlap)}")
        seen.extend(layers)
    if sorted(seen) != list(range(len(seen))):
        raise ValueError("GLM-5.3 层组未按真实层归属恰好覆盖连续层集合 0..N-1")


def _performance_spec() -> dict:
    return {
        "status": perf_rules.PERFORMANCE_STATUS,
        "note": perf_rules.PERFORMANCE_NOTE,
        "engine": perf_rules.ENGINE,
    }


def _attach(group: dict, group_id: str) -> dict:
    return perf_rules.engine.attach_performance(
        group, perf_rules.block_rules(group_id), f"层组 {group_id}")


def _group_for_block(block: dict) -> str:
    """单层入口按真实算子构成选规则集（不看名字猜）。"""
    ids = {op["id"] for op in block.get("operators", [])}
    has_indexer = any(op.startswith("g_ix_") for op in ids)
    has_moe = "g_moe_router" in ids
    if has_indexer:
        return "full-moe" if has_moe else "full-dense"
    return "shared-moe"


def _variant_group(variant: dict, scenario: dict) -> dict:
    block = variant["structure"]["block"]
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
            "condition": ("同组真实层按顺序各执行一次；本表只保留代表层结构模板，"
                          "不复制 78 层行"),
            "note": "循环归属来自真实 layer_indices，模板复用不表示共享权重；"
                    "shared indexer 层复用上一层 full 层的 top-k，但仍是独立层实例。",
        },
        template_note="同组每层是独立 GlmMoeDsaDecoderLayer 实例；模板复用不代表共享权重。",
        scope_note=(f"分析代表层 {variant['representative_layer_index']} 的 Block 结构；"
                    f"真实层索引 [{', '.join(str(i) for i in layers)}]，"
                    f"共 {len(layers)} 层，均属本组同构模板；"
                    f"indexer_type={variant.get('indexer_type')}。"),
    )
    return _attach(group, variant["id"])


def _single_group(structure: dict, scenario: dict) -> dict:
    block = structure["block"]
    scope = structure["scope"]
    layer = scope["layer_index"]
    indexer_type = block.get("indexer_type", scope.get("indexer_type"))
    group = _group_from_block(
        block,
        group_id=f"layer-{layer}",
        label=(f"单层 layer {layer}（indexer={indexer_type} + {scope['moe_type']}）"),
        layer_indices=[layer],
        representative=layer,
        loop={
            "count": 1,
            "label": "单层阅读表",
            "condition": "本表只分析该层一次；整网三组层型循环见分层入口/整网归属",
            "note": "单层入口不汇总整网循环，避免与分层入口重复计数。",
        },
        template_note="单层样板：同一 Block 模板复用不代表共享权重。",
        scope_note=(f"单层范围 layer_index={layer}；attention_type={scope['attention_type']}；"
                    f"indexer_type={indexer_type}；moe_type={scope['moe_type']}；"
                    "整网层组归属见分层入口。"),
    )
    return _attach(group, _group_for_block(block))


def _glm_network_notes() -> list[str]:
    return [
        "网络外围（Embedding/默认 position_ids/因果 mask/交错 RoPE/Final RMSNorm/"
        "logits slice/LM head）与 78 层真实层序、三组层型归属同源；"
        "层内结构由各层组代表层 Block 表示。",
        "同构模板复用不表示共享权重：同组每层独立实例；tie_word_embeddings=false，"
        "embed_tokens 与 lm_head 权重也独立。",
        "三组真实层归属：full-dense（0,1,2，3 层）、shared-moe（57 层）、"
        "full-moe（18 层），共 78 层；full indexer 层 21 个（含 3 个 dense 层），"
        "shared 层 57 个。",
        "shared indexer 层复用上一层 full 层的 top-k（层间传递，不是持久缓存）；"
        "full 层另有 [B,L,128] indexer 键缓存。",
        "MTP：checkpoint 的 model.layers.78.* 是训练期多 token 预测头，本地实现显式忽略"
        "且 forward 只循环 0..77 —— 不在本主干，不铺开。",
        "视觉/图像输入路径不在本票范围；本表只覆盖 token 输入到 logits 的语言主干。",
        "本区列出归属、循环次数与串行汇总；成本已按用户 CUBE/VEC 规则接入（层组小计×"
        "真实层数 + 入口/出口外围段），父组摘要与子项不重复计入整网成本。",
    ]



def _sheet(phase_id: str, name: str, title: str, scenario: dict,
           groups: list[dict], network: dict | None) -> dict:
    return _common_sheet(
        phase_id, name, title, scenario, groups, network,
        net_label="整网串行汇总（来自 network_overview；不铺开 78 层）",
        net_notes=_glm_network_notes(), input_source_suffix="GLM53_SCENARIOS",
        performance_engine=perf_rules.ENGINE,
        periphery_group=(perf_rules.network_group(network)
                         if network is not None else None),
        serial_order=(perf_rules.serial_order() if network is not None else None))


def build_structures_workbook_spec(container: dict, network: dict | None = None) -> dict:
    """三组容器 → 两页工作簿 spec（每页三组，同一结构不同场景）。"""
    _validate_glm_attribution(container)
    sheets = []
    for phase_id, name, subtitle, scenario_id in SHEET_SCENARIOS:
        scenario = scenario_by_id(scenario_id)
        groups = [_variant_group(variant, scenario) for variant in container["variants"]]
        sheets.append(_sheet(phase_id, name, f"{name}（{subtitle}）", scenario,
                             groups, network))
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
        sheets.append(_sheet(phase_id, name, f"{name}（{subtitle}，单层样板）",
                             scenario, [_single_group(structure, scenario)], network))
    return {
        "schema": "operator_workbook.v2",
        "model": structure["model"],
        "performance": _performance_spec(),
        "disclaimer": perf_rules.DISCLAIMER,
        "sheets": sheets,
    }


def export_structures_workbook(container: dict, network: dict | None = None) -> bytes:
    from structure_reader.export.operator_workbook import export_workbook

    return export_workbook(build_structures_workbook_spec(container, network))


def export_single_structure_workbook(structure: dict, network: dict | None = None) -> bytes:
    from structure_reader.export.operator_workbook import export_workbook

    return export_workbook(build_structure_workbook_spec(structure, network))
