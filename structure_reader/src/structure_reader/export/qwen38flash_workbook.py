"""Qwen3.8-Flash-Next 模型适配：把同一份参数化 Block 结构声明成通用工作簿 spec。

薄适配，不创建 Workbook、不写 sheet、不保存文件：唯一导出实现仍是
``export/operator_workbook.export_workbook``；模型无关的公共声明在
``export/hf_workbook_common.py``。这里只声明 Qwen3.8-Flash-Next 自己的映射与文案：

- 两页 Prefill/Decode、三组真实层型（gdn-moe / gdn-ple-moe / qsa-moe）；
- 派生式映射：L_kv/L_idx/N_pool/B_sel/V_q/K_sel/K_sel_last/K_out/T_tok/
  S_rec/S_conv/S_ple_conv/L_ngram_ctx → 本组参数行；
- 层组循环次数来自真实 layer_indices（35/1/12，共 48），不机械复制 48 层或 512 专家；
- 成本列=用户 CUBE/VEC 规则耗时（表内可改吞吐）；未知/非法输入显示未知，不按 0 计。
- GDN 组的派生量是固定状态 S_rec/S_conv（+ PLE 的 S_ple_conv/L_ngram_ctx）；
  QSA 组才引用 kvcachelen/seqlen 推导 Lkv/Lidx/块预算/tail 规模。不要把一组的时间/
  空间公式套到另一组。
"""
from __future__ import annotations

from structure_reader.export import qwen38flash_perf_rules as perf_rules
from structure_reader.export.hf_workbook_common import (
    SHEET_SCENARIOS,
    group_from_block as _common_group_from_block,
    sheet as _common_sheet,
)
from structure_reader.extract.hf_scenario import scenario_by_id

__all__ = [
    "SHEET_SCENARIOS",
    "DERIVATION_TO_PARAM",
    "build_structure_workbook_spec",
    "build_structures_workbook_spec",
    "export_single_structure_workbook",
    "export_structures_workbook",
]

# Qwen 的派生 id → Block 参数行 id（复用同一行，不新增重复符号）。
DERIVATION_TO_PARAM = {
    "L_kv": "Lkv",
    "L_idx": "Lidx",
    "N_pool": "Npool",
    "B_sel": "Bsel",
    "V_q": "Vq",
    "N_q": "Nq",
    "B_q": "Bq",
    "R_tail": "Rtail",
    "K_sel": "Ksel",
    "K_sel_last": "Ksel_last",
    "K_out": "Kout",
    "T_tok": "Ttok",
    "S_rec": "Srec",
    "S_conv": "Sconv",
    "S_ple_conv": "Sple",
    "L_ngram_ctx": "Lctx",
}

# 所有派生式都已对应到 Block 参数行；不额外生成“旁路”参数行。
EXTRA_PARAM_IDS: dict[str, str] = {}

INPUT_IDS = {
    "B": "batch_size",
    "seqlen": "seqlen",
    "kvcachelen": "kvcachelen",
    # 数据相关的命中专家数是本表可改输入（默认未知）；参数行引用同一单元格，
    # 避免“参数区未知、输入区有值”两套口径。
    "N_hit": "expert_hit",
}

_DERIVED_SOURCE = [{
    "path": "structure_reader/src/structure_reader/extract/qwen38flash_hf.py",
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


def _validate_qwen_attribution(container: dict) -> None:
    """Qwen 适配严格校验真实层归属与注意力层型（不借用 GLM/DeepSeek 契约）。"""
    variants = container.get("variants", [])
    if not variants:
        raise ValueError("Qwen3.8-Flash-Next 容器没有 variants")
    seen: list[int] = []
    for variant in variants:
        layers = list(variant.get("layer_indices", []))
        if not layers:
            raise ValueError(f"层组 {variant.get('id')} 缺少真实 layer_indices")
        representative = variant.get("representative_layer_index")
        if representative is None or representative not in layers:
            raise ValueError(f"层组 {variant.get('id')} 的代表层必须属于真实 layer_indices")
        block = (variant.get("structure") or {}).get("block") or {}
        attention_type = variant.get("attention_type")
        if attention_type != block.get("attention_type"):
            raise ValueError(
                f"层组 {variant.get('id')} 的 attention_type 与 Block 结构不一致: "
                f"{attention_type!r} vs {block.get('attention_type')!r}")
        if variant.get("moe_type") != block.get("moe_type"):
            raise ValueError(f"层组 {variant.get('id')} 的 moe_type 与 Block 结构不一致")
        if attention_type not in ("linear_attention", "indexed_attention"):
            raise ValueError(f"层组 {variant.get('id')} 的 attention_type 非法: {attention_type!r}")
        if variant.get("ple") != block.get("ple"):
            raise ValueError(f"层组 {variant.get('id')} 的 PLE 标记与 Block 结构不一致")
        if attention_type == "indexed_attention" and variant.get("ple"):
            raise ValueError(f"层组 {variant.get('id')} 是 QSA，不应有 PLE")
        overlap = set(seen) & set(layers)
        if overlap:
            raise ValueError(f"层组 {variant.get('id')} 与其它组层索引重叠: {sorted(overlap)}")
        seen.extend(layers)
    if sorted(seen) != list(range(len(seen))):
        raise ValueError("Qwen3.8-Flash-Next 层组未按真实层归属恰好覆盖连续层集合 0..N-1")
    if len(seen) != 48:
        raise ValueError(f"Qwen3.8-Flash-Next 主干必须是 48 层，实得 {len(seen)}")


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
    ids = {op["id"] for op in block.get("operators", [])}
    if block.get("attention_type") == "indexed_attention":
        return "qsa-moe"
    return "gdn-ple-moe" if any(op.startswith("g_ple_") for op in ids) else "gdn-moe"


def _variant_group(variant: dict, scenario: dict) -> dict:
    block = variant["structure"]["block"]
    layers = list(variant["layer_indices"])
    if variant.get("attention_type") == "linear_attention":
        kind = "GDN + PLE" if variant.get("ple") else "GDN"
    else:
        kind = "QSA"
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
                          "不复制 48 层行"),
            "note": "循环归属来自真实 layer_indices，模板复用不表示共享权重；"
                    "GDN 层有固定 conv/recurrent 状态（PLE 层另有 2 个固定状态），"
                    "QSA 层有随历史增长的 K/V + indexer key 缓存，二者不能互相套用。",
        },
        template_note="同组每层是独立 Qwen4ExpTextDecoderLayer 实例；模板复用不代表共享权重。",
        scope_note=(f"分析代表层 {variant['representative_layer_index']} 的 Block 结构；"
                    f"真实层索引 [{', '.join(str(i) for i in layers)}]，"
                    f"共 {len(layers)} 层，均属本组同构模板；"
                    f"attention={variant.get('attention_type')}（{kind}）；"
                    f"moe_type={variant.get('moe_type')}；ple={variant.get('ple')}。"),
    )
    return _attach(group, variant["id"])


def _single_group(structure: dict, scenario: dict) -> dict:
    block = structure["block"]
    scope = structure["scope"]
    layer = scope["layer_index"]
    kind = "GDN + PLE" if scope.get("ple") else (
        "GDN" if scope["attention_type"] == "linear_attention" else "QSA")
    group = _group_from_block(
        block,
        group_id=f"layer-{layer}",
        label=f"单层 layer {layer}（{kind} + TopK MoE）",
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
                    f"ple={scope.get('ple')}；moe_type={scope['moe_type']}；"
                    "整网层组归属见分层入口。"),
    )
    return _attach(group, _group_for_block(block))


def _qwen_network_notes() -> list[str]:
    return [
        "网络外围（Embedding/PLE token id/4 行 position_ids/两套 mask/4 路 Gated "
        "Residual 展开/48 层/hyper_connection_mixer/logits slice/LM head）与真实层序、"
        "三组层型归属同源；层内结构由各层组代表层 Block 表示。",
        "同构模板复用不表示共享权重：同组每层独立实例；tie_word_embeddings=false，"
        "embed_tokens 与 lm_head 权重也独立。",
        "三组真实层归属：gdn-moe（35 层）、gdn-ple-moe（1 层，layer 1）、qsa-moe"
        "（3,7,11,...,47，12 层），共 48 层；没有 dense MLP 层。",
        "GN 状态固定大小（conv 40,960、recurrent 786,432 每 batch），不随 kvcachelen "
        "增长；QSA 的 K/V [B,2,L,256] 与 indexer key [B,L,128] 随历史增长。"
        "不能把一组的时间/空间公式套到另一组。",
        "Gated Residual 是 4 路 hyper-connections：读门是逐元素 [B,S,4,H]，写门是每支路"
        "标量 [B,S,4]；子层输出按写门展开加回原 4 路流。不是普通标量残差，也不是 mHC/"
        "Sinkhorn。",
        "网络末端没有 Final RMSNorm：hyper_connection_mixer 用读门加权 mean 收拢后直接 "
        "lm_head；成本/结构表都不能补一个不存在的 norm。",
        "QSA：partial RoPE 64/256、2 KV 头 + repeat 12；indexer 每 4 个可见 token 均值"
        "压成 micro-block，relu(q·k).sum(-1)/√128 打分，每 query 最多 512 块（2048 token）"
        "+ 不完整 tail，pad 到 2051。",
        "PLE：仅 layer 1；16 个哈希头（bigram 8 + trigram 8）×160 维；上下文 [B,2] + "
        "dilation-3 depthwise 短卷积 state [B,10240,9]；EOS 切断 n-gram。",
        "MTP：配置声明 1 层 full_attention，但本地 HF 实现没有 MTP 模块，加载时忽略 "
        "mtp.*，forward 只循环 0..47 —— 不在本主干，不铺开。",
        "视觉/图像输入路径不在本票范围；本表只覆盖 token 输入到 logits 的语言主干。",
        "本区列出归属、循环次数与串行汇总；成本已按用户 CUBE/VEC 规则接入（层组小计×"
        "真实层数 + 入口/出口外围段），父组摘要与子项不重复计入整网成本。",
    ]



def _sheet(phase_id: str, name: str, title: str, scenario: dict,
           groups: list[dict], network: dict | None) -> dict:
    return _common_sheet(
        phase_id, name, title, scenario, groups, network,
        net_label="整网串行汇总（来自 network_overview；不铺开 48 层）",
        net_notes=_qwen_network_notes(), input_source_suffix="QWEN38FLASH_SCENARIOS",
        performance_engine=perf_rules.ENGINE,
        periphery_group=(perf_rules.network_group(network)
                         if network is not None else None),
        serial_order=(perf_rules.serial_order() if network is not None else None))


def build_structures_workbook_spec(container: dict, network: dict | None = None) -> dict:
    """三组容器 → 两页工作簿 spec（每页三组，同一结构不同场景）。"""
    _validate_qwen_attribution(container)
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
