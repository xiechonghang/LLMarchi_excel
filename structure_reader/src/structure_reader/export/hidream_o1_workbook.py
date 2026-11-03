"""HiDream-O1-Image 模型适配：把同一份参数化 Block 结构声明成单 sheet 工作簿 spec。

薄适配，不创建 Workbook、不写 sheet、不保存文件：唯一导出实现仍是
``export/operator_workbook.export_workbook``；模型无关的公共声明在
``export/hf_workbook_common.py``。本模块只声明 HiDream-O1-Image 自己的映射与文案：

- **单 sheet**（``Generation``）：HiDream 是单次网络 forward，没有 Prefill/Decode
  两个阶段；这里只声明一个 generation 工作表，不伪造第二页。
- 两组真实重复层组：``uit-decoder``（36 层）与 ``vit-tower``（27 blocks，条件执行）。
- 可填写输入是图像尺寸/prompt 长度/参考图 patch 数/条件图 token 数/B/精度/位宽；
  不是文本模型的 seqlen/kvcachelen（本网络没有 KV cache）。
- 派生式：S_pix/S_gen/S_tot/N_ar/N_gt/S_vis_m/H_q/H_kv/D_hv/H_v3/H_v4/V_pin/
  P_in/P_ca；由本表输入用活公式重算，未给定保持未知。
- 成本列=用户 CUBE/VEC 规则耗时（表内可改吞吐）；未知/非法输入显示未知，不按 0 计。
"""
from __future__ import annotations

from structure_reader.export import hidream_o1_perf_rules as perf_rules
from structure_reader.export.hf_workbook_common import (
    derived_note as _derived_note,
    group_from_block as _common_group_from_block,
    throughput_inputs,
)

__all__ = [
    "INPUT_IDS",
    "DERIVATION_TO_PARAM",
    "SHEET_ID",
    "SHEET_NAME",
    "build_structures_workbook_spec",
    "export_structures_workbook",
    "export_single_structure_workbook",
]

SHEET_ID = "generation"
SHEET_NAME = "Generation"

# 派生 id → Block 参数行 id（复用同一行，不新增重复符号）。
DERIVATION_TO_PARAM = {
    "S_pix": "Spix",
    "S_gen": "Sgen",
    "S_tot": "Stot",
    "N_ar": "Nar",
    "N_gt": "Ngt",
    "N_vis": "Nvis",
    "S_vis_m": "Svism",
    "H_q": "Hq",
    "H_kv": "Hkv",
    "D_hv": "Dhv",
    "H_v3": "Hv3",
    "H_v4": "Hv4",
    "V_pin": "Vpin",
    "P_in": "Pin",
    "P_ca": "Pca",
    "S_attn_elems": "Sattn",
}

# 所有派生式都已对应到 Block 参数行；不额外生成“旁路”参数行。
EXTRA_PARAM_IDS: dict[str, str] = {}

INPUT_IDS = {
    "Himg": "image_height",
    "Wimg": "image_width",
    "Ptxt": "prompt_tokens",
    "Kref": "ref_images",
    "Sref": "ref_patch_tokens",
    "Svis": "cond_vision_tokens",
    "S_i": "cond_image_tokens",
    "B": "batch_size",
}

# 整网一次性任务组里不作为“重复成本任务”的循环父节点：36 层与 27 blocks 的
# 明细已在各自层组的 Block 模板里按真实次数表达。
LOOP_PARENT_OPS = ("net_layer_loop", "net_vision_loop")

_DERIVED_SOURCE = [{
    "path": "structure_reader/src/structure_reader/extract/hidream_o1_hf.py",
    "locator": "build_block_structure._derived_lengths",
    "kind": "derived",
    "note": "场景派生式；同一份结构按本表输入在 Excel 中重算",
}]

_PERF_NOTE = (
    "性能按用户 2026-10-29 确认的 CUBE/VEC 粗估：CUBE=B·M·K·N/吞吐、VEC=M·N/吞吐"
    "（默认 4096 MAC/ms、64 元素/ms，本页可改）；连续按比例、不向 64 取整；"
    "未知/非法输入不按 0 计。参考图/视觉塔分支先判 Kref：Kref=0 时该分支整段计 0。"
)

SHEET_NOTES = (
    _PERF_NOTE,
    "本表只分析 HiDream-O1-Image 的**一次**网络 forward（文本 + 像素 patch + 时间步 → x_pred）："
    "50/28 步采样循环、scheduler 更新、CFG 两次调用、噪声注入与像素反拼都在外部，"
    "不乘步数、不计算完整出图耗时。",
    "图像尺寸会吸附到官方预定义分辨率（utils:190-200）并按 32 对齐；本表按填写的 "
    "image_height/image_width 直接算 patch 数，填写值不是官方分辨率时请自行确认。",
    "参考图任务（Kref>0）才执行视觉塔（27 个 ViT block + DeepStack merger），并把参考图的 "
    "32×32 patch 拼进同一个序列；t2i（Kref=0）时 ref_patch_tokens/cond_vision_tokens 为 0。",
    "本网络没有 KV cache，也没有 seqlen/kvcachelen 语义：序列长度是 prompt_tokens + "
    "目标 patch + 参考图 patch。",
    "性能：逐算子 CUBE/VEC 分类与用户规则耗时；层组小计与整网串行汇总；"
    "未知/非法输入显示未知（原因），不按 0 计。",
)

_NETWORK_NOTES = (
    "整网路径（来自 network_overview）：文本 embedding → 时间步正弦 + 2 层 MLP → tms 位置替换 "
    "→ 像素 patch 两段低秩投影 → 文本/patch 拼接 → MRoPE + AR/生成行划分 → 36 层 decoder → "
    "Final RMSNorm → final_layer2 → 按生成 token 取 x_pred。",
    "终点是 x_pred，不是 logits：生成路径不调用 lm_head；也不存在 VAE 或独立文本编码器。",
    "视觉塔（27 blocks）是条件分支：仅参考图/编辑/IP 任务（Kref>0）执行；t2i 时整段跳过。",
    "两组真实归属：uit-decoder 36 层（0..35，总是执行）、vit-tower 27 blocks（0..26，条件执行）；"
    "两个索引空间各自从 0 开始，不能把 ViT block 当成 decoder 的第 36..62 层。",
    "时间步只通过 tms token 的嵌入替换进入网络；没有 adaLN/时间步调制 MLP。",
    "单次 forward 没有 KV cache 读写；跨采样步的 z/cond_image_embeds 属于外部循环。",
    "本区列出归属、循环次数与串行汇总；成本已按用户 CUBE/VEC 规则接入（层组小计×"
    "真实层数 + 入口/出口外围段），父组摘要与子项不重复计入整网成本。",
)


def _performance_spec() -> dict:
    return {
        "status": perf_rules.PERFORMANCE_STATUS,
        "note": perf_rules.PERFORMANCE_NOTE,
        "engine": perf_rules.ENGINE,
    }


def _sheet_inputs() -> list[dict]:
    """单页可填写输入：图像规模/prompt/参考图 + 通用性能配置。"""
    return [
        {
            "id": "image_height", "symbol": "Himg", "label": "目标图像高度（像素）",
            "value": 1024, "kind": "integer",
            "source": "示例 1024（可改；pipeline 会把尺寸吸附到预定义分辨率并 32 对齐）",
            "note": _derived_note("必须按 32 对齐；patch 数 = (Himg/32)×(Wimg/32)。"),
            "editable": True,
        },
        {
            "id": "image_width", "symbol": "Wimg", "label": "目标图像宽度（像素）",
            "value": 1024, "kind": "integer",
            "source": "示例 1024（可改；32 对齐）",
            "note": _derived_note("与高度一起决定目标 patch 数；不是文本模型的序列长度。"),
            "editable": True,
        },
        {
            "id": "prompt_tokens", "symbol": "Ptxt",
            "label": "文本 prompt token 数（含 chat template/boi/tms）",
            "value": 128, "kind": "integer",
            "source": "示例 128（可改；由实际 tokenizer 结果决定）",
            "note": _derived_note("序列 = prompt_tokens + 目标 patch + 参考图 patch。"),
            "editable": True,
        },
        {
            "id": "ref_images", "symbol": "Kref", "label": "参考图数量 Kref",
            "value": 0, "kind": "integer",
            "source": "示例 Kref=0（文生图；编辑/IP 改为实际数量）",
            "note": _derived_note("Kref>0 时视觉塔与参考图 patch 分支执行；t2i 跳过视觉塔。"),
            "editable": True,
        },
        {
            "id": "ref_patch_tokens", "symbol": "Sref",
            "label": "参考图像素 patch token 总数（数据相关）",
            "value": 0, "kind": "integer",
            "source": "示例 0（Kref=0）；Kref>0 时填实际值",
            "note": _derived_note("参考图先 32 对齐再 patch 化；数量取决于参考图数量/长宽比。"),
            "editable": True,
        },
        {
            "id": "cond_vision_tokens", "symbol": "Svis",
            "label": "VLM 条件图 patch token 总数（merge 前，数据相关）",
            "value": 0, "kind": "integer",
            "source": "示例 0（Kref=0）；Kref>0 时填实际值",
            "note": _derived_note("条件图按 16×16 patch 编码，merge 后 /4 写回文本序列；"
                                  "Kref>0 时必须 >0 且能被 merge 单元（4）整除。"),
            "editable": True,
        },
        {
            "id": "cond_image_tokens", "symbol": "S_i",
            "label": "单张条件图 patch token 数（逐段长度；等长时填同一值）",
            "value": 0, "kind": "integer",
            "source": "示例 0（Kref=0）；Kref>0 等长时填每图 token 数",
            "note": _derived_note("官方按 cu_seqlens 逐图分段做注意力；本输入要求 Kref×S_i = Svis。"
                                  "两图不等长时无法用单值表示，分段/总注意力规模保持未知。"),
            "editable": True,
        },
        {
            "id": "batch_size", "symbol": "B", "label": "batch size",
            "value": 1, "kind": "integer", "source": "示例 batch=1（可改）",
            "note": _derived_note("示例性能配置，可改；不声称代表用户实际负载。"),
            "editable": True,
        },
        {
            "id": "compute_precision", "symbol": "精度", "label": "计算精度",
            "value": "FP32", "kind": "text",
            "source": "本工程 FP32 阅读特化（可改；官方推理默认 bfloat16）",
            "note": _derived_note("用户 CUBE/VEC 粗估规则不含精度折扣，改本项不改变耗时。"),
            "editable": True,
        },
        {
            "id": "weight_storage_bits", "symbol": "Wbits",
            "label": "权重存储位宽（bit）", "value": 32, "kind": "integer",
            "source": "示例 32 bit（可改）",
            "note": _derived_note("仅为用户公式准备的输入，不对应已核实硬件能力。"),
            "editable": True,
        },
        {
            "id": "activation_storage_bits", "symbol": "Abits",
            "label": "激活存储位宽（bit）", "value": 32, "kind": "integer",
            "source": "示例 32 bit（可改）",
            "note": _derived_note("用户 CUBE/VEC 粗估规则不含带宽/位宽项，改本项不改变耗时。"),
            "editable": True,
        },
        *throughput_inputs(perf_rules.ENGINE),
    ]



def _group_from_block(block: dict, *, group_id: str, label: str,
                      layer_indices: list[int], representative: int,
                      loop: dict, template_note: str, scope_note: str) -> dict:
    return _common_group_from_block(
        block, group_id=group_id, label=label, layer_indices=layer_indices,
        representative=representative, loop=loop, template_note=template_note,
        scope_note=scope_note, input_ids=INPUT_IDS,
        derivation_to_param=DERIVATION_TO_PARAM,
        extra_param_ids=EXTRA_PARAM_IDS, derived_source=_DERIVED_SOURCE)


def _validate_hidream_attribution(container: dict) -> None:
    """严格校验真实层归属（decoder 与 ViT 是两个独立索引空间）。"""
    variants = container.get("variants", [])
    if not variants:
        raise ValueError("HiDream-O1-Image 容器没有 variants")
    ids = [variant.get("id") for variant in variants]
    if ids != ["uit-decoder", "vit-tower"]:
        raise ValueError(f"HiDream-O1-Image 层组必须且只能是 uit-decoder/vit-tower，实得 {ids}")
    expected = {
        "uit-decoder": (list(range(36)), "uit_decoder", "hybrid_causal_full", "dense_swiglu"),
        "vit-tower": (list(range(27)), "vit_tower", "bidirectional_vision", "dense_gelu_mlp"),
    }
    for variant in variants:
        gid = variant["id"]
        layers, kind, attention_type, moe_type = expected[gid]
        if list(variant.get("layer_indices", [])) != layers:
            raise ValueError(f"层组 {gid} 的真实层索引必须是 {layers[0]}..{layers[-1]}")
        representative = variant.get("representative_layer_index")
        if representative not in layers:
            raise ValueError(f"层组 {gid} 的代表层必须属于本组真实层索引")
        block = (variant.get("structure") or {}).get("block") or {}
        if block.get("kind") != kind:
            raise ValueError(f"层组 {gid} 的 Block kind 必须是 {kind}，实得 {block.get('kind')!r}")
        if variant.get("attention_type") != attention_type or block.get("attention_type") != attention_type:
            raise ValueError(f"层组 {gid} 的 attention_type 必须一致为 {attention_type}")
        if variant.get("moe_type") != moe_type or block.get("moe_type") != moe_type:
            raise ValueError(f"层组 {gid} 的 moe_type 必须一致为 {moe_type}")


def _group_loop(variant: dict) -> dict:
    if variant["id"] == "uit-decoder":
        return {
            "count": 36,
            "label": "decoder 层循环（36 层逐层实例化）",
            "condition": ("36 层按 layer_order 顺序各执行一次；本表只保留代表层模板，"
                          "不复制 36 层行"),
            "note": ("循环归属来自真实 layer_indices 0..35；模板复用不表示共享权重。"
                     "单次 forward 没有 KV cache：层间不写回任何缓存。"),
        }
    return {
        "count": 27,
        "label": "视觉塔 block 循环（27 个 block，条件执行）",
        "condition": ("仅参考图/编辑/IP 任务（Kref>0）执行；27 个 block 顺序各执行一次，"
                      "本表只保留代表 block 模板"),
        "note": ("循环归属来自真实 block 索引 0..26（视觉塔自己的索引空间，与 decoder 层号无关）。"
                 "第 8/16/24 个 block 之后另有 DeepStack merger 读取，见整网说明。"),
    }


def _variant_group(variant: dict) -> dict:
    block = variant["structure"]["block"]
    layers = list(variant["layer_indices"])
    execution = ("总是执行" if variant["id"] == "uit-decoder"
                 else "仅参考图/编辑任务（Kref>0）")
    group = _group_from_block(
        block,
        group_id=variant["id"],
        label=variant["label"],
        layer_indices=layers,
        representative=variant["representative_layer_index"],
        loop=_group_loop(variant),
        template_note=("同组每个 block 是独立实例；模板复用不代表共享权重。"),
        scope_note=(f"分析代表层 {variant['representative_layer_index']} 的 Block 结构；"
                    f"真实{'层' if variant['id'] == 'uit-decoder' else 'block'}索引 "
                    f"[{', '.join(str(i) for i in layers)}]，共 {len(layers)} 个；"
                    f"执行条件：{execution}；attention_type={variant.get('attention_type')}；"
                    f"moe_type={variant.get('moe_type')}。"),
    )
    if variant["id"] == "vit-tower":
        group["execution_condition"] = dict(perf_rules._EXEC_CONDITION)
    return perf_rules.engine.attach_performance(
        group, perf_rules.block_rules(variant["id"]), f"层组 {variant['id']}")


def _network_section(network: dict) -> dict:
    from structure_reader.export.hf_workbook_common import network_section as _net

    return _net(
        network,
        label="整网串行汇总（来自 network_overview；不铺开 36 层与 27 blocks）",
        notes=list(_NETWORK_NOTES),
        performance_engine=perf_rules.ENGINE,
        serial_order=perf_rules.serial_order(),
        count_note=("两组是不同重复结构：uit-decoder 36 层总是执行，vit-tower 27 blocks "
                    "仅 Kref>0 执行；不把两者相加成 63 层，整网 decoder 主干仍声明 36 层。"))


def _deepstack_src() -> dict:
    """DeepStack 整网行的源码引用（复用提取层的 MODELING 常量与 _src，不另造路径）。"""
    from structure_reader.extract.hidream_o1_hf import _src

    return _src(
        "762-774/913-919/938-946",
        "ViT tap：layer_num ∈ deepstack_visual_indexes（8/16/24）时 merger 读取 hidden_states；"
        "decoder 侧在 layer_idx < len(deepstack_visual_embeds)=3 时调用 _deepstack_process，"
        "把 visual_embeds 加到 visual_pos_masks 指向的条件图占位行（源码 913-919/938-946，"
        "Python 循环的 layer_idx 从 0 起）。")


def _network_group(network: dict) -> dict:
    """整网一次性任务组：必要外围网络算子（时间步/像素投影/视觉塔入口出口/Final）。

    复用同一 ``group_from_block`` 路径（同一导出器），参数/张量/算子全部来自
    network_overview 的 ``path``；循环父节点（36 层、27 blocks）不重复成行，
    它们已由两个真实层组的 Block 模板按真实次数表达。
    """
    path = dict(network["path"])
    kept = [op for op in path.get("operators", [])
            if op["id"] not in LOOP_PARENT_OPS]
    # 整网一次性 DeepStack 行（3 个 tap 的读取 + decoder 前 3 层注入）：层组模板
    # （36 层/27 blocks）对它是条件执行、按模板次数会重复计，因此层组内只作引用，
    # 在这里按真实 3 次收一次。张量/参数都取自同源 network_overview。
    anchor = next(index for index, op in enumerate(kept)
                  if op["id"] == "net_attn_partition")
    kept.insert(anchor + 1, {
        "id": perf_rules.DEEPSTACK_OP_ID,
        "module_path": "Qwen3VLTextModel.forward._deepstack_process（layer_idx<3）",
        "name": "DeepStack 条件图注入（3 个 tap，仅 decoder 前 3 层）",
        "op": "scatter-add",
        "inputs": ["n_vtaps"],
        "outputs": ["n_hidden"],
        "weights": [],
        "parameter_refs": ["Ndh", "Svis", "Hv", "B", "Svism", "H"],
        "source_refs": [_deepstack_src()],
        "analysis_status": "analyzed",
        "note": ("整网一次性：3 个 tap 从视觉塔 block 8/16/24 读取，注入 decoder "
                 "前 3 层的条件图占位行；层组模板不重复计。Kref=0 时整段不执行。"),
    })
    path["operators"] = kept
    path["attention_type"] = "one_shot_network"
    path["moe_type"] = "mixed_tasks"
    group = _group_from_block(
        path,
        group_id="network-ops",
        label="整网一次性任务（时间步/像素投影/视觉塔入口出口/Final 输出）",
        layer_indices=[],
        representative=None,
        loop={
            "count": 1,
            "label": "整网一次（每项任务按一次执行）",
            "condition": ("时间步嵌入、像素投影、视觉塔入口/merger、Final RMSNorm/linear、"
                          "生成 token 选择按整网一次执行；视觉塔相关项仅 Kref>0。"
                          "层循环父节点（36 层/27 blocks）已由层组表达，不在此重复"),
            "note": ("本组不是层组，只是单次 forward 的外围任务清单；父循环不重复计成本，"
                     "成本已按用户 CUBE/VEC 规则单列（视觉塔相关项仅 Kref>0）。"),
        },
        template_note="各任务在整网各执行一次；×3 的 DeepStack merger 在行说明里标注，不复制 3 行。",
        scope_note=("整网一次性任务组（不是 36 层的一部分）；视觉塔入口/出口条件：Kref>0；"
                    "参数/张量与 network_overview 同源，shape 由本表输入活算。"),
    )
    group["segments"] = [dict(segment) for segment in perf_rules.network_segments()]
    return perf_rules.engine.attach_performance(
        group, perf_rules.network_rules(), "整网一次性任务组")


def _sheet(network: dict, groups: list[dict]) -> dict:
    return {
        "id": SHEET_ID,
        "name": SHEET_NAME,
        "title": "单次 forward（文本 + 像素 patch + 时间步 → x_pred）",
        "notes": list(SHEET_NOTES) + [
            "本页 3 个分组：整网一次性任务（network-ops，不是层组）+ 两个真实重复层组；"
            "循环父节点（36 层 / 27 blocks）不重复成行。",
        ],
        "inputs": _sheet_inputs(),
        "groups": groups,
        "network": _network_section(network),
    }


def _validate_network_group(network: dict, groups: list[dict]) -> None:
    network_groups = [g for g in groups if g["id"] == "network-ops"]
    if len(network_groups) != 1:
        raise ValueError("HiDream-O1-Image 单 sheet 必须恰好有一个 network-ops 一次性任务组")
    group = network_groups[0]
    op_ids = {op["id"] for op in group["operators"]}
    if op_ids & set(LOOP_PARENT_OPS):
        raise ValueError("network-ops 组不应包含层循环父节点（会与层组重复计成本）")
    path_op_ids = {op["id"] for op in network["path"]["operators"]}
    allowed = path_op_ids | {perf_rules.DEEPSTACK_OP_ID}
    if not op_ids:
        raise ValueError("network-ops 组没有算子")
    if not op_ids.issubset(allowed):
        raise ValueError("network-ops 组出现 network_overview 与声明行之外的算子")
    if perf_rules.DEEPSTACK_OP_ID not in op_ids:
        raise ValueError("network-ops 组缺少整网一次性 DeepStack 行")
    for required in ("net_t_lin1", "net_t_lin2", "net_patch_lin1", "net_patch_lin2",
                     "net_vision_patch", "net_vmerge_norm", "net_ds_norm",
                     "net_final_norm", "net_final_lin"):
        if required not in op_ids:
            raise ValueError(f"network-ops 组缺少必要外围任务 {required}")


def build_structures_workbook_spec(container: dict, network: dict | None = None) -> dict:
    """两组容器 → **单页**工作簿 spec（同一结构；不伪造 Prefill/Decode 第二页）。"""
    _validate_hidream_attribution(container)
    if network is None:
        raise ValueError("HiDream-O1-Image 单 sheet 工作簿必须提供 network_overview（整网归属）")
    groups = [_network_group(network), *[_variant_group(v) for v in container["variants"]]]
    _validate_network_group(network, groups)
    return {
        "schema": "operator_workbook.v2",
        "model": container["model"],
        "performance": _performance_spec(),
        "disclaimer": perf_rules.DISCLAIMER,
        "sheets": [_sheet(network, groups)],
    }


def export_structures_workbook(container: dict, network: dict | None = None) -> bytes:
    from structure_reader.export.operator_workbook import export_workbook

    return export_workbook(build_structures_workbook_spec(container, network))


def export_single_structure_workbook(structure: dict, network: dict | None = None) -> bytes:
    raise ValueError(
        "HiDream-O1-Image 只提供分层浏览入口（--all-variants）：单层长页未实现，"
        "不伪造第二套页面；请使用分层入口")
