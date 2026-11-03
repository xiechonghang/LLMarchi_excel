"""HF 分层浏览页的模型 profile：把「模型特有的展示规则」从通用页面装配中分开。

页面（``render/hf_structure_page.py``）只做层级/锚点/路由/装配；每个模型通过
profile 提供：

- 层型短名与层格文案、scope 头部附加信息；
- 整网/层组/Block 的模型说明文字与层组事实行；
- 层型选择表（DeepSeek 的压缩器三选一 / GLM-5.3 的 full|shared indexer）；
- 模块分组规则（``diagram.module_groups`` 的 rules）、总览布局（positions/frames）、
  模块图构造与模块讲解；
- 源码命名表（可交由 ``diagram_source_labels.use_source_table`` 临时启用）。

``DEEPSEEK_PAGE_PROFILE`` 复刻既有文案与行为（默认值），保证 DeepSeek 页面输出
不变；``GLM53_PAGE_PROFILE`` 使用 GLM-5.3 自己的事实。页面不把某个模型的分组/
讲解/缓存字段当成通用规则。
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Callable

from structure_reader.render.diagram import (
    DEFAULT_LEGEND,
    build_csa_composite_graph,
    build_module_graph,
    build_overview_graph,
    module_groups as default_module_groups,
)
from structure_reader.render.glm53_narrative import (
    module_narrative as glm_module_narrative,
    overview_narrative as glm_overview_narrative,
)
from structure_reader.render.glm53_source_labels import source_table as glm_source_table
from structure_reader.render.glm53flash_narrative import (
    module_narrative as flash_module_narrative,
    overview_narrative as flash_overview_narrative,
)
from structure_reader.render.glm53flash_source_labels import source_table as flash_source_table
from structure_reader.render.hidream_o1_narrative import (
    module_narrative as hidream_module_narrative,
    overview_narrative as hidream_overview_narrative,
)
from structure_reader.render.hidream_o1_source_labels import source_table as hidream_source_table
from structure_reader.render.qwen38flash_narrative import (
    module_narrative as qwen_module_narrative,
    overview_narrative as qwen_overview_narrative,
)
from structure_reader.render.qwen38flash_source_labels import source_table as qwen_source_table
from structure_reader.render.module_narrative import (
    module_narrative as deepseek_module_narrative,
    overview_narrative as deepseek_overview_narrative,
)

__all__ = ["PageProfile", "DEEPSEEK_PAGE_PROFILE", "GLM53_PAGE_PROFILE", "profile_for"]


@dataclass(frozen=True)
class PageProfile:
    model_key: str
    # 层型短名与层格
    layer_cell: Callable[[dict], str]
    group_type_header: str
    group_heading: str
    scope_meta: Callable[[dict], str]
    # 说明文字（HTML 片段）
    network_extra: Callable[[int], str]
    group_block_note: str
    block_diagram_note: str
    group_facts_extra: Callable[[dict], str]
    # 层型选择表（压缩器 / indexer）
    variant_selection: Callable[[dict], str]
    # 模块分组/图/讲解
    module_groups: Callable[[dict], list]
    overview_graph: Callable[[dict], dict]
    module_graph: Callable[[dict, dict, dict], dict]
    nested_module_note: Callable[[str, dict], str]
    module_narrative: Callable[[dict, str], str]
    overview_narrative: Callable[[dict], str]
    network_loop_node: Callable[[dict, str], None]
    network_groups_label: str
    legend: tuple
    source_table: Callable[[list, dict], dict | None]
    group_css: str = ""
    # 展示层文案（非 Decoder 模型的替换点）：默认值复刻既有页面字符串，
    # 不传时输出与既有候选逐字节一致；只改文字，不改结构/路由/锚点。
    network_title: str = "整网：一次 forward（token → logits）"
    network_note: Callable[[int], str] | None = None
    block_title: str = "Block（Decoder Layer）"
    block_enter_label: str = "进入 Block（Decoder Layer） →"
    group_type_label: str = "层型"
    scenario_label: str = "分析场景（seqlen / kvcachelen）"
    scenario_panel_summary: str = "场景派生（当前 seqlen/kvcachelen 绑定；点击展开窗口/缓存规模）"
    scenario_panel_title: str = "场景派生（seqlen / kvcachelen 绑定）"
    header_shape_note: str = (
        "shape 保留符号 seqlen（本次新 token 数）与 kvcachelen（调用前逻辑历史 token 数）。")
    # None 表示沿用页面内置的整段 <p> 文案（保证既有候选字节不变）；非 Decoder
    # 模型给出自己的文案，整段替换（含未知项说明）。
    scenario_panel_hint: str | None = None


# ---------------------------------------------------------------------------
# DeepSeek（默认；与既有页面字符串一致）
# ---------------------------------------------------------------------------


def _ds_layer_cell(entry: dict) -> str:
    attn_short = {
        "sliding_attention": "SWA",
        "compressed_sparse_attention": "CSA",
        "heavily_compressed_attention": "HCA",
    }
    moe_short = {"hash_moe": "Hash", "moe": "TopK"}
    return (f"{attn_short.get(str(entry.get('attention_type', '')), entry.get('attention_type', ''))}"
            f" · {moe_short.get(str(entry.get('moe_type', '')), entry.get('moe_type', ''))}")


def _ds_scope_meta(scope: dict) -> str:
    return f" · hc_mult <b>{html.escape(str(scope.get('hc_mult', '')))}</b>"


def _ds_network_extra(layer_count: int) -> str:
    return ('<p class="dg-note">mHC 是所选实现的真实机制：网络入口把单流扩展为 hc_mult 路流，'
            f"{layer_count} 层内由双 HyperConnection 混合（见 Block 层），"
            "循环后由 HyperHead 收拢、再经 Final RMSNorm。终点是 lm_head 的 logits；"
            "采样/generate 循环与 loss 不在本主干。</p>")


def _ds_group_block_note() -> str:
    return ('<p class="dg-note">Block＝一个 <code>DeepseekV4DecoderLayer</code>：双 mHC 残差 + '
            "Attention（含缓存/压缩器/Indexer）+ MoE。下面只列模块清单，进入 Block 层看真实"
            "分支/残差/缓存连接。</p>")


def _ds_block_diagram_note() -> str:
    return ('<p class="dg-note">本图为模块导航（虚线框，非计算算子）：层内横向并列的是真实并行支路'
            "（Q / 共享 KV / 压缩器→Indexer），容器边界表示真实包含关系；"
            "持久状态存储框是同一 <code>past_key_values.layers[layer_idx]</code> 载体，"
            "读前态/写后态接入对应模块；残差段用 post/comb 把单流广播回多流。</p>")


def _ds_compressor_selection(item: dict) -> str:
    attention_type = str(item.get("attention_type", ""))
    mapping = {
        "compressed_sparse_attention": ("DeepseekV4CSACompressor（含 Lightning Indexer）",
                                        "Ca/Cb 重叠窗口 + Indexer top-k + overlap 状态"),
        "heavily_compressed_attention": ("DeepseekV4HCACompressor（无 Indexer）",
                                         "非重叠 compress_rate=128 + position_bias，无 overlap"),
        "sliding_attention": ("None（无压缩器）", "只用主 KV 滑动窗口"),
    }
    chosen, detail = mapping.get(attention_type, ("未知", "未分析"))
    rows = []
    for key, label, note in (
        ("compressed_sparse_attention", "CSA", "Ca/Cb 重叠 + Indexer"),
        ("heavily_compressed_attention", "HCA", "非重叠 128，无 Indexer"),
        ("sliding_attention", "SWA", "无压缩器"),
    ):
        mark = "✓ 当前生效" if key == attention_type else ""
        rows.append(f'<tr><td class="mono">{key}</td><td>{label}</td><td>{note}</td>'
                    f"<td>{mark}</td></tr>")
    return (f'<details class="fold" id="block-compressor-{html.escape(str(item["id"]))}">'
            '<summary>压缩器选择（按实际选中层型，配置驱动）</summary>'
            f'<p class="dg-note">本组 <code>config.layer_types[layer_idx]="{html.escape(attention_type)}"</code> '
            f"→ <b>{html.escape(chosen)}</b>：{html.escape(detail)}。</p>"
            '<div class="table-scroll"><table class="params"><thead><tr>'
            "<th>映射键</th><th>压缩器</th><th>结构</th><th>本组</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div></details>")


def _ds_module_graph(block: dict, group: dict, by_id: dict) -> dict:
    if group["id"] == "attn_csa" and "attn_indexer" in by_id:
        return build_csa_composite_graph(block, group, by_id["attn_indexer"])
    return build_module_graph(block, group)


def _ds_nested_note(mid: str, by_id: dict) -> str:
    if mid == "attn_indexer" and "attn_csa" in by_id:
        return ('<p class="dg-note">模块图见「CSA 压缩器」的复合图'
                "（Lightning Indexer 作为嵌套子容器画在其中）；本节的算子卡与"
                "锚点仍属于同一份结构。</p>")
    return ""


def _ds_loop_node(node: dict, loop_count: str) -> None:
    node["sub"] = f"Loop(×{loop_count}) · 四类层组"
    node["w"] = 300.0
    node["wrap"] = 38
    node["title"] = "for layer in self.layers（modeling:1357-1366）· 层顺序与四组见下方"


DEEPSEEK_PAGE_PROFILE = PageProfile(
    model_key="deepseek_v4_flash_0731_hf",
    layer_cell=_ds_layer_cell,
    group_type_header="层型 × MLP",
    group_heading="四类真实层组（真实层索引与顺序）",
    scope_meta=_ds_scope_meta,
    network_extra=_ds_network_extra,
    group_block_note=_ds_group_block_note(),
    block_diagram_note=_ds_block_diagram_note(),
    group_facts_extra=lambda item: "",
    variant_selection=_ds_compressor_selection,
    module_groups=lambda block: default_module_groups(block),
    overview_graph=lambda block: build_overview_graph(block),
    module_graph=_ds_module_graph,
    nested_module_note=_ds_nested_note,
    module_narrative=deepseek_module_narrative,
    overview_narrative=deepseek_overview_narrative,
    network_loop_node=_ds_loop_node,
    network_groups_label="四类层组",
    legend=DEFAULT_LEGEND,
    source_table=lambda structures, network: None,
    group_css=(
        ".np-swa-hash a{border-left:4px solid #215c36}\n"
        ".np-csa-hash a{border-left:4px solid #7a3b0e}\n"
        ".np-hca-topk a{border-left:4px solid #1f4e79}\n"
        ".np-csa-topk a{border-left:4px solid #6d3f8f}\n"),
)


# ---------------------------------------------------------------------------
# GLM-5.3（glm_moe_dsa）
# ---------------------------------------------------------------------------

_GLM_MODULE_RULES = (
    ("glm_attn_norm", "RMSNorm（input_layernorm）", ("g_input_norm",)),
    ("glm_attn_q", "Attention — Q 支路（低秩 + 交错部分 RoPE）",
     ("g_q_a_proj", "g_q_a_norm", "g_q_b_proj", "g_q_view", "g_q_split",
      "g_q_rope", "g_q_cat")),
    ("glm_attn_kv", "Attention — 压缩 KV 支路（latent 512 + k_rot 64）",
     ("g_kv_a_proj", "g_kv_a_split", "g_kv_a_norm", "g_k_pass_view", "g_kv_rope")),
    ("glm_attn_cache", "持久缓存（压缩 latent 写入 + kv_b 展开）",
     ("g_cache_update", "g_kv_b_proj", "g_kv_view", "g_kv_split",
      "g_k_rot_expand", "g_key_cat")),
    ("glm_attn_indexer", "DSA Indexer（full 层：打分 / top-k）",
     ("g_ix_q_proj", "g_ix_q_view", "g_ix_q_split", "g_ix_k_proj",
      "g_ix_k_norm", "g_ix_k_view", "g_ix_k_split", "g_ix_rope",
      "g_ix_q_cat", "g_ix_k_cat", "g_ix_cache_update", "g_ix_scores",
      "g_ix_relu", "g_ix_weights", "g_ix_sum", "g_ix_mask", "g_ix_topk")),
    ("glm_attn_mask", "稀疏掩码（top-k → index_mask → +因果 mask）",
     ("g_index_mask_scatter", "g_sparse_mask_apply")),
    ("glm_attn_core", "eager 注意力核心（QK → mask → softmax → AV）",
     ("g_qk", "g_mask_add", "g_softmax", "g_av", "g_attn_transpose")),
    ("glm_attn_out", "输出投影（拼头 + o_proj）", ("g_attn_reshape", "g_o_proj")),
    ("glm_res_attn", "Attention 残差（普通相加）", ("g_res_attn",)),
    ("glm_norm_ffn", "RMSNorm（post_attention_layernorm）", ("g_post_norm",)),
    ("glm_mlp_dense", "Dense MLP（gate/up/down，宽度 12288）",
     ("g_dense_gate", "g_dense_up", "g_dense_act", "g_dense_down")),
    ("glm_moe_route", "MoE 路由（sigmoid + 修正偏置 + top-8）",
     ("g_moe_flatten", "g_moe_router", "g_moe_sigmoid", "g_moe_bias_add",
      "g_moe_group_score", "g_moe_group_topk", "g_moe_group_scatter",
      "g_moe_group_expand", "g_moe_group_reshape",
      "g_moe_group_masked_fill", "g_moe_topk", "g_moe_weight_gather",
      "g_moe_weight_norm", "g_moe_weight_scale")),
    ("glm_moe_expert", "MoE 专家模板（单份；不复制 256 份）",
     ("g_moe_expert_mask", "g_moe_expert_hit", "g_moe_expert_gather_indices",
      "g_moe_select_tokens", "g_moe_expert_gate_up",
      "g_moe_expert_chunk", "g_moe_expert_act", "g_moe_expert_down",
      "g_moe_expert_weight", "g_moe_result_init", "g_moe_index_add",
      "g_moe_routed_view")),
    ("glm_moe_shared", "共享专家（宽度 2048，并行支路）",
     ("g_moe_shared_gate", "g_moe_shared_up", "g_moe_shared_act",
      "g_moe_shared_down")),
    ("glm_moe_merge", "MoE 汇合（routed + shared）", ("g_moe_merge",)),
    ("glm_res_ffn", "FFN/MoE 残差（普通相加）", ("g_res_ffn",)),
)

# 总览布局：层=自上而下的主数据流，列=真实并列支路（Q / 压缩 KV / indexer 并行）。
_GLM_OVERVIEW_POS = {
    "g_hidden_in": (0, 0.0),
    "g_position_ids": (0, 1.0),
    "g_attention_mask": (0, 2.0),
    "g_cos": (0, 3.0),
    "g_sin": (0, 3.7),
    "g_prev_topk_indices": (0, 4.6),
    "glm_attn_norm": (1, 0.0),
    "glm_attn_q": (2, 0.0),
    "glm_attn_kv": (2, 1.0),
    "glm_attn_indexer": (2, 2.0),
    "glm_attn_cache": (3, 1.0),
    "glm_attn_mask": (3, 2.0),
    "glm_attn_core": (4, 0.6),
    "glm_attn_out": (5, 0.0),
    "glm_res_attn": (6, 0.0),
    "glm_norm_ffn": (7, 0.0),
    "glm_mlp_dense": (8, 0.0),
    "glm_moe_route": (8, 0.0),
    "glm_moe_shared": (8, 1.2),
    "glm_moe_expert": (9, 0.0),
    "glm_moe_merge": (10, 0.0),
    "glm_res_ffn": (11, 0.0),
    "g_hidden_out": (12, 0.0),
}


def _glm_overview_graph(block: dict) -> dict:
    present = {g["id"] for g in default_module_groups(block, _GLM_MODULE_RULES)}
    positions = {key: pos for key, pos in _GLM_OVERVIEW_POS.items()}
    attention_children = ["m:glm_attn_q", "m:glm_attn_kv", "m:glm_attn_cache"]
    frames = []
    if "glm_attn_indexer" in present:
        attention_children.append("frame:indexer")
        frames.append(("frame:indexer", "DSA Indexer（self.indexer，full 层）",
                       "wq_b / wk / k_norm / weights_proj + top-k", "indexer",
                       ["m:glm_attn_indexer"]))
    attention_children += ["m:glm_attn_mask", "m:glm_attn_core", "m:glm_attn_out"]
    frames.insert(0, ("frame:attention", "Attention（GlmMoeDsaAttention）",
                      "MLA + DSA 稀疏选择；缓存存压缩 latent", "attn",
                      attention_children))
    return build_overview_graph(block, rules=_GLM_MODULE_RULES, positions=positions,
                                frames=frames)


def _glm_layer_cell(entry: dict) -> str:
    idx = {"full": "Full", "shared": "Shared"}.get(str(entry.get("indexer_type", "")),
                                                  str(entry.get("indexer_type", "")))
    moe = {"dense": "Dense", "sparse": "TopK MoE"}.get(str(entry.get("moe_type", "")),
                                                       str(entry.get("moe_type", "")))
    return f"DSA({idx}) · {moe}"


def _glm_scope_meta(scope: dict) -> str:
    full = len(scope.get("indexer_full_layers", []) or [])
    shared = len(scope.get("indexer_shared_layers", []) or [])
    return (f" · full indexer <b>{full} 层</b> · shared indexer <b>{shared} 层</b>")


def _glm_network_extra(layer_count: int) -> str:
    return ('<p class="dg-note">本模型是标准单流残差：网络入口不做多流扩展，'
            f"{layer_count} 层内是两处普通残差相加（没有 mHC/HyperConnection/Sinkhorn）；"
            "循环后直接 Final RMSNorm，再切 logits 位置、过 lm_head。"
            "层间还单向传递 DSA 的 top-k：full indexer 层产出、shared 层复用。"
            "终点是 lm_head 的 logits；采样/generate 循环与 loss 不在本主干。</p>")


def _glm_group_block_note() -> str:
    return ('<p class="dg-note">Block＝一个 <code>GlmMoeDsaDecoderLayer</code>：两处普通残差 + '
            "MLA(DSA) + Dense/TopK MoE。full 层含 indexer 子模块，shared 层复用上一层 top-k；"
            "下面只列模块清单，进入 Block 层看真实分支/残差/缓存连接。</p>")


def _glm_block_diagram_note() -> str:
    return ('<p class="dg-note">本图为模块导航（虚线框，非计算算子）：层内横向并列的是真实并行支路'
            "（Q 低秩 / 压缩 KV latent / DSA indexer），容器边界表示真实包含关系；"
            "缓存载体是同一 <code>past_key_values.layers[layer_idx]</code>（DynamicIndexedLayer）："
            "主 KV 存 512+64 压缩 latent，indexer 另存 [B,L,128] 键。"
            "残差是普通相加，没有多流回写。</p>")


def _glm_group_facts_extra(item: dict) -> str:
    indexer = html.escape(str(item.get("indexer_type", "")))
    return (f'<tr><th>indexer 模式</th><td class="mono">{indexer}</td>'
            "<th>注意力层型</th><td class=\"mono\">indexed_attention（MLA + DSA）</td></tr>")


def _glm_variant_selection(item: dict) -> str:
    indexer = str(item.get("indexer_type", ""))
    mapping = {
        "full": ("实例化 GlmMoeDsaIndexer（wq_b/wk/k_norm/weights_proj）",
                 "本层运行 indexer 打分与 top-k，并写 [B,L,128] indexer 键缓存"),
        "shared": ("不实例化 indexer（self.indexer=None）",
                   "复用上一层 full indexer 的 prev_topk_indices，不写 indexer 缓存"),
    }
    chosen, detail = mapping.get(indexer, ("未知", "未分析"))
    rows = []
    for key, label, note in (
        ("full", "运行本层 indexer", "wq_b/wk/k_norm/weights_proj + top-k + 键缓存"),
        ("shared", "复用上一层 top-k", "无 indexer 模块/缓存，消费 prev_topk_indices"),
    ):
        mark = "✓ 当前生效" if key == indexer else ""
        rows.append(f'<tr><td class="mono">{key}</td><td>{label}</td><td>{note}</td>'
                    f"<td>{mark}</td></tr>")
    return (f'<details class="fold" id="block-indexer-{html.escape(str(item["id"]))}">'
            '<summary>DSA indexer 模式（按实际层，配置驱动）</summary>'
            f'<p class="dg-note">本组 <code>config.indexer_types[layer_idx]="{html.escape(indexer)}"</code> '
            f"→ <b>{html.escape(chosen)}</b>：{html.escape(detail)}。</p>"
            '<div class="table-scroll"><table class="params"><thead><tr>'
            "<th>映射键</th><th>行为</th><th>结构</th><th>本组</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div></details>")


def _glm_loop_node(node: dict, loop_count: str) -> None:
    node["sub"] = f"Loop(×{loop_count}) · 三组层型"
    node["w"] = 300.0
    node["wrap"] = 38
    node["title"] = ("for i, decoder_layer in enumerate(self.layers[:num_hidden_layers])"
                     "（modeling:712-722）· 层顺序与三组见下方")


GLM53_PAGE_PROFILE = PageProfile(
    model_key="glm_5_3_hf",
    layer_cell=_glm_layer_cell,
    group_type_header="indexer × MLP",
    group_heading="三组真实层型组合（真实层索引与顺序）",
    scope_meta=_glm_scope_meta,
    network_extra=_glm_network_extra,
    group_block_note=_glm_group_block_note(),
    block_diagram_note=_glm_block_diagram_note(),
    group_facts_extra=_glm_group_facts_extra,
    variant_selection=_glm_variant_selection,
    module_groups=lambda block: default_module_groups(block, _GLM_MODULE_RULES),
    overview_graph=_glm_overview_graph,
    module_graph=lambda block, group, by_id: build_module_graph(block, group),
    nested_module_note=lambda mid, by_id: "",
    module_narrative=glm_module_narrative,
    overview_narrative=glm_overview_narrative,
    network_loop_node=_glm_loop_node,
    network_groups_label="三组层型",
    legend=(
        ("opbox", "计算算子框（源码名 + 实际计算类型）"),
        ("navbox", "导航框（总览模块，虚线——非计算算子）"),
        ("frame", "容器边界（真实包含：Attention 内嵌 DSA Indexer（full 层）；非计算算子）"),
        ("dot", "端口圆点（tensor/权重端点；源码名+shape 在箭头上）"),
        ("flow", "数据流（激活/索引/掩码张量，箭头=producer→consumer）"),
        ("weight", "权重/参数引用（虚线；不改变数据流方向）"),
        ("cache", "缓存读写（点线；主 KV 压缩 latent 512+64 / indexer 键缓存 [B,L,128]）"),
    ),
    source_table=lambda structures, network: glm_source_table(
        [item["structure"]["block"] for item in structures] + [network["path"]]),
    group_css=(
        ".np-full-dense a{border-left:4px solid #215c36}\n"
        ".np-shared-moe a{border-left:4px solid #1f4e79}\n"
        ".np-full-moe a{border-left:4px solid #7a3b0e}\n"),
)


_PROFILES = {
    "deepseek_v4_flash_0731_hf": DEEPSEEK_PAGE_PROFILE,
    "glm_5_3_hf": GLM53_PAGE_PROFILE,
}


def profile_for(model_key: str) -> PageProfile:
    try:
        return _PROFILES[model_key]
    except KeyError as exc:
        raise ValueError(
            f"模型 {model_key!r} 没有分层浏览 profile（不借用其它模型的展示规则）"
        ) from exc


# ---------------------------------------------------------------------------
# GLM-5.3-Flash（glm5_next）：KDA + MLA/DSA 混合、4 路 mHC、NoPE、k-pool indexer
# ---------------------------------------------------------------------------

_FLASH_MODULE_RULES = (
    ("mhc_attn", "mHC 残差流（attn_hc：坍缩 + pre/post/comb）",
     ("g_mhc_attn_in_norm", "g_mhc_attn_fn", "g_mhc_attn_split",
      "g_mhc_attn_gate", "g_mhc_attn_sinkhorn", "g_mhc_attn_collapse")),
    ("attn_norm", "RMSNorm（input_layernorm）", ("g_input_norm",)),
    ("kda_proj", "KDA — Q/K/V 投影与拼接",
     ("g_kda_mask_apply", "g_kda_q_proj", "g_kda_k_proj", "g_kda_v_proj",
      "g_kda_qkv_cat", "g_kda_qkv_transpose")),
    ("kda_conv", "KDA — 因果短卷积（固定 conv state）",
     ("g_kda_conv_state_update", "g_kda_conv1d", "g_kda_qkv_view",
      "g_kda_qkv_split")),
    ("kda_gate", "KDA — forget gate / beta",
     ("g_kda_f_a_proj", "g_kda_f_b_proj", "g_kda_dt_add", "g_kda_decay",
      "g_kda_b_proj", "g_kda_beta")),
    ("kda_scan", "KDA — delta-rule 扫描（recurrent/chunk）",
     ("g_kda_q_l2", "g_kda_k_l2", "g_kda_q_scale", "g_kda_scan",
      "g_kda_state_write")),
    ("kda_out", "KDA — gated RMSNorm + o_proj",
     ("g_kda_g_a_proj", "g_kda_g_b_proj", "g_kda_gate_view", "g_kda_o_norm",
      "g_kda_o_reshape", "g_kda_o_proj")),
    ("dsa_q", "MLA — Q 低秩（NoPE）",
     ("g_dsa_q_a_proj", "g_dsa_q_a_norm", "g_dsa_q_b_proj", "g_dsa_q_view")),
    ("dsa_kv", "MLA — 压缩 KV latent（NoPE）",
     ("g_dsa_kv_a_proj", "g_dsa_kv_a_split", "g_dsa_kv_a_norm",
      "g_dsa_k_pass_view", "g_dsa_k_rot_view")),
    ("dsa_cache", "持久缓存（压缩 latent 写入 + kv_b 展开）",
     ("g_dsa_cache_kv_update", "g_dsa_kv_b_proj", "g_dsa_kv_view",
      "g_dsa_kv_split", "g_dsa_k_rot_expand", "g_dsa_key_cat")),
    ("dsa_indexer", "DSA Indexer — 投影 / packed 缓存 / 可见性",
     ("g_dsa_ix_q_proj", "g_dsa_ix_q_view", "g_dsa_ix_k_proj",
      "g_dsa_ix_k_norm", "g_dsa_ix_gate", "g_dsa_ix_valid", "g_dsa_ix_pack",
      "g_dsa_ix_cache_update", "g_dsa_ix_valid_keys", "g_dsa_ix_visible")),
    ("dsa_pool", "DSA Indexer — k-pool 压缩与池打分",
     ("g_dsa_ix_pool", "g_dsa_ix_pool_end", "g_dsa_ix_valid_candidates",
      "g_dsa_ix_scores", "g_dsa_ix_relu", "g_dsa_ix_weights", "g_dsa_ix_sum",
      "g_dsa_ix_mask")),
    ("dsa_select", "DSA Indexer — 池 top-k / 展开 / tail / pad",
     ("g_dsa_ix_topk", "g_dsa_ix_selected_valid", "g_dsa_ix_select_pools",
      "g_dsa_ix_flatten", "g_dsa_ix_invalid", "g_dsa_ix_tail",
      "g_dsa_ix_concat_tail", "g_dsa_ix_pad", "g_dsa_ix_query_mask")),
    ("dsa_mask", "稀疏可见掩码（counts scatter；True=可见）",
     ("g_dsa_mask_counts", "g_dsa_sparse_mask", "g_dsa_eager_mask")),
    ("dsa_core", "eager 注意力核心（QK → mask → softmax → AV）",
     ("g_dsa_qk", "g_dsa_mask_add", "g_dsa_softmax", "g_dsa_av",
      "g_dsa_attn_transpose")),
    ("dsa_out", "输出投影（拼头 + o_proj）",
     ("g_dsa_attn_reshape", "g_dsa_o_proj")),
    ("res_attn", "mHC 残差混合（Attention 段）",
     ("g_mhc_attn_expand", "g_mhc_attn_comb_mix", "g_mhc_attn_add")),
    ("mhc_ffn", "mHC 残差流（ffn_hc：坍缩 + pre/post/comb）",
     ("g_mhc_ffn_in_norm", "g_mhc_ffn_fn", "g_mhc_ffn_split",
      "g_mhc_ffn_gate", "g_mhc_ffn_sinkhorn", "g_mhc_ffn_collapse")),
    ("norm_ffn", "RMSNorm（post_attention_layernorm）", ("g_post_norm",)),
    ("mlp_dense", "Dense MLP（含 SwiGLU clamp）",
     ("g_dense_gate", "g_dense_up", "g_dense_clamp", "g_dense_act",
      "g_dense_down")),
    ("moe_route", "MoE 路由（sigmoid + 修正偏置 + top-8）",
     ("g_moe_flatten", "g_moe_router", "g_moe_sigmoid", "g_moe_bias_add",
      "g_moe_group_score", "g_moe_group_topk", "g_moe_group_scatter",
      "g_moe_group_expand", "g_moe_group_reshape", "g_moe_group_masked_fill",
      "g_moe_topk", "g_moe_weight_gather", "g_moe_weight_norm",
      "g_moe_weight_scale")),
    ("moe_expert", "MoE 专家模板（单份；不复制 288 份）",
     ("g_moe_expert_mask", "g_moe_expert_hit", "g_moe_expert_gather_indices",
      "g_moe_select_tokens", "g_moe_expert_gate_up", "g_moe_expert_chunk",
      "g_moe_expert_clamp", "g_moe_expert_act", "g_moe_expert_down",
      "g_moe_expert_weight", "g_moe_result_init", "g_moe_index_add",
      "g_moe_routed_view")),
    ("moe_shared", "共享专家（宽度 2048，并行支路）",
     ("g_moe_shared_gate", "g_moe_shared_up", "g_moe_shared_clamp",
      "g_moe_shared_act", "g_moe_shared_down")),
    ("moe_merge", "MoE 汇合（routed + shared）", ("g_moe_merge",)),
    ("res_ffn", "mHC 残差混合（FFN 段）",
     ("g_mhc_ffn_expand", "g_mhc_ffn_comb_mix", "g_mhc_ffn_add")),
)

_FLASH_MODULE_TITLES = {rule[0]: rule[1] for rule in _FLASH_MODULE_RULES}

_FLASH_OVERVIEW_POS = {
    "g_hidden_streams_in": (0, 0.0),
    "g_attention_mask": (0, 0.7),
    "mhc_attn": (1, 0.0),
    "attn_norm": (2, 0.0),
    "kda_proj": (3, 0.0),
    "kda_conv": (3, 1.0),
    "kda_gate": (3, 2.0),
    "kda_scan": (4, 0.0),
    "kda_out": (4, 1.0),
    "dsa_q": (3, 0.0),
    "dsa_kv": (3, 1.0),
    "dsa_cache": (3, 2.0),
    "dsa_indexer": (4, 0.0),
    "dsa_pool": (4, 1.0),
    "dsa_select": (4, 2.0),
    "dsa_mask": (5, 0.0),
    "dsa_core": (5, 1.0),
    "dsa_out": (5, 2.0),
    "res_attn": (5, 0.0),
    "mhc_ffn": (6, 0.0),
    "norm_ffn": (7, 0.0),
    "mlp_dense": (8, 0.0),
    "moe_route": (8, 0.0),
    "moe_shared": (8, 1.2),
    "moe_expert": (8, 2.6),
    "moe_merge": (9, 0.0),
    "res_ffn": (10, 0.0),
    "g_hidden_streams_out": (11, 0.0),
}


def _flash_module_groups(block: dict) -> list:
    return default_module_groups(block, _FLASH_MODULE_RULES)


def _flash_overview_graph(block: dict) -> dict:
    present = {g["id"] for g in _flash_module_groups(block)}
    positions = {key: pos for key, pos in _FLASH_OVERVIEW_POS.items()}
    is_dsa = block.get("kind") == "dsa"
    if is_dsa:
        # DSA 的 7 个模块分两行排（3+4），避免总览图横向溢出到必须拖动才能看全；
        # 后续残差/MLP 行相应下移一行。
        positions.update({
            "dsa_q": (3, 0.0), "dsa_kv": (3, 1.0), "dsa_cache": (3, 2.0),
            "dsa_indexer": (4, 0.0), "dsa_pool": (4, 1.0), "dsa_select": (4, 2.0),
            "dsa_mask": (5, 0.0), "dsa_core": (5, 1.0), "dsa_out": (5, 2.0),
            "res_attn": (6, 0.0), "mhc_ffn": (7, 0.0), "norm_ffn": (8, 0.0),
            "moe_route": (9, 0.0), "moe_shared": (9, 1.2), "moe_expert": (9, 2.6),
            "moe_merge": (10, 0.0), "res_ffn": (11, 0.0),
            "g_hidden_streams_out": (12, 0.0)})
    if is_dsa:
        attention_children = ["m:dsa_q", "m:dsa_kv", "m:dsa_cache",
                              "frame:indexer", "m:dsa_mask", "m:dsa_core",
                              "m:dsa_out"]
        attn_sub = ("MLA（Q 低秩/压缩 latent/NoPE）+ DSA（k-pool/池 top-k/tail）；"
                    "indexer 是 self.indexer 子模块")
    else:
        attention_children = ["m:kda_proj", "m:kda_conv", "m:kda_gate",
                              "m:kda_scan", "m:kda_out"]
        attn_sub = "KDA 线性注意力：q/k/v + 因果短卷积 + delta-rule 扫描"
    frames = [
        ("frame:attention", "Attention（KDA / MLA+DSA）", attn_sub, "attn",
         tuple(attention_children)),
        ("frame:mhc_attn", "attn_hc（mHC）",
         "4 路流 → 无权重 RMSNorm → pre/post/comb + 坍缩", "mhc", ("m:mhc_attn",)),
        ("frame:mhc_ffn", "ffn_hc（mHC）",
         "4 路流 → 无权重 RMSNorm → pre/post/comb + 坍缩", "mhc", ("m:mhc_ffn",)),
    ]
    if is_dsa:
        frames.append(("frame:indexer", "DSA Indexer（self.indexer）",
                       "投影/packed 缓存 → k-pool 池打分 → 池 top-k/tail",
                       "indexer", ("m:dsa_indexer", "m:dsa_pool", "m:dsa_select")))
    # 只保留当前 Block 实际存在的模块/帧（build_overview_graph 会再过滤 children）
    _ = present
    return build_overview_graph(block, rules=_FLASH_MODULE_RULES,
                                positions=positions, frames=frames)


def _flash_layer_cell(entry: dict) -> str:
    attn = str(entry.get("attention_type", ""))
    moe = {"dense": "Dense", "sparse": "TopK MoE"}.get(str(entry.get("moe_type", "")),
                                                       str(entry.get("moe_type", "")))
    if attn == "linear_attention":
        return f"KDA · {moe}"
    if attn == "indexed_attention":
        return f"DSA(Full) · {moe}"
    return f"{attn} · {moe}"


def _flash_scope_meta(scope: dict) -> str:
    kda = len(scope.get("kda_layers", []) or [])
    dsa = len(scope.get("dsa_layers", []) or [])
    return f" · KDA <b>{kda} 层</b> · MLA/DSA <b>{dsa} 层</b> · mHC <b>4 路流</b>"


def _flash_network_extra(layer_count: int) -> str:
    return ('<p class="dg-note">本模型是混合注意力 + 4 路 mHC：网络入口把 embedding '
            f'广播成 hc_mult=4 路流，{layer_count} 层内每层两处 attn_hc/ffn_hc '
            "做 pre/post/comb 混合（comb 过 Sinkhorn）；循环后 hc_head 取 4 路均值、"
            "再 Final RMSNorm。KDA 层状态固定大小，MLA/DSA 层的压缩 latent 与 indexer "
            "packed 缓存随历史增长。NoPE：没有 rotary embedding。终点是 lm_head 的 logits；"
            "采样/generate 循环、视觉路径与 MTP 不在本主干。</p>")


def _flash_group_block_note() -> str:
    return ('<p class="dg-note">Block＝一个 <code>Glm5NextTextDecoderLayer</code>：'
            "attn_hc 坍缩 → Attention（KDA 或 MLA/DSA）→ post/comb 混合回 4 路流 → "
            "ffn_hc 坍缩 → Dense/TopK MoE → post/comb 混合。mHC 不是普通残差；"
            "下面只列模块清单，进入 Block 层看真实分支/缓存连接。</p>")


def _flash_block_diagram_note() -> str:
    return ('<p class="dg-note">本图为模块导航（虚线框，非计算算子）：层内横向并列的是真实并行支路'
            "（KDA 的投影/门控/扫描，或 MLA 的 Q/压缩 KV/indexer/mask/core）；"
            "frame 表示真实包含关系（Attention ⊃ DSA Indexer；attn_hc/ffn_hc 是 mHC 子模块）。"
            "KDA 缓存是同一 <code>past_key_values.layers[layer_idx]</code> 的 conv/recurrent "
            "状态；DSA 缓存是压缩 latent + indexer packed 状态。"
            "残差由 post/comb 在两个站点混合，不是标量相加。</p>")


def _flash_group_facts_extra(item: dict) -> str:
    attn = str(item.get("attention_type", ""))
    indexer = str(item.get("indexer_type", ""))
    if attn == "linear_attention":
        impl = "Glm5NextTextLinearAttention（KDA）"
        state = ("conv state [B,3×NhL×DhL,Kconv] + recurrent state [B,NhL,DhL,DhL]；"
                 "大小固定")
        indexer_row = "none（KDA 层没有 DSA indexer）"
    else:
        impl = "Glm5NextTextAttention（MLA + DSA）"
        state = ("主 KV 压缩 latent [B,1,L,512] + indexer packed [B,L,257]；随历史增长")
        indexer_row = "full（11 个 DSA 层各自实例化 indexer）"
    return (f'<tr><th>attention 实现</th><td class="mono">{html.escape(impl)}</td>'
            f"<th>indexer</th><td class=\"mono\">{html.escape(indexer_row)}</td></tr>"
            f'<tr><th>状态/缓存</th><td colspan="3" class="mono">{html.escape(state)}</td></tr>')


def _flash_variant_selection(item: dict) -> str:
    attn = str(item.get("attention_type", ""))
    rows = []
    mapping = (
        ("linear_attention", "KDA 线性注意力",
         "q/k/v + 因果短卷积 + forget gate + delta-rule 扫描；无 indexer；固定状态"),
        ("indexed_attention", "MLA + DSA 稀疏注意力",
         "Q 低秩 + NoPE；压缩 latent + kv_b 展开；k-pool indexer + 池 top-k/tail"),
    )
    for key, label, note in mapping:
        mark = "✓ 当前生效" if key == attn else ""
        rows.append(f'<tr><td class="mono">{key}</td><td>{label}</td><td>{note}</td>'
                    f"<td>{mark}</td></tr>")
    return (f'<details class="fold" id="block-attention-{html.escape(str(item["id"]))}">'
            '<summary>注意力层型（按实际 layer_types，配置驱动）</summary>'
            f'<p class="dg-note">本组 <code>config.layer_types</code> 规范后为 '
            f'<b>{html.escape(attn)}</b>；DSA 的 indexer 模式为 '
            f'<code>{html.escape(str(item.get("indexer_type", "")))}</code>。'
            "两类层的状态/缓存规模不同，不能互相套用。</p>"
            '<div class="table-scroll"><table class="params"><thead><tr>'
            "<th>layer_type</th><th>实现</th><th>结构</th><th>本组</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div></details>")


def _flash_loop_node(node: dict, loop_count: str) -> None:
    node["sub"] = f"Loop(×{loop_count}) · 三组层型（KDA/DSA）"
    node["w"] = 330.0
    node["wrap"] = 40
    node["title"] = ("for i, decoder_layer in enumerate(self.layers[:num_hidden_layers])"
                     "（modeling:1495-1507）· 34 层 KDA + 11 层 DSA，层顺序与三组见下方")


GLM53FLASH_PAGE_PROFILE = PageProfile(
    model_key="glm_5_3_flash_hf",
    layer_cell=_flash_layer_cell,
    group_type_header="注意力 × MLP",
    group_heading="三组真实层型组合（KDA / DSA × Dense/MoE）",
    scope_meta=_flash_scope_meta,
    network_extra=_flash_network_extra,
    group_block_note=_flash_group_block_note(),
    block_diagram_note=_flash_block_diagram_note(),
    group_facts_extra=_flash_group_facts_extra,
    variant_selection=_flash_variant_selection,
    module_groups=_flash_module_groups,
    overview_graph=_flash_overview_graph,
    module_graph=lambda block, group, by_id: build_module_graph(block, group),
    nested_module_note=lambda mid, by_id: "",
    module_narrative=flash_module_narrative,
    overview_narrative=flash_overview_narrative,
    network_loop_node=_flash_loop_node,
    network_groups_label="三组层型",
    legend=(
        ("opbox", "计算算子框（源码名 + 实际计算类型）"),
        ("navbox", "导航框（总览模块，虚线——非计算算子）"),
        ("frame", "容器边界（真实包含：Attention ⊃ DSA Indexer；attn_hc/ffn_hc 是 mHC 子模块）"),
        ("dot", "端口圆点（tensor/权重端点；源码名+shape 在箭头上）"),
        ("flow", "数据流（激活/索引/掩码张量，箭头=producer→consumer）"),
        ("weight", "权重/参数引用（虚线；不改变数据流方向）"),
        ("cache", "缓存读写（点线；KDA conv/recurrent 固定状态 / DSA latent+packed 缓存）"),
    ),
    source_table=lambda structures, network: flash_source_table(
        [item["structure"]["block"] for item in structures] + [network["path"]]),
    group_css=(
        ".np-kda-dense a{border-left:4px solid #215c36}\n"
        ".np-dsa-moe a{border-left:4px solid #7a3b0e}\n"
        ".np-kda-moe a{border-left:4px solid #1f4e79}\n"),
)

_PROFILES["glm_5_3_flash_hf"] = GLM53FLASH_PAGE_PROFILE


# ---------------------------------------------------------------------------
# Qwen3.8-Flash-Next（qwen4_exp）：Gated DeltaNet + QSA、4 路 Gated Residual、
# N-gram PLE、softmax MoE
# ---------------------------------------------------------------------------

_QWEN_GDN_RULES = (
    ("ple_ngram", "N-gram PLE — 上下文 / 哈希 / 查表",
     ("g_ple_context_update", "g_ple_history", "g_ple_shift1", "g_ple_shift2",
      "g_ple_bigram", "g_ple_trigram", "g_ple_concat_ids", "g_ple_slice",
      "g_ple_gather", "g_ple_flatten")),
    ("ple_gate", "N-gram PLE — 投影 / 门控 / 膨胀短卷积 / 加回",
     ("g_ple_key_proj", "g_ple_norm_key", "g_ple_key_view", "g_ple_value_proj",
      "g_ple_norm_query", "g_ple_query_view", "g_ple_gate", "g_ple_gate_sign",
      "g_ple_gated_value", "g_ple_gated_value_flat", "g_ple_norm_conv",
      "g_ple_mask", "g_ple_conv_state_update", "g_ple_conv",
      "g_ple_value_mask_flatten", "g_ple_add", "g_ple_stream_add")),
    ("mhc_attn", "Gated Residual（attn_hc：读门 [hc_count,hidden_size] + 写门 [hc_count]）",
     ("g_hc_attn_norm", "g_hc_attn_down", "g_hc_attn_down_act", "g_hc_attn_up",
      "g_hc_attn_read_sigmoid", "g_hc_attn_read_view", "g_hc_attn_norm_view",
      "g_hc_attn_mix", "g_hc_attn_inject_linear", "g_hc_attn_inject")),
    ("gdn_proj", "GDN — Q/K/V 合并投影 + z/beta/decay 投影",
     ("g_gdn_mask", "g_gdn_qkv_proj", "g_gdn_qkv_transpose", "g_gdn_z_proj",
      "g_gdn_z_view", "g_gdn_b_proj", "g_gdn_a_proj")),
    ("gdn_conv", "GDN — 因果短卷积（固定 conv state）",
     ("g_gdn_conv_state_update", "g_gdn_conv", "g_gdn_conv_transpose", "g_gdn_split")),
    ("gdn_gate", "GDN — beta / decay / q,k 头扩展",
     ("g_gdn_beta", "g_gdn_decay", "g_gdn_q_repeat", "g_gdn_k_repeat")),
    ("gdn_scan", "GDN — gated delta-rule 扫描（chunk/recurrent）",
     ("g_gdn_q_l2", "g_gdn_k_l2", "g_gdn_q_scale", "g_gdn_scan",
      "g_gdn_state_write")),
    ("gdn_out", "GDN — Gated RMSNorm + o_proj",
     ("g_gdn_o_norm", "g_gdn_o_reshape", "g_gdn_out_proj")),
    ("res_attn", "Gated Residual 回写（Attention 段）",
     ("g_hc_attn_expand", "g_hc_attn_expand_flat", "g_hc_attn_add")),
    ("mhc_ffn", "Gated Residual（ffn_hc：读门 [hc_count,hidden_size] + 写门 [hc_count]）",
     ("g_hc_ffn_norm", "g_hc_ffn_down", "g_hc_ffn_down_act", "g_hc_ffn_up",
      "g_hc_ffn_read_sigmoid", "g_hc_ffn_read_view", "g_hc_ffn_norm_view",
      "g_hc_ffn_mix", "g_hc_ffn_inject_linear", "g_hc_ffn_inject")),
    ("moe_route", "MoE 路由（softmax 512 选 10 + 归一）",
     ("g_moe_flatten", "g_moe_router", "g_moe_softmax", "g_moe_topk",
      "g_moe_weight_norm")),
    ("moe_expert", "MoE 专家模板（单份；不复制 512 份）",
     ("g_moe_expert_mask", "g_moe_expert_hit", "g_moe_expert_gather_indices",
      "g_moe_select_tokens", "g_moe_expert_gate_up", "g_moe_expert_chunk",
      "g_moe_expert_act", "g_moe_expert_down", "g_moe_expert_weight",
      "g_moe_result_init", "g_moe_index_add")),
    ("moe_shared", "共享专家（640 宽 + sigmoid 标量门）",
     ("g_moe_shared_gate", "g_moe_shared_up", "g_moe_shared_act",
      "g_moe_shared_down", "g_moe_shared_gate_scalar", "g_moe_shared_gated")),
    ("moe_merge", "MoE 汇合（routed + 共享专家）",
     ("g_moe_merge", "g_moe_merge_view")),
    ("res_ffn", "Gated Residual 回写（FFN 段）",
     ("g_hc_ffn_expand", "g_hc_ffn_expand_flat", "g_hc_ffn_add")),
)

_QWEN_QSA_RULES = (
    ("mhc_attn", "Gated Residual（attn_hc：读门 [hc_count,hidden_size] + 写门 [hc_count]）",
     ("g_hc_attn_norm", "g_hc_attn_down", "g_hc_attn_down_act", "g_hc_attn_up",
      "g_hc_attn_read_sigmoid", "g_hc_attn_read_view", "g_hc_attn_norm_view",
      "g_hc_attn_mix", "g_hc_attn_inject_linear", "g_hc_attn_inject")),
    ("qsa_q", "QSA — Q / gate 合并投影 + Q RMSNorm",
     ("g_qsa_q_proj", "g_qsa_q_gate_view", "g_qsa_q_gate_chunk",
      "g_qsa_gate_view", "g_qsa_q_norm", "g_qsa_q_transpose")),
    ("qsa_kv", "QSA — K/V 投影（2 头，K RMSNorm）",
     ("g_qsa_k_proj", "g_qsa_k_view", "g_qsa_k_norm", "g_qsa_k_transpose",
      "g_qsa_v_proj", "g_qsa_v_view")),
    ("qsa_rope", "QSA — partial RoPE（前 64/256 维）",
     ("g_qsa_q_rope", "g_qsa_k_rope")),
    ("qsa_cache", "QSA — 主 K/V 缓存写入（随历史线性增长）",
     ("g_qsa_cache_update",)),
    ("qsa_indexer", "QSA Indexer — QK 投影 / 逐 token key 缓存 / 可见性",
     ("g_qsa_ix_qk_proj", "g_qsa_ix_split", "g_qsa_ix_q_norm",
      "g_qsa_ix_q_rope", "g_qsa_ix_cache_update",
      "g_qsa_ix_visible", "g_qsa_ix_query_guard")),
    ("qsa_pool", "QSA Indexer — micro-block 均值池化 / 块打分",
     ("g_qsa_ix_block_view", "g_qsa_ix_pool", "g_qsa_ix_pool_norm",
      "g_qsa_ix_block_rope", "g_qsa_ix_scores")),
    ("qsa_select", "QSA Indexer — 块 top-k / 展开 / tail / pad",
     ("g_qsa_ix_block_topk", "g_qsa_ix_select_blocks", "g_qsa_ix_tail",
      "g_qsa_ix_concat", "g_qsa_ix_pad", "g_qsa_ix_mask_bool")),
    ("qsa_mask", "稀疏 mask bool→eager additive，再与 causal mask 合并",
     ("g_qsa_ix_mask_float", "g_qsa_mask_combine")),
    ("qsa_core", "eager 注意力核心（GQA repeat / QK / mask / softmax / AV）",
     ("g_qsa_repeat_kv", "g_qsa_qk", "g_qsa_mask_add", "g_qsa_softmax",
      "g_qsa_av", "g_qsa_attn_transpose", "g_qsa_attn_reshape")),
    ("qsa_out", "QSA — 输出 sigmoid 门 + o_proj",
     ("g_qsa_gate_sigmoid", "g_qsa_gate_mul", "g_qsa_o_proj")),
    ("res_attn", "Gated Residual 回写（Attention 段）",
     ("g_hc_attn_expand", "g_hc_attn_expand_flat", "g_hc_attn_add")),
    ("mhc_ffn", "Gated Residual（ffn_hc：读门 [hc_count,hidden_size] + 写门 [hc_count]）",
     ("g_hc_ffn_norm", "g_hc_ffn_down", "g_hc_ffn_down_act", "g_hc_ffn_up",
      "g_hc_ffn_read_sigmoid", "g_hc_ffn_read_view", "g_hc_ffn_norm_view",
      "g_hc_ffn_mix", "g_hc_ffn_inject_linear", "g_hc_ffn_inject")),
    ("moe_route", "MoE 路由（softmax 512 选 10 + 归一）",
     ("g_moe_flatten", "g_moe_router", "g_moe_softmax", "g_moe_topk",
      "g_moe_weight_norm")),
    ("moe_expert", "MoE 专家模板（单份；不复制 512 份）",
     ("g_moe_expert_mask", "g_moe_expert_hit", "g_moe_expert_gather_indices",
      "g_moe_select_tokens", "g_moe_expert_gate_up", "g_moe_expert_chunk",
      "g_moe_expert_act", "g_moe_expert_down", "g_moe_expert_weight",
      "g_moe_result_init", "g_moe_index_add")),
    ("moe_shared", "共享专家（640 宽 + sigmoid 标量门）",
     ("g_moe_shared_gate", "g_moe_shared_up", "g_moe_shared_act",
      "g_moe_shared_down", "g_moe_shared_gate_scalar", "g_moe_shared_gated")),
    ("moe_merge", "MoE 汇合（routed + 共享专家）",
     ("g_moe_merge", "g_moe_merge_view")),
    ("res_ffn", "Gated Residual 回写（FFN 段）",
     ("g_hc_ffn_expand", "g_hc_ffn_expand_flat", "g_hc_ffn_add")),
)

_QWEN_GDN_POS = {
    "g_hidden_streams_in": (0, 0.0),
    "g_conv_mask": (0, 1.0),
    "ple_ngram": (1, 0.0),
    "ple_gate": (2, 0.0),
    "mhc_attn": (3, 0.0),
    "gdn_proj": (4, 0.0),
    "gdn_conv": (4, 1.0),
    "gdn_gate": (4, 2.0),
    "gdn_scan": (5, 0.0),
    "gdn_out": (5, 1.0),
    "res_attn": (6, 0.0),
    "mhc_ffn": (7, 0.0),
    "moe_route": (8, 0.0),
    "moe_shared": (8, 1.2),
    "moe_expert": (8, 2.6),
    "moe_merge": (9, 0.0),
    "res_ffn": (10, 0.0),
    "g_hidden_streams_out": (11, 0.0),
}

_QWEN_QSA_POS = {
    "g_hidden_streams_in": (0, 0.0),
    "g_causal_mask": (0, 0.8),
    "g_position_cos": (0, 1.6),
    "mhc_attn": (1, 0.0),
    "qsa_q": (2, 0.0),
    "qsa_kv": (2, 1.0),
    "qsa_rope": (3, 0.0),
    "qsa_cache": (3, 1.0),
    "qsa_indexer": (3, 2.0),
    "qsa_pool": (4, 0.0),
    "qsa_select": (4, 1.0),
    "qsa_mask": (5, 0.0),
    "qsa_core": (5, 1.0),
    "qsa_out": (5, 2.0),
    "res_attn": (6, 0.0),
    "mhc_ffn": (7, 0.0),
    "moe_route": (8, 0.0),
    "moe_shared": (8, 1.2),
    "moe_expert": (8, 2.6),
    "moe_merge": (9, 0.0),
    "res_ffn": (10, 0.0),
    "g_hidden_streams_out": (11, 0.0),
}


def _qwen_module_rules(block: dict):
    return _QWEN_QSA_RULES if block.get("kind") == "qsa" else _QWEN_GDN_RULES


def _qwen_module_groups(block: dict) -> list:
    return default_module_groups(block, _qwen_module_rules(block))


def _qwen_overview_graph(block: dict) -> dict:
    rules = _qwen_module_rules(block)
    present = {g["id"] for g in _qwen_module_groups(block)}
    positions = dict(_QWEN_QSA_POS if block.get("kind") == "qsa" else _QWEN_GDN_POS)
    frames = []
    if "ple_ngram" in present or "ple_gate" in present:
        frames.append(("frame:ple", "N-gram PLE（layer 1）",
                       "上下文 [B,2] + 16 头哈希 + dilation-3 短卷积",
                       "indexer", tuple(f"m:{mid}" for mid in ("ple_ngram", "ple_gate")
                                         if mid in present)))
    frames.append(("frame:attn_hc", "attn_hc（Gated Residual）",
                   "4 路流 → 分组 RMSNorm → 逐元素读门 → 加权 mean；写门 [4]",
                   "mhc", ("m:mhc_attn",)))
    if block.get("kind") == "qsa":
        attention_children = tuple(f"m:{mid}" for mid in (
            "qsa_q", "qsa_kv", "qsa_rope", "qsa_cache", "qsa_indexer",
            "qsa_pool", "qsa_select", "qsa_mask", "qsa_core", "qsa_out")
            if mid in present)
        attention_sub = ("QSA：q/gate + 2 头 KV + partial RoPE + micro-block indexer + "
                         "eager attention")
        frames.append(("frame:indexer", "QSA Indexer（self.indexer）",
                       "逐 token key → 4-token 均值块 → relu(q·k) 打分 → 块 top-k/tail",
                       "indexer", tuple(f"m:{mid}" for mid in (
                           "qsa_indexer", "qsa_pool", "qsa_select") if mid in present)))
    else:
        attention_children = tuple(f"m:{mid}" for mid in (
            "gdn_proj", "gdn_conv", "gdn_gate", "gdn_scan", "gdn_out")
            if mid in present)
        attention_sub = "Gated DeltaNet：qkv + 因果短卷积 + gated delta-rule"
    frames.append(("frame:attention", "Attention（GDN / QSA）", attention_sub, "attn",
                   attention_children))
    frames.append(("frame:ffn_hc", "ffn_hc（Gated Residual）",
                   "4 路流 → 分组 RMSNorm → 逐元素读门 → 加权 mean；写门 [4]",
                   "mhc", ("m:mhc_ffn",)))
    frames.append(("frame:moe", "TopK MoE + 共享专家",
                   "softmax 512 选 10 + 单份专家模板 + 共享专家标量门",
                   "attn", tuple(f"m:{mid}" for mid in (
                       "moe_route", "moe_shared", "moe_expert", "moe_merge")
                       if mid in present)))
    return build_overview_graph(block, rules=rules, positions=positions, frames=frames)


def _qwen_layer_cell(entry: dict) -> str:
    attn = "QSA" if entry.get("attention_type") == "indexed_attention" else "GDN"
    ple = " + PLE" if entry.get("ple") else ""
    return f"{attn}{ple} · TopK MoE"


def _qwen_scope_meta(scope: dict) -> str:
    gdn = len(scope.get("gdn_layers", []) or [])
    qsa = len(scope.get("qsa_layers", []) or [])
    ple = len(scope.get("ple_layers", []) or [])
    return (f" · GDN <b>{gdn} 层</b> · QSA <b>{qsa} 层</b> · "
            f"PLE <b>{ple} 层</b> · Gated Residual <b>4 路流</b>")


def _qwen_network_extra(layer_count: int) -> str:
    return ('<p class="dg-note">本模型是 12×(3×Gated DeltaNet + 1×QSA) 混合注意力 + '
            f'4 路 Gated Residual：网络入口把 embedding 复制成 hc_count=4 路，'
            f'{layer_count} 层内每层两处 attn_hc/ffn_hc 做逐元素读门 + 每支路标量写门；'
            "循环后 hyper_connection_mixer 用同一读门加权 mean 收拢，**没有 Final RMSNorm**，"
            "直接切 logits 位置并过 lm_head。GDN 状态固定大小，QSA 的 K/V + indexer key "
            "随历史增长。终点是 lm_head 的 logits；采样/generate 循环、视觉路径与 MTP "
            "不在本主干。</p>")


def _qwen_group_block_note() -> str:
    return ('<p class="dg-note">Block＝一个 <code>Qwen4ExpTextDecoderLayer</code>：'
            "attn_hc 读门坍缩 → Attention（GDN 或 QSA，layer 1 先加 PLE）→ 写门展开回 "
            "4 路流 → ffn_hc 读门坍缩 → TopK MoE → 写门展开。Gated Residual 不是普通"
            "标量残差，也不是多路混合残差模板；下面只列模块清单，进入 Block 层看真实"
            "分支/缓存连接。</p>")


def _qwen_block_diagram_note() -> str:
    return ('<p class="dg-note">本图为模块导航（虚线框，非计算算子）：层内横向并列的是真实'
            "并行支路（GDN 的投影/卷积/门控/扫描，或 QSA 的 q/kv/rope/cache/indexer/"
            "mask/core）；frame 表示真实包含关系（Attention ⊃ QSA Indexer；attn_hc/ffn_hc "
            "是 Gated Residual 子模块；PLE 是 layer 1 的前置支路）。"
            "GDN 缓存是 conv/recurrent 固定状态；QSA 缓存是 K/V + indexer key。"
            "残差由读门/写门在两个站点混合，不是标量相加。</p>")


def _qwen_group_facts_extra(item: dict) -> str:
    attn = str(item.get("attention_type", ""))
    ple = bool(item.get("ple"))
    if attn == "linear_attention":
        impl = "Qwen4ExpTextGatedDeltaNet" + (" + Qwen4ExpTextPLELayer" if ple else "")
        state = ("conv state [B,10240,4] + recurrent state [B,48,128,128]（固定）"
                 + ("；PLE 另有 [B,10240,9] 与 [B,2]" if ple else ""))
    else:
        impl = "Qwen4ExpTextAttention + Qwen4ExpTextQSAIndexer"
        state = "K/V [B,2,L,256] + indexer key [B,L,128]（随历史增长）"
    return (f'<tr><th>attention 实现</th><td class="mono">{html.escape(impl)}</td>'
            f"<th>PLE</th><td class=\"mono\">{'是（layer 1）' if ple else '否'}</td></tr>"
            f'<tr><th>状态/缓存</th><td colspan="3" class="mono">{html.escape(state)}</td></tr>')


def _qwen_variant_selection(item: dict) -> str:
    attn = str(item.get("attention_type", ""))
    rows = []
    mapping = (
        ("linear_attention", "Gated DeltaNet 线性注意力",
         "q/k/v 合并投影 + 因果短卷积 + beta/decay + gated delta-rule；固定状态"),
        ("indexed_attention", "Qwen Sparse Attention（QSA）",
         "q/gate + partial RoPE + 2 头 KV + micro-block indexer + eager attention"),
    )
    for key, label, note in mapping:
        mark = "✓ 当前生效" if key == attn else ""
        rows.append(f'<tr><td class="mono">{key}</td><td>{label}</td><td>{note}</td>'
                    f"<td>{mark}</td></tr>")
    return (f'<details class="fold" id="block-attention-{html.escape(str(item["id"]))}">'
            '<summary>注意力层型（按实际 layer_types，配置驱动）</summary>'
            f'<p class="dg-note">本组 <code>config.layer_types</code> 规范后为 '
            f'<b>{html.escape(attn)}</b>；PLE='
            f'<code>{html.escape(str(item.get("ple", False)))}</code>。'
            "GDN 与 QSA 的状态/缓存规模不同，不能互相套用。</p>"
            '<div class="table-scroll"><table class="params"><thead><tr>'
            "<th>layer_type</th><th>实现</th><th>结构</th><th>本组</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div></details>")


def _qwen_loop_node(node: dict, loop_count: str) -> None:
    node["sub"] = f"Loop(×{loop_count}) · 三组层型（GDN/QSA）"
    node["w"] = 330.0
    node["wrap"] = 40
    node["title"] = ("for layer_idx, decoder_layer in enumerate(self.layers[:num_hidden_layers])"
                     "（modeling:1473-1482）· 36 层 GDN + 12 层 QSA，层顺序与三组见下方")


QWEN38FLASH_PAGE_PROFILE = PageProfile(
    model_key="qwen_3_8_flash_next_hf",
    layer_cell=_qwen_layer_cell,
    group_type_header="注意力 × PLE",
    group_heading="三组真实层型组合（GDN / GDN+PLE / QSA）",
    scope_meta=_qwen_scope_meta,
    network_extra=_qwen_network_extra,
    group_block_note=_qwen_group_block_note(),
    block_diagram_note=_qwen_block_diagram_note(),
    group_facts_extra=_qwen_group_facts_extra,
    variant_selection=_qwen_variant_selection,
    module_groups=_qwen_module_groups,
    overview_graph=_qwen_overview_graph,
    module_graph=lambda block, group, by_id: build_module_graph(block, group),
    nested_module_note=lambda mid, by_id: "",
    module_narrative=qwen_module_narrative,
    overview_narrative=qwen_overview_narrative,
    network_loop_node=_qwen_loop_node,
    network_groups_label="三组层型",
    legend=(
        ("opbox", "计算算子框（源码名 + 实际计算类型）"),
        ("navbox", "导航框（总览模块，虚线——非计算算子）"),
        ("frame", "容器边界（真实包含：Attention ⊃ QSA Indexer；attn_hc/ffn_hc 是 Gated Residual；PLE 在 layer 1）"),
        ("dot", "端口圆点（tensor/权重端点；源码名+shape 在箭头上）"),
        ("flow", "数据流（激活/索引/掩码张量，箭头=producer→consumer）"),
        ("weight", "权重/参数引用（虚线；不改变数据流方向）"),
        ("cache", "缓存读写（点线；GDN conv/recurrent 固定状态 / QSA K/V+indexer key）"),
    ),
    source_table=lambda structures, network: qwen_source_table(
        [item["structure"]["block"] for item in structures] + [network["path"]]),
    group_css=(
        ".np-gdn-moe a{border-left:4px solid #215c36}\n"
        ".np-gdn-ple-moe a{border-left:4px solid #7a3b0e}\n"
        ".np-qsa-moe a{border-left:4px solid #1f4e79}\n"),
)

_PROFILES["qwen_3_8_flash_next_hf"] = QWEN38FLASH_PAGE_PROFILE


# ---------------------------------------------------------------------------
# HiDream-O1-Image（Pixel-level Unified Transformer，单次 forward）
# ---------------------------------------------------------------------------

_HIDREAM_DECODER_RULES = (
    ("d_norm_attn", "RMSNorm（input_layernorm）", ("d_in_norm",)),
    ("d_attn_q", "Attention — Q 支路（q_proj → 逐头 q_norm → MRoPE）",
     ("d_q_proj", "d_q_view", "d_q_norm", "d_q_rope")),
    ("d_attn_k", "Attention — K 支路（k_proj → 逐头 k_norm → MRoPE）",
     ("d_k_proj", "d_k_view", "d_k_norm", "d_k_rope")),
    ("d_attn_v", "Attention — V 支路（v_proj → view；无 norm/RoPE）",
     ("d_v_proj", "d_v_view")),
    ("d_attn_ar", "第 1 趟：AR 子集因果注意力",
     ("d_gather_ar", "d_qk_ar", "d_softmax_ar", "d_av_ar")),
    ("d_attn_full", "第 2 趟：全序列双向注意力",
     ("d_qk_full", "d_softmax_full", "d_av_full")),
    ("d_attn_merge", "两趟合并（AR 行覆盖）", ("d_merge_ar",)),
    ("d_attn_out", "拼头 + o_proj", ("d_head_merge", "d_o_proj")),
    ("d_res_attn", "Attention 残差（标量相加）", ("d_res_attn",)),
    ("d_norm_ffn", "RMSNorm（post_attention_layernorm）", ("d_post_norm",)),
    ("d_ffn_mlp", "SwiGLU MLP（gate/up 12288 → down）",
     ("d_gate_proj", "d_up_proj", "d_silu", "d_swiglu_mul", "d_down_proj")),
    ("d_res_ffn", "FFN 残差（标量相加）", ("d_res_ffn",)),
    ("d_deepstack", "条件 DeepStack 注入（layer 0..2）", ("d_deepstack_add",)),
)

_HIDREAM_DECODER_POS = {
    "t:d_hidden_in": (0, 0.0),
    "t:d_pos_cos": (0, 1.0),
    "t:d_pos_sin": (0, 2.0),
    "t:d_ar_index": (0, 3.0),
    "t:d_visual_mask": (0, 4.2),
    "t:d_ds_embeds": (0, 5.2),
    "m:d_norm_attn": (1, 0.0),
    "m:d_attn_q": (2, 0.0),
    "m:d_attn_k": (2, 1.0),
    "m:d_attn_v": (2, 2.0),
    "m:d_attn_ar": (3, 0.0),
    "m:d_attn_full": (3, 1.0),
    "m:d_attn_merge": (4, 0.5),
    "m:d_attn_out": (5, 0.0),
    "m:d_res_attn": (6, 0.0),
    "m:d_norm_ffn": (7, 0.0),
    "m:d_ffn_mlp": (8, 0.0),
    "m:d_res_ffn": (9, 0.0),
    "m:d_deepstack": (10, 0.0),
    "t:d_hidden_out": (11, 0.0),
}

_HIDREAM_VISION_RULES = (
    ("v_norm_attn", "LayerNorm（norm1）", ("v_norm1",)),
    ("v_attn_qkv", "qkv 合并投影 + 拆分（3456，bias）", ("v_qkv_proj", "v_qkv_split")),
    ("v_attn_rope", "2D rotary（q/k）", ("v_q_rope", "v_k_rope")),
    ("v_attn_seg", "按图分段（cu_seqlens 切分/拼接）", ("v_seg_split", "v_seg_concat")),
    ("v_attn_core", "逐段非因果注意力（[Nhv,S_i,S_i]：QK→softmax→AV）",
     ("v_qk", "v_softmax", "v_av")),
    ("v_attn_out", "拼头 + proj", ("v_head_merge", "v_proj")),
    ("v_res_attn", "Attention 残差（标量相加）", ("v_res1",)),
    ("v_norm_ffn", "LayerNorm（norm2）", ("v_norm2",)),
    ("v_ffn_mlp", "GELU-tanh MLP（4304）", ("v_fc1", "v_gelu", "v_fc2")),
    ("v_res_ffn", "FFN 残差（标量相加）", ("v_res2",)),
    ("v_deepstack", "DeepStack tap（block 8/16/24）", ("v_deepstack_tap",)),
)

_HIDREAM_VISION_POS = {
    "t:v_hidden_in": (0, 0.0),
    "t:v_pos_emb": (0, 1.2),
    "t:v_cu_seqlens": (0, 2.4),
    "m:v_norm_attn": (1, 0.0),
    "m:v_attn_qkv": (2, 0.0),
    "m:v_attn_rope": (3, 0.0),
    "m:v_attn_seg": (4, 0.0),
    "m:v_attn_core": (5, 0.0),
    "m:v_attn_out": (6, 0.0),
    "m:v_res_attn": (7, 0.0),
    "m:v_norm_ffn": (8, 0.0),
    "m:v_ffn_mlp": (9, 0.0),
    "m:v_res_ffn": (10, 0.0),
    "m:v_deepstack": (11, 0.0),
}


def _hidream_module_rules(block: dict):
    return _HIDREAM_DECODER_RULES if block.get("kind") == "uit_decoder" else _HIDREAM_VISION_RULES


def _hidream_module_groups(block: dict) -> list:
    return default_module_groups(block, _hidream_module_rules(block))


def _hidream_overview_graph(block: dict) -> dict:
    rules = _hidream_module_rules(block)
    present = {g["id"] for g in _hidream_module_groups(block)}
    if block.get("kind") == "uit_decoder":
        positions = dict(_HIDREAM_DECODER_POS)
        frames = [
            ("frame:attention", "Attention（两趟 flash）",
             "第 1 趟 AR 子集 causal + 第 2 趟全序列双向 + index_copy",
             "attn", tuple(f"m:{mid}" for mid in (
                 "d_attn_q", "d_attn_k", "d_attn_v", "d_attn_ar", "d_attn_full",
                 "d_attn_merge", "d_attn_out") if mid in present)),
            ("frame:ffn", "SwiGLU MLP",
             "gate/up 12288 → SiLU(gate)×up → down 4096", "attn",
             tuple(f"m:{mid}" for mid in ("d_ffn_mlp",) if mid in present)),
        ]
    else:
        positions = dict(_HIDREAM_VISION_POS)
        frames = [
            ("frame:attention", "Attention（非因果、逐图分段）",
             "cu_seqlens 切段 → 段内 [Nhv,S_i,S_i] → 逐段 concat",
             "attn", tuple(f"m:{mid}" for mid in (
                 "v_attn_qkv", "v_attn_rope", "v_attn_seg", "v_attn_core",
                 "v_attn_out") if mid in present)),
        ]
    return build_overview_graph(block, rules=rules, positions=positions, frames=frames)


def _hidream_layer_cell(entry: dict) -> str:
    if entry.get("attention_type") == "bidirectional_vision":
        return "ViT block"
    return "UIT · Dense"


def _hidream_scope_meta(scope: dict) -> str:
    return (f" · ViT <b>{html.escape(str(scope.get('vision_block_count', '')))} blocks</b>"
            f" · 单次 forward · 输出 <b>x_pred</b>")


def _hidream_network_note(layer_count: int) -> str:
    return (
        '<p class="dg-note">本层只显示整网主路径与层组入口。节点/边来自按源码逐句记录的'
        "算子与张量引用；箭头标签是源码变量名与符号 shape，点节点可跳到下方算子卡，"
        f"点箭头标签可跳到张量索引。层循环体（{layer_count} 层）不在此铺开，"
        "两类层组的层内结构见各自的 Block/算子层。视觉塔（27 blocks）是条件分支："
        "仅参考图/编辑/IP 任务（Kref>0）执行，不占 36 层的位置。</p>")


def _hidream_network_extra(layer_count: int) -> str:
    return (
        '<p class="dg-note">这是一次网络 forward：文本 token 与像素 patch（32×32×3）在同一个序列里，'
        "时间步通过替换 1 个 tms token 的嵌入进入（不是 adaLN），输出是每个生成 token 的 "
        "32×32×3 像素预测 x_pred，<b>不经过 lm_head</b>，也没有 VAE/独立文本编码器。"
        "注意力是同一层内的行级差异：AR 文本行只对 AR 文本因果，生成 token 行对全部 token 双向"
        f"（flash 两趟 + index_copy）。单次 forward 没有 KV cache；50/28 步采样、CFG、scheduler、"
        "噪声注入与像素反拼都在外部，不乘步数。</p>")


def _hidream_group_block_note() -> str:
    return ('<p class="dg-note">Block 有两种真实形态：统一主干 <code>Qwen3VLTextDecoderLayer</code>'
            "（RMSNorm → q/k/v → 两趟注意力 → o_proj → 残差 → RMSNorm → SwiGLU → 残差 → "
            "条件 DeepStack）与视觉塔 <code>Qwen3VLVisionBlock</code>（LayerNorm → qkv 合并投影 → "
            "2D rotary → 非因果分图注意力 → proj → 残差 → LayerNorm → GELU-tanh MLP → 残差）。"
            "下面只列模块清单，进入 Block 层看真实分支/残差/条件依赖。</p>")


def _hidream_block_diagram_note() -> str:
    return ('<p class="dg-note">本图为模块导航（虚线框，非计算算子）：横向并列的是真实并行支路'
            "（q/k/v 三条投影；两趟注意力），frame 表示真实包含关系（Attention ⊃ 两趟注意力）。"
            "主干没有 KV cache 载体、没有 MoE 专家模板；DeepStack 注入是层后条件算子，"
            "只作用于条件图占位行。</p>")


def _hidream_group_facts_extra(item: dict) -> str:
    if item.get("attention_type") == "bidirectional_vision":
        impl = "Qwen3VLVisionBlock（modeling:266-290）"
        execution = "仅参考图/编辑/IP 任务（Kref>0）"
        state = "无 KV cache；DeepStack 特征是一次 forward 内的一次性输出"
    else:
        impl = "Qwen3VLTextDecoderLayer（modeling:496-539）"
        execution = "总是执行（36 层）"
        state = "无 KV cache（use_cache=False）、无持久状态；时间步不以 adaLN 进入层内"
    return (f'<tr><th>实现</th><td class="mono">{html.escape(impl)}</td>'
            f"<th>执行条件</th><td>{html.escape(execution)}</td></tr>"
            f'<tr><th>状态/缓存</th><td colspan="3" class="mono">{html.escape(state)}</td></tr>')


def _hidream_variant_selection(item: dict) -> str:
    gid = html.escape(str(item.get("id")))
    if item.get("attention_type") == "bidirectional_vision":
        rows = (
            '<tr><td class="mono">LayerNorm</td><td>norm1、norm2</td>'
            "<td>视觉塔用 LayerNorm（不是 RMSNorm）</td></tr>"
            '<tr><td class="mono">qkv 合并</td><td>Linear(1152→3456, bias)</td>'
            "<td>一次投影拆 q/k/v；head_dim 72</td></tr>"
            '<tr><td class="mono">2D rotary</td><td>行/列各 36 维</td>'
            "<td>只作用于 q/k；v 不做</td></tr>"
            '<tr><td class="mono">非因果</td><td>is_causal=False</td>'
            "<td>按 cu_seqlens 分图，每张图内部全对全</td></tr>"
        )
        return (f'<details class="fold" id="block-attention-{gid}">'
                "<summary>视觉塔结构与执行条件（配置驱动）</summary>"
                '<p class="dg-note">本组只在 K&gt;0 时执行；t2i（Kref=0）不跑视觉塔。'
                "Block 内部没有 causal mask，也没有 KV cache。</p>"
                '<div class="table-scroll"><table class="params"><thead><tr>'
                "<th>项</th><th>事实</th><th>说明</th></tr></thead><tbody>"
                + rows + "</tbody></table></div></details>")
    rows = (
        '<tr><td class="mono">AR 文本行</td><td>只对 AR 文本行做因果注意力</td>'
        "<td>token_types=0；第 1 趟只在 AR 子集上 causal</td></tr>"
        '<tr><td class="mono">生成 token 行</td><td>对全部 token 做双向注意力</td>'
        "<td>token_types&gt;0（tms 也是）；第 2 趟全序列 non-causal</td></tr>"
        '<tr><td class="mono">flash 两趟</td><td>causal(AR) + full(all) + index_copy</td>'
        "<td>pipeline 固定 use_flash_attn=True（pipeline:341）</td></tr>"
        '<tr><td class="mono">eager 4D mask</td><td>生成行置 0、其余上三角 -inf</td>'
        "<td>use_flash_attn=False 的等价实现（modeling:1577-1589）</td></tr>"
    )
    return (f'<details class="fold" id="block-attention-{gid}">'
            "<summary>注意力语义（同一层内的行级差异，不是两种层型）</summary>"
            '<p class="dg-note">本模型所有 36 层都是同一结构；差别只在这份结构内的两趟注意力'
            "与条件 DeepStack。没有 KV cache、没有 MoE、没有时间步调制。</p>"
            '<div class="table-scroll"><table class="params"><thead><tr>'
            "<th>行/实现</th><th>可见范围</th><th>说明</th></tr></thead><tbody>"
            + rows + "</tbody></table></div></details>")


def _hidream_loop_node(node: dict, loop_count: str) -> None:
    node["sub"] = f"Loop(×{loop_count}) · 统一主干（36 层）"
    node["w"] = 330.0
    node["wrap"] = 40
    node["title"] = ("for layer_idx, decoder_layer in enumerate(text_model.layers)"
                     "（modeling:1368-1378）· 36 层同构，层顺序见下方")


HIDREAM_O1_PAGE_PROFILE = PageProfile(
    model_key="hidream_o1_image_hf",
    layer_cell=_hidream_layer_cell,
    group_type_header="注意力 × MLP",
    group_heading="两类真实重复层组（统一主干 36 层 / 视觉塔 27 blocks）",
    scope_meta=_hidream_scope_meta,
    network_extra=_hidream_network_extra,
    group_block_note=_hidream_group_block_note(),
    block_diagram_note=_hidream_block_diagram_note(),
    group_facts_extra=_hidream_group_facts_extra,
    variant_selection=_hidream_variant_selection,
    module_groups=_hidream_module_groups,
    overview_graph=_hidream_overview_graph,
    module_graph=lambda block, group, by_id: build_module_graph(block, group),
    nested_module_note=lambda mid, by_id: "",
    module_narrative=hidream_module_narrative,
    overview_narrative=hidream_overview_narrative,
    network_loop_node=_hidream_loop_node,
    network_groups_label="两类层组",
    legend=(
        ("opbox", "计算算子框（源码名 + 实际计算类型）"),
        ("navbox", "导航框（总览模块，虚线——非计算算子）"),
        ("frame", "容器边界（真实包含：Attention ⊃ 两趟注意力；MLP 是层内子模块）"),
        ("dot", "端口圆点（tensor/权重端点；源码名+shape 在箭头上）"),
        ("flow", "数据流（激活/索引/掩码张量，箭头=producer→consumer）"),
        ("weight", "权重/参数引用（虚线；不改变数据流方向）"),
        ("cond", "条件执行（DeepStack 仅 layer 0..2 且 Kref>0；视觉塔仅 Kref>0）"),
    ),
    source_table=lambda structures, network: hidream_source_table(
        [item["structure"]["block"] for item in structures] + [network["path"]]),
    group_css=(
        ".np-uit-decoder a{border-left:4px solid #215c36}\n"
        ".np-vit-tower a{border-left:4px solid #7a3b0e}\n"),
    network_title="整网：一次 forward（文本 + 像素 patch + 时间步 → x_pred）",
    network_note=_hidream_network_note,
    block_title="Block（统一 Transformer / ViT）",
    block_enter_label="进入 Block（统一 Transformer / ViT） →",
    group_type_label="注意力 × MLP",
    scenario_label="分析场景（图像尺寸 / prompt / 参考图）",
    header_shape_note=("shape 保留符号：图像高/宽、prompt token 数、参考图 patch 数、"
                       "条件图 token 数；本网络没有 KV cache。"),
    scenario_panel_summary="场景派生（当前图像尺寸/prompt/参考图绑定；点击展开 patch/序列规模）",
    scenario_panel_title="场景派生（图像尺寸 / prompt / 参考图绑定）",
    scenario_panel_hint=(
        "结构 shape 始终保留符号；下表只是把场景取值代入固定配置后的派生规模。"
        "未给定图像尺寸/参考图规模时如实标未知，不默认成 0。"),
)

_PROFILES["hidream_o1_image_hf"] = HIDREAM_O1_PAGE_PROFILE
