"""DeepSeek-V4-Flash HF Decoder 语义投影：既有事实 → 可追溯 raw 语义图（T12）。

复用：``rules/deepseek_v4_hf.HfModelValues``（有效配置/来源指纹）、
``extract/deepseek_v4_hf.build_layer_structure``（``hf_layer_structure.v1`` 的
张量/算子/派生/缓存事实、四组层归属）。本模块**不**重新解析源码、不实例化模型、
不读权重；它把既有事实投影成语义阶段图，并做两件既有事实粒度不足的事：

1. 在语义层恢复可辨认粒度：``kv_proj`` → Linear + RMSNorm，``kv_head_view`` →
   View + Transpose，``q_b_proj`` → Linear + View + Transpose，``qk_scaled`` →
   Kᵀ Transpose + MatMul + scale，``av_matmul`` → MatMul + eager Transpose，
   ``attn_inv_rope`` → 两个 layout Transpose + 共轭 RoPE。
2. 用全称轴（batch_size/seqlen/kvcachelen/hidden_size/head_dim/
   num_attention_heads/num_key_value_heads/q_lora_rank/qk_rope_head_dim…）描述
   形状，并把旧内部别名（B/S/Dh/Nkv…）只留在旧事实映射里。

每个语义节点带结构化源码来源（仓内相对路径 + 行区间）、旧算子/张量映射和模块
路径；旧 block 事实中的每个算子都必须被至少一个语义节点覆盖（导出校验函数
``legacy_coverage``）。四组真实层（swa-hash / csa-hash / hca-topk / csa-topk）
各自声明自身语义差异（compressor/indexer 存在性与 RoPE 类型），不伪造统一结构。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from structure_reader.extract.deepseek_v4_hf import build_layer_structure
from structure_reader.rules.deepseek_v4_hf import (
    HfModelValues,
    layer_group_assignments,
    load_hf_model_values,
)
from structure_reader.semantic.axes import AxisCatalog, AxisDef
from structure_reader.semantic.bundle import (
    BundleIssue,
    GraphBuilder,
    SemanticBundle,
    SourceRef,
)

__all__ = [
    "SemanticProjectionError",
    "LAYER_GROUP_LAYERS",
    "build_semantic_bundle",
    "build_all_layer_group_bundles",
    "legacy_coverage",
    "semantic_shape_facts",
    "NamedLinearStep",
    "FfnExpansion",
    "ffn_named_linear_steps",
    "ffn_tensor_facts",
]

TRANSFORMERS_REPO = "transformers"
ROOT_REPO = "root"
_CONFIG_PATH = "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json"

# 四组代表层（与 LAYER_GROUP_SPECS 一致；只取真实存在的层）。
LAYER_GROUP_LAYERS: dict[str, int] = {
    "swa-hash": 0,
    "csa-hash": 2,
    "hca-topk": 3,
    "csa-topk": 4,
}

_MODELING = "src/transformers/models/deepseek_v4/modeling_deepseek_v4.py"
_CACHE_UTILS = "src/transformers/cache_utils.py"
_MASKING_UTILS = "src/transformers/masking_utils.py"

# 注意力区必须被投影覆盖的既有算子（未归属即失败，不允许静默漏掉事实）。
_REQUIRED_COVERED_PREFIXES = (
    "attn_hc_",
    "attn_norm",
    "q_",
    "rope_",
    "kv_",
    "cache_",
    "c_",
    "i_",
    "mask_",
    "hca_",
    "repeat_kv",
    "qk_",
    "sink_",
    "av_",
    "attn_inv_rope",
    "attn_group_reshape",
    "o_a_",
    "o_b_",
    "attn_residual_",
    "ffn_",
    "moe_",
    "shared_",
)


class SemanticProjectionError(ValueError):
    """既有事实无法投影为语义图，或旧算子覆盖不完整（具名拒绝）。"""


# ---------------------------------------------------------------------------
# 来源
# ---------------------------------------------------------------------------


def _locator_lines(locator: str) -> tuple[int, int | None]:
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", locator.strip())
    if not match:
        raise SemanticProjectionError(f"无法解释来源行定位: {locator!r}")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else None
    return start, end


def _source_ref(path: str, locator: str) -> SourceRef:
    line, end_line = _locator_lines(locator)
    if path.startswith("hflib/transformers/"):
        return SourceRef(
            repo_key=TRANSFORMERS_REPO,
            path=path[len("hflib/transformers/"):],
            line=line,
            end_line=end_line,
        )
    return SourceRef(repo_key=ROOT_REPO, path=path, line=line, end_line=end_line)


def _op_sources(op: Mapping[str, Any]) -> tuple[SourceRef, ...]:
    refs: list[SourceRef] = []
    seen: set[tuple] = set()
    for source in op.get("source_refs", ()):
        for part in re.split(r"[/,]", source["locator"]):
            ref = _source_ref(source["path"], part)
            key = (ref.repo_key, ref.path, ref.line, ref.end_line)
            if key not in seen:
                seen.add(key)
                refs.append(ref)
    if not refs:
        raise SemanticProjectionError(f"算子 {op.get('id')!r} 没有来源，拒绝伪造")
    return tuple(refs)


def _line_ref(path: str, line: int, end_line: int | None = None) -> SourceRef:
    return SourceRef(repo_key=TRANSFORMERS_REPO, path=path, line=line, end_line=end_line)


# ---------------------------------------------------------------------------
# 轴目录
# ---------------------------------------------------------------------------


def _axis_catalog(
    values: HfModelValues, *, has_compressor: bool, has_indexer: bool, compress_rate: int
) -> AxisCatalog:
    catalog = AxisCatalog()
    catalog.register(
        AxisDef(
            "batch_size",
            "批大小（调用输入）",
            "config/forward",
            "config.json（batch 维）+ modeling:800 input_shape",
            aliases=("B",),
        )
    )
    catalog.register(
        AxisDef(
            "seqlen",
            "本次新输入 token 数（seqlen）",
            "forward 输入",
            "modeling:800-808 input_shape/hidden_shape",
            aliases=("S", "T"),
        )
    )
    catalog.register(
        AxisDef(
            "kvcachelen",
            "调用前逻辑历史 token 数（kvcachelen）",
            "cache 状态",
            "modeling:189-201 cache.update + cache_utils:237-274",
            aliases=("Lkv",),
        )
    )
    catalog.register(
        AxisDef(
            "hidden_size",
            "隐藏维 hidden_size",
            "config",
            f"{_CONFIG_PATH}#hidden_size",
            value=values.hidden_size,
            aliases=("H",),
        )
    )
    catalog.register(
        AxisDef(
            "head_dim",
            "每头维 head_dim",
            "config",
            f"{_CONFIG_PATH}#head_dim",
            value=values.head_dim,
            aliases=("Dh",),
        )
    )
    catalog.register(
        AxisDef(
            "num_attention_heads",
            "查询头数 num_attention_heads",
            "config",
            f"{_CONFIG_PATH}#num_attention_heads",
            value=values.num_attention_heads,
            aliases=("Nh",),
        )
    )
    catalog.register(
        AxisDef(
            "num_key_value_heads",
            "KV 头数 num_key_value_heads（当前固定 1）",
            "config",
            f"{_CONFIG_PATH}#num_key_value_heads；modeling:772 单 KV 头广播",
            value=values.num_key_value_heads,
            display_value=True,
            aliases=("Nkv",),
        )
    )
    catalog.register(
        AxisDef(
            "q_lora_rank",
            "Q 低秩维 q_lora_rank",
            "config",
            f"{_CONFIG_PATH}#q_lora_rank",
            value=values.q_lora_rank,
            aliases=("Rq",),
        )
    )
    catalog.register(
        AxisDef(
            "qk_rope_head_dim",
            "RoPE 通道数 qk_rope_head_dim",
            "config",
            f"{_CONFIG_PATH}#qk_rope_head_dim",
            value=values.qk_rope_head_dim,
            aliases=("Rk",),
        )
    )
    catalog.register(
        AxisDef(
            "o_groups",
            "输出分组数 o_groups",
            "config",
            f"{_CONFIG_PATH}#o_groups",
            value=values.o_groups,
            aliases=("G",),
        )
    )
    catalog.register(
        AxisDef(
            "o_lora_rank",
            "分组输出低秩维 o_lora_rank",
            "config",
            f"{_CONFIG_PATH}#o_lora_rank",
            value=values.o_lora_rank,
            aliases=("Ro",),
        )
    )
    catalog.register(
        AxisDef(
            "sliding_window",
            "滑窗长度 sliding_window",
            "config",
            f"{_CONFIG_PATH}#sliding_window",
            value=values.sliding_window,
            aliases=("W",),
        )
    )
    catalog.register(
        AxisDef(
            "position_batch",
            "position_ids / mask 的批维（固定 1）",
            "建模实现",
            "modeling:142-153 position_ids[...,None]；:730-739 sinks.expand",
            value=1,
            display_value=True,
            name="1",
        )
    )
    catalog.register(
        AxisDef(
            "hc_mult",
            "mHC 流数 hc_mult",
            "config",
            f"{_CONFIG_PATH}#hc_mult",
            value=values.hc_mult,
            aliases=("Nhc",),
        )
    )
    catalog.register(
        AxisDef(
            "hc_mult_j",
            "comb 源流维（hc_mult 的 j 轴，与 k 轴独立）",
            "config",
            f"{_CONFIG_PATH}#hc_mult；modeling:976-979 comb[..., j, k]",
            value=values.hc_mult,
            name="hc_mult（源流 j）",
        )
    )
    catalog.register(
        AxisDef(
            "hc_mult_k",
            "comb 目标流维（hc_mult 的 k 轴，与 j 轴独立）",
            "config",
            f"{_CONFIG_PATH}#hc_mult；modeling:976-979 comb[..., j, k]",
            value=values.hc_mult,
            name="hc_mult（目标流 k）",
        )
    )
    catalog.register(
        AxisDef(
            "qk_rope_half",
            "qk_rope_head_dim / 2（interleaved 每 pair 一个 θ）",
            "派生",
            "modeling:339-356 apply_rotary_pos_emb（repeat_interleave 2）",
            expression="qk_rope_head_dim // 2",
            name="qk_rope_head_dim / 2",
            display_override="qk_rope_head_dim / 2",
        )
    )
    catalog.register(
        AxisDef(
            "attention_heads_total",
            "num_attention_heads * head_dim",
            "派生",
            "modeling:785 q_b_proj 输出特征维",
            expression="num_attention_heads * head_dim",
            display_override="num_attention_heads * head_dim",
        )
    )
    catalog.register(
        AxisDef(
            "o_group_width",
            "num_attention_heads * head_dim // o_groups",
            "派生",
            "modeling:867-869 reshape(B,T,G,-1)",
            expression="num_attention_heads * head_dim // o_groups",
            name="num_attention_heads * head_dim / o_groups",
            display_override="num_attention_heads * head_dim / o_groups",
        )
    )
    catalog.register(
        AxisDef(
            "o_groups_total",
            "o_groups * o_lora_rank",
            "派生",
            "modeling:869 o_b_proj 输入特征维",
            expression="o_groups * o_lora_rank",
            display_override="o_groups * o_lora_rank",
        )
    )
    catalog.register(
        AxisDef(
            "kv_cache_len",
            "min(kvcachelen, sliding_window - 1)（调用前持久滑窗尾部）",
            "派生",
            "modeling:189-201 主 KV 持久态 keys 尾部",
            expression="min(kvcachelen, sliding_window - 1)",
            name="min(kvcachelen, sliding_window - 1)",
            display_override="kv_cache_len",
        )
    )
    catalog.register(
        AxisDef(
            "kv_cache_len_post",
            "min(kvcachelen + seqlen, sliding_window - 1)（调用后持久态）",
            "派生",
            "modeling:189-201 keys=full[:, :, -(W-1):, :]",
            expression="min(kvcachelen + seqlen, sliding_window - 1)",
            name="min(kvcachelen + seqlen, sliding_window - 1)",
            display_override="kv_cache_len_post",
        )
    )
    catalog.register(
        AxisDef(
            "kv_length",
            "本次 Attention 可读滑窗长度 = 历史尾部 + seqlen",
            "派生",
            "modeling:189-201 full=cat([keys, kv], dim=-2)",
            expression="min(kvcachelen, sliding_window - 1) + seqlen",
            name="min(kvcachelen, sliding_window - 1) + seqlen",
            display_override="kv_length",
        )
    )
    if has_compressor:
        catalog.register(
            AxisDef(
                "compress_rate",
                "压缩率（CSA 4 / HCA 128，来自 config.compress_ratios）",
                "config",
                f"{_CONFIG_PATH}#compress_ratios",
                value=compress_rate,
            )
        )
        catalog.register(
            AxisDef(
                "num_compressed",
                "压缩条目数（历史已闭窗 + 本次新闭窗）",
                "cache 状态",
                "modeling:620-700 compressor.forward（compressed_kv 轴）",
            )
        )
        catalog.register(
            AxisDef(
                "kv_length_plus_compressed",
                "滑窗长度 + 压缩条目数",
                "派生",
                "modeling:824-829 kv=cat([kv,compressed_kv],dim=2)",
                expression="kv_length + num_compressed",
                name="kv_length + num_compressed",
                display_override="kv_length + num_compressed",
            )
        )
        catalog.register(
            AxisDef(
                "compressor_cache_len",
                "压缩器状态长度（buffer + 已压缩条目 + overlap）",
                "cache 状态",
                "modeling:203-237 cache_utils 压缩器状态",
            )
        )
        catalog.register(
            AxisDef(
                "compressor_cache_len_post",
                "压缩器状态长度（调用后）",
                "cache 状态",
                "modeling:203-237",
                expression="compressor_cache_len + num_compressed",
                name="compressor_cache_len + num_compressed",
                display_override="compressor_cache_len + num_compressed",
            )
        )
    if has_indexer:
        catalog.register(
            AxisDef(
                "index_n_heads",
                "Indexer 头数 index_n_heads",
                "config",
                f"{_CONFIG_PATH}#index_n_heads",
                value=values.index_n_heads,
                aliases=("Ih",),
            )
        )
        catalog.register(
            AxisDef(
                "index_head_dim",
                "Indexer 头维 index_head_dim",
                "config",
                f"{_CONFIG_PATH}#index_head_dim",
                value=values.index_head_dim,
                aliases=("Id",),
            )
        )
        catalog.register(
            AxisDef(
                "index_topk",
                "Indexer top-k 上限 index_topk",
                "config",
                f"{_CONFIG_PATH}#index_topk",
                value=values.index_topk,
                aliases=("Ki",),
            )
        )
        catalog.register(
            AxisDef(
                "num_selected",
                "min(index_topk, num_compressed)",
                "派生",
                "modeling:565-583 top_k=min(Ki, compressed_len)",
                expression="min(index_topk, num_compressed)",
                name="min(index_topk, num_compressed)",
                display_override="min(index_topk, num_compressed)",
            )
        )
    # 单 KV 头广播后的头数：表达式在化简规则 R1 下可（带 divisible 前提）简化为
    # num_attention_heads；不满足前提时保留表达式。
    catalog.register(
        AxisDef(
            "kv_broadcast_heads",
            "num_key_value_heads * (num_attention_heads // num_key_value_heads)",
            "派生",
            "modeling:702-711 repeat_kv expand+reshape",
            expression="num_key_value_heads * (num_attention_heads // num_key_value_heads)",
            name="num_attention_heads",
        )
    )
    return catalog


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------


@dataclass
class _Context:
    values: HfModelValues
    block: Mapping[str, Any]
    ops: Mapping[str, Mapping[str, Any]]
    tensors: Mapping[str, Mapping[str, Any]]
    group: Mapping[str, Any]
    layer_type: str
    has_compressor: bool
    has_indexer: bool
    compress_rate: int
    rope_type: str
    builder: GraphBuilder
    guards: Mapping[str, str]
    transitions: Mapping[str, str]
    kv_len_axis: str
    known: frozenset[str] = frozenset()

    @property
    def attention_type(self) -> str:
        return self.group["attention_type"]

    def legacy(self, *refs: Any) -> tuple[str, ...]:
        """只保留真实存在的旧事实引用（不同层组的张量命名有差异）。"""
        flat: list[str] = []
        for ref in refs:
            if isinstance(ref, (list, tuple)):
                flat.extend(ref)
            else:
                flat.append(ref)
        return tuple(ref for ref in flat if ref in self.known)

    def old_tensor(self, *candidates: str) -> str | None:
        for candidate in candidates:
            if f"tensor:{candidate}" in self.known:
                return candidate
        return None

    def op_sources(self, *op_ids: str) -> tuple[SourceRef, ...]:
        refs: list[SourceRef] = []
        seen: set[tuple] = set()
        for op_id in op_ids:
            op = self.ops.get(op_id)
            if op is None:
                raise SemanticProjectionError(f"既有事实缺少算子 {op_id!r}")
            for ref in _op_sources(op):
                key = (ref.repo_key, ref.path, ref.line, ref.end_line)
                if key not in seen:
                    seen.add(key)
                    refs.append(ref)
        return tuple(refs)


def _ops_ids(ctx: _Context, prefixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            op_id
            for op_id in ctx.ops
            if any(op_id == prefix or op_id.startswith(prefix) for prefix in prefixes)
        )
    )


def _tensor(ctx: _Context, old_id: str) -> Mapping[str, Any]:
    if old_id not in ctx.tensors:
        raise SemanticProjectionError(f"既有事实缺少张量 {old_id!r}")
    return ctx.tensors[old_id]


def _build_layer_nodes(ctx: _Context) -> None:
    b = ctx.builder
    module = "model.layers.%d" % ctx.block["layer_index"]

    b.tensor(
        "hidden_streams_in",
        ("batch_size", "seqlen", "hc_mult", "hidden_size"),
        role="boundary",
        label="hidden_streams（层输入，mHC 多流）",
        note="mHC 把 hidden 扩成 hc_mult 条流；旧的 [B,seqlen,Nhc,H] 事实",
    )
    b.tensor(
        "attn_collapsed",
        ("batch_size", "seqlen", "hidden_size"),
        label="attn_hc 坍缩流",
        note="Σ_n pre_n·stream_n（坍缩不合并 RoPE/权重计算）",
    )
    b.tensor(
        "attn_mhc_post",
        ("batch_size", "seqlen", "hc_mult"),
        label="attn_hc post（坍缩后的扩展权重）",
    )
    b.tensor(
        "attn_mhc_comb",
        ("batch_size", "seqlen", "hc_mult_j", "hc_mult_k"),
        label="attn_hc comb（流混合矩阵）",
    )
    b.tensor(
        "ffn_mhc_post",
        ("batch_size", "seqlen", "hc_mult"),
        label="ffn_hc post（扩展权重）",
    )
    b.tensor(
        "ffn_mhc_comb",
        ("batch_size", "seqlen", "hc_mult_j", "hc_mult_k"),
        label="ffn_hc comb（流混合矩阵）",
    )
    b.tensor("attn_normed", ("batch_size", "seqlen", "hidden_size"), label="input_layernorm 输出")
    b.tensor("attn_block_output", ("batch_size", "seqlen", "hidden_size"), label="Attention 输出")
    b.tensor(
        "attn_out_streams",
        ("batch_size", "seqlen", "hc_mult", "hidden_size"),
        label="attn 段残差混合输出",
    )
    b.tensor("ffn_collapsed", ("batch_size", "seqlen", "hidden_size"), label="ffn_hc 坍缩流")
    b.tensor("ffn_normed", ("batch_size", "seqlen", "hidden_size"), label="post_attention_layernorm 输出")
    b.tensor("mlp_output", ("batch_size", "seqlen", "hidden_size"), label="MoE/FFN 输出")
    b.tensor(
        "hidden_streams_out",
        ("batch_size", "seqlen", "hc_mult", "hidden_size"),
        role="boundary",
        label="hidden_streams（层输出）",
    )

    b.node(
        "layer.input",
        op_type="input",
        granularity="external",
        label="hidden_streams",
        kind="external",
        module_path=module,
        sources=_op_sources(ctx.ops["attn_hc_flatten"]),
        outputs=(("streams", "hidden_streams_in"),),
        legacy_refs=ctx.legacy("tensor:hidden_streams_in", "op:attn_hc_flatten"),
        note="DecoderLayer 输入（Attention 前 mHC 流）；[batch_size, seqlen, hc_mult, hidden_size]",
    )
    b.node(
        "layer.attn_hc",
        op_type="HyperConnection",
        granularity="named_module",
        label="attn_hc · mHC 流坍缩（pre/post/comb）",
        kind="composite",
        module_path=f"{module}.attn_hc",
        sources=ctx.op_sources(
            "attn_hc_flatten",
            "attn_hc_project",
            "attn_hc_split",
            "attn_hc_pre",
            "attn_hc_post",
            "attn_hc_comb",
            "attn_hc_collapse",
        ),
        inputs=(("streams", "hidden_streams_in"),),
        outputs=(
            ("collapsed", "attn_collapsed"),
            ("post", "attn_mhc_post"),
            ("comb", "attn_mhc_comb"),
        ),
        legacy_refs=tuple(
            f"op:{op_id}"
            for op_id in (
                "attn_hc_flatten",
                "attn_hc_project",
                "attn_hc_split",
                "attn_hc_pre",
                "attn_hc_post",
                "attn_hc_comb",
                "attn_hc_collapse",
            )
        )
        + ("tensor:attn_mhc_collapsed",),
        params=("hc_mult", "hidden_size"),
        note=(
            "mHC：flatten+input_norm → F.linear(fn) → split(pre/post/comb) → "
            "门控/Sinkhorn → collapsed=Σpre·stream；内部微步骤按具名模块保留，"
            "不在主图展开"
        ),
    )
    b.node(
        "layer.input_layernorm",
        op_type="DeepseekV4RMSNorm",
        granularity="named_instance",
        label="input_layernorm · RMSNorm",
        kind="rmsnorm",
        module_path=f"{module}.input_layernorm",
        sources=_op_sources(ctx.ops["attn_norm"]),
        inputs=(("x", "attn_collapsed"),),
        outputs=(("y", "attn_normed"),),
        legacy_refs=ctx.legacy("op:attn_norm", "tensor:attn_norm_weight", "tensor:attn_normed"),
        params=("hidden_size", "rms_norm_eps"),
        note="weight [hidden_size]；fp32 归一后乘权重并转回 activation dtype（精度不变）",
    )
    b.node(
        "layer.attn_residual",
        op_type="HyperConnection",
        granularity="named_module",
        label="attn 残差：post 扩展 + combᵀ 混合",
        kind="composite",
        module_path=module,
        sources=ctx.op_sources(
            "attn_residual_combT", "attn_residual_expand", "attn_residual_mix"
        ),
        inputs=(
            ("y", "attn_block_output"),
            ("residual", "hidden_streams_in"),
            ("post", "attn_mhc_post"),
            ("comb", "attn_mhc_comb"),
        ),
        outputs=(("streams", "attn_out_streams"),),
        legacy_refs=ctx.legacy(
            "op:attn_residual_combT",
            "op:attn_residual_expand",
            "op:attn_residual_mix",
            "tensor:attn_out_streams",
        ),
        params=("hc_mult", "hidden_size"),
        note=(
            "hc post/comb 来自 attn_hc（同一 HyperConnection 前向的另外两个输出）："
            "combᵀ @ residual + post.unsqueeze(-1)·output；不是普通恒等残差"
        ),
    )
    b.node(
        "layer.ffn_hc",
        op_type="HyperConnection",
        granularity="named_module",
        label="ffn_hc · mHC 流坍缩（pre/post/comb）",
        kind="composite",
        module_path=f"{module}.ffn_hc",
        sources=ctx.op_sources(
            "ffn_hc_flatten",
            "ffn_hc_project",
            "ffn_hc_split",
            "ffn_hc_pre",
            "ffn_hc_post",
            "ffn_hc_comb",
            "ffn_hc_collapse",
        ),
        inputs=(("streams", "attn_out_streams"),),
        outputs=(
            ("collapsed", "ffn_collapsed"),
            ("post", "ffn_mhc_post"),
            ("comb", "ffn_mhc_comb"),
        ),
        legacy_refs=tuple(
            f"op:{op_id}"
            for op_id in (
                "ffn_hc_flatten",
                "ffn_hc_project",
                "ffn_hc_split",
                "ffn_hc_pre",
                "ffn_hc_post",
                "ffn_hc_comb",
                "ffn_hc_collapse",
            )
        )
        + ("tensor:ffn_mhc_collapsed",),
        params=("hc_mult", "hidden_size"),
        note="与 attn_hc 同族但权重独立；MoE 前第二次坍缩/门控",
    )
    b.node(
        "layer.post_attention_layernorm",
        op_type="DeepseekV4RMSNorm",
        granularity="named_instance",
        label="post_attention_layernorm · RMSNorm",
        kind="rmsnorm",
        module_path=f"{module}.post_attention_layernorm",
        sources=_op_sources(ctx.ops["ffn_norm"]),
        inputs=(("x", "ffn_collapsed"),),
        outputs=(("y", "ffn_normed"),),
        legacy_refs=ctx.legacy("op:ffn_norm", "tensor:ffn_norm_weight", "tensor:ffn_normed"),
        params=("hidden_size", "rms_norm_eps"),
        note="weight [hidden_size]；fp32 归一后转回 activation dtype",
    )
    moe_ops = _ops_ids(ctx, ("moe_", "shared_"))
    b.node(
        "layer.moe",
        op_type="DeepseekV4SparseMoeBlock",
        granularity="named_module",
        label="MoE FFN · 路由 + 专家 + 共享专家"
        + ("（Hash 门控）" if ctx.group["moe_type"] == "hash_moe" else "（TopK 门控）"),
        kind="composite",
        module_path=f"{module}.mlp",
        sources=ctx.op_sources(*moe_ops),
        inputs=(("x", "ffn_normed"),),
        outputs=(("y", "mlp_output"),),
        legacy_refs=tuple(f"op:{op_id}" for op_id in moe_ops) + ("tensor:mlp_output",),
        params=(
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "hidden_size",
        ),
        note=(
            "路由/分派/单专家 SwiGLU 模板/加权汇合/共享专家按具名模块保留；"
            "每专家 token 数 N_e 是运行期数据相关量，不虚构为常量"
        ),
    )
    b.node(
        "layer.ffn_residual",
        op_type="HyperConnection",
        granularity="named_module",
        label="FFN 残差：post 扩展 + combᵀ 混合",
        kind="composite",
        module_path=module,
        sources=ctx.op_sources(
            "ffn_residual_combT", "ffn_residual_expand", "ffn_residual_mix"
        ),
        inputs=(
            ("y", "mlp_output"),
            ("residual", "attn_out_streams"),
            ("post", "ffn_mhc_post"),
            ("comb", "ffn_mhc_comb"),
        ),
        outputs=(("streams", "hidden_streams_out"),),
        legacy_refs=ctx.legacy(
            "op:ffn_residual_combT",
            "op:ffn_residual_expand",
            "op:ffn_residual_mix",
            "tensor:hidden_streams_out",
        ),
        params=("hc_mult", "hidden_size"),
        note=(
            "hc post/comb 来自 ffn_hc 的另外两个输出；第二次 mHC 残差混合后回到 "
            "[batch_size, seqlen, hc_mult, hidden_size]"
        ),
    )
    b.node(
        "layer.output",
        op_type="output",
        granularity="external",
        label="hidden_streams_out",
        kind="external_out",
        module_path=module,
        sources=_op_sources(ctx.ops["ffn_residual_mix"]),
        inputs=(("streams", "hidden_streams_out"),),
        legacy_refs=ctx.legacy("tensor:hidden_streams_out", "op:ffn_residual_mix"),
        note="DecoderLayer 输出；下一层/最终 head 不属于本图范围",
    )


def _build_attention_nodes(ctx: _Context) -> None:
    b = ctx.builder
    values = ctx.values
    module = "model.layers.%d.self_attn" % ctx.block["layer_index"]
    kv_len = ctx.kv_len_axis
    b.tensor(
        "rope_cos",
        ("position_batch", "seqlen", "qk_rope_half"),
        label="cos（" + ("compress" if ctx.rope_type == "compress" else "main") + " RoPE）",
    )
    b.tensor(
        "rope_sin",
        ("position_batch", "seqlen", "qk_rope_half"),
        label="sin（" + ("compress" if ctx.rope_type == "compress" else "main") + " RoPE）",
    )

    # --- Q 低秩路径 -------------------------------------------------------
    b.tensor("q_a", ("batch_size", "seqlen", "q_lora_rank"), label="q_a_proj 输出")
    b.tensor("q_a_normed", ("batch_size", "seqlen", "q_lora_rank"), label="q_a_norm 输出")
    b.tensor("q_b_out", ("batch_size", "seqlen", "attention_heads_total"), label="q_b_proj 输出")
    b.tensor(
        "q_viewed",
        ("batch_size", "seqlen", "num_attention_heads", "head_dim"),
        label="q.view 输出",
    )
    b.tensor(
        "q_head",
        ("batch_size", "num_attention_heads", "seqlen", "head_dim"),
        label="q.transpose(1,2) 输出",
    )
    b.tensor(
        "q_normed",
        ("batch_size", "num_attention_heads", "seqlen", "head_dim"),
        label="q_b_norm 输出",
    )
    b.tensor(
        "q_rope",
        ("batch_size", "num_attention_heads", "seqlen", "head_dim"),
        label="q（部分 RoPE 后）",
    )
    b.node(
        "attn.q_a_proj",
        op_type="nn.Linear",
        granularity="named_instance",
        label="q_a_proj · Linear",
        kind="linear",
        module_path=f"{module}.q_a_proj",
        sources=_op_sources(ctx.ops["q_a_proj"]),
        inputs=(("x", "attn_normed"),),
        outputs=(("y", "q_a"),),
        legacy_refs=ctx.legacy("op:q_a_proj", "tensor:q_a", "tensor:q_weight"),
        params=("hidden_size", "q_lora_rank"),
        note="bias=False；weight [q_lora_rank, hidden_size]",
    )
    b.node(
        "attn.q_a_norm",
        op_type="DeepseekV4RMSNorm",
        granularity="named_instance",
        label="q_a_norm · RMSNorm",
        kind="rmsnorm",
        module_path=f"{module}.q_a_norm",
        sources=_op_sources(ctx.ops["q_a_norm"]),
        inputs=(("x", "q_a"),),
        outputs=(("y", "q_a_normed"),),
        legacy_refs=ctx.legacy("op:q_a_norm", "tensor:q_a_norm_weight", "tensor:q_a_normed"),
        params=("q_lora_rank", "rms_norm_eps"),
        note="weight [q_lora_rank]；低秩 Q 的秩归一，与 kv_norm 是两个实例",
    )
    b.node(
        "attn.q_b_proj",
        op_type="nn.Linear",
        granularity="named_instance",
        label="q_b_proj · Linear",
        kind="linear",
        module_path=f"{module}.q_b_proj",
        sources=_op_sources(ctx.ops["q_b_proj"]),
        inputs=(("x", "q_a_normed"),),
        outputs=(("y", "q_b_out"),),
        legacy_refs=ctx.legacy("op:q_b_proj", "tensor:q_weight", "tensor:q_b_weight"),
        params=("q_lora_rank", "num_attention_heads", "head_dim"),
        note="weight [num_attention_heads * head_dim, q_lora_rank]；view/transpose 是紧随的独立语义节点",
    )
    b.node(
        "attn.q_view",
        op_type="view",
        granularity="layout_op",
        label="q · View（拆分头维）",
        kind="view",
        module_path=module,
        sources=_op_sources(ctx.ops["q_b_proj"]),
        inputs=(("x", "q_b_out"),),
        outputs=(("y", "q_viewed"),),
        legacy_refs=ctx.legacy("op:q_b_proj", "tensor:q_head"),
        params=("num_attention_heads", "head_dim"),
        note=".view(*hidden_shape)：hidden_shape=(*input_shape, -1, head_dim)",
    )
    b.node(
        "attn.q_transpose",
        op_type="transpose",
        granularity="layout_op",
        label="q · Transpose（交换轴 1、2）",
        kind="transpose",
        module_path=module,
        sources=_op_sources(ctx.ops["q_b_proj"]),
        inputs=(("x", "q_viewed"),),
        outputs=(("y", "q_head"),),
        legacy_refs=ctx.legacy("op:q_b_proj", "tensor:q_head"),
        note=".transpose(1, 2)：把 seqlen 与 num_attention_heads 轴交换（64 头非单位轴）",
    )
    b.node(
        "attn.q_b_norm",
        op_type="DeepseekV4UnweightedRMSNorm",
        granularity="named_instance",
        label="q_b_norm · RMSNorm（无权重）",
        kind="rmsnorm",
        module_path=f"{module}.q_b_norm",
        sources=_op_sources(ctx.ops["q_b_norm"]),
        inputs=(("x", "q_head"),),
        outputs=(("y", "q_normed"),),
        legacy_refs=ctx.legacy("op:q_b_norm", "tensor:q_normed"),
        params=("head_dim", "rms_norm_eps"),
        note="无 learnable weight；沿最后一维 head_dim 做 fp32 RMS 归一",
    )
    b.node(
        "attn.q_rope",
        op_type="apply_rotary_pos_emb",
        granularity="named_op",
        label="RoPE · Q（尾部 qk_rope_head_dim）",
        kind="rope",
        module_path=module,
        sources=_op_sources(ctx.ops["q_rope_apply"]),
        inputs=(("x", "q_normed"), ("cos", "rope_cos"), ("sin", "rope_sin")),
        outputs=(("y", "q_rope"),),
        legacy_refs=ctx.legacy("op:q_rope_apply", "tensor:q_rope"),
        params=("qk_rope_head_dim",),
        note=(
            "interleaved RoPE：只旋转尾部 qk_rope_head_dim 通道，nope 部分不动；"
            "cos/sin 从 [1,seqlen,qk_rope_head_dim/2] repeat_interleave(2) 扩展"
        ),
    )

    # --- KV 路径（共享单投影）--------------------------------------------
    b.tensor("kv_linear", ("batch_size", "seqlen", "head_dim"), label="kv_proj 输出")
    b.tensor("kv_normed", ("batch_size", "seqlen", "head_dim"), label="kv_norm 输出")
    b.tensor(
        "kv_viewed",
        ("batch_size", "seqlen", "num_key_value_heads", "head_dim"),
        label="kv.view 输出",
    )
    b.tensor(
        "kv_head",
        ("batch_size", "num_key_value_heads", "seqlen", "head_dim"),
        label="kv.transpose(1,2) 输出",
    )
    b.tensor(
        "kv_rope",
        ("batch_size", "num_key_value_heads", "seqlen", "head_dim"),
        label="kv（部分 RoPE 后）",
    )
    b.node(
        "attn.kv_proj",
        op_type="nn.Linear",
        granularity="named_instance",
        label="kv_proj · Linear",
        kind="linear",
        module_path=f"{module}.kv_proj",
        sources=_op_sources(ctx.ops["kv_proj"]),
        inputs=(("x", "attn_normed"),),
        outputs=(("y", "kv_linear"),),
        legacy_refs=ctx.legacy(
            "op:kv_proj",
            "tensor:kv_proj_out",
            "tensor:kv_main",
            "tensor:kv_weight",
            "tensor:kv_weight_attn",
        ),
        params=("hidden_size", "head_dim", "num_key_value_heads"),
        note=(
            "bias=False；num_key_value_heads=1 时直接投影到 head_dim，"
            "weight [head_dim, hidden_size]——不是 num_key_value_heads*head_dim"
        ),
    )
    b.node(
        "attn.kv_norm",
        op_type="DeepseekV4RMSNorm",
        granularity="named_instance",
        label="kv_norm · RMSNorm",
        kind="rmsnorm",
        module_path=f"{module}.kv_norm",
        sources=_op_sources(ctx.ops["kv_proj"]),
        inputs=(("x", "kv_linear"),),
        outputs=(("y", "kv_normed"),),
        legacy_refs=ctx.legacy(
            "op:kv_proj",
            "tensor:kv_norm_weight",
            "tensor:kv_proj_out",
            "tensor:kv_main",
        ),
        params=("head_dim", "rms_norm_eps"),
        note="weight [head_dim]；与 kv_proj 分开辨认（源码相邻两行不代表合并为一个计算）",
    )
    b.node(
        "attn.kv_view",
        op_type="view",
        granularity="layout_op",
        label="kv · View（增加单 KV 头轴）",
        kind="view",
        module_path=module,
        sources=_op_sources(ctx.ops["kv_head_view"]),
        inputs=(("x", "kv_normed"),),
        outputs=(("y", "kv_viewed"),),
        legacy_refs=ctx.legacy("op:kv_head_view", "tensor:kv_head"),
        params=("num_key_value_heads", "head_dim"),
        note=".view(*hidden_shape)（B,T,1,Dh）：把 head_dim 拆成单 KV 头轴 + head_dim",
    )
    b.node(
        "attn.kv_transpose",
        op_type="transpose",
        granularity="layout_op",
        label="kv · Transpose（交换轴 1、2）",
        kind="transpose",
        module_path=module,
        sources=_op_sources(ctx.ops["kv_head_view"]),
        inputs=(("x", "kv_viewed"),),
        outputs=(("y", "kv_head"),),
        legacy_refs=ctx.legacy("op:kv_head_view", "tensor:kv_head"),
        note=".transpose(1, 2)；单 KV 头时只移动单位轴，是 R5 可证明化简的候选",
    )
    b.node(
        "attn.kv_rope",
        op_type="apply_rotary_pos_emb",
        granularity="named_op",
        label="RoPE · KV（同一张量送 K/V）",
        kind="rope",
        module_path=module,
        sources=_op_sources(ctx.ops["kv_rope_apply"]),
        inputs=(("x", "kv_head"), ("cos", "rope_cos"), ("sin", "rope_sin")),
        outputs=(("y", "kv_rope"),),
        legacy_refs=ctx.legacy("op:kv_rope_apply", "tensor:kv_rope"),
        params=("qk_rope_head_dim",),
        note="K=V 同一张量：RoPE 后同时作为 K、V；不存在两条独立 K/V 投影",
    )

    # --- RoPE cos/sin（辅助输入，短旁枝）----------------------------------
    rope_gen_op = "rope_compress" if ctx.rope_type == "compress" else "rope_main"
    b.node(
        "attn.rope_freqs",
        op_type="DeepseekV4RotaryEmbedding",
        granularity="named_source",
        label="cos / sin · " + ("compress RoPE" if ctx.rope_type == "compress" else "main RoPE"),
        kind="aux",
        module_path=module,
        sources=_op_sources(ctx.ops[rope_gen_op]),
        outputs=(("cos", "rope_cos"), ("sin", "rope_sin")),
        legacy_refs=ctx.legacy(f"op:{rope_gen_op}", "tensor:cos_q", "tensor:sin_q"),
        params=("qk_rope_head_dim", "sliding_window"),
        note=(
            "rotary_emb 由模型外层按 rope_type 传入：SWA 用 main θ="
            f"{values.rope_main_theta}；CSA/HCA 用 compress θ={values.rope_compress_theta}"
            "（YaRN）；本层取 "
            + ("compress" if ctx.rope_type == "compress" else "main")
        ),
    )

    # --- 缓存状态 ---------------------------------------------------------
    b.tensor(
        "cache_keys",
        ("batch_size", "num_key_value_heads", "kv_cache_len", "head_dim"),
        role="cache",
        label="cache.keys/values（调用前，W-1 尾部）",
    )
    b.tensor(
        "kv_full",
        ("batch_size", "num_key_value_heads", "kv_length", "head_dim"),
        label="cache.update 返回 full",
    )
    b.tensor(
        "cache_keys_post",
        ("batch_size", "num_key_value_heads", "kv_cache_len_post", "head_dim"),
        role="cache",
        label="cache.keys/values（调用后）",
    )
    if ctx.has_compressor:
        b.tensor(
            "compressor_cache",
            ("batch_size", "compressor_cache_len", "head_dim"),
            role="cache",
            label="compressor 状态（调用前：buffer/压缩条目/overlap）",
        )
        b.tensor(
            "compressor_cache_post",
            ("batch_size", "compressor_cache_len_post", "head_dim"),
            role="cache",
            label="compressor 状态（调用后）",
        )
        b.tensor(
            "compressed_kv",
            ("batch_size", "num_key_value_heads", "num_compressed", "head_dim"),
            label="compressed_kv（本次调用新条目 + 历史条目）",
        )
        b.tensor(
            "block_bias",
            ("batch_size", "position_batch", "seqlen", "num_compressed"),
            role="mask",
            label="block_bias（压缩条目可见性）",
        )
    if ctx.has_indexer:
        b.tensor(
            "indexer_cache",
            ("batch_size", "compressor_cache_len", "index_head_dim"),
            role="cache",
            label="indexer 状态（调用前）",
        )
        b.tensor(
            "indexer_cache_post",
            ("batch_size", "compressor_cache_len_post", "index_head_dim"),
            role="cache",
            label="indexer 状态（调用后）",
        )
        b.tensor(
            "indexer_topk",
            ("batch_size", "seqlen", "num_selected"),
            role="index",
            label="indexer top-k 索引",
        )

    cache_in_outputs: list[tuple[str, str]] = [("kv", "cache_keys")]
    if ctx.has_compressor:
        cache_in_outputs.append(("compressor", "compressor_cache"))
    if ctx.has_indexer:
        cache_in_outputs.append(("indexer", "indexer_cache"))
    b.node(
        "attn.cache_in",
        op_type="state",
        granularity="state_boundary",
        label="cache · 调用前状态",
        kind="state_in",
        module_path=module,
        sources=_op_sources(ctx.ops["cache_update"]),
        outputs=tuple(cache_in_outputs),
        legacy_refs=ctx.legacy("tensor:cache_keys", "tensor:cache_cumlen", "op:cache_update"),
        note="跨调用状态输入：主 KV 持久滑窗尾部（以及压缩器/Indexer 状态）",
    )
    b.node(
        "attn.cache_update",
        op_type="DeepseekV4CSACache.update",
        granularity="named_module",
        label="cache.update · 主 KV（K=V）",
        kind="cache_update",
        module_path=module,
        sources=_op_sources(ctx.ops["cache_update"]),
        inputs=(("kv", "kv_rope"), ("cache", "cache_keys")),
        outputs=(("full", "kv_full"), ("cache", "cache_keys_post")),
        legacy_refs=ctx.legacy("op:cache_update", "tensor:kv_full", "tensor:cache_keys_post"),
        params=("sliding_window", "kvcachelen", "seqlen"),
        note=(
            "full=cat([keys, kv],dim=-2)；keys 保留 W-1 尾部；K=V 同存储。"
            "本条使用 kvcachelen 表达历史尾部，不把新 KV 写成 seqlen+kvcachelen 的总 KV"
        ),
    )
    cache_out_inputs: list[tuple[str, str]] = [("kv", "cache_keys_post")]
    if ctx.has_compressor:
        cache_out_inputs.append(("compressor", "compressor_cache_post"))
    if ctx.has_indexer:
        cache_out_inputs.append(("indexer", "indexer_cache_post"))
    b.node(
        "attn.cache_out",
        op_type="state",
        granularity="state_boundary",
        label="cache · 调用后状态",
        kind="state_out",
        module_path=module,
        sources=_op_sources(ctx.ops["cache_update"]),
        inputs=tuple(cache_out_inputs),
        legacy_refs=ctx.legacy("tensor:cache_keys_post", "tensor:cache_keys"),
        note="跨调用状态输出：下一调用读取；不因排版隐藏缓存依赖",
    )

    if ctx.has_compressor:
        compressor_label = (
            "compressor · CSA"
            if ctx.attention_type == "compressed_sparse_attention"
            else "compressor · HCA"
        )
        compressor_ops = _ops_ids(ctx, ("c_", "hca_"))
        b.node(
            "attn.compressor",
            op_type="DeepseekV4CSACompressor"
            if ctx.attention_type == "compressed_sparse_attention"
            else "DeepseekV4HCACompressor",
            granularity="named_module",
            label=compressor_label,
            kind="composite",
            module_path=f"{module}.compressor",
            sources=ctx.op_sources(*compressor_ops),
            inputs=(("x", "attn_normed"), ("cache", "compressor_cache")),
            outputs=(("compressed", "compressed_kv"), ("cache", "compressor_cache_post")),
            legacy_refs=ctx.legacy(
                tuple(f"op:{op_id}" for op_id in compressor_ops)
                + (
                    "tensor:cache_compressed_c_post",
                    "tensor:c_block_bias",
                    "tensor:block_bias",
                )
            ),
            params=("compress_rate", "head_dim", "num_key_value_heads"),
            note=(
                "具名计算：kv_proj/gate_proj → 窗口 buffer 合并/闭窗 → Ca/Cb 重叠布局 → "
                "softmax 加权汇聚 → kv_norm → 压缩条目 RoPE → 状态追加 → block_bias。"
                f"本次 compress_rate={ctx.compress_rate}；状态读写在 state 边表达"
            ),
            condition="layer_type != 'sliding_attention'",
        )
        if ctx.has_indexer:
            indexer_ops = _ops_ids(ctx, ("i_",))
            b.node(
                "attn.indexer",
                op_type="DeepseekV4Indexer",
                granularity="named_module",
                label="indexer · 稀疏选择",
                kind="composite",
                module_path=f"{module}.compressor.indexer",
                sources=ctx.op_sources(*indexer_ops),
                inputs=(("x", "attn_normed"), ("q", "q_a_normed"), ("cache", "indexer_cache")),
                outputs=(("topk", "indexer_topk"), ("cache", "indexer_cache_post")),
                legacy_refs=tuple(f"op:{op_id}" for op_id in indexer_ops)
                + ("tensor:indexer_topk",),
                params=("index_n_heads", "index_head_dim", "index_topk", "compress_rate"),
                note=(
                    "具名计算：q_b_proj+RoPE、KV/gate 投影、压缩条目、scorer（relu(Q·Kᵀ)·Ih^-0.5"
                    "·weights 求和）、future/causal mask、topk、-1 哨兵；输出索引而非 Value"
                ),
                condition="layer_type == 'compressed_sparse_attention'",
            )
        if ctx.has_compressor:
            bias_ops = _ops_ids(ctx, ("c_block_bias", "hca_block_bias"))
            if ctx.has_indexer:
                # CSA：compressed_kv + Indexer topk；只生成可见性 mask。
                bias_inputs = (("compressed", "compressed_kv"), ("topk", "indexer_topk"))
                bias_type = "DeepseekV4CSACompressor.block_bias"
                bias_label = "block_bias · CSA 可见性"
                bias_condition = "layer_type == 'compressed_sparse_attention'"
                bias_note = (
                    "Indexer top-k 只决定压缩条目的可见性（0/-inf block_bias），"
                    "不裁剪主 KV 或压缩 KV；compressor 的压缩轴不因 topk 改变。"
                )
                bias_params = ("compress_rate", "index_topk")
                bias_legacy = ("tensor:c_block_bias", "tensor:block_bias", "tensor:indexer_topk")
            else:
                # HCA：position_ids + compressed_kv；无 Indexer/TopK。
                b.tensor(
                    "position_ids",
                    ("batch_size", "seqlen"),
                    role="input",
                    label="position_ids（HCA 因果可见性）",
                )
                b.node(
                    "attn.position_ids",
                    op_type="input",
                    granularity="boundary",
                    label="position_ids",
                    kind="composite",
                    module_path=module,
                    sources=_op_sources(ctx.ops["hca_block_bias"]),
                    outputs=(("position_ids", "position_ids"),),
                    legacy_refs=ctx.legacy("op:hca_block_bias"),
                    note="HCA block_bias 按 position_ids 构建因果阈值；不是 Indexer 输入",
                )
                bias_inputs = (("compressed", "compressed_kv"), ("position_ids", "position_ids"))
                bias_type = "DeepseekV4HCACompressor.block_bias"
                bias_label = "block_bias · HCA（None 分支）"
                bias_condition = "layer_type == 'heavily_compressed_attention'"
                bias_note = (
                    "HCA 用 position_ids 计算每 query 的压缩条目因果阈值；"
                    "seq_len==1 或 compressed_len==0 时返回 None，mask_extend 走零扩展。"
                    "无 Indexer/TopK，不能套用 CSA 条件。"
                )
                bias_params = ("compress_rate",)
                bias_legacy = ("tensor:block_bias",)
            b.node(
                "attn.block_bias",
                op_type=bias_type,
                granularity="named_op",
                label=bias_label,
                kind="composite",
                module_path=f"{module}.compressor",
                sources=ctx.op_sources(*bias_ops),
                inputs=bias_inputs,
                outputs=(("bias", "block_bias"),),
                legacy_refs=ctx.legacy(
                    *(f"op:{op_id}" for op_id in bias_ops), *bias_legacy,
                ),
                params=bias_params,
                note=bias_note,
                condition=bias_condition,
            )
            if not ctx.has_indexer:
                b.edge(
                    "e.position_ids.block_bias", "attn.position_ids", "position_ids",
                    "attn.block_bias", "position_ids", kind="data",
                )

        b.tensor(
            "kv_all",
            ("batch_size", "num_key_value_heads", "kv_length_plus_compressed", "head_dim"),
            label="kv（主 KV + 压缩条目）",
        )
        b.node(
            "attn.kv_concat",
            op_type="torch.cat",
            granularity="layout_op",
            label="kv · Concat（压缩条目）",
            kind="concat",
            module_path=module,
            sources=_op_sources(ctx.ops["kv_concat"]),
            inputs=(("main", "kv_full"), ("compressed", "compressed_kv")),
            outputs=(("y", "kv_all"),),
            legacy_refs=ctx.legacy("op:kv_concat", "tensor:kv_all"),
            params=("kv_length", "num_compressed"),
            note="拼接发生在 cache.update 之后：Attention 读到的 KV = 滑窗 full + 全部压缩条目",
            condition="compressor is not None",
        )

    # --- mask -------------------------------------------------------------
    if ctx.has_compressor:
        b.tensor(
            "mask_model",
            ("batch_size", "position_batch", "seqlen", "kv_length"),
            role="mask",
            label="模型级滑动因果 mask",
        )
        mask_extend_op = "mask_extend" if "mask_extend" in ctx.ops else "c_mask_extend"
        b.tensor(
            "mask_ext",
            ("batch_size", "position_batch", "seqlen", "kv_length_plus_compressed"),
            role="mask",
            label="扩展后的 mask（含压缩条目 block_bias）",
        )
        b.node(
            "attn.mask",
            op_type="masking_utils.create_sliding_window_causal_mask",
            granularity="named_source",
            label="mask · 滑动因果掩码",
            kind="aux",
            module_path="masking_utils",
            sources=_op_sources(ctx.ops["mask_model_build"]),
            outputs=(("y", "mask_model"),),
            legacy_refs=ctx.legacy("op:mask_model_build", "tensor:mask_model"),
            params=("sliding_window", "kvcachelen", "seqlen"),
            note="0 可见 / -inf 屏蔽；q_offset=累积历史，kv_offset=max(cumulative-W+1,0)",
        )
        b.node(
            "attn.mask_extend",
            op_type="torch.cat",
            granularity="layout_op",
            label="mask · Cat（block_bias）",
            kind="concat",
            module_path=module,
            sources=_op_sources(ctx.ops[mask_extend_op]),
            inputs=(("mask", "mask_model"), ("bias", "block_bias")),
            outputs=(("y", "mask_ext"),),
            legacy_refs=ctx.legacy(f"op:{mask_extend_op}", "tensor:mask_ext"),
            params=("kv_length", "num_compressed"),
            note="block_bias 为压缩条目携带每 query 因果性 + Indexer 可见性；不零填充",
            condition="compressor is not None",
        )
        mask_tensor = "mask_ext"
    else:
        b.tensor(
            "mask_model",
            ("batch_size", "position_batch", "seqlen", "kv_length"),
            role="mask",
            label="模型级滑动因果 mask",
        )
        b.node(
            "attn.mask",
            op_type="masking_utils.create_sliding_window_causal_mask",
            granularity="named_source",
            label="mask · 滑动因果掩码",
            kind="aux",
            module_path="masking_utils",
            sources=_op_sources(ctx.ops["mask_model_build"]),
            outputs=(("y", "mask_model"),),
            legacy_refs=ctx.legacy("op:mask_model_build", "tensor:mask_model"),
            params=("sliding_window", "kvcachelen", "seqlen"),
            note="0 可见 / -inf 屏蔽；SWA 层无压缩条目，不需要扩展",
        )
        mask_tensor = "mask_model"

    # --- attention core ---------------------------------------------------
    b.tensor(
        "kv_repeated",
        ("batch_size", "kv_broadcast_heads", kv_len, "head_dim"),
        label="repeat_kv(kv)：单 KV 头广播到全部查询头",
    )
    b.node(
        "attn.repeat_kv",
        op_type="repeat_kv",
        granularity="named_op",
        label="repeat_kv · 单 KV 头广播",
        kind="broadcast",
        module_path="eager_attention_forward.repeat_kv",
        sources=_op_sources(ctx.ops["repeat_kv"]),
        inputs=(("x", "kv_all" if ctx.has_compressor else "kv_full"),),
        outputs=(("y", "kv_repeated"),),
        legacy_refs=ctx.legacy("op:repeat_kv", "tensor:kv_repeated"),
        params=("num_key_value_heads", "num_attention_heads"),
        note=(
            "expand+reshape：num_key_value_heads → num_attention_heads"
            "（不是两次独立 K/V 投影；n_rep=1 时才退化为恒等）"
        ),
    )
    b.tensor(
        "kv_transposed",
        ("batch_size", "kv_broadcast_heads", "head_dim", kv_len),
        label="Kᵀ（最后两个矩阵轴交换）",
    )
    b.node(
        "attn.kv_transposed",
        op_type="transpose",
        granularity="layout_op",
        label="K · Transpose（交换最后两轴）",
        kind="transpose",
        module_path="eager_attention_forward",
        sources=_op_sources(ctx.ops["qk_scaled"]),
        inputs=(("x", "kv_repeated"),),
        outputs=(("y", "kv_transposed"),),
        legacy_refs=ctx.legacy("op:qk_scaled", "tensor:kv_repeated"),
        note="key_states.transpose(2, 3)：只交换最后两个矩阵轴（R7 的候选）",
    )
    b.tensor(
        "qk_logits",
        ("batch_size", "num_attention_heads", "seqlen", kv_len),
        label="attn_weights（QKᵀ）",
    )
    b.tensor(
        "qk_scaled",
        ("batch_size", "num_attention_heads", "seqlen", kv_len),
        label="attn_weights × scaling",
    )
    b.tensor(
        "qk_masked",
        ("batch_size", "num_attention_heads", "seqlen", kv_len),
        label="attn_weights + mask",
    )
    b.node(
        "attn.qk_matmul",
        op_type="torch.matmul",
        granularity="named_op",
        label="QKᵀ · MatMul",
        kind="matmul",
        module_path="eager_attention_forward",
        sources=_op_sources(ctx.ops["qk_scaled"]),
        inputs=(("q", "q_rope"), ("k", "kv_transposed")),
        outputs=(("y", "qk_logits"),),
        legacy_refs=ctx.legacy("op:qk_scaled", "tensor:qk_logits"),
        params=("num_attention_heads", "head_dim"),
        note="Q 与 Kᵀ 的 batch/head 轴保持；只合并最后两轴的转置语义",
        attributes={
            "lhs_port": "q",
            "rhs_port": "k",
            "lhs_symbol": "Q",
            "rhs_symbol": "K",
            "matmul": "qk",
        },
    )
    b.node(
        "attn.scale",
        op_type="mul",
        granularity="named_op",
        label="scale · × head_dim^-0.5",
        kind="scale",
        module_path="eager_attention_forward",
        sources=_op_sources(ctx.ops["qk_scaled"]),
        inputs=(("x", "qk_logits"),),
        outputs=(("y", "qk_scaled"),),
        legacy_refs=ctx.legacy("op:qk_scaled", "tensor:qk_logits"),
        params=("head_dim",),
        note="scaling=head_dim**-0.5；与 MatMul 分开表达（不声称设备已融合）",
    )
    b.node(
        "attn.mask_add",
        op_type="add",
        granularity="named_op",
        label="mask · Add（加法型 mask）",
        kind="add",
        module_path="eager_attention_forward",
        sources=_op_sources(ctx.ops["mask_add"]),
        inputs=(("x", "qk_scaled"), ("mask", mask_tensor)),
        outputs=(("y", "qk_masked"),),
        legacy_refs=ctx.legacy("op:mask_add", "tensor:qk_masked"),
        params=("kv_length",),
        note="0 可见 / -inf 屏蔽；加法生成新张量，不原地改 qk_logits",
    )
    b.tensor("sinks", ("num_attention_heads",), role="weight", label="sinks（每头可学习标量）")
    b.tensor(
        "attn_probs",
        ("batch_size", "num_attention_heads", "seqlen", kv_len),
        label="attn_probs（softmax，去 sink）",
    )
    b.node(
        "attn.sinks",
        op_type="parameter",
        granularity="named_source",
        label="sinks · [num_attention_heads]",
        kind="aux",
        module_path=module,
        sources=_op_sources(ctx.ops["sink_softmax"]),
        outputs=(("y", "sinks"),),
        legacy_refs=ctx.legacy("tensor:sinks", "op:sink_softmax"),
        note="每头一个可学习标量参与 softmax 竞争，但不产出输出（去 sink 切片）",
    )
    b.node(
        "attn.sink_softmax",
        op_type="F.softmax",
        granularity="named_op",
        label="softmax · cat(sinks) + 稳定化 + 去 sink",
        kind="softmax",
        module_path="eager_attention_forward",
        sources=_op_sources(ctx.ops["sink_softmax"]),
        inputs=(("x", "qk_masked"), ("sinks", "sinks")),
        outputs=(("y", "attn_probs"),),
        legacy_refs=ctx.legacy(
            "op:sink_softmax", "tensor:attn_probs", "tensor:combined_logits"
        ),
        params=("num_attention_heads",),
        note="cat(sinks) → 减 max → softmax → 去 sink；dropout 在推理期为 p=0",
    )
    b.tensor(
        "av_out",
        ("batch_size", "num_attention_heads", "seqlen", "head_dim"),
        label="attn_output（AV，头维在前）",
    )
    b.tensor(
        "av_out_bshd",
        ("batch_size", "seqlen", "num_attention_heads", "head_dim"),
        label="attn_output.transpose(1,2).contiguous()",
    )
    b.node(
        "attn.av_matmul",
        op_type="torch.matmul",
        granularity="named_op",
        label="AV · MatMul",
        kind="matmul",
        module_path="eager_attention_forward",
        sources=_op_sources(ctx.ops["av_matmul"]),
        inputs=(("p", "attn_probs"), ("v", "kv_repeated")),
        outputs=(("y", "av_out"),),
        legacy_refs=ctx.legacy("op:av_matmul", "tensor:attn_head_out"),
        params=("num_attention_heads", "head_dim"),
        note="V 与 K 是同一张量（K=V 共享 KV）；概率轴与 KV 轴闭合",
        attributes={
            "lhs_port": "p",
            "rhs_port": "v",
            "lhs_symbol": "P",
            "rhs_symbol": "V",
            "matmul": "av",
        },
    )
    b.node(
        "attn.out_transpose",
        op_type="transpose",
        granularity="layout_op",
        label="Transpose(1,2) · eager 输出 layout",
        kind="transpose",
        module_path="eager_attention_forward",
        sources=_op_sources(ctx.ops["av_matmul"]),
        inputs=(("x", "av_out"),),
        outputs=(("y", "av_out_bshd"),),
        legacy_refs=ctx.legacy("op:av_matmul", "tensor:attn_head_out"),
        note="attn_output.transpose(1, 2).contiguous()；纯 layout 修正",
    )
    b.tensor(
        "av_out_bhsd",
        ("batch_size", "num_attention_heads", "seqlen", "head_dim"),
        label="attn_output.transpose(1,2)（逆 RoPE 输入）",
    )
    b.node(
        "attn.rope_in_transpose",
        op_type="transpose",
        granularity="layout_op",
        label="Transpose(1,2) · 逆 RoPE 前 layout",
        kind="transpose",
        module_path=module,
        sources=_op_sources(ctx.ops["attn_inv_rope"]),
        inputs=(("x", "av_out_bshd"),),
        outputs=(("y", "av_out_bhsd"),),
        legacy_refs=ctx.legacy("op:attn_inv_rope", "tensor:attn_head_out"),
        note="apply_rotary_pos_emb 期望 [B,H,S,D]；与 eager 输出 Transpose 复合为恒等（R3 候选）",
    )
    b.tensor(
        "attn_rope_inv",
        ("batch_size", "num_attention_heads", "seqlen", "head_dim"),
        label="attn_output（逆 RoPE 后）",
    )
    b.node(
        "attn.inv_rope",
        op_type="apply_rotary_pos_emb",
        granularity="named_op",
        label="RoPE · 输出逆旋转（-sin 共轭）",
        kind="rope",
        module_path=module,
        sources=_op_sources(ctx.ops["attn_inv_rope"]),
        inputs=(("x", "av_out_bhsd"), ("cos", "rope_cos"), ("sin", "rope_sin")),
        outputs=(("y", "attn_rope_inv"),),
        legacy_refs=ctx.legacy("op:attn_inv_rope", "tensor:attn_rope_inv"),
        params=("qk_rope_head_dim",),
        note="用 -sin 在 query 位置共轭抵消 V 的 RoPE，使输出只依赖相对距离",
    )
    b.tensor(
        "attn_rope_inv_bshd",
        ("batch_size", "seqlen", "num_attention_heads", "head_dim"),
        label="attn_output（逆 RoPE 后，[B,S,H,D]）",
    )
    b.node(
        "attn.rope_out_transpose",
        op_type="transpose",
        granularity="layout_op",
        label="Transpose(1,2) · 逆 RoPE 后 layout",
        kind="transpose",
        module_path=module,
        sources=_op_sources(ctx.ops["attn_inv_rope"]),
        inputs=(("x", "attn_rope_inv"),),
        outputs=(("y", "attn_rope_inv_bshd"),),
        legacy_refs=ctx.legacy("op:attn_inv_rope", "tensor:attn_rope_inv"),
        note="逆 RoPE 后再 transpose 回 [B,seqlen,H,D]，随后 reshape 分组",
    )
    b.tensor(
        "attn_grouped",
        ("batch_size", "seqlen", "o_groups", "o_group_width"),
        label="attn_output.reshape(B,T,G,-1)",
    )
    b.tensor(
        "o_a_out",
        ("batch_size", "seqlen", "o_groups_total"),
        label="o_a_proj 输出（组低秩）",
    )
    b.node(
        "attn.group_reshape",
        op_type="reshape",
        granularity="layout_op",
        label="Reshape · 分组（B,T,G,·）",
        kind="reshape",
        module_path=module,
        sources=_op_sources(ctx.ops["attn_group_reshape"]),
        inputs=(("x", "attn_rope_inv_bshd"),),
        outputs=(("y", "attn_grouped"),),
        legacy_refs=ctx.legacy("op:attn_group_reshape", "tensor:attn_grouped"),
        params=("o_groups", "num_attention_heads", "head_dim"),
        note="attn_output.reshape(*input_shape, o_groups, -1)；为分组输出投影准备",
    )
    b.node(
        "attn.o_a_proj",
        op_type="DeepseekV4GroupedLinear",
        granularity="named_instance",
        label="o_a_proj · GroupedLinear",
        kind="grouped_linear",
        module_path=f"{module}.o_a_proj",
        sources=_op_sources(ctx.ops["o_a_proj"]),
        inputs=(("x", "attn_grouped"),),
        outputs=(("y", "o_a_out"),),
        legacy_refs=ctx.legacy("op:o_a_proj", "tensor:o_a_weight", "tensor:o_a_out"),
        params=("o_groups", "o_lora_rank", "num_attention_heads", "head_dim"),
        note="块对角分组线性：每组 (num_attention_heads*head_dim/o_groups) → o_lora_rank",
    )
    b.node(
        "attn.o_b_proj",
        op_type="nn.Linear",
        granularity="named_instance",
        label="o_b_proj · Linear",
        kind="linear",
        module_path=f"{module}.o_b_proj",
        sources=_op_sources(ctx.ops["o_b_proj"]),
        inputs=(("x", "o_a_out"),),
        outputs=(("y", "attn_block_output"),),
        legacy_refs=ctx.legacy("op:o_b_proj", "tensor:o_b_weight", "tensor:attn_block_output"),
        params=("o_groups", "o_lora_rank", "hidden_size"),
        note="weight [hidden_size, o_groups * o_lora_rank]；flatten 后输出 [batch_size, seqlen, hidden_size]",
    )


# ---------------------------------------------------------------------------
# 边 / guard / 状态 / alias
# ---------------------------------------------------------------------------


def _build_edges(ctx: _Context) -> None:
    b = ctx.builder
    guards = ctx.guards
    transitions = ctx.transitions
    compressor_guard = guards.get("g.compressor")
    indexer_guard = guards.get("g.indexer")

    def edge(edge_id: str, src: str, src_port: str, dst: str, dst_port: str, **kwargs: Any) -> None:
        b.edge(edge_id, src, src_port, dst, dst_port, **kwargs)

    # Layer 主干
    edge("e.layer.in.hc", "layer.input", "streams", "layer.attn_hc", "streams")
    edge("e.layer.hc.norm", "layer.attn_hc", "collapsed", "layer.input_layernorm", "x")
    edge("e.layer.norm.q_a", "layer.input_layernorm", "y", "attn.q_a_proj", "x")
    edge("e.layer.norm.kv", "layer.input_layernorm", "y", "attn.kv_proj", "x")
    edge("e.layer.norm.input", "layer.input", "streams", "layer.attn_residual", "residual")
    edge("e.layer.attn.residual", "attn.o_b_proj", "y", "layer.attn_residual", "y")
    edge("e.layer.attn_hc.post", "layer.attn_hc", "post", "layer.attn_residual", "post")
    edge("e.layer.attn_hc.comb", "layer.attn_hc", "comb", "layer.attn_residual", "comb")
    edge("e.layer.residual.ffn_hc", "layer.attn_residual", "streams", "layer.ffn_hc", "streams")
    edge(
        "e.layer.residual.residual",
        "layer.attn_residual",
        "streams",
        "layer.ffn_residual",
        "residual",
    )
    edge("e.layer.ffn_hc.norm", "layer.ffn_hc", "collapsed", "layer.post_attention_layernorm", "x")
    edge("e.layer.norm2.moe", "layer.post_attention_layernorm", "y", "layer.moe", "x")
    edge("e.layer.moe.residual", "layer.moe", "y", "layer.ffn_residual", "y")
    edge("e.layer.ffn_hc.post", "layer.ffn_hc", "post", "layer.ffn_residual", "post")
    edge("e.layer.ffn_hc.comb", "layer.ffn_hc", "comb", "layer.ffn_residual", "comb")
    edge("e.layer.residual.out", "layer.ffn_residual", "streams", "layer.output", "streams")

    # Q 低秩路径
    edge("e.q.a.norm", "attn.q_a_proj", "y", "attn.q_a_norm", "x")
    edge("e.q.norm.b", "attn.q_a_norm", "y", "attn.q_b_proj", "x")
    edge("e.q.b.view", "attn.q_b_proj", "y", "attn.q_view", "x")
    edge("e.q.view.transpose", "attn.q_view", "y", "attn.q_transpose", "x")
    edge("e.q.transpose.norm", "attn.q_transpose", "y", "attn.q_b_norm", "x")
    edge("e.q.norm.rope", "attn.q_b_norm", "y", "attn.q_rope", "x")
    edge("e.ropes.cos.q", "attn.rope_freqs", "cos", "attn.q_rope", "cos")
    edge("e.ropes.sin.q", "attn.rope_freqs", "sin", "attn.q_rope", "sin")

    # KV 路径
    edge("e.kv.linear.norm", "attn.kv_proj", "y", "attn.kv_norm", "x")
    edge("e.kv.norm.view", "attn.kv_norm", "y", "attn.kv_view", "x")
    edge("e.kv.view.transpose", "attn.kv_view", "y", "attn.kv_transpose", "x")
    edge("e.kv.transpose.rope", "attn.kv_transpose", "y", "attn.kv_rope", "x")
    edge("e.ropes.cos.kv", "attn.rope_freqs", "cos", "attn.kv_rope", "cos")
    edge("e.ropes.sin.kv", "attn.rope_freqs", "sin", "attn.kv_rope", "sin")
    edge("e.kv.rope.cache", "attn.kv_rope", "y", "attn.cache_update", "kv")

    # 缓存状态（state 边，不隐藏副作用）
    edge(
        "e.state.kv.read",
        "attn.cache_in",
        "kv",
        "attn.cache_update",
        "cache",
        kind="state",
        transition_ref=transitions["tr.attn.kv"],
    )
    edge(
        "e.state.kv.write",
        "attn.cache_update",
        "cache",
        "attn.cache_out",
        "kv",
        kind="state",
        transition_ref=transitions["tr.attn.kv"],
    )

    if ctx.has_compressor:
        edge(
            "e.state.compressor.read",
            "attn.cache_in",
            "compressor",
            "attn.compressor",
            "cache",
            kind="state",
            transition_ref=transitions["tr.attn.compressor"],
            guard_ref=compressor_guard,
        )
        edge(
            "e.state.compressor.write",
            "attn.compressor",
            "cache",
            "attn.cache_out",
            "compressor",
            kind="state",
            transition_ref=transitions["tr.attn.compressor"],
            guard_ref=compressor_guard,
        )
        edge(
            "e.layer.norm.compressor",
            "layer.input_layernorm",
            "y",
            "attn.compressor",
            "x",
            guard_ref=compressor_guard,
        )
        if ctx.has_indexer:
            edge(
                "e.state.indexer.read",
                "attn.cache_in",
                "indexer",
                "attn.indexer",
                "cache",
                kind="state",
                transition_ref=transitions["tr.attn.indexer"],
                guard_ref=indexer_guard,
            )
            edge(
                "e.state.indexer.write",
                "attn.indexer",
                "cache",
                "attn.cache_out",
                "indexer",
                kind="state",
                transition_ref=transitions["tr.attn.indexer"],
                guard_ref=indexer_guard,
            )
            edge(
                "e.layer.norm.indexer",
                "layer.input_layernorm",
                "y",
                "attn.indexer",
                "x",
                guard_ref=indexer_guard,
            )
            edge(
                "e.q.a_norm.indexer",
                "attn.q_a_norm",
                "y",
                "attn.indexer",
                "q",
                guard_ref=indexer_guard,
            )
            edge(
                "e.indexer.block_bias",
                "attn.indexer",
                "topk",
                "attn.block_bias",
                "topk",
                guard_ref=indexer_guard,
            )
        edge(
            "e.cache.concat.main",
            "attn.cache_update",
            "full",
            "attn.kv_concat",
            "main",
            guard_ref=compressor_guard,
        )
        edge(
            "e.compressor.concat",
            "attn.compressor",
            "compressed",
            "attn.kv_concat",
            "compressed",
            guard_ref=compressor_guard,
        )
        edge(
            "e.concat.repeat",
            "attn.kv_concat",
            "y",
            "attn.repeat_kv",
            "x",
            guard_ref=compressor_guard,
        )
    else:
        edge("e.cache.repeat", "attn.cache_update", "full", "attn.repeat_kv", "x")

    # mask
    if ctx.has_compressor:
        edge("e.mask.model", "attn.mask", "y", "attn.mask_extend", "mask")
        edge(
            "e.compressor.block_bias2",
            "attn.compressor",
            "compressed",
            "attn.block_bias",
            "compressed",
            guard_ref=compressor_guard,
        )
        edge(
            "e.block_bias.mask2",
            "attn.block_bias",
            "bias",
            "attn.mask_extend",
            "bias",
            guard_ref=compressor_guard,
        )
        edge(
            "e.mask.extend.add",
            "attn.mask_extend",
            "y",
            "attn.mask_add",
            "mask",
            guard_ref=compressor_guard,
        )
    else:
        edge("e.mask.model", "attn.mask", "y", "attn.mask_add", "mask")

    # attention core
    edge("e.repeat.kt", "attn.repeat_kv", "y", "attn.kv_transposed", "x")
    edge("e.repeat.av", "attn.repeat_kv", "y", "attn.av_matmul", "v")
    edge("e.q.rope.qk", "attn.q_rope", "y", "attn.qk_matmul", "q")
    edge("e.kt.qk", "attn.kv_transposed", "y", "attn.qk_matmul", "k")
    edge("e.qk.scale", "attn.qk_matmul", "y", "attn.scale", "x")
    edge("e.scale.mask", "attn.scale", "y", "attn.mask_add", "x")
    edge("e.mask.add.softmax", "attn.mask_add", "y", "attn.sink_softmax", "x")
    edge("e.sinks.softmax", "attn.sinks", "y", "attn.sink_softmax", "sinks")
    edge("e.softmax.av", "attn.sink_softmax", "y", "attn.av_matmul", "p")
    edge("e.av.transpose", "attn.av_matmul", "y", "attn.out_transpose", "x")
    edge("e.transpose.rope_in", "attn.out_transpose", "y", "attn.rope_in_transpose", "x")
    edge("e.rope_in.rope", "attn.rope_in_transpose", "y", "attn.inv_rope", "x")
    edge("e.ropes.cos.inv", "attn.rope_freqs", "cos", "attn.inv_rope", "cos")
    edge("e.ropes.sin.inv", "attn.rope_freqs", "sin", "attn.inv_rope", "sin")
    edge("e.rope.out_transpose", "attn.inv_rope", "y", "attn.rope_out_transpose", "x")
    edge("e.transpose2.reshape", "attn.rope_out_transpose", "y", "attn.group_reshape", "x")
    edge("e.reshape.o_a", "attn.group_reshape", "y", "attn.o_a_proj", "x")
    edge("e.o_a.o_b", "attn.o_a_proj", "y", "attn.o_b_proj", "x")


def _build_guards(ctx: _Context) -> dict[str, str]:
    b = ctx.builder
    guards: dict[str, str] = {}
    guards["g.compressor"] = b.guard(
        "g.compressor",
        "layer_type != 'sliding_attention'",
        (_line_ref(_MODELING, 794, 796), _line_ref(_MODELING, 745, 749)),
    )
    guards["g.indexer"] = b.guard(
        "g.indexer",
        "layer_type == 'compressed_sparse_attention'",
        (_line_ref(_MODELING, 615, 615), _line_ref(_MODELING, 586, 590)),
    )
    return guards


def _build_state(ctx: _Context) -> dict[str, str]:
    b = ctx.builder
    transitions: dict[str, str] = {}
    transitions["tr.attn.kv"] = b.state_transition(
        "tr.attn.kv", "state.attn.kv", "attn.kv.before", "attn.kv.after"
    )
    if ctx.has_compressor:
        transitions["tr.attn.compressor"] = b.state_transition(
            "tr.attn.compressor",
            "state.attn.compressor",
            "attn.compressor.before",
            "attn.compressor.after",
        )
        if ctx.has_indexer:
            transitions["tr.attn.indexer"] = b.state_transition(
                "tr.attn.indexer",
                "state.attn.indexer",
                "attn.indexer.before",
                "attn.indexer.after",
            )
    return transitions


def _build_aliases(ctx: _Context) -> None:
    tensor_id = "kv_all" if ctx.has_compressor else "kv_full"
    ctx.builder.alias("alias.kv.key", tensor_id, "key")
    ctx.builder.alias("alias.kv.value", tensor_id, "value")


# ---------------------------------------------------------------------------
# 装配与校验
# ---------------------------------------------------------------------------


def _context(values: HfModelValues, block: Mapping[str, Any], group: Mapping[str, Any]) -> _Context:
    attention_type = group["attention_type"]
    has_compressor = attention_type != "sliding_attention"
    has_indexer = attention_type == "compressed_sparse_attention"
    compress_rate = (
        values.compress_rate_csa
        if attention_type == "compressed_sparse_attention"
        else values.compress_rate_hca
    )
    rope_type = "main" if attention_type == "sliding_attention" else "compress"
    template_id = (
        f"{values.model_key}:{group['id']}:layer{block['layer_index']}"
    )
    catalog = _axis_catalog(
        values, has_compressor=has_compressor, has_indexer=has_indexer, compress_rate=compress_rate
    )
    builder = GraphBuilder(template_id, catalog)
    ctx = _Context(
        values=values,
        block=block,
        ops={op["id"]: op for op in block["operators"]},
        tensors={tensor["id"]: tensor for tensor in block["tensors"]},
        group=group,
        layer_type=attention_type,
        has_compressor=has_compressor,
        has_indexer=has_indexer,
        compress_rate=compress_rate,
        rope_type=rope_type,
        builder=builder,
        guards={},
        transitions={},
        kv_len_axis="kv_length_plus_compressed" if has_compressor else "kv_length",
        known=frozenset(
            [f"op:{op_id}" for op_id in (op["id"] for op in block["operators"])]
            + [f"tensor:{tensor_id}" for tensor_id in (t["id"] for t in block["tensors"])]
        ),
    )
    return ctx


def _metadata(ctx: _Context) -> dict[str, Any]:
    values = ctx.values
    divisible = (
        values.num_key_value_heads > 0
        and values.num_attention_heads % values.num_key_value_heads == 0
    )
    attention_ops = tuple(
        sorted(
            op_id
            for op_id in ctx.ops
            if any(op_id == prefix or op_id.startswith(prefix) for prefix in _REQUIRED_COVERED_PREFIXES)
            and not op_id.startswith(("attn_hc_", "ffn_", "moe_", "shared_", "attn_residual_"))
            and op_id not in ("attn_norm",)
        )
    )
    return {
        "model_key": values.model_key,
        "display_name": values.display_name,
        "implementation": values.implementation,
        "layer_key": f"layer.{ctx.block['layer_index']}",
        "layer_index": ctx.block["layer_index"],
        "layer_group": ctx.group["id"],
        "layer_group_label": ctx.group["label"],
        "attention_type": ctx.layer_type,
        "moe_type": ctx.group["moe_type"],
        "rope_type": ctx.rope_type,
        "has_compressor": ctx.has_compressor,
        "has_indexer": ctx.has_indexer,
        "compress_rate": ctx.compress_rate if ctx.has_compressor else None,
        "template_id": ctx.builder.template_id,
        "source_revision": values.source_revision,
        "config_path": _CONFIG_PATH,
        "analysis_path": "eager",
        "shape_facts": [
            ["divisible", "num_attention_heads", "num_key_value_heads"]
        ]
        if divisible
        else [],
        "shape_facts_note": (
            f"num_attention_heads={values.num_attention_heads}，"
            f"num_key_value_heads={values.num_key_value_heads}："
            + ("可整除，广播头数表达式可化简" if divisible else "不可整除，广播头数表达式保留")
        ),
        "legacy_tensor_map": {
            "tensor:attn_normed": "attn_normed",
            "tensor:q_a": "q_a",
            "tensor:q_a_normed": "q_a_normed",
            "tensor:q_normed": "q_normed",
            "tensor:q_rope": "q_rope",
            "tensor:kv_normed": ctx.old_tensor("kv_proj_out", "kv_main") or "kv_main",
            "tensor:kv_head": "kv_head",
            "tensor:kv_rope": "kv_rope",
            "tensor:kv_full": "kv_full",
            "tensor:mask_model": "mask_model",
            "tensor:kv_repeated": "kv_repeated",
            "tensor:qk_logits": "qk_logits",
            "tensor:qk_masked": "qk_masked",
            "tensor:attn_probs": "attn_probs",
            "tensor:av_out_bshd": "attn_head_out",
            "tensor:attn_rope_inv_bshd": "attn_rope_inv",
            "tensor:attn_grouped": "attn_grouped",
            "tensor:o_a_out": "o_a_out",
            "tensor:attn_block_output": "attn_block_output",
            "tensor:attn_out_streams": "attn_out_streams",
            "tensor:hidden_streams_out": "hidden_streams_out",
        },
        "views": {
            "kv": {
                "old_ops": ["kv_proj", "kv_head_view", "kv_rope_apply"],
                "extra_old_ops": [f"rope_{ctx.rope_type}"],
                "start_tensor": "attn_normed",
                "end_tensor": "kv_rope",
                "title": "KV 新投影路径（Attention 接收的 hidden_states → 新 KV）",
            },
            "attention": {
                "old_ops": list(attention_ops),
                "extra_old_ops": ["c_block_bias", "hca_block_bias"],
                "start_tensor": "attn_normed",
                "end_tensor": "attn_block_output",
                "title": "DeepSeek Attention（Q 低秩路径 + 共享 KV + 压缩/索引 + QKᵀ/AV + 输出投影）",
            },
        },
        "focus_notes": [
            "KV 新投影阶段只用 seqlen 表达本次新 token；kvcachelen 只出现在缓存/历史尾部与 Attention 键轴",
            "num_key_value_heads=1 是当前配置固定值，图上显示 1 并在说明中给出参数名",
            "多个 Linear 同时显示实例名与操作类型（如 kv_proj · Linear）",
        ],
    }


def build_semantic_bundle(
    workspace_root: str | Path,
    layer_index: int = LAYER_GROUP_LAYERS["csa-topk"],
    model_key: str = "deepseek_v4_flash_0731_hf",
) -> SemanticBundle:
    """把既有 ``hf_layer_structure.v1`` 事实投影为 raw 语义图（四组层之一）。"""
    values = load_hf_model_values(workspace_root, model_key=model_key)
    document = build_layer_structure(workspace_root, layer_index, model_key)
    block = document["block"]
    groups = {group["id"]: group for group in layer_group_assignments(values)}
    match = None
    for group in groups.values():
        if layer_index in group["layer_indices"]:
            match = group
            break
    if match is None:
        raise SemanticProjectionError(
            f"layer {layer_index} 不属于任何真实层组（{sorted(groups)}）"
        )
    ctx = _context(values, block, match)
    ctx.guards = _build_guards(ctx)
    ctx.transitions = _build_state(ctx)
    _build_layer_nodes(ctx)
    _build_attention_nodes(ctx)
    _build_edges(ctx)
    _build_aliases(ctx)
    bundle = ctx.builder.build(_metadata(ctx))
    issues = [
        issue
        for issue in legacy_coverage(bundle, block)
        if issue.code in ("legacy-op-missing", "legacy-ref-unknown")
    ]
    if issues:
        raise SemanticProjectionError(
            "旧事实覆盖/引用不完整：" + "; ".join(issue.message for issue in issues)
        )
    return bundle


def build_all_layer_group_bundles(
    workspace_root: str | Path,
    model_key: str = "deepseek_v4_flash_0731_hf",
) -> dict[str, SemanticBundle]:
    """四组真实层各自一份 raw 语义图（按 group id 排序返回）。"""
    values = load_hf_model_values(workspace_root, model_key=model_key)
    groups = {group["id"]: group for group in layer_group_assignments(values)}
    bundles: dict[str, SemanticBundle] = {}
    for group_id in sorted(groups):
        representative = groups[group_id]["representative_layer_index"]
        bundles[group_id] = build_semantic_bundle(
            workspace_root, representative, model_key=model_key
        )
    return bundles


def legacy_coverage(bundle: SemanticBundle, block: Mapping[str, Any]) -> tuple[BundleIssue, ...]:
    """旧事实覆盖校验：每个既有算子都被语义节点覆盖、每条旧引用真实存在、秩一致。"""
    issues: list[BundleIssue] = []
    known_ops = {f"op:{op['id']}" for op in block["operators"]}
    known_tensors = {f"tensor:{tensor['id']}" for tensor in block["tensors"]}
    known = known_ops | known_tensors
    covered_ops: set[str] = set()
    for node_id, meta in bundle.node_meta.items():
        for ref in meta.legacy_refs:
            if ref not in known:
                issues.append(
                    BundleIssue(
                        code="legacy-ref-unknown",
                        path=f"node_meta[{node_id}].legacy_refs",
                        message=f"旧事实映射 {ref!r} 在 block 事实中不存在",
                    )
                )
            elif ref.startswith("op:"):
                covered_ops.add(ref)
    for op in block["operators"]:
        if f"op:{op['id']}" not in covered_ops:
            issues.append(
                BundleIssue(
                    code="legacy-op-missing",
                    path=f"operators[{op['id']}]",
                    message=f"既有算子 {op['id']} 没有被任何语义节点覆盖",
                )
            )
    # 秩一致：语义张量与旧张量的映射（metadata.legacy_tensor_map）
    tensor_map = bundle.metadata.get("legacy_tensor_map") or {}
    semantic_tensors = bundle.tensors
    old_tensors = {tensor["id"]: tensor for tensor in block["tensors"]}
    for semantic_id, old_id in tensor_map.items():
        if semantic_id not in semantic_tensors or old_id not in old_tensors:
            issues.append(
                BundleIssue(
                    code="legacy-tensor-map-unknown",
                    path=f"metadata.legacy_tensor_map[{semantic_id}]",
                    message=f"张量映射 {semantic_id} → {old_id} 的任一端不存在",
                )
            )
            continue
        if len(semantic_tensors[semantic_id].axes) != len(old_tensors[old_id]["shape"]):
            issues.append(
                BundleIssue(
                    code="legacy-tensor-rank-mismatch",
                    path=f"tensors[{semantic_id}]",
                    message=(
                        f"秩不一致：语义 {len(semantic_tensors[semantic_id].axes)} 维 vs "
                        f"旧事实 {len(old_tensors[old_id]['shape'])} 维"
                    ),
                )
            )
    return tuple(issues)


def semantic_shape_facts(bundle: SemanticBundle) -> tuple[tuple, ...]:
    """从 bundle 元数据取出可判定形状事实（供化简规则使用）。"""
    records = bundle.metadata.get("shape_facts") or []
    return tuple(tuple(record) for record in records)


# ---------------------------------------------------------------------------
# FFN 具名计算投影（T13 返修）：共享专家三个 Linear + 专家模板的融合参数身份
# ---------------------------------------------------------------------------
#
# 旧事实把共享专家写成 `linear+clamp×2` / `silu+mul+linear` 两个复合算子，FFN 层图
# 只能显示 DeepseekV4MLP 复合框。这里从**同一批旧事实**（算子标记、权重张量名/形状、
# 源码行）展开出具名 Linear；条件不成立时保留复合算子并给拒绝原因，不猜模块。
# 展开不修改 block 的 operators/tensors，因此 Excel 粒度、公式与字节不变。


@dataclass(frozen=True)
class NamedLinearStep:
    """一条具名计算步骤（共享专家 Linear / clamp / SiLU·mul，或专家模板 Linear）。"""

    step_id: str
    label: str
    op_type: str
    module_path: str
    sources: tuple[SourceRef, ...]
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[tuple[str, str], ...]
    weight: Mapping[str, Any] | None = None
    note: str = ""
    # 该步骤产生的中间张量（同源派生：名称/形状/来源；形状由输入与权重推导，不抄样例常数）。
    tensor_facts: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class FfnExpansion:
    """一个复合 FFN 算子的展开结果：applied 或 rejected（附原因）。"""

    op_id: str
    rule_id: str
    status: str  # applied | rejected
    reason: str
    steps: tuple[NamedLinearStep, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "rule_id": self.rule_id,
            "status": self.status,
            "reason": self.reason,
            "steps": [
                {
                    "step_id": step.step_id,
                    "label": step.label,
                    "op_type": step.op_type,
                    "module_path": step.module_path,
                    "sources": [source.location for source in step.sources],
                    "inputs": list(step.inputs),
                    "outputs": list(step.outputs),
                    "weight": dict(step.weight) if step.weight else None,
                    "note": step.note,
                    "tensor_facts": [dict(fact) for fact in step.tensor_facts],
                }
                for step in self.steps
            ],
        }


def _layer_module_prefix(block: Mapping[str, Any]) -> str:
    index = block.get("layer_index")
    if index is None:
        raise SemanticProjectionError("block 缺少 layer_index，无法给出真实模块路径")
    return f"model.layers.{index}.mlp"


def _weight_of(block: Mapping[str, Any], tensor_id: str) -> Mapping[str, Any] | None:
    for tensor in block.get("tensors", ()):
        if tensor.get("id") == tensor_id:
            return tensor
    return None


def _weight_steps(
    block: Mapping[str, Any], op: Mapping[str, Any], suffix: str
) -> Mapping[str, Any] | None:
    for weight_id in op.get("weights", ()):
        tensor = _weight_of(block, weight_id)
        if tensor is None:
            continue
        name = str(tensor.get("name") or "")
        if name.endswith(suffix):
            return tensor
    return None


def _tensor_shape(block: Mapping[str, Any], tensor_id: str) -> tuple[str, ...]:
    tensor = _weight_of(block, tensor_id)
    return tuple(tensor.get("shape", ())) if tensor else ()


def _tensor_alias_shape(block: Mapping[str, Any], tensor_id: str) -> tuple[str, ...]:
    tensor = _weight_of(block, tensor_id)
    if tensor is None:
        raise SemanticProjectionError(f"既有事实缺少张量 {tensor_id!r}，无法派生形状")
    return tuple(str(axis) for axis in tensor.get("shape", ()))


def _linear_output_shape(
    input_shape: tuple[str, ...], weight_shape: tuple[str, ...]
) -> tuple[tuple[str, ...] | None, str]:
    """nn.Linear 输出形状 = 输入前缀 + 权重输出维（权重视为 [out, in]）。

    无法证明时返回 (None, 原因)，由调用方显式标注未知，不猜常量。
    """
    if not input_shape:
        return None, "输入张量缺少形状，无法派生 Linear 输出"
    if len(weight_shape) != 2:
        return None, f"权重形状 {list(weight_shape)} 不是二维 [out,in]，无法派生 Linear 输出"
    return tuple(input_shape[:-1]) + (weight_shape[0],), ""


def ffn_named_linear_steps(block: Mapping[str, Any]) -> tuple[FfnExpansion, ...]:
    """把 FFN 复合算子展开为具名计算；每条规则有成立条件与拒绝原因。

    R-FFN-1 共享专家 gate/up：op 标记为 `linear+clamp×2`，且权重张量真实名称分别以
      `.gate_proj.weight` / `.up_proj.weight` 结尾 → 两个 Linear + 一个双端 clamp。
    R-FFN-2 共享专家 down：op 标记包含 `silu+mul+linear`，权重以 `.down_proj.weight`
      结尾 → SiLU、SiLU·mul、down Linear。
    R-FFN-3 专家模板融合参数：op 标记为 `linear` 且权重名是 `gate_up_proj[...]` /
      `down_proj[...]` → 仅改名为真实参数身份（不虚构每专家实例）。
    不满足条件的复合算子原样保留（rejected + reason），不猜测模块或权重。
    """
    prefix = _layer_module_prefix(block)
    expansions: list[FfnExpansion] = []
    for op in block.get("operators", ()):
        op_id = str(op.get("id"))
        module_path = str(op.get("module_path") or "")
        if not (
            op_id.startswith("moe_")
            or op_id.startswith("shared_")
            or "MLP" in module_path
            or "Experts" in module_path
        ):
            continue
        marker = str(op.get("op") or "")
        weights = [w for w in (op.get("weights") or ())]
        sources = _op_sources(op)
        if marker.replace(" ", "") in ("linear+clamp×2", "linear+clampx2"):
            gate = _weight_steps(block, op, ".gate_proj.weight")
            up = _weight_steps(block, op, ".up_proj.weight")
            if gate is None or up is None:
                expansions.append(
                    FfnExpansion(
                        op_id=op_id,
                        rule_id="R-FFN-1",
                        status="rejected",
                        reason=(
                            "权重张量名不是 gate_proj.weight/up_proj.weight，"
                            "不能证明是两个具名 Linear；保留复合算子"
                        ),
                    )
                )
                continue
            gate_name = str(gate["name"]).removesuffix(".weight")
            up_name = str(up["name"]).removesuffix(".weight")
            gate_shape = tuple(gate.get("shape", ()))
            up_shape = tuple(up.get("shape", ()))
            inputs = tuple(op.get("inputs", ()))
            outputs = tuple(op.get("outputs", ()))
            if len(inputs) != 1 or len(outputs) < 2:
                expansions.append(
                    FfnExpansion(
                        op_id=op_id,
                        rule_id="R-FFN-1",
                        status="rejected",
                        reason="复合算子的输入/输出数量与 gate/up 双投影不符；保留复合算子",
                    )
                )
                continue
            gate_out = f"sem:{op_id}.gate_proj"
            up_out = f"sem:{op_id}.up_proj"
            input_shape = _tensor_alias_shape(block, str(inputs[0]))
            gate_shape, gate_reason = _linear_output_shape(input_shape, gate_shape)
            up_shape, up_reason = _linear_output_shape(input_shape, up_shape)
            gate_source = [source.location for source in sources[:3]]
            gate_fact = {
                "id": gate_out,
                "name": f"{gate_name}(x)",
                "shape": list(gate_shape) if gate_shape else [],
                "source": gate_source,
                "reason": gate_reason,
                "derivation": f"nn.Linear 输出 = 输入前缀 + 权重输出维（{gate['name']}）",
            }
            up_fact = {
                "id": up_out,
                "name": f"{up_name}(x)",
                "shape": list(up_shape) if up_shape else [],
                "source": gate_source,
                "reason": up_reason,
                "derivation": f"nn.Linear 输出 = 输入前缀 + 权重输出维（{up['name']}）",
            }
            steps = (
                NamedLinearStep(
                    step_id=f"{op_id}.gate_proj",
                    label="gate_proj · Linear",
                    op_type="Linear",
                    module_path=f"{prefix}.shared_experts.gate_proj",
                    sources=sources,
                    inputs=((inputs[0], str(inputs[0])),),
                    outputs=(("y", gate_out),),
                    weight={
                        "tensor_id": gate["id"],
                        "name": gate["name"],
                        "shape": list(gate.get("shape", ())),
                    },
                    note=(
                        f"nn.Linear(hidden_size → intermediate_size)，weight {gate['name']} "
                        f"{list(gate.get('shape', ()))}；输出张量 {gate_out}（{gate_fact['derivation']}）"
                    ),
                    tensor_facts=(gate_fact,),
                ),
                NamedLinearStep(
                    step_id=f"{op_id}.up_proj",
                    label="up_proj · Linear",
                    op_type="Linear",
                    module_path=f"{prefix}.shared_experts.up_proj",
                    sources=sources,
                    inputs=((inputs[0], str(inputs[0])),),
                    outputs=(("y", up_out),),
                    weight={
                        "tensor_id": up["id"],
                        "name": up["name"],
                        "shape": list(up.get("shape", ())),
                    },
                    note=(
                        f"nn.Linear(hidden_size → intermediate_size)，weight {up['name']} "
                        f"{list(up.get('shape', ()))}；输出张量 {up_out}（{up_fact['derivation']}）"
                    ),
                    tensor_facts=(up_fact,),
                ),
                NamedLinearStep(
                    step_id=f"{op_id}.clamp",
                    label="clamp · 双端限幅",
                    op_type="clamp",
                    module_path=f"{prefix}.shared_experts",
                    sources=sources,
                    inputs=((gate_out, gate_out), (up_out, up_out)),
                    outputs=((str(outputs[0]), str(outputs[0])), (str(outputs[1]), str(outputs[1]))),
                    note=(
                        "gate=gate_proj(x).clamp(max=swiglu_limit)；"
                        "up=up_proj(x).clamp(min=-swiglu_limit, max=swiglu_limit)"
                    ),
                ),
            )
            expansions.append(
                FfnExpansion(
                    op_id=op_id,
                    rule_id="R-FFN-1",
                    status="applied",
                    reason=(
                        f"权重名 {gate_name}.weight / {up_name}.weight 与源码 "
                        "DeepseekV4MLP.gate_proj/up_proj 一致；按源码行拆成两个 Linear + clamp"
                    ),
                    steps=steps,
                )
            )
            continue
        if "silu+mul+linear" in marker:
            down = _weight_steps(block, op, ".down_proj.weight")
            if down is None:
                expansions.append(
                    FfnExpansion(
                        op_id=op_id,
                        rule_id="R-FFN-2",
                        status="rejected",
                        reason="权重名不是 down_proj.weight，不能证明具名 down Linear；保留复合算子",
                    )
                )
                continue
            inputs = tuple(op.get("inputs", ()))
            outputs = tuple(op.get("outputs", ()))
            if len(inputs) != 2 or len(outputs) < 2:
                expansions.append(
                    FfnExpansion(
                        op_id=op_id,
                        rule_id="R-FFN-2",
                        status="rejected",
                        reason="输入/输出数量与 SiLU·mul+down 不符；保留复合算子",
                    )
                )
                continue
            silu_out = f"sem:{op_id}.silu"
            silu_input_shape = _tensor_alias_shape(block, str(inputs[0]))
            silu_source = [source.location for source in sources[:3]]
            silu_fact = {
                "id": silu_out,
                "name": f"{str(inputs[0])} 的 SiLU",
                "shape": list(silu_input_shape),
                "source": silu_source,
                "reason": "",
                "derivation": "激活逐元素，形状与输入一致（SiLU 不改变形状）",
            }
            steps = (
                NamedLinearStep(
                    step_id=f"{op_id}.silu",
                    label="SiLU · 激活",
                    op_type="activation",
                    module_path=f"{prefix}.shared_experts",
                    sources=sources,
                    inputs=((str(inputs[0]), str(inputs[0])),),
                    outputs=(("y", silu_out),),
                    note=(
                        f"act_fn(gate)，gate 已 clamp；输出张量 {silu_out}"
                        "（激活保持输入形状）"
                    ),
                    tensor_facts=(silu_fact,),
                ),
                NamedLinearStep(
                    step_id=f"{op_id}.mul",
                    label="mul · SiLU(gate)×up",
                    op_type="mul",
                    module_path=f"{prefix}.shared_experts",
                    sources=sources,
                    inputs=((silu_out, silu_out), (str(inputs[1]), str(inputs[1]))),
                    outputs=(("y", str(outputs[0])),),
                    note=f"逐元素相乘，输出 {outputs[0]}",
                ),
                NamedLinearStep(
                    step_id=f"{op_id}.down_proj",
                    label="down_proj · Linear",
                    op_type="Linear",
                    module_path=f"{prefix}.shared_experts.down_proj",
                    sources=sources,
                    inputs=((str(outputs[0]), str(outputs[0])),),
                    outputs=(("y", str(outputs[1])),),
                    weight={
                        "tensor_id": down["id"],
                        "name": down["name"],
                        "shape": list(down.get("shape", ())),
                    },
                    note=(
                        f"nn.Linear(intermediate_size → hidden_size)，weight {down['name']} "
                        f"{list(down.get('shape', ()))}"
                    ),
                ),
            )
            expansions.append(
                FfnExpansion(
                    op_id=op_id,
                    rule_id="R-FFN-2",
                    status="applied",
                    reason=(
                        f"权重名 {down['name']} 与源码 DeepseekV4MLP.down_proj 一致；"
                        "按源码行拆成 SiLU·mul 与具名 down Linear"
                    ),
                    steps=steps,
                )
            )
            continue
        if marker.strip() == "linear" and weights:
            weight = _weight_of(block, str(weights[0]))
            name = str(weight.get("name") or "") if weight else ""
            if name.startswith("gate_up_proj[") or name.startswith("down_proj["):
                expansions.append(
                    FfnExpansion(
                        op_id=op_id,
                        rule_id="R-FFN-3",
                        status="applied",
                        reason=(
                            f"专家模板的 F.linear 使用融合参数 {name}：按参数身份命名，"
                            "不虚构每专家实例"
                        ),
                        steps=(
                            NamedLinearStep(
                                step_id=f"{op_id}.linear",
                                label=(
                                    "gate_up_proj · Linear" if name.startswith("gate_up_proj")
                                    else "down_proj · Linear"
                                ),
                                op_type="Linear",
                                module_path=f"{prefix}.experts.{name.split('[')[0]}",
                                sources=sources,
                                inputs=tuple(
                                    (str(t), str(t)) for t in op.get("inputs", ())
                                ),
                                outputs=tuple(
                                    (str(t), str(t)) for t in op.get("outputs", ())
                                ),
                                weight={
                                    "tensor_id": weight["id"],
                                    "name": weight["name"],
                                    "shape": list(weight.get("shape", ())),
                                },
                                note=(
                                    "F.linear 单专家切片（融合参数，第 e 个专家）；"
                                    "专家命中与每专家 token 数仍为运行期未知量"
                                ),
                            ),
                        ),
                    )
                )
                continue
        if marker in ("linear", "linear+softplus+topk", "gather+greater+scatter"):
            expansions.append(
                FfnExpansion(
                    op_id=op_id,
                    rule_id="R-FFN-4",
                    status="rejected",
                    reason="权重身份不满足 R-FFN-1/2/3 的成立条件；保留原算子",
                )
            )
    return tuple(expansions)


def ffn_tensor_facts(block: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """FFN 展开登记的中间张量事实（同源派生：名称/形状/来源/推导说明）。

    供构建层与证据脚本核对“新张量已在语义投影中登记”，不复制常量。
    """
    facts: dict[str, dict[str, Any]] = {}
    for item in ffn_named_linear_steps(block):
        if item.status != "applied":
            continue
        for step in item.steps:
            for fact in step.tensor_facts:
                facts[str(fact["id"])] = {
                    "id": fact["id"],
                    "name": fact["name"],
                    "shape": list(fact.get("shape", ())),
                    "sources": list(fact.get("source", ())),
                    "derivation": fact.get("derivation", ""),
                    "reason": fact.get("reason", ""),
                    "step": step.step_id,
                }
    return facts
