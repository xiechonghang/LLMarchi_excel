"""产品分层 Archify 图的模型声明：只声明「哪些模块/来源属于该模型」，不复制构建器。

每个模型声明：

- 显示名与源码仓库（URL/revision/本地检出根/路径前缀）；
- 层组/Block/子图说明文案（进入页面时显示）；
- 子图分段规则：层键、页面标题、真实模块 id（来自同模型 ``PageProfile`` 的
  ``module_groups``）、是否同时交付 raw/normalized、可选的入口/出口张量；
- 整网辅助算子（位置编码、mask、RoPE 等）前缀：在整网图中作为短旁枝。

``render/archify_layers.py`` 用这些声明驱动**同一套**图构建/布局/身份代码；模型
差异不落到另一套构建器或手抄事实。模块 id 与 ``render/hf_model_profile.py`` 的
分组规则一一对应；如果模型 profile 改了分组，这里的 id 缺失会当场报错，不会静默
退化成一张“其他”图。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "GraphRepository",
    "SegmentSpec",
    "ModelGraphProfile",
    "MODEL_GRAPH_PROFILES",
    "graph_profile_for",
]


@dataclass(frozen=True)
class GraphRepository:
    """Archify 来源证据：源码检出根 + 仓库身份（相对工作区根）。"""

    url: str
    revision: str
    root_rel: str
    path_prefix: str
    link_mode: str = "local-only"

    def as_finalize(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "revision": self.revision,
            "link_mode": self.link_mode,
        }


@dataclass(frozen=True)
class SegmentSpec:
    """一个 Block 内部子图：真实模块集合 + 页面入口文案。"""

    level: str
    page_label: str
    title: str
    modules: tuple[str, ...]
    note: str = ""
    pair_raw: bool = False
    entry: str | None = None
    exit: str | None = None

    def to_request(self, variant: str, form: str = "normalized") -> dict[str, Any]:
        return {
            "level": self.level,
            "variant": variant,
            "form": form,
            "title": self.title,
        }


@dataclass(frozen=True)
class ModelGraphProfile:
    model_key: str
    display: str
    repository: GraphRepository
    network_title: str
    network_side_prefixes: tuple[str, ...]
    group_note: str
    block_note: str
    segments: Callable[[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]],
                       list[SegmentSpec]]
    levels: tuple[str, ...]
    # 整网阶段（label, tag, 真实 op ids）：只用于把很长的单次 forward 收成入口级
    # 阶段，细节仍逐 op 列在节点详情并由层组/Block 图展开。空表示逐 op 出图。
    network_phases: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    build_network_title: Callable[[Mapping[str, Any]], str] | None = None
    # False：源码检出不是 git top-level（HiDream 官方实现是嵌套目录），不声明
    # Archify 仓库证据；来源以 reports/source_manifest.json + 独立 sources JSON 核对。
    repository_evidence: bool = True


# ---------------------------------------------------------------------------
# 分段助手
# ---------------------------------------------------------------------------


def _ids(modules: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(module["id"]): module for module in modules}


def _present(modules: Sequence[Mapping[str, Any]], *wanted: str) -> tuple[str, ...]:
    available = _ids(modules)
    return tuple(name for name in wanted if name in available)


def _attention_modules(
    modules: Sequence[Mapping[str, Any]], *candidates: str
) -> tuple[str, ...]:
    present = _present(modules, *candidates)
    if not present:
        raise ValueError(
            "模型声明与 PageProfile 模块分组不一致：以下候选模块都不存在 "
            f"{candidates}"
        )
    return present


def _level_title(display: str, group: Mapping[str, Any], suffix: str) -> str:
    label = group.get("label") or group.get("id")
    return f"{display} · {label} — {suffix}"


def _kind(block: Mapping[str, Any], group: Mapping[str, Any], key: str) -> str:
    value = group.get(key)
    if value:
        return str(value)
    scope = (block.get("scope") or {})
    return str(scope.get(key, ""))


# ---------------------------------------------------------------------------
# GLM-5.3（glm_moe_dsa）：共享/独立 DSA indexer
# ---------------------------------------------------------------------------


def _glm53_segments(block, group, modules):
    present = _ids(modules)
    specs: list[SegmentSpec] = []
    full_indexer = "glm_attn_indexer" in present
    attention = _attention_modules(
        modules,
        "glm_attn_norm", "glm_attn_q", "glm_attn_kv", "glm_attn_mask",
        "glm_attn_core", "glm_attn_out", "glm_res_attn",
    )
    specs.append(
        SegmentSpec(
            level="attention",
            page_label="Attention 内部（Q/压缩 KV → indexer 稀疏 mask → eager 核心 → o_proj）",
            title=_level_title(
                "GLM-5.3", group,
                "Attention 内部计算（full DSA indexer）" if full_indexer
                else "Attention 内部计算（shared indexer：复用上一 full 层 top-k）",
            ),
            modules=attention,
            note=(
                "Q 支路、压缩 KV 支路与 DSA Indexer 并行；indexer 的 top-k 只经稀疏"
                " mask 进入注意力核心，不改变主计算方向。shared 层不实例化 indexer，"
                "prev_topk_indices 作为边界输入。"
            ),
            pair_raw=True,
        )
    )
    if full_indexer:
        specs.append(
            SegmentSpec(
                level="indexer",
                page_label="DSA Indexer（full 层：打分 / top-k → index_mask）",
                title=_level_title("GLM-5.3", group, "DSA Indexer 打分与 top-k"),
                modules=_present(
                    modules, "glm_attn_indexer", "glm_attn_mask"),
                note=(
                    "indexer 是独立打分器：q/k 低秩投影、k_norm、键缓存、分数 relu、"
                    "top-k；top-k 选中位置在 mask 中置为可见。"
                ),
            )
        )
    kv = _present(modules, "glm_attn_kv", "glm_attn_cache")
    if kv:
        specs.append(
            SegmentSpec(
                level="kv",
                page_label="压缩 KV 支路与持久缓存（latent 写入 + kv_b 展开）",
                title=_level_title("GLM-5.3", group, "压缩 KV 与持久缓存"),
                modules=kv,
                note=(
                    "缓存保存 kv_a_layernorm 后的压缩 latent 与 RoPE k_rot，不是展开后的"
                    " K；kv_b 展开后与 k_rot 拼成注意力 K。"
                ),
            )
        )
    if "glm_mlp_dense" in present:
        ffn = _present(
            modules, "glm_norm_ffn", "glm_mlp_dense", "glm_res_ffn")
        suffix, kind_note = "Dense FFN（gate/up/down）", "dense"
    else:
        ffn = _present(
            modules, "glm_norm_ffn", "glm_moe_route", "glm_moe_expert",
            "glm_moe_shared", "glm_moe_merge", "glm_res_ffn")
        suffix, kind_note = "MoE（路由 / 单专家模板 / 共享专家 / 汇合）", "sparse"
    if ffn:
        specs.append(
            SegmentSpec(
                level="ffn",
                page_label="FFN / MoE 内部计算（专家模板单份，不复制 E 份）",
                title=_level_title("GLM-5.3", group, suffix),
                modules=ffn,
                note=(
                    f"moe_type={kind_note}；专家模板只画一份，路由命中数保持数据相关；"
                    "共享专家是并行支路，汇合后进入 FFN 残差。"
                ),
            )
        )
    return specs


# ---------------------------------------------------------------------------
# GLM-5.3-Flash（glm5_next）：KDA / DSA 混合 + mHC
# ---------------------------------------------------------------------------


def _flash_segments(block, group, modules):
    present = _ids(modules)
    specs: list[SegmentSpec] = []
    kind = str(block.get("kind") or "")
    if kind == "kda":
        attention = _attention_modules(
            modules, "attn_norm", "kda_proj", "kda_conv", "kda_gate",
            "kda_scan", "kda_out", "res_attn")
        specs.append(
            SegmentSpec(
                level="attention",
                page_label="KDA 线性注意力（QKV → 因果短卷积 → delta-rule 扫描 → 门控输出）",
                title=_level_title("GLM-5.3-Flash", group, "KDA 线性注意力内部计算"),
                modules=attention,
                note=(
                    "KDA 没有 softmax 注意力矩阵；conv state 与 recurrent state 是两层"
                    "固定大小状态，按调用前/后分开。"
                ),
                pair_raw=True,
            )
        )
        state = _present(modules, "kda_conv", "kda_scan")
        if state:
            specs.append(
                SegmentSpec(
                    level="kv",
                    page_label="KDA 跨调用状态（conv state / recurrent state 前态后态）",
                    title=_level_title("GLM-5.3-Flash", group, "KDA 跨调用状态"),
                    modules=state,
                    note=(
                        "conv state 长度固定 linear_conv_kernel_dim；recurrent state [batch_size,linear_num_heads,linear_head_dim,linear_head_dim] 固定，"
                        "不随历史长度增长；两者都不是 KV cache。"
                    ),
                )
            )
    else:
        attention = _attention_modules(
            modules, "attn_norm", "dsa_q", "dsa_kv", "dsa_cache", "dsa_core",
            "dsa_out", "res_attn")
        specs.append(
            SegmentSpec(
                level="attention",
                page_label="DSA 稀疏注意力（MLA Q / 压缩 KV / eager 核心 → o_proj）",
                title=_level_title("GLM-5.3-Flash", group, "DSA 注意力内部计算"),
                modules=attention,
                note=(
                    "本模型 NoPE（Rk=0），转置/单位轴表达按源码保留；主 KV 缓存保存"
                    "压缩 latent，indexer 缓存另存 packed 键。"
                ),
                pair_raw=True,
            )
        )
        kv = _present(modules, "dsa_kv", "dsa_cache")
        if kv:
            specs.append(
                SegmentSpec(
                    level="kv",
                    page_label="主 KV 缓存（压缩 latent 前态/后态 + kv_b 展开）",
                    title=_level_title("GLM-5.3-Flash", group, "主 KV 缓存"),
                    modules=kv,
                    note="压缩 latent 追加当前 seqlen 后写回；缓存前态/后态分开表达。",
                )
            )
        indexer = _present(
            modules, "dsa_indexer", "dsa_pool", "dsa_select", "dsa_mask")
        if indexer:
            specs.append(
                SegmentSpec(
                    level="indexer",
                    page_label="DSA Indexer（packed 缓存 / k-pool 池化 / 池 top-k / 可见 mask）",
                    title=_level_title("GLM-5.3-Flash", group, "DSA Indexer 与池选择"),
                    modules=indexer,
                    note=(
                        "packed 缓存是 k(128)+gate(128)+valid(1)；池内均值打分、池 top-k"
                        " 展开后再与 tail 拼接，超出预算部分填 -1。"
                    ),
                )
            )
    mhc = _present(modules, "mhc_attn", "res_attn", "mhc_ffn", "res_ffn")
    if mhc:
        specs.append(
            SegmentSpec(
                level="mhc",
                page_label="mHC 超连接与两段残差（坍缩 → pre/post/comb → 写回）",
                title=_level_title("GLM-5.3-Flash", group, "mHC 多流混合与残差"),
                modules=mhc,
                note=(
                    "4 路残差流在 attn/ffn 两个站点各自坍缩、生成 pre/post/comb 系数，"
                    "再展开写回；这是真实数据依赖，不是排版回流。"
                ),
            )
        )
    if "mlp_dense" in present:
        ffn = _present(modules, "norm_ffn", "mlp_dense", "res_ffn")
        suffix, kind_note = "Dense MLP（SwiGLU clamp）", "dense"
    else:
        ffn = _present(
            modules, "norm_ffn", "moe_route", "moe_expert", "moe_shared",
            "moe_merge", "res_ffn")
        suffix, kind_note = "MoE（路由 / 单专家模板 / 共享专家 / 汇合）", "sparse"
    if ffn:
        specs.append(
            SegmentSpec(
                level="ffn",
                page_label="FFN / MoE 内部计算（专家模板单份）",
                title=_level_title("GLM-5.3-Flash", group, suffix),
                modules=ffn,
                note=f"moe_type={kind_note}；路由命中数保持数据相关，不推断真实专家。",
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Qwen3.8-Flash-Next（qwen4_exp）：GDN / QSA + Gated Residual + PLE
# ---------------------------------------------------------------------------


def _qwen_segments(block, group, modules):
    present = _ids(modules)
    specs: list[SegmentSpec] = []
    kind = str(block.get("kind") or "")
    if kind == "gdn":
        attention = _attention_modules(
            modules, "gdn_proj", "gdn_conv", "gdn_gate", "gdn_scan",
            "gdn_out", "res_attn")
        specs.append(
            SegmentSpec(
                level="attention",
                page_label="GDN 线性注意力（QKV/z/beta/decay → 短卷积 → gated delta-rule → 门控输出）",
                title=_level_title("Qwen3.8-Flash-Next", group, "GDN 线性注意力内部计算"),
                modules=attention,
                note=(
                    "GDN 没有 softmax 注意力矩阵；conv state 与 recurrent state 固定"
                    "大小，按调用前/后分开。"
                ),
                pair_raw=True,
            )
        )
        state = _present(modules, "gdn_conv", "gdn_scan")
        if state:
            specs.append(
                SegmentSpec(
                    level="kv",
                    page_label="GDN 跨调用状态（conv state / recurrent state 前态后态）",
                    title=_level_title("Qwen3.8-Flash-Next", group, "GDN 跨调用状态"),
                    modules=state,
                    note="conv state 固定 linear_conv_kernel_dim；recurrent state [batch_size,linear_num_value_heads,linear_key_head_dim,linear_value_head_dim] 固定。",
                )
            )
    else:
        attention = _attention_modules(
            modules, "qsa_q", "qsa_kv", "qsa_rope", "qsa_cache",
            "qsa_core", "qsa_out", "res_attn")
        specs.append(
            SegmentSpec(
                level="attention",
                page_label="QSA 稀疏注意力（Q/K/V → partial RoPE → 块 top-k mask → eager 核心 → 门控 o_proj）",
                title=_level_title("Qwen3.8-Flash-Next", group, "QSA 注意力内部计算"),
                modules=attention,
                note=(
                    "QSA 的 indexer 按 micro-block 均值池化后选块；mask bool→additive 与"
                    " causal mask 合并，再进入 eager 核心。"
                ),
                pair_raw=True,
            )
        )
        kv = _present(modules, "qsa_kv", "qsa_cache")
        if kv:
            specs.append(
                SegmentSpec(
                    level="kv",
                    page_label="主 K/V 缓存（随历史线性增长，前态/后态分开）",
                    title=_level_title("Qwen3.8-Flash-Next", group, "主 K/V 缓存"),
                    modules=kv,
                    note="K/V 缓存 [batch_size,num_key_value_heads,kv_sequence_length,head_dim]；更新线性追加当前 seqlen。",
                )
            )
        indexer = _present(
            modules, "qsa_indexer", "qsa_pool", "qsa_select", "qsa_mask")
        if indexer:
            specs.append(
                SegmentSpec(
                    level="indexer",
                    page_label="QSA Indexer（逐 token key 缓存 / micro-block 池化 / 块 top-k / mask）",
                    title=_level_title("Qwen3.8-Flash-Next", group, "QSA Indexer 与块选择"),
                    modules=indexer,
                    note=(
                        "indexer key 是逐 token 缓存 [batch_size,indexer_kv_sequence_length,indexer_head_dim]，不是 pooled block key；"
                        "无完整块的 query 走 tail 分支。"
                    ),
                )
            )
    ple = _present(modules, "ple_ngram", "ple_gate")
    if ple:
        specs.append(
            SegmentSpec(
                level="ple",
                page_label="N-gram PLE（上下文 / 哈希查表 / 门控 / 膨胀短卷积 / 加回）",
                title=_level_title("Qwen3.8-Flash-Next", group, "N-gram PLE 与门控残差"),
                modules=ple,
                note=(
                    "PLE 只在第 2 层（index 1）执行；上下文 [batch_size,ngram_context_len] 与短卷积 state 都是"
                    "跨调用状态，按前态/后态表达。"
                ),
            )
        )
    hc = _present(modules, "mhc_attn", "res_attn", "mhc_ffn", "res_ffn")
    if hc:
        specs.append(
            SegmentSpec(
                level="mhc",
                page_label="Gated Residual（读门 [hc_count,hidden_size] + 写门 [hc_count]，attn/ffn 两段）",
                title=_level_title("Qwen3.8-Flash-Next", group, "门控残差与两段写回"),
                modules=hc,
                note=(
                    "4 路流在 attn/ffn 站点各自读门混合、生成写门并写回；门控是真实"
                    "数据依赖，不是排版回流。"
                ),
            )
        )
    ffn = _present(
        modules, "mhc_ffn", "moe_route", "moe_expert", "moe_shared",
        "moe_merge", "res_ffn")
    # mhc_ffn 已在 HC 子图；FFN 子图从 post-norm 之后开始。
    ffn = tuple(name for name in ffn if name != "mhc_ffn")
    if ffn:
        specs.append(
            SegmentSpec(
                level="ffn",
                page_label="FFN / MoE 内部计算（路由 / 单专家模板 / 共享专家 / 汇合）",
                title=_level_title("Qwen3.8-Flash-Next", group, "FFN / MoE 内部计算"),
                modules=ffn,
                note=(
                    "路由 softmax 512 选 10；专家模板单份，共享专家 640 宽 + sigmoid"
                    " 标量门。"
                ),
            )
        )
    return specs


# ---------------------------------------------------------------------------
# HiDream-O1-Image（单次生成网络）：UiT 主干 + 条件 ViT
# ---------------------------------------------------------------------------


def _hidream_segments(block, group, modules):
    present = _ids(modules)
    kind = str(block.get("kind") or "")
    specs: list[SegmentSpec] = []
    if kind == "uit_decoder":
        attention = _present(
            modules, "d_norm_attn", "d_attn_q", "d_attn_k", "d_attn_v",
            "d_attn_ar", "d_attn_full", "d_attn_merge", "d_attn_out",
            "d_res_attn")
        specs.append(
            SegmentSpec(
                level="attention",
                page_label="两趟注意力（AR 子集因果 + 全序列双向 → 合并 → o_proj）",
                title=_level_title("HiDream-O1-Image", group, "两趟注意力内部计算"),
                modules=attention,
                note=(
                    "同一层内两趟 flash 注意力：AR 行只对 AR 子集因果，生成行对全序列"
                    "双向；index_copy 用 AR 行覆盖第 1 趟结果。"
                ),
                pair_raw=True,
            )
        )
        ffn = _present(modules, "d_norm_ffn", "d_ffn_mlp", "d_res_ffn")
        if ffn:
            specs.append(
                SegmentSpec(
                    level="ffn",
                    page_label="SwiGLU MLP（gate/up 12288 → SiLU×up → down）",
                    title=_level_title("HiDream-O1-Image", group, "SwiGLU FFN"),
                    modules=ffn,
                )
            )
        deepstack = _present(modules, "d_deepstack")
        if deepstack:
            specs.append(
                SegmentSpec(
                    level="deepstack",
                    page_label="条件 DeepStack 注入（layer 0..2 且 Kref>0）",
                    title=_level_title("HiDream-O1-Image", group, "DeepStack 条件注入"),
                    modules=deepstack,
                    note=(
                        "DeepStack 只在 visual_pos_masks 为 True 的条件图行上相加；"
                        "t2i（Kref=0）时该路径不执行。"
                    ),
                )
            )
    else:
        attention = _present(
            modules, "v_norm_attn", "v_attn_qkv", "v_attn_rope",
            "v_attn_seg", "v_attn_core", "v_attn_out", "v_res_attn")
        specs.append(
            SegmentSpec(
                level="attention",
                page_label="视觉塔注意力（qkv 合并投影 → 2D rotary → 按图分段非因果注意力）",
                title=_level_title("HiDream-O1-Image", group, "视觉塔注意力内部计算"),
                modules=attention,
                note=(
                    "按 cu_seqlens 逐图切段做非因果注意力，不是 decoder 因果 mask；"
                    "视觉塔只在参考图/编辑任务执行。"
                ),
                pair_raw=True,
            )
        )
        ffn = _present(modules, "v_norm_ffn", "v_ffn_mlp", "v_res_ffn")
        if ffn:
            specs.append(
                SegmentSpec(
                    level="ffn",
                    page_label="视觉 MLP（GELU-tanh，4304）与残差",
                    title=_level_title("HiDream-O1-Image", group, "视觉塔 MLP"),
                    modules=ffn,
                )
            )
        deepstack = _present(modules, "v_deepstack")
        if deepstack:
            specs.append(
                SegmentSpec(
                    level="deepstack",
                    page_label="DeepStack tap（block 8/16/24 特征抽取）",
                    title=_level_title("HiDream-O1-Image", group, "视觉塔 DeepStack tap"),
                    modules=deepstack,
                    note="第 8/16/24 个 ViT block 输出被 DeepStack merger 抽走，是辅助出口。",
                )
            )
    return specs


# ---------------------------------------------------------------------------
# 声明表
# ---------------------------------------------------------------------------


_HF_REPOSITORY = GraphRepository(
    url="https://github.com/huggingface/transformers.git",
    revision="536ecc007387a50e77603bb5d92100e9b07514cc",
    root_rel="hflib/transformers",
    path_prefix="hflib/transformers/",
)

_HIDREAM_REPOSITORY = GraphRepository(
    url="https://github.com/HiDream-ai/HiDream-O1-Image.git",
    revision="2c2d29ff729e48f33e41f49edfdbd81d5ac103b4",
    root_rel="models/HiDream/HiDream-O1-Image",
    path_prefix="models/HiDream/HiDream-O1-Image/",
)


MODEL_GRAPH_PROFILES: dict[str, ModelGraphProfile] = {
    "deepseek_v4_flash_0731_hf": ModelGraphProfile(
        model_key="deepseek_v4_flash_0731_hf",
        display="DeepSeek V4",
        repository=_HF_REPOSITORY,
        network_title="DeepSeek V4 · 整网主路径（token → logits）",
        network_side_prefixes=("net_position_ids", "net_mask_build"),
        group_note="",
        block_note=(
            "模块级张量数据流：mHC 的 pre/post/comb 作为真实依赖进入两段残差，"
            "缓存以调用前/后状态表达；Attention 内部与 FFN/MoE/mHC 细节在下方子图。"
        ),
        segments=lambda block, group, modules: [],
        levels=("network", "group", "decoder", "kv", "attention", "moe", "mhc"),
    ),
    "glm_5_3_hf": ModelGraphProfile(
        model_key="glm_5_3_hf",
        display="GLM-5.3",
        repository=_HF_REPOSITORY,
        network_title="GLM-5.3 · 整网主路径（token → logits）",
        network_side_prefixes=("net_position_ids", "net_mask_build", "net_rope"),
        group_note=(
            "同组共用一份 Block 模板（模板复用不代表共享权重）；full 层实例化独立"
            " DSA indexer，shared 层直接复用上一 full 层的 top-k 索引。"
        ),
        block_note=(
            "模块级张量数据流：Q 支路、压缩 KV 与 DSA Indexer 并行，稀疏 mask 在"
            "注意力核心前汇合；主 KV 与 indexer 缓存以调用前/后状态表达。"
        ),
        segments=_glm53_segments,
        levels=("network", "group", "decoder", "attention", "indexer", "kv", "ffn"),
    ),
    "glm_5_3_flash_hf": ModelGraphProfile(
        model_key="glm_5_3_flash_hf",
        display="GLM-5.3-Flash",
        repository=_HF_REPOSITORY,
        network_title="GLM-5.3-Flash · 整网主路径（token → logits）",
        network_side_prefixes=("net_position_ids", "net_mask_build"),
        group_note=(
            "KDA 线性注意力与 MLA/DSA 稀疏注意力是两种真实层型；4 路 mHC 残差流"
            "在每个站点坍缩/写回，同组模板复用不代表共享权重。"
        ),
        block_note=(
            "模块级张量数据流：先做 mHC 坍缩，再进入 KDA 或 DSA 注意力；conv/"
            "recurrent state 与主 KV/indexer 缓存都在调用前/后分开表达。"
        ),
        segments=_flash_segments,
        levels=("network", "group", "decoder", "attention", "indexer", "kv", "mhc", "ffn"),
    ),
    "qwen_3_8_flash_next_hf": ModelGraphProfile(
        model_key="qwen_3_8_flash_next_hf",
        display="Qwen3.8-Flash-Next",
        repository=_HF_REPOSITORY,
        network_title="Qwen3.8-Flash-Next · 整网主路径（token → logits）",
        network_side_prefixes=("net_position_ids", "net_mask_build"),
        group_note=(
            "GDN 线性注意力与 QSA 稀疏注意力是两种真实层型；Gated Residual 的读写"
            "门、N-gram PLE 仅在真实执行层出现。"
        ),
        block_note=(
            "模块级张量数据流：GDN/QSA 计算与 Gated Residual 读写门是真实依赖；"
            "conv/recurrent state、KV/indexer 缓存与 PLE 上下文都在调用前/后分开。"
        ),
        segments=_qwen_segments,
        levels=("network", "group", "decoder", "attention", "indexer", "kv", "ple", "mhc", "ffn"),
    ),
    "hidream_o1_image_hf": ModelGraphProfile(
        model_key="hidream_o1_image_hf",
        display="HiDream-O1-Image",
        repository=_HIDREAM_REPOSITORY,
        network_title="HiDream-O1-Image · 单次生成网络（文本 + 像素 patch + 时间步 → x_pred）",
        network_side_prefixes=(),
        group_note=(
            "统一主干 36 层与条件视觉塔 27 blocks 是两个独立索引空间；视觉塔只在"
            "参考图/编辑任务（Kref>0）执行。"
        ),
        block_note=(
            "模块级张量数据流：同一层内两趟注意力（AR 因果 + 全序列双向）合并后"
            "进入 o_proj；DeepStack 只在条件图行上注入。单次 forward 没有 KV cache。"
        ),
        segments=_hidream_segments,
        levels=("network", "group", "decoder", "attention", "ffn", "deepstack"),
        repository_evidence=False,
        network_phases=(
            ("文本 token 嵌入", "文本输入",
             ("net_text_embed",)),
            ("时间步嵌入（正弦→两层 Linear）", "时间步",
             ("net_t_freq", "net_t_lin1", "net_t_silu", "net_t_lin2")),
            ("视觉 patch + 2D RoPE + ViT 27 blocks", "条件视觉塔",
             ("net_vision_patch", "net_vision_pos", "net_vision_rope",
              "net_vision_loop")),
            ("视觉 merge（LN→fc1→GELU→fc2）", "视觉 merger",
             ("net_vmerge_norm", "net_vmerge_view", "net_vmerge_fc1",
              "net_vmerge_gelu", "net_vmerge_fc2")),
            ("DeepStack merge（3 taps）", "DeepStack",
             ("net_ds_view", "net_ds_norm", "net_ds_fc1",
              "net_ds_gelu", "net_ds_fc2")),
            ("条件散射 + 时间步 token 替换", "条件构造",
             ("net_cond_scatter", "net_tms_substitute")),
            ("像素 patch 投影（proj1→proj2）", "生成 patch",
             ("net_patch_lin1", "net_patch_lin2")),
            ("文本/生成 token 拼接", "序列拼接",
             ("net_concat",)),
            ("位置 RoPE + AR 行索引", "注意力分区",
             ("net_rope", "net_attn_partition")),
            ("36 层 UiT 主干（单次 forward）", "UiT 主干",
             ("net_layer_loop",)),
            ("Final RMSNorm + 像素预测", "输出投影",
             ("net_final_norm", "net_final_lin")),
            ("按 token_types 取生成 patch", "输出选择",
             ("net_select_gen",)),
        ),
    ),
}


def graph_profile_for(model_key: str) -> ModelGraphProfile:
    try:
        return MODEL_GRAPH_PROFILES[model_key]
    except KeyError as exc:
        raise ValueError(
            f"模型 {model_key!r} 没有 Archify 图声明：不借用其它模型的分段/来源规则"
        ) from exc
