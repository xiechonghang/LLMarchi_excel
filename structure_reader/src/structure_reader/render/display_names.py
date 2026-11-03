"""HTML 展示用完整维度/参数名称（反馈12）。

只做显示层 token 替换：JSON/源码变量/锚点 id 不变。
- 参数 id → 完整名称（config 正式字段名）
- shape 表达式中的符号 token → 完整名称；支持 2Dh 这类数字前缀
- 不替换源码快照/HTML 属性/已有完整名称（幂等）
"""
from __future__ import annotations

import html
import re

# 参数 id → 完整名称（与 config/source 字段一致）
PARAM_NAMES = {
    "B": "batch_size",
    "T": "sequence_length",
    "P": "past_sequence_length",
    "H": "hidden_size",
    "Nh": "num_attention_heads",
    "Nkv": "num_key_value_heads",
    "Dh": "head_dim",
    "Rq": "q_lora_rank",
    "Rk": "qk_rope_head_dim",
    "G": "o_groups",
    "Ro": "o_lora_rank",
    "M": "compress_rate",
    "Ki": "index_topk",
    "Ih": "index_n_heads",
    "Id": "index_head_dim",
    "W": "sliding_window",
    "Nhc": "hc_mult",
    # 整网总览（token→logits）参数
    "Nlay": "num_hidden_layers",
    "Ssel": "logits_sequence_length",
    "Lk": "logits_to_keep",
    "Sit": "hc_sinkhorn_iters",
    "Eps": "hc_eps",
    "EpsN": "rms_norm_eps",
    "E": "n_routed_experts",
    "K": "num_experts_per_tok",
    "Es": "moe_intermediate_size",
    "Lim": "swiglu_limit",
    "Scl": "routed_scaling_factor",
    "ThetaC": "compress_rope_theta",
    "Yf": "rope_scaling_factor",
    # 派生参数 id（JSON parameters 实际清单）
    "Nwin": "num_full_windows",
    "Nwin_i": "num_full_windows_indexer",
    "Ncomp": "num_compressed_entries",
    "Lkv": "kv_sequence_length",
    "Lmask": "mask_sequence_length",
    "Tbuf": "buffer_length",
    "Tbufp": "buffer_length_after",
    "Tbufi": "indexer_buffer_length",
    "Tbufip": "indexer_buffer_length_after",
    "Tcc": "compressor_entries_before",
    "Tci": "indexer_entries_before",
    "Vocab": "vocab_size",
    "ThetaM": "rope_theta",
    # GLM-5.3（glm_moe_dsa）：与配置字段同名的显式 id，避免借用 DeepSeek 符号
    "Dq": "qk_head_dim",
    "Dn": "qk_nope_head_dim",
    "Dv": "v_head_dim",
    "Rkv": "kv_lora_rank",
    "Is": "intermediate_size",
    "Ng": "n_group",
    "Ktg": "topk_group",
    "EpsA": "sub_norm_eps",
    "Lidx": "indexer_kv_sequence_length",
    "Ksel": "num_selected_entries",
    "Ttok": "flattened_token_count",
    # GLM-5.3-Flash（glm5_next）：KDA / k-pool / mHC 的显式 id
    "Hc": "hc_mult",
    "NhL": "linear_num_heads",
    "DhL": "linear_head_dim",
    "Kconv": "linear_conv_kernel_dim",
    "Kpool": "index_kpool",
    "Ipack": "indexer_packed_width",
    "Npool": "num_candidate_pools",
    "SelP": "num_selected_pools",
    "Kout": "indexer_output_width",
    "Srec": "recurrent_state_elements",
    "Sconv": "conv_state_elements",
    "Swi": "swiglu_limit",
    "GateLB": "linear_gate_lower_bound",
    "Sit": "hc_sinkhorn_iters",
    "Mhc": "mhc_coefficient_count",
    "Vq": "query_visible_length",
    "Ksel_last": "num_selected_entries_last_query",
    # Qwen3.8-Flash-Next（qwen4_exp）：GDN / QSA / PLE / Gated Residual
    "NkH": "linear_num_key_heads",
    "NvH": "linear_num_value_heads",
    "Dk": "linear_key_head_dim",
    "DvH": "linear_value_head_dim",
    "Gmul": "qk_to_v_head_repeat",
    "Dqkv": "gdn_qkv_width",
    "Dz": "gdn_value_width",
    "Pfrac": "partial_rotary_factor",
    "Nrep": "num_key_value_groups",
    "Iqh": "indexer_n_heads",
    "Ikh": "indexer_kv_heads",
    "Btop": "indexer_block_topk",
    "Kc": "indexer_compress_ratio",
    "Drot": "partial_rotary_dim",
    "Ngm": "ngram_size",
    "Hng": "heads_per_ngram",
    "Ngh": "ngram_heads",
    "Dgh": "ngram_head_dim",
    "Dple": "ple_embed_dim",
    "Kple": "ple_conv_kernel_size",
    "Dild": "ple_conv_dilation",
    "Lpconv": "ple_short_conv_state_len",
    "Lctx": "ngram_context_len",
    "Vng": "ngram_vocab_size_base",
    "Vngpad": "ngram_padded_vocab_size",
    "Eos": "eos_token_id",
    "Sple": "ple_conv_state_elements",
    "Dqw": "query_width",
    "Dkv": "kv_width",
    "Bsel": "num_selected_blocks",
    "Nq": "num_query_complete_blocks",
    "Bq": "num_query_selected_blocks",
    "Rtail": "query_tail_tokens",
    "Rhc": "hc_lowrank",
    "Npos": "max_position_embeddings",
    # HiDream-O1-Image（单次生成网络）局部符号：全部有 modeling/config 定义
    "S": "seqlen",
    "Stot": "total_sequence_length",
    "S_tot": "total_sequence_length",
    "Hq": "q_projection_width",
    "H_q": "q_projection_width",
    "Hkv": "kv_projection_width",
    "H_kv": "kv_projection_width",
    "Sar": "num_ar_tokens",
    "S_ar": "num_ar_tokens",
    "Nar": "num_ar_tokens",
    "N_ar": "num_ar_tokens",
    "Ngt": "num_generation_tokens",
    "N_gt": "num_generation_tokens",
    "Fi": "intermediate_size",
    "Pin": "pixel_patch_input_width",
    "P_in": "pixel_patch_input_width",
    "Vpin": "condition_pixel_input_width",
    "V_pin": "condition_pixel_input_width",
    "Nhv": "num_vision_attention_heads",
    "Dhv": "vision_head_dim",
    "D_hv": "vision_head_dim",
    "Hv": "vision_hidden_size",
    "Hv3": "vision_qkv_width",
    "H_v3": "vision_qkv_width",
    "H_v4": "merged_unit_width",
    "Fv": "vision_intermediate_size",
    "Pv": "vision_patch_size",
    "Tp": "temporal_patch_size",
    "Mv": "spatial_merge_size",
    "Npm": "vision_position_embeddings",
    "Nv": "vision_block_count",
    "Nvis": "deepstack_visual_tokens",
    "Svis": "vision_patch_tokens",
    "Sgen": "generation_token_count",
    "S_gen": "generation_token_count",
    "Spix": "target_pixel_tokens",
    "S_pix": "target_pixel_tokens",
    "Ncnd": "num_condition_images",
    "Sref": "reference_patch_tokens",
    "Sref": "reference_patch_tokens",
    "S_i": "per_image_patch_tokens",
    "S_vis_m": "merged_visual_tokens",
    "S_attn": "attention_score_elements",
    "Ndh": "deepstack_layers",
    "Ppix": "pixel_patch_size",
    "Cin": "pixel_channel_count",
    "Fq": "timestep_sinusoid_dim",
    "Tms": "timestep_token_id",
    "P_ca": "pixel_patch_lowrank_dim",
    "Ptxt": "text_prompt_tokens",
    "Himg": "target_image_height",
    "Wimg": "target_image_width",
    "Kref": "num_reference_images",
}

# shape 表达式 token（含派生轴）；值必须能自解释
TOKENS = {
    **PARAM_NAMES,
    "N_win": "num_full_windows",
    "Nwin": "num_full_windows",
    "N_win_i": "num_full_windows_indexer",
    "N_comp": "num_compressed_entries",
    "Tbuf": "buffer_length",
    "Tbufp": "buffer_length_after",
    "Tbuf_i": "indexer_buffer_length",
    "Tbufip": "indexer_buffer_length_after",
    "Lkv": "kv_sequence_length",
    "Lmask": "mask_sequence_length",
    "Ncomp": "num_compressed_entries",
    "Nwin": "num_full_windows",
    "Nwin_i": "num_full_windows_indexer",
    "Tcc": "compressor_entries_before",
    "Tci": "indexer_entries_before",
    "Vocab": "vocab_size",
    "ThetaM": "rope_theta",
    "Tci_post": "indexer_entries_after",
    "L_kv": "kv_sequence_length",
    "L_mask": "mask_sequence_length",
    "N_hit": "num_hit_experts",
    "N_e": "num_expert_tokens",
    "K_sel": "num_selected_entries",
    "N_kv": "kv_sequence_length",
    "Vocab": "vocab_size",
    "ThetaM": "rope_theta",
    # GLM-5.3 显式符号（含下划线写法）
    "L_idx": "indexer_kv_sequence_length",
    "T_tok": "flattened_token_count",
    # GLM-5.3-Flash 显式符号（含下划线写法）
    "S_rec": "recurrent_state_elements",
    "S_conv": "conv_state_elements",
    "I_pack": "indexer_packed_width",
    "N_pool": "num_candidate_pools",
    "K_out": "indexer_output_width",
    "Mhc": "mhc_coefficient_count",
    "V_q": "query_visible_length",
    "K_sel_last": "num_selected_entries_last_query",
    "B_sel": "num_selected_blocks",
    "N_q": "num_query_complete_blocks",
    "B_q": "num_query_selected_blocks",
    "R_tail": "query_tail_tokens",
    "D_qkv": "gdn_qkv_width",
    "D_z": "gdn_value_width",
    "D_rot": "partial_rotary_dim",
    "D_q": "query_width",
    "D_kv": "kv_width",
    "N_ngram_heads": "ngram_heads",
    "S_ple_conv": "ple_conv_state_elements",
    # HiDream-O1-Image：shape 里出现的压缩写法（与上面参数表同义）
    "S": "seqlen",
    "Stot": "total_sequence_length",
    "S_tot": "total_sequence_length",
    "Hq": "q_projection_width",
    "H_q": "q_projection_width",
    "Hkv": "kv_projection_width",
    "H_kv": "kv_projection_width",
    "Sar": "num_ar_tokens",
    "S_ar": "num_ar_tokens",
    "Nar": "num_ar_tokens",
    "N_ar": "num_ar_tokens",
    "Ngt": "num_generation_tokens",
    "N_gt": "num_generation_tokens",
    "Fi": "intermediate_size",
    "Pin": "pixel_patch_input_width",
    "P_in": "pixel_patch_input_width",
    "Vpin": "condition_pixel_input_width",
    "V_pin": "condition_pixel_input_width",
    "Nhv": "num_vision_attention_heads",
    "Dhv": "vision_head_dim",
    "D_hv": "vision_head_dim",
    "Hv": "vision_hidden_size",
    "Hv3": "vision_qkv_width",
    "H_v3": "vision_qkv_width",
    "H_v4": "merged_unit_width",
    "Fv": "vision_intermediate_size",
    "Pv": "vision_patch_size",
    "Tp": "temporal_patch_size",
    "Mv": "spatial_merge_size",
    "Npm": "vision_position_embeddings",
    "Nv": "vision_block_count",
    "Nvis": "deepstack_visual_tokens",
    "Svis": "vision_patch_tokens",
    "Sgen": "generation_token_count",
    "S_gen": "generation_token_count",
    "Spix": "target_pixel_tokens",
    "S_pix": "target_pixel_tokens",
    "Ncnd": "num_condition_images",
    "Sref": "reference_patch_tokens",
    "S_i": "per_image_patch_tokens",
    "S_vis_m": "merged_visual_tokens",
    "S_attn": "attention_score_elements",
    "Ndh": "deepstack_layers",
    "Ppix": "pixel_patch_size",
    "Cin": "pixel_channel_count",
    "Fq": "timestep_sinusoid_dim",
    "Tms": "timestep_token_id",
    "P_ca": "pixel_patch_lowrank_dim",
    "Ptxt": "text_prompt_tokens",
    "Himg": "target_image_height",
    "Wimg": "target_image_width",
    "Kref": "num_reference_images",
}

_TOKEN_RE = re.compile(r"(\d*)([A-Za-z_][A-Za-z0-9_]*)")


def _expand(match: re.Match) -> str:
    digits, ident = match.group(1), match.group(2)
    name = TOKENS.get(ident)
    if not name:
        return match.group(0)
    return f"{digits}×{name}" if digits else name


def shape_text_raw(shape) -> str:
    raw = ",".join(str(a) for a in shape)
    text = _TOKEN_RE.sub(_expand, raw)
    # 前后状态：Tbuf' / Tbuf_i' 后缀用 _after 表达
    text = text.replace("buffer_length'", "buffer_length_after")
    text = text.replace("indexer_buffer_length'", "indexer_buffer_length_after")
    return text


def shape_text(shape) -> str:
    """shape 列表 → 完整名称文本（逗号分隔，已 HTML 转义）。"""
    return html.escape(shape_text_raw(shape))


def display_axis(axis: str) -> str:
    """单轴显示名（保留为一个轴；min(P,W-1) 内部逗号不拆）。"""
    return expand_display_text(str(axis))


def display_shape(shape) -> str:
    return "[" + shape_text(shape) + "]"


def full_name(pid: str) -> str:
    return PARAM_NAMES.get(pid, pid)


def param_label(pid: str, value=None) -> str:
    name = full_name(pid)
    if value is None:
        return html.escape(name)
    return html.escape(f"{name}={value}")


_SHAPE_BRACKET_RE = re.compile(r"\[([^\[\]:.(){}]+)\]")
_PART_RE = re.compile(r"[\w×÷+\-·/']+")


def expand_shape_text(text: str) -> str:
    """说明文字里的 [B, T, Nhc, H] 形态 → 完整名称（未转义；源码片段不匹配则原样）。"""
    def rep(m: re.Match) -> str:
        parts = [p.strip() for p in m.group(1).split(",")]
        if not parts or not all(_PART_RE.fullmatch(p) for p in parts):
            return m.group(0)
        if not any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) and p in TOKENS for p in parts):
            return m.group(0)
        return "[" + shape_text_raw(parts) + "]"
    return _SHAPE_BRACKET_RE.sub(rep, text)


def expand_display_text(text: str) -> str:
    """说明/注释文字：先展开 [shape]，再把独立参数 token 换成完整名称（未转义）。"""
    text = expand_shape_text(text)
    text = text.replace("；G=o_groups", "").replace("; G=o_groups", "")

    def repl(m: re.Match) -> str:
        digits, ident = m.group(1), m.group(2)
        name = TOKENS.get(ident)
        if not name:
            return m.group(0)
        start, end = m.span()
        prev = text[start - 1] if start > 0 else ""
        nxt = text[end:end + 1]
        rest = text[end:end + 3]
        # 转置/幂：.T、^T、k^T 保持源码数学语义
        if prev and prev in ".^":
            return m.group(0)
        if ident == "T" and nxt == "ᵀ":
            return m.group(0)
        # K 只在明确表示“专家数”（K 个专家 / K experts）时展开；Q/K 投影、
        # K 归一、K 交错等键语义一律保留源码 K。
        if ident == "K":
            after = rest.lstrip()
            if not after.startswith(("个", "专家", "expert")):
                return m.group(0)
        return f"{digits}×{name}" if digits else name

    token_re = re.compile(r"(?<![A-Za-z0-9_])(\d*)([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_])")
    text = token_re.sub(repl, text)
    text = text.replace("buffer_length'", "buffer_length_after")
    text = text.replace("indexer_buffer_length'", "indexer_buffer_length_after")
    return text
