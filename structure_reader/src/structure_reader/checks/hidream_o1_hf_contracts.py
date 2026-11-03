"""HiDream-O1-Image 结构与契约检查（发现真实错误，不照实现写镜像断言）。

检查分三层：

- ``validate_hf_reading_document``：单次 forward 层文档（``hidream_o1_layer_document.v1``）
  的身份、单阶段声明、骨架/数据流闭合；拒绝 Prefill/Decode 两阶段投影与 KV cache。
- ``validate_hf_layer_structure``：Block 语义锚点 ——
  decoder：GQA 32/8、逐头 q/k RMSNorm、MRoPE 只施加 q/k、两趟注意力 + index_copy、
  两个 RMSNorm 站点与标量残差、SwiGLU 12288、条件 DeepStack、无 MoE/无缓存/无 adaLN；
  vision：LayerNorm、qkv 合并投影（bias）、2D rotary、非因果分图注意力、GELU-tanh 4304、
  条件 DeepStack tap、无缓存。
- ``validate_hf_network_overview``：整网只到 x_pred（不是 logits）、不含 lm_head/采样
  scheduler/VAE/独立文本编码器、36 层真实顺序、两层层组真实归属、视觉塔条件执行。

这些检查会拒绝“套用其它模型事实”的改法（MoE/线性注意力/KV cache/adaLN/VAE 等），
但不会复制实现里的每个表达式。
"""
from __future__ import annotations

from structure_reader.checks.hf_contract_core import (
    check_flow_units,
    check_input_identity,
    check_model_identity,
    check_unit,
)
from structure_reader.extract.hidream_o1_hf import DOCUMENT_SCHEMA
from structure_reader.rules.hidream_o1_hf import SOURCE_FILES

__all__ = [
    "validate_hf_reading_document",
    "validate_hf_layer_structure",
    "validate_hf_network_overview",
]

# 其它模型机制的禁词：HiDream-O1-Image 的这份实现里不存在。
_FORBIDDEN_BLOCK_TOKENS = (
    "moe", "mhc", "sinkhorn", "gated_residual", "kda", "delta", "sliding_window",
    "adaLN", "adaln", "q_lora", "kv_lora", "compressor", "indexer", "vae", "clip",
    "t5_encoder", "text_encoder",
)

_DECODER_REQUIRED_OPS = (
    "d_in_norm", "d_q_proj", "d_k_proj", "d_v_proj", "d_q_norm", "d_k_norm",
    "d_q_rope", "d_k_rope", "d_gather_ar", "d_qk_ar", "d_softmax_ar", "d_av_ar",
    "d_qk_full", "d_softmax_full", "d_av_full", "d_merge_ar", "d_o_proj",
    "d_res_attn", "d_post_norm", "d_gate_proj", "d_up_proj", "d_silu",
    "d_swiglu_mul", "d_down_proj", "d_res_ffn", "d_deepstack_add",
)

_VISION_REQUIRED_OPS = (
    "v_norm1", "v_qkv_proj", "v_qkv_split", "v_q_rope", "v_k_rope", "v_seg_split",
    "v_qk", "v_softmax", "v_av", "v_seg_concat", "v_proj", "v_res1", "v_norm2",
    "v_fc1", "v_gelu", "v_fc2", "v_res2", "v_deepstack_tap",
)

_NETWORK_REQUIRED_OPS = (
    "net_text_embed", "net_t_freq", "net_t_lin1", "net_t_lin2", "net_tms_substitute",
    "net_patch_lin1", "net_patch_lin2", "net_concat", "net_rope", "net_attn_partition",
    "net_layer_loop", "net_final_norm", "net_final_lin", "net_select_gen",
)
_NETWORK_VISION_OPS = (
    "net_vision_patch", "net_vision_pos", "net_vision_rope", "net_vision_loop",
    "net_vmerge_norm", "net_vmerge_view", "net_vmerge_fc1", "net_vmerge_gelu",
    "net_vmerge_fc2", "net_ds_view", "net_ds_norm", "net_ds_fc1", "net_ds_gelu",
    "net_ds_fc2", "net_cond_scatter",
)


def _op_index(unit: dict) -> dict:
    return {op.get("id"): op for op in unit.get("operators", [])}


def _param_index(unit: dict) -> dict:
    return {param.get("id"): param for param in unit.get("parameters", [])}


def _tensor_index(unit: dict) -> dict:
    return {tensor.get("id"): tensor for tensor in unit.get("tensors", [])}


def _numeric(param: dict):
    value = param.get("value")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _text(unit: dict) -> str:
    """算子/张量/参数的 id、类型与模块路径拼成检索串（禁词检查用；不含说明文字）。"""
    parts: list[str] = []
    for key in ("operators", "tensors", "parameters"):
        for item in unit.get(key, []):
            parts.append(str(item.get("id", "")))
            parts.append(str(item.get("name", "")))
            parts.append(str(item.get("op", "")))
            parts.append(str(item.get("module_path", "")))
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# 单次 forward 层文档
# ---------------------------------------------------------------------------


def validate_hf_reading_document(document: dict, input_shas: dict[str, str]) -> list[str]:
    issues: list[str] = []
    if not isinstance(document, dict):
        return ["document 必须是对象"]
    if document.get("schema") != DOCUMENT_SCHEMA:
        issues.append(f"document.schema 必须是 {DOCUMENT_SCHEMA!r}")
    issues.extend(check_model_identity(document.get("model") or {}))
    issues.extend(check_input_identity(input_shas, SOURCE_FILES))

    scope = document.get("scope") or {}
    if scope.get("analysis") != "single_forward":
        issues.append("scope.analysis 必须是 single_forward（本票只分析一次 forward）")
    phases = scope.get("phases")
    if phases != ["generation"]:
        issues.append(f"scope.phases 必须且只能是 ['generation']，实得 {phases!r}；"
                      "不把单次网络投影成 Prefill/Decode 两阶段")
    block = document.get("block")
    if not isinstance(block, dict):
        return issues + ["document.block 缺失"]
    issues.extend(check_unit(block, "block"))
    issues.extend(check_flow_units([block]))

    # 单次 forward 没有 KV cache：不允许出现 cache 张量或缓存更新算子。
    for tensor in block.get("tensors", []):
        if tensor.get("kind") == "cache":
            issues.append(f"单次 forward 不应有 cache 张量：{tensor.get('id')}")
    for op in block.get("operators", []):
        if "cache" in str(op.get("id", "")).lower() or "_cache" in str(op.get("op", "")).lower():
            issues.append(f"单次 forward 不应有缓存读写算子：{op.get('id')}")
    return issues


# ---------------------------------------------------------------------------
# Block 语义
# ---------------------------------------------------------------------------


def _check_common_block(structure: dict) -> list[str]:
    issues: list[str] = []
    block = structure.get("block") or {}
    scope = structure.get("scope") or {}
    if structure.get("schema") != "hf_layer_structure.v1":
        issues.append("structure.schema 必须是 hf_layer_structure.v1")
    if not block:
        return issues + ["structure.block 缺失"]
    params = _param_index(block)
    if block.get("kind") == "uit_decoder":
        for param_id in ("B", "Ptxt", "Himg", "Wimg"):
            if param_id not in params:
                issues.append(f"缺少输入规模参数 {param_id}（结构必须保留可填写输入）")
    elif block.get("kind") == "vit_tower" and "Svis" not in params:
        issues.append("视觉塔必须保留输入规模参数 Svis（条件图 patch 数）")
    text = _text(block)
    for token in _FORBIDDEN_BLOCK_TOKENS:
        if token.lower() in text:
            issues.append(f"Block 出现不属于本模型的机制/模块: {token}")
    if scope.get("analysis_path") != "eager":
        issues.append("scope.analysis_path 必须是 eager（静态结构阅读口径）")
    return issues


def _check_decoder(structure: dict) -> list[str]:
    issues: list[str] = []
    block = structure["block"]
    ops = _op_index(block)
    params = _param_index(block)
    tensors = _tensor_index(block)
    if block.get("attention_type") != "hybrid_causal_full":
        issues.append(f"decoder attention_type 必须是 hybrid_causal_full，实得 {block.get('attention_type')!r}")
    if block.get("moe_type") != "dense_swiglu":
        issues.append(f"decoder moe_type 必须是 dense_swiglu，实得 {block.get('moe_type')!r}")
    missing = [oid for oid in _DECODER_REQUIRED_OPS if oid not in ops]
    if missing:
        issues.append(f"decoder 缺少必要算子: {missing}")

    expected = {"H": 4096, "Nh": 32, "Nkv": 8, "Dh": 128, "Hq": 4096, "Hkv": 1024,
                "Fi": 12288, "Ndh": 3, "Nlay": 36, "Ppix": 32}
    for pid, want in expected.items():
        got = _numeric(params.get(pid, {}))
        if got != want:
            issues.append(f"参数 {pid} 必须是 {want}（实得 {got}）")

    # q/k 有逐头 RMSNorm；v 没有 norm、没有 RoPE。
    if "d_q_norm" not in ops or "d_k_norm" not in ops:
        issues.append("q/k 必须各有逐头 RMSNorm")
    for op in block.get("operators", []):
        if op.get("op") == "rmsnorm" and any(str(ref).startswith("d_v") for ref in op.get("inputs", [])):
            issues.append(f"V 不应做 norm：{op.get('id')}")
        if "rope" in str(op.get("id", "")).lower() and "d_v" in [str(x) for x in op.get("inputs", [])]:
            issues.append(f"V 不应做 RoPE：{op.get('id')}")
    if ops.get("d_q_rope", {}).get("outputs") != ["d_q"] or \
            ops.get("d_k_rope", {}).get("outputs") != ["d_k"]:
        issues.append("MRoPE 必须作用在 q 与 k 上")

    # 两趟注意力 + index_copy：AR 子集 causal、全序列双向。
    if "d_merge_ar" not in ops:
        issues.append("缺少两趟注意力的 index_copy 合并算子")
    else:
        note = str(ops["d_merge_ar"].get("note", ""))
        if "4D" not in note and "mask" not in note:
            issues.append("两趟合并算子必须说明 eager 4D mask 的等价实现")
    if not tensors.get("d_score_ar") or not tensors.get("d_score_full"):
        issues.append("缺少 AR 子集与全序列两套 attention score 张量")
    else:
        ar = tensors["d_score_ar"].get("shape", [])
        full = tensors["d_score_full"].get("shape", [])
        if "Nar" not in ar:
            issues.append("AR 子集 score 的序列轴必须是 Nar（不是全长）")
        if "Stot" not in full:
            issues.append("全序列 score 的序列轴必须是 Stot")

    # 两个 RMSNorm 站点 + 两个标量残差；没有 MoE / 门控残差。
    for oid in ("d_in_norm", "d_post_norm"):
        if ops.get(oid, {}).get("op") != "rmsnorm":
            issues.append(f"{oid} 必须是 rmsnorm")
    for oid in ("d_res_attn", "d_res_ffn"):
        if ops.get(oid, {}).get("op") != "add":
            issues.append(f"{oid} 必须是普通标量 add")
    if "d_res_ffn" in ops and ops["d_res_ffn"].get("inputs") != ["d_h1", "d_ffn"]:
        issues.append("FFN 残差必须加回 attention 残差后的单流（不是多路混合）")

    # DeepStack 注入必须是条件算子（K>0 且 layer 0..2）。
    ds = ops.get("d_deepstack_add", {})
    ds_note = str(ds.get("note", ""))
    if "Kref>0" not in ds_note or "0,1,2" not in ds_note:
        issues.append("DeepStack 注入必须显式条件：参考图任务且 layer ∈ {0,1,2}")

    # 不允许 KV cache / lm_head / logits。
    if any(t.get("kind") == "cache" for t in block.get("tensors", [])):
        issues.append("decoder Block 不应有 cache 张量")
    if any("logit" in str(op.get("id", "")).lower() or "lm_head" in str(op.get("module_path", ""))
           for op in block.get("operators", [])):
        issues.append("decoder Block 不含 lm_head/logits（生成路径输出 x_pred）")

    for pid, token in (("Himg", "Himg % Ppix == 0"), ("Wimg", "Wimg % Ppix == 0"),
                       ("Sref", "Sref"), ("Svis", "Svis % (Mv * Mv) == 0")):
        if token not in str(params.get(pid, {}).get("consistency", "")):
            issues.append(f"参数 {pid} 缺少输入一致性约束（{token}）")

    # q/k/v 权重形状必须是 (Hq,H)/(Hkv,H)/(Hkv,H)。
    for tid, want in (("d_w_q", ["Hq", "H"]), ("d_w_k", ["Hkv", "H"]), ("d_w_v", ["Hkv", "H"]),
                      ("d_w_o", ["H", "Hq"]), ("d_w_down", ["H", "Fi"])):
        got = tensors.get(tid, {}).get("shape")
        if got != want:
            issues.append(f"权重 {tid} 形状必须是 {want}（实得 {got}）")
    return issues


def _check_vision(structure: dict) -> list[str]:
    issues: list[str] = []
    block = structure["block"]
    ops = _op_index(block)
    params = _param_index(block)
    tensors = _tensor_index(block)
    if block.get("attention_type") != "bidirectional_vision":
        issues.append(f"vision attention_type 必须是 bidirectional_vision，实得 {block.get('attention_type')!r}")
    if block.get("moe_type") != "dense_gelu_mlp":
        issues.append(f"vision moe_type 必须是 dense_gelu_mlp，实得 {block.get('moe_type')!r}")
    missing = [oid for oid in _VISION_REQUIRED_OPS if oid not in ops]
    if missing:
        issues.append(f"vision 缺少必要算子: {missing}")
    expected = {"Hv": 1152, "Nhv": 16, "Dhv": 72, "Hv3": 3456, "Fv": 4304,
                "Pv": 16, "Tp": 2, "Mv": 2, "Nv": 27, "Npm": 2304}
    for pid, want in expected.items():
        got = _numeric(params.get(pid, {}))
        if got != want:
            issues.append(f"vision 参数 {pid} 必须是 {want}（实得 {got}）")
    for oid in ("v_norm1", "v_norm2"):
        if ops.get(oid, {}).get("op") != "layernorm":
            issues.append(f"{oid} 必须是 LayerNorm（视觉塔不是 RMSNorm）")
    if ops.get("v_qkv_proj", {}).get("op") != "linear":
        issues.append("v_qkv_proj 必须是合并的 qkv 线性投影")
    if len(ops.get("v_qkv_proj", {}).get("weights", [])) < 2:
        issues.append("视觉 qkv 投影 bias=True，必须声明 bias 权重")
    if ops.get("v_gelu", {}).get("op") != "gelu-tanh":
        issues.append("视觉 MLP 激活必须是 GELU-tanh")
    if ops.get("v_qk", {}).get("op") != "matmul-scaled":
        issues.append("视觉注意力核心必须是 matmul-scaled")
    note = str(ops.get("v_qk", {}).get("note", ""))
    if "非因果" not in note:
        issues.append("视觉注意力必须显式非因果（is_causal=False）")
    tap = ops.get("v_deepstack_tap", {})
    if "8,16,24" not in str(tap.get("note", "")):
        issues.append("DeepStack tap 必须说明 block 8/16/24 条件")
    if any(t.get("kind") == "cache" for t in block.get("tensors", [])):
        issues.append("vision Block 不应有 cache 张量")
    for tid, want in (("v_w_qkv", ["Hv3", "Hv"]), ("v_w_fc1", ["Fv", "Hv"]),
                      ("v_w_fc2", ["Hv", "Fv"])):
        got = tensors.get(tid, {}).get("shape")
        if got != want:
            issues.append(f"视觉权重 {tid} 形状必须是 {want}（实得 {got}）")

    # 逐图分段：cu_seqlens/切分/局部 shape/拼接必须真实存在，不能平方总长。
    if "v_cu_seqlens" not in tensors:
        issues.append("视觉注意力缺少 cu_seqlens/每图边界张量")
    elif tensors["v_cu_seqlens"].get("shape") != ["Kref+1"]:
        issues.append("cu_seqlens 必须是一维 [Kref+1]（cumsum 左 pad 0），"
                      f"实得 {tensors['v_cu_seqlens'].get('shape')}")
    if "v_seg_split" not in ops or "v_seg_concat" not in ops:
        issues.append("视觉注意力缺少逐段切分/拼接算子")
    for tid in ("v_scores", "v_probs"):
        got = tensors.get(tid, {}).get("shape")
        if got != ["Nhv", "S_i", "S_i"]:
            issues.append(f"{tid} 必须是逐段局部 shape [Nhv,S_i,S_i]（实得 {got}）")
    if tensors.get("v_scores", {}).get("shape") == ["Nhv", "Svis", "Svis"]:
        issues.append("视觉 score 不能按所有图总长平方 [Nhv,Svis,Svis]")
    for tid in ("v_q_seg", "v_k_seg", "v_v_seg"):
        got = tensors.get(tid, {}).get("shape")
        if got != ["Nhv", "S_i", "Dhv"]:
            issues.append(f"{tid} 必须是第 i 段局部 shape [Nhv,S_i,Dhv]（实得 {got}）")
    seg = str(ops.get("v_seg_split", {}).get("note", ""))
    if "S_i" not in seg or "K" not in seg:
        issues.append("v_seg_split 必须说明逐段长度 S_i 与段数 K")
    s_i = params.get("S_i", {})
    s_i_consistency = str(s_i.get("consistency", ""))
    if "Svis" not in s_i_consistency or "Kref * S_i == Svis" not in s_i_consistency:
        issues.append("每图长度 S_i 必须有与总长 Svis 的一致性约束")
    if "S_i % (Mv * Mv) == 0" not in s_i_consistency:
        issues.append("每图长度 S_i 必须有 merge 单元整除约束（S_i % 4 == 0）")
    if "S_attn" not in {p.get("symbol") for p in block.get("parameters", [])}:
        issues.append("缺少逐图注意力总规模参数 S_attn")
    attn = params.get("Sattn", {})
    if "Nhv * Kref * S_i * S_i" not in str(attn.get("expression", "")):
        issues.append("S_attn 必须是 Σ 逐图 Nhv·S_i²（Kref*S_i²），不能由总长平方")
    guard_deriv = {d.get("id"): d for d in block.get("derivations", [])}.get("S_attn_elems", {})
    guard_text = str(guard_deriv.get("guard", ""))
    if "Kref * S_i == Svis" not in guard_text or "S_i % (Mv * Mv) == 0" not in guard_text:
        issues.append("逐图注意力总规模必须带 Kref*S_i=Svis 与每图 merge 整除守卫")
    svis = params.get("Svis", {})
    if "Svis % (Mv * Mv) == 0" not in str(svis.get("consistency", "")):
        issues.append("Svis 必须带 merge 单元整除约束")
    return issues


def validate_hf_layer_structure(structure: dict) -> list[str]:
    issues = _check_common_block(structure)
    block = structure.get("block") or {}
    if block.get("kind") == "uit_decoder":
        issues.extend(_check_decoder(structure))
    elif block.get("kind") == "vit_tower":
        issues.extend(_check_vision(structure))
    else:
        issues.append(f"未知 Block kind: {block.get('kind')!r}（只支持 uit_decoder/vit_tower）")
    # 场景派生必须按本模型的输入规模轴求值，而不是 seqlen/kvcachelen。
    derivations = (structure.get("scenario_derivations") or {})
    if not derivations:
        issues.append("缺少 scenario_derivations")
    else:
        first = next(iter(derivations.values()))
        ids = {item.get("id") for item in first}
        if "kvcachelen" in " ".join(str(item.get("expression", "")) for item in first):
            issues.append("HiDream 场景不应引用 kvcachelen（本网络没有 KV cache）")
        if block.get("kind") == "uit_decoder" and not ({"S_pix", "S_gen", "S_tot"} & ids):
            issues.append("decoder 场景派生必须包含 patch/序列长度（S_pix/S_gen/S_tot）")
        if block.get("kind") == "vit_tower" and "S_vis_m" not in ids:
            issues.append("视觉塔场景派生必须包含 merge 后 token 数（S_vis_m）")
    return issues


# ---------------------------------------------------------------------------
# 整网
# ---------------------------------------------------------------------------


def validate_hf_network_overview(network: dict, input_shas: dict[str, str]) -> list[str]:
    issues: list[str] = []
    if not isinstance(network, dict):
        return ["network 必须是对象"]
    if network.get("schema") != "hf_network_overview.v1":
        issues.append("network.schema 必须是 hf_network_overview.v1")
    issues.extend(check_model_identity(network.get("model") or {}))
    issues.extend(check_input_identity(input_shas, SOURCE_FILES))
    path = network.get("path")
    if not isinstance(path, dict):
        return issues + ["network.path 缺失"]
    issues.extend(check_unit(path, "network.path"))
    issues.extend(check_flow_units([path]))
    ops = _op_index(path)
    tensors = _tensor_index(path)
    params = _param_index(path)

    missing = [oid for oid in _NETWORK_REQUIRED_OPS if oid not in ops]
    if missing:
        issues.append(f"整网缺少必要算子: {missing}")
    for oid in _NETWORK_VISION_OPS:
        op = ops.get(oid)
        if op is None:
            issues.append(f"整网缺少视觉支路算子: {oid}")
            continue
        if "Kref>0" not in str(op.get("note", "")):
            issues.append(f"视觉支路算子 {oid} 必须显式条件（Kref>0 才执行）")
    for oid in ("net_vision_merge", "net_deepstack_merge"):
        if oid in ops:
            issues.append(f"merger 不能压成黑箱算子 {oid}（需 Norm/view/Linear/GELU/Linear 明细）")
    # 主线：norm 在 view 之前（1152）；DeepStack：view 在 norm 之前（4608）
    if ops.get("net_vmerge_norm", {}).get("op") != "layernorm" or \
            ops.get("net_vmerge_view", {}).get("inputs") != ["n_vmerge_normed"]:
        issues.append("主线 merger 必须是 Norm(1152) → view → Linear → GELU → Linear")
    if ops.get("net_ds_view", {}).get("inputs") != ["n_vtaps"] or \
            ops.get("net_ds_norm", {}).get("inputs") != ["n_ds_units"]:
        issues.append("DeepStack merger 必须是 view → Norm(4608) → Linear → GELU → Linear")
    for oid in ("net_vmerge_gelu", "net_ds_gelu"):
        if ops.get(oid, {}).get("op") not in ("gelu",):
            issues.append(f"{oid} 必须是 GELU 具名算子")
    for tid in ("n_vmerge_units", "n_vmerge_h", "n_vmerge_act",
                "n_ds_units", "n_ds_normed", "n_ds_h", "n_ds_act"):
        if tid not in tensors:
            issues.append(f"merger 缺少中间张量 {tid}")
    ds_note = " ".join(str(op.get("note", "")) for oid, op in ops.items()
                       if oid.startswith("net_ds_"))
    if "×3" not in ds_note or "8/16/24" not in ds_note:
        issues.append("DeepStack merger 必须说明 ×3 与 tap 8/16/24 归属")

    # 终点必须是 x_pred，而不是 logits；生成路径不调用 lm_head。
    for op in path.get("operators", []):
        module = str(op.get("module_path", "")).lower()
        if "lm_head" in module or "logits" in str(op.get("name", "")).lower():
            issues.append(f"整网生成路径不应包含 lm_head/logits：{op.get('id')}")
        for token in ("scheduler", "guidance", "cfg", "noise_inject", "sampling_step",
                      "vae", "clip", "text_encoder"):
            if token in module:
                issues.append(f"整网不应包含外部采样/其它架构模块（{token}）：{op.get('id')}")
    if "n_x_pred" not in tensors:
        issues.append("缺少 x_pred 输出张量")
    if not any(op.get("id") == "net_select_gen" for op in path.get("operators", [])):
        issues.append("缺少按生成 token 取 x_pred 的算子")

    for pid, want in ({"H": 4096, "Nlay": 36, "Hv": 1152, "Nv": 27, "Ppix": 32,
                       "Pin": 3072, "Pca": 1024, "Fq": 256, "Tms": 151673}.items()):
        got = _numeric(params.get(pid, {}))
        if got != want:
            issues.append(f"整网参数 {pid} 必须是 {want}（实得 {got}）")

    # 不分析外部采样：不允许把 50/28 步写成网络节点或成本。
    for op in path.get("operators", []):
        if op.get("id") in ("net_layer_loop",):
            continue
        if "step" in str(op.get("id", "")).lower() and "timestep" not in str(op.get("id", "")):
            issues.append(f"整网不应有采样步算子：{op.get('id')}")

    scope = network.get("scope") or {}
    if scope.get("layer_count") != 36:
        issues.append(f"scope.layer_count 必须是 36（实得 {scope.get('layer_count')}）")
    if scope.get("vision_block_count") != 27:
        issues.append(f"scope.vision_block_count 必须是 27（实得 {scope.get('vision_block_count')}）")
    if scope.get("tms_token_id") != 151673:
        issues.append("scope.tms_token_id 必须是 151673")
    if list(scope.get("deepstack_visual_indexes") or []) != [8, 16, 24]:
        issues.append("scope.deepstack_visual_indexes 必须是 [8,16,24]")

    groups = {group.get("id"): group for group in network.get("groups", [])}
    if set(groups) != {"uit-decoder", "vit-tower"}:
        issues.append(f"整网层组必须且只能是 uit-decoder/vit-tower（实得 {sorted(groups)}）")
    else:
        decoder = groups["uit-decoder"]
        vision = groups["vit-tower"]
        if decoder.get("layer_indices") != list(range(36)):
            issues.append("decoder 层的真实索引必须是 0..35")
        if vision.get("layer_indices") != list(range(27)):
            issues.append("vision block 的真实索引必须是 0..26（独立索引空间）")
        if "Kref>0" not in str(vision.get("execution", "")):
            issues.append("视觉得层组必须标注条件执行（K>0）")
        if "always" not in str(decoder.get("execution", "")):
            issues.append("decoder 层组必须标注总是执行")

    layer_order = network.get("layer_order") or []
    if [entry.get("index") for entry in layer_order] != list(range(36)):
        issues.append("layer_order 必须是 36 层真实顺序 0..35")
    if any(entry.get("group_id") != "uit-decoder" for entry in layer_order):
        issues.append("layer_order 只能包含 decoder 层（视觉塔是并行条件分支，不是 36 层的一部分）")

    # 位置/掩码输入必须存在。
    for tid in ("n_position_ids", "n_token_types", "n_timestep", "n_vinputs"):
        if tid not in tensors:
            issues.append(f"整网缺少边界张量 {tid}")
    pos = tensors.get("n_position_ids", {}).get("shape", [])
    if pos[:1] != [3]:
        issues.append("n_position_ids 的第一轴必须是 3（MRoPE 三轴）")
    return issues
