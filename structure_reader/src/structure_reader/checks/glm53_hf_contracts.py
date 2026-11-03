"""GLM-5.3（glm_moe_dsa）HF 阅读数据语义契约检查（静态，不运行模型）。

检查对象是 ``extract/glm53_hf.py`` / ``extract/glm53_network.py`` 产出的文档；
任何一条不通过即返回问题串，由构建层具名拒绝（不静默修正、不发布检查未过的候选）。

与 DeepSeek 的 checks 分开：这里的语义锚点是 GLM-5.3 的真实实现事实，不能把
mHC/压缩器/Hash 路由/单 KV 头等 DeepSeek 机制当成通用规则；反过来，如果文档里
出现那些机制，本检查会拒绝（说明误抄了别的模型）。

覆盖：
1. 身份：官方仓库 revision、model_type、实现路径、层数 78。
2. 骨架/数据流闭合（复用 checks/hf_contract_core）。
3. 参数事实：192+64=256、V=256、q_lora=2048、kv_lora=512、indexer 32×128、
   index_topk=2048、256 专家选 8、Es=2048、Is=12288、scaling=1/16、两套 eps、
   theta=8e6。
4. Attention 链顺序与缓存语义：q 低秩→部分交错 RoPE；KV 只缓存压缩 latent
   （512+64），缓存后 kv_b 展开；K=cat(nope,rope)；softmax 前加稀疏 mask。
5. DSA：full 层 indexer 的算子/缓存/带 bias LayerNorm；shared 层不得有 indexer
   算子/缓存，必须消费 prev_topk_indices。
6. MoE：sigmoid + correction bias + top-8 + 归一 + ×2.5；单份专家模板；共享专家
   并行相加；dense 层不得有 MoE 算子。
7. 残差/边界：两处普通残差；没有 mHC/压缩器/滑动窗口等其它模型事实。
8. 场景派生量在具体场景下的取值（空缓存/非零历史/超过 index_topk）。
9. 输入身份完整（SOURCE_FILES 每个键都有实际 SHA256）。
"""
from __future__ import annotations

from structure_reader.checks import hf_contract_core as hf_core
from structure_reader.rules.glm53_hf import MODEL_REPOSITORY, SOURCE_FILES

__all__ = [
    "validate_hf_layer_structure",
    "validate_hf_reading_document",
    "validate_hf_network_overview",
]

_EXPECTED_PARAMS = {
    "H": 6144, "Nh": 64, "Nkv": 64, "Dq": 256, "Dn": 192, "Rk": 64, "Dv": 256,
    "Rq": 2048, "Rkv": 512, "Ih": 32, "Id": 128, "Ki": 2048, "E": 256, "K": 8,
    "Es": 2048, "Is": 12288, "Scl": 2.5, "Ng": 1, "Ktg": 1, "EpsN": 1e-5,
    "EpsA": 1e-6, "ThetaM": 8000000.0, "Vocab": 154880,
}

# 场景 → 派生量期望（独立手算，不读回文档里的值再比较）
_EXPECTED_DERIVATIONS = {
    "prefill-empty": {"L_kv": 8, "L_idx": 8, "L_mask": 8, "K_sel": 8},
    "decode-nonzero": {"L_kv": 4097, "L_idx": 4097, "L_mask": 4097, "K_sel": 2048},
    "decode-long": {"L_kv": 131073, "L_idx": 131073, "L_mask": 131073, "K_sel": 2048},
    "decode-multi": {"L_kv": 4096, "L_idx": 4096, "L_mask": 4096, "K_sel": 2048},
}

_FORBIDDEN_ID_TOKENS = ("mhc", "sink", "sliding", "csa", "hca", "kv_full",
                        "block_bias", "tid2eid", "compress_rate", "overlap",
                        "num_full_windows")


def _ids(document: dict, key: str) -> list[str]:
    return [str(item.get("id", "")) for item in document.get(key, [])]


def _param_map(block: dict) -> dict:
    return {p.get("id"): p for p in block.get("parameters", [])}


def _tensor_map(block: dict) -> dict:
    return {t.get("id"): t for t in block.get("tensors", [])}


def _op_map(block: dict) -> dict:
    return {o.get("id"): o for o in block.get("operators", [])}


def _op_order(block: dict) -> dict:
    return {o.get("id"): index for index, o in enumerate(block.get("operators", []))}


def _check_identity(model: dict, *, require_revision: bool = True) -> list[str]:
    issues = hf_core.check_model_identity(model)
    if model.get("model_key") != "glm_5_3_hf":
        issues.append(f"model.model_key 必须是 glm_5_3_hf: {model.get('model_key')!r}")
    config_path = str(model.get("config_path", ""))
    if "zai-org/GLM-5.3/config.json" not in config_path:
        issues.append(f"model.config_path 必须指向官方 GLM-5.3 config: {config_path!r}")
    if require_revision and model.get("model_revision") != MODEL_REPOSITORY["revision"]:
        issues.append(
            f"model.model_revision 必须是官方 revision {MODEL_REPOSITORY['revision']}: "
            f"{model.get('model_revision')!r}"
        )
    return issues


def _check_no_foreign_facts(block: dict, label: str) -> list[str]:
    """拒绝把其它模型的机制写进 GLM-5.3 结构（id 级别，不误伤说明文字）。"""
    issues: list[str] = []
    for kind in ("operators", "tensors"):
        for item in block.get(kind, []):
            item_id = str(item.get("id", "")).lower()
            for token in _FORBIDDEN_ID_TOKENS:
                if token in item_id:
                    issues.append(f"{label}: {kind} id {item.get('id')!r} 含其它模型机制 {token!r}")
    return issues


def _check_param_facts(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    params = _param_map(block)
    for pid, expected in _EXPECTED_PARAMS.items():
        if pid not in params:
            issues.append(f"{label}: 缺少参数 {pid}")
            continue
        value = params[pid].get("value")
        if isinstance(expected, float):
            if not isinstance(value, (int, float)) or abs(float(value) - expected) > 1e-12:
                issues.append(f"{label}: 参数 {pid}={value!r}，期望 {expected}")
        elif value != expected:
            issues.append(f"{label}: 参数 {pid}={value!r}，期望 {expected}")
    # 关键派生关系：qk_head_dim = qk_nope + qk_rope；head_dim 不按 192 解释
    dq, dn, rk = (params.get(k, {}).get("value") for k in ("Dq", "Dn", "Rk"))
    if all(isinstance(v, int) for v in (dq, dn, rk)) and dq != dn + rk:
        issues.append(f"{label}: Dq={dq} != Dn+Rk={dn}+{rk}（qk_head_dim 派生错误）")
    if rk != 64:
        issues.append(f"{label}: Rk={rk!r}；运行期 head_dim 必须是 qk_rope_head_dim=64，"
                      "不能按根配置保存值 192 解释 RoPE")
    # 场景参数保留符号
    for pid in ("seqlen", "kvcachelen"):
        if params.get(pid, {}).get("value") is not None:
            issues.append(f"{label}: 结构参数 {pid} 必须保留符号（value=None）")
    # 数据相关量不得默认 0
    for pid in ("N_hit", "N_e"):
        if pid in params and params[pid].get("value") is not None:
            issues.append(f"{label}: 数据相关参数 {pid} 不得预填数值（当前 {params[pid].get('value')!r}）")
    return issues


def _check_derivations(document: dict) -> list[str]:
    """派生式语义 + 具体场景取值（独立期望）。"""
    from structure_reader.scenario_evaluator import ExpressionError, check_expression

    issues: list[str] = []
    block = document.get("block", {})
    derivations = {d.get("id"): d for d in block.get("derivations", [])}
    for did in ("L_kv", "L_idx", "L_mask", "K_sel", "T_tok"):
        if did not in derivations:
            issues.append(f"缺少派生量 {did}")
            continue
        expression = str(derivations[did].get("expression", ""))
        try:
            check_expression(expression)
        except ExpressionError as exc:
            issues.append(f"派生量 {did} 表达式非法: {exc}")
        if "min(kvcachelen" in expression or " W" in f" {expression}" or "%" in expression:
            issues.append(f"派生量 {did} 混入了滑动窗口/压缩语义: {expression!r}")
    lkv_expr = str(derivations.get("L_kv", {}).get("expression", ""))
    if lkv_expr.replace(" ", "") != "kvcachelen+seqlen":
        issues.append(f"L_kv 必须是 kvcachelen + seqlen（无窗口/压缩）: {lkv_expr!r}")
    if str(derivations.get("K_sel", {}).get("expression", "")) != "min(Ki, L_idx)":
        issues.append("K_sel 必须是 min(Ki, L_idx)")

    evaluated = document.get("scenario_derivations", {})
    for scenario_id, expected in _EXPECTED_DERIVATIONS.items():
        rows = evaluated.get(scenario_id)
        if not isinstance(rows, list):
            issues.append(f"缺少场景 {scenario_id} 的派生求值")
            continue
        by_id = {row.get("id"): row for row in rows}
        for did, value in expected.items():
            row = by_id.get(did)
            if row is None:
                issues.append(f"场景 {scenario_id} 缺少派生结果 {did}")
            elif row.get("state") != "determined" or row.get("value") != value:
                issues.append(
                    f"场景 {scenario_id} 的 {did} 期望 {value}，实得 "
                    f"{row.get('value')!r}（{row.get('state')}）")
    # 符号场景必须如实未知
    symbolic = {row.get("id"): row
                for row in (evaluated.get("prefill-symbolic") or [])}
    if symbolic.get("L_kv", {}).get("state") != "unknown":
        issues.append("符号 seqlen 场景下 L_kv 必须未知（不默认 0）")
    return issues


def _check_attention_chain(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    order = _op_order(block)
    tensors = _tensor_map(block)
    params = _param_map(block)

    def need(oid: str) -> bool:
        if oid not in ops:
            issues.append(f"{label}: 缺少算子 {oid}")
            return False
        return True

    def before(first: str, second: str, note: str) -> None:
        if first in order and second in order and order[first] >= order[second]:
            issues.append(f"{label}: 顺序错误：{note}（{first} 必须在 {second} 之前）")

    chain = [
        "g_input_norm", "g_q_a_proj", "g_q_a_norm", "g_q_b_proj", "g_q_view",
        "g_q_split", "g_q_rope", "g_q_cat", "g_kv_a_proj", "g_kv_a_split",
        "g_kv_a_norm", "g_k_pass_view", "g_kv_rope", "g_cache_update", "g_kv_b_proj",
        "g_kv_view", "g_kv_split", "g_k_rot_expand", "g_key_cat", "g_qk", "g_mask_add",
        "g_softmax", "g_av", "g_attn_transpose", "g_attn_reshape", "g_o_proj",
        "g_res_attn", "g_post_norm",
    ]
    missing = [oid for oid in chain if not need(oid)]
    if missing:
        return issues
    for first, second in zip(chain, chain[1:]):
        before(first, second, f"{first} → {second}")

    # Q 低秩链：q_a_norm 在 q_b_proj 之前，q_rope 只消费 rope 段
    if "g_q_residual" not in ops["g_q_b_proj"].get("inputs", []):
        issues.append(f"{label}: q_b_proj 必须消费 q_a_layernorm 输出（q_residual）")
    if "g_q_rot" not in ops["g_q_rope"].get("inputs", []):
        issues.append(f"{label}: Q RoPE 必须只作用于 split 出的 q_rot")
    if ops["g_q_rope"].get("op") != "rope-interleaved":
        issues.append(f"{label}: Q RoPE 必须是交错布局（rope-interleaved）")
    if ops["g_kv_rope"].get("op") != "rope-interleaved":
        issues.append(f"{label}: K RoPE 必须是交错布局（rope-interleaved）")
    for tid in ("g_cos", "g_sin"):
        if tensors.get(tid, {}).get("shape", [None])[-1] != "Rk":
            issues.append(f"{label}: {tid} 的尾维必须是 Rk=64（不是 head_dim 192）")

    # 缓存语义：update 存的是 latent（512/64），不是展开后的多头 K/V
    cache_update = ops["g_cache_update"]
    outs = cache_update.get("outputs", [])
    if sorted(outs) != ["g_cache_k_pass_post", "g_cache_k_rot_post"]:
        issues.append(f"{label}: cache_update 输出必须是压缩 latent 缓存: {outs}")
    for tid, axis in (("g_cache_k_pass_pre", "Rkv"), ("g_cache_k_rot_pre", "Rk")):
        tensor = tensors.get(tid, {})
        if tensor.get("kind") != "cache":
            issues.append(f"{label}: {tid} 必须是 cache 张量")
        if tensor.get("shape", [None])[-1] != axis:
            issues.append(f"{label}: {tid} 的尾维必须是 {axis}（缓存压缩 latent）")
        if "Nh" in tensor.get("shape", []):
            issues.append(f"{label}: {tid} 不应带 Nh 轴（缓存不是展开后的多头 K/V）")
    if "g_k_pass_4d" not in cache_update.get("inputs", []):
        issues.append(f"{label}: cache_update 必须消费单头 view 后的 k_pass（4D），"
                      "不能把 3D latent 直接当 4D 缓存输入")
    if "g_k_pass_normed" not in ops.get("g_k_pass_view", {}).get("inputs", []):
        issues.append(f"{label}: 归一化后的 k_pass 必须先显式 view 成 [B,1,seqlen,Rkv]")
    if tensors.get("g_k_pass_normed", {}).get("shape") != ["B", "seqlen", "Rkv"]:
        issues.append(f"{label}: RMSNorm 输出的 k_pass 必须是 3D [B,seqlen,Rkv]")
    if tensors.get("g_k_pass_4d", {}).get("shape") != ["B", "1", "seqlen", "Rkv"]:
        issues.append(f"{label}: k_pass 的缓存输入必须是 4D [B,1,seqlen,Rkv]")
    if "g_cache_k_pass_post" not in ops["g_kv_b_proj"].get("inputs", []):
        issues.append(f"{label}: kv_b_proj 必须消费缓存返回的 latent（展开发生在缓存之后）")
    before("g_cache_update", "g_kv_b_proj", "缓存写入先于 kv_b 展开")
    if "g_k_nope" not in ops["g_key_cat"].get("inputs", []) or \
            "g_k_rot_expanded" not in ops["g_key_cat"].get("inputs", []):
        issues.append(f"{label}: key 必须是 cat(k_nope, k_rot)")
    if tensors.get("g_k_nope", {}).get("shape", [None])[-1] != "Dn":
        issues.append(f"{label}: k_nope 尾维必须是 Dn=192")
    if tensors.get("g_v", {}).get("shape", [None])[-1] != "Dv":
        issues.append(f"{label}: value 尾维必须是 Dv=256")

    # 注意力核心：QK×scaling → +稀疏 mask → softmax → AV
    if "g_key" not in ops["g_qk"].get("inputs", []):
        issues.append(f"{label}: QK 未消费 key")
    if "Dq" not in ops["g_qk"].get("parameter_refs", []):
        issues.append(f"{label}: QK 缩放必须引用 Dq（1/sqrt(256)=1/16）")
    if params.get("Dq", {}).get("value") != 256:
        issues.append(f"{label}: Dq 必须是 256 才能得到 1/16 缩放")
    if "1/16" not in str(ops["g_qk"].get("note", "")):
        issues.append(f"{label}: QK 说明必须写明 scaling=1/16")
    if "g_sparse_mask" not in ops["g_mask_add"].get("inputs", []):
        issues.append(f"{label}: 注意力必须加 DSA 稀疏 mask（不能只做普通因果注意力）")
    if "g_attn_masked" not in ops["g_softmax"].get("inputs", []):
        issues.append(f"{label}: softmax 必须消费已加 mask 的 logits")
    if "g_v" not in ops["g_av"].get("inputs", []):
        issues.append(f"{label}: AV 未消费 value")
    before("g_softmax", "g_av", "softmax 先于 AV")
    if "g_attn_flat" not in ops["g_o_proj"].get("inputs", []):
        issues.append(f"{label}: o_proj 必须消费拼头后的 attn_flat")
    if tensors.get("g_attn_flat", {}).get("shape", [None])[-1] != "Nh×Dv":
        issues.append(f"{label}: attn_flat 尾维必须是 Nh×Dv=64×256")
    return issues


def _check_residual(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    if ops.get("g_res_attn", {}).get("op") != "add" or \
            ops.get("g_res_ffn", {}).get("op") != "add":
        issues.append(f"{label}: 两处残差都必须是普通 add（GLM-5.3 没有 mHC 混合）")
    if "g_hidden_in" not in ops["g_res_attn"].get("inputs", []):
        issues.append(f"{label}: Attention 残差必须消费归一化前的层输入")
    if "g_hidden_mid" not in ops["g_res_ffn"].get("inputs", []) or \
            "g_mlp_out" not in ops["g_res_ffn"].get("inputs", []):
        issues.append(f"{label}: FFN/MoE 残差必须是 hidden_mid + mlp_out")
    extra_adds = [oid for oid, op in ops.items()
                  if op.get("op") == "add" and oid not in ("g_res_attn", "g_res_ffn",
                                                           "g_moe_bias_add", "g_moe_merge")]
    if extra_adds:
        issues.append(f"{label}: 出现未预期的额外加法算子 {extra_adds}")
    # post_attention_layernorm 的输入必须是第一次残差之后
    if "g_hidden_mid" not in ops["g_post_norm"].get("inputs", []):
        issues.append(f"{label}: post_attention_layernorm 必须消费第一次残差后的张量")
    return issues


def _check_indexer(block: dict, label: str, indexer_type: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    tensors = _tensor_map(block)
    order = _op_order(block)
    indexer_ops = sorted(oid for oid in ops if oid.startswith("g_ix_"))
    has_indexer_cache = any(tid.startswith("g_cache_indexer_keys") for tid in tensors)
    topk = ops.get("g_ix_topk_indices")
    if indexer_type == "full":
        required = {
            "g_ix_q_proj", "g_ix_q_view", "g_ix_q_split", "g_ix_k_proj",
            "g_ix_k_norm", "g_ix_k_view", "g_ix_k_split", "g_ix_rope",
            "g_ix_q_cat", "g_ix_k_cat", "g_ix_cache_update", "g_ix_scores",
            "g_ix_relu", "g_ix_weights", "g_ix_sum", "g_ix_mask", "g_ix_topk",
        }
        missing = sorted(required - set(indexer_ops))
        if missing:
            issues.append(f"{label}: full indexer 缺少算子 {missing}")
        if not has_indexer_cache:
            issues.append(f"{label}: full indexer 必须有 [B,Lidx,Id] 键缓存")
        k_norm = ops.get("g_ix_k_norm", {})
        if k_norm.get("op") != "layernorm":
            issues.append(f"{label}: indexer k_norm 必须是 LayerNorm（不是 RMSNorm）")
        if sorted(k_norm.get("weights", [])) != ["g_ix_k_norm_bias", "g_ix_k_norm_weight"]:
            issues.append(f"{label}: indexer k_norm 必须同时有 weight 与 bias")
        if "g_q_residual" not in ops.get("g_ix_q_proj", {}).get("inputs", []):
            issues.append(f"{label}: indexer Q 必须复用 q_residual（q_a_layernorm 输出）")
        if "g_attn_normed" not in ops.get("g_ix_k_proj", {}).get("inputs", []):
            issues.append(f"{label}: indexer K 必须消费 Attention 的 hidden_states")
        if "g_attention_mask" not in ops.get("g_ix_mask", {}).get("inputs", []):
            issues.append(f"{label}: indexer top-k 必须先加因果 mask")
        if tensors.get("g_ix_topk_indices", {}).get("shape", [None])[-1] != "Ksel":
            issues.append(f"{label}: top-k 输出尾维必须是 Ksel")
        if "g_ix_topk_indices" not in ops.get("g_index_mask_scatter", {}).get("inputs", []):
            issues.append(f"{label}: 稀疏 mask 必须消费本层 indexer 的 top-k")
        # 缓存写入必须在打分之前
        if order.get("g_ix_cache_update", 0) >= order.get("g_ix_scores", 0):
            issues.append(f"{label}: indexer 缓存追加必须先于打分")
    else:
        if indexer_ops:
            issues.append(f"{label}: shared 层不得实例化 indexer 算子: {indexer_ops}")
        if has_indexer_cache:
            issues.append(f"{label}: shared 层不得有 indexer 键缓存")
        if "g_prev_topk_indices" not in ops.get("g_index_mask_scatter", {}).get("inputs", []):
            issues.append(f"{label}: shared 层必须消费上一 full 层的 prev_topk_indices")
        if "g_prev_topk_indices" not in tensors:
            issues.append(f"{label}: shared 层缺少 prev_topk_indices 边界张量")
        else:
            note = str(tensors["g_prev_topk_indices"].get("note", ""))
            if "上一" not in note and "复用" not in note:
                issues.append(f"{label}: prev_topk_indices 必须说明复用上一 full 层")
    return issues


def _check_mask(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    tensors = _tensor_map(block)
    scatter = ops.get("g_index_mask_scatter", {})
    if scatter.get("op") != "scatter+unsqueeze":
        issues.append(f"{label}: top-k → 稀疏掩码必须是 scatter 后 unsqueeze(1)")
    index_mask = tensors.get("g_index_mask", {})
    if index_mask.get("shape") != ["B", "1", "seqlen", "Lidx"]:
        issues.append(f"{label}: index_mask 必须是 unsqueeze 后的 4D [B,1,seqlen,Lidx]")
    note = str(index_mask.get("note", "")) + " " + str(scatter.get("note", ""))
    if "True" not in note or "屏蔽" not in note:
        issues.append(f"{label}: index_mask 必须说明 True=屏蔽（选中位置为 False），"
                      "不能写成 True=选中")
    if "False" not in note or "选中" not in note:
        issues.append(f"{label}: index_mask 必须说明 top-k 选中位置 scatter 成 False")
    sparse = tensors.get("g_sparse_mask", {})
    if sparse.get("shape") != ["B", "1", "seqlen", "Lidx"]:
        issues.append(f"{label}: 稀疏 mask 必须是 4D [B,1,seqlen,Lidx]")
    apply = ops.get("g_sparse_mask_apply", {})
    if apply.get("op") != "masked_fill":
        issues.append(f"{label}: 因果 mask 与稀疏掩码合并必须是 masked_fill")
    if sorted(apply.get("inputs", [])) != ["g_attention_mask", "g_index_mask"]:
        issues.append(f"{label}: 稀疏 mask 必须由因果 mask 与 index_mask 合并")
    return issues


def _check_moe(block: dict, label: str, moe_type: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    tensors = _tensor_map(block)
    order = _op_order(block)
    moe_ops = sorted(oid for oid in ops if oid.startswith("g_moe_"))
    if moe_type == "dense":
        if moe_ops:
            issues.append(f"{label}: dense 层不得出现 MoE 算子: {moe_ops}")
        for oid in ("g_dense_gate", "g_dense_up", "g_dense_act", "g_dense_down"):
            if oid not in ops:
                issues.append(f"{label}: dense 层缺少 {oid}")
        if "g_mlp_out" not in ops.get("g_dense_down", {}).get("outputs", []):
            issues.append(f"{label}: dense MLP 输出必须是 g_mlp_out")
        return issues

    required = {
        "g_moe_flatten", "g_moe_router", "g_moe_sigmoid", "g_moe_bias_add",
        "g_moe_group_score", "g_moe_group_topk", "g_moe_group_scatter",
        "g_moe_group_expand", "g_moe_group_reshape",
        "g_moe_group_masked_fill", "g_moe_topk", "g_moe_weight_gather",
        "g_moe_weight_norm", "g_moe_weight_scale", "g_moe_expert_mask",
        "g_moe_expert_hit", "g_moe_expert_gather_indices",
        "g_moe_select_tokens", "g_moe_expert_gate_up", "g_moe_expert_chunk",
        "g_moe_expert_act", "g_moe_expert_down", "g_moe_expert_weight",
        "g_moe_result_init", "g_moe_index_add", "g_moe_routed_view",
        "g_moe_shared_gate", "g_moe_shared_up", "g_moe_shared_act",
        "g_moe_shared_down", "g_moe_merge",
    }
    missing = sorted(required - set(moe_ops))
    if missing:
        issues.append(f"{label}: MoE 缺少算子 {missing}")
    if ops.get("g_moe_sigmoid", {}).get("op") != "sigmoid":
        issues.append(f"{label}: 路由打分必须是 sigmoid（不是 softmax/sqrt-softplus）")
    if order.get("g_moe_bias_add", 0) >= order.get("g_moe_topk", 0):
        issues.append(f"{label}: correction bias 必须加在 top-k 选择之前")
    if order.get("g_moe_weight_norm", 0) >= order.get("g_moe_weight_scale", 0):
        issues.append(f"{label}: 归一化必须在 routed_scaling_factor 缩放之前")
    if "K" not in ops.get("g_moe_topk", {}).get("parameter_refs", []):
        issues.append(f"{label}: top-k 必须引用 K（每 token 选择数）")
    if "g_topk_indices" not in ops.get("g_moe_weight_gather", {}).get("inputs", []):
        issues.append(f"{label}: 路由权重必须按选中的索引 gather")
    # 组掩码必须保留 scatter→expand→reshape，最终 [Ttok,E]
    for oid, expected in (("g_group_mask", ["Ttok", "Ng"]),
                          ("g_group_mask_expanded", ["Ttok", "Ng", "E÷Ng"]),
                          ("g_score_mask", ["Ttok", "E"])):
        if tensors.get(oid, {}).get("shape") != expected:
            issues.append(f"{label}: {oid} 形状必须是 {expected}，"
                          f"实得 {tensors.get(oid, {}).get('shape')!r}")
    for first, second in (("g_moe_group_scatter", "g_moe_group_expand"),
                          ("g_moe_group_expand", "g_moe_group_reshape"),
                          ("g_moe_group_reshape", "g_moe_group_masked_fill")):
        if order.get(first, 0) >= order.get(second, 0):
            issues.append(f"{label}: 组掩码顺序必须 {first} → {second}")
    if "g_score_mask" not in ops.get("g_moe_group_masked_fill", {}).get("inputs", []):
        issues.append(f"{label}: masked_fill 必须消费 reshape 后的 [Ttok,E] score_mask")
    # MoE token 分配：完整 one-hot 矩阵 + 两个真实索引贯通
    if tensors.get("g_expert_mask", {}).get("shape") != ["E+1", "K", "Ttok"]:
        issues.append(f"{label}: expert_mask 必须是完整 [E+1,K,Ttok] 分配矩阵")
    if ops.get("g_moe_expert_mask", {}).get("op") != "one_hot+permute":
        issues.append(f"{label}: 专家分配必须是 one_hot(top_k,E+1).permute(2,1,0)")
    if ops.get("g_moe_expert_hit", {}).get("op") != "sum+nonzero":
        issues.append(f"{label}: 命中专家集合必须由完整分配矩阵 sum+nonzero 得到")
    gather = ops.get("g_moe_expert_gather_indices", {})
    if gather.get("op") != "where":
        issues.append(f"{label}: 必须用 torch.where(expert_mask[expert_idx]) 取两个索引")
    if sorted(gather.get("outputs", [])) != ["g_expert_token_idx", "g_expert_topk_pos"]:
        issues.append(f"{label}: where 必须同时输出 token_idx 与 top_k_pos")
    if tensors.get("g_expert_token_idx", {}).get("shape") != ["N_e"] or \
            tensors.get("g_expert_topk_pos", {}).get("shape") != ["N_e"]:
        issues.append(f"{label}: token_idx/top_k_pos 必须是 [N_e] 数据相关索引")
    if "g_expert_token_idx" not in ops.get("g_moe_select_tokens", {}).get("inputs", []):
        issues.append(f"{label}: gather token 行必须用 token_idx（不能只用命中集合）")
    weight_inputs = ops.get("g_moe_expert_weight", {}).get("inputs", [])
    for required in ("g_expert_token_idx", "g_expert_topk_pos", "g_topk_weights"):
        if required not in weight_inputs:
            issues.append(f"{label}: 专家加权必须用 top_k_weights[token_idx, top_k_pos]，"
                          f"缺少 {required}")
    index_inputs = ops.get("g_moe_index_add", {}).get("inputs", [])
    if "g_expert_token_idx" not in index_inputs:
        issues.append(f"{label}: index_add 必须按 token_idx 回写")
    if "g_routed_init" not in index_inputs:
        issues.append(f"{label}: index_add 必须有显式 zeros_like 初值张量")
    for first, second in (("g_moe_expert_gather_indices", "g_moe_select_tokens"),
                          ("g_moe_expert_gather_indices", "g_moe_expert_weight"),
                          ("g_moe_expert_weight", "g_moe_index_add")):
        if order.get(first, 0) >= order.get(second, 0):
            issues.append(f"{label}: 专家分配顺序必须 {first} → {second}")
    gate_up = tensors.get("g_moe_experts_gate_up", {})
    if gate_up.get("shape") != ["E", "2Es", "H"]:
        issues.append(f"{label}: 专家 gate_up 形状必须是 [E,2Es,H]（单份模板，不是 E 份子图）")
    down = tensors.get("g_moe_experts_down", {})
    if down.get("shape") != ["E", "H", "Es"]:
        issues.append(f"{label}: 专家 down 形状必须是 [E,H,Es]")
    if "g_routed" not in ops.get("g_moe_merge", {}).get("inputs", []) or \
            "g_shared_out" not in ops.get("g_moe_merge", {}).get("inputs", []):
        issues.append(f"{label}: MoE 汇合必须是 routed + 共享专家")
    if "g_ffn_in" not in ops.get("g_moe_shared_gate", {}).get("inputs", []):
        issues.append(f"{label}: 共享专家必须消费 routed 之前的 residuals（并行支路）")
    if any("hash" in oid for oid in ops):
        issues.append(f"{label}: GLM-5.3 没有 hash 路由，文档出现 hash 算子")
    if any("hash" in tid for tid in tensors):
        issues.append(f"{label}: GLM-5.3 没有 tid2eid 查表，文档出现 hash 张量")
    if tensors.get("g_expert_hit", {}).get("shape") != ["N_hit"]:
        issues.append(f"{label}: 命中专家数必须保留数据相关符号 N_hit")
    return issues


def _check_source_refs(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    for op in block.get("operators", []):
        refs = op.get("source_refs", [])
        if not refs:
            issues.append(f"{label}: 算子 {op.get('id')} 缺少源码定位")
            continue
        for ref in refs:
            if not ref.get("path") or not ref.get("locator"):
                issues.append(f"{label}: 算子 {op.get('id')} 的 source_ref 缺少 path/locator")
    for element in ("parameters", "tensors"):
        for item in block.get(element, []):
            for ref in item.get("source_refs", []):
                if not ref.get("path") or not ref.get("locator"):
                    issues.append(f"{label}: {element} {item.get('id')} 的 source_ref 不完整")
    return issues


def _check_block(block: dict, label: str, *, indexer_type: str, moe_type: str) -> list[str]:
    issues: list[str] = []
    issues.extend(hf_core.check_unit(block, label))
    issues.extend(hf_core.check_flow_units([block]))
    issues.extend(_check_no_foreign_facts(block, label))
    issues.extend(_check_param_facts(block, label))
    issues.extend(_check_attention_chain(block, label))
    issues.extend(_check_residual(block, label))
    issues.extend(_check_mask(block, label))
    issues.extend(_check_indexer(block, label, indexer_type))
    issues.extend(_check_moe(block, label, moe_type))
    issues.extend(_check_source_refs(block, label))
    # 所有 id 以 g_ 前缀命名空间隔离，避免与其它模型事实/展示规则混用
    for element in ("operators", "tensors", "parameters"):
        for item in block.get(element, []):
            if element != "parameters" and not str(item.get("id", "")).startswith("g_"):
                issues.append(f"{label}: {element} id {item.get('id')!r} 未使用 g_ 命名空间")
    # 缓存注记必须写清压缩 latent 与 indexer 缓存
    cache_blob = " ".join(block.get("cache_notes", []))
    for token in ("压缩 latent", "indexer", "kvcachelen", "滑动窗口"):
        if token not in cache_blob:
            issues.append(f"{label}: cache_notes 缺少关键事实 {token!r}")
    return issues


def validate_hf_layer_structure(document: dict) -> list[str]:
    """单层组 Block 结构（hf_layer_structure.v1）的 GLM-5.3 语义检查。"""
    issues: list[str] = []
    if document.get("schema") != "hf_layer_structure.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    issues.extend(_check_identity(document.get("model", {})))
    scope = document.get("scope", {})
    if scope.get("analysis_path") != "eager":
        issues.append(f"scope.analysis_path 必须是 eager: {scope.get('analysis_path')!r}")
    layer_index = scope.get("layer_index")
    if not isinstance(layer_index, int) or layer_index < 0:
        issues.append("scope.layer_index 必须是非负整数")
    if scope.get("attention_type") != "indexed_attention":
        issues.append(f"scope.attention_type 必须是 indexed_attention: {scope.get('attention_type')!r}")
    indexer_type = scope.get("indexer_type")
    if indexer_type not in ("full", "shared"):
        issues.append(f"scope.indexer_type 必须是 full/shared: {indexer_type!r}")
        indexer_type = "full"
    moe_type = scope.get("moe_type")
    if moe_type not in ("dense", "sparse"):
        issues.append(f"scope.moe_type 必须是 dense/sparse: {moe_type!r}")
        moe_type = "sparse"
    if "phases" in document:
        issues.append("结构文档不得包含 phases（prefill/decode 只是场景投影）")
    scenarios = scope.get("scenarios", [])
    ids = [s.get("id") for s in scenarios]
    if len(ids) != len(set(ids)) or not ids:
        issues.append("场景预设 id 缺失或重复")
    for sid in ("prefill-empty", "decode-nonzero"):
        if sid not in ids:
            issues.append(f"缺少必要场景 {sid}")
    for scenario in scenarios:
        if "seqlen" not in scenario or "kvcachelen" not in scenario:
            issues.append(f"场景 {scenario.get('id')} 缺 seqlen/kvcachelen")
    if scope.get("default_scenario") not in ids:
        issues.append("default_scenario 必须是已声明场景之一")
    prefill = next((s for s in scenarios if s.get("id") == "prefill-empty"), {})
    if prefill.get("kvcachelen") != 0:
        issues.append("空缓存 Prefill 场景的 kvcachelen 必须是 0")

    block = document.get("block")
    if not isinstance(block, dict):
        issues.append("block 缺失")
        return issues
    if block.get("layer_index") != layer_index:
        issues.append("block.layer_index 与 scope.layer_index 不一致")
    if block.get("indexer_type") != indexer_type or block.get("moe_type") != moe_type:
        issues.append("block 的 indexer_type/moe_type 与 scope 不一致")
    issues.extend(_check_block(block, "block", indexer_type=indexer_type, moe_type=moe_type))
    issues.extend(_check_derivations(document))
    return issues


def validate_hf_reading_document(document: dict, input_shas: dict[str, str]) -> list[str]:
    """同源 prefill/decode 投影（hf_decoder_reading.v1）的 GLM-5.3 检查。"""
    issues: list[str] = []
    if document.get("schema") != "hf_decoder_reading.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    issues.extend(_check_identity(document.get("model", {})))
    scope = document.get("scope", {})
    if scope.get("analysis_path") != "eager":
        issues.append(f"scope.analysis_path 必须是 eager: {scope.get('analysis_path')!r}")
    if scope.get("attention_type") != "indexed_attention":
        issues.append("scope.attention_type 必须是 indexed_attention")
    indexer_type = scope.get("indexer_type")
    if indexer_type not in ("full", "shared"):
        issues.append(f"scope.indexer_type 必须是 full/shared: {indexer_type!r}")
        indexer_type = "full"
    moe_type = scope.get("moe_type")
    if moe_type not in ("dense", "sparse"):
        issues.append(f"scope.moe_type 必须是 dense/sparse: {moe_type!r}")
        moe_type = "sparse"
    phases = document.get("phases", [])
    if [p.get("id") for p in phases] != ["prefill", "decode"]:
        issues.append("phases 必须依次为 prefill、decode")
    for phase in phases:
        pid = phase.get("id", "?")
        issues.extend(hf_core.check_unit(phase, pid))
        issues.extend(hf_core.check_flow_units([phase]))
        issues.extend(_check_no_foreign_facts(phase, pid))
        issues.extend(_check_attention_chain(phase, pid))
        issues.extend(_check_residual(phase, pid))
        issues.extend(_check_indexer(phase, pid, indexer_type))
        issues.extend(_check_moe(phase, pid, moe_type))
        if phase.get("indexer_type") != indexer_type:
            issues.append(f"{pid}: 投影丢失 indexer_type")
        binding = phase.get("scenario_binding", {})
        if pid == "prefill" and binding.get("kvcachelen") != 0:
            issues.append("prefill 投影的 kvcachelen 必须是 0（空缓存）")
        if pid == "decode" and binding.get("seqlen") != 1:
            issues.append("decode 投影的 seqlen 必须是 1（单 token 预设）")
        if not phase.get("scenario_derivations"):
            issues.append(f"{pid}: 缺少场景派生量")
    issues.extend(hf_core.check_input_identity(input_shas, SOURCE_FILES))
    return issues


def validate_hf_network_overview(document: dict, input_shas: dict[str, str]) -> list[str]:
    """整网总览（hf_network_overview.v1）的 GLM-5.3 检查。"""
    issues: list[str] = []
    if document.get("schema") != "hf_network_overview.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    issues.extend(_check_identity(document.get("model", {})))
    scope = document.get("scope", {})
    if scope.get("analysis_path") != "eager":
        issues.append("scope.analysis_path 必须是 eager")
    if scope.get("entry_class") != "GlmMoeDsaForCausalLM":
        issues.append(f"scope.entry_class 必须是 GlmMoeDsaForCausalLM: {scope.get('entry_class')!r}")
    if scope.get("main_module_class") != "GlmMoeDsaModel":
        issues.append("scope.main_module_class 必须是 GlmMoeDsaModel")
    layer_count = scope.get("layer_count")
    if layer_count != 78:
        issues.append(f"scope.layer_count 必须是 78（主干 0..77）: {layer_count!r}")
        layer_count = 78
    path = document.get("path")
    if not isinstance(path, dict):
        issues.append("path 缺失（整网总览必须提供单条 token→logits 路径）")
        return issues
    if path.get("id") != "network":
        issues.append(f"path.id 必须是 'network': {path.get('id')!r}")
    issues.extend(hf_core.check_unit(path, "network"))
    issues.extend(hf_core.check_flow_units([path]))
    issues.extend(_check_no_foreign_facts(path, "network"))
    ops = _op_map(path)
    order = _op_order(path)
    chain = ["net_embed", "net_position_ids", "net_mask_build", "net_rope",
             "net_layer_loop", "net_final_norm", "net_logits_slice", "net_lm_head"]
    for oid in chain:
        if oid not in ops:
            issues.append(f"network: 缺少算子 {oid}")
    for first, second in zip(chain, chain[1:]):
        if order.get(first, 0) >= order.get(second, 0):
            issues.append(f"network: 顺序错误 {first} 必须在 {second} 之前")
    loop = ops.get("net_layer_loop", {})
    for required in ("cos_rope", "sin_rope", "causal_mask", "past_key_values"):
        if required not in loop.get("inputs", []):
            issues.append(f"network: 层循环未消费 {required}")
    if "Nlay" not in loop.get("parameter_refs", []):
        issues.append("network: 层循环必须引用 Nlay")
    if "g_hidden_in" in ops:
        issues.append("network: 不应出现 Block 级算子")
    if "glm" not in str(document.get("model", {}).get("display_name", "")).lower():
        issues.append("network: 显示名必须标明 GLM-5.3")
    blob = " ".join(scope.get("notes", []))
    for token in ("MTP", "indexed_attention", "单流"):
        if token not in blob:
            issues.append(f"network: scope.notes 缺少关键边界 {token!r}")
    if "视觉" not in blob and "图像" not in blob:
        issues.append("network: scope.notes 未声明视觉/图像路径不在范围")

    groups = document.get("groups", [])
    if not isinstance(groups, list) or len(groups) != 3:
        issues.append(f"groups 必须是三种真实层型组合（实得 {len(groups) if isinstance(groups, list) else '?'}）")
        return issues
    covered: list[int] = []
    for group in groups:
        gid = group.get("id")
        indices = group.get("layer_indices", [])
        if not gid:
            issues.append("层组缺少 id")
        if not indices:
            issues.append(f"层组 {gid} 没有真实层")
        if group.get("count") != len(indices):
            issues.append(f"层组 {gid} count 与 layer_indices 长度不符")
        if group.get("representative_layer_index") not in indices:
            issues.append(f"层组 {gid} 代表层不属于该组")
        if group.get("indexer_type") not in ("full", "shared"):
            issues.append(f"层组 {gid} indexer_type 非法")
        covered.extend(indices)
    if sorted(covered) != list(range(layer_count)) or len(covered) != len(set(covered)):
        issues.append(f"三组未恰好覆盖 0..{layer_count - 1}（实得 {len(covered)} 项）")
    first_indices = [min(group.get("layer_indices") or [layer_count]) for group in groups]
    if first_indices != sorted(first_indices):
        issues.append("层组顺序必须按各组首层索引递增")
    expected_ids = {group.get("id") for group in groups}
    if expected_ids != {"full-dense", "shared-moe", "full-moe"}:
        issues.append(f"层组 id 必须是 full-dense/shared-moe/full-moe: {sorted(expected_ids)}")

    order_entries = document.get("layer_order", [])
    if [entry.get("index") for entry in order_entries] != list(range(layer_count)):
        issues.append("layer_order 必须逐一列出 0..77 的真实层顺序")
    group_of = {index: group["id"] for group in groups for index in group.get("layer_indices", [])}
    indexer_of = {index: group["indexer_type"] for group in groups
                  for index in group.get("layer_indices", [])}
    for entry in order_entries:
        index = entry.get("index")
        if entry.get("group_id") != group_of.get(index):
            issues.append(f"layer_order[{index}] 的组与 groups 归属不符")
        if entry.get("indexer_type") != indexer_of.get(index):
            issues.append(f"layer_order[{index}] 的 indexer_type 与 groups 归属不符")
        if entry.get("attention_type") != "indexed_attention":
            issues.append(f"layer_order[{index}] 的 attention_type 必须是 indexed_attention")

    full_from_groups = sorted(index for group in groups if group.get("indexer_type") == "full"
                              for index in group.get("layer_indices", []))
    scope_full = sorted(scope.get("indexer_full_layers", []))
    if full_from_groups != scope_full:
        issues.append("scope.indexer_full_layers 与层组 full 层集合不一致")
    if len(full_from_groups) != 21:
        issues.append(f"full indexer 层数应为 21，实得 {len(full_from_groups)}")
    if scope.get("dense_layers") != [0, 1, 2]:
        issues.append(f"dense 层必须是 0/1/2: {scope.get('dense_layers')!r}")
    shared_from_groups = sorted(index for group in groups if group.get("indexer_type") == "shared"
                                for index in group.get("layer_indices", []))
    if shared_from_groups != sorted(scope.get("indexer_shared_layers", [])):
        issues.append("scope.indexer_shared_layers 与层组 shared 层集合不一致")
    if len(shared_from_groups) != 57:
        issues.append(f"shared indexer 层数应为 57，实得 {len(shared_from_groups)}")

    detail = document.get("detail", {})
    href = detail.get("href")
    if not isinstance(href, str) or not href.endswith(".html"):
        issues.append(f"detail.href 必须是详图 HTML 入口: {href!r}")
    detail_groups = detail.get("groups", [])
    if [g.get("id") for g in detail_groups] != [g.get("id") for g in groups]:
        issues.append("detail.groups 必须与层组同序一一对应")
    for group in detail_groups:
        anchors = group.get("anchors", {})
        for key in ("group", "block", "operators", "tensors"):
            if not anchors.get(key):
                issues.append(f"detail.groups[{group.get('id')}].anchors.{key} 缺失")

    # MTP 边界必须显式存在且说明不执行
    params = _param_map(path)
    mtp = params.get("Nmtp")
    if mtp is None:
        issues.append("network: 缺少 MTP 边界参数 Nmtp")
    else:
        note = str(mtp.get("note", "")) + str(mtp.get("meaning", ""))
        if "MTP" not in note or "不" not in note:
            issues.append("network: MTP 参数必须说明不在主干/不执行")
    issues.extend(hf_core.check_input_identity(input_shas, SOURCE_FILES))
    return issues
