"""Qwen3.8-Flash-Next（qwen4_exp）HF 阅读数据语义契约检查（静态，不运行模型）。

检查对象是 ``extract/qwen38flash_hf.py`` / ``extract/qwen38flash_network.py`` 产出的
文档；任何一条不通过即返回问题串，由构建层具名拒绝（不静默修正、不发布检查未过的
候选）。

与 GLM-5.3 / GLM-5.3-Flash / DeepSeek 的 checks 分开：这里的语义锚点是 Qwen3.8-
Flash-Next 的真实实现事实，不能把 GLM-5.3 的 MLA/DSA/普通残差、GLM-5.3-Flash 的
KDA/mHC/NoPE、DeepSeek 的 CSA/HCA/滑窗/Hash，或名称相近的 Qwen3-Next/Qwen3.5/
Qwen3-MoE 当成通用规则。反过来，如果本模型文档里出现那些机制，本检查会拒绝。

覆盖：
1. 身份：官方 Qwen3.8-Flash-Next revision、model_type/architectures、text_config、
   实现路径、48 层。
2. 骨架/数据流闭合（复用 checks/hf_contract_core）。
3. 参数事实：H=2560、48 层、GDN 16 QK/48 V 头×128、conv 4；QSA 24 Q/2 KV×256、
   partial RoPE 64/256；indexer 4 Q/1 KV×128、budget 2048、compress 4、block 512；
   Gated Residual 4 路、lowrank 320；PLE 16 头×160、vocab base 20M、conv 9/dilation 3；
   MoE 512 选 10、专家 640、共享 640；eps=1e-6。
4. 层归属：三组 gdn-moe(35)/gdn-ple-moe(1)/qsa-moe(12) 恰好覆盖 48 层；
   QSA 层恰为 3,7,11,...,47；PLE 层 0-based 1。
5. GDN：q/k/v 合并投影 → 短卷积 + conv state → beta/decay → q/k repeat/L2Norm/缩放 →
   gated delta-rule 扫描（chunk/recurrent）+ recurrent state 写回 → Gated RMSNorm →
   o_proj；没有 RoPE/indexer/attention/普通残差；scan note 必须先衰减再算误差且方向 A^T k。
6. QSA：q/gate 合并投影、q/k RMSNorm、partial RoPE 64 维；KV 2 头 + repeat_kv；
   indexer 逐 token raw key 缓存、micro-block 均值池化、block key RoPE、
   relu(q·k).sum/√128 打分、块 top-k（≤512）+ tail + pad 2051、True=可见 mask；
   eager QK/softmax/AV + 输出 sigmoid 门；没有 mHC/Sinkhorn/KDA/DSA。
7. Gated Residual：attn/ffn 两站点都有分组 norm → 低秩读门 [Hc,H] → 加权 mean →
   写门 [Hc] → 展开加到原 4 路流；不是普通 add 残差，也不是 mHC/Sinkhorn。
8. PLE：只在 PLE 组出现；16 个哈希头、160 维、上下文 [B,2]、dilation-3 短卷积 state 9；
   非 PLE 组不得出现 PLE 算子。
9. MoE：softmax 路由 + top-10 + 归一；没有 correction bias / routed scaling factor /
   SwiGLU clamp；共享专家有 sigmoid 标量门；单份专家模板。
10. 场景派生量在具体场景下的取值（空缓存/非零历史/超过 2048/多 token）。
11. 整网：token→logits；4 路 repeat；mixer 读门加权 mean；**没有 Final RMSNorm**；
   没有视觉/MTP 执行；48 层真实顺序。
12. 输入身份完整（SOURCE_FILES 每个键都有实际 SHA256）。
"""
from __future__ import annotations

from structure_reader.checks import hf_contract_core as hf_core
from structure_reader.rules.qwen38flash_hf import (
    MODEL_REPOSITORY,
    SOURCE_FILES,
    gdn_layer_indices,
    ple_layer_indices,
    qsa_layer_indices,
)

__all__ = [
    "validate_hf_layer_structure",
    "validate_hf_reading_document",
    "validate_hf_network_overview",
]

_EXPECTED_COMMON = {
    "H": 2560, "Hc": 4, "Rhc": 320,
    "E": 512, "K": 10, "Es": 640, "Ess": 640, "EpsN": 1e-6,
}

_EXPECTED_GDN = {
    "NkH": 16, "NvH": 48, "Dk": 128, "DvH": 128, "Kconv": 4,
    "Gmul": 3, "Dqkv": 10240, "Dz": 6144,
}

_EXPECTED_QSA = {
    "Nh": 24, "Nkv": 2, "Dh": 256, "Drot": 64, "Pfrac": 0.25,
    "Theta": 10000000.0, "Nrep": 12, "Dqw": 6144, "Dkv": 512,
    "Iqh": 4, "Ikh": 1, "Id": 128, "Ki": 2048, "Kc": 4, "Btop": 512,
}

_EXPECTED_PLE = {
    "Ngm": 3, "Hng": 8, "Ngh": 16, "Dgh": 160, "Dple": 2560,
    "Kple": 4, "Dild": 3, "Lpconv": 9, "Lctx": 2,
    "Vng": 20000000, "Eos": 248044,
}

# 场景 → 派生量期望（独立手算，不读回文档里的值再比较）。
_QSA_SCENARIO_EXPECTED = {
    "prefill-empty": {"L_kv": 8, "L_idx": 8, "N_pool": 2, "B_sel": 2,
                      "K_sel_last": 8, "K_out": 2051},
    "decode-nonzero": {"L_kv": 4097, "L_idx": 4097, "N_pool": 1024, "B_sel": 512,
                       "K_sel_last": 2049, "K_out": 2051},
    "decode-long": {"L_kv": 131073, "L_idx": 131073, "N_pool": 32768, "B_sel": 512,
                    "K_sel_last": 2049, "K_out": 2051},
    "decode-multi": {"L_kv": 4096, "L_idx": 4096, "N_pool": 1024, "B_sel": 512,
                     "K_sel_last": 2048, "K_out": 2051},
}

_QSA_EXPECTED_FORMULAS = {
    "L_kv": "kvcachelen + seqlen",
    "L_idx": "kvcachelen + seqlen",
    "N_pool": "L_idx // Kc",
    "B_sel": "min(Btop, N_pool)",
    "V_q": "kvcachelen + j + 1",
    "N_q": "V_q // Kc",
    "B_q": "min(Btop, N_q)",
    "R_tail": "V_q % Kc",
    "K_sel": "min(V_q, Ki + (V_q % Kc))",
    "K_sel_last": "min(L_idx, Ki + (L_idx % Kc))",
    "K_out": "Ki + Kc - 1",
    "T_tok": "B * seqlen",
}
_GDN_EXPECTED_FORMULAS = {
    "S_rec": "NvH * Dk * DvH",
    "S_conv": "Dqkv * Kconv",
    "S_ple_conv": "Dqkv * Lpconv",
    "L_ngram_ctx": "Ngm - 1",
    "T_tok": "B * seqlen",
}

_GDN_SCENARIO_EXPECTED = {
    "S_rec": 786432, "S_conv": 40960,
}
_PLE_SCENARIO_EXPECTED = {
    "S_ple_conv": 92160, "L_ngram_ctx": 2,
}

_FORBIDDEN_ALL = ("mhc", "sinkhorn", "kda", "dsa", "csa", "hca", "swa",
                  "tid2eid", "compress_rate", "num_full_windows")
_FORBIDDEN_GDN = ("rope", "rotary", "indexer", "qsa", "attention_mask_4d")
_FORBIDDEN_QSA = ("delta_rule", "conv_state", "gate_lower_bound")


def _ids(document: dict, key: str) -> list[str]:
    return [str(item.get("id", "")) for item in document.get(key, [])]


def _param_map(block: dict) -> dict:
    return {p.get("id"): p for p in block.get("parameters", [])}


def _tensor_map(block: dict) -> dict:
    return {t.get("id"): t for t in block.get("tensors", [])}


def _op_map(block: dict) -> dict:
    return {o.get("id"): o for o in block.get("operators", [])}


def _op_order(block: dict) -> dict:
    return {o.get("id"): i for i, o in enumerate(block.get("operators", []))}


def _check_identity(model: dict, *, require_revision: bool = True) -> list[str]:
    issues = hf_core.check_model_identity(model)
    if model.get("model_key") != "qwen_3_8_flash_next_hf":
        issues.append(f"model.model_key 必须是 qwen_3_8_flash_next_hf: {model.get('model_key')!r}")
    config_path = str(model.get("config_path", ""))
    if "Qwen/Qwen3.8-Flash-Next/config.json" not in config_path:
        issues.append(f"model.config_path 必须指向官方 Qwen3.8-Flash-Next config: {config_path!r}")
    if require_revision and model.get("model_revision") != MODEL_REPOSITORY["revision"]:
        issues.append(
            f"model.model_revision 必须是官方 revision {MODEL_REPOSITORY['revision']}: "
            f"{model.get('model_revision')!r}"
        )
    return issues


def _check_no_foreign_facts(block: dict, label: str, kind: str) -> list[str]:
    """拒绝把其它模型的机制写进本模型结构（id 级别，不误伤说明文字）。"""
    issues: list[str] = []
    forbidden = list(_FORBIDDEN_ALL)
    forbidden += _FORBIDDEN_GDN if kind == "gdn" else _FORBIDDEN_QSA
    for key in ("operators", "tensors"):
        for item in block.get(key, []):
            item_id = str(item.get("id", "")).lower()
            for token in forbidden:
                if token in item_id:
                    issues.append(f"{label}: {key} id {item.get('id')!r} 含其它模型机制 {token!r}")
    return issues


def _check_param_facts(block: dict, label: str, kind: str, *, legacy: bool = False) -> list[str]:
    issues: list[str] = []
    params = _param_map(block)
    expected = dict(_EXPECTED_COMMON)
    if kind == "gdn":
        expected.update(_EXPECTED_GDN)
        if block.get("ple"):
            expected.update(_EXPECTED_PLE)
    else:
        expected.update(_EXPECTED_QSA)
    for pid, want in sorted(expected.items()):
        if pid not in params:
            issues.append(f"{label}: 缺少参数 {pid}")
            continue
        got = params[pid].get("value")
        if isinstance(want, float):
            if not isinstance(got, (int, float)) or abs(float(got) - want) > 1e-12:
                issues.append(f"{label}: 参数 {pid}={got!r}，期望 {want}")
        elif got != want:
            issues.append(f"{label}: 参数 {pid}={got!r}，期望 {want}")
    if kind == "gdn":
        for pid in ("Srec", "Sconv"):
            if pid not in params:
                issues.append(f"{label}: GDN 缺少派生参数 {pid}")
        if block.get("ple"):
            for pid in ("Sple", "Lctx"):
                if pid not in params:
                    issues.append(f"{label}: PLE 缺少派生参数 {pid}")
    else:
        for pid in ("kvcachelen", "Lkv", "Lidx", "Npool", "Bsel", "j", "Vq",
                    "Ksel", "Ksel_last", "Kout"):
            if pid not in params:
                issues.append(f"{label}: QSA 缺少派生参数 {pid}")
        for pid in ("j", "Vq", "Ksel", "Ksel_last"):
            if params.get(pid, {}).get("value") is not None:
                issues.append(f"{label}: 逐 query 参数 {pid} 必须保留符号（value=None）")
        j_domain = params.get("j", {}).get("domain", {})
        if (j_domain.get("kind") != "integer" or j_domain.get("min") != 0
                or j_domain.get("max_symbol") != "seqlen"):
            issues.append(f"{label}: j 必须声明 integer 域 0<=j<seqlen（导出守卫依据）")
    if not legacy:
        for pid in ("seqlen",):
            if params.get(pid, {}).get("value") is not None:
                issues.append(f"{label}: 结构参数 {pid} 必须保留符号（value=None）")
    if not legacy and kind == "qsa" and params.get("kvcachelen", {}).get("value") is not None:
        issues.append(f"{label}: QSA 结构参数 kvcachelen 必须保留符号（value=None）")
    for pid in ("N_hit", "N_e"):
        if pid in params and params[pid].get("value") is not None:
            issues.append(f"{label}: 数据相关参数 {pid} 不得预填数值")
    return issues


def _check_derivations(document: dict, kind: str, *, ple: bool = False) -> list[str]:
    from structure_reader.scenario_evaluator import ExpressionError, check_expression

    issues: list[str] = []
    block = document.get("block", {})
    derivations = {d.get("id"): d for d in block.get("derivations", [])}
    expected_formulas = _GDN_EXPECTED_FORMULAS if kind == "gdn" else _QSA_EXPECTED_FORMULAS
    if kind == "gdn":
        required = ("S_rec", "S_conv", "T_tok")
        expected = dict(_GDN_SCENARIO_EXPECTED)
        if ple:
            required += ("S_ple_conv", "L_ngram_ctx")
            expected.update(_PLE_SCENARIO_EXPECTED)
        for did in ("L_kv", "L_idx", "K_sel", "B_sel"):
            if did in derivations:
                issues.append(f"GDN 层不应有 QSA 派生量 {did}")
    else:
        required = ("L_kv", "L_idx", "N_pool", "B_sel", "V_q", "K_sel",
                    "K_sel_last", "K_out", "T_tok")
        expected = _QSA_SCENARIO_EXPECTED
    for did in required:
        entry = derivations.get(did)
        if entry is None:
            issues.append(f"缺少派生量 {did}")
            continue
        try:
            check_expression(str(entry.get("expression", "")))
        except ExpressionError as exc:
            issues.append(f"派生量 {did} 表达式不在白名单: {exc}")
        expected_formula = expected_formulas.get(did)
        actual_formula = " ".join(str(entry.get("expression", "")).split())
        if expected_formula is not None and actual_formula != expected_formula:
            issues.append(
                f"派生量 {did} 表达式必须是 {expected_formula!r}，实得 {actual_formula!r}")

    evaluated = document.get("scenario_derivations", {})
    if kind == "gdn":
        # GDN/PLE 的固定状态量在每个场景都相同（不依赖 seqlen/kvcachelen）。
        for sid, entries_list in evaluated.items():
            entries = {e.get("id"): e for e in entries_list}
            for did, value in expected.items():
                entry = entries.get(did)
                if entry is None:
                    issues.append(f"场景 {sid} 缺少 {did}")
                    continue
                if entry.get("state") != "determined" or entry.get("value") != value:
                    issues.append(
                        f"场景 {sid} 的 {did}={entry.get('value')!r}"
                        f"（{entry.get('state')}），期望 {value}")
    else:
        for sid, want in expected.items():
            entries = {e.get("id"): e for e in evaluated.get(sid, [])}
            for did, value in want.items():
                entry = entries.get(did)
                if entry is None:
                    issues.append(f"场景 {sid} 缺少 {did}")
                    continue
                if entry.get("state") != "determined" or entry.get("value") != value:
                    issues.append(
                        f"场景 {sid} 的 {did}={entry.get('value')!r}"
                        f"（{entry.get('state')}），期望 {value}")
    # 符号场景：与 seqlen 有关的 QSA 量必须未知，而不是 0。
    if kind == "qsa":
        sym = {e.get("id"): e for e in evaluated.get("prefill-symbolic", [])}
        for did in ("L_kv", "L_idx", "N_pool", "V_q", "N_q", "B_q", "R_tail",
                    "K_sel", "K_sel_last"):
            entry = sym.get(did)
            if entry is None:
                issues.append(f"prefill-symbolic 缺少 {did}")
            elif entry.get("state") == "determined":
                issues.append(f"prefill-symbolic 的 {did} 不应有确定值")
    return issues


def _check_gdn_block(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    tensors = _tensor_map(block)
    required_ops = (
        "g_hc_attn_norm", "g_hc_attn_down", "g_hc_attn_down_act", "g_hc_attn_up",
        "g_hc_attn_read_sigmoid", "g_hc_attn_read_view", "g_hc_attn_norm_view",
        "g_hc_attn_mix", "g_hc_attn_inject_linear",
        "g_hc_attn_inject", "g_hc_attn_expand", "g_hc_attn_expand_flat",
        "g_hc_attn_add",
        "g_gdn_mask", "g_gdn_qkv_proj", "g_gdn_qkv_transpose", "g_gdn_z_proj",
        "g_gdn_b_proj", "g_gdn_a_proj", "g_gdn_conv_state_update", "g_gdn_conv",
        "g_gdn_split", "g_gdn_beta", "g_gdn_decay", "g_gdn_q_repeat",
        "g_gdn_k_repeat", "g_gdn_q_l2", "g_gdn_k_l2", "g_gdn_q_scale",
        "g_gdn_scan", "g_gdn_state_write", "g_gdn_o_norm", "g_gdn_o_reshape",
        "g_gdn_out_proj", "g_hc_ffn_norm", "g_hc_ffn_read_sigmoid",
        "g_hc_ffn_read_view", "g_hc_ffn_norm_view", "g_hc_ffn_mix",
        "g_hc_ffn_expand_flat", "g_hc_ffn_add",
        "g_moe_router", "g_moe_softmax", "g_moe_topk", "g_moe_weight_norm",
        "g_moe_expert_gate_up", "g_moe_expert_act", "g_moe_expert_down",
        "g_moe_shared_gate_scalar", "g_moe_merge",
    )
    for oid in required_ops:
        if oid not in ops:
            issues.append(f"{label}: 缺少 GDN 算子 {oid}")
    # GDN 绝不能出现 QSA/RoPE/indexer。
    for oid in ops:
        if oid.startswith("g_qsa_") or "rope" in oid.lower() or "indexer" in oid.lower():
            issues.append(f"{label}: GDN 层不应有 QSA/RoPE/indexer 算子 {oid}")
    scan = ops.get("g_gdn_scan", {})
    note = str(scan.get("note", ""))
    for token in ("A = S", "A^T k", "delta", "chunk", "recurrent"):
        if token not in note:
            issues.append(f"{label}: g_gdn_scan note 缺少真实递推事实 {token!r}")
    if "S_{t-1} k" in note and "A^T k" not in note:
        issues.append(f"{label}: g_gdn_scan 误差方向错误")
    # 读门/写门形状
    read = tensors.get("g_hc_attn_read", {})
    if read.get("shape") != ["B", "seqlen", "Hc", "H"]:
        issues.append(f"{label}: 读门形状必须是 [B,seqlen,Hc,H]（逐元素），实得 {read.get('shape')}")
    inject = tensors.get("g_hc_attn_inject", {})
    if inject.get("shape") != ["B", "seqlen", "Hc"]:
        issues.append(f"{label}: 写门形状必须是 [B,seqlen,Hc]（每支路标量），实得 {inject.get('shape')}")
    # GDN 状态固定
    for tid in ("g_gdn_conv_state_pre", "g_gdn_state_pre"):
        if tid not in tensors:
            issues.append(f"{label}: 缺少状态张量 {tid}")
    streams = tensors.get("g_hidden_streams_in", {})
    if streams.get("shape") != ["B", "seqlen", "Hc×H"]:
        issues.append(f"{label}: 层间 hidden 必须是 flat [B,seqlen,Hc×H]，实得 "
                      f"{streams.get('shape')}")
    add = ops.get("g_hc_attn_add", {})
    if add.get("inputs") and len(add["inputs"]) == 2:
        rhs = tensors.get(add["inputs"][1], {})
        if rhs.get("shape") != ["B", "seqlen", "Hc×H"]:
            issues.append(f"{label}: Attention 残差右项必须 flatten 到 [B,seqlen,Hc×H]")
    if block.get("ple"):
        pids = _param_map(block)
        for pid in ("Ngm", "Hng", "Ngh", "Dgh", "Dple", "Kple", "Dild", "Lpconv",
                    "Lctx", "Vng", "Eos"):
            if pid not in pids:
                issues.append(f"{label}: PLE 缺少参数 {pid}")
        for oid in ("g_ple_context_update", "g_ple_bigram", "g_ple_trigram",
                    "g_ple_gather", "g_ple_conv_state_update", "g_ple_conv",
                    "g_ple_stream_add"):
            if oid not in ops:
                issues.append(f"{label}: 缺少 PLE 算子 {oid}")
        conv = ops.get("g_ple_conv", {})
        if "dilation=3" not in str(conv.get("note", "")):
            issues.append(f"{label}: PLE 短卷积 note 必须说明 dilation=3")
    else:
        for oid in ops:
            if oid.startswith("g_ple_"):
                issues.append(f"{label}: 非 PLE 组不应出现 PLE 算子 {oid}")
    # MoE 事实：softmax，无 bias/scaling/clamp
    for oid in ("g_moe_bias_add", "g_moe_weight_scale", "g_moe_expert_clamp",
                "g_moe_group_score", "g_moe_group_topk"):
        if oid in ops:
            issues.append(f"{label}: Qwen MoE 不应有算子 {oid}（softmax + top-10 + 归一，无 bias/scale/clamp）")
    router = ops.get("g_moe_router", {})
    if "softmax" not in str(router.get("note", "")).lower():
        issues.append(f"{label}: g_moe_router note 应说明 softmax 路由")
    return issues


def _check_qsa_block(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    tensors = _tensor_map(block)
    required_ops = (
        "g_hc_attn_norm", "g_hc_attn_mix", "g_hc_attn_inject", "g_hc_attn_add",
        "g_qsa_q_proj", "g_qsa_q_gate_chunk", "g_qsa_q_norm", "g_qsa_q_transpose",
        "g_qsa_k_proj", "g_qsa_k_norm", "g_qsa_v_proj", "g_qsa_q_rope",
        "g_qsa_k_rope", "g_qsa_cache_update", "g_qsa_ix_qk_proj", "g_qsa_ix_split",
        "g_qsa_ix_q_norm", "g_qsa_ix_q_rope", "g_qsa_ix_cache_update",
        "g_qsa_ix_visible", "g_qsa_ix_query_guard", "g_qsa_ix_block_view",
        "g_qsa_ix_pool",
        "g_qsa_ix_block_rope", "g_qsa_ix_scores", "g_qsa_ix_block_topk",
        "g_qsa_ix_select_blocks", "g_qsa_ix_tail", "g_qsa_ix_concat",
        "g_qsa_ix_pad", "g_qsa_ix_mask_bool", "g_qsa_ix_mask_float",
        "g_qsa_mask_combine", "g_qsa_repeat_kv",
        "g_qsa_qk", "g_qsa_mask_add", "g_qsa_softmax", "g_qsa_av",
        "g_qsa_gate_sigmoid", "g_qsa_gate_mul", "g_qsa_o_proj",
        "g_hc_ffn_norm", "g_hc_ffn_mix", "g_hc_ffn_add", "g_moe_softmax",
    )
    for oid in required_ops:
        if oid not in ops:
            issues.append(f"{label}: 缺少 QSA 算子 {oid}")
    for oid in ops:
        if oid.startswith("g_gdn_") or "delta_rule" in oid or "conv_state" in oid:
            issues.append(f"{label}: QSA 层不应有 GDN 算子 {oid}")
    # indexer 评分是 relu 求和，不是 softmax
    scores = ops.get("g_qsa_ix_scores", {})
    snote = str(scores.get("note", ""))
    for token in ("relu", "sum", "sqrt"):
        if token not in snote.lower():
            issues.append(f"{label}: indexer 打分 note 缺少 {token!r}")
    if "softmax" in snote.lower() and "不是 softmax" not in snote.lower():
        issues.append(f"{label}: indexer 打分不应是 softmax")
    # 块 top-k 预算
    topk = ops.get("g_qsa_ix_block_topk", {})
    if "512" not in str(topk.get("note", "")):
        issues.append(f"{label}: 块 top-k note 应说明每 query 最多 512 块")
    # mask 语义 True=可见
    mask = ops.get("g_qsa_ix_mask_bool", {})
    if "True=可见" not in str(mask.get("note", "")):
        issues.append(f"{label}: selected mask note 必须说明 True=可见")
    convert = ops.get("g_qsa_ix_mask_float", {})
    if "0.0" not in str(convert.get("note", "")) or "min" not in str(convert.get("note", "")):
        issues.append(f"{label}: selected mask 必须先转 0/min 再与 causal 相加")
    combine = ops.get("g_qsa_mask_combine", {})
    if "g_qsa_ix_mask_float" not in combine.get("inputs", []):
        issues.append(f"{label}: mask 合并必须消费 float selected mask")
    # partial RoPE 只转 64
    for oid in ("g_qsa_q_rope", "g_qsa_k_rope", "g_qsa_ix_q_rope",
                "g_qsa_ix_block_rope"):
        note = str(ops.get(oid, {}).get("note", ""))
        if "64" not in note:
            issues.append(f"{label}: {oid} note 必须说明只转前 64 维")
    # KV 2 头
    kpre = tensors.get("g_qsa_k_pre", {})
    if kpre.get("shape") != ["B", "Nkv", "kvcachelen", "Dh"]:
        issues.append(f"{label}: K 缓存形状必须是 [B,Nkv,kvcachelen,Dh]，实得 {kpre.get('shape')}")
    # 逐 query 局部维，不得用全长 N_pool/B_sel 充当 pool/topk 实际 shape
    if tensors.get("g_qsa_ix_block_tokens", {}).get("shape") != ["B", "seqlen", "Nq", "Kc"]:
        issues.append(f"{label}: block tokens 必须用逐 query N_q，实得 "
                      f"{tensors.get('g_qsa_ix_block_tokens', {}).get('shape')}")
    if tensors.get("g_qsa_ix_block_idx", {}).get("shape") != ["B", "seqlen", "Bq"]:
        issues.append(f"{label}: block top-k 必须用逐 query B_q，实得 "
                      f"{tensors.get('g_qsa_ix_block_idx', {}).get('shape')}")
    if "g_qsa_ix_query_guard" not in ops:
        issues.append(f"{label}: 缺少逐 query 零块守卫")
    if "g_qsa_ix_k_norm" in ops:
        issues.append(f"{label}: 不应有逐 token indexer K Norm")
    # flat 层间边界
    streams = tensors.get("g_hidden_streams_in", {})
    if streams.get("shape") != ["B", "seqlen", "Hc×H"]:
        issues.append(f"{label}: 层间 hidden 必须是 flat [B,seqlen,Hc×H]，实得 "
                      f"{streams.get('shape')}")
    return issues


def _check_block(block: dict, label: str, *, legacy: bool = False) -> list[str]:
    issues: list[str] = []
    kind = block.get("kind")
    if kind not in ("gdn", "qsa"):
        issues.append(f"{label}: kind 必须是 gdn/qsa: {kind!r}")
        return issues
    ple = bool(block.get("ple"))
    if kind == "qsa" and ple:
        issues.append(f"{label}: QSA 层不能有 PLE")
    issues.extend(_check_param_facts(block, label, kind, legacy=legacy))
    issues.extend(_check_no_foreign_facts(block, label, kind))
    if kind == "gdn":
        issues.extend(_check_gdn_block(block, label))
    else:
        issues.extend(_check_qsa_block(block, label))
    if not block.get("cache_notes"):
        issues.append(f"{label}: cache_notes 为空")
    if not block.get("coverage_notes"):
        issues.append(f"{label}: coverage_notes 为空")
    if not legacy:
        if block.get("attention_type") != (
            "linear_attention" if kind == "gdn" else "indexed_attention"
        ):
            issues.append(f"{label}: attention_type 与 kind 不一致")
        if block.get("moe_type") != "sparse":
            issues.append(f"{label}: moe_type 必须是 sparse")
        if block.get("ple") != ple:
            issues.append(f"{label}: ple 标记不一致")
    return issues


def _check_scope(scope: dict) -> list[str]:
    issues: list[str] = []
    if scope.get("analysis_path") != "eager":
        issues.append("scope.analysis_path 必须是 eager")
    if scope.get("layer_count") != 48:
        issues.append(f"scope.layer_count 必须是 48: {scope.get('layer_count')!r}")
    if scope.get("hidden_size") != 2560:
        issues.append(f"scope.hidden_size 必须是 2560: {scope.get('hidden_size')!r}")
    if scope.get("vocab_size") != 248320:
        issues.append(f"scope.vocab_size 必须是 248320: {scope.get('vocab_size')!r}")
    if scope.get("hc_count") != 4:
        issues.append(f"scope.hc_count 必须是 4: {scope.get('hc_count')!r}")
    return issues


def validate_hf_layer_structure(document: dict, input_shas: dict[str, str] | None = None) -> list[str]:
    """一份参数化 Block 结构（hf_layer_structure.v1）的 Qwen 检查。"""
    issues: list[str] = []
    if document.get("schema") != "hf_layer_structure.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    issues.extend(_check_identity(document.get("model", {})))
    block = document.get("block")
    if not isinstance(block, dict):
        issues.append("block 缺失")
        return issues
    issues.extend(_check_block(block, "block"))
    scope = document.get("scope", {})
    if scope.get("analysis_path") != "eager":
        issues.append("scope.analysis_path 必须是 eager")
    if scope.get("layer_index") != block.get("layer_index"):
        issues.append("scope.layer_index 与 block 不一致")
    scenarios = scope.get("scenarios", [])
    if not scenarios:
        issues.append("scope.scenarios 缺失")
    ids = [s.get("id") for s in scenarios]
    for sid in ("prefill-empty", "prefill-symbolic", "decode-nonzero",
                "decode-long", "decode-multi"):
        if sid not in ids:
            issues.append(f"缺少必要场景 {sid}")
    if scope.get("default_scenario") not in ids:
        issues.append("default_scenario 必须是已声明场景之一")
    prefill = next((s for s in scenarios if s.get("id") == "prefill-empty"), {})
    if prefill.get("kvcachelen") != 0:
        issues.append("prefill-empty 的 kvcachelen 必须是 0")
    decode = next((s for s in scenarios if s.get("id") == "decode-nonzero"), {})
    if decode.get("seqlen") != 1:
        issues.append("decode-nonzero 的 seqlen 必须是 1")
    issues.extend(_check_derivations(document, str(block.get("kind")),
                                     ple=bool(block.get("ple"))))
    if input_shas is not None:
        issues.extend(hf_core.check_input_identity(input_shas, SOURCE_FILES))
    return issues


def validate_hf_reading_document(document: dict, input_shas: dict[str, str]) -> list[str]:
    """同源 prefill/decode 投影（hf_decoder_reading.v1）的 Qwen 检查。"""
    issues: list[str] = []
    if document.get("schema") != "hf_decoder_reading.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    issues.extend(_check_identity(document.get("model", {})))
    scope = document.get("scope", {})
    if scope.get("analysis_path") != "eager":
        issues.append("scope.analysis_path 必须是 eager")
    attention_type = scope.get("attention_type")
    if attention_type not in ("linear_attention", "indexed_attention"):
        issues.append("scope.attention_type 非法")
    moe_type = scope.get("moe_type")
    if moe_type != "sparse":
        issues.append("scope.moe_type 必须是 sparse（本模型没有 dense 层）")
    if attention_type == "indexed_attention" and scope.get("ple"):
        issues.append("QSA 层不能有 PLE")
    phases = document.get("phases", [])
    if [p.get("id") for p in phases] != ["prefill", "decode"]:
        issues.append("phases 必须依次为 prefill、decode")
    for phase in phases:
        pid = phase.get("id", "?")
        issues.extend(hf_core.check_unit(phase, pid))
        issues.extend(hf_core.check_flow_units([phase]))
        issues.extend(_check_block(phase, pid, legacy=True))
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
    """整网总览（hf_network_overview.v1）的 Qwen 检查。"""
    issues: list[str] = []
    if document.get("schema") != "hf_network_overview.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    issues.extend(_check_identity(document.get("model", {})))
    scope = document.get("scope", {})
    issues.extend(_check_scope(scope))
    if scope.get("entry_class") != "Qwen4ExpForConditionalGeneration":
        issues.append(f"scope.entry_class 必须是 Qwen4ExpForConditionalGeneration: "
                      f"{scope.get('entry_class')!r}")
    if scope.get("main_module_class") != "Qwen4ExpTextModel":
        issues.append("scope.main_module_class 必须是 Qwen4ExpTextModel")
    if scope.get("decoder_layer_class") != "Qwen4ExpTextDecoderLayer":
        issues.append("scope.decoder_layer_class 必须是 Qwen4ExpTextDecoderLayer")
    if len(scope.get("gdn_layers", [])) != 36:
        issues.append(f"scope.gdn_layers 必须是 36 层: {len(scope.get('gdn_layers', []))}")
    if scope.get("qsa_layers") != [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]:
        issues.append(f"scope.qsa_layers 必须是 3,7,...,47: {scope.get('qsa_layers')!r}")
    if scope.get("ple_layers") != [1]:
        issues.append(f"scope.ple_layers 必须是 [1]（第 2 层）: {scope.get('ple_layers')!r}")
    groups = document.get("groups", [])
    counts = {g.get("id"): len(g.get("layer_indices", [])) for g in groups}
    if counts != {"gdn-moe": 35, "gdn-ple-moe": 1, "qsa-moe": 12}:
        issues.append(f"三组层数必须是 35/1/12，实得 {counts}")
    covered = sorted(i for g in groups for i in g.get("layer_indices", []))
    if covered != list(range(48)):
        issues.append(f"三组层索引必须恰好覆盖 0..47，实得 {covered}")
    order = document.get("layer_order", [])
    if len(order) != 48 or [x.get("index") for x in order] != list(range(48)):
        issues.append("layer_order 必须逐层列出 0..47")
    else:
        for entry in order:
            expected_type = ("indexed_attention" if entry["index"] % 4 == 3
                             else "linear_attention")
            if entry.get("attention_type") != expected_type:
                issues.append(f"layer_order[{entry['index']}] 层型错误: "
                              f"{entry.get('attention_type')!r}，期望 {expected_type}")
            expected_ple = entry["index"] == 1
            if bool(entry.get("ple")) != expected_ple:
                issues.append(f"layer_order[{entry['index']}] PLE 标记错误")
    path = document.get("path")
    if not isinstance(path, dict):
        issues.append("path 缺失（整网总览必须提供单条 token→logits 路径）")
        return issues
    if path.get("id") != "network":
        issues.append(f"path.id 必须是 'network': {path.get('id')!r}")
    issues.extend(hf_core.check_unit(path, "network"))
    issues.extend(hf_core.check_flow_units([path]))
    ops = _op_map(path)
    order_map = _op_order(path)
    chain = ["net_embed", "net_position_ids", "net_mask_build", "net_ple_input_ids",
             "net_hidden_repeat", "net_layer_loop", "net_mixer_mix",
             "net_logits_slice", "net_lm_head"]
    for oid in chain:
        if oid not in ops:
            issues.append(f"network: 缺少算子 {oid}")
    for first, second in zip(chain, chain[1:]):
        if order_map.get(first, 0) >= order_map.get(second, 0):
            issues.append(f"network: 算子顺序错误 {first} 必须在 {second} 之前")
    for oid in ("net_hc_expand", "net_hc_head", "net_final_norm"):
        if oid in ops:
            issues.append(f"network: Qwen 不应有算子 {oid}（4 路 repeat + mixer，且没有 Final RMSNorm）")
    loop = ops.get("net_layer_loop", {})
    if "48" not in str(loop.get("note", "")):
        issues.append("network: 层循环 note 必须说明 48 层")
    mixer = ops.get("net_mixer_mix", {})
    if "mean" not in str(mixer.get("note", "")):
        issues.append("network: mixer note 必须说明加权 mean（不是普通 sum/mean）")
    # 没有视觉/MTP 执行
    for oid in ops:
        low = oid.lower()
        if "visual" in low or "vision" in low or "mtp" in low:
            issues.append(f"network: 不应执行视觉/MTP 算子 {oid}")
    issues.extend(hf_core.check_input_identity(input_shas, SOURCE_FILES))
    return issues
