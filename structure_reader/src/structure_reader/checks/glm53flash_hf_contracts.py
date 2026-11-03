"""GLM-5.3-Flash（glm5_next）HF 阅读数据语义契约检查（静态，不运行模型）。

检查对象是 ``extract/glm53flash_hf.py`` / ``extract/glm53flash_network.py`` 产出的
文档；任何一条不通过即返回问题串，由构建层具名拒绝（不静默修正、不发布检查未过的
候选）。

与 GLM-5.3 / DeepSeek 的 checks 分开：这里的语义锚点是 Flash 的真实实现事实，
不能把 GLM-5.3 的「全 MLA/DSA + 普通残差 + 部分 RoPE」或 DeepSeek 的
「CSA/HCA/滑窗/Hash」当成通用规则。反过来，如果 Flash 文档里出现那些机制，
本检查会拒绝（说明误抄了别的模型）。

覆盖：
1. 身份：官方 Flash revision、model_type/architectures、text_config、实现路径、45 层。
2. 骨架/数据流闭合（复用 checks/hf_contract_core）。
3. 参数事实：H=4096、Nh=64、Dn/Dv/Dq=256/256/256、Rk=0（NoPE）、Rq=1536、
   Rkv=512、Ih=32/Id=128、Ipack=257、Ki=2048、Kpool=4、288 专家选 8、
   Es=2048、Is=12288、scaling=1/16、Swi=10、hc_mult=4、Sinkhorn=20、
   eps=1e-5/1e-6、KDA 64×128×4。
4. 层归属：三组 kda-dense(3)/dsa-moe(11)/kda-moe(31) 恰好覆盖 45 层；
   DSA 层恰为 3,7,11,…,43；dense 层 0/1/2；indexer_types 全 full。
5. KDA：q/k/v 投影 → 短卷积 + conv state → forget gate/beta → L2Norm/缩放 →
   delta-rule 扫描（chunk/recurrent）+ recurrent state 写回 → gated RMSNorm →
   o_proj；没有 RoPE/indexer/普通残差。
6. MLA/DSA：Q 低秩 + NoPE（无 split/rope 算子）；KV 只缓存压缩 latent（512 + 0）；
   kv_b 在缓存之后展开；indexer packed 缓存；k-pool 池 top-k + tail；
   selected counts 稀疏 mask（True=可见）；eager 注意力。
7. mHC：attn/ffn 两处都有 in_norm→fn→split→pre/post 门控 + Sinkhorn→collapse，
   以及 expand/comb_mix/add；不是普通 add 残差。
8. MoE：sigmoid + correction bias + top-8 + 归一 + ×2.5；单份专家模板；共享专家
   并行相加；专家/共享/稠密 SwiGLU 都有 clamp；dense 层没有 MoE 算子。
9. 场景派生量在具体场景下的取值（空缓存/非零历史/超过 index_topk/多 token）。
10. 输入身份完整（SOURCE_FILES 每个键都有实际 SHA256）。
"""
from __future__ import annotations

from structure_reader.checks import hf_contract_core as hf_core
from structure_reader.rules.glm53flash_hf import (
    MODEL_REPOSITORY,
    SOURCE_FILES,
    dsa_layer_indices,
    kda_layer_indices,
)

__all__ = [
    "validate_hf_layer_structure",
    "validate_hf_reading_document",
    "validate_hf_network_overview",
]

_EXPECTED_PARAMS = {
    "H": 4096, "Hc": 4, "NhL": 64, "DhL": 128, "Kconv": 4, "GateLB": -5.0,
    "Nh": 64, "Dn": 256, "Dv": 256, "Dq": 256, "Rk": 0, "Rq": 1536, "Rkv": 512,
    "Ih": 32, "Id": 128, "Ipack": 257, "Ki": 2048, "Kpool": 4,
    "E": 288, "K": 8, "Es": 2048, "Is": 12288, "Scl": 2.5, "Ng": 1, "Ktg": 1,
    "Swi": 10.0, "EpsN": 1e-5, "EpsA": 1e-6, "Sit": 20, "Mhc": 24,
}

# DSA 场景 → 派生量期望（独立手算，不读回文档里的值再比较）。
_DSA_EXPECTED = {
    "prefill-empty": {"L_kv": 8, "L_idx": 8, "N_pool": 2, "SelP": 2,
                      "K_sel_last": 8, "K_out": 2051},
    "decode-nonzero": {"L_kv": 4097, "L_idx": 4097, "N_pool": 1024, "SelP": 512,
                       "K_sel_last": 2049, "K_out": 2051},
    "decode-long": {"L_kv": 131073, "L_idx": 131073, "N_pool": 32768, "SelP": 512,
                    "K_sel_last": 2049, "K_out": 2051},
    "decode-multi": {"L_kv": 4096, "L_idx": 4096, "N_pool": 1024, "SelP": 512,
                     "K_sel_last": 2048, "K_out": 2051},
}

_KDA_EXPECTED = {
    "S_rec": {"prefill-empty": 1048576, "decode-nonzero": 1048576,
              "decode-long": 1048576, "decode-multi": 1048576},
    "S_conv": {"prefill-empty": 98304, "decode-nonzero": 98304,
               "decode-long": 98304, "decode-multi": 98304},
}

_FORBIDDEN_ID_TOKENS = ("hash", "tid2eid", "sliding", "csa", "hca", "swa",
                        "compress_rate", "overlap", "num_full_windows", "rope")


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
    if model.get("model_key") != "glm_5_3_flash_hf":
        issues.append(f"model.model_key 必须是 glm_5_3_flash_hf: {model.get('model_key')!r}")
    config_path = str(model.get("config_path", ""))
    if "zai-org/GLM-5.3-Flash/config.json" not in config_path:
        issues.append(f"model.config_path 必须指向官方 GLM-5.3-Flash config: {config_path!r}")
    if require_revision and model.get("model_revision") != MODEL_REPOSITORY["revision"]:
        issues.append(
            f"model.model_revision 必须是官方 revision {MODEL_REPOSITORY['revision']}: "
            f"{model.get('model_revision')!r}"
        )
    return issues


def _check_no_foreign_facts(block: dict, label: str) -> list[str]:
    """拒绝把其它模型的机制写进 Flash 结构（id 级别，不误伤说明文字）。"""
    issues: list[str] = []
    for kind in ("operators", "tensors"):
        for item in block.get(kind, []):
            item_id = str(item.get("id", "")).lower()
            for token in _FORBIDDEN_ID_TOKENS:
                if token in item_id:
                    issues.append(f"{label}: {kind} id {item.get('id')!r} 含其它模型机制 {token!r}")
    return issues


def _check_param_facts(block: dict, label: str, kind: str, *,
                       legacy: bool = False) -> list[str]:
    issues: list[str] = []
    params = _param_map(block)
    needed = set(_EXPECTED_PARAMS)
    if kind == "kda":
        needed -= {"Nh", "Dn", "Dv", "Dq", "Rk", "Rq", "Rkv", "Ih", "Id",
                   "Ipack", "Ki", "Kpool", "EpsA"}
    else:
        needed -= {"NhL", "DhL", "Kconv", "GateLB", "Srec", "Sconv"}
    for pid in sorted(needed):
        expected = _EXPECTED_PARAMS[pid]
        if pid not in params:
            issues.append(f"{label}: 缺少参数 {pid}")
            continue
        value = params[pid].get("value")
        if isinstance(expected, float):
            if not isinstance(value, (int, float)) or abs(float(value) - expected) > 1e-12:
                issues.append(f"{label}: 参数 {pid}={value!r}，期望 {expected}")
        elif value != expected:
            issues.append(f"{label}: 参数 {pid}={value!r}，期望 {expected}")
    if kind == "kda":
        for pid in ("Srec", "Sconv"):
            if pid not in params:
                issues.append(f"{label}: KDA 缺少派生参数 {pid}")
    else:
        for pid in ("Lkv", "Lidx", "Npool", "SelP", "j", "Vq", "Ksel",
                    "Ksel_last", "Kout"):
            if pid not in params:
                issues.append(f"{label}: DSA 缺少派生参数 {pid}")
        for pid in ("j", "Vq", "Ksel", "Ksel_last"):
            if params.get(pid, {}).get("value") is not None:
                issues.append(f"{label}: 逐 query 参数 {pid} 必须保留符号（value=None）")
        j_domain = params.get("j", {}).get("domain", {})
        if (j_domain.get("kind") != "integer" or j_domain.get("min") != 0
                or j_domain.get("max_symbol") != "seqlen"):
            issues.append(f"{label}: j 必须声明 integer 域 0<=j<seqlen（导出守卫依据）")
        # NoPE 与缩放
        params_dq = params.get("Dq", {}).get("value")
        params_dn = params.get("Dn", {}).get("value")
        params_rk = params.get("Rk", {}).get("value")
        if None not in (params_dq, params_dn, params_rk) and params_dq != params_dn + params_rk:
            issues.append(f"{label}: Dq={params_dq} != Dn+Rk={params_dn}+{params_rk}")
        if params_rk != 0:
            issues.append(f"{label}: Rk={params_rk!r}；Flash 必须是 NoPE（qk_rope_head_dim=0）")
    if not legacy:
        for pid in ("seqlen",):
            if params.get(pid, {}).get("value") is not None:
                issues.append(f"{label}: 结构参数 {pid} 必须保留符号（value=None）")
        if kind == "dsa" and params.get("kvcachelen", {}).get("value") is not None:
            issues.append(f"{label}: DSA 结构参数 kvcachelen 必须保留符号（value=None）")
    for pid in ("N_hit", "N_e"):
        if pid in params and params[pid].get("value") is not None:
            issues.append(f"{label}: 数据相关参数 {pid} 不得预填数值")
    return issues


def _check_derivations(document: dict, kind: str) -> list[str]:
    from structure_reader.scenario_evaluator import ExpressionError, check_expression

    issues: list[str] = []
    block = document.get("block", {})
    derivations = {d.get("id"): d for d in block.get("derivations", [])}
    if kind == "kda":
        required = ("T_tok", "S_rec", "S_conv")
        expected_map = _KDA_EXPECTED
        for did in ("L_kv", "L_idx", "K_sel"):
            if did in derivations:
                issues.append(f"KDA 层不应有 DSA 派生量 {did}")
    else:
        required = ("L_kv", "L_idx", "N_pool", "SelP", "V_q", "K_sel",
                    "K_sel_last", "K_out", "T_tok")
        expected_map = _DSA_EXPECTED
    for did in required:
        if did not in derivations:
            issues.append(f"缺少派生量 {did}")
            continue
        try:
            check_expression(str(derivations[did].get("expression", "")))
        except ExpressionError as exc:
            issues.append(f"派生量 {did} 表达式非法: {exc}")
    evaluated = document.get("scenario_derivations", {})
    if kind == "kda":
        for did, scenarios in expected_map.items():
            for scenario_id, value in scenarios.items():
                rows = evaluated.get(scenario_id) or []
                row = next((r for r in rows if r.get("id") == did), None)
                if row is None:
                    issues.append(f"KDA 场景 {scenario_id} 缺少派生结果 {did}")
                elif row.get("state") != "determined" or row.get("value") != value:
                    issues.append(f"KDA 场景 {scenario_id} 的 {did} 期望 {value}，"
                                  f"实得 {row.get('value')!r}（{row.get('state')}）")
        # 符号场景必须如实未知（B 也是独立性能输入，T_tok 不绑定场景）
        symbolic = {row.get("id"): row for row in (evaluated.get("prefill-symbolic") or [])}
        if symbolic.get("T_tok", {}).get("state") != "unknown":
            issues.append("KDA 符号 seqlen 场景下 T_tok 必须未知")
        return issues

    # DSA 表达式语义
    lkv_expr = str(derivations.get("L_kv", {}).get("expression", "")).replace(" ", "")
    if lkv_expr != "kvcachelen+seqlen":
        issues.append(f"L_kv 必须是 kvcachelen + seqlen: {lkv_expr!r}")
    lidx_expr = str(derivations.get("L_idx", {}).get("expression", "")).replace(" ", "")
    if lidx_expr != "kvcachelen+seqlen":
        issues.append(f"L_idx 必须是 kvcachelen + seqlen: {lidx_expr!r}")
    if str(derivations.get("V_q", {}).get("expression", "")).replace(" ", "") != \
            "kvcachelen+j+1":
        issues.append("V_q 必须是 kvcachelen + j + 1（逐 query 可见长度）")
    if str(derivations.get("K_sel", {}).get("expression", "")).replace(" ", "") != \
            "min(V_q,Ki+(V_q%Kpool))":
        issues.append("K_sel 必须是 min(V_q, Ki + (V_q % Kpool))（逐 query，不是全长统一值）")
    if str(derivations.get("K_sel_last", {}).get("expression", "")).replace(" ", "") != \
            "min(L_idx,Ki+(L_idx%Kpool))":
        issues.append("K_sel_last 必须是 min(L_idx, Ki + (L_idx % Kpool))（仅最后 query）")
    if str(derivations.get("K_out", {}).get("expression", "")).replace(" ", "") != \
            "Ki+Kpool-1":
        issues.append("K_out 必须是 Ki + Kpool - 1")
    for scenario_id, expected in expected_map.items():
        rows = evaluated.get(scenario_id) or []
        by_id = {row.get("id"): row for row in rows}
        for did in ("V_q", "K_sel"):
            row = by_id.get(did)
            if row is None or row.get("state") != "unknown":
                issues.append(f"场景 {scenario_id} 的 {did} 必须未知（query 索引 j 未给定）")
        for did, value in expected.items():
            row = by_id.get(did)
            if row is None:
                issues.append(f"场景 {scenario_id} 缺少派生结果 {did}")
            elif row.get("state") != "determined" or row.get("value") != value:
                issues.append(f"场景 {scenario_id} 的 {did} 期望 {value}，"
                              f"实得 {row.get('value')!r}（{row.get('state')}）")
    symbolic = {row.get("id"): row for row in (evaluated.get("prefill-symbolic") or [])}
    for did in ("L_kv", "N_pool", "V_q", "K_sel"):
        if symbolic.get(did, {}).get("state") != "unknown":
            issues.append(f"符号 seqlen 场景下 {did} 必须未知（不默认 0）")
    return issues


def _check_mhc(block: dict, label: str) -> list[str]:
    """两处 mHC 的真实权重/坍缩/展开/混合；拒绝普通残差 add。"""
    issues: list[str] = []
    ops = _op_map(block)
    tensors = _tensor_map(block)
    params = _param_map(block)
    order = _op_order(block)
    if params.get("Mhc", {}).get("value") != 24:
        issues.append(f"{label}: Mhc 必须是 (2+Hc)*Hc=24，实得 "
                      f"{params.get('Mhc', {}).get('value')!r}")

    def need(oid: str) -> bool:
        if oid not in ops:
            issues.append(f"{label}: 缺少 mHC 算子 {oid}")
            return False
        return True

    for site in ("attn", "ffn"):
        chain = [f"g_mhc_{site}_{name}" for name in
                 ("in_norm", "fn", "split", "gate", "sinkhorn", "collapse",
                  "expand", "comb_mix", "add")]
        missing = [oid for oid in chain if not need(oid)]
        if missing:
            continue
        for first, second in zip(chain, chain[1:]):
            if order.get(first, 0) >= order.get(second, 0):
                issues.append(f"{label}: mHC {site} 顺序错误 {first} 必须在 {second} 之前")
        if tensors.get(f"g_mhc_{site}_fn_weight", {}).get("shape") != ["Mhc", "Hc×H"]:
            issues.append(f"{label}: mHC {site} fn 必须是 [Mhc,Hc×H]（24×16384）")
        if tensors.get(f"g_mhc_{site}_base", {}).get("shape") != ["Mhc"]:
            issues.append(f"{label}: mHC {site} base 必须是 [Mhc]（24）")
        if tensors.get(f"g_mhc_{site}_mixes", {}).get("shape") != ["B", "seqlen", "Mhc"]:
            issues.append(f"{label}: mHC {site} mixes 必须是 [B,seqlen,Mhc]（24 个系数）")
        if tensors.get(f"g_mhc_{site}_pre", {}).get("shape") != ["B", "seqlen", "Hc"]:
            issues.append(f"{label}: mHC {site} pre 必须是 [B,seqlen,Hc]")
        if tensors.get(f"g_mhc_{site}_post", {}).get("shape") != ["B", "seqlen", "Hc"]:
            issues.append(f"{label}: mHC {site} post 必须是 [B,seqlen,Hc]")
        if tensors.get(f"g_mhc_{site}_comb", {}).get("shape") != ["B", "seqlen", "Hc", "Hc"]:
            issues.append(f"{label}: mHC {site} comb 必须是 [B,seqlen,Hc,Hc]")
        if tensors.get(f"g_mhc_{site}_collapsed", {}).get("shape") != ["B", "seqlen", "H"]:
            issues.append(f"{label}: mHC {site} collapsed 必须是 [B,seqlen,H]")
        if ops[f"g_mhc_{site}_in_norm"].get("op") != "rmsnorm-unweighted":
            issues.append(f"{label}: mHC {site} 输入归一必须是无权重 RMSNorm")
        if ops[f"g_mhc_{site}_sinkhorn"].get("op") != "sinkhorn":
            issues.append(f"{label}: mHC {site} comb 必须过 Sinkhorn")
        if "Sit" not in ops[f"g_mhc_{site}_sinkhorn"].get("parameter_refs", []):
            issues.append(f"{label}: mHC {site} Sinkhorn 必须引用 Sit（hc_sinkhorn_iters）")
        if ops[f"g_mhc_{site}_add"].get("op") != "add":
            issues.append(f"{label}: mHC {site} 流相加必须是 add")
    if tensors.get("g_hidden_streams_out", {}).get("shape") != ["B", "seqlen", "Hc", "H"]:
        issues.append(f"{label}: 层输出必须保持 [B,seqlen,Hc,H] 的 4 路流")
    if "g_res_attn" in ops or "g_res_ffn" in ops:
        issues.append(f"{label}: Flash 没有普通单流残差算子 g_res_attn/g_res_ffn")
    extra_adds = [oid for oid, op in ops.items()
                  if op.get("op") == "add" and oid.startswith("g_res_")]
    if extra_adds:
        issues.append(f"{label}: 出现普通残差 add {extra_adds}")
    return issues


def _check_kda(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    tensors = _tensor_map(block)
    order = _op_order(block)
    chain = [
        "g_input_norm", "g_kda_mask_apply",
        "g_kda_q_proj", "g_kda_k_proj", "g_kda_v_proj", "g_kda_qkv_cat",
        "g_kda_qkv_transpose", "g_kda_conv_state_update", "g_kda_conv1d",
        "g_kda_qkv_view", "g_kda_qkv_split",
        "g_kda_f_a_proj", "g_kda_f_b_proj", "g_kda_dt_add", "g_kda_decay",
        "g_kda_b_proj", "g_kda_beta",
        "g_kda_q_l2", "g_kda_k_l2", "g_kda_q_scale", "g_kda_scan",
        "g_kda_state_write",
        "g_kda_g_a_proj", "g_kda_g_b_proj", "g_kda_gate_view",
        "g_kda_o_norm", "g_kda_o_reshape", "g_kda_o_proj",
    ]
    missing = [oid for oid in chain if oid not in ops]
    if missing:
        issues.append(f"{label}: KDA 缺少算子 {missing}")
        return issues
    for first, second in zip(chain, chain[1:]):
        if order[first] >= order[second]:
            issues.append(f"{label}: KDA 顺序错误 {first} 必须在 {second} 之前")
    # 短卷积 + 固定状态
    if ops["g_kda_conv1d"].get("op") != "conv1d":
        issues.append(f"{label}: KDA 必须有因果短卷积 conv1d")
    if ops["g_kda_conv_state_update"].get("op") != "cache_update":
        issues.append(f"{label}: KDA conv state 必须是 cache_update")
    if "g_kda_conv_weight" not in ops["g_kda_conv1d"].get("weights", []):
        issues.append(f"{label}: KDA 卷积必须引用 conv1d.weight")
    conv_shape = tensors.get("g_kda_conv_state_pre", {}).get("shape")
    if conv_shape != ["B", "3×NhL×DhL", "Kconv"]:
        issues.append(f"{label}: conv state 必须是 [B,3×NhL×DhL,Kconv]（固定大小）")
    state_shape = tensors.get("g_kda_state_pre", {}).get("shape")
    if state_shape != ["B", "NhL", "DhL", "DhL"]:
        issues.append(f"{label}: recurrent state 必须是 [B,NhL,DhL,DhL]（固定大小）")
    # 扫描语义
    scan = ops["g_kda_scan"]
    if scan.get("op") != "scan":
        issues.append(f"{label}: KDA 核心必须是 scan（递归/分块），不能伪装成 Attention")
    for required in ("g_kda_state_pre", "g_kda_g", "g_kda_beta", "g_kda_k_hat",
                     "g_kda_v", "g_kda_q_scaled"):
        if required not in scan.get("inputs", []):
            issues.append(f"{label}: KDA scan 未消费 {required}")
    if sorted(scan.get("outputs", [])) != ["g_kda_core_out", "g_kda_state_new"]:
        issues.append(f"{label}: KDA scan 必须同时产出 core_out 与 state_new")
    note = str(scan.get("note", ""))
    for token in ("delta", "chunk", "recurrent", "A = S_{t-1}", "exp(g",
                  "A^T", "S_t = A", "seq_len==1", "空缓存"):
        if token not in note:
            issues.append(f"{label}: KDA scan 说明缺少真实语义 {token!r}")
    if "S_t = S_{t-1}" in note:
        issues.append(f"{label}: KDA scan 仍把未衰减 S_{{t-1}} 写进状态更新公式")
    if ops["g_kda_state_write"].get("op") != "cache_update":
        issues.append(f"{label}: KDA recurrent state 写回必须是 cache_update")
    if ops["g_kda_decay"].get("op") != "sigmoid+scale":
        issues.append(f"{label}: KDA forget gate 必须是 -5*sigmoid(exp(A_log)*g)")
    if "GateLB" not in ops["g_kda_decay"].get("parameter_refs", []):
        issues.append(f"{label}: KDA forget gate 必须引用 GateLB（-5 下界）")
    if ops["g_kda_q_l2"].get("op") != "l2norm" or ops["g_kda_k_l2"].get("op") != "l2norm":
        issues.append(f"{label}: KDA q/k 必须在 kernel 内 L2Norm")
    if ops["g_kda_o_norm"].get("op") != "rmsnorm-gated":
        issues.append(f"{label}: KDA 输出必须是 gated RMSNorm")
    if "g_kda_gate" not in ops["g_kda_o_norm"].get("inputs", []):
        issues.append(f"{label}: KDA gated RMSNorm 必须消费输出门")
    # 不是 MLA/DSA
    if any(oid.startswith("g_ix_") or oid.startswith("g_dsa_") for oid in ops):
        issues.append(f"{label}: KDA 层不得出现 MLA/DSA indexer/attention 算子")
    if any("g_cache_kv" in tid or "g_cache_krot" in tid for tid in tensors):
        issues.append(f"{label}: KDA 层不得出现 MLA latent 缓存")
    if "g_qk" in ops or "g_index_mask_scatter" in ops:
        issues.append(f"{label}: KDA 层不得出现 QK 稀疏注意力算子")
    return issues


def _check_dsa(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    tensors = _tensor_map(block)
    order = _op_order(block)
    chain = [
        "g_input_norm", "g_dsa_q_a_proj", "g_dsa_q_a_norm", "g_dsa_q_b_proj",
        "g_dsa_q_view", "g_dsa_kv_a_proj", "g_dsa_kv_a_split", "g_dsa_kv_a_norm",
        "g_dsa_k_pass_view", "g_dsa_k_rot_view", "g_dsa_cache_kv_update",
        "g_dsa_kv_b_proj", "g_dsa_kv_view", "g_dsa_kv_split", "g_dsa_k_rot_expand",
        "g_dsa_key_cat", "g_dsa_ix_q_proj", "g_dsa_ix_q_view", "g_dsa_ix_k_proj",
        "g_dsa_ix_k_norm", "g_dsa_ix_gate", "g_dsa_ix_valid", "g_dsa_ix_pack",
        "g_dsa_ix_cache_update", "g_dsa_ix_valid_keys", "g_dsa_ix_visible",
        "g_dsa_ix_pool", "g_dsa_ix_pool_end", "g_dsa_ix_valid_candidates",
        "g_dsa_ix_scores", "g_dsa_ix_relu", "g_dsa_ix_weights",
        "g_dsa_ix_sum", "g_dsa_ix_mask", "g_dsa_ix_topk", "g_dsa_ix_selected_valid",
        "g_dsa_ix_select_pools", "g_dsa_ix_flatten", "g_dsa_ix_invalid",
        "g_dsa_ix_tail", "g_dsa_ix_concat_tail", "g_dsa_ix_pad",
        "g_dsa_ix_query_mask", "g_dsa_mask_counts",
        "g_dsa_sparse_mask", "g_dsa_eager_mask", "g_dsa_qk", "g_dsa_mask_add",
        "g_dsa_softmax", "g_dsa_av", "g_dsa_attn_transpose", "g_dsa_attn_reshape",
        "g_dsa_o_proj",
    ]
    missing = [oid for oid in chain if oid not in ops]
    if missing:
        issues.append(f"{label}: DSA 缺少算子 {missing}")
        return issues
    for first, second in zip(chain, chain[1:]):
        if order[first] >= order[second]:
            issues.append(f"{label}: DSA 顺序错误 {first} 必须在 {second} 之前")
    # NoPE：没有 split/rope
    for forbidden in ("g_q_split", "g_q_rope", "g_q_cat", "g_kv_rope",
                      "g_ix_rope", "g_ix_q_split", "g_ix_k_split"):
        if forbidden in ops:
            issues.append(f"{label}: Flash 是 NoPE，不应出现 {forbidden}")
    if tensors.get("g_dsa_q", {}).get("shape") != ["B", "Nh", "seqlen", "Dq"]:
        issues.append(f"{label}: Q 必须是 [B,Nh,seqlen,Dq]（NoPE 无 rope 段）")
    if tensors.get("g_dsa_key", {}).get("shape") != ["B", "Nh", "Lkv", "Dq"]:
        issues.append(f"{label}: key 必须是 [B,Nh,Lkv,Dq]")
    if tensors.get("g_dsa_k_nope", {}).get("shape", [None])[-1] != "Dn":
        issues.append(f"{label}: k_nope 尾维必须是 Dn=256")
    if tensors.get("g_dsa_v", {}).get("shape", [None])[-1] != "Dv":
        issues.append(f"{label}: value 尾维必须是 Dv=256")
    # 缓存：压缩 latent，不是展开 K/V
    for tid, axis in (("g_cache_kv_pre", "Rkv"), ("g_cache_krot_pre", "Rk")):
        tensor = tensors.get(tid, {})
        if tensor.get("kind") != "cache":
            issues.append(f"{label}: {tid} 必须是 cache 张量")
        if tensor.get("shape", [None])[-1] != axis:
            issues.append(f"{label}: {tid} 尾维必须是 {axis}")
        if "Nh" in tensor.get("shape", []):
            issues.append(f"{label}: {tid} 不应带 Nh 轴（缓存不是展开后的多头 K/V）")
    if "g_dsa_k_pass_4d" not in ops["g_dsa_cache_kv_update"].get("inputs", []):
        issues.append(f"{label}: 缓存 update 必须消费单头 view 后的 k_pass")
    if "g_cache_kv_post" not in ops["g_dsa_kv_b_proj"].get("inputs", []):
        issues.append(f"{label}: kv_b_proj 必须消费缓存返回的 latent（展开在缓存之后）")
    if order["g_dsa_cache_kv_update"] >= order["g_dsa_kv_b_proj"]:
        issues.append(f"{label}: 缓存写入必须先于 kv_b 展开")
    # indexer packed / NoPE
    if tensors.get("g_dsa_ix_packed", {}).get("shape") != ["B", "seqlen", "Ipack"]:
        issues.append(f"{label}: indexer packed 必须是 [B,seqlen,Ipack]")
    if tensors.get("g_dsa_ix_cache_pre", {}).get("shape") != ["B", "kvcachelen", "Ipack"]:
        issues.append(f"{label}: indexer packed 缓存前态必须是 [B,kvcachelen,Ipack]")
    if ops["g_dsa_ix_k_norm"].get("op") != "layernorm":
        issues.append(f"{label}: indexer k_norm 必须是 LayerNorm")
    if sorted(ops["g_dsa_ix_k_norm"].get("weights", [])) != \
            ["g_dsa_ix_k_norm_bias", "g_dsa_ix_k_norm_weight"]:
        issues.append(f"{label}: indexer k_norm 必须同时有 weight 与 bias")
    if "g_dsa_q_residual" not in ops["g_dsa_ix_q_proj"].get("inputs", []):
        issues.append(f"{label}: indexer Q 必须复用 q_residual")
    if "g_attn_normed" not in ops["g_dsa_ix_k_proj"].get("inputs", []):
        issues.append(f"{label}: indexer K 必须消费 Attention 的 hidden_states")
    # 池/tail/掩码语义
    if ops["g_dsa_ix_pool"].get("op") != "pool+softmax":
        issues.append(f"{label}: DSA indexer 必须是 k-pool 压缩（pool+softmax）")
    pool = ops["g_dsa_ix_pool"]
    if "g_dsa_ix_ape" not in pool.get("weights", []):
        issues.append(f"{label}: DSA k-pool 必须引用 index_kpool_compress_ape 权重")
    if "g_dsa_ix_visible" in pool.get("inputs", []):
        issues.append(f"{label}: DSA k-pool 不应直接依赖逐 query visible；只依赖 packed valid")
    if "g_dsa_ix_cache_post" not in pool.get("inputs", []):
        issues.append(f"{label}: DSA k-pool 必须消费 packed 缓存")
    if "g_dsa_ix_pool_end" not in ops or "g_dsa_ix_valid_candidates" not in ops:
        issues.append(f"{label}: DSA 缺少 pool_end/valid_candidates 具名依赖")
    elif "g_dsa_ix_pool_indices" not in ops["g_dsa_ix_pool_end"].get("inputs", []):
        issues.append(f"{label}: pool_end 必须由 pool_indices 求得")
    elif sorted(ops["g_dsa_ix_valid_candidates"].get("inputs", [])) != \
            ["g_dsa_ix_pool_end", "g_dsa_ix_pool_valid", "g_dsa_ix_visible"]:
        issues.append(f"{label}: valid_candidates 必须 gather(visible,pool_end) & pool_valid")
    mask = ops["g_dsa_ix_mask"]
    if sorted(mask.get("inputs", [])) != ["g_dsa_ix_index_scores", "g_dsa_ix_valid_candidates"]:
        issues.append(f"{label}: 池 mask 必须只消费 index_scores + valid_candidates")
    if "g_dsa_ix_selected_valid" not in ops:
        issues.append(f"{label}: DSA 缺少 selected_valid 来源")
    elif sorted(ops["g_dsa_ix_selected_valid"].get("inputs", [])) != \
            ["g_dsa_ix_selected", "g_dsa_ix_valid_candidates"]:
        issues.append(f"{label}: selected_valid 必须由 valid_candidates.gather(selected) 得到")
    if "g_dsa_ix_selected_valid" not in ops["g_dsa_ix_invalid"].get("inputs", []):
        issues.append(f"{label}: 无效池填 -1 必须消费 selected_valid")
    if "g_dsa_ix_attention_mask" in ops:
        issues.append(f"{label}: 不应出现伪造的 attention_mask 算子")
    if "g_dsa_ix_padded_indices" not in ops["g_dsa_ix_query_mask"].get("inputs", []) or \
            "g_attention_mask" not in ops["g_dsa_ix_query_mask"].get("inputs", []):
        issues.append(f"{label}: topk 必须经 attention_mask 的 query padding 末端过滤")
    if ops["g_dsa_ix_pad"].get("outputs") != ["g_dsa_ix_padded_indices"]:
        issues.append(f"{label}: pad 只产出 padded 索引；query 过滤是后续具名算子")
    if tensors.get("g_dsa_ix_pool_keys", {}).get("shape") != ["B", "Npool", "Id"]:
        issues.append(f"{label}: pool_keys 必须是 [B,Npool,Id]")
    if tensors.get("g_dsa_ix_pool_indices", {}).get("shape") != ["B", "Npool", "Kpool"]:
        issues.append(f"{label}: pool_indices 必须是 [B,Npool,Kpool]")
    if ops["g_dsa_ix_topk"].get("op") != "topk":
        issues.append(f"{label}: DSA indexer 池选择必须是 topk")
    if "Ki" not in ops["g_dsa_ix_topk"].get("parameter_refs", []):
        issues.append(f"{label}: DSA 池选择必须引用 Ki")
    if tensors.get("g_dsa_ix_tail_indices", {}).get("shape") != ["B", "seqlen", "Kpool-1"]:
        issues.append(f"{label}: tail 必须是 [B,seqlen,Kpool-1]")
    if tensors.get("g_dsa_topk_indices", {}).get("shape") != ["B", "seqlen", "Kout"]:
        issues.append(f"{label}: topk 输出必须是 [B,seqlen,Kout]")
    if ops["g_dsa_ix_visible"].get("op") != "compare+and":
        issues.append(f"{label}: DSA 因果可见性必须是 compare+and（kv_pos <= q_pos & valid）")
    counts = ops["g_dsa_mask_counts"]
    if counts.get("op") != "scatter+add":
        issues.append(f"{label}: 稀疏 mask 必须由 selected counts scatter_add 得到")
    if tensors.get("g_dsa_sparse_mask", {}).get("shape") != ["B", "1", "seqlen", "Lkv"]:
        issues.append(f"{label}: 稀疏 mask 必须是 [B,1,seqlen,Lkv]")
    mask_note = str(tensors.get("g_dsa_sparse_mask", {}).get("note", "")) + " " + \
        str(ops["g_dsa_sparse_mask"].get("note", ""))
    if "True" not in mask_note or "可见" not in mask_note:
        issues.append(f"{label}: 稀疏 mask 必须说明 True=可见（不是 True=屏蔽）")
    if "g_dsa_eager_mask" not in ops["g_dsa_mask_add"].get("inputs", []):
        issues.append(f"{label}: eager 注意力必须加稀疏可见 mask")
    # 注意力核心
    if "g_dsa_key" not in ops["g_dsa_qk"].get("inputs", []):
        issues.append(f"{label}: QK 未消费 key")
    if "Dq" not in ops["g_dsa_qk"].get("parameter_refs", []):
        issues.append(f"{label}: QK 缩放必须引用 Dq（1/sqrt(256)=1/16）")
    if "1/16" not in str(ops["g_dsa_qk"].get("note", "")):
        issues.append(f"{label}: QK 说明必须写明 scaling=1/16")
    if "g_dsa_attn_flat" not in ops["g_dsa_o_proj"].get("inputs", []):
        issues.append(f"{label}: o_proj 必须消费拼头后的 attn_flat")
    if tensors.get("g_dsa_attn_flat", {}).get("shape", [None])[-1] != "Nh×Dv":
        issues.append(f"{label}: attn_flat 尾维必须是 Nh×Dv=64×256")
    return issues


def _check_dense_mlp(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    for oid in ("g_dense_gate", "g_dense_up", "g_dense_clamp", "g_dense_act",
                "g_dense_down"):
        if oid not in ops:
            issues.append(f"{label}: dense 层缺少 {oid}")
    if "g_mlp_out" not in ops.get("g_dense_down", {}).get("outputs", []):
        issues.append(f"{label}: dense MLP 输出必须是 g_mlp_out")
    if any(oid.startswith("g_moe_") for oid in ops):
        issues.append(f"{label}: dense 层不得出现 MoE 算子")
    if "Swi" not in ops.get("g_dense_clamp", {}).get("parameter_refs", []):
        issues.append(f"{label}: dense SwiGLU 必须引用 Swi（swiglu_limit）")
    return issues


def _check_moe(block: dict, label: str) -> list[str]:
    issues: list[str] = []
    ops = _op_map(block)
    tensors = _tensor_map(block)
    order = _op_order(block)
    required = {
        "g_moe_flatten", "g_moe_router", "g_moe_sigmoid", "g_moe_bias_add",
        "g_moe_group_score", "g_moe_group_topk", "g_moe_group_scatter",
        "g_moe_group_expand", "g_moe_group_reshape", "g_moe_group_masked_fill",
        "g_moe_topk", "g_moe_weight_gather", "g_moe_weight_norm",
        "g_moe_weight_scale", "g_moe_expert_mask", "g_moe_expert_hit",
        "g_moe_expert_gather_indices", "g_moe_select_tokens",
        "g_moe_expert_gate_up", "g_moe_expert_chunk", "g_moe_expert_clamp",
        "g_moe_expert_act", "g_moe_expert_down", "g_moe_expert_weight",
        "g_moe_result_init", "g_moe_index_add", "g_moe_routed_view",
        "g_moe_shared_gate", "g_moe_shared_up", "g_moe_shared_clamp",
        "g_moe_shared_act", "g_moe_shared_down", "g_moe_merge",
    }
    missing = sorted(required - set(ops))
    if missing:
        issues.append(f"{label}: MoE 缺少算子 {missing}")
        return issues
    if ops["g_moe_sigmoid"].get("op") != "sigmoid":
        issues.append(f"{label}: 路由打分必须是 sigmoid")
    if order["g_moe_bias_add"] >= order["g_moe_topk"]:
        issues.append(f"{label}: correction bias 必须加在 top-k 之前")
    if order["g_moe_weight_norm"] >= order["g_moe_weight_scale"]:
        issues.append(f"{label}: 归一化必须在 routed_scaling_factor 之前")
    if "g_topk_indices" not in ops["g_moe_weight_gather"].get("inputs", []):
        issues.append(f"{label}: 路由权重必须按选中的索引 gather")
    for oid, expected in (("g_group_mask", ["Ttok", "Ng"]),
                          ("g_group_mask_expanded", ["Ttok", "Ng", "E÷Ng"]),
                          ("g_score_mask", ["Ttok", "E"])):
        if tensors.get(oid, {}).get("shape") != expected:
            issues.append(f"{label}: {oid} 形状必须是 {expected}")
    if tensors.get("g_expert_mask", {}).get("shape") != ["E+1", "K", "Ttok"]:
        issues.append(f"{label}: expert_mask 必须是完整 [E+1,K,Ttok] 分配矩阵")
    if ops["g_moe_expert_mask"].get("op") != "one_hot+permute":
        issues.append(f"{label}: 专家分配必须是 one_hot(top_k,E+1).permute(2,1,0)")
    if ops["g_moe_expert_gather_indices"].get("op") != "where":
        issues.append(f"{label}: 必须用 torch.where 取两个索引")
    if sorted(ops["g_moe_expert_gather_indices"].get("outputs", [])) != \
            ["g_expert_token_idx", "g_expert_topk_pos"]:
        issues.append(f"{label}: where 必须同时输出 token_idx 与 top_k_pos")
    for required_tid in ("g_expert_token_idx", "g_expert_topk_pos"):
        if tensors.get(required_tid, {}).get("shape") != ["N_e"]:
            issues.append(f"{label}: {required_tid} 必须是 [N_e] 数据相关索引")
    for required_tid in ("g_expert_token_idx", "g_expert_topk_pos", "g_topk_weights"):
        if required_tid not in ops["g_moe_expert_weight"].get("inputs", []):
            issues.append(f"{label}: 专家加权必须用 top_k_weights[token_idx, top_k_pos]")
    if "g_expert_token_idx" not in ops["g_moe_index_add"].get("inputs", []):
        issues.append(f"{label}: index_add 必须按 token_idx 回写")
    if "g_routed_init" not in ops["g_moe_index_add"].get("inputs", []):
        issues.append(f"{label}: index_add 必须有显式 zeros_like 初值")
    if tensors.get("g_moe_experts_gate_up", {}).get("shape") != ["E", "2Es", "H"]:
        issues.append(f"{label}: 专家 gate_up 必须是 [E,2Es,H]")
    if tensors.get("g_moe_experts_down", {}).get("shape") != ["E", "H", "Es"]:
        issues.append(f"{label}: 专家 down 必须是 [E,H,Es]")
    if "g_routed" not in ops["g_moe_merge"].get("inputs", []) or \
            "g_shared_out" not in ops["g_moe_merge"].get("inputs", []):
        issues.append(f"{label}: MoE 汇合必须是 routed + 共享专家")
    if "g_ffn_in" not in ops["g_moe_shared_gate"].get("inputs", []):
        issues.append(f"{label}: 共享专家必须消费 routed 之前的 residuals")
    if "Swi" not in ops["g_moe_expert_clamp"].get("parameter_refs", []):
        issues.append(f"{label}: 专家 SwiGLU 必须引用 Swi")
    if "Swi" not in ops["g_moe_shared_clamp"].get("parameter_refs", []):
        issues.append(f"{label}: 共享专家 SwiGLU 必须引用 Swi")
    if any("hash" in oid for oid in ops):
        issues.append(f"{label}: Flash 没有 hash 路由")
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


def _check_block(block: dict, label: str, *, legacy: bool = False) -> list[str]:
    issues: list[str] = []
    kind = block.get("kind")
    if kind not in ("kda", "dsa"):
        issues.append(f"{label}: block.kind 必须是 kda/dsa: {kind!r}")
        kind = "kda" if block.get("attention_type") == "linear_attention" else "dsa"
    issues.extend(hf_core.check_unit(block, label))
    issues.extend(hf_core.check_flow_units([block]))
    issues.extend(_check_no_foreign_facts(block, label))
    issues.extend(_check_param_facts(block, label, kind, legacy=legacy))
    issues.extend(_check_mhc(block, label))
    if kind == "kda":
        issues.extend(_check_kda(block, label))
        if block.get("moe_type") == "dense":
            issues.extend(_check_dense_mlp(block, label))
        else:
            issues.extend(_check_moe(block, label))
    else:
        issues.extend(_check_dsa(block, label))
        issues.extend(_check_moe(block, label))
    issues.extend(_check_source_refs(block, label))
    for element in ("operators", "tensors", "parameters"):
        for item in block.get(element, []):
            if element != "parameters" and not str(item.get("id", "")).startswith("g_"):
                issues.append(f"{label}: {element} id {item.get('id')!r} 未使用 g_ 命名空间")
    cache_blob = " ".join(block.get("cache_notes", []))
    for token in ("kvcachelen", "状态", "缓存"):
        if token not in cache_blob:
            issues.append(f"{label}: cache_notes 缺少关键事实 {token!r}")
    return issues


def validate_hf_layer_structure(document: dict) -> list[str]:
    """单层组 Block 结构（hf_layer_structure.v1）的 Flash 语义检查。"""
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
    attention_type = scope.get("attention_type")
    if attention_type not in ("linear_attention", "indexed_attention"):
        issues.append(f"scope.attention_type 必须是 linear_attention/indexed_attention: "
                      f"{attention_type!r}")
    indexer_type = scope.get("indexer_type")
    if attention_type == "indexed_attention":
        if indexer_type != "full":
            issues.append(f"DSA 层 scope.indexer_type 必须是 full: {indexer_type!r}")
    elif indexer_type != "none":
        issues.append(f"KDA 层 scope.indexer_type 必须是 none: {indexer_type!r}")
    moe_type = scope.get("moe_type")
    if moe_type not in ("dense", "sparse"):
        issues.append(f"scope.moe_type 必须是 dense/sparse: {moe_type!r}")
    if "phases" in document:
        issues.append("结构文档不得包含 phases（prefill/decode 只是场景投影）")
    scenarios = scope.get("scenarios", [])
    ids = [s.get("id") for s in scenarios]
    if len(ids) != len(set(ids)) or not ids:
        issues.append("场景预设 id 缺失或重复")
    for sid in ("prefill-empty", "decode-nonzero", "decode-long", "decode-multi"):
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
    if block.get("attention_type") != attention_type or block.get("moe_type") != moe_type:
        issues.append("block 的 attention_type/moe_type 与 scope 不一致")
    if block.get("indexer_type") != indexer_type:
        issues.append("block 的 indexer_type 与 scope 不一致")
    kind = block.get("kind")
    issues.extend(_check_block(block, "block"))
    issues.extend(_check_derivations(document, kind if kind in ("kda", "dsa") else
                                     ("kda" if attention_type == "linear_attention" else "dsa")))
    return issues


def validate_hf_reading_document(document: dict, input_shas: dict[str, str]) -> list[str]:
    """同源 prefill/decode 投影（hf_decoder_reading.v1）的 Flash 检查。"""
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
    indexer_type = scope.get("indexer_type")
    if attention_type == "indexed_attention" and indexer_type != "full":
        issues.append("DSA 层 scope.indexer_type 必须是 full")
    if attention_type == "linear_attention" and indexer_type != "none":
        issues.append("KDA 层 scope.indexer_type 必须是 none")
    moe_type = scope.get("moe_type")
    if moe_type not in ("dense", "sparse"):
        issues.append("scope.moe_type 非法")
    phases = document.get("phases", [])
    if [p.get("id") for p in phases] != ["prefill", "decode"]:
        issues.append("phases 必须依次为 prefill、decode")
    for phase in phases:
        pid = phase.get("id", "?")
        issues.extend(hf_core.check_unit(phase, pid))
        issues.extend(hf_core.check_flow_units([phase]))
        issues.extend(_check_no_foreign_facts(phase, pid))
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
    """整网总览（hf_network_overview.v1）的 Flash 检查。"""
    issues: list[str] = []
    if document.get("schema") != "hf_network_overview.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    issues.extend(_check_identity(document.get("model", {})))
    scope = document.get("scope", {})
    if scope.get("analysis_path") != "eager":
        issues.append("scope.analysis_path 必须是 eager")
    if scope.get("entry_class") != "Glm5NextForConditionalGeneration":
        issues.append(f"scope.entry_class 必须是 Glm5NextForConditionalGeneration: "
                      f"{scope.get('entry_class')!r}")
    if scope.get("main_module_class") != "Glm5NextTextModel":
        issues.append("scope.main_module_class 必须是 Glm5NextTextModel")
    if scope.get("decoder_layer_class") != "Glm5NextTextDecoderLayer":
        issues.append("scope.decoder_layer_class 必须是 Glm5NextTextDecoderLayer")
    layer_count = scope.get("layer_count")
    if layer_count != 45:
        issues.append(f"scope.layer_count 必须是 45（主干 0..44）: {layer_count!r}")
        layer_count = 45
    if scope.get("hc_mult") != 4:
        issues.append(f"scope.hc_mult 必须是 4: {scope.get('hc_mult')!r}")
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
    chain = ["net_embed", "net_position_ids", "net_mask_build", "net_hc_expand",
             "net_layer_loop", "net_hc_head", "net_final_norm", "net_logits_slice",
             "net_lm_head"]
    for oid in chain:
        if oid not in ops:
            issues.append(f"network: 缺少算子 {oid}")
    for first, second in zip(chain, chain[1:]):
        if order.get(first, 0) >= order.get(second, 0):
            issues.append(f"network: 顺序错误 {first} 必须在 {second} 之前")
    loop = ops.get("net_layer_loop", {})
    for required in ("hidden_streams", "padding_mask", "position_ids", "past_key_values"):
        if required not in loop.get("inputs", []):
            issues.append(f"network: 层循环未消费 {required}")
    if "Nlay" not in loop.get("parameter_refs", []):
        issues.append("network: 层循环必须引用 Nlay")
    if "g_hidden_in" in ops:
        issues.append("network: 不应出现 Block 级算子")
    if "net_rope" in ops or any("rope" in oid for oid in ops):
        issues.append("network: Flash 是 NoPE，不应出现网络级 RoPE 算子")
    if "glm" not in str(document.get("model", {}).get("display_name", "")).lower():
        issues.append("network: 显示名必须标明 GLM-5.3-Flash")
    blob = " ".join(scope.get("notes", []))
    for token in ("MTP", "KDA", "DSA", "mHC", "NoPE", "45"):
        if token not in blob:
            issues.append(f"network: scope.notes 缺少关键边界 {token!r}")
    if "视觉" not in blob and "图像" not in blob:
        issues.append("network: scope.notes 未声明视觉/图像路径不在范围")
    tensors = _tensor_map(path)
    if tensors.get("hidden_streams", {}).get("shape") != ["B", "seqlen", "Hc", "H"]:
        issues.append("network: hidden_streams 必须是 [B,seqlen,Hc,H]（4 路 mHC）")
    if tensors.get("logits", {}).get("shape") != ["B", "Ssel", "Vocab"]:
        issues.append("network: logits 必须是 [B,Ssel,Vocab]")

    groups = document.get("groups", [])
    if not isinstance(groups, list) or len(groups) != 3:
        issues.append(f"groups 必须是三种真实层型组合（实得 "
                      f"{len(groups) if isinstance(groups, list) else '?'}）")
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
        if group.get("attention_type") not in ("linear_attention", "indexed_attention"):
            issues.append(f"层组 {gid} attention_type 非法")
        covered.extend(indices)
    if sorted(covered) != list(range(layer_count)) or len(covered) != len(set(covered)):
        issues.append(f"三组未恰好覆盖 0..{layer_count - 1}（实得 {len(covered)} 项）")
    first_indices = [min(group.get("layer_indices") or [layer_count]) for group in groups]
    if first_indices != sorted(first_indices):
        issues.append("层组顺序必须按各组首层索引递增")
    expected_ids = {group.get("id") for group in groups}
    if expected_ids != {"kda-dense", "dsa-moe", "kda-moe"}:
        issues.append(f"层组 id 必须是 kda-dense/dsa-moe/kda-moe: {sorted(expected_ids)}")

    order_entries = document.get("layer_order", [])
    if [entry.get("index") for entry in order_entries] != list(range(layer_count)):
        issues.append("layer_order 必须逐一列出 0..44 的真实层顺序")
    group_of = {index: group["id"] for group in groups for index in group.get("layer_indices", [])}
    for entry in order_entries:
        index = entry.get("index")
        if entry.get("group_id") != group_of.get(index):
            issues.append(f"layer_order[{index}] 的组与 groups 归属不符")
        attention = entry.get("attention_type")
        if attention not in ("linear_attention", "indexed_attention"):
            issues.append(f"layer_order[{index}] attention_type 非法")
        if attention == "indexed_attention" and entry.get("indexer_type") != "full":
            issues.append(f"layer_order[{index}] DSA 必须是 full indexer")
        if attention == "linear_attention" and entry.get("indexer_type") != "none":
            issues.append(f"layer_order[{index}] KDA 不应有 indexer")

    dsa_from_groups = sorted(index for group in groups
                             if group.get("attention_type") == "indexed_attention"
                             for index in group.get("layer_indices", []))
    if dsa_from_groups != dsa_layer_indices(_ValuesProxy()):
        issues.append(f"DSA 层必须是 3,7,11,…,43: {dsa_from_groups}")
    if dsa_from_groups != sorted(scope.get("indexer_full_layers", [])):
        issues.append("scope.indexer_full_layers 与层组 DSA 集合不一致")
    if len(dsa_from_groups) != 11:
        issues.append(f"DSA 层数应为 11，实得 {len(dsa_from_groups)}")
    if scope.get("indexer_shared_layers"):
        issues.append("scope.indexer_shared_layers 必须为空（本配置全 full）")
    if scope.get("dense_layers") != [0, 1, 2]:
        issues.append(f"dense 层必须是 0/1/2: {scope.get('dense_layers')!r}")
    kda_from_groups = sorted(index for group in groups
                             if group.get("attention_type") == "linear_attention"
                             for index in group.get("layer_indices", []))
    if kda_from_groups != kda_layer_indices(_ValuesProxy()):
        issues.append(f"KDA 层集合不符: {kda_from_groups}")
    if len(kda_from_groups) != 34:
        issues.append(f"KDA 层数应为 34，实得 {len(kda_from_groups)}")

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


_OFFICIAL_LAYER_TYPES = [
    "indexed_attention" if (i % 4 == 3 and i < 44) else "linear_attention"
    for i in range(45)
]
assert len(_OFFICIAL_LAYER_TYPES) == 45
assert [i for i, t in enumerate(_OFFICIAL_LAYER_TYPES) if t == "indexed_attention"] == list(range(3, 44, 4))


class _ValuesProxy:
    """只提供 layer_types 的只读代理，用于独立列出官方 KDA/DSA 层号。"""

    num_hidden_layers = 45
    layer_types = _OFFICIAL_LAYER_TYPES
