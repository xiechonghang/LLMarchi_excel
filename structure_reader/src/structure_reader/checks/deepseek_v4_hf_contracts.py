"""HF 阅读数据语义契约检查（里程碑一）。

对 reading_document 做结构一致性与关键语义核查；任何一条不通过即给出
问题串（由 build 层具名拒绝），不静默修正。检查范围：

1. 文档骨架：schema/模型身份/两阶段顺序/引用闭合（通用图检查的 HF 等价）。
2. 数据流闭合（机器核查，不依赖说明文字）：
   - 每个算子的 activation/index/mask 输入必须有更早算位的 producer；
     boundary/weight/cache 豁免（外部输入、权重声明、缓存调用前状态）；
   - 算子不得把同一张量同时作为输入与输出（原地状态更新必须以
     调用前/调用后两个张量分别表达）；
   - eager AV 的概率输入必须是去 sink 概率张量（回放 wrong_av 变异拒绝）。
3. 关键语义锚点（按源码事实抽查阅读数据是否表达）：
   - mHC comb 残差方向必须出现 transpose（comb.T）语义；
   - 共享 KV：Nkv=1 且 cache_update 的 K/V 同输入；
   - 缓存读写顺序：先 update 主 KV 再拼接压缩条目；
   - eager 链顺序：QK→scale/mask→sink 拼接→softmax→去 sink→AV；
   - 压缩条目确定性 RoPE 位置（i·M+first_window_position）；
   - Indexer -1 哨兵与因果阈值；
   - 专家 clamp/SiLU·mul 合并 gate_up 与 index_add 汇合；
   - 部分旋转维（Rk=64 < Dh=512）与逆 RoPE（-sin）。
4. 来源清单与输入身份一致（SOURCE_FILES 键齐备；build 层另行复算指纹）。
"""

from __future__ import annotations

from pathlib import Path

from structure_reader.checks import hf_contract_core as hf_core
from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES

__all__ = [
    "validate_hf_layer_structure",
    "validate_hf_reading_document",
    "validate_hf_network_overview",
]

# 数据流检查的 producer 豁免类型：boundary=诚实外部输入边界；weight=权重声明；
# cache=缓存调用前状态（调用后状态由写算子产出，同样豁免缺 producer）。
_FLOW_EXEMPT_KINDS = frozenset({"boundary", "weight", "cache"})


def validate_hf_reading_document(document: dict, input_shas: dict[str, str]) -> list[str]:
    """返回问题串列表；空列表 = 通过。"""
    issues: list[str] = []
    issues.extend(_check_skeleton(document))
    issues.extend(_check_flow_units(document.get("phases", [])))
    issues.extend(_check_semantics(document))
    issues.extend(_check_inputs(input_shas))
    return issues


def validate_hf_network_overview(document: dict, input_shas: dict[str, str]) -> list[str]:
    """整网总览（hf_network_overview.v1）契约检查；空列表 = 通过。

    检查范围：身份/schema、单条 token→logits 路径的引用闭合与数据流、
    四类层组与真实层顺序的恰好覆盖、外围与终点必须对应源码锚点、
    到既有 Block/算子详图的往返目标、输入身份完整。
    """
    issues: list[str] = []
    issues.extend(_check_network_skeleton(document))
    path = document.get("path")
    if isinstance(path, dict):
        issues.extend(_check_unit(path, path.get("id", "network")))
        issues.extend(_check_flow_units([path]))
    issues.extend(_check_network_semantics(document))
    issues.extend(_check_inputs(input_shas))
    return issues


def _check_unit(unit: dict, label: str) -> list[str]:
    """一个阶段/路径的骨架检查（通用实现在 checks/hf_contract_core.py）。"""
    return hf_core.check_unit(unit, label)


def _check_skeleton(document: dict) -> list[str]:
    issues = []
    if document.get("schema") != "hf_decoder_reading.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    model = document.get("model", {})
    for field in ("model_key", "display_name", "implementation", "source_revision", "config_path"):
        if not isinstance(model.get(field), str) or not model[field]:
            issues.append(f"model.{field} 缺失或非字符串")
    if model.get("implementation") != "transformers":
        issues.append(f"model.implementation 必须是 transformers: {model.get('implementation')!r}")
    scope = document.get("scope", {})
    if scope.get("analysis_path") != "eager":
        issues.append(f"scope.analysis_path 必须是 eager: {scope.get('analysis_path')!r}")
    if not isinstance(scope.get("layer_index"), int) or scope["layer_index"] < 0:
        issues.append("scope.layer_index 必须是非负整数")
    phases = document.get("phases", [])
    if [p.get("id") for p in phases] != ["prefill", "decode"]:
        issues.append("phases 必须依次为 prefill、decode")
    for phase in phases:
        issues.extend(_check_unit(phase, phase.get("id", "?")))
    return issues


def _check_flow_units(units) -> list[str]:
    """数据流闭合检查（通用实现在 checks/hf_contract_core.py）。"""
    return hf_core.check_flow_units(units)


def _check_network_skeleton(document: dict) -> list[str]:
    """整网总览骨架：身份、单路径、四类层组恰好覆盖、真实层序、详情往返目标。"""
    issues: list[str] = []
    if document.get("schema") != "hf_network_overview.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    model = document.get("model", {})
    for field in ("model_key", "display_name", "implementation", "source_revision", "config_path"):
        if not isinstance(model.get(field), str) or not model[field]:
            issues.append(f"model.{field} 缺失或非字符串")
    if model.get("implementation") != "transformers":
        issues.append(f"model.implementation 必须是 transformers: {model.get('implementation')!r}")
    scope = document.get("scope", {})
    if scope.get("analysis_path") != "eager":
        issues.append(f"scope.analysis_path 必须是 eager: {scope.get('analysis_path')!r}")
    if scope.get("entry_class") != "DeepseekV4ForCausalLM":
        issues.append(f"scope.entry_class 必须是 DeepseekV4ForCausalLM: {scope.get('entry_class')!r}")
    layer_count = scope.get("layer_count")
    if not isinstance(layer_count, int) or layer_count <= 0:
        issues.append("scope.layer_count 必须是正整数")
        layer_count = 0

    path = document.get("path")
    if not isinstance(path, dict):
        issues.append("path 缺失（整网总览必须提供单条 token→logits 路径）")
    elif path.get("id") != "network":
        issues.append(f"path.id 必须是 'network': {path.get('id')!r}")

    groups = document.get("groups", [])
    if not isinstance(groups, list) or len(groups) != 4:
        issues.append(f"groups 必须是四类真实层组（实得 {len(groups) if isinstance(groups, list) else '?'} 组）")
        return issues
    covered: list[int] = []
    for group in groups:
        gid = group.get("id")
        indices = group.get("layer_indices", [])
        if not gid:
            issues.append("层组缺少 id")
        if not indices:
            issues.append(f"层组 {gid} 没有真实层（空组不能作为模板）")
        if group.get("count") != len(indices):
            issues.append(f"层组 {gid} count={group.get('count')!r} 与 layer_indices 长度 {len(indices)} 不符")
        if group.get("representative_layer_index") not in indices:
            issues.append(f"层组 {gid} 代表层 {group.get('representative_layer_index')!r} 不属于该组")
        covered.extend(indices)
    if sorted(covered) != list(range(layer_count)):
        issues.append(
            f"四类层组未恰好覆盖 0..{layer_count - 1}（实得 {len(covered)} 项，"
            f"去重 {len(set(covered))}）"
        )
    if len(covered) != len(set(covered)):
        issues.append("层组之间层索引重复")
    first_indices = [min(group.get("layer_indices") or [layer_count])
                     for group in groups]
    if first_indices != sorted(first_indices):
        issues.append("四类层组顺序未按各组首层索引递增（应与首层出现顺序一致）")

    order = document.get("layer_order", [])
    if [entry.get("index") for entry in order] != list(range(layer_count)):
        issues.append("layer_order 必须逐一列出 0..N-1 的真实层顺序")
    group_of = {index: group["id"] for group in groups
                for index in group.get("layer_indices", [])}
    for entry in order:
        index = entry.get("index")
        if entry.get("group_id") != group_of.get(index):
            issues.append(f"layer_order[{index}] 的组 {entry.get('group_id')!r} 与 groups 归属不符")
    expected_ids = [group["id"] for group in groups]
    detail = document.get("detail", {})
    href = detail.get("href")
    if not isinstance(href, str) or not href.endswith(".html"):
        issues.append(f"detail.href 必须是既有详图 HTML 入口: {href!r}")
    detail_groups = detail.get("groups", [])
    if [g.get("id") for g in detail_groups] != expected_ids:
        issues.append("detail.groups 必须与四类层组同序一一对应")
    for group in detail_groups:
        anchors = group.get("anchors", {})
        for key in ("group", "block", "operators", "tensors"):
            value = anchors.get(key)
            if not isinstance(value, str) or not value:
                issues.append(f"detail.groups[{group.get('id')}].anchors.{key} 缺失")
    return issues


def _check_network_semantics(document: dict) -> list[str]:
    """整网外围/终点/连接的关键锚点：必须对应所选源码，不能是普通残差模板。"""
    issues: list[str] = []
    path = document.get("path", {})
    ops = {o.get("id"): o for o in path.get("operators", [])}
    tensors = {t.get("id"): t for t in path.get("tensors", [])}
    params = {p.get("id"): p for p in path.get("parameters", [])}

    required = (
        "net_embed", "net_position_ids", "net_mask_build", "net_rope_main",
        "net_rope_compress", "net_hc_expand", "net_layer_loop", "net_hc_head_norm",
        "net_hc_head_project", "net_hc_head_pre", "net_hc_head_collapse",
        "net_final_norm", "net_logits_slice", "net_lm_head",
    )
    for oid in required:
        if oid not in ops:
            issues.append(f"network: 缺少整网主干算子 {oid}")

    def locators(oid: str) -> str:
        return " ".join(str(r.get("locator", "")) for r in ops.get(oid, {}).get("source_refs", []))

    def files(oid: str) -> str:
        return " ".join(str(r.get("path", "")) for r in ops.get(oid, {}).get("source_refs", []))

    # 1) 入口 embedding 与终点 lm_head/logits 必须对应 modeling 真实行
    for oid, needle in (("net_embed", "1333"), ("net_lm_head", "1511"),
                        ("net_logits_slice", "1510"),
                        ("net_final_norm", "1368"),
                        ("net_layer_loop", "1357-1366"),
                        ("net_hc_head_collapse", "1003"),
                        ("net_hc_expand", "1351")):
        if oid in ops and needle not in locators(oid):
            issues.append(f"network: {oid} 来源未定位到 modeling:{needle}，实得 {locators(oid)!r}")
        if oid in ops and "modeling_deepseek_v4.py" not in files(oid):
            issues.append(f"network: {oid} 来源必须指向 modeling_deepseek_v4.py")

    # 2) 入口/终点张量
    for tid, needles in (
        ("input_ids", ("input_ids",)),
        ("inputs_embeds", ("inputs_embeds",)),
        ("hidden_streams", ("hidden_states", "mHC")),
        ("causal_mask", ("causal_mask",)),
        ("logits", ("logits",)),
    ):
        info = tensors.get(tid)
        if info is None:
            issues.append(f"network: 缺少边界/主干张量 {tid}")
            continue
        blob = str(info.get("name", "")) + str(info.get("note", ""))
        for needle in needles:
            if needle not in blob:
                issues.append(f"network: 张量 {tid} 未表达 {needle!r}")

    if "logits" in tensors:
        if tensors["logits"].get("parameter_refs")[:1] != ["Vocab"] and "Vocab" not in \
                tensors["logits"].get("parameter_refs", []):
            issues.append("network: logits 形状必须引用词表参数 Vocab")
        if tensors["logits"].get("shape", [])[:2] != ["B", "Ssel"]:
            issues.append(f"network: logits 形状应为 [B,S,Vocab]，实得 {tensors['logits'].get('shape')!r}")

    # 3) mHC 不是普通残差：网络入口扩展 + 出口 HyperHead 收拢，且层循环保持多流
    if "hidden_streams" in tensors:
        if "Nhc" not in tensors["hidden_streams"].get("parameter_refs", []):
            issues.append("network: mHC 流扩展张量必须引用 hc_mult 参数 Nhc")
        if len(tensors["hidden_streams"].get("shape", [])) != 4:
            issues.append("network: mHC 流张量应为 4 维 [B,seqlen,Nhc,H]")
    if "hc_head_collapsed" not in tensors:
        issues.append("network: 缺少 HyperHead 收拢后的单流张量 hc_head_collapsed")
    else:
        if tensors["hc_head_collapsed"].get("shape") != ["B", "seqlen", "H"]:
            issues.append(
                f"network: HyperHead 收拢后应为 [B,seqlen,H]，实得 "
                f"{tensors['hc_head_collapsed'].get('shape')!r}")
    loop_outputs = set(ops.get("net_layer_loop", {}).get("outputs", []))
    for tid in ("streams_after_layers", "past_key_values_post"):
        if tid not in loop_outputs:
            issues.append(f"network: 层循环必须输出 {tid}")

    # 4) 层循环真实连接：层输入（mask/位置/两组 RoPE/input_ids/cache）与次数
    loop_inputs = set(ops.get("net_layer_loop", {}).get("inputs", []))
    for tid in ("hidden_streams", "causal_mask", "position_ids", "cos_main", "sin_main",
                "cos_compress", "sin_compress", "input_ids", "past_key_values"):
        if tid not in loop_inputs:
            issues.append(f"network: 层循环未接入真实依赖 {tid}")
    if params.get("Nlay", {}).get("value") != document.get("scope", {}).get("layer_count"):
        issues.append("network: 参数 Nlay（循环次数）必须等于 scope.layer_count")
    if "Nlay" not in ops.get("net_layer_loop", {}).get("parameter_refs", []):
        issues.append("network: 层循环算子必须引用循环次数参数 Nlay")

    # 5) lm_head 与 embedding 是两份独立权重（tie_word_embeddings=false）
    if "embed_tokens_weight" in tensors and "lm_head_weight" in tensors:
        note = (str(tensors["embed_tokens_weight"].get("note", ""))
                + str(tensors["lm_head_weight"].get("note", "")))
        if "tie_word_embeddings=false" not in note:
            issues.append("network: 必须说明 tie_word_embeddings=false，embed/lm_head 权重独立")
        if tensors["embed_tokens_weight"].get("shape") != tensors["lm_head_weight"].get("shape"):
            issues.append("network: embed_tokens 与 lm_head 权重形状应同为 [Vocab,H]")

    # 6) 采样/训练不在主干（终点是 logits）
    if "logits" not in ops.get("net_lm_head", {}).get("outputs", []):
        issues.append("network: net_lm_head 必须直接产出 logits")

    # 7) logits_to_keep 越界：正整数长度必须按 min(seqlen, Lk) 裁剪
    #    （源码 :1510 slice(-Lk, None)；seqlen=1、Lk=5 仍只有 1 个位置）
    if "net_logits_slice" in ops:
        slice_blob = (
            str(ops["net_logits_slice"].get("note", ""))
            + str(params.get("Ssel", {}).get("expression", ""))
            + str(params.get("Ssel", {}).get("note", ""))
        )
        if "min(seqlen, Lk)" not in slice_blob:
            issues.append(
                "network: logits 切片未表达越界裁剪 min(seqlen, Lk)"
                "（源码 :1510 slice(-Lk, None)，Lk 大于 seqlen 时不扩大长度）"
            )
    return issues


def _check_semantics(document: dict) -> list[str]:
    """关键语义锚点：每个阶段抽查阅读数据确实表达了源码确定的结构事实。"""
    issues = []
    for phase in document.get("phases", []):
        pid = phase.get("id", "?")
        ops = {o.get("id"): o for o in phase.get("operators", [])}
        tensors = {t.get("id"): t for t in phase.get("tensors", [])}
        params = {p.get("id"): p for p in phase.get("parameters", [])}
        notes_blob = " ".join(
            str(o.get("note", "")) + str(o.get("name", "")) for o in phase.get("operators", [])
        )

        def op_note(*op_ids: str) -> str:
            return " ".join(
                str(ops[oid].get("note", "")) for oid in op_ids if oid in ops
            )

        # 1) mHC comb 转置方向
        for seg in ("attn", "ffn"):
            mix_id = f"{seg}_residual_mix"
            if mix_id not in ops:
                issues.append(f"{pid}: 缺少 {mix_id}（mHC 残差混合算子）")
            else:
                blob = op_note(mix_id, f"{seg}_residual_combT") + str(ops[mix_id].get("name", ""))
                if "transpose" not in blob and "combᵀ" not in blob and "comb.T" not in blob:
                    issues.append(f"{pid}: {mix_id} 未表达 comb 转置方向（源码 :1178/:1184）")

        # 2) 共享 KV（Nkv=1）与 K/V 同输入缓存更新
        nkv = params.get("Nkv", {}).get("value")
        if nkv != 1:
            issues.append(f"{pid}: 参数 Nkv 必须=1（共享 KV MQA），实得 {nkv!r}")
        if "cache_update" not in ops:
            issues.append(f"{pid}: 缺少 cache_update（主 KV 缓存更新）")
        else:
            upd = ops["cache_update"]
            if upd.get("inputs", [])[:1] != ["kv_rope"] and "kv_rope" not in upd.get("inputs", []):
                issues.append(f"{pid}: cache_update 输入应含 kv_rope（K=V 同一张量）")

        # 3) 缓存读与压缩条目拼接顺序（cat 在 update 之后）
        if "cache_update" in ops and "kv_all" in tensors:
            order_upd = next(
                (i for i, o in enumerate(phase["operators"]) if o.get("id") == "cache_update"), -1
            )
            order_cat = next(
                (i for i, o in enumerate(phase["operators"]) if o.get("id") == "kv_concat"),
                -1,
            )
            if order_cat == -1:
                issues.append(f"{pid}: 缺少主 KV+压缩条目拼接算子（kv_concat）")
            elif order_cat < order_upd:
                issues.append(f"{pid}: 压缩条目拼接必须发生在缓存 update 之后（源码 :822→:829）")

        # 4) eager 链关键锚点
        required_chain = ("qk_scaled", "mask_add", "sink_softmax", "av_matmul")
        for oid in required_chain:
            if oid not in ops:
                issues.append(f"{pid}: 缺少 eager 链算子 {oid}")
        if "sink_softmax" in ops:
            blob = str(ops["sink_softmax"].get("note", "")) + str(ops["sink_softmax"].get("op", ""))
            for token in ("cat", "softmax", "sink"):
                if token not in blob.lower():
                    issues.append(f"{pid}: sink_softmax 未表达 {token}（sink 拼接→softmax→去 sink）")
        # 4b) AV 概率输入必须是去 sink 概率张量（拒绝把 q_a 等张量冒充概率输入）
        if "av_matmul" in ops:
            av_inputs = ops["av_matmul"].get("inputs", [])
            if not av_inputs or av_inputs[0] != "attn_probs":
                issues.append(
                    f"{pid}: av_matmul 概率输入必须是 attn_probs（去 sink 概率），"
                    f"实得 {av_inputs[:1]!r}"
                )
            if "kv_repeated" not in av_inputs:
                issues.append(f"{pid}: av_matmul 输入应含 kv_repeated（广播后的 V）")

        # 5) 压缩条目确定性 RoPE 位置
        attn_type = document.get("scope", {}).get("attention_type")
        if attn_type == "sliding_attention":
            if "c_rope" in ops:
                issues.append(f"{pid}: SWA 不应有压缩条目 RoPE c_rope")
        elif "c_rope" in ops:
            blob = str(ops["c_rope"].get("note", "")) + str(ops["c_rope"].get("op", ""))
            if "first_window_position" not in blob and "i·M" not in blob and "i*M" not in blob:
                issues.append(f"{pid}: c_rope 未表达确定性压缩位置（i·M+first_window_position）")
        else:
            issues.append(f"{pid}: 缺少压缩条目 RoPE 算子 c_rope")

        attn_type = document.get("scope", {}).get("attention_type")
        # 6) Indexer 哨兵与 top-k（仅 CSA；HCA 无 Indexer）
        if attn_type in ("heavily_compressed_attention", "sliding_attention"):
            pass
        elif "i_topk" in ops:
            blob = str(ops["i_topk"].get("note", "")) + str(ops["i_topk"].get("op", ""))
            if "-1" not in blob:
                issues.append(f"{pid}: i_topk 未表达 -1 哨兵（源码 :574-581）")
        else:
            issues.append(f"{pid}: 缺少 Indexer top-k 算子 i_topk")

        # 7) 专家模板链（单份模板内的必要算子行；clamp 同类合并、SiLU 具名）
        expert_chain = ("moe_select_tokens", "moe_expert_slice", "moe_expert_gu",
                        "moe_expert_clamp", "moe_expert_act", "moe_expert_down",
                        "moe_expert_weight", "moe_combine")
        for oid in expert_chain:
            if oid not in ops:
                issues.append(f"{pid}: 缺少专家模板链算子 {oid}")
        # 7b) 寻址链：当前专家二维切片与路由权重双索引（HF :1045-1052）
        if "moe_expert_slice" in ops:
            slice_op = ops["moe_expert_slice"]
            if set(slice_op.get("outputs", [])) != {"expert_gate_up_e", "expert_down_e"}:
                issues.append(f"{pid}: moe_expert_slice 必须输出当前专家二维切片 "
                              "expert_gate_up_e/expert_down_e")
            if "hit_experts" not in slice_op.get("inputs", []):
                issues.append(f"{pid}: moe_expert_slice 未消费命中专家索引 hit_experts（e 来源）")
        else:
            issues.append(f"{pid}: 两个专家 Linear 不得直接消费 3D 整库（缺切片算子）")
        if "moe_expert_gu" in ops:
            if ops["moe_expert_gu"].get("weights") != ["expert_gate_up_e"]:
                issues.append(f"{pid}: moe_expert_gu 权重必须是当前专家切片 "
                              f"expert_gate_up_e，实得 {ops['moe_expert_gu'].get('weights')!r}")
        if "moe_expert_down" in ops:
            if ops["moe_expert_down"].get("weights") != ["expert_down_e"]:
                issues.append(f"{pid}: moe_expert_down 权重必须是当前专家切片 "
                              f"expert_down_e，实得 {ops['moe_expert_down'].get('weights')!r}")
        if "moe_expert_weight" in ops:
            w_inputs = set(ops["moe_expert_weight"].get("inputs", []))
            if "expert_token_idx" not in w_inputs or "expert_topk_pos" not in w_inputs:
                issues.append(f"{pid}: moe_expert_weight 路由权重寻址需 token_idx+top_k_pos "
                              f"双索引（top_k_weights[token_idx, top_k_pos, None]），实得 {sorted(w_inputs)}")
        if "moe_expert_clamp" in ops:
            blob = str(ops["moe_expert_clamp"].get("op", "")) + str(ops["moe_expert_clamp"].get("note", ""))
            for token in ("chunk", "clamp"):
                if token not in blob.lower():
                    issues.append(f"{pid}: moe_expert_clamp 未表达 {token}")
        if "moe_expert_act" in ops:
            blob = str(ops["moe_expert_act"].get("op", "")) + str(ops["moe_expert_act"].get("note", ""))
            for token in ("silu", "mul"):
                if token not in blob.lower():
                    issues.append(f"{pid}: moe_expert_act 未表达 {token}")
        if "moe_combine" in ops:
            combine = ops["moe_combine"]
            blob = str(combine.get("op", "")) + str(combine.get("note", ""))
            if "index_add" not in blob:
                issues.append(f"{pid}: moe_combine 未表达 index_add 汇合")
            if "expert_token_idx" not in combine.get("inputs", []):
                issues.append(f"{pid}: moe_combine 未消费 token_idx（index_add 寻址）")
            if "topk_weights" in combine.get("inputs", []):
                issues.append(f"{pid}: moe_combine 重复消费 topk_weights（加权应在模板内）")
        if "moe_routed_view" not in ops:
            issues.append(f"{pid}: 缺少 routed 还原 3D 视图算子 moe_routed_view（源码 :1133）")
        else:
            merge = ops.get("moe_merge")
            if merge is None:
                issues.append(f"{pid}: 缺少 moe_merge（路由+共享汇合）")
            elif "routed_view" not in merge.get("inputs", []):
                issues.append(f"{pid}: moe_merge 必须消费 view 后的 routed_view（源码 :1133-1134）")
        if "moe_select_tokens" in ops:
            if "ffn_flat" not in ops["moe_select_tokens"].get("inputs", []):
                issues.append(f"{pid}: 专家模板必须经 moe_select_tokens 读真实 token 行")

        # 8) 部分旋转与逆 RoPE
        rk = params.get("Rk", {}).get("value")
        dh = params.get("Dh", {}).get("value")
        if rk is None or dh is None or not (0 < rk < dh):
            issues.append(f"{pid}: 部分旋转参数异常（Rk={rk!r}, Dh={dh!r}，需 0<Rk<Dh）")
        if "attn_inv_rope" not in ops:
            issues.append(f"{pid}: 缺少逆 RoPE 算子 attn_inv_rope（conjugate -sin）")
        else:
            blob = str(ops["attn_inv_rope"].get("note", "")) + str(ops["attn_inv_rope"].get("op", ""))
            if "-sin" not in blob and "conjugate" not in blob and "共轭" not in blob:
                issues.append(f"{pid}: attn_inv_rope 未表达共轭旋转（-sin）")

        # 8b) HCA 专属：非重叠 128、head_dim、无 Indexer/overlap、历史返回、bias 条件
        if attn_type == "heavily_compressed_attention":
            if params.get("M", {}).get("value") != 128:
                issues.append(f"{pid}: HCA compress_rate 必须=128，实得 {params.get('M', {}).get('value')!r}")
            for t in tensors.values():
                if str(t.get("id", "")).startswith(("i_", "cache_overlap_i")):
                    issues.append(f"{pid}: HCA 不应有 Indexer/overlap 张量 {t.get('id')}")
            if "cache_buffer_i_kv" in tensors or "cache_overlap_c_kv" in tensors:
                issues.append(f"{pid}: HCA 不应有 indexer/overlap 缓存张量")
            if "cache_values" in tensors:
                issues.append(f"{pid}: HCA 不应新增独立 cache_values（self.values=self.keys 别名）")
            upd = ops.get("cache_update", {})
            blob = str(upd.get("note", ""))
            if "self.values = self.keys" not in blob and "同一存储" not in blob:
                issues.append(f"{pid}: HCA cache_update 未表达 values=keys 同存储")
            if "c_rope_gen" not in ops:
                issues.append(f"{pid}: HCA 缺少条目位置生成算子 c_rope_gen")
            else:
                rg = ops["c_rope_gen"]
                if "cache_entry_count_c" not in rg.get("inputs", []):
                    issues.append(f"{pid}: c_rope_gen 必须读 entry_count 生成 first_window_position")
                if "first_window_position" not in rg.get("outputs", []):
                    issues.append(f"{pid}: c_rope_gen 未输出 first_window_position")
            cro = ops.get("c_rope", {})
            if "position_ids" in cro.get("inputs", []):
                issues.append(f"{pid}: c_rope 不得把 query position_ids 当条目位置")
            if not any("c_cos" == x or "c_sin" == x for x in cro.get("inputs", [])):
                issues.append(f"{pid}: c_rope 未消费条目 cos/sin")
            if "kv_proj" in ops:
                if "kv_norm_weight" not in ops["kv_proj"].get("weights", []):
                    issues.append(f"{pid}: HCA kv_proj 未绑定 kv_norm_weight（Linear+RMSNorm）")
                if "rmsnorm" not in str(ops["kv_proj"].get("op", "")):
                    issues.append(f"{pid}: HCA kv_proj 类型未列 RMSNorm")
            mm = ops.get("mask_model_build", {})
            refs = " ".join(str(r.get("path", "")) for r in mm.get("source_refs", []))
            if "masking_utils.py" not in refs:
                issues.append(f"{pid}: mask_model_build 来源应为 masking_utils.py")
            cm = ops.get("c_mask_extend", {})
            cref = " ".join(str(r.get("locator", "")) for r in cm.get("source_refs", []))
            if "837-841" not in cref:
                issues.append(f"{pid}: c_mask_extend 来源应为 Attention.forward:837-841")
            if any("cache_buffer_c_kv_post2" == t.get("id") for t in tensors.values()):
                issues.append(f"{pid}: 残留无消费者假张量 cache_buffer_c_kv_post2")
            for tid in ("c_kv", "c_gate"):
                shp = tensors.get(tid, {}).get("shape", [])
                if not shp or shp[-1] != "Dh":
                    issues.append(f"{pid}: HCA {tid} 末维应为 head_dim(Dh)，实得 {shp!r}")
            for tid in ("c_kv_weight", "c_gate_weight"):
                shp = tensors.get(tid, {}).get("shape", [])
                if shp != ["Dh", "H"]:
                    issues.append(f"{pid}: HCA {tid} 应为 [head_dim, hidden_size]，实得 {shp!r}")
            pb = tensors.get("c_position_bias", {}).get("shape", [])
            if pb != ["M", "Dh"]:
                issues.append(f"{pid}: HCA position_bias 应为 [compress_rate, head_dim]，实得 {pb!r}")
            up = ops.get("c_update_states", {})
            blob = str(up.get("note", "")) + str(up.get("op", ""))
            if "num_full_windows=0" not in blob or "历史" not in blob:
                issues.append(f"{pid}: HCA update_states 未表达 N_win=0 仍返回历史")
            bb = str(ops.get("hca_block_bias", {}).get("note", ""))
            if "None" not in bb:
                issues.append(f"{pid}: HCA block_bias 未表达 None 条件")
            if any(str(o.get("id", "")).startswith("i_") for o in phase.get("operators", [])):
                issues.append(f"{pid}: HCA 不应包含 Indexer 算子（i_*）")

        # 8c) SWA：无压缩/Indexer、独立 keys/values、main RoPE
        if attn_type == "sliding_attention":
            for t in tensors.values():
                tid = str(t.get("id", ""))
                if tid.startswith(("c_", "i_", "cache_buffer_c", "cache_entry_count")) or tid in (
                        "cache_compressed_c", "block_bias", "kv_all"):
                    issues.append(f"{pid}: SWA 不应有压缩/Indexer 元素 {tid}")
            for o in ops:
                if o.startswith(("c_", "i_")):
                    issues.append(f"{pid}: SWA 不应有压缩/Indexer 算子 {o}")
            if "cache_values" not in tensors:
                issues.append(f"{pid}: SWA 缺少独立 cache_values 字段")
            else:
                note = str(tensors["cache_values"].get("note", ""))
                if "独立" not in note:
                    issues.append(f"{pid}: SWA cache_values 未注明独立持久字段")
            if params.get("ThetaM", {}).get("value") != 10000:
                issues.append(f"{pid}: SWA main RoPE theta 应为 10000")
            if "kv_concat" in ops:
                issues.append(f"{pid}: SWA 不应有压缩 KV 拼接 kv_concat")
        # 8d) Hash：input_ids/tid2eid 查表 + scores gather，无 TopK/bias
        if document.get("scope", {}).get("moe_type") == "hash_moe":
            if "input_ids" not in tensors:
                issues.append(f"{pid}: Hash 缺少 input_ids 边界")
            if "tid2eid" not in tensors:
                issues.append(f"{pid}: Hash 缺少 tid2eid 持久 buffer")
            if "topk_indices" not in tensors:
                issues.append(f"{pid}: Hash 缺少查表 indices")
            lookup = ops.get("moe_hash_lookup")
            if lookup is None:
                issues.append(f"{pid}: Hash 缺少 tid2eid 查表算子")
            elif "input_ids" not in lookup.get("inputs", []):
                issues.append(f"{pid}: Hash 查表未消费 input_ids")
            if "router_bias" in tensors:
                issues.append(f"{pid}: Hash 不应有 correction bias")
            r = ops.get("moe_router", {})
            if "topk" in str(r.get("op", "")).lower():
                issues.append(f"{pid}: Hash 路由算子类型不应是 TopK")

        # 9) sink 参数与共享 KV 投影（单 KV 头）
        if "sinks" not in tensors:
            issues.append(f"{pid}: 缺少 sinks 权重张量（每头注意力 sink）")
        if "kv_proj" in ops:
            blob = str(ops["kv_proj"].get("note", ""))
            if "num_key_value_heads=1" not in blob and "单 KV 头" not in blob:
                issues.append(f"{pid}: kv_proj 未说明单 KV 头投影")
    return issues


def _check_inputs(input_shas: dict[str, str]) -> list[str]:
    """输入身份检查（通用实现；键集来自本模型 SOURCE_FILES）。"""
    return hf_core.check_input_identity(input_shas, SOURCE_FILES)


def validate_hf_layer_structure(document: dict) -> list[str]:
    """分层浏览的 Block 结构契约检查（hf_layer_structure.v1）；空列表 = 通过。

    检查重点（对应 T03 验收）：
    - 一份结构：只允许一个 block 单元，不得有按 prefill/decode 复制的 phases；
    - 场景参数身份：seqlen=本次新 token 数、kvcachelen=逻辑历史数（空缓存 Prefill 为 0），
      结构里保留符号、不带阶段写死的取值；
    - shape 词表统一：shape 轴不得残留 T/P 旧符号（避免「只改显示名」）；
    - 派生量：表达式在白名单求值器内，具体场景下可确定，符号场景下如实未知；
    - 场景预设：id 唯一、含 seqlen/kvcachelen、默认场景存在。
    """
    from structure_reader.scenario_evaluator import ExpressionError, check_expression

    issues: list[str] = []
    if document.get("schema") != "hf_layer_structure.v1":
        issues.append(f"schema 非法: {document.get('schema')!r}")
    model = document.get("model", {})
    for field in ("model_key", "display_name", "implementation", "source_revision", "config_path"):
        if not isinstance(model.get(field), str) or not model[field]:
            issues.append(f"model.{field} 缺失或非字符串")
    if model.get("implementation") != "transformers":
        issues.append(f"model.implementation 必须是 transformers: {model.get('implementation')!r}")
    scope = document.get("scope", {})
    if scope.get("analysis_path") != "eager":
        issues.append(f"scope.analysis_path 必须是 eager: {scope.get('analysis_path')!r}")
    if not isinstance(scope.get("layer_index"), int) or scope["layer_index"] < 0:
        issues.append("scope.layer_index 必须是非负整数")
    if not scope.get("attention_type") or not scope.get("moe_type"):
        issues.append("scope.attention_type/moe_type 必须写明真实层型")
    if "phases" in document:
        issues.append("结构文档不得包含 phases（prefill/decode 只是场景投影，不是两套结构）")

    block = document.get("block")
    if not isinstance(block, dict):
        issues.append("block 缺失（必须是一份参数化 Block 结构）")
        return issues
    if block.get("id") != "block":
        issues.append(f"block.id 必须是 'block'（一份结构）: {block.get('id')!r}")
    issues.extend(_check_unit(block, "block"))

    params = {p.get("id"): p for p in block.get("parameters", [])}
    for pid in ("seqlen", "kvcachelen"):
        if pid not in params:
            issues.append(f"缺少场景参数 {pid}")
        elif params[pid].get("value") is not None:
            issues.append(f"结构参数 {pid} 必须保留符号（value=None），场景预设另行绑定")
    seq_note = str(params.get("seqlen", {}).get("note", "")) + str(
        params.get("seqlen", {}).get("meaning", ""))
    kv_note = str(params.get("kvcachelen", {}).get("note", "")) + str(
        params.get("kvcachelen", {}).get("meaning", ""))
    if "新 token" not in seq_note or "不含" not in seq_note:
        issues.append("seqlen 必须定义为本次新 token 数（不含此前历史）")
    for token in ("逻辑历史", "空缓存", "不是缓存字节数"):
        if token not in kv_note:
            issues.append(f"kvcachelen 定义缺少语义 {token!r}")
    if "不等于" not in kv_note and "物理保留长度" not in kv_note:
        issues.append("kvcachelen 必须说明不等于物理保留长度")

    # shape 词表：不得残留 T/P（旧符号），且必须真的引用 seqlen/kvcachelen
    legacy_axes: set[str] = set()
    used_axes: set[str] = set()
    for tensor in block.get("tensors", []):
        for axis in tensor.get("shape", []):
            text = str(axis)
            used_axes.update(text.replace("(", " ").replace(")", " ").replace("+", " ")
                             .replace("-", " ").replace("×", " ").replace("÷", " ")
                             .replace(",", " ").split())
    for axis in used_axes:
        if axis in ("T", "P"):
            legacy_axes.add(axis)
    if legacy_axes:
        issues.append(f"shape 轴仍使用旧符号 {sorted(legacy_axes)}（应统一为 seqlen/kvcachelen）")
    if not used_axes & {"seqlen", "kvcachelen"}:
        issues.append("shape 未引用 seqlen/kvcachelen（参数化结构缺失）")

    derivations = block.get("derivations", [])
    if not derivations:
        issues.append("缺少场景派生量 derivations（窗口/缓存规模无从推导）")
    for item in derivations:
        try:
            check_expression(str(item.get("expression", "")))
        except ExpressionError as exc:
            issues.append(f"派生量 {item.get('id')} 表达式非法: {exc}")
        if not item.get("label"):
            issues.append(f"派生量 {item.get('id')} 缺少 label")

    # 压缩条目的因果边界：官方源码屏蔽 w >= (t+1)//M，因此可见条件是严格小于；
    # 说明里写成 w ≤ 会在「条目恰在阈值」时把未来 token 当成可见，必须拒绝。
    if scope.get("attention_type") != "sliding_attention":
        thr_ids = {"future_threshold", "first_query_threshold"}
        by_id = {d.get("id"): d for d in derivations}
        missing = sorted(thr_ids - set(by_id))
        if missing:
            issues.append(f"缺少因果阈值派生量 {missing}（压缩条目可见性边界）")
        blob = " ".join(str(by_id.get(i, {}).get("note", "")) for i in sorted(thr_ids))
        if "w < (t+1)//M" not in blob:
            issues.append("因果阈值说明必须写严格可见条件 w < (t+1)//M"
                          "（源码屏蔽 w >= threshold）")
        if "等于阈值" not in blob:
            issues.append("因果阈值说明必须写明「w 等于阈值即屏蔽」（含等于）")
        if "w ≤" in blob or "w≤" in blob:
            issues.append("因果阈值说明不得把可见条件写成 w ≤ (t+1)//M")

    scenarios = scope.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        issues.append("scope.scenarios 缺失（至少一个场景预设）")
        return issues
    ids = [s.get("id") for s in scenarios]
    if len(ids) != len(set(ids)):
        issues.append("场景 id 重复")
    if scope.get("default_scenario") not in ids:
        issues.append(f"默认场景 {scope.get('default_scenario')!r} 不在场景列表中")
    for scene in scenarios:
        if not scene.get("label"):
            issues.append(f"场景 {scene.get('id')} 缺少 label")
        for key in ("seqlen", "kvcachelen"):
            if key not in scene:
                issues.append(f"场景 {scene.get('id')} 缺少 {key}")
            elif scene[key] is not None and (
                    isinstance(scene[key], bool) or not isinstance(scene[key], int)):
                issues.append(f"场景 {scene.get('id')} 的 {key} 必须是整数或 None")

    evaluated = document.get("scenario_derivations", {})
    if set(evaluated) != set(ids):
        issues.append("scenario_derivations 必须与场景列表一一对应")
    for scene in scenarios:
        rows = {r.get("id"): r for r in evaluated.get(scene.get("id"), [])}
        if set(rows) != {d.get("id") for d in derivations}:
            issues.append(f"场景 {scene.get('id')} 派生结果与 derivations 不一一对应")
            continue
        concrete = scene.get("seqlen") is not None and scene.get("kvcachelen") is not None
        if concrete:
            unknown = [rid for rid, row in rows.items() if row.get("state") != "determined"]
            if unknown:
                issues.append(
                    f"场景 {scene.get('id')} 参数已给定却仍有未知派生量 {unknown}")
        else:
            symbolic = [rid for rid, row in rows.items() if row.get("state") != "determined"]
            if not symbolic:
                issues.append(f"场景 {scene.get('id')} 保留符号却没有任何未知派生量（不诚实）")
            for rid in symbolic:
                if not rows[rid].get("unknown_reason"):
                    issues.append(f"场景 {scene.get('id')} 未知派生量 {rid} 缺少原因")
    return issues
