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

from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES

__all__ = ["validate_hf_reading_document"]

# 数据流检查的 producer 豁免类型：boundary=诚实外部输入边界；weight=权重声明；
# cache=缓存调用前状态（调用后状态由写算子产出，同样豁免缺 producer）。
_FLOW_EXEMPT_KINDS = frozenset({"boundary", "weight", "cache"})


def validate_hf_reading_document(document: dict, input_shas: dict[str, str]) -> list[str]:
    """返回问题串列表；空列表 = 通过。"""
    issues: list[str] = []
    issues.extend(_check_skeleton(document))
    issues.extend(_check_flow(document))
    issues.extend(_check_semantics(document))
    issues.extend(_check_inputs(input_shas))
    return issues


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
        pid = phase.get("id", "?")
        pids = [p.get("id") for p in phase.get("parameters", [])]
        if len(pids) != len(set(pids)):
            issues.append(f"{pid}: 参数 id 重复")
        tids = [t.get("id") for t in phase.get("tensors", [])]
        if len(tids) != len(set(tids)):
            issues.append(f"{pid}: 张量 id 重复")
        oids = [o.get("id") for o in phase.get("operators", [])]
        if len(oids) != len(set(oids)):
            issues.append(f"{pid}: 算子 id 重复")
        param_index = set(pids)
        tensor_index = set(tids)
        for t in phase.get("tensors", []):
            for ref in t.get("parameter_refs", []):
                if ref not in param_index:
                    issues.append(f"{pid}: 张量 {t.get('id')} 参数引用 {ref!r} 无法解析")
        for o in phase.get("operators", []):
            for kind, refs in (
                ("inputs", o.get("inputs", [])),
                ("outputs", o.get("outputs", [])),
                ("weights", o.get("weights", [])),
            ):
                for ref in refs:
                    if ref not in tensor_index:
                        issues.append(f"{pid}: 算子 {o.get('id')} {kind} 引用 {ref!r} 无法解析")
            for ref in o.get("parameter_refs", []):
                if ref not in param_index:
                    issues.append(f"{pid}: 算子 {o.get('id')} 参数引用 {ref!r} 无法解析")
            if o.get("analysis_status") not in ("analyzed", "internal_unanalyzed"):
                issues.append(f"{pid}: 算子 {o.get('id')} 分析状态非法")
        if not phase.get("cache_notes"):
            issues.append(f"{pid}: cache_notes 为空（需说明缓存前提/读写/持久区别）")
        if not phase.get("coverage_notes"):
            issues.append(f"{pid}: coverage_notes 为空（需说明覆盖与未覆盖）")
    return issues


def _check_flow(document: dict) -> list[str]:
    """数据流闭合（机器核查）：producer 存在且在消费之前；无输入输出同 id 自环。

    与源码核对原则一致：不能以说明文字替代真实连接；真实 cache 原地更新以
    调用前/调用后两个张量表达，不落入自环。
    """
    issues = []
    for phase in document.get("phases", []):
        pid = phase.get("id", "?")
        tensors = {t.get("id"): t for t in phase.get("tensors", [])}
        producers: dict[str, list[int]] = {}
        for i, op in enumerate(phase.get("operators", [])):
            for tid in op.get("outputs", []):
                producers.setdefault(tid, []).append(i)
        for i, op in enumerate(phase.get("operators", [])):
            oid = op.get("id", "?")
            for tid in op.get("inputs", []):
                info = tensors.get(tid)
                if info is None:
                    continue  # 引用闭合由骨架检查负责
                if info.get("kind") in _FLOW_EXEMPT_KINDS:
                    continue
                sites = producers.get(tid, [])
                if not sites:
                    issues.append(f"{pid}: 算子 {oid} 的输入 {tid!r} 无 producer（数据流断开）")
                elif all(j >= i for j in sites):
                    issues.append(
                        f"{pid}: 算子 {oid} 的输入 {tid!r} 的 producer "
                        f"{[op2.get('id') for j, op2 in enumerate(phase['operators']) if j in sites]}"
                        f" 不在其之前（阅读顺序断开）"
                    )
            overlap = sorted(set(op.get("inputs", [])) & set(op.get("outputs", [])))
            if overlap:
                issues.append(
                    f"{pid}: 算子 {oid} 输入输出同 id {overlap}（原地更新须以"
                    "调用前/调用后张量分离表达）"
                )
    return issues


def _check_inputs(input_shas: dict[str, str]) -> list[str]:
    """输入身份：SOURCE_FILES 各键必须提供实际 SHA（build 层另行复算指纹）。"""
    issues = []
    if not isinstance(input_shas, dict) or not input_shas:
        issues.append("input_shas 缺失（须提供 SOURCE_FILES 各输入的实际 SHA256）")
        return issues
    for key in SOURCE_FILES:
        value = input_shas.get(key)
        if not isinstance(value, str) or len(value) != 64:
            issues.append(f"input_shas[{key!r}] 缺失或非 SHA256（来源身份不完整）")
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
        if "c_rope" in ops:
            blob = str(ops["c_rope"].get("note", "")) + str(ops["c_rope"].get("op", ""))
            if "first_window_position" not in blob and "i·M" not in blob and "i*M" not in blob:
                issues.append(f"{pid}: c_rope 未表达确定性压缩位置（i·M+first_window_position）")
        else:
            issues.append(f"{pid}: 缺少压缩条目 RoPE 算子 c_rope")

        # 6) Indexer 哨兵与 top-k
        if "i_topk" in ops:
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

        # 9) sink 参数与共享 KV 投影（单 KV 头）
        if "sinks" not in tensors:
            issues.append(f"{pid}: 缺少 sinks 权重张量（每头注意力 sink）")
        if "kv_proj" in ops:
            blob = str(ops["kv_proj"].get("note", ""))
            if "num_key_value_heads=1" not in blob and "单 KV 头" not in blob:
                issues.append(f"{pid}: kv_proj 未说明单 KV 头投影")
    return issues
