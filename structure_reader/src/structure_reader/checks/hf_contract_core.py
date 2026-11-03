"""HF 阅读文档的通用契约检查（模型无关部分）。

只放「任何 HF 阅读文档都必须成立」的骨架/数据流/身份检查；模型语义锚点
（mHC 方向、DSA indexer 复用、MoE 打分方式等）留在各模型自己的 checks 模块，
避免把某个模型的机制当成通用规则。

DeepSeek 与 GLM-5.3 的 checks 都调用这里，不各写一份相同的数据流校验。
"""
from __future__ import annotations

__all__ = [
    "FLOW_EXEMPT_KINDS",
    "check_unit",
    "check_flow_units",
    "check_model_identity",
    "check_input_identity",
]

# 数据流检查的 producer 豁免类型：boundary=诚实外部输入边界；weight=权重声明；
# cache=缓存调用前状态（调用后状态由写算子产出，同样豁免缺 producer）。
FLOW_EXEMPT_KINDS = frozenset({"boundary", "weight", "cache"})


def check_unit(unit: dict, label: str) -> list[str]:
    """一个阶段/路径的骨架检查：id 唯一、引用闭合、分析状态、注记非空。"""
    issues: list[str] = []
    pids = [p.get("id") for p in unit.get("parameters", [])]
    if len(pids) != len(set(pids)):
        issues.append(f"{label}: 参数 id 重复")
    tids = [t.get("id") for t in unit.get("tensors", [])]
    if len(tids) != len(set(tids)):
        issues.append(f"{label}: 张量 id 重复")
    oids = [o.get("id") for o in unit.get("operators", [])]
    if len(oids) != len(set(oids)):
        issues.append(f"{label}: 算子 id 重复")
    param_index = set(pids)
    tensor_index = set(tids)
    for t in unit.get("tensors", []):
        for ref in t.get("parameter_refs", []):
            if ref not in param_index:
                issues.append(f"{label}: 张量 {t.get('id')} 参数引用 {ref!r} 无法解析")
    for o in unit.get("operators", []):
        for kind, refs in (
            ("inputs", o.get("inputs", [])),
            ("outputs", o.get("outputs", [])),
            ("weights", o.get("weights", [])),
        ):
            for ref in refs:
                if ref not in tensor_index:
                    issues.append(f"{label}: 算子 {o.get('id')} {kind} 引用 {ref!r} 无法解析")
        for ref in o.get("parameter_refs", []):
            if ref not in param_index:
                issues.append(f"{label}: 算子 {o.get('id')} 参数引用 {ref!r} 无法解析")
        if o.get("analysis_status") not in ("analyzed", "internal_unanalyzed"):
            issues.append(f"{label}: 算子 {o.get('id')} 分析状态非法")
    if not unit.get("cache_notes"):
        issues.append(f"{label}: cache_notes 为空（需说明缓存前提/读写/持久区别）")
    if not unit.get("coverage_notes"):
        issues.append(f"{label}: coverage_notes 为空（需说明覆盖与未覆盖）")
    return issues


def check_flow_units(units) -> list[str]:
    """数据流闭合：producer 存在且在消费之前；无输入输出同 id 自环。

    不能以说明文字替代真实连接；真实 cache 原地更新以调用前/调用后两个张量
    表达，不落入自环。units 为阶段/路径列表（阅读文档传 phases，整网总览传
    单条 path）。
    """
    issues: list[str] = []
    for phase in units:
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
                if info.get("kind") in FLOW_EXEMPT_KINDS:
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


def check_model_identity(model: dict) -> list[str]:
    """模型身份字段：五个必需字符串 + implementation=transformers。"""
    issues: list[str] = []
    for field in ("model_key", "display_name", "implementation", "source_revision", "config_path"):
        if not isinstance(model.get(field), str) or not model[field]:
            issues.append(f"model.{field} 缺失或非字符串")
    if model.get("implementation") != "transformers":
        issues.append(f"model.implementation 必须是 transformers: {model.get('implementation')!r}")
    return issues


def check_input_identity(input_shas: dict[str, str], source_files: dict[str, str]) -> list[str]:
    """输入身份：SOURCE_FILES 各键必须提供实际 SHA（build 层另行复算指纹）。"""
    issues: list[str] = []
    if not isinstance(input_shas, dict) or not input_shas:
        issues.append("input_shas 缺失（须提供 SOURCE_FILES 各输入的实际 SHA256）")
        return issues
    for key in source_files:
        value = input_shas.get(key)
        if not isinstance(value, str) or len(value) != 64:
            issues.append(f"input_shas[{key!r}] 缺失或非 SHA256（来源身份不完整）")
    return issues
