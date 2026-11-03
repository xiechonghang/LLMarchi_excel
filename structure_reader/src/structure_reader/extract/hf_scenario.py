"""分析场景与同源投影：``seqlen``/``kvcachelen`` 场景预设、派生量求值、兼容阶段投影。

本模块是 T03「一份参数化结构 + 场景参数」的数据侧实现，只做纯计算，不读源码、
不建图、不做文件 I/O：

- **一份结构**：``extract/deepseek_v4_hf.py``（经 ``m2_hca.py``/``m2_swa_hash.py``）
  为每个真实层组构建**一份** Block 结构（参数/张量/算子/条件），shape 保留符号
  ``seqlen``（本次新 token 数）与 ``kvcachelen``（调用前逻辑历史 token 数）。
  Prefill/Decode 不再各建一份结构。
- **场景**：``SCENARIO_PRESETS`` 只绑定 ``seqlen``/``kvcachelen`` 的取值；结构本身
  不随之改变。派生量（窗口数、压缩条目、可读 KV 长度等）由
  ``scenario_evaluator`` 的有限白名单求值器按固定配置与「从空态连续正常执行」
  前提推导；变量缺失或除零时返回 unknown，**不默认成 0/False**。
- **兼容投影**：``project_legacy_phase`` 把同一份结构投影成既有
  ``hf_decoder_reading.v1`` 的 prefill/decode 阶段（Excel 两表、单层阅读页、
  可选 ONNX 仍消费该冻结接口），阶段差异只是场景取值，不是第二套结构事实。

``kvcachelen`` 是逻辑历史 token 总数，不是物理保留长度（主 KV 只持久
``min(kvcachelen, sliding_window-1)`` 个位置）也不是缓存字节数。
"""

from __future__ import annotations

import copy

from structure_reader.scenario_evaluator import TriState, evaluate_tri

__all__ = [
    "STRUCTURE_SCHEMA",
    "SCENARIO_PRESETS",
    "DEFAULT_SCENARIO_ID",
    "LEGACY_PHASE_VALUES",
    "scenario_by_id",
    "evaluate_derivations",
    "scenario_instance_line",
    "project_legacy_phase",
    "structure_document",
]

STRUCTURE_SCHEMA = "hf_layer_structure.v1"

# 少量有区别的场景（不是长度组合枚举）：
# - 空缓存 Prefill（kvcachelen=0）：示例取值与符号取值各一，后者验证 unknown 诚实；
# - 非零逻辑历史单 token Decode：窗口未闭合 / 闭窗对齐各一；
# - 非零历史多 token Decode：验证统一结构没有偷留 +1/T=1 假设。
SCENARIO_PRESETS = (
    {
        "id": "prefill-empty",
        "label": "空缓存 Prefill（示例 seqlen=8）",
        "seqlen": 8,
        "kvcachelen": 0,
        "note": "kvcachelen=0 即空缓存起步；seqlen=8 只是示例取值，结构 shape 仍保留符号 seqlen。",
    },
    {
        "id": "prefill-symbolic",
        "label": "空缓存 Prefill（符号 seqlen，未给数值）",
        "seqlen": None,
        "kvcachelen": 0,
        "note": "seqlen 保持符号：与 seqlen 有关的派生量如实标为未知，不默认成 0。",
    },
    {
        "id": "decode-nonzero",
        "label": "非零历史单 token Decode（kvcachelen=4096，窗口未闭合）",
        "seqlen": 1,
        "kvcachelen": 4096,
        "note": "kvcachelen=4096 是示例历史长度；与 M 不整除对齐时本调用不闭窗，只写回 buffer。",
    },
    {
        "id": "decode-closed",
        "label": "非零历史单 token Decode（kvcachelen=4095，闭窗对齐）",
        "seqlen": 1,
        "kvcachelen": 4095,
        "note": "kvcachelen+seqlen 恰为 M 的整数倍：本调用闭窗（CSA M=4、HCA M=128 皆对齐）。",
    },
    {
        "id": "decode-multi",
        "label": "非零历史多 token Decode（seqlen=4，kvcachelen=4092）",
        "seqlen": 4,
        "kvcachelen": 4092,
        "note": "多 token Decode 场景：检查统一结构按 seqlen 表达，而不是固定 +1/T=1。",
    },
)

DEFAULT_SCENARIO_ID = "prefill-empty"

# 兼容投影的既有两阶段取值：prefill 空缓存起步（kvcachelen=0），decode 单 token
# 预设（seqlen=1）。两者都只是场景预设，不改变符号 seqlen/kvcachelen 的结构事实。
LEGACY_PHASE_VALUES = {
    "prefill": {
        "scenario_id": "legacy-prefill",
        "label": "Prefill 场景投影",
        "seqlen": None,
        "kvcachelen": 0,
        "note": "空缓存 Prefill 预设：kvcachelen=0；seqlen 保留符号（由实际调用决定）。",
    },
    "decode": {
        "scenario_id": "legacy-decode",
        "label": "Decode 场景投影",
        "seqlen": 1,
        "kvcachelen": None,
        "note": "Decode 单 token 预设：seqlen=1；kvcachelen 保留符号（调用前的逻辑历史由调用序列决定）。",
    },
}


def scenario_by_id(scenario_id: str) -> dict:
    for preset in SCENARIO_PRESETS:
        if preset["id"] == scenario_id:
            return dict(preset)
    raise ValueError(f"未知场景 {scenario_id!r}；可用：{[p['id'] for p in SCENARIO_PRESETS]}")


def _base_env(parameters: list[dict]) -> dict[str, int]:
    """参数表中已确定的整数常量（不含序列维度与数据相关量）。"""
    env: dict[str, int] = {}
    for param in parameters:
        value = param.get("value")
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        env[str(param.get("id"))] = value
    return env


def evaluate_derivations(derivations: list[dict], parameters: list[dict],
                         scenario: dict, *,
                         bind_keys: tuple[str, ...] | None = None) -> list[dict]:
    """按场景绑定可填写输入后求值全部派生量（三态，未知不默认 0）。

    默认绑定 ``seqlen``/``kvcachelen``（既有文本模型的场景轴）。非 Decoder 的
    单次网络（HiDream-O1-Image）用 ``bind_keys`` 声明自己的输入规模轴（图像
    高/宽、prompt token 数、参考图 patch 数等），不借用 seqlen/kvcachelen 语义。

    派生量可以链式引用前面已确定的派生 id；前面的量未知时，依赖它的量也如实
    未知（不把 unknown 当 0 继续算）。
    """
    keys = tuple(bind_keys) if bind_keys else ("seqlen", "kvcachelen")
    env = _base_env(parameters)
    for key in keys:
        env.pop(key, None)
    for key in keys:
        value = scenario.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            env[key] = value
    results: list[dict] = []
    for item in derivations:
        expr = str(item.get("expression", ""))
        # 可选一致性守卫：白名单布尔表达式。为 False 表示本场景的输入组合
        # 不满足该派生量的前提（例如每图长度 × 图数 ≠ 总长），此时如实未知，
        # 不把不可信输入代入公式。旧模型没有 guard 键，行为不变。
        guard = item.get("guard")
        guard_state = None
        if guard:
            guard_state = evaluate_tri(str(guard), dict(env))
            if (not guard_state.is_known) or guard_state.value is not True:
                reason = (f"输入不满足约束：{guard}" if guard_state.is_known
                          else f"约束无法求值：{guard_state.unknown_reason}")
                results.append({
                    "id": item.get("id", ""),
                    "label": item.get("label", ""),
                    "expression": expr,
                    "note": item.get("note", ""),
                    "state": "unknown",
                    "unknown_reason": reason,
                })
                continue
        state: TriState = evaluate_tri(expr, dict(env))
        entry = {
            "id": item.get("id", ""),
            "label": item.get("label", ""),
            "expression": expr,
            "note": item.get("note", ""),
            "state": state.state,
        }
        if guard:
            entry["guard"] = str(guard)
        if state.is_known:
            entry["value"] = state.value
            if isinstance(state.value, int) and not isinstance(state.value, bool):
                env[str(item.get("id"))] = state.value
        else:
            entry["unknown_reason"] = state.unknown_reason
        results.append(entry)
    return results


def scenario_instance_line(scenario: dict, evaluated: list[dict]) -> str:
    """把一个场景的派生结果压成一行可核对文字（未知项如实标注原因，不默认 0）。"""
    head = (f"{scenario.get('label', scenario.get('id', ''))}："
            f"seqlen={_fmt_value(scenario.get('seqlen'))}、"
            f"kvcachelen={_fmt_value(scenario.get('kvcachelen'))}")
    if not evaluated:
        return head + "（本层型无压缩窗口派生量）"
    parts = [f"{entry['label']}={entry.get('value')}"
             for entry in evaluated if entry.get("state") == "determined"]
    unknown = [entry for entry in evaluated if entry.get("state") != "determined"]
    if unknown:
        reasons = sorted({entry.get("unknown_reason", "未知") for entry in unknown})
        parts.append(f"其余 {len(unknown)} 项未知（{('；'.join(reasons))}）")
    return head + " → " + "；".join(parts)


def _fmt_value(value) -> str:
    if value is None:
        return "符号（未给定）"
    return str(value)


def project_legacy_phase(structure: dict, phase_id: str) -> dict:
    """把同一份 Block 结构投影成既有 ``hf_decoder_reading.v1`` 阶段。

    只替换场景取值与阶段标题、追加场景实例说明；tensors/operators/连接完全来自
    同一份结构，不复制第二套阶段事实。
    """
    if phase_id not in LEGACY_PHASE_VALUES:
        raise ValueError(f"未知兼容阶段 {phase_id!r}")
    spec = LEGACY_PHASE_VALUES[phase_id]
    phase = copy.deepcopy(structure)
    phase["id"] = phase_id
    base_title = str(structure.get("title", "Block 结构"))
    phase["title"] = f"{spec['label']}（{base_title}）"
    phase["scenario_id"] = spec["scenario_id"]
    phase["scenario_binding"] = {
        "seqlen": spec["seqlen"], "kvcachelen": spec["kvcachelen"],
        "note": spec["note"],
    }
    for param in phase.get("parameters", []):
        pid = param.get("id")
        if pid in ("seqlen", "kvcachelen"):
            param["value"] = spec[pid]
            param["expression"] = (
                f"{pid} = {'0（空缓存 Prefill）' if spec[pid] == 0 else '1（Decode 单 token 预设）'}"
                if spec[pid] is not None else f"{pid} 保留符号（由实际调用/调用序列决定）"
            )
            param["note"] = spec["note"]
    evaluated = evaluate_derivations(
        phase.get("derivations", []), phase.get("parameters", []), spec)
    phase["scenario_derivations"] = evaluated
    phase["cache_notes"] = list(phase.get("cache_notes", [])) + [
        "本阶段场景实例：" + scenario_instance_line(spec, evaluated),
    ]
    return phase


def structure_document(values, block: dict, scenarios=SCENARIO_PRESETS, *,
                       bind_keys: tuple[str, ...] | None = None,
                       default_scenario: str | None = None) -> dict:
    """把一份 Block 结构包装成 ``hf_layer_structure.v1``（场景预设与派生值在此求值）。

    ``bind_keys``/``default_scenario`` 只影响场景绑定轴与默认场景标识；既有文本
    模型不传时保持原行为（绑定 seqlen/kvcachelen，默认第一个场景）。
    """
    evaluated = {
        preset["id"]: evaluate_derivations(
            block.get("derivations", []), block.get("parameters", []), preset,
            bind_keys=bind_keys)
        for preset in scenarios
    }
    return {
        "schema": STRUCTURE_SCHEMA,
        "model": {
            "model_key": values.model_key,
            "display_name": values.display_name,
            "implementation": values.implementation,
            "source_revision": values.source_revision,
            "config_path": values.config_path,
            # 可选身份：只有声明了上游模型仓库 revision 的适配才记录（DeepSeek 未声明，
            # 保持既有 JSON 字节不变；GLM-5.3 记录 zai-org/GLM-5.3 revision）。
            **({"model_revision": values.model_revision}
               if getattr(values, "model_revision", None) else {}),
        },
        "scope": {
            "layer_index": block.get("layer_index"),
            "attention_type": block.get("attention_type"),
            "moe_type": block.get("moe_type"),
            # 可选层型维度：GLM-5.3 用 full/shared indexer 区分同一 attention_type 的
            # 层型；DeepSeek 没有该字段时不写键，保持既有 JSON 字节不变。
            **({"indexer_type": block["indexer_type"]}
               if "indexer_type" in block else {}),
            "analysis_path": "eager",
            "scenarios": [dict(preset) for preset in scenarios],
            "default_scenario": (default_scenario or scenarios[0]["id"]),
            "notes": block.get("scope_notes", []),
        },
        "block": block,
        "scenario_derivations": evaluated,
    }
