"""build 流水线（工作包 C）：inspect→classify→extract→validate→render→export。

复用包 B 的真实 API（build_reference_graph / build_layer_plan /
build_scenario_views / validate_model_contracts）与 S0 的内容寻址/原子提交
模式（.work 暂存 → os.replace 原子提交 → 同输入幂等复用 → 冲突拒绝 →
_failed 诊断）。

成功目录 outputs/<model_key>/<analysis_id>/，analysis_id 对输入扫描（各输入
SHA256）、适配器/规则/渲染/导出工具源码、场景集内容寻址；不含绝对路径/
时间戳/解释器安装路径。同输入重跑复用并核对产物完整性；不同输入/工具版本
得到不同目录（隔离）；失败写 outputs/_failed/<run_id>/ 并返回非零，不落
正式页、不覆盖成功结果。

场景实例化（门禁前必补，依据 model.py:334-365）：
- prefill remainder 主块/尾块分支与具体状态槽位（T%r 尾块写 kv_state/
  score_state 的 offset 槽位）；
- decode 连续调用的同一 state_id 版本链（v1→v2→…）；
- 未触发更新/触发池化写入/缓存容量/历史前提逐场景验证。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from structure_reader.checks.deepseek_v4_contracts import validate_model_contracts
from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_reference import (
    TEMPLATE_ID,
    build_reference_graph,
    build_scenario_views,
)
from structure_reader.layer_plan import build_layer_plan
from structure_reader.rules.deepseek_v4_flash_0731 import (
    GUARD_DEFS,
    SCENARIOS,
    ModelValues,
    scenario_base_env,
)
from structure_reader.scenario_evaluator import evaluate_tri

__all__ = [
    "BuildError",
    "BuildResult",
    "run_build",
]


class BuildError(ValueError):
    """build 失败（具名）：输入/提取/校验/落盘任何一步不通过。"""


# ---------------------------------------------------------------------------
# 场景实例化：压缩路径细化（model.py:331-383 真实控制流）
# ---------------------------------------------------------------------------

def _compress_instances(env: dict[str, int], values: ModelValues) -> dict:
    """一个场景的压缩路径实例化：主块/尾块/槽位/池化来源/写入槽位。

    prefill（start_pos==0，model.py:331-348）：
    - should_compress = T>=r；remainder = T%r；cutoff = T-remainder
    - 主序列块 N_blk = T//r：unflatten 成 (N_blk, r) 块 + APE，overlap 变换
      （前半取前一块 overlap 半，model.py:313-320），softmax 加权汇聚
    - remainder>0：尾块 [cutoff:] 写 kv_state/score_state 的 offset 槽位
      （overlap 时 offset=r，model.py:335,340-341）——历史起点前提显式
    decode（start_pos>0，model.py:350-365）：
    - should_compress = (start_pos+1)%r==0；每 token 写状态槽位
      start_pos%r（非 overlap）或 r+start_pos%r（overlap，model.py:353-354）
    - 触发时从状态读 2r 槽池化出 1 条压缩条目，写压缩区槽位 start_pos//r
      （model.py:382）
    未触发：状态更新照常，无 cache 写/返回值（366-367）。
    """
    past = env["P"]
    tokens = env["T"]
    ratio = env["r"]
    start_pos = past
    is_prefill = start_pos == 0
    end_pos = start_pos + tokens

    if is_prefill:
        remainder = tokens % ratio
        cutoff = tokens - remainder
        n_blk = tokens // ratio
        should_compress = tokens >= ratio
        # overlap=True（CSA ratio==4，model.py:296）：状态槽位 offset=ratio
        # 与 should_compress 无关（model.py:335）——尾块总是写 r..r+rem-1。
        # 主序列末块取 kv_state[:r]（overlap 前半来源，model.py:337-338），
        # 尾块写后 r 槽（offset=4 起，model.py:340-341）。
        state_slots = {
            "layout_note": (
                "overlap 布局：state[:, :r]=overlap 窗（主序列末块），"
                "state[:, r:]=当前窗（尾块）；model.py:308-309,337-341"
            ),
            "main_last_block_slots": list(range(0, ratio))
            if should_compress and cutoff >= ratio else [],
            "main_last_block_note": (
                "主序列末块 → 前半 r 槽（model.py:337-338）"
                if should_compress and cutoff >= ratio else "无主块写入"
            ),
        }
        if remainder > 0:
            state_slots["remainder_write"] = {
                "tensor": "kv_state/score_state",
                "slots": [ratio + i for i in range(remainder)],
                "note": (
                    f"尾块写槽 {ratio}..{ratio + remainder - 1}"
                    "（offset=ratio，model.py:335,340-341），供后续 decode 池化"
                ),
            }
        else:
            state_slots["remainder_write"] = None
        write_cache = bool(should_compress)
        cache_slots = (
            list(range(n_blk)) if should_compress else []
        )  # prefill 写压缩区 0..T//r-1（model.py:380）
        return {
            "phase": "prefill",
            "should_compress": should_compress,
            "remainder": remainder,
            "cutoff": cutoff,
            "n_blk": n_blk,
            "state_slots": state_slots,
            "pool_source": "blockify→overlap→softmax 池化（prefill 主序列块）"
            if should_compress else None,
            "cache_write_slots": cache_slots,
            "cache_write_capacity_note": (
                f"压缩区容量 N_cc={values.max_seq_len // ratio} 条；"
                f"本调用写 {len(cache_slots)} 条"
            ),
        }

    # decode：逐 token
    should_compress = (start_pos + 1) % ratio == 0
    state_slot = ratio + start_pos % ratio  # overlap：r+start_pos%r（model.py:353）
    state_slots = {
        "current_token_write": {
            "tensor": "kv_state/score_state",
            "slots": [state_slot],
            "note": f"当前 token 写槽位 r+start_pos%r={state_slot}（model.py:353-354）",
        }
    }
    if should_compress:
        # 触发后状态前移：state[:, :r] = state[:, r:]（model.py:359-360）
        state_slots["shift_after_pool"] = {
            "note": "池化后前移：state[:r] = state[r:]（model.py:359-360）",
            "slots_before_shift": list(range(0, 2 * ratio)),
        }
    return {
        "phase": "decode",
        "should_compress": should_compress,
        "remainder": 0,
        "n_blk": 0,
        "state_slots": state_slots,
        "pool_source": (
            "kv_state/score_state 历史 2r 槽 + 当前槽（model.py:356-358）"
            if should_compress else None
        ),
        "cache_write_slots": ([start_pos // ratio] if should_compress else []),
        "cache_write_capacity_note": (
            f"压缩区容量 N_cc={values.max_seq_len // ratio} 条；"
            f"本调用{'写压缩区槽位 ' + str(start_pos // ratio) if should_compress else '不写压缩区'}"
        ),
    }


def _instantiate_scenarios(values: ModelValues) -> list[dict]:
    """全部场景的实例化数据：guard 激活 + 压缩路径细化 + Indexer 边界。"""
    from structure_reader.extract.deepseek_v4_reference import indexer_visibility
    instances = []
    for scenario in SCENARIOS:
        env = scenario_base_env(values, scenario)
        env["r"] = 4
        activations = {}
        for guard in GUARD_DEFS:
            result = evaluate_tri(guard.predicate_expr, env)
            activations[guard.stable_id] = (
                {"state": "true" if result.value else "false"}
                if result.is_known
                else {"state": "unknown", "reason": result.unknown_reason}
            )
        instances.append({
            "scenario_id": scenario.scenario_id,
            "description": scenario.description,
            "env": dict(env),
            "covers": list(scenario.covers),
            "guard_activations": activations,
            "compression": _compress_instances(env, values),
            "indexer": indexer_visibility(env),
            "derived": {
                "N_tokens": env["B"] * env["T"],
                "N_ctx": env["T"] + env["T"] // env["r"],
                "N_blk": env["T"] // env["r"],
                "N_vis": (env["P"] + env["T"]) // env["r"],
                "N_sel_c": min(env["Ki"], (env["P"] + env["T"]) // env["r"]),
                "remainder": env["T"] % env["r"],
                "decode_trigger": 1 if env["start_pos"] > 0 and (
                    env["start_pos"] + 1) % env["r"] == 0 else 0,
            },
            "history_premise": (
                "历史连续调用前提：P 个历史 token 已按同一窗口/压缩规则处理；"
                "cache 实际内容 unknown（本图不验证数值），仅结构路径已核对"
            ),
        })
    return instances


def _decode_version_chain(values: ModelValues, length: int = 4) -> list[dict]:
    """decode 连续调用的同一 state_id 版本链（门禁前必补）。

    以 compress_kv_state 为例：P=1 起连续 decode，每步把当前 token 写
    r+start_pos%r 槽位（model.py:353），恰逢 (start_pos+1)%r==0 时池化出
    1 条压缩条目并前移状态槽（model.py:359-360）。同一 state_id
    （csa_topk:compress_kv_state）的版本序列：
    v_hist（上一步 final）→ v_updated（本步写入后、待池化）→
    v_final（触发=shift 后；非触发=updated 本身）。
    每步给 before/updated/final 的版本 ID 引用与跨步关系
    （step i+1 的 before 引用 step i 的 final）。
    """
    chain = []
    ratio = 4
    state_id = "csa_topk:compress_kv_state"
    prev_final_ref = f"state:{state_id}:v_hist#P0"  # 首步历史（调用前前提）
    for step in range(length):
        start_pos = 1 + step
        should_compress = (start_pos + 1) % ratio == 0
        before_ref = prev_final_ref
        updated_ref = f"state:{state_id}:v_updated#P{start_pos}"
        # 未触发：final = updated（状态只累积）；触发：final = shifted
        final_state = "shifted" if should_compress else "updated"
        final_ref = (
            f"state:{state_id}:v_final#P{start_pos}" if should_compress
            else updated_ref
        )
        chain.append({
            "step": step + 1,
            "start_pos": start_pos,
            "state_id": state_id,
            "before_ref": before_ref,
            "updated_ref": updated_ref,
            "final_ref": final_ref,
            "final_state": final_state,
            "write_slot": ratio + start_pos % ratio,
            "should_compress": should_compress,
            "pool_from_state": should_compress,
            "cache_write_slot": start_pos // ratio if should_compress else None,
            "evidence": "model.py:350-360,382",
        })
        prev_final_ref = final_ref
    return chain


# ---------------------------------------------------------------------------
# build 编排
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildResult:
    """一次成功 build 的产物位置与摘要。"""

    output_dir: Path
    analysis_id: str
    reused: bool
    model_key: str
    scenario_set: str


def _tool_sources_hash(project_root: Path) -> dict[str, str]:
    """影响图/页面/表格内容的工具源码哈希（进入 analysis_id）。"""
    relatives = (
        "src/structure_reader/extract/deepseek_v4_reference.py",
        "src/structure_reader/extract/deepseek_v4_variants.py",
        "src/structure_reader/extract/evidence.py",
        # B4：DSpark 六组件与整体集成（影响 data/dspark/*、页面、validation）
        "src/structure_reader/extract/deepseek_v4_dspark.py",
        "src/structure_reader/extract/dspark_stage.py",
        "src/structure_reader/extract/dspark_endpoints.py",
        "src/structure_reader/extract/dspark_exit.py",
        "src/structure_reader/extract/dspark_markov.py",
        "src/structure_reader/extract/dspark_confidence.py",
        "src/structure_reader/extract/dspark_integration.py",
        "src/structure_reader/checks/dspark_contracts.py",
        "src/structure_reader/layer_plan.py",
        "src/structure_reader/checks/deepseek_v4_contracts.py",
        "src/structure_reader/checks/graph_contracts.py",
        "src/structure_reader/rules/deepseek_v4_flash_0731.py",
        "src/structure_reader/scenario_evaluator.py",
        "src/structure_reader/graph_document.py",
        "src/structure_reader/graph_ir.py",
        "src/structure_reader/build_pipeline.py",
        "src/structure_reader/render/structure_page.py",
        "src/structure_reader/export/semantic_tables.py",
        # CLI/入口/注册/扫描：manifest 的 tools 块与其一致，否则同输入
        # 重跑会因 run_manifest 内容变化而伪冲突（identity 必须覆盖全部
        # 影响产物的源码，含清单自身内容）
        "src/structure_reader/cli.py",
        "src/structure_reader/inventory.py",
        "src/structure_reader/model_registry.py",
        "src/structure_reader/__init__.py",
        "scripts/read_model_structure.py",
    )
    hashes = {}
    for rel in relatives:
        path = project_root / rel
        hashes[rel] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file()
            else "MISSING"
        )
    return hashes


def _scenario_set_content(scenario_set: str) -> dict:
    if scenario_set != "standard":
        raise BuildError(f"未知场景集: {scenario_set!r}（当前仅 standard）")
    return {"name": scenario_set, "scenarios": [s.scenario_id for s in SCENARIOS]}


def run_build(model_key: str, workspace_root: Path, scenario_set: str = "standard",
              project_root: Path | None = None) -> BuildResult:
    """一条命令连通 inspect 基线核对 → classify → extract → validate。

    render/export 产物由调用方（CLI）在同一 analysis 目录写盘前传入本函数
    返回的 bundle；本函数负责图/规划/场景/校验与 analysis_id。
    """
    workspace_root = Path(workspace_root)
    project_root = project_root or Path(__file__).resolve().parents[2]

    # 1) 输入扫描 + 固定基线核对（inspect 同源；失配即失败）
    from structure_reader.inventory import build_inventory
    from structure_reader.model_registry import (
        StructureReaderError,
        find_workspace_root,
    )
    try:
        inv = build_inventory(model_key, workspace_root=workspace_root)
    except StructureReaderError as exc:
        raise BuildError(f"输入扫描失败: {exc}") from exc
    bad = [item for item in inv["baseline_comparison"]
           if item["status"] in ("mismatch", "schema_unrecognized")]
    if bad:
        raise BuildError(
            "输入基线失配: "
            + "; ".join(f"{item['name']}={item['status']}" for item in bad)
            + "；输入已变化，图不得继续构建"
        )

    # 2) classify：四组层规划（含真实提取 build_graph=True）
    values = None
    plan = build_layer_plan(workspace_root, build_graph=True)
    values = load_values_via_rules(workspace_root)

    # 3) extract：代表图与证据（build_layer_plan 内部已构建；此处取实例）
    graph, registry, values2 = build_reference_graph(workspace_root)
    # 4) validate：通用引用 + 模型契约
    ref_issues = validate_graph_references(graph)
    if ref_issues:
        raise BuildError(f"通用引用检查未通过: {ref_issues[:5]}")
    contract_issues = validate_model_contracts(graph)
    if contract_issues:
        raise BuildError(f"模型契约未通过: {contract_issues[:5]}")

    # 5) 场景实例化
    scenarios = _instantiate_scenarios(values)
    chain = _decode_version_chain(values)

    # 6) analysis_id：输入 SHA + 工具源码 + 场景集
    input_shas = {
        item["role"]: item["sha256"] for item in inv["inputs"]
    }
    identity = {
        "schema": "structure_reader_build_identity_v1",
        "model_key": model_key,
        "scenario_set": _scenario_set_content(scenario_set),
        "input_sha256": input_shas,
        "tool_sources": _tool_sources_hash(project_root),
    }
    analysis_id = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return BuildResult(
        output_dir=project_root / "outputs" / model_key / analysis_id,
        analysis_id=analysis_id,
        reused=False,  # CLI 层再判复用
        model_key=model_key,
        scenario_set=scenario_set,
    )


def load_values_via_rules(workspace_root: Path) -> ModelValues:
    from structure_reader.rules.deepseek_v4_flash_0731 import load_model_values
    return load_model_values(workspace_root)
