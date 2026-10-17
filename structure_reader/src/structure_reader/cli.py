"""structure_reader CLI：当前仅公开 S0 ``inspect`` 入口。

设计约束（S0 第二轮开发单 02）：
- 仅标准库；inspect 真正调用 :func:`structure_reader.inventory.build_inventory`。
- 产物唯一为 ``<project>/outputs/_inventory/<model_key>/<content_id>/`` 下的
  ``input_manifest.json`` / ``capability_report.json`` / ``run_manifest.json`` /
  ``README.md``；不生成后续阶段（模型图/HTML/CSV）的空占位。
- ``content_id`` 为完整 SHA256，覆盖规范化扫描输入、注册 JSON、基线、影响
  输出的库/入口源码与 Python 版本；不含绝对工作区位置、时间戳或解释器
  安装路径等偶然字段，因此产物文件对同一输入字节级确定。
- 同输入重跑复用同目录并先核对既有文件一致性；不一致（被篡改/不完整）
  拒绝覆盖，非零退出并在 ``outputs/_failed/<run_id>/`` 留诊断记录。
- 产物一律临时写入 ``<project>/.work/``，成功后原子提交到正式目录。
- 状态：``success``（全部已声明基线一致）/ ``unreviewed``（缺基线，未审查，
  不表示通过）/ ``failed``（不产生正式扫描包）。
- 退出码：0 = success/unreviewed；1 = failed；2 = argparse 参数错误。
- 工程（outputs/.work 根）从本模块源码位置定位，与调用方 cwd 无关；
  ``--workspace-root`` 仅改变扫描读取的注册与输入所在工作区。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .inventory import build_inventory
from .model_registry import (
    StructureReaderError,
    find_workspace_root,
    get_model_config_path,
    get_model_entry,
)

__all__ = ["main"]

_INVENTORY_SCHEMA_TAG = "structure_reader_inventory_identity_v1"
_CONTENT_FILES = ("README.md", "capability_report.json", "input_manifest.json", "run_manifest.json")
_BAD_BASELINE_STATUSES = ("mismatch", "schema_unrecognized")


class OutputConsistencyError(Exception):
    """既存扫描包内容与本输入不一致（疑似篡改或不完整），拒绝覆盖。"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path:
    """``src/structure_reader/cli.py`` → 工程根（``structure_reader/``）。"""
    return Path(__file__).resolve().parents[2]


def _outputs_root() -> Path:
    return _project_root() / "outputs"


def _canonical_json_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pretty_json_bytes(obj: object) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _tool_identity() -> dict:
    """影响输出内容的工具身份：版本、Python 版本与相关源码文件内容哈希。"""
    project = _project_root()
    relative_sources = (
        "src/structure_reader/__init__.py",
        "src/structure_reader/model_registry.py",
        "src/structure_reader/inventory.py",
        "src/structure_reader/cli.py",
        "scripts/read_model_structure.py",
    )
    sources: dict[str, str | None] = {}
    for relative in relative_sources:
        path = project / relative
        sources[relative] = _sha256_file(path) if path.is_file() else None
    return {
        "structure_reader_version": __version__,
        "python_version": sys.version.split()[0],
        "sources": sources,
    }


def _scrub_workspace_paths(value: object, workspace_root: Path) -> object:
    """从字符串中剔除内嵌的绝对工作区路径（git 报错等偶然字段）。"""
    if not isinstance(value, str):
        return value
    root = str(workspace_root)
    if root + "/" in value:
        value = value.replace(root + "/", "")
    if root in value:
        value = value.replace(root, ".")
    return value


_PATH_VALUE_KEYS = {"path", "source", "source_path", "dataclass_source", "run_config_path"}


def _relativize_source_paths(obj: object, workspace_root: Path) -> object:
    """把来源承载字段（path/source/source_path 等）的绝对路径改为工作区相对。

    只转换明确承载路径的字段值；模型 config 原样数据与默认值表达式不动。
    """
    if isinstance(obj, dict):
        return {
            key: (_scrub_workspace_paths(value, workspace_root)
                  if isinstance(key, str) and key in _PATH_VALUE_KEYS
                  else _relativize_source_paths(value, workspace_root))
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_relativize_source_paths(item, workspace_root) for item in obj]
    return obj


def _normalized_hf_checkout(inv: dict, workspace_root: Path) -> dict:
    hf = dict(inv["revisions"]["hf_checkout"])
    hf.pop("checkout_root", None)  # 绝对路径属偶然字段，不进内容寻址产物
    if "error" in hf:
        hf["error"] = _scrub_workspace_paths(hf["error"], workspace_root)
    if "dirty_paths" in hf:
        hf["dirty_paths"] = sorted(hf["dirty_paths"] or [])
    return hf


def _identity_document(model_key: str, inv: dict, workspace_root: Path,
                       registry_path: Path) -> dict:
    """content_id 的规范化输入：同输入必得同字节序列，不含偶然字段。"""
    baselines = []
    for item in inv["baseline_comparison"]:
        if item["status"] == "missing":
            baselines.append(
                {"name": item["name"], "file_sha256": None, "declared_source_files": None}
            )
            continue
        declared = {
            entry["path"]: entry["declared_sha256"]
            for entry in item.get("file_comparison", [])
        }
        baselines.append(
            {
                "name": item["name"],
                "file_sha256": _sha256_file(Path(item["path"])),
                "declared_source_files": declared,
            }
        )
    hf = _normalized_hf_checkout(inv, workspace_root)
    capabilities = inv["capabilities"]
    return {
        "schema": _INVENTORY_SCHEMA_TAG,
        "model_key": model_key,
        "display_model_id": inv["display_model_id"],
        "registry_config": {
            "path": registry_path.resolve().relative_to(workspace_root).as_posix(),
            "sha256": _sha256_file(registry_path),
        },
        "inputs": sorted(
            (
                {
                    "role": item["role"],
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "size_bytes": item["size_bytes"],
                }
                for item in inv["inputs"]
            ),
            key=lambda item: item["role"],
        ),
        "hf_checkout": {
            key: hf.get(key)
            for key in ("commit", "commit_available", "dirty", "dirty_paths", "error")
        },
        "weights_actual": {
            "safetensors_count": inv["weights"]["actual"]["safetensors_count"],
            "shard_pattern_matches": sorted(inv["weights"]["actual"]["shard_pattern_matches"]),
            "index_present": inv["weights"]["actual"]["index_present"],
        },
        "baselines": sorted(baselines, key=lambda item: item["name"]),
        "environment": {
            "python_version": capabilities["interpreter"]["python_version"],
            "python_implementation": capabilities["interpreter"]["implementation"],
            "platform": capabilities["interpreter"]["platform"],
            "optional_packages": {
                name: spec.get("available") if isinstance(spec, dict) else None
                for name, spec in capabilities["optional_packages"].items()
            },
        },
        "tools": _tool_identity(),
    }


# ---------------------------------------------------------------------------
# 产物文档（全部确定性：不含绝对路径/时间戳/解释器安装路径）
# ---------------------------------------------------------------------------


def _input_manifest_document(model_key: str, inv: dict, entry: dict,
                             workspace_root: Path) -> dict:
    inputs = sorted(
        (
            {
                "role": item["role"],
                "path": item["path"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
                "description": item["description"],
            }
            for item in inv["inputs"]
        ),
        key=lambda item: item["role"],
    )
    weights = inv["weights"]
    configs = inv["configs"]
    return {
        "schema": "input_manifest_v2",
        "model_key": model_key,
        "display_model_id": inv["display_model_id"],
        "path_convention": "所有 path 相对工作区根；本文件不含绝对路径与时间戳，保证内容寻址确定性",
        "inputs": inputs,
        "revisions": {
            "model_manifest": {
                "path": inv["revisions"]["model_manifest"]["path"],
                "declared_repo_id": inv["revisions"]["model_manifest"].get("declared_repo_id"),
                "declared_revision": inv["revisions"]["model_manifest"].get("declared_revision"),
            },
            "hf_checkout": _normalized_hf_checkout(inv, workspace_root),
        },
        "configs": {
            "raw_root_config": _relativize_source_paths(configs["raw_root_config"], workspace_root),
            "constructor_state": _relativize_source_paths(configs["constructor_state"], workspace_root),
            "entrypoint_overrides": _relativize_source_paths(configs["entrypoint_overrides"], workspace_root),
            "notes": [
                "构造态 = ModelArgs dataclass 默认值（AST 静态读取）+ inference/config.json 运行配置覆盖",
                "入口覆盖（如 interactive 分支 max_seq_len=65536/max_batch_size=1）只记录不并入构造态，"
                "其来源/条件/符号见 entrypoint_overrides",
                "所有来源路径已规范化为工作区相对路径",
            ],
        },
        "raw_to_effective_pairs": {
            "equivalent_pairs": inv["raw_to_effective_pairs"]["equivalent_pairs"],
            "non_equivalent_pairs": inv["raw_to_effective_pairs"]["non_equivalent_pairs"],
            "notes": [
                "等价对照对的 match/取值由扫描库在扫描时计算；本文件原样持久化其结论",
            ],
        },
        "weights": {
            "report_basis": weights["report_basis"],
            "actual": {
                "safetensors_count": weights["actual"]["safetensors_count"],
                "shard_pattern": weights["actual"]["shard_pattern"],
                "shard_pattern_matches": sorted(weights["actual"]["shard_pattern_matches"]),
                "index_present": weights["actual"]["index_present"],
                "index_file": entry["weights"]["index_file"],
            },
            "manifest_declared": weights["manifest_declared"],
            "consistency": weights["consistency"],
        },
    }


def _capability_document(inv: dict) -> dict:
    capabilities = inv["capabilities"]
    interpreter = capabilities["interpreter"]
    return {
        "schema": "capability_report_v2",
        "interpreter": {
            "python_version": interpreter["python_version"],
            "implementation": interpreter["implementation"],
            "platform": interpreter["platform"],
        },
        "core_dependencies": capabilities["core_dependencies"],
        "optional_packages": capabilities["optional_packages"],
        "implemented_capabilities": capabilities["implemented_capabilities"],
        "not_implemented": capabilities["not_implemented"],
        "environment_note": "解释器可执行文件的绝对安装路径（python_executable）属运行环境偶然信息，只在 CLI stdout 摘要出现，不写入内容寻址产物",
    }


def _readme_document(model_key: str, content_id: str, status: str, inv: dict) -> bytes:
    display = inv["display_model_id"]
    weights = inv["weights"]["actual"]
    consistency = inv["weights"]["consistency"]
    if status == "success":
        review_line = "success —— 全部已声明的源码基线比对一致（reviewed）"
    else:
        review_line = (
            "unreviewed —— 源码基线缺失，本扫描未经基线审查；unreviewed 不表示通过"
        )
    lines = [
        f"# {display} — S0 输入扫描产物（input inventory）",
        "",
        f"- 内容 ID: `{content_id}`（完整 SHA256；同输入重跑得到同一目录）",
        f"- 模型键: `{model_key}`",
        "- 阶段: S0 输入扫描 —— **本目录不是模型结构图**",
        f"- 审查状态: {review_line}",
        "",
        "## 阅读顺序",
        "",
        "1. `run_manifest.json` — 状态、完整内容 ID 及其构成（content_id_basis）、文件清单、工具版本",
        "2. `input_manifest.json` — 输入扫描：各输入文件的角色/工作区相对路径/SHA256；"
        "revision 声明与 HF checkout 状态；实际权重现状",
        "3. `capability_report.json` — 当前实现能力与未实现项",
        "",
        "## 配置消费说明",
        "",
        "`input_manifest.json` 直接持久化配置消费数据：raw 根配置（HF 命名原样）、"
        "构造态（ModelArgs dataclass 默认值 + run_config 覆盖，逐字段含来源与行号）、"
        "入口覆盖（interactive 分支 max_seq_len=65536/max_batch_size=1 等，只记录不并入构造态）、"
        "以及 raw↔effective 对照对结论（4096/4 等构造值 vs interactive 覆盖值的来源均可在"
        "对应节查到）。来源路径已规范化为工作区相对路径。",
        "",
        "## 实际权重状态",
        "",
        f"磁盘现存 {weights['safetensors_count']} 个 safetensors"
        f"（匹配分片模式 {len(weights['shard_pattern_matches'])} 个）；"
        f"索引文件{'存在' if weights['index_present'] else '缺失'}；"
        f"与下载清单声明一致性: **{consistency['status']}**。"
        "以磁盘现存为准，下载清单不冒充现存权重。",
        "",
        "## 尚无的内容",
        "",
        "- 尚无模型图 / HTML 阅读页 / CSV 表格（分属 S1+/S5 等后续阶段产物）",
        "- 本目录没有为后续阶段预留的空占位目录",
        "",
        "## 一致性与复现",
        "",
        "同输入重跑会得到相同内容 ID 并复用本目录（先核对既有文件一致性）；"
        "若既存内容与本输入不符（被篡改或不完整），命令拒绝覆盖、非零退出，"
        "并在 `outputs/_failed/` 留具名诊断记录。",
        "",
        "路径约定：产物内所有路径相对工作区根，不含绝对路径、时间戳与解释器安装路径。",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _run_manifest_document(status: str, content_id: str, identity: dict,
                           doc_hashes: dict[str, str]) -> dict:
    files = [
        {"name": name, "sha256": doc_hashes[name]}
        for name in _CONTENT_FILES
        if name != "run_manifest.json"
    ]
    files.append({"name": "run_manifest.json", "sha256": None, "note": "本文件"})
    return {
        "schema": "run_manifest_v1",
        "artifact": {
            "kind": "input_inventory",
            "stage": "S0",
            "note": "S0 输入扫描产物；不是模型结构图",
        },
        "status": status,
        "reviewed": status == "success",
        "model_key": identity["model_key"],
        "display_model_id": identity["display_model_id"],
        "content_id": content_id,
        "content_id_schema": _INVENTORY_SCHEMA_TAG,
        "content_id_basis": identity,
        "files": files,
        "tools": identity["tools"],
    }


# ---------------------------------------------------------------------------
# 落盘：.work 暂存 → 原子提交；既存目录先核对一致性
# ---------------------------------------------------------------------------


def _materialize(target: Path, documents: dict[str, bytes]) -> str:
    """提交产物包（键可含子目录路径）；返回 "created" 或 "reused"。
    内容冲突抛 :class:`OutputConsistencyError`。"""
    if target.exists():
        return _verify_existing(target, documents)
    work_root = _project_root() / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="build-", dir=work_root))
    try:
        for name, data in documents.items():
            dest = work_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(work_dir, target)
        except OSError:
            if target.exists():
                return _verify_existing(target, documents)
            raise
        return "created"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _verify_existing(target: Path, documents: dict[str, bytes]) -> str:
    conflicts = []
    for name, data in documents.items():
        path = target / name
        if not path.is_file():
            conflicts.append({"file": name, "problem": "missing"})
        else:
            actual = _sha256_bytes(path.read_bytes())
            if actual != _sha256_bytes(data):
                conflicts.append({"file": name, "expected_sha256": _sha256_bytes(data), "actual_sha256": actual})
    if conflicts:
        raise OutputConsistencyError(
            f"既存产物包与本输入不一致，拒绝覆盖: {target}; 冲突: "
            + "; ".join(str(item) for item in conflicts)
        )
    return "reused"


# ---------------------------------------------------------------------------
# 失败记录：outputs/_failed/<run_id>/
# ---------------------------------------------------------------------------


def _write_failed_record(model_key: str, kind: str, detail: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{stamp}Z-{uuid.uuid4().hex[:8]}"
    record_dir = _outputs_root() / "_failed" / run_id
    record_dir.mkdir(parents=True)
    manifest = {
        "schema": "failed_run_v1",
        "status": "failed",
        "stage": "S0",
        "run_id": run_id,
        "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_key": model_key,
        "error": {"kind": kind, **detail},
        "tools": _tool_identity(),
    }
    (record_dir / "run_manifest.json").write_bytes(_pretty_json_bytes(manifest))
    return record_dir


def _emit(args: argparse.Namespace, text_lines: list[str], payload: dict) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for line in text_lines:
            print(line)


def _fail_exit(args: argparse.Namespace, model_key: str, kind: str,
               detail: dict, message: str) -> int:
    record_dir = _write_failed_record(model_key, kind, detail)
    _emit(
        args,
        [
            "status: failed",
            f"error: {kind}: {message}",
            f"failed_record: {record_dir}",
            "未产生正式扫描包（outputs/_inventory 下无本输入的新目录）",
        ],
        {
            "status": "failed",
            "error": {"kind": kind, "message": message},
            "failed_record_dir": str(record_dir),
        },
    )
    return 1


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def _cmd_inspect(args: argparse.Namespace) -> int:
    model_key = args.model
    try:
        workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else find_workspace_root()
        inventory = build_inventory(model_key, workspace_root=workspace_root)
    except StructureReaderError as exc:
        return _fail_exit(
            args, model_key, exc.__class__.__name__,
            {"named_structure_reader_error": True, "message": str(exc), "traceback": traceback.format_exc()},
            str(exc),
        )
    except Exception as exc:  # 未预期异常也留具名记录，不裸 traceback
        return _fail_exit(
            args, model_key, exc.__class__.__name__,
            {"named_structure_reader_error": False, "message": str(exc), "traceback": traceback.format_exc()},
            str(exc),
        )

    baseline_statuses = {item["name"]: item["status"] for item in inventory["baseline_comparison"]}
    bad_baselines = {name: status for name, status in baseline_statuses.items()
                     if status in _BAD_BASELINE_STATUSES}
    if bad_baselines:
        kind = ("baseline_mismatch" if "mismatch" in bad_baselines.values()
                else "baseline_schema_unrecognized")
        return _fail_exit(
            args, model_key, kind,
            {"baseline_statuses": baseline_statuses,
             "baseline_comparison": inventory["baseline_comparison"]},
            f"源码基线比对未通过: {bad_baselines}；不产生成功扫描包",
        )

    if baseline_statuses and all(status == "match" for status in baseline_statuses.values()):
        status = "success"
    else:
        status = "unreviewed"  # 缺基线（含完全未声明基线）：可输出诊断扫描，但不标 reviewed

    try:
        entry = get_model_entry(model_key, workspace_root=workspace_root)
        registry_path = get_model_config_path(model_key, workspace_root=workspace_root)
        identity = _identity_document(model_key, inventory, workspace_root, registry_path)
        content_id = _sha256_bytes(_canonical_json_bytes(identity))
        documents = {
            "input_manifest.json": _pretty_json_bytes(
                _input_manifest_document(model_key, inventory, entry, workspace_root)
            ),
            "capability_report.json": _pretty_json_bytes(_capability_document(inventory)),
            "README.md": _readme_document(model_key, content_id, status, inventory),
        }
        doc_hashes = {name: _sha256_bytes(data) for name, data in documents.items()}
        documents["run_manifest.json"] = _pretty_json_bytes(
            _run_manifest_document(status, content_id, identity, doc_hashes)
        )
        target = _outputs_root() / "_inventory" / model_key / content_id
        outcome = _materialize(target, documents)
    except OutputConsistencyError as exc:
        return _fail_exit(
            args, model_key, "output_consistency_conflict",
            {"message": str(exc), "output_dir": str(target)},
            str(exc),
        )
    except Exception as exc:
        return _fail_exit(
            args, model_key, exc.__class__.__name__,
            {"named_structure_reader_error": False, "message": str(exc), "traceback": traceback.format_exc()},
            str(exc),
        )

    final_hashes = {name: doc_hashes.get(name) or _sha256_bytes(documents[name])
                    for name in _CONTENT_FILES}
    equivalent = inventory["raw_to_effective_pairs"]["equivalent_pairs"]
    equivalent_match = sum(1 for item in equivalent if item.get("match"))
    non_equivalent = inventory["raw_to_effective_pairs"]["non_equivalent_pairs"]
    weights_actual = inventory["weights"]["actual"]
    reviewed = status == "success"
    payload = {
        "status": status,
        "reviewed": reviewed,
        "stage": "S0 输入扫描（input inventory）",
        "model_key": model_key,
        "display_model_id": inventory["display_model_id"],
        "workspace_root": str(workspace_root),
        "content_id": content_id,
        "output_dir": str(target),
        "reused": outcome == "reused",
        "files": list(_CONTENT_FILES),
        "inputs_count": len(inventory["inputs"]),
        "baseline_statuses": baseline_statuses,
        "weights": {
            "safetensors_count": weights_actual["safetensors_count"],
            "index_present": weights_actual["index_present"],
            "consistency": inventory["weights"]["consistency"]["status"],
        },
    }
    _emit(
        args,
        [
            f"status: {status}" + ("" if reviewed else "（缺基线，未审查；unreviewed 不表示通过）"),
            "阶段: S0 输入扫描（input inventory；不是模型结构图）",
            f"model: {model_key} ({inventory['display_model_id']})",
            f"workspace_root: {workspace_root}",
            f"content_id: {content_id}",
            f"output_dir: {target}",
            "files:",
            *(f"  - {name}  sha256={final_hashes[name]}" for name in _CONTENT_FILES),
            f"inputs: {len(inventory['inputs'])} 个（含 model_manifest）",
            f"raw↔effective 对照: 等价对 {equivalent_match}/{len(equivalent)} match；"
            f"非等价对照 {len(non_equivalent)} 条（全量配置数据持久化属后续阶段）",
            f"weights: 磁盘 {weights_actual['safetensors_count']} 个 safetensors；"
            f"索引{'存在' if weights_actual['index_present'] else '缺失'}；"
            f"与清单一致性 {inventory['weights']['consistency']['status']}",
            f"baselines: {baseline_statuses if baseline_statuses else '（未声明基线）'}",
            f"interpreter: Python {inventory['capabilities']['interpreter']['python_version']}"
            f"（{inventory['capabilities']['interpreter']['implementation']}）"
            f" @ {inventory['capabilities']['interpreter']['python_executable']}",
            f"result: {outcome}" + ("（既有目录一致，幂等复用）" if outcome == "reused" else ""),
        ],
        payload,
    )
    return 0


# ---------------------------------------------------------------------------
# build（工作包 C）：连通 classify→extract→validate→场景→render→export
# ---------------------------------------------------------------------------

_BUILD_FILES = (
    "README.md", "run_manifest.json",
    "data/input_manifest.json", "data/layer_plan.json",
    "data/semantic_graph.json", "data/instance_bindings.json",
    "data/scenarios.json", "data/variants.json",
    "reports/reading_guide.md", "reports/validation_report.json",
    "reports/source_evidence.json", "reports/coverage_report.json",
    "viewer/structure.html",
    "tables/connections.csv", "tables/tensor_axes.csv",
    "logs/build.log",
)


def _build_identity_document(model_key: str, inv: dict, workspace_root: Path,
                             scenario_set: str) -> dict:
    """build 的 analysis_id 规范化输入（不含绝对路径/时间戳）。"""
    from .build_pipeline import _tool_sources_hash
    return {
        "schema": "structure_reader_build_identity_v1",
        "model_key": model_key,
        "scenario_set": scenario_set,
        "input_sha256": {item["role"]: item["sha256"] for item in inv["inputs"]},
        "tool_sources": _tool_sources_hash(_project_root()),
    }


def _build_documents(model_key: str, workspace_root: Path, scenario_set: str,
                     analysis_id: str, inv: dict, build_log: list[str]) -> dict:
    """生成全部 V1 交付文件（字节 → 落盘 dict）。"""
    from .build_pipeline import (
        _decode_version_chain,
        _instantiate_scenarios,
    )
    from .export.semantic_tables import (
        export_connections_csv,
        export_tensor_axes_csv,
    )
    from .build_pipeline import BuildError
    from .checks.deepseek_v4_contracts import build_validation_report, validate_model_contracts
    from .checks.graph_contracts import validate_graph_references
    from .extract.deepseek_v4_reference import build_reference_graph
    from .graph_document import graph_to_json
    from .layer_plan import build_layer_plan
    from .render.structure_page import render_structure_html
    from .rules.deepseek_v4_flash_0731 import load_model_values

    values = load_model_values(workspace_root)
    plan = build_layer_plan(workspace_root, values=values, build_graph=True)
    graph, registry, values = build_reference_graph(workspace_root, values=values)
    # V2：四类变体图全部构建进产物（任一失败即整体失败）
    from .extract.deepseek_v4_reference import build_scenario_views
    from .extract.deepseek_v4_variants import (
        build_all_variant_graphs, VARIANT_SPECS)
    variant_graphs = build_all_variant_graphs(workspace_root, values=values)
    build_log.append(
        f"extract: {len(graph.nodes)} nodes / {len(graph.edges)} edges / "
        f"{len(registry.all())} evidence entries / "
        f"{len(variant_graphs)} variants"
    )
    ref_issues = validate_graph_references(graph)
    if ref_issues:
        raise BuildError(f"通用引用检查未通过: {ref_issues[:5]}")
    contract_issues = validate_model_contracts(graph)
    if contract_issues:
        raise BuildError(f"模型契约未通过: {contract_issues[:5]}")
    build_log.append("validate: graph references OK, model contracts OK")

    # B4：DSpark 整体集成（六组件合并图 + 跨图连接 + 场景状态 + 集成契约）
    # ——先于 coverage/validation 构建（两者引用其事实；失败即整体失败）
    from .checks.dspark_contracts import build_dspark_validation_report
    from .extract.dspark_integration import (
        build_dspark_integration_graph,
        dspark_scenario_views,
        integration_document,
    )
    dspark = build_dspark_integration_graph(workspace_root, values=values)
    dspark_validation = build_dspark_validation_report(dspark.graph)
    # 防线顺序：先核实际图本身（original 契约），再核受控错误检出——
    # 坏图传入时具名报契约违规（而非先报 mutation 注入异常），诊断更准确
    if dspark_validation["original_contracts"]["status"] != "pass":
        raise BuildError(
            "DSpark 集成契约未通过: "
            + "; ".join(
                f"{i['code']}@{i['path']}"
                for i in dspark_validation["original_contracts"]["issues"])
            + "；不落成功报告")
    if not dspark_validation["controlled_errors"]["all_detected"]:
        raise BuildError(
            "DSpark 集成受控错误未全部检出: "
            + "; ".join(
                f"{r['mutation_id']} missing {r['missing_codes']}"
                for r in dspark_validation["controlled_errors"]["results"]
                if not r["detected"])
            + "；不落成功报告")
    dspark_integration_doc = integration_document(dspark)
    # 实例角色/参数绑定面板数据（从实际集成图 + 展开绑定表派生，全部
    # 联查真实图节点/张量；进页面 dsparkView 与 data/dspark/integration.json）
    from .extract.dspark_integration import dspark_role_bindings
    dspark_role_bindings_doc = dspark_role_bindings(dspark)
    # 层规划 DSpark 读出关系与实际集成图接线一致断言（防旧一对一映射；
    # 不一致即整体失败，不落成功报告）
    from .layer_plan import assert_dspark_readout_binding
    assert_dspark_readout_binding(plan, dspark.graph)
    # 集成图场景视图：全部 standard 场景对集成图真实求值（guard 三态 +
    # 窗口/槽位/草稿位置），落盘 data/dspark/scenarios.json
    from .rules.deepseek_v4_flash_0731 import SCENARIOS, scenario_base_env
    dspark_views = dspark_scenario_views(dspark, [
        dict(scenario_base_env(values, s), scenario_id=s.scenario_id)
        for s in SCENARIOS
    ])
    build_log.append(
        "dspark_integration: " + f"{len(dspark.graph.nodes)} nodes / "
        f"{len(dspark.graph.edges)} edges / "
        f"{len(dspark.connections)} cross-graph connections / "
        f"{len(dspark.registry.all())} evidence entries / "
        f"contracts OK / {len(dspark_validation['controlled_errors']['results'])}"
        " mutations all detected")

    # 证据合并发布：主图 61 条 + DSpark 集成 122 条同 ID 逐字段一致合并
    # （冲突具名拒绝，不允许后写覆盖为另一语义），同一份注册表进页面
    # payload 与 reports/source_evidence.json——五图（主图/四变体/DSpark）
    # 全部证据引用在此闭合。
    from .extract.dspark_integration import merge_evidence_registries
    merged_registry = merge_evidence_registries([registry, dspark.registry])

    scenarios = _instantiate_scenarios(values)
    chain = _decode_version_chain(values)
    display = inv["display_model_id"]

    plan_dict = plan.to_dict()
    plan_dict["decode_version_chain"] = chain
    scenarios_doc = {
        "schema": "scenario_instances_v1",
        "scenario_set": scenario_set,
        "default_scenario": "decode_p128",
        "instances": scenarios,
    }
    evidence_doc = {
        "schema": "source_evidence_v1",
        "entries": [e.to_dict() for e in merged_registry.all()],
    }
    # 证据闭合断言：五图（主图/四变体/DSpark）节点/边/guard/状态迁移/
    # 别名/折叠边界引用的全部 ev_ 前缀证据 ID 必须都在合并注册表内（轴
    # 引用中的 "model.py:NNN" 源码定位文本非 registry ID，按前缀区分，
    # 不误当缺失证据）。任一悬空即整体失败——不落成功报告。
    def _evidence_closure(gobj) -> list[str]:
        from structure_reader.graph_document import SemanticGraph
        assert isinstance(gobj, SemanticGraph)
        ids: set[str] = set()
        for n in gobj.nodes:
            ids.update(r for r in n.evidence_refs if r.startswith("ev_"))
        for e in gobj.edges:
            ids.update(r for r in e.evidence_refs if r.startswith("ev_"))
        for guard in gobj.guards:
            ids.update(r for r in guard.evidence_refs if r.startswith("ev_"))
        for tr in gobj.state_transitions:
            ids.update(r for r in tr.evidence_refs if r.startswith("ev_"))
        for alias in gobj.alias_relations:
            ids.update(r for r in alias.evidence_refs if r.startswith("ev_"))
        for fg in gobj.fold_groups:
            for bm in fg.boundary_mappings:
                ids.update(r for r in bm.evidence_refs if r.startswith("ev_"))
        for ax in gobj.named_axes:
            if ax.evidence_ref.startswith("ev_"):
                ids.add(ax.evidence_ref)
        for t in gobj.tensors:
            for ax in t.axes:
                if ax.evidence_ref.startswith("ev_"):
                    ids.add(ax.evidence_ref)
        known = {e.evidence_id for e in merged_registry.all()}
        return sorted(ids - known)

    dangling = _evidence_closure(graph)
    for _key, (vg_i, _vr_i, _vv_i) in variant_graphs.items():
        dangling += _evidence_closure(vg_i)
    dangling += _evidence_closure(dspark.graph)
    if dangling:
        raise BuildError(
            "证据引用未闭合（悬空 ev_ 引用）: " + ", ".join(dangling[:10])
            + "；不落成功报告")
    coverage_doc = {
        "schema": "coverage_report_v1",
        "extracted": list(plan.coverage.extracted),
        "pending_extraction": list(plan.coverage.pending_extraction),
        "uncovered": list(plan.coverage.uncovered),
        "dspark": {
            "status": "extracted_integrated",
            "template_id": dspark.graph.template_id,
            "note": "DSpark 整体集成图已构建并通过集成契约（首入口/公共 "
                    "stage×3/末出口 head/Markov/Confidence 真实合并 + 14 "
                    "跨图连接；data/dspark/ 与阅读页 DSpark 图同源）",
        },
        "evidence_coverage_counts": {
            c: len(merged_registry.by_coverage(c))
            for c in ("located", "source_checked", "derived", "uncovered")
        },
        "evidence_registry": {
            "main_entries": len(registry.all()),
            "dspark_entries": len(dspark.registry.all()),
            "merged_entries": len(merged_registry.all()),
            "note": "主图 61 条与 DSpark 集成 122 条同 ID 逐字段一致合并"
                    "（冲突具名拒绝）；五图全部 ev_ 证据引用在此闭合",
        },
        "notes": [
            "kernel.py 的 tilelang JIT kernel 数值实现未核对（不透明边界）",
            "cache 实际内容 / 每专家 token 数 / Indexer 实际可见条目数 unknown",
            "DSpark 实际 token/采样/置信度数值 unknown（静态结构分析）",
            "本产物为静态结构分析，不含运行推理或数值等价验证",
        ],
    }
    # D4：validation_report 来自 build 期实际运行的受控错误（逐项 detected），
    # 不写死计数/结果；任何漏检按检查失败策略处理
    validation_doc = build_validation_report(graph)
    if not validation_doc["controlled_errors"]["all_detected"]:
        raise BuildError(
            "受控错误验证未全部检出: "
            + "; ".join(
                f"{r['mutation_id']} missing {r['missing_codes']}"
                for r in validation_doc["controlled_errors"]["results"]
                if not r["detected"])
            + "；不落成功报告")
    reading_guide = (
        f"# {display} 结构阅读指南\n\n"
        "打开 `viewer/structure.html`（双击或 file:// 打开；单文件自包含，"
        "无外网/CDN/依赖）。\n\n"
        "## 打开与导航\n\n"
        "- 首屏 L1 图默认显示 CSA·TopK（layer 4 代表模板）的模块总览；"
        "工具条\"图\"按钮切换四张真实图：CSA·TopK / CSA·Hash / HCA·TopK / "
        "SWA·Hash（同一渲染管线，切换后按所选图重建）。\n"
        "- 视图：\"模块总览\"（默认，模块盒+模块间投影边）→ 点击模块进入"
        "局部展开（真实节点/边有向图）→ \"完整图\"（全节点/边拓扑）。\n"
        "- 场景：默认 decode_p128；图工具条与下方\"场景\"面板两套按钮同步"
        "（统一入口），guard 三态/压缩路径细节随场景切换。\n"
        "- 可读性：局部展开与完整图默认适配保证节点标签实际 ≥12px；横向"
        "溢出可拖拽平移（起点从输入侧开始）；\"定位选中\"把选中节点/边居中；"
        "\"查看全局范围\"显式整体缩小（想一眼看全图时用）；\"适配视图\"恢复"
        "可读默认。\n"
        "- 详情：点击节点/边/模块/投影边看真实端口/张量/证据（近图面板）；"
        "图下方 L1 节点表/场景面板/状态版本/层规划/证据为补充。\n"
        "- L0 整体层序：Embedding → HC Expand → 43 层（四组 2/1/20/20）→ "
        "HC Head → Final RMSNorm → LM Head；DSpark 外部读出（40/41/42）。\n"
        "- DSpark 整体图（工具条\"图\"按钮 DS）：首入口（40/41/42 readout "
        "有序 concat → stage0 唯一 main_proj/main_norm → 唯一 main_x 三 "
        "stage 分发；noise+seed 首列 → 共享 embed → HC 展开）→ 公共 stage "
        "模板三实例 mtp.0/1/2（同一模板结构，参数路径/窗口 cache 逐实例唯一"
        "；可展开看 decode 双 mHC+DSparkAttention/MoE 完整链与 prefill "
        "passthrough）→ 末出口（hc_head→norm→共享 head full_logits→Markov "
        "K5 递推→Confidence）→ 返回三元组 output_ids/logits/confidence。"
        "跨图连接 14 条边可联查（data/dspark/integration.json）。\n"
        "- 轴/证据：每个节点/边可追到来源证据（行号+原文内嵌页面）；"
        "逻辑/物理布局与 dtype 在 tensor_axes.csv 与轴定义分开。\n\n"
        "## 覆盖边界（诚实声明）\n\n"
        "- 四类主干模板与 DSpark 整体图均已提取（均可视阅读；DSpark 为"
        "六组件真实合并集成，跨图连接/实例/角色/场景状态同源落盘）。\n"
        "- kernel 数值、fast_hadamard_transform 未覆盖——页面直接标出。\n"
        "- 场景/压缩路径逐步实例化面板为主图（CSA·TopK）数据；变体图场景"
        "面板显示该图真实 env/guard/派生数量与前提（同源 data/variants/）。\n"
    ).encode("utf-8")
    readme = (
        f"# {display} — 模型结构阅读产物\n\n"
        f"- analysis_id: `{analysis_id}`（内容寻址；同输入+同工具版本复用）\n"
        f"- 场景集: {scenario_set}（默认显示 decode_p128）\n\n"
        "## 阅读\n\n"
        "1. `viewer/structure.html` — 自包含离线页（双击/file:// 打开）："
        "四图统一可视切换（CSA·TopK 主图 + CSA·Hash / HCA·TopK / SWA·Hash）"
        " + DSpark 整体图（工具条 DS 按钮）、模块总览/局部展开/完整图三视图、"
        "场景切换与选择详情、证据查看\n"
        "2. `reports/reading_guide.md` — 阅读顺序与覆盖边界\n"
        "3. `data/semantic_graph.json` + `data/variants/<key>/semantic_graph.json`"
        " + `data/dspark/semantic_graph.json`（DSpark 整体集成图）+ "
        "`tables/*.csv` + `data/variants/<key>/*.csv` + "
        "`data/dspark/*.csv` — 同源数据（ID 可联查）\n\n"
        "## 页面使用（viewer/structure.html）\n\n"
        "- \"图\"按钮切五图（四主干 + DSpark 整体）；\"模块总览/完整图\""
        "切视图；场景按钮两处同步。\n"
        "- 拖拽平移、滚轮/按钮缩放；\"定位选中\"居中所选；\"查看全局范围\""
        "整体缩小；\"适配视图\"恢复可读默认（节点标签实际 ≥12px）。\n"
        "- 点击节点/边/模块/投影边出详情（端口/张量/轴/证据，近图面板）。\n\n"
        "## 边界（诚实声明）\n\n"
        "- 静态结构分析：不执行模型代码，不声称推理/运行追踪/数值等价已验证\n"
        "- cache 实际内容、每专家 token 数、Indexer 实际可见条目数 unknown；"
        "DSpark 实际 token/采样/置信度数值 unknown\n"
        "- kernel 内部数值、fast_hadamard_transform 未覆盖；DSpark 整体图"
        "为静态结构集成（HF 实现无对应类，见 data/dspark/integration.json "
        "hf_difference）；完整 V2 独立门禁与用户阅读验收待完成\n"
        "（见 coverage_report）\n"
    ).encode("utf-8")

    graph_json = graph_to_json(graph)
    # V2：四类完整图数据（页面切换：每变体图 JSON + 适用场景视图）
    variant_graph_json = {
        key: json.loads(graph_to_json(vg))
        for key, (vg, _vr, _vv) in variant_graphs.items()
    }
    # V2：四类完整图落盘（每变体 graph JSON + 适用场景视图 + 同源 CSV）
    variant_docs: dict[str, bytes] = {}
    variant_views: dict[str, list] = {}
    for key, (vg, vreg, vv) in variant_graphs.items():
        spec = VARIANT_SPECS[key]
        vviews = build_scenario_views(
            vv, vreg, compress_ratio=spec.compress_ratio,
            has_indexer=spec.has_indexer, has_compress=spec.compress_ratio > 0)
        variant_views[key] = [
            {"scenario_id": x.scenario_id, "env": x.env,
             "guard_activations": {a.guard_id: a.active_for_scenario
                                   for a in x.guard_activations},
             "derived": x.derived, "indexer": x.indexer,
             "premise": x.premise}
            for x in vviews
        ]
        variant_docs[f"data/variants/{key}/semantic_graph.json"] = (
            graph_to_json(vg) + "\n").encode("utf-8")
        variant_docs[f"data/variants/{key}/scenarios.json"] = _pretty_json_bytes({
            "schema": "scenario_instances_v1",
            "variant": key,
            "compress_ratio": spec.compress_ratio,
            "default_scenario": "decode_p128" if spec.compress_ratio else "decode_p2",
            "instances": variant_views[key],
        })
        variant_docs[f"data/variants/{key}/connections.csv"] = \
            export_connections_csv(vg)
        variant_docs[f"data/variants/{key}/tensor_axes.csv"] = \
            export_tensor_axes_csv(vg)
    # 四类场景聚合进主 scenarios.json（页面切换用）
    scenarios_doc["variant_views"] = variant_views

    html = render_structure_html(
        graph_json=graph_json, plan_dict=plan_dict,
        scenarios_json=json.dumps(scenarios_doc, ensure_ascii=False),
        evidence_json=json.dumps(evidence_doc, ensure_ascii=False),
        model_key=model_key, analysis_id=analysis_id, display_name=display,
        variant_graphs=variant_graph_json,
        variant_views=variant_views,
        dspark_graph_json=graph_to_json(dspark.graph),
        dspark_view_json=json.dumps(
            {"views": dspark_views,
             "integration": {
                 "connections": dspark_integration_doc["connections"],
                 "instances": dspark_integration_doc["instances"],
                 "boundaries": dspark_integration_doc["boundaries"],
                 "role_bindings": dspark_role_bindings_doc},
             "integration_doc_note":
                 "完整组件映射（历史 scope 与当前绑定的关系）见 "
                 "data/dspark/integration.json"},
            ensure_ascii=False),
    )
    docs = {
        "README.md": readme,
        "data/input_manifest.json": _pretty_json_bytes(
            _input_manifest_document(model_key, inv, get_model_entry(
                model_key, workspace_root=workspace_root), workspace_root)
        ),
        "data/layer_plan.json": _pretty_json_bytes(plan_dict),
        "data/semantic_graph.json": (graph_json + "\n").encode("utf-8"),
        "data/variants.json": _pretty_json_bytes({
            "schema": "variant_graphs_v1",
            "variants": {
                key: {
                    "template_id": g.template_id,
                    "node_count": len(g.nodes),
                    "edge_count": len(g.edges),
                    "representative_layer": spec.representative_layer,
                    "compress_ratio": spec.compress_ratio,
                    "has_indexer": spec.has_indexer,
                    "gate_kind": spec.gate_kind,
                }
                for key, (g, _r, _v) in variant_graphs.items()
                for spec in [VARIANT_SPECS[key]]
            },
        }),
        "data/instance_bindings.json": _pretty_json_bytes({
            "schema": "instance_bindings_v1",
            "bindings": [b.to_dict() for b in plan.bindings],
            "decode_version_chain": chain,
        }),
        "data/scenarios.json": _pretty_json_bytes(scenarios_doc),
        "reports/reading_guide.md": reading_guide,
        "reports/validation_report.json": _pretty_json_bytes(validation_doc),
        "reports/source_evidence.json": _pretty_json_bytes(evidence_doc),
        "reports/coverage_report.json": _pretty_json_bytes(coverage_doc),
        "viewer/structure.html": html,
        "tables/connections.csv": export_connections_csv(graph),
        "tables/tensor_axes.csv": export_tensor_axes_csv(graph),
    }
    docs.update(variant_docs)
    docs["data/dspark/semantic_graph.json"] = (
        graph_to_json(dspark.graph) + "\n").encode("utf-8")
    docs["data/dspark/integration.json"] = _pretty_json_bytes(
        {**dspark_integration_doc,
         "role_bindings": dspark_role_bindings_doc})
    docs["data/dspark/scenarios.json"] = _pretty_json_bytes({
        "schema": "dspark_scenario_views_v1",
        "template_id": dspark.graph.template_id,
        "default_scenario": "decode_p128",
        "views": dspark_views,
    })
    docs["data/dspark/connections.csv"] = \
        export_connections_csv(dspark.graph)
    docs["data/dspark/tensor_axes.csv"] = \
        export_tensor_axes_csv(dspark.graph)
    docs["reports/dspark_validation.json"] = _pretty_json_bytes(
        dspark_validation)
    docs["logs/build.log"] = ("\n".join(build_log) + "\n").encode("utf-8")
    return docs


def _cmd_build(args: argparse.Namespace) -> int:
    from .build_pipeline import BuildError

    model_key = args.model
    build_log: list[str] = []
    target: Path | None = None
    try:
        workspace_root = (Path(args.workspace_root).resolve()
                          if args.workspace_root else find_workspace_root())
        from .inventory import build_inventory
        inv = build_inventory(model_key, workspace_root=workspace_root)
        bad = [item for item in inv["baseline_comparison"]
               if item["status"] in ("mismatch", "schema_unrecognized")]
        if bad:
            return _fail_exit(
                args, model_key,
                "baseline_mismatch" if any(i["status"] == "mismatch" for i in bad)
                else "baseline_schema_unrecognized",
                {"baseline_comparison": inv["baseline_comparison"]},
                "输入基线失配；不构建",
            )
        build_log.append(
            "inputs: " + ", ".join(
                f"{i['role']}={i['sha256'][:12]}" for i in inv["inputs"])
        )
        identity = _build_identity_document(model_key, inv, workspace_root,
                                            args.scenario_set)
        analysis_id = _sha256_bytes(_canonical_json_bytes(identity))
        target = _outputs_root() / model_key / analysis_id
        docs = _build_documents(model_key, workspace_root, args.scenario_set,
                                analysis_id, inv, build_log)
        doc_hashes = {name: _sha256_bytes(data) for name, data in docs.items()}
        manifest = {
            "schema": "build_run_manifest_v1",
            "artifact": {"kind": "structure_reading", "stage": "V1"},
            "status": "success",
            "model_key": model_key,
            "display_model_id": inv["display_model_id"],
            "analysis_id": analysis_id,
            "scenario_set": args.scenario_set,
            "content_id_basis": identity,
            "files": [{"name": n, "sha256": doc_hashes[n]}
                      for n in sorted(docs) if n != "run_manifest.json"]
            + [{"name": "run_manifest.json", "sha256": None, "note": "本文件"}],
            "tools": _tool_identity(),
        }
        docs["run_manifest.json"] = _pretty_json_bytes(manifest)
        outcome = _materialize(target, docs)
    except OutputConsistencyError as exc:
        return _fail_exit(args, model_key, "output_consistency_conflict",
                          {"message": str(exc), "output_dir": str(target)},
                          str(exc))
    except BuildError as exc:
        return _fail_exit(args, model_key, "build_error",
                          {"message": str(exc), "traceback": traceback.format_exc()},
                          str(exc))
    except Exception as exc:
        return _fail_exit(args, model_key, exc.__class__.__name__,
                          {"message": str(exc),
                           "traceback": traceback.format_exc()}, str(exc))
    _emit(
        args,
        [
            "status: success",
            f"model: {model_key}",
            f"analysis_id: {analysis_id}",
            f"output_dir: {target}",
            f"page: {target / 'viewer' / 'structure.html'}",
            f"result: {outcome}" + ("（同输入幂等复用，产物已核对）" if outcome == "reused" else ""),
        ],
        {"status": "success", "model_key": model_key,
         "analysis_id": analysis_id, "output_dir": str(target),
         "page": str(target / "viewer" / "structure.html"),
         "reused": outcome == "reused"},
    )
    return 0


# ---------------------------------------------------------------------------
# argparse 入口
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="read_model_structure",
        description=(
            "structure_reader：模型特征结构阅读工具。"
            "已实现：inspect（S0 输入扫描）、build（结构阅读产物："
            "四图统一离线页 + 代表图/变体图数据 + 同源 CSV）"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser(
        "inspect", help="扫描模型输入并产出内容寻址的清单目录（outputs/_inventory/）"
    )
    inspect_parser.add_argument(
        "--model", required=True,
        help="模型键（structure_reader/configs/models/<model_key>.json 的文件名 stem）",
    )
    inspect_parser.add_argument(
        "--workspace-root", default=None,
        help="工作区根目录；缺省从包位置向上定位（同时含 hf_docs/ 与 models/），不依赖 cwd",
    )
    inspect_parser.add_argument(
        "--json", action="store_true", help="以 JSON 摘要输出（status 与实际路径；非整份模型数据）"
    )
    build_parser = commands.add_parser(
        "build", help="构建结构阅读产物：四图统一阅读页（变体切换/模块总览/"
                      "局部展开/完整图/场景/详情）+ 自包含离线 HTML + 同源 CSV"
                      "（outputs/<model_key>/<analysis_id>/）"
    )
    build_parser.add_argument(
        "--model", required=True,
        help="模型键（structure_reader/configs/models/<model_key>.json 的文件名 stem）",
    )
    build_parser.add_argument(
        "--scenario-set", default="standard",
        help="场景集（当前仅 standard：prefill T=3/4/5/127/128/129 + decode P=2/3/4/127/128）",
    )
    build_parser.add_argument(
        "--workspace-root", default=None,
        help="工作区根目录；缺省从包位置向上定位，不依赖 cwd",
    )
    build_parser.add_argument(
        "--json", action="store_true", help="以 JSON 摘要输出"
    )
    hf_parser = commands.add_parser(
        "build-hf",
        help="HF decoder 阅读样板构建（transformers/eager）：reading_document + 单文件离线 HTML"
             " + 两表 xlsx（outputs/<model_key>/<analysis_id>/）",
    )
    hf_parser.add_argument(
        "--model", required=True,
        help="模型键（当前仅 deepseek_v4_flash_0731_hf）",
    )
    hf_parser.add_argument(
        "--layer-index", type=int, default=4,
        help="样板层索引（默认 4：CSA + TopK MoE；其余变体属里程碑二）",
    )
    hf_parser.add_argument(
        "--workspace-root", default=None,
        help="工作区根目录；缺省从包位置向上定位，不依赖 cwd",
    )
    hf_parser.add_argument(
        "--json", action="store_true", help="以 JSON 摘要输出"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口；返回退出码（0 成功，1 failed，2 参数错误）。"""
    parser = _build_parser()
    args = parser.parse_args(argv)  # 非法参数由 argparse 以退出码 2 终止
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "build-hf":
        return _cmd_build_hf(args)
    parser.error(f"未知命令: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


def _cmd_build_hf(args: argparse.Namespace) -> int:
    """HF 阅读样板构建命令：提取→检查→HTML/xlsx→原子落盘→失败隔离。"""
    model_key = args.model
    try:
        workspace_root = (
            Path(args.workspace_root).resolve() if args.workspace_root else find_workspace_root()
        )
    except StructureReaderError as exc:
        return _fail_exit(args, model_key, exc.__class__.__name__,
                          {"named_structure_reader_error": True, "message": str(exc),
                           "traceback": traceback.format_exc()}, str(exc))
    try:
        from .build_hf_decoder import HfBuildError, run_hf_build

        result, documents = run_hf_build(
            workspace_root, _project_root(), layer_index=args.layer_index, model_key=model_key,
        )
        target = result.output_dir
        outcome = _materialize(target, documents)
    except HfBuildError as exc:
        return _fail_exit(
            args, model_key, "hf_build_error",
            {"named_structure_reader_error": True, "message": str(exc),
             "traceback": traceback.format_exc()},
            str(exc),
        )
    except OutputConsistencyError as exc:
        return _fail_exit(
            args, model_key, "output_consistency_conflict",
            {"message": str(exc), "output_dir": str(target)},
            str(exc),
        )
    except Exception as exc:  # 未预期异常也留具名记录
        return _fail_exit(
            args, model_key, exc.__class__.__name__,
            {"named_structure_reader_error": False, "message": str(exc),
             "traceback": traceback.format_exc()},
            str(exc),
        )

    deliverables = documents["run_manifest.json"]
    _emit(
        args,
        [
            "status: success",
            f"阶段: HF decoder 阅读样板（transformers/eager；里程碑一）",
            f"model: {model_key}（layer_idx={args.layer_index}）",
            f"workspace_root: {workspace_root}",
            f"analysis_id: {result.analysis_id}",
            f"output_dir: {target}",
            "files:",
            *(f"  - {name}  sha256={__import__('hashlib').sha256(data).hexdigest()}"
              for name, data in sorted(documents.items())),
            f"result: {outcome}" + ("（既有目录一致，幂等复用）" if outcome == "reused" else ""),
        ],
        {
            "status": "success",
            "stage": "HF decoder 阅读样板（里程碑一）",
            "model_key": model_key,
            "layer_index": args.layer_index,
            "analysis_id": result.analysis_id,
            "output_dir": str(target),
            "reused": outcome == "reused",
            "files": sorted(documents),
        },
    )
    return 0
