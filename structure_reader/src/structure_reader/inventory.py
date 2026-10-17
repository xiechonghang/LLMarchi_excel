"""S0 输入扫描：文件清单、构造态配置、权重存在性与能力报告。

纯标准库（Python >= 3.11）；不导入/执行模型代码，不使用 eval/exec，
不依赖 torch。所有返回值均为 JSON 可序列化的纯数据 dict。

对外入口为 :func:`build_inventory`；可运行入口见 ``scripts/read_model_structure.py``
（inspect 命令，内容寻址产物落 ``outputs/_inventory/``），使用文档见 ``docs/``。
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

from .model_registry import StructureReaderError, get_model_entry

__all__ = [
    "MissingInputError",
    "InvalidConfigError",
    "ConstructorStateError",
    "build_inventory",
]


class MissingInputError(StructureReaderError):
    """注册的输入文件在磁盘上缺失。"""


class InvalidConfigError(StructureReaderError):
    """输入文件存在但不是合法 JSON（或基线 schema 不可识别）。"""


class ConstructorStateError(StructureReaderError):
    """无法用受限 AST 解析恢复构造态（不支持的默认值表达式等）。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_file(path: Path, role: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MissingInputError(f"无法读取输入文件（role={role}）: {path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidConfigError(f"输入文件（role={role}）不是合法 JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidConfigError(f"输入文件（role={role}）顶层应为 JSON object: {path}")
    return data


def _source_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MissingInputError(f"无法读取输入文件: {path}") from exc


def _parse_source(source_text: str, source_path: Path) -> ast.Module:
    try:
        return ast.parse(source_text, filename=str(source_path))
    except SyntaxError as exc:
        raise ConstructorStateError(f"源码解析失败: {source_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# 受限 AST 字面量求值（不用 eval/exec）
# ---------------------------------------------------------------------------

_LITERAL_CALLS = {"tuple": "tuple", "list": "list", "dict": "dict"}
_ARITH_OPS = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod}


def _ast_literal(node: ast.AST, source_path: Path) -> object:
    """把 AST 节点转换为受限 Python 字面量；空 ``tuple()`` 等调用也被支持。"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_ast_literal(item, source_path) for item in node.elts]
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise ConstructorStateError(f"{source_path}: 不支持字典展开默认值")
        return {
            _ast_literal(key, source_path): _ast_literal(value, source_path)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_literal(node.operand, source_path)
        if isinstance(inner, (int, float)):
            return -inner
    if isinstance(node, ast.Name) and node.id in ("None", "True", "False"):
        return {"None": None, "True": True, "False": False}[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _LITERAL_CALLS
        and not node.args
        and not node.keywords
    ):
        return {"tuple": [], "list": [], "dict": {}}[_LITERAL_CALLS[node.func.id]]
    raise ConstructorStateError(
        f"{source_path}: 不支持的默认值表达式（行 {getattr(node, 'lineno', '?')}）: "
        f"{type(node).__name__}"
    )


def _fold_binop(node: ast.BinOp, source_path: Path) -> object | None:
    """对常量二元算术做受限折叠（仅 + - * / // %）；失败返回 None，不抛错不 eval。"""
    if type(node.op) not in _ARITH_OPS:
        return None
    try:
        left = _ast_literal(node.left, source_path)
        right = _ast_literal(node.right, source_path)
    except ConstructorStateError:
        return None
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    if isinstance(left, bool) or isinstance(right, bool):
        return None
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.FloorDiv):
        return left // right
    return left % right


def _literal_entry(node: ast.AST, source_path: Path) -> dict:
    """把默认值/赋值节点转成 {value, original_type, expression} 记录。

    非字面量（如变量名）返回 ``is_literal=False`` 的记录，不在此处抛错；
    是否视为致命错误由调用方决定（dataclass 默认值严格，入口赋值宽松）。
    """
    if isinstance(node, ast.BinOp):
        folded = _fold_binop(node, source_path)
        expression = ast.unparse(node)
        if folded is not None:
            return {
                "value": folded,
                "original_type": type(folded).__name__,
                "expression": expression,
                "is_literal": True,
            }
        return {
            "value": None,
            "original_type": None,
            "expression": expression,
            "is_literal": False,
            "note": "表达式超出受限常量折叠范围，未求值",
        }
    try:
        value = _ast_literal(node, source_path)
    except ConstructorStateError:
        return {
            "value": None,
            "original_type": None,
            "expression": ast.unparse(node),
            "is_literal": False,
            "note": "值不是字面量（变量/调用等），静态不求值",
        }
    if isinstance(node, ast.Tuple):
        original_type = "tuple"
    elif isinstance(node, ast.List):
        original_type = "list"
    elif isinstance(node, ast.Dict):
        original_type = "dict"
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _LITERAL_CALLS
    ):
        original_type = _LITERAL_CALLS[node.func.id]
    else:
        original_type = type(value).__name__
    return {
        "value": value,
        "original_type": original_type,
        "expression": ast.unparse(node),
        "is_literal": True,
    }


# ---------------------------------------------------------------------------
# ModelArgs dataclass 默认值（AST）
# ---------------------------------------------------------------------------


def _read_dataclass_defaults(source_path: Path, dataclass_symbol: str) -> list[dict]:
    """按声明顺序读取 dataclass 字段的字面量默认值。

    返回字段列表：{"field", "has_default", "value", "original_type",
    "expression", "line"}。value 统一 JSON 序列化（tuple → list，
    original_type 保留原始类型名）。
    """
    tree = _parse_source(_source_text(source_path), source_path)
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == dataclass_symbol):
            continue
        decorator_names = []
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Name):
                decorator_names.append(target.id)
        if not any(name.startswith("dataclass") for name in decorator_names):
            raise ConstructorStateError(
                f"{source_path}: 类 {dataclass_symbol} 未标注 @dataclass"
            )
        fields = []
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            field = {
                "field": stmt.target.id,
                "has_default": stmt.value is not None,
                "value": None,
                "original_type": None,
                "expression": None,
                "line": stmt.lineno,
            }
            if stmt.value is not None:
                literal = _literal_entry(stmt.value, source_path)
                field["value"] = literal["value"]
                field["original_type"] = literal["original_type"]
                field["expression"] = literal["expression"]
                if not literal["is_literal"]:
                    raise ConstructorStateError(
                        f"{source_path}: 字段 {field['field']} 默认值不是受限字面量（行 {stmt.lineno}）"
                    )
            fields.append(field)
        if not fields:
            raise ConstructorStateError(
                f"{source_path}: 类 {dataclass_symbol} 中未找到字段声明"
            )
        return fields
    raise ConstructorStateError(
        f"{source_path}: 未找到名为 {dataclass_symbol} 的类定义"
    )


# ---------------------------------------------------------------------------
# 入口脚本：构造调用点与入口覆盖
# ---------------------------------------------------------------------------


def _scan_entrypoint(source_path: Path, dataclass_symbol: str) -> dict:
    """扫描入口脚本中的构造调用、条件覆盖与入口参数来源。

    - ``constructor_call``：ModelArgs(...) 调用点（行号与调用表达式）。
    - ``overrides``：``args.<field> = ...`` 赋值；带所在 if 的条件（符号/源码/行号）。
      非字面量值（如 ``args.temperature = temperature``）记录值符号而不求值。
    - ``entrypoint_parameters``：顶层函数签名中带默认值的参数。
    """
    tree = _parse_source(_source_text(source_path), source_path)
    constructor_calls = []
    open_calls = []
    entrypoint_parameters = {}
    overrides = []
    if_stack: list[dict] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == dataclass_symbol:
                    constructor_calls.append(
                        {
                            "symbol": dataclass_symbol,
                            "line": child.lineno,
                            "call_source": ast.unparse(child),
                        }
                    )
                if isinstance(child.func, ast.Name) and child.func.id == "open":
                    open_calls.append({"line": child.lineno, "call_source": ast.unparse(child)})
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = []
                arguments = child.args
                positional = list(arguments.posonlyargs) + list(arguments.args)
                defaults = list(arguments.defaults)
                if defaults:
                    padded = [None] * (len(positional) - len(defaults)) + defaults
                    for arg, default in zip(positional, padded):
                        if default is not None:
                            literal = _literal_entry(default, source_path)
                            params.append(
                                {
                                    "parameter": arg.arg,
                                    "default_value": literal["value"],
                                    "line": arg.lineno,
                                }
                            )
                for kwarg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
                    if default is not None:
                        literal = _literal_entry(default, source_path)
                        params.append(
                            {
                                "parameter": kwarg.arg,
                                "default_value": literal["value"],
                                "line": kwarg.lineno,
                            }
                        )
                if params:
                    entrypoint_parameters[child.name] = params
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "args"
                    ):
                        record = {
                            "field": target.attr,
                            "line": child.lineno,
                            "statement": ast.unparse(child),
                            "guards": [dict(g) for g in if_stack],
                        }
                        if child.value is not None:
                            literal = _literal_entry(child.value, source_path)
                            record["value"] = literal["value"]
                            record["original_type"] = literal["original_type"]
                            record["expression"] = literal["expression"]
                            record["is_literal"] = literal["is_literal"]
                            if not literal["is_literal"]:
                                record["note"] = literal.get("note", "值非受限字面量，未求值")
                        overrides.append(record)
                        break
            pushed = False
            if isinstance(child, ast.If):
                if_stack.append(
                    {
                        "condition_source": ast.unparse(child.test),
                        "line": child.lineno,
                    }
                )
                pushed = True
            visit(child)
            if pushed:
                if_stack.pop()

    visit(tree)
    if not constructor_calls:
        raise ConstructorStateError(
            f"{source_path}: 未找到 {dataclass_symbol}(...) 构造调用"
        )
    return {
        "constructor_call": constructor_calls[0],
        "constructor_call_count": len(constructor_calls),
        "config_open_calls": open_calls,
        "entrypoint_parameters": entrypoint_parameters,
        "overrides": overrides,
    }


# ---------------------------------------------------------------------------
# 配置扫描
# ---------------------------------------------------------------------------


def _get_dotted(data: dict, dotted: str):
    current: object = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None, False
        current = current[part]
    return current, True


def _raw_config_section(raw: dict, raw_path: Path, hf_config_class: str | None,
                        hf_config_source: str) -> dict:
    return {
        "path": str(raw_path),
        "role": "root_config",
        "naming_convention": "HF（hidden_size/num_hidden_layers 等）",
        "hf_consumer": {
            "config_class": hf_config_class,
            "source": hf_config_source,
            "note": "根 config 由 HF 配置类消费；参考实现入口不直接读取本文件",
        },
        "fields": raw,
        "notes": [
            "本节原样保存根 config；与参考构造态（ModelArgs 命名）的映射见 raw_to_effective_pairs",
        ],
    }


def _constructor_state_section(defaults: list[dict], run_config: dict,
                               run_config_path: Path, dataclass_symbol: str,
                               dataclass_source: str) -> dict:
    """组装构造态：ModelArgs 默认值 + run_config 覆盖，逐字段记录来源。"""
    fields: dict[str, dict] = {}
    for order, default in enumerate(defaults):
        name = default["field"]
        if name in run_config:
            entry = {
                "value": run_config[name],
                "source": "run_config",
                "source_path": str(run_config_path),
                "order": order,
            }
        elif default["has_default"]:
            entry = {
                "value": default["value"],
                "source": "dataclass_default",
                "source_path": str(dataclass_source),
                "line": default["line"],
                "expression": default["expression"],
                "order": order,
            }
            if default["original_type"] is not None:
                entry["original_type"] = default["original_type"]
        else:
            raise ConstructorStateError(
                f"构造态字段 {name} 无 dataclass 默认值且未出现在运行配置中，"
                f"参考实现构造将失败"
            )
        fields[name] = entry
    unconsumed = sorted(set(run_config) - set(fields))
    return {
        "dataclass_symbol": dataclass_symbol,
        "dataclass_source": str(dataclass_source),
        "run_config_path": str(run_config_path),
        "merge_rule": "ModelArgs(**run_config)：run_config 键覆盖同名字段默认值；"
        "仅 dataclass 默认值存在的字段保持默认值",
        "fields": fields,
        "run_config_keys_not_consumed": unconsumed,
        "evidence": {
            "method": "AST 静态读取字面量默认值；未导入/执行模型代码，未使用 eval/exec",
        },
    }


def _hf_config_class_info(configuration_source: Path) -> dict:
    """从 HF configuration 源码定位消费根 config 的配置类与 model_type。"""
    tree = _parse_source(_source_text(configuration_source), configuration_source)
    classes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        model_type = None
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "model_type"
                and isinstance(stmt.value, ast.Constant)
            ):
                model_type = stmt.value.value
        if model_type is not None:
            classes.append({"class": node.name, "model_type": model_type, "line": node.lineno})
    return {
        "classes_declaring_model_type": classes,
        "note": "仅静态定位类名与 model_type；字段级消费映射属后续阶段",
    } if classes else {"classes_declaring_model_type": [], "note": "未找到声明 model_type 的配置类"}


# ---------------------------------------------------------------------------
# revision / git / 权重 / 能力 / 基线
# ---------------------------------------------------------------------------


def _git_checkout_info(checkout_root: Path, source_root: Path) -> dict:
    info: dict = {
        "checkout_root": str(checkout_root),
        "method": "git（仅读取 commit 与 dirty 状态，不做任何写操作）",
        "commit": None,
        "commit_available": False,
        "dirty": None,
        "dirty_paths": [],
    }
    try:
        commit_run = subprocess.run(
            ["git", "-C", str(checkout_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if commit_run.returncode == 0:
            info["commit"] = commit_run.stdout.strip()
            info["commit_available"] = True
        else:
            info["error"] = (commit_run.stderr or commit_run.stdout).strip() or "git rev-parse 失败"
            return info
        relative = source_root
        try:
            relative = source_root.relative_to(checkout_root)
        except ValueError:
            pass
        status_run = subprocess.run(
            ["git", "-C", str(checkout_root), "status", "--porcelain", "--", str(relative)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if status_run.returncode == 0:
            dirty_paths = [line[3:] for line in status_run.stdout.splitlines() if line.strip()]
            info["dirty"] = bool(dirty_paths)
            info["dirty_paths"] = dirty_paths
        else:
            info["error"] = (status_run.stderr or status_run.stdout).strip() or "git status 失败"
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["error"] = f"无法执行 git: {exc}"
    return info


def _weights_report(reference_root: Path, weights_spec: dict, manifest: dict) -> dict:
    """按磁盘现存文件计数 safetensors 与索引；manifest 声明数单独记录。"""
    shard_files = sorted(p.name for p in reference_root.glob(weights_spec["safetensors_pattern"]))
    all_safetensors = sorted(p.name for p in reference_root.glob("*.safetensors"))
    index_path = reference_root / weights_spec["index_file"]
    declared_files = manifest.get("downloaded_files", [])
    declared_shards = sorted(f for f in declared_files if f.endswith(".safetensors"))
    index_declared = weights_spec["index_file"] in declared_files
    actual_count = len(all_safetensors)
    declared_count = len(declared_shards)
    if actual_count == declared_count:
        consistency = {"status": "match", "note": "磁盘现存 shard 数与 manifest 声明一致"}
    else:
        consistency = {
            "status": "mismatch",
            "note": f"磁盘现存 {actual_count} 个 safetensors 文件，manifest 声明 {declared_count} 个；"
            "以磁盘现存为准，下载清单不冒充现存权重",
        }
    return {
        "report_basis": "磁盘现存文件计数；下载清单仅作声明对照",
        "actual": {
            "safetensors_count": actual_count,
            "shard_pattern": weights_spec["safetensors_pattern"],
            "shard_pattern_matches": shard_files,
            "index_present": index_path.is_file(),
            "index_path": str(index_path),
        },
        "manifest_declared": {
            "safetensors_count": declared_count,
            "index_declared": index_declared,
            "declared_files_total": len(declared_files),
        },
        "consistency": consistency,
    }


def _capability_report() -> dict:
    """基于当前解释器与 find_spec 的能力报告；不导入重型包、不执行模型。"""
    optional = {}
    for package in ("torch", "safetensors", "transformers", "openpyxl"):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError) as exc:
            optional[package] = {"available": False, "error": str(exc)}
        else:
            optional[package] = {
                "available": spec is not None,
                "module_spec": spec.name if spec is not None else None,
            }
    return {
        "interpreter": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "core_dependencies": {
            "stdlib_only": True,
            "torch_required": False,
            "min_python": "3.11",
        },
        "optional_packages": optional,
        "implemented_capabilities": [
            "input_file_inventory（角色/相对路径/SHA256）",
            "model_manifest_revision_and_hf_git_state",
            "raw_root_config_preservation",
            "constructor_state_via_ast（ModelArgs 默认值 + run_config 覆盖）",
            "entrypoint_override_scan（条件/符号/行号）",
            "weights_existence_report",
            "capability_report",
            "source_baseline_comparison（缺失/匹配/失配）",
            "cli_inspect_command（内容寻址产物/.work 暂存提交/幂等复用/_failed 失败记录）",
            "s0_usage_docs（docs/ 用户指南/开发者指南/输出说明/进度；inspect 不自动更新基线与文档）",
        ],
        "not_implemented": [
            {"feature": "semantic_graph", "stage": "S1+", "note": "图 IR/轴/条件尚未实现"},
            {"feature": "layer_plan", "stage": "S2", "note": "模板分组与实例绑定尚未实现"},
            {"feature": "reading_page", "stage": "S5", "note": "阅读页/渲染尚未实现"},
            {"feature": "scenario_evaluation", "stage": "S1+", "note": "场景条件求值尚未实现；入口覆盖未并入构造态"},
            {"feature": "runtime_tracing", "stage": "未排期", "note": "运行追踪证据未实现"},
            {"feature": "numeric_verification", "stage": "未排期", "note": "数值等价验证未实现"},
            {"feature": "skill", "stage": "S8", "note": "项目 Skill 未实现"},
        ],
    }


def _compare_source_baselines(workspace_root: Path, model_entry: dict,
                              inputs: list[dict]) -> list[dict]:
    """与已存在的源码基线比较；基线缺失记录 missing，不创建、不覆盖。"""
    baseline_root = workspace_root / "structure_reader" / "configs" / "source_baselines"
    actual_by_path = {item["path"]: item["sha256"] for item in inputs}
    results = []
    for name in model_entry["source_baselines"]:
        baseline_path = baseline_root / f"{name}.json"
        if not baseline_path.is_file():
            results.append(
                {
                    "name": name,
                    "path": str(baseline_path),
                    "status": "missing",
                    "note": "基线文件不存在；固定基线在 structure_reader/configs/source_baselines 维护，inspect 不自动更新",
                }
            )
            continue
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidConfigError(f"审查基线不是合法 JSON: {baseline_path}: {exc}") from exc
        if not isinstance(baseline, dict):
            results.append(
                {
                    "name": name,
                    "path": str(baseline_path),
                    "status": "schema_unrecognized",
                    "note": f"基线顶层必须是 JSON 对象；实际为 {type(baseline).__name__}",
                }
            )
            continue
        declared = baseline.get("source_files")
        if isinstance(declared, dict):
            declared = [{"path": path, "sha256": sha} for path, sha in declared.items()]
        if not isinstance(declared, list):
            results.append(
                {
                    "name": name,
                    "path": str(baseline_path),
                    "status": "schema_unrecognized",
                    "note": "基线缺少可识别的 source_files 字段（期望 list 或 {path: sha256}）",
                }
            )
            continue
        file_results = []
        for item in declared:
            path = item.get("path")
            declared_sha = item.get("sha256")
            actual_sha = actual_by_path.get(path)
            if actual_sha is None:
                status = "unverifiable"
                note = "本轮未扫描该路径"
            elif actual_sha == declared_sha:
                status = "match"
                note = None
            else:
                status = "mismatch"
                note = "实际 SHA256 与基线声明不一致"
            file_results.append(
                {
                    "path": path,
                    "declared_sha256": declared_sha,
                    "actual_sha256": actual_sha,
                    "status": status,
                    "note": note,
                }
            )
        statuses = {item["status"] for item in file_results}
        if not file_results:
            overall = "schema_unrecognized"
        elif statuses == {"match"}:
            overall = "match"
        elif "mismatch" in statuses or "unverifiable" in statuses:
            overall = "mismatch"
        else:
            overall = "schema_unrecognized"
        results.append(
            {
                "name": name,
                "path": str(baseline_path),
                "status": overall,
                "file_comparison": file_results,
            }
        )
    return results


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------


def build_inventory(model_key: str, workspace_root: str | Path | None = None) -> dict:
    """扫描指定模型的输入并返回可 JSON 序列化的 inventory dict。

    参数:
        model_key: 注册表中的模型标识（当前仅 ``deepseek_v4_flash_0731``）。
        workspace_root: 工作区根目录；缺省时按本包位置向上定位，不依赖 cwd。

    返回 dict 顶层键:
        schema_version / artifact / model_key / display_model_id /
        workspace_root / inputs / revisions / configs / raw_to_effective_pairs /
        weights / capabilities / baseline_comparison / known_limits

    - ``inputs``: 每个输入文件 {role, path(工作区相对), absolute_path,
      sha256, size_bytes, description}。
    - ``configs.raw_root_config``: 根 config 原样保存（HF 命名）+ HF 消费方。
    - ``configs.constructor_state``: ModelArgs 默认值 + run_config 覆盖，
      逐字段 {value, source, source_path, ...}。
    - ``configs.entrypoint_overrides``: generate.py 的条件覆盖
      （guard 条件/符号/行号），未并入构造态。

    异常: UnknownModelModelError / MissingInputError / InvalidConfigError /
    ConstructorStateError（均为 StructureReaderError 子类，CLI 可据此非零退出）。
    """
    entry = get_model_entry(model_key, workspace_root=workspace_root)
    root = Path(entry["resolved"]["workspace_root"])
    missing = []
    inputs: list[dict] = []
    for item in entry["inputs"]:
        absolute = Path(entry["resolved"][item["role"]])
        if not absolute.is_file():
            missing.append(f"{item['role']}: {absolute}")
            continue
        inputs.append(
            {
                "role": item["role"],
                "path": item["path"],
                "absolute_path": str(absolute),
                "sha256": _sha256_file(absolute),
                "size_bytes": absolute.stat().st_size,
                "description": item["description"],
            }
        )
    if missing:
        raise MissingInputError("缺少注册的输入文件: " + "; ".join(missing))

    inputs_by_role = {item["role"]: item["absolute_path"] for item in inputs}

    # 扫描必需的消费角色；注册缺 role 属坏配置，具名报错而非裸 KeyError。
    _REQUIRED_CONSUMPTION_ROLES = (
        "root_config",
        "run_config",
        "reference_model_source",
        "reference_entrypoint_source",
        "hf_configuration_source",
    )
    absent_roles = [role for role in _REQUIRED_CONSUMPTION_ROLES if role not in inputs_by_role]
    if absent_roles:
        raise InvalidConfigError(
            "注册缺少扫描必需的输入角色: "
            + ", ".join(absent_roles)
            + f"；现有角色: {', '.join(sorted(inputs_by_role)) or '（无）'}"
        )

    manifest_path = Path(entry["resolved"]["model_manifest"]) if "model_manifest" in entry["resolved"] else root / entry["model_manifest"]
    manifest = _load_json_file(manifest_path, "model_manifest")
    manifest_input = {
        "role": "model_manifest",
        "path": entry["model_manifest"],
        "absolute_path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
        "description": "下载清单：声明 revision 与文件列表；不作为现存权重证据",
    }
    inputs.append(manifest_input)
    inputs_by_role["model_manifest"] = str(manifest_path)

    raw_config = _load_json_file(Path(inputs_by_role["root_config"]), "root_config")
    run_config = _load_json_file(Path(inputs_by_role["run_config"]), "run_config")

    constructor = entry["constructor"]
    defaults = _read_dataclass_defaults(
        Path(inputs_by_role["reference_model_source"]), constructor["dataclass_symbol"]
    )
    entrypoint = _scan_entrypoint(
        Path(inputs_by_role["reference_entrypoint_source"]), constructor["dataclass_symbol"]
    )
    hf_config_info = _hf_config_class_info(Path(inputs_by_role["hf_configuration_source"]))
    hf_classes = hf_config_info["classes_declaring_model_type"]
    hf_config_class = hf_classes[0]["class"] if hf_classes else None

    configs = {
        "raw_root_config": _raw_config_section(
            raw_config,
            Path(inputs_by_role["root_config"]),
            hf_config_class,
            inputs_by_role["hf_configuration_source"],
        ),
        "constructor_state": _constructor_state_section(
            defaults,
            run_config,
            Path(inputs_by_role["run_config"]),
            constructor["dataclass_symbol"],
            inputs_by_role["reference_model_source"],
        ),
        "entrypoint_overrides": {
            "source": inputs_by_role["reference_entrypoint_source"],
            **entrypoint,
            "notes": [
                "overrides 未并入 constructor_state；场景求值由后续阶段实现",
                "config 文件路径由调用方经 main(config=...)/CLI --config 指定；"
                "注册表按工程布局固定为 inference/config.json",
            ],
        },
    }

    pairs_report = {"equivalent_pairs": [], "non_equivalent_pairs": []}
    for raw_field, effective_field in entry["raw_to_effective_pairs"]["equivalent_pairs"]:
        raw_value, found = _get_dotted(raw_config, raw_field)
        effective = configs["constructor_state"]["fields"].get(effective_field)
        effective_value = effective["value"] if effective else None
        pairs_report["equivalent_pairs"].append(
            {
                "raw_field": raw_field,
                "effective_field": effective_field,
                "raw_value": raw_value,
                "raw_field_found": found,
                "effective_value": effective_value,
                "match": bool(found and raw_value == effective_value),
            }
        )
    for raw_field, effective_field, note in entry["raw_to_effective_pairs"]["non_equivalent_pairs"]:
        raw_value, found = _get_dotted(raw_config, raw_field)
        effective = configs["constructor_state"]["fields"].get(effective_field)
        effective_value = effective["value"] if effective else None
        pairs_report["non_equivalent_pairs"].append(
            {
                "raw_field": raw_field,
                "effective_field": effective_field,
                "raw_value": raw_value,
                "raw_field_found": found,
                "effective_value": effective_value,
                "note": note,
            }
        )

    weights = _weights_report(root / entry["reference_root"], entry["weights"], manifest)
    capabilities = _capability_report()
    baselines = _compare_source_baselines(root, entry, inputs)

    known_limits = [
        "全部结论来自静态文件读取；模型代码未导入、未执行",
        "根 config 与运行 config 的字段级消费映射只覆盖已审查的 raw↔effective 对照对",
        "声明 revision 与本地内容哈希分别存放；未核对线上一致性",
        "源码基线缺失时比较结果为 missing，missing 不视为审查通过（unreviewed）；"
        "固定基线在 structure_reader/configs/source_baselines 维护，inspect 不自动更新",
        "入口覆盖（interactive/temperature）只记录不合并；场景求值未实现",
    ]

    return {
        "schema_version": "1.0",
        "artifact": {
            "kind": "input_inventory",
            "stage": "S0",
            "note": "S0 输入扫描产物；不是正式模型结构图",
        },
        "model_key": entry["model_key"],
        "display_model_id": entry["display_model_id"],
        "workspace_root": str(root),
        "inputs": inputs,
        "revisions": {
            "model_manifest": {
                "path": entry["model_manifest"],
                "declared_repo_id": manifest.get("repo_id"),
                "declared_revision": manifest.get("revision"),
                "verification": {
                    "online_consistency_checked": False,
                    "note": "声明 revision 与本地文件哈希分别存放；未核对线上一致性",
                },
            },
            "hf_checkout": _git_checkout_info(
                Path(entry["resolved"]["hf_checkout_root_dir"]),
                Path(entry["resolved"]["hf_source_root_dir"]),
            ),
        },
        "configs": configs,
        "raw_to_effective_pairs": pairs_report,
        "weights": weights,
        "capabilities": capabilities,
        "baseline_comparison": baselines,
        "known_limits": known_limits,
    }
