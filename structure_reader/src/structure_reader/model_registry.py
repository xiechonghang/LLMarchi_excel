"""模型注册表：structure_reader/configs/models/<model_key>.json 是唯一模型定义数据源。

注册 JSON 声明输入文件角色/路径（相对工作区根）、目录元数据、构造入口、
weights、审查基线名称与 raw↔effective 对照对；``MODEL_REGISTRY`` 按需从
JSON 加载，本模块不再手写第二份模型数据。注册配置固定位于
``<workspace_root>/structure_reader/configs/models/``（与 source_baselines
同为 structure_reader 包内 configs 布局），键为配置文件名 stem。工作区根
按本文件位置向上定位（同时含有 ``hf_docs/`` 与 ``models/`` 的最近祖先
目录），不依赖调用方 cwd。

本模块只声明事实与入口，不做文件扫描（见 ``inventory.py``）。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "StructureReaderError",
    "UnknownModelModelError",
    "ModelConfigNotFoundError",
    "ModelConfigParseError",
    "ModelConfigSchemaError",
    "MODEL_REGISTRY",
    "find_workspace_root",
    "get_model_entry",
    "get_model_config_path",
    "list_model_keys",
]


class StructureReaderError(Exception):
    """structure_reader 具名异常基类；CLI 可据此统一非零退出。"""


class UnknownModelModelError(StructureReaderError):
    """注册表中不存在请求的模型标识。"""


class ModelConfigNotFoundError(StructureReaderError):
    """模型标识已知，但当前工作区缺少其注册配置 JSON。"""


class ModelConfigParseError(StructureReaderError):
    """注册配置存在但不是合法 JSON。"""


class ModelConfigSchemaError(StructureReaderError):
    """注册配置缺少必要字段或字段类型不符合约定。"""


_REGISTRY_DIRNAME = Path("structure_reader") / "configs" / "models"

_REQUIRED_STR_FIELDS = (
    "model_key",
    "display_model_id",
    "reference_root",
    "model_manifest",
    "hf_checkout_root",
    "hf_source_root",
)

_REQUIRED_CONSTRUCTOR_FIELDS = (
    "dataclass_symbol",
    "dataclass_source",
    "config_override_file",
    "entrypoint_source",
)

_REQUIRED_INPUT_FIELDS = ("role", "path", "description")

_REQUIRED_WEIGHTS_FIELDS = ("safetensors_pattern", "index_file")


def _registry_keys(config_dir: Path) -> list[str]:
    """列出目录下可用的注册键（= 配置文件名 stem）；不解析文件内容。"""
    if not config_dir.is_dir():
        return []
    return sorted(p.stem for p in config_dir.glob("*.json"))


def _load_model_entry(model_key: str, config_path: Path) -> dict:
    """读取并校验单个注册 JSON，返回不含 ``resolved`` 的注册项（每次新解析）。"""
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelConfigNotFoundError(f"无法读取注册配置 {config_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ModelConfigParseError(f"注册配置不是合法 UTF-8 文本: {config_path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelConfigParseError(
            f"注册配置不是合法 JSON: {config_path} 第 {exc.lineno} 行第 {exc.colno} 列: {exc.msg}"
        ) from exc
    _validate_model_config(model_key, data, config_path)
    return data


def _validate_model_config(model_key: str, data: object, config_path: Path) -> None:
    """校验注册 JSON 必要字段；不满足抛 :class:`ModelConfigSchemaError`。"""
    where = f"注册配置 {config_path}"
    if not isinstance(data, dict):
        raise ModelConfigSchemaError(f"{where}: 顶层必须是 JSON 对象")
    for field in _REQUIRED_STR_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ModelConfigSchemaError(
                f"{where}: 缺少或不合法的必要字段 {field!r}（需非空字符串）"
            )
    if data["model_key"] != model_key:
        raise ModelConfigSchemaError(
            f"{where}: model_key={data['model_key']!r} 与注册键 {model_key!r}（文件名）不一致"
        )
    constructor = data.get("constructor")
    if not isinstance(constructor, dict):
        raise ModelConfigSchemaError(f"{where}: constructor 必须是对象")
    for field in _REQUIRED_CONSTRUCTOR_FIELDS:
        value = constructor.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ModelConfigSchemaError(f"{where}: constructor.{field} 需非空字符串")
    inputs = data.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ModelConfigSchemaError(f"{where}: inputs 必须是非空数组")
    roles: list[str] = []
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise ModelConfigSchemaError(f"{where}: inputs[{index}] 必须是对象")
        for field in _REQUIRED_INPUT_FIELDS:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ModelConfigSchemaError(f"{where}: inputs[{index}].{field} 需非空字符串")
        roles.append(item["role"])
    duplicated = sorted({role for role in roles if roles.count(role) > 1})
    if duplicated:
        raise ModelConfigSchemaError(f"{where}: inputs 存在重复 role: {', '.join(duplicated)}")
    weights = data.get("weights")
    if not isinstance(weights, dict):
        raise ModelConfigSchemaError(f"{where}: weights 必须是对象")
    for field in _REQUIRED_WEIGHTS_FIELDS:
        value = weights.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ModelConfigSchemaError(f"{where}: weights.{field} 需非空字符串")
    baselines = data.get("source_baselines")
    if not isinstance(baselines, list) or not all(
        isinstance(name, str) and name.strip() for name in baselines
    ):
        raise ModelConfigSchemaError(f"{where}: source_baselines 必须是非空字符串数组（可为空数组）")
    pairs = data.get("raw_to_effective_pairs")
    if not isinstance(pairs, dict):
        raise ModelConfigSchemaError(f"{where}: raw_to_effective_pairs 必须是对象")
    equivalent = pairs.get("equivalent_pairs")
    if not isinstance(equivalent, list):
        raise ModelConfigSchemaError(f"{where}: raw_to_effective_pairs.equivalent_pairs 必须是数组")
    for index, pair in enumerate(equivalent):
        if not (
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(isinstance(field, str) and field.strip() for field in pair)
        ):
            raise ModelConfigSchemaError(
                f"{where}: equivalent_pairs[{index}] 必须是 [raw_field, effective_field] 两个字符串"
            )
    non_equivalent = pairs.get("non_equivalent_pairs")
    if not isinstance(non_equivalent, list):
        raise ModelConfigSchemaError(
            f"{where}: raw_to_effective_pairs.non_equivalent_pairs 必须是数组"
        )
    for index, pair in enumerate(non_equivalent):
        if not (
            isinstance(pair, (list, tuple))
            and len(pair) == 3
            and all(isinstance(field, str) and field.strip() for field in pair)
        ):
            raise ModelConfigSchemaError(
                f"{where}: non_equivalent_pairs[{index}] "
                "必须是 [raw_field, effective_field, note] 三个字符串"
            )


class _JsonModelRegistry(Mapping):
    """按需从注册 JSON 加载的只读模型注册视图；键为配置文件名 stem。

    首次访问时定位默认工作区根，之后按需解析单个 JSON；不在导入时读盘，
    也不为键列表解析未请求的模型配置。
    """

    def __init__(self) -> None:
        self._config_dir: Path | None = None

    def _get_config_dir(self) -> Path:
        if self._config_dir is None:
            self._config_dir = find_workspace_root() / _REGISTRY_DIRNAME
        return self._config_dir

    def _available_keys(self) -> list[str]:
        return _registry_keys(self._get_config_dir())

    def __iter__(self):
        return iter(self._available_keys())

    def __len__(self) -> int:
        return len(self._available_keys())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._available_keys()

    def __getitem__(self, key: str):
        if not isinstance(key, str) or key not in self:
            raise KeyError(key)
        return _load_model_entry(key, self._get_config_dir() / f"{key}.json")

    def __repr__(self) -> str:
        return f"_JsonModelRegistry(config_dir={self._get_config_dir()})"


MODEL_REGISTRY = _JsonModelRegistry()


def find_workspace_root(start: str | Path | None = None) -> Path:
    """向上定位工作区根目录（同时含 ``hf_docs/`` 与 ``models/`` 的最近祖先）。

    ``start`` 缺省时从本文件位置向上查找，因此结果不依赖调用方 cwd。
    找不到时抛出 :class:`StructureReaderError`。
    """
    if start is None:
        current = Path(__file__).resolve()
    else:
        current = Path(start).resolve()
    if current.is_dir():
        candidates = [current, *current.parents]
    else:
        candidates = list(current.parents)
    for candidate in candidates:
        if (candidate / "hf_docs").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise StructureReaderError(
        "无法定位工作区根目录（需要同时包含 hf_docs/ 与 models/ 的目录）；"
        "可显式传入 workspace_root 参数"
    )


def list_model_keys(workspace_root: str | Path | None = None) -> list[str]:
    """返回已注册模型标识（= 该工作区 ``structure_reader/configs/models/`` 下注册 JSON 的文件名 stem）。

    ``workspace_root`` 缺省时定位默认工作区根；结果不依赖调用方 cwd。
    """
    if workspace_root is None:
        return list(MODEL_REGISTRY)
    return _registry_keys(Path(workspace_root) / _REGISTRY_DIRNAME)


def get_model_entry(model_key: str, workspace_root: str | Path | None = None) -> dict:
    """取模型注册项：实际读取该工作区 ``structure_reader/configs/models/<model_key>.json`` 并解析为绝对路径。

    ``workspace_root`` 缺省时按本文件位置向上定位工作区根，不依赖 cwd。
    异常：未知模型抛 :class:`UnknownModelModelError`；模型已知但该工作区缺少
    注册 JSON 抛 :class:`ModelConfigNotFoundError`；JSON 损坏抛
    :class:`ModelConfigParseError`；必要字段不合法抛
    :class:`ModelConfigSchemaError`（后三者均为 :class:`StructureReaderError` 子类）。

    返回 dict 为注册项的每次新解析副本，附加 ``resolved`` 段：
    ``{role: 绝对路径}``、目录键（``reference_root_dir``/``hf_checkout_root_dir``/
    ``hf_source_root_dir``/``model_manifest``）与 ``workspace_root``。
    """
    if not isinstance(model_key, str) or not model_key:
        raise UnknownModelModelError(f"未知模型 {model_key!r}；模型标识必须是非空字符串")
    root = Path(workspace_root) if workspace_root is not None else find_workspace_root()
    config_dir = root / _REGISTRY_DIRNAME
    config_path = config_dir / f"{model_key}.json"
    if not config_path.is_file():
        available = _registry_keys(config_dir)
        # 仅用于错误分类：默认注册表不可定位时按未知模型报告，不影响正常路径。
        try:
            known = model_key in MODEL_REGISTRY
        except StructureReaderError:
            known = False
        if known:
            raise ModelConfigNotFoundError(
                f"模型 {model_key!r} 已注册，但该工作区缺少注册配置: {config_path}"
            )
        raise UnknownModelModelError(
            f"未知模型 {model_key!r}；该工作区可用模型: "
            f"{', '.join(available) if available else '（无注册配置）'}"
        )
    entry = _load_model_entry(model_key, config_path)
    resolved = {item["role"]: str(root / item["path"]) for item in entry["inputs"]}
    resolved["reference_root_dir"] = str(root / entry["reference_root"])
    resolved["hf_checkout_root_dir"] = str(root / entry["hf_checkout_root"])
    resolved["hf_source_root_dir"] = str(root / entry["hf_source_root"])
    resolved["model_manifest"] = str(root / entry["model_manifest"])
    resolved["workspace_root"] = str(root)
    entry["resolved"] = resolved
    return entry


def get_model_config_path(model_key: str, workspace_root: str | Path | None = None) -> Path:
    """返回该模型注册 JSON 的路径（不读取内容）；未知模型抛 :class:`UnknownModelModelError`。

    仅做存在性与归属判断，不解析 JSON，因此坏 JSON/缺字段的诊断仍由
    :func:`get_model_entry`/扫描库负责。
    """
    if not isinstance(model_key, str) or not model_key:
        raise UnknownModelModelError(f"未知模型 {model_key!r}；模型标识必须是非空字符串")
    root = Path(workspace_root) if workspace_root is not None else find_workspace_root()
    config_path = root / _REGISTRY_DIRNAME / f"{model_key}.json"
    if not config_path.is_file():
        raise UnknownModelModelError(f"未知模型 {model_key!r}；缺少注册配置: {config_path}")
    return config_path
