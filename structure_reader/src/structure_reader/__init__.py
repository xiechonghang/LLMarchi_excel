"""structure_reader：模型特征结构阅读工具（开发中，当前仅 S0 输入扫描）。"""

from .model_registry import (
    MODEL_REGISTRY,
    ModelConfigNotFoundError,
    ModelConfigParseError,
    ModelConfigSchemaError,
    StructureReaderError,
    UnknownModelModelError,
    find_workspace_root,
    get_model_entry,
    list_model_keys,
)
from .inventory import (
    ConstructorStateError,
    InvalidConfigError,
    MissingInputError,
    build_inventory,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelConfigNotFoundError",
    "ModelConfigParseError",
    "ModelConfigSchemaError",
    "StructureReaderError",
    "UnknownModelModelError",
    "MissingInputError",
    "InvalidConfigError",
    "ConstructorStateError",
    "find_workspace_root",
    "get_model_entry",
    "list_model_keys",
    "build_inventory",
]

__version__ = "0.1.0"
