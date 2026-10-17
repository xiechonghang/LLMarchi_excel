"""独立测试共用支持：路径定位、隔离工作区构造、期望数据加载。

本模块只使用标准库，不导入产品实现之外的任何运行依赖。
期望值来自 tests/fixtures/expected_inventory_s0.json（由测试方独立读取原始
模型文件后固定），不由被测实现生成。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent  # structure_reader/tests -> workspace root
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
FIXTURES_DIR = TESTS_DIR / "fixtures"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_KEY = "deepseek_v4_flash_0731"
REGISTRY_REL = Path("structure_reader") / "configs" / "models" / f"{MODEL_KEY}.json"

# 注册表声明的 7 个输入角色；扫描产物应另有 model_manifest，共 8 项。
REGISTERED_INPUT_ROLES = (
    "root_config",
    "run_config",
    "reference_model_source",
    "reference_kernel_source",
    "reference_entrypoint_source",
    "hf_modeling_source",
    "hf_configuration_source",
)
MANIFEST_ROLE = "model_manifest"
EXPECTED_INPUT_ROLES = REGISTERED_INPUT_ROLES + (MANIFEST_ROLE,)

REFERENCE_ROOT_REL = Path("models") / "deepseek-ai" / "DeepSeek-V4-Flash-0731"


def load_fixture(name: str = "expected_inventory_s0.json") -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def registry_json_text() -> str:
    return (REPO_ROOT / REGISTRY_REL).read_text(encoding="utf-8")


def registry_json() -> dict:
    return json.loads(registry_json_text())


def _symlink_dir(link: Path, target: Path) -> None:
    if not link.exists():
        link.symlink_to(target, target_is_directory=True)


def assert_outside_repo(path: Path) -> None:
    """防御性检查：任何写/删操作的目标解析后都不得落在真实工作区内。

    独立测试的负例必须只作用于临时副本；一旦发现路径穿过符号链接回到真实
    工作区，立即失败而不是继续执行破坏性操作。
    """
    resolved = Path(path).resolve()
    repo = REPO_ROOT.resolve()
    if resolved == repo or repo in resolved.parents:
        raise AssertionError(
            f"拒绝在真实工作区内执行写/删操作: {path} -> {resolved}"
        )


def safe_unlink(path: Path) -> None:
    assert_outside_repo(path)
    Path(path).unlink()


def safe_write_text(path: Path, text: str) -> None:
    assert_outside_repo(path)
    Path(path).write_text(text, encoding="utf-8")


def make_isolated_workspace(
    dest: Path,
    *,
    display_model_id: str | None = None,
    raw_json_text: str | None = None,
    drop_registry_json: bool = False,
    registry_mutate=None,
) -> Path:
    """构造隔离临时工作区；参考模型输入为**真实副本**，不指向真实文件。

    - 复制（或改写）注册 JSON 到 ``dest/structure_reader/configs/models/``；
    - ``models/deepseek-ai/DeepSeek-V4-Flash-0731`` 整体复制到临时副本，
      因此负例可安全删除/改写其中文件，绝不影响真实输入；
    - ``hflib/`` 以符号链接只读引用（负例不写它）；
    - ``hf_docs/`` 建空目录，使临时目录满足工作区根判定。
    """
    dest = Path(dest).resolve()  # 统一符号链接（macOS /var → /private/var），便于与 CLI 输出比较
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "hf_docs").mkdir(exist_ok=True)
    _symlink_dir(dest / "hflib", REPO_ROOT / "hflib")

    copied_root = dest / "models" / "deepseek-ai" / REFERENCE_ROOT_REL.name
    if not copied_root.exists():
        copied_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            REPO_ROOT / REFERENCE_ROOT_REL,
            copied_root,
            ignore=shutil.ignore_patterns("tokenizer.json", ".DS_Store"),
        )

    registry_dir = dest / REGISTRY_REL.parent
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = dest / REGISTRY_REL
    if drop_registry_json:
        if registry_path.exists():
            registry_path.unlink()
        return dest

    if raw_json_text is not None:
        registry_path.write_text(raw_json_text, encoding="utf-8")
        return dest

    data = registry_json()
    if display_model_id is not None:
        data["display_model_id"] = display_model_id
    if registry_mutate is not None:
        registry_mutate(data)
    registry_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dest


def make_baseline_dir(workspace: Path) -> Path:
    baseline_dir = workspace / "structure_reader" / "configs" / "source_baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    return baseline_dir


# ---------------------------------------------------------------------------
# 隔离工程（CLI 用）：复制产品包与入口，使 outputs/.work 落在临时工程内
# ---------------------------------------------------------------------------

PROJECT_MODULES = ("__init__.py", "cli.py", "inventory.py", "model_registry.py")
PROJECT_SRC_REL = Path("src") / "structure_reader"
PROJECT_ENTRY_REL = Path("scripts") / "read_model_structure.py"
REAL_OUTPUTS_DIR = REPO_ROOT / "structure_reader" / "outputs"


def make_isolated_project(dest: Path) -> Path:
    """复制产品包 + 入口脚本到临时工程，使 CLI 的 outputs/.work 只落在 dest 内。

    CLI 的工程根由 ``cli.py`` 位置（parents[2]）决定，因此只复制
    ``src/structure_reader/`` 与 ``scripts/read_model_structure.py``；
    ``configs`` 与输入由 ``--workspace-root`` 指向的工作区提供。
    """
    dest = Path(dest).resolve()  # 与 CLI 的工程根解析一致（macOS /var → /private/var）
    src_dir = dest / PROJECT_SRC_REL
    src_dir.mkdir(parents=True, exist_ok=True)
    for name in PROJECT_MODULES:
        shutil.copy2(
            REPO_ROOT / "structure_reader" / "src" / "structure_reader" / name,
            src_dir / name,
        )
    entry = dest / PROJECT_ENTRY_REL
    entry.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        REPO_ROOT / "structure_reader" / "scripts" / "read_model_structure.py",
        entry,
    )
    return dest


def run_cli(project: Path, args: list[str], cwd: Path | None = None,
            timeout: int = 300) -> subprocess.CompletedProcess:
    """用当前解释器运行隔离工程内的入口脚本；返回 CompletedProcess。"""
    entry = Path(project) / PROJECT_ENTRY_REL
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, str(entry), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(cwd if cwd is not None else project),
    )


def cli_inventory_root(project: Path, model_key: str = MODEL_KEY) -> Path:
    return Path(project).resolve() / "outputs" / "_inventory" / model_key


def cli_package_dirs(project: Path, model_key: str = MODEL_KEY) -> list[Path]:
    root = cli_inventory_root(project, model_key)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def cli_failed_dirs(project: Path) -> list[Path]:
    root = Path(project).resolve() / "outputs" / "_failed"
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def snapshot_tree(root: Path) -> dict[str, str]:
    """递归记录 root 下所有文件的相对路径 → SHA256；用于确认未被写改。"""
    root = Path(root)
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# 真实工程的“产品产物”区：CLI 只允许写这两个子目录（且必须经隔离工程改写目标）。
# outputs/_coordination/ 是协调与验收记录区，由本轮报告/日志合法写入，不纳入本断言。
REAL_PRODUCT_OUTPUT_DIRS = (
    REPO_ROOT / "structure_reader" / "outputs" / "_inventory",
    REPO_ROOT / "structure_reader" / "outputs" / "_failed",
)


def snapshot_product_outputs() -> dict[str, str]:
    merged: dict[str, str] = {}
    for root in REAL_PRODUCT_OUTPUT_DIRS:
        for rel, digest in snapshot_tree(root).items():
            merged[f"{root.name}/{rel}"] = digest
    return merged


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
