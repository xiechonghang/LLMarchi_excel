"""结构阅读器 S0：CLI ``inspect`` 独立测试。

覆盖 v3 S0 验收中 CLI 相关条款：
- 帮助只列实际命令；参数错误退出码 2；具名错误退出码 1 不裸 traceback；
- inspect 真正调用扫描库；产物唯一为 outputs/_inventory/<key>/<content_id>/ 四文件，
  不生成后续阶段空占位；
- 配置来源（raw/constructor/entrypoint）与能力探测**直接持久化在扫描包内**可读；
- content_id 为完整 SHA256、内容寻址、同输入稳定、异输入变化、不含绝对路径；
- 同输入重跑幂等复用；既存内容被篡改时拒绝覆盖并留具名失败记录。

安全约束（本文件的所有负例）：
- 只写临时副本：CLI 在**隔离工程**（复制 src/ + scripts/）内运行，outputs/.work
  只落在临时工程；工作区为真实工作区（只读）或真实临时副本。
- 所有写/删前用 ``_support.assert_outside_repo`` 校验解析后路径，绝不改真实输入/真实产物。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support  # noqa: E402
from _support import (  # noqa: E402
    MODEL_KEY,
    REFERENCE_ROOT_REL,
    REPO_ROOT,
    cli_failed_dirs,
    cli_inventory_root,
    cli_package_dirs,
    load_json,
    make_baseline_dir,
    make_isolated_project,
    make_isolated_workspace,
    run_cli,
)

REAL_WORKSPACE = REPO_ROOT
REAL_BASELINES = REPO_ROOT / "structure_reader" / "configs" / "source_baselines"
CONTENT_FILES = ("README.md", "capability_report.json", "input_manifest.json", "run_manifest.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_args(workspace: Path, extra: list[str] | None = None) -> list[str]:
    args = ["inspect", "--model", MODEL_KEY, "--workspace-root", str(workspace)]
    if extra:
        args.extend(extra)
    return args


def make_workspace_with_baselines(dest: Path) -> Path:
    workspace = make_isolated_workspace(dest)
    baseline_dir = make_baseline_dir(workspace)
    for source in sorted(REAL_BASELINES.glob("*.json")):
        shutil.copy2(source, baseline_dir / source.name)
    return workspace


class CliSuccessMixin:
    """在隔离工程内对真实工作区（只读）执行一次成功的 inspect。"""

    tmp: tempfile.TemporaryDirectory
    project: Path
    workspace: Path
    result: object
    payload: dict
    package: Path

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.project = make_isolated_project(root / "proj")
        cls.workspace = REAL_WORKSPACE
        cls.real_outputs_before = _support.snapshot_product_outputs()
        cls.result = run_cli(cls.project, inspect_args(cls.workspace, ["--json"]))
        cls.payload = json.loads(cls.result.stdout)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class CliArgparseTest(unittest.TestCase):
    """参数错误与入口行为（不产生任何产物）。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.project = make_isolated_project(Path(cls.tmp.name) / "proj")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_help_lists_only_inspect_and_exits_zero(self):
        result = run_cli(self.project, ["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        text = result.stdout
        self.assertIn("inspect", text)
        # 只列实际命令：不得把后续阶段命令暴露为可用子命令
        for absent in ("validate", "render", "export-tables", "graph", "extract"):
            self.assertNotRegex(
                text, rf"(?m)^\s+{re.escape(absent)}\b", f"帮助不应列出未实现命令 {absent}"
            )

    def test_subcommand_help_exits_zero(self):
        result = run_cli(self.project, ["inspect", "--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", result.stdout)
        self.assertIn("--workspace-root", result.stdout)

    def test_no_command_exits_2(self):
        self.assertEqual(run_cli(self.project, []).returncode, 2)

    def test_unknown_command_exits_2(self):
        self.assertEqual(run_cli(self.project, ["bogus"]).returncode, 2)

    def test_inspect_without_model_exits_2(self):
        self.assertEqual(run_cli(self.project, ["inspect"]).returncode, 2)

    def test_inspect_model_without_value_exits_2(self):
        self.assertEqual(run_cli(self.project, ["inspect", "--model"]).returncode, 2)

    def test_argparse_errors_create_no_output(self):
        run_cli(self.project, ["inspect"])
        run_cli(self.project, [])
        self.assertEqual(cli_package_dirs(self.project), [])
        self.assertEqual(cli_failed_dirs(self.project), [])

    def test_entry_is_cwd_independent(self):
        with tempfile.TemporaryDirectory() as other_cwd:
            result = run_cli(
                self.project, inspect_args(REAL_WORKSPACE, ["--json"]), cwd=Path(other_cwd)
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "success")
        # 产物仍落在隔离工程内，而不是调用方 cwd
        self.assertTrue(str(payload["output_dir"]).startswith(str(self.project)))


class CliInspectPackageTest(CliSuccessMixin, unittest.TestCase):
    """成功扫描包的结构、内容与确定性。"""

    def test_exit_zero_and_status_success(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)
        self.assertEqual(self.payload["status"], "success")
        self.assertTrue(self.payload["reviewed"])
        self.assertEqual(self.payload["model_key"], MODEL_KEY)

    def test_outputs_only_inside_isolated_project(self):
        dirs = cli_package_dirs(self.project)
        self.assertEqual(len(dirs), 1)
        self.assertTrue(str(dirs[0]).startswith(str(self.project)))
        self.assertTrue(str(self.payload["output_dir"]).startswith(str(self.project)))

    def test_real_project_outputs_untouched(self):
        self.assertEqual(
            _support.snapshot_product_outputs(),
            self.real_outputs_before,
            "CLI 不得写入真实工程的 outputs/_inventory 或 _failed（隔离工程才能写）",
        )

    def test_package_has_exactly_the_four_s0_files(self):
        dirs = cli_package_dirs(self.project)
        self.assertEqual(len(dirs), 1)
        package = dirs[0]
        names = sorted(p.name for p in package.iterdir())
        self.assertEqual(names, sorted(CONTENT_FILES))
        # 不得为后续阶段预留空占位目录
        for later in ("html", "csv", "graph", "tables", "figures"):
            self.assertFalse((package / later).exists(), later)

    def test_content_id_is_full_sha256_and_matches_dir_and_manifest(self):
        content_id = self.payload["content_id"]
        self.assertRegex(content_id, SHA256_RE)
        package = Path(self.payload["output_dir"])
        self.assertEqual(package.name, content_id)
        manifest = load_json(package / "run_manifest.json")
        self.assertEqual(manifest["content_id"], content_id)

    def test_run_manifest_file_hashes_match_actual_bytes(self):
        package = Path(self.payload["output_dir"])
        manifest = load_json(package / "run_manifest.json")
        listed = {item["name"]: item["sha256"] for item in manifest["files"]}
        self.assertEqual(sorted(listed), sorted(CONTENT_FILES))
        for name in CONTENT_FILES:
            actual = sha256_file(package / name)
            if name == "run_manifest.json":
                self.assertIsNone(listed[name], "自指文件不应登记自身哈希")
            else:
                self.assertEqual(listed[name], actual, name)

    def test_input_manifest_persists_raw_constructor_and_entrypoint(self):
        package = Path(self.payload["output_dir"])
        manifest = load_json(package / "input_manifest.json")
        configs = manifest["configs"]
        # raw 根配置：HF 命名原样
        raw = configs["raw_root_config"]
        self.assertEqual(raw["fields"]["hidden_size"], 4096)
        self.assertEqual(raw["fields"]["num_hidden_layers"], 43)
        self.assertNotIn("inference/config.json", raw["path"])
        # 构造态：run_config 覆盖 + dataclass 默认值来源
        fields = configs["constructor_state"]["fields"]
        for name, value in (("n_layers", 43), ("n_mtp_layers", 3),
                            ("n_hash_layers", 3), ("hc_mult", 4)):
            self.assertEqual(fields[name]["value"], value, name)
            self.assertEqual(fields[name]["source"], "run_config", name)
            self.assertFalse(str(fields[name]["source_path"]).startswith("/"), name)
        self.assertEqual(fields["max_seq_len"]["value"], 4096)
        self.assertEqual(fields["max_seq_len"]["source"], "dataclass_default")
        self.assertEqual(fields["max_batch_size"]["value"], 4)
        self.assertEqual(fields["max_batch_size"]["source"], "dataclass_default")
        # 入口覆盖：只记录、不并入构造态
        overrides = configs["entrypoint_overrides"]["overrides"]
        by_field = {item["field"]: item for item in overrides}
        self.assertEqual(by_field["max_seq_len"]["value"], 65536)
        self.assertEqual(by_field["max_batch_size"]["value"], 1)
        for name in ("max_seq_len", "max_batch_size"):
            guards = " ".join(g["condition_source"] for g in by_field[name]["guards"])
            self.assertIn("interactive", guards, name)
        self.assertFalse(by_field["temperature"]["is_literal"])
        self.assertIn("temperature", by_field["temperature"]["expression"])

    def test_input_manifest_persists_pair_conclusions(self):
        package = Path(self.payload["output_dir"])
        manifest = load_json(package / "input_manifest.json")
        pairs = manifest["raw_to_effective_pairs"]
        by_effective = {p["effective_field"]: p for p in pairs["equivalent_pairs"]}
        self.assertEqual(by_effective["dim"]["raw_value"], 4096)
        self.assertTrue(by_effective["dim"]["match"])
        non = {p["effective_field"]: p for p in pairs["non_equivalent_pairs"]}
        self.assertEqual(non["n_mtp_layers"]["raw_value"], 1)
        self.assertEqual(non["n_mtp_layers"]["effective_value"], 3)

    def test_capability_report_has_actual_probe_not_placeholder(self):
        package = Path(self.payload["output_dir"])
        report = load_json(package / "capability_report.json")
        self.assertEqual(report["schema"], "capability_report_v2")
        self.assertTrue(report["core_dependencies"]["stdlib_only"])
        self.assertFalse(report["core_dependencies"]["torch_required"])
        # 实际探测器结果：至少包含四个被探测包且含 available 布尔值
        self.assertEqual(
            sorted(report["optional_packages"]),
            ["openpyxl", "safetensors", "torch", "transformers"],
        )
        for name, spec in report["optional_packages"].items():
            self.assertIn("available", spec, name)
        self.assertTrue(report["implemented_capabilities"])
        self.assertIn("python_version", report["interpreter"])

    def test_artifacts_contain_no_absolute_paths(self):
        package = Path(self.payload["output_dir"])
        forbidden = {str(REPO_ROOT), str(self.project), sys.executable}
        for name in CONTENT_FILES:
            text = (package / name).read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, f"{name} 不应含绝对路径片段 {token!r}")

    def test_readme_marks_stage_not_structure_graph(self):
        package = Path(self.payload["output_dir"])
        readme = (package / "README.md").read_text(encoding="utf-8")
        self.assertIn("不是模型结构图", readme)
        self.assertIn("S0", readme)
        self.assertIn("内容 ID", readme)

    def test_package_json_files_are_valid_and_navigation_complete(self):
        package = Path(self.payload["output_dir"])
        for name in ("run_manifest.json", "input_manifest.json", "capability_report.json"):
            document = load_json(package / name)
            self.assertIsInstance(document, dict, name)
            self.assertIn("schema", document, name)
        readme = (package / "README.md").read_text(encoding="utf-8")
        for sibling in ("run_manifest.json", "input_manifest.json", "capability_report.json"):
            self.assertIn(sibling, readme, f"README 导航应指向 {sibling}")

    def test_rerun_is_idempotent_reuse(self):
        second = run_cli(self.project, inspect_args(self.workspace, ["--json"]))
        self.assertEqual(second.returncode, 0, second.stderr)
        payload = json.loads(second.stdout)
        self.assertTrue(payload["reused"])
        self.assertEqual(payload["content_id"], self.payload["content_id"])
        self.assertEqual(cli_package_dirs(self.project), [Path(self.payload["output_dir"])])
        self.assertEqual(cli_failed_dirs(self.project), [])

    def test_text_output_reports_status_and_files(self):
        result = run_cli(self.project, inspect_args(self.workspace))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status: success", result.stdout)
        for name in CONTENT_FILES:
            self.assertIn(name, result.stdout)


class CliUnreviewedTest(unittest.TestCase):
    """缺基线的明确降级：unreviewed，可出诊断包但不标通过。"""

    def test_missing_baselines_yield_unreviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_isolated_project(root / "proj")
            workspace = make_isolated_workspace(root / "ws")  # 明确无 source_baselines
            result = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "unreviewed")
            self.assertFalse(payload["reviewed"])
            self.assertTrue(
                all(v == "missing" for v in payload["baseline_statuses"].values())
            )
            package = Path(payload["output_dir"])
            self.assertTrue(package.is_dir())
            readme = (package / "README.md").read_text(encoding="utf-8")
            self.assertIn("unreviewed", readme)
            self.assertIn("不表示通过", readme)
            manifest = load_json(package / "run_manifest.json")
            self.assertEqual(manifest["status"], "unreviewed")
            self.assertFalse(manifest["reviewed"])


class CliContentIdTest(unittest.TestCase):
    """content_id 的内容寻址性质。"""

    def test_same_inputs_same_content_id_across_workspace_paths(self):
        ids = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(2):
                project = make_isolated_project(root / f"proj{index}")
                workspace = make_workspace_with_baselines(root / f"ws{index}")
                result = run_cli(project, inspect_args(workspace, ["--json"]))
                self.assertEqual(result.returncode, 0, result.stderr)
                ids.append(json.loads(result.stdout)["content_id"])
        self.assertRegex(ids[0], SHA256_RE)
        self.assertEqual(ids[0], ids[1], "同一输入在不同绝对路径工作区应得同一 content_id")

    def test_changed_input_changes_content_id(self):
        ids = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(2):
                project = make_isolated_project(root / f"proj{index}")
                workspace = make_isolated_workspace(root / f"ws{index}")
                if index == 1:
                    target = workspace / REFERENCE_ROOT_REL / "inference" / "config.json"
                    _support.assert_outside_repo(target)
                    target.write_text(target.read_text(encoding="utf-8") + "\n",
                                      encoding="utf-8")
                result = run_cli(project, inspect_args(workspace, ["--json"]))
                self.assertEqual(result.returncode, 0, result.stderr)
                ids.append(json.loads(result.stdout)["content_id"])
        self.assertNotEqual(ids[0], ids[1], "输入字节变化必须改变 content_id")


class CliConsistencyAndFailureTest(unittest.TestCase):
    """拒绝覆盖与具名失败记录；全部在隔离工程/临时副本内。"""

    def test_tampered_package_refused_and_failed_record_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_isolated_project(root / "proj")
            workspace = make_workspace_with_baselines(root / "ws")
            first = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(first.returncode, 0, first.stderr)
            package = Path(json.loads(first.stdout)["output_dir"])

            target = package / "README.md"
            _support.assert_outside_repo(target)
            target.write_text("tampered content", encoding="utf-8")

            second = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(second.returncode, 1, second.stdout)
            payload = json.loads(second.stdout)
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error"]["kind"], "output_consistency_conflict")

            # 拒绝覆盖：被篡改文件保持原样，且没有产生第二个目录
            self.assertEqual(target.read_text(encoding="utf-8"), "tampered content")
            self.assertEqual(cli_package_dirs(project), [package])
            failed = cli_failed_dirs(project)
            self.assertEqual(len(failed), 1)
            record = load_json(failed[0] / "run_manifest.json")
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["model_key"], MODEL_KEY)
            self.assertIn("output_consistency_conflict", json.dumps(record, ensure_ascii=False))

    def test_baseline_mismatch_exits_1_without_success_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_isolated_project(root / "proj")
            workspace = make_isolated_workspace(root / "ws")
            baseline_dir = make_baseline_dir(workspace)
            for name in ("deepseek_v4_flash_0731_reference",
                         "deepseek_v4_flash_0731_hf_comparison"):
                (baseline_dir / f"{name}.json").write_text(
                    json.dumps({"source_files": [
                        {"path": f"{REFERENCE_ROOT_REL.as_posix()}/config.json",
                         "sha256": "0" * 64}]}),
                    encoding="utf-8",
                )
            result = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(result.returncode, 1, result.stdout)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["error"]["kind"], "baseline_mismatch")
            self.assertEqual(cli_package_dirs(project), [])
            self.assertEqual(len(cli_failed_dirs(project)), 1)

    def test_modified_input_against_fixed_baseline_is_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_isolated_project(root / "proj")
            workspace = make_workspace_with_baselines(root / "ws")
            target = workspace / REFERENCE_ROOT_REL / "inference" / "config.json"
            _support.assert_outside_repo(target)
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            result = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertEqual(json.loads(result.stdout)["error"]["kind"], "baseline_mismatch")

    def test_unknown_model_exits_1_with_named_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_isolated_project(root / "proj")
            workspace = make_isolated_workspace(root / "ws")
            result = run_cli(
                project,
                ["inspect", "--model", "no_such_model",
                 "--workspace-root", str(workspace), "--json"],
            )
            self.assertEqual(result.returncode, 1, result.stdout)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "failed")
            kind = payload["error"]["kind"]
            self.assertTrue(kind.endswith("Error"), kind)
            self.assertEqual(cli_package_dirs(project, "no_such_model"), [])
            self.assertEqual(len(cli_failed_dirs(project)), 1)
            self.assertNotIn("Traceback", result.stderr)

    def test_bad_registry_json_exits_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_isolated_project(root / "proj")
            workspace = make_isolated_workspace(root / "ws")
            registry = workspace / _support.REGISTRY_REL
            _support.safe_write_text(registry, "{ this is not json")
            result = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertEqual(json.loads(result.stdout)["error"]["kind"], "ModelConfigParseError")

    def test_missing_input_exits_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_isolated_project(root / "proj")
            workspace = make_isolated_workspace(root / "ws")
            target = workspace / REFERENCE_ROOT_REL / "inference" / "config.json"
            _support.safe_unlink(target)
            result = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertEqual(json.loads(result.stdout)["error"]["kind"], "MissingInputError")

    def test_failed_run_never_writes_real_project(self):
        real_outputs_before = _support.snapshot_product_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_isolated_project(root / "proj")
            workspace = make_isolated_workspace(root / "ws")
            result = run_cli(
                project,
                ["inspect", "--model", "no_such_model",
                 "--workspace-root", str(workspace)],
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(len(cli_failed_dirs(project)), 1)
        self.assertEqual(
            _support.snapshot_product_outputs(), real_outputs_before,
            "失败记录也不得写入真实工程 outputs/_inventory 或 _failed",
        )


class CliSafetyTest(unittest.TestCase):
    """隔离前提的回归护栏：负例只作用于真实临时副本。"""

    def test_isolated_workspace_models_is_real_copy_not_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(Path(tmp) / "ws")
            models = workspace / "models"
            self.assertTrue(models.is_dir())
            self.assertFalse(models.is_symlink(), "models 必须是实体副本，不能是符号链接")
            real_config = REAL_WORKSPACE / REFERENCE_ROOT_REL / "inference" / "config.json"
            copy_config = workspace / REFERENCE_ROOT_REL / "inference" / "config.json"
            self.assertNotEqual(real_config.resolve(), copy_config.resolve())

    def test_guard_rejects_real_paths_and_allows_temp(self):
        real_config = REAL_WORKSPACE / REFERENCE_ROOT_REL / "inference" / "config.json"
        with self.assertRaises(AssertionError):
            _support.assert_outside_repo(real_config)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(Path(tmp) / "ws")
            _support.assert_outside_repo(workspace / REFERENCE_ROOT_REL / "inference" / "config.json")

    def test_real_workspace_inputs_not_modified_by_cli_success_run(self):
        tracked = [
            REAL_WORKSPACE / REFERENCE_ROOT_REL / "config.json",
            REAL_WORKSPACE / REFERENCE_ROOT_REL / "inference" / "config.json",
            REAL_WORKSPACE / REFERENCE_ROOT_REL / "inference" / "model.py",
            REAL_WORKSPACE / REFERENCE_ROOT_REL / "inference" / "generate.py",
            REAL_WORKSPACE / "structure_reader" / "configs" / "models" / f"{MODEL_KEY}.json",
        ]
        before = {str(p): sha256_file(p) for p in tracked}
        with tempfile.TemporaryDirectory() as tmp:
            project = make_isolated_project(Path(tmp) / "proj")
            result = run_cli(project, inspect_args(REAL_WORKSPACE, ["--json"]))
            self.assertIn(result.returncode, (0, 1))
        after = {str(p): sha256_file(p) for p in tracked}
        self.assertEqual(before, after, "真实输入/注册不得被 CLI 改动")


# ---------------------------------------------------------------------------
# 定版补充（verification_task_03_s0_complete）：
# 未实现命令、缺消费 role、坏/非法基线 schema、基线不自建自更新、
# content_id 构成（注册 / 基线 / 工具源码）、低 Python 版本守卫、真实工作区 inspect。
# ---------------------------------------------------------------------------

REQUIRED_CONSUMPTION_ROLES = (
    "root_config",
    "run_config",
    "reference_model_source",
    "reference_entrypoint_source",
    "hf_configuration_source",
)


def build_scenario(root: Path, name: str, *, baselines: str = "none",
                   registry_mutate=None, baseline_writer=None, tool_patch: bool = False):
    """构造隔离工程 + 隔离工作区；可选写入基线/改注册/改工具源码。

    ``baselines``: "none"（不写基线）/ "real"（复制真实固定基线）。
    ``baseline_writer``: 接收基线目录路径，自行写入（用于坏 schema 负例）。
    全部写操作都在临时副本内，并先经 ``assert_outside_repo``。
    """
    project = make_isolated_project(root / f"{name}-proj")
    workspace = make_isolated_workspace(root / f"{name}-ws")
    if baselines == "real":
        baseline_dir = make_baseline_dir(workspace)
        for source in sorted(REAL_BASELINES.glob("*.json")):
            shutil.copy2(source, baseline_dir / source.name)
    if baseline_writer is not None:
        baseline_writer(make_baseline_dir(workspace))
    if registry_mutate is not None:
        registry = workspace / _support.REGISTRY_REL
        _support.assert_outside_repo(registry)
        data = json.loads(registry.read_text(encoding="utf-8"))
        registry_mutate(data)
        registry.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if tool_patch:
        source = project / _support.PROJECT_SRC_REL / "inventory.py"
        _support.assert_outside_repo(source)
        source.write_text(
            source.read_text(encoding="utf-8") + "\n# harness independent probe\n",
            encoding="utf-8",
        )
    return project, workspace


class CliUnimplementedCommandTest(unittest.TestCase):
    """未实现命令必须非零退出，且不产生任何成功模型图/扫描包。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.project = make_isolated_project(Path(cls.tmp.name) / "proj")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_later_stage_commands_exit_2_without_artifacts(self):
        for command in ("build", "validate", "render", "export-tables"):
            result = run_cli(self.project, [command])
            self.assertEqual(result.returncode, 2, f"{command} 应退出 2")
            self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(cli_package_dirs(self.project), [])
        self.assertEqual(cli_failed_dirs(self.project), [])

    def test_help_does_not_advertise_unimplemented_commands(self):
        # V1 起 build 已实现并列入帮助；仍未实现的独立子命令不得出现
        help_text = run_cli(self.project, ["--help"]).stdout
        self.assertRegex(help_text, r"(?m)^\s+build\b")
        for command in ("validate", "render", "export-tables"):
            self.assertNotRegex(help_text, rf"(?m)^\s+{re.escape(command)}\b", command)


class CliMissingConsumptionRoleTest(unittest.TestCase):
    """注册缺必需消费 role → 具名 InvalidConfigError，非裸 KeyError。"""

    def test_each_required_role_missing_is_named_error(self):
        for role in REQUIRED_CONSUMPTION_ROLES:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)

                def drop_role(data, role=role):
                    data["inputs"] = [i for i in data["inputs"] if i["role"] != role]

                project, workspace = build_scenario(
                    root, "drop", registry_mutate=drop_role
                )
                result = run_cli(project, inspect_args(workspace, ["--json"]))
                self.assertEqual(result.returncode, 1, result.stdout)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["error"]["kind"], "InvalidConfigError", role)
                self.assertIn(role, payload["error"]["message"])
                self.assertEqual(cli_package_dirs(project), [], role)
                self.assertEqual(len(cli_failed_dirs(project)), 1, role)

    def test_keyerror_never_leaks_to_stdout(self):
        def drop_role(data):
            data["inputs"] = [i for i in data["inputs"] if i["role"] != "run_config"]

        with tempfile.TemporaryDirectory() as tmp:
            project, workspace = build_scenario(Path(tmp), "nr", registry_mutate=drop_role)
            result = run_cli(project, inspect_args(workspace))
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("KeyError", result.stdout)
            self.assertNotIn("Traceback", result.stdout)


class CliBaselineSchemaTest(unittest.TestCase):
    """空/坏 schema 基线 → failed；坏 JSON → 具名 InvalidConfigError。"""

    REFERENCE_NAME = "deepseek_v4_flash_0731_reference"

    def _run_with_baseline_text(self, text):
        def writer(baseline_dir):
            (baseline_dir / f"{self.REFERENCE_NAME}.json").write_text(text, encoding="utf-8")

        tmp = tempfile.TemporaryDirectory()
        project, workspace = build_scenario(Path(tmp.name), "bl", baseline_writer=writer)
        result = run_cli(project, inspect_args(workspace, ["--json"]))
        return tmp, project, workspace, result

    def test_empty_object_baseline_is_schema_unrecognized(self):
        tmp, project, _, result = self._run_with_baseline_text("{}")
        try:
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertEqual(
                json.loads(result.stdout)["error"]["kind"], "baseline_schema_unrecognized"
            )
            self.assertEqual(cli_package_dirs(project), [])
            self.assertEqual(len(cli_failed_dirs(project)), 1)
        finally:
            tmp.cleanup()

    def test_non_object_top_level_baseline_is_schema_unrecognized(self):
        tmp, project, _, result = self._run_with_baseline_text("[]")
        try:
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertEqual(
                json.loads(result.stdout)["error"]["kind"], "baseline_schema_unrecognized"
            )
        finally:
            tmp.cleanup()

    def test_empty_source_files_baseline_is_schema_unrecognized(self):
        tmp, project, _, result = self._run_with_baseline_text(
            json.dumps({"baseline_name": self.REFERENCE_NAME, "source_files": []})
        )
        try:
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertEqual(
                json.loads(result.stdout)["error"]["kind"], "baseline_schema_unrecognized"
            )
        finally:
            tmp.cleanup()

    def test_malformed_baseline_json_is_named_error(self):
        tmp, project, _, result = self._run_with_baseline_text("{ not json")
        try:
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertEqual(
                json.loads(result.stdout)["error"]["kind"], "InvalidConfigError"
            )
        finally:
            tmp.cleanup()

    def test_failed_baseline_leaves_no_apparently_successful_package(self):
        tmp, project, _, result = self._run_with_baseline_text("{}")
        try:
            self.assertEqual(result.returncode, 1)
            self.assertEqual(cli_package_dirs(project), [],
                             "非法基线不得遗留貌似成功的扫描包/图")
            record = load_json(cli_failed_dirs(project)[0] / "run_manifest.json")
            self.assertEqual(record["status"], "failed")
            self.assertIn("baseline", json.dumps(record, ensure_ascii=False))
        finally:
            tmp.cleanup()


class CliBaselineImmutabilityTest(unittest.TestCase):
    """inspect 不自动创建、不自动更新基线。"""

    def test_missing_baselines_are_not_auto_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, workspace = build_scenario(Path(tmp), "nb", baselines="none")
            baseline_dir = workspace / "structure_reader" / "configs" / "source_baselines"
            self.assertFalse(baseline_dir.exists())
            result = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "unreviewed")
            self.assertFalse(baseline_dir.exists(), "缺基线时 inspect 不得自动创建基线")

    def test_existing_baselines_are_not_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, workspace = build_scenario(Path(tmp), "ub", baselines="real")
            baseline_dir = workspace / "structure_reader" / "configs" / "source_baselines"
            before = {p.name: sha256_file(p) for p in sorted(baseline_dir.glob("*.json"))}
            result = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "success")
            after = {p.name: sha256_file(p) for p in sorted(baseline_dir.glob("*.json"))}
            self.assertEqual(before, after, "inspect 不得自动更新基线文件")


class CliContentIdCompositionTest(unittest.TestCase):
    """content_id 必须包含注册 JSON、基线内容与影响输出的工具源码。"""

    @staticmethod
    def _content_id(root: Path, name: str, **kwargs) -> str:
        project, workspace = build_scenario(root, name, baselines="real", **kwargs)
        result = run_cli(project, inspect_args(workspace, ["--json"]))
        payload = json.loads(result.stdout)
        if result.returncode != 0:
            raise AssertionError(f"场景 {name} 应成功: {payload}")
        return payload["content_id"]

    def test_registry_json_change_changes_content_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._content_id(root, "base")

            def set_display(data):
                data["display_model_id"] = "Independent-Probe-7331"

            changed = self._content_id(root, "reg", registry_mutate=set_display)
            self.assertRegex(base, SHA256_RE)
            self.assertNotEqual(base, changed, "改注册 JSON 必须改变 content_id")

    def test_baseline_content_change_changes_content_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._content_id(root, "base")

            def add_metadata(baseline_dir):
                for path in sorted(baseline_dir.glob("*.json")):
                    data = json.loads(path.read_text(encoding="utf-8"))
                    data["independent_probe_note"] = "harness content id probe"
                    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

            changed = self._content_id(root, "bl", baseline_writer=add_metadata)
            self.assertNotEqual(base, changed, "改基线文件内容必须改变 content_id")

    def test_tool_source_change_changes_content_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._content_id(root, "base")
            changed = self._content_id(root, "tool", tool_patch=True)
            self.assertNotEqual(base, changed, "改影响输出的源码必须改变 content_id")

    def test_input_content_change_changes_content_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._content_id(root, "base")
            project, workspace = build_scenario(root, "inp", baselines="none")
            target = workspace / REFERENCE_ROOT_REL / "inference" / "config.json"
            _support.assert_outside_repo(target)
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            result = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotEqual(
                base, json.loads(result.stdout)["content_id"],
                "改输入内容必须改变 content_id",
            )

    def test_content_id_basis_covers_registry_baseline_and_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            project, workspace = build_scenario(Path(tmp), "basis", baselines="real")
            result = run_cli(project, inspect_args(workspace, ["--json"]))
            self.assertEqual(result.returncode, 0, result.stderr)
            package = Path(json.loads(result.stdout)["output_dir"])
            basis = load_json(package / "run_manifest.json")["content_id_basis"]
            self.assertEqual(basis["registry_config"]["path"],
                             _support.REGISTRY_REL.as_posix())
            self.assertRegex(basis["registry_config"]["sha256"], SHA256_RE)
            self.assertTrue(basis["baselines"])
            for entry in basis["baselines"]:
                self.assertRegex(entry["file_sha256"], SHA256_RE)
                self.assertIsNotNone(entry["declared_source_files"])
            self.assertIn("src/structure_reader/cli.py", basis["tools"]["sources"])
            self.assertRegex(
                basis["tools"]["sources"]["src/structure_reader/cli.py"], SHA256_RE
            )
            # 偶然字段不得进入 identity
            serialized = json.dumps(basis, ensure_ascii=False)
            self.assertNotIn(str(REPO_ROOT), serialized)
            self.assertNotIn(sys.executable, serialized)
            self.assertNotIn("recorded_at", serialized)
            self.assertNotIn("timestamp", serialized)


class CliLowPythonGuardTest(unittest.TestCase):
    """薄入口的 Python 版本守卫（本机确有 3.9.6，故为实机实测）。"""

    LEGACY_PYTHON = Path("/usr/bin/python3")

    def test_real_legacy_interpreter_refused(self):
        if not self.LEGACY_PYTHON.exists():
            self.skipTest("本机无 /usr/bin/python3，低版本守卫只能静态检查")
        version = subprocess.run(
            [str(self.LEGACY_PYTHON), "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        if not version.startswith("3.") or tuple(map(int, version.split("."))) >= (3, 11):
            self.skipTest(f"{self.LEGACY_PYTHON} 为 {version}，不低于 3.11")
        entry = REPO_ROOT / "structure_reader" / "scripts" / "read_model_structure.py"
        for argv in (["--help"], ["inspect", "--model", MODEL_KEY]):
            result = subprocess.run(
                [str(self.LEGACY_PYTHON), str(entry), *argv],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 1, argv)
            message = result.stdout + result.stderr
            self.assertIn("3.11", message)
            self.assertIn(version, message)

    def test_version_parser_and_minimum_are_consistent(self):
        import importlib.util

        entry = REPO_ROOT / "structure_reader" / "scripts" / "read_model_structure.py"
        spec = importlib.util.spec_from_file_location("sr_entry_under_test", entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(tuple(module.MIN_PYTHON), (3, 11))
        self.assertEqual(module._version_tuple("3.9.6"), (3, 9, 6))
        self.assertEqual(module._version_tuple("3.11"), (3, 11))
        self.assertLess(module._version_tuple("3.9.6"), tuple(module.MIN_PYTHON))
        self.assertGreaterEqual(module._version_tuple("3.12.14"), tuple(module.MIN_PYTHON))


class CliRealWorkspaceRunTest(unittest.TestCase):
    """真实 inspect（产品自身工程、非隔离工程）：从其他 cwd 运行并给出实际路径。"""

    def test_documented_inspect_from_other_cwd(self):
        entry = REPO_ROOT / "structure_reader" / "scripts" / "read_model_structure.py"
        with tempfile.TemporaryDirectory() as other_cwd:
            result = subprocess.run(
                [sys.executable, str(entry), "inspect", "--model", MODEL_KEY, "--json"],
                capture_output=True, text=True, timeout=300,
                cwd=other_cwd,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "success")
        self.assertTrue(payload["reviewed"])
        self.assertEqual(
            payload["baseline_statuses"],
            {
                "deepseek_v4_flash_0731_reference": "match",
                "deepseek_v4_flash_0731_hf_comparison": "match",
            },
        )
        package = Path(payload["output_dir"])
        self.assertTrue(package.is_dir(), package)
        self.assertRegex(package.name, SHA256_RE)
        self.assertEqual(package.name, payload["content_id"])
        self.assertEqual(sorted(p.name for p in package.iterdir()), sorted(CONTENT_FILES))
        self.assertIsInstance(payload["reused"], bool)
        self.assertIn("inputs_count", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
