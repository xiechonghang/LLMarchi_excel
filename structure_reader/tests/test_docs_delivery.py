"""结构阅读器 S0：交付文档、工程元数据与基线元数据范围的独立测试。

对应 verification_task_03_s0_complete §6：
- 文档命令需可实际复现；README 与 docs 正确区分入口/库/原始输入/配置/产物/测试/临时；
- pyproject 无重型运行依赖；.gitignore 覆盖暂存/生成目录；
- progress 标待验收，不自批 S0 或后续阶段；
- 基线 metadata 只能声明 S0 输入身份与配置消费审查，不得声称整个模型结构语义已审核。

本文件只读产品文件，不写任何真实路径。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support  # noqa: E402
from _support import FIXTURES_DIR, REPO_ROOT  # noqa: E402

PROJECT = REPO_ROOT / "structure_reader"
DOCS = PROJECT / "docs"
ENTRY = PROJECT / "scripts" / "read_model_structure.py"
LATER_STAGE_COMMANDS = ("validate", "render", "export-tables")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class DeliveryInventoryTest(unittest.TestCase):
    """15 份定版产品文件与文档应全部存在且非空。"""

    EXPECTED = (
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "configs/models/deepseek_v4_flash_0731.json",
        "configs/source_baselines/deepseek_v4_flash_0731_reference.json",
        "configs/source_baselines/deepseek_v4_flash_0731_hf_comparison.json",
        "docs/developer_guide.md",
        "docs/output_reference.md",
        "docs/progress.md",
        "docs/user_guide.md",
        "scripts/read_model_structure.py",
        "src/structure_reader/__init__.py",
        "src/structure_reader/cli.py",
        "src/structure_reader/inventory.py",
        "src/structure_reader/model_registry.py",
    )

    def test_all_expected_product_files_exist_nonempty(self):
        self.assertEqual(len(self.EXPECTED), 15)
        for relative in self.EXPECTED:
            path = PROJECT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertGreater(path.stat().st_size, 0, relative)


class ReadmeScopeTest(unittest.TestCase):
    """README 必须区分各类目录职责，且不夸大能力。"""

    @classmethod
    def setUpClass(cls):
        cls.text = read(PROJECT / "README.md")

    def test_distinguishes_all_responsibility_areas(self):
        for token in ("scripts/", "src/", "configs/", "models/", "hflib/",
                      "tests/", "outputs/", ".work/"):
            self.assertIn(token, self.text, f"README 应说明 {token} 的职责")

    def test_states_s0_scope_and_missing_features(self):
        self.assertIn("S0", self.text)
        self.assertIn("不是用户最终阅读的模型结构图", self.text)
        for feature in ("build", "validate", "render", "export-tables"):
            self.assertIn(feature, self.text, feature)

    def test_build_claim_matches_reality(self):
        # build 声称可用时必须指向真实产物路径；未实现命令不得声称可用
        text = self.text
        if "build" in text and "已实现" in text:
            self.assertIn("outputs/", text)

    def test_no_heavy_runtime_dependency_claimed(self):
        self.assertIn("不需要 torch", self.text)
        self.assertIn("其他版本未实测", self.text)


class UserGuideCommandTest(unittest.TestCase):
    """user_guide 的命令必须可实际复现，且只列已实现命令。"""

    @classmethod
    def setUpClass(cls):
        cls.text = read(DOCS / "user_guide.md")

    def test_documented_help_commands_reproduce(self):
        with tempfile.TemporaryDirectory() as cwd:
            for argv in (["--help"], ["inspect", "--help"]):
                result = subprocess.run(
                    [sys.executable, str(ENTRY), *argv],
                    capture_output=True, text=True, timeout=120, cwd=cwd,
                )
                self.assertEqual(result.returncode, 0, (argv, result.stderr))
                self.assertIn("inspect", result.stdout)

    def test_documented_inspect_command_is_well_formed(self):
        self.assertRegex(
            self.text,
            r"read_model_structure\.py inspect --model deepseek_v4_flash_0731",
        )
        self.assertIn("--workspace-root", self.text)
        self.assertIn("--json", self.text)

    def test_unimplemented_commands_not_documented_as_usage(self):
        for command in LATER_STAGE_COMMANDS:
            self.assertNotRegex(
                self.text,
                rf"read_model_structure\.py\s+{re.escape(command)}\b",
                f"user_guide 不应给出未实现命令 {command} 的用法",
            )
        self.assertIn("尚未实现", self.text)

    def test_build_documented_as_usable(self):
        # V1：build 已实现并写入 user_guide（含场景集与产物说明）
        self.assertIn("read_model_structure.py build", self.text)
        self.assertIn("--model deepseek_v4_flash_0731", self.text)
        self.assertIn("--scenario-set standard", self.text)
        self.assertIn("decode_p128", self.text)

    def test_documents_exit_codes_and_states(self):
        for token in ("success", "unreviewed", "failed"):
            self.assertIn(token, self.text, token)
        self.assertIn("不表示通过", self.text)
        for code in ("0", "1", "2"):
            self.assertIn(code, self.text, code)

    def test_states_output_is_not_structure_graph(self):
        self.assertIn("不要期待在本目录找到模型结构图", self.text)


class OutputReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = read(DOCS / "output_reference.md")

    def test_lists_actual_schemas_and_files(self):
        for token in ("run_manifest_v1", "input_manifest_v2",
                      "capability_report_v2", "failed_run_v1"):
            self.assertIn(token, self.text, token)
        for name in ("run_manifest.json", "input_manifest.json",
                     "capability_report.json", "README.md"):
            self.assertIn(name, self.text, name)

    def test_distinguishes_declared_vs_measured_evidence(self):
        self.assertIn("实测证据", self.text)
        self.assertIn("未线上核验", self.text)
        self.assertIn("AST 静态读取", self.text)


class DeveloperGuideTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = read(DOCS / "developer_guide.md")

    def test_states_readonly_rule_and_negative_isolation(self):
        self.assertIn("只读", self.text)
        self.assertIn("隔离副本", self.text)
        self.assertIn("assert_outside_repo", self.text)

    def test_states_baseline_not_auto_updated(self):
        self.assertIn("不得自动更新", self.text)

    def test_states_content_id_contract(self):
        self.assertIn("content_id", self.text)
        self.assertIn("绝对工作区路径", self.text)
        self.assertIn("output_consistency_conflict", self.text)

    def test_lists_known_limitations(self):
        self.assertIn("已知限制", self.text)
        self.assertIn("未实现", self.text)


class ProjectMetadataTest(unittest.TestCase):
    def test_pyproject_has_no_heavy_runtime_dependencies(self):
        text = read(PROJECT / "pyproject.toml")
        self.assertRegex(text, r"(?m)^\s*dependencies\s*=\s*\[\s*\]")
        project_section = text.split("[project.optional-dependencies]")[0]
        for heavy in ("torch", "transformers", "safetensors"):
            self.assertNotIn(heavy, project_section.lower(), heavy)
        self.assertIn('requires-python = ">=3.11"', text)

    def test_gitignore_covers_generated_and_staging_dirs(self):
        text = read(PROJECT / ".gitignore")
        for token in ("outputs/", ".work/", "__pycache__/"):
            self.assertIn(token, text, token)


# ---------------------------------------------------------------------------
# 进度文件生命周期契约
#
# 设计原则（verification_04 修正）：测试只断言“结构完备 + 状态合法 + 声称通过
# 必须有验收记录导航”这条规则，**不写死任何阶段的当前状态**。S0 被批准、
# 或进入 S1+ 时，真实 progress.md 的合法变化不应导致测试误报。
# 规则本身由 tests/fixtures/ 的三份独立示例固定（待验收 / 经批准 / 违规），
# 期望值不取自 outputs 或任何运行时产物，也不自动批准阶段。
# ---------------------------------------------------------------------------

STAGES = tuple(f"S{i}" for i in range(9))
LEGAL_STATUSES = ("pending", "in_progress", "passed", "failed")
# 明确验收记录导航 = 指向 audit/ 下验收记录文件的路径/链接（审核日志或能力状态）。
# 只提到“验收记录”字样不算导航；必须是实际记录文件路径。
ACCEPTANCE_NAV_RE = re.compile(r"audit/[^\s)\]\"'<>]*\.(?:md|json)")
ACCEPTANCE_RECORD_FILES = ("final_acceptance.md", "capability_status.json")


def parse_stage_rows(text: str) -> dict[str, str]:
    """从 Markdown 表格抽出 阶段 → 状态单元格。"""
    rows: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"\|\s*(S\d)\b[^|]*\|([^|]*)\|", line)
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


def status_token(cell: str) -> str | None:
    """返回状态单元格中唯一合法的状态词；含糊或缺失返回 None。"""
    found = [status for status in LEGAL_STATUSES if re.search(rf"\b{status}\b", cell)]
    return found[0] if len(found) == 1 else None


def has_acceptance_navigation(text: str) -> bool:
    """文档是否给出明确验收记录导航（指向 audit/ 下记录文件的链接/路径）。

    仅出现“验收记录”“审核日志”等字样**不算**导航——必须是实际记录文件路径，
    这样才能拒绝“自述通过但无可核验记录”的进度文件。
    """
    return ACCEPTANCE_NAV_RE.search(text) is not None


def validate_progress(text: str) -> dict:
    """校验进度文件结构；返回 {rows, passed_stages, issues, has_acceptance_navigation}。

    规则：每个阶段一行且状态合法；任何阶段标 passed 时必须存在验收记录导航。
    """
    rows = parse_stage_rows(text)
    issues: list[str] = []
    for stage in STAGES:
        if stage not in rows:
            issues.append(f"缺少阶段行: {stage}")
            continue
        if status_token(rows[stage]) is None:
            issues.append(f"{stage} 状态不合法或含糊: {rows[stage]!r}")
    passed_stages = [
        stage for stage in STAGES
        if stage in rows and status_token(rows[stage]) == "passed"
    ]
    navigation = has_acceptance_navigation(text)
    if passed_stages and not navigation:
        issues.append(
            "阶段标 passed 但无明确验收记录导航: " + ", ".join(passed_stages)
        )
    return {
        "rows": rows,
        "passed_stages": passed_stages,
        "issues": issues,
        "has_acceptance_navigation": navigation,
    }


class ProgressLifecycleContractTest(unittest.TestCase):
    """进度契约：只查结构/合法状态/验收记录导航，不写死当前阶段。"""

    @classmethod
    def setUpClass(cls):
        cls.real_text = read(DOCS / "progress.md")
        cls.pending_text = read(FIXTURES_DIR / "progress_pending_acceptance.md")
        cls.approved_text = read(FIXTURES_DIR / "progress_codex_approved.md")
        cls.invalid_text = read(FIXTURES_DIR / "progress_passed_without_acceptance.md")

    # --- 规则由独立固件固定 ---

    def test_pending_fixture_cannot_claim_passed(self):
        result = validate_progress(self.pending_text)
        self.assertEqual(result["issues"], [], result["issues"])
        self.assertEqual(result["passed_stages"], [], "待验收固件不得有阶段标 passed")

    def test_approved_fixture_allows_passed_with_acceptance_navigation(self):
        result = validate_progress(self.approved_text)
        self.assertEqual(result["issues"], [], result["issues"])
        self.assertIn("S0", result["passed_stages"])
        self.assertTrue(result["has_acceptance_navigation"])

    def test_rule_rejects_passed_without_acceptance_record(self):
        result = validate_progress(self.invalid_text)
        self.assertIn("S0", result["passed_stages"])
        self.assertTrue(
            any("验收记录" in issue for issue in result["issues"]), result["issues"]
        )

    def test_rule_accepts_both_lifecycle_states(self):
        # 待验收与已批准都必须结构合法：契约不偏向任何当前阶段。
        self.assertEqual(validate_progress(self.pending_text)["issues"], [])
        self.assertEqual(validate_progress(self.approved_text)["issues"], [])

    # --- 真实 progress：只验证完整性 / 合法状态 / 验收记录导航 ---

    def test_real_progress_lists_every_stage_exactly_once(self):
        rows = parse_stage_rows(self.real_text)
        self.assertEqual(sorted(rows), sorted(STAGES))

    def test_real_progress_statuses_are_legal_and_consistent(self):
        result = validate_progress(self.real_text)
        self.assertEqual(result["issues"], [], result["issues"])

    def test_real_progress_has_acceptance_record_navigation(self):
        self.assertTrue(
            has_acceptance_navigation(self.real_text),
            "progress 必须给出明确验收记录导航（指向 audit/ 下记录文件的链接）",
        )
        self.assertTrue(
            any(name in self.real_text for name in ACCEPTANCE_RECORD_FILES),
            "progress 应引用审核日志或能力状态记录文件",
        )

    def test_no_passed_stage_lacks_acceptance_navigation(self):
        result = validate_progress(self.real_text)
        if result["passed_stages"]:
            self.assertTrue(result["has_acceptance_navigation"])


class BaselineMetadataScopeTest(unittest.TestCase):
    """基线只能声明 S0 输入身份/配置消费审查，不得声称整模型语义已审。"""

    def _baseline(self, name: str) -> dict:
        return json.loads(read(PROJECT / "configs" / "source_baselines" / name))

    def test_baseline_kind_is_input_identity_only(self):
        for name in ("deepseek_v4_flash_0731_reference.json",
                     "deepseek_v4_flash_0731_hf_comparison.json"):
            data = self._baseline(name)
            self.assertEqual(data["baseline_kind"], "input_identity_baseline", name)
            self.assertEqual(data["schema_version"], "1.0", name)

    def test_review_scope_explicitly_excludes_semantics(self):
        for name in ("deepseek_v4_flash_0731_reference.json",
                     "deepseek_v4_flash_0731_hf_comparison.json"):
            data = self._baseline(name)
            scope = data["review_scope"]
            self.assertTrue(scope["not_covered"], name)
            not_covered = " ".join(scope["not_covered"])
            self.assertTrue(
                any(token in not_covered for token in ("语义", "结构")), name
            )
            self.assertTrue(scope["pending"], name)

    def test_fixation_policy_states_no_semantic_review(self):
        reference = self._baseline("deepseek_v4_flash_0731_reference.json")
        self.assertIn("不做语义审查", reference["fixation_policy"])
        self.assertIn("不会更新", reference["fixation_policy"])

    def test_declared_revision_kept_separate_from_local_hash(self):
        reference = self._baseline("deepseek_v4_flash_0731_reference.json")
        metadata = reference["metadata"]
        self.assertIn("manifest_revision_declared", metadata)
        self.assertIn("不可互相替代", metadata["note"])

    def test_source_files_are_identity_entries_not_semantic_claims(self):
        for name in ("deepseek_v4_flash_0731_reference.json",
                     "deepseek_v4_flash_0731_hf_comparison.json"):
            data = self._baseline(name)
            self.assertTrue(data["source_files"], name)
            for entry in data["source_files"]:
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$", name)
                self.assertIn("path", entry, name)
                self.assertIn("role", entry, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
