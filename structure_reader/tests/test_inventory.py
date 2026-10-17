"""结构阅读器 S0：输入扫描（inventory）独立测试。

期望值独立取自原始模型文件与 v3 S0 条款（见 fixtures/expected_inventory_s0.json），
不由被测实现输出反推。所有负例只在隔离临时副本上进行，真实输入只读。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support  # noqa: E402,F401
from _support import (
    EXPECTED_INPUT_ROLES,
    MANIFEST_ROLE,
    MODEL_KEY,
    REGISTERED_INPUT_ROLES,
    REGISTRY_REL,
    REPO_ROOT,
    REFERENCE_ROOT_REL,
    SRC_DIR,
    make_baseline_dir,
    make_isolated_workspace,
)

from structure_reader.inventory import (
    InvalidConfigError,
    MissingInputError,
    build_inventory,
)

FIXTURE = _support.load_fixture()
PYTHON = sys.executable


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ScanCacheMixin:
    _cached = None

    @classmethod
    def inventory(cls) -> dict:
        if cls._cached is None:
            cls._cached = build_inventory(MODEL_KEY, workspace_root=REPO_ROOT)
        return cls._cached


class InventoryTopLevelTest(ScanCacheMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = cls.inventory()

    def test_schema_and_artifact_identity(self):
        self.assertEqual(self.inv["schema_version"], "1.0")
        self.assertEqual(self.inv["artifact"]["kind"], "input_inventory")
        self.assertEqual(self.inv["artifact"]["stage"], "S0")
        self.assertEqual(self.inv["model_key"], MODEL_KEY)
        self.assertEqual(self.inv["display_model_id"], "DeepSeek-V4-Flash-0731")
        self.assertEqual(
            Path(self.inv["workspace_root"]).resolve(), REPO_ROOT.resolve()
        )

    def test_known_limits_declared(self):
        limits = self.inv["known_limits"]
        self.assertIsInstance(limits, list)
        self.assertTrue(limits)
        joined = " ".join(limits)
        self.assertIn("静态", joined)


class InputIdentityTest(ScanCacheMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = cls.inventory()

    def test_exactly_eight_inputs_with_expected_roles(self):
        roles = [item["role"] for item in self.inv["inputs"]]
        self.assertEqual(len(roles), FIXTURE["inputs"]["expected_count"])
        self.assertEqual(set(roles), set(EXPECTED_INPUT_ROLES))
        self.assertEqual(len(roles), len(set(roles)), "角色不得重复")
        for registered in REGISTERED_INPUT_ROLES:
            self.assertIn(registered, roles)
        self.assertIn(MANIFEST_ROLE, roles)

    def test_sha256_and_size_are_real_and_verifiable(self):
        pattern = re.compile(FIXTURE["inputs"]["sha256_regex"])
        for item in self.inv["inputs"]:
            self.assertRegex(item["sha256"], pattern, item["role"])
            self.assertGreater(item["size_bytes"], 0, item["role"])
            actual = Path(item["absolute_path"])
            self.assertTrue(actual.is_file(), item["role"])
            self.assertEqual(actual.stat().st_size, item["size_bytes"])
            self.assertEqual(
                sha256_of(actual), item["sha256"], f"{item['role']} SHA 应可独立复算"
            )

    def test_paths_are_workspace_relative_and_nonempty(self):
        for item in self.inv["inputs"]:
            self.assertTrue(item["path"], item["role"])
            self.assertFalse(
                os.path.isabs(item["path"]), f"{item['role']} 的 path 应为工作区相对路径"
            )
            self.assertTrue(item["description"], item["role"])

    def test_manifest_input_is_declared_as_declaration_not_weights(self):
        by_role = {item["role"]: item for item in self.inv["inputs"]}
        manifest = by_role[MANIFEST_ROLE]
        self.assertEqual(manifest["path"], f"{REFERENCE_ROOT_REL.as_posix()}/metadata_manifest.json")
        self.assertIn("声明", manifest["description"])


class RawVsConstructorStateTest(ScanCacheMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = cls.inventory()
        cls.configs = cls.inv["configs"]
        cls.fields = cls.configs["constructor_state"]["fields"]

    def test_raw_root_config_preserved_with_hf_names(self):
        raw = self.configs["raw_root_config"]
        self.assertEqual(raw["role"], "root_config")
        self.assertTrue(raw["path"].endswith("config.json"))
        self.assertNotIn("inference/config.json", raw["path"])
        for field, value in FIXTURE["raw_root_config"].items():
            self.assertEqual(raw["fields"][field], value, field)
        self.assertEqual(raw["hf_consumer"]["config_class"], "DeepseekV4Config")

    def test_run_config_is_separate_input_from_raw_root_config(self):
        raw_path = Path(self.configs["raw_root_config"]["path"])
        run_path = Path(self.configs["constructor_state"]["run_config_path"])
        self.assertNotEqual(raw_path, run_path)
        self.assertTrue(str(run_path).endswith("inference/config.json"))

    def test_constructor_state_layer_values_and_sources(self):
        for field in ("n_layers", "n_mtp_layers", "n_hash_layers", "hc_mult"):
            expected = FIXTURE["constructor_state_expected"][field]
            actual = self.fields[field]
            self.assertEqual(actual["value"], expected["value"], field)
            self.assertEqual(actual["source"], expected["source"], field)
            self.assertIn("inference/config.json", actual["source_path"], field)

    def test_defaults_max_seq_len_and_batch_size_come_from_dataclass(self):
        for field in ("max_seq_len", "max_batch_size"):
            expected = FIXTURE["constructor_state_expected"][field]
            actual = self.fields[field]
            self.assertEqual(actual["value"], expected["value"], field)
            self.assertEqual(actual["source"], expected["source"], field)
            self.assertIn("inference/model.py", actual["source_path"], field)

    def test_run_config_keys_all_consumed(self):
        self.assertEqual(
            self.configs["constructor_state"]["run_config_keys_not_consumed"], []
        )

    def test_no_field_without_source(self):
        for name, entry in self.fields.items():
            self.assertIn(entry["source"], {"run_config", "dataclass_default"}, name)
            self.assertTrue(entry["source_path"], name)

    def test_ast_method_declared_no_exec(self):
        evidence = self.configs["constructor_state"]["evidence"]
        self.assertIn("AST", evidence["method"])
        self.assertIn("未使用 eval/exec", evidence["method"])


class EntrypointOverrideTest(ScanCacheMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = cls.inventory()
        cls.overrides = cls.inv["configs"]["entrypoint_overrides"]

    def _find(self, field):
        matches = [o for o in self.overrides["overrides"] if o["field"] == field]
        self.assertTrue(matches, f"未记录入口覆盖 {field}")
        return matches[0]

    def test_constructor_call_recorded(self):
        call = self.overrides["constructor_call"]
        self.assertEqual(call["symbol"], "ModelArgs")
        self.assertIn("ModelArgs(**json.load(f))", call["call_source"])

    def test_interactive_guard_records_65536_and_1(self):
        for field, expected in (
            ("max_seq_len", FIXTURE["entrypoint_interactive_overrides"]["max_seq_len"]),
            ("max_batch_size", FIXTURE["entrypoint_interactive_overrides"]["max_batch_size"]),
        ):
            record = self._find(field)
            self.assertEqual(record["value"], expected["value"], field)
            guards = " ".join(g["condition_source"] for g in record["guards"])
            self.assertIn(expected["guard_contains"], guards, field)

    def test_interactive_overrides_not_merged_into_constructor_state(self):
        fields = self.inv["configs"]["constructor_state"]["fields"]
        self.assertEqual(fields["max_seq_len"]["value"], 4096)
        self.assertEqual(fields["max_batch_size"]["value"], 4)

    def test_temperature_override_keeps_symbol_not_evaluated(self):
        record = self._find("temperature")
        expected = FIXTURE["entrypoint_interactive_overrides"]["temperature"]
        self.assertFalse(record["is_literal"])
        self.assertIsNone(record["value"])
        self.assertIn(expected["expression_contains"], record["expression"])

    def test_overrides_notes_state_not_merged(self):
        notes = " ".join(self.overrides["notes"])
        self.assertIn("未并入", notes)


class RawToEffectivePairsTest(ScanCacheMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = cls.inventory()
        cls.pairs = cls.inv["raw_to_effective_pairs"]

    def test_equivalent_pairs_report_match_with_values(self):
        by_effective = {p["effective_field"]: p for p in self.pairs["equivalent_pairs"]}
        self.assertEqual(by_effective["dim"]["raw_value"], 4096)
        self.assertEqual(by_effective["dim"]["effective_value"], 4096)
        self.assertTrue(by_effective["dim"]["match"])
        self.assertEqual(by_effective["n_layers"]["raw_value"], 43)
        self.assertEqual(by_effective["n_layers"]["effective_value"], 43)
        self.assertTrue(by_effective["n_layers"]["match"])
        for pair in self.pairs["equivalent_pairs"]:
            self.assertTrue(pair["raw_field_found"], pair["raw_field"])
            self.assertTrue(pair["match"], pair["effective_field"])

    def test_n_mtp_layers_recorded_as_non_equivalent(self):
        by_effective = {p["effective_field"]: p for p in self.pairs["non_equivalent_pairs"]}
        mtp = by_effective["n_mtp_layers"]
        self.assertEqual(
            mtp["raw_value"], FIXTURE["non_equivalent"]["num_nextn_predict_layers"]
        )
        self.assertEqual(
            mtp["effective_value"], FIXTURE["non_equivalent"]["n_mtp_layers"]
        )
        self.assertIn("不同", mtp["note"])

    def test_max_seq_len_not_substituted_by_max_position_embeddings(self):
        by_effective = {p["effective_field"]: p for p in self.pairs["non_equivalent_pairs"]}
        entry = by_effective["max_seq_len"]
        self.assertEqual(entry["raw_value"], 1048576)
        self.assertEqual(entry["effective_value"], 4096)
        self.assertIn("不得用", entry["note"])


class WeightsAndRevisionTest(ScanCacheMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = cls.inventory()

    def test_local_weights_actually_zero_and_index_missing(self):
        actual = self.inv["weights"]["actual"]
        self.assertEqual(
            actual["safetensors_count"],
            FIXTURE["weights"]["actual_safetensors_count"],
        )
        self.assertFalse(actual["index_present"])
        self.assertEqual(actual["shard_pattern_matches"], [])
        self.assertFalse(Path(actual["index_path"]).exists())

    def test_manifest_declaration_recorded_separately(self):
        declared = self.inv["weights"]["manifest_declared"]
        self.assertEqual(
            declared["safetensors_count"],
            FIXTURE["weights"]["manifest_declared_safetensors_count"],
        )
        self.assertTrue(declared["index_declared"])
        self.assertIn("声明", self.inv["weights"]["report_basis"])
        self.assertIn("磁盘现存", self.inv["weights"]["report_basis"])

    def test_consistency_reports_mismatch_not_fake_presence(self):
        consistency = self.inv["weights"]["consistency"]
        self.assertEqual(consistency["status"], "mismatch")
        self.assertIn("不冒充", consistency["note"])

    def test_revision_declared_but_not_online_verified(self):
        revision = self.inv["revisions"]["model_manifest"]
        self.assertEqual(revision["declared_repo_id"], "deepseek-ai/DeepSeek-V4-Flash-0731")
        self.assertEqual(
            revision["declared_revision"], FIXTURE["weights"]["manifest_revision"]
        )
        self.assertFalse(revision["verification"]["online_consistency_checked"])

    def test_hf_checkout_state_recorded_without_write(self):
        hf = self.inv["revisions"]["hf_checkout"]
        self.assertIn("commit", hf)
        self.assertIn("dirty", hf)
        self.assertIn("不做任何写操作", hf["method"])


class CapabilityAndStdlibTest(ScanCacheMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = cls.inventory()

    def test_capability_declares_stdlib_only(self):
        caps = self.inv["capabilities"]
        self.assertTrue(caps["core_dependencies"]["stdlib_only"])
        self.assertFalse(caps["core_dependencies"]["torch_required"])
        self.assertIn("interpreter", caps)

    def test_scan_does_not_import_torch_transformers_or_reference_model(self):
        code = (
            "import sys, json;"
            f"sys.path.insert(0, {str(SRC_DIR)!r});"
            "from structure_reader.inventory import build_inventory;"
            f"inv = build_inventory({MODEL_KEY!r}, workspace_root={str(REPO_ROOT)!r});"
            "banned = [m for m in ('torch','transformers','safetensors','model',"
            "'encoding_dsv4','kernel') if m in sys.modules];"
            "print('BANNED_MODULES=' + json.dumps(banned))"
        )
        proc = subprocess.run(
            [PYTHON, "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("BANNED_MODULES=[]", proc.stdout)

    def test_source_uses_ast_not_eval_exec(self):
        import ast as ast_module

        source = (
            SRC_DIR / "structure_reader" / "inventory.py"
        ).read_text(encoding="utf-8")
        tree = ast_module.parse(source)
        for node in ast_module.walk(tree):
            if isinstance(node, ast_module.Call) and isinstance(node.func, ast_module.Name):
                self.assertNotIn(
                    node.func.id, {"eval", "exec", "compile"}, node.func.id
                )


class BaseLineTest(ScanCacheMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv = cls.inventory()

    def test_all_baselines_missing_and_not_created(self):
        # 前置条件修正（2026-10-12，开发单 03）：固定基线现已落盘于
        # structure_reader/configs/source_baselines/，真实目录不再“缺基线”。
        # 原断言假定真实目录缺基线，已被真实前置条件推翻；这里改在**隔离副本**上
        # 明确构造“无基线”场景，保持 missing 契约不变，不放宽、不隐藏缺陷。
        # 真实目录在“基线存在且一致”下的行为由 test_real_baselines_match_and_not_modified 覆盖。
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(Path(tmp))
            baseline_dir = workspace / "structure_reader" / "configs" / "source_baselines"
            self.assertFalse(baseline_dir.exists(), "隔离副本初始应明确无基线目录")
            inventory = build_inventory(MODEL_KEY, workspace_root=workspace)
            statuses = {
                entry["name"]: entry["status"]
                for entry in inventory["baseline_comparison"]
            }
            self.assertEqual(
                set(statuses), set(FIXTURE["source_baseline_names"])
            )
            for name, status in statuses.items():
                self.assertEqual(status, "missing", name)
            self.assertFalse(
                baseline_dir.exists(), "扫描不得自动创建基线目录/文件"
            )

    def test_real_baselines_match_and_not_modified(self):
        # 固定基线已落盘：真实工作区应为 match，且扫描不得改动基线文件。
        baseline_dir = REPO_ROOT / "structure_reader" / "configs" / "source_baselines"
        before = {
            path.name: sha256_of(path)
            for path in sorted(baseline_dir.glob("*.json"))
        }
        self.assertEqual(
            sorted(before), sorted(f"{n}.json" for n in FIXTURE["source_baseline_names"])
        )
        statuses = {
            entry["name"]: entry["status"]
            for entry in self.inv["baseline_comparison"]
        }
        for name, status in statuses.items():
            self.assertEqual(status, "match", name)
        after = {
            path.name: sha256_of(path)
            for path in sorted(baseline_dir.glob("*.json"))
        }
        self.assertEqual(before, after, "扫描不得创建或改动基线文件")

    def test_baseline_mismatch_detected_on_isolated_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(Path(tmp))
            baseline_dir = make_baseline_dir(workspace)
            wrong_sha = "0" * 64
            for name in FIXTURE["source_baseline_names"]:
                (baseline_dir / f"{name}.json").write_text(
                    json.dumps(
                        {
                            "source_files": [
                                {
                                    "path": f"{REFERENCE_ROOT_REL.as_posix()}/config.json",
                                    "sha256": wrong_sha,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
            inventory = build_inventory(MODEL_KEY, workspace_root=workspace)
            statuses = {
                entry["name"]: entry["status"]
                for entry in inventory["baseline_comparison"]
            }
            for name, status in statuses.items():
                self.assertEqual(status, "mismatch", name)
            detail = inventory["baseline_comparison"][0]["file_comparison"][0]
            self.assertNotEqual(detail["actual_sha256"], wrong_sha)

    def test_baseline_match_detected_on_isolated_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(Path(tmp))
            baseline_dir = make_baseline_dir(workspace)
            rel = f"{REFERENCE_ROOT_REL.as_posix()}/config.json"
            good_sha = sha256_of(REPO_ROOT / rel)
            for name in FIXTURE["source_baseline_names"]:
                (baseline_dir / f"{name}.json").write_text(
                    json.dumps({"source_files": {rel: good_sha}}),
                    encoding="utf-8",
                )
            inventory = build_inventory(MODEL_KEY, workspace_root=workspace)
            for entry in inventory["baseline_comparison"]:
                self.assertEqual(entry["status"], "match", entry["name"])


class NegativeInputTest(unittest.TestCase):
    def test_missing_input_raises_on_isolated_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(Path(tmp))
            target = workspace / REFERENCE_ROOT_REL / "inference" / "config.json"
            self.assertTrue(target.is_file())
            _support.safe_unlink(target)
            self.assertFalse(target.exists())
            with self.assertRaises(MissingInputError):
                build_inventory(MODEL_KEY, workspace_root=workspace)

    def test_malformed_model_json_raises_on_isolated_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(Path(tmp))
            target = workspace / REFERENCE_ROOT_REL / "inference" / "config.json"
            _support.safe_unlink(target)
            _support.safe_write_text(target, "{ broken json")
            with self.assertRaises(InvalidConfigError):
                build_inventory(MODEL_KEY, workspace_root=workspace)

    def test_negative_cases_do_not_touch_real_inputs(self):
        busiest = REPO_ROOT / REFERENCE_ROOT_REL / "inference" / "config.json"
        before = sha256_of(busiest)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(Path(tmp))
            target = workspace / REFERENCE_ROOT_REL / "inference" / "config.json"
            self.assertNotEqual(target.resolve(), busiest.resolve())
            _support.safe_unlink(target)
            with self.assertRaises(MissingInputError):
                build_inventory(MODEL_KEY, workspace_root=workspace)
        self.assertEqual(sha256_of(busiest), before)


class CwdIndependenceTest(unittest.TestCase):
    def test_default_scan_is_cwd_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                inventory = build_inventory(MODEL_KEY)
            finally:
                os.chdir(previous)
        self.assertEqual(
            Path(inventory["workspace_root"]).resolve(), REPO_ROOT.resolve()
        )
        for item in inventory["inputs"]:
            self.assertTrue(
                str(Path(item["absolute_path"]).resolve()).startswith(
                    str(REPO_ROOT.resolve())
                ),
                item["role"],
            )


class RepeatStabilityTest(unittest.TestCase):
    def test_repeat_scan_core_manifest_stable(self):
        first = build_inventory(MODEL_KEY, workspace_root=REPO_ROOT)
        second = build_inventory(MODEL_KEY, workspace_root=REPO_ROOT)

        def core(inv):
            return {
                "inputs": sorted(
                    (i["role"], i["path"], i["sha256"], i["size_bytes"])
                    for i in inv["inputs"]
                ),
                "fields": {
                    name: (entry["value"], entry["source"])
                    for name, entry in inv["configs"]["constructor_state"]["fields"].items()
                },
                "overrides": [
                    (
                        o["field"],
                        o.get("value"),
                        o.get("is_literal"),
                        tuple(g["condition_source"] for g in o["guards"]),
                    )
                    for o in inv["configs"]["entrypoint_overrides"]["overrides"]
                ],
                "weights_actual": inv["weights"]["actual"],
                "weights_declared": inv["weights"]["manifest_declared"],
                "baselines": [
                    (e["name"], e["status"]) for e in inv["baseline_comparison"]
                ],
            }

        self.assertEqual(core(first), core(second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
