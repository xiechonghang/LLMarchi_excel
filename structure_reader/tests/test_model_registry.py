"""结构阅读器 S0：模型注册表独立测试。

契约（来自 v3 S0 条款与第二轮开发单 01）：
- 工作区根按包位置定位，不依赖调用方 cwd；
- configs/models/<key>.json 是唯一模型定义数据源，改动必须真实作用于
  get_model_entry 与扫描；
- 未知模型、缺注册 JSON、坏 JSON、必要字段缺失一律用具名 StructureReaderError 报告；
- 所有负例只在隔离临时副本上执行，不写真实输入。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _support  # noqa: E402,F401  (把 src/ 加入 sys.path)
from _support import (
    EXPECTED_INPUT_ROLES,
    MODEL_KEY,
    REGISTERED_INPUT_ROLES,
    REGISTRY_REL,
    REPO_ROOT,
    make_isolated_workspace,
)

from structure_reader.model_registry import (
    StructureReaderError,
    UnknownModelModelError,
    find_workspace_root,
    get_model_entry,
    list_model_keys,
)


class WorkspaceRootTest(unittest.TestCase):
    def test_default_root_is_package_location_not_cwd(self):
        expected = REPO_ROOT.resolve()
        self.assertEqual(find_workspace_root().resolve(), expected)
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertEqual(find_workspace_root().resolve(), expected)
            finally:
                os.chdir(previous)

    def test_explicit_start_dir(self):
        self.assertEqual(
            find_workspace_root(REPO_ROOT).resolve(), REPO_ROOT.resolve()
        )


class RegistryLookupTest(unittest.TestCase):
    def test_model_key_registered(self):
        self.assertIn(MODEL_KEY, list_model_keys())

    def test_entry_resolves_all_registered_roles_under_root(self):
        entry = get_model_entry(MODEL_KEY, workspace_root=REPO_ROOT)
        self.assertEqual(entry["model_key"], MODEL_KEY)
        resolved = entry["resolved"]
        for role in REGISTERED_INPUT_ROLES:
            self.assertIn(role, resolved)
            self.assertTrue(
                Path(resolved[role]).is_absolute(), f"{role} 应为绝对路径"
            )
            self.assertTrue(
                str(resolved[role]).startswith(str(REPO_ROOT.resolve())),
                f"{role} 应位于工作区根之下",
            )
        for key in (
            "reference_root_dir",
            "hf_checkout_root_dir",
            "hf_source_root_dir",
            "workspace_root",
        ):
            self.assertIn(key, resolved)
        self.assertEqual(
            Path(resolved["workspace_root"]).resolve(), REPO_ROOT.resolve()
        )

    def test_unknown_model_raises_named_error(self):
        with self.assertRaises(UnknownModelModelError) as ctx:
            get_model_entry("no_such_model", workspace_root=REPO_ROOT)
        self.assertIsInstance(ctx.exception, StructureReaderError)


class RegistryJsonIsSourceOfTruthTest(unittest.TestCase):
    """第二轮开发单 01 的交付判据：JSON 改动必须作用到注册查询与扫描。"""

    def test_display_model_id_from_isolated_registry_json(self):
        probe = "Independent-Probe-0421"
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(
                Path(tmp), display_model_id=probe
            )
            entry = get_model_entry(MODEL_KEY, workspace_root=workspace)
            self.assertEqual(entry["display_model_id"], probe)
            self.assertEqual(
                entry["resolved"]["workspace_root"], str(workspace)
            )

    def test_registry_json_edit_reaches_inventory_scan(self):
        from structure_reader.inventory import build_inventory

        probe = "Independent-Scan-Probe-7"
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(
                Path(tmp), display_model_id=probe
            )
            inventory = build_inventory(MODEL_KEY, workspace_root=workspace)
            self.assertEqual(inventory["display_model_id"], probe)
            self.assertEqual(inventory["workspace_root"], str(workspace))

    def test_real_registry_json_semantics_preserved(self):
        # 隔离副本只改 display_model_id，其余输入身份不应漂移。
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(
                Path(tmp), display_model_id="Independent-Probe-0421"
            )
            entry = get_model_entry(MODEL_KEY, workspace_root=workspace)
            roles = {item["role"] for item in entry["inputs"]}
            self.assertEqual(roles, set(REGISTERED_INPUT_ROLES))
            real = get_model_entry(MODEL_KEY, workspace_root=REPO_ROOT)
            self.assertEqual(
                Path(entry["resolved"]["root_config"]).name,
                Path(real["resolved"]["root_config"]).name,
            )


class RegistryJsonFailureTest(unittest.TestCase):
    """坏注册 JSON 必须具名报错，且只在临时副本上进行。"""

    def test_missing_registry_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(
                Path(tmp), drop_registry_json=True
            )
            self.assertFalse((workspace / REGISTRY_REL).exists())
            with self.assertRaises(StructureReaderError):
                get_model_entry(MODEL_KEY, workspace_root=workspace)

    def test_malformed_registry_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(
                Path(tmp), raw_json_text="{ this is not json"
            )
            with self.assertRaises(StructureReaderError):
                get_model_entry(MODEL_KEY, workspace_root=workspace)

    def test_registry_json_non_object_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(
                Path(tmp), raw_json_text='["not", "an", "object"]'
            )
            with self.assertRaises(StructureReaderError):
                get_model_entry(MODEL_KEY, workspace_root=workspace)

    def test_registry_json_missing_required_field_raises(self):
        def drop_constructor(data):
            data.pop("constructor", None)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(
                Path(tmp), registry_mutate=drop_constructor
            )
            with self.assertRaises(StructureReaderError):
                get_model_entry(MODEL_KEY, workspace_root=workspace)

    def test_registry_json_missing_inputs_raises(self):
        def drop_inputs(data):
            data.pop("inputs", None)

        with tempfile.TemporaryDirectory() as tmp:
            workspace = make_isolated_workspace(
                Path(tmp), registry_mutate=drop_inputs
            )
            with self.assertRaises(StructureReaderError):
                get_model_entry(MODEL_KEY, workspace_root=workspace)


class ShippedRegistryJsonTest(unittest.TestCase):
    def test_shipped_json_parses_and_has_contract_fields(self):
        data = _support.registry_json()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["model_key"], MODEL_KEY)
        for field in (
            "display_model_id",
            "constructor",
            "inputs",
            "weights",
            "source_baselines",
            "raw_to_effective_pairs",
        ):
            self.assertIn(field, data, f"注册 JSON 缺少字段 {field}")
        for field in ("dataclass_symbol", "dataclass_source",
                      "config_override_file", "entrypoint_source"):
            self.assertIn(field, data["constructor"])
        roles = [item["role"] for item in data["inputs"]]
        self.assertEqual(set(roles), set(REGISTERED_INPUT_ROLES))
        self.assertEqual(len(roles), len(REGISTERED_INPUT_ROLES))
        self.assertEqual(
            set(data["raw_to_effective_pairs"]),
            {"equivalent_pairs", "non_equivalent_pairs"},
        )
        # 逐项为 [raw, effective] 或 [raw, effective, note]
        for pair in data["raw_to_effective_pairs"]["equivalent_pairs"]:
            self.assertEqual(len(pair), 2, pair)
        for pair in data["raw_to_effective_pairs"]["non_equivalent_pairs"]:
            self.assertEqual(len(pair), 3, pair)

    def test_json_equivalent_pairs_match_hardcoded_registry_contract(self):
        # JSON 与代码中的声明若重复，必须一致；否则说明存在两份不同定义。
        from structure_reader.model_registry import MODEL_REGISTRY

        data = _support.registry_json()
        if MODEL_KEY not in MODEL_REGISTRY:
            self.skipTest("MODEL_REGISTRY 未包含该模型")
        hard = MODEL_REGISTRY[MODEL_KEY]
        json_pairs = [
            tuple(pair) for pair in data["raw_to_effective_pairs"]["equivalent_pairs"]
        ]
        hard_pairs = [tuple(pair) for pair in hard["raw_to_effective_pairs"]["equivalent_pairs"]]
        self.assertEqual(json_pairs, hard_pairs)
        json_non = [tuple(p) for p in data["raw_to_effective_pairs"]["non_equivalent_pairs"]]
        hard_non = [tuple(p) for p in hard["raw_to_effective_pairs"]["non_equivalent_pairs"]]
        self.assertEqual(json_non, hard_non)
        self.assertEqual(
            [i["role"] for i in data["inputs"]],
            [i["role"] for i in hard["inputs"]],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
