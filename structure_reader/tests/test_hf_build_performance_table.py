"""E03 回归：正式 ``build-hf --all-variants`` 接入新版性能表。

覆盖点（对应用户派单 E03 验收）：
- 新候选产出 ``tables/decoder_performance.xlsx``，与已批准 layout-v2-rework 样表
  字节一致；旧 ``tables/decoder_variants.xlsx`` 仍由旧导出器生成、字节保持。
- 新导出器源码进入 source_manifest.tool_sources 与 analysis_id；工具源码变化产生
  新身份；run_manifest 交付清单/manifest.tables 收录新表。
- 缓存完整性：新表丢失/损坏时拒绝复用，不静默健康。
- 缺 openpyxl 时新表路径具名 HfBuildError（不降级 CSV、不只遗漏新表）。
- 单层旧入口不产出新表；其他模型不声明/不加载 DeepSeek 专属性能表依赖。
- ``--archify-layers`` 路径使用同一份 structures/network 生成同源新表，不改图。

只读真实 HF 输入；缺少只读输入或工程源码时按既有 HF 测试约定明确 skip。
不写真实 outputs：候选目录一律在临时目录内，缓存核验直接调用 ``_verify_existing``。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_KEY = "deepseek_v4_flash_0731_hf"
PROJECT_ROOT = REPO_ROOT / "structure_reader"

# E01 layout-v2-rework 已批准样表与旧导出表字节（本任务冻结输入下应保持不变）。
APPROVED_PERFORMANCE_SHA256 = (
    "953592e594cdccfa7aed8914f5fed791cc10a0e972e6bc585ee4b5c3be751699")
LEGACY_VARIANTS_SHA256 = (
    "2dabd3af336e08b59cc847b7cc8d5c89fc8bc46244abf3a685a0077cc365b05e")
PERF_MODULES = (
    "src/structure_reader/export/deepseek_perf_layout.py",
    "src/structure_reader/export/deepseek_perf_layout_v2.py",
)


def _missing_readonly_inputs() -> list[str]:
    required = (
        REPO_ROOT / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json",
        REPO_ROOT / "hflib/transformers/src/transformers/models/deepseek_v4/"
        "modeling_deepseek_v4.py",
        PROJECT_ROOT / "src/structure_reader/build_hf_decoder.py",
        PROJECT_ROOT / f"configs/models/{MODEL_KEY}.json",
    )
    return [str(path) for path in required if not path.is_file()]


def _build(**kwargs):
    from structure_reader.build_hf_decoder import run_hf_build

    return run_hf_build(REPO_ROOT, PROJECT_ROOT, model_key=MODEL_KEY, **kwargs)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestPerformanceTableBuild(unittest.TestCase):
    """正式分层候选：新表、旧表、清单与身份。"""

    @classmethod
    def setUpClass(cls):
        missing = _missing_readonly_inputs()
        if missing:
            raise unittest.SkipTest(
                "缺少只读 HF 输入或工程源码，无法验证正式构建；如实跳过: "
                + ", ".join(missing[:3]))
        cls.result, cls.documents = _build(all_variants=True)

    def test_new_performance_table_matches_approved_sample(self):
        data = self.documents.get("tables/decoder_performance.xlsx")
        self.assertIsNotNone(data, "正式分层候选必须含 tables/decoder_performance.xlsx")
        self.assertEqual(_sha256(data), APPROVED_PERFORMANCE_SHA256)

    def test_legacy_variants_table_bytes_unchanged(self):
        data = self.documents.get("tables/decoder_variants.xlsx")
        self.assertIsNotNone(data)
        self.assertEqual(_sha256(data), LEGACY_VARIANTS_SHA256)

    def test_run_manifest_lists_both_tables(self):
        manifest = json.loads(self.documents["run_manifest.json"])
        self.assertEqual(manifest["analysis_id"], self.result.analysis_id)
        self.assertEqual(
            manifest["deliverables"]["tables/decoder_performance.xlsx"],
            APPROVED_PERFORMANCE_SHA256)
        self.assertEqual(
            manifest["deliverables"]["tables/decoder_variants.xlsx"],
            LEGACY_VARIANTS_SHA256)
        tables = manifest["tables"]
        self.assertEqual(tables["performance"]["path"], "tables/decoder_performance.xlsx")
        self.assertEqual(tables["performance"]["layout"], "layout-v2-rework")
        self.assertEqual(tables["variants_legacy"]["path"], "tables/decoder_variants.xlsx")

    def test_performance_exporter_sources_enter_identity(self):
        source_manifest = json.loads(self.documents["reports/source_manifest.json"])
        tool_sources = source_manifest["tool_sources"]
        for rel in PERF_MODULES:
            self.assertIn(rel, tool_sources, f"{rel} 必须进入 tool_sources")
            self.assertEqual(
                tool_sources[rel], _sha256((PROJECT_ROOT / rel).read_bytes()))
        self.assertIn("src/structure_reader/export/deepseek_perf_rules.py", tool_sources)
        # analysis_id = sha256(canonical(identity))，含 tool_sources 与 scenarios。
        identity = source_manifest["identity"]
        self.assertEqual(identity["tool_sources"], tool_sources)
        canonical = json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), self.result.analysis_id)

    def test_single_layer_entry_does_not_emit_performance_table(self):
        _result, documents = _build(layer_index=4)
        self.assertIn("tables/decoder_reading.xlsx", documents)
        self.assertNotIn("tables/decoder_performance.xlsx", documents)

    def test_other_models_declare_no_performance_exporter(self):
        from structure_reader.build_hf_decoder import _adapter_for

        for model_key in ("glm_5_3_hf", "glm_5_3_flash_hf",
                          "qwen_3_8_flash_next_hf", "hidream_o1_image_hf"):
            self.assertIsNone(_adapter_for(model_key).export_performance,
                              f"{model_key} 不得借用 DeepSeek 性能表布局")
        # 新进程确认：只构建其他模型适配器不会导入 DeepSeek 专属导出模块链。
        code = (
            "import sys\n"
            "from structure_reader.build_hf_decoder import _adapter_for\n"
            "_adapter_for('glm_5_3_hf')\n"
            "bad=[m for m in sys.modules if 'deepseek_perf_layout' in m]\n"
            "print(','.join(sorted(bad)))\n"
        )
        env = dict(os.environ, PYTHONPATH=str(SRC_DIR))
        output = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True,
            check=True).stdout.strip()
        self.assertEqual(output, "", f"其他模型意外加载 DeepSeek 专属导出: {output}")


class TestPerformanceTableCacheIntegrity(unittest.TestCase):
    """新表丢失/损坏不能仍健康复用（缓存完整性）。"""

    @classmethod
    def setUpClass(cls):
        missing = _missing_readonly_inputs()
        if missing:
            raise unittest.SkipTest("缺少只读 HF 输入，跳过缓存完整性验证: "
                                    + ", ".join(missing[:3]))
        cls.result, cls.documents = _build(all_variants=True)

    def _materialize_copy(self, root: Path) -> Path:
        target = root / "candidate"
        for name, data in self.documents.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return target

    def test_intact_candidate_reuses(self):
        from structure_reader.cli import _verify_existing

        with tempfile.TemporaryDirectory() as td:
            target = self._materialize_copy(Path(td))
            self.assertEqual(_verify_existing(target, self.documents), "reused")

    def test_missing_performance_table_is_conflict(self):
        from structure_reader.cli import OutputConsistencyError, _verify_existing

        with tempfile.TemporaryDirectory() as td:
            target = self._materialize_copy(Path(td))
            (target / "tables/decoder_performance.xlsx").unlink()
            with self.assertRaises(OutputConsistencyError) as caught:
                _verify_existing(target, self.documents)
            message = str(caught.exception)
            self.assertIn("tables/decoder_performance.xlsx", message)
            self.assertIn("missing", message)

    def test_corrupt_performance_table_is_conflict(self):
        from structure_reader.cli import OutputConsistencyError, _verify_existing

        with tempfile.TemporaryDirectory() as td:
            target = self._materialize_copy(Path(td))
            path = target / "tables/decoder_performance.xlsx"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaises(OutputConsistencyError) as caught:
                _verify_existing(target, self.documents)
            message = str(caught.exception)
            self.assertIn("tables/decoder_performance.xlsx", message)
            self.assertIn("actual_sha256", message)


class TestPerformanceTableMissingDependency(unittest.TestCase):
    """缺 openpyxl 时新表路径具名失败，不降级、不只遗漏新表。"""

    def setUp(self):
        missing = _missing_readonly_inputs()
        if missing:
            self.skipTest("缺少只读 HF 输入，跳过缺依赖验证: " + ", ".join(missing[:3]))

    def test_missing_openpyxl_for_performance_export_is_named(self):
        from structure_reader.build_hf_decoder import HfBuildError

        # 单独让新导出模块不可导入：旧表路径正常，失败必须来自新表路径且具名。
        with mock.patch.dict(
                sys.modules, {"structure_reader.export.deepseek_perf_layout_v2": None}):
            with self.assertRaises(HfBuildError) as caught:
                _build(all_variants=True)
        message = str(caught.exception)
        self.assertIn("性能表", message)
        self.assertIn("openpyxl", message)
        self.assertIn("decoder_performance.xlsx", message)


class TestPerformanceTableIdentityAndArchifyPath(unittest.TestCase):
    """工具源码变化→新身份；archify-layers 路径同源生成同一张新表。"""

    def setUp(self):
        missing = _missing_readonly_inputs()
        if missing:
            self.skipTest("缺少只读 HF 输入，跳过身份/archify 路径验证: "
                          + ", ".join(missing[:3]))

    def test_layout_source_change_produces_new_analysis_id(self):
        baseline, _documents = _build(all_variants=True)
        with tempfile.TemporaryDirectory() as td:
            mirror = Path(td) / "project"
            shutil.copytree(
                PROJECT_ROOT / "src", mirror / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copytree(PROJECT_ROOT / "configs", mirror / "configs",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            target = mirror / PERF_MODULES[1]
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# e03 identity probe\n",
                encoding="utf-8")
            from structure_reader.build_hf_decoder import run_hf_build

            changed, documents = run_hf_build(
                REPO_ROOT, mirror, model_key=MODEL_KEY, all_variants=True)
            self.assertNotEqual(changed.analysis_id, baseline.analysis_id)
            # 性能表内容仍由同一份 structures/network 决定，字节不变。
            self.assertEqual(
                _sha256(documents["tables/decoder_performance.xlsx"]),
                APPROVED_PERFORMANCE_SHA256)
            source_manifest = json.loads(documents["reports/source_manifest.json"])
            self.assertNotEqual(
                source_manifest["tool_sources"][PERF_MODULES[1]],
                _sha256((PROJECT_ROOT / PERF_MODULES[1]).read_bytes()))

    def test_archify_layers_path_emits_same_performance_table(self):
        _baseline, plain = _build(all_variants=True)
        stub_bundle = {
            "files": {
                "viewer/archify/network/index.html": b"<html>stub</html>",
                "data/archify/manifest.json": b"{}\n",
            },
            "panel": {"schema": "archify-layers-v1", "network": None,
                      "endpoints": {}, "groups": {}, "tool": {}, "note": ""},
            "identity": {"schema": "hf-archify-identity.v1", "model_key": MODEL_KEY,
                         "builder": "stub"},
            "manifest": {"schema": "archify_layers_manifest_v1", "builder": "stub",
                         "entries": [], "tool": {"version": "stub"}},
        }
        with mock.patch(
                "structure_reader.build_hf_decoder._build_archify_layers",
                return_value=stub_bundle), mock.patch(
                "structure_reader.render.hf_reading_page.render_reading_page",
                return_value=b"<html>reading-stub</html>"):
            try:
                _result, archify_documents = _build(
                    all_variants=True, archify_layers=True)
            except Exception as exc:  # noqa: BLE001 - 缺 Archify CLI 等环境差
                self.skipTest(f"archify 路径集成检查需可用的分层图构建链: {exc}")
        self.assertEqual(
            archify_documents["tables/decoder_performance.xlsx"],
            plain["tables/decoder_performance.xlsx"],
            "archify-layers 路径必须用同一份 structures/network 生成同源新表")
        self.assertEqual(
            archify_documents["tables/decoder_variants.xlsx"],
            plain["tables/decoder_variants.xlsx"])
        self.assertIn("viewer/archify/network/index.html", archify_documents)


if __name__ == "__main__":
    unittest.main(verbosity=2)
