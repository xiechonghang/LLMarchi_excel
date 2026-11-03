"""T01 回归：xlsx 导出器导入失败时，单层与四变体都必须具名 HfBuildError。

背景：operator_workbook 在模块顶层导入 openpyxl。若导出器 import 落在
``_export_workbook`` 的 try 之外，缺 openpyxl 时单层路径会抛 ModuleNotFoundError，
绕过具名失败隔离（四变体因在包装内调用导出器而不受影响）。本回归同时约束两条
既有构建入口：失败时不返回候选字节、不落盘。

只在临时工程根上运行，不写真实 outputs；不导入/执行 HF 模型。缺少只读 HF 输入
或工程源码（如未准备 models/hflib 的新克隆）时按既有 HF 测试约定明确 skip，
不以 HfConfigError 误报为缺 openpyxl 分支的结果。
"""

from __future__ import annotations

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


def _missing_readonly_inputs() -> list[str]:
    """沿用既有 HF 就绪检查，并补上本回归所需的工程源码/配置。"""
    root = REPO_ROOT
    required = (
        root / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json",
        root / "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py",
        root / "structure_reader/src/structure_reader/build_hf_decoder.py",
        root / f"structure_reader/configs/models/{MODEL_KEY}.json",
    )
    return [str(path) for path in required if not path.is_file()]


class TestMissingOpenpyxlNamedFailure(unittest.TestCase):
    """模拟 openpyxl 缺失：两条构建路径均具名失败且不产生候选目录。"""

    def setUp(self):
        missing = _missing_readonly_inputs()
        if missing:
            self.skipTest(
                "缺少只读 HF 输入或工程源码，无法验证缺 openpyxl 分支；如实跳过: "
                + ", ".join(missing[:3])
                + (f"（共 {len(missing)} 项）" if len(missing) > 3 else "")
            )

    def _build_with_openpyxl_blocked(self, **kwargs) -> str:
        from structure_reader.build_hf_decoder import HfBuildError, run_hf_build

        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "project"
            project.mkdir()
            (project / "src").symlink_to(
                REPO_ROOT / "structure_reader/src", target_is_directory=True)
            (project / "configs").symlink_to(
                REPO_ROOT / "structure_reader/configs", target_is_directory=True)
            # 导出器可能已被其它测试导入；清掉以便重新执行其顶层 openpyxl import。
            saved = sys.modules.pop("structure_reader.export.operator_workbook", None)
            try:
                with mock.patch.dict(sys.modules, {"openpyxl": None}):
                    with self.assertRaises(HfBuildError) as caught:
                        run_hf_build(REPO_ROOT, project, model_key=MODEL_KEY, **kwargs)
            finally:
                if saved is not None:
                    sys.modules["structure_reader.export.operator_workbook"] = saved
            self.assertFalse(
                (project / "outputs").exists(),
                "失败构建不得落盘候选（run_hf_build 只返回字节，由 CLI 提交）",
            )
        return str(caught.exception)

    def test_single_layer_missing_openpyxl_is_named(self):
        message = self._build_with_openpyxl_blocked(layer_index=4)
        self.assertIn("xlsx 导出必需依赖缺失", message)
        self.assertIn("openpyxl", message)

    def test_variants_missing_openpyxl_is_named(self):
        message = self._build_with_openpyxl_blocked(all_variants=True)
        self.assertIn("xlsx 导出必需依赖缺失", message)
        self.assertIn("openpyxl", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
