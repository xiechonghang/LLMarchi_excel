"""M2 门禁回归：ONNX 模块的 checker/shape/simplify 失败必须阻断整组构建。

T05 接口重构后，模型事实由 Adapter 声明；本测试不再调用私有
``_run_variants(..., model_key)``，改走公共 ``run_hf_build(all_variants=True)``
的真实分层构建路径。``_load_onnx_stack`` 只被替换为假导入（避免测试环境必须安装
可选的 onnx/onnxsim/onnxruntime），而逐模块的 checker/shape/simplify 校验仍是
真实实现（``_build_onnx_document``），负例仍要求首个 SWA+Hash 组被拒绝。
"""
import unittest
from pathlib import Path
from unittest import mock

from structure_reader.build_hf_decoder import HfBuildError, run_hf_build

PROJECT = Path(__file__).resolve().parents[1]
WS = PROJECT.parent
MODEL_KEY = "deepseek_v4_flash_0731_hf"


def _fake(checks):
    mods = {}
    for i in range(16):
        mid = "attn_q" if i == 0 else f"m{i}"
        mods[mid] = {"status": "ok", "checks": checks if i == 0 else
                     {"checker": "ok", "shape_inference": "ok",
                      "simplify": {"returned": True}}}
    return {"files": {}, "bundle": {},
            "report": {"modules": {"prefill": mods, "decode": mods}}}


def _build_with_fake_stack(checks):
    """公共入口 + 假 ONNX 导入；逐模块错误检查仍走真实 _build_onnx_document。"""
    fake_stack = ({"enabled": True}, lambda *args, **kwargs: _fake(checks))
    with mock.patch("structure_reader.build_hf_decoder._load_onnx_stack",
                    return_value=fake_stack):
        return run_hf_build(WS, PROJECT, model_key=MODEL_KEY,
                            all_variants=True, onnx_modules=True)


class TestM2Guard(unittest.TestCase):
    def test_first_group_swa_simplify_failure_blocks(self):
        bad = {"checker": "ok", "shape_inference": "ok",
               "simplify": {"returned": False}}
        with self.assertRaises(HfBuildError) as cm:
            _build_with_fake_stack(bad)
        msg = str(cm.exception)
        self.assertIn("swa-hash", msg)
        self.assertIn("attn_q", msg)
        self.assertIn("onnxsim", msg)

    def test_first_group_swa_shape_failure_blocks(self):
        bad = {"checker": "ok", "shape_inference": "failed: bad",
               "simplify": {"returned": True}}
        with self.assertRaises(HfBuildError) as cm:
            _build_with_fake_stack(bad)
        self.assertIn("swa-hash", str(cm.exception))
        self.assertIn("checker/shape", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
