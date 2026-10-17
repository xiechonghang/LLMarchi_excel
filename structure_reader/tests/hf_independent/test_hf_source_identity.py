"""Input-identity checks for the independent HF decoder expectations.

Verifies that the pinned inputs (model config + Transformers checkout) still
hash to the identities this independent expectation was derived from, and that
input files are located as separate sources (model package vs. Transformers
source), never mixed.
"""

import unittest
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import hf_expectations as hx  # noqa: E402


class SourceIdentityTests(unittest.TestCase):
    def test_all_pinned_sources_exist(self):
        for name, path in hx.SOURCE_PATHS.items():
            with self.subTest(source=name):
                self.assertTrue(path.is_file(), f"missing pinned source: {path}")

    def test_frozen_sha256_identity(self):
        for name, expected in hx.FROZEN_SHA256.items():
            with self.subTest(source=name):
                actual = hx.sha256_file(hx.SOURCE_PATHS[name])
                self.assertEqual(
                    actual,
                    expected,
                    f"{name} identity drifted; independent expectations must be re-derived",
                )

    def test_model_config_is_model_type_deepseek_v4(self):
        raw = hx.load_raw_config()
        self.assertEqual(raw["model_type"], "deepseek_v4")
        self.assertEqual(raw["architectures"], ["DeepseekV4ForCausalLM"])

    def test_config_and_transformers_sources_are_separate_roots(self):
        config = hx.SOURCE_PATHS["config"].resolve()
        modeling = hx.SOURCE_PATHS["modeling"].resolve()
        models_root = (hx.WORKSPACE_ROOT / "models").resolve()
        hflib_root = (hx.WORKSPACE_ROOT / "hflib").resolve()
        # config lives under models/, modeling under hflib/transformers
        self.assertTrue(str(config).startswith(str(models_root)), config)
        self.assertFalse(str(config).startswith(str(hflib_root)), config)
        self.assertTrue(str(modeling).startswith(str(hflib_root)), modeling)
        self.assertFalse(str(modeling).startswith(str(models_root)), modeling)

    def test_model_inference_reference_is_not_the_hf_source(self):
        """The model package's inference/model.py must not be used as an HF fact source."""
        reference = hx.MODEL_DIR / "inference" / "model.py"
        self.assertTrue(reference.is_file(), "expected a reference inference/model.py to exist")
        self.assertNotEqual(reference.resolve(), hx.SOURCE_PATHS["modeling"].resolve())
        # independent facts must never be read from the reference package
        for path in hx.SOURCE_PATHS.values():
            self.assertNotIn("inference", path.parts)


if __name__ == "__main__":
    unittest.main()
