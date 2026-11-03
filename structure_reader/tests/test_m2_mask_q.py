import unittest
import numpy as np


def _ctx(scope, tensors, params, rope):
    from structure_reader.onnx_module.builders import Ctx
    phase = {"id": "prefill", "tensors": [{"id": k, "shape": v} for k, v in tensors.items()],
             "parameters": [{"id": k, "value": v} for k, v in params.items()], "operators": []}
    return Ctx({"scope": scope}, phase, {}, rope, rope)


ROPE = {"inv_freq": np.array([1.0], np.float32), "dim": 2, "attention_factor": 1.0,
        "rope_type": "default", "theta": 10000}


class TestMaskQProduction(unittest.TestCase):
    def _run(self, ctx, builder, feeds):
        import onnxruntime as ort
        from structure_reader.onnx_module.build import build_module
        res = build_module(ctx, "m", builder)
        outs = {}
        for key in ("model", "sim"):
            s = ort.InferenceSession(res[key].SerializeToString(), providers=["CPUExecutionProvider"])
            outs[key] = dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds)))
        return outs, res

    def test_swa_mask_generated(self):
        from structure_reader.onnx_module.builders_m2 import build_swa_mask
        W = 4
        tensors = {"mask_model": ["B", "1", "T", "Lmask"]}
        ctx = _ctx({"attention_type": "sliding_attention"}, tensors, {"W": W}, ROPE)
        feeds = {"B": np.array(2, np.int64), "T": np.array(3, np.int64),
                 "cache_cumulative_length": np.array(10, np.int64)}
        out, res = self._run(ctx, build_swa_mask, feeds)
        L = min(10, W - 1) + 3
        self.assertEqual(out["model"]["mask_model"].shape, (2, 1, 3, L))
        base = max(10 - (W - 1), 0)
        keys = np.arange(L) + base
        q = np.arange(10, 13)
        ok = (keys[None, :] <= q[:, None]) & (keys[None, :] > q[:, None] - W)
        exp = np.where(ok, 0.0, -np.inf).astype(np.float32)[None, None]
        for key in ("model", "sim"):
            np.testing.assert_array_equal(np.isneginf(out[key]["mask_model"]), np.broadcast_to(~ok[None, None], (2, 1, 3, L)))
        self.assertFalse(np.isfinite(out["model"]["mask_model"]).all())

    def test_hca_mask_and_bias(self):
        from structure_reader.onnx_module.builders_m2 import build_hca_mask
        W = 4
        tensors = {"mask_model": ["B", "1", "T", "Lmask"], "block_bias": ["B", "1", "T", "Ncomp"],
                   "mask_ext": ["B", "1", "T", "Lmask+Ncomp"]}
        ctx = _ctx({"attention_type": "heavily_compressed_attention"}, tensors, {"W": W}, ROPE)
        feeds = {"B": np.array(1, np.int64), "T": np.array(2, np.int64),
                 "cache_cumulative_length": np.array(5, np.int64),
                 "block_bias": np.array([[[[0.0, -np.inf, 0.0],
                                           [0.0, -np.inf, 0.0]]]], np.float32)}
        out, _ = self._run(ctx, build_hca_mask, feeds)
        L = min(5, W - 1) + 2
        self.assertEqual(out["model"]["mask_ext"].shape, (1, 1, 2, L + 3))
        np.testing.assert_allclose(out["model"]["mask_ext"][..., :L], out["model"]["mask_model"])
        np.testing.assert_array_equal(np.isneginf(out["model"]["mask_ext"][..., L:]),
                                      np.isneginf(feeds["block_bias"][:, :, :2, :]))

    def test_q_group_by_scope(self):
        import json
        from pathlib import Path
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.onnx_module.builders import Ctx
        from structure_reader.onnx_module.build import build_module
        from structure_reader.onnx_module.builders import build_attn_q
        from structure_reader.onnx_module.modeling import derive_main_rope, derive_compress_rope
        ws = Path(__file__).resolve().parents[2]
        cfg = json.loads((ws / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json").read_text())
        for idx, group in ((0, "rope_main"), (3, "rope_compress")):
            doc = build_reading_document(ws, idx)
            ph = doc["phases"][0]
            ctx = Ctx(doc, ph, cfg, derive_main_rope(cfg), derive_compress_rope(cfg))
            res = build_module(ctx, "attn_q", build_attn_q)
            labels = {k: v["label"] for k, v in res["mapping"].items()}
            self.assertIn(group, labels)
            if group == "rope_main":
                self.assertIn("main", labels[group])
                self.assertNotIn("YaRN", labels[group])
            else:
                self.assertIn("YaRN", labels[group])


if __name__ == "__main__":
    unittest.main()
