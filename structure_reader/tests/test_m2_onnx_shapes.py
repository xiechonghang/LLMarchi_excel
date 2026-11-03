import unittest
import numpy as np

from structure_reader.onnx_module.build import display_dim, set_static_dims


def _ctx(scope, tensors, params, rope, phase_id="prefill"):
    from structure_reader.onnx_module.builders import Ctx
    phase = {"id": phase_id, "tensors": [{"id": k, "shape": v} for k, v in tensors.items()],
             "parameters": [{"id": k, "value": v} for k, v in params.items()], "operators": []}
    return Ctx({"scope": scope}, phase, {}, rope, rope)


def _run(ctx, builder, feeds):
    import onnxruntime as ort
    from structure_reader.onnx_module.build import build_module
    res = build_module(ctx, "m", builder)
    outs = {}
    for key in ("model", "sim"):
        s = ort.InferenceSession(res[key].SerializeToString(), providers=["CPUExecutionProvider"])
        outs[key] = dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds)))
    return outs


def _rms(x, w, eps):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def _rope(x, pos, inv, rk, sign=1.0):
    # x [B,1,T,Dh] / [B,Nh,T,Dh]；last rk interleaved
    cos = np.cos(pos[..., None] * inv)[None, None, :, :]
    sin = np.sin(pos[..., None] * inv)[None, None, :, :]
    last = x[..., -rk:]
    x1, x2 = last[..., 0::2], last[..., 1::2]
    r1 = x1 * cos - sign * x2 * sin
    r2 = sign * x1 * sin + x2 * cos
    rot = np.stack([r1, r2], -1).reshape(*last.shape)
    return np.concatenate([x[..., :-rk], rot], -1)


class TestOnnxShapes(unittest.TestCase):
    def test_static_dims(self):
        set_static_dims({"H": 4096, "Dh": 512, "Rq": 1024, "Rk": 64, "Nh": 32,
                         "Nhc": 4, "G": 8, "Ro": 256, "M": 128, "W": 128,
                         "E": 256, "K": 6, "Es": 2048, "Vocab": 129280})
        self.assertEqual(display_dim("H"), 4096)
        self.assertEqual(display_dim("Dh"), 512)
        self.assertEqual(display_dim("Rk÷2"), 32)
        self.assertEqual(display_dim("2Dh"), 1024)
        self.assertEqual(display_dim("Nhc×H"), 16384)
        self.assertEqual(display_dim("1"), 1)
        self.assertEqual(display_dim("B×T"), "batch_size×sequence_length")
        self.assertEqual(display_dim("min(P+T,W-1)"),
                         "min(past_sequence_length+sequence_length,sliding_window-1)")
        self.assertEqual(display_dim("Tbuf'"), "buffer_length_after")
        self.assertEqual(display_dim("unk__3"), "unresolved_dimension_3")

    def test_kv_numeric(self):
        from structure_reader.onnx_module.builders import build_attn_kv
        B, T, Dh, H, Rk = 2, 3, 6, 6, 4
        rng = np.random.default_rng(11)
        x = rng.normal(size=(B, T, H)).astype(np.float32)
        w = rng.normal(size=(Dh, H)).astype(np.float32)
        wn = rng.normal(size=(Dh,)).astype(np.float32)
        pos = np.arange(T, dtype=np.int64)[None, :]
        inv = (1.0 / (10000.0 ** (np.arange(0, Rk, 2) / Rk))).astype(np.float32)
        rope = {"inv_freq": inv, "dim": Rk, "attention_factor": 1.0,
                "rope_type": "default", "theta": 10000}
        tensors = {"attn_normed": ["B", "T", "H"], "kv_weight_attn": ["Dh", "H"],
                   "kv_norm_weight": ["Dh"], "kv_rope": ["B", "1", "T", "Dh"],
                   "position_ids": ["1", "T"]}
        ctx = _ctx({"attention_type": "sliding_attention"}, tensors,
                   {"Dh": Dh, "Rk": Rk, "H": H, "EpsN": 1e-6}, rope)
        out = _run(ctx, build_attn_kv, {"attn_normed": x, "kv_weight_attn": w,
                                        "kv_norm_weight": wn, "position_ids": pos})
        kv = _rms(x @ w.T, wn, 1e-6)[:, None, :, :]
        exp = _rope(kv, pos, inv, Rk)
        for key in ("model", "sim"):
            self.assertEqual(out[key]["kv_rope"].shape, (B, 1, T, Dh))
            np.testing.assert_allclose(out[key]["kv_rope"], exp, rtol=1e-5, atol=1e-5)

    def test_decode_t1_nonzero_position(self):
        from structure_reader.onnx_module.builders import build_attn_kv, build_attn_out
        B, T, Dh, H, Rk = 2, 1, 4, 7, 4
        rng = np.random.default_rng(21)
        pos = np.array([[7]], np.int64)
        inv = (1.0 / (10000.0 ** (np.arange(0, Rk, 2) / Rk))).astype(np.float32)
        rope = {"inv_freq": inv, "dim": Rk, "attention_factor": 1.0,
                "rope_type": "default", "theta": 10000}
        x = rng.normal(size=(B, T, H)).astype(np.float32)
        w = rng.normal(size=(Dh, H)).astype(np.float32)
        wn = rng.normal(size=(Dh,)).astype(np.float32)
        tensors = {"position_ids": ["1", "T"], "attn_normed": ["B", "T", "H"],
                   "kv_weight_attn": ["Dh", "H"], "kv_norm_weight": ["Dh"],
                   "kv_rope": ["B", "1", "T", "Dh"]}
        ctx = _ctx({"attention_type": "sliding_attention"}, tensors,
                   {"Dh": Dh, "Rk": Rk, "H": H, "EpsN": 1e-6}, rope, "decode")
        out = _run(ctx, build_attn_kv, {"attn_normed": x, "kv_weight_attn": w,
                                        "kv_norm_weight": wn, "position_ids": pos})
        exp = _rope(_rms(x @ w.T, wn, 1e-6)[:, None, :, :], pos, inv, Rk)
        for key in ("model", "sim"):
            np.testing.assert_allclose(out[key]["kv_rope"], exp, rtol=1e-5, atol=1e-5)
            self.assertEqual(out[key]["kv_rope"].shape, (B, 1, T, Dh))

        Nh, G, Ro = 3, 3, 2
        x2 = rng.normal(size=(B, T, Nh, Dh)).astype(np.float32)
        oa = rng.normal(size=(G * Ro, Nh * Dh // G)).astype(np.float32)
        ob = rng.normal(size=(Dh, G * Ro)).astype(np.float32)
        tensors2 = {"position_ids": ["1", "T"], "attn_head_out": ["B", "T", "Nh", "Dh"],
                    "attn_rope_inv": ["B", "T", "Nh", "Dh"],
                    "attn_grouped": ["B", "T", "G", "Nh×Dh÷G"],
                    "o_a_out": ["B", "T", "G×Ro"], "o_a_weight": ["G×Ro", "Nh×Dh÷G"],
                    "o_b_weight": ["Dh", "G×Ro"], "attn_block_output": ["B", "T", "Dh"]}
        ctx2 = _ctx({"attention_type": "sliding_attention"}, tensors2,
                    {"Nh": Nh, "Dh": Dh, "G": G, "Ro": Ro, "H": Dh, "Rk": Rk}, rope, "decode")
        out2 = _run(ctx2, build_attn_out, {"attn_head_out": x2, "o_a_weight": oa,
                                           "o_b_weight": ob, "position_ids": pos})
        h = _rope(x2.transpose(0, 2, 1, 3), pos, inv, Rk, sign=-1.0).transpose(0, 2, 1, 3)
        wg = oa.reshape(G, Ro, Nh * Dh // G)
        y = np.einsum("btgi,gri->btgr", h.reshape(B, T, G, Nh * Dh // G), wg)
        exp2 = y.reshape(B, T, G * Ro) @ ob.T
        for key in ("model", "sim"):
            np.testing.assert_allclose(out2[key]["attn_block_output"], exp2, rtol=1e-5, atol=1e-5)
            self.assertEqual(out2[key]["attn_block_output"].shape, (B, T, Dh))

    def test_out_inverse_rope_grouped(self):
        from structure_reader.onnx_module.builders import build_attn_out
        B, T, Nh, Dh, G, Ro, H, Rk = 2, 3, 2, 6, 3, 2, 6, 4
        rng = np.random.default_rng(12)
        x = rng.normal(size=(B, T, Nh, Dh)).astype(np.float32)
        oa = rng.normal(size=(G * Ro, Nh * Dh // G)).astype(np.float32)
        ob = rng.normal(size=(H, G * Ro)).astype(np.float32)
        pos = np.arange(T, dtype=np.int64)[None, :]
        inv = (1.0 / (10000.0 ** (np.arange(0, Rk, 2) / Rk))).astype(np.float32)
        rope = {"inv_freq": inv, "dim": Rk, "attention_factor": 1.0,
                "rope_type": "default", "theta": 10000}
        tensors = {"position_ids": ["1", "T"], "attn_head_out": ["B", "T", "Nh", "Dh"],
                   "attn_rope_inv": ["B", "T", "Nh", "Dh"],
                   "attn_grouped": ["B", "T", "G", "Nh×Dh÷G"],
                   "o_a_out": ["B", "T", "G×Ro"], "o_a_weight": ["G×Ro", "Nh×Dh÷G"],
                   "o_b_weight": ["H", "G×Ro"], "attn_block_output": ["B", "T", "H"]}
        ctx = _ctx({"attention_type": "sliding_attention"}, tensors,
                   {"Nh": Nh, "Dh": Dh, "G": G, "Ro": Ro, "H": H, "Rk": Rk}, rope)
        out = _run(ctx, build_attn_out, {"attn_head_out": x, "o_a_weight": oa,
                                         "o_b_weight": ob, "position_ids": pos})
        h = x.transpose(0, 2, 1, 3)
        h = _rope(h, pos, inv, Rk, sign=-1.0)
        h = h.transpose(0, 2, 1, 3)
        grouped = h.reshape(B, T, G, Nh * Dh // G)
        wg = oa.reshape(G, Ro, Nh * Dh // G)
        y = np.einsum("btgi,gri->btgr", grouped, wg)
        exp = (y.reshape(B, T, G * Ro) @ ob.T)
        for key in ("model", "sim"):
            self.assertEqual(out[key]["attn_block_output"].shape, (B, T, H))
            np.testing.assert_allclose(out[key]["attn_block_output"], exp, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
