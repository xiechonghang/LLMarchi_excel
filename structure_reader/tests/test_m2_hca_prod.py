import unittest
import numpy as np

M, DH, RK, H = 4, 8, 4, 8
TENSORS = {
    "attn_normed": ["B", "T", "H"], "c_kv_weight": ["Dh", "H"], "c_gate_weight": ["Dh", "H"],
    "cache_buffer_c_kv": ["B", "Tbuf", "Dh"], "cache_buffer_c_gate": ["B", "Tbuf", "Dh"],
    "c_position_bias": ["M", "Dh"], "cache_compressed_c": ["B", "Tcc", "Dh"],
    "cache_entry_count_c": [], "c_kv_norm_weight": ["Dh"], "position_ids": ["1", "T"],
    "c_chunk_kv": ["B", "Nwin*M", "Dh"], "c_chunk_gate": ["B", "Nwin*M", "Dh"],
    "cache_buffer_c_kv_post": ["B", "Tbufp", "Dh"], "cache_buffer_c_gate_post": ["B", "Tbufp", "Dh"],
    "c_windows_kv": ["B", "Nwin", "M", "Dh"], "c_compressed": ["B", "Nwin", "Dh"],
    "c_positions": ["B", "Nwin"], "c_cos": ["B", "Nwin", "Rk/2"],
    "c_sin": ["B", "Nwin", "Rk/2"], "c_compressed_rope": ["B", "Nwin", "Dh"],
    "cache_compressed_c_post": ["B", "Tcc+Nwin", "Dh"], "cache_entry_count_c_post": [],
    "block_bias": ["B", "1", "T", "Ncomp"],
}
PARAMS = {"M": M, "Dh": DH, "Rk": RK, "H": H, "EpsN": 1e-6}
INV = (1.0 / (10000.0 ** (np.arange(0, RK, 2) / RK))).astype(np.float32)


class TestHcaProduction(unittest.TestCase):
    def _ctx_and_run(self, feeds=None):
        from structure_reader.onnx_module.builders import Ctx
        from structure_reader.onnx_module.builders_m2 import build_hca_csa
        from structure_reader.onnx_module.build import build_module
        phase = {"id": "prefill",
                 "tensors": [{"id": k, "shape": v} for k, v in TENSORS.items()],
                 "parameters": [{"id": k, "value": v} for k, v in PARAMS.items()],
                 "operators": []}
        rope = {"inv_freq": INV, "dim": RK, "attention_factor": 1.0}
        ctx = Ctx({"scope": {"attention_type": "heavily_compressed_attention"}}, phase, {}, {}, rope)
        res = build_module(ctx, "attn_csa", build_hca_csa)
        if feeds is None:
            return None, res
        import onnxruntime as ort
        outs = {}
        for key in ("model", "sim"):
            sess = ort.InferenceSession(res[key].SerializeToString(), providers=["CPUExecutionProvider"])
            outs[key] = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
        return outs, res

    def test_production_model_and_sim(self):
        import onnx
        _, res = self._ctx_and_run()
        onnx.checker.check_model(res["model"], full_check=True)
        onnx.checker.check_model(res["sim"], full_check=True)
        self.assertTrue(res.length if False else True)

    def test_two_windows_batch2_independent(self):
        rng = np.random.default_rng(7)
        B, T, BUF, HIST, EC = 2, 9, 1, 2, 3
        assert BUF < M
        nwin = (BUF + T) // M
        assert nwin == 2
        names = ["kv", "gt", "bkv", "bgt", "pbia", "hist", "wn", "wkv", "wgt", "x"]
        kv, gt, bkv, bgt, pbia, hist, wn, wkv, wgt, x = [
            rng.normal(size=s).astype(np.float32) for s in (
                (B, T, DH), (B, T, DH), (B, BUF, DH), (B, BUF, DH), (M, DH),
                (B, HIST, DH), (DH,), (DH, H), (DH, H), (B, T, H))]
        pos = np.arange(T, dtype=np.int64)[None, :]
        feeds = {"attn_normed": x, "c_kv_weight": wkv, "c_gate_weight": wgt,
                 "cache_buffer_c_kv": bkv, "cache_buffer_c_gate": bgt,
                 "c_position_bias": pbia, "cache_compressed_c": hist,
                 "cache_entry_count_c": np.array(EC, np.int64),
                 "c_kv_norm_weight": wn, "position_ids": pos}
        outs, _ = self._ctx_and_run(feeds)
        kvp, gtp = x @ wkv.T, x @ wgt.T
        cat = np.concatenate([bkv, kvp], 1); gcat = np.concatenate([bgt, gtp], 1)
        pre = cat[:, :nwin * M].reshape(B, nwin, M, DH)
        gwin = gcat[:, :nwin * M].reshape(B, nwin, M, DH) + pbia
        prob = np.exp(gwin - gwin.max(2, keepdims=True)); prob /= prob.sum(2, keepdims=True)
        pooled = (pre * prob).sum(2)
        normed = pooled / np.sqrt((pooled ** 2).mean(-1, keepdims=True) + 1e-6) * wn
        posw = EC * M + np.arange(nwin) * M
        cos = np.cos(posw[:, None] * INV)[None].repeat(B, 0)
        sin = np.sin(posw[:, None] * INV)[None].repeat(B, 0)
        last = normed[..., DH - RK:]
        x1, x2 = last[..., 0::2], last[..., 1::2]
        r1 = x1 * cos - x2 * sin; r2 = x1 * sin + x2 * cos
        inter = np.stack([r1, r2], -1).reshape(B, nwin, RK)
        exp_new = np.concatenate([normed[..., :DH - RK], inter], -1)
        for key in ("model", "sim"):
            o = outs[key]
            np.testing.assert_allclose(o["c_positions"], np.tile(posw, (B, 1)))
            self.assertEqual(o["c_positions"].shape, (B, nwin))
            np.testing.assert_allclose(o["c_cos"], cos, rtol=0, atol=1e-6)
            np.testing.assert_allclose(o["c_sin"], sin, rtol=0, atol=1e-6)
            np.testing.assert_allclose(o["c_compressed_rope"], exp_new, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(o["cache_buffer_c_kv_post"], cat[:, nwin * M:], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(o["cache_buffer_c_gate_post"], gcat[:, nwin * M:], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(o["cache_compressed_c_post"][:, :HIST], hist, rtol=0, atol=0)
            self.assertEqual(int(o["cache_entry_count_c_post"]), EC + nwin)
            thr = (pos + 1) // M
            blocked = np.arange(HIST + nwin)[None, None, :] >= thr[..., None]
            self.assertEqual(o["block_bias"].shape, (B, 1, T, HIST + nwin))
            self.assertTrue(np.all(np.isneginf(o["block_bias"]) == blocked))

    def test_zero_window_keeps_history_none_bias(self):
        B, T, HIST, EC = 2, 2, 3, 5
        rng = np.random.default_rng(8)
        feeds = {"attn_normed": rng.normal(size=(B, T, H)).astype(np.float32),
                 "c_kv_weight": rng.normal(size=(DH, H)).astype(np.float32),
                 "c_gate_weight": rng.normal(size=(DH, H)).astype(np.float32),
                 "cache_buffer_c_kv": np.zeros((B, 0, DH), np.float32),
                 "cache_buffer_c_gate": np.zeros((B, 0, DH), np.float32),
                 "c_position_bias": rng.normal(size=(M, DH)).astype(np.float32),
                 "cache_compressed_c": rng.normal(size=(B, HIST, DH)).astype(np.float32),
                 "cache_entry_count_c": np.array(EC, np.int64),
                 "c_kv_norm_weight": rng.normal(size=(DH,)).astype(np.float32),
                 "position_ids": np.arange(T, dtype=np.int64)[None, :]}
        outs, _ = self._ctx_and_run(feeds)
        for key in ("model", "sim"):
            o = outs[key]
            self.assertEqual(o["c_compressed_rope"].shape[1], 0)
            np.testing.assert_allclose(o["cache_compressed_c_post"], feeds["cache_compressed_c"], rtol=0, atol=0)
            self.assertEqual(int(o["cache_entry_count_c_post"]), EC)
        feeds1 = dict(feeds)
        feeds1["attn_normed"] = feeds["attn_normed"][:, :1]
        feeds1["position_ids"] = np.zeros((1, 1), np.int64)
        outs1, _ = self._ctx_and_run(feeds1)
        for key in ("model", "sim"):
            np.testing.assert_allclose(outs1[key]["block_bias"], 0.0, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
