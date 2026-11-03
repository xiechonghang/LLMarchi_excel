import unittest
import numpy as np


class TestM2Onnx(unittest.TestCase):
    def _run(self, g, feeds):
        import onnx
        import onnxruntime as ort
        from structure_reader.onnx_module.build import gb_to_model
        m = gb_to_model(g, g.module_id, "prefill")
        onnx.checker.check_model(m, full_check=True)
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        outs = sess.run(None, feeds)
        return dict(zip([o.name for o in sess.get_outputs()], outs)), m

    def test_hca_windows_history_bias(self):
        import onnx
        from onnxsim import simplify
        from structure_reader.onnx_module.builders_m2 import build_hca_compressor
        rng = np.random.default_rng(0)
        dh, m, hist, seq = 8, 4, 2, 5
        kv = rng.normal(size=(1, seq, dh)).astype(np.float32)
        gate = rng.normal(size=(1, seq, dh)).astype(np.float32)
        pb = rng.normal(size=(m, dh)).astype(np.float32)
        ch = rng.normal(size=(1, hist, dh)).astype(np.float32)
        w = rng.normal(size=(dh,)).astype(np.float32)
        feeds = {"kv": kv, "gate": gate, "buffer_kv": np.zeros((1, 0, dh), np.float32),
                 "buffer_gate": np.zeros((1, 0, dh), np.float32), "position_bias": pb,
                 "compressed_kv": ch, "entry_count": np.array(0, np.int64),
                 "kv_norm_weight": w, "position_ids": np.arange(seq)[None, :].astype(np.int64)}
        out, model = self._run(build_hca_compressor(seq=seq, hist=hist, m=m), feeds)
        self.assertEqual(out["new_compressed"].shape, (1, 1, dh))
        self.assertEqual(int(out["entry_count_post"]), 1)
        np.testing.assert_allclose(out["compressed_post"][:, :hist], ch, rtol=0, atol=0)
        np.testing.assert_allclose(out["buffer_kv_post"], kv[:, -1:], rtol=0, atol=0)
        bias = out["block_bias"]
        self.assertEqual(bias.shape, (1, 1, seq, hist + 1))
        # j >= (pos+1)//4 屏蔽
        for pos in range(seq):
            thr = (pos + 1) // m
            np.testing.assert_array_equal((bias[0, 0, pos] < -1e8), np.arange(hist + 1) >= thr)
        sim = simplify(model)[0]
        onnx.checker.check_model(sim, full_check=True)
        import onnxruntime as ort
        sess = ort.InferenceSession(sim.SerializeToString(), providers=["CPUExecutionProvider"])
        out2 = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
        np.testing.assert_allclose(out2["new_compressed"], out["new_compressed"], rtol=0, atol=1e-5)

    def test_hca_zero_window_keeps_history(self):
        from structure_reader.onnx_module.builders_m2 import build_hca_compressor
        rng = np.random.default_rng(1)
        dh, m, hist, seq = 8, 4, 3, 2
        kv = rng.normal(size=(1, seq, dh)).astype(np.float32)
        ch = rng.normal(size=(1, hist, dh)).astype(np.float32)
        feeds = {"kv": kv, "gate": rng.normal(size=(1, seq, dh)).astype(np.float32),
                 "buffer_kv": np.zeros((1, 0, dh), np.float32),
                 "buffer_gate": np.zeros((1, 0, dh), np.float32),
                 "position_bias": rng.normal(size=(m, dh)).astype(np.float32),
                 "compressed_kv": ch, "entry_count": np.array(2, np.int64),
                 "kv_norm_weight": rng.normal(size=(dh,)).astype(np.float32),
                 "position_ids": np.arange(seq)[None, :].astype(np.int64)}
        out, _ = self._run(build_hca_compressor(seq=seq, hist=hist, m=m), feeds)
        self.assertEqual(out["new_compressed"].shape, (1, 0, dh))
        self.assertEqual(int(out["entry_count_post"]), 2)
        np.testing.assert_allclose(out["compressed_post"], ch, rtol=0, atol=0)
        # 历史存在时 compressed_len>0，block_bias 仍按阈值生成（非 None）
        self.assertEqual(out["block_bias"].shape, (1, 1, seq, hist))
        # query_length==1 时 HF block_bias=None → 消费侧零扩展
        feeds1 = dict(feeds)
        feeds1["kv"] = feeds["kv"][:, :1]
        feeds1["gate"] = feeds["gate"][:, :1]
        feeds1["position_ids"] = np.zeros((1, 1), np.int64)
        out1, _ = self._run(build_hca_compressor(seq=1, hist=hist, m=m), feeds1)
        np.testing.assert_allclose(out1["block_bias"], 0.0, rtol=0, atol=0)

    def test_hash_table_selection_not_topk(self):
        from structure_reader.onnx_module.builders_m2 import build_hash_route
        table = np.array([[1, 2], [0, 3], [2, 0], [3, 1], [0, 2], [3, 1]], np.int64)
        ids = np.array([0, 5], np.int64)          # 选 1,2 与 3,1
        flat = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], np.float32)
        w = np.zeros((4, 4), np.float32)
        # 让 expert 0 得分最高：TopK 会选 0，但表选的是 1/2 与 3/1
        w[0] = np.array([10.0, 0, 0, 0], np.float32)
        w[1] = np.array([0, 1.0, 0, 0], np.float32)
        w[2] = np.array([0, 0, 1.0, 0], np.float32)
        w[3] = np.array([0, 0, 0, 1.0], np.float32)
        feeds = {"flat": flat, "input_ids": ids, "tid2eid": table,
                 "router_weight": w}
        out, _ = self._run(build_hash_route(n=2, h=4, e=4, k=2, vocab=6), feeds)
        np.testing.assert_array_equal(out["indices"], table[ids])
        topk = np.argsort(-out["router_scores"], axis=1)[:, :2]
        self.assertFalse(np.array_equal(np.sort(out["indices"], axis=1), np.sort(topk, axis=1)))
        g = np.take_along_axis(out["router_scores"], out["indices"], axis=1)
        exp = g / (g.sum(axis=1, keepdims=True) + 1e-20) * 2.5
        np.testing.assert_allclose(out["weights"], exp, rtol=1e-5, atol=1e-6)

    def test_swa_main_rope_independent_values(self):
        from structure_reader.onnx_module.builders_m2 import build_swa_attention
        rng = np.random.default_rng(2)
        b, t, c, nh, dh, rk = 1, 3, 2, 2, 8, 2
        q = rng.normal(size=(b, nh, t, dh)).astype(np.float32)
        kv = rng.normal(size=(b, 1, t, dh)).astype(np.float32)
        ck = rng.normal(size=(b, 1, c, dh)).astype(np.float32)
        cv = rng.normal(size=(b, 1, c, dh)).astype(np.float32)
        mask = np.zeros((1, 1, t, c + t), np.float32)
        feeds = {"q": q, "kv": kv, "cache_keys": ck, "cache_values": cv,
                 "position_ids": np.arange(t)[None, :].astype(np.int64), "mask": mask}
        out, m = self._run(build_swa_attention(seq=t, cache=c, nh=nh, dh=dh, rk=rk), feeds)
        self.assertEqual(out["attn_out"].shape, (b, t, nh, dh))
        self.assertEqual(out["full_keys"].shape[2], c + t)
        self.assertFalse(np.allclose(out["full_keys"], out["full_values"]))
        self.assertEqual(out["keys_post"].shape, (b, 1, c, dh))
        self.assertEqual(out["values_post"].shape, (b, 1, c, dh))
        # 无压缩器输入
        self.assertFalse(any("compressed" in i.name or "entry_count" in i.name for i in m.graph.input))


if __name__ == "__main__":
    unittest.main()
