import unittest
import numpy as np


def _ctx(scope, tensors, params, rope):
    from structure_reader.onnx_module.builders import Ctx
    phase = {"id": "prefill", "tensors": [{"id": k, "shape": v} for k, v in tensors.items()],
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
    return outs, res


ROPE = {"inv_freq": np.array([1.0], np.float32), "dim": 2, "attention_factor": 1.0}


class TestSwaHashProduction(unittest.TestCase):
    def test_swa_cache_tail_independent_values(self):
        from structure_reader.onnx_module.builders_m2 import build_swa_cache
        B, L0, T, W, DH = 2, 2, 3, 8, 6
        rng = np.random.default_rng(3)
        ck = rng.normal(size=(B, 1, L0, DH)).astype(np.float32)
        cv = rng.normal(size=(B, 1, L0, DH)).astype(np.float32) + 1.0  # 前态 keys != values
        kr = rng.normal(size=(B, 1, T, DH)).astype(np.float32)
        tensors = {"cache_keys": ["B", "1", "L0", "Dh"], "cache_values": ["B", "1", "L0", "Dh"],
                   "kv_rope": ["B", "1", "T", "Dh"], "kv_full": ["B", "1", "Lkv", "Dh"],
                   "cache_keys_post": ["B", "1", "Wm1", "Dh"],
                   "cache_values_post": ["B", "1", "Wm1", "Dh"],
                   "cache_cumlen": [], "cache_cumlen_post": []}
        ctx = _ctx({"attention_type": "sliding_attention"}, tensors, {"W": W, "Dh": DH}, ROPE)
        out, res = _run(ctx, build_swa_cache,
                        {"cache_keys": ck, "cache_values": cv, "kv_rope": kr,
                         "cache_cumlen": np.array(2, np.int64)})
        fk = np.concatenate([ck, kr], 2)
        fv = np.concatenate([cv, kr], 2)
        self.assertEqual(out["model"]["cache_keys_post"].shape[2], 5)  # 2+3 (<W-1)
        for key in ("model", "sim"):
            np.testing.assert_allclose(out[key]["kv_full"], fk, rtol=0, atol=0)
            np.testing.assert_allclose(out[key]["cache_keys_post"], fk, rtol=0, atol=0)
            np.testing.assert_allclose(out[key]["cache_values_post"], fv, rtol=0, atol=0)
            self.assertEqual(int(out[key]["cache_cumlen_post"]), 5)

    def test_swa_core_sink_and_av(self):
        from structure_reader.onnx_module.builders_m2 import build_swa_core
        B, NH, T, LKV, DH = 2, 3, 4, 5, 6
        rng = np.random.default_rng(4)
        q = rng.normal(size=(B, NH, T, DH)).astype(np.float32)
        kv = rng.normal(size=(B, 1, LKV, DH)).astype(np.float32)
        mask = np.zeros((1, 1, T, LKV), np.float32)
        sinks = rng.normal(size=(NH,)).astype(np.float32)
        tensors = {"q_rope": ["B", "Nh", "T", "Dh"], "kv_full": ["B", "1", "Lkv", "Dh"],
                   "mask_model": ["1", "1", "T", "Lkv"], "sinks": ["Nh"],
                   "attn_probs": ["B", "Nh", "T", "Lkv"], "attn_head_out": ["B", "T", "Nh", "Dh"]}
        ctx = _ctx({"attention_type": "sliding_attention"}, tensors, {"Nh": NH, "Dh": DH}, ROPE)
        out, _ = _run(ctx, build_swa_core, {"q_rope": q, "kv_full": kv, "mask_model": mask, "sinks": sinks})
        v = np.repeat(kv, NH, axis=1)
        logits = q @ np.transpose(v, (0, 1, 3, 2)) / np.sqrt(DH) + mask
        sink = np.broadcast_to(sinks[None, :, None, None], (B, NH, T, 1))
        cat = np.concatenate([logits, sink], -1)
        prob = np.exp(cat - cat.max(-1, keepdims=True)); prob /= prob.sum(-1, keepdims=True)
        exp = prob[..., :-1] @ v
        for key in ("model", "sim"):
            np.testing.assert_allclose(out[key]["attn_probs"], prob[..., :-1], rtol=1e-5, atol=1e-5)
            self.assertTrue(np.all(out[key]["attn_probs"].sum(-1) <= 1.0))
            self.assertEqual(out[key]["attn_head_out"].shape, (B, T, NH, DH))
            np.testing.assert_allclose(out[key]["attn_head_out"],
                                       exp.transpose(0, 2, 1, 3), rtol=1e-5, atol=1e-5)

    def test_hash_table_not_topk_dispatch(self):
        from structure_reader.onnx_module.builders_m2 import build_hash_route_prod
        B, T, H, E, K, V = 2, 3, 5, 4, 2, 6
        rng = np.random.default_rng(5)
        table = np.array([[1, 2], [0, 3], [2, 0], [3, 1], [0, 2], [3, 1]], np.int64)
        ids = np.array([[0, 5, 2], [1, 4, 3]], np.int64)
        flat = rng.normal(size=(B * T, H)).astype(np.float32)
        w = rng.normal(size=(E, H)).astype(np.float32)
        tensors = {"ffn_normed": ["B", "T", "H"], "ffn_flat": ["B*T", "H"],
                   "input_ids": ["B", "T"], "tid2eid": ["Vocab", "K"],
                   "router_weight": ["E", "H"], "topk_indices": ["B*T", "K"],
                   "topk_weights": ["B*T", "K"], "router_logits": ["B*T", "E"],
                   "router_scores": ["B*T", "E"], "expert_mask": ["E+1", "K", "B*T"],
                   "hit_experts": ["N_hit", "1"]}
        ctx = _ctx({"attention_type": "compressed_sparse_attention", "moe_type": "hash_moe"},
                   tensors, {"E": E, "K": K, "H": H, "Scl": 2.5, "Eps": 1e-6}, ROPE)
        normed = flat.reshape(B, T, H)
        out, _ = _run(ctx, build_hash_route_prod,
                      {"ffn_normed": normed, "input_ids": ids, "tid2eid": table,
                       "router_weight": w})
        exp_idx = table[ids.reshape(-1)]
        logits = flat @ w.T
        scores = np.sqrt(np.log1p(np.exp(logits)))
        g = np.take_along_axis(scores, exp_idx, 1)
        exp_w = g / (g.sum(1, keepdims=True) + 1e-20) * 2.5
        for key in ("model", "sim"):
            o = out[key]
            np.testing.assert_array_equal(o["topk_indices"], exp_idx)
            np.testing.assert_allclose(o["topk_weights"], exp_w, rtol=1e-5, atol=1e-5)
            self.assertFalse(np.array_equal(np.sort(exp_idx, 1),
                                            np.sort(np.argsort(-o["router_scores"], 1)[:, :K], 1)))
            self.assertEqual(o["topk_indices"].dtype.kind, "i")
            np.testing.assert_allclose(o["ffn_flat"], normed.reshape(B * T, H), rtol=0, atol=0)
            onehot = np.eye(E + 1)[exp_idx]  # [N,K,E+1]
            self.assertEqual(o["expert_mask"].shape, (E + 1, K, B * T))
            np.testing.assert_allclose(o["expert_mask"], onehot.transpose(2, 1, 0), rtol=0, atol=0)
            hit = np.unique(exp_idx)
            self.assertEqual(o["hit_experts"].shape, (len(hit), 1))
            np.testing.assert_array_equal(o["hit_experts"][:, 0], hit)

    def test_all_four_groups_build_all_no_failed(self):
        import json
        from pathlib import Path
        from structure_reader.extract.deepseek_v4_hf import build_reading_document
        from structure_reader.onnx_module.build import build_all
        project = Path(__file__).resolve().parents[1]
        ws = project.parent
        cfg = json.loads((ws / "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json").read_text())
        for idx, count in ((0, 16), (2, 18), (3, 17), (4, 18)):
            doc = build_reading_document(ws, idx)
            res = build_all(ws, project, doc, cfg)
            for ph in ("prefill", "decode"):
                self.assertEqual(len(res["report"]["modules"][ph]), count)
                self.assertFalse(any(st.get("status") == "failed"
                                     for st in res["report"]["modules"][ph].values()))


if __name__ == "__main__":
    unittest.main()
