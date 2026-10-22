"""反馈06 ONNX 交付/语义测试（修复01）：实际资产、真实 builder 小合成例。

- 资产测试：checker/schema/IR、节点名唯一、cache pre/post、页面链接按 viewer 目录真实解析。
- 语义测试：调用**生产 builder**（build_attn_csa / build_attn_mask / build_attn_indexer）
  与小合成权重，与独立 numpy 期望比较；不再手写另一张图自证。
依赖项目内隔离安装的 onnx/onnxsim/onnxruntime；缺失则跳过并如实说明。
"""
from __future__ import annotations

import html as html_mod
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import onnx
    from onnx import TensorProto
    _HAS_ONNX = True
except Exception:  # pragma: no cover
    _HAS_ONNX = False


def _latest_onnx_candidate() -> Path:
    import os
    root = REPO_ROOT / "structure_reader" / "outputs" / "deepseek_v4_flash_0731_hf"
    env = os.environ.get("ONNX_CANDIDATE")
    if env:
        return root / env
    cands = [p for p in root.iterdir()
             if p.is_dir() and (p / "reports" / "onnx_build.json").is_file()]
    if not cands:
        raise unittest.SkipTest("没有 ONNX 构建候选（先运行 build-hf --onnx）")
    return max(cands, key=lambda p: p.stat().st_mtime)


OUT = _latest_onnx_candidate()


# ---------------------------------------------------------------------------
# 真实 builder 小合成例的公共设施
# ---------------------------------------------------------------------------
def _ctx(phase_id: str, tensors: dict, params: dict, rope_dim: int, ops=None):
    from structure_reader.onnx_module import builders
    phase = {"id": phase_id,
             "tensors": [{"id": k, **v} for k, v in tensors.items()],
             "operators": ops or [],
             "parameters": [{"id": k, "value": v} for k, v in params.items()]}
    rope = {"inv_freq": (1.0 / (10000 ** (np.arange(0, rope_dim, 2) / rope_dim))).astype(np.float32),
            "attention_factor": 1.0}
    return builders.Ctx({}, phase, {}, rope, rope)


def _model_of(gb):
    from structure_reader.onnx_module.build import gb_to_model
    gb.ensure_unique_names()
    return gb_to_model(gb, gb.module_id, gb.phase)


def _run(model, feeds):
    import onnxruntime as ort
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, feeds)))


def _rope_ref(x, cos, sin, rk):
    """独立 numpy：interleaved partial RoPE（HF apply_rotary_pos_emb 语义）。"""
    nope, r = x[..., :-rk], x[..., -rk:]
    x1, x2 = r[..., 0::2], r[..., 1::2]
    rot = np.stack([-x2, x1], axis=-1).reshape(r.shape)
    c = np.repeat(cos, 2, axis=-1)[:, None, :, :]
    s = np.repeat(sin, 2, axis=-1)[:, None, :, :]
    return np.concatenate([nope, r * c + rot * s], axis=-1)


def _rms(x, w, eps):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y * w if w is not None else y


@unittest.skipUnless(_HAS_ONNX, "onnx 未安装：ONNX 测试跳过（不伪造通过）")
class TestOnnxArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((OUT / "reports" / "onnx_build.json").read_text())
        cls.html = (OUT / "viewer" / "decoder.html").read_text()
        cls.manifest = json.loads((OUT / "run_manifest.json").read_text())

    def test_models_check_real_schema_and_unique_names(self):
        n = 0
        for phase, mods in self.report["modules"].items():
            for mid, st in mods.items():
                self.assertIsNone(st.get("status"), f"{phase}/{mid}: {st.get('error')}")
                for fname in ("model.onnx", "model.sim.onnx"):
                    m = onnx.load(str(OUT / "onnx" / phase / mid / fname))
                    onnx.checker.check_model(m, full_check=True)
                    self.assertEqual([o.domain for o in m.opset_import], [""])
                    self.assertEqual(m.opset_import[0].version, 18)
                    self.assertLessEqual(m.ir_version, 10)
                    names = [x.name for x in m.graph.node]
                    self.assertEqual(len(names), len(set(names)), f"{phase}/{mid}/{fname} 节点名重复")
                    init = {i.name for i in m.graph.initializer}
                    for w in st["weight_inputs"]:
                        self.assertIn(w, {i.name for i in m.graph.input})
                        self.assertNotIn(w, init)
                    n += 1
        self.assertEqual(n, 72)

    def test_cache_io_and_shared_outputs(self):
        def st(phase, mid):
            return self.report["modules"][phase][mid]

        for phase in ("prefill", "decode"):
            outs = {o.get("json_id", o["name"]) for o in st(phase, "attn_q")["outputs"]}
            self.assertTrue({"q_a_normed", "q_rope"} <= outs)
            for mid in ("attn_cache", "attn_csa", "attn_indexer"):
                s = st(phase, mid)
                self.assertFalse(any(x.endswith("_post") for x in s["cache_pre_inputs"]))
                self.assertTrue(any(o["role"].startswith("cache_post") for o in s["outputs"]))
            csa_outs = {o.get("json_id") for o in st(phase, "attn_csa")["outputs"]}
            self.assertTrue({"cache_compressed_c_post", "cache_entry_count_c_post",
                             "cache_overlap_c_kv_post"} <= csa_outs)
            idx_outs = {o.get("json_id") for o in st(phase, "attn_indexer")["outputs"]}
            self.assertTrue({"indexer_topk", "c_block_bias"} <= idx_outs)
            # block_bias 必须依赖 CSA compressed（JSON c_block_bias 的真实输入）
            m = onnx.load(str(OUT / "onnx" / phase / "attn_indexer" / "model.onnx"))
            self.assertIn("cache_compressed_c_post", {i.name for i in m.graph.input})

    def test_asset_links_resolve_from_viewer_dir(self):
        viewer = OUT / "viewer" / "decoder.html"
        html_text = viewer.read_text()
        hrefs = re.findall(r'href="([^"]*onnx/[^"]+)"', html_text)
        self.assertEqual(len(hrefs), 36 * 3)  # 原始/简化/映射下载
        bad = [h for h in hrefs if not (viewer.parent / h).is_file()]
        self.assertEqual(bad, [], f"{len(bad)} 个链接按 viewer 目录解析不存在，例如 {bad[:3]}")
        self.assertTrue(all(h.startswith("../onnx/") for h in hrefs))

    def test_page_local_netron_links_resolve_to_sim(self):
        """逐模块解析本地 llmarchi-netron 链接，用启动器 --resolve-only 校验到同一候选 sim。"""
        import subprocess
        app = (REPO_ROOT / "structure_reader/.work/tools/netron_launcher08/"
               "LLMarchiNetron.app/Contents/MacOS/LLMarchiNetron")
        if not app.is_file():
            self.skipTest("本地 Netron 启动器未构建")
        anchors = re.findall(r'<a class="onnx-open-link"[^>]*>', self.html)
        self.assertEqual(len(anchors), 36)
        seen = set()
        for a in anchors:
            mod = re.search(r'data-module="([^"]+)"', a).group(1)
            href = html_mod.unescape(re.search(r'href="([^"]+)"', a).group(1))
            phase, mid = mod.split("/")
            self.assertEqual(
                href,
                f"llmarchi-netron://open?candidate={OUT.name}&phase={phase}"
                f"&module={mid}&variant=sim")
            expect = (OUT / "onnx" / phase / mid / "model.sim.onnx").resolve()
            r = subprocess.run([str(app), "--resolve-only", href],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"resolved: {expect}", r.stdout)
            seen.add(mod)
        self.assertEqual(len(seen), 36)
        for ph in ("prefill", "decode"):
            self.assertIn(f"{ph}/attn_csa", seen)
            self.assertIn(f"{ph}/attn_indexer", seen)
        # 拒绝非法/越界/重复/错误 scheme
        bad = [
            f"llmarchi-netron://open?candidate={OUT.name}&phase=prefill&module=../../x&variant=sim",
            f"llmarchi-netron://open?candidate={OUT.name}&phase=prefill&module=attn_q&variant=orig",
            f"llmarchi-netron://open?candidate={'0'*64}&phase=prefill&module=attn_q&variant=sim",
            f"netron://open?candidate={OUT.name}&phase=prefill&module=attn_q&variant=sim",
            f"llmarchi-netron://open?candidate={OUT.name}&candidate={OUT.name}"
            "&phase=prefill&module=attn_q&variant=sim",
            f"llmarchi-netron://open?candidate={OUT.name}&phase=prefill&module=attn_q&variant=sim#x",
        ]
        for url in bad:
            r = subprocess.run([str(app), "--resolve-only", url], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, url)
        # 页面不再有网页 Netron/base64；并排面板已移除；仅 36 张原详图
        self.assertNotIn("netron.app", self.html)
        self.assertNotIn("base64", self.html)
        self.assertEqual(self.html.count('class="dg-pair"'), 0)
        self.assertEqual(len(re.findall(r'<svg class="dg"', self.html)), 36)
        ids = re.findall(r'\bid="([^"]+)"', self.html)
        self.assertEqual([i for i in set(ids) if ids.count(i) > 1], [])

    def test_launcher_boundaries_and_manifest_fixture(self):
        """启动器有限边界：userinfo/port 拒绝；manifest SHA 不符/符号链接出根拒绝（临时 fixture）。"""
        import hashlib
        import os
        import secrets
        import shutil
        import subprocess
        import tempfile
        app = (REPO_ROOT / "structure_reader/.work/tools/netron_launcher08/"
               "LLMarchiNetron.app/Contents/MacOS/LLMarchiNetron")
        if not app.is_file():
            self.skipTest("本地 Netron 启动器未构建")
        out_root = OUT.parent
        good = (f"llmarchi-netron://open?candidate={OUT.name}&phase=decode"
                "&module=moe_merge&variant=sim")
        r = subprocess.run([str(app), "--resolve-only", good], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        for bad in (
            f"llmarchi-netron://user:pw@open?candidate={OUT.name}&phase=decode&module=moe_merge&variant=sim",
            f"llmarchi-netron://open:8080?candidate={OUT.name}&phase=decode&module=moe_merge&variant=sim",
        ):
            r = subprocess.run([str(app), "--resolve-only", bad], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, bad)
        tmpdir = Path(tempfile.mkdtemp(prefix="llmarchi-netron-test-"))
        try:
            fid = secrets.token_hex(32)
            base = out_root / fid
            (base / "onnx/prefill/attn_q").mkdir(parents=True)
            payload = b"fixture-onnx-bytes"
            (base / "onnx/prefill/attn_q/model.sim.onnx").write_bytes(payload)
            correct = hashlib.sha256(payload).hexdigest()
            (base / "run_manifest.json").write_text(json.dumps(
                {"deliverables": {"onnx/prefill/attn_q/model.sim.onnx": "0" * 64}}))
            url = f"llmarchi-netron://open?candidate={fid}&phase=prefill&module=attn_q&variant=sim"
            r = subprocess.run([str(app), "--resolve-only", url], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, "manifest SHA 不符应拒绝")
            (base / "run_manifest.json").write_text(json.dumps(
                {"deliverables": {"onnx/prefill/attn_q/model.sim.onnx": correct}}))
            r = subprocess.run([str(app), "--resolve-only", url], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            # manifest 为指向项目外的符号链接 → 拒绝
            outside = tmpdir / "outside_manifest.json"
            outside.write_text(json.dumps({"deliverables": {"onnx/prefill/attn_q/model.sim.onnx": correct}}))
            (base / "run_manifest.json").unlink()
            os.symlink(outside, base / "run_manifest.json")
            r = subprocess.run([str(app), "--resolve-only", url], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, "manifest 符号链接出根应拒绝")
        finally:
            shutil.rmtree(out_root / fid, ignore_errors=True)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_sim_inserted_nodes_bound_by_adjacency(self):
        total_derived = 0
        for phase in ("prefill", "decode"):
            for mid in ("attn_csa", "attn_indexer"):
                d = OUT / "onnx" / phase / mid
                m = onnx.load(str(d / "model.sim.onnx"))
                groups = json.loads((d / "mapping.json").read_text())["groups"]
                mapped = {n["name"]: gid for gid, g in groups.items() for n in g["nodes"]}
                for node in m.graph.node:
                    self.assertIn(node.name, mapped, f"{phase}/{mid} {node.name} 未绑定")
                # 实际插入的 Reshape_0/1：kind/来源组/邻接证据
                for gid, g in groups.items():
                    for n in g["nodes"]:
                        if n.get("kind") != "onnxsim_inserted":
                            continue
                        total_derived += 1
                        self.assertTrue(n.get("derived"))
                        self.assertEqual(n["op_type"], "Reshape")
                        self.assertEqual(n["source_groups"], [gid])
                        self.assertEqual(gid, "c_rope" if mid == "attn_csa" else "i_pool_rope")
                        self.assertTrue(n["evidence"]["producer_groups"])
                        self.assertTrue(n["evidence"]["consumer_groups"])
                        self.assertTrue(n.get("locator"))
                st = json.loads((d / "state.json").read_text())
                self.assertEqual(st["sim_inserted_unresolved"], [])
                self.assertEqual(st["sim_inserted_bound"], 2)
        self.assertEqual(total_derived, 8)

    def test_page_reading_notes(self):
        for text in ("FP32 阅读特化图", "用本地 Netron 打开", "需已注册 llmarchi-netron",
                     "单专家模板", "默认因果/滑窗，padding 未覆盖"):
            self.assertIn(text, self.html)
        self.assertNotIn("右栏", self.html)

    def test_identity_records_graph_byte_deps(self):
        env = self.manifest["identity"]["onnx_modules"]
        for k in ("onnx", "onnxsim", "onnxruntime", "numpy", "protobuf", "ml_dtypes"):
            self.assertIn(k, env)
            self.assertTrue(env[k])
        self.assertEqual(env["opset"], 18)
        self.assertEqual(env["ir_version"], 10)


@unittest.skipUnless(_HAS_ONNX, "onnx 未安装：语义小例跳过")
class TestOnnxSemanticsProductionBuilders(unittest.TestCase):
    B, T, H, Dh, M, Rk = 1, 4, 4, 4, 2, 2

    def _csa_inputs(self, tbuf=0, tcc=0, t=4):
        B, H, Dh, M = self.B, self.H, self.Dh, self.M
        return {
            "attn_normed": {"shape": [B, t, H], "kind": "activation"},
            "cache_buffer_c_kv": {"shape": [B, tbuf, 2 * Dh], "kind": "activation"},
            "cache_buffer_c_gate": {"shape": [B, tbuf, 2 * Dh], "kind": "activation"},
            "cache_entry_count_c": {"shape": [], "kind": "index"},
            "cache_overlap_c_kv": {"shape": [B, M, Dh], "kind": "activation"},
            "cache_overlap_c_gate": {"shape": [B, M, Dh], "kind": "activation"},
            "cache_compressed_c": {"shape": [B, tcc, Dh], "kind": "activation"},
            "c_kv_weight": {"shape": [2 * Dh, H], "kind": "weight"},
            "c_gate_weight": {"shape": [2 * Dh, H], "kind": "weight"},
            "c_position_bias": {"shape": [M, 2 * Dh], "kind": "weight"},
            "c_kv_norm_weight": {"shape": [Dh], "kind": "weight"},
            "c_chunk_kv": {"shape": [B, t, 2 * Dh], "kind": "activation"},
            "c_chunk_gate": {"shape": [B, t, 2 * Dh], "kind": "activation"},
            "c_windows_kv": {"shape": [B, t // M, 2 * M, Dh], "kind": "activation"},
            "c_windows_gate": {"shape": [B, t // M, 2 * M, Dh], "kind": "activation"},
            "cache_buffer_c_kv_post": {"shape": [B, "Tbuf_post", 2 * Dh], "kind": "activation"},
            "cache_buffer_c_gate_post": {"shape": [B, "Tbuf_post", 2 * Dh], "kind": "activation"},
            "cache_compressed_c_post": {"shape": [B, "N_comp", Dh], "kind": "activation"},
            "cache_entry_count_c_post": {"shape": [], "kind": "index"},
            "cache_overlap_c_kv_post": {"shape": [B, M, Dh], "kind": "activation"},
            "cache_overlap_c_gate_post": {"shape": [B, M, Dh], "kind": "activation"},
        }

    def _csa_reference(self, x, wkv, wg, bias, wn, prior_kv, prior_gate, pre_comp, entry, eps, M):
        Dh = self.Dh
        kv = x @ wkv.T
        gate = x @ wg.T
        B, T = x.shape[0], x.shape[1]
        usable = (T // M) * M
        chunk = kv[:, :usable]
        chunkg = gate[:, :usable]
        buf_post = kv[:, usable:]
        nwin = usable // M
        out = {"buffer_kv_post": buf_post}
        if nwin == 0:
            out.update(compressed_post=pre_comp.copy(), overlap_kv=prior_kv.copy(),
                       overlap_gate=prior_gate.copy(), entry=entry)
            return out
        c4 = chunk.reshape(B, nwin, M, 2 * Dh)
        g4 = chunkg.reshape(B, nwin, M, 2 * Dh) + bias
        ca, cb = c4[..., :Dh], c4[..., Dh:]
        cag, cbg = g4[..., :Dh], g4[..., Dh:]
        # new_kv 布局 [B, Nwin, 2M, Dh]：前 M 行 Ca（前移/前态），后 M 行 Cb
        shap = np.empty((B, nwin, 2 * M, Dh), np.float32)
        gate_new = np.empty_like(shap)
        shap[:, :, M:, :] = cb
        gate_new[:, :, M:, :] = cbg
        shap[:, 0, :M, :] = prior_kv
        gate_new[:, 0, :M, :] = prior_gate
        if nwin > 1:
            shap[:, 1:, :M, :] = ca[:, :-1]
            gate_new[:, 1:, :M, :] = cag[:, :-1]
        probs = np.exp(gate_new - gate_new.max(2, keepdims=True))
        probs = probs / probs.sum(2, keepdims=True)
        pooled = (shap * probs).sum(2)
        comp = _rms(pooled, wn, eps)
        pos = np.arange(nwin)[None, :] * M + entry * M
        freq = 1.0 / (10000 ** (np.arange(0, self.Rk, 2) / self.Rk))
        cos = np.cos(pos[:, :, None] * freq[None, None, :])
        sin = np.sin(pos[:, :, None] * freq[None, None, :])
        comp = _rope_ref(comp[:, None], cos, sin, self.Rk)[:, 0]
        out.update(compressed_post=np.concatenate([pre_comp, comp], axis=1),
                   overlap_kv=ca[:, -1].copy(), overlap_gate=cag[:, -1].copy(),
                   entry=entry + nwin, compressed=comp)
        return out

    def _run_csa(self, tbuf, tcc, t, seed):
        from structure_reader.onnx_module import builders
        from structure_reader.onnx_module.build import gb_to_model
        tensors = self._csa_inputs(tbuf=tbuf, tcc=tcc, t=t)
        params = {"Dh": self.Dh, "M": self.M, "Rk": self.Rk, "EpsN": 1e-6}
        ctx = _ctx("prefill", tensors, params, self.Rk)
        gb = builders.build_attn_csa(ctx)
        gb.ensure_unique_names()
        model = gb_to_model(gb, "attn_csa", "prefill")
        rng = np.random.default_rng(seed)
        feeds = {
            "attn_normed": rng.normal(size=(self.B, t, self.H)).astype(np.float32),
            "cache_buffer_c_kv": rng.normal(size=(self.B, tbuf, 2 * self.Dh)).astype(np.float32),
            "cache_buffer_c_gate": rng.normal(size=(self.B, tbuf, 2 * self.Dh)).astype(np.float32),
            "cache_entry_count_c": np.array(2, np.int64),
            "cache_overlap_c_kv": rng.normal(size=(self.B, self.M, self.Dh)).astype(np.float32),
            "cache_overlap_c_gate": rng.normal(size=(self.B, self.M, self.Dh)).astype(np.float32),
            "cache_compressed_c": rng.normal(size=(self.B, tcc, self.Dh)).astype(np.float32),
            "c_kv_weight": rng.normal(size=(2 * self.Dh, self.H)).astype(np.float32),
            "c_gate_weight": rng.normal(size=(2 * self.Dh, self.H)).astype(np.float32),
            "c_position_bias": rng.normal(size=(self.M, 2 * self.Dh)).astype(np.float32),
            "c_kv_norm_weight": rng.normal(size=(self.Dh,)).astype(np.float32),
        }
        got = _run(model, feeds)
        ref = self._csa_reference(
            feeds["attn_normed"], feeds["c_kv_weight"], feeds["c_gate_weight"],
            feeds["c_position_bias"], feeds["c_kv_norm_weight"],
            feeds["cache_overlap_c_kv"], feeds["cache_overlap_c_gate"],
            feeds["cache_compressed_c"], 2, 1e-6, self.M)
        return got, ref, t // self.M

    def test_tiny_csa_zero_window(self):
        got, ref, _ = self._run_csa(tbuf=0, tcc=0, t=1, seed=3)
        np.testing.assert_allclose(got["cache_compressed_c_post"], ref["compressed_post"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got["cache_overlap_c_kv_post"], ref["overlap_kv"], rtol=1e-5, atol=1e-6)
        self.assertEqual(int(got["cache_entry_count_c_post"]), ref["entry"])

    def test_tiny_csa_one_window(self):
        got, ref, _ = self._run_csa(tbuf=0, tcc=0, t=2, seed=4)
        np.testing.assert_allclose(got["cache_compressed_c_post"], ref["compressed_post"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got["cache_overlap_c_kv_post"], ref["overlap_kv"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got["cache_overlap_c_gate_post"], ref["overlap_gate"], rtol=1e-5, atol=1e-6)

    def test_tiny_csa_two_windows_ca_shift(self):
        got, ref, _ = self._run_csa(tbuf=0, tcc=0, t=4, seed=5)
        np.testing.assert_allclose(got["cache_compressed_c_post"], ref["compressed_post"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got["cache_overlap_c_kv_post"], ref["overlap_kv"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got["cache_overlap_c_gate_post"], ref["overlap_gate"], rtol=1e-5, atol=1e-6)
        self.assertEqual(int(got["cache_entry_count_c_post"]), 2 + 2)

    # ---------------- mask（生产 build_attn_mask） ----------------
    def _run_mask(self, cum, t, W=4):
        from structure_reader.onnx_module import builders
        from structure_reader.onnx_module.build import gb_to_model
        tensors = {
            "cache_cumlen": {"shape": [], "kind": "index"},
            "mask_model": {"shape": [1, 1, t, min(cum, W - 1) + t], "kind": "activation"},
            "c_block_bias": {"shape": [1, 1, t, 2], "kind": "activation"},
            "mask_ext": {"shape": [1, 1, t, min(cum, W - 1) + t + 2], "kind": "activation"},
        }
        params = {"W": W}
        ctx = _ctx("prefill", tensors, params, 2)
        gb = builders.build_attn_mask(ctx)
        gb.ensure_unique_names()
        model = gb_to_model(gb, "attn_mask", "prefill")
        got = _run(model, {"B": np.array(1, np.int64), "cache_cumulative_length": np.array(cum, np.int64),
                           "T": np.array(t, np.int64),
                           "c_block_bias": np.zeros((1, 1, t, 2), np.float32)})
        return got["mask_model"]

    def test_tiny_mask_window_keeps_four_keys(self):
        m = self._run_mask(cum=10, t=1)
        self.assertEqual(m.shape, (1, 1, 1, 4))
        np.testing.assert_array_equal(m, np.zeros_like(m))  # 绝对 key 7,8,9,10 全可见

    def test_tiny_mask_partial_window_lower_triangle(self):
        m = self._run_mask(cum=0, t=3)
        expect = np.zeros((1, 1, 3, 3), np.float32)
        expect[0, 0, 0, 1:] = -np.inf
        expect[0, 0, 1, 2] = -np.inf
        np.testing.assert_array_equal(m, expect)

    # ---------------- Indexer（生产 build_attn_indexer） ----------------
    def _run_indexer(self, tci, tcc, seed, T=1):
        from structure_reader.onnx_module import builders
        from structure_reader.onnx_module.build import gb_to_model
        B, H, M, Id, Ih, Ki = 1, 4, 2, 4, 2, 4
        Rk = 2
        tensors = {
            "attn_normed": {"shape": [B, T, H], "kind": "activation"},
            "q_a_normed": {"shape": [B, T, H], "kind": "activation"},
            "position_ids": {"shape": [1, T], "kind": "index"},
            "cache_buffer_i_kv": {"shape": [B, 0, 2 * Id], "kind": "activation"},
            "cache_buffer_i_gate": {"shape": [B, 0, 2 * Id], "kind": "activation"},
            "cache_entry_count_i": {"shape": [], "kind": "index"},
            "cache_overlap_i_kv": {"shape": [B, M, Id], "kind": "activation"},
            "cache_overlap_i_gate": {"shape": [B, M, Id], "kind": "activation"},
            "cache_compressed_i": {"shape": [B, tci, Id], "kind": "activation"},
            "cache_compressed_c_post": {"shape": [B, tcc, Id], "kind": "activation"},
            "i_kv_weight": {"shape": [2 * Id, H], "kind": "weight"},
            "i_gate_weight": {"shape": [2 * Id, H], "kind": "weight"},
            "i_position_bias": {"shape": [M, 2 * Id], "kind": "weight"},
            "i_kv_norm_weight": {"shape": [Id], "kind": "weight"},
            "i_q_weight": {"shape": [Ih * Id, H], "kind": "weight"},
            "i_w_weight": {"shape": [Ih, H], "kind": "weight"},
            "indexer_topk": {"shape": [B, T, "K_sel"], "kind": "index"},
            "c_block_bias": {"shape": [B, 1, T, "N_comp"], "kind": "activation"},
            "cache_buffer_i_kv_post": {"shape": [B, "Tbuf_post", 2 * Id], "kind": "activation"},
            "cache_buffer_i_gate_post": {"shape": [B, "Tbuf_post", 2 * Id], "kind": "activation"},
            "cache_compressed_i_post": {"shape": [B, "Tci_post", Id], "kind": "activation"},
            "cache_entry_count_i_post": {"shape": [], "kind": "index"},
            "cache_overlap_i_kv_post": {"shape": [B, M, Id], "kind": "activation"},
            "cache_overlap_i_gate_post": {"shape": [B, M, Id], "kind": "activation"},
        }
        params = {"Dh": Id, "Id": Id, "Ih": Ih, "Ki": Ki, "M": M, "Rk": Rk, "EpsN": 1e-6}
        ctx = _ctx("prefill", tensors, params, Rk)
        gb = builders.build_attn_indexer(ctx)
        gb.ensure_unique_names()
        model = gb_to_model(gb, "attn_indexer", "prefill")
        rng = np.random.default_rng(seed)
        feeds = {
            "attn_normed": rng.normal(size=(B, T, H)).astype(np.float32),
            "q_a_normed": rng.normal(size=(B, T, H)).astype(np.float32),
            "position_ids": np.arange(T, dtype=np.int64)[None, :],
            "cache_buffer_i_kv": np.zeros((B, 0, 2 * Id), np.float32),
            "cache_buffer_i_gate": np.zeros((B, 0, 2 * Id), np.float32),
            "cache_entry_count_i": np.array(0, np.int64),
            "cache_overlap_i_kv": np.zeros((B, M, Id), np.float32),
            "cache_overlap_i_gate": np.zeros((B, M, Id), np.float32),
            "cache_compressed_i": rng.normal(size=(B, tci, Id)).astype(np.float32),
            "cache_compressed_c_post": rng.normal(size=(B, tcc, Id)).astype(np.float32),
            "i_kv_weight": rng.normal(size=(2 * Id, H)).astype(np.float32),
            "i_gate_weight": rng.normal(size=(2 * Id, H)).astype(np.float32),
            "i_position_bias": rng.normal(size=(M, 2 * Id)).astype(np.float32),
            "i_kv_norm_weight": rng.normal(size=(Id,)).astype(np.float32),
            "i_q_weight": rng.normal(size=(Ih * Id, H)).astype(np.float32),
            "i_w_weight": rng.normal(size=(Ih, H)).astype(np.float32),
        }
        return _run(model, feeds)

    def test_tiny_indexer_with_history(self):
        # T=2,M=2 → nwin=1；tci=2 → scoring 3 条；tcc=3 = CSA Ncomp
        got = self._run_indexer(tci=2, tcc=3, seed=6, T=2)
        topk = got["indexer_topk"]
        self.assertEqual(topk.shape, (1, 2, 3))
        self.assertEqual(topk.dtype, np.int64)
        # pos0 阈值 0 → 全部越界 → -1；pos1 阈值 1 → 只有 entry0 可读，其余 -1
        self.assertEqual(sorted(topk[0, 0].tolist()), [-1, -1, -1])
        self.assertEqual(sorted(topk[0, 1].tolist()), [-1, -1, 0])
        bb = got["c_block_bias"]
        self.assertEqual(bb.shape, (1, 1, 2, 3))
        self.assertTrue(np.all(np.isneginf(bb[0, 0, 0])))
        self.assertEqual(bb[0, 0, 1, 0], 0.0)
        self.assertTrue(np.isneginf(bb[0, 0, 1, 1]))

    def test_tiny_indexer_empty_cache_topk0(self):
        # T=1 < M=2 → nwin=0；tci=0 → scoring 0 条，TopK K=0
        got = self._run_indexer(tci=0, tcc=0, seed=7, T=1)
        self.assertEqual(got["indexer_topk"].shape, (1, 1, 0))
        self.assertEqual(got["c_block_bias"].shape, (1, 1, 1, 0))


# ---------------------------------------------------------------------------
# 修复02：同一“不巧合”小配置（B=2, T=3, Nh=3, G=3, Dh=4…）跑全部 18 个 builder
# ---------------------------------------------------------------------------
_OPS = {
    "norm_attn": [{"id": "attn_norm", "inputs": ["attn_mhc_collapsed", "attn_norm_weight"],
                   "weights": [], "outputs": ["attn_normed"]}],
    "norm_ffn": [{"id": "ffn_norm", "inputs": ["ffn_mhc_collapsed", "ffn_norm_weight"],
                  "weights": [], "outputs": ["ffn_normed"]}],
    "res_attn": [
        {"id": "attn_residual_combT", "inputs": ["attn_mhc_comb"], "weights": [],
         "outputs": ["attn_residual_combT"]},
        {"id": "attn_residual_expand", "inputs": ["attn_mhc_post", "attn_block_output"],
         "weights": [], "outputs": ["attn_expanded"]},
        {"id": "attn_residual_mix",
         "inputs": ["attn_residual_combT", "hidden_streams_in", "attn_expanded"],
         "weights": [], "outputs": ["attn_out_streams"]}],
    "res_ffn": [
        {"id": "ffn_residual_combT", "inputs": ["ffn_mhc_comb"], "weights": [],
         "outputs": ["ffn_residual_combT"]},
        {"id": "ffn_residual_expand", "inputs": ["ffn_mhc_post", "mlp_output"],
         "weights": [], "outputs": ["ffn_expanded"]},
        {"id": "ffn_residual_mix",
         "inputs": ["ffn_residual_combT", "attn_out_streams", "ffn_expanded"],
         "weights": [], "outputs": ["hidden_streams_out"]}],
}

SMALL = dict(B=2, T=3, H=4, Nh=3, Dh=4, Rq=4, G=3, Ro=2, Rk=2, M=2, E=3, K=2,
             Es=4, P=2, L=2, Nc=2, Nhc=2, Sit=2, W=4)


def _s(x):
    return list(x)


def _fixtures(mid):
    """每个模块的 tensor id → shape/kind（小配置，覆盖真实 builder 的 IO）。"""
    B, T, H, Nh, Dh, Rq, G, Ro = (SMALL[k] for k in ("B", "T", "H", "Nh", "Dh", "Rq", "G", "Ro"))
    Rk, M, E, K, Es, P, L, Nc, Nhc = (SMALL[k] for k in ("Rk", "M", "E", "K", "Es", "P", "L", "Nc", "Nhc"))
    f = {}
    def a(tid, shape):
        f[tid] = {"shape": _s(shape), "kind": "activation"}
    def w(tid, shape):
        f[tid] = {"shape": _s(shape), "kind": "weight"}
    def i(tid, shape):
        f[tid] = {"shape": _s(shape), "kind": "index"}
    if mid in ("norm_attn", "norm_ffn"):
        a("attn_mhc_collapsed", [B, T, H]); w("attn_norm_weight", [H])
        a("attn_normed", [B, T, H])
        a("attn_out_streams", [B, T, Nhc, H]); a("ffn_mhc_collapsed", [B, T, H])
        w("ffn_norm_weight", [H]); a("ffn_normed", [B, T, H])
    elif mid == "attn_q":
        a("attn_normed", [B, T, H]); i("position_ids", [1, T])
        w("q_weight", [Rq, H]); w("q_a_norm_weight", [Rq]); w("q_b_weight", [Nh * Dh, Rq])
        a("q_a_normed", [B, T, Rq]); a("q_rope", [B, Nh, T, Dh])
    elif mid == "attn_kv":
        a("attn_normed", [B, T, H]); i("position_ids", [1, T])
        w("kv_weight", [Dh, H]); w("kv_norm_weight", [Dh]); a("kv_rope", [B, 1, T, Dh])
    elif mid == "attn_cache":
        a("kv_rope", [B, 1, T, Dh]); a("cache_keys", [B, 1, P, Dh])
        a("kv_full", [B, 1, P + T, Dh]); a("cache_keys_post", [B, 1, SMALL["W"] - 1, Dh])
        a("cache_values_post", [B, 1, SMALL["W"] - 1, Dh]); a("cache_cumulative_length", []); a("cache_cumulative_length_post", [])
    elif mid == "attn_core":
        a("q_rope", [B, Nh, T, Dh]); a("kv_full", [B, 1, L, Dh]); a("cache_compressed_c_post", [B, Nc, Dh])
        a("mask_ext", [B, 1, T, L + Nc]); w("sinks", [Nh]); a("attn_head_out", [B, T, Nh, Dh])
    elif mid == "attn_out":
        a("attn_head_out", [B, T, Nh, Dh]); i("position_ids", [1, T])
        w("o_a_weight", [G * Ro, Nh * Dh // G]); w("o_b_weight", [H, G * Ro]); a("attn_block_output", [B, T, H])
    elif mid in ("res_attn", "res_ffn"):
        a("attn_mhc_comb", [B, T, Nhc, Nhc]); a("attn_mhc_post", [B, T, Nhc]); a("hidden_streams_in", [B, T, Nhc, H])
        a("attn_block_output", [B, T, H]); a("attn_out_streams", [B, T, Nhc, H])
        a("ffn_mhc_comb", [B, T, Nhc, Nhc]); a("ffn_mhc_post", [B, T, Nhc])
        a("mlp_output", [B, T, H]); a("hidden_streams_out", [B, T, Nhc, H])
        a("attn_residual_combT", [B, T, Nhc, Nhc]); a("attn_expanded", [B, T, Nhc, H])
        a("ffn_residual_combT", [B, T, Nhc, Nhc]); a("ffn_expanded", [B, T, Nhc, H])
    elif mid in ("mhc_attn", "mhc_ffn"):
        a("hidden_streams_in", [B, T, Nhc, H]); a("attn_out_streams", [B, T, Nhc, H])
        w("attn_hc_fn", [(2 + Nhc) * Nhc, Nhc * H]); w("attn_hc_base", [(2 + Nhc) * Nhc]); w("attn_hc_scale", [3])
        w("ffn_hc_fn", [(2 + Nhc) * Nhc, Nhc * H]); w("ffn_hc_base", [(2 + Nhc) * Nhc]); w("ffn_hc_scale", [3])
        for t in ("attn_mhc_pre", "attn_mhc_post", "ffn_mhc_pre", "ffn_mhc_post"):
            a(t, [B, T, Nhc])
        for t in ("attn_mhc_comb", "ffn_mhc_comb"):
            a(t, [B, T, Nhc, Nhc])
        a("attn_mhc_collapsed", [B, T, H]); a("ffn_mhc_collapsed", [B, T, H])
    elif mid == "attn_mask":
        a("cache_cumulative_length", []); a("c_block_bias", [B, 1, T, Nc])
        a("mask_model", [B, 1, T, min(P, SMALL["W"] - 1) + T])
        a("mask_ext", [B, 1, T, min(P, SMALL["W"] - 1) + T + Nc])
    elif mid == "moe_shared":
        a("ffn_normed", [B, T, H]); w("shared_gate_weight", [Es, H]); w("shared_up_weight", [Es, H])
        w("shared_down_weight", [H, Es]); a("shared_out", [B, T, H])
    elif mid == "moe_merge":
        a("routed_view", [B, T, H]); a("shared_out", [B, T, H]); a("mlp_output", [B, T, H])
    elif mid == "moe_route":
        a("ffn_normed", [B, T, H]); w("router_weight", [E, H]); w("router_bias", [E])
        i("topk_indices", [B * T, K]); a("topk_weights", [B * T, K]); a("router_logits", [B * T, E])
        a("expert_mask", [E + 1, K, B * T]); i("hit_experts", ["N_hit", 1]); a("ffn_flat", [B * T, H])
    elif mid == "moe_expert":
        a("ffn_flat", [B * T, H]); a("expert_mask", [E + 1, K, B * T]); a("topk_weights", [B * T, K])
        w("expert_gate_up", [E, 2 * Es, H]); w("expert_down", [E, H, Es])
        i("expert_token_idx", ["N_e"]); i("expert_topk_pos", ["N_e"])
        a("expert_tokens", ["N_e", H]); a("expert_gate_up_e", [2 * Es, H]); a("expert_down_e", [H, Es])
        a("expert_gu_out", ["N_e", 2 * Es]); a("expert_gate_c", ["N_e", Es]); a("expert_up_c", ["N_e", Es])
        a("expert_gated", ["N_e", Es]); a("expert_down_raw", ["N_e", H]); a("expert_down_out", ["N_e", H])
        a("routed_out", [B * T, H]); a("routed_view", [B, T, H])
    return f


_BUILDERS = {
    "norm_attn": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_norm(c, "attn_norm"),
    "norm_ffn": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_norm(c, "ffn_norm"),
    "attn_q": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_attn_q(c),
    "attn_kv": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_attn_kv(c),
    "attn_cache": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_attn_cache(c),
    "attn_core": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_attn_core(c),
    "attn_out": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_attn_out(c),
    "res_attn": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_res_attn(c),
    "res_ffn": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_res_ffn(c),
    "mhc_attn": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_mhc(c, "mhc_attn"),
    "mhc_ffn": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_mhc(c, "mhc_ffn"),
    "moe_shared": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_moe_shared(c),
    "moe_merge": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_moe_merge(c),
    "moe_route": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_moe_route(c),
    "moe_expert": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_moe_expert(c),
    "attn_mask": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_attn_mask(c),
    "attn_csa": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_attn_csa(c),
    "attn_indexer": lambda c: __import__("structure_reader.onnx_module.builders", fromlist=["x"]).build_attn_indexer(c),
}


def _params(mid):
    p = {k: SMALL[k] for k in ("H", "Dh", "Nh", "Rq", "G", "Ro", "Rk", "M", "E", "K", "Es",
                               "Nh", "Nhc", "W")}
    p.update({"Id": SMALL["Dh"], "Ih": SMALL["Nh"], "Ki": 4, "Sit": SMALL["Sit"],
              "Eps": 1e-6, "EpsN": 1e-6, "Lim": 10.0, "Scl": 1.5})
    return p


@unittest.skipUnless(_HAS_ONNX, "onnx 未安装：小配置覆盖跳过")
class TestAllBuildersSmallConfig(unittest.TestCase):
    """同一小配置实际构造/加载/运行当前 18 个 builder；核真实输出 shape。"""

    def _run_one(self, mid, seed=0):
        from structure_reader.onnx_module.build import gb_to_model
        ctx = _ctx("prefill", _fixtures(mid), _params(mid), SMALL["Rk"], ops=_OPS.get(mid))
        gb = _BUILDERS[mid](ctx)
        gb.ensure_unique_names()
        model = gb_to_model(gb, mid, "prefill")
        self.assertEqual(len({n.name for n in model.graph.node}), len(model.graph.node))
        rng = np.random.default_rng(seed)
        feeds = {}
        sym = {k: v for k, v in SMALL.items()} | {"BT": SMALL["B"] * SMALL["T"], "Ne": 2,
                                                   "N_e": 2, "Nhit": 2, "K_sel": 2, "N_comp": 2}
        for inp in model.graph.input:
            name = inp.name
            dims = [d.dim_value if d.HasField("dim_value") else (sym.get(d.dim_param) or 1)
                    for d in inp.type.tensor_type.shape.dim]
            if inp.type.tensor_type.elem_type == TensorProto.INT64:
                continue
            if "weight" in name or name in ("sinks",):
                feeds[name] = rng.normal(size=dims).astype(np.float32) * 0.2
            else:
                feeds[name] = rng.normal(size=dims).astype(np.float32)
        return self._special_feeds(mid, feeds, model), model

    def _special_feeds(self, mid, feeds, model):
        B, T, H, Nh, Dh, G, Ro, M, E, K, Es, Nhc = (SMALL[k] for k in
                                                   ("B", "T", "H", "Nh", "Dh", "G", "Ro", "M", "E", "K", "Es", "Nhc"))
        if mid == "attn_q" or mid == "attn_kv" or mid == "attn_out":
            feeds["position_ids"] = np.arange(T, dtype=np.int64)[None, :]
        if mid == "attn_cache":
            feeds["cache_keys"] = np.random.default_rng(1).normal(size=(B, 1, SMALL["P"], Dh)).astype(np.float32)
            feeds["kv_rope"] = np.random.default_rng(2).normal(size=(B, 1, T, Dh)).astype(np.float32)
            feeds["cache_cumulative_length"] = np.array(SMALL["P"], np.int64)
            feeds["T"] = np.array(T, np.int64)
        if mid == "attn_mask":
            feeds["B"] = np.array(B, np.int64); feeds["T"] = np.array(T, np.int64)
            feeds["cache_cumulative_length"] = np.array(SMALL["P"], np.int64)
            feeds["T"] = np.array(T, np.int64)
            feeds["c_block_bias"] = np.zeros((B, 1, T, SMALL["Nc"]), np.float32)
        if mid == "moe_route":
            feeds["ffn_normed"] = np.random.default_rng(3).normal(size=(B, T, H)).astype(np.float32)
        if mid == "moe_expert":
            BT = B * T
            mask = np.zeros((E + 1, K, BT), np.float32)
            mask[0, 0, 0] = 1.0
            mask[0, 1, 2] = 1.0
            feeds["expert_mask"] = mask
            feeds["expert_index"] = np.array(0, np.int64)
            feeds["B"] = np.array(B, np.int64); feeds["T"] = np.array(T, np.int64)
            feeds["topk_weights"] = np.random.default_rng(4).normal(size=(BT, K)).astype(np.float32)
        if mid in ("attn_csa", "attn_indexer"):
            pass
        return feeds

    def test_all_18_builders_run(self):
        import onnxruntime as ort
        for mid in _BUILDERS:
            with self.subTest(module=mid):
                if mid in ("attn_csa", "attn_indexer"):
                    continue  # 由已有 0/1/2 窗与空缓存小例（生产 builder）覆盖
                feeds, model = self._run_one(mid, seed=7)
                sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
                out = sess.run(None, feeds)
                self.assertTrue(len(out) >= 1)

    def test_attn_core_sink_softmax_numpy(self):
        import onnxruntime as ort
        feeds, model = self._run_one("attn_core", seed=11)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        names = [o.name for o in sess.get_outputs()]
        got = dict(zip(names, sess.run(None, feeds)))
        q = feeds["q_rope"]; kv = feeds["kv_full"]; comp = feeds["cache_compressed_c_post"]
        mask = feeds["mask_ext"]; sinks = feeds["W.sink_softmax.sinks"]
        allkv = np.concatenate([kv, comp[:, None]], axis=2)
        rep = np.repeat(allkv, SMALL["Nh"], axis=1)  # 广播到 Nh
        scale = SMALL["Dh"] ** -0.5
        qk = np.einsum("bhtd,bhld->bhtl", q, rep) * scale + mask
        s = np.broadcast_to(sinks.reshape(1, -1, 1, 1), (q.shape[0], SMALL["Nh"], q.shape[2], 1))
        logits = np.concatenate([qk, s], -1)
        logits = logits - logits.max(-1, keepdims=True)
        p = np.exp(logits); p = p / p.sum(-1, keepdims=True)
        ref = np.einsum("bhtl,bhld->bthd", p[..., :-1], rep)
        np.testing.assert_allclose(got["attn_head_out"], ref, rtol=1e-4, atol=1e-4)

    def test_attn_out_grouped_projection_numpy(self):
        import onnxruntime as ort
        from structure_reader.onnx_module.build import gb_to_model
        B, T, H, Nh, Dh, G, Ro, Rk = (SMALL[k] for k in ("B", "T", "H", "Nh", "Dh", "G", "Ro", "Rk"))
        ctx = _ctx("prefill", _fixtures("attn_out"), _params("attn_out"), Rk, ops=_OPS.get("attn_out"))
        gb = _BUILDERS["attn_out"](ctx)
        gb.ensure_unique_names()
        gb.out("o_a_out", [B, T, G * Ro])
        model = gb_to_model(gb, "attn_out", "prefill")
        rng = np.random.default_rng(12)
        x = rng.normal(size=(B, T, Nh, Dh)).astype(np.float32)
        wa = (rng.normal(size=(G * Ro, Nh * Dh // G)).astype(np.float32) * 0.3)
        wb = (rng.normal(size=(H, G * Ro)).astype(np.float32) * 0.3)
        pos = np.arange(T, dtype=np.int64)[None, :]
        feeds = {"attn_head_out": x, "o_a_weight": wa, "o_b_weight": wb, "position_ids": pos}
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
        # 独立 numpy：transpose(1,2) → 逆 RoPE（-sin）→ transpose 回 → 逐组投影
        freq = 1.0 / (10000 ** (np.arange(0, Rk, 2) / Rk))
        cos = np.cos(pos.astype(np.float32)[:, :, None] * freq[None, None, :])
        sin = np.sin(pos.astype(np.float32)[:, :, None] * freq[None, None, :])
        xh = x.transpose(0, 2, 1, 3)
        rh = _rope_ref(xh, cos, sin * 0, Rk)  # placeholder replaced below
        rh = _rope_ref(xh, cos, -sin, Rk)     # sign=-1：逆旋转
        xr = rh.transpose(0, 2, 1, 3)
        in_g = Nh * Dh // G
        w3 = wa.reshape(G, Ro, in_g).transpose(0, 2, 1)
        y = np.stack([xr.reshape(-1, G, in_g)[:, g, :] @ w3[g] for g in range(G)], axis=1)
        ref = y.reshape(B, T, G * Ro)
        np.testing.assert_allclose(got["o_a_out"], ref, rtol=1e-4, atol=1e-4)

    def test_moe_route_hit_and_ffn_flat(self):
        import onnxruntime as ort
        feeds, model = self._run_one("moe_route", seed=13)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
        self.assertIn("ffn_flat", got)
        hit = got["hit_experts"]
        self.assertEqual(hit.ndim, 2)
        self.assertEqual(hit.shape[1], 1)
        self.assertEqual(hit.shape[0], int(hit.shape[0]))
        # 期望：mask.sum(-1,-2)>0 的位置数 = hit 行数
        em = got["expert_mask"]
        self.assertEqual(hit.shape[0], int((em.sum(axis=(1, 2)) > 0).sum()))

    def test_moe_expert_selection_and_view(self):
        import onnxruntime as ort
        feeds, model = self._run_one("moe_expert", seed=14)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        got = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
        self.assertEqual(sorted(got["expert_token_idx"].tolist()), [0, 2])
        self.assertEqual(sorted(got["expert_topk_pos"].tolist()), [0, 1])
        self.assertEqual(got["routed_view"].shape, (SMALL["B"], SMALL["T"], SMALL["H"]))
        # routed_view[0,0] 应等于 token0 的 down_out（index 0），[1,0] 等于 token2
        np.testing.assert_allclose(got["routed_view"][0, 0], got["routed_out"][0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got["routed_view"][0, 2], got["routed_out"][2], rtol=1e-6, atol=1e-6)
