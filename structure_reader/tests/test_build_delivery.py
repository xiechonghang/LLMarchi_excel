"""工作包 C 测试：真实 build、失败隔离、幂等复用、图表同源。"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _support  # noqa: E402
from _support import REPO_ROOT  # noqa: E402

PROJECT = REPO_ROOT / "structure_reader"
ENTRY = PROJECT / "scripts" / "read_model_structure.py"
PYTHON = sys.executable
MODEL_KEY = "deepseek_v4_flash_0731"


def _run_build(*extra: str, workspace_root: str | None = None) -> subprocess.CompletedProcess:
    argv = [PYTHON, str(ENTRY), "build", "--model", MODEL_KEY, *extra]
    if workspace_root:
        argv += ["--workspace-root", workspace_root]
    return subprocess.run(argv, capture_output=True, text=True, timeout=600,
                          cwd=str(REPO_ROOT))


class TestBuildReal(unittest.TestCase):
    """真实 build：成功、幂等复用、产物齐套、图表同源。"""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_build("--json")
        cls.payload = json.loads(cls.result.stdout)
        cls.out = Path(cls.payload["output_dir"])

    def test_exit_zero_and_success(self):
        self.assertEqual(self.result.returncode, 0, self.result.stdout)
        self.assertEqual(self.payload["status"], "success")

    def test_full_v1_tree(self):
        expected = (
            "README.md", "run_manifest.json",
            "data/input_manifest.json", "data/layer_plan.json",
            "data/semantic_graph.json", "data/instance_bindings.json",
            "data/scenarios.json",
            "data/dspark/semantic_graph.json",
            "data/dspark/integration.json", "data/dspark/scenarios.json",
            "data/dspark/connections.csv", "data/dspark/tensor_axes.csv",
            "reports/reading_guide.md", "reports/validation_report.json",
            "reports/dspark_validation.json",
            "reports/source_evidence.json", "reports/coverage_report.json",
            "viewer/structure.html",
            "tables/connections.csv", "tables/tensor_axes.csv",
            "logs/build.log",
        )
        for rel in expected:
            path = self.out / rel
            self.assertTrue(path.is_file(), rel)
            self.assertGreater(path.stat().st_size, 0, rel)

    def test_idempotent_reuse(self):
        again = _run_build("--json")
        self.assertEqual(again.returncode, 0)
        payload2 = json.loads(again.stdout)
        self.assertEqual(payload2["analysis_id"], self.payload["analysis_id"])
        self.assertTrue(payload2["reused"])

    def test_semantic_graph_canonical(self):
        from structure_reader.graph_document import graph_from_json, graph_to_json
        text = (self.out / "data" / "semantic_graph.json").read_text(encoding="utf-8")
        graph = graph_from_json(text)
        self.assertEqual(graph_to_json(graph), text.rstrip("\n"))
        self.assertEqual(graph.template_id, "deepseek_v4_flash_0731:csa_topk")

    def test_csv_graph_same_source(self):
        """CSV 与图 JSON 同源：connections 全部边 ID 一致、tensor_axes 联查一致。"""
        graph = json.loads(
            (self.out / "data" / "semantic_graph.json").read_text(encoding="utf-8"))
        conn_rows = list(csv.DictReader(io.StringIO(
            (self.out / "tables" / "connections.csv").read_text(encoding="utf-8"))))
        edge_ids = {e["stable_id"] for e in graph["edges"]}
        self.assertEqual({r["edge_id"] for r in conn_rows}, edge_ids)
        # 三方一致抽查：边张量 = CSV 张量
        edge_tensor = {e["stable_id"]: e["tensor_id"] for e in graph["edges"]}
        for r in conn_rows[:50]:
            self.assertEqual(edge_tensor[r["edge_id"]], r["tensor_id"])
        axes_rows = list(csv.DictReader(io.StringIO(
            (self.out / "tables" / "tensor_axes.csv").read_text(encoding="utf-8"))))
        tensor_ids = {t["stable_id"] for t in graph["tensors"]}
        self.assertLessEqual({r["tensor_id"] for r in axes_rows}, tensor_ids)
        # 轴定义与 named_axes 一致（联查）
        axis_by_symbol = {a["symbol"]: a for a in graph["named_axes"]}
        for r in axes_rows[:60]:
            axis = axis_by_symbol[r["axis_symbol"]]
            self.assertEqual(r["axis_name"], axis["name"])

    def test_scenarios_complete(self):
        doc = json.loads(
            (self.out / "data" / "scenarios.json").read_text(encoding="utf-8"))
        ids = {s["scenario_id"] for s in doc["instances"]}
        self.assertEqual(ids, {"prefill_t3", "prefill_t4", "prefill_t5",
                               "prefill_t127", "prefill_t128", "prefill_t129",
                               "decode_p2", "decode_p3", "decode_p4",
                               "decode_p126", "decode_p127", "decode_p128"})
        self.assertEqual(doc["default_scenario"], "decode_p128")
        d128 = next(s for s in doc["instances"]
                    if s["scenario_id"] == "decode_p128")
        # 门禁前必补：decode 具体状态槽位（model.py:353）
        self.assertEqual(
            d128["compression"]["state_slots"]["current_token_write"]["slots"],
            [4])
        t5 = next(s for s in doc["instances"]
                  if s["scenario_id"] == "prefill_t5")
        # remainder 尾块槽位（model.py:340-341）
        self.assertEqual(
            t5["compression"]["state_slots"]["remainder_write"]["slots"], [4])
        # 版本链在 instance_bindings
        ib = json.loads((self.out / "data" / "instance_bindings.json")
                        .read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(ib["decode_version_chain"]), 4)
        self.assertTrue(all(c["state_id"] == "csa_topk:compress_kv_state"
                            for c in ib["decode_version_chain"]))

    def test_page_self_contained_and_data_embedded(self):
        html = (self.out / "viewer" / "structure.html").read_text(encoding="utf-8")
        for bad in ("http://", "https://", "cdn.", "integrity="):
            self.assertNotIn(bad, html, bad)
        # 数据内嵌（同源：图 JSON 在页面 payload 里）
        graph_text = (self.out / "data" / "semantic_graph.json").read_text(encoding="utf-8")
        template_id = json.loads(graph_text)["template_id"]
        self.assertIn(template_id, html)
        # 未覆盖模板明确标注
        self.assertIn("pending_extraction", html)
        self.assertIn("未覆盖", html)
        # 量化范围与 offset/-1 规则出现在页面数据（不只藏源码）
        self.assertIn("act_quant", html)
        self.assertIn("-1", html)

    def test_honest_coverage(self):
        cov = json.loads((self.out / "reports" / "coverage_report.json")
                         .read_text(encoding="utf-8"))
        # V2+B4：四类主干 extracted；DSpark 整体集成（pending 清空、
        # dspark 块记录集成事实）
        self.assertEqual(
            sorted(cov["extracted"]),
            sorted(["deepseek_v4_flash_0731:swa_hash",
                    "deepseek_v4_flash_0731:csa_hash",
                    "deepseek_v4_flash_0731:csa_topk",
                    "deepseek_v4_flash_0731:hca_topk"]))
        self.assertEqual(cov["pending_extraction"], [])
        self.assertEqual(cov["dspark"]["status"], "extracted_integrated")
        self.assertTrue(any("kernel" in u for u in cov["uncovered"]))
        val = json.loads((self.out / "reports" / "validation_report.json")
                         .read_text(encoding="utf-8"))
        self.assertGreaterEqual(val["controlled_errors"]["spec_count"], 26)
        self.assertTrue(val["controlled_errors"]["all_detected"])
        # B4：DSpark 集成契约与受控错误（同 build 失败策略，逐项 detected）
        dval = json.loads((self.out / "reports" / "dspark_validation.json")
                          .read_text(encoding="utf-8"))
        self.assertTrue(dval["original_contracts"]["status"] == "pass")
        self.assertTrue(dval["controlled_errors"]["all_detected"])
        self.assertGreaterEqual(dval["controlled_errors"]["catalog_count"], 12)

    def test_only_implemented_commands_in_help(self):
        result = subprocess.run([PYTHON, str(ENTRY), "--help"],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0)
        self.assertIn("build", result.stdout)
        self.assertIn("inspect", result.stdout)
        for not_impl in ("validate", "render", "export-tables"):
            self.assertNotIn(not_impl, result.stdout)


class TestBuildFailures(unittest.TestCase):
    """失败路径：未知模型/基线失配隔离/产物篡改拒绝。"""

    def test_unknown_model(self):
        result = _run_build("--model", "no_such_model")
        self.assertEqual(result.returncode, 1)
        self.assertIn("failed", result.stdout)

    def test_tampered_output_rejected(self):
        first = _run_build("--json")
        self.assertEqual(first.returncode, 0)
        out = Path(json.loads(first.stdout)["output_dir"])
        readme = out / "README.md"
        original = readme.read_bytes()
        try:
            readme.write_bytes(original + b"tampered\n")
            second = _run_build("--json")
            self.assertEqual(second.returncode, 1)
            self.assertIn("output_consistency_conflict", second.stdout)
        finally:
            readme.write_bytes(original)
        third = _run_build("--json")
        self.assertEqual(third.returncode, 0)

    def test_failed_run_leaves_record_no_output(self):
        result = _run_build("--model", "no_such_model")
        self.assertEqual(result.returncode, 1)
        self.assertIn("failed_record", result.stdout)


class TestPagePayloadAndSchema(unittest.TestCase):
    """codex_review_01：payload 浏览器行为、JS schema、槽位语义、面板数据。"""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_build("--json")
        assert cls.result.returncode == 0, cls.result.stdout
        cls.out = Path(json.loads(cls.result.stdout)["output_dir"])
        cls.html = (cls.out / "viewer" / "structure.html").read_text(encoding="utf-8")

    def _payload_raw(self) -> str:
        import re
        m = re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            self.html, re.S)
        return m.group(1)

    def test_payload_is_valid_json_as_browser_parses(self):
        # 浏览器 JSON.parse(textContent) 不做 HTML 实体解码；
        # 原始 script 内容必须直接是合法 JSON（不得先 html.unescape）
        import re
        payload = json.loads(self._payload_raw())
        self.assertEqual(payload["modelKey"], MODEL_KEY)

    def test_payload_has_no_html_escaping(self):
        raw = self._payload_raw()
        self.assertNotIn("&quot;", raw)
        self.assertNotIn("&amp;", raw)

    def test_scenarios_js_schema_matches_doc(self):
        # scenarios_doc 是对象 {instances, default_scenario, ...}：
        # JS 不得把它当数组 find/for..of
        self.assertNotIn("P.scenarios.find", self.html)
        self.assertNotIn("for (const s of P.scenarios)", self.html)
        doc = json.loads((self.out / "data" / "scenarios.json")
                         .read_text(encoding="utf-8"))
        # 全部 11 场景实例必须有 derived（renderScen 读取）
        for inst in doc["instances"]:
            self.assertIn("derived", inst, inst["scenario_id"])
            self.assertIn("N_tokens", inst["derived"])

    def test_t3_remainder_slots_are_4_5_6(self):
        # model.py:335 offset=ratio（overlap 与 should_compress 无关）：
        # T=3 尾块写槽 4/5/6，不是 0/1/2
        doc = json.loads((self.out / "data" / "scenarios.json")
                         .read_text(encoding="utf-8"))
        t3 = next(i for i in doc["instances"]
                  if i["scenario_id"] == "prefill_t3")
        slots = t3["compression"]["state_slots"]["remainder_write"]["slots"]
        self.assertEqual(slots, [4, 5, 6])
        t5 = next(i for i in doc["instances"]
                  if i["scenario_id"] == "prefill_t5")
        # T=5：主块写前 r 槽（overlap 前半来源），尾块写槽 4
        self.assertEqual(
            t5["compression"]["state_slots"]["remainder_write"]["slots"], [4])

    def test_tensor_axis_state_panels_have_data(self):
        # 张量/轴/状态面板：页面数据必须含完整轴名/意义/来源/逻辑物理布局
        payload = json.loads(self._payload_raw())
        tensors = payload["graph"]["tensors"]
        axis_names = {a["name"] for t in tensors for a in t["axes"]}
        self.assertIn("hidden_size", axis_names)
        # L0 结构事实从语义数据派生（JS 常量手写四组计数不可作为唯一来源）
        self.assertIn("plan", payload)
        self.assertIn("groups", payload["plan"])

    def test_state_version_panel_data(self):
        payload = json.loads(self._payload_raw())
        trs = payload["graph"]["state_transitions"]
        self.assertTrue(trs)
        versions = payload["graph"]["state_versions"]
        self.assertTrue(versions)
        # 同源：页面的层规划含版本链
        self.assertIn("decode_version_chain", payload["plan"])

    def test_js_data_paths_script_verification(self):
        """脚本级 JS 行为验证（非浏览器交互验收）：模拟浏览器解析与全部
        初始化/切换/展开的数据访问路径，11 场景不抛路径异常。"""
        payload = json.loads(self._payload_raw())
        G = payload["graph"]
        SC = payload["scenarios"]
        SCEN = SC["instances"]
        # renderScen：default 可定位、每场景字段完整
        self.assertIn(SC["default_scenario"],
                      {s["scenario_id"] for s in SCEN})
        for s in SCEN:
            self.assertIn(s["compression"]["phase"], ("prefill", "decode"))
            for k in ("N_tokens", "N_ctx", "N_blk", "N_vis", "N_sel_c",
                      "remainder"):
                self.assertIn(k, s["derived"], s["scenario_id"])
            self.assertIn("visible_upper", s["indexer"])
        # showEv/showTensor：节点证据与端口张量全部可解析
        EVID = payload["evidence"]["entries"]
        ev_ids = {e["evidence_id"] for e in EVID}
        for n in G["nodes"]:
            for r in n["evidence_refs"]:
                self.assertIn(r, ev_ids)
        tensor_ids = {t["stable_id"] for t in G["tensors"]}
        for n in G["nodes"]:
            for pt in n["input_ports"] + n["output_ports"]:
                self.assertIn(pt["tensor_id"], tensor_ids)
        # 状态面板：transitions + 版本链
        self.assertTrue(G["state_transitions"])
        for c in payload["plan"]["decode_version_chain"]:
            self.assertIn("write_slot", c)
        # L0 派生：43 绑定、唯一 extracted 绑定（layer 4）、world_size
        self.assertEqual(len(payload["plan"]["bindings"]), 43)
        extracted = [b for b in payload["plan"]["bindings"]
                     if b["template_status"] == "extracted"]
        self.assertEqual(sorted(b["layer_id"] for b in extracted),
                         [0, 2, 3, 4])
        self.assertEqual(G["world_size"], 1)
        # evidence schema：JS 用 .entries（对象不是数组）
        self.assertNotIn("for (const e of P.evidence)", self.html)
        self.assertIn("P.evidence.entries", self.html)


if __name__ == "__main__":
    unittest.main()
