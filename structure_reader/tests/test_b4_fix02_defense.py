"""B4 定点补修02 回归：装配图必要链全删目录 / 角色参数真实绑定 / 状态
三态 / 自包含页面测试。

覆盖 codex_b4_review02 四项缺口的关闭证据（自包含，不依赖 coordination
目录）：

1. 装配图必要链：review02 的 71 条 head/Markov/Confidence 必要边在
   本测试内从真实图动态逐条删除（不是只测三个具名 ID）——每份删图
   refs=0、D16_CHAIN_EDGE_MISSING 具名拒绝；正常图 D1-D17 全通过。
   data→control 降级（抽样全量）同样逐条检出。测试自包含：D16 目录
   与注入路径都以当前源码常量为准，不读 coordination（旧 JSON 全部
   只读保留给 Codex 回放，产品测试不依赖）。
2. 集成不变量 D17：dtype/轴目录、共享张量边集（w1/w2/output_ids）、
   win_cache 版本索引——负例经受控错误目录检出。
3. 角色参数：q_norm 等 11 项 inline_op_param（激活不是参数）；24 项
   完整；param_tensor 项 dtype/shape/consumer 联查；last 专有参数
   markov_w2/confidence checkpoint bf16（model.py:728-729/810-811）、
   hc_head_base/scale 消费 gate（fn 才进 mix）。
4. 状态三态：before_state_known 恒 "unknown"；after_slot_count 数值
   分字段；prefill T3/decode P2/P127/P128 断言。
5. 自包含页面测试：页面由临时隔离 _build_documents 生成（不扫真实
   outputs；无已有 candidate 不 ValueError）。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_page_test_module():
    """以文件路径加载 tests/test_b4_fix01_page（tests 非 import 包，
    与既有 per-file 执行口径一致）。"""
    spec = importlib.util.spec_from_file_location(
        "test_b4_fix01_page",
        Path(__file__).resolve().parent / "test_b4_fix01_page.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

import _support  # noqa: E402
from _support import REPO_ROOT  # noqa: E402

from structure_reader.checks.dspark_contracts import (  # noqa: E402
    apply_dspark_mutation,
    build_dspark_validation_report,
    dspark_integration_error_specs,
    validate_dspark_integration_contracts,
)
from structure_reader.checks.graph_contracts import (  # noqa: E402
    validate_graph_references,
)
from structure_reader.extract.dspark_integration import (  # noqa: E402
    build_dspark_integration_graph,
    dspark_role_bindings,
    dspark_scenario_views,
)
from structure_reader.model_registry import find_workspace_root  # noqa: E402
from structure_reader.rules.deepseek_v4_flash_0731 import (  # noqa: E402
    SCENARIOS,
    load_model_values,
    scenario_base_env,
)


class TestAssemblyNecessaryChain(unittest.TestCase):
    """阻断1：装配图必要链边逐删目录（71 条动态全删）+ 正常图通过。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.graph = cls.bundle.graph
        from structure_reader.checks.dspark_contracts import (
            _NECESSARY_CHAIN_EDGES)
        cls.chain_ids = tuple(e[0] for e in _NECESSARY_CHAIN_EDGES)

    def test_original_passes_full_contracts(self):
        self.assertEqual(
            validate_dspark_integration_contracts(self.graph), ())
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_every_necessary_edge_deleted_is_named_rejected(self):
        """71 条必要链边逐条从真实装配图删除：refs=0 + D16 具名拒绝。
        这是 codex_b4_review02「71 条删除后 refs0 集成语义零 issue」的
        完整关闭——不是只测三个具名 ID。"""
        self.assertEqual(len(self.chain_ids), 71)
        for eid in self.chain_ids:
            with self.subTest(edge=eid):
                bad = replace(self.graph, edges=tuple(
                    e for e in self.graph.edges if e.stable_id != eid))
                self.assertEqual(validate_graph_references(bad), (), eid)
                codes = {i.code for i in
                         validate_dspark_integration_contracts(bad)}
                self.assertIn("D16_CHAIN_EDGE_MISSING", codes, eid)
                # 端点/张量核验不误报（删边只报缺失）
                self.assertNotIn("D16_CHAIN_EDGE_ENDPOINTS", codes, eid)

    def test_every_necessary_edge_downgrade_to_control_detected(self):
        """必要链 data→control 降级逐条检出（必要数据流不得降级）。"""
        edge_by_id = {e.stable_id: e for e in self.graph.edges}
        for eid in self.chain_ids:
            with self.subTest(edge=eid):
                new_edges = []
                for e in self.graph.edges:
                    if e.stable_id == eid:
                        e = replace(e, kind="control")
                    new_edges.append(e)
                bad = replace(self.graph, edges=tuple(new_edges))
                self.assertEqual(validate_graph_references(bad), (), eid)
                codes = {i.code for i in
                         validate_dspark_integration_contracts(bad)}
                self.assertIn("D16_CHAIN_EDGE_KIND", codes, eid)

    def test_review02_catalog_all_detected(self):
        """36 项受控错误目录（含 review02 四具名删除断链 + 降级/改接/
        D17 负例）全检出且引用合法。"""
        report = build_dspark_validation_report(self.graph)
        self.assertTrue(report["original_contracts"]["status"] == "pass")
        ce = report["controlled_errors"]
        self.assertTrue(ce["all_detected"])
        self.assertGreaterEqual(ce["catalog_count"], 36)
        ids = {r["mutation_id"] for r in ce["results"]}
        for mid in ("e_cf_concat_float_deleted", "e_cf_proj_squeeze_deleted",
                    "e_ex_mix_gate_deleted", "e_mk_s2_add_sample_deleted",
                    "e_ex_mix_gate_to_control",
                    "e_mk_s2_add_sample_unguarded",
                    "e_cf_proj_squeeze_rewired", "e_ex_mix_gate_rewired",
                    "e_ex_scale_gate_tensor_swapped", "mk_w2_param_copied",
                    "mk_step0_logits_updated_bf16",
                    "confidence_out_axes_swapped"):
            self.assertIn(mid, ids)
        for spec in dspark_integration_error_specs():
            mutated = apply_dspark_mutation(self.graph, spec.mutation_id)
            self.assertEqual(validate_graph_references(mutated), (),
                             spec.mutation_id)


class TestInvariantsD17(unittest.TestCase):
    """D17：dtype/轴目录 + 共享张量边集 + win_cache 版本索引。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph

    def test_win_cache_version_index(self):
        versions = {v.stable_id for v in self.graph.state_versions}
        for i in range(3):
            for label in ("v_hist", "v_after"):
                self.assertIn(
                    f"state:dspark:win_cache:mtp.{i}:{label}", versions)


class TestRoleBindingsRealParams(unittest.TestCase):
    """阻断2：角色参数真实绑定（激活≠参数；24 项完整；dtype/consumer
    联查）。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.rb = dspark_role_bindings(cls.bundle)
        cls.graph = cls.bundle.graph
        cls.tensor_ids = {t.stable_id: t for t in cls.graph.tensors}
        cls.node_ids = {n.stable_id for n in cls.graph.nodes}

    def test_24_template_param_specs_complete(self):
        self.assertEqual(len(self.rb["template_param_bindings"]), 24)
        slots = {p["param_slot"] for p in self.rb["template_param_bindings"]}
        # review02 role_spec_omissions 十项全部在内
        for slot in ("attn.wq_a.weight", "attn.wq_b.weight",
                     "attn.wo_a.weight", "attn.wo_b.weight",
                     "ffn.experts.*.w1.weight", "ffn.experts.*.w2.weight",
                     "ffn.experts.*.w3.weight",
                     "ffn.shared_experts.w1.weight",
                     "ffn.shared_experts.w2.weight",
                     "ffn.shared_experts.w3.weight"):
            self.assertIn(slot, slots)

    def test_q_norm_is_inline_param_not_activation(self):
        qn = next(p for p in self.rb["template_param_bindings"]
                  if p["spec_id"] == "stage.q_norm_w")
        for x in qn["instances"]:
            # 激活张量 tensor:mtp{i}.stage:qr 不得再被标为参数张量
            self.assertNotIn("graph_tensor", x)
            self.assertEqual(x["binding"], "inline_op_param")
            self.assertIn("mtp%d.dspark.q_norm" % x["stage_id"],
                          x["consumer_nodes"])
        # qr 张量本身是 bf16 激活（面板不应把它当参数）
        qr = self.tensor_ids["tensor:mtp0.stage:qr"]
        self.assertEqual(qr.dtype, "bf16")

    def test_param_tensor_bindings_dtype_shape_consumer_verified(self):
        for pb in self.rb["template_param_bindings"]:
            for x in pb["instances"]:
                if x["binding"] != "param_tensor":
                    continue
                t = self.tensor_ids[x["graph_tensor"]]
                self.assertEqual(x["graph_dtype"], t.dtype)
                self.assertEqual(t.dtype, pb["dtype"])
                for cn in x["consumer_nodes"]:
                    self.assertIn(cn, self.node_ids)
                    node = next(n for n in self.graph.nodes
                                if n.stable_id == cn)
                    self.assertTrue(
                        any(p.tensor_id == x["graph_tensor"]
                            for p in node.input_ports),
                        f"{pb['spec_id']}: {cn} 不消费 {x['graph_tensor']}")

    def test_inline_expert_params_wildcard_no_fake_instances(self):
        for spec_id in ("stage.expert_w1_w", "stage.shared_w1_w"):
            pb = next(p for p in self.rb["template_param_bindings"]
                      if p["spec_id"] == spec_id)
            self.assertEqual(
                {x["binding"] for x in pb["instances"]},
                {"inline_op_param"})
            for x in pb["instances"]:
                self.assertNotIn("graph_tensor", x)
                self.assertNotIn("graph_tensor", list(x))

    def test_last_params_checkpoint_dtype_and_consumer_ports(self):
        by_module = {p["module"]: p for p in self.rb["last_only_params"]}
        # model.py:728-729：lm_head checkpoint bf16、runtime fp32
        self.assertEqual(by_module["markov_head.markov_w2"]["dtype"], "fp32")
        self.assertEqual(
            by_module["markov_head.markov_w2"]["checkpoint_dtype"], "bf16")
        # model.py:810-811：confidence proj checkpoint bf16、runtime fp32
        self.assertEqual(
            by_module["confidence_head.proj"]["checkpoint_dtype"], "bf16")
        # hc_head_base/scale 真实消费端口在 gate（fn 才进 mix）
        for m in ("hc_head_base", "hc_head_scale"):
            self.assertEqual(by_module[m]["consumer_node"],
                             "exit.hc_head.gate", m)
        self.assertEqual(by_module["hc_head_fn"]["consumer_node"],
                         "exit.hc_head.mix")
        # 逐字段联查：消费节点有该张量输入端口 + 图中有携带边
        edges = self.graph.edges
        node_by_id = {n.stable_id: n for n in self.graph.nodes}
        for p in self.rb["last_only_params"]:
            node = node_by_id[p["consumer_node"]]
            self.assertTrue(
                any(pp.tensor_id == p["graph_tensor"]
                    for pp in node.input_ports), p["module"])
            self.assertTrue(
                any(e.tensor_id == p["graph_tensor"]
                    and e.dst_node_id == p["consumer_node"]
                    for e in edges), p["module"])


class TestScenarioTriState(unittest.TestCase):
    """阻断3：before 三态 unknown / after 数量分字段 / 假设显式化。"""

    @classmethod
    def setUpClass(cls):
        root = find_workspace_root()
        cls.bundle = build_dspark_integration_graph(root)
        cls.values = load_model_values(root)
        cls.views = dspark_scenario_views(cls.bundle, [
            dict(scenario_base_env(cls.values, s), scenario_id=s.scenario_id)
            for s in SCENARIOS
        ])
        cls.by_id = {v["scenario_id"]: v for v in cls.views}

    def test_before_state_known_always_unknown_tri_state(self):
        for sid, view in self.by_id.items():
            for st in view["stage_states"]:
                self.assertEqual(st["before_state_known"], "unknown", sid)
                self.assertIsInstance(st["before_state_known"], str, sid)
                self.assertIn("显式假设", st["before_state_note"], sid)
                self.assertIn("unknown", st["before_state_note"], sid)

    def test_after_slot_count_numeric_separate(self):
        self.assertEqual(self.by_id["decode_p2"]["stage_states"][0]
                         ["after_slot_count"], 3)
        self.assertEqual(self.by_id["decode_p127"]["stage_states"][0]
                         ["after_slot_count"], 128)
        self.assertEqual(self.by_id["decode_p128"]["stage_states"][0]
                         ["after_slot_count"], 128)
        self.assertEqual(self.by_id["prefill_t3"]["stage_states"][0]
                         ["after_slot_count"], 3)
        # window_cache.valid_slot_count 满窗时是 W（环形覆盖后的有效
        # 容量），after_slot_count 是 P+1 覆盖数量口径——两个数字分开
        self.assertEqual(self.by_id["decode_p127"]["window_cache"]
                         ["valid_slot_count"], 128)
        self.assertEqual(self.by_id["decode_p128"]["window_cache"]
                         ["valid_slot_count"], 128)

    def test_prefill_t3_no_before_after_contradiction(self):
        """prefill_t3：window_cache 说 unknown、stage_states 不再说
        before=true（补修01 的互相矛盾关闭）。"""
        view = self.by_id["prefill_t3"]
        self.assertEqual(view["window_cache"]["valid_length_state"],
                         "unknown")
        for st in view["stage_states"]:
            self.assertEqual(st["before_state_known"], "unknown")
            self.assertNotIn("前提成立", st["before_state_note"])

    def test_p127_p128_assumption_wording(self):
        for sid in ("decode_p127", "decode_p128"):
            view = self.by_id[sid]
            wc = view["window_cache"]
            self.assertIn("显式假设", wc["valid_length_premise"], sid)
            self.assertIn("不证明写前历史", wc["valid_length_premise"], sid)
            self.assertEqual(wc["valid_slot_count"], 128)
        # 数值阈值只用于 after 数量，不写成历史已成立断言
        for sid in ("decode_p127", "decode_p128", "decode_p2", "prefill_t3"):
            self.assertNotIn("前提成立",
                             self.by_id[sid]["stage_states"][0]
                             ["before_state_note"], sid)
            self.assertNotIn("前提成立",
                             self.by_id[sid]["window_cache"]
                             .get("valid_length_premise", ""), sid)


class TestPageBuildSelfContained(unittest.TestCase):
    """阻断4：页面测试自包含——临时隔离 _build_documents 生成当前源码
    页面（无已有 candidate 不 ValueError；改源码测到新页）。"""

    def test_build_isolated_page_produces_current_payload(self):
        import json
        import re as _re
        import tempfile
        page_mod = _load_page_test_module()
        with tempfile.TemporaryDirectory(prefix="b4-fix02-selfcontained-") as td:
            page, build_log = page_mod._build_isolated_page(
                Path(td) / "outputs_root" / "deepseek_v4_flash_0731")
            self.assertTrue(page.is_file())
            self.assertTrue(any("dspark_integration" in line
                                for line in build_log))
            html = page.read_text(encoding="utf-8")
            self.assertIn('id="payload"', html)
            # 页面 payload 是当前源码的真实 role_bindings（24 项 + 三态）
            payload = json.loads(_re.search(
                r'<script id="payload" type="application/json">([\s\S]*?)'
                r'</script>', html).group(1))
            rb = payload["dsparkView"]["integration"]["role_bindings"]
            self.assertEqual(len(rb["template_param_bindings"]), 24)
            qn = next(p for p in rb["template_param_bindings"]
                      if p["spec_id"] == "stage.q_norm_w")
            self.assertEqual(qn["instances"][0]["binding"],
                             "inline_op_param")
            views = payload["dsparkView"]["views"]
            v3 = next(v for v in views if v["scenario_id"] == "prefill_t3")
            self.assertEqual(v3["stage_states"][0]["before_state_known"],
                             "unknown")

    def test_page_test_module_uses_isolated_build(self):
        """test_b4_fix01_page 不再有 mtime 选旧 candidate 的
        _latest_output_dir 依赖。"""
        page_mod = _load_page_test_module()
        self.assertFalse(hasattr(page_mod, "_latest_output_dir"))
        source = Path(page_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("_latest_output_dir", source)
        self.assertIn("_build_isolated_page", source)


if __name__ == "__main__":
    unittest.main()
