"""gate_repairs/task01 红测：D1/D3/D4/G1/G2/D2（从独立报告与源码建立）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.model_registry import find_workspace_root

ROOT = find_workspace_root()


class TestD1GroupStatusConsistent(unittest.TestCase):
    def test_group_extracted_after_real_build(self):
        # D1：build_layer_plan（build_graph=True）后组记录须与提取事实一致
        from structure_reader.layer_plan import build_layer_plan
        plan = build_layer_plan(ROOT, build_graph=True)
        by_id = {g.group_id: g for g in plan.groups}
        self.assertEqual(by_id["csa_topk"].template_status, "extracted",
                         "真实提取后组记录必须标 extracted")
        # V2：四类全部真实构建 → 全部 extracted；build_graph=False 才全 pending
        for gid in ("swa_hash", "csa_hash", "hca_topk"):
            self.assertEqual(by_id[gid].template_status, "extracted", gid)

    def test_group_pending_without_build(self):
        # build_graph=False：无提取事实，组记录不得宣称 extracted
        from structure_reader.layer_plan import build_layer_plan
        plan = build_layer_plan(ROOT, build_graph=False)
        by_id = {g.group_id: g for g in plan.groups}
        for g in plan.groups:
            self.assertEqual(g.template_status, "pending_extraction")


class TestD3PublishChecksExecute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_reference import (
            build_reference_graph)
        cls.graph, _r, _v = build_reference_graph(ROOT)

    def _drop_edge(self, edge_id):
        from structure_reader.checks.deepseek_v4_contracts import (
            apply_contract_mutation)
        return apply_contract_mutation(self.graph, "shift_unconnected")

    def test_delete_shift_publish_detected(self):
        # D3：删 shift→publish 边必须检出（正常图下也要执行，不缩进死代码）
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        from structure_reader.graph_document import SemanticGraph
        from dataclasses import fields as dc_fields
        kept = [e for e in self.graph.edges
                if e.stable_id != "attn.compressor.shift_publish"]
        kwargs = {f.name: getattr(self.graph, f.name)
                  for f in dc_fields(SemanticGraph)}
        kwargs["edges"] = kept
        mutated = SemanticGraph(**kwargs)
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertIn("C13_FINAL_PUBLISH_SOURCE", codes,
                      "删 shift→publish 必须报 C13_FINAL_PUBLISH_SOURCE")

    def test_delete_updated_publish_detected(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        from structure_reader.graph_document import SemanticGraph
        from dataclasses import fields as dc_fields
        kept = [e for e in self.graph.edges
                if e.stable_id != "attn.compressor.updated_publish"]
        kwargs = {f.name: getattr(self.graph, f.name)
                  for f in dc_fields(SemanticGraph)}
        kwargs["edges"] = kept
        mutated = SemanticGraph(**kwargs)
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertIn("C13_FINAL_PUBLISH_SOURCE", codes,
                      "删 updated→publish 必须报 C13_FINAL_PUBLISH_SOURCE")

    def test_normal_graph_no_issues(self):
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        self.assertEqual(validate_model_contracts(self.graph), ())


class TestD4ValidationReportDynamic(unittest.TestCase):
    def test_docstring_count_matches_specs(self):
        # D4：模块 docstring 不得写死旧数量
        import structure_reader.checks.deepseek_v4_contracts as mod
        from structure_reader.checks.deepseek_v4_contracts import (
            contract_error_specs)
        self.assertNotIn("8 项受控错误", mod.__doc__)
        self.assertGreaterEqual(len(contract_error_specs()), 23)

    def test_build_report_records_actual_run(self):
        # validation_report 的 controlled_errors 必须来自 build 期实际运行
        # （每项 detected 布尔），不能写死 all_detected:true
        from structure_reader.checks.deepseek_v4_contracts import (
            contract_error_specs)
        out = self._build()
        val = json.loads((out / "reports" / "validation_report.json")
                         .read_text(encoding="utf-8"))
        ce = val["controlled_errors"]
        self.assertIn("results", ce)
        self.assertGreaterEqual(len(ce["results"]), 26)
        for item in ce["results"]:
            self.assertIn("mutation_id", item)
            self.assertIsInstance(item["detected"], bool)
        # all_detected 是推导值不是字面量
        self.assertEqual(ce["all_detected"],
                         all(r["detected"] for r in ce["results"]))

    def _build(self):
        import json
        import subprocess
        r = subprocess.run(
            [sys.executable,
             str(ROOT / "structure_reader" / "scripts" / "read_model_structure.py"),
             "build", "--model", "deepseek_v4_flash_0731",
             "--scenario-set", "standard", "--json"],
            capture_output=True, text=True, timeout=600, cwd=str(ROOT))
        self.assertEqual(r.returncode, 0, r.stdout)
        return Path(json.loads(r.stdout)["output_dir"])

    def test_report_generated_from_actual_mutation_run(self):
        # 直接调用 _build_validation_report：漏检时 all_detected=false
        from structure_reader.checks.deepseek_v4_contracts import build_validation_report as _build_validation_report
        from structure_reader.checks.deepseek_v4_contracts import (
            apply_contract_mutation, contract_error_specs)
        from structure_reader.extract.deepseek_v4_reference import (
            build_reference_graph)
        graph, _r, _v = build_reference_graph(ROOT)
        report_ok = _build_validation_report(graph)
        self.assertTrue(report_ok["controlled_errors"]["all_detected"])
        from structure_reader.checks.deepseek_v4_contracts import (
            contract_error_specs as _specs)
        # csa_topk 图（V1 提取器）：不适用 mutation（hash 依赖等）被产品按
        # template/spec 过滤——实际 apply 数 ≤ 总数，且逐项 detected 布尔
        total = len(_specs())
        applied = len(report_ok["controlled_errors"]["results"])
        self.assertLess(applied, total, "不适用 mutation 应被过滤")
        self.assertGreater(applied, total - 5,
                           "过滤只应剔除少量不适用项")
        for item in report_ok["controlled_errors"]["results"]:
            self.assertIsInstance(item["detected"], bool)
        # 漏检一个 mutation
        original = apply_contract_mutation
        def broken(g, mid):
            if mid == "qr_edge_deleted":
                return g
            return original(g, mid)
        import structure_reader.checks.deepseek_v4_contracts as cmod
        cmod.apply_contract_mutation = broken
        try:
            report_leak = _build_validation_report(graph)
        finally:
            cmod.apply_contract_mutation = original
        self.assertFalse(report_leak["controlled_errors"]["all_detected"])


import json  # noqa: E402


class TestG1GuardPredicateSemantics(unittest.TestCase):
    """G1：guard 谓词语义核验（改坏 >= 成 >、互换 prefill/decode 可检出）。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_reference import (
            build_reference_graph)
        cls.graph, _r, _v = build_reference_graph(ROOT)
        cls.preds = {g.stable_id: g.predicate_expr for g in cls.graph.guards}

    def _assert_semantic(self, guard_id, *, true_envs, false_envs):
        from structure_reader.scenario_evaluator import evaluate_tri
        base = {"B": 1, "r": 4, "W": 128, "Ki": 512}
        for env in true_envs:
            r = evaluate_tri(self.preds[guard_id], {**base, **env})
            self.assertTrue(r.is_known and r.value,
                            f"{guard_id} 在 {env} 应为 true，谓词 "
                            f"{self.preds[guard_id]!r}")
        for env in false_envs:
            r = evaluate_tri(self.preds[guard_id], {**base, **env})
            self.assertFalse(r.is_known and r.value,
                             f"{guard_id} 在 {env} 应为 false，谓词 "
                             f"{self.preds[guard_id]!r}")

    def test_prefill_decode_phase_boundary(self):
        # T3/T4 边界：prefill vs decode 相位互换必须检出
        self._assert_semantic(
            "g_prefill",
            true_envs=[{"T": 3, "P": 0, "start_pos": 0},
                       {"T": 4, "P": 0, "start_pos": 0}],
            false_envs=[{"T": 1, "P": 2, "start_pos": 2},
                        {"T": 1, "P": 128, "start_pos": 128}])
        self._assert_semantic(
            "g_decode",
            true_envs=[{"T": 1, "P": 2, "start_pos": 2},
                       {"T": 1, "P": 128, "start_pos": 128}],
            false_envs=[{"T": 3, "P": 0, "start_pos": 0}])

    def test_compress_prefill_boundary_ge(self):
        # g_compress_prefill: T>=r —— T4 true / T3 false（改坏 >= 成 > 即检出）
        self._assert_semantic(
            "g_compress_prefill",
            true_envs=[{"T": 4, "P": 0, "start_pos": 0},
                       {"T": 5, "P": 0, "start_pos": 0}],
            false_envs=[{"T": 3, "P": 0, "start_pos": 0}])

    def test_window_wrap_boundary(self):
        # g_window_wrap: T>W —— W-1/W false、W+1 true
        self._assert_semantic(
            "g_window_wrap",
            true_envs=[{"T": 129, "P": 0, "start_pos": 0}],
            false_envs=[{"T": 127, "P": 0, "start_pos": 0},
                        {"T": 128, "P": 0, "start_pos": 0}])

    def test_csa_indexer_conditions(self):
        # r 条件与 Indexer 存在（CSA 代表层 r=4 恒真——真值环境核验）；
        # 错条件的检出由受控错误注入（mutation 换谓词 → 语义期望假值环境
        # 不满足即具名报错）承载，此处验证谓词在真值环境成立
        self._assert_semantic(
            "g_compress_csa",
            true_envs=[{"T": 4, "P": 0, "start_pos": 0},
                       {"T": 5, "P": 0, "start_pos": 0}],
            false_envs=[])
        self._assert_semantic(
            "g_has_indexer",
            true_envs=[{"T": 4, "P": 0, "start_pos": 0}],
            false_envs=[])

    def test_decode_trigger_boundaries(self):
        # decode 触发：P127 true、P2/P128 false
        self._assert_semantic(
            "g_compress_decode",
            true_envs=[{"T": 1, "P": 127, "start_pos": 127}],
            false_envs=[{"T": 1, "P": 2, "start_pos": 2},
                        {"T": 1, "P": 128, "start_pos": 128}])

    def test_mutated_predicates_detected(self):
        # 改坏谓词的受控错误必须被契约检出（不只 ID 存在/常量比较）
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        from structure_reader.graph_document import SemanticGraph
        from dataclasses import fields as dc_fields
        from structure_reader.graph_ir import Guard

        def rebuild_with(new_preds):
            new_guards = [Guard(stable_id=g.stable_id,
                                predicate_expr=new_preds.get(g.stable_id,
                                                             g.predicate_expr),
                                evidence_refs=g.evidence_refs)
                          for g in self.graph.guards]
            kwargs = {f.name: getattr(self.graph, f.name)
                      for f in dc_fields(SemanticGraph)}
            kwargs["guards"] = new_guards
            return SemanticGraph(**kwargs)

        cases = [
            ("ge_broken", "g_compress_prefill", "start_pos == 0 and T > r"),
            ("phase_swapped", "g_prefill", "start_pos > 0"),
            ("wrap_broken", "g_window_wrap", "T >= W"),
            ("indexer_broken", "g_has_indexer", "r > 0"),
        ]
        for name, gid, bad in cases:
            mutated_preds = {gid: bad}
            codes = {i.code for i in
                     validate_model_contracts(rebuild_with(mutated_preds))}
            self.assertTrue(
                any(c.startswith("C16_") for c in codes),
                f"{name}: 改坏 {gid} 谓词未被检出，得到 {codes}")


class TestG2AxisSemantics(unittest.TestCase):
    """G2：核心轴语义约束（qr 末轴 Rq 错成 H 可检出；unknown 不掩盖）。"""

    @classmethod
    def setUpClass(cls):
        from structure_reader.extract.deepseek_v4_reference import (
            build_reference_graph)
        cls.graph, _r, _v = build_reference_graph(ROOT)

    def test_qr_last_axis_is_rq(self):
        qr = next(t for t in self.graph.tensors if t.name == "csa_topk:qr")
        self.assertEqual(qr.axes[-1].symbol, "Rq")

    def test_mutated_axis_detected(self):
        # qr 末轴 Rq 错成 H 必须检出（数字同为 1024? 否——H=4096≠Rq=1024；
        # 更隐蔽的是换 symbol 但值相同的情况也要靠 symbol 约束检出）
        from structure_reader.checks.deepseek_v4_contracts import (
            validate_model_contracts)
        from structure_reader.graph_document import SemanticGraph
        from dataclasses import fields as dc_fields
        from structure_reader.graph_ir import NamedAxis, Tensor
        axis_by = {a.symbol: a for a in self.graph.named_axes}
        new_tensors = []
        for t in self.graph.tensors:
            if t.name == "csa_topk:qr":
                axes = list(t.axes)
                axes[-1] = axis_by["H"]  # 错成 H
                symbols = [a.symbol for a in axes]
                t = Tensor(name=t.name, dtype=t.dtype, axes=axes,
                           logical_layout=symbols, physical_layout=symbols)
            new_tensors.append(t)
        kwargs = {f.name: getattr(self.graph, f.name)
                  for f in dc_fields(SemanticGraph)}
        kwargs["tensors"] = new_tensors
        mutated = SemanticGraph(**kwargs)
        codes = {i.code for i in validate_model_contracts(mutated)}
        self.assertTrue(any(c.startswith("C17_") for c in codes),
                        f"qr 末轴错成 H 未被检出，得到 {codes}")

    def test_unknown_axes_keep_semantics(self):
        # unknown 轴不得被语义约束伪造成已知
        ne = next(a for a in self.graph.named_axes if a.symbol == "N_e")
        self.assertIsNone(ne.value)
        self.assertIsNone(ne.expression)


if __name__ == "__main__":
    unittest.main()
