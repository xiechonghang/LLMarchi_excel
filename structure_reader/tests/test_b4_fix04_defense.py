"""B4 定点补修04 回归：可信轴值/完整 rank-layout-shape 目录（D19）+
guard 谓词可信定义（D20）。

覆盖 codex_b4_review04 两项缺口的关闭证据（测试自包含：全部期望来自
当前源码独立维护目录与真实图动态变异，不读 .work/coordination——旧
JSON 全部只读保留给 Codex 回放）：

1. D19 可信轴目录：named_axes 逐轴 kind/value/expression/unknown
   （V/Rm/H/K 等配置轴固定值逐字、派生轴 canonical 表达式词法锚定 +
   引用轴实际值公式交叉、N_e 数据相关 unknown 不转 0/false）；轴集合
   闭合（多/少具名拒绝）。目录与 rules AXIS_DEFS 同源冲突核验。
2. D19 张量完整规格目录：全部 264 张量（entry 17 + exit 12 + cf 5 +
   mk 顶层 5 + mk 步 30 + stage 65×3）dtype + 轴符号序（rank）+
   logical/physical layout 逐字段；seed[B] rank1 契约、
   output_ids[B,N_draft+1]、Confidence proj_w[C1_conf,H_plus_Rm] 等
   原组件契约落到实际发布图；张量集合闭合。V 全量同步 129280→129281
   与 seed [B]→[B,C1_exit]（review04 两坏图等价注入）具名拒绝。
3. D20 guard 谓词可信定义：四 guard canonical 谓词词法 + evidence
   锁定 + 白名单求值器真/假边界环境（两相位误并、窗口 inwin/wrap
   互换、阈值偏移）+ 定义/引用闭合 + unknown 不转 false；
   g_decode start_pos>0→>=0（review04 坏图等价注入）具名拒绝。
4. 正常图通过 + 受控错误目录全检出（catalog ≥ 71）。
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _support  # noqa: E402
from _support import REPO_ROOT  # noqa: E402

from structure_reader.checks.dspark_contracts import (  # noqa: E402
    _MK_STEP_TENSOR_TEMPLATE,
    _STAGE_TENSOR_TEMPLATE,
    _TOP_TENSOR_TEMPLATE,
    _TRUSTED_AXIS_DEFS,
    _TRUSTED_GUARD_DEFS,
    _TRUSTED_GUARD_SEMANTICS,
    apply_dspark_mutation,
    build_dspark_validation_report,
    dspark_integration_error_specs,
    dspark_trusted_tensor_catalog,
    validate_dspark_integration_contracts,
)
from structure_reader.checks.graph_contracts import (  # noqa: E402
    validate_graph_references,
)
from structure_reader.extract.dspark_integration import (  # noqa: E402
    build_dspark_integration_graph,
)
from structure_reader.scenario_evaluator import evaluate_tri  # noqa: E402

PROJECT = REPO_ROOT / "structure_reader"


class TestTrustedAxisCatalogD19(unittest.TestCase):
    """D19a：可信轴目录构成 + 与 rules AXIS_DEFS 同源冲突核验。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.graph = cls.bundle.graph

    def test_axis_catalog_shape_and_kinds(self):
        """37 轴逐条 kind 正确：场景轴 expression、N_e unknown、其余
        配置/集成轴 value；目录数与图轴数一致。"""
        self.assertEqual(len(_TRUSTED_AXIS_DEFS), 37)
        self.assertEqual(len(self.graph.named_axes), 37)
        for symbol in ("B", "T", "P"):
            self.assertEqual(_TRUSTED_AXIS_DEFS[symbol][0], "expression",
                             symbol)
        self.assertEqual(_TRUSTED_AXIS_DEFS["N_e"][0], "unknown")
        for symbol in ("V", "Rm", "H", "K", "W", "N_draft",
                       "H_plus_Rm", "N_draft_plus_1"):
            self.assertEqual(_TRUSTED_AXIS_DEFS[symbol][0], "value",
                             symbol)

    def test_axis_catalog_matches_rules_axis_defs(self):
        """目录配置常量值与 rules AXIS_DEFS 同源：任一漂移即本测试失败
        （rules AXIS_DEFS 与 run config 的交叉核对在 load_model_values
        具名执行，两目录值同源一致，防独立漂移）。"""
        from structure_reader.rules.deepseek_v4_flash_0731 import (
            AXIS_BY_SYMBOL)
        for symbol, (kind, expected) in _TRUSTED_AXIS_DEFS.items():
            if kind != "value" or symbol not in AXIS_BY_SYMBOL:
                continue
            rules_val = AXIS_BY_SYMBOL[symbol].value
            self.assertEqual(rules_val, expected, symbol)

    def test_derived_formulas_consistent_with_expressions(self):
        """值型派生轴的公式交叉目录与 expression 轴 canonical 互洽：
        N_mix/N_hcH/D_g/GRo/N_tgtH/N_kvcat/N_qf 同时出现在两目录且
        图内声明一致（expression 轴）。"""
        for symbol, formula in (("N_mix", "(2 + N_hc) * N_hc"),
                                ("N_hcH", "N_hc * H"),
                                ("D_g", "Hq * D // G"),
                                ("GRo", "G * Ro"),
                                ("N_tgtH", "H * N_tgt"),
                                ("N_kvcat", "W + N_draft"),
                                ("N_qf", "Hq * D")):
            self.assertIn(symbol, _TRUSTED_AXIS_DEFS, symbol)
            self.assertEqual(_TRUSTED_AXIS_DEFS[symbol][1], formula,
                             symbol)
            self.assertEqual(_TRUSTED_AXIS_DEFS[symbol][0], "expression",
                             symbol)
            ax = next(a for a in self.graph.named_axes
                      if a.symbol == symbol)
            self.assertEqual(ax.expression, formula, symbol)


class TestTrustedTensorSpecD19(unittest.TestCase):
    """D19b：全部 264 张量完整规格目录 + 集合闭合。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph
        cls.catalog = dspark_trusted_tensor_catalog()

    def test_catalog_size_matches_graph(self):
        """目录条目数 = 图张量数 = 264（stage 65×3 + mk 步 6×5 + 顶层
        39）；模板构成 65/6/39 逐段断言。"""
        self.assertEqual(len(_STAGE_TENSOR_TEMPLATE), 65)
        self.assertEqual(len(_MK_STEP_TENSOR_TEMPLATE), 6)
        self.assertEqual(len(_TOP_TENSOR_TEMPLATE), 39)
        self.assertEqual(len(self.catalog), 264)
        self.assertEqual(len(self.graph.tensors), 264)

    def test_every_tensor_matches_catalog_fieldwise(self):
        """逐张量 dtype + 轴符号序 + logical/physical layout 逐字段
        （全量 264 条遍历，不抽样）。"""
        for t in self.graph.tensors:
            with self.subTest(tensor=t.stable_id):
                self.assertIn(t.stable_id, self.catalog, t.stable_id)
                want_dt, want_axes = self.catalog[t.stable_id]
                self.assertEqual(t.dtype, want_dt, t.stable_id)
                self.assertEqual(tuple(a.symbol for a in t.axes),
                                 tuple(want_axes), t.stable_id)
                self.assertEqual(list(t.logical_layout),
                                 list(want_axes), t.stable_id)
                self.assertEqual(list(t.physical_layout),
                                 list(want_axes), t.stable_id)

    def test_key_contract_tensors_explicit(self):
        """原组件契约关键张量显式断言（seed rank1 int64、output_ids
        [B,N_draft+1]、Confidence 权重 [C1_conf,H_plus_Rm] fp32、head
        权重 [V,H]、embed 权重 [V,H] bf16、main_proj_w fp8）。"""
        catalog = self.catalog
        self.assertEqual(catalog["tensor:entry:seed_ids"],
                         ("int64", ("B",)))
        self.assertEqual(catalog["tensor:mk:output_ids"],
                         ("int64", ("B", "N_draft_plus_1")))
        self.assertEqual(catalog["tensor:cf:proj_w"],
                         ("fp32", ("C1_conf", "H_plus_Rm")))
        self.assertEqual(catalog["tensor:exit:head_w"], ("fp32", ("V", "H")))
        self.assertEqual(catalog["tensor:entry:embed_w"], ("bf16", ("V", "H")))
        self.assertEqual(catalog["tensor:entry:main_proj_w"],
                         ("fp8_e4m3fn", ("H", "N_tgtH")))
        self.assertEqual(catalog["tensor:mtp0.stage:win_cache"],
                         ("bf16", ("B", "W", "D")))

    def test_synced_vocab_axis_mutant_named_rejected(self):
        """review04 mutant_synced_vocab_axis 等价注入：V 全局同步
        129280→129281（named+张量内嵌、symbol/引用合法）→
        D19_AXIS_VALUE_MISMATCH 具名拒绝，original 语义非空。"""
        def bump(a):
            return replace(a, value=129281) if a.symbol == "V" else a
        bad = replace(self.graph,
                      named_axes=tuple(bump(a) for a in self.graph.named_axes),
                      tensors=tuple(
                          replace(t, axes=tuple(bump(a) for a in t.axes))
                          for t in self.graph.tensors))
        self.assertEqual(validate_graph_references(bad), ())
        codes = {i.code for i in validate_dspark_integration_contracts(bad)}
        self.assertIn("D19_AXIS_VALUE_MISMATCH", codes)

    def test_seed_rank2_mutant_named_rejected(self):
        """review04 mutant_seed_rank2 等价注入：seed [B]→[B,C1_exit]
        （借用已有轴、引用合法）→ D19_TENSOR_SPEC 具名拒绝。"""
        c1e = next(a for a in self.graph.named_axes
                   if a.symbol == "C1_exit")
        bad = replace(self.graph, tensors=tuple(
            replace(t, axes=(*t.axes, c1e),
                    logical_layout=(*t.logical_layout, "C1_exit"),
                    physical_layout=(*t.physical_layout, "C1_exit"))
            if t.stable_id == "tensor:entry:seed_ids" else t
            for t in self.graph.tensors))
        self.assertEqual(validate_graph_references(bad), ())
        codes = {i.code for i in validate_dspark_integration_contracts(bad)}
        self.assertIn("D19_TENSOR_SPEC", codes)

    def test_system_axis_negatives(self):
        """D19 系统负例：Rm 同步漂移、N_mix 表达式词法漂移（值同）、
        N_e 未知压 0、layout 转置、stage 逐实例轴序漂移、mk 步张量
        漂移、dtype 漂移、幻影轴/张量——逐项具名检出且引用合法。"""
        for mid in ("rm_axis_sync_257", "n_mix_expression_drift",
                    "ne_axis_collapsed_zero", "cf_proj_w_layout_transposed",
                    "mtp1_win_cache_axes_drift", "mk_step3_written_axes_bv",
                    "seed_axis_dtype_retyped", "phantom_axis_added",
                    "phantom_tensor_added"):
            with self.subTest(mutation=mid):
                mutated = apply_dspark_mutation(self.graph, mid)
                self.assertEqual(validate_graph_references(mutated), (),
                                 mid)
                codes = {i.code for i in
                         validate_dspark_integration_contracts(mutated)}
                self.assertNotEqual(codes, set(), mid)


class TestGuardPredicateD20(unittest.TestCase):
    """D20：四 guard canonical 谓词 + 语义边界 + 闭合 + unknown 三态。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph
        cls.guards = {gd.stable_id: gd for gd in cls.graph.guards}

    def test_guard_catalog_exact(self):
        """恰四 guard 且 canonical/evidence 逐条与图一致。"""
        self.assertEqual(set(self.guards), set(_TRUSTED_GUARD_DEFS))
        for gid, (canonical, ev) in _TRUSTED_GUARD_DEFS.items():
            self.assertEqual(self.guards[gid].predicate_expr, canonical,
                             gid)
            self.assertEqual(tuple(sorted(self.guards[gid].evidence_refs)),
                             tuple(sorted(ev)), gid)

    def test_guard_semantics_boundary_envs(self):
        """canonical 谓词在真/假边界环境求值正确（相位互斥、窗口
        inwin/wrap 互换、阈值 ±1 逐对覆盖）。"""
        for gid, (true_envs, false_envs) in _TRUSTED_GUARD_SEMANTICS.items():
            pred = self.guards[gid].predicate_expr
            for env in true_envs:
                full = {"B": 1, "T": 1, "P": 0,
                        "start_pos": env.get("P", 0), "W": 128, **env}
                r = evaluate_tri(pred, full)
                self.assertTrue(r.is_known and r.value, (gid, env))
            for env in false_envs:
                full = {"B": 1, "T": 1, "P": 0,
                        "start_pos": env.get("P", 0), "W": 128, **env}
                r = evaluate_tri(pred, full)
                self.assertTrue(r.is_known and not r.value, (gid, env))

    def test_decode_guard_prefill_mutation_named_rejected(self):
        """review04 mutant_decode_guard_includes_prefill 等价注入：
        g_decode start_pos > 0 → >= 0（guard_ref ID 全不变）→
        canonical + FALSE_CASE_TRUE 双码具名拒绝。"""
        bad = replace(self.graph, guards=tuple(
            replace(gd, predicate_expr="start_pos >= 0")
            if gd.stable_id == "g_decode" else gd
            for gd in self.graph.guards))
        self.assertEqual(validate_graph_references(bad), ())
        codes = {i.code for i in validate_dspark_integration_contracts(bad)}
        self.assertIn("D20_GUARD_PREDICATE_CANONICAL", codes)
        self.assertIn("D20_GUARD_FALSE_CASE_TRUE", codes)

    def test_system_guard_negatives(self):
        """D20 系统负例：窗口阈值偏移、inwin/wrap 互换、prefill 放宽、
        evidence 删减——逐项具名检出且引用合法。"""
        for mid in ("window_guard_threshold_shifted", "inwin_wrap_swapped",
                    "g_prefill_widened_neq", "g_decode_evidence_dropped"):
            with self.subTest(mutation=mid):
                mutated = apply_dspark_mutation(self.graph, mid)
                self.assertEqual(validate_graph_references(mutated), (),
                                 mid)
                codes = {i.code for i in
                         validate_dspark_integration_contracts(mutated)}
                self.assertNotEqual(codes, set(), mid)

    def test_unknown_not_collapsed_to_false(self):
        """变量缺失时求值必须 unknown（evaluate_tri 契约）：g_decode
        缺 start_pos 环境 → unknown 非 false；graph guard_activations
        为空（场景判断不写入模板）。"""
        r = evaluate_tri("start_pos > 0", {"B": 1, "T": 1, "P": 0, "W": 128})
        self.assertFalse(r.is_known)
        self.assertEqual(r.state, "unknown")
        self.assertEqual(self.graph.guard_activations, ())


class TestOriginalAndCatalog(unittest.TestCase):
    """正常图零 issue + 受控错误目录（71 项）全检出。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph

    def test_original_passes_full_contracts(self):
        self.assertEqual(
            validate_dspark_integration_contracts(self.graph), ())
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_extended_catalog_all_detected(self):
        report = build_dspark_validation_report(self.graph)
        self.assertEqual(report["original_contracts"]["status"], "pass")
        ce = report["controlled_errors"]
        self.assertTrue(ce["all_detected"])
        self.assertEqual(ce["catalog_count"],
                         len(dspark_integration_error_specs()))
        self.assertGreaterEqual(ce["catalog_count"], 71)
        ids = {r["mutation_id"] for r in ce["results"]}
        for mid in ("synced_vocab_axis_129281", "seed_rank2_borrowed_axis",
                    "g_decode_predicate_includes_prefill"):
            self.assertIn(mid, ids)
        for spec in dspark_integration_error_specs():
            mutated = apply_dspark_mutation(self.graph, spec.mutation_id)
            self.assertEqual(validate_graph_references(mutated), (),
                             spec.mutation_id)


class TestCoordinationIndependent(unittest.TestCase):
    """本轮防线测试源码无 coordination 目录 I/O。"""

    def test_fix04_defense_has_no_coordination_io(self):
        import re as _re
        src = Path(__file__).resolve().read_text(encoding="utf-8")
        for m in _re.finditer(
                r"(?:open|read_text|write_text)\(([^)]*)", src):
            args = m.group(1)
            self.assertNotIn(".work", args)
            self.assertNotIn("coordination", args)


if __name__ == "__main__":
    unittest.main()
