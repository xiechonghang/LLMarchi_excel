"""B4 定点补修05 回归：轴来源语义目录（D19 来源扩展）+
显式张量秩标注与实际轴数一致（D19 秩标注核对）。

覆盖 codex_b4_review05 两项缺口的关闭证据（测试自包含：全部期望来自
当前源码独立维护目录与真实图动态变异，不读 .work/coordination——旧
JSON 全部只读保留给 Codex 回放）：

1. 轴来源语义（codex_b4_review05 阻断1 的关闭）：37 轴逐轴
   name/meaning/origin/evidence_ref 可信目录（named_axes 与全部张量
   内嵌轴同目录核验）；N_e unknown_reason 可信原文锁定；rules
   AXIS_DEFS 覆盖的 23 轴与 AXIS_BY_SYMBOL 四字段同源核验（两目录
   同源，防独立漂移）。V 数值/符号正确但四字段借用真实 H 轴的同步
   误写（review05 坏图等价注入）具名拒绝——发布页/CSV 消费这些字段，
   不能因数值和 symbol 正确放行错误阅读事实；正确图字节不受影响。
2. 显式秩标注（codex_b4_review05 阻断2 的关闭）：可选
   global_rank/local_rank 非 None 必须等于实际 len(axes)，两字段分别
   核对；None 合法保留（未标注不是错，不把 None 改写为 0、不把 99
   当并行 rank）。seed/Confidence 权重/Markov 初中末步、各字段单独
   错、合法 None、正确明确秩（global_rank=local_rank=len(axes)）全
   覆盖。
3. 正常图通过 + 受控错误目录全检出（catalog ≥ 80，含本轮 9 项新增）。
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
    _TRUSTED_AXIS_SOURCES,
    _TRUSTED_AXIS_UNKNOWN_REASONS,
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
from structure_reader.graph_ir import Tensor  # noqa: E402

PROJECT = REPO_ROOT / "structure_reader"


class TestTrustedAxisSourceCatalog(unittest.TestCase):
    """来源目录构成 + 与 rules AXIS_DEFS 同源 + unknown_reason 锚定。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph
        cls.axes = {a.symbol: a for a in cls.graph.named_axes}

    def test_catalog_covers_all_37_axes(self):
        """目录恰覆盖 37 轴，且与 _TRUSTED_AXIS_SOURCES 键集一致。"""
        self.assertEqual(len(_TRUSTED_AXIS_SOURCES), 37)
        self.assertEqual(set(_TRUSTED_AXIS_SOURCES), set(self.axes))
        self.assertEqual(len(_TRUSTED_AXIS_UNKNOWN_REASONS), 1)
        self.assertIn("N_e", _TRUSTED_AXIS_UNKNOWN_REASONS)

    def test_original_axes_match_catalog_fieldwise(self):
        """正常图 37 轴四字段 + N_e unknown_reason 逐轴逐字段匹配
        （原图零违规的静态前提）。"""
        for symbol, want in _TRUSTED_AXIS_SOURCES.items():
            ax = self.axes[symbol]
            got = (ax.name, ax.meaning, ax.origin, ax.evidence_ref)
            self.assertEqual(got, want, symbol)
        self.assertEqual(self.axes["N_e"].unknown_reason,
                         _TRUSTED_AXIS_UNKNOWN_REASONS["N_e"])

    def test_catalog_same_source_with_rules_axis_defs(self):
        """rules AXIS_DEFS 覆盖的轴（23 条）name/meaning/origin/
        evidence_ref 与 AXIS_BY_SYMBOL 同源：任一字段漂移即本测试失败
        （rules 与 run config 交叉核对在 load_model_values 具名执行，
        两目录来源同源，防独立漂移；集成专属轴不在 rules，独立维护）。"""
        from structure_reader.rules.deepseek_v4_flash_0731 import (
            AXIS_BY_SYMBOL)
        shared = [s for s in _TRUSTED_AXIS_SOURCES if s in AXIS_BY_SYMBOL]
        self.assertEqual(len(shared), 23)
        for symbol in shared:
            ax = AXIS_BY_SYMBOL[symbol]
            want = _TRUSTED_AXIS_SOURCES[symbol]
            got = (ax.name, ax.meaning, ax.origin, ax.evidence_ref)
            self.assertEqual(got, want, symbol)

    def test_vocab_axis_semantics_explicit(self):
        """V/H 来源语义显式断言（review05 混淆对：V=vocab_size/
        词表大小/run config vocab_size/run_config.json:2，H=hidden_size/
        隐藏维度/run config dim/run_config.json:3——互不为对方）。"""
        self.assertEqual(_TRUSTED_AXIS_SOURCES["V"],
                         ("vocab_size", "词表大小", "run config vocab_size",
                          "run_config.json:2"))
        self.assertEqual(_TRUSTED_AXIS_SOURCES["H"],
                         ("hidden_size", "隐藏维度", "run config dim",
                          "run_config.json:3"))
        self.assertNotEqual(_TRUSTED_AXIS_SOURCES["V"],
                            _TRUSTED_AXIS_SOURCES["H"])

    def test_tensor_embedded_axes_match_catalog(self):
        """全部 264 张量内嵌轴四字段与 named 同一可信目录逐字段匹配
        （发布页/CSV 直接消费内嵌字段）。"""
        catalog = _TRUSTED_AXIS_SOURCES
        for t in self.graph.tensors:
            for a_index, ax in enumerate(t.axes):
                with self.subTest(tensor=t.stable_id, axis=ax.symbol):
                    got = (ax.name, ax.meaning, ax.origin, ax.evidence_ref)
                    self.assertEqual(got, catalog[ax.symbol],
                                     (t.stable_id, a_index))


class TestAxisSourceMutations(unittest.TestCase):
    """来源语义同步误写具名拒绝（数值/符号正确不放行错误阅读事实）。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph

    def _codes(self, mutated):
        self.assertEqual(validate_graph_references(mutated), ())
        return {i.code for i in
                validate_dspark_integration_contracts(mutated)}

    def test_review05_source_catalog_mutations_detected(self):
        """V 借 H 四字段 / Rm 借 D / H_plus_Rm 含义漂移 / N_e 未知原因
        误导文本 / N_draft_plus_1 name 漂移——named + 张量内嵌同步注入
        （与 review05 坏图同构），逐项具名检出且引用合法。"""
        expectations = {
            "vocab_source_claims_hidden":
                ("D19_AXIS_SOURCE_MISMATCH", "D19_TENSOR_AXIS_SOURCE"),
            "rm_source_misattributed":
                ("D19_AXIS_SOURCE_MISMATCH", "D19_TENSOR_AXIS_SOURCE"),
            "h_plus_rm_meaning_drifted":
                ("D19_AXIS_SOURCE_MISMATCH", "D19_TENSOR_AXIS_SOURCE"),
            "ne_unknown_reason_misleading":
                ("D19_AXIS_UNKNOWN_REASON_MISMATCH",),
            "n_draft_plus_1_name_drifted":
                ("D19_AXIS_SOURCE_MISMATCH", "D19_TENSOR_AXIS_SOURCE"),
        }
        for mid, want_codes in expectations.items():
            with self.subTest(mutation=mid):
                codes = self._codes(apply_dspark_mutation(self.graph, mid))
                for c in want_codes:
                    self.assertIn(c, codes, (mid, c))

    def test_vocab_claims_hidden_named_and_embedded(self):
        """V 四字段借用 H：named_axes 与全部含 V 张量内嵌轴同时检出
        （发布页 render/structure_page.py 轴表与 export/
        semantic_tables.py tensor_axes CSV 实际消费这些字段）。"""
        def f(a):
            if a.symbol != "V":
                return a
            h = next(x for x in self.graph.named_axes if x.symbol == "H")
            return replace(a, name=h.name, meaning=h.meaning,
                           origin=h.origin, evidence_ref=h.evidence_ref)
        bad = replace(self.graph,
                      named_axes=tuple(f(a) for a in self.graph.named_axes),
                      tensors=tuple(
                          replace(t, axes=tuple(f(a) for a in t.axes))
                          for t in self.graph.tensors))
        codes = self._codes(bad)
        self.assertIn("D19_AXIS_SOURCE_MISMATCH", codes)
        self.assertIn("D19_TENSOR_AXIS_SOURCE", codes)
        named = [i for i in validate_dspark_integration_contracts(bad)
                 if i.code == "D19_AXIS_SOURCE_MISMATCH"]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].path, "named_axes[axis:V]")
        embedded = [i for i in validate_dspark_integration_contracts(bad)
                    if i.code == "D19_TENSOR_AXIS_SOURCE"]
        self.assertGreaterEqual(len(embedded), 20)

    def test_source_single_field_drift_detected(self):
        """仅单字段漂移（evidence_ref 或 origin）同样具名检出（目录
        逐字段核对，非整体比较）。"""
        def f(a):
            return replace(a, evidence_ref="model.py:1") \
                if a.symbol == "K" else a
        bad = replace(self.graph,
                      named_axes=tuple(f(a) for a in self.graph.named_axes),
                      tensors=tuple(
                          replace(t, axes=tuple(f(a) for a in t.axes))
                          for t in self.graph.tensors))
        codes = self._codes(bad)
        self.assertIn("D19_AXIS_SOURCE_MISMATCH", codes)
        self.assertIn("D19_TENSOR_AXIS_SOURCE", codes)

    def test_correct_graph_bytes_unchanged(self):
        """正确图 canonical 字节不被本轮检查改动（不为通过而改图；
        original 契约零 issue 的图序列化前后一致）。"""
        from structure_reader.graph_document import graph_to_json
        before = graph_to_json(self.graph)
        self.assertEqual(
            validate_dspark_integration_contracts(self.graph), ())
        self.assertEqual(graph_to_json(self.graph), before)


class TestTensorRankMetadata(unittest.TestCase):
    """可选秩标注 global_rank/local_rank 与实际轴数一致（两字段分别）。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph
        cls.tensors = {t.stable_id: t for t in cls.graph.tensors}

    def _codes(self, mutated):
        self.assertEqual(validate_graph_references(mutated), ())
        return {i.code for i in
                validate_dspark_integration_contracts(mutated)}

    def test_original_all_none_and_passes(self):
        """正确图 264 张量两秩标注全 None（None=未标注合法，不是错），
        original 契约零 issue。"""
        self.assertEqual(validate_dspark_integration_contracts(self.graph),
                         ())
        for t in self.graph.tensors:
            self.assertIsNone(t.global_rank, t.stable_id)
            self.assertIsNone(t.local_rank, t.stable_id)

    def test_review05_rank99_catalog_mutations_detected(self):
        """seed / Confidence 权重 / Markov 初中末步秩标注 99 注入
        （与 review05 system_probe 同构；Markov step4 末步由
        mk_step2_rank_metadata99 覆盖中步，另有初步 seed/出口负例），
        逐项具名检出且引用合法。"""
        for mid in ("seed_rank_metadata99", "cf_proj_w_rank_metadata_borrowed",
                    "mk_step2_rank_metadata99", "entry_logits_rank_metadata0"):
            with self.subTest(mutation=mid):
                codes = self._codes(
                    apply_dspark_mutation(self.graph, mid))
                self.assertIn("D19_TENSOR_RANK_METADATA", codes)

    def test_seed_rank99_both_fields_named(self):
        """seed axes [B] 不变、双秩标注 99：两字段各自具名拒绝（两
        字段分别核对，不是合并比较）。"""
        bad = replace(self.graph, tensors=tuple(
            replace(t, global_rank=99, local_rank=99)
            if t.stable_id == "tensor:entry:seed_ids" else t
            for t in self.graph.tensors))
        issues = [i for i in validate_dspark_integration_contracts(bad)
                  if i.code == "D19_TENSOR_RANK_METADATA"]
        paths = {i.path for i in issues}
        self.assertEqual(paths, {"tensors[tensor:entry:seed_ids].global_rank",
                                 "tensors[tensor:entry:seed_ids].local_rank"})

    def test_single_field_mismatch_detected(self):
        """仅 global_rank 矛盾（local_rank 保持 None）也具名拒绝；
        仅 local_rank 矛盾同样检出（各字段独立）。"""
        t0 = next(t for t in self.graph.tensors
                  if t.stable_id == "tensor:cf:proj_w")
        for field, wrong, right in (("global_rank", 99, None),
                                    ("local_rank", 99, None)):
            with self.subTest(field=field):
                kw = {field: wrong}
                bad = replace(self.graph, tensors=tuple(
                    replace(t, **kw) if t.stable_id == "tensor:cf:proj_w"
                    else t for t in self.graph.tensors))
                issues = [i for i in
                          validate_dspark_integration_contracts(bad)
                          if i.code == "D19_TENSOR_RANK_METADATA"]
                self.assertEqual(len(issues), 1)
                self.assertEqual(issues[0].path,
                                 f"tensors[tensor:cf:proj_w].{field}")

    def test_one_correct_one_wrong_detected(self):
        """global_rank=2（恰等于实际 rank）+ local_rank=99：两字段分别
        核对，正确字段不放行错误字段（cf:proj_w 实际 rank2）。"""
        bad = replace(self.graph, tensors=tuple(
            replace(t, global_rank=2, local_rank=99)
            if t.stable_id == "tensor:cf:proj_w" else t
            for t in self.graph.tensors))
        issues = [i for i in validate_dspark_integration_contracts(bad)
                  if i.code == "D19_TENSOR_RANK_METADATA"]
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].path,
                         "tensors[tensor:cf:proj_w].local_rank")

    def test_none_preserved_and_explicit_rank_accepted(self):
        """None 合法保留（不要求标注、不改写为 0）；显式正确秩
        global_rank=local_rank=len(axes) 通过（真实标注可用，当且仅当
        与轴数一致）。"""
        seed = self.tensors["tensor:entry:seed_ids"]
        self.assertEqual(len(seed.axes), 1)
        explicitly = replace(seed, global_rank=1, local_rank=1)
        self.assertEqual(explicitly.global_rank, 1)
        # 全图正确 + seed 显式 rank1（与实际一致）仍零 issue
        bad_but_valid = replace(self.graph, tensors=tuple(
            replace(t, global_rank=len(t.axes), local_rank=len(t.axes))
            for t in self.graph.tensors))
        self.assertEqual(
            validate_dspark_integration_contracts(bad_but_valid), ())
        self.assertIsNone(seed.global_rank)  # 原图 None 未被改写

    def test_rank_metadata_checks_run_before_tensor_spec_gate(self):
        """秩标注核对独立于 TENSOR_SPEC 门（spec 通过分支内仍执行；
        seed 双 99 注入时 axes/dtype/layout 全部正确、唯一检出码是
        RANK_METADATA——不是借 SPEC 偶然检出）。"""
        mutated = apply_dspark_mutation(self.graph, "seed_rank_metadata99")
        issues = validate_dspark_integration_contracts(mutated)
        codes = {i.code for i in issues}
        self.assertEqual(codes, {"D19_TENSOR_RANK_METADATA"})


class TestOriginalAndCatalog05(unittest.TestCase):
    """正常图零 issue + 受控错误目录（80 项）全检出。"""

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
        self.assertGreaterEqual(ce["catalog_count"], 80)
        ids = {r["mutation_id"] for r in ce["results"]}
        for mid in ("vocab_source_claims_hidden", "rm_source_misattributed",
                    "ne_unknown_reason_misleading", "seed_rank_metadata99",
                    "cf_proj_w_rank_metadata_borrowed",
                    "mk_step2_rank_metadata99",
                    "entry_logits_rank_metadata0"):
            self.assertIn(mid, ids)
        for spec in dspark_integration_error_specs():
            mutated = apply_dspark_mutation(self.graph, spec.mutation_id)
            self.assertEqual(validate_graph_references(mutated), (),
                             spec.mutation_id)

    def test_tensor_catalog_unchanged_264(self):
        """本轮不扩不缩张量目录：仍 264 条，stage 65×3 + mk 6×5 + 顶层
        39（fix04 成果保持，本轮只加来源/秩标注两维度核验）。"""
        catalog = dspark_trusted_tensor_catalog()
        self.assertEqual(len(catalog), 264)
        self.assertEqual(len(self.graph.tensors), 264)
        self.assertEqual(catalog["tensor:entry:seed_ids"], ("int64", ("B",)))


class TestCoordinationIndependent(unittest.TestCase):
    """本轮防线测试源码无 coordination 目录 I/O。"""

    def test_fix05_defense_has_no_coordination_io(self):
        import re as _re
        src = Path(__file__).resolve().read_text(encoding="utf-8")
        for m in _re.finditer(
                r"(?:open|read_text|write_text)\(([^)]*)", src):
            args = m.group(1)
            self.assertNotIn(".work", args)
            self.assertNotIn("coordination", args)


if __name__ == "__main__":
    unittest.main()
