"""B4 定点补修03 回归：入口/stage 装配必要链（D18）+ 装配图算子身份
（D15 扩展）+ 有向状态版本迁移（D17 方向核验）+ 全图逐删零放行。

覆盖 codex_b4_review03 四项缺口的关闭证据（测试自包含：全部期望来自
当前源码独立维护目录与真实图动态变异，不读 .work/coordination——旧
JSON 全部只读保留给 Codex 回放）：

1. D18 入口/stage 装配必要链：首入口 19 条绝对 ID + stage 模板 84 条
   ×3 实例 + 跨图改接 3×3 条，逐条从真实装配图删除（每份 refs=0、
   D18_CHAIN_EDGE_MISSING 具名拒绝）；kind/guard/迁移引用目录逐边
   期望（合法分支编码在目录内：cache 写三分支 guard 各异、prefill
   passthrough g_prefill、路由 select/control、main 共享路径无 guard）。
2. D18 降级/去 guard/guard 换枝/迁移引用错位/双边轮换改接负例逐项
   检出（data→control、control→data、select→data 同目录核出）。
3. D15 装配图算子目录：Markov 五步全算子（read_col/embed/bias/slice/
   add/logits_write/sample= gumbel_max_sample/write_col）+ Confidence
   squeeze/首入口 draft 链算子身份——sample 换 argmax（边/张量不变）
   具名拒绝；正例全目录逐一匹配。
4. D17 有向版本迁移：win_cache before=v_hist/after=v_after 逐字段
   有向（方向反转即拒绝，不比无序集合）；Markov logits/output_ids 第
   i 步迁移 before=v{i}/after=v{i+1} 有向；stage0 反转不影响其它实例
   （系统负例按实例逐个注入）。
5. 全图 364 条边逐删扫描：refs0 且语义契约零 issue 的删除必须为空
   （review03「202 条删除放行」的完整关闭——不是只测具名 ID）。
6. 张量 dtype 目录扩展（stage HC fp32 升链/骨架 bf16/路由索引整型/
   entry 关键张量）负例检出。
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
    _OP_CATALOG,
    apply_dspark_mutation,
    build_dspark_validation_report,
    dspark_assembly_chain_edges,
    dspark_integration_error_specs,
    validate_dspark_integration_contracts,
)
from structure_reader.checks.graph_contracts import (  # noqa: E402
    validate_graph_references,
)
from structure_reader.extract.dspark_integration import (  # noqa: E402
    build_dspark_integration_graph,
)

PROJECT = REPO_ROOT / "structure_reader"


class TestAssemblyNecessaryChainD18(unittest.TestCase):
    """阻断1：D18 入口/stage 装配必要链逐删目录 + 正常图通过。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.graph = cls.bundle.graph
        cls.d18_ids = tuple(e[0] for e in dspark_assembly_chain_edges())

    def test_catalog_shape(self):
        """目录构成：entry 19 + stage 内部 84×3 + 跨图改接 3×3 = 280。"""
        from structure_reader.checks.dspark_contracts import (
            _ENTRY_CHAIN_EDGES, _STAGE_TEMPLATE_EDGES)
        self.assertEqual(len(_ENTRY_CHAIN_EDGES), 19)
        self.assertEqual(len(_STAGE_TEMPLATE_EDGES), 84)
        self.assertEqual(len(self.d18_ids), 280)
        self.assertEqual(len(set(self.d18_ids)), 280)  # ID 无重复

    def test_original_passes_full_contracts(self):
        self.assertEqual(
            validate_dspark_integration_contracts(self.graph), ())
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_every_d18_edge_deleted_is_named_rejected(self):
        """172 条装配必要链边逐条从真实装配图删除：refs=0 + D18 具名
        拒绝（entry/stage/cache 全覆盖，不抽样）。"""
        for eid in self.d18_ids:
            with self.subTest(edge=eid):
                bad = replace(self.graph, edges=tuple(
                    e for e in self.graph.edges if e.stable_id != eid))
                self.assertEqual(validate_graph_references(bad), (), eid)
                codes = {i.code for i in
                         validate_dspark_integration_contracts(bad)}
                self.assertIn("D18_CHAIN_EDGE_MISSING", codes, eid)
                # 删边只报缺失：端点/张量核验不误报
                self.assertNotIn("D18_CHAIN_EDGE_ENDPOINTS", codes, eid)
                self.assertNotIn("D18_CHAIN_EDGE_TENSOR", codes, eid)

    def test_entry_edges_unconditional_no_guard(self):
        """首入口边两相位无条件：目录全部 guard=None 且实际图一致
        （挂 guard 即 D18_CHAIN_EDGE_GUARD 拒绝）。"""
        from structure_reader.checks.dspark_contracts import (
            _ENTRY_CHAIN_EDGES)
        edge_by_id = {e.stable_id: e for e in self.graph.edges}
        for row in _ENTRY_CHAIN_EDGES:
            eid, _, _, _, _, _, kind, guard, transition = row
            self.assertIsNone(guard, eid)
            self.assertEqual(kind, "data", eid)
            self.assertIsNone(transition, eid)
            self.assertIsNone(edge_by_id[eid].guard_ref, eid)

    def test_cache_write_three_branch_guards_distinct(self):
        """cache 写三分支合法 guard 逐实例保持在目录内（inwin/wrap/
        decode 不并枝）；slot 坐标消费 g_decode；写后读 state 边引用
        本实例迁移。"""
        from structure_reader.checks.dspark_contracts import (
            _stage_chain_edges_for)
        catalog = {e[0]: e for e in _stage_chain_edges_for(1)}
        for eid, want_guard in (("mtp1.e_ds_cachew_inwin",
                                 "g_dspark_prefill_inwin"),
                                ("mtp1.e_ds_cachew_wrap", "g_window_wrap"),
                                ("mtp1.e_ds_cachew_dec", "g_decode"),
                                ("mtp1.e_slot_cachew", "g_decode")):
            self.assertEqual(catalog[eid][7], want_guard, eid)
            self.assertEqual(catalog[eid][6], "data", eid)
        self.assertEqual(catalog["mtp1.e_ds_cache_state"][6], "state")
        self.assertEqual(catalog["mtp1.e_ds_cache_state"][8],
                         "tr_dspark_cache_mtp1")
        # 实际图同值
        edge_by_id = {e.stable_id: e for e in self.graph.edges}
        for eid, want_guard in (("mtp1.e_ds_cachew_inwin",
                                 "g_dspark_prefill_inwin"),
                                ("mtp1.e_ds_cachew_wrap", "g_window_wrap"),
                                ("mtp1.e_ds_cachew_dec", "g_decode")):
            self.assertEqual(edge_by_id[eid].guard_ref, want_guard, eid)

    def test_stage0_vs_middle_rewired_sources_distinct(self):
        """stage0 与 stage1/2 的改接边源不同（编码在目录内）：stage0
        来自 entry.stage_feed_0.h_out、stage1/2 来自前一 stage 真实
        h 输出；prefill passthrough g_prefill 全实例。"""
        from structure_reader.checks.dspark_contracts import (
            _stage_rewired_edges_for)
        by_i = {i: {e[0]: e for e in _stage_rewired_edges_for(i)}
                for i in range(3)}
        self.assertEqual(by_i[0]["mtp0.e_st_hcin_hcpre"][1],
                         "entry.stage_feed_0")
        self.assertEqual(by_i[1]["mtp1.e_st_hcin_hcpre"][1],
                         "mtp0.stage.hc_post_ffn.combine")
        self.assertEqual(by_i[2]["mtp2.e_st_hcin_hcpre"][1],
                         "mtp1.stage.hc_post_ffn.combine")
        for i in range(3):
            self.assertEqual(
                by_i[i][f"mtp{i}.e_st_prefill_in"][7], "g_prefill")

    def test_d18_downgrades_and_rewires_detected(self):
        """D18 负例逐项检出：kind 降级/升格、去 guard、guard 换枝
        （合法分支并枝）、迁移引用跨实例错位、双边轮换改接。"""
        for mid in ("e_ep_norm_feed1_to_control",
                    "mtp0_e_st_dispatch_gate_to_data",
                    "mtp0_e_ds_idx_select_to_data",
                    "mtp1_e_st_hcpre_a_out_unguarded",
                    "mtp2_e_st_cachew_dec_unguarded",
                    "mtp0_e_ds_cachew_inwin_guard_swapped",
                    "mtp0_e_st_prefill_in_guard_swapped",
                    "mtp1_e_ds_cache_state_transition_ref_dangling",
                    "e_ep_cat_proj_rewired",
                    "mtp0_e_st_norm_xsrc_rewired"):
            with self.subTest(mutation=mid):
                mutated = apply_dspark_mutation(self.graph, mid)
                self.assertEqual(validate_graph_references(mutated), (),
                                 mid)
                codes = {i.code for i in
                         validate_dspark_integration_contracts(mutated)}
                self.assertNotEqual(codes, set(), mid)

    def test_every_kind_change_on_d18_edges_detected(self):
        """D18 目录全部 data 边 kind 改 control 逐条检出（同 D16 口径
        的全量降级核验）。"""
        d18_rows = {e[0]: e for e in dspark_assembly_chain_edges()}
        for eid, row in d18_rows.items():
            kind = row[6]
            if kind != "data":
                continue  # select/control/state 另行由 kind 期望核出
            with self.subTest(edge=eid):
                new_edges = [replace(e, kind="control")
                             if e.stable_id == eid else e
                             for e in self.graph.edges]
                bad = replace(self.graph, edges=tuple(new_edges))
                self.assertEqual(validate_graph_references(bad), (), eid)
                codes = {i.code for i in
                         validate_dspark_integration_contracts(bad)}
                self.assertIn("D18_CHAIN_EDGE_KIND", codes, eid)


class TestAssemblyOpIdentityD15(unittest.TestCase):
    """阻断2：装配图算子身份（Markov 五步全算子 + Confidence squeeze +
    首入口 draft 链）。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph
        cls.nodes = {n.stable_id: n for n in cls.graph.nodes}

    def test_markov_five_step_ops_in_catalog_and_match(self):
        """五步 8 算子全在 D15 目录且实际图逐一匹配（gumbel_max_sample
        身份在装配图保持，非仅组件图）。"""
        want = {"read_col": "col_read", "embed": "embedding",
                "bias": "linear", "slice": "logits_slice",
                "add": "bias_add", "logits_write": "logits_write",
                "sample": "gumbel_max_sample", "write_col": "col_write"}
        catalog = dict(_OP_CATALOG)
        for s in range(5):
            for suffix, op in want.items():
                nid = f"mk.step{s}.{suffix}"
                self.assertEqual(catalog.get(nid), op, nid)
                self.assertEqual(self.nodes[nid].op_type, op, nid)

    def test_sample_op_swap_to_argmax_detected(self):
        """mk.step2.sample gumbel_max_sample→argmax（边/张量/端口全不
        变，仅算子身份偷换）：D15_OP_MISMATCH 具名拒绝。"""
        bad = replace(self.graph, nodes=tuple(
            replace(n, op_type="argmax")
            if n.stable_id == "mk.step2.sample" else n
            for n in self.graph.nodes))
        self.assertEqual(validate_graph_references(bad), ())
        codes = {i.code for i in validate_dspark_integration_contracts(bad)}
        self.assertIn("D15_OP_MISMATCH", codes)
        # 其余四步同目录核出（系统负例不限于 step2）
        for s in (0, 1, 3, 4):
            bad_s = replace(self.graph, nodes=tuple(
                replace(n, op_type="argmax")
                if n.stable_id == f"mk.step{s}.sample" else n
                for n in self.graph.nodes))
            codes_s = {i.code for i in
                       validate_dspark_integration_contracts(bad_s)}
            self.assertIn("D15_OP_MISMATCH", codes_s, f"step{s}")

    def test_confidence_and_entry_ops_identity(self):
        """cf.squeeze=squeeze_last 与首入口 draft 链算子身份在装配图
        保持（目录 + 实际图双核）。"""
        catalog = dict(_OP_CATALOG)
        for nid, op in (("cf.squeeze", "squeeze_last"),
                        ("entry.noise_fill", "fill_const"),
                        ("entry.first_col_write", "index_write"),
                        ("entry.embed_lookup", "embedding"),
                        ("entry.hc_expand", "unsqueeze_repeat"),
                        ("entry.stage_feed_0", "stage_dispatch")):
            self.assertEqual(catalog.get(nid), op, nid)
            self.assertEqual(self.nodes[nid].op_type, op, nid)


class TestDirectedStateVersionsD17(unittest.TestCase):
    """阻断3：有向状态版本迁移逐字段核验（不比无序集合）。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph

    def _reversed_transition_graph(self, tid: str):
        transitions = []
        for t in self.graph.state_transitions:
            if t.stable_id == tid:
                t = replace(t, before_version_id=t.after_version_id,
                            after_version_id=t.before_version_id)
            transitions.append(t)
        return replace(self.graph, state_transitions=tuple(transitions))

    def test_win_cache_direction_fieldwise(self):
        """三实例 win_cache 迁移 before=v_hist/after=v_after 有向逐字段
        （方向反转即 D17_WIN_CACHE_TRANSITION_DIRECTION）。"""
        trs = {t.stable_id: t for t in self.graph.state_transitions}
        for i in range(3):
            tid = f"tr_dspark_cache_mtp{i}"
            self.assertEqual(
                trs[tid].before_version_id,
                f"state:dspark:win_cache:mtp.{i}:v_hist", tid)
            self.assertEqual(
                trs[tid].after_version_id,
                f"state:dspark:win_cache:mtp.{i}:v_after", tid)

    def test_win_cache_reversal_per_instance_detected(self):
        """stage0/1/2 各自反转逐一具名拒绝（系统负例按实例，不靠其它
        实例受控失败偶然挡住）；反转只报本实例迁移路径，不误伤其它
        实例的迁移检查。"""
        for i in range(3):
            with self.subTest(stage=i):
                bad = self._reversed_transition_graph(
                    f"tr_dspark_cache_mtp{i}")
                self.assertEqual(validate_graph_references(bad), ())
                issues = validate_dspark_integration_contracts(bad)
                dir_issues = [x for x in issues if x.code ==
                              "D17_WIN_CACHE_TRANSITION_DIRECTION"]
                want_path = "state_transitions[tr_dspark_cache_mtp%d]" % i
                self.assertEqual([x.path for x in dir_issues], [want_path],
                                 f"stage{i} 反转只具名本实例迁移")

    def test_markov_transitions_direction_fieldwise(self):
        """Markov logits/output_ids 第 i 步迁移 before=v{i}/after=v{i+1}
        有向逐字段（方向反转即拒绝）。"""
        trs = {t.stable_id: t for t in self.graph.state_transitions}
        for state, prefix in (("mk:logits", "tr_mk_logits_s"),
                              ("mk:output_ids", "tr_mk_ids_s")):
            for s in range(5):
                tid = f"{prefix}{s}"
                self.assertEqual(trs[tid].before_version_id,
                                 f"state:{state}:v{s}", tid)
                self.assertEqual(trs[tid].after_version_id,
                                 f"state:{state}:v{s + 1}", tid)

    def test_markov_transition_reversal_detected(self):
        """logits/output_ids 各抽步反转逐一具名拒绝（每条迁移都测，
        10 条全量）。"""
        for state, prefix in (("mk:logits", "tr_mk_logits_s"),
                              ("mk:output_ids", "tr_mk_ids_s")):
            for s in range(5):
                with self.subTest(transition=f"{prefix}{s}"):
                    bad = self._reversed_transition_graph(f"{prefix}{s}")
                    self.assertEqual(validate_graph_references(bad), ())
                    codes = {c.code for c in
                             validate_dspark_integration_contracts(bad)}
                    self.assertIn("D17_MARKOV_TRANSITION_DIRECTION", codes)


class TestZeroDeletionAcceptance(unittest.TestCase):
    """阻断4：全图 364 条边逐删扫描零放行（review03「202 条删除放行」
    的完整关闭）+ 受控错误目录（55 项）全检出。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_dspark_integration_graph().graph

    def test_no_edge_deletion_passes_silent(self):
        """逐边删除：不存在 refs0 且语义契约零 issue 的删除（可选边/
        必要边/改接边全部在 D16/D18 或其它契约目录内具名检出）。"""
        for e in self.graph.edges:
            with self.subTest(edge=e.stable_id):
                bad = replace(self.graph, edges=tuple(
                    x for x in self.graph.edges
                    if x.stable_id != e.stable_id))
                refs_ok = not validate_graph_references(bad)
                issues = validate_dspark_integration_contracts(bad)
                self.assertTrue(
                    refs_ok is False or bool(issues),
                    f"边 {e.stable_id} 删除后 refs0 且零语义 issue——"
                    f"装配语义放行")

    def test_extended_catalog_all_detected(self):
        """55 项受控错误目录（review03 五坏图等价 + D18 降级/改接/相位
        分支 + D17 有向迁移反转）全检出且注入后引用合法。"""
        report = build_dspark_validation_report(self.graph)
        self.assertEqual(report["original_contracts"]["status"], "pass")
        ce = report["controlled_errors"]
        self.assertTrue(ce["all_detected"])
        self.assertEqual(ce["catalog_count"],
                         len(dspark_integration_error_specs()))
        self.assertGreaterEqual(ce["catalog_count"], 55)
        ids = {r["mutation_id"] for r in ce["results"]}
        for mid in ("e_ep_cat_proj_deleted",
                    "mtp0_e_st_hcpre_a_out_deleted",
                    "mtp0_e_ds_cache_state_deleted",
                    "mk_step2_sample_argmax",
                    "mtp0_cache_transition_reversed",
                    "mk_ids_s2_transition_reversed",
                    "mk_logits_s0_transition_reversed",
                    "entry_main_proj_w_dtype_swapped",
                    "mtp1_stage_x_hc_attn_flat_bf16"):
            self.assertIn(mid, ids)
        for spec in dspark_integration_error_specs():
            mutated = apply_dspark_mutation(self.graph, spec.mutation_id)
            self.assertEqual(validate_graph_references(mutated), (),
                             spec.mutation_id)


class TestCoordinationIndependent(unittest.TestCase):
    """产品测试不读 coordination：防线测试源码无 coordination 目录 I/O
    （旧 JSON 只读保留给 Codex 回放，测试行为不因本地文件存在与否变化）。
    检查 I/O 模式（read_text/open 的目标路径），不误伤 docstring 里的
    历史溯源文字。"""

    def test_defense_tests_have_no_coordination_io(self):
        import re as _re
        for name in ("test_b4_fix02_defense.py", "test_b4_fix03_defense.py",
                     "test_b4_fix01_defense.py"):
            src = (Path(__file__).resolve().parent / name).read_text(
                encoding="utf-8")
            # 任何 open/read_text/write_text 调用的实参里不得出现
            # coordination 目录路径段（.work / codex_b4_review*）
            for m in _re.finditer(
                    r"(?:open|read_text|write_text)\(([^)]*)", src):
                args = m.group(1)
                self.assertNotIn(".work", args, name)
                self.assertNotIn("coordination", args, name)


if __name__ == "__main__":
    unittest.main()
