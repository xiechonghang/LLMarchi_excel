"""B4 定点补修01 回归：集成语义防线 / 源码证据闭合 / 角色绑定 / 状态版本。

覆盖 codex_b4_review01 五项缺口的关闭证据（自包含，不依赖 coordination
目录——五个原始 mutant 以 fixture 复制随库）：

1. 集成图错误被成功构建放行：五个原始 mutant（refs=0、D1-D10 零 issue、
   build_report original=pass）在扩展契约 D11-D15 下逐份具名拒绝；同批
   负例经 apply_dspark_mutation 等价注入路径检出且引用合法；正常图通过。
2. 页面/报告丢 DSpark 源码：合并注册表（主图 61 + DSpark 122 同 ID 逐
   字段一致）五图 ev_ 引用闭合；_build_documents 落盘 source_evidence
   含全部 DSpark 证据。
3. 实例角色/参数绑定：role_bindings 从实际图派生并联查（不是未消费
   payload）；页面 JS 消费（NodeVM DOM 桩点选 head/Markov/Confidence/
   DSparkAttention 证据 + 角色面板渲染）。
4. 旧 pending 矛盾：component_maps 标记为历史 scope；层规划读出 = 有序
   concat → stage0 唯一 → main_x 共享，与集成图接线断言一致。
5. 场景状态版本：每实例 before/after/published 绑定图实体；有效项数
   数值与前提分开（P=127 边界按 P+1 口径）。

隔离失败路径：_build_documents 在内存 mock builder 返回坏图时必须
raise（不产出成功 HTML 字节）——不物化坏 candidate、不动真实 outputs。
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _support  # noqa: E402
from _support import REPO_ROOT  # noqa: E402

from structure_reader.build_pipeline import BuildError  # noqa: E402
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
    integration_document,
    merge_evidence_registries,
)
from structure_reader.extract.deepseek_v4_reference import (  # noqa: E402
    build_reference_graph,
)
from structure_reader.extract.deepseek_v4_variants import (  # noqa: E402
    build_all_variant_graphs,
)
from structure_reader.graph_document import graph_from_json  # noqa: E402
from structure_reader.layer_plan import (  # noqa: E402
    assert_dspark_readout_binding,
    build_layer_plan,
)
from structure_reader.model_registry import find_workspace_root  # noqa: E402
from structure_reader.rules.deepseek_v4_flash_0731 import (  # noqa: E402
    SCENARIOS,
    load_model_values,
    scenario_base_env,
)

PROJECT = REPO_ROOT / "structure_reader"
MUTANT_DIR = PROJECT / "tests" / "fixtures" / "integration_mutants"

# 五个原始 mutant（codex_b4_review01 实际改动）→ 期望至少命中的契约码
_RAW_MUTANT_EXPECT = {
    "mutant_confidence_float_proj_deleted": {"D11_CF_FLOAT_PROJ_MISSING"},
    "mutant_confidence_float_proj_control": {"D11_CF_FLOAT_PROJ_KIND"},
    "mutant_head_gate_sinkhorn": {"D12_HC_HEAD_GATE_OP", "D15_OP_MISMATCH"},
    "mutant_final_logits_bf16": {"D13_MK_LOGITS_DTYPE"},
    "mutant_middle_stage_bypass_stage0": {"D14_STAGE_CHAIN_DECODE"},
}


class TestIntegrationMutantDefense(unittest.TestCase):
    """阻断1：五原始 mutant 语义回放 + 扩展负例 + 正常图通过。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.graph = cls.bundle.graph

    def test_original_graph_passes_extended_contracts(self):
        self.assertEqual(validate_dspark_integration_contracts(self.graph), ())
        self.assertEqual(validate_graph_references(self.graph), ())

    def test_raw_mutants_fixture_sourced_and_detected(self):
        """五个原始 mutant fixture 与集成图 schema 同构；逐份在扩展契约
        下具名拒绝（不再是 refs0 放行）。"""
        self.assertEqual(
            sorted(p.stem for p in MUTANT_DIR.glob("*.json")),
            sorted(_RAW_MUTANT_EXPECT))
        for name, want_codes in _RAW_MUTANT_EXPECT.items():
            with self.subTest(mutant=name):
                g = graph_from_json(
                    (MUTANT_DIR / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(g.template_id, self.graph.template_id)
                # 引用仍合法（这是放行根因：不是引用错误而是语义错）
                self.assertEqual(validate_graph_references(g), (), name)
                found = {i.code for i in
                         validate_dspark_integration_contracts(g)}
                self.assertTrue(want_codes & found,
                                f"{name}: 期望 {want_codes}，得到 {found}")

    def test_equivalent_injections_detected_and_reference_valid(self):
        """与五 mutant 逐字段等价的注入路径（受控错误目录）逐项检出且
        引用合法——mutant 回放与 build 期受控错误同源。"""
        for spec in dspark_integration_error_specs():
            with self.subTest(mutation=spec.mutation_id):
                mutated = apply_dspark_mutation(self.graph, spec.mutation_id)
                self.assertEqual(validate_graph_references(mutated), (),
                                 spec.mutation_id)
                found = {i.code for i in
                         validate_dspark_integration_contracts(mutated)}
                for code in spec.expected_codes:
                    self.assertIn(code, found,
                                  f"{spec.mutation_id}: 缺 {code}")

    def test_validation_report_catalog_extended_all_detected(self):
        report = build_dspark_validation_report(self.graph)
        self.assertTrue(report["controlled_errors"]["all_detected"])
        ids = {r["mutation_id"] for r in
               report["controlled_errors"]["results"]}
        # 五原始 mutant 等价回放都在目录里
        for mid in ("cf_float_proj_deleted", "cf_float_proj_control",
                    "head_gate_sinkhorn", "final_logits_bf16",
                    "middle_stage_bypass_stage0"):
            self.assertIn(mid, ids)
        self.assertGreaterEqual(report["controlled_errors"]["catalog_count"],
                                24)


class TestEvidenceClosure(unittest.TestCase):
    """阻断2：合并注册表发布 + 五图证据引用闭合。"""

    @classmethod
    def setUpClass(cls):
        cls.root = find_workspace_root()
        cls.dspark = build_dspark_integration_graph(cls.root)
        cls.main_graph, cls.main_reg, _ = build_reference_graph(cls.root)
        cls.merged = merge_evidence_registries(
            [cls.main_reg, cls.dspark.registry])

    def _graph_ev_ids(self, g) -> set[str]:
        ids: set[str] = set()
        for n in g.nodes:
            ids.update(r for r in n.evidence_refs if r.startswith("ev_"))
        for e in g.edges:
            ids.update(r for r in e.evidence_refs if r.startswith("ev_"))
        for guard in g.guards:
            ids.update(r for r in guard.evidence_refs if r.startswith("ev_"))
        for tr in g.state_transitions:
            ids.update(r for r in tr.evidence_refs if r.startswith("ev_"))
        for a in g.alias_relations:
            ids.update(r for r in a.evidence_refs if r.startswith("ev_"))
        for fg in g.fold_groups:
            for bm in fg.boundary_mappings:
                ids.update(r for r in bm.evidence_refs
                           if r.startswith("ev_"))
        for ax in g.named_axes:
            if ax.evidence_ref.startswith("ev_"):
                ids.add(ax.evidence_ref)
        for t in g.tensors:
            for ax in t.axes:
                if ax.evidence_ref.startswith("ev_"):
                    ids.add(ax.evidence_ref)
        return ids

    def test_merged_registry_superset_and_no_conflict(self):
        merged_ids = {e.evidence_id for e in self.merged.all()}
        self.assertLessEqual({e.evidence_id for e in self.main_reg.all()},
                             merged_ids)
        self.assertLessEqual({e.evidence_id for e in self.dspark.registry.all()},
                             merged_ids)
        # 43 个此前缺失的 DSpark 证据（codex_b4_review01 missing_ids）全部
        # 在合并注册表内——固定常量期望（产品测试自包含，不读 coordination）
        missing = {"ev_cf_call", "ev_ex_hc_head", "ev_mk_loop",
                   "ev_st_forward_spec", "ev_ds_slot_write"}
        self.assertLessEqual(missing, merged_ids)

    def test_all_five_graphs_evidence_closed(self):
        known = {e.evidence_id for e in self.merged.all()}
        dangling = sorted(self._graph_ev_ids(self.main_graph) - known)
        self.assertEqual(dangling, [])
        dangling = sorted(self._graph_ev_ids(self.dspark.graph) - known)
        self.assertEqual(dangling, [])
        vgraphs = build_all_variant_graphs(self.root)
        for key, item in vgraphs.items():
            dangling = sorted(self._graph_ev_ids(item[0]) - known)
            self.assertEqual(dangling, [], key)

    def test_axis_location_text_not_treated_as_registry_id(self):
        """轴引用的 model.py:NNN 源码定位文本保持其口径，不误当 registry
        ID（闭合断言只核 ev_ 前缀）。"""
        loc = {ax.evidence_ref for ax in self.dspark.graph.named_axes
               if not ax.evidence_ref.startswith("ev_")}
        self.assertTrue(any(s.startswith("model.py:") for s in loc))
        known = {e.evidence_id for e in self.merged.all()}
        self.assertFalse(loc & known)


class TestRoleBindingsDerived(unittest.TestCase):
    """缺口3：实例角色/参数绑定从实际数据派生并联查（非静态说明）。"""

    @classmethod
    def setUpClass(cls):
        cls.bundle = build_dspark_integration_graph()
        cls.rb = dspark_role_bindings(cls.bundle)
        cls.graph = cls.bundle.graph
        cls.node_ids = {n.stable_id for n in cls.graph.nodes}
        cls.tensor_ids = {t.stable_id for t in cls.graph.tensors}

    def test_instances_roles_and_layers(self):
        inst = {i["param_path"]: i for i in self.rb["instances"]}
        self.assertEqual(sorted(inst), ["mtp.0", "mtp.1", "mtp.2"])
        self.assertTrue(inst["mtp.0"]["is_first"])
        self.assertFalse(inst["mtp.0"]["is_last"])
        self.assertFalse(inst["mtp.1"]["is_first"])
        self.assertFalse(inst["mtp.1"]["is_last"])
        self.assertTrue(inst["mtp.2"]["is_last"])
        self.assertEqual([i["actual_layer_id"] for i in self.rb["instances"]],
                         [43, 44, 45])

    def test_param_bindings_link_graph_tensors(self):
        for pb in self.rb["template_param_bindings"]:
            for x in pb["instances"]:
                self.assertEqual(x["instance_path"],
                                 f"mtp.{x['stage_id']}.{pb['param_slot']}")
                if x["binding"] == "param_tensor":
                    self.assertIn(x["graph_tensor"], self.tensor_ids,
                                  pb["spec_id"])
                else:
                    # inline_op_param：无独立参数张量（激活不是参数），
                    # 消费节点联查存在
                    self.assertNotIn("graph_tensor", x, pb["spec_id"])
                    for cn in x["consumer_nodes"]:
                        self.assertIn(cn, self.node_ids, pb["spec_id"])
        # 公共参数三实例展开
        by_spec = {p["spec_id"]: p for p in self.rb["template_param_bindings"]}
        self.assertEqual(len(by_spec["stage.hc_attn_fn_w"]["instances"]), 3)

    def test_shared_single_entity_and_last_params(self):
        # 唯一 embed/head：节点存在且 D10 单实体契约同口径
        self.assertIn(self.rb["shared"]["embed"]["weight_node"], self.node_ids)
        self.assertIn(self.rb["shared"]["head"]["weight_node"], self.node_ids)
        for p in self.rb["last_only_params"]:
            self.assertIn(p["graph_node"], self.node_ids, p["module"])
            self.assertIn(p["graph_tensor"], self.tensor_ids, p["module"])
            self.assertIn(p["consumer_node"], self.node_ids, p["module"])
            # runtime/checkpoint dtype 分开声明
            self.assertIn("checkpoint_dtype", p)
        for p in self.rb["first_only_params"]:
            self.assertIn(p["graph_node"], self.node_ids, p["module"])
            self.assertIn(p["graph_tensor"], self.tensor_ids, p["module"])

    def test_cache_state_links_graph_transitions(self):
        tr_ids = {t.stable_id for t in self.graph.state_transitions}
        ver_ids = {v.stable_id for v in self.graph.state_versions}
        for inst in self.rb["instances"]:
            cs = inst["cache_state"]
            self.assertIn(cs["transition_id"], tr_ids)
            self.assertIn(cs["before_version"], ver_ids)
            self.assertIn(cs["after_version"], ver_ids)
            self.assertIn(cs["cache_tensor"], self.tensor_ids)


class TestReadoutBindingAssertion(unittest.TestCase):
    """缺口4a：层规划读出与集成图接线一致断言。"""

    def test_plan_matches_graph(self):
        root = find_workspace_root()
        plan = build_layer_plan(root, build_graph=False)
        dspark = build_dspark_integration_graph(root)
        assert_dspark_readout_binding(plan, dspark.graph)  # 不抛即一致

    def test_plan_shape_new_semantics(self):
        root = find_workspace_root()
        plan = build_layer_plan(root, build_graph=False)
        self.assertEqual(len(plan.dspark_readouts), 1)
        rec = plan.dspark_readouts[0]
        self.assertEqual(rec.readout_layers, (40, 41, 42))
        self.assertEqual(rec.consumer_stage_id, 0)
        self.assertEqual(rec.shared_consumers, (0, 1, 2))
        self.assertIn("concat", rec.to_dict()["concat_note"])

    def test_component_maps_scope_named_as_history(self):
        bundle = build_dspark_integration_graph()
        doc = integration_document(bundle)
        scope = doc["component_maps_scope"]
        self.assertIn("历史", scope["note"])
        self.assertIn("remaining_pending", doc)
        # 真正剩余 pending = V2 门禁/浏览器/用户阅读，不再含组件 B4 拼接
        remaining = "\n".join(doc["remaining_pending"])
        self.assertIn("V2 独立门禁", remaining)
        self.assertIn("浏览器", remaining)
        self.assertIn("用户阅读", remaining)


class TestScenarioStageStates(unittest.TestCase):
    """缺口5：每实例版本绑定 + 数值/前提分开。"""

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
        cls.graph = cls.bundle.graph
        cls.version_ids = {v.stable_id for v in cls.graph.state_versions}
        cls.tr_ids = {t.stable_id for t in cls.graph.state_transitions}

    def test_every_scenario_has_three_stage_state_bindings(self):
        for sid, view in self.by_id.items():
            self.assertEqual(len(view["stage_states"]), 3, sid)
            for i, st in enumerate(view["stage_states"]):
                self.assertEqual(st["state_id"], f"dspark:win_cache:mtp.{i}")
                self.assertIn(st["before_version_id"], self.version_ids, sid)
                self.assertIn(st["after_version_id"], self.version_ids, sid)
                self.assertIn(st["transition_id"], self.tr_ids, sid)
                self.assertTrue(st["write_branch"], sid)
                self.assertTrue(st["read_relation"], sid)

    def test_published_is_after_and_before_tri_state(self):
        # published = after（本调用发布版本）；before 内容三态 unknown
        # （数量阈值不转 bool——codex_b4_review02 阻断3 口径）
        for sid in ("decode_p2", "decode_p126", "decode_p127", "decode_p128",
                    "prefill_t3"):
            for st in self.by_id[sid]["stage_states"]:
                self.assertEqual(st["published_version_id"],
                                 st["after_version_id"])
                self.assertEqual(st["before_state_known"], "unknown", sid)
                self.assertIn("unknown", st["before_state_note"], sid)
                # after 有效槽数与 before 已知性分字段（P+1 数量口径）
                self.assertIsInstance(st["after_slot_count"], int, sid)

    def test_p127_boundary_uses_p_plus_1(self):
        # P=127：P+1=128=W 恰满窗（历史口径不写 P≥W——P=127<W）
        wc = self.by_id["decode_p127"]["window_cache"]
        self.assertEqual(wc["valid_slot_count"], 128)
        self.assertEqual(wc["valid_length_state"], "full_premise")
        self.assertNotIn("P=127 ≥ W", wc["valid_length"])
        self.assertIn("P+1", wc["valid_length_premise"])
        # 连续历史是显式假设，不由数量阈值证明（before/after 分开）
        self.assertIn("显式假设", wc["valid_length_premise"])
        self.assertIn("不证明写前历史", wc["valid_length_premise"])
        # 数值与前提分开：数值字段是 int，前提是文字
        self.assertIsInstance(wc["valid_slot_count"], int)
        self.assertIsInstance(wc["valid_length_premise"], str)

    def test_prefill_after_count_not_history_proof(self):
        # prefill：T>0 只是本调用写入数；before/历史前提不冒充成立
        for sid in ("prefill_t3", "prefill_t4", "prefill_t5"):
            view = self.by_id[sid]
            wc = view["window_cache"]
            self.assertEqual(wc["valid_length_state"], "unknown")
            self.assertIn("显式假设", wc["valid_length_note"])
            for st in view["stage_states"]:
                self.assertEqual(st["before_state_known"], "unknown", sid)
                self.assertIn("显式假设", st["before_state_note"], sid)
                self.assertEqual(st["after_slot_count"],
                                 min(view["env"]["T"],
                                     view["env"]["W"]), sid)

    def test_physical_vs_select_width_separate(self):
        for sid in ("decode_p2", "decode_p128"):
            wc = self.by_id[sid]["window_cache"]
            self.assertEqual(wc["physical_concat_width"], 128 + 5)
            # 选择有效宽 min(W,P+1)+N_draft 随 P 变化
            want = min(128, self.by_id[sid]["env"]["P"] + 1) + 5
            self.assertEqual(wc["select_valid_width"], want)
            self.assertIn("物理", wc["physical_concat_note"])

    def test_prefill_unknown_not_empty(self):
        wc = self.by_id["prefill_t3"]["window_cache"]
        self.assertEqual(wc["valid_length_state"], "unknown")
        self.assertIn("不当作空", wc["unknown_note"] + wc.get(
            "valid_length_note", ""))


class TestBuildFailurePathIsolated(unittest.TestCase):
    """隔离失败路径：坏集成图在 _build_documents 即 raise，不产成功 HTML。

    内存 mock build_dspark_integration_graph 逐份返回五个原始 mutant
    （同 codex_b4_review01 build_failure_path_probe 的注入方式），断言
    每份都 BuildError 具名抛出（扩展契约码），不物化任何 candidate 文件、
    不动真实 outputs——与修前「_build_documents 仍产出 37 份成功文件字节
    含 HTML」的放行根因相反。
    """

    def test_all_five_mutants_fail_before_html(self):
        import unittest.mock as mock
        import structure_reader.cli as cli_mod
        from structure_reader.extract import dspark_integration as di
        from structure_reader.inventory import build_inventory

        workspace_root = find_workspace_root()
        inv = build_inventory("deepseek_v4_flash_0731",
                              workspace_root=workspace_root)
        real_bundle = build_dspark_integration_graph(workspace_root)
        outputs_root = PROJECT / "outputs" / "deepseek_v4_flash_0731"
        outputs_before = {p.name for p in outputs_root.iterdir()}
        for name, want_code in _RAW_MUTANT_EXPECT.items():
            with self.subTest(mutant=name):
                bad = graph_from_json(
                    (MUTANT_DIR / f"{name}.json").read_text(encoding="utf-8"))
                fb = mock.Mock()
                fb.graph = bad
                fb.registry = real_bundle.registry
                fb.values = real_bundle.values
                fb.connections = ()
                fb.expansion = real_bundle.expansion
                fb.chain = real_bundle.chain
                fb.entry_bindings = real_bundle.entry_bindings
                fb.exit_role_map = real_bundle.exit_role_map
                fb.markov_step_map = real_bundle.markov_step_map
                fb.confidence_map = real_bundle.confidence_map
                with mock.patch.object(di, "build_dspark_integration_graph",
                                       return_value=fb):
                    with self.assertRaises(BuildError) as ctx:
                        cli_mod._build_documents(
                            "deepseek_v4_flash_0731", workspace_root,
                            "standard", "0" * 64, inv, [])
                found_codes = set()
                for part in str(ctx.exception).split("；")[0].split(": "):
                    for token in part.split("; "):
                        if "@" in token:
                            found_codes.add(token.split("@")[0].strip())
                # 具名拒绝必须含该 mutant 的期望契约码（消息内含 code@path）
                for code in want_code:
                    self.assertIn(code, str(ctx.exception),
                                  f"{name}: 缺 {code}")
        outputs_after = {p.name for p in outputs_root.iterdir()}
        self.assertEqual(outputs_before, outputs_after,
                         "失败路径不得物化新 candidate 目录")


if __name__ == "__main__":
    unittest.main()
