"""DeepSeek 规则模块测试：轴字典↔config 交叉核对、层分组、守卫白名单、场景环境。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.model_registry import find_workspace_root
from structure_reader.rules.deepseek_v4_flash_0731 import (
    AXIS_DEFS,
    GUARD_DEFS,
    SCENARIOS,
    RulesError,
    derive_layer_groups,
    load_model_values,
    scenario_base_env,
)
from structure_reader.scenario_evaluator import evaluate_tri


class TestAxisDictionary(unittest.TestCase):
    def setUp(self):
        self.values = load_model_values(find_workspace_root())

    def test_load_values(self):
        self.assertEqual(self.values.dim, 4096)
        self.assertEqual(self.values.n_layers, 43)
        self.assertEqual(self.values.n_hash_layers, 3)
        self.assertEqual(self.values.compress_ratios[2], 4)
        self.assertEqual(self.values.compress_ratios[3], 128)
        self.assertEqual(len(self.values.compress_ratios), 46)

    def test_axis_defs_wellformed(self):
        symbols = [a.symbol for a in AXIS_DEFS]
        self.assertEqual(len(symbols), len(set(symbols)), "轴 symbol 重复")
        for axis in AXIS_DEFS:
            axis.validate()  # 非法即抛 RulesError

    def test_axis_config_crosscheck(self):
        # load_model_values 内部已交叉核对（不一致抛 RulesError）；能走到这里即通过
        self.assertEqual(self.values.hc_mult, 4)
        self.assertEqual(self.values.window_size, 128)
        self.assertEqual(self.values.max_seq_len, 4096)  # ModelArgs 默认


class TestLayerGroups(unittest.TestCase):
    def setUp(self):
        self.values = load_model_values(find_workspace_root())
        self.groups = derive_layer_groups(self.values)

    def test_four_groups(self):
        by_id = {g.group_id: g for g in self.groups}
        self.assertEqual(
            sorted(by_id), ["csa_hash", "csa_topk", "hca_topk", "swa_hash"]
        )
        self.assertEqual(len(by_id["swa_hash"].member_layers), 2)
        self.assertEqual(len(by_id["csa_hash"].member_layers), 1)
        self.assertEqual(len(by_id["hca_topk"].member_layers), 20)
        self.assertEqual(len(by_id["csa_topk"].member_layers), 20)

    def test_union_and_disjoint(self):
        covered = sorted(m for g in self.groups for m in g.member_layers)
        self.assertEqual(covered, list(range(43)))

    def test_specific_members(self):
        by_id = {g.group_id: g for g in self.groups}
        self.assertEqual(by_id["swa_hash"].member_layers, (0, 1))
        self.assertEqual(by_id["csa_hash"].member_layers, (2,))
        self.assertIn(4, by_id["csa_topk"].member_layers)
        self.assertNotIn(4, by_id["hca_topk"].member_layers)
        # hash 门控只出现在前 3 层
        for g in self.groups:
            if g.gate_kind == "hash":
                self.assertTrue(all(m < 3 for m in g.member_layers))

    def test_indexer_flags(self):
        by_id = {g.group_id: g for g in self.groups}
        self.assertTrue(by_id["csa_topk"].has_indexer)
        self.assertTrue(by_id["csa_hash"].has_indexer)
        self.assertFalse(by_id["hca_topk"].has_indexer)
        self.assertFalse(by_id["swa_hash"].has_indexer)

    def test_template_status_pending_extraction(self):
        # 默认（无提取器声明）：任何分组都不得标 extracted
        for g in self.groups:
            self.assertEqual(
                g.template_status, "pending_extraction",
                f"{g.group_id} 尚未提取，不得标 extracted",
            )

    def test_extracted_only_after_real_extraction(self):
        # extracted_groups 允许真实提取器在核验成功后标记
        groups = derive_layer_groups(self.values, extracted_groups={"csa_topk"})
        by_id = {g.group_id: g for g in groups}
        self.assertEqual(by_id["csa_topk"].template_status, "extracted")
        self.assertEqual(by_id["swa_hash"].template_status, "pending_extraction")

    def test_unknown_extracted_group_rejected(self):
        # 提取器不能声明从未推导出的分组
        with self.assertRaises(RulesError):
            derive_layer_groups(self.values, extracted_groups={"hca_hash"})

    def _with_ratios(self, ratios):
        return type(self.values)(
            **{**self.values.__dict__, "compress_ratios": tuple(ratios)}
        )

    def test_unknown_main_ratio_rejected(self):
        # 主干层出现未知 compress_ratio：具名 RulesError，不裸 KeyError
        ratios = list(self.values.compress_ratios[:43])
        ratios[10] = 7
        with self.assertRaises(RulesError):
            derive_layer_groups(self._with_ratios(ratios + [0, 0, 0]))

    def test_unsupported_combination_rejected(self):
        # 层 0 在 hash 门控范围（<3），ratio=128 → hca_hash 未支持组合
        ratios = list(self.values.compress_ratios[:43])
        ratios[0] = 128
        with self.assertRaises(RulesError) as ctx:
            derive_layer_groups(self._with_ratios(ratios + [0, 0, 0]))
        self.assertIn("hca_hash", str(ctx.exception))

    def test_tail_dspark_ratio_boundary(self):
        # 真实尾部 3 项为 0：合法（DSparkAttention 要求 ratio==0）
        self.assertEqual(self.values.compress_ratios[43:], (0, 0, 0))
        self.assertEqual(len(self.groups), 4)
        # 尾部 DSpark stage ratio 非 0：具名拒绝，指向源码依据
        ratios = list(self.values.compress_ratios)
        ratios[43] = 4
        with self.assertRaises(RulesError) as ctx:
            derive_layer_groups(self._with_ratios(ratios))
        self.assertIn("DSpark", str(ctx.exception))


class TestGuards(unittest.TestCase):
    def test_guard_defs_valid(self):
        for guard in GUARD_DEFS:
            guard.validate()

    def test_guard_ids_unique(self):
        ids = [g.stable_id for g in GUARD_DEFS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_guards_evaluable_with_scenario_env(self):
        values = load_model_values(find_workspace_root())
        for scenario in SCENARIOS:
            env = dict(scenario_base_env(values, scenario))
            env["r"] = 4  # CSA·TopK 代表层压缩比
            for guard in GUARD_DEFS:
                result = evaluate_tri(guard.predicate_expr, env)
                self.assertTrue(
                    result.is_known,
                    f"场景 {scenario.scenario_id} 守卫 {guard.stable_id} 未知: "
                    f"{result.unknown_reason}",
                )

    def test_compress_prefill_skip_guard(self):
        # 未触发路径守卫：T<r 时为真（compressor 返回 None）
        values = load_model_values(find_workspace_root())
        env = scenario_base_env(values, next(
            s for s in SCENARIOS if s.scenario_id == "prefill_t3"))
        env["r"] = 4
        self.assertIs(evaluate_tri("start_pos == 0 and T < r", env).value, True)


class TestScenarios(unittest.TestCase):
    def test_scenario_ids_unique(self):
        ids = [s.scenario_id for s in SCENARIOS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_expected_scenarios_present(self):
        ids = {s.scenario_id for s in SCENARIOS}
        self.assertLessEqual(
            {"prefill_t3", "prefill_t4", "prefill_t5", "prefill_t127",
             "prefill_t128", "prefill_t129",
             "decode_p2", "decode_p3", "decode_p4", "decode_p127",
             "decode_p128"}, ids
        )

    def test_decode_env_consistent(self):
        for s in SCENARIOS:
            self.assertEqual(s.env["start_pos"], s.env["P"])
            if s.scenario_id.startswith("decode"):
                self.assertEqual(s.env["T"], 1)
            else:
                self.assertEqual(s.env["P"], 0)
                self.assertEqual(s.env["start_pos"], 0)


if __name__ == "__main__":
    unittest.main()
