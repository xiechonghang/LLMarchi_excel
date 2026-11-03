"""T11 回归：四模型 CUBE/VEC 规则、QSA 逐 query 和、scan/conv 几何与条件语义。

覆盖：
- 每个模型规则模块的每个算子规则都在白名单语法内、引用的维度项非空；
- QSA ΣNq/ΣBq/Σtail 闭式与**逐查询枚举**一致（13 组边界，含 512 块上界前后）；
- KDA/GDN scan 的逐 token 几何（两个矩阵-向量 + rank-1 更新）、depthwise conv 几何；
- HiDream 条件：先判 Kref 再依赖分支尺寸（公式顺序）、Kref=0 的层组/串行条件守卫；
- MoE N_hit 相容守卫（≤min(E,B·S·K)、非空 ≥K）；
- 表达式节点上限仍拒绝超大表达式（T11 将上限提到 160 只为 QSA 精确和）。

运行：cd structure_reader && PYTHONPATH=src python3 tests/test_perf_engine_models.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent
SRC_DIR = REPO_ROOT / "structure_reader" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from structure_reader.export import (  # noqa: E402
    glm53_perf_rules,
    glm53flash_perf_rules,
    hidream_o1_perf_rules,
    perf_engine,
    qwen38flash_perf_rules,
)
from structure_reader.scenario_evaluator import (  # noqa: E402
    ExpressionError,
    check_expression,
    evaluate_bool,
    evaluate_int,
    expression_names,
)

MODULES = (glm53_perf_rules, glm53flash_perf_rules,
           qwen38flash_perf_rules, hidream_o1_perf_rules)


def _all_rules(module):
    for group_id, rules in module.block_rule_groups().items():
        for op_id, rule in rules.items():
            yield group_id, op_id, rule
    for op_id, rule in module.network_rules().items():
        yield "net-periphery", op_id, rule


def _brute_qsa(P, S, d=4, c=512):
    sums = {"Nq": 0, "Bq": 0, "tail": 0}
    for j in range(S):
        v = P + j + 1
        nq = v // d
        sums["Nq"] += nq
        sums["Bq"] += min(c, nq)
        sums["tail"] += v % d
    return sums


class RuleSyntaxTest(unittest.TestCase):
    def test_every_rule_is_whitelisted_and_has_terms(self):
        count = 0
        for module in MODULES:
            for group_id, op_id, rule in _all_rules(module):
                count += 1
                self.assertIn(rule["class"], ("cube", "vec", "container"),
                              (module.__name__, group_id, op_id))
                if rule["class"] == "container":
                    self.assertTrue(rule["reason"])
                    continue
                self.assertTrue(rule["terms"], (module.__name__, op_id))
                for term in rule["terms"]:
                    keys = ("b", "m", "k", "n") if rule["class"] == "cube" else ("m", "n")
                    for key in keys:
                        check_expression(term[key])
                    if "label" in term:
                        self.assertTrue(term["label"])
                if isinstance(rule.get("repeat"), str):
                    check_expression(rule["repeat"])
                for guard in rule.get("guards", []):
                    check_expression(guard)
                conditional = rule.get("conditional") or {}
                if conditional:
                    check_expression(conditional["when"])
        self.assertGreater(count, 250)


class QsaSumsTest(unittest.TestCase):
    CASES = [(0, 1), (0, 3), (0, 4), (0, 8), (1, 7), (2046, 8), (2047, 8),
             (2048, 8), (2051, 5), (4092, 4), (4096, 1), (4096, 4096),
             (0, 4096)]

    def test_closed_form_matches_per_query_enumeration(self):
        sums = qwen38flash_perf_rules.qsa_sum_expressions()
        for P, S in self.CASES:
            env = {"kvcachelen": P, "seqlen": S, "Kc": 4, "Btop": 512}
            expected = _brute_qsa(P, S)
            self.assertEqual(evaluate_int(sums["sum_Nq"], env), expected["Nq"],
                             (P, S, "sum_Nq"))
            self.assertEqual(evaluate_int(sums["sum_Bq"], env), expected["Bq"],
                             (P, S, "sum_Bq"))
            self.assertEqual(evaluate_int(sums["sum_tail"], env), expected["tail"],
                             (P, S, "sum_tail"))

    def test_per_query_terms_include_batch(self):
        rules = qwen38flash_perf_rules.block_rules("qsa-moe")
        for op_id in ("g_qsa_ix_pool", "g_qsa_ix_pool_norm", "g_qsa_ix_block_rope",
                      "g_qsa_ix_block_topk", "g_qsa_ix_select_blocks",
                      "g_qsa_ix_tail", "g_qsa_ix_block_view"):
            rule = rules[op_id]
            self.assertTrue(any(term["m"].startswith("B*") for term in rule["terms"]),
                            (op_id, rule["terms"]))
        concat = rules["g_qsa_ix_concat"]
        self.assertTrue(all(term["m"].startswith("B*") for term in concat["terms"]))
        # 主 QK/AV 仍是完整 Lkv，不用 Ksel 冒充收缩轴
        self.assertEqual(rules["g_qsa_qk"]["terms"][0]["n"], "Lkv")
        self.assertEqual(rules["g_qsa_av"]["terms"][0]["k"], "Lkv")

    def test_each_query_rows_for_indexer_scores_and_gate(self):
        qsa = qwen38flash_perf_rules.block_rules("qsa-moe")
        self.assertEqual(qsa["g_qsa_ix_scores"]["terms"][0]["m"], "1")
        self.assertEqual(qsa["g_qsa_ix_scores"]["terms"][0]["b"], "B*Iqh")
        ple = qwen38flash_perf_rules.block_rules("gdn-ple-moe")["g_ple_gate"]
        self.assertEqual(ple["terms"][0]["b"], "B*seqlen*Hc")
        self.assertEqual(ple["terms"][0]["m"], "1")


class ScanGeometryTest(unittest.TestCase):
    def _scan_work(self, rule, env):
        total = 0
        for term in rule["terms"]:
            product = 1
            for key in ("b", "m", "k", "n"):
                names = expression_names(term[key])
                product *= evaluate_int(term[key], {name: env[name] for name in names})
            total += product
        return total

    def test_kda_scan_three_matrix_terms(self):
        rule = glm53flash_perf_rules.block_rules("kda-dense")["g_kda_scan"]
        self.assertEqual(len(rule["terms"]), 3)
        self.assertTrue(all(term["b"] == "B*NhL*seqlen" for term in rule["terms"]))
        self.assertEqual([term["m"] for term in rule["terms"]], ["1", "1", "DhL"])
        self.assertEqual([term["k"] for term in rule["terms"]], ["DhL", "DhL", "1"])
        self.assertEqual([term["n"] for term in rule["terms"]], ["DhL", "DhL", "DhL"])
        work = self._scan_work(rule, {"B": 1, "NhL": 64, "seqlen": 8, "DhL": 128})
        self.assertEqual(work, 3 * 1 * 64 * 8 * 128 * 128)

    def test_gdn_scan_three_matrix_terms(self):
        rule = qwen38flash_perf_rules.block_rules("gdn-moe")["g_gdn_scan"]
        self.assertEqual(len(rule["terms"]), 3)
        self.assertTrue(all(term["b"] == "B*NvH*seqlen" for term in rule["terms"]))
        self.assertEqual([term["m"] for term in rule["terms"]], ["1", "1", "Dk"])
        work = self._scan_work(rule, {"B": 1, "NvH": 48, "seqlen": 8, "Dk": 128,
                                      "DvH": 128})
        self.assertEqual(work, 3 * 1 * 48 * 8 * 128 * 128)

    def test_depthwise_conv_geometry(self):
        kda = glm53flash_perf_rules.block_rules("kda-dense")["g_kda_conv1d"]
        term = kda["terms"][0]
        self.assertEqual(term["b"], "B*3*NhL*DhL")
        self.assertEqual(term["k"], "Kconv")
        self.assertEqual(term["n"], "1")
        gdn = qwen38flash_perf_rules.block_rules("gdn-moe")["g_gdn_conv"]
        self.assertEqual(gdn["terms"][0]["b"], "B*Dqkv")
        ple = qwen38flash_perf_rules.block_rules("gdn-ple-moe")["g_ple_conv"]
        self.assertEqual(ple["terms"][0]["k"], "Kple")


class HidreamConditionTest(unittest.TestCase):
    def test_branch_rows_are_conditional_and_guard_consistency(self):
        vit = hidream_o1_perf_rules.block_rules("vit-tower")
        for op_id, rule in vit.items():
            if op_id == "v_deepstack_tap":
                continue
            self.assertEqual(rule["conditional"]["when"], "Kref >= 1", op_id)
            text = " ".join(rule["guards"])
            self.assertIn("S_i", text)
            self.assertIn("Kref*S_i == Svis", text)
        tap = vit["v_deepstack_tap"]
        self.assertEqual(tap["charge"]["charged_at"],
                         hidream_o1_perf_rules.DEEPSTACK_OP_ID)
        decoder = hidream_o1_perf_rules.block_rules("uit-decoder")
        self.assertEqual(decoder["d_deepstack_add"]["charge"]["mode"], "excluded")
        network = hidream_o1_perf_rules.network_rules()
        deep = network[hidream_o1_perf_rules.DEEPSTACK_OP_ID]
        self.assertEqual(len(deep["terms"]), 2)
        self.assertEqual(deep["conditional"]["when"], "Kref >= 1")

    def test_serial_and_group_condition_guard_consistency(self):
        order = hidream_o1_perf_rules.serial_order()
        vit = next(entry for entry in order if entry.get("group") == "vit-tower")
        condition = vit["condition"]
        self.assertEqual(condition["when"], "Kref >= 1")
        self.assertTrue(any("Svis" in guard for guard in condition["guards"]))
        self.assertTrue(evaluate_bool(condition["guards"][0],
                                      {"Kref": 0, "Svis": 0}))
        self.assertFalse(evaluate_bool(condition["guards"][0],
                                       {"Kref": 0, "Svis": 64}))
        self.assertTrue(evaluate_bool(condition["guards"][0],
                                      {"Kref": 1, "Svis": 1024}))

    def test_net_vmerge_norm_is_before_view(self):
        rule = hidream_o1_perf_rules.network_rules()["net_vmerge_norm"]
        self.assertEqual(rule["terms"][0]["m"], "Svis")
        self.assertEqual(rule["terms"][0]["n"], "Hv")


class MoeGuardTest(unittest.TestCase):
    def test_nhit_bounds(self):
        for module, group in ((glm53_perf_rules, "shared-moe"),
                              (glm53flash_perf_rules, "dsa-moe"),
                              (qwen38flash_perf_rules, "gdn-moe")):
            rule = module.block_rules(group)["g_moe_expert_gather_indices"]
            guards = rule["guards"]
            env = {"B": 1, "seqlen": 8, "E": 256, "K": 8, "N_hit": 8}
            base = {"B": 1, "seqlen": 8,
                    "E": 256 if module is not qwen38flash_perf_rules else 512,
                    "K": 8 if module is not qwen38flash_perf_rules else 10}
            self.assertTrue(all(evaluate_bool(guard, {**base, "N_hit": base["K"]})
                                for guard in guards))
            self.assertFalse(evaluate_bool(guards[0], {**base, "N_hit": 10 ** 6}))
            self.assertFalse(evaluate_bool(guards[1], {**base, "N_hit": 0}))
            self.assertEqual(rule["terms"][0]["m"], "N_hit*K")
            self.assertIn("各层统一命中数", rule["note"])


class ExpressionLimitTest(unittest.TestCase):
    def test_qsa_sum_fits_but_absurd_expression_rejected(self):
        sums = qwen38flash_perf_rules.qsa_sum_expressions()
        for expr in sums.values():
            check_expression(expr)          # 约 100 节点，上限 160 内
        with self.assertRaises(ExpressionError):
            check_expression("1" + "+1" * 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
