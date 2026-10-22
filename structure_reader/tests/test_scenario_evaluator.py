"""scenario_evaluator 白名单求值器测试：合法求值、具名拒绝、三态行为与求值边界。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structure_reader.scenario_evaluator import (
    ExpressionError,
    evaluate_bool,
    evaluate_int,
    evaluate_tri,
    check_expression,
)


class TestEvaluateInt(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(evaluate_int("1 + 2 * 3", {}), 7)
        self.assertEqual(evaluate_int("(1 + 2) * 3", {}), 9)
        self.assertEqual(evaluate_int("10 // 3", {}), 3)
        self.assertEqual(evaluate_int("10 % 3", {}), 1)
        self.assertEqual(evaluate_int("-5 + 2", {}), -3)
        self.assertEqual(evaluate_int("B * T", {"B": 2, "T": 128}), 256)

    def test_min_max(self):
        self.assertEqual(evaluate_int("min(3, 5)", {}), 3)
        self.assertEqual(evaluate_int("max(3, 5)", {}), 5)
        self.assertEqual(evaluate_int("min(T, W)", {"T": 300, "W": 128}), 128)
        self.assertEqual(evaluate_int("min(Ki, (P + T) // r)", {"Ki": 512, "P": 3, "T": 4, "r": 4}), 1)

    def test_min_max_single_arg(self):
        # 单参数返回该参数本身（不是内置 min/max 的可迭代重载）
        self.assertEqual(evaluate_int("min(1)", {}), 1)
        self.assertEqual(evaluate_int("max(1)", {}), 1)
        self.assertEqual(evaluate_int("min(-7)", {}), -7)
        self.assertEqual(evaluate_int("min(T)", {"T": 42}), 42)
        self.assertEqual(evaluate_int("max(min(3), 9)", {}), 9)
        self.assertEqual(evaluate_int("min(max(5), min(2))", {}), 2)

    def test_variable_missing_raises(self):
        with self.assertRaises(ExpressionError):
            evaluate_int("T + 1", {})

    def test_division_by_zero_raises(self):
        with self.assertRaises(ExpressionError):
            evaluate_int("10 // X", {"X": 0})

    def test_non_int_result_raises(self):
        with self.assertRaises(ExpressionError):
            evaluate_int("1 < 2", {})

    def test_env_none_equivalent_to_empty(self):
        self.assertEqual(evaluate_int("1 + 2"), 3)
        self.assertEqual(evaluate_int("1 + 2", None), 3)
        with self.assertRaises(ExpressionError):
            evaluate_int("T + 1")

    def test_env_bool_value_rejected(self):
        # bool 与 int 严格区分：True 是布尔不是 1
        with self.assertRaises(ExpressionError):
            evaluate_int("x + 1", {"x": True})

    def test_env_non_int_value_rejected(self):
        for bad in ("5", 5.0, None, [5], {"v": 5}):
            with self.assertRaises(ExpressionError, msg=repr(bad)):
                evaluate_int("x + 1", {"x": bad})

    def test_env_non_mapping_rejected(self):
        with self.assertRaises(ExpressionError):
            evaluate_int("x + 1", "x=5")
        with self.assertRaises(ExpressionError):
            evaluate_int("x + 1", [("x", 5)])

    def test_env_non_identifier_key_rejected(self):
        with self.assertRaises(ExpressionError):
            evaluate_int("x + 1", {"not-a-name": 5})

    def test_env_value_size_limit(self):
        with self.assertRaises(ExpressionError):
            evaluate_int("x", {"x": 1 << 63})
        self.assertEqual(evaluate_int("x", {"x": (1 << 63) - 1}), (1 << 63) - 1)

    def test_int_literal_size_limit(self):
        with self.assertRaises(ExpressionError):
            check_expression("9223372036854775808")
        self.assertEqual(evaluate_int("9223372036854775807", {}), (1 << 63) - 1)
        self.assertEqual(evaluate_int("-9223372036854775807", {}), -((1 << 63) - 1))

    def test_intermediate_overflow_rejected(self):
        # 中间结果同样受限：2^32 * 2^32 = 2^64 超限具名拒绝，不静默用大整数
        with self.assertRaises(ExpressionError):
            evaluate_int("x * x", {"x": 4294967296})
        self.assertEqual(
            evaluate_int("x * x", {"x": 2147483647}), 4611686014132420609
        )

    def test_min_max_no_args_rejected(self):
        with self.assertRaises(ExpressionError):
            check_expression("min()")
        with self.assertRaises(ExpressionError):
            check_expression("max()")


class TestEvaluateBool(unittest.TestCase):
    def test_comparisons(self):
        self.assertTrue(evaluate_bool("start_pos == 0", {"start_pos": 0}))
        self.assertFalse(evaluate_bool("start_pos == 0", {"start_pos": 3}))
        self.assertTrue(evaluate_bool("(start_pos + 1) % r == 0", {"start_pos": 127, "r": 4}))
        self.assertFalse(evaluate_bool("(start_pos + 1) % r == 0", {"start_pos": 126, "r": 4}))
        self.assertTrue(evaluate_bool("T > W", {"T": 129, "W": 128}))

    def test_and_or(self):
        self.assertTrue(evaluate_bool("start_pos == 0 and T >= r", {"start_pos": 0, "T": 5, "r": 4}))
        self.assertFalse(evaluate_bool("start_pos == 0 and T >= r", {"start_pos": 0, "T": 3, "r": 4}))
        self.assertTrue(evaluate_bool("start_pos > 0 or T >= r", {"start_pos": 1, "T": 3, "r": 4}))

    def test_non_bool_result_raises(self):
        with self.assertRaises(ExpressionError):
            evaluate_bool("1 + 1", {})

    def test_and_or_non_bool_operand_rejected(self):
        # and/or 操作数必须是布尔结果：整数/布尔常量不是合法操作数
        with self.assertRaises(ExpressionError):
            evaluate_bool("1 and T >= r", {"T": 5, "r": 4})
        with self.assertRaises(ExpressionError):
            evaluate_bool("T >= r or 0", {"T": 5, "r": 4})

    def test_bool_env_value_rejected(self):
        with self.assertRaises(ExpressionError):
            evaluate_bool("x", {"x": True})
        with self.assertRaises(ExpressionError):
            evaluate_bool("x == 1", {"x": True})

    def test_and_or_short_circuit(self):
        # 短路：已确定的 False/True 不被后续操作数中的除零掩盖
        self.assertIs(
            evaluate_bool("T == 0 and (10 // X) == 0", {"T": 1, "X": 0}), False
        )
        self.assertIs(
            evaluate_bool("T == 1 or (10 // X) == 0", {"T": 1, "X": 0}), True
        )
        # 不短路时除零照常具名报错，不静默
        with self.assertRaises(ExpressionError):
            evaluate_bool("T == 0 and (10 // X) == 0", {"T": 0, "X": 0})


class TestEvaluateTri(unittest.TestCase):
    def test_determined(self):
        result = evaluate_tri("T >= r", {"T": 5, "r": 4})
        self.assertTrue(result.is_known)
        self.assertIs(result.value, True)

    def test_missing_var_unknown_not_false(self):
        result = evaluate_tri("C >= Ki", {"Ki": 512})
        self.assertFalse(result.is_known)
        self.assertEqual(result.state, "unknown")
        self.assertIn("C", result.unknown_reason)

    def test_division_unknown(self):
        result = evaluate_tri("10 // X", {"X": 0})
        self.assertFalse(result.is_known)
        self.assertIn("除零", result.unknown_reason)

    def test_bad_syntax_unknown(self):
        # env 可省略（None = 空环境）；非法表达式 → unknown（带原因）
        for result in (evaluate_tri("import os"), evaluate_tri("import os", None)):
            self.assertFalse(result.is_known)
            self.assertEqual(result.state, "unknown")
            self.assertIn("表达式非法", result.unknown_reason)

    def test_env_contract_error_raises_not_unknown(self):
        # 环境类型错误是调用方程序错误：抛 ExpressionError，不得掩盖成 unknown
        with self.assertRaises(ExpressionError):
            evaluate_tri("T >= 1", "not a mapping")
        with self.assertRaises(ExpressionError):
            evaluate_tri("T >= 1", {"T": "5"})
        with self.assertRaises(ExpressionError):
            evaluate_tri("T >= 1", {"T": True})

    def test_short_circuit_keeps_determined(self):
        # and 左支已确定为 False：整式确定为 False，右支除零不掩盖
        result = evaluate_tri("T == 0 and (10 // X) == 0", {"T": 1, "X": 0})
        self.assertTrue(result.is_known)
        self.assertIs(result.value, False)

    def test_missing_var_not_swallowed_by_short_circuit(self):
        # 预检保守：即使短路本可跳过缺失变量，也报 unknown（数据不可知优先显式化）
        result = evaluate_tri("T == 0 and (10 // X) == 0", {"T": 1})
        self.assertFalse(result.is_known)
        self.assertIn("X", result.unknown_reason)

    def test_min_max_single_arg_in_tri(self):
        # 三态路径同样不能让单参数 min/max 触发裸 TypeError
        self.assertTrue(evaluate_tri("min(1)").is_known)
        result = evaluate_tri("max(1)", None)
        self.assertTrue(result.is_known)
        self.assertEqual(result.value, 1)
        result2 = evaluate_tri("min(min(T))", {"T": 9})
        self.assertTrue(result2.is_known)
        self.assertEqual(result2.value, 9)

    def test_min_max_in_tri_env_free(self):
        # min/max 函数名不应被缺失变量预检当作变量；常量参数无缺失即确定
        result = evaluate_tri("min(T, 4)", {})
        self.assertFalse(result.is_known)
        self.assertIn("T", result.unknown_reason)
        result2 = evaluate_tri("min(4, 2)", {})
        self.assertTrue(result2.is_known)
        self.assertEqual(result2.value, 2)


class TestWhitelistRejections(unittest.TestCase):
    """白名单外语法必须具名拒绝。"""

    def test_reject_attribute_access(self):
        with self.assertRaises(ExpressionError):
            check_expression("x.__class__")

    def test_reject_call_nonwhitelisted(self):
        with self.assertRaises(ExpressionError):
            check_expression("__import__('os').system('ls')")

    def test_reject_lambda(self):
        with self.assertRaises(ExpressionError):
            check_expression("(lambda: 1)()")

    def test_reject_string_constant(self):
        with self.assertRaises(ExpressionError):
            check_expression("'abc' + 'def'")

    def test_reject_float_constant(self):
        with self.assertRaises(ExpressionError):
            check_expression("1.5 + 2")

    def test_reject_bool_constant(self):
        with self.assertRaises(ExpressionError):
            check_expression("True and False")

    def test_reject_subscript(self):
        with self.assertRaises(ExpressionError):
            check_expression("xs[0]", )

    def test_reject_chained_comparison(self):
        with self.assertRaises(ExpressionError):
            check_expression("1 < x < 10")

    def test_reject_power_op(self):
        with self.assertRaises(ExpressionError):
            check_expression("2 ** 10")

    def test_reject_newline(self):
        with self.assertRaises(ExpressionError):
            check_expression("1 +\n2")

    def test_reject_empty(self):
        with self.assertRaises(ExpressionError):
            check_expression("   ")

    def test_reject_excess_depth(self):
        # 真实深 AST：嵌套调用与左结合运算链都形成真实嵌套深度，必须拒绝
        deep_calls = "min(" * 40 + "1" + ", 1)" * 40
        with self.assertRaises(ExpressionError):
            check_expression(deep_calls)
        deep_chain = "1" + " + 1" * 200
        with self.assertRaises(ExpressionError):
            check_expression(deep_chain)

    def test_parens_are_not_ast_depth(self):
        # 括号是语法层分组，不构成 AST 节点/深度；30 层括号包住小表达式
        # 不触碰任何上限（深度限制按真实 AST 嵌套计，不按括号数）
        self.assertEqual(evaluate_int("(" * 30 + "1" + ")" * 30, {}), 1)

    def test_reject_huge_int_literal(self):
        with self.assertRaises(ExpressionError):
            check_expression("9223372036854775808")

    def test_eval_import_is_rejected_everywhere(self):
        # 非白名单内容在任何位置都拒绝：调用、操作数、嵌套
        for expr in ("__import__('os')", "1 + __import__('os')", "min(__import__('os'), 1)"):
            with self.assertRaises(ExpressionError, msg=expr):
                check_expression(expr)


if __name__ == "__main__":
    unittest.main()
