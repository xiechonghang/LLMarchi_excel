"""S1 01a Guard / GuardActivation 独立验证：静态条件身份、谓词原文、三态与场景往返。

范围严格限定于 `results/development_01a_guard_probe.md` 声明的两类增量：
- `Guard`：显式 `stable_id`（字段，非派生 property）、`predicate_expr` 两必填字符串
  （非空、无首尾空白）+ 有序可选 `evidence_refs`（仅 list/tuple、各项非空、字符串不拆解、
  dict 不取键）；`predicate_expr` 是**原文**——只存储、**不执行/不求值/不导入模型代码**；
  `to_dict` 恰 3 键；`from_dict` 拒绝非对象/缺字段/`evidence_refs` 非 JSON 数组；frozen、可哈希、
  重复与跨进程字节稳定。
- `GuardActivation`：`guard_id`、`scenario_id`、`active_for_scenario` 三必填 + 可选
  `unknown_reason`；`active_for_scenario` 限 `true|false|unknown` **三态严格区分**、大小写敏感；
  `unknown` **不得被当作 false** 且必须携带非空 `unknown_reason`；确定 `true`/`false` 时
  **不允许**携带 `unknown_reason`；非法取值（含 bool、大小写变体、空白）受控拒绝并回显值；
  `to_dict` 恰 4 键；frozen、可哈希、往返与字节稳定。
- 交叉约定：构造带 `guard_ref` 的 Edge，检查局部字段往返；`Guard` 静态身份不随场景变化。

**明确属图级检查、本单不通过也不声称**（见 `GuardGraphLevelGapsTest`）：
`GuardActivation.guard_id` / `Edge.guard_ref` 指向实体的**存在性**、同一场景对同一 guard 的
**重复/冲突结果**、`GuardActivation` 的场景一致性、图级 `stable_id` 唯一性、谓词求值/条件路径。

不在本范围：AliasRelation、SemanticGraph、`graph_to_json`/`graph_from_json`、图级 canonical
序列化、alias/fold/template-instance、01b 检查器、01c schema、有限求值器（开发单 02）、
源码核验（开发单 03）、真实 DeepSeek 结构提取。

纪律：数据为**合成契约样例（synthetic contract fixture）**，非 DeepSeek 结构；期望值来自
任务单与开发报告声明的接口，不取自产品输出（不复制产品输出为 golden）；只用标准库；
不安装/导入 jsonschema、torch、transformers、safetensors。
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, MISSING, fields
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import structure_reader.graph_ir as gir  # noqa: E402

GUARD_REQUIRED = ("stable_id", "predicate_expr")
GUARD_TO_DICT_KEYS = {"stable_id", "predicate_expr", "evidence_refs"}
GA_REQUIRED = ("guard_id", "scenario_id", "active_for_scenario")
GA_TO_DICT_KEYS = {"guard_id", "scenario_id", "active_for_scenario", "unknown_reason"}
THREE_STATES = ("true", "false", "unknown")
CANONICAL_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}
FORBIDDEN_RUNTIME_MODULES = ("torch", "transformers", "jsonschema", "safetensors")


def guard(**overrides):
    """合成契约样例：因果掩码条件（谓词为原文，非可执行）。"""
    spec = {
        "stable_id": "guard:causal_mask",
        "predicate_expr": "T > W（条件原文，非可执行）",
        "evidence_refs": ["fixture:guard/ev1", "fixture:guard/ev2"],
    }
    spec.update(overrides)
    return gir.Guard(**spec)


def activation(**overrides):
    """合成契约样例：prefill 场景下 guard 为 true。"""
    spec = {
        "guard_id": "guard:causal_mask",
        "scenario_id": "scenario:prefill",
        "active_for_scenario": "true",
    }
    spec.update(overrides)
    return gir.GuardActivation(**spec)


def unknown_activation(**overrides):
    """合成契约样例：上下文不足而 unknown（必须带理由）。"""
    spec = {
        "guard_id": "guard:causal_mask",
        "scenario_id": "scenario:unknown_ctx",
        "active_for_scenario": "unknown",
        "unknown_reason": "源码未给出该分支条件取值",
    }
    spec.update(overrides)
    return gir.GuardActivation(**spec)


def canonical(item):
    """仅用于稳定性比较：对 to_dict() 做确定性 JSON 编码（非产品 API）。"""
    return json.dumps(item.to_dict(), **CANONICAL_KWARGS)


class GuardTestCase(unittest.TestCase):
    """公共断言辅助。"""

    def assert_located(self, exc, *needles):
        message = str(exc)
        self.assertTrue(
            any(needle in message for needle in needles),
            f"错误信息未定位到 {needles!r}；实际信息：{message!r}",
        )

    def assertValueErrorLocated(self, callable_, *needles):
        with self.assertRaises(ValueError) as ctx:
            callable_()
        self.assert_located(ctx.exception, *needles)
        return ctx.exception


# ---------------------------------------------------------------------------
# 1. Guard 接口快照：字段、必填、显式 stable_id、frozen、序列化键集
# ---------------------------------------------------------------------------
class GuardInterfaceTest(GuardTestCase):
    def test_guard_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.Guard.__dataclass_params__.frozen)
        self.assertEqual(tuple(f.name for f in fields(gir.Guard)),
                         ("stable_id", "predicate_expr", "evidence_refs"))

    def test_required_fields_and_optional_default(self):
        by_name = {f.name: f for f in fields(gir.Guard)}
        for required in GUARD_REQUIRED:
            with self.subTest(required=required):
                self.assertIs(by_name[required].default, MISSING)
        self.assertEqual(by_name["evidence_refs"].default, ())
        minimal = gir.Guard(stable_id="guard:x", predicate_expr="a > b")
        self.assertEqual(minimal.evidence_refs, ())

    def test_stable_id_is_explicit_field_not_derived_property(self):
        static = inspect.getattr_static(gir.Guard, "stable_id", None)
        self.assertNotIsInstance(static, property)
        item = guard(stable_id="plain-guard-id")
        self.assertEqual(item.stable_id, "plain-guard-id")
        self.assertNotEqual(item.stable_id, item.predicate_expr)

    def test_to_dict_has_exactly_three_keys(self):
        payload = guard().to_dict()
        self.assertEqual(set(payload), GUARD_TO_DICT_KEYS)
        self.assertEqual(len(payload), 3)

    def test_frozen_prevents_mutation_and_is_hashable(self):
        item = guard()
        for attribute, new_value in (("stable_id", "other"),
                                     ("predicate_expr", "x > y"),
                                     ("evidence_refs", ())):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, attribute, new_value)
        self.assertEqual(hash(item), hash(guard()))
        self.assertEqual(len({item, guard()}), 1)

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.Guard(stable_id="g", predicate_expr="a", scenario_id="s")

    def test_guard_carries_no_scenario_or_activation_state(self):
        """静态身份：Guard 本身不含场景/判断结果字段（结果由 GuardActivation 承载）。"""
        field_names = {f.name for f in fields(gir.Guard)}
        for forbidden in ("scenario_id", "active_for_scenario", "unknown_reason",
                          "activation", "active"):
            with self.subTest(field=forbidden):
                self.assertNotIn(forbidden, field_names)
                self.assertFalse(hasattr(guard(), forbidden))


# ---------------------------------------------------------------------------
# 2. Guard 谓词原文：只存储、不执行、不求值
# ---------------------------------------------------------------------------
class GuardPredicateTextTest(GuardTestCase):
    def test_predicate_preserved_verbatim_and_roundtrip(self):
        item = guard()
        self.assertEqual(item.predicate_expr, "T > W（条件原文，非可执行）")
        payload = item.to_dict()
        self.assertEqual(payload["predicate_expr"], "T > W（条件原文，非可执行）")
        back = gir.Guard.from_dict(payload)
        self.assertEqual(back, item)
        self.assertEqual(back.predicate_expr, item.predicate_expr)

    def test_predicate_with_operators_and_unicode_is_not_normalized(self):
        for expr in ("(T > W) and (T != 0)", "seq_len <= max_position_embeddings",
                     "x % 2 == 0 or 中文条件", "a\t==\tb", "line1 and\nline2"):
            with self.subTest(expr=expr):
                item = guard(predicate_expr=expr)
                self.assertEqual(item.predicate_expr, expr)
                self.assertEqual(gir.Guard.from_dict(item.to_dict()).predicate_expr, expr)

    def test_multi_line_predicate_allowed_but_surrounding_whitespace_rejected(self):
        """边界观察：谓词原文允许多行；仅拒绝首尾空白，不因换行而拒绝。"""
        item = guard(predicate_expr="cond_a\ncond_b")
        self.assertEqual(item.predicate_expr, "cond_a\ncond_b")
        exc = self.assertValueErrorLocated(
            lambda: guard(predicate_expr="\ncond_a"), "predicate_expr"
        )
        self.assertIn("空白", str(exc))

    def test_predicate_is_never_evaluated(self):
        """只存储不执行：把有副作用的表达式放进谓词，构造后不得产生任何副作用。"""
        marker = Path(tempfile.gettempdir()) / f"dsh_guard_predicate_{os.getpid()}.marker"
        if marker.exists():
            marker.unlink()
        try:
            expr = f"__import__('pathlib').Path({str(marker)!r}).write_text('pwned') or True"
            item = guard(predicate_expr=expr)
            self.assertEqual(item.predicate_expr, expr)
            self.assertEqual(gir.Guard.from_dict(item.to_dict()).predicate_expr, expr)
            self.assertFalse(marker.exists(), "谓词被求值/执行了（出现副作用）")
        finally:
            if marker.exists():
                marker.unlink()

    def test_guard_exposes_no_evaluation_method(self):
        for name in ("evaluate", "eval", "apply", "is_active", "check", "truth",
                     "activate", "resolve"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(gir.Guard, name))
        self.assertFalse(callable(guard()), "Guard 实例不应可调用（无求值入口）")

    def test_importing_product_module_does_not_import_model_runtimes(self):
        """导入产品不引入 torch/transformers/jsonschema/safetensors（不执行模型代码）。"""
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "import structure_reader.graph_ir\n"
            f"mods = [m for m in {FORBIDDEN_RUNTIME_MODULES!r} if m in sys.modules]\n"
            "print(json.dumps(mods))\n"
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout), [])

    def test_guard_section_source_contains_no_dynamic_execution(self):
        """静态检查：Guard 段源码不含 eval/exec/__import__（与行为测试互补）。"""
        source = (SRC_DIR / "structure_reader" / "graph_ir.py").read_text(encoding="utf-8")
        start = source.index("class Guard:")
        end = source.index("class GuardActivation:")
        section = source[start:end]
        for token in ("eval(", "exec(", "__import__", "compile("):
            with self.subTest(token=token):
                self.assertNotIn(token, section)

    def test_required_text_fields_must_be_nonempty_strings(self):
        for field_name in GUARD_REQUIRED:
            for bad in ("", "   ", "\t\n", None, 123, True, b"x", ["a"]):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: guard(**{f: b}), field_name
                    )

    def test_required_text_fields_reject_surrounding_whitespace(self):
        for field_name in GUARD_REQUIRED:
            for bad in (" value", "value ", "\tvalue", "value\n"):
                with self.subTest(field=field_name, bad=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: guard(**{f: b}), field_name
                    )
                    self.assertIn("空白", str(exc))

    def test_internal_whitespace_allowed(self):
        item = guard(predicate_expr="a > b and c < d")
        self.assertEqual(item.predicate_expr, "a > b and c < d")


# ---------------------------------------------------------------------------
# 3. Guard 有序 evidence_refs（两道闸门）
# ---------------------------------------------------------------------------
class GuardEvidenceRefsTest(GuardTestCase):
    def test_default_empty_and_roundtrip(self):
        item = guard(evidence_refs=[])
        self.assertEqual(item.evidence_refs, ())
        self.assertEqual(item.to_dict()["evidence_refs"], [])
        self.assertEqual(gir.Guard.from_dict(item.to_dict()).evidence_refs, ())

    def test_order_preserved(self):
        refs = ["fixture:guard/a", "fixture:guard/b", "fixture:guard/c"]
        item = guard(evidence_refs=refs)
        self.assertEqual(list(item.evidence_refs), refs)
        self.assertEqual(item.to_dict()["evidence_refs"], refs)
        self.assertEqual(list(gir.Guard.from_dict(item.to_dict()).evidence_refs), refs)
        reversed_item = guard(evidence_refs=list(reversed(refs)))
        self.assertNotEqual(canonical(item), canonical(reversed_item))

    def test_items_must_be_nonempty_strings_with_index(self):
        for bad, needle in (([None], "evidence_refs[0]"), ([""], "evidence_refs[0]"),
                            (["   "], "evidence_refs[0]"), ([5], "evidence_refs[0]"),
                            ([[]], "evidence_refs[0]"),
                            (["ok", None], "evidence_refs[1]")):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: guard(evidence_refs=b), needle
                )

    def test_container_must_be_list_or_tuple(self):
        for bad in ({"fixture:a"}, {"fixture:a": 1}, (x for x in ["fixture:a"])):
            with self.subTest(bad=type(bad).__name__):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: guard(evidence_refs=b), "evidence_refs"
                )
                self.assertIn("list/tuple", str(exc))
                self.assertIn(type(bad).__name__, str(exc))

    def test_string_container_is_not_split_into_characters(self):
        exc = self.assertValueErrorLocated(
            lambda: guard(evidence_refs="fixture:guard/ev1"), "evidence_refs"
        )
        self.assertIn("list/tuple", str(exc))
        self.assertIn("str", str(exc))

    def test_dict_container_keys_are_not_taken(self):
        exc = self.assertValueErrorLocated(
            lambda: guard(evidence_refs={"fixture:guard/ev1": 1}), "evidence_refs"
        )
        self.assertIn("dict", str(exc))

    def test_list_and_tuple_accepted(self):
        self.assertEqual(guard(evidence_refs=["fixture:a"]).evidence_refs, ("fixture:a",))
        self.assertEqual(guard(evidence_refs=("fixture:a",)).evidence_refs, ("fixture:a",))

    def test_duplicates_allowed(self):
        item = guard(evidence_refs=["fixture:same", "fixture:same"])
        self.assertEqual(item.evidence_refs, ("fixture:same", "fixture:same"))
        self.assertEqual(gir.Guard.from_dict(item.to_dict()).evidence_refs,
                         ("fixture:same", "fixture:same"))

    def test_evidence_refs_frozen_to_tuple_with_defensive_copy(self):
        refs = ["fixture:guard/ev1"]
        item = guard(evidence_refs=refs)
        self.assertIsInstance(item.evidence_refs, tuple)
        refs.append("fixture:guard/ev3")
        self.assertEqual(item.evidence_refs, ("fixture:guard/ev1",))


# ---------------------------------------------------------------------------
# 4. Guard 往返、JSON 数组闸门与字节稳定
# ---------------------------------------------------------------------------
class GuardRoundTripTest(GuardTestCase):
    def test_roundtrip_equality_and_dict_equality(self):
        item = guard()
        back = gir.Guard.from_dict(item.to_dict())
        self.assertEqual(back, item)
        self.assertEqual(back.to_dict(), item.to_dict())

    def test_repeated_canonical_serialization_byte_identical(self):
        item = guard()
        first = canonical(item)
        self.assertEqual(first.encode("utf-8"), canonical(item).encode("utf-8"))
        self.assertEqual(first, canonical(gir.Guard.from_dict(item.to_dict())))

    def test_canonical_bytes_deterministic_across_processes(self):
        item = guard(predicate_expr="T > W and 中文条件")
        expected = canonical(item)
        literal = repr(json.dumps(item.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import Guard\n"
            f"data = json.loads({literal})\n"
            "g = Guard.from_dict(data)\n"
            "sys.stdout.write(json.dumps(g.to_dict(), ensure_ascii=False, sort_keys=True,"
            " separators=(',', ':')))\n"
        )
        for seed in ("0", "1", "12345"):
            with self.subTest(pythonhashseed=seed):
                proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                      text=True, env=dict(os.environ, PYTHONHASHSEED=seed),
                                      timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout, expected)

    def test_from_dict_requires_json_object(self):
        for bad in ([], "guard", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: gir.Guard.from_dict(b),
                                             "guard", "对象")

    def test_from_dict_missing_required_fields_named(self):
        for missing in GUARD_REQUIRED:
            payload = guard().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.Guard.from_dict(p), missing
                )

    def test_from_dict_evidence_refs_must_be_json_array(self):
        for bad in (None, 5, True, "ev1", "", {"fixture:a": 1}):
            payload = guard().to_dict()
            payload["evidence_refs"] = bad
            with self.subTest(bad=bad):
                exc = self.assertValueErrorLocated(
                    lambda p=payload: gir.Guard.from_dict(p), "evidence_refs"
                )
                self.assertIn("JSON 数组", str(exc))

    def test_from_dict_optional_evidence_refs_absent_default(self):
        payload = {"stable_id": "g0", "predicate_expr": "a > b"}
        item = gir.Guard.from_dict(payload)
        self.assertEqual(item.evidence_refs, ())
        self.assertEqual(item, gir.Guard(stable_id="g0", predicate_expr="a > b"))

    def test_from_dict_extra_keys_ignored(self):
        payload = guard().to_dict()
        payload["scenario_id"] = "scenario:prefill"
        payload["active_for_scenario"] = "true"
        self.assertEqual(gir.Guard.from_dict(payload), guard())

    def test_distinct_guards_do_not_collapse(self):
        a = guard(stable_id="g_a")
        b = guard(stable_id="g_b")
        c = guard(stable_id="g_a", predicate_expr="c > d")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b, c}), 3)


# ---------------------------------------------------------------------------
# 5. GuardActivation 三态：true / false / unknown 严格区分
# ---------------------------------------------------------------------------
class GuardActivationThreeStateTest(GuardTestCase):
    def test_three_states_accepted_and_roundtrip(self):
        items = [activation(active_for_scenario="true"),
                 activation(active_for_scenario="false"),
                 unknown_activation()]
        self.assertEqual({item.active_for_scenario for item in items}, set(THREE_STATES))
        for item in items:
            with self.subTest(state=item.active_for_scenario):
                self.assertEqual(gir.GuardActivation.from_dict(item.to_dict()), item)

    def test_states_are_distinct_facts_not_interchangeable(self):
        true_item = activation(active_for_scenario="true")
        false_item = activation(active_for_scenario="false")
        unknown_item = unknown_activation()
        self.assertNotEqual(true_item, false_item)
        self.assertNotEqual(false_item, unknown_item)
        self.assertNotEqual(true_item, unknown_item)
        self.assertEqual(len({canonical(true_item), canonical(false_item),
                              canonical(unknown_item)}), 3)

    def test_unknown_is_not_treated_as_false(self):
        unknown_item = unknown_activation()
        self.assertEqual(unknown_item.active_for_scenario, "unknown")
        self.assertNotEqual(unknown_item.active_for_scenario, "false")
        self.assertIsNotNone(unknown_item.unknown_reason)
        payload = unknown_item.to_dict()
        self.assertEqual(payload["active_for_scenario"], "unknown")
        self.assertNotEqual(payload["active_for_scenario"], "false")
        # false 不需要理由，unknown 必须要有：二者不是同一事实
        false_item = activation(active_for_scenario="false")
        self.assertIsNone(false_item.unknown_reason)
        self.assertNotEqual(canonical(false_item), canonical(unknown_item))

    def test_state_is_case_sensitive_and_echoes_value(self):
        for bad in ("True", "TRUE", "False", "UNKNOWN", "Unknown", "Yes", "on"):
            with self.subTest(bad=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: activation(active_for_scenario=b),
                    "active_for_scenario",
                )
                self.assertIn(repr(bad), str(exc))

    def test_bool_values_rejected_as_non_string(self):
        for bad in (True, False):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: activation(active_for_scenario=b),
                    "active_for_scenario",
                )

    def test_invalid_values_rejected_with_field_name(self):
        for bad in ("", "   ", "yes", "no", "0", "1", "null", None, 5, ["true"],
                    {"active_for_scenario": "true"}):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: activation(active_for_scenario=b),
                    "active_for_scenario",
                )

    def test_surrounding_whitespace_rejected_before_whitelist(self):
        for bad in (" true", "true ", "\ttrue", "false\n", " unknown "):
            with self.subTest(bad=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: activation(active_for_scenario=b),
                    "active_for_scenario",
                )
                self.assertIn("空白", str(exc))

    def test_error_message_lists_all_three_legal_values(self):
        exc = self.assertValueErrorLocated(
            lambda: activation(active_for_scenario="maybe"), "active_for_scenario"
        )
        for state in THREE_STATES:
            self.assertIn(state, str(exc))
        self.assertIn("maybe", str(exc))

    def test_state_whitelist_precedes_unknown_reason_rule(self):
        exc = self.assertValueErrorLocated(
            lambda: activation(active_for_scenario="bogus", unknown_reason="r"),
            "active_for_scenario",
        )
        self.assertIn("bogus", str(exc))

    def test_from_dict_path_enforces_three_state_rules(self):
        payload = activation().to_dict()
        payload["active_for_scenario"] = "UNKNOWN"
        self.assertValueErrorLocated(
            lambda: gir.GuardActivation.from_dict(payload), "active_for_scenario"
        )
        payload2 = activation().to_dict()
        payload2["active_for_scenario"] = "unknown"  # 无 reason
        self.assertValueErrorLocated(
            lambda: gir.GuardActivation.from_dict(payload2), "unknown_reason"
        )


# ---------------------------------------------------------------------------
# 6. GuardActivation 的 unknown_reason 规则
# ---------------------------------------------------------------------------
class GuardActivationUnknownReasonTest(GuardTestCase):
    def test_unknown_with_reason_accepted_and_roundtrip(self):
        item = unknown_activation()
        self.assertEqual(item.active_for_scenario, "unknown")
        self.assertEqual(item.unknown_reason, "源码未给出该分支条件取值")
        payload = item.to_dict()
        self.assertEqual(payload["unknown_reason"], "源码未给出该分支条件取值")
        self.assertEqual(gir.GuardActivation.from_dict(payload), item)

    def test_unknown_reason_not_constrained_to_ascii(self):
        item = unknown_activation(unknown_reason="源码未给出 · 待 02 求值器")
        self.assertEqual(gir.GuardActivation.from_dict(item.to_dict()).unknown_reason,
                         "源码未给出 · 待 02 求值器")

    def test_unknown_requires_nonempty_reason(self):
        for bad in (None, "", "   ", "\t", 5, ["r"], True):
            with self.subTest(reason=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: gir.GuardActivation(
                        guard_id="guard:x", scenario_id="scenario:s",
                        active_for_scenario="unknown", unknown_reason=b),
                    "unknown_reason", "unknown",
                )
                self.assertIn("unknown", str(exc))

    def test_unknown_reason_rejects_surrounding_whitespace(self):
        for bad in (" reason", "reason ", "\treason"):
            with self.subTest(reason=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: unknown_activation(unknown_reason=b),
                    "unknown_reason",
                )
                self.assertIn("空白", str(exc))

    def test_definite_true_forbids_reason(self):
        for bad in ("", "   ", "理由", 5):
            with self.subTest(reason=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: activation(active_for_scenario="true",
                                             unknown_reason=b),
                    "unknown_reason",
                )

    def test_definite_false_forbids_reason(self):
        for bad in ("", "   ", "理由", 5):
            with self.subTest(reason=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: activation(active_for_scenario="false",
                                             unknown_reason=b),
                    "unknown_reason",
                )

    def test_definite_states_default_reason_is_none(self):
        for state in ("true", "false"):
            with self.subTest(state=state):
                item = activation(active_for_scenario=state)
                self.assertIsNone(item.unknown_reason)
                self.assertIsNone(item.to_dict()["unknown_reason"])

    def test_from_dict_forbids_reason_on_definite_state(self):
        for state in ("true", "false"):
            payload = activation(active_for_scenario=state).to_dict()
            payload["unknown_reason"] = "理由"
            with self.subTest(state=state):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.GuardActivation.from_dict(p), "unknown_reason"
                )


# ---------------------------------------------------------------------------
# 7. 场景 ID 往返与 Guard 静态身份不随场景变化
# ---------------------------------------------------------------------------
class GuardActivationScenarioIdentityTest(GuardTestCase):
    def test_guard_id_and_scenario_id_preserved_verbatim_and_roundtrip(self):
        item = activation()
        payload = item.to_dict()
        self.assertEqual(payload["guard_id"], "guard:causal_mask")
        self.assertEqual(payload["scenario_id"], "scenario:prefill")
        back = gir.GuardActivation.from_dict(payload)
        self.assertEqual((back.guard_id, back.scenario_id),
                         ("guard:causal_mask", "scenario:prefill"))

    def test_activation_has_exactly_four_fields_no_stable_id(self):
        """边界观察：GuardActivation 无 stable_id；身份为 (guard_id, scenario_id) 组合。"""
        field_names = {f.name for f in fields(gir.GuardActivation)}
        self.assertEqual(field_names, GA_TO_DICT_KEYS)
        self.assertNotIn("stable_id", field_names)

    def test_guard_static_identity_unchanged_across_scenarios(self):
        """同一 Guard 在两个场景下的激活不改变 Guard 模板身份（无场景字段、哈希不变）。"""
        template = guard()
        before_hash = hash(template)
        for scenario in ("scenario:prefill", "scenario:decode", "scenario:训练"):
            result = activation(guard_id=template.stable_id, scenario_id=scenario)
            with self.subTest(scenario=scenario):
                self.assertEqual(result.guard_id, template.stable_id)
                self.assertEqual(hash(template), before_hash)
                self.assertEqual(template, guard())
                self.assertFalse(hasattr(template, "scenario_id"))

    def test_same_guard_different_scenarios_are_distinct_facts(self):
        a = activation(scenario_id="scenario:prefill")
        b = activation(scenario_id="scenario:decode")
        self.assertEqual(a.guard_id, b.guard_id)
        self.assertNotEqual(a, b)
        self.assertNotEqual(canonical(a), canonical(b))

    def test_guard_id_references_guard_stable_id_as_string(self):
        template = guard()
        result = activation(guard_id=template.stable_id)
        self.assertIsInstance(result.guard_id, str)
        self.assertEqual(result.guard_id, template.stable_id)

    def test_guard_id_and_scenario_id_required_text_rules(self):
        for field_name in ("guard_id", "scenario_id"):
            for bad in ("", "   ", None, 5, True, b"s"):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: activation(**{f: b}), field_name
                    )
            for bad in (" value", "value ", "\tvalue"):
                with self.subTest(field=field_name, whitespace=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: activation(**{f: b}), field_name
                    )
                    self.assertIn("空白", str(exc))

    def test_same_guard_scenario_different_state_are_distinct_no_dedup(self):
        """图级缺口（不算通过）：同一 (guard, scenario) 的两个不同结果本地均可构造。"""
        t = activation(active_for_scenario="true")
        f = activation(active_for_scenario="false")
        self.assertEqual((t.guard_id, t.scenario_id), (f.guard_id, f.scenario_id))
        self.assertNotEqual(t, f)


# ---------------------------------------------------------------------------
# 8. GuardActivation 往返、frozen 与字节稳定
# ---------------------------------------------------------------------------
class GuardActivationRoundTripTest(GuardTestCase):
    def test_frozen_prevents_mutation_and_is_hashable(self):
        item = unknown_activation()
        for attribute, new_value in (("guard_id", "other"), ("scenario_id", "other"),
                                     ("active_for_scenario", "false"),
                                     ("unknown_reason", None)):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, attribute, new_value)
        self.assertEqual(hash(item), hash(unknown_activation()))
        self.assertEqual(len({item, unknown_activation()}), 1)

    def test_to_dict_has_exactly_four_keys(self):
        for item in (activation(), unknown_activation()):
            with self.subTest(state=item.active_for_scenario):
                payload = item.to_dict()
                self.assertEqual(set(payload), GA_TO_DICT_KEYS)
                self.assertEqual(len(payload), 4)

    def test_roundtrip_equality_for_all_three_states(self):
        for item in (activation(active_for_scenario="true"),
                     activation(active_for_scenario="false"),
                     unknown_activation()):
            with self.subTest(state=item.active_for_scenario):
                back = gir.GuardActivation.from_dict(item.to_dict())
                self.assertEqual(back, item)
                self.assertEqual(back.to_dict(), item.to_dict())

    def test_repeated_canonical_serialization_byte_identical(self):
        item = unknown_activation()
        first = canonical(item)
        self.assertEqual(first.encode("utf-8"), canonical(item).encode("utf-8"))
        self.assertEqual(first, canonical(gir.GuardActivation.from_dict(item.to_dict())))

    def test_canonical_bytes_deterministic_across_processes(self):
        item = unknown_activation(unknown_reason="源码未给出 · 中文理由")
        expected = canonical(item)
        literal = repr(json.dumps(item.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import GuardActivation\n"
            f"data = json.loads({literal})\n"
            "a = GuardActivation.from_dict(data)\n"
            "sys.stdout.write(json.dumps(a.to_dict(), ensure_ascii=False, sort_keys=True,"
            " separators=(',', ':')))\n"
        )
        for seed in ("0", "1", "12345"):
            with self.subTest(pythonhashseed=seed):
                proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                      text=True, env=dict(os.environ, PYTHONHASHSEED=seed),
                                      timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout, expected)

    def test_from_dict_requires_json_object(self):
        for bad in ([], "activation", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: gir.GuardActivation.from_dict(b), "guard 判断", "对象"
                )

    def test_from_dict_missing_required_fields_named(self):
        for missing in GA_REQUIRED:
            payload = activation().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.GuardActivation.from_dict(p), missing
                )

    def test_from_dict_optional_unknown_reason_absent_defaults_none(self):
        payload = {"guard_id": "guard:x", "scenario_id": "scenario:s",
                   "active_for_scenario": "true"}
        item = gir.GuardActivation.from_dict(payload)
        self.assertIsNone(item.unknown_reason)

    def test_from_dict_extra_keys_ignored(self):
        payload = unknown_activation().to_dict()
        payload["semantic_graph_id"] = "graph:1"
        payload["created_by"] = "claude"
        self.assertEqual(gir.GuardActivation.from_dict(payload), unknown_activation())

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.GuardActivation(guard_id="g", scenario_id="s",
                                active_for_scenario="true", reason="r")


# ---------------------------------------------------------------------------
# 9. Edge 的 guard_ref 局部往返（不解析引用存在性）
# ---------------------------------------------------------------------------
class EdgeGuardRefTest(GuardTestCase):
    @staticmethod
    def _data_edge(**overrides):
        spec = {
            "stable_id": "edge:q_out->attn_in@layer0",
            "src_node_id": "call:q_proj@layer0",
            "src_port": "q_out",
            "dst_node_id": "call:attn@layer0",
            "dst_port": "q_in",
            "tensor_id": "tensor:q",
            "kind": "data",
        }
        spec.update(overrides)
        return gir.Edge(**spec)

    @staticmethod
    def _state_edge(**overrides):
        spec = {
            "stable_id": "edge:kv_state@layer1",
            "src_node_id": "call:attn@layer0",
            "src_port": "kv_out",
            "dst_node_id": "call:attn@layer1",
            "dst_port": "kv_in",
            "tensor_id": "tensor:kv",
            "kind": "state",
            "state_transition_ref": "trans:kv_cache@step1",
        }
        spec.update(overrides)
        return gir.Edge(**spec)

    def test_data_edge_with_guard_ref_roundtrip(self):
        item = self._data_edge(guard_ref="guard:causal_mask")
        self.assertEqual(item.guard_ref, "guard:causal_mask")
        back = gir.Edge.from_dict(item.to_dict())
        self.assertEqual(back, item)
        self.assertEqual(back.guard_ref, "guard:causal_mask")

    def test_state_edge_with_guard_ref_allowed_and_roundtrip(self):
        item = self._state_edge(guard_ref="guard:cache_valid")
        self.assertEqual(item.kind, "state")
        self.assertEqual(item.guard_ref, "guard:cache_valid")
        self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)

    def test_guard_ref_absent_key_means_none(self):
        payload = self._data_edge().to_dict()
        del payload["guard_ref"]
        self.assertIsNone(gir.Edge.from_dict(payload).guard_ref)

    def test_guard_ref_optional_default_none(self):
        self.assertIsNone(self._data_edge().guard_ref)
        self.assertIsNone(self._data_edge().to_dict()["guard_ref"])

    def test_guard_ref_rejects_empty_whitespace_and_non_string(self):
        for bad in ("", "   ", "\t", 5, True, ["guard:x"]):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: self._data_edge(guard_ref=b), "guard_ref"
                )
        # 边界观察：guard_ref 复用 Edge 通用"非空字符串或 None"规则，首尾空白被拒但
        # 错误信息用该通用措辞（不含"空白"字样）；此处只断言字段级定位。
        for bad in (" guard:x", "guard:x "):
            with self.subTest(bad=bad):
                exc = self.assertValueErrorLocated(
                    lambda b=bad: self._data_edge(guard_ref=b), "guard_ref"
                )
                self.assertIn("guard_ref", str(exc))

    def test_guard_ref_is_opaque_copy_of_guard_stable_id(self):
        template = guard()
        item = self._data_edge(guard_ref=template.stable_id)
        self.assertIsInstance(item.guard_ref, str)
        self.assertEqual(item.guard_ref, template.stable_id)

    def test_reference_to_nonexistent_guard_is_locally_accepted(self):
        """图级缺口（不算通过）：guard_ref 指向存在的 Guard 本地不检查。"""
        item = self._data_edge(guard_ref="guard:does_not_exist")
        self.assertEqual(item.guard_ref, "guard:does_not_exist")
        self.assertEqual(gir.Edge.from_dict(item.to_dict()), item)


# ---------------------------------------------------------------------------
# 10. 图级缺口（本单不通过、不声称）
# ---------------------------------------------------------------------------
class GuardGraphLevelGapsTest(GuardTestCase):
    def test_guard_stable_id_uniqueness_not_enforced(self):
        first = guard(stable_id="g_dup")
        second = guard(stable_id="g_dup")
        self.assertEqual(first, second)
        self.assertEqual(len([first, second]), 2)

    def test_activation_reference_existence_not_checked(self):
        item = activation(guard_id="guard:not_in_any_graph",
                          scenario_id="scenario:not_in_any_graph")
        self.assertEqual(item.guard_id, "guard:not_in_any_graph")
        self.assertEqual(gir.GuardActivation.from_dict(item.to_dict()), item)

    def test_duplicate_activation_for_same_guard_scenario_allowed(self):
        first = activation()
        second = activation()
        self.assertEqual(first, second)
        self.assertEqual(len([first, second]), 2)

    def test_conflicting_activations_for_same_guard_scenario_allowed(self):
        """图级缺口：同一 (guard, scenario) 同时 true/false 的冲突本地不检测。"""
        t = activation(active_for_scenario="true")
        f = activation(active_for_scenario="false")
        u = unknown_activation(scenario_id="scenario:prefill")
        self.assertEqual(t.scenario_id, f.scenario_id)
        self.assertEqual({t.active_for_scenario, f.active_for_scenario, u.active_for_scenario},
                         set(THREE_STATES))

    def test_no_scenario_applicability_inference(self):
        """图级缺口：guard 在某场景是否"适用"不由本类推导，只保存显式结果。"""
        item = unknown_activation()
        self.assertEqual(item.active_for_scenario, "unknown")
        payload = item.to_dict()
        self.assertNotIn("applicable", payload)
        self.assertNotIn("resolved", payload)

    def test_serialized_guard_contains_no_evaluated_boolean(self):
        """只存储不执行：序列化里只有谓词原文与三态标签，没有求值后的布尔字段。"""
        payload = guard().to_dict()
        self.assertNotIn("value", payload)
        self.assertNotIn("evaluated", payload)
        self.assertNotIn("result", payload)
        self.assertIsInstance(payload["predicate_expr"], str)
        ga_payload = unknown_activation().to_dict()
        self.assertEqual(ga_payload["active_for_scenario"], "unknown")
        self.assertIsInstance(ga_payload["active_for_scenario"], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
