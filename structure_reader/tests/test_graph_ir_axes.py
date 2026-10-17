"""S1 具名轴探针（NamedAxis）独立验证：正反例与 canonical JSON 往返。

范围严格限定于 `results/development_01a_axis_probe.md` 声明的探针增量：
- 每条轴的必填元数据（symbol / name / 中文 meaning / origin / evidence_ref）；
- 合法确定值 0 与“未知”严格区分（未知不得以 0 代替）；value 与 expression 互斥；
  unknown_reason 的必填/禁用规则；
- 显式稳定 ID（axis:{symbol}）一致性与重复轴拒绝；
- 轴顺序语义（axes 数组顺序即轴顺序，canonical 化不得重排）；
- canonical JSON 往返与字节稳定；
- 无重依赖（本体仅标准库）且不执行任意表达式。

**不在本测试范围**（不在此声称已证）：节点/边/端口/张量/guard 三态/state 前后版本/
alias/fold/template-instance 表示、JSON Schema 文件与图检查器、有限表达式求值器、
源码语义核验、真实 DeepSeek 结构提取、阅读页/CSV/Skill。

纪律：
- 全部轴数据为**合成契约样例（synthetic contract fixture）**，origin/evidence_ref 显式
  标注 synthetic，避免被误读为真实 DeepSeek 结构或源码证据；
- 期望值来自派工文档与开发报告声明的接口，不取自产品生成的输出（不复制产品输出为 golden）；
- 只使用标准库；不安装/导入 jsonschema、torch、transformers、safetensors。
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
PRODUCT_FILE = SRC_DIR / "structure_reader" / "graph_ir.py"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import structure_reader.graph_ir as gir  # noqa: E402

# 合成契约样例的显式标注：既非 DeepSeek 结构，也非源码证据。
SYNTHETIC_ORIGIN = "synthetic_contract_fixture(非DeepSeek结构)"
SYNTHETIC_EVIDENCE = "fixture:synthetic_contract/axis/hidden_size"
SCHEMA_VERSION = "semantic-graph/v1"
KIND_NAMED_AXES = "named_axes"

BLOCKED_HEAVY_MODULES = (
    "jsonschema",
    "torch",
    "transformers",
    "safetensors",
    "numpy",
    "scipy",
    "pandas",
)


def make_axis(**overrides):
    """构造一条合法的已知轴（默认 hidden_size=4096）。"""
    spec = {
        "symbol": "hidden_size",
        "name": "hidden_size",
        "meaning": "隐藏维度大小（合成契约样例，非 DeepSeek 结构）",
        "origin": SYNTHETIC_ORIGIN,
        "evidence_ref": SYNTHETIC_EVIDENCE,
        "value": 4096,
    }
    spec.update(overrides)
    return gir.NamedAxis(**spec)


def make_unknown(reason="该轴数值未在来源中给出（合成契约样例）", **overrides):
    """构造一条合法未知轴：value/expression 皆空 + 非空原因。"""
    spec = {"value": None, "expression": None, "unknown_reason": reason}
    spec.update(overrides)
    return make_axis(**spec)


def make_expression(expr="2 * hidden_size", **overrides):
    """构造一条合法“有限表达式”已知轴。"""
    spec = {"value": None, "expression": expr, "unknown_reason": None}
    spec.update(overrides)
    return make_axis(**spec)


def envelope(axes, schema=SCHEMA_VERSION, kind=KIND_NAMED_AXES):
    return json.dumps({"schema": schema, "kind": kind, "axes": axes}, ensure_ascii=False)


class AxisTestCase(unittest.TestCase):
    """公共断言辅助。"""

    def assert_located(self, exc, *needles):
        """受控破坏必须得到具名/可定位的错误信息，而不是笼统失败。"""
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
# 1. 接口快照：核对开发报告声明的公开接口（只读事实，不推断未来 API）
# ---------------------------------------------------------------------------
class InterfaceSnapshotTest(AxisTestCase):
    def test_schema_constant_is_pinned_version(self):
        self.assertEqual(gir.SCHEMA, SCHEMA_VERSION)

    def test_public_entrypoints_exist(self):
        self.assertTrue(callable(gir.NamedAxis))
        self.assertTrue(callable(gir.axes_to_json))
        self.assertTrue(callable(gir.axes_from_json))

    def test_named_axis_field_set_matches_declaration(self):
        names = {f.name for f in fields(gir.NamedAxis)}
        self.assertEqual(
            names,
            {
                "symbol",
                "name",
                "meaning",
                "origin",
                "evidence_ref",
                "value",
                "expression",
                "unknown_reason",
            },
        )

    def test_named_axis_is_frozen_and_hashable(self):
        axis = make_axis()
        with self.assertRaises(FrozenInstanceError):
            axis.symbol = "别的"  # type: ignore[misc]
        self.assertEqual(len({axis}), 1)  # 可哈希，未被可变状态击穿

    def test_stable_id_derivation_axis_prefix(self):
        self.assertEqual(make_axis(symbol="hidden_size").stable_id, "axis:hidden_size")
        self.assertEqual(make_unknown(symbol="kv_lora_rank").stable_id, "axis:kv_lora_rank")


# ---------------------------------------------------------------------------
# 2. 中文含义与来源引用
# ---------------------------------------------------------------------------
class ChineseMeaningAndOriginTest(AxisTestCase):
    def test_meaning_preserved_verbatim_and_not_ascii_escaped(self):
        axis = make_axis()
        text = gir.axes_to_json([axis])
        self.assertIn("隐藏维度大小", text)
        self.assertNotIn("\\u", text)  # ensure_ascii=False：中文原样落盘
        back = gir.axes_from_json(text)[0]
        self.assertEqual(back.meaning, axis.meaning)

    def test_all_required_metadata_roundtrip(self):
        axis = make_axis(
            symbol="n_hc_src",
            name="N_hc_src",
            meaning="源侧 HC 流数量（合成契约样例）",
            origin=SYNTHETIC_ORIGIN,
            evidence_ref="fixture:synthetic_contract/axis/n_hc_src",
            value=4,
        )
        back = gir.axes_from_json(gir.axes_to_json([axis]))[0]
        self.assertEqual(back, axis)
        for field_name in ("symbol", "name", "meaning", "origin", "evidence_ref"):
            self.assertEqual(getattr(back, field_name), getattr(axis, field_name), field_name)

    def test_required_text_fields_reject_empty_whitespace_and_non_string(self):
        for field_name in ("symbol", "name", "meaning", "origin", "evidence_ref"):
            for bad in ("", "   ", "\t\n", 123, None):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: make_axis(**{f: b}), field_name
                    )

    def test_symbol_rejects_surrounding_whitespace(self):
        exc = self.assertValueErrorLocated(lambda: make_axis(symbol=" hidden_size "), "symbol")
        self.assertIn("空白", str(exc))

    def test_from_dict_missing_metadata_field_is_named(self):
        for field_name in ("stable_id", "symbol", "name", "meaning", "origin", "evidence_ref"):
            data = make_axis().to_dict()
            del data[field_name]
            with self.subTest(field=field_name):
                self.assertValueErrorLocated(
                    lambda d=data: gir.NamedAxis.from_dict(d), field_name
                )

    def test_meaning_with_quotes_and_newline_roundtrips(self):
        tricky = '含义含"引号"、反斜杠\\与换行\n第二行（合成契约样例）'
        axis = make_axis(meaning=tricky)
        text = gir.axes_to_json([axis])
        self.assertNotIn("\n", text)  # 结构字符被 JSON 转义，不破坏单行文档
        back = gir.axes_from_json(text)[0]
        self.assertEqual(back.meaning, tricky)

    def test_meaning_requires_nonempty_not_cjk(self):
        """接口保证是“非空字符串”，并非强制中文。

        这是如实记录的边界：中文可读性由写入方（提取器/契约样例）保证，
        通用校验器不会拒绝纯英文含义。此处只固定当前真实行为，防止把
        “有中文含义字段”误报成“校验器强制中文”。
        """
        axis = make_axis(meaning="hidden dim size (synthetic, ASCII only)")
        back = gir.axes_from_json(gir.axes_to_json([axis]))[0]
        self.assertEqual(back.meaning, axis.meaning)


# ---------------------------------------------------------------------------
# 3. 合法数值 0 与“未知”的区别
# ---------------------------------------------------------------------------
class ZeroVersusUnknownTest(AxisTestCase):
    def test_zero_is_a_legal_known_value(self):
        axis = make_axis(value=0)
        self.assertFalse(axis.is_unknown)
        self.assertEqual(axis.value, 0)
        self.assertIsNone(axis.unknown_reason)

    def test_zero_survives_roundtrip_as_zero(self):
        axis = make_axis(value=0)
        text = gir.axes_to_json([axis])
        self.assertIn('"value":0', text)
        back = gir.axes_from_json(text)[0]
        self.assertFalse(back.is_unknown)
        self.assertEqual(back.value, 0)
        self.assertIsInstance(back.value, int)
        self.assertFalse(isinstance(back.value, bool))

    def test_unknown_serializes_null_value_with_reason(self):
        axis = make_unknown(reason="来源未给出该轴数值（合成契约样例）")
        self.assertTrue(axis.is_unknown)
        text = gir.axes_to_json([axis])
        self.assertIn('"value":null', text)
        self.assertIn('"expression":null', text)
        self.assertIn("来源未给出该轴数值", text)

    def test_zero_and_unknown_never_collapse(self):
        known_zero = make_axis(value=0)
        unknown = make_unknown()
        text_zero = gir.axes_to_json([known_zero])
        text_unknown = gir.axes_to_json([unknown])
        self.assertNotEqual(text_zero, text_unknown)
        self.assertNotEqual(text_zero.encode(), text_unknown.encode())

        back_zero = gir.axes_from_json(text_zero)[0]
        back_unknown = gir.axes_from_json(text_unknown)[0]
        self.assertFalse(back_zero.is_unknown)
        self.assertTrue(back_unknown.is_unknown)
        self.assertIsNone(back_unknown.value)
        self.assertNotEqual(back_zero.value, back_unknown.value)
        # 未知不得退化为 0/false
        self.assertIsNot(back_unknown.value, 0)
        self.assertIsNot(back_unknown.value, False)
        self.assertFalse(back_unknown.unknown_reason is None)

    def test_unknown_roundtrip_stays_unknown(self):
        axis = make_unknown(reason="容量取决于运行配置（合成契约样例）")
        back = gir.axes_from_json(gir.axes_to_json([axis]))[0]
        self.assertEqual(back, axis)
        self.assertTrue(back.is_unknown)
        self.assertEqual(back.unknown_reason, axis.unknown_reason)

    def test_missing_or_null_value_key_means_unknown_only_with_reason(self):
        data = make_unknown().to_dict()
        for key in ("value", "expression"):
            del data[key]
        with self.subTest(case="keys absent"):
            self.assertTrue(gir.NamedAxis.from_dict(data).is_unknown)
        data["value"] = None
        data["expression"] = None
        with self.subTest(case="null values"):
            self.assertTrue(gir.NamedAxis.from_dict(data).is_unknown)

    def test_unknown_without_reason_rejected_on_construct_and_parse(self):
        self.assertValueErrorLocated(
            lambda: gir.NamedAxis(
                symbol="capacity",
                name="capacity",
                meaning="缓存容量（合成契约样例）",
                origin=SYNTHETIC_ORIGIN,
                evidence_ref=SYNTHETIC_EVIDENCE,
            ),
            "unknown_reason",
        )
        for bad in (None, "", "   "):
            data = {
                "stable_id": "axis:capacity",
                "symbol": "capacity",
                "name": "capacity",
                "meaning": "缓存容量（合成契约样例）",
                "origin": SYNTHETIC_ORIGIN,
                "evidence_ref": SYNTHETIC_EVIDENCE,
                "value": None,
                "expression": None,
                "unknown_reason": bad,
            }
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda d=data: gir.NamedAxis.from_dict(d), "unknown_reason"
                )

    def test_non_integer_values_rejected(self):
        # bool 是 int 子类，必须被显式拒绝，否则 true/false 会冒充数值
        for bad in (True, False, "4", 4.0, 4.5, [4], {"value": 4}):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: make_axis(value=b), "value")


# ---------------------------------------------------------------------------
# 4. value/expression 互斥 与 unknown_reason 规则
# ---------------------------------------------------------------------------
class MutualExclusionTest(AxisTestCase):
    def test_value_and_expression_are_mutually_exclusive(self):
        exc = self.assertValueErrorLocated(
            lambda: make_axis(value=4096, expression="2 * hidden_size"), "value", "expression"
        )
        self.assertIn("互斥", str(exc))

    def test_value_only_and_expression_only_are_both_legal_and_distinct(self):
        by_value = make_axis(value=4096)
        by_expr = make_expression(expr="2 * hidden_size")
        self.assertFalse(by_value.is_unknown)
        self.assertFalse(by_expr.is_unknown)
        self.assertIsNone(by_expr.value)
        self.assertEqual(by_expr.expression, "2 * hidden_size")
        text_value = gir.axes_to_json([by_value])
        text_expr = gir.axes_to_json([by_expr])
        self.assertNotEqual(text_value, text_expr)
        self.assertEqual(gir.axes_from_json(text_expr)[0].expression, "2 * hidden_size")
        self.assertEqual(gir.axes_from_json(text_value)[0].value, 4096)

    def test_expression_rejects_blank_and_multiline(self):
        for bad in ("", "   ", "2 * hidden_size\n+ 1", "a\nexpr"):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: make_axis(value=None, expression=b), "expression"
                )

    def test_known_axis_forbids_unknown_reason(self):
        for built in (
            lambda: make_axis(value=0, unknown_reason="不该出现"),
            lambda: make_expression(unknown_reason="不该出现"),
        ):
            with self.subTest(built=built):
                self.assertValueErrorLocated(built, "unknown_reason")

    def test_unknown_reason_on_unknown_axis_roundtrips_chinese(self):
        axis = make_unknown(reason="该轴数值依赖运行配置，来源未给出（合成契约样例）")
        text = gir.axes_to_json([axis])
        self.assertIn("依赖运行配置", text)
        back = gir.axes_from_json(text)[0]
        self.assertEqual(back.unknown_reason, axis.unknown_reason)
        self.assertTrue(back.is_unknown)


# ---------------------------------------------------------------------------
# 5. 显式稳定 ID
# ---------------------------------------------------------------------------
class StableIdTest(AxisTestCase):
    def test_stable_id_is_embedded_in_serialized_axes(self):
        axis = make_axis(symbol="hidden_size")
        text = gir.axes_to_json([axis])
        payload = json.loads(text)
        self.assertEqual(payload["axes"][0]["stable_id"], "axis:hidden_size")
        self.assertEqual(payload["axes"][0]["stable_id"], axis.to_dict()["stable_id"])

    def test_tampered_stable_id_rejected(self):
        data = make_axis().to_dict()
        data["stable_id"] = "axis:other_symbol"
        self.assertValueErrorLocated(
            lambda: gir.NamedAxis.from_dict(data), "stable_id", "symbol"
        )

    def test_stable_id_must_differ_from_symbol_prefix_form(self):
        data = make_axis(symbol="hidden_size").to_dict()
        data["stable_id"] = "hidden_size"  # 缺 axis: 前缀
        self.assertValueErrorLocated(lambda: gir.NamedAxis.from_dict(data), "stable_id")

    def test_stable_id_has_no_time_or_random_component(self):
        first = make_axis(symbol="hidden_size")
        second = gir.NamedAxis(
            symbol="hidden_size",
            name="hidden_size",
            meaning="隐藏维度大小（合成契约样例，非 DeepSeek 结构）",
            origin=SYNTHETIC_ORIGIN,
            evidence_ref=SYNTHETIC_EVIDENCE,
            value=4096,
        )
        self.assertEqual(first.stable_id, second.stable_id)
        self.assertEqual(first.stable_id, "axis:hidden_size")
        self.assertEqual(gir.axes_to_json([first]), gir.axes_to_json([second]))

    def test_stable_id_and_bytes_deterministic_across_processes(self):
        """更换 PYTHONHASHSEED 与进程后，ID 与 canonical 字节必须完全一致。"""
        axis = make_axis(symbol="hidden_size")
        expected = gir.axes_to_json([axis])
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import NamedAxis, axes_to_json\n"
            "axis = NamedAxis(\n"
            "    symbol='hidden_size',\n"
            "    name='hidden_size',\n"
            f"    meaning={make_axis().meaning!r},\n"
            f"    origin={SYNTHETIC_ORIGIN!r},\n"
            f"    evidence_ref={SYNTHETIC_EVIDENCE!r},\n"
            "    value=4096,\n"
            ")\n"
            "sys.stdout.write(axis.stable_id + '\\n' + axes_to_json([axis]))\n"
        )
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            with self.subTest(pythonhashseed=seed):
                proc = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=60,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                stable_id, _, payload = proc.stdout.partition("\n")
                self.assertEqual(stable_id, "axis:hidden_size")
                self.assertEqual(payload, expected)


# ---------------------------------------------------------------------------
# 6. 重复轴
# ---------------------------------------------------------------------------
class DuplicateAxisTest(AxisTestCase):
    def test_duplicate_symbol_rejected_and_located_on_serialize(self):
        dup = [make_axis(symbol="hidden_size"), make_axis(symbol="hidden_size")]
        exc = self.assertValueErrorLocated(lambda: gir.axes_to_json(dup), "axis:hidden_size")
        self.assertIn("重复", str(exc))

    def test_duplicate_detected_when_not_adjacent(self):
        axes = [
            make_axis(symbol="n_hc_src", value=4),
            make_axis(symbol="hidden_size", value=4096),
            make_unknown(symbol="n_hc_src"),
        ]
        self.assertValueErrorLocated(lambda: gir.axes_to_json(axes), "axis:n_hc_src")

    def test_same_symbol_with_different_metadata_still_duplicate(self):
        axes = [
            make_axis(symbol="hidden_size", meaning="隐藏维度（合成契约样例）", value=4096),
            make_axis(symbol="hidden_size", meaning="另一处同名轴（合成契约样例）", value=2048),
        ]
        self.assertValueErrorLocated(lambda: gir.axes_to_json(axes), "axis:hidden_size")

    def test_duplicate_rejected_on_deserialize(self):
        one = make_axis().to_dict()
        text = envelope([one, dict(one)])
        self.assertValueErrorLocated(lambda: gir.axes_from_json(text), "axis:hidden_size")

    def test_distinct_symbols_with_equal_values_are_allowed(self):
        axes = [
            make_axis(symbol="n_hc_src", name="N_hc_src", value=4),
            make_axis(symbol="n_hc_dst", name="N_hc_dst", value=4),
        ]
        back = gir.axes_from_json(gir.axes_to_json(axes))
        self.assertEqual([a.symbol for a in back], ["n_hc_src", "n_hc_dst"])
        self.assertEqual({a.value for a in back}, {4})


# ---------------------------------------------------------------------------
# 7. 轴顺序
# ---------------------------------------------------------------------------
class AxisOrderTest(AxisTestCase):
    def test_array_order_is_axis_order_and_not_sorted(self):
        symbols = ["n_hc_src", "hidden_size", "kv_lora_rank"]
        axes = [make_axis(symbol=s) for s in symbols]
        payload = json.loads(gir.axes_to_json(axes))
        self.assertEqual([a["symbol"] for a in payload["axes"]], symbols)
        self.assertNotEqual(symbols, sorted(symbols))  # 样例确实是乱序，能证明未重排

    def test_reverse_order_yields_different_bytes_and_preserves_order(self):
        forward = [make_axis(symbol=s) for s in ("n_hc_src", "hidden_size")]
        backward = list(reversed(forward))
        text_forward = gir.axes_to_json(forward)
        text_backward = gir.axes_to_json(backward)
        self.assertNotEqual(text_forward, text_backward)
        self.assertEqual(
            [a.symbol for a in gir.axes_from_json(text_forward)],
            ["n_hc_src", "hidden_size"],
        )
        self.assertEqual(
            [a.symbol for a in gir.axes_from_json(text_backward)],
            ["hidden_size", "n_hc_src"],
        )

    def test_order_preserved_through_deserialization_is_order_sensitive(self):
        axes = [make_axis(symbol=s) for s in ("hidden_size", "n_hc_dst", "n_hc_src")]
        back = gir.axes_from_json(gir.axes_to_json(axes))
        self.assertEqual(back, axes)  # 列表相等按位置比较
        self.assertNotEqual(list(reversed(back)), axes)

    def test_canonical_form_sorts_object_keys_but_not_axis_array(self):
        axes = [make_axis(symbol="n_hc_src"), make_axis(symbol="hidden_size")]
        text = gir.axes_to_json(axes)
        pairs = json.loads(text, object_pairs_hook=lambda p: p)
        keys = [k for k, _ in pairs]
        self.assertEqual(keys, sorted(keys))  # 对象键排序（canonical）
        axis_pairs = dict(pairs)["axes"]
        self.assertEqual([dict(a)["symbol"] for a in axis_pairs], ["n_hc_src", "hidden_size"])

    def test_compact_separators_no_padding_whitespace(self):
        text = gir.axes_to_json([make_axis(symbol="hidden_size")])
        self.assertNotIn('", "', text)
        self.assertNotIn('": "', text)

    def test_repeated_serialization_is_byte_identical(self):
        axes = [
            make_axis(symbol="n_hc_src", value=4),
            make_expression(symbol="hidden_size", expr="2 * hidden_size"),
            make_unknown(symbol="kv_lora_rank"),
        ]
        first = gir.axes_to_json(axes)
        second = gir.axes_to_json(axes)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))


# ---------------------------------------------------------------------------
# 8. canonical JSON 往返与受控破坏（信封/项）
# ---------------------------------------------------------------------------
class CanonicalRoundTripTest(AxisTestCase):
    def test_mixed_axes_roundtrip_equality(self):
        axes = [
            make_axis(symbol="n_hc_src", name="N_hc_src", value=4),
            make_axis(symbol="n_hc_dst", name="N_hc_dst", value=4),
            make_axis(symbol="hidden_size", value=0),  # 合法 0，不是未知
            make_expression(symbol="head_dim", expr="hidden_size // num_heads"),
            make_unknown(symbol="kv_lora_rank"),
        ]
        back = gir.axes_from_json(gir.axes_to_json(axes))
        self.assertEqual(back, axes)
        self.assertFalse(back[2].is_unknown)
        self.assertTrue(back[4].is_unknown)

    def test_serialize_deserialize_serialize_byte_stable(self):
        axes = [make_axis(symbol="hidden_size"), make_unknown(symbol="capacity")]
        once = gir.axes_to_json(axes)
        twice = gir.axes_to_json(gir.axes_from_json(once))
        self.assertEqual(once.encode("utf-8"), twice.encode("utf-8"))

    def test_empty_axis_list_roundtrips(self):
        text = gir.axes_to_json([])
        self.assertEqual(json.loads(text)["axes"], [])
        self.assertEqual(gir.axes_from_json(text), [])

    def test_tuple_input_accepted(self):
        axes = (make_axis(symbol="hidden_size"), make_unknown(symbol="capacity"))
        self.assertEqual([a.symbol for a in gir.axes_from_json(gir.axes_to_json(axes))],
                         ["hidden_size", "capacity"])

    def test_envelope_schema_must_match(self):
        good = make_axis().to_dict()
        self.assertValueErrorLocated(
            lambda: gir.axes_from_json(envelope([good], schema="semantic-graph/v2")), "schema"
        )
        self.assertValueErrorLocated(
            lambda: gir.axes_from_json(envelope([good], schema=None)), "schema"
        )

    def test_envelope_kind_must_match(self):
        good = make_axis().to_dict()
        self.assertValueErrorLocated(
            lambda: gir.axes_from_json(envelope([good], kind="graph")), "kind"
        )
        self.assertValueErrorLocated(
            lambda: gir.axes_from_json(envelope([good], kind=None)), "kind"
        )

    def test_top_level_must_be_object(self):
        for text in ("[]", "[1, 2]", '"text"', "3"):
            with self.subTest(text=text):
                self.assertValueErrorLocated(lambda t=text: gir.axes_from_json(t), "顶层")

    def test_axes_must_be_list(self):
        for payload in (
            {"schema": SCHEMA_VERSION, "kind": KIND_NAMED_AXES},
            {"schema": SCHEMA_VERSION, "kind": KIND_NAMED_AXES, "axes": {"a": 1}},
            {"schema": SCHEMA_VERSION, "kind": KIND_NAMED_AXES, "axes": "axis"},
        ):
            with self.subTest(payload=payload):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.axes_from_json(json.dumps(p, ensure_ascii=False)),
                    "axes",
                )

    def test_axis_item_must_be_object(self):
        for item in ("axis", 3, None, ["stable_id"]):
            with self.subTest(item=item):
                text = envelope([item])
                self.assertValueErrorLocated(
                    lambda t=text: gir.axes_from_json(t), "轴记录", "对象"
                )

    def test_malformed_json_rejected_as_value_error(self):
        for text in ("{not json", "", "{\"schema\": }"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as ctx:
                    gir.axes_from_json(text)
                # 可归类（json.JSONDecodeError 是 ValueError 子类），不静默返回半成品
                self.assertIsInstance(ctx.exception, ValueError)


# ---------------------------------------------------------------------------
# 9. 无重依赖 / 不执行任意表达式
# ---------------------------------------------------------------------------
class NoHeavyDependencyTest(AxisTestCase):
    ALLOWED_STDLIB_ROOTS = {"__future__", "json", "collections", "dataclasses", "typing"}
    FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}

    @classmethod
    def setUpClass(cls):
        cls.source = PRODUCT_FILE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(PRODUCT_FILE))

    def test_module_level_imports_are_stdlib_only(self):
        imported = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    self.fail(f"graph_ir.py 不应有相对导入：level={node.level} module={node.module}")
                imported.add(node.module or "")
        self.assertTrue(imported, "应至少导入 json")
        for name in sorted(imported):
            root = name.split(".")[0]
            with self.subTest(module=name):
                self.assertIn(root, self.ALLOWED_STDLIB_ROOTS)
                self.assertNotIn(root, BLOCKED_HEAVY_MODULES)

    def test_source_has_no_eval_exec_compile(self):
        offenders = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.FORBIDDEN_CALLS:
                    offenders.append((node.func.id, node.lineno))
        self.assertEqual(offenders, [], f"发现任意代码执行调用：{offenders}")

    def _run_blocked_subprocess(self, body: str):
        blocker = (
            "import sys\n"
            "import importlib.abc\n"
            f"BLOCKED = {set(BLOCKED_HEAVY_MODULES)!r}\n"
            "class _Blocker(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in BLOCKED:\n"
            "            raise ImportError('blocked heavy dependency: ' + fullname)\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Blocker())\n"
        )
        script = blocker + body + (
            "\nloaded = sorted(m for m in BLOCKED if m in sys.modules)\n"
            "if loaded:\n"
            "    raise SystemExit('HEAVY_LOADED:' + ','.join(loaded))\n"
            "print('NO_HEAVY_DEPENDENCY_OK')\n"
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_no_blocked_module_loaded_via_package_import(self):
        body = (
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "import structure_reader.graph_ir as gir\n"
            "assert gir.axes_to_json([]).startswith('{')\n"
        )
        proc = self._run_blocked_subprocess(body)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NO_HEAVY_DEPENDENCY_OK", proc.stdout)

    def test_no_blocked_module_loaded_via_direct_file_import(self):
        body = (
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('graph_ir_probe', {str(PRODUCT_FILE)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "sys.modules['graph_ir_probe'] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "assert mod.SCHEMA == 'semantic-graph/v1'\n"
        )
        proc = self._run_blocked_subprocess(body)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NO_HEAVY_DEPENDENCY_OK", proc.stdout)

    def test_pyproject_declares_no_runtime_dependencies(self):
        import tomllib

        data = tomllib.loads((PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(data["project"]["dependencies"], [])
        optional = data["project"].get("optional-dependencies", {})
        declared = " ".join(str(v) for v in optional.values()).lower()
        for heavy in ("jsonschema", "torch", "transformers", "safetensors"):
            self.assertNotIn(heavy, declared, heavy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
