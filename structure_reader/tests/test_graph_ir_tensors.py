"""S1 01a Tensor 探针独立验证：字段、轴/布局顺序、未知与 shape、往返与负例。

范围严格限定于 `results/development_01a_tensor_probe.md` 声明的 Tensor 增量：
- 张量 name 与轴 name 分属不同命名空间；`stable_id` 形如 `tensor:{name}`；
- `dtype` 必填非空；`axes` 有序且冻结为 tuple，允许未知轴，张量内 symbol 不得重复；
- 同数值不同 symbol 的轴按对象各自保留、不合并；
- `logical_layout` / `physical_layout` 各为 axes symbol 的完整排列，顺序即布局顺序，
  允许互不相同（转置），不排序不重排；
- 未知轴以 NamedAxis 的 `unknown_reason` 表达，`value=0` 是合法确定值，二者不混淆；
- 派生 `shape`：全部轴有确定 value 时按 axes 顺序给出，否则为 `None`（不求值表达式）；
- `global_rank` / `local_rank` 为可选非负整数标注（bool 与非整数被拒）；
- `to_dict` / `from_dict` 稳定往返（dataclass 相等、重复与跨进程字节一致）；frozen。

**不在本测试范围**（不在此声称已证）：Node / Edge / SemanticGraph 及图级构造与唯一性、
`graph_to_json`/`graph_from_json`、Port↔Tensor 引用一致性、alias/fold、guard 三态、state
前后版本、template/instance、JSON Schema 与图检查器、表达式求值器、根 API 导出、
真实 DeepSeek 结构提取与阅读页。

纪律：
- 数据为**合成契约样例（synthetic contract fixture）**，非 DeepSeek 结构；
- 期望值来自派工文档与开发报告声明的接口，不取自产品生成的输出（不复制产品输出为 golden）；
- 只用标准库；不安装/导入 jsonschema、torch、transformers、safetensors；
- 不因“尚未实现的类型不存在”而断言（Claude 可能并行加入 Node/Edge），只固定 Tensor 已声明契约。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, MISSING, fields
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import structure_reader.graph_ir as gir  # noqa: E402

ORIGIN = "synthetic_contract_fixture(非DeepSeek结构)"
CANONICAL_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}
TENSOR_TO_DICT_KEYS = {
    "stable_id", "name", "dtype", "axes",
    "logical_layout", "physical_layout", "global_rank", "local_rank",
}


def evidence_ref(symbol):
    return f"fixture:synthetic_contract/tensor/axis/{symbol}"


def known_axis(symbol, axis_name, meaning, value):
    return gir.NamedAxis(symbol=symbol, name=axis_name, meaning=meaning,
                         origin=ORIGIN, evidence_ref=evidence_ref(symbol), value=value)


def unknown_axis(symbol, axis_name, meaning,
                 reason="该轴数值未在来源中给出（合成契约样例）"):
    return gir.NamedAxis(symbol=symbol, name=axis_name, meaning=meaning, origin=ORIGIN,
                         evidence_ref=evidence_ref(symbol), value=None, expression=None,
                         unknown_reason=reason)


def expr_axis(symbol, axis_name, meaning, expr="seq_len * 2"):
    return gir.NamedAxis(symbol=symbol, name=axis_name, meaning=meaning, origin=ORIGIN,
                         evidence_ref=evidence_ref(symbol), value=None, expression=expr)


def make_axes():
    """合成 4 轴样例：B/T/N_hc/H（通用结构名词，非 DeepSeek 语义）。"""
    return [
        known_axis("B", "batch_size", "批大小（合成契约样例，非 DeepSeek 结构）", 1),
        known_axis("T", "seq_len", "序列长度（合成契约样例，非 DeepSeek 结构）", 128),
        known_axis("N_hc", "hc_streams", "HC 流数量（合成契约样例，非 DeepSeek 结构）", 4),
        known_axis("H", "hidden_size", "隐藏维度大小（合成契约样例，非 DeepSeek 结构）", 4096),
    ]


DEFAULT_LOGICAL = ("B", "T", "N_hc", "H")
DEFAULT_PHYSICAL = ("B", "H", "T", "N_hc")  # 与 logical 不同的排列（转置式）


def make_tensor(axes=None, logical_layout=None, physical_layout=None, **overrides):
    spec = {
        "name": "hidden_states",
        "dtype": "bfloat16",
        "axes": make_axes() if axes is None else axes,
        "logical_layout": DEFAULT_LOGICAL if logical_layout is None else logical_layout,
        "physical_layout": DEFAULT_PHYSICAL if physical_layout is None else physical_layout,
    }
    spec.update(overrides)
    return gir.Tensor(**spec)


def tensor_with_axes(axes, physical=None, **overrides):
    """用给定轴列表构造张量，布局自动匹配该列表（physical 默认取倒序）。"""
    logical = [axis.symbol for axis in axes]
    if physical is None:
        physical = list(reversed(logical))
    return make_tensor(axes=list(axes), logical_layout=logical,
                       physical_layout=physical, **overrides)


def canonical(tensor):
    """仅用于稳定性比较：对 to_dict() 做确定性 JSON 编码（非产品 API）。"""
    return json.dumps(tensor.to_dict(), **CANONICAL_KWARGS)


class TensorTestCase(unittest.TestCase):
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
# 1. 接口快照：字段、必填/默认可选、frozen、派生属性
# ---------------------------------------------------------------------------
class TensorInterfaceSnapshotTest(TensorTestCase):
    def test_tensor_is_frozen_dataclass_with_declared_field_order(self):
        self.assertTrue(gir.Tensor.__dataclass_params__.frozen)
        self.assertEqual(
            tuple(f.name for f in fields(gir.Tensor)),
            ("name", "dtype", "axes", "logical_layout", "physical_layout",
             "global_rank", "local_rank"),
        )

    def test_required_fields_and_optional_rank_defaults(self):
        required = {"name", "dtype", "axes", "logical_layout", "physical_layout"}
        by_name = {f.name: f for f in fields(gir.Tensor)}
        for field_name in sorted(required):
            with self.subTest(required=field_name):
                self.assertIs(by_name[field_name].default, MISSING)
        for rank_name in ("global_rank", "local_rank"):
            with self.subTest(optional=rank_name):
                self.assertIsNone(by_name[rank_name].default)
        tensor = make_tensor()
        self.assertIsNone(tensor.global_rank)
        self.assertIsNone(tensor.local_rank)

    def test_frozen_prevents_mutation_and_is_hashable(self):
        tensor = make_tensor()
        for attribute, new_value in (("name", "other"), ("dtype", "float32"),
                                     ("axes", ()), ("global_rank", 1)):
            with self.subTest(attribute=attribute):
                with self.assertRaises(FrozenInstanceError):
                    setattr(tensor, attribute, new_value)
        self.assertEqual(hash(tensor), hash(make_tensor()))
        self.assertEqual(len({tensor, make_tensor()}), 1)

    def test_axes_frozen_to_tuple_with_defensive_copy(self):
        axes_list = make_axes()
        tensor = make_tensor(axes=axes_list, logical_layout=DEFAULT_LOGICAL,
                             physical_layout=DEFAULT_PHYSICAL)
        self.assertIsInstance(tensor.axes, tuple)
        axes_list.append(known_axis("Z", "extra", "额外轴（合成契约样例）", 1))
        self.assertEqual(len(tensor.axes), 4)  # 外部列表后续修改不影响已构造张量

    def test_stable_id_and_shape_are_properties(self):
        tensor = make_tensor()
        self.assertEqual(tensor.stable_id, "tensor:hidden_states")
        self.assertIsInstance(type(tensor).shape, property)
        self.assertEqual(tensor.shape, (1, 128, 4, 4096))

    def test_unexpected_constructor_keyword_rejected(self):
        with self.assertRaises(TypeError):
            gir.Tensor(name="t", dtype="float32", axes=[], logical_layout=[],
                       physical_layout=[], symbol="H")


# ---------------------------------------------------------------------------
# 2. 张量 name / dtype 与“张量名 ≠ 轴名”命名空间
# ---------------------------------------------------------------------------
class TensorNameDtypeTest(TensorTestCase):
    def test_name_and_dtype_must_be_nonempty_strings(self):
        for field_name in ("name", "dtype"):
            for bad in ("", "   ", "\t\n", None, 123, b"x"):
                with self.subTest(field=field_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: make_tensor(**{f: b}), field_name
                    )

    def test_name_and_dtype_reject_surrounding_whitespace(self):
        for field_name in ("name", "dtype"):
            for bad in (" value", "value ", "\tvalue", "value\n"):
                with self.subTest(field=field_name, bad=bad):
                    exc = self.assertValueErrorLocated(
                        lambda f=field_name, b=bad: make_tensor(**{f: b}), field_name
                    )
                    self.assertIn("空白", str(exc))

    def test_common_dtype_strings_accepted(self):
        for dtype in ("float32", "bfloat16", "float16", "int8", "uint8", "float8_e4m3fn"):
            with self.subTest(dtype=dtype):
                tensor = make_tensor(dtype=dtype)
                self.assertEqual(tensor.dtype, dtype)
                self.assertEqual(gir.Tensor.from_dict(tensor.to_dict()).dtype, dtype)

    def test_tensor_name_namespace_is_separate_from_axis_name(self):
        tensor = make_tensor(name="seq_len")  # 与第二个轴的 name 同名
        axis_names = [axis.name for axis in tensor.axes]
        self.assertIn("seq_len", axis_names)
        self.assertEqual(tensor.name, "seq_len")
        payload = tensor.to_dict()
        self.assertEqual(payload["name"], "seq_len")          # 张量名
        self.assertEqual(payload["axes"][1]["name"], "seq_len")  # 轴名
        self.assertEqual(payload["axes"][1]["symbol"], "T")      # 轴 symbol
        self.assertEqual(payload["stable_id"], "tensor:seq_len")
        self.assertEqual(payload["axes"][1]["stable_id"], "axis:T")
        self.assertEqual(gir.Tensor.from_dict(payload).name, "seq_len")

    def test_tensor_name_may_equal_axis_symbol_without_collision(self):
        tensor = make_tensor(name="H")  # 与第四轴 symbol 相同
        self.assertEqual(tensor.stable_id, "tensor:H")
        self.assertEqual(tensor.axes[3].stable_id, "axis:H")
        self.assertNotEqual(tensor.stable_id, tensor.axes[3].stable_id)
        self.assertEqual([axis.symbol for axis in gir.Tensor.from_dict(tensor.to_dict()).axes],
                         ["B", "T", "N_hc", "H"])


# ---------------------------------------------------------------------------
# 3. 稳定 ID
# ---------------------------------------------------------------------------
class TensorStableIdTest(TensorTestCase):
    def test_stable_id_derivation_and_serialization(self):
        tensor = make_tensor(name="kv_cache")
        self.assertEqual(tensor.stable_id, "tensor:kv_cache")
        payload = tensor.to_dict()
        self.assertEqual(payload["stable_id"], "tensor:kv_cache")
        self.assertEqual([a["stable_id"] for a in payload["axes"]],
                         ["axis:B", "axis:T", "axis:N_hc", "axis:H"])
        self.assertEqual(gir.Tensor.from_dict(payload), tensor)

    def test_tampered_stable_id_rejected(self):
        payload = make_tensor().to_dict()
        payload["stable_id"] = "tensor:other"
        self.assertValueErrorLocated(
            lambda: gir.Tensor.from_dict(payload), "stable_id", "name"
        )

    def test_missing_stable_id_named_in_error(self):
        payload = make_tensor().to_dict()
        del payload["stable_id"]
        self.assertValueErrorLocated(lambda: gir.Tensor.from_dict(payload), "stable_id")

    def test_stable_id_is_deterministic_and_name_derived(self):
        first = make_tensor()
        second = make_tensor()
        self.assertEqual(first.stable_id, second.stable_id)
        self.assertEqual(canonical(first), canonical(second))
        self.assertNotEqual(make_tensor(name="kv_cache").stable_id, first.stable_id)


# ---------------------------------------------------------------------------
# 4. 轴顺序、中文含义/来源、同值不同 symbol、重复轴
# ---------------------------------------------------------------------------
class TensorAxisOrderMetadataTest(TensorTestCase):
    def test_axes_order_preserved_and_not_sorted(self):
        axes = [
            known_axis("H", "hidden_size", "隐藏维度大小（合成契约样例）", 4096),
            known_axis("B", "batch_size", "批大小（合成契约样例）", 1),
            known_axis("N_hc", "hc_streams", "HC 流数量（合成契约样例）", 4),
            known_axis("T", "seq_len", "序列长度（合成契约样例）", 128),
        ]
        symbols = [axis.symbol for axis in axes]
        self.assertNotEqual(symbols, sorted(symbols))  # 样例确实乱序，能证明未重排
        tensor = tensor_with_axes(axes)
        self.assertEqual([axis.symbol for axis in tensor.axes], symbols)
        self.assertEqual([a["symbol"] for a in tensor.to_dict()["axes"]], symbols)
        self.assertEqual([a.symbol for a in gir.Tensor.from_dict(tensor.to_dict()).axes],
                         symbols)
        self.assertEqual(tensor.shape, (4096, 1, 4, 128))
        self.assertEqual(tensor.logical_layout, tuple(symbols))

    def test_reversed_axes_produce_different_tensor_and_roundtrip(self):
        axes = make_axes()
        forward = tensor_with_axes(axes)
        backward = tensor_with_axes(list(reversed(axes)))
        self.assertNotEqual(forward, backward)
        self.assertNotEqual(canonical(forward), canonical(backward))
        self.assertEqual([a.symbol for a in gir.Tensor.from_dict(forward.to_dict()).axes],
                         ["B", "T", "N_hc", "H"])
        self.assertEqual([a.symbol for a in gir.Tensor.from_dict(backward.to_dict()).axes],
                         ["H", "N_hc", "T", "B"])

    def test_chinese_meaning_origin_and_evidence_ref_preserved_raw(self):
        tensor = make_tensor()
        text = canonical(tensor)
        self.assertIn("隐藏维度大小", text)
        self.assertIn(ORIGIN, text)
        self.assertIn("fixture:synthetic_contract/tensor/axis/H", text)
        self.assertNotIn("\\u", text)
        back = gir.Tensor.from_dict(json.loads(text))
        self.assertEqual([a.meaning for a in back.axes], [a.meaning for a in tensor.axes])
        self.assertEqual([a.origin for a in back.axes], [a.origin for a in tensor.axes])
        self.assertEqual([a.evidence_ref for a in back.axes],
                         [a.evidence_ref for a in tensor.axes])

    def test_unknown_axis_reason_survives_tensor_roundtrip(self):
        axes = [
            known_axis("B", "batch_size", "批大小（合成契约样例）", 1),
            unknown_axis("T", "seq_len", "序列长度（合成契约样例）",
                         reason="变长，来源未给出具体值（合成契约样例）"),
        ]
        tensor = tensor_with_axes(axes)
        self.assertTrue(tensor.axes[1].is_unknown)
        self.assertIn("变长，来源未给出具体值", canonical(tensor))
        back = gir.Tensor.from_dict(tensor.to_dict())
        self.assertTrue(back.axes[1].is_unknown)
        self.assertEqual(back.axes[1].unknown_reason, tensor.axes[1].unknown_reason)
        self.assertEqual(back, tensor)

    def test_same_value_different_symbol_not_merged(self):
        axes = [
            known_axis("s", "streams_src", "源侧流数量（合成契约样例）", 4096),
            known_axis("h", "streams_dst", "目标侧流数量（合成契约样例）", 4096),
        ]
        tensor = tensor_with_axes(axes)
        self.assertEqual(len(tensor.axes), 2)
        self.assertEqual({axis.symbol for axis in tensor.axes}, {"s", "h"})
        self.assertEqual(tensor.shape, (4096, 4096))
        payload = tensor.to_dict()
        self.assertEqual([a["symbol"] for a in payload["axes"]], ["s", "h"])
        self.assertEqual([a["stable_id"] for a in payload["axes"]], ["axis:s", "axis:h"])
        back = gir.Tensor.from_dict(payload)
        self.assertEqual(len(back.axes), 2)
        self.assertNotEqual(back.axes[0], back.axes[1])
        self.assertEqual(back.logical_layout, ("s", "h"))

    def test_duplicate_axis_symbol_rejected_adjacent_and_non_adjacent(self):
        h = known_axis("H", "hidden_size", "隐藏维度大小（合成契约样例）", 4096)
        h_other = known_axis("H", "hidden_size_dup", "另一处同名轴（合成契约样例）", 2048)
        b = known_axis("B", "batch_size", "批大小（合成契约样例）", 1)
        for axes in ([h, h_other], [h, b, h_other]):
            with self.subTest(count=len(axes)):
                exc = self.assertValueErrorLocated(
                    lambda a=axes: tensor_with_axes(a), "axis:H"
                )
                self.assertIn("重复", str(exc))

    def test_duplicate_symbol_with_different_metadata_still_rejected(self):
        axes = [
            known_axis("H", "hidden_size", "隐藏维度大小（合成契约样例）", 4096),
            unknown_axis("H", "hidden_size_unknown", "同名但未知的轴（合成契约样例）"),
        ]
        self.assertValueErrorLocated(lambda: tensor_with_axes(axes), "axis:H")

    def test_axes_must_be_list_or_tuple(self):
        for bad in ("BTNH", {"B": 1}, {known_axis("B", "b", "批大小（合成契约样例）", 1)},
                    (a for a in make_axes())):
            with self.subTest(bad=type(bad).__name__):
                self.assertValueErrorLocated(lambda b=bad: make_tensor(axes=b), "axes")

    def test_axis_elements_must_be_named_axis(self):
        payload = {"symbol": "B", "name": "batch_size", "meaning": "批大小（合成契约样例）",
                   "origin": ORIGIN, "evidence_ref": evidence_ref("B"), "value": 1}
        b = known_axis("B", "batch_size", "批大小（合成契约样例）", 1)
        # 直接构造：axes 元素校验先于布局校验，故布局可给占位值
        for axes, needle in (([payload], "axes[0]"), (["B"], "axes[0]"),
                             ([b, "H"], "axes[1]"), ([b, None], "axes[1]")):
            with self.subTest(needle=needle):
                self.assertValueErrorLocated(
                    lambda a=axes: gir.Tensor(name="t", dtype="float32", axes=a,
                                              logical_layout=["B", "H"],
                                              physical_layout=["H", "B"]),
                    needle,
                )


# ---------------------------------------------------------------------------
# 5. 未知轴 / value=0 / shape
# ---------------------------------------------------------------------------
class TensorUnknownAndShapeTest(TensorTestCase):
    def test_shape_known_follows_axes_order(self):
        axes = [
            known_axis("H", "hidden_size", "隐藏维度大小（合成契约样例）", 4096),
            known_axis("B", "batch_size", "批大小（合成契约样例）", 1),
        ]
        self.assertEqual(tensor_with_axes(axes).shape, (4096, 1))

    def test_zero_value_axis_is_known_and_yields_zero_in_shape(self):
        axes = [
            known_axis("B", "batch_size", "批大小（合成契约样例）", 0),
            known_axis("T", "seq_len", "序列长度（合成契约样例）", 0),
        ]
        tensor = tensor_with_axes(axes)
        self.assertEqual(tensor.shape, (0, 0))
        self.assertFalse(any(axis.is_unknown for axis in tensor.axes))
        self.assertIn('"value":0', canonical(tensor))
        self.assertEqual(gir.Tensor.from_dict(tensor.to_dict()).shape, (0, 0))

    def test_unknown_axis_makes_shape_none_not_zero(self):
        axes = [
            known_axis("B", "batch_size", "批大小（合成契约样例）", 1),
            unknown_axis("N_hc", "hc_streams", "HC 流数量（合成契约样例）"),
            known_axis("H", "hidden_size", "隐藏维度大小（合成契约样例）", 4096),
        ]
        tensor = tensor_with_axes(axes)
        self.assertIsNone(tensor.shape)
        self.assertNotEqual(tensor.shape, (1, 0, 4096))
        text = canonical(tensor)
        self.assertIn('"value":null', text)
        self.assertIn("该轴数值未在来源中给出", text)
        self.assertIsNone(gir.Tensor.from_dict(tensor.to_dict()).shape)

    def test_expression_axis_makes_shape_none_without_evaluation(self):
        axes = [
            known_axis("B", "batch_size", "批大小（合成契约样例）", 1),
            expr_axis("T", "seq_len", "序列长度（合成契约样例）", expr="seq_len * 2"),
        ]
        tensor = tensor_with_axes(axes)
        self.assertFalse(tensor.axes[1].is_unknown)  # 有表达式 ≠ 未知
        self.assertIsNone(tensor.shape)              # 但不求值 → 形状未知
        self.assertIn('"expression":"seq_len * 2"', canonical(tensor))
        self.assertIsNone(gir.Tensor.from_dict(tensor.to_dict()).shape)

    def test_unknown_and_zero_never_collapse(self):
        unknown_tensor = tensor_with_axes([
            known_axis("B", "batch_size", "批大小（合成契约样例）", 1),
            unknown_axis("N_hc", "hc_streams", "HC 流数量（合成契约样例）"),
        ])
        zero_tensor = tensor_with_axes([
            known_axis("B", "batch_size", "批大小（合成契约样例）", 1),
            known_axis("N_hc", "hc_streams", "HC 流数量（合成契约样例）", 0),
        ])
        self.assertIsNone(unknown_tensor.shape)
        self.assertEqual(zero_tensor.shape, (1, 0))
        self.assertNotEqual(unknown_tensor, zero_tensor)
        self.assertNotEqual(canonical(unknown_tensor), canonical(zero_tensor))
        self.assertIsNone(unknown_tensor.axes[1].value)
        self.assertEqual(zero_tensor.axes[1].value, 0)
        self.assertFalse(isinstance(zero_tensor.axes[1].value, bool))
        self.assertEqual(gir.Tensor.from_dict(unknown_tensor.to_dict()).shape, None)
        self.assertEqual(gir.Tensor.from_dict(zero_tensor.to_dict()).shape, (1, 0))

    def test_empty_axes_scalar_tensor_shape_is_empty_tuple(self):
        """边界观察：空 axes 与空布局被接受，shape 为 ()（秩 0 张量）。

        派工未规定空轴是否合法；此处固定当前真实行为，若后续收紧须重新分发测试单。
        """
        tensor = make_tensor(axes=[], logical_layout=[], physical_layout=[])
        self.assertEqual(tensor.axes, ())
        self.assertEqual(tensor.shape, ())
        self.assertEqual(gir.Tensor.from_dict(tensor.to_dict()), tensor)

    def test_rank_not_cross_checked_against_axis_count(self):
        """边界观察：global_rank/local_rank 只做“非负整数或 None”，不与 len(axes) 交叉校验。"""
        tensor = make_tensor(global_rank=99, local_rank=0)
        self.assertEqual(tensor.global_rank, 99)
        self.assertEqual(tensor.local_rank, 0)
        self.assertEqual(len(tensor.axes), 4)
        self.assertEqual(gir.Tensor.from_dict(tensor.to_dict()), tensor)


# ---------------------------------------------------------------------------
# 6. logical_layout / physical_layout 顺序与转置
# ---------------------------------------------------------------------------
class TensorLayoutTest(TensorTestCase):
    def test_logical_and_physical_layout_preserved_including_transpose(self):
        tensor = make_tensor()
        self.assertEqual(tuple(tensor.logical_layout), ("B", "T", "N_hc", "H"))
        self.assertEqual(tuple(tensor.physical_layout), ("B", "H", "T", "N_hc"))
        self.assertNotEqual(tuple(tensor.logical_layout), tuple(tensor.physical_layout))
        payload = tensor.to_dict()
        self.assertEqual(payload["logical_layout"], ["B", "T", "N_hc", "H"])
        self.assertEqual(payload["physical_layout"], ["B", "H", "T", "N_hc"])
        back = gir.Tensor.from_dict(payload)
        self.assertEqual(tuple(back.logical_layout), ("B", "T", "N_hc", "H"))
        self.assertEqual(tuple(back.physical_layout), ("B", "H", "T", "N_hc"))
        self.assertEqual(back, tensor)

    def test_layout_order_not_sorted(self):
        tensor = make_tensor(logical_layout=("H", "N_hc", "T", "B"),
                             physical_layout=("T", "B", "H", "N_hc"))
        self.assertEqual(tuple(tensor.logical_layout), ("H", "N_hc", "T", "B"))
        self.assertEqual(tuple(tensor.physical_layout), ("T", "B", "H", "N_hc"))
        payload = tensor.to_dict()
        self.assertEqual(payload["logical_layout"], ["H", "N_hc", "T", "B"])
        self.assertEqual(payload["physical_layout"], ["T", "B", "H", "N_hc"])

    def test_equal_layouts_allowed(self):
        tensor = make_tensor(logical_layout=DEFAULT_LOGICAL,
                             physical_layout=DEFAULT_LOGICAL)
        self.assertEqual(tuple(tensor.logical_layout), tuple(tensor.physical_layout))
        self.assertEqual(gir.Tensor.from_dict(tensor.to_dict()), tensor)

    def test_reversed_layout_allowed_and_preserved(self):
        reversed_layout = tuple(reversed(DEFAULT_LOGICAL))
        tensor = make_tensor(logical_layout=DEFAULT_LOGICAL,
                             physical_layout=reversed_layout)
        self.assertEqual(tuple(tensor.physical_layout), ("H", "N_hc", "T", "B"))
        self.assertEqual(tuple(gir.Tensor.from_dict(tensor.to_dict()).physical_layout),
                         ("H", "N_hc", "T", "B"))

    def test_layout_missing_symbol_rejected(self):
        for layout_name in ("logical_layout", "physical_layout"):
            with self.subTest(layout=layout_name):
                exc = self.assertValueErrorLocated(
                    lambda n=layout_name: make_tensor(**{n: ("B", "T", "N_hc")}),
                    layout_name,
                )
                self.assertIn("排列", str(exc))

    def test_layout_extra_foreign_symbol_rejected(self):
        for layout_name in ("logical_layout", "physical_layout"):
            with self.subTest(layout=layout_name):
                self.assertValueErrorLocated(
                    lambda n=layout_name: make_tensor(
                        **{n: ("B", "T", "N_hc", "H", "Hq")}),
                    layout_name,
                )

    def test_layout_duplicate_entry_rejected(self):
        self.assertValueErrorLocated(
            lambda: make_tensor(physical_layout=("B", "B", "N_hc", "H")),
            "physical_layout",
        )

    def test_layout_must_be_list_or_tuple(self):
        for bad in ("BTNH", {"B": 0, "T": 1, "N_hc": 2, "H": 3}, {"B", "T", "N_hc", "H"}):
            with self.subTest(bad=type(bad).__name__):
                self.assertValueErrorLocated(
                    lambda b=bad: make_tensor(logical_layout=b), "logical_layout"
                )
        # 直接构造，避免 make_tensor 的“None 表示用默认值”语义掩盖本用例
        self.assertValueErrorLocated(
            lambda: gir.Tensor(name="t", dtype="float32", axes=[], logical_layout=None,
                               physical_layout=[]),
            "logical_layout",
        )

    def test_layout_entry_must_be_symbol_string(self):
        self.assertValueErrorLocated(
            lambda: make_tensor(physical_layout=("B", "T", "N_hc", 4096)),
            "physical_layout[",
        )
        self.assertValueErrorLocated(
            lambda: make_tensor(physical_layout=("B", "T", "N_hc", None)),
            "physical_layout[",
        )
        self.assertValueErrorLocated(
            lambda: make_tensor(physical_layout=("B", "T", ["N_hc"], "H")),
            "physical_layout[",
        )


# ---------------------------------------------------------------------------
# 7. global_rank / local_rank
# ---------------------------------------------------------------------------
class TensorRankTest(TensorTestCase):
    def test_ranks_default_none_and_roundtrip(self):
        tensor = make_tensor()
        self.assertIsNone(tensor.global_rank)
        self.assertIsNone(tensor.local_rank)
        payload = tensor.to_dict()
        self.assertIsNone(payload["global_rank"])
        self.assertIsNone(payload["local_rank"])
        self.assertEqual(gir.Tensor.from_dict(payload), tensor)
        for key in ("global_rank", "local_rank"):
            reduced = dict(payload)
            del reduced[key]
            with self.subTest(removed=key):
                self.assertEqual(gir.Tensor.from_dict(reduced), tensor)

    def test_non_negative_ranks_accepted(self):
        for value in (0, 1, 4, 128):
            with self.subTest(rank=value):
                tensor = make_tensor(global_rank=value, local_rank=value)
                self.assertEqual(tensor.global_rank, value)
                self.assertEqual(tensor.local_rank, value)
                self.assertEqual(gir.Tensor.from_dict(tensor.to_dict()), tensor)

    def test_negative_rank_rejected_naming_field(self):
        for rank_name in ("global_rank", "local_rank"):
            for bad in (-1, -128):
                with self.subTest(field=rank_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda n=rank_name, b=bad: make_tensor(**{n: b}), rank_name
                    )

    def test_bool_rank_rejected(self):
        for rank_name in ("global_rank", "local_rank"):
            for bad in (True, False):
                with self.subTest(field=rank_name, bad=bad):
                    self.assertValueErrorLocated(
                        lambda n=rank_name, b=bad: make_tensor(**{n: b}), rank_name
                    )

    def test_non_integer_rank_rejected(self):
        for bad in (4.0, "4", [4], {"rank": 4}):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(
                    lambda b=bad: make_tensor(global_rank=b), "global_rank"
                )

    def test_rank_fields_validated_independently(self):
        self.assertValueErrorLocated(
            lambda: make_tensor(global_rank=1, local_rank=-1), "local_rank"
        )
        self.assertValueErrorLocated(
            lambda: make_tensor(global_rank=-1, local_rank=1), "global_rank"
        )


# ---------------------------------------------------------------------------
# 8. to_dict / from_dict 稳定往返与受控破坏
# ---------------------------------------------------------------------------
class TensorRoundTripTest(TensorTestCase):
    def mixed_tensor(self):
        axes = [
            known_axis("B", "batch_size", "批大小（合成契约样例）", 1),
            known_axis("T", "seq_len", "序列长度（合成契约样例）", 0),
            unknown_axis("N_hc", "hc_streams", "HC 流数量（合成契约样例）"),
            expr_axis("H", "hidden_size", "隐藏维度大小（合成契约样例）", expr="d_model // 2"),
        ]
        return tensor_with_axes(axes, physical=["H", "N_hc", "T", "B"],
                                global_rank=4, local_rank=1)

    def test_to_dict_shape_and_keys(self):
        payload = make_tensor().to_dict()
        self.assertEqual(set(payload), TENSOR_TO_DICT_KEYS)
        self.assertEqual(payload["name"], "hidden_states")
        self.assertEqual(payload["dtype"], "bfloat16")
        self.assertEqual(len(payload["axes"]), 4)

    def test_roundtrip_equality_for_mixed_tensor(self):
        tensor = self.mixed_tensor()
        self.assertIsNone(tensor.shape)  # 含未知与表达式轴
        back = gir.Tensor.from_dict(tensor.to_dict())
        self.assertEqual(back, tensor)
        self.assertEqual(back.to_dict(), tensor.to_dict())
        self.assertIsNone(back.shape)
        self.assertEqual(tuple(back.physical_layout), ("H", "N_hc", "T", "B"))
        self.assertEqual(back.global_rank, 4)
        self.assertTrue(back.axes[2].is_unknown)
        self.assertFalse(back.axes[3].is_unknown)

    def test_repeated_canonical_serialization_byte_identical(self):
        tensor = self.mixed_tensor()
        first = canonical(tensor)
        second = canonical(tensor)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        self.assertEqual(first, canonical(gir.Tensor.from_dict(tensor.to_dict())))

    def test_canonical_bytes_deterministic_across_processes(self):
        tensor = self.mixed_tensor()
        expected = canonical(tensor)
        literal = repr(json.dumps(tensor.to_dict(), ensure_ascii=False))
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(SRC_DIR)!r})\n"
            "from structure_reader.graph_ir import Tensor\n"
            f"data = json.loads({literal})\n"
            "t = Tensor.from_dict(data)\n"
            "sys.stdout.write(json.dumps(t.to_dict(), ensure_ascii=False, sort_keys=True,"
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
        for bad in ([], "tensor", 3, None, True):
            with self.subTest(bad=bad):
                self.assertValueErrorLocated(lambda b=bad: gir.Tensor.from_dict(b),
                                             "张量", "对象")

    def test_from_dict_missing_fields_named(self):
        for missing in ("stable_id", "name", "dtype", "axes",
                        "logical_layout", "physical_layout"):
            payload = make_tensor().to_dict()
            del payload[missing]
            with self.subTest(missing=missing):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.Tensor.from_dict(p), missing
                )

    def test_from_dict_axes_must_be_json_array(self):
        for bad in ({"B": 1}, "BTNH", 3, None):
            payload = make_tensor().to_dict()
            payload["axes"] = bad
            with self.subTest(bad=type(bad).__name__):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.Tensor.from_dict(p), "axes"
                )

    def test_from_dict_rejects_bad_nested_axis_record(self):
        base = make_tensor().to_dict()
        cases = []
        broken_item = dict(base)
        broken_item["axes"] = [{"symbol": "B"}]
        cases.append(("轴项非对象", broken_item, ("轴记录", "对象")))
        bad_stable = make_tensor().to_dict()
        bad_stable["axes"][0]["stable_id"] = "axis:other"
        cases.append(("轴stable_id不一致", bad_stable, ("stable_id", "symbol")))
        missing_reason = make_tensor().to_dict()
        missing_reason["axes"][1] = {
            "stable_id": "axis:T", "symbol": "T", "name": "seq_len",
            "meaning": "序列长度（合成契约样例）", "origin": ORIGIN,
            "evidence_ref": evidence_ref("T"), "value": None, "expression": None,
            "unknown_reason": None,
        }
        cases.append(("未知轴缺原因", missing_reason, ("unknown_reason",)))
        for label, payload, needles in cases:
            with self.subTest(case=label):
                self.assertValueErrorLocated(
                    lambda p=payload: gir.Tensor.from_dict(p), *needles
                )

    def test_from_dict_revalidates_layout_and_rank(self):
        bad_layout = make_tensor().to_dict()
        bad_layout["physical_layout"] = ["B", "T", "N_hc"]
        self.assertValueErrorLocated(
            lambda: gir.Tensor.from_dict(bad_layout), "physical_layout"
        )
        bad_rank = make_tensor().to_dict()
        bad_rank["global_rank"] = -1
        self.assertValueErrorLocated(lambda: gir.Tensor.from_dict(bad_rank), "global_rank")
        dup_axis = make_tensor().to_dict()
        dup_axis["axes"][1] = dict(dup_axis["axes"][0])
        self.assertValueErrorLocated(lambda: gir.Tensor.from_dict(dup_axis), "axis:B")

    def test_from_dict_extra_keys_ignored(self):
        """边界观察：未知多余键被忽略（前向兼容），不影响往返相等。"""
        payload = make_tensor().to_dict()
        payload["shape"] = [1, 128, 4, 4096]
        payload["unexpected_field"] = "ignored?"
        self.assertEqual(gir.Tensor.from_dict(payload), make_tensor())
        self.assertEqual(gir.Tensor.from_dict(payload).to_dict(), make_tensor().to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
