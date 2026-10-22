"""S1 01a Tensor 形状修正独立验证：logical_shape / physical_shape / shape 别名。

范围严格限定于 `results/development_01a_tensor_shape_fix.md` 声明的修复增量
（差距报告建议 A + C + D）：
- `logical_shape` 按 **axes 顺序**给出确定值；
- `physical_shape` 按 **physical_layout 符号顺序**重排确定 value（排列式布局，覆盖转置）；
- `shape` 是 `logical_shape` 的**别名**（既有调用与测试不破坏）；
- 任一轴 `value is None`（未知轴或仅 expression 轴）→ 两者均 `None`，不求值表达式；
- `value=0` 是确定值、参与形状；空 axes → 两形状均为 `()`；
- **序列化字节契约不变**：`to_dict` 仍恰 8 键、无任何 shape 键；canonical 字节与
  修复前一致（此处用手工推导的 golden 字面量固定，不复制产品输出）。

**仍明确未实现（本轮只固定现状、不作为缺陷）**：`world_size>1` 时全局/本 rank 轴长度不可表达
（无 `world_size`、无 per-rank 长度字段）；`global_rank`/`local_rank` 只是张量秩标注，
不与 `len(axes)` 交叉校验；首版 `world_size=1` 分析假设**尚待图/scenario 批次记录**；
分片/融合/重复维度的物理布局超出排列式布局契约。

不在本范围：Node/Edge/SemanticGraph、图级序列化、01b 检查器、01c schema、表达式求值器、
模板/实例绑定、guard 三态、state、alias、fold、真实 DeepSeek 结构提取。

纪律：数据为**合成契约样例（synthetic contract fixture）**，非 DeepSeek 结构；
只用标准库；期望值来自派工/开发报告与手工推导的 canonical 规则，不复制产品输出为 golden。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import structure_reader.graph_ir as gir  # noqa: E402

ORIGIN = "synthetic_contract_fixture(非DeepSeek结构)"
CANONICAL_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}
TO_DICT_KEYS = {
    "stable_id", "name", "dtype", "axes",
    "logical_layout", "physical_layout", "global_rank", "local_rank",
}
SHAPE_KEYS = {"shape", "logical_shape", "physical_shape"}


def axis(symbol, name, meaning, value):
    return gir.NamedAxis(symbol=symbol, name=name, meaning=meaning, origin=ORIGIN,
                         evidence_ref=f"fixture:shape/{symbol}", value=value)


def unknown_axis(symbol, name, meaning, reason="该轴数值未在来源中给出（合成契约样例）"):
    return gir.NamedAxis(symbol=symbol, name=name, meaning=meaning, origin=ORIGIN,
                         evidence_ref=f"fixture:shape/{symbol}", value=None,
                         expression=None, unknown_reason=reason)


def expr_axis(symbol, name, meaning, expr="seq_len * 2"):
    return gir.NamedAxis(symbol=symbol, name=name, meaning=meaning, origin=ORIGIN,
                         evidence_ref=f"fixture:shape/{symbol}", value=None, expression=expr)


def axes_bth():
    """3 轴合成样例：B/T/H（通用结构名词，非 DeepSeek 语义）。"""
    return [axis("B", "batch_size", "批大小（合成契约样例）", 1),
            axis("T", "seq_len", "序列长度（合成契约样例）", 4),
            axis("H", "hidden_size", "隐藏维度（合成契约样例）", 4096)]


def transpose_tensor():
    """转置样例：logical=(B,T,H)，physical=(H,T,B)。"""
    return gir.Tensor(name="kv", dtype="bfloat16", axes=axes_bth(),
                      logical_layout=("B", "T", "H"), physical_layout=("H", "T", "B"))


def canonical(tensor):
    """仅用于稳定性比较：对 to_dict() 做确定性 JSON 编码（非产品 API）。"""
    return json.dumps(tensor.to_dict(), **CANONICAL_KWARGS)


# 手工推导的 canonical golden（keys 排序 + 紧凑分隔符 + ensure_ascii=False）。
# 与修复前快照的字节契约对照用；推导自 fixtures 值，不取自产品输出。
EXPECTED_TRANSPOSE_CANONICAL = (
    '{"axes":['
    '{"evidence_ref":"fixture:shape/B","expression":null,"meaning":"批大小（合成契约样例）",'
    f'"name":"batch_size","origin":"{ORIGIN}","stable_id":"axis:B","symbol":"B",'
    '"unknown_reason":null,"value":1},'
    '{"evidence_ref":"fixture:shape/T","expression":null,"meaning":"序列长度（合成契约样例）",'
    f'"name":"seq_len","origin":"{ORIGIN}","stable_id":"axis:T","symbol":"T",'
    '"unknown_reason":null,"value":4},'
    '{"evidence_ref":"fixture:shape/H","expression":null,"meaning":"隐藏维度（合成契约样例）",'
    f'"name":"hidden_size","origin":"{ORIGIN}","stable_id":"axis:H","symbol":"H",'
    '"unknown_reason":null,"value":4096}'
    '],"dtype":"bfloat16","global_rank":null,"local_rank":null,'
    '"logical_layout":["B","T","H"],"name":"kv","physical_layout":["H","T","B"],'
    '"stable_id":"tensor:kv"}'
)


class ShapeTestCase(unittest.TestCase):
    """公共断言辅助。"""

    def assert_shapes(self, tensor, logical, physical):
        self.assertEqual(tensor.logical_shape, logical)
        self.assertEqual(tensor.physical_shape, physical)
        self.assertEqual(tensor.shape, logical)
        self.assertEqual(tensor.shape, tensor.logical_shape)


# ---------------------------------------------------------------------------
# 1. 接口快照：属性形态与别名
# ---------------------------------------------------------------------------
class TensorShapeInterfaceTest(ShapeTestCase):
    def test_shapes_are_readonly_properties(self):
        for attribute in ("shape", "logical_shape", "physical_shape"):
            with self.subTest(attribute=attribute):
                self.assertIsInstance(
                    inspect_getattr_static(gir.Tensor, attribute), property
                )

    def test_shape_is_alias_of_logical_shape(self):
        tensor = transpose_tensor()
        self.assertEqual(tensor.shape, tensor.logical_shape)
        self.assertIsNotNone(tensor.shape)

    def test_field_set_unchanged_by_shape_fix(self):
        self.assertEqual(
            tuple(f.name for f in fields(gir.Tensor)),
            ("name", "dtype", "axes", "logical_layout", "physical_layout",
             "global_rank", "local_rank"),
        )

    def test_shape_properties_do_not_mutate_serialization(self):
        tensor = transpose_tensor()
        before = canonical(tensor)
        _ = tensor.logical_shape, tensor.physical_shape, tensor.shape
        self.assertEqual(canonical(tensor), before)

    def test_shape_values_are_plain_int_tuples(self):
        tensor = transpose_tensor()
        for name in ("logical_shape", "physical_shape", "shape"):
            with self.subTest(shape=name):
                value = getattr(tensor, name)
                self.assertIsInstance(value, tuple)
                self.assertTrue(all(type(item) is int for item in value))


def inspect_getattr_static(cls, name):
    import inspect

    return inspect.getattr_static(cls, name, None)


# ---------------------------------------------------------------------------
# 2. logical_shape：按 axes 顺序
# ---------------------------------------------------------------------------
class LogicalShapeTest(ShapeTestCase):
    def test_logical_shape_follows_axes_order(self):
        axes = [axis("H", "hidden_size", "隐藏维度（合成契约样例）", 4096),
                axis("B", "batch_size", "批大小（合成契约样例）", 1),
                axis("T", "seq_len", "序列长度（合成契约样例）", 4)]
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes,
                            logical_layout=("H", "B", "T"),
                            physical_layout=("H", "B", "T"))
        self.assertEqual([a.symbol for a in tensor.axes], ["H", "B", "T"])
        self.assert_shapes(tensor, (4096, 1, 4), (4096, 1, 4))

    def test_logical_shape_ignores_physical_layout(self):
        base = transpose_tensor()
        other = gir.Tensor(name="kv", dtype="bfloat16", axes=axes_bth(),
                           logical_layout=("B", "T", "H"),
                           physical_layout=("T", "B", "H"))
        self.assertEqual(base.logical_shape, other.logical_shape)
        self.assertEqual(base.logical_shape, (1, 4, 4096))
        self.assertNotEqual(base.physical_shape, other.physical_shape)

    def test_reversed_axes_reverse_logical_shape(self):
        axes = axes_bth()
        forward = gir.Tensor(name="t", dtype="float32", axes=axes,
                             logical_layout=("B", "T", "H"),
                             physical_layout=("B", "T", "H"))
        backward = gir.Tensor(name="t", dtype="float32", axes=list(reversed(axes)),
                              logical_layout=("H", "T", "B"),
                              physical_layout=("H", "T", "B"))
        self.assertEqual(forward.logical_shape, (1, 4, 4096))
        self.assertEqual(backward.logical_shape, (4096, 4, 1))

    def test_single_axis_tensor(self):
        tensor = gir.Tensor(name="v", dtype="float32",
                            axes=[axis("H", "hidden_size", "隐藏维度（合成契约样例）", 4096)],
                            logical_layout=("H",), physical_layout=("H",))
        self.assert_shapes(tensor, (4096,), (4096,))

    def test_shape_alias_matches_for_multiple_fixtures(self):
        fixtures = [
            transpose_tensor(),
            gir.Tensor(name="t", dtype="float32", axes=axes_bth(),
                       logical_layout=("T", "H", "B"), physical_layout=("B", "H", "T")),
            gir.Tensor(name="s", dtype="float32", axes=[], logical_layout=[],
                       physical_layout=[]),
        ]
        for tensor in fixtures:
            with self.subTest(name=tensor.name, axes=len(tensor.axes)):
                self.assertEqual(tensor.shape, tensor.logical_shape)


# ---------------------------------------------------------------------------
# 3. physical_shape：按 physical_layout 重排
# ---------------------------------------------------------------------------
class PhysicalShapeTest(ShapeTestCase):
    def test_transpose_example(self):
        """差距报告 CE1 的转置例：logical (1,4,4096) / physical (4096,4,1)。"""
        self.assert_shapes(transpose_tensor(), (1, 4, 4096), (4096, 4, 1))

    def test_physical_order_follows_layout_symbol_by_symbol(self):
        tensor = transpose_tensor()
        by_symbol = {a.symbol: a.value for a in tensor.axes}
        self.assertEqual(tensor.physical_shape,
                         tuple(by_symbol[s] for s in tensor.physical_layout))

    def test_equal_layouts_give_equal_shapes(self):
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes_bth(),
                            logical_layout=("B", "T", "H"),
                            physical_layout=("B", "T", "H"))
        self.assert_shapes(tensor, (1, 4, 4096), (1, 4, 4096))
        self.assertEqual(tensor.logical_shape, tensor.physical_shape)

    def test_reversed_physical_layout(self):
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes_bth(),
                            logical_layout=("B", "T", "H"),
                            physical_layout=("H", "T", "B"))
        self.assertEqual(tensor.physical_shape, (4096, 4, 1))

    def test_physical_shape_changes_with_layout_only(self):
        axes = axes_bth()
        base = gir.Tensor(name="t", dtype="float32", axes=axes,
                          logical_layout=("B", "T", "H"),
                          physical_layout=("B", "T", "H"))
        moved = gir.Tensor(name="t", dtype="float32", axes=axes,
                           logical_layout=("B", "T", "H"),
                           physical_layout=("B", "H", "T"))
        self.assertEqual(base.physical_shape, (1, 4, 4096))
        self.assertEqual(moved.physical_shape, (1, 4096, 4))
        self.assertEqual(base.logical_shape, moved.logical_shape)

    def test_four_axis_plan_style_transpose(self):
        """plan v3 §4.2 第 232 行的 [B,T,N_hc,H] ↔ [B,H,T,N_hc] 形状重排。"""
        axes = [axis("B", "batch_size", "批大小（合成契约样例）", 1),
                axis("T", "seq_len", "序列长度（合成契约样例）", 128),
                axis("N_hc", "hc_streams", "HC 流数量（合成契约样例）", 4),
                axis("H", "hidden_size", "隐藏维度（合成契约样例）", 4096)]
        tensor = gir.Tensor(name="hidden_states", dtype="bfloat16", axes=axes,
                            logical_layout=("B", "T", "N_hc", "H"),
                            physical_layout=("B", "H", "T", "N_hc"))
        self.assert_shapes(tensor, (1, 128, 4, 4096), (1, 4096, 128, 4))

    def test_physical_shape_is_permutation_of_logical(self):
        tensor = transpose_tensor()
        self.assertEqual(sorted(tensor.physical_shape), sorted(tensor.logical_shape))
        self.assertNotEqual(tensor.physical_shape, tensor.logical_shape)


# ---------------------------------------------------------------------------
# 4. 未知轴 / 表达式轴的传播
# ---------------------------------------------------------------------------
class UnknownPropagationTest(ShapeTestCase):
    def test_unknown_axis_makes_both_shapes_none(self):
        axes = [axis("B", "batch_size", "批大小（合成契约样例）", 1),
                unknown_axis("H", "hidden_size", "隐藏维度（合成契约样例）")]
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes,
                            logical_layout=("B", "H"), physical_layout=("H", "B"))
        self.assert_shapes(tensor, None, None)
        self.assertIsNone(tensor.shape)

    def test_expression_axis_makes_both_shapes_none_without_evaluation(self):
        axes = [axis("B", "batch_size", "批大小（合成契约样例）", 1),
                expr_axis("T", "seq_len", "序列长度（合成契约样例）", expr="seq_len * 2")]
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes,
                            logical_layout=("B", "T"), physical_layout=("T", "B"))
        self.assertFalse(tensor.axes[1].is_unknown)   # 有表达式 ≠ 未知
        self.assert_shapes(tensor, None, None)
        self.assertIn('"expression":"seq_len * 2"', canonical(tensor))

    def test_unknown_wins_over_known_and_zero(self):
        axes = [axis("B", "batch_size", "批大小（合成契约样例）", 0),
                unknown_axis("H", "hidden_size", "隐藏维度（合成契约样例）")]
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes,
                            logical_layout=("B", "H"), physical_layout=("H", "B"))
        self.assertIsNone(tensor.logical_shape)
        self.assertIsNone(tensor.physical_shape)
        self.assertNotEqual(tensor.logical_shape, (0, 0))

    def test_all_known_gives_known_shapes(self):
        tensor = transpose_tensor()
        self.assertIsNotNone(tensor.logical_shape)
        self.assertIsNotNone(tensor.physical_shape)

    def test_none_ness_survives_roundtrip(self):
        axes = [unknown_axis("H", "hidden_size", "隐藏维度（合成契约样例）"),
                axis("B", "batch_size", "批大小（合成契约样例）", 1)]
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes,
                            logical_layout=("H", "B"), physical_layout=("B", "H"))
        back = gir.Tensor.from_dict(tensor.to_dict())
        self.assertEqual(back, tensor)
        self.assertIsNone(back.logical_shape)
        self.assertIsNone(back.physical_shape)

    def test_unknown_axis_reason_preserved_in_bytes(self):
        axes = [unknown_axis("H", "hidden_size", "隐藏维度（合成契约样例）",
                             reason="来源未给出具体值（合成契约样例）")]
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes,
                            logical_layout=("H",), physical_layout=("H",))
        text = canonical(tensor)
        self.assertIn('"value":null', text)
        self.assertIn("来源未给出具体值", text)


# ---------------------------------------------------------------------------
# 5. value=0 与空 axes
# ---------------------------------------------------------------------------
class ZeroAndEmptyAxesTest(ShapeTestCase):
    def test_zero_is_known_and_participates_in_shapes(self):
        axes = [axis("B", "batch_size", "批大小（合成契约样例）", 0),
                axis("H", "hidden_size", "隐藏维度（合成契约样例）", 4096)]
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes,
                            logical_layout=("B", "H"), physical_layout=("H", "B"))
        self.assert_shapes(tensor, (0, 4096), (4096, 0))
        self.assertIn(0, tensor.logical_shape)
        self.assertIn(0, tensor.physical_shape)

    def test_zero_in_reordered_physical_position(self):
        axes = [axis("B", "batch_size", "批大小（合成契约样例）", 0),
                axis("T", "seq_len", "序列长度（合成契约样例）", 4),
                axis("H", "hidden_size", "隐藏维度（合成契约样例）", 4096)]
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes,
                            logical_layout=("B", "T", "H"),
                            physical_layout=("H", "B", "T"))
        self.assertEqual(tensor.physical_shape, (4096, 0, 4))
        self.assertIsNotNone(tensor.physical_shape)  # 0 不等于未知

    def test_zero_does_not_mask_unknown(self):
        axes = [axis("B", "batch_size", "批大小（合成契约样例）", 0),
                unknown_axis("H", "hidden_size", "隐藏维度（合成契约样例）")]
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes,
                            logical_layout=("B", "H"), physical_layout=("H", "B"))
        self.assertIsNone(tensor.logical_shape)

    def test_empty_axes_give_empty_tuple_not_none(self):
        tensor = gir.Tensor(name="scalar", dtype="float32", axes=[],
                            logical_layout=[], physical_layout=[])
        self.assertEqual(tensor.logical_shape, ())
        self.assertEqual(tensor.physical_shape, ())
        self.assertEqual(tensor.shape, ())
        self.assertIsNotNone(tensor.logical_shape)

    def test_empty_axes_roundtrip(self):
        tensor = gir.Tensor(name="scalar", dtype="float32", axes=[],
                            logical_layout=[], physical_layout=[])
        back = gir.Tensor.from_dict(tensor.to_dict())
        self.assertEqual(back, tensor)
        self.assertEqual(back.logical_shape, ())
        self.assertEqual(back.physical_shape, ())


# ---------------------------------------------------------------------------
# 6. 序列化字节契约不变（8 键 / 无 shape 键 / golden 字面量 / 往返）
# ---------------------------------------------------------------------------
class SerializationUnchangedTest(ShapeTestCase):
    def test_to_dict_still_exactly_eight_keys_without_shapes(self):
        payload = transpose_tensor().to_dict()
        self.assertEqual(set(payload), TO_DICT_KEYS)
        self.assertEqual(set(payload) & SHAPE_KEYS, set())

    def test_canonical_bytes_match_pre_fix_literal_golden(self):
        """与修复前字节契约对照：golden 为手工推导字面量（非产品输出复制）。"""
        self.assertEqual(canonical(transpose_tensor()), EXPECTED_TRANSPOSE_CANONICAL)

    def test_repeated_serialization_byte_identical(self):
        tensor = transpose_tensor()
        self.assertEqual(canonical(tensor).encode("utf-8"),
                         canonical(tensor).encode("utf-8"))

    def test_canonical_bytes_deterministic_across_processes(self):
        tensor = transpose_tensor()
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

    def test_roundtrip_preserves_shapes(self):
        tensor = transpose_tensor()
        back = gir.Tensor.from_dict(tensor.to_dict())
        self.assertEqual(back, tensor)
        self.assert_shapes(back, (1, 4, 4096), (4096, 4, 1))
        self.assertEqual(canonical(back), EXPECTED_TRANSPOSE_CANONICAL)

    def test_from_dict_ignores_injected_shape_keys(self):
        """边界观察：shape 键不落盘；若被注入则被忽略（差距报告 §5A 的另一可选规则是
        “出现必须等于派生值，否则具名拒绝”，当前实现取忽略，本测试固定现状）。"""
        payload = transpose_tensor().to_dict()
        payload["logical_shape"] = [999, 999, 999]
        payload["physical_shape"] = [999, 999, 999]
        back = gir.Tensor.from_dict(payload)
        self.assertEqual(back, transpose_tensor())
        self.assertEqual(set(back.to_dict()), TO_DICT_KEYS)
        self.assertEqual(back.logical_shape, (1, 4, 4096))


# ---------------------------------------------------------------------------
# 7. 并行维度缺口：world_size>1 全局/本 rank 长度仍未实现（现状固定，非缺陷）
# ---------------------------------------------------------------------------
class ParallelDimensionGapTest(ShapeTestCase):
    def test_world_size_keyword_not_supported(self):
        """差距报告建议 B 未实施：Tensor 无 world_size 字段。"""
        with self.assertRaises(TypeError):
            gir.Tensor(name="t", dtype="float32", axes=axes_bth(),
                       logical_layout=("B", "T", "H"), physical_layout=("H", "T", "B"),
                       world_size=8)

    def test_named_axis_has_no_local_value_field(self):
        """per-rank 轴长度无处承载：NamedAxis 无 local_value 字段。"""
        self.assertNotIn("local_value", {f.name for f in fields(gir.NamedAxis)})
        with self.assertRaises(TypeError):
            gir.NamedAxis(symbol="H", name="hidden_size", meaning="隐藏维度（合成契约样例）",
                          origin=ORIGIN, evidence_ref="fixture:shape/H",
                          value=4096, local_value=512)

    def test_rank_fields_are_annotations_not_per_rank_lengths(self):
        """4 轴 + rank 99 合法：rank 不与 len(axes) 交叉校验，也不表示 per-rank 长度。"""
        tensor = gir.Tensor(name="t", dtype="float32", axes=axes_bth(),
                            logical_layout=("B", "T", "H"),
                            physical_layout=("H", "T", "B"),
                            global_rank=99, local_rank=0)
        self.assertEqual(tensor.global_rank, 99)
        self.assertEqual(tensor.local_rank, 0)
        self.assertEqual(len(tensor.axes), 3)
        # 注意：这里没有 512（4096/8）这类 per-rank 长度可言
        self.assertEqual(tensor.logical_shape, (1, 4, 4096))

    def test_docstring_documents_world_size_assumption_gap(self):
        doc = gir.Tensor.__doc__ or ""
        self.assertIn("world_size", doc)
        self.assertIn("world_size=1", doc)
        module_doc = gir.__doc__ or ""
        self.assertIn("world_size", module_doc)

    def test_sharded_physical_layout_still_rejected(self):
        """分片物理布局（重复/外来符号）仍超出排列式契约。"""
        for layout in (("B", "T", "H", "H"), ("B", "T", "H_local")):
            with self.subTest(layout=layout):
                with self.assertRaises(ValueError):
                    gir.Tensor(name="t", dtype="float32", axes=axes_bth(),
                               logical_layout=("B", "T", "H"), physical_layout=layout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
