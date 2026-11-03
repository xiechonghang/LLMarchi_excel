"""确定性索引映射探针：用小型确定性张量索引核对布局改写是否保持值/顺序。

不依赖 numpy/torch、不读权重：把轴符号映射为小整数维度（单位轴=1，其余=2，
元素总数上限 1024），按行主序模拟 View/Reshape/Transpose 链，得到“输出每个
位置来自输入的哪个线性下标”的元组；再与替换后的单次 Reshape（恒等映射）比较。
一致才算证明成立；不能证明时规则必须拒绝。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from structure_reader.graph_ir import NamedAxis

MAX_PROBE_ELEMENTS = 1024


class ProbeError(ValueError):
    """探针无法构造或元素数不一致（调用方应拒绝改写）。"""


def extents_for(
    symbols: Iterable[str],
    axes_by_symbol: Mapping[str, NamedAxis],
) -> dict[str, int]:
    extents: dict[str, int] = {}
    for symbol in symbols:
        axis = axes_by_symbol.get(symbol)
        extents[symbol] = 1 if axis is not None and axis.value == 1 else 2
    return extents


def _prod(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _shape(symbols: Sequence[str], extents: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(extents[symbol] for symbol in symbols)


def _transpose_flat(
    flat: list[int],
    in_axes: Sequence[str],
    out_axes: Sequence[str],
    extents: Mapping[str, int],
) -> list[int]:
    if sorted(in_axes) != sorted(out_axes):
        raise ProbeError("转置前后轴集合不同")
    in_shape = _shape(in_axes, extents)
    out_shape = _shape(out_axes, extents)
    if _prod(in_shape) != len(flat):
        raise ProbeError("转置输入元素数与张量不一致")
    strides: dict[str, int] = {}
    stride = 1
    for axis, dim in zip(reversed(in_axes), reversed(in_shape)):
        strides[axis] = stride
        stride *= dim
    result: list[int] = []
    for linear in range(_prod(out_shape)):
        remainder = linear
        multi: dict[str, int] = {}
        for axis, dim in zip(reversed(out_axes), reversed(out_shape)):
            multi[axis] = remainder % dim
            remainder //= dim
        source = sum(multi[axis] * strides[axis] for axis in in_axes)
        result.append(flat[source])
    return result


def simulate(
    input_axes: Sequence[str],
    steps: Sequence[tuple[str, Sequence[str]]],
    extents: Mapping[str, int],
) -> tuple[int, ...]:
    """按行主序模拟布局链；返回输出位置→输入线性下标。"""
    current = tuple(input_axes)
    try:
        total = _prod(_shape(current, extents))
    except KeyError as error:
        raise ProbeError(f"缺少轴 {error} 的维度") from error
    if total > MAX_PROBE_ELEMENTS:
        raise ProbeError(f"探针元素数 {total} 超过上限 {MAX_PROBE_ELEMENTS}")
    flat = list(range(total))
    for kind, out_axes in steps:
        out_axes = tuple(out_axes)
        if kind in ("view", "reshape"):
            if _prod(_shape(current, extents)) != _prod(_shape(out_axes, extents)):
                raise ProbeError("Reshape 前后元素数不同")
        elif kind == "transpose":
            flat = _transpose_flat(flat, current, out_axes, extents)
        else:
            raise ProbeError(f"探针不支持的布局操作 {kind!r}")
        current = out_axes
    return tuple(flat)


def identity_chain(
    input_axes: Sequence[str],
    steps: Sequence[tuple[str, Sequence[str]]],
    extents: Mapping[str, int],
) -> tuple[bool, str]:
    """布局链是否等价于恒等（即一次同序 Reshape）。返回 (是否成立, 说明)。"""
    try:
        mapping = simulate(input_axes, steps, extents)
    except ProbeError as error:
        return False, f"探针无法构造：{error}"
    expected = tuple(range(len(mapping)))
    if mapping == expected:
        size = len(mapping)
        return True, f"index-map 恒等（{size} 个位置，首项 {mapping[:4]}）"
    return False, f"index-map 非恒等（首项 {mapping[:6]} ≠ {expected[:6]}）"


def last_two_swap_chain(
    input_axes: Sequence[str],
    output_axes: Sequence[str],
    extents: Mapping[str, int],
) -> tuple[bool, str]:
    """Transpose 是否恰好只交换最后两个矩阵轴（按索引映射核对）。"""
    if len(input_axes) < 2 or len(output_axes) != len(input_axes):
        return False, "秩不足或转置前后秩不同"
    expected_axes = tuple(input_axes[:-2]) + (input_axes[-1], input_axes[-2])
    if tuple(output_axes) != expected_axes:
        return False, f"轴顺序不是最后两轴交换：{tuple(output_axes)} ≠ {expected_axes}"
    try:
        mapping = simulate(input_axes, [("transpose", output_axes)], extents)
    except ProbeError as error:
        return False, f"探针无法构造：{error}"
    return True, f"index-map 最后两轴交换（{len(mapping)} 个位置）"
