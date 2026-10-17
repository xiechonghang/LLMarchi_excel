"""有限表达式求值器（垂直切片版）——白名单 AST，整数/布尔，三态结果。

设计约束：
- 只接受白名单语法：整数常量、变量名、+ - * // % 一元正负、
  min/max 调用、比较（== != < <= > >=）、and/or 组合。其余节点一律具名拒绝。
- 不 eval、不 exec、不访问属性、不索引、不调用白名单外函数；
  非法表达式在解析/校验期具名拒绝，不存在任何动态执行路径。
- 深度/节点数上限按真实 AST 嵌套计（括号不构成 AST 深度，也不设
  文本长度/括号数上限），防御超深表达式。
- 环境（env）契约：值必须是整数且不得是 bool，绝对值 ≤ 2^63-1；布尔、
  字符串、浮点、任意对象一律在入口具名拒绝——任意对象运算与模型代码
  执行均不可能发生。env=None 等价空环境。
- 整数量级：字面量、env 值与中间结果都限制在 ±(2^63-1)（形状计算远小于
  此），超限具名拒绝，不用任意精度大整数静默继续。
- 类型语义：bool 与 int 严格区分。布尔只能作为比较结果、and/or 操作数或
  表达式最终结果；算术、比较、min/max 参数中出现布尔即具名拒绝。
- min/max 至少需要一个整数参数；单参数返回该参数本身。
- 整数除法用 FloorDiv（形状计算语义）；除零在求值期具名报错。
- 三态：determined(value) / unknown(reason)——unknown 不被当作 False。
  and/or 短路求值：已确定的 False/True 不被后续 unknown 掩盖
  （`假 and <除零>` == False）；变量缺失在预检期保守返回 unknown，
  不因短路被吞掉（见 evaluate_tri）。

本模块不求值任何非白名单内容，不导入模型源码，不执行模型代码。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "ExpressionError",
    "TriState",
    "evaluate_int",
    "evaluate_bool",
    "evaluate_tri",
    "check_expression",
]

_MAX_DEPTH = 24
_MAX_NODES = 96
_MAX_ABS = (1 << 63) - 1  # 字面量/env 值/中间结果的量级上限（int64）
_ALLOWED_FUNCS = ("min", "max")
_ALLOWED_CMPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

_FUNC_IMPL = {
    "min": min,
    "max": max,
}


class ExpressionError(ValueError):
    """表达式不在白名单内、环境/结果类型不符或求值失败（具名拒绝）。"""


@dataclass(frozen=True)
class TriState:
    """一次三态求值结果：state 限 determined|unknown。"""

    state: str
    value: Any = None
    unknown_reason: str = ""

    @property
    def is_known(self) -> bool:
        return self.state == "determined"


def _reject_bool(what: str) -> ExpressionError:
    return ExpressionError(f"{what}不接受布尔值；bool 与 int 严格区分")


def _walk(node: ast.AST, depth: int) -> int:
    """白名单校验 + 返回节点计数；非法节点具名拒绝。"""
    if depth > _MAX_DEPTH:
        raise ExpressionError(f"表达式嵌套超过上限 {_MAX_DEPTH}")
    count = 1
    if isinstance(node, ast.Expression):
        count += _walk(node.body, depth + 1)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ExpressionError(
                f"只允许整数常量，得到 {node.value!r}（{type(node.value).__name__}）"
            )
        if abs(node.value) > _MAX_ABS:
            raise ExpressionError("整数字面量超出量级上限 ±(2^63-1)")
    elif isinstance(node, ast.Name):
        if not node.id or not node.id.isidentifier():
            raise ExpressionError(f"非法变量名 {node.id!r}")
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            raise ExpressionError(
                f"二元运算只允许 + - * // %，得到 {type(node.op).__name__}"
            )
        count += _walk(node.left, depth + 1)
        count += _walk(node.right, depth + 1)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise ExpressionError(f"一元运算只允许 +/-，得到 {type(node.op).__name__}")
        count += _walk(node.operand, depth + 1)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ExpressionError(
                f"只允许调用 {'/'.join(_ALLOWED_FUNCS)}，得到 "
                f"{getattr(node.func, 'id', type(node.func).__name__)!r}"
            )
        if node.keywords:
            raise ExpressionError("不允许关键字参数")
        if not node.args:
            raise ExpressionError(f"{node.func.id} 至少需要一个参数")
        for arg in node.args:
            count += _walk(arg, depth + 1)
    elif isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ExpressionError("只允许单次比较（不允许 a < b < c 链）")
        if not isinstance(node.ops[0], tuple(_ALLOWED_CMPS)):
            raise ExpressionError(
                f"比较只允许 == != < <= > >=，得到 {type(node.ops[0]).__name__}"
            )
        count += _walk(node.left, depth + 1)
        count += _walk(node.comparators[0], depth + 1)
    elif isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ExpressionError(f"布尔组合只允许 and/or，得到 {type(node.op).__name__}")
        for value in node.values:
            if not isinstance(value, (ast.Compare, ast.BoolOp)):
                # 静态约束：and/or 操作数必须是布尔产生式（比较或布尔组合）。
                # env 不接受布尔、常量布尔也被拒，所以布尔只可能来自比较；
                # 静态拒绝保证与短路求值顺序无关。
                raise ExpressionError(
                    "and/or 操作数必须是比较或布尔组合，"
                    f"得到 {type(value).__name__}"
                )
            count += _walk(value, depth + 1)
    else:
        raise ExpressionError(
            f"表达式包含白名单外的语法节点: {type(node).__name__}"
        )
    if count > _MAX_NODES:
        raise ExpressionError(f"表达式节点数超过上限 {_MAX_NODES}")
    return count


def _parse(expr: str) -> ast.Expression:
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("表达式必须是非空字符串")
    if "\n" in expr:
        raise ExpressionError("表达式必须是单行")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"表达式语法错误: {expr!r}: {exc}") from exc
    _walk(tree, 0)
    return tree


def _validate_env(env: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """环境契约：映射、合法变量名、非 bool 整数、量级受限；违规具名拒绝。

    环境类型错误是调用方程序错误，抛 :class:`ExpressionError` 而不是
    unknown——unknown 保留给数据不可知（变量缺失/除零）。
    """
    if env is None:
        return {}
    if not isinstance(env, Mapping):
        raise ExpressionError(f"env 必须是映射，得到 {type(env).__name__}")
    for key, value in env.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ExpressionError(f"env 变量名非法: {key!r}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ExpressionError(
                f"env 变量 {key} 必须是整数，得到 {type(value).__name__}"
                "（bool 与 int 严格区分，不接受任意对象）"
            )
        if abs(value) > _MAX_ABS:
            raise ExpressionError(f"env 变量 {key} 超出量级上限 ±(2^63-1)")
    return env


def _eval(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise KeyError(node.id)
        return env[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, env)
        right = _eval(node.right, env)
        if isinstance(left, bool) or isinstance(right, bool):
            raise _reject_bool("算术运算")
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        elif isinstance(node.op, ast.FloorDiv):
            value = left // right
        else:
            value = left % right
        if abs(value) > _MAX_ABS:
            raise ExpressionError("中间结果超出量级上限 ±(2^63-1)")
        return value
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, env)
        if isinstance(value, bool):
            raise _reject_bool("一元运算")
        value = -value if isinstance(node.op, ast.USub) else +value
        if abs(value) > _MAX_ABS:
            raise ExpressionError("中间结果超出量级上限 ±(2^63-1)")
        return value
    if isinstance(node, ast.Call):
        args = [_eval(arg, env) for arg in node.args]
        if any(isinstance(arg, bool) for arg in args):
            raise _reject_bool(f"{node.func.id} 参数")
        # args 是非空整数序列（白名单静态保证 ≥1 参数）。内置 min/max 单
        # 参数按可迭代对象解释，必须传序列而不是 *args 展开——单参数返回
        # 该参数本身。
        return _FUNC_IMPL[node.func.id](args)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        right = _eval(node.comparators[0], env)
        if isinstance(left, bool) or isinstance(right, bool):
            raise _reject_bool("比较")
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        return left >= right
    if isinstance(node, ast.BoolOp):
        # 短路求值：and 遇 False / or 遇 True 立即返回，与被求值表达式的
        # 源码语义一致；后续操作数里的除零等 unknown 不会掩盖已确定的值。
        is_and = isinstance(node.op, ast.And)
        for value_node in node.values:
            value = _eval(value_node, env)
            if not isinstance(value, bool):
                raise ExpressionError(
                    "and/or 操作数必须是布尔结果（比较或布尔组合），"
                    f"得到 {type(value).__name__}"
                )
            if is_and and not value:
                return False
            if not is_and and value:
                return True
        return is_and
    raise ExpressionError(f"求值遇到未支持的节点: {type(node).__name__}")  # pragma: no cover


def check_expression(expr: str) -> None:
    """仅校验白名单合法性，不求值；非法抛 :class:`ExpressionError`。"""
    _parse(expr)


def _env_missing(expr: str, env: Mapping[str, Any]) -> str | None:
    tree = _parse(expr)
    needed = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
              and not isinstance(n.ctx, ast.Store)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in _ALLOWED_FUNCS:
            needed.discard(n.id)
    missing = sorted(needed - set(env))
    if missing:
        return f"缺少变量: {', '.join(missing)}"
    return None


def evaluate_int(expr: str, env: Mapping[str, Any] | None = None) -> int:
    """求值整数表达式。

    env=None 等价空环境。变量缺失/除零/结果非整数抛
    :class:`ExpressionError`；环境类型或量级错误在入口即具名拒绝。
    """
    checked = _validate_env(env)
    tree = _parse(expr)
    try:
        value = _eval(tree.body, checked)
    except KeyError as exc:
        raise ExpressionError(f"缺少变量: {exc.args[0]}") from exc
    except ZeroDivisionError as exc:
        raise ExpressionError(f"除零: {expr!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExpressionError(
            f"表达式 {expr!r} 结果不是整数: {value!r}（{type(value).__name__}）"
        )
    return value


def evaluate_bool(expr: str, env: Mapping[str, Any] | None = None) -> bool:
    """求值布尔表达式（守卫判断）。

    env=None 等价空环境。变量缺失/除零/结果非布尔抛
    :class:`ExpressionError`；环境类型或量级错误在入口即具名拒绝。
    """
    checked = _validate_env(env)
    tree = _parse(expr)
    try:
        value = _eval(tree.body, checked)
    except KeyError as exc:
        raise ExpressionError(f"缺少变量: {exc.args[0]}") from exc
    except ZeroDivisionError as exc:
        raise ExpressionError(f"除零: {expr!r}") from exc
    if not isinstance(value, bool):
        raise ExpressionError(f"表达式 {expr!r} 结果不是布尔值: {value!r}")
    return value


def evaluate_tri(expr: str, env: Mapping[str, Any] | None = None) -> TriState:
    """三态求值：变量缺失或除零 → unknown（带原因），绝不静默当作 False。

    API 契约：
    - env 可省略（None = 空环境）；表达式引用缺失变量 → unknown。
    - 表达式非法（语法错误或白名单外）→ unknown，原因含"表达式非法"。
    - 环境类型/量级错误不是 unknown：直接抛 :class:`ExpressionError`
      （调用方程序错误必须显式修复，不能掩盖成数据不可知）。
    - and/or 短路：已确定的 False/True 不被后续除零 unknown 掩盖。
    """
    checked = _validate_env(env)
    try:
        tree = _parse(expr)
    except ExpressionError as exc:
        return TriState(state="unknown", unknown_reason=f"表达式非法: {exc}")
    missing = _env_missing(expr, checked)
    if missing:
        return TriState(state="unknown", unknown_reason=missing)
    try:
        value = _eval(tree.body, checked)
    except ZeroDivisionError:
        return TriState(state="unknown", unknown_reason="求值遇到除零")
    except ExpressionError as exc:
        return TriState(state="unknown", unknown_reason=str(exc))
    if isinstance(value, bool):
        return TriState(state="determined", value=value)
    if isinstance(value, int):
        return TriState(state="determined", value=value)
    return TriState(
        state="unknown",
        unknown_reason=f"结果类型不支持: {type(value).__name__}",
    )
