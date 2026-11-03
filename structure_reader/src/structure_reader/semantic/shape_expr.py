"""有限、可判定的形状表达式：解析、化简、求值与确定性比较。

T12 的形状表达式化简只处理**有限白名单**的整数形状代数，不做任意符号证明：

- 语法：整数常量、变量名、``+ - * // %``、``min(...)`` / ``max(...)``
  （两参或多参——多参被折叠成嵌套二参调用）、括号。拒绝属性访问、函数调用
  以外的节点、字符串/浮点、幂运算与其它语法（``ShapeExprError`` 具名拒绝）。
- 化简：常量折叠、``x+0``/``x*1``/``x*0``/``x-0``/``x-x``/``x//1``、
  ``min(x,x)``/``max(x,x)``、``min``/``max`` 常量折叠；以及**带前提**的
  ``a * (b // a) == b``——仅当调用方显式给出 ``divisible(b, a)`` 事实时成立
  （否则保持原表达式，例如 5 // 2 不可反推）。
- 求值：只在小整数环境上求值（测试与场景校验用），缺变量/除零具名拒绝。
- 不把未知规模变成示例常量：无 ``value`` 的变量保持符号。

本模块不执行模型代码、不读权重、不依赖 numpy/torch。
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping

Expr = tuple
"""表达式节点：``('int', n)`` / ``('var', name)`` / ``(op, left, right)``。"""


class ShapeExprError(ValueError):
    """形状表达式非法、求值失败或事实不足（具名拒绝）。"""


_ALLOWED_BINOPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
}
_ALLOWED_CALLS = {"min", "max"}
_PRECEDENCE = {
    "add": 1,
    "sub": 1,
    "mul": 2,
    "floordiv": 2,
    "mod": 2,
    "min": 3,
    "max": 3,
    "int": 4,
    "var": 4,
}


def _convert(node: ast.AST) -> Expr:
    if isinstance(node, ast.Expression):
        return _convert(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ShapeExprError("只允许整数常量")
        return ("int", node.value)
    if isinstance(node, ast.Name):
        if not node.id.isidentifier():
            raise ShapeExprError(f"非法变量名 {node.id!r}")
        return ("var", node.id)
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ShapeExprError(f"不允许的运算符 {type(node.op).__name__}")
        return (op, _convert(node.left), _convert(node.right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
            raise ShapeExprError("只允许 min(...) / max(...) 调用")
        if node.keywords:
            raise ShapeExprError("min/max 不接受关键字参数")
        if len(node.args) < 1:
            raise ShapeExprError("min/max 至少一个参数")
        result = _convert(node.args[0])
        for argument in node.args[1:]:
            result = (node.func.id, result, _convert(argument))
        return result
    raise ShapeExprError(f"不允许的表达式节点 {type(node).__name__}")


def parse(text: str) -> Expr:
    """解析单行形状表达式；非法语法具名拒绝（不回退到 eval）。"""
    if not isinstance(text, str) or not text.strip():
        raise ShapeExprError("表达式必须是非空字符串")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ShapeExprError(f"表达式语法错误: {text!r}") from error
    return _convert(tree)


def is_constant(expr: Expr) -> bool:
    if expr[0] == "int":
        return True
    if expr[0] in ("var",):
        return False
    if expr[0] in ("min", "max"):
        return all(is_constant(item) for item in expr[1:])
    return is_constant(expr[1]) and is_constant(expr[2])


def free_symbols(expr: Expr) -> tuple[str, ...]:
    """按首次出现顺序返回自由变量名（去重、确定性）。"""
    seen: list[str] = []

    def walk(node: Expr) -> None:
        if node[0] == "var":
            if node[1] not in seen:
                seen.append(node[1])
            return
        if node[0] == "int":
            return
        for child in node[1:]:
            walk(child)

    walk(expr)
    return tuple(seen)


def _wrap(child: Expr, parent_op: str, side: str) -> str:
    text = to_text(child)
    child_precedence = _PRECEDENCE.get(child[0], 4)
    parent_precedence = _PRECEDENCE[parent_op]
    if child_precedence < parent_precedence:
        return f"({text})"
    if child_precedence == parent_precedence and side == "right":
        if parent_op in ("sub", "floordiv", "mod"):
            return f"({text})"
        if parent_op == "mul" and child[0] in ("floordiv", "mod"):
            return f"({text})"
    return text


def to_text(expr: Expr) -> str:
    kind = expr[0]
    if kind == "int":
        return str(expr[1])
    if kind == "var":
        return expr[1]
    if kind in ("min", "max"):
        return f"{kind}({', '.join(to_text(item) for item in expr[1:])})"
    left, right = expr[1], expr[2]
    symbol = {"add": "+", "sub": "-", "mul": "*", "floordiv": "//", "mod": "%"}[kind]
    return f"{_wrap(left, kind, 'left')} {symbol} {_wrap(right, kind, 'right')}"


def _const(expr: Expr) -> int | None:
    return expr[1] if expr[0] == "int" else None


def _simplify_pair(op: str, left: Expr, right: Expr, facts: frozenset[tuple]) -> Expr:
    left_const, right_const = _const(left), _const(right)
    if op == "add":
        if right_const == 0:
            return left
        if left_const == 0:
            return right
        if left_const is not None and right_const is not None:
            return ("int", left_const + right_const)
        return (op, left, right)
    if op == "sub":
        if right_const == 0:
            return left
        if left == right:
            return ("int", 0)
        if left_const is not None and right_const is not None:
            return ("int", left_const - right_const)
        return (op, left, right)
    if op == "mul":
        if left_const == 0 or right_const == 0:
            return ("int", 0)
        if right_const == 1:
            return left
        if left_const == 1:
            return right
        if left_const is not None and right_const is not None:
            return ("int", left_const * right_const)
        # 带前提的单位化：a * (b // a) == b 仅当 divisible(b, a) 已知。
        for factor, quotient in ((left, right), (right, left)):
            if quotient[0] == "floordiv" and quotient[2] == factor:
                if ("divisible", to_text(quotient[1]), to_text(factor)) in facts:
                    return quotient[1]
        return (op, left, right)
    if op == "floordiv":
        if right_const == 0:
            raise ShapeExprError("形状表达式除零")
        if right_const == 1:
            return left
        if left_const == 0:
            return ("int", 0)
        if left_const is not None and right_const is not None:
            return ("int", left_const // right_const)
        if left == right:
            return ("int", 1)
        return (op, left, right)
    if op == "mod":
        if right_const == 0:
            raise ShapeExprError("形状表达式取模零")
        if right_const == 1:
            return ("int", 0)
        if left_const is not None and right_const is not None:
            return ("int", left_const % right_const)
        return (op, left, right)
    raise ShapeExprError(f"未知运算符 {op!r}")


def simplify(expr: Expr, facts: Iterable[tuple] = ()) -> Expr:
    """按白名单代数化简；``facts`` 为可判定前提（如 ``("divisible", b, a)``）。"""
    fact_set = frozenset(facts)
    kind = expr[0]
    if kind in ("int", "var"):
        return expr
    if kind in ("min", "max"):
        items = [simplify(item, fact_set) for item in expr[1:]]
        if len(items) == 1:
            return items[0]
        if all(_const(item) is not None for item in items):
            values = [_const(item) for item in items]
            return ("int", min(values) if kind == "min" else max(values))
        if all(item == items[0] for item in items):
            return items[0]
        return (kind, *items)
    left = simplify(expr[1], fact_set)
    right = simplify(expr[2], fact_set)
    return _simplify_pair(kind, left, right, fact_set)


def evaluate(expr: Expr, env: Mapping[str, int]) -> int:
    """在整数环境上求值；缺变量或除零具名拒绝。"""
    kind = expr[0]
    if kind == "int":
        return expr[1]
    if kind == "var":
        if expr[1] not in env:
            raise ShapeExprError(f"缺少变量 {expr[1]!r} 的取值")
        value = env[expr[1]]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ShapeExprError(f"变量 {expr[1]!r} 必须是整数")
        return value
    if kind in ("min", "max"):
        values = [evaluate(item, env) for item in expr[1:]]
        return min(values) if kind == "min" else max(values)
    left, right = evaluate(expr[1], env), evaluate(expr[2], env)
    if kind == "add":
        return left + right
    if kind == "sub":
        return left - right
    if kind == "mul":
        return left * right
    if kind == "floordiv":
        if right == 0:
            raise ShapeExprError("求值时除零")
        return left // right
    if kind == "mod":
        if right == 0:
            raise ShapeExprError("求值时取模零")
        return left % right
    raise ShapeExprError(f"未知运算符 {kind!r}")


def equivalent_on_grid(
    left: Expr,
    right: Expr,
    envs: Iterable[Mapping[str, int]],
) -> bool:
    """在给定的确定性小整数环境集合上逐点比较两个表达式。"""
    return all(evaluate(left, env) == evaluate(right, env) for env in envs)
