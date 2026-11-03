"""具名轴目录：源码/config 全称、固定值/符号/派生表达式、来源与旧别名映射。

模型无关接口：:class:`AxisCatalog` 只要求调用方按全称登记轴；DeepSeek 适配器
声明自己的轴、来源与旧内部别名（``B``/``S``/``Dh`` 等）。语义图只允许引用目录
中的**全称** symbol；旧别名只用于旧事实映射与“不向主图泄露”的校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from structure_reader.graph_ir import NamedAxis


class AxisError(ValueError):
    """轴未登记、别名被当作全称或 axis 定义非法（具名拒绝）。"""


@dataclass(frozen=True)
class AxisDef:
    """一条轴的全称定义。

    value 固定常量；expression 派生表达式（单行有限表达式）；二者皆空即符号长度
    （静态分析保留符号，必须在 NamedAxis 里给出 unknown_reason，不以 0 代替）。
    display_value=True 时固定值直接显示（例如单 KV 头轴显示 ``1``）。
    """

    symbol: str
    meaning: str
    origin: str
    evidence_ref: str
    value: int | None = None
    expression: str | None = None
    name: str = ""
    display_value: bool = False
    display_override: str | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol.isidentifier():
            raise AxisError(f"轴 symbol 必须是标识符: {self.symbol!r}")
        if self.value is not None and self.expression is not None:
            raise AxisError(f"轴 {self.symbol}: value 与 expression 互斥")
        for alias in self.aliases:
            if not alias:
                raise AxisError(f"轴 {self.symbol}: 旧别名不能为空")
        if not self.name:
            object.__setattr__(self, "name", self.symbol)

    def to_named_axis(self) -> NamedAxis:
        unknown_reason = None
        if self.value is None and self.expression is None:
            unknown_reason = "符号长度：由调用时输入/配置决定，静态分析保留符号，不当作未知常量"
        return NamedAxis(
            symbol=self.symbol,
            name=self.name,
            meaning=self.meaning,
            origin=self.origin,
            evidence_ref=self.evidence_ref,
            value=self.value,
            expression=self.expression,
            unknown_reason=unknown_reason,
        )

    def display(self) -> str:
        if self.display_override is not None:
            return self.display_override
        if self.value is not None and self.display_value:
            return str(self.value)
        if self.expression is not None:
            return self.expression
        return self.name


@dataclass
class AxisCatalog:
    """全称轴目录 + 旧别名映射（别名不是合法 symbol）。"""

    _axes: dict[str, AxisDef] = field(default_factory=dict)
    _aliases: dict[str, str] = field(default_factory=dict)

    def register(self, definition: AxisDef) -> None:
        if definition.symbol in self._axes:
            raise AxisError(f"轴重复登记: {definition.symbol}")
        if definition.symbol in self._aliases:
            raise AxisError(f"全称轴使用了已登记的旧别名: {definition.symbol}")
        self._axes[definition.symbol] = definition
        for alias in definition.aliases:
            if alias in self._axes:
                raise AxisError(f"旧别名与全称轴冲突: {alias}")
            if alias in self._aliases and self._aliases[alias] != definition.symbol:
                raise AxisError(f"旧别名重复映射: {alias}")
            self._aliases[alias] = definition.symbol

    def extend(self, definitions: tuple[AxisDef, ...]) -> "AxisCatalog":
        for definition in definitions:
            self.register(definition)
        return self

    def symbols(self) -> tuple[str, ...]:
        return tuple(self._axes)

    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def definition(self, symbol: str) -> AxisDef:
        if symbol in self._aliases:
            raise AxisError(
                f"{symbol!r} 是旧内部别名，主图/语义轴必须使用全称 "
                f"{self._aliases[symbol]!r}"
            )
        if symbol not in self._axes:
            raise AxisError(f"未登记的轴 symbol: {symbol!r}")
        return self._axes[symbol]

    def named_axis(self, symbol: str) -> NamedAxis:
        return self.definition(symbol).to_named_axis()

    def display(self, symbol: str) -> str:
        return self.definition(symbol).display()

    def shape_text(self, symbols: tuple[str, ...]) -> str:
        return "[" + ", ".join(self.display(symbol) for symbol in symbols) + "]"

    def to_json(self) -> list[dict]:
        return [
            {
                "symbol": definition.symbol,
                "name": definition.name,
                "meaning": definition.meaning,
                "origin": definition.origin,
                "evidence_ref": definition.evidence_ref,
                "value": definition.value,
                "expression": definition.expression,
                "display_value": definition.display_value,
                "display_override": definition.display_override,
                "aliases": list(definition.aliases),
            }
            for definition in self._axes.values()
        ]

    @classmethod
    def from_json(cls, records: list[dict]) -> "AxisCatalog":
        catalog = cls()
        for record in records:
            catalog.register(
                AxisDef(
                    symbol=record["symbol"],
                    meaning=record["meaning"],
                    origin=record["origin"],
                    evidence_ref=record["evidence_ref"],
                    value=record.get("value"),
                    expression=record.get("expression"),
                    name=record.get("name", ""),
                    display_value=bool(record.get("display_value", False)),
                    display_override=record.get("display_override"),
                    aliases=tuple(record.get("aliases", ())),
                )
            )
        return catalog
