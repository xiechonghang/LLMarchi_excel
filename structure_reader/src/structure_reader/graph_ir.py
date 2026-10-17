"""具名轴、端口、张量、节点、边、状态版本、guard、alias 与折叠组图 IR 数据模型——小单01 01a 探针增量。

本文件当前实际实现（仅以下十二项）：
- NamedAxis 标准 frozen dataclass 与稳定 JSON 往返（canonical 序列化，
  axes_to_json/axes_from_json）。不变量：value 与 expression 互斥；二者皆空即
  未知轴，必须给出非空 unknown_reason，未知不得以 0 代替（value=0 是合法
  确定值，与未知严格区分）。轴顺序语义保留：canonical JSON 只排序对象键，
  axes 数组顺序即轴顺序，不重排。
- Port 标准 frozen dataclass：name、direction（限 input|output）、tensor_id，
  必填字符串非空，to_dict/from_dict 可往返。
- Tensor 标准 frozen dataclass：name（张量名，与轴名分属不同命名空间）、dtype、
  有序 axes（构造时冻结为 tuple，顺序即逻辑维度顺序，不重排；允许未知轴，由
  NamedAxis 的 unknown_reason 表达）、有序 logical_layout 与 physical_layout
  （各为 axes 符号的完整排列，顺序即布局顺序，不排序不重排；两布局允许不同，
  如转置）、global_rank/local_rank（当前仅是张量秩（维度个数）标注，None 表示
  未标注，不与 len(axes) 交叉校验，也不表示并行下全局/本 rank 的轴长度）与
  派生 shape 属性：logical_shape（按 axes 顺序）与 physical_shape（按
  physical_layout 符号顺序重排），任一轴未知或仅有 expression 则为 None，本文件
  不求值表达式；shape 为 logical_shape 的别名。布局仅支持轴符号排列；分片/融合
  与 world_size>1 时不同 rank 轴长度的情况未实现，后续图/scenario 批次应记录
  首版 world_size=1 分析假设。同数值但不同 symbol 的轴按对象保留，不合并。
  stable_id 形如 tensor:{name}，随序列化落盘并在读取时校验一致。
- Node 标准 frozen dataclass：显式稳定 stable_id（必填字段，要求模板局部唯一，
  同算子多次调用不得共用；模板级唯一性校验留待图级构造）、op_type 与
  granularity（非空字符串，granularity 词表待后续批次收敛）、按方向分列的
  input_ports/output_ports（构造时冻结为 tuple，顺序即端口顺序，不重排；
  每个端口方向必须与所在列表一致，各列表内端口名不得重复，跨方向同名允许，
  direction 可区分）、可选有序 evidence_refs（原样保留顺序）。
  to_dict/from_dict 可往返。

- Edge 标准 frozen dataclass：显式稳定 stable_id（必填，图级唯一性留待检查器）、
  生产端 src_node_id/src_port 与消费端 dst_node_id/dst_port（端点为"节点 ID+端口名"
  对，本文件只查非空与首尾空白，不校验存在性）、tensor_id（引用张量 stable_id，
  是 ID 不是显示名称，不校验存在性，也不以数值 shape 判连接相容——相容性属图级，
  且按语义轴判断，不按数值相等）、kind（限 data|state|select|control）、可选
  guard_ref（预期指向 Guard.stable_id，本文件只查非空，不校验存在性）、有序
  evidence_refs（仅接受 list/tuple，不拆字符串）与可选 state_transition_ref
  （引用 StateTransition.stable_id）。使用规则：data/select/control 边连接
  生产/消费端口；state 边表达跨调用状态前后版本，必须携带 state_transition_ref，
  data/select/control 边禁止携带（Edge 侧仅存引用字符串并做上述 kind 约束；
  引用存在性、无版本循环检查属图级）。to_dict/from_dict 可往返。
- StateVersion 标准 frozen dataclass：显式稳定 stable_id、state_id、version_label
  三必填字符串（类型/非空/首尾空白检查）。这是状态在某一时刻的版本标识，
  不携带 capacity/valid_length 等 cache 属性（未实现）。
- StateTransition 标准 frozen dataclass：显式稳定 stable_id、state_id、
  before_version_id、after_version_id 与有序可选 evidence_refs（仅 list/tuple，
  不拆字符串）。before_version_id 与 after_version_id 必须不同（有向更新）；
  引用的 state_id/version 一致性、跨调用先后与无版本环属图级检查器，
  本文件不伪造判断。与 Edge 的约定：state 边的 state_transition_ref 引用
  StateTransition.stable_id，普通（data/select/control）边禁止引用（Edge 侧
  规则不变，本单不改 Edge 公共字段/序列化）。

- AliasRelation 标准 frozen dataclass：显式稳定 stable_id、base_tensor_id、
  alias_tensor_id、relation_kind（限 same_tensor|view|slice）、可选 alias_role、
  可选 transform_expr 与有序 evidence_refs（仅 list/tuple，不拆字符串）。
  局部不变量按 relation_kind 分叉：same_tensor 要求两个 tensor ID 相同且
  alias_role 非空（表达 DeepSeek 共享 KV：K/V 两角色引用同一 tensor_id，
  各自 alias_role 如 key/value 说明角色，不额外造两个投影/权重）；view/slice
  要求两个 tensor ID 不同且 transform_expr 非空（表达缓存 view/slice：
  transform_expr 存原文，不执行 Python 表达式，不声称独立存储）。
  to_dict/from_dict 可往返。

- FoldBoundaryMapping 标准 frozen dataclass：一条折叠边界映射——direction
  （限 input|output）、外部端口名 external_port、边类别 edge_kind（限
  data|state|select|control）、内部节点 ID internal_node_id、内部端口名
  internal_port（均必填非空无首尾空白）与有序 evidence_refs（仅 list/tuple，
  不拆字符串，各项非空，顺序保留）。state/select/control 边界同 data
  一样必须显式映射，不静默省略。
- FoldGroup 标准 frozen dataclass：显式稳定 stable_id、有序 member_node_ids
  （构造时冻结为 tuple，组内不得重复）、有序 boundary_mappings（冻结为 tuple
  的 FoldBoundaryMapping 序列）。局部不变量：每条映射的 internal_node_id 必须
  属于本组；同一外部边界键 (direction, external_port, edge_kind) 至多一条映射，
  出现第二次（即使映射完全相同）即具名拒绝——端口名相同但 direction 或
  edge_kind 不同的边界键可区分，各允许一条。to_dict/from_dict 可往返。
  图级检查器随后验证外部 Edge 存在性、旁路边与实际端口引用存在性，
  本文件不虚构连通性；折叠组不删除旁路（旁路是否被正确保留由图级核验）。

尚未实现，本文件不声称：SemanticGraph、graph_to_json/graph_from_json、图级
（含节点/张量/边/guard/alias/折叠组集合的）序列化与唯一性校验、折叠组的外部
Edge/旁路边/实际端口引用存在性检查、折叠前后连通性等价证明、alias 的张量引用
存在性检查、同一角色重复绑定、存储别名与源码证据的一致性、view 轴变换相容性、
端口方向/跨调用版本顺序检查、Node stable_id 的模板级唯一性检查、predicate_expr
与 transform_expr 的求值（有限求值器在开发单 02，本文件不执行/导入模型代码）、
state/cache 数值属性（capacity、valid_length、压缩 remainder/overlap、
Indexer 状态）、template/instance 绑定、schema 校验器、检查器、根 API 导出
（留待本批后续与 01b/01c）。
遵循 01a 派工既定选择：标准库 dataclass、semantic-graph/v1、显式稳定 ID。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

SCHEMA = "semantic-graph/v1"
_KIND_NAMED_AXES = "named_axes"


@dataclass(frozen=True)
class NamedAxis:
    """一条具名轴。value 与 expression 互斥；二者皆空即未知轴。"""

    symbol: str
    name: str
    meaning: str
    origin: str
    evidence_ref: str
    value: Optional[int] = None
    expression: Optional[str] = None
    unknown_reason: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ("symbol", "name", "meaning", "origin", "evidence_ref"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
        if self.symbol != self.symbol.strip():
            raise ValueError("symbol 不允许首尾空白")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, int):
                raise ValueError("value 必须是整数")
        if self.value is not None and self.expression is not None:
            raise ValueError("value 与 expression 互斥，只能确定其一")
        if self.expression is not None:
            if not self.expression.strip() or "\n" in self.expression:
                raise ValueError("expression 必须是单行非空有限表达式")
        if self.value is not None or self.expression is not None:
            if self.unknown_reason:
                raise ValueError("已知轴不允许填写 unknown_reason")
        elif not (isinstance(self.unknown_reason, str) and self.unknown_reason.strip()):
            raise ValueError("未知轴必须给出非空 unknown_reason，不得以 0 代替未知")

    @property
    def stable_id(self) -> str:
        """显式稳定 ID：由 symbol 派生，序列化落盘并在读取时校验一致。"""
        return f"axis:{self.symbol}"

    @property
    def is_unknown(self) -> bool:
        return self.value is None and self.expression is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "symbol": self.symbol,
            "name": self.name,
            "meaning": self.meaning,
            "value": self.value,
            "expression": self.expression,
            "origin": self.origin,
            "evidence_ref": self.evidence_ref,
            "unknown_reason": self.unknown_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NamedAxis":
        if not isinstance(data, dict):
            raise ValueError("轴记录必须是 JSON 对象")
        required = ("stable_id", "symbol", "name", "meaning", "origin", "evidence_ref")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"轴记录缺少字段: {', '.join(missing)}")
        if data["stable_id"] != f"axis:{data['symbol']}":
            raise ValueError("stable_id 与 symbol 不一致")
        return cls(
            symbol=data["symbol"],
            name=data["name"],
            meaning=data["meaning"],
            origin=data["origin"],
            evidence_ref=data["evidence_ref"],
            value=data.get("value"),
            expression=data.get("expression"),
            unknown_reason=data.get("unknown_reason"),
        )


_DIRECTION_INPUT = "input"
_DIRECTION_OUTPUT = "output"
_DIRECTIONS = (_DIRECTION_INPUT, _DIRECTION_OUTPUT)


@dataclass(frozen=True)
class Port:
    """一个端口：图节点上具名、有方向的连接点，指向一个张量。

    direction 限 input|output；name 与 tensor_id 为非空字符串（不允许首尾空白）。
    """

    name: str
    direction: str
    tensor_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name 必须是非空字符串")
        if self.name != self.name.strip():
            raise ValueError("name 不允许首尾空白")
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"direction 必须是 {' 或 '.join(_DIRECTIONS)}，得到 {self.direction!r}"
            )
        if not isinstance(self.tensor_id, str) or not self.tensor_id.strip():
            raise ValueError("tensor_id 必须是非空字符串")
        if self.tensor_id != self.tensor_id.strip():
            raise ValueError("tensor_id 不允许首尾空白")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "tensor_id": self.tensor_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Port":
        if not isinstance(data, dict):
            raise ValueError("端口记录必须是 JSON 对象")
        required = ("name", "direction", "tensor_id")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"端口记录缺少字段: {', '.join(missing)}")
        return cls(
            name=data["name"],
            direction=data["direction"],
            tensor_id=data["tensor_id"],
        )


@dataclass(frozen=True)
class Tensor:
    """一个张量：具名、带 dtype，按有序轴描述形状，并带有序布局与秩标注。

    name 是张量名（与轴名分属不同命名空间）；stable_id 形如 tensor:{name}。
    axes 顺序即张量逻辑维度顺序，构造时冻结为 tuple，不排序不重排；允许未知轴，
    由 NamedAxis 的 unknown_reason 表达。logical_layout 与 physical_layout 是
    axes 符号的完整排列（每个符号恰好一次，允许两布局互不相同，如转置），
    原样保留——本文件仅支持排列式布局；分片/融合/重复维度的物理布局
    （如 TP 把 H 切成两维、QKV 融合）超出当前契约，留待后续 shard/fuse 规范，
    故 world_size>1 时不同 rank 轴长度不同的情况当前不可表达。
    global_rank/local_rank 当前只是张量秩（维度个数）标注（None 表示未标注），
    不与 len(axes) 交叉校验，也不表示并行下全局/本 rank 的轴长度；并行维度
    区分（world_size、per-rank 长度）未实现，首版场景按 world_size=1 假设分析，
    该假设应记录在后续图/scenario 批次中。shape 为 logical_shape 的别名。
    logical_shape 按 axes 顺序给出确定值；physical_shape 按 physical_layout
    符号顺序重排确定 value；任一轴未知或仅有 expression 时均为 None，
    不执行表达式求值。同数值但不同 symbol 的轴按对象保留，不合并。
    """

    name: str
    dtype: str
    axes: Sequence[NamedAxis]
    logical_layout: Sequence[str]
    physical_layout: Sequence[str]
    global_rank: Optional[int] = None
    local_rank: Optional[int] = None

    def __post_init__(self) -> None:
        for field_name in ("name", "dtype"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            if field_value != field_value.strip():
                raise ValueError(f"{field_name} 不允许首尾空白")
        if not isinstance(self.axes, (list, tuple)):
            raise ValueError("axes 必须是 NamedAxis 的 list/tuple 序列")
        axes = tuple(self.axes)
        seen: set[str] = set()
        for index, axis in enumerate(axes):
            if not isinstance(axis, NamedAxis):
                raise ValueError(f"axes[{index}] 必须是 NamedAxis")
            if axis.stable_id in seen:
                raise ValueError(f"axes 内 stable_id 重复: {axis.stable_id}")
            seen.add(axis.stable_id)
        symbols = [axis.symbol for axis in axes]
        for layout_name in ("logical_layout", "physical_layout"):
            layout_value = getattr(self, layout_name)
            if not isinstance(layout_value, (list, tuple)):
                raise ValueError(f"{layout_name} 必须是轴 symbol 的 list/tuple 序列")
            layout = tuple(layout_value)
            for index, entry in enumerate(layout):
                if not isinstance(entry, str):
                    raise ValueError(f"{layout_name}[{index}] 必须是轴 symbol 字符串")
            if sorted(layout) != sorted(symbols):
                raise ValueError(
                    f"{layout_name} 必须是本张量 axes 符号的完整排列（每个恰好一次）"
                )
            object.__setattr__(self, layout_name, layout)
        for rank_name in ("global_rank", "local_rank"):
            rank = getattr(self, rank_name)
            if rank is not None and (
                isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            ):
                raise ValueError(f"{rank_name} 必须是非负整数或 None")
        object.__setattr__(self, "axes", axes)

    @property
    def stable_id(self) -> str:
        """显式稳定 ID：由 name 派生，序列化落盘并在读取时校验一致。"""
        return f"tensor:{self.name}"

    def _shape_by_layout(self, layout: Sequence[str]) -> Optional[tuple[int, ...]]:
        """按给定布局符号顺序重排轴确定值；任一轴无确定 value 则 None。"""
        values_by_symbol = {}
        for axis in self.axes:
            if axis.value is None:
                return None
            values_by_symbol[axis.symbol] = axis.value
        return tuple(values_by_symbol[symbol] for symbol in layout)

    @property
    def logical_shape(self) -> Optional[tuple[int, ...]]:
        """按 axes 顺序的逻辑形状；含未知轴或表达式轴时为 None（不求值表达式）。"""
        return self._shape_by_layout([axis.symbol for axis in self.axes])

    @property
    def physical_shape(self) -> Optional[tuple[int, ...]]:
        """按 physical_layout 符号顺序的物理形状；任一轴无确定 value 时为 None。"""
        return self._shape_by_layout(self.physical_layout)

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """logical_shape 的别名（按 axes 顺序），保留以兼容既有调用与测试。"""
        return self.logical_shape

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "name": self.name,
            "dtype": self.dtype,
            "axes": [axis.to_dict() for axis in self.axes],
            "logical_layout": list(self.logical_layout),
            "physical_layout": list(self.physical_layout),
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tensor":
        if not isinstance(data, dict):
            raise ValueError("张量记录必须是 JSON 对象")
        required = (
            "stable_id",
            "name",
            "dtype",
            "axes",
            "logical_layout",
            "physical_layout",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"张量记录缺少字段: {', '.join(missing)}")
        if data["stable_id"] != f"tensor:{data['name']}":
            raise ValueError("stable_id 与 name 不一致")
        raw_axes = data["axes"]
        if not isinstance(raw_axes, list):
            raise ValueError("axes 必须是 JSON 数组")
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            axes=[NamedAxis.from_dict(item) for item in raw_axes],
            logical_layout=data["logical_layout"],
            physical_layout=data["physical_layout"],
            global_rank=data.get("global_rank"),
            local_rank=data.get("local_rank"),
        )


@dataclass(frozen=True)
class Node:
    """一个节点：模板局部唯一的算子调用点，按方向分列端口并保留顺序。

    stable_id 是显式稳定 ID（必填字段）：模板局部唯一，同算子多次调用不得
    共用；本文件无模板上下文，模板级唯一性留待图级构造与检查器校验。
    op_type 与 granularity 为非空字符串（granularity 词表待后续批次收敛）。
    input_ports/output_ports 按方向分列，各冻结为 tuple，顺序即端口顺序，
    不排序不重排；每个端口 direction 必须与所在列表一致，各列表内端口名
    不得重复，跨方向同名允许（direction 可区分）。evidence_refs 为可选有序
    引用列表，原样保留顺序。
    """

    stable_id: str
    op_type: str
    granularity: str
    input_ports: Sequence[Port] = ()
    output_ports: Sequence[Port] = ()
    evidence_refs: Sequence[str] = ()

    def __post_init__(self) -> None:
        for field_name in ("stable_id", "op_type", "granularity"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            if field_value != field_value.strip():
                raise ValueError(f"{field_name} 不允许首尾空白")
        expected_directions = {
            "input_ports": _DIRECTION_INPUT,
            "output_ports": _DIRECTION_OUTPUT,
        }
        for ports_name, expected in expected_directions.items():
            ports_value = getattr(self, ports_name)
            if not isinstance(ports_value, (list, tuple)):
                raise ValueError(f"{ports_name} 必须是 Port 的 list/tuple 序列")
            ports = tuple(ports_value)
            seen_names: set[str] = set()
            for index, port in enumerate(ports):
                if not isinstance(port, Port):
                    raise ValueError(f"{ports_name}[{index}] 必须是 Port")
                if port.direction != expected:
                    raise ValueError(
                        f"{ports_name}[{index}] 的 direction 必须是 {expected}，"
                        f"得到 {port.direction!r}（{port.name!r}）"
                    )
                if port.name in seen_names:
                    raise ValueError(f"{ports_name} 内端口名重复: {port.name}")
                seen_names.add(port.name)
            object.__setattr__(self, ports_name, ports)
        if not isinstance(self.evidence_refs, (list, tuple)):
            raise ValueError(
                "evidence_refs 必须是有序 list/tuple 引用列表，"
                f"得到 {type(self.evidence_refs).__name__}"
            )
        evidence = tuple(self.evidence_refs)
        for index, ref in enumerate(evidence):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"evidence_refs[{index}] 必须是非空字符串")
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "op_type": self.op_type,
            "granularity": self.granularity,
            "input_ports": [port.to_dict() for port in self.input_ports],
            "output_ports": [port.to_dict() for port in self.output_ports],
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        if not isinstance(data, dict):
            raise ValueError("节点记录必须是 JSON 对象")
        required = ("stable_id", "op_type", "granularity")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"节点记录缺少字段: {', '.join(missing)}")
        for key in ("input_ports", "output_ports", "evidence_refs"):
            value = data.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{key} 必须是 JSON 数组，得到 {type(value).__name__}")
        return cls(
            stable_id=data["stable_id"],
            op_type=data["op_type"],
            granularity=data["granularity"],
            input_ports=[
                Port.from_dict(item) for item in data.get("input_ports", [])
            ],
            output_ports=[
                Port.from_dict(item) for item in data.get("output_ports", [])
            ],
            evidence_refs=list(data.get("evidence_refs", [])),
        )


_EDGE_KIND_DATA = "data"
_EDGE_KIND_STATE = "state"
_EDGE_KIND_SELECT = "select"
_EDGE_KIND_CONTROL = "control"
_EDGE_KINDS = (_EDGE_KIND_DATA, _EDGE_KIND_STATE, _EDGE_KIND_SELECT, _EDGE_KIND_CONTROL)


@dataclass(frozen=True)
class Edge:
    """一条边：生产节点/端口到消费节点/端口的有向连接。

    stable_id 是显式稳定 ID（必填字段），图级唯一性留待检查器。端点是
    （节点 ID, 端口名）对：本文件只查非空与首尾空白，不校验节点/端口是否
    存在、端口方向是否与连接一致——引用完整性与方向核验属图级检查器。
    tensor_id 引用一个张量的 stable_id，是 ID 不是显示名称；不以数值 shape
    判断连接相容（相容性按语义轴与布局判断，属图级）。
    kind 限 data|state|select|control。使用规则：data/select/control 边连接
    生产/消费端口；state 边表达跨调用状态前后版本，必须携带
    state_transition_ref（引用 StateTransition.stable_id），data/select/control
    边禁止携带该引用。guard_ref 可选，预期指向 Guard.stable_id（本文件只查
    非空，存在性属图级检查器）。evidence_refs 为有序引用列表，仅接受
    list/tuple，各项非空字符串。
    """

    stable_id: str
    src_node_id: str
    src_port: str
    dst_node_id: str
    dst_port: str
    tensor_id: str
    kind: str
    evidence_refs: Sequence[str] = ()
    guard_ref: Optional[str] = None
    state_transition_ref: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
            "stable_id",
            "src_node_id",
            "src_port",
            "dst_node_id",
            "dst_port",
            "tensor_id",
            "kind",
        ):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            if field_value != field_value.strip():
                raise ValueError(f"{field_name} 不允许首尾空白")
        if self.kind not in _EDGE_KINDS:
            raise ValueError(
                f"kind 必须是 {' 或 '.join(_EDGE_KINDS)}，得到 {self.kind!r}"
            )
        for ref_name in ("guard_ref", "state_transition_ref"):
            ref = getattr(self, ref_name)
            if ref is not None and (
                not isinstance(ref, str) or not ref.strip() or ref != ref.strip()
            ):
                raise ValueError(f"{ref_name} 必须是非空字符串或 None")
        if not isinstance(self.evidence_refs, (list, tuple)):
            raise ValueError(
                "evidence_refs 必须是有序 list/tuple 引用列表，"
                f"得到 {type(self.evidence_refs).__name__}"
            )
        evidence = tuple(self.evidence_refs)
        for index, ref in enumerate(evidence):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"evidence_refs[{index}] 必须是非空字符串")
        object.__setattr__(self, "evidence_refs", evidence)
        if self.kind == _EDGE_KIND_STATE and self.state_transition_ref is None:
            raise ValueError(
                "state 边必须携带 state_transition_ref 指明状态版本引用"
            )
        if self.kind != _EDGE_KIND_STATE and self.state_transition_ref is not None:
            raise ValueError(
                f"{self.kind} 边不允许携带 state_transition_ref（仅 state 边使用）"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "src_node_id": self.src_node_id,
            "src_port": self.src_port,
            "dst_node_id": self.dst_node_id,
            "dst_port": self.dst_port,
            "tensor_id": self.tensor_id,
            "kind": self.kind,
            "evidence_refs": list(self.evidence_refs),
            "guard_ref": self.guard_ref,
            "state_transition_ref": self.state_transition_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Edge":
        if not isinstance(data, dict):
            raise ValueError("边记录必须是 JSON 对象")
        required = (
            "stable_id",
            "src_node_id",
            "src_port",
            "dst_node_id",
            "dst_port",
            "tensor_id",
            "kind",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"边记录缺少字段: {', '.join(missing)}")
        for key in ("evidence_refs",):
            value = data.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{key} 必须是 JSON 数组，得到 {type(value).__name__}")
        return cls(
            stable_id=data["stable_id"],
            src_node_id=data["src_node_id"],
            src_port=data["src_port"],
            dst_node_id=data["dst_node_id"],
            dst_port=data["dst_port"],
            tensor_id=data["tensor_id"],
            kind=data["kind"],
            evidence_refs=list(data.get("evidence_refs", [])),
            guard_ref=data.get("guard_ref"),
            state_transition_ref=data.get("state_transition_ref"),
        )


@dataclass(frozen=True)
class StateVersion:
    """一个状态在某一时刻的版本标识（身份，不含容量等 cache 属性）。

    stable_id 为显式稳定 ID；state_id 标识所属状态，version_label 标识该状态
    内的版本（如 before/after 或步次标签，词表未约束）。三者必填，类型/非空/
    首尾空白检查。本类只完成版本标识，不携带 capacity、valid_length 等属性；
    引用一致性与跨状态版本关系属图级检查器。
    """

    stable_id: str
    state_id: str
    version_label: str

    def __post_init__(self) -> None:
        for field_name in ("stable_id", "state_id", "version_label"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            if field_value != field_value.strip():
                raise ValueError(f"{field_name} 不允许首尾空白")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "state_id": self.state_id,
            "version_label": self.version_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateVersion":
        if not isinstance(data, dict):
            raise ValueError("状态版本记录必须是 JSON 对象")
        required = ("stable_id", "state_id", "version_label")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"状态版本记录缺少字段: {', '.join(missing)}")
        return cls(
            stable_id=data["stable_id"],
            state_id=data["state_id"],
            version_label=data["version_label"],
        )


@dataclass(frozen=True)
class StateTransition:
    """一次状态更新：同一 state_id 上从 before 版本到 after 版本的有向变化。

    stable_id 为显式稳定 ID；state_id、before_version_id、after_version_id 必填。
    before_version_id 与 after_version_id 必须不同（有向更新，方向即更新方向）；
    二者引用的版本是否存在、是否属于同一 state_id 的合法序列、跨调用先后与
    无版本环属图级检查器，本文件不伪造判断。evidence_refs 为有序引用列表，
    仅接受 list/tuple（不拆字符串、不取字典键），各项非空字符串，顺序保留。
    Edge 约定：state 边的 state_transition_ref 引用本类 stable_id，
    data/select/control 边禁止引用（约束在 Edge 侧执行）。
    """

    stable_id: str
    state_id: str
    before_version_id: str
    after_version_id: str
    evidence_refs: Sequence[str] = ()

    def __post_init__(self) -> None:
        for field_name in ("stable_id", "state_id", "before_version_id", "after_version_id"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            if field_value != field_value.strip():
                raise ValueError(f"{field_name} 不允许首尾空白")
        if self.before_version_id == self.after_version_id:
            raise ValueError(
                "before_version_id 与 after_version_id 必须不同"
                f"（有向更新），得到 {self.before_version_id!r}"
            )
        if not isinstance(self.evidence_refs, (list, tuple)):
            raise ValueError(
                "evidence_refs 必须是有序 list/tuple 引用列表，"
                f"得到 {type(self.evidence_refs).__name__}"
            )
        evidence = tuple(self.evidence_refs)
        for index, ref in enumerate(evidence):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"evidence_refs[{index}] 必须是非空字符串")
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "state_id": self.state_id,
            "before_version_id": self.before_version_id,
            "after_version_id": self.after_version_id,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateTransition":
        if not isinstance(data, dict):
            raise ValueError("状态更新记录必须是 JSON 对象")
        required = ("stable_id", "state_id", "before_version_id", "after_version_id")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"状态更新记录缺少字段: {', '.join(missing)}")
        for key in ("evidence_refs",):
            value = data.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{key} 必须是 JSON 数组，得到 {type(value).__name__}")
        return cls(
            stable_id=data["stable_id"],
            state_id=data["state_id"],
            before_version_id=data["before_version_id"],
            after_version_id=data["after_version_id"],
            evidence_refs=list(data.get("evidence_refs", [])),
        )


_GUARD_ACTIVE_TRUE = "true"
_GUARD_ACTIVE_FALSE = "false"
_GUARD_ACTIVE_UNKNOWN = "unknown"
_GUARD_ACTIVE_STATES = (
    _GUARD_ACTIVE_TRUE,
    _GUARD_ACTIVE_FALSE,
    _GUARD_ACTIVE_UNKNOWN,
)


@dataclass(frozen=True)
class Guard:
    """一个静态条件的身份定义：条件表达式原文与证据，不含任何场景判断结果。

    stable_id 为显式稳定 ID（Edge.guard_ref 预期指向它，存在性属图级检查器）；
    predicate_expr 是源码/规则中的条件表达式**原文**（非可执行 Python），本文件
    不求值、不执行、不导入模型代码——有限求值器在开发单 02。模板身份不随
    场景切换改变；某场景下的判断结果由 GuardActivation 承载。evidence_refs 为
    有序引用列表，仅接受 list/tuple（不拆字符串、不取字典键），各项非空。
    """

    stable_id: str
    predicate_expr: str
    evidence_refs: Sequence[str] = ()

    def __post_init__(self) -> None:
        for field_name in ("stable_id", "predicate_expr"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            if field_value != field_value.strip():
                raise ValueError(f"{field_name} 不允许首尾空白")
        if not isinstance(self.evidence_refs, (list, tuple)):
            raise ValueError(
                "evidence_refs 必须是有序 list/tuple 引用列表，"
                f"得到 {type(self.evidence_refs).__name__}"
            )
        evidence = tuple(self.evidence_refs)
        for index, ref in enumerate(evidence):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"evidence_refs[{index}] 必须是非空字符串")
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "predicate_expr": self.predicate_expr,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Guard":
        if not isinstance(data, dict):
            raise ValueError("guard 记录必须是 JSON 对象")
        required = ("stable_id", "predicate_expr")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"guard 记录缺少字段: {', '.join(missing)}")
        for key in ("evidence_refs",):
            value = data.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{key} 必须是 JSON 数组，得到 {type(value).__name__}")
        return cls(
            stable_id=data["stable_id"],
            predicate_expr=data["predicate_expr"],
            evidence_refs=list(data.get("evidence_refs", [])),
        )


@dataclass(frozen=True)
class GuardActivation:
    """某场景下一个 guard 的判断结果：明确三态 true|false|unknown。

    guard_id 引用 Guard.stable_id、scenario_id 标识场景（均为非空字符串；
    引用存在性、同一场景/guard 重复结果与结果冲突属图级检查器，本文件不做）。
    active_for_scenario 限 true|false|unknown 三态：unknown 不被当作 false，
    必须携带非空 unknown_reason 说明为何未知；确定 true/false 时不应携带
    unknown_reason（一律具名拒绝）。本类只保存判断结果，不推导、不求值。
    """

    guard_id: str
    scenario_id: str
    active_for_scenario: str
    unknown_reason: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ("guard_id", "scenario_id", "active_for_scenario"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            if field_value != field_value.strip():
                raise ValueError(f"{field_name} 不允许首尾空白")
        if self.active_for_scenario not in _GUARD_ACTIVE_STATES:
            raise ValueError(
                f"active_for_scenario 必须是 {' 或 '.join(_GUARD_ACTIVE_STATES)}"
                f"（三态，unknown 不当 false），得到 {self.active_for_scenario!r}"
            )
        reason = self.unknown_reason
        if self.active_for_scenario == _GUARD_ACTIVE_UNKNOWN:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    "unknown 必须给出非空 unknown_reason，不得被当作 false"
                )
            if reason != reason.strip():
                raise ValueError("unknown_reason 不允许首尾空白")
        elif reason is not None:
            raise ValueError("确定 true/false 不允许携带 unknown_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard_id": self.guard_id,
            "scenario_id": self.scenario_id,
            "active_for_scenario": self.active_for_scenario,
            "unknown_reason": self.unknown_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuardActivation":
        if not isinstance(data, dict):
            raise ValueError("guard 判断记录必须是 JSON 对象")
        required = ("guard_id", "scenario_id", "active_for_scenario")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"guard 判断记录缺少字段: {', '.join(missing)}")
        return cls(
            guard_id=data["guard_id"],
            scenario_id=data["scenario_id"],
            active_for_scenario=data["active_for_scenario"],
            unknown_reason=data.get("unknown_reason"),
        )


_ALIAS_KIND_SAME = "same_tensor"
_ALIAS_KIND_VIEW = "view"
_ALIAS_KIND_SLICE = "slice"
_ALIAS_KINDS = (_ALIAS_KIND_SAME, _ALIAS_KIND_VIEW, _ALIAS_KIND_SLICE)


@dataclass(frozen=True)
class AliasRelation:
    """一条张量别名关系：共享同一存储或 view/slice 派生，按 relation_kind 表达。

    stable_id 为显式稳定 ID；base_tensor_id 与 alias_tensor_id 引用张量
    stable_id（存在性属图级检查器，本文件不查）。relation_kind 限
    same_tensor|view|slice，局部不变量按 kind 分叉：
    - same_tensor：两个 tensor ID 必须相同，alias_role 必填非空——表达共享 KV
      这类"两个消费角色引用同一存储"（如 K、V 角色各自一条 alias_role=key/value
      的 same_tensor 关系，不额外造两个投影或两个权重）；
    - view/slice：两个 tensor ID 必须不同，transform_expr 必填非空——记录派生
      关系的变换表达式**原文**，本文件不执行 Python 表达式，也不声称 alias 是
      独立存储。
    alias_role 与 transform_expr 在不要求的 kind 下必须为 None（具名拒绝）。
    evidence_refs 为有序引用列表，仅接受 list/tuple（不拆字符串、不取字典键），
    各项非空。图级核验（张量存在性、同一角色重复绑定、存储别名与源码证据
    一致性、view 轴变换相容性）属 01b 或源码核验，本文件不假判。
    """

    stable_id: str
    base_tensor_id: str
    alias_tensor_id: str
    relation_kind: str
    alias_role: Optional[str] = None
    transform_expr: Optional[str] = None
    evidence_refs: Sequence[str] = ()

    def __post_init__(self) -> None:
        for field_name in ("stable_id", "base_tensor_id", "alias_tensor_id", "relation_kind"):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            if field_value != field_value.strip():
                raise ValueError(f"{field_name} 不允许首尾空白")
        if self.relation_kind not in _ALIAS_KINDS:
            raise ValueError(
                f"relation_kind 必须是 {' 或 '.join(_ALIAS_KINDS)}，得到 {self.relation_kind!r}"
            )
        for opt_name in ("alias_role", "transform_expr"):
            opt = getattr(self, opt_name)
            if opt is not None and (
                not isinstance(opt, str) or not opt.strip() or opt != opt.strip()
            ):
                raise ValueError(f"{opt_name} 必须是非空字符串或 None")
        if self.relation_kind == _ALIAS_KIND_SAME:
            if self.base_tensor_id != self.alias_tensor_id:
                raise ValueError(
                    f"same_tensor 要求 base_tensor_id 与 alias_tensor_id 相同，"
                    f"得到 {self.base_tensor_id!r} 与 {self.alias_tensor_id!r}"
                )
            if self.alias_role is None:
                raise ValueError("same_tensor 要求非空 alias_role 说明消费角色")
            if self.transform_expr is not None:
                raise ValueError("same_tensor 不允许携带 transform_expr")
        else:
            if self.base_tensor_id == self.alias_tensor_id:
                raise ValueError(
                    f"{self.relation_kind} 要求 base_tensor_id 与 alias_tensor_id 不同，"
                    f"得到 {self.base_tensor_id!r}"
                )
            if self.transform_expr is None:
                raise ValueError(
                    f"{self.relation_kind} 要求非空 transform_expr 记录变换表达式原文"
                )
            if self.alias_role is not None:
                raise ValueError(f"{self.relation_kind} 不允许携带 alias_role")
        if not isinstance(self.evidence_refs, (list, tuple)):
            raise ValueError(
                "evidence_refs 必须是有序 list/tuple 引用列表，"
                f"得到 {type(self.evidence_refs).__name__}"
            )
        evidence = tuple(self.evidence_refs)
        for index, ref in enumerate(evidence):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"evidence_refs[{index}] 必须是非空字符串")
        object.__setattr__(self, "evidence_refs", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "base_tensor_id": self.base_tensor_id,
            "alias_tensor_id": self.alias_tensor_id,
            "relation_kind": self.relation_kind,
            "alias_role": self.alias_role,
            "transform_expr": self.transform_expr,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AliasRelation":
        if not isinstance(data, dict):
            raise ValueError("别名关系记录必须是 JSON 对象")
        required = ("stable_id", "base_tensor_id", "alias_tensor_id", "relation_kind")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"别名关系记录缺少字段: {', '.join(missing)}")
        for key in ("evidence_refs",):
            value = data.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{key} 必须是 JSON 数组，得到 {type(value).__name__}")
        return cls(
            stable_id=data["stable_id"],
            base_tensor_id=data["base_tensor_id"],
            alias_tensor_id=data["alias_tensor_id"],
            relation_kind=data["relation_kind"],
            alias_role=data.get("alias_role"),
            transform_expr=data.get("transform_expr"),
            evidence_refs=list(data.get("evidence_refs", [])),
        )


@dataclass(frozen=True)
class FoldBoundaryMapping:
    """一条折叠边界映射：外部输入/输出端口对应到折叠组内部节点端口。

    direction 限 input|output（从折叠组外部视角：input 是外部流入组内，
    output 是组内流出外部）；external_port 为外部端口名；edge_kind 限
    data|state|select|control——四类边界一律显式映射，state/select/control
    不静默省略（旁路是否保留由图级检查器核验）；internal_node_id 与
    internal_port 指向组内节点的端口（归属与存在性由 FoldGroup 与图级校验）。
    evidence_refs 为有序来源证据引用：仅接受 list/tuple（str 不拆解、dict 不取
    键），各项非空字符串，顺序保留。全部必填字段非空字符串，不允许首尾空白。
    """

    direction: str
    external_port: str
    edge_kind: str
    internal_node_id: str
    internal_port: str
    evidence_refs: Sequence[str] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "external_port",
            "internal_node_id",
            "internal_port",
        ):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
            if field_value != field_value.strip():
                raise ValueError(f"{field_name} 不允许首尾空白")
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"direction 必须是 {' 或 '.join(_DIRECTIONS)}，得到 {self.direction!r}"
            )
        if self.edge_kind not in _EDGE_KINDS:
            raise ValueError(
                f"edge_kind 必须是 {' 或 '.join(_EDGE_KINDS)}，得到 {self.edge_kind!r}"
            )
        if not isinstance(self.evidence_refs, (list, tuple)):
            raise ValueError(
                "evidence_refs 必须是有序 list/tuple 引用列表，"
                f"得到 {type(self.evidence_refs).__name__}"
            )
        evidence = tuple(self.evidence_refs)
        for index, ref in enumerate(evidence):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"evidence_refs[{index}] 必须是非空字符串")
        object.__setattr__(self, "evidence_refs", evidence)

    def boundary_key(self) -> tuple[str, str, str]:
        """外部边界键 (direction, external_port, edge_kind)：FoldGroup 查重用。"""
        return (self.direction, self.external_port, self.edge_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "external_port": self.external_port,
            "edge_kind": self.edge_kind,
            "internal_node_id": self.internal_node_id,
            "internal_port": self.internal_port,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FoldBoundaryMapping":
        if not isinstance(data, dict):
            raise ValueError("折叠边界映射记录必须是 JSON 对象")
        required = ("direction", "external_port", "edge_kind", "internal_node_id", "internal_port")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"折叠边界映射记录缺少字段: {', '.join(missing)}")
        for key in ("evidence_refs",):
            value = data.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{key} 必须是 JSON 数组，得到 {type(value).__name__}")
        return cls(
            direction=data["direction"],
            external_port=data["external_port"],
            edge_kind=data["edge_kind"],
            internal_node_id=data["internal_node_id"],
            internal_port=data["internal_port"],
            evidence_refs=list(data.get("evidence_refs", [])),
        )


@dataclass(frozen=True)
class FoldGroup:
    """一个折叠组：被折叠为单一线条的内部节点集合及其边界端口映射。

    stable_id 为显式稳定 ID；member_node_ids 为有序组内节点 ID（冻结为 tuple，
    组内不得重复）；boundary_mappings 为有序 FoldBoundaryMapping 序列（冻结为
    tuple，顺序保留不重排）。局部不变量：每条映射的 internal_node_id 必须属于
    本组；同一外部边界键 (direction, external_port, edge_kind) 至多一条映射，
    出现第二次即具名拒绝（不区分映射是否完全相同）。外部 Edge 存在性、旁路边
    与实际端口引用存在性属图级检查器，本文件不虚构连通性。
    """

    stable_id: str
    member_node_ids: Sequence[str]
    boundary_mappings: Sequence[FoldBoundaryMapping]

    def __post_init__(self) -> None:
        if not isinstance(self.stable_id, str) or not self.stable_id.strip():
            raise ValueError("stable_id 必须是非空字符串")
        if self.stable_id != self.stable_id.strip():
            raise ValueError("stable_id 不允许首尾空白")
        if not isinstance(self.member_node_ids, (list, tuple)):
            raise ValueError("member_node_ids 必须是字符串的 list/tuple 序列")
        members = tuple(self.member_node_ids)
        seen: set[str] = set()
        for index, node_id in enumerate(members):
            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError(f"member_node_ids[{index}] 必须是非空字符串")
            if node_id != node_id.strip():
                raise ValueError(f"member_node_ids[{index}] 不允许首尾空白")
            if node_id in seen:
                raise ValueError(f"member_node_ids 组内节点 ID 重复: {node_id}")
            seen.add(node_id)
        if not isinstance(self.boundary_mappings, (list, tuple)):
            raise ValueError(
                "boundary_mappings 必须是 FoldBoundaryMapping 的 list/tuple 序列"
            )
        mappings = tuple(self.boundary_mappings)
        seen_keys: dict[tuple[str, str, str], FoldBoundaryMapping] = {}
        for index, mapping in enumerate(mappings):
            if not isinstance(mapping, FoldBoundaryMapping):
                raise ValueError(
                    f"boundary_mappings[{index}] 必须是 FoldBoundaryMapping"
                )
            if mapping.internal_node_id not in seen:
                raise ValueError(
                    f"boundary_mappings[{index}] 的内部节点 "
                    f"{mapping.internal_node_id!r} 不属于本折叠组"
                )
            key = mapping.boundary_key()
            if key in seen_keys:
                previous = seen_keys[key]
                raise ValueError(
                    f"边界 (direction={key[0]}, external_port={key[1]}, "
                    f"edge_kind={key[2]}) 存在冲突的重复映射："
                    f"{previous.internal_node_id}.{previous.internal_port} 与 "
                    f"{mapping.internal_node_id}.{mapping.internal_port}"
                )
            seen_keys[key] = mapping
        object.__setattr__(self, "member_node_ids", members)
        object.__setattr__(self, "boundary_mappings", mappings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "member_node_ids": list(self.member_node_ids),
            "boundary_mappings": [
                mapping.to_dict() for mapping in self.boundary_mappings
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FoldGroup":
        if not isinstance(data, dict):
            raise ValueError("折叠组记录必须是 JSON 对象")
        required = ("stable_id", "member_node_ids", "boundary_mappings")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"折叠组记录缺少字段: {', '.join(missing)}")
        for key in ("member_node_ids", "boundary_mappings"):
            value = data.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{key} 必须是 JSON 数组，得到 {type(value).__name__}")
        return cls(
            stable_id=data["stable_id"],
            member_node_ids=list(data["member_node_ids"]),
            boundary_mappings=[
                FoldBoundaryMapping.from_dict(item)
                for item in data["boundary_mappings"]
            ],
        )


def _require_unique_symbols(axes: Sequence[NamedAxis]) -> None:
    seen: set[str] = set()
    for axis in axes:
        if axis.stable_id in seen:
            raise ValueError(f"stable_id 重复: {axis.stable_id}")
        seen.add(axis.stable_id)


def axes_to_json(axes: Sequence[NamedAxis]) -> str:
    """canonical 序列化：sort_keys 仅排序对象键；axes 数组顺序即轴顺序，刻意不重排。"""
    _require_unique_symbols(axes)
    envelope = {
        "schema": SCHEMA,
        "kind": _KIND_NAMED_AXES,
        "axes": [axis.to_dict() for axis in axes],
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def axes_from_json(text: str) -> list[NamedAxis]:
    """反序列化：保留数组顺序；校验 schema/kind 与 stable_id 一致性。"""
    envelope = json.loads(text)
    if not isinstance(envelope, dict):
        raise ValueError("顶层必须是 JSON 对象")
    if envelope.get("schema") != SCHEMA:
        raise ValueError(f"schema 不符，期望 {SCHEMA}")
    if envelope.get("kind") != _KIND_NAMED_AXES:
        raise ValueError(f"kind 不符，期望 {_KIND_NAMED_AXES}")
    raw_axes = envelope.get("axes")
    if not isinstance(raw_axes, list):
        raise ValueError("缺少 axes 数组")
    axes = [NamedAxis.from_dict(item) for item in raw_axes]
    _require_unique_symbols(axes)
    return axes
