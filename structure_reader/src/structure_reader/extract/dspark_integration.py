"""DeepSeek-V4-Flash-0731 DSpark 整体集成图（工作包 B · 步骤 B4）。

数据来源：models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/{model.py,
generate.py} 与 inference/config.json（固定 SHA256 基线，
:mod:`structure_reader.extract.evidence` 逐条核对原文）。静态取证，
不导入/不执行模型代码。

B4 范围（整体集成的图装配层；组件 B1/B2/B3a/B3b 均已组件范围认可，
本模块只做真实整体装配，不重做任何组件内部图）：
1. 真实合并六个已认可组件图为一张整体图（template_id
   ``deepseek_v4_flash_0731:dspark_integration``）：
   - 首入口 entry（B3a）：主干 40/41/42 readout mean → 有序 concat →
     stage0 唯一 main_proj/main_norm → 唯一 main_x 三 stage 分发；
     noise+seed 首列 → 共享 embed 查表 → HC 展开 → stage0 草稿 h；
   - 公共 stage 模板三实例（B2）：mtp.0/mtp.1/mtp.2 独立节点/张量/边/
     状态命名空间（参数路径 mtp.{i}.*、state_id dspark:win_cache:mtp.{i}
     实例唯一），同一模板结构（逐实例节点多重集相等，机器可检）；
   - 末出口 exit（B3b head）+ Markov（B3b）+ Confidence（B3b）：
     stage2 h_out → hc_head → pre-norm x → norm → 共享 head full_logits
     → Markov K=5 递推（output_ids/logits 显式版本链）→ Confidence
     两来源（exit:prenorm_x + mk:markov_embed）；
   - forward_head 返回三元组 output_ids[B,K+1] / logits[B,K,V] /
     confidence[B,K] 为同一集成图的三 个 graph_output，端口/张量可联查。
2. 跨图连接 = 14 条真实改接边（引用合法、源/目标/张量/kind/guard 逐条
   记录在连接目录，可由图自身证伪）：
   - main_x 分发 ×3：entry.stage_feed_{i}.main_x_out → mtp{i} 注意力
     main 输入（两相位，无 guard）；
   - 草稿 h 进 stage0（decode 骨架 g_decode / prefill passthrough
     g_prefill 两边）与 stage 串接 s0→s1→s2（decode 真实 h 链、prefill
     passthrough 链分开四条）；
   - stage2 → exit h_in（g_decode）；exit.head → Markov 初始 logits；
     entry.seed → Markov output_ids 首列；exit:prenorm_x → Confidence
     hidden；mk:markov_embed → Confidence markov_embed。
   组件边界 graph_input/graph_output 节点（stage.main_x_src、
   stage.x_hc_in_src、exit.h_out_src、exit.logits_out、mk.seed_src、
   mk.init_logits_src、mk.embed_out、cf.hidden_src、cf.embed_src）在
   集成图中移除，其消费/生产边改接到真实跨图源/汇——替换记录在
   integration_doc.dropped_boundaries，可联查。
3. 集成图整体外边界（闭世界，机器可检）：graph_input 恰 4 个
   （entry.input_h_40/41/42 主干读出 + entry.seed_src 种子 token），
   graph_output 恰 3 个（三元组）。prefill 末 stage h_out（933-934
   提前 return，无 head 消费）如实保留为无消费输出端口，不画假边。
4. 相位：decode 专用链（stage 双 mHC 骨架、草稿 Q/KV、sparse_attn、
   exit/markov/confidence）全部挂可求值 g_decode（start_pos > 0）；
   prefill passthrough 链挂 g_prefill；main_x 分发/readout/embed 两相位
   无条件（forward_spec 930 先 forward_embed、931-932 loop 两相位执行、
   933-934 prefill 在 loop 后 return）。
5. 轴合并：六图同名轴定义逐字典一致才合并（探针核对 35 轴）；唯一已知
   冲突 C1（stage=hc_pre keepdim 单例 / exit=hc_head_scale 单元宽 /
   conf=proj 输出单元宽）按图限定改名 C1_exit / C1_conf（值/语义/证据
   原样保留，改名记录在 integration_doc.axis_qualifications）；其余任何
   未声明冲突一律具名拒绝，不静默改名。
6. 证据合并：六组件注册表逐条合并（evidence_id 重复时逐字段一致才接受，
   不一致具名拒绝）；跨图边补挂两侧组件已注册证据（不发明新证据）。

不做（本模块不声称）：场景状态实例化（build_pipeline 落盘层）、集成契约
受控错误（checks 层）、阅读页/CSV（render/export 层）、CLI 接入（cli 层）、
HF 实现覆盖声明（HF 无 DSpark 类，named diff 在 integration_doc）、
kernel 数值/实际 token/采样/置信度数值（unknown 保持）。

可调用入口：
- build_dspark_integration_graph() → DSparkIntegrationBundle（集成图 +
  合并证据 + ModelValues + 14 连接记录 + 五组件映射文档 + 集成文档）
- integration_document(bundle) → dict（data/dspark/integration.json 落盘
  结构：连接目录/实例命名空间/边界闭世界/轴限定/HF named diff/组件映射）

失败异常接口（以实际接口为准）：证据核对失败抛
structure_reader.extract.evidence.EvidenceError；合并冲突、改接目标不存在、
validate_graph_references 未通过、集成文档结构不合法抛
structure_reader.extract.deepseek_v4_reference.ExtractionError
（ValueError 子类）。均具名拒绝、失败不给图。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_reference import ExtractionError
from structure_reader.extract.dspark_confidence import (
    build_confidence_map,
    build_dspark_confidence_graph,
)
from structure_reader.extract.dspark_endpoints import (
    build_dspark_entry_graph,
    build_entry_bindings,
)
from structure_reader.extract.dspark_exit import (
    build_dspark_exit_graph,
    build_exit_role_map,
)
from structure_reader.extract.dspark_markov import (
    build_dspark_markov_graph,
    build_markov_step_map,
)
from structure_reader.extract.dspark_stage import (
    build_chain_map,
    build_dspark_stage_template,
)
from structure_reader.extract.evidence import (
    EvidenceBase,
    EvidenceRegistry,
    load_evidence_base,
)
from structure_reader.graph_document import SemanticGraph
from structure_reader.graph_ir import (
    AliasRelation,
    Edge,
    FoldBoundaryMapping,
    FoldGroup,
    Guard,
    NamedAxis,
    Node,
    Port,
    StateTransition,
    StateVersion,
    Tensor,
)
from structure_reader.model_registry import find_workspace_root
from structure_reader.rules.deepseek_v4_flash_0731 import (
    ModelValues,
    load_model_values,
)

__all__ = [
    "TEMPLATE_ID",
    "IntegrationConnection",
    "DSparkIntegrationBundle",
    "build_dspark_integration_graph",
    "integration_document",
    "merge_evidence_registries",
    "dspark_role_bindings",
]

TEMPLATE_ID = "deepseek_v4_flash_0731:dspark_integration"

# stage 实例数与参数路径（与 B2 derive_stage_bindings 同源；固定 3 实例）
_N_STAGES = 3
_STAGE_PARAM_PATHS = tuple(f"mtp.{i}" for i in range(_N_STAGES))
_STAGE_PREFIXES = tuple(f"mtp{i}" for i in range(_N_STAGES))
_STAGE_STATE_IDS = tuple(
    f"dspark:win_cache:{path}" for path in _STAGE_PARAM_PATHS)


# ---------------------------------------------------------------------------
# 跨图连接记录（每条 = 一次真实改接；src/dst/tensor/kind/guard 与图边
# 一一对应，可由图自身证伪）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntegrationConnection:
    """一条跨图连接：组件边界到真实跨图源/汇的改接记录。"""

    connection_id: str
    src_node: str
    src_port: str
    src_tensor: str
    src_template: str
    dst_node: str
    dst_port: str
    dst_tensor: str
    dst_template: str
    edge_id: str          # 集成图中承载本连接的真实边 stable_id
    kind: str
    guard_ref: str | None
    phase: str            # both | decode | prefill
    evidence: tuple[str, ...]
    note: str

    def to_dict(self) -> dict:
        return {
            "connection_id": self.connection_id,
            "src": {"node": self.src_node, "port": self.src_port,
                    "tensor": self.src_tensor, "template": self.src_template},
            "dst": {"node": self.dst_node, "port": self.dst_port,
                    "tensor": self.dst_tensor, "template": self.dst_template},
            "edge_id": self.edge_id,
            "kind": self.kind,
            "guard_ref": self.guard_ref,
            "phase": self.phase,
            "evidence": list(self.evidence),
            "note": self.note,
        }


# 组件模板 ID（只读引用各组件模块的 template_id 常量，不复制字符串）
_T_ENTRY = "deepseek_v4_flash_0731:dspark_entry"
_T_STAGE = "deepseek_v4_flash_0731:dspark_stage"
_T_EXIT = "deepseek_v4_flash_0731:dspark_exit"
_T_MARKOV = "deepseek_v4_flash_0731:dspark_markov"
_T_CONF = "deepseek_v4_flash_0731:dspark_confidence"


def _connections() -> tuple[IntegrationConnection, ...]:
    """14 条跨图连接目录（与 _rewire_table 一一对应）。"""
    conns: list[IntegrationConnection] = []
    # main_x 唯一生产（entry.main_norm）三 stage 分发（两相位无条件）
    for i in range(_N_STAGES):
        conns.append(IntegrationConnection(
            connection_id=f"conn.main_x.s{i}",
            src_node="entry.stage_feed_%d" % i, src_port="main_x_out",
            src_tensor="tensor:entry:main_x", src_template=_T_ENTRY,
            dst_node=f"mtp{i}.dspark.x_main_in_src", dst_port="x",
            dst_tensor="tensor:entry:main_x", dst_template=_T_STAGE,
            edge_id=f"mtp{i}.e_st_mainx_xsrc", kind="data", guard_ref=None,
            phase="both",
            evidence=("ev_st_forward_spec", "ev_st_main_proj"),
            note="同一 main_x 第三实参透传 stage i（forward_spec 931-932）；"
                 "唯一生产 mtp.0.main_proj/main_norm（entry 图）",
        ))
    # 草稿 h 进 stage0：decode 骨架 / prefill passthrough 两边
    conns.append(IntegrationConnection(
        connection_id="conn.h.entry.s0",
        src_node="entry.stage_feed_0", src_port="h_out",
        src_tensor="tensor:entry:entry_h", src_template=_T_ENTRY,
        dst_node="mtp0.stage.hc_pre_attn.flatten", dst_port="x",
        dst_tensor="tensor:entry:entry_h", dst_template=_T_STAGE,
        edge_id="mtp0.e_st_hcin_hcpre", kind="data", guard_ref="g_decode",
        phase="decode",
        evidence=("ev_st_forward_spec", "ev_st_forward_embed"),
        note="forward_embed 产出的 HC 展开流进 stage0 decode 骨架"
             "（非原始 transformer.embed.y）",
    ))
    conns.append(IntegrationConnection(
        connection_id="conn.h.entry.s0.prefill",
        src_node="entry.stage_feed_0", src_port="h_out",
        src_tensor="tensor:entry:entry_h", src_template=_T_ENTRY,
        dst_node="mtp0.stage.prefill_passthrough", dst_port="x",
        dst_tensor="tensor:entry:entry_h", dst_template=_T_STAGE,
        edge_id="mtp0.e_st_prefill_in", kind="data", guard_ref="g_prefill",
        phase="prefill",
        evidence=("ev_st_forward_spec", "ev_ds_block_prefill"),
        note="prefill（start_pos==0）h 进 stage0 passthrough"
             "（DSparkBlock.forward 845-849 只建 cache）",
    ))
    for i in (1, 2):
        conns.append(IntegrationConnection(
            connection_id=f"conn.h.chain.s{i - 1}.s{i}",
            src_node=f"mtp{i - 1}.stage.hc_post_ffn.combine",
            src_port="y", src_tensor=f"tensor:mtp{i - 1}.stage:x_hc_out",
            src_template=_T_STAGE,
            dst_node=f"mtp{i}.stage.hc_pre_attn.flatten", dst_port="x",
            dst_tensor=f"tensor:mtp{i - 1}.stage:x_hc_out",
            dst_template=_T_STAGE,
            edge_id=f"mtp{i}.e_st_hcin_hcpre", kind="data",
            guard_ref="g_decode", phase="decode",
            evidence=("ev_st_forward_spec", "ev_ds_block_prefill"),
            note=f"decode 草稿 h 串接 stage{i - 1}→{i}"
                 "（for layer in self.mtp: h = layer(h, ...)）",
        ))
        conns.append(IntegrationConnection(
            connection_id=f"conn.h.chain.s{i - 1}.s{i}.prefill",
            src_node=f"mtp{i - 1}.stage.prefill_passthrough",
            src_port="y",
            src_tensor=f"tensor:mtp{i - 1}.stage:x_hc_prefill_out",
            src_template=_T_STAGE,
            dst_node=f"mtp{i}.stage.prefill_passthrough", dst_port="x",
            dst_tensor=f"tensor:mtp{i - 1}.stage:x_hc_prefill_out",
            dst_template=_T_STAGE,
            edge_id=f"mtp{i}.e_st_prefill_in", kind="data",
            guard_ref="g_prefill", phase="prefill",
            evidence=("ev_st_forward_spec", "ev_ds_block_prefill"),
            note=f"prefill passthrough 链 stage{i - 1}→{i}（各 stage "
                 "prefill 分支返回原 h，不算草稿双 mHC 计算）",
        ))
    # 末出口：stage2 h_out → exit hc_head（仅 decode；933-934 prefill
    # 提前 return，stage2 prefill h_out 无消费如实保留）
    conns.append(IntegrationConnection(
        connection_id="conn.exit.h_in",
        src_node="mtp2.stage.hc_post_ffn.combine", src_port="y",
        src_tensor="tensor:mtp2.stage:x_hc_out", src_template=_T_STAGE,
        dst_node="exit.hc_head.flatten", dst_port="x",
        dst_tensor="tensor:mtp2.stage:x_hc_out", dst_template=_T_EXIT,
        edge_id="e_ex_hin_flat", kind="data", guard_ref="g_decode",
        phase="decode",
        evidence=("ev_st_forward_spec", "ev_st_forward_head"),
        note="末 stage h_out 进 forward_head（mtp[-1] 实参，935）",
    ))
    # Markov：初始 logits 来自共享 head；种子列来自 entry.seed
    conns.append(IntegrationConnection(
        connection_id="conn.markov.init_logits",
        src_node="exit.head", src_port="y",
        src_tensor="tensor:exit:logits", src_template=_T_EXIT,
        dst_node="mk.step0.slice", dst_port="logits",
        dst_tensor="tensor:exit:logits", dst_template=_T_MARKOV,
        edge_id="e_mk_s0_slice_read", kind="data", guard_ref="g_decode",
        phase="decode",
        evidence=("ev_mk_loop", "ev_ex_parallel_head"),
        note="forward_head 863 head full_logits=True 返回的初始 logits "
             "[B,K,V] 直接进 Markov 第 0 步切片（867-868）；in-place 更新"
             "由 mk:logits 版本链承载",
    ))
    conns.append(IntegrationConnection(
        connection_id="conn.markov.seed",
        src_node="entry.seed_src", src_port="y",
        src_tensor="tensor:entry:seed_ids", src_template=_T_ENTRY,
        dst_node="mk.output_ids_init", dst_port="seed",
        dst_tensor="tensor:entry:seed_ids", dst_template=_T_MARKOV,
        edge_id="e_mk_seed_init", kind="data", guard_ref="g_decode",
        phase="decode",
        evidence=("ev_mk_boundary_premise", "ev_st_forward_head"),
        note="output_ids 第 0 列 = input_ids（864-865）＝entry 种子 token"
             "（与首列写同一来源）",
    ))
    # Confidence 两来源（真实绑定，非 exit:norm_x / 非采样后 embed）
    conns.append(IntegrationConnection(
        connection_id="conn.conf.hidden",
        src_node="exit.hc_head.reduce", src_port="y",
        src_tensor="tensor:exit:prenorm_x", src_template=_T_EXIT,
        dst_node="cf.concat", dst_port="hidden",
        dst_tensor="tensor:exit:prenorm_x", dst_template=_T_CONF,
        edge_id="e_cf_hidden_in", kind="data", guard_ref="g_decode",
        phase="decode",
        evidence=("ev_cf_call", "ev_ex_hc_head"),
        note="hidden = hc_head 输出 pre-norm x（862，863 norm 之前）——"
             "非 exit:norm_x、非普通 stage h_out",
    ))
    conns.append(IntegrationConnection(
        connection_id="conn.conf.markov_embed",
        src_node="mk.stack_embed", src_port="stacked",
        src_tensor="tensor:mk:markov_embed", src_template=_T_MARKOV,
        dst_node="cf.concat", dst_port="embed",
        dst_tensor="tensor:mk:markov_embed", dst_template=_T_CONF,
        edge_id="e_cf_embed_in", kind="data", guard_ref="g_decode",
        phase="decode",
        evidence=("ev_mk_embed_stack", "ev_cf_call"),
        note="markov_embed = 872 五步各自当前输入 token 查表 embed 的有序"
             " stack——非本步新采样 token 查表替代",
    ))
    return tuple(conns)


# 集成图移除的组件边界节点（替换记录，可联查）与移除的组件边界边
_DROPPED_NODES = (
    *(f"mtp{i}.stage.main_x_src" for i in range(_N_STAGES)),
    *(f"mtp{i}.stage.x_hc_in_src" for i in range(_N_STAGES)),
    "exit.h_out_src", "exit.logits_out",
    "mk.seed_src", "mk.init_logits_src", "mk.embed_out",
    "cf.hidden_src", "cf.embed_src",
)
_DROPPED_EDGES = ("e_ex_logits_out", "e_mk_stack_out")
_DROPPED_TEMPLATES = {
    "mtp{i}.stage.main_x_src": "stage.main_x_src（B2 模板 graph_input，由"
                               " conn.main_x.s{i} 替换）",
    "mtp{i}.stage.x_hc_in_src": "stage.x_hc_in_src（B2 模板 graph_input，由"
                                " conn.h.entry.s0 / conn.h.chain.* 替换）",
    "exit.h_out_src": "exit graph_input（B3b 外部边界，由 conn.exit.h_in "
                      "替换）",
    "exit.logits_out": "exit graph_output（B3b exit:logits 边界，并入 "
                       "conn.markov.init_logits 与三元组 mk.logits_out）",
    "mk.seed_src": "markov graph_input（B3b 外部边界，由 conn.markov.seed "
                   "替换）",
    "mk.init_logits_src": "markov graph_input（B3b 外部边界，由 "
                          "conn.markov.init_logits 替换）",
    "mk.embed_out": "markov graph_output（B3b mk:markov_embed 边界，由 "
                    "conn.conf.markov_embed 替换）",
    "cf.hidden_src": "confidence graph_input（B3b 外部边界，由 "
                     "conn.conf.hidden 替换）",
    "cf.embed_src": "confidence graph_input（B3b 外部边界，由 "
                    "conn.conf.markov_embed 替换）",
}

# 集成图闭世界外边界
_EXTERNAL_INPUTS = (
    ("entry.input_h_40", "y", "tensor:entry:src_h_40",
     "主干层 40 hc 流 h [B,T,N_hc,H]（917-921 readout 边界）"),
    ("entry.input_h_41", "y", "tensor:entry:src_h_41",
     "主干层 41 hc 流 h [B,T,N_hc,H]"),
    ("entry.input_h_42", "y", "tensor:entry:src_h_42",
     "主干层 42 hc 流 h [B,T,N_hc,H]"),
    ("entry.seed_src", "y", "tensor:entry:seed_ids",
     "种子 token [B]（主干 hc_head 采样，forward_spec input_ids）"),
)
_TRIPLE_OUTPUTS = (
    ("mk.output_ids_out", "ids", "tensor:mk:output_ids",
     "output_ids [B,K+1]（forward_head 874 返回第 1 项；v5 终版）"),
    ("mk.logits_out", "x", "tensor:mk:logits",
     "logits [B,K,V]（forward_head 874 返回第 2 项；in-place 更新 v5 终版）"),
    ("cf.confidence_out", "x", "tensor:cf:confidence",
     "confidence [B,K]（forward_head 874 返回第 3 项）"),
)


# ---------------------------------------------------------------------------
# 轴合并（同名轴逐字典一致才合并；唯一已知冲突 C1 按图限定改名）
# ---------------------------------------------------------------------------

_AXIS_CONFLICT_POLICY = {
    # symbol → {template_id 片段 → 限定名}；不在表内的冲突一律拒绝
    "C1": {"dspark_exit": "C1_exit", "dspark_confidence": "C1_conf"},
}


def _merge_axes(graph_axes: list[tuple[str, dict[str, NamedAxis]]]
                ) -> tuple[list[NamedAxis], dict[str, dict[str, str]]]:
    """按模板顺序合并轴字典；返回 (合并轴列表, {template: {旧symbol: 新symbol}})。

    同名轴逐 to_dict 字典一致 → 合并共享；差异若在冲突策略表内 → 按表
    限定改名；其余差异具名拒绝（不静默改名）。
    """
    merged: dict[str, NamedAxis] = {}
    renames: dict[str, dict[str, str]] = {}
    for template_id, axes in graph_axes:
        renames[template_id] = {}
        for symbol, axis in axes.items():
            if symbol not in merged:
                merged[symbol] = axis
                continue
            if axis.to_dict() == merged[symbol].to_dict():
                continue
            policy = _AXIS_CONFLICT_POLICY.get(symbol)
            tag = None
            if policy:
                for frag, qualified in policy.items():
                    if frag in template_id:
                        tag = qualified
                        break
            if tag is None:
                raise ExtractionError(
                    f"轴 {symbol} 在 {template_id} 与已合并定义不一致且无"
                    f"冲突策略: {axis.to_dict()} vs {merged[symbol].to_dict()}")
            renames[template_id][symbol] = tag
            merged[tag] = NamedAxis(
                symbol=tag, name=axis.name, meaning=axis.meaning,
                origin=axis.origin, evidence_ref=axis.evidence_ref,
                value=axis.value, expression=axis.expression,
                unknown_reason=axis.unknown_reason,
            )
    order = {a.symbol: idx for idx, a in enumerate(
        _AXIS_ORDER_HINT) if a in merged} if False else None
    return list(merged.values()), renames


_AXIS_ORDER_HINT: tuple[str, ...] = ()  # 占位（保持合并确定性由构建顺序保证）


# ---------------------------------------------------------------------------
# stage 实例命名空间改写（B2 模板 → mtp{i} 实例）
# ---------------------------------------------------------------------------


def _instantiate_stage(stage: SemanticGraph, i: int
                       ) -> tuple[SemanticGraph, dict[str, str]]:
    """stage 模板 → 第 i 实例：节点/边/张量/状态/别名/折叠全部加 mtp{i}
    命名空间；guard 共享不改。返回 (实例图, 张量映射)。"""
    n = f"mtp{i}"

    def node_id(nid: str) -> str:
        return f"{n}.{nid}"

    def tensor_id(tid: str) -> str:
        if not tid.startswith("tensor:stage:"):
            raise ExtractionError(
                f"stage 模板张量名前缀意外: {tid!r}（实例化中止，不静默合并）")
        return f"tensor:{n}.stage:{tid[len('tensor:stage:'):]}"  # noqa: ECE001

    tensors = [
        Tensor(name=t.stable_id[len("tensor:"):].__str__().replace(
            "stage:", f"{n}.stage:", 1)
            if t.name.startswith("stage:") else t.name,
            dtype=t.dtype, axes=tuple(t.axes),
            logical_layout=t.logical_layout, physical_layout=t.physical_layout,
            global_rank=t.global_rank, local_rank=t.local_rank)
        for t in stage.tensors
    ]
    tmap = {old.stable_id: new.stable_id
            for old, new in zip(stage.tensors, tensors)}
    nodes = [
        Node(stable_id=node_id(x.stable_id), op_type=x.op_type,
             granularity=x.granularity,
             input_ports=tuple(Port(p.name, p.direction, tmap[p.tensor_id])
                               for p in x.input_ports),
             output_ports=tuple(Port(p.name, p.direction, tmap[p.tensor_id])
                                for p in x.output_ports),
             evidence_refs=tuple(x.evidence_refs))
        for x in stage.nodes
    ]
    edges = []
    for e in stage.edges:
        tr = e.state_transition_ref
        edges.append(Edge(
            stable_id=f"{n}.{e.stable_id}",
            src_node_id=node_id(e.src_node_id), src_port=e.src_port,
            dst_node_id=node_id(e.dst_node_id), dst_port=e.dst_port,
            tensor_id=tmap[e.tensor_id], kind=e.kind,
            evidence_refs=tuple(e.evidence_refs), guard_ref=e.guard_ref,
            state_transition_ref=(f"tr_dspark_cache_{n}"
                                  if tr == "tr_dspark_cache" else tr)))
    versions = [
        StateVersion(
            stable_id=v.stable_id.replace("dspark:win_cache",
                                          f"dspark:win_cache:mtp.{i}"),
            state_id=v.state_id.replace("dspark:win_cache",
                                        f"dspark:win_cache:mtp.{i}"),
            version_label=v.version_label)
        for v in stage.state_versions
    ]
    transitions = [
        StateTransition(
            stable_id=f"tr_dspark_cache_{n}" if tr.stable_id == "tr_dspark_cache"
            else tr.stable_id,
            state_id=tr.state_id.replace("dspark:win_cache",
                                         f"dspark:win_cache:mtp.{i}"),
            before_version_id=tr.before_version_id.replace(
                "dspark:win_cache", f"dspark:win_cache:mtp.{i}"),
            after_version_id=tr.after_version_id.replace(
                "dspark:win_cache", f"dspark:win_cache:mtp.{i}"),
            evidence_refs=tuple(tr.evidence_refs))
        for tr in stage.state_transitions
    ]
    aliases = [
        AliasRelation(
            stable_id=f"{a.stable_id}_{n}" if i else a.stable_id,
            base_tensor_id=tmap[a.base_tensor_id],
            alias_tensor_id=tmap[a.alias_tensor_id]
            if a.alias_tensor_id else None,
            relation_kind=a.relation_kind, alias_role=a.alias_role,
            transform_expr=a.transform_expr,
            evidence_refs=tuple(a.evidence_refs))
        for a in stage.alias_relations
    ]
    folds = [
        FoldGroup(
            stable_id=f"{fg.stable_id}_{n}",
            member_node_ids=[node_id(m) for m in fg.member_node_ids],
            boundary_mappings=[
                FoldBoundaryMapping(
                    direction=bm.direction, external_port=bm.external_port,
                    edge_kind=bm.edge_kind,
                    internal_node_id=node_id(bm.internal_node_id),
                    internal_port=bm.internal_port,
                    evidence_refs=tuple(bm.evidence_refs))
                for bm in fg.boundary_mappings])
        for fg in stage.fold_groups
    ]
    inst = SemanticGraph(
        template_id=stage.template_id, named_axes=stage.named_axes,
        tensors=tensors, nodes=nodes, edges=edges,
        state_versions=versions, state_transitions=transitions,
        guards=stage.guards, guard_activations=(),
        alias_relations=aliases, fold_groups=folds)
    return inst, tmap


# ---------------------------------------------------------------------------
# 改接表：边 → 新 src/dst/张量；节点端口张量改写；移除集合
# ---------------------------------------------------------------------------


def _rewire_table() -> dict[str, dict]:
    """14 条改接的机器可执行规格（与 _connections 一一对应）。

    每项：edge（改写后的边 stable_id）→ {src?, dst?, tensor?, ports?, ev_add?}。
    ports：[(node_id, port_name, 新 tensor_id)]（边与两端端口三方一致）。
    """
    table: dict[str, dict] = {}
    for i in range(_N_STAGES):
        table[f"mtp{i}.e_st_mainx_xsrc"] = {
            "src": ("entry.stage_feed_%d" % i, "main_x_out"),
            "tensor": "tensor:entry:main_x",
            "ports": [(f"mtp{i}.dspark.x_main_in_src", "x",
                       "tensor:entry:main_x")],
        }
    table["mtp0.e_st_hcin_hcpre"] = {
        "src": ("entry.stage_feed_0", "h_out"),
        "tensor": "tensor:entry:entry_h",
        "ports": [("mtp0.stage.hc_pre_attn.flatten", "x",
                   "tensor:entry:entry_h")],
    }
    table["mtp0.e_st_prefill_in"] = {
        "src": ("entry.stage_feed_0", "h_out"),
        "tensor": "tensor:entry:entry_h",
        "ports": [("mtp0.stage.prefill_passthrough", "x",
                   "tensor:entry:entry_h")],
    }
    for i in (1, 2):
        table[f"mtp{i}.e_st_hcin_hcpre"] = {
            "src": (f"mtp{i - 1}.stage.hc_post_ffn.combine", "y"),
            "tensor": f"tensor:mtp{i - 1}.stage:x_hc_out",
            "ports": [(f"mtp{i}.stage.hc_pre_attn.flatten", "x",
                       f"tensor:mtp{i - 1}.stage:x_hc_out")],
        }
        table[f"mtp{i}.e_st_prefill_in"] = {
            "src": (f"mtp{i - 1}.stage.prefill_passthrough", "y"),
            "tensor": f"tensor:mtp{i - 1}.stage:x_hc_prefill_out",
            "ports": [(f"mtp{i}.stage.prefill_passthrough", "x",
                       f"tensor:mtp{i - 1}.stage:x_hc_prefill_out")],
        }
    table["e_ex_hin_flat"] = {
        "src": ("mtp2.stage.hc_post_ffn.combine", "y"),
        "tensor": "tensor:mtp2.stage:x_hc_out",
        "ports": [("exit.hc_head.flatten", "x", "tensor:mtp2.stage:x_hc_out")],
    }
    table["e_mk_s0_slice_read"] = {
        "src": ("exit.head", "y"),
        "tensor": "tensor:exit:logits",
        "ports": [("mk.step0.slice", "logits", "tensor:exit:logits")],
        "ev_add": ("ev_ex_parallel_head",),
    }
    table["e_mk_seed_init"] = {
        "src": ("entry.seed_src", "y"),
        "tensor": "tensor:entry:seed_ids",
        "ports": [("mk.output_ids_init", "seed", "tensor:entry:seed_ids")],
        "ev_add": ("ev_st_forward_head",),
    }
    table["e_cf_hidden_in"] = {
        "src": ("exit.hc_head.reduce", "y"),
        "tensor": "tensor:exit:prenorm_x",
        "ports": [("cf.concat", "hidden", "tensor:exit:prenorm_x")],
        "ev_add": ("ev_ex_hc_head",),
    }
    table["e_cf_embed_in"] = {
        "src": ("mk.stack_embed", "stacked"),
        "tensor": "tensor:mk:markov_embed",
        "ports": [("cf.concat", "embed", "tensor:mk:markov_embed")],
        "ev_add": ("ev_mk_embed_stack",),
    }
    # main_x 改接边补挂跨图锚点证据
    for i in range(_N_STAGES):
        table[f"mtp{i}.e_st_mainx_xsrc"]["ev_add"] = (
            "ev_st_forward_spec", "ev_st_main_proj")
    for eid in ("mtp0.e_st_hcin_hcpre", "mtp0.e_st_prefill_in"):
        table[eid].setdefault("ev_add", ())
    return table


# ---------------------------------------------------------------------------
# 合并证据注册表
# ---------------------------------------------------------------------------


def _merge_registries(registries: list[EvidenceRegistry]) -> EvidenceRegistry:
    """六组件证据逐条合并：ID 冲突时逐字段一致才接受，不一致具名拒绝。"""
    return merge_evidence_registries(registries)


def merge_evidence_registries(
        registries: list[EvidenceRegistry]) -> EvidenceRegistry:
    """公开证据注册表合并（DSpark 六组件与主图合并发布共用同一条规则）：
    ID 冲突时逐字段一致才接受（同 ID 必须同语义），不一致具名拒绝——
    不允许后写覆盖为另一语义。"""
    merged = EvidenceRegistry(registries[0]._base, registries[0]._root)
    seen: dict[str, dict] = {}
    for reg in registries:
        for ev in reg.all():
            d = ev.to_dict()
            if ev.evidence_id in seen:
                if seen[ev.evidence_id] != d:
                    raise ExtractionError(
                        f"证据 {ev.evidence_id} 跨组件定义不一致，拒绝合并")
                continue
            seen[ev.evidence_id] = d
            merged.register(ev)
    return merged


def _merge_guards(graphs: list[SemanticGraph]) -> tuple[Guard, ...]:
    """guard 合并：同 ID 同谓词 → 证据并集；同 ID 异谓词具名拒绝。"""
    by_id: dict[str, Guard] = {}
    for g in graphs:
        for guard in g.guards:
            if guard.stable_id in by_id:
                prev = by_id[guard.stable_id]
                if prev.predicate_expr != guard.predicate_expr:
                    raise ExtractionError(
                        f"guard {guard.stable_id} 谓词不一致: "
                        f"{prev.predicate_expr!r} vs {guard.predicate_expr!r}")
                ev = tuple(sorted(set(prev.evidence_refs)
                                  | set(guard.evidence_refs)))
                by_id[guard.stable_id] = Guard(
                    stable_id=guard.stable_id,
                    predicate_expr=guard.predicate_expr, evidence_refs=ev)
            else:
                by_id[guard.stable_id] = guard
    return tuple(by_id[sid] for sid in sorted(by_id))


# ---------------------------------------------------------------------------
# 集成图构建
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DSparkIntegrationBundle:
    """B4 集成交付包：整体图 + 合并证据 + 数值 + 连接目录 + 组件映射。"""

    graph: SemanticGraph
    registry: EvidenceRegistry
    values: ModelValues
    connections: tuple[IntegrationConnection, ...]
    chain: object            # dspark_stage.ChainMap
    entry_bindings: object   # dspark_endpoints.EntryBindings
    exit_role_map: dict
    markov_step_map: dict
    confidence_map: dict
    expansion: object        # dspark_stage.StageExpansion


def build_dspark_integration_graph(
    workspace_root: Path | None = None,
    *,
    values: ModelValues | None = None,
    evidence_base: EvidenceBase | None = None,
) -> DSparkIntegrationBundle:
    """构建 DSpark 整体集成图：六组件真实合并 + 14 跨图改接 + 引用检查。

    任何证据核对失败、合并冲突、改接目标不存在或
    validate_graph_references 未通过都抛具名异常——失败不给图。
    """
    root = Path(workspace_root) if workspace_root is not None else find_workspace_root()
    values = values or load_model_values(root)
    evidence_base = evidence_base or load_evidence_base(root)

    entry = build_dspark_entry_graph(root, values=values,
                                     evidence_base=evidence_base)
    stage = build_dspark_stage_template(root, values=values,
                                        evidence_base=evidence_base)
    exitg = build_dspark_exit_graph(root, values=values,
                                    evidence_base=evidence_base)
    markov = build_dspark_markov_graph(root, values=values,
                                       evidence_base=evidence_base)
    conf = build_dspark_confidence_graph(root, values=values,
                                         evidence_base=evidence_base)

    registry = _merge_registries([entry.registry, stage.registry,
                                  exitg.registry, markov.registry,
                                  conf.registry])

    # 轴合并（闭世界冲突策略；stage 三实例同定义共享）
    axis_inputs = [
        (_T_ENTRY, {a.symbol: a for a in entry.graph.named_axes}),
    ]
    axis_inputs.append((_T_STAGE, {a.symbol: a for a in stage.graph.named_axes}))
    axis_inputs.append((_T_EXIT, {a.symbol: a for a in exitg.graph.named_axes}))
    axis_inputs.append((_T_MARKOV, {a.symbol: a
                                    for a in markov.graph.named_axes}))
    axis_inputs.append((_T_CONF, {a.symbol: a for a in conf.graph.named_axes}))
    merged_axes_list, axis_renames = _merge_axes(axis_inputs)
    merged_axes = {a.symbol: a for a in merged_axes_list}
    if len(merged_axes) != len(merged_axes_list):
        raise ExtractionError("合并轴 symbol 重复（限定改名后仍冲突）")

    def axes_for(template_id: str, axes: tuple[NamedAxis, ...]
                 ) -> tuple[NamedAxis, ...]:
        ren = axis_renames.get(template_id, {})
        if not ren:
            return axes
        return tuple(merged_axes[ren.get(a.symbol, a.symbol)] for a in axes)

    # 组件实体合并（entry / stage×3 / exit / markov / conf）
    parts: list[SemanticGraph] = [entry.graph]
    stage_insts = []
    for i in range(_N_STAGES):
        inst, _tmap = _instantiate_stage(stage.graph, i)
        stage_insts.append(inst)
        parts.append(inst)
    parts.append(exitg.graph)
    parts.append(markov.graph)
    parts.append(conf.graph)

    tensors: list[Tensor] = []
    nodes: list[Node] = []
    edges: list[Edge] = []
    versions: list[StateVersion] = []
    transitions: list[StateTransition] = []
    aliases: list[AliasRelation] = []
    folds: list[FoldGroup] = []
    for part, tid in zip(parts, [_T_ENTRY, _T_STAGE, _T_STAGE, _T_STAGE,
                                 _T_EXIT, _T_MARKOV, _T_CONF]):
        ren = axis_renames.get(tid, {})
        for t in part.tensors:
            axes = axes_for(tid, tuple(t.axes))
            layouts = tuple(ren.get(s, s) for s in t.logical_layout), \
                tuple(ren.get(s, s) for s in t.physical_layout)
            tensors.append(Tensor(name=t.name, dtype=t.dtype, axes=axes,
                                  logical_layout=layouts[0],
                                  physical_layout=layouts[1],
                                  global_rank=t.global_rank,
                                  local_rank=t.local_rank))
        nodes.extend(part.nodes)
        edges.extend(part.edges)
        versions.extend(part.state_versions)
        transitions.extend(part.state_transitions)
        aliases.extend(part.alias_relations)
        folds.extend(part.fold_groups)

    guards = _merge_guards(parts)

    graph = SemanticGraph(
        template_id=TEMPLATE_ID, named_axes=merged_axes_list,
        tensors=tensors, nodes=nodes, edges=edges,
        state_versions=versions, state_transitions=transitions,
        guards=guards, guard_activations=(),
        alias_relations=aliases, fold_groups=folds)

    # 改接：重写 14 条边 + 端口张量 + 移除边界节点/边/孤儿张量
    graph = _apply_rewires(graph)

    issues = validate_graph_references(graph)
    if issues:
        raise ExtractionError(
            "集成图通用引用检查未通过: "
            + "; ".join(f"[{x.code}] {x.path}: {x.message}"
                        for x in issues[:8]))

    chain = build_chain_map(root, values=values)
    entry_bindings = build_entry_bindings(values)
    exit_role_map = build_exit_role_map(values)
    markov_step_map = build_markov_step_map(values)
    confidence_map = build_confidence_map(values)

    return DSparkIntegrationBundle(
        graph=graph, registry=registry, values=values,
        connections=_connections(),
        chain=chain, entry_bindings=entry_bindings,
        exit_role_map=exit_role_map, markov_step_map=markov_step_map,
        confidence_map=confidence_map, expansion=stage.expansion)


def _apply_rewires(graph: SemanticGraph) -> SemanticGraph:
    """执行 14 条改接 + 边界节点/边移除 + 孤儿张量清理（具名核对目标存在）。"""
    table = _rewire_table()
    drop_nodes = set(_DROPPED_NODES)
    drop_edges = set(_DROPPED_EDGES)

    node_by_id = {n.stable_id: n for n in graph.nodes}
    for nid in drop_nodes:
        if nid not in node_by_id:
            raise ExtractionError(f"改接目标：待移除边界节点不存在: {nid}")
    # 端口张量改写集合
    port_rw: dict[tuple[str, str], str] = {}
    for spec in table.values():
        for nid, pname, tid in spec.get("ports", ()):
            if nid in drop_nodes:
                raise ExtractionError(f"改接目标端口所在节点被移除: {nid}")
            port_rw[(nid, pname)] = tid
    new_nodes = []
    for n in graph.nodes:
        if n.stable_id in drop_nodes:
            continue
        changed = False
        ins, outs = [], []
        for p in n.input_ports:
            tid = port_rw.get((n.stable_id, p.name), p.tensor_id)
            changed |= tid != p.tensor_id
            ins.append(Port(p.name, p.direction, tid))
        for p in n.output_ports:
            tid = port_rw.get((n.stable_id, p.name), p.tensor_id)
            changed |= tid != p.tensor_id
            outs.append(Port(p.name, p.direction, tid))
        new_nodes.append(
            Node(stable_id=n.stable_id, op_type=n.op_type,
                 granularity=n.granularity, input_ports=tuple(ins),
                 output_ports=tuple(outs),
                 evidence_refs=tuple(n.evidence_refs))
            if changed else n)

    edge_by_id = {e.stable_id: e for e in graph.edges}
    for eid in drop_edges:
        if eid not in edge_by_id:
            raise ExtractionError(f"改接目标：待移除边界边不存在: {eid}")
    new_edges = []
    for e in graph.edges:
        if e.stable_id in drop_edges:
            continue
        spec = table.get(e.stable_id)
        if spec is None:
            new_edges.append(e)
            continue
        src_node, src_port = spec.get("src", (e.src_node_id, e.src_port))
        dst_node, dst_port = spec.get("dst", (e.dst_node_id, e.dst_port))
        tid = spec.get("tensor", e.tensor_id)
        ev = tuple(e.evidence_refs) + tuple(spec.get("ev_add", ()))
        new_edges.append(Edge(
            stable_id=e.stable_id, src_node_id=src_node, src_port=src_port,
            dst_node_id=dst_node, dst_port=dst_port, tensor_id=tid,
            kind=e.kind, evidence_refs=ev, guard_ref=e.guard_ref,
            state_transition_ref=e.state_transition_ref))

    # 移除孤儿张量：只丢因本次改接而失去端口引用的张量（组件图中本就
    # 未被端口引用的张量原样保留，不引入新的清理语义——route_slot 等）。
    pre_used: set[str] = set()
    for n in graph.nodes:
        for p in (*n.input_ports, *n.output_ports):
            pre_used.add(p.tensor_id)
    used: set[str] = set()
    for n in new_nodes:
        for p in (*n.input_ports, *n.output_ports):
            used.add(p.tensor_id)
    kept_tensors, dropped_tensors = [], []
    for t in graph.tensors:
        orphaned = (t.stable_id in pre_used) and (t.stable_id not in used)
        (kept_tensors if not orphaned else dropped_tensors).append(t)
    known_orphans = {
        *(f"tensor:mtp{i}.stage:main_x" for i in range(_N_STAGES)),
        *(f"tensor:mtp{i}.stage:x_hc_in" for i in range(_N_STAGES)),
        "tensor:exit:h_out", "tensor:mk:init_logits", "tensor:mk:seed_ids",
        "tensor:cf:hidden_in", "tensor:cf:markov_embed_in",
    }
    unexpected = {t.stable_id for t in dropped_tensors} - known_orphans
    if unexpected:
        raise ExtractionError(
            f"改接后出现未声明的孤儿张量（拒绝静默丢弃）: {sorted(unexpected)}")

    return SemanticGraph(
        template_id=graph.template_id, named_axes=graph.named_axes,
        tensors=kept_tensors, nodes=new_nodes, edges=new_edges,
        state_versions=graph.state_versions,
        state_transitions=graph.state_transitions,
        guards=graph.guards, guard_activations=(),
        alias_relations=[
            a for a in graph.alias_relations
            if a.base_tensor_id in used and
            (a.alias_tensor_id is None or a.alias_tensor_id in used)],
        fold_groups=graph.fold_groups)


# ---------------------------------------------------------------------------
# DSpark 场景状态视图（prefill 建窗口 cache / decode 草稿 K5 状态实例化）
# ---------------------------------------------------------------------------


def dspark_scenario_views(bundle: DSparkIntegrationBundle,
                          scenario_envs: list[dict]) -> list[dict]:
    """对每个场景 env 真实求值集成图 guard，生成 DSpark 场景状态视图。

    每场景返回 {scenario_id, env, guard_activations, window_cache,
    draft_kv, positions, phase}；guard 三态与坐标由
    :func:`scenario_evaluator.evaluate_tri` 对集成图 guards 原文逐条求值，
    不硬写 passed。prefill（start_pos==0）窗口 cache 构建分支与 decode
    草稿分支由真实活动边推导：capacity=W 槽恒定（r=0 无压缩区），
    valid_length/有效槽位/本次写槽按分支区分；无初始条件保持 unknown。
    """
    from structure_reader.scenario_evaluator import evaluate_tri

    g = bundle.graph
    guard_by_id = {x.stable_id: x for x in g.guards}
    # 每实例 cache 状态身份/版本/迁移（从图实体派生，供场景视图逐实例
    # 绑定 before/after/published 版本）
    tr_by_state = {tr.state_id: tr for tr in g.state_transitions
                   if tr.state_id.startswith("dspark:")}
    views = []
    for env in scenario_envs:
        eval_env = {k: v for k, v in env.items() if k != "scenario_id"}
        acts = {}
        for gid, guard in guard_by_id.items():
            result = evaluate_tri(guard.predicate_expr, eval_env)
            acts[gid] = (
                {"state": "true" if result.value else "false"}
                if result.is_known
                else {"state": "unknown", "reason": result.unknown_reason}
            )
        start_pos = env["start_pos"]
        is_prefill = start_pos == 0
        w = env["W"]
        k = env["N_draft"]
        t = env["T"]
        p = env["P"]
        # before 内容已知性（codex_b4_review02 阻断3 三态拆分）：
        # - before_state_known 一律 "unknown"：无任何实际初始状态证据
        #   （静态分析不验证连续历史），P/T 数值不能转换成 bool 语义——
        #   P+1≥W 只是"本调用写入后的覆盖数量口径"，不能由数量阈值
        #   反推写前历史连续成立（P=127 写后 128 槽不能推出写前 127 槽
        #   已满），prefill T>0 更只是本调用写入数；
        # - after_slot_count 可数：本调用写入后的环形覆盖数量口径
        #   （prefill min(T,W)、decode min(P+1, W)——P+1 是含本调用的
        #   累计写次数，超过 W 时全部槽已被覆盖，有效槽数封顶 W；仅此
        #   用途，不能由它反推写前历史连续成立）；
        # - 连续历史是显式假设（history_assumption），unknown 不冒充。
        after_slot_count = (min(t, w) if is_prefill
                            else min(p + 1, w))
        stage_states = []
        for i, path in enumerate(_STAGE_PARAM_PATHS):
            tr = tr_by_state[f"dspark:win_cache:{path}"]
            state_id = f"dspark:win_cache:{path}"
            stage_states.append({
                "stage_id": i,
                "param_path": path,
                "state_id": state_id,
                "cache_tensor": f"tensor:mtp{i}.stage:win_cache",
                "transition_id": tr.stable_id,
                "before_version_id": tr.before_version_id,
                "after_version_id": tr.after_version_id,
                "published_version_id": tr.after_version_id,
                "write_branch": (
                    f"prefill：整段建窗（{'顺序' if t <= w else '回绕'}写）；"
                    if is_prefill else
                    f"decode：单槽写 slot {p % w}（{p} % {w}）"),
                "read_relation": (
                    "published 后读取：本调用写槽并入 v_after，本调用内"
                    "读窗为 v_hist∪本步写入（同一 state_id 前后版本）"
                    if not is_prefill else
                    "prefill 只建窗不读旧内容（DSparkBlock 845-849）"),
                "before_state_known": "unknown",
                "before_state_note": (
                    "before 内容 known/unknown 无实际初始状态证据"
                    "（unknown 保持，不转 bool）：连续历史是显式假设"
                    "（见 history_assumption），本调用写入后 "
                    f"{'min(T, W)=' + str(min(t, w)) if is_prefill else 'P+1=' + str(p + 1)}"
                    " 槽可数（覆盖数量口径），写前内容 unknown"),
                "after_slot_count": after_slot_count,
            })
        if is_prefill:
            inwin = t <= w
            window_cache = {
                "capacity_slots": w,
                "compress_zone_slots": 0,
                "note": "DSpark stage 层 compress_ratio==0（model.py:753）："
                        "cache 容量=窗口 W 槽，无压缩区",
                "branch": ("sequential" if inwin else "wraparound"),
                "branch_evidence": ("model.py:765" if inwin
                                    else "model.py:767-768"),
                "written_slots_this_call": (
                    list(range(t)) if inwin else list(range(w))),
                "wrap_note": None if inwin else
                f"回绕写：末 W={w} 个 token split([{w - (t % w)}, {t % w}])，"
                f"前 {w - (t % w)} 个进槽 {t % w}..{w - 1}、"
                f"末 {t % w} 个进槽 0..{t % w - 1}（model.py:767-768，"
                "cutoff=T%W）；本次调用覆盖全部 W 槽",
                # 数值与前提分开：written 本调用可数；valid_length 数值
                # 只在连续历史假设下成立，before 内容 unknown（不当作空）
                "valid_slot_count": (
                    min(t, w) if is_prefill else None),
                "valid_length_state": "unknown" if is_prefill else None,
                "valid_length_note": (
                    f"本次写入 min(T, W)={min(t, w)} 槽；此前内容 unknown"
                    "（无初始状态证据，不当作空；连续历史为显式假设）"
                    if is_prefill else "见 decode 分支"),
                "unknown_note": "无初始条件：调用前槽位内容 unknown（不当作空）",
            }
            draft_kv = {
                "active": False,
                "note": "prefill 在 769 早退（整条草稿链 g_decode 屏蔽）："
                        "无草稿 Q/KV、无选择索引、无 sparse_attn",
            }
            positions = {
                "main_rope": "P..P+T-1（start_pos=0：0..T-1）",
                "draft_rope": None,
                "slot_write": None,
                "note": "prefill 无草稿位置/无单槽坐标（771 起于相位分支后）",
            }
        else:
            sel_width = f"min(W, P + 1) + N_draft = min({w}, {p + 1}) + {k}"
            sel_value = min(w, p + 1) + k
            # P+1 仅用于 after 有效数量口径（写后可推出）；连续历史本身
            # 是显式假设（history_assumption），不由数量阈值证明
            premise_ok = p + 1 >= w
            window_cache = {
                "capacity_slots": w,
                "compress_zone_slots": 0,
                "branch": "decode_single_slot",
                "branch_evidence": "model.py:783",
                "written_slots_this_call": [p % w],
                # 数值与前提分开：premise_ok（P+1≥W 数量阈值满足）时 after
                # 有效项数可数（若连续历史假设成立）；未满足前提明确
                # unknown，不冒充满窗也不当作空
                "valid_slot_count": w if premise_ok else p + 1,
                "valid_length_state": ("full_premise" if premise_ok
                                       else "partial"),
                "valid_length_premise": (
                    "连续历史显式假设（本图不验证）：若历史 P 次调用已按"
                    f"同一窗口规则连续写入，则 P+1≥W（P={p} 时 P+1={p + 1}，"
                    "W=" + str(w) + "）数量阈值满足——本调用写入后全部 W 槽"
                    "被环形覆盖（覆盖数量按 P+1 而非 P；P+1 仅用于 after "
                    "有效数量，不证明写前历史；before 内容仍 unknown）"
                    if premise_ok else
                    "连续历史显式假设（本图不验证）：P+1≥W 数量阈值未满足"
                    f"（P+1={p + 1} < W={w}）——after 仅 {p + 1} 槽可数"
                    "（含本调用写入），其余槽内容 unknown（不当作空也不"
                    "当作满；before 内容 unknown）"),
                "valid_length": (
                    f"满 W={w} 槽有效（显式假设 + 数量阈值 P+1={p + 1} ≥ W；"
                    "after 口径）"
                    if premise_ok else
                    f"after {p + 1} 槽确定有效，其余 W-{p + 1} 槽内容 unknown"),
                "unknown_note": "cache 实际内容 unknown（连续历史显式假设，"
                                "本图不验证数值）",
                # 物理宽（W+N_draft）与选择有效宽（min(W,P+1)+N_draft）分开
                "physical_concat_width": w + k,
                "physical_concat_note": (
                    f"decode KV concat 物理宽 W+N_draft={w + k}"
                    "（model.py:784）：主窗口与草稿 KV 拼接——是物理布局"
                    "宽，不是有效选择数"),
                "select_valid_width": sel_value,
                "select_valid_expr": sel_width,
                "select_valid_note": (
                    f"每 query 有效选择宽 min(W,P+1)+N_draft={sel_value}"
                    "（model.py:746）；物理 concat 宽 W+N_draft 与之分开，"
                    "不得混用"),
            }
            draft_kv = {
                "active": True,
                "k_steps": k,
                "select_width_expr": "min(W, P + 1) + N_draft",
                "select_width_value": sel_value,
                "no_minus_one_padding": True,
                "note": "选择索引无 -1 无效填充（与 get_window_topk_idxs "
                        "不同）；宽随 P 变化，非固定值（model.py:746）",
            }
            positions = {
                "main_rope": f"P..P+T-1（{p}..{p + t - 1}）",
                "draft_rope": f"P+T..P+T+N_draft-1（{p + t}..{p + t + k - 1}）",
                "slot_write": f"P % W = {p} % {w} = {p % w}",
                "position_offset": start_pos + t,
                "note": "草稿位置 offset = start_pos+seqlen（model.py:772）",
            }
        views.append({
            "scenario_id": env["scenario_id"],
            "env": dict(env),
            "guard_activations": acts,
            "phase": "prefill" if is_prefill else "decode",
            "window_cache": window_cache,
            "draft_kv": draft_kv,
            "positions": positions,
            # 每实例 cache 状态版本绑定（before/after/published + 前提）
            "stage_states": stage_states,
            "forward_spec_flow": (
                "forward_embed → stage loop（prefill：各 stage 只建 cache、"
                "h passthrough）→ 933 提前 return（无 head/markov/confidence）"
                if is_prefill else
                "forward_embed → stage loop（真实 h 链 mtp0→1→2）→ "
                "forward_head（hc_head→norm→head full_logits→Markov K5→"
                "Confidence）→ 返回三元组"),
            "history_premise": "连续历史显式假设：P 个历史 token 已按同一"
                               "窗口规则写入环形槽位（本图不验证，P 数值"
                               "不证明历史）；cache 实际内容 unknown——"
                               "before/写前已知性保持 unknown，写后有效"
                               "槽数按数量口径单独给出（stage_states"
                               ".after_slot_count / window_cache."
                               "valid_slot_count）",
        })
    return views


# ---------------------------------------------------------------------------
# 集成文档（data/dspark/integration.json 落盘结构）
# ---------------------------------------------------------------------------


def integration_document(bundle: DSparkIntegrationBundle) -> dict:
    """集成文档：连接目录/实例命名空间/边界闭世界/轴限定/HF named diff/
    组件映射（可直接 JSON 落盘，确定性：无时间戳/绝对路径）。"""
    g = bundle.graph
    inst_docs = []
    for i, path in enumerate(_STAGE_PARAM_PATHS):
        binding = bundle.chain.to_dict()["stage_bindings"][i]
        inst_docs.append({
            "stage_id": i,
            "param_path": path,
            "actual_layer_id": binding["actual_layer_id"],
            "node_prefix": _STAGE_PREFIXES[i],
            "state_id": _STAGE_STATE_IDS[i],
            "is_first": binding["is_first"],
            "is_last": binding["is_last"],
            "role_modules": list(binding["role_modules"]),
            "shared": {"embed": "shared.embed（Transformer.embed 同一对象，"
                                "903 绑定非复制）",
                       "head": "shared.head（Transformer.head 同一对象，"
                               "904 绑定非复制）"},
        })
    inputs_doc = []
    for nid, port, tid, desc in _EXTERNAL_INPUTS:
        t = next(x for x in g.tensors if x.stable_id == tid)
        inputs_doc.append({
            "node": nid, "port": port, "tensor": tid,
            "dtype": t.dtype, "shape_symbols": [a.symbol for a in t.axes],
            "note": desc,
        })
    outputs_doc = []
    for nid, port, tid, desc in _TRIPLE_OUTPUTS:
        t = next(x for x in g.tensors if x.stable_id == tid)
        outputs_doc.append({
            "node": nid, "port": port, "tensor": tid,
            "dtype": t.dtype, "shape_symbols": [a.symbol for a in t.axes],
            "note": desc,
        })
    dropped = []
    for nid in _DROPPED_NODES:
        dropped.append({"node": nid,
                        "replaced_by": _DROPPED_TEMPLATES.get(
                            nid.split(".stage.")[0] + ".stage." +
                            nid.split(".stage.")[-1]
                            if ".stage." in nid else nid,
                            _DROPPED_TEMPLATES.get(nid, nid))})
    renames = []
    for ax in g.named_axes:
        if ax.symbol in ("C1_exit", "C1_conf"):
            renames.append({"qualified_symbol": ax.symbol,
                            "name": ax.name, "meaning": ax.meaning,
                            "origin": ax.origin, "evidence_ref": ax.evidence_ref})
    return {
        "schema": "dspark-integration/v1",
        "template_id": TEMPLATE_ID,
        "model_key": "deepseek_v4_flash_0731",
        "instances": inst_docs,
        "connections": [c.to_dict() for c in bundle.connections],
        "boundaries": {
            "external_inputs": inputs_doc,
            "triple_outputs": outputs_doc,
            "dropped_boundaries": dropped,
            "prefill_last_h_out_note":
                "prefill（start_pos==0）forward_spec 933-934 提前 return："
                "stage2 prefill passthrough 输出无下游消费，如实保留为"
                "无消费端口，不画假边",
        },
        "axis_qualifications": {
            "policy": "同名轴逐字典一致才合并；唯一已知冲突 C1 按图限定改名；"
                      "其余冲突一律具名拒绝",
            "qualified": renames,
        },
        "graph_stats": {
            "nodes": len(g.nodes), "edges": len(g.edges),
            "tensors": len(g.tensors),
            "evidence_entries": len(bundle.registry.all()),
        },
        "ratios_consumers": bundle.chain.to_dict()["ratios_consumers"],
        "hf_difference": {
            "named": [
                "reference run config n_mtp_layers=3（inference/config.json），"
                "消费方 = inference model.py DSparkBlock 实例数"
                "（model.py:901 range(args.n_mtp_layers)）→ 本集成图三实例",
                "HF root config num_nextn_predict_layers=1"
                "（config.json:31），消费方 = checkpoint nextn.* 单 MTP 段"
                "布局（v3 MTPLoader 惯例，非推理图实例数）",
                "HF transformers modeling（modeling_deepseek_v4.py）无 DSpark"
                " 类，且 _keys_to_ignore_on_load_unexpected 忽略 mtp.*"
                "（modeling_deepseek_v4.py:1253）",
            ],
            "conclusion": "本 V2 DSpark 集成图来自 reference inference 源码"
                          "静态取证；不声称、也不暗示 HF 实现已覆盖 0731 "
                          "DSpark/MTP 完整实现。两个配置字段来源/消费者不同，"
                          "不互相换算。",
        },
        "unknowns": [
            "kernel.py tilelang JIT kernel（sparse_attn/act_quant 等）数值"
            "实现未核对（不透明 kernel 边界）",
            "output_ids/logits/confidence 实际 token 与数值未知（静态分析，"
            "不执行推理）",
            "win_cache 实际内容 unknown（历史连续调用前提，不当作空）",
            "每专家实际 token 数 N_e 数据相关静态未知",
        ],
        "component_maps": {
            "chain_map": bundle.chain.to_dict(),
            "entry_bindings": bundle.entry_bindings.to_dict(),
            "exit_role_map": bundle.exit_role_map,
            "markov_step_map": bundle.markov_step_map,
            "confidence_map": bundle.confidence_map,
        },
        "component_maps_scope": {
            "note": "component_maps 是各组件包（B2/B3a/B3b）认可时的"
                    "历史 scope 记录：其中 scope.pending 与 "
                    "stage_bindings[].pending_subgraphs 描述的是该组件包"
                    "当时的边界，不是当前整体集成的接线状态。",
            "superseded_by": {
                "current_bindings": "instances + role_bindings（本文档/"
                                    "页面 dsparkView.integration，来自 "
                                    "实际集成图与展开绑定表）",
                "real_wiring": "connections（14 条跨图连接 = 集成图真实"
                               "边）与 boundaries（闭世界外边界）",
                "chain_map.pending_subgraphs": (
                    "B2 组件包认可时 first/last 角色专有模块的内部图"
                    "pending；当前整体集成中 main_proj/main_norm/"
                    "forward_embed 入口链与 hc_head/norm/head/Markov/"
                    "Confidence 出口链已是真实集成图节点/边（见本图），"
                    "仅各内部 kernel 数值仍 unknown"),
                "exit_role_map.scope.pending / markov_step_map.scope."
                "pending / confidence_map.scope.pending": (
                    "B3b 各组件包认可时的 pending（head 出口图/三元组"
                    "拼接/跨图集成）；当前 B4 已真实集成，正式剩余 "
                    "pending 见 remaining_pending"),
            },
        },
        "remaining_pending": [
            "正式 V2 独立门禁（非组件级，全产物）",
            "真实浏览器验收（当前 blocked；NodeVM 只算 JS 逻辑执行）",
            "用户阅读验收（不代签）",
            "kernel.py tilelang JIT kernel 数值实现（不透明 kernel 边界，"
            "unknown 保持）",
        ],
    }


# ---------------------------------------------------------------------------
# 实例角色/参数绑定（页面 DSpark 实例面板数据源；从实际集成图 + 展开绑定
# 表派生，全部联查真实图节点/端口/张量，非静态说明文字）
# ---------------------------------------------------------------------------

# 模板参数 spec → 集成图参数节点（entry/exit 前缀）与 stage 实例张量后缀。
# 节点/张量 ID 在构建期逐一核验存在，漂移即 ExtractionError（目录与图
# 不许静默漂移）。
#
# binding 三种口径（codex_b4_review02 阻断2：存在性检查不够——面板宣称
# 参数联查就必须联查真正的参数实体，禁止把激活当参数张量）：
# - "param_tensor"：集成图独立参数张量 tensor:mtp{i}.stage:<suffix>，构建
#   期联查 dtype/轴/消费边（graph_tensor + consumer_nodes 逐字段核验）；
# - "inline_op_param"：权重内联在算子内（无独立图张量；如 wq_a/wq_b/
#   wo_a/wo_b/expert/shared 的 Linear.weight、q_norm 的 RMSNorm weight），
#   记录消费节点（联查存在）+ 源码口径，明确标注"参数属于内联算子"，
#   不虚构张量 ID；
# - "inline_op_shared_param"：内联且跨实例共享引用（同 param_tensor
#   消费端口核验；wkv_w/kv_norm_w 属此，param_tensor 口径单列）。
_ROLE_PARAM_ANCHORS: tuple[tuple[str, str, str], ...] = (
    # (spec_id, entry/exit 参数节点 stable_id, stage 模板张量后缀)
    ("stage.hc_attn_fn_w", None, "hc_attn_fn_w"),
    ("stage.hc_attn_scale_w", None, "hc_attn_scale_w"),
    ("stage.hc_attn_base_w", None, "hc_attn_base_w"),
    ("stage.hc_ffn_fn_w", None, "hc_ffn_fn_w"),
    ("stage.hc_ffn_scale_w", None, "hc_ffn_scale_w"),
    ("stage.hc_ffn_base_w", None, "hc_ffn_base_w"),
    ("stage.attn_norm_w", None, "attn_norm_w"),
    ("stage.ffn_norm_w", None, "ffn_norm_w"),
    ("stage.wkv_w", None, "wkv_w"),
    ("stage.kv_norm_w", None, "kv_norm_w"),
    ("stage.attn_sink_w", None, "attn_sink_param"),
    ("stage.gate_weight_w", None, "gate_weight_w"),
    ("stage.gate_bias_w", None, "gate_bias_w"),
    # q_norm 权重是 dspark.q_norm（rms_norm 算子）的内联参数：集成图只有
    # 激活张量 qr[B,N_draft,Rq]（bf16）——补修01 曾把激活误标为参数张量，
    # 本轮改为 inline_op_param 口径（后缀 None 表示无参数张量）
    ("stage.q_norm_w", None, None),
    ("stage.wq_a_w", None, None),
    ("stage.wq_b_w", None, None),
    ("stage.wo_a_w", None, None),
    ("stage.wo_b_w", None, None),
    # 路由/shared 专家权重内联在 swiglu_expert 算子（E=256 份实例，
    # 模板身份用通配路径，不造全部专家实例）
    ("stage.expert_w1_w", None, None),
    ("stage.expert_w2_w", None, None),
    ("stage.expert_w3_w", None, None),
    ("stage.shared_w1_w", None, None),
    ("stage.shared_w2_w", None, None),
    ("stage.shared_w3_w", None, None),
)

# 内联算子参数的消费节点联查表（spec_id → stage 模板节点后缀元组；
# 构建期逐一核验节点存在，漂移即 ExtractionError）
_INLINE_PARAM_CONSUMERS: dict[str, tuple[str, ...]] = {
    "stage.q_norm_w": ("dspark.q_norm",),
    "stage.wq_a_w": ("dspark.wq_a",),
    "stage.wq_b_w": ("dspark.wq_b",),
    "stage.wo_a_w": ("dspark.o_einsum",),
    "stage.wo_b_w": ("dspark.wo_b",),
    "stage.expert_w1_w": ("stage.moe.expert_template",),
    "stage.expert_w2_w": ("stage.moe.expert_template",),
    "stage.expert_w3_w": ("stage.moe.expert_template",),
    "stage.shared_w1_w": ("stage.moe.shared_experts",),
    "stage.shared_w2_w": ("stage.moe.shared_experts",),
    "stage.shared_w3_w": ("stage.moe.shared_experts",),
}


def dspark_role_bindings(bundle: DSparkIntegrationBundle) -> dict:
    """实例角色/参数绑定面板数据（页面 dsparkView.integration.role_bindings）。

    从实际集成图与 B2 展开绑定表真实派生：
    - 逐实例（mtp.0/1/2）：actual_layer_id 43/44/45、first/last 角色、
      state_id 与 win_cache 状态版本对/迁移；
    - 公共模板参数 → 实例参数路径（mtp.{i}.{slot}）与集成图张量联查
      （tensor:mtp{i}.stage:<suffix>，逐条核验存在，漂移具名拒绝）；
    - 唯一 embed/head 共享对象（三实例同一 entry.embed_w / exit.head_w，
      不复制）；
    - last 专有 norm/hc_head/Markov/Confidence 参数 → exit/mk/cf 参数
      节点+张量（runtime dtype 与 checkpoint 存储 dtype 分开声明）。
    纯派生无时间戳；引用非法/锚点缺失一律 ExtractionError。
    """
    g = bundle.graph
    nodes = {n.stable_id: n for n in g.nodes}
    tensors = {t.stable_id: t for t in g.tensors}
    expansion = bundle.expansion.to_dict()
    specs_by_id = {s["spec_id"]: s for s in expansion["param_specs"]}
    chain = bundle.chain.to_dict()["stage_bindings"]
    transitions = {tr.stable_id: tr for tr in g.state_transitions}
    versions = {v.stable_id: v for v in g.state_versions}

    def _require_node(nid: str) -> None:
        if nid not in nodes:
            raise ExtractionError(f"角色绑定锚点节点缺失: {nid}")

    def _require_tensor(tid: str) -> None:
        if tid not in tensors:
            raise ExtractionError(f"角色绑定锚点张量缺失: {tid}")

    def _require_consumer_binding(graph_node: str, graph_tensor: str,
                                  consumer_node: str, module: str) -> None:
        """consumer_node 必须真实消费 graph_tensor（存在该张量的输入端口
        且有携带张量的边从参数节点/算子到消费者）——逐字段联查，不只
        assert 节点存在（codex_b4_review02 阻断2 口径）。"""
        _require_node(graph_node)
        _require_tensor(graph_tensor)
        _require_node(consumer_node)
        cons = nodes[consumer_node]
        if not any(p.tensor_id == graph_tensor for p in cons.input_ports):
            raise ExtractionError(
                f"角色绑定 {module}: 消费节点 {consumer_node} 无消费 "
                f"{graph_tensor} 的输入端口（实际端口张量 "
                f"{[p.tensor_id for p in cons.input_ports]}）")
        if not any(e.tensor_id == graph_tensor
                   and e.dst_node_id == consumer_node for e in g.edges):
            raise ExtractionError(
                f"角色绑定 {module}: 图中无 {graph_tensor} → {consumer_node} "
                f"的携带边")

    inst_docs = []
    for i, path in enumerate(_STAGE_PARAM_PATHS):
        binding = chain[i]
        state_id = _STAGE_STATE_IDS[i]
        tr = transitions[f"tr_dspark_cache_mtp{i}"]
        # win_cache 状态版本联查（tr before/after 版本必须存在于版本表）
        for vid in (tr.before_version_id, tr.after_version_id):
            if vid not in versions:
                raise ExtractionError(f"win_cache 版本缺失: {vid}")
        cache_tid = f"tensor:mtp{i}.stage:win_cache"
        _require_tensor(cache_tid)
        inst_docs.append({
            "stage_id": i,
            "param_path": path,
            "actual_layer_id": binding["actual_layer_id"],
            "node_prefix": _STAGE_PREFIXES[i],
            "is_first": binding["is_first"],
            "is_last": binding["is_last"],
            "role_modules": list(binding["role_modules"]),
            "state_id": state_id,
            "cache_state": {
                "cache_tensor": cache_tid,
                "transition_id": tr.stable_id,
                "before_version": tr.before_version_id,
                "after_version": tr.after_version_id,
                "version_labels": ["v_hist", "v_after"],
                "note": "win_cache 每实例独立身份（B2 展开；内容 unknown）",
            },
            "entry_chain_live": i == 0,
            "entry_chain_note": (
                "main_proj/main_norm/forward_embed 是本实例（stage0）真实"
                "集成图链（entry.* 节点）" if i == 0 else
                "本实例无入口专有模块（共享 main_x 第三参透传 + 共享 embed）"),
            "exit_chain_live": binding["is_last"],
            "exit_chain_note": (
                "norm/hc_head/共享 head full_logits/Markov/Confidence 是"
                "本实例（stage2）真实集成图链（exit.*/mk.*/cf.* 节点）"
                if binding["is_last"] else
                "本实例无出口专有模块（stage 串接后 h 进下一实例）"),
        })

    # 公共模板参数 → 实例路径 + 集成图联查（param_tensor 逐条核验
    # dtype/shape/instance_path 消费边；inline_op_param 联查消费节点）
    tensor_by_id = {t.stable_id: t for t in g.tensors}
    param_bindings_docs = []
    for spec_id, _anchor, suffix in _ROLE_PARAM_ANCHORS:
        spec = specs_by_id[spec_id]
        entry = {
            "spec_id": spec_id,
            "param_slot": spec["param_slot"],
            "dtype": spec["dtype"],
            "shape": list(spec.get("shape", ())),
            "note": spec["note"],
            "evidence": list(spec["evidence"]),
        }
        per_inst = []
        for i, path in enumerate(_STAGE_PARAM_PATHS):
            rec = {
                "stage_id": i,
                "instance_path": f"{path}.{spec['param_slot']}",
            }
            if suffix is not None:
                # param_tensor 口径：真实参数张量联查（dtype/shape 逐字段
                # 核验 spec 口径；消费边必须指向真实消费者）
                tid = f"tensor:mtp{i}.stage:{suffix}"
                _require_tensor(tid)
                t = tensor_by_id[tid]
                if t.dtype != spec["dtype"]:
                    raise ExtractionError(
                        f"角色绑定 {spec_id} 实例 {i} dtype 与 spec 不一致: "
                        f"图 {t.dtype!r} vs spec {spec['dtype']!r}")
                want_shape = list(spec.get("shape", ()))
                got_known = [ax.value for ax in t.axes
                             if ax.value is not None]
                known_want = [d for d in want_shape if d is not None]
                if want_shape and len(t.axes) != len(want_shape):
                    raise ExtractionError(
                        f"角色绑定 {spec_id} 实例 {i} 秩与 spec 不一致: "
                        f"图 {len(t.axes)} vs spec {len(want_shape)}")
                if want_shape and known_want and \
                        got_known != known_want and got_known:
                    raise ExtractionError(
                        f"角色绑定 {spec_id} 实例 {i} 已知维与 spec 不一致: "
                        f"图 {got_known} vs spec {known_want}")
                consumers = sorted({e.dst_node_id for e in g.edges
                                    if e.tensor_id == tid})
                for cn in consumers:
                    _require_node(cn)
                rec.update({
                    "binding": "param_tensor",
                    "graph_tensor": tid,
                    "graph_dtype": t.dtype,
                    "consumer_nodes": consumers,
                })
            else:
                # inline_op_param 口径：权重内联在算子内（无独立图张量，
                # 不虚构张量 ID）；联查消费节点存在
                consumers = tuple(
                    f"mtp{i}.{nid}" for nid in
                    _INLINE_PARAM_CONSUMERS[spec_id])
                for cn in consumers:
                    _require_node(cn)
                rec.update({
                    "binding": "inline_op_param",
                    "consumer_nodes": list(consumers),
                    "inline_note": (
                        "参数内联在本实例算子内（集成图无独立参数张量；"
                        "激活张量不是参数）——链接消费节点联查"),
                })
            per_inst.append(rec)
        entry["instances"] = per_inst
        entry["binding_summary"] = (
            "param_tensor ×%d" % sum(1 for r in per_inst
                                     if r["binding"] == "param_tensor")
            if suffix is not None else "inline_op_param（三实例同构）")
        param_bindings_docs.append(entry)

    # 唯一 embed/head 共享对象（三实例同一，节点逐一核验）
    for nid in ("entry.embed_w", "entry.embed_lookup", "exit.head_w",
                "exit.head"):
        _require_node(nid)
    shared = {
        "embed": {
            "checkpoint_ref": "Transformer.embed（model.py:903 三实例同一"
                              "对象非复制）",
            "weight_node": "entry.embed_w",
            "lookup_node": "entry.embed_lookup",
            "weight_tensor": "tensor:entry:embed_w",
            "instance_refs": ["mtp.0.embed", "mtp.1.embed", "mtp.2.embed"],
        },
        "head": {
            "checkpoint_ref": "Transformer.head（model.py:904 三实例同一"
                              "对象非复制；forward_head 863 full_logits="
                              "True）",
            "weight_node": "exit.head_w",
            "linear_node": "exit.head",
            "weight_tensor": "tensor:exit:head_w",
            "instance_refs": ["mtp.0.head", "mtp.1.head", "mtp.2.head"],
        },
    }
    for key, sec in shared.items():
        _require_node(sec["weight_node"])
        _require_tensor(sec["weight_tensor"])

    # last 专有参数 → 参数节点/张量（runtime/checkpoint dtype 分开）
    last_binding = chain[2]
    last_params = [
        {
            "module": "norm",
            "param_path": "mtp.2.norm.weight",
            "graph_node": "exit.norm_w",
            "graph_tensor": "tensor:exit:norm_w",
            "consumer_node": "exit.norm",
            "dtype": "fp32",
            "checkpoint_dtype": "bf16",
            "dtype_note": "RMSNorm 模块 fp32；checkpoint 存 bf16"
                          "（model.py:716-717 源码注释）",
        },
        {
            "module": "hc_head_fn",
            "param_path": "mtp.2.hc_head_fn",
            "graph_node": "exit.hc_head_fn_w",
            "graph_tensor": "tensor:exit:hc_head_fn_w",
            "consumer_node": "exit.hc_head.mix",
            "dtype": "fp32",
            "checkpoint_dtype": "fp32",
            "dtype_note": "set_dtype(torch.float32) 块内创建"
                          "（model.py:838-841）",
        },
        {
            "module": "hc_head_base",
            "param_path": "mtp.2.hc_head_base",
            "graph_node": "exit.hc_head_base_w",
            "graph_tensor": "tensor:exit:hc_head_base_w",
            "consumer_node": "exit.hc_head.gate",
            "dtype": "fp32",
            "checkpoint_dtype": "fp32",
            "dtype_note": "同 hc_head_fn（set_dtype 块）",
        },
        {
            "module": "hc_head_scale",
            "param_path": "mtp.2.hc_head_scale",
            "graph_node": "exit.hc_head_scale_w",
            "graph_tensor": "tensor:exit:hc_head_scale_w",
            "consumer_node": "exit.hc_head.gate",
            "dtype": "fp32",
            "checkpoint_dtype": "fp32",
            "dtype_note": "shape-(1,) 单元宽（C1_exit）",
        },
        {
            "module": "markov_head.markov_w1",
            "param_path": "mtp.2.markov_head.markov_w1.weight",
            "graph_node": "mk.w1_param",
            "graph_tensor": "tensor:mk:w1_w",
            "consumer_node": "mk.step0.embed",
            "dtype": "bf16",
            "checkpoint_dtype": "bf16",
            "dtype_note": "ParallelEmbedding [V,Rm] bf16；五步逐步查表"
                          "（mk.step{0..4}.embed 同一张量）",
        },
        {
            "module": "markov_head.markov_w2",
            "param_path": "mtp.2.markov_head.markov_w2.weight",
            "graph_node": "mk.w2_param",
            "graph_tensor": "tensor:mk:w2_w",
            "consumer_node": "mk.step0.bias",
            "dtype": "fp32",
            "checkpoint_dtype": "bf16",
            "dtype_note": "ParallelHead weight runtime fp32（model.py:729）；"
                          "checkpoint lm_head 存 bf16（model.py:728-729 源码"
                          "注释）——mk:logits fp32 的来源；五步 bias 相加同"
                          "张量",
        },
        {
            "module": "confidence_head.proj",
            "param_path": "mtp.2.confidence_head.proj.weight",
            "graph_node": "cf.proj_w",
            "graph_tensor": "tensor:cf:proj_w",
            "consumer_node": "cf.proj",
            "dtype": "fp32",
            "checkpoint_dtype": "bf16",
            "dtype_note": "runtime fp32（model.py:811 dtype=torch.float32，"
                          "fp32 confidence score）；checkpoint 存 bf16"
                          "（model.py:810 源码注释，与既有 Confidence 严格"
                          "map 一致）；输出维 1（C1_conf）",
        },
    ]
    for p in last_params:
        _require_consumer_binding(p["graph_node"], p["graph_tensor"],
                                  p["consumer_node"], p["module"])

    # first 专有参数（stage0 main_proj/main_norm）
    for nid in ("entry.main_proj_w", "entry.main_norm_w"):
        _require_node(nid)
    first_params = [
        {
            "module": "main_proj",
            "param_path": "mtp.0.main_proj.weight",
            "graph_node": "entry.main_proj_w",
            "graph_tensor": "tensor:entry:main_proj_w",
            "consumer_node": "entry.main_proj",
            "dtype": "fp8_e4m3fn",
            "checkpoint_dtype": "fp8_e4m3fn",
            "dtype_note": "Transformer default_dtype fp8（Linear 默认）；"
                          "入宽 H×3（40/41/42 readout concat）",
        },
        {
            "module": "main_norm",
            "param_path": "mtp.0.main_norm.weight",
            "graph_node": "entry.main_norm_w",
            "graph_tensor": "tensor:entry:main_norm_w",
            "consumer_node": "entry.main_norm",
            "dtype": "fp32",
            "checkpoint_dtype": "bf16",
            "dtype_note": "RMSNorm 模块 fp32；checkpoint 存 bf16",
        },
    ]
    for p in first_params:
        _require_consumer_binding(p["graph_node"], p["graph_tensor"],
                                  p["consumer_node"], p["module"])

    return {
        "schema": "dspark-role-bindings/v1",
        "template_id": TEMPLATE_ID,
        "n_stages": _N_STAGES,
        "n_main_layers": bundle.values.n_layers,
        "instances": inst_docs,
        "template_param_bindings": param_bindings_docs,
        "shared": shared,
        "first_only_params": first_params,
        "last_only_params": last_params,
        "derivation_note": (
            "全部字段从实际集成图（节点/张量/状态迁移）与 B2 展开绑定表"
            "派生：instance_path 联查 mtp.{i}.{slot}，graph_node/"
            "graph_tensor 为集成图真实实体（构建期逐一核验存在）；非"
            "静态说明文字"),
        "unknown_note": "cache 实际内容/推理数值 unknown（静态结构分析）",
    }
