"""DeepSeek-V4-Flash-0731 主干四类变体提取（V2 工作包 A）。

复用已验证的 CSA·TopK 提取器（deepseek_v4_reference），以变体配置参数化
真实拓扑差异（源码 model.py 推导，不只换标题）：

- swa_hash（layer 0/1 代表 0）：compress_ratio=0 → 无 Compressor/Indexer；
  纯滑动窗口（YaRN 关闭、基础 rope_theta，model.py:483-485）；Hash 门控
  （tid2eid token-ID 查表，无选择偏置，model.py:561-564）。
- csa_hash（layer 2 代表）：r=4 → overlap Compressor + 学习式 Indexer；
  Hash 门控。
- csa_topk（layer 4 代表）：V1 已验证（委托原提取器）。
- hca_topk（layer 3 代表）：r=128 → 压缩但无学习式 Indexer
  （model.py:474-477 else 分支），固定压缩位置选择
  get_compress_topk_idxs（model.py:519）；TopK 门控（有选择偏置）。

保留双 mHC、低秩 Q/O、共享 KV 单投影、专家 shared 流；专家实际选择/每专家
token 数（N_e）诚实 unknown。extracted 只在对应模板图构建成功后标记。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_reference import (  # noqa: F401
    ExtractionError,
    EvidenceBase,
    EvidenceRegistry,
    SourceEvidence,
    _GraphBuilder,
    _add_compressor,
    _add_compressor_states,
    _add_flow,
    _add_moe,
    _add_tensors,
    _evidence_defs,
    _graph_axes,
    build_reference_graph,
    indexer_visibility,
    load_evidence_base,
)
from structure_reader.graph_document import SemanticGraph
from structure_reader.graph_ir import (
    AliasRelation,
    Edge,
    Guard,
    GuardActivation,
    NamedAxis,
    Node,
    Port,
    StateTransition,
    StateVersion,
    Tensor,
)
from structure_reader.rules.deepseek_v4_flash_0731 import (
    GUARD_DEFS,
    ModelValues,
    load_model_values,
)
from structure_reader.scenario_evaluator import evaluate_tri

__all__ = [
    "VariantKey",
    "VARIANT_SPECS",
    "build_variant_graph",
    "build_all_variant_graphs",
]

VARIANT_KEY_SWA_HASH = "swa_hash"
VARIANT_KEY_CSA_HASH = "csa_hash"
VARIANT_KEY_CSA_TOPK = "csa_topk"
VARIANT_KEY_HCA_TOPK = "hca_topk"

VariantKey = str


@dataclass(frozen=True)
class VariantSpec:
    """一个主干变体的真实拓扑配置（源码推导，非标签）。"""

    key: VariantKey
    template_id: str
    representative_layer: int
    compress_ratio: int  # 0 = 无压缩（SWA）
    has_indexer: bool
    gate_kind: str  # hash | topk
    hash_layers: bool  # tid2eid token-ID 路由（model.py:561-564,582）
    yarn_enabled: bool  # 压缩层用 YaRN+compress_rope_theta；SWA 用基础 theta


VARIANT_SPECS: dict[VariantKey, VariantSpec] = {
    VARIANT_KEY_SWA_HASH: VariantSpec(
        key=VARIANT_KEY_SWA_HASH,
        template_id="deepseek_v4_flash_0731:swa_hash",
        representative_layer=0,
        compress_ratio=0,
        has_indexer=False,
        gate_kind="hash",
        hash_layers=True,
        yarn_enabled=False,  # model.py:483-485 SWA 关闭 YaRN
    ),
    VARIANT_KEY_CSA_HASH: VariantSpec(
        key=VARIANT_KEY_CSA_HASH,
        template_id="deepseek_v4_flash_0731:csa_hash",
        representative_layer=2,
        compress_ratio=4,
        has_indexer=True,
        gate_kind="hash",
        hash_layers=True,
        yarn_enabled=True,
    ),
    VARIANT_KEY_CSA_TOPK: VariantSpec(
        key=VARIANT_KEY_CSA_TOPK,
        template_id="deepseek_v4_flash_0731:csa_topk",
        representative_layer=4,
        compress_ratio=4,
        has_indexer=True,
        gate_kind="topk",
        hash_layers=False,
        yarn_enabled=True,
    ),
    VARIANT_KEY_HCA_TOPK: VariantSpec(
        key=VARIANT_KEY_HCA_TOPK,
        template_id="deepseek_v4_flash_0731:hca_topk",
        representative_layer=3,
        compress_ratio=128,
        has_indexer=False,
        gate_kind="topk",
        hash_layers=False,
        yarn_enabled=True,
    ),
}


def build_variant_graph(
    workspace_root: Path,
    spec: VariantSpec,
    *,
    values: ModelValues | None = None,
    evidence_base: EvidenceBase | None = None,
) -> tuple[SemanticGraph, EvidenceRegistry, ModelValues]:
    """构建一个变体的代表图：复用 V1 骨架 + 变体差异注入。

    swa_hash：V1 骨架去掉 Compressor/Indexer 子图（cache 只剩窗口槽）；
    csa_hash/csa_topk：完整 V1 拓扑；hca_topk：压缩但 Indexer 替换为
    固定压缩位置选择（get_compress_topk_idxs，model.py:519），无学习式
    Q/weights_proj/自有 Compressor。
    """
    if spec.key == VARIANT_KEY_CSA_TOPK:
        # V1 已验证路径：委托原提取器（不重复构建）
        return build_reference_graph(workspace_root, values=values,
                                     evidence_base=evidence_base)

    workspace_root = Path(workspace_root)
    values = values or load_model_values(workspace_root)
    evidence_base = evidence_base or load_evidence_base(workspace_root)
    registry = EvidenceRegistry(evidence_base, workspace_root)
    for ev in _evidence_defs():
        registry.register(ev)

    axes = _graph_axes(values, compress_ratio=spec.compress_ratio)
    b = _GraphBuilder(axes)

    # 变体差异在张量集合与流构造前生效：按 spec 过滤 V1 骨架。
    t = _add_tensors(b, values, tprefix=spec.template_id.rsplit(":", 1)[-1],
                     compress_ratio=spec.compress_ratio)
    if spec.compress_ratio == 0:
        _strip_compression_tensors(b, t)
    if not spec.has_indexer:
        _strip_indexer_tensors(b, t)
    if spec.compress_ratio != 4:
        # HCA r=128 无 overlap（model.py:296 overlap = ratio==4）：
        # 剥离 overlap 专属张量（blockify 后直接池化）
        for role in ("comp_overlap", "idx_comp_overlap"):
            t.pop(role, None)
            b.remove_tensor(f"csa_topk:{role}")
    if spec.hash_layers:
        # Hash 门控：token-ID 输入（model.py:582 tid2eid[input_ids]；
        # forward 实参 input_ids.flatten()，model.py:637）
        tid = b.tensor(
            f"{spec.template_id.rsplit(':', 1)[-1]}:gate_input_ids",
            "int64", ("B", "T"))
        t["gate_input_ids"] = tid
        # input_ids 源节点（块级输入，model.py:919 layer(h, start_pos, input_ids)）
        b.node("block.input_ids_src", "passthrough", "tensor_op",
               (), (("y", tid),), ())
    _add_variant_flow(b, t, values, spec)
    if spec.compress_ratio != 0:
        # 状态目录按实例实际存在声明（HCA 无 Indexer 三状态）
        _add_compressor_states(b, with_indexer=spec.has_indexer)
    if spec.compress_ratio == 0:
        b.state("csa_topk:attn_window_cache", "tr_attn_cache_win",
                ("ev_prefill_cache_write", "ev_decode_cache_read"),
                "SWA 窗口 cache：v_hist 历史 → v_after 本调用写入后"
                "（无压缩区，model.py:479 仅窗口槽）")
    guards = _guard_objects_variant(registry, spec)

    for ev_id in b.evidence_ids:
        if ev_id not in registry:
            raise ExtractionError(f"图引用了未注册证据: {ev_id!r}")
    for axis in axes.values():
        if (axis.evidence_ref.startswith("ev_")
                and axis.evidence_ref not in registry):
            raise ExtractionError(
                f"轴 {axis.symbol} 引用了未注册证据: {axis.evidence_ref!r}"
            )

    graph = b.build(guards, spec.template_id)
    return graph, registry, values


def _strip_compression_tensors(b: _GraphBuilder, t: dict[str, str]) -> None:
    """SWA：移除压缩相关张量（构建前过滤；cache 只剩窗口槽）。"""
    for role in ("comp_kv_proj", "comp_score_proj", "comp_ape",
                 "comp_main_slice", "comp_tail_slice",
                 "comp_main_score_slice", "comp_tail_score_slice",
                 "comp_main_score_aped", "comp_tail_score_aped",
                 "comp_dec_score_aped", "comp_dec_token", "comp_dec_token_score",
                 "comp_ape_tail_rows", "comp_ape_dec_row",
                 "comp_blocks", "comp_overlap", "comp_pooled",
                 "comp_normed", "comp_out", "comp_return",
                 "comp_state_kv_hist", "comp_state_score_hist",
                 "comp_state_kv_updated", "comp_state_score_updated",
                 "comp_state_kv_final", "comp_state_score_final",
                 "comp_cache", "comp_idxs", "attn_cache"):
        t.pop(role, None)
        b.remove_tensor(f"csa_topk:{role}")


def _strip_indexer_tensors(b: _GraphBuilder, t: dict[str, str]) -> None:
    """HCA：移除学习式 Indexer 张量（保留 attention 压缩链）。"""
    for role in ("idx_q_flat", "idx_q", "idx_weights",
                 "idx_cache", "idx_cache_visible", "idx_scores",
                 "idx_score", "idx_score_masked",
                 "idx_comp_kv_proj", "idx_comp_score_proj", "idx_comp_ape",
                 "idx_comp_blocks", "idx_comp_overlap", "idx_comp_pooled",
                 "idx_comp_normed", "idx_comp_out", "idx_comp_return",
                 "idx_comp_main_slice", "idx_comp_tail_slice",
                 "idx_comp_main_score_slice", "idx_comp_tail_score_slice",
                 "idx_comp_main_score_aped", "idx_comp_tail_score_aped",
                 "idx_comp_dec_score_aped", "idx_comp_dec_token",
                 "idx_comp_dec_token_score", "idx_comp_ape_tail_rows",
                 "idx_comp_ape_dec_row",
                 "idx_comp_state_kv_hist", "idx_comp_state_score_hist",
                 "idx_comp_state_kv_updated", "idx_comp_state_score_updated",
                 "idx_comp_state_kv_final", "idx_comp_state_score_final"):
        t.pop(role, None)
        b.remove_tensor(f"csa_topk:{role}")


def _add_variant_flow(b: _GraphBuilder, t: dict[str, str],
                      values: ModelValues, spec: VariantSpec) -> None:
    """按变体 spec 构建流：V1 骨架（内部按张量角色是否缺失自适应跳过）+
    变体专属分支（Hash 门控 / HCA 固定压缩选择）。"""
    # V1 骨架 flow：_add_flow 内部所有对 t[role] 的引用按缺失跳过——
    # 用 KeyError 捕获跳过过于脆弱，改为显式过滤：包一层 role lookup
    # 缺失角色抛 ExtractionError（骨架不该在变体中引用被剥离的角色），
    # 因此这里直接调用 _add_flow 的一个参数化包装。
    _add_flow_variant(b, t, values, spec)


def _add_flow_variant(b: _GraphBuilder, t: dict[str, str],
                      values: ModelValues, spec: VariantSpec) -> None:
    """变体 flow：复用 V1 _add_flow 主体，压缩/Indexer 段按 spec 跳过，
    追加变体专属节点（Hash tid2eid / HCA 固定压缩选择）。"""
    rd = spec.representative_layer

    def node(sid: str, op: str, gr: str, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid: str, src, dst, role: str, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    # ---- 双 mHC + attention 共用骨架（无压缩/Indexer 的部分照抄 V1）----
    node("block.hc_pre_attn.flatten", "flatten", "tensor_op",
         (("x", t["x_hc_in"]),),
         (("x_flat", t["x_hc_attn_flat"]), ("residual", t["x_hc_in"])),
         ("ev_hc_pre",))
    node("block.hc_pre_attn.mix", "linear", "parameter_op",
         (("x", t["x_hc_attn_flat"]),), (("y", t["attn_mixes"]),),
         ("ev_hc_pre",))
    node("block.hc_pre_attn.sinkhorn", "sinkhorn_split", "opaque_kernel",
         (("mixes", t["attn_mixes"]),),
         (("pre", t["attn_pre"]), ("post", t["attn_post"]),
          ("comb", t["attn_comb"])),
         ("ev_hc_pre", "ev_hc_sinkhorn_kernel"))
    node("block.hc_pre_attn.reduce", "weighted_sum", "tensor_op",
         (("pre", t["attn_pre"]), ("x", t["x_hc_in"])),
         (("y", t["x_attn_in"]),), ("ev_hc_pre",))
    edge("e_hcpre_a_flat", ("block.hc_pre_attn.flatten", "x_flat"),
         ("block.hc_pre_attn.mix", "x"), "x_hc_attn_flat")
    edge("e_hcpre_a_mix", ("block.hc_pre_attn.mix", "y"),
         ("block.hc_pre_attn.sinkhorn", "mixes"), "attn_mixes")
    edge("e_hcpre_a_pre", ("block.hc_pre_attn.sinkhorn", "pre"),
         ("block.hc_pre_attn.reduce", "pre"), "attn_pre")
    edge("e_hcpre_a_x", ("block.hc_pre_attn.flatten", "residual"),
         ("block.hc_pre_attn.reduce", "x"), "x_hc_in")
    edge("e_hcpre_a_out", ("block.hc_pre_attn.reduce", "y"),
         ("attn_norm", "x"), "x_attn_in")
    edge("e_hcpre_a_post", ("block.hc_pre_attn.sinkhorn", "post"),
         ("block.hc_post_attn.combine", "post"), "attn_post")
    edge("e_hcpre_a_comb", ("block.hc_pre_attn.sinkhorn", "comb"),
         ("block.hc_post_attn.combine", "comb"), "attn_comb")

    node("attn_norm", "rms_norm", "parameter_op",
         (("x", t["x_attn_in"]),), (("y", t["x_attn_normed"]),),
         ("ev_attn_norm",))

    node("attn.wq_a", "linear", "parameter_op",
         (("x", t["x_attn_normed"]),), (("y", t["qr"]),), ("ev_attn_init",))
    edge("e_norm_wqa", ("attn_norm", "y"), ("attn.wq_a", "x"), "x_attn_normed")
    node("attn.q_norm", "rms_norm", "parameter_op",
         (("x", t["qr"]),), (("y", t["qr"]),), ("ev_attn_init",))
    edge("e_wqa_qnorm", ("attn.wq_a", "y"), ("attn.q_norm", "x"), "qr")
    node("attn.wq_b", "linear", "parameter_op",
         (("x", t["qr"]),), (("y", t["q_flat"]),), ("ev_attn_init",))
    edge("e_qnorm_wqb", ("attn.q_norm", "y"), ("attn.wq_b", "x"), "qr")
    node("attn.q_unflatten", "unflatten", "tensor_op",
         (("x", t["q_flat"]),), (("y", t["q"]),), ("ev_q_path",))
    edge("e_wqb_unfl", ("attn.wq_b", "y"), ("attn.q_unflatten", "x"), "q_flat")
    node("attn.q_rms_scale", "rms_scale", "tensor_op",
         (("x", t["q"]),), (("y", t["q"]),), ("ev_q_path",))
    edge("e_q_scale", ("attn.q_unflatten", "y"), ("attn.q_rms_scale", "x"), "q")
    node("attn.q_rope", "rope_partial", "tensor_op",
         (("x", t["q"]),), (("y", t["q_rope"]),),
         ("ev_q_path", "ev_apply_rotary"))
    edge("e_q_rope", ("attn.q_rms_scale", "y"), ("attn.q_rope", "x"), "q")

    node("attn.wkv", "linear", "parameter_op",
         (("x", t["x_attn_normed"]),), (("y", t["kv_single"]),),
         ("ev_kv_path",))
    edge("e_norm_wkv", ("attn_norm", "y"), ("attn.wkv", "x"), "x_attn_normed")
    node("attn.kv_norm", "rms_norm", "parameter_op",
         (("x", t["kv_single"]),), (("y", t["kv_normed"]),), ("ev_kv_path",))
    edge("e_wkv_norm", ("attn.wkv", "y"), ("attn.kv_norm", "x"), "kv_single")
    node("attn.kv_rope", "rope_partial", "tensor_op",
         (("x", t["kv_normed"]),), (("y", t["kv_quant"]),),
         ("ev_kv_path", "ev_apply_rotary"))
    edge("e_kv_rope", ("attn.kv_norm", "y"), ("attn.kv_rope", "x"), "kv_normed")
    node("attn.kv_quant", "act_quant_fp8", "quant_sim",
         (("x", t["kv_quant"]),), (("y", t["kv_quant"]),), ("ev_kv_path",))
    edge("e_kv_quant", ("attn.kv_rope", "y"), ("attn.kv_quant", "x"), "kv_quant")
    b.alias("alias_kv_key", relation_kind="same_tensor",
            tensor_id=t["kv_quant"], alias_role="key",
            evidence=("ev_kv_path",))
    b.alias("alias_kv_value", relation_kind="same_tensor",
            tensor_id=t["kv_quant"], alias_role="value",
            evidence=("ev_kv_path",))

    node("attn.win_topk", "window_index", "index_op",
         (), (("y", t["win_idxs"]),), ("ev_window_topk",))

    # ---- 变体差异：压缩链（CSA×2 / HCA 有，SWA 无）----
    if spec.compress_ratio != 0:
        _add_compressor(b, t, values, prefix="attn.compressor",
                        x_role="x_attn_normed", cache_role="comp_cache",
                        role_prefix="comp_", rotate=False,
                        tr_kv="tr_comp_kv", tr_score="tr_comp_score",
                        tr_kv_shift="tr_comp_kv_shift",
                        tr_score_shift="tr_comp_score_shift",
                        overlap=spec.compress_ratio == 4)
        edge("e_comp_in", ("attn_norm", "y"), ("attn.compressor.wkv", "x"),
             "x_attn_normed")
        edge("e_comp_in_gate", ("attn_norm", "y"),
             ("attn.compressor.wgate", "x"), "x_attn_normed")
        b.alias("alias_comp_cache_slice", relation_kind="slice",
                tensor_id=t["attn_cache"], other_tensor_id=t["comp_cache"],
                transform_expr="attn_cache[:, W:]（压缩区视图，同存储非独立拷贝）",
                evidence=("ev_lazy_bind",))
        if spec.has_indexer:
            _add_indexer_flow(b, t, values, spec)
        else:
            _add_fixed_compress_select_flow(b, t, values, spec)
        _add_cache_and_concat(b, t, values, spec)
    else:
        # SWA：cache 只有窗口槽（无压缩区，model.py:479 无 2 项）
        node("attn.cache_update", "cache_write", "state_op",
             (("kv", t["kv_quant"]),),
             (("cache_win", t["win_cache"]),),
             ("ev_prefill_cache_write", "ev_decode_cache_read"))
        edge("e_kv_cache", ("attn.kv_quant", "y"),
             ("attn.cache_update", "kv"), "kv_quant")
        node("attn.cache_read", "cache_read", "state_op",
             (("cache_win", t["win_cache"]),),
             (("kv_decode", t["win_cache"]),),
             ("ev_decode_cache_read",))
        edge("e_cache_state_w", ("attn.cache_update", "cache_win"),
             ("attn.cache_read", "cache_win"), "win_cache", kind="state",
             state_transition_ref="tr_attn_cache_win")

    # ---- 变体差异：gate 分支（Hash tid2eid vs TopK bias 选择）----
    _add_moe_variant(b, t, values, spec)

    # ---- sparse_attn 与输出（含 r=128 的 HCA 压缩上下文）----
    _add_sparse_and_out(b, t, values, spec)


def _add_indexer_flow(b: _GraphBuilder, t: dict[str, str],
                      values: ModelValues, spec: VariantSpec) -> None:
    """CSA 系：学习式 Indexer（qr 双消费；与 V1 相同的真实子图）。"""
    def node(sid: str, op: str, gr: str, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid: str, src, dst, role: str, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    node("attn.indexer.wq_b", "linear", "parameter_op",
         (("x", t["qr"]),), (("y", t["idx_q_flat"]),), ("ev_idx_q",))
    edge("e_idx_wqb", ("attn.q_norm", "y"), ("attn.indexer.wq_b", "x"), "qr")
    node("attn.indexer.q_unflatten", "unflatten", "tensor_op",
         (("x", t["idx_q_flat"]),), (("y", t["idx_q"]),), ("ev_idx_q",))
    edge("e_idx_unfl", ("attn.indexer.wq_b", "y"),
         ("attn.indexer.q_unflatten", "x"), "idx_q_flat")
    node("attn.indexer.q_rope", "rope_partial", "tensor_op",
         (("x", t["idx_q"]),), (("y", t["idx_q"]),),
         ("ev_idx_q", "ev_apply_rotary"))
    edge("e_idx_qrope", ("attn.indexer.q_unflatten", "y"),
         ("attn.indexer.q_rope", "x"), "idx_q")
    node("attn.indexer.q_rotate", "hadamard_rotate", "quant_sim",
         (("x", t["idx_q"]),), (("y", t["idx_q"]),),
         ("ev_idx_q", "ev_idx_rotate"))
    edge("e_idx_rot", ("attn.indexer.q_rope", "y"),
         ("attn.indexer.q_rotate", "x"), "idx_q")
    node("attn.indexer.q_quant", "act_quant_fp4", "quant_sim",
         (("x", t["idx_q"]),), (("y", t["idx_q"]),), ("ev_idx_q",))
    edge("e_idx_quant", ("attn.indexer.q_rotate", "y"),
         ("attn.indexer.q_quant", "x"), "idx_q")
    _add_compressor(b, t, values, prefix="attn.indexer.compressor",
                    x_role="x_attn_normed", cache_role="idx_cache",
                    role_prefix="idx_comp_", rotate=True,
                    tr_kv="tr_idx_kv", tr_score="tr_idx_score",
                    tr_kv_shift="tr_idx_kv_shift",
                    tr_score_shift="tr_idx_score_shift")
    edge("e_idx_comp_in", ("attn_norm", "y"),
         ("attn.indexer.compressor.wkv", "x"), "x_attn_normed")
    edge("e_idx_comp_in_gate", ("attn_norm", "y"),
         ("attn.indexer.compressor.wgate", "x"), "x_attn_normed")
    node("attn.indexer.weights_proj", "linear", "parameter_op",
         (("x", t["x_attn_normed"]),), (("y", t["idx_weights"]),),
         ("ev_indexer_init",))
    edge("e_idx_wproj", ("attn_norm", "y"),
         ("attn.indexer.weights_proj", "x"), "x_attn_normed")
    node("attn.indexer.cache_slice", "slice_view", "tensor_op",
         (("x", t["idx_cache"]),),
         (("y", t["idx_cache_visible"]),), ("ev_idx_score",))
    edge("e_idx_cache_slice", ("attn.indexer.compressor.cache_out", "cache"),
         ("attn.indexer.cache_slice", "x"), "idx_cache")
    edge("e_idx_cache_state_r", ("attn.indexer.compressor.cache_out", "cache"),
         ("attn.indexer.cache_slice", "x"), "idx_cache", kind="state",
         state_transition_ref="tr_idx_cache")
    node("attn.indexer.score", "einsum_relu", "tensor_op",
         (("q", t["idx_q"]), ("cache", t["idx_cache_visible"])),
         (("scores", t["idx_scores"]),), ("ev_idx_score",))
    edge("e_idx_q_score", ("attn.indexer.q_quant", "y"),
         ("attn.indexer.score", "q"), "idx_q")
    edge("e_idx_cache_score", ("attn.indexer.cache_slice", "y"),
         ("attn.indexer.score", "cache"), "idx_cache_visible")
    node("attn.indexer.head_reduce", "weighted_head_sum", "tensor_op",
         (("scores", t["idx_scores"]), ("weights", t["idx_weights"])),
         (("y", t["idx_score"]),), ("ev_idx_score",))
    edge("e_idx_hred", ("attn.indexer.score", "scores"),
         ("attn.indexer.head_reduce", "scores"), "idx_scores")
    edge("e_idx_w_hred", ("attn.indexer.weights_proj", "y"),
         ("attn.indexer.head_reduce", "weights"), "idx_weights")
    node("attn.indexer.causal_mask", "causal_mask", "tensor_op",
         (("x", t["idx_score"]),), (("y", t["idx_score_masked"]),),
         ("ev_idx_causal",))
    edge("e_idx_mask", ("attn.indexer.head_reduce", "y"),
         ("attn.indexer.causal_mask", "x"), "idx_score",
         guard_ref="g_causal_mask")
    edge("e_idx_mask_topk", ("attn.indexer.causal_mask", "y"),
         ("attn.indexer.topk", "x_masked"), "idx_score_masked",
         guard_ref="g_causal_mask")
    edge("e_idx_direct_topk", ("attn.indexer.head_reduce", "y"),
         ("attn.indexer.topk", "x"), "idx_score", guard_ref="g_decode")
    node("attn.indexer.topk", "topk_select", "index_op",
         (("x_masked", t["idx_score_masked"]), ("x", t["idx_score"])),
         (("indices", t["comp_idxs"]),), ("ev_idx_topk",))
    edge("e_idx_select", ("attn.indexer.topk", "indices"),
         ("attn.select_cat", "compress"), "comp_idxs", kind="select")


def _add_fixed_compress_select_flow(b: _GraphBuilder, t: dict[str, str],
                                    values: ModelValues,
                                    spec: VariantSpec) -> None:
    """HCA：无学习式 Indexer——固定压缩位置选择
    get_compress_topk_idxs（model.py:274-282,519），无 Q/weights_proj/
    自有 Compressor。"""
    def node(sid: str, op: str, gr: str, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid: str, src, dst, role: str, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    # 固定压缩位置选择（model.py:518-519 else 分支）
    node("attn.compress_topk_fixed", "fixed_compress_index", "index_op",
         (), (("y", t["comp_idxs"]),), ("ev_compress_topk_fixed",))
    edge("e_idx_select", ("attn.compress_topk_fixed", "y"),
         ("attn.select_cat", "compress"), "comp_idxs", kind="select")


def _add_cache_and_concat(b: _GraphBuilder, t: dict[str, str],
                          values: ModelValues, spec: VariantSpec) -> None:
    """压缩变体共用的 cache 写读与上下文拼接（V1 相同真实路径）。"""
    def node(sid: str, op: str, gr: str, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid: str, src, dst, role: str, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    node("attn.cache_update", "cache_write", "state_op",
         (("kv", t["kv_quant"]), ("comp_out", t["comp_cache"])),
         (("cache_full", t["attn_cache"]),),
         ("ev_prefill_cache_write", "ev_decode_cache_read",
          "ev_comp_cache_write"))
    edge("e_kv_cache", ("attn.kv_quant", "y"),
         ("attn.cache_update", "kv"), "kv_quant")
    edge("e_comp_cache_w", ("attn.compressor.cache_out", "cache"),
         ("attn.cache_update", "comp_out"), "comp_cache",
         guard_ref="g_compress_prefill")
    edge("e_comp_cache_w_dec", ("attn.compressor.cache_out", "cache"),
         ("attn.cache_update", "comp_out"), "comp_cache",
         guard_ref="g_compress_decode")
    edge("e_cache_state_w", ("attn.cache_update", "cache_full"),
         ("attn.cache_read", "cache_full"), "attn_cache", kind="state",
         state_transition_ref="tr_attn_cache")

    node("attn.select_cat", "concat", "tensor_op",
         (("win", t["win_idxs"]), ("compress", t["comp_idxs"])),
         (("y", t["select_idxs"]),), ("ev_indexer_call",))
    edge("e_cat_win", ("attn.win_topk", "y"), ("attn.select_cat", "win"),
         "win_idxs")

    node("attn.cache_read", "cache_read", "state_op",
         (("cache_full", t["attn_cache"]),),
         (("kv_decode", t["kv_concat"]),),
         ("ev_decode_cache_read",))
    edge("e_sparse_kv_decode", ("attn.cache_read", "kv_decode"),
         ("attn.sparse_attn", "kv_win"), "kv_concat", kind="data",
         guard_ref="g_decode")
    node("attn.kv_concat", "concat", "tensor_op",
         (("win_kv", t["kv_quant"]), ("compress", t["comp_return"])),
         (("y", t["kv_concat"]),), ("ev_prefill_compress",
                                    "ev_decode_cache_read"))
    edge("e_concat_win", ("attn.kv_quant", "y"),
         ("attn.kv_concat", "win_kv"), "kv_quant", guard_ref="g_prefill")
    edge("e_concat_comp", ("attn.compressor.return", "y"),
         ("attn.kv_concat", "compress"), "comp_return",
         guard_ref="g_compress_prefill")
    edge("e_concat_sparse", ("attn.kv_concat", "y"),
         ("attn.sparse_attn", "kv_win"), "kv_concat", guard_ref="g_prefill")


def _add_moe_variant(b: _GraphBuilder, t: dict[str, str],
                     values: ModelValues, spec: VariantSpec) -> None:
    """MoE 变体 gate：Hash（tid2eid token-ID 查表，无偏置）vs TopK
    （偏置选择）。两者共享 score→gather 权重→dispatch→专家→汇合。"""
    def node(sid: str, op: str, gr: str, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid: str, src, dst, role: str, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    edge("e_ffnnorm_gate", ("ffn_norm", "y"), ("moe.gate.score", "x"),
         "x_ffn_normed")
    node("moe.gate.score", "moe_gate_score", "parameter_op",
         (("x", t["x_ffn_normed"]),), (("scores", t["gate_scores"]),),
         ("ev_gate_forward",))
    if spec.hash_layers:
        # Hash：tid2eid 查表（token-ID 依赖），无选择偏置
        # （model.py:561-564 bias=None, 581-582 indices = tid2eid[input_ids]）
        node("moe.gate.tid2eid", "hash_lookup", "index_op",
             (("input_ids", t["gate_input_ids"]),),
             (("indices", t["gate_indices"]),),
             ("ev_gate_init",))
        edge("e_ids_gate", ("block.input_ids_src", "y"),
             ("moe.gate.tid2eid", "input_ids"), "gate_input_ids")
        node("moe.gate.gather", "gather_weights", "tensor_op",
             (("original", t["gate_scores"]), ("indices", t["gate_indices"])),
             (("weights", t["gate_weights"]),),
             ("ev_gate_forward", "ev_weight_norm"))
        edge("e_orig_gather", ("moe.gate.score", "scores"),
             ("moe.gate.gather", "original"), "gate_scores")
        edge("e_idx_gather", ("moe.gate.tid2eid", "indices"),
             ("moe.gate.gather", "indices"), "gate_indices")
    else:
        # TopK：偏置只进选择不进 gather（V1 已验证结构）
        node("moe.gate.bias", "add_bias", "parameter_op",
             (("x", t["gate_scores"]),), (("y", t["gate_scores_biased"]),),
             ("ev_gate_forward",))
        edge("e_score_bias", ("moe.gate.score", "scores"),
             ("moe.gate.bias", "x"), "gate_scores")
        node("moe.gate.topk", "topk_select", "index_op",
             (("x", t["gate_scores_biased"]),),
             (("indices", t["gate_indices"]),), ("ev_gate_forward",))
        edge("e_bias_topk", ("moe.gate.bias", "y"), ("moe.gate.topk", "x"),
             "gate_scores_biased")
        node("moe.gate.gather", "gather_weights", "tensor_op",
             (("original", t["gate_scores"]), ("indices", t["gate_indices"])),
             (("weights", t["gate_weights"]),),
             ("ev_gate_forward", "ev_weight_norm"))
        edge("e_orig_gather", ("moe.gate.score", "scores"),
             ("moe.gate.gather", "original"), "gate_scores")
        edge("e_idx_gather", ("moe.gate.topk", "indices"),
             ("moe.gate.gather", "indices"), "gate_indices")

    node("moe.dispatch", "token_dispatch", "index_op",
         (("indices", t["gate_indices"]),), (("counts", t["n_e"]),),
         ("ev_moe_dispatch",))
    # dispatch 的 indices 来自 gate 的索引生产者（hash=tid2eid / topk=topk），
    # gather 只消费不产出（model.py:584-585 indices 先产、weights 后 gather）
    idx_producer = ("moe.gate.tid2eid" if spec.hash_layers
                    else "moe.gate.topk")
    edge("e_idx_dispatch", (idx_producer, "indices"),
         ("moe.dispatch", "indices"), "gate_indices")
    node("moe.expert_template", "swiglu_expert", "parameter_op",
         (("x", t["x_ffn_normed"]), ("weights", t["gate_weights"])),
         (("y", t["expert_out"]),), ("ev_expert", "ev_moe_dispatch"))
    edge("e_gate_expert", ("ffn_norm", "y"), ("moe.expert_template", "x"),
         "x_ffn_normed")
    edge("e_w_expert", ("moe.gate.gather", "weights"),
         ("moe.expert_template", "weights"), "gate_weights")
    node("moe.shared_experts", "swiglu_expert", "parameter_op",
         (("x", t["x_ffn_normed"]),), (("y", t["shared_out"]),),
         ("ev_moe_shared",))
    edge("e_gate_shared", ("ffn_norm", "y"), ("moe.shared_experts", "x"),
         "x_ffn_normed")
    node("moe.out", "accumulate_add", "tensor_op",
         (("routed", t["expert_out"]), ("shared", t["shared_out"])),
         (("y", t["moe_out"]),), ("ev_moe_shared",))
    edge("e_expert_acc", ("moe.expert_template", "y"), ("moe.out", "routed"),
         "expert_out")
    edge("e_shared_acc", ("moe.shared_experts", "y"), ("moe.out", "shared"),
         "shared_out")


def _add_sparse_and_out(b: _GraphBuilder, t: dict[str, str],
                        values: ModelValues, spec: VariantSpec) -> None:
    """sparse_attn → inverse RoPE → 分组低秩 O → hc_post → FFN hc 轮。"""
    def node(sid: str, op: str, gr: str, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid: str, src, dst, role: str, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    if spec.compress_ratio != 0:
        sparse_inputs = (
            (("q", t["q_rope"]), ("kv_win", t["kv_concat"]),
             ("win_idxs", t["win_idxs"]),
             ("compress_idxs", t["comp_idxs"]),
             ("topk_idxs", t["select_idxs"]),
             ("sink", t["attn_sink_param"])))
        edge("e_sparse_winidx", ("attn.win_topk", "y"),
             ("attn.sparse_attn", "win_idxs"), "win_idxs")
        edge("e_sparse_topk", ("attn.select_cat", "y"),
             ("attn.sparse_attn", "topk_idxs"), "select_idxs")
    else:
        sparse_inputs = (
            (("q", t["q_rope"]), ("kv_win", t["win_cache"]),
             ("win_idxs", t["win_idxs"]),
             ("sink", t["attn_sink_param"])))
        edge("e_sparse_winidx", ("attn.win_topk", "y"),
             ("attn.sparse_attn", "win_idxs"), "win_idxs")
    node("attn.sparse_attn", "sparse_attention", "opaque_kernel",
         sparse_inputs, (("o", t["o_rope"]),),
         ("ev_sparse_attn_kernel", "ev_prefill_compress",
          "ev_decode_cache_read"))
    edge("e_sparse_q", ("attn.q_rope", "y"), ("attn.sparse_attn", "q"),
         "q_rope")
    if spec.compress_ratio == 0:
        # SWA：prefill 当前窗口 kv_quant 即上下文（无压缩拼接）；
        # decode 读 cache。kv_win 端口张量统一 win_cache。
        edge("e_sparse_kv_decode", ("attn.cache_read", "kv_decode"),
             ("attn.sparse_attn", "kv_win"), "win_cache", kind="data",
             guard_ref="g_decode")
        # prefill：本调用窗口 KV 写入 cache 后即为本调用上下文
        # （model.py:523-525 写 → 533 读）
        edge("e_sparse_kv_prefill", ("attn.cache_update", "cache_win"),
             ("attn.cache_read", "cache_win"), "win_cache", kind="state",
             state_transition_ref="tr_attn_cache_win",
             guard_ref="g_prefill")
    # sink 独立参数
    node("attn.attn_sink", "attn_sink_param", "parameter_op",
         (), (("y", t["attn_sink_param"]),), ("ev_attn_init",))
    edge("e_sparse_sink", ("attn.attn_sink", "y"),
         ("attn.sparse_attn", "sink"), "attn_sink_param")

    node("attn.o_rope_inv", "rope_partial_inverse", "tensor_op",
         (("x", t["o_rope"]),), (("y", t["o_rope"]),),
         ("ev_rope_inv", "ev_apply_rotary"))
    edge("e_sparse_oinv", ("attn.sparse_attn", "o"),
         ("attn.o_rope_inv", "x"), "o_rope")
    node("attn.o_reshape", "reshape", "tensor_op",
         (("x", t["o_rope"]),), (("y", t["o_grouped"]),), ("ev_o_group",))
    edge("e_oinv_reshape", ("attn.o_rope_inv", "y"),
         ("attn.o_reshape", "x"), "o_rope")
    node("attn.o_einsum", "grouped_lowrank_einsum", "parameter_op",
         (("x", t["o_grouped"]),), (("y", t["o_lora"]),), ("ev_o_group",))
    edge("e_reshape_einsum", ("attn.o_reshape", "y"),
         ("attn.o_einsum", "x"), "o_grouped")
    node("attn.o_flatten", "flatten", "tensor_op",
         (("x", t["o_lora"]),), (("y", t["o_flat"]),), ("ev_o_group",))
    edge("e_einsum_flat", ("attn.o_einsum", "y"), ("attn.o_flatten", "x"),
         "o_lora")
    node("attn.wo_b", "linear", "parameter_op",
         (("x", t["o_flat"]),), (("y", t["attn_out"]),), ("ev_o_group",))
    edge("e_flat_wob", ("attn.o_flatten", "y"), ("attn.wo_b", "x"), "o_flat")

    node("block.hc_post_attn.combine", "hc_combine", "tensor_op",
         (("x", t["attn_out"]), ("post", t["attn_post"]),
          ("comb", t["attn_comb"]), ("residual", t["x_hc_in"])),
         (("y", t["x_attn_post"]),), ("ev_hc_post",))
    edge("e_wob_post", ("attn.wo_b", "y"),
         ("block.hc_post_attn.combine", "x"), "attn_out")
    edge("e_residual_a", ("block.hc_pre_attn.flatten", "residual"),
         ("block.hc_post_attn.combine", "residual"), "x_hc_in")

    node("block.hc_pre_ffn.flatten", "flatten", "tensor_op",
         (("x", t["x_attn_post"]),),
         (("y", t["x_ffn_flat_hc"]), ("residual", t["x_attn_post"])),
         ("ev_hc_pre",))
    edge("e_hcpost_a_out", ("block.hc_post_attn.combine", "y"),
         ("block.hc_pre_ffn.flatten", "x"), "x_attn_post")
    node("block.hc_pre_ffn.mix", "linear", "parameter_op",
         (("x", t["x_ffn_flat_hc"]),), (("y", t["ffn_mixes"]),),
         ("ev_hc_pre",))
    node("block.hc_pre_ffn.sinkhorn", "sinkhorn_split", "opaque_kernel",
         (("mixes", t["ffn_mixes"]),),
         (("pre", t["ffn_pre"]), ("post", t["ffn_post"]),
          ("comb", t["ffn_comb"])),
         ("ev_hc_pre", "ev_hc_sinkhorn_kernel"))
    node("block.hc_pre_ffn.reduce", "weighted_sum", "tensor_op",
         (("pre", t["ffn_pre"]), ("x", t["x_attn_post"])),
         (("y", t["x_ffn"]),), ("ev_hc_pre",))
    edge("e_hcpre_f_flat", ("block.hc_pre_ffn.flatten", "y"),
         ("block.hc_pre_ffn.mix", "x"), "x_ffn_flat_hc")
    edge("e_hcpre_f_mix", ("block.hc_pre_ffn.mix", "y"),
         ("block.hc_pre_ffn.sinkhorn", "mixes"), "ffn_mixes")
    edge("e_hcpre_f_pre", ("block.hc_pre_ffn.sinkhorn", "pre"),
         ("block.hc_pre_ffn.reduce", "pre"), "ffn_pre")
    edge("e_hcpre_f_x", ("block.hc_pre_ffn.flatten", "residual"),
         ("block.hc_pre_ffn.reduce", "x"), "x_attn_post")
    edge("e_hcpre_f_out", ("block.hc_pre_ffn.reduce", "y"),
         ("ffn_norm", "x"), "x_ffn")
    edge("e_hcpre_f_post", ("block.hc_pre_ffn.sinkhorn", "post"),
         ("block.hc_post_ffn.combine", "post"), "ffn_post")
    edge("e_hcpre_f_comb", ("block.hc_pre_ffn.sinkhorn", "comb"),
         ("block.hc_post_ffn.combine", "comb"), "ffn_comb")

    node("ffn_norm", "rms_norm", "parameter_op",
         (("x", t["x_ffn"]),), (("y", t["x_ffn_normed"]),),
         ("ev_attn_norm",))

    node("block.hc_post_ffn.combine", "hc_combine", "tensor_op",
         (("x", t["moe_out"]), ("post", t["ffn_post"]),
          ("comb", t["ffn_comb"]), ("residual", t["x_attn_post"])),
         (("y", t["x_hc_out"]),), ("ev_hc_post",))
    edge("e_moe_post", ("moe.out", "y"),
         ("block.hc_post_ffn.combine", "x"), "moe_out")
    edge("e_residual_f", ("block.hc_pre_ffn.flatten", "residual"),
         ("block.hc_post_ffn.combine", "residual"), "x_attn_post")


def _guard_objects_variant(registry: EvidenceRegistry,
                           spec: VariantSpec) -> tuple[Guard, ...]:
    """按变体过滤 guard：SWA 无压缩/Indexer guard；HCA 无 Indexer guard。"""
    skip = set()
    if spec.compress_ratio == 0:
        skip |= {"g_compress_csa", "g_has_indexer", "g_compress_prefill",
                 "g_compress_prefill_skip", "g_compress_prefill_remainder",
                 "g_compress_decode", "g_decode_no_compress", "g_causal_mask"}
    elif not spec.has_indexer:
        skip |= {"g_has_indexer", "g_causal_mask"}
    return tuple(
        Guard(stable_id=g.stable_id, predicate_expr=g.predicate_expr,
              evidence_refs=(g.evidence_ref,))
        for g in GUARD_DEFS if g.stable_id not in skip
    )


def build_all_variant_graphs(
    workspace_root: Path,
    *,
    values: ModelValues | None = None,
) -> dict[str, tuple[SemanticGraph, EvidenceRegistry, ModelValues]]:
    """构建全部四类变体代表图（csa_topk 委托 V1）。"""
    values = values or load_model_values(Path(workspace_root))
    results = {}
    for key, spec in VARIANT_SPECS.items():
        results[key] = build_variant_graph(workspace_root, spec, values=values)
    return results
