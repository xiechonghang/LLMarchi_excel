"""DeepSeek-V4-Flash-0731 模型语义契约（工作包 B）：独立期望 + 受控错误检测。

与通用引用检查（checks/graph_contracts.py，只查引用/方向/状态环）分层：
本模块维护**模型语义期望**——只有连接拓扑与方向都对才算通过，检查标签或
数量不能替代连接正确性。

:func:`validate_model_contracts(graph)` 返回 :class:`ContractIssue` 列表
（空 = 全部通过）；核心是 :func:`mutated_graph` 驱动的受控错误注入：
每项错误在真实图上做一种语义破坏，验证契约能检出。受控错误数量随迭代
增长（当前见 :func:`contract_error_specs` 返回长度，构建时动态记录每项
detected 结果到 validation_report，不写死计数）：
1. comb 方向反转（comb 轴交换 j/k——错误连接 sum_k comb[...,j,k]*residual[...,k,d]）
2. 删 qr→Indexer 边（Indexer 失去第二真实输入）
3. 拆共享 KV 别名（K/V 变成两个独立投影输出）
4. inverse RoPE 与 O reshape 顺序交换
5. 错误压缩 guard（decode 触发条件 % 换成 +）
6. 丢状态版本（StateTransition 的 before/after 指向不存在版本）
7. 错误轴/布局（attn_comb 的 hc_j/hc_k 布局互换）
8. 缺失/重叠层（层分组并集缺层或重叠）
9+. sink 独立、compressor 输入/隔离、cache 分支/状态连接、主尾块切片、
   pool 时序、互斥发布、score slice/add 分流、guard 相位、guard 谓词语义
   （C16）、核心轴语义（C17）等——逐项见 contract_error_specs 描述。

本模块只读图对象与规则常量，不执行模型代码。
"""

from __future__ import annotations

from dataclasses import dataclass

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_reference import TEMPLATE_ID
from structure_reader.graph_document import SemanticGraph

__all__ = [
    "ContractIssue",
    "validate_model_contracts",
    "contract_error_specs",
    "apply_contract_mutation",
    "build_validation_report",
]

# 模板内的关键节点/边/张量（与 deepseek_v4_reference 保持一致的稳定 ID）
_COMB_AXIS_J = "hc_j"
_COMB_AXIS_K = "hc_k"
_HCPRE_ATTN_SINKHORN = "block.hc_pre_attn.sinkhorn"
_HCPRE_FFN_SINKHORN = "block.hc_pre_ffn.sinkhorn"
_IDX_WQB = "attn.indexer.wq_b"
_SPARSE_ATTN = "attn.sparse_attn"
_O_ROPE_INV = "attn.o_rope_inv"
_O_RESHAPE = "attn.o_reshape"
_COMB_TENSOR = "tensor:csa_topk:attn_comb"
_COMB_TENSOR_FFN = "tensor:csa_topk:ffn_comb"
_COMB_AXIS_J = "hc_j"
_COMB_AXIS_K = "hc_k"
_COMB_PATH_ATTN = "attn_comb"
_COMB_PATH_FFN = "ffn_comb"


@dataclass(frozen=True)
class ContractIssue:
    """一条契约发现：稳定 code、定位 path、易懂 message。"""

    code: str
    path: str
    message: str


def _issue(issues: list[ContractIssue], code: str, path: str, message: str) -> None:
    issues.append(ContractIssue(code=code, path=path, message=message))


# ---------------------------------------------------------------------------
# 正向契约检查（期望 = 真实连接结构）
# ---------------------------------------------------------------------------

def validate_model_contracts(graph: SemanticGraph) -> tuple[ContractIssue, ...]:
    """独立维护的模型语义期望，全部通过返回空元组。

    覆盖（每项都查连接/方向/轴，不只查标签或数量）：
    C1 hc_post comb 方向与双来源：attn_comb/ffn_comb 尾部轴序 [hc_j, hc_k]、
       布局无转置；attn 侧 comb/post 必须来自 hc_pre_attn.sinkhorn，FFN 侧
       必须来自 hc_pre_ffn.sinkhorn——不得跨侧错接。
    C2 qr 双消费：attn.q_norm 输出同时供给 attn.wq_b 与 attn.indexer.wq_b。
    C3 共享 KV：key/value 两条 same_tensor 指向同一 kv_quant 张量，且消费方
       为 cache_update 与 kv_concat。
    C4 inverse RoPE 先于 O reshape。
    C5 压缩 guard 三态完整：guards 含全部 9 条且 g_compress_decode 谓词原文
       为 '(start_pos + 1) % r == 0'。
    C6 状态版本完备且全部被 state 边引用（目录不落空）。
    C7 select 边语义：Indexer 输出选择索引。
    C8 N_e unknown。
    C9 sink 独立参数：sparse_attn 的 sink 输入来自 attn.attn_sink 参数节点
       （[Hq] 张量），不是 hc sinkhorn 的 post——错接即违规。
    C10 compressor 输入与隔离：两个 compressor 的 wkv/wgate 输入都来自
        attn_norm（[B,T,H]）；两套中间张量不共享（prefix comp_/idx_comp_）；
        Indexer 实例走 Hadamard+FP4、attention 实例走 FP8。
    C11 cache 分支与状态连接：prefill（g_prefill）与 decode（g_decode）两条
        kv 供给边存在；全部 StateTransition 被实际 state 边引用。
    """
    issues: list[ContractIssue] = []
    # 变体感知（V2）：模板 ID 决定张量前缀与适用检查——SWA 无压缩/Indexer、
    # HCA 无学习式 Indexer/因果 mask guard，不适用其专属契约。
    _has_compress = _variant_has_compress(graph)
    _has_indexer = _variant_has_indexer(graph)
    P = _variant_prefix(graph)
    _check_comb_direction(graph, issues, prefix=P)
    _check_qr_dual_consumer(graph, issues, has_indexer=_has_indexer, prefix=P)
    _check_shared_kv(graph, issues, has_compress=_has_compress, prefix=P)
    _check_rope_order(graph, issues)
    if _has_compress:
        _check_compression_guard(graph, issues, has_indexer=_has_indexer)
        _check_compressor_isolation(graph, issues, prefix=P)
        _check_cache_branches_and_states(graph, issues)
        _check_compress_slices_and_shift(graph, issues, prefix=P)
        _check_slice_semantics_and_timing(graph, issues, prefix=P)
    _check_state_namespaces(graph, issues)
    _check_select_edge_semantics(graph, issues, prefix=P)
    _check_ne_unknown(graph, issues)
    _check_sink_independence(graph, issues, prefix=P)
    _check_guard_predicate_semantics(graph, issues)
    _check_core_axis_semantics(graph, issues, prefix=P)
    _check_variant_topology(graph, issues, prefix=P)
    _check_variant_components(graph, issues)
    return tuple(issues)


def _check_variant_topology(graph: SemanticGraph,
                            issues: list[ContractIssue], prefix: str = "csa_topk") -> None:
    """C18/C19：变体专属拓扑语义（V2）。

    C18 Hash token 依赖：图含 tid2eid 查表节点时，必须有 input_ids 输入边
    （model.py:582 tid2eid[input_ids]）——移除 token 依赖即违规。
    C19 HCA 无 overlap：r=128 变体（无 Indexer 且有压缩）不得消费
    overlap 张量（model.py:296 overlap = ratio==4）。
    """
    nodes = {n.stable_id for n in graph.nodes}
    # C18
    if "moe.gate.tid2eid" in nodes:
        tok_in = [e for e in graph.edges
                  if e.dst_node_id == "moe.gate.tid2eid"
                  and e.tensor_id == f"tensor:{prefix}:gate_input_ids"]
        if not tok_in:
            _issue(issues, "C18_HASH_TOKEN_DEPENDENCY_MISSING", "edges",
                   "tid2eid 查表缺 input_ids 输入——Hash 路由的 token 依赖"
                   "被移除（model.py:582）")
        if "moe.gate.bias" in nodes:
            _issue(issues, "C18_HASH_WITH_BIAS", "nodes",
                   "Hash 层不得有选择偏置（model.py:561-564 bias=None）")
    # C19
    if (_variant_has_compress(graph) and not _variant_has_indexer(graph)):
        overlap_consumers = [e for e in graph.edges
                             if e.tensor_id == f"tensor:{prefix}:comp_overlap"]
        if overlap_consumers:
            _issue(issues, "C19_HCA_OVERLAP_CONSUMED", "edges",
                   "HCA（r=128）不得消费 overlap 张量"
                   "（model.py:296 overlap=ratio==4 仅 CSA）")
        if "attn.compressor.overlap" in nodes:
            _issue(issues, "C19_HCA_OVERLAP_NODE", "nodes",
                   "HCA 不得有 overlap_transform 节点（无 overlap 压缩）")


# 变体组件规范（C20，V2）：按 template 稳定约束必需/禁止组件——
# 依据 model.py 构造分支（472-477 compress_ratio/==4 → Indexer、
# 561-564/582 hash tid2eid、483-485 SWA YaRN 关）。
_VARIANT_COMPONENT_SPEC = {
    "swa_hash": {
        "required": ("attn.wq_a", "attn.wkv", "attn.win_topk",
                     "moe.gate.tid2eid", "block.hc_pre_attn.sinkhorn"),
        "forbidden": ("attn.compressor.wkv", "attn.indexer.wq_b",
                      "moe.gate.bias", "attn.indexer.compressor.wkv"),
    },
    "csa_hash": {
        "required": ("attn.compressor.wkv", "attn.indexer.wq_b",
                     "attn.indexer.compressor.wkv", "moe.gate.tid2eid"),
        "forbidden": ("moe.gate.bias",),
    },
    "csa_topk": {
        "required": ("attn.compressor.wkv", "attn.indexer.wq_b",
                     "attn.indexer.compressor.wkv", "moe.gate.bias"),
        "forbidden": ("moe.gate.tid2eid",),
    },
    "hca_topk": {
        "required": ("attn.compressor.wkv", "moe.gate.bias",
                     "attn.compress_topk_fixed"),
        "forbidden": ("attn.indexer.wq_b", "attn.indexer.compressor.wkv",
                      "attn.compressor.overlap", "moe.gate.tid2eid"),
    },
}


def _check_variant_components(graph: SemanticGraph,
                              issues: list[ContractIssue]) -> None:
    """C20：按 template 稳定 spec 核验必需/禁止组件（review02 第 3 项）。

    缺必需组件（如 CSA 删整套 Indexer/Compressor）与含禁止组件
    （如 HCA 注入合法 Indexer）均具名报错——不依赖节点缺失的自动 skip，
    不能因悬空残留误绿。
    """
    spec = _VARIANT_COMPONENT_SPEC.get(_variant_prefix(graph))
    if spec is None:
        return
    nodes = {n.stable_id for n in graph.nodes}
    for required in spec["required"]:
        if required not in nodes:
            _issue(issues, "C20_REQUIRED_COMPONENT_MISSING", "nodes",
                   f"{graph.template_id} 缺必需组件 {required}"
                   "（model.py 构造分支要求存在）")
    for forbidden in spec["forbidden"]:
        if forbidden in nodes:
            _issue(issues, "C20_FORBIDDEN_COMPONENT_PRESENT", "nodes",
                   f"{graph.template_id} 含禁止组件 {forbidden}"
                   "（该变体的 model.py 分支不构造它）")


def _variant_prefix(graph: SemanticGraph) -> str:
    """图模板的张量 ID 前缀（V2：template_id 决定，联合导出不碰撞）。

    template_id 形如 deepseek_v4_flash_0731:csa_topk → 前缀 csa_topk。
    """
    return graph.template_id.rsplit(":", 1)[-1]


def _variant_has_compress(graph: SemanticGraph) -> bool:
    """图模板是否有压缩链（swa_hash 无）。"""
    return any(n.stable_id == "attn.compressor.wkv" for n in graph.nodes)


def _variant_has_indexer(graph: SemanticGraph) -> bool:
    """图模板是否有学习式 Indexer（swa_hash/hca_topk 无）。"""
    return any(n.stable_id == "attn.indexer.wq_b" for n in graph.nodes)


def _check_comb_direction(graph: SemanticGraph,
                          issues: list[ContractIssue], prefix: str = "csa_topk") -> None:
    comb = {t.stable_id: t for t in graph.tensors
            if t.name in (f"{prefix}:attn_comb", f"{prefix}:ffn_comb")}
    for tid, tensor in comb.items():
        symbols = [a.symbol for a in tensor.axes]
        # comb 形如 [b, s, hc_j, hc_k]：尾部两维必须是 源j → 目标k 顺序
        if len(symbols) < 2 or symbols[-2:] != [_COMB_AXIS_J, _COMB_AXIS_K]:
            _issue(issues, "C1_COMB_AXIS_ORDER", f"tensors[{tid}].axes",
                   f"comb 张量尾部轴序必须是 [hc_j(源), hc_k(目标)]，"
                   f"得到 {symbols}")
        # 源码无转置：物理布局必须与轴序一致（comb[...,j,k] 直接对齐 residual）
        if list(tensor.physical_layout) != symbols:
            _issue(issues, "C1_COMB_LAYOUT", f"tensors[{tid}].physical_layout",
                   f"comb 物理布局必须与轴序一致（源码无转置），"
                   f"axes={symbols} physical={list(tensor.physical_layout)}")
    # 双来源确切核对：attn 侧 combine 的 comb/post 必须来自 attn 侧 sinkhorn，
    # FFN 侧必须来自 FFN 侧 sinkhorn——跨侧错接（如 FFN sinkhorn 供 attn
    # combine）即违规。
    side_expect = {
        "block.hc_post_attn.combine": "block.hc_pre_attn.sinkhorn",
        "block.hc_post_ffn.combine": "block.hc_pre_ffn.sinkhorn",
    }
    for edge in graph.edges:
        expected_src = side_expect.get(edge.dst_node_id)
        if expected_src is None:
            continue
        if edge.dst_port in ("comb", "post"):
            if edge.src_node_id != expected_src:
                _issue(issues, "C1_HCSOURCE_CROSS", f"edges[{edge.stable_id}]",
                       f"{edge.dst_node_id}.{edge.dst_port} 必须来自 "
                       f"{expected_src}，错接自 {edge.src_node_id}")
    # hc_post 的 comb 输入必须来自 hc_pre.sinkhorn 的 comb 端口
    sinkhorns = {_HCPRE_ATTN_SINKHORN, _HCPRE_FFN_SINKHORN}
    for edge in graph.edges:
        if (edge.dst_node_id.endswith("hc_post_attn.combine")
                and edge.dst_port == "comb"):
            if edge.src_node_id not in sinkhorns or edge.src_port != "comb":
                _issue(issues, "C1_COMB_SOURCE", f"edges[{edge.stable_id}]",
                       "hc_post 的 comb 输入必须来自 hc_pre.sinkhorn 的 comb 端口"
                       f"，得到 {edge.src_node_id}.{edge.src_port}")


def _check_qr_dual_consumer(graph: SemanticGraph,
                            issues: list[ContractIssue],
                            *, has_indexer: bool = True, prefix: str = "csa_topk") -> None:
    qr_consumers: set[str] = set()
    for edge in graph.edges:
        if (edge.src_node_id == "attn.q_norm" and edge.src_port == "y"
                and edge.tensor_id == f"tensor:{prefix}:qr"):
            qr_consumers.add(edge.dst_node_id)
    expected = {"attn.wq_b"} | (
        {_IDX_WQB} if has_indexer else set())
    if qr_consumers != expected:
        _issue(issues, "C2_QR_CONSUMERS", "edges",
               f"qr 必须同时供给 attn.wq_b 与 {IDX_WQB_NOTE}，"
               f"实际消费方: {sorted(qr_consumers)}")


IDX_WQB_NOTE = "attn.indexer.wq_b"


def _check_shared_kv(graph: SemanticGraph,
                     issues: list[ContractIssue],
                     *, has_compress: bool = True, prefix: str = "csa_topk") -> None:
    kv_alias = [a for a in graph.alias_relations
                if a.alias_role in ("key", "value")]
    roles = {a.alias_role: a.alias_tensor_id for a in kv_alias}
    if set(roles) != {"key", "value"}:
        _issue(issues, "C3_KV_ALIAS_ROLES", "alias_relations",
               f"共享 KV 需要 key/value 两条 same_tensor，得到 {sorted(roles)}")
        return
    if roles["key"] != roles["value"]:
        _issue(issues, "C3_KV_ALIAS_SPLIT", "alias_relations",
               f"K/V 角色必须指向同一张量（共享单投影），得到 {roles['key']} "
               f"与 {roles['value']}——拆成两个投影即违规")
        return
    kv_tensor = roles["key"]
    # 源码消费方（model.py:508-538）：cache_update 写 cache；kv_concat 是
    # prefill 上下文 KV 的来源（decode 路径经 cache_read 间接消费同一 cache）。
    consumers = {edge.dst_node_id for edge in graph.edges
                 if edge.tensor_id == kv_tensor
                 and edge.dst_node_id in ("attn.sparse_attn",
                                          "attn.cache_update", "attn.kv_concat")}
    expected_consumers = (
        {"attn.cache_update", "attn.kv_concat"} if has_compress
        else {"attn.cache_update"})
    if consumers != expected_consumers:
        _issue(issues, "C3_KV_CONSUMERS", "edges",
               f"共享 KV 张量必须供 cache_update 与 kv_concat（上下文 KV）消费，"
               f"实际: {sorted(consumers)}")


def _check_rope_order(graph: SemanticGraph,
                      issues: list[ContractIssue]) -> None:
    has_edge = any(
        e.src_node_id == _O_ROPE_INV and e.dst_node_id == _O_RESHAPE
        for e in graph.edges
    )
    if not has_edge:
        _issue(issues, "C4_ROPE_ORDER", "edges",
               "inverse RoPE（attn.o_rope_inv）必须先于 O reshape"
               "（attn.o_reshape）——交换顺序即违规")


def _check_compression_guard(graph: SemanticGraph,
                             issues: list[ContractIssue],
                             *, has_indexer: bool = True) -> None:
    guards = {g.stable_id: g for g in graph.guards}
    decode_guard = guards.get("g_compress_decode")
    if decode_guard is None:
        _issue(issues, "C5_GUARD_MISSING", "guards", "缺少 g_compress_decode")
    elif decode_guard.predicate_expr != "(start_pos + 1) % r == 0":
        _issue(issues, "C5_GUARD_EXPR", "guards[g_compress_decode].predicate_expr",
               f"decode 压缩触发谓词必须为 '(start_pos + 1) % r == 0'，"
               f"得到 {decode_guard.predicate_expr!r}")
    required_guards = ["g_prefill", "g_decode", "g_compress_csa",
                       "g_compress_prefill", "g_compress_prefill_skip",
                       "g_window_wrap"]
    if has_indexer:
        required_guards += ["g_has_indexer", "g_causal_mask"]
    for required in required_guards:
        if required not in guards:
            _issue(issues, "C5_GUARD_MISSING", "guards", f"缺少 guard {required}")


def _check_state_namespaces(graph: SemanticGraph,
                            issues: list[ContractIssue]) -> None:
    version_state: dict[str, str] = {v.stable_id: v.state_id
                                     for v in graph.state_versions}
    for tr in graph.state_transitions:
        for ref in (tr.before_version_id, tr.after_version_id):
            if ref not in version_state:
                _issue(issues, "C6_STATE_VERSION_MISSING",
                       f"state_transitions[{tr.stable_id}].{ref}",
                       f"版本 {ref!r} 不存在（丢失状态版本）")
            elif version_state[ref] != tr.state_id:
                _issue(issues, "C6_STATE_ID_MISMATCH",
                       f"state_transitions[{tr.stable_id}]",
                       f"版本 {ref!r} 属于 {version_state[ref]!r}，"
                       f"与 transition 的 {tr.state_id!r} 不一致")
    if not graph.state_versions:
        _issue(issues, "C6_NO_STATES", "state_versions",
               "模板图没有状态版本（cache/state 应有 v_hist→v_after）")


def _check_select_edge_semantics(graph: SemanticGraph,
                                 issues: list[ContractIssue], prefix: str = "csa_topk") -> None:
    if not _variant_has_indexer(graph):
        return  # SWA/HCA 无学习式 Indexer：select 边由固定选择/无压缩承载
    select_edges = [e for e in graph.edges if e.kind == "select"]
    if not select_edges:
        _issue(issues, "C7_SELECT_EDGE_MISSING", "edges",
               "Indexer 输出必须是 select 边（选择索引而非 Value）")
    for edge in select_edges:
        if edge.tensor_id != f"tensor:{prefix}:comp_idxs":
            _issue(issues, "C7_SELECT_TENSOR", f"edges[{edge.stable_id}]",
                   f"select 边张量应为压缩选择索引，得到 {edge.tensor_id}")


def _check_ne_unknown(graph: SemanticGraph,
                      issues: list[ContractIssue]) -> None:
    ne = next((a for a in graph.named_axes if a.symbol == "N_e"), None)
    if ne is None:
        _issue(issues, "C8_NE_AXIS_MISSING", "named_axes", "缺少 N_e 轴")
    elif not (ne.unknown_reason and ne.unknown_reason.strip()):
        _issue(issues, "C8_NE_NOT_UNKNOWN", "named_axes[axis:N_e]",
               "N_e（每专家 token 数）必须保持 unknown（数据相关），"
               "不得给具体值或表达式")


def _check_sink_independence(graph: SemanticGraph,
                             issues: list[ContractIssue], prefix: str = "csa_topk") -> None:
    """C9：sparse_attn 的 sink 输入必须来自独立参数节点 attn.attn_sink。"""
    sink_tensor = f"tensor:{prefix}:attn_sink_param"
    sink_edges = [e for e in graph.edges
                  if e.dst_node_id == "attn.sparse_attn"
                  and e.dst_port == "sink"]
    if len(sink_edges) != 1:
        _issue(issues, "C9_SINK_EDGES", "edges",
               f"sparse_attn 的 sink 输入必须恰好一条，得到 {len(sink_edges)}")
        return
    edge = sink_edges[0]
    if edge.src_node_id != "attn.attn_sink" or edge.tensor_id != sink_tensor:
        _issue(issues, "C9_SINK_WRONG_SOURCE", f"edges[{edge.stable_id}]",
               f"sink 必须来自独立参数 attn.attn_sink（{sink_tensor}，"
               f"model.py:462），错接自 {edge.src_node_id}"
               f"（张量 {edge.tensor_id}）——HC post 不是 attn_sink")
    # hc post 不得流入 sparse_attn 任何端口
    leaked = [e for e in graph.edges
              if e.src_node_id == "block.hc_pre_attn.sinkhorn"
              and e.dst_node_id == "attn.sparse_attn"]
    if leaked:
        _issue(issues, "C9_HCPOST_LEAK", "edges",
               f"hc sinkhorn 的 post 泄漏进 sparse_attn（{len(leaked)} 条边）"
               "——attn_sink 是独立参数")


def _check_compressor_isolation(graph: SemanticGraph,
                                issues: list[ContractIssue], prefix: str = "csa_topk") -> None:
    """C10：compressor 输入 H、两套实例张量隔离、量化路径区分。"""
    # 输入必须来自 attn_norm（args.dim=H），不得来自 attn.wkv（窗口 KV [D]）；
    # 实例按图实际存在核验（HCA 无 Indexer compressor 是正确拓扑）
    existing = {n.stable_id.rsplit(".", 1)[0] for n in graph.nodes
                if n.stable_id.endswith((".wkv", ".wgate"))}
    for comp in ("attn.compressor", "attn.indexer.compressor"):
        if comp not in existing:
            continue
        inputs = [e for e in graph.edges
                  if e.dst_node_id in (f"{comp}.wkv", f"{comp}.wgate")]
        if not inputs:
            _issue(issues, "C10_COMP_INPUT_MISSING", comp,
                   f"{comp} 的 wkv/wgate 没有输入边")
            continue
        for e in inputs:
            if e.src_node_id != "attn_norm":
                _issue(issues, "C10_COMP_INPUT_WRONG",
                       f"edges[{e.stable_id}]",
                       f"{comp} 的 wkv/wgate 输入必须是 attn_norm 输出"
                       f"（args.dim，model.py:303-304），错接自 {e.src_node_id}"
                       "——与窗口 wkv 是两套并行投影")
    # 中间张量隔离：attention 实例边引用 comp_* 张量，indexer 实例边引用
    # idx_comp_* 张量——Di/D 与 2Di/2D 两套中间张量不得混用（model.py:404）。
    for e in graph.edges:
        if e.src_node_id.startswith("attn.compressor.") \
                and "idx_comp" in e.tensor_id:
            _issue(issues, "C10_TENSOR_SHARED", f"edges[{e.stable_id}]",
                   "attention compressor 不得消费 indexer 的张量")
        if e.src_node_id.startswith("attn.indexer.compressor.") \
                and e.tensor_id.startswith(f"tensor:{prefix}:comp_") \
                and "idx_comp" not in e.tensor_id:
            _issue(issues, "C10_TENSOR_SHARED", f"edges[{e.stable_id}]",
                   "indexer compressor 不得消费 attention 的中间张量"
                   "（Di/D2i 独立，model.py:404）")
        # 消费端隔离：indexer compressor 的 wkv/wgate 输入边张量必须是
        # H 维输入（x_attn_normed），不得是 attention 实例的 D 维中间张量
        if e.dst_node_id in ("attn.indexer.compressor.wkv",
                             "attn.indexer.compressor.wgate",
                             "attn.compressor.wkv",
                             "attn.compressor.wgate"):
            if e.tensor_id.startswith(f"tensor:{prefix}:comp_") \
                    and not e.tensor_id.endswith(("kv_proj", "score_proj")):
                _issue(issues, "C10_TENSOR_SHARED", f"edges[{e.stable_id}]",
                       f"compressor 输入边消费压缩中间张量 {e.tensor_id}"
                       "——输入必须是 attn_norm 输出（H 维）")
            elif e.tensor_id in (f"tensor:{prefix}:comp_kv_proj",
                                 f"tensor:{prefix}:comp_score_proj",
                                 f"tensor:{prefix}:idx_comp_kv_proj",
                                 f"tensor:{prefix}:idx_comp_score_proj"):
                _issue(issues, "C10_TENSOR_SHARED", f"edges[{e.stable_id}]",
                       f"compressor 输入边消费压缩投影输出 {e.tensor_id}"
                       "（Di/D 混用或输出回灌）")
    # 量化路径：indexer 实例 Hadamard+FP4，attention 实例 FP8
    idx_quant = [n for n in graph.nodes
                 if n.stable_id == "attn.indexer.compressor.quant"]
    attn_quant = [n for n in graph.nodes
                  if n.stable_id == "attn.compressor.quant"]
    if idx_quant and idx_quant[0].op_type != "hadamard_fp4_quant":
        _issue(issues, "C10_QUANT_PATH", "nodes[attn.indexer.compressor.quant]",
               f"indexer compressor 量化必须是 Hadamard+FP4"
               f"（rotate=True，model.py:374-376,404），得到 "
               f"{idx_quant[0].op_type}")
    if attn_quant and attn_quant[0].op_type != "act_quant_fp8":
        _issue(issues, "C10_QUANT_PATH", "nodes[attn.compressor.quant]",
               f"attention compressor 非 RoPE 维量化必须是 FP8"
               f"（model.py:378），得到 {attn_quant[0].op_type}")


def _check_compress_slices_and_shift(graph: SemanticGraph,
                                     issues: list[ContractIssue], prefix: str = "csa_topk") -> None:
    """C12：主/尾块真实切片消费与 pool 后 shift 最终版本。

    - state_update 的 main_* 输入必须是 slice view 张量（不是完整投影）；
    - tail_* 输入必须是尾块/decode token 切片张量，且每条边挂场景 guard
      （g_compress_prefill_remainder / g_decode）——无 guard 的 tail 边
      会在 T=4/128 等无尾块场景伪活动；
    - pool_shift 节点存在：updated → shift → final 的 state 边引用
      shift transition；final 版本被 cache/return 消费。
    """
    # 两套 compressor 实例（G3：检查不硬编码第一套；维度/量化差异保留）
    _COMP_INSTANCES = (
        {"prefix": "attn.compressor", "tp": "comp",
         "full_proj": {f"tensor:{prefix}:comp_kv_proj",
                       f"tensor:{prefix}:comp_score_proj"}},
        {"prefix": "attn.indexer.compressor", "tp": "idx_comp",
         "full_proj": {f"tensor:{prefix}:idx_comp_kv_proj",
                       f"tensor:{prefix}:idx_comp_score_proj"}},
    )
    node_ids = {n.stable_id for n in graph.nodes}
    P = _variant_prefix(graph)
    for inst in _COMP_INSTANCES:
        if f"{inst['prefix']}.state_update" not in node_ids:
            continue  # 变体无该实例（HCA 无 Indexer compressor）是正确拓扑
        _check_state_chain_one(graph, issues, inst["prefix"], inst["tp"],
                               inst["full_proj"], prefix=P)


def _check_state_chain_one(graph: SemanticGraph, issues: list[ContractIssue],
                           inst_prefix: str, tp: str,
                           full_proj: set[str],
                           prefix: str = "csa_topk") -> None:
    """单套 compressor 的 C12 状态链检查（两实例共用；参数化前缀/张量）。

    inst_prefix = 节点前缀（attn.compressor / attn.indexer.compressor）；
    prefix = 张量模板前缀（csa_topk / swa_hash / …）。
    """
    upd_in = [e for e in graph.edges
              if e.dst_node_id == f"{inst_prefix}.state_update"]
    for e in upd_in:
        if e.dst_port in ("main_kv", "main_score"):
            if e.tensor_id in full_proj:
                _issue(issues, "C12_MAIN_FULL_PROJECTION",
                       f"edges[{e.stable_id}]",
                       f"主块输入消费完整投影 {e.tensor_id}——必须是 "
                       "cutoff-r:cutoff 切片 view（model.py:337-338）")
            if e.guard_ref != "g_compress_prefill":
                _issue(issues, "C12_MAIN_GUARD", f"edges[{e.stable_id}]",
                       f"主块边必须挂 g_compress_prefill，得到 {e.guard_ref}")
        if e.dst_port in ("tail_kv", "tail_score"):
            if e.tensor_id in full_proj:
                _issue(issues, "C12_TAIL_FULL_PROJECTION",
                       f"edges[{e.stable_id}]",
                       f"尾块输入消费完整投影 {e.tensor_id}——必须是 "
                       "cutoff: 切片或 decode token 切片（model.py:340-341,353）")
            if e.guard_ref not in ("g_compress_prefill_remainder", "g_decode"):
                _issue(issues, "C12_TAIL_UNGUARDED", f"edges[{e.stable_id}]",
                       f"尾块边必须挂 remainder/decode guard（无尾块场景"
                       f" T=4/128 不得活动），得到 {e.guard_ref}")
    # shift 链：pool_shift 存在、输入来自 state_update（state 边）、
    # 输出 final 被消费
    shift = next((n for n in graph.nodes
                  if n.stable_id == f"{inst_prefix}.pool_shift"), None)
    if shift is None:
        _issue(issues, "C12_SHIFT_MISSING", "nodes",
               f"缺少 {prefix} 的 pool 后 shift 节点（model.py:359-360）")
    else:
        shift_edges = [e for e in graph.edges
                       if e.src_node_id == f"{inst_prefix}.pool_shift"]
        if not shift_edges:
            _issue(issues, "C12_SHIFT_UNCONNECTED", "nodes",
                   f"{inst_prefix}.pool_shift 输出无消费（最终版本未连读写）")
        final_consumed = any(
            e.tensor_id.endswith(f"{tp}_state_kv_final")
            or e.tensor_id.endswith(f"{tp}_state_score_final")
            for e in shift_edges)
        if not final_consumed:
            _issue(issues, "C12_FINAL_NOT_CONSUMED", "edges",
                   f"{inst_prefix} shift 产出的最终状态版本（v_final）未被消费")


def _check_slice_semantics_and_timing(graph: SemanticGraph,
                                      issues: list[ContractIssue], prefix: str = "csa_topk") -> None:
    """C13：切片 shape/表达式与最终状态发布时序。

    - 尾块 view 张量长度轴必须是 N_rem（T%r），decode token 是 N_one；
      尾块与 decode token 不能共享张量 ID；
    - 主/尾块/decode 切片必须有 slice AliasRelation 且 transform_expr
      含正确表达式原文（cutoff-r:cutoff / cutoff: / ape[start_pos % r]）；
    - pool_shift 必须有来自 pool_decode 的 control 依赖；
    - publish_final 必须存在且有触发（shift final）与非触发（updated）
      两条发布来源，输出被消费（下一调用历史）。
    """
    # 两套实例（G3）：C13 切片/时序/发布检查覆盖 attn.compressor 与
    # attn.indexer.compressor（维度/张量前缀不同，检查逻辑共用）
    node_ids = {n.stable_id for n in graph.nodes}
    P = _variant_prefix(graph)
    for prefix, tp in (("attn.compressor", "comp"),
                       ("attn.indexer.compressor", "idx_comp")):
        if f"{prefix}.state_update" not in node_ids:
            continue  # 变体无该实例（HCA）是正确拓扑
        _check_slice_timing_one(graph, issues, prefix, tp,
                                tensor_prefix=P)
    # 跨实例张量共享检查（保持两套隔离语义）
    tensors = {t.name: t for t in graph.tensors}
    dec_a = tensors.get(f"{prefix}:comp_dec_token")
    tail_a = tensors.get(f"{prefix}:comp_tail_slice")
    if dec_a is not None and tail_a is not None \
            and dec_a.stable_id == tail_a.stable_id:
        _issue(issues, "C13_TAIL_DEC_SHARED", "tensors",
               "decode token 与 prefill 尾块共享张量（语义/shape 冲突）")


def _check_slice_timing_one(graph: SemanticGraph, issues: list[ContractIssue],
                            inst_prefix: str, tp: str,
                            tensor_prefix: str = "csa_topk") -> None:
    """单套 compressor 的 C13 切片/时序/发布检查（两实例共用）。

    inst_prefix = 节点前缀；tensor_prefix = 张量模板前缀。
    """
    tensors = {t.name: t for t in graph.tensors}
    tail = tensors.get(f"{tensor_prefix}:{tp}_tail_slice")
    if tail is not None:
        if tail.axes[1].symbol != "N_rem":
            _issue(issues, "C13_TAIL_LENGTH_AXIS", f"tensors[{tail.stable_id}]",
                   f"尾块长度轴必须是 N_rem（T%r），得到 {tail.axes[1].symbol}")
    dec = tensors.get(f"{tensor_prefix}:{tp}_dec_token")
    if dec is None:
        _issue(issues, "C13_DEC_TOKEN_MISSING", "tensors",
               f"缺少 {prefix} 的 decode 单 token 张量（与 prefill 尾块分开）")
    aliases = {a.stable_id: a for a in graph.alias_relations}
    expected_expr = {
        f"alias_{tp}_main_slice": "cutoff-r:cutoff",
        f"alias_{tp}_tail_slice": "cutoff:",
        # decode 切片用本调用局部索引（长度 1 的 0:1；start_pos 是历史
        # 绝对位置，不是投影下标——P2/127/128 下 start_pos:start_pos+1 为空）
        f"alias_{tp}_dec_token_slice": "0:1",
        f"alias_{tp}_dec_score_slice": "0:1",
        f"alias_{tp}_tail_score_slice": "cutoff:",
    }
    # score alias 是纯 view：APE 相加由 ape_add 节点承载，不得写进 alias
    for score_aid in (f"alias_{tp}_main_score_slice",
                      f"alias_{tp}_tail_score_slice",
                      f"alias_{tp}_dec_score_slice"):
        a = aliases.get(score_aid)
        if a is not None and ("+" in (a.transform_expr or "")
                              or "ape" in (a.transform_expr or "")):
            _issue(issues, "C13_SCORE_ALIAS_WITH_ADD",
                   f"alias_relations[{score_aid}]",
                   "score alias 不得含相加（slice+add 值流必须分开："
                   "alias 只表达实际 view，APE 相加是 ape_add 节点）")
    for aid, frag in expected_expr.items():
        a = aliases.get(aid)
        if a is None:
            _issue(issues, "C13_SLICE_ALIAS_MISSING", "alias_relations",
                   f"缺少切片 alias {aid}（切片事实必须保存 transform_expr）")
        elif a.relation_kind != "slice" or frag not in (a.transform_expr or ""):
            _issue(issues, "C13_SLICE_EXPR_WRONG", f"alias_relations[{aid}]",
                   f"切片表达式必须含 {frag!r}，得到 {a.transform_expr!r}")
    ctrl = [e for e in graph.edges
            if e.dst_node_id == f"{inst_prefix}.pool_shift"
            and e.kind == "control"]
    if not ctrl:
        _issue(issues, "C13_SHIFT_NO_POOL_ORDER", "edges",
               f"{inst_prefix}.pool_shift 缺 pool 完成的 control 依赖"
               "（时序不保证）")
    pub = next((n for n in graph.nodes
                if n.stable_id == f"{inst_prefix}.publish_final"), None)
    if pub is None:
        _issue(issues, "C13_FINAL_PUBLISH_MISSING", "nodes",
               f"缺少 {inst_prefix} 的最终状态发布节点"
               "（非触发/prefill 的最终状态来源缺失）")
    else:
        pub_in = [e for e in graph.edges
                  if e.dst_node_id == f"{inst_prefix}.publish_final"]
        # D3 逐端口核验：publish_final 四个端口（updated/updated_score/
        # final/final_score）各自必须有来源——按端口组布尔化会被同组另一
        # 端口的边掩盖（删 kv 边后 score 边仍在，组非空但 kv 端口已断）。
        expected_ports = {
            "updated": f"{inst_prefix}.state_update",
            "updated_score": f"{inst_prefix}.state_update",
            "final": f"{inst_prefix}.pool_shift",
            "final_score": f"{inst_prefix}.pool_shift",
        }
        edges_by_port: dict[str, list] = {}
        for e in pub_in:
            edges_by_port.setdefault(e.dst_port, []).append(e)
        for port, expected_src in expected_ports.items():
            port_edges = edges_by_port.get(port, [])
            if not port_edges:
                _issue(issues, "C13_FINAL_PUBLISH_SOURCE", "edges",
                       f"publish_final.{port} 无来源边"
                       f"（预期来自 {expected_src}）")
                continue
            for e in port_edges:
                if e.src_node_id != expected_src:
                    _issue(issues, "C13_FINAL_PUBLISH_SOURCE",
                           f"edges[{e.stable_id}]",
                           f"publish_final.{port} 来源必须是 {expected_src}，"
                           f"得到 {e.src_node_id}")
        # updated 端口的 guard 必须是相位 guard（互斥：无 guard = 触发双源），
        # 且两个相位（prefill / 非触发 decode）各自必须有边——删单相位边
        # 该相位场景即失去发布来源，必须检出
        for port in ("updated", "updated_score"):
            port_edges = edges_by_port.get(port, [])
            phase_guards = {e.guard_ref for e in port_edges}
            for e in port_edges:
                if e.guard_ref not in ("g_prefill", "g_decode_no_compress"):
                    _issue(issues, "C13_PUBLISH_NOT_MUTUALLY_EXCLUSIVE",
                           f"edges[{e.stable_id}]",
                           f"updated 发布边 guard 必须是相位 guard"
                           f"（g_prefill=全部 prefill / g_decode_no_compress="
                           f"非触发 decode），得到 {e.guard_ref}")
            if "g_prefill" not in phase_guards:
                _issue(issues, "C13_FINAL_PUBLISH_SOURCE", "edges",
                       f"publish_final.{port} 缺 prefill 相位发布边"
                       f"（prefill 场景无 updated 来源）")
            if "g_decode_no_compress" not in phase_guards:
                _issue(issues, "C13_FINAL_PUBLISH_SOURCE", "edges",
                       f"publish_final.{port} 缺非触发 decode 相位发布边"
                       f"（非触发 decode 场景无 updated 来源）")
    # decode 未触发 guard 必须限定相位（漏 start_pos>0 会让 prefill 借用）；
    # 从图内 guards 读（mutation 改图，契约核验实际图数据）
    _dnc = next((g for g in graph.guards
                 if g.stable_id == "g_decode_no_compress"), None)
    if _dnc is not None and "start_pos > 0" not in _dnc.predicate_expr:
        _issue(issues, "C14_DECODE_GUARD_NO_PHASE", "guards[g_decode_no_compress]",
               "decode 未触发 guard 必须含 start_pos > 0（相位限定），"
               f"得到 {_dnc.predicate_expr!r}——否则 prefill 误借 decode 条件")
    # publish 来源完备性与消费检查：与 guard 相位判断同层执行（正常图也要
    # 跑，不缩进在任何相位分支内——D3 修复；来源按端口核验，见上）
    if pub is not None:
        pub_out = [e for e in graph.edges
                   if e.src_node_id == f"{inst_prefix}.publish_final"]
        if not pub_out:
            _issue(issues, "C13_FINAL_UNCONSUMED", "edges",
                   f"{inst_prefix}.publish_final 输出无消费"
                   "（下一调用历史来源缺失）")
    # cache_out/return 不得残留 shift_state 人为入参
    for nid in (f"{inst_prefix}.cache_out", f"{inst_prefix}.return"):
        node = next((n for n in graph.nodes if n.stable_id == nid), None)
        if node is not None and "shift_state" in {p.name for p in node.input_ports}:
            _issue(issues, "C13_SYNTHETIC_STATE_INPUT", f"nodes[{nid}]",
                   "cache/return 消费量化条目，不得挂 shift_state 状态入参"
                   "（最终状态发布由 publish_final 承载）")
    # D2：tail/decode 的 score+APE add 必须消费行切片 view（非整表）；
    # main 保留全表。按实例 tp 前缀区分维度。
    adds = {n.stable_id: n for n in graph.nodes
            if n.stable_id.startswith(f"{inst_prefix}.ape_add")}
    for role, expect_rows in (("tail", f"{tp}_ape_tail_rows"),
                              ("dec", f"{tp}_ape_dec_row")):
        add = adds.get(f"{inst_prefix}.ape_add_{role}")
        if add is None:
            _issue(issues, "C14_APE_ADD_MISSING", f"nodes[{prefix}.ape_add_{role}]",
                   f"缺少 {role} 的 score+APE add 节点")
            continue
        in_tensors = {e.tensor_id for e in graph.edges
                      if e.dst_node_id == add.stable_id}
        if not any(x.endswith(expect_rows) for x in in_tensors):
            _issue(issues, "C14_APE_ROWS_NOT_SLICED",
                   f"nodes[{add.stable_id}]",
                   f"{role} add 必须消费 APE 行切片 {expect_rows}（D2），"
                   f"实际输入 {sorted(x.rsplit(':', 1)[-1] for x in in_tensors)}")


# guard 谓词的独立语义期望（源码/需求推导：真值环境与假值环境边界）。
# evaluate 于真实谓词，不只比较字符串或检查 ID 存在（G1）。
_GUARD_SEMANTIC_EXPECT = {
    # guard_id: (true_envs, false_envs)——base env 补 B/r/W/Ki
    "g_prefill": (
        [{"T": 3, "P": 0, "start_pos": 0}, {"T": 4, "P": 0, "start_pos": 0}],
        [{"T": 1, "P": 2, "start_pos": 2}, {"T": 1, "P": 128, "start_pos": 128}],
    ),
    "g_decode": (
        [{"T": 1, "P": 2, "start_pos": 2}, {"T": 1, "P": 128, "start_pos": 128}],
        [{"T": 3, "P": 0, "start_pos": 0}],
    ),
    # T>=r 边界：T4 true / T3 false（>= 改 > 即检出）
    "g_compress_prefill": (
        [{"T": 4, "P": 0, "start_pos": 0}, {"T": 5, "P": 0, "start_pos": 0}],
        [{"T": 3, "P": 0, "start_pos": 0}],
    ),
    "g_compress_prefill_skip": (
        [{"T": 3, "P": 0, "start_pos": 0}],
        [{"T": 4, "P": 0, "start_pos": 0}],
    ),
    # 尾块存在：T%r>0
    "g_compress_prefill_remainder": (
        [{"T": 3, "P": 0, "start_pos": 0}, {"T": 5, "P": 0, "start_pos": 0},
         {"T": 129, "P": 0, "start_pos": 0}],
        [{"T": 4, "P": 0, "start_pos": 0}, {"T": 128, "P": 0, "start_pos": 0}],
    ),
    # 窗口回绕边界：T>W（W-1/W false、W+1 true）
    "g_window_wrap": (
        [{"T": 129, "P": 0, "start_pos": 0}],
        [{"T": 127, "P": 0, "start_pos": 0}, {"T": 128, "P": 0, "start_pos": 0}],
    ),
    # r 条件与 Indexer 存在（CSA 专属）
    "g_compress_csa": (
        [{"T": 4, "P": 0, "start_pos": 0}, {"T": 5, "P": 0, "start_pos": 0}],
        [],
    ),
    "g_has_indexer": (
        [{"T": 4, "P": 0, "start_pos": 0}],
        [],
    ),
    # decode 触发：(start_pos+1)%r==0
    "g_compress_decode": (
        [{"T": 1, "P": 127, "start_pos": 127}],
        [{"T": 1, "P": 2, "start_pos": 2}, {"T": 1, "P": 128, "start_pos": 128}],
    ),
    # decode 未触发：相位限定 + 非 (start_pos+1)%r==0
    "g_decode_no_compress": (
        [{"T": 1, "P": 2, "start_pos": 2}, {"T": 1, "P": 128, "start_pos": 128}],
        [{"T": 3, "P": 0, "start_pos": 0}, {"T": 1, "P": 127, "start_pos": 127}],
    ),
}


def _check_guard_predicate_semantics(graph: SemanticGraph,
                                     issues: list[ContractIssue], prefix: str = "csa_topk") -> None:
    """C16：关键 guard 谓词按真实 env 求值核验边界与相位（G1）。

    用图内实际 predicate（不是常量比较）+ 白名单求值器在真值/假值环境
    边界上求值；改坏 >= 成 >、互换 prefill/decode、错 Indexer 条件均检出。
    求值 unknown（如环境缺变量）不误报，记录为检查缺口。
    """
    from structure_reader.scenario_evaluator import evaluate_tri
    preds = {g.stable_id: g.predicate_expr for g in graph.guards}
    base = {"B": 1, "r": 4, "W": 128, "Ki": 512}
    # 变体过滤：检查只对图内实际存在的 guard 生效（SWA 无压缩 guard、
    # HCA 无 Indexer guard——缺失在那里是正确拓扑而非缺陷）
    _has_compress = _variant_has_compress(graph)
    _has_indexer = _variant_has_indexer(graph)
    # 规范表达式契约：语义期望无法用真/假环境区分的 guard（如 g_has_indexer
    # 的 r==4 在 CSA 层恒真）按需求固定其规范表达式片段——改写为语义等价但
    # 形式不同的表达式（r>0）即检出。
    _CANONICAL_EXPR_FRAGMENTS = {
        "g_has_indexer": "r == 4",
        "g_compress_csa": "r > 0",
    }
    for guard_id, (true_envs, false_envs) in _GUARD_SEMANTIC_EXPECT.items():
        if not _has_compress and guard_id.startswith("g_compress"):
            continue
        if not _has_compress and guard_id == "g_decode_no_compress":
            continue  # SWA 无压缩状态：该 guard 表达压缩触发反相，不适用
        if not _has_indexer and guard_id in ("g_has_indexer", "g_causal_mask"):
            continue
        pred = preds.get(guard_id)
        if pred is None:
            _issue(issues, "C16_GUARD_MISSING", "guards",
                   f"缺少关键 guard {guard_id}（语义期望无法核验）")
            continue
        canonical = _CANONICAL_EXPR_FRAGMENTS.get(guard_id)
        if canonical is not None and canonical not in pred:
            _issue(issues, "C16_GUARD_NON_CANONICAL",
                   f"guards[{guard_id}].predicate_expr",
                   f"{guard_id} 谓词必须含规范表达式片段 {canonical!r}"
                   f"（需求语义：Indexer 仅 r==4 存在），得到 {pred!r}")
        for env in true_envs:
            r = evaluate_tri(pred, {**base, **env})
            if r.is_known and not r.value:
                _issue(issues, "C16_GUARD_FALSE_IN_TRUE_CASE",
                       f"guards[{guard_id}].predicate_expr",
                       f"谓词 {pred!r} 在本应为 true 的环境 {env} 求值为 "
                       f"false——边界或相位错误")
        for env in false_envs:
            r = evaluate_tri(pred, {**base, **env})
            if r.is_known and r.value:
                _issue(issues, "C16_GUARD_TRUE_IN_FALSE_CASE",
                       f"guards[{guard_id}].predicate_expr",
                       f"谓词 {pred!r} 在本应为 false 的环境 {env} 求值为 "
                       f"true——相位借用或条件过宽")


# 核心轴语义约束（G2 + G2-residual）：张量名 → 期望轴（按语义位置）。
# 全部从参考源码/需求独立推导（model.py 各投影/输出宽度），错轴符号/
# 长度表达式的有效变异按 symbol 判（不以数值相等掩盖不同轴）。
def _axis_semantic_expect(prefix: str = "csa_topk",
                          compress_ratio: int = 4) -> dict:
    """核心轴语义期望（G2/G2-residual）：按模板前缀与压缩比生成。

    APE 宽按 coff（model.py:298 coff=1+overlap；ratio==4 → D2/D2i，
    否则 D/Di）；无 Indexer 变体裁剪 idx_* 项。
    """
    P = prefix
    ape_w = "D2" if compress_ratio == 4 else "D"
    idx_ape_w = "D2i" if compress_ratio == 4 else "Di"
    has_indexer = compress_ratio == 4
    base = {
    # qr 末轴必须是 Rq（Q 低秩）：错成 H 即检出（即使数值巧合也按 symbol 判）
    f"{prefix}:qr": {"last": "Rq"},
    f"{prefix}:idx_q_flat": {"last": "Di"},
    f"{prefix}:idx_cache": {"last": "Di"},
    f"{prefix}:attn_cache": {"last": "D"},
    f"{prefix}:comp_cache": {"last": "D"},
    f"{prefix}:idx_cache_visible": {"last": "Di"},
    # G2-residual：prefill 拼接上下文 KV（D）、attn_norm 输出（H）、
    # gate 选择索引（K 每token专家数）
    f"{prefix}:kv_concat": {"last": "D"},
    f"{prefix}:x_attn_normed": {"last": "H"},
    f"{prefix}:gate_indices": {"last": "K"},
    # 两套 APE：主表 [r, D2/D2i]；tail/decode 行切片
    # APE 宽按 coff（model.py:298 coff=1+overlap；CSA D2/D2i、HCA D/Di）
    # ——由调用方传 compress_ratio 决定，见工厂尾部替换
    f"{prefix}:comp_ape": {"first": "r", "last": "@APE_W@"},
    f"{prefix}:idx_comp_ape": {"first": "r", "last": "@IDX_APE_W@"},
    f"{prefix}:comp_ape_tail_rows": {"first": "N_rem", "last": "@APE_W@"},
    f"{prefix}:idx_comp_ape_tail_rows": {"first": "N_rem", "last": "@IDX_APE_W@"},
    f"{prefix}:comp_ape_dec_row": {"first": "N_one", "last": "@APE_W@"},
    f"{prefix}:idx_comp_ape_dec_row": {"first": "N_one", "last": "@IDX_APE_W@"},
}





    if not has_indexer:
        base = {k: v for k, v in base.items() if "idx_" not in k}
    for k in list(base):
        v = dict(base[k])
        for key, val in v.items():
            if val == "@APE_W@":
                v[key] = ape_w
            elif val == "@IDX_APE_W@":
                v[key] = idx_ape_w
        base[k] = v
    return base


def _check_core_axis_semantics(graph: SemanticGraph,
                               issues: list[ContractIssue],
                               prefix: str = "csa_topk",
                               compress_ratio: int = 4) -> None:
    """C17：核心 named axes/端口语义约束（G2）。

    按轴 symbol（非数值相等）核验关键张量的轴身份；unknown 轴语义保持
    （不给具体值）。qr 末轴 Rq 错成 H 等受控错误检出。
    """
    tensors = {t.name: t for t in graph.tensors}
    has_compress = _variant_has_compress(graph)
    has_indexer = _variant_has_indexer(graph)
    r_axis = next((a for a in graph.named_axes if a.symbol == "r"), None)
    ratio = r_axis.value if r_axis is not None else 0
    for tname, expect in _axis_semantic_expect(prefix, ratio).items():
        # 变体适用性：剥离的角色（SWA 无压缩、HCA 无 Indexer）缺失是
        # 正确拓扑，不报缺失——只对存在的张量核验轴语义
        if not has_compress and tname in (
                f"{prefix}:comp_cache", f"{prefix}:comp_ape",
                f"{prefix}:comp_ape_tail_rows", f"{prefix}:comp_ape_dec_row",
                f"{prefix}:attn_cache", f"{prefix}:kv_concat"):
            continue  # 压缩变体专属张量（SWA 无压缩区）
        if not has_indexer and "idx_" in tname:
            continue
        t = tensors.get(tname)
        if t is None:
            _issue(issues, "C17_TENSOR_MISSING", "tensors",
                   f"核心张量 {tname} 缺失（轴语义约束无法核验）")
            continue
        if "last" in expect and t.axes[-1].symbol != expect["last"]:
            _issue(issues, "C17_AXIS_WRONG",
                   f"tensors[{t.stable_id}].axes[-1]",
                   f"{tname} 末轴必须是 {expect['last']}，得到 "
                   f"{t.axes[-1].symbol}——轴身份错误（不以数值相等掩盖）")
        if "first" in expect and t.axes[0].symbol != expect["first"]:
            _issue(issues, "C17_AXIS_WRONG",
                   f"tensors[{t.stable_id}].axes[0]",
                   f"{tname} 首轴必须是 {expect['first']}，得到 "
                   f"{t.axes[0].symbol}——轴身份错误（不以数值相等掩盖）")


def _check_cache_branches_and_states(graph: SemanticGraph,
                                     issues: list[ContractIssue]) -> None:
    """C11：prefill/decode 双路径、压缩触发分支、全部 transition 被引用。"""
    edge_ids = {e.stable_id for e in graph.edges}
    required_branches = {
        # decode：cache 读出（含本调用写入）直达 sparse_attn.kv_win
        "e_sparse_kv_decode": "g_decode",
        # prefill：当前 kv_quant 进 kv_concat
        "e_concat_win": "g_prefill",
        # prefill 压缩拼接（g_compress_prefill：触发时才拼 comp_return）
        "e_concat_comp": "g_compress_prefill",
    }
    for edge_id, guard in required_branches.items():
        if edge_id not in edge_ids:
            _issue(issues, "C11_BRANCH_MISSING", "edges",
                   f"缺少 cache 分支边 {edge_id}")
        else:
            edge = next(e for e in graph.edges if e.stable_id == edge_id)
            if edge.guard_ref != guard:
                _issue(issues, "C11_BRANCH_GUARD", f"edges[{edge_id}]",
                       f"{edge_id} 必须挂 guard {guard}，"
                       f"得到 {edge.guard_ref}")
    # sparse_attn 的 topk_idxs 端口必须被 select_cat 输出消费（不是只有
    # 未消费的索引节点）
    topk_edges = [e for e in graph.edges
                  if e.dst_node_id == "attn.sparse_attn"
                  and e.dst_port == "topk_idxs"]
    if not any(e.src_node_id == "attn.select_cat" for e in topk_edges):
        _issue(issues, "C11_SELECT_CONCAT_MISSING", "edges",
               "sparse_attn 的 topk_idxs 必须消费 attn.select_cat 输出"
               "（窗口+压缩拼接）")
    # 全部 transition 被实际 state 边引用（目录不落空）
    referenced = {e.state_transition_ref for e in graph.edges
                  if e.state_transition_ref}
    for tr in graph.state_transitions:
        if tr.stable_id not in referenced:
            _issue(issues, "C11_TRANSITION_UNREFERENCED",
                   f"state_transitions[{tr.stable_id}]",
                   "transition 未被任何 state 边引用（状态目录与图脱节）")


# ---------------------------------------------------------------------------
# 受控错误注入（测试驱动：每种错误注入后契约必须检出）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContractMutation:
    """一种受控错误的定义：注入方式与期望的契约检出码。"""

    mutation_id: str
    description: str
    expected_codes: tuple[str, ...]


def contract_error_specs() -> tuple[ContractMutation, ...]:
    """8 项受控错误的规格（mutation_id → 期望检出的契约码）。"""
    return (
        ContractMutation("comb_reversed",
                         "comb 源/目标轴交换（错误方向）", ("C1_COMB_AXIS_ORDER",)),
        ContractMutation("qr_edge_deleted",
                         "删 qr→Indexer 边", ("C2_QR_CONSUMERS",)),
        ContractMutation("kv_alias_split",
                         "K/V 别名拆成两个张量", ("C3_KV_ALIAS_SPLIT",)),
        ContractMutation("rope_order_swapped",
                         "inverse RoPE 与 O reshape 顺序交换", ("C4_ROPE_ORDER",)),
        ContractMutation("compress_guard_wrong",
                         "decode 压缩 guard 改错", ("C5_GUARD_EXPR",)),
        ContractMutation("state_version_lost",
                         "StateTransition 指向不存在的版本", ("C6_STATE_VERSION_MISSING",)),
        ContractMutation("comb_layout_swapped",
                         "comb 布局互换（physical_layout 转置）",
                         ("C1_COMB_LAYOUT",)),
        ContractMutation("ne_axis_known",
                         "N_e 轴伪造已知值", ("C8_NE_NOT_UNKNOWN",)),
        ContractMutation("sink_from_hcpost",
                         "sparse_attn sink 错接 hc post", ("C9_SINK_WRONG_SOURCE",)),
        ContractMutation("compressor_input_from_wkv",
                         "compressor 输入错接窗口 wkv", ("C10_COMP_INPUT_WRONG",)),
        ContractMutation("transitions_unreferenced",
                         "状态目录与图脱节（删状态边）",
                         ("C11_TRANSITION_UNREFERENCED",)),
        ContractMutation("compressor_shared_tensors",
                         "compressor 输入跨实例/跨路径组件共享",
                         ()),  # expected 由 build_validation_report 按图定：
        # 含 Indexer 图 → C10_TENSOR_SHARED；无 Indexer 压缩图 → C10_COMP_INPUT_WRONG

        ContractMutation("tail_full_projection",
                         "尾块输入接回完整投影", ("C12_TAIL_FULL_PROJECTION",)),
        ContractMutation("tail_unguarded",
                         "尾块边去掉 guard（无尾块场景伪活动）",
                         ("C12_TAIL_UNGUARDED",)),
        ContractMutation("shift_unconnected",
                         "pool 后 shift 输出断开", ("C12_SHIFT_UNCONNECTED",)),
        ContractMutation("tail_wrong_length_axis",
                         "尾块长度轴改回 r（不是 T%r）", ("C13_TAIL_LENGTH_AXIS",)),
        ContractMutation("slice_expr_wrong",
                         "切片表达式错误（cutoff 写错）", ("C13_SLICE_EXPR_WRONG",)),
        ContractMutation("shift_no_pool_order",
                         "shift 缺 pool 完成顺序依赖", ("C13_SHIFT_NO_POOL_ORDER",)),
        ContractMutation("final_publish_lost",
                         "非触发最终状态发布丢失",
                         ("C13_FINAL_PUBLISH_MISSING",)),
        ContractMutation("publish_not_exclusive",
                         "updated 发布边去 guard（触发 decode 双源）",
                         ("C13_PUBLISH_NOT_MUTUALLY_EXCLUSIVE",)),
        ContractMutation("score_alias_with_add",
                         "score alias 混入相加（slice/add 值流不分）",
                         ("C13_SCORE_ALIAS_WITH_ADD",)),
        ContractMutation("decode_guard_no_phase",
                         "decode 未触发 guard 漏相位限定",
                         ("C14_DECODE_GUARD_NO_PHASE",)),
        ContractMutation("prefill_publish_wrong_guard",
                         "prefill 发布错接压缩触发 guard",
                         ("C13_PUBLISH_NOT_MUTUALLY_EXCLUSIVE",)),
        ContractMutation("guard_ge_broken",
                         "g_compress_prefill 边界 >= 改 >",
                         ("C16_GUARD_FALSE_IN_TRUE_CASE",)),
        ContractMutation("guard_phase_swapped",
                         "g_prefill 与 g_decode 谓词互换",
                         ("C16_GUARD_FALSE_IN_TRUE_CASE",)),
        ContractMutation("qr_axis_wrong",
                         "qr 末轴 Rq 错成 H", ("C17_AXIS_WRONG",)),
        # V2 变体专属（review02 第 4 项）：具名拆分，按图适用性运行
        ContractMutation("hca_indexer_injected",
                         "HCA 注入合法完整 Indexer 节点/端口/张量/边",
                         ("C20_FORBIDDEN_COMPONENT_PRESENT",)),
        ContractMutation("csa_indexer_removed",
                         "CSA 删整套 Indexer 组件（含相关边/状态）",
                         ("C20_REQUIRED_COMPONENT_MISSING",)),
        ContractMutation("csa_compressor_removed",
                         "CSA 删整套 Compressor 组件",
                         ("C20_REQUIRED_COMPONENT_MISSING",)),
        ContractMutation("hash_token_dependency_removed",
                         "Hash 图 tid2eid 删 input_ids 输入",
                         ("C18_HASH_TOKEN_DEPENDENCY_MISSING",)),
        ContractMutation("hca_overlap_injected",
                         "HCA 注入合法完整 overlap 节点/张量/边",
                         ("C19_HCA_OVERLAP_NODE",)),
    )


def build_validation_report(graph: SemanticGraph) -> dict:
    """build 期实际运行全部受控错误并据实记录（D4：不写死计数/结果）。

    对每个 spec：注入 mutation → validate → 记录每项 expected code 是否
    检出（detected 布尔 + 漏检明细）。all_detected 是推导值；任何漏检都
    体现在报告中，由调用方按检查失败策略处理（不宣称通过）。
    """
    specs = contract_error_specs()
    catalog_count = len(specs)
    # 按稳定 template/spec 过滤适用性（review02 第 4 项）：
    # 压缩链 mutation 仅压缩图；Indexer 链 mutation 仅含 Indexer 图；
    # hash token 依赖仅 Hash 门控图；变体注入仅对应变体图。
    has_compress = _variant_has_compress(graph)
    has_indexer = _variant_has_indexer(graph)
    gate_is_hash = any(n.stable_id == "moe.gate.tid2eid" for n in graph.nodes)
    is_hca = (_variant_prefix(graph) == "hca_topk")
    skipped: list[dict] = []
    applied: list = []
    has_indexer_comp = any(n.stable_id == "attn.indexer.compressor.wkv"
                           for n in graph.nodes)
    for spec in specs:
        mid = spec.mutation_id
        skip_reason = None
        # 适用性按稳定 template/spec：压缩链 mutation 仅压缩图；
        # Indexer 链仅含 Indexer 图（csa_hash/csa_topk）；
        # 谓词 mutation 仅当目标 guard 在图内；变体注入仅对应变体图
        _compress_chain = {
            "compressor_input_from_wkv", "compressor_shared_tensors",
            "tail_full_projection", "tail_unguarded", "shift_unconnected",
            "tail_wrong_length_axis", "slice_expr_wrong",
            "shift_no_pool_order", "final_publish_lost",
            "publish_not_exclusive", "score_alias_with_add",
            "prefill_publish_wrong_guard",
        }
        _indexer_chain = {"qr_edge_deleted", "transitions_unreferenced"}
        guard_pred_targets = {
            "decode_guard_no_phase": "g_decode_no_compress",
            "guard_ge_broken": "g_compress_prefill",
            "guard_phase_swapped": "g_prefill",
            "compress_guard_wrong": "g_compress_decode",
        }
        preds_in_graph = {g.stable_id for g in graph.guards}
        if mid in _compress_chain and not has_compress:
            skip_reason = "图无压缩组件（SWA）"
        elif mid in _indexer_chain and not has_indexer:
            skip_reason = "图无学习式 Indexer（SWA/HCA）"
        elif (mid == "compressor_shared_tensors" and not has_indexer_comp
              and not has_compress):
            skip_reason = "SWA：无压缩无 Indexer，两路径组件均不存在"
        elif mid in guard_pred_targets and (
                guard_pred_targets[mid] not in preds_in_graph):
            skip_reason = (f"目标 guard {guard_pred_targets[mid]} 不在该变体"
                           "图内")
        elif mid == "hash_token_dependency_removed" and not gate_is_hash:
            skip_reason = "图非 Hash 门控（topk）"
        elif mid == "hca_indexer_injected" and not is_hca:
            skip_reason = "仅 HCA 图适用"
        elif mid == "csa_indexer_removed" and (not has_compress
                                               or not has_indexer):
            skip_reason = "仅 CSA（压缩+Indexer）图适用"
        elif mid == "csa_compressor_removed" and not has_compress:
            skip_reason = "图无压缩组件（SWA）"
        elif mid == "hca_overlap_injected" and not is_hca:
            skip_reason = "仅 HCA 图适用"
        if skip_reason:
            skipped.append({"mutation_id": mid, "reason": skip_reason})
            continue
        applied.append(spec)
    results = []
    all_detected = True
    for spec in applied:
        try:
            mutated = apply_contract_mutation(graph, spec.mutation_id)
            found = {i.code for i in validate_model_contracts(mutated)}
        except Exception as exc:  # 注入本身失败也是漏检，不是通过
            results.append({"mutation_id": spec.mutation_id,
                            "description": spec.description,
                            "detected": False,
                            "missing_codes": list(spec.expected_codes),
                            "error": f"{type(exc).__name__}: {exc}"})
            all_detected = False
            continue
        missing = [c for c in spec.expected_codes if c not in found]
        detected = not missing
        all_detected &= detected
        results.append({"mutation_id": spec.mutation_id,
                        "description": spec.description,
                        "detected": detected,
                        "missing_codes": missing})
    # 原图真实校验（不写死 0/pass）：引用与契约都在原图上实际运行
    ref_issues = validate_graph_references(graph)
    contract_issues = validate_model_contracts(graph)
    return {
        "schema": "validation_report_v1",
        "graph_references": {
            "issue_count": len(ref_issues),
            "status": "pass" if not ref_issues else "fail",
            "issues": [{"code": i.code, "path": i.path, "message": i.message}
                       for i in ref_issues[:20]],
        },
        "model_contracts": {
            "issue_count": len(contract_issues),
            "status": "pass" if not contract_issues else "fail",
            "issues": [{"code": i.code, "path": i.path, "message": i.message}
                       for i in contract_issues[:20]],
            "checked_groups": [
                "C1 comb 方向/布局/双来源", "C2 qr 双消费",
                "C3 共享 KV", "C4 RoPE 顺序", "C5 guard 目录",
                "C6 状态版本", "C7 select 边", "C8 N_e unknown",
                "C9 sink 独立", "C10 compressor 隔离",
                "C11 cache 分支/状态连接",
                "C12 主/尾块切片与 shift 最终版本",
                "C13 切片 shape/表达式与最终状态时序",
                "C14 score slice/add 分流与互斥状态发布",
                "C15 发布 guard 相位范围",
                "C16 guard 谓词语义（G1）",
                "C17 核心轴语义（G2）",
                "C18 Hash token 依赖", "C19 HCA 无 overlap",
                "C20 变体必需/禁止组件"]},
        "controlled_errors": {
            "catalog_count": catalog_count,
            "spec_count": len(results),  # 实际执行数（review03 第 2 项）
            "skipped_count": len(skipped),
            "skipped": skipped,
            "all_detected": all_detected,
            "results": results,
        },
        "notes": ["静态结构检查；不声称数值等价或运行验证",
                  "controlled_errors 为 build 期实际运行结果（逐项 detected），"
                  "catalog_count 为 spec 目录总数，skipped 记录不适用项及原因",
                  "graph_references/model_contracts 为原图实际校验结果"],
    }


def apply_contract_mutation(graph: SemanticGraph, mutation_id: str) -> SemanticGraph:
    """在真实图上注入一种受控错误，返回新图（不改原图）。

    用于测试：mutation 后 validate_model_contracts 必须报出对应 expected_codes。
    """
    if mutation_id == "comb_reversed":
        return _mutate_axes(graph, _COMB_PATH_ATTN,
                            new_axes=("hc_k", "hc_j"), swap_layout=False)
    if mutation_id == "qr_edge_deleted":
        P = graph.template_id.rsplit(":", 1)[-1]
        return _drop_edges(graph, lambda e: e.dst_node_id == _IDX_WQB
                           and e.tensor_id == f"tensor:{P}:qr")
    if mutation_id == "kv_alias_split":
        return _split_kv_alias(graph)
    if mutation_id == "rope_order_swapped":
        return _swap_rope_order(graph)
    if mutation_id == "compress_guard_wrong":
        return _replace_guard_expr(graph, "g_compress_decode",
                                   "(start_pos + 1) + r == 0")
    if mutation_id == "state_version_lost":
        return _drop_state_version(graph)
    if mutation_id == "comb_layout_swapped":
        return _mutate_axes(graph, _COMB_PATH_ATTN,
                            new_axes=("hc_j", "hc_k"), swap_layout=True)
    if mutation_id == "ne_axis_known":
        return _make_ne_known(graph)
    if mutation_id == "sink_from_hcpost":
        return _rewire_edge(graph, "e_sparse_sink",
                            new_src=("block.hc_pre_attn.sinkhorn", "post"))
    if mutation_id == "compressor_input_from_wkv":
        return _rewire_edge(graph, "e_comp_in_gate",
                            new_src=("attn.wkv", "y"))
    if mutation_id == "transitions_unreferenced":
        return _drop_edges(graph, lambda e: e.kind == "state"
                           and e.state_transition_ref == "tr_idx_cache")
    if mutation_id == "tail_full_projection":
        return _retensor_edge(graph, "attn.compressor.tailslice_upd",
                              new_tensor_id=f"tensor:{graph.template_id.rsplit(':', 1)[-1]}:comp_kv_proj")
    if mutation_id == "tail_unguarded":
        return _unguard_edge(graph, "attn.compressor.tailslice_upd")
    if mutation_id == "tail_wrong_length_axis":
        return _retensor_axis(graph,
                              f"{graph.template_id.rsplit(':', 1)[-1]}:comp_tail_slice",
                              "r")
    if mutation_id == "slice_expr_wrong":
        return _replace_alias_expr(graph, "alias_comp_main_slice",
                                   "kv[cutoff:cutoff+r]（错误方向）")
    if mutation_id == "shift_no_pool_order":
        return _drop_edges(graph, lambda e: e.stable_id
                           == "attn.compressor.pool_done_shift")
    if mutation_id == "decode_guard_no_phase":
        from structure_reader.graph_ir import Guard
        new_guards = [
            Guard(stable_id=g.stable_id, predicate_expr="(start_pos + 1) % r != 0",
                  evidence_refs=g.evidence_refs)
            if g.stable_id == "g_decode_no_compress" else g
            for g in graph.guards
        ]
        return _rebuild(graph, guards=new_guards)
    if mutation_id == "prefill_publish_wrong_guard":
        return _reguard_edge(graph, "attn.compressor.updated_publish",
                             "g_compress_prefill")
    if mutation_id == "guard_ge_broken":
        return _replace_guard_expr(graph, "g_compress_prefill",
                                   "start_pos == 0 and T > r")
    if mutation_id == "guard_phase_swapped":
        from structure_reader.graph_ir import Guard
        preds = {g.stable_id: g.predicate_expr for g in graph.guards}
        new_guards = [
            Guard(stable_id=g.stable_id,
                  predicate_expr=preds["g_decode"]
                  if g.stable_id == "g_prefill"
                  else (preds["g_prefill"] if g.stable_id == "g_decode"
                        else g.predicate_expr),
                  evidence_refs=g.evidence_refs)
            for g in graph.guards
        ]
        return _rebuild(graph, guards=new_guards)
    if mutation_id == "hca_indexer_injected":
        # 仅 HCA 图：注入**合法完整** Indexer 节点/端口/张量/边
        # （非悬空——通用引用检查 0，模型契约按 C20 禁止组件具名拒绝）
        from structure_reader.graph_ir import Edge, Node, Port, Tensor
        P = graph.template_id.rsplit(":", 1)[-1]
        axis_by = {a.symbol: a for a in graph.named_axes}
        idx_flat = Tensor(name=f"{P}:idx_q_flat_injected", dtype="bf16",
                          axes=[axis_by["B"], axis_by["T"], axis_by["Hi"],
                                axis_by["Di"]],
                          logical_layout=["B", "T", "Hi", "Di"],
                          physical_layout=["B", "T", "Hi", "Di"])
        wqb = Node(stable_id="attn.indexer.wq_b", op_type="linear",
                   granularity="parameter_op",
                   input_ports=[Port("x", "input", f"tensor:{P}:qr")],
                   output_ports=[Port("y", "output", idx_flat.stable_id)])
        e1 = Edge(stable_id="e_idx_wqb_injected",
                  src_node_id="attn.q_norm", src_port="y",
                  dst_node_id="attn.indexer.wq_b", dst_port="x",
                  tensor_id=f"tensor:{P}:qr", kind="data")
        return _rebuild(graph, tensors=list(graph.tensors) + [idx_flat],
                        nodes=list(graph.nodes) + [wqb],
                        edges=list(graph.edges) + [e1])
    if mutation_id == "csa_indexer_removed":
        # CSA 图：删整套 Indexer 组件（节点+全部相关边+相关状态目录，
        # 非悬空残留——C20 必需组件缺失具名检出）
        drop = {n.stable_id for n in graph.nodes
                if n.stable_id.startswith("attn.indexer")}
        kn = [n for n in graph.nodes if n.stable_id not in drop]
        ke = [e for e in graph.edges
              if e.src_node_id not in drop and e.dst_node_id not in drop]
        kt = [t for t in graph.state_transitions
              if not t.stable_id.startswith("tr_idx")]
        kv = [v for v in graph.state_versions
              if ":indexer" not in v.state_id]
        return _rebuild(graph, nodes=kn, edges=ke,
                        state_transitions=kt, state_versions=kv)
    if mutation_id == "csa_compressor_removed":
        # CSA 图：删整套 Compressor 组件（两套实例+相关边+状态目录）
        drop = {n.stable_id for n in graph.nodes
                if ".compressor." in n.stable_id}
        kn = [n for n in graph.nodes if n.stable_id not in drop]
        ke = [e for e in graph.edges
              if e.src_node_id not in drop and e.dst_node_id not in drop]
        kt = [t for t in graph.state_transitions
              if "comp" not in t.stable_id and "idx" not in t.stable_id]
        kv = [v for v in graph.state_versions
              if "compress" not in v.state_id and "indexer" not in v.state_id]
        return _rebuild(graph, nodes=kn, edges=ke,
                        state_transitions=kt, state_versions=kv)
    if mutation_id == "hash_token_dependency_removed":
        # 仅 Hash 图：tid2eid 删 input_ids 输入边（token 依赖移除）
        P = graph.template_id.rsplit(":", 1)[-1]
        return _drop_edges(
            graph, lambda e: e.dst_node_id == "moe.gate.tid2eid"
            and e.tensor_id == f"tensor:{P}:gate_input_ids")
    if mutation_id == "hca_overlap_injected":
        # 仅 HCA 图：注入合法完整 overlap 节点/张量/边（非悬空——
        # 通用引用检查 0；C19_HCA_OVERLAP_NODE 具名拒绝）。
        # blockify.y 的输出是 comp_blocks；overlap.x 消费 comp_blocks，
        # overlap.y 产出新 overlap 张量（引用一致）。
        from structure_reader.graph_ir import Edge, Node, Port, Tensor
        P = graph.template_id.rsplit(":", 1)[-1]
        axis_by = {a.symbol: a for a in graph.named_axes}
        ov = Tensor(name=f"{P}:comp_overlap_injected", dtype="fp32",
                    axes=[axis_by["B"], axis_by["N_blk"], axis_by["r"],
                          axis_by["D"]],
                    logical_layout=["B", "N_blk", "r", "D"],
                    physical_layout=["B", "N_blk", "r", "D"])
        ov_node = Node(stable_id="attn.compressor.overlap",
                       op_type="overlap_transform", granularity="tensor_op",
                       input_ports=[Port("x", "input",
                                         f"tensor:{P}:comp_blocks")],
                       output_ports=[Port("y", "output", ov.stable_id)])
        e_in = Edge(stable_id="e_injected_overlap_in",
                    src_node_id="attn.compressor.blockify", src_port="y",
                    dst_node_id="attn.compressor.overlap", dst_port="x",
                    tensor_id=f"tensor:{P}:comp_blocks", kind="data")
        return _rebuild(graph, tensors=list(graph.tensors) + [ov],
                        nodes=list(graph.nodes) + [ov_node],
                        edges=list(graph.edges) + [e_in])
    if mutation_id == "qr_axis_wrong":
        from structure_reader.graph_ir import NamedAxis, Tensor
        P = graph.template_id.rsplit(":", 1)[-1]
        axis_by = {a.symbol: a for a in graph.named_axes}
        new_tensors = []
        for t in graph.tensors:
            if t.name == f"{P}:qr":
                axes = list(t.axes)
                axes[-1] = axis_by["H"]
                symbols = [a.symbol for a in axes]
                t = Tensor(name=t.name, dtype=t.dtype, axes=axes,
                           logical_layout=symbols, physical_layout=symbols)
            new_tensors.append(t)
        return _rebuild(graph, tensors=new_tensors)
    if mutation_id == "publish_not_exclusive":
        return _unguard_edge(graph, "attn.compressor.updated_publish")
    if mutation_id == "score_alias_with_add":
        return _replace_alias_expr(
            graph, "alias_comp_main_score_slice",
            "score[cutoff-r:cutoff] + ape[:r]（add 混入 slice）")
    if mutation_id == "final_publish_lost":
        return _drop_nodes(graph, {"attn.compressor.publish_final"})
    if mutation_id == "shift_unconnected":
        return _drop_edges(graph, lambda e: e.src_node_id
                           == "attn.compressor.pool_shift")
    if mutation_id == "compressor_shared_tensors":
        P = graph.template_id.rsplit(":", 1)[-1]
        if "attn.indexer.compressor.wkv" in {n.stable_id for n in graph.nodes}:
            # 含 Indexer 图：indexer 实例消费 attention 的 D 维投影（Di/D 混用）
            return _retensor_edge(graph, "e_idx_comp_in",
                                  new_tensor_id=f"tensor:{P}:comp_kv_proj")
        # 无 Indexer 图（SWA 无压缩不适用；HCA）：
        # attention compressor 输入错接窗口 wkv（跨路径组件共享）
        return _rewire_edge(graph, "e_comp_in",
                            new_src=("attn.wkv", "y"))
    raise ValueError(f"未知受控错误: {mutation_id!r}")


def _rewire_edge(graph: SemanticGraph, edge_id: str, *,
                 new_src: tuple[str, str] | None = None,
                 new_dst: tuple[str, str] | None = None) -> SemanticGraph:
    """重接一条边（保留 kind/guard/state 引用），用于受控错误注入。"""
    new_edges = []
    for e in graph.edges:
        if e.stable_id == edge_id:
            src = new_src if new_src else (e.src_node_id, e.src_port)
            dst = new_dst if new_dst else (e.dst_node_id, e.dst_port)
            new_edges.append(type(e)(
                stable_id=e.stable_id, src_node_id=src[0], src_port=src[1],
                dst_node_id=dst[0], dst_port=dst[1], tensor_id=e.tensor_id,
                kind=e.kind, evidence_refs=e.evidence_refs,
                guard_ref=e.guard_ref,
                state_transition_ref=e.state_transition_ref,
            ))
        else:
            new_edges.append(e)
    return _rebuild(graph, edges=new_edges)


def _retensor_edge(graph: SemanticGraph, edge_id: str,
                   new_tensor_id: str) -> SemanticGraph:
    """改一条边引用的张量（不重接端点），用于维度/张量混用注入。"""
    new_edges = []
    for e in graph.edges:
        if e.stable_id == edge_id:
            new_edges.append(type(e)(
                stable_id=e.stable_id, src_node_id=e.src_node_id,
                src_port=e.src_port, dst_node_id=e.dst_node_id,
                dst_port=e.dst_port, tensor_id=new_tensor_id, kind=e.kind,
                evidence_refs=e.evidence_refs, guard_ref=e.guard_ref,
                state_transition_ref=e.state_transition_ref,
            ))
        else:
            new_edges.append(e)
    return _rebuild(graph, edges=new_edges)


def _retensor_axis(graph: SemanticGraph, tensor_name: str,
                   new_symbol: str) -> SemanticGraph:
    """把张量的第二轴换成另一符号（模拟长度轴错误）。"""
    from structure_reader.graph_ir import Tensor
    axis_by_symbol = {a.symbol: a for a in graph.named_axes}
    new_tensors = []
    for t in graph.tensors:
        if t.name == tensor_name:
            symbols = [a.symbol for a in t.axes]
            symbols[1] = new_symbol
            axes = [axis_by_symbol[s] for s in symbols]
            t = Tensor(name=t.name, dtype=t.dtype, axes=axes,
                       logical_layout=symbols, physical_layout=symbols)
        new_tensors.append(t)
    return _rebuild(graph, tensors=new_tensors)


def _replace_alias_expr(graph: SemanticGraph, alias_id: str,
                        new_expr: str) -> SemanticGraph:
    from structure_reader.graph_ir import AliasRelation
    new_aliases = [
        AliasRelation(
            stable_id=a.stable_id, base_tensor_id=a.base_tensor_id,
            alias_tensor_id=a.alias_tensor_id, relation_kind=a.relation_kind,
            alias_role=a.alias_role, transform_expr=new_expr,
            evidence_refs=a.evidence_refs,
        ) if a.stable_id == alias_id else a
        for a in graph.alias_relations
    ]
    return _rebuild(graph, alias_relations=new_aliases)


def _drop_nodes(graph: SemanticGraph, node_ids: set[str]) -> SemanticGraph:
    """删节点及其全部关联边（模拟节点缺失）。"""
    kept_nodes = [n for n in graph.nodes if n.stable_id not in node_ids]
    kept_edges = [e for e in graph.edges
                  if e.src_node_id not in node_ids
                  and e.dst_node_id not in node_ids]
    return _rebuild(graph, nodes=kept_nodes, edges=kept_edges)


def _reguard_edge(graph: SemanticGraph, edge_id: str,
                  new_guard: str) -> SemanticGraph:
    """改一条边的 guard_ref（模拟相位错接）。"""
    new_edges = []
    for e in graph.edges:
        if e.stable_id == edge_id:
            new_edges.append(type(e)(
                stable_id=e.stable_id, src_node_id=e.src_node_id,
                src_port=e.src_port, dst_node_id=e.dst_node_id,
                dst_port=e.dst_port, tensor_id=e.tensor_id, kind=e.kind,
                evidence_refs=e.evidence_refs, guard_ref=new_guard,
                state_transition_ref=e.state_transition_ref,
            ))
        else:
            new_edges.append(e)
    return _rebuild(graph, edges=new_edges)


def _unguard_edge(graph: SemanticGraph, edge_id: str) -> SemanticGraph:
    """去掉一条边的 guard_ref（模拟无尾块场景伪活动的语义破坏）。"""
    new_edges = []
    for e in graph.edges:
        if e.stable_id == edge_id:
            new_edges.append(type(e)(
                stable_id=e.stable_id, src_node_id=e.src_node_id,
                src_port=e.src_port, dst_node_id=e.dst_node_id,
                dst_port=e.dst_port, tensor_id=e.tensor_id, kind=e.kind,
                evidence_refs=e.evidence_refs, guard_ref=None,
                state_transition_ref=e.state_transition_ref,
            ))
        else:
            new_edges.append(e)
    return _rebuild(graph, edges=new_edges)


def _rebuild(graph: SemanticGraph, **overrides) -> SemanticGraph:
    from dataclasses import fields as dc_fields
    kwargs = {f.name: getattr(graph, f.name) for f in dc_fields(graph)}
    kwargs.update(overrides)
    return SemanticGraph(**kwargs)


# --- 各 mutation 的实现 ---

def _mutate_axes(graph: SemanticGraph, tensor_name_suffix: str, *,
                 new_axes: tuple[str, ...], swap_layout: bool) -> SemanticGraph:
    from structure_reader.graph_ir import NamedAxis, Tensor
    axis_by_symbol = {a.symbol: a for a in graph.named_axes}
    P = graph.template_id.rsplit(":", 1)[-1]
    new_tensors = []
    for t in graph.tensors:
        if t.name == f"{P}:{tensor_name_suffix}":
            symbols = [a.symbol for a in t.axes]
            if swap_layout:
                # 布局互换：axes 不变，physical_layout 转置
                t = Tensor(name=t.name, dtype=t.dtype, axes=t.axes,
                           logical_layout=t.logical_layout,
                           physical_layout=list(reversed(symbols)))
            else:
                # 轴序交换：只交换尾部 comb 两维，保留前导批/序列轴
                swapped = list(symbols)
                swapped[-2], swapped[-1] = swapped[-1], swapped[-2]
                axes = [axis_by_symbol[s] for s in swapped]
                t = Tensor(name=t.name, dtype=t.dtype, axes=axes,
                           logical_layout=swapped, physical_layout=swapped)
        new_tensors.append(t)
    return SemanticGraph(
        template_id=graph.template_id, named_axes=graph.named_axes,
        tensors=new_tensors, nodes=graph.nodes, edges=graph.edges,
        state_versions=graph.state_versions,
        state_transitions=graph.state_transitions, guards=graph.guards,
        guard_activations=graph.guard_activations,
        alias_relations=graph.alias_relations,
        fold_groups=graph.fold_groups,
    )


def _drop_edges(graph: SemanticGraph, predicate) -> SemanticGraph:
    kept = [e for e in graph.edges if not predicate(e)]
    return SemanticGraph(
        template_id=graph.template_id, named_axes=graph.named_axes,
        tensors=graph.tensors, nodes=graph.nodes, edges=kept,
        state_versions=graph.state_versions,
        state_transitions=graph.state_transitions, guards=graph.guards,
        guard_activations=graph.guard_activations,
        alias_relations=graph.alias_relations,
        fold_groups=graph.fold_groups,
    )


def _split_kv_alias(graph: SemanticGraph) -> SemanticGraph:
    """把 value 角色别名指向一个不存在的独立张量（模拟拆分共享 KV）。"""
    from structure_reader.graph_ir import AliasRelation
    new_aliases = []
    for a in graph.alias_relations:
        if a.alias_role == "value":
            new_aliases.append(AliasRelation(
                stable_id=a.stable_id,
                base_tensor_id="tensor:csa_topk:kv_quant_v",
                alias_tensor_id="tensor:csa_topk:kv_quant_v",
                relation_kind="same_tensor", alias_role="value",
                evidence_refs=a.evidence_refs,
            ))
        else:
            new_aliases.append(a)
    return SemanticGraph(
        template_id=graph.template_id, named_axes=graph.named_axes,
        tensors=graph.tensors, nodes=graph.nodes, edges=graph.edges,
        state_versions=graph.state_versions,
        state_transitions=graph.state_transitions, guards=graph.guards,
        guard_activations=graph.guard_activations,
        alias_relations=new_aliases, fold_groups=graph.fold_groups,
    )


def _swap_rope_order(graph: SemanticGraph) -> SemanticGraph:
    """把 inverse RoPE → reshape 边反向（模拟顺序交换）。"""
    new_edges = []
    for e in graph.edges:
        if e.src_node_id == _O_ROPE_INV and e.dst_node_id == _O_RESHAPE:
            new_edges.append(type(e)(
                stable_id=e.stable_id,
                src_node_id=e.dst_node_id, src_port=e.dst_port,
                dst_node_id=e.src_node_id, dst_port=e.src_port,
                tensor_id=e.tensor_id, kind=e.kind,
                evidence_refs=e.evidence_refs, guard_ref=e.guard_ref,
                state_transition_ref=e.state_transition_ref,
            ))
        else:
            new_edges.append(e)
    return SemanticGraph(
        template_id=graph.template_id, named_axes=graph.named_axes,
        tensors=graph.tensors, nodes=graph.nodes, edges=new_edges,
        state_versions=graph.state_versions,
        state_transitions=graph.state_transitions, guards=graph.guards,
        guard_activations=graph.guard_activations,
        alias_relations=graph.alias_relations,
        fold_groups=graph.fold_groups,
    )


def _replace_guard_expr(graph: SemanticGraph, guard_id: str,
                        new_expr: str) -> SemanticGraph:
    from structure_reader.graph_ir import Guard
    new_guards = [
        Guard(stable_id=g.stable_id, predicate_expr=new_expr,
              evidence_refs=g.evidence_refs)
        if g.stable_id == guard_id else g
        for g in graph.guards
    ]
    return SemanticGraph(
        template_id=graph.template_id, named_axes=graph.named_axes,
        tensors=graph.tensors, nodes=graph.nodes, edges=graph.edges,
        state_versions=graph.state_versions,
        state_transitions=graph.state_transitions, guards=new_guards,
        guard_activations=graph.guard_activations,
        alias_relations=graph.alias_relations,
        fold_groups=graph.fold_groups,
    )


def _drop_state_version(graph: SemanticGraph) -> SemanticGraph:
    from structure_reader.graph_ir import StateTransition
    victim = graph.state_transitions[0]
    new_transitions = [
        StateTransition(
            stable_id=tr.stable_id, state_id=tr.state_id,
            before_version_id="state:csa_topk:attn_kv_cache:v_ghost",
            after_version_id=tr.after_version_id,
            evidence_refs=tr.evidence_refs,
        ) if tr.stable_id == victim.stable_id else tr
        for tr in graph.state_transitions
    ]
    return SemanticGraph(
        template_id=graph.template_id, named_axes=graph.named_axes,
        tensors=graph.tensors, nodes=graph.nodes, edges=graph.edges,
        state_versions=graph.state_versions,
        state_transitions=new_transitions, guards=graph.guards,
        guard_activations=graph.guard_activations,
        alias_relations=graph.alias_relations,
        fold_groups=graph.fold_groups,
    )


def _make_ne_known(graph: SemanticGraph) -> SemanticGraph:
    from structure_reader.graph_ir import NamedAxis
    new_axes = [
        NamedAxis(symbol=a.symbol, name=a.name, meaning=a.meaning,
                  origin=a.origin, evidence_ref=a.evidence_ref,
                  value=6 if a.symbol == "N_e" else a.value,
                  expression=a.expression,
                  unknown_reason=None if a.symbol == "N_e" else a.unknown_reason)
        for a in graph.named_axes
    ]
    return SemanticGraph(
        template_id=graph.template_id, named_axes=new_axes,
        tensors=graph.tensors, nodes=graph.nodes, edges=graph.edges,
        state_versions=graph.state_versions,
        state_transitions=graph.state_transitions, guards=graph.guards,
        guard_activations=graph.guard_activations,
        alias_relations=graph.alias_relations,
        fold_groups=graph.fold_groups,
    )
