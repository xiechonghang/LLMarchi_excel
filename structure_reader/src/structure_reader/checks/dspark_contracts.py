"""DeepSeek-V4-Flash-0731 DSpark 集成契约检查与受控错误（工作包 B · 步骤 B4）。

独立维护的集成语义期望（不派生自提取器输出）：对
``deepseek_v4_flash_0731:dspark_integration`` 整体图核验：
- D1 闭世界边界：graph_input 恰 4（40/41/42 readout + seed）、graph_output
  恰 3（output_ids/logits/confidence 三元组）；
- D2 readout 有序：readout_concat 端口 s0/s1/s2 依序消费 mean_40/41/42
  输出（40/41/42 丢失或换序即违规）；
- D3 main_x 唯一生产与共享分发：main_x 张量唯一生产者 entry.main_norm，
  三 stage x_main_in_src 消费同一张量；
- D4 stage 三实例：节点命名空间 mtp0./mtp1./mtp2.，逐实例 (op_type,
  granularity, 端口签名) 多重集与公共模板一致（同构模板不同实例），
  win_cache state_id 逐实例唯一；
- D5 相位：decode 草稿链（草稿 Q/KV/sparse_attn/exit/markov/confidence）
  全部挂 g_decode；prefill passthrough 链挂 g_prefill——去 guard 即检出
  （prefill 误启完整 draft）；
- D6 末出口来源：exit hc_head 输入必须来自 mtp2 stage h_out；Confidence
  hidden 必须是 prenorm_x（exit.hc_head.reduce 产出，非 exit:norm_x）；
- D7 Markov 版本链：logits/output_ids 各 6 版本；第 i 步切片读第 i-1 步
  写节点输出（i=0 读 exit.head）——版本跳接即检出；output_ids v5 终版
  消费于三元组输出；
- D8 Confidence embed 来源：mk.stack_embed.stacked（当前 token 查表有序
  stack），非单步 embed、非采样后 token；
- D9 必要边 kind：14 条跨图连接边 kind=data（改 control 即检出）；
- D10 共享 embed/head 单实体：embedding_weight / head weight 各恰一节点。
- D11 Confidence float→proj 必要链（codex_b4_review01 mutant 回放）：边
  e_cf_float_proj 必须存在（src=cf.to_float.y、dst=cf.proj.x、张量
  tensor:cf:concat_f32、kind=data、g_decode）——删除（Confidence fp32
  投影断链）或降级 control 即检出；
- D12 exit hc_head 门控算子身份：exit.hc_head.gate op_type 必须
  sigmoid_gate（forward_head 头部是 sigmoid 门控加权汇聚；Sinkhorn 只在
  stage hc_pre/hc_post 侧）——算子篡改即检出；
- D13 Markov logits dtype：tensor:mk:logits 必须 fp32（markov_w2
  ParallelHead fp32，model.py:722-724）——dtype 篡改即检出；
- D14 stage 有序串接（decode + prefill 双链）：mtp{i}（i=1,2）的
  e_st_hcin_hcpre / e_st_prefill_in 必须分别来自 mtp{i-1} 对应输出节点
  与张量——跳过中间 stage 改接入口输出即检出（同轴同步篡改含张量核验）；
- D15 集成后节点 op 目录核验：关键语义算子（readout_concat、main_proj、
  main_norm、exit 门控/norm/head、Markov stack、Confidence float_cast/
  concat/proj、stage 骨架 hc_pre/hc_post/sparse_attn/moe）op_type 必须
  与目录一致——集成后组件内部算子被篡改（独立组件校验看不到装配图）
  即检出；
- D16 装配图必要链边目录（codex_b4_review02：71 条 head/Markov/
  Confidence 组件内部必要边逐条核验——组件必要边验证在独立组件图上
  本来正确，但必须应用于装配后实际图）：每条必须存在、端点/张量与
  独立维护目录逐字段一致、kind=data、guard=g_decode、两端端口张量与
  边三方一致（同轴同步重接即检出）。删除任一条（如
  e_cf_concat_float / e_mk_s2_add_sample / e_ex_mix_gate /
  e_cf_proj_squeeze）或 data→control 降级均具名拒绝；
- D17 集成不变量：出口链关键张量 dtype/轴目录（tensor:cf:concat_f32
  fp32 / cf:proj_out fp32 / cf:confidence fp32 / mk:step{i}_logits_updated
  fp32 / exit:logits fp32 / cf:concat bf16 / exit:prenorm_x bf16 /
  mk:markov_embed bf16 / mk:output_ids int64）+ stage 关键张量 dtype
  （HC 展开 flat/mixes/归一化因子 fp32、骨架 bf16、gate_indices int64）
  + 共享参数端口单实体（五步 bias.w 同 tensor:mk:w2_w、五步 embed.w 同
  tensor:mk:w1_w、exit.head.w / cf.proj.w / exit.norm.w /
  exit.hc_head.mix.w 各自锚定唯一参数张量——复制改接即检出）+
  win_cache 状态版本索引（每实例恰 v_hist/v_after 两版本）+ 迁移有向
  逐字段（win_cache before=v_hist/after=v_after；Markov logits/output_ids
  第 i 步迁移 before=v{i}/after=v{i+1}——方向反转即检出，不比无序集合）；
- D18 入口/stage 组件装配必要链目录（codex_b4_review03 阻断1 的完整
  关闭：D16 出口链之外，首入口与三 stage 公共组件的既有必要语义必须
  可靠迁移到装配后命名空间——entry 19 条绝对 ID；stage 模板 84 条内部
  边按实例 mtp0/1/2 展开为装配命名空间 + 9 条跨图改接边按改接后真实
  期望覆盖；每条核验存在、端点、张量、kind、guard、state 迁移引用、
  两端端口张量三方一致。合法分支编码在目录内：cache 写三分支 guard
  各异（inwin/wrap/decode）、prefill passthrough g_prefill、路由
  select/control kind、cache 读写 state kind + tr_dspark_cache_mtp{i}
  迁移引用——删除/降级/改接/换 guard/换迁移即具名检出，可选边不当
  必需、合法分支不误判。

- D19 可信轴/完整 shape 目录（codex_b4_review04 阻断1 的关闭：装配图
  此前只守 dtype 与出口轴 symbol，named_axes 与全部张量内嵌轴的**值**
  及关键张量完整 rank/logical/physical layout 漏检——V 全量同步
  129280→129281、entry seed [B]→[B,1]（借用已有 C1_exit 轴、引用全部
  合法）均 refs0/original 零违规、真实 CLI rc0/HTML1。目录独立维护、
  不从被检图刷新"正确期望"：配置常量轴值与 rules AXIS_DEFS 同源（其
  与 run config 的交叉核对在 load_model_values 具名执行，ModelValues
  数值链闭合），集成专属轴独立声明；两目录漂移即 D19_AXIS_TRUST_
  CONFLICT。派生轴词法锚定 canonical 表达式，值型派生轴（H_plus_Rm、
  N_draft_plus_1 等）按 canonical 公式对引用轴实际值重算交叉校验；
  场景轴 B/T/P 保持表达式、N_e 保持数据相关 unknown（unknown_reason
  非空，未知不转 0/false）。全部 264 张量逐条独立目录：dtype + 轴符号
  序（=rank）+ logical/physical layout（本模板无转置，布局=轴序）——
  固定 shape 不凭 symbol 相同放行已知维漂移；seed[B]、
  output_ids[B,N_draft+1]、Confidence 权重[1,H+Rm] 等原组件契约落到
  实际发布图逐字段核验。named_axes 符号集与张量集合闭合（多/少/漂移
  均具名拒绝）。codes：D19_AXIS_SET_MISMATCH / D19_AXIS_KIND_MISMATCH /
  D19_AXIS_VALUE_MISMATCH / D19_AXIS_EXPRESSION_MISMATCH /
  D19_AXIS_UNKNOWN_REASON_MISSING / D19_AXIS_UNKNOWN_REASON_MISMATCH /
  D19_DERIVED_AXIS_CROSSCHECK / D19_AXIS_TRUST_CONFLICT /
  D19_TENSOR_MISSING / D19_TENSOR_UNEXPECTED / D19_TENSOR_SPEC /
  D19_TENSOR_RANK_METADATA / D19_AXIS_SOURCE_MISMATCH /
  D19_TENSOR_AXIS_SOURCE。轴来源语义目录（codex_b4_review05 阻断1：
  逐轴 name/meaning/origin/evidence_ref 锁定 + N_e unknown_reason
  原文锁定，named 与张量内嵌同目录核验——数值/符号正确不等于来源
  语义正确，V 四字段借用真实 H 轴同步误写即检出）；张量可选秩标注
  global_rank/local_rank 非 None 必须等于实际轴数（两字段分别核对，
  None 合法保留，不当并行 rank，codex_b4_review05 阻断2）。
- D20 guard 谓词可信定义目录（codex_b4_review04 阻断2 的关闭：D5/D16/
  D18 只守 guard_ref ID，谓词语义坏仍发布——g_decode start_pos>0→
  >=0 后 prefill start_pos=0 误启整条草稿出口，refs0/original 零违规。
  对集成图实际四 guard（g_decode/g_prefill/g_dspark_prefill_inwin/
  g_window_wrap）来源可信核验：g_prefill/g_decode canonical 与 rules
  GUARD_DEFS 逐字一致；窗口两分支 canonical = prefill 相位合取 +
  T<=W/T>W 边界。三层核验：词法锁定 canonical 谓词原文（evidence 引用
  一并锁定）；白名单求值器对真/假边界环境实际求值（两相位误并、
  inwin/wrap 互换、阈值偏移即检出；不执行参考模型）；闭合=定义集=
  引用集=四 guard，变量缺失必须 unknown 不转 false。
  codes：D20_GUARD_MISSING / D20_GUARD_UNEXPECTED /
  D20_GUARD_PREDICATE_CANONICAL / D20_GUARD_EVIDENCE /
  D20_GUARD_TRUE_CASE_FALSE / D20_GUARD_FALSE_CASE_TRUE /
  D20_GUARD_UNKNOWN_COLLAPSED / D20_GUARD_REFERENCE_CLOSED。

受控错误（mutation → 期望检出码）：全部对真实集成图注入、引用合法
（validate_graph_references 不因注入报新引用错误），语义违规由本模块
正向检查检出；D4 口径与 build_validation_report 相同（逐项 detected，
漏检即整体失败）。目录含 codex_b4_review01 五个原始 mutant 的逐字段
等价回放（mutant_confidence_float_proj_deleted / _control /
head_gate_sinkhorn / final_logits_bf16 / middle_stage_bypass_stage0）、
codex_b4_review02 四个具名删除例的等价注入（e_cf_concat_float /
e_cf_proj_squeeze / e_ex_mix_gate / e_mk_s2_add_sample）与必要边删改 /
同轴同步篡改扩展负例、codex_b4_review03 五份实际坏图等价注入（
e_ep_cat_proj / mtp0.e_st_hcpre_a_out / mtp0.e_ds_cache_state 删除 +
mk.step2.sample 算子换 argmax + stage0 cache 迁移方向反转）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_reference import ExtractionError
from structure_reader.graph_document import SemanticGraph
from structure_reader.graph_ir import (Edge, Guard, NamedAxis, Node, Port,
                                       StateTransition, Tensor)

__all__ = [
    "validate_dspark_integration_contracts",
    "dspark_integration_error_specs",
    "build_dspark_validation_report",
    "apply_dspark_mutation",
]

_INTEGRATION_TEMPLATE = "deepseek_v4_flash_0731:dspark_integration"

# decode-only 必须挂 g_decode 的边目录（组件认可语义：去 guard = prefill
# 误启完整草稿链；按真实边 ID 全量核对，不抽样）
_DECODE_EDGES_STAGE = (
    "e_ds_xin_wqa", "e_ds_qa_qnorm", "e_ds_qnorm_wqb", "e_ds_wqb_unfl",
    "e_ds_unfl_scale", "e_ds_scale_rope", "e_ds_xin_wkvd", "e_ds_draft_wkv",
    "e_ds_draft_norm_rope", "e_ds_draft_rope_quant", "e_ds_read_cat",
    "e_ds_draft_cat", "e_ds_sparse_q", "e_ds_cat_sparse", "e_ds_idx_select",
    "e_ds_sparse_sink", "e_ds_sparse_o", "e_ds_oinv_reshape",
    "e_ds_reshape_einsum", "e_ds_einsum_flat", "e_ds_flat_wob",
)
_DECODE_EDGES_TOP = (
    "e_ex_hin_flat", "e_ex_prenorm_norm", "e_ex_norm_head",
    "e_cf_hidden_in", "e_cf_embed_in", "e_mk_seed_init", "e_mk_s0_slice_read",
)

# 14 条跨图连接边（kind 必须 = data；与 dspark_integration._rewire_table
# 的 edge_id 一致；idempotent 目录，不派生自提取器）
_CONNECTION_EDGE_IDS = (
    "mtp0.e_st_mainx_xsrc", "mtp1.e_st_mainx_xsrc", "mtp2.e_st_mainx_xsrc",
    "mtp0.e_st_hcin_hcpre", "mtp0.e_st_prefill_in",
    "mtp1.e_st_hcin_hcpre", "mtp1.e_st_prefill_in",
    "mtp2.e_st_hcin_hcpre", "mtp2.e_st_prefill_in",
    "e_ex_hin_flat", "e_mk_s0_slice_read", "e_mk_seed_init",
    "e_cf_hidden_in", "e_cf_embed_in",
)

# D15 集成后关键算子 op 目录（装配图节点身份；不派生自提取器，独立维护。
# 覆盖六组件合并后首入口/公共 stage/末出口的语义算子——组件内部算子在
# 集成装配图上被篡改时，独立组件校验看不到装配图，由本目录检出）
_STAGE_OP_ENTRIES: tuple[tuple[str, str], ...] = (
    ("stage.hc_pre_attn.mix", "linear"),
    ("stage.hc_pre_attn.sinkhorn", "sinkhorn_split"),
    ("stage.hc_pre_attn.reduce", "weighted_sum"),
    ("stage.hc_pre_attn.flatten", "flatten"),
    ("stage.attn_norm", "rms_norm"),
    ("stage.hc_post_attn.combine", "hc_combine"),
    ("stage.hc_pre_ffn.mix", "linear"),
    ("stage.hc_pre_ffn.sinkhorn", "sinkhorn_split"),
    ("stage.hc_post_ffn.combine", "hc_combine"),
    ("stage.ffn_norm", "rms_norm"),
    ("stage.prefill_passthrough", "passthrough"),
    ("dspark.sparse_attn", "sparse_attention"),
    ("stage.moe.gate.score", "moe_gate_score"),
    ("stage.moe.expert_template", "swiglu_expert"),
    ("stage.moe.shared_experts", "swiglu_expert"),
    ("dspark.cache_update", "cache_write"),
    ("dspark.cache_read", "cache_read"),
    ("dspark.wkv_w", "linear_weight"),
)

_OP_CATALOG: tuple[tuple[str, str], ...] = (
    # 首入口
    ("entry.readout_concat", "concat"),
    ("entry.readout_mean_40", "reduce_mean"),
    ("entry.readout_mean_41", "reduce_mean"),
    ("entry.readout_mean_42", "reduce_mean"),
    ("entry.main_proj", "linear"),
    ("entry.main_norm", "rms_norm"),
    ("entry.embed_lookup", "embedding"),
    ("entry.embed_w", "embedding_weight"),
    # 公共 stage 模板（三实例逐一核验：骨架双 mHC/sinkhorn 混合、norm、
    # sparse_attn、MoE、win_cache 读写）
    *(("mtp%d.%s" % (i, suffix), op)
      for i in range(3) for suffix, op in _STAGE_OP_ENTRIES),
    # 末出口 head
    ("exit.hc_head.gate", "sigmoid_gate"),
    ("exit.hc_head.reduce", "weighted_sum"),
    ("exit.norm", "rms_norm"),
    ("exit.head", "linear"),
    ("exit.head_w", "linear_weight"),
    # Markov
    ("mk.stack_embed", "stack"),
    ("mk.w1_param", "markov_w1_param"),
    ("mk.w2_param", "markov_w2_param"),
    # Markov 五步内部算子（K5 中步递推每步的算子身份必须在装配图保持——
    # 独立组件校验看不到装配图；codex_b4_review03 mutant_sample_argmax
    # 的关闭：gumbel_max_sample 换 argmax 边/张量不变即检出）
    *(("mk.step%d.%s" % (s, suffix), op)
      for s in range(5) for suffix, op in (
          ("read_col", "col_read"), ("embed", "embedding"),
          ("bias", "linear"), ("slice", "logits_slice"),
          ("add", "bias_add"), ("logits_write", "logits_write"),
          ("sample", "gumbel_max_sample"), ("write_col", "col_write"))),
    ("mk.output_ids_init", "tensor_init"),
    ("mk.output_ids_out", "graph_output"),
    ("mk.logits_out", "graph_output"),
    # Confidence
    ("cf.concat", "concat"),
    ("cf.to_float", "float_cast"),
    ("cf.proj", "linear"),
    ("cf.proj_w", "confidence_proj_weight"),
    ("cf.squeeze", "squeeze_last"),
    ("cf.confidence_out", "graph_output"),
    # 首入口 draft 链算子（e_ep 必要链端点算子身份）
    ("entry.noise_fill", "fill_const"),
    ("entry.first_col_write", "index_write"),
    ("entry.embed_lookup", "embedding"),
    ("entry.hc_expand", "unsqueeze_repeat"),
    ("entry.stage_feed_0", "stage_dispatch"),
    ("entry.stage_feed_1", "stage_dispatch"),
    ("entry.stage_feed_2", "stage_dispatch"),
)


def _prefill_chain_edges(graph: SemanticGraph) -> tuple[str, ...]:
    return tuple(
        f"mtp{i}.{eid}" for i in range(3)
        for eid in ("e_st_prefill_in",))


# ---------------------------------------------------------------------------
# D16 装配图必要链边目录（codex_b4_review02 阻断1 的完整关闭：71 条
# head/Markov/Confidence 组件内部必要边在装配图上逐条核验。独立维护、
# 不派生自提取器；每项 (edge_id, src_node, src_port, dst_node, dst_port,
# tensor_id)。kind/guard 由检查统一要求 data/g_decode；端口张量三方
# 一致由检查对图实体核对。来源 = codex_b4_review02/result.json
# edge_deletions 全清单（71 条真实删除断链核验），逐条抄录端点。）
# ---------------------------------------------------------------------------
_EXIT_CHAIN_EDGES: tuple[tuple[str, str, str, str, str, str], ...] = (
    # hc_head flatten→factor/mix、参数→mix/factor、mix→gate、参数→gate、
    # gate→reduce、flatten→reduce（forward_head sigmoid 门控加权汇聚链）
    ("e_ex_flat_factor", "exit.hc_head.flatten", "x_flat",
     "exit.hc_head.factor", "x", "tensor:exit:hc_flat"),
    ("e_ex_flat_mix", "exit.hc_head.flatten", "x_flat",
     "exit.hc_head.mix", "x", "tensor:exit:hc_flat"),
    ("e_ex_fnw_mix", "exit.hc_head_fn_w", "y",
     "exit.hc_head.mix", "w", "tensor:exit:hc_head_fn_w"),
    ("e_ex_factor_mix", "exit.hc_head.factor", "y",
     "exit.hc_head.mix", "factor", "tensor:exit:hc_factor"),
    ("e_ex_mix_gate", "exit.hc_head.mix", "y",
     "exit.hc_head.gate", "x", "tensor:exit:hc_mixes"),
    ("e_ex_scale_gate", "exit.hc_head_scale_w", "y",
     "exit.hc_head.gate", "scale", "tensor:exit:hc_head_scale_w"),
    ("e_ex_base_gate", "exit.hc_head_base_w", "y",
     "exit.hc_head.gate", "base", "tensor:exit:hc_head_base_w"),
    ("e_ex_gate_reduce", "exit.hc_head.gate", "y",
     "exit.hc_head.reduce", "pre", "tensor:exit:hc_gate"),
    ("e_ex_flat_reduce", "exit.hc_head.flatten", "x_flat",
     "exit.hc_head.reduce", "x", "tensor:exit:hc_flat"),
    ("e_ex_normw_norm", "exit.norm_w", "y",
     "exit.norm", "w", "tensor:exit:norm_w"),
    ("e_ex_headw_head", "exit.head_w", "y",
     "exit.head", "w", "tensor:exit:head_w"),
)
_MARKOV_CHAIN_EDGES: tuple[tuple[str, str, str, str, str, str], ...] = (
    # 参数→五步（共享参数单实体分发；D17 同步核验张量唯一）
    *(("e_mk_s%d_w1_embed" % s, "mk.w1_param", "y", f"mk.step{s}.embed", "w",
       "tensor:mk:w1_w") for s in range(5)),
    *(("e_mk_s%d_w2_bias" % s, "mk.w2_param", "y", f"mk.step{s}.bias", "w",
       "tensor:mk:w2_w") for s in range(5)),
    # 步内链：read_col→embed→bias、slice→add、bias→add、add→write/sample、
    # sample→write_col（K5 中步递推每条数据边）
    *(("e_mk_s%d_read_embed" % s, f"mk.step{s}.read_col", "token",
       f"mk.step{s}.embed", "x", f"tensor:mk:step{s}_col_read")
      for s in range(5)),
    *(("e_mk_s%d_embed_bias" % s, f"mk.step{s}.embed", "y",
       f"mk.step{s}.bias", "x", f"tensor:mk:step{s}_embed") for s in range(5)),
    *(("e_mk_s%d_slice_gather" % s, f"mk.step{s}.slice", "slice",
       f"mk.step{s}.add", "x", f"tensor:mk:step{s}_logits_slice")
      for s in range(5)),
    *(("e_mk_s%d_bias_add" % s, f"mk.step{s}.bias", "y",
       f"mk.step{s}.add", "bias", f"tensor:mk:step{s}_bias") for s in range(5)),
    *(("e_mk_s%d_add_write" % s, f"mk.step{s}.add", "y",
       f"mk.step{s}.logits_write", "updated",
       f"tensor:mk:step{s}_logits_updated") for s in range(5)),
    *(("e_mk_s%d_add_sample" % s, f"mk.step{s}.add", "y",
       f"mk.step{s}.sample", "logits", f"tensor:mk:step{s}_logits_updated")
      for s in range(5)),
    *(("e_mk_s%d_sample_write" % s, f"mk.step{s}.sample", "token",
       f"mk.step{s}.write_col", "token", f"tensor:mk:step{s}_written")
      for s in range(5)),
    # output_ids 版本链（write_col i → read_col i+1）与首列
    *(("e_mk_s%d_write_ids" % s, f"mk.step{s}.write_col", "y",
       f"mk.step{s + 1}.read_col", "ids", "tensor:mk:output_ids")
      for s in range(4)),
    ("e_mk_init_col0", "mk.output_ids_init", "y",
     "mk.step0.read_col", "ids", "tensor:mk:output_ids"),
    # embed stack（Confidence 来源的五步有序 stack 输入）
    *(("e_mk_s%d_embed_stack" % s, f"mk.step{s}.embed", "y",
       "mk.stack_embed", f"in{s}", f"tensor:mk:step{s}_embed")
      for s in range(5)),
    # logits 终版出口
    ("e_mk_logits_out", "mk.step4.logits_write", "y",
     "mk.logits_out", "x", "tensor:mk:logits"),
)
_CONFIDENCE_CHAIN_EDGES: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("e_cf_concat_float", "cf.concat", "y", "cf.to_float", "x",
     "tensor:cf:concat"),
    ("e_cf_w_proj", "cf.proj_w", "y", "cf.proj", "w", "tensor:cf:proj_w"),
    ("e_cf_proj_squeeze", "cf.proj", "y", "cf.squeeze", "x",
     "tensor:cf:proj_out"),
    ("e_cf_squeeze_out", "cf.squeeze", "y", "cf.confidence_out", "x",
     "tensor:cf:confidence"),
)
_NECESSARY_CHAIN_EDGES: tuple[tuple[str, str, str, str, str, str], ...] = (
    *_EXIT_CHAIN_EDGES, *_MARKOV_CHAIN_EDGES, *_CONFIDENCE_CHAIN_EDGES)


# ---------------------------------------------------------------------------
# D18 入口/stage 组件装配必要链目录（codex_b4_review03 阻断1：出口链 71 条
# 之外，首入口与三 stage 公共组件内的既有必要语义必须可靠迁移到装配后
# 命名空间。独立维护（端点/张量/相位分支抄录自已认可组件构建语义，不从
# 提取器输出生成）；kind/guard/迁移引用逐边编码——合法分支（cache 写
# 三分支各 guard、prefill passthrough g_prefill、路由 select/control、
# cache 读写 state+迁移引用）在目录内显式保持，不当可选、不误并分支。
# 每项 9 元组：(edge_id, src_node, src_port, dst_node, dst_port,
# tensor_id, kind, guard(None=必须无 guard), state_transition_ref)。
# ---------------------------------------------------------------------------

# D18a 首入口 19 条（绝对 ID，两相位无条件：guard 必须缺席、kind=data）
_ENTRY_CHAIN_EDGES: tuple[tuple[str, str, str, str, str, str, str, str | None, str | None], ...] = (
    # 主干 readout 边界（917-925）：外部 h → mean → 有序 concat
    ("e_ep_h40_mean", "entry.input_h_40", "y", "entry.readout_mean_40",
     "x", "tensor:entry:src_h_40", "data", None, None),
    ("e_ep_h41_mean", "entry.input_h_41", "y", "entry.readout_mean_41",
     "x", "tensor:entry:src_h_41", "data", None, None),
    ("e_ep_h42_mean", "entry.input_h_42", "y", "entry.readout_mean_42",
     "x", "tensor:entry:src_h_42", "data", None, None),
    ("e_ep_mean40_cat", "entry.readout_mean_40", "y",
     "entry.readout_concat", "s0", "tensor:entry:mean_h_40", "data", None,
     None),
    ("e_ep_mean41_cat", "entry.readout_mean_41", "y",
     "entry.readout_concat", "s1", "tensor:entry:mean_h_41", "data", None,
     None),
    ("e_ep_mean42_cat", "entry.readout_mean_42", "y",
     "entry.readout_concat", "s2", "tensor:entry:mean_h_42", "data", None,
     None),
    # readout → main_proj → main_norm → 唯一 main_x 三 stage
    ("e_ep_cat_proj", "entry.readout_concat", "y", "entry.main_proj",
     "x", "tensor:entry:main_hidden", "data", None, None),
    ("e_ep_projw_proj", "entry.main_proj_w", "y", "entry.main_proj",
     "w", "tensor:entry:main_proj_w", "data", None, None),
    ("e_ep_proj_norm", "entry.main_proj", "y", "entry.main_norm",
     "x", "tensor:entry:main_proj_y", "data", None, None),
    ("e_ep_normw_norm", "entry.main_norm_w", "y", "entry.main_norm",
     "w", "tensor:entry:main_norm_w", "data", None, None),
    ("e_ep_norm_feed0", "entry.main_norm", "y", "entry.stage_feed_0",
     "x", "tensor:entry:main_x", "data", None, None),
    ("e_ep_norm_feed1", "entry.main_norm", "y", "entry.stage_feed_1",
     "x", "tensor:entry:main_x", "data", None, None),
    ("e_ep_norm_feed2", "entry.main_norm", "y", "entry.stage_feed_2",
     "x", "tensor:entry:main_x", "data", None, None),
    # noise/seed → 首列写 → embed → HC 展开 → stage0
    ("e_ep_noise_firstcol", "entry.noise_fill", "y",
     "entry.first_col_write", "dst", "tensor:entry:draft_noise", "data",
     None, None),
    ("e_ep_seed_firstcol", "entry.seed_src", "y",
     "entry.first_col_write", "src", "tensor:entry:seed_ids", "data", None,
     None),
    ("e_ep_firstcol_embed", "entry.first_col_write", "y",
     "entry.embed_lookup", "x", "tensor:entry:draft_input_ids", "data",
     None, None),
    ("e_ep_embedw_embed", "entry.embed_w", "y", "entry.embed_lookup",
     "w", "tensor:entry:embed_w", "data", None, None),
    ("e_ep_embed_hcexp", "entry.embed_lookup", "y", "entry.hc_expand",
     "x", "tensor:entry:embed_y", "data", None, None),
    ("e_ep_hcexp_feed0", "entry.hc_expand", "y", "entry.stage_feed_0",
     "h", "tensor:entry:entry_h", "data", None, None),
)

# D18b stage 公共模板 84 条内部边（mtp{i} 命名空间展开；节点/张量角色名
# 为模板局部名，展开时加 mtp{i}. / tensor:mtp{i}.stage: 前缀）
_STAGE_TEMPLATE_EDGES: tuple[tuple[str, str, str, str, str, str, str, str | None, str | None], ...] = (
    # ===== 共享参数双调用点（wkv/kv_norm：main 两相位无条件、draft decode）=====
    ("e_ds_wkv_w_main", "dspark.wkv_w", "y", "dspark.wkv_main", "w",
     "wkv_w", "data", None, None),
    ("e_ds_wkv_w_draft", "dspark.wkv_w", "y", "dspark.wkv_draft", "w",
     "wkv_w", "data", "g_decode", None),
    ("e_ds_kvnorm_w_main", "dspark.kv_norm_w", "y", "dspark.kv_norm_main",
     "w", "kv_norm_w", "data", None, None),
    ("e_ds_kvnorm_w_draft", "dspark.kv_norm_w", "y", "dspark.kv_norm_draft",
     "w", "kv_norm_w", "data", "g_decode", None),
    # ===== main KV 链（两相位共享，758-761）=====
    ("e_ds_main_in_wkv", "dspark.x_main_in_src", "y", "dspark.wkv_main",
     "x", "main_x", "data", None, None),
    ("e_ds_main_wkv", "dspark.wkv_main", "y", "dspark.kv_norm_main", "x",
     "main_kv_proj", "data", None, None),
    ("e_ds_main_norm_rope", "dspark.kv_norm_main", "y", "dspark.main_rope",
     "x", "main_kv_normed", "data", None, None),
    ("e_ds_main_rope_quant", "dspark.main_rope", "y", "dspark.main_quant",
     "x", "main_kv_quant", "data", None, None),
    # ===== 窗口 cache 写三分支（合法分支 guard 各异，不并枝）+ 写槽坐标
    # + 写后读（state 边挂 tr_dspark_cache_{mtp{i}} 迁移引用）=====
    ("e_ds_cachew_inwin", "dspark.main_quant", "y", "dspark.cache_update",
     "kv", "main_kv_quant", "data", "g_dspark_prefill_inwin", None),
    ("e_ds_cachew_wrap", "dspark.main_quant", "y", "dspark.cache_update",
     "kv", "main_kv_quant", "data", "g_window_wrap", None),
    ("e_ds_cachew_dec", "dspark.main_quant", "y", "dspark.cache_update",
     "kv", "main_kv_quant", "data", "g_decode", None),
    ("e_slot_cachew", "dspark.slot_src", "y", "dspark.cache_update", "slot",
     "slot_write", "data", "g_decode", None),
    ("e_ds_cache_state", "dspark.cache_update", "cache_full",
     "dspark.cache_read", "cache_full", "win_cache", "state", "g_decode",
     "tr_dspark_cache"),
    # ===== decode 草稿 Q（774-777）=====
    ("e_ds_xin_wqa", "dspark.x_draft_single_src", "y", "dspark.wq_a", "x",
     "x_draft_single", "data", "g_decode", None),
    ("e_ds_qa_qnorm", "dspark.wq_a", "y", "dspark.q_norm", "x", "qr",
     "data", "g_decode", None),
    ("e_ds_qnorm_wqb", "dspark.q_norm", "y", "dspark.wq_b", "x", "qr",
     "data", "g_decode", None),
    ("e_ds_wqb_unfl", "dspark.wq_b", "y", "dspark.q_unflatten", "x",
     "q_flat", "data", "g_decode", None),
    ("e_ds_unfl_scale", "dspark.q_unflatten", "y", "dspark.q_rms_scale",
     "x", "q", "data", "g_decode", None),
    ("e_ds_scale_rope", "dspark.q_rms_scale", "y", "dspark.q_rope", "x",
     "q", "data", "g_decode", None),
    ("e_pos_qrope", "dspark.pos0_src", "y", "dspark.q_rope", "pos",
     "draft_pos0", "data", "g_decode", None),
    # ===== decode 草稿 KV（778-780）=====
    ("e_ds_xin_wkvd", "dspark.x_draft_single_src", "y", "dspark.wkv_draft",
     "x", "x_draft_single", "data", "g_decode", None),
    ("e_ds_draft_wkv", "dspark.wkv_draft", "y", "dspark.kv_norm_draft", "x",
     "draft_kv_proj", "data", "g_decode", None),
    ("e_ds_draft_norm_rope", "dspark.kv_norm_draft", "y", "dspark.kv_rope",
     "x", "draft_kv_normed", "data", "g_decode", None),
    ("e_pos_kvrope", "dspark.pos0_src", "y", "dspark.kv_rope", "pos",
     "draft_pos0", "data", "g_decode", None),
    ("e_ds_draft_rope_quant", "dspark.kv_rope", "y", "dspark.kv_quant", "x",
     "draft_kv_quant", "data", "g_decode", None),
    # ===== 窗口读 + 草稿 concat → sparse_attn → O 链（784-791）=====
    ("e_ds_read_cat", "dspark.cache_read", "y", "dspark.kv_cat", "win",
     "kv_win_read", "data", "g_decode", None),
    ("e_ds_draft_cat", "dspark.kv_quant", "y", "dspark.kv_cat", "draft",
     "draft_kv_quant", "data", "g_decode", None),
    ("e_ds_sparse_q", "dspark.q_rope", "y", "dspark.sparse_attn", "q",
     "q_rope", "data", "g_decode", None),
    ("e_ds_cat_sparse", "dspark.kv_cat", "y", "dspark.sparse_attn", "kv",
     "kv_concat", "data", "g_decode", None),
    ("e_ds_idx_select", "dspark.idx_build", "y", "dspark.sparse_attn",
     "topk_idxs", "topk_idxs", "select", "g_decode", None),
    ("e_ds_sparse_sink", "dspark.attn_sink", "y", "dspark.sparse_attn",
     "sink", "attn_sink_param", "data", "g_decode", None),
    ("e_ds_sparse_o", "dspark.sparse_attn", "o", "dspark.o_rope_inv", "x",
     "o_rope", "data", "g_decode", None),
    ("e_pos_oinv", "dspark.pos0_src", "y", "dspark.o_rope_inv", "pos",
     "draft_pos0", "data", "g_decode", None),
    ("e_ds_oinv_reshape", "dspark.o_rope_inv", "y", "dspark.o_reshape", "x",
     "o_rope", "data", "g_decode", None),
    ("e_ds_reshape_einsum", "dspark.o_reshape", "y", "dspark.o_einsum", "x",
     "o_grouped", "data", "g_decode", None),
    ("e_ds_einsum_flat", "dspark.o_einsum", "y", "dspark.o_flatten", "x",
     "o_lora", "data", "g_decode", None),
    ("e_ds_flat_wob", "dspark.o_flatten", "y", "dspark.wo_b", "x", "o_flat",
     "data", "g_decode", None),
    # ===== Block 骨架：hc_pre（attention 侧）=====
    ("e_st_hcpre_a_factor", "stage.hc_pre_attn.flatten", "x_flat",
     "stage.hc_pre_attn.factor", "x", "x_hc_attn_flat", "data", "g_decode",
     None),
    ("e_st_hcpre_a_flat", "stage.hc_pre_attn.flatten", "x_flat",
     "stage.hc_pre_attn.mix", "x", "x_hc_attn_flat", "data", "g_decode",
     None),
    ("e_st_hcpre_a_fn_w", "stage.hc_attn_fn_w", "y",
     "stage.hc_pre_attn.mix", "w", "hc_attn_fn_w", "data", "g_decode", None),
    ("e_st_hcpre_a_factor_in", "stage.hc_pre_attn.factor", "y",
     "stage.hc_pre_attn.mix", "factor", "attn_norm_factor", "data",
     "g_decode", None),
    ("e_st_hcpre_a_mix", "stage.hc_pre_attn.mix", "y",
     "stage.hc_pre_attn.sinkhorn", "mixes", "attn_mixes", "data",
     "g_decode", None),
    ("e_st_hcpre_a_scale_w", "stage.hc_attn_scale_w", "y",
     "stage.hc_pre_attn.sinkhorn", "scale", "hc_attn_scale_w", "data",
     "g_decode", None),
    ("e_st_hcpre_a_base_w", "stage.hc_attn_base_w", "y",
     "stage.hc_pre_attn.sinkhorn", "base", "hc_attn_base_w", "data",
     "g_decode", None),
    ("e_st_hcpre_a_pre", "stage.hc_pre_attn.sinkhorn", "pre",
     "stage.hc_pre_attn.reduce", "pre", "attn_pre", "data", "g_decode", None),
    ("e_st_hcpre_a_x", "stage.hc_pre_attn.flatten", "residual",
     "stage.hc_pre_attn.reduce", "x", "x_hc_in", "data", "g_decode", None),
    ("e_st_hcpre_a_out", "stage.hc_pre_attn.reduce", "y",
     "stage.attn_norm", "x", "x_attn_in", "data", "g_decode", None),
    ("e_st_attnnorm_w", "stage.attn_norm_w", "y", "stage.attn_norm", "w",
     "attn_norm_w", "data", "g_decode", None),
    # attn_norm 输出 = attention decode x（同一存储非拷贝，697-699）
    ("e_st_norm_xsrc", "stage.attn_norm", "y", "dspark.x_draft_single_src",
     "x", "x_draft_single", "data", "g_decode", None),
    # ===== hc_post（attention 残差至 post，700）=====
    ("e_st_attn_out_post", "dspark.wo_b", "y",
     "stage.hc_post_attn.combine", "x", "attn_out", "data", "g_decode", None),
    ("e_st_hcpost_a_post", "stage.hc_pre_attn.sinkhorn", "post",
     "stage.hc_post_attn.combine", "post", "attn_post", "data", "g_decode",
     None),
    ("e_st_hcpost_a_comb", "stage.hc_pre_attn.sinkhorn", "comb",
     "stage.hc_post_attn.combine", "comb", "attn_comb", "data", "g_decode",
     None),
    ("e_st_hcpost_a_residual", "stage.hc_pre_attn.flatten", "residual",
     "stage.hc_post_attn.combine", "residual", "x_hc_in", "data",
     "g_decode", None),
    ("e_st_hcpost_a_out", "stage.hc_post_attn.combine", "y",
     "stage.hc_pre_ffn.flatten", "x", "x_attn_post", "data", "g_decode",
     None),
    # ===== hc_pre（ffn 侧，702）=====
    ("e_st_hcpre_f_factor", "stage.hc_pre_ffn.flatten", "x_flat",
     "stage.hc_pre_ffn.factor", "x", "x_ffn_flat_hc", "data", "g_decode",
     None),
    ("e_st_hcpre_f_flat", "stage.hc_pre_ffn.flatten", "x_flat",
     "stage.hc_pre_ffn.mix", "x", "x_ffn_flat_hc", "data", "g_decode", None),
    ("e_st_hcpre_f_fn_w", "stage.hc_ffn_fn_w", "y",
     "stage.hc_pre_ffn.mix", "w", "hc_ffn_fn_w", "data", "g_decode", None),
    ("e_st_hcpre_f_factor_in", "stage.hc_pre_ffn.factor", "y",
     "stage.hc_pre_ffn.mix", "factor", "ffn_norm_factor", "data",
     "g_decode", None),
    ("e_st_hcpre_f_mix", "stage.hc_pre_ffn.mix", "y",
     "stage.hc_pre_ffn.sinkhorn", "mixes", "ffn_mixes", "data", "g_decode",
     None),
    ("e_st_hcpre_f_scale_w", "stage.hc_ffn_scale_w", "y",
     "stage.hc_pre_ffn.sinkhorn", "scale", "hc_ffn_scale_w", "data",
     "g_decode", None),
    ("e_st_hcpre_f_base_w", "stage.hc_ffn_base_w", "y",
     "stage.hc_pre_ffn.sinkhorn", "base", "hc_ffn_base_w", "data",
     "g_decode", None),
    ("e_st_hcpre_f_pre", "stage.hc_pre_ffn.sinkhorn", "pre",
     "stage.hc_pre_ffn.reduce", "pre", "ffn_pre", "data", "g_decode", None),
    ("e_st_hcpre_f_x", "stage.hc_pre_ffn.flatten", "residual",
     "stage.hc_pre_ffn.reduce", "x", "x_attn_post", "data", "g_decode", None),
    ("e_st_hcpre_f_out", "stage.hc_pre_ffn.reduce", "y", "stage.ffn_norm",
     "x", "x_ffn", "data", "g_decode", None),
    ("e_st_ffnnorm_w", "stage.ffn_norm_w", "y", "stage.ffn_norm", "w",
     "ffn_norm_w", "data", "g_decode", None),
    # ===== MoE TopK gate → 专家模板 → 汇聚（703-704；合法 kind：路由
    # select、counts 门控 control）=====
    ("e_st_ffnnorm_gate", "stage.ffn_norm", "y", "stage.moe.gate.score",
     "x", "x_ffn_normed", "data", "g_decode", None),
    ("e_st_gatew_score", "stage.gate_weight_w", "y",
     "stage.moe.gate.score", "w", "gate_weight_w", "data", "g_decode", None),
    ("e_st_score_bias", "stage.moe.gate.score", "scores",
     "stage.moe.gate.bias", "x", "gate_scores", "data", "g_decode", None),
    ("e_st_gateb_bias", "stage.gate_bias_w", "y", "stage.moe.gate.bias",
     "b", "gate_bias_w", "data", "g_decode", None),
    ("e_st_bias_topk", "stage.moe.gate.bias", "y", "stage.moe.gate.topk",
     "x", "gate_scores_biased", "data", "g_decode", None),
    ("e_st_orig_gather", "stage.moe.gate.score", "scores",
     "stage.moe.gate.gather", "original", "gate_scores", "data", "g_decode",
     None),
    ("e_st_idx_gather", "stage.moe.gate.topk", "indices",
     "stage.moe.gate.gather", "indices", "gate_indices", "data", "g_decode",
     None),
    ("e_st_idx_dispatch", "stage.moe.gate.topk", "indices",
     "stage.moe.dispatch", "indices", "gate_indices", "data", "g_decode",
     None),
    ("e_st_gate_expert", "stage.ffn_norm", "y",
     "stage.moe.expert_template", "x", "x_ffn_normed", "data", "g_decode",
     None),
    ("e_st_w_expert", "stage.moe.gate.gather", "weights",
     "stage.moe.expert_template", "weights", "gate_weights", "data",
     "g_decode", None),
    ("e_st_dispatch_select", "stage.moe.dispatch", "routing",
     "stage.moe.expert_template", "routing", "route_select", "select",
     "g_decode", None),
    ("e_st_dispatch_gate", "stage.moe.dispatch", "counts",
     "stage.moe.expert_template", "gate_counts", "n_e", "control",
     "g_decode", None),
    ("e_st_gate_shared", "stage.ffn_norm", "y",
     "stage.moe.shared_experts", "x", "x_ffn_normed", "data", "g_decode",
     None),
    ("e_st_expert_acc", "stage.moe.expert_template", "y", "stage.moe.out",
     "routed", "expert_out", "data", "g_decode", None),
    ("e_st_shared_acc", "stage.moe.shared_experts", "y", "stage.moe.out",
     "shared", "shared_out", "data", "g_decode", None),
    # ===== hc_post（ffn 残差至 post，705）=====
    ("e_st_moe_post", "stage.moe.out", "y", "stage.hc_post_ffn.combine",
     "x", "moe_out", "data", "g_decode", None),
    ("e_st_hcpost_f_post", "stage.hc_pre_ffn.sinkhorn", "post",
     "stage.hc_post_ffn.combine", "post", "ffn_post", "data", "g_decode",
     None),
    ("e_st_hcpost_f_comb", "stage.hc_pre_ffn.sinkhorn", "comb",
     "stage.hc_post_ffn.combine", "comb", "ffn_comb", "data", "g_decode",
     None),
    ("e_st_hcpost_f_residual", "stage.hc_pre_ffn.flatten", "residual",
     "stage.hc_post_ffn.combine", "residual", "x_attn_post", "data",
     "g_decode", None),
)


def _stage_chain_edges_for(i: int
                           ) -> tuple[tuple[str, str, str, str, str, str,
                                            str, str | None, str | None], ...]:
    """stage 模板 84 条内部边 → 第 i 实例装配命名空间（节点 mtp{i}.、
    张量 tensor:mtp{i}.stage:、迁移 tr_dspark_cache_mtp{i}）。"""
    n = f"mtp{i}"
    return tuple(
        (f"{n}.{eid}", f"{n}.{src_node}", src_port, f"{n}.{dst_node}",
         dst_port, f"tensor:{n}.stage:{role}", kind, guard,
         f"tr_dspark_cache_{n}" if transition == "tr_dspark_cache"
         else transition)
        for (eid, src_node, src_port, dst_node, dst_port, role, kind,
             guard, transition) in _STAGE_TEMPLATE_EDGES)


def _stage_rewired_edges_for(i: int
                             ) -> tuple[tuple[str, str, str, str, str, str,
                                              str, str | None, str | None], ...]:
    """3 条跨图改接边的装配后真实期望（_rewire_table 的边界改接语义；
    stage0 与 stage1/2 源不同——编码在目录内，不做单一模板改接）。"""
    n = f"mtp{i}"
    mainx = (f"{n}.e_st_mainx_xsrc", f"entry.stage_feed_{i}", "main_x_out",
             f"{n}.dspark.x_main_in_src", "x", "tensor:entry:main_x", "data",
             None, None)
    if i == 0:
        hcin = (f"{n}.e_st_hcin_hcpre", "entry.stage_feed_0", "h_out",
                f"{n}.stage.hc_pre_attn.flatten", "x", "tensor:entry:entry_h",
                "data", "g_decode", None)
        prefill = (f"{n}.e_st_prefill_in", "entry.stage_feed_0", "h_out",
                   f"{n}.stage.prefill_passthrough", "x",
                   "tensor:entry:entry_h", "data", "g_prefill", None)
    else:
        hcin = (f"{n}.e_st_hcin_hcpre", f"mtp{i - 1}.stage.hc_post_ffn.combine",
                "y", f"{n}.stage.hc_pre_attn.flatten", "x",
                f"tensor:mtp{i - 1}.stage:x_hc_out", "data", "g_decode", None)
        prefill = (f"{n}.e_st_prefill_in",
                   f"mtp{i - 1}.stage.prefill_passthrough", "y",
                   f"{n}.stage.prefill_passthrough", "x",
                   f"tensor:mtp{i - 1}.stage:x_hc_prefill_out", "data",
                   "g_prefill", None)
    return (mainx, hcin, prefill)


def dspark_assembly_chain_edges(
) -> tuple[tuple[str, str, str, str, str, str, str, str | None, str | None], ...]:
    """D18 完整目录：首入口 19 + stage 模板 84×3 实例 + 跨图改接 3×3。

    装配图 entry/stage 命名空间的全部必要边（删任一条即断链）；出口链
    71 条由 D16 目录、14 条跨图连接 kind 由 D9 另行核验。"""
    edges: list[tuple[str, str, str, str, str, str, str, str | None,
                      str | None]] = list(_ENTRY_CHAIN_EDGES)
    for i in range(3):
        edges.extend(_stage_chain_edges_for(i))
        edges.extend(_stage_rewired_edges_for(i))
    return tuple(edges)


# D17a 出口链关键张量 dtype/轴目录（fp32 显式升链与 bf16 激活分开——
# dtype 篡改/降精度断链即检出；不派生自提取器）
_NECESSARY_TENSOR_DTYPES: tuple[tuple[str, str], ...] = (
    ("tensor:cf:concat", "bf16"),
    ("tensor:cf:concat_f32", "fp32"),
    ("tensor:cf:proj_out", "fp32"),
    ("tensor:cf:confidence", "fp32"),
    ("tensor:cf:proj_w", "fp32"),
    ("tensor:exit:logits", "fp32"),
    ("tensor:exit:prenorm_x", "bf16"),
    ("tensor:exit:norm_x", "bf16"),
    ("tensor:exit:head_w", "fp32"),
    ("tensor:exit:norm_w", "fp32"),
    ("tensor:mk:logits", "fp32"),
    ("tensor:mk:markov_embed", "bf16"),
    ("tensor:mk:w1_w", "bf16"),
    ("tensor:mk:w2_w", "fp32"),
    ("tensor:mk:output_ids", "int64"),
    *(("tensor:mk:step%d_logits_updated" % s, "fp32") for s in range(5)),
    # stage 关键张量 dtype（每实例；HC 展开 fp32 升链 / 骨架 bf16 /
    # 路由索引整型分开——降精度/类型篡改即检出）
    *(("tensor:mtp%d.stage:%s" % (i, tid), dt)
      for i in range(3)
      for (tid, dt) in (
          ("x_hc_in", "bf16"), ("x_hc_attn_flat", "fp32"),
          ("attn_norm_factor", "fp32"), ("attn_mixes", "fp32"),
          ("attn_pre", "fp32"), ("attn_post", "fp32"),
          ("attn_comb", "fp32"), ("x_attn_in", "bf16"),
          ("x_attn_post", "bf16"), ("x_ffn_flat_hc", "fp32"),
          ("ffn_norm_factor", "fp32"), ("ffn_mixes", "fp32"),
          ("ffn_pre", "fp32"), ("ffn_post", "fp32"),
          ("ffn_comb", "fp32"), ("x_ffn", "bf16"),
          ("x_ffn_normed", "bf16"), ("gate_scores", "fp32"),
          ("gate_indices", "int64"), ("gate_weights", "fp32"),
          ("expert_out", "fp32"), ("shared_out", "fp32"),
          ("moe_out", "bf16"), ("x_hc_out", "bf16"),
          ("x_hc_prefill_out", "bf16"), ("win_cache", "bf16"),
          ("main_kv_quant", "bf16"), ("kv_win_read", "bf16"),
          ("q_rope", "bf16"), ("kv_concat", "bf16"),
          ("slot_write", "int64"), ("draft_pos0", "int64"),
          ("topk_idxs", "int32"),
          ("hc_attn_fn_w", "fp32"), ("hc_ffn_fn_w", "fp32"),
          ("attn_norm_w", "fp32"), ("ffn_norm_w", "fp32"),
          ("gate_weight_w", "bf16"), ("gate_bias_w", "fp32"))),
    # 首入口关键张量 dtype
    ("tensor:entry:main_x", "bf16"),
    ("tensor:entry:main_hidden", "bf16"),
    ("tensor:entry:main_proj_w", "fp8_e4m3fn"),
    ("tensor:entry:entry_h", "bf16"),
    ("tensor:entry:seed_ids", "int64"),
    ("tensor:entry:draft_input_ids", "int64"),
    ("tensor:entry:embed_w", "bf16"),
)

# D17b 共享参数/版本张量边集锚定：张量 → 恰这些 (生产节点, 消费节点) 边
# 多重集（生产者唯一性 + 全部消费者端口一次核清；共享参数复制、复制改接、
# 版本链绕接即检出。与 D10 互补——D10 按算子类别找实体，本目录按张量
# 完整边集定位全部生产/消费）
_SHARED_TENSOR_EDGE_SETS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("tensor:mk:w2_w",
     tuple(("mk.w2_param", f"mk.step{s}.bias") for s in range(5))),
    ("tensor:mk:w1_w",
     tuple(("mk.w1_param", f"mk.step{s}.embed") for s in range(5))),
    ("tensor:mk:output_ids",
     (("mk.output_ids_init", "mk.step0.read_col"),
      *tuple((f"mk.step{s}.write_col", f"mk.step{s + 1}.read_col")
             for s in range(4)),
      ("mk.step4.write_col", "mk.output_ids_out"))),
)

# D17c 出口三元组轴目录（轴 symbol 逐序一致；篡改输出形状即检出）
_TRIPLE_TENSOR_AXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("tensor:mk:logits", ("B", "N_draft", "V")),
    ("tensor:mk:output_ids", ("B", "N_draft_plus_1")),
    ("tensor:cf:confidence", ("B", "N_draft")),
)


# ---------------------------------------------------------------------------
# D19 可信轴/完整 shape 目录（codex_b4_review04 阻断1：装配图独立维护的
# 可信期望，不从被检图或 fresh builder 输出刷新。三部分：
# a) named_axes 逐轴定义：kind = value（固定值，必须逐字相等）/
#    expression（canonical 表达式词法锚定）/ unknown（unknown_reason
#    锚定，未知不转 0/false）。配置常量轴值与 rules AXIS_DEFS 同源——
#    AXIS_DEFS 与 run config 的交叉核对在 load_model_values 具名执行；
#    集成专属轴（hc_j/hc_k/HC3/C1*/H_plus_Rm/T_in/N_kvcat/N_sel_ds/
#    N_draft_plus_1 等）在此独立声明。派生轴（expression）另附
#    canonical 公式（用轴符号写），对图内引用轴的实际值重算交叉校验
#    （表达式串相同但引用轴值漂移时仍检出值不一致）。
# b) 全部张量逐条目录：dtype + 轴符号序（rank=轴数）+ logical_layout +
#    physical_layout。stage 张量按公共模板声明一次、三实例展开（三实例
#    同构在 D4 核验过，本目录按同一模板逐实例核验）；mk 步张量按角色
#    模板 ×5 步展开。
# c) 图级闭合：named_axes 符号集恰为目录；张量集合恰为目录展开集。
# seed[B]/output_ids[B,N_draft+1]/Confidence proj_w[C1_conf,H_plus_Rm]
# 等原组件契约经此目录守在实际发布图。
# ---------------------------------------------------------------------------

# named_axes 逐轴：symbol → (kind, expected)。
# kind="value": expected=int 固定值（与 rules AXIS_DEFS 常量同源）
# kind="expression": expected=canonical 表达式原文
# kind="unknown": expected=None（unknown_reason 非空由检查要求）
_AXIS_KIND_VALUE = "value"
_AXIS_KIND_EXPRESSION = "expression"
_AXIS_KIND_UNKNOWN = "unknown"
_TRUSTED_AXIS_DEFS: dict[str, tuple[str, object]] = {
    # 场景相关轴：表达式（值随场景变化，模板不预设数值）
    "B": (_AXIS_KIND_EXPRESSION, "B"),
    "T": (_AXIS_KIND_EXPRESSION, "T"),
    "P": (_AXIS_KIND_EXPRESSION, "P"),
    # 配置常量轴（rules AXIS_DEFS _CONFIG_AXES 同源子集：DSpark 六组件
    # 合并轴字典实际声明的主干子集；I/Mcap/N_tokens/r 属主图模板轴，
    # 不在集成模板轴字典）
    "H": (_AXIS_KIND_VALUE, 4096),
    "N_hc": (_AXIS_KIND_VALUE, 4),
    "Hq": (_AXIS_KIND_VALUE, 64),
    "D": (_AXIS_KIND_VALUE, 512),
    "D_rope": (_AXIS_KIND_VALUE, 64),
    "D_nope": (_AXIS_KIND_VALUE, 448),
    "Rq": (_AXIS_KIND_VALUE, 1024),
    "Ro": (_AXIS_KIND_VALUE, 1024),
    "G": (_AXIS_KIND_VALUE, 8),
    "W": (_AXIS_KIND_VALUE, 128),
    "E": (_AXIS_KIND_VALUE, 256),
    "K": (_AXIS_KIND_VALUE, 6),
    "V": (_AXIS_KIND_VALUE, 129280),
    "N_tgt": (_AXIS_KIND_VALUE, 3),
    "N_draft": (_AXIS_KIND_VALUE, 5),
    "Rm": (_AXIS_KIND_VALUE, 256),
    # 静态未知轴（数据相关；unknown 不当 0/false）
    "N_e": (_AXIS_KIND_UNKNOWN, None),
    # 派生轴：canonical 表达式 + 对引用轴实际值重算交叉
    "N_mix": (_AXIS_KIND_EXPRESSION, "(2 + N_hc) * N_hc"),
    "N_hcH": (_AXIS_KIND_EXPRESSION, "N_hc * H"),
    "D_g": (_AXIS_KIND_EXPRESSION, "Hq * D // G"),
    "GRo": (_AXIS_KIND_EXPRESSION, "G * Ro"),
    "N_tgtH": (_AXIS_KIND_EXPRESSION, "H * N_tgt"),
    "N_kvcat": (_AXIS_KIND_EXPRESSION, "W + N_draft"),
    "N_qf": (_AXIS_KIND_EXPRESSION, "Hq * D"),
    "N_sel_ds": (_AXIS_KIND_EXPRESSION, "min(W, P + 1) + N_draft"),
    # 集成专属固定值轴（组件推导命名空间）
    "C1": (_AXIS_KIND_VALUE, 1),
    "C1_conf": (_AXIS_KIND_VALUE, 1),
    "C1_exit": (_AXIS_KIND_VALUE, 1),
    "HC3": (_AXIS_KIND_VALUE, 3),
    "H_plus_Rm": (_AXIS_KIND_VALUE, 4352),  # = H + Rm（值型声明+交叉）
    "N_draft_plus_1": (_AXIS_KIND_VALUE, 6),  # = N_draft + 1
    "T_in": (_AXIS_KIND_VALUE, 1),
    "hc_j": (_AXIS_KIND_VALUE, 4),
    "hc_k": (_AXIS_KIND_VALUE, 4),
}

# 值型集成专属轴的派生公式交叉（D19_DERIVED_AXIS_CROSSCHECK）：对图内
# 引用轴的实际值按公式重算；与 expression 轴的求值核验互补——表达式
# 串未变但引用轴值漂移（如 H 漂移导致 H_plus_Rm 应变而未变）即检出。
# 公式用轴符号写，经白名单求值器求值（不执行任意代码）。
_TRUSTED_AXIS_DERIVED_FORMULAS: dict[str, str] = {
    "H_plus_Rm": "H + Rm",
    "N_draft_plus_1": "N_draft + 1",
    "D_nope": "D - D_rope",
    "HC3": "N_hc - 1",
    "N_mix": "(2 + N_hc) * N_hc",
    "N_hcH": "N_hc * H",
    "D_g": "Hq * D // G",
    "GRo": "G * Ro",
    "N_tgtH": "H * N_tgt",
    "N_kvcat": "W + N_draft",
    "N_qf": "Hq * D",
}

# D19 可信轴来源目录（codex_b4_review05 阻断1 的关闭：D19 此前只锁
# kind/value/expression/unknown_reason，轴的 name/meaning/origin/
# evidence_ref 来源语义可被同步误写仍出成功页——V 数值 129280 与符号
# 不变、四字段整体借用真实 H 轴（hidden_size/隐藏维度/run config dim/
# run_config.json:3）后，词表被发布页与 tensor_axes CSV 解释为隐藏维度、
# 来源错指 dim。目录逐轴锁定四字段，与 _TRUSTED_AXIS_DEFS 同一独立维护
# 原则：不从被检图或 fresh builder 输出刷新；rules AXIS_DEFS 覆盖的
# 配置/派生/未知轴（23 条）name/meaning/origin/evidence_ref 与
# AXIS_BY_SYMBOL 同源（同源冲突核验在测试与 D19_SOURCE 检查中具名），
# 集成专属轴（C1*/HC3/H_plus_Rm/N_draft_plus_1/T_in/hc_j/hc_k、
# N_kvcat/N_sel_ds/N_qf/N_tgtH 及集成侧 N_e 覆盖）独立声明。
# N_e 的 unknown_reason 单独锚定（可信原因原文）：真实"数据相关未知"
# 不得被替换成"已知 0"等误导文本。）
_TRUSTED_AXIS_SOURCES: dict[str, tuple[str, str, str, str]] = {
    # symbol → (name, meaning, origin, evidence_ref)
    "B": ("batch_size", "批大小", "场景输入", "scenarios"),
    "T": ("seq_len", "本次调用序列长度", "场景输入", "scenarios"),
    "P": ("past_len", "调用前已处理长度（历史 cache）", "场景输入",
          "scenarios"),
    "H": ("hidden_size", "隐藏维度", "run config dim", "run_config.json:3"),
    "N_hc": ("hc_streams", "mHC 超连接流数", "run config hc_mult",
             "run_config.json:33"),
    "Hq": ("query_heads", "注意力查询头数", "run config n_heads",
           "run_config.json:12"),
    "D": ("head_dim", "注意力头维度", "run config head_dim",
          "run_config.json:20"),
    "D_rope": ("rope_head_dim", "RoPE 旋转维度", "run config rope_head_dim",
               "run_config.json:21"),
    "D_nope": ("nope_head_dim", "非 RoPE 头维度（D−D_rope）",
               "推导 head_dim−rope_head_dim", "model.py:455"),
    "Rq": ("q_lora_rank", "Q 低秩秩", "run config q_lora_rank",
           "run_config.json:19"),
    "Ro": ("o_lora_rank", "分组低秩 O 投影秩", "run config o_lora_rank",
           "run_config.json:23"),
    "G": ("output_groups", "O 投影分组数", "run config o_groups",
          "run_config.json:22"),
    "W": ("window_size", "滑动窗口长度", "run config window_size",
          "run_config.json:24"),
    "E": ("routed_experts", "路由专家总数", "run config n_routed_experts",
          "run_config.json:13"),
    "K": ("experts_per_token", "每 token 激活专家数",
          "run config n_activated_experts", "run_config.json:15"),
    "V": ("vocab_size", "词表大小", "run config vocab_size",
          "run_config.json:2"),
    "N_tgt": ("dspark_target_layers", "DSpark 主干读出目标层数",
              "run config dspark_target_layer_ids", "run_config.json:10"),
    "N_draft": ("dspark_block_size", "DSpark 每步草稿 token 数",
                "run config dspark_block_size", "run_config.json:8"),
    "Rm": ("dspark_markov_rank", "Markov 低秩秩",
           "run config dspark_markov_rank", "run_config.json:11"),
    # 静态未知轴：来源四字段 + unknown_reason 可信原文单独锚定
    "N_e": ("tokens_per_expert", "路由到单个专家的 token 数",
            "数据相关（gate 路由结果）", "model.py:639-645"),
    "N_mix": ("hc_mix_dim", "mHC 混合参数维度 (2+N_hc)×N_hc",
              "推导 model.py:670", "model.py:670"),
    "N_hcH": ("hc_stream_hidden", "mHC 展平维度 N_hc×H", "推导 model.py:683",
              "model.py:683"),
    "D_g": ("group_head_dim", "O 分组前每组的紧凑头维度 Hq×D/G",
            "推导 model.py:542", "model.py:542-546"),
    "GRo": ("group_lora_flat", "O 分组低秩展平维度 G×Ro", "推导 model.py:547",
            "model.py:546-547"),
    "N_tgtH": ("main_proj_in_width",
               "main_proj 输入宽 H×N_tgt（三个 readout 有序 concat）",
               "推导 model.py:832", "ev_st_main_proj"),
    "N_kvcat": ("dspark_kv_concat_width",
                "decode KV concat 物理宽 = 窗口 cache W 槽 + 草稿 N_draft"
                "（model.py:784）", "推导 model.py:784", "ev_ds_decode"),
    "N_qf": ("q_flat_width", "wq_b 输出展平宽度 Hq×D", "推导 model.py:465",
             "ev_ds_draft_q"),
    "N_sel_ds": ("dspark_select_width",
                 "DSpark 选择索引物理宽（槽位数）：decode "
                 "min(W, P+1)+N_draft（窗口段+草稿段拼接，model.py:746）；"
                 "无 -1 填充、全部有效；仅 decode（745 assert start_pos>0），"
                 "prefill 无索引（769 早于 782）", "推导 model.py:743-747",
                 "ev_ds_topk"),
    # 集成专属固定值轴（组件推导命名空间）
    "C1": ("keepdim_singleton",
           "hc_pre 归一化因子 keepdim 单例轴（size 1，广播到 N_mix）",
           "推导 model.py:683", "model.py:683"),
    "C1_conf": ("unit_width",
                "单元宽（confidence proj 输出维 out_features=1）",
                "推导 model.py:811", "ev_cf_conf_head"),
    "C1_exit": ("unit_width",
                "单元宽（shape-(1,) 参数，如 hc_head_scale torch.empty(1)）",
                "推导 model.py:841", "model.py:841"),
    "HC3": ("hc_scale_triad",
            "hc_scale 三元组轴（pre/post/comb 三权重，torch.empty(3)）",
            "推导 model.py:677", "model.py:677"),
    "H_plus_Rm": ("confidence_input_dim",
                  "confidence proj 输入宽 H+Rm（Linear input_dim）",
                  "推导 model.py:837", "ev_cf_conf_head"),
    "N_draft_plus_1": ("dspark_block_size_plus_1",
                       "output_ids 列宽 K+1（草稿步数+种子列）",
                       "推导 model.py:864", "ev_mk_loop"),
    "T_in": ("spec_seed_width",
             "forward_spec seed input_ids 元素数（rank-1 [B] 调用前提，"
             "855 首列 [B] 槽写入）",
             "调用前提（derived，ev_ep_entry_premise）",
             "ev_ep_entry_premise"),
    "hc_j": ("hc_source_stream", "mHC comb 源流下标 j", "推导 model.py:692",
             "model.py:692"),
    "hc_k": ("hc_target_stream", "mHC comb 目标流下标 k", "推导 model.py:692",
             "model.py:692"),
}

# N_e 可信 unknown_reason 原文（unknown 轴的原因语义锁定：真实"数据
# 相关未知"不得替换为"已知 0"等误导文本；value/expression kind 漂移
# 由 D19_AXIS_KIND_MISMATCH 另行检出）
_TRUSTED_AXIS_UNKNOWN_REASONS: dict[str, str] = {
    "N_e": "依赖每 token 的路由选择，静态分析未知",
}

# stage 公共张量模板（65 条/实例）：role → (dtype, axes 符号序)。
# logical/physical layout 在本模板全部 = 轴符号序（无转置声明），目录
# 声明一次，三实例 mtp{i}.stage: 前缀展开核验。
_STAGE_TENSOR_TEMPLATE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("attn_comb", "fp32", ("B", "N_draft", "hc_j", "hc_k")),
    ("attn_mixes", "fp32", ("B", "N_draft", "N_mix")),
    ("attn_norm_factor", "fp32", ("B", "N_draft", "C1")),
    ("attn_norm_w", "fp32", ("H",)),
    ("attn_out", "bf16", ("B", "N_draft", "H")),
    ("attn_post", "fp32", ("B", "N_draft", "N_hc")),
    ("attn_pre", "fp32", ("B", "N_draft", "N_hc")),
    ("attn_sink_param", "fp32", ("Hq",)),
    ("draft_kv_normed", "bf16", ("B", "N_draft", "D")),
    ("draft_kv_proj", "bf16", ("B", "N_draft", "D")),
    ("draft_kv_quant", "bf16", ("B", "N_draft", "D")),
    ("draft_pos0", "int64", ()),
    ("expert_out", "fp32", ("B", "N_draft", "H")),
    ("ffn_comb", "fp32", ("B", "N_draft", "hc_j", "hc_k")),
    ("ffn_mixes", "fp32", ("B", "N_draft", "N_mix")),
    ("ffn_norm_factor", "fp32", ("B", "N_draft", "C1")),
    ("ffn_norm_w", "fp32", ("H",)),
    ("ffn_post", "fp32", ("B", "N_draft", "N_hc")),
    ("ffn_pre", "fp32", ("B", "N_draft", "N_hc")),
    ("gate_bias_w", "fp32", ("E",)),
    ("gate_indices", "int64", ("B", "N_draft", "K")),
    ("gate_scores", "fp32", ("B", "N_draft", "E")),
    ("gate_scores_biased", "fp32", ("B", "N_draft", "E")),
    ("gate_weight_w", "bf16", ("E", "H")),
    ("gate_weights", "fp32", ("B", "N_draft", "K")),
    ("hc_attn_base_w", "fp32", ("N_mix",)),
    ("hc_attn_fn_w", "fp32", ("N_mix", "N_hcH")),
    ("hc_attn_scale_w", "fp32", ("HC3",)),
    ("hc_ffn_base_w", "fp32", ("N_mix",)),
    ("hc_ffn_fn_w", "fp32", ("N_mix", "N_hcH")),
    ("hc_ffn_scale_w", "fp32", ("HC3",)),
    ("kv_concat", "bf16", ("B", "N_kvcat", "D")),
    ("kv_norm_w", "fp32", ("D",)),
    ("kv_win_read", "bf16", ("B", "W", "D")),
    ("main_kv_normed", "bf16", ("B", "T", "D")),
    ("main_kv_proj", "bf16", ("B", "T", "D")),
    ("main_kv_quant", "bf16", ("B", "T", "D")),
    ("main_x", "bf16", ("B", "T", "H")),
    ("moe_out", "bf16", ("B", "N_draft", "H")),
    ("n_e", "int64", ("E",)),
    ("o_flat", "bf16", ("B", "N_draft", "GRo")),
    ("o_grouped", "bf16", ("B", "N_draft", "G", "D_g")),
    ("o_lora", "bf16", ("B", "N_draft", "G", "Ro")),
    ("o_rope", "bf16", ("B", "N_draft", "Hq", "D")),
    ("q", "bf16", ("B", "N_draft", "Hq", "D")),
    ("q_flat", "bf16", ("B", "N_draft", "N_qf")),
    ("q_rope", "bf16", ("B", "N_draft", "Hq", "D")),
    ("qr", "bf16", ("B", "N_draft", "Rq")),
    ("route_select", "int64", ("N_e",)),
    ("route_slot", "fp32", ("N_e",)),
    ("shared_out", "fp32", ("B", "N_draft", "H")),
    ("slot_write", "int64", ()),
    ("topk_idxs", "int32", ("B", "N_draft", "N_sel_ds")),
    ("win_cache", "bf16", ("B", "W", "D")),
    ("wkv_w", "fp8_e4m3fn", ("D", "H")),
    ("x_attn_in", "bf16", ("B", "N_draft", "H")),
    ("x_attn_post", "bf16", ("B", "N_draft", "N_hc", "H")),
    ("x_draft_single", "bf16", ("B", "N_draft", "H")),
    ("x_ffn", "bf16", ("B", "N_draft", "H")),
    ("x_ffn_flat_hc", "fp32", ("B", "N_draft", "N_hcH")),
    ("x_ffn_normed", "bf16", ("B", "N_draft", "H")),
    ("x_hc_attn_flat", "fp32", ("B", "N_draft", "N_hcH")),
    ("x_hc_in", "bf16", ("B", "N_draft", "N_hc", "H")),
    ("x_hc_out", "bf16", ("B", "N_draft", "N_hc", "H")),
    ("x_hc_prefill_out", "bf16", ("B", "N_draft", "N_hc", "H")),
)

# mk 五步张量角色模板：role → (dtype, axes)。每步 s 前缀展开。
_MK_STEP_TENSOR_TEMPLATE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("bias", "fp32", ("B", "V")),
    ("col_read", "int64", ("B",)),
    ("embed", "bf16", ("B", "Rm")),
    ("logits_slice", "fp32", ("B", "V")),
    ("logits_updated", "fp32", ("B", "V")),
    ("written", "int64", ("B",)),
)

# 顶层命名空间张量目录：完整 stable_id → (dtype, axes)。
_TOP_TENSOR_TEMPLATE: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # entry
    ("tensor:entry:draft_input_ids", "int64", ("B", "N_draft")),
    ("tensor:entry:draft_noise", "int64", ("B", "N_draft")),
    ("tensor:entry:embed_w", "bf16", ("V", "H")),
    ("tensor:entry:embed_y", "bf16", ("B", "N_draft", "H")),
    ("tensor:entry:entry_h", "bf16", ("B", "N_draft", "N_hc", "H")),
    ("tensor:entry:main_hidden", "bf16", ("B", "T", "N_tgtH")),
    ("tensor:entry:main_norm_w", "fp32", ("H",)),
    ("tensor:entry:main_proj_w", "fp8_e4m3fn", ("H", "N_tgtH")),
    ("tensor:entry:main_proj_y", "bf16", ("B", "T", "H")),
    ("tensor:entry:main_x", "bf16", ("B", "T", "H")),
    ("tensor:entry:mean_h_40", "bf16", ("B", "T", "H")),
    ("tensor:entry:mean_h_41", "bf16", ("B", "T", "H")),
    ("tensor:entry:mean_h_42", "bf16", ("B", "T", "H")),
    ("tensor:entry:seed_ids", "int64", ("B",)),
    ("tensor:entry:src_h_40", "bf16", ("B", "T", "N_hc", "H")),
    ("tensor:entry:src_h_41", "bf16", ("B", "T", "N_hc", "H")),
    ("tensor:entry:src_h_42", "bf16", ("B", "T", "N_hc", "H")),
    # exit
    ("tensor:exit:hc_factor", "fp32", ("B", "N_draft", "C1_exit")),
    ("tensor:exit:hc_flat", "fp32", ("B", "N_draft", "N_hcH")),
    ("tensor:exit:hc_gate", "fp32", ("B", "N_draft", "N_hc")),
    ("tensor:exit:hc_head_base_w", "fp32", ("N_hc",)),
    ("tensor:exit:hc_head_fn_w", "fp32", ("N_hc", "N_hcH")),
    ("tensor:exit:hc_head_scale_w", "fp32", ("C1_exit",)),
    ("tensor:exit:hc_mixes", "fp32", ("B", "N_draft", "N_hc")),
    ("tensor:exit:head_w", "fp32", ("V", "H")),
    ("tensor:exit:logits", "fp32", ("B", "N_draft", "V")),
    ("tensor:exit:norm_w", "fp32", ("H",)),
    ("tensor:exit:norm_x", "bf16", ("B", "N_draft", "H")),
    ("tensor:exit:prenorm_x", "bf16", ("B", "N_draft", "H")),
    # confidence
    ("tensor:cf:concat", "bf16", ("B", "N_draft", "H_plus_Rm")),
    ("tensor:cf:concat_f32", "fp32", ("B", "N_draft", "H_plus_Rm")),
    ("tensor:cf:confidence", "fp32", ("B", "N_draft")),
    ("tensor:cf:proj_out", "fp32", ("B", "N_draft", "C1_conf")),
    ("tensor:cf:proj_w", "fp32", ("C1_conf", "H_plus_Rm")),
    # markov 顶层
    ("tensor:mk:logits", "fp32", ("B", "N_draft", "V")),
    ("tensor:mk:markov_embed", "bf16", ("B", "N_draft", "Rm")),
    ("tensor:mk:output_ids", "int64", ("B", "N_draft_plus_1")),
    ("tensor:mk:w1_w", "bf16", ("V", "Rm")),
    ("tensor:mk:w2_w", "fp32", ("V", "Rm")),
)


def dspark_trusted_tensor_catalog() -> dict[str, tuple[str, tuple[str, ...]]]:
    """D19 完整张量目录：三 stage 实例 + mk 五步 + 顶层命名空间展开。

    返回 stable_id → (dtype, axes 符号序)；logical/physical layout 期望
    = axes 符号序（本模板声明无转置，由 D19_TENSOR_SPEC 逐字段核验）。
    """
    catalog: dict[str, tuple[str, tuple[str, ...]]] = {}
    for i in range(3):
        for role, dt, axes in _STAGE_TENSOR_TEMPLATE:
            catalog[f"tensor:mtp{i}.stage:{role}"] = (dt, axes)
    for s in range(5):
        for role, dt, axes in _MK_STEP_TENSOR_TEMPLATE:
            catalog[f"tensor:mk:step{s}_{role}"] = (dt, axes)
    for tid, dt, axes in _TOP_TENSOR_TEMPLATE:
        catalog[tid] = (dt, axes)
    return catalog


# ---------------------------------------------------------------------------
# D20 guard 谓词可信定义目录（codex_b4_review04 阻断2：集成图实际四
# guard 的来源可信期望，不从被检图刷新。g_prefill/g_decode canonical
# 谓词与 rules GUARD_DEFS 同源（GUARD_DEFS 与 model.py 行号证据绑定，
# 变更时轴字典校验路径会具名失败）；窗口两分支 canonical = prefill
# 相位合取 + T<=W/T>W 边界（g_window_wrap 的 start_pos == 0 前缀 =
# DSpark prefill 专属分支，model.py:523-528）。evidence 引用一并锁定。
# ---------------------------------------------------------------------------

_TRUSTED_GUARD_DEFS: dict[str, tuple[str, tuple[str, ...]]] = {
    "g_decode": ("start_pos > 0", ("ev_ds_decode", "ev_ex_spec_return")),
    "g_prefill": ("start_pos == 0", ("ev_ds_prefill",)),
    "g_dspark_prefill_inwin": ("start_pos == 0 and T <= W",
                               ("ev_ds_prefill",)),
    "g_window_wrap": ("start_pos == 0 and T > W", ("ev_ds_prefill",)),
}

# guard 语义边界期望：guard_id → (true_envs, false_envs)。base 环境
# {B,T,P,start_pos,W} 由检查构造；边界成对覆盖：相位互斥（prefill/
# decode 误并）、窗口 inwin/wrap 互换、阈值 ±1（T==W 属 inwin 非 wrap、
# start_pos==0 属 prefill 非 decode）。求值 unknown（变量缺失）不误报，
# 由 D20_GUARD_UNKNOWN_COLLAPSED 另行核验 unknown 不被压成 false。
_TRUSTED_GUARD_SEMANTICS: dict[str, tuple[tuple[dict, ...],
                                          tuple[dict, ...]]] = {
    "g_prefill": (
        ({"T": 3, "P": 0}, {"T": 128, "P": 0}, {"T": 129, "P": 0}),
        ({"T": 1, "P": 2}, {"T": 1, "P": 128}, {"T": 1, "P": 1}),
    ),
    "g_decode": (
        ({"T": 1, "P": 2}, {"T": 1, "P": 128}, {"T": 1, "P": 1}),
        ({"T": 3, "P": 0}, {"T": 128, "P": 0}, {"T": 129, "P": 0}),
    ),
    "g_dspark_prefill_inwin": (
        ({"T": 3, "P": 0}, {"T": 128, "P": 0}),
        ({"T": 129, "P": 0}, {"T": 1, "P": 2}, {"T": 1, "P": 128}),
    ),
    "g_window_wrap": (
        ({"T": 129, "P": 0},),
        ({"T": 128, "P": 0}, {"T": 127, "P": 0},
         {"T": 1, "P": 2}, {"T": 1, "P": 128}),
    ),
}


@dataclass(frozen=True)
class DsparkContractIssue:
    """一条集成契约发现：稳定 code、精确 path、易懂 message。"""

    code: str
    path: str
    message: str


def _issue(issues: list[DsparkContractIssue], code: str, path: str,
           message: str) -> None:
    issues.append(DsparkContractIssue(code=code, path=path, message=message))


def _edge_by_id(graph: SemanticGraph) -> dict[str, Edge]:
    return {e.stable_id: e for e in graph.edges}


def _node_by_id(graph: SemanticGraph) -> dict[str, Node]:
    return {n.stable_id: n for n in graph.nodes}


def _edges_into(graph: SemanticGraph, node_id: str, port: str) -> list[Edge]:
    return [e for e in graph.edges
            if e.dst_node_id == node_id and e.dst_port == port]


def _edges_out(graph: SemanticGraph, node_id: str, port: str) -> list[Edge]:
    return [e for e in graph.edges
            if e.src_node_id == node_id and e.src_port == port]


def _tensor_producers(graph: SemanticGraph, tensor_id: str) -> list[Edge]:
    return [e for e in graph.edges if e.tensor_id == tensor_id
            and e.kind in ("data", "select")]


def _required_decode_edges(graph: SemanticGraph) -> list[str]:
    edges = [f"mtp{i}.{eid}" for i in range(3) for eid in _DECODE_EDGES_STAGE]
    edges += list(_DECODE_EDGES_TOP)
    return edges


def validate_dspark_integration_contracts(
        graph: SemanticGraph) -> tuple[DsparkContractIssue, ...]:
    """独立集成契约：全部通过返回空元组；不执行模型代码。"""
    issues: list[DsparkContractIssue] = []
    if graph.template_id != _INTEGRATION_TEMPLATE:
        return issues  # 非集成图不适用本契约（调用方按模板过滤）
    nodes = _node_by_id(graph)
    edges = _edge_by_id(graph)
    graph_tensors = {t.stable_id: t for t in graph.tensors}

    # D1 闭世界边界
    gi = sorted(n.stable_id for n in graph.nodes if n.op_type == "graph_input")
    go = sorted(n.stable_id for n in graph.nodes if n.op_type == "graph_output")
    if gi != ["entry.input_h_40", "entry.input_h_41", "entry.input_h_42",
              "entry.seed_src"]:
        _issue(issues, "D1_INPUT_CLOSED_WORLD", "graph_input",
               f"整体图外输入必须恰为 40/41/42 readout + seed，得到 {gi}")
    if go != ["cf.confidence_out", "mk.logits_out", "mk.output_ids_out"]:
        _issue(issues, "D1_OUTPUT_TRIPLE", "graph_output",
               f"整体图输出必须恰为 output_ids/logits/confidence 三元组，"
               f"得到 {go}")

    # D2 readout 有序（40/41/42 → s0/s1/s2）
    concat_in = {}
    for port, src_node, want_tensor in (
            ("s0", "entry.readout_mean_40", "tensor:entry:mean_h_40"),
            ("s1", "entry.readout_mean_41", "tensor:entry:mean_h_41"),
            ("s2", "entry.readout_mean_42", "tensor:entry:mean_h_42")):
        got = _edges_into(graph, "entry.readout_concat", port)
        if len(got) != 1 or got[0].src_node_id != src_node \
                or got[0].tensor_id != want_tensor:
            _issue(issues, "D2_READOUT_ORDER",
                   f"entry.readout_concat.{port}",
                   f"readout 端口 {port} 必须依序消费 {src_node} 输出 "
                   f"（{want_tensor}），得到 "
                   f"{[(e.src_node_id, e.tensor_id) for e in got]}")

    # D3 main_x 唯一生产 + 三 stage 共享消费
    # （生产 = entry.main_norm.y 三条分发边；stage_feed*.main_x_out 是
    # 同一张量的转发端口，不计为第二生产者）
    mx_edges = _tensor_producers(graph, "tensor:entry:main_x")
    fwd = {e.stable_id for e in mx_edges if e.src_port == "main_x_out"}
    produced = [e for e in mx_edges if e.stable_id not in fwd]
    if [e.src_node_id for e in produced] != ["entry.main_norm"] * 3 or \
            {e.dst_node_id for e in produced} != {
                "entry.stage_feed_0", "entry.stage_feed_1",
                "entry.stage_feed_2"}:
        _issue(issues, "D3_MAIN_X_SINGLE_PRODUCER", "tensor:entry:main_x",
               f"main_x 必须由 entry.main_norm 唯一生产并分发三 stage，"
               f"得到 {[(e.src_node_id, e.dst_node_id) for e in produced]}")
    if len(fwd) != 3:
        _issue(issues, "D3_MAIN_X_DISPATCH", "tensor:entry:main_x",
               f"main_x 必须经三 stage dispatch 转发（main_x_out），"
               f"得到 {len(fwd)} 条转发边")

    # D4 stage 三实例：命名空间 + 模板同构 + 状态唯一
    for i, prefix in enumerate(("mtp0.", "mtp1.", "mtp2.")):
        inst_nodes = [n for n in graph.nodes if n.stable_id.startswith(prefix)]
        if not inst_nodes:
            _issue(issues, "D4_STAGE_INSTANCE_MISSING", prefix,
                   f"stage 实例 {i} 无节点")
            continue
    sigs = []
    for i, prefix in enumerate(("mtp0.", "mtp1.", "mtp2.")):
        sig = sorted(
            (n.op_type, n.granularity,
             tuple((p.direction, p.name) for p in n.input_ports),
             tuple((p.direction, p.name) for p in n.output_ports))
            for n in graph.nodes if n.stable_id.startswith(prefix)
            and not n.stable_id.startswith(f"mtp{i}.dspark.x_"))
        sigs.append(sig)
    if not (sigs[0] == sigs[1] == sigs[2]):
        _issue(issues, "D4_STAGE_TEMPLATE_HOMOGENEOUS", "mtp{0,1,2}",
               "三实例的 (op_type, granularity, 端口签名) 多重集必须一致"
               "（公共模板实例化），实际不一致")
    states = sorted({t.state_id for t in graph.state_transitions
                     if t.state_id.startswith("dspark:")})
    if states != ["dspark:win_cache:mtp.0", "dspark:win_cache:mtp.1",
                  "dspark:win_cache:mtp.2"]:
        _issue(issues, "D4_STAGE_STATE_UNIQUE", "state_transitions",
               f"win_cache state_id 必须逐实例唯一 mtp.0/1/2，得到 {states}")

    # D5 相位 guard：decode 草稿链全部 g_decode
    for eid in _required_decode_edges(graph):
        e = edges.get(eid)
        if e is None:
            _issue(issues, "D5_DECODE_EDGE_MISSING", f"edges[{eid}]",
                   "decode 草稿链必要边缺失")
            continue
        if e.guard_ref != "g_decode":
            _issue(issues, "D5_DECODE_GUARD_REQUIRED", f"edges[{eid}]",
                   f"decode 专用边必须挂 g_decode，得到 guard_ref="
                   f"{e.guard_ref!r}（去 guard = prefill 误启完整草稿链）")
    for eid in _prefill_chain_edges(graph):
        e = edges.get(eid)
        if e is not None and e.guard_ref != "g_prefill":
            _issue(issues, "D5_PREFILL_GUARD_REQUIRED", f"edges[{eid}]",
                   f"prefill passthrough 链必须挂 g_prefill，得到 "
                   f"{e.guard_ref!r}")

    # D6 末出口来源（mtp2 h_out；Confidence hidden = prenorm_x）
    e = edges.get("e_ex_hin_flat")
    if e is None or e.src_node_id != "mtp2.stage.hc_post_ffn.combine" \
            or e.tensor_id != "tensor:mtp2.stage:x_hc_out":
        _issue(issues, "D6_EXIT_SOURCE", "edges[e_ex_hin_flat]",
               "exit hc_head 输入必须来自 mtp2 stage h_out"
               "（mtp2.stage.hc_post_ffn.combine.y），换 stage 即违规")
    e = edges.get("e_cf_hidden_in")
    if e is None or e.src_node_id != "exit.hc_head.reduce" \
            or e.tensor_id != "tensor:exit:prenorm_x":
        _issue(issues, "D6_CONFIDENCE_HIDDEN_PRENORM", "edges[e_cf_hidden_in]",
               "Confidence hidden 必须是 prenorm_x（exit.hc_head.reduce.y，"
               "norm 之前），接 exit:norm_x 即违规")

    # D7 Markov 版本链
    lv = sorted(v.stable_id for v in graph.state_versions
                if v.state_id == "mk:logits")
    ov = sorted(v.stable_id for v in graph.state_versions
                if v.state_id == "mk:output_ids")
    if lv != [f"state:mk:logits:v{i}" for i in range(6)]:
        _issue(issues, "D7_LOGITS_VERSION_CHAIN", "state_versions",
               f"mk:logits 必须有 v0..v5 六版本，得到 {lv}")
    if ov != [f"state:mk:output_ids:v{i}" for i in range(6)]:
        _issue(issues, "D7_OUTPUT_IDS_VERSION_CHAIN", "state_versions",
               f"mk:output_ids 必须有 v0..v5 六版本，得到 {ov}")
    prev_write = "exit.head"
    for i in range(5):
        e = edges.get(f"e_mk_s{i}_slice_read")
        if e is None or e.src_node_id != prev_write:
            _issue(issues, "D7_STEP_VERSION_LINK", f"edges[e_mk_s{i}_slice_read]",
                   f"第 {i} 步 logits 切片必须读第 {i - 1} 步写节点"
                   f"（{prev_write}）输出（i=0 读 exit.head），版本跳接即违规")
        prev_write = f"mk.step{i}.logits_write"
    final_ids = _edges_into(graph, "mk.output_ids_out", "ids")
    if len(final_ids) != 1 or final_ids[0].src_node_id != "mk.step4.write_col":
        _issue(issues, "D7_FINAL_IDS_CONSUME_V5", "mk.output_ids_out.ids",
               "output_ids 三元组输出必须消费第 4 步写后的终版（v5）")

    # D8 Confidence embed 来源（stack_embed，非单步/非采样后）
    e = edges.get("e_cf_embed_in")
    if e is None or e.src_node_id != "mk.stack_embed" \
            or e.src_port != "stacked" or e.tensor_id != "tensor:mk:markov_embed":
        _issue(issues, "D8_CONFIDENCE_EMBED_STACK", "edges[e_cf_embed_in]",
               "Confidence markov_embed 必须来自 mk.stack_embed.stacked"
               "（五步当前 token 查表有序 stack），单步 embed 或采样后"
               "token 查表即违规")

    # D9 必要边 kind=data
    for eid in _CONNECTION_EDGE_IDS:
        e = edges.get(eid)
        if e is None:
            _issue(issues, "D9_CONNECTION_EDGE_MISSING", f"edges[{eid}]",
                   "跨图连接必要边缺失")
        elif e.kind != "data":
            _issue(issues, "D9_CONNECTION_EDGE_KIND", f"edges[{eid}]",
                   f"跨图连接必要边必须 kind=data（必要数据流不得降为"
                   f" control），得到 {e.kind!r}")

    # D10 共享 embed/head 单实体
    embed_nodes = [n.stable_id for n in graph.nodes
                   if n.op_type == "embedding_weight"]
    if embed_nodes != ["entry.embed_w"]:
        _issue(issues, "D10_SHARED_EMBED_SINGLE", "embedding_weight",
               f"共享 embed 权重必须恰一节点 entry.embed_w（三实例 mtp.{i}"
               f".embed 同一对象非复制），得到 {embed_nodes}")
    head_w = [n.stable_id for n in graph.nodes
              if n.op_type == "linear_weight"
              and any(p.tensor_id == "tensor:exit:head_w"
                      for p in n.output_ports)]
    if head_w != ["exit.head_w"]:
        _issue(issues, "D10_SHARED_HEAD_SINGLE", "exit.head_w",
               f"共享 head 权重必须恰一节点 exit.head_w（mtp.2.head 同一"
               f"对象非复制），得到 {head_w}")

    # D11 Confidence float→proj 必要链（float→proj 断链/降级即检出）
    e = edges.get("e_cf_float_proj")
    if e is None:
        _issue(issues, "D11_CF_FLOAT_PROJ_MISSING", "edges[e_cf_float_proj]",
               "Confidence fp32 投影必要边 e_cf_float_proj（cf.to_float.y→"
               "cf.proj.x，tensor:cf:concat_f32）缺失——float→proj 断链")
    else:
        if (e.src_node_id, e.src_port) != ("cf.to_float", "y") or \
                (e.dst_node_id, e.dst_port) != ("cf.proj", "x"):
            _issue(issues, "D11_CF_FLOAT_PROJ_ENDPOINTS",
                   "edges[e_cf_float_proj]",
                   f"e_cf_float_proj 必须连 cf.to_float.y→cf.proj.x，得到 "
                   f"{e.src_node_id}.{e.src_port}→{e.dst_node_id}."
                   f"{e.dst_port}")
        if e.tensor_id != "tensor:cf:concat_f32":
            _issue(issues, "D11_CF_FLOAT_PROJ_TENSOR",
                   "edges[e_cf_float_proj]",
                   f"e_cf_float_proj 必须承载 tensor:cf:concat_f32（fp32 "
                   f"concat 输出），得到 {e.tensor_id!r}")
        if e.kind != "data":
            _issue(issues, "D11_CF_FLOAT_PROJ_KIND", "edges[e_cf_float_proj]",
                   f"e_cf_float_proj 必须 kind=data（必要数据流），得到 "
                   f"{e.kind!r}")
    # 端口三方一致：cf.proj.x 端口张量必须与边一致（同轴同步篡改）
    proj = nodes.get("cf.proj")
    if proj is not None:
        xin = [p for p in proj.input_ports if p.name == "x"]
        if len(xin) != 1 or xin[0].tensor_id != "tensor:cf:concat_f32":
            _issue(issues, "D11_CF_PROJ_PORT_TENSOR", "nodes[cf.proj].x",
                   "cf.proj 输入端口 x 必须消费 tensor:cf:concat_f32（float "
                   "投影链同轴），改接其它张量即违规")

    # D12 exit hc_head 门控算子身份（集成后节点 op 核验）
    gate = nodes.get("exit.hc_head.gate")
    if gate is None:
        _issue(issues, "D12_HC_HEAD_GATE_MISSING", "exit.hc_head.gate",
               "exit hc_head 门控节点缺失")
    elif gate.op_type != "sigmoid_gate":
        _issue(issues, "D12_HC_HEAD_GATE_OP", "exit.hc_head.gate",
               f"hc_head 门控必须 op=sigmoid_gate（forward_head sigmoid 加权"
               f"汇聚；Sinkhorn 只在 stage hc_pre/hc_post 侧），得到 "
               f"{gate.op_type!r}")

    # D13 Markov logits dtype（fp32，ParallelHead fp32）
    t = graph_tensors.get("tensor:mk:logits")
    if t is None:
        _issue(issues, "D13_MK_LOGITS_TENSOR_MISSING",
               "tensors[tensor:mk:logits]", "Markov logits 张量缺失")
    elif t.dtype != "fp32":
        _issue(issues, "D13_MK_LOGITS_DTYPE", "tensors[tensor:mk:logits]",
               f"tensor:mk:logits 必须 fp32（markov_w2 ParallelHead fp32），"
               f"得到 {t.dtype!r}")

    # D14 stage 有序串接（decode + prefill 双链；i=1,2 逐条核验来源）
    for i in (1, 2):
        e = edges.get(f"mtp{i}.e_st_hcin_hcpre")
        want_src = f"mtp{i - 1}.stage.hc_post_ffn.combine"
        want_t = f"tensor:mtp{i - 1}.stage:x_hc_out"
        if e is None or e.src_node_id != want_src or e.tensor_id != want_t:
            got = (f"{e.src_node_id}→{e.dst_node_id} t={e.tensor_id}"
                   if e is not None else "边缺失")
            _issue(issues, "D14_STAGE_CHAIN_DECODE",
                   f"edges[mtp{i}.e_st_hcin_hcpre]",
                   f"stage{i} decode 输入必须来自 stage{i - 1} 真实 h 输出"
                   f"（{want_src}.y，{want_t}），跳过中间 stage 改接其它源"
                   f"即违规；得到 {got}")
        e = edges.get(f"mtp{i}.e_st_prefill_in")
        want_src = f"mtp{i - 1}.stage.prefill_passthrough"
        want_t = f"tensor:mtp{i - 1}.stage:x_hc_prefill_out"
        if e is None or e.src_node_id != want_src or e.tensor_id != want_t:
            got = (f"{e.src_node_id}→{e.dst_node_id} t={e.tensor_id}"
                   if e is not None else "边缺失")
            _issue(issues, "D14_STAGE_CHAIN_PREFILL",
                   f"edges[mtp{i}.e_st_prefill_in]",
                   f"stage{i} prefill 输入必须来自 stage{i - 1} passthrough"
                   f"（{want_src}.y，{want_t}）；得到 {got}")

    # D15 集成后关键算子 op 目录（装配图上组件内部算子篡改即检出）
    for nid, want_op in _OP_CATALOG:
        n = nodes.get(nid)
        if n is None:
            _issue(issues, "D15_OP_NODE_MISSING", f"nodes[{nid}]",
                   f"关键语义节点 {nid} 缺失（op 目录核验不通过）")
        elif n.op_type != want_op:
            _issue(issues, "D15_OP_MISMATCH", f"nodes[{nid}]",
                   f"关键算子 {nid} 必须 op={want_op}，得到 {n.op_type!r}")
    # readout concat 三端口 op 面身份（40/41/42 有序消费已由 D2 核验）

    # D16 装配图必要链边目录（codex_b4_review02 阻断1：组件必要边验证
    # 必须应用于装配后实际图——删边/降级/改接/同轴漂移逐条具名检出）
    for eid, src, sport, dst, dport, tid in _NECESSARY_CHAIN_EDGES:
        e = edges.get(eid)
        if e is None:
            _issue(issues, "D16_CHAIN_EDGE_MISSING", f"edges[{eid}]",
                   f"必要链边 {eid}（{src}.{sport}→{dst}.{dport}，{tid}）"
                   f"缺失——装配图断链")
            continue
        if (e.src_node_id, e.src_port) != (src, sport) or \
                (e.dst_node_id, e.dst_port) != (dst, dport):
            _issue(issues, "D16_CHAIN_EDGE_ENDPOINTS", f"edges[{eid}]",
                   f"必要链边 {eid} 必须连 {src}.{sport}→{dst}.{dport}"
                   f"（{tid}），得到 {e.src_node_id}.{e.src_port}→"
                   f"{e.dst_node_id}.{e.dst_port}")
        if e.tensor_id != tid:
            _issue(issues, "D16_CHAIN_EDGE_TENSOR", f"edges[{eid}]",
                   f"必要链边 {eid} 必须承载 {tid}，得到 {e.tensor_id!r}")
        if e.kind != "data":
            _issue(issues, "D16_CHAIN_EDGE_KIND", f"edges[{eid}]",
                   f"必要链边 {eid} 必须 kind=data（必要数据流不得降级），"
                   f"得到 {e.kind!r}")
        if e.guard_ref != "g_decode":
            _issue(issues, "D16_CHAIN_EDGE_GUARD", f"edges[{eid}]",
                   f"必要链边 {eid} 属 decode 出口链，必须挂 g_decode，"
                   f"得到 guard_ref={e.guard_ref!r}")
        # 两端端口张量三方一致（同轴同步重接在引用合法下语义违规）
        for role, nid, port_name, direction in (
                ("src", e.src_node_id, e.src_port, "output"),
                ("dst", e.dst_node_id, e.dst_port, "input")):
            n = nodes.get(nid)
            if n is None:
                continue  # 节点缺失由通用引用检查具名报告
            p = next((p for p in (n.output_ports if direction == "output"
                                  else n.input_ports)
                      if p.name == port_name), None)
            if p is None or p.tensor_id != e.tensor_id:
                _issue(issues, "D16_CHAIN_EDGE_PORT_TENSOR",
                       f"nodes[{nid}].{port_name}",
                       f"必要链边 {eid} 的 {role} 端口 {nid}.{port_name} "
                       f"张量必须与边一致（{e.tensor_id}），得到 "
                       f"{p.tensor_id!r}" if p else f"端口缺失")

    # D17 集成不变量：dtype/轴目录 + 共享张量边集 + win_cache 版本索引
    for tid, want in _NECESSARY_TENSOR_DTYPES:
        t = graph_tensors.get(tid)
        if t is None:
            _issue(issues, "D17_TENSOR_MISSING", f"tensors[{tid}]",
                   f"出口链关键张量 {tid} 缺失")
        elif t.dtype != want:
            _issue(issues, "D17_TENSOR_DTYPE", f"tensors[{tid}]",
                   f"{tid} 必须 {want}，得到 {t.dtype!r}")
    for tid, want_axes in _TRIPLE_TENSOR_AXES:
        t = graph_tensors.get(tid)
        if t is not None:
            got = tuple(a.symbol for a in t.axes)
            if got != want_axes:
                _issue(issues, "D17_TRIPLE_AXES", f"tensors[{tid}]",
                       f"出口三元组张量 {tid} 轴必须 {want_axes}，得到 {got}")
    for tid, want_dsts in _SHARED_TENSOR_EDGE_SETS:
        carrying = sorted((e.src_node_id, e.dst_node_id) for e in graph.edges
                          if e.tensor_id == tid)
        if carrying != sorted(want_dsts):
            _issue(issues, "D17_SHARED_TENSOR_EDGE_SET",
                   f"tensors[{tid}]",
                   f"共享张量 {tid} 的边集必须恰为 {sorted(want_dsts)}，"
                   f"得到 {carrying}（生产唯一/消费端口一次核清）")
    for i in range(3):
        state_id = f"dspark:win_cache:mtp.{i}"
        vs = sorted(v.stable_id for v in graph.state_versions
                    if v.state_id == state_id)
        if vs != [f"state:{state_id}:v_after", f"state:{state_id}:v_hist"]:
            _issue(issues, "D17_WIN_CACHE_VERSION_SET", "state_versions",
                   f"{state_id} 必须恰 v_hist/v_after 两版本，得到 {vs}")
        trs = [t for t in graph.state_transitions
               if t.state_id == state_id]
        if len(trs) != 1 or trs[0].stable_id != f"tr_dspark_cache_mtp{i}":
            _issue(issues, "D17_WIN_CACHE_TRANSITION",
                   f"state_transitions[tr_dspark_cache_mtp{i}]",
                   f"{state_id} 迁移必须唯一 tr_dspark_cache_mtp{i}，"
                   f"得到 {len(trs)} 条")
        elif (trs[0].before_version_id, trs[0].after_version_id) != (
                f"state:{state_id}:v_hist",
                f"state:{state_id}:v_after"):
            # 有向逐字段：before 必须是 v_hist（读历史）、after 必须是
            # v_after（本调用写入后发布）——方向反转（v_after→v_hist）
            # 不是同一迁移（codex_b4_review03 mutant 的关闭：不比无序
            # 集合，写前/写后语义有向）
            _issue(issues, "D17_WIN_CACHE_TRANSITION_DIRECTION",
                   f"state_transitions[tr_dspark_cache_mtp{i}]",
                   f"{state_id} 迁移必须有向 v_hist→v_after"
                   f"（before=读历史、after=本调用写后发布），得到 "
                   f"{trs[0].before_version_id}→{trs[0].after_version_id}")
    # Markov 版本迁移有向逐字段：第 i 步 before=v{i}/after=v{i+1}——
    # 版本名称集合相同不代表方向正确（反转即检出）
    for state, prefix in (("mk:logits", "tr_mk_logits_s"),
                          ("mk:output_ids", "tr_mk_ids_s")):
        for s in range(5):
            want = (f"state:{state}:v{s}", f"state:{state}:v{s + 1}")
            trs = [t for t in graph.state_transitions
                   if t.stable_id == f"{prefix}{s}"]
            if len(trs) != 1:
                _issue(issues, "D17_MARKOV_TRANSITION_MISSING",
                       f"state_transitions[{prefix}{s}]",
                       f"{state} 第 {s} 步迁移 {prefix}{s} 必须恰一条，"
                       f"得到 {len(trs)} 条")
            elif (trs[0].before_version_id, trs[0].after_version_id) != want:
                _issue(issues, "D17_MARKOV_TRANSITION_DIRECTION",
                       f"state_transitions[{prefix}{s}]",
                       f"{prefix}{s} 必须有向 {want[0]}→{want[1]}"
                       f"（第 {s} 步消费 v{s}、产出 v{s + 1}），得到 "
                       f"{trs[0].before_version_id}→{trs[0].after_version_id}")

    # D18 入口/stage 组件装配必要链（codex_b4_review03 阻断1 的完整关闭：
    # entry 19 条绝对 ID + stage 模板 84 条×3 实例 + 跨图改接 3×3 条。
    # 每条核验：存在、端点、张量、kind、guard（目录 None = 必须无 guard，
    # 两相位无条件；目录 g_* = 必须恰该 guard，合法分支不并枝不误并）、
    # state 边迁移引用（cache 写后读必须引用本实例 tr_dspark_cache_mtp{i}）、
    # 两端端口张量三方一致。目录语义=已认可组件构建语义迁移到装配命名
    # 空间，不从提取器输出生成。）
    for (eid, src, sport, dst, dport, tid, kind, guard, transition
         ) in dspark_assembly_chain_edges():
        e = edges.get(eid)
        if e is None:
            _issue(issues, "D18_CHAIN_EDGE_MISSING", f"edges[{eid}]",
                   f"装配必要链边 {eid}（{src}.{sport}→{dst}.{dport}，"
                   f"{tid}）缺失——装配图断链")
            continue
        if (e.src_node_id, e.src_port) != (src, sport) or \
                (e.dst_node_id, e.dst_port) != (dst, dport):
            _issue(issues, "D18_CHAIN_EDGE_ENDPOINTS", f"edges[{eid}]",
                   f"装配必要链边 {eid} 必须连 {src}.{sport}→{dst}.{dport}"
                   f"（{tid}），得到 {e.src_node_id}.{e.src_port}→"
                   f"{e.dst_node_id}.{e.dst_port}")
        if e.tensor_id != tid:
            _issue(issues, "D18_CHAIN_EDGE_TENSOR", f"edges[{eid}]",
                   f"装配必要链边 {eid} 必须承载 {tid}，得到 "
                   f"{e.tensor_id!r}")
        if e.kind != kind:
            _issue(issues, "D18_CHAIN_EDGE_KIND", f"edges[{eid}]",
                   f"装配必要链边 {eid} 必须 kind={kind}（目录语义：必要"
                   f"数据流/state 版本推进/路由因果分支），得到 {e.kind!r}")
        if e.guard_ref != guard:
            _issue(issues, "D18_CHAIN_EDGE_GUARD", f"edges[{eid}]",
                   f"装配必要链边 {eid} guard 必须为 "
                   f"{guard if guard else '无（两相位无条件）'}（合法相位"
                   f"分支编码在目录内：cache 写三分支/prefill passthrough/"
                   f"main 共享路径不并枝），得到 {e.guard_ref!r}")
        if e.state_transition_ref != transition:
            _issue(issues, "D18_CHAIN_EDGE_TRANSITION_REF", f"edges[{eid}]",
                   f"装配必要链边 {eid} state 边迁移引用必须为 "
                   f"{transition!r}（cache 写后读指向本实例迁移），得到 "
                   f"{e.state_transition_ref!r}")
        # 两端端口张量三方一致（同轴同步重接在引用合法下语义违规）
        for role, nid, port_name, direction in (
                ("src", e.src_node_id, e.src_port, "output"),
                ("dst", e.dst_node_id, e.dst_port, "input")):
            n = nodes.get(nid)
            if n is None:
                continue  # 节点缺失由通用引用检查具名报告
            p = next((p for p in (n.output_ports if direction == "output"
                                  else n.input_ports)
                      if p.name == port_name), None)
            if p is None or p.tensor_id != e.tensor_id:
                _issue(issues, "D18_CHAIN_EDGE_PORT_TENSOR",
                       f"nodes[{nid}].{port_name}",
                       f"装配必要链边 {eid} 的 {role} 端口 {nid}."
                       f"{port_name} 张量必须与边一致（{e.tensor_id}），"
                       f"得到 {p.tensor_id!r}" if p else f"端口缺失")

    # D19 可信轴/完整 shape 目录（codex_b4_review04 阻断1：装配图独立
    # 维护的可信期望——named_axes 逐轴 kind/value/expression/unknown、
    # 值型派生轴公式交叉、全部张量 dtype+轴序+双 layout 逐字段、图级
    # 集合闭合。不从被检图或 fresh builder 输出刷新正确期望；固定 shape
    # 不凭 symbol 相同放行已知维漂移。）
    from structure_reader.scenario_evaluator import ExpressionError, evaluate_int

    graph_axes = {a.symbol: a for a in graph.named_axes}
    if set(graph_axes) != set(_TRUSTED_AXIS_DEFS):
        _issue(issues, "D19_AXIS_SET_MISMATCH", "named_axes",
               f"named_axes 符号集必须恰为可信轴目录（{len(_TRUSTED_AXIS_DEFS)}"
               f" 条），多出 {sorted(set(graph_axes) - set(_TRUSTED_AXIS_DEFS))}、"
               f"缺少 {sorted(set(_TRUSTED_AXIS_DEFS) - set(graph_axes))}")
    axis_values: dict[str, int] = {}
    for symbol, (kind, expected) in _TRUSTED_AXIS_DEFS.items():
        ax = graph_axes.get(symbol)
        if ax is None:
            continue  # 缺失已由 D19_AXIS_SET_MISMATCH 具名
        if kind == _AXIS_KIND_VALUE:
            if ax.value is None or ax.expression is not None:
                _issue(issues, "D19_AXIS_KIND_MISMATCH",
                       f"named_axes[axis:{symbol}]",
                       f"轴 {symbol} 必须是固定值轴（value={expected}），"
                       f"得到 value={ax.value!r} expression={ax.expression!r}")
            elif ax.value != expected:
                _issue(issues, "D19_AXIS_VALUE_MISMATCH",
                       f"named_axes[axis:{symbol}]",
                       f"轴 {symbol} 可信值必须 {expected}，得到 {ax.value}"
                       "（不从被检图刷新正确期望）")
            else:
                axis_values[symbol] = ax.value
        elif kind == _AXIS_KIND_EXPRESSION:
            if ax.expression is None or ax.value is not None:
                _issue(issues, "D19_AXIS_KIND_MISMATCH",
                       f"named_axes[axis:{symbol}]",
                       f"轴 {symbol} 必须是表达式轴（canonical "
                       f"{expected!r}），得到 value={ax.value!r} "
                       f"expression={ax.expression!r}")
            elif ax.expression != expected:
                _issue(issues, "D19_AXIS_EXPRESSION_MISMATCH",
                       f"named_axes[axis:{symbol}]",
                       f"轴 {symbol} canonical 表达式必须 {expected!r}，"
                       f"得到 {ax.expression!r}")
        else:  # unknown
            if ax.value is not None or ax.expression is not None:
                _issue(issues, "D19_AXIS_KIND_MISMATCH",
                       f"named_axes[axis:{symbol}]",
                       f"轴 {symbol} 静态未知（数据相关），不得给出 value/"
                       f"expression（未知不转 0/false），得到 value="
                       f"{ax.value!r} expression={ax.expression!r}")
            elif not (ax.unknown_reason and ax.unknown_reason.strip()):
                _issue(issues, "D19_AXIS_UNKNOWN_REASON_MISSING",
                       f"named_axes[axis:{symbol}]",
                       f"未知轴 {symbol} 必须给出非空 unknown_reason")
            else:
                want_reason = _TRUSTED_AXIS_UNKNOWN_REASONS.get(symbol)
                if want_reason is not None and \
                        ax.unknown_reason != want_reason:
                    _issue(issues, "D19_AXIS_UNKNOWN_REASON_MISMATCH",
                           f"named_axes[axis:{symbol}].unknown_reason",
                           f"未知轴 {symbol} 可信 unknown_reason 必须 "
                           f"{want_reason!r}，得到 {ax.unknown_reason!r}"
                           "（真实未知原因不得替换成'已知 0'等误导文本）")
        # 可信来源定义（codex_b4_review05 阻断1 的关闭：数值/符号正确
        # 不等于来源语义正确——V 四字段借用真实 H 轴同步误写时，发布页
        # 与 tensor_axes CSV 把词表解释为隐藏维度、来源错指 dim。逐字段
        # 与独立维护目录核对；张量内嵌轴同目录另查（与 named 的一致性
        # 仍由通用 V2 保留，此处直接锚可信源）。）
        want_src = _TRUSTED_AXIS_SOURCES.get(symbol)
        if want_src is not None:
            got_src = (ax.name, ax.meaning, ax.origin, ax.evidence_ref)
            if got_src != want_src:
                _issue(issues, "D19_AXIS_SOURCE_MISMATCH",
                       f"named_axes[axis:{symbol}]",
                       f"轴 {symbol} 可信来源定义 name/meaning/origin/"
                       f"evidence_ref 必须为 {want_src}，得到 {got_src}"
                       "（数值/符号正确不等于来源语义正确；不从被检图"
                       "刷新正确期望）")
        # expression 轴对图内引用轴实际值重算交叉（表达式串相同但引用
        # 轴值漂移时仍检出值不一致）
        formula = _TRUSTED_AXIS_DERIVED_FORMULAS.get(symbol)
        if formula is not None and ax is not None:
            if ax.is_unknown:
                _issue(issues, "D19_DERIVED_AXIS_CROSSCHECK",
                       f"named_axes[axis:{symbol}]",
                       f"派生轴 {symbol} 不得为 unknown（{formula} 可静态"
                       "交叉）")
            else:
                refs_ok = True
                env: dict[str, int] = {}
                import re as _re
                for dep in sorted(set(_re.findall(r"[A-Za-z_][A-Za-z_0-9]*",
                                                  formula))):
                    dep_ax = graph_axes.get(dep)
                    if dep_ax is None or dep_ax.value is None:
                        refs_ok = False
                        _issue(issues, "D19_DERIVED_AXIS_CROSSCHECK",
                               f"named_axes[axis:{symbol}]",
                               f"派生轴 {symbol} 公式 {formula} 的引用轴 "
                               f"{dep} 在图中无确定值，无法交叉核验")
                    else:
                        env[dep] = dep_ax.value
                if refs_ok and symbol in env and symbol not in formula:
                    pass  # 标量轴不参与公式
                if refs_ok:
                    actual = (ax.value if ax.value is not None
                              else None)
                    if ax.value is not None:
                        try:
                            want = evaluate_int(formula, env)
                        except ExpressionError as exc:
                            refs_ok = False
                            _issue(issues, "D19_DERIVED_AXIS_CROSSCHECK",
                                   f"named_axes[axis:{symbol}]",
                                   f"派生轴 {symbol} 公式 {formula} 求值"
                                   f"失败: {exc}")
                        if refs_ok and actual != want:
                            _issue(issues, "D19_DERIVED_AXIS_CROSSCHECK",
                                   f"named_axes[axis:{symbol}]",
                                   f"派生轴 {symbol} 值 {actual} 与公式 "
                                   f"{formula}（引用轴实际值 {env}）重算 "
                                   f"{want} 不一致——表达式串/值漂移")
    # 张量完整规格目录：dtype + 轴符号序（rank）+ logical/physical layout
    # + 可选秩标注（global_rank/local_rank 非 None 必须等于实际轴数，两
    # 字段分别核对；None 合法保留——首版 world_size=1，None=未标注不是
    # 错，不得把 None 改写为 0 或把任意非负整数当并行 rank 放行）
    trusted_tensors = dspark_trusted_tensor_catalog()
    graph_tensors_full = {t.stable_id: t for t in graph.tensors}
    for tid, (want_dt, want_axes) in trusted_tensors.items():
        t = graph_tensors_full.get(tid)
        if t is None:
            _issue(issues, "D19_TENSOR_MISSING", f"tensors[{tid}]",
                   f"可信张量目录条目 {tid} 缺失")
            continue
        got_axes = tuple(a.symbol for a in t.axes)
        if t.dtype != want_dt or got_axes != want_axes:
            _issue(issues, "D19_TENSOR_SPEC", f"tensors[{tid}]",
                   f"{tid} 可信规格 dtype={want_dt} axes={list(want_axes)}，"
                   f"得到 dtype={t.dtype!r} axes={list(got_axes)}"
                   "（完整 rank/layout/shape 逐字段，不凭 symbol 相同放行"
                   "已知维漂移）")
        else:
            syms = list(want_axes)
            if list(t.logical_layout) != syms or \
                    list(t.physical_layout) != syms:
                _issue(issues, "D19_TENSOR_SPEC", f"tensors[{tid}]",
                       f"{tid} layout 必须为轴符号序（logical="
                       f"{list(t.logical_layout)} physical="
                       f"{list(t.physical_layout)}，目录期望 {syms}）——"
                       "rank-layout-shape 完整核验")
        # 可选秩标注与实际轴数一致（两字段分别检查，None 合法跳过；
        # 与 IR 的"仅秩标注、非负整数"契约互补——IR 不与 len(axes) 交叉，
        # 标注矛盾在此具名拒绝）
        actual_rank = len(t.axes)
        for rank_field, got_rank in (("global_rank", t.global_rank),
                                     ("local_rank", t.local_rank)):
            if got_rank is not None and got_rank != actual_rank:
                _issue(issues, "D19_TENSOR_RANK_METADATA",
                       f"tensors[{tid}].{rank_field}",
                       f"{tid} 显式 {rank_field}={got_rank} 与实际轴数 "
                       f"{actual_rank} 矛盾（None=未标注合法保留；非 None "
                       "必须等于 len(axes)，不得当并行 rank/任意标注放行）")
        # 张量内嵌轴来源定义与 named 同一可信目录（发布页/CSV 直接消费
        # 内嵌字段；仅靠 V2 与 named 一致不够——named 被同步误写时两者
        # 一致但语义错，此处直接锚可信源，与 D19_AXIS_SOURCE_MISMATCH
        # 同目录）
        for a_index, ax in enumerate(t.axes):
            want_src = _TRUSTED_AXIS_SOURCES.get(ax.symbol)
            if want_src is None:
                continue
            got_src = (ax.name, ax.meaning, ax.origin, ax.evidence_ref)
            if got_src != want_src:
                _issue(issues, "D19_TENSOR_AXIS_SOURCE",
                       f"tensors[{tid}].axes[{a_index}]",
                       f"{tid} 内嵌轴 {ax.symbol} 可信来源定义必须为 "
                       f"{want_src}，得到 {got_src}（与 named_axes 同一"
                       "可信目录；数值/符号正确不等于来源语义正确）")
    unexpected = sorted(set(graph_tensors_full) - set(trusted_tensors))
    if unexpected:
        _issue(issues, "D19_TENSOR_UNEXPECTED", "tensors",
               f"图内出现可信张量目录之外的张量: {unexpected[:8]}"
               f"（共 {len(unexpected)} 条；目录闭合，不从被检图刷新）")

    # D20 guard 谓词可信定义（codex_b4_review04 阻断2：guard_ref ID
    # 正确不守谓词语义——g_decode start_pos>0→>=0 后 prefill 误启
    # decode。四 guard canonical 谓词词法 + evidence 锁定 + 白名单求值
    # 器真/假边界核验 + 定义/引用闭合；不执行参考模型。）
    guard_by_id = {gd.stable_id: gd for gd in graph.guards}
    if set(guard_by_id) != set(_TRUSTED_GUARD_DEFS):
        _issue(issues, "D20_GUARD_MISSING", "guards",
               f"guards 定义集必须恰为可信四 guard 目录，多出 "
               f"{sorted(set(guard_by_id) - set(_TRUSTED_GUARD_DEFS))}、"
               f"缺少 {sorted(set(_TRUSTED_GUARD_DEFS) - set(guard_by_id))}")
    for gid, (canonical, want_ev) in _TRUSTED_GUARD_DEFS.items():
        gd = guard_by_id.get(gid)
        if gd is None:
            _issue(issues, "D20_GUARD_MISSING", f"guards[{gid}]",
                   f"可信 guard {gid} 缺失（D5/D16/D18 的 guard_ref ID "
                   "核验不覆盖谓词语义）")
            continue
        if gd.predicate_expr != canonical:
            _issue(issues, "D20_GUARD_PREDICATE_CANONICAL",
                   f"guards[{gid}].predicate_expr",
                   f"{gid} 谓词必须逐字等于可信 canonical {canonical!r}，"
                   f"得到 {gd.predicate_expr!r}（guard_ref ID 不变时谓词"
                   "语义漂移即检出：两相位误并/阈值偏移/边界互换）")
        if tuple(sorted(gd.evidence_refs)) != tuple(sorted(want_ev)):
            _issue(issues, "D20_GUARD_EVIDENCE", f"guards[{gid}].evidence_refs",
                   f"{gid} 证据引用必须 {sorted(want_ev)}，得到 "
                   f"{sorted(gd.evidence_refs)}")
        # 白名单求值器真/假边界核验（base 环境 + 场景变量；不执行模型）
        true_envs, false_envs = _TRUSTED_GUARD_SEMANTICS[gid]
        for env in true_envs:
            full = {"B": 1, "T": 1, "P": 0, "start_pos": env.get("P", 0),
                    "W": 128, **env}
            from structure_reader.scenario_evaluator import evaluate_tri
            r = evaluate_tri(gd.predicate_expr, full)
            if r.is_known and not r.value:
                _issue(issues, "D20_GUARD_TRUE_CASE_FALSE",
                       f"guards[{gid}].predicate_expr",
                       f"谓词 {gd.predicate_expr!r} 在应为 true 的环境 "
                       f"{env} 求值为 false——相位借用或边界错误")
        for env in false_envs:
            full = {"B": 1, "T": 1, "P": 0, "start_pos": env.get("P", 0),
                    "W": 128, **env}
            from structure_reader.scenario_evaluator import evaluate_tri
            r = evaluate_tri(gd.predicate_expr, full)
            if r.is_known and r.value:
                _issue(issues, "D20_GUARD_FALSE_CASE_TRUE",
                       f"guards[{gid}].predicate_expr",
                       f"谓词 {gd.predicate_expr!r} 在应为 false 的环境 "
                       f"{env} 求值为 true——两相位误并/窗口分支互换/"
                       "阈值偏移")
    # 引用闭合：边挂的 guard_ref 必须都在可信四 guard 内（新 guard 借
    # 引用混入即检出）
    edge_guard_refs = {e.guard_ref for e in graph.edges
                       if e.guard_ref is not None}
    stray = sorted(edge_guard_refs - set(_TRUSTED_GUARD_DEFS))
    if stray:
        _issue(issues, "D20_GUARD_REFERENCE_CLOSED", "edges",
               f"边引用的 guard 必须都在可信四 guard 目录内，发现目录外 "
               f"引用: {stray}")
    # unknown 不转 false：谓词含未定义变量时求值必须 unknown（抽查
    # canonical 谓词变量集与场景环境覆盖——start_pos/T/W/P 全在场景 env
    # 中可达，任何 guard 若引用目录外变量，在其变量缺失环境下求值应得
    # unknown 而非 false）
    for gid, gd in guard_by_id.items():
        if gid not in _TRUSTED_GUARD_DEFS:
            continue
        from structure_reader.scenario_evaluator import evaluate_tri, _parse
        try:
            tree = _parse(gd.predicate_expr)
            import ast as _ast
            needed = {n.id for n in _ast.walk(tree)
                      if isinstance(n, _ast.Name)
                      and not isinstance(n.ctx, _ast.Store)}
        except Exception:
            continue  # 非法表达式已由 canonical 词法锁定检出
        empty_env = {"B": 1, "T": 1, "P": 0, "start_pos": 0, "W": 128,
                     "r": 4}
        missing = needed - set(empty_env)
        if missing:
            r = evaluate_tri(gd.predicate_expr, empty_env)
            if r.is_known:
                _issue(issues, "D20_GUARD_UNKNOWN_COLLAPSED",
                       f"guards[{gid}].predicate_expr",
                       f"谓词 {gd.predicate_expr!r} 引用环境外变量 "
                       f"{sorted(missing)}，缺失时必须 unknown，求值却得 "
                       f"{r.value!r}——unknown 被压成确定值")

    return tuple(issues)


# ---------------------------------------------------------------------------
# 受控错误注入
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DsparkMutation:
    """一种集成受控错误：注入方式与期望检出的契约码。"""

    mutation_id: str
    description: str
    expected_codes: tuple[str, ...]


def dspark_integration_error_specs() -> tuple[DsparkMutation, ...]:
    return (
        DsparkMutation("entry_readout_reordered",
                       "readout concat 端口 s0/s1 交换（40/41 换序）",
                       ("D2_READOUT_ORDER",)),
        DsparkMutation("entry_readout_dropped",
                       "删 mean_41→concat 边（target 41 读出丢失）",
                       ("D2_READOUT_ORDER",)),
        DsparkMutation("main_x_skip_norm",
                       "stage0 main_x 改接 main_proj 输出（跳过 main_norm，"
                       "main_x 唯一生产破坏）",
                       ("D3_MAIN_X_SINGLE_PRODUCER",)),
        DsparkMutation("stage_exit_wrong_stage",
                       "exit 输入改接 mtp1 h_out（末出口 stage 角色错）",
                       ("D6_EXIT_SOURCE",)),
        DsparkMutation("prefill_draft_unguarded",
                       "草稿 Q 输入边去 guard（prefill 误启完整 draft）",
                       ("D5_DECODE_GUARD_REQUIRED",)),
        DsparkMutation("confidence_postnorm",
                       "Confidence hidden 改接 exit:norm_x（postnorm 替换）",
                       ("D6_CONFIDENCE_HIDDEN_PRENORM",)),
        DsparkMutation("confidence_sampled_embed",
                       "Confidence embed 改接第 4 步单步 embed（非有序 "
                       "stack）",
                       ("D8_CONFIDENCE_EMBED_STACK",)),
        DsparkMutation("logits_version_skipped",
                       "第 2 步切片改读第 0 步写输出（logits 版本断开）",
                       ("D7_STEP_VERSION_LINK",)),
        DsparkMutation("connection_edge_to_control",
                       "Confidence hidden 跨图边 kind 改 control",
                       ("D9_CONNECTION_EDGE_KIND",)),
        DsparkMutation("exit_h_in_lost",
                       "删 exit h_in 边（末出口丢失）",
                       ("D5_DECODE_EDGE_MISSING", "D6_EXIT_SOURCE")),
        DsparkMutation("triple_output_ids_lost",
                       "删 output_ids 终版输出边（三元组断链）",
                       ("D7_FINAL_IDS_CONSUME_V5",)),
        DsparkMutation("shared_embed_copied",
                       "注入第二份 embed 权重并改接（共享 embed 复制）",
                       ("D10_SHARED_EMBED_SINGLE",)),
        # --- codex_b4_review01 五个原始 mutant 逐字段等价回放 ---
        DsparkMutation("cf_float_proj_deleted",
                       "删必要 e_cf_float_proj（显式 float→proj 断链）",
                       ("D11_CF_FLOAT_PROJ_MISSING",)),
        DsparkMutation("cf_float_proj_control",
                       "该必要 data 边 e_cf_float_proj 改 control",
                       ("D11_CF_FLOAT_PROJ_KIND",)),
        DsparkMutation("head_gate_sinkhorn",
                       "exit.hc_head.gate op_type sigmoid_gate→sinkhorn",
                       ("D12_HC_HEAD_GATE_OP", "D15_OP_MISMATCH")),
        DsparkMutation("final_logits_bf16",
                       "tensor:mk:logits dtype fp32→bf16",
                       ("D13_MK_LOGITS_DTYPE",)),
        DsparkMutation("middle_stage_bypass_stage0",
                       "mtp1.e_st_hcin_hcpre 及目标端口同步改接 "
                       "entry.stage_feed_0.h_out，跳过 stage0 真实 h 输出",
                       ("D14_STAGE_CHAIN_DECODE",)),
        # --- 必要边删改 / 同轴同步篡改扩展负例 ---
        DsparkMutation("cf_float_proj_endpoints_swapped",
                       "e_cf_float_proj 两端互换（to_float 消费 proj 输出）",
                       ("D11_CF_FLOAT_PROJ_ENDPOINTS",)),
        DsparkMutation("cf_float_proj_tensor_swapped",
                       "e_cf_float_proj 及两端端口同步改接 bf16 concat"
                       "（fp32 显式升链断开，同轴同步篡改）",
                       ("D11_CF_FLOAT_PROJ_TENSOR", "D11_CF_PROJ_PORT_TENSOR")),
        DsparkMutation("stage2_chain_to_entry",
                       "mtp2.e_st_hcin_hcpre 改接 mtp0 输出（跳过 stage1）",
                       ("D14_STAGE_CHAIN_DECODE",)),
        DsparkMutation("stage1_prefill_to_entry",
                       "mtp1.e_st_prefill_in 改接 entry h_out（跳过 stage0 "
                       "passthrough）",
                       ("D14_STAGE_CHAIN_PREFILL",)),
        DsparkMutation("head_gate_op_swap",
                       "exit.norm op 改 passthrough（末出口 norm 失效）",
                       ("D15_OP_MISMATCH",)),
        DsparkMutation("stack_embed_to_single",
                       "mk.stack_embed op 改 concat（有序 stack 失效）",
                       ("D15_OP_MISMATCH",)),
        DsparkMutation("exit_head_op_removed",
                       "exit.head op 改 sinkhorn_split（共享 head 算子篡改）",
                       ("D15_OP_MISMATCH",)),
        # --- codex_b4_review02 具名删除断链（D16 装配图必要链目录）---
        DsparkMutation("e_cf_concat_float_deleted",
                       "删 e_cf_concat_float（concat→to_float 断链）",
                       ("D16_CHAIN_EDGE_MISSING",)),
        DsparkMutation("e_cf_proj_squeeze_deleted",
                       "删 e_cf_proj_squeeze（proj→squeeze 断链）",
                       ("D16_CHAIN_EDGE_MISSING",)),
        DsparkMutation("e_ex_mix_gate_deleted",
                       "删 e_ex_mix_gate（hc_head mixes→sigmoid 门控断链）",
                       ("D16_CHAIN_EDGE_MISSING",)),
        DsparkMutation("e_mk_s2_add_sample_deleted",
                       "删 e_mk_s2_add_sample（K5 中步更新 logits→sample "
                       "断链）",
                       ("D16_CHAIN_EDGE_MISSING",)),
        # --- D16 降级/改接扩展负例（与删除断链同目录同检出口）---
        DsparkMutation("e_ex_mix_gate_to_control",
                       "e_ex_mix_gate kind data→control（必要流降级）",
                       ("D16_CHAIN_EDGE_KIND",)),
        DsparkMutation("e_mk_s2_add_sample_unguarded",
                       "e_mk_s2_add_sample 去 guard（decode 出口链相位 "
                       "守卫缺失）",
                       ("D16_CHAIN_EDGE_GUARD",)),
        DsparkMutation("e_cf_proj_squeeze_rewired",
                       "e_cf_proj_squeeze/e_cf_squeeze_out 双边轮换改接"
                       "（proj 直进输出、squeeze 回环，绕过 squeeze；"
                       "端点偏离目录）",
                       ("D16_CHAIN_EDGE_ENDPOINTS",)),
        DsparkMutation("e_ex_mix_gate_rewired",
                       "e_ex_mix_gate/e_ex_gate_reduce 双边轮换改接"
                       "（mixes 直进 reduce、gate 回环；端点偏离目录）",
                       ("D16_CHAIN_EDGE_ENDPOINTS",)),
        DsparkMutation("e_ex_scale_gate_tensor_swapped",
                       "e_ex_scale_gate 与两端端口同步改接 base 权重张量"
                       "（scale/base 参数错位，同轴同步篡改）",
                       ("D16_CHAIN_EDGE_TENSOR",)),
        DsparkMutation("mk_w2_param_copied",
                       "注入第二份 markov_w2 权重并改接 step4.bias"
                       "（共享参数复制）",
                       ("D17_SHARED_TENSOR_EDGE_SET",)),
        DsparkMutation("mk_step0_logits_updated_bf16",
                       "tensor:mk:step0_logits_updated dtype fp32→bf16"
                       "（中步降精度断链）",
                       ("D17_TENSOR_DTYPE",)),
        DsparkMutation("confidence_out_axes_swapped",
                       "tensor:cf:confidence 轴序 [B,N_draft]→"
                       "[N_draft,B]（出口三元组形状篡改）",
                       ("D17_TRIPLE_AXES",)),
        # --- codex_b4_review03 五份实际坏图等价注入（refs0 引用合法、
        # 原始契约必具名 fail；实际 CLI 临时隔离 rc1/0HTML）---
        DsparkMutation("e_ep_cat_proj_deleted",
                       "删 e_ep_cat_proj（readout concat→main_proj 入口"
                       "必要链断链）",
                       ("D18_CHAIN_EDGE_MISSING",)),
        DsparkMutation("mtp0_e_st_hcpre_a_out_deleted",
                       "删 mtp0.e_st_hcpre_a_out（stage0 hc_pre_attn 汇聚"
                       "→attn_norm 断链）",
                       ("D18_CHAIN_EDGE_MISSING",)),
        DsparkMutation("mtp0_e_ds_cache_state_deleted",
                       "删 mtp0.e_ds_cache_state（stage0 cache 写后读"
                       "state 边断链）",
                       ("D18_CHAIN_EDGE_MISSING",)),
        DsparkMutation("mk_step2_sample_argmax",
                       "mk.step2.sample op gumbel_max_sample→argmax（边/"
                       "张量不变，算子身份偷换）",
                       ("D15_OP_MISMATCH",)),
        DsparkMutation("mtp0_cache_transition_reversed",
                       "stage0 cache 迁移方向反转 v_hist→v_after 改 "
                       "v_after→v_hist（有向逐字段核验拒绝）",
                       ("D17_WIN_CACHE_TRANSITION_DIRECTION",)),
        # --- D18 降级/改接/相位分支扩展负例 ---
        DsparkMutation("e_ep_norm_feed1_to_control",
                       "e_ep_norm_feed1 kind data→control（main_x 分发"
                       "降级）",
                       ("D18_CHAIN_EDGE_KIND",)),
        DsparkMutation("mtp1_e_st_hcpre_a_out_unguarded",
                       "mtp1.e_st_hcpre_a_out 去 guard（decode-only 骨架"
                       "边相位守卫缺失）",
                       ("D18_CHAIN_EDGE_GUARD",)),
        DsparkMutation("mtp2_e_st_cachew_dec_unguarded",
                       "mtp2.e_ds_cachew_dec 去 guard（decode 单槽写分支"
                       "守卫缺失）",
                       ("D18_CHAIN_EDGE_GUARD",)),
        DsparkMutation("mtp0_e_ds_cachew_inwin_guard_swapped",
                       "mtp0.e_ds_cachew_inwin guard 改 g_decode（prefill"
                       " 顺序写分支并枝到 decode——合法分支篡改）",
                       ("D18_CHAIN_EDGE_GUARD",)),
        DsparkMutation("mtp0_e_st_prefill_in_guard_swapped",
                       "mtp0.e_st_prefill_in guard g_prefill→g_decode"
                       "（prefill passthrough 并枝 decode）",
                       ("D18_CHAIN_EDGE_GUARD",)),
        DsparkMutation("mtp1_e_ds_cache_state_transition_ref_dangling",
                       "mtp1.e_ds_cache_state 迁移引用改 stage0 的 "
                       "tr_dspark_cache_mtp0（跨实例状态引用错位）",
                       ("D18_CHAIN_EDGE_TRANSITION_REF",)),
        DsparkMutation("e_ep_cat_proj_rewired",
                       "e_ep_cat_proj/e_ep_proj_norm 双边轮换改接（concat"
                       " 直进 main_norm、proj 回环；端点偏离目录）",
                       ("D18_CHAIN_EDGE_ENDPOINTS",)),
        DsparkMutation("mtp0_e_st_dispatch_gate_to_data",
                       "mtp0.e_st_dispatch_gate kind control→data（counts"
                       " 门控语义篡改）",
                       ("D18_CHAIN_EDGE_KIND",)),
        DsparkMutation("mtp0_e_ds_idx_select_to_data",
                       "mtp0.e_ds_idx_select kind select→data（选择因果"
                       "语义篡改）",
                       ("D18_CHAIN_EDGE_KIND",)),
        DsparkMutation("mtp1_stage_x_hc_attn_flat_bf16",
                       "tensor:mtp1.stage:x_hc_attn_flat dtype fp32→bf16"
                       "（stage HC 展开 fp32 升链降精度）",
                       ("D17_TENSOR_DTYPE",)),
        DsparkMutation("entry_main_proj_w_dtype_swapped",
                       "tensor:entry:main_proj_w dtype fp8_e4m3fn→bf16"
                       "（主投影 fp8 权重类型篡改）",
                       ("D17_TENSOR_DTYPE",)),
        DsparkMutation("mtp0_e_st_norm_xsrc_rewired",
                       "mtp0.e_st_norm_xsrc/e_ds_xin_wqa 双边轮换改接"
                       "（attn_norm 直进 wq_a、源节点回环；端点偏离目录）",
                       ("D18_CHAIN_EDGE_ENDPOINTS",)),
        DsparkMutation("mk_ids_s2_transition_reversed",
                       "tr_mk_ids_s2 方向反转 v2→v3 改 v3→v2（Markov "
                       "output_ids 版本推进有向核验拒绝）",
                       ("D17_MARKOV_TRANSITION_DIRECTION",)),
        DsparkMutation("mk_logits_s0_transition_reversed",
                       "tr_mk_logits_s0 方向反转 v0→v1 改 v1→v0（Markov"
                       " logits 版本推进有向核验拒绝）",
                       ("D17_MARKOV_TRANSITION_DIRECTION",)),
        # --- codex_b4_review04 三份实际坏图等价注入（refs0 引用合法、
        # 原始契约必具名 fail；实际 CLI 临时隔离 rc1/0HTML）---
        DsparkMutation("synced_vocab_axis_129281",
                       "V 轴 named+全部张量内嵌值同步 129280→129281"
                       "（symbol/引用合法；与可信输入 vocab_size 矛盾）",
                       ("D19_AXIS_VALUE_MISMATCH",)),
        DsparkMutation("seed_rank2_borrowed_axis",
                       "entry seed 轴 [B]→[B,C1_exit]（rank1 契约破坏、"
                       "借用已有轴、logical/physical 同步、引用全合法）",
                       ("D19_TENSOR_SPEC",)),
        DsparkMutation("g_decode_predicate_includes_prefill",
                       "g_decode 谓词 start_pos > 0 改 >= 0（guard_ref ID"
                       " 不变；prefill start_pos=0 误启 decode 出口）",
                       ("D20_GUARD_PREDICATE_CANONICAL",
                        "D20_GUARD_FALSE_CASE_TRUE")),
        # --- D19/D20 扩展系统负例（同步数值/表达式篡改、layout 漂移、
        # guard 窗口互换/证据漂移/unknown 压缩）---
        DsparkMutation("rm_axis_sync_257",
                       "Rm 全局 256→257（w1_w/w2_w/markov_embed/step*_embed"
                       " 同步漂移；值型轴逐字核出）",
                       ("D19_AXIS_VALUE_MISMATCH",)),
        DsparkMutation("n_mix_expression_drift",
                       "N_mix canonical 表达式改 N_hc * H（值同 24 但"
                       "词法漂移；派生公式交叉同时检出）",
                       ("D19_AXIS_EXPRESSION_MISMATCH",)),
        DsparkMutation("ne_axis_collapsed_zero",
                       "N_e 数据相关未知轴给 value=0（未知不转 0/false）",
                       ("D19_AXIS_KIND_MISMATCH",)),
        DsparkMutation("cf_proj_w_layout_transposed",
                       "tensor:cf:proj_w physical_layout 转置"
                       "[C1_conf,H_plus_Rm]→[H_plus_Rm,C1_conf]（rank 同、"
                       "layout 漂移）",
                       ("D19_TENSOR_SPEC",)),
        DsparkMutation("mtp1_win_cache_axes_drift",
                       "tensor:mtp1.stage:win_cache 轴序 [B,W,D]→[B,D,W]"
                       "（仅 stage1 实例；逐实例核验）",
                       ("D19_TENSOR_SPEC",)),
        DsparkMutation("mk_step3_written_axes_bv",
                       "tensor:mk:step3_written 轴 [B]→[B,V]（步张量"
                       "模板逐条核验）",
                       ("D19_TENSOR_SPEC",)),
        DsparkMutation("seed_axis_dtype_retyped",
                       "tensor:entry:seed_ids dtype int64→int32（rank 同、"
                       "dtype 漂移）",
                       ("D19_TENSOR_SPEC",)),
        DsparkMutation("phantom_axis_added",
                       "注入新 named axis X=7（轴集合闭合拒绝）",
                       ("D19_AXIS_SET_MISMATCH",)),
        DsparkMutation("phantom_tensor_added",
                       "注入新张量 tensor:entry:phantom（张量目录闭合拒绝）",
                       ("D19_TENSOR_UNEXPECTED",)),
        DsparkMutation("window_guard_threshold_shifted",
                       "g_window_wrap 谓词 T > W 改 T >= W（阈值偏移："
                       "T==W 属 inwin 非 wrap）",
                       ("D20_GUARD_PREDICATE_CANONICAL",
                        "D20_GUARD_FALSE_CASE_TRUE")),
        DsparkMutation("inwin_wrap_swapped",
                       "g_dspark_prefill_inwin 与 g_window_wrap 谓词互换"
                       "（窗口两分支互换）",
                       ("D20_GUARD_PREDICATE_CANONICAL",)),
        DsparkMutation("g_prefill_widened_neq",
                       "g_prefill 谓词 start_pos == 0 改 start_pos <= 0"
                       "（canonical 词法锁定；相位边界漂移）",
                       ("D20_GUARD_PREDICATE_CANONICAL",)),
        DsparkMutation("g_decode_evidence_dropped",
                       "g_decode 证据引用删 ev_ex_spec_return（evidence "
                       "锁定核验）",
                       ("D20_GUARD_EVIDENCE",)),
        # --- codex_b4_review05 阻断1 扩展负例：轴来源语义（name/meaning/
        # origin/evidence_ref 同步误写、数值/符号不变仍检出；N_e 未知原因
        # 误导文本）---
        DsparkMutation("vocab_source_claims_hidden",
                       "V 来源四字段借用真实 H 轴（hidden_size/隐藏维度/"
                       "run config dim/run_config.json:3，named+张量内嵌"
                       "同步；数值 129280/符号不变）——词表被解释为隐藏维度",
                       ("D19_AXIS_SOURCE_MISMATCH", "D19_TENSOR_AXIS_SOURCE")),
        DsparkMutation("rm_source_misattributed",
                       "Rm 来源四字段借用 D 轴（head_dim/注意力头维度/"
                       "run config head_dim/run_config.json:20，named+张量"
                       "内嵌同步；数值 256 不变）——Markov 秩来源错指",
                       ("D19_AXIS_SOURCE_MISMATCH", "D19_TENSOR_AXIS_SOURCE")),
        DsparkMutation("h_plus_rm_meaning_drifted",
                       "派生轴 H_plus_Rm 含义/来源漂移（confidence 输入宽"
                       "→hidden 拼接宽度、ev_cf_conf_head→model.py:1，数值"
                       "4352 不变）",
                       ("D19_AXIS_SOURCE_MISMATCH",
                        "D19_TENSOR_AXIS_SOURCE")),
        DsparkMutation("ne_unknown_reason_misleading",
                       "N_e unknown_reason 替换为误导文本'已知为 0（静态 "
                       "可求值）'（unknown kind 保持、原因语义漂移）",
                       ("D19_AXIS_UNKNOWN_REASON_MISMATCH",)),
        DsparkMutation("n_draft_plus_1_name_drifted",
                       "N_draft_plus_1 name 漂移（dspark_block_size_plus_1"
                       "→N_draft_plus_one、named+内嵌同步）",
                       ("D19_AXIS_SOURCE_MISMATCH",
                        "D19_TENSOR_AXIS_SOURCE")),
        # --- codex_b4_review05 阻断2 扩展负例：显式秩标注与轴数矛盾
        # （None 合法保留；非 None 必须等于 len(axes)，两字段分别核对）---
        DsparkMutation("seed_rank_metadata99",
                       "tensor:entry:seed_ids global_rank/local_rank 均 "
                       "99（axes [B] rank1 不变；可选秩标注矛盾）",
                       ("D19_TENSOR_RANK_METADATA",)),
        DsparkMutation("cf_proj_w_rank_metadata_borrowed",
                       "tensor:cf:proj_w global_rank=2（恰与 rank 相同）/"
                       "local_rank=99（矛盾）——两字段分别核对",
                       ("D19_TENSOR_RANK_METADATA",)),
        DsparkMutation("mk_step2_rank_metadata99",
                       "tensor:mk:step2_embed 双秩标注 99（Markov 中步）",
                       ("D19_TENSOR_RANK_METADATA",)),
        DsparkMutation("entry_logits_rank_metadata0",
                       "tensor:exit:logits 双秩标注 0（把 None 未标注改写"
                       "为 0 的等价负例：0≠3）",
                       ("D19_TENSOR_RANK_METADATA",)),
    )


def apply_dspark_mutation(graph: SemanticGraph, mutation_id: str
                          ) -> SemanticGraph:
    """在真实集成图上注入一种受控错误，返回新图（不改原图）。

    注入保持引用合法（validate_graph_references 不报新引用错误）；语义
    违规由 validate_dspark_integration_contracts 检出。
    """
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    tensors = list(graph.tensors)
    transitions = list(graph.state_transitions)
    named_axes = list(graph.named_axes)
    guards = list(graph.guards)
    tensor_by_id = {t.stable_id: t for t in tensors}

    def replace_transition(tid: str) -> None:
        """一条迁移方向反转（版本引用仍存在且同 state_id，引用合法）；
        before/after 必须不同（StateTransition 有向更新约束）。"""
        for idx, tr in enumerate(transitions):
            if tr.stable_id == tid:
                transitions[idx] = StateTransition(
                    stable_id=tr.stable_id, state_id=tr.state_id,
                    before_version_id=tr.after_version_id,
                    after_version_id=tr.before_version_id,
                    evidence_refs=tuple(tr.evidence_refs))
                return
        raise ExtractionError(f"mutation {mutation_id}: 迁移不存在 {tid}")

    def replace_edge(eid: str, **kw) -> None:
        for idx, e in enumerate(edges):
            if e.stable_id == eid:
                edges[idx] = Edge(
                    stable_id=e.stable_id,
                    src_node_id=kw.get("src_node_id", e.src_node_id),
                    src_port=kw.get("src_port", e.src_port),
                    dst_node_id=kw.get("dst_node_id", e.dst_node_id),
                    dst_port=kw.get("dst_port", e.dst_port),
                    tensor_id=kw.get("tensor_id", e.tensor_id),
                    kind=kw.get("kind", e.kind),
                    evidence_refs=tuple(e.evidence_refs),
                    guard_ref=kw.get("guard_ref", e.guard_ref),
                    state_transition_ref=e.state_transition_ref)
                return
        raise ExtractionError(f"mutation {mutation_id}: 边不存在 {eid}")

    def replace_port(node_id: str, port: str, tensor_id: str) -> None:
        for idx, n in enumerate(nodes):
            if n.stable_id != node_id:
                continue
            ins, outs, changed = [], [], False
            for p in n.input_ports:
                if p.name == port and p.direction == "input":
                    ins.append(Port(p.name, p.direction, tensor_id))
                    changed = True
                else:
                    ins.append(p)
            for p in n.output_ports:
                if p.name == port and p.direction == "output":
                    outs.append(Port(p.name, p.direction, tensor_id))
                    changed = True
                else:
                    outs.append(p)
            if changed:
                nodes[idx] = Node(
                    stable_id=n.stable_id, op_type=n.op_type,
                    granularity=n.granularity, input_ports=tuple(ins),
                    output_ports=tuple(outs),
                    evidence_refs=tuple(n.evidence_refs))
            return
        raise ExtractionError(f"mutation {mutation_id}: 节点不存在 {node_id}")

    if mutation_id == "entry_readout_reordered":
        # 端口交换（张量随边走，引用保持合法）：mean_40 输出（mean_h_40）
        # 进 s1、mean_41 输出（mean_h_41）进 s0 —— D2 依序消费被破坏
        e0 = next(e for e in edges if e.stable_id == "e_ep_mean40_cat")
        e1 = next(e for e in edges if e.stable_id == "e_ep_mean41_cat")
        replace_edge("e_ep_mean40_cat", dst_port=e1.dst_port)
        replace_edge("e_ep_mean41_cat", dst_port=e0.dst_port)
        replace_port("entry.readout_concat", "s0", e1.tensor_id)
        replace_port("entry.readout_concat", "s1", e0.tensor_id)
    elif mutation_id == "entry_readout_dropped":
        edges[:] = [e for e in edges if e.stable_id != "e_ep_mean41_cat"]
    elif mutation_id == "main_x_skip_norm":
        replace_edge("e_ep_norm_feed0", src_node_id="entry.main_proj",
                     tensor_id="tensor:entry:main_proj_y")
        replace_port("entry.stage_feed_0", "x", "tensor:entry:main_proj_y")
    elif mutation_id == "stage_exit_wrong_stage":
        replace_edge("e_ex_hin_flat", src_node_id="mtp1.stage.hc_post_ffn.combine",
                     tensor_id="tensor:mtp1.stage:x_hc_out")
        replace_port("exit.hc_head.flatten", "x", "tensor:mtp1.stage:x_hc_out")
    elif mutation_id == "prefill_draft_unguarded":
        replace_edge("mtp0.e_ds_xin_wqa", guard_ref=None)
    elif mutation_id == "confidence_postnorm":
        replace_edge("e_cf_hidden_in", src_node_id="exit.norm",
                     tensor_id="tensor:exit:norm_x")
        replace_port("cf.concat", "hidden", "tensor:exit:norm_x")
    elif mutation_id == "confidence_sampled_embed":
        # 引用合法：边与 concat.embed 端口张量同步改为第 4 步单步 embed
        replace_edge("e_cf_embed_in", src_node_id="mk.step4.embed",
                     src_port="y", tensor_id="tensor:mk:step4_embed")
        replace_port("cf.concat", "embed", "tensor:mk:step4_embed")
    elif mutation_id == "logits_version_skipped":
        replace_edge("e_mk_s2_slice_read", src_node_id="mk.step0.logits_write")
    elif mutation_id == "connection_edge_to_control":
        replace_edge("e_cf_hidden_in", kind="control")
    elif mutation_id == "exit_h_in_lost":
        edges[:] = [e for e in edges if e.stable_id != "e_ex_hin_flat"]
    elif mutation_id == "triple_output_ids_lost":
        edges[:] = [e for e in edges if e.stable_id != "e_mk_s4_write_ids"]
    elif mutation_id == "shared_embed_copied":
        src = tensor_by_id["tensor:entry:embed_w"]
        copy = Tensor(name="entry:embed_w_copy", dtype=src.dtype,
                      axes=tuple(src.axes),
                      logical_layout=tuple(src.logical_layout),
                      physical_layout=tuple(src.physical_layout))
        tensors.append(copy)
        nodes.append(Node(stable_id="entry.embed_w_copy",
                          op_type="embedding_weight", granularity="parameter_op",
                          output_ports=(Port("y", "output", copy.stable_id),),
                          evidence_refs=("ev_ep_shared_embed",)))
        replace_edge("e_ep_embedw_embed", src_node_id="entry.embed_w_copy",
                     tensor_id=copy.stable_id)
        replace_port("entry.embed_lookup", "w", copy.stable_id)
    elif mutation_id == "cf_float_proj_deleted":
        edges[:] = [e for e in edges if e.stable_id != "e_cf_float_proj"]
    elif mutation_id == "cf_float_proj_control":
        replace_edge("e_cf_float_proj", kind="control")
    elif mutation_id == "cf_float_proj_endpoints_swapped":
        # 引用合法的错接重排：float 链直进 squeeze（绕过 proj），proj 输出
        # 回环自身输入——e_cf_float_proj 端点不再 to_float.y→proj.x，
        # D11 端点核验检出
        replace_edge("e_cf_float_proj",
                     dst_node_id="cf.squeeze", dst_port="x")
        replace_edge("e_cf_proj_squeeze",
                     dst_node_id="cf.proj", dst_port="x")
        replace_port("cf.squeeze", "x", "tensor:cf:concat_f32")
        replace_port("cf.proj", "x", "tensor:cf:proj_out")
    elif mutation_id == "cf_float_proj_tensor_swapped":
        # 同轴同步篡改：边与两端端口三方一致地改接 bf16 concat（引用合法、
        # 无 V3 三方不一致），fp32 显式升链断开由 D11 张量核验检出
        replace_edge("e_cf_float_proj", tensor_id="tensor:cf:concat")
        replace_port("cf.to_float", "y", "tensor:cf:concat")
        replace_port("cf.proj", "x", "tensor:cf:concat")
    elif mutation_id == "head_gate_sinkhorn":
        for idx, n in enumerate(nodes):
            if n.stable_id == "exit.hc_head.gate":
                nodes[idx] = Node(
                    stable_id=n.stable_id, op_type="sinkhorn_split",
                    granularity=n.granularity, input_ports=n.input_ports,
                    output_ports=n.output_ports,
                    evidence_refs=tuple(n.evidence_refs))
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 节点不存在 exit.hc_head.gate")
    elif mutation_id == "final_logits_bf16":
        for idx, t in enumerate(tensors):
            if t.stable_id == "tensor:mk:logits":
                tensors[idx] = Tensor(
                    name=t.name, dtype="bf16", axes=tuple(t.axes),
                    logical_layout=t.logical_layout,
                    physical_layout=t.physical_layout,
                    global_rank=t.global_rank, local_rank=t.local_rank)
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 张量不存在 tensor:mk:logits")
    elif mutation_id == "middle_stage_bypass_stage0":
        # 与 codex_b4_review01/mutant_middle_stage_bypass_stage0.json 逐
        # 字段等价：mtp1 decode 输入改接 entry.stage_feed_0.h_out，目标
        # 端口张量同步（引用合法）；stage0 真实 h 输出被跳过
        replace_edge("mtp1.e_st_hcin_hcpre",
                     src_node_id="entry.stage_feed_0", src_port="h_out",
                     tensor_id="tensor:entry:entry_h")
        replace_port("mtp1.stage.hc_pre_attn.flatten", "x",
                     "tensor:entry:entry_h")
    elif mutation_id == "stage2_chain_to_entry":
        replace_edge("mtp2.e_st_hcin_hcpre",
                     src_node_id="mtp0.stage.hc_post_ffn.combine",
                     tensor_id="tensor:mtp0.stage:x_hc_out")
        replace_port("mtp2.stage.hc_pre_attn.flatten", "x",
                     "tensor:mtp0.stage:x_hc_out")
    elif mutation_id == "stage1_prefill_to_entry":
        replace_edge("mtp1.e_st_prefill_in",
                     src_node_id="entry.stage_feed_0", src_port="h_out",
                     tensor_id="tensor:entry:entry_h")
        replace_port("mtp1.stage.prefill_passthrough", "x",
                     "tensor:entry:entry_h")
    elif mutation_id == "head_gate_op_swap":
        for idx, n in enumerate(nodes):
            if n.stable_id == "exit.norm":
                nodes[idx] = Node(
                    stable_id=n.stable_id, op_type="passthrough",
                    granularity=n.granularity, input_ports=n.input_ports,
                    output_ports=n.output_ports,
                    evidence_refs=tuple(n.evidence_refs))
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 节点不存在 exit.norm")
    elif mutation_id == "stack_embed_to_single":
        for idx, n in enumerate(nodes):
            if n.stable_id == "mk.stack_embed":
                nodes[idx] = Node(
                    stable_id=n.stable_id, op_type="concat",
                    granularity=n.granularity, input_ports=n.input_ports,
                    output_ports=n.output_ports,
                    evidence_refs=tuple(n.evidence_refs))
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 节点不存在 mk.stack_embed")
    elif mutation_id == "exit_head_op_removed":
        for idx, n in enumerate(nodes):
            if n.stable_id == "exit.head":
                nodes[idx] = Node(
                    stable_id=n.stable_id, op_type="sinkhorn_split",
                    granularity=n.granularity, input_ports=n.input_ports,
                    output_ports=n.output_ports,
                    evidence_refs=tuple(n.evidence_refs))
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 节点不存在 exit.head")
    elif mutation_id in ("e_cf_concat_float_deleted",
                         "e_cf_proj_squeeze_deleted",
                         "e_ex_mix_gate_deleted",
                         "e_mk_s2_add_sample_deleted"):
        eid = mutation_id[:-len("_deleted")]
        if not any(e.stable_id == eid for e in edges):
            raise ExtractionError(f"mutation {mutation_id}: 边不存在 {eid}")
        edges[:] = [e for e in edges if e.stable_id != eid]
    elif mutation_id == "e_ex_mix_gate_to_control":
        replace_edge("e_ex_mix_gate", kind="control")
    elif mutation_id == "e_mk_s2_add_sample_unguarded":
        replace_edge("e_mk_s2_add_sample", guard_ref=None)
    elif mutation_id == "e_cf_proj_squeeze_rewired":
        # 双边轮换（引用合法的错接重排，各边自带张量落到同张量端口）：
        # proj 输出直进 confidence_out（绕过 squeeze）、squeeze 输出回环
        # 自身输入——两条必要链边端点偏离目录，D16 端点核验检出
        replace_edge("e_cf_proj_squeeze", dst_node_id="cf.confidence_out",
                     dst_port="x")
        replace_edge("e_cf_squeeze_out", dst_node_id="cf.squeeze",
                     dst_port="x")
        replace_port("cf.confidence_out", "x", "tensor:cf:proj_out")
        replace_port("cf.squeeze", "x", "tensor:cf:confidence")
    elif mutation_id == "e_ex_mix_gate_rewired":
        # 双边轮换：mixes 直进 reduce.pre（绕过 sigmoid 门控）、gate 输出
        # 回环自身输入——两条必要链边端点偏离目录，D16 端点核验检出
        replace_edge("e_ex_mix_gate", dst_node_id="exit.hc_head.reduce",
                     dst_port="pre")
        replace_edge("e_ex_gate_reduce", dst_node_id="exit.hc_head.gate",
                     dst_port="x")
        replace_port("exit.hc_head.reduce", "pre", "tensor:exit:hc_mixes")
        replace_port("exit.hc_head.gate", "x", "tensor:exit:hc_gate")
    elif mutation_id == "e_ex_scale_gate_tensor_swapped":
        replace_edge("e_ex_scale_gate", tensor_id="tensor:exit:hc_head_base_w")
        replace_port("exit.hc_head.gate", "scale",
                     "tensor:exit:hc_head_base_w")
        replace_port("exit.hc_head_scale_w", "y",
                     "tensor:exit:hc_head_base_w")
    elif mutation_id == "mk_w2_param_copied":
        src = tensor_by_id["tensor:mk:w2_w"]
        copy = Tensor(name="mk:w2_w_copy", dtype=src.dtype,
                      axes=tuple(src.axes),
                      logical_layout=tuple(src.logical_layout),
                      physical_layout=tuple(src.physical_layout))
        tensors.append(copy)
        nodes.append(Node(stable_id="mk.w2_param_copy",
                          op_type="markov_w2_param", granularity="parameter_op",
                          output_ports=(Port("y", "output", copy.stable_id),),
                          evidence_refs=("ev_mk_w2_param",)))
        replace_edge("e_mk_s4_w2_bias", src_node_id="mk.w2_param_copy",
                     tensor_id=copy.stable_id)
        replace_port("mk.step4.bias", "w", copy.stable_id)
    elif mutation_id == "mk_step0_logits_updated_bf16":
        for idx, t in enumerate(tensors):
            if t.stable_id == "tensor:mk:step0_logits_updated":
                tensors[idx] = Tensor(
                    name=t.name, dtype="bf16", axes=tuple(t.axes),
                    logical_layout=t.logical_layout,
                    physical_layout=t.physical_layout,
                    global_rank=t.global_rank, local_rank=t.local_rank)
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 张量不存在 "
                f"tensor:mk:step0_logits_updated")
    elif mutation_id == "confidence_out_axes_swapped":
        for idx, t in enumerate(tensors):
            if t.stable_id == "tensor:cf:confidence":
                axes = tuple(reversed(tuple(t.axes)))
                tensors[idx] = Tensor(
                    name=t.name, dtype=t.dtype, axes=axes,
                    logical_layout=tuple(reversed(t.logical_layout)),
                    physical_layout=tuple(reversed(t.physical_layout)),
                    global_rank=t.global_rank, local_rank=t.local_rank)
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 张量不存在 tensor:cf:confidence")
    elif mutation_id in ("e_ep_cat_proj_deleted",
                         "mtp0_e_st_hcpre_a_out_deleted",
                         "mtp0_e_ds_cache_state_deleted"):
        eid = {"e_ep_cat_proj_deleted": "e_ep_cat_proj",
               "mtp0_e_st_hcpre_a_out_deleted": "mtp0.e_st_hcpre_a_out",
               "mtp0_e_ds_cache_state_deleted":
                   "mtp0.e_ds_cache_state"}[mutation_id]
        if not any(e.stable_id == eid for e in edges):
            raise ExtractionError(f"mutation {mutation_id}: 边不存在 {eid}")
        edges[:] = [e for e in edges if e.stable_id != eid]
    elif mutation_id == "mk_step2_sample_argmax":
        # 边/张量/端口全部不变，仅算子身份偷换（codex_b4_review03
        # mutant_sample_argmax 等价）——引用合法，D15 目录检出
        for idx, n in enumerate(nodes):
            if n.stable_id == "mk.step2.sample":
                nodes[idx] = Node(
                    stable_id=n.stable_id, op_type="argmax",
                    granularity=n.granularity, input_ports=n.input_ports,
                    output_ports=n.output_ports,
                    evidence_refs=tuple(n.evidence_refs))
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 节点不存在 mk.step2.sample")
    elif mutation_id == "mtp0_cache_transition_reversed":
        # stage0 cache 迁移方向反转（codex_b4_review03
        # mutant_cache_transition_reversed 等价）——版本引用仍存在、
        # 仍属同一 state_id（引用合法），方向由 D17 有向核验拒绝
        replace_transition("tr_dspark_cache_mtp0")
    elif mutation_id in ("e_ep_norm_feed1_to_control",
                         "mtp0_e_st_dispatch_gate_to_data",
                         "mtp0_e_ds_idx_select_to_data"):
        eid = {"e_ep_norm_feed1_to_control": "e_ep_norm_feed1",
               "mtp0_e_st_dispatch_gate_to_data": "mtp0.e_st_dispatch_gate",
               "mtp0_e_ds_idx_select_to_data":
                   "mtp0.e_ds_idx_select"}[mutation_id]
        replace_edge(eid, kind="control" if mutation_id.endswith(
            "to_control") else "data")
    elif mutation_id in ("mtp1_e_st_hcpre_a_out_unguarded",
                         "mtp2_e_st_cachew_dec_unguarded"):
        eid = {"mtp1_e_st_hcpre_a_out_unguarded": "mtp1.e_st_hcpre_a_out",
               "mtp2_e_st_cachew_dec_unguarded":
                   "mtp2.e_ds_cachew_dec"}[mutation_id]
        replace_edge(eid, guard_ref=None)
    elif mutation_id in ("mtp0_e_ds_cachew_inwin_guard_swapped",
                         "mtp0_e_st_prefill_in_guard_swapped"):
        eid = {"mtp0_e_ds_cachew_inwin_guard_swapped":
                   "mtp0.e_ds_cachew_inwin",
               "mtp0_e_st_prefill_in_guard_swapped":
                   "mtp0.e_st_prefill_in"}[mutation_id]
        replace_edge(eid, guard_ref="g_decode")
    elif mutation_id == "mtp1_e_ds_cache_state_transition_ref_dangling":
        # 跨实例状态引用错位：stage1 cache 写后读引用 stage0 迁移
        # （引用本身存在，但语义错位——state_id 不一致属通用 V4 检查，
        # 这里注入保持引用合法的等价错位：同实例迁移的边引用改指另一
        # 条已定义迁移必须同 state_id，故用 D18 目录核验引用错位）
        for idx, e in enumerate(edges):
            if e.stable_id == "mtp1.e_ds_cache_state":
                edges[idx] = Edge(
                    stable_id=e.stable_id,
                    src_node_id=e.src_node_id, src_port=e.src_port,
                    dst_node_id=e.dst_node_id, dst_port=e.dst_port,
                    tensor_id=e.tensor_id, kind=e.kind,
                    evidence_refs=tuple(e.evidence_refs),
                    guard_ref=e.guard_ref,
                    state_transition_ref="tr_dspark_cache_mtp0")
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 边不存在 mtp1.e_ds_cache_state")
    elif mutation_id == "e_ep_cat_proj_rewired":
        # 双边轮换（引用合法的错接重排，各边自带张量落到同张量端口）：
        # concat 输出直进 main_norm（绕过 proj）、proj 输出回环自身输入
        replace_edge("e_ep_cat_proj", dst_node_id="entry.main_norm",
                     dst_port="x")
        replace_edge("e_ep_proj_norm", dst_node_id="entry.main_proj",
                     dst_port="x")
        replace_port("entry.main_norm", "x", "tensor:entry:main_hidden")
        replace_port("entry.main_proj", "x", "tensor:entry:main_proj_y")
    elif mutation_id == "mtp0_e_st_norm_xsrc_rewired":
        # 双边轮换：attn_norm 输出直进 wq_a（绕过源节点消费端口）、
        # 源节点输出回环自身输入
        replace_edge("mtp0.e_st_norm_xsrc",
                     dst_node_id="mtp0.dspark.wq_a", dst_port="x")
        replace_edge("mtp0.e_ds_xin_wqa",
                     dst_node_id="mtp0.dspark.x_draft_single_src",
                     dst_port="x")
        replace_port("mtp0.dspark.wq_a", "x",
                     "tensor:mtp0.stage:x_draft_single")
        replace_port("mtp0.dspark.x_draft_single_src", "x",
                     "tensor:mtp0.stage:x_draft_single")
    elif mutation_id == "mtp1_stage_x_hc_attn_flat_bf16":
        for idx, t in enumerate(tensors):
            if t.stable_id == "tensor:mtp1.stage:x_hc_attn_flat":
                tensors[idx] = Tensor(
                    name=t.name, dtype="bf16", axes=tuple(t.axes),
                    logical_layout=t.logical_layout,
                    physical_layout=t.physical_layout,
                    global_rank=t.global_rank, local_rank=t.local_rank)
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 张量不存在 "
                f"tensor:mtp1.stage:x_hc_attn_flat")
    elif mutation_id == "entry_main_proj_w_dtype_swapped":
        for idx, t in enumerate(tensors):
            if t.stable_id == "tensor:entry:main_proj_w":
                tensors[idx] = Tensor(
                    name=t.name, dtype="bf16", axes=tuple(t.axes),
                    logical_layout=t.logical_layout,
                    physical_layout=t.physical_layout,
                    global_rank=t.global_rank, local_rank=t.local_rank)
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: 张量不存在 "
                f"tensor:entry:main_proj_w")
    elif mutation_id in ("mk_ids_s2_transition_reversed",
                         "mk_logits_s0_transition_reversed"):
        # Markov 版本迁移方向反转（版本引用仍存在且同 state_id，引用
        # 合法；有向推进语义由 D17 方向核验拒绝）
        replace_transition(
            {"mk_ids_s2_transition_reversed": "tr_mk_ids_s2",
             "mk_logits_s0_transition_reversed":
                 "tr_mk_logits_s0"}[mutation_id])
    elif mutation_id == "synced_vocab_axis_129281":
        # codex_b4_review04 mutant_synced_vocab_axis 等价：named_axes
        # 与全部张量内嵌 V 轴值同步 129280→129281（symbol/引用合法）
        def bump_v(a: NamedAxis) -> NamedAxis:
            return replace(a, value=129281) if a.symbol == "V" else a
        named_axes = [bump_v(a) for a in graph.named_axes]
        tensors = [replace(t, axes=tuple(bump_v(a) for a in t.axes))
                   for t in tensors]
    elif mutation_id == "seed_rank2_borrowed_axis":
        # codex_b4_review04 mutant_seed_rank2 等价：entry seed 轴
        # [B]→[B,C1_exit]（借用已有轴、logical/physical 同步、引用合法）
        c1e = next(a for a in graph.named_axes if a.symbol == "C1_exit")
        tensors = [
            replace(t, axes=(*t.axes, c1e),
                    logical_layout=(*t.logical_layout, "C1_exit"),
                    physical_layout=(*t.physical_layout, "C1_exit"))
            if t.stable_id == "tensor:entry:seed_ids" else t
            for t in tensors]
    elif mutation_id == "g_decode_predicate_includes_prefill":
        # codex_b4_review04 mutant_decode_guard_includes_prefill 等价：
        # g_decode 谓词 start_pos > 0 → start_pos >= 0（guard_ref ID 全
        # 保持；prefill start_pos=0 误启整条草稿出口）
        guards = list(graph.guards)
        for idx, gd in enumerate(guards):
            if gd.stable_id == "g_decode":
                guards[idx] = Guard(
                    stable_id=gd.stable_id,
                    predicate_expr="start_pos >= 0",
                    evidence_refs=tuple(gd.evidence_refs))
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: guard 不存在 g_decode")
    elif mutation_id == "rm_axis_sync_257":
        def bump_rm(a: NamedAxis) -> NamedAxis:
            return replace(a, value=257) if a.symbol == "Rm" else a
        named_axes = [bump_rm(a) for a in graph.named_axes]
        tensors = [replace(t, axes=tuple(bump_rm(a) for a in t.axes))
                   for t in tensors]
    elif mutation_id == "n_mix_expression_drift":
        def drift_nmix(a: NamedAxis) -> NamedAxis:
            # 词法漂移（值不变：24==24）；named_axes 与张量内嵌轴定义
            # 同步改（V2 同定义规则），D19 canonical/公式交叉核出
            return replace(a, expression="N_hc * H") if a.symbol == "N_mix" \
                else a
        named_axes = [drift_nmix(a) for a in named_axes]
        tensors = [replace(t, axes=tuple(drift_nmix(a) for a in t.axes))
                   for t in tensors]
    elif mutation_id == "ne_axis_collapsed_zero":
        def collapse_ne(a: NamedAxis) -> NamedAxis:
            return replace(a, value=0, unknown_reason=None) \
                if a.symbol == "N_e" else a
        named_axes = [collapse_ne(a) for a in named_axes]
        tensors = [replace(t, axes=tuple(collapse_ne(a) for a in t.axes))
                   for t in tensors]
    elif mutation_id == "cf_proj_w_layout_transposed":
        tensors = [
            replace(t, physical_layout=tuple(reversed(t.physical_layout)))
            if t.stable_id == "tensor:cf:proj_w" else t
            for t in tensors]
    elif mutation_id == "mtp1_win_cache_axes_drift":
        tensors = [
            replace(t, axes=(t.axes[0], t.axes[2], t.axes[1]),
                    logical_layout=(t.logical_layout[0],
                                    t.logical_layout[2],
                                    t.logical_layout[1]),
                    physical_layout=(t.physical_layout[0],
                                     t.physical_layout[2],
                                     t.physical_layout[1]))
            if t.stable_id == "tensor:mtp1.stage:win_cache" else t
            for t in tensors]
    elif mutation_id == "mk_step3_written_axes_bv":
        tensors = [
            replace(t, axes=(*t.axes, next(
                a for a in graph.named_axes if a.symbol == "V")),
                    logical_layout=(*t.logical_layout, "V"),
                    physical_layout=(*t.physical_layout, "V"))
            if t.stable_id == "tensor:mk:step3_written" else t
            for t in tensors]
    elif mutation_id == "seed_axis_dtype_retyped":
        tensors = [replace(t, dtype="int32")
                   if t.stable_id == "tensor:entry:seed_ids" else t
                   for t in tensors]
    elif mutation_id == "phantom_axis_added":
        named_axes = [*graph.named_axes,
                      NamedAxis(symbol="X", name="phantom",
                                meaning="注入轴", origin="mutation",
                                evidence_ref="mutation", value=7)]
    elif mutation_id == "phantom_tensor_added":
        tensors = [*tensors,
                   Tensor(name="entry:phantom", dtype="bf16",
                          axes=(next(a for a in graph.named_axes
                                     if a.symbol == "H"),),
                          logical_layout=("H",), physical_layout=("H",))]
    elif mutation_id == "window_guard_threshold_shifted":
        guards = list(graph.guards)
        for idx, gd in enumerate(guards):
            if gd.stable_id == "g_window_wrap":
                guards[idx] = Guard(
                    stable_id=gd.stable_id,
                    predicate_expr="start_pos == 0 and T >= W",
                    evidence_refs=tuple(gd.evidence_refs))
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: guard 不存在 g_window_wrap")
    elif mutation_id == "inwin_wrap_swapped":
        guards = list(graph.guards)
        exprs = {gd.stable_id: gd.predicate_expr for gd in guards}
        for idx, gd in enumerate(guards):
            if gd.stable_id == "g_dspark_prefill_inwin":
                guards[idx] = Guard(
                    stable_id=gd.stable_id,
                    predicate_expr=exprs["g_window_wrap"],
                    evidence_refs=tuple(gd.evidence_refs))
            elif gd.stable_id == "g_window_wrap":
                guards[idx] = Guard(
                    stable_id=gd.stable_id,
                    predicate_expr=exprs["g_dspark_prefill_inwin"],
                    evidence_refs=tuple(gd.evidence_refs))
    elif mutation_id == "g_prefill_widened_neq":
        guards = list(graph.guards)
        for idx, gd in enumerate(guards):
            if gd.stable_id == "g_prefill":
                guards[idx] = Guard(
                    stable_id=gd.stable_id,
                    predicate_expr="start_pos <= 0",
                    evidence_refs=tuple(gd.evidence_refs))
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: guard 不存在 g_prefill")
    elif mutation_id == "g_decode_evidence_dropped":
        guards = list(graph.guards)
        for idx, gd in enumerate(guards):
            if gd.stable_id == "g_decode":
                guards[idx] = Guard(
                    stable_id=gd.stable_id,
                    predicate_expr=gd.predicate_expr,
                    evidence_refs=("ev_ds_decode",))
                break
        else:
            raise ExtractionError(
                f"mutation {mutation_id}: guard 不存在 g_decode")
    elif mutation_id == "vocab_source_claims_hidden":
        # codex_b4_review05 mutant_vocab_source_claims_hidden 等价：V
        # 来源四字段借用真实 H 轴（named + 张量内嵌同步；数值/符号不变）
        def v_claims_hidden(a: NamedAxis) -> NamedAxis:
            if a.symbol != "V":
                return a
            h = next(x for x in graph.named_axes if x.symbol == "H")
            return replace(a, name=h.name, meaning=h.meaning,
                           origin=h.origin, evidence_ref=h.evidence_ref)
        named_axes = [v_claims_hidden(a) for a in named_axes]
        tensors = [replace(t, axes=tuple(v_claims_hidden(a) for a in t.axes))
                   for t in tensors]
    elif mutation_id == "rm_source_misattributed":
        def rm_claims_d(a: NamedAxis) -> NamedAxis:
            if a.symbol != "Rm":
                return a
            d = next(x for x in graph.named_axes if x.symbol == "D")
            return replace(a, name=d.name, meaning=d.meaning,
                           origin=d.origin, evidence_ref=d.evidence_ref)
        named_axes = [rm_claims_d(a) for a in named_axes]
        tensors = [replace(t, axes=tuple(rm_claims_d(a) for a in t.axes))
                   for t in tensors]
    elif mutation_id == "h_plus_rm_meaning_drifted":
        def hprm_drift(a: NamedAxis) -> NamedAxis:
            if a.symbol != "H_plus_Rm":
                return a
            return replace(a, meaning="hidden 拼接宽度", origin="model.py:1",
                           evidence_ref="model.py:1")
        named_axes = [hprm_drift(a) for a in named_axes]
        tensors = [replace(t, axes=tuple(hprm_drift(a) for a in t.axes))
                   for t in tensors]
    elif mutation_id == "ne_unknown_reason_misleading":
        def ne_reason_bad(a: NamedAxis) -> NamedAxis:
            return replace(a, unknown_reason="已知为 0（静态可求值）") \
                if a.symbol == "N_e" else a
        named_axes = [ne_reason_bad(a) for a in named_axes]
        tensors = [replace(t, axes=tuple(ne_reason_bad(a) for a in t.axes))
                   for t in tensors]
    elif mutation_id == "n_draft_plus_1_name_drifted":
        def ndp1_drift(a: NamedAxis) -> NamedAxis:
            return replace(a, name="N_draft_plus_one") \
                if a.symbol == "N_draft_plus_1" else a
        named_axes = [ndp1_drift(a) for a in named_axes]
        tensors = [replace(t, axes=tuple(ndp1_drift(a) for a in t.axes))
                   for t in tensors]
    elif mutation_id == "seed_rank_metadata99":
        # codex_b4_review05 mutant_seed_rank_metadata99 等价：seed axes/
        # 双 layout 不变，仅可选秩标注两字段均改 99（与 rank1 矛盾）
        tensors = [
            replace(t, global_rank=99, local_rank=99)
            if t.stable_id == "tensor:entry:seed_ids" else t
            for t in tensors]
    elif mutation_id == "cf_proj_w_rank_metadata_borrowed":
        # 两字段分别核对：global_rank=2（恰与实际 rank 相同，不得放行
        # 另一字段的矛盾）+ local_rank=99
        tensors = [
            replace(t, global_rank=2, local_rank=99)
            if t.stable_id == "tensor:cf:proj_w" else t
            for t in tensors]
    elif mutation_id == "mk_step2_rank_metadata99":
        tensors = [
            replace(t, global_rank=99, local_rank=99)
            if t.stable_id == "tensor:mk:step2_embed" else t
            for t in tensors]
    elif mutation_id == "entry_logits_rank_metadata0":
        # None 改写为 0 的等价负例：exit:logits 实际 rank3，标注 0 矛盾
        tensors = [
            replace(t, global_rank=0, local_rank=0)
            if t.stable_id == "tensor:exit:logits" else t
            for t in tensors]
    else:
        raise ExtractionError(f"未知 DSpark 集成 mutation: {mutation_id}")

    mutated = SemanticGraph(
        template_id=graph.template_id, named_axes=tuple(named_axes),
        tensors=tensors, nodes=nodes, edges=edges,
        state_versions=graph.state_versions,
        state_transitions=tuple(transitions),
        guards=tuple(guards), guard_activations=(),
        alias_relations=graph.alias_relations,
        fold_groups=graph.fold_groups)
    ref_issues = validate_graph_references(mutated)
    if ref_issues:
        raise ExtractionError(
            f"mutation {mutation_id} 注入后引用不合法（受控错误必须引用合法）: "
            + "; ".join(f"[{x.code}] {x.path}" for x in ref_issues[:3]))
    return mutated


def build_dspark_validation_report(graph: SemanticGraph) -> dict:
    """build 期实际运行全部集成受控错误并据实记录（D4 口径）。"""
    specs = dspark_integration_error_specs()
    results = []
    all_detected = True
    for spec in specs:
        try:
            mutated = apply_dspark_mutation(graph, spec.mutation_id)
            found = {i.code for i in
                     validate_dspark_integration_contracts(mutated)}
        except Exception as exc:
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
    original = validate_dspark_integration_contracts(graph)
    return {
        "schema": "dspark_integration_validation_v1",
        "template_applies": graph.template_id == _INTEGRATION_TEMPLATE,
        "original_contracts": {
            "issue_count": len(original),
            "status": "pass" if not original else "fail",
            "issues": [{"code": i.code, "path": i.path, "message": i.message}
                       for i in original[:20]],
        },
        "controlled_errors": {
            "catalog_count": len(specs),
            "all_detected": all_detected,
            "results": results,
        },
        "notes": ["静态结构检查；不声称数值等价或运行验证",
                  "受控错误为 build 期对真实集成图实际注入并重新校验的结果"
                  "（逐项 detected），不写死通过"],
    }
