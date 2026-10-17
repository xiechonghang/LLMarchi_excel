"""DeepSeek-V4-Flash-0731 DSparkAttention 真实子图提取器（工作包 B · 步骤 B1）。

数据来源：models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/{model.py,kernel.py}
与 inference/config.json（固定 SHA256 基线，:mod:`structure_reader.extract.evidence`
逐条核对原文）。固定参考函数：model.py:743-792（get_dspark_topk_idxs 与
DSparkAttention.forward），静态取证，不导入/不执行模型代码。

B1 范围（整体 B 仍 in_progress，本模块不是完整 DSpark 交付）：
只构建 DSparkAttention（MTP draft stage 的注意力，model.py:750-752 继承
Attention、stage 层 compress_ratio==0）的注意力子图。公共 Block/hc_pre/
hc_post/mHC、三 stage 首末入口、Markov/Confidence 头、全量集成由后续
步骤续接，本模块不声称覆盖。

B2 修正（本模块）：位置/槽位坐标表示——删除把坐标标量伪装成 bf16 数据
张量 view 的两条 AliasRelation（int64 坐标不会从 bf16 cache/q 内存别名而
来），改为 bundle 级 CoordinateExpression 显式记录（可求值原文 + 图内真实
消费端口 + 证据，明确为坐标值而非 shape 轴/存储关系）与公开 JSON 往返
（bundle_to_json/bundle_from_json）；真实存储别名仅保留 kv_cache[:bsz]
batch view。公共 stage/三实例绑定/链路图在本包 dspark_stage 模块（B2 新增）。

图覆盖（对照 model.py:743-792 逐条，行号以证据条目锚定）：
- 前提：assert self.compress_ratio == 0（753）→ 窗口 cache 容量 = W
  （479 条件式 else 分支），无压缩区、无 compressor/indexer（DSpark 无
  压缩路径；压缩相关轴不出现在本图）。
- main 路径（main_x [B,T,H]，两相位同源）：共享单投影 KV
  wkv→kv_norm（759）→ 尾 rd 维部分 RoPE（760）→ 非 RoPE 维 FP8 量化
  （761，block 64，in-place）。
- prefill（start_pos==0，763-769）：仅 cache 构建——T<=W 顺序写（765）、
  T>W 环形回绕写（767-768），随后 return x 原样（769）；无 Q、无索引、
  无 sparse_attn。DSparkBlock.forward prefill 分支只调 attn 不走 Block
  双 hc 前向（845-849）。x 参数在 prefill 是 hc 展开流 [B,N_draft,N_hc,H]
  （858-861 forward_embed），decode 是 hc_pre 后单流 [B,N_draft,H]——
  两相位形状不同是真实源码事实，图以两个输入张量分别记录。
- decode（start_pos>0，771-792）：草稿位置 offset——草稿 Q/KV 的 RoPE
  取 freqs_cis[start_pos+seqlen : start_pos+seqlen+block_size]（772），
  即草稿位置从 P+T 起、共 N_draft 个；草稿低秩 Q wq_a→q_norm→wq_b→
  unflatten→手写 RMS 缩放→部分 RoPE（774-777）；草稿 KV 复用同一
  self.wkv/self.kv_norm 实例（778，与 759 同一调用目标）→ RoPE →
  非 RoPE 维 FP8（780）；选择索引 get_dspark_topk_idxs（782，743-747）：
  宽 min(W, P+1)+N_draft、无 -1 无效填充（与 get_window_topk_idxs 的
  -1 填充语义不同）、仅 decode（745 assert start_pos > 0）、按
  (win,bsz,block_size,start_pos) lru_cache——宽度随 P 变化非固定值；
  主 cache 槽 start_pos % W 环形更新（783）；窗口 cache 与草稿 KV
  concat（784，物理宽 W+N_draft）；sparse_attn（785，不透明 kernel
  边界）→ inverse RoPE（786，conj）→ reshape G 组/分组低秩 einsum/
  flatten/wo_b（788-791）。
- 共享参数：wkv/kv_norm 在 main 与 draft 是同一实例两次调用——图以
  共享权重张量节点（dspark.wkv_w/dspark.kv_norm_w）扇出两条边到两个
  调用点表达；单调用点的 wq_a/q_norm/wq_b/wo_b 沿用参考图约定
  （linear/rms_norm 节点即参数算子，不展开权重张量）。
- 状态：窗口 cache 单一 state_id（v_hist 历史累积 → v_after 本调用
  写入后）；历史连续调用前提，内容 unknown，不当作空。
- 相位链（Node 无 guard 字段，用既有 Edge guard 表达）：草稿 Q/KV 内部
  链、索引、cache 写后读 state 边、concat、sparse_attn 与 O 链的整个
  decode 运行时链全部挂 g_decode——prefill 在 769 早退，整条草稿链
  不可达（以测试对序列化图逐边求值验证，不靠声称）；图输入经显式源
  节点消费：x_draft_hc→prefill_return 消费边挂 g_prefill（prefill
  passthrough 只能 prefill），x_draft_single→草稿 Q/KV 调用点挂
  g_decode，x_main_in→wkv_main 两相位共享不挂；共享参数节点静态身份
  两相位可见，但其到 decode-only 调用点的消费边（draft 扇出、sink→
  sparse_attn）挂 g_decode，到 main 调用点的边不挂——不把计算边留成
  prefill 运行。g_window_wrap 谓词在本图限定 prefill
  （start_pos == 0 and T > W）——decode 主 seqlen=1 是实际调用前提，
  非法长 decode 不激活 prefill 回绕分支；与 rules GUARD_DEFS 的
  g_window_wrap（T > W）相差相位限定，rules 只读不改。
- 位置/缓存槽位坐标（具名标量，零轴张量，不充当张量维度大小）：
  decode 本次写槽 P % W（783）、草稿 RoPE 位置起点 P + T（772）——
  以标量张量 + bundle 级 CoordinateExpression（可求值原文 + 真实消费
  端口，见 bundle.coordinate_expressions）+ 消费边到 cache_update.slot
  与 q_rope/kv_rope/o_rope_inv.pos 端口表达；坐标是 int64 整数值，
  不是 bf16 数据张量的 view/别名（B2 修正，无此类虚假 AliasRelation）；
  下一次写槽 (P+T)%W 属下一调用前提，不混入。
- 符号轴/未知数诚实：B/T/P 为场景表达式轴；选择索引物理宽与 KV concat
  宽是表达式轴（随相位/场景求值），不固定值；prefill 不存在索引
  （769 早于 782），索引/注意力张量的消费边全部挂 g_decode。
- world_size=1 分析假设（graph.world_size=1；n_local_heads=Hq）。

新增模块给出简明可调用 build_dspark_attention_graph 入口，返回
DSparkAttentionBundle(graph, registry, values, coordinate_expressions)；
graph 保证 validate_graph_references 零问题、canonical JSON 往返一致
（测试覆盖）；bundle 公开 JSON 往返（bundle_to_json/bundle_from_json）
携带坐标表达式原文，篡改可从往返结果读出检出（测试覆盖）。

失败异常接口（以实际接口为准）：证据核对失败抛
structure_reader.extract.evidence.EvidenceError（ValueError 子类）；
图构造错误、validate_graph_references 未通过或 bundle JSON 结构不合法
抛 structure_reader.extract.deepseek_v4_reference.ExtractionError
（ValueError 子类）。两类均具名拒绝、失败不给图/不给 bundle。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.deepseek_v4_reference import (
    ExtractionError,
    _GraphBuilder,
)
from structure_reader.extract.evidence import (
    EvidenceBase,
    EvidenceRegistry,
    SourceEvidence,
    load_evidence_base,
)
from structure_reader.graph_document import (
    SemanticGraph,
    graph_from_json,
    graph_to_json,
)
from structure_reader.graph_ir import Guard, NamedAxis
from structure_reader.model_registry import find_workspace_root
from structure_reader.rules.deepseek_v4_flash_0731 import (
    ModelValues,
    load_model_values,
)
from structure_reader.scenario_evaluator import check_expression

__all__ = [
    "TEMPLATE_ID",
    "EXPR_SEL_WIDTH",
    "EXPR_KV_CONCAT_WIDTH",
    "EXPR_SLOT_WRITE",
    "EXPR_DRAFT_POS0",
    "CoordinateExpression",
    "DSparkAttentionBundle",
    "build_dspark_attention_graph",
    "bundle_to_json",
    "bundle_from_json",
]

TEMPLATE_ID = "deepseek_v4_flash_0731:dspark_attention"
# stage 模块复用本模板时的显式别名（同一模板 ID，不发明第二份）
ATTENTION_TEMPLATE_ID = TEMPLATE_ID

# DSpark 选择索引物理宽（get_dspark_topk_idxs，model.py:746）：
# 窗口段 min(W, P+1) + 草稿段 N_draft。仅 decode（P>0，745）；
# 无 -1 填充——所有槽位有效（与 get_window_topk_idxs 不同）。
EXPR_SEL_WIDTH = "min(W, P + 1) + N_draft"
# decode KV concat 物理宽（model.py:784）：窗口 cache W 槽 + 草稿 N_draft。
EXPR_KV_CONCAT_WIDTH = "W + N_draft"
# decode 本次写入槽（model.py:783）：kv_cache[:bsz, start_pos % win]。
# 具名标量坐标，场景求值用，不充当张量维度大小。
EXPR_SLOT_WRITE = "P % W"
# 草稿 RoPE 位置起点（model.py:772）：freqs_cis 从 start_pos+seqlen 起
# 共 N_draft 个（P+T .. P+T+N_draft-1）。下一次写槽 (P+T)%W 属下一调用
# 前提，不混入本图。
EXPR_DRAFT_POS0 = "P + T"


# ---------------------------------------------------------------------------
# 证据目录：每条 = qualified symbol + 行区间 + 原文片段 + 核对结论 + 等级
# 全部锚定 V4.0（DeepSeek-V4-Flash-0731）源码，注册时逐条对原文核对。
# ---------------------------------------------------------------------------

_E = SourceEvidence


def _evidence_defs() -> tuple[SourceEvidence, ...]:
    m = "model.py"
    k = "kernel.py"
    return (
        _E("ev_ds_class", m, "DSparkAttention", 750, 753,
           "class DSparkAttention(Attention):\n"
           "def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor):\n"
           "assert self.compress_ratio == 0",
           "DSparkAttention 继承 Attention（wq_a/q_norm/wq_b/wkv/kv_norm/wo_a/wo_b/"
           "attn_sink/窗口 cache 全部来自基类 __init__）；显式要求 compress_ratio==0"
           "（MTP stage 无压缩）", "located"),
        _E("ev_ds_main_kv", m, "DSparkAttention.forward main kv", 758, 761,
           "main_freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]\n"
           "main_kv = self.kv_norm(self.wkv(main_x))\n"
           "apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)\n"
           "act_quant(main_kv[..., :-rd], 64, scale_fmt, scale_dtype, True)",
           "main 路径：main_x 经共享单投影 wkv→kv_norm；main 位置 P..P+T-1；"
           "尾部 rd 维 RoPE；非 RoPE 维 FP8 量化（block 64，in-place）", "source_checked"),
        _E("ev_ds_prefill", m, "DSparkAttention.forward prefill", 763, 769,
           "if start_pos == 0:\n"
           "if seqlen <= win:\n"
           "self.kv_cache[:bsz, :seqlen] = main_kv\n"
           "cutoff = seqlen % win\n"
           "self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = main_kv[:, -win:].split([win - cutoff, cutoff], dim=1)\n"
           "return x",
           "prefill（start_pos==0）仅 cache 构建：T<=W 顺序写前 seqlen 槽、"
           "T>W 环形回绕写（cutoff=T%W），随后 return x 原样——无 Q/索引/"
           "sparse_attn（769 早于 774/782/785）", "source_checked"),
        _E("ev_ds_draft_pos", m, "DSparkAttention.forward draft positions", 771, 772,
           "bsz, block_size, _ = x.size()\n"
           "freqs_cis = self.freqs_cis[start_pos+seqlen:start_pos+seqlen+block_size]",
           "草稿形状 [B, N_draft, H]；草稿 Q/KV 的 RoPE 位置 = start_pos+seqlen 起"
           "共 block_size 个（P+T .. P+T+N_draft-1）——草稿位置 offset = P+T；"
           "位置起点即主 KV 覆盖 P..P+T-1 之后的下一位置", "located"),
        _E("ev_ds_draft_q", m, "DSparkAttention.forward draft q", 774, 777,
           "q = self.q_norm(self.wq_a(x))\n"
           "q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))\n"
           "q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)\n"
           "apply_rotary_emb(q[..., -rd:], freqs_cis)",
           "草稿低秩 Q：wq_a→q_norm→wq_b→unflatten→手写 RMS 缩放→尾部 rd 维 RoPE", "source_checked"),
        _E("ev_ds_draft_kv", m, "DSparkAttention.forward draft kv", 778, 780,
           "kv = self.kv_norm(self.wkv(x))\n"
           "apply_rotary_emb(kv[..., -rd:], freqs_cis)\n"
           "act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)",
           "草稿 KV 复用同一 self.wkv/self.kv_norm 实例（与 759 同一调用目标，"
           "非第二套参数）→ RoPE → 非 RoPE 维 FP8（block 64）", "source_checked"),
        _E("ev_ds_decode", m, "DSparkAttention.forward decode cache+cat", 782, 784,
           "topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos)\n"
           "self.kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)\n"
           "kv = torch.cat([self.kv_cache[:bsz], kv], dim=1)",
           "decode：选择索引（743-747）；主 cache 单槽 start_pos%W 环形更新"
           "（写后读）；窗口 cache 与草稿 KV concat（物理宽 W+N_draft）", "source_checked"),
        _E("ev_ds_sparse", m, "DSparkAttention.forward sparse_attn", 785, 786,
           "o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)\n"
           "apply_rotary_emb(o[..., -rd:], freqs_cis, True)",
           "sparse_attn 输入 (q, kv_concat, attn_sink, topk_idxs)；inverse RoPE"
           "（conj）在输出上、先于 O reshape/分组低秩", "located"),
        _E("ev_ds_slot_write", m, "DSparkAttention.forward slot write", 783, 783,
           "self.kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)",
           "decode 本次写入槽 = start_pos % W（P % W）：真实本次写槽坐标，"
           "非下一次写槽（(P+T)%W 属下一调用，772 起点之后）", "source_checked"),
        _E("ev_ds_ogroup", m, "DSparkAttention.forward o projection", 788, 791,
           "o = o.view(bsz, block_size, self.n_local_groups, -1)\n"
           "wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)\n"
           "o = torch.einsum(\"bsgd,grd->bsgr\", o, wo_a)\n"
           "x = self.wo_b(o.flatten(2))",
           "inverse RoPE 之后：reshape G 组（每组 D_g=Hq×D/G）、分组低秩 einsum"
           "（G×Ro）、展平 GRo 后 wo_b", "located"),
        _E("ev_ds_topk", m, "get_dspark_topk_idxs", 743, 747,
           "@lru_cache(1)\n"
           "def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int):\n"
           "assert start_pos > 0\n"
           "matrix = torch.cat([torch.arange(min(window_size, start_pos + 1)), window_size + torch.arange(block_size)])\n"
           "return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()",
           "DSpark 选择索引：宽 = min(W, P+1) + N_draft（窗口段 + 草稿段拼接）；"
           "无 -1 无效填充（所有槽位有效，与 get_window_topk_idxs 的 -1 填充"
           "不同）；仅 decode（assert start_pos > 0）；lru_cache 按"
           " (win,bsz,block_size,start_pos) 缓存——宽度随 P 变化，非固定值", "source_checked"),
        _E("ev_ds_block_prefill", m, "DSparkBlock.forward", 845, 849,
           "def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor, main_x: torch.Tensor) -> torch.Tensor:\n"
           "if start_pos > 0:\n"
           "return super().forward(x, start_pos, input_ids, main_x)\n"
           "# only compute KV cache in prefill stage\n"
           "return self.attn(x, start_pos, main_x)",
           "Block 级相位调度：prefill 只调 attn（cache 构建）不走 Block 双 hc "
           "前向；decode 走 Block.forward（hc_pre→attn_norm→attn→hc_post）；"
           "两相位 attn 的 x 参数形状不同（prefill hc 展开流 / decode 单流）", "located"),
        _E("ev_ds_attn_init", m, "Attention.__init__ (DSpark 基类)", 462, 469,
           "self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))\n"
           "self.wq_a = Linear(self.dim, self.q_lora_rank)\n"
           "self.q_norm = RMSNorm(self.q_lora_rank, self.eps)\n"
           "self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)\n"
           "self.wkv = Linear(self.dim, self.head_dim)\n"
           "self.kv_norm = RMSNorm(self.head_dim, self.eps)\n"
           "self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.bfloat16)\n"
           "self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)",
           "参数清单：attn_sink [Hq] fp32、低秩 Q（wq_a/q_norm/wq_b）、共享单投影"
           " KV（wkv/kv_norm）、分组低秩 O（wo_a/wo_b）", "located"),
        _E("ev_ds_rope", m, "apply_rotary_emb", 238, 243,
           "def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:\n"
           "if inverse:\n"
           "freqs_cis = freqs_cis.conj()",
           "RoPE 正/逆共用实现，inverse 用共轭；部分维度（尾部 rd）旋转", "located"),
        _E("ev_ds_sparse_kernel", k, "sparse_attn", 355, 357,
           "def sparse_attn(\n"
           "q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float\n"
           ") -> torch.Tensor:",
           "sparse_attn 接口签名已定位核对", "located",
           note="kernel.py sparse_attn_kernel（tilelang JIT）的 gather/在线 "
                "softmax/sink 数值实现未核对；本条只声称接口签名与调用边界定位"),
        _E("ev_ds_capacity", m, "kv_cache_size (r=0)", None, None, None,
           "MTP stage 层 compress_ratio==0 → kv_cache_size = W + 0 = W"
           "（窗口槽，无压缩区）", "derived",
           derivation="model.py:479 kv_cache_size = args.window_size + "
                      "(args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)"
                      " 的 else 分支；DSparkAttention assert compress_ratio==0（model.py:753）；"
                      "run config compress_ratios 尾部 3 项为 0（run_config.json:39）"),
        _E("ev_ds_kernel_numerics", k, "tilelang JIT kernels", None, None, None,
           "kernel.py 的 tilelang JIT kernel（act_quant/fp8_gemm/sparse_attn_kernel）"
           "数值实现", "uncovered",
           note="全部 tilelang kernel 的数值实现未核对；本图只在调用边界消费其"
                "接口语义，不声称 kernel 内部语义已验证（不透明 kernel 边界）"),
    )


# ---------------------------------------------------------------------------
# 轴：rules 轴字典同源子集 + DSpark 图专属新轴
# ---------------------------------------------------------------------------

def _axes(values: ModelValues) -> dict[str, NamedAxis]:
    """图轴集合。B/T/P 场景表达式轴与配置常量轴从 rules AXIS_DEFS 同源复制
    （避免第二份定义漂移）；N_qf 与 DSpark 两个新轴在本模块定义。无 r/
    N_cap 等压缩轴——DSparkAttention compress_ratio==0（753），无压缩路径。
    N_mix/N_hcH/hc_j/hc_k 块级 mHC 轴同源复制（stage 模板 Block 骨架共享，
    B1 图不含 hc 链但无害）。"""
    from structure_reader.rules.deepseek_v4_flash_0731 import AXIS_DEFS
    wanted = ("B", "T", "P", "N_hc", "H", "Hq", "D", "D_rope", "D_nope",
              "Rq", "Ro", "G", "W", "N_draft", "D_g", "GRo",
              "N_mix", "N_hcH", "hc_j", "hc_k", "N_e", "E", "K")
    axes: dict[str, NamedAxis] = {}
    for a in AXIS_DEFS:
        if a.symbol in wanted:
            axes[a.symbol] = NamedAxis(
                symbol=a.symbol, name=a.name, meaning=a.meaning,
                origin=a.origin, evidence_ref=a.evidence_ref,
                value=a.value, expression=a.expression,
                unknown_reason=a.unknown_reason,
            )
    missing = [s for s in wanted if s not in axes and s not in ("hc_j", "hc_k")]
    if missing:
        raise ExtractionError(f"rules 轴字典缺少 DSpark 需要的轴: {missing}")
    # hc_j/hc_k：与参考提取 _graph_axes 同源语义（rules 无静态定义，按
    # hc_mult 派生；model.py:692 comb 源/目标流下标）。evidence_ref 用
    # rules 派生轴惯例（源码行锚点），ev_ 引用由使用这些轴的节点/边携带。
    hc_v = values.hc_mult
    axes["hc_j"] = NamedAxis("hc_j", "hc_source_stream",
                             "mHC comb 源流下标 j", "推导 model.py:692",
                             "model.py:692", value=hc_v)
    axes["hc_k"] = NamedAxis("hc_k", "hc_target_stream",
                             "mHC comb 目标流下标 k", "推导 model.py:692",
                             "model.py:692", value=hc_v)
    axes["N_qf"] = NamedAxis(
        "N_qf", "q_flat_width", "wq_b 输出展平宽度 Hq×D",
        "推导 model.py:465", "ev_ds_draft_q", expression="Hq * D")
    axes["N_sel_ds"] = NamedAxis(
        "N_sel_ds", "dspark_select_width",
        "DSpark 选择索引物理宽（槽位数）：decode min(W, P+1)+N_draft"
        "（窗口段+草稿段拼接，model.py:746）；无 -1 填充、全部有效；"
        "仅 decode（745 assert start_pos>0），prefill 无索引（769 早于 782）",
        "推导 model.py:743-747", "ev_ds_topk", expression=EXPR_SEL_WIDTH)
    axes["N_kvcat"] = NamedAxis(
        "N_kvcat", "dspark_kv_concat_width",
        "decode KV concat 物理宽 = 窗口 cache W 槽 + 草稿 N_draft"
        "（model.py:784）",
        "推导 model.py:784", "ev_ds_decode",
        expression=EXPR_KV_CONCAT_WIDTH)
    return axes


# ---------------------------------------------------------------------------
# 图构建器
# ---------------------------------------------------------------------------

def _add_attn_tensors(b: _GraphBuilder, axes: dict[str, NamedAxis], *,
                      prefix: str, draft_in_role: str = "x_draft_single",
                      main_in_role: str = "x_main_in") -> dict[str, str]:
    """DSparkAttention 内部张量角色 → stable_id（B1/stage 公共模板同源）。

    prefix 是张量名前缀（B1 "dspark" / stage "stage"）；draft_in_role 允许
    调用方按语境命名 decode attn 输入张量（B1 是图输入 x_draft_single，
    stage 是 attn_norm 输出 x_attn_normed——同一算子输入角色）。形状轴按
    源码真实选择。返回 role（无前缀）→ stable_id。
    """
    t: dict[str, str] = {}

    def add(role: str, dtype: str, syms: tuple[str, ...],
            evidence: tuple[str, ...] = ()) -> str:
        tid = b.tensor(f"{prefix}:{role}", dtype, syms, evidence)
        t[role] = tid
        return tid

    # 输入（B1：图输入张量；stage：x_main_in 图输入 / x_draft_single 为
    # attn_norm 输出中间张量——消费关系由调用方建边表达）
    add(main_in_role, "bf16", ("B", "T", "H"), ())         # main_x [B,T,H]
    add(draft_in_role, "bf16", ("B", "N_draft", "H"), ())  # decode x
    # 共享权重（wkv/kv_norm 同一实例两次调用：main 759 / draft 778）
    # Linear 权重 dtype：config dtype="fp8" → Transformer default_dtype
    # fp8_e4m3fn（model.py:887）→ Linear weight fp8（129-160）；RMSNorm
    # weight fp32（model.py:191-193）。scale 张量不展开（claim 级注记）。
    add("wkv_w", "fp8_e4m3fn", ("D", "H"), ("ev_ds_attn_init",))
    add("kv_norm_w", "fp32", ("D",), ("ev_ds_attn_init",))
    # main 路径
    add("main_kv_proj", "bf16", ("B", "T", "D"), ("ev_ds_main_kv",))
    add("main_kv_normed", "bf16", ("B", "T", "D"), ("ev_ds_main_kv",))
    add("main_kv_quant", "bf16", ("B", "T", "D"), ("ev_ds_main_kv",))
    # decode 草稿路径
    add("qr", "bf16", ("B", "N_draft", "Rq"), ("ev_ds_draft_q",))
    add("q_flat", "bf16", ("B", "N_draft", "N_qf"), ("ev_ds_draft_q",))
    add("q", "bf16", ("B", "N_draft", "Hq", "D"), ("ev_ds_draft_q",))
    add("q_rope", "bf16", ("B", "N_draft", "Hq", "D"), ("ev_ds_draft_q",))
    add("draft_kv_proj", "bf16", ("B", "N_draft", "D"), ("ev_ds_draft_kv",))
    add("draft_kv_normed", "bf16", ("B", "N_draft", "D"), ("ev_ds_draft_kv",))
    add("draft_kv_quant", "bf16", ("B", "N_draft", "D"), ("ev_ds_draft_kv",))
    # cache 与选择索引
    add("win_cache", "bf16", ("B", "W", "D"), ("ev_ds_prefill", "ev_ds_capacity"))
    add("kv_win_read", "bf16", ("B", "W", "D"), ("ev_ds_decode",))
    add("topk_idxs", "int32", ("B", "N_draft", "N_sel_ds"), ("ev_ds_topk",))
    add("kv_concat", "bf16", ("B", "N_kvcat", "D"), ("ev_ds_decode",))
    add("attn_sink_param", "fp32", ("Hq",), ("ev_ds_attn_init",))
    # 具名标量坐标（零轴张量，值语义由 bundle 级 CoordinateExpression
    # 原文承载并公开 JSON 往返，不充当任何张量的维度大小；int64 坐标值
    # 不是任何 bf16 数据张量的存储 view）：
    add("slot_write", "int64", (), ("ev_ds_slot_write",))   # decode 写槽 P%W（783）
    add("draft_pos0", "int64", (), ("ev_ds_draft_pos",))    # 草稿 RoPE 起点 P+T（772）
    # 输出链
    add("o_rope", "bf16", ("B", "N_draft", "Hq", "D"), ("ev_ds_sparse",))
    add("o_grouped", "bf16", ("B", "N_draft", "G", "D_g"), ("ev_ds_ogroup",))
    add("o_lora", "bf16", ("B", "N_draft", "G", "Ro"), ("ev_ds_ogroup",))
    add("o_flat", "bf16", ("B", "N_draft", "GRo"), ("ev_ds_ogroup",))
    add("attn_out", "bf16", ("B", "N_draft", "H"), ("ev_ds_ogroup",))
    return t


def _add_tensors(b: _GraphBuilder, axes: dict[str, NamedAxis]) -> dict[str, str]:
    """B1 张量集合：公共 Attention 张量 + B1 特有 prefill hc 展开流输入。"""
    t = _add_attn_tensors(b, axes, prefix="dspark")
    # prefill 分支 attn 的 x 是 hc 展开流 [B, N_draft, N_hc, H]
    # （DSparkBlock.forward_embed unsqueeze+repeat，model.py:858-861）
    tid = b.tensor("dspark:x_draft_hc", "bf16", ("B", "N_draft", "N_hc", "H"), ())
    t["x_draft_hc"] = tid
    return t


def _add_attn_flow(b: _GraphBuilder, t: dict[str, str], *,
                   src_op: str = "graph_input",
                   include_prefill: bool = False,
                   fed_sources: bool = False,
                   main_in_role: str = "x_main_in",
                   draft_in_role: str = "x_draft_single") -> tuple[str, ...]:
    """DSparkAttention 内部流（B1 与 stage 公共模板同源，单一实现）。

    src_op：输入源节点的 op_type——B1 图中是图输入（"graph_input"）；
    stage 图中是子图输入供给（"subgraph_input"）。节点/边 ID 与连线两种
    模式完全一致（dspark.*/e_ds_*），不复制第二份连线。
    include_prefill：B1 含 prefill return 子图（x_draft_hc passthrough，
    769）置 True；stage 图 prefill 由 Block 级 passthrough 承载
    （DSparkBlock.forward 849），attention 内部不重复建模，置 False。
    fed_sources：True 时两个输入源节点带输入端口 x（骨架数据流从外部
    喂入源节点，stage 用）；False（B1）源节点只输出。
    返回创建的节点 ID（stage 用作 fold 组成员）。"""
    created: list[str] = []

    def node(sid, op, gr, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        created.append(sid)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid, src, dst, role, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    # ===== 共享权重节点（wkv/kv_norm 同一实例两次调用）=====
    node("dspark.wkv_w", "linear_weight", "parameter_op",
         (), (("y", t["wkv_w"]),), ("ev_ds_attn_init",))
    node("dspark.kv_norm_w", "rms_norm_weight", "parameter_op",
         (), (("y", t["kv_norm_w"]),), ("ev_ds_attn_init",))
    # 参数静态身份两相位可见；到 main 调用点的消费边不挂相位 guard
    # （main 路径两相位共享），到 draft 调用点的边挂 g_decode（draft
    # 只在 decode 实例化，771 起于相位分支之后）。
    edge("e_ds_wkv_w_main", ("dspark.wkv_w", "y"), ("dspark.wkv_main", "w"),
         "wkv_w")
    edge("e_ds_wkv_w_draft", ("dspark.wkv_w", "y"), ("dspark.wkv_draft", "w"),
         "wkv_w", guard_ref="g_decode")
    edge("e_ds_kvnorm_w_main", ("dspark.kv_norm_w", "y"),
         ("dspark.kv_norm_main", "w"), "kv_norm_w")
    edge("e_ds_kvnorm_w_draft", ("dspark.kv_norm_w", "y"),
         ("dspark.kv_norm_draft", "w"), "kv_norm_w", guard_ref="g_decode")

    # ===== main 路径（两相位同源；main_x [B,T,H]）=====
    # 输入经显式源节点消费（B1：图输入；stage：骨架经源节点 x 端口喂入
    # 的子图输入）：main 路径 758-761 在相位分支之前两相位共享，消费边
    # 不挂相位 guard。main_in_role/draft_in_role 与 _add_attn_tensors 的
    # 角色参数对应（B1 默认 x_main_in/x_draft_single；stage main_x/
    # x_draft_single）。
    node("dspark.x_main_in_src", src_op, "tensor_op",
         (("x", t[main_in_role]),) if fed_sources else (),
         (("y", t[main_in_role]),), ())
    node("dspark.wkv_main", "linear", "parameter_op",
         (("x", t[main_in_role]), ("w", t["wkv_w"])),
         (("y", t["main_kv_proj"]),), ("ev_ds_main_kv",))
    edge("e_ds_main_in_wkv", ("dspark.x_main_in_src", "y"),
         ("dspark.wkv_main", "x"), main_in_role)
    node("dspark.kv_norm_main", "rms_norm", "parameter_op",
         (("x", t["main_kv_proj"]), ("w", t["kv_norm_w"])),
         (("y", t["main_kv_normed"]),), ("ev_ds_main_kv",))
    edge("e_ds_main_wkv", ("dspark.wkv_main", "y"),
         ("dspark.kv_norm_main", "x"), "main_kv_proj")
    node("dspark.main_rope", "rope_partial", "tensor_op",
         (("x", t["main_kv_normed"]),), (("y", t["main_kv_quant"]),),
         ("ev_ds_main_kv", "ev_ds_rope"))
    edge("e_ds_main_norm_rope", ("dspark.kv_norm_main", "y"),
         ("dspark.main_rope", "x"), "main_kv_normed")
    node("dspark.main_quant", "act_quant_fp8", "quant_sim",
         (("x", t["main_kv_quant"]),), (("y", t["main_kv_quant"]),),
         ("ev_ds_main_kv",))
    edge("e_ds_main_rope_quant", ("dspark.main_rope", "y"),
         ("dspark.main_quant", "x"), "main_kv_quant")

    # ===== 窗口 cache 写（三真实分支：prefill T<=W 顺序 / prefill T>W 环形 /
    # decode 单槽 P%W）与读 =====
    # decode 写槽坐标：具名标量（零轴张量，值语义由 bundle 级
    # CoordinateExpression 原文 EXPR_SLOT_WRITE 承载并公开 JSON 往返，
    # 不充当任何张量的维度大小，也不是数据张量的存储 view），经真实消费
    # 边接入 cache_update.slot 端口（783 start_pos % win）。
    node("dspark.slot_src", "scalar_expr", "derived_expr",
         (), (("y", t["slot_write"]),), ("ev_ds_slot_write",))
    node("dspark.cache_update", "cache_write", "state_op",
         (("kv", t["main_kv_quant"]), ("slot", t["slot_write"])),
         (("cache_full", t["win_cache"]),),
         ("ev_ds_prefill", "ev_ds_decode", "ev_ds_capacity"))
    edge("e_ds_cachew_inwin", ("dspark.main_quant", "y"),
         ("dspark.cache_update", "kv"), "main_kv_quant",
         guard_ref="g_dspark_prefill_inwin")
    edge("e_ds_cachew_wrap", ("dspark.main_quant", "y"),
         ("dspark.cache_update", "kv"), "main_kv_quant",
         guard_ref="g_window_wrap")
    edge("e_ds_cachew_dec", ("dspark.main_quant", "y"),
         ("dspark.cache_update", "kv"), "main_kv_quant",
         guard_ref="g_decode")
    edge("e_slot_cachew", ("dspark.slot_src", "y"),
         ("dspark.cache_update", "slot"), "slot_write",
         guard_ref="g_decode", ev=("ev_ds_slot_write",))
    node("dspark.cache_read", "cache_read", "state_op",
         (("cache_full", t["win_cache"]),), (("y", t["kv_win_read"]),),
         ("ev_ds_decode",))
    # 写后读同一存储版本对（783 写 → 784 读）：cache_read 只在 decode
    # 实例化（prefill 769 早退），写后读边挂 g_decode——相位条件在边上
    # 明确，不在 transition 上声称。
    edge("e_ds_cache_state", ("dspark.cache_update", "cache_full"),
         ("dspark.cache_read", "cache_full"), "win_cache", kind="state",
         guard_ref="g_decode", state_transition_ref="tr_dspark_cache")
    # 真实 view：784 self.kv_cache[:bsz]（batch 前缀 view；world_size=1 全 batch）
    b.alias("alias_ds_cache_read_view", relation_kind="view",
            tensor_id=t["win_cache"], other_tensor_id=t["kv_win_read"],
            transform_expr="kv_cache[:bsz]（batch 前缀 view，model.py:784）",
            evidence=("ev_ds_decode",))
    # 写槽坐标表达式：场景求值原文（P % W，model.py:783 本次写入槽；
    # 下一次写槽 (P+T)%W 属下一调用前提，不混入本图）——以 bundle 级
    # CoordinateExpression 显式记录（见 build_dspark_attention_graph），
    # 不再伪装成数据张量 view：坐标 int64 值不是 bf16 cache 内存的别名。
    # 草稿 RoPE 位置起点坐标表达式（P + T，model.py:772）：q_rope 的
    # 位置起点是标量 draft_pos0（kv_rope/o_rope_inv 消费同一坐标）；
    # 同样以 CoordinateExpression 记录，无 view alias。

    # ===== prefill return x（769）=====
    # prefill 分支 attn 返回 x 原样（hc 展开流 [B,N_draft,N_hc,H]），
    # 未做注意力计算；图输入经显式源节点消费，passthrough 返回边挂
    # g_prefill——prefill passthrough 只能 prefill。（仅 B1 图；stage 图
    # 的 prefill 在 Block 级 passthrough，attention 内部不重复建模。）
    if include_prefill:
        node("dspark.x_draft_hc_src", "graph_input", "tensor_op",
             (), (("y", t["x_draft_hc"]),), ())
        node("dspark.prefill_return", "passthrough", "tensor_op",
             (("x", t["x_draft_hc"]),), (("y", t["x_draft_hc"]),),
             ("ev_ds_prefill",))
        edge("e_ds_hc_in_pref", ("dspark.x_draft_hc_src", "y"),
             ("dspark.prefill_return", "x"), "x_draft_hc",
             guard_ref="g_prefill", ev=("ev_ds_block_prefill",))

    # ===== decode 草稿 Q（774-777）=====
    # 输入 x_draft_single 经显式源节点消费（B1 图输入 / stage 骨架喂入）；
    # wq_a/wkv_draft 调用点只在 decode 实例化（771 起于相位分支之后），
    # 消费边挂 g_decode。
    node("dspark.x_draft_single_src", src_op, "tensor_op",
         (("x", t[draft_in_role]),) if fed_sources else (),
         (("y", t[draft_in_role]),), ())
    node("dspark.wq_a", "linear", "parameter_op",
         (("x", t[draft_in_role]),), (("y", t["qr"]),),
         ("ev_ds_draft_q", "ev_ds_attn_init"))
    edge("e_ds_xin_wqa", ("dspark.x_draft_single_src", "y"),
         ("dspark.wq_a", "x"), draft_in_role, guard_ref="g_decode")
    node("dspark.q_norm", "rms_norm", "parameter_op",
         (("x", t["qr"]),), (("y", t["qr"]),), ("ev_ds_draft_q",))
    edge("e_ds_qa_qnorm", ("dspark.wq_a", "y"), ("dspark.q_norm", "x"), "qr",
         guard_ref="g_decode")
    node("dspark.wq_b", "linear", "parameter_op",
         (("x", t["qr"]),), (("y", t["q_flat"]),), ("ev_ds_draft_q",))
    edge("e_ds_qnorm_wqb", ("dspark.q_norm", "y"), ("dspark.wq_b", "x"), "qr",
         guard_ref="g_decode")
    node("dspark.q_unflatten", "unflatten", "tensor_op",
         (("x", t["q_flat"]),), (("y", t["q"]),), ("ev_ds_draft_q",))
    edge("e_ds_wqb_unfl", ("dspark.wq_b", "y"),
         ("dspark.q_unflatten", "x"), "q_flat", guard_ref="g_decode")
    node("dspark.q_rms_scale", "rms_scale", "tensor_op",
         (("x", t["q"]),), (("y", t["q"]),), ("ev_ds_draft_q",))
    edge("e_ds_unfl_scale", ("dspark.q_unflatten", "y"),
         ("dspark.q_rms_scale", "x"), "q", guard_ref="g_decode")
    # 草稿 RoPE 位置起点：具名标量（bundle 级 CoordinateExpression 原文
    # EXPR_DRAFT_POS0，model.py:772 freqs_cis 切片自 start_pos+seqlen 起，
    # 不是 q 数据张量的存储 view）经真实消费边接入 rope 算子 pos 端口；
    # 只 decode。
    node("dspark.pos0_src", "scalar_expr", "derived_expr",
         (), (("y", t["draft_pos0"]),), ("ev_ds_draft_pos",))
    node("dspark.q_rope", "rope_partial", "tensor_op",
         (("x", t["q"]), ("pos", t["draft_pos0"])), (("y", t["q_rope"]),),
         ("ev_ds_draft_q", "ev_ds_draft_pos", "ev_ds_rope"))
    edge("e_ds_scale_rope", ("dspark.q_rms_scale", "y"),
         ("dspark.q_rope", "x"), "q", guard_ref="g_decode")
    edge("e_pos_qrope", ("dspark.pos0_src", "y"),
         ("dspark.q_rope", "pos"), "draft_pos0", guard_ref="g_decode",
         ev=("ev_ds_draft_pos",))

    # ===== decode 草稿 KV（778-780，复用 wkv_w/kv_norm_w）=====
    node("dspark.wkv_draft", "linear", "parameter_op",
         (("x", t[draft_in_role]), ("w", t["wkv_w"])),
         (("y", t["draft_kv_proj"]),), ("ev_ds_draft_kv",))
    edge("e_ds_xin_wkvd", ("dspark.x_draft_single_src", "y"),
         ("dspark.wkv_draft", "x"), draft_in_role, guard_ref="g_decode")
    node("dspark.kv_norm_draft", "rms_norm", "parameter_op",
         (("x", t["draft_kv_proj"]), ("w", t["kv_norm_w"])),
         (("y", t["draft_kv_normed"]),), ("ev_ds_draft_kv",))
    edge("e_ds_draft_wkv", ("dspark.wkv_draft", "y"),
         ("dspark.kv_norm_draft", "x"), "draft_kv_proj", guard_ref="g_decode")
    node("dspark.kv_rope", "rope_partial", "tensor_op",
         (("x", t["draft_kv_normed"]), ("pos", t["draft_pos0"])),
         (("y", t["draft_kv_quant"]),),
         ("ev_ds_draft_kv", "ev_ds_draft_pos", "ev_ds_rope"))
    edge("e_ds_draft_norm_rope", ("dspark.kv_norm_draft", "y"),
         ("dspark.kv_rope", "x"), "draft_kv_normed", guard_ref="g_decode")
    edge("e_pos_kvrope", ("dspark.pos0_src", "y"),
         ("dspark.kv_rope", "pos"), "draft_pos0", guard_ref="g_decode",
         ev=("ev_ds_draft_pos",))
    node("dspark.kv_quant", "act_quant_fp8", "quant_sim",
         (("x", t["draft_kv_quant"]),), (("y", t["draft_kv_quant"]),),
         ("ev_ds_draft_kv",))
    edge("e_ds_draft_rope_quant", ("dspark.kv_rope", "y"),
         ("dspark.kv_quant", "x"), "draft_kv_quant", guard_ref="g_decode")

    # ===== 选择索引与 sink =====
    node("dspark.idx_build", "dspark_index", "index_op",
         (), (("y", t["topk_idxs"]),), ("ev_ds_topk",))
    node("dspark.attn_sink", "attn_sink_param", "parameter_op",
         (), (("y", t["attn_sink_param"]),), ("ev_ds_attn_init",))

    # ===== 窗口 cache + 草稿 KV concat（784）=====
    node("dspark.kv_cat", "concat", "tensor_op",
         (("win", t["kv_win_read"]), ("draft", t["draft_kv_quant"])),
         (("y", t["kv_concat"]),), ("ev_ds_decode",))
    edge("e_ds_read_cat", ("dspark.cache_read", "y"),
         ("dspark.kv_cat", "win"), "kv_win_read", guard_ref="g_decode")
    edge("e_ds_draft_cat", ("dspark.kv_quant", "y"),
         ("dspark.kv_cat", "draft"), "draft_kv_quant", guard_ref="g_decode")

    # ===== sparse_attn（不透明 kernel，785）=====
    node("dspark.sparse_attn", "sparse_attention", "opaque_kernel",
         (("q", t["q_rope"]), ("kv", t["kv_concat"]),
          ("topk_idxs", t["topk_idxs"]), ("sink", t["attn_sink_param"])),
         (("o", t["o_rope"]),),
         ("ev_ds_sparse", "ev_ds_sparse_kernel", "ev_ds_kernel_numerics"))
    edge("e_ds_sparse_q", ("dspark.q_rope", "y"),
         ("dspark.sparse_attn", "q"), "q_rope", guard_ref="g_decode")
    edge("e_ds_cat_sparse", ("dspark.kv_cat", "y"),
         ("dspark.sparse_attn", "kv"), "kv_concat", guard_ref="g_decode")
    edge("e_ds_idx_select", ("dspark.idx_build", "y"),
         ("dspark.sparse_attn", "topk_idxs"), "topk_idxs", kind="select",
         guard_ref="g_decode")
    # sink 是静态参数身份（两相位可见），但其唯一消费点是 decode-only 的
    # sparse_attn（769 早退），消费边挂 g_decode——参数身份不等于计算边
    # prefill 运行。
    edge("e_ds_sparse_sink", ("dspark.attn_sink", "y"),
         ("dspark.sparse_attn", "sink"), "attn_sink_param",
         guard_ref="g_decode")
    edge("e_ds_sparse_o", ("dspark.sparse_attn", "o"),
         ("dspark.o_rope_inv", "x"), "o_rope", guard_ref="g_decode")

    # ===== inverse RoPE → O 分组低秩（786-791）=====
    # inverse RoPE 与 Q/KV RoPE 同源：o 的 RoPE 位置也是草稿位置
    # P+T 起 N_draft 个（786 复用 772 同一 freqs_cis），pos 端口消费
    # 同一标量；O 链全部 decode-only（769 早退）。
    node("dspark.o_rope_inv", "rope_partial_inverse", "tensor_op",
         (("x", t["o_rope"]), ("pos", t["draft_pos0"])),
         (("y", t["o_rope"]),),
         ("ev_ds_sparse", "ev_ds_rope"))
    edge("e_pos_oinv", ("dspark.pos0_src", "y"),
         ("dspark.o_rope_inv", "pos"), "draft_pos0", guard_ref="g_decode",
         ev=("ev_ds_sparse",))
    edge("e_ds_oinv_reshape", ("dspark.o_rope_inv", "y"),
         ("dspark.o_reshape", "x"), "o_rope", guard_ref="g_decode")
    node("dspark.o_reshape", "reshape", "tensor_op",
         (("x", t["o_rope"]),), (("y", t["o_grouped"]),), ("ev_ds_ogroup",))
    edge("e_ds_reshape_einsum", ("dspark.o_reshape", "y"),
         ("dspark.o_einsum", "x"), "o_grouped", guard_ref="g_decode")
    node("dspark.o_einsum", "grouped_lowrank_einsum", "parameter_op",
         (("x", t["o_grouped"]),), (("y", t["o_lora"]),), ("ev_ds_ogroup",))
    edge("e_ds_einsum_flat", ("dspark.o_einsum", "y"),
         ("dspark.o_flatten", "x"), "o_lora", guard_ref="g_decode")
    node("dspark.o_flatten", "flatten", "tensor_op",
         (("x", t["o_lora"]),), (("y", t["o_flat"]),), ("ev_ds_ogroup",))
    edge("e_ds_flat_wob", ("dspark.o_flatten", "y"),
         ("dspark.wo_b", "x"), "o_flat", guard_ref="g_decode")
    node("dspark.wo_b", "linear", "parameter_op",
         (("x", t["o_flat"]),), (("y", t["attn_out"]),), ("ev_ds_ogroup",))
    return tuple(created)


def _add_flow(b: _GraphBuilder, t: dict[str, str]) -> None:
    """B1 图流：graph_input 源 + prefill return 子图（stage 同源复用
    _add_attn_flow 的单一实现，参数不同）。"""
    _add_attn_flow(b, t, src_op="graph_input", include_prefill=True)


def _attn_coords(t: dict[str, str], *, node_map: dict, tensor_ids: set[str]
                 ) -> tuple[CoordinateExpression, ...]:
    """坐标值显式记录（B1/stage 公共）：表达式与图内真实消费端口核对
    （消费端口不存在或标量张量不在图中即拒绝——记录必须可由图自身证伪）。"""
    coords = (
        CoordinateExpression(
            coordinate_id="dspark:slot_write",
            expression=EXPR_SLOT_WRITE,
            scalar_tensor_id=t["slot_write"],
            consumer_node_id="dspark.cache_update",
            consumer_port="slot",
            evidence_refs=("ev_ds_slot_write",),
            claim="decode 本次写入槽坐标（model.py:783 start_pos % win）："
                  "int64 标量坐标值，经 cache_update.slot 端口消费；"
                  "非 shape 轴、非 bf16 数据张量的存储 view/别名",
        ),
        CoordinateExpression(
            coordinate_id="dspark:draft_pos0",
            expression=EXPR_DRAFT_POS0,
            scalar_tensor_id=t["draft_pos0"],
            consumer_node_id="dspark.q_rope",
            consumer_port="pos",
            evidence_refs=("ev_ds_draft_pos",),
            claim="草稿 RoPE 位置起点坐标（model.py:772 start_pos+seqlen）："
                  "int64 标量坐标值，经 q_rope/kv_rope/o_rope_inv 的 pos "
                  "端口消费（同一坐标三处消费）；非 shape 轴、非 bf16 "
                  "数据张量的存储 view/别名",
        ),
    )
    for c in coords:
        if c.scalar_tensor_id not in tensor_ids:
            raise ExtractionError(
                f"坐标记录 {c.coordinate_id} 的标量张量不在图中: "
                f"{c.scalar_tensor_id}")
        n = node_map.get(c.consumer_node_id)
        if n is None:
            raise ExtractionError(
                f"坐标记录 {c.coordinate_id} 的消费节点不在图中: "
                f"{c.consumer_node_id}")
        if not any(p.name == c.consumer_port and p.tensor_id == c.scalar_tensor_id
                   for p in n.input_ports):
            raise ExtractionError(
                f"坐标记录 {c.coordinate_id} 声称的消费端口 "
                f"{c.consumer_node_id}.{c.consumer_port} 不消费其标量张量")
        check_expression(c.expression)
    return coords


def _states(b: _GraphBuilder) -> None:
    """窗口 cache 状态：v_hist 历史累积 → v_after 本调用写入后。

    历史连续调用前提：P 个历史 token 已按同一窗口规则写入环形槽位；
    未给此前提时 cache 内容 unknown（不当作空）。"""
    b.state("dspark:win_cache", "tr_dspark_cache",
            ("ev_ds_prefill", "ev_ds_decode"),
            "窗口 cache（容量 W，r=0 无压缩区）：prefill 顺序/环形写、decode "
            "单槽 start_pos%W 更新；v_hist 历史累积 → v_after 本调用写入后；"
            "历史连续调用前提，内容 unknown")


def _guards() -> tuple[Guard, ...]:
    """相位/分支 guard。g_window_wrap 在本图限定 prefill
    （start_pos == 0 and T > W）：回绕写只发生在 prefill 分支（767-768），
    decode 主 seqlen=1 是实际调用前提，非法长 decode 不激活 prefill 分支；
    与 rules GUARD_DEFS 的 g_window_wrap（T > W）相差相位限定——rules
    只读不改，本图谓词自足求值，不引用 rules 实例。谓词与 GUARD_DEFS
    同文的部分（g_prefill/g_decode/g_dspark_prefill_inwin）保持原文。
    evidence_refs 指向本模块注册的 DSpark 证据条目。"""
    for expr in ("start_pos == 0", "start_pos > 0",
                 "start_pos == 0 and T <= W", "start_pos == 0 and T > W",
                 EXPR_SLOT_WRITE, EXPR_DRAFT_POS0):
        check_expression(expr)
    return (
        Guard("g_prefill", "start_pos == 0", ("ev_ds_prefill",)),
        Guard("g_decode", "start_pos > 0", ("ev_ds_decode",)),
        Guard("g_dspark_prefill_inwin", "start_pos == 0 and T <= W",
              ("ev_ds_prefill",)),
        Guard("g_window_wrap", "start_pos == 0 and T > W",
              ("ev_ds_prefill",)),
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoordinateExpression:
    """一个具名坐标值在图中的显式记录（B2 修正引入）。

    坐标（decode 写槽、草稿 RoPE 起点等）是 **整数值**（int64 标量），
    不是任何 bf16 数据张量的存储 view/别名——本记录显式声明这一事实：
    - coordinate_id：模板局部稳定 ID（如 "dspark:slot_write"）；
    - expression：可求值原文（非 Python 执行；求值器为 scenario_evaluator）；
    - scalar_tensor_id：承载该坐标的零轴标量张量（int64）；
    - consumer_node_id / consumer_port：图内真实消费端口（坐标被算子
      端口消费，而非 shape 轴或存储关系）；
    - evidence_refs：锚定原文的证据条目；
    - claim：语义声明（坐标值，非 shape 轴、非存储别名）。
    """

    coordinate_id: str
    expression: str
    scalar_tensor_id: str
    consumer_node_id: str
    consumer_port: str
    evidence_refs: tuple[str, ...]
    claim: str

    def to_dict(self) -> dict:
        return {
            "coordinate_id": self.coordinate_id,
            "expression": self.expression,
            "scalar_tensor_id": self.scalar_tensor_id,
            "consumer_node_id": self.consumer_node_id,
            "consumer_port": self.consumer_port,
            "evidence_refs": list(self.evidence_refs),
            "claim": self.claim,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "CoordinateExpression":
        try:
            refs = doc["evidence_refs"]
        except (KeyError, TypeError) as exc:
            raise ExtractionError(f"坐标表达式记录缺字段: {exc}") from exc
        if not isinstance(refs, list) or not refs:
            raise ExtractionError("坐标表达式记录 evidence_refs 必须是非空 list")
        obj = cls(
            coordinate_id=doc["coordinate_id"],
            expression=doc["expression"],
            scalar_tensor_id=doc["scalar_tensor_id"],
            consumer_node_id=doc["consumer_node_id"],
            consumer_port=doc["consumer_port"],
            evidence_refs=tuple(refs),
            claim=doc["claim"],
        )
        for f in ("coordinate_id", "expression", "scalar_tensor_id",
                  "consumer_node_id", "consumer_port", "claim"):
            v = getattr(obj, f)
            if not isinstance(v, str) or not v.strip():
                raise ExtractionError(f"坐标表达式记录字段 {f} 必须是非空字符串")
        if obj.scalar_tensor_id != doc["scalar_tensor_id"]:
            raise ExtractionError("scalar_tensor_id 前后不一致")
        return obj


@dataclass(frozen=True)
class DSparkAttentionBundle:
    """B1/B2 交付包：DSparkAttention 子图 + 已核对证据注册表 + 模型数值
    + 具名坐标值显式记录（B2：坐标不是数据张量的 view）。"""

    graph: SemanticGraph
    registry: EvidenceRegistry
    values: ModelValues
    coordinate_expressions: tuple[CoordinateExpression, ...] = ()


def bundle_to_json(bundle: DSparkAttentionBundle) -> str:
    """bundle 公开规范序列化：图 canonical JSON + 坐标表达式记录。

    规范化（键排序、无空白），重复构建逐字节一致；坐标表达式原文随
    往返携带，篡改可从读回结果检出（测试覆盖）。"""
    doc = {
        "schema": "dspark-bundle/v1",
        "template_id": bundle.graph.template_id,
        "graph": json.loads(graph_to_json(bundle.graph)),
        "coordinate_expressions": [c.to_dict()
                                   for c in bundle.coordinate_expressions],
    }
    return json.dumps(doc, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def bundle_from_json(text: str) -> DSparkAttentionBundle:
    """读回 bundle：图 canonical JSON 解析回图，坐标表达式记录重建。

    结构不合法（缺 schema/template_id/graph/坐标记录字段、坐标表达式
    缺失或空、记录非 list）具名抛 :class:`ExtractionError`；证据注册表
    与模型数值不随 JSON 往返（注册时核对原文，往返后再核对是无谓重复，
    由调用方需要时另行构建）。"""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"bundle JSON 不合法: {exc}") from exc
    if not isinstance(doc, dict):
        raise ExtractionError("bundle JSON 顶层必须是对象")
    if doc.get("schema") != "dspark-bundle/v1":
        raise ExtractionError(
            f"bundle schema 不符，期望 dspark-bundle/v1，得到 {doc.get('schema')!r}")
    tid = doc.get("template_id")
    if not isinstance(tid, str) or not tid.strip():
        raise ExtractionError("bundle template_id 必须是非空字符串")
    if "graph" not in doc or not isinstance(doc["graph"], dict):
        raise ExtractionError("bundle 必须携带 graph 对象")
    graph = graph_from_json(json.dumps(doc["graph"], ensure_ascii=False,
                                       sort_keys=True, separators=(",", ":")))
    if graph.template_id != tid:
        raise ExtractionError("bundle template_id 与图 template_id 不一致")
    raw_coords = doc.get("coordinate_expressions")
    if not isinstance(raw_coords, list) or not raw_coords:
        raise ExtractionError(
            "bundle 必须携带非空 coordinate_expressions 记录列表")
    coords = tuple(CoordinateExpression.from_dict(c) for c in raw_coords)
    seen: set[str] = set()
    for c in coords:
        if c.coordinate_id in seen:
            raise ExtractionError(f"坐标表达式记录 ID 重复: {c.coordinate_id}")
        seen.add(c.coordinate_id)
    return DSparkAttentionBundle(graph=graph, registry=None, values=None,
                                 coordinate_expressions=coords)


def build_dspark_attention_graph(
    workspace_root: Path | None = None,
    *,
    values: ModelValues | None = None,
    evidence_base: EvidenceBase | None = None,
) -> DSparkAttentionBundle:
    """构建 DSparkAttention 真实子图：证据核对 → 图构建 → 通用引用检查。

    返回 :class:`DSparkAttentionBundle`，附两条坐标值显式记录
    （dspark:slot_write → cache_update.slot、dspark:draft_pos0 →
    q_rope/kv_rope/o_rope_inv.pos）；记录的 consumer/标量张量必须真实
    存在于图（构建期核对，不存在即拒绝）。任何证据核对失败、图构造错误
    或 validate_graph_references 未通过都抛 :class:`ExtractionError`——
    失败不给图。
    """
    root = Path(workspace_root) if workspace_root is not None else find_workspace_root()
    values = values or load_model_values(root)
    evidence_base = evidence_base or load_evidence_base(root)
    registry = EvidenceRegistry(evidence_base, root)
    for ev in _evidence_defs():
        registry.register(ev)

    axes = _axes(values)
    b = _GraphBuilder(axes)
    t = _add_tensors(b, axes)
    _add_flow(b, t)
    _states(b)
    guards = _guards()

    # guard 引用的证据必须已注册（引用完整性）
    for g in guards:
        for ref in g.evidence_refs:
            if ref not in registry:
                raise ExtractionError(
                    f"guard {g.stable_id} 引用了未注册证据: {ref!r}")
    # 全部图引用的证据必须已注册
    for ev_id in b.evidence_ids:
        if ev_id not in registry:
            raise ExtractionError(f"图引用了未注册证据: {ev_id!r}")
    for axis in axes.values():
        if (axis.evidence_ref.startswith("ev_")
                and axis.evidence_ref not in registry):
            raise ExtractionError(
                f"轴 {axis.symbol} 引用了未注册证据: {axis.evidence_ref!r}"
            )

    graph = b.build(guards, TEMPLATE_ID)

    # 坐标值显式记录：表达式与图内真实消费端口构建期核对（消费端口不存在
    # 或标量张量不在图中即拒绝——记录必须可由图自身证伪）
    coords = _attn_coords(
        t, node_map={n.stable_id: n for n in graph.nodes},
        tensor_ids={x.stable_id for x in graph.tensors})

    return DSparkAttentionBundle(graph=graph, registry=registry, values=values,
                                 coordinate_expressions=coords)
