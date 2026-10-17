"""DeepSeek-V4-Flash-0731 真实参考提取器（工作包 B）：layer 4 CSA·TopK 代表图。

数据来源：models/deepseek-ai/DeepSeek-V4-Flash-0731/inference/{model.py,kernel.py,config.json}
（固定 SHA256 基线，:mod:`structure_reader.extract.evidence` 逐条核对原文）。
本模块按 model.py 的 Block/Attention/Compressor/Indexer/Gate/Expert/MoE 真实
控制流构建 layer 4（compress_ratio=4 → CSA，layer 4 ≥ n_hash_layers=3 → TopK
门控）的语义图模板，不复制 synthetic fixtures。

图覆盖（对照 task.md 逐条）：
- Block 两次 hc_pre → norm → module → hc_post（model.py:695-707）；mHC comb
  的源 j / 目标 k 是两个独立轴（hc_j/hc_k），方向为
  sum_j comb[...,j,k] * residual[...,j,d]（model.py:692，source_checked）。
- Attention：wq_a/q_norm 低秩 qr（qr 同时供给 wq_b 与 Indexer 两个消费方）、
  wq_b/查询 RMS 缩放/部分 RoPE、共享单投影 KV（K/V 同一张量，same_tensor
  alias）/kv_norm/部分 RoPE/非 RoPE 维 FP8 量化模拟、窗口选择索引、
  sparse_attn 的 sink 与不透明 kernel 边界、inverse RoPE 先于 reshape/分组
  低秩 O einsum/wo_b。
- Compressor（overlap）：wkv/wgate/APE、overlap 前后窗口、softmax 加权汇聚、
  norm/部分 RoPE/量化、独立 kv_state/score_state 前后状态与注意力 cache
  压缩区 slice alias（model.py:497）；prefill/decode guard 区分，未触发
  （should_compress=False）时 compressor 返回 None，不伪造输出值。
- Indexer：x 与 qr 两个真实输入、独立 wq_b/旋转/FP4 量化、weights_proj、
  自有 Compressor（rotate=True，与 attention Compressor 同模板不同实例参数）
  与独立 cache、score/relu/加权汇聚、prefill 因果掩码、topk 索引/offset 与
  -1 无效标记；输出 select 索引而非 Value。
- MoE TopK Gate：score（sqrtsoftplus）、选择偏置不影响权重（gather 自
  original_scores）、indices/weights、token dispatch（每专家 token 数 N_e
  数据相关 unknown）、单专家 SwiGLU 模板（不展开 256 份）、加权累加、
  shared expert 路径与汇合。

场景数据（build_scenario_views）：prefill T=3/4/5/127/128/129，decode
P=2/3/4/127/128；guard 三态激活（evaluate_tri 白名单求值）；Indexer 可见
条目/有效选择边界；历史连续调用前提显式记录，未给 cache 前提的状态内容
unknown，不当作空。world_size=1（graph.world_size=1 + 视图记录假设）。

本模块不导入/不执行模型代码，只读源码文本与 config。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from structure_reader.checks.graph_contracts import validate_graph_references
from structure_reader.extract.evidence import (
    EvidenceBase,
    EvidenceError,
    EvidenceRegistry,
    SourceEvidence,
    load_evidence_base,
)
from structure_reader.graph_document import SemanticGraph
from structure_reader.graph_ir import (
    AliasRelation,
    Edge,
    FoldBoundaryMapping,
    FoldGroup,
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
    SCENARIOS,
    ModelValues,
    derive_layer_groups,
    load_model_values,
    scenario_base_env,
)
from structure_reader.scenario_evaluator import evaluate_int, evaluate_tri

__all__ = [
    "ExtractionError",
    "ReferenceGraphBundle",
    "ScenarioView",
    "build_reference_graph",
    "build_scenario_views",
    "indexer_visibility",
]

TEMPLATE_ID = "deepseek_v4_flash_0731:csa_topk"
REPRESENTATIVE_LAYER = 4


# ---------------------------------------------------------------------------
# 阶段统一表达式（有限求值器白名单内）——图轴与场景视图同源求值的唯一来源。
#
# 相位由 P 编码：P==0 → prefill（start_pos==0），P>0 → decode（start_pos>0），
# 与 g_prefill 谓词同一判据（场景 env 恒有 start_pos==P，rules.py 场景目录）。
# 真实语义（model.py:261-271 get_window_topk_idxs / 274-282
# get_compress_topk_idxs / 513-519 forward）：
# - 窗口选择索引物理宽：prefill min(T,W)；decode 恒 W 槽位（不足 -1 填充，
#   两个 decode 分支宽都是 W）——槽位数 ≠ 有效数。
# - 窗口有效数上限：min(P+T, W)（prefill 逐 query 递增到 min(T,W)；
#   decode 单 query = min(P+1, W)）。
# - 固定压缩选择宽（HCA）：prefill T//r（P=0 → (P+T)//r；逐 query 有效
#   (s+1)//r 且 -1 掩码）；decode (P+1)//r（T=1 → (P+T)//r；全有效）。
# - 学习式压缩选择宽（CSA）：min(Ki, (P+T)//r)（model.py:433 topk k）。
# - 压缩条目 offset：prefill kv.size(1)=T；decode win=W（model.py:515）。
# ---------------------------------------------------------------------------
EXPR_WINDOW_WIDTH = "min(T, W) + min(P, 1) * (W - min(T, W))"
EXPR_WINDOW_VALID = "min(P + T, W)"
EXPR_COMPRESS_WIDTH_FIXED = "(P + T) // r"
EXPR_COMPRESS_WIDTH_LEARNED = "min(Ki, (P + T) // r)"
EXPR_SELECT_WIDTH_FIXED = EXPR_WINDOW_WIDTH + " + " + EXPR_COMPRESS_WIDTH_FIXED
EXPR_SELECT_WIDTH_LEARNED = (EXPR_WINDOW_WIDTH + " + "
                             + EXPR_COMPRESS_WIDTH_LEARNED)
EXPR_SELECT_VALID_UPPER_FIXED = (EXPR_WINDOW_VALID + " + "
                                 + EXPR_COMPRESS_WIDTH_FIXED)
EXPR_SELECT_VALID_UPPER_LEARNED = (EXPR_WINDOW_VALID + " + "
                                   + EXPR_COMPRESS_WIDTH_LEARNED)
EXPR_COMPRESS_OFFSET_FIXED = "T + min(P, 1) * (W - T)"


class ExtractionError(ValueError):
    """参考图提取失败：证据核对、图构造或引用检查不通过（具名拒绝）。"""


# ---------------------------------------------------------------------------
# 证据目录：每条 = qualified symbol + 行区间 + 原文片段 + 核对结论 + 等级
# ---------------------------------------------------------------------------

_E = SourceEvidence  # 简写


def _evidence_defs() -> tuple[SourceEvidence, ...]:
    """图与契约引用的全部证据条目（注册时逐条对源码原文核对）。"""
    m = "model.py"
    k = "kernel.py"
    return (
        # --- Block / mHC ---
        _E("ev_block_forward", m, "Block.forward", 695, 707,
           "residual = x\nx, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)\nx = self.attn_norm(x)\nx = self.attn(x, start_pos, *attn_args)\nx = self.hc_post(x, residual, post, comb)",
           "Block 两次 hc_pre→norm→module→hc_post：attention 残差至 post_attn，FFN 第二次残差至 post_ffn", "source_checked"),
        _E("ev_hc_pre", m, "Block.hc_pre", 680, 688,
           "mixes = F.linear(x, hc_fn) * rsqrt\npre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)\ny = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)",
           "hc_pre：N_mix 混合参数 → Sinkhorn 分裂 pre/post/comb；pre 加权汇聚 hc 流得单流 y", "located"),
        _E("ev_hc_post", m, "Block.hc_post", 690, 693,
           "y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)",
           "hc_post 逐元素核对：comb.unsqueeze(-1)=[b,s,hc_j,hc_k,1] 与 residual.unsqueeze(-2)=[b,s,1,hc_j,d] 对齐相乘后 sum(dim=2) 消去源下标 j，保留目标下标 k——方向为 sum_j comb[...,j,k] * residual[...,j,d]，不能凭长度同为 4 的 shape 判断", "source_checked"),
        _E("ev_hc_params", m, "Block.__init__ hc params", 670, 678,
           "mix_hc = (2 + hc_mult) * hc_mult\nhc_dim = hc_mult * args.dim",
           "N_mix=(2+N_hc)×N_hc、N_hcH=N_hc×H 参数形状依据；hc_attn_fn/hc_ffn_fn 独立参数", "located"),
        _E("ev_hc_sinkhorn_kernel", k, "hc_split_sinkhorn", 430, 441,
           "def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):",
           "hc_split_sinkhorn 调用接口与 (pre, post, comb) 返回三元组已定位核对；kernel 内部数值生成另见 ev_idx_rotate 同源 uncovered 说明", "located",
           note="kernel.py hc_split_sinkhorn_kernel 内部 Sinkhorn 迭代与归一化细节未核对（uncovered 见模块级说明）；本条只声称接口定位"),
        _E("ev_attn_norm", m, "RMSNorm", 197, 202,
           "var = x.square().mean(-1, keepdim=True)\nx = x * torch.rsqrt(var + self.eps)",
           "RMSNorm：块内 attn_norm/ffn_norm/q_norm/kv_norm/compressor.norm 共用实现", "located"),
        # --- Attention ---
        _E("ev_attn_init", m, "Attention.__init__", 445, 470,
           "self.wq_a = Linear(self.dim, self.q_lora_rank)\nself.q_norm = RMSNorm(self.q_lora_rank, self.eps)\nself.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)\nself.wkv = Linear(self.dim, self.head_dim)\nself.kv_norm = RMSNorm(self.head_dim, self.eps)",
           "低秩 Q（wq_a→q_norm→wq_b）、共享单投影 KV（wkv 一个投影供 K/V）、O 分组低秩（wo_a/wo_b）、attn_sink 参数", "located"),
        _E("ev_compress_branch", m, "Attention.__init__ compressor/indexer", 472, 477,
           "if self.compress_ratio:\n    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)\n    if self.compress_ratio == 4:\n        self.indexer = Indexer(args, self.compress_ratio)",
           "compress_ratio≠0 建 Compressor；==4 建 Indexer（CSA 专属）；layer 4 ratio=4 → 两者都有", "located"),
        _E("ev_layer_ratio", m, "Attention.__init__ compress_ratio", 459, 459,
           "self.compress_ratio = args.compress_ratios[layer_id]",
           "层压缩比取自 config compress_ratios[layer]；layer 4 → 4（run_config.json:39 第 5 项）", "located"),
        _E("ev_cache_capacity", m, "Attention.__init__ kv_cache", 479, 480,
           'kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)',
           "注意力 cache 容量 = W + Mcap//r 条（窗口槽 + 压缩区）；N_cap 轴依据", "located"),
        _E("ev_lazy_bind", m, "Attention.forward lazy bind", 496, 500,
           "self.compressor.kv_cache = self.kv_cache[:, win:]\nself.compressor.freqs_cis = self.freqs_cis",
           "压缩区 slice alias：compressor 的 cache 是注意力 cache[:, win:] 视图（同存储，非独立拷贝）；首次调用惰性绑定", "located"),
        _E("ev_q_path", m, "Attention.forward q", 502, 505,
           "qr = q = self.q_norm(self.wq_a(x))\nq = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))\nq *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)\napply_rotary_emb(q[..., -rd:], freqs_cis)",
           "qr = q_norm(wq_a(x)) 保存；wq_b 展开 → 手写 RMS 缩放 → 尾部 rd 维部分 RoPE", "located"),
        _E("ev_kv_path", m, "Attention.forward kv", 508, 512,
           "kv = self.wkv(x)\nkv = self.kv_norm(kv)\napply_rotary_emb(kv[..., -rd:], freqs_cis)\nact_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)",
           "共享单投影 KV：一个 wkv 投影 + kv_norm + 部分 RoPE；非 RoPE 维 FP8 量化模拟（QAT，in-place）", "located"),
        _E("ev_window_topk", m, "get_window_topk_idxs", 261, 271,
           "if start_pos >= window_size - 1:\n    start_pos %= window_size\nelif start_pos > 0:\n    matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)",
           "窗口选择索引：回绕环形槽位；未填满时 -1 无效填充——槽位数量 ≠ 有效选择数", "source_checked"),
        _E("ev_offset", m, "Attention.forward offset", 515, 515,
           "offset = kv.size(1) if start_pos == 0 else win",
           "压缩条目索引偏移：prefill 压缩条目拼在窗口 KV 之后（offset=T），decode 偏移 = 窗口槽位数 W", "located"),
        _E("ev_indexer_call", m, "Attention.forward indexer call", 516, 520,
           "compress_topk_idxs = self.indexer(x, qr, start_pos, offset).int()\nelse:\n    compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)\ntopk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)",
           "Indexer 实参 (x, qr, start_pos, offset)：qr 是第二真实输入；选择索引 cat 到窗口索引之后", "located"),
        _E("ev_prefill_cache_write", m, "Attention.forward prefill cache", 523, 528,
           "self.kv_cache[:bsz, :seqlen] = kv\nelse:\n    cutoff = seqlen % win\n    self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)",
           "prefill 窗口 cache 写入：不超窗顺序写，超窗（T>W）环形回绕写", "source_checked"),
        _E("ev_prefill_compress", m, "Attention.forward prefill compress", 529, 533,
           "if (kv_compress := self.compressor(x, start_pos)) is not None:\n    kv = torch.cat([kv, kv_compress], dim=1)\no = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)",
           "prefill：compressor 返回 None（T<r）则不拼接；sparse_attn 输入为 [窗口 kv | 压缩 kv] 与 sink", "located"),
        _E("ev_decode_cache_read", m, "Attention.forward decode", 534, 538,
           "self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)\nif self.compress_ratio:\n    self.compressor(x, start_pos)\no = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)",
           "decode：当前 token 写窗口槽位 start_pos%W；sparse_attn 读整个 cache（窗口+压缩区）", "located"),
        _E("ev_rope_inv", m, "Attention.forward inverse rope", 539, 539,
           "apply_rotary_emb(o[..., -rd:], freqs_cis, True)",
           "inverse RoPE（conj 旋转）在 sparse_attn 输出上、必须先于 O 的 reshape/分组 einsum/wo_b（apply_rotary_emb inverse=True 用共轭，model.py:238-243）", "source_checked"),
        _E("ev_apply_rotary", m, "apply_rotary_emb", 238, 250,
           "if inverse:\n    freqs_cis = freqs_cis.conj()",
           "RoPE 正/逆共用实现，inverse 用共轭；部分维度（尾部 rd）旋转", "located"),
        _E("ev_o_group", m, "Attention.forward o projection", 542, 547,
           "o = o.view(bsz, seqlen, self.n_local_groups, -1)\nwo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)\no = torch.einsum(\"bsgd,grd->bsgr\", o, wo_a)\nx = self.wo_b(o.flatten(2))",
           "inverse RoPE 之后：reshape 成 G 组（每组 D_g=Hq×D/G 紧凑头）、分组低秩 einsum（G×Ro）、展平 GRo 后 wo_b 投影回 H", "source_checked"),
        _E("ev_sparse_attn_kernel", k, "sparse_attn", 355, 370,
           "def sparse_attn(\n    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float\n) -> torch.Tensor:",
           "sparse_attn 接口签名已定位核对；按 topk_idxs 聚集/在线 softmax/sink 偏置的 kernel 内部数值过程 uncovered——不透明 kernel 边界", "located",
           note="kernel.py sparse_attn_kernel 的 gather/online-softmax 数值实现未核对（uncovered 边界：数值语义不声称已验证）；本条只声称接口签名定位"),
        # --- Compressor ---
        _E("ev_comp_init", m, "Compressor.__init__", 285, 311,
           "self.overlap = compress_ratio == 4\ncoff = 1 + self.overlap\nself.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))\nself.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)\nself.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)",
           "overlap = (ratio==4)（CSA 专属）；coff=2 → wkv/wgate 输出 2D 宽（前半 overlap 压缩、后半常规）；APE 形状 [r, 2D]", "located"),
        _E("ev_comp_states", m, "Compressor.__init__ state buffers", 307, 310,
           'self.register_buffer("kv_state", torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)',
           "独立压缩状态 kv_state/score_state：[B, coff×r=2r 槽, 2D]；overlap 时前 ratio 槽为 overlap 窗、后 ratio 槽为当前窗", "located"),
        _E("ev_overlap_transform", m, "Compressor.overlap_transform", 313, 320,
           "new_tensor[:, :, ratio:] = tensor[:, :, :, d:]\nnew_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]",
           "overlap 变换：后半取本块常规压缩、前半取前一块的 overlap 半——块间重叠窗口", "source_checked"),
        _E("ev_comp_prefill", m, "Compressor.forward prefill", 331, 348,
           "should_compress = seqlen >= ratio\nremainder = seqlen % ratio\ncutoff = seqlen - remainder\noffset = ratio if overlap else 0",
           "prefill：T≥r 触发；remainder 尾块写入 kv_state/score_state 供后续调用（不丢弃）；主序列 unflatten 成块 + APE + overlap 变换 + softmax(dim=2) 加权汇聚", "source_checked"),
        _E("ev_comp_decode", m, "Compressor.forward decode", 350, 365,
           "should_compress = (start_pos + 1) % self.compress_ratio == 0\nscore += self.ape[start_pos % ratio]",
           "decode：逐 token 累积；恰能整除 r 时触发压缩（softmax 加权汇聚后输出一条压缩条目）并前移状态槽", "source_checked"),
        _E("ev_comp_no_trigger", m, "Compressor.forward not should_compress", 366, 367,
           "if not should_compress:\n    return",
           "未触发压缩时返回 None——调用方不拼接、不产出压缩条目；不存在伪造的压缩输出", "located"),
        _E("ev_comp_out", m, "Compressor.forward output path", 368, 383,
           "kv = self.norm(kv.to(dtype))\napply_rotary_emb(kv[..., -rd:], freqs_cis)\nact_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)",
           "压缩条目：RMSNorm → 部分 RoPE（prefill 用 freqs_cis[:cutoff:ratio]，decode 用 start_pos+1-ratio 单点）→ 非 RoPE 维量化 → 写 cache 压缩区", "source_checked"),
        _E("ev_comp_cache_write", m, "Compressor.forward cache write", 379, 382,
           "self.kv_cache[:bsz, :seqlen // ratio] = kv\nelse:\n    self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)",
           "压缩条目写入 cache：prefill 写 0..T//r-1，decode 写槽位 start_pos//r——写的是 slice alias（注意力 cache 压缩区）", "located"),
        # --- Indexer ---
        _E("ev_indexer_init", m, "Indexer.__init__", 386, 406,
           "self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)\nself.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)\nself.compressor = Compressor(args, compress_ratio, self.head_dim, True)\nself.register_buffer(\"kv_cache\", torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim), persistent=False)",
           "Indexer 自有 wq_b（与 attention wq_b 不同参数）、weights_proj、自有 Compressor（rotate=True）与独立 kv_cache（Mcap//r 条，头维 Di）", "located"),
        _E("ev_indexer_forward", m, "Indexer.forward", 408, 439,
           "def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):",
           "Indexer 输入 (x, qr, start_pos, offset)：x 供自有 compressor 与 weights_proj，qr 供 wq_b——两个真实输入", "located"),
        _E("ev_idx_q", m, "Indexer.forward q path", 417, 422,
           "q = self.wq_b(qr)\nq = q.unflatten(-1, (self.n_local_heads, self.head_dim))\napply_rotary_emb(q[..., -rd:], freqs_cis)\nq = rotate_activation(q)\nfp4_act_quant(q, fp4_block_size, True)",
           "Indexer Q：独立 wq_b(qr) → 展开 → 部分 RoPE → Hadamard 旋转 → FP4 量化模拟", "located"),
        _E("ev_idx_rotate", m, "rotate_activation", 253, 257,
           'from fast_hadamard_transform import hadamard_transform\nreturn hadamard_transform(x, scale=x.size(-1) ** -0.5)',
           "Hadamard 旋转调用语义（尺度 1/sqrt(dim)）已定位核对；外部库 kernel 数值过程 uncovered", "located",
           note="外部库 fast_hadamard_transform 的 kernel 数值未核对（uncovered 边界）；本条只声称 model.py 调用点定位"),
        _E("ev_idx_score", m, "Indexer.forward score", 424, 427,
           'index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])\nindex_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)',
           "打分：Q×压缩 KV（可见条目 end_pos//r）→ relu → 按 head 加权汇聚；产出标量分数而非 Value——输出是选择索引", "source_checked"),
        _E("ev_idx_causal", m, "Indexer.forward causal mask", 430, 432,
           "mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio\nindex_score += torch.where(mask, float(\"-inf\"), 0)",
           "prefill 因果掩码：位置 s 不可见条目 e ≥ (s+1)//r——因果可见性按 (s+1)//r 逐位置判定", "source_checked"),
        _E("ev_idx_topk", m, "Indexer.forward topk", 433, 438,
           'topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]',
           "有效选择数 = min(Ki, end_pos//r)：Ki 是上限不是每 query 有效数；条目不足时实际选择更少；-1 标记无效槽位；decode 直接加 offset", "source_checked"),
        # --- MoE ---
        _E("ev_gate_init", m, "Gate.__init__ topk branch", 556, 567,
           "self.hash = layer_id < args.n_hash_layers\nself.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))\nif self.hash:\n    self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)\nelse:\n    self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))",
           "layer 4 ≥ n_hash_layers=3 → TopK 门控（hash 层才有 tid2eid 查表）；TopK 层有选择偏置", "located"),
        _E("ev_gate_forward", m, "Gate.forward", 569, 589,
           "scores = linear(x.float(), self.weight.float())\noriginal_scores = scores\nif self.bias is not None:\n    scores = scores + self.bias\nindices = scores.topk(self.topk, dim=-1)[1]\nweights = original_scores.gather(1, indices)",
           "sqrtsoftplus 打分（571-576 F.softplus(scores).sqrt()）；偏置只加在选择用分数上，路由权重 gather 自 original_scores——偏置影响选择不影响权重", "source_checked"),
        _E("ev_weight_norm", m, "Gate.forward weight norm", 586, 588,
           "weights /= weights.sum(dim=-1, keepdim=True)\nweights *= self.route_scale",
           "非 softmax 计分：权重按选中项归一化再乘 route_scale=1.5（run_config.json:17）", "located"),
        _E("ev_expert", m, "Expert.forward SwiGLU", 601, 611,
           "gate = self.w1(x).float()\nup = self.w3(x).float()\nif self.swiglu_limit > 0:\n    up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)\n    gate = torch.clamp(gate, max=self.swiglu_limit)\nx = F.silu(gate) * up\nif weights is not None:\n    x = weights * x\nreturn self.w2(x.to(dtype))",
           "单专家 SwiGLU 模板（w1/w2/w3 + swiglu_limit 截断 + 权重乘）；256 个专家共用该模板，不逐份展开", "source_checked"),
        _E("ev_moe_dispatch", m, "MoE.forward dispatch", 634, 645,
           "x = x.view(-1, self.dim)\nweights, indices = self.gate(x, input_ids.flatten())\ncounts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()\nfor i in range(self.experts_start_idx, self.experts_end_idx):\n    if counts[i] == 0:\n        continue\n    expert = self.experts[i]\n    idx, top = torch.where(indices == i)\n    y[idx] += expert(x[idx], weights[idx, top, None])",
           "token 展平（N_tokens=B×T，model.py:636）→ gate → bincount 计数 → 按专家分发（counts[i]==0 跳过）；每专家 token 数 N_e 数据相关 unknown", "source_checked"),
        _E("ev_moe_shared", m, "MoE.forward shared expert", 648, 649,
           "y += self.shared_experts(x)\nreturn y.type_as(x).view(shape)",
           "shared expert（无路由权重乘）与路由专家输出相加后 view 回原形状汇合", "located"),
        # --- config / 派生轴 ---
        _E("ev_cfg_ratios", "run_config.json", "compress_ratios", 39, 39,
           '"compress_ratios": [0, 0, 4, 128, 4,',
           "compress_ratios[4]=4 → layer 4 为 CSA；[0]=0,[1]=0（SWA），[3]=128（HCA），[2]=4（CSA hash）", "located"),
        _E("ev_cfg_nhash", "run_config.json", "n_hash_layers", 6, 6,
           '"n_hash_layers": 3,',
           "n_hash_layers=3：层 0-2 Hash 门控，层 3-42 TopK 门控", "located"),
        _E("ev_cfg_route_scale", "run_config.json", "route_scale", 17, 17,
           '"route_scale": 1.5,',
           "route_scale=1.5（Gate 权重缩放）", "located"),
        _E("ev_compress_topk_fixed", m, "get_compress_topk_idxs", 274, 282,
           "def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):",
           "固定压缩位置选择（HCA 无学习式 Indexer 时使用，model.py:518-519 else 分支）："
           "prefill 按块序号因果掩码、decode 取已完成条目序号 + offset；-1 无效标记",
           "source_checked"),
        _E("ev_kernel_numerics", k, "tilelang JIT kernels", None, None, None,
           "kernel.py 的 tilelang JIT kernel（act_quant/fp8_gemm/fp4_gemm/"
           "sparse_attn_kernel/hc_split_sinkhorn_kernel）数值实现",
           "uncovered",
           note="全部 tilelang kernel 的数值实现未核对；本图只在调用边界消费其"
                "接口语义，不声称 kernel 内部语义已验证（不透明 kernel 边界）"),
        _E("ev_derived_d_nope", m, "nope_head_dim", None, None, None,
           "D_nope = D − D_rope（head_dim − rope_head_dim）", "derived",
           derivation="model.py:455 self.nope_head_dim = args.head_dim - args.rope_head_dim；512−64=448"),
        _E("ev_derived_d2", m, "compress proj width", None, None, None,
           "D2 = 2×D（overlap 时 coff=1+overlap=2，wkv/wgate 输出 coff×head_dim）", "derived",
           derivation="model.py:298 coff = 1 + self.overlap（ratio==4 → overlap=True → coff=2）；303-304 Linear(self.dim, coff * self.head_dim)"),
        _E("ev_derived_d_g", m, "group head dim", None, None, None,
           "D_g = Hq×D//G（wo_a 输入为每组紧凑头维 n_heads*head_dim//n_groups）", "derived",
           derivation="model.py:468 ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, ...)；64×512//8=4096"),
        _E("ev_derived_gro", m, "group lora flat", None, None, None,
           "GRo = G×Ro（einsum 输出 bsgr 展平最后一两维后为 G×Ro）", "derived",
           derivation="model.py:546-547 o=[b,s,G,Ro] → o.flatten(2)=[b,s,G*Ro] → wo_b；8×1024=8192"),
        _E("ev_derived_n_mix", m, "mix_hc", None, None, None,
           "N_mix = (2+N_hc)×N_hc = 24（hc_fn 参数第一维）", "derived",
           derivation="model.py:670 mix_hc = (2 + hc_mult) * hc_mult；hc_mult=4"),
        _E("ev_derived_n_cap", m, "kv_cache_size", None, None, None,
           "N_cap = W + Mcap//r = 128 + 1024 = 1140（注意力 cache 总槽位）", "derived",
           derivation="model.py:479 window_size + max_seq_len // compress_ratio；128 + 4096//4"),
        _E("ev_derived_n_cs", m, "compress state slots", None, None, None,
           "N_cs = coff×r = 2r = 8（kv_state/score_state 槽位；overlap 前 r 槽 + 当前 r 槽）", "derived",
           derivation="model.py:298 coff=1+overlap、309 register_buffer(... coff * compress_ratio ...)"),
        _E("ev_derived_n_vis", m, "indexer visible entries", None, None, None,
           "N_vis = (P+T)//r = end_pos//r（Indexer 打分可见的压缩条目数）", "derived",
           derivation="model.py:426 self.kv_cache[:bsz, :end_pos // ratio]，end_pos = start_pos + seqlen（413）"),
        _E("ev_derived_n_sel", m, "indexer select width", None, None, None,
           "N_sel_c = min(Ki, (P+T)//r)：每 query 压缩选择宽度 = min(上限 Ki, 可见条目)", "derived",
           derivation="model.py:433 index_score.topk(min(self.index_topk, end_pos // ratio), ...)"),
        _E("ev_derived_n_ctx", m, "prefill attention context", None, None, None,
           "N_ctx = T + T//r（prefill 拼接窗口 KV 与压缩 KV 后的上下文条目数）", "derived",
           derivation="model.py:530-531 kv = torch.cat([kv, kv_compress], dim=1)；kv [b,T,d] + kv_compress [b,T//r,d]"),
        _E("ev_derived_n_blk", m, "prefill fresh blocks", None, None, None,
           "N_blk = T//r（prefill 主序列 unflatten 成块数；remainder 尾块单独入状态）", "derived",
           derivation="model.py:343-344 kv.unflatten(1, (-1, ratio))，cutoff = seqlen - seqlen%ratio（334）"),
        _E("ev_derived_n_qf", m, "wq_b output width", None, None, None,
           "N_qf = Hq×D = 32768（wq_b 输出展平宽）", "derived",
           derivation="model.py:465 ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)"),
    )


# ---------------------------------------------------------------------------
# 轴：复用 rules 轴字典 + 图专属新轴
# ---------------------------------------------------------------------------

def _axes_from_rules(values: ModelValues) -> dict[str, NamedAxis]:
    from structure_reader.rules.deepseek_v4_flash_0731 import AXIS_DEFS
    axes: dict[str, NamedAxis] = {}
    for a in AXIS_DEFS:
        axes[a.symbol] = NamedAxis(
            symbol=a.symbol, name=a.name, meaning=a.meaning, origin=a.origin,
            evidence_ref=a.evidence_ref, value=a.value,
            expression=a.expression, unknown_reason=a.unknown_reason,
        )
    return axes


def _graph_axes(values: ModelValues, *,
                compress_ratio: int = 4) -> dict[str, NamedAxis]:
    """图轴集合；compress_ratio 决定 r 轴值与压缩派生轴适用性（V2 变体）。

    r 轴：CSA r=4 / HCA r=128（model.py:459 self.compress_ratio =
    args.compress_ratios[layer_id]）；SWA r=0 无压缩轴。
    N_cap/N_vis/N_sel 等压缩派生轴只对压缩变体有效（SWA 不引入除零）。
    """
    axes = _axes_from_rules(values)
    # 窗口选择索引物理宽（槽位数 ≠ 有效数）：四类变体共享（SWA 纯窗口也用，
    # model.py:261-271 get_window_topk_idxs 对所有层同源）
    axes["N_win"] = NamedAxis(
        "N_win", "window_index_width",
        "窗口选择索引物理宽（槽位数）：prefill min(T,W)、decode 恒 W——"
        "槽位数 ≠ 有效数（不足 -1 填充，model.py:261-271 "
        "get_window_topk_idxs）",
        "推导 model.py:261-271,268",
        "ev_window_topk", expression=EXPR_WINDOW_WIDTH)
    # r 轴按变体真实值（rules 的 _AXIS_R_CSA 是 CSA 代表层固定 4）
    if compress_ratio:
        axes["r"] = NamedAxis(
            symbol="r", name="compress_ratio",
            meaning=f"压缩比（compress_ratios[layer]={compress_ratio}）",
            origin="run config compress_ratios[layer]",
            evidence_ref="run_config.json:39", value=compress_ratio)
    else:
        axes.pop("r", None)  # SWA：无压缩轴
    hc_v = values.hc_mult
    # 块级轴（hc/mHC）所有变体都需要（SWA 也有双 mHC）
    axes["hc_j"] = NamedAxis("hc_j", "hc_source_stream", "mHC comb 源流下标 j",
                             "推导 model.py:692", "ev_hc_post", value=hc_v)
    axes["hc_k"] = NamedAxis("hc_k", "hc_target_stream", "mHC comb 目标流下标 k",
                             "推导 model.py:692", "ev_hc_post", value=hc_v)
    if not compress_ratio:
        # SWA：压缩派生轴全部不适用（不引入除零/无关 Ki 项）；
        # 块级轴（hc/N_qf 等）保留
        for sym in ("N_cap", "N_cs", "N_vis", "N_sel_c", "N_sel",
                    "N_cc", "N_rem", "D2", "D2i"):
            axes.pop(sym, None)
    elif compress_ratio != 4:
        # HCA（no-overlap，coff=1）：D2/D2i/N_cs 仅 CSA overlap 存在
        # （model.py:298 coff=1+overlap，ratio==4 → 2）；状态槽数即 r
        for sym in ("D2", "D2i", "N_cs"):
            axes.pop(sym, None)
    # 压缩派生轴按变体 coff 参数化含义（model.py:298 coff=1+overlap）：
    # CSA（overlap）→ D2=2*D / N_cs=2*r / D2i=2*Di；HCA（no-overlap）→
    # 投影/状态宽为 D（coff=1），无 D2/D2i/N_cs（状态槽数即 r）；
    # SWA → 全部压缩派生轴不适用（窗口 cache 容量即 W，无压缩项）。
    if compress_ratio:
        proj_w = "D2" if compress_ratio == 4 else "D"
        proj_w_expr = "2 * D" if compress_ratio == 4 else "D"
        extra = (
            NamedAxis("hc_j", "hc_source_stream", "mHC comb 源流下标 j", "推导 model.py:692",
                      "ev_hc_post", value=hc_v),
            NamedAxis("hc_k", "hc_target_stream", "mHC comb 目标流下标 k", "推导 model.py:692",
                      "ev_hc_post", value=hc_v),
            NamedAxis("N_cap", "attn_cache_capacity", "注意力 cache 总槽位（窗口+压缩区）",
                      "推导 model.py:479", "ev_derived_n_cap", expression="W + Mcap // r"),
            NamedAxis("N_vis", "compress_visible_entries", "压缩可见条目数 (P+T)//r"
                      + ("（学习式 Indexer 打分可见，model.py:413,426）" if compress_ratio == 4
                         else "（固定因果位置可见，get_compress_topk_idxs，model.py:274-282,518-519）"),
                      ("推导 model.py:413,426" if compress_ratio == 4
                       else "推导 model.py:274-282,518-519"),
                      "ev_derived_n_vis", expression="(P + T) // r"),
            NamedAxis("N_sel_c", "compress_select_width",
                      ("每 query 压缩选择宽度 min(Ki, N_vis)（学习式 topk，model.py:433；"
                       "Ki 是上限不是每 query 有效数）"
                       if compress_ratio == 4 else
                       "每 query 压缩选择物理宽（固定因果位置选择，无学习式 topk/"
                       "无 Ki 上限）：prefill T//r（逐 query 有效数 (s+1)//r 且"
                       " -1 掩码），decode (P+1)//r——model.py:274-282,518-519；"
                       "相位由 P 编码（P=0 prefill / P>0 decode，T=1）"),
                      ("推导 model.py:433" if compress_ratio == 4
                       else "推导 model.py:274-282,518-519"),
                      "ev_derived_n_sel",
                      expression=(EXPR_COMPRESS_WIDTH_LEARNED
                                  if compress_ratio == 4
                                  else EXPR_COMPRESS_WIDTH_FIXED)),
            NamedAxis("N_sel", "select_width_total",
                      ("每 query 总选择宽度 = 窗口选择物理宽 N_win + 压缩 N_sel_c"
                       "（model.py:268,520 cat 窗口+压缩索引）"
                       if compress_ratio == 4 else
                       "每 query 总选择物理宽 = 窗口 N_win（prefill min(T,W)，"
                       "decode W，model.py:268）+ 固定压缩 (P+T)//r"
                       "（prefill T//r，decode (P+1)//r，model.py:274-282）"),
                      ("推导 model.py:513,520" if compress_ratio == 4
                       else "推导 model.py:513,268,274-282"),
                      "ev_window_topk",
                      expression=(EXPR_SELECT_WIDTH_LEARNED
                                  if compress_ratio == 4
                                  else EXPR_SELECT_WIDTH_FIXED)),
            NamedAxis("N_cc", "compress_cache_capacity", "压缩 cache 容量 Mcap//r 条",
                      "推导 model.py:405,479", "ev_cache_capacity", expression="Mcap // r"),
            NamedAxis("N_rem", "prefill_remainder_len", "prefill 尾块长度 T%r（逐场景变化）",
                      "推导 model.py:333", "ev_comp_prefill", expression="T % r"),
            NamedAxis("N_one", "decode_single_token", "decode 每步单 token（seqlen=1）",
                      "推导 model.py:913,960（forward(input_ids, start_pos) 单 token 增量）",
                      "ev_decode_cache_read", value=1),
            NamedAxis("N_ctx", "prefill_attn_context", "prefill 注意力上下文条目数 T + T//r",
                      "推导 model.py:530-531", "ev_derived_n_ctx", expression="T + T // r"),
            NamedAxis("N_blk", "prefill_fresh_blocks", "prefill 新序列压缩块数 T//r",
                      "推导 model.py:334,343", "ev_derived_n_blk", expression="T // r"),
            NamedAxis("N_qf", "q_flat_width", "wq_b 输出展平宽度 Hq×D",
                      "推导 model.py:465", "ev_derived_n_qf", expression="Hq * D"),
        )
        if compress_ratio == 4:
            extra += (
                NamedAxis("N_cs", "compress_state_slots", "压缩状态槽位（overlap 2r）",
                          "推导 model.py:298-309", "ev_derived_n_cs", expression="2 * r"),
                NamedAxis("D2i", "indexer_compress_proj_dim", "Indexer 压缩投影宽度（overlap 时 2×Di）",
                          "推导 model.py:303-304,404（head_dim=Di）", "ev_derived_d2",
                          expression="2 * Di"),
            )
        for axis in extra:
            axes[axis.symbol] = axis
    else:
        # SWA：窗口语义轴（无压缩派生项；cache 容量 = W）
        extra = (
            NamedAxis("hc_j", "hc_source_stream", "mHC comb 源流下标 j", "推导 model.py:692",
                      "ev_hc_post", value=hc_v),
            NamedAxis("hc_k", "hc_target_stream", "mHC comb 目标流下标 k", "推导 model.py:692",
                      "ev_hc_post", value=hc_v),
            NamedAxis("N_qf", "q_flat_width", "wq_b 输出展平宽度 Hq×D",
                      "推导 model.py:465", "ev_derived_n_qf", expression="Hq * D"),
        )
        for axis in extra:
            axes[axis.symbol] = axis
    return axes


# ---------------------------------------------------------------------------
# 图构建器
# ---------------------------------------------------------------------------

class _GraphBuilder:
    """累积实体并做即时一致性检查；build() 时跑通用引用检查器。"""

    def __init__(self, axes: dict[str, NamedAxis]) -> None:
        self._axes = axes
        self._tensors: dict[str, Tensor] = {}
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        self._edge_list: list[Edge] = []
        self._versions: list[StateVersion] = []
        self._transitions: list[StateTransition] = []
        self._aliases: list[AliasRelation] = []
        self._fold_groups: list[FoldGroup] = []
        self._evidence_ids: set[str] = set()

    def require_evidence(self, evidence_id: str) -> str:
        self._evidence_ids.add(evidence_id)
        return evidence_id

    def axis(self, symbol: str) -> NamedAxis:
        return self._axes[symbol]

    def remove_tensor(self, name: str) -> None:
        """构建期剥离张量（变体过滤：该变体不存在的角色）。仅用于
        _add_tensors 之后、flow 构建之前的过滤窗口；若已有边/节点引用
        该张量则拒绝（防止剥离产生悬空）。"""
        tid = f"tensor:{name}"
        if tid not in self._tensors:
            return
        for e in self._edge_list:
            if e.tensor_id == tid:
                raise ExtractionError(
                    f"cannot remove tensor {name}: edge {e.stable_id} 引用它")
        del self._tensors[tid]

    def tensor(self, name: str, dtype: str, axis_symbols: tuple[str, ...],
               evidence: tuple[str, ...] = ()) -> str:
        # Tensor 无 evidence_refs 字段（证据在各轴定义与节点/边上）；
        # evidence 参数仅用于 require_evidence 登记，保证引用完整性。
        for ev_id in evidence:
            self.require_evidence(ev_id)
        axes = [self._axes[s] for s in axis_symbols]
        symbols = list(axis_symbols)
        t = Tensor(name=name, dtype=dtype, axes=axes,
                   logical_layout=symbols, physical_layout=symbols)
        tid = t.stable_id
        if tid in self._tensors:
            raise ExtractionError(f"张量名重复: {name}")
        self._tensors[tid] = t
        return tid

    def node(self, stable_id: str, op_type: str, granularity: str,
             inputs: tuple[tuple[str, str], ...] = (),
             outputs: tuple[tuple[str, str], ...] = (),
             evidence: tuple[str, ...] = ()) -> str:
        node = Node(
            stable_id=stable_id, op_type=op_type, granularity=granularity,
            input_ports=[Port(n, "input", tid) for n, tid in inputs],
            output_ports=[Port(n, "output", tid) for n, tid in outputs],
            evidence_refs=evidence,
        )
        if node.stable_id in self._nodes:
            raise ExtractionError(f"节点 ID 重复: {stable_id}")
        self._nodes[node.stable_id] = node
        return node.stable_id

    def edge(self, stable_id: str, src: tuple[str, str], dst: tuple[str, str],
             tensor_id: str, kind: str = "data", *, guard_ref: str | None = None,
             state_transition_ref: str | None = None,
             evidence: tuple[str, ...] = ()) -> str:
        e = Edge(stable_id=stable_id, src_node_id=src[0], src_port=src[1],
                 dst_node_id=dst[0], dst_port=dst[1], tensor_id=tensor_id,
                 kind=kind, guard_ref=guard_ref,
                 state_transition_ref=state_transition_ref, evidence_refs=evidence)
        if e.stable_id in self._edges:
            raise ExtractionError(f"边 ID 重复: {stable_id}")
        self._edges[e.stable_id] = e
        self._edge_list.append(e)
        return e.stable_id

    def state(self, state_id: str, transition_id: str, evidence: tuple[str, ...],
              premise: str, *, pool_shift: bool = False) -> None:
        """一个状态 + 本次调用更新的版本序列。

        默认两版本：v_hist（历史累积）→ v_after（本调用后）。
        pool_shift=True（压缩状态）：三版本——v_hist（历史）→ v_updated
        （本调用写入后、待池化）→ v_final（触发池化+shift 后最终版本，
        连续调用的下一历史）；transition 表达 hist→updated，shift 语义由
        pool_shift 节点与 final 版本承载（model.py:353-360）。
        """
        before = StateVersion(stable_id=f"state:{state_id}:v_hist",
                              state_id=state_id, version_label="v_hist")
        after = StateVersion(stable_id=f"state:{state_id}:v_updated"
                             if pool_shift else f"state:{state_id}:v_after",
                             state_id=state_id,
                             version_label="v_updated" if pool_shift else "v_after")
        self._versions.extend([before, after])
        if pool_shift:
            final = StateVersion(stable_id=f"state:{state_id}:v_final",
                                 state_id=state_id, version_label="v_final")
            self._versions.append(final)
            self._transitions.append(StateTransition(
                stable_id=f"{transition_id}_shift", state_id=state_id,
                before_version_id=after.stable_id,
                after_version_id=final.stable_id,
                evidence_refs=evidence,
            ))
        self._transitions.append(StateTransition(
            stable_id=transition_id, state_id=state_id,
            before_version_id=before.stable_id,
            after_version_id=after.stable_id,
            evidence_refs=evidence,
        ))

    def alias(self, stable_id: str, *, relation_kind: str, tensor_id: str,
              alias_role: str | None = None, transform_expr: str | None = None,
              other_tensor_id: str | None = None, evidence: tuple[str, ...] = ()) -> None:
        base = tensor_id
        alias = tensor_id if relation_kind == "same_tensor" else other_tensor_id
        self._aliases.append(AliasRelation(
            stable_id=stable_id, base_tensor_id=base, alias_tensor_id=alias,
            relation_kind=relation_kind, alias_role=alias_role,
            transform_expr=transform_expr, evidence_refs=evidence,
        ))

    def fold_group(self, stable_id: str, member_ids: tuple[str, ...],
                   mappings: tuple[FoldBoundaryMapping, ...]) -> None:
        self._fold_groups.append(FoldGroup(stable_id=stable_id,
                                           member_node_ids=member_ids,
                                           boundary_mappings=mappings))

    def build(self, guards: tuple[Guard, ...], template_id: str) -> SemanticGraph:
        graph = SemanticGraph(
            template_id=template_id,
            named_axes=list(self._axes.values()),
            tensors=list(self._tensors.values()),
            nodes=list(self._nodes.values()),
            edges=self._edge_list,
            state_versions=self._versions,
            state_transitions=self._transitions,
            guards=list(guards),
            alias_relations=self._aliases,
            fold_groups=self._fold_groups,
        )
        issues = validate_graph_references(graph)
        if issues:
            detail = "; ".join(f"{i.code}@{i.path}: {i.message}" for i in issues[:10])
            raise ExtractionError(f"通用引用检查未通过（{len(issues)} 项）: {detail}")
        return graph

    @property
    def evidence_ids(self) -> set[str]:
        return set(self._evidence_ids)



# ---------------------------------------------------------------------------
# 张量定义（layer 4 CSA·TopK 模板的全部中间张量）
# ---------------------------------------------------------------------------

def _add_tensors(b: _GraphBuilder, values: ModelValues, *,
                 tprefix: str = "csa_topk",
                 compress_ratio: int = 4) -> dict[str, str]:
    """按连接需求定义张量；返回 张量角色名 → stable_id。

    tprefix 区分模板（联合导出/页面不碰撞）；compress_ratio 决定压缩
    轴语义：CSA r=4 overlap → coff=2（投影宽 D2=2D、状态槽 N_cs=2r）；
    HCA r=128 无 overlap → coff=1（投影宽 D、状态槽 r=128）；
    SWA r=0 无压缩（压缩张量由变体剥离）。轴 symbol 按变体真实选择，
    不靠 spec 元数据声明。
    """
    t: dict[str, str] = {}

    def add(role: str, dtype: str, axes: tuple[str, ...],
            evidence: tuple[str, ...] = ()) -> str:
        tid = b.tensor(f"{tprefix}:{role}", dtype, axes, evidence)
        t[role] = tid
        return tid

    # 压缩轴按变体真实选择（model.py:298 coff = 1 + overlap；296 overlap = ratio==4）
    overlap = compress_ratio == 4
    coff = 2 if overlap else 1
    if compress_ratio:
        proj_w = "D2" if overlap else "D"        # wkv/wgate 投影宽
        state_w = "D2" if overlap else "D"       # kv_state/score_state 槽宽
        n_cs = "N_cs" if overlap else "r"        # 状态槽数：coff*r（2r）或 r
        n_cs_val = 2 * compress_ratio if overlap else compress_ratio
        proj_sym = "D2" if overlap else "D"

    # 块级 / hc_pre（attention 侧）
    add("x_hc_in", "bf16", ("B", "T", "N_hc", "H"), ())
    add("x_hc_attn_flat", "fp32", ("B", "T", "N_hcH"), ("ev_hc_pre",))
    add("attn_mixes", "fp32", ("B", "T", "N_mix"), ("ev_hc_pre",))
    add("attn_pre", "fp32", ("B", "T", "N_hc"), ("ev_hc_pre",))
    add("attn_post", "fp32", ("B", "T", "N_hc"), ("ev_hc_pre",))
    add("attn_comb", "fp32", ("B", "T", "hc_j", "hc_k"), ("ev_hc_post",))
    add("x_attn_in", "bf16", ("B", "T", "H"), ("ev_hc_pre",))
    add("x_attn_normed", "bf16", ("B", "T", "H"), ("ev_attn_norm",))
    # Q 路径
    add("qr", "bf16", ("B", "T", "Rq"), ("ev_q_path",))
    add("q_flat", "bf16", ("B", "T", "N_qf"), ("ev_q_path",))
    add("q", "bf16", ("B", "T", "Hq", "D"), ("ev_q_path",))
    add("q_rope", "bf16", ("B", "T", "Hq", "D"), ("ev_q_path",))
    # 共享 KV 单投影（一个 wkv、K/V 同一张量）
    add("kv_single", "bf16", ("B", "T", "D"), ("ev_kv_path",))
    add("kv_normed", "bf16", ("B", "T", "D"), ("ev_kv_path",))
    add("kv_quant", "bf16", ("B", "T", "D"), ("ev_kv_path",))
    # cache 与选择索引
    add("win_cache", "bf16", ("B", "W", "D"), ("ev_prefill_cache_write",))
    if compress_ratio:
        # 压缩变体：窗口+压缩区联合 cache（model.py:479 W + Mcap//r）；
        # SWA cache 只有窗口槽（win_cache），无 N_cap/压缩选择索引
        add("attn_cache", "bf16", ("B", "N_cap", "D"), ("ev_cache_capacity",))
        add("comp_idxs", "int32", ("B", "T", "N_sel_c"), ("ev_idx_topk",))
        add("select_idxs", "int32", ("B", "T", "N_sel"), ("ev_indexer_call",))
    # 窗口选择索引：物理宽 N_win（prefill min(T,W) / decode W 槽位，
    # model.py:261-271——槽位数 ≠ 有效数，不足 -1 填充）；SWA 同源
    add("win_idxs", "int32", ("B", "T", "N_win"), ("ev_window_topk",))
    # attn_sink：独立可学习参数 [Hq]（model.py:462），不是 hc sinkhorn 的 post
    add("attn_sink_param", "fp32", ("Hq",), ("ev_attn_init",))
    # attention Compressor：输入 H（args.dim）；投影/状态宽与槽数按变体
    # coff（CSA overlap=2 → D2/2r；HCA no-overlap=1 → D/r）
    if compress_ratio:
        add("comp_kv_proj", "fp32", ("B", "T", proj_sym), ("ev_comp_init",))
        add("comp_score_proj", "fp32", ("B", "T", proj_sym), ("ev_comp_init",))
        add("comp_ape", "fp32", ("r", proj_sym), ("ev_comp_init",))
        # blockify（unflatten）后、overlap 前仍是 r 槽 × coff·D 宽
        #（model.py:343-344）
        add("comp_blocks", "fp32", ("B", "N_blk", "r", proj_sym),
            ("ev_comp_prefill",))
        if overlap:
            # overlap 变换后才 [B, N_blk, 2r, D]（model.py:313-320）
            add("comp_overlap", "fp32", ("B", "N_blk", "N_cs", "D"),
                ("ev_overlap_transform",))
        add("comp_pooled", "fp32", ("B", "N_blk", "D"), ("ev_comp_prefill",))
        add("comp_normed", "bf16", ("B", "N_blk", "D"), ("ev_comp_out",))
        add("comp_out", "bf16", ("B", "N_blk", "D"), ("ev_comp_out",))
    if compress_ratio:
        # 本次调用新压缩条目（prefill 拼接进 sparse_attn / 写压缩区）
        add("comp_return", "bf16", ("B", "N_blk", "D"), ("ev_comp_cache_write",))
        # 主/尾块切片 view（model.py:337-341）：主序列末块 cutoff-r:cutoff、
        # 尾块 cutoff:；decode 当前 token 单独路径。宽随 coff（CSA D2 / HCA D）
        add("comp_main_slice", "fp32", ("B", "r", proj_sym), ("ev_comp_prefill",))
        add("comp_tail_slice", "fp32", ("B", "N_rem", proj_sym), ("ev_comp_prefill",))
        add("comp_main_score_slice", "fp32", ("B", "r", proj_sym), ("ev_comp_prefill",))
        add("comp_tail_score_slice", "fp32", ("B", "N_rem", proj_sym), ("ev_comp_prefill",))
        add("comp_dec_token", "fp32", ("B", "N_one", proj_sym), ("ev_comp_decode",))
        add("comp_dec_token_score", "fp32", ("B", "N_one", proj_sym), ("ev_comp_decode",))
        add("comp_main_score_aped", "fp32", ("B", "r", proj_sym), ("ev_comp_prefill",))
        add("comp_tail_score_aped", "fp32", ("B", "N_rem", proj_sym), ("ev_comp_prefill",))
        add("comp_dec_score_aped", "fp32", ("B", "N_one", proj_sym), ("ev_comp_decode",))
        add("comp_ape_tail_rows", "fp32", ("N_rem", proj_sym), ("ev_comp_prefill",))
        add("comp_ape_dec_row", "fp32", ("N_one", proj_sym), ("ev_comp_decode",))
        # 压缩增量状态：hist → updated → final；槽数/宽按 coff
        add("comp_state_kv_hist", "fp32", ("B", n_cs, state_w), ("ev_comp_states",))
        add("comp_state_score_hist", "fp32", ("B", n_cs, state_w), ("ev_comp_states",))
        add("comp_state_kv_updated", "fp32", ("B", n_cs, state_w), ("ev_comp_states",))
        add("comp_state_score_updated", "fp32", ("B", n_cs, state_w), ("ev_comp_states",))
        add("comp_state_kv_final", "fp32", ("B", n_cs, state_w), ("ev_comp_states",))
        add("comp_state_score_final", "fp32", ("B", n_cs, state_w), ("ev_comp_states",))
        # 压缩区容量 Mcap//r 条（注意力 cache[:, W:] 视图，model.py:479,497）
        add("comp_cache", "bf16", ("B", "N_cc", "D"), ("ev_comp_cache_write",))
        # prefill 拼接后的注意力上下文 KV（当前窗口 + 本次新压缩条目）
        add("kv_concat", "bf16", ("B", "N_ctx", "D"), ("ev_prefill_compress",))
    if compress_ratio and overlap:
        # Indexer 自有 Compressor（仅 CSA：head_dim=Di、rotate=True，model.py:404；
        # 与 attention 同 overlap 语义）
        add("idx_comp_kv_proj", "fp32", ("B", "T", "D2i"), ("ev_comp_init", "ev_indexer_init"))
        add("idx_comp_score_proj", "fp32", ("B", "T", "D2i"), ("ev_comp_init", "ev_indexer_init"))
        add("idx_comp_ape", "fp32", ("r", "D2i"), ("ev_comp_init", "ev_indexer_init"))
        add("idx_comp_blocks", "fp32", ("B", "N_blk", "r", "D2i"), ("ev_comp_prefill",))
        add("idx_comp_overlap", "fp32", ("B", "N_blk", "N_cs", "Di"), ("ev_overlap_transform",))
        add("idx_comp_pooled", "fp32", ("B", "N_blk", "Di"), ("ev_comp_prefill",))
        add("idx_comp_normed", "bf16", ("B", "N_blk", "Di"), ("ev_comp_out",))
        add("idx_comp_out", "bf16", ("B", "N_blk", "Di"), ("ev_comp_out",))
        add("idx_comp_main_slice", "fp32", ("B", "r", "D2i"), ("ev_comp_prefill",))
        add("idx_comp_tail_slice", "fp32", ("B", "N_rem", "D2i"), ("ev_comp_prefill",))
        add("idx_comp_main_score_slice", "fp32", ("B", "r", "D2i"), ("ev_comp_prefill",))
        add("idx_comp_tail_score_slice", "fp32", ("B", "N_rem", "D2i"), ("ev_comp_prefill",))
        add("idx_comp_dec_token", "fp32", ("B", "N_one", "D2i"), ("ev_comp_decode",))
        add("idx_comp_dec_token_score", "fp32", ("B", "N_one", "D2i"), ("ev_comp_decode",))
        add("idx_comp_main_score_aped", "fp32", ("B", "r", "D2i"), ("ev_comp_prefill",))
        add("idx_comp_tail_score_aped", "fp32", ("B", "N_rem", "D2i"), ("ev_comp_prefill",))
        add("idx_comp_dec_score_aped", "fp32", ("B", "N_one", "D2i"), ("ev_comp_decode",))
        add("idx_comp_ape_tail_rows", "fp32", ("N_rem", "D2i"), ("ev_comp_prefill",))
        add("idx_comp_ape_dec_row", "fp32", ("N_one", "D2i"), ("ev_comp_decode",))
        add("idx_comp_state_kv_hist", "fp32", ("B", "N_cs", "D2i"), ("ev_comp_states", "ev_indexer_init"))
        add("idx_comp_state_score_hist", "fp32", ("B", "N_cs", "D2i"), ("ev_comp_states", "ev_indexer_init"))
        add("idx_comp_state_kv_updated", "fp32", ("B", "N_cs", "D2i"), ("ev_comp_states", "ev_indexer_init"))
        add("idx_comp_state_score_updated", "fp32", ("B", "N_cs", "D2i"), ("ev_comp_states", "ev_indexer_init"))
        add("idx_comp_state_kv_final", "fp32", ("B", "N_cs", "D2i"), ("ev_comp_states", "ev_indexer_init"))
        add("idx_comp_state_score_final", "fp32", ("B", "N_cs", "D2i"), ("ev_comp_states", "ev_indexer_init"))
        add("idx_comp_return", "bf16", ("B", "N_blk", "Di"), ("ev_comp_cache_write",))
        add("idx_cache", "bf16", ("B", "N_cc", "Di"), ("ev_indexer_init",))
        add("idx_cache_visible", "bf16", ("B", "N_vis", "Di"), ("ev_idx_score",))
        add("idx_q_flat", "bf16", ("B", "T", "Hi", "Di"), ("ev_idx_q",))
        add("idx_q", "bf16", ("B", "T", "Hi", "Di"), ("ev_idx_q",))
        add("idx_weights", "bf16", ("B", "T", "Hi"), ("ev_indexer_init",))
        add("idx_scores", "fp32", ("B", "T", "Hi", "N_vis"), ("ev_idx_score",))
        add("idx_score", "fp32", ("B", "T", "N_vis"), ("ev_idx_score",))
        add("idx_score_masked", "fp32", ("B", "T", "N_vis"), ("ev_idx_causal",))
    # hc_pre（ffn 侧）
    add("x_attn_post", "bf16", ("B", "T", "N_hc", "H"), ("ev_hc_post",))
    add("x_ffn_flat_hc", "fp32", ("B", "T", "N_hcH"), ("ev_hc_pre",))
    add("ffn_mixes", "fp32", ("B", "T", "N_mix"), ("ev_hc_pre",))
    add("ffn_pre", "fp32", ("B", "T", "N_hc"), ("ev_hc_pre",))
    add("ffn_post", "fp32", ("B", "T", "N_hc"), ("ev_hc_pre",))
    add("ffn_comb", "fp32", ("B", "T", "hc_j", "hc_k"), ("ev_hc_post",))
    add("x_ffn", "bf16", ("B", "T", "H"), ("ev_hc_pre",))
    add("x_ffn_normed", "bf16", ("B", "T", "H"), ("ev_attn_norm",))
    # attention 输出侧
    add("o_rope", "bf16", ("B", "T", "Hq", "D"), ("ev_rope_inv",))
    add("o_grouped", "bf16", ("B", "T", "G", "D_g"), ("ev_o_group",))
    add("o_lora", "bf16", ("B", "T", "G", "Ro"), ("ev_o_group",))
    add("o_flat", "bf16", ("B", "T", "GRo"), ("ev_o_group",))
    add("attn_out", "bf16", ("B", "T", "H"), ("ev_o_group",))
    # MoE
    add("gate_scores", "fp32", ("B", "T", "E"), ("ev_gate_forward",))
    add("gate_scores_biased", "fp32", ("B", "T", "E"), ("ev_gate_forward",))
    add("gate_indices", "int64", ("B", "T", "K"), ("ev_gate_forward",))
    add("gate_weights", "fp32", ("B", "T", "K"), ("ev_gate_forward",))
    add("n_e", "int64", ("E",), ())  # 每专家 token 数：数据相关 unknown（轴 N_e）
    add("expert_out", "fp32", ("B", "T", "H"), ("ev_moe_dispatch",))
    add("shared_out", "fp32", ("B", "T", "H"), ("ev_moe_shared",))
    add("moe_out", "bf16", ("B", "T", "H"), ("ev_moe_shared",))
    add("x_hc_out", "bf16", ("B", "T", "N_hc", "H"), ())
    return t


# ---------------------------------------------------------------------------
# 节点与边（按 Block.forward 真实控制流）
# ---------------------------------------------------------------------------

def _add_flow(b: _GraphBuilder, t: dict[str, str], values: ModelValues) -> None:
    def node(sid: str, op: str, gr: str, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid: str, src, dst, role: str, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], kind, **kw)

    # ===== hc_pre（attention 侧）=====
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

    # ===== attn_norm =====
    node("attn_norm", "rms_norm", "parameter_op",
         (("x", t["x_attn_in"]),), (("y", t["x_attn_normed"]),),
         ("ev_attn_norm",))

    # ===== Q 路径（qr 分叉给 wq_b 与 Indexer）=====
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

    # ===== 共享 KV 单投影 =====
    node("attn.wkv", "linear", "parameter_op",
         (("x", t["x_attn_normed"]),), (("y", t["kv_single"]),), ("ev_kv_path",))
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
    # K/V 共享单投影：两个消费角色引用同一 kv_quant（same_tensor alias）
    b.alias("alias_kv_key", relation_kind="same_tensor",
            tensor_id=t["kv_quant"], alias_role="key",
            evidence=("ev_kv_path",))
    b.alias("alias_kv_value", relation_kind="same_tensor",
            tensor_id=t["kv_quant"], alias_role="value",
            evidence=("ev_kv_path",))

    # ===== 窗口选择 =====
    node("attn.win_topk", "window_index", "index_op",
         (), (("y", t["win_idxs"]),), ("ev_window_topk",))

    # ===== attention Compressor =====
    # 输入是 attn_norm 输出（args.dim，model.py:303-304），与窗口 wkv 并行的
    # 另一套投影（wkv/wgate 各自 Linear(H, 2D)），不串在窗口 wkv 之后。
    _add_compressor(b, t, values, prefix="attn.compressor",
                    x_role="x_attn_normed", cache_role="comp_cache",
                    role_prefix="comp_", rotate=False,
                    tr_kv="tr_comp_kv", tr_score="tr_comp_score",
                    tr_kv_shift="tr_comp_kv_shift",
                    tr_score_shift="tr_comp_score_shift")
    edge("e_comp_in", ("attn_norm", "y"), ("attn.compressor.wkv", "x"),
         "x_attn_normed")
    edge("e_comp_in_gate", ("attn_norm", "y"), ("attn.compressor.wgate", "x"),
         "x_attn_normed")
    # 压缩区 slice alias：compressor cache 是注意力 cache[:, W:] 视图（同存储）
    b.alias("alias_comp_cache_slice", relation_kind="slice",
            tensor_id=t["attn_cache"], other_tensor_id=t["comp_cache"],
            transform_expr="attn_cache[:, W:]（压缩区视图，同存储非独立拷贝）",
            evidence=("ev_lazy_bind",))

    # ===== Indexer（x 与 qr 两个真实输入）=====
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
    # 自有 Compressor（head_dim=Di、rotate=True，model.py:404）：输入 x 独立实例
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
    # Indexer 独立 cache 是自有存储（不同注意力 cache，model.py:405 独立 buffer）
    node("attn.indexer.weights_proj", "linear", "parameter_op",
         (("x", t["x_attn_normed"]),), (("y", t["idx_weights"]),),
         ("ev_indexer_init",))
    edge("e_idx_wproj", ("attn_norm", "y"),
         ("attn.indexer.weights_proj", "x"), "x_attn_normed")
    # 可见前缀视图：打分只读 cache[:end_pos//r]（model.py:426）——
    # 真实 slice 节点从 idx_cache 产出 N_vis 前缀，score 消费该张量
    node("attn.indexer.cache_slice", "slice_view", "tensor_op",
         (("x", t["idx_cache"]),),
         (("y", t["idx_cache_visible"]),), ("ev_idx_score",))
    edge("e_idx_cache_slice", ("attn.indexer.compressor.cache_out", "cache"),
         ("attn.indexer.cache_slice", "x"), "idx_cache")
    # slice 读取挂 Indexer cache 状态（写后读，同 state_id）
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
    # prefill 因果掩码（model.py:430-432，g_causal_mask）与 decode 无掩码直连
    # （model.py:437-438 直接 topk+offset）按 guard 分支。
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
    # ===== cache 更新与读取（单一 state_id 的版本对：写 → 读）=====
    # model.py:523-528/535 写窗口槽 + 压缩条目写压缩区（slice alias 同存储）；
    # model.py:533 prefill 读当前 kv_quant；539 decode 读整个更新后 cache。
    node("attn.cache_update", "cache_write", "state_op",
         (("kv", t["kv_quant"]), ("comp_out", t["comp_cache"])),
         (("cache_full", t["attn_cache"]),),
         ("ev_prefill_cache_write", "ev_decode_cache_read",
          "ev_comp_cache_write"))
    edge("e_kv_cache", ("attn.kv_quant", "y"),
         ("attn.cache_update", "kv"), "kv_quant")
    # 压缩条目写入进同一 cache_update（压缩区 slice alias 语义由 alias 表达）
    edge("e_comp_cache_w", ("attn.compressor.cache_out", "cache"),
         ("attn.cache_update", "comp_out"), "comp_cache",
         guard_ref="g_compress_prefill")
    edge("e_comp_cache_w_dec", ("attn.compressor.cache_out", "cache"),
         ("attn.cache_update", "comp_out"), "comp_cache",
         guard_ref="g_compress_decode")
    # state 写读边（同一条边表达 v_hist → v_after 的写后读，
    # model.py:535 写 → 538-539 读；生产者 cache_update → 消费者 cache_read）
    edge("e_cache_state_w", ("attn.cache_update", "cache_full"),
         ("attn.cache_read", "cache_full"), "attn_cache", kind="state",
         state_transition_ref="tr_attn_cache")

    # ===== 选择索引拼接（窗口 + 压缩）=====
    node("attn.select_cat", "concat", "tensor_op",
         (("win", t["win_idxs"]), ("compress", t["comp_idxs"])),
         (("y", t["select_idxs"]),), ("ev_indexer_call",))
    edge("e_cat_win", ("attn.win_topk", "y"), ("attn.select_cat", "win"),
         "win_idxs")
    edge("e_cat_comp", ("attn.indexer.topk", "indices"),
         ("attn.select_cat", "compress"), "comp_idxs")
    # Indexer 产出选择索引（select 语义），不是 Value
    edge("e_idx_select", ("attn.indexer.topk", "indices"),
         ("attn.sparse_attn", "compress_idxs"), "comp_idxs", kind="select")

    # ===== cache 读与上下文 KV 拼接 =====
    # decode：cache 读出（含本调用写入）直接供 kv_win（model.py:539）
    node("attn.cache_read", "cache_read", "state_op",
         (("cache_full", t["attn_cache"]),),
         (("kv_decode", t["kv_concat"]),),
         ("ev_decode_cache_read",))
    edge("e_sparse_kv_decode", ("attn.cache_read", "kv_decode"),
         ("attn.sparse_attn", "kv_win"), "kv_concat", kind="data",
         guard_ref="g_decode")
    # prefill：当前 kv_quant + 触发时的压缩返回值 → kv_concat（model.py:530-531）
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

    # ===== sparse_attn（不透明 kernel）=====
    node("attn.sparse_attn", "sparse_attention", "opaque_kernel",
         (("q", t["q_rope"]), ("kv_win", t["kv_concat"]),
          ("win_idxs", t["win_idxs"]),
          ("compress_idxs", t["comp_idxs"]), ("topk_idxs", t["select_idxs"]),
          ("sink", t["attn_sink_param"])),
         (("o", t["o_rope"]),),
         ("ev_sparse_attn_kernel", "ev_prefill_compress",
          "ev_decode_cache_read"))
    edge("e_sparse_q", ("attn.q_rope", "y"), ("attn.sparse_attn", "q"), "q_rope")
    edge("e_sparse_winidx", ("attn.win_topk", "y"),
         ("attn.sparse_attn", "win_idxs"), "win_idxs")
    edge("e_sparse_topk", ("attn.select_cat", "y"),
         ("attn.sparse_attn", "topk_idxs"), "select_idxs")
    # sink 是独立参数 [Hq]（model.py:462），与 hc post 无关
    node("attn.attn_sink", "attn_sink_param", "parameter_op",
         (), (("y", t["attn_sink_param"]),), ("ev_attn_init",))
    edge("e_sparse_sink", ("attn.attn_sink", "y"),
         ("attn.sparse_attn", "sink"), "attn_sink_param")

    # ===== inverse RoPE 先于 O 分组低秩 =====
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

    # ===== hc_post（attention 残差至 post）=====
    node("block.hc_post_attn.combine", "hc_combine", "tensor_op",
         (("x", t["attn_out"]), ("post", t["attn_post"]),
          ("comb", t["attn_comb"]), ("residual", t["x_hc_in"])),
         (("y", t["x_attn_post"]),), ("ev_hc_post",))
    edge("e_wob_post", ("attn.wo_b", "y"),
         ("block.hc_post_attn.combine", "x"), "attn_out")
    edge("e_residual_a", ("block.hc_pre_attn.flatten", "residual"),
         ("block.hc_post_attn.combine", "residual"), "x_hc_in")

    # ===== hc_pre（ffn 侧）=====
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

    # ===== ffn_norm =====
    node("ffn_norm", "rms_norm", "parameter_op",
         (("x", t["x_ffn"]),), (("y", t["x_ffn_normed"]),),
         ("ev_attn_norm",))

    # ===== MoE TopK =====
    _add_moe(b, t, values)

    # ===== hc_post（ffn 残差至 post）=====
    node("block.hc_post_ffn.combine", "hc_combine", "tensor_op",
         (("x", t["moe_out"]), ("post", t["ffn_post"]),
          ("comb", t["ffn_comb"]), ("residual", t["x_attn_post"])),
         (("y", t["x_hc_out"]),), ("ev_hc_post",))
    edge("e_moe_post", ("moe.out", "y"),
         ("block.hc_post_ffn.combine", "x"), "moe_out")
    edge("e_residual_f", ("block.hc_pre_ffn.flatten", "residual"),
         ("block.hc_post_ffn.combine", "residual"), "x_attn_post")


def _add_compressor(b: _GraphBuilder, t: dict[str, str], values: ModelValues,
                    prefix: str, *, x_role: str, cache_role: str,
                    role_prefix: str, rotate: bool,
                    tr_kv: str, tr_score: str,
                    tr_kv_shift: str, tr_score_shift: str,
                    overlap: bool = True) -> None:
    """Compressor 模板（attention 与 indexer 两实例），状态 I/O 真实连通。

    源码事实（model.py:285-383）：
    - wkv/wgate 输入 dim=H（与窗口 wkv 并行的两套投影）；
    - 状态读取（hist）→ 本次投影/APE 更新槽位（state_update）→ 更新后版本
      （after）；触发时 decode 池化从状态读历史+当前（350-360），prefill 池化
      走 blockify/overlap（331-348）；prefill remainder 暂存进状态（339-341）；
    - 未触发（366-367 return None）：状态更新照常、cache 写/返回值不产出
      ——cache_out 与 return 的输入边挂触发 guard；
    - rotate=True（Indexer）走 Hadamard+FP4，否则非 RoPE 维 FP8。
    输入通过 x_role 由调用方边连接。
    """
    p = role_prefix
    def node(sid: str, op: str, gr: str, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(f"{prefix}.{sid}", op, gr, inputs, outputs, ev)

    def edge(sid: str, src, dst, role: str, kind="data", **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(f"{prefix}.{sid}", (f"{prefix}.{src[0]}", src[1]),
                      (f"{prefix}.{dst[0]}", dst[1]), t[role], kind, **kw)

    quant_op = "hadamard_fp4_quant" if rotate else "act_quant_fp8"
    quant_ev = ("ev_comp_out", "ev_idx_rotate") if rotate else ("ev_comp_out",)

    node("wkv", "linear", "parameter_op",
         (("x", t[x_role]),), (("y", t[f"{p}kv_proj"]),), ("ev_comp_init",))
    node("wgate", "linear", "parameter_op",
         (("x", t[x_role]),), (("y", t[f"{p}score_proj"]),), ("ev_comp_init",))
    node("ape", "additive_positional_embed", "parameter_op",
         (), (("y", t[f"{p}ape"]),), ("ev_comp_init",))
    # 历史状态读出（v_hist 版本锚点：无入边、两条 state 出边带 transition 引用）
    node("state_hist", "state_read", "state_op",
         (("next_hist", t[f"{p}state_kv_final"]),
          ("next_hist_score", t[f"{p}state_score_final"])),
         (("kv_state", t[f"{p}state_kv_hist"]),
          ("score_state", t[f"{p}state_score_hist"])),
         ("ev_comp_states", "ev_comp_decode"))
    # 主/尾块真实切片（model.py:337-341）：主末块取 kv[cutoff-r:cutoff]、
    # 尾块取 kv[cutoff:]；slice 节点消费完整投影、产出 view 张量。
    # prefill 主块切片挂 g_compress_prefill（cutoff>=r 才有）；尾块切片挂
    # g_compress_prefill（remainder>0 才有）；decode 当前 token 用独立节点
    # （start_pos 单槽）。
    node("main_slice", "slice_view", "tensor_op",
         (("x", t[f"{p}kv_proj"]),),
         (("y", t[f"{p}main_slice"]),),
         ("ev_comp_prefill",))
    node("main_score_slice", "slice_view", "tensor_op",
         (("x", t[f"{p}score_proj"]),),
         (("y", t[f"{p}main_score_slice"]),),
         ("ev_comp_prefill",))
    node("tail_slice", "slice_view", "tensor_op",
         (("x", t[f"{p}kv_proj"]),),
         (("y", t[f"{p}tail_slice"]),),
         ("ev_comp_prefill",))
    node("tail_score_slice", "slice_view", "tensor_op",
         (("x", t[f"{p}score_proj"]),),
         (("y", t[f"{p}tail_score_slice"]),),
         ("ev_comp_prefill",))
    node("decode_token_slice", "slice_view", "tensor_op",
         (("x", t[f"{p}kv_proj"]),),
         (("y", t[f"{p}dec_token"]),),
         ("ev_comp_decode",))
    node("decode_token_score_slice", "slice_view", "tensor_op",
         (("x", t[f"{p}score_proj"]),),
         (("y", t[f"{p}dec_token_score"]),),
         ("ev_comp_decode",))
    # 切片事实保存为 AliasRelation（slice + transform_expr 原文，model.py:337-341,351,353）；
    # 图/CSV 导出可核查，不靠 stable_id 承载语义。
    b.alias(f"alias_{p}main_slice", relation_kind="slice",
            tensor_id=t[f"{p}kv_proj"], other_tensor_id=t[f"{p}main_slice"],
            transform_expr="kv[cutoff-r:cutoff]（主序列末块，cutoff=T-T%r）",
            evidence=("ev_comp_prefill",))
    b.alias(f"alias_{p}tail_slice", relation_kind="slice",
            tensor_id=t[f"{p}kv_proj"], other_tensor_id=t[f"{p}tail_slice"],
            transform_expr="kv[cutoff:]（prefill 尾块，长度 remainder=T%r）",
            evidence=("ev_comp_prefill",))
    b.alias(f"alias_{p}main_score_slice", relation_kind="slice",
            tensor_id=t[f"{p}score_proj"], other_tensor_id=t[f"{p}main_score_slice"],
            transform_expr="score[cutoff-r:cutoff]（主块 score 切片 view；加法在后续节点，model.py:344）",
            evidence=("ev_comp_prefill",))
    b.alias(f"alias_{p}tail_score_slice", relation_kind="slice",
            tensor_id=t[f"{p}score_proj"], other_tensor_id=t[f"{p}tail_score_slice"],
            transform_expr="score[cutoff:]（尾块 score 切片 view；加法在后续节点，model.py:341）",
            evidence=("ev_comp_prefill",))
    b.alias(f"alias_{p}dec_token_slice", relation_kind="slice",
            tensor_id=t[f"{p}kv_proj"], other_tensor_id=t[f"{p}dec_token"],
            transform_expr="本调用 kv[:, 0:1]（局部索引，长度 1 恰取本次单 token；start_pos 是历史绝对位置不作投影下标，model.py:353）",
            evidence=("ev_comp_decode",))
    b.alias(f"alias_{p}dec_score_slice", relation_kind="slice",
            tensor_id=t[f"{p}score_proj"], other_tensor_id=t[f"{p}dec_token_score"],
            transform_expr="本调用 score[:, 0:1]（局部索引 view；加法在后续节点，model.py:351,353）",
            evidence=("ev_comp_decode",))
    # score+APE 相加节点（add 产生新值；slice alias 只表达 view）：
    # main：score 切片 + ape 全量 r 行（model.py:344 score = score.unflatten(1,(-1,ratio)) + self.ape）
    node("ape_add_main", "add", "tensor_op",
         (("score_slice", t[f"{p}main_score_slice"]), ("ape", t[f"{p}ape"])),
         (("y", t[f"{p}main_score_aped"]),), ("ev_comp_prefill",))
    # tail：score 切片 + ape 前 remainder 行切片 view（model.py:341 ape[:remainder]；
    # D2：行范围以 slice alias + 行切片张量表达，add 消费切片非整表）
    node("ape_add_tail", "add", "tensor_op",
         (("score_slice", t[f"{p}tail_score_slice"]),
          ("ape_rows", t[f"{p}ape_tail_rows"])),
         (("y", t[f"{p}tail_score_aped"]),), ("ev_comp_prefill",))
    # decode：score 切片 + ape 单行切片 view（model.py:351 score += ape[start_pos % ratio]；
    # D2：单行以 ape[start_pos%r : start_pos%r+1] 切片表达）
    node("ape_add_dec", "add", "tensor_op",
         (("score_slice", t[f"{p}dec_token_score"]),
          ("ape_rows", t[f"{p}ape_dec_row"])),
         (("y", t[f"{p}dec_score_aped"]),), ("ev_comp_decode",))
    # 状态更新：hist + 主块切片（前 r 槽）+ 尾块切片（后 r 槽 offset=ratio）
    # → updated（本调用写入后、待池化）。未触发也执行。
    node("state_update", "state_update", "state_op",
         (("hist_kv", t[f"{p}state_kv_hist"]),
          ("hist_score", t[f"{p}state_score_hist"]),
          ("main_kv", t[f"{p}main_slice"]),
          ("main_score_aped", t[f"{p}main_score_aped"]),
          ("tail_kv", t[f"{p}tail_slice"]),
          ("tail_score_aped", t[f"{p}tail_score_aped"]),
          ("dec_kv", t[f"{p}dec_token"]),
          ("dec_score_aped", t[f"{p}dec_score_aped"])),
         (("kv_state", t[f"{p}state_kv_updated"]),
          ("score_state", t[f"{p}state_score_updated"])),
         ("ev_comp_states", "ev_comp_decode", "ev_comp_prefill",
          "ev_comp_no_trigger"))
    # prefill 路径：blockify → overlap → 池化（g_compress_prefill 触发）
    node("blockify", "unflatten", "tensor_op",
         (("kv", t[f"{p}kv_proj"]),),
         (("y", t[f"{p}blocks"]),), ("ev_comp_prefill",))
    if overlap:
        node("overlap", "overlap_transform", "tensor_op",
             (("x", t[f"{p}blocks"]), ("score", t[f"{p}score_proj"]),
              ("ape", t[f"{p}ape"])),
             (("y", t[f"{p}overlap"]),),
             ("ev_overlap_transform", "ev_comp_prefill"))
    if overlap:
        node("pool_prefill", "softmax_weighted_pool", "tensor_op",
             (("x", t[f"{p}overlap"]),),
             (("y", t[f"{p}pooled"]),), ("ev_comp_prefill",))
    else:
        # 无 overlap（r=128）：blockify 后直接池化（model.py:361-365 else）
        node("pool_prefill", "softmax_weighted_pool", "tensor_op",
             (("x", t[f"{p}blocks"]),),
             (("y", t[f"{p}pooled"]),), ("ev_comp_prefill",))
    # decode 路径：触发时读 updated 状态（历史 2r 槽 + 当前槽位，
    # model.py:355-358）→ 池化
    node("pool_decode", "softmax_weighted_pool_from_state", "tensor_op",
         (("kv_state", t[f"{p}state_kv_updated"]),
          ("score_state", t[f"{p}state_score_updated"]),
          ("ape", t[f"{p}ape"])),
         (("y", t[f"{p}pooled"]),), ("ev_comp_decode",))
    node("norm", "rms_norm", "parameter_op",
         (("x", t[f"{p}pooled"]),), (("y", t[f"{p}normed"]),),
         ("ev_comp_out",))
    node("rope", "rope_partial", "tensor_op",
         (("x", t[f"{p}normed"]),), (("y", t[f"{p}normed"]),),
         ("ev_comp_out", "ev_apply_rotary"))
    node("quant", quant_op, "quant_sim",
         (("x", t[f"{p}normed"]),), (("y", t[f"{p}out"]),), quant_ev)
    # 触发时才写 cache / 产出返回值（未触发 return None，366-367）
    node("cache_out", "cache_publish", "state_op",
         (("x", t[f"{p}out"]),), (("cache", t[cache_role]),),
         ("ev_comp_cache_write",))
    node("return", "compress_return", "state_op",
         (("x", t[f"{p}out"]),), (("y", t[f"{p}return"]),),
         ("ev_comp_cache_write", "ev_comp_no_trigger"))

    edge("wkv_blockify", ("wkv", "y"), ("blockify", "kv"), f"{p}kv_proj",
         guard_ref="g_compress_prefill")
    if overlap:
        edge("wgate_overlap", ("wgate", "y"), ("overlap", "score"),
             f"{p}score_proj", guard_ref="g_compress_prefill")
        edge("ape_overlap", ("ape", "y"), ("overlap", "ape"), f"{p}ape",
             guard_ref="g_compress_prefill")
        edge("blockify_overlap", ("blockify", "y"), ("overlap", "x"),
             f"{p}blocks", guard_ref="g_compress_prefill")
        edge("overlap_pool", ("overlap", "y"), ("pool_prefill", "x"),
             f"{p}overlap", guard_ref="g_compress_prefill")
    else:
        # 无 overlap（r=128）：blockify 直接池化（model.py:361-365 else 分支）
        edge("blockify_pool", ("blockify", "y"), ("pool_prefill", "x"),
             f"{p}blocks", guard_ref="g_compress_prefill")
    edge("pool_prefill_norm", ("pool_prefill", "y"), ("norm", "x"),
         f"{p}pooled", guard_ref="g_compress_prefill")
    edge("state_hist_upd", ("state_hist", "kv_state"),
         ("state_update", "hist_kv"), f"{p}state_kv_hist", kind="state",
         state_transition_ref=tr_kv)
    edge("state_hist_upd_s", ("state_hist", "score_state"),
         ("state_update", "hist_score"), f"{p}state_score_hist", kind="state",
         state_transition_ref=tr_score)
    # updated → pool_decode（触发时读更新后状态池化；g_compress_decode）
    edge("upd_pool_kv", ("state_update", "kv_state"),
         ("pool_decode", "kv_state"), f"{p}state_kv_updated",
         guard_ref="g_compress_decode")
    edge("upd_pool_score", ("state_update", "score_state"),
         ("pool_decode", "score_state"), f"{p}state_score_updated",
         guard_ref="g_compress_decode")
    # 主/尾块切片消费（model.py:337-341）：
    # 主末块 slice（cutoff-r:cutoff）——仅 prefill 主块存在（g_compress_prefill）
    # APE 行切片（D2）：tail 取前 N_rem 行、decode 取 start_pos%r 单行
    node("ape_tail_rows_slice", "slice_view", "tensor_op",
         (("x", t[f"{p}ape"]),), (("y", t[f"{p}ape_tail_rows"]),),
         ("ev_comp_prefill",))
    node("ape_dec_row_slice", "slice_view", "tensor_op",
         (("x", t[f"{p}ape"]),), (("y", t[f"{p}ape_dec_row"]),),
         ("ev_comp_decode",))
    b.alias(f"alias_{p}ape_tail_rows", relation_kind="slice",
            tensor_id=t[f"{p}ape"], other_tensor_id=t[f"{p}ape_tail_rows"],
            transform_expr="ape[:remainder]（前 remainder 行，remainder=T%r，model.py:341）",
            evidence=("ev_comp_prefill",))
    b.alias(f"alias_{p}ape_dec_row", relation_kind="slice",
            tensor_id=t[f"{p}ape"], other_tensor_id=t[f"{p}ape_dec_row"],
            transform_expr="ape[start_pos % r : start_pos % r + 1]（单行，model.py:351）",
            evidence=("ev_comp_decode",))
    edge("ape_tail_rows_add", ("ape_tail_rows_slice", "y"),
         ("ape_add_tail", "ape_rows"), f"{p}ape_tail_rows",
         guard_ref="g_compress_prefill_remainder")
    edge("ape_dec_row_add", ("ape_dec_row_slice", "y"),
         ("ape_add_dec", "ape_rows"), f"{p}ape_dec_row", guard_ref="g_decode")
    edge("wkv_mainslice", ("wkv", "y"), ("main_slice", "x"), f"{p}kv_proj",
         guard_ref="g_compress_prefill")
    edge("wgate_mainslice", ("wgate", "y"), ("main_score_slice", "x"),
         f"{p}score_proj", guard_ref="g_compress_prefill")
    edge("mainslice_upd", ("main_slice", "y"), ("state_update", "main_kv"),
         f"{p}main_slice", guard_ref="g_compress_prefill")
    edge("mainscore_slice_add", ("main_score_slice", "y"),
         ("ape_add_main", "score_slice"), f"{p}main_score_slice",
         guard_ref="g_compress_prefill")
    # main 消费 APE 全 r 表（model.py:344 + self.ape 全量；D2 只拆 tail/dec 行切片）
    edge("ape_add_main_ape", ("ape", "y"), ("ape_add_main", "ape"),
         f"{p}ape", guard_ref="g_compress_prefill")
    edge("ape_add_main_upd", ("ape_add_main", "y"),
         ("state_update", "main_score_aped"), f"{p}main_score_aped",
         guard_ref="g_compress_prefill")
    # 尾块 slice（cutoff:）——仅 prefill 有 remainder 时活动（g_compress_prefill
    # 的活动性由场景 T 决定；T=3 主块 slice 不活动但尾块活动）
    edge("wkv_tailslice", ("wkv", "y"), ("tail_slice", "x"), f"{p}kv_proj",
         guard_ref="g_compress_prefill_remainder")
    edge("wgate_tailslice", ("wgate", "y"), ("tail_score_slice", "x"),
         f"{p}score_proj", guard_ref="g_compress_prefill_remainder")
    edge("tailslice_upd", ("tail_slice", "y"), ("state_update", "tail_kv"),
         f"{p}tail_slice", guard_ref="g_compress_prefill_remainder")
    edge("tailscore_slice_add", ("tail_score_slice", "y"),
         ("ape_add_tail", "score_slice"), f"{p}tail_score_slice",
         guard_ref="g_compress_prefill_remainder")
    edge("ape_add_tail_upd", ("ape_add_tail", "y"),
         ("state_update", "tail_score_aped"), f"{p}tail_score_aped",
         guard_ref="g_compress_prefill_remainder")
    # decode 当前 token：独立 slice 节点 → 尾块端口（g_decode）
    edge("wkv_dectoken", ("wkv", "y"), ("decode_token_slice", "x"),
         f"{p}kv_proj", guard_ref="g_decode")
    edge("wgate_dectoken", ("wgate", "y"), ("decode_token_score_slice", "x"),
         f"{p}score_proj", guard_ref="g_decode")
    edge("dectoken_upd", ("decode_token_slice", "y"),
         ("state_update", "dec_kv"), f"{p}dec_token", guard_ref="g_decode")
    edge("dectoken_score_add", ("decode_token_score_slice", "y"),
         ("ape_add_dec", "score_slice"), f"{p}dec_token_score",
         guard_ref="g_decode")
    edge("ape_add_dec_upd", ("ape_add_dec", "y"),
         ("state_update", "dec_score_aped"), f"{p}dec_score_aped",
         guard_ref="g_decode")

    # decode 池化从状态历史读（g_compress_decode）

    edge("ape_pool_decode", ("ape", "y"), ("pool_decode", "ape"),
         f"{p}ape", guard_ref="g_compress_decode")
    edge("pool_decode_norm", ("pool_decode", "y"), ("norm", "x"),
         f"{p}pooled", guard_ref="g_compress_decode")
    # pool 后 shift（model.py:359-360）：state[:, :r] = state[:, r:]，
    # 产出最终版本 v_final（连续调用的下一历史）；输入来自 pool_decode
    # 完成（同 guard），输出被 cache/return 写入与后续调用的 state_hist
    # 消费（state 边引用同一 transition）
    node("pool_shift", "state_shift", "state_op",
         (("updated_kv", t[f"{p}state_kv_updated"]),
          ("updated_score", t[f"{p}state_score_updated"]),
          ("ctrl", t[f"{p}pooled"])),
         (("kv_state", t[f"{p}state_kv_final"]),
          ("score_state", t[f"{p}state_score_final"])),
         ("ev_comp_decode", "ev_comp_states"))
    # pool 完成先后依赖：pool_decode → pool_shift 的 control 边
    # （shift 必须在池化完成之后，读取前移前状态 model.py:359-360）
    edge("pool_done_shift", ("pool_decode", "y"),
         ("pool_shift", "ctrl"), f"{p}pooled", kind="control",
         guard_ref="g_compress_decode")
    edge("upd_shift_kv", ("state_update", "kv_state"),
         ("pool_shift", "updated_kv"), f"{p}state_kv_updated",
         kind="state", state_transition_ref=tr_kv_shift,
         guard_ref="g_compress_decode")
    edge("upd_shift_score", ("state_update", "score_state"),
         ("pool_shift", "updated_score"), f"{p}state_score_updated",
         kind="state", state_transition_ref=tr_score_shift,
         guard_ref="g_compress_decode")
    # shift 输出（final）→ cache/return 写入消费（触发时产出条目用最终状态）
    # 最终状态发布（publish_final）：下一调用状态的真实来源。
    # 触发 decode：shift 后 final（g_compress_decode）；非触发 decode 与
    # prefill：updated（无 guard——状态更新照常发布）。与版本链绑定：
    # 发布的版本即下一调用的 v_hist（同 state_id 延续，model.py:353-360,362-363）。
    node("publish_final", "state_publish", "state_op",
         (("final", t[f"{p}state_kv_final"]),
          ("final_score", t[f"{p}state_score_final"]),
          ("updated", t[f"{p}state_kv_updated"]),
          ("updated_score", t[f"{p}state_score_updated"])),
         (("kv_state", t[f"{p}state_kv_final"]),
          ("score_state", t[f"{p}state_score_final"])),
         ("ev_comp_states", "ev_comp_decode", "ev_comp_no_trigger"))
    edge("shift_publish", ("pool_shift", "kv_state"),
         ("publish_final", "final"), f"{p}state_kv_final",
         guard_ref="g_compress_decode")
    edge("shift_publish_score", ("pool_shift", "score_state"),
         ("publish_final", "final_score"), f"{p}state_score_final",
         guard_ref="g_compress_decode")
    # 互斥发布：updated 是非触发路径（prefill 全部挂 g_prefill 相位 guard +
    # 非触发 decode 挂 g_decode_no_compress）；触发 decode（g_compress_decode）
    # 只收 shifted。三个 guard 集合按相位/分支互斥（prefill / decode 未触发 /
    # decode 触发）。
    edge("updated_publish", ("state_update", "kv_state"),
         ("publish_final", "updated"), f"{p}state_kv_updated",
         guard_ref="g_prefill")
    edge("updated_publish_dec", ("state_update", "kv_state"),
         ("publish_final", "updated"), f"{p}state_kv_updated",
         guard_ref="g_decode_no_compress")
    edge("updated_publish_score", ("state_update", "score_state"),
         ("publish_final", "updated_score"), f"{p}state_score_updated",
         guard_ref="g_prefill")
    edge("updated_publish_score_dec", ("state_update", "score_state"),
         ("publish_final", "updated_score"), f"{p}state_score_updated",
         guard_ref="g_decode_no_compress")
    # 发布输出 = 下一调用历史：control 依赖表达"发布后下一调用才可读"
    # （state_hist 是 v_hist 版本锚点，无 input 端口；跨步版本链由
    # tr_kv/tr_score 的 hist→updated 序列与 v_final 标签承载）
    edge("publish_hist_next", ("publish_final", "kv_state"),
         ("state_hist", "next_hist"), f"{p}state_kv_final", kind="control")
    edge("publish_hist_next_score", ("publish_final", "score_state"),
         ("state_hist", "next_hist_score"), f"{p}state_score_final",
         kind="control")
    edge("norm_rope", ("norm", "y"), ("rope", "x"), f"{p}normed")
    edge("rope_quant", ("rope", "y"), ("quant", "x"), f"{p}normed")
    # 触发 guard：未触发时 cache/return 无输入（366-367 不产值）
    edge("quant_cache", ("quant", "y"), ("cache_out", "x"), f"{p}out",
         guard_ref="g_compress_prefill")
    edge("quant_cache_dec", ("quant", "y"), ("cache_out", "x"), f"{p}out",
         guard_ref="g_compress_decode")
    edge("quant_return", ("quant", "y"), ("return", "x"), f"{p}out",
         guard_ref="g_compress_prefill")
    edge("quant_return_dec", ("quant", "y"), ("return", "x"), f"{p}out",
         guard_ref="g_compress_decode")


def _add_moe(b: _GraphBuilder, t: dict[str, str], values: ModelValues) -> None:
    """MoE TopK Gate：score→偏置选择→gather 原始分→dispatch→专家模板→汇合。"""
    def node(sid: str, op: str, gr: str, inputs=(), outputs=(), ev=()):
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.node(sid, op, gr, inputs, outputs, ev)

    def edge(sid: str, src, dst, role: str, **kw):
        ev = kw.pop("ev", ())
        for ev_id in ev:
            b.require_evidence(ev_id)
        return b.edge(sid, src, dst, t[role], **kw)

    edge("e_ffnnorm_gate", ("ffn_norm", "y"), ("moe.gate.score", "x"),
         "x_ffn_normed")
    node("moe.gate.score", "moe_gate_score", "parameter_op",
         (("x", t["x_ffn_normed"]),), (("scores", t["gate_scores"]),),
         ("ev_gate_forward",))
    node("moe.gate.bias", "add_bias", "parameter_op",
         (("x", t["gate_scores"]),), (("y", t["gate_scores_biased"]),),
         ("ev_gate_forward",))
    edge("e_score_bias", ("moe.gate.score", "scores"), ("moe.gate.bias", "x"),
         "gate_scores")
    node("moe.gate.topk", "topk_select", "index_op",
         (("x", t["gate_scores_biased"]),), (("indices", t["gate_indices"]),),
         ("ev_gate_forward",))
    edge("e_bias_topk", ("moe.gate.bias", "y"), ("moe.gate.topk", "x"),
         "gate_scores_biased")
    node("moe.gate.gather", "gather_weights", "tensor_op",
         (("original", t["gate_scores"]), ("indices", t["gate_indices"])),
         (("weights", t["gate_weights"]),), ("ev_gate_forward",
                                              "ev_weight_norm"))
    edge("e_orig_gather", ("moe.gate.score", "scores"),
         ("moe.gate.gather", "original"), "gate_scores")
    edge("e_idx_gather", ("moe.gate.topk", "indices"),
         ("moe.gate.gather", "indices"), "gate_indices")
    # 关键结构事实：偏置不进 gather（选择偏置不影响权重）
    node("moe.dispatch", "token_dispatch", "index_op",
         (("indices", t["gate_indices"]),), (("counts", t["n_e"]),),
         ("ev_moe_dispatch",))
    edge("e_idx_dispatch", ("moe.gate.topk", "indices"),
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


# ---------------------------------------------------------------------------
# 守卫与入口
# ---------------------------------------------------------------------------

def _add_compressor_states(b: _GraphBuilder, *,
                           with_indexer: bool = True) -> None:
    """状态版本对（before=历史累积 v_hist，after=本调用后 v_after）。

    transition ID 会被实际 state 边引用（cache 读写、压缩状态更新、
    Indexer cache 打分），未连接的 transition 在 build 时被契约拒绝。
    """
    ev_cache = ("ev_comp_states", "ev_comp_cache_write")
    # 单一 state_id：写（cache_update）与读（cache_read）引用同一 transition，
    # 写后读是同一存储的版本更新，不建平行 state。
    b.state("csa_topk:attn_kv_cache", "tr_attn_cache", ev_cache,
            "注意力 cache（窗口+压缩区）：v_hist 历史累积 → v_after 本调用写入后；"
            "历史连续调用前提，内容 unknown")
    b.state("csa_topk:compress_kv_state", "tr_comp_kv",
            ("ev_comp_states", "ev_comp_decode"),
            "Compressor kv_state：v_hist 历史 → v_updated 本调用写入（待池化）"
            "→ v_final 触发池化+shift 后最终版本（下一调用历史）",
            pool_shift=True)
    b.state("csa_topk:compress_score_state", "tr_comp_score",
            ("ev_comp_states", "ev_comp_decode"),
            "Compressor score_state：同 kv_state 三版本序列",
            pool_shift=True)
    if with_indexer:
        b.state("csa_topk:indexer_kv_cache", "tr_idx_cache",
                ("ev_indexer_init", "ev_comp_cache_write", "ev_idx_score"),
                "Indexer 独立 cache（model.py:405 独立 buffer）")
        b.state("csa_topk:indexer_compress_kv_state", "tr_idx_kv",
                ("ev_indexer_init", "ev_comp_states"),
                "Indexer 自有 Compressor 的 kv_state（三版本序列）",
                pool_shift=True)
        b.state("csa_topk:indexer_compress_score_state", "tr_idx_score",
                ("ev_indexer_init", "ev_comp_states"),
                "Indexer 自有 Compressor 的 score_state（三版本序列）",
                pool_shift=True)


def _guard_objects(registry: EvidenceRegistry) -> tuple[Guard, ...]:
    """rules 守卫目录 → 图 Guard 实体（predicate 原文 + 证据引用）。"""
    return tuple(
        Guard(stable_id=g.stable_id, predicate_expr=g.predicate_expr,
              evidence_refs=(g.evidence_ref,))
        for g in GUARD_DEFS
    )


def build_reference_graph(
    workspace_root: Path,
    *,
    values: ModelValues | None = None,
    evidence_base: EvidenceBase | None = None,
) -> tuple[SemanticGraph, EvidenceRegistry, ModelValues]:
    """构建 layer 4 CSA·TopK 代表图：证据核对 → 图构建 → 通用引用检查。

    返回 (graph, evidence_registry, values)。任何证据核对失败、图构造错误或
    通用引用检查未通过都抛 :class:`ExtractionError`——失败不给图。
    """
    workspace_root = Path(workspace_root)
    values = values or load_model_values(workspace_root)
    evidence_base = evidence_base or load_evidence_base(workspace_root)
    registry = EvidenceRegistry(evidence_base, workspace_root)
    for ev in _evidence_defs():
        registry.register(ev)

    axes = _graph_axes(values)
    b = _GraphBuilder(axes)
    t = _add_tensors(b, values)
    _add_flow(b, t, values)
    _add_compressor_states(b)
    guards = _guard_objects(registry)

    # 全部图引用的证据必须已注册：节点/边（require_evidence）与轴 evidence_ref
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
    return graph, registry, values


# ---------------------------------------------------------------------------
# 场景视图：guard 激活 + Indexer 可见性边界
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioView:
    """一个场景下的模板视图：guard 激活三态 + 派生数量 + 前提记录。"""

    scenario_id: str
    env: dict[str, int]
    guard_activations: tuple[GuardActivation, ...]
    derived: dict[str, int]
    indexer: dict[str, int | str]
    premise: str


def indexer_visibility(env: dict[str, int]) -> dict[str, int | str]:
    """Indexer 可见条目与有效选择边界（model.py:426,433）。

    C_hist：本调用前已完成压缩条目数——历史相关 unknown（轴 C），未给历史
    前提时不当作 0。C_fresh = (P+T)//r 是本调用输入可见的已完成条目上限
    （含历史写入）；可见条目 = C_hist + 新压缩条目。
    """
    past = env["P"]
    tokens = env["T"]
    ratio = env["r"]
    topk = env["Ki"]
    end_pos = past + tokens
    visible_from_input = end_pos // ratio
    fresh_blocks = visible_from_input - past // ratio if past % ratio == 0 else (
        end_pos // ratio - (past // ratio + 1)
    )
    return {
        "visible_upper": visible_from_input,  # (P+T)//r：本调用可见条目上限
        "hist_dependent": 1,  # 1=True：实际可见量依赖历史（C_hist unknown）
        "select_width": min(topk, visible_from_input),  # min(Ki, (P+T)//r)
        "capacity_note": (
            f"Ki={topk} 是上限；条目不足时有效选择 = min(Ki, (P+T)//r)"
        ),
    }


def build_scenario_views(
    values: ModelValues, registry: EvidenceRegistry,
    *, compress_ratio: int = 4, has_indexer: bool = True,
    has_compress: bool = True,
) -> tuple[ScenarioView, ...]:
    """全部场景的 guard 三态激活与派生数量（evaluate_tri 白名单求值）。

    - guard 表达式环境 = 场景变量 + 模型常量 + r（按变体 compress_ratio；
      V2：CSA 4 / HCA 128；SWA 无压缩不产视图）。
    - unknown 一律携带原因，不当作 False。
    - 历史连续调用前提显式记录；未给前提时 cache 内容 unknown。
    - has_indexer=False（HCA）：Indexer 字段记录固定压缩位置选择语义
      （get_compress_topk_idxs，model.py:518-519,274-282），不借 Indexer Ki。
    """
    if compress_ratio == 0:
        # SWA：真实窗口场景（无压缩 guard/state——明确不适用，非空占位）。
        # guard 仅 prefill/decode 相位 + 窗口回绕；派生数量按窗口语义
        # （model.py:513 get_window_topk_idxs、523-528/535 cache 写读）
        views: list[ScenarioView] = []
        swa_guards = ["g_prefill", "g_decode", "g_window_wrap"]
        for scenario in SCENARIOS:
            env = scenario_base_env(values, scenario)
            activations: list[GuardActivation] = []
            for guard in GUARD_DEFS:
                if guard.stable_id not in swa_guards:
                    continue  # 压缩/Indexer guard 在 SWA 明确不适用
                result = evaluate_tri(guard.predicate_expr, env)
                if result.is_known:
                    activations.append(GuardActivation(
                        guard_id=guard.stable_id,
                        scenario_id=scenario.scenario_id,
                        active_for_scenario="true" if result.value else "false",
                    ))
                else:
                    activations.append(GuardActivation(
                        guard_id=guard.stable_id,
                        scenario_id=scenario.scenario_id,
                        active_for_scenario="unknown",
                        unknown_reason=result.unknown_reason,
                    ))
            end_pos = env["P"] + env["T"]
            derived = {
                "N_tokens": env["B"] * env["T"],
                # SWA 无压缩：上下文条目 = min(窗口容量 W, 已处理 token)
                "N_ctx": min(env["W"], end_pos) if env["start_pos"] > 0
                else env["T"],
                "decode_trigger": 0,
                # 窗口环形槽位（model.py:523-528,535）：已填充槽位数 =
                # min(W, end_pos)；下一写入槽位 = end_pos % W（回绕）
                "window_slots_used": min(env["W"], end_pos),
                "window_next_slot": end_pos % env["W"],
                "window_valid_length": min(env["W"], end_pos),
                # 窗口选择索引与图轴同源求值：物理宽 N_win（prefill
                # min(T,W) / decode W 槽位）≠ 有效数上限（-1 填充不占有效）
                "N_win": evaluate_int(EXPR_WINDOW_WIDTH, env),
                "N_sel": evaluate_int(EXPR_WINDOW_WIDTH, env),
                "N_sel_valid_upper": evaluate_int(EXPR_WINDOW_VALID, env),
                # 无压缩派生字段：N_vis/N_sel_c/N_blk/remainder 为压缩专属，
                # SWA 不携带（不得以 0 伪装成压缩统计）
            }
            premise = (
                "历史连续调用前提：P 个历史 token 已按同一窗口规则处理；"
                "cache 实际内容 unknown（本图不验证数值），仅结构连接已核对。"
                "SWA 无压缩路径：压缩/Indexer guard 与状态明确不适用，"
                "cache 容量 = W（model.py:479 无压缩区项）。"
            )
            indexer_info = {
                "selection_mode": "window_only",
                "note": ("SWA 纯滑动窗口：选择 = 窗口索引（环形槽位，"
                         "model.py:513），无压缩选择项；物理宽与有效数分离"),
                "capacity_note": f"cache 容量 = W={env['W']}（无压缩区）",
                # 与图轴 N_win/N_sel 同源求值（槽位数 ≠ 有效数）
                "window_width": evaluate_int(EXPR_WINDOW_WIDTH, env),
                "select_width": evaluate_int(EXPR_WINDOW_WIDTH, env),
                "select_valid_upper": evaluate_int(EXPR_WINDOW_VALID, env),
                "visible_upper": min(env["W"], end_pos),
                "hist_dependent": 1,
            }
            views.append(ScenarioView(
                scenario_id=scenario.scenario_id, env=dict(env),
                guard_activations=tuple(activations), derived=derived,
                indexer=indexer_info, premise=premise,
            ))
        return tuple(views)
    views: list[ScenarioView] = []
    for scenario in SCENARIOS:
        env = scenario_base_env(values, scenario)
        env["r"] = compress_ratio  # 按变体真实压缩比
        activations: list[GuardActivation] = []
        for guard in GUARD_DEFS:
            result = evaluate_tri(guard.predicate_expr, env)
            if result.is_known:
                activations.append(GuardActivation(
                    guard_id=guard.stable_id, scenario_id=scenario.scenario_id,
                    active_for_scenario="true" if result.value else "false",
                ))
            else:
                activations.append(GuardActivation(
                    guard_id=guard.stable_id, scenario_id=scenario.scenario_id,
                    active_for_scenario="unknown",
                    unknown_reason=result.unknown_reason,
                ))
        derived = {
            "N_tokens": env["B"] * env["T"],
            "N_ctx": env["T"] + env["T"] // env["r"],
            "N_blk": env["T"] // env["r"],
            "N_vis": (env["P"] + env["T"]) // env["r"],
            "remainder": env["T"] % env["r"],
            "decode_trigger": 1 if env["start_pos"] > 0 and (
                env["start_pos"] + 1) % env["r"] == 0 else 0,
        }
        # 选择宽度与图轴同源求值（evaluate_int 白名单；相位由 P 编码，
        # prefill P=0 / decode P>0 且 T=1——与 g_prefill 判据一致）：
        # 物理宽 N_win/N_sel_c/N_sel 与有效数上限分开（-1 填充槽位不占有效数）
        derived["N_win"] = evaluate_int(EXPR_WINDOW_WIDTH, env)
        if has_indexer:
            # 学习式 Indexer：有效选择 = min(Ki, N_vis)（model.py:433，
            # Ki 是上限不是每 query 有效数）
            derived["N_sel_c"] = evaluate_int(EXPR_COMPRESS_WIDTH_LEARNED, env)
            derived["N_sel"] = evaluate_int(EXPR_SELECT_WIDTH_LEARNED, env)
            derived["N_sel_valid_upper"] = evaluate_int(
                EXPR_SELECT_VALID_UPPER_LEARNED, env)
        else:
            # HCA 固定压缩位置选择（model.py:518-519,274-282）：
            # 无学习式 topk/Ki——(P+T)//r 在 prefill（P=0）即 T//r、
            # decode（T=1）即 (P+1)//r，逐 query 有效数 (s+1)//r 且 -1 掩码
            derived["N_sel_c"] = evaluate_int(EXPR_COMPRESS_WIDTH_FIXED, env)
            derived["N_sel"] = evaluate_int(EXPR_SELECT_WIDTH_FIXED, env)
            derived["N_sel_valid_upper"] = evaluate_int(
                EXPR_SELECT_VALID_UPPER_FIXED, env)
            derived["selection_mode"] = "fixed_causal_positions"
        premise = (
            "历史连续调用前提：P 个历史 token 已按同一窗口/压缩规则处理；"
            "本图未验证该前提下的实际 cache 内容（cache 数值 unknown），"
            "仅结构连接已核对。未给此前提时 cache 内容完全 unknown。"
        )
        if has_indexer:
            indexer_info = indexer_visibility(env)
        else:
            # HCA：固定压缩位置选择（model.py:274-282,518-519）——
            # 压缩物理宽/每 query 有效数分相位；全部与图轴同源求值：
            # 窗口物理宽 N_win（prefill min(T,W)、decode W，model.py:268）；
            # 压缩宽 (P+T)//r；offset prefill T（kv.size(1)）、decode W
            ratio = env["r"]
            compress_width = evaluate_int(EXPR_COMPRESS_WIDTH_FIXED, env)
            per_query_valid = ("(s+1)//r（-1 掩码）" if env["start_pos"] == 0
                               else str(compress_width))
            window_width = evaluate_int(EXPR_WINDOW_WIDTH, env)
            offset = evaluate_int(EXPR_COMPRESS_OFFSET_FIXED, env)
            if env["start_pos"] == 0:
                note = (f"prefill 固定因果位置选择：压缩物理宽 {compress_width}"
                        f"（逐位置 (s+1)//r 掩码，-1 无效），offset={offset}"
                        f"（kv.size(1)）")
            else:
                note = (f"decode 固定位置：压缩物理宽 {compress_width}"
                        f"（条目 0..{compress_width - 1} + offset={offset}=W）")
            indexer_info = {
                "selection_mode": "fixed_causal_positions",
                "compress_width": compress_width,
                "per_query_valid": per_query_valid,
                "window_width": window_width,
                "offset": offset,
                "select_total_width": evaluate_int(EXPR_SELECT_WIDTH_FIXED, env),
                "select_valid_upper": evaluate_int(
                    EXPR_SELECT_VALID_UPPER_FIXED, env),
                "visible_upper": compress_width,
                "hist_dependent": 0,
                "capacity_note": "固定压缩位置选择（非学习式 topk；"
                                 f"r={ratio}，model.py:518-519,274-282）",
                "note": note,
            }
        views.append(ScenarioView(
            scenario_id=scenario.scenario_id, env=dict(env),
            guard_activations=tuple(activations), derived=derived,
            indexer=indexer_info, premise=premise,
        ))
    return tuple(views)
