"""GLM-5.3-Flash 的 CUBE/VEC 粗估规则（声明式，由唯一导出器消费）。

用户 2026-10-29 确认的规则：CUBE ``B·M·K·N/吞吐``（默认 4096 MAC/ms）、
VEC ``M·N/吞吐``（默认 64 元素/ms），按比例、不向 64 取整。

本模块只声明 GLM-5.3-Flash 自己的语义；构造器/校验来自
``export/perf_engine.py``，不解析模型、不写工作簿、不计算耗时。

源码口径（本地官方 transformers `glm5_next` 文本主干）：

- 4 路 mHC：``fn`` 是 ``[(2+Hc)·Hc, Hc·H]`` 的一次线性；``collapsed=(pre·streams).sum``
  后进单流；子层输出按 post 展开、再与 ``combᵀ·residual`` 相加（4×4 流间混合）。
  Sinkhorn 迭代（Sit=20）属 comb 的内部步骤，不在本粗估另收费（与 T10 同口径）。
- KDA（线性注意力）：depthwise causal conv1d（核长 Kconv=4，通道 3·NhL·DhL）后做
  delta-rule 递推。递推按源码给出的**逐 token 等价数学核**声明三个矩阵项：
  ``mem=A^T·k_t``（矩阵-向量）、``o_t=S_t^T·q_t``（矩阵-向量）、
  ``S_t=A+k_t·δ_tᵀ``（rank-1 状态更新 = Dk·Dv 次乘加/头/token）。
  ``A=S⊙exp(g)`` 与 ``δ=β⊙(v-mem)`` 是逐元素伴随项，不另收费。
  实际实现走 chunk（64 长度 pad、三角 mask、分块并行）——本轮只做跨 XPU 理论粗估，
  采用递推数学核，不计算法改写的附加工作，**不声称 chunk 路径的 MAC 与递推完全相同**。
- DSA（稀疏索引注意力）：indexer 先把 packed keys（k|gate|valid，宽 Ipack=257）按
  Kpool=4 个可见 token 一池做 softmax 加权池化，再逐 query 对 Npool 个池打分并
  topk(SelP=min(Ki//Kpool,Npool))；选中池展开 + 当前不完整 tail 后 pad 到
  Kout=Ki+Kpool-1 的固定索引宽度。主 QK/AV 仍是完整 Lkv 矩阵 + 稀疏 mask，
  不能用 Ksel 冒充主注意力的收缩轴。NoPE：Rk=0，没有 RoPE 调用。
- MoE：sigmoid 路由 + 组选择 + TopK(K=8)，专家计算按总路由分配数 Ttok·K；
  依赖命中专家数的只有“每命中专家扫描分配矩阵”一项（本表 N_hit，默认未知）。
"""
from __future__ import annotations

from structure_reader.export import perf_engine as engine
from structure_reader.export.perf_engine import cube, cube_terms, vec

__all__ = [
    "ENGINE",
    "PERFORMANCE_STATUS",
    "PERFORMANCE_NOTE",
    "THROUGHPUT_NOTE",
    "DISCLAIMER",
    "block_rules",
    "network_rules",
    "network_group",
    "serial_order",
    "rules_report",
    "rule_symbols",
    "block_rule_groups",
]

ENGINE = engine.engine_defaults()
PERFORMANCE_STATUS = engine.PERFORMANCE_STATUS
PERFORMANCE_NOTE = engine.PERFORMANCE_NOTE
THROUGHPUT_NOTE = engine.THROUGHPUT_NOTE
DISCLAIMER = engine.disclaimer("GLM-5.3-Flash")

_RMS_NOTE = "RMSNorm 按工作张量元素数计一次（读+归一写）；不反映真实 kernel 融合。"
_LN_NOTE = "LayerNorm（有 bias）按工作张量元素数计一次。"
_MHC_NOTE = ("mHC 流数 Hc 是真实张量轴（[B,seqlen,Hc,H]）；按最后一次读写的工作张量"
             "计一次元素数，不把流数当 batch 重复计。")
_MHC_MIX_NOTE = ("matmul(combᵀ, residual)：每个 (batch,token) 上 [Hc,Hc]×[Hc,H] 的批量"
                 "矩阵乘，B=B·seqlen、M=Hc、K=Hc、N=H；post 展开/相加不另收费。")

_MHC_ATTN = {
    "g_mhc_attn_in_norm": vec("B*seqlen", "Hc*H",
                              "无权重 RMSNorm 作用在 4 路流拼接上（Hc·H）。"
                              + _RMS_NOTE),
    "g_mhc_attn_fn": cube("B", "seqlen", "Hc*H", "Mhc",
                          "F.linear(flat[B,S,Hc·H], fn[Mhc,Hc·H])：一次线性投影输出 "
                          "pre/post/comb 三段系数；K=Hc·H、N=Mhc=(2+Hc)·Hc。" ),
    "g_mhc_attn_split": vec("B*seqlen", "Mhc",
                            "split 出 pre(Hc)/post(Hc)/comb(Hc×Hc) 的元素数。" ),
    "g_mhc_attn_gate": vec("B*seqlen", "2*Hc",
                           "pre=sigmoid+eps、post=2·sigmoid：每 token 2·Hc 个门控系数。" ),
    "g_mhc_attn_sinkhorn": vec("B*seqlen*Hc", "Hc",
                               "comb 的 Sinkhorn 行/列归一：按 [B,S,Hc,Hc] 的逻辑元素数"
                               "计一次；Sit=20 次迭代属内部算法步骤，不在本粗估另收费"
                               "（与 T10 的 hc_comb 同口径）。" ),
    "g_mhc_attn_collapse": vec("B*seqlen*Hc", "H", _MHC_NOTE + " 坍缩回单流。" ),
    "g_mhc_attn_expand": vec("B*seqlen*Hc", "H",
                               "post 按流广播到子层输出 [B,S,Hc,H]。" ),
    "g_mhc_attn_comb_mix": cube("B*seqlen", "Hc", "Hc", "H", _MHC_MIX_NOTE),
    "g_mhc_attn_add": vec("B*seqlen*Hc", "H", "两处 mHC 结果逐元素相加。" ),
}

_MHC_FFN = {
    key.replace("attn", "ffn"): value for key, value in _MHC_ATTN.items()
}

_KDA = {
    "g_kda_mask_apply": vec("B*seqlen", "H",
                            "padding 置零（不是因果 mask；因果性由短卷积+递推顺序保证）。" ),
    "g_kda_q_proj": cube("B", "seqlen", "H", "NhL*DhL",
                         "self.q_proj：Linear(H, NhL·DhL)，K=H、N=NhL·DhL。" ),
    "g_kda_k_proj": cube("B", "seqlen", "H", "NhL*DhL", "self.k_proj：同形 Linear。" ),
    "g_kda_v_proj": cube("B", "seqlen", "H", "NhL*DhL", "self.v_proj：同形 Linear。" ),
    "g_kda_qkv_cat": vec("B*seqlen", "3*NhL*DhL",
                         "q/k/v 沿特征维拼接（qkv_dim=3·8192）的元素数。" ),
    "g_kda_qkv_transpose": vec("B*seqlen", "3*NhL*DhL",
                               "transpose 把序列维放到最后供 depthwise conv 使用。" ),
    "g_kda_conv_state_update": vec("B", "3*NhL*DhL*Kconv",
                                   "conv state [B,3·NhL·DhL,Kconv] 的写回规模；"
                                   "状态大小固定、不随 kvcachelen 增长。" ),
    "g_kda_conv1d": cube("B*3*NhL*DhL", "seqlen", "Kconv", "1",
                         "depthwise causal conv1d：groups=Cin=Cout=3·NhL·DhL，"
                         "B_eff=batch·groups、M=输出位置数 seqlen、"
                         "K=(Cin/groups)·核长=Kconv、N=Cout/groups=1；"
                         "conv state 提供左侧 Kconv-1 个历史。" ),
    "g_kda_qkv_view": vec("B*seqlen", "3*NhL*DhL",
                          "卷积输出 transpose 回 [B,S,3·NhL·DhL]。" ),
    "g_kda_qkv_split": vec("B*seqlen", "3*NhL*DhL",
                           "按 8192 均分成 q/k/v 三块。" ),
    "g_kda_f_a_proj": cube("B", "seqlen", "H", "DhL",
                           "forget_gate.f_a_proj：Linear(H,DhL) 低秩压缩。" ),
    "g_kda_f_b_proj": cube("B", "seqlen", "DhL", "NhL*DhL",
                           "forget_gate.f_b_proj：Linear(DhL,NhL·DhL) 升维。" ),
    "g_kda_dt_add": vec("B*seqlen", "NhL*DhL", "加 dt_bias（每通道偏置）。" ),
    "g_kda_decay": vec("B*seqlen", "NhL*DhL",
                       "g=-5·sigmoid(exp(A_log)·g_raw)：逐元素衰减 log 系数。" ),
    "g_kda_b_proj": cube("B", "seqlen", "H", "NhL",
                         "self.b_proj：Linear(H,NhL)，每头一个 beta logit。" ),
    "g_kda_beta": vec("B*seqlen", "NhL", "sigmoid(beta logits)。" ),
    "g_kda_q_l2": vec("B*seqlen*NhL", "DhL", "q 的 L2 归一（fp32，在 kernel 内）。" ),
    "g_kda_k_l2": vec("B*seqlen*NhL", "DhL", "k 的 L2 归一。" ),
    "g_kda_q_scale": vec("B*seqlen*NhL", "DhL", "q 乘 DhL^-0.5。" ),
    "g_kda_scan": cube_terms([
        {"label": "mem=A^T·k_t（逐 token 矩阵-向量）", "b": "B*NhL*seqlen", "m": "1",
         "k": "DhL", "n": "DhL"},
        {"label": "o_t=S_t^T·q_t（逐 token 矩阵-向量）", "b": "B*NhL*seqlen", "m": "1",
         "k": "DhL", "n": "DhL"},
        {"label": "S_t=A+k_t·δ_tᵀ（逐 token rank-1 更新）", "b": "B*NhL*seqlen",
         "m": "DhL", "k": "1", "n": "DhL"},
    ], ("delta-rule 逐 token 等价数学核：衰减 A=S⊙exp(g)（逐元素）、mem=A^T·k_t、"
        "δ=β⊙(v-mem)（逐元素）、rank-1 写回 S_t=A+k_t·δ_tᵀ、输出 o_t=S_t^T·q_t。"
        "**每个 token 的状态矩阵不同**，因此矩阵几何按 token 展开：两个矩阵-向量项 "
        "B_eff=B·NhL·seqlen、M=1、K=DhL、N=DhL；rank-1 项 B_eff 同上、M=DhL、K=1、"
        "N=DhL。三个矩阵项各占一行，总量=3·B·NhL·seqlen·DhL·DhL（与原口径相同）；"
        "逐元素伴随项不另收费。"
        "实际实现走 chunk（64 长度 pad/三角 mask/分块并行），本轮只做跨 XPU 理论粗估，"
        "采用递推数学核，不计算法改写附加工作，也不声称 chunk 实际 MAC 与递推相同。")),
    "g_kda_state_write": vec("B*NhL", "DhL*DhL",
                             "recurrent state [B,NhL,DhL,DhL] 的写回规模；"
                             "状态大小固定、不随 kvcachelen 增长。" ),
    "g_kda_g_a_proj": cube("B", "seqlen", "H", "DhL",
                           "输出门 g_a_proj：Linear(H,DhL) 低秩。" ),
    "g_kda_g_b_proj": cube("B", "seqlen", "DhL", "NhL*DhL",
                           "输出门 g_b_proj：Linear(DhL,NhL·DhL)。" ),
    "g_kda_gate_view": vec("B*seqlen", "NhL*DhL", "gate view 成 [B,S,NhL,DhL]。" ),
    "g_kda_o_norm": vec("B*seqlen*NhL", "DhL",
                        "gated RMSNorm：按 head_dim DhL 归一一次再乘 sigmoid(gate)。"
                        + _RMS_NOTE),
    "g_kda_o_reshape": vec("B*seqlen", "NhL*DhL", "把 NhL×DhL 拼成 8192。" ),
    "g_kda_o_proj": cube("B", "seqlen", "NhL*DhL", "H",
                         "self.o_proj：Linear(NhL·DhL,H)，K=8192、N=H。" ),
}

_DENSE = {
    "g_dense_gate": cube("B", "seqlen", "H", "Is",
                         "nn.Linear(H,Is)：dense 层宽度 intermediate_size=12288。" ),
    "g_dense_up": cube("B", "seqlen", "H", "Is", "nn.Linear(H,Is)，与 gate 并行。" ),
    "g_dense_clamp": vec("B*seqlen", "2*Is",
                         "gate.clamp(max=Swi) 与 up.clamp(±Swi)：两段各 Is。" ),
    "g_dense_act": vec("B*seqlen", "Is", "SiLU(gate.clamp)·up.clamp。" ),
    "g_dense_down": cube("B", "seqlen", "Is", "H", "nn.Linear(Is,H)。" ),
}

_DSA = {
    "g_dsa_q_a_proj": cube("B", "seqlen", "H", "Rq",
                           "self.q_a_proj：Linear(H,Rq)，K=H、N=1536。" ),
    "g_dsa_q_a_norm": vec("B*seqlen", "Rq", "q_a_layernorm（eps=1e-5）。" + _RMS_NOTE),
    "g_dsa_q_b_proj": cube("B", "seqlen", "Rq", "Nh*Dq",
                           "self.q_b_proj：Linear(Rq,Nh·Dq)，输出 64×256。" ),
    "g_dsa_q_view": vec("B*seqlen*Nh", "Dq",
                        "view+transpose 到 [B,Nh,S,Dq]；NoPE 没有 q split/RoPE。" ),
    "g_dsa_kv_a_proj": cube("B", "seqlen", "H", "Rkv+Rk",
                            "kv_a_proj_with_mqa：Linear(H,Rkv+Rk)，NoPE 时 Rk=0。" ),
    "g_dsa_kv_a_split": vec("B*seqlen", "Rkv+Rk", "拆出 latent(Rkv) 与 k_rot(Rk=0)。" ),
    "g_dsa_kv_a_norm": vec("B*seqlen", "Rkv", "只归一 latent 段。" + _RMS_NOTE),
    "g_dsa_k_pass_view": vec("B*seqlen", "Rkv", "view 成 [B,1,S,Rkv]。" ),
    "g_dsa_k_rot_view": vec("B*seqlen", "Rk",
                            "NoPE：Rk=0，这个 4D 张量最后一维为 0（合法零维度）。" ),
    "g_dsa_cache_kv_update": vec("B*seqlen", "Rkv+Rk",
                                 "只计本次写入的 latent+k_rot（NoPE 时 Rk=0）。" ),
    "g_dsa_kv_b_proj": cube("B", "Lkv", "Rkv", "Nh*(Dn+Dv)",
                            "kv_b_proj 展开全部 Lkv 个缓存位置：M=Lkv、K=Rkv、"
                            "N=Nh·(Dn+Dv)=64×512。" ),
    "g_dsa_kv_view": vec("B*Nh*Lkv", "Dn+Dv", "按头重排的元素数。" ),
    "g_dsa_kv_split": vec("B*Nh*Lkv", "Dn+Dv", "拆成 k_nope(Dn) 与 value(Dv)。" ),
    "g_dsa_k_rot_expand": vec("B*Nh*Lkv", "Rk", "NoPE：0 维广播后仍是 0 维。" ),
    "g_dsa_key_cat": vec("B*Nh*Lkv", "Dq", "k_nope 与 0 维 k_rot 拼接（Dq=Dn+Rk）。" ),
    "g_dsa_ix_q_proj": cube("B", "seqlen", "Rq", "Ih*Id",
                            "indexer.wq_b：Linear(Rq,Ih·Id)。" ),
    "g_dsa_ix_q_view": vec("B*seqlen", "Ih*Id", "view 成 [B,S,Ih,Id]；NoPE 无 RoPE。" ),
    "g_dsa_ix_k_proj": cube("B", "seqlen", "H", "Id", "indexer.wk：Linear(H,Id)。" ),
    "g_dsa_ix_k_norm": vec("B*seqlen", "Id",
                           "indexer.k_norm 是 LayerNorm（有 bias）。" + _LN_NOTE),
    "g_dsa_ix_gate": cube("B", "seqlen", "H", "Id",
                          "index_kpool_compress_gate：Linear(H,Id)，输出池内 softmax 权重 logits。" ),
    "g_dsa_ix_valid": vec("B*seqlen", "1", "padding mask 取 valid 通道并入 packed。" ),
    "g_dsa_ix_pack": vec("B*seqlen", "Ipack",
                         "cat([k, gate, valid]) 成宽 257 的 packed 状态。" ),
    "g_dsa_ix_cache_update": vec("B*seqlen", "Ipack",
                                 "只计本次追加的 packed 行 [B,Lidx,257]。" ),
    "g_dsa_ix_valid_keys": vec("B*Lidx", "1", "取 packed 最后一维 valid。" ),
    "g_dsa_ix_visible": vec("B*seqlen", "Lidx",
                            "逐 query 构造 causal & valid_keys 的可见性矩阵。" ),
    "g_dsa_ix_pool": vec("B*Npool*Kpool", "Id",
                         "池化：每池 Kpool 个 token 的 Id 维 key 做 softmax 加权求和"
                         "（只依赖 packed 状态与 APE，不依赖逐 query 可见性）；"
                         "不完整尾池不作为候选池。" ),
    "g_dsa_ix_pool_end": vec("B*Npool", "1", "取每池最后一个 token 索引并 clamp。" ),
    "g_dsa_ix_valid_candidates": vec("B*seqlen", "Npool",
                                     "pool_end 的逐 query 可见性 & pool_valid。" ),
    "g_dsa_ix_scores": cube("B*Ih", "seqlen", "Id", "Npool",
                            "torch.matmul(q[B,S,Ih,Id], pool_keys[B,Npool,Id]ᵀ)："
                            "头数作 batch，M=S、K=Id、N=Npool。" ),
    "g_dsa_ix_relu": vec("B*seqlen*Ih", "Npool", "relu 打分张量元素数。" ),
    "g_dsa_ix_weights": cube("B", "seqlen", "H", "Ih",
                             "indexer.weights_proj：Linear(H,Ih)。" ),
    "g_dsa_ix_sum": cube("B*seqlen", "1", "Ih", "Npool",
                         "每个 (batch, query) 各有一份 scores[Ih,Npool] 与 weights[Ih]，"
                         "按头加权求和是逐 query 的一次 [1,Ih]×[Ih,Npool] 收缩："
                         "B_eff=B·seqlen、M=1（每 query 一行）、K=Ih、N=Npool；"
                         "不是跨 query 共用的单个 [Ih,Npool] 矩阵。" ),
    "g_dsa_ix_mask": vec("B*seqlen", "Npool", "无效池位置置 -inf。" ),
    "g_dsa_ix_topk": vec("B*seqlen", "Npool",
                         "逐 query 在 Npool 个池分数上 topk(min(Ki//Kpool,Npool))；"
                         "按整张 [B,S,Npool] 扫描一次。" ),
    "g_dsa_ix_selected_valid": vec("B*seqlen", "SelP", "选中池的 valid gather。" ),
    "g_dsa_ix_select_pools": vec("B*seqlen*SelP", "Kpool",
                                 "每个选中池取回 Kpool 个原始 token 索引。" ),
    "g_dsa_ix_flatten": vec("B*seqlen*SelP", "Kpool", "池内索引 flatten（4×SelP）。" ),
    "g_dsa_ix_invalid": vec("B*seqlen*SelP", "Kpool", "无效池位置填 -1。" ),
    "g_dsa_ix_tail": vec("B*seqlen", "Kpool",
                         "追加当前不完整池的可见 token（宽度 Kpool-1，按 Kpool 上限计）。" ),
    "g_dsa_ix_concat_tail": vec("B*seqlen", "SelP*Kpool+Kpool",
                                "选中池展开（SelP·Kpool）与 tail（≤Kpool）拼接。" ),
    "g_dsa_ix_pad": vec("B*seqlen", "Kout",
                        "pad 到固定索引宽度 Kout=Ki+Kpool-1（含 -1 填充）。" ),
    "g_dsa_ix_query_mask": vec("B*seqlen", "Kout",
                               "padding query 的整行 top-k 置 -1（保留 query padding 语义）。" ),
    "g_dsa_mask_counts": vec("B*seqlen", "Kout+Lkv",
                             "把 Kout 个索引 scatter 成每 query 每 key 的计数，"
                             "再对 Lkv 个位置做 ne(0)。" ),
    "g_dsa_sparse_mask": vec("B*seqlen", "Lkv", "counts≠0 → bool mask 并 unsqueeze。" ),
    "g_dsa_eager_mask": vec("B*seqlen", "Lkv", "eager 路径把 bool mask 转 0/min。" ),
    "g_dsa_qk": cube("B*Nh", "seqlen", "Dq", "Lkv",
                     "主注意力仍是**完整 Lkv** 矩阵：头数作 batch、M=S、K=Dq、N=Lkv；"
                     "indexer 的稀疏选择只通过 mask 生效，不能用 Ksel 当收缩轴。" ),
    "g_dsa_mask_add": vec("B*Nh*seqlen", "Lkv", "稀疏 mask 加到 [B,Nh,S,Lkv] 打分。" ),
    "g_dsa_softmax": vec("B*Nh*seqlen", "Lkv", "softmax 在全部 Lkv 上。" ),
    "g_dsa_av": cube("B*Nh", "seqlen", "Lkv", "Dv",
                     "AV 也是完整 Lkv：M=S、K=Lkv、N=Dv。" ),
    "g_dsa_attn_transpose": vec("B*seqlen*Nh", "Dv", "transpose(1,2) 元素数。" ),
    "g_dsa_attn_reshape": vec("B*seqlen", "Nh*Dv", "Nh×Dv 拼成一维。" ),
    "g_dsa_o_proj": cube("B", "seqlen", "Nh*Dv", "H", "self.o_proj：Linear(Nh·Dv,H)。" ),
}

_MOE = {
    "g_moe_flatten": vec("Ttok", "H", "hidden.view(Ttok,H) 的元素数；residuals 保留原形状。" ),
    "g_moe_router": cube("B", "seqlen", "H", "E",
                         "F.linear(x.float(), W.float())：K=H、N=E（float32 路由）。" ),
    "g_moe_sigmoid": vec("Ttok", "E", "sigmoid 打分。" ),
    "g_moe_bias_add": vec("Ttok", "E", "加 e_score_correction_bias（只参与选择）。" ),
    "g_moe_group_score": vec("Ttok", "E", "每组 top2 求和：按 [Ttok,E] 扫描一次。" ),
    "g_moe_group_topk": vec("Ttok", "Ng", "在 Ng 个组分数上 topk(Ktg)。" ),
    "g_moe_group_scatter": vec("Ttok", "Ng", "组掩码 zeros+scatter。" ),
    "g_moe_group_expand": vec("Ttok", "E", "组掩码 expand 到 [Ttok,E]。" ),
    "g_moe_group_reshape": vec("Ttok", "E", "3D→2D reshape。" ),
    "g_moe_group_masked_fill": vec("Ttok", "E", "组外专家 masked_fill(-inf)。" ),
    "g_moe_topk": vec("Ttok", "E", "每 token 对 E 个分数 topk(K=8)。" ),
    "g_moe_weight_gather": vec("Ttok", "K", "gather 未加偏置的路由权重。" ),
    "g_moe_weight_norm": vec("Ttok", "K", "top-k 权重归一。" ),
    "g_moe_weight_scale": vec("Ttok", "K", "乘 routed_scaling_factor。" ),
    "g_moe_expert_mask": vec("(E+1)*K", "Ttok", "one_hot+permute 的完整分配矩阵物化。" ),
    "g_moe_expert_hit": vec("(E+1)*K", "Ttok", "对分配矩阵 sum+nonzero 求命中集合。" ),
    "g_moe_expert_gather_indices": engine.expert_hit_rule(
        experts="E", top_k="K", tokens="Ttok", slice_rows="K", lower_text="K",
        note=("对每个命中专家取 (top_k_pos, token_idx)：每次扫描 [K,Ttok]；"
              "总量=N_hit·K·Ttok。单值是各层统一命中数假设，"
              "实际路由分布不由本工具推断。")),
    "g_moe_select_tokens": vec("Ttok*K", "H", "按总分配行数 gather token 行。" ),
    "g_moe_expert_gate_up": cube("1", "Ttok*K", "H", "2*Es",
                                 "gate/up 打包成 [E,2Es,H] 的合并 Linear："
                                 "M=总路由分配数 Ttok·K、K=H、N=2Es。" ),
    "g_moe_expert_chunk": vec("Ttok*K", "2*Es", "拆成 gate/up 各 Es。" ),
    "g_moe_expert_clamp": vec("Ttok*K", "2*Es", "SwiGLU 双端 clamp（gate 上限 Swi、up ±Swi）。" ),
    "g_moe_expert_act": vec("Ttok*K", "Es", "SiLU(gate.clamp)·up.clamp。" ),
    "g_moe_expert_down": cube("1", "Ttok*K", "Es", "H", "down_proj[e]：K=Es、N=H。" ),
    "g_moe_expert_weight": vec("Ttok*K", "H", "按 (token_idx,top_k_pos) 加权。" ),
    "g_moe_result_init": vec("Ttok", "H", "zeros_like(Ttok,H)。" ),
    "g_moe_index_add": vec("Ttok*K", "H", "按 token_idx 累加各分配贡献。" ),
    "g_moe_routed_view": vec("Ttok", "H", "view 回 [B,seqlen,H]。" ),
    "g_moe_shared_gate": cube("B", "seqlen", "H", "Es", "共享专家 gate_proj。" ),
    "g_moe_shared_up": cube("B", "seqlen", "H", "Es", "共享专家 up_proj。" ),
    "g_moe_shared_clamp": vec("B*seqlen", "2*Es", "共享专家 SwiGLU clamp。" ),
    "g_moe_shared_act": vec("B*seqlen", "Es", "SiLU(gate.clamp)·up.clamp。" ),
    "g_moe_shared_down": cube("B", "seqlen", "Es", "H", "共享专家 down_proj。" ),
    "g_moe_merge": vec("B*seqlen", "H", "routed + shared 相加。" ),
}

_RULES_BY_GROUP = {
    "kda-dense": {**_MHC_ATTN, **_MHC_FFN, "g_input_norm": vec(
        "B*seqlen", "H", "带权重 RMSNorm（层输入来自 mHC 坍缩）。" + _RMS_NOTE),
        **_KDA, **_DENSE},
    "dsa-moe": {**_MHC_ATTN, **_MHC_FFN, "g_input_norm": vec(
        "B*seqlen", "H", "带权重 RMSNorm（层输入来自 mHC 坍缩）。" + _RMS_NOTE),
        **_DSA, **_MOE, "g_post_norm": vec(
            "B*seqlen", "H", "post_attention_layernorm（FFN 前的 mHC 坍缩单流）。"
            + _RMS_NOTE)},
    "kda-moe": {**_MHC_ATTN, **_MHC_FFN, "g_input_norm": vec(
        "B*seqlen", "H", "带权重 RMSNorm（层输入来自 mHC 坍缩）。" + _RMS_NOTE),
        **_KDA, **_MOE, "g_post_norm": vec(
            "B*seqlen", "H", "post_attention_layernorm（FFN 前的 mHC 坍缩单流）。"
            + _RMS_NOTE)},
}
# kda-dense 的 dense MLP 后也有 post_attention_layernorm
_RULES_BY_GROUP["kda-dense"]["g_post_norm"] = vec(
    "B*seqlen", "H", "post_attention_layernorm（FFN 前的 mHC 坍缩单流）。" + _RMS_NOTE)

# ------------------------------------------------------------------ 整网
_NETWORK = {
    "net_embed": vec("B*seqlen", "H", "token→[B,seqlen,H] 嵌入查表与写出。" ),
    "net_position_ids": vec("seqlen", "1",
                            "NoPE：源码不构造 rotary_emb，position_ids 只被转发；"
                            "按 [1,seqlen] 个标量计。" ),
    "net_mask_build": vec("B*seqlen", "1+Lkv",
                          "一次构建 2D padding mask 并转发；Lkv 项为调用方 2D mask 的"
                          "拼接宽度（本模型没有网络级 4D 因果 mask）。" ),
    "net_hc_expand": vec("B*seqlen*Hc", "H",
                         "把单流 embedding 广播成 Hc 路相同流的物化规模。" ),
    "net_layer_loop": engine.container(
        "45 层循环容器：34 层 KDA + 11 层 DSA，层内成本由三组层型小计×真实层数"
        "串行汇总，本行不重复计成本。"),
    "net_hc_head": vec("B*seqlen*Hc", "H", "4 路流按均值收拢成单流。" ),
    "net_final_norm": vec("B*seqlen", "H", "Final RMSNorm。" + _RMS_NOTE),
    "net_logits_slice": vec("B*Ssel", "H", "只对保留位置做后续投影。" ),
    "net_lm_head": cube("B", "Ssel", "H", "Vocab", "LM Head：K=H、N=Vocab。" ),
}

_NETWORK_SEGMENTS = (
    {"id": "entry", "label": "整网入口外围",
     "operator_ids": ["net_embed", "net_position_ids", "net_mask_build", "net_hc_expand"],
     "subtotal": True},
    {"id": "loop", "label": "45 层循环（容器，不单独计成本）",
     "operator_ids": ["net_layer_loop"], "subtotal": False},
    {"id": "exit", "label": "整网出口外围",
     "operator_ids": ["net_hc_head", "net_final_norm", "net_logits_slice", "net_lm_head"],
     "subtotal": True},
)


def block_rule_groups() -> dict:
    return _RULES_BY_GROUP


def block_rules(group_id: str) -> dict[str, dict]:
    if group_id not in _RULES_BY_GROUP:
        raise ValueError(f"未知 GLM-5.3-Flash 层组 id {group_id!r}：没有 CUBE/VEC 规则")
    return _RULES_BY_GROUP[group_id]


def network_rules() -> dict[str, dict]:
    return _NETWORK


def network_segments() -> tuple:
    return tuple(dict(segment) for segment in _NETWORK_SEGMENTS)


def serial_order() -> list[dict]:
    return [
        {"kind": "segment", "group": "net-periphery", "segment": "entry"},
        {"kind": "group", "group": "kda-dense"},
        {"kind": "group", "group": "dsa-moe"},
        {"kind": "group", "group": "kda-moe"},
        {"kind": "segment", "group": "net-periphery", "segment": "exit"},
    ]


def network_group(network: dict) -> dict:
    import copy

    path = network["path"]
    rules = network_rules()
    raw_operators = list(path.get("operators", []))
    missing = [op["id"] for op in raw_operators if op["id"] not in rules]
    if missing:
        raise ValueError(f"整网外围算子缺少 CUBE/VEC 规则: {missing}")
    operators = [{**op, "performance": copy.deepcopy(rules[op["id"]])}
                 for op in raw_operators]
    parameters = engine.periphery_parameters(
        network,
        input_ids={"B": "batch_size", "seqlen": "seqlen",
                   "kvcachelen": "kvcachelen"},
        derived_source=[{
            "path": "structure_reader/outputs/<candidate>/data/network_overview.json",
            "locator": "path.parameters",
            "kind": "source",
            "note": "整网外围参数；取自同源 network_overview，不另造一份",
        }])
    return {
        "id": "net-periphery",
        "label": "整网外围（token→logits 的入口/出口外围算子）",
        "attention_type": "（整网外围）",
        "moe_type": "（不适用）",
        "layer_indices": [],
        "representative_layer_index": None,
        "loop": None,
        "template_note": ("整网外围来自同一份 network_overview.json；"
                          "层内结构仍由三个层组代表层 Block 表达，不铺开 45 层。"),
        "scope_note": ("范围：一次 forward 的入口外围（embedding/位置/padding mask/"
                       "4 路 mHC 展开）+ 出口外围（均值收拢/Final RMSNorm/logits 切片/"
                       "LM Head）；45 层循环本体不在此块计成本。"),
        "parameters": parameters,
        "tensors": list(path.get("tensors", [])),
        "operators": operators,
        "cache_notes": list(path.get("cache_notes", [])),
        "coverage_notes": list(path.get("coverage_notes", [])),
        "segments": [dict(segment) for segment in _NETWORK_SEGMENTS],
        "kind": "periphery",
    }


def rule_symbols(rule: dict) -> list[str]:
    return engine.rule_symbols(rule)


def rules_report() -> list[dict]:
    return engine.rules_report(_RULES_BY_GROUP, _NETWORK)
