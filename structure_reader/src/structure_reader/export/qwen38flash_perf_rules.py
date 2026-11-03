"""Qwen3.8-Flash-Next 的 CUBE/VEC 粗估规则（声明式，由唯一导出器消费）。

用户 2026-10-29 确认的规则：CUBE ``B·M·K·N/吞吐``（默认 4096 MAC/ms）、
VEC ``M·N/吞吐``（默认 64 元素/ms），按比例、不向 64 取整。

本模块只声明 Qwen3.8-Flash-Next 自己的语义；构造器/校验来自
``export/perf_engine.py``，不解析模型、不写工作簿、不计算耗时。

源码口径（本地官方 transformers `qwen4_exp` 文本主干）：

- Gated Residual（4 路 hyper-connection）：读门是逐元素 ``[B,S,Hc,H]``（低秩
  down→SiLU→up），写门是每支路标量 ``[B,S,Hc]``；不是 mHC/Sinkhorn。
- GDN（线性注意力）：depthwise causal conv1d（Dqkv=10240 通道、核长 4、dilation=1）
  后做 gated delta-rule 递推。递推按**逐 token 等价数学核**声明三个矩阵项：
  ``mem=A^T·k_t``、``o_t=S_t^T·q_t``、``S_t=A+k_t·δ_tᵀ``（rank-1 更新）；
  逐元素伴随项（衰减/δ/门控）不另收费。实际实现走 chunk（64 长度 pad、三角 mask、
  分块并行），本轮只做跨 XPU 理论粗估，采用递推数学核，不计算法改写附加工作，
  **不声称 chunk 实际 MAC 与递推相同**。
- QSA（稀疏索引注意力）：indexer 在 ``forward`` 内**逐 batch、逐 query** 循环
  （modeling_qwen4_exp.py:721）：第 j 个 query 的可见长 ``V_q=P+j+1``，
  完整 micro-block 数 ``N_q=floor(V_q/Kc)``，选中块数 ``B_q=min(Btop,N_q)``，
  选中 token 数 ``Ksel=Kc·B_q+(V_q % Kc)``。整次 S 个查询的成本按精确和汇总：
  ``ΣNq``/``ΣBq``/``Σtail`` 用整除前缀闭式（见 ``_sum_*`` 表达式），
  不用最后一个 query 的 ``Ksel_last·S`` 或某个固定 j 冒充整网。
  主 QK/AV 仍是**完整 Lkv** 矩阵 + 稀疏 mask（:798/802），不能用 Ksel 当主注意力
  的收缩轴；逻辑选中数与 pad 后的固定索引宽度 Kout 也分开表达。
- MoE：softmax 路由 + TopK(K=10)，专家计算按总路由分配数 Ttok·K；共享专家并行，
  另有标量门；依赖命中专家数的只有“每命中专家扫描分配矩阵”一项（本表 N_hit）。
- PLE（仅 layer 1）：16 个 n-gram 哈希头（bigram 8 + trigram 8）×160 维；
  dilation=3 depthwise 短卷积（核长 4，有效感受野 10）。
"""
from __future__ import annotations

from structure_reader.export import perf_engine as engine
from structure_reader.export.perf_engine import cube, cube_terms, vec, vec_terms

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
    "qsa_sum_expressions",
]

ENGINE = engine.engine_defaults()
PERFORMANCE_STATUS = engine.PERFORMANCE_STATUS
PERFORMANCE_NOTE = engine.PERFORMANCE_NOTE
THROUGHPUT_NOTE = engine.THROUGHPUT_NOTE
DISCLAIMER = engine.disclaimer("Qwen3.8-Flash-Next")

_RMS_NOTE = "RMSNorm 按工作张量元素数计一次（读+归一写）；不反映真实 kernel 融合。"

# ---------------------------------------------------------------- QSA 精确和
#
# 依据 modeling_qwen4_exp.py:718-758 的逐 query 循环：
#   V_q = P + j + 1 (j=0..S-1)，N_q = V_q // Kc，B_q = min(Btop, N_q)，
#   tail = V_q % Kc，Ksel = Kc·B_q + tail。
# 对 j 求和用整除前缀闭式：G(n) = Σ_{v=0}^{n-1} floor(v/Kc)
#   = Kc·q·(q-1)/2 + q·r，q=n//Kc、r=n%Kc。
# 于是 ΣNq = G(P+1+S) - G(P+1)；
#      ΣBq = [G(min(P+1+S, Btop·Kc)) - G(P+1)]⁺ + min(S, [P+1+S-Btop·Kc]⁺)·Btop；
#      ΣVq = S·(2P+S+1)/2；Σtail = ΣVq - Kc·ΣNq。
# 这些是精确整数和（非取整近似），与 /private/tmp/llmarchi-qsa-sums-reference.py
# 的 13 组枚举核对一致（见 T11-cube-vec-verify/qsa-sums-check.json）。
_A = "(kvcachelen+1)"


def _G(n: str) -> str:
    return (f"(Kc*({n}//Kc)*(({n}//Kc)-1)//2"
            f"+({n}//Kc)*({n}%Kc))")


_SUM_NQ = f"({_G(f'({_A}+seqlen)')}-{_G(_A)})"
_SUM_BQ = (f"(max(0,{_G(f'min({_A}+seqlen,Btop*Kc)')}-{_G(_A)})"
           f"+min(seqlen,max(0,{_A}+seqlen-Btop*Kc))*Btop)")
_SUM_VQ = f"(seqlen*(2*{_A}+seqlen-1)//2)"
_SUM_TAIL = f"({_SUM_VQ}-Kc*{_SUM_NQ})"


def qsa_sum_expressions() -> dict:
    """QSA 逐 query 和的闭式（供证据脚本独立复算，不复制产品规则）。"""
    return {"sum_Nq": _SUM_NQ, "sum_Bq": _SUM_BQ,
            "sum_Vq": _SUM_VQ, "sum_tail": _SUM_TAIL}


# ---------------------------------------------------------------- Gated Residual
_HC_ATTN = {
    "g_hc_attn_norm": vec("B*seqlen", "Hc*H",
                          "每个残差流独立 RMSNorm（group_size=H）后仍 flat。"
                          + _RMS_NOTE),
    "g_hc_attn_down": cube("B", "seqlen", "Hc*H", "Rhc",
                           "读门低秩下投影：K=Hc·H=10240、N=Rhc=320。" ),
    "g_hc_attn_down_act": vec("B*seqlen", "Rhc",
                              "先除 Hc 再 SiLU（真实缩放，不是显示修正）。" ),
    "g_hc_attn_up": cube("B", "seqlen", "Rhc", "Hc*H",
                         "读门上投影回 10240 维：K=Rhc、N=Hc·H。" ),
    "g_hc_attn_read_sigmoid": vec("B*seqlen", "Hc*H", "读门 sigmoid（逐元素）。" ),
    "g_hc_attn_read_view": vec("B*seqlen", "Hc*H",
                               "flat 读门 unflatten 成 [B,S,Hc,H] 的元素数。" ),
    "g_hc_attn_norm_view": vec("B*seqlen", "Hc*H",
                               "normed unflatten 成 [B,S,Hc,H] 的元素数。" ),
    "g_hc_attn_mix": vec("B*seqlen*Hc", "H",
                         "读门加权后取 mean 坍缩回单流：按 [B,S,Hc,H] 元素数计一次。" ),
    "g_hc_attn_inject_linear": cube("B", "seqlen", "Hc*H", "Hc",
                                    "写门投影：K=Hc·H、N=Hc（每支路一个标量）。" ),
    "g_hc_attn_inject": vec("B*seqlen", "Hc", "2·sigmoid(logits/Hc) 的每支路标量门。" ),
    "g_hc_attn_expand": vec("B*seqlen*Hc", "H", "子层输出按写门展开到 [B,S,Hc,H]。" ),
    "g_hc_attn_expand_flat": vec("B*seqlen", "Hc*H", "展开结果 flatten。" ),
    "g_hc_attn_add": vec("B*seqlen", "Hc*H",
                         "与其 4 路残差流 hyper_input 相加（flat 边界）。" ),
}

_HC_FFN = {key.replace("attn", "ffn"): value for key, value in _HC_ATTN.items()}

# ---------------------------------------------------------------- GDN
_GDN = {
    "g_gdn_mask": vec("B*seqlen", "H",
                      "padding 置零（不是因果 mask；因果性由短卷积+递推顺序保证）。" ),
    "g_gdn_qkv_proj": cube("B", "seqlen", "H", "Dqkv",
                           "in_proj_qkv：一次 Linear(H,Dqkv=10240) 出 q/k/v 三段。" ),
    "g_gdn_qkv_transpose": vec("B*seqlen", "Dqkv",
                               "transpose 把序列维放到最后供 depthwise conv。" ),
    "g_gdn_z_proj": cube("B", "seqlen", "H", "Dz",
                         "in_proj_z：Linear(H,Dz=6144) 输出门 z。" ),
    "g_gdn_z_view": vec("B*seqlen", "Dz", "z view 成 [B,S,NvH,DvH]。" ),
    "g_gdn_b_proj": cube("B", "seqlen", "H", "NvH",
                         "in_proj_b：Linear(H,NvH=48)，每 v 头一个 beta logit。" ),
    "g_gdn_a_proj": cube("B", "seqlen", "H", "NvH",
                         "in_proj_a：Linear(H,NvH)，每 v 头一个 decay logit。" ),
    "g_gdn_conv_state_update": vec("B", "Dqkv*Kconv",
                                   "conv state [B,Dqkv,Kconv] 的写回规模；"
                                   "状态大小固定、不随 kvcachelen 增长。" ),
    "g_gdn_conv": cube("B*Dqkv", "seqlen", "Kconv", "1",
                       "depthwise causal conv1d：groups=Cin=Cout=Dqkv，"
                       "B_eff=batch·groups、M=输出位置数 seqlen、"
                       "K=(Cin/groups)·核长=Kconv=4、N=Cout/groups=1；"
                       "conv state 提供左侧 Kconv-1 个历史。" ),
    "g_gdn_conv_transpose": vec("B*seqlen", "Dqkv", "卷积输出 transpose 回 [B,S,Dqkv]。" ),
    "g_gdn_split": vec("B*seqlen", "Dqkv",
                       "按 q(2048)/k(2048)/v(6144) 拆分；q/k 16 头、v 48 头。" ),
    "g_gdn_beta": vec("B*seqlen", "NvH", "sigmoid(beta logits)。" ),
    "g_gdn_decay": vec("B*seqlen", "NvH",
                       "g=-exp(A_log)·softplus(a+dt_bias)：每 v 头一个 log 衰减系数。" ),
    "g_gdn_q_repeat": vec("B*seqlen*NvH", "Dk",
                          "q 头 16→48 的 repeat_interleave 物化规模。" ),
    "g_gdn_k_repeat": vec("B*seqlen*NvH", "Dk", "k 头 16→48 的 repeat_interleave。" ),
    "g_gdn_q_l2": vec("B*seqlen*NvH", "Dk", "q 的 L2 归一（fp32，在 kernel 内）。" ),
    "g_gdn_k_l2": vec("B*seqlen*NvH", "Dk", "k 的 L2 归一。" ),
    "g_gdn_q_scale": vec("B*seqlen*NvH", "Dk", "q 乘 Dk^-0.5。" ),
    "g_gdn_scan": cube_terms([
        {"label": "mem=A^T·k_t（逐 token 矩阵-向量）", "b": "B*NvH*seqlen", "m": "1",
         "k": "Dk", "n": "DvH"},
        {"label": "o_t=S_t^T·q_t（逐 token 矩阵-向量）", "b": "B*NvH*seqlen", "m": "1",
         "k": "Dk", "n": "DvH"},
        {"label": "S_t=A+k_t·δ_tᵀ（逐 token rank-1 更新）", "b": "B*NvH*seqlen",
         "m": "Dk", "k": "1", "n": "DvH"},
    ], ("gated delta-rule 逐 token 等价数学核：衰减 A=S⊙exp(g)（逐元素）、"
        "mem=A^T·k_t、δ=β⊙(v-mem)（逐元素）、rank-1 写回 S_t=A+k_t·δ_tᵀ、"
        "输出 o_t=S_t^T·q_t。**每个 token 的状态矩阵不同**，矩阵几何按 token 展开："
        "两个矩阵-向量项 B_eff=B·NvH·seqlen、M=1、K=Dk、N=DvH；rank-1 项 B_eff 同上、"
        "M=Dk、K=1、N=DvH。三个矩阵项各占一行，总量=3·B·NvH·seqlen·Dk·DvH"
        "（与原口径相同）；逐元素伴随项不另收费。"
        "实际实现走 chunk（64 长度 pad/三角 mask/分块并行），本轮只做跨 XPU 理论粗估，"
        "采用递推数学核，不计算法改写附加工作，也不声称 chunk 实际 MAC 与递推相同。")),
    "g_gdn_state_write": vec("B*NvH", "Dk*DvH",
                             "recurrent state [B,NvH,Dk,DvH] 的写回规模；"
                             "状态大小固定、不随 kvcachelen 增长。" ),
    "g_gdn_o_norm": vec("B*seqlen*NvH", "DvH",
                        "gated RMSNorm：按 head_dim DvH 归一一次再乘 sigmoid(z)。"
                        + _RMS_NOTE),
    "g_gdn_o_reshape": vec("B*seqlen", "Dz", "把 NvH×DvH 拼成 6144。" ),
    "g_gdn_out_proj": cube("B", "seqlen", "Dz", "H",
                           "out_proj：Linear(Dz,H)，K=6144、N=H=2560。" ),
}

# ---------------------------------------------------------------- QSA
_QSA = {
    "g_qsa_q_proj": cube("B", "seqlen", "H", "2*Nh*Dh",
                         "self.q_proj：一次 Linear(H,2·Nh·Dh=12288) 同时出 q 与输出门。" ),
    "g_qsa_q_gate_view": vec("B*seqlen", "2*Nh*Dh", "view 成 [B,S,Nh,2·Dh]。" ),
    "g_qsa_q_gate_chunk": vec("B*seqlen*Nh", "2*Dh",
                              "沿最后一维均分成 q/ gate 两个 Dh。" ),
    "g_qsa_gate_view": vec("B*seqlen", "Nh*Dh", "gate reshape 成 [B,S,Dqw=6144]。" ),
    "g_qsa_q_norm": vec("B*seqlen*Nh", "Dh", "Q 用 RMSNorm(Dh)；V 不归一。" + _RMS_NOTE),
    "g_qsa_q_transpose": vec("B*seqlen*Nh", "Dh", "transpose 成 [B,Nh,S,Dh]。" ),
    "g_qsa_k_proj": cube("B", "seqlen", "H", "Dkv",
                         "self.k_proj：Linear(H,Dkv=512)，2 个 KV 头。" ),
    "g_qsa_k_view": vec("B*seqlen", "Dkv", "view 成 [B,S,Nkv,Dh]。" ),
    "g_qsa_k_norm": vec("B*seqlen*Nkv", "Dh", "K 用 RMSNorm(Dh)。" + _RMS_NOTE),
    "g_qsa_k_transpose": vec("B*seqlen*Nkv", "Dh", "transpose 成 [B,Nkv,S,Dh]。" ),
    "g_qsa_v_proj": cube("B", "seqlen", "H", "Dkv",
                         "self.v_proj：Linear(H,Dkv)，与 K 同宽、没有 norm。" ),
    "g_qsa_v_view": vec("B*seqlen*Nkv", "Dh", "view+transpose 成 [B,Nkv,S,Dh]。" ),
    "g_qsa_q_rope": vec("B*seqlen*Nh", "Dh",
                        "partial RoPE（Drot=64/256）：按整张量元素数计一次"
                        "（用户粗规则按元素数、不按旋转分量比例折算）。" ),
    "g_qsa_k_rope": vec("B*seqlen*Nkv", "Dh", "K 的 partial RoPE（只转前 64 维）。" ),
    "g_qsa_cache_update": vec("B*seqlen", "2*Nkv*Dh",
                              "只计本次写入的 K 与 V 新行（2×Nkv×Dh）；"
                              "QSA 缓存随历史线性增长。" ),
    "g_qsa_ix_qk_proj": cube("B", "seqlen", "H", "(Iqh+Ikh)*Id",
                             "indexer.index_qk_proj：Linear(H,(Iqh+Ikh)·Id=640)。" ),
    "g_qsa_ix_split": vec("B*seqlen", "(Iqh+Ikh)*Id",
                          "拆成 q(4 头) 与 token key(1 头) 的 [B,S,Iqh+Ikh,Id]。" ),
    "g_qsa_ix_q_norm": vec("B*seqlen*Iqh", "Id", "indexer q 的 RMSNorm(128)。" ),
    "g_qsa_ix_q_rope": vec("B*seqlen*Iqh", "Id",
                           "indexer q 的 partial RoPE（64/128），按整张量计一次。" ),
    "g_qsa_ix_cache_update": vec("B*seqlen", "Id",
                                 "只计本次追加的 indexer raw key 行 [B,Lidx,Id]。" ),
    "g_qsa_ix_visible": vec("B*seqlen", "Lkv",
                            "逐 query 的 nonzero(可见位置)：每次扫描 Lkv 个 mask 元素。" ),
    "g_qsa_ix_query_guard": vec("B*seqlen", "1", "N_q>0 的逐 query 守卫（标量）。" ),
    "g_qsa_ix_block_view": vec(f"B*{_SUM_VQ}", "1",
                               "把每 query 的可见索引 view 成 (N_q,Kc)："
                               "按 B·ΣV_q 计（ΣV_q 是单 batch 内各 query 可见长之和，"
                               "不枚举 j）。" ),
    "g_qsa_ix_pool": vec(f"B*Kc*{_SUM_NQ}", "Id",
                         "逐 query 均值池化：读 N_q·Kc 个可见 token 的 Id 维 key；"
                         "总量=B·Kc·ΣN_q·Id（ΣN_q 为单 batch 内各 query 完整块数之和）。" ),
    "g_qsa_ix_pool_norm": vec(f"B*{_SUM_NQ}", "Id",
                              "只对 pooled key 做 RMSNorm(Id)：ΣN_q·Id。" ),
    "g_qsa_ix_block_rope": vec(f"B*{_SUM_NQ}", "Id",
                               "块 key 的 partial RoPE（位置取块首 token）：ΣN_q·Id。" ),
    "g_qsa_ix_scores": cube("B*Iqh", "1", "Id", _SUM_NQ,
                            "逐 query 的 q[Iqh,Id] × block_keys[N_q,Id]ᵀ："
                            "每次一个矩阵-向量，M=1、K=Id、N=N_q；"
                            "ΣN_q 是**单 batch 内**对 S 个查询的精确和，"
                            "再乘 batch（B·Iqh 已含），不用 Ksel_last·S。" ),
    "g_qsa_ix_block_topk": vec(f"B*{_SUM_NQ}", "1",
                               "逐 query 在 N_q 个块分数上 topk(min(Btop,N_q))；"
                               "按块分数元素数 ΣN_q 扫描。" ),
    "g_qsa_ix_select_blocks": vec(f"B*Kc*{_SUM_BQ}", "1",
                                  "逐 query 取回 B_q 个块的 Kc 个原始 token 索引："
                                  "B·Kc·ΣB_q（ΣB_q=min(Btop,N_q) 的单 batch 精确和）。" ),
    "g_qsa_ix_tail": vec(f"B*{_SUM_TAIL}", "1",
                         "逐 query 的不完整 tail：B·Σ(V_q % Kc)=B·Σtail（单 batch 内求和）。" ),
    "g_qsa_ix_concat": vec_terms([
        {"label": "选中块展开（Kc·ΣB_q）", "m": f"B*Kc*{_SUM_BQ}", "n": "1"},
        {"label": "tail 追加（Σtail）", "m": f"B*{_SUM_TAIL}", "n": "1"},
    ], ("selected_tokens 由“块展开”与“tail”两段拼接；两段各占一行，"
        "合计=Kc·ΣB_q+Σtail（=ΣKsel），不用固定 Kout 或 K_sel_last 冒充。")),
    "g_qsa_ix_pad": vec("B*seqlen", "Kout",
                        "把有效选中项 pad 到固定索引宽度 Kout=Ki+Kc-1（含 -1 填充）："
                        "逻辑选中数与 pad 后的轴分开表达。" ),
    "g_qsa_ix_mask_bool": vec("B*seqlen", "Lkv+Kout",
                              "selected_token_mask 的 zeros[·,Lkv+1] + scatter(Kout) "
                              "+ 切片；按 Lkv+Kout 计。" ),
    "g_qsa_ix_mask_float": vec("B*seqlen", "Lkv",
                               "eager 路径 where(bool,0,min) 覆盖 [B,1,S,Lkv]。" ),
    "g_qsa_mask_combine": vec("B*seqlen", "Lkv",
                              "稀疏选择 mask 与主 causal mask 相加（只收窄）。" ),
    "g_qsa_repeat_kv": vec("B*Nh*Lkv", "2*Dh",
                           "repeat_kv 把 2 个 KV 头扩成 24 头：K/V 各一份，2×Dh。" ),
    "g_qsa_qk": cube("B*Nh", "seqlen", "Dh", "Lkv",
                     "主注意力仍是**完整 Lkv**：头数作 batch、M=S、K=Dh、N=Lkv；"
                     "indexer 的稀疏选择只通过 mask 生效，不能用 Ksel 当收缩轴。" ),
    "g_qsa_mask_add": vec("B*Nh*seqlen", "Lkv", "mask 加到 [B,Nh,S,Lkv] 打分。" ),
    "g_qsa_softmax": vec("B*Nh*seqlen", "Lkv", "softmax 在全部 Lkv 上。" ),
    "g_qsa_av": cube("B*Nh", "seqlen", "Lkv", "Dh", "AV 也是完整 Lkv。" ),
    "g_qsa_attn_transpose": vec("B*seqlen*Nh", "Dh", "输出 transpose 回序列维在前。" ),
    "g_qsa_attn_reshape": vec("B*seqlen", "Dqw", "Nh×Dh 拼成 6144。" ),
    "g_qsa_gate_sigmoid": vec("B*seqlen", "Dqw", "输出门 sigmoid。" ),
    "g_qsa_gate_mul": vec("B*seqlen", "Dqw", "attn·sigmoid(gate) 逐元素。" ),
    "g_qsa_o_proj": cube("B", "seqlen", "Dqw", "H", "self.o_proj：Linear(Dqw,H)。" ),
}

# ---------------------------------------------------------------- MoE
_MOE = {
    "g_moe_flatten": vec("Ttok", "H", "hidden.view(Ttok,H) 的元素数。" ),
    "g_moe_router": cube("B", "seqlen", "H", "E",
                         "F.linear(x,W)：K=H、N=E=512（softmax 路由，无 correction bias）。" ),
    "g_moe_softmax": vec("Ttok", "E", "softmax(router_logits)。" ),
    "g_moe_topk": vec("Ttok", "E", "每 token 对 E 个概率 topk(K=10)。" ),
    "g_moe_weight_norm": vec("Ttok", "K", "top-10 权重归一（无 routed scaling）。" ),
    "g_moe_expert_mask": vec("(E+1)*K", "Ttok", "one_hot+permute 的完整分配矩阵物化。" ),
    "g_moe_expert_hit": vec("(E+1)*K", "Ttok", "对分配矩阵 sum+nonzero 求命中集合。" ),
    "g_moe_expert_gather_indices": engine.expert_hit_rule(
        experts="E", top_k="K", tokens="Ttok", slice_rows="K", lower_text="K",
        note=("对每个命中专家取 (top_k_pos, token_idx)：每次扫描 [K,Ttok]；"
              "总量=N_hit·K·Ttok。单值是各层统一命中数假设，"
              "实际路由分布不由本工具推断。")),
    "g_moe_select_tokens": vec("Ttok*K", "H", "按总分配行数 gather token 行。" ),
    "g_moe_expert_gate_up": cube("1", "Ttok*K", "H", "2*Es",
                                 "gate/up 打包 [E,2Es,H] 的合并 Linear："
                                 "M=Ttok·K、K=H、N=2·Es=1280。" ),
    "g_moe_expert_chunk": vec("Ttok*K", "2*Es", "拆成 gate/up 各 640。" ),
    "g_moe_expert_act": vec("Ttok*K", "Es", "SiLU(gate)·up（Qwen 无 clamp）。" ),
    "g_moe_expert_down": cube("1", "Ttok*K", "Es", "H", "down_proj[e]：K=Es、N=H。" ),
    "g_moe_expert_weight": vec("Ttok*K", "H", "按 (token_idx,top_k_pos) 加权。" ),
    "g_moe_result_init": vec("Ttok", "H", "zeros_like(Ttok,H)。" ),
    "g_moe_index_add": vec("Ttok*K", "H", "按 token_idx 累加各分配贡献。" ),
    "g_moe_shared_gate": cube("B", "seqlen", "H", "Ess", "共享专家 gate_proj。" ),
    "g_moe_shared_up": cube("B", "seqlen", "H", "Ess", "共享专家 up_proj。" ),
    "g_moe_shared_act": vec("B*seqlen", "Ess", "SiLU(gate)·up（共享专家）。" ),
    "g_moe_shared_down": cube("B", "seqlen", "Ess", "H", "共享专家 down_proj。" ),
    "g_moe_shared_gate_scalar": cube("B", "seqlen", "H", "1",
                                     "shared_expert_gate：Linear(H,1) 标量门。" ),
    "g_moe_shared_gated": vec("B*seqlen", "H", "sigmoid(标量门) 广播乘共享专家输出。" ),
    "g_moe_merge": vec("Ttok", "H", "routed 专家输出 + 共享专家输出（展平域）。" ),
    "g_moe_merge_view": vec("Ttok", "H", "view 回 [B,seqlen,H]。" ),
}

# ---------------------------------------------------------------- PLE（layer 1）
_PLE = {
    "g_ple_context_update": vec("B", "Lctx", "上下文缓存写回（最近 2 个 token id）。" ),
    "g_ple_history": vec("B*(seqlen+Lctx)", "1",
                         "cat([previous_context, input_ids]) 的 token 数；"
                         "历史 2 位只参与 n-gram 哈希。" ),
    "g_ple_shift1": vec("B*(seqlen+Lctx)", "1", "右移 1，跨 EOS 位置填 eos。" ),
    "g_ple_shift2": vec("B*(seqlen+Lctx)", "1", "右移 2，跨 EOS 位置填 eos。" ),
    "g_ple_bigram": vec("B*(seqlen+Lctx)", "Hng",
                        "bigram 哈希（8 头）：xor/乘子/mod/偏移。" ),
    "g_ple_trigram": vec("B*(seqlen+Lctx)", "Hng",
                         "trigram 哈希（8 头）：当前+前 1+前 2。" ),
    "g_ple_concat_ids": vec("B*(seqlen+Lctx)", "Ngh",
                            "bigram 8 + trigram 8 的 id 拼接（16 头）。" ),
    "g_ple_slice": vec("B*seqlen", "Ngh", "只取当前 seqlen 的 n-gram id。" ),
    "g_ple_gather": vec("B*seqlen*Ngh", "Dgh",
                        "n-gram 查表：每头一个 160 维 embedding；"
                        "不按词表大小乘。" ),
    "g_ple_flatten": vec("B*seqlen", "Dple", "16×160=2560 拼宽。" ),
    "g_ple_key_proj": cube("B", "seqlen", "Dple", "Dqkv",
                           "key_proj：Linear(Dple,Dqkv=10240)。" ),
    "g_ple_norm_key": vec("B*seqlen", "Dqkv", "4 路流各自归一（group_size=H）。"
                          + _RMS_NOTE),
    "g_ple_key_view": vec("B*seqlen", "Dqkv", "unflatten 成 [B,S,Hc,H]。" ),
    "g_ple_value_proj": cube("B", "seqlen", "Dple", "H",
                             "value_proj：Linear(Dple,H=2560) 单流 value。" ),
    "g_ple_norm_query": vec("B*seqlen", "Dqkv", "query 就是 4 路 flat 残差流，分组归一。"
                            + _RMS_NOTE),
    "g_ple_query_view": vec("B*seqlen", "Dqkv", "unflatten 成 [B,S,Hc,H]。" ),
    "g_ple_gate": cube("B*seqlen*Hc", "1", "H", "1",
                       "每个 (batch, token, 流) 各自一份 key/query，逐 token 独立点积："
                       "B_eff=B·seqlen·Hc、M=1、K=H、N=1；"
                       "不是跨 token 共用的 [seqlen·Hc,H] 单矩阵。" ),
    "g_ple_gate_sign": vec("B*seqlen", "Hc", "abs/clamp_min/sqrt/sign 的每流标量。" ),
    "g_ple_gated_value": vec("B*seqlen*Hc", "H", "sigmoid(门)·value 广播到 [B,S,Hc,H]。" ),
    "g_ple_gated_value_flat": vec("B*seqlen", "Dqkv", "flatten 成 flat 10240。" ),
    "g_ple_norm_conv": vec("B*seqlen", "Dqkv", "卷积前的分组 RMSNorm。" + _RMS_NOTE),
    "g_ple_mask": vec("B*seqlen", "2*Dqkv",
                      "gated_value 与 normed 两个张量按 padding mask 置零。" ),
    "g_ple_conv_state_update": vec("B", "Dqkv*Lpconv",
                                   "PLE 短卷积 state [B,Dqkv,Lpconv=9] 的写回；"
                                   "状态大小固定。" ),
    "g_ple_conv": cube("B*Dqkv", "seqlen", "Kple", "1",
                       "depthwise conv1d（dilation=D_dilate=3、核长 Kple=4）："
                       "groups=Cin=Cout=Dqkv、B_eff=batch·groups、M=seqlen、"
                       "K=(Cin/groups)·核长=Kple、N=1；有效感受野 10。" ),
    "g_ple_value_mask_flatten": vec("B*seqlen", "Dqkv",
                                    "加回前把 gated_value 保持 flat。" ),
    "g_ple_add": vec("B*seqlen", "Dqkv", "gated_value + short_conv(normed)。" ),
    "g_ple_stream_add": vec("B*seqlen", "Dqkv",
                            "PLE 作为前置加性支路加回 4 路 flat 残差流。" ),
}

_RULES_BY_GROUP = {
    "gdn-moe": {**_HC_ATTN, **_GDN, **_HC_FFN, **_MOE},
    "gdn-ple-moe": {**_PLE, **_HC_ATTN, **_GDN, **_HC_FFN, **_MOE},
    "qsa-moe": {**_HC_ATTN, **_QSA, **_HC_FFN, **_MOE},
}

# ------------------------------------------------------------------ 整网
_NETWORK = {
    "net_embed": vec("B*seqlen", "H", "token→[B,seqlen,H] 嵌入查表与写出。" ),
    "net_position_ids": vec("4*B*seqlen", "1",
                            "默认 4 行 position_ids：[4,B,seqlen] 个标量；"
                            "text-only 时 4 行相同但仍物化。" ),
    "net_mask_build": vec("B*seqlen", "Lkv+1",
                          "一次构建两套 mask：4D causal [B,1,S,Lkv] 与 2D conv mask "
                          "[B,S]；按 Lkv+1 计。" ),
    "net_ple_input_ids": vec("B*seqlen", "1",
                             "PLE 用 token id：padding 位置替换成 eos。" ),
    "net_hidden_repeat": vec("B*seqlen*Hc", "H",
                             "单流 embedding 沿最后一维复制 4 份的物化规模。" ),
    "net_layer_loop": engine.container(
        "48 层循环容器：36 层 GDN（含 layer 1 的 PLE）+ 12 层 QSA，"
        "层内成本由三组层型小计×真实层数串行汇总，本行不重复计成本。"),
    "net_mixer_mix": cube_terms([
        {"label": "读门低秩下投影（Hc·H→Rhc）", "b": "B", "m": "seqlen",
         "k": "Hc*H", "n": "Rhc"},
        {"label": "读门上投影（Rhc→Hc·H）", "b": "B", "m": "seqlen",
         "k": "Rhc", "n": "Hc*H"},
    ], ("网络末端 hyper_connection_mixer：分组 RMSNorm → 低秩下投影 → SiLU → 上投影 → "
        "sigmoid 读门 → 4 路加权 mean 收拢成单流。两个矩阵项各占一行；"
        "norm/sigmoid/mean 不另收费。本模型没有 Final RMSNorm。")),
    "net_logits_slice": vec("B*Ssel", "H", "只对保留位置做后续投影。" ),
    "net_lm_head": cube("B", "Ssel", "H", "Vocab", "LM Head：K=H、N=Vocab=248320。" ),
}

_NETWORK_SEGMENTS = (
    {"id": "entry", "label": "整网入口外围",
     "operator_ids": ["net_embed", "net_position_ids", "net_mask_build",
                      "net_ple_input_ids", "net_hidden_repeat"],
     "subtotal": True},
    {"id": "loop", "label": "48 层循环（容器，不单独计成本）",
     "operator_ids": ["net_layer_loop"], "subtotal": False},
    {"id": "exit", "label": "整网出口外围",
     "operator_ids": ["net_mixer_mix", "net_logits_slice", "net_lm_head"],
     "subtotal": True},
)


def block_rule_groups() -> dict:
    return _RULES_BY_GROUP


def block_rules(group_id: str) -> dict[str, dict]:
    if group_id not in _RULES_BY_GROUP:
        raise ValueError(f"未知 Qwen3.8-Flash-Next 层组 id {group_id!r}：没有 CUBE/VEC 规则")
    return _RULES_BY_GROUP[group_id]


def network_rules() -> dict[str, dict]:
    return _NETWORK


def network_segments() -> tuple:
    return tuple(dict(segment) for segment in _NETWORK_SEGMENTS)


def serial_order() -> list[dict]:
    return [
        {"kind": "segment", "group": "net-periphery", "segment": "entry"},
        {"kind": "group", "group": "gdn-moe"},
        {"kind": "group", "group": "gdn-ple-moe"},
        {"kind": "group", "group": "qsa-moe"},
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
                          "层内结构仍由三个层组代表层 Block 表达，不铺开 48 层。"),
        "scope_note": ("范围：一次 forward 的入口外围（embedding/4 行位置/"
                       "两套 mask/PLE token id/4 路 Gated Residual 展开）+ "
                       "出口外围（hyper_connection_mixer/logits 切片/LM Head）；"
                       "48 层循环本体不在此块计成本。"),
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
