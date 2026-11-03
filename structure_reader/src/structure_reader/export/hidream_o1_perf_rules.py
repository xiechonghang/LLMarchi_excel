"""HiDream-O1-Image 的 CUBE/VEC 粗估规则（声明式，由唯一导出器消费）。

用户 2026-10-29 确认的规则：CUBE ``B·M·K·N/吞吐``（默认 4096 MAC/ms）、
VEC ``M·N/吞吐``（默认 64 元素/ms），按比例、不向 64 取整。

本模块只声明 HiDream-O1-Image（单次 forward）自己的语义；构造器/校验来自
``export/perf_engine.py``，不解析模型、不写工作簿、不计算耗时。

源码口径（官方 GitHub `models/qwen3_vl_transformers.py` / `utils`）：

- 单次 forward：文本 token + 像素 patch + 时间步 → x_pred。采样步数/CFG/scheduler/
  噪声注入在外部，不乘步数。
- 文本主干 36 层：**两趟注意力**——AR 文本行做 causal 子集 [B,Nh,N_ar,N_ar]，
  全序列再做 non-causal [B,Nh,S_tot,S_tot]（AR 行稍后被第 1 趟结果覆盖）；
  两趟都是真实执行，分别计入。
- 视觉塔 27 blocks：**仅 Kref>0 执行**（t2i 时整段跳过）。逐图分段做非因果注意力，
  段长 S_i=Svis/Kref（要求 Kref 段等长且 Kref·S_i=Svis）；不等长/缺失时相关项未知。
  条件判定优先于分支专有尺寸：Kref=0 时这些行在表内计 0（条件），整网汇总对
  vit-tower 的贡献也按条件直接记 0，不因 S_i 空白而给主网总计丢未知。
- DeepStack：decoder 前 3 层（layer_idx < 3）把 3 个 tap 注入条件图占位行；
  这不是每层都执行，层模板×36 会重复计 33 次，因此层组内该行只作引用，
  整网在 ``net_deepstack_inject`` 一次性按 3 个 tap 收费（含 ViT 侧 tap 读取）。
- 视觉塔 block 的 DeepStack tap 只在 block 8/16/24 读取，同样在整网行收一次。
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
    "DEEPSTACK_OP_ID",
]

ENGINE = engine.engine_defaults()
PERFORMANCE_STATUS = engine.PERFORMANCE_STATUS
PERFORMANCE_NOTE = engine.PERFORMANCE_NOTE
THROUGHPUT_NOTE = engine.THROUGHPUT_NOTE
DISCLAIMER = engine.disclaimer("HiDream-O1-Image")

DEEPSTACK_OP_ID = "net_deepstack_inject"

_RMS_NOTE = "RMSNorm 按工作张量元素数计一次（读+归一写）；不反映真实 kernel 融合。"
_LN_NOTE = "LayerNorm（均值+方差，有 weight/bias）按工作张量元素数计一次。"

# 参考图/视觉塔分支的执行条件：先判 Kref，再依赖该分支专有尺寸。
_REF_CONDITION = {
    "when": "Kref >= 1",
    "when_text": "Kref=0（t2i）时参考图/视觉塔分支不执行，本项计 0",
}
# 条件一致性：Kref=0 必须 Svis=0；Kref>0 必须段等长且 Kref·S_i=Svis。
_REF_GUARD = [
    "((Kref == 0) and (Svis == 0)) or "
    "((Kref >= 1) and (Svis >= 1) and (S_i >= 1) and (Kref*S_i == Svis))",
]
_REF_GUARD_TEXT = ("Kref=0 要求 Svis=0（分支不执行）；Kref>0 要求 Svis>=1、S_i>=1 且 "
                   "Kref·S_i=Svis（逐图等长）。不满足即未知，不把不同长度平均成单值。")
# 不使用逐图段长的整网任务/层内注入：只需要 Kref 与 Svis 的条件一致性。
_NET_GUARD = [
    "((Kref == 0) and (Svis == 0)) or ((Kref >= 1) and (Svis >= 1))",
]
_NET_GUARD_TEXT = ("Kref=0 要求 Svis=0（分支不执行）；Kref>0 要求 Svis>=1。"
                   "不满足即未知，不按 0 或单值平均。")
# 每图 patch 数：由 Svis 与 Kref 推出的等长段长（不直接用 S_i，以免 Kref=0 时
# 分支专有输入空白污染 0）。
_SI = "(Svis//max(Kref,1))"
_DEEPSTACK_CHARGE = {
    "mode": "excluded",
    "charged_at": DEEPSTACK_OP_ID,
    "reason": ("该算子只在 decoder 前 3 层（layer_idx<3）执行；层模板×36 会重复计 33 次，"
               "因此本行只作引用，整网在 net_deepstack_inject 按 3 个 tap 收一次。"),
}
# 整网条件（层组总计/串行贡献使用）：除 Kref 外还要求 Kref=0 时 Svis=0，
# 这样“Kref=0 但 Svis>0”这种矛盾输入不会被静默当成 0 分支。
_EXEC_CONDITION = {
    "when": "Kref >= 1",
    "when_text": "Kref=0（t2i）时参考图/视觉塔分支不执行，本项计 0",
    "guards": ["(Kref != 0) or (Svis == 0)"],
}
_TAP_CHARGE = {
    "mode": "excluded",
    "charged_at": DEEPSTACK_OP_ID,
    "reason": ("tap 读取只发生在 block 8/16/24（3/27）；层模板×27 会重复计 24 次，"
               "因此本行只作引用，整网在 net_deepstack_inject 按 3 个 tap 收一次。"),
}

# ---------------------------------------------------------------- 文本主干 36 层
_DECODER = {
    "d_in_norm": vec("B*Stot", "H", "逐 token RMSNorm(H)，没有时间步调制。" + _RMS_NOTE),
    "d_q_proj": cube("B", "Stot", "H", "Hq", "q_proj：Linear(H,Hq=4096)，无 bias。" ),
    "d_q_view": vec("B*Stot", "Hq", "q.view(B,S,Nh,Dh) 的元素数。" ),
    "d_q_norm": vec("B*Stot*Nh", "Dh", "逐头 RMSNorm(Dh)（QK-norm）。" + _RMS_NOTE),
    "d_q_rope": vec("B*Stot*Nh", "Dh",
                    "MRoPE 作用在整 128 维（按元素数计一次，不按旋转分量比例折算）。" ),
    "d_k_proj": cube("B", "Stot", "H", "Hkv",
                     "k_proj：Linear(H,Hkv=1024)，8 个 KV 头、无 bias。" ),
    "d_k_view": vec("B*Stot", "Hkv", "k.view(B,S,Nkv,Dh) 的元素数。" ),
    "d_k_norm": vec("B*Stot*Nkv", "Dh", "逐头 RMSNorm(Dh)。" + _RMS_NOTE),
    "d_k_rope": vec("B*Stot*Nkv", "Dh", "key 侧 MRoPE（V 不做）。" ),
    "d_v_proj": cube("B", "Stot", "H", "Hkv", "v_proj：Linear(H,Hkv)，无 bias。" ),
    "d_v_view": vec("B*Stot", "Hkv", "view+transpose 的元素数；V 不做 norm/RoPE。" ),
    "d_gather_ar": vec("B*(Nh+2*Nkv)*Nar", "Dh",
                       "三份 index-select：q[Nh]、k[Nkv]、v[Nkv] 的 AR 行子集；"
                       "第 1 趟只在 AR 行上算。" ),
    "d_qk_ar": cube("B*Nh", "Nar", "Dh", "Nar",
                    "第 1 趟 QK：AR 行 causal 子集 [B,Nh,N_ar,N_ar]；"
                    "M=N_ar、K=Dh、N=N_ar。" ),
    "d_softmax_ar": vec("B*Nh*Nar", "Nar", "AR 子集 softmax。" ),
    "d_av_ar": cube("B*Nh", "Nar", "Nar", "Dh", "AR 子集 AV → [B,N_ar,Nh,Dh]。" ),
    "d_qk_full": cube("B*Nh", "Stot", "Dh", "Stot",
                      "第 2 趟 QK：全序列 non-causal [B,Nh,S_tot,S_tot]；"
                      "O(S_tot²) 的规模就在这里。" ),
    "d_softmax_full": vec("B*Nh*Stot", "Stot", "全序列双向 softmax（无 mask 加法）。" ),
    "d_av_full": cube("B*Nh", "Stot", "Stot", "Dh", "全序列 AV。" ),
    "d_merge_ar": vec("B*Nar*Nh", "Dh",
                      "把第 1 趟 AR 行结果写回（index-copy 的 [N_ar,Nh,Dh] 规模）。" ),
    "d_head_merge": vec("B*Stot", "Hq", "32 个头拼回 4096 维。" ),
    "d_o_proj": cube("B", "Stot", "Hq", "H", "o_proj：Linear(Hq,H)，无 bias。" ),
    "d_res_attn": vec("B*Stot", "H", "普通标量残差相加。" ),
    "d_post_norm": vec("B*Stot", "H", "第二个 RMSNorm 站点。" + _RMS_NOTE),
    "d_gate_proj": cube("B", "Stot", "H", "Fi", "gate_proj：Linear(H,Fi=12288)。" ),
    "d_up_proj": cube("B", "Stot", "H", "Fi", "up_proj：与 gate 并行、同形。" ),
    "d_silu": vec("B*Stot", "Fi", "SiLU 只作用于 gate 支路。" ),
    "d_swiglu_mul": vec("B*Stot", "Fi", "SiLU(gate)·up（无 clamp）。" ),
    "d_down_proj": cube("B", "Stot", "Fi", "H", "down_proj：Linear(Fi,H)。" ),
    "d_res_ffn": vec("B*Stot", "H", "第二个残差站点。" ),
    "d_deepstack_add": vec("B*Nvis", "H",
                           "把 1 个 tap 注入条件图占位行（每 batch 每 tap 的 Svism×H）；"
                           "只在前 3 层执行，整网在 net_deepstack_inject 按 3 个 tap 收一次。",
                           conditional=_REF_CONDITION, guards=_NET_GUARD,
                           guard_text=_NET_GUARD_TEXT, charge=_DEEPSTACK_CHARGE),
}

# ---------------------------------------------------------------- 视觉塔 27 blocks
_VIT = {
    "v_norm1": vec("Svis", "Hv", "LayerNorm(Hv)，与文本主干 RMSNorm 不同。" + _LN_NOTE,
                   conditional=_REF_CONDITION, guards=_REF_GUARD,
                   guard_text=_REF_GUARD_TEXT),
    "v_qkv_proj": cube("1", "Svis", "Hv", "Hv3",
                       "qkv 合并一次 Linear(Hv,3·Hv=3456)，bias=True。",
                       conditional=_REF_CONDITION, guards=_REF_GUARD,
                       guard_text=_REF_GUARD_TEXT),
    "v_qkv_split": vec("Svis", "Hv3", "一次 view+permute 拆成 q/k/v 三块。",
                       conditional=_REF_CONDITION, guards=_REF_GUARD,
                       guard_text=_REF_GUARD_TEXT),
    "v_q_rope": vec("Nhv*Svis", "Dhv", "2D rotary（行/列各 36 维）作用在整 72 维。",
                    conditional=_REF_CONDITION, guards=_REF_GUARD,
                    guard_text=_REF_GUARD_TEXT),
    "v_k_rope": vec("Nhv*Svis", "Dhv", "k 的 2D rotary（v 不做）。",
                    conditional=_REF_CONDITION, guards=_REF_GUARD,
                    guard_text=_REF_GUARD_TEXT),
    "v_seg_split": vec("Svis", "Nhv*Dhv",
                       "按 cu_seqlens 逐图切段；段长 S_i=Svis/Kref（等长前提见守卫）。",
                       conditional=_REF_CONDITION, guards=_REF_GUARD,
                       guard_text=_REF_GUARD_TEXT),
    "v_qk": cube("Nhv*Kref", _SI, "Dhv", _SI,
                 "逐图非因果 QK：第 i 段 [Nhv,S_i,S_i]，总量 Σ_i Nhv·S_i²·Dh；"
                 "等长时 = Kref·Nhv·S_i²·Dh；不是把不同图拼成一个全连接矩阵。",
                 conditional=_REF_CONDITION, guards=_REF_GUARD,
                 guard_text=_REF_GUARD_TEXT),
    "v_softmax": vec(f"Nhv*Kref*{_SI}", _SI, "逐段 softmax（非因果、无 padding）。",
                     conditional=_REF_CONDITION, guards=_REF_GUARD,
                     guard_text=_REF_GUARD_TEXT),
    "v_av": cube("Nhv*Kref", _SI, _SI, "Dhv", "逐段 AV，输出 [S_i,Nhv,Dhv]。",
                 conditional=_REF_CONDITION, guards=_REF_GUARD,
                 guard_text=_REF_GUARD_TEXT),
    "v_seg_concat": vec("Svis", "Nhv*Dhv", "Kref 段按原顺序拼回总长 Svis。",
                        conditional=_REF_CONDITION, guards=_REF_GUARD,
                        guard_text=_REF_GUARD_TEXT),
    "v_head_merge": vec("Svis", "Hv", "16 个头拼回 1152 维。",
                        conditional=_REF_CONDITION, guards=_REF_GUARD,
                        guard_text=_REF_GUARD_TEXT),
    "v_proj": cube("1", "Svis", "Hv", "Hv", "proj：Linear(Hv,Hv)，bias=True。",
                   conditional=_REF_CONDITION, guards=_REF_GUARD,
                   guard_text=_REF_GUARD_TEXT),
    "v_res1": vec("Svis", "Hv", "普通标量残差。",
                  conditional=_REF_CONDITION, guards=_REF_GUARD,
                  guard_text=_REF_GUARD_TEXT),
    "v_norm2": vec("Svis", "Hv", "MLP 前 LayerNorm。" + _LN_NOTE,
                   conditional=_REF_CONDITION, guards=_REF_GUARD,
                   guard_text=_REF_GUARD_TEXT),
    "v_fc1": cube("1", "Svis", "Hv", "Fv", "linear_fc1：Linear(Hv,Fv=4304)，bias=True。",
                  conditional=_REF_CONDITION, guards=_REF_GUARD,
                  guard_text=_REF_GUARD_TEXT),
    "v_gelu": vec("Svis", "Fv", "GELU（tanh 近似），不是 SiLU。",
                  conditional=_REF_CONDITION, guards=_REF_GUARD,
                  guard_text=_REF_GUARD_TEXT),
    "v_fc2": cube("1", "Svis", "Fv", "Hv", "linear_fc2：Linear(Fv,Hv)，bias=True。",
                  conditional=_REF_CONDITION, guards=_REF_GUARD,
                  guard_text=_REF_GUARD_TEXT),
    "v_res2": vec("Svis", "Hv", "第二个残差站点。",
                  conditional=_REF_CONDITION, guards=_REF_GUARD,
                  guard_text=_REF_GUARD_TEXT),
    "v_deepstack_tap": vec("Svis", "Hv",
                           "DeepStack tap 读取（block 8/16/24）；整网在 "
                           "net_deepstack_inject 按 3 个 tap 收一次。",
                           conditional=_REF_CONDITION, guards=_REF_GUARD,
                           guard_text=_REF_GUARD_TEXT, charge=_TAP_CHARGE),
}

# ---------------------------------------------------------------- 整网一次性任务
_NETWORK = {
    "net_text_embed": vec("B*Ptxt", "H", "文本 token 查表与写出。" ),
    "net_t_freq": vec("B", "Fq", "时间步正弦编码（每 batch 一个 Fq 向量）。" ),
    "net_t_lin1": cube("B", "1", "Fq", "H", "t_mlp[0]：Linear(Fq,H)，bias=True。" ),
    "net_t_silu": vec("B", "H", "单层 SiLU。" ),
    "net_t_lin2": cube("B", "1", "H", "H", "t_mlp[2]：Linear(H,H)，bias=True。" ),
    "net_vision_patch": cube("1", "Svis", "Vpin", "Hv",
                             "条件 Kref>0：3D 卷积把每个 patch 的 Cin·Tp·Pv·Pv=1536 "
                             "像素映射到 Hv；按一次卷积收缩计。",
                             conditional=_REF_CONDITION, guards=_NET_GUARD,
                             guard_text=_NET_GUARD_TEXT),
    "net_vision_pos": vec("Svis", "Hv", "条件 Kref>0：按图网格插值加位置嵌入。",
                          conditional=_REF_CONDITION, guards=_NET_GUARD,
                          guard_text=_NET_GUARD_TEXT),
    "net_vision_rope": vec("Svis", "Dhv", "条件 Kref>0：2D rotary cos/sin 生成。",
                           conditional=_REF_CONDITION, guards=_NET_GUARD,
                           guard_text=_NET_GUARD_TEXT),
    "net_vmerge_norm": vec("Svis", "Hv",
                           "条件 Kref>0：主线 merger 的 norm 在 view **之前**，作用在每 token "
                           "的 1152 维上（Svis×Hv，不是 merge 后的 Svism 行）。"
                           + _LN_NOTE,
                           conditional=_REF_CONDITION, guards=_NET_GUARD,
                           guard_text=_NET_GUARD_TEXT),
    "net_vmerge_view": vec("Svis", "Hv", "条件 Kref>0：每 4 个相邻 token 拼成 4608 维单元。",
                           conditional=_REF_CONDITION, guards=_NET_GUARD,
                           guard_text=_NET_GUARD_TEXT),
    "net_vmerge_fc1": cube("1", "Svism", "Hv4", "Hv4",
                           "条件 Kref>0：Linear(4608,4608)，bias=True。",
                           conditional=_REF_CONDITION, guards=_NET_GUARD,
                           guard_text=_NET_GUARD_TEXT),
    "net_vmerge_gelu": vec("Svism", "Hv4", "条件 Kref>0：merger 固定用 GELU。",
                           conditional=_REF_CONDITION, guards=_NET_GUARD,
                           guard_text=_NET_GUARD_TEXT),
    "net_vmerge_fc2": cube("1", "Svism", "Hv4", "H",
                           "条件 Kref>0：Linear(4608,4096)，bias=True，写回文本序列。",
                           conditional=_REF_CONDITION, guards=_NET_GUARD,
                           guard_text=_NET_GUARD_TEXT),
    "net_ds_view": vec("Ndh*Svis", "Hv", "条件 Kref>0：3 个 DeepStack merger 的 view。",
                       conditional=_REF_CONDITION, guards=_NET_GUARD,
                       guard_text=_NET_GUARD_TEXT),
    "net_ds_norm": vec("Ndh*Svism", "Hv4",
                       "条件 Kref>0：postshuffle LayerNorm 作用在 4608 维（view 之后）×3。"
                       + _LN_NOTE,
                       conditional=_REF_CONDITION, guards=_NET_GUARD,
                       guard_text=_NET_GUARD_TEXT),
    "net_ds_fc1": cube("Ndh", "Svism", "Hv4", "Hv4",
                       "条件 Kref>0：3 个 merger 各执行一次 Linear(4608,4608)。",
                       conditional=_REF_CONDITION, guards=_NET_GUARD,
                       guard_text=_NET_GUARD_TEXT),
    "net_ds_gelu": vec("Ndh*Svism", "Hv4", "条件 Kref>0：每个 merger 一次 GELU。",
                       conditional=_REF_CONDITION, guards=_NET_GUARD,
                       guard_text=_NET_GUARD_TEXT),
    "net_ds_fc2": cube("Ndh", "Svism", "Hv4", "H",
                       "条件 Kref>0：Linear(4608,4096)。",
                       conditional=_REF_CONDITION, guards=_NET_GUARD,
                       guard_text=_NET_GUARD_TEXT),
    "net_cond_scatter": vec("B*Svism", "H",
                            "条件 Kref>0：把条件图 embedding 替换到 image placeholder 行；"
                            "t2i 时 pixel_values=None，本算子等价恒等（计 0）。",
                            conditional=_REF_CONDITION, guards=_NET_GUARD,
                            guard_text=_NET_GUARD_TEXT),
    "net_tms_substitute": vec("B", "1", "时间步通过替换 1 个 tms token 的嵌入进入网络。" ),
    "net_patch_lin1": cube("B", "Sgen", "Pin", "Pca",
                           "x_embedder.proj1：Linear(Pin=3072,Pca=1024)，bias=False。" ),
    "net_patch_lin2": cube("B", "Sgen", "Pca", "H",
                           "x_embedder.proj2：Linear(1024,4096)，bias=True；"
                           "目标噪声 patch 与参考图 patch 共用本嵌入。" ),
    "net_concat": vec("B*Stot", "H", "文本在前、生成 token 在后拼接。" ),
    "net_rope": vec("B*Stot", "Dh", "MRoPE 三轴按 [24,20,20] 交织生成 cos/sin。" ),
    "net_attn_partition": vec("B*Stot", "1", "AR/生成行划分 idx_ar（层外只算一次）。" ),
    DEEPSTACK_OP_ID: engine.with_extra({"class": "vec", "terms": [
        {"label": "ViT tap 读取（3 个 tap）", "m": "Ndh*Svis", "n": "Hv"},
        {"label": "decoder 前 3 层注入（3 个 tap）", "m": "Ndh*B*Svism", "n": "H"},
    ], "note": ("整网一次性：DeepStack 的 3 个 tap 从视觉塔（block 8/16/24）读取，"
                "并注入 decoder 前 3 层（layer_idx<3）的条件图占位行。"
                "层组模板（36 层/27 blocks）只作引用，不按模板次数重复计；"
                "两个逻辑工作张量各占一行。Kref=0 时整段不执行（计 0）。"),
        "conditional": dict(_REF_CONDITION), "guards": list(_NET_GUARD),
        "guard_text": _NET_GUARD_TEXT}, None, None, None, None, ""),
    "net_vision_loop": engine.container(
        "27 个视觉塔 block 的循环父节点：block 成本由 vit-tower 层组小计×27 表达，"
        "本行不重复计成本（Kref=0 时整段不执行）。"),
    "net_layer_loop": engine.container(
        "36 层 decoder 循环容器：层内成本由 uit-decoder 层组小计×36 表达，"
        "本行不重复计成本。"),
    "net_final_norm": vec("B*Stot", "H", "Final RMSNorm。" + _RMS_NOTE),
    "net_final_lin": cube("B", "Stot", "H", "Pin",
                          "final_layer2：Linear(4096,3072)，作用于全部 token。" ),
    "net_select_gen": vec("B*Spix", "Pin", "只取生成 token 的 x_pred。" ),
}

_RULES_BY_GROUP = {
    "uit-decoder": _DECODER,
    "vit-tower": _VIT,
    "network-ops": _NETWORK,
}

# 网络一次性任务组的段落划分（算子顺序取自 network_overview，循环父节点已剔除，
# DeepStack 整网行插在 entry 末尾）。
_ENTRY_OPERATOR_IDS = [
    "net_text_embed", "net_t_freq", "net_t_lin1", "net_t_silu", "net_t_lin2",
    "net_vision_patch", "net_vision_pos", "net_vision_rope",
    "net_vmerge_norm", "net_vmerge_view", "net_vmerge_fc1", "net_vmerge_gelu",
    "net_vmerge_fc2", "net_ds_view", "net_ds_norm", "net_ds_fc1", "net_ds_gelu",
    "net_ds_fc2", "net_cond_scatter", "net_tms_substitute",
    "net_patch_lin1", "net_patch_lin2", "net_concat", "net_rope",
    "net_attn_partition", DEEPSTACK_OP_ID,
]
_EXIT_OPERATOR_IDS = ["net_final_norm", "net_final_lin", "net_select_gen"]


def block_rule_groups() -> dict:
    return _RULES_BY_GROUP


def block_rules(group_id: str) -> dict[str, dict]:
    if group_id not in _RULES_BY_GROUP:
        raise ValueError(f"未知 HiDream-O1-Image 分组 id {group_id!r}：没有 CUBE/VEC 规则")
    return _RULES_BY_GROUP[group_id]


def network_rules() -> dict[str, dict]:
    return _NETWORK


def network_segments() -> tuple:
    return (
        {"id": "entry", "label": "整网一次性入口任务（文本/时间步/视觉塔入口/merger/拼接）",
         "operator_ids": list(_ENTRY_OPERATOR_IDS), "subtotal": True},
        {"id": "exit", "label": "整网一次性出口任务（Final RMSNorm/final_layer2/生成 token 选择）",
         "operator_ids": list(_EXIT_OPERATOR_IDS), "subtotal": True},
    )


def serial_order() -> list[dict]:
    """串行求和：入口外围 → 视觉塔（条件 Kref>0）→ 36 层 decoder → 出口外围。"""
    return [
        {"kind": "segment", "group": "network-ops", "segment": "entry"},
        {"kind": "group", "group": "vit-tower", "condition": dict(_EXEC_CONDITION)},
        {"kind": "group", "group": "uit-decoder"},
        {"kind": "segment", "group": "network-ops", "segment": "exit"},
    ]


def rule_symbols(rule: dict) -> list[str]:
    return engine.rule_symbols(rule)


def rules_report() -> list[dict]:
    return engine.rules_report(_RULES_BY_GROUP, _NETWORK)
