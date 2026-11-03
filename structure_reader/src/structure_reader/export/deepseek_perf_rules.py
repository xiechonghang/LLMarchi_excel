"""DeepSeek-V4-Flash 的 CUBE/VEC 粗估规则（声明式，由唯一导出器消费）。

用户 2026-10-29 确认的首版规则（写进 Excel、可在表内改吞吐）：

- CUBE（含矩阵乘/线性投影/批量矩阵乘的算子）：``耗时 ms = B·M·K·N / 吞吐``；
  基准 ``B=1, M=1, K=64, N=64`` 耗时 1 ms，即默认吞吐 ``4096 MAC/ms``。
- VEC（不含矩阵计算的算子）：``耗时 ms = M·N / 吞吐``；基准 ``M=1, N=64``
  耗时 1 ms，即默认吞吐 ``64 元素/ms``。
- 连续按比例计算，不向 64 取整；不加带宽/并行/融合/通信/精度折扣。

本模块只**声明**模型语义适配：每个算子的类别（CUBE/VEC）、真实收缩维度
（``b/m/k/n`` 或 ``m/n``）与来源说明。不做解析、不写工作簿、不计算耗时——公式由
``export/operator_workbook.py`` 写入 Excel 并由 Excel 重算。维度表达式只使用
``scenario_evaluator`` 的白名单语法（整数、+ - * // %、min/max、比较、and/or），
引用本组参数 id/符号或本表可填写输入（``B``/``seqlen``/``kvcachelen``/``N_hit``）。

维度口径（完整依据见各规则 note 与 T10-cube-vec-result.md）：

- Linear（``Y = X·Wᵀ``，权重存 ``[N,K]``）：``B=batch, M=序列/行数, K=in, N=out``，
  batch 只出现一次（不把 batch 再乘进 M）。
- 注意力打分/加权：头数作矩阵 batch（``B=B·Nh``），``M=查询位置数``，
  ``K=head_dim``/可见 KV 长度，``N=可见 KV 长度``/``head_dim``；不把输出元素数当 MACs。
- 分组线性（``o_a_proj``）：``B=B·G``，``K=Nh·Dh/G``（每组输入维），``N=Ro``。
- 复合算子：矩阵部分计入 CUBE，所有矩阵项都列入 terms（或同形项用 repeat 表示次数）；
  伴随的 view/transpose/add/激活等非矩阵部分本粗估不另收费（note 注明）。
- MoE 专家模板：不填运行期每专家 token 分布；用源码可证明的**总路由分配数**
  ``B·seqlen·K``（每 token K 个专家分配）表达专家计算的 CUBE 总量。
  ``moe_expert_slice`` 每个命中专家读取固定大小权重，总量只依赖运行期 ``N_hit``，
  因此用本表输入 ``N_hit``（默认未知）；未给定即未知，不填假设值。
- 条件执行（HCA block_bias）：条件可判定，未物化时计 0 并写明原因；
  条件依赖的输入未给定仍是未知。

规则按层组 id 声明：四个层组共享大部分算子规则，注意力/压缩器/Indexer/整网外围
按各自源码形状覆盖。构建时由适配层校验“本组每个算子都有规则、规则引用的符号都在
本组声明”，缺失即具名失败，避免用字符串前缀命中少数项便声称全覆盖。
"""
from __future__ import annotations

import copy

from structure_reader.export import perf_engine as engine
from structure_reader.export.perf_engine import (
    attach_performance,
    container,
    cube,
    cube_terms,
    rule_symbols,
    rules_report as _rules_report,
    vec,
)

__all__ = [
    "ENGINE",
    "PERFORMANCE_STATUS",
    "PERFORMANCE_NOTE",
    "DISCLAIMER",
    "THROUGHPUT_NOTE",
    "block_rules",
    "network_rules",
    "network_group",
    "serial_order",
    "rules_report",
    "rule_symbols",
]

# 用户确认的默认吞吐（表内可改，见 sheet_inputs 的 CUBE/VEC 吞吐行）。
ENGINE = {
    "kind": "cube_vec",
    "cube_throughput_input": "cube_mac_per_ms",
    "vec_throughput_input": "vec_elem_per_ms",
    "cube_default": 4096,
    "vec_default": 64,
    "cube_unit": "MAC/ms",
    "vec_unit": "元素/ms",
    "source": ("用户 2026-10-29 确认：CUBE B1/M1/K64/N64 = 1 ms（4096 MAC/ms）；"
               "VEC M1/N64 = 1 ms（64 元素/ms）；按比例、不向 64 取整"),
}

PERFORMANCE_STATUS = "已按用户 CUBE/VEC 规则接入（2026-10-29 确认）"
PERFORMANCE_NOTE = (
    "CUBE 耗时 = B·M·K·N ÷ 本表 CUBE 吞吐（默认 4096 MAC/ms）；"
    "VEC 耗时 = M·N ÷ 本表 VEC 吞吐（默认 64 元素/ms）；连续按比例、不向 64 取整；"
    "吞吐与 B/seqlen/kvcachelen 均为可改单元格，改后由 Excel 重算；"
    "不计带宽、并行、融合、通信与精度折扣，不等同实测性能")
THROUGHPUT_NOTE = (
    "用户确认基准：CUBE B1/M1/K64/N64 = 1 ms（4096 MAC/ms）；VEC M1/N64 = 1 ms"
    "（64 元素/ms）；连续按比例、不向 64 取整。可改本单元格，公式随改随算；"
    "≤0、空白或文本时相关成本显示未知，不按 0 计")
DISCLAIMER = (
    "本工作簿由 structure_reader 唯一导出链生成：结构、参数、层组归属来自同源事实文档；"
    "CUBE/VEC 分类与维度映射是 DeepSeek 适配层声明的模型语义规则，导出器只按规则写公式。"
    "耗时=用户 2026-10-29 确认的粗估规则（CUBE B·M·K·N/吞吐、VEC M·N/吞吐），"
    "不含带宽/并行/融合/通信/精度折扣，不等同实测；未知或非法输入显示“未知（原因）”，"
    "不默认 0，也不用忽略空白的 SUM 冒充汇总。"
)


# ---------------------------------------------------------------- 规则构造
#
# 构造器/校验/清单/整网外围参数都来自模型无关的 ``export/perf_engine.py``（T11 提炼）；
# 本模块只保留 DeepSeek 自己的语义声明，不复制第二套构造逻辑。

_cube = cube
_vec = vec
_cube_terms = cube_terms
_container = container


_HC_NOTE = ("mHC 流数 Nhc 是真实张量轴（modeling_deepseek_v4.py:969-1000 的 "
            "flatten/split/collapse）；按最后一次读取/写出的工作张量计一次元素数，"
            "不把流数当 batch 重复计。")
_HC_PROJECT_NOTE = ("F.linear(flattened[B,seqlen,Nhc·H], fn[(2+Nhc)·Nhc, Nhc·H])："
                    "一次矩阵乘，K=Nhc·H、N=(2+Nhc)·Nhc；随后的 split/view 不另收费。")
_HC_COMB_NOTE = ("comb 的逻辑输入是 [B,seqlen,Nhc,Nhc]，按该具名算子的输入元素数计一次"
                 "（与 Norm/Softmax 同口径）。源码在 softmax 后做 Sinkhorn-Knopp 行/列归一"
                 "（:975-978，Sit 次迭代）属内部算法步骤，不在本粗估另收费；"
                 "这是首版元素数粗估的限制，不代表真实算法复杂度。")
_HC_MIX_NOTE = ("In@combᵀ：每个 (batch, token) 上 [Nhc,H]×[Nhc,Nhc] 的批量矩阵乘，"
                "B=B·seqlen、M=Nhc、K=Nhc、N=H；post⊙Out 与相加项不另收费。")
_RMS_NOTE = "RMSNorm 按工作张量元素数计一次（读+归一写）；不反映真实 kernel 融合。"

# 整网一次性项：层组行只表示引用，成本在整网外围收一次；层组小计不重复计。
_CHARGE_MASK = {
    "mode": "excluded",
    "charged_at": "net_mask_build",
    "reason": ("模型级 sliding causal mask 在 DeepseekV4Model.forward 构建一次"
               "（modeling_deepseek_v4.py:1344-1350），循环内把同一 mask 传给全部层"
               "（:1352-1362）；层组行只是引用，整网只在 net_mask_build 收一次，"
               "层组小计与×层数都不重复收。"),
}
_CHARGE_ROPE_MAIN = {
    "mode": "excluded",
    "charged_at": "net_rope_main",
    "reason": ("position_embeddings['main'] 在 DeepseekV4Model.forward 计算一次"
               "（modeling_deepseek_v4.py:1351-1354），作为字典传给每层"
               "（:1355-1362）；层组行只是引用，整网只在 net_rope_main 收一次。"),
}
_CHARGE_ROPE_COMPRESS = {
    "mode": "excluded",
    "charged_at": "net_rope_compress",
    "reason": ("position_embeddings['compress'] 与 main 同一次在 model.forward 计算"
               "（modeling_deepseek_v4.py:1351-1354）并传给每层；层组行只是引用，"
               "整网只在 net_rope_compress 收一次。"),
}

_BASE: dict[str, dict] = {
    "attn_hc_flatten": _vec("B*seqlen*Nhc", "H", _HC_NOTE + " 展平+fp32 归一。"),
    "attn_hc_project": _cube("B", "seqlen", "Nhc*H", "(2+Nhc)*Nhc", _HC_PROJECT_NOTE),
    "attn_hc_split": _vec("B*seqlen", "(2+Nhc)*Nhc",
                          "split 出 pre/post/comb 三段的元素数（comb 为 Nhc×Nhc）；"
                          "base/scale 切片按标量参数不计。"),
    "attn_hc_pre": _vec("B*seqlen", "Nhc",
                         "sigmoid(pre_w·scale+base)+eps：每 token Nhc 个门控系数。"),
    "attn_hc_post": _vec("B*seqlen", "Nhc",
                          "2·sigmoid(post_w·scale+base)：每 token Nhc 个门控系数。"),
    "attn_hc_comb": _vec("B*seqlen*Nhc", "Nhc", _HC_COMB_NOTE),
    "attn_hc_collapse": _vec("B*seqlen*Nhc", "H",
                              "collapsed=(pre·stream).sum(Nhc)：按 Nhc 路流的元素数计一次读。"),
    "attn_norm": _vec("B*seqlen", "H", "标准 RMSNorm（带权重）。" + _RMS_NOTE),
    "q_a_proj": _cube("B", "seqlen", "H", "Rq",
                      "nn.Linear(H,Rq,bias=False)：权重存 [Rq,H]，按 Y=X·Wᵀ，K=H、N=Rq。"),
    "q_a_norm": _vec("B*seqlen", "Rq", "带权重 RMSNorm。" + _RMS_NOTE),
    "q_b_proj": _cube("B", "seqlen", "Rq", "Nh*Dh",
                      "nn.Linear(Rq, Nh·Dh)；输出 view/transpose 成 [B,Nh,seqlen,Dh] "
                      "是同一结果的视图，不另收费。"),
    "q_b_norm": _vec("B*Nh*seqlen", "Dh",
                      "无权重 RMSNorm，作用在 [B,Nh,seqlen,Dh] 的末维 Dh；"
                      "头维并入前导元素数。"),
    "q_rope_apply": _vec("B*Nh*seqlen", "Dh",
                          "partial RoPE 只旋转末维前 Rk 个分量，按整张量元素数计一次"
                          "（用户粗规则按元素数、不按旋转分量比例折算）。"),
    "kv_proj": _cube("B", "seqlen", "H", "Dh",
                     "nn.Linear(H,Dh)（单 KV 头）+ 随后 RMSNorm；矩阵部分按一次线性计，"
                     "紧随的 RMSNorm 不另收费。"),
    "kv_head_view": _vec("B*seqlen", "Dh", "view/transpose 到 [B,1,seqlen,Dh] 的元素数。"),
    "kv_rope_apply": _vec("B*seqlen", "Dh", "K=V 共享；按 [B,seqlen,Dh] 计一次。"),
    "cache_update": _vec("B*seqlen", "Dh",
                         "只计本次新写入 cache 的 kv 行（读调用前 cache 的视图不另收费）；"
                         "cache 物理保留长度/字节数不在本粗估计费口径内。"),
    "mask_model_build": _vec(
        "B*seqlen", "Lmask",
        "模型级 4D mask 的物化规模 [B,1,seqlen,L_mask]；循环内各层复用同一 mask"
        "（:1362），整网在 net_mask_build 收一次，本行不计入层组小计。",
        charge=_CHARGE_MASK),
    "attn_inv_rope": _vec("B*seqlen*Nh", "Dh", "输出逆 RoPE，按 [B,seqlen,Nh,Dh] 元素数。"),
    "attn_group_reshape": _vec("B*seqlen*Nh", "Dh",
                                "按 o_groups 的 reshape（视图变换），按元素数计一次。"),
    "o_a_proj": _cube("B*G", "seqlen", "Nh*Dh//G", "Ro",
                      "DeepseekV4GroupedLinear：torch.bmm(x[G,B·S, Nh·Dh/G], "
                      "w[G, Nh·Dh/G, Ro])，块对角分组矩阵乘；"
                      "B=B·G、K=Nh·Dh/G、N=Ro，不把 G 再乘进 M。"),
    "o_b_proj": _cube("B", "seqlen", "G*Ro", "H",
                      "nn.Linear(G·Ro, H)：把分组输出混合到 hidden；K=G·Ro、N=H。"),
    "attn_residual_combT": _vec("B*seqlen", "Nhc*Nhc", "combᵀ 是对称性不可省的显式转置。"),
    "attn_residual_expand": _vec("B*seqlen*Nhc", "H",
                                  "post.unsqueeze(-1)·output.unsqueeze(-2) 的广播扩张规模"
                                  " [B,seqlen,Nhc,H]；不拿较小标量输入冒充。"),
    "attn_residual_mix": _cube("B*seqlen", "Nhc", "Nhc", "H", _HC_MIX_NOTE),
    "ffn_hc_flatten": _vec("B*seqlen*Nhc", "H", _HC_NOTE + " 展平+fp32 归一。"),
    "ffn_hc_project": _cube("B", "seqlen", "Nhc*H", "(2+Nhc)*Nhc", _HC_PROJECT_NOTE),
    "ffn_hc_split": _vec("B*seqlen", "(2+Nhc)*Nhc",
                          "split 出 pre/post/comb 三段的元素数；base/scale 标量不计。"),
    "ffn_hc_pre": _vec("B*seqlen", "Nhc", "sigmoid(pre_w·scale+base)+eps。"),
    "ffn_hc_post": _vec("B*seqlen", "Nhc", "2·sigmoid(post_w·scale+base)。"),
    "ffn_hc_comb": _vec("B*seqlen*Nhc", "Nhc", _HC_COMB_NOTE),
    "ffn_hc_collapse": _vec("B*seqlen*Nhc", "H", "collapsed=(pre·stream).sum(Nhc)。"),
    "ffn_norm": _vec("B*seqlen", "H", "标准 RMSNorm（带权重）。" + _RMS_NOTE),
    "moe_flatten": _vec("B*seqlen", "H", "hidden.view(B·seqlen,H) 的元素数。"),
    "moe_router": _cube("B", "seqlen", "H", "E",
                        "F.linear(flat[B·seqlen,H], router.weight[E,H])：K=H、N=E；"
                        "随后 sqrtsoftplus/TopK/gather/归一（每 token E 或 K 个标量）"
                        "不另收费。"),
    "moe_dispatch": _vec("(E+1)*K", "B*seqlen",
                          "F.one_hot(indices[B·seqlen,K], E+1).permute → [E+1,K,B·seqlen]"
                          " 的确定性物化规模；hit=nonzero(...) 的选择开销不另收费。"),
    "moe_select_tokens": _vec("B*seqlen*K", "H",
                               "按总路由分配行数 B·seqlen·K 读取真实 token 行；"
                               "每命中专家对 one_hot 掩码的 where 扫描是 N_hit 依赖的"
                               "实现开销，本粗估不计（不填 N_hit 分布）。"),
    "moe_expert_gu": _cube("1", "B*seqlen*K", "H", "2*Es",
                            "合并 gate|up 一次 Linear [2Es,H]；用源码可证明的总路由分配数"
                            " M=B·seqlen·K（每 token K 个专家分配，"
                            "num_experts_per_tok）表达总量，不填运行期每专家 token 分布。"),
    "moe_expert_clamp": _vec("B*seqlen*K", "2*Es",
                              "chunk + 双端 clamp：按总路由分配行数×2Es 计。"),
    "moe_expert_act": _vec("B*seqlen*K", "Es", "SiLU(gate)·up：按总分配行数×Es 计。"),
    "moe_expert_down": _cube("1", "B*seqlen*K", "Es", "H",
                              "down_proj[e] [H,Es]：K=Es、N=H；总量按总路由分配数。"),
    "moe_expert_weight": _vec("B*seqlen*K", "H",
                               "按 (token_idx, top_k_pos) gather 路由权重后乘："
                               "总分配行数×H 个元素（权重只乘一次，combine 不重复）。"),
    "moe_combine": _vec("B*seqlen*K", "H",
                         "零张量起步按 token 行 index_add 各分配贡献：总分配行数×H 次累加。"),
    "moe_routed_view": _vec("B*seqlen", "H", "routed.view(B,T,H) 的元素数。"),
    "shared_gate_up": _cube("B", "seqlen", "H", "Es",
                             "共享专家 gate_proj/up_proj 两个同形 Linear([Es,H])，"
                             "同形重复 2 次（repeat=2），合成一次 N=2·Es 的等价矩阵乘；"
                             "clamp 不另收费。", repeat="2"),
    "shared_act_down": _cube("B", "seqlen", "Es", "H",
                              "down_proj [H,Es]：K=Es、N=H；SiLU·mul 不另收费。"),
    "moe_merge": _vec("B*seqlen", "H", "routed+shared 逐元素相加。"),
    "ffn_residual_combT": _vec("B*seqlen", "Nhc*Nhc", "combᵀ 显式转置。"),
    "ffn_residual_expand": _vec("B*seqlen*Nhc", "H",
                                 "post⊙out 广播扩张到 [B,seqlen,Nhc,H]。"),
    "ffn_residual_mix": _cube("B*seqlen", "Nhc", "Nhc", "H", _HC_MIX_NOTE),
}

_EXPERT_SLICE_NOTE = (
    "每个命中专家读取 gate_up[e]（2·Es·H）+ down[e]（H·Es）共 3·Es·H 个元素；"
    "每命中专家成本恒定，总量=命中专家数×该常量。命中专家数 N_hit 是运行期数据相关量，"
    "由本表输入给出（默认未知）；未给定即未知、不填假设值。本表的 N_hit 是"
    "“各层统一命中数”假设，实际路由分布不由本工具推断。")


def _expert_slice_rule(lower: str, lower_text: str) -> dict:
    """专家权重切片：N_hit 必须与总路由分配数相容，否则未知。

    相容性来自源码可证明的边界：每 token 至多 K 个专家分配，
    因此命中专家数 ≤ min(E, B·seqlen·K)；非空调用时 TopK 至少 K 个不同专家命中、
    Hash 至少 1 个。违反这些边界说明输入不可实现，不能给出“完整总计”。
    """
    return _vec(
        "N_hit", "3*Es*H", _EXPERT_SLICE_NOTE,
        guards=[
            "N_hit <= min(E, B*seqlen*K)",
            f"(B*seqlen*K == 0) or (N_hit >= {lower})",
        ],
        guard_text=("N_hit 必须 ≤ min(E, B·seqlen·K)（总路由分配数），"
                    f"非空调用还需 ≥ {lower_text}；单值是各层统一的命中数假设，"
                    "实际路由不由本工具推断"),
    )


# 注意力打分/加权：普通 SWA 的可见 KV 轴是 Lkv；压缩层是 Lkv+Ncomp。
_QK_NOTE = ("torch.matmul(query[B,Nh,S,Dh], key[B,Nh,KV,Dh]ᵀ)：头数作矩阵 batch"
            "（B=B·Nh），M=S、K=Dh、N=KV；KV 轴口径见括号。")
_AV_NOTE = ("torch.matmul(probs[B,Nh,S,KV], value[B,Nh,KV,Dh])：B=B·Nh、M=S、"
            "K=KV、N=Dh；K=V 共享同一 KV 张量，不重复计。")

_ATTN_SWA = {
    "rope_main": _vec(
        "seqlen", "Rk",
        "main RoPE 的 cos/sin 生成：每位置 Rk÷2 个分量×2（cos+sin）=Rk 个；"
        "B=1 广播，不乘 batch。整网在 net_rope_main 收一次，本行不计入层组小计。",
        charge=_CHARGE_ROPE_MAIN),
    "repeat_kv": _vec("B*Nh*Lkv", "Dh",
                       "单 KV 头 expand+reshape 到 Nh 头的物化规模 [B,Nh,Lkv,Dh]。"),
    "qk_scaled": _cube("B*Nh", "seqlen", "Dh", "Lkv", _QK_NOTE + " KV 轴=Lkv（无压缩条目）。"),
    "mask_add": _vec("B*Nh*seqlen", "Lkv", "加 mask：按打分张量 [B,Nh,S,Lkv] 元素数。"),
    "sink_softmax": _vec("B*Nh*seqlen", "Lkv",
                          "cat sink→减 max→softmax→去 sink：按打分张量元素数计一次"
                          "（sink 列 Nh/S 个标量不另计）。"),
    "av_matmul": _cube("B*Nh", "seqlen", "Lkv", "Dh", _AV_NOTE + " KV 轴=Lkv。"),
}

_ATTN_COMPRESSED = {
    "rope_compress": _vec(
        "seqlen", "Rk",
        "compress RoPE 的 cos/sin 生成：每位置 Rk÷2×2=Rk 个；B=1 广播。"
        "整网在 net_rope_compress 收一次，本行不计入层组小计。",
        charge=_CHARGE_ROPE_COMPRESS),
    "repeat_kv": _vec("B*Nh*(Lkv+Ncomp)", "Dh",
                       "单 KV 头 expand+reshape 到 Nh 头；KV 轴=主 KV Lkv+压缩条目 Ncomp。"),
    "qk_scaled": _cube("B*Nh", "seqlen", "Dh", "Lkv+Ncomp",
                        _QK_NOTE + " KV 轴=Lkv+Ncomp（cache.update 之后拼接的压缩条目）。"),
    "mask_add": _vec("B*Nh*seqlen", "Lkv+Ncomp",
                      "加 mask/block_bias：按打分张量元素数（KV 轴=Lkv+Ncomp）。"),
    "sink_softmax": _vec("B*Nh*seqlen", "Lkv+Ncomp",
                          "cat sink→softmax→去 sink：按打分张量元素数计一次。"),
    "av_matmul": _cube("B*Nh", "seqlen", "Lkv+Ncomp", "Dh", _AV_NOTE + " KV 轴=Lkv+Ncomp。"),
    "kv_concat": _vec("B*(Lkv+Ncomp)", "Dh",
                       "主 KV 与压缩条目拼接到同一 KV 轴的元素数；拼接后由 repeat_kv 广播。"),
    "mask_extend": _vec("B*seqlen", "Lmask+Ncomp",
                         "模型级 mask 与 block_bias 拼接后的可见性矩阵元素数。"),
}

# CSA（compress_rate_csa=4）：Ca/Cb 重叠布局、条目宽 2·M、步长 M。
_CSA_COMPRESSOR = {
    "c_kv_gate_proj": _cube("B", "seqlen", "H", "2*Dh",
                             "kv_proj/gate_proj 两个同形 Linear([2Dh,H])（Ca|Cb 并存），"
                             "同形重复 2 次（repeat=2）；不把 2·Dh 当两次输出再乘次数。",
                             repeat="2"),
    "c_store_window": _vec("B*(seqlen+Tbuf)", "2*Dh",
                            "窗口缓冲合并：新投影 seqlen 行 + 调用前缓冲 Tbuf 行"
                            "（Tbuf=kvcachelen % M），每行 2·Dh 个元素；"
                            "不含整段历史重写。"),
    "c_window_view": _vec("B*Nwin*2*M", "Dh",
                           "Ca/Cb 重叠布局 new_kv[B,N_win,2·M,Dh] 的工作张量规模"
                           "（M 为配置 compress_rate_csa）；仅在 N_win>0 时物化，"
                           "N_win=0 时元素数为 0。"),
    "c_pool": _vec("B*Nwin*2*M", "Dh",
                    "(new_kv·softmax(gate,dim=2)).sum(dim=2) 再 RMSNorm："
                    "按 [B,N_win,2·M,Dh] 元素数计一次。"),
    "c_rope_gen": _vec("B*Nwin", "Rk", "条目位置 cos/sin：每条目 Rk÷2×2=Rk 个。"),
    "c_rope": _vec("B*Nwin", "Dh", "本调用新条目 RoPE 的元素数。"),
    "c_update_states": _vec("B*Nwin", "Dh",
                             "只计本次追加的新条目（读调用前累计条目的视图不另收费）。"),
    "i_kv_gate_proj": _cube("B", "seqlen", "H", "2*Id",
                             "Indexer kv_proj/gate_proj 两个同形 Linear([2Id,H])，"
                             "repeat=2。", repeat="2"),
    "i_store_window": _vec("B*(seqlen+Tbufi)", "2*Id",
                            "Indexer 窗口缓冲合并：seqlen+Tbuf_i 行，每行 2·Id 元素。"),
    "i_window_view": _vec("B*Nwin_i*2*M", "Id",
                           "Indexer Ca/Cb 布局 new_kv[B,N_win_i,2·M,Id] 的元素数。"),
    "i_pool_rope": _vec("B*Nwin_i*2*M", "Id",
                         "Indexer 条目 softmax 汇聚+RMSNorm+RoPE：按 "
                         "[B,N_win_i,2·M,Id] 元素数计一次。"),
    "i_update_states": _vec("B*Nwin_i", "Id", "只计本次追加的 Indexer 新条目。"),
    "i_q_proj": _cube("B", "seqlen", "Rq", "Ih*Id",
                       "indexer.q_b_proj Linear(Rq, Ih·Id)：K=Rq、N=Ih·Id；"
                       "随后的 view+compress RoPE 不另收费。"),
    "i_scorer": _cube_terms([
        {"label": "q·kᴵᶜᵒᵐᵖ（压分）", "b": "B*Ih", "m": "seqlen", "k": "Id",
         "n": "Tci+Nwin_i"},
        {"label": "weights_proj（每头权重）", "b": "B", "m": "seqlen", "k": "H", "n": "Ih"},
    ], ("matmul(q[B,S,Ih,Id], comp_kv[B,T,Id]ᵀ) 得 [B,S,Ih,T]：头数作 batch、K=Id、"
        "N=Tci+Nwin_i；另有 weights_proj Linear(H→Ih) 的第二项矩阵乘（K=H、N=Ih）。"
        "两项都列入，不只取第一项；relu/scale/加权求和不另收费。")),
    "i_topk": _vec("B*seqlen", "Tci+Nwin_i",
                    "对 scores[B,seqlen,Tci+Nwin_i] 做 mask+topk+where："
                    "按整张打分张量元素数计一次。"),
    "c_block_bias": _vec("B*seqlen", "Ncomp+1",
                          "block_bias 先按 [B,1,seqlen,Ncomp+1] 填 -inf 再 scatter 选中项"
                          "（+1 为 -1 哨兵列）；scatter 的 Kisel 个索引不另收费。"),
}

# HCA（compress_rate_hca=128）：非重叠窗口，无 Ca/Cb、无 Indexer。
_HCA_COMPRESSOR = {
    "c_kv_proj": _cube("B", "seqlen", "H", "Dh",
                        "HCA compressor.kv_proj Linear(H,Dh)：K=H、N=Dh。"),
    "c_gate_proj": _cube("B", "seqlen", "H", "Dh",
                          "HCA compressor.gate_proj Linear(H,Dh)：K=H、N=Dh。"),
    "c_store_window": _vec("B*(seqlen+Tbuf)", "Dh",
                            "窗口缓冲合并：seqlen+Tbuf 行，每行 Dh 个元素。"),
    "c_window_view": _vec("B*Nwin*M", "Dh",
                           "非重叠窗口 new_kv[B,Nwin,M,Dh] 的元素数（M=compress_rate_hca）。"),
    "c_pool": _vec("B*Nwin*M", "Dh",
                    "沿 compress_rate 轴 softmax 汇聚再 RMSNorm：按 [B,Nwin,M,Dh] 元素数。"),
    "c_rope_gen": _vec("B*Nwin", "Rk", "条目位置 cos/sin：每条目 Rk 个。"),
    "c_rope": _vec("B*Nwin", "Dh", "本调用新条目 RoPE 的元素数。"),
    "c_update_states": _vec("B*Nwin", "Dh", "只计本次追加的新条目。"),
    "hca_block_bias": _vec(
        "B*seqlen", "Ncomp",
        "HCA 因果 block_bias 的可见性矩阵 [B,1,seqlen,Ncomp]；源码仅在 "
        "seqlen!=1 且调用后条目数>0 时物化（:block_bias），未物化时本项计 0"
        "（合法条件不执行，不是缺失）；条件依赖的输入未给定仍是未知。",
        conditional={"when": "seqlen != 1 and Ncomp > 0",
                     "when_text": "seqlen==1 或调用后条目数为 0 时不物化 block_bias"}),
    "c_mask_extend": _vec("B*seqlen", "Lmask+Ncomp",
                           "模型级 mask 与 block_bias 拼接/零扩展后的可见性矩阵元素数。"),
}

_HASH = {
    "moe_hash_lookup": _vec("B*seqlen", "K",
                             "tid2eid[input_ids] 查表结果 [B,seqlen,K] 的元素数；"
                             "查表不是分数 TopK，也不根据权重推断路由。"),
}

_RULES_BY_GROUP: dict[str, dict] = {
    "swa-hash": {**_BASE, **_ATTN_SWA, **_HASH,
                 "moe_expert_slice": _expert_slice_rule("1", "每 token 命中数（Hash 为 1）")},
    "csa-hash": {**_BASE, **_ATTN_COMPRESSED, **_CSA_COMPRESSOR, **_HASH,
                 "moe_expert_slice": _expert_slice_rule("1", "每 token 命中数（Hash 为 1）")},
    "hca-topk": {**_BASE, **_ATTN_COMPRESSED, **_HCA_COMPRESSOR,
                 "moe_expert_slice": _expert_slice_rule("K", "每 token 命中数 K")},
    "csa-topk": {**_BASE, **_ATTN_COMPRESSED, **_CSA_COMPRESSOR,
                 "moe_expert_slice": _expert_slice_rule("K", "每 token 命中数 K")},
}

# 单层入口（layer-N）没有层组 id：按 Block 的 attention_type 与**真实算子构成**
# （是否有 Indexer 打分 / Hash 查表）选择同一套层组规则，不看名字猜。
_GROUP_BY_ATTENTION = {
    "sliding_attention": "swa-hash",
    "heavily_compressed_attention": "hca-topk",
}

_NETWORK: dict[str, dict] = {
    "net_embed": _vec("B*seqlen", "H",
                       "token→[B,seqlen,H] 嵌入查表与写出：按输出元素数计一次；"
                       "不按词表大小乘。"),
    "net_position_ids": _vec("seqlen", "1",
                              "默认位置序列 arange+add+unsqueeze 产出 [1,seqlen] 个标量；"
                              "B=1 广播。"),
    "net_mask_build": _vec("B*seqlen", "Lmask",
                            "模型级滑动因果 mask 物化 [B,1,seqlen,L_mask]；"
                            "只建一次、循环内复用。"),
    "net_rope_main": _vec("seqlen", "Rk",
                           "main RoPE 频率：每位置 Rk÷2 个分量×2（cos+sin）=Rk 个。"),
    "net_rope_compress": _vec("seqlen", "Rk",
                               "compress RoPE 频率：每位置 Rk 个（cos+sin）；B=1 广播。"),
    "net_hc_expand": _vec("B*seqlen*Nhc", "H",
                           "单流 unsqueeze+expand+contiguous 到 [B,seqlen,Nhc,H] 的规模。"),
    "net_layer_loop": _container(
        "43 层循环容器：层内成本由四类层组小计×层数串行汇总，"
        "本行不重复计成本（避免父子双计）。"),
    "net_hc_head_norm": _vec("B*seqlen*Nhc", "H",
                              "HyperHead 收拢前展平+无权重 RMSNorm 的元素数。"),
    "net_hc_head_project": _cube("B", "seqlen", "Nhc*H", "Nhc",
                                  "F.linear(flat[B,seqlen,Nhc·H], hc_fn[Nhc, Nhc·H])："
                                  "K=Nhc·H、N=Nhc。"),
    "net_hc_head_pre": _vec("B*seqlen", "Nhc", "sigmoid(mixes·scale+base)+eps。"),
    "net_hc_head_collapse": _vec("B*seqlen*Nhc", "H",
                                  "(pre·x).sum(Nhc)：mHC 流回到单流。"),
    "net_final_norm": _vec("B*seqlen", "H", "共享 Final RMSNorm（在 HyperHead 收拢之后）。"),
    "net_logits_slice": _vec("B*Ssel", "H",
                              "只对保留位置做后续投影：切片后 [B,S_sel,H] 的元素数"
                              "（S_sel 见参数区，默认 Lk=0 即 seqlen）。"),
    "net_lm_head": _cube("B", "Ssel", "H", "Vocab",
                          "nn.Linear(H,Vocab,bias=False)：K=H、N=Vocab；"
                          "tie_word_embeddings=false，与 embed_tokens 是两份独立权重。"),
}

_NETWORK_SEGMENTS = (
    {"id": "entry", "label": "整网入口外围",
     "operator_ids": ["net_embed", "net_position_ids", "net_mask_build",
                      "net_rope_main", "net_rope_compress", "net_hc_expand"],
     "subtotal": True},
    {"id": "loop", "label": "43 层循环（容器，不单独计成本）",
     "operator_ids": ["net_layer_loop"], "subtotal": False},
    {"id": "exit", "label": "整网出口外围",
     "operator_ids": ["net_hc_head_norm", "net_hc_head_project", "net_hc_head_pre",
                      "net_hc_head_collapse", "net_final_norm", "net_logits_slice",
                      "net_lm_head"],
     "subtotal": True},
)


# ---------------------------------------------------------------- 对外接口


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def block_rules(group_id: str) -> dict[str, dict]:
    """层组 id → {算子 id: 规则}。未知组 id 具名失败，不静默给空规则。"""
    if group_id not in _RULES_BY_GROUP:
        raise ValueError(f"未知 DeepSeek 层组 id {group_id!r}：没有 CUBE/VEC 规则")
    return _RULES_BY_GROUP[group_id]


def group_id_for_block(block: dict) -> str:
    """单层入口：按 Block 的真实算子构成选择同一套层组规则。

    单层文档的 ``moe_type`` 只写 ``moe``（不细分 hash/topk），因此用源码可核的算子
    存在性判定：有 ``i_scorer`` 即带 Indexer 的 CSA，有 ``moe_hash_lookup`` 即 Hash 路由。
    """
    attention = block.get("attention_type", "")
    if attention in _GROUP_BY_ATTENTION:
        return _GROUP_BY_ATTENTION[attention]
    if attention == "compressed_sparse_attention":
        operators = {op["id"] for op in block.get("operators", [])}
        return "csa-hash" if "moe_hash_lookup" in operators else "csa-topk"
    raise ValueError(
        f"Block attention_type={attention!r} 没有对应的 CUBE/VEC 规则："
        f"已声明 {sorted({*_GROUP_BY_ATTENTION, 'compressed_sparse_attention'})}")


def network_rules() -> dict[str, dict]:
    return _NETWORK


def network_segments() -> tuple:
    return tuple(dict(segment) for segment in _NETWORK_SEGMENTS)


def serial_order() -> list[dict]:
    """整网串行求和顺序：入口外围 → 四层组（真实层序）→ 出口外围。"""
    return [
        {"kind": "segment", "group": "net-periphery", "segment": "entry"},
        {"kind": "group", "group": "swa-hash"},
        {"kind": "group", "group": "csa-hash"},
        {"kind": "group", "group": "hca-topk"},
        {"kind": "group", "group": "csa-topk"},
        {"kind": "segment", "group": "net-periphery", "segment": "exit"},
    ]


def network_group(network: dict) -> dict:
    """从 network_overview 构造“整网外围”块（真实算子/张量/参数，不改事实）。"""
    path = network["path"]
    rules = network_rules()
    raw_operators = list(path.get("operators", []))
    missing = [op["id"] for op in raw_operators if op["id"] not in rules]
    if missing:
        raise ValueError(f"整网外围算子缺少 CUBE/VEC 规则: {missing}")
    # 不修改同源结构输入：每个算子复制后挂规则，network_overview 对象保持原样
    # （同进程其它调用、结构 JSON 与后续导出都不被污染）。
    operators = []
    for op in raw_operators:
        copied = {**op, "performance": copy.deepcopy(rules[op["id"]])}
        operators.append(copied)

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

    segments = []
    for segment in _NETWORK_SEGMENTS:
        segments.append({
            "id": segment["id"], "label": segment["label"],
            "operator_ids": list(segment["operator_ids"]),
            "subtotal": segment["subtotal"],
        })
    return {
        "id": "net-periphery",
        "label": "整网外围（token→logits 的入口/出口外围算子）",
        "attention_type": "（整网外围）",
        "moe_type": "（不适用）",
        "layer_indices": [],
        "representative_layer_index": None,
        "loop": None,
        "template_note": ("整网外围来自同一份 network_overview.json；"
                          "层内结构仍由四个层组代表层 Block 表达，不铺开 43 层。"),
        "scope_note": ("范围：一次 forward 的入口外围（embedding/位置/mask/RoPE/"
                       "mHC 扩展）+ 出口外围（HyperHead 收拢/FinalNorm/logits 切片/"
                       "LM Head）；层循环本体不在此块计成本。"),
        "parameters": parameters,
        "tensors": list(path.get("tensors", [])),
        "operators": operators,
        "cache_notes": list(path.get("cache_notes", [])),
        "coverage_notes": list(path.get("coverage_notes", [])),
        "segments": segments,
        "kind": "periphery",
    }


def rules_report() -> list[dict]:
    """规则清单（可审查/可写 JSON 证据），由公共基座展开。"""
    return _rules_report(_RULES_BY_GROUP, _NETWORK)
