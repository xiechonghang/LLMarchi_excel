"""GLM-5.3 的 CUBE/VEC 粗估规则（声明式，由唯一导出器消费）。

用户 2026-10-29 确认的规则（T10 首版，T11 覆盖本模型）：

- CUBE（含矩阵计算的算子）：``耗时 ms = B·M·K·N / 吞吐``，默认 4096 MAC/ms。
- VEC（不含矩阵计算的算子）：``耗时 ms = M·N / 吞吐``，默认 64 元素/ms。
- 连续按比例、不向 64 取整；不加带宽/并行/融合/通信/精度折扣。

本模块只**声明** GLM-5.3 自己的语义（分类、真实收缩维度、来源说明），构造器与
校验来自模型无关的 ``export/perf_engine.py``；不解析模型、不写工作簿、不计算耗时。
维度表达式只用 ``scenario_evaluator`` 白名单语法，引用本组参数行或本表可填写输入
（``B``/``seqlen``/``kvcachelen``/``N_hit``）。

本模型的源码口径（依据 hflib 本地官方 transformers 检出 `glm_moe_dsa`）：

- MLA 注意力：``q_a_proj``/``q_b_proj`` 低秩 Q；``kv_a_proj_with_mqa`` 输出
  ``Rkv+Rk``；``kv_b_proj`` 把 **全部 Lkv 个缓存位置**的 512 维 latent 展开成
  ``Nh×(Dn+Dv)``；``qk`` 的 KV 轴是 Lkv（本模型无压缩条目、无滑动窗口）。
- DSA Indexer：full 层每层独立算 q/k、打分、top-k；shared 层复用上一 full 层的
  top-k，只把索引散进 mask（``g_index_mask_scatter``），不重复算 Indexer。
- MoE：sigmoid 路由 + 组选择（Ng=1 仍执行）+ TopK(K=8)；专家计算用源码可证明的
  总路由分配数 ``Ttok·K``（``Ttok = B·seqlen``）表达总量；依赖运行期命中专家数的
  只有“每命中专家扫描分配矩阵”一项，使用本表 ``N_hit``（默认未知，带相容守卫）。
  GLM-5.3 的结构里没有独立的“命中专家权重切片”算子，因此专家权重读取不另收
  （与 CUBE 的 MAC 口径一致；用户规则不含带宽项）。
- shared-moe 的共享专家是并行支路（宽 Es），与 routed 专家相加。
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
DISCLAIMER = engine.disclaimer("GLM-5.3")

_RMS_NOTE = "RMSNorm 按工作张量元素数计一次（读+归一写）；不反映真实 kernel 融合。"
_LN_NOTE = "LayerNorm（有 bias）按工作张量元素数计一次。"

# ------------------------------------------------------------------ 注意力公共
_COMMON = {
    "g_input_norm": vec("B*seqlen", "H", "带权重 RMSNorm。" + _RMS_NOTE),
    "g_q_a_proj": cube("B", "seqlen", "H", "Rq",
                       "nn.Linear(H,Rq,bias=False)：权重存 [Rq,H]，Y=X·Wᵀ，"
                       "K=H、N=Rq。"),
    "g_q_a_norm": vec("B*seqlen", "Rq",
                      "带权重 RMSNorm（eps=1e-6）。" + _RMS_NOTE),
    "g_q_b_proj": cube("B", "seqlen", "Rq", "Nh*Dq",
                       "nn.Linear(Rq,Nh·Dq)；输出 view/transpose 成 [B,Nh,seqlen,Dq] "
                       "是同一结果的视图，不另收费。"),
    "g_q_view": vec("B*seqlen*Nh", "Dq",
                    "view+transpose 到 [B,Nh,seqlen,Dq] 的元素数。" ),
    "g_q_split": vec("B*seqlen*Nh", "Dq",
                     "split 成 nope(Dn)/rope(Rk) 两段，按整张量元素数计一次。"),
    "g_q_rope": vec("B*seqlen*Nh", "Rk",
                    "交错 RoPE 只旋转末维 Rk 个分量，按该段元素数计一次；"
                    "cos/sin 由网络级 rotary_emb 生成，不在这里重复计。"),
    "g_q_cat": vec("B*seqlen*Nh", "Dq",
                   "nope+rope 拼回完整头维（Dq）的元素数。" ),
    "g_kv_a_proj": cube("B", "seqlen", "H", "Rkv+Rk",
                        "nn.Linear(H, Rkv+Rk)（MLA 压缩 latent + 单头 k_rot）；"
                        "K=H、N=Rkv+Rk。" ),
    "g_kv_a_split": vec("B*seqlen", "Rkv+Rk",
                        "把压缩输出拆成 latent(Rkv) 与 k_rot(Rk)。" ),
    "g_kv_a_norm": vec("B*seqlen", "Rkv",
                       "只归一 latent 段（Rkv），k_rot 不归一。" + _RMS_NOTE),
    "g_k_pass_view": vec("B*seqlen", "Rkv",
                         "view 成 [B,1,seqlen,Rkv] 的元素数。" ),
    "g_kv_rope": vec("B*seqlen", "Rk", "单头 k_rot 的 RoPE 元素数。" ),
    "g_cache_update": vec("B*seqlen", "Rkv+Rk",
                          "只计本次新写入 cache 的 latent+k_rot 行；调用前缓存只读视图"
                          "不另收费，物理保留长度/字节不在本口径内。" ),
    "g_kv_b_proj": cube("B", "Lkv", "Rkv", "Nh*(Dn+Dv)",
                        "kv_b_proj 把**全部 Lkv 个缓存位置**的 Rkv 维 latent 展开成 "
                        "Nh·(Dn+Dv)：K=Rkv、N=Nh·(Dn+Dv)、M=Lkv；"
                        "每次调用都重算历史位置，不能只算新 token。" ),
    "g_kv_view": vec("B*Nh*Lkv", "Dn+Dv",
                     "把 [B,1,Lkv,Nh·(Dn+Dv)] 按头重排的元素数。" ),
    "g_kv_split": vec("B*Nh*Lkv", "Dn+Dv",
                      "拆成 k_nope(Dn) 与 value(Dv)；两项之和=Dn+Dv。" ),
    "g_k_rot_expand": vec("B*Nh*Lkv", "Rk",
                          "单头 k_rot 广播成 Nh 份（expand）的物化规模。" ),
    "g_key_cat": vec("B*Nh*Lkv", "Dq",
                     "k_nope+expanded k_rot 拼成完整 key（Dq=Dn+Rk）。" ),
    "g_index_mask_scatter": vec("B*seqlen", "Lidx",
                                "把本层 top-k 索引（full 层自算、shared 层复用上一层）"
                                "散进全 True 布尔矩阵 [B,1,seqlen,Lidx]。" ),
    "g_sparse_mask_apply": vec("B*seqlen", "Lidx",
                               "用 index_mask 收窄因果 mask（masked_fill），"
                               "按 [B,1,seqlen,Lidx] 元素数计一次；top-k 只能收窄、"
                               "不能放大因果可见范围。" ),
    "g_qk": cube("B*Nh", "seqlen", "Dq", "Lkv",
                 "torch.matmul(q[B,Nh,S,Dq], k[B,Nh,Lkv,Dq]ᵀ)：头数作矩阵 batch，"
                 "M=S、K=Dq、N=Lkv（KV 轴=全部已缓存位置）。" ),
    "g_mask_add": vec("B*Nh*seqlen", "Lkv",
                      "把因果/稀疏 mask 加到打分张量 [B,Nh,S,Lkv]。" ),
    "g_softmax": vec("B*Nh*seqlen", "Lkv",
                     "对 Lkv 个位置做 softmax（float32），按打分张量元素数计一次。" ),
    "g_av": cube("B*Nh", "seqlen", "Lkv", "Dv",
                 "torch.matmul(probs[B,Nh,S,Lkv], v[B,Nh,Lkv,Dv])："
                 "M=S、K=Lkv、N=Dv；K/V 共享同一缓存位置。" ),
    "g_attn_transpose": vec("B*seqlen*Nh", "Dv",
                            "transpose(1,2)+contiguous 的元素数。" ),
    "g_attn_reshape": vec("B*seqlen", "Nh*Dv",
                          "把 Nh×Dv 拼成一维（16384）的元素数。" ),
    "g_o_proj": cube("B", "seqlen", "Nh*Dv", "H",
                     "nn.Linear(Nh·Dv, H)：K=Nh·Dv、N=H。" ),
    "g_res_attn": vec("B*seqlen", "H", "标准残差相加（两次残差都是普通加法）。" ),
    "g_post_norm": vec("B*seqlen", "H", "带权重 RMSNorm。" + _RMS_NOTE),
    "g_res_ffn": vec("B*seqlen", "H", "MLP/MoE 输出加回残差。" ),
}

_DENSE = {
    "g_dense_gate": cube("B", "seqlen", "H", "Is",
                         "nn.Linear(H,Is)：dense 层宽度 intermediate_size=12288。" ),
    "g_dense_up": cube("B", "seqlen", "H", "Is", "nn.Linear(H,Is)，与 gate 并行。" ),
    "g_dense_act": vec("B*seqlen", "Is",
                       "SiLU(gate)·up；本模型无 clamp，按一个工作张量计一次。" ),
    "g_dense_down": cube("B", "seqlen", "Is", "H",
                         "nn.Linear(Is,H)：K=Is、N=H。" ),
}

_INDEXER = {
    "g_ix_q_proj": cube("B", "seqlen", "Rq", "Ih*Id",
                        "indexer.wq_b：Linear(Rq, Ih·Id)，K=Rq、N=Ih·Id。" ),
    "g_ix_q_view": vec("B*seqlen", "Ih*Id", "view 成 [B,S,Ih,Id] 的元素数。" ),
    "g_ix_q_split": vec("B*seqlen*Ih", "Id", "split 成 rope(Rk)/nope(Id-Rk) 两段。" ),
    "g_ix_k_proj": cube("B", "seqlen", "H", "Id",
                        "indexer.wk：Linear(H,Id)，单头、与主 KV 缓存独立。" ),
    "g_ix_k_norm": vec("B*seqlen", "Id",
                       "indexer.k_norm 是 LayerNorm（有 bias），不是 RMSNorm。"
                       + _LN_NOTE),
    "g_ix_k_view": vec("B*seqlen", "Id", "unsqueeze 成伪头维的元素数。" ),
    "g_ix_k_split": vec("B*seqlen", "Id", "split 成 rope(Rk)/nope(Id-Rk)。" ),
    "g_ix_rope": vec("B*seqlen*(Ih+1)", "Rk",
                     "indexer 的 q(Ih 头) 与 k(1 伪头) 共用同一 cos/sin 做交错 RoPE；"
                     "按 (Ih+1) 份 Rk 段计。" ),
    "g_ix_q_cat": vec("B*seqlen*Ih", "Id", "q 拼回 Id。" ),
    "g_ix_k_cat": vec("B*seqlen", "Id", "k 拼回 Id 后进入 indexer 缓存。" ),
    "g_ix_cache_update": vec("B*seqlen", "Id",
                             "只计本次追加的 indexer key 行（3D 单头 [B,Lidx,Id]）。" ),
    "g_ix_scores": cube("B*Ih", "seqlen", "Id", "Lidx",
                        "torch.matmul(q[B,S,Ih,Id], keys[B,1,Lidx,Id]ᵀ)："
                        "头数作 batch，M=S、K=Id、N=Lidx（全部历史 indexer key）。" ),
    "g_ix_relu": vec("B*seqlen*Ih", "Lidx", "relu 打分张量元素数。" ),
    "g_ix_weights": cube("B", "seqlen", "H", "Ih",
                         "indexer.weights_proj：Linear(H,Ih)，K=H、N=Ih（每头一个权重）。" ),
    "g_ix_sum": cube("B*seqlen", "1", "Ih", "Lidx",
                     "每个 (batch, query) 各有一份 scores[Ih,Lidx] 与 weights[Ih]，"
                     "按头加权求和是逐 query 的一次 [1,Ih]×[Ih,Lidx] 收缩："
                     "B_eff=B·seqlen、M=1（每 query 一行）、K=Ih、N=Lidx；"
                     "不是跨 query 共用的单个 [Ih,Lidx] 矩阵。" ),
    "g_ix_mask": vec("B*seqlen", "Lidx",
                     "把因果 mask 加到 index_scores[B,S,Lidx]（只取 batch/head 0 切片）。" ),
    "g_ix_topk": vec("B*seqlen", "Lidx",
                     "对 [B,S,Lidx] 做 topk(min(Ki,Lidx))：按整张打分张量扫描一次；"
                     "Ksel=min(Ki,Lidx) 见参数区，输出索引宽度不重复计。" ),
}

_MOE = {
    "g_moe_flatten": vec("Ttok", "H", "hidden.view(Ttok,H) 的元素数。" ),
    "g_moe_router": cube("B", "seqlen", "H", "E",
                         "F.linear(x.float(), W.float())：K=H、N=E；本行只计 Linear。"
                         "随后的 sigmoid/偏置/TopK/归一已各自单列为 VEC 行计费，"
                         "不在这里重复收。" ),
    "g_moe_sigmoid": vec("Ttok", "E", "sigmoid 打分张量元素数。" ),
    "g_moe_bias_add": vec("Ttok", "E", "scores + e_score_correction_bias（偏置只参与选择）。" ),
    "g_moe_group_score": vec("Ttok", "E",
                             "view 成 [Ttok,Ng,E/Ng] 后每组取 top2 求和：按 [Ttok,E] "
                             "元素数扫描一次（Ng=1 仍执行该分支）。" ),
    "g_moe_group_topk": vec("Ttok", "Ng", "在 Ng 个组分数上 topk(Ktg)。" ),
    "g_moe_group_scatter": vec("Ttok", "Ng", "组掩码 zeros+scatter。" ),
    "g_moe_group_expand": vec("Ttok", "E", "组掩码 expand 到 [Ttok,E]。"),
    "g_moe_group_reshape": vec("Ttok", "E", "3D→2D reshape 的元素数。" ),
    "g_moe_group_masked_fill": vec("Ttok", "E", "组外专家 masked_fill(-inf)。" ),
    "g_moe_topk": vec("Ttok", "E", "每 token 对 E 个分数 topk(K=8)。" ),
    "g_moe_weight_gather": vec("Ttok", "K", "按索引 gather 未加偏置的路由权重。" ),
    "g_moe_weight_norm": vec("Ttok", "K", "top-k 权重归一（+1e-20）。" ),
    "g_moe_weight_scale": vec("Ttok", "K", "乘 routed_scaling_factor。" ),
    "g_moe_expert_mask": vec("(E+1)*K", "Ttok",
                             "one_hot(indices,E+1).permute → [E+1,K,Ttok] 的确定性"
                             "物化规模；E+1 的最后一槽在循环中跳过。" ),
    "g_moe_expert_hit": vec("(E+1)*K", "Ttok",
                            "对完整分配矩阵 sum+nonzero 求命中专家集合；"
                            "扫描规模与 E/K/Ttok 成正比，不依赖命中数。" ),
    "g_moe_expert_gather_indices": engine.expert_hit_rule(
        experts="E", top_k="K", tokens="Ttok", slice_rows="K", lower_text="K",
        note=("对每个命中专家从完整分配矩阵取 (top_k_pos, token_idx)："
              "每次扫描 [K,Ttok]；总量=N_hit·K·Ttok，只依赖运行期命中专家数。"
              "单值是各层统一命中数假设，实际路由分布不由本工具推断。")),
    "g_moe_select_tokens": vec("Ttok*K", "H",
                               "按总路由分配行数 Ttok·K gather 真实 token 行；"
                               "每命中专家的 where 扫描已单列，不重复。" ),
    "g_moe_expert_gate_up": cube("1", "Ttok*K", "H", "2*Es",
                                 "合并 gate|up 一次 Linear([2Es,H])：用源码可证明的"
                                 "总路由分配数 M=Ttok·K（每 token K 个分配）表达总量，"
                                 "不填运行期每专家 token 分布；按专家索引取切片，"
                                 "不复制 E 份。" ),
    "g_moe_expert_chunk": vec("Ttok*K", "2*Es", "沿最后一维均分成 gate/up 两个 Es。" ),
    "g_moe_expert_act": vec("Ttok*K", "Es", "SiLU(gate)·up（按总分配行数）。" ),
    "g_moe_expert_down": cube("1", "Ttok*K", "Es", "H",
                              "down_proj[e] [H,Es]：K=Es、N=H；总量按总分配数。" ),
    "g_moe_expert_weight": vec("Ttok*K", "H",
                               "按 (token_idx, top_k_pos) 取路由权重后相乘："
                               "总分配行数×H 个元素，权重只乘一次。" ),
    "g_moe_result_init": vec("Ttok", "H", "zeros_like(Ttok,H)。" ),
    "g_moe_index_add": vec("Ttok*K", "H",
                           "按 token_idx 把各分配贡献加回对应 token 行："
                           "总分配行数×H 次累加。" ),
    "g_moe_routed_view": vec("Ttok", "H", "routed.view(B,seqlen,H) 的元素数。" ),
    "g_moe_shared_gate": cube("B", "seqlen", "H", "Es",
                              "共享专家 gate_proj：Linear(H,Es)，与 routed 并行。" ),
    "g_moe_shared_up": cube("B", "seqlen", "H", "Es", "共享专家 up_proj。" ),
    "g_moe_shared_act": vec("B*seqlen", "Es", "SiLU(gate)·up（共享专家）。" ),
    "g_moe_shared_down": cube("B", "seqlen", "Es", "H", "共享专家 down_proj。" ),
    "g_moe_merge": vec("B*seqlen", "H", "routed + shared 逐元素相加。" ),
}

_RULES_BY_GROUP = {
    "full-dense": {**_COMMON, **_INDEXER, **_DENSE},
    "shared-moe": {**_COMMON, **_MOE},
    "full-moe": {**_COMMON, **_INDEXER, **_MOE},
}

# ------------------------------------------------------------------ 整网
_NETWORK = {
    "net_embed": vec("B*seqlen", "H",
                     "token→[B,seqlen,H] 嵌入查表与写出：按输出元素数计一次；"
                     "不按词表大小乘。" ),
    "net_position_ids": vec("seqlen", "1",
                            "默认 position_ids 为 arange+past 的 [1,seqlen] 个标量；"
                            "B=1 广播，不乘 batch。" ),
    "net_mask_build": vec("B*seqlen", "Lmask",
                          "模型级因果 mask 物化 [B,1,seqlen,Lmask]；只建一次、"
                          "循环内复用（层内稀疏化见层组行）。" ),
    "net_rope": vec("seqlen", "Rk",
                    "唯一一组 RoPE 频率：每位置 Rk÷2 对（cos+sin）=Rk 个；"
                    "主注意力与 indexer 共用，只在网络入口算一次。" ),
    "net_layer_loop": engine.container(
        "78 层循环容器：层内成本由三组层型小计×真实层数串行汇总，"
        "本行不重复计成本（避免父子双计）。"),
    "net_final_norm": vec("B*seqlen", "H",
                          "Final RMSNorm（本模型没有 HyperHead/mHC 收拢）。"
                          + _RMS_NOTE),
    "net_logits_slice": vec("B*Ssel", "H",
                            "只对保留位置做后续投影：切片后 [B,S_sel,H] 的元素数"
                            "（S_sel 见参数区，默认 Lk=0 即 seqlen）。" ),
    "net_lm_head": cube("B", "Ssel", "H", "Vocab",
                        "nn.Linear(H,Vocab,bias=False)：K=H、N=Vocab；"
                        "tie_word_embeddings=false，与 embed_tokens 是两份独立权重。" ),
}

_NETWORK_SEGMENTS = (
    {"id": "entry", "label": "整网入口外围",
     "operator_ids": ["net_embed", "net_position_ids", "net_mask_build", "net_rope"],
     "subtotal": True},
    {"id": "loop", "label": "78 层循环（容器，不单独计成本）",
     "operator_ids": ["net_layer_loop"], "subtotal": False},
    {"id": "exit", "label": "整网出口外围",
     "operator_ids": ["net_final_norm", "net_logits_slice", "net_lm_head"],
     "subtotal": True},
)


def block_rule_groups() -> dict:
    return _RULES_BY_GROUP


def block_rules(group_id: str) -> dict[str, dict]:
    """层组 id → {算子 id: 规则}。未知组 id 具名失败。"""
    if group_id not in _RULES_BY_GROUP:
        raise ValueError(f"未知 GLM-5.3 层组 id {group_id!r}：没有 CUBE/VEC 规则")
    return _RULES_BY_GROUP[group_id]


def network_rules() -> dict[str, dict]:
    return _NETWORK


def network_segments() -> tuple:
    return tuple(dict(segment) for segment in _NETWORK_SEGMENTS)


def serial_order() -> list[dict]:
    """整网串行求和顺序：入口外围 → 三组层型（真实层序）→ 出口外围。"""
    return [
        {"kind": "segment", "group": "net-periphery", "segment": "entry"},
        {"kind": "group", "group": "full-dense"},
        {"kind": "group", "group": "shared-moe"},
        {"kind": "group", "group": "full-moe"},
        {"kind": "segment", "group": "net-periphery", "segment": "exit"},
    ]


def network_group(network: dict) -> dict:
    """从 network_overview 构造“整网外围”块（真实算子/张量/参数，不改事实）。"""
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
                          "层内结构仍由三个层组代表层 Block 表达，不铺开 78 层。"),
        "scope_note": ("范围：一次 forward 的入口外围（embedding/位置/因果 mask/"
                       "RoPE）+ 出口外围（Final RMSNorm/logits 切片/LM Head）；"
                       "78 层循环本体不在此块计成本。"),
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
