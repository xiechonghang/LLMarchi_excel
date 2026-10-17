"""DeepSeek-V4-Flash HF decoder 阅读数据提取（里程碑一样板：layer_idx=4，CSA+TopK MoE）。

从 rules/deepseek_v4_hf.py 解释的有效配置与本地官方 Transformers 检出源码，
构建 hf_reading_contract_v1 冻结接口的 reading_document（dict）。HTML 与
Excel 消费同一份文档；本模块不生成渲染文件、不执行模型、不加载权重。

样板范围（真实 layer_idx=4）：
- 结构组合 compressed_sparse_attention + moe（TopK 路由），两处 mHC。
- prefill：T 个新 token、空缓存起步的完整 Decoder 路径。
- decode：T=1 个新 token（符号 T 保留）、P 个历史 token 缓存前提的完整 Decoder 路径。

表达事实（modeling_deepseek_v4.py 行号按本地检出 536ecc00…，下称 modeling）：
- DecoderLayer（:1161-1185）：attn_hc→input_layernorm→Attention→post 扩展 +
  comb.T 残差混合→ffn_hc→post_attention_layernorm→MoE→post 扩展 + comb.T
  残差混合。comb 转置方向由 :1177-1179/:1183-1185
  matmul(comb.transpose(-1,-2), hidden_states) 确定（双随机但非对称）。
- HyperConnection（:940-984）：flatten+input_norm→F.linear(fn)→split(pre/post/comb)
  →pre=sigmoid(w·scale+b)+eps、post=2·sigmoid(w·scale+b)、comb=softmax(w·scale+b)+eps
  →Sinkhorn 列归一 + (Sit-1)×[行归一→列归一]（:976-979）→collapsed=Σ_n pre_n·stream_n。
- Attention（:798-870）：q_a_proj→q_a_norm→q_b_proj→view(B,T,Nh,Dh).transpose(1,2)
  →q_b_norm(Unweighted)→部分RoPE（尾部 Rq=64 通道，interleaved 配对，
  rotate_half 用 0::2/1::2 交错；:332-356）；kv=kv_norm(kv_proj(hidden))
  →view/transpose→部分RoPE→cache.update(kv,kv) 返回 full（K=V 同存储）
  →压缩器→kv=cat([kv,compressed_kv],dim=2)（:829）→mask 扩展（:837-841）
  →eager_attention_forward（:714-742，repeat_kv→QK^T·scale→+mask→cat sinks→
  减max→softmax→去sink→dropout(p=0)→AV→transpose(1,2)）→逆RoPE（conjugate
  -sin，:859-865）→reshape(B,T,G,-1)→o_a(GroupedLinear bmm)→flatten→o_b_proj。
- CSA 压缩器（:620-699）：kv_proj/gate_proj(→2·Dh)→cache.store_compression_weights
  （buffer 合并→取 M 对齐前缀→余量回写 buffer，:203-224）→view(B,n_win,M,2Dh)
  →+position_bias→Ca/Cb 重叠布局（new_kv 前半=前窗 Ca、后半=本窗 Cb；
  首窗前半由 cache.overlap 提供，首调用为零值/-inf gate，:647-666）→
  softmax(fp32) 加权汇聚→kv_norm→压缩条目 RoPE（位置 i·M+first_window_position）
  →update_compressor_states 追加→compressed_kv→block_bias（Indexer top-k
  集合+因果阈值，scatter 0/-inf，:690-699）。
- Lightning Indexer（:504-583）：kv_proj/gate_proj(→2·Id)→同 Ca/Cb 重叠布局→
  softmax 汇聚→kv_norm→RoPE→q_b_proj(q_residual)→q RoPE→scorer
  （:452-456，ReLU(Q·K^T)·Id^-0.5·weights_proj 求和→[B,S,T]）→future_mask
  （entry>=causal_threshold 处 -inf）→topk(min(Ki,len))→无效位 -1 哨兵。
- HCA 缓存层与 CSA 缓存层（modeling:156-297；DynamicSlidingWindowLayer 基类
  cache_utils:211-294）：主 KV 持久保存 W-1=127 尾部、本次可读 full；
  compressor/indexer 两套 buffer_kv/buffer_gate/compressed_kv/entry_count；
  CSA 额外 overlap_kv/overlap_gate（跨调用最后满窗 Ca 尾）。
- MoE（:1117-1134）：TopKRouter（:1076-1083，flat→linear(weight)→sqrtsoftplus
  →topk(scores+bias)→gather→归一(+1e-20)→×Scl）→Experts（:1038-1053，
  one_hot mask→命中专家循环→F.linear(gate_up[e])→chunk→clamp/SiLU·mul→
  down→×top_k_weights→index_add）→shared_experts（DeepseekV4MLP :1006-1021，
  clamp/SiLU/mul/down）→routed+shared。
- 模型级 mask（masking_utils:1078-1162 + Cache.get_mask_sizes cache_utils:276-286）：
  create_sliding_window_causal_mask 用首个 sliding 层尺寸；eager_mask 生成 4D
  float（0 可见 / -inf 屏蔽），q_offset=累积历史、kv_offset=max(cumulative-W+1,0)。
  Attention 内在压缩条目拼接后再扩展（cat block_bias 或 0 填充，:837-841）。

缓存阶段区别：
- prefill：缓存起步为空（keys 空 tensor、entry_count=0、buffer=None、overlap=None）；
  update 惰性初始化后 full=T 个新 KV。
- decode：缓存已含历史（keys 为 W-1 尾部、entry_count=N_h、buffer 为上次余量、
  overlap 为上次满窗 Ca 尾）；full=cat(keys_tail, 新kv)。

本样板所有算子均按源码显式表达，无 internal_unanalyzed 黑盒（黑盒机制保留在
接口上，当前文档不使用；量化执行/设备 kernel 明确列于 coverage_notes 未覆盖）。
"""

from __future__ import annotations

from pathlib import Path

from structure_reader.rules.deepseek_v4_hf import (
    SOURCE_FILES,
    HfModelValues,
    load_hf_model_values,
)

__all__ = [
    "ReadingDocError",
    "SAMPLE_LAYER_INDEX",
    "build_reading_document",
]

SAMPLE_LAYER_INDEX = 4
SCHEMA = "hf_decoder_reading.v1"


class ReadingDocError(ValueError):
    """阅读数据构建失败（配置/范围/事实来源不一致），具名拒绝。"""


def build_reading_document(
    workspace_root: str | Path,
    layer_index: int = SAMPLE_LAYER_INDEX,
    model_key: str = "deepseek_v4_flash_0731_hf",
) -> dict:
    """构建冻结接口 reading_document（prefill/decode 两阶段，算子按真实阅读顺序）。

    layer_index 必须是 CSA+TopK 组合层（本样板目标）；其他组合属里程碑二，
    本提取器明确拒绝而非画错结构。
    """
    values = load_hf_model_values(workspace_root, model_key=model_key)
    if not 0 <= layer_index < values.num_hidden_layers:
        raise ReadingDocError(
            f"layer_index {layer_index} 越界（0..{values.num_hidden_layers - 1}）"
        )
    attention_type = values.layer_type(layer_index)
    moe_type = values.mlp_layer_type(layer_index)
    if attention_type != "compressed_sparse_attention" or moe_type != "moe":
        raise ReadingDocError(
            f"layer_index={layer_index} 是 {attention_type}+{moe_type}；"
            "本提取器样板范围限定 CSA+TopK（其余变体属里程碑二，不拼造组合）"
        )

    prefill = _PhaseBuilder(values, layer_index, "prefill").build()
    decode = _PhaseBuilder(values, layer_index, "decode").build()
    return {
        "schema": SCHEMA,
        "model": {
            "model_key": values.model_key,
            "display_name": values.display_name,
            "implementation": values.implementation,
            "source_revision": values.source_revision,
            "config_path": values.config_path,
        },
        "scope": {
            "layer_index": layer_index,
            "attention_type": attention_type,
            "moe_type": moe_type,
            "analysis_path": "eager",
            "notes": [
                "样板范围：单个 decoder 层（layer_idx=4，compressed_sparse_attention + TopK MoE，两处 mHC）；"
                "整网 Embedding/43 层展开/HyperHead/Final Norm/LM head 属里程碑三，明确未覆盖。",
                "实现路径：transformers eager（modeling_deepseek_v4.py:1195-1213 声明仅支持 eager："
                "_supports_flash_attn/_supports_sdpa/_supports_flex_attn 均为 False）；"
                "根配置声明 FP8 量化（quantization_config），本样板只表达 HF 逻辑 eager 路径，"
                "不分析量化执行与设备 kernel。",
                "静态结构一致为目标：不加载权重、不运行模型、不做数值等价；B/T/P 等调用维度保留符号。",
                "hash_moe 层（0/1/2）的 HashRouter input_ids 查表路由属其他变体，不在本样板；"
                "SWA 层（0/1）无压缩器；本层缓存结构为 DeepseekV4CSACache。",
                "同一权重/激活张量可被多个算子引用（共享引用非重复定义）；阶段间 id 按阶段作用域解释。",
            ],
        },
        "phases": [prefill, decode],
    }


# ---------------------------------------------------------------------------
# 阶段构建
# ---------------------------------------------------------------------------


class _PhaseBuilder:
    """单阶段阅读数据累积器：参数→边界/缓存张量→算子按真实阅读顺序。

    prefill 与 decode 同构（同一模板方法）；差异点：
    - P 参数值（0 vs 符号）、T 参数值（null vs 1）。
    - 缓存张量 note 与读写算子 note（空起步 vs 历史前提）。
    - N_win/N_comp/N_kv/L_mask 的派生式相同、示例前提取决于阶段。
    """

    def __init__(self, values: HfModelValues, layer_idx: int, phase: str):
        self.values = values
        self.layer_idx = layer_idx
        self.phase = phase
        self.is_prefill = phase == "prefill"
        self.parameters: list[dict] = []
        self.tensors: list[dict] = []
        self.operators: list[dict] = []
        self.cache_notes: list[str] = []
        self.coverage_notes: list[str] = []

    # ------------------------------------------------------------------
    # 记录辅助
    # ------------------------------------------------------------------

    def param(self, pid, symbol, meaning, value, expression, note, source_refs) -> str:
        self.parameters.append({
            "id": pid, "symbol": symbol, "meaning": meaning, "value": value,
            "expression": expression, "note": note, "source_refs": source_refs,
        })
        return pid

    def tensor(self, tid, name, shape, parameter_refs, kind, note) -> str:
        self.tensors.append({
            "id": tid, "name": name, "shape": shape, "parameter_refs": parameter_refs,
            "kind": kind, "note": note,
        })
        return tid

    def op(self, oid, module_path, name, op, inputs, outputs, weights,
           parameter_refs, source_refs, note, analysis_status="analyzed") -> str:
        self.operators.append({
            "id": oid, "module_path": module_path, "name": name, "op": op,
            "inputs": inputs, "outputs": outputs, "weights": weights,
            "parameter_refs": parameter_refs, "source_refs": source_refs,
            "analysis_status": analysis_status, "note": note,
        })
        return oid

    def src(self, locator, note) -> dict:
        return {
            "path": SOURCE_FILES["modeling_deepseek_v4.py"],
            "locator": locator, "kind": "source", "note": note,
        }

    def srcf(self, file_key, locator, note) -> dict:
        return {"path": SOURCE_FILES[file_key], "locator": locator, "kind": "source", "note": note}

    def cfg(self, locator, note) -> dict:
        return {
            "path": SOURCE_FILES["config.json"], "locator": locator,
            "kind": "config", "note": note,
        }

    # ------------------------------------------------------------------
    # 模板方法：按真实阅读顺序构建
    # ------------------------------------------------------------------

    def build(self) -> dict:
        v = self.values
        title = (
            f"Prefill Decoder（layer_idx={self.layer_idx}，{v.num_hidden_layers} 层主干代表样板："
            f"CSA + TopK MoE + 双 mHC，缓存空起步）"
            if self.is_prefill
            else f"Decode Decoder（layer_idx={self.layer_idx}，CSA + TopK MoE + 双 mHC；"
                 f"历史缓存前提，T=1）"
        )
        self._build_parameters()
        self._build_boundary_and_cache()
        self._decoder_layer_flow()
        self._build_cache_notes()
        self._build_coverage_notes()
        return {
            "id": self.phase,
            "title": title,
            "parameters": self.parameters,
            "tensors": self.tensors,
            "operators": self.operators,
            "cache_notes": self.cache_notes,
            "coverage_notes": self.coverage_notes,
        }

    # ------------------------------------------------------------------
    # 参数区
    # ------------------------------------------------------------------

    def _build_parameters(self) -> None:
        v = self.values
        cfg, src = self.cfg, self.src

        self.param("B", "B", "batch 批大小", None, None,
                   "调用输入维度，静态分析保留符号", [])
        if self.is_prefill:
            self.param("T", "T", "prefill 新 token 数（本次输入序列长度）", None, None,
                       "调用输入维度；压缩窗口数、mask 形状均由 T 派生", [])
            self.param("P", "P", "历史缓存 token 数", 0, "P=0（prefill 空缓存起步）",
                       "首次调用无历史；结构前提，非运行时探值", [])
        else:
            self.param("T", "T", "decode 本次 token 数", 1, "T=1（decode 单 token）",
                       "shape 符号仍写 T 保持同构表达；值/表达式记录 T=1", [])
            self.param("P", "P", "decode 起步前历史 token 数", None,
                       "P=T_prev+T_prev2+…（历史调用累积）",
                       "缓存内容有效性来自调用历史，不由本图证明", [])
        self.param("H", "H", "hidden_size 隐层维度", v.hidden_size, None,
                   "config.hidden_size", [cfg("hidden_size", "根配置字段")])
        self.param("Nh", "Nh", "num_attention_heads 注意力头数", v.num_attention_heads, None,
                   "config.num_attention_heads", [cfg("num_attention_heads", "根配置字段")])
        self.param("Nkv", "Nkv", "num_key_value_heads KV 头数（共享 KV MQA）",
                   v.num_key_value_heads, None,
                   "=1：kv_proj 投影到单 KV 头且 K/V 读同一张量；广播到全部 Nh 头",
                   [cfg("num_key_value_heads", "根配置字段"),
                    src("776", "self.num_key_value_groups = config.num_attention_heads")])
        self.param("Dh", "Dh", "head_dim 每头维度", v.head_dim, None,
                   "config.head_dim", [cfg("head_dim", "根配置字段")])
        self.param("Rq", "Rq", "q_lora_rank 低秩 Q 秩", v.q_lora_rank, None,
                   "config.q_lora_rank", [cfg("q_lora_rank", "根配置字段")])
        self.param("Rk", "Rk", "qk_rope_head_dim 旋转维（部分 RoPE）", v.qk_rope_head_dim,
                   "Rk = int(Dh × 64/512)",
                   "legacy qk_rope_head_dim=64 / head_dim=512 派生；RoPE 只作用每头尾部 64 通道",
                   [cfg("qk_rope_head_dim", "legacy 字段"),
                    {"path": SOURCE_FILES["configuration_deepseek_v4.py"], "locator": "270-278",
                     "kind": "derived", "note": "__post_init__ partial_rotary_factor 派生"}])
        self.param("G", "G", "o_groups 分组输出投影组数", v.o_groups, None,
                   "config.o_groups", [cfg("o_groups", "根配置字段")])
        self.param("Ro", "Ro", "o_lora_rank 每组输出中间维", v.o_lora_rank, None,
                   "config.o_lora_rank", [cfg("o_lora_rank", "根配置字段")])
        self.param("M", "M", "compress_rate_csa 压缩窗口宽", v.compress_rate_csa, None,
                   "compress_rates['compressed_sparse_attention']=4（default_compress_rates，无 legacy 覆盖）",
                   [{"path": SOURCE_FILES["configuration_deepseek_v4.py"], "locator": "145/244-250",
                     "kind": "derived", "note": "default_compress_rates 默认字典"}])
        self.param("Ki", "Ki", "index_topk 每查询保留压缩条目数", v.index_topk, None,
                   "config.index_topk；实际 top_k=min(Ki, compressed_len)",
                   [cfg("index_topk", "根配置字段"), src("566", "top_k = min(self.index_topk, compressed_len)")])
        self.param("Ih", "Ih", "index_n_heads Indexer 查询头数", v.index_n_heads, None,
                   "config.index_n_heads", [cfg("index_n_heads", "根配置字段")])
        self.param("Id", "Id", "index_head_dim Indexer 头维", v.index_head_dim, None,
                   "config.index_head_dim", [cfg("index_head_dim", "根配置字段")])
        self.param("W", "W", "sliding_window 滑动窗口", v.sliding_window, None,
                   "config.sliding_window；主 KV 滑动分支与模型级 mask 窗口",
                   [cfg("sliding_window", "根配置字段"),
                    self.srcf("cache_utils.py", "211-294",
                              "DynamicSlidingWindowLayer：持久 W-1=127 尾部，本次可读 full")])
        self.param("Nhc", "Nhc", "hc_mult mHC 流数", v.hc_mult, None,
                   "config.hc_mult；mHC 权重投影输出 (2+Nhc)×Nhc",
                   [cfg("hc_mult", "根配置字段"),
                    src("926-938", "fn 权重 (2+Nhc)*Nhc × Nhc*H；base/scale 参数")])
        self.param("Sit", "Sit", "hc_sinkhorn_iters Sinkhorn 迭代次数", v.hc_sinkhorn_iters, None,
                   "config.hc_sinkhorn_iters；先列归一 1 次，再循环 Sit-1 次 [行归一→列归一]",
                   [cfg("hc_sinkhorn_iters", "根配置字段"), src("976-979", "comb 归一循环")])
        self.param("Eps", "Eps", "hc_eps Sinkhorn/sigmoid 权重数值下限", v.hc_eps, None,
                   "config.hc_eps", [cfg("hc_eps", "根配置字段")])
        self.param("E", "E", "n_routed_experts 路由专家总数（参数规模，非执行数）",
                   v.n_routed_experts, None,
                   "config.n_routed_experts（attribute_map num_local_experts→n_routed_experts）；"
                   "实际命中专家集合取决于运行时路由，静态保留符号",
                   [cfg("n_routed_experts", "根配置字段"),
                    {"path": SOURCE_FILES["configuration_deepseek_v4.py"], "locator": "104-107",
                     "kind": "derived", "note": "attribute_map"},
                    src("1030", "self.num_experts = config.num_local_experts")])
        self.param("K", "K", "num_experts_per_tok 每 token TopK 专家数", v.num_experts_per_tok, None,
                   "config.num_experts_per_tok（与 E 不同参数：E 是专家参数规模，K 是每 token 选择数）",
                   [cfg("num_experts_per_tok", "根配置字段")])
        self.param("Es", "Es", "moe_intermediate_size 专家中间维", v.moe_intermediate_size, None,
                   "config.moe_intermediate_size（attribute_map intermediate_size→moe_intermediate_size）",
                   [cfg("moe_intermediate_size", "根配置字段"),
                    {"path": SOURCE_FILES["configuration_deepseek_v4.py"], "locator": "104-107",
                     "kind": "derived", "note": "attribute_map"}])
        self.param("Lim", "Lim", "swiglu_limit 专家 gate/up clamp 上限", v.swiglu_limit, None,
                   "config.swiglu_limit", [cfg("swiglu_limit", "根配置字段")])
        self.param("Scl", "Scl", "routed_scaling_factor 路由权重缩放", v.routed_scaling_factor, None,
                   "config.routed_scaling_factor", [cfg("routed_scaling_factor", "根配置字段")])
        self.param("EpsN", "EpsN", "rms_norm_eps RMSNorm epsilon", v.rms_norm_eps, None,
                   "config.rms_norm_eps", [cfg("rms_norm_eps", "根配置字段")])
        self.param("ThetaC", "θc", "compress RoPE base（YaRN）", v.rope_compress_theta, None,
                   "rope_parameters.compress.rope_theta=compress_rope_theta；"
                   "本层 CSA 压缩器/Indexer/Attention 共用同一 rope_layer_type='compress'",
                   [cfg("compress_rope_theta", "根配置字段"),
                    {"path": SOURCE_FILES["configuration_deepseek_v4.py"], "locator": "288-307",
                     "kind": "derived", "note": "__post_init__ compress 构造（attention_factor 强制 1.0）"}])
        self.param("Yf", "Yf", "YaRN factor", v.rope_compress_factor, None,
                   "rope_scaling.factor；inv_freq 插值/外推混合比例",
                   [cfg("rope_scaling.factor", "根配置字段"),
                    self.srcf("modeling_rope_utils.py", "345-483", "_compute_yarn_parameters")])
        self.param("Nwin", "N_win", "本调用压缩器系列新闭窗数（数据相关）", None,
                   "N_win = floor((buffer_len['compressor'] + T) / M)（buffer_len 为调用前窗口缓冲余量）",
                   "store_compression_weights 取 M 对齐前缀长度//M；buffer_len 取决于调用历史，"
                   "静态保留派生式；prefill 空缓存起步 buffer_len=0",
                   [src("216-224", "usable=(kv.shape[1]//M)*M")])
        self.param("Nwin_i", "N_win_i", "本调用 Indexer 系列新闭窗数（数据相关）", None,
                   "N_win_i = floor((buffer_len['indexer'] + T) / M)",
                   "Indexer 与压缩器共用同窗宽 M，但 buffer/entry_count/overlap 按 'indexer' 键独立维护，"
                   "历史余量可不同，故闭窗数独立成参",
                   [src("519-523", "store_compression_weights('indexer', kv, gate)"),
                    src("268-276", "buffer_kv/buffer_gate['indexer'] 独立键位")])
        self.param("Ncomp", "N_comp", "压缩条目总数（调用后持久累计，数据相关）", None,
                   "N_comp = entry_count('compressor')（调用后 = 调用前 + N_win）",
                   "update_compressor_states 追加本调用新条目并累加计数",
                   [src("226-237", "update_compressor_states")])
        self.param("Lkv", "L_kv", "Attention 主 KV 本次可读长度（数据相关）", None,
                   "L_kv = min(P, W-1) + T（full=cat(keys持久尾部, kv_new)）",
                   "keys 持久保存最近 min(累计, W-1) 个位置（短历史不足 W-1 时全保留）；"
                   "full 是本次参与 Attention 的完整视图，与调用后持久长度 min(P+T, W-1) 分开表达",
                   [src("189-201", "HCACache.update 返回 full；持久 keys=full[:, :, -(W-1):, :]",
                      )])
        self.param("Lmask", "L_mask", "模型级 mask 的 KV 长度（数据相关）", None,
                   "L_mask = min(P, W-1) + T（is_full=(P≥W) 时 W-1+T，否则 P+T；"
                   "按首个 sliding 层（层0）的 cumulative_length=P 计算）",
                   "create_sliding_window_causal_mask 按首个 sliding 层尺寸构造 4D float mask；"
                   "本层 Attention 内拼接压缩条目后再扩展",
                   [self.srcf("masking_utils.py", "1078-1162", "create_sliding_window_causal_mask"),
                    self.srcf("cache_utils.py", "276-286",
                              "get_mask_sizes：is_full→W-1+query，否则 cumulative+query")])
        # ---- 缓存长度符号（运行时数据相关，全图 shape 引用，含义/公式在此统一定义）----
        self.param("Tbuf", "Tbuf", "压缩器窗口缓冲余量长度（调用前）", None,
                   "Tbuf = len(buffer_kv['compressor'])（调用前，<M 或 0）",
                   "store_compression_weights 合并后取 M 对齐前缀，余量即 Tbuf；"
                   "prefill 起步 None（0 长度），decode 起步为上次余量",
                   [src("216-224", "usable=(kv.shape[1]//M)*M；buffer=kv[:, usable:]")])
        self.param("Tbufp", "Tbuf'", "压缩器窗口缓冲余量长度（调用后）", None,
                   "Tbuf' = (Tbuf + T) % M",
                   "本调用写回 buffer 的余量（c_store_window 输出 cache_buffer_c_kv_post）",
                   [src("216-224", "usable=(kv.shape[1]//M)*M；buffer=kv[:, usable:]")])
        self.param("Tbufi", "Tbuf_i", "Indexer 窗口缓冲余量长度（调用前）", None,
                   "Tbuf_i = len(buffer_kv['indexer'])（调用前）",
                   "'indexer' 键独立缓冲，与 Tbuf 可不同",
                   [src("519-523", "store_compression_weights('indexer', kv, gate)")])
        self.param("Tbufip", "Tbuf_i'", "Indexer 窗口缓冲余量长度（调用后）", None,
                   "Tbuf_i' = (Tbuf_i + T) % M",
                   "i_store_window 写回余量（cache_buffer_i_kv_post）",
                   [src("519-523", "store_compression_weights('indexer', kv, gate)")])
        self.param("Tcc", "Tcc", "压缩器已发射压缩条目数（调用前）", None,
                   "Tcc = entry_count['compressor']（调用前）",
                   "prefill 起步 None（0 条）；调用后 N_comp=Tcc+N_win",
                   [src("226-237", "update_compressor_states 追加并累加")])
        self.param("Tci", "Tci", "Indexer 已发射压缩条目数（调用前）", None,
                   "Tci = entry_count['indexer']（调用前）",
                   "Indexer 系列（'indexer' 键）；调用后条目数=Tci+N_win_i；"
                   "i_scores/topk 轴使用追加后条目（:556-566），保持 Tci 只表调用前数量",
                   [src("556-558", "compressed_kv = update_compressor_states('indexer', compressed)")])

    # ------------------------------------------------------------------
    # 边界与缓存张量
    # ------------------------------------------------------------------

    def _build_boundary_and_cache(self) -> None:
        prefill = self.is_prefill
        self.tensor(
            "input_ids", "input_ids", ["B", "T"], ["B", "T"], "boundary",
            "输入 token id（本层 MoE 为 TopK 路由不消费它；整网 embedding 使用；"
            "HashRouter 层才直接查表）",
        )
        self.tensor(
            "position_ids", "position_ids", ["1", "T"], ["T"], "boundary",
            "本调用绝对位置（modeling:1335-1337）：arange(T)+P 后 unsqueeze(0)，默认 [1,T]"
            "（prefill P=0、decode P）；供 RoPE cos/sin、模型级 mask 与 Indexer 因果阈值消费；"
            "本样板按该默认路径表达（cos/sin 亦为 [1,T,Rk÷2]）；若调用方传入自定义批次位置"
            "（[B,T]），cos/sin 与阈值随输入 batch 维变化——本图不含该可选输入组合",
        )
        self.tensor(
            "hidden_streams_in", "hidden_streams（层输入）",
            ["B", "T", "Nhc", "H"], ["Nhc", "H"], "boundary",
            "上一 decoder 层输出的 Nhc 路流（层0 来自 embedding 后 expand，modeling:1351）；"
            "单层样板的诚实外部输入边界，无本图内 producer",
        )
        self.tensor(
            "hidden_streams_out", "hidden_streams（层输出）",
            ["B", "T", "Nhc", "H"], ["Nhc", "H"], "boundary",
            "本层最终输出（ffn 段 mHC 混合后），进入下一层",
        )
        start_note = (
            "prefill 起步为空（惰性初始化，keys 为 0 长度 tensor）；shape 的持久轴"
            " min(P,W-1) 在 prefill 取 0"
            if prefill else "decode 起步已含历史：最近 min(P, W-1) 个位置的 KV 尾部"
            "（历史不足 W-1 时全保留）"
        )
        self.tensor(
            "cache_keys", "CSACache.keys/values（主 KV 滑动持久态，调用前）",
            ["B", "1", "min(P,W-1)", "Dh"], ["Lkv", "Dh", "W"], "cache",
            f"共享 KV（K=V 同存储，单 KV 头）调用前状态；{start_note}",
        )
        self.tensor(
            "cache_keys_post", "CSACache.keys/values（主 KV 滑动持久态，调用后）",
            ["B", "1", "min(P+T,W-1)", "Dh"], ["Lkv", "Dh", "W"], "cache",
            "update 后持久保存 full 的最后 W-1 个位置（累计不足 W-1 时全量，min(P+T,W-1)）；"
            "与本次可读 full（L_kv）长度分开表达；进入下一调用作为其调用前状态",
        )
        self.tensor(
            "cache_cumlen", "cumulative_length（主 KV 累计长度）",
            [], ["W"], "cache",
            "已累计 token 数（含本调用新 token 前）：决定 get_mask_sizes 的 kv_offset/kv_length 与"
            "持久保存量；prefill 起步 0，decode 起步 P",
        )
        self.tensor(
            "cache_entry_count_c", "entry_count['compressor']（调用前）",
            [], ["M"], "cache",
            "压缩器已发射条目数（调用前）：×M = 下一窗口首 token 绝对位置（first_window_position）；"
            "与 position_ids 分开维护以保持 prefill→decode→prefill 一致；"
            + ("prefill 起步 0" if prefill else "decode 起步 = 历史闭窗数"),
        )
        self.tensor(
            "cache_entry_count_c_post", "entry_count['compressor']（调用后）",
            [], ["M"], "cache",
            "调用后 = 调用前 + N_win；下一调用以其为 first_window_position 基准",
        )
        self.tensor(
            "cache_entry_count_i", "entry_count['indexer']（调用前）",
            [], ["M"], "cache",
            "Indexer 系列同语义窗口计数（调用前，'indexer' 键独立维护）",
        )
        self.tensor(
            "cache_entry_count_i_post", "entry_count['indexer']（调用后）",
            [], ["M"], "cache",
            "调用后 = 调用前 + N_win_i",
        )
        self.tensor(
            "cache_buffer_c_kv", "buffer_kv['compressor']（调用前）",
            ["B", "Tbuf", "2Dh"], ["M"], "cache",
            "两次满窗之间到达的源 token 投影（kv_proj 输出，2·Dh 特征=Ca|Cb 并存）；"
            + ("起步 None（Tbuf=0）" if prefill else "起步为上次调用余量（<M 条）"),
        )
        self.tensor(
            "cache_buffer_c_gate", "buffer_gate['compressor']（调用前）",
            ["B", "Tbuf", "2Dh"], ["M"], "cache",
            "与 buffer_c_kv 配对的 gate 投影缓冲",
        )
        self.tensor(
            "cache_buffer_c_kv_post", "buffer_kv['compressor']（调用后余量）",
            ["B", "Tbuf'", "2Dh"], ["M"], "cache",
            "store_compression_weights 写回：本调用合并后取 M 对齐前缀，余量 (Tbuf+T)%M 条"
            "（符号表达）留作下调用 buffer",
        )
        self.tensor(
            "cache_buffer_c_gate_post", "buffer_gate['compressor']（调用后余量）",
            ["B", "Tbuf'", "2Dh"], ["M"], "cache",
            "与 buffer_c_kv_post 配对",
        )
        self.tensor(
            "cache_buffer_i_kv", "buffer_kv['indexer']（调用前）",
            ["B", "Tbuf_i", "2Id"], ["Id", "M"], "cache",
            "Indexer 系列窗口缓冲（2·Id 特征），与压缩器系列独立",
        )
        self.tensor(
            "cache_buffer_i_gate", "buffer_gate['indexer']（调用前）",
            ["B", "Tbuf_i", "2Id"], ["Id", "M"], "cache",
            "与 buffer_i_kv 配对",
        )
        self.tensor(
            "cache_buffer_i_kv_post", "buffer_kv['indexer']（调用后余量）",
            ["B", "Tbuf_i'", "2Id"], ["Id", "M"], "cache",
            "Indexer 系列 store_compression_weights 写回余量",
        )
        self.tensor(
            "cache_buffer_i_gate_post", "buffer_gate['indexer']（调用后余量）",
            ["B", "Tbuf_i'", "2Id"], ["Id", "M"], "cache",
            "与 buffer_i_kv_post 配对",
        )
        self.tensor(
            "cache_compressed_c", "compressed_kv['compressor']（调用前）",
            ["B", "Tcc", "Dh"], ["M", "Dh"], "cache",
            "CSA 压缩器已发射条目（调用前，RMSNorm+RoPE 后）；"
            + ("起步 None（Tcc=0）" if prefill else "起步已含历史条目"),
        )
        self.tensor(
            "cache_compressed_c_post", "compressed_kv['compressor']（调用后全量）",
            ["B", "N_comp", "Dh"], ["M", "Dh"], "cache",
            "update_compressor_states 返回的累计条目=cat(调用前, 本调用新条目)；"
            "即本调用 Attention 拼接到 KV 轴的长程条目与 block_bias 的列基准",
        )
        self.tensor(
            "cache_compressed_i", "compressed_kv['indexer']（调用前）",
            ["B", "Tci", "Id"], ["Id", "M"], "cache",
            "Indexer 压缩条目（Id 维，供 scorer 打分）调用前状态；跨调用增长",
        )
        self.tensor(
            "cache_compressed_i_post", "compressed_kv['indexer']（调用后全量）",
            ["B", "Tci+N_win_i", "Id"], ["Id", "M"], "cache",
            "update_compressor_states('indexer') 返回累计条目；scorer 打分与 "
            "top_k=min(Ki, 本轴长度) 的基准",
        )
        self.tensor(
            "cache_overlap_c_kv", "overlap_kv['compressor']（调用前）",
            ["B", "M", "Dh"], ["M", "Dh"], "cache",
            "上一闭窗调用的最后满窗 Ca 尾 chunk[:, -1, :, :Dh]（modeling:290，chunk 为 "
            "[B,N_win,M,2Dh] 视图后取末窗 Ca 半）→[B,M,Dh]；"
            "仅当该调用 N_win>0 才写入；"
            + ("prefill 起步 None（首闭窗调用 new_kv[0] 前半为零值/-inf gate，softmax 权重 0）"
               if prefill else "decode 起步为上次满窗 Ca 尾 [B,M,Dh]"),
        )
        self.tensor(
            "cache_overlap_c_gate", "overlap_gate['compressor']（调用前）",
            ["B", "M", "Dh"], ["M", "Dh"], "cache",
            "与 overlap_c_kv 配对的 gate（[B,M,Dh]）；调用前状态",
        )
        self.tensor(
            "cache_overlap_c_kv_post", "overlap_kv['compressor']（调用后）",
            ["B", "M", "Dh"], ["M", "Dh"], "cache",
            "本调用 N_win>0 时持久化最后满窗 Ca 尾（[B,M,Dh]）；N_win=0 时本调用不读不写，"
            "保持调用前状态（历史保留）",
        )
        self.tensor(
            "cache_overlap_c_gate_post", "overlap_gate['compressor']（调用后）",
            ["B", "M", "Dh"], ["M", "Dh"], "cache",
            "与 overlap_c_kv_post 配对；同样仅 N_win>0 时更新",
        )
        self.tensor(
            "cache_overlap_i_kv", "overlap_kv['indexer']（调用前）",
            ["B", "M", "Id"], ["Id", "M"], "cache",
            "Indexer 系列同一机制（chunk 维度 2·Id，Ca 尾 [B,M,Id]）；"
            "仅当该调用 N_win_i>0 才写入",
        )
        self.tensor(
            "cache_overlap_i_gate", "overlap_gate['indexer']（调用前）",
            ["B", "M", "Id"], ["Id", "M"], "cache",
            "与 overlap_i_kv 配对",
        )
        self.tensor(
            "cache_overlap_i_kv_post", "overlap_kv['indexer']（调用后）",
            ["B", "M", "Id"], ["Id", "M"], "cache",
            "Indexer 系列本调用最后满窗 Ca 尾（[B,M,Id]）；N_win_i=0 时不读不写",
        )
        self.tensor(
            "cache_overlap_i_gate_post", "overlap_gate['indexer']（调用后）",
            ["B", "M", "Id"], ["Id", "M"], "cache",
            "与 overlap_i_kv_post 配对",
        )

    # ------------------------------------------------------------------
    # Decoder 层主流程
    # ------------------------------------------------------------------

    def _decoder_layer_flow(self) -> None:
        prefill = self.is_prefill
        # ==== attn mHC ====
        self._mhc_block("attn", "attn_hc", "hidden_streams_in")
        self._rmsnorm_block("attn", "input_layernorm")
        self._attention_block()
        self._mhc_residual_block("attn")
        # ==== ffn mHC ====
        self._mhc_block("ffn", "ffn_hc", "attn_out_streams")
        self._rmsnorm_block("ffn", "post_attention_layernorm")
        self._moe_block()
        self._mhc_residual_block("ffn")

    def _mhc_block(self, site: str, hc_name: str, in_tid: str) -> None:
        """mHC 权重生成与流坍缩。site: attn/ffn；输入流张量 in_tid。"""
        src = self.src
        pretty = "attn_hc（注意力前）" if site == "attn" else "ffn_hc（MoE 前）"
        p = f"{site}"

        self.tensor(
            f"{p}_mhc_flat", "flattened", ["B", "T", "Nhc×H"], ["Nhc", "H"], "activation",
            "hidden_streams.view(B, T, -1).float()（:955）",
        )
        self.tensor(
            f"{p}_mhc_normed", "flattened(input_norm)", ["B", "T", "Nhc×H"], ["Nhc", "H"],
            "activation",
            "UnweightedRMSNorm（eps=EpsN，:71-72）：x*rsqrt(mean(x²)+eps)，无可学习权重",
        )
        self.tensor(
            f"{p}_mhc_weights", "mhc_projection", ["B", "T", "(2+Nhc)×Nhc"], ["Nhc", "H"],
            "activation",
            "F.linear(flattened, fn)（:958）；随后 split 出原始 pre_w/post_w/comb_w",
        )
        self.tensor(
            f"{p}_mhc_pre_w", "pre_w（原始拆分权重）", ["B", "T", "Nhc"], ["Nhc"], "activation",
            "flattened.split([hc, hc, hc*hc]) 第一段（:960）——未经 affine/激活的原始值",
        )
        self.tensor(
            f"{p}_mhc_post_w", "post_w（原始拆分权重）", ["B", "T", "Nhc"], ["Nhc"], "activation",
            "split 第二段（:960）",
        )
        self.tensor(
            f"{p}_mhc_comb_w", "comb_w（原始拆分矩阵权重）", ["B", "T", "Nhc", "Nhc"],
            ["Nhc"], "activation",
            "split 第三段后 view(..., hc, hc)（:960/:964）——矩阵权重，与 pre/post 的向量本质不同",
        )
        self.tensor(
            f"{p}_mhc_pre", "pre（坍缩权重系数）", ["B", "T", "Nhc"], ["Nhc"], "activation",
            "pre=sigmoid(pre_w·pre_scale+pre_b)+Eps（:968）",
        )
        self.tensor(
            f"{p}_mhc_post", "post（扩展权重系数）", ["B", "T", "Nhc"], ["Nhc"], "activation",
            "post=2·sigmoid(post_w·post_scale+post_b)（:969）；返回给残差段",
        )
        self.tensor(
            f"{p}_mhc_comb", "comb（混合矩阵系数）", ["B", "T", "Nhc", "Nhc"], ["Nhc"], "activation",
            "comb=softmax(comb_w·comb_scale+comb_b)+Eps（:970）→Sinkhorn 列归一+"
            "(Sit-1)×[行归一→列归一]（:976-979）；返回给残差段，使用时转置",
        )
        self.tensor(
            f"{p}_mhc_collapsed", "collapsed（坍缩流）", ["B", "T", "H"], ["Nhc", "H"],
            "activation",
            "collapsed=Σ_n pre_n·stream_n（:983）；进入 norm→子层",
        )
        self.tensor(
            f"{p}_hc_fn", f"{hc_name}.fn", ["(2+Nhc)×Nhc", "Nhc×H"], ["Nhc", "H"], "weight",
            "HyperConnection 权重矩阵（fn）",
        )
        self.tensor(
            f"{p}_hc_base", f"{hc_name}.base", ["(2+Nhc)×Nhc"], ["Nhc"], "weight",
            "HyperConnection 偏置（split 后 pre_b/post_b/comb_b，:961）",
        )
        self.tensor(
            f"{p}_hc_scale", f"{hc_name}.scale", ["3"], ["Nhc"], "weight",
            "三输出各自缩放（unbind 后 pre_scale/post_scale/comb_scale，:962）",
        )
        self.op(
            f"{p}_hc_flatten", f"DeepseekV4DecoderLayer.{hc_name}", pretty + "：流展平+归一",
            "view+float+rmsnorm", [in_tid], [f"{p}_mhc_normed"], [], ["Nhc", "EpsN"],
            [src("954-956", "view(B,T,-1).float() → input_norm")],
            "UnweightedRMSNorm：x*rsqrt(mean(x²)+EpsN)，fp32 计算",
        )
        self.op(
            f"{p}_hc_project", f"DeepseekV4DecoderLayer.{hc_name}", pretty + "：权重投影",
            "linear", [f"{p}_mhc_normed"], [f"{p}_mhc_weights"], [f"{p}_hc_fn"],
            ["Nhc", "H"], [src("958", "F.linear(flattened, self.fn.float())")],
            "权重生成单层感知机（fn.float() 精度）",
        )
        self.op(
            f"{p}_hc_split", f"DeepseekV4DecoderLayer.{hc_name}", pretty + "：原始权重拆分",
            "split+view",
            [f"{p}_mhc_weights"], [f"{p}_mhc_pre_w", f"{p}_mhc_post_w", f"{p}_mhc_comb_w"],
            [], ["Nhc"],
            [src("960-965", "flattened.split([hc, hc, hc*hc], dim=-1)；"
                 "comb_w.view(*shape[:-1], hc, hc) 矩阵化；base.split 同段切出 pre_b/post_b/comb_b")],
            "原始拆分权重（pre_w/post_w 向量、comb_w 矩阵）；affine/激活在后续算子，"
            "base 切片（pre_b/post_b/comb_b）与 scale 切片（:961-962）在此步一起完成",
        )
        self.op(
            f"{p}_hc_pre", f"DeepseekV4DecoderLayer.{hc_name}", pretty + "：pre 门控系数",
            "affine+sigmoid+add",
            [f"{p}_mhc_pre_w", f"{p}_hc_base", f"{p}_hc_scale"], [f"{p}_mhc_pre"], [],
            ["Nhc", "Eps"], [src("968", "pre=sigmoid(pre_w*pre_scale+pre_b)+Eps")],
            "坍缩流权重系数（本算子输出消费于 hc_collapse）；取 base/scale 的 pre 段切片",
        )
        self.op(
            f"{p}_hc_post", f"DeepseekV4DecoderLayer.{hc_name}", pretty + "：post 门控系数",
            "affine+sigmoid+mul",
            [f"{p}_mhc_post_w", f"{p}_hc_base", f"{p}_hc_scale"], [f"{p}_mhc_post"], [],
            ["Nhc"], [src("969", "post=2*sigmoid(post_w*post_scale+post_b)")],
            "范围 (0,2)；返回给残差段；取 base/scale 的 post 段切片",
        )
        self.op(
            f"{p}_hc_comb", f"DeepseekV4DecoderLayer.{hc_name}", pretty + "：comb 混合系数+Sinkhorn",
            "affine+softmax+sinkhorn",
            [f"{p}_mhc_comb_w", f"{p}_hc_base", f"{p}_hc_scale"], [f"{p}_mhc_comb"], [],
            ["Nhc", "Sit", "Eps"],
            [src("970-979", "softmax(dim=-1)+Eps；列归一；(Sit-1)×[行归一→列归一]")],
            "双随机投影约束残差传播非扩张；float 精度（DecoderLayer .to(dtype) 转回）；"
            "取 base/scale 的 comb 段切片",
        )
        self.op(
            f"{p}_hc_collapse", f"DeepseekV4DecoderLayer.{hc_name}", pretty + "：流坍缩",
            "broadcast-mul+sum",
            [f"{p}_mhc_pre", in_tid], [f"{p}_mhc_collapsed"], [], ["Nhc", "H"],
            [src("983", "collapsed=(pre.unsqueeze(-1)*hidden_streams).sum(dim=2).to(dtype)")],
            "Nhc 路流加权求和为单流；mHC 返回 (post, comb, collapsed)",
        )

    def _rmsnorm_block(self, site: str, ln_name: str) -> None:
        p = site
        src = self.src
        label = "注意力子层前" if site == "attn" else "MoE 子层前"
        self.tensor(
            f"{p}_normed", f"{ln_name}(collapsed)", ["B", "T", "H"], ["H", "EpsN"],
            "activation",
            "RMSNorm：fp32 计算、weight fp32 乘、出口 .to(input_dtype)（:54-60）",
        )
        self.tensor(
            f"{p}_norm_weight", f"{ln_name}.weight", ["H"], ["H"], "weight",
            "RMSNorm 可学习缩放（fp32 保持）",
        )
        self.op(
            f"{p}_norm", f"DeepseekV4DecoderLayer.{ln_name}", f"{label} RMSNorm", "rmsnorm",
            [f"{p}_mhc_collapsed", f"{p}_norm_weight"], [f"{p}_normed"], [],
            ["H", "EpsN"],
            [src("46-63", "pow(2).mean(-1)→rsqrt(var+eps)→weight*→to(dtype)")],
            "collapsed=Σpre·stream 未再归一（权重生成路径已有 input_norm），此处标准 RMSNorm",
        )

    def _attention_block(self) -> None:
        """Attention 完整链：低秩 Q、共享 KV、部分 RoPE、缓存更新、CSA 压缩器、
        Indexer、mask 扩展、eager QK→scale→mask→sink→softmax→AV、逆 RoPE、分组输出投影。"""
        prefill = self.is_prefill
        src = self.src
        v = self.values
        P = f"attn"

        # --- Q 支路 ---
        self.tensor(
            "q_a", "q_a_proj(hidden)", ["B", "T", "Rq"], ["Rq", "H"], "activation",
            "低秩 Q 下投影",
        )
        self.tensor(
            "q_a_normed", "q_a_norm(q_a)", ["B", "T", "Rq"], ["Rq", "EpsN"], "activation",
            "RMSNorm(q_lora_rank, eps=EpsN)（:784）；即 HF :813 的 q_residual，"
            "同时供 Indexer 的 q_b_proj 使用（:561）",
        )
        self.tensor(
            "q_a_norm_weight", "q_a_norm.weight", ["Rq"], ["Rq"], "weight",
            "DeepseekV4RMSNorm 带权缩放（fp32 保持；:784）",
        )
        self.tensor(
            "q_weight", "q_a_proj.weight", ["Rq", "H"], ["Rq", "H"], "weight",
            "Linear(H, Rq, bias=False)",
        )
        self.tensor(
            "q_b_weight", "q_b_proj.weight", ["Nh×Dh", "Rq"], ["Nh", "Dh", "Rq"], "weight",
            "Linear(Rq, Nh*Dh, bias=False)",
        )
        self.tensor(
            "q_head", "q.view/transpose", ["B", "Nh", "T", "Dh"], ["Nh", "Dh"], "activation",
            "q_b_proj(q_a_normed).view(B,T,-1,Dh).transpose(1,2)（:814）",
        )
        self.tensor(
            "q_normed", "q_b_norm(q_head)", ["B", "Nh", "T", "Dh"], ["Nh", "Dh"],
            "activation",
            "UnweightedRMSNorm（:786，:815）——按最后维 Dh 归一，无权重",
        )
        self.tensor(
            "q_rope", "q（部分 RoPE 后）", ["B", "Nh", "T", "Dh"], ["Nh", "Dh", "Rk"],
            "activation",
            "apply_rotary_pos_emb：尾部 Rk=64 通道旋转（interleaved 配对 cos/sin "
            "repeat_interleave(2)；rotate_half 用 x[..., 0::2]/x[..., 1::2] 交错，"
            ":332-356）；前导 Dh-Rk 通道不动",
        )
        self.op(
            "q_a_proj", "DeepseekV4Attention", "低秩 Q 下投影", "linear",
            ["attn_normed"], ["q_a"], ["q_weight"], ["Rq", "H"],
            [src("783", "self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)"),
             src("813", "q_residual = self.q_a_norm(self.q_a_proj(hidden_states))")],
            "q_residual 同时供 Indexer 的 q_b_proj 使用（:561）",
        )
        self.op(
            "q_a_norm", "DeepseekV4Attention.q_a_norm", "Q 秩归一", "rmsnorm",
            ["q_a"], ["q_a_normed"], ["q_a_norm_weight"], ["Rq", "EpsN"],
            [src("784", "self.q_a_norm = DeepseekV4RMSNorm(q_lora_rank, eps=rms_norm_eps)")],
            "DeepseekV4RMSNorm（带权重）；输出 q_residual 复用于 Indexer 查询投影",
        )
        self.op(
            "q_b_proj", "DeepseekV4Attention", "Q 上投影+头重排", "linear+view+transpose",
            ["q_a_normed"], ["q_head"], ["q_b_weight"], ["Nh", "Dh", "Rq"],
            [src("785", "nn.Linear(q_lora_rank, num_heads*head_dim, bias=False)"),
             src("814", ".view(*hidden_shape).transpose(1, 2)")],
            "hidden_shape=(B,T,-1,Dh)",
        )
        self.op(
            "q_b_norm", "DeepseekV4Attention.q_b_norm", "Q 头归一", "rmsnorm-unweighted",
            ["q_head"], ["q_normed"], [], ["Nh", "Dh", "EpsN"],
            [src("786", "self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=rms_norm_eps)"),
             src("815", "q = self.q_b_norm(q)")],
            "无权重 RMSNorm",
        )

        # --- RoPE（compress 型，本层共用）---
        self.tensor(
            "cos_q", "cos（compress）", ["1", "T", "Rk÷2"], ["Rk", "ThetaC", "Yf"], "activation",
            "freqs=position_ids[...,None]·inv_freq→cos（modeling:148-153）；"
            "默认 position_ids [1,T] → cos [1,T,Rk÷2]（与本图 position_ids 一致）；"
            "attention_scaling=1.0（yarn 时强制）；inv_freq 由 yarn 混合公式计算（theta=θc）",
        )
        self.tensor(
            "sin_q", "sin（compress）", ["1", "T", "Rk÷2"], ["Rk", "ThetaC", "Yf"], "activation",
            "同 cos；interleaved RoPE 每 pair 一个 θ，apply 时 repeat_interleave(2)"
            "并 unsqueeze(1) 广播到头维",
        )
        self.op(
            "rope_compress", "DeepseekV4RotaryEmbedding", "compress RoPE cos/sin 生成",
            "outer+cos/sin", ["position_ids"], ["cos_q", "sin_q"], [],
            ["Rk", "ThetaC", "Yf"],
            [src("140-153", "freqs=position_ids[...,None].float()*inv_freq；cos/sin×attention_scaling"),
             self.srcf("modeling_rope_utils.py", "446-483",
                       "_compute_yarn_parameters：插值/外推混合 inv_freq（theta=θc, factor=Yf）")],
            "模型级 rotary_emb 缓存 main/compress 两套 inv_freq；本层全部取 compress（:774、:811）",
        )
        self.op(
            "q_rope_apply", "apply_rotary_pos_emb", "Q 部分 RoPE", "rope-interleaved",
            ["q_normed", "cos_q", "sin_q"], ["q_rope"], [], ["Rk"],
            [src("339-356", "cos.repeat_interleave(2,-1).unsqueeze(1)；nope/rope 切分；"
                 "rotated=(rope*cos+rotate_half(rope)*sin).to(dtype)")],
            "只旋转尾部 Rk=64 通道；fp32 计算",
        )

        # --- KV 支路（共享 KV）---
        self.tensor(
            "kv_proj_out", "kv_norm(kv_proj(hidden))", ["B", "T", "Dh"], ["Dh", "H"],
            "activation",
            "单 KV 头投影+归一",
        )
        self.tensor(
            "kv_weight", "kv_proj.weight", ["Dh", "H"], ["Dh", "H"], "weight",
            "Linear(H, Dh, bias=False)（单 KV 头：输出即 head_dim 而非 Nkv*Dh）",
        )
        self.tensor(
            "kv_norm_weight", "kv_norm.weight", ["Dh"], ["Dh"], "weight",
            "RMSNorm(Dh, eps=EpsN)",
        )
        self.tensor(
            "kv_head", "kv.view/transpose", ["B", "1", "T", "Dh"], ["Nkv", "Dh"], "activation",
            "view(B,T,-1,Dh).transpose(1,2)（:818）；1 个 KV 头",
        )
        self.tensor(
            "kv_rope", "kv（部分 RoPE 后）", ["B", "1", "T", "Dh"], ["Nkv", "Dh", "Rk"],
            "activation",
            "K=V 同一张量；RoPE 后写缓存",
        )
        self.op(
            "kv_proj", "DeepseekV4Attention", "共享 KV 投影+归一", "linear+rmsnorm",
            ["attn_normed"], ["kv_proj_out"], ["kv_weight", "kv_norm_weight"], ["Dh", "H"],
            [src("787-788", "kv_proj=Linear(hidden,Dh)）；kv_norm=RMSNorm(Dh)")],
            "num_key_value_heads=1：直接投影到 head_dim（单 KV 头）",
        )
        self.op(
            "kv_head_view", "DeepseekV4Attention", "KV 头视图重排", "view+transpose",
            ["kv_proj_out"], ["kv_head"], [], ["Nkv", "Dh"],
            [src("818", ".view(*hidden_shape).transpose(1, 2)；hidden_shape=(B,T,-1,Dh)")],
            "view(B,T,1,Dh).transpose(1,2)→[B,1,T,Dh]；1 个 KV 头",
        )
        self.op(
            "kv_rope_apply", "apply_rotary_pos_emb", "KV 部分 RoPE", "rope-interleaved",
            ["kv_head", "cos_q", "sin_q"], ["kv_rope"], [], ["Nkv", "Rk"],
            [src("818-819", "kv = apply_rotary_pos_emb(kv, cos, sin)")],
            "K=V 共享，V 也带 RoPE（后续用逆 RoPE 抵消）",
        )

        # --- 缓存更新 ---
        self.tensor(
            "kv_full", "cache.update 返回 full", ["B", "1", "L_kv", "Dh"], ["Lkv", "Dh", "W"],
            "cache",
            "full=cat(keys, kv_rope)（:198）：本次参与 Attention 的完整 KV 视图，"
            "L_kv=min(P,W-1)+T；"
            + ("prefill 首调用 keys 空，full=本调用 T 个新 KV"
               if prefill else "decode：keys 为 min(P,W-1) 尾部历史 + 本调用 1 个新 KV"),
        )
        self.op(
            "cache_update", "DeepseekV4CSACache.update", "主 KV 缓存更新（K=V）", "cat+slice",
            ["kv_rope", "cache_keys"], ["kv_full", "cache_keys_post"], [], ["Lkv", "Dh", "W"],
            [src("189-201", "full=cat([keys, kv],dim=-2)；keys=full[:, :, -(W-1):, :]；"
                  "values=keys（K=V 同存储）；返回 full, full"),
             self.srcf("cache_utils.py", "237-274", "DynamicSlidingWindowLayer.update 同族语义")],
            "读调用前持久态 cache_keys、写调用后持久态 cache_keys_post=min(P+T,W-1) 尾部"
            "（原地状态更新以调用前/调用后两个张量分别表达）；本次可读 full 与持久保存长度不同；"
            + ("prefill 起步惰性初始化（keys 为空 tensor）"
               if prefill else "decode 起步 keys 已含历史尾部"),
        )

        # --- CSA 压缩器 ---
        self.tensor(
            "c_kv", "compressor.kv_proj(hidden)", ["B", "T", "2Dh"], ["Dh", "H"], "activation",
            "2·Dh 特征：[..., :Dh]=Ca（供下一窗）、[..., Dh:]=Cb（供本窗）",
        )
        self.tensor(
            "c_gate", "compressor.gate_proj(hidden)", ["B", "T", "2Dh"], ["Dh", "H"],
            "activation",
            "与 c_kv 配对的 softmax gate 投影",
        )
        self.tensor(
            "c_kv_weight", "compressor.kv_proj.weight", ["2Dh", "H"], ["Dh", "H"], "weight",
            "Linear(H, 2*Dh, bias=False)；FP8 混合精度声明下保持 BF16"
            "（_keep_in_fp32_modules 非严格列表，:1246-1252）",
        )
        self.tensor(
            "c_gate_weight", "compressor.gate_proj.weight", ["2Dh", "H"], ["Dh", "H"], "weight",
            "Linear(H, 2*Dh, bias=False)",
        )
        self.tensor(
            "c_position_bias", "compressor.position_bias", ["M", "2Dh"], ["M", "Dh"], "weight",
            "可学习窗口内位置偏置（加到 gate 上）",
        )
        self.tensor(
            "c_kv_norm_weight", "compressor.kv_norm.weight", ["Dh"], ["Dh"], "weight",
            "RMSNorm(Dh, eps=EpsN)",
        )
        self.tensor(
            "c_chunk_kv", "chunk_kv（M 对齐前缀）", ["B", "N_win×M", "2Dh"], ["M"], "activation",
            "cat(buffer_kv, kv) 后取 M 对齐前缀；余量回写 buffer（store_compression_weights，"
            ":216-224）",
        )
        self.tensor(
            "c_chunk_gate", "chunk_gate（M 对齐前缀）", ["B", "N_win×M", "2Dh"], ["M"],
            "activation", "与 c_chunk_kv 配对",
        )
        self.tensor(
            "c_windows_kv", "new_kv（Ca/Cb 重叠布局）",
            ["B", "N_win", "2M", "Dh"], ["M", "Dh"], "activation",
            "view(B,N_win,M,2Dh)→重排：后半 [M:]=本窗 Cb，前半 [:M]=前窗 Ca；"
            "窗 0 前半由 cache.overlap 提供（首调用零值，:647-666）",
        )
        self.tensor(
            "c_windows_gate", "new_gate（Ca/Cb 重叠布局 gate）",
            ["B", "N_win", "2M", "Dh"], ["M", "Dh"], "activation",
            "本窗 Cb gate+position_bias；前窗 Ca gate；窗 0 前半 -inf（softmax 权重 0）",
        )
        self.tensor(
            "c_compressed", "compressed（本调用新条目，RoPE 前）", ["B", "N_win", "Dh"],
            ["M", "Dh"], "activation",
            "kv_norm(Σ_slot softmax(gate)·kv)（fp32 softmax）",
        )
        self.tensor(
            "c_compressed_rope", "compressed（本调用新条目，RoPE 后）", ["B", "N_win", "Dh"],
            ["M", "Dh", "Rk"], "activation",
            "apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)（:677）",
        )
        self.tensor(
            "c_cos", "压缩条目 RoPE cos", ["B", "N_win", "Rk÷2"], ["Rk", "ThetaC"],
            "activation",
            "positions=arange(N_win)·M+first_window_position（确定性绝对位置，:673-675）；"
            "first_window_position=entry_count['compressor']（调用前）·M",
        )
        self.tensor(
            "c_sin", "压缩条目 RoPE sin", ["B", "N_win", "Rk÷2"], ["Rk", "ThetaC"],
            "activation", "与 c_cos 配对",
        )
        self.tensor(
            "c_block_bias", "block_bias（压缩条目可见性）",
            ["B", "1", "T", "N_comp"], ["M", "Ki"], "mask",
            "Indexer top-k 集合+因果阈值：查询 t 只可见选中且 w≤(t+1)//M 的条目；"
            "0 可见/-inf 屏蔽（:692-699）",
        )
        self.op(
            "c_kv_gate_proj", "DeepseekV4CSACompressor", "压缩器 KV/gate 投影", "linear×2",
            ["attn_normed"], ["c_kv", "c_gate"], ["c_kv_weight", "c_gate_weight"], ["Dh", "H"],
            [src("609-616", "kv_proj=Linear(H,2Dh)；gate_proj=Linear(H,2Dh)；"
                 "position_bias=(M,2Dh)"),
             src("632-633", "kv=self.kv_proj(hidden_states)；gate=self.gate_proj(hidden_states)")],
            "每 token 产 Ca|Cb 两序列并存于一张量",
        )
        self.op(
            "c_store_window", "DeepseekV4CSACache.store_compression_weights", "窗口缓冲合并+闭窗",
            "cat+slice",
            ["c_kv", "c_gate", "cache_buffer_c_kv", "cache_buffer_c_gate",
             "cache_entry_count_c"],
            ["c_chunk_kv", "c_chunk_gate", "cache_buffer_c_kv_post",
             "cache_buffer_c_gate_post"], [],
            ["M"],
            [src("203-224", "cat([buffered, kv],dim=1)→usable=(len//M)*M→"
                  "buffer=kv[:, usable:]→返回 kv[:, :usable], first_window_position=entry_count·M")],
            "读调用前 buffer/entry_count，写调用后 buffer 余量（<M 条）；"
            "first_window_position=entry_count['compressor']（调用前）·M（闭窗绝对起点）；"
            + ("prefill 起步 buffer=None"
               if prefill else "decode 起步 buffer 含上次余量"),
        )
        self.op(
            "c_window_view", "DeepseekV4CSACompressor", "窗口视图重排（Ca/Cb 重叠布局）",
            "view+assign",
            ["c_chunk_kv", "c_chunk_gate", "cache_overlap_c_kv", "cache_overlap_c_gate"],
            ["c_windows_kv", "c_windows_gate", "cache_overlap_c_kv_post",
             "cache_overlap_c_gate_post"],
            ["c_position_bias"], ["M", "Dh"],
            [src("641-666", "view(B,n_win,M,-1)；+position_bias；new_kv[:, :, M:]=chunk[..., Dh:]；"
                  "new_kv[:, 1:, :M]=chunk[:, :-1, :, :Dh]；prior=overlap 填窗 0 前半"),
             src("278-292", "update_overlap_state：读 prior_kv/prior_gate，"
                  "持久化本调用 chunk[:, -1, :, :Dh]（[B,M,Dh]）及配对 gate")],
            "Ca/Cb 重叠布局：条目宽 2M、步长 M；整个分支仅在 N_win>0（本调用闭窗）时执行——"
            "N_win=0 时 overlap 读写都不发生（保留调用前状态/首调用缺省），"
            "chunk_kv.shape[1]==0 走零条目分支（:678-679）；"
            "读调用前 overlap [B,M,Dh]（窗 0 前半），写调用后 overlap（本调用最后满窗 Ca 尾）",
        )
        self.op(
            "c_pool", "DeepseekV4CSACompressor", "压缩条目 softmax 汇聚+归一", "softmax+mul+sum+rmsnorm",
            ["c_windows_kv", "c_windows_gate", "c_kv_norm_weight"], ["c_compressed"], [],
            ["M", "Dh", "EpsN"],
            [src("668-672", "kv_norm((new_kv*new_gate.softmax(dim=2, dtype=fp32).to(dtype)).sum(dim=2))")],
            "softmax fp32 稳定性（bf16 会坍缩相近对）",
        )
        self.op(
            "c_rope_gen", "DeepseekV4CSACompressor.rotary_emb", "压缩条目 RoPE cos/sin 生成",
            "arange+outer+cos/sin",
            ["cache_entry_count_c"], ["c_cos", "c_sin"], [], ["Rk", "ThetaC", "M"],
            [src("673-675", "positions=arange(n_windows)*compress_rate+first_window_position"
                  "（first_window_position=entry_count·M）"),
             src("676", "cos, sin = self.rotary_emb(compressed, position_ids=positions, "
                  "layer_type='compress')"),
             src("140-153", "freqs=positions[...,None]·inv_freq→cos/sin（[B,N_win,Rk÷2]）")],
            "压缩条目 RoPE 的确定性位置链：entry_count（调用前）→first_window_position→positions；"
            "同一 'compress' rope（θc）与查询/主 KV 共用",
        )
        self.op(
            "c_rope", "apply_rotary_pos_emb（CSA 压缩条目）", "压缩条目 RoPE", "rope-interleaved",
            ["c_compressed", "c_cos", "c_sin"], ["c_compressed_rope"], [], ["Rk", "ThetaC"],
            [src("677", "compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)")],
            "确定性绝对位置 i·M+first_window_position（i=窗口序号）保证跨调用拼接因果正确；"
            "索引器系列同机制",
        )
        self.op(
            "c_update_states", "DeepseekV4CSACache.update_compressor_states", "压缩条目追加",
            "cat+add",
            ["c_compressed_rope", "cache_compressed_c", "cache_entry_count_c"],
            ["cache_compressed_c_post", "cache_entry_count_c_post"], [], ["M"],
            [src("226-237", "compressed_kv=cat([...],dim=1)；entry_count+=compressed.shape[1]")],
            "读调用前累计/计数，写调用后全量 cache_compressed_c_post（即 Attention 可读条目）"
            "与调用后计数",
        )
        # --- Lightning Indexer ---
        # （块偏置 c_block_bias 在 Indexer top-k 之后构建，与源码 :690→:692-699 顺序一致）
        self.tensor(
            "i_kv", "indexer.kv_proj(hidden)", ["B", "T", "2Id"], ["Id", "H"], "activation",
            "Indexer 压缩 KV 投影（2·Id 特征 Ca|Cb）",
        )
        self.tensor(
            "i_gate", "indexer.gate_proj(hidden)", ["B", "T", "2Id"], ["Id", "H"], "activation",
            "Indexer gate 投影",
        )
        self.tensor(
            "i_kv_weight", "indexer.kv_proj.weight", ["2Id", "H"], ["Id", "H"], "weight",
            "Linear(H, 2*Id, bias=False)",
        )
        self.tensor(
            "i_gate_weight", "indexer.gate_proj.weight", ["2Id", "H"], ["Id", "H"], "weight",
            "Linear(H, 2*Id, bias=False)",
        )
        self.tensor(
            "i_position_bias", "indexer.position_bias", ["M", "2Id"], ["Id", "M"], "weight",
            "窗口内位置偏置（M, 2Id）",
        )
        self.tensor(
            "i_kv_norm_weight", "indexer.kv_norm.weight", ["Id"], ["Id"], "weight",
            "RMSNorm(Id, eps=EpsN)",
        )
        self.tensor(
            "i_chunk_kv", "indexer chunk_kv", ["B", "N_win_i×M", "2Id"], ["Id", "M"], "activation",
            "indexer 系列 store_compression_weights 结果（'indexer' 键独立计数/缓冲）",
        )
        self.tensor(
            "i_chunk_gate", "indexer chunk_gate", ["B", "N_win_i×M", "2Id"], ["Id", "M"],
            "activation", "与 i_chunk_kv 配对的 gate M 对齐前缀",
        )
        self.tensor(
            "i_windows_kv", "indexer new_kv（Ca/Cb 布局）", ["B", "N_win_i", "2M", "Id"],
            ["Id", "M"], "activation",
            "同 CSA 压缩器重叠布局，head_dim→Id",
        )
        self.tensor(
            "i_windows_gate", "indexer new_gate", ["B", "N_win_i", "2M", "Id"], ["Id", "M"],
            "activation", "同上；窗 0 前半 -inf",
        )
        self.tensor(
            "i_compressed", "indexer compressed（本调用新条目）", ["B", "N_win_i", "Id"],
            ["Id", "M"], "activation",
            "kv_norm(Σ softmax(gate)·kv) 后按 i·M+first_window_position RoPE",
        )
        self.tensor(
            "i_q", "indexer q（RoPE 后）", ["B", "T", "Ih", "Id"], ["Ih", "Id", "Rq"],
            "activation",
            "q_b_proj(q_residual).view(B,T,-1,Id).transpose(1,2)→RoPE→transpose 回 [B,T,Ih,Id]"
            "（:561-562）；scorer 以 [B,T,Ih,Id]×[B,1,Id,Tci] 广播相乘",
        )
        self.tensor(
            "i_q_weight", "indexer.q_b_proj.weight", ["Ih×Id", "Rq"], ["Ih", "Id", "Rq"],
            "weight", "Linear(Rq, Ih*Id, bias=False)；输入是 Attention 的 q_residual",
        )
        self.tensor(
            "i_scores", "indexer scores", ["B", "T", "Tci+N_win_i"], ["Id", "Ih"], "activation",
            "scorer：Σ_h w·ReLU(q·K^T)·Id^-0.5（:452-456）；K 用追加后全量条目 "
            "cache_compressed_i_post（:556-566），分数轴=Tci+N_win_i（含本调用新增，"
            "prefill 旧 Tci=0 也覆盖新条目）",
        )
        self.tensor(
            "i_w_weight", "indexer.scorer.weights_proj.weight", ["Ih", "H"], ["Ih", "H"],
            "weight", "Linear(H, Ih, bias=False)；每查询头权重 w",
        )
        self.tensor(
            "indexer_topk", "top_k_indices", ["B", "T", "min(Ki,Tci+N_win_i)"], ["Ki", "Id", "M"],
            "index",
            "每查询选中的压缩条目索引；实际 top_k=min(Ki, 本调用更新后 Indexer 条目数"
            " Tci+N_win_i)（:566）；无效位 -1 哨兵（future_mask -inf→topk→where，"
            ":574-581）",
        )
        self.op(
            "i_kv_gate_proj", "DeepseekV4Indexer", "Indexer KV/gate 投影", "linear×2",
            ["attn_normed"], ["i_kv", "i_gate"], ["i_kv_weight", "i_gate_weight"], ["Id", "H"],
            [src("496-498", "kv_proj=Linear(H,2Id)；gate_proj=Linear(H,2Id)；"
                  "position_bias=(M,2Id)")],
            "Indexer 独立小型压缩器（与外层压缩器同窗但对 'indexer' 键独立计数）",
        )
        self.op(
            "i_store_window", "DeepseekV4CSACache.store_compression_weights", "Indexer 窗口缓冲",
            "cat+slice",
            ["i_kv", "i_gate", "cache_buffer_i_kv", "cache_buffer_i_gate",
             "cache_entry_count_i"],
            ["i_chunk_kv", "i_chunk_gate", "cache_buffer_i_kv_post",
             "cache_buffer_i_gate_post"], [], ["Id", "M"],
            [src("519-523", "store_compression_weights('indexer', kv, gate)")],
            "与压缩器同一缓冲机制、不同键名（kv/gate 前缀都返回）；"
            "first_window_position 来自 entry_count['indexer']（调用前）；overlap 状态同理",
        )
        self.op(
            "i_window_view", "DeepseekV4Indexer", "Indexer Ca/Cb 重叠布局", "view+assign",
            ["i_chunk_kv", "i_chunk_gate", "cache_overlap_i_kv", "cache_overlap_i_gate"],
            ["i_windows_kv", "i_windows_gate", "cache_overlap_i_kv_post",
             "cache_overlap_i_gate_post"],
            ["i_position_bias"], ["Id", "M"],
            [src("528-543", "new_kv[:, :, M:]=chunk[..., Id:]；new_kv[:, 1:, :M]=前窗 Ca；"
                  "update_overlap_state('indexer') 读 prior 填窗 0 前半并持久化本调用 Ca 尾")],
            "与外层压缩器同构（head_dim→Id）；仅 N_win_i>0（本调用 Indexer 闭窗）时执行，"
            "N_win_i=0 时 overlap 读写都不发生（保留历史/首调用缺省，chunk 零条目分支）；"
            "读调用前 overlap [B,M,Id]、写调用后 overlap（前后状态分离表达）",
        )
        self.op(
            "i_pool_rope", "DeepseekV4Indexer", "Indexer 压缩条目汇聚+RoPE",
            "softmax+mul+sum+rmsnorm+rope",
            ["i_windows_kv", "i_windows_gate", "i_kv_norm_weight", "cache_entry_count_i"],
            ["i_compressed"], [],
            ["Id", "M", "EpsN", "Rk", "ThetaC"],
            [src("545-552", "kv_norm((new_kv*new_gate.softmax(dim=2)).sum(dim=2))；"
                  "positions=arange(n_win)*compress_rate+first_window_position；"
                  "rotary_emb('compress')；apply_rotary_pos_emb")],
            "压缩条目 RoPE 与查询 RoPE 同 θ（θc）保证平移不变（:479-485）；"
            "位置链经 entry_count['indexer']（调用前）→first_window_position",
        )
        self.op(
            "i_update_states", "DeepseekV4CSACache.update_compressor_states", "Indexer 条目追加",
            "cat+add",
            ["i_compressed", "cache_compressed_i", "cache_entry_count_i"],
            ["cache_compressed_i_post", "cache_entry_count_i_post"], [], ["Id", "M"],
            [src("556-558", "compressed_kv = update_compressor_states('indexer', compressed)")],
            "读调用前累计/计数，写调用后全量 cache_compressed_i_post（scorer 与 topk 消费）",
        )
        self.op(
            "i_q_proj", "DeepseekV4Indexer", "Indexer 查询投影+RoPE", "linear+view+rope",
            ["q_a_normed", "cos_q", "sin_q"], ["i_q"], ["i_q_weight"], ["Ih", "Id", "Rq", "Rk"],
            [src("500", "q_b_proj=Linear(q_lora_rank, Ih*Id)"),
             src("560-562", "cos_q,sin_q=rotary_emb(hidden, position_ids)；q=q_b_proj(q_residual)…"
                  "apply_rotary_pos_emb(q, cos_q, sin_q)")],
            "输入是归一后的 q_residual（=q_a_normed，HF :813/:561）；"
            "查询位置=当前 position_ids（与压缩条目的确定性位置不同）",
        )
        self.op(
            "i_scorer", "DeepseekV4IndexerScorer", "Indexer 打分", "matmul+relu+mul+sum",
            ["i_q", "cache_compressed_i_post", "attn_normed", "i_w_weight"],
            ["i_scores"], [], ["Id", "Ih"],
            [src("443-456", "scores=matmul(q.float(), compressed_kv.T.float().unsqueeze(1))；"
                  "relu·Id^-0.5；weights=weights_proj(hidden)·Ih^-0.5；"
                  "(scores*w.unsqueeze(-1)).sum(dim=2)")],
            "fp32 打分；输出 [B,T,Tci+N_win_i]",
        )
        self.op(
            "i_topk", "DeepseekV4Indexer", "Indexer top-k 选择", "mask+topk+where",
            ["i_scores", "position_ids", "cache_compressed_i_post"], ["indexer_topk"], [],
            ["Ki", "M"],
            [src("565-583", "top_k=min(Ki, compressed_len)；future_mask=entry_indices>="
                  "(position_ids+1)//M；masked_fill(-inf)→topk→invalid→-1 哨兵")],
            "compressed_len=本调用更新后条目数（cache_compressed_i_post 轴长）；"
            "查询 t 不见因果阈值之后的条目；选不中的置 -1 由块偏置处理",
        )

        # --- 压缩条目块偏置（Indexer 之后，源码 :690→:692-699 顺序）---
        self.op(
            "c_block_bias", "DeepseekV4CSACompressor", "压缩条目块偏置构建", "scatter",
            ["cache_compressed_c_post", "indexer_topk"], ["c_block_bias"], [], ["M", "Ki"],
            [src("690-699", "valid=top_k_indices>=0；safe_indices=where(valid, idx, N_comp)；"
                  "block_bias=full(-inf).scatter_(-1, safe_indices, 0.0)[..., :N_comp]")],
            "Indexer 选中的条目置 0（可见），其余 -inf；-1 哨兵映射到越界列后裁掉；"
            "N_comp=本调用更新后压缩器条目数",
        )

        # --- KV 拼接与 mask 扩展 ---
        self.tensor(
            "kv_all", "kv（主 KV + 压缩条目）", ["B", "1", "L_kv+N_comp", "Dh"],
            ["Lkv", "Dh", "M", "W"], "activation",
            "cat([kv_full, compressed_kv], dim=2)（:829）；K=V；压缩条目在 KV 轴尾部",
        )
        self.op(
            "kv_concat", "DeepseekV4Attention", "主 KV 与压缩条目拼接", "cat",
            ["kv_full", "cache_compressed_c_post"], ["kv_all"], [], ["Lkv", "M", "Dh"],
            [src("824-829", "compressed_kv, block_bias = self.compressor(...)；"
                  "kv = torch.cat([kv, compressed_kv], dim=2)")],
            "拼接发生在缓存 update 之后（Attention 读到的 KV = 滑动 full + 更新后全部压缩条目）；"
            "K=V 共享（V 同一 view 参与下游 AV）",
        )
        self.tensor(
            "mask_ext", "attention_mask（扩展后）",
            ["B", "1", "T", "L_mask+N_comp"], ["Lmask", "M", "W"], "mask",
            "模型级滑动因果 mask（sliding 层尺寸）cat block_bias（:837-841）；"
            "0 可见/-inf 屏蔽；eager float mask",
        )
        self.tensor(
            "mask_model", "模型级 4D float mask", ["B", "1", "T", "L_mask"], ["Lmask", "W"],
            "mask",
            "create_sliding_window_causal_mask：sliding_window_causal 函数（kv_idx>q_idx-W"
            " 且 kv_idx≤q_idx）+可选 padding；eager_mask 生成 0/-inf float；"
            "q_offset=累计历史、kv_offset=max(cumulative-W+1,0)",
        )
        self.op(
            "mask_model_build", "masking_utils.create_sliding_window_causal_mask",
            "模型级滑动因果 mask 构建", "mask-materialize",
            ["cache_cumlen"], ["mask_model"], [], ["Lmask", "W"],
            [self.srcf("masking_utils.py", "1078-1162",
                       "sliding_window_causal_mask_function(W)=and_masks(sliding_overlay, causal)"),
             self.srcf("masking_utils.py", "531-560",
                       "eager_mask：bool→0/-inf float 4D"),
             self.srcf("cache_utils.py", "276-286",
                       "get_mask_sizes：is_full→W-1+T else cumulative+T；offset=max(cum-W+1,0)")],
            "按首个 sliding 层（层0）尺寸；本层共用该 mask（:1341-1350 所有层同一 mask）",
        )
        self.op(
            "mask_extend", "DeepseekV4Attention", "mask 扩展到压缩条目", "cat",
            ["mask_model", "c_block_bias"], ["mask_ext"], [], ["Lmask", "M"],
            [src("831-841", "kv.shape[2]>mask.shape[-1] 时：block_bias 存在则 cat；"
                 "否则 F.pad(0)")],
            "CSA 有 block_bias → cat；HCA/无 Indexer 层走 0 填充分支（本层用 cat）",
        )

        # --- eager attention 核心 ---
        self.tensor(
            "kv_repeated", "repeat_kv(kv)", ["B", "Nh", "L_kv+N_comp", "Dh"],
            ["Nh", "Lkv", "Dh", "M", "W"], "activation",
            "num_key_value_groups=Nh：单 KV 头广播到 Nh 头（:702-711 expand+reshape）；"
            "n_rep=Nh/1=Nh",
        )
        self.tensor(
            "qk_logits", "attn_weights（QK^T·scale）", ["B", "Nh", "T", "L_kv+N_comp"],
            ["Nh", "Lkv", "Dh", "M"], "activation",
            "matmul(q, k^T)·Dh^-0.5（:726 scaling=head_dim^-0.5）",
        )
        self.tensor(
            "sinks", "sinks 参数", ["Nh"], ["Nh"], "weight",
            "每头可学习注意力 sink（gpt-oss 风格；:793 nn.Parameter(Nh)）",
        )
        self.tensor(
            "combined_logits", "cat(attn_weights, sinks)", ["B", "Nh", "T", "L_kv+N_comp+1"],
            ["Nh", "Lkv", "Dh", "M"], "activation",
            "sink 拼接在 KV 轴尾（:730-731）；KV 轴长 L_kv+N_comp（概率序列轴）+1（sink）",
        )
        self.tensor(
            "probs", "softmax 后概率", ["B", "Nh", "T", "L_kv+N_comp+1"], ["Nh", "Lkv", "Dh", "M"],
            "activation",
            "combined-max 减最大值（防 bf16 溢出，:736）→softmax(dim=-1)（:737）",
        )
        self.tensor(
            "attn_probs", "scores（去 sink）", ["B", "Nh", "T", "L_kv+N_comp"],
            ["Nh", "Lkv", "Dh", "M"], "activation",
            "probs[..., :-1]（:738）——概率序列轴=L_kv+N_comp，与 AV 的 KV 轴闭合；"
            "dropout p=0（推理）",
        )
        self.tensor(
            "attn_head_out", "attn_output（头维）", ["B", "T", "Nh", "Dh"], ["Nh", "Dh"],
            "activation", "matmul(attn_probs, v)→[B,Nh,T,Dh]→transpose(1,2).contiguous()"
            "→[B,T,Nh,Dh]（:740-741）",
        )
        self.tensor(
            "qk_masked", "attn_weights（+mask 后）", ["B", "Nh", "T", "L_kv+N_comp"],
            ["Nh", "Lkv", "Dh", "M"], "activation",
            "attn_weights + attention_mask（:727-728，加法型 mask 生成新张量）",
        )
        self.op(
            "repeat_kv", "eager_attention_forward", "KV 头广播", "expand+reshape",
            ["kv_all"], ["kv_repeated"], [], ["Nh", "Lkv"],
            [src("702-711", "repeat_kv：hidden[:, :, None].expand(...).reshape(B, Nkv*n_rep, S, Dh)"),
             src("724-725", "key_states=repeat_kv(key, num_key_value_groups)；value 同")],
            "n_rep=Nh（num_key_value_groups，:776）",
        )
        self.op(
            "qk_scaled", "eager_attention_forward", "QK^T 缩放", "matmul+mul",
            ["q_rope", "kv_repeated"], ["qk_logits"], [], ["Nh", "Dh", "M"],
            [src("726", "attn_weights = matmul(query, key_states.transpose(2,3)) * scaling")],
            "scaling=head_dim^-0.5（:781）",
        )
        self.op(
            "mask_add", "eager_attention_forward", "mask 加法", "add",
            ["qk_logits", "mask_ext"], ["qk_masked"], [], ["Lmask", "M", "W"],
            [src("727-728", "if attention_mask is not None: attn_weights = attn_weights + attention_mask")],
            "加法型 mask（0 可见/-inf 屏蔽）；+ 生成新张量（输出 qk_masked）",
        )
        self.op(
            "sink_softmax", "eager_attention_forward", "sink 拼接+稳定 softmax",
            "cat+sub+softmax+slice",
            ["qk_masked", "sinks"], ["attn_probs"], [], ["Nh", "M"],
            [src("730-739", "sinks.expand；cat(dim=-1)；combined-=max；softmax(dtype)；"
                  "scores=probs[..., :-1]；dropout(p=0).to(v dtype)")],
            "sink 为每头可学习标量参与 softmax 竞争但不产出输出（去 sink 切片）；"
            "推理 training=False→dropout p=0.0（:852）",
        )
        self.op(
            "av_matmul", "eager_attention_forward", "AV 加权汇合", "matmul+transpose",
            ["attn_probs", "kv_repeated"], ["attn_head_out"], [], ["Nh", "Dh"],
            [src("740-741", "attn_output=matmul(attn_weights, value_states)；"
                  "transpose(1,2).contiguous()")],
            "概率输入是去 sink 后的 attn_probs（L_kv+N_comp 轴与 V 的 KV 轴闭合）；"
            "V 与 K 同一张量（K=V 共享 KV）；输出经 transpose(1,2) 为 [B,T,Nh,Dh]",
        )

        # --- 逆 RoPE 与分组输出投影 ---
        self.tensor(
            "attn_rope_inv", "attn_output（逆 RoPE 后）", ["B", "T", "Nh", "Dh"], ["Nh", "Dh", "Rk"],
            "activation",
            "conjugate 旋转（-sin）抵消 V 上的 RoPE（:859-865）；transpose 配对为 layout "
            "修正——apply 期望 [B,S,H,D]，attention 输出转置进去、旋转后转置回来，"
            "前后都是 [B,T,Nh,Dh]",
        )
        self.tensor(
            "attn_grouped", "attn_output.reshape(B,T,G,-1)", ["B", "T", "G", "Nh×Dh÷G"],
            ["Nh", "Dh", "G"], "activation",
            "每组 Nh/G=8 头、Dh=512 → 组内 4096 维（:867 reshape）",
        )
        self.tensor(
            "o_a_weight", "o_a_proj.weight（GroupedLinear）", ["G×Ro", "Nh×Dh÷G"], ["G", "Ro"],
            "weight",
            "DeepseekV4GroupedLinear(Nh*Dh/G, G*Ro, G)（:789-791）；forward 按 "
            "weight.view(G,-1,Dh_g).transpose(1,2) 分组 bmm（:323-329）",
        )
        self.tensor(
            "o_a_out", "o_a_proj 输出", ["B", "T", "G×Ro"], ["G", "Ro"], "activation",
            "flatten(2) 后接 o_b_proj（:868）",
        )
        self.tensor(
            "o_b_weight", "o_b_proj.weight", ["H", "G×Ro"], ["H", "G", "Ro"], "weight",
            "Linear(G*Ro, H, bias=False)",
        )
        self.tensor(
            "attn_block_output", "Attention 输出（o_b_proj）", ["B", "T", "H"], ["H"],
            "activation",
            "o_b_proj 输出（:869），即残差段的 attn_output（进入 attn 段 mHC 混合）",
        )
        self.op(
            "attn_inv_rope", "DeepseekV4Attention", "输出逆 RoPE", "rope-conjugate",
            ["attn_head_out", "cos_q", "sin_q"], ["attn_rope_inv"], [], ["Rk"],
            [src("859-865", "apply_rotary_pos_emb(attn_output.transpose(1,2), cos, -sin)"
                  ".transpose(1,2)")],
            "K=V 时 V 已带 RoPE；查询位置用 -sin 共轭旋转，使输出只依赖相对距离；"
            "transpose 配对为 layout 修正（apply 期望 [B,S,H,D]）",
        )
        self.op(
            "attn_group_reshape", "DeepseekV4Attention", "分组 reshape", "reshape",
            ["attn_rope_inv"], ["attn_grouped"], [], ["Nh", "Dh", "G"],
            [src("867", "grouped = attn_output.reshape(*input_shape, o_groups, -1)")],
            "[B,T,Nh,Dh]→[B,T,G,Nh·Dh/G]；G=o_groups",
        )
        self.op(
            "o_a_proj", "DeepseekV4GroupedLinear", "分组输出下投影", "reshape+bmm",
            ["attn_grouped"], ["o_a_out"], ["o_a_weight"], ["G", "Ro", "Nh", "Dh"],
            [src("319-329", "w=weight.view(G,-1,hidden).transpose(1,2)；"
                  "x=reshape(-1,G,hidden).transpose(0,1)；y=bmm(x,w)；reshape 回")],
            "8 组各 4096→1024（Ro=1024）；块对角分组线性",
        )
        self.op(
            "o_b_proj", "DeepseekV4Attention", "输出混合投影", "linear+flatten",
            ["o_a_out"], ["attn_block_output"], ["o_b_weight"], ["H", "G", "Ro"],
            [src("792", "o_b_proj=Linear(G*Ro, H, bias=False)"),
             src("868-869", "o_a_proj→flatten(2)；o_b_proj(grouped)")],
            "G·Ro=8192 → H=4096；输出直接进入 attn 段残差混合（真实 producer→consumer 连接）",
        )

    def _mhc_residual_block(self, site: str) -> None:
        """post 扩展 + comb.T 残差混合。attn 段输出→ffn 流；ffn 段输出→层输出。"""
        src = self.src
        p = site
        if site == "attn":
            block_out = "attn_block_output"
            residual_in = "hidden_streams_in"
            out_streams = "attn_out_streams"
            post_tid = "attn_mhc_post"
            comb_tid = "attn_mhc_comb"
            self.tensor(
                "attn_out_streams", "hidden_states（attn 段混合输出）",
                ["B", "T", "Nhc", "H"], ["Nhc", "H"], "activation",
                "post 扩展 + comb.T 混合后的流，即 ffn_hc 的输入",
            )
        else:
            block_out = "mlp_output"
            residual_in = "attn_out_streams"
            out_streams = "hidden_streams_out"
            post_tid = "ffn_mhc_post"
            comb_tid = "ffn_mhc_comb"
        self.tensor(
            f"{p}_expanded", "post.unsqueeze(-1)·output.unsqueeze(-2)",
            ["B", "T", "Nhc", "H"], ["Nhc", "H"], "activation",
            "块输出按 post 加权的流扩展",
        )
        self.tensor(
            f"{p}_combT", "comb.transpose(-1,-2)", ["B", "T", "Nhc", "Nhc"], ["Nhc"],
            "activation",
            "comb 显式转置结果（rank 与 comb 相同 [B,T,Nhc,Nhc]），供残差 matmul 消费",
        )
        self.tensor(
            f"{p}_mixed", "comb.T @ residual", ["B", "T", "Nhc", "H"], ["Nhc", "H"],
            "activation",
            "matmul(combᵀ, residual_streams)（:1178/:1184）；"
            "sum_j comb[j,k]·residual[j,d]——对第一个流轴求和；"
            "双随机非对称，方向由源码确定",
        )
        self.op(
            f"{p}_residual_combT", "DeepseekV4DecoderLayer.forward", "comb 转置",
            "transpose", [comb_tid], [f"{p}_combT"], [], ["Nhc"],
            [src("1178/1184", "comb.to(dtype).transpose(-1, -2)")],
            "combᵀ 显式转置算子（先转置再混合，阅读顺序与源码一致）；"
            "Sinkhorn 后双随机但非对称，转置不可省",
        )
        self.op(
            f"{p}_residual_expand", "DeepseekV4DecoderLayer.forward",
            ("attn 残差：post 扩展" if site == "attn" else "ffn 残差：post 扩展"),
            "broadcast-mul",
            [post_tid, block_out], [f"{p}_expanded"], [], ["Nhc", "H"],
            [src("1177", "post.unsqueeze(-1) * attn_output.unsqueeze(-2)")
             if site == "attn" else
             src("1183", "post.unsqueeze(-1) * mlp_output.unsqueeze(-2)")],
            "块输出扩展为 Nhc 路流",
        )
        self.op(
            f"{p}_residual_mix", "DeepseekV4DecoderLayer.forward",
            ("attn 残差：comb.T 混合+相加" if site == "attn" else "ffn 残差：comb.T 混合+相加"),
            "matmul+add",
            [f"{p}_combT", residual_in, f"{p}_expanded"], [out_streams], [],
            ["Nhc", "H"],
            [src("1177-1179/1183-1185",
                 "post.unsqueeze(-1)*output.unsqueeze(-2) + matmul(comb.transpose(-1,-2), hidden_states)")],
            "Mixed = post⊙Out + In@combᵀ；fp32 权重先 .to(dtype)（:1174-1176）"
            + ("；输出即本层最终流" if site == "ffn" else "；输出进入 ffn_hc"),
        )

    def _moe_block(self) -> None:
        """MoE：TopK 路由→专家分派/合并 gate_up→clamp/SiLU/mul→down→加权汇合→共享专家。"""
        src = self.src
        srcf = self.srcf
        v = self.values
        prefill = self.is_prefill

        self.tensor(
            "ffn_flat", "flat=hidden.view(-1, H)", ["B×T", "H"], ["H"], "activation",
            "SparseMoeBlock.forward view（:1128）；专家模板/路由的 token 输入源",
        )
        self.op(
            "moe_flatten", "DeepseekV4SparseMoeBlock", "MoE 输入展平", "view",
            ["ffn_normed"], ["ffn_flat"], [], ["H"],
            [src("1125-1128", "residual=hidden_states；batch, seq_len, hidden_dim=shape；"
                  "flat=hidden_states.view(-1, hidden_dim)")],
            "residual 同一张量供共享专家（:1127/:1134）；展开为 token 行 [B×T, H]",
        )
        self.tensor(
            "router_weight", "router.weight", ["E", "H"], ["E", "H"], "weight",
            "TopKRouter.weight（E×H，:1071）",
        )
        self.tensor(
            "router_bias", "e_score_correction_bias", ["E"], ["E"], "weight",
            "TopKRouter.e_score_correction_bias（buffer，init zeros；:1074）——只参与选择，"
            "不参与权重值（gather 自 scores）",
        )
        self.tensor(
            "router_logits", "logits", ["B×T", "E"], ["E", "H"], "activation",
            "F.linear(flat, weight)（:1078）",
        )
        self.tensor(
            "router_scores", "scores（sqrtsoftplus）", ["B×T", "E"], ["E"], "activation",
            "score_fn(logits)=sqrt(softplus(logits))（activations.py:217-221）",
        )
        self.tensor(
            "topk_indices", "indices", ["B×T", "K"], ["K", "E"], "index",
            "topk(scores+bias, K, sorted=False).indices（:1080）",
        )
        self.tensor(
            "topk_weights", "weights×Scl", ["B×T", "K"], ["K", "E", "Scl"], "activation",
            "scores.gather(indices)→/(sum+1e-20)（:1081-1082）→×routed_scaling_factor"
            "（:1083）；归一无视 bias（gather 自原 scores）",
        )
        self.op(
            "moe_router", "DeepseekV4TopKRouter", "TopK 路由打分与选择",
            "linear+softplus+sqrt+topk+gather+norm",
            ["ffn_flat", "router_bias"], ["router_logits", "topk_indices", "topk_weights"],
            ["router_weight"], ["E", "K", "Scl", "H"],
            [src("1065-1083", "flat→linear→score_fn→topk(scores+bias)→gather→"
                 "归一+1e-20→×routed_scaling_factor"),
             srcf("activations.py", "217-221", "SqrtSoftplusActivation=sqrt(softplus(x))")],
            "scoring_func='sqrtsoftplus'；e_score_correction_bias 参与选择不参与权重；"
            "命中专家集合运行时决定（保留符号）",
        )
        self.tensor(
            "expert_mask", "one_hot mask", ["E+1", "K", "B×T"], ["E", "K"], "index",
            "F.one_hot(top_k_index, E+1).permute(2,1,0)（:1043）；+1 维为哨兵位",
        )
        self.tensor(
            "hit_experts", "hit（命中专家）", ["N_hit", "1"], ["E", "K"], "index",
            "mask.sum(dim=(-1,-2))>0 的 nonzero（:1044）；N_hit 运行时数据相关"
            "（静态符号表达，不填值）",
        )
        self.tensor(
            "expert_tokens", "选中 token 行（模板输入）", ["N_e", "H"], ["H"], "activation",
            "hidden_states[token_idx]（:1050）——当前命中专家 e 的 N_e 行真实 token 输入；"
            "N_e 运行时数据相关（保留符号）",
        )
        self.tensor(
            "expert_token_idx", "token_idx", ["N_e"], ["K"], "index",
            "where(mask[expert_idx])[1]（:1049）——选中 token 在 flat [B×T] 的行索引；"
            "供 index_add 汇合寻址",
        )
        self.tensor(
            "expert_topk_pos", "top_k_pos", ["N_e"], ["K"], "index",
            "where(mask[expert_idx])[0]（:1049）——选中 token 在 K 个路由槽中的位置；"
            "供路由权重 gather 寻址",
        )
        self.tensor(
            "expert_gate_up", "experts.gate_up_proj", ["E", "2Es", "H"], ["E", "Es", "H"],
            "weight",
            "合并 gate_up 权重 3D 张量（:1033）",
        )
        self.tensor(
            "expert_down", "experts.down_proj", ["E", "H", "Es"], ["E", "H", "Es"], "weight",
            "down 权重 3D 张量（:1034）",
        )
        self.tensor(
            "expert_gu_out", "F.linear(x, gate_up[e])", ["N_e", "2Es"], ["Es"], "activation",
            "选中 token 行的合并 gate_up 投影（:1050）",
        )
        self.tensor(
            "expert_gate_c", "gate（clamp 后）", ["N_e", "Es"], ["Es", "Lim"], "activation",
            "chunk(2) 前半→gate.clamp(max=Lim)（:1059-1060）",
        )
        self.tensor(
            "expert_up_c", "up（clamp 后）", ["N_e", "Es"], ["Es", "Lim"], "activation",
            "chunk(2) 后半→up.clamp(min=-Lim, max=Lim)（:1061）",
        )
        self.tensor(
            "expert_gated", "SiLU(gate)·up", ["N_e", "Es"], ["Es", "Lim"], "activation",
            "act_fn(gate)·up（:1062，_apply_gate）",
        )
        self.tensor(
            "expert_down_raw", "down 输出（未加权）", ["N_e", "H"], ["Es", "H"], "activation",
            "F.linear(gated, down_proj[expert_idx])（:1051 前半）",
        )
        self.tensor(
            "expert_down_out", "down 输出×路由权重", ["N_e", "H"], ["Es", "H", "K", "Scl"],
            "activation",
            "expert_down_raw·top_k_weights[token_idx, top_k_pos, None]（:1051 后半）——"
            "路由权重在真实位置（top_k_pos 寻址）相乘",
        )
        self.tensor(
            "routed_out", "routed（加权汇合，flat）", ["B×T", "H"], ["H"], "activation",
            "final=zeros_like(flat)（:1041）起步零张量，index_add_(0, token_idx, current)"
            "（:1052）；仍为 flat 行视图",
        )
        self.tensor(
            "routed_view", "routed.view(B,T,H)", ["B", "T", "H"], ["H"], "activation",
            "experts 输出还原 3D 视图（:1133），与 shared_out 同形后相加（:1134）",
        )
        self.op(
            "moe_dispatch", "DeepseekV4Experts", "专家分派（one_hot+命中循环）",
            "one_hot+nonzero+where",
            ["topk_indices", "ffn_flat"], ["expert_mask", "hit_experts"], [], ["E", "K"],
            [src("1038-1049", "zeros_like→no_grad→one_hot.permute→greater.sum.nonzero→"
                  "for expert_idx in hit（哨兵 E 跳过）→where(mask[e]) 取 token/pos")],
            "数据相关分派：N_hit/N_e 为运行时值（保留符号不填数）；"
            "同构专家共用下面一份模板链，不逐专家复制",
        )
        self.op(
            "moe_select_tokens", "DeepseekV4Experts（专家模板）", "选 token 行与索引",
            "where+index",
            ["ffn_flat", "expert_mask", "hit_experts"],
            ["expert_tokens", "expert_token_idx", "expert_topk_pos"], [], ["K", "H"],
            [src("1049", "top_k_pos, token_idx = torch.where(mask[expert_idx])；"
                  "hidden_states[token_idx] 行选取")],
            "每命中专家 e 一组（token_idx, top_k_pos）；模板按该数据相关索引读取真实 token 行，"
            "不复制 E 份",
        )
        self.tensor(
            "expert_gate_up_e", "gate_up_proj[expert_idx=e]", ["2Es", "H"], ["Es", "H"],
            "weight",
            "当前命中专家 e 的二维切片（:1050 gate_up_proj[expert_idx]），由整库 "
            "expert_gate_up [E,2Es,H] 按命中索引选择",
        )
        self.tensor(
            "expert_down_e", "down_proj[expert_idx=e]", ["H", "Es"], ["H", "Es"], "weight",
            "当前命中专家 e 的二维切片（:1051 down_proj[expert_idx]），由整库 "
            "expert_down [E,H,Es] 按命中索引选择",
        )
        self.op(
            "moe_expert_slice", "DeepseekV4Experts（专家模板）", "当前专家权重切片选择",
            "index",
            ["expert_gate_up", "expert_down", "hit_experts"],
            ["expert_gate_up_e", "expert_down_e"], [], ["E", "Es", "H"],
            [src("1045-1051", "for expert_idx in hit: … F.linear(…, self.gate_up_proj[expert_idx])、"
                  "F.linear(…, self.down_proj[expert_idx])")],
            "e=hit_experts 当前项（命中专家循环）；每命中专家执行一次本选择与后续模板行，"
            "模板仍只一份、不复制 E 份；两个 Linear 真实消费此二维切片而非整库",
        )
        self.op(
            "moe_expert_gu", "DeepseekV4Experts（专家模板）", "合并 gate_up 投影", "linear",
            ["expert_tokens"], ["expert_gu_out"], ["expert_gate_up_e"], ["Es", "H"],
            [src("1050", "current = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])")],
            "一次 Linear 出 2·Es（gate|up 并存）；权重为当前专家二维切片 gate_up[e] [2Es,H]",
        )
        self.op(
            "moe_expert_clamp", "DeepseekV4Experts（专家模板）", "chunk+双端 clamp", "chunk+clamp",
            ["expert_gu_out"], ["expert_gate_c", "expert_up_c"], [], ["Es", "Lim"],
            [src("1058-1061", "gate, up = gate_up.chunk(2, dim=-1)；"
                  "gate.clamp(max=Lim)；up.clamp(min=-Lim, max=Lim)")],
            "同类向量运算（chunk 与两处 clamp）合并为一行；上界/双边 clamp 语义不同如注",
        )
        self.op(
            "moe_expert_act", "DeepseekV4Experts（专家模板）", "SiLU·mul", "silu+mul",
            ["expert_gate_c", "expert_up_c"], ["expert_gated"], [], ["Es", "Lim"],
            [src("1062", "return self.act_fn(gate) * up")],
            "act_fn=ACT2FN['silu']（:1035）",
        )
        self.op(
            "moe_expert_down", "DeepseekV4Experts（专家模板）", "down 投影", "linear",
            ["expert_gated"], ["expert_down_raw"], ["expert_down_e"], ["Es", "H"],
            [src("1051", "F.linear(current, self.down_proj[expert_idx])")],
            "Es→H 线性；权重为当前专家二维切片 down[e] [H,Es]",
        )
        self.op(
            "moe_expert_weight", "DeepseekV4Experts（专家模板）", "路由权重加权", "gather+mul",
            ["expert_down_raw", "topk_weights", "expert_token_idx", "expert_topk_pos"],
            ["expert_down_out"], [],
            ["K", "Scl", "H"],
            [src("1051", "* top_k_weights[token_idx, top_k_pos, None]")],
            "top_k_weights 按 (token_idx 行, top_k_pos 列) 双索引 gather 出 [N_e,1] 后乘"
            "（路由权重只乘一次）；token_idx 与 top_k_pos 均为真实寻址输入",
        )
        self.op(
            "moe_combine", "DeepseekV4Experts", "专家输出 index_add 汇合", "index_add",
            ["expert_down_out", "expert_token_idx"], ["routed_out"], [], ["H", "K"],
            [src("1041", "final = torch.zeros_like(hidden_states)"),
             src("1052", "final.index_add_(0, token_idx, current.to(final.dtype))")],
            "零张量起步按 token 行索引累加各命中专家贡献（token_idx 为真实寻址输入）；"
            "topk_weights 已在模板加权步消费，此处不再重复",
        )
        self.op(
            "moe_routed_view", "DeepseekV4SparseMoeBlock", "routed 还原 3D 视图", "view",
            ["routed_out"], ["routed_view"], [], ["H"],
            [src("1133", "routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)")],
            "[B×T,H]→[B,T,H]；与 shared_out 同形后相加",
        )
        # 共享专家（DeepseekV4MLP）
        self.tensor(
            "shared_gate", "shared.gate_proj(x).clamp(max=Lim)", ["B", "T", "Es"],
            ["Es", "Lim", "H"], "activation",
            "gate 投影+clamp（:1019）",
        )
        self.tensor(
            "shared_up", "shared.up_proj(x).clamp(±Lim)", ["B", "T", "Es"], ["Es", "Lim", "H"],
            "activation", "up 投影+clamp（:1020）",
        )
        self.tensor(
            "shared_gate_weight", "shared_experts.gate_proj.weight", ["Es", "H"], ["Es", "H"],
            "weight", "Linear(H, Es)（intermediate_size=moe_intermediate_size，:1011-1012）",
        )
        self.tensor(
            "shared_up_weight", "shared_experts.up_proj.weight", ["Es", "H"], ["Es", "H"],
            "weight", "Linear(H, Es)",
        )
        self.tensor(
            "shared_down_weight", "shared_experts.down_proj.weight", ["H", "Es"], ["H", "Es"],
            "weight", "Linear(Es, H)",
        )
        self.tensor(
            "shared_act", "SiLU(gate)·up", ["B", "T", "Es"], ["Es", "Lim"], "activation",
            "act_fn(gate)·up（:1021）",
        )
        self.tensor(
            "shared_out", "shared down 输出", ["B", "T", "H"], ["H", "Es"], "activation",
            "down_proj(act·up)（:1021）",
        )
        self.tensor(
            "mlp_output", "MoE 输出（routed+shared）", ["B", "T", "H"], ["H"], "activation",
            "routed + shared_experts(residual)（:1134）；进入 ffn 残差段",
        )
        self.op(
            "shared_gate_up", "DeepseekV4MLP", "共享专家 gate/up 投影+clamp", "linear+clamp×2",
            ["ffn_normed"], ["shared_gate", "shared_up"],
            ["shared_gate_weight", "shared_up_weight"], ["Es", "Lim", "H"],
            [src("1006-1021", "gate_proj/up_proj Linear(H,Es)；clamp(max=Lim)/clamp(min=-Lim,max=Lim)")],
            "residual=hidden_states（MoE 输入，:1127）；n_shared_experts=1 个实例",
        )
        self.op(
            "shared_act_down", "DeepseekV4MLP", "共享专家激活+down", "silu+mul+linear",
            ["shared_gate", "shared_up"], ["shared_act", "shared_out"],
            ["shared_down_weight"], ["Es", "Lim", "H"],
            [src("1015/1021", "act_fn=ACT2FN['silu']；down_proj(act_fn(gate)·up)")],
            "标准 SwiGLU 形（无 bias：mlp_bias=False）",
        )
        self.op(
            "moe_merge", "DeepseekV4SparseMoeBlock", "路由+共享汇合", "add",
            ["routed_view", "shared_out"], ["mlp_output"], [], ["H"],
            [src("1133-1134", "routed=experts(...).view(batch, seq_len, hidden_dim)；"
                  "return routed + shared_experts(residual)")],
            "TopK 路由层无 input_ids 查表（is_hash=False 分支，:1131-1132）；"
            "routed_view 由 moe_routed_view 显式还原 3D 后与 shared_out 相加",
        )

    # ------------------------------------------------------------------
    # 阶段注记
    # ------------------------------------------------------------------

    def _build_cache_notes(self) -> None:
        prefill = self.is_prefill
        if prefill:
            self.cache_notes = [
                "prefill 缓存起步为空：DynamicCache(config) 按层型分发建层（cache_utils.py:1808-1896），"
                "CSA 层为 DeepseekV4CSACache（modeling:247-297）；update 首调用惰性初始化，keys 为 0 长度 tensor。",
                "主 KV：update 返回 full=cat(keys, kv_new)（modeling:198）——prefill full=T 个新 KV；"
                "持久保存的是 full 的最后 W-1=127 个位置（:199），本次可读与持久长度不同。",
                "压缩器窗口缓冲 buffer_kv/buffer_gate['compressor']：起步 None；本调用取 M 对齐前缀闭窗，"
                "余量（T%M 个）留作下调用 buffer（:216-224）。",
                "压缩条目 compressed_kv['compressor']：起步 None；本调用新增 N_win=T//M 条"
                "（T≥M 时），entry_count 由 0 增至 N_win（:226-237）。",
                "CSA overlap：首调用 overlap_kv/gate=None→new_kv 窗 0 前半为零值/-inf gate"
                "（softmax 权重 0，:647-666 注释）；调用后持久化最后满窗 Ca 尾（:278-292）。",
                "Indexer 系列同构：'indexer' 键独立 buffer/compressed/entry_count/overlap（:268-276）。",
                "position_ids=arange(T)+P（modeling:1335-1337；prefill P=0）；"
                "mask 尺寸 get_mask_sizes(T)（cache_utils:276-286）。",
            ]
        else:
            self.cache_notes = [
                "decode 缓存起步已含历史：keys 为最近 min(P, W-1) 个位置尾部（历史足够长时即 W-1=127）、"
                "cumulative_length=P、entry_count=N_hist、buffer 为上次余量、overlap 为上次满窗 Ca 尾——"
                "全部为调用历史的结构前提，内容有效性由历史调用保证，本图不验证数值。",
                "主 KV：full=cat(keys 尾部, 新 kv)（modeling:198）——decode L_kv=min(P,W-1)+1"
                "（P≥W-1 时即 W 个位置视图）；持久 keys_post=min(P+1, W-1) 个位置"
                "（含新 token，最老一个滑出）。",
                "压缩器窗口缓冲：历史余量 buffer（<M 条）+本调用 1 token；仅当合计达 M 条才闭窗"
                "（N_win = floor((buffer_len+1)/M)，即 (P+1)%M==0 对齐触发时），否则只更新 buffer。",
                "压缩条目：N_comp=entry_count 调用后值；本调用最多新增 1 条（数据相关）；"
                "Attention 读到的条目全量随 update_compressor_states 返回（:556-558/:681-683）。",
                "CSA overlap：调用前 overlap 为上次满窗 Ca 尾；闭窗时先读后写"
                "（update_overlap_state 返回 prior 再持久本调用，:278-292；前后状态分离表达）。",
                "Indexer 系列同构：'indexer' 键独立 buffer/compressed/entry_count/overlap"
                "（:268-276），N_win_i 与压缩器 N_win 因历史余量不同而独立。",
                "position_ids=P（单 token，modeling:1335-1337）；"
                "mask get_mask_sizes(1)：is_full（P≥W）时 kv_length=W-1+1=W、"
                "kv_offset=P-W+1（cache_utils:276-286）。",
                "future_threshold：decode 查询 t=P 的因果阈值 (P+1)//M 决定可见压缩条目上界"
                "（modeling:432-439/:574-581）。",
            ]

    def _build_coverage_notes(self) -> None:
        common = [
            "已表达：DecoderLayer 全链（双 mHC 权重生成/流坍缩/残差混合、双 RMSNorm）、"
            "Attention 低秩 Q/共享 KV/部分 RoPE（interleaved）/缓存更新/CSA 压缩器"
            "（窗口缓冲/Ca-Cb 重叠/softmax 汇聚/条目 RoPE）/Lightning Indexer"
            "（独立压缩/scorer/topk/-1 哨兵）/mask 扩展/eager QK→scale→mask→sink→softmax→AV/"
            "逆 RoPE/分组输出投影、TopK MoE（sqrtsoftplus 路由/one_hot 分派/合并 gate_up/"
            "clamp/SiLU·mul/down/加权 index_add/共享专家 SwiGLU）。",
            "comb 转置方向、压缩条目确定性 RoPE 位置、overlap 状态语义、sink 去除、"
            "bias 只影响选择不影响权重值等关键语义均按源码核对（source_refs 逐算子给出）。",
            "未覆盖（明确列出，不宣称已分析）：整网 Embedding/43 层循环/HyperHead/Final Norm/"
            "LM head/logits（里程碑三）；SWA/Hash 变体层（里程碑二）；"
            "量化执行路径（FP8/FP4 kernel、设备后端、_keep_in_fp32_modules 的加载侧行为）；"
            "训练路径（dropout>0、aux loss、梯度）；生成循环/采样；"
            "torch.compile/_is_stateful 回滚语义；数值等价与权重内容。",
            "本样板所有算子 analysis_status=analyzed（无 internal_unanalyzed 黑盒）；"
            "黑盒机制保留在接口上供后续使用。",
            "数据相关量（N_win/N_comp/N_kv/L_mask/命中专家数/每专家 token 数）保留符号与派生式，"
            "不填运行时假设值。",
        ]
        if self.is_prefill:
            self.coverage_notes = [
                "prefill 阶段覆盖：T 个新 token、空缓存起步的完整 Decoder 路径"
                "（mHC→norm→Attention（含压缩器/Indexer/缓存）→残差→mHC→norm→MoE→残差）。",
                *common,
            ]
        else:
            self.coverage_notes = [
                "decode 阶段覆盖：T=1、历史缓存前提的完整 Decoder 路径（与 prefill 同构模板；"
                "差异在缓存起步状态、窗口触发条件、mask 尺寸与因果阈值）。",
                *common,
            ]
