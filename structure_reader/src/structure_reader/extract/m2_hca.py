"""HCA（heavily_compressed_attention，代表层 3）Block 结构提取。

只实现 HCA+TopK：复用公共 mHC/Norm/专家/残差；HCA 压缩器按 modeling:359-440
真实构造（kv/gate 各 head_dim、非重叠 compress_rate=128、position_bias、
无 Indexer/overlap）。与 CSA 共用同一套 Block 结构基类与场景参数化。
"""
from __future__ import annotations

from structure_reader.extract.deepseek_v4_hf import BlockStructureBuilder
from structure_reader.rules.deepseek_v4_hf import HfModelValues

_HCA_NEEDED = {
    "B", "seqlen", "kvcachelen", "H", "Nh", "Nkv", "Dh", "Rq", "Rk", "G", "Ro", "M",
    "Nhc", "Sit", "Eps", "EpsN", "E", "K", "Es", "Lim", "Scl", "W",
    "Lkv", "Lmask", "Ncomp", "Nwin", "Tbuf", "Tbufp", "Tcc", "ThetaC", "Yf",
}


class HCABlockBuilder(BlockStructureBuilder):
    def _build_parameters(self) -> None:
        super()._build_parameters()
        self.parameters = [p for p in self.parameters if p["id"] in _HCA_NEEDED]
        for p in self.parameters:
            if p["id"] == "M":
                p["value"] = self.values.compress_rate_hca
                p["meaning"] = "HCA 非重叠压缩窗口宽（每 compress_rate 个 token 聚合 1 条）"
                p["expression"] = "default_compress_rates['heavily_compressed_attention']=128"
                p["note"] = ("值来自 configuration default_compress_rates 的 HCA 项；"
                             "compress_ratios=128 只是层型选择，不是 scalar 覆盖")
                p["source_refs"] = [
                    self.cfg("compress_ratios", "128 → heavily_compressed_attention"),
                    self.src("359-440", "HCACompressor：非重叠窗口 compress_rate"),
                ]

    # ---- 边界与 HCA 专用缓存 ----
    def _build_boundary_and_cache(self) -> None:
        src, cfg = self.src, self.cfg
        self.tensor("input_ids", "input_ids", ["B", "seqlen"], ["B", "seqlen"], "boundary",
                    "输入 token id；HCA+TopK 层不消费（Hash 层才查 tid2eid）")
        self.tensor("position_ids", "position_ids", ["1", "seqlen"], ["seqlen"], "boundary",
                    "本调用绝对位置 [1,seqlen]；供 compress RoPE、mask 与 block_bias 阈值")
        self.tensor("hidden_streams_in", "hidden_streams（层输入）", ["B", "seqlen", "Nhc", "H"],
                    ["Nhc", "H"], "boundary", "上一 decoder 层输出的 Nhc 路流")
        self.tensor("hidden_streams_out", "hidden_streams（层输出）", ["B", "seqlen", "Nhc", "H"],
                    ["Nhc", "H"], "boundary", "本层最终输出")
        self.tensor("cache_keys", "DeepseekV4HCACache.keys（主 KV 持久，调用前）",
                    ["B", "1", "min(kvcachelen,W-1)", "Dh"], ["Lkv", "Dh", "W"], "cache",
                    "主 KV 滑动持久态；空缓存 Prefill 起步为 0 长度，"
                    "非零历史 Decode 起步为最近 min(kvcachelen,W-1) 个位置")

        self.tensor("cache_keys_post", "DeepseekV4HCACache.keys（调用后）",
                    ["B", "1", "min(kvcachelen+seqlen,W-1)", "Dh"], ["Lkv", "Dh", "W"], "cache",
                    "update 后 keys 只保留最后 W-1 个位置（min(kvcachelen+seqlen,W-1)）")
        self.tensor("cache_values_post", "DeepseekV4HCACache.values（调用后，keys 别名）",
                    ["B", "1", "min(kvcachelen+seqlen,W-1)", "Dh"], ["Lkv", "Dh", "W"], "cache",
                    "self.values = self.keys（modeling:193/197/201-202）：同一存储，别名不新增")
        self.tensor("cache_cumlen", "cumulative_length（主 KV 累计长度）", [], ["W"], "cache",
                    "已累计 token 数（含本调用前）：决定 mask kv_offset/kv_length")
        self.tensor("cache_buffer_c_kv", "buffer_kv['compressor']（HCA，调用前）",
                    ["B", "Tbuf", "Dh"], ["M", "Dh"], "cache",
                    "HCA 窗口缓冲（head_dim，不是 CSA 的 2·head_dim）")
        self.tensor("cache_buffer_c_gate", "buffer_gate['compressor']（HCA，调用前）",
                    ["B", "Tbuf", "Dh"], ["M", "Dh"], "cache",
                    "与 buffer_kv 配对的 gate 缓冲（head_dim）")
        self.tensor("cache_entry_count_c", "entry_count['compressor']（调用前）", [], ["M"], "cache",
                    "HCA 已发射条数；first_window_position=entry_count*compress_rate")
        self.tensor("cache_compressed_c", "compressed_kv['compressor']（调用前）",
                    ["B", "Tcc", "Dh"], ["M", "Dh"], "cache",
                    "HCA 已发射压缩条目（RoPE 后）；无 Indexer 条目")
        self.tensor("cache_buffer_c_kv_post", "buffer_kv['compressor']（调用后余量）",
                    ["B", "Tbuf'", "Dh"], ["M", "Dh"], "cache",
                    "merge 后取 compress_rate 对齐前缀，余量 (Tbuf+T)%M 写回")
        self.tensor("cache_buffer_c_gate_post", "buffer_gate['compressor']（调用后余量）",
                    ["B", "Tbuf'", "Dh"], ["M", "Dh"], "cache", "配对 gate 余量")
        self.tensor("cache_compressed_c_post", "compressed_kv['compressor']（调用后全量）",
                    ["B", "Ncomp", "Dh"], ["M", "Dh"], "cache",
                    "append 本调用新条目；num_full_windows=0 时原样返回此前全量历史")
        self.tensor("cache_entry_count_c_post", "entry_count['compressor']（调用后）", [], ["M"],
                    "cache", "调用后 = 调用前 + num_full_windows")

    def _build_derivations(self) -> None:
        """HCA 专用派生量：非重叠 compress_rate=128 窗口，无 Indexer/overlap。"""
        from structure_reader.extract.deepseek_v4_hf import (
            _EXPR_LKV, _EXPR_NCOMP, _EXPR_NWIN, _EXPR_PERSIST, _EXPR_POS_FIRST,
            _EXPR_TBUF, _EXPR_TBUF_POST,
        )
        common_note = ("按固定配置 + 「空态起步、连续正常执行」前提推导；"
                       "不覆盖裁剪/回滚后的自定义快照")
        self.derivation("L_kv", "本次可读主 KV 长度", _EXPR_LKV,
                        "持久尾部（最多 W-1）+ 本调用新 token；物理保留长度不等于 kvcachelen。"
                        + common_note)
        self.derivation("L_mask", "模型级 mask KV 长度", _EXPR_LKV,
                        "与 L_kv 同式。" + common_note)
        self.derivation("L_kv_post", "调用后持久主 KV 长度", _EXPR_PERSIST,
                        "update 后只持久 full 的最后 W-1 个位置。" + common_note)
        self.derivation("Tbuf", "调用前窗口缓冲余量", _EXPR_TBUF,
                        "连续执行时 buffer 余量即 kvcachelen % M（M=compress_rate=128）。" + common_note)
        self.derivation("Tbuf_post", "调用后窗口缓冲余量", _EXPR_TBUF_POST,
                        "store_compression_weights 写回 (Tbuf+seqlen)%M 条。" + common_note)
        self.derivation("N_win", "本调用新闭窗数", _EXPR_NWIN,
                        "非重叠窗口：闭窗数 = (Tbuf+seqlen)//M；为 0 时不新增条目、buffer 只写回。"
                        + common_note)
        self.derivation("N_comp", "调用后压缩条目总数", _EXPR_NCOMP,
                        "调用前条目（kvcachelen//M）+ 本调用新闭窗；update_compressor_states "
                        "返回历史全量。" + common_note)
        self.derivation("first_window_position", "本调用首窗绝对位置", _EXPR_POS_FIRST,
                        "entry_count（调用前）×M；条目位置 = i·M + first_window_position。"
                        + common_note)
        self.derivation("future_threshold", "末查询的因果阈值",
                        "(kvcachelen + seqlen) // M",
                        "查询 t 可见 w < (t+1)//M；w 等于阈值（w = (t+1)//M）即屏蔽"
                        "（含等于，源码 :437-439 block_bias masked_fill(entry>=threshold)）；"
                        "末查询阈值为 (kvcachelen+seqlen)//M。" + common_note)
        self.derivation("first_query_threshold", "本调用首个查询的因果阈值",
                        "(kvcachelen + 1) // M",
                        "首个查询 t=kvcachelen：可见 w < (t+1)//M；w 等于阈值"
                        "（w = (t+1)//M）即屏蔽（含等于，源码 :437-439）。" + common_note)
        self.derivation("block_bias", "HCA block_bias 是否物化",
                        "seqlen != 1 and N_comp > 0",
                        "源码 modeling:432-439：seq_len==1 或 compressed_len==0 时返回 None"
                        "（不物化），否则按因果阈值生成 0/-inf 矩阵。" + common_note)

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "HCA 主 KV 使用 DeepseekV4HCACache（modeling:156）：self.values = self.keys 同一存储；"
            "update 返回 full（min(kvcachelen,W-1)+seqlen）与持久 tail（min(kvcachelen+seqlen,W-1)）；"
            "独立 keys/values 字段只属于 SWA 基类。",
            "compressor 只有 'compressor' 键：buffer_kv/buffer_gate/entry_count/compressed_kv；"
            "无 indexer 键、无 overlap/CaCb。",
            "N_win=0 时仍返回此前全量 compressed_kv（历史条目），entry_count 不变；"
            "非零历史 Decode 只写回 buffer，不新增条目。",
            "block_bias 仅在 seqlen != 1 且调用后条目数 > 0 时物化；seqlen=1 或条目为 0 时为 None。",
        ]

    def _build_coverage_notes(self) -> None:
        self.coverage_notes = [
            "覆盖：HCA 压缩器（非重叠 128 窗口、position_bias、softmax 聚合、条目 RoPE、历史追加）、"
            "共享 KV 缓存、sliding mask+block_bias、eager QK→scale→mask→sink→softmax→AV、"
            "逆 RoPE、分组输出投影、mHC、TopK MoE 与共享专家。",
            "未覆盖：SWA/Hash 路由、HCA 以外变体、量化执行、整网外围与 token→logits 见整网层。",
        ]

    # ---- HCA 注意力链 ----
    def _attention_block(self) -> None:
        src, cfg = self.src, self.cfg
        start = len(self.operators)
        # Q 低秩 + compress RoPE
        self.tensor("q_a", "q_a_proj(hidden)", ["B", "seqlen", "Rq"], ["Rq", "H"], "activation",
                    "低秩 Q 下投影")
        self.tensor("q_weight", "q_a_proj.weight", ["Rq", "H"], ["Rq", "H"], "weight",
                    "Linear(H,Rq,bias=False)")
        self.tensor("q_a_norm_weight", "q_a_norm.weight", ["Rq"], ["Rq"], "weight",
                    "DeepseekV4RMSNorm(q_lora_rank)")
        self.op("q_a_proj", "DeepseekV4Attention.q_a_proj", "q_a_proj", "linear",
                ["attn_normed", "q_weight"], ["q_a"], ["q_weight"], ["Rq", "H"],
                [src("813-814", "self.q_a_proj(hidden_states)")], "Linear 无 bias")
        self.tensor("q_a_normed", "q_a_norm(q_a)", ["B", "seqlen", "Rq"], ["Rq", "EpsN"],
                    "activation", "q_residual（HCA 仅签名接收，实际不再使用）")
        self.op("q_a_norm", "DeepseekV4Attention.q_a_norm", "q_a_norm", "rmsnorm",
                ["q_a", "q_a_norm_weight"], ["q_a_normed"], ["q_a_norm_weight"],
                ["Rq", "EpsN"], [src("813-814", "self.q_a_norm(self.q_a_proj(...))")],
                "HCA 不使用 q_residual（无 Indexer），不画伪边")
        self.tensor("q_b_weight", "q_b_proj.weight", ["Nh×Dh", "Rq"], ["Nh", "Dh", "Rq"],
                    "weight", "Linear(Rq,Nh*Dh,bias=False)")
        self.tensor("q_head", "q.view/transpose", ["B", "Nh", "seqlen", "Dh"], ["Nh", "Dh"],
                    "activation", "q_b_proj→view(B,T,-1,Dh)→transpose(1,2)")
        self.op("q_b_proj", "DeepseekV4Attention.q_b_proj", "q_b_proj（linear+view/transpose）",
                "linear+view+transpose",
                ["q_a_normed", "q_b_weight"], ["q_head"], ["q_b_weight"], ["Nh", "Dh", "Rq"],
                [src("813-814", "q_b_proj(q_residual).view(...).transpose(1,2)")],
                "view/transpose 后 [B,Nh,seqlen,Dh]")
        self.tensor("q_normed", "q_b_norm(q_head)", ["B", "Nh", "seqlen", "Dh"], ["Dh"], "activation",
                    "UnweightedRMSNorm")
        self.op("q_b_norm", "DeepseekV4Attention.q_b_norm", "q_b_norm", "rmsnorm",
                ["q_head"], ["q_normed"], [], ["Dh"], [src("786,815", "UnweightedRMSNorm")],
                "无权重 RMSNorm")
        self.tensor("cos_q", "cos（compress）", ["1", "seqlen", "Rk÷2"], ["Rk", "ThetaC", "Yf"],
                    "activation", "compress YaRN 的 cos")
        self.tensor("sin_q", "sin（compress）", ["1", "seqlen", "Rk÷2"], ["Rk", "ThetaC", "Yf"],
                    "activation", "compress YaRN 的 sin")
        self.op("rope_compress", "DeepseekV4RotaryEmbedding", "compress RoPE cos/sin", "rope",
                ["position_ids"], ["cos_q", "sin_q"], [], ["Rk", "ThetaC", "Yf"],
                [src("148-153", "position_ids·inv_freq→cos/sin（compress）")],
                "HCA 使用 compress YaRN（θ=compress_rope_theta）")
        self.tensor("q_rope", "q（部分 RoPE 后）", ["B", "Nh", "seqlen", "Dh"], ["Nh", "Dh", "Rk"],
                    "activation", "末 Rk 维 interleaved 旋转")
        self.op("q_rope_apply", "apply_rotary_pos_emb", "q partial RoPE", "rope",
                ["q_normed", "cos_q", "sin_q"], ["q_rope"], [], ["Rk", "Dh"],
                [src("339-356", "apply_rotary_pos_emb")], "部分旋转（末 64 维）")

        # 主 Attention KV（与 HCA 压缩器自己的 kv_proj 分离）
        self.tensor("kv_weight_attn", "self.kv_proj.weight（Attention 主 KV）", ["Dh", "H"],
                    ["Dh", "H"], "weight", "Linear(H,head_dim,bias=False)")
        self.tensor("kv_norm_weight", "self.kv_norm.weight", ["Dh"], ["Dh"], "weight",
                    "RMSNorm(head_dim)")
        self.tensor("kv_main", "self.kv_norm(self.kv_proj(hidden))", ["B", "seqlen", "Dh"],
                    ["Dh", "H"], "activation", "主 KV 投影+归一（单 KV 头）")
        self.op("kv_proj", "DeepseekV4Attention.kv_proj", "Attention 主 KV 投影+RMSNorm",
                "linear+rmsnorm",
                ["attn_normed", "kv_weight_attn", "kv_norm_weight"], ["kv_main"],
                ["kv_weight_attn", "kv_norm_weight"], ["Dh", "H", "EpsN"],
                [src("817-818", "self.kv_norm(self.kv_proj(hidden_states))")],
                "Linear + RMSNorm 复合；num_key_value_heads=1 单 KV 头；K=V 同一张量")
        self.tensor("kv_head", "kv.view/transpose", ["B", "1", "seqlen", "Dh"], ["Dh"], "activation",
                    "view(B,T,1,Dh).transpose(1,2)")
        self.op("kv_head_view", "DeepseekV4Attention.kv_head_view", "主 KV view/transpose",
                "view", ["kv_main"], ["kv_head"], [], ["Dh"],
                [src("818", "view(B,T,-1,Dh).transpose(1,2)")], "单 KV 头")
        self.tensor("kv_rope", "kv（部分 RoPE 后，主 KV）", ["B", "1", "seqlen", "Dh"],
                    ["Dh", "Rk"], "activation", "主 KV 末 Rk 维旋转（compress RoPE）")
        self.op("kv_rope_apply", "apply_rotary_pos_emb", "主 KV partial RoPE", "rope",
                ["kv_head", "cos_q", "sin_q"], ["kv_rope"], [], ["Rk", "Dh"],
                [src("339-356", "apply_rotary_pos_emb(kv)")], "与 Q 同一 compress rope")

        # HCA 压缩器自己的 kv/gate：各 head_dim

        self.tensor("c_kv_weight", "compressor.kv_proj.weight", ["Dh", "H"], ["Dh", "H"],
                    "weight", "HCA Linear(H,head_dim)（不是 CSA 的 2·head_dim）")
        self.tensor("c_gate_weight", "compressor.gate_proj.weight", ["Dh", "H"], ["Dh", "H"],
                    "weight", "HCA Linear(H,head_dim)")
        self.tensor("c_kv", "compressor.kv_proj(hidden)", ["B", "seqlen", "Dh"], ["Dh", "H"],
                    "activation", "HCA kv 投影（head_dim）")
        self.tensor("c_gate", "compressor.gate_proj(hidden)", ["B", "seqlen", "Dh"], ["Dh", "H"],
                    "activation", "HCA gate 投影（head_dim）")
        self.op("c_kv_proj", "DeepseekV4HCACompressor.kv_proj", "HCA kv_proj", "linear",
                ["attn_normed", "c_kv_weight"], ["c_kv"], ["c_kv_weight"], ["Dh", "H"],
                [src("359-440", "HCA kv_proj 输出 head_dim；单 KV 头")],
                "Linear(H,head_dim)；num_key_value_heads=1 共享 KV")
        self.op("c_gate_proj", "DeepseekV4HCACompressor.gate_proj", "HCA gate_proj", "linear",
                ["attn_normed", "c_gate_weight"], ["c_gate"], ["c_gate_weight"], ["Dh", "H"],
                [src("359-440", "HCA gate_proj 输出 head_dim")], "Linear(H,head_dim)")
        self.tensor("c_position_bias", "compressor.position_bias", ["M", "Dh"], ["M", "Dh"],
                    "weight", "HCA position_bias[compress_rate, head_dim]")
        self.tensor("c_kv_norm_weight", "compressor.kv_norm.weight", ["Dh"], ["Dh"], "weight",
                    "RMSNorm(head_dim)")
        self.tensor("c_chunk_kv", "chunk_kv（M 对齐前缀）", ["B", "Nwin×M", "Dh"], ["M"], "activation",
                    "cat(buffer_kv, kv) 后取 compress_rate 对齐前缀")
        self.tensor("c_chunk_gate", "chunk_gate（M 对齐前缀）", ["B", "Nwin×M", "Dh"], ["M"], "activation",
                    "与 chunk_kv 配对")

        self.op("c_store_window", "DeepseekV4HCACompressor.store_compression_weights",
                "HCA 缓冲合并/对齐前缀/余量写回", "store_compression_weights",
                ["c_kv", "c_gate", "cache_buffer_c_kv", "cache_buffer_c_gate"],
                ["c_chunk_kv", "c_chunk_gate", "cache_buffer_c_kv_post", "cache_buffer_c_gate_post"],
                [], ["M", "Tbuf"], [src("203-224,359-440", "store_compression_weights")],
                "num_full_windows = (Tbuf+T)//M；余量写回 buffer")
        self.tensor("c_windows_kv", "new_kv（非重叠窗口）", ["B", "Nwin", "M", "Dh"], ["M", "Dh"],
                    "activation", "view(B,Nwin,M,Dh)；无 Ca/Cb 重叠")
        self.tensor("c_windows_gate", "new_gate（非重叠窗口）", ["B", "Nwin", "M", "Dh"], ["M", "Dh"],
                    "activation", "gate + position_bias")
        self.op("c_window_view", "DeepseekV4HCACompressor.window_view", "HCA 窗口视图",
                "window_view", ["c_chunk_kv", "c_chunk_gate", "c_position_bias"],
                ["c_windows_kv", "c_windows_gate"], ["c_position_bias"], ["M", "Dh"],
                [src("359-440", "view(B,Nwin,M,Dh)；gate+position_bias")],
                "非重叠；无 Ca/Cb、无 overlap")
        self.tensor("c_compressed", "compressed（本调用新条目）", ["B", "Nwin", "Dh"], ["M", "Dh"],
                    "activation", "softmax(gate,M)·kv 求和 → RMSNorm")
        self.tensor("c_compressed_rope", "compressed（条目 RoPE 后）", ["B", "Nwin", "Dh"],
                    ["M", "Dh"], "activation", "应用确定性 compress RoPE")
        self.op("c_pool", "DeepseekV4HCACompressor.pool", "HCA 窗口 softmax 汇聚", "pool",
                ["c_windows_kv", "c_windows_gate", "c_kv_norm_weight"], ["c_compressed"],
                ["c_kv_norm_weight"], ["M", "Dh"],
                [src("359-440", "softmax(dim=M)→*kv→sum→kv_norm")],
                "沿 compress_rate 轴 softmax；RMSNorm；无重叠槽")
        self.tensor("first_window_position", "first_window_position", [], ["M"],
                    "activation", "调用前 entry_count*M（store_compression_weights modeling:216 返回）")
        self.tensor("c_positions", "compressed positions", ["B", "Nwin"], ["M", "Nwin"],
                    "activation", "arange(num_full_windows)*M + first_window_position，expand(batch,-1)")
        self.tensor("c_cos", "cos（compress, 条目位置）", ["B", "Nwin", "Rk÷2"], ["Rk", "ThetaC"],
                    "activation", "条目 compress RoPE 的 cos")
        self.tensor("c_sin", "sin（compress, 条目位置）", ["B", "Nwin", "Rk÷2"], ["Rk", "ThetaC"],
                    "activation", "条目 compress RoPE 的 sin")
        self.op("c_rope_gen", "DeepseekV4HCACompressor.rope", "HCA 条目位置/cos/sin",
                "entry_count*M+range+mul+add+expand+rope",
                ["cache_entry_count_c"], ["first_window_position", "c_positions", "c_cos", "c_sin"],
                [], ["M", "Nwin", "Rk", "ThetaC"],
                [src("216,416-420", "first_window_position=entry_count*M；positions=arange(Nwin)*M+first")],
                "位置来自 entry_count 与窗口序号；query position_ids 不参与条目 RoPE")
        self.op("c_rope", "DeepseekV4HCACompressor.rope", "HCA 条目 RoPE", "rope",
                ["c_compressed", "c_cos", "c_sin"], ["c_compressed_rope"], [], ["Rk", "ThetaC"],
                [src("416-420", "apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin)")],
                "确定性位置 i·M+first_window_position（compress RoPE）；N_win=0 时 compressed 为 "
                "new_zeros[batch,0,head_dim]，不做 norm/RoPE")
        self.op("c_update_states", "DeepseekV4HCACache.update_compressor_states",
                "HCA 条目追加/返回历史", "update_compressor_states",
                ["c_compressed_rope", "cache_compressed_c", "cache_entry_count_c"],
                ["cache_compressed_c_post", "cache_entry_count_c_post"], [], ["M"],
                [src("226-237", "update_compressor_states")],
                "append 新条目；num_full_windows=0 时返回此前全量历史；entry_count += N_win")
        self.tensor("block_bias", "block_bias（HCA 压缩条目可见性）", ["B", "1", "seqlen", "Ncomp"],
                    ["M"], "activation",
                    "条件张量：seqlen==1 或调用后条目数==0 时为 None（源码 :432-439），"
                    "否则为 [B,1,seqlen,N_comp] 的 0/-inf；可见 w < (t+1)//M，等于阈值即屏蔽")
        self.op("hca_block_bias", "DeepseekV4HCACompressor.block_bias", "HCA 因果 block bias",
                "block_bias", ["position_ids", "cache_compressed_c_post"], ["block_bias"], [], ["M"],
                [src("432-439", "seq_len==1 or compressed_len==0 → None；"
                      "否则 masked_fill(entry_indices >= causal_threshold, -inf)")],
                "仅在 seqlen!=1 且调用后条目数>0 时物化；seqlen==1 或条目为 0 时返回 None；"
                "查询 t 只可见 w < (t+1)//M，w 等于阈值即屏蔽（含等于）；值为 0/-inf，不做 TopK")
        self.op("mask_model_build", "create_sliding_window_causal_mask",
                "模型级 sliding causal mask", "mask_build",
                ["position_ids", "cache_cumlen"], ["mask_model"], [], ["Lmask", "W"],
                [self.srcf("masking_utils.py", "1078-1162", "create_sliding_window_causal_mask"),
                 self.srcf("cache_utils.py", "276-286", "get_mask_sizes"),
                 src("1341-1348", "调用 mask 构造")],
                "mask 来自 masking_utils.py:1078-1162；尺寸由 cache_utils.py:276-286 提供")
        self.op("c_mask_extend", "DeepseekV4Attention.forward", "mask 与 block_bias 拼接/零扩展",
                "concat+pad", ["kv_all", "mask_model", "block_bias"], ["mask_ext"], [],
                ["Lmask", "M"],
                [src("837-841", "kv.shape[2]>mask.shape[-1]：block_bias 则 cat，否则 F.pad 0")],
                "block_bias 为 None（seqlen==1 或条目为 0）时 F.pad 零扩展；非 None 时 cat；"
                "依赖本次 kv_all 长度")

        # 主 KV 缓存（keys/values 独立持久字段）
        self.tensor("cache_cumlen_post", "cumulative_length（调用后）", [], ["W"], "cache",
                    "调用后 = kvcachelen + seqlen；下次 mask 起步")
        self.op("cache_update", "DeepseekV4HCACache.update", "主 KV 缓存更新", "cache_update",
                ["kv_rope", "cache_keys", "cache_cumlen"],
                ["kv_full", "cache_keys_post", "cache_values_post", "cache_cumlen_post"],
                [], ["W", "Lkv"], [src("193,197,201-202", "update：self.values = self.keys；keys 持久 tail")],
                "HCA：self.values = self.keys（同一存储，modeling:193/197/201-202），"
                "不新增独立 values 字段；full 与持久 tail 分开。独立 keys/values 仅 SWA 基类")
        self.tensor("kv_full", "cache.update 返回 full", ["B", "1", "Lkv", "Dh"], ["Lkv", "Dh", "W"],
                    "cache", "本次可读 full=cat(keys,kv_rope)")
        self.op("kv_concat", "DeepseekV4Attention.forward", "KV 轴拼接压缩条目", "concat",
                ["kv_full", "cache_compressed_c_post"], ["kv_all"], [], ["Lkv", "Ncomp"],
                [src("822-829", "cat([kv, compressed_kv], dim=2)")],
                "HCA 条目 unsqueeze 到单 KV 头后再拼接")
        self.tensor("kv_all", "key_states / value_states", ["B", "1", "Lkv+Ncomp", "Dh"],
                    ["Lkv", "Ncomp", "Dh"], "activation", "K=V 同输入")
        self.tensor("kv_repeated", "repeat_kv(kv)", ["B", "Nh", "Lkv+Ncomp", "Dh"],
                    ["Nh", "Lkv", "Dh"], "activation", "单 KV 头广播到 Nh 头")
        self.op("repeat_kv", "repeat_kv", "KV 头广播", "repeat_kv", ["kv_all"], ["kv_repeated"], [],
                ["Nh"], [src("702-711", "repeat_kv")], "expand+reshape（非复制权重）")
        self.tensor("qk_logits", "attn_weights（QKᵀ·scale）", ["B", "Nh", "seqlen", "Lkv+Ncomp"],
                    ["Nh", "Dh"], "activation", "matmul(q,k^T)*Dh^-0.5")
        self.op("qk_scaled", "eager_attention_forward", "QKᵀ·scale", "matmul",
                ["q_rope", "kv_repeated"], ["qk_logits"], [], ["Dh"],
                [src("726", "matmul(query,key.T)*scaling")], "scaling=head_dim^-0.5")
        self.tensor("mask_ext", "attention_mask（含 block_bias）", ["B", "1", "seqlen", "Lmask+Ncomp"],
                    ["Lmask", "M"], "mask", "sliding mask 与 HCA block_bias 拼接")
        self.tensor("mask_model", "模型级 sliding causal mask", ["B", "1", "seqlen", "Lmask"], ["Lmask", "W"],
                    "mask", "HCA 主 KV 的滑窗因果 mask")
        self.tensor("qk_masked", "attn_weights（+mask）", ["B", "Nh", "seqlen", "Lkv+Ncomp"],
                    ["Nh", "Lkv"], "activation", "qk + mask_ext")
        self.op("mask_add", "eager_attention_forward", "加 mask+block_bias", "add",
                ["qk_logits", "mask_ext"], ["qk_masked"], [], ["Lmask", "M"],
                [src("727-728", "attn_weights + attention_mask")], "0/-inf 加法 mask")
        self.tensor("sinks", "sinks 参数", ["Nh"], ["Nh"], "weight", "每头注意力 sink")
        self.op("sink_softmax", "eager_attention_forward", "cat sink→softmax→去 sink", "softmax",
                ["qk_masked", "sinks"], ["attn_probs"], ["sinks"], ["Nh"],
                [src("730-738", "cat(sinks)→softmax→[...,:-1]")],
                "cat sink 列 → 减 max → softmax → 去 sink")
        self.tensor("attn_rope_inv", "attn_output（逆 RoPE 后）", ["B", "seqlen", "Nh", "Dh"],
                    ["Nh", "Dh", "Rk"], "activation", "共轭旋转 -sin 后")
        self.tensor("attn_probs", "scores（去 sink）", ["B", "Nh", "seqlen", "Lkv+Ncomp"],
                    ["Nh", "Lkv"], "activation", "probs[..., :-1]")
        self.tensor("attn_head_out", "attn_output（头维）", ["B", "seqlen", "Nh", "Dh"], ["Nh", "Dh"],
                    "activation", "matmul(attn_probs,v)→transpose(1,2)")
        self.op("av_matmul", "eager_attention_forward", "AV", "matmul",
                ["attn_probs", "kv_repeated"], ["attn_head_out"], [], ["Nh", "Dh"],
                [src("739-741", "matmul(attn_weights,value_states)")], "去 sink 概率×V")
        self.op("attn_inv_rope", "DeepseekV4Attention.forward", "逆 RoPE（-sin）", "rope",
                ["attn_head_out", "cos_q", "sin_q"], ["attn_rope_inv"], [], ["Rk", "Dh"],
                [src("859-865", "apply_rotary_pos_emb(attn_output.transpose(1,2), cos, -sin)")],
                "共轭旋转 -sin；前后 transpose 配对")
        self.tensor("attn_grouped", "attn_output.reshape(B,T,G,-1)", ["B", "seqlen", "G", "Nh×Dh÷G"],
                    ["Nh", "Dh", "G"], "activation", "分组输出投影输入")
        self.op("attn_group_reshape", "DeepseekV4Attention.forward", "分组 reshape", "reshape",
                ["attn_rope_inv"], ["attn_grouped"], [], ["G", "Nh", "Dh"],
                [src("867", "reshape(*input_shape, o_groups, -1)")], "按 o_groups 分组")
        self.tensor("o_a_weight", "o_a_proj.weight（GroupedLinear）", ["G×Ro", "Nh×Dh÷G"],
                    ["G", "Ro", "Nh", "Dh"], "weight", "分组块对角")
        self.tensor("o_b_weight", "o_b_proj.weight", ["H", "G×Ro"], ["H", "G", "Ro"], "weight",
                    "Linear(G*Ro,H,bias=False)")
        self.tensor("o_a_out", "o_a_proj 输出", ["B", "seqlen", "G×Ro"], ["G", "Ro"], "activation",
                    "flatten(2) 后接 o_b_proj")
        self.tensor("attn_block_output", "Attention 输出（o_b_proj）", ["B", "seqlen", "H"], ["H"],
                    "activation", "进入 attn 段 mHC 残差")
        self.op("o_a_proj", "DeepseekV4GroupedLinear.forward", "o_a 分组线性（+flatten）",
                "grouped_linear+flatten",
                ["attn_grouped", "o_a_weight"], ["o_a_out"], ["o_a_weight"], ["G", "Ro", "Nh", "Dh"],
                [src("300-329", "reshape(-1,G,in).transpose(0,1)→bmm→transpose")], "按组 bmm")
        self.op("o_b_proj", "DeepseekV4Attention.o_b_proj", "o_b 输出投影", "linear",
                ["o_a_out", "o_b_weight"], ["attn_block_output"], ["o_b_weight"], ["H", "G", "Ro"],
                [src("869", "self.o_b_proj(grouped)")], "输出 [B,seqlen,H]")
        # 源码顺序：主 KV → cache.update → compressor → mask 扩展 → 拼接 → 核心
        order = [
            "q_a_proj", "q_a_norm", "q_b_proj", "q_b_norm", "rope_compress", "q_rope_apply",
            "kv_proj", "kv_head_view", "kv_rope_apply", "cache_update",
            "c_kv_proj", "c_gate_proj", "c_store_window", "c_window_view", "c_pool",
            "c_rope_gen", "c_rope", "c_update_states", "hca_block_bias",
            "mask_model_build", "kv_concat", "c_mask_extend",
            "repeat_kv", "qk_scaled", "mask_add", "sink_softmax", "av_matmul",
            "attn_inv_rope", "attn_group_reshape", "o_a_proj", "o_b_proj",
        ]
        self.operators[start:] = sorted(
            self.operators[start:],
            key=lambda o: order.index(o["id"]) if o["id"] in order else 999)
