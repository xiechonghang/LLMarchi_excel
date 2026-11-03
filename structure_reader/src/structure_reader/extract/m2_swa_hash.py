"""SWA+Hash（层0/1）与 CSA+Hash（层2）Block 结构提取。只数据，不做展示/导出。"""
from __future__ import annotations

from structure_reader.extract.deepseek_v4_hf import BlockStructureBuilder

_SWA_NEEDED = {"B", "seqlen", "kvcachelen", "H", "Nh", "Nkv", "Dh", "Rq", "Rk", "G", "Ro",
               "Nhc", "Sit", "Eps", "EpsN", "E", "K", "Es", "Lim", "Scl", "W",
               "Lkv", "Lmask", "ThetaM", "Vocab"}


def _add_hash_params(b: BlockStructureBuilder) -> None:
    v = b.values
    b.param("Vocab", "Vocab", "词表大小（Hash tid2eid 第一维）", v.vocab_size,
            "config.vocab_size", "tid2eid 持久 buffer 形状 [vocab_size, num_experts_per_tok]",
            [b.cfg("vocab_size", "config.vocab_size")])


class SWABlockBuilder(BlockStructureBuilder):
    mlp_type = "hash_moe"

    def _build_parameters(self) -> None:
        super()._build_parameters()
        self.parameters = [p for p in self.parameters if p["id"] in _SWA_NEEDED]
        v = self.values
        self.param("ThetaM", "θm", "main RoPE theta（SWA 普通 RoPE）", v.rope_main_theta,
                   "config.rope_theta=10000", "SWA 使用 main rope_type='default'，非 compress YaRN",
                   [self.cfg("rope_theta", "config.rope_theta")])
        _add_hash_params(self)

    def _build_boundary_and_cache(self) -> None:
        self.tensor("input_ids", "input_ids", ["B", "seqlen"], ["B", "seqlen"], "boundary",
                    "HashRouter 必需外部输入 [B,seqlen]；沿 reshape 查 tid2eid")
        self.tensor("position_ids", "position_ids", ["1", "seqlen"], ["seqlen"], "boundary",
                    "本调用绝对位置 [1,seqlen]；SWA 用 main RoPE")
        self.tensor("hidden_streams_in", "hidden_streams（层输入）", ["B", "seqlen", "Nhc", "H"],
                    ["Nhc", "H"], "boundary", "上一 decoder 层输出的 Nhc 路流")
        self.tensor("hidden_streams_out", "hidden_streams（层输出）", ["B", "seqlen", "Nhc", "H"],
                    ["Nhc", "H"], "boundary", "本层最终输出")
        self.tensor("cache_keys", "DynamicSlidingWindowLayer.keys（调用前）",
                    ["B", "1", "min(kvcachelen,W-1)", "Dh"], ["Lkv", "Dh", "W"], "cache",
                    "SWA 主 KV 调用前状态：空缓存 Prefill（kvcachelen=0）为 0 长度，"
                    "非零历史 Decode 为最近 min(kvcachelen,W-1) 个位置")
        self.tensor("cache_values", "DynamicSlidingWindowLayer.values（调用前）",
                    ["B", "1", "min(kvcachelen,W-1)", "Dh"], ["Lkv", "Dh", "W"], "cache",
                    "SWA 的 keys/values 是独立持久字段（不是 self.values=self.keys）；"
                    "本次输入 K/V 数值都来自 kv_rope")
        self.tensor("cache_keys_post", "DynamicSlidingWindowLayer.keys（调用后）",
                    ["B", "1", "min(kvcachelen+seqlen,W-1)", "Dh"], ["Lkv", "Dh", "W"], "cache",
                    "full_key 的持久 tail（min(kvcachelen+seqlen,W-1)）")
        self.tensor("cache_values_post", "DynamicSlidingWindowLayer.values（调用后）",
                    ["B", "1", "min(kvcachelen+seqlen,W-1)", "Dh"], ["Lkv", "Dh", "W"], "cache",
                    "full_value 的持久 tail（独立字段）")
        self.tensor("cache_cumlen", "cumulative_length（主 KV）", [], ["W"], "cache",
                    "决定 mask offset/length；空缓存 Prefill 为 0，非零历史 Decode 为 kvcachelen")
        self.tensor("cache_cumlen_post", "cumulative_length（调用后）", [], ["W"], "cache",
                    "调用后 = kvcachelen + seqlen")

    def _attention_block(self) -> None:
        src = self.src
        self.tensor("q_a", "q_a_proj(hidden)", ["B", "seqlen", "Rq"], ["Rq", "H"], "activation", "低秩 Q")
        self.tensor("q_weight", "q_a_proj.weight", ["Rq", "H"], ["Rq", "H"], "weight", "Linear(H,Rq)")
        self.tensor("q_a_norm_weight", "q_a_norm.weight", ["Rq"], ["Rq"], "weight", "RMSNorm")
        self.op("q_a_proj", "DeepseekV4Attention.q_a_proj", "q_a_proj", "linear",
                ["attn_normed", "q_weight"], ["q_a"], ["q_weight"], ["Rq", "H"],
                [src("813-814", "q_a_proj")], "Linear 无 bias")
        self.tensor("q_a_normed", "q_a_norm(q_a)", ["B", "seqlen", "Rq"], ["Rq", "EpsN"], "activation",
                    "q_residual（SWA 无 Indexer，不消费）")
        self.op("q_a_norm", "DeepseekV4Attention.q_a_norm", "q_a_norm", "rmsnorm",
                ["q_a", "q_a_norm_weight"], ["q_a_normed"], ["q_a_norm_weight"], ["Rq", "EpsN"],
                [src("813-814", "q_a_norm")], "无 Indexer 消费")
        self.tensor("q_b_weight", "q_b_proj.weight", ["Nh×Dh", "Rq"], ["Nh", "Dh"], "weight", "Linear")
        self.tensor("q_head", "q.view/transpose", ["B", "Nh", "seqlen", "Dh"], ["Nh", "Dh"], "activation",
                    "q_b_proj→view→transpose")
        self.op("q_b_proj", "DeepseekV4Attention.q_b_proj", "q_b_proj（+view/transpose）",
                "linear+view+transpose", ["q_a_normed", "q_b_weight"], ["q_head"],
                ["q_b_weight"], ["Nh", "Dh", "Rq"], [src("813-814", "q_b_proj")], "复合步骤")
        self.tensor("q_normed", "q_b_norm", ["B", "Nh", "seqlen", "Dh"], ["Dh"], "activation",
                    "UnweightedRMSNorm")
        self.op("q_b_norm", "DeepseekV4Attention.q_b_norm", "q_b_norm", "rmsnorm",
                ["q_head"], ["q_normed"], [], ["Dh"], [src("786,815", "q_b_norm")], "无权重")
        self.tensor("cos_q", "cos（main）", ["1", "seqlen", "Rk÷2"], ["Rk", "ThetaM"], "activation",
                    "main RoPE cos（theta=10000）")
        self.tensor("sin_q", "sin（main）", ["1", "seqlen", "Rk÷2"], ["Rk", "ThetaM"], "activation",
                    "main RoPE sin")
        self.op("rope_main", "DeepseekV4RotaryEmbedding", "main RoPE cos/sin", "rope",
                ["position_ids"], ["cos_q", "sin_q"], [], ["Rk", "ThetaM"],
                [src("148-153", "default rope（theta=rope_theta）")], "SWA 用 main，不是 compress")
        self.tensor("q_rope", "q（部分 RoPE）", ["B", "Nh", "seqlen", "Dh"], ["Nh", "Dh", "Rk"],
                    "activation", "末 Rk 维 interleaved")
        self.op("q_rope_apply", "apply_rotary_pos_emb", "q partial RoPE", "rope",
                ["q_normed", "cos_q", "sin_q"], ["q_rope"], [], ["Rk", "Dh"],
                [src("339-356", "apply_rotary_pos_emb")], "partial")
        # 主 KV（K=V 同一 kv_rope；持久字段独立）
        self.tensor("kv_weight_attn", "self.kv_proj.weight", ["Dh", "H"], ["Dh", "H"], "weight",
                    "Linear(H,head_dim)")
        self.tensor("kv_norm_weight", "self.kv_norm.weight", ["Dh"], ["Dh"], "weight", "RMSNorm")
        self.tensor("kv_main", "self.kv_norm(self.kv_proj(hidden))", ["B", "seqlen", "Dh"], ["Dh", "H"],
                    "activation", "Linear+RMSNorm")
        self.op("kv_proj", "DeepseekV4Attention.kv_proj", "Attention 主 KV 投影+RMSNorm",
                "linear+rmsnorm", ["attn_normed", "kv_weight_attn", "kv_norm_weight"], ["kv_main"],
                ["kv_weight_attn", "kv_norm_weight"], ["Dh", "H", "EpsN"],
                [src("817-818", "self.kv_norm(self.kv_proj(...))")],
                "Linear+RMSNorm；num_key_value_heads=1 单 KV 头；K=V 同一张量")
        self.tensor("kv_head", "kv.view/transpose", ["B", "1", "seqlen", "Dh"], ["Dh"], "activation",
                    "view(B,T,1,Dh).transpose(1,2)")
        self.op("kv_head_view", "DeepseekV4Attention.kv_head_view", "主 KV view", "view",
                ["kv_main"], ["kv_head"], [], ["Dh"], [src("818", "view/transpose")], "单 KV 头")
        self.tensor("kv_rope", "kv（main RoPE 后）", ["B", "1", "seqlen", "Dh"], ["Dh", "Rk"],
                    "activation", "main RoPE partial")
        self.op("kv_rope_apply", "apply_rotary_pos_emb", "KV partial RoPE", "rope",
                ["kv_head", "cos_q", "sin_q"], ["kv_rope"], [], ["Rk", "Dh"],
                [src("339-356", "apply_rotary_pos_emb(kv)")], "K=V 同一 kv_rope")
        # SWA cache：keys/values 独立字段，返回 full（K/V 同值）
        self.op("cache_update", "DynamicSlidingWindowLayer.update", "SWA 缓存更新", "cache_update",
                ["kv_rope", "cache_keys", "cache_values", "cache_cumlen"],
                ["kv_full", "cache_keys_post", "cache_values_post", "cache_cumlen_post"], [],
                ["W", "Lkv"],
                [self.srcf("cache_utils.py", "211-294", "DynamicSlidingWindowLayer.update")],
                "SWA：keys/values 独立持久字段；输入 K/V 都是 kv_rope（数值相等），"
                "但不是 self.values=self.keys；返回 full[0] 供共享 K/V")
        self.tensor("kv_full", "cache.update 返回 full", ["B", "1", "Lkv", "Dh"], ["Lkv", "Dh"],
                    "cache", "本次可读 K/V full；无压缩条目")
        self.tensor("mask_model", "sliding causal mask", ["B", "1", "seqlen", "Lmask"],
                    ["Lmask", "W"], "mask", "SWA mask（无 block_bias）")
        self.op("mask_model_build", "create_sliding_window_causal_mask", "sliding mask", "mask_build",
                ["position_ids", "cache_cumlen"], ["mask_model"], [], ["Lmask", "W"],
                [self.srcf("masking_utils.py", "1078-1162", "create_sliding_window_causal_mask")],
                "mask 只覆盖主 KV Lkv，无压缩条目")
        self.tensor("kv_repeated", "repeat_kv", ["B", "Nh", "Lkv", "Dh"], ["Nh", "Lkv", "Dh"],
                    "activation", "单 KV 头广播")
        self.op("repeat_kv", "repeat_kv", "KV 广播", "repeat_kv", ["kv_full"], ["kv_repeated"], [],
                ["Nh"], [src("702-711", "repeat_kv")], "expand+reshape")
        self.tensor("qk_logits", "attn_weights", ["B", "Nh", "seqlen", "Lkv"], ["Nh", "Dh"], "activation",
                    "QKᵀ·scale")
        self.op("qk_scaled", "eager_attention_forward", "QKᵀ·scale", "matmul",
                ["q_rope", "kv_repeated"], ["qk_logits"], [], ["Dh"], [src("726", "qk*scaling")],
                "Lkv 轴无压缩条目")
        self.tensor("qk_masked", "attn_weights+mask", ["B", "Nh", "seqlen", "Lkv"], ["Lmask"],
                    "activation", "只有主 mask")
        self.op("mask_add", "eager_attention_forward", "加 mask", "add",
                ["qk_logits", "mask_model"], ["qk_masked"], [], ["Lmask"],
                [src("727-728", "attn_weights + mask")], "SWA 无 block_bias")
        self.tensor("sinks", "sinks", ["Nh"], ["Nh"], "weight", "每头 sink")
        self.op("sink_softmax", "eager_attention_forward", "sink softmax", "softmax",
                ["qk_masked", "sinks"], ["attn_probs"], ["sinks"], ["Nh"],
                [src("730-738", "cat sink→softmax→[...,:-1]")], "cat/softmax/去 sink")
        self.tensor("attn_probs", "scores", ["B", "Nh", "seqlen", "Lkv"], ["Nh"], "activation",
                    "去 sink 概率")
        self.tensor("attn_head_out", "attn_output", ["B", "seqlen", "Nh", "Dh"], ["Nh", "Dh"],
                    "activation", "AV 后 transpose")
        self.op("av_matmul", "eager_attention_forward", "AV", "matmul",
                ["attn_probs", "kv_repeated"], ["attn_head_out"], [], ["Nh", "Dh"],
                [src("739-741", "matmul(probs,v)")], "无压缩 V")
        self.op("attn_inv_rope", "DeepseekV4Attention.forward", "逆 RoPE（-sin, main）", "rope",
                ["attn_head_out", "cos_q", "sin_q"], ["attn_rope_inv"], [], ["Rk", "Dh"],
                [src("859-865", "cos,-sin")], "main RoPE 共轭")
        self.tensor("attn_rope_inv", "反旋转后", ["B", "seqlen", "Nh", "Dh"], ["Nh", "Dh"], "activation",
                    "-sin 后")
        self.tensor("attn_grouped", "grouped", ["B", "seqlen", "G", "Nh×Dh÷G"], ["G", "Nh"], "activation",
                    "reshape 分组")
        self.op("attn_group_reshape", "DeepseekV4Attention.forward", "分组 reshape", "reshape",
                ["attn_rope_inv"], ["attn_grouped"], [], ["G", "Nh", "Dh"],
                [src("867", "reshape(*shape,G,-1)")], "按 o_groups")
        self.tensor("o_a_weight", "o_a_proj.weight", ["G×Ro", "Nh×Dh÷G"], ["G", "Ro"], "weight",
                    "GroupedLinear")
        self.tensor("o_b_weight", "o_b_proj.weight", ["H", "G×Ro"], ["H", "G", "Ro"], "weight",
                    "Linear(G*Ro,H)")
        self.tensor("o_a_out", "o_a_proj 输出", ["B", "seqlen", "G×Ro"], ["G", "Ro"], "activation",
                    "flatten(2)")
        self.tensor("attn_block_output", "Attention 输出", ["B", "seqlen", "H"], ["H"], "activation",
                    "进入 attn 残差")
        self.op("o_a_proj", "DeepseekV4GroupedLinear.forward", "o_a 分组+flatten",
                "grouped_linear+flatten", ["attn_grouped", "o_a_weight"], ["o_a_out"],
                ["o_a_weight"], ["G", "Ro"], [src("300-329", "grouped linear")], "按组 bmm")
        self.op("o_b_proj", "DeepseekV4Attention.o_b_proj", "o_b 输出投影", "linear",
                ["o_a_out", "o_b_weight"], ["attn_block_output"], ["o_b_weight"], ["H", "G", "Ro"],
                [src("869", "o_b_proj")], "[B,seqlen,H]")

    def _build_derivations(self) -> None:
        """SWA 派生量：无压缩窗口/Indexer/buffer，只推导主 KV 与 mask 长度。"""
        from structure_reader.extract.deepseek_v4_hf import _EXPR_LKV, _EXPR_PERSIST

        common_note = ("按固定配置 + 「空态起步、连续正常执行」前提推导；"
                       "不覆盖裁剪/回滚后的自定义快照")
        self.derivation("L_kv", "本次可读主 KV 长度", _EXPR_LKV,
                        "持久尾部（最多 W-1）+ 本调用新 token；SWA 无压缩条目。" + common_note)
        self.derivation("L_mask", "模型级 mask KV 长度", _EXPR_LKV,
                        "与 L_kv 同式；SWA mask 只覆盖主 KV。" + common_note)
        self.derivation("L_kv_post", "调用后持久主 KV 长度", _EXPR_PERSIST,
                        "update 后 keys/values 只持久 full 的最后 W-1 个位置。" + common_note)

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "SWA 用 DynamicSlidingWindowLayer（cache_utils:211-294）：keys/values 是独立持久字段；"
            "本次输入 K/V 都是 kv_rope（数值相等），但不是 self.values=self.keys。",
            "无 compressor/buffer/entry_count/Indexer/overlap；mask 只覆盖主 KV L_kv。",
            "HashRouter：input_ids [B,seqlen] → tid2eid [Vocab,K] 查表 indices；learned scores gather，"
            "无 TopK 分数选择与 correction bias。",
            "空缓存 Prefill（kvcachelen=0）起步为 0 长度；非零历史 Decode 起步为最近 "
            "min(kvcachelen,W-1) 个位置尾部，本图不验证缓存内容数值。",
        ]

    def _build_coverage_notes(self) -> None:
        self.coverage_notes = [
            "覆盖：SWA attention（main RoPE、独立 keys/values、sliding mask）、Hash 路由与专家模板、"
            "共享专家、mHC、逆 RoPE、分组输出。",
            "未覆盖：HCA/CSA 压缩器（本层无）、量化执行；整网外围与 token→logits 见整网层。",
        ]


class CSAHashBlockBuilder(BlockStructureBuilder):
    mlp_type = "hash_moe"

    def _build_boundary_and_cache(self) -> None:
        super()._build_boundary_and_cache()
        for t in self.tensors:
            if t["id"] == "input_ids":
                t["note"] = "HashRouter 真实输入 [B,seqlen]：reshape 后查 tid2eid 选择专家"

    def _build_parameters(self) -> None:
        super()._build_parameters()
        _add_hash_params(self)

    def _build_cache_notes(self) -> None:
        self.cache_notes = [
            "CSA 压缩器/Indexer 缓存与 Attention 主 KV 同既有 CSA 事实（同一份结构，场景绑定取值）。",
            "HashRouter：input_ids [B,seqlen] → tid2eid [Vocab,K] 查表 indices；无 TopK/bias。",
            "空缓存 Prefill（kvcachelen=0）起步为空；非零历史 Decode 起步含历史尾部、"
            "压缩条目与 buffer 余量，本图不验证缓存内容数值。",
        ]

    def _build_coverage_notes(self) -> None:
        self.coverage_notes = [
            "覆盖：CSA attention（含 Indexer）、Hash 路由、专家模板、共享专家、mHC。",
            "未覆盖：SWA/HCA 变体（本层非）、量化执行；整网外围与 token→logits 见整网层。",
        ]
