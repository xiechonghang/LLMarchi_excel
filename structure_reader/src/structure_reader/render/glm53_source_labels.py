"""GLM-5.3（glm_moe_dsa）图示源码命名表（展示层，静态阅读）。

只提供「名称」：算子框的源码调用名/表达式、边上张量的源码变量名；不定义算子、
边、shape——连接与维度事实仍全部来自 ``extract/glm53_hf.py`` 的结构文档。
行号按本地官方 transformers 检出 536ecc00… 的
``src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py``（下称 modeling）；
每条都是从该文件静态读出的事实，不是从 DeepSeek 实现推断的。

页面渲染通过 ``diagram_source_labels.use_source_table`` 临时启用本表，退出即复原；
不注册全局模型、不影响 DeepSeek 页面的既有输出。
"""
from __future__ import annotations

__all__ = ["MODELING_FILE", "OP_SOURCE", "TENSOR_SOURCE", "tensor_label_table"]

MODELING_FILE = (
    "hflib/transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py"
)

# 算子 id → (源码调用名/表达式, 行号定位)
OP_SOURCE: dict[str, tuple[str, str]] = {
    # Block：RMSNorm 与 MLA
    "g_input_norm": ("self.input_layernorm(hidden_states)", "597-598"),
    "g_q_a_proj": ("self.q_a_proj(hidden_states)", "383"),
    "g_q_a_norm": ("q_resid = self.q_a_layernorm(q_a_proj(x))", "383"),
    "g_q_b_proj": ("self.q_b_proj(q_resid)", "384"),
    "g_q_view": ("q_states.view(B,S,Nh,Dq).transpose(1,2)", "381-384"),
    "g_q_split": ("torch.split(q_states,[qk_nope,qk_rope],-1)", "385"),
    "g_q_rope": ("apply_rotary_pos_emb_interleave(q_rot,k_rot,cos,sin)", "392-394"),
    "g_q_cat": ("torch.cat((q_pass, q_rot), dim=-1)", "400"),
    "g_kv_a_proj": ("self.kv_a_proj_with_mqa(hidden_states)", "387"),
    "g_kv_a_split": ("compressed_kv.split([kv_lora_rank,qk_rope],-1)", "388"),
    "g_kv_a_norm": ("self.kv_a_layernorm(kv_pass)", "390"),
    "g_k_pass_view": ("k_pass.view(batch_size, 1, seq_length, kv_lora_rank)", "390"),
    "g_kv_rope": ("apply_rotary_pos_emb_interleave(q_rot,k_rot,cos,sin)", "392-394"),
    "g_cache_update": ("past_key_values.update(k_pass,k_rot,layer_idx)", "396-398"),
    "g_kv_b_proj": ("self.kv_b_proj(k_pass)（全部 Lkv 位置）", "402/351-360"),
    "g_kv_view": ("kv_nope.view(key_shape).transpose(1,2)", "358-360"),
    "g_kv_split": ("torch.split(kv_nope,[qk_nope,v_head],-1)", "361"),
    "g_k_rot_expand": ("k_rot.expand(-1, num_heads, -1, -1)", "362"),
    "g_key_cat": ("cat([k_nope, k_rot]) → key_states", "365-368"),
    # DSA indexer（仅 full 层）
    "g_ix_q_proj": ("self.indexer.wq_b(q_resid)", "213"),
    "g_ix_q_view": ("q.view(B,S,index_n_heads,index_head_dim)", "214"),
    "g_ix_q_split": ("torch.split(q,[qk_rope,head_dim-qk_rope],-1)", "215"),
    "g_ix_k_proj": ("self.indexer.wk(hidden_states)", "217"),
    "g_ix_k_norm": ("self.indexer.k_norm(k)（LayerNorm，带 bias）", "180/217"),
    "g_ix_k_view": ("k.unsqueeze(2)", "217"),
    "g_ix_k_split": ("torch.split(k,[qk_rope,head_dim-qk_rope],-1)", "218"),
    "g_ix_rope": ("apply_rotary_pos_emb_interleave(...,unsqueeze_dim=2)", "220-222"),
    "g_ix_q_cat": ("torch.cat([q_rot, q_pass], dim=-1)", "222"),
    "g_ix_k_cat": ("torch.cat([k_rot, k_pass], dim=-1).squeeze(2)", "223"),
    "g_ix_cache_update": ("past_key_values.update_indexer(k, layer_idx)", "225-226"),
    "g_ix_scores": ("matmul(q.float(),k.float().T.unsqueeze(1))*softmax_scale", "228"),
    "g_ix_relu": ("F.relu(scores)", "229"),
    "g_ix_weights": ("self.weights_proj(hidden).float()*(n_heads**-0.5)", "232"),
    "g_ix_sum": ("matmul(weights.unsqueeze(-2), scores).squeeze(-2)", "233"),
    "g_ix_mask": ("index_scores + attention_mask[:,0]", "235-239/410"),
    "g_ix_topk": ("index_scores.topk(min(index_topk,L)).indices.int()", "241-242"),
    # 稀疏 mask 与注意力核心
    "g_index_mask_scatter": ("new_ones((B,S,L),bool).scatter(-1,top_k,False)", "421-425"),
    "g_sparse_mask_apply": ("attention_mask.masked_fill(index_mask, min)", "426-430"),
    "g_qk": ("matmul(query,key_states.transpose(2,3))*scaling", "270"),
    "g_mask_add": ("attn_weights + attention_mask", "271-272"),
    "g_softmax": ("softmax(attn_weights,dim=-1,dtype=float32)", "274"),
    "g_av": ("matmul(attn_weights, value_states)", "276"),
    "g_attn_transpose": ("attn_output.transpose(1, 2).contiguous()", "277"),
    "g_attn_reshape": ("attn_output.reshape(B,seqlen,-1)", "449"),
    "g_o_proj": ("self.o_proj(attn_output)", "450"),
    "g_res_attn": ("hidden_states = residual + hidden_states", "610"),
    "g_post_norm": ("self.post_attention_layernorm(hidden_states)", "613"),
    # 稠密 MLP
    "g_dense_gate": ("self.gate_proj(x)", "466"),
    "g_dense_up": ("self.up_proj(x)", "466"),
    "g_dense_act": ("self.act_fn(gate) * up", "466"),
    "g_dense_down": ("self.down_proj(act_fn(gate)*up)", "466"),
    # MoE 路由
    "g_moe_flatten": ("hidden_states.view(-1, hidden_dim)", "569"),
    "g_moe_router": ("F.linear(x.float(), self.weight.float())", "485"),
    "g_moe_sigmoid": ("router_logits.sigmoid()", "486"),
    "g_moe_bias_add": ("scores + self.e_score_correction_bias", "487"),
    "g_moe_group_score": ("view(-1,n_group,E/n_group).topk(2).sum(-1)", "488-492"),
    "g_moe_group_topk": ("topk(group_scores, k=topk_group)", "493"),
    "g_moe_group_scatter": ("group_mask.scatter_(1, group_idx, 1)", "494-495"),
    "g_moe_group_expand": ("group_mask.unsqueeze(-1).expand(-1,Ng,E//Ng)", "496-499"),
    "g_moe_group_reshape": (".reshape(-1, num_experts)", "500"),
    "g_moe_group_masked_fill": ("masked_fill(~score_mask, -inf)", "501"),
    "g_moe_topk": ("topk(scores_for_choice, k=top_k)", "502"),
    "g_moe_weight_gather": ("scores.gather(1, topk_indices)", "503"),
    "g_moe_weight_norm": ("topk_weights /= sum(-1)+1e-20", "504-506"),
    "g_moe_weight_scale": ("topk_weights * routed_scaling_factor", "507"),
    # MoE 专家（单份模板）
    "g_moe_expert_mask": ("one_hot(top_k_index,E+1).permute(2,1,0)", "531-533"),
    "g_moe_expert_hit": ("expert_mask.sum(-1,-2).nonzero()", "534-535"),
    "g_moe_expert_gather_indices": ("torch.where(expert_mask[expert_idx])", "540"),
    "g_moe_result_init": ("final_hidden_states = zeros_like(hidden_states)", "530"),
    "g_moe_select_tokens": ("hidden_states[token_idx]", "540-541"),
    "g_moe_expert_gate_up": ("F.linear(x, gate_up_proj[expert_idx])", "542"),
    "g_moe_expert_chunk": (".chunk(2, dim=-1)", "542"),
    "g_moe_expert_act": ("self.act_fn(gate) * up", "543"),
    "g_moe_expert_down": ("F.linear(x, down_proj[expert_idx])", "544"),
    "g_moe_expert_weight": ("* top_k_weights[token_idx, pos]", "545"),
    "g_moe_index_add": ("final.index_add_(0, token_idx, current)", "546"),
    "g_moe_routed_view": (".view(*orig_shape)", "570"),
    "g_moe_shared_gate": ("self.shared_experts.gate_proj(residuals)", "571/466"),
    "g_moe_shared_up": ("self.shared_experts.up_proj(residuals)", "466"),
    "g_moe_shared_act": ("self.act_fn(gate) * up", "466"),
    "g_moe_shared_down": ("self.shared_experts.down_proj(act)", "466"),
    "g_moe_merge": ("routed + self.shared_experts(residuals)", "571-572"),
    "g_res_ffn": ("hidden_states = residual + hidden_states", "615"),
    # 整网总览
    "net_embed": ("self.embed_tokens(input_ids)", "685-686"),
    "net_position_ids": ("arange(seqlen) + past_seen_tokens", "691-694"),
    "net_mask_build": ("create_causal_mask(..., allow_is_causal_skip=False)", "696-706"),
    "net_rope": ("self.rotary_emb(hidden_states, position_ids)", "708-709"),
    "net_layer_loop": ("for layer in self.layers[:num_hidden_layers]", "712-722"),
    "net_final_norm": ("self.norm(hidden_states)", "724"),
    "net_logits_slice": ("hidden_states[:, slice_indices, :]", "790"),
    "net_lm_head": ("self.lm_head(hidden_states[:, slice_indices, :])", "791"),
}


# 张量 id → 源码变量/短表达式（图上箭头标签用；完整说明仍在算子卡与张量表）。
# 名字保持简短是刻意的：图上的边标签空间有限，长表达式会互相压盖。
TENSOR_SOURCE: dict[str, str] = {
    "g_hidden_in": "hidden_states（层输入）",
    "g_position_ids": "position_ids",
    "g_attention_mask": "attention_mask",
    "g_cos": "cos",
    "g_sin": "sin",
    "g_prev_topk_indices": "prev_topk_indices",
    "g_attn_normed": "hidden_states",
    "g_q_a": "self.q_a_proj(hidden_states)",
    "g_q_residual": "q_resid",
    "g_q_flat": "q_states",
    "g_q": "q",
    "g_q_pass": "q_pass",
    "g_q_rot": "q_rot",
    "g_q_rot_emb": "q_rot",
    "g_q_full": "query_states",
    "g_kv_compressed": "compressed_kv",
    "g_kv_pass": "kv_pass",
    "g_k_pass_normed": "k_pass",
    "g_k_rot": "k_rot",
    "g_k_rot_emb": "k_rot",
    "g_cache_k_pass_pre": "self.keys",
    "g_cache_k_pass_post": "self.keys",
    "g_cache_k_rot_pre": "self.values",
    "g_cache_k_rot_post": "self.values",
    "g_kv_nope_flat": "self.kv_b_proj(k_pass)",
    "g_kv_nope": "kv_nope",
    "g_k_nope": "k_nope",
    "g_v": "value_states",
    "g_k_rot_expanded": "k_rot",
    "g_key": "key_states",
    "g_ix_q_flat": "self.indexer.wq_b(q_resid)",
    "g_ix_q": "q",
    "g_ix_q_rot": "q_rot",
    "g_ix_q_pass": "q_pass",
    "g_ix_k_proj": "self.indexer.wk(hidden_states)",
    "g_ix_k_normed": "k_norm(k)",
    "g_ix_k": "k",
    "g_ix_k_rot": "k_rot",
    "g_ix_k_pass": "k_pass",
    "g_ix_q_rot_emb": "q_rot",
    "g_ix_k_rot_emb": "k_rot",
    "g_ix_q_cat": "q",
    "g_ix_k_cat": "k",
    "g_cache_indexer_keys_pre": "self.indexer_keys",
    "g_cache_indexer_keys_post": "self.indexer_keys",
    "g_ix_scores": "scores",
    "g_ix_relu": "scores",
    "g_ix_weights": "weights",
    "g_ix_index_scores": "index_scores",
    "g_ix_masked_scores": "index_scores",
    "g_ix_topk_indices": "topk_indices",
    "g_index_mask": "index_mask",
    "g_sparse_mask": "attention_mask",
    "g_attn_logits": "attn_weights",
    "g_attn_masked": "attn_weights",
    "g_attn_probs": "attn_weights",
    "g_attn_head_out": "attn_output",
    "g_attn_flat": "attn_output",
    "g_attn_out": "self.o_proj(attn_output)",
    "g_hidden_mid": "hidden_states",
    "g_ffn_in": "hidden_states",
    "g_dense_gate": "self.gate_proj(x)",
    "g_dense_up": "self.up_proj(x)",
    "g_dense_act": "act_fn(gate)*up",
    "g_mlp_out": "mlp_out",
    "g_moe_flat": "hidden_states",
    "g_router_logits": "router_logits",
    "g_router_scores": "scores",
    "g_scores_for_choice": "scores_for_choice",
    "g_group_scores": "group_scores",
    "g_group_idx": "group_idx",
    "g_score_mask": "score_mask",
    "g_masked_scores": "scores_for_choice",
    "g_topk_indices": "topk_indices",
    "g_topk_weights_raw": "topk_weights",
    "g_topk_weights_norm": "topk_weights",
    "g_topk_weights": "topk_weights",
    "g_expert_mask": "expert_mask",
    "g_expert_hit": "expert_hit",
    "g_expert_token_idx": "token_idx",
    "g_expert_topk_pos": "top_k_pos",
    "g_group_mask": "group_mask",
    "g_group_mask_expanded": "score_mask(expand)",
    "g_k_pass_4d": "k_pass",
    "g_attn_heads_seq": "attn_output",
    "g_routed_init": "final_hidden_states",
    "g_token_states": "current_state",
    "g_gate_up": "gate_up",
    "g_expert_gate": "gate",
    "g_expert_up": "up",
    "g_expert_hidden": "current_hidden_states",
    "g_expert_out": "current_hidden_states",
    "g_expert_weighted": "current_hidden_states",
    "g_routed_flat": "final_hidden_states",
    "g_routed": "hidden_states",
    "g_shared_gate": "shared.gate_proj(residuals)",
    "g_shared_up": "shared.up_proj(residuals)",
    "g_shared_act": "act_fn(gate)*up",
    "g_shared_out": "shared_experts(residuals)",
    "g_hidden_out": "hidden_states（层输出）",
    # 整网
    "input_ids": "input_ids",
    "attention_mask": "attention_mask",
    "past_key_values": "past_key_values",
    "past_key_values_post": "past_key_values",
    "inputs_embeds": "inputs_embeds",
    "position_ids": "position_ids",
    "causal_mask": "causal_mask",
    "cos_rope": "position_embeddings cos",
    "sin_rope": "position_embeddings sin",
    "hidden_after_layers": "hidden_states（全部层之后）",
    "final_hidden": "hidden_states（Final RMSNorm 后）",
    "lm_input": "hidden_states[:, slice_indices, :]",
    "logits": "logits",
}


def tensor_label_table(blocks) -> dict:
    """从结构文档的张量名生成展示表（名称即源码变量/表达式；locator 由算子表承担）。

    返回 ``{tensor_id: (显示名, "")}``。同名张量在多个层组复用且名称一致；若出现
    同名不同义，保留第一个并在值里带组信息由调用方检查（本模型 id 全局唯一）。
    """
    table: dict[str, tuple[str, str]] = {}
    for block in blocks:
        for tensor in block.get("tensors", []):
            tid = tensor.get("id")
            if not tid:
                continue
            label = TENSOR_SOURCE.get(tid) or str(tensor.get("name") or tid)
            if tid in table and table[tid][0] != label:
                raise ValueError(f"张量 {tid} 在不同层组名称不一致: {table[tid][0]!r} vs {label!r}")
            table[tid] = (label, "")
    return table


def source_table(blocks) -> dict:
    """页面使用的完整源码命名表（算子表达式 + 张量名 + modeling 路径）。"""
    return {
        "modeling_file": MODELING_FILE,
        "ops": dict(OP_SOURCE),
        "tensors": tensor_label_table(blocks),
    }
