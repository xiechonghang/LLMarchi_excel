"""GLM-5.3-Flash 模块/总览中文结构讲解（事实全部读 Block 文档）。

- 只读 ``operators`` / ``tensors`` / ``parameters``，不引入模型之外的结论；
- 可见名复用 ``display_shape`` 的完整名称，内链使用当前 Block 的真实作用域 id
  （``block-<组id>``），与参数表/张量表/模块详情同源；
- 引用不存在的张量/参数直接抛 KeyError（不静默造对象，避免讲错结构）；
- 文本按「KDA/DSA × dense/MoE」分支：Flash 的 KDA 状态、mHC 4 路流、NoPE、
  k-pool indexer、SwiGLU clamp 分别叙述，不套 GLM-5.3/DeepSeek 的机制。
"""
from __future__ import annotations

import html

from structure_reader.render.display_names import display_shape, param_label

__all__ = ["module_narrative", "overview_narrative"]


class _C:
    def __init__(self, block: dict):
        self.block = block
        self.pid = str(block.get("id") or "block")
        self.tensors = {t["id"]: t for t in block.get("tensors", [])}
        self.params = {p["id"]: p for p in block.get("parameters", [])}
        self.ops = {o["id"]: o for o in block.get("operators", [])}

    def sh(self, tid: str) -> str:
        if tid not in self.tensors:
            raise KeyError(f"narrative: 未知 tensor {tid}")
        return display_shape(self.tensors[tid]["shape"])

    def t(self, tid: str) -> str:
        if tid not in self.tensors:
            raise KeyError(f"narrative: 未知 tensor {tid}")
        return (f'<a class="mono narr-link" title="{html.escape(tid)}" '
                f'href="#t-{self.pid}-{html.escape(tid)}">'
                f'{html.escape(str(self.tensors[tid].get("name", tid)))}</a>')

    def p(self, pid: str) -> str:
        if pid not in self.params:
            raise KeyError(f"narrative: 未知参数 {pid}")
        return (f'<a class="mono narr-link" title="{html.escape(pid)}" '
                f'href="#p-{self.pid}-{html.escape(pid)}">'
                f'{param_label(pid, self.params[pid].get("value"))}</a>')

    def m(self, mid: str, label: str) -> str:
        return (f'<a class="narr-link" href="#mod-{self.pid}-{html.escape(mid)}">'
                f'{html.escape(label)}</a>')

    def wrap(self, module: str, lead: str, steps: list[str] | None = None) -> str:
        lis = "".join(f"<li>{s}</li>" for s in (steps or []))
        return (f'<div class="narr" data-module="{self.pid}/{html.escape(module)}">'
                f'<p class="narr-lead">{lead}</p>'
                + (f'<ol class="narr-steps">{lis}</ol>' if lis else "")
                + "</div>")


def _kind(c: _C) -> str:
    return "kda" if c.block.get("kind") == "kda" else "dsa"


def _moe(c: _C) -> bool:
    return c.block.get("moe_type") == "sparse"


def _mhc_attn(c: _C) -> str:
    lead = (f"Attention 前的 mHC（attn_hc）：4 路流 {c.t('g_hidden_streams_in')} "
            f"{c.sh('g_hidden_streams_in')} 先展平/无权重 RMSNorm，再由 "
            f"{c.t('g_mhc_attn_fn_weight')} 投影出 (2+Hc)×Hc={c.p('Mhc')} 个系数 "
            f"{c.sh('g_mhc_attn_mixes')}（pre 4 + post 4 + comb 16）。")
    return c.wrap("mhc_attn", lead, [
        f"pre = sigmoid(...)+hc_eps、post = 2*sigmoid(...)，comb 先 softmax 再做 "
        f"{c.p('Sit')} 轮 Sinkhorn 行列归一，得到双随机矩阵 {c.t('g_mhc_attn_comb')} "
        f"{c.sh('g_mhc_attn_comb')}。",
        f"坍缩后的单流 {c.t('g_mhc_attn_collapsed')} {c.sh('g_mhc_attn_collapsed')} 进入 "
        "input_layernorm；这不是普通单流残差，残差在 Attention 之后才由 post/comb 混合。"])


def _attn_norm(c: _C) -> str:
    lead = (f"层输入坍缩成 {c.t('g_mhc_attn_collapsed')} 后做 RMSNorm（eps={c.p('EpsN')}）："
            f"{c.t('g_attn_normed')} {c.sh('g_attn_normed')}。")
    return c.wrap("attn_norm", lead, [
        "mHC 的残差保留的是 4 路流，不是这个归一化后的单流；不要把 mHC 讲成普通 "
        "residual + norm。"])


def _kda_proj(c: _C) -> str:
    lead = (f"KDA 先把 hidden 的 padding 位置置零（{c.t('g_attention_mask')} 是 2D padding "
            f"mask，不是因果 mask），再分别做 q/k/v 投影："
            f"{c.sh('g_kda_q_flat')} → 拼接 {c.sh('g_kda_qkv')} → transpose 成 "
            f"{c.sh('g_kda_qkv_t')} 供 depthwise 因果短卷积（kernel={c.p('Kconv')}）。")
    return c.wrap("kda_proj", lead, [
        "q/k/v 投影都是 hidden→64×128 的线性层；拼接顺序固定 q/k/v，"
        "转置只是把序列维移到最后一维供 Conv1d 使用。",
        "KDA 的因果性不是靠 4D causal mask；短卷积与 delta-rule 递推顺序共同保证左侧上下文。"])


def _kda_conv(c: _C) -> str:
    lead = (f"卷积前先更新 conv state：{c.t('g_kda_conv_state_pre')} "
            f"{c.sh('g_kda_conv_state_pre')} 保存最近 {c.p('Kconv')} 个 token 的 qkv 通道；"
            f"首次调用按需左补零，续接时与前态拼接（cache_utils:1099-1149）。")
    return c.wrap("kda_conv", lead, [
        f"卷积输出 {c.t('g_kda_conv_out')} {c.sh('g_kda_conv_out')} 只取最后 seqlen 步，"
        f"再 split/view 成 q/k/v {c.sh('g_kda_q')}。",
        "depthwise、causal（只看左侧 Kconv-1 个历史 + 当前）；conv state 大小固定，"
        "不随 kvcachelen 增长。"])


def _kda_gate(c: _C) -> str:
    lead = (f"forget gate：f_a/f_b 低秩投影得到 {c.sh('g_kda_f_b')}，加 dt_bias "
            f"{c.sh('g_kda_g_raw')}，再算 g = {c.p('GateLB')} * sigmoid(exp(A_log) * g_raw) "
            f"= {c.t('g_kda_g')} {c.sh('g_kda_g')}（范围 (-5,0)）。")
    return c.wrap("kda_gate", lead, [
        f"beta = sigmoid(b_proj(x)) = {c.t('g_kda_beta')} {c.sh('g_kda_beta')}；"
        "g 是每 token/头/通道的 log decay，beta 是新 value 的写入门控。",
        "这两个量都属于 KDA 的 delta-rule，不是 MLA/DSA 概念。"])


def _kda_scan(c: _C) -> str:
    lead = (f"delta-rule 扫描：q/k 在 kernel 内 L2Norm（{c.t('g_kda_q_hat')}、"
            f"{c.t('g_kda_k_hat')}），q 再乘 {c.p('DhL')}^-0.5；状态 "
            f"{c.t('g_kda_state_pre')} {c.sh('g_kda_state_pre')} 与 q/k/v/g/beta 一起进入 "
            "<code>scan</code>。")
    return c.wrap("kda_scan", lead, [
        "真实更新：先衰减 A = S_{t-1} ⊙ exp(g_t)；mem = A^T k_t；"
        "delta = beta_t ⊙ (v_t - mem)；S_t = A + k_t delta^T；o_t = S_t^T q_t。"
        f"误差项用已衰减状态 A（不是未衰减 S_{{t-1}}），方向是 A^T k；输出 "
        f"{c.t('g_kda_core_out')} {c.sh('g_kda_core_out')} 与本次新状态 "
        f"{c.t('g_kda_state_new')} {c.sh('g_kda_state_new')}。",
        "recurrent 仅在 cache 已有状态且 seq_len==1 时逐 token 执行；空缓存首次调用"
        "（含 P=0/S=1）与多 token 续接走 chunk 分块（chunk_size=64，含 pad 与三角 mask）"
        "——同一递推的不同执行形式，不是普通 Attention。",
        "新状态经 <code>update_recurrent_state</code> 写回 cache（copy_ 到同一静态地址）；"
        "状态大小固定，不随 kvcachelen 增长。"])


def _kda_out(c: _C) -> str:
    lead = (f"输出门 g_a/g_b 得到 {c.t('g_kda_gate')} {c.sh('g_kda_gate')}；"
            f"gated RMSNorm 把 {c.t('g_kda_core_out')} 按 head_dim 归一后乘 sigmoid(gate)，"
            f"得 {c.t('g_kda_o_normed')} {c.sh('g_kda_o_normed')}。")
    return c.wrap("kda_out", lead, [
        f"拼头 {c.sh('g_kda_o_flat')} 后 o_proj 回 {c.t('g_attn_out')} "
        f"{c.sh('g_attn_out')}；这是 KDA 的 gated output norm，不是普通 RMSNorm。"])


def _dsa_q(c: _C) -> str:
    lead = (f"MLA 的 Q 低秩：q_a_proj {c.sh('g_dsa_q_a')} → q_a_layernorm（RMSNorm，"
            f"eps={c.p('EpsN')}）{c.t('g_dsa_q_residual')} {c.sh('g_dsa_q_residual')} → "
            f"q_b_proj {c.sh('g_dsa_q_flat')} → view/transpose {c.t('g_dsa_q')} "
            f"{c.sh('g_dsa_q')}。")
    return c.wrap("dsa_q", lead, [
        f"本模型是 **NoPE**（qk_rope_head_dim={c.p('Rk')}）：没有 q_pass/q_rot split，"
        "也没有 apply_rotary_pos_emb；Q 的最后一维就是 nope 256 维。",
        f"q_residual 同时被 {c.m('dsa_indexer', 'DSA indexer')} 的 wq_b 复用。"])


def _dsa_kv(c: _C) -> str:
    lead = (f"压缩 KV latent：kv_a_proj_with_mqa 输出 {c.sh('g_dsa_kv_compressed')} "
            f"（{c.p('Rkv')}+{c.p('Rk')} 维），拆成 latent {c.sh('g_dsa_kv_pass')} 与 "
            f"k_rot {c.sh('g_dsa_k_rot')}；latent 过 kv_a_layernorm（eps={c.p('EpsN')}）"
            f"得 {c.t('g_dsa_k_pass_normed')} {c.sh('g_dsa_k_pass_normed')}。")
    return c.wrap("dsa_kv", lead, [
        f"NoPE：k_rot 是 0 维；源码仍保留这个槽位并 view 成 {c.sh('g_dsa_k_rot_4d')}。",
        f"归一化后单独 view 成单头 4D：{c.t('g_dsa_k_pass_4d')} {c.sh('g_dsa_k_pass_4d')}，"
        "缓存写入的是这个 4D 张量，不是 3D latent。"])


def _dsa_cache(c: _C) -> str:
    lead = (f"缓存写入：update(k_pass, k_rot) 把**压缩 latent** 追加到 "
            f"{c.t('g_cache_kv_post')} {c.sh('g_cache_kv_post')} 与 "
            f"{c.t('g_cache_krot_post')} {c.sh('g_cache_krot_post')}；每 token 每层只存 "
            f"{c.p('Rkv')}+{c.p('Rk')} 个值，不是 64 头展开后的 K/V。")
    return c.wrap("dsa_cache", lead, [
        f"使用前才对**全部 Lkv 个位置**做 kv_b_proj 展开：{c.sh('g_dsa_kv_nope_flat')} → "
        f"{c.t('g_dsa_kv_nope')} {c.sh('g_dsa_kv_nope')} → 拆出 k_nope {c.sh('g_dsa_k_nope')} "
        f"与 value {c.sh('g_dsa_v')} {c.sh('g_dsa_v')}。",
        f"单头 k_rot（0 维）广播到 {c.p('Nh')} 头后与 k_nope 拼成 key {c.t('g_dsa_key')} "
        f"{c.sh('g_dsa_key')}；NoPE 时 key 实际等于 k_nope，但源码的拼接语义保留。"])


def _dsa_indexer(c: _C) -> str:
    lead = (f"DSA indexer：wq_b(q_residual) {c.sh('g_dsa_ix_q_flat')}、wk(hidden) "
            f"{c.sh('g_dsa_ix_k_proj_out')}，k_norm 是**带 bias 的 LayerNorm**"
            f"（{c.p('Id')} 维，eps={c.p('EpsA')}）；gate_scores "
            f"{c.sh('g_dsa_ix_gate_scores')} 与 valid 通道一起打包成 "
            f"{c.t('g_dsa_ix_packed')} {c.sh('g_dsa_ix_packed')} 写入 packed 缓存。")
    return c.wrap("dsa_indexer", lead, [
        f"packed 缓存 {c.t('g_dsa_ix_cache_pre')} → {c.t('g_dsa_ix_cache_post')} "
        f"{c.sh('g_dsa_ix_cache_post')}；valid 通道取回后构造 "
        f"{c.t('g_dsa_ix_visible')} {c.sh('g_dsa_ix_visible')}（kv_pos ≤ q_pos & valid）。",
        f"indexer 也是 NoPE（qk_rope_head_dim={c.p('Rk')}），没有 RoPE/split 步骤。",
        f"具体选中哪些池由后续打分决定，是数据相关量；图上不猜具体 token。"])


def _dsa_pool(c: _C) -> str:
    lead = (f"k-pool 压缩：每 {c.p('Kpool')} 个连续 token 用 softmax(gate+"
            f"{c.t('g_dsa_ix_ape')}) 加权成一个候选池 "
            f"{c.t('g_dsa_ix_pool_keys')} {c.sh('g_dsa_ix_pool_keys')}；"
            f"池内 token 不能单独被选中，索引记为 {c.sh('g_dsa_ix_pool_indices')}。")
    return c.wrap("dsa_pool", lead, [
        f"每 query 对池打分：{c.sh('g_dsa_ix_scores')} → ReLU → 头权重加权求和 "
        f"{c.sh('g_dsa_ix_index_scores')}；由 pool_indices 求 pool_end "
        f"{c.t('g_dsa_ix_pool_end')}，再 gather visible 并与 pool_valid 相与得到 "
        f"{c.t('g_dsa_ix_valid_candidates')} {c.sh('g_dsa_ix_valid_candidates')}；"
        f"masked_fill 只消费 valid_candidates → {c.sh('g_dsa_ix_masked_scores')}。",
        "pool 本身只依赖 packed 的 k/gate/valid 与 APE，不直接依赖逐 query visible；"
        "不完整尾池由 tail 单独处理。"])


def _dsa_select(c: _C) -> str:
    lead = (f"池 top-k：每 query 选 {c.p('SelP')} 个池 "
            f"{c.t('g_dsa_ix_selected')} {c.sh('g_dsa_ix_selected')}，再用 "
            f"{c.t('g_dsa_ix_selected_valid')} {c.sh('g_dsa_ix_selected_valid')} "
            f"= valid_candidates.gather(selected) 判断有效性，展开成 "
            f"{c.sh('g_dsa_ix_flat_indices')}；无效池位置按 selected_valid 填 -1。")
    return c.wrap("dsa_select", lead, [
        f"当前不完整池的 tail {c.t('g_dsa_ix_tail_indices')} {c.sh('g_dsa_ix_tail_indices')} "
        f"直接追加，pad 到 {c.sh('g_dsa_ix_padded_indices')}，最后按 query "
        f"{c.t('g_attention_mask')} 做末端过滤得到 "
        f"{c.t('g_dsa_topk_indices')} {c.sh('g_dsa_topk_indices')}；"
        f"K_sel(j) = min(V_q, Ki + (V_q % Kpool))，V_q=P+j+1，是逐 query 值；"
        f"只有 j=seqlen-1 才等于 {c.p('Ksel_last')}，不是固定 2048。",
        "tail 只包含当前 query 所在的不完整池，不会与已选池重复；输出宽度固定 "
        "Ki+Kpool-1=2051，mask 构建时忽略 -1；query padding 整行置 -1。"])


def _dsa_mask(c: _C) -> str:
    lead = (f"用 top-k 索引构造 counts：{c.t('g_dsa_mask_counts')} "
            f"{c.sh('g_dsa_mask_counts')} 在选中位置 scatter_add 1，再 "
            f"{c.t('g_dsa_sparse_mask')} {c.sh('g_dsa_sparse_mask')} = counts.ne(0).unsqueeze(1)。")
    return c.wrap("dsa_mask", lead, [
        f"**True=可见**（被选中的 key）；eager 路径再 where 成 "
        f"{c.t('g_dsa_eager_mask')} {c.sh('g_dsa_eager_mask')}（True→0、False→min）。"
        "这与 ones+scatter(False) 的语义相反，不能混用。",
        "top-k 只能收窄可见范围，不能放大因果范围；selected counts 已排除 -1/越界位置。"])


def _dsa_core(c: _C) -> str:
    lead = (f"QKᵀ×scaling：{c.sh('g_dsa_attn_logits')}（scaling=Dq^-0.5=1/16，NoPE 无 "
            f"mscale）→ 加稀疏 mask {c.sh('g_dsa_attn_masked')} → softmax(fp32) → AV "
            f"{c.sh('g_dsa_attn_head_out')}。")
    return c.wrap("dsa_core", lead, [
        f"AV 之后 eager 仍做 transpose(1,2).contiguous()：{c.t('g_dsa_attn_head_out')} → "
        f"{c.t('g_dsa_attn_heads_seq')} {c.sh('g_dsa_attn_heads_seq')}。",
        f"num_key_value_heads 与 query 头数相同，repeat_kv 是恒等；value 头维 "
        f"{c.sh('g_dsa_v')}。"])


def _dsa_out(c: _C) -> str:
    lead = (f"拼头：{c.t('g_dsa_attn_heads_seq')} {c.sh('g_dsa_attn_heads_seq')} → reshape "
            f"{c.t('g_dsa_attn_flat')} {c.sh('g_dsa_attn_flat')}（Nh×Dv=64×256）→ o_proj "
            f"{c.sh('g_attn_out')}。")
    return c.wrap("dsa_out", lead, [
        "Attention 内部没有逆 RoPE、没有分组输出投影，也没有 sink。"])


def _res_attn(c: _C) -> str:
    lead = (f"Attention 后的 mHC 混合：post 展开 {c.t('g_mhc_attn_expanded')} "
            f"{c.sh('g_mhc_attn_expanded')}，combᵀ 混合 residual 流 "
            f"{c.t('g_mhc_attn_combined')} {c.sh('g_mhc_attn_combined')}，两者相加得 "
            f"{c.t('g_hidden_streams_mid')} {c.sh('g_hidden_streams_mid')}。")
    return c.wrap("res_attn", lead, [
        "这是 4 路流之间的混合（post/comb），不是标量残差相加；不要写成 "
        "hidden + attn_output。"])


def _mhc_ffn(c: _C) -> str:
    lead = (f"FFN 前的 mHC（ffn_hc）：4 路流 {c.t('g_hidden_streams_mid')} 再展平/无权重 "
            f"RMSNorm，投影出 pre/post/comb，Sinkhorn 后坍缩成 "
            f"{c.t('g_mhc_ffn_collapsed')} {c.sh('g_mhc_ffn_collapsed')}。")
    return c.wrap("mhc_ffn", lead, [
        f"attn_hc 与 ffn_hc 是两个独立实例（各自 fn/base/scale），不共享权重；"
        f"comb 形状 {c.sh('g_mhc_ffn_comb')}。"])


def _norm_ffn(c: _C) -> str:
    lead = (f"post_attention_layernorm：{c.t('g_mhc_ffn_collapsed')} → "
            f"{c.t('g_ffn_in')} {c.sh('g_ffn_in')}（RMSNorm，eps={c.p('EpsN')}；"
            "权重与 input_layernorm 独立）。")
    return c.wrap("norm_ffn", lead)


def _dense(c: _C) -> str:
    lead = (f"Dense MLP（本层 mlp_layer_types=dense）：gate/up 各 {c.sh('g_dense_gate')}"
            f"（intermediate_size={c.p('Is')}），先 clamp（gate≤{c.p('Swi')}、up±{c.p('Swi')}）"
            f"再 SiLU(gate)×up，down_proj 回 {c.sh('g_mlp_out')}。")
    return c.wrap("mlp_dense", lead, [
        "SwiGLU clamp 是本模型的真实算子；不能省略或套用无 clamp 的 MLP。"])


def _moe_route(c: _C) -> str:
    lead = (f"MoE 路由：展平 {c.sh('g_moe_flat')} → float32 linear {c.sh('g_router_logits')} "
            f"→ sigmoid {c.sh('g_router_scores')} → 加 e_score_correction_bias（只影响选择）→ "
            f"组分数/组掩码（n_group=1，单组掩码全通过）→ topk {c.sh('g_topk_indices')}。")
    return c.wrap("moe_route", lead, [
        f"组掩码：{c.t('g_group_mask')} {c.sh('g_group_mask')} → expand "
        f"{c.t('g_group_mask_expanded')} {c.sh('g_group_mask_expanded')} → reshape "
        f"{c.t('g_score_mask')} {c.sh('g_score_mask')}；最终必须与 scores_for_choice 同型 "
        "[Ttok,E]。",
        f"权重取 sigmoid 原分数后归一（+1e-20）再乘 routed_scaling_factor={c.p('Scl')}；"
        f"每 token 选 {c.p('K')} 个专家（专家总数 {c.p('E')}）。",
        "打分是 sigmoid，不是 softmax，也不是 sqrt(softplus)。"])


def _moe_expert(c: _C) -> str:
    lead = (f"专家模板（只画一份，不复制 {c.p('E')} 份）：one_hot(top_k_index,E+1) 后 "
            f"permute 得完整分配矩阵 {c.t('g_expert_mask')} {c.sh('g_expert_mask')}；"
            f"sum+nonzero 得命中专家 {c.t('g_expert_hit')} {c.sh('g_expert_hit')}。")
    return c.wrap("moe_expert", lead, [
        f"对每个命中专家，torch.where 给出两个真实索引：top_k_pos "
        f"{c.t('g_expert_topk_pos')} {c.sh('g_expert_topk_pos')} 与 token_idx "
        f"{c.t('g_expert_token_idx')} {c.sh('g_expert_token_idx')}；按 token_idx gather "
        f"token 行 {c.sh('g_token_states')}。",
        f"专家 SwiGLU 先 clamp（gate≤{c.p('Swi')}、up±{c.p('Swi')}）再 silu(gate)*up；"
        f"输出乘 top_k_weights[token_idx, top_k_pos] 后以 token_idx index_add 回 "
        f"{c.t('g_routed_flat')} {c.sh('g_routed_flat')}。",
        "命中专家数、每专家 token 数、具体索引都是数据相关量，保持未知；"
        "不按 288 个专家复制子图。"])


def _moe_shared(c: _C) -> str:
    lead = (f"共享专家（与 routed 并行）：输入是 routed 之前的 residuals "
            f"{c.t('g_ffn_in')}，gate/up {c.sh('g_shared_gate')} → clamp → SiLU×up → "
            f"down_proj → {c.t('g_shared_out')} {c.sh('g_shared_out')}。")
    return c.wrap("moe_shared", lead, [
        f"宽度 = moe_intermediate_size×n_shared_experts = {c.sh('g_shared_act')}；"
        "共享专家也做 SwiGLU clamp。"])


def _moe_merge(c: _C) -> str:
    lead = (f"汇合：routed {c.t('g_routed')} + 共享专家 {c.t('g_shared_out')} = "
            f"{c.t('g_mlp_out')} {c.sh('g_mlp_out')}。")
    return c.wrap("moe_merge", lead)


def _res_ffn(c: _C) -> str:
    lead = (f"FFN 后的 mHC 混合：post 展开 {c.t('g_mhc_ffn_expanded')}，combᵀ 混合 "
            f"{c.t('g_mhc_ffn_combined')}，相加得层输出 {c.t('g_hidden_streams_out')} "
            f"{c.sh('g_hidden_streams_out')}。")
    return c.wrap("res_ffn", lead, [
        "层输出仍是 4 路流；网络末端才用 hc_head 均值收拢并做 Final RMSNorm。"])


_BUILDERS = {
    "mhc_attn": _mhc_attn,
    "attn_norm": _attn_norm,
    "kda_proj": _kda_proj,
    "kda_conv": _kda_conv,
    "kda_gate": _kda_gate,
    "kda_scan": _kda_scan,
    "kda_out": _kda_out,
    "dsa_q": _dsa_q,
    "dsa_kv": _dsa_kv,
    "dsa_cache": _dsa_cache,
    "dsa_indexer": _dsa_indexer,
    "dsa_pool": _dsa_pool,
    "dsa_select": _dsa_select,
    "dsa_mask": _dsa_mask,
    "dsa_core": _dsa_core,
    "dsa_out": _dsa_out,
    "res_attn": _res_attn,
    "mhc_ffn": _mhc_ffn,
    "norm_ffn": _norm_ffn,
    "mlp_dense": _dense,
    "moe_route": _moe_route,
    "moe_expert": _moe_expert,
    "moe_shared": _moe_shared,
    "moe_merge": _moe_merge,
    "res_ffn": _res_ffn,
}


def module_narrative(block: dict, module_id: str) -> str:
    fn = _BUILDERS.get(module_id)
    if fn is None:
        return ""
    return fn(_C(block))


def overview_narrative(block: dict) -> str:
    """Block 总览导语：按本组真实层型组合叙述端到端数据流。"""
    c = _C(block)
    kind = _kind(c)
    if kind == "kda":
        attn = (f"KDA（q/k/v → 因果短卷积 → forget gate/beta → delta-rule 扫描 → "
                f"gated RMSNorm → o_proj）")
        attn_mods = (f"{c.m('kda_proj', '投影')}、{c.m('kda_conv', '短卷积')}、"
                     f"{c.m('kda_gate', 'forget gate')}、{c.m('kda_scan', 'delta-rule 扫描')}、"
                     f"{c.m('kda_out', 'gated norm')}")
    else:
        attn = ("MLA（Q 低秩 + NoPE；KV 只缓存 512 维压缩 latent，用时 kv_b 展开全量）"
                "+ DSA（k-pool 压缩 / 池 top-k / tail / counts 稀疏 mask）")
        attn_mods = (f"{c.m('dsa_q', 'Q 低秩')}、{c.m('dsa_kv', '压缩 KV')}、"
                     f"{c.m('dsa_cache', '缓存/展开')}、{c.m('dsa_indexer', 'indexer 投影')}、"
                     f"{c.m('dsa_pool', 'k-pool 打分')}、{c.m('dsa_select', '池 top-k/tail')}、"
                     f"{c.m('dsa_mask', '稀疏 mask')}、{c.m('dsa_core', 'QK/softmax/AV')}")
    mlp = ("dense MLP（gate/up 宽度 12288，SwiGLU clamp）" if not _moe(c) else
           "TopK MoE（288 专家选 8 + 并行共享专家 2048，SwiGLU clamp）")
    lead = (f"结构级（场景未绑定）：seqlen 与 kvcachelen 保留符号。端到端：4 路流 "
            f"{c.t('g_hidden_streams_in')} {c.sh('g_hidden_streams_in')} → attn_hc 坍缩 → "
            f"input RMSNorm → {attn} → post/comb 混合回 4 路流 → ffn_hc 坍缩 → "
            f"post RMSNorm → {mlp} → post/comb 混合 → 4 路流 "
            f"{c.t('g_hidden_streams_out')} {c.sh('g_hidden_streams_out')}。")
    steps = [
        f"Attention 段：{attn_mods}；mHC 权重/坍缩在 {c.m('mhc_attn', 'attn_hc')}。",
        f"残差段：{c.m('res_attn', 'mHC 混合')} 用 post/comb 把子层输出与 4 路输入流混合，"
        "不是普通单流残差相加。",
        (f"MLP 段：{c.m('mlp_dense', 'dense MLP')} 直接输出；"
         if not _moe(c) else
         f"MoE 段：{c.m('moe_route', '路由')} 选专家，{c.m('moe_expert', '专家模板')} 与 "
         f"{c.m('moe_shared', '共享专家')} 并行，{c.m('moe_merge', '汇合')}相加；")
        + f"{c.m('res_ffn', '第二次 mHC 混合')} 得到 4 路流层输出。",
    ]
    return c.wrap("overview", lead, steps)
