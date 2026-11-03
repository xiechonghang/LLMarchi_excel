"""GLM-5.3 模块/总览中文结构讲解（事实全部读 Block 文档）。

- 只读 ``operators`` / ``tensors`` / ``parameters``，不引入模型之外的结论；
- 可见名复用 ``display_shape`` 的完整名称（qk_head_dim 等），内链使用当前
  Block 的真实作用域 id（``block-<组id>``），与参数表/张量表/模块详情同源；
- 引用不存在的张量/参数直接抛 KeyError（不静默造对象，避免讲错结构）；
- 文本按「层型组合」分支：full/shared indexer、dense/TopK MoE 分别叙述。
"""
from __future__ import annotations

import html

from structure_reader.render.display_names import display_shape, param_label

__all__ = ["module_narrative", "overview_narrative"]


class _C:
    def __init__(self, block: dict):
        self.block = block
        # 页面渲染前会把 Block id 作用域化为 block-<组id>；讲解/引用链接必须用
        # 同一真实 id，否则 #p-/t-/mod- 会缺组命名空间并退回 network。
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


def _idx(c: _C) -> str:
    return c.block.get("indexer_type", "")


def _moe(c: _C) -> str:
    return c.block.get("moe_type", "")


def _norm_attn(c: _C) -> str:
    lead = (f"层输入 {c.t('g_hidden_in')} {c.sh('g_hidden_in')} 先做 RMSNorm（eps="
            f"{c.p('EpsN')}，fp32 方差归一后乘可学习 weight）：输出 "
            f"{c.t('g_attn_normed')} {c.sh('g_attn_normed')}。")
    return c.wrap("glm_attn_norm", lead, [
        "残差用的是归一化前的 g_hidden_in，不是归一化后的张量；本模型没有 mHC "
        "多流/门控混合，第一次残差在 Attention 之后做普通加法。"])


def _attn_q(c: _C) -> str:
    lead = (f"Q 支路：{c.t('g_attn_normed')} {c.sh('g_attn_normed')} → q_a_proj "
            f"{c.sh('g_q_a')} → q_a_layernorm（RMSNorm，源码默认 eps=1e-6）→ "
            f"{c.t('g_q_residual')} {c.sh('g_q_residual')} → q_b_proj "
            f"{c.sh('g_q_flat')} → view/transpose {c.sh('g_q')}。")
    return c.wrap("glm_attn_q", lead, [
        f"按头维拆成 nope {c.sh('g_q_pass')} 与 rope {c.sh('g_q_rot')}；只有 rope 段做"
        f"**交错** RoPE（cos/sin 前 {c.p('Rk')}÷2 个角，按 0::2/1::2 成对旋转），"
        f"再拼回 {c.t('g_q_full')} {c.sh('g_q_full')}。",
        f"q_residual 同时被 {c.m('glm_attn_indexer', 'DSA indexer')} 的 wq_b 复用"
        if _idx(c) == "full" else
        "本层是 shared indexer：不实例化 indexer，q_residual 只进主注意力。"])


def _attn_kv(c: _C) -> str:
    lead = (f"共享 KV 支路（MLA）：{c.t('g_attn_normed')} → kv_a_proj_with_mqa "
            f"{c.sh('g_kv_compressed')}，拆成 latent {c.sh('g_kv_pass')} 与 k_rot "
            f"{c.sh('g_k_rot')}；latent 过 kv_a_layernorm（eps=1e-6）得 "
            f"{c.t('g_k_pass_normed')} {c.sh('g_k_pass_normed')}。")
    return c.wrap("glm_attn_kv", lead, [
        f"归一化后单独 view 成单头 4D：{c.t('g_k_pass_normed')} → "
        f"{c.t('g_k_pass_4d')} {c.sh('g_k_pass_4d')}，缓存 update 消费这个 4D 张量；"
        "不能把 3D latent 直接当 4D 缓存输入。",
        f"k_rot 做与 Q 相同的交错 RoPE，得 {c.t('g_k_rot_emb')} {c.sh('g_k_rot_emb')}；"
        "kv_a_proj_with_mqa 只输出 512+64 维，这是 MLA 压缩 KV 的入口。"])


def _attn_cache(c: _C) -> str:
    lead = (f"缓存写入：update(k_pass, k_rot) 把 **压缩 latent** 追加到 "
            f"{c.t('g_cache_k_pass_post')} {c.sh('g_cache_k_pass_post')} 与 "
            f"{c.t('g_cache_k_rot_post')} {c.sh('g_cache_k_rot_post')}；"
            "每 token 每层只存 512+64 个值，不是 64 头展开后的 K/V。")
    return c.wrap("glm_attn_cache", lead, [
        f"使用前才对**全部 Lkv 个位置**做 kv_b_proj 展开：{c.sh('g_kv_nope_flat')} → "
        f"{c.t('g_kv_nope')} {c.sh('g_kv_nope')} → 拆出 k_nope {c.sh('g_k_nope')} 与 "
        f"value {c.sh('g_v')} {c.sh('g_v')}。",
        f"单头 k_rot 广播到 {c.p('Nh')} 头后与 k_nope 拼成 key "
        f"{c.t('g_key')} {c.sh('g_key')}；K=V 不共享：key 是 192+64，value 是 256。",
        f"Lkv = kvcachelen + seqlen（{c.p('Lkv')}）；本模型无滑动窗口/压缩条目，"
        "缓存线性保留全部历史，所以每次调用都会重算全量 kv_b 展开。"])


def _indexer(c: _C) -> str:
    if _idx(c) != "full":
        return c.wrap("glm_attn_indexer",
                      "本层是 shared indexer：不创建 indexer 模块、没有 indexer 键缓存；"
                      f"直接消费 {c.t('g_prev_topk_indices')} {c.sh('g_prev_topk_indices')}"
                      "（上一层 full indexer 的 top-k，层间单向传递）。")
    lead = (f"本层是 full indexer：独立轻量投影 wq_b(q_residual) {c.sh('g_ix_q_flat')} 与 "
            f"wk({c.t('g_attn_normed')}) {c.sh('g_ix_k_proj')}；"
            f"k_norm 是**带 bias 的 LayerNorm**（{c.p('Id')} 维，不是 RMSNorm）。")
    return c.wrap("glm_attn_indexer", lead, [
        f"q/k 各自拆 64/64 做**交错** RoPE 后拼回；k 以 [B,Lidx,128] 追加进 indexer 键缓存 "
        f"{c.t('g_cache_indexer_keys_post')} {c.sh('g_cache_indexer_keys_post')}，"
        "shared 层没有这份缓存。",
        f"打分 relu(q·kᵀ×128^-0.5) {c.sh('g_ix_relu')} → 头权重加权求和 "
        f"{c.sh('g_ix_index_scores')} → 加因果 mask → topk(min(index_topk,Lidx)) 得 "
        f"{c.t('g_ix_topk_indices')} {c.sh('g_ix_topk_indices')}。",
        f"K_sel = min({c.p('Ki')}, Lidx)：Lidx 不超过 2048 时全选（掩码不删任何位置），"
        "超过时只保留 2048 个。"])


def _mask(c: _C) -> str:
    source = ("本层 top-k" if _idx(c) == "full" else
              "上一层 full indexer 的 top-k")
    lead = (f"用{source}构造 index_mask：`new_ones` 先全 True，再把选中位置 scatter 成 "
            f"False（True=屏蔽）→ unsqueeze(1) 得 {c.t('g_index_mask')} "
            f"{c.sh('g_index_mask')}（4D）。")
    return c.wrap("glm_attn_mask", lead, [
        f"与网络级因果 mask 合并为 {c.t('g_sparse_mask')} {c.sh('g_sparse_mask')}："
        "float 路径 `masked_fill(index_mask, min)` 把 True 位置置 min，"
        "False（被 top-k 选中）位置保留因果 mask 的值；bool 路径是 `& ~index_mask`，语义等价。",
        "top-k 只能收窄可见范围，不能放大因果范围；eager 路径仍计算与 Lkv 等长的 QKᵀ，"
        "只是被 mask 掉未选中的位置（不是跳过计算）。"])


def _core(c: _C) -> str:
    lead = (f"QKᵀ×scaling：{c.sh('g_attn_logits')}（scaling=qk_head_dim^-0.5=1/16，"
            f"rope_type=default 不乘 YaRN mscale）→ 加稀疏 mask {c.sh('g_attn_masked')} → "
            f"softmax(fp32) → AV {c.sh('g_attn_head_out')} {c.sh('g_attn_head_out')}。")
    return c.wrap("glm_attn_core", lead, [
        f"AV 之后 eager 仍做 transpose(1,2).contiguous()：{c.t('g_attn_head_out')} "
        f"[{c.sh('g_attn_head_out')}] → {c.t('g_attn_heads_seq')} "
        f"{c.sh('g_attn_heads_seq')}；把头维移到 seqlen 之后，下一步才能按 seqlen reshape。",
        f"num_key_value_heads={c.p('Nkv')} 与 query 头数相同，repeat_kv 是恒等（不复制 KV 头）；"
        f"value 头维 {c.sh('g_v')}，所以 AV 输出是 256 维而不是 192+64。"])


def _out(c: _C) -> str:
    lead = (f"拼头：{c.t('g_attn_heads_seq')} {c.sh('g_attn_heads_seq')} → reshape "
            f"{c.t('g_attn_flat')} {c.sh('g_attn_flat')}（Nh×Dv=64×256）→ o_proj "
            f"{c.sh('g_attn_out')}。")
    return c.wrap("glm_attn_out", lead, [
        "reshape 的输入是已经 transpose(1,2).contiguous 的 [B,seqlen,Nh,Dv]；"
        "Attention 内部没有逆 RoPE、没有分组输出投影，也没有 sink。"])


def _res_attn(c: _C) -> str:
    lead = (f"第一次残差：{c.t('g_hidden_in')} + {c.t('g_attn_out')} = "
            f"{c.t('g_hidden_mid')} {c.sh('g_hidden_mid')}。这是普通加法，"
            "没有 mHC 的 post/comb 门控与多流回写。")
    return c.wrap("glm_res_attn", lead)


def _norm_ffn(c: _C) -> str:
    lead = (f"post_attention_layernorm：{c.t('g_hidden_mid')} → "
            f"{c.t('g_ffn_in')} {c.sh('g_ffn_in')}（RMSNorm，eps={c.p('EpsN')}；"
            "权重与 input_layernorm 独立）。")
    return c.wrap("glm_norm_ffn", lead)


def _dense(c: _C) -> str:
    lead = (f"Dense MLP（本层 mlp_layer_types=dense）：gate/up 各 "
            f"{c.sh('g_dense_gate')}（intermediate_size=12288，不是专家宽度 2048），"
            f"SiLU(gate)×up 后 down_proj 回 {c.sh('g_mlp_out')}。")
    return c.wrap("glm_mlp_dense", lead, [
        "本模型没有 swiglu_limit clamp；hidden_act=silu 直接作用。"])


def _moe_route(c: _C) -> str:
    lead = (f"MoE 路由：展平 {c.sh('g_moe_flat')} → float32 linear {c.sh('g_router_logits')} "
            f"→ sigmoid {c.sh('g_router_scores')} → 加 e_score_correction_bias（只影响选择）→ "
            f"组分数/组掩码（n_group=1，单组掩码全通过）→ topk {c.sh('g_topk_indices')}。")
    return c.wrap("glm_moe_route", lead, [
        f"组掩码：{c.t('g_group_mask')} {c.sh('g_group_mask')}（scatter 选中组=1）→ expand "
        f"{c.t('g_group_mask_expanded')} {c.sh('g_group_mask_expanded')} → reshape "
        f"{c.t('g_score_mask')} {c.sh('g_score_mask')}；最终必须与 scores_for_choice 同型 "
        "[Ttok,E]，masked_fill 才不会广播错维。",
        f"权重取 sigmoid 原分数后归一（+1e-20）再乘 routed_scaling_factor={c.p('Scl')}："
        f"{c.sh('g_topk_weights')}。每 token 选 {c.p('K')} 个专家（专家总数 {c.p('E')}）。",
        "打分是 sigmoid，不是 softmax，也不是 sqrt(softplus)；组路由在 n_group=1 时"
        "只是源码会执行的一段等价分支。"])


def _moe_expert(c: _C) -> str:
    lead = (f"专家模板（只画一份，不复制 {c.p('E')} 份）：one_hot(top_k_index,E+1) 后 "
            f"permute 得完整分配矩阵 {c.t('g_expert_mask')} {c.sh('g_expert_mask')}；"
            f"sum+nonzero 得命中专家 {c.t('g_expert_hit')} {c.sh('g_expert_hit')}。")
    return c.wrap("glm_moe_expert", lead, [
        f"对每个命中专家，`torch.where(expert_mask[expert_idx])` 给出两个真实索引："
        f"top_k_pos {c.t('g_expert_topk_pos')} {c.sh('g_expert_topk_pos')} 与 token_idx "
        f"{c.t('g_expert_token_idx')} {c.sh('g_expert_token_idx')}；"
        f"按 token_idx gather token 行 {c.sh('g_token_states')}。",
        f"专家输出乘 top_k_weights[token_idx, top_k_pos]（两个索引都要；只用命中集合无法确定"
        f"乘哪个槽），再以 token_idx index_add 回 {c.t('g_routed_flat')} "
        f"{c.sh('g_routed_flat')}。",
        "gate/up 在权重里打包成 [E,2Es,H]、down 是 [E,H,Es]；按专家索引切片，"
        "图的形状保留 E 维，不按 256 个专家复制子图。",
        f"命中专家数、每专家 token 数、top_k_pos/token_idx 的具体取值都是数据相关量，"
        "保持未知，不根据权重推断实际路由、不猜具体 token。"])


def _moe_shared(c: _C) -> str:
    lead = (f"共享专家（与 routed 并行，不是串行）：输入是 routed 之前的 residuals "
            f"{c.t('g_ffn_in')}，gate/up {c.sh('g_shared_gate')} → SiLU×up → down_proj "
            f"→ {c.t('g_shared_out')} {c.sh('g_shared_out')}。")
    return c.wrap("glm_moe_shared", lead, [
        f"宽度 = moe_intermediate_size×n_shared_experts = {c.sh('g_shared_act')}，"
        "比 routed 单专家宽，但不需要路由。"])


def _moe_merge(c: _C) -> str:
    lead = (f"汇合：routed {c.t('g_routed')} + 共享专家 {c.t('g_shared_out')} = "
            f"{c.t('g_mlp_out')} {c.sh('g_mlp_out')}，随后进入第二次普通残差。")
    return c.wrap("glm_moe_merge", lead)


def _res_ffn(c: _C) -> str:
    lead = (f"第二次残差：{c.t('g_hidden_mid')} + {c.t('g_mlp_out')} = "
            f"{c.t('g_hidden_out')} {c.sh('g_hidden_out')}，即本层输出。")
    return c.wrap("glm_res_ffn", lead, [
        "两处残差都是普通加法；层内没有额外的流混合/归一化，也没有把 Attention "
        "输出回写成多流。"])


_BUILDERS = {
    "glm_attn_norm": _norm_attn,
    "glm_attn_q": _attn_q,
    "glm_attn_kv": _attn_kv,
    "glm_attn_cache": _attn_cache,
    "glm_attn_indexer": _indexer,
    "glm_attn_mask": _mask,
    "glm_attn_core": _core,
    "glm_attn_out": _out,
    "glm_res_attn": _res_attn,
    "glm_norm_ffn": _norm_ffn,
    "glm_mlp_dense": _dense,
    "glm_moe_route": _moe_route,
    "glm_moe_expert": _moe_expert,
    "glm_moe_shared": _moe_shared,
    "glm_moe_merge": _moe_merge,
    "glm_res_ffn": _res_ffn,
}


def module_narrative(block: dict, module_id: str) -> str:
    fn = _BUILDERS.get(module_id)
    if fn is None:
        return ""
    return fn(_C(block))


def overview_narrative(block: dict) -> str:
    """Block 总览导语：按本组真实层型组合叙述端到端数据流。"""
    c = _C(block)
    idx = _idx(c)
    moe = _moe(c)
    indexer = ("full DSA indexer（本层运行 indexer 并写 [B,L,128] 键缓存）"
               if idx == "full" else
               "shared DSA indexer（复用上一层 full 层的 top-k，无 indexer 模块/缓存）")
    mlp = ("dense MLP（gate/up 宽度 12288）" if moe == "dense" else
           "TopK MoE（256 专家选 8 + 并行共享专家 2048）")
    lead = (f"结构级（场景未绑定）：seqlen 与 kvcachelen 保留符号。端到端："
            f"{c.t('g_hidden_in')} {c.sh('g_hidden_in')} → input RMSNorm → "
            f"MLA（Q 低秩+交错部分 RoPE；KV 只缓存 512+64 压缩 latent，用时 kv_b 展开全量）"
            f" + {indexer} → 稀疏 mask → QKᵀ/softmax/AV → o_proj → 第一次普通残差 → "
            f"post RMSNorm → {mlp} → 第二次普通残差 → {c.t('g_hidden_out')} "
            f"{c.sh('g_hidden_out')}。")
    steps = [
        f"Attention 段：{c.m('glm_attn_q', 'Q 支路')}、{c.m('glm_attn_kv', '压缩 KV 支路')}"
        + (f"、{c.m('glm_attn_indexer', 'DSA indexer')}" if idx == "full" else "")
        + f"并行；缓存写入在 {c.m('glm_attn_cache', '缓存/展开')}，稀疏 mask 在 "
          f"{c.m('glm_attn_mask', '掩码')}，核心在 {c.m('glm_attn_core', 'QK/softmax/AV')}。",
        f"残差段：{c.m('glm_res_attn', '普通相加')} 把 Attention 输出加回层输入——"
        "本模型没有 mHC 多流/门控混合。",
        (f"MLP 段：{c.m('glm_mlp_dense', 'dense MLP')} 直接输出；"
         if moe == "dense" else
         f"MoE 段：{c.m('glm_moe_route', '路由')} 选专家，{c.m('glm_moe_expert', '专家模板')}"
         f"与 {c.m('glm_moe_shared', '共享专家')} 并行，{c.m('glm_moe_merge', '汇合')}相加；")
        + f"{c.m('glm_res_ffn', '第二次残差')} 得到层输出。",
    ]
    return c.wrap("overview", lead, steps)
