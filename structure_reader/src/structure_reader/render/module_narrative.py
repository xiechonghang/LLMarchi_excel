"""Decoder 模块/总览中文结构讲解（反馈11，修复01）。

事实全部读 phase 的 operators/tensors/parameters；可见名复用 tensor_source_label，
内链仍用语义 id；不存在的 key 直接报错，不静默造对象。
"""
from __future__ import annotations

import html

from structure_reader.render.diagram_source_labels import tensor_source_label
from structure_reader.render.display_names import display_shape, param_label


def _num(c: "_Ctx", pid: str) -> int:
    p = c.params[pid]
    v = p.get("value")
    if v is None:
        raise KeyError(f"narrative: 参数 {pid} 无当前值，不能写死数字")
    return int(v)


def _tensor_variant(phase: dict) -> str | None:
    ids = {o.get("id", "") for o in phase.get("operators", [])}
    if "c_kv_proj" in ids:
        return "hca"
    if "rope_main" in ids:
        return "swa_hash" if "moe_hash_lookup" in ids else "swa"
    if "moe_hash_lookup" in ids:
        return "csa_hash"
    return None


class _Ctx:
    def __init__(self, phase: dict):
        self.phase = phase
        self.pid = phase["id"]
        self.tensors = {t["id"]: t for t in phase["tensors"]}
        self.params = {p["id"]: p for p in phase["parameters"]}
        self.groups = {g["id"]: g for g in _groups(phase)}

    # ---- 形状/值 ----
    def sh(self, tid: str) -> str:
        t = self.tensors.get(tid)
        if t is None:
            raise KeyError(f"narrative: 未知 tensor {tid}")
        return display_shape(t["shape"])

    def v(self, pid: str) -> str:
        p = self.params.get(pid)
        if p is None:
            raise KeyError(f"narrative: 未知参数 {pid}")
        return param_label(pid, p.get("value"))

    def n(self, pid: str) -> int:
        return _num(self, pid)

    # ---- 内联锚点：显示源码别名，链接仍用语义 id ----
    def name(self, tid: str) -> str:
        t = self.tensors.get(tid)
        if t is None:
            raise KeyError(f"narrative: 未知 tensor {tid}")
        label = tensor_source_label(tid, _tensor_variant(self.phase)) or t.get("name") or tid
        return html.escape(label)

    def t(self, tid: str) -> str:
        if tid not in self.tensors:
            raise KeyError(f"narrative: 未知 tensor {tid}")
        return (f'<a class="mono narr-link" title="{html.escape(tid)}" '
                f'href="#t-{self.pid}-{html.escape(tid)}">{self.name(tid)}</a>')

    def p(self, pid: str) -> str:
        if pid not in self.params:
            raise KeyError(f"narrative: 未知参数 {pid}")
        return (f'<a class="mono narr-link" title="{html.escape(pid)}" '
                f'href="#p-{self.pid}-{html.escape(pid)}">{self.v(pid)}</a>')

    def m(self, mid: str, label: str | None = None) -> str:
        return (f'<a class="narr-link" href="#mod-{self.pid}-{html.escape(mid)}">'
                f'{html.escape(label or mid)}</a>')

    def wrap(self, module: str, lead: str, steps: list[str], sub: str = "") -> str:
        lis = "".join(f"<li>{s}</li>" for s in steps)
        return (f'<div class="narr" data-module="{self.pid}/{html.escape(module)}">'
                f'<p class="narr-lead">{lead}</p>'
                + (f'<ol class="narr-steps">{lis}</ol>' if steps else "")
                + sub + "</div>")


def _groups(phase: dict) -> list[dict]:
    from structure_reader.render.diagram import module_groups
    return module_groups(phase)


# ---------------------------------------------------------------------------
def _mhc(c: _Ctx, module: str, src: str, block: str, out: str,
         residual_mod: str, is_attn: bool) -> str:
    pre = "attn" if is_attn else "ffn"
    nhc, H = c.n("Nhc"), c.n("H")
    coef, flat = (2 + nhc) * nhc, nhc * H
    rk = c.n("Rk")
    where = "Attention 前" if is_attn else "MoE 前"
    src_desc = ("本层输入的多流 hidden_streams" if is_attn
                else "本层 Attention 混合输出 attn_out_streams")
    lead = (
        f"本段是{where}的 mHC（多流超连接）权重生成与坍缩。输入是{src_desc} "
        f"{c.t(src)} {c.sh(src)}（当前 {c.v('Nhc')}、{c.v('H')}，展平后 {nhc}×{H}={flat}）；"
        f"学习参数 {c.t(pre + '_hc_fn')} {c.sh(pre + '_hc_fn')}、"
        f"{c.t(pre + '_hc_base')} {c.sh(pre + '_hc_base')}、"
        f"{c.t(pre + '_hc_scale')} {c.sh(pre + '_hc_scale')} 参与生成，"
        f"它们与本次算出的 pre/post/comb 系数不同。本段生成三组系数并把多流压成单流；"
        f"回写由 {c.m(residual_mod, 'mHC 残差混合')} 完成，本段不执行 Attention/MoE、也不回写多流。"
    )
    steps = [
        f"展平 + float + 无权重 RMSNorm（eps={c.p('EpsN')}）：{c.sh(src)} → {c.sh(pre + '_mhc_normed')}。",
        f"F.linear：{c.sh(pre + '_mhc_normed')} × {c.sh(pre + '_hc_fn')}ᵀ → {c.sh(pre + '_mhc_weights')}，"
        f"即 (2+hc_mult)×hc_mult=(2+{nhc})×{nhc}={coef} 个系数（不是 {coef} 路 hidden 流）。",
        f"按最后一维拆成 pre_w {c.sh(pre + '_mhc_pre_w')}、post_w {c.sh(pre + '_mhc_post_w')}、"
        f"comb_w {c.sh(pre + '_mhc_comb_w')}；分别用 base/scale 的对应片段仿射："
        f"pre → sigmoid + {c.p('Eps')}，post → 2·sigmoid，comb → softmax 后做 {c.p('Sit')} 步行/列归一"
        f"（近似双随机，不宣称数值等价）。得到 pre {c.sh(pre + '_mhc_pre')}、post {c.sh(pre + '_mhc_post')}、"
        f"comb {c.sh(pre + '_mhc_comb')}。",
        f"pre 只在本段内部消费：与 {c.t(src)} 逐流相乘并沿流轴求和 → "
        f"{c.t(pre + '_mhc_collapsed')} {c.sh(pre + '_mhc_collapsed')}，进入后续归一/子层。",
        f"本段返回 post {c.sh(pre + '_mhc_post')}、comb {c.sh(pre + '_mhc_comb')}、"
        f"collapsed {c.sh(pre + '_mhc_collapsed')} 三项；pre 不返回。",
        f"回写在 {c.m(residual_mod, 'mHC 残差混合')}：post 与 {c.t(block)} {c.sh(block)} 广播成 "
        f"{c.sh(pre + '_expanded')}；comb 转置后左乘原多流 matmul(comb.T, streams)，两支相加 → "
        f"{c.t(out)} {c.sh(out)}"
        + ("，完成 attn 段闭环并进入 ffn 段。" if is_attn else "，即本层最终输出。"),
    ]
    return c.wrap(module, lead, steps)


def _norm(c: _Ctx, module: str, src: str, out: str, weight: str) -> str:
    lead = (f"{c.m(module, 'RMSNorm')} 把 {c.t(src)} {c.sh(src)} 逐 token 归一："
            f"x·rsqrt(mean(x²)+{c.p('EpsN')})，再乘学习缩放 {c.t(weight)}；"
            f"输出 {c.t(out)} {c.sh(out)}，shape 不变。")
    return c.wrap(module, lead, [])


def _attn_q(c: _Ctx) -> str:
    rk = c.n("Rk")
    lead = (f"Q 支路：{c.t('attn_normed')} {c.sh('attn_normed')} 经低秩投影压到 {c.sh('q_a')}，"
            f"归一后升到 {c.sh('q_head')}，只对每头最后 {rk} 维做部分 RoPE；"
            f"输出 {c.t('q_a_normed')}（q_residual，供 Indexer）与 {c.t('q_rope')} {c.sh('q_rope')}。")
    steps = [
        f"q_a_proj → {c.t('q_a')} {c.sh('q_a')}；q_a_norm（{c.t('q_a_norm_weight')}）→ {c.t('q_a_normed')}。",
        f"q_b_proj → {c.sh('q_head')}，view/transpose；q_b_norm 无权重 → {c.t('q_normed')}。",
        f"compress YaRN（{c.p('ThetaC')}，factor={c.n('Yf')}）对最后 {rk} 维 interleaved 旋转 → {c.t('q_rope')}。",
        f"去向：{c.t('q_rope')} → {c.m('attn_core')}；{c.t('q_a_normed')} → {c.m('attn_indexer')} 的 q 投影。",
    ]
    return c.wrap("attn_q", lead, steps)


def _attn_kv(c: _Ctx) -> str:
    rk = c.n("Rk")
    lead = (f"共享 KV：{c.t('attn_normed')} {c.sh('attn_normed')} → {c.t('kv_proj_out')} "
            f"{c.sh('kv_proj_out')} → view {c.sh('kv_head')} → RoPE（最后 {rk} 维）→ "
            f"{c.t('kv_rope')} {c.sh('kv_rope')}。V4 中 K=V 同一张量，缓存也共用。")
    return c.wrap("attn_kv", lead, [])


def _attn_cache(c: _Ctx) -> str:
    lead = (f"缓存更新：本次 {c.t('kv_rope')} {c.sh('kv_rope')} 接到持久 keys 尾部 → "
            f"{c.t('kv_full')} {c.sh('kv_full')}（kv_sequence_length=min(past_sequence_length, sliding_window-1)+sequence_length，"
            f"{c.p('W')}）。持久 keys 只留最后 sliding_window-1 个位置："
            f"{c.t('cache_keys_post')} {c.sh('cache_keys_post')}。")
    steps = [
        f"源码 :192-201 中 <code>self.values = self.keys</code>（同一存储）、"
        f"<code>self.cumulative_length += key_states.shape[-2]</code>（:197）；"
        f"因此只列 keys 侧，values 是别名，不新增存储。",
        f"累计长度前态 {c.t('cache_cumlen')} {c.sh('cache_cumlen')}；调用后为前态 + sequence_length，"
        f"作为下次窗口/mask 起步（不另造 post 张量）。",
        f"去向：{c.t('kv_full')} → {c.m('attn_core')}；keys_post 等作为下一次调用的前态。",
    ]
    return c.wrap("attn_cache", lead, steps)


def _attn_csa(c: _Ctx) -> str:
    lead = (f"CSA 压缩器：把 {c.t('attn_normed')} {c.sh('attn_normed')} 的 token 按 {c.p('M')} 个一组压缩成"
            f"长程条目，追加到 {c.t('cache_compressed_c_post')} {c.sh('cache_compressed_c_post')}；"
            f"窗口不足（num_full_windows=0）时不产生新条目。同一次调用里包含 Lightning Indexer（见下），"
            f"它从压缩条目中为每个 query 选 top-k。")
    steps = [
        f"投影：→ {c.t('c_kv')} {c.sh('c_kv')} 与 gate；与 {c.t('cache_buffer_c_kv')} "
        f"{c.sh('cache_buffer_c_kv')} 拼接后取 M 对齐前缀，余量回写 buffer。",
        f"窗口：chunk view 成 [batch_size, num_full_windows, compress_rate, 2×head_dim]；后半 Cb 归本窗、前半 Ca 前移；窗 0 前半来自 "
        f"{c.t('cache_overlap_c_kv')}；gate 加 {c.t('c_position_bias')} 后沿 2M 轴 softmax。",
        f"汇聚：softmax(gate)×kv 沿 2M 求和 → {c.sh('c_compressed')}；RMSNorm（{c.t('c_kv_norm_weight')}）"
        f"后按 first_window_position 做 compress RoPE。",
        f"持久化：新条目追加到 {c.t('cache_compressed_c_post')}；entry_count += num_full_windows；overlap 写回最后满窗 Ca 尾"
        f"（仅 num_full_windows>0）。输出给 {c.m('attn_core')} 的 KV 轴，并作为 {c.m('attn_indexer')} block_bias 的列基准。",
    ]
    return c.wrap("attn_csa", lead, steps, sub=_indexer(c))


def _indexer(c: _Ctx) -> str:
    ki = c.n("Ki")
    lead = (f'<div class="narr narr-sub" data-module="{c.pid}/attn_indexer">'
            f'<p class="narr-lead">其中 <b>Lightning Indexer</b>（包含在 CSA 内，不抬成平级）：'
            f"{c.t('attn_normed')}（hidden_states）经 kv_proj/gate_proj 生成<b>独立的压缩键</b> "
            f"{c.t('cache_compressed_i_post')} {c.sh('cache_compressed_i_post')}（与 CSA 压缩条目分属不同 key/轴）；"
            f"q_residual {c.t('q_a_normed')} 经 q_b_proj 生成<b>查询</b> {c.t('i_q')} {c.sh('i_q')}；"
            f"{c.t('attn_normed')} 另经 scorer.weights_proj（{c.t('i_w_weight')} {c.sh('i_w_weight')}）"
            f"得到每个头的权重。query 与压缩键内积后 ReLU/缩放，乘头权重并沿头维归约得到分数，"
            f"再做 top-k（k=min({c.p('Ki')}, 本轴长度)）与 block_bias。</p>"
            '<ol class="narr-steps">'
            f"<li>压缩键：{c.t('attn_normed')} → kv_proj/gate_proj 的窗口压缩（独立于 CSA 条目）→ "
            f"{c.sh('cache_compressed_i_post')}。</li>"
            f"<li>查询与权重：{c.t('q_a_normed')} 经 q_b_proj（{c.t('i_q_weight')} {c.sh('i_q_weight')}）→ "
            f"{c.t('i_q')}；{c.t('attn_normed')} 经 {c.t('i_w_weight')} → 每头权重。</li>"
            f"<li>打分：relu(q·Kᵀ)·index_head_dim^-0.5 × 权重·index_n_heads^-0.5，沿头维求和 → {c.t('i_scores')} {c.sh('i_scores')}。</li>"
            f"<li>top-k：按 causal_threshold=(pos+1)//M 屏蔽未来条目，k=min(index_topk, indexer_entries_before+num_full_windows_indexer)（当前 index_topk={ki}）；"
            f"越界下标置 -1 → {c.t('indexer_topk')}。</li>"
            f"<li>block_bias：按 <b>CSA</b> 的条目数 num_compressed_entries 建 {c.sh('c_block_bias')} 的 -inf，选中列置 0；"
            f"与滑窗 mask 拼成 {c.t('mask_ext')}。CSA 与 Indexer 的条目轴各自独立，不互相等同。</li></ol></div>")
    return lead


def _attn_mask(c: _Ctx) -> str:
    lead = (f"mask 构建：由累计长度 {c.t('cache_cumlen')} 与 {c.p('W')} 生成 {c.t('mask_model')} "
            f"{c.sh('mask_model')}（0 可读 / -inf 屏蔽，含绝对 key 偏移）；再与 block_bias "
            f"{c.sh('c_block_bias')} 拼接成 {c.t('mask_ext')} {c.sh('mask_ext')}。只覆盖默认因果/滑窗，不含 2D padding。")
    return c.wrap("attn_mask", lead, [])


def _attn_core(c: _Ctx) -> str:
    lead = (f"注意力核心：Q {c.sh('q_rope')} 与拼接并复制到 num_attention_heads 头的 KV {c.t('kv_repeated')} "
            f"{c.sh('kv_repeated')} 做 QKᵀ·scale，加 mask 后拼 sink 做 softmax，去 sink 列再对 V 加权，"
            f"输出 {c.t('attn_head_out')} {c.sh('attn_head_out')}。")
    steps = [
        f"concat：{c.sh('kv_full')} + {c.sh('cache_compressed_c_post')} → 本次 KV 轴，复制到 num_attention_heads 头。",
        f"QK：{c.sh('q_rope')} × KVᵀ → {c.sh('qk_logits')}，乘 head_dim^-0.5。",
        f"mask+sink：加 {c.sh('mask_ext')}；sinks reshape+expand 到 [batch_size, num_attention_heads, sequence_length, 1]，拼最后一列；"
        f"减最大值后 softmax，取 [..., :-1] → {c.t('attn_probs')}。",
        f"AV：{c.sh('attn_probs')} × V → transpose → {c.t('attn_head_out')}；去向 {c.m('attn_out')}。",
    ]
    return c.wrap("attn_core", lead, steps)


def _attn_out(c: _Ctx) -> str:
    g = c.n("G")
    lead = (f"输出投影：{c.sh('attn_head_out')} 先 transpose 成 [batch_size, num_attention_heads, sequence_length, head_dim]，用 -sin 做逆 RoPE，再转回；"
            f"reshape 成 {c.sh('attn_grouped')} 后按 o_groups={g} 组做分组线性 → {c.sh('o_a_out')}，"
            f"再 o_b_proj → {c.t('attn_block_output')} {c.sh('attn_block_output')}，进入残差段。")
    return c.wrap("attn_out", lead, [])


def _residual(c: _Ctx, module: str, comb: str, post: str, block: str, base: str,
              expanded: str, out: str) -> str:
    lead = (f"残差回写：{c.t(post)} {c.sh(post)} 与 {c.t(block)} {c.sh(block)} 广播成 "
            f"{c.t(expanded)} {c.sh(expanded)}；{c.t(comb)} {c.sh(comb)} 转置后左乘 {c.t(base)} "
            f"{c.sh(base)}；两支相加得 {c.t(out)} {c.sh(out)}。这是 mHC 闭环的回写半段。")
    return c.wrap(module, lead, [])


def _moe_route(c: _Ctx) -> str:
    lead = (f"路由：{c.t('ffn_flat')} {c.sh('ffn_flat')} × {c.t('router_weight')} {c.sh('router_weight')}ᵀ → "
            f"{c.t('router_logits')} {c.sh('router_logits')}；sqrt(softplus) 后加 {c.t('router_bias')}，"
            f"TopK 选 {c.p('K')} 个专家 → indices/weights "
            f"[batch_size×sequence_length, num_experts_per_tok]。"
            f"注意 bias 只参与“选哪些专家”（indices）；最终权重是从<b>未加 bias 的 scores</b> gather 再归一 × "
            f"routed_scaling={c.p('Scl')}。")
    return c.wrap("moe_route", lead, [])


def _moe_expert(c: _Ctx) -> str:
    lim = c.p("Lim")
    lead = (f"单专家模板（只展开一个命中专家，不复制 n_routed_experts 份）：从 expert_mask 取该专家的行，nonzero 得 "
            f"token 下标与 top-k 槽位；选中 token {c.sh('expert_tokens')} 过本专家 "
            f"{c.t('expert_gate_up_e')} {c.sh('expert_gate_up_e')} → {c.sh('expert_gu_out')}；"
            f"拆成 gate/up {c.sh('expert_gate_c')}；<b>gate 只 clamp 上界 Lim</b>，"
            f"<b>up 范围 [-Lim, Lim]</b>（{lim}）；SiLU(gate)·up → 过 {c.t('expert_down_e')} "
            f"{c.sh('expert_down_e')} → {c.sh('expert_down_raw')}，乘路由权重后 scatter-add 回 "
            f"{c.t('routed_out')} {c.sh('routed_out')}，view 成 {c.t('routed_view')} {c.sh('routed_view')}。")
    return c.wrap("moe_expert", lead, [])


def _moe_shared(c: _Ctx) -> str:
    lim = c.p("Lim")
    lead = (f"共享专家（与专家模板并行）：{c.sh('ffn_normed')} → gate/up {c.sh('shared_gate')} / "
            f"{c.sh('shared_up')}（clamp：gate 只上界 Lim，up 范围 [-Lim, Lim]；{lim}）；SiLU(gate)·up → "
            f"{c.sh('shared_act')}，down → {c.t('shared_out')} {c.sh('shared_out')}。")
    return c.wrap("moe_shared", lead, [])


def _moe_merge(c: _Ctx) -> str:
    lead = (f"MoE 汇合：加权专家输出 {c.t('routed_view')} {c.sh('routed_view')} 与共享专家 "
            f"{c.t('shared_out')} {c.sh('shared_out')} 逐元素相加 → {c.t('mlp_output')} {c.sh('mlp_output')}，"
            f"进入 ffn 段 mHC 残差。")
    return c.wrap("moe_merge", lead, [])


def overview_narrative(phase: dict) -> str:
    """总览端到端短讲解（不新增图/表）。"""
    c = _Ctx(phase)
    kv_len = c.params.get("kvcachelen", {}).get("value")
    seq_len = c.params.get("seqlen", {}).get("value")
    seq_text = ("seqlen 保留符号（由实际调用决定）" if seq_len is None
                else f"seqlen={seq_len}")
    if kv_len is None and seq_len is None:
        premise = ("结构级（场景未绑定）：seqlen 与 kvcachelen 都保留符号；"
                   "顶部场景条绑定取值后，窗口/缓存规模在场景面板给出，结构本身不随场景改变。")
    elif kv_len == 0:
        premise = (f"空缓存 Prefill：kvcachelen=0（调用前无历史），{seq_text}；"
                   "mask/压缩窗口规模由场景参数派生。")
    elif kv_len is None:
        premise = ("非零逻辑历史：kvcachelen 保留符号（调用前已处理的逻辑历史 token 数），"
                   f"{seq_text}；cache 前态已含历史，闭窗与否由 kvcachelen 与窗口宽推导。")
    else:
        premise = (f"非零逻辑历史：kvcachelen={kv_len}，{seq_text}；"
                   "cache 前态已含历史，闭窗与否由 kvcachelen 与窗口宽推导。")
    ids = {o.get("id", "") for o in phase.get("operators", [])}
    if "c_kv_proj" in ids:
        attn = "HCA 压缩器（非重叠 128，无 Indexer）"
    elif "c_window_view" in ids:
        attn = "CSA/Indexer"
    else:
        attn = "SWA 主 KV（无压缩器/Indexer）"
    route = "Hash 查表路由" if "moe_hash_lookup" in ids else "TopK 路由"
    lead = (f"{premise} 端到端：输入 {c.t('hidden_streams_in')} {c.sh('hidden_streams_in')} 先经 mHC 坍缩与归一，"
            f"进入 Attention 的 Q/KV 并行支路（{attn} 与主 KV 缓存并行接入），"
            f"再经输出投影与残差回写成多流；随后第二次 mHC 坍缩进入 MoE（{route} + 单专家模板 + 并行共享专家），"
            f"汇合后经第二次（最后一次）残差回写得到 {c.t('hidden_streams_out')} {c.sh('hidden_streams_out')}。")
    attn_parts = [c.m("attn_q"), c.m("attn_kv")]
    for mid, label in (("attn_csa", "压缩器"), ("attn_indexer", "Indexer")):
        if mid in c.groups:
            attn_parts.append(c.m(mid))
    attn_txt = "、".join(attn_parts)
    steps = [
        f"Attention 段：{attn_txt} 并行；"
        f"缓存前态经 {c.m('attn_cache')} 与 {c.m('attn_mask')} 接入 {c.m('attn_core')}，"
        f"再由 {c.m('attn_out')} 投影。",
        f"残差段：{c.m('res_attn')} 用 post/comb 把 Attention 单流广播回多流，得 {c.sh('attn_out_streams')}。",
        f"MoE 段：{c.m('moe_route')} 选专家，{c.m('moe_expert')} 单专家模板与 {c.m('moe_shared')} 并行，"
        f"{c.m('moe_merge')} 相加；{c.m('res_ffn')} 再次回写。",
    ]
    return c.wrap("overview", lead, steps)


_BUILDERS = {
    "norm_attn": lambda c: _norm(c, "norm_attn", "attn_mhc_collapsed", "attn_normed", "attn_norm_weight"),
    "norm_ffn": lambda c: _norm(c, "norm_ffn", "ffn_mhc_collapsed", "ffn_normed", "ffn_norm_weight"),
    "mhc_attn": lambda c: _mhc(c, "mhc_attn", "hidden_streams_in", "attn_block_output",
                               "attn_out_streams", "res_attn", True),
    "mhc_ffn": lambda c: _mhc(c, "mhc_ffn", "attn_out_streams", "mlp_output",
                              "hidden_streams_out", "res_ffn", False),
    "attn_q": _attn_q,
    "attn_kv": _attn_kv,
    "attn_cache": _attn_cache,
    "attn_csa": _attn_csa,
    "attn_indexer": lambda c: "",
    "attn_mask": _attn_mask,
    "attn_core": _attn_core,
    "attn_out": _attn_out,
    "res_attn": lambda c: _residual(c, "res_attn", "attn_mhc_comb", "attn_mhc_post",
                                    "attn_block_output", "hidden_streams_in", "attn_expanded",
                                    "attn_out_streams"),
    "res_ffn": lambda c: _residual(c, "res_ffn", "ffn_mhc_comb", "ffn_mhc_post",
                                   "mlp_output", "attn_out_streams", "ffn_expanded",
                                   "hidden_streams_out"),
    "moe_route": _moe_route,
    "moe_expert": _moe_expert,
    "moe_shared": _moe_shared,
    "moe_merge": _moe_merge,
}


def _variant(phase: dict) -> str:
    ids = {o.get("id", "") for o in phase.get("operators", [])}
    if "c_kv_proj" in ids:
        return "hca"
    if "rope_main" in ids:
        return "swa"
    if "moe_hash_lookup" in ids:
        return "hash"
    return "csa"


def _attn_q_swa(c: _Ctx) -> str:
    rk = c.n("Rk")
    lead = (f"Q 支路（SWA）：{c.t('attn_normed')} {c.sh('attn_normed')} → {c.t('q_a')} "
            f"{c.sh('q_a')} → {c.t('q_head')} {c.sh('q_head')}，只对末 {rk} 维做 main RoPE；"
            f"输出 {c.t('q_rope')} {c.sh('q_rope')}。无 Indexer 消费 q_residual。")
    steps = [f"main RoPE（theta={c.n('ThetaM')}）→ {c.t('q_rope')}；无 compress YaRN。",
             f"去向：{c.t('q_rope')} → {c.m('attn_core')}。"]
    return c.wrap("attn_q", lead, steps)


def _attn_cache_swa(c: _Ctx) -> str:
    lead = (f"SWA 缓存（DynamicSlidingWindowLayer）：{c.t('kv_rope')} {c.sh('kv_rope')} 同时作为 "
            f"K 与 V（数值相等），分别写入持久 {c.t('cache_keys')} 与 {c.t('cache_values')} "
            f"（独立字段，不是同一存储）；返回 full {c.t('kv_full')} {c.sh('kv_full')}。")
    steps = [f"写后持久 tail：{c.t('cache_keys_post')} {c.sh('cache_keys_post')}、"
             f"{c.t('cache_values_post')} {c.sh('cache_values_post')}，"
             f"长度 min(past_sequence_length+sequence_length, sliding_window-1)；"
             f"mask 只用主 KV 的 kv_sequence_length 轴，无压缩条目。",
             f"去向：{c.t('kv_full')} → {c.m('attn_core')}；无 kv_concat。"]
    return c.wrap("attn_cache", lead, steps)


def _attn_q_hca(c: _Ctx) -> str:
    rk = c.n("Rk")
    lead = (f"Q 支路（HCA）：{c.t('attn_normed')} {c.sh('attn_normed')} → {c.t('q_a')} "
            f"{c.sh('q_a')} → {c.t('q_head')} {c.sh('q_head')}，末 {rk} 维 compress RoPE → "
            f"{c.t('q_rope')} {c.sh('q_rope')}；HCA 无 Indexer，q_residual 不参与选择。")
    steps = [f"去向：{c.t('q_rope')} → {c.m('attn_core')}。"]

    return c.wrap("attn_q", lead, steps)


def _attn_kv_variant(c: _Ctx) -> str:
    rk = c.n("Rk")
    lead = (f"共享 KV：{c.t('attn_normed')} {c.sh('attn_normed')} → {c.t('kv_main')} "
            f"{c.sh('kv_main')}（Linear+RMSNorm，单 KV 头）→ view {c.sh('kv_head')} → "
            f"末 {rk} 维 RoPE → {c.t('kv_rope')} {c.sh('kv_rope')}；K=V 同一张量。")
    return c.wrap("attn_kv", lead, [])


def _attn_mask_swa(c: _Ctx) -> str:
    lead = (f"mask（SWA）：{c.t('mask_model')} {c.sh('mask_model')} 只覆盖主 KV，"
            f"无 block_bias、无压缩条目扩展。")
    return c.wrap("attn_mask", lead, [])


def _attn_csa_hca(c: _Ctx) -> str:
    lead = (f"HCA 压缩器（非重叠 128）：输入 {c.t('attn_normed')} {c.sh('attn_normed')} 与 "
            f"kv/gate 权重 [head_dim, hidden_size] 投影得 {c.t('c_kv')} {c.sh('c_kv')}、"
            f"{c.t('c_gate')} {c.sh('c_gate')}；与旧 buffer 合并后按 {c.p('M')} 个 token 一组 "
            f"view {c.sh('c_windows_kv')}；"
            f"gate+position_bias 沿窗口 softmax 聚合→归一→条目 RoPE→追加 "
            f"{c.t('cache_compressed_c_post')} {c.sh('cache_compressed_c_post')}。无 Indexer/overlap。")
    steps = [f"条目位置来自 first_window_position + arange(n_windows)×compress_rate"
             f"（{c.t('c_positions')} {c.sh('c_positions')}），不是 query position_ids。",
             f"num_full_windows=0 时不产生新条目，仍返回此前全量历史；"
             f"block_bias 在 query_length==1 或 compressed_len==0 时为 None，否则按 "
             f"entry_indices >= (position_ids+1)//compress_rate 置 -inf。",
             f"去向：{c.t('cache_compressed_c_post')} → {c.m('attn_core')} 的 KV 拼接。"]

    return c.wrap("attn_csa", lead, steps)


def _attn_mask_hca(c: _Ctx) -> str:
    lead = (f"mask：{c.t('mask_model')} {c.sh('mask_model')} 只覆盖主 KV；"
            f"HCA 压缩条目另由 {c.t('block_bias')}（非 None 时）拼接，"
            f"None（query_length==1 或 compressed_len==0）时 F.pad 零扩展。")
    return c.wrap("attn_mask", lead, [])


def _attn_core_swa(c: _Ctx) -> str:
    lead = (f"核心：QKᵀ·scale {c.t('qk_logits')} {c.sh('qk_logits')} → 加 sliding mask → "
            f"cat sink→softmax→去 sink → AV（{c.t('kv_repeated')}，kv_sequence_length 轴，无压缩条目）→ "
            f"{c.t('attn_head_out')}。")
    return c.wrap("attn_core", lead, [])


def _moe_route_hash(c: _Ctx) -> str:
    lead = (f"Hash 路由：{c.t('input_ids')} {c.sh('input_ids')} reshape 后查持久表 "
            f"{c.t('tid2eid')} {c.sh('tid2eid')} 得 {c.t('topk_indices')} {c.sh('topk_indices')}；"
            f"另一路 learned weight[batch×sequence_length, n_routed_experts] 打分得 "
            f"{c.t('router_scores')} {c.sh('router_scores')}；两路按 indices gather→weights.sum 归一"
            f"(+1e-20)→×routed_scaling_factor。无 TopK 分数选择、无 correction bias。")
    return c.wrap("moe_route", lead, [])


_VARIANT_BUILDERS = {
    ("swa", "attn_q"): _attn_q_swa,
    ("swa", "attn_cache"): _attn_cache_swa,
    ("swa", "attn_core"): _attn_core_swa,
    ("swa", "attn_mask"): _attn_mask_swa,
    ("hca", "attn_q"): _attn_q_hca,
    ("hca", "attn_kv"): _attn_kv_variant,
    ("swa", "attn_kv"): _attn_kv_variant,
    ("hca", "attn_csa"): _attn_csa_hca,
    ("hca", "attn_mask"): _attn_mask_hca,
    ("hash", "moe_route"): _moe_route_hash,
    ("swa", "moe_route"): _moe_route_hash,
}


def module_narrative(phase: dict, module_id: str) -> str:
    c = _Ctx(phase)
    fn = _VARIANT_BUILDERS.get((_variant(phase), module_id))
    if fn is None:
        fn = _BUILDERS.get(module_id)
    return fn(c) if fn else ""
