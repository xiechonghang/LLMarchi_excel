"""Qwen3.8-Flash-Next 模块/总览中文结构讲解（事实全部读 Block 文档）。

- 只读 ``operators`` / ``tensors`` / ``parameters``，不引入模型之外的结论；
- 可见名复用 ``display_shape`` 的完整名称，内链使用当前 Block 的真实作用域 id
  （``block-<组id>``），与参数表/张量表/模块详情同源；
- 引用不存在的张量/参数直接抛 KeyError（不静默造对象，避免讲错结构）；
- 文本按「GDN/QSA × PLE/MoE」分支：Gated Residual 4 路流、GDN 固定状态、
  QSA micro-block indexer、N-gram PLE、softmax MoE 分别叙述，不套其它模型机制。
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
    return "gdn" if c.block.get("kind") == "gdn" else "qsa"


def _ple(c: _C) -> bool:
    return bool(c.block.get("ple"))


def _ple_ngram(c: _C) -> str:
    lead = (f"N-gram PLE 先取上一 token 上下文 {c.t('g_ple_context_pre')} "
            f"{c.sh('g_ple_context_pre')}（state_idx=2，初始 eos），与当前 ids 拼成 "
            f"{c.t('g_ple_history')} {c.sh('g_ple_history')}；右移 1/2 位时 EOS 会切断"
            " n-gram（跨段位置填 eos）。")
    return c.wrap("ple_ngram", lead, [
        f"16 个哈希头（bigram 8 + trigram 8）：token×每层乘子后 XOR，再按每头素数码表"
        f"取模并加偏移，得到 {c.t('g_ple_ids')} {c.sh('g_ple_ids')}；"
        f"每头 {c.p('Dgh')} 维，拼成 {c.p('Dple')}。",
        f"查表 {c.t('g_ple_emb')} 后只取当前 seqlen；历史 2 位只参与哈希，不产生自己的 PLE 输出。",
        f"码表总大小按下界 {c.p('Vng')} 后的 16 个素数求和再 pad；这是固定配置量，不是运行时路由。"])


def _ple_gate(c: _C) -> str:
    lead = (f"PLE 把 n-gram embedding 投影成 key {c.sh('g_ple_key_normed')} 与 value "
            f"{c.sh('g_ple_value')}；query 是当前 4 路流 {c.sh('g_ple_query_normed')}。"
            f"每流 gate = (key·query)/√H = {c.t('g_ple_gate')} {c.sh('g_ple_gate')}，"
            f"再做 abs/clamp/sqrt/sign 得 {c.t('g_ple_gate_signed')}。")
    return c.wrap("ple_gate", lead, [
        f"value 广播到 4 路，乘 sigmoid(gate) 得 {c.t('g_ple_gated_value')} "
        f"{c.sh('g_ple_gated_value')}；先 flatten 回 10240，再分组 RMSNorm 后进 "
        f"dilation=3、kernel=4 的 depthwise 短卷积（state 9 步）。",
        f"输出 = flatten 后的 gated value + flat 短卷积输出（{c.t('g_ple_out')} "
        f"{c.sh('g_ple_out')}），再加回 flat 4 路流 {c.t('g_hidden_streams_ple')}；"
        "PLE 是层输入的前置加性支路，不是替换 hidden，也不是普通 attention。"])


def _mhc_attn(c: _C) -> str:
    lead = (f"Attention 前的 Gated Residual（attn_hc）：4 路流 {c.t('g_hidden_streams_in')} "
            f"{c.sh('g_hidden_streams_in')} 是 flat 层间边界；分组 RMSNorm 在 flat 上做，"
            f"经 10240→{c.p('Rhc')}→10240 + sigmoid 得到**逐元素读门** "
            f"{c.t('g_hc_attn_read')} {c.sh('g_hc_attn_read')}。")
    return c.wrap("mhc_attn", lead, [
        f"读门先 unflatten 成 [B,S,4,H] 再与 normed 的 4D 视图相乘；四路流按读门加权后"
        f"取 mean，坍缩成 {c.t('g_hc_attn_mixed')} "
        f"{c.sh('g_hc_attn_mixed')} 进入 GDN/QSA；这不是普通单流残差，也不是"
        "多路混合残差的 pre/post/comb 模板。",
        f"block_inject 10240→4 得到每支路标量写门 {c.t('g_hc_attn_inject')} "
        f"{c.sh('g_hc_attn_inject')}（2×sigmoid(·/4)），供 Attention 输出写回 4 路流。"])


def _gdn_proj(c: _C) -> str:
    lead = (f"GDN 先把 padding 位置置零（{c.t('g_conv_mask')} 是 2D padding mask，"
            f"不是因果 mask），再用一个线性层出 q/k/v {c.t('g_gdn_qkv_flat')} "
            f"{c.sh('g_gdn_qkv_flat')}；同时投影输出门 z、beta、decay logits。")
    return c.wrap("gdn_proj", lead, [
        f"qkv 宽度是 {c.p('Dqkv')} = 2×{c.p('NkH')}×{c.p('Dk')} + {c.p('NvH')}×{c.p('DvH')}；"
        "转置成 [B,10240,S] 供 depthwise 短卷积。",
        f"z 是 {c.p('Dz')} 维、beta/decay 各 {c.p('NvH')} 维；它们不是 attention 的 Q/K/V，"
        "不能套用 QSA 的 q/gate 结构。"])


def _gdn_conv(c: _C) -> str:
    lead = (f"卷积前更新 conv state {c.t('g_gdn_conv_state_pre')} "
            f"{c.sh('g_gdn_conv_state_pre')}：保存最近 {c.p('Kconv')} 个 token 的 10240 通道；"
            f"首次调用左补零，续接时与前态拼接（cache_utils:1099-1149）。")
    return c.wrap("gdn_conv", lead, [
        f"depthwise 因果短卷积 + SiLU 得到 {c.t('g_gdn_conv_out')} "
        f"{c.sh('g_gdn_conv_out')}，只取最后 seqlen 步；转置后 split 成 q/k "
        f"{c.sh('g_gdn_q')} 与 v {c.sh('g_gdn_v')}。",
        "q/k 是 16 头、v 是 48 头；因果性由卷积只看左侧 + 递推顺序保证，没有 4D causal mask。"])


def _gdn_gate(c: _C) -> str:
    lead = (f"beta = sigmoid(b) = {c.t('g_gdn_beta')} {c.sh('g_gdn_beta')}；"
            f"g = -exp(A_log)·softplus(a+dt_bias) = {c.t('g_gdn_g')} {c.sh('g_gdn_g')}，"
            "是每 token/头的 log decay（≤0）。")
    return c.wrap("gdn_gate", lead, [
        f"q/k 先 repeat_interleave {c.p('Gmul')} 次从 16 头到 48 头"
        f"（{c.t('g_gdn_q_rep')} / {c.t('g_gdn_k_rep')}）；同头复制不是权重共享。",
        "beta 是 delta-rule 的写入门控；g 控制旧状态的衰减。两者都属于 GDN，不是 QSA 概念。"])


def _gdn_scan(c: _C) -> str:
    lead = (f"q/k 在 kernel 内 L2Norm（{c.t('g_gdn_q_hat')}、{c.t('g_gdn_k_hat')}），"
            f"q 再乘 {c.p('Dk')}^-0.5；状态 {c.t('g_gdn_state_pre')} "
            f"{c.sh('g_gdn_state_pre')} 与 q/k/v/g/beta 进入 gated delta-rule 扫描。")
    return c.wrap("gdn_scan", lead, [
        "真实更新：先衰减 A = S_{t-1} ⊙ exp(g_t)；mem = A^T k_t；"
        "delta = beta_t ⊙ (v_t - mem)；S_t = A + k_t delta^T；o_t = S_t^T q_t。"
        "误差项用已衰减状态 A（不是未衰减 S_{t-1}），方向是 A^T k。"
        f"输出 {c.t('g_gdn_core_out')} {c.sh('g_gdn_core_out')} 与新状态 "
        f"{c.t('g_gdn_state_new')} {c.sh('g_gdn_state_new')}。",
        "recurrent 仅在 cache 已有状态且 seq_len==1 时逐 token 执行；空缓存首次调用"
        "（含 P=0/S=1）与多 token 续接走 chunk 分块（chunk_size=64，含 pad 与三角 mask）"
        "——同一递推的不同执行形式，不是普通 Attention。",
        f"新状态经 <code>update_recurrent_state</code> 写回 cache；状态大小固定，"
        "不随 kvcachelen 增长。"])


def _gdn_out(c: _C) -> str:
    lead = (f"Gated RMSNorm：{c.t('g_gdn_core_out')} 按 head_dim 归一后乘 sigmoid(z)，"
            f"得 {c.t('g_gdn_o_normed')} {c.sh('g_gdn_o_normed')}；"
            f"拼头 {c.sh('g_gdn_o_flat')} 后 o_proj 回 {c.t('g_attn_out')}。")
    return c.wrap("gdn_out", lead, [
        f"输出门 activation 来自 config.output_gate_type=sigmoid；不是 silu，也不是普通 RMSNorm。",
        "o_proj 6144→2560；之后由 attn_hc 的写门把单流输出展开回 4 路流。"])


def _qsa_q(c: _C) -> str:
    lead = (f"QSA 的 q_proj 一次输出 2×{c.p('Nh')}×{c.p('Dh')}=12288 维"
            f"（{c.t('g_qsa_q_flat')}），按头 view 后 chunk 成 q 与 gate："
            f"q {c.sh('g_qsa_q_raw')}、gate {c.sh('g_qsa_gate_raw')}。")
    return c.wrap("qsa_q", lead, [
        f"q 做 RMSNorm({c.p('Dh')}) 得 {c.t('g_qsa_q_normed')}，再转成 "
        f"{c.sh('g_qsa_q')}；gate 拼成 {c.sh('g_qsa_gate')}，最后逐元素 sigmoid 乘 attention 输出。",
        f"{c.p('Nh')} 个 query 头、{c.p('Nkv')} 个 KV 头；GQA 由后续 repeat_kv 复制 {c.p('Nrep')} 份。"])


def _qsa_kv(c: _C) -> str:
    lead = (f"K/V 各投影到 2×256=512 维；K 做 RMSNorm({c.p('Dh')})，V 不归一。"
            f"K {c.sh('g_qsa_k')}、V {c.sh('g_qsa_v')} 都是 2 头。")
    return c.wrap("qsa_kv", lead, [
        f"K/V 是逐层独立的 2 头缓存；attention 时 repeat 到 {c.p('Nh')} 头，"
        "缓存本身仍只存 2 头。",
        "V 没有 q_norm/k_norm；不要把 GDN 的 Gated RMSNorm 或 MLA 的低秩 KV 套到这里。"])


def _qsa_rope(c: _C) -> str:
    lead = (f"partial RoPE：只旋转 q/k 的前 {c.p('Drot')} 维（256×0.25），"
            f"后 192 维 nope 原样 cat 回；q {c.t('g_qsa_q_rope')}、k {c.t('g_qsa_k_rope')}。")
    return c.wrap("qsa_rope", lead, [
        f"cos/sin 来自 4 行 position_ids 的 mRoPE 重组；text-only 时三行 grid 相同，"
        "等价于同一组 1D RoPE 的前 64 维。",
        "RoPE 是 partial：只转前 64 维，其余 192 维是 nope 段；不是全维 RoPE。视觉网格 M-RoPE 不在本票范围。"])


def _qsa_cache(c: _C) -> str:
    lead = (f"主 K/V 缓存 update：把当前 K/V 追加到 {c.t('g_qsa_k_post')} / "
            f"{c.t('g_qsa_v_post')}，长度 Lkv={c.p('Lkv')}；QSA 读取全部 Lkv 做 masked softmax。")
    return c.wrap("qsa_cache", lead, [
        f"缓存随历史线性增长（与 GDN 的固定 recurrent state 不同）；"
        "indexer 的 selected mask 只收窄可见范围，不删除 K/V。",
        f"每个 QSA 层独立缓存；{len(c.block.get('layer_indices', [])) if c.block.get('layer_indices') else ''}"
        "本组层不共享权重或缓存。"])


def _qsa_indexer(c: _C) -> str:
    lead = (f"QSA indexer：index_qk_proj 输出 640 维（{c.sh('g_qsa_ix_qk')}），"
            f"拆成 {c.p('Iqh')} 个 query 头与 {c.p('Ikh')} 个共享 key 头；"
            f"q/k 各 RMSNorm({c.p('Id')})，q 前 {c.p('Drot')} 维做 RoPE。")
    return c.wrap("qsa_indexer", lead, [
        f"逐 token raw key 写入 indexer 缓存 {c.t('g_qsa_indexer_keys_post')} "
        f"{c.sh('g_qsa_indexer_keys_post')}；缓存的是**未池化** key，block key 在每次调用内"
        "由均值 + norm + RoPE 得到。",
        f"可见 token {c.t('g_qsa_ix_visible')} 来自主 causal mask；indexer 只在该集合内"
        "选块，不重新定义因果性。"])


def _qsa_pool(c: _C) -> str:
    lead = (f"micro-block：每 {c.p('Kc')} 个连续可见 token 取均值压成一块"
            f"（{c.t('g_qsa_ix_pooled_keys')} {c.sh('g_qsa_ix_pooled_keys')}）；"
            f"块 key 做 RMSNorm 后用块首位置做前 {c.p('Drot')} 维 RoPE。")
    return c.wrap("qsa_pool", lead, [
        f"分数 = relu(q·k).sum(-1)/sqrt({c.p('Id')})（{c.t('g_qsa_ix_scores')} "
        f"{c.sh('g_qsa_ix_scores')}）：4 个 indexer 头求和，不是 softmax，也不是加权 gate。",
        f"块内 token 不能单独被选中；每个 query 的完整块数是 N_q=floor(V_q/{c.p('Kc')})，"
        "不是全长 N_pool；N_q=0 时 pool/norm/RoPE/score 分支不执行，全部可见 token 走 tail。"])


def _qsa_select(c: _C) -> str:
    lead = (f"每 query 对块分数 top-k：最多 {c.p('Btop')} 块 = {c.p('Ki')} token"
            f"（{c.t('g_qsa_ix_block_idx')} {c.sh('g_qsa_ix_block_idx')}）；"
            "展开成 token 后追加不完整 tail。")
    return c.wrap("qsa_select", lead, [
        f"top-k 宽度是逐 query 的 B_q=min({c.p('Btop')},N_q)，不是全长 B_sel；"
        f"选择索引 pad 到 {c.p('Kout')}=2051（空位 -1），scatter 成 bool "
        f"{c.t('g_qsa_ix_mask_bool')} {c.sh('g_qsa_ix_mask_bool')}；True=可见。",
        f"K_sel 是逐 query 量：min(V_q, {c.p('Ki')}+V_q%{c.p('Kc')})；"
        f"最后一个 query 的 {c.p('Ksel_last')} 不等于所有 query 的统一值，"
        "也不是 2048 固定常数。"])


def _qsa_mask(c: _C) -> str:
    lead = (f"selected mask 先做 eager 转换：bool {c.t('g_qsa_ix_mask_bool')} "
            f"→ {c.t('g_qsa_ix_mask_float')}（True→0.0、False→finfo.min），"
            f"再与 causal mask 相加得 {c.t('g_qsa_mask_combined')} "
            f"{c.sh('g_qsa_mask_combined')}。")
    return c.wrap("qsa_mask", lead, [
        "源码 :766-769 先 where(True,0,min)，:848-851 才做 float 相加；"
        "不能把 True=可见的 bool mask 直接加到 float causal mask。",
        "selected mask 只能收窄 causal mask：被因果屏蔽的位置不会因为被 indexer 选中而变可见。",
        "QSA 的稀疏性来自这里的 mask，不是删除 K/V 或改缓存长度。"])


def _qsa_core(c: _C) -> str:
    lead = (f"eager attention：K/V repeat_kv {c.p('Nrep')} 份到 {c.p('Nh')} 头，"
            f"QKᵀ×{c.p('Dh')}^-0.5 后加合并 mask，再 softmax(fp32)、AV。")
    return c.wrap("qsa_core", lead, [
        f"softmax 仍在全部 Lkv 上；非选中位置是 min，概率为 0。"
        f"Dk 输出 {c.sh('g_qsa_attn_flat')}。",
        "attention 核心是标准 GQA + mask，不是 GDN 的递推，也不是 MLA 的压缩 latent。"])


def _qsa_out(c: _C) -> str:
    lead = (f"输出先乘 sigmoid(gate)（{c.t('g_qsa_gated')} {c.sh('g_qsa_gated')}），"
            f"再 o_proj 6144→2560；gate 与 attention 输出同宽，逐元素作用。")
    return c.wrap("qsa_out", lead, [
        "没有额外缩放/softmax；输出门来自同一个 q_proj 的 gate 段。",
        "之后由 attn_hc 的写门把单流输出展开回 4 路流。"])


def _res_attn(c: _C) -> str:
    lead = (f"Attention 输出按写门 {c.t('g_hc_attn_inject')} 展开到 4D "
            f"（{c.t('g_hc_attn_expanded')}），再显式 flatten 成 "
            f"{c.t('g_hc_attn_expanded_flat')}，加到 attn_hc 返回的原 flat 4 路流 "
            f"{c.t('g_hidden_streams_in')}，得 {c.t('g_hidden_streams_mid')}。")
    return c.wrap("res_attn", lead, [
        "写门是每支路标量，不是逐元素 [4,H]；读门才是逐元素。",
        "层间边界是 flat [B,S,Hc*H]；4D 广播结果必须先 flatten 才能与 flat 残差相加。",
        "这不是普通标量残差：残差底座是原 4 路流，不是坍缩后的单流。"])


def _mhc_ffn(c: _C) -> str:
    lead = (f"FFN 前的 Gated Residual（ffn_hc）：对 {c.t('g_hidden_streams_mid')} "
            f"{c.sh('g_hidden_streams_mid')} 做同样的分组 norm / 低秩读门 / 加权 mean，"
            f"得到 {c.t('g_hc_ffn_mixed')} {c.sh('g_hc_ffn_mixed')} 进入 MoE。")
    return c.wrap("mhc_ffn", lead, [
        f"ffn_hc 与 attn_hc 是两份独立权重；结构同构但参数不同。",
        f"写门 {c.t('g_hc_ffn_inject')} {c.sh('g_hc_ffn_inject')} 供 MoE 输出写回 4 路流。"])


def _moe_route(c: _C) -> str:
    lead = (f"MoE 路由：展平 {c.t('g_moe_flat')} {c.sh('g_moe_flat')} 后 linear 到 "
            f"{c.p('E')} 个专家，softmax 得 {c.t('g_router_probs')}；"
            f"每 token 选 {c.p('K')} 个并归一化。")
    return c.wrap("moe_route", lead, [
        f"没有 correction bias、没有 routed scaling factor；这是 softmax 路由，"
        "不是 sigmoid 打分 + correction bias 的路由。",
        f"top-{c.p('K')} 权重和为 1；路由索引是数据相关量，图上不猜具体专家。"])


def _moe_expert(c: _C) -> str:
    lead = (f"专家模板只有一份：one-hot 分配矩阵 {c.t('g_expert_mask')} "
            f"{c.sh('g_expert_mask')}（E+1 的最后一槽跳过），按 (top_k_pos, token_idx) "
            f"gather 命中 token；专家宽度 {c.p('Es')}。")
    return c.wrap("moe_expert", lead, [
        f"SwiGLU 是 silu(gate)*up，**没有 clamp**（与带 clamp 的变体不同）；"
        "down 后乘该 token 对该专家的路由权重，再 index_add 回 token 行。",
        f"命中专家数 N_hit 与每专家 token 数 N_e 都是数据相关量，保留未知；"
        "不按 512 个专家复制图。"])


def _moe_shared(c: _C) -> str:
    lead = (f"共享专家：{c.p('Ess')} 宽 SwiGLU，与 routed 专家并行；"
            f"另有 2560→1 的 sigmoid 标量门 {c.t('g_shared_gate_scalar')} "
            f"{c.sh('g_shared_gate_scalar')}。")
    return c.wrap("moe_shared", lead, [
        "共享专家不参与 top-k 路由；它的输出先乘自己的标量门，再与 routed 输出相加。",
        "共享专家是独立权重，不是第 513 个路由专家。"])


def _moe_merge(c: _C) -> str:
    lead = (f"MoE 汇合：routed 输出 + 门控后的共享专家输出（{c.t('g_mlp_flat')}），"
            f"再 reshape 回 {c.t('g_mlp_out')} {c.sh('g_mlp_out')}。")
    return c.wrap("moe_merge", lead, [
        "父组与子项不重复计数：本模块只表示单次 MoE 的汇合，不汇总整网成本。",
        "之后由 ffn_hc 的写门展开回 4 路流。"])


def _res_ffn(c: _C) -> str:
    lead = (f"FFN/MoE 输出按写门 {c.t('g_hc_ffn_inject')} 展开并 flatten 成 "
            f"{c.t('g_hc_ffn_expanded_flat')}，再加到 flat "
            f"{c.t('g_hidden_streams_mid')}，得到层输出 {c.t('g_hidden_streams_out')}。")
    return c.wrap("res_ffn", lead, [
        "层输出仍是 flat 4 路流；下一层继续用 attn_hc/ffn_hc，不在这里收拢。",
        "网络末端才由 hyper_connection_mixer 加权 mean 成单流，且没有 Final RMSNorm。"])


_MODULES = {
    "ple_ngram": _ple_ngram,
    "ple_gate": _ple_gate,
    "mhc_attn": _mhc_attn,
    "gdn_proj": _gdn_proj,
    "gdn_conv": _gdn_conv,
    "gdn_gate": _gdn_gate,
    "gdn_scan": _gdn_scan,
    "gdn_out": _gdn_out,
    "qsa_q": _qsa_q,
    "qsa_kv": _qsa_kv,
    "qsa_rope": _qsa_rope,
    "qsa_cache": _qsa_cache,
    "qsa_indexer": _qsa_indexer,
    "qsa_pool": _qsa_pool,
    "qsa_select": _qsa_select,
    "qsa_mask": _qsa_mask,
    "qsa_core": _qsa_core,
    "qsa_out": _qsa_out,
    "res_attn": _res_attn,
    "mhc_ffn": _mhc_ffn,
    "moe_route": _moe_route,
    "moe_expert": _moe_expert,
    "moe_shared": _moe_shared,
    "moe_merge": _moe_merge,
    "res_ffn": _res_ffn,
}


def module_narrative(block: dict, module_id: str) -> str:
    fn = _MODULES.get(module_id)
    if fn is None:
        raise KeyError(f"narrative: 未知模块 {module_id}")
    return fn(_C(block))


def overview_narrative(block: dict) -> str:
    c = _C(block)
    kind = "GDN" if _kind(c) == "gdn" else "QSA"
    ple = " + N-gram PLE" if _ple(c) else ""
    lead = (f"本 Block 是 <b>{kind}{ple} + TopK MoE</b>：输入是 4 路 Gated Residual 流 "
            f"{c.t('g_hidden_streams_in')}；attn_hc 读门加权 mean 成单流，"
            f"{kind} 输出由写门展开回原 4 路流；ffn_hc 再做一次同样的读门/写门，"
            f"最后进入 MoE。")
    steps = [
        f"读门是逐元素 [4,H]、写门是每支路标量 [4]；两者都不是普通残差系数。",
        f"GDN 状态固定（conv 40,960 + recurrent 786,432 每 batch）；"
        if _kind(c) == "qsa" else
        f"QSA 的 K/V + indexer key 随历史增长；indexer 只选块，不删除缓存。",
        "专家只有一份模板，不按 512 个专家复制；数据相关路由量保留未知。",
    ]
    if _ple(c):
        steps.insert(0, "PLE 在 attn_hc 之前把 n-gram 特征加回 4 路流；上下文 [B,2] 与短卷积 state 9 步都是固定状态。")
    return c.wrap("overview", lead, steps)
