"""HiDream-O1-Image 模块/总览中文结构讲解（事实全部读 Block 文档）。

- 只读 ``operators`` / ``tensors`` / ``parameters``，不引入模型之外的结论；
- 可见名复用 ``display_shape``；内链使用当前 Block 的真实作用域 id
  （``block-<组 id>``），与参数表/张量表/模块详情同源；
- 文本按 Block kind 分支：``uit_decoder``（统一 Transformer，两趟注意力 + SwiGLU +
  条件 DeepStack）与 ``vit_tower``（LayerNorm + qkv 合并 + GELU-tanh + DeepStack tap）
  分别叙述，不套其它模型机制。
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
        return display_shape(self.tensors.get(tid, {}).get("shape", []))

    def t(self, tid: str) -> str:
        name = self.tensors.get(tid, {}).get("name", tid)
        return (f'<a class="mono narr-link" title="{html.escape(tid)}" '
                f'href="#t-{self.pid}-{html.escape(tid)}">{html.escape(str(name))}</a>')

    def p(self, pid: str) -> str:
        param = self.params.get(pid)
        if param is None:
            return html.escape(pid)
        return (f'<a class="mono narr-link" title="{html.escape(pid)}" '
                f'href="#p-{self.pid}-{html.escape(pid)}">'
                f'{param_label(pid, param.get("value"))}</a>')

    def m(self, mid: str, label: str) -> str:
        return (f'<a class="narr-link" href="#mod-{self.pid}-{html.escape(mid)}">'
                f'{html.escape(label)}</a>')

    def wrap(self, module: str, lead: str, steps: list[str] | None = None) -> str:
        lis = "".join(f"<li>{s}</li>" for s in (steps or []))
        return (f'<div class="narr" data-module="{self.pid}/{html.escape(module)}">'
                f'<p class="narr-lead">{lead}</p>'
                + (f'<ol class="narr-steps">{lis}</ol>' if lis else "")
                + "</div>")


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------


def overview_narrative(block: dict) -> str:
    c = _C(block)
    kind = str(block.get("kind"))
    if kind == "uit_decoder":
        lead = (f"本图是<b>模块导航</b>（虚线框，非计算算子）：一个统一 Transformer "
                f"Block 内，{c.t('d_hidden_in')} {c.sh('d_hidden_in')} 先经 "
                f"{c.m('d_norm_attn', 'RMSNorm')}，再进入 Q/K/V 三条<b>并行</b>投影支路；"
                f"注意力拆成两趟：{c.m('d_attn_ar', 'AR 子集因果')}与"
                f"{c.m('d_attn_full', '全序列双向')}，由"
                f"{c.m('d_attn_merge', 'index_copy')}合成一个输出。")
        steps = [
            f"Q/K 各自做逐头 RMSNorm（{c.sh('d_q_head')}）后施加 MRoPE；V 不做 norm、不做 RoPE。",
            f"AR 行只看 AR 行（{c.sh('d_score_ar')}），生成行看全部 token（{c.sh('d_score_full')}）；"
            "这是<b>行级差异</b>，不是两种层型。",
            f"拼接 32 头后 o_proj，加到 {c.t('d_hidden_in')} 上（标量残差），再经第二个 RMSNorm 进 "
            f"{c.m('d_ffn_mlp', 'SwiGLU MLP')}；FFN 残差同样加回单流。",
            f"参考图任务且 layer 0..2 时，{c.m('d_deepstack', 'DeepStack 注入')}在条件图占位行上"
            "再加视觉特征；t2i 或其它层不执行（恒等）。",
            "层内没有 MoE、没有 KV cache、没有时间步调制/adaLN；时间步在整网层以 tms token 位置替换表达。",
        ]
        return c.wrap("overview", lead, steps)
    lead = (f"本图是<b>视觉塔</b>一个 block 的模块导航：{c.t('v_hidden_in')} {c.sh('v_hidden_in')} 经 "
            f"{c.m('v_norm_attn', 'LayerNorm')} 后做一次合并的 "
            f"{c.m('v_attn_qkv', 'qkv 投影（3456，bias）')}，拆成 q/k/v；q/k 施加 2D rotary，"
            f"再按 {c.t('v_cu_seqlens')} <b>逐图切段</b>，段内做非因果注意力，最后逐段拼回。")
    steps = [
        f"每段 score 是 [Nhv,S_i,S_i]（第 i 张图自己的 S_i），总规模是 Σ_i Nhv·S_i²，"
        "不是把所有图拼成一个 (Σ S_i)² 的矩阵。",
        f"注意力输出按头拼回 1152 维后 proj，加到输入上（标量残差）。",
        f"第二个 LayerNorm 后进 {c.m('v_ffn_mlp', 'GELU-tanh MLP（4304）')}，残差加回。",
        f"block 8/16/24 之后的 {c.t('v_tap')} 会再被 DeepStack merger 读取；主线继续流向下一 block。",
        "视觉塔只在参考图/编辑/IP 任务（Kref>0）执行；t2i 时整段跳过。",
    ]
    return c.wrap("overview", lead, steps)


# ---------------------------------------------------------------------------
# per-module narrative
# ---------------------------------------------------------------------------


def _decoder(c: _C, mid: str) -> str | None:
    if mid == "d_norm_attn":
        return c.wrap(mid, (
            f"层输入 {c.t('d_hidden_in')} 做逐 token RMSNorm（{c.p('H')} 维，eps={c.p('EpsN')}），"
            f"输出 {c.t('d_normed')} 同时喂给 q/k/v 三条投影。"), [
            "Qwen3-VL 用 RMSNorm，不是 LayerNorm；没有 bias，也没有时间步调制。",
        ])
    if mid == "d_attn_q":
        return c.wrap(mid, (
            f"<code>q_proj</code> 把 {c.sh('d_normed')} 映到 {c.sh('d_q_lin')}，按 "
            f"{c.p('Nh')} 个头拆成 {c.sh('d_q_head')}，每个头做 RMSNorm({c.p('Dh')})，"
            f"再施加 MRoPE 得到 {c.t('d_q')}。"), [
            "q_norm 是逐头 QK-norm（Qwen3 系列机制），只对 head_dim 归一。",
            "MRoPE 为 interleaved [24,20,20]，三轴位置来自 position_ids（目标图像 patch 从 4096 起）。",
        ])
    if mid == "d_attn_k":
        return c.wrap(mid, (
            f"<code>k_proj</code> 输出 {c.sh('d_k_lin')}，按 {c.p('Nkv')} 个 KV 头拆开后同样做"
            f"逐头 RMSNorm 与 MRoPE，得到 {c.t('d_k')}。"), [
            f"GQA：{c.p('Nh')} 个 Q 头共享 {c.p('Nkv')} 个 KV 头；K 的权重仍是 8 份，不复制成 32 份。",
        ])
    if mid == "d_attn_v":
        return c.wrap(mid, (
            f"<code>v_proj</code> 输出 {c.sh('d_v_lin')}，只做 view/transpose 得到 {c.t('d_v')}。"), [
            "V 不做 QK-norm、不做 RoPE：这两步只作用于 q/k。",
        ])
    if mid == "d_attn_ar":
        return c.wrap(mid, (
            f"第 1 趟：按 {c.t('d_ar_index')} 取出 AR 行，只在 AR 子集内部做 causal 注意力"
            f"（{c.sh('d_score_ar')} → {c.sh('d_ctx_ar')}）。"), [
            "文本在前、生成 token 在后的布局下，AR 行左侧没有生成 token，所以“AR 子集 causal”"
            "等价于“AR 行只对文本行因果”。",
            "flash 内核把 QK/softmax/AV 融合执行；这里按语义拆成三个具名算子。",
        ])
    if mid == "d_attn_full":
        return c.wrap(mid, (
            f"第 2 趟：全序列 non-causal 注意力（{c.sh('d_score_full')} → {c.sh('d_ctx_full')}），"
            "所有行都能看到全部 token。"), [
            f"这一趟是 O({c.p('Stot')}²) 的规模来源；生成 token 行只在这一趟里有效。",
        ])
    if mid == "d_attn_merge":
        return c.wrap(mid, (
            f"<code>out_full[:, idx_ar] = out_ar</code>：把 AR 行换回第 1 趟结果，得到"
            f" {c.t('d_ctx')}。"), [
            "use_flash_attn=False 时，同一语义由 4D additive mask 一次算出"
            "（生成行置 0、其余上三角 -inf）；不另建第二套层结构。",
        ])
    if mid == "d_attn_out":
        return c.wrap(mid, (
            f"32 个头拼回 {c.sh('d_attn_flat')} 后过 <code>o_proj</code> 得 {c.t('d_attn')}。"), [
            "o_proj 无 bias；没有输出门/缩放（输出直接进残差）。",
        ])
    if mid == "d_res_attn":
        return c.wrap(mid, f"{c.t('d_hidden_in')} + {c.t('d_attn')}，得到 {c.t('d_h1')}。", [
            "普通标量残差相加；没有多路流、没有读门/写门。",
        ])
    if mid == "d_norm_ffn":
        return c.wrap(mid, f"第二个 RMSNorm（{c.p('H')} 维）后进 MLP。", [
            "Qwen3-VL 的 norm 位置是 pre-norm：两个 RMSNorm 分别在 attention/MLP 之前。",
        ])
    if mid == "d_ffn_mlp":
        return c.wrap(mid, (
            f"<code>gate_proj</code>/<code>up_proj</code> 各把 {c.sh('d_h1n')} 映到 {c.sh('d_gate')}；"
            f"<code>SiLU(gate)*up</code> 后由 <code>down_proj</code> 映回 {c.sh('d_ffn')}。"), [
            "SwiGLU，无 bias、无 clamp；是 dense MLP，不是 MoE/共享专家。",
        ])
    if mid == "d_res_ffn":
        return c.wrap(mid, f"{c.t('d_h1')} + {c.t('d_ffn')}，得到 {c.t('d_h2')}。", [
            "层输出到此完整；DeepStack 注入在循环体里发生在本层之后。",
        ])
    if mid == "d_deepstack":
        return c.wrap(mid, (
            f"参考图任务且在 layer 0..2 时，把 {c.t('d_ds_embeds')} 加到 "
            f"{c.t('d_visual_mask')} 为 True 的行上，得到 {c.t('d_hidden_out')}。"), [
            "只改条件图占位行；其它 token 原样传递（t2i 或 layer≥3 时等价恒等）。",
            f"3 个特征分别对应视觉塔 block 8/16/24，注入到 decoder 前 {c.p('Ndh')} 层。",
        ])
    return None


def _vision(c: _C, mid: str) -> str | None:
    if mid == "v_norm_attn":
        return c.wrap(mid, (
            f"视觉塔输入 {c.t('v_hidden_in')} 做 LayerNorm({c.p('Hv')})，得到 {c.t('v_normed1')}。"), [
            "视觉塔用 LayerNorm；文本主干用 RMSNorm，两者不混用。",
        ])
    if mid == "v_attn_qkv":
        return c.wrap(mid, (
            f"一次 <code>qkv</code> 线性投影（{c.sh('v_qkv')}，bias=True）后 reshape/permute 拆成 "
            f"q/k/v（{c.sh('v_q')}）。"), [
            "q/k/v 共享一次投影，不是三个独立 Linear；head_dim=72。",
        ])
    if mid == "v_attn_rope":
        return c.wrap(mid, (
            f"2D rotary：行/列坐标各占 {c.p('Dhv')}/2 维，cos/sin 由 grid 构造，只作用于 q/k。"), [
            "与文本主干的 MRoPE 不是同一套位置机制：这里是视觉 2D rotary。",
        ])
    if mid == "v_attn_seg":
        return c.wrap(mid, (
            f"按 {c.t('v_cu_seqlens')} 的差分长度把 q/k/v 切成 Kref 段"
            f"（第 i 段 {c.t('v_q_seg')} {c.sh('v_q_seg')}）；逐段算完后拼回"
            f" {c.t('v_ctx')} {c.sh('v_ctx')}。"), [
            "循环边界来自 cu_seqlens（每图 h*w 的 cumsum），段数 = 条件图数量 Kref。",
            "段长度 S_i 是数据相关量：等长时 Kref×S_i=Svis；不等长时不能用单值表示，"
            "逐段规模保留未知。",
            "不同图之间不互相注意；拼接只恢复顺序，不引入跨图信息。",
        ])
    if mid == "v_attn_core":
        return c.wrap(mid, (
            f"第 i 段做非因果 QK^T×Dhv^-0.5、softmax、AV（{c.t('v_scores')} {c.sh('v_scores')} → "
            f"{c.t('v_ctx_seg')} {c.sh('v_ctx_seg')}）。"), [
            "score 是逐段局部 shape，不是 [Nhv,Svis,Svis]；总元素 Σ_i Nhv·S_i²。",
            "flash 路径同样按段分块调用，只是把三步融合在一个内核里。",
        ])
    if mid == "v_attn_out":
        return c.wrap(mid, f"16 个头拼回 {c.sh('v_ctx_flat')}，再 proj 回 {c.sh('v_attn')}。", [
            "proj 带 bias。",
        ])
    if mid == "v_res_attn":
        return c.wrap(mid, f"{c.t('v_hidden_in')} + {c.t('v_attn')}，得到 {c.t('v_h1')}。", [])
    if mid == "v_norm_ffn":
        return c.wrap(mid, f"第二个 LayerNorm 后进 MLP。", [])
    if mid == "v_ffn_mlp":
        return c.wrap(mid, (
            f"fc1 把 {c.sh('v_normed2')} 映到 {c.sh('v_fc1')}，GELU(tanh) 后 fc2 映回 "
            f"{c.sh('v_fc2')}。"), [
            "GELU-tanh + 两层 MLP；不是 SwiGLU。",
        ])
    if mid == "v_res_ffn":
        return c.wrap(mid, f"{c.t('v_h1')} + {c.t('v_fc2')}，得到 {c.t('v_out')}。", [
            "block 输出继续流向下一 block。",
        ])
    if mid == "v_deepstack":
        return c.wrap(mid, (
            f"block 序号 ∈ {{8,16,24}} 时，{c.t('v_out')} 的同一状态再被对应 DeepStack merger 读取"
            f"（{c.t('v_tap')}）。"), [
            "tap 不改变主线：v_out 仍作为下一 block 的输入。",
            "merger 本身在整网层（一次性模块），不在 27 次 block 模板里。",
        ])
    return None


def module_narrative(block: dict, module_id: str) -> str:
    c = _C(block)
    kind = str(block.get("kind"))
    result = _decoder(c, module_id) if kind == "uit_decoder" else _vision(c, module_id)
    return result or ""
