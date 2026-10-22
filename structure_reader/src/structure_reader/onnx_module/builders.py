"""18 个模块的显式 ONNX builder（源码事实直译；无自定义域算子）。

约定：
- 权重是 graph input（真实 config 形状，不写实值）；派生数学常数用 initializer；
- cache 状态 pre 为输入、post 为输出；N_win/N_comp 等动态维用符号；
- 每个节点归属一个 group（= JSON 算子 id），供右图语义分组显示；
- 只用 opset 18 标准算子名（Clip / Sigmoid+Mul / ScatterND(reduction=add) 等）。
"""
from __future__ import annotations

import numpy as np

from structure_reader.onnx_module.modeling import (
    DTYPE, FLOAT, INT64, GB, clip, linear, rmsnorm, rope_apply, rope_cos_sin, silu)

INT_MAX = np.iinfo(np.int64).max
IDX = "int64"
INDEX_OUTS = {"topk_indices", "hit_experts", "expert_token_idx",
              "expert_topk_pos", "indexer_topk"}


class Ctx:
    def __init__(self, doc: dict, phase: dict, config: dict, rope_main: dict,
                 rope_compress: dict):
        self.doc = doc
        self.phase = phase
        self.pid = phase["id"]
        self.config = config
        self.rope_main = rope_main
        self.rope_compress = rope_compress
        self.rope = rope_compress
        self.ops = {o["id"]: o for o in phase["operators"]}
        self.tensors = {t["id"]: t for t in phase["tensors"]}
        self.params = {p["id"]: p for p in phase["parameters"]}

    def sh(self, tid):
        return list(self.tensors[tid]["shape"])

    def p(self, pid):
        return self.params[pid].get("value")

    def weight(self, g: GB, group: str, tid: str) -> str:
        return g.inp(tid, self.sh(tid), DTYPE, "weight")

    def act_in(self, g: GB, tid: str, name: str | None = None, shape=None) -> str:
        return g.inp(name or tid, shape or self.sh(tid), DTYPE, "activation")

    def cache_in(self, g: GB, tid: str, dtype=DTYPE) -> str:
        return g.inp(tid, self.sh(tid), dtype, "cache_pre")

    def pos_in(self, g: GB) -> str:
        if not any(i["name"] == "position_ids" for i in g.inputs):
            g.inp("position_ids", self.sh("position_ids"), INT64, "activation")
        return "position_ids"

    def out_t(self, g: GB, tid: str, role="activation", name: str | None = None,
              shape=None, dtype: str = DTYPE) -> str:
        return g.out(name or tid, self.sh(tid) if shape is None else shape, dtype, role,
                     json_id=tid)


# ---------------------------------------------------------------------------
def build_norm(ctx: Ctx, op_id: str) -> GB:
    g = GB(op_id, ctx.pid)
    x = ctx.act_in(g, ctx.ops[op_id]["inputs"][0])
    w_tid = (ctx.ops[op_id]["weights"] or ctx.ops[op_id]["inputs"][1:])[0]
    w = ctx.weight(g, op_id, w_tid)
    rmsnorm(g, op_id, x, ctx.ops[op_id]["outputs"][0], eps=ctx.p("EpsN"), weight=w)
    g.set_group_label(op_id, "DeepseekV4RMSNorm（fp32：x*rsqrt(mean(x²)+eps)*weight）")
    ctx.out_t(g, ctx.ops[op_id]["outputs"][0])
    return g


# ---------------------------------------------------------------------------
def build_attn_q(ctx: Ctx) -> GB:
    g = GB("attn_q", ctx.pid)
    Nh, Dh, Rq, Rk = (int(ctx.p(k)) for k in ("Nh", "Dh", "Rq", "Rk"))
    x = ctx.act_in(g, "attn_normed")
    qa = linear(g, "q_a_proj", x, ctx.weight(g, "q_a_proj", "q_weight"), "q_a")
    g.set_group_label("q_a_proj", "self.q_a_proj(hidden_states)：Linear 无 bias")
    qr = rmsnorm(g, "q_a_norm", qa, "q_a_normed", ctx.p("EpsN"),
                 ctx.weight(g, "q_a_norm", "q_a_norm_weight"))
    g.set_group_label("q_a_norm", "DeepseekV4RMSNorm(q_lora_rank)")
    qflat = linear(g, "q_b_proj", qr, ctx.weight(g, "q_b_proj", "q_b_weight"), "q_b_flat")
    g.node("Reshape", [qflat, g.const("q_b_proj/rs", np.array([0, 0, Nh, Dh], np.int64))],
           ["q_b_4d"], group="q_b_proj")
    qhead = g.node("Transpose", ["q_b_4d"], ["q_head"], group="q_b_proj", perm=[0, 2, 1, 3])
    g.set_group_label("q_b_proj", "self.q_b_proj→view(B,T,Nh,Dh)→transpose(1,2)")
    qn = rmsnorm(g, "q_b_norm", qhead, "q_normed", ctx.p("EpsN"))
    g.set_group_label("q_b_norm", "DeepseekV4UnweightedRMSNorm（无 weight）")
    ctx.pos_in(g)
    rope_cos_sin(g, "rope_compress", "position_ids", ctx.rope, "cos_q", "sin_q")
    g.set_group_label("rope_compress", "DeepseekV4RotaryEmbedding（compress YaRN，θ=160000）")
    rope_apply(g, "q_rope_apply", qn, "cos_q", "sin_q", Rk, Dh, "q_rope")
    g.set_group_label("q_rope_apply", "apply_rotary_pos_emb（interleaved，仅末 64 维）")
    # q_residual 是 Indexer/CSA 共用的边界输出，不能只导出 q_rope
    ctx.out_t(g, "q_a_normed")
    ctx.out_t(g, "q_rope")
    return g


def build_attn_kv(ctx: Ctx) -> GB:
    g = GB("attn_kv", ctx.pid)
    Dh, Rk = int(ctx.p("Dh")), int(ctx.p("Rk"))
    x = ctx.act_in(g, "attn_normed")
    proj = linear(g, "kv_proj", x, ctx.weight(g, "kv_proj", "kv_weight"), "kv_proj_linear")
    kvn = rmsnorm(g, "kv_proj", proj, "kv_proj_out", ctx.p("EpsN"),
                  ctx.weight(g, "kv_proj", "kv_norm_weight"))
    g.set_group_label("kv_proj", "self.kv_proj→self.kv_norm（共享 KV，K=V）")
    g.node("Reshape", [kvn, g.const("kv_head_view/rs", np.array([0, 0, 1, Dh], np.int64))],
           ["kv_4d"], group="kv_head_view")
    kvh = g.node("Transpose", ["kv_4d"], ["kv_head"], group="kv_head_view", perm=[0, 2, 1, 3])
    g.set_group_label("kv_head_view", "view(B,T,1,Dh).transpose(1,2)")
    ctx.pos_in(g)
    rope_cos_sin(g, "kv_rope_apply", "position_ids", ctx.rope, "cos_q", "sin_q")
    rope_apply(g, "kv_rope_apply", kvh, "cos_q", "sin_q", Rk, Dh, "kv_rope")
    g.set_group_label("kv_rope_apply", "与 Q 共用 compress rope 的 interleaved RoPE")
    ctx.out_t(g, "kv_rope")
    return g


def build_attn_cache(ctx: Ctx) -> GB:
    g = GB("attn_cache", ctx.pid)
    W = int(ctx.p("W"))
    kv = ctx.act_in(g, "kv_rope")
    keys = ctx.cache_in(g, "cache_keys")
    full = g.node("Concat", [keys, kv], ["kv_full"], group="cache_update", axis=2)
    g.node("Slice", [full, g.const("cache_update/neg_w", np.array([-(W - 1)], np.int64)),
                     g.const("cache_update/end_max", np.array([INT_MAX], np.int64)),
                     g.const("cache_update/ax2", np.array([2], np.int64))],
           ["cache_keys_post"], group="cache_update")
    g.set_group_label("cache_update", "update(): full=cat(keys,kv)；持久 keys 只留 W-1 尾；values=keys")
    ctx.out_t(g, "kv_full", role="cache_view")
    ctx.out_t(g, "cache_keys_post", role="cache_post")
    g.node("Identity", ["cache_keys_post"], ["cache_values_post"], group="cache_update")
    ctx.out_t(g, "cache_values_post", role="cache_post_alias", shape=ctx.sh("cache_keys_post"))
    # 源码 :197 的 self.cumulative_length += T：原 JSON 无该 post 张量，ONNX 派生输出
    cum = g.inp("cache_cumulative_length", [], INT64, "cache_pre")
    T = g.inp("T", [], INT64, "shape")
    g.node("Add", [cum, T], ["cache_cumulative_length_post"], group="cache_update")
    ctx.out_t(g, "cache_cumulative_length_post", role="derived_cache_post", shape=[], dtype="int64")
    return g


def build_attn_core(ctx: Ctx) -> GB:
    g = GB("attn_core", ctx.pid)
    Nh, Dh = int(ctx.p("Nh")), int(ctx.p("Dh"))
    scale = float(Dh) ** -0.5
    q = ctx.act_in(g, "q_rope")
    kvf = ctx.act_in(g, "kv_full", shape=ctx.sh("kv_full"))
    comp = ctx.act_in(g, "cache_compressed_c_post")
    comp_u = g.node("Unsqueeze", [comp], ["compressed_u"], group="kv_concat", axes=[1])
    kvall = g.node("Concat", [kvf, comp_u], ["kv_all"], group="kv_concat", axis=2)
    g.set_group_label("kv_concat", "cat([kv_full, compressed_kv], dim=2)")
    rp = g.node("Reshape", [kvall, g.const("repeat_kv/rs", np.array([0, 1, 1, -1, Dh], np.int64))],
                ["kv_5d"], group="repeat_kv")
    exp_shape = g.node("Concat", [g.node("Shape", [kvall], ["kv_shape"], group="repeat_kv"),
                                  g.const("repeat_kv/rest", np.array([], np.int64))],
                      ["tmp_unused"], group="repeat_kv") if False else None
    ex = g.node("Expand", [rp, g.const("repeat_kv/exp", np.array([1, Nh, 1, 1, 1], np.int64))],
                ["kv_exp"], group="repeat_kv")
    g.node("Reshape", [ex, g.const("repeat_kv/rs2", np.array([0, Nh, -1, Dh], np.int64))],
           ["kv_repeated"], group="repeat_kv")
    g.set_group_label("repeat_kv", "MQA KV 复制到 Nh 头（Expand+Reshape）")
    kvt = g.node("Transpose", ["kv_repeated"], ["kv_T"], group="qk_scaled", perm=[0, 1, 3, 2])
    qk = g.node("MatMul", [q, kvt], ["qk_matmul"], group="qk_scaled")
    g.node("Mul", [qk, g.const("qk_scaled/scale", np.float32(scale))], ["qk_logits"], group="qk_scaled")
    g.set_group_label("qk_scaled", "matmul(query, key.T) * head_dim**-0.5")
    mask = ctx.act_in(g, "mask_ext")
    qkm = g.node("Add", ["qk_logits", mask], ["qk_masked"], group="mask_add")
    g.set_group_label("mask_add", "attn_weights + attention_mask")
    sinks = g.inp("W.sink_softmax.sinks", ["Nh"], DTYPE, "weight")
    sinks4 = g.node("Reshape", [sinks, g.const("sink_softmax/sk", np.array([1, -1, 1, 1], np.int64))],
                    ["sinks_4d"], group="sink_softmax")
    # HF:730 sinks.reshape(1,-1,1,1).expand(B,-1,T,-1) —— 必须显式 Expand 到 query 的 B/T
    qk_shape = g.node("Shape", [qkm], ["qk_shape"], group="sink_softmax")
    sB = g.node("Slice", [qk_shape, g.const("sink_softmax/b0", np.array([0], np.int64)),
                          g.const("sink_softmax/b1", np.array([1], np.int64)),
                          g.const("sink_softmax/ax0", np.array([0], np.int64))],
                ["sink_B"], group="sink_softmax")
    sT = g.node("Slice", [qk_shape, g.const("sink_softmax/t0", np.array([2], np.int64)),
                          g.const("sink_softmax/t1", np.array([3], np.int64)),
                          g.const("sink_softmax/ax1", np.array([0], np.int64))],
                ["sink_T"], group="sink_softmax")
    sink_shape = g.node("Concat", [sB, g.const("sink_softmax/nh", np.array([Nh], np.int64)), sT,
                                   g.const("sink_softmax/one1", np.array([1], np.int64))],
                        ["sink_shape"], group="sink_softmax", axis=0)
    sinks_exp = g.node("Expand", [sinks4, sink_shape], ["sinks_exp"], group="sink_softmax")
    cat = g.node("Concat", [qkm, sinks_exp], ["logits_sink"], group="sink_softmax", axis=-1)
    mx = g.node("ReduceMax", [cat], ["logits_max"], group="sink_softmax", axes=[-1], keepdims=1)
    sub = g.node("Sub", [cat, mx], ["logits_shift"], group="sink_softmax")
    probs = g.node("Softmax", [sub], ["probs_sink"], group="sink_softmax", axis=-1)
    g.node("Slice", [probs, g.const("sink_softmax/s0", np.array([0], np.int64)),
                     g.const("sink_softmax/s1", np.array([-1], np.int64)),
                     g.const("sink_softmax/ax", np.array([-1], np.int64))],
           ["attn_probs"], group="sink_softmax")
    g.set_group_label("sink_softmax", "cat(sinks)→减 max→softmax→去掉 sink 列")
    o = g.node("MatMul", ["attn_probs", "kv_repeated"], ["av_out"], group="av_matmul")
    g.node("Transpose", [o], ["attn_head_out"], group="av_matmul", perm=[0, 2, 1, 3])
    g.set_group_label("av_matmul", "matmul(probs, values).transpose(1,2)")
    ctx.out_t(g, "attn_head_out")
    return g


def build_attn_out(ctx: Ctx) -> GB:
    g = GB("attn_out", ctx.pid)
    Nh, Dh, G, Ro, Rk = (int(ctx.p(k)) for k in ("Nh", "Dh", "G", "Ro", "Rk"))
    x = ctx.act_in(g, "attn_head_out")
    ctx.pos_in(g)
    # HF:865 逆 RoPE 前后必须 transpose(1,2)：helper 期望 [B,Nh,T,Dh]，模块边界是 [B,T,Nh,Dh]
    xh = g.node("Transpose", [x], ["attn_heads"], group="attn_inv_rope", perm=[0, 2, 1, 3])
    rope_cos_sin(g, "attn_inv_rope", "position_ids", ctx.rope, "cos_q", "sin_q")
    rope_out = rope_apply(g, "attn_inv_rope", xh, "cos_q", "sin_q", Rk, Dh, "attn_rope_inv_h",
                          sign=-1.0)
    g.node("Transpose", [rope_out], ["attn_rope_inv"], group="attn_inv_rope", perm=[0, 2, 1, 3])
    g.set_group_label("attn_inv_rope", "attn.transpose(1,2)→apply(position=-i)→transpose(1,2)")
    g.node("Reshape", ["attn_rope_inv",
                       g.const("attn_group_reshape/rs", np.array([0, 0, G, Nh * Dh // G], np.int64))],
           ["attn_grouped"], group="attn_group_reshape")
    g.set_group_label("attn_group_reshape", "reshape(B,S,G,Nh*Dh/G)")
    # HF:300-329：x.reshape(-1,G,in).transpose(0,1) → bmm(w[G,in,Ro]) → transpose 回 → reshape(B,S,G*Ro)
    in_g = Nh * Dh // G
    oa = g.inp("o_a_weight", [G * Ro, in_g], DTYPE, "weight")
    w3 = g.node("Reshape", [oa, g.const("o_a_proj/wrs", np.array([G, Ro, in_g], np.int64))],
                ["o_a_w3"], group="o_a_proj")
    wT = g.node("Transpose", [w3], ["o_a_WT"], group="o_a_proj", perm=[0, 2, 1])
    xr = g.node("Reshape", ["attn_grouped", g.const("o_a_proj/xrs", np.array([-1, G, in_g], np.int64))],
                ["o_a_xr"], group="o_a_proj")
    xt = g.node("Transpose", [xr], ["o_a_xt"], group="o_a_proj", perm=[1, 0, 2])
    y = g.node("MatMul", ["o_a_xt", wT], ["o_a_bmm"], group="o_a_proj")
    yt = g.node("Transpose", [y], ["o_a_yt"], group="o_a_proj", perm=[1, 0, 2])
    xshape = g.node("Shape", ["attn_rope_inv"], ["o_a_xshape"], group="o_a_proj")
    bt = g.node("Slice", [xshape, g.const("o_a_proj/s0", np.array([0], np.int64)),
                          g.const("o_a_proj/s1", np.array([2], np.int64)),
                          g.const("o_a_proj/ax", np.array([0], np.int64))],
                ["o_a_bt"], group="o_a_proj")
    oshape = g.node("Concat", [bt, g.const("o_a_proj/ro", np.array([G * Ro], np.int64))],
                    ["o_a_oshape"], group="o_a_proj", axis=0)
    g.node("Reshape", [yt, oshape], ["o_a_out"], group="o_a_proj")
    g.set_group_label("o_a_proj", "DeepseekV4GroupedLinear（reshape/transpose + bmm 按组）")
    y = linear(g, "o_b_proj", "o_a_out", ctx.weight(g, "o_b_proj", "o_b_weight"), "o_b_linear")
    g.node("Reshape", [y, g.const("o_b_proj/rs", np.array([0, 0, -1], np.int64))],
           ["attn_block_output"], group="o_b_proj")
    g.set_group_label("o_b_proj", "self.o_b_proj→flatten")
    ctx.out_t(g, "attn_block_output")
    return g


# ---------------------------------------------------------------------------
def build_mhc(ctx: Ctx, module: str) -> GB:
    g = GB(module, ctx.pid)
    p = "attn_hc" if module == "mhc_attn" else "ffn_hc"
    H, Nhc, Sit = int(ctx.p("H")), int(ctx.p("Nhc")), int(ctx.p("Sit"))
    eps, hce = ctx.p("EpsN"), ctx.p("Eps")
    in_tid = "hidden_streams_in" if module == "mhc_attn" else "attn_out_streams"
    out_tid = "attn_mhc_collapsed" if module == "mhc_attn" else "ffn_mhc_collapsed"
    x = ctx.act_in(g, in_tid)
    flat = g.node("Reshape", [x, g.const(f"{module}/flat", np.array([0, 0, Nhc * H], np.int64))],
                  [f"{module}_flat"], group=f"{p}_flatten")
    normed = rmsnorm(g, f"{p}_flatten", flat, f"{module}_normed", eps)
    g.set_group_label(f"{p}_flatten", "view(B,T,Nhc*H).float + UnweightedRMSNorm")
    fn = ctx.weight(g, module, "attn_hc_fn" if module == "mhc_attn" else "ffn_hc_fn")
    wts = linear(g, f"{p}_project", normed, fn, f"{module}_w")
    g.set_group_label(f"{p}_project", "F.linear(flattened, self.fn)：系数 (2+Nhc)*Nhc")
    # split
    g.node("Slice", [wts, g.const(f"{module}/p0", np.array([0], np.int64)),
                     g.const(f"{module}/p1", np.array([Nhc], np.int64)),
                     g.const(f"{module}/axl", np.array([-1], np.int64))],
           [f"{module}_pre_w"], group=f"{p}_split")
    g.node("Slice", [wts, g.const(f"{module}/q0", np.array([Nhc], np.int64)),
                     g.const(f"{module}/q1", np.array([2 * Nhc], np.int64)),
                     g.const(f"{module}/axl2", np.array([-1], np.int64))],
           [f"{module}_post_w"], group=f"{p}_split")
    cw = g.node("Slice", [wts, g.const(f"{module}/r0", np.array([2 * Nhc], np.int64)),
                          g.const(f"{module}/r1", np.array([2 * Nhc + Nhc * Nhc], np.int64)),
                          g.const(f"{module}/axl3", np.array([-1], np.int64))],
                [f"{module}_comb_w"], group=f"{p}_split")
    g.node("Reshape", [cw, g.const(f"{module}/crs", np.array([0, 0, Nhc, Nhc], np.int64))],
           [f"{module}_comb_w4"], group=f"{p}_split")
    g.set_group_label(f"{p}_split", "split([pre,post,comb]) + comb.view(Nhc,Nhc)")
    base = ctx.weight(g, module, "attn_hc_base" if module == "mhc_attn" else "ffn_hc_base")
    scale = ctx.weight(g, module, "attn_hc_scale" if module == "mhc_attn" else "ffn_hc_scale")
    bpre = g.node("Slice", [base, g.const(f"{module}/b0", np.array([0], np.int64)),
                            g.const(f"{module}/b1", np.array([Nhc], np.int64))],
                  [f"{module}_base_pre"], group=f"{p}_split")
    bpost = g.node("Slice", [base, g.const(f"{module}/b2", np.array([Nhc], np.int64)),
                             g.const(f"{module}/b3", np.array([2 * Nhc], np.int64))],
                   [f"{module}_base_post"], group=f"{p}_split")
    bcomb = g.node("Slice", [base, g.const(f"{module}/b4", np.array([2 * Nhc], np.int64)),
                             g.const(f"{module}/b5", np.array([2 * Nhc + Nhc * Nhc], np.int64))],
                   [f"{module}_base_comb"], group=f"{p}_split")
    bcomb4 = g.node("Reshape", [bcomb, g.const(f"{module}/brs", np.array([Nhc, Nhc], np.int64))],
                    [f"{module}_base_comb4"], group=f"{p}_split")
    spre = g.node("Gather", [scale, g.const(f"{module}/s0", np.array(0, np.int64))], [f"{module}_sp"], group=f"{p}_split")
    spost = g.node("Gather", [scale, g.const(f"{module}/s1", np.array(1, np.int64))], [f"{module}_sq"], group=f"{p}_split")
    scomb = g.node("Gather", [scale, g.const(f"{module}/s2", np.array(2, np.int64))], [f"{module}_sr"], group=f"{p}_split")
    g.set_group_label(f"{p}_split", "base/scale 按 [pre,post,comb] 拆分")

    pw = g.node("Mul", [f"{module}_pre_w", spre], [f"{module}_pre_sw"], group=f"{p}_pre")
    pa = g.node("Add", [pw, bpre], [f"{module}_pre_ab"], group=f"{p}_pre")
    ps = g.node("Sigmoid", [pa], [f"{module}_pre_sig"], group=f"{p}_pre")
    pre = g.node("Add", [ps, g.const(f"{module}/hce", np.float32(hce))], [f"{module}_pre"], group=f"{p}_pre")
    g.set_group_label(f"{p}_pre", "pre = sigmoid(pre_w*scale+base) + hc_eps")
    qwq = g.node("Mul", [f"{module}_post_w", spost], [f"{module}_post_sw"], group=f"{p}_post")
    qaq = g.node("Add", [qwq, bpost], [f"{module}_post_ab"], group=f"{p}_post")
    qsq = g.node("Sigmoid", [qaq], [f"{module}_post_sig"], group=f"{p}_post")
    post = g.node("Mul", [qsq, g.const(f"{module}/two", np.float32(2.0))], [f"{module}_post"], group=f"{p}_post")
    g.set_group_label(f"{p}_post", "post = 2*sigmoid(post_w*scale+base)")
    cws = g.node("Mul", [f"{module}_comb_w4", scomb], [f"{module}_comb_sw"], group=f"{p}_comb")
    cwa = g.node("Add", [cws, bcomb4], [f"{module}_comb_ab"], group=f"{p}_comb")
    csm = g.node("Softmax", [cwa], [f"{module}_comb_sm"], group=f"{p}_comb", axis=-1)
    comb = g.node("Add", [csm, g.const(f"{module}/hce2", np.float32(hce))], [f"{module}_comb"], group=f"{p}_comb")
    g.set_group_label(f"{p}_comb", "comb = softmax(comb_w*scale+base,-1) + hc_eps")

    # Sinkhorn-Knopp：源码先按列归一，再循环 (Sit-1) 次行/列归一
    cur = g.node("Add", [g.node("ReduceSum", [comb], [f"{module}_sk0"], group=f"{p}_comb",
                               axes=[-2], keepdims=1),
                         g.const(f"{module}/hce3", np.float32(hce))],
                 [f"{module}_sk0d"], group=f"{p}_comb")
    cur = g.node("Div", [comb, cur], [f"{module}_sk0n"], group=f"{p}_comb")
    for i in range(Sit - 1):
        r = g.node("ReduceSum", [cur], [f"{module}_skr{i}"], group=f"{p}_comb", axes=[-1], keepdims=1)
        r = g.node("Add", [r, g.const(f"{module}/he{i}", np.float32(hce))], [f"{module}_skra{i}"], group=f"{p}_comb")
        cur = g.node("Div", [cur, r], [f"{module}_skrn{i}"], group=f"{p}_comb")
        c = g.node("ReduceSum", [cur], [f"{module}_skc{i}"], group=f"{p}_comb", axes=[-2], keepdims=1)
        c = g.node("Add", [c, g.const(f"{module}/hc{i}", np.float32(hce))], [f"{module}_skca{i}"], group=f"{p}_comb")
        cur = g.node("Div", [cur, c], [f"{module}_skcn{i}"], group=f"{p}_comb")
    comb_s = g.node("Identity", [cur], [f"{module}_comb_sink"], group=f"{p}_comb")
    g.set_group_label(f"{p}_comb", f"Sinkhorn-Knopp ×{Sit}（静态展开，行/列归一）")
    pre3 = g.node("Unsqueeze", [pre], [f"{module}_pre_u"], group=f"{p}_collapse", axes=[-1])
    mul = g.node("Mul", [pre3, x], [f"{module}_pre_x"], group=f"{p}_collapse")
    collapsed = g.node("ReduceSum", [mul], [out_tid], group=f"{p}_collapse", axes=[2], keepdims=0)
    g.set_group_label(f"{p}_collapse", "collapsed = sum(pre.unsqueeze(-1) * streams, dim=2)")
    pre_id = "attn_mhc_pre" if module == "mhc_attn" else "ffn_mhc_pre"
    post_id = "attn_mhc_post" if module == "mhc_attn" else "ffn_mhc_post"
    comb_id = "attn_mhc_comb" if module == "mhc_attn" else "ffn_mhc_comb"
    g.node("Identity", [pre], [pre_id], group=f"{p}_pre")
    g.node("Identity", [post], [post_id], group=f"{p}_post")
    g.node("Identity", [comb_s], [comb_id], group=f"{p}_comb")
    for tid in (pre_id, post_id, comb_id, out_tid):
        ctx.out_t(g, tid)
    return g
    return g


# ---------------------------------------------------------------------------
def _residual(ctx: Ctx, module: str, comb_id: str, expand_id: str, mix_id: str,
              out_id: str) -> GB:
    g = GB(module, ctx.pid)
    comb = ctx.act_in(g, ctx.ops[comb_id]["inputs"][0])
    combT = g.node("Transpose", [comb], [ctx.ops[comb_id]["outputs"][0]], group=comb_id, perm=[0, 1, 3, 2])
    g.set_group_label(comb_id, "comb.transpose(-1,-2)（Sinkhorn 后方向敏感）")
    post = ctx.act_in(g, ctx.ops[expand_id]["inputs"][0])
    blk = ctx.act_in(g, ctx.ops[expand_id]["inputs"][1])
    p3 = g.node("Unsqueeze", [post], ["post_u"], group=expand_id, axes=[-1])
    b3 = g.node("Unsqueeze", [blk], ["blk_u"], group=expand_id, axes=[-2])
    exp = g.node("Mul", [p3, b3], [ctx.ops[expand_id]["outputs"][0]], group=expand_id)
    g.set_group_label(expand_id, "post.unsqueeze(-1) * block.unsqueeze(-2)")
    base = ctx.act_in(g, ctx.ops[mix_id]["inputs"][1])
    mix = g.node("MatMul", [combT, base], ["comb_mix"], group=mix_id)
    g.node("Add", [exp, mix], [out_id], group=mix_id)
    g.set_group_label(mix_id, "matmul(comb.T, streams) + post*block")
    ctx.out_t(g, out_id)
    return g


def build_res_attn(ctx: Ctx) -> GB:
    return _residual(ctx, "res_attn", "attn_residual_combT", "attn_residual_expand",
                     "attn_residual_mix", "attn_out_streams")


def build_res_ffn(ctx: Ctx) -> GB:
    return _residual(ctx, "res_ffn", "ffn_residual_combT", "ffn_residual_expand",
                     "ffn_residual_mix", "hidden_streams_out")


# ---------------------------------------------------------------------------
def build_moe_shared(ctx: Ctx) -> GB:
    g = GB("moe_shared", ctx.pid)
    Es, Lim = int(ctx.p("Es")), float(ctx.p("Lim"))
    x = ctx.act_in(g, "ffn_normed")
    gu = linear(g, "shared_gate_up", x, ctx.weight(g, "shared_gate_up", "shared_gate_weight"), "shared_gate")
    up = linear(g, "shared_gate_up", x, ctx.weight(g, "shared_gate_up", "shared_up_weight"), "shared_up_raw")
    clip(g, "shared_gate_up", gu, "shared_gate_c", -Lim, Lim)
    clip(g, "shared_gate_up", up, "shared_up", -Lim, Lim)
    g.set_group_label("shared_gate_up", "gate/up 两个 Linear + swiglu_limit clamp")
    act = silu(g, "shared_act_down", "shared_gate_c", "shared_act")
    gated = g.node("Mul", [act, "shared_up"], ["shared_gated"], group="shared_act_down")
    down = linear(g, "shared_act_down", gated, ctx.weight(g, "shared_act_down", "shared_down_weight"),
                  "shared_out")
    g.set_group_label("shared_act_down", "silu(gate)*up → down_proj")
    ctx.out_t(g, "shared_out")
    return g


def build_moe_merge(ctx: Ctx) -> GB:
    g = GB("moe_merge", ctx.pid)
    r = ctx.act_in(g, "routed_view")
    s = ctx.act_in(g, "shared_out")
    g.node("Add", [r, s], ["mlp_output"], group="moe_merge")
    g.set_group_label("moe_merge", "routed + shared_experts")
    ctx.out_t(g, "mlp_output")
    return g


def build_moe_route(ctx: Ctx) -> GB:
    g = GB("moe_route", ctx.pid)
    H, E, K = int(ctx.p("H")), int(ctx.p("E")), int(ctx.p("K"))
    Scl = float(ctx.p("Scl"))
    x = ctx.act_in(g, "ffn_normed")
    g.node("Reshape", [x, g.const("moe_flatten/rs", np.array([-1, H], np.int64))],
           ["ffn_flat"], group="moe_flatten")
    g.set_group_label("moe_flatten", "view(-1, H)")
    rw = ctx.weight(g, "moe_router", "router_weight")
    bias = ctx.weight(g, "moe_router", "router_bias")
    wt = g.node("Transpose", [rw], ["router_WT"], group="moe_router", perm=[1, 0])
    logits = g.node("MatMul", ["ffn_flat", wt], ["router_logits"], group="moe_router")
    sp = g.node("Softplus", [logits], ["router_softplus"], group="moe_router")
    scores = g.node("Sqrt", [sp], ["router_scores"], group="moe_router")
    sc = g.node("Add", [scores, bias], ["router_scores_bias"], group="moe_router")
    topv, topi = g.node("TopK", [sc, g.const("moe_router/k", np.array([K], np.int64))],
                        ["topk_vals", "topk_indices"], group="moe_router", axis=-1)
    w = g.node("GatherElements", [scores, topi], ["topk_weights_raw"], group="moe_router", axis=-1)
    ssum = g.node("ReduceSum", [w], ["topk_w_sum"], group="moe_router", axes=[-1], keepdims=1)
    ssum = g.node("Add", [ssum, g.const("moe_router/eps", np.float32(1e-20))], ["topk_w_den"], group="moe_router")
    wn = g.node("Div", [w, ssum], ["topk_weights_norm"], group="moe_router")
    g.node("Mul", [wn, g.const("moe_router/scl", np.float32(Scl))], ["topk_weights"], group="moe_router")
    g.set_group_label("moe_router", "sqrt(softplus(logits))→+bias→TopK→gather→归一×routed_scaling")
    oh = g.node("OneHot", [topi, g.const("moe_dispatch/depth", np.array(E + 1, np.int64)),
                           g.const("moe_dispatch/values", np.array([0.0, 1.0], np.float32))],
                ["one_hot_btke"], group="moe_dispatch", axis=-1)
    g.node("Transpose", [oh], ["expert_mask"], group="moe_dispatch", perm=[2, 1, 0])
    g.set_group_label("moe_dispatch", "one_hot(idx,E+1).permute(2,1,0) + nonzero(hit)")
    mask_sum = g.node("ReduceSum", ["expert_mask"], ["mask_sum"], group="moe_dispatch",
                      axes=[-1, -2], keepdims=0)
    g.node("Greater", [mask_sum, g.const("moe_dispatch/zero", np.float32(0.0))],
           ["hit_bool"], group="moe_dispatch")
    nz = g.node("NonZero", ["hit_bool"], ["hit_nz"], group="moe_dispatch")
    g.node("Transpose", [nz], ["hit_experts"], group="moe_dispatch", perm=[1, 0])
    g.set_group_label("moe_dispatch", "one_hot→permute→nonzero→transpose 成 [N_hit,1]（HF:1044）")
    ctx.out_t(g, "ffn_flat", role="boundary")
    for tid in ("router_logits", "topk_indices", "topk_weights", "expert_mask", "hit_experts"):
        ctx.out_t(g, tid, shape=ctx.sh(tid), dtype=IDX if tid in INDEX_OUTS else DTYPE)
    return g


def build_moe_expert(ctx: Ctx) -> GB:
    g = GB("moe_expert", ctx.pid)
    Es, Lim, H = int(ctx.p("Es")), float(ctx.p("Lim")), int(ctx.p("H"))
    flat = ctx.act_in(g, "ffn_flat")
    gate_up = g.inp("W.expert_gate_up", ctx.sh("expert_gate_up"), DTYPE, "weight")
    down = g.inp("W.expert_down", ctx.sh("expert_down"), DTYPE, "weight")
    eidx = g.inp("expert_index", [], INT64, "index")
    emask = ctx.act_in(g, "expert_mask")
    tw = ctx.act_in(g, "topk_weights")
    b_dim = g.inp("B", [], INT64, "shape")
    t_dim = g.inp("T", [], INT64, "shape")
    g.node("Gather", [gate_up, eidx], ["expert_gate_up_e"], group="moe_expert_slice", axis=0)
    g.node("Gather", [down, eidx], ["expert_down_e"], group="moe_expert_slice", axis=0)
    g.set_group_label("moe_expert_slice", "按命中专家下标切 gate_up/down（模板一次）")
    # HF:1050 top_k_pos, token_idx = where(mask[expert_idx])；mask[expert_idx] 形状 [K, BT]
    me = g.node("Gather", [emask, eidx], ["expert_row"], group="moe_select_tokens", axis=0)
    nz = g.node("NonZero", [me], ["expert_nz"], group="moe_select_tokens")
    krow = g.node("Slice", [nz, g.const("moe_select_tokens/k0", np.array([0], np.int64)),
                            g.const("moe_select_tokens/k1", np.array([1], np.int64)),
                            g.const("moe_select_tokens/ax0", np.array([0], np.int64))],
                  ["expert_k_row"], group="moe_select_tokens")
    trow = g.node("Slice", [nz, g.const("moe_select_tokens/t0", np.array([1], np.int64)),
                            g.const("moe_select_tokens/t1", np.array([2], np.int64)),
                            g.const("moe_select_tokens/ax1", np.array([0], np.int64))],
                  ["expert_t_row"], group="moe_select_tokens")
    tpos = g.node("Squeeze", [krow, g.const("moe_select_tokens/sq0", np.array([0], np.int64))],
                  ["expert_topk_pos"], group="moe_select_tokens")
    tidx = g.node("Squeeze", [trow, g.const("moe_select_tokens/sq1", np.array([0], np.int64))],
                  ["expert_token_idx"], group="moe_select_tokens")
    g.node("Gather", [flat, tidx], ["expert_tokens"], group="moe_select_tokens", axis=0)
    g.set_group_label("moe_select_tokens",
                      "expert_mask[expert_idx].nonzero → top_k_pos/token_idx → hidden_states[token_idx]")
    guT = g.node("Transpose", ["expert_gate_up_e"], ["gu_T"], group="moe_expert_gu", perm=[1, 0])
    g.node("MatMul", ["expert_tokens", guT], ["expert_gu_out"], group="moe_expert_gu")
    g.set_group_label("moe_expert_gu", "F.linear(tokens, gate_up[e])")
    gate = g.node("Slice", ["expert_gu_out", g.const("moe_expert_clamp/a0", np.array([0], np.int64)),
                            g.const("moe_expert_clamp/a1", np.array([Es], np.int64)),
                            g.const("moe_expert_clamp/a2", np.array([-1], np.int64))],
                  ["expert_gate_c"], group="moe_expert_clamp")
    up = g.node("Slice", ["expert_gu_out", g.const("moe_expert_clamp/b0", np.array([Es], np.int64)),
                          g.const("moe_expert_clamp/b1", np.array([2 * Es], np.int64)),
                          g.const("moe_expert_clamp/b2", np.array([-1], np.int64))],
                ["expert_up_c"], group="moe_expert_clamp")
    g.node("Clip", [gate, g.const("moe_expert_clamp/gmin", np.float32(-3.4e38)),
                    g.const("moe_expert_clamp/gmax", np.float32(Lim))],
           ["expert_gate_c_clip"], group="moe_expert_clamp")
    clip(g, "moe_expert_clamp", up, "expert_up_c_clip", -Lim, Lim)
    g.set_group_label("moe_expert_clamp", "chunk(2) + swiglu_limit clamp（Clip）")
    act = silu(g, "moe_expert_act", "expert_gate_c_clip", "expert_silu")
    g.node("Mul", [act, "expert_up_c_clip"], ["expert_gated"], group="moe_expert_act")
    g.set_group_label("moe_expert_act", "silu(gate)*up（Sigmoid+Mul）")
    dnT = g.node("Transpose", ["expert_down_e"], ["dn_T"], group="moe_expert_down", perm=[1, 0])
    g.node("MatMul", ["expert_gated", dnT], ["expert_down_raw"], group="moe_expert_down")
    g.set_group_label("moe_expert_down", "F.linear(gated, down[e])")
    ti_u = g.node("Unsqueeze", [tidx], ["ti_u"], group="moe_expert_weight", axes=[-1])
    tp_u = g.node("Unsqueeze", [tpos], ["tp_u"], group="moe_expert_weight", axes=[-1])
    idx2 = g.node("Concat", [ti_u, tp_u], ["w_idx"], group="moe_expert_weight", axis=-1)
    w = g.node("GatherND", [tw, idx2], ["expert_w"], group="moe_expert_weight")
    wu = g.node("Unsqueeze", [w], ["expert_w_u"], group="moe_expert_weight", axes=[-1])
    g.node("Mul", ["expert_down_raw", wu], ["expert_down_out"], group="moe_expert_weight")
    g.set_group_label("moe_expert_weight", "topk_weights[token,pos] 加权")
    zeros = g.node("Mul", [flat, g.const("moe_combine/zero", np.float32(0.0))],
                   ["zeros_flat"], group="moe_combine")
    g.node("ScatterND", ["zeros_flat", ti_u, "expert_down_out"], ["routed_out"],
           group="moe_combine", reduction="add")
    g.set_group_label("moe_combine", "index_add_(0, token_idx, down_out)（ScatterND reduction=add）")
    rv_shape = g.node("Concat", [g.node("Unsqueeze", [b_dim], ["b_u"], group="moe_routed_view", axes=[0]),
                                 g.node("Unsqueeze", [t_dim], ["t_u"], group="moe_routed_view", axes=[0]),
                                 g.const("moe_routed_view/h", np.array([H], np.int64))],
                       ["rv_shape"], group="moe_routed_view", axis=0)
    g.node("Reshape", ["routed_out", rv_shape], ["routed_view"], group="moe_routed_view")
    g.set_group_label("moe_routed_view", "view(B,T,H)")
    for tid in ("expert_tokens", "expert_token_idx", "expert_topk_pos", "expert_gate_up_e",
                "expert_down_e", "expert_gu_out", "expert_gate_c", "expert_up_c",
                "expert_gated", "expert_down_raw", "expert_down_out", "routed_out", "routed_view"):
        ctx.out_t(g, tid, shape=ctx.sh(tid), dtype=IDX if tid in INDEX_OUTS else DTYPE)
    return g


# ---------------------------------------------------------------------------
# 6 CSA / Indexer（窗口/cache；N_win=0 用零窗口 + Slice/Where 保持语义，不用假分支）
# ---------------------------------------------------------------------------
def _vec(g: GB, name: str, scalar: str, group: str) -> str:
    """0-D 标量 → [1]（Slice/Range 的 starts/ends 必须是 1-D）。"""
    return g.node("Unsqueeze", [scalar], [name], group=group, axes=[0])


def _window_common(g: GB, ctx: Ctx, prefix: str, x: str, kv_w: str, gate_w: str,
                   pos_bias: str, norm_w: str, buf_kv: str, buf_gate: str,
                   entry: str, ov_kv: str, ov_gate: str, comp_pre: str,
                   Dh: int, M: int):
    """store_compression_weights → window view → pool → rope → update_compressor_states。"""
    kv = linear(g, f"{prefix}_kv_gate_proj", x, kv_w, f"{prefix}_kv")
    gate = linear(g, f"{prefix}_kv_gate_proj", x, gate_w, f"{prefix}_gate")
    g.set_group_label(f"{prefix}_kv_gate_proj", "kv_proj + gate_proj（两个 Linear，2×Dh）")
    cat_kv = g.node("Concat", [buf_kv, kv], [f"{prefix}_cat_kv"], group=f"{prefix}_store_window", axis=1)
    cat_gate = g.node("Concat", [buf_gate, gate], [f"{prefix}_cat_gate"], group=f"{prefix}_store_window", axis=1)
    shape = g.node("Shape", [cat_kv], [f"{prefix}_cat_shape"], group=f"{prefix}_store_window")
    bdim = g.node("Slice", [shape, g.const(f"{prefix}/b0", np.array([0], np.int64)),
                            g.const(f"{prefix}/b1", np.array([1], np.int64)),
                            g.const(f"{prefix}/ax0", np.array([0], np.int64))],
                  [f"{prefix}_bdim"], group=f"{prefix}_store_window")
    total = g.node("Gather", [shape, g.const(f"{prefix}/idx1", np.array(1, np.int64))],
                   [f"{prefix}_total"], group=f"{prefix}_store_window")
    total_f = g.node("Cast", [total], [f"{prefix}_total_f"], group=f"{prefix}_store_window", to=1)
    m_f = g.const(f"{prefix}/m_f", np.float32(M))
    fl = g.node("Div", [total_f, m_f], [f"{prefix}_div"], group=f"{prefix}_store_window")
    fl = g.node("Floor", [fl], [f"{prefix}_floor"], group=f"{prefix}_store_window")
    usable_f = g.node("Mul", [fl, m_f], [f"{prefix}_usable_f"], group=f"{prefix}_store_window")
    usable = g.node("Cast", [usable_f], [f"{prefix}_usable"], group=f"{prefix}_store_window", to=7)
    nwin = g.node("Cast", [fl], [f"{prefix}_nwin"], group=f"{prefix}_store_window", to=7)
    usable_v = _vec(g, f"{prefix}_usable_v", usable, f"{prefix}_store_window")
    g.node("Slice", [cat_kv, g.const(f"{prefix}/s0", np.array([0], np.int64)), usable_v,
                     g.const(f"{prefix}/ax1", np.array([1], np.int64))],
           [f"{prefix}_chunk_kv"], group=f"{prefix}_store_window")
    g.node("Slice", [cat_gate, g.const(f"{prefix}/s0b", np.array([0], np.int64)), usable_v,
                     g.const(f"{prefix}/ax1b", np.array([1], np.int64))],
           [f"{prefix}_chunk_gate"], group=f"{prefix}_store_window")
    g.node("Slice", [cat_kv, usable_v, g.const(f"{prefix}/imax", np.array([INT_MAX], np.int64)),
                     g.const(f"{prefix}/ax1c", np.array([1], np.int64))],
           [f"{prefix}_buffer_kv_post"], group=f"{prefix}_store_window")
    g.node("Slice", [cat_gate, usable_v, g.const(f"{prefix}/imax2", np.array([INT_MAX], np.int64)),
                     g.const(f"{prefix}/ax1d", np.array([1], np.int64))],
           [f"{prefix}_buffer_gate_post"], group=f"{prefix}_store_window")
    g.set_group_label(f"{prefix}_store_window",
                      "store_compression_weights：usable=floor((Tbuf+T)/M)*M；chunk 前缀 + 余量回写")
    # 零窗口补齐，保证 N_win=0 时 reshape/softmax 形状合法；输出最后按 nwin 切回
    pad_shape = g.node("Concat", [bdim, g.const(f"{prefix}/pad_tail", np.array([M, 2 * Dh], np.int64))],
                       [f"{prefix}_pad_shape"], group=f"{prefix}_window_view", axis=0)
    zero = g.const(f"{prefix}/zero", np.float32(0.0))
    pad = g.node("Expand", [zero, f"{prefix}_pad_shape"], [f"{prefix}_pad"], group=f"{prefix}_window_view")
    catp = g.node("Concat", [cat_kv, pad], [f"{prefix}_catp_kv"], group=f"{prefix}_window_view", axis=1)
    gatep = g.node("Concat", [cat_gate, pad], [f"{prefix}_catp_gate"], group=f"{prefix}_window_view", axis=1)
    one = g.const(f"{prefix}/one", np.int64(1))
    neff = g.node("Max", [nwin, one], [f"{prefix}_neff"], group=f"{prefix}_window_view")
    ueff = g.node("Mul", [neff, g.const(f"{prefix}/m_i", np.int64(M))], [f"{prefix}_ueff"], group=f"{prefix}_window_view")
    ueff_v = _vec(g, f"{prefix}_ueff_v", ueff, f"{prefix}_window_view")
    neff_v = _vec(g, f"{prefix}_neff_v", neff, f"{prefix}_window_view")
    ch = g.node("Slice", [catp, g.const(f"{prefix}/z0", np.array([0], np.int64)), ueff_v,
                          g.const(f"{prefix}/ax1e", np.array([1], np.int64))],
                [f"{prefix}_chunk_eff_kv"], group=f"{prefix}_window_view")
    cg = g.node("Slice", [gatep, g.const(f"{prefix}/z0b", np.array([0], np.int64)), ueff_v,
                          g.const(f"{prefix}/ax1f", np.array([1], np.int64))],
                [f"{prefix}_chunk_eff_gate"], group=f"{prefix}_window_view")
    neff_u = g.node("Unsqueeze", [neff], [f"{prefix}_neff_u"], group=f"{prefix}_window_view", axes=[0])
    tgt = g.node("Concat", [bdim, neff_u, g.const(f"{prefix}/mtail", np.array([M, 2 * Dh], np.int64))],
                 [f"{prefix}_rs"], group=f"{prefix}_window_view", axis=0)
    kv4 = g.node("Reshape", [ch, tgt], [f"{prefix}_kv4"], group=f"{prefix}_window_view")
    gate4 = g.node("Reshape", [cg, tgt], [f"{prefix}_gate4"], group=f"{prefix}_window_view")
    gate4 = g.node("Add", [gate4, pos_bias], [f"{prefix}_gate4b"], group=f"{prefix}_window_view")
    g.set_group_label(f"{prefix}_window_view",
                      "chunk.view(B,N_win,M,2Dh)；gate + position_bias；Ca|Cb 两半 + 前一窗 Ca 前移")
    ca = g.node("Slice", [kv4, g.const(f"{prefix}/ca0", np.array([0], np.int64)),
                          g.const(f"{prefix}/ca1", np.array([Dh], np.int64)),
                          g.const(f"{prefix}/ax3", np.array([3], np.int64))],
                [f"{prefix}_ca"], group=f"{prefix}_window_view")
    cb = g.node("Slice", [kv4, g.const(f"{prefix}/cb0", np.array([Dh], np.int64)),
                          g.const(f"{prefix}/cb1", np.array([2 * Dh], np.int64)),
                          g.const(f"{prefix}/ax3b", np.array([3], np.int64))],
                [f"{prefix}_cb"], group=f"{prefix}_window_view")
    cag = g.node("Slice", [gate4, g.const(f"{prefix}/cag0", np.array([0], np.int64)),
                           g.const(f"{prefix}/cag1", np.array([Dh], np.int64)),
                           g.const(f"{prefix}/ax3c", np.array([3], np.int64))],
                 [f"{prefix}_cag"], group=f"{prefix}_window_view")
    cbg = g.node("Slice", [gate4, g.const(f"{prefix}/cbg0", np.array([Dh], np.int64)),
                           g.const(f"{prefix}/cbg1", np.array([2 * Dh], np.int64)),
                           g.const(f"{prefix}/ax3d", np.array([3], np.int64))],
                 [f"{prefix}_cbg"], group=f"{prefix}_window_view")
    ovu = g.node("Unsqueeze", [ov_kv], [f"{prefix}_ov_u"], group=f"{prefix}_window_view", axes=[1])
    ovgu = g.node("Unsqueeze", [ov_gate], [f"{prefix}_ovg_u"], group=f"{prefix}_window_view", axes=[1])
    cacat = g.node("Concat", [ovu, ca], [f"{prefix}_ca_cat"], group=f"{prefix}_window_view", axis=1)
    cgcat = g.node("Concat", [ovgu, cag], [f"{prefix}_cg_cat"], group=f"{prefix}_window_view", axis=1)
    cash = g.node("Slice", [cacat, g.const(f"{prefix}/s0c", np.array([0], np.int64)), neff_v,
                            g.const(f"{prefix}/ax1g", np.array([1], np.int64))],
                  [f"{prefix}_ca_shift"], group=f"{prefix}_window_view")
    cgsh = g.node("Slice", [cgcat, g.const(f"{prefix}/s0d", np.array([0], np.int64)), neff_v,
                            g.const(f"{prefix}/ax1h", np.array([1], np.int64))],
                  [f"{prefix}_cg_shift"], group=f"{prefix}_window_view")
    newkv = g.node("Concat", [cash, cb], [f"{prefix}_new_kv"], group=f"{prefix}_window_view", axis=2)
    newgate = g.node("Concat", [cgsh, cbg], [f"{prefix}_new_gate"], group=f"{prefix}_window_view", axis=2)
    probs = g.node("Softmax", [newgate], [f"{prefix}_probs"], group=f"{prefix}_pool", axis=2)
    mul = g.node("Mul", [newkv, probs], [f"{prefix}_probs_kv"], group=f"{prefix}_pool")
    pooled = g.node("ReduceSum", [mul], [f"{prefix}_pooled"], group=f"{prefix}_pool", axes=[2], keepdims=0)
    g.set_group_label(f"{prefix}_pool", "softmax(gate,dim=2) * kv → sum(dim=2) → RMSNorm")
    comp_eff = rmsnorm(g, f"{prefix}_pool", pooled, f"{prefix}_compressed_eff", ctx.p("EpsN"), norm_w)
    # positions = index*M + entry_count*M
    rng = g.node("Range", [g.const(f"{prefix}/r0", np.int64(0)), neff, g.const(f"{prefix}/r1", np.int64(1))],
                 [f"{prefix}_idx"], group=f"{prefix}_rope_gen")
    pos = g.node("Mul", [rng, g.const(f"{prefix}/m_i2", np.int64(M))], [f"{prefix}_idx_m"], group=f"{prefix}_rope_gen")
    first = g.node("Mul", [entry, g.const(f"{prefix}/m_i3", np.int64(M))], [f"{prefix}_first"], group=f"{prefix}_rope_gen")
    pos = g.node("Add", [pos, first], [f"{prefix}_positions"], group=f"{prefix}_rope_gen")
    pos = g.node("Unsqueeze", [pos], [f"{prefix}_pos_u"], group=f"{prefix}_rope_gen", axes=[0])
    g.set_group_label(f"{prefix}_rope_gen", "positions = arange(N_win)*M + entry_count*M")
    rope_cos_sin(g, f"{prefix}_rope_gen", f"{prefix}_pos_u", ctx.rope, f"{prefix}_cos", f"{prefix}_sin")
    cr = g.node("Unsqueeze", [comp_eff], [f"{prefix}_comp_u"], group=f"{prefix}_rope", axes=[1])
    rope_apply(g, f"{prefix}_rope", cr, f"{prefix}_cos", f"{prefix}_sin", int(ctx.p("Rk")), Dh,
               f"{prefix}_rope_out")
    g.node("Squeeze", [f"{prefix}_rope_out", g.const(f"{prefix}/sq_ax", np.array([1], np.int64))],
           [f"{prefix}_compressed_eff_rope"], group=f"{prefix}_rope")
    g.set_group_label(f"{prefix}_rope", "apply_rotary_pos_emb(compressed.unsqueeze(1))")
    # 只保留真实 N_win 条（N_win=0 → 空）；overlap 用 Where 保持前态
    nwin_v = _vec(g, f"{prefix}_nwin_v", nwin, f"{prefix}_update_states")
    g.node("Slice", [f"{prefix}_compressed_eff_rope", g.const(f"{prefix}/o0", np.array([0], np.int64)), nwin_v,
                     g.const(f"{prefix}/ax1i", np.array([1], np.int64))],
           [f"{prefix}_compressed"], group=f"{prefix}_update_states")
    cat = g.node("Concat", [comp_pre, f"{prefix}_compressed"], [f"{prefix}_compressed_post"],
                 group=f"{prefix}_update_states", axis=1)
    entry_post = g.node("Add", [entry, nwin], [f"{prefix}_entry_count_post"], group=f"{prefix}_update_states")
    g.set_group_label(f"{prefix}_update_states",
                      "update_compressor_states：cat(前态, 本次)；entry_count += N_win（可 0）")
    last = g.node("Sub", [neff, one], [f"{prefix}_last"], group=f"{prefix}_overlap")
    last = g.node("Unsqueeze", [last], [f"{prefix}_last_u"], group=f"{prefix}_overlap", axes=[0])
    last1 = g.node("Add", [last, one], [f"{prefix}_last1"], group=f"{prefix}_overlap")
    cand = g.node("Slice", [ca, last, last1, g.const(f"{prefix}/ax1j", np.array([1], np.int64))],
                  [f"{prefix}_ov_cand"], group=f"{prefix}_overlap")
    cand = g.node("Squeeze", [cand, g.const(f"{prefix}/sq1", np.array([1], np.int64))],
                  [f"{prefix}_ov_cand_s"], group=f"{prefix}_overlap")
    candg = g.node("Slice", [cag, last, last1, g.const(f"{prefix}/ax1k", np.array([1], np.int64))],
                   [f"{prefix}_ovg_cand"], group=f"{prefix}_overlap")
    candg = g.node("Squeeze", [candg, g.const(f"{prefix}/sq2", np.array([1], np.int64))],
                   [f"{prefix}_ovg_cand_s"], group=f"{prefix}_overlap")
    cond = g.node("Greater", [nwin, g.const(f"{prefix}/z_i", np.int64(0))], [f"{prefix}_nwin_gt0"],
                  group=f"{prefix}_overlap")
    g.node("Where", [cond, cand, ov_kv], [f"{prefix}_overlap_kv_post"], group=f"{prefix}_overlap")
    g.node("Where", [cond, candg, ov_gate], [f"{prefix}_overlap_gate_post"], group=f"{prefix}_overlap")
    g.set_group_label(f"{prefix}_overlap",
                      "仅 N_win>0 写回最后满窗 Ca 尾；N_win=0 保持前态（Where）")
    gmap = ({"kv_gate_proj": "c_kv_gate_proj", "store_window": "c_store_window",
             "window_view": "c_window_view", "pool": "c_pool", "rope_gen": "c_rope_gen",
             "rope": "c_rope", "update_states": "c_update_states", "overlap": "c_update_states"}
            if prefix == "c" else
            {"kv_gate_proj": "i_kv_gate_proj", "store_window": "i_store_window",
             "window_view": "i_window_view", "pool": "i_pool_rope", "rope_gen": "i_pool_rope",
             "rope": "i_pool_rope", "update_states": "i_update_states", "overlap": "i_update_states"})
    for suf, target in gmap.items():
        src_key = f"{prefix}_{suf}"
        if src_key not in g.groups or src_key == target:
            continue
        dst = g.groups.setdefault(target, {"label": g.groups[src_key]["label"], "nodes": []})
        dst["nodes"].extend(g.groups[src_key]["nodes"])
        for nd in dst["nodes"]:
            if nd["name"].startswith(f"{src_key}/"):
                nd["name"] = nd["name"].replace(f"{src_key}/", f"{target}/", 1)
        del g.groups[src_key]
    return {"kv_post": f"{prefix}_buffer_kv_post", "gate_post": f"{prefix}_buffer_gate_post",
            "comp_post": f"{prefix}_compressed_post", "entry_post": f"{prefix}_entry_count_post",
            "ovkv_post": f"{prefix}_overlap_kv_post", "ovg_post": f"{prefix}_overlap_gate_post"}


def build_attn_csa(ctx: Ctx) -> GB:
    g = GB("attn_csa", ctx.pid)
    Dh, M = int(ctx.p("Dh")), int(ctx.p("M"))
    x = ctx.act_in(g, "attn_normed")
    names = _window_common(
        g, ctx, "c", x,
        ctx.weight(g, "c_kv_gate_proj", "c_kv_weight"),
        ctx.weight(g, "c_kv_gate_proj", "c_gate_weight"),
        ctx.weight(g, "c_position_bias", "c_position_bias"),
        ctx.weight(g, "c_pool", "c_kv_norm_weight"),
        g.inp("cache_buffer_c_kv", ctx.sh("cache_buffer_c_kv"), DTYPE, "cache_pre"),
        g.inp("cache_buffer_c_gate", ctx.sh("cache_buffer_c_gate"), DTYPE, "cache_pre"),
        g.inp("cache_entry_count_c", ctx.sh("cache_entry_count_c"), INT64, "cache_pre"),
        g.inp("cache_overlap_c_kv", ctx.sh("cache_overlap_c_kv"), DTYPE, "cache_pre"),
        g.inp("cache_overlap_c_gate", ctx.sh("cache_overlap_c_gate"), DTYPE, "cache_pre"),
        g.inp("cache_compressed_c", ctx.sh("cache_compressed_c"), DTYPE, "cache_pre"),
        Dh, M)
    for key, jid, dt in (("kv_post", "cache_buffer_c_kv_post", "float32"),
                         ("gate_post", "cache_buffer_c_gate_post", "float32"),
                         ("comp_post", "cache_compressed_c_post", "float32"),
                         ("entry_post", "cache_entry_count_c_post", "int64"),
                         ("ovkv_post", "cache_overlap_c_kv_post", "float32"),
                         ("ovg_post", "cache_overlap_c_gate_post", "float32")):
        g.node("Identity", [names[key]], [jid], group="c_update_states")
        ctx.out_t(g, jid, role="cache_post", dtype=dt)
    return g


def build_attn_indexer(ctx: Ctx) -> GB:
    g = GB("attn_indexer", ctx.pid)
    Id, Ih, Ki, M = int(ctx.p("Id")), int(ctx.p("Ih")), int(ctx.p("Ki")), int(ctx.p("M"))
    x = ctx.act_in(g, "attn_normed")
    qres = ctx.act_in(g, "q_a_normed")
    names = _window_common(
        g, ctx, "i", x,
        ctx.weight(g, "i_kv_gate_proj", "i_kv_weight"),
        ctx.weight(g, "i_kv_gate_proj", "i_gate_weight"),
        ctx.weight(g, "i_position_bias", "i_position_bias"),
        ctx.weight(g, "i_pool_rope", "i_kv_norm_weight"),
        g.inp("cache_buffer_i_kv", ctx.sh("cache_buffer_i_kv"), DTYPE, "cache_pre"),
        g.inp("cache_buffer_i_gate", ctx.sh("cache_buffer_i_gate"), DTYPE, "cache_pre"),
        g.inp("cache_entry_count_i", ctx.sh("cache_entry_count_i"), INT64, "cache_pre"),
        g.inp("cache_overlap_i_kv", ctx.sh("cache_overlap_i_kv"), DTYPE, "cache_pre"),
        g.inp("cache_overlap_i_gate", ctx.sh("cache_overlap_i_gate"), DTYPE, "cache_pre"),
        g.inp("cache_compressed_i", ctx.sh("cache_compressed_i"), DTYPE, "cache_pre"),
        Id, M)
    # i_q_proj（q_residual → Ih×Id，reshape/rope）
    qw = ctx.weight(g, "i_q_proj", "i_q_weight")
    qflat = linear(g, "i_q_proj", qres, qw, "i_q_flat")
    g.node("Reshape", [qflat, g.const("i_q_proj/rs", np.array([0, 0, Ih, Id], np.int64))],
           ["i_q_4d"], group="i_q_proj")
    g.node("Transpose", ["i_q_4d"], ["i_q_head"], group="i_q_proj", perm=[0, 2, 1, 3])
    g.set_group_label("i_q_proj", "q_b_proj(q_residual).view(B,T,Ih,Id).transpose(1,2)")
    ctx.pos_in(g)
    rope_cos_sin(g, "i_q_proj", "position_ids", ctx.rope, "cos_q_i", "sin_q_i")
    rope_apply(g, "i_q_proj", "i_q_head", "cos_q_i", "sin_q_i", int(ctx.p("Rk")), Id, "i_q_rope_h")
    g.node("Transpose", ["i_q_rope_h"], ["i_q"], group="i_q_proj", perm=[0, 2, 1, 3])
    # scorer
    comp = names["comp_post"]
    compT = g.node("Transpose", [comp], ["comp_T"], group="i_scorer", perm=[0, 2, 1])
    compU = g.node("Unsqueeze", [compT], ["comp_U"], group="i_scorer", axes=[1])
    qf = g.node("Cast", ["i_q"], ["i_q_f"], group="i_scorer", to=1)
    cf = g.node("Cast", [compU], ["comp_f"], group="i_scorer", to=1)
    sc = g.node("MatMul", [qf, cf], ["scores_raw"], group="i_scorer")
    relu = g.node("Relu", [sc], ["scores_relu"], group="i_scorer")
    sc = g.node("Mul", [relu, g.const("i_scorer/scale", np.float32(Id ** -0.5))], ["scores_scaled"],
                group="i_scorer")
    wproj = linear(g, "i_scorer", x, ctx.weight(g, "i_scorer", "i_w_weight"), "scorer_w")
    wsc = g.node("Mul", [wproj, g.const("i_scorer/wsc", np.float32(Ih ** -0.5))], ["scorer_w_scaled"],
                 group="i_scorer")
    wu = g.node("Unsqueeze", [wsc], ["scorer_w_u"], group="i_scorer", axes=[-1])
    term = g.node("Mul", [sc, wu], ["scores_weighted"], group="i_scorer")
    g.node("ReduceSum", [term], ["index_scores"], group="i_scorer", axes=[2], keepdims=0)
    g.set_group_label("i_scorer", "relu(q·Kᵀ)*Id^-0.5 × weights(Ih^-0.5) → sum(heads)")
    # topk with causal threshold
    pos = ctx.pos_in(g)
    pos1 = g.node("Add", [pos, g.const("i_topk/one", np.int64(1))], ["pos_p1"], group="i_topk")
    thr = g.node("Div", [pos1, g.const("i_topk/m", np.int64(M))], ["causal_thr"], group="i_topk")
    comp_len = g.node("Gather", [g.node("Shape", [comp], ["comp_shape"], group="i_topk"),
                                 g.const("i_topk/idx1", np.array(1, np.int64))],
                      ["comp_len"], group="i_topk")
    rng = g.node("Range", [g.const("i_topk/z", np.int64(0)), comp_len, g.const("i_topk/o", np.int64(1))],
                 ["entry_idx"], group="i_topk")
    tru = g.node("Unsqueeze", [thr], ["thr_u"], group="i_topk", axes=[-1])
    fut = g.node("GreaterOrEqual", [g.node("Unsqueeze", [rng], ["rng_u"], group="i_topk", axes=[0]),
                                    tru], ["future_mask"], group="i_topk")
    neg = g.const("i_topk/ninf", np.float32(float("-inf")))
    masked = g.node("Where", [fut, neg, "index_scores"], ["scores_masked"], group="i_topk")
    kk = g.node("Min", [g.const("i_topk/ki", np.int64(Ki)), comp_len], ["topk_k"], group="i_topk")
    kk = g.node("Reshape", [kk, g.const("i_topk/krs", np.array([1], np.int64))], ["topk_k1"], group="i_topk")
    tv, ti = g.node("TopK", [masked, kk], ["topk_vals", "topk_idx_raw"], group="i_topk", axis=-1)
    inv = g.node("Greater", [ti, g.node("Sub", [tru, g.const("i_topk/one2", np.int64(1))], ["thr_m1"], group="i_topk")],
                 ["invalid"], group="i_topk")
    neg1 = g.const("i_topk/neg1", np.int64(-1))
    topk = g.node("Where", [inv, neg1, ti], ["indexer_topk"], group="i_topk")
    g.set_group_label("i_topk", "causal_threshold=(pos+1)//M；越界置 -1；TopK(k=min(Ki,N_comp))")
    # block_bias：长度取 **CSA** compressed（cache_compressed_c_post），不是 Indexer 自己的；
    # 哨兵列 N+1 → scatter 0 → 最后切回 [..., :N_comp]（源码 :690-706）。
    csa_comp = ctx.act_in(g, "cache_compressed_c_post")
    csa_len = g.node("Gather", [g.node("Shape", [csa_comp], ["csa_shape"], group="c_block_bias"),
                                g.const("cb/idx1", np.array(1, np.int64))],
                     ["csa_comp_len"], group="c_block_bias")
    safe = g.node("Where", [inv, g.node("Expand", [csa_len,
                                              g.node("Shape", [topk], ["topk_shape"], group="c_block_bias")],
                                     ["comp_len_shape"], group="c_block_bias"), topk],
                  ["safe_idx"], group="c_block_bias")
    n1 = g.node("Add", [csa_len, g.const("cb/one", np.int64(1))], ["comp_len_p1"], group="c_block_bias")
    bb_shape = g.node("Concat", [g.node("Slice", [g.node("Shape", [csa_comp], ["cshape"], group="c_block_bias"),
                                                  g.const("cb/b0", np.array([0], np.int64)),
                                                  g.const("cb/b1", np.array([1], np.int64)),
                                                  g.const("cb/ax0", np.array([0], np.int64))],
                                        ["cB"], group="c_block_bias"),
                                  g.const("cb/one1", np.array([1], np.int64)),
                                  g.node("Slice", [g.node("Shape", [topk], ["tshape"], group="c_block_bias"),
                                                   g.const("cb/t0", np.array([1], np.int64)),
                                                   g.const("cb/t1", np.array([2], np.int64)),
                                                   g.const("cb/ax1", np.array([0], np.int64))],
                                         ["cT"], group="c_block_bias"),
                                  g.node("Unsqueeze", [n1], ["n1_u"], group="c_block_bias", axes=[0])],
                        ["bb_shape"], group="c_block_bias", axis=0)
    bb = g.node("Expand", [neg, bb_shape], ["bb_full"], group="c_block_bias")
    sidx = g.node("Unsqueeze", [safe], ["safe_u"], group="c_block_bias", axes=[1])
    zval = g.const("cb/zero", np.float32(0.0))
    # ScatterElements updates 必须与 indices 同 rank/可广播：显式 Expand 成同 shape 的 0
    updates = g.node("Expand", [zval, g.node("Shape", [sidx], ["sidx_shape"], group="c_block_bias")],
                     ["cb_updates"], group="c_block_bias")
    bb = g.node("ScatterElements", [bb, sidx, updates], ["bb_scattered"], group="c_block_bias", axis=-1)
    csa_len_v = _vec(g, "cb_csa_len_v", csa_len, "c_block_bias")
    g.node("Slice", [bb, g.const("cb/s0", np.array([0], np.int64)), csa_len_v,
                     g.const("cb/ax2", np.array([-1], np.int64))],
           ["c_block_bias"], group="c_block_bias")
    g.set_group_label("c_block_bias", "-inf 张量 + ScatterElements(置 0) + 去尾列 → per-query block bias")
    ctx.out_t(g, "indexer_topk", dtype="int64")
    ctx.out_t(g, "c_block_bias")
    for key, jid, dt in (("kv_post", "cache_buffer_i_kv_post", "float32"),
                         ("gate_post", "cache_buffer_i_gate_post", "float32"),
                         ("comp_post", "cache_compressed_i_post", "float32"),
                         ("entry_post", "cache_entry_count_i_post", "int64"),
                         ("ovkv_post", "cache_overlap_i_kv_post", "float32"),
                         ("ovg_post", "cache_overlap_i_gate_post", "float32")):
        g.node("Identity", [names[key]], [jid], group="i_update_states")
        ctx.out_t(g, jid, role="cache_post", dtype=dt)
    return g


def build_attn_mask(ctx: Ctx) -> GB:
    g = GB("attn_mask", ctx.pid)
    W = int(ctx.p("W"))
    B = g.inp("B", [], INT64, "shape")
    cum = g.inp("cache_cumulative_length", [], INT64, "cache_pre")
    T = g.inp("T", [], INT64, "shape")
    pos = g.node("Add", [cum, g.node("Range", [g.const("mask_model_build/z", np.int64(0)), T,
                                             g.const("mask_model_build/o", np.int64(1))],
                                     ["pos_rng"], group="mask_model_build")], ["positions"], group="mask_model_build")
    L = g.node("Min", [cum, g.const("mask_model_build/wm1", np.int64(W - 1))], ["pm1"], group="mask_model_build")
    L = g.node("Add", [L, T], ["L_mask"], group="mask_model_build")
    k = g.node("Range", [g.const("mask_model_build/z2", np.int64(0)), L,
                         g.const("mask_model_build/o2", np.int64(1))], ["key_idx"], group="mask_model_build")
    # key 的绝对位置 = max(0, P-(W-1)) + k（本地窗口内 k 不是绝对位置）
    base = g.node("Sub", [cum, g.const("mask_model_build/wm1b", np.int64(W - 1))], ["win_base0"],
                  group="mask_model_build")
    base = g.node("Max", [base, g.const("mask_model_build/z3", np.int64(0))], ["win_base"],
                  group="mask_model_build")
    kg = g.node("Add", [k, base], ["key_global"], group="mask_model_build")
    ku = g.node("Unsqueeze", [kg], ["k_u"], group="mask_model_build", axes=[0])
    pu = g.node("Unsqueeze", [pos], ["p_u"], group="mask_model_build", axes=[-1])
    le = g.node("LessOrEqual", [ku, pu], ["causal_ok"], group="mask_model_build")
    lower = g.node("Sub", [pu, g.const("mask_model_build/w", np.int64(W))], ["lower"], group="mask_model_build")
    gt = g.node("Greater", [ku, lower], ["window_ok"], group="mask_model_build")
    ok = g.node("And", [le, gt], ["mask_ok"], group="mask_model_build")
    z = g.const("mask_model_build/zero", np.float32(0.0))
    ninf = g.const("mask_model_build/ninf", np.float32(float("-inf")))
    m2 = g.node("Where", [ok, z, ninf], ["mask2d"], group="mask_model_build")
    m3 = g.node("Unsqueeze", [m2], ["mask3d"], group="mask_model_build", axes=[0])
    m4 = g.node("Unsqueeze", ["mask3d"], ["mask4d"], group="mask_model_build", axes=[0])
    bb = g.node("Concat", [g.node("Unsqueeze", [B], ["B_u"], group="mask_model_build", axes=[0]),
                           g.const("mask_model_build/shape_rest", np.array([1], np.int64)),
                           g.node("Unsqueeze", [T], ["T_u"], group="mask_model_build", axes=[0]),
                           g.node("Unsqueeze", [L], ["L_u"], group="mask_model_build", axes=[0])],
                ["mask_shape"], group="mask_model_build", axis=0)
    g.node("Expand", ["mask4d", "mask_shape"], ["mask_model"], group="mask_model_build")
    g.set_group_label("mask_model_build",
                      "causal + sliding-window mask（Range/Compare/Where）；2D padding 路径未覆盖")
    bias = ctx.act_in(g, "c_block_bias")
    g.node("Concat", ["mask_model", bias], ["mask_ext"], group="mask_extend", axis=-1)
    g.set_group_label("mask_extend", "cat(mask, block_bias, dim=-1)")
    ctx.out_t(g, "mask_model")
    ctx.out_t(g, "mask_ext")
    return g
