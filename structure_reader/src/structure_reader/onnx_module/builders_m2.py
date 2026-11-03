"""M2 变体 ONNX 数学 builder（HCA 压缩器 / SWA attention / Hash 路由）。

FP32 阅读特化；不加载权重、不执行 HF。只生成真实 ModelProto 供小例数值验证。
"""
from __future__ import annotations

import numpy as np

from structure_reader.onnx_module.modeling import (
    DTYPE, FLOAT, GB, INT64, linear, rmsnorm, rope_apply, rope_cos_sin)


def _shape_of(g: GB, x: str, group: str) -> str:
    return g.node("Shape", [x], [f"{group}/shape"], group=group)


def _slice(g: GB, x: str, start, end, axis: int, group: str, out: str) -> str:
    s = start if isinstance(start, str) else g.const(f"{group}/{out}_s", np.array(start, np.int64))
    if isinstance(end, str):
        e = end
    elif end == 2 ** 31 - 1:
        e = g.const(f"{group}/{out}_e", np.array([2 ** 31 - 1], np.int64))
    else:
        e = g.const(f"{group}/{out}_e", np.array(end, np.int64))
    ax = g.const(f"{group}/{out}_ax", np.array([axis], np.int64))
    return g.node("Slice", [x, s, e, ax], [out], group=group)


def build_hca_compressor(*, batch: int = 1, seq: int = 5,
                         buf: int = 0, hist: int = 2,
                         dh: int = 8, m: int = 4, rk: int = 2,
                         rope: dict | None = None, eps: float = 1e-6) -> GB:
    """HCA（定形 fixture）：buffer+kv→M 对齐前缀/余量→非重叠 view→
    gate+position_bias softmax→RMSNorm→entry_count*M+arange 位置 RoPE→追加历史。"""
    rope = rope or {"inv_freq": (1.0 / (10000.0 ** (np.arange(0, rk, 2) / rk))).astype(np.float32),
                    "dim": rk, "attention_factor": 1.0}
    assert isinstance(seq, int) and isinstance(buf, int) and isinstance(hist, int)
    nwin = (seq + buf) // m
    usable = nwin * m
    g = GB("hca_compressor", "prefill")
    kv = g.inp("kv", [batch, seq, dh], FLOAT)
    gate = g.inp("gate", [batch, seq, dh], FLOAT)
    bkv = g.inp("buffer_kv", [batch, buf, dh], FLOAT)
    bgate = g.inp("buffer_gate", [batch, buf, dh], FLOAT)
    pbia = g.inp("position_bias", [m, dh], FLOAT)
    chist = g.inp("compressed_kv", [batch, hist, dh], FLOAT)
    ecount = g.inp("entry_count", [], INT64)
    posq = g.inp("position_ids", [1, seq], INT64)

    cat_kv = g.node("Concat", [bkv, kv], ["cat_kv"], group="merge", axis=1)
    cat_gt = g.node("Concat", [bgate, gate], ["cat_gate"], group="merge", axis=1)
    pre_kv = _slice(g, cat_kv, [0], [usable], 1, "windows", "pre_kv")
    pre_gt = _slice(g, cat_gt, [0], [usable], 1, "windows", "pre_gt")
    rem_kv = _slice(g, cat_kv, [usable], [2 ** 31 - 1], 1, "windows", "buffer_kv_post")
    rem_gt = _slice(g, cat_gt, [usable], [2 ** 31 - 1], 1, "windows", "rem_gate")
    shp = g.const("win_shape", np.array([batch, nwin, m, dh], np.int64))
    win_kv = g.node("Reshape", [pre_kv, shp], ["win_kv"], group="windows")
    win_gt = g.node("Reshape", [pre_gt, shp], ["win_gt"], group="windows")
    biased = g.node("Add", [win_gt, pbia], ["biased_gate"], group="pool")
    soft = g.node("Softmax", [biased], ["gate_prob"], group="pool", axis=2)
    prod = g.node("Mul", [win_kv, soft], ["weighted"], group="pool")
    pooled = g.node("ReduceSum", [prod], ["pooled"], group="pool", axes=[2], keepdims=0)
    norms = g.inp("kv_norm_weight", [dh], FLOAT)
    compressed = rmsnorm(g, "norm", pooled, "compressed", eps, norms)
    ar = g.const("arange_nwin", np.arange(nwin, dtype=np.int64))
    first = g.node("Mul", [ecount, g.const("m", np.array(m, np.int64))], ["first"], group="rope")
    positions = g.node("Add", [first, ar], ["positions"], group="rope")
    pos2 = g.node("Unsqueeze", [positions], ["pos2"], group="rope", axes=[0])
    cos, sin = "cos", "sin"
    rope_cos_sin(g, "rope", pos2, rope, cos, sin)
    ck = g.node("Unsqueeze", [compressed], ["compressed_b"], group="rope", axes=[1])
    ckr = rope_apply(g, "rope", ck, cos, sin, rk, dh, "compressed_rope")
    ckr = g.node("Squeeze", [ckr], ["new_compressed"], group="rope", axes=[1])
    new_hist = g.node("Concat", [chist, ckr], ["compressed_post"], group="append", axis=1)
    ec_post = g.node("Add", [ecount, g.const("nwin_c", np.array(nwin, np.int64))], ["entry_count_post"],
                     group="append")
    # 消费侧 bias（None→零扩展）：entry j >= (pos+1)//M → -inf；seq==1 时全 0
    ent = g.const("entries", np.arange(hist + nwin, dtype=np.int64))
    thr = g.node("Div", [g.node("Add", [posq, g.const("one", np.array(1, np.int64))],
                                ["pos1"], group="bias"),
                         g.const("m2", np.array(m, np.int64))], ["thr"], group="bias")
    if seq == 1:
        bias0 = g.const("zero_bias", np.zeros((1, 1, 1, hist + nwin), np.float32))
        bias = g.node("Identity", [bias0], ["block_bias"], group="bias")
    else:
        ent4 = g.node("Unsqueeze", [ent], ["ent4"], group="bias", axes=[0, 1, 2])
        thr4 = g.node("Unsqueeze", [thr], ["thr4a"], group="bias", axes=[-1])
        thr4 = g.node("Unsqueeze", [thr4], ["thr4"], group="bias", axes=[0])
        cmp = g.node("GreaterOrEqual", [ent4, thr4], ["blocked"], group="bias")
        neg = g.const("neg_inf", np.array(-1e9, np.float32))
        zero = g.const("zero_f", np.array(0.0, np.float32))
        bias = g.node("Where", [cmp, neg, zero], ["block_bias"], group="bias")

    g.out("new_compressed", [batch, nwin, dh], FLOAT)
    g.out("buffer_kv_post", [batch, max(0, seq + buf - usable), dh], FLOAT)
    g.out("compressed_post", [batch, hist + nwin, dh], FLOAT)
    g.out("entry_count_post", [], INT64)
    g.out("block_bias", [batch, 1, seq, hist + nwin], FLOAT)
    return g


def build_swa_attention(*, batch: int = 1, seq: int = 3, cache: int = 2,
                        nh: int = 2, dh: int = 8, rk: int = 2,
                        rope: dict | None = None) -> GB:
    """SWA：main partial RoPE；keys/values 独立持久字段；无压缩拼接。"""
    rope = rope or {"inv_freq": (1.0 / (10000.0 ** (np.arange(0, rk, 2) / rk))).astype(np.float32),
                    "dim": rk, "attention_factor": 1.0}
    g = GB("swa_attention", "prefill")
    q = g.inp("q", [batch, nh, seq, dh], FLOAT)
    kv = g.inp("kv", [batch, 1, seq, dh], FLOAT)
    ck = g.inp("cache_keys", [batch, 1, cache, dh], FLOAT)
    cv = g.inp("cache_values", [batch, 1, cache, dh], FLOAT)
    pos = g.inp("position_ids", [1, seq], INT64)
    mask = g.inp("mask", [1, 1, seq, "Lkv"], FLOAT)
    cos, sin = "cos", "sin"
    rope_cos_sin(g, "rope", pos, rope, cos, sin)
    qr = rope_apply(g, "qrope", q, cos, sin, rk, dh, "q_rope")
    kr = rope_apply(g, "kvrope", kv, cos, sin, rk, dh, "kv_rope")
    full_k = g.node("Concat", [ck, kr], ["full_keys"], group="cache", axis=2)
    full_v = g.node("Concat", [cv, kr], ["full_values"], group="cache", axis=2)
    rep = g.const("rep", np.array([1, nh, 1, 1], np.int64))
    vrep = g.node("Tile", [full_v, rep], ["v_repeated"], group="core")
    logits = g.node("MatMul", [qr, g.node("Transpose", [full_k], ["k_t"], group="core", perm=[0, 1, 3, 2])],
                    ["logits"], group="core")
    scale = g.const("scale", np.float32(1.0 / np.sqrt(dh)))
    logits = g.node("Mul", [logits, scale], ["scaled"], group="core")
    masked = g.node("Add", [logits, mask], ["masked"], group="core")
    probs = g.node("Softmax", [masked], ["probs"], group="core", axis=-1)
    out = g.node("MatMul", [probs, vrep], ["attn"], group="core")
    inv = rope_apply(g, "inv", out, cos, sin, rk, dh, "attn_inv", sign=-1.0)
    inv = g.node("Transpose", [inv], ["attn_out"], group="out", perm=[0, 2, 1, 3])
    # 持久 tail：每字段独立切最后 W-1 个（这里取 cache）
    tail_k = _slice(g, full_k, [-cache], [2 ** 31 - 1], 2, "cache", "keys_post")
    tail_v = _slice(g, full_v, [-cache], [2 ** 31 - 1], 2, "cache", "values_post")
    g.out("attn_out", ["B", "T", "Nh", dh], FLOAT)
    g.out("keys_post", ["B", "1", "Lkv_tail", dh], FLOAT)
    g.out("values_post", ["B", "1", "Lkv_tail", dh], FLOAT)
    g.out("full_keys", ["B", "1", "Lkv", dh], FLOAT)
    g.out("full_values", ["B", "1", "Lkv", dh], FLOAT)
    return g


def build_hash_route(*, n: int = 2, h: int = 4, e: int = 4, k: int = 2,
                     vocab: int = 6, eps: float = 1e-20, scale: float = 2.5) -> GB:
    """Hash：input_ids[INT64]+tid2eid[INT64] Gather→indices；learned scores GatherElements；
    无 TopK、无 correction bias。"""
    g = GB("hash_route", "prefill")
    flat = g.inp("flat", [n, h], FLOAT)
    ids = g.inp("input_ids", [n], INT64)
    table = g.inp("tid2eid", [vocab, k], INT64)
    w = g.inp("router_weight", [e, h], FLOAT)
    idx = g.node("Gather", [table, ids], ["indices"], group="lookup", axis=0)
    logits = linear(g, "score", flat, w, "router_logits")
    sp = g.node("Softplus", [logits], ["softplus"], group="score")
    scores = g.node("Sqrt", [sp], ["router_scores"], group="score")
    weights = g.node("GatherElements", [scores, idx], ["gathered"], group="gather", axis=1)
    wsum = g.node("ReduceSum", [weights], ["wsum"], group="gather", axes=[1], keepdims=1)
    wsum = g.node("Add", [wsum, g.const("eps", np.float32(eps))], ["wsum_eps"], group="gather")
    normed = g.node("Div", [weights, wsum], ["normed"], group="gather")
    scaled = g.node("Mul", [normed, g.const("scale", np.float32(scale))], ["weights"], group="gather")
    g.out("indices", [n, k], INT64)
    g.out("weights", [n, k], FLOAT)
    g.out("router_logits", [n, e], FLOAT)
    g.out("router_scores", [n, e], FLOAT)
    return g


def build_hca_csa(ctx) -> GB:
    """生产 HCA 压缩器（Ctx/doc 参数；动态 B/T/Tbuf/Tcc/Nwin；无 Indexer/overlap）。"""
    m = int(ctx.p("M")); dh = int(ctx.p("Dh")); rk = int(ctx.p("Rk"))
    h = int(ctx.p("H")); eps = float(ctx.p("EpsN"))
    rope = ctx.rope_compress
    g = GB("attn_csa", ctx.pid)
    x = ctx.act_in(g, "attn_normed")
    wkv = ctx.weight(g, "c_kv_proj", "c_kv_weight")
    wg = ctx.weight(g, "c_gate_proj", "c_gate_weight")
    bkv = ctx.cache_in(g, "cache_buffer_c_kv")
    bgt = ctx.cache_in(g, "cache_buffer_c_gate")
    pbia = g.inp("c_position_bias", [m, dh], DTYPE, "weight")
    chist = ctx.cache_in(g, "cache_compressed_c")
    ec = g.inp("cache_entry_count_c", [], INT64, "cache_pre")
    wnorm = ctx.weight(g, "c_pool", "c_kv_norm_weight")
    posq = ctx.pos_in(g)

    kv = linear(g, "c_kv_proj", x, wkv, "c_kv")
    gt = linear(g, "c_gate_proj", x, wg, "c_gate")
    ckv = g.node("Concat", [bkv, kv], ["c_chunk_kv_all"], group="c_store_window", axis=1)
    cgt = g.node("Concat", [bgt, gt], ["c_chunk_gate_all"], group="c_store_window", axis=1)
    tot = g.node("Gather", [g.node("Shape", [ckv], ["tot_shape"], group="c_store_window"),
                            g.const("one", np.array(1, np.int64))], ["tot"],
                 group="c_store_window")
    mc = g.const("m", np.array(m, np.int64))
    nwin = g.node("Div", [tot, mc], ["nwin"], group="c_store_window")
    usable = g.node("Mul", [nwin, mc], ["usable"], group="c_store_window")
    usable1 = g.node("Unsqueeze", [usable], ["usable_u"], group="c_store_window", axes=[0])
    pre_kv = _slice(g, ckv, [0], usable1, 1, "c_store_window", "c_chunk_kv")
    pre_gt = _slice(g, cgt, [0], usable1, 1, "c_store_window", "c_chunk_gate")
    rem_kv = _slice(g, ckv, usable1, [2 ** 31 - 1], 1, "c_store_window", "cache_buffer_c_kv_post")
    rem_gt = _slice(g, cgt, usable1, [2 ** 31 - 1], 1, "c_store_window", "cache_buffer_c_gate_post")
    shp = g.node("Concat", [g.const("b0", np.array([0], np.int64)),
                            g.node("Unsqueeze", [nwin], ["nwin_u"], group="c_window_view", axes=[0]),
                            g.const("md", np.array([m, dh], np.int64))], ["win_shape"],
                  group="c_window_view", axis=0)
    wk = g.node("Reshape", [pre_kv, shp], ["c_windows_kv"], group="c_window_view")
    wg2 = g.node("Reshape", [pre_gt, shp], ["c_windows_gate"], group="c_window_view")
    biased = g.node("Add", [wg2, pbia], ["c_gate_bias"], group="c_pool")
    soft = g.node("Softmax", [biased], ["c_gate_prob"], group="c_pool", axis=2)
    prod = g.node("Mul", [wk, soft], ["c_weighted"], group="c_pool")
    pooled = g.node("ReduceSum", [prod], ["c_pooled"], group="c_pool", axes=[2], keepdims=0)
    comp = rmsnorm(g, "c_pool", pooled, "c_compressed", eps, wnorm)

    ar = g.node("Range", [g.const("zero", np.array(0, np.int64)), nwin,
                          g.const("one64", np.array(1, np.int64))], ["arange"], group="c_rope_gen")
    ar_m = g.node("Mul", [ar, mc], ["arange_m"], group="c_rope_gen")
    first = g.node("Mul", [ec, mc], ["first_window_position"], group="c_rope_gen")
    positions = g.node("Add", [first, ar_m], ["c_pos_1d"], group="c_rope_gen")
    pos2 = g.node("Unsqueeze", [positions], ["c_pos2"], group="c_rope_gen", axes=[0])
    bdim = g.node("Gather", [g.node("Shape", [x], ["x_shape"], group="c_rope_gen"),
                             g.const("zero_i", np.array(0, np.int64))], ["bdim"], group="c_rope_gen")
    reps2 = g.node("Concat", [g.node("Unsqueeze", [bdim], ["bdim_u"], group="c_rope_gen", axes=[0]),
                              g.const("one_rep", np.array([1], np.int64))], ["reps2"],
                   group="c_rope_gen", axis=0)
    posb = g.node("Tile", [pos2, reps2], ["c_positions"], group="c_rope_gen")
    posf = g.node("Cast", [posb], ["c_pos_f"], group="c_rope_gen", to=1)
    inv = g.const("c_inv_freq", rope["inv_freq"])
    freqs = g.node("Mul", [g.node("Unsqueeze", [posf], ["c_pos_u"], group="c_rope_gen", axes=[-1]),
                           inv], ["c_freqs"], group="c_rope_gen")
    cos = g.node("Cos", [freqs], ["c_cos"], group="c_rope_gen")
    sin = g.node("Sin", [freqs], ["c_sin"], group="c_rope_gen")
    nope = g.node("Slice", [comp, g.const("n0", np.array([0], np.int64)),
                            g.const("n1", np.array([dh - rk], np.int64)),
                            g.const("nax", np.array([2], np.int64))], ["c_nope"], group="c_rope")
    last = g.node("Slice", [comp, g.const("r0", np.array([dh - rk], np.int64)),
                            g.const("r1", np.array([dh], np.int64)),
                            g.const("rax", np.array([2], np.int64))], ["c_last"], group="c_rope")
    ev = g.const("idx_even", np.arange(0, rk, 2, dtype=np.int64))
    od = g.const("idx_odd", np.arange(1, rk, 2, dtype=np.int64))
    x1 = g.node("Gather", [last, ev], ["c_x1"], group="c_rope", axis=2)
    x2 = g.node("Gather", [last, od], ["c_x2"], group="c_rope", axis=2)
    c3 = cos
    s3 = sin
    r1 = g.node("Sub", [g.node("Mul", [x1, c3], ["c_r1a"], group="c_rope"),
                        g.node("Mul", [x2, s3], ["c_r1b"], group="c_rope")], ["c_rot1"], group="c_rope")
    r2 = g.node("Add", [g.node("Mul", [x1, s3], ["c_r2a"], group="c_rope"),
                        g.node("Mul", [x2, c3], ["c_r2b"], group="c_rope")], ["c_rot2"], group="c_rope")
    pair = g.node("Concat", [g.node("Unsqueeze", [r1], ["c_r1u"], group="c_rope", axes=[3]),
                             g.node("Unsqueeze", [r2], ["c_r2u"], group="c_rope", axes=[3])],
                  ["c_pairs"], group="c_rope", axis=3)
    rot = g.node("Reshape", [pair, g.const("rot_shape", np.array([0, 0, rk], np.int64))],
                 ["c_rot"], group="c_rope")
    ckr = g.node("Concat", [nope, rot], ["c_compressed_rope"], group="c_rope", axis=2)
    new_hist = g.node("Concat", [chist, ckr], ["cache_compressed_c_post"],
                      group="c_update_states", axis=1)
    ecp = g.node("Add", [ec, nwin], ["cache_entry_count_c_post"], group="c_update_states")
    # bias：entry >= (pos+1)//M；T==1 或 comp_len==0 → 零扩展（HF None）
    comp_len = g.node("Gather", [g.node("Shape", [new_hist], ["hist_shape"], group="hca_block_bias"),
                                 g.const("two", np.array(1, np.int64))], ["comp_len"],
                      group="hca_block_bias")
    ent = g.node("Range", [g.const("entry_zero", np.array(0, np.int64)),
                           g.node("Unsqueeze", [comp_len], ["cl_u"], group="hca_block_bias", axes=[0]),
                           g.const("b1", np.array(1, np.int64))], ["entries"],
                 group="hca_block_bias")
    thr = g.node("Div", [g.node("Add", [posq, g.const("one64b", np.array(1, np.int64))],
                                ["pos1"], group="hca_block_bias"), mc], ["thr"],
                 group="hca_block_bias")
    ent4 = g.node("Unsqueeze", [ent], ["ent4"], group="hca_block_bias", axes=[0, 1, 2])
    thr4 = g.node("Unsqueeze", [thr], ["thr4a"], group="hca_block_bias", axes=[-1])
    thr4 = g.node("Unsqueeze", [thr4], ["thr4"], group="hca_block_bias", axes=[0])
    cmp = g.node("GreaterOrEqual", [ent4, thr4], ["blocked"], group="hca_block_bias")
    neg = g.const("neg_inf", np.float32(-np.inf))
    zero = g.const("zero_f", np.float32(0.0))
    raw = g.node("Where", [cmp, neg, zero], ["bias_raw"], group="hca_block_bias")
    tlen = g.node("Gather", [g.node("Shape", [posq], ["pos_shape"], group="hca_block_bias"),
                             g.const("one2", np.array(1, np.int64))], ["tlen"],
                  group="hca_block_bias")
    single = g.node("Equal", [tlen, g.const("one_l", np.array(1, np.int64))], ["is_single"],
                    group="hca_block_bias")
    empty = g.node("Equal", [comp_len, g.const("zero_l", np.array(0, np.int64))], ["is_empty"],
                   group="hca_block_bias")
    none = g.node("Or", [single, empty], ["is_none"], group="hca_block_bias")
    bias = g.node("Where", [none, zero, raw], ["bias_b1"], group="hca_block_bias")
    reps4 = g.node("Concat", [g.node("Unsqueeze", [bdim], ["bdim_u2"], group="hca_block_bias", axes=[0]),
                              g.const("ones4", np.array([1, 1, 1], np.int64))], ["reps4"],
                   group="hca_block_bias", axis=0)
    bias = g.node("Tile", [bias, reps4], ["block_bias"], group="hca_block_bias")

    ctx.out_t(g, "c_chunk_kv", name="c_chunk_kv")
    ctx.out_t(g, "c_chunk_gate", name="c_chunk_gate")
    ctx.out_t(g, "cache_buffer_c_kv_post", name="cache_buffer_c_kv_post")
    ctx.out_t(g, "cache_buffer_c_gate_post", name="cache_buffer_c_gate_post")
    ctx.out_t(g, "c_windows_kv", name="c_windows_kv")
    ctx.out_t(g, "c_compressed", name="c_compressed")
    ctx.out_t(g, "c_positions", name="c_positions", dtype=INT64)
    ctx.out_t(g, "c_compressed_rope", name="c_compressed_rope")
    ctx.out_t(g, "cache_compressed_c_post", name="cache_compressed_c_post")
    ctx.out_t(g, "cache_entry_count_c_post", name="cache_entry_count_c_post", dtype=INT64)
    ctx.out_t(g, "c_cos", name="c_cos")
    ctx.out_t(g, "c_sin", name="c_sin")
    ctx.out_t(g, "block_bias", name="block_bias")
    return g


def _rope_of(ctx):
    return ctx.rope_main if ctx.doc.get("scope", {}).get("attention_type") == "sliding_attention" else ctx.rope_compress


def build_swa_cache(ctx) -> GB:
    """DynamicSlidingWindowLayer：keys/values 独立持久，tail=W-1，与调用前长度无关。"""
    g = GB("attn_cache", ctx.pid)
    ck = ctx.cache_in(g, "cache_keys")
    cv = ctx.cache_in(g, "cache_values")
    kr = ctx.act_in(g, "kv_rope")
    fk = g.node("Concat", [ck, kr], ["kv_full"], group="cache_update", axis=2)
    fv = g.node("Concat", [cv, kr], ["full_values"], group="cache_update", axis=2)
    wkeep = int(ctx.p("W")) - 1
    lk = g.node("Gather", [g.node("Shape", [fk], ["fk_shape"], group="cache_update"),
                           g.const("two_i", np.array(2, np.int64))], ["lk"], group="cache_update")
    start = g.node("Max", [g.node("Sub", [lk, g.const("keep", np.array(wkeep, np.int64))],
                                 ["raw_start"], group="cache_update"),
                           g.const("zero_i", np.array(0, np.int64))], ["start"], group="cache_update")
    s1 = g.node("Unsqueeze", [start], ["start_u"], group="cache_update", axes=[0])
    e1 = g.const("int_max", np.array([2 ** 31 - 1], np.int64))
    ax = g.const("ax2", np.array([2], np.int64))
    kp = g.node("Slice", [fk, s1, e1, ax], ["cache_keys_post"], group="cache_update")
    vp = g.node("Slice", [fv, s1, e1, ax], ["cache_values_post"], group="cache_update")
    cl = g.node("Gather", [g.node("Shape", [kr], ["kr_shape"], group="cache_update"),
                           g.const("two2", np.array(2, np.int64))], ["tlen"], group="cache_update")
    cum = g.node("Add", [ctx.cache_in(g, "cache_cumlen", dtype=INT64), cl],
                 ["cache_cumlen_post"], group="cache_update")
    ctx.out_t(g, "kv_full", name="kv_full")
    ctx.out_t(g, "cache_keys_post", name="cache_keys_post")
    ctx.out_t(g, "cache_values_post", name="cache_values_post")
    ctx.out_t(g, "cache_cumlen_post", name="cache_cumlen_post", dtype="int64")
    return g


def build_swa_core(ctx) -> GB:
    """SWA eager core：直接 kv_full(Lkv)，mask_model；sink cat→softmax→去 sink；V=K full。"""
    g = GB("attn_core", ctx.pid)
    q = ctx.act_in(g, "q_rope")
    kv = ctx.act_in(g, "kv_full")
    mask = ctx.act_in(g, "mask_model")
    sinks = ctx.weight(g, "sink_softmax", "sinks")
    nh = int(ctx.p("Nh"))
    rep = g.const("rep", np.array([1, nh, 1, 1], np.int64))
    kr = g.node("Tile", [kv, rep], ["kv_repeated"], group="repeat_kv")
    kt = g.node("Transpose", [kr], ["k_t"], group="qk_scaled", perm=[0, 1, 3, 2])
    logits = g.node("MatMul", [q, kt], ["qk_logits"], group="qk_scaled")
    dh = int(ctx.p("Dh"))
    logits = g.node("Mul", [logits, g.const("scale", np.float32(1.0 / np.sqrt(dh)))],
                    ["qk_scaled"], group="qk_scaled")
    masked = g.node("Add", [logits, mask], ["qk_masked"], group="mask_add")
    sink3 = g.node("Reshape", [sinks, g.const("sink_shape", np.array([1, nh, 1, 1], np.int64))],
                   ["sink3"], group="sink_softmax")
    bdim = g.node("Gather", [g.node("Shape", [q], ["q_shape"], group="sink_softmax"),
                             g.const("b_ax", np.array(0, np.int64))], ["bdim"], group="sink_softmax")
    tl = g.node("Gather", [g.node("Shape", [q], ["q_shape2"], group="sink_softmax"),
                           g.const("t_ax", np.array(2, np.int64))], ["tlen"], group="sink_softmax")
    reps = g.node("Concat", [g.node("Unsqueeze", [bdim], ["bdim_u"], group="sink_softmax", axes=[0]),
                             g.const("mid_reps", np.array([1], np.int64)),
                             g.node("Unsqueeze", [tl], ["tl_u"], group="sink_softmax", axes=[0]),
                             g.const("last_rep", np.array([1], np.int64))], ["sink_reps"],
                  group="sink_softmax", axis=0)
    sk = g.node("Tile", [sink3, reps], ["sink_full"], group="sink_softmax")
    cat = g.node("Concat", [masked, sk], ["with_sink"], group="sink_softmax", axis=-1)
    prob = g.node("Softmax", [cat], ["prob_sink"], group="sink_softmax", axis=-1)
    L = g.node("Gather", [g.node("Shape", [prob], ["p_shape"], group="sink_softmax"),
                          g.const("last_ax", np.array(3, np.int64))], ["plen"], group="sink_softmax")
    end = g.node("Sub", [L, g.const("one_l", np.array(1, np.int64))], ["end"], group="sink_softmax")
    e1 = g.node("Unsqueeze", [end], ["end_u"], group="sink_softmax", axes=[0])
    probs = g.node("Slice", [prob, g.const("zero_v", np.array([0], np.int64)), e1,
                             g.const("ax3", np.array([3], np.int64))], ["attn_probs"],
                   group="sink_softmax")
    out = g.node("MatMul", [probs, kr], ["attn_head_raw"], group="av_matmul")
    out = g.node("Transpose", [out], ["attn_head_out"], group="av_matmul", perm=[0, 2, 1, 3])
    ctx.out_t(g, "attn_probs", name="attn_probs")
    ctx.out_t(g, "attn_head_out", name="attn_head_out")
    return g


def build_hash_route_prod(ctx) -> GB:
    """生产 Hash 路由：ffn_normed→flat；input_ids+tid2eid→indices；scores gather。"""
    g = GB("moe_route", ctx.pid)
    normed = ctx.act_in(g, "ffn_normed")
    flat = g.node("Reshape", [normed, g.const("flat_shape", np.array([-1, int(ctx.p("H"))], np.int64))],
                  ["ffn_flat"], group="moe_flatten")
    ids = g.inp("input_ids", ctx.sh("input_ids"), INT64, "activation")
    table = g.inp("tid2eid", ctx.sh("tid2eid"), INT64, "persistent_buffer")
    w = ctx.weight(g, "moe_router", "router_weight")
    idx = g.node("Reshape", [ids, g.const("flat_ids", np.array([-1], np.int64))],
                 ["ids_flat"], group="moe_hash_lookup")
    idx = g.node("Gather", [table, idx], ["topk_indices"], group="moe_hash_lookup", axis=0)
    logits = linear(g, "moe_router", flat, w, "router_logits")
    sp = g.node("Softplus", [logits], ["softplus"], group="moe_router")
    scores = g.node("Sqrt", [sp], ["router_scores"], group="moe_router")
    wt = g.node("GatherElements", [scores, idx], ["gathered"], group="moe_router", axis=1)
    sm = g.node("ReduceSum", [wt], ["wsum"], group="moe_router", axes=[1], keepdims=1)
    sm = g.node("Add", [sm, g.const("eps", np.float32(1e-20))], ["wsum_eps"], group="moe_router")
    nw = g.node("Div", [wt, sm], ["normed"], group="moe_router")
    nw = g.node("Mul", [nw, g.const("scl", np.float32(float(ctx.p("Scl"))))], ["topk_weights"],
                group="moe_router")
    e = int(ctx.p("E"))
    onehot = g.node("OneHot", [idx, g.const("depth", np.array(e + 1, np.int64)),
                               g.const("oh_vals", np.array([0.0, 1.0], np.float32))],
                    ["oh"], group="moe_dispatch", axis=-1)
    mask = g.node("Transpose", [onehot], ["expert_mask"], group="moe_dispatch", perm=[2, 1, 0])
    hit = g.node("ReduceSum", [mask], ["hit_sum"], group="moe_dispatch", axes=[1, 2], keepdims=0)
    hit = g.node("Cast", [hit], ["hit_bool"], group="moe_dispatch", to=7)
    hit = g.node("NonZero", [hit], ["hit_nz"], group="moe_dispatch")
    hit = g.node("Transpose", [hit], ["hit_experts"], group="moe_dispatch", perm=[1, 0])
    ctx.out_t(g, "ffn_flat", name="ffn_flat")
    ctx.out_t(g, "topk_indices", name="topk_indices", dtype=INT64)
    ctx.out_t(g, "topk_weights", name="topk_weights")
    ctx.out_t(g, "router_logits", name="router_logits")
    ctx.out_t(g, "router_scores", name="router_scores")
    ctx.out_t(g, "expert_mask", name="expert_mask")
    ctx.out_t(g, "hit_experts", name="hit_experts", dtype=INT64)
    return g


def build_hca_cache(ctx) -> GB:
    """HCA cache：values=keys 同存储（别名），keys 持久 tail W-1。"""
    g = GB("attn_cache", ctx.pid)
    ck = ctx.cache_in(g, "cache_keys")
    kr = ctx.act_in(g, "kv_rope")
    fk = g.node("Concat", [ck, kr], ["kv_full"], group="cache_update", axis=2)
    wkeep = int(ctx.p("W")) - 1
    lk = g.node("Gather", [g.node("Shape", [fk], ["fk_shape"], group="cache_update"),
                           g.const("two_i", np.array(2, np.int64))], ["lk"], group="cache_update")
    start = g.node("Max", [g.node("Sub", [lk, g.const("keep", np.array(wkeep, np.int64))],
                                 ["raw"], group="cache_update"),
                           g.const("zero_i", np.array(0, np.int64))], ["start"], group="cache_update")
    s1 = g.node("Unsqueeze", [start], ["start_u"], group="cache_update", axes=[0])
    kp = g.node("Slice", [fk, s1, g.const("imax", np.array([2 ** 31 - 1], np.int64)),
                          g.const("ax2", np.array([2], np.int64))], ["cache_keys_post"], group="cache_update")
    vp = g.node("Identity", [kp], ["cache_values_post"], group="cache_update")
    cl = g.node("Gather", [g.node("Shape", [kr], ["kr_shape"], group="cache_update"),
                           g.const("two2", np.array(2, np.int64))], ["tlen"], group="cache_update")
    cum = g.node("Add", [ctx.cache_in(g, "cache_cumlen", dtype=INT64), cl],
                 ["cache_cumlen_post"], group="cache_update")
    ctx.out_t(g, "kv_full", name="kv_full")
    ctx.out_t(g, "cache_keys_post", name="cache_keys_post")
    ctx.out_t(g, "cache_values_post", name="cache_values_post", role="cache_post_alias")
    ctx.out_t(g, "cache_cumlen_post", name="cache_cumlen_post", dtype=INT64)
    return g


def build_swa_mask(ctx) -> GB:
    from structure_reader.onnx_module.builders import _build_mask_model
    g = GB("attn_mask", ctx.pid)
    _build_mask_model(ctx, g)
    ctx.out_t(g, "mask_model", name="mask_model")
    return g


def build_hca_mask(ctx) -> GB:
    from structure_reader.onnx_module.builders import _build_mask_model
    g = GB("attn_mask", ctx.pid)
    _build_mask_model(ctx, g)
    b = ctx.act_in(g, "block_bias")
    g.node("Concat", ["mask_model", b], ["mask_ext"], group="c_mask_extend", axis=-1)
    ctx.out_t(g, "mask_model", name="mask_model")
    ctx.out_t(g, "mask_ext", name="mask_ext")
    return g
