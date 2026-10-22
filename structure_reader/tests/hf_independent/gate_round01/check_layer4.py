"""Independent layer4 (CSA + TopK MoE + two mHC) facts checker for HF M1 gate round01.

Expectations are derived *here* from the pinned HF source text and the root
config.json (read-only). The reading document is the object under test; nothing
is copied from the fresh builder, product validators or coordination probes.

Roles are resolved by structure (module_path + op tokens + graph connectivity).
When a role cannot be resolved, a `manual` issue is emitted instead of guessing.

No torch/transformers import; no model execution.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

TRANSFORMERS_HEAD = "536ecc007387a50e77603bb5d92100e9b07514cc"
CONFIG_REL = "models/deepseek-ai/DeepSeek-V4-Flash-0731/config.json"
MODELING_REL = "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py"
CONFIGURATION_REL = "hflib/transformers/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py"

EXPECTED_PARAMS = {
    "H": 4096, "Nh": 64, "Nkv": 1, "Dh": 512, "Rq": 1024, "Rk": 64,
    "G": 8, "Ro": 1024, "M": 4, "Ki": 512, "Ih": 64, "Id": 128, "W": 128,
    "Nhc": 4, "Sit": 20, "E": 256, "K": 6, "Es": 2048, "Lim": 10.0, "Scl": 1.5,
    "ThetaC": 160000, "Yf": 16,
}
EXPECTED_FLOAT_PARAMS = {"Eps": 1e-6, "EpsN": 1e-6}


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    phase: str | None = None
    subject: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _err(code, msg, phase=None, subject=None):
    return Issue("error", code, msg, phase, subject)


def _manual(code, msg, phase=None, subject=None):
    return Issue("manual", code, msg, phase, subject)


def _info(code, msg, phase=None, subject=None):
    return Issue("info", code, msg, phase, subject)


def _warn(code, msg, phase=None, subject=None):
    return Issue("warning", code, msg, phase, subject)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_expected(workspace_root: Path) -> dict:
    raw = json.loads((workspace_root / CONFIG_REL).read_text(encoding="utf-8"))
    n = raw["num_hidden_layers"]
    mapping = {0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}
    layer_types = [mapping[r] for r in raw["compress_ratios"]][:n]
    n_hash = raw["num_hash_layers"]
    mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash)
    return {
        "raw": raw,
        "layer_index": 4,
        "attention_type": layer_types[4],
        "moe_type": mlp_layer_types[4],
        "layer_types": layer_types,
        "mlp_layer_types": mlp_layer_types,
        "persistent_cap": raw["sliding_window"] - 1,
        "nope_dim": raw["head_dim"] - raw["qk_rope_head_dim"],
    }


class Phase:
    def __init__(self, phase):
        self.d = phase
        self.id = phase["id"]
        self.tensors = {t["id"]: t for t in phase["tensors"]}
        self.ops = phase["operators"]
        self.producers = {}
        self.consumers = {}
        self.op_by_id = {o["id"]: o for o in self.ops}
        for o in self.ops:
            for tid in o["outputs"]:
                self.producers.setdefault(tid, []).append(o)
            for tid in o["inputs"]:
                self.consumers.setdefault(tid, []).append(o)

    def shape(self, tid):
        return self.tensors.get(tid, {}).get("shape", [])

    def kind(self, tid):
        return self.tensors.get(tid, {}).get("kind")

    def producer(self, tid):
        p = self.producers.get(tid)
        return p[0] if p else None

    def module_ops(self, token):
        return [o for o in self.ops if token in (o.get("module_path") or "")]

    def find(self, *tokens, all_of=False, exclude=()):
        out = []
        for o in self.ops:
            op = o.get("op") or ""
            hit = all(t in op for t in tokens) if all_of else any(t in op for t in tokens)
            if hit and not any(x in op for x in exclude):
                out.append(o)
        return out


def _one(cands, code, phase, what, issues):
    if len(cands) == 1:
        return cands[0]
    issues.append(_manual(code, f"cannot uniquely resolve {what} (candidates={[c['id'] for c in cands]})", phase.id))
    return None


def _need(op, code, phase, what, issues):
    if op is None:
        issues.append(_manual(code, f"could not resolve {what}", phase.id))
    return op


# ---------------------------------------------------------------------------
# identity / scope / parameters
# ---------------------------------------------------------------------------

def check_identity(document, workspace_root, issues):
    model = document.get("model", {})
    if model.get("model_key") != "deepseek_v4_flash_0731_hf":
        issues.append(_err("model_key", f"unexpected model_key {model.get('model_key')!r}"))
    if model.get("implementation") != "transformers":
        issues.append(_err("implementation", f"implementation must be transformers, got {model.get('implementation')!r}"))
    if model.get("source_revision") != TRANSFORMERS_HEAD:
        issues.append(_err("source_revision", f"source_revision {model.get('source_revision')!r} != {TRANSFORMERS_HEAD}"))
    for rel in (CONFIG_REL, MODELING_REL, CONFIGURATION_REL):
        if not (workspace_root / rel).is_file():
            issues.append(_err("input_missing", f"input not found: {rel}"))


def check_scope(document, workspace_root, issues):
    scope = document.get("scope", {})
    if scope.get("layer_index") != 4:
        issues.append(_err("scope_layer", f"layer_index must be 4, got {scope.get('layer_index')!r}"))
    if scope.get("attention_type") != "compressed_sparse_attention":
        issues.append(_err("scope_attn", f"attention_type {scope.get('attention_type')!r} != compressed_sparse_attention"))
    if scope.get("moe_type") != "moe":
        issues.append(_err("scope_moe", f"moe_type {scope.get('moe_type')!r} != moe"))
    if scope.get("analysis_path") != "eager":
        issues.append(_err("scope_path", f"analysis_path must be eager, got {scope.get('analysis_path')!r}"))
    if [p.get("id") for p in document.get("phases", [])] != ["prefill", "decode"]:
        issues.append(_err("phase_order", f"phases must be [prefill, decode], got {[p.get('id') for p in document.get('phases', [])]}"))


def check_parameters(phase: Phase, issues):
    params = {p["id"]: p for p in phase.d["parameters"]}
    for pid, val in EXPECTED_PARAMS.items():
        if pid not in params:
            issues.append(_err("param_missing", f"parameter {pid!r} missing", phase.id))
        elif params[pid].get("value") != val:
            issues.append(_err("param_value", f"parameter {pid!r} value {params[pid].get('value')!r} != {val!r}", phase.id, pid))
    for pid, val in EXPECTED_FLOAT_PARAMS.items():
        if pid in params and params[pid].get("value") is not None:
            try:
                if abs(float(params[pid]["value"]) - val) > 1e-12:
                    issues.append(_err("param_value", f"parameter {pid!r} value {params[pid].get('value')!r} != {val!r}", phase.id, pid))
            except (TypeError, ValueError):
                issues.append(_err("param_value", f"parameter {pid!r} value not numeric", phase.id, pid))
    for pid in ("H", "Nh", "Dh", "Rq", "Es", "E", "ThetaC"):
        if pid in params:
            kinds = {s.get("kind") for s in params[pid].get("source_refs", [])}
            if not kinds & {"config", "derived"}:
                issues.append(_err("param_source", f"parameter {pid!r} lacks config/derived source_ref (got {sorted(kinds)})", phase.id, pid))


# ---------------------------------------------------------------------------
# mHC
# ---------------------------------------------------------------------------

def check_mhc_site(phase: Phase, issues, site):
    ops = phase.module_ops(f"{site}_hc")
    if not ops:
        issues.append(_manual("mhc_resolve", f"no ops with module_path containing {site}_hc", phase.id))
        return
    flatten = _one([o for o in ops if "rmsnorm" in o["op"]], "mhc_resolve", phase, f"{site} flatten+rmsnorm op", issues)
    project = _one([o for o in ops if o["op"] == "linear"], "mhc_resolve", phase, f"{site} weight project op", issues)
    split = _one([o for o in ops if "split" in o["op"]], "mhc_resolve", phase, f"{site} split op", issues)
    pre = _one([o for o in ops if "sigmoid" in o["op"] and "add" in o["op"]], "mhc_resolve", phase, f"{site} pre op", issues)
    post = _one([o for o in ops if "sigmoid" in o["op"] and "mul" in o["op"]], "mhc_resolve", phase, f"{site} post op", issues)
    comb = _one([o for o in ops if "sinkhorn" in o["op"]], "mhc_resolve", phase, f"{site} comb/sinkhorn op", issues)
    collapse = _one([o for o in ops if "sum" in o["op"]], "mhc_resolve", phase, f"{site} collapse op", issues)
    for op, what in ((flatten, "flatten"), (project, "project"), (split, "split"), (pre, "pre"), (post, "post"), (comb, "comb"), (collapse, "collapse")):
        _need(op, "mhc_resolve", phase, f"{site} {what} op", issues)
    # split raw weights must feed pre/post/comb
    if split and pre and post and comb:
        raws = set(split["outputs"])
        if len(raws) != 3:
            issues.append(_err("mhc_split", f"{site} split {split['id']!r} must emit 3 raw weights, got {len(raws)}", phase.id, split["id"]))
        for op, label in ((pre, "pre"), (post, "post"), (comb, "comb")):
            if not (set(op["inputs"]) & raws):
                issues.append(_err("mhc_split_link", f"{site} {label} op {op['id']!r} does not consume a raw split weight", phase.id, op["id"]))
    # pre must collapse (consume) the stream input: collapse consumes pre output + streams
    if collapse and pre:
        if pre["outputs"][0] not in collapse["inputs"]:
            issues.append(_err("mhc_collapse_link", f"{site} collapse {collapse['id']!r} does not consume pre {pre['outputs'][0]!r}", phase.id, collapse["id"]))
    # weight shapes
    proj_w = f"{site}_hc_fn"
    base_w = f"{site}_hc_base"
    scale_w = f"{site}_hc_scale"
    for tid, want in ((proj_w, ["(2+Nhc)×Nhc", "Nhc×H"]), (base_w, ["(2+Nhc)×Nhc"]), (scale_w, ["3"])):
        if tid not in phase.tensors:
            issues.append(_err("mhc_weight_missing", f"{tid} missing", phase.id, tid))
        elif phase.shape(tid) != want:
            issues.append(_err("mhc_weight_shape", f"{tid} shape {phase.shape(tid)} != {want}", phase.id, tid))
    # comb.T chain
    if comb:
        comb_t = comb["outputs"][0]
        transpose = [o for o in phase.ops if o["op"] == "transpose" and comb_t in o["inputs"]]
        transpose = _one(transpose, "combT_resolve", phase, f"{site} comb transpose op", issues)
        if transpose:
            dst = transpose["outputs"][0]
            sshape, dshape = phase.shape(comb_t), phase.shape(dst)
            expected = list(sshape)
            if len(expected) >= 2:
                expected[-2], expected[-1] = expected[-1], expected[-2]
            if dshape != expected:
                issues.append(_err("combT_rank", f"{site}: transpose {comb_t}{sshape} -> {dst}{dshape}, expected {expected}", phase.id, transpose["id"]))
            if not phase.consumers.get(dst):
                issues.append(_err("combT_isolated", f"{site}: transpose output {dst!r} never consumed", phase.id, transpose["id"]))
            mix = [o for o in phase.ops if "matmul" in o["op"] and "add" in o["op"] and dst in o["inputs"]]
            if not mix:
                issues.append(_err("combT_mix_link", f"{site}: no residual mix consumes transposed comb {dst!r}", phase.id, transpose["id"]))
            else:
                issues.append(_info("combT", f"{site}: comb.T consumed by residual mix {mix[0]['id']}", phase.id, transpose["id"]))


# ---------------------------------------------------------------------------
# Q path
# ---------------------------------------------------------------------------

def check_q_path(phase: Phase, issues):
    q_b = _one([o for o in phase.ops if "linear" in o["op"] and "transpose" in o["op"] and "matmul" not in o["op"]], "q_resolve", phase, "Q up-projection op", issues)
    if not q_b:
        return
    if len(q_b["inputs"]) != 1:
        issues.append(_manual("q_resolve", f"Q up-projection {q_b['id']!r} has {len(q_b['inputs'])} inputs", phase.id, q_b["id"]))
        return
    q_norm = phase.producer(q_b["inputs"][0])
    if q_norm is None or "rmsnorm" not in (q_norm.get("op") or ""):
        issues.append(_err("q_norm_chain", f"Q up-projection input {q_b['inputs'][0]!r} is not produced by an RMSNorm op", phase.id, q_b["id"]))
        return
    # weighted RMSNorm: weight [Rq] either in weights or an input of kind weight
    w = q_norm.get("weights", []) + [i for i in q_norm["inputs"] if phase.kind(i) == "weight"]
    if not any(phase.shape(t) == ["Rq"] for t in w):
        issues.append(_err("q_norm_weight", f"Q RMSNorm {q_norm['id']!r} lacks a learned weight [Rq] (source: DeepseekV4RMSNorm(q_lora_rank))", phase.id, q_norm["id"]))
    # down projection feeds the norm
    act_in = [i for i in q_norm["inputs"] if phase.kind(i) != "weight"]
    q_a = phase.producer(act_in[0]) if act_in else None
    if q_a is None or q_a.get("op") != "linear":
        issues.append(_err("q_down_chain", f"Q RMSNorm input is not produced by a linear down-projection", phase.id, q_norm["id"]))
    else:
        if not any(phase.shape(x) == ["Rq", "H"] for x in q_a.get("weights", [])):
            issues.append(_err("q_down_weight", f"Q down-projection {q_a['id']!r} lacks weight [Rq, H]", phase.id, q_a["id"]))
    if not any(phase.shape(x) == ["Nh×Dh", "Rq"] for x in q_b.get("weights", [])):
        issues.append(_err("q_up_weight", f"Q up-projection {q_b['id']!r} lacks weight [Nh×Dh, Rq]", phase.id, q_b["id"]))
    q_b_norm = _one([o for o in phase.ops if "unweighted" in o["op"]], "q_resolve", phase, "Q head unweighted-norm op", issues)
    if q_b_norm and q_b_norm.get("weights"):
        issues.append(_err("q_b_norm_weight", f"Q head norm {q_b_norm['id']!r} must be unweighted", phase.id, q_b_norm["id"]))


# ---------------------------------------------------------------------------
# KV / attention
# ---------------------------------------------------------------------------

def check_kv_attention(phase: Phase, issues):
    for o in phase.ops:
        if o["id"] in ("v_proj", "v_b_proj", "value_proj") or o["id"].startswith("v_"):
            issues.append(_err("shared_kv", f"found a separate value projection op {o['id']!r}; V4 shares K=V"))
    repeat = _one(phase.find("expand"), "attn_resolve", phase, "repeat_kv op", issues)
    av = _one([o for o in phase.ops if "matmul" in o["op"] and "transpose" in o["op"]], "attn_resolve", phase, "AV op", issues)
    softmax = _one([o for o in phase.ops if "softmax" in o["op"] and "cat" in o["op"]], "attn_resolve", phase, "attention softmax op", issues)
    inv = _one(phase.find("conjugate"), "attn_resolve", phase, "inverse-RoPE op", issues)
    if repeat and av:
        kv_rep = repeat["outputs"][0]
        if kv_rep not in av["inputs"]:
            issues.append(_err("shared_kv_value", f"AV {av['id']!r} does not consume the shared KV {kv_rep!r}", phase.id, av["id"]))
        else:
            issues.append(_info("shared_kv", f"AV consumes shared KV {kv_rep!r} (K=V)", phase.id, av["id"]))
    if softmax and av:
        probs = softmax["outputs"][0]
        if probs not in av["inputs"]:
            issues.append(_err("av_inputs", f"AV {av['id']!r} inputs {av['inputs']} do not include softmax output {probs!r}", phase.id, av["id"]))
    if inv:
        if not any("cos" in i for i in inv["inputs"]) or not any("sin" in i for i in inv["inputs"]):
            issues.append(_err("inv_rope_inputs", f"inverse-RoPE {inv['id']!r} must consume cos/sin", phase.id, inv["id"]))
    # cache read view vs persistent
    if repeat:
        kv_all = repeat["inputs"][0]
        cat = phase.producer(kv_all)
        if cat is None:
            issues.append(_err("kv_chain", f"repeat input {kv_all!r} has no producer", phase.id, repeat["id"]))
        else:
            cache_reads = [i for i in cat["inputs"] if phase.kind(i) == "cache"]
            if not cache_reads:
                issues.append(_err("kv_chain", f"KV concat {cat['id']!r} has no cache read input", phase.id, cat["id"]))
            for cr in cache_reads:
                sh = phase.shape(cr)
                if sh and sh[-2] == "L_kv":
                    issues.append(_info("kv_read_view", f"read KV view {cr}={sh}", phase.id, cr))
    # sliding KV update resolved through the attention chain: repeat_kv -> kv_all -> concat -> cache read -> update
    if repeat:
        kv_all = repeat["inputs"][0]
        cat = phase.producer(kv_all)
        cache_reads = [i for i in cat["inputs"] if phase.kind(i) == "cache"] if cat else []
        upd = phase.producer(cache_reads[0]) if cache_reads else None
        if upd is None:
            issues.append(_manual("kv_resolve", "could not resolve the sliding KV cache update op via the attention chain", phase.id))
        else:
            post = upd["outputs"][1] if len(upd["outputs"]) > 1 else None
            if post and phase.kind(post) != "cache":
                issues.append(_err("kv_persist_kind", f"persistent KV output {post!r} should be a cache tensor", phase.id, upd["id"]))
            expr = phase.shape(post)[-2] if post and phase.shape(post) else None
            if expr is not None:
                toks = [t for t in re.split(r"[^A-Za-z0-9_]+", expr) if t]
                if "min(" not in expr or "W-1" not in expr or "T" not in toks:
                    issues.append(_err("kv_persist_len", f"persistent KV {post!r} length {expr!r} should be min(prior+T, W-1) (must depend on the new T and cap W-1)", phase.id, post))
                else:
                    issues.append(_info("kv_persist_len", f"persistent KV {post}={phase.shape(post)}", phase.id, post))
    if not any("flatten" in o["op"] for o in phase.ops):
        issues.append(_manual("grouped_out", "could not resolve grouped output projection op", phase.id))


def check_prob_axis(phase: Phase, issues):
    av = _one([o for o in phase.ops if "matmul" in o["op"] and "transpose" in o["op"]], "prob_axis_resolve", phase, "AV op", issues)
    repeat = _one(phase.find("expand"), "prob_axis_resolve", phase, "repeat_kv op", issues)
    softmax = _one([o for o in phase.ops if "softmax" in o["op"] and "cat" in o["op"]], "prob_axis_resolve", phase, "softmax op", issues)
    if not (av and repeat and softmax):
        return
    probs = softmax["outputs"][0]
    kv_rep = repeat["outputs"][0]
    pshape, kshape = phase.shape(probs), phase.shape(kv_rep)
    if pshape and kshape:
        key_axis = pshape[-1]
        kv_seq_axis = kshape[-2] if len(kshape) >= 3 else kshape[-1]
        if key_axis != kv_seq_axis:
            issues.append(_err("prob_axis", f"softmax {probs!r} key axis {key_axis!r} != KV seq axis {kv_seq_axis!r}", phase.id, probs))


# ---------------------------------------------------------------------------
# CSA / Indexer cache
# ---------------------------------------------------------------------------

def check_csa_indexer(phase: Phase, issues):
    windows = [o for o in phase.ops if o["op"] == "view+assign"]
    if len(windows) != 2:
        issues.append(_manual("window_resolve", f"expected 2 window-view ops (CSA + Indexer), got {len(windows)}", phase.id))
    for o in windows:
        label = "indexer" if "Indexer" in (o.get("module_path") or "") else "csa"
        overlap_in = [i for i in o["inputs"] if phase.kind(i) == "cache"]
        overlap_out = [i for i in o["outputs"] if phase.kind(i) == "cache"]
        for t in overlap_in:
            sh = phase.shape(t)
            if len(sh) != 3 or sh[1] != "M":
                issues.append(_err("overlap_shape", f"{label} overlap input {t}={sh} != [B, M, dim]", phase.id, t))
        if not overlap_in:
            issues.append(_err("overlap_missing", f"{label} window view {o['id']!r} has no overlap cache input", phase.id, o["id"]))
        if not overlap_out:
            issues.append(_err("overlap_missing", f"{label} window view {o['id']!r} has no overlap cache output", phase.id, o["id"]))
    # Indexer query consumes normalized Q residual
    i_q = _one([o for o in phase.ops if "linear" in o["op"] and "rope" in o["op"] and "transpose" not in o["op"]], "indexer_query_resolve", phase, "indexer query projection op", issues)
    q_norms = [o for o in phase.ops if o["op"] == "rmsnorm" and (o.get("module_path") or "").endswith("q_a_norm")]
    if i_q:
        if not q_norms:
            issues.append(_manual("indexer_query_norm", "could not resolve q_a_norm to compare", phase.id, i_q["id"]))
        else:
            qn_out = q_norms[0]["outputs"][0]
            if qn_out not in i_q["inputs"]:
                issues.append(_err("indexer_query_norm", f"indexer query {i_q['id']!r} inputs {i_q['inputs']} lack normalized Q residual {qn_out!r}", phase.id, i_q["id"]))
            else:
                issues.append(_info("indexer_query_norm", f"indexer query consumes normalized Q residual {qn_out!r} + RoPE cos/sin", phase.id, i_q["id"]))
        w = i_q.get("weights", []) + [i for i in i_q["inputs"] if phase.kind(i) == "weight"]
        if not any(phase.shape(t) == ["Ih×Id", "Rq"] for t in w):
            issues.append(_err("indexer_weight", f"indexer query {i_q['id']!r} lacks weight [Ih×Id, Rq]", phase.id, i_q["id"]))
    # scorer consumes post-update indexer entries
    scorer = _one([o for o in phase.ops if "relu" in o["op"]], "scorer_resolve", phase, "indexer scorer op", issues)
    if scorer:
        cache_inputs = [i for i in scorer["inputs"] if phase.kind(i) == "cache"]
        if not cache_inputs:
            issues.append(_err("scorer_input", f"indexer scorer {scorer['id']!r} consumes no compressed-entry cache input", phase.id, scorer["id"]))
        else:
            for t in cache_inputs:
                sh = phase.shape(t)
                if sh and ("+" in sh[-2]):
                    issues.append(_info("scorer_new_len", f"scorer consumes {t}={sh} (includes new window)", phase.id, t))
    # topk min length
    topk = _one([o for o in phase.ops if "topk" in o["op"] and "where" in o["op"]], "topk_resolve", phase, "indexer top-k op", issues)
    if topk:
        sh = phase.shape(topk["outputs"][0])
        if sh and "min(" not in sh[-1]:
            issues.append(_err("indexer_topk_min", f"topk output {topk['outputs'][0]!r} last axis {sh[-1]!r} should be min(Ki, entries)", phase.id, topk["outputs"][0]))
    # block_bias after topk
    bias = _one([o for o in phase.ops if "scatter" in o["op"]], "block_bias_resolve", phase, "block_bias op", issues)
    if bias and topk:
        idx = {o["id"]: i for i, o in enumerate(phase.ops)}
        if idx[bias["id"]] < idx[topk["id"]]:
            issues.append(_err("block_bias_order", f"block_bias {bias['id']!r} precedes top-k {topk['id']!r}", phase.id, bias["id"]))
    # entry_count and window params present
    for pid in ("Nwin", "Ncomp", "Tbuf", "Tbufp"):
        if pid not in {p["id"] for p in phase.d["parameters"]}:
            issues.append(_err("cache_param_missing", f"CSA cache derived parameter {pid!r} missing", phase.id))


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

def check_moe(phase: Phase, issues):
    router = _one(phase.find("softplus"), "moe_resolve", phase, "router op", issues)
    if router:
        w = router.get("weights", []) + [i for i in router["inputs"] if phase.kind(i) == "weight"]
        if not any(phase.shape(t) == ["E", "H"] for t in w):
            issues.append(_err("router_weight", f"router {router['id']!r} lacks weight [E, H]", phase.id, router["id"]))
        if len(router["outputs"]) < 3:
            issues.append(_err("router_outputs", f"router {router['id']!r} must emit logits/indices/weights", phase.id, router["id"]))
        if "Scl" not in router.get("parameter_refs", []):
            issues.append(_err("router_scale", f"router {router['id']!r} parameter_refs {router.get('parameter_refs')} missing Scl (routed scaling)", phase.id, router["id"]))
    select = _one([o for o in phase.ops if "where" in o["op"] and "index" in o["op"]], "moe_resolve", phase, "token selection op", issues)
    slice_op = _one([o for o in phase.ops if o["op"] == "index"], "moe_resolve", phase, "expert slice op", issues)
    gu = _one([o for o in phase.ops if o["op"] == "linear" and phase.shape(o["outputs"][0])[-1:] == ["2Es"]], "moe_resolve", phase, "expert gate_up linear op", issues)
    down = _one([o for o in phase.ops if o["op"] == "linear" and phase.shape(o["outputs"][0]) == ["N_e", "H"]], "moe_resolve", phase, "expert down linear op", issues)
    clamp = _one([o for o in phase.ops if "chunk" in o["op"]], "moe_resolve", phase, "expert clamp op", issues)
    act = _one([o for o in phase.ops if o["op"] == "silu+mul"], "moe_resolve", phase, "expert activation op", issues)
    weight_op = _one([o for o in phase.ops if o["op"] == "gather+mul"], "moe_resolve", phase, "expert weight-apply op", issues)
    combine = _one([o for o in phase.ops if "index_add" in o["op"]], "moe_resolve", phase, "expert combine op", issues)
    if slice_op:
        got = sorted(tuple(phase.shape(t)) for t in slice_op["outputs"])
        expect = sorted([("2Es", "H"), ("H", "Es")])
        if got != expect:
            issues.append(_err("expert_slice_shape", f"expert slice outputs {got} != {expect} (gate_up[e]->[2Es,H], down[e]->[H,Es])", phase.id, slice_op["id"]))
        # slice must select using hit experts
        if not any(phase.kind(i) == "index" for i in slice_op["inputs"]):
            issues.append(_err("expert_slice_index", f"expert slice {slice_op['id']!r} has no hit-experts index input", phase.id, slice_op["id"]))
    # one expert template only
    gu_like = [o for o in phase.ops if o["op"] == "linear" and phase.shape(o["outputs"][0])[-1:] == ["2Es"]]
    down_like = [o for o in phase.ops if o["op"] == "linear" and phase.shape(o["outputs"][0]) == ["N_e", "H"]]
    if len(gu_like) != 1 or len(down_like) != 1:
        issues.append(_err("expert_template_count", f"expected exactly one gate_up/down template op, got gu={len(gu_like)} down={len(down_like)}", phase.id))
    # gu consumes selected routed tokens; down consumes activation
    if gu and select and (set(gu["inputs"]) & set(select["outputs"])):
        issues.append(_info("expert_input_linkage", f"expert gate_up consumes selected routed tokens from {select['id']!r}", phase.id, gu["id"]))
    elif gu:
        issues.append(_err("expert_input_linkage", f"expert gate_up {gu['id']!r} inputs {gu['inputs']} lack selected routed tokens from {select['id']!r}" if select else "expert gate_up lacks routed-token input", phase.id, gu["id"]))
    if gu and act and down:
        if act["outputs"][0] not in down["inputs"]:
            issues.append(_err("expert_gu_down", f"expert down {down['id']!r} does not consume activation {act['outputs'][0]!r}", phase.id, down["id"]))
    # weight op: same addressing as selection, applies top-k weights once
    if weight_op and select:
        outs = set(select["outputs"])
        if not outs & set(weight_op["inputs"]):
            issues.append(_err("expert_addressing", f"weight op {weight_op['id']!r} does not consume selection indices from {select['id']!r}", phase.id, weight_op["id"]))
        if not any("topk_weight" in i for i in weight_op["inputs"]):
            issues.append(_err("expert_weight_apply", f"weight op {weight_op['id']!r} does not consume top-k weights", phase.id, weight_op["id"]))
        # multiply once: exactly one output and it is produced here
        if len(weight_op["outputs"]) != 1:
            issues.append(_err("expert_weight_once", f"weight op {weight_op['id']!r} should have exactly one output", phase.id, weight_op["id"]))
        # raw down output consumed only here (weights applied once)
        raw = [i for i in weight_op["inputs"] if "down_raw" in i]
        if raw and len(phase.consumers.get(raw[0], [])) != 1:
            issues.append(_err("expert_weight_once", f"expert down raw {raw[0]!r} consumed by {len(phase.consumers.get(raw[0], []))} ops; weights must be applied once", phase.id, weight_op["id"]))
    if combine and weight_op:
        if weight_op["outputs"][0] not in combine["inputs"]:
            issues.append(_err("expert_combine", f"combine {combine['id']!r} does not consume weighted expert output {weight_op['outputs'][0]!r}", phase.id, combine["id"]))
    # shared expert + merge
    shared_ops = phase.module_ops("DeepseekV4MLP")
    if len(shared_ops) < 2:
        issues.append(_manual("shared_expert", f"could not resolve both shared-expert ops (module DeepseekV4MLP): {[o['id'] for o in shared_ops]}", phase.id))
    merge = _one([o for o in phase.ops if o["op"] == "add" and "SparseMoeBlock" in (o.get("module_path") or "")], "moe_resolve", phase, "shared/routed merge op", issues)
    if merge and len(merge["inputs"]) != 2:
        issues.append(_err("moe_merge", f"merge {merge['id']!r} should add exactly routed + shared", phase.id, merge["id"]))


# ---------------------------------------------------------------------------
# Redundant declaration adjudication
# ---------------------------------------------------------------------------

def check_declarations(phase: Phase, issues):
    referenced, consumed = set(), set()
    for o in phase.ops:
        referenced.update(o["inputs"]); referenced.update(o["outputs"]); referenced.update(o["weights"])
        consumed.update(o["inputs"])
    input_shapes = {tuple(phase.shape(i)) for i in (phase.ops[0]["inputs"] if phase.ops else [])}
    passthrough = {t["id"] for t in phase.tensors.values() if t["kind"] == "activation" and tuple(t["shape"]) in input_shapes}
    subsumed = {
        "mhc_flat": "view+float+rmsnorm",
        "combined_logits": "softmax",
        "probs": "softmax",
        "attn_mixed": "transpose",
        "ffn_mixed": "transpose",
        "router_scores": "softplus",
        "shared_act": "silu",
    }
    byproduct = {"router_logits"}
    for t in phase.tensors.values():
        tid, kind = t["id"], t["kind"]
        if kind != "activation":
            continue
        if tid == "input_ids" or (tid in referenced and (tid in consumed or tid in passthrough)):
            continue
        suffix = next((k for k in subsumed if tid == k or tid.endswith(k)), None)
        if suffix and any(subsumed[suffix] in (o.get("op") or "") for o in phase.ops):
            issues.append(_info("redundant_declaration", f"{tid!r} is a declared intermediate subsumed by a composite op (token {subsumed[suffix]!r})", phase.id, tid))
            continue
        if tid in byproduct:
            issues.append(_info("byproduct", f"{tid!r} is a source byproduct (router logits returned and ignored by the MoE block)", phase.id, tid))
            continue
        issues.append(_err("unresolved_declaration", f"activation {tid!r} declared but unused and not an adjudicated redundant intermediate", phase.id, tid))


def check_phase(phase: Phase, issues):
    check_parameters(phase, issues)
    check_mhc_site(phase, issues, "attn")
    check_mhc_site(phase, issues, "ffn")
    check_q_path(phase, issues)
    check_kv_attention(phase, issues)
    check_prob_axis(phase, issues)
    check_csa_indexer(phase, issues)
    check_moe(phase, issues)
    check_declarations(phase, issues)


def check_layer4(document, workspace_root: Path):
    workspace_root = Path(workspace_root)
    issues: list[Issue] = []
    check_identity(document, workspace_root, issues)
    check_scope(document, workspace_root, issues)
    for p in document.get("phases", []):
        check_phase(Phase(p), issues)
    return issues


__all__ = ["Issue", "check_layer4", "load_expected", "Phase", "sha256_file", "_err", "_warn", "_info", "_manual"]
