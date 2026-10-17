"""Independent edge/role expectation derived from the reading_document only.

Implements the documented display contract (edges strictly from
operators.inputs/outputs/weights; external ports for unproduced/boundary/cache
inputs; output ports for unconsumed outputs; cache roles from real
producer/consumer references). Used to verify the rendered figures; the
expected values are computed here, not copied from the product renderer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations


def index(phase: dict):
    tensors = {t["id"]: t for t in phase["tensors"]}
    producers, consumers, wconsumers = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, op in enumerate(phase["operators"]):
        for tid in op.get("outputs", []):
            producers[tid].append(i)
        for tid in op.get("inputs", []):
            consumers[tid].append(i)
        for tid in op.get("weights", []):
            wconsumers[tid].append(i)
    return tensors, producers, consumers, wconsumers


def edge_kind(tid, tensors):
    k = tensors.get(tid, {}).get("kind", "activation")
    return "weight" if k == "weight" else ("cache" if k == "cache" else "flow")


def expected_module_edges(phase, op_ids):
    """Counter of (tid, kind) for the module figure of this op set."""
    tensors, producers, consumers, wconsumers = index(phase)
    ops = {o["id"]: o for o in phase["operators"]}
    S = set(op_ids)
    counts = Counter()
    for oid in S:
        op = ops[oid]
        for tid in op.get("inputs", []):
            internal = any(phase["operators"][p]["id"] in S for p in producers.get(tid, []))
            if not internal:
                counts[(tid, edge_kind(tid, tensors))] += 1
        for tid in op.get("weights", []):
            if not any(phase["operators"][p]["id"] in S for p in producers.get(tid, [])):
                counts[(tid, "weight")] += 1
        for tid in op.get("outputs", []):
            downstream = {phase["operators"][c]["id"]
                          for c in consumers.get(tid, []) + wconsumers.get(tid, [])
                          if phase["operators"][c]["id"] in S and phase["operators"][c]["id"] != oid}
            if not downstream:
                counts[(tid, edge_kind(tid, tensors))] += 1
            else:
                # one deduped edge per downstream dst (distinct src/dst/tid)
                for _ in downstream:
                    counts[(tid, edge_kind(tid, tensors))] += 1
    return counts


def transitive_cache_ops(phase, family_prefixes, active_cache_ops):
    """Inner cache op set by shared-tensor transitive closure (independent)."""
    tensors, producers, consumers, wconsumers = index(phase)
    all_cache = {tid for tid, t in tensors.items() if t.get("kind") == "cache"}

    def family_of(tid):
        for fam, prefixes in family_prefixes.items():
            for p in prefixes:
                if tid == p or tid.startswith(p):
                    return fam
        return None

    fam = {tid for tid in all_cache if family_of(tid)}
    cache_ops = {o["id"]: o for o in phase["operators"] if o["id"] in active_cache_ops}
    inner = {oid for oid, op in cache_ops.items()
             if (set(op.get("inputs", [])) | set(op.get("outputs", []))) & fam and family_of(next(iter((set(op.get('inputs',[]))|set(op.get('outputs',[]))) & fam))) }
    # start from ops that directly touch a cache tensor of any family, then closure per family is
    # done by the caller on the figure op set; here we just return family tensor roles.
    return fam, family_of


def cache_roles(phase, family_tids, cache_op_ids):
    """tid -> role string, from real producer/consumer references (independent)."""
    tensors, producers, consumers, wconsumers = index(phase)
    roles = {}
    for tid in family_tids:
        if not producers.get(tid):
            roles[tid] = "调用前状态"
            continue
        outside = [phase["operators"][c]["id"] for c in consumers.get(tid, [])
                   if phase["operators"][c]["id"] not in cache_op_ids]
        if not outside:
            roles[tid] = "调用后回写"
        else:
            roles[tid] = "本次读视图" + ("·兼调用后保存" if tid.endswith("_post") else "")
    return roles


def downstream_edges(phase, read_view_tids, inner_op_ids):
    """(tid, downstream op id) for family-external consumers of read-view cache tensors."""
    tensors, producers, consumers, wconsumers = index(phase)
    out = []
    for tid in read_view_tids:
        for c in consumers.get(tid, []):
            cid = phase["operators"][c]["id"]
            if cid not in inner_op_ids:
                out.append((tid, cid))
    return out
