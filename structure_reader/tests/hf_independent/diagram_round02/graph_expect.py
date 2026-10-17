"""Independent directed-edge expectation from the reading_document (round02).

Returns, per op set, the exact directed edges (src_node, dst_node, tid, kind)
that a faithful producer->consumer rendering must contain, using node ids
"op:<id>" / "t:<tid>". Derived only from JSON inputs/outputs/weights.
"""

from __future__ import annotations

from collections import defaultdict


def _index(phase):
    producers, consumers, wconsumers = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, op in enumerate(phase["operators"]):
        for t in op.get("outputs", []):
            producers[t].append(i)
        for t in op.get("inputs", []):
            consumers[t].append(i)
        for t in op.get("weights", []):
            wconsumers[t].append(i)
    return producers, consumers, wconsumers


def kind_of(phase, tid):
    k = {t["id"]: t.get("kind") for t in phase["tensors"]}.get(tid, "activation")
    return "weight" if k == "weight" else ("cache" if k == "cache" else "flow")


def expected_edges(phase, op_ids):
    producers, consumers, wconsumers = _index(phase)
    ops = {o["id"]: o for o in phase["operators"]}
    S = set(op_ids)
    out = set()
    for oid in S:
        op = ops[oid]
        for t in op.get("inputs", []):
            if not any(phase["operators"][p]["id"] in S for p in producers.get(t, [])):
                out.add((f"t:{t}", f"op:{oid}", t, kind_of(phase, t)))
        for t in op.get("weights", []):
            if not any(phase["operators"][p]["id"] in S for p in producers.get(t, [])):
                out.add((f"t:{t}", f"op:{oid}", t, "weight"))
        for t in op.get("outputs", []):
            downstream = {phase["operators"][c]["id"]
                          for c in consumers.get(t, []) + wconsumers.get(t, [])
                          if phase["operators"][c]["id"] in S and phase["operators"][c]["id"] != oid}
            if downstream:
                for cid in downstream:
                    out.add((f"op:{oid}", f"op:{cid}", t, kind_of(phase, t)))
            else:
                out.add((f"op:{oid}", f"t:{t}", t, kind_of(phase, t)))
    return out


def expected_cache_downstream(phase, family_tids, inner_op_ids):
    _producers, consumers, _w = _index(phase)
    out = set()
    for t in family_tids:
        for c in consumers.get(t, []):
            cid = phase["operators"][c]["id"]
            if cid not in inner_op_ids:
                out.add((f"t:{t}", f"d:{cid}", t, "flow"))
    return out
