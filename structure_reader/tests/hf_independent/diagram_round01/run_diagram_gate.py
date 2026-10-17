#!/usr/bin/env python3
"""Independent verification of the M1 decoder HTML connection/cache diagrams.

Checks the frozen candidate's rendered SVG figures against expectations derived
solely from the reading_document, plus static linkage and label geometry.
Diagnostic; exit 1 on any `error`.

Usage: run_diagram_gate.py --candidate <dir> [--json-out <file>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import parse_html as ph  # noqa: E402
import graph_expect as ge  # noqa: E402

CACHE_PREFIXES = {
    "main": ("cache_keys", "cache_cumlen", "kv_full"),
    "csa": ("cache_buffer_c_", "cache_compressed_c", "cache_entry_count_c", "cache_overlap_c_"),
    "indexer": ("cache_buffer_i_", "cache_compressed_i", "cache_entry_count_i", "cache_overlap_i_"),
}
ACTIVE_CACHE_OPS = {
    "cache_update", "c_store_window", "c_window_view", "c_rope_gen", "c_rope", "c_update_states",
    "i_store_window", "i_window_view", "i_pool_rope", "i_update_states", "mask_model_build",
}
CONDITION_OPS = {
    "c_window_view": "仅 N_win>0", "i_window_view": "仅 N_win_i>0",
    "c_update_states": "追加 N_win 条（可为 0）", "i_update_states": "追加 N_win_i 条（可为 0）",
    "c_store_window": "余量回写（每次调用）", "i_store_window": "余量回写（每次调用）",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def char_w(ch: str) -> float:
    return 10.0 if ord(ch) > 0x2E80 else 5.6


def line_w(line: str) -> float:
    return sum(char_w(c) for c in line)


def check_phase(phase: dict, figures: list, issues: list, report: dict):
    tensors = {t["id"]: t for t in phase["tensors"]}
    params = {p["id"] for p in phase["parameters"]}
    ops = {o["id"]: o for o in phase["operators"]}
    # module grouping from rendered module figures (drives overview expectation)
    group_of = {}
    for f in figures:
        if f.kind == "module":
            for oid in f.op_ids():
                group_of[oid] = f.name
    # ---- module + cache figures: edge tensor multiset must equal expectation
    for f in figures:
        if f.kind in ("module", "cache"):
            op_ids = f.op_ids()
            exp = ge.expected_module_edges(phase, op_ids)
            act = Counter(f.edge_hrefs)
            if exp != act:
                missing = exp - act
                extra = act - exp
                issues.append(("error", "edge_mismatch",
                               f"{f.anchor}: expected {sum(exp.values())} edges, got {sum(act.values())}; "
                               f"missing={list(missing.items())[:6]} extra={list(extra.items())[:6]}"))
            report.setdefault("figures", {})[f.anchor] = {"phase": f.phase, "kind": f.kind,
                                                          "op_count": len(op_ids), "expected_edges": sum(exp.values()),
                                                          "actual_edges": sum(act.values())}
    # ---- cache figures: roles, downstream chips, conditions
    for f in figures:
        if f.kind != "cache":
            continue
        fam = f.name
        fam_tids = {t for t, tv in tensors.items()
                    if tv.get("kind") == "cache" and any(t == p or t.startswith(p) for p in CACHE_PREFIXES[fam])}
        inner_op_ids = set(f.op_ids())
        roles = ge.cache_roles(phase, fam_tids, inner_op_ids)
        # state node sub text must carry role
        for n in f.nodes:
            m = re.match(r"#t-[^-]+-(.+)$", n.href)
            if not m or m.group(1) not in fam_tids:
                continue
            tid = m.group(1)
            if roles.get(tid) and roles[tid].split("·")[0] not in n.sub:
                issues.append(("error", "cache_role",
                               f"{f.anchor}: state {tid} sub {n.sub!r} missing role {roles[tid]!r}"))
        # downstream chips must equal family-external consumers of read-view tensors
        read_view = [t for t in fam_tids if roles.get(t, "").startswith("本次读视图")]
        exp_ds = sorted({cid for t in read_view for (_t, cid) in ge.downstream_edges(phase, [t], inner_op_ids)})
        act_ds = sorted(set(f.downstream_ids()))
        if exp_ds != act_ds:
            issues.append(("error", "cache_downstream",
                           f"{f.anchor}: downstream chips {act_ds} != expected external consumers {exp_ds}"))
        total_edges = len(re.findall(r'<path class="dg-edge ', f.raw))
        unwrapped = total_edges - len(f.edge_hrefs)
        if unwrapped != len(exp_ds):
            issues.append(("error", "cache_downstream_edges",
                           f"{f.anchor}: {unwrapped} unlinked downstream edge(s) vs {len(exp_ds)} expected external consumer(s)"))
        report.setdefault("cache", {})[f.anchor] = {"family": fam, "roles": roles,
                                                    "downstream": act_ds, "state_tensors": sorted(fam_tids)}
    # ---- overview: modules / boundaries / aggregate edge count
    for f in figures:
        if f.kind != "overview":
            continue
        module_nodes = sorted({re.sub(r"^mod-" + re.escape(f.phase) + r"-", "", n.href.lstrip("#"))
                               for n in f.nodes if n.href.startswith("#mod-")})
        exp_modules = sorted({g for g in group_of.values()})
        if module_nodes != exp_modules:
            issues.append(("error", "overview_modules", f"{f.anchor}: modules {module_nodes} != {exp_modules}"))
        boundary = {n.href for n in f.nodes if n.kind == "boundary"}
        for need in ("#t-" + f.phase + "-hidden_streams_in", "#t-" + f.phase + "-hidden_streams_out"):
            if need not in boundary:
                issues.append(("error", "overview_boundary", f"{f.anchor}: missing boundary node {need}"))
        # every overview edge label must name real phase tensors ("N 张量：a，b…" / "读写 N 状态")
        for L in f.labels:
            full = L.full or "".join(L.lines)
            if full.startswith("读写 "):
                continue
            m = re.search(r"张量：(.*)$", full)
            if not m:
                issues.append(("error", "overview_label", f"{f.anchor}: unrecognised edge label {full!r}"))
                continue
            for nm in [x for x in re.split(r"[，,、…]", m.group(1)) if x.strip()]:
                if nm.strip() not in tensors:
                    issues.append(("error", "overview_label_tensor", f"{f.anchor}: label names unknown tensor {nm!r}"))
        report.setdefault("overview", {})[f.anchor] = {"modules": module_nodes,
                                                       "edge_labels": len([l for l in f.labels])}
    # ---- conditions present in cache figures
    for oid, text in CONDITION_OPS.items():
        if oid not in ops:
            continue
        present = any(any(text in (l.full or "") for l in f.labels) for f in figures)
        if not present:
            issues.append(("warning", "condition_label", f"condition label for {oid!r} ({text}) not found in any figure"))


def geometry_findings(figures: list) -> dict:
    out_canvas, node_overlap, spacing_bad, label_out = [], [], [], []
    anchor_in_node, bbox_overlap = [], []
    for f in figures:
        for n in f.nodes:
            if n.x < -0.5 or n.y < -0.5 or n.x + n.w > f.width + 0.5 or n.y + n.h > f.height + 0.5:
                out_canvas.append((f.anchor, n.href))
        for a, b in ((a, b) for i, a in enumerate(f.nodes) for b in f.nodes[i + 1:]):
            if a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h:
                node_overlap.append((f.anchor, a.href, b.href))
        byx = defaultdict(list)
        for L in f.labels:
            byx[L.x].append(L)
        for x, ls in byx.items():
            ls.sort(key=lambda s: s.y)
            prev = None
            for L in ls:
                if prev is not None:
                    gap = L.y - (prev.y + 11 * (len(prev.lines) - 1))
                    if gap < 13 - 0.01:
                        spacing_bad.append((f.anchor, round(gap, 2)))
                prev = L
        for L in f.labels:
            if L.y + 11 * (len(L.lines) - 1) + 2 > f.height + 0.5:
                label_out.append((f.anchor, "".join(L.lines)[:24]))
            for n in f.nodes:
                if n.x <= L.x <= n.x + n.w and n.y <= L.y <= n.y + n.h:
                    anchor_in_node.append((f.kind, f.anchor, "".join(L.lines)[:36], n.href))
                    break
            w = max((line_w(x) for x in L.lines), default=0)
            l, rr = L.x - w / 2, L.x + w / 2
            top, bot = L.y - 9, L.y + 11 * (len(L.lines) - 1) + 3
            for n in f.nodes:
                if l < n.x + n.w and n.x < rr and top < n.y + n.h and n.y < bot:
                    bbox_overlap.append((f.kind, f.anchor, "".join(L.lines)[:36], n.href))
                    break
    return {"figures": len(figures), "node_out_of_canvas": out_canvas, "node_overlap": node_overlap,
            "label_spacing_violations": spacing_bad, "label_out_of_canvas": label_out,
            "label_anchor_in_node": anchor_in_node, "label_bbox_overlap_heuristic": bbox_overlap}


def geometry(figures: list, issues: list, report: dict):
    f = geometry_findings(figures)
    if f["node_out_of_canvas"]:
        issues.append(("error", "node_out_of_canvas", f"{len(f['node_out_of_canvas'])} node rect(s) outside canvas: {f['node_out_of_canvas'][:5]}"))
    if f["node_overlap"]:
        issues.append(("error", "node_overlap", f"{len(f['node_overlap'])} node-node rect overlap(s): {f['node_overlap'][:5]}"))
    if f["label_spacing_violations"]:
        issues.append(("error", "label_spacing", f"{len(f['label_spacing_violations'])} same-x label spacing < 13px: {f['label_spacing_violations'][:5]}"))
    if f["label_out_of_canvas"]:
        issues.append(("error", "label_out_of_canvas", f"{len(f['label_out_of_canvas'])} label(s) beyond canvas: {f['label_out_of_canvas'][:5]}"))
    if f["label_anchor_in_node"]:
        issues.append(("error", "label_occluded",
                       f"{len(f['label_anchor_in_node'])} edge-label anchors lie inside a node rect drawn after the labels "
                       f"(opaque fill covers the glyphs); examples: {f['label_anchor_in_node'][:6]}"))
    report["geometry"] = {"figures": f["figures"], "node_out_of_canvas": len(f["node_out_of_canvas"]),
                          "node_overlap": len(f["node_overlap"]),
                          "label_spacing_violations": len(f["label_spacing_violations"]),
                          "label_out_of_canvas": len(f["label_out_of_canvas"]),
                          "label_anchor_in_node": len(f["label_anchor_in_node"]),
                          "label_bbox_overlap_heuristic": len(f["label_bbox_overlap_heuristic"]),
                          "anchor_in_node_all": f["label_anchor_in_node"],
                          "bbox_overlap_examples": f["label_bbox_overlap_heuristic"][:12],
                          "limitation": "static SVG coordinates + DOM paint order; not a browser font/CSS visual pass"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--workspace-root", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    cand = Path(args.candidate).resolve()
    root = Path(args.workspace_root).resolve() if args.workspace_root else cand.parents[3]
    doc = json.loads((cand / "data/reading_document.json").read_text(encoding="utf-8"))
    html = (cand / "viewer/decoder.html").read_text(encoding="utf-8")
    figures = ph.parse(html)
    issues: list = []
    report = {"candidate": str(cand), "figure_count": len(figures),
              "figure_kinds": dict(Counter(f.kind for f in figures))}
    # linkage integrity
    ids = re.findall(r'\bid="([^"]+)"', html)
    hrefs = re.findall(r'href="#([^"]+)"', html)
    missing = sorted(set(hrefs) - set(ids))
    if len(ids) != len(set(ids)):
        issues.append(("error", "duplicate_html_id", "duplicate element ids in HTML"))
    if missing:
        issues.append(("error", "dangling_href", f"{len(missing)} missing link targets: {missing[:5]}"))
    marker_ids = re.findall(r'<marker id="([^"]+)"', html)
    if len(marker_ids) != len(set(marker_ids)):
        issues.append(("error", "duplicate_marker", "duplicate SVG marker ids"))
    url_targets = set(re.findall(r'url\(#([^)]+)\)', html))
    if url_targets - set(marker_ids):
        issues.append(("error", "marker_url", f"url(#) targets without marker def: {sorted(url_targets - set(marker_ids))[:5]}"))
    report["linkage"] = {"ids": len(ids), "unique_ids": len(set(ids)), "hrefs": len(hrefs),
                         "missing_targets": missing, "markers": len(marker_ids),
                         "external_refs": re.findall(r'(?:src|href)="(https?://[^"]+)"', html)[:5]}
    if pf_ok := ('id="phase-prefill"' in html and 'id="phase-decode"' in html):
        if html.index('id="phase-prefill"') > html.index('id="phase-decode"'):
            issues.append(("error", "phase_order", "decode before prefill"))
    else:
        issues.append(("error", "phase_sections", "missing phase section anchors"))
    for phase in doc["phases"]:
        pf = [f for f in figures if f.phase == phase["id"]]
        check_phase(phase, pf, issues, report)
    geometry(figures, issues, report)

    report["issues"] = [{"severity": s, "code": c, "message": m} for (s, c, m) in issues]
    errors = [i for i in issues if i[0] == "error"]
    report["summary"] = {"error": len(errors), "warning": sum(1 for i in issues if i[0] == "warning")}
    print(f"candidate {cand}")
    print(f"figures {report['figure_count']} kinds {report['figure_kinds']}")
    print(f"linkage {report['linkage']}")
    print(f"geometry {json.dumps(report['geometry'], ensure_ascii=False)[:600]}")
    print(f"issues error={report['summary']['error']} warning={report['summary']['warning']}")
    for i in issues:
        print(f"  {i[0].upper():7s} {i[1]}: {i[2][:200]}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
