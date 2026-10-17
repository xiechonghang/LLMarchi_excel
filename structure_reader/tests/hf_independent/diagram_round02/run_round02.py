#!/usr/bin/env python3
"""Round02 independent checks: diagram direction + label-block occlusion + regressions.

Diagnostic; exit 1 on any `error`.
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
import parse_svg as ps  # noqa: E402
import graph_expect as gx  # noqa: E402

CACHE_PREFIXES = {
    "main": ("cache_keys", "cache_cumlen", "kv_full"),
    "csa": ("cache_buffer_c_", "cache_compressed_c", "cache_entry_count_c", "cache_overlap_c_"),
    "indexer": ("cache_buffer_i_", "cache_compressed_i", "cache_entry_count_i", "cache_overlap_i_"),
}
CONDITION_TEXTS = ("仅 N_win>0", "仅 N_win_i>0", "追加 N_win 条（可为 0）", "追加 N_win_i 条（可为 0）",
                   "余量回写（每次调用）")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def node_id(n: ps.Node) -> str:
    m = re.match(r"#op-[^-]+-(.+)$", n.href)
    if m and n.kind == "op":
        return "op:" + m.group(1)
    if m and n.kind == "downstream":
        return "d:" + m.group(1)
    m = re.match(r"#t-[^-]+-(.+)$", n.href)
    return "t:" + m.group(1) if m else n.href


def resolve_edge_endpoints(fig: ps.Figure):
    out_map, in_map = {}, {}
    for n in fig.nodes:
        out_map[round(n.out_anchor[0], 1), round(n.out_anchor[1], 1)] = node_id(n)
        in_map[round(n.in_anchor[0], 1), round(n.in_anchor[1], 1)] = node_id(n)

    def nearest(mp, pt):
        key = (round(pt[0], 1), round(pt[1], 1))
        if key in mp:
            return mp[key]
        best, bd = None, 1.5
        for (x, y), nid in mp.items():
            d = abs(x - pt[0]) + abs(y - pt[1])
            if d < bd:
                best, bd = nid, d
        return best

    resolved = []
    for e in fig.edges:
        src = nearest(out_map, e.start)
        dst = nearest(in_map, e.end)
        tid = e.tid
        if tid is None and src and src.startswith("t:"):
            tid = src[2:]
        resolved.append({"edge": e, "src": src, "dst": dst, "tid": tid})
    return resolved


def check_phase(phase, figures, issues, report):
    tensors = {t["id"]: t for t in phase["tensors"]}
    params = {p["id"] for p in phase["parameters"]}
    for fig in figures:
        if fig.kind not in ("module", "cache"):
            continue
        op_ids = set(fig.op_ids())
        exp = gx.expected_edges(phase, op_ids)
        if fig.kind == "cache":
            fam = fig.name
            fam_tids = {t for t, tv in tensors.items() if tv.get("kind") == "cache"
                        and any(t == p or t.startswith(p) for p in CACHE_PREFIXES[fam])}
            exp |= gx.expected_cache_downstream(phase, fam_tids, op_ids)
        res = resolve_edge_endpoints(fig)
        geo = set()
        unresolved = []
        for r in res:
            if not r["src"] or not r["dst"]:
                unresolved.append((r["edge"].kind, r["edge"].start, r["edge"].end))
                continue
            geo.add((r["src"], r["dst"], r["tid"], r["edge"].kind))
        missing = exp - geo
        extra = geo - exp
        reversed_edges = [m for m in missing if (m[1], m[0], m[2], m[3]) in extra]
        if reversed_edges:
            issues.append(("error", "edge_reversed", f"{fig.anchor}: reversed edge(s) {reversed_edges[:4]}"))
        real_missing = [m for m in missing if m not in reversed_edges]
        if real_missing:
            issues.append(("error", "edge_missing", f"{fig.anchor}: {len(real_missing)} expected edge(s) absent: {real_missing[:4]}"))
        if extra:
            real_extra = [x for x in extra if (x[1], x[0], x[2], x[3]) not in missing]
            if real_extra:
                issues.append(("error", "edge_unexpected", f"{fig.anchor}: {len(real_extra)} unexpected edge(s): {real_extra[:4]}"))
        if unresolved:
            issues.append(("error", "edge_unresolved", f"{fig.anchor}: {len(unresolved)} edge endpoint(s) not at any node anchor: {unresolved[:3]}"))
        report.setdefault("direction", {})[fig.anchor] = {"expected": len(exp), "geometry": len(geo), "missing": len(real_missing),
                                                          "extra": len(extra) - len(reversed_edges), "reversed": len(reversed_edges),
                                                          "unresolved": len(unresolved)}
    # cache roles + conditions
    for fig in figures:
        if fig.kind != "cache":
            continue
        fam = fig.name
        op_ids = set(fig.op_ids())
        prod, cons = defaultdict(list), defaultdict(list)
        for i, op in enumerate(phase["operators"]):
            for t in op.get("outputs", []):
                prod[t].append(i)
            for t in op.get("inputs", []):
                cons[t].append(i)
        fam_tids = {t for t, tv in tensors.items() if tv.get("kind") == "cache"
                    and any(t == p or t.startswith(p) for p in CACHE_PREFIXES[fam])}
        for n in fig.nodes:
            tid = None
            m = re.match(r"#t-[^-]+-(.+)$", n.href)
            if m:
                tid = m.group(1)
            if tid not in fam_tids:
                continue
            if not prod.get(tid):
                role = "调用前状态"
            else:
                outside = [phase["operators"][c]["id"] for c in cons.get(tid, []) if phase["operators"][c]["id"] not in op_ids]
                role = ("本次读视图" + ("·兼调用后保存" if tid.endswith("_post") else "")) if outside else "调用后回写"
            if role.split("·")[0] not in n.sub:
                issues.append(("error", "cache_role", f"{fig.anchor}: {tid} sub {n.sub!r} != role {role!r}"))
        report.setdefault("cache_roles", {})[fig.anchor] = "checked"
    for text in CONDITION_TEXTS:
        if not any(text in l.full for f in figures for l in f.labels):
            issues.append(("warning", "condition_label", f"condition text {text!r} not found"))


def geometry(figures, issues, report):
    out_canvas, node_overlap, spacing_bad, label_out, anchor_in_node, block_hits = [], [], [], [], [], []
    for fig in figures:
        for n in fig.nodes:
            if n.x < -0.5 or n.y < -0.5 or n.x + n.w > fig.width + 0.5 or n.y + n.h > fig.height + 0.5:
                out_canvas.append((fig.anchor, n.href))
        for i, a in enumerate(fig.nodes):
            for b in fig.nodes[i + 1:]:
                if a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h:
                    node_overlap.append((fig.anchor, a.href, b.href))
        byx = defaultdict(list)
        for L in fig.labels:
            byx[L.x].append(L)
        for x, ls in byx.items():
            ls.sort(key=lambda s: s.y)
            prev = None
            for L in ls:
                if prev is not None and L.y - (prev.y + 11 * (len(prev.lines) - 1)) < 13 - 0.01:
                    spacing_bad.append((fig.anchor, round(L.y - (prev.y + 11 * (len(prev.lines) - 1)), 2)))
                prev = L
        for L in fig.labels:
            if L.y + 11 * (len(L.lines) - 1) + 2 > fig.height + 0.5:
                label_out.append((fig.anchor, L.full[:24]))
            half = ps.label_half_w(L.lines)
            left, right = L.x - half, L.x + half
            top, bot = L.y - 8, L.y + 11 * (len(L.lines) - 1) + 2
            for n in fig.nodes:
                if n.x <= L.x <= n.x + n.w and n.y <= L.y <= n.y + n.h:
                    anchor_in_node.append((fig.kind, fig.anchor, L.full[:32], n.href))
                    break
            for n in fig.nodes:
                if left < n.x + n.w and n.x < right and top < n.y + n.h and n.y < bot:
                    ow = min(right, n.x + n.w) - max(left, n.x)
                    oh = min(bot, n.y + n.h) - max(top, n.y)
                    # visible text half-width (metric minus the +4 detection margin)
                    vhalf = max(half - 4.0, 0.1)
                    # min font scale s (actual/estimated) at which the tail leaves the rect
                    s_candidates = [(n.x - L.x) / vhalf] if n.x > L.x else []
                    s_candidates.append(((L.x) - (n.x + n.w)) / vhalf) if (n.x + n.w) < L.x else None
                    s_min = min([s for s in s_candidates if s > 0], default=99.0)
                    block_hits.append((fig.kind, fig.anchor, L.full[:32], n.href, round(ow, 1), round(oh, 1), round(s_min, 2)))
                    break
    for name, coll in (("node_out_of_canvas", out_canvas), ("node_overlap", node_overlap),
                       ("label_spacing", spacing_bad), ("label_out_of_canvas", label_out),
                       ("anchor_in_node", anchor_in_node)):
        if coll:
            issues.append(("error", name, f"{len(coll)}: {coll[:4]}"))
    # robust occlusion: text tail stays inside a node unless the real font is >30% narrower than the metric
    robust = [h for h in block_hits if h[6] < 0.7]
    if robust:
        issues.append(("error", "label_block_overlap",
                       f"{len(robust)} edge-label block(s) robustly overlap a later-painted node rect (tail inside node unless font >30% narrower than the renderer metric): {robust[:8]}"))
    report["geometry"] = {"figures": len(figures), "node_out_of_canvas": len(out_canvas), "node_overlap": len(node_overlap),
                          "label_spacing_violations": len(spacing_bad), "label_out_of_canvas": len(label_out),
                          "anchor_in_node": len(anchor_in_node), "label_block_overlap": len(block_hits),
                          "label_block_overlap_robust": len(robust),
                          "block_overlap_all": block_hits, "block_overlap_robust": robust,
                          "anchor_in_node_all": anchor_in_node,
                          "limitation": "static SVG coordinates + DOM paint order + renderer width metric; not a browser font pass"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--workspace-root", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    cand = Path(args.candidate).resolve()
    root = Path(args.workspace_root).resolve() if args.workspace_root else cand.parents[3]
    doc = json.loads((cand / "data/reading_document.json").read_text(encoding="utf-8"))
    html = (cand / "viewer/decoder.html").read_text(encoding="utf-8")
    figures = ps.parse(html)
    issues = []
    report = {"candidate": str(cand), "figure_count": len(figures), "figure_kinds": dict(Counter(f.kind for f in figures))}
    # linkage
    ids = re.findall(r'\bid="([^"]+)"', html)
    hrefs = re.findall(r'href="#([^"]+)"', html)
    missing_targets = sorted(set(hrefs) - set(ids))
    markers = re.findall(r'<marker id="([^"]+)"', html)
    url_targets = set(re.findall(r'url\(#([^)]+)\)', html))
    report["linkage"] = {"ids": len(ids), "unique_ids": len(set(ids)), "hrefs": len(hrefs), "missing_targets": missing_targets,
                         "markers": len(markers), "marker_dupes": len(markers) - len(set(markers)),
                         "marker_urls_unresolved": sorted(url_targets - set(markers)),
                         "external_refs": re.findall(r'(?:src|href)="(https?://[^"]+)"', html)[:5]}
    if len(ids) != len(set(ids)):
        issues.append(("error", "duplicate_id", "duplicate DOM ids"))
    if missing_targets:
        issues.append(("error", "dangling_href", f"{len(missing_targets)} missing: {missing_targets[:5]}"))
    if url_targets - set(markers):
        issues.append(("error", "marker_url", f"{sorted(url_targets - set(markers))[:5]}"))
    if "id=\"phase-prefill\"" in html and "id=\"phase-decode\"" in html:
        if html.index('id="phase-prefill"') > html.index('id="phase-decode"'):
            issues.append(("error", "phase_order", "decode before prefill"))
    else:
        issues.append(("error", "phase_sections", "missing phase anchors"))
    for phase in doc["phases"]:
        check_phase(phase, [f for f in figures if f.phase == phase["id"]], issues, report)
    geometry(figures, issues, report)
    report["issues"] = [{"severity": s, "code": c, "message": m} for s, c, m in issues]
    report["summary"] = {"error": sum(1 for i in issues if i[0] == "error"), "warning": sum(1 for i in issues if i[0] == "warning")}
    print(f"candidate {cand}")
    print(f"figures {report['figure_count']} {report['figure_kinds']}")
    print(f"linkage {report['linkage']}")
    g = report["geometry"]
    print(f"geometry anchor_in_node={g['anchor_in_node']} block_overlap={g['label_block_overlap']} node_overlap={g['node_overlap']} spacing={g['label_spacing_violations']} out_canvas={g['node_out_of_canvas']} label_out={g['label_out_of_canvas']}")
    print(f"issues error={report['summary']['error']} warning={report['summary']['warning']}")
    for s, c, m in issues:
        print(f"  {s.upper():7s} {c}: {m[:220]}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if report["summary"]["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
