#!/usr/bin/env python3
"""ONNX-style HTML display independent check (onnx_view_round01).

Parses the real SVG (compute rects, 24px IO identifiers, tensor/weight/cache/
downstream dots), verifies direction against JSON expectations (reusing
diagram_round02/graph_expect.py), visible text completeness, static colour
pairing, and identity/same-source. The alias table is read as the object under
test (AST, no product import); true connections/shapes come from the frozen JSON.

Usage: onnx_view_check.py --candidate <dir> [--workspace-root <root>] [--json-out <file>]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND02 = HERE.parent / "diagram_round02"
for p in (str(HERE), str(ROUND02)):
    if p not in sys.path:
        sys.path.insert(0, p)
import graph_expect as gx  # noqa: E402

CACHE_PREFIXES = {
    "main": ("cache_keys", "cache_cumlen", "kv_full"),
    "csa": ("cache_buffer_c_", "cache_compressed_c", "cache_entry_count_c", "cache_overlap_c_"),
    "indexer": ("cache_buffer_i_", "cache_compressed_i", "cache_entry_count_i", "cache_overlap_i_"),
}

_SVG = re.compile(r'<svg class="dg".*?</svg>', re.S)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

class Elem:
    def __init__(self, kind, node_id, x, y, w, h, labels, ports, sub, dashed_nav=False):
        self.kind = kind          # compute | io | dot | nav | navlane
        self.node_id = node_id
        self.x, self.y, self.w, self.h = x, y, w, h
        self.labels = labels
        self.ports = ports
        self.sub = sub
        self.dashed_nav = dashed_nav

    def out_anchor(self):
        if self.kind == "dot":
            return (self.x + 6, self.y)
        return (self.x + self.w, self.y + self.h / 2)

    def in_anchor(self):
        if self.kind == "dot":
            return (self.x - 6, self.y)
        return (self.x - 3, self.y + self.h / 2)


class Edge:
    def __init__(self, href, kind, cond, x1, y1, x2, y2):
        self.href, self.kind, self.cond = href, kind, cond
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.tid = None
        if href:
            m = re.match(r"#t-[^-]+-(.+)$", href)
            self.tid = m.group(1) if m else None

    @property
    def start(self):
        return (self.x1, self.y1)

    @property
    def end(self):
        return (self.x2, self.y2)


class Figure:
    def __init__(self, anchor, phase, kind, name, width, height):
        self.anchor, self.phase, self.kind, self.name = anchor, phase, kind, name
        self.width, self.height = width, height
        self.elems, self.edges, self.labels = [], [], []

    def op_ids(self):
        return [e.node_id[3:] for e in self.elems if e.kind == "compute"]


def resolve_edges(fig):
    """Map each edge path start/end to real node anchors -> (src, dst, tid, kind, resolved)."""
    out_map, in_map = {}, {}
    for e in fig.elems:
        out_map[(round(e.out_anchor()[0], 1), round(e.out_anchor()[1], 1))] = e.node_id
        in_map[(round(e.in_anchor()[0], 1), round(e.in_anchor()[1], 1))] = e.node_id
    res = []
    for ed in fig.edges:
        def near(mp, pt):
            key = (round(pt[0], 1), round(pt[1], 1))
            if key in mp:
                return mp[key]
            best, bd = None, 0.6
            for (x, y), nid in mp.items():
                d = abs(x - pt[0]) + abs(y - pt[1])
                if d < bd:
                    best, bd = nid, d
            return best
        src, dst = near(out_map, ed.start), near(in_map, ed.end)
        tid = ed.tid or (src[2:] if src and src.startswith("t:") else None)
        res.append((src, dst, tid, ed.kind, bool(src and dst)))
    return res


def parse(text: str):
    ids = [(m.start(), m.group(1)) for m in re.finditer(r'\bid="([^"]+)"', text)]

    def anchor_for(pos):
        best = ""
        for p, nm in ids:
            if p < pos:
                best = nm
            else:
                break
        return best

    figs = []
    for m in _SVG.finditer(text):
        svg = m.group(0)
        anchor = anchor_for(m.start())
        phase = "prefill" if "-prefill" in anchor or anchor == "phase-prefill" else ("decode" if "-decode" in anchor or anchor == "phase-decode" else None)
        if anchor.startswith("cache-"):
            kind, name = "cache", anchor.split("-", 2)[2]
        elif anchor.startswith("mod-"):
            kind, name = "module", anchor.split("-", 2)[2]
        else:
            kind, name = "overview", "overview"
        vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        fig = Figure(anchor, phase, kind, name, float(vb.group(1)) if vb else 0.0, float(vb.group(2)) if vb else 0.0)
        # compute + nav + navlane groups
        for gm in re.finditer(r'<a href="(#(?:op|mod|cache)-[^"]+)"><g class="dg-node (\w+)" transform="translate\(([-\d.]+),([-\d.]+)\)"><rect width="([\d.]+)" height="([\d.]+)"[^>]*/>(.*?)</g></a>', svg, re.S):
            href, cls, x, y, w, h, inner = gm.group(1), gm.group(2), float(gm.group(3)), float(gm.group(4)), float(gm.group(5)), float(gm.group(6)), gm.group(7)
            labels = [__import__("html").unescape(z) for z in re.findall(r'<text class="nlabel"[^>]*>([^<]*)</text>', inner)]
            sub = [__import__("html").unescape(z) for z in re.findall(r'<text class="nsub"[^>]*>([^<]*)</text>', inner)]
            if cls == "op":
                e = Elem("compute", "op:" + re.match(r"#op-[^-]+-(.+)$", href).group(1), x, y, w, h, labels, [], sub)
            elif cls in ("nav", "navlane"):
                e = Elem(cls, ("m:" if cls == "nav" else "c:") + href.split("-", 2)[2], x, y, w, h, labels, [], sub, dashed_nav=True)
            else:
                e = Elem("compute", "op:" + re.sub(r"^#\w+-[^-]+-", "", href), x, y, w, h, labels, [], sub)
            fig.elems.append(e)
        # io identifiers (24px)
        for gm in re.finditer(r'<a href="(#t-[^"]+)"><g class="dg-port io" transform="translate\(([-\d.]+),([-\d.]+)\)"><rect width="([\d.]+)" height="([\d.]+)"[^>]*/>(.*?)</g></a>', svg, re.S):
            href, x, y, w, h, inner = gm.group(1), float(gm.group(2)), float(gm.group(3)), float(gm.group(4)), float(gm.group(5)), gm.group(6)
            tid = re.match(r"#t-[^-]+-(.+)$", href).group(1)
            pl = [__import__("html").unescape(z) for z in re.findall(r'<text class="portlabel"[^>]*>([^<]*)</text>', inner)]
            fig.elems.append(Elem("io", "t:" + tid, x, y, w, h, [], pl, []))
        # dots
        for gm in re.finditer(r'<a href="(#(?:t|op)-[^"]+)"><g class="dg-dot (\w+)" transform="translate\(([-\d.]+),([-\d.]+)\)"><circle r="([\d.]+)"', svg):
            href, cls, x, y, r = gm.group(1), gm.group(2), float(gm.group(3)), float(gm.group(4)), float(gm.group(5))
            if href.startswith("#t-"):
                nid = "t:" + re.match(r"#t-[^-]+-(.+)$", href).group(1)
            else:
                nid = "d:" + re.match(r"#op-[^-]+-(.+)$", href).group(1)
            fig.elems.append(Elem("dot", nid, x, y, 2 * r, 2 * r, [], [], [], dashed_nav=(cls == "downstream")))
        # edges
        for em in re.finditer(r'(?:<a href="(#t-[^"]+)">)?<path class="dg-edge (\w+)( cond)?" d="M([-\d.]+),([-\d.]+) C[^"]* ([-\d.]+),([-\d.]+)"', svg):
            fig.edges.append(Edge(em.group(1), em.group(2), bool(em.group(3)), float(em.group(4)), float(em.group(5)), float(em.group(6)), float(em.group(7))))
        # labels
        for lm in re.finditer(r'<text class="(elabel[^\"]*)" x="([-\d.]+)" y="([-\d.]+)"[^>]*>(.*?)</text>', svg, re.S):
            body = lm.group(4)
            lines = [__import__("html").unescape(z) for z in re.findall(r"<tspan[^>]*>([^<]*)</tspan>", body)]
            t = re.search(r"<title>([^<]*)</title>", body)
            full = __import__("html").unescape(t.group(1)) if t else "".join(lines)
            fig.labels.append({"cls": lm.group(1), "x": float(lm.group(2)), "y": float(lm.group(3)), "lines": lines,
                               "full": full, "cond": "elabel-cond" in body or "elabel-cond" in lm.group(1)})
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# alias table (inspected object; read via AST, not imported)
# ---------------------------------------------------------------------------

def load_aliases(project_root: Path):
    src = (project_root / "src/structure_reader/render/diagram_source_labels.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    op_source, tensor_source, type_map = {}, {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                continue
            if name == "OP_SOURCE":
                op_source = {k: v[0] for k, v in val.items()}
            elif name == "_T":
                tensor_source = {k: v[0] for k, v in val.items()}
            elif name == "_TYPE_MAP":
                type_map = val
    return op_source, tensor_source, type_map


# ---------------------------------------------------------------------------
# colour contrast
# ---------------------------------------------------------------------------

def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lum(hexs):
    hexs = hexs.lstrip("#")
    r, g, b = int(hexs[0:2], 16), int(hexs[2:4], 16), int(hexs[4:6], 16)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a, b):
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check(candidate: Path, workspace_root: Path):
    issues = []
    report = {}
    doc = json.loads((candidate / "data/reading_document.json").read_text(encoding="utf-8"))
    html = (candidate / "viewer/decoder.html").read_text(encoding="utf-8")
    figs = parse(html)
    report["figure_kinds"] = dict(Counter(f.kind for f in figs))
    all_elems = [e for f in figs for e in f.elems]
    report["elem_classes"] = dict(Counter(e.kind for e in all_elems))
    report["dot_kinds_html"] = dict(Counter(re.findall(r'<g class="dg-dot (\w+)"', html)))

    # 1) classification
    op_boxes = [e for f in figs if f.kind in ("module", "cache") for e in f.elems if e.kind == "compute"]
    if not op_boxes:
        issues.append(("error", "no_compute_boxes", "no dg-node op compute boxes parsed"))
    small_boxes = [e for e in op_boxes if e.h < 24]
    if small_boxes:
        issues.append(("error", "compute_box_too_small", f"{len(small_boxes)} compute boxes <24px"))
    ios = [e for e in all_elems if e.kind == "io"]
    bad_io = [e for e in ios if abs(e.h - 24) > 0.5]
    if bad_io:
        issues.append(("error", "io_not_24px", f"{len(bad_io)} IO identifiers not 24px tall"))
    io_ids = {e.node_id for e in ios}
    report["io_ids"] = sorted(io_ids)
    if not any("hidden_streams_in" in i for i in io_ids) or not any("hidden_streams_out" in i for i in io_ids):
        issues.append(("error", "hidden_states_io", f"hidden_states input/output IO identifiers missing: {sorted(io_ids)}"))
    dot_elems = [e for e in all_elems if e.kind == "dot"]
    report["dot_parsed"] = len(dot_elems)
    navs = [e for e in all_elems if e.kind in ("nav", "navlane")]
    report["nav_elems"] = len(navs)

    # 2) direction (module/cache), reuse graph_expect
    direction = {}
    for phase in doc["phases"]:
        tensors = {t["id"]: t for t in phase["tensors"]}
        for f in figs:
            if f.phase != phase["id"] or f.kind not in ("module", "cache"):
                continue
            op_ids = set(f.op_ids())
            exp = gx.expected_edges(phase, op_ids)
            if f.kind == "cache":
                fam_tids = {t for t, tv in tensors.items() if tv.get("kind") == "cache" and any(t == p or t.startswith(p) for p in CACHE_PREFIXES[f.name])}
                exp |= gx.expected_cache_downstream(phase, fam_tids, op_ids)
            geo, unresolved = set(), []
            for src, dst, tid, kind, ok in resolve_edges(f):
                if not ok:
                    unresolved.append((kind,))
                    continue
                geo.add((src, dst, tid, kind))
            missing = exp - geo
            extra = geo - exp
            rev = [m for m in missing if (m[1], m[0], m[2], m[3]) in extra]
            real_missing = [m for m in missing if m not in rev]
            real_extra = [x for x in extra if (x[1], x[0], x[2], x[3]) not in missing]
            direction[f.anchor] = {"exp": len(exp), "geo": len(geo), "missing": len(real_missing), "extra": len(real_extra), "reversed": len(rev), "unresolved": len(unresolved)}
            if rev:
                issues.append(("error", "edge_reversed", f"{f.anchor}: {rev[:3]}"))
            if real_missing:
                issues.append(("error", "edge_missing", f"{f.anchor}: {real_missing[:3]}"))
            if real_extra:
                issues.append(("error", "edge_extra", f"{f.anchor}: {real_extra[:3]}"))
            if unresolved:
                issues.append(("error", "edge_unresolved", f"{f.anchor}: {unresolved[:3]}"))
    report["direction_total"] = {k: sum(v[k] for v in direction.values()) for k in ("exp", "geo", "missing", "extra", "reversed", "unresolved")}
    report["direction_figures"] = len(direction)

    # overview endpoints (resolve only, listed separately)
    ov = {}
    for f in figs:
        if f.kind != "overview":
            continue
        out_map, in_map = {}, {}
        for e in f.elems:
            out_map[(round(e.out_anchor()[0], 1), round(e.out_anchor()[1], 1))] = e.node_id
            in_map[(round(e.in_anchor()[0], 1), round(e.in_anchor()[1], 1))] = e.node_id
        res = 0
        for ed in f.edges:
            if (round(ed.start[0], 1), round(ed.start[1], 1)) in out_map and (round(ed.end[0], 1), round(ed.end[1], 1)) in in_map:
                res += 1
        ov[f.anchor] = {"edges": len(f.edges), "resolved": res, "width": f.width}
        if res != len(f.edges):
            issues.append(("warning", "overview_endpoints", f"{f.anchor}: {len(f.edges) - res} endpoints unresolved"))
    report["overview"] = ov

    # 3) visible label completeness (name+shape, condition simultaneously)
    op_source, tensor_source, type_map = load_aliases(workspace_root / "structure_reader")
    label_full = [l["full"] for f in figs for l in f.labels]
    label_count = len(label_full)
    report["edge_label_count"] = label_count
    cond_labels = sum(1 for f in figs for l in f.labels if l["cond"])
    report["cond_label_count"] = cond_labels
    unmapped, missing_labels = 0, []
    for phase in doc["phases"]:
        tensors = {t["id"]: t for t in phase["tensors"]}
        for f in figs:
            if f.phase != phase["id"]:
                continue
            for ed in f.edges:
                if not ed.tid:
                    continue
                name = tensor_source.get(ed.tid)
                sh = tensors.get(ed.tid, {}).get("shape")
                if name is None or sh is None:
                    unmapped += 1
                    continue
                body = f"{name} [{','.join(sh)}]" if sh else name
                # labels may carry a role/condition prefix line before name+shape; require the
                # exact name+shape suffix to be present (no truncation of the visible text)
                ok = any(l == body or l.endswith(body) for l in label_full)
                if not ok:
                    missing_labels.append((f.anchor, ed.tid, body, ed.cond))
    report["unmapped_tid"] = unmapped
    if missing_labels:
        issues.append(("error", "edge_label_missing", f"{len(missing_labels)} edge labels lack visible name+shape(+cond): {missing_labels[:5]}"))

    # 4) long op names complete (both phases) + type present
    focus_ops = ["moe_expert_gu", "moe_expert_slice", "i_topk", "attn_residual_mix", "ffn_residual_mix"]
    op_boxes_by_id = {(f.phase, e.node_id): e for f in figs for e in f.elems if e.kind == "compute"}
    for oid in focus_ops:
        for ph in ("prefill", "decode"):
            e = op_boxes_by_id.get((ph, "op:" + oid))
            if e is None:
                issues.append(("error", "focus_op_missing", f"{ph}/{oid} compute box missing"))
                continue
            joined = "".join(e.labels)
            expected = op_source.get(oid)
            if expected and joined != expected:
                issues.append(("error", "op_name_truncated", f"{ph}/{oid}: visible {joined!r} != source {expected!r}"))
            if not e.sub:
                issues.append(("error", "op_type_missing", f"{ph}/{oid}: compute type label missing"))

    # 5) source-name correctness (static HF source)
    modeling = (workspace_root / "hflib/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py").read_text(encoding="utf-8").splitlines()
    name_checks = {
        "moe_expert_gu": "F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])",
        "i_scorer": "(scores * weights.unsqueeze(-1)).sum(dim=2)",
        "q_a_norm": "self.q_a_norm(self.q_a_proj(hidden_states))",
        "moe_expert_down": "F.linear(current, self.down_proj[expert_idx])",
    }
    for oid, expect_src in name_checks.items():
        label = op_source.get(oid)
        # compare against the actual source line (normalise spaces)
        src_norm = re.sub(r"\s+", "", "".join(modeling))
        exp_norm = re.sub(r"\s+", "", expect_src)
        if exp_norm not in src_norm:
            issues.append(("error", "name_source_mismatch", f"{oid}: '{expect_src}' not found in modeling source"))
    report["name_source_checks"] = list(name_checks)

    # 6) colours (static pairing, light + dark)
    style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    def decl(selector, prop):
        ms = re.findall(re.escape(selector) + r"\s*\{([^}]*)\}", style)
        val = None
        for body in ms:
            for part in body.split(";"):
                if prop in part and ":" in part:
                    val = part.split(":", 1)[1].strip()
        return val
    light_card = decl(":root", "--card") or "#ffffff"
    dark_card = "#1d2126"
    pairs = {
        "light": {
            "nlabel/opbox": ("#1f2933", "#fcfdfe", "text"),
            "nsub/opbox": ("#4a5568", "#fcfdfe", "text"),
            "portlabel/io": ("#1f2933", "#e9f7f0", "text"),
            "elabel/halo": ("#33404d", "#f6f7f9", "text"),
            "flow/opbox": ("#1f4e79", "#fcfdfe", "line"),
            "weight/opbox": ("#b08d57", "#fcfdfe", "line"),
            "cache/opbox": ("#2b5d8f", "#fcfdfe", "line"),
        },
        "dark": {
            "nlabel/opbox": ("#1f2933", "#fcfdfe", "text"),
            "nsub/opbox": ("#4a5568", "#fcfdfe", "text"),
            "portlabel/io": ("#1f2933", "#e9f7f0", "text"),
            "elabel/halo": ("#d9e2ea", "#14171b", "text"),
            "flow/card": ("#7fa8d9", dark_card, "line"),
            "weight/card": ("#d4b183", dark_card, "line"),
            "cache/card": ("#9fc2e8", dark_card, "line"),
        },
    }
    colour = {}
    for theme, items in pairs.items():
        for name, (fg, bg, kind) in items.items():
            ratio = round(contrast(fg, bg), 2)
            colour[f"{theme}:{name}"] = ratio
            thr = 4.5 if kind == "text" else 3.0
            if ratio < thr:
                issues.append(("error", "colour_contrast", f"{theme}:{name} contrast {ratio} < {thr}"))
    report["colour_contrast"] = colour

    # 7) linkage / regressions
    ids = re.findall(r'\bid="([^"]+)"', html)
    hrefs = re.findall(r'href="#([^"]+)"', html)
    markers = re.findall(r'<marker id="([^"]+)"', html)
    url_targets = set(re.findall(r'url\(#([^)]+)\)', html))
    report["linkage"] = {"ids": len(ids), "unique": len(set(ids)), "hrefs": len(hrefs),
                         "missing": sorted(set(hrefs) - set(ids))[:5], "markers": len(markers),
                         "marker_dupes": len(markers) - len(set(markers)), "marker_url_unresolved": sorted(url_targets - set(markers))[:5],
                         "external": re.findall(r'(?:src|href)="(https?://[^"]+)"', html)[:5]}
    if len(ids) != len(set(ids)) or (set(hrefs) - set(ids)) or len(markers) != len(set(markers)) or (url_targets - set(markers)):
        issues.append(("error", "linkage", f"linkage problem: {report['linkage']}"))
    if "id=\"phase-prefill\"" in html and "id=\"phase-decode\"" in html:
        if html.index('id="phase-prefill"') > html.index('id="phase-decode"'):
            issues.append(("error", "phase_order", "decode before prefill"))
    else:
        issues.append(("error", "phase_sections", "missing phase anchors"))

    report["summary"] = {"error": sum(1 for i in issues if i[0] == "error"), "warning": sum(1 for i in issues if i[0] == "warning")}
    report["issues"] = [{"severity": s, "code": c, "message": m} for s, c, m in issues]
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--workspace-root", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    cand = Path(args.candidate).resolve()
    root = Path(args.workspace_root).resolve() if args.workspace_root else cand.parents[3]
    report = check(cand, root)
    print(f"candidate {cand}")
    print(f"figures {report['figure_kinds']} elems {report['elem_classes']}")
    print(f"io {len(report['io_ids'])} dots {report['dot_parsed']} nav {report['nav_elems']}")
    print(f"direction {report['direction_total']} figures {report['direction_figures']}")
    print(f"overview {report['overview']}")
    print(f"labels {report['edge_label_count']} cond {report['cond_label_count']} unmapped {report['unmapped_tid']}")
    print(f"colour {report['colour_contrast']}")
    print(f"linkage {report['linkage']}")
    print(f"issues error={report['summary']['error']} warning={report['summary']['warning']}")
    for i in report["issues"]:
        print(f"  {i['severity'].upper():7s} {i['code']}: {i['message'][:200]}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if report["summary"]["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
