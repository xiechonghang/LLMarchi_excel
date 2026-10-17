"""Static parser for the decoder page diagrams (no browser, no JS).

Extracts, per embedded ``<svg class="dg">`` figure: the anchor id from the
nearest preceding element id, op/tensor/downstream node hrefs, edge tensor hrefs
+ kinds, edge labels, node rectangles, and edge-label texts with coordinates.
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field


@dataclass
class Node:
    href: str
    kind: str
    x: float
    y: float
    w: float
    h: float
    nlabel_lines: list[str]
    sub: str


@dataclass
class EdgeLabel:
    x: float
    y: float
    lines: list[str]
    full: str
    cond: bool


@dataclass
class Figure:
    anchor: str
    phase: str | None
    kind: str  # overview | module | cache
    name: str
    width: float
    height: float
    markers: list[str]
    edge_hrefs: list[tuple[str, str]]  # (tid, kind) for edges wrapped in <a href="#t-...">
    labelled_unwrapped: list[str]  # labels of edge paths without href
    nodes: list[Node] = field(default_factory=list)
    labels: list[EdgeLabel] = field(default_factory=list)
    raw: str = ""

    def op_ids(self) -> list[str]:
        out = []
        for n in self.nodes:
            if n.kind != "op":
                continue
            m = re.match(r"#op-([^-]+)-(.+)$", n.href)
            if m and m.group(2):
                out.append(m.group(2))
        return out

    def tensor_ids(self) -> list[str]:
        out = []
        for n in self.nodes:
            if n.kind not in ("port", "boundary", "weight", "cache"):
                continue
            m = re.match(r"#t-([^-]+)-(.+)$", n.href)
            if m and m.group(2):
                out.append(m.group(2))
        return out

    def downstream_ids(self) -> list[str]:
        out = []
        for n in self.nodes:
            if n.kind != "downstream":
                continue
            m = re.match(r"#op-([^-]+)-(.+)$", n.href)
            if m and m.group(2):
                out.append(m.group(2))
        return out


_SVG_RE = re.compile(r'<svg class="dg".*?</svg>', re.S)
_ID_RE = re.compile(r'\bid="([^"]+)"')


def _near_anchor_ids(html: str):
    return [(m.start(), m.group(1)) for m in _ID_RE.finditer(html)]


def _anchor_for(pos: int, ids) -> str:
    best = ""
    for p, name in ids:
        if p < pos:
            best = name
        else:
            break
    return best


def _phase_of(anchor: str) -> str | None:
    for ph in ("prefill", "decode"):
        if f"-{ph}" in anchor or anchor == f"phase-{ph}":
            return ph
    return None


def parse(html: str) -> list[Figure]:
    ids = _near_anchor_ids(html)
    figures: list[Figure] = []
    for m in _SVG_RE.finditer(html):
        svg = m.group(0)
        anchor = _anchor_for(m.start(), ids)
        phase = _phase_of(anchor)
        if anchor.startswith("cache-"):
            kind, name = "cache", anchor.split("-", 2)[2]
        elif anchor.startswith("mod-"):
            kind, name = "module", anchor.split("-", 2)[2]
        else:
            kind, name = "overview", "overview"
        vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        width, height = (float(vb.group(1)), float(vb.group(2))) if vb else (0.0, 0.0)
        markers = re.findall(r'id="(dg-arr-[a-z]+-[^"]+)"', svg)
        edge_hrefs = re.findall(r'<a href="#t-([^-]+)-([^"]+)"><path class="dg-edge ([a-z]+)', svg)
        edge_hrefs = [(f"{a}-{b}" if False else b, c) for (a, b, c) in edge_hrefs]
        # labels of edge paths that are NOT wrapped in an href
        unwrapped = re.findall(r'(?<!</a>)<path class="dg-edge [a-z]+"[^>]*marker-end="url\(#dg-arr-[a-z]+-[^"]+\)"/>', svg)
        nodes = []
        for nm in re.finditer(
            r'<a href="([^"]+)"><g class="dg-node ([a-z]+)" transform="translate\(([-\d.]+),([-\d.]+)\)">'
            r'<rect width="([\d.]+)" height="([\d.]+)"[^>]*/>', svg):
            href, nkind, x, y, w, h = nm.group(1), nm.group(2), float(nm.group(3)), float(nm.group(4)), float(nm.group(5)), float(nm.group(6))
            block_end = svg.find("</g></a>", nm.end())
            block = svg[nm.end():block_end if block_end > 0 else nm.end() + 2000]
            lines = [_html.unescape(x) for x in re.findall(r'<text class="nlabel"[^>]*>([^<]*)</text>', block)]
            sub = re.search(r'<text class="nsub"[^>]*>([^<]*)</text>', block)
            nodes.append(Node(href, nkind, x, y, w, h, lines, _html.unescape(sub.group(1)) if sub else ""))
        labels = []
        for lm in re.finditer(
            r'<text class="(elabel[^"]*)" x="([-\d.]+)" y="([-\d.]+)" text-anchor="middle">(.*?)</text>', svg, re.S):
            cls, x, y, body = lm.group(1), float(lm.group(2)), float(lm.group(3)), lm.group(4)
            lines = [_html.unescape(x) for x in re.findall(r"<tspan[^>]*>([^<]*)</tspan>", body)]
            full = re.search(r"<title>([^<]*)</title>", body)
            labels.append(EdgeLabel(x, y, lines, _html.unescape(full.group(1)) if full else "".join(lines), "cond" in cls))
        figures.append(Figure(anchor, phase, kind, name, width, height, markers, edge_hrefs,
                              unwrapped, nodes, labels, svg))
    return figures
