"""Static parser for decoder-page SVG diagrams (round02, self-contained).

Extracts per ``<svg class="dg">`` figure: anchor id, nodes (href/kind/rect),
edge paths (start/end/kind/href) and edge labels with coordinates. Also exposes
the renderer's label-block width metric (CJK 10 / latin 5.5 + 4 half) so the
block-occlusion question can be judged consistently with the implementation.
"""

from __future__ import annotations

import html as H
import re
from dataclasses import dataclass, field

NODE_W = 172.0


@dataclass
class Node:
    href: str
    kind: str
    x: float
    y: float
    w: float
    h: float
    nlabel_lines: list
    sub: str = ""

    @property
    def out_anchor(self):
        return (self.x + self.w, self.y + self.h / 2)

    @property
    def in_anchor(self):
        return (self.x, self.y + self.h / 2)


@dataclass
class Edge:
    x1: float
    y1: float
    cx1: float
    cy1: float
    cx2: float
    cy2: float
    x2: float  # = true end x - 3 (marker offset)
    y2: float
    kind: str
    href: str | None
    tid: str | None
    cond: bool

    @property
    def start(self):
        return (self.x1, self.y1)

    @property
    def end(self):
        return (self.x2 + 3.0, self.y2)


@dataclass
class Label:
    x: float
    y: float
    lines: list
    full: str
    cond: bool


@dataclass
class Figure:
    anchor: str
    phase: str | None
    kind: str
    name: str
    width: float
    height: float
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    raw: str = ""

    def op_ids(self):
        return [re.match(r"#op-[^-]+-(.+)$", n.href).group(1) for n in self.nodes
                if n.kind == "op" and re.match(r"#op-[^-]+-(.+)$", n.href)]

    def downstream_ids(self):
        return [re.match(r"#op-[^-]+-(.+)$", n.href).group(1) for n in self.nodes
                if n.kind == "downstream" and re.match(r"#op-[^-]+-(.+)$", n.href)]


def block_half_w(text: str) -> float:
    return max(sum(10.0 if ord(c) > 0x2E80 else 5.5 for c in line) for line in [text]) / 2 + 4.0


def label_half_w(lines) -> float:
    return max(sum(10.0 if ord(c) > 0x2E80 else 5.5 for c in line) for line in lines) / 2 + 4.0


_SVG = re.compile(r'<svg class="dg".*?</svg>', re.S)


def parse(text: str) -> list:
    ids = [(m.start(), m.group(1)) for m in re.finditer(r'\bid="([^"]+)"', text)]

    def anchor_for(pos):
        best = ""
        for p, name in ids:
            if p < pos:
                best = name
            else:
                break
        return best

    figures = []
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
        width, height = (float(vb.group(1)), float(vb.group(2))) if vb else (0.0, 0.0)
        nodes = []
        for nm in re.finditer(
                r'<a href="([^"]+)"><g class="dg-node ([a-z]+)" transform="translate\(([-\d.]+),([-\d.]+)\)">'
                r'<rect width="([\d.]+)" height="([\d.]+)"', svg):
            href, nk, x, y, w, hh = nm.group(1), nm.group(2), float(nm.group(3)), float(nm.group(4)), float(nm.group(5)), float(nm.group(6))
            blk_end = svg.find("</g></a>", nm.end())
            blk = svg[nm.end():blk_end if blk_end > 0 else nm.end() + 1500]
            lines = [H.unescape(x) for x in re.findall(r'<text class="nlabel"[^>]*>([^<]*)</text>', blk)]
            subm = re.search(r'<text class="nsub"[^>]*>([^<]*)</text>', blk)
            nodes.append(Node(href, nk, x, y, w, hh, lines, H.unescape(subm.group(1)) if subm else ""))
        edges = []
        # wrapped edges carry href (tensor id); unwrapped (downstream) do not
        for em in re.finditer(r'(?:<a href="(#[^"]*)">)?<path class="dg-edge ([a-z]+)( cond)?" d="M([-\d.]+),([-\d.]+) C([-\d.]+),([-\d.]+) ([-\d.]+),([-\d.]+) ([-\d.]+),([-\d.]+)"', svg):
            href = em.group(1)
            tid = None
            if href:
                mm = re.match(r"#t-[^-]+-(.+)$", href)
                tid = mm.group(1) if mm else None
            edges.append(Edge(float(em.group(4)), float(em.group(5)), float(em.group(6)), float(em.group(7)),
                              float(em.group(8)), float(em.group(9)), float(em.group(10)), float(em.group(11)),
                              em.group(2), href, tid, bool(em.group(3))))
        labels = []
        for lm in re.finditer(r'<text class="(elabel[^"]*)" x="([-\d.]+)" y="([-\d.]+)" text-anchor="middle">(.*?)</text>', svg, re.S):
            cls, x, y, body = lm.group(1), float(lm.group(2)), float(lm.group(3)), lm.group(4)
            lines = [H.unescape(t) for t in re.findall(r"<tspan[^>]*>([^<]*)</tspan>", body)]
            full = H.unescape(re.search(r"<title>([^<]*)</title>", body).group(1)) if "<title>" in body else "".join(lines)
            labels.append(Label(x, y, lines, full, "cond" in cls))
        figures.append(Figure(anchor, phase, kind, name, width, height, nodes, edges, labels, svg))
    return figures
