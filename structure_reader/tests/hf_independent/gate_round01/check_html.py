"""Static, independently-provable HTML checks (no browser)."""

from __future__ import annotations

import re

from check_layer4 import Issue, _err, _warn  # type: ignore


def check_html(html_text: str, document: dict):
    issues: list[Issue] = []
    ids = set(re.findall(r'\bid="([^"]+)"', html_text))
    hrefs = re.findall(r'href="#([^"]+)"', html_text)
    missing = sorted(set(hrefs) - ids)
    if missing:
        issues.append(_err("html_dangling_href", f"{len(hrefs)} in-page links reference {len(missing)} missing target ids, e.g. {missing[:8]}"))
    pf = 'id="phase-prefill"'
    dc = 'id="phase-decode"'
    if pf in html_text and dc in html_text:
        if html_text.index(pf) > html_text.index(dc):
            issues.append(_err("html_phase_order", "decode phase appears before prefill phase"))
    else:
        issues.append(_err("html_phase_sections", 'missing id="phase-prefill" or id="phase-decode"'))
    n_sections = len(re.findall(r'<section[^>]*\bphase\b', html_text))
    if n_sections != 2:
        issues.append(_err("html_phase_count", f"expected exactly two phase sections, found {n_sections}"))
    has_blackbox = any(o.get("analysis_status") == "internal_unanalyzed" for p in document.get("phases", []) for o in p.get("operators", []))
    if has_blackbox and "内部未分析" not in html_text:
        issues.append(_err("html_blackbox_marker", "document has internal_unanalyzed ops but HTML lacks 内部未分析"))
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html_text)
    if external:
        issues.append(_warn("html_external_ref", f"{len(external)} external http(s) resource reference(s); offline expected: {external[:5]}"))
    return issues


__all__ = ["check_html"]
