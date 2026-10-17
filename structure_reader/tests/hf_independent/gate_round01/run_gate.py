#!/usr/bin/env python3
"""Run the round01 independent gate checks against an explicit candidate.

Diagnostic only (not a signed pass). Reads the candidate named on the command
line; never selects by mtime. Exit 1 when any `error` severity issue is found.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import check_layer4 as cl  # noqa: E402
import check_html as ch  # noqa: E402
import check_xlsx as cx  # noqa: E402

REL_DOC = Path("data/reading_document.json")
REL_HTML = Path("viewer/decoder.html")
REL_XLSX = Path("tables/decoder_reading.xlsx")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--workspace-root", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    cand = Path(args.candidate).resolve()
    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else cand.parents[3]
    if not cand.is_dir():
        print(f"candidate not found: {cand}", file=sys.stderr)
        return 2

    report = {"candidate": str(cand), "workspace_root": str(workspace_root), "files": {}, "issues": [], "summary": {}}
    issues = []
    doc_path = cand / REL_DOC
    if doc_path.is_file():
        report["files"]["reading_document.json"] = cl.sha256_file(doc_path)
        document = json.loads(doc_path.read_text(encoding="utf-8"))
    else:
        document = {"phases": []}
        issues.append(cl.Issue("error", "missing_document", f"not found: {doc_path}"))

    issues += cl.check_layer4(document, workspace_root)

    html_path = cand / REL_HTML
    if html_path.is_file():
        report["files"]["decoder.html"] = cl.sha256_file(html_path)
        issues += ch.check_html(html_path.read_text(encoding="utf-8"), document)
    else:
        issues.append(cl.Issue("error", "missing_html", f"not found: {html_path}"))

    xlsx_path = cand / REL_XLSX
    if xlsx_path.is_file():
        report["files"]["decoder_reading.xlsx"] = cl.sha256_file(xlsx_path)
        issues += cx.check_xlsx(xlsx_path, document)
    else:
        issues.append(cl.Issue("error", "missing_xlsx", f"not found: {xlsx_path}"))

    report["issues"] = [i.as_dict() for i in issues]
    report["summary"] = {"total": len(issues), **Counter(i.severity for i in issues), "by_code": dict(Counter(i.code for i in issues))}

    print(f"candidate: {cand}")
    for n, h in report["files"].items():
        print(f"  {n}: {h}")
    s = report["summary"]
    print(f"issues: total={s['total']} error={s.get('error',0)} warning={s.get('warning',0)} manual={s.get('manual',0)} info={s.get('info',0)}")
    print("by code:", json.dumps(s["by_code"], ensure_ascii=False))
    for i in issues:
        if i.severity in ("error", "warning", "manual"):
            print(f"  {i.severity.upper():7s} {i.code} [{i.phase}/{i.subject}]: {i.message}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("report:", args.json_out)
    return 1 if s.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
