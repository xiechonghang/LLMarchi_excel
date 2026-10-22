#!/usr/bin/env python3
"""Run the independent HF product checks against an explicitly named candidate.

This is a diagnostic tool, not a formal acceptance gate. It reads the candidate
directory given on the command line (never a "latest" or mtime-selected
directory) and reports issues. Exit code 1 means issues were found (expected
for known-bad candidates); it does NOT mean the product passed.

Usage:
    <engineering-python> run_product_checks.py --candidate <dir> [--json-out <file>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import check_reading_document as crd  # noqa: E402

REL_DOC = Path("data") / "reading_document.json"
REL_HTML = Path("viewer") / "decoder.html"
REL_XLSX = Path("tables") / "decoder_reading.xlsx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Independent HF reading-document checks")
    ap.add_argument("--candidate", required=True, help="explicit candidate directory to inspect (read-only)")
    ap.add_argument("--json-out", default=None, help="optional path to write the machine-readable report")
    args = ap.parse_args(argv)

    cand = Path(args.candidate).resolve()
    if not cand.is_dir():
        print(f"candidate directory not found: {cand}", file=sys.stderr)
        return 2

    report = {"candidate": str(cand), "files": {}, "issues": [], "summary": {}}
    issues = []

    doc_path = cand / REL_DOC
    if not doc_path.is_file():
        issues.append(crd.Issue("error", "missing_reading_document", f"not found: {doc_path}"))
        document = {"phases": []}
    else:
        report["files"]["reading_document.json"] = sha256(doc_path)
        document = json.loads(doc_path.read_text(encoding="utf-8"))
        issues += crd.check_reading_document(document)

    html_path = cand / REL_HTML
    if html_path.is_file():
        report["files"]["decoder.html"] = sha256(html_path)
        issues += crd.check_html(html_path.read_text(encoding="utf-8"), document)
    else:
        issues.append(crd.Issue("error", "missing_html", f"not found: {html_path}"))

    xlsx_path = cand / REL_XLSX
    if xlsx_path.is_file():
        report["files"]["decoder_reading.xlsx"] = sha256(xlsx_path)
        issues += crd.check_workbook(xlsx_path, document)
    else:
        issues.append(crd.Issue("error", "missing_xlsx", f"not found: {xlsx_path}"))

    report["issues"] = [i.as_dict() for i in issues]
    report["summary"] = crd.summarize(issues)

    print(f"candidate: {cand}")
    for name, digest in report["files"].items():
        print(f"  {name}: sha256 {digest}")
    print(f"issues: total={report['summary']['total']} error={report['summary'].get('error',0)} "
          f"warning={report['summary'].get('warning',0)} manual={report['summary'].get('manual',0)}")
    print("by code:")
    for code, n in sorted(report["summary"]["by_code"].items()):
        print(f"  {code}: {n}")
    print("details:")
    for i in issues:
        loc = f" [{i.phase}/{i.subject}]" if i.phase or i.subject else ""
        print(f"  - {i.severity.upper():7s} {i.code}{loc}: {i.message}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written: {out}")

    return 1 if report["summary"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
