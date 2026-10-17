#!/usr/bin/env python3
"""Recompute the diagram_gate/round01 freeze fingerprints (begin/end)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXPECTED_FREEZE = "ee8d5c0aff7133f04eb4e2d06dad3e0cd0a7a27ec4d45c6e335fb2405564e8ca"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    gate = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    fz = gate / "freeze.json"
    out = {"freeze_json_sha256": sha(fz), "freeze_json_ok": sha(fz) == EXPECTED_FREEZE}
    freeze = json.loads(fz.read_text(encoding="utf-8"))
    missing, mismatch = [], []
    for rel, expected in freeze["files"].items():
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
        elif sha(p) != expected:
            mismatch.append(rel)
    old_missing, old_mismatch = [], []
    for rel, expected in freeze.get("protected_old_candidates", {}).items():
        p = ROOT / rel
        if not p.is_file():
            old_missing.append(rel)
        elif sha(p) != expected:
            old_mismatch.append(rel)
    out.update({"files_total": len(freeze["files"]), "files_missing": missing, "files_mismatch": mismatch,
                "protected_total": len(freeze.get("protected_old_candidates", {})),
                "protected_missing": old_missing, "protected_mismatch": old_mismatch,
                "candidate": freeze.get("candidate")})
    out["ok"] = out["freeze_json_ok"] and not missing and not mismatch and not old_missing and not old_mismatch
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
