#!/usr/bin/env python3
"""Recompute the round01 freeze fingerprints (begin/end).

Verifies freeze.json + task.md SHA256, every frozen file, and that protected old
candidates are unchanged. Writes/prints a JSON summary; exit 1 on any drift.

Usage: verify_freeze.py <gate-dir> [json-out]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    gate_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    freeze_path = gate_dir / "freeze.json"
    task_path = gate_dir / "task.md"
    expected_freeze = "3c0e7d2c7b0bad6bd8f4f20050d7a230e30b2f9177e942d576ced0dc382c05ca"
    expected_task = "758f003feb2dd27c0717ee9b530b4b5b4a89b17e4b5cf35793afd9ad59742e2f"
    out = {"freeze_json_sha256": sha(freeze_path), "freeze_json_ok": sha(freeze_path) == expected_freeze,
           "task_md_sha256": sha(task_path), "task_md_ok": sha(task_path) == expected_task}
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    missing, mismatch = [], []
    for rel, expected in freeze["files"].items():
        p = WORKSPACE_ROOT / rel
        if not p.is_file():
            missing.append(rel)
        elif sha(p) != expected:
            mismatch.append(rel)
    old_missing, old_mismatch = [], []
    for rel, expected in freeze.get("protected_old_candidates", {}).items():
        p = WORKSPACE_ROOT / rel
        if not p.is_file():
            old_missing.append(rel)
        elif sha(p) != expected:
            old_mismatch.append(rel)
    out.update({
        "files_total": len(freeze["files"]), "files_missing": missing, "files_mismatch": mismatch,
        "protected_total": len(freeze.get("protected_old_candidates", {})),
        "protected_missing": old_missing, "protected_mismatch": old_mismatch,
        "candidate": freeze.get("candidate"), "source_revision": freeze.get("source_revision"),
    })
    out["ok"] = out["freeze_json_ok"] and out["task_md_ok"] and not missing and not mismatch and not old_missing and not old_mismatch
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
