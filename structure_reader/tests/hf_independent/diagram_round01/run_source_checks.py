#!/usr/bin/env python3
"""Same-source + build-stability checks for the diagram gate round01."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROJECT = ROOT / "structure_reader"
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

CAND = PROJECT / "outputs/deepseek_v4_flash_0731_hf/32af1e923cca4f8c1d535786e4b612d4ccfdfe736c47aeb3671230f77f2d5733"
OLD = PROJECT / "outputs/deepseek_v4_flash_0731_hf/29d70d15971b524bc41cf334fa3768ab851186f68005549628ea638dd51abbd4"
FIXED_ID = "32af1e923cca4f8c1d535786e4b612d4ccfdfe736c47aeb3671230f77f2d5733"
DATA_FILES = ("data/reading_document.json", "data/effective_config.json", "tables/decoder_reading.xlsx")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    report = {"candidate": str(CAND), "fixed_id": FIXED_ID}
    # 1) data files byte-identical to the already-verified 29d70 candidate
    report["data_identical_to_29d70"] = {f: ((CAND / f).read_bytes() == (OLD / f).read_bytes()) for f in DATA_FILES}
    # 2) run_manifest deliverables + tool_sources identity vs actual files
    manifest = json.loads((CAND / "run_manifest.json").read_text(encoding="utf-8"))
    report["deliverable_hashes_ok"] = {name: (sha(CAND / name) == expected)
                                      for name, expected in manifest["deliverables"].items()}
    report["tool_sources_ok"] = {name: (sha(PROJECT / name) == expected)
                                 for name, expected in manifest["identity"]["tool_sources"].items()}
    report["tool_sources_count"] = len(manifest["identity"]["tool_sources"])
    report["analysis_id_ok"] = manifest["analysis_id"] == FIXED_ID
    # 3) current renderer/build reproduces the candidate bytes exactly
    from structure_reader.build_hf_decoder import run_hf_build
    _, docs = run_hf_build(ROOT, PROJECT, layer_index=4, model_key="deepseek_v4_flash_0731_hf")
    report["current_render_matches_candidate"] = {name: ((CAND / name).read_bytes() == data) for name, data in docs.items()}
    # 4) two real builds: same fixed id + reused
    argv = [sys.executable, str(PROJECT / "scripts/read_model_structure.py"), "build-hf",
            "--model", "deepseek_v4_flash_0731_hf", "--json"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    builds = []
    for tag in ("build1", "build2"):
        p = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True, env=env)
        try:
            payload = json.loads(p.stdout)
        except Exception:
            payload = None
        builds.append({"tag": tag, "argv": argv, "cwd": str(ROOT), "returncode": p.returncode,
                       "payload": payload, "stderr": p.stderr})
    report["builds"] = builds
    report["build_stable_reused"] = bool(
        all(b["returncode"] == 0 and b["payload"] and b["payload"]["analysis_id"] == FIXED_ID for b in builds)
        and builds[1]["payload"].get("reused") is True)
    ok = (all(report["data_identical_to_29d70"].values()) and all(report["deliverable_hashes_ok"].values())
          and all(report["tool_sources_ok"].values()) and report["analysis_id_ok"]
          and all(report["current_render_matches_candidate"].values()) and report["build_stable_reused"])
    report["ok"] = ok
    print(json.dumps({k: (v if not isinstance(v, dict) else f"{sum(v.values())}/{len(v)} ok")
                      for k, v in report.items() if k not in ("builds",)}, ensure_ascii=False, indent=2))
    print("builds:", [(b["tag"], b["returncode"], (b["payload"] or {}).get("reused")) for b in builds])
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
