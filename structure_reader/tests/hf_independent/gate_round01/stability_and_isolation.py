#!/usr/bin/env python3
"""M1 round01: build stability + limited isolation-failure replay.

- two real `build-hf` invocations (subprocess, raw argv/cwd/rc) must yield the
  same fixed analysis_id and the second must reuse the existing directory.
- current build bytes must equal the frozen candidate deliverables.
- three limited isolation replays (temp project/workspace only, real candidate
  and inputs untouched): wrong AV input, HF source sigmoid->cos drift, unknown
  model key. Each must fail with rc != 0 and produce 0 HTML.

Writes a JSON report; prints a summary. Diagnostic only, not a signed pass.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[3]
PROJECT_ROOT = WORKSPACE_ROOT / "structure_reader"
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CANDIDATE = WORKSPACE_ROOT / "structure_reader/outputs/deepseek_v4_flash_0731_hf/29d70d15971b524bc41cf334fa3768ab851186f68005549628ea638dd51abbd4"
FIXED_ID = "29d70d15971b524bc41cf334fa3768ab851186f68005549628ea638dd51abbd4"
SCRIPT = PROJECT_ROOT / "scripts/read_model_structure.py"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _temp_project(td: str) -> Path:
    project = Path(td) / "project"
    project.mkdir()
    (project / "src").symlink_to(PROJECT_ROOT / "src", target_is_directory=True)
    (project / "configs").symlink_to(PROJECT_ROOT / "configs", target_is_directory=True)
    return project


def _cli_replay(argv, project, patches):
    from structure_reader import cli
    stdout, stderr = io.StringIO(), io.StringIO()
    stack = [patch.object(cli, "_project_root", return_value=project)]
    stack += [patch(target, **kwargs) for target, kwargs in patches]
    for p in stack:
        p.start()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            rc = cli.main(argv)
    finally:
        for p in reversed(stack):
            p.stop()
    return {"cli_argv": argv, "cli_returncode": rc,
            "cli_html_count": len(list(project.rglob("*.html"))),
            "cli_stderr_tail": stderr.getvalue()[-300:]}


def run_build(tag: str) -> dict:
    argv = [sys.executable, str(SCRIPT), "build-hf", "--model", "deepseek_v4_flash_0731_hf", "--json"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(argv, cwd=str(WORKSPACE_ROOT), capture_output=True, text=True, env=env)
    payload = None
    try:
        payload = json.loads(proc.stdout)
    except Exception:
        pass
    return {"tag": tag, "argv": argv, "cwd": str(WORKSPACE_ROOT), "returncode": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr, "payload": payload}


def replay_source_drift() -> dict:
    from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES, HfConfigError
    from structure_reader.build_hf_decoder import HfBuildError, run_hf_build
    with tempfile.TemporaryDirectory(prefix="hf_gate_src_") as td:
        ws = Path(td) / "ws"
        ws.mkdir()
        (ws / "models").symlink_to(WORKSPACE_ROOT / "models", target_is_directory=True)
        for key, rel in SOURCE_FILES.items():
            if rel.startswith("models/"):
                continue  # provided by the ws/models symlink
            src = WORKSPACE_ROOT / rel
            dest = ws / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if key == "modeling_deepseek_v4.py":
                text = src.read_text(encoding="utf-8").replace("sigmoid", "cos")
                dest.write_text(text, encoding="utf-8")
            else:
                dest.symlink_to(src)
        project = _temp_project(td)
        raised = None
        try:
            run_hf_build(ws, project, layer_index=4, model_key="deepseek_v4_flash_0731_hf")
        except (HfBuildError, HfConfigError) as exc:
            raised = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # pragma: no cover
            raised = f"UNEXPECTED {type(exc).__name__}: {exc}"
        cli = _cli_replay(["build-hf", "--model", "deepseek_v4_flash_0731_hf", "--workspace-root", str(ws), "--json"], project, [])
        return {"replay": "source_sigmoid_to_cos", "isolated_project": str(project), "error": raised,
                "rc_equiv": 1 if raised and not raised.startswith("UNEXPECTED") else 0,
                "html_count": len(list(project.rglob("*.html"))), **cli}


def replay_unknown_model() -> dict:
    from structure_reader.build_hf_decoder import HfBuildError, run_hf_build
    with tempfile.TemporaryDirectory(prefix="hf_gate_unknown_") as td:
        project = _temp_project(td)
        raised = None
        try:
            run_hf_build(WORKSPACE_ROOT, project, layer_index=4, model_key="deepseek_v4_flash_0731_hf_TYPO")
        except HfBuildError as exc:
            raised = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # pragma: no cover
            raised = f"UNEXPECTED {type(exc).__name__}: {exc}"
        cli = _cli_replay(["build-hf", "--model", "deepseek_v4_flash_0731_hf_TYPO", "--json"], project, [])
        return {"replay": "unknown_model_key", "isolated_project": str(project), "error": raised,
                "rc_equiv": 1 if raised and not raised.startswith("UNEXPECTED") else 0,
                "html_count": len(list(project.rglob("*.html"))), **cli}


def replay_wrong_av() -> dict:
    from structure_reader.checks.deepseek_v4_hf_contracts import validate_hf_reading_document
    from structure_reader.build_hf_decoder import HfBuildError, run_hf_build
    from structure_reader.rules.deepseek_v4_hf import SOURCE_FILES
    doc = json.loads((CANDIDATE / "data/reading_document.json").read_text(encoding="utf-8"))
    inputs = {k: _sha(WORKSPACE_ROOT / v) for k, v in SOURCE_FILES.items()}
    mutant = copy.deepcopy(doc)
    av = next(o for o in mutant["phases"][0]["operators"] if o["id"] == "av_matmul")
    av["inputs"][0] = "q_a"  # swap probabilities for the raw Q
    issues = validate_hf_reading_document(mutant, inputs)
    with tempfile.TemporaryDirectory(prefix="hf_gate_wrongav_") as td:
        project = _temp_project(td)
        raised = None
        with patch("structure_reader.build_hf_decoder.build_reading_document", return_value=mutant):
            try:
                run_hf_build(WORKSPACE_ROOT, project, layer_index=4, model_key="deepseek_v4_flash_0731_hf")
            except HfBuildError as exc:
                raised = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # pragma: no cover
                raised = f"UNEXPECTED {type(exc).__name__}: {exc}"
        cli = _cli_replay(
            ["build-hf", "--model", "deepseek_v4_flash_0731_hf", "--json"], project,
            [("structure_reader.build_hf_decoder.build_reading_document", {"return_value": mutant})],
        )
        return {"replay": "wrong_av_input", "mutated_input": "av_matmul.inputs[0]=q_a",
                "validator_issues": issues[:6], "validator_issue_count": len(issues),
                "error": raised, "rc_equiv": 1 if raised and not raised.startswith("UNEXPECTED") else 0,
                "html_count": len(list(project.rglob("*.html"))), **cli}


def main() -> int:
    report = {"candidate": str(CANDIDATE), "fixed_id": FIXED_ID}
    b1 = run_build("build1")
    b2 = run_build("build2")
    report["builds"] = [b1, b2]
    ok_stable = (
        b1["returncode"] == 0 and b2["returncode"] == 0
        and b1["payload"] and b2["payload"]
        and b1["payload"]["analysis_id"] == FIXED_ID == b2["payload"]["analysis_id"]
        and b2["payload"].get("reused") is True
    )
    report["build_stable_and_reused"] = ok_stable
    # current build bytes vs frozen candidate
    from structure_reader.build_hf_decoder import run_hf_build
    _, docs = run_hf_build(WORKSPACE_ROOT, PROJECT_ROOT, layer_index=4, model_key="deepseek_v4_flash_0731_hf")
    same = {name: (CANDIDATE / name).read_bytes() == data for name, data in docs.items()}
    report["current_bytes_match"] = same
    report["current_bytes_all_match"] = all(same.values())
    report["replays"] = [replay_wrong_av(), replay_source_drift(), replay_unknown_model()]

    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "build1": {k: b1[k] for k in ("returncode",)},
        "build2": {k: b2[k] for k in ("returncode",)},
        "analysis_ids": [b1["payload"]["analysis_id"] if b1["payload"] else None, b2["payload"]["analysis_id"] if b2["payload"] else None],
        "build2_reused": b2["payload"].get("reused") if b2["payload"] else None,
        "build_stable_and_reused": ok_stable,
        "current_bytes_all_match": report["current_bytes_all_match"],
        "replays": [{k: r[k] for k in ("replay", "rc_equiv", "html_count", "error")} for r in report["replays"]],
    }, ensure_ascii=False, indent=2))
    return 0 if ok_stable and report["current_bytes_all_match"] and all(r["rc_equiv"] == 1 and r["html_count"] == 0 for r in report["replays"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
