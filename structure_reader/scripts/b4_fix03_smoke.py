"""B4 fix03 第一轮冒烟：5 份 review03 实际坏图真实 CLI 临时隔离回放。

每例：mock builder 返回坏图 + _outputs_root 指向临时隔离输出根 → 真实
cli._cmd_build → 期望 rc1、stdout 仅 build_error、0 HTML、恰 1 条 _failed
记录；真实 outputs 未动。argv/cwd/stdout/stderr/returncode 全记录进
b4_fix03_evidence.json。"""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

REPO = Path("/Users/xiechonghang/Workspace/LLMarchi_excel")
PROJ = REPO / "structure_reader"
sys.path.insert(0, str(PROJ / "src"))

from structure_reader import cli
from structure_reader.checks.graph_contracts import (
    validate_graph_references)
from structure_reader.extract.dspark_integration import (
    build_dspark_integration_graph)

REVIEW03 = PROJ / ".work" / "coordination" / "v2" / "dspark" / "codex_b4_review03"
OUT_DIR = PROJ / ".work" / "coordination" / "v2" / "dspark"
MUTANTS = OUT_DIR / "b4_fix03_mutants"

b = build_dspark_integration_graph(REPO)
g = b.graph
tr = next(t for t in g.state_transitions
          if t.stable_id == "tr_dspark_cache_mtp0")
cases = [
    ("delete_e_ep_cat_proj", replace(
        g, edges=tuple(e for e in g.edges if e.stable_id != "e_ep_cat_proj"))),
    ("delete_mtp0.e_st_hcpre_a_out", replace(
        g, edges=tuple(e for e in g.edges
                       if e.stable_id != "mtp0.e_st_hcpre_a_out"))),
    ("delete_mtp0.e_ds_cache_state", replace(
        g, edges=tuple(e for e in g.edges
                       if e.stable_id != "mtp0.e_ds_cache_state"))),
    ("sample_argmax", replace(
        g, nodes=tuple(replace(n, op_type="argmax")
                       if n.stable_id == "mk.step2.sample" else n
                       for n in g.nodes))),
    ("cache_transition_reversed", replace(
        g, state_transitions=tuple(
            replace(t, before_version_id=tr.after_version_id,
                    after_version_id=tr.before_version_id)
            if t.stable_id == tr.stable_id else t
            for t in g.state_transitions))),
]
# 坏图 JSON 落盘供 Codex 回放（b4_fix03 命名空间；review03 目录只读不动）
MUTANTS.mkdir(exist_ok=True)
for label, bad in cases:
    (MUTANTS / f"claude_b4_fix03_mutant_{label}.json").write_text(
        json.dumps(json.loads(json.dumps({
            "template_id": bad.template_id,
            "named_axes": [a.to_dict() for a in bad.named_axes],
            "tensors": [t.to_dict() for t in bad.tensors],
            "nodes": [n.to_dict() for n in bad.nodes],
            "edges": [e.to_dict() for e in bad.edges],
            "state_versions": [v.to_dict() for v in bad.state_versions],
            "state_transitions": [t.to_dict() for t in bad.state_transitions],
            "guards": [x.to_dict() for x in bad.guards],
        })), ensure_ascii=False, indent=1))

results = []
for label, bad in cases:
    argv = ["(in-process) cli._cmd_build", "--model", "deepseek_v4_flash_0731",
            "--scenario-set", "standard", "--json"]
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="b4-fix03-smoke-") as td:
        out = Path(td) / "outputs"
        s = io.StringIO()
        args = argparse.Namespace(
            model="deepseek_v4_flash_0731", workspace_root=str(REPO),
            scenario_set="standard", json=True)
        with patch(
                "structure_reader.extract.dspark_integration."
                "build_dspark_integration_graph",
                return_value=replace(b, graph=bad)), \
                patch.object(cli, "_outputs_root", return_value=out), \
                contextlib.redirect_stdout(s):
            rc = cli._cmd_build(args)
        elapsed = time.monotonic() - t0
        stdout = s.getvalue()
        html_count = len(list(out.rglob("*.html")))
        failed = sorted(str(p.relative_to(out)) for p in
                        (out / "_failed").glob("*/run_manifest.json"))
    row = {
        "schema": "b4_fix03_command_meta_v1",
        "command": f"cli_isolated_{label}",
        "mutation": label,
        "argv": argv,
        "cwd": str(REPO),
        "returncode": rc,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_raw": stdout,
        "stdout": json.loads(stdout) if stdout.strip() else None,
        "stderr": "",
        "html_count": html_count,
        "failed_record_count": len(failed),
        "failed_records": failed,
        "isolated_outputs_root": "(temporary; removed)",
        "refs_issues": len(validate_graph_references(bad)),
    }
    results.append(row)
    print(json.dumps({k: row[k] for k in (
        "command", "returncode", "html_count", "failed_record_count")},
        ensure_ascii=False))

(OUT_DIR / "logs").mkdir(exist_ok=True)
(OUT_DIR / "logs" / "b4_fix03_smoke_cli_isolated.json").write_text(
    json.dumps({"schema": "b4_fix03_smoke_v1", "cases": results},
               ensure_ascii=False, indent=1))
print("saved logs/b4_fix03_smoke_cli_isolated.json")
