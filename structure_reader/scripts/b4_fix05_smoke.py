"""B4 fix05 冒烟：2 份 review05 实际坏图真实 CLI 临时隔离回放。

每例：mock builder 返回坏图 + _outputs_root 指向临时隔离输出根 → 真实
cli._cmd_build → 期望 rc1、stdout 仅 build_error、0 HTML、恰 1 条 _failed
记录；真实 outputs 未动。argv/cwd/stdout/stderr/returncode 全记录进
logs/b4_fix05_smoke_cli_isolated.json。坏图静态 JSON 落盘
b4_fix05_mutants/ 供 Codex 回放（review05 目录只读不动）。

同时记录注入等价核验：claude_b4_fix05_mutant_* 与 review05 原始 JSON
canonical 字节一致性（本脚本不再复制原始文件，等价注入在受控目录内）。"""
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
from structure_reader.graph_document import graph_from_json, graph_to_json

REVIEW05 = PROJ / ".work" / "coordination" / "v2" / "dspark" / "codex_b4_review05"
OUT_DIR = PROJ / ".work" / "coordination" / "v2" / "dspark"
MUTANTS = OUT_DIR / "b4_fix05_mutants"
LOGS = OUT_DIR / "logs"

b = build_dspark_integration_graph(REPO)

cases = []
# review05 两份实际坏图原样回放（只读 review05 目录）
for name in ("mutant_vocab_source_claims_hidden",
             "mutant_seed_rank_metadata99"):
    cases.append((name.removeprefix("mutant_"),
                  graph_from_json((REVIEW05 / f"{name}.json").read_text())))

MUTANTS.mkdir(exist_ok=True)
equivalence = []
for label, bad in cases:
    (MUTANTS / f"claude_b4_fix05_mutant_{label}.json").write_text(
        graph_to_json(bad))
    equivalence.append({
        "mutation": label,
        "review05_source": f"codex_b4_review05/mutant_{label}.json",
        "saved_copy": f"b4_fix05_mutants/claude_b4_fix05_mutant_{label}.json",
        "canonical_bytes_equal": graph_to_json(bad) == graph_to_json(
            graph_from_json(
                (REVIEW05 / f"mutant_{label}.json").read_text())),
    })

results = []
for label, bad in cases:
    argv = ["(in-process) cli._cmd_build", "--model", "deepseek_v4_flash_0731",
            "--scenario-set", "standard", "--json"]
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="b4-fix05-smoke-") as td:
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
        failed_dir_exists = (out / "_failed").is_dir()
    row = {
        "schema": "b4_fix05_command_meta_v1",
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
        "failed_dir_exists": failed_dir_exists,
        "isolated_outputs_root": "(temporary; removed)",
        "refs_issues": len(validate_graph_references(bad)),
    }
    results.append(row)
    print(json.dumps({k: row[k] for k in (
        "command", "returncode", "html_count", "failed_record_count")},
        ensure_ascii=False))

LOGS.mkdir(exist_ok=True)
(LOGS / "b4_fix05_smoke_cli_isolated.json").write_text(
    json.dumps({"schema": "b4_fix05_smoke_v1",
                "review05_graphs_replayed": equivalence,
                "cases": results},
               ensure_ascii=False, indent=1))
print("saved logs/b4_fix05_smoke_cli_isolated.json")
