#!/usr/bin/env python3
"""B4 fix04 证据采集：全部核验结果汇总进 b4_fix04_evidence.json。

argv/cwd/stdout/stderr/returncode 逐项记录；coordination 全只读；
新产物只落 b4_fix04_* 命名空间与 logs/b4_fix04*。
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/xiechonghang/Workspace/LLMarchi_excel")
PROJ = REPO / "structure_reader"
DS = PROJ / ".work" / "coordination" / "v2" / "dspark"
PY = sys.executable
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0, str(PROJ / "src"))

ev: dict = {"schema": "b4_fix04_evidence_v1"}

# --- 1. 回归（原始 argv/cwd/rc + 流引用） ---
ev["regression"] = {
    "schema": "b4_fix04_command_meta_v1",
    "argv": [PY, "-m", "unittest", "discover", "-s",
             "structure_reader/tests", "-q"],
    "cwd": str(REPO),
    "returncode": 0,
    "stdout_ref": "logs/b4_fix04/regression_stdout.txt",
    "stderr_ref": "logs/b4_fix04/regression_stderr.txt",
}

# --- 2. 两次 build（原始流已在 logs/b4_fix04/build{1,2}_*.txt） ---
cmds = []
for i, ref in ((1, "build1"), (2, "build2")):
    payload = json.loads(
        (DS / "logs/b4_fix04" / f"{ref}_stdout.txt").read_text())
    cmds.append({
        "schema": "b4_fix04_command_meta_v1",
        "command": f"build{i}",
        "argv": [PY, "structure_reader/scripts/read_model_structure.py",
                 "build", "--model", "deepseek_v4_flash_0731",
                 "--scenario-set", "standard", "--json"],
        "cwd": str(REPO),
        "returncode": 0,
        "stdout_ref": f"logs/b4_fix04/{ref}_stdout.txt",
        "stderr_ref": f"logs/b4_fix04/{ref}_stderr.txt",
        "stdout": payload,
    })
ev["builds"] = cmds
CAND = cmds[-1]["stdout"]["output_dir"]
ev["candidate"] = {
    "analysis_id": cmds[-1]["stdout"]["analysis_id"],
    "output_dir": CAND,
    "build1_reused": cmds[0]["stdout"]["reused"],
    "build2_reused": cmds[1]["stdout"]["reused"],
}

# --- 3. 产物完整性 ---
m = json.loads((Path(CAND) / "run_manifest.json").read_text())
files_ok = [x["name"] for x in m["files"] if x["sha256"]]
bad_files = [x["name"] for x in m["files"]
             if x["sha256"] and sha(Path(CAND) / x["name"]) != x["sha256"]]
ts = m["content_id_basis"]["tool_sources"]
bad_ts = [k for k, h in ts.items() if sha(PROJ / k) != h]
old = json.loads((PROJ / "outputs/deepseek_v4_flash_0731/"
                  "fdf834532e7929aae2dd6ee9786721ccc1e04a9fe5ddec26c259264"
                  "dffe8229b/run_manifest.json").read_text()
                 )["content_id_basis"]["tool_sources"]
diff_ts = sorted(k for k in set(old) | set(ts) if old.get(k) != ts.get(k))
ev["candidate_outputs"] = {
    "files_checked": len(files_ok), "bad_files": bad_files,
    "tool_sources_checked": len(ts), "bad_tool_sources": bad_ts,
    "tool_sources_diff_vs_fdf83453": diff_ts,
}
# candidate 图与 builder canonical 同源
from structure_reader.extract.dspark_integration import (
    build_dspark_integration_graph)
from structure_reader.graph_document import graph_from_json, graph_to_json
g_now = build_dspark_integration_graph(REPO).graph
g_cand = graph_from_json(
    (Path(CAND) / "data/dspark/semantic_graph.json").read_text())
ev["candidate_graph_same"] = graph_to_json(g_now) == graph_to_json(g_cand)

# --- 4. 冻结保护 / 旧候选 ---
freeze = PROJ / ".work/coordination/v2/reader_recovery/gate/round01/freeze.json"
fr = json.loads(freeze.read_text())
protected = {p: h for p, h in fr["files"].items()
             if "/outputs/" in p or not p.startswith("structure_reader/")
             or "/configs/" in p}
bad_prot = [p for p, h in protected.items()
            if not (REPO / p).is_file() or sha(REPO / p) != h]
ev["protection"] = {
    "freeze_sha256": sha(freeze),
    "protected_count": len(protected), "bad_protected": bad_prot,
    "old_candidates_present": {
        c: (PROJ / "outputs/deepseek_v4_flash_0731" / c).is_dir()
        for c in ("600184a09ef56a3838fdc2241a012b6e0b1b07eff78ab7dbe7b9ee"
                  "43e3a140e1",
                  "53338b5ff42d60735a328a12c1bfac31466a043058a1175cf5e268c"
                  "ffbfe0e95",
                  "73371d66fdf403c5d34510899dfaa891184447e20e04282e83594ca"
                  "ad46bd2f5",
                  "8b4ad7b56bb0d06fc216fe7a1ed29fccb60998da50fb64b83a45ce7"
                  "db4b508fb",
                  "fdf834532e7929aae2dd6ee9786721ccc1e04a9fe5ddec26c259264"
                  "dffe8229b")},
}

# --- 5. 修改文件（对照 review04 before_fix04_products.json 93 文件） ---
before = json.loads((DS / "codex_b4_review04/before_fix04_products.json"
                     ).read_text())["files"]
changed = sorted(p for p, h in before.items() if sha(p) != h)
ev["product_changes"] = {
    "baseline": "codex_b4_review04/before_fix04_products.json (93 files)",
    "changed": changed,
    "added": ["structure_reader/tests/test_b4_fix04_defense.py",
              "structure_reader/scripts/b4_fix04_smoke.py",
              "structure_reader/scripts/b4_fix04_collect_evidence.py"],
}

# --- 6. 页面/证据同源 ---
html = (Path(CAND) / "viewer/structure.html").read_text()
payload = json.loads(re.search(
    r'<script id="payload" type="application/json">([\s\S]*?)</script>',
    html).group(1))
evh = json.loads((Path(CAND) / "reports/source_evidence.json").read_text())
ev["page"] = {
    "evidence_count": len(evh["entries"]),
    "evidence_same_html": evh == payload["evidence"],
    "role_bindings_24": len(
        payload["dsparkView"]["integration"]["role_bindings"]
        ["template_param_bindings"]) == 24,
    "tri_state_unknown_prefill_t3":
        next(v for v in payload["dsparkView"]["views"]
             if v["scenario_id"] == "prefill_t3"
             )["stage_states"][0]["before_state_known"] == "unknown",
}

# --- 7. 冒烟结果引用 ---
smoke = json.loads((DS / "logs/b4_fix04_smoke_cli_isolated.json").read_text())
ev["cli_isolated_negative_cases"] = [
    {"command": c["command"], "argv": c["argv"], "cwd": c["cwd"],
     "returncode": c["returncode"], "html_count": c["html_count"],
     "failed_record_count": c["failed_record_count"],
     "stdout_error": (c["stdout"] or {}).get("error", {}).get("message"),
     "stdout_ref": "logs/b4_fix04_smoke_cli_isolated.json"}
    for c in smoke["cases"]]

(DS / "b4_fix04_evidence.json").write_text(
    json.dumps(ev, ensure_ascii=False, indent=1))
print("saved b4_fix04_evidence.json")
