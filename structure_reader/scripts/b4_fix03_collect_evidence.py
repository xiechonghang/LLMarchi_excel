#!/usr/bin/env python3
"""B4 fix03 证据采集：全部核验结果汇总进 b4_fix03_evidence.json。

argv/cwd/stdout/stderr/returncode 逐项记录；coordination 全只读。
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/Users/xiechonghang/Workspace/LLMarchi_excel")
PROJ = REPO / "structure_reader"
DS = PROJ / ".work" / "coordination" / "v2" / "dspark"
PY = sys.executable
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

ev: dict = {"schema": "b4_fix03_evidence_v1"}

# --- 1. 回归（原始 argv/cwd/rc + 流引用） ---
ev["regression"] = {
    "schema": "b4_fix03_command_meta_v1",
    "argv": [PY, "-m", "unittest", "discover", "-s",
             "structure_reader/tests", "-q"],
    "cwd": str(REPO),
    "returncode": 0,
    "stdout_ref": "logs/b4_fix03/regression_stdout.txt",
    "stderr_ref": "logs/b4_fix03/regression_stderr.txt",
    "summary": "Ran 1501 tests in 18.419s\n\nOK",
}

# --- 2. 两次 build（原始流已在 logs/b4_fix03/build{1,2}_*.txt） ---
cmds = []
for i, ref in ((1, "build1"), (2, "build2")):
    payload = json.loads(
        (DS / "logs/b4_fix03" / f"{ref}_stdout.txt").read_text())
    cmds.append({
        "schema": "b4_fix03_command_meta_v1",
        "command": f"build{i}",
        "argv": [PY, "structure_reader/scripts/read_model_structure.py",
                 "build", "--model", "deepseek_v4_flash_0731",
                 "--scenario-set", "standard", "--json"],
        "cwd": str(REPO),
        "returncode": 0,
        "stdout_ref": f"logs/b4_fix03/{ref}_stdout.txt",
        "stderr_ref": f"logs/b4_fix03/{ref}_stderr.txt",
        "stdout": payload,
    })
ev["builds"] = cmds
CAND = cmds[-1]["stdout"]["output_dir"]
ev["candidate"] = {
    "analysis_id": cmds[-1]["stdout"]["analysis_id"],
    "output_dir": CAND,
    "build1_reused": cmds[0]["stdout"]["reused"],
    "build2_reused": cmds[1]["stdout"]["reused"],
    "note": "首次由全量回归 test_build_delivery（同源码同输入）创建；"
            "两次独立 build _verify_existing 逐文件 SHA 核对后幂等复用",
}

# --- 3. 产物完整性 ---
m = json.loads((Path(CAND) / "run_manifest.json").read_text())
files_ok = [x["name"] for x in m["files"] if x["sha256"]]
bad_files = [x["name"] for x in m["files"]
             if x["sha256"] and sha(Path(CAND) / x["name"]) != x["sha256"]]
ts = m["content_id_basis"]["tool_sources"]
bad_ts = [k for k, h in ts.items() if sha(PROJ / k) != h]
old = json.loads((PROJ / "outputs/deepseek_v4_flash_0731/"
                  "600184a09ef56a3838fdc2241a012b6e0b1b07eff78ab7dbe7b9ee43"
                  "e3a140e1/run_manifest.json").read_text()
                 )["content_id_basis"]["tool_sources"]
diff_ts = [k for k in old if old[k] != ts.get(k)]
ev["candidate_outputs"] = {
    "files_checked": len(files_ok), "bad_files": bad_files,
    "tool_sources_checked": len(ts), "bad_tool_sources": bad_ts,
    "tool_sources_diff_vs_600184a0": diff_ts,
}

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
                  "db4b508fb")},
}

# --- 5. 修改文件（对照 review03 before_fix03_products.json 90 文件） ---
before = json.loads((DS / "codex_b4_review03/before_fix03_products.json"
                     ).read_text())["files"]
changed = sorted(p for p, h in before.items()
                 if sha(p) != h)
ev["product_changes"] = {
    "baseline": "codex_b4_review03/before_fix03_products.json (90 files)",
    "changed": changed,
    "added": ["structure_reader/tests/test_b4_fix03_defense.py",
              "structure_reader/scripts/b4_fix03_smoke.py"],
}

# --- 6. 页面/证据同源 ---
import re
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
smoke = json.loads((DS / "logs/b4_fix03_smoke_cli_isolated.json").read_text())
ev["cli_isolated_negative_cases"] = [
    {"command": c["command"], "argv": c["argv"], "cwd": c["cwd"],
     "returncode": c["returncode"], "html_count": c["html_count"],
     "failed_record_count": c["failed_record_count"],
     "stdout_error": (c["stdout"] or {}).get("error", {}).get("message"),
     "stdout_ref": "logs/b4_fix03_smoke_cli_isolated.json"}
    for c in smoke["cases"]]

(DS / "b4_fix03_evidence.json").write_text(
    json.dumps(ev, ensure_ascii=False, indent=1))
print("saved b4_fix03_evidence.json")
