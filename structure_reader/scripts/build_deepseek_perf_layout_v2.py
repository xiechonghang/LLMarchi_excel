#!/usr/bin/env python3
"""layout-v2：共用模板 + 参数层派生维度的 DeepSeek Excel 样表入口。

只读冻结 layer_structures.json / network_overview.json，复用
hf_workbook.build_structures_workbook_spec 与 deepseek_perf_layout_v2 适配器。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from structure_reader.export.deepseek_perf_layout_v2 import (  # noqa: E402
    build_perf_layout_v2_workbook,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-structures", required=True, type=Path)
    parser.add_argument("--network-overview", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    container = json.loads(args.layer_structures.read_text(encoding="utf-8"))
    network = json.loads(args.network_overview.read_text(encoding="utf-8"))
    payload = build_perf_layout_v2_workbook(container, network)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    summary = {
        "schema": "deepseek_perf_layout.v2",
        "output": {"path": str(args.output), "sha256": _sha256(args.output),
                   "bytes": args.output.stat().st_size},
        "inputs": {
            "layer_structures": {"path": str(args.layer_structures),
                                 "sha256": _sha256(args.layer_structures)},
            "network_overview": {"path": str(args.network_overview),
                                 "sha256": _sha256(args.network_overview)},
        },
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": summary["output"]["sha256"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
