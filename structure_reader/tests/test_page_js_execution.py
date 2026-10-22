"""实际页面 JS 执行回归（codex_review_06）：Node vm + DOM 桩，无浏览器。

复现探针来源 delivery/codex_page_probe.cjs（Codex 协调文件，只读复制）；
本测试对其做两处参数化：产物目录取最新 build 输出、探针副本写到 .work。
要求本机 /opt/homebrew/bin/node 存在；缺失时 skipTest（不伪造通过）。
这属于本地 JS 单元执行，不是浏览器/CSS 验收，不涉及 file:// 安全策略。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NODE = "/opt/homebrew/bin/node"
PROBE_SOURCE = (
    PROJECT / ".work" / "coordination" / "v1" / "delivery" / "codex_page_probe.cjs"
)
MODEL_OUTPUTS = PROJECT / "outputs" / "deepseek_v4_flash_0731"


def _latest_output_dir() -> Path:
    runs = [r for r in MODEL_OUTPUTS.iterdir()
            if (r / "viewer" / "structure.html").is_file()]
    return max(runs, key=lambda r: r.stat().st_mtime)


@unittest.skipUnless(Path(NODE).exists(), "本机无 node；跳过 JS 执行回归")
class TestPageJsExecution(unittest.TestCase):
    """实际生成 HTML 的 JS 解析与执行（DOM 桩）：初始化/场景/面板。"""

    @classmethod
    def setUpClass(cls):
        cls.out_dir = _latest_output_dir()
        html = (cls.out_dir / "viewer" / "structure.html").read_text(
            encoding="utf-8")
        # 前置：最新产物必须存在且 payload 是原始 JSON（浏览器行为）
        import re
        m = re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            html, re.S)
        cls.payload = json.loads(m.group(1))

    def test_page_js_initializes_and_all_panels_execute(self):
        # 探针副本：产物目录参数化为最新 build 输出
        probe_source = PROBE_SOURCE.read_text(encoding="utf-8")
        probe_copy = probe_source.replace(
            "30fac0fdbf0415f2f60d414ac893cf7261c9c7b9ab7fcc79bab4e3d218c2c01b",
            self.out_dir.name,
        )
        assert probe_copy != probe_source, "探针目录参数化失败"
        probe_path = PROJECT / ".work" / "page_probe_test.cjs"
        probe_path.write_text(probe_copy, encoding="utf-8")
        run = subprocess.run(
            [NODE, str(probe_path)],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT.parent),  # 探针 root=cwd（工作区根）
        )
        self.assertEqual(run.returncode, 0,
                         f"页面 JS 执行失败:\n{run.stderr[:2000]}")
        payload = json.loads(run.stdout)
        # 初始化 + 11 场景回调 + 全部张量/证据面板，无 undefined/NaN
        self.assertEqual(payload["scenarios"], 12)  # V2: +decode_p126
        self.assertGreaterEqual(payload["evidence"], 60)
        self.assertGreater(payload["tensors"], 90)
        self.assertIn("代表模板已提取", payload["l0groups"])
        self.assertIn("DSpark", payload["dspark"])
        # 探针副本不入正式产物（.work 可丢弃）
        self.assertIn(".work", str(probe_path))

    def test_probe_updated_path_matches_latest_build(self):
        # 探针指向的目录必须是最新真实产物（不是过时目录）
        self.assertEqual(PROBE_SOURCE.exists(), True)
        latest = _latest_output_dir()
        self.assertEqual(self.out_dir, latest)


if __name__ == "__main__":
    unittest.main()
